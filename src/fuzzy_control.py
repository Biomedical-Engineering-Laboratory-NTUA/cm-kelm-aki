"""
Fuzzy Adaptive control for the DE backbone (population selection layer).
======================================================================

Motivation
----------
Our fitness function is a NOISY estimate (3-fold internal cross-validation
AUPRC), not a deterministic value. Two consequences follow directly from
the fuzzy-adaptive-DE literature:

1. Liu & Lampinen (2005), "A Fuzzy Adaptive Differential Evolution
   Algorithm", Soft Computing -- introduced FADE: two fuzzy logic
   controllers that adapt the mutation factor F and crossover rate CR
   generation-to-generation, based on the relative change in decision
   variables and in objective values, instead of using fixed F/CR.

2. More recent work (e.g. Type-3 fuzzy DE variants, 2025) specifically
   highlights that fuzzy-controlled parameter adaptation is *especially*
   beneficial "under noisy conditions" -- directly applicable here, since
   a fixed greedy accept-if-better rule can reject a genuinely better
   genome purely because of CV-fold noise, or accept a worse one by luck.

This module implements two independent fuzzy mechanisms, both usable as
drop-in replacements for the fixed-parameter logic in CMEELM/CMKELM's DE
loop:

  FuzzyDEController   -- adapts F and CR each generation from two crisp
                         inputs: population diversity and improvement
                         rate (stagnation), via a small Mamdani-style
                         fuzzy rule base with triangular membership
                         functions and centroid defuzzification.

  fuzzy_accept()       -- replaces the crisp "trial_fit > target_fit"
                         acceptance test with a fuzzy margin: trials that
                         are only marginally worse are accepted with some
                         probability rather than always rejected,
                         reducing sensitivity to fitness-evaluation noise.

No external fuzzy-logic library is required (hand-rolled, dependency-free,
so it drops into the existing sklearn/numpy-only environment).
"""

import numpy as np


# ----------------------------------------------------------------------
# Generic triangular membership function
# ----------------------------------------------------------------------
def _tri(x, a, b, c):
    """Triangular membership: 0 at a, 1 at b, 0 at c (a <= b <= c)."""
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b > a else 1.0
    return (c - x) / (c - b) if c > b else 1.0


class FuzzyDEController:
    """
    Fuzzy Adaptive DE parameter control (FADE-style, Liu & Lampinen 2005).

    Inputs (computed once per generation from the current population):
      diversity   -- normalized std of the population's genomes (0..1,
                      1 = fully spread out, 0 = fully converged)
      improvement -- normalized generation-to-generation best-fitness
                      gain (0..1, 0 = stagnant, 1 = large improvement)

    Outputs:
      F  in [F_min, F_max]   -- mutation scale factor
      CR in [CR_min, CR_max] -- crossover rate

    Rule-of-thumb encoded in the rule base (consistent with the FADE
    literature): low diversity + low improvement (stagnation) -> INCREASE
    F to escape the local optimum; high diversity + high improvement
    (healthy progress) -> DECREASE F to exploit the promising region more
    carefully. CR follows a similar but independent rule set.
    """

    def __init__(self, F_min=0.3, F_max=0.9, CR_min=0.3, CR_max=0.9):
        self.F_min, self.F_max = F_min, F_max
        self.CR_min, self.CR_max = CR_min, CR_max
        self._prev_best_fitness = None
        self._diversity_history = []

    @staticmethod
    def _memberships_low_mid_high(x):
        """x assumed already normalized to [0, 1]."""
        low = _tri(x, -0.01, 0.0, 0.5)
        mid = _tri(x, 0.0, 0.5, 1.0)
        high = _tri(x, 0.5, 1.0, 1.01)
        return {"low": low, "mid": mid, "high": high}

    def compute_diversity(self, population):
        """Mean per-gene std, normalized by the population's own genome-value range."""
        std_per_gene = population.std(axis=0)
        value_range = population.max(axis=0) - population.min(axis=0)
        value_range = np.where(value_range < 1e-9, 1e-9, value_range)
        normalized = np.clip(std_per_gene / value_range, 0, 1)
        return float(normalized.mean())

    def compute_improvement(self, current_best_fitness):
        """Normalized best-fitness gain since the previous generation."""
        if self._prev_best_fitness is None:
            improvement = 0.5  # neutral prior on the first generation
        else:
            raw_gain = current_best_fitness - self._prev_best_fitness
            # squashed into [0, 1]; scale (10.0) assumes AUPRC-scale fitness deltas
            improvement = 1.0 / (1.0 + np.exp(-10.0 * raw_gain)) if raw_gain != 0 else 0.5
        self._prev_best_fitness = current_best_fitness
        return float(np.clip(improvement, 0, 1))

    def update(self, population, current_best_fitness):
        """Call once per generation; returns (F, CR) for that generation."""
        diversity = self.compute_diversity(population)
        improvement = self.compute_improvement(current_best_fitness)

        div_m = self._memberships_low_mid_high(diversity)
        imp_m = self._memberships_low_mid_high(improvement)

        # Mamdani rule base (min for AND, max for aggregation), mapped onto
        # a 3-level output scale {low, mid, high} for both F and CR, then
        # defuzzified via a simple weighted average (centroid of a 3-point
        # discrete set -- equivalent in effect to centroid-of-singletons).
        rules_F = {
            "low": max(min(div_m["low"], imp_m["low"]),      # stagnant+converged -> explore more
                       min(div_m["low"], imp_m["mid"])),
            "mid": max(min(div_m["mid"], imp_m["mid"]),
                       min(div_m["high"], imp_m["low"])),
            "high": max(min(div_m["high"], imp_m["high"]),   # diverse+improving -> keep exploring
                        min(div_m["low"], imp_m["high"])),
        }
        # NOTE on direction: low diversity + low improvement (stuck) should
        # INCREASE F to jump out -- so "low" diversity/improvement maps to
        # a comparatively higher F output level than mid-range cases. We
        # therefore invert the low/high output labels for F specifically:
        F_low, F_mid, F_high = rules_F["high"], rules_F["mid"], rules_F["low"]
        F = self._defuzzify(F_low, F_mid, F_high, self.F_min, self.F_max)

        rules_CR = {
            "low": max(min(div_m["high"], imp_m["high"]), min(div_m["mid"], imp_m["high"])),
            "mid": min(div_m["mid"], imp_m["mid"]),
            "high": max(min(div_m["low"], imp_m["low"]), min(div_m["low"], imp_m["mid"])),
        }
        CR = self._defuzzify(rules_CR["low"], rules_CR["mid"], rules_CR["high"],
                              self.CR_min, self.CR_max)

        self._diversity_history.append(diversity)
        return F, CR

    @staticmethod
    def _defuzzify(w_low, w_mid, w_high, out_min, out_max):
        """Weighted-average defuzzification over three output singletons."""
        total_w = w_low + w_mid + w_high
        if total_w < 1e-9:
            return (out_min + out_max) / 2.0
        levels = np.array([out_min, (out_min + out_max) / 2.0, out_max])
        weights = np.array([w_low, w_mid, w_high])
        return float(np.dot(levels, weights) / total_w)


def fuzzy_accept(target_fitness, trial_fitness, noise_estimate, rng):
    """
    Fuzzy/soft replacement for the crisp `trial_fitness > target_fitness`
    DE acceptance test, aware of fitness-evaluation noise.

    If the trial clearly beats the target (beyond the noise band), accept
    outright. If it clearly loses, reject outright. If the difference
    falls WITHIN the estimated noise band (both plausibly "equal" given
    CV-fold variance), accept with a fuzzy membership-based probability
    rather than a hard cutoff -- this is the "population selection" fuzzy
    mechanism requested, distinct from the F/CR controller above.

    noise_estimate: a rough standard-deviation estimate of the fitness
    function's own noise (e.g. std of AUPRC across the internal CV folds
    for one evaluation) -- pass 0.0 to fall back to the crisp rule.
    """
    diff = trial_fitness - target_fitness
    if noise_estimate <= 1e-9:
        return diff > 0

    # membership: how "clearly better" is the trial, scaled by noise
    membership_better = _tri(diff, -noise_estimate, noise_estimate, 3 * noise_estimate)
    if diff >= 3 * noise_estimate:
        return True
    if diff <= -noise_estimate:
        return False
    accept_probability = np.clip(membership_better, 0.0, 1.0)
    return rng.random() < accept_probability
