"""
Adaptive Cultural-Memetic DE for full-cohort Nystrom-KELM.
==========================================================

Motivation (results of 26-27 Sep 2026)
--------------------------------------
* The static CM-KELM search saturates early: the best fitness plateaus
  around generation 5, and doubling the budget (40/30) adds nothing.
* Averaging models from 3 independent seeds gives AUROC 0.850 against 0.841
  for one seed, and a population ensemble from ONE run reaches 0.848.
  Diversity is what the ensemble feeds on, and convergence destroys it.

Goal: keep the diversity of independent runs inside a single run, without
losing solution quality. Target: single-run population ensemble >= 0.850 / 0.248.

Mechanisms (each an independent flag, for ablation)
---------------------------------------------------
K  adaptive_ks      Adaptive knowledge-source selection. For every offspring
                    ONE influence is drawn from
                      normative   (soft clip to the accepted individuals' ranges)
                      situational (pull toward the exemplar)
                      topographic (DE partners from the offspring's own island)
                      free        (plain DE, whole population, no influence)
                    using probability matching (Thierens 2005): each source's
                    quality is an exponential moving average of the fitness
                    gains it produced, and p_k = p_min + (1 - K p_min) q_k / sum(q).
                    Domain knowledge (protected clinical features) is a hard
                    constraint and is always applied.
A  adaptive_accept  Diversity-driven acceptance. D = mean pairwise Hamming
                    distance of the feature masks, normalised by its initial
                    value D0. When D falls, the belief space learns from a
                    larger share of the population (accept 0.2 -> 0.5) and the
                    situational pull weakens (0.05 -> 0), which slows the
                    premature convergence seen from generation 5 onwards.
H  hetero_islands   Heterogeneous islands. Each topographic island starts from
                    a different knowledge-source profile (normative-,
                    situational-, topographic- and exploration-leaning) and,
                    with K on, adapts its own probabilities. Island champions
                    then differ systematically, which is what the ensemble needs.

R  robust_fitness   Noisy-optimisation mode (added 27 Sep 2026). The confirmatory
                    runs showed the search fitting the noise of a FIXED inner CV
                    (same 3 folds, same landmarks): CV fitness keeps rising while
                    test performance does not. With R the folds are re-drawn every
                    generation and the whole population is re-evaluated on the new
                    split, so parents and trials are always compared on the same
                    folds but no split can be memorised. The final ranking uses the
                    mean of 3 fresh splits (reduces the winner's curse).
L  robust_landmarks with R, also re-draw the Nystrom landmarks every generation.
Z  fuzzy_accept     FADE-style acceptance: a trial within the estimated noise band
                    (median std of the fold scores) is accepted with a fuzzy
                    probability instead of a hard ">" test.

All flags off reproduces the original static scheme (all influences applied
to every offspring), so the class is a strict superset of FullCohortCMKELM.

The run records per-generation traces (knowledge-source probabilities per
island, diversity, acceptance fraction, best fitness) in `trace_` for the
figures of the paper.
"""

import numpy as np

from cmeelm import CulturalBeliefSpace, memetic_local_search
from fuzzy_control import fuzzy_accept
from kelm_variant import KELMEncoding
from nystrom_cmkelm import nystrom_kelm_fitness
from nystrom_kelm import NystromKELM

SOURCES = ("normative", "situational", "topographic", "free")
ISLAND_PROFILES = np.array([   # starting probabilities for heterogeneous islands
    [0.55, 0.15, 0.15, 0.15],  # normative-leaning
    [0.15, 0.55, 0.15, 0.15],  # situational-leaning
    [0.15, 0.15, 0.55, 0.15],  # topographic-leaning
    [0.15, 0.15, 0.15, 0.55],  # exploration-leaning
])


def mask_diversity(pop, encoding):
    masks = np.array([encoding.decode(g)[2] for g in pop])
    n = len(masks)
    if n < 2:
        return 0.0
    d = (masks[:, None, :] != masks[None, :, :]).sum(axis=2)
    return float(d[np.triu_indices(n, 1)].mean() / masks.shape[1])


class AdaptiveCulturalCMKELM:
    def __init__(self, population_size=20, generations=15, F=0.6, CR=0.7,
                 memetic_top_frac=0.1, use_memetic=True, n_landmarks=1000,
                 complexity_penalty=0.001, protected_feature_indices=None,
                 n_cells=4, migration_every=5, random_state=0, rank=None, eig_rtol=None,
                 adaptive_ks=True, adaptive_accept=True, hetero_islands=True,
                 robust_fitness=False, robust_landmarks=False, fuzzy=False,
                 backend="cpu", precision="fp64", multimeme=None, meme_budget=3,
                 pm_alpha=0.3, p_min=0.05, situational_strength=0.05):
        self.robust_fitness, self.robust_landmarks, self.fuzzy = robust_fitness, robust_landmarks, fuzzy
        self.backend, self.precision = backend, precision   # 'gpu' -> gpu_fitness.NystromFitness
        self.multimeme, self.meme_budget = multimeme, meme_budget   # hybrid_memes.py
        self.population_size = population_size
        self.generations = generations
        self.F, self.CR = F, CR
        self.memetic_top_frac = memetic_top_frac
        self.use_memetic = use_memetic
        self.n_landmarks = n_landmarks
        self.complexity_penalty = complexity_penalty
        self.protected_feature_indices = protected_feature_indices or []
        self.n_cells = n_cells
        self.migration_every = migration_every
        self.rank, self.eig_rtol = rank, eig_rtol
        self.adaptive_ks = adaptive_ks
        self.adaptive_accept = adaptive_accept
        self.hetero_islands = hetero_islands
        self.pm_alpha, self.p_min = pm_alpha, p_min
        self.situational_strength = situational_strength
        self.rng = np.random.default_rng(random_state)
        self.encoding = self.belief = self.best_model = None
        self.trace_ = []

    # ------------------------------------------------------------------
    def _initial_probs(self):
        k = len(SOURCES)
        if self.hetero_islands:
            return np.array([ISLAND_PROFILES[c % len(ISLAND_PROFILES)] for c in range(self.n_cells)])
        return np.full((self.n_cells, k), 1.0 / k)

    def _update_probs(self, probs, quality, rewards, counts):
        k = len(SOURCES)
        for c in range(self.n_cells):
            for s in range(k):
                if counts[c, s] > 0:
                    quality[c, s] = ((1 - self.pm_alpha) * quality[c, s]
                                     + self.pm_alpha * rewards[c, s] / counts[c, s])
            q = quality[c]
            if q.sum() > 0:
                probs[c] = self.p_min + (1 - k * self.p_min) * q / q.sum()
        return probs, quality

    def _partners(self, i, cell, gen, use_island):
        n = self.population_size
        whole = np.array([j for j in range(n) if j != i])
        if not use_island or gen % self.migration_every == 0:
            return whole
        pool = np.flatnonzero(cell == cell[i])
        pool = pool[pool != i]
        return pool if len(pool) >= 3 else whole

    # ------------------------------------------------------------------
    def fit(self, X, y):
        n_features = X.shape[1]
        self.encoding = KELMEncoding(n_features)
        gsize = self.encoding.total_size
        self.belief = CulturalBeliefSpace(
            gsize, n_cells=self.n_cells, migration_every=self.migration_every,
            protected_feature_indices=self.protected_feature_indices, mask_gene_offset=2)

        self._cv_seed = 0
        self._noise = []

        if self.backend != "cpu":
            if self.rank is not None or self.eig_rtol is not None:
                raise ValueError("rank / eig_rtol are not supported on the GPU backend")
            from gpu_fitness import NystromFitness
            self.fitness_ = NystromFitness(X, y, self.encoding, self.n_landmarks,
                                           self.complexity_penalty,
                                           backend="cpu" if self.backend == "cpu-xp" else self.backend,
                                           precision=self.precision)

        self.n_evals_ = 0

        def fit_fn(g):
            self.n_evals_ += 1
            lm = self._cv_seed if self.robust_landmarks else 0
            if self.backend == "cpu":
                f, sd = nystrom_kelm_fitness(g, X, y, self.encoding, self.n_landmarks,
                                             self.complexity_penalty, rank=self.rank,
                                             eig_rtol=self.eig_rtol, cv_seed=self._cv_seed,
                                             lm_seed=lm, return_fold_std=True)
            else:
                f, sd = self.fitness_(g, cv_seed=self._cv_seed, lm_seed=lm, return_fold_std=True)
            self._noise.append(sd)
            return f

        pop = np.array([self.belief.enforce_domain_knowledge(self.encoding.random_individual(self.rng))
                        for _ in range(self.population_size)])
        fit = np.array([fit_fn(g) for g in pop])
        self.belief.update(pop, fit)                     # also fixes the island map
        cell = np.asarray(self.belief.cell_assignment) % self.n_cells

        k = len(SOURCES)
        probs = self._initial_probs()
        quality = probs.copy()                           # neutral start = initial profile
        d0 = max(mask_diversity(pop, self.encoding), 1e-9)
        static = not (self.adaptive_ks or self.hetero_islands)
        mm = None
        if self.multimeme:
            from hybrid_memes import IslandMultimeme
            mm = IslandMultimeme(self.multimeme, self.n_cells, n_features,
                                 self.protected_feature_indices, budget=self.meme_budget)

        for gen in range(self.generations):
            if self.robust_fitness and gen > 0:
                # new split for this generation; re-score parents on it so that
                # parent/trial comparisons stay paired
                self._cv_seed = 1000 + gen
                fit = np.array([fit_fn(g) for g in pop])
            noise = float(np.median(self._noise[-200:])) if self._noise else 0.0
            div = mask_diversity(pop, self.encoding)
            ratio = float(np.clip(div / d0, 0.0, 1.0))
            if self.adaptive_accept:
                self.belief.accept_fraction = 0.2 + 0.3 * (1.0 - ratio)
                sit_strength = self.situational_strength * ratio
            else:
                sit_strength = self.situational_strength
            self.belief.update(pop, fit)

            rewards = np.zeros((self.n_cells, k))
            counts = np.zeros((self.n_cells, k))
            new_pop, new_fit = pop.copy(), fit.copy()
            for i in range(self.population_size):
                c = cell[i]
                if static:
                    src = None
                    partners = self._partners(i, cell, gen, use_island=True)
                else:
                    src = int(self.rng.choice(k, p=probs[c]))
                    partners = self._partners(i, cell, gen, use_island=(SOURCES[src] == "topographic"))

                r1, r2, r3 = pop[self.rng.choice(partners, 3, replace=False)]
                mutant = r1 + self.F * (r2 - r3)
                if static:
                    mutant = self.belief.clip_to_norms(mutant)
                    mutant = self.belief.pull_to_exemplar(mutant, strength=sit_strength)
                elif SOURCES[src] == "normative":
                    mutant = self.belief.clip_to_norms(mutant)
                elif SOURCES[src] == "situational":
                    # applied to ~1/4 of offspring instead of all, so a stronger pull
                    # (0.08-0.20); it still weakens with diversity loss when A is on
                    mutant = self.belief.pull_to_exemplar(mutant, strength=max(sit_strength, 0.02) * 4)

                cross = self.rng.random(gsize) < self.CR
                if not static:
                    cross[self.rng.integers(gsize)] = True  # at least one gene from the mutant
                trial = self.belief.enforce_domain_knowledge(np.where(cross, mutant, pop[i]))
                t_fit = fit_fn(trial)
                if src is not None:
                    counts[c, src] += 1
                    rewards[c, src] += max(0.0, t_fit - fit[i])
                accept = (fuzzy_accept(fit[i], t_fit, noise, self.rng) if self.fuzzy
                          else t_fit > fit[i])
                if accept:
                    new_pop[i], new_fit[i] = trial, t_fit
            pop, fit = new_pop, new_fit

            if self.adaptive_ks:
                probs, quality = self._update_probs(probs, quality, rewards, counts)

            meme_log = None
            if mm is not None:
                pop, fit, meme_log = mm.step(
                    pop, fit, cell, fit_fn, self.rng, self.belief.enforce_domain_knowledge, d0,
                    noise, self.belief.accept_fraction, lambda P: mask_diversity(P, self.encoding))
            elif self.use_memetic:
                top = np.argsort(fit)[::-1][:max(1, int(self.population_size * self.memetic_top_frac))]
                for idx in top:
                    ref, ref_fit = memetic_local_search(pop[idx], fit_fn, step_size=0.1,
                                                        max_iters=5, rng=self.rng)
                    ref = self.belief.enforce_domain_knowledge(ref)
                    if ref_fit > fit[idx]:
                        pop[idx], fit[idx] = ref, ref_fit

            self.trace_.append({
                "gen": gen + 1, "best": float(fit.max()), "mean": float(fit.mean()),
                "diversity": div, "diversity_ratio": ratio,
                "accept_fraction": float(self.belief.accept_fraction),
                "probs": probs.round(4).tolist(), "noise": noise, "memes": meme_log,
            })
            ptxt = " | ".join("/".join(f"{v:.2f}" for v in probs[c]) for c in range(self.n_cells))
            print(f"  Gen {gen + 1}/{self.generations}: best {fit.max():.4f}  mean {fit.mean():.4f}  "
                  f"div {div:.3f} ({ratio:.2f})  acc {self.belief.accept_fraction:.2f}  "
                  f"p[N/S/T/F] {ptxt}", flush=True)
            if meme_log:
                print("    memes: " + "  ".join(f"I{e['island']}:{e['meme']}({e['gain']:+.4f})"
                                                for e in meme_log), flush=True)

        if self.robust_fitness:
            # final ranking on the mean of 3 fresh splits (less winner's curse)
            scores = []
            for sd_ in (7001, 7002, 7003):
                self._cv_seed = sd_
                scores.append([fit_fn(g) for g in pop])
            fit = np.mean(scores, axis=0)
            print(f"  final re-ranking on 3 fresh splits: best {fit.max():.4f}", flush=True)

        if mm is not None:
            self.meme_usage_ = mm.usage.tolist()
        # same attributes as FullCohortCMKELM (population_ensemble.py relies on them)
        self.final_population_, self.final_fitnesses_ = pop.copy(), fit.copy()
        self.cell_assignment_ = cell.copy()
        best = int(np.argmax(fit))
        self.best_genome = pop[best]
        self.best_gamma_, self.best_lambda_, self.best_mask_ = self.encoding.decode(self.best_genome)
        self.best_model = NystromKELM(gamma=self.best_gamma_, lam=self.best_lambda_,
                                      n_landmarks=self.n_landmarks, random_state=0,
                                      rank=self.rank, eig_rtol=self.eig_rtol).fit(X * self.best_mask_, y)
        return self

    def predict_proba(self, X):
        return self.best_model.predict_proba(X * self.best_mask_)
