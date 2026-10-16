"""
Hybrid multimeme local search with per-island meme selection.
=============================================================

Replaces the single Gaussian hill-climb of CM-KELM with a pool of three
local searchers ("memes") of different character, and lets every island
choose which one to apply to its champion in each generation.

Memes (all stateful PER ISLAND, so a meme continues where it stopped the
previous time the island used it -- short calls interleaved with evolution
add up to a long local-search trajectory):

  hc    Gaussian hill-climb on the whole genome (the original CM-KELM meme).
  tabu  Culturally guided tabu search with guided-local-search (GLS) penalties
        on the FEATURE MASK. Candidate moves are single bit flips:
          drop j  (j in mask)    weight ~ (1 + p_j) * (1 - f_j)
          add j   (j not in mask) weight ~ f_j
        where f_j is the frequency of feature j in the belief space's accepted
        set (the cultural "situational" knowledge) and p_j the GLS penalty of
        feature j. Flipped features become tabu for `tenure` calls (tabu moves
        are sampled only when every move is tabu). When a call finds no
        improvement (a local optimum for this island), the penalty of the kept
        feature with the highest GLS utility c_j / (1 + p_j), c_j = 1 - f_j, is
        increased: features the culture does not support are pushed out.
        Protected clinical features (domain knowledge) are never dropped.
  sa    Simulated annealing on (log gamma, log lambda) with Metropolis moves.
        The initial temperature is the estimated fitness noise (median SD of
        the fold scores), the step size the island's spread of the two
        hyperparameters; T cools geometrically and is re-heated when the island
        stagnates.

Controllers (which meme an island uses in a generation):

  fuzzy    zero-order Sugeno rule base on two island descriptors
             d = island mask diversity / initial global diversity
             s = generations since the island best last improved
           R1  d HIGH and s FRESH -> hc      (diverse and progressing: refine)
           R2  d LOW              -> tabu    (masks collapsed: structured moves)
           R3  s STAGNANT         -> sa      (stuck: move the kernel/ridge)
           R4  d LOW and s STAGNANT -> tabu 0.5, sa 0.5
  credit   probability matching on the improvement per evaluation of each meme
  fc       0.5 fuzzy + 0.5 credit (default: rules for the prior, credit to adapt)
  uniform  random meme
  hc | tabu | sa   a single fixed meme (ablations)

Budget: `budget` fitness evaluations per island per generation. The default
(3 x 4 islands = 12) equals the evaluations the static CM-KELM spends in its
memetic step (top 10 % of 20 = 2 individuals x 6 evaluations), so hybrid and
static runs are compared at the same total number of fitness evaluations.
"""

import numpy as np

MEMES = ("hc", "tabu", "sa")


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _mask_of(genome):
    m = (genome[2:] > 0)
    if not m.any():
        m = np.zeros_like(m)
        m[np.argmax(genome[2:])] = True
    return m


def _flip(genome, j, on):
    g = genome.copy()
    v = abs(g[2 + j]) + 0.5
    g[2 + j] = v if on else -v
    return g


# ----------------------------------------------------------------------
# memes
# ----------------------------------------------------------------------
class GaussHC:
    def __init__(self, step=0.1):
        self.step = step

    def run(self, genome, fit, fit_fn, budget, rng, ctx):
        best, best_fit = genome.copy(), fit
        for _ in range(budget):
            cand = ctx["repair"](best + rng.normal(0, self.step, best.shape))
            f = fit_fn(cand)
            if f > best_fit:
                best, best_fit = cand, f
        return best, best_fit


class MaskTabuGLS:
    def __init__(self, n_features, protected, tenure=4, gls_step=1.0):
        self.n = n_features
        self.protected = set(protected)
        self.tenure = tenure
        self.gls_step = gls_step
        self.pen = np.zeros(n_features)
        self.tabu_until = np.zeros(n_features, dtype=int)
        self.calls = 0
        self.best_seen = -np.inf

    def run(self, genome, fit, fit_fn, budget, rng, ctx):
        self.calls += 1
        self.best_seen = max(self.best_seen, fit)
        freq = ctx["feature_freq"]
        mask = _mask_of(genome)
        w = np.where(mask, (1.0 + self.pen) * (1.0 - freq) + 0.05, freq + 0.05)
        for j in self.protected:
            if mask[j]:
                w[j] = 0.0
        tabu = self.tabu_until >= self.calls
        w_nt = np.where(tabu, 0.0, w)
        if w_nt.sum() > 0:
            w = w_nt
        k = min(budget, int((w > 0).sum()))
        if k == 0:
            return genome, fit
        cands = rng.choice(self.n, size=k, replace=False, p=w / w.sum())
        best, best_fit, best_j = genome, fit, None
        for j in cands:
            g = ctx["repair"](_flip(genome, j, on=not mask[j]))
            f = fit_fn(g)
            if f > best_fit:
                best, best_fit, best_j = g, f, j
            if f > self.best_seen:                      # aspiration (also frees tabu)
                self.best_seen = f
        if best_j is not None:
            self.tabu_until[best_j] = self.calls + self.tenure
        else:
            # local optimum for this island: GLS penalty on the kept feature with the
            # highest utility c_j / (1 + p_j), c_j = 1 - cultural support
            kept = [j for j in np.flatnonzero(mask) if j not in self.protected]
            if kept:
                util = (1.0 - freq[kept]) / (1.0 + self.pen[kept])
                self.pen[kept[int(np.argmax(util))]] += self.gls_step
        return best, best_fit


class HyperSA:
    def __init__(self, cooling=0.85, reheat_after=3):
        self.T = None
        self.cooling = cooling
        self.reheat_after = reheat_after

    def run(self, genome, fit, fit_fn, budget, rng, ctx):
        T0 = max(ctx["noise"], 1e-4)
        if self.T is None or ctx["stagnation"] >= self.reheat_after:
            self.T = T0
        sigma = np.maximum(ctx["hyper_spread"], 0.05)
        cur, cur_fit = genome.copy(), fit
        best, best_fit = genome.copy(), fit
        for _ in range(budget):
            cand = cur.copy()
            cand[:2] = cur[:2] + rng.normal(0, sigma)
            cand = ctx["repair"](cand)
            f = fit_fn(cand)
            if f > cur_fit or rng.random() < np.exp((f - cur_fit) / self.T):
                cur, cur_fit = cand, f
            if f > best_fit:
                best, best_fit = cand, f
        self.T *= self.cooling
        return best, best_fit


# ----------------------------------------------------------------------
# controllers
# ----------------------------------------------------------------------
def _low(x):
    return float(np.clip((0.6 - x) / 0.4, 0, 1))


def _high(x):
    return float(np.clip((x - 0.3) / 0.4, 0, 1))


def fuzzy_preferences(d, s):
    stag = min(1.0, s / 3.0)
    fresh = 1.0 - stag
    rules = [                      # (firing strength, output weights hc/tabu/sa)
        (_high(d) * fresh, (1.0, 0.0, 0.0)),
        (_low(d), (0.0, 1.0, 0.0)),
        (stag, (0.0, 0.0, 1.0)),
        (_low(d) * stag, (0.0, 0.5, 0.5)),
    ]
    num = np.zeros(3)
    den = 0.0
    for w, out in rules:
        num += w * np.asarray(out)
        den += w
    pref = num / den if den > 1e-12 else np.full(3, 1 / 3)
    return pref


class MemeController:
    def __init__(self, mode, n_cells, alpha=0.3, p_min=0.1):
        if mode not in ("fuzzy", "credit", "fc", "uniform") + MEMES:
            raise ValueError(f"unknown multimeme controller '{mode}'")
        self.mode, self.alpha, self.p_min = mode, alpha, p_min
        self.q = np.full((n_cells, 3), 1e-3)            # improvement per evaluation (EMA)

    def probs(self, c, d, s):
        if self.mode in MEMES:
            p = np.zeros(3)
            p[MEMES.index(self.mode)] = 1.0
            return p
        if self.mode == "uniform":
            return np.full(3, 1 / 3)
        credit = self.q[c] / self.q[c].sum()
        if self.mode == "credit":
            p = credit
        else:
            fz = fuzzy_preferences(d, s)
            p = fz if self.mode == "fuzzy" else 0.5 * fz + 0.5 * credit
        p = self.p_min + (1 - 3 * self.p_min) * p / p.sum()
        return p

    def reward(self, c, m, gain, evals):
        self.q[c, m] = (1 - self.alpha) * self.q[c, m] + self.alpha * max(gain, 0.0) / max(evals, 1)


class IslandMultimeme:
    """Holds the memes of every island and applies one per island per generation."""

    def __init__(self, mode, n_cells, n_features, protected, budget=3):
        self.ctrl = MemeController(mode, n_cells)
        self.budget = budget
        self.memes = [{"hc": GaussHC(), "tabu": MaskTabuGLS(n_features, protected), "sa": HyperSA()}
                      for _ in range(n_cells)]
        self.island_best = np.full(n_cells, -np.inf)
        self.stagnation = np.zeros(n_cells, dtype=int)
        self.usage = np.zeros((n_cells, 3), dtype=int)

    def step(self, pop, fit, cell, fit_fn, rng, repair, d0, noise, accept_fraction, mask_div_fn):
        n_cells = len(self.memes)
        order = np.argsort(fit)[::-1]
        top = pop[order[:max(2, int(len(pop) * accept_fraction))]]
        freq = np.array([_mask_of(g) for g in top]).mean(axis=0)
        log = []
        for c in range(n_cells):
            members = np.flatnonzero(cell == c)
            if len(members) == 0:
                continue
            i = members[int(np.argmax(fit[members]))]
            d = float(np.clip(mask_div_fn(pop[members]) / d0, 0, 1)) if len(members) > 1 else 0.0
            s = int(self.stagnation[c])
            p = self.ctrl.probs(c, d, s)
            m = int(rng.choice(3, p=p))
            ctx = {"feature_freq": freq, "noise": noise, "stagnation": s, "repair": repair,
                   "hyper_spread": pop[members][:, :2].std(axis=0) if len(members) > 1 else np.full(2, 0.3)}
            g, f = self.memes[c][MEMES[m]].run(pop[i], fit[i], fit_fn, self.budget, rng, ctx)
            gain = f - fit[i]
            self.ctrl.reward(c, m, gain, self.budget)
            self.usage[c, m] += 1
            if f > fit[i]:
                pop[i], fit[i] = g, f
            best_c = fit[members].max()
            if best_c > self.island_best[c] + 1e-12:
                self.island_best[c], self.stagnation[c] = best_c, 0
            else:
                self.stagnation[c] += 1
            log.append({"island": c, "meme": MEMES[m], "p": np.round(p, 3).tolist(),
                        "d": round(d, 3), "s": s, "gain": float(gain)})
        return pop, fit, log
