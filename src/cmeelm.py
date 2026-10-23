"""
CM-EELM: Cultural-Memetic Evolutionary Extreme Learning Machine
==================================================================

Used by the article for CulturalBeliefSpace and memetic_local_search, which the
Nystrom CM-KELM (nystrom_cmkelm.py) imports. The ELM, DEOptimizer and CMEELM classes
are the original random-hidden-layer prototype and are not used for any result.

Core implementation, built to be validated on public data before being
pointed at the real INSPIRE cohort.

Components
----------
- ELM: single-hidden-layer feedforward network with analytically-solved
  output weights (Moore-Penrose pseudo-inverse).
- DEOptimizer: Differential Evolution backbone that searches the ELM
  hidden-layer weights/biases and a binary feature-selection mask.
- CulturalBeliefSpace: normative + situational knowledge that narrows
  the DE mutation range around historically good regions.
- memetic_local_search: Lamarckian hill-climbing applied to the top-K%
  of each generation.
- CMEELM: the full pipeline, with toggles so every ablation step
  (Standard ELM -> DE-ELM -> CA-DE-ELM -> MA-DE-ELM -> full CM-EELM)
  can be produced from the same code path.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold


# ----------------------------------------------------------------------
# Extreme Learning Machine
# ----------------------------------------------------------------------
class ELM:
    """Single-hidden-layer feedforward network with closed-form output layer."""

    def __init__(self, n_hidden=None, W=None, b=None, activation="sigmoid"):
        self.n_hidden = n_hidden if n_hidden is not None else (W.shape[1] if W is not None else None)
        self.W = W          # (n_features, n_hidden)
        self.b = b          # (n_hidden,)
        self.activation = activation
        self.beta = None    # (n_hidden, n_outputs), solved analytically

    @staticmethod
    def _act(x, kind):
        if kind == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        if kind == "relu":
            return np.maximum(0, x)
        if kind == "tanh":
            return np.tanh(x)
        raise ValueError(f"unknown activation {kind}")

    def _hidden(self, X):
        return self._act(X @ self.W + self.b, self.activation)

    def fit(self, X, y, ridge=1e-3):
        """Closed-form output weights via regularized pseudo-inverse."""
        H = self._hidden(X)
        y_onehot = y.reshape(-1, 1).astype(float)
        # Ridge-regularized least squares (more stable than raw pinv)
        I = np.eye(H.shape[1])
        self.beta = np.linalg.solve(H.T @ H + ridge * I, H.T @ y_onehot)
        return self

    def predict_proba(self, X):
        H = self._hidden(X)
        raw = H @ self.beta
        return 1.0 / (1.0 + np.exp(-np.clip(raw.ravel(), -500, 500)))


# ----------------------------------------------------------------------
# Individual encoding: flat vector <-> (W, b, feature mask)
# ----------------------------------------------------------------------
class Encoding:
    """
    Packs/unpacks an individual's genome into ELM-ready arrays.

    The hidden layer size L is now itself evolved: the genome is sized for
    a fixed MAXIMUM number of hidden neurons (max_hidden), plus one extra
    "L-control" gene per potential neuron that acts as an on/off gate.
    Only the neurons whose gate is active are used -- this lets DE grow or
    shrink the effective network capacity instead of it being fixed by a
    hyperparameter chosen in advance.
    """

    def __init__(self, n_features, max_hidden, min_hidden=5):
        self.n_features = n_features
        self.max_hidden = max_hidden
        self.min_hidden = min_hidden
        self.w_size = n_features * max_hidden
        self.b_size = max_hidden
        self.mask_size = n_features
        self.gate_size = max_hidden  # one gate gene per potential hidden neuron
        self.total_size = self.w_size + self.b_size + self.mask_size + self.gate_size

    def random_individual(self, rng):
        w = rng.normal(0, 1, self.w_size)
        b = rng.normal(0, 1, self.b_size)
        mask_logits = rng.normal(0, 1, self.mask_size)
        # bias initial gates so a random individual starts with a
        # reasonable (not maximal, not empty) number of active neurons
        gate_logits = rng.normal(-0.3, 1, self.gate_size)
        return np.concatenate([w, b, mask_logits, gate_logits])

    def decode(self, genome):
        w_flat = genome[: self.w_size]
        b = genome[self.w_size: self.w_size + self.b_size]
        mask_logits = genome[self.w_size + self.b_size: self.w_size + self.b_size + self.mask_size]
        gate_logits = genome[self.w_size + self.b_size + self.mask_size:]

        w = w_flat.reshape(self.n_features, self.max_hidden)
        mask = (mask_logits > 0).astype(float)
        if mask.sum() == 0:
            mask[np.argmax(mask_logits)] = 1.0

        gate = gate_logits > 0
        if gate.sum() < self.min_hidden:
            # force-activate the top-scoring neurons up to min_hidden
            top = np.argsort(gate_logits)[::-1][: self.min_hidden]
            gate = np.zeros(self.max_hidden, dtype=bool)
            gate[top] = True

        w_active = w[:, gate]
        b_active = b[gate]
        return w_active, b_active, mask, int(gate.sum())


# ----------------------------------------------------------------------
# Cultural Algorithm belief space
# ----------------------------------------------------------------------
class CulturalBeliefSpace:
    """
    Implements all four knowledge sources from the original CM-EELM design:

    - Normative knowledge: per-gene [lo, hi] range seen in good individuals.
      Narrows DE's mutation range around historically good regions.
    - Situational knowledge: best genome found so far (exemplar). Pulls
      mutants toward it.
    - Topographic knowledge: the population is partitioned into `n_cells`
      "islands" that explore semi-independently, with periodic migration --
      prevents the whole population from collapsing onto one region too
      early (a common DE failure mode called premature convergence).
    - Domain knowledge: hard clinical constraints -- specific feature
      indices that must never be switched off by the feature-selection
      mask, regardless of what the search discovers (e.g. baseline
      creatinine should never be excluded from an AKI predictor).
    """

    def __init__(self, genome_size, accept_fraction=0.2,
                 n_cells=4, migration_every=5, migration_frac=0.1,
                 protected_feature_indices=None, mask_gene_offset=None):
        self.genome_size = genome_size
        self.accept_fraction = accept_fraction
        self.norm_lo = np.full(genome_size, -np.inf)
        self.norm_hi = np.full(genome_size, np.inf)
        self.exemplar = None
        self.exemplar_fitness = -np.inf

        # topographic knowledge
        self.n_cells = n_cells
        self.migration_every = migration_every
        self.migration_frac = migration_frac
        self.cell_assignment = None  # set on first update()

        # domain knowledge
        self.protected_feature_indices = protected_feature_indices or []
        self.mask_gene_offset = mask_gene_offset  # where mask genes start in the genome

    def update(self, population, fitnesses):
        order = np.argsort(fitnesses)[::-1]
        n_accept = max(1, int(len(population) * self.accept_fraction))
        accepted = population[order[:n_accept]]
        self.norm_lo = accepted.min(axis=0)
        self.norm_hi = accepted.max(axis=0)
        if fitnesses[order[0]] > self.exemplar_fitness:
            self.exemplar_fitness = fitnesses[order[0]]
            self.exemplar = population[order[0]].copy()

        if self.cell_assignment is None or len(self.cell_assignment) != len(population):
            rng = np.random.default_rng(0)
            self.cell_assignment = rng.integers(0, self.n_cells, len(population))

    def clip_to_norms(self, genome, softness=0.25):
        """Softly pull a genome back toward the normative range instead of a hard clip."""
        lo, hi = self.norm_lo, self.norm_hi
        below = genome < lo
        above = genome > hi
        out = genome.copy()
        out[below] = genome[below] + softness * (lo[below] - genome[below])
        out[above] = genome[above] + softness * (hi[above] - genome[above])
        return out

    def pull_to_exemplar(self, genome, strength=0.1):
        if self.exemplar is None:
            return genome
        return genome + strength * (self.exemplar - genome)

    def enforce_domain_knowledge(self, genome):
        """Force protected feature genes to stay switched ON (positive logit)."""
        if not self.protected_feature_indices or self.mask_gene_offset is None:
            return genome
        out = genome.copy()
        for idx in self.protected_feature_indices:
            gene_pos = self.mask_gene_offset + idx
            if out[gene_pos] <= 0:
                out[gene_pos] = 0.5  # push just above the on/off threshold
        return out

    def topographic_partners(self, individual_idx, generation):
        """
        Returns the index pool an individual may draw DE partners from:
        its own island most of the time, the whole population only on
        migration generations. This is what keeps islands semi-independent.
        """
        is_migration_gen = (generation % self.migration_every == 0)
        if is_migration_gen or self.cell_assignment is None:
            return None  # signal: use the whole population
        my_cell = self.cell_assignment[individual_idx]
        pool = np.where(self.cell_assignment == my_cell)[0]
        return pool if len(pool) >= 4 else None  # fall back if island too small

    def island_best_index(self, individual_idx, fitnesses, generation):
        """
        For PSO's lbest variant: returns the index of the best-fitness
        individual within the same island as individual_idx, or the global
        best on migration generations / if islands aren't set up yet.
        """
        is_migration_gen = (generation % self.migration_every == 0)
        if is_migration_gen or self.cell_assignment is None:
            return int(np.argmax(fitnesses))
        my_cell = self.cell_assignment[individual_idx]
        pool = np.where(self.cell_assignment == my_cell)[0]
        if len(pool) == 0:
            return int(np.argmax(fitnesses))
        return int(pool[np.argmax(fitnesses[pool])])


# ----------------------------------------------------------------------
# Memetic local search
# ----------------------------------------------------------------------
def memetic_local_search(genome, fitness_fn, step_size=0.05, max_iters=5, rng=None, repair=None):
    """Lamarckian hill-climbing: small Gaussian perturbations, keep if better.

    repair: optional function applied to every candidate BEFORE it is evaluated
    (e.g. CulturalBeliefSpace.enforce_domain_knowledge), so that the stored fitness
    always belongs to the genome that is kept."""
    rng = rng or np.random.default_rng()
    repair = repair or (lambda g: g)
    best = repair(genome.copy())
    best_fit = fitness_fn(best)
    for _ in range(max_iters):
        candidate = repair(best + rng.normal(0, step_size, size=best.shape))
        cand_fit = fitness_fn(candidate)
        if cand_fit > best_fit:
            best, best_fit = candidate, cand_fit
    return best, best_fit


# ----------------------------------------------------------------------
# Shared fitness function (used by every metaheuristic backbone)
# ----------------------------------------------------------------------
def elm_fitness(genome, X, y, encoding, complexity_penalty=0.001, n_splits=3):
    """AUPRC via internal k-fold CV, minus a mask-size / network-size penalty."""
    w, b, mask, n_active = encoding.decode(genome)
    Xm = X * mask
    elm = ELM(n_active, W=w, b=b)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = []
    for tr_idx, val_idx in skf.split(Xm, y):
        elm.fit(Xm[tr_idx], y[tr_idx])
        p = elm.predict_proba(Xm[val_idx])
        try:
            scores.append(average_precision_score(y[val_idx], p))
        except ValueError:
            scores.append(0.0)
    penalty = complexity_penalty * (mask.sum() + 0.1 * n_active)
    return float(np.mean(scores) - penalty)


# ----------------------------------------------------------------------
# Full CM-EELM pipeline (with ablation toggles) -- Differential Evolution backbone
# ----------------------------------------------------------------------
class CMEELM:
    """
    use_cultural: enable the belief-space-guided mutation
    use_memetic:  enable local search on the top-K% each generation
    Setting both False + population_size=1/generations=1-ish reduces to
    something close to a Standard ELM baseline (random init, no search).
    """

    def __init__(self, max_hidden=60, min_hidden=5, population_size=30, generations=40,
                 F=0.6, CR=0.7, memetic_top_frac=0.1,
                 use_cultural=True, use_memetic=True,
                 complexity_penalty=0.001, random_state=0):
        self.max_hidden = max_hidden
        self.min_hidden = min_hidden
        self.population_size = population_size
        self.generations = generations
        self.F = F
        self.CR = CR
        self.memetic_top_frac = memetic_top_frac
        self.use_cultural = use_cultural
        self.use_memetic = use_memetic
        self.complexity_penalty = complexity_penalty
        self.rng = np.random.default_rng(random_state)

        self.encoding = None
        self.belief = None
        self.best_genome = None
        self.best_elm = None
        self.fitness_history_ = []

    def _fitness(self, genome, X, y, encoding):
        return elm_fitness(genome, X, y, encoding, self.complexity_penalty)

    def fit(self, X, y):
        n_features = X.shape[1]
        self.encoding = Encoding(n_features, self.max_hidden, min_hidden=self.min_hidden)
        genome_size = self.encoding.total_size
        self.belief = CulturalBeliefSpace(genome_size)

        pop = np.array([self.encoding.random_individual(self.rng)
                         for _ in range(self.population_size)])
        fit_fn = lambda g: self._fitness(g, X, y, self.encoding)
        fitnesses = np.array([fit_fn(ind) for ind in pop])

        for gen in range(self.generations):
            if self.use_cultural:
                self.belief.update(pop, fitnesses)

            new_pop = pop.copy()
            for i in range(self.population_size):
                idxs = [j for j in range(self.population_size) if j != i]
                r1, r2, r3 = pop[self.rng.choice(idxs, 3, replace=False)]
                mutant = r1 + self.F * (r2 - r3)

                if self.use_cultural:
                    mutant = self.belief.clip_to_norms(mutant)
                    mutant = self.belief.pull_to_exemplar(mutant, strength=0.05)

                cross_mask = self.rng.random(pop.shape[1]) < self.CR
                trial = np.where(cross_mask, mutant, pop[i])

                trial_fit = fit_fn(trial)
                if trial_fit > fitnesses[i]:
                    new_pop[i] = trial
                    fitnesses[i] = trial_fit
            pop = new_pop

            if self.use_memetic:
                top_k = max(1, int(self.population_size * self.memetic_top_frac))
                top_idx = np.argsort(fitnesses)[::-1][:top_k]
                for idx in top_idx:
                    refined, refined_fit = memetic_local_search(
                        pop[idx], fit_fn, step_size=0.05, max_iters=5, rng=self.rng)
                    if refined_fit > fitnesses[idx]:
                        pop[idx] = refined
                        fitnesses[idx] = refined_fit

            self.fitness_history_.append(fitnesses.max())

        best_i = np.argmax(fitnesses)
        self.best_genome = pop[best_i]
        w, b, mask, n_active = self.encoding.decode(self.best_genome)
        self.best_mask_ = mask
        self.best_n_hidden_ = n_active
        self.best_elm = ELM(n_active, W=w, b=b).fit(X * mask, y)
        return self

    def predict_proba(self, X):
        return self.best_elm.predict_proba(X * self.best_mask_)
