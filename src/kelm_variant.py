"""
Kernel ELM (KELM) architecture variant
======================================================================

Used by the article only for the genome encoding (Encoding: log-gamma, log-lambda
and one selection gene per variable; log-gamma and log-lambda are clipped to
[-10, 5] when decoded). The exact-KELM classes and kelm_fitness below are the
small-sample prototypes that preceded the Nystrom version (nystrom_cmkelm.py);
they are kept for completeness and are not used for any result.

Standard ELM projects inputs through a RANDOM hidden layer (W, b), which
is fast but can be a weak nonlinear feature map for some problems (as our
hard-dataset stress test suggested).

KELM (Huang et al., 2012) replaces that random projection with an
explicit kernel function -- mathematically related to SVM's kernel trick,
but still solved in closed form (kernel ridge regression), so it keeps
ELM's training-speed advantage over SVM's iterative QP solver.

  f(x) = sum_i alpha_i * K(x, x_i),   alpha = (K + lambda*I)^-1 y

This module implements:
  - KELM: the base kernel-ELM model (RBF kernel)
  - CMKELM: the same DE + Cultural + Memetic scaffolding as CMEELM, but
    evolving kernel hyperparameters (log-gamma, log-lambda) and a feature
    mask instead of random hidden weights -- a MUCH smaller genome than
    CMEELM's, since there is no (n_features x n_hidden) weight matrix to
    search.
  - Class-weighted regularization option, since AKI-like problems are
    imbalanced and standard ridge regression implicitly favors the
    majority class.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score

from cmeelm import CulturalBeliefSpace, memetic_local_search


# ----------------------------------------------------------------------
# Kernel ELM base model
# ----------------------------------------------------------------------
class KELM:
    def __init__(self, gamma=0.1, lam=1.0, class_weighted=True):
        self.gamma = gamma
        self.lam = lam
        self.class_weighted = class_weighted
        self.X_train = None
        self.alpha = None

    @staticmethod
    def _rbf_kernel(A, B, gamma):
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b, computed via broadcasting
        A_sq = np.sum(A ** 2, axis=1).reshape(-1, 1)
        B_sq = np.sum(B ** 2, axis=1).reshape(1, -1)
        sq_dists = A_sq + B_sq - 2 * A @ B.T
        sq_dists = np.maximum(sq_dists, 0)
        return np.exp(-gamma * sq_dists)

    def fit(self, X, y):
        self.X_train = X
        K = self._rbf_kernel(X, X, self.gamma)
        n = X.shape[0]

        if self.class_weighted:
            # inverse-frequency weights so the minority class contributes
            # comparably to the majority class in the ridge solve.
            # Correct weighted dual form: alpha = (lambda*I + W@K)^-1 @ (W @ target)
            counts = np.bincount(y.astype(int))
            w = np.where(y == 1, n / (2.0 * counts[1]), n / (2.0 * counts[0]))
            W = np.diag(w)
            target = 2 * y - 1
            self.alpha = np.linalg.solve(self.lam * np.eye(n) + W @ K, W @ target)
        else:
            self.alpha = np.linalg.solve(K + self.lam * np.eye(n), (2 * y - 1))
        return self

    def predict_proba(self, X):
        K = self._rbf_kernel(X, self.X_train, self.gamma)
        raw = K @ self.alpha
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -500, 500)))


# ----------------------------------------------------------------------
# Genome encoding for KELM: [log_gamma, log_lambda, mask_logits...]
# ----------------------------------------------------------------------
class KELMEncoding:
    def __init__(self, n_features):
        self.n_features = n_features
        self.total_size = 2 + n_features

    def random_individual(self, rng):
        log_gamma = rng.uniform(np.log(1e-3), np.log(10))
        log_lambda = rng.uniform(np.log(1e-3), np.log(100))
        mask_logits = rng.normal(0, 1, self.n_features)
        return np.concatenate([[log_gamma, log_lambda], mask_logits])

    def decode(self, genome):
        log_gamma, log_lambda = genome[0], genome[1]
        gamma = np.exp(np.clip(log_gamma, -10, 5))
        lam = np.exp(np.clip(log_lambda, -10, 5))
        mask_logits = genome[2:]
        mask = (mask_logits > 0).astype(float)
        if mask.sum() == 0:
            mask[np.argmax(mask_logits)] = 1.0
        return gamma, lam, mask


def kelm_fitness(genome, X, y, encoding, complexity_penalty=0.001, n_splits=2, max_train=400):
    gamma, lam, mask = encoding.decode(genome)
    Xm = X * mask
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = []
    for tr_idx, val_idx in skf.split(Xm, y):
        # subsample the training fold for speed during the search itself --
        # KELM's O(n^3) solve makes full-size fitness evaluation too slow
        # for a demo-scale search budget; the FINAL model is still refit on
        # the full training data after search (see CMKELM.fit).
        if len(tr_idx) > max_train:
            rng = np.random.default_rng(0)
            tr_idx = rng.choice(tr_idx, max_train, replace=False)
        model = KELM(gamma=gamma, lam=lam).fit(Xm[tr_idx], y[tr_idx])
        p = model.predict_proba(Xm[val_idx])
        try:
            scores.append(average_precision_score(y[val_idx], p))
        except ValueError:
            scores.append(0.0)
    penalty = complexity_penalty * mask.sum()
    return float(np.mean(scores) - penalty)


# ----------------------------------------------------------------------
# DE + Cultural + Memetic search over KELM hyperparameters + feature mask
# ----------------------------------------------------------------------
class CMKELM:
    def __init__(self, population_size=20, generations=15, F=0.6, CR=0.7,
                 memetic_top_frac=0.1, use_cultural=True, use_memetic=True,
                 use_fuzzy=False, complexity_penalty=0.001, protected_feature_indices=None,
                 n_cells=4, migration_every=5, random_state=0):
        self.population_size = population_size
        self.generations = generations
        self.F = F
        self.CR = CR
        self.memetic_top_frac = memetic_top_frac
        self.use_cultural = use_cultural
        self.use_memetic = use_memetic
        self.use_fuzzy = use_fuzzy
        self.complexity_penalty = complexity_penalty
        self.protected_feature_indices = protected_feature_indices or []
        self.n_cells = n_cells
        self.migration_every = migration_every
        self.rng = np.random.default_rng(random_state)
        self._fuzzy_ctrl = None

        self.encoding = None
        self.belief = None
        self.best_genome = None
        self.best_model = None

    def fit(self, X, y):
        n_features = X.shape[1]
        self.encoding = KELMEncoding(n_features)
        genome_size = self.encoding.total_size
        self.belief = CulturalBeliefSpace(
            genome_size, n_cells=self.n_cells, migration_every=self.migration_every,
            protected_feature_indices=self.protected_feature_indices,
            mask_gene_offset=2,  # genome layout: [log_gamma, log_lambda, mask...]
        )

        pop = np.array([self.encoding.random_individual(self.rng)
                         for _ in range(self.population_size)])
        fit_fn = lambda g: kelm_fitness(g, X, y, self.encoding, self.complexity_penalty)
        # domain knowledge applies unconditionally, even without use_cultural,
        # since it encodes a hard clinical requirement rather than a search heuristic
        pop = np.array([self.belief.enforce_domain_knowledge(ind) for ind in pop])
        fitnesses = np.array([fit_fn(ind) for ind in pop])

        if self.use_fuzzy:
            from fuzzy_control import FuzzyDEController, fuzzy_accept
            self._fuzzy_ctrl = FuzzyDEController()

        for gen in range(self.generations):
            if self.use_cultural:
                self.belief.update(pop, fitnesses)

            if self.use_fuzzy:
                F_gen, CR_gen = self._fuzzy_ctrl.update(pop, float(fitnesses.max()))
            else:
                F_gen, CR_gen = self.F, self.CR

            new_pop = pop.copy()
            for i in range(self.population_size):
                # topographic knowledge: draw DE partners from the same
                # island most generations, whole population on migration gens
                partner_pool = self.belief.topographic_partners(i, gen) if self.use_cultural else None
                if partner_pool is None:
                    partner_pool = np.array([j for j in range(self.population_size) if j != i])
                else:
                    partner_pool = partner_pool[partner_pool != i]
                    if len(partner_pool) < 3:
                        partner_pool = np.array([j for j in range(self.population_size) if j != i])

                r1, r2, r3 = pop[self.rng.choice(partner_pool, 3, replace=False)]
                mutant = r1 + F_gen * (r2 - r3)

                if self.use_cultural:
                    mutant = self.belief.clip_to_norms(mutant)
                    mutant = self.belief.pull_to_exemplar(mutant, strength=0.05)

                cross_mask = self.rng.random(genome_size) < CR_gen
                trial = np.where(cross_mask, mutant, pop[i])
                trial = self.belief.enforce_domain_knowledge(trial)  # hard clinical constraint

                trial_fit = fit_fn(trial)
                if self.use_fuzzy:
                    # noise_estimate: fixed small default here; a refined version could
                    # pass the actual std across the internal CV folds from kelm_fitness
                    accepted = fuzzy_accept(fitnesses[i], trial_fit, noise_estimate=0.01, rng=self.rng)
                else:
                    accepted = trial_fit > fitnesses[i]
                if accepted:
                    new_pop[i] = trial
                    fitnesses[i] = trial_fit
            pop = new_pop

            if self.use_memetic:
                top_k = max(1, int(self.population_size * self.memetic_top_frac))
                top_idx = np.argsort(fitnesses)[::-1][:top_k]
                for idx in top_idx:
                    refined, refined_fit = memetic_local_search(
                        pop[idx], fit_fn, step_size=0.1, max_iters=5, rng=self.rng,
                        repair=self.belief.enforce_domain_knowledge)
                    if refined_fit > fitnesses[idx]:
                        pop[idx] = refined
                        fitnesses[idx] = refined_fit

        best_i = np.argmax(fitnesses)
        self.best_genome = pop[best_i]
        gamma, lam, mask = self.encoding.decode(self.best_genome)
        self.best_mask_ = mask
        self.best_gamma_, self.best_lambda_ = gamma, lam
        self.best_model = KELM(gamma=gamma, lam=lam).fit(X * mask, y)
        return self

    def predict_proba(self, X):
        return self.best_model.predict_proba(X * self.best_mask_)


# ----------------------------------------------------------------------
# GA backbone for KELM (same encoding/belief-space/memetic scaffolding)
# ----------------------------------------------------------------------
class GAKELM:
    def __init__(self, population_size=20, generations=15, crossover_rate=0.8,
                 mutation_rate=0.15, mutation_std=0.3, elite_frac=0.1,
                 memetic_top_frac=0.1, use_cultural=True, use_memetic=True,
                 complexity_penalty=0.001, random_state=0):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.elite_frac = elite_frac
        self.memetic_top_frac = memetic_top_frac
        self.use_cultural = use_cultural
        self.use_memetic = use_memetic
        self.complexity_penalty = complexity_penalty
        self.rng = np.random.default_rng(random_state)
        self.encoding = None
        self.belief = None
        self.best_genome = None
        self.best_model = None

    def _tournament_select(self, pop, fitnesses, pool_indices, k=3):
        k = min(k, len(pool_indices))
        idxs = self.rng.choice(pool_indices, k, replace=False)
        return pop[idxs[np.argmax(fitnesses[idxs])]]

    def fit(self, X, y):
        n_features = X.shape[1]
        self.encoding = KELMEncoding(n_features)
        genome_size = self.encoding.total_size
        self.belief = CulturalBeliefSpace(genome_size, n_cells=4, migration_every=5)

        pop = np.array([self.encoding.random_individual(self.rng)
                         for _ in range(self.population_size)])
        fit_fn = lambda g: kelm_fitness(g, X, y, self.encoding, self.complexity_penalty)
        fitnesses = np.array([fit_fn(ind) for ind in pop])
        n_elite = max(1, int(self.population_size * self.elite_frac))

        for gen in range(self.generations):
            if self.use_cultural:
                self.belief.update(pop, fitnesses)
            order = np.argsort(fitnesses)[::-1]
            new_pop = [pop[i].copy() for i in order[:n_elite]]

            child_idx = len(new_pop)
            while len(new_pop) < self.population_size:
                if self.use_cultural:
                    pool = self.belief.topographic_partners(child_idx % self.population_size, gen)
                    if pool is None or len(pool) < 3:
                        pool = np.arange(self.population_size)
                else:
                    pool = np.arange(self.population_size)

                p1 = self._tournament_select(pop, fitnesses, pool)
                p2 = self._tournament_select(pop, fitnesses, pool)
                if self.rng.random() < self.crossover_rate:
                    alpha = self.rng.random(genome_size)
                    child = alpha * p1 + (1 - alpha) * p2
                else:
                    child = p1.copy()
                mutate_mask = self.rng.random(genome_size) < self.mutation_rate
                child[mutate_mask] += self.rng.normal(0, self.mutation_std, mutate_mask.sum())
                if self.use_cultural:
                    child = self.belief.clip_to_norms(child)
                    child = self.belief.pull_to_exemplar(child, strength=0.05)
                new_pop.append(child)
                child_idx += 1

            pop = np.array(new_pop[: self.population_size])
            fitnesses = np.array([fit_fn(ind) for ind in pop])

            if self.use_memetic:
                top_k = max(1, int(self.population_size * self.memetic_top_frac))
                top_idx = np.argsort(fitnesses)[::-1][:top_k]
                for idx in top_idx:
                    refined, refined_fit = memetic_local_search(
                        pop[idx], fit_fn, step_size=0.1, max_iters=5, rng=self.rng)
                    if refined_fit > fitnesses[idx]:
                        pop[idx] = refined
                        fitnesses[idx] = refined_fit

        best_i = np.argmax(fitnesses)
        self.best_genome = pop[best_i]
        gamma, lam, mask = self.encoding.decode(self.best_genome)
        self.best_mask_ = mask
        self.best_model = KELM(gamma=gamma, lam=lam).fit(X * mask, y)
        return self

    def predict_proba(self, X):
        return self.best_model.predict_proba(X * self.best_mask_)


# ----------------------------------------------------------------------
# PSO backbone for KELM
# ----------------------------------------------------------------------
class PSOKELM:
    def __init__(self, population_size=20, generations=15, inertia=0.7, c1=1.5, c2=1.5,
                 velocity_clip=1.0, memetic_top_frac=0.1, use_cultural=True, use_memetic=True,
                 complexity_penalty=0.001, random_state=0):
        self.population_size = population_size
        self.generations = generations
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.velocity_clip = velocity_clip
        self.memetic_top_frac = memetic_top_frac
        self.use_cultural = use_cultural
        self.use_memetic = use_memetic
        self.complexity_penalty = complexity_penalty
        self.rng = np.random.default_rng(random_state)
        self.encoding = None
        self.belief = None
        self.best_genome = None
        self.best_model = None

    def fit(self, X, y):
        n_features = X.shape[1]
        self.encoding = KELMEncoding(n_features)
        genome_size = self.encoding.total_size
        self.belief = CulturalBeliefSpace(genome_size, n_cells=4, migration_every=5)
        fit_fn = lambda g: kelm_fitness(g, X, y, self.encoding, self.complexity_penalty)

        pos = np.array([self.encoding.random_individual(self.rng)
                         for _ in range(self.population_size)])
        vel = self.rng.normal(0, 0.1, pos.shape)
        fitnesses = np.array([fit_fn(p) for p in pos])
        pbest_pos = pos.copy()
        pbest_fit = fitnesses.copy()
        gbest_i = np.argmax(pbest_fit)
        gbest_pos = pbest_pos[gbest_i].copy()
        gbest_fit = pbest_fit[gbest_i]

        for gen in range(self.generations):
            if self.use_cultural:
                self.belief.update(pos, fitnesses)

            if self.use_cultural:
                # lbest variant: each particle is pulled toward the best
                # individual within its own island, not the global best --
                # except on migration generations, where islands mix.
                target_pos = np.array([
                    pbest_pos[self.belief.island_best_index(i, pbest_fit, gen)]
                    for i in range(self.population_size)
                ])
            else:
                target_pos = np.tile(gbest_pos, (self.population_size, 1))

            r1 = self.rng.random(pos.shape)
            r2 = self.rng.random(pos.shape)
            vel = (self.inertia * vel + self.c1 * r1 * (pbest_pos - pos)
                   + self.c2 * r2 * (target_pos - pos))
            vel = np.clip(vel, -self.velocity_clip, self.velocity_clip)
            pos = pos + vel
            if self.use_cultural:
                pos = np.array([self.belief.clip_to_norms(p) for p in pos])
            fitnesses = np.array([fit_fn(p) for p in pos])

            if self.use_memetic:
                top_k = max(1, int(self.population_size * self.memetic_top_frac))
                top_idx = np.argsort(fitnesses)[::-1][:top_k]
                for idx in top_idx:
                    refined, refined_fit = memetic_local_search(
                        pos[idx], fit_fn, step_size=0.1, max_iters=5, rng=self.rng)
                    if refined_fit > fitnesses[idx]:
                        pos[idx] = refined
                        fitnesses[idx] = refined_fit

            improved = fitnesses > pbest_fit
            pbest_pos[improved] = pos[improved]
            pbest_fit[improved] = fitnesses[improved]
            if pbest_fit.max() > gbest_fit:
                gbest_i = np.argmax(pbest_fit)
                gbest_pos = pbest_pos[gbest_i].copy()
                gbest_fit = pbest_fit[gbest_i]

        self.best_genome = gbest_pos
        gamma, lam, mask = self.encoding.decode(self.best_genome)
        self.best_mask_ = mask
        self.best_model = KELM(gamma=gamma, lam=lam).fit(X * mask, y)
        return self

    def predict_proba(self, X):
        return self.best_model.predict_proba(X * self.best_mask_)
