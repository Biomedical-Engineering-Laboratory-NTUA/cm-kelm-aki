"""
CM-KELM: evolutionary search over a class-weighted Nystrom-KELM on the full training cohort.
=========================================================================================

Genome [log gamma, log lambda, z_1..z_d]; variable j is selected when z_j > 0.
Fitness: mean AUPRC of a stratified 3-fold CV on the training set (fixed folds
and landmarks) minus 0.001 x the number of selected variables.

Search: DE/rand/1/bin (F = 0.6, CR = 0.7, a trial replaces its target only if its
fitness is higher), guided by the cultural belief space of cmeelm.py
(normative ranges, situational exemplar, topographic islands, domain knowledge:
protected clinical variables are never deselected), and a Lamarckian Gaussian
hill-climb on the best 10 % of the population after each generation.
use_cultural=False / use_memetic=False give the component ablations; the
protection of the clinical variables is applied in every variant.
The number of landmarks is fixed during the search.
"""

import numpy as np

from cmeelm import CulturalBeliefSpace, memetic_local_search
from kelm_variant import KELMEncoding
from nystrom_kelm import NystromKELM
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score


def nystrom_kelm_fitness(genome, X, y, encoding, n_landmarks, complexity_penalty=0.001, n_splits=3,
                         rank=None, eig_rtol=None, cv_seed=0, lm_seed=0, return_fold_std=False):
    """Same structure as kelm_variant.kelm_fitness, but fits NystromKELM
    instead of exact KELM -- the only change needed to make full-cohort
    evolutionary search computationally feasible."""
    gamma, lam, mask = encoding.decode(genome)
    Xm = X * mask
    # cv_seed / lm_seed = 0 reproduce the original fixed-split fitness exactly
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)
    scores = []
    for tr_idx, val_idx in skf.split(Xm, y):
        model = NystromKELM(gamma=gamma, lam=lam, n_landmarks=n_landmarks,
                             random_state=lm_seed, rank=rank, eig_rtol=eig_rtol).fit(Xm[tr_idx], y[tr_idx])
        p = model.predict_proba(Xm[val_idx])
        try:
            scores.append(average_precision_score(y[val_idx], p))
        except ValueError:
            scores.append(0.0)
    penalty = complexity_penalty * mask.sum()
    if return_fold_std:
        return float(np.mean(scores) - penalty), float(np.std(scores))
    return float(np.mean(scores) - penalty)


class FullCohortCMKELM:
    """
    DE + Cultural + Memetic search over KELM hyperparameters + feature
    mask, using Nystrom-approximated KELM as the base model so the FULL
    training cohort can be used (not a subsample).

    Same public interface as kelm_variant.CMKELM (fit/predict_proba),
    so it is a drop-in replacement wherever full-cohort training is
    needed instead of the subsample-based exact-KELM version.
    """

    def __init__(self, population_size=20, generations=15, F=0.6, CR=0.7,
                 memetic_top_frac=0.1, use_cultural=True, use_memetic=True,
                 n_landmarks=1000, complexity_penalty=0.001,
                 protected_feature_indices=None, n_cells=4, migration_every=5,
                 random_state=0, rank=None, eig_rtol=None, backend="cpu", precision="fp64"):
        self.backend, self.precision = backend, precision   # 'gpu' -> gpu_fitness.NystromFitness
        self.rank = rank            # rank-k Nystrom (see nystrom_kelm.NystromKELM)
        self.eig_rtol = eig_rtol
        self.population_size = population_size
        self.generations = generations
        self.F = F
        self.CR = CR
        self.memetic_top_frac = memetic_top_frac
        self.use_cultural = use_cultural
        self.use_memetic = use_memetic
        self.n_landmarks = n_landmarks
        self.complexity_penalty = complexity_penalty
        self.protected_feature_indices = protected_feature_indices or []
        self.n_cells = n_cells
        self.migration_every = migration_every
        self.rng = np.random.default_rng(random_state)

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
            protected_feature_indices=self.protected_feature_indices, mask_gene_offset=2,
        )

        if self.backend == "cpu":
            fit_fn = lambda g: nystrom_kelm_fitness(g, X, y, self.encoding, self.n_landmarks,
                                                      self.complexity_penalty,
                                                      rank=self.rank, eig_rtol=self.eig_rtol)
        else:
            if self.rank is not None or self.eig_rtol is not None:
                raise ValueError("rank / eig_rtol are not supported on the GPU backend")
            from gpu_fitness import NystromFitness
            self.fitness_ = NystromFitness(X, y, self.encoding, self.n_landmarks,
                                           self.complexity_penalty,
                                           backend="cpu" if self.backend == "cpu-xp" else self.backend,
                                           precision=self.precision)
            fit_fn = self.fitness_

        pop = np.array([self.encoding.random_individual(self.rng) for _ in range(self.population_size)])
        pop = np.array([self.belief.enforce_domain_knowledge(ind) for ind in pop])
        fitnesses = np.array([fit_fn(ind) for ind in pop])

        for gen in range(self.generations):
            if self.use_cultural:
                self.belief.update(pop, fitnesses)

            new_pop = pop.copy()
            for i in range(self.population_size):
                partner_pool = self.belief.topographic_partners(i, gen) if self.use_cultural else None
                if partner_pool is None:
                    partner_pool = np.array([j for j in range(self.population_size) if j != i])
                else:
                    partner_pool = partner_pool[partner_pool != i]
                    if len(partner_pool) < 3:
                        partner_pool = np.array([j for j in range(self.population_size) if j != i])

                r1, r2, r3 = pop[self.rng.choice(partner_pool, 3, replace=False)]
                mutant = r1 + self.F * (r2 - r3)

                if self.use_cultural:
                    mutant = self.belief.clip_to_norms(mutant)
                    mutant = self.belief.pull_to_exemplar(mutant, strength=0.05)

                cross_mask = self.rng.random(genome_size) < self.CR
                trial = np.where(cross_mask, mutant, pop[i])
                trial = self.belief.enforce_domain_knowledge(trial)

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
                        pop[idx], fit_fn, step_size=0.1, max_iters=5, rng=self.rng)
                    refined = self.belief.enforce_domain_knowledge(refined)
                    if refined_fit > fitnesses[idx]:
                        pop[idx] = refined
                        fitnesses[idx] = refined_fit

            print(f"  Gen {gen+1}/{self.generations}: best fitness so far = {fitnesses.max():.4f}", flush=True)

        # keep the whole final population (and the island map) so that
        # population_ensemble.py can build an ensemble from this single run
        self.final_population_ = pop.copy()
        self.final_fitnesses_ = fitnesses.copy()
        self.cell_assignment_ = (None if self.belief.cell_assignment is None
                                 else np.array(self.belief.cell_assignment).copy())
        self._X_fit_shape = X.shape

        best_i = np.argmax(fitnesses)
        self.best_genome = pop[best_i]
        gamma, lam, mask = self.encoding.decode(self.best_genome)
        self.best_mask_ = mask
        self.best_gamma_, self.best_lambda_ = gamma, lam
        self.best_model = NystromKELM(gamma=gamma, lam=lam, n_landmarks=self.n_landmarks,
                                       random_state=0, rank=self.rank,
                                       eig_rtol=self.eig_rtol).fit(X * mask, y)
        return self

    def predict_proba(self, X):
        return self.best_model.predict_proba(X * self.best_mask_)
