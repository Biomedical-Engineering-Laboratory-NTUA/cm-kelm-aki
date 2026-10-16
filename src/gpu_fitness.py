"""
GPU (CuPy) fitness evaluation for full-cohort Nystrom-KELM search.
=================================================================

Why: after the CPU optimisations, one fitness evaluation (3-fold inner CV,
m=1000, ~24k training rows per fold) costs ~1 s on the 24-core EPYC, and a
full run (pop 20, gen 15) 5-15 min. The planned hybrid multimeme search
(stronger local search, more evaluations per generation) multiplies that.
This module moves the whole fitness evaluation -- kernel blocks, K_mm
eigendecomposition, feature map, Gram matrix, ridge solve and validation
prediction -- onto the GPU, with the data and the CV folds resident there.

Design
------
* One code path for both backends: `xp` is numpy (backend='cpu') or cupy
  (backend='gpu'). The numpy path reproduces nystrom_kelm.NystromKELM
  (default settings) operation by operation, so the GPU can be validated
  against it and against the existing CPU fitness function.
* Landmarks and folds are drawn exactly as on the CPU (NumPy RNG,
  StratifiedKFold), so in fp64 a GPU run follows the same search trajectory
  as a CPU run with the same seed; test_gpu_fitness.py and a replicated run
  check this.
* precision='fp64' (default): everything in double precision, same algorithm
  as the CPU code (reproduces CPU runs).
* precision='mixed': the RTX 6000 Ada has low FP64 throughput (1/64 of FP32).
  First version (FP32 feature map U = K_nm K_mm^{-1/2}) failed on the GPU for
  very small gamma (fitness errors up to 1e-2, 27 Sep): K_mm^{-1/2} has entries
  up to 1e5 and FP32 round-off is amplified. Current version: equivalent
  subset-of-regressors system, FP32 only for the O(n m^2) Gram product used as
  a preconditioner, FP64-residual conjugate gradient, automatic FP64 fallback
  if the residual test fails. Accuracy is therefore guaranteed by the residual
  check, not assumed.
* AUPRC is computed on the host with sklearn on the ~12k validation scores
  (a 100 KB transfer), so it is bit-compatible with every other number we
  report.

Not (yet) supported on the GPU path: rank-k truncation, eig_rtol, k-means++
landmarks. They raise instead of silently falling back.
"""

import time

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

try:
    import cupy as cp
except ImportError:  # CPU-only machine: the numpy path still works
    cp = None


def get_xp(backend):
    if backend == "cpu":
        return np
    if backend == "gpu":
        if cp is None:
            raise ImportError("backend='gpu' needs CuPy: pip install 'cupy-cuda12x[ctk]'")
        return cp
    raise ValueError(f"unknown backend '{backend}'")


def _to_host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


class XPNystromKELM:
    """Nystrom-KELM on numpy or cupy arrays. Same maths as
    nystrom_kelm.NystromKELM with rank=None, eig_rtol=None, class_weighted=True,
    landmark_strategy='random'."""

    def __init__(self, gamma, lam, n_landmarks, random_state=0, backend="gpu", precision="fp64"):
        self.gamma, self.lam, self.n_landmarks = gamma, lam, n_landmarks
        self.random_state = random_state
        self.xp = get_xp(backend)
        if precision not in ("fp64", "mixed"):
            raise ValueError("precision must be 'fp64' or 'mixed'")
        self.precision = precision

    def _rbf(self, A, B, a_sq=None):
        xp = self.xp
        if a_sq is None:
            a_sq = xp.sum(A * A, axis=1)
        b_sq = xp.sum(B * B, axis=1)
        d = a_sq[:, None] + b_sq[None, :] - 2.0 * (A @ B.T)
        xp.maximum(d, 0, out=d)
        d *= -self.gamma
        xp.exp(d, out=d)
        return d

    def fit(self, X, y, landmark_idx=None):
        """X: (n, d) xp array (fp64). y: (n,) int xp array of 0/1.
        landmark_idx: optional host index array; default reproduces
        np.random.default_rng(random_state).choice(n, m, replace=False)."""
        xp = self.xp
        n = X.shape[0]
        m = min(self.n_landmarks, n)
        if landmark_idx is None:
            landmark_idx = np.random.default_rng(self.random_state).choice(n, m, replace=False)
        L = X[xp.asarray(landmark_idx)]

        Kmm = self._rbf(L, L)
        Kmm[xp.arange(m), xp.arange(m)] += 1e-8

        y_f = y.astype(xp.float64)
        n1 = float(_to_host(y_f.sum()))
        n0 = n - n1
        w = xp.where(y == 1, n / (2.0 * n1), n / (2.0 * n0))
        sw = xp.sqrt(w)
        target = 2.0 * y_f - 1.0

        if self.precision == "fp64":
            eigvals, eigvecs = xp.linalg.eigh(Kmm)          # (mixed mode needs no eigh)
            eigvals = xp.clip(eigvals, 1e-10, None)
            F = eigvecs / xp.sqrt(eigvals)                  # m x m, fp64
            Knm = self._rbf(X, L)
            Us = (Knm @ F) * sw[:, None]
            G = Us.T @ Us
            rhs = Us.T @ (target * sw)
            del Knm, Us
            G[xp.arange(m), xp.arange(m)] += self.lam
            alpha = xp.linalg.solve(G, rhs)
            self.beta_ = F @ alpha
        else:
            self.beta_ = self._solve_mixed(X, L, Kmm, sw, target)
        self.landmarks_ = L
        return self

    def _solve_mixed(self, X, L, Kmm, sw, target, tol=1e-10, maxit=25):
        """Mixed precision, accuracy guaranteed by an FP64 residual check.

        Solves the algebraically equivalent subset-of-regressors system
            (Ks^T Ks + lam K_mm) beta = Ks^T (sw * t),   Ks = diag(sw) K_nm
        (equal to the U-space solution; verified to ~1e-7 in fitness).
        The only O(n m^2) product, Ks^T Ks, is formed in FP32 and used ONLY as
        a preconditioner; preconditioned CG then iterates with FP64 residuals
        (two FP64 mat-vecs per step, O(n m)). If the relative residual does not
        reach `tol` within `maxit` steps (very ill-conditioned K_mm, e.g. tiny
        gamma), the system is rebuilt fully in FP64 -- never a silent loss of
        accuracy. Counters: self.cg_iters_, self.fallback_."""
        xp = self.xp
        m = L.shape[0]
        Ks = self._rbf(X, L) * sw[:, None]                 # FP64, n x m (cheap: d=37)
        Ks32 = Ks.astype(xp.float32)
        C = (Ks32.T @ Ks32).astype(xp.float64)             # the expensive product, FP32
        del Ks32
        C = 0.5 * (C + C.T)
        lamK = self.lam * Kmm
        b = Ks.T @ (target * sw)
        P = xp.linalg.inv(C + lamK)                         # preconditioner (approx. inverse)

        def A(v):
            return Ks.T @ (Ks @ v) + lamK @ v

        x = P @ b
        r = b - A(x)
        z = P @ r
        p = z.copy()
        rz = float(r @ z)
        bn = float(xp.linalg.norm(b)) or 1.0
        self.cg_iters_, self.fallback_ = 0, False
        for it in range(maxit):
            if float(xp.linalg.norm(r)) / bn < tol:
                break
            Ap = A(p)
            alpha = rz / float(p @ Ap)
            x += alpha * p
            r -= alpha * Ap
            z = P @ r
            rz_new = float(r @ z)
            p = z + (rz_new / rz) * p
            rz = rz_new
            self.cg_iters_ = it + 1
        if float(xp.linalg.norm(r)) / bn >= tol:
            self.fallback_ = True
            x = xp.linalg.solve(Ks.T @ Ks + lamK, b)        # full FP64
        return x


    def decision(self, X_new):
        # FP64 in both modes: O(n_val m d + n_val m) work, and beta can have large
        # cancelling components when K_mm is ill-conditioned
        return self._rbf(X_new, self.landmarks_) @ self.beta_

    def predict_proba(self, X_new):
        """Returns a HOST numpy array (same convention as NystromKELM)."""
        raw = _to_host(self.decision(X_new))
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -500, 500)))


class NystromFitness:
    """Callable fitness with data and folds resident on the device.

    f = NystromFitness(Xtr, ytr, encoding, n_landmarks=1000, backend='gpu')
    value           = f(genome)                              # cv_seed=0, lm_seed=0
    value, fold_std = f(genome, cv_seed=1003, lm_seed=0, return_fold_std=True)

    Identical definition to nystrom_cmkelm.nystrom_kelm_fitness: mean inner
    CV-AUPRC over `n_splits` stratified folds minus complexity_penalty * |mask|.
    """

    def __init__(self, X, y, encoding, n_landmarks, complexity_penalty=0.001, n_splits=3,
                 backend="gpu", precision="fp64"):
        self.xp = get_xp(backend)
        self.backend, self.precision = backend, precision
        self.encoding = encoding
        self.n_landmarks = n_landmarks
        self.complexity_penalty = complexity_penalty
        self.n_splits = n_splits
        self.y_host = np.asarray(y).astype(int)
        self.X = self.xp.asarray(np.ascontiguousarray(X, dtype=np.float64))
        self.y = self.xp.asarray(self.y_host)
        self._folds = {}
        self.n_evals, self.seconds = 0, 0.0
        self.n_fits, self.n_fallback, self.cg_iters = 0, 0, 0     # mixed-precision diagnostics

    def _get_folds(self, cv_seed):
        if cv_seed not in self._folds:
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=cv_seed)
            folds = []
            for tr, va in skf.split(np.zeros(len(self.y_host)), self.y_host):
                folds.append((tr, va, self.xp.asarray(tr), self.xp.asarray(va)))
            self._folds[cv_seed] = folds
        return self._folds[cv_seed]

    def __call__(self, genome, cv_seed=0, lm_seed=0, return_fold_std=False):
        t0 = time.perf_counter()
        xp = self.xp
        gamma, lam, mask = self.encoding.decode(genome)
        Xm = self.X * xp.asarray(mask)
        scores = []
        for tr_h, va_h, tr, va in self._get_folds(cv_seed):
            model = XPNystromKELM(gamma, lam, self.n_landmarks, random_state=lm_seed,
                                  backend=self.backend, precision=self.precision)
            model.fit(Xm[tr], self.y[tr])
            if self.precision == "mixed":
                self.n_fits += 1
                self.n_fallback += int(model.fallback_)
                self.cg_iters += model.cg_iters_
            p = model.predict_proba(Xm[va])
            try:
                scores.append(average_precision_score(self.y_host[va_h], p))
            except ValueError:
                scores.append(0.0)
        value = float(np.mean(scores) - self.complexity_penalty * mask.sum())
        self.n_evals += 1
        self.seconds += time.perf_counter() - t0
        if return_fold_std:
            return value, float(np.std(scores))
        return value

    def summary(self):
        per = self.seconds / max(self.n_evals, 1)
        txt = f"{self.backend}/{self.precision}: {self.n_evals} evaluations, {self.seconds:.1f} s ({per:.3f} s each)"
        if self.precision == "mixed" and self.n_fits:
            txt += (f"; CG {self.cg_iters / self.n_fits:.1f} iters/fit, FP64 fallback in "
                    f"{self.n_fallback}/{self.n_fits} fits")
        return txt
