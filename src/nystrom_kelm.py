"""
Class-weighted Nystrom kernel extreme learning machine (KELM).
==============================================================

Exact KELM needs an n x n solve, impractical at n ~ 35,000 when it is repeated
hundreds of times inside the evolutionary search. The Nystrom method
(Williams & Seeger 2001) uses m << n landmarks:
  1. select m landmarks from the training set (uniform random by default);
  2. K_mm = V Lambda V^T (eigendecomposition) and K_nm (training points x landmarks);
  3. explicit feature map U = K_nm V Lambda^{-1/2} (n x m);
  4. class-weighted ridge solve (U^T W U + lambda I) alpha = U^T W t,
     t in {-1, +1}, W_ii = n / (2 n_c(i)), so that both classes carry equal weight;
  5. predicted risk sigma(k(x, L) V Lambda^{-1/2} alpha), sigma the logistic function.
Cost O(n m^2 + m^3) instead of O(n^3). For numerical stability a ridge of 1e-8
is added to K_mm and its eigenvalues are clipped at 1e-10.

Options not used in the paper: rank-k Nystrom (`rank`), a relative eigenvalue
threshold (`eig_rtol`) and k-means++ landmarks (`landmark_strategy="kmeanspp"`).
"""

import numpy as np


class NystromKELM:
    def __init__(self, gamma=0.1, lam=1.0, n_landmarks=500,
                 class_weighted=True, landmark_strategy="random", random_state=0,
                 rank=None, eig_rtol=None):
        """
        rank     : keep only the top-`rank` eigenpairs of K_mm (rank-k Nystrom).
                   The feature map becomes n x k instead of n x m, so the two
                   dominant O(n*m^2) steps drop to O(n*m*k) and O(n*k^2).
        eig_rtol : alternatively (or additionally) drop eigenpairs with
                   lambda_i <= eig_rtol * lambda_max (the C++ backend uses 1e-10).
        Both None (default) reproduces the original behaviour exactly
        (all m eigenpairs, absolute clip at 1e-10).
        """
        self.gamma = gamma
        self.lam = lam
        self.n_landmarks = n_landmarks
        self.class_weighted = class_weighted
        self.landmark_strategy = landmark_strategy  # "random" or "kmeanspp"
        self.rank = rank
        self.eig_rtol = eig_rtol
        self.rng = np.random.default_rng(random_state)
        self.Kmm_inv_sqrt_ = None
        self.alpha_ = None
        self.beta_ = None

    @staticmethod
    def _rbf_kernel(A, B, gamma):
        A_sq = np.sum(A ** 2, axis=1).reshape(-1, 1)
        B_sq = np.sum(B ** 2, axis=1).reshape(1, -1)
        sq_dists = np.maximum(A_sq + B_sq - 2 * A @ B.T, 0)
        return np.exp(-gamma * sq_dists)

    def _select_landmarks(self, X, m):
        n = X.shape[0]
        if self.landmark_strategy == "random":
            # uniform random landmarks (default, used in the paper)
            idx = self.rng.choice(n, m, replace=False)
            return X[idx]

        elif self.landmark_strategy == "kmeanspp":
            # optional k-means++ seeding (not used in the paper)
            centers_idx = [self.rng.integers(n)]
            closest_sq_dist = np.sum((X - X[centers_idx[0]]) ** 2, axis=1)
            for _ in range(1, m):
                probs = closest_sq_dist / (closest_sq_dist.sum() + 1e-12)
                next_idx = self.rng.choice(n, p=probs)
                centers_idx.append(next_idx)
                new_sq_dist = np.sum((X - X[next_idx]) ** 2, axis=1)
                closest_sq_dist = np.minimum(closest_sq_dist, new_sq_dist)
            return X[centers_idx]

        else:
            raise ValueError(f"Unknown landmark_strategy: {self.landmark_strategy}")

    def _inv_sqrt_factor(self, Kmm):
        """Returns F (m x k) with K_mm^{-1/2} = F F^T restricted to the kept
        subspace, i.e. F = V_k diag(lambda_k^{-1/2}).

        The original code formed the full m x m matrix V diag() V^T, which
        costs an extra O(m^3) GEMM and is unnecessary: U = K_nm F differs from
        K_nm V diag() V^T only by the orthogonal rotation V^T, and the ridge
        solution (hence every prediction) is invariant to that rotation.
        """
        m = Kmm.shape[0]
        if self.rank is not None and self.rank < m:
            # partial spectrum (LAPACK dsyevr): only the top-k eigenpairs,
            # the deterministic counterpart of a randomized SVD
            from scipy.linalg import eigh as sp_eigh
            eigvals, eigvecs = sp_eigh(Kmm, subset_by_index=[m - self.rank, m - 1],
                                       driver="evr", check_finite=False)
        else:
            eigvals, eigvecs = np.linalg.eigh(Kmm)
        if self.eig_rtol is not None:
            keep = eigvals > self.eig_rtol * eigvals.max()
            eigvals, eigvecs = eigvals[keep], eigvecs[:, keep]
        elif self.rank is None:
            eigvals = np.clip(eigvals, 1e-10, None)  # original behaviour
        self.eigvals_ = eigvals[::-1].copy()          # descending, for diagnostics
        self.effective_rank_ = eigvals.size
        return eigvecs / np.sqrt(eigvals)

    def fit(self, X, y):
        n = X.shape[0]
        m = min(self.n_landmarks, n)

        self.landmarks_ = self._select_landmarks(X, m)

        Kmm = self._rbf_kernel(self.landmarks_, self.landmarks_, self.gamma)
        Kmm += 1e-8 * np.eye(m)  # small ridge for numerical stability

        F = self._inv_sqrt_factor(Kmm)          # m x k
        self.Kmm_inv_sqrt_ = F
        k = F.shape[1]

        Knm = self._rbf_kernel(X, self.landmarks_, self.gamma)
        U = Knm @ F                              # n x k explicit feature map

        target = 2 * y - 1
        if self.class_weighted:
            counts = np.bincount(y.astype(int))
            w = np.where(y == 1, n / (2.0 * counts[1]), n / (2.0 * counts[0]))
            # W never materialized (see earlier performance fix). Scaling rows
            # by sqrt(w) makes the Gram matrix a symmetric A^T A product, which
            # NumPy dispatches to BLAS syrk (half the flops of a general GEMM),
            # as the C++ backend does with cblas_dsyrk.
            sw = np.sqrt(w)
            Us = U * sw[:, None]
            G = Us.T @ Us
            rhs = Us.T @ (target * sw)
        else:
            G = U.T @ U
            rhs = U.T @ target
        G[np.diag_indices_from(G)] += self.lam
        self.alpha_ = np.linalg.solve(G, rhs)
        # fold F into the weights once, so prediction is O(n_new*m) instead of
        # O(n_new*m*k) -- matters because every fitness evaluation predicts
        # on a ~12,000-case validation fold (again as in the C++ version)
        self.beta_ = F @ self.alpha_
        return self

    def predict_proba(self, X_new):
        Knm_new = self._rbf_kernel(X_new, self.landmarks_, self.gamma)
        raw = Knm_new @ self.beta_
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -500, 500)))
