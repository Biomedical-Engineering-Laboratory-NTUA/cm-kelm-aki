//
// MIT License

// Copyright (c) 2026 Massimo Cafaro, Ph.D.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// October 20, 2026
// Nystrom-KELM v0.9.3
//
// This code implements, in C++ using the CUDA Toolkit and related libraries,
// a Nystrom-approximated Kernel Estreme Learning Machine
// by using a standard kernel ridge regression; e.g. Williams & Seeger 2001
//

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>



// error checking macros
#define CUDA_CHECK(call)                                                                                                                                                                               \
  do {                                                                                                                                                                                                 \
    cudaError_t err__ = (call);                                                                                                                                                                        \
    if (err__ != cudaSuccess) {                                                                                                                                                                        \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> " << cudaGetErrorString(err__) << "\n";                                                                                     \
      std::exit(1);                                                                                                                                                                                    \
    }                                                                                                                                                                                                  \
  } while (0)

#define CUBLAS_CHECK(call)                                                                                                                                                                             \
  do {                                                                                                                                                                                                 \
    cublasStatus_t s__ = (call);                                                                                                                                                                       \
    if (s__ != CUBLAS_STATUS_SUCCESS) {                                                                                                                       \
      std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " -> code " << s__ << "\n";                                                                                                    \
      std::exit(1);                                                                                                                                                                                    \
    }                                                                                                                                                                                                  \
  } while (0)

#define CUSOLVER_CHECK(call)                                                                                                                                                                           \
  do {                                                                                                                                                                                                 \
    cusolverStatus_t s__ = (call);                                                                                                                                                                     \
    if (s__ != CUSOLVER_STATUS_SUCCESS) {                                                                                                                     \
      std::cerr << "cuSOLVER error at " << __FILE__ << ":" << __LINE__ << " -> code " << s__ << "\n";                                                                                              \
      std::exit(1);                                                                                                                                                                                    \
    }                                                                                                                                                                                                  \
  } while (0)

// CUDA kernel to compute the L2 squared norm of rows (row-major layout)
__global__ void compute_row_norms_kernel(const double *__restrict__ X, double *__restrict__ out, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double *row = X + static_cast<size_t>(i) * d;
    double s = 0.0;
    for (int j = 0; j < d; ++j) s += row[j] * row[j];
    out[i] = s;
}

// CUDA kernel for RBF matrix (column-major layout for K)
__global__ void rbf_exp_kernel(double *__restrict__ K, const double *__restrict__ asq, const double *__restrict__ bsq, int n, int m, double gamma) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(n) * m;
    if (idx >= total) return;

    int i = static_cast<int>(idx % n);
    int j = static_cast<int>(idx / n);
    double val = asq[i] + bsq[j] + K[idx];
    K[idx] = exp(fmax(0.0, val) * (-gamma));
}

// CUDA kernel for inverse square root of eigenvalues
// (tol is computed on the host from the largest eigenvalue, as in the MPI+CUDA version,
//  so that no thread reads eigvals[m-1] while another one overwrites it)
__global__ void compute_inv_sqrt_kernel(double *__restrict__ eigvals, int m, double tol) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    double v = eigvals[i];
    if(v > tol) {
      eigvals[i] = 1.0 / std::sqrt(v);
    }
    else {
      eigvals[i] = 0.0;
    }
}

// CUDA kernel to scale matrix V columns (column-major layout)
__global__ void scale_columns_kernel(double *__restrict__ V, const double *__restrict__ scale, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    double s = scale[i];
    for (int j = 0; j < m; ++j) {
        V[i * m + j] *= s; // column-major indexing to get column i
    }
}

// CUDA kernel to scale matrix U rows (column-major layout)
__global__ void weighted_scale_rows_kernel(double *__restrict__ U, double *__restrict__ target_w, const double *__restrict__ sqrt_w, const double *__restrict__ target, int n, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double sw = sqrt_w[i];
    for (int j = 0; j < m; ++j) {
        U[j * n + i] *= sw;
    }
    target_w[i] = target[i] * sw;
}

// CUDA kernel for regularization of main diagonal
__global__ void add_diag_kernel(double *__restrict__ A, double val, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    A[i * m + i] += val;
}

// CUDA kernel for landmark extraction
__global__ void gather_landmarks_kernel(const double *__restrict__ X, double *__restrict__ landmarks, const int *__restrict__ indices, int m, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const double *src = X + static_cast<size_t>(indices[i]) * d;
    double *dst = landmarks + static_cast<size_t>(i) * d;
    for (int j = 0; j < d; ++j) dst[j] = src[j];
}

// CUDA kernel for binary target mapping: from {0, 1} to {-1, 1}
__global__ void map_target_kernel(double *__restrict__ target, const double *__restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    target[i] = 2.0 * y[i] - 1.0;
}

// CUDA kernel for numerically stable (through clamping) sigmoid
__global__ void sigmoid_kernel(double *__restrict__ out, const double *__restrict__ raw, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double v = fmin(500.0, fmax(-500.0, raw[i]));
    out[i] = 1.0 / (1.0 + exp(-v));
}

// Nystrom-KELM CUDA
class NystromKELM_Cuda {
public:
    double gamma_, lambda_;
    int n_landmarks_;
    bool class_weighted_;
    unsigned int seed_;

    // pointers to device memory
    double *d_X = nullptr, *d_y = nullptr;
    double *d_landmarks = nullptr, *d_Kmm = nullptr, *d_eigvals = nullptr;
    double *d_V_scaled = nullptr, *d_Kmm_inv_sqrt = nullptr, *d_Knm = nullptr;
    double *d_U = nullptr, *d_LHS = nullptr, *d_RHS = nullptr, *d_alpha = nullptr;
    double *d_asq = nullptr, *d_bsq = nullptr, *d_sqrt_w = nullptr, *d_target = nullptr, *d_target_w = nullptr;
    double *d_probs = nullptr;
    int *d_indices = nullptr;
    int *d_devInfo = nullptr; // on device allocated pointer for cuSOLVER

    // tracking capacity for safe dynamic reallocations
    size_t knm_capacity_ = 0;
    size_t probs_capacity_ = 0;
    size_t asq_capacity_ = 0;
    size_t bsq_capacity_ = 0;
    int m_cached_ = 0;

    cublasHandle_t cublas_ = nullptr;
    cusolverDnHandle_t cusolver_ = nullptr;
    cudaStream_t stream_ = nullptr;

    NystromKELM_Cuda(double gamma, double lambda, int n_landmarks, bool cw, unsigned int sd = 42)
        : gamma_(gamma), lambda_(lambda), n_landmarks_(n_landmarks), class_weighted_(cw), seed_(sd) {
        CUBLAS_CHECK(cublasCreate(&cublas_));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver_));
        CUDA_CHECK(cudaStreamCreate(&stream_));
        CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
        CUSOLVER_CHECK(cusolverDnSetStream(cusolver_, stream_));
    }

    void free_all_gpu_memory() {
        if (d_X) { cudaFree(d_X); d_X = nullptr; }
        if (d_y) { cudaFree(d_y); d_y = nullptr; }
        if (d_landmarks) { cudaFree(d_landmarks); d_landmarks = nullptr; }
        if (d_Kmm) { cudaFree(d_Kmm); d_Kmm = nullptr; }
        if (d_eigvals) { cudaFree(d_eigvals); d_eigvals = nullptr; }
        if (d_V_scaled) { cudaFree(d_V_scaled); d_V_scaled = nullptr; }
        if (d_Kmm_inv_sqrt) { cudaFree(d_Kmm_inv_sqrt); d_Kmm_inv_sqrt = nullptr; }
        if (d_Knm) { cudaFree(d_Knm); d_Knm = nullptr; knm_capacity_ = 0; }
        if (d_U) { cudaFree(d_U); d_U = nullptr; }
        if (d_LHS) { cudaFree(d_LHS); d_LHS = nullptr; }
        if (d_RHS) { cudaFree(d_RHS); d_RHS = nullptr; }
        if (d_alpha) { cudaFree(d_alpha); d_alpha = nullptr; }
        if (d_asq) { cudaFree(d_asq); d_asq = nullptr; asq_capacity_ = 0; }
        if (d_bsq) { cudaFree(d_bsq); d_bsq = nullptr; bsq_capacity_ = 0; }
        if (d_sqrt_w) { cudaFree(d_sqrt_w); d_sqrt_w = nullptr; }
        if (d_target) { cudaFree(d_target); d_target = nullptr; }
        if (d_target_w) { cudaFree(d_target_w); d_target_w = nullptr; }
        if (d_probs) { cudaFree(d_probs); d_probs = nullptr; probs_capacity_ = 0; }
        if (d_indices) { cudaFree(d_indices); d_indices = nullptr; }
        if (d_devInfo) { cudaFree(d_devInfo); d_devInfo = nullptr; }
    }

    ~NystromKELM_Cuda() {
        free_all_gpu_memory();
        if (cublas_) cublasDestroy(cublas_);
        if (cusolver_) cusolverDnDestroy(cusolver_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    void ensure_asq_capacity(size_t req_n) {
        if (req_n > asq_capacity_) {
            if (d_asq) CUDA_CHECK(cudaFree(d_asq));
            CUDA_CHECK(cudaMalloc(&d_asq, req_n * sizeof(double)));
            asq_capacity_ = req_n;
        }
    }

    void ensure_bsq_capacity(size_t req_m) {
        if (req_m > bsq_capacity_) {
            if (d_bsq) CUDA_CHECK(cudaFree(d_bsq));
            CUDA_CHECK(cudaMalloc(&d_bsq, req_m * sizeof(double)));
            bsq_capacity_ = req_m;
        }
    }

    void compute_rbf_kernel(const double *A, const double *B, double *K, int n, int m, int d) {
        const double alpha_gemm = -2.0, beta_gemm = 0.0;

        ensure_asq_capacity(static_cast<size_t>(n));
        ensure_bsq_capacity(static_cast<size_t>(m));

        // compute -2 * A^T * B (n x m column-major matrix)
        CUBLAS_CHECK(cublasDgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, n, m, d, &alpha_gemm, A, d, B, d, &beta_gemm, K, n));

        int threads = 256;
        compute_row_norms_kernel<<<(n + threads - 1) / threads, threads, 0, stream_>>>(A, d_asq, n, d);
        compute_row_norms_kernel<<<(m + threads - 1) / threads, threads, 0, stream_>>>(B, d_bsq, m, d);

        size_t total = static_cast<size_t>(n) * m;
        rbf_exp_kernel<<<(total + threads - 1) / threads, threads, 0, stream_>>>(K, d_asq, d_bsq, n, m, gamma_);
    }

    void fit(const double *h_X, const double *h_y, int n, int d) {

        // free device memory, required if we invoke the fit method multiple times
        free_all_gpu_memory();

        int m = std::min(n_landmarks_, n);
        m_cached_ = m;
        size_t sz = sizeof(double);

        CUDA_CHECK(cudaMalloc(&d_X, static_cast<size_t>(n) * d * sz));
        CUDA_CHECK(cudaMalloc(&d_y, static_cast<size_t>(n) * sz));

        ensure_asq_capacity(static_cast<size_t>(n));
        ensure_bsq_capacity(static_cast<size_t>(m));

        CUDA_CHECK(cudaMalloc(&d_sqrt_w, static_cast<size_t>(n) * sz));
        CUDA_CHECK(cudaMalloc(&d_target, static_cast<size_t>(n) * sz));
        CUDA_CHECK(cudaMalloc(&d_target_w, static_cast<size_t>(n) * sz));
        CUDA_CHECK(cudaMalloc(&d_landmarks, static_cast<size_t>(m) * d * sz));
        CUDA_CHECK(cudaMalloc(&d_Kmm, static_cast<size_t>(m) * m * sz));
        CUDA_CHECK(cudaMalloc(&d_eigvals, static_cast<size_t>(m) * sz));
        CUDA_CHECK(cudaMalloc(&d_V_scaled, static_cast<size_t>(m) * m * sz));
        CUDA_CHECK(cudaMalloc(&d_Kmm_inv_sqrt, static_cast<size_t>(m) * m * sz));
        CUDA_CHECK(cudaMalloc(&d_LHS, static_cast<size_t>(m) * m * sz));
        CUDA_CHECK(cudaMalloc(&d_RHS, static_cast<size_t>(m) * sz));
        CUDA_CHECK(cudaMalloc(&d_alpha, static_cast<size_t>(m) * sz));
        CUDA_CHECK(cudaMalloc(&d_indices, static_cast<size_t>(n) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_devInfo, sizeof(int)));

        CUDA_CHECK(cudaMemcpyAsync(d_X, h_X, static_cast<size_t>(n) * d * sz, cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_y, h_y, static_cast<size_t>(n) * sz, cudaMemcpyHostToDevice, stream_));

        // Landmark selection with user-defined seed for reproducibility
        std::vector<int> h_indices(n);
        std::iota(h_indices.begin(), h_indices.end(), 0);
        std::mt19937 rng(seed_);
        std::shuffle(h_indices.begin(), h_indices.end(), rng);
        CUDA_CHECK(cudaMemcpyAsync(d_indices, h_indices.data(), static_cast<size_t>(n) * sizeof(int), cudaMemcpyHostToDevice, stream_));

        gather_landmarks_kernel<<<(m + 255) / 256, 256, 0, stream_>>>(d_X, d_landmarks, d_indices, m, d);

        // Kmm matrix and regularization for numerical stability
        compute_rbf_kernel(d_landmarks, d_landmarks, d_Kmm, m, m, d);
        add_diag_kernel<<<(m + 255) / 256, 256, 0, stream_>>>(d_Kmm, 1e-8, m);

        // Spectral decomposition: Kmm = V * Λ * V^T
        int temp_lwork = 0;
        CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(cusolver_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, m, d_Kmm, m, d_eigvals, &temp_lwork));
        double *ws = nullptr;
        CUDA_CHECK(cudaMalloc(&ws, static_cast<size_t>(temp_lwork) * sizeof(double)));
        CUSOLVER_CHECK(cusolverDnDsyevd(cusolver_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, m, d_Kmm, m, d_eigvals, ws, temp_lwork, d_devInfo));

        // stream sync before deallocating the cuSOLVER workspace
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        CUDA_CHECK(cudaFree(ws));

        // check if the spectral decomposition failed
        int h_info = 0;
        CUDA_CHECK(cudaMemcpy(&h_info, d_devInfo, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_info != 0) {
            throw std::runtime_error("cusolverDnDsyevd failed: devInfo = " + std::to_string(h_info));
        }

        // threshold of the truncated pseudo-inverse, from the largest eigenvalue (ascending order)
        double h_max_eig = 0.0;
        CUDA_CHECK(cudaMemcpy(&h_max_eig, d_eigvals + (m - 1), sizeof(double), cudaMemcpyDeviceToHost));
        const double tol = std::max(1e-12, 1e-10 * h_max_eig);

        // in-place computing of the inverse square root of eigenvalues
        compute_inv_sqrt_kernel<<<(m + 255) / 256, 256, 0, stream_>>>(d_eigvals, m, tol);
        CUDA_CHECK(cudaGetLastError());

        // Copy V to d_V_scaled and scale the columns: d_V_scaled = V * diag(Λ^{-1/2})
        CUDA_CHECK(cudaMemcpyAsync(d_V_scaled, d_Kmm, static_cast<size_t>(m) * m * sz, cudaMemcpyDeviceToDevice, stream_));
        scale_columns_kernel<<<(m + 255) / 256, 256, 0, stream_>>>(d_V_scaled, d_eigvals, m);

        // Kmm_inv_sqrt = (V * Λ^{-1/2}) * V^T = Kmm^{-1/2}
        const double a1 = 1.0, b0 = 0.0;
        CUBLAS_CHECK(cublasDgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_T, m, m, m,
                                 &a1, d_V_scaled, m, d_Kmm, m, &b0, d_Kmm_inv_sqrt, m));

        // feature map U = Knm * Kmm^{-1/2} (n x m)
        CUDA_CHECK(cudaMalloc(&d_Knm, static_cast<size_t>(n) * m * sz));
        knm_capacity_ = static_cast<size_t>(n) * m;
        compute_rbf_kernel(d_X, d_landmarks, d_Knm, n, m, d);

        CUDA_CHECK(cudaMalloc(&d_U, static_cast<size_t>(n) * m * sz));
        CUBLAS_CHECK(cublasDgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N, n, m, m,
                                 &a1, d_Knm, n, d_Kmm_inv_sqrt, m, &b0, d_U, n));

        if (class_weighted_) {
            std::vector<double> h_y(n);
            CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, n * sizeof(double), cudaMemcpyDeviceToHost));
            double c0 = 0, c1 = 0;
            for (double v : h_y) (v == 1.0) ? ++c1 : ++c0;
            double w0 = n / (2.0 * std::max(1.0, c0)), w1 = n / (2.0 * std::max(1.0, c1));

            std::vector<double> h_sw(n), h_t(n), h_tw(n);
            for (int i = 0; i < n; ++i) {
                h_sw[i] = std::sqrt((h_y[i] == 1.0) ? w1 : w0);
                h_t[i] = 2.0 * h_y[i] - 1.0;
                h_tw[i] = h_t[i] * h_sw[i];
            }

            CUDA_CHECK(cudaMemcpyAsync(d_sqrt_w, h_sw.data(), n * sz, cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(d_target, h_t.data(), n * sz, cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(d_target_w, h_tw.data(), n * sz, cudaMemcpyHostToDevice, stream_));
            weighted_scale_rows_kernel<<<(n + 255) / 256, 256, 0, stream_>>>(d_U, d_target_w, d_sqrt_w, d_target, n, m);

            // LHS = U^T * U (m x m) through cublasDsyrk
            CUBLAS_CHECK(cublasDsyrk(cublas_, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, m, n, &a1, d_U, n, &b0, d_LHS, m));
            // RHS = U^T * target_w
            CUBLAS_CHECK(cublasDgemv(cublas_, CUBLAS_OP_T, n, m, &a1, d_U, n, d_target_w, 1, &b0, d_RHS, 1));
        } else {
            map_target_kernel<<<(n + 255) / 256, 256, 0, stream_>>>(d_target, d_y, n);
            CUBLAS_CHECK(cublasDsyrk(cublas_, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, m, n, &a1, d_U, n, &b0, d_LHS, m));
            CUBLAS_CHECK(cublasDgemv(cublas_, CUBLAS_OP_T, n, m, &a1, d_U, n, d_target, 1, &b0, d_RHS, 1));
        }

        // regularization for numerical stability (main diagonal)
        add_diag_kernel<<<(m + 255) / 256, 256, 0, stream_>>>(d_LHS, lambda_, m);

        int lwork = 0;
        CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(cusolver_, CUBLAS_FILL_MODE_UPPER, m, d_LHS, m, &lwork));
        double *work_potrf = nullptr;
        CUDA_CHECK(cudaMalloc(&work_potrf, static_cast<size_t>(lwork) * sizeof(double)));
        CUSOLVER_CHECK(cusolverDnDpotrf(cusolver_, CUBLAS_FILL_MODE_UPPER, m, d_LHS, m, work_potrf, lwork, d_devInfo));

        CUDA_CHECK(cudaStreamSynchronize(stream_));
        CUDA_CHECK(cudaFree(work_potrf));

        // check that LHS matrix is SPD
        CUDA_CHECK(cudaMemcpy(&h_info, d_devInfo, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_info != 0) {
            throw std::runtime_error("Cholesky failed (LHS is not positive defined), info = " + std::to_string(h_info));
        }

        // solve triangular system
        CUSOLVER_CHECK(cusolverDnDpotrs(cusolver_, CUBLAS_FILL_MODE_UPPER, m, 1, d_LHS, m, d_RHS, m, d_devInfo));

        CUDA_CHECK(cudaStreamSynchronize(stream_));
        CUDA_CHECK(cudaMemcpy(&h_info, d_devInfo, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_info != 0) {
            throw std::runtime_error("cusolverDnDpotrs failed: info = " + std::to_string(h_info));
        }

        CUDA_CHECK(cudaMemcpyAsync(d_alpha, d_RHS, m * sizeof(double), cudaMemcpyDeviceToDevice, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    void predict_proba(const double *h_X_new, int n_new, int d, std::vector<double> &probs_out) {
        if (m_cached_ == 0 || !d_landmarks || !d_Kmm_inv_sqrt || !d_alpha) {
            throw std::runtime_error("The model must be trained before calling this method.");
        }

        size_t sz = sizeof(double);

        // safe dynamic reallocation of capacity
        if (static_cast<size_t>(n_new) * m_cached_ > knm_capacity_) {
            if (d_Knm) CUDA_CHECK(cudaFree(d_Knm));
            CUDA_CHECK(cudaMalloc(&d_Knm, static_cast<size_t>(n_new) * m_cached_ * sz));
            knm_capacity_ = static_cast<size_t>(n_new) * m_cached_;
        }

        if (static_cast<size_t>(n_new) > probs_capacity_) {
            if (d_probs) CUDA_CHECK(cudaFree(d_probs));
            CUDA_CHECK(cudaMalloc(&d_probs, static_cast<size_t>(n_new) * sz));
            probs_capacity_ = static_cast<size_t>(n_new);
        }

        double *d_Xn = nullptr;
        CUDA_CHECK(cudaMalloc(&d_Xn, static_cast<size_t>(n_new) * d * sz));
        CUDA_CHECK(cudaMemcpyAsync(d_Xn, h_X_new, static_cast<size_t>(n_new) * d * sz, cudaMemcpyHostToDevice, stream_));

        // compute_rbf_kernel will reallocate d_asq if n_new > n_train
        compute_rbf_kernel(d_Xn, d_landmarks, d_Knm, n_new, m_cached_, d);

        double *d_raw = nullptr;
        CUDA_CHECK(cudaMalloc(&d_raw, static_cast<size_t>(n_new) * sz));
        const double a1 = 1.0, b0 = 0.0;

        // prediction: raw = Knm_new * Kmm^{-1/2} * alpha
        // step 1: temp = Kmm_inv_sqrt * alpha (m-dimensional vector stored in d_RHS)
        CUBLAS_CHECK(cublasDgemv(cublas_, CUBLAS_OP_N, m_cached_, m_cached_,
                                 &a1, d_Kmm_inv_sqrt, m_cached_, d_alpha, 1, &b0, d_RHS, 1));
        // step 2: raw = Knm_new * temp (n_new-dimensional vector)
        CUBLAS_CHECK(cublasDgemv(cublas_, CUBLAS_OP_N, n_new, m_cached_,
                                 &a1, d_Knm, n_new, d_RHS, 1, &b0, d_raw, 1));

        sigmoid_kernel<<<(n_new + 255) / 256, 256, 0, stream_>>>(d_probs, d_raw, n_new);

        CUDA_CHECK(cudaFree(d_raw));
        CUDA_CHECK(cudaFree(d_Xn));

        CUDA_CHECK(cudaStreamSynchronize(stream_));
        probs_out.resize(n_new);
        CUDA_CHECK(cudaMemcpy(probs_out.data(), d_probs, static_cast<size_t>(n_new) * sz, cudaMemcpyDeviceToHost));
    }
};

// Helpers for CSV parsing
static std::string trim_str(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n\"'");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n\"'");
    return str.substr(first, (last - first + 1));
}

static std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> row;
    std::string field;
    bool in_quotes = false;
    for (size_t i = 0; i < line.length(); ++i) {
        char c = line[i];
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            row.push_back(trim_str(field));
            field.clear();
        } else {
            field.push_back(c);
        }
    }
    row.push_back(trim_str(field));
    return row;
}

static bool is_numeric_field(const std::string& s) {
    std::string t = trim_str(s);
    if (t.empty()) return false;
    char* endptr = nullptr;
    std::strtod(t.c_str(), &endptr);
    return endptr != t.c_str() && *endptr == '\0';
}

void read_csv(const std::string &filename, std::vector<double> &X_host, std::vector<double> &y_host, int &n_rows, int &n_features) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) lines.push_back(line);
    }

    if (lines.size() < 2) throw std::runtime_error("CSV file has no data rows");

    std::vector<std::string> headers = parse_csv_line(lines[0]);
    if (headers.size() < 2) throw std::runtime_error("CSV file must have at least 2 columns (features and target)");

    // Locate the 'target' column (case-insensitive search)
    int target_col = -1;
    for (size_t i = 0; i < headers.size(); ++i) {
        std::string h = headers[i];
        std::transform(h.begin(), h.end(), h.begin(), [](unsigned char c) { return std::tolower(c); });
        if (h == "target") {
            target_col = static_cast<int>(i);
            break;
        }
    }

    if (target_col == -1) {
        throw std::runtime_error("CSV file must contain a column named 'target'");
    }

    n_rows = static_cast<int>(lines.size()) - 1;
    const int n_cols = static_cast<int>(headers.size());

    std::vector<std::vector<std::string>> raw_rows;
    raw_rows.reserve(n_rows);
    for (int i = 0; i < n_rows; ++i) {
        std::vector<std::string> row = parse_csv_line(lines[i + 1]);
        if (static_cast<int>(row.size()) != n_cols) {
            throw std::runtime_error("Row " + std::to_string(i + 2) + " column count mismatch");
        }
        raw_rows.push_back(row);
    }

    // Process target column
    std::vector<std::string> raw_y(n_rows);
    std::set<std::string> unique_targets_set;
    for (int i = 0; i < n_rows; ++i) {
        raw_y[i] = raw_rows[i][target_col];
        unique_targets_set.insert(raw_y[i]);
    }

    y_host.assign(n_rows, 0.0);
    bool target_is_numeric = true;
    for (const auto& val : unique_targets_set) {
        if (!is_numeric_field(val)) {
            target_is_numeric = false;
            break;
        }
    }

    if (target_is_numeric) {
        bool already_binary_01 = true;
        for (const auto& val : unique_targets_set) {
            double d = std::stod(val);
            if (d != 0.0 && d != 1.0) {
                already_binary_01 = false;
                break;
            }
        }

        if (already_binary_01) {
            for (int i = 0; i < n_rows; ++i) {
                y_host[i] = std::stod(raw_y[i]);
            }
        } else {
            std::vector<double> num_vals;
            for (const auto& val : unique_targets_set) num_vals.push_back(std::stod(val));
            std::sort(num_vals.begin(), num_vals.end());

            std::map<double, double> target_map;
            for (size_t i = 0; i < num_vals.size(); ++i) {
                target_map[num_vals[i]] = static_cast<double>(i);
            }
            for (int i = 0; i < n_rows; ++i) {
                y_host[i] = target_map[std::stod(raw_y[i])];
            }
        }
    } else {
        std::vector<std::string> unique_targets(unique_targets_set.begin(), unique_targets_set.end());
        std::map<std::string, double> target_map;
        for (size_t i = 0; i < unique_targets.size(); ++i) {
            target_map[unique_targets[i]] = static_cast<double>(i);
        }
        for (int i = 0; i < n_rows; ++i) {
            y_host[i] = target_map[raw_y[i]];
        }
    }

    // Process feature columns (all columns except 'target')
    std::vector<int> feature_cols;
    for (int j = 0; j < n_cols; ++j) {
        if (j != target_col) {
            feature_cols.push_back(j);
        }
    }

    struct FeatureColInfo {
        bool is_numeric;
        std::vector<std::string> unique_categories;
        std::map<std::string, int> cat_to_idx;
    };

    std::vector<FeatureColInfo> col_infos(feature_cols.size());
    int total_features = 0;

    for (size_t k = 0; k < feature_cols.size(); ++k) {
        int col_idx = feature_cols[k];
        bool numeric = true;
        std::set<std::string> cat_set;

        for (int i = 0; i < n_rows; ++i) {
            const std::string& val = raw_rows[i][col_idx];
            cat_set.insert(val);
            if (!is_numeric_field(val)) {
                numeric = false;
            }
        }

        col_infos[k].is_numeric = numeric;
        if (numeric) {
            total_features += 1;
        } else {
            col_infos[k].unique_categories.assign(cat_set.begin(), cat_set.end());
            for (size_t c = 0; c < col_infos[k].unique_categories.size(); ++c) {
                col_infos[k].cat_to_idx[col_infos[k].unique_categories[c]] = static_cast<int>(c);
            }
            total_features += static_cast<int>(col_infos[k].unique_categories.size());
        }
    }

    n_features = total_features;
    X_host.assign(static_cast<size_t>(n_rows) * n_features, 0.0);

    for (int i = 0; i < n_rows; ++i) {
        int current_out_col = 0;
        for (size_t k = 0; k < feature_cols.size(); ++k) {
            int col_idx = feature_cols[k];
            const std::string& val = raw_rows[i][col_idx];

            if (col_infos[k].is_numeric) {
                X_host[static_cast<size_t>(i) * n_features + current_out_col++] = std::stod(val);
            } else {
                int cat_idx = col_infos[k].cat_to_idx[val];
                int num_cats = static_cast<int>(col_infos[k].unique_categories.size());
                X_host[static_cast<size_t>(i) * n_features + current_out_col + cat_idx] = 1.0;
                current_out_col += num_cats;
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        std::cout << "Usage: " << argv[0] << " <n_landmarks> [seed]\n";
        return 1;
    }

    int n_landmarks = std::stoi(argv[1]);
    unsigned long seed = 42; // Default seed for reproducibility
    if (argc == 3) {
        seed = std::stoul(argv[2]);
    }


    std::string filename = "./data/diabetes.csv";

    std::vector<double> X_host, y_host;
    int n = 0, d = 0;
    try {
        read_csv(filename, X_host, y_host, n, d);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }


#if 0
    std::cerr << "Matrix dimensions: " << n << " " << d << "\n";
    std::cerr << "Landmarks: " << n_landmarks << "\n";
    std::cerr << "Seed: " << seed << "\n";
#endif

    NystromKELM_Cuda model(0.1, 1.0, n_landmarks, true, seed);

    auto t0 = std::chrono::high_resolution_clock::now();
    model.fit(X_host.data(), y_host.data(), n, d);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Training Time: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    std::vector<double> probs;
    t0 = std::chrono::high_resolution_clock::now();
    model.predict_proba(X_host.data(), n, d, probs);
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Prediction Time: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    std::cout << "First 10 probabilities:\n";
    for (int i = 0; i < std::min(10, (int)probs.size()); ++i)
        std::cout << " [" << i << "] " << std::fixed << std::setprecision(7) << probs[i] << "\n";

    return 0;
}
