// MIT License

// Copyright (c) 2026 Massimo Cafaro, Ph.D., University of Salento, Lecce, Italy, ORCID: 0000-0003-1118-7109

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

// October 18, 2026
// Nystrom-KELM v0.5
//
// This code implements, in C++ using MPI, the CUDA Toolkit and related libraries,
// a Nystrom-approximated Kernel Estreme Learning Machine
// by using a standard kernel ridge regression; e.g. Williams & Seeger 2001
//
// The code actually reimplements a corresponding Python version originally developed by
//
// Jason Digalakis
// PhD, BSc Applied Informatics
// Postdoctoral Researcher
// Biomedical Engineering Laboratory
// School of Electrical and Computer Engineering
// National Technical University of Athens
//
// using numpy. The Python version works as follows:
//
//  1. Select m landmark points from the training set.
//  2. Compute K_mm (m x m kernel matrix among landmarks) and
//     K_nm (n x m kernel matrix between all training points and landmarks).
//  3. Form U = K_nm @ K_mm^{-1/2} (an n x m explicit low-rank feature map
//     approximating the implicit kernel feature space).
//  4. Solve ordinary (m-dimensional) ridge regression in U-space:
//     alpha = (U^T U + lambda I_m)^{-1} U^T t -- an m x m solve, not n x n.
//  5. Predict via f(x) = k(x, landmarks) @ K_mm^{-1/2} @ alpha.
//
//
//  This code has been developed by
//
//  Massimo cafaro, Ph.D.
//  Associate Professor
//  Department of Engineering for Innovation
//  University of Salento, Lecce, Italy
//





#include "nystrom.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

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
    if (s__ != CUBLAS_STATUS_SUCCESS) {                                                                                                                                                                \
      std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " -> code " << s__ << "\n";                                                                                                    \
      std::exit(1);                                                                                                                                                                                    \
    }                                                                                                                                                                                                  \
  } while (0)

#define CUSOLVER_CHECK(call)                                                                                                                                                                           \
  do {                                                                                                                                                                                                 \
    cusolverStatus_t s__ = (call);                                                                                                                                                                     \
    if (s__ != CUSOLVER_STATUS_SUCCESS) {                                                                                                                                                              \
      std::cerr << "cuSOLVER error at " << __FILE__ << ":" << __LINE__ << " -> code " << s__ << "\n";                                                                                               \
      std::exit(1);                                                                                                                                                                                    \
    }                                                                                                                                                                                                  \
  } while (0)

// CUDA kernel to compute the L2 squared norm of rows
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
__global__ void compute_inv_sqrt_truncated_kernel(double *__restrict__ eigvals, int m, double tol) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    double v = eigvals[i];
    if (v > tol) {
        eigvals[i] = 1.0 / sqrt(v);
    } else {
        eigvals[i] = 0.0;
    }
}

// CUDA kernel to scale matrix V columns
__global__ void scale_columns_kernel(double *__restrict__ V, const double *__restrict__ scale, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    double s = scale[i];
    for (int j = 0; j < m; ++j) {
        V[i * m + j] *= s;
    }
}

// CUDA kernel to scale matrix U rows
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

//  CUDA kernel for binary target mapping: from {0, 1} to {-1, 1}
__global__ void map_target_kernel(double *__restrict__ target, const double *__restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    target[i] = 2.0 * y[i] - 1.0;
}

//  CUDA kernel for numerically stable (through clamping) sigmoid
__global__ void sigmoid_kernel(double *__restrict__ out, const double *__restrict__ raw, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double v = fmin(500.0, fmax(-500.0, raw[i]));
    out[i] = 1.0 / (1.0 + exp(-v));
}

void set_cuda_device(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
}

int get_cuda_device_count() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

// NystromKELM_Cuda
struct NystromKELM_Cuda::Impl {
    double gamma_, lambda_;
    int n_landmarks_;
    bool class_weighted_;
    int m_cached_ = 0;

    cublasHandle_t cublas_ = nullptr;
    cusolverDnHandle_t cusolver_ = nullptr;
    cudaStream_t stream_ = nullptr;

    double *d_X_loc = nullptr, *d_y_loc = nullptr;
    double *d_landmarks = nullptr, *d_Kmm = nullptr, *d_eigvals = nullptr;
    double *d_V_scaled = nullptr, *d_Kmm_inv_sqrt = nullptr, *d_Knm_loc = nullptr;
    double *d_U_loc = nullptr, *d_LHS = nullptr, *d_RHS = nullptr;
    double *d_alpha = nullptr, *d_beta = nullptr;
    double *d_asq = nullptr, *d_bsq = nullptr;
    double *d_sqrt_w = nullptr, *d_target = nullptr, *d_target_w = nullptr;
    int *d_devInfo = nullptr;

    size_t asq_capacity_ = 0, bsq_capacity_ = 0;

    Impl(double gamma, double lambda, int n_landmarks, bool cw)
        : gamma_(gamma), lambda_(lambda), n_landmarks_(n_landmarks), class_weighted_(cw) {
        CUBLAS_CHECK(cublasCreate(&cublas_));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver_));
        CUDA_CHECK(cudaStreamCreate(&stream_));
        CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
        CUSOLVER_CHECK(cusolverDnSetStream(cusolver_, stream_));
    }

    ~Impl() {
        free_memory();
        if (cublas_) cublasDestroy(cublas_);
        if (cusolver_) cusolverDnDestroy(cusolver_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    void free_memory() {
        if (d_X_loc) cudaFree(d_X_loc);
        if (d_y_loc) cudaFree(d_y_loc);
        if (d_landmarks) cudaFree(d_landmarks);
        if (d_Kmm) cudaFree(d_Kmm);
        if (d_eigvals) cudaFree(d_eigvals);
        if (d_V_scaled) cudaFree(d_V_scaled);
        if (d_Kmm_inv_sqrt) cudaFree(d_Kmm_inv_sqrt);
        if (d_Knm_loc) cudaFree(d_Knm_loc);
        if (d_U_loc) cudaFree(d_U_loc);
        if (d_LHS) cudaFree(d_LHS);
        if (d_RHS) cudaFree(d_RHS);
        if (d_alpha) cudaFree(d_alpha);
        if (d_beta) cudaFree(d_beta);
        if (d_asq) cudaFree(d_asq);
        if (d_bsq) cudaFree(d_bsq);
        if (d_sqrt_w) cudaFree(d_sqrt_w);
        if (d_target) cudaFree(d_target);
        if (d_target_w) cudaFree(d_target_w);
        if (d_devInfo) cudaFree(d_devInfo);
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
        ensure_asq_capacity(n);
        ensure_bsq_capacity(m);

		    // compute -2 * A^T * B (n x m  column-major matrix)
        CUBLAS_CHECK(cublasDgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, n, m, d, &alpha_gemm, A, d, B, d, &beta_gemm, K, n));

        int threads = 256;
        compute_row_norms_kernel<<<(n + threads - 1) / threads, threads, 0, stream_>>>(A, d_asq, n, d);
        compute_row_norms_kernel<<<(m + threads - 1) / threads, threads, 0, stream_>>>(B, d_bsq, m, d);

        size_t total = static_cast<size_t>(n) * m;
        rbf_exp_kernel<<<(total + threads - 1) / threads, threads, 0, stream_>>>(K, d_asq, d_bsq, n, m, gamma_);
    }
};

NystromKELM_Cuda::NystromKELM_Cuda(double gamma, double lambda, int n_landmarks, bool cw)
    : pimpl_(new Impl(gamma, lambda, n_landmarks, cw)) {}

NystromKELM_Cuda::~NystromKELM_Cuda() { delete pimpl_; }

void NystromKELM_Cuda::compute_local_matrices(
    const double *h_X_loc, const double *h_y_loc,
    int n_loc, int n_global, int d,
    const std::vector<double> &h_landmarks,
    double c0_glob, double c1_glob,
    std::vector<double> &h_LHS_loc, std::vector<double> &h_RHS_loc)
{
    int m = std::min(pimpl_->n_landmarks_, n_global);
    pimpl_->m_cached_ = m;
    size_t sz = sizeof(double);

    pimpl_->free_memory();

    CUDA_CHECK(cudaMalloc(&pimpl_->d_X_loc, static_cast<size_t>(n_loc) * d * sz));
    CUDA_CHECK(cudaMalloc(&pimpl_->d_y_loc, static_cast<size_t>(n_loc) * sz));
    CUDA_CHECK(cudaMalloc(&pimpl_->d_landmarks, static_cast<size_t>(m) * d * sz));
    CUDA_CHECK(cudaMalloc(&pimpl_->d_Kmm, static_cast<size_t>(m) * m * sz));
    CUDA_CHECK(cudaMalloc(&pimpl_->d_eigvals, static_cast<size_t>(m) * sz));
    CUDA_CHECK(cudaMalloc(&pimpl_->d_V_scaled, static_cast<size_t>(m) * m * sz));
    CUDA_CHECK(cudaMalloc(&pimpl_->d_Kmm_inv_sqrt, static_cast<size_t>(m) * m * sz));
    CUDA_CHECK(cudaMalloc(&pimpl_->d_Knm_loc, static_cast<size_t>(n_loc) * m * sz));
    CUDA_CHECK(cudaMalloc(&pimpl_->d_U_loc, static_cast<size_t>(n_loc) * m * sz));
    CUDA_CHECK(cudaMalloc(&pimpl_->d_LHS, static_cast<size_t>(m) * m * sz));
    CUDA_CHECK(cudaMalloc(&pimpl_->d_RHS, static_cast<size_t>(m) * sz));
    CUDA_CHECK(cudaMalloc(&pimpl_->d_alpha, static_cast<size_t>(m) * sz));
    CUDA_CHECK(cudaMalloc(&pimpl_->d_beta, static_cast<size_t>(m) * sz));
    CUDA_CHECK(cudaMalloc(&pimpl_->d_sqrt_w, static_cast<size_t>(n_loc) * sz));
    CUDA_CHECK(cudaMalloc(&pimpl_->d_target, static_cast<size_t>(n_loc) * sz));
    CUDA_CHECK(cudaMalloc(&pimpl_->d_target_w, static_cast<size_t>(n_loc) * sz));
    CUDA_CHECK(cudaMalloc(&pimpl_->d_devInfo, sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(pimpl_->d_X_loc, h_X_loc, static_cast<size_t>(n_loc) * d * sz, cudaMemcpyHostToDevice, pimpl_->stream_));
    CUDA_CHECK(cudaMemcpyAsync(pimpl_->d_y_loc, h_y_loc, static_cast<size_t>(n_loc) * sz, cudaMemcpyHostToDevice, pimpl_->stream_));
    CUDA_CHECK(cudaMemcpyAsync(pimpl_->d_landmarks, h_landmarks.data(), static_cast<size_t>(m) * d * sz, cudaMemcpyHostToDevice, pimpl_->stream_));

    pimpl_->compute_rbf_kernel(pimpl_->d_landmarks, pimpl_->d_landmarks, pimpl_->d_Kmm, m, m, d);
    add_diag_kernel<<<(m + 255) / 256, 256, 0, pimpl_->stream_>>>(pimpl_->d_Kmm, 1e-8, m);

    int temp_lwork = 0;
    CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(pimpl_->cusolver_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, m, pimpl_->d_Kmm, m, pimpl_->d_eigvals, &temp_lwork));
    double *ws = nullptr;
    CUDA_CHECK(cudaMalloc(&ws, static_cast<size_t>(temp_lwork) * sz));
    CUSOLVER_CHECK(cusolverDnDsyevd(pimpl_->cusolver_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, m, pimpl_->d_Kmm, m, pimpl_->d_eigvals, ws, temp_lwork, pimpl_->d_devInfo));
    CUDA_CHECK(cudaStreamSynchronize(pimpl_->stream_));
    CUDA_CHECK(cudaFree(ws));

    double h_max_eig = 0.0;
    CUDA_CHECK(cudaMemcpy(&h_max_eig, pimpl_->d_eigvals + (m - 1), sz, cudaMemcpyDeviceToHost));
    double tol = std::max(1e-12, 1e-10 * h_max_eig);
    compute_inv_sqrt_truncated_kernel<<<(m + 255) / 256, 256, 0, pimpl_->stream_>>>(pimpl_->d_eigvals, m, tol);

    CUDA_CHECK(cudaMemcpyAsync(pimpl_->d_V_scaled, pimpl_->d_Kmm, static_cast<size_t>(m) * m * sz, cudaMemcpyDeviceToDevice, pimpl_->stream_));
    scale_columns_kernel<<<(m + 255) / 256, 256, 0, pimpl_->stream_>>>(pimpl_->d_V_scaled, pimpl_->d_eigvals, m);

    const double a1 = 1.0, b0 = 0.0;
    CUBLAS_CHECK(cublasDgemm(pimpl_->cublas_, CUBLAS_OP_N, CUBLAS_OP_T, m, m, m, &a1, pimpl_->d_V_scaled, m, pimpl_->d_Kmm, m, &b0, pimpl_->d_Kmm_inv_sqrt, m));

    pimpl_->compute_rbf_kernel(pimpl_->d_X_loc, pimpl_->d_landmarks, pimpl_->d_Knm_loc, n_loc, m, d);
    CUBLAS_CHECK(cublasDgemm(pimpl_->cublas_, CUBLAS_OP_N, CUBLAS_OP_N, n_loc, m, m, &a1, pimpl_->d_Knm_loc, n_loc, pimpl_->d_Kmm_inv_sqrt, m, &b0, pimpl_->d_U_loc, n_loc));

    if (pimpl_->class_weighted_) {
        double w0 = n_global / (2.0 * std::max(1.0, c0_glob));
        double w1 = n_global / (2.0 * std::max(1.0, c1_glob));

        std::vector<double> h_sw(n_loc), h_t(n_loc), h_tw(n_loc);
        for (int i = 0; i < n_loc; ++i) {
            h_sw[i] = std::sqrt((h_y_loc[i] == 1.0) ? w1 : w0);
            h_t[i] = 2.0 * h_y_loc[i] - 1.0;
            h_tw[i] = h_t[i] * h_sw[i];
        }

        CUDA_CHECK(cudaMemcpyAsync(pimpl_->d_sqrt_w, h_sw.data(), n_loc * sz, cudaMemcpyHostToDevice, pimpl_->stream_));
        CUDA_CHECK(cudaMemcpyAsync(pimpl_->d_target, h_t.data(), n_loc * sz, cudaMemcpyHostToDevice, pimpl_->stream_));
        CUDA_CHECK(cudaMemcpyAsync(pimpl_->d_target_w, h_tw.data(), n_loc * sz, cudaMemcpyHostToDevice, pimpl_->stream_));

        weighted_scale_rows_kernel<<<(n_loc + 255) / 256, 256, 0, pimpl_->stream_>>>(pimpl_->d_U_loc, pimpl_->d_target_w, pimpl_->d_sqrt_w, pimpl_->d_target, n_loc, m);
        CUBLAS_CHECK(cublasDsyrk(pimpl_->cublas_, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, m, n_loc, &a1, pimpl_->d_U_loc, n_loc, &b0, pimpl_->d_LHS, m));
        CUBLAS_CHECK(cublasDgemv(pimpl_->cublas_, CUBLAS_OP_T, n_loc, m, &a1, pimpl_->d_U_loc, n_loc, pimpl_->d_target_w, 1, &b0, pimpl_->d_RHS, 1));
    } else {
        map_target_kernel<<<(n_loc + 255) / 256, 256, 0, pimpl_->stream_>>>(pimpl_->d_target, pimpl_->d_y_loc, n_loc);
        CUBLAS_CHECK(cublasDsyrk(pimpl_->cublas_, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, m, n_loc, &a1, pimpl_->d_U_loc, n_loc, &b0, pimpl_->d_LHS, m));
        CUBLAS_CHECK(cublasDgemv(pimpl_->cublas_, CUBLAS_OP_T, n_loc, m, &a1, pimpl_->d_U_loc, n_loc, pimpl_->d_target, 1, &b0, pimpl_->d_RHS, 1));
    }

    CUDA_CHECK(cudaStreamSynchronize(pimpl_->stream_));

    h_LHS_loc.resize(m * m);
    h_RHS_loc.resize(m);

    CUDA_CHECK(cudaMemcpy(h_LHS_loc.data(), pimpl_->d_LHS, static_cast<size_t>(m) * m * sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_RHS_loc.data(), pimpl_->d_RHS, static_cast<size_t>(m) * sz, cudaMemcpyDeviceToHost));
}

void NystromKELM_Cuda::solve_global_system(
    const std::vector<double> &h_LHS_glob,
    const std::vector<double> &h_RHS_glob)
{
    int m = pimpl_->m_cached_;
    size_t sz = sizeof(double);

    CUDA_CHECK(cudaMemcpyAsync(pimpl_->d_LHS, h_LHS_glob.data(), static_cast<size_t>(m) * m * sz, cudaMemcpyHostToDevice, pimpl_->stream_));
    CUDA_CHECK(cudaMemcpyAsync(pimpl_->d_RHS, h_RHS_glob.data(), static_cast<size_t>(m) * sz, cudaMemcpyHostToDevice, pimpl_->stream_));

    add_diag_kernel<<<(m + 255) / 256, 256, 0, pimpl_->stream_>>>(pimpl_->d_LHS, pimpl_->lambda_, m);

    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(pimpl_->cusolver_, CUBLAS_FILL_MODE_UPPER, m, pimpl_->d_LHS, m, &lwork));
    double *work_potrf = nullptr;
    CUDA_CHECK(cudaMalloc(&work_potrf, static_cast<size_t>(lwork) * sz));
    CUSOLVER_CHECK(cusolverDnDpotrf(pimpl_->cusolver_, CUBLAS_FILL_MODE_UPPER, m, pimpl_->d_LHS, m, work_potrf, lwork, pimpl_->d_devInfo));
    CUDA_CHECK(cudaStreamSynchronize(pimpl_->stream_));
    CUDA_CHECK(cudaFree(work_potrf));

    int h_info = 0;
    CUDA_CHECK(cudaMemcpy(&h_info, pimpl_->d_devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        throw std::runtime_error("Cholesky failed on LHS (info = " + std::to_string(h_info) + ")");
    }

    CUSOLVER_CHECK(cusolverDnDpotrs(pimpl_->cusolver_, CUBLAS_FILL_MODE_UPPER, m, 1, pimpl_->d_LHS, m, pimpl_->d_RHS, m, pimpl_->d_devInfo));
    CUDA_CHECK(cudaMemcpyAsync(pimpl_->d_alpha, pimpl_->d_RHS, m * sz, cudaMemcpyDeviceToDevice, pimpl_->stream_));

    const double a1 = 1.0, b0 = 0.0;
    CUBLAS_CHECK(cublasDgemv(pimpl_->cublas_, CUBLAS_OP_N, m, m, &a1, pimpl_->d_Kmm_inv_sqrt, m, pimpl_->d_alpha, 1, &b0, pimpl_->d_beta, 1));
    CUDA_CHECK(cudaStreamSynchronize(pimpl_->stream_));
}

void NystromKELM_Cuda::predict_proba_local(
    const double *h_X_loc, int n_loc, int d,
    std::vector<double> &h_probs_loc)
{
    size_t sz = sizeof(double);
    double *d_Xn_loc = nullptr, *d_Knm_new = nullptr, *d_raw = nullptr, *d_probs_loc = nullptr;

    CUDA_CHECK(cudaMalloc(&d_Xn_loc, static_cast<size_t>(n_loc) * d * sz));
    CUDA_CHECK(cudaMalloc(&d_Knm_new, static_cast<size_t>(n_loc) * pimpl_->m_cached_ * sz));
    CUDA_CHECK(cudaMalloc(&d_raw, static_cast<size_t>(n_loc) * sz));
    CUDA_CHECK(cudaMalloc(&d_probs_loc, static_cast<size_t>(n_loc) * sz));

    CUDA_CHECK(cudaMemcpyAsync(d_Xn_loc, h_X_loc, static_cast<size_t>(n_loc) * d * sz, cudaMemcpyHostToDevice, pimpl_->stream_));

    pimpl_->compute_rbf_kernel(d_Xn_loc, pimpl_->d_landmarks, d_Knm_new, n_loc, pimpl_->m_cached_, d);

    const double a1 = 1.0, b0 = 0.0;
    CUBLAS_CHECK(cublasDgemv(pimpl_->cublas_, CUBLAS_OP_N, n_loc, pimpl_->m_cached_, &a1, d_Knm_new, n_loc, pimpl_->d_beta, 1, &b0, d_raw, 1));

    sigmoid_kernel<<<(n_loc + 255) / 256, 256, 0, pimpl_->stream_>>>(d_probs_loc, d_raw, n_loc);

    h_probs_loc.resize(n_loc);
    CUDA_CHECK(cudaMemcpy(h_probs_loc.data(), d_probs_loc, static_cast<size_t>(n_loc) * sz, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_Xn_loc));
    CUDA_CHECK(cudaFree(d_Knm_new));
    CUDA_CHECK(cudaFree(d_raw));
    CUDA_CHECK(cudaFree(d_probs_loc));
}
