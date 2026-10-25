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




#ifndef NYSTROM_CUDA_H
#define NYSTROM_CUDA_H

#include <vector>

void set_cuda_device(int device_id);
int get_cuda_device_count();

class NystromKELM_Cuda {
public:
    NystromKELM_Cuda(double gamma, double lambda, int n_landmarks, bool class_weighted);
    ~NystromKELM_Cuda();

    // Compute locally on GPU the LHS (m x m) and RHS (m x 1) matrices
    void compute_local_matrices(
        const double *h_X_loc, const double *h_y_loc,
        int n_loc, int n_global, int d,
        const std::vector<double> &h_landmarks,
        double c0_glob, double c1_glob,
        std::vector<double> &h_LHS_loc, std::vector<double> &h_RHS_loc
    );

    // Solve on GPU the global system after MPI_Allreduce
    void solve_global_system(
        const std::vector<double> &h_LHS_glob,
        const std::vector<double> &h_RHS_glob
    );

    // Probabilistic forecast on local data
    void predict_proba_local(
        const double *h_X_loc, int n_loc, int d,
        std::vector<double> &h_probs_loc
    );

private:
    struct Impl;
    Impl *pimpl_;
};

#endif // NYSTROM_CUDA_H
