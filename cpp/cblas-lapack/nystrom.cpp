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
// Nystrom-KELM v0.7
//
// This code implements, in C++ using the CBLAS and LAPACK libraries,
// a Nystrom-approximated Kernel Estreme Learning Machine
// by using a standard kernel ridge regression; e.g. Williams & Seeger 2001
//
//
// Nystrom-KELM v0.6
//
// This code implements, in C++ using the CBLAS and LAPACK libraries,
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

#include <algorithm>
#include <cblas.h>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <lapacke.h>
#include <map>
#include <memory>
#include <numeric>
#include <omp.h>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Matrix datatype (Row-Major layout)
struct Matrix {
    int rows, cols;
    std::vector<double> data;

    Matrix(int r, int c) : rows(r), cols(c), data(static_cast<size_t>(r) * c, 0.0) {}

    double& at(int r, int c) { return data[static_cast<size_t>(r) * cols + c]; }
    const double& at(int r, int c) const { return data[static_cast<size_t>(r) * cols + c]; }
    double* ptr() { return data.data(); }
    const double* ptr() const { return data.data(); }
};

// Helper: Trim surrounding whitespace and quote characters
static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n\"'");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n\"'");
    return str.substr(first, (last - first + 1));
}

// Helper: Parse a single CSV row while preserving text inside quotes
static std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> row;
    std::string field;
    bool in_quotes = false;
    for (size_t i = 0; i < line.length(); ++i) {
        char c = line[i];
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            row.push_back(trim(field));
            field.clear();
        } else {
            field.push_back(c);
        }
    }
    row.push_back(trim(field));
    return row;
}

// Helper: Test whether a string field represents a valid floating-point number
static bool is_numeric_field(const std::string& s) {
    std::string t = trim(s);
    if (t.empty()) return false;
    char* endptr = nullptr;
    std::strtod(t.c_str(), &endptr);
    return endptr != t.c_str() && *endptr == '\0';
}

// Dynamic CSV file reader supporting arbitrary 'target' column and mixed categorical/numerical features
Matrix read_csv_into_matrix(const std::string& filename, std::vector<double>& y) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    std::string line;
    if (!std::getline(file, line)) throw std::runtime_error("Empty CSV file");

    std::vector<std::string> headers = parse_csv_line(line);
    if (headers.size() < 2) throw std::runtime_error("Need at least 2 columns");

    // Locate the 'target' column index (case-insensitive search)
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

    // Read all data rows into string tokens
    std::vector<std::vector<std::string>> raw_rows;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        std::vector<std::string> row = parse_csv_line(line);
        if (row.size() != headers.size()) {
            throw std::runtime_error("Column mismatch in CSV row");
        }
        raw_rows.push_back(row);
    }

    if (raw_rows.empty()) {
        throw std::runtime_error("No data rows found in CSV file");
    }

    int n_rows = static_cast<int>(raw_rows.size());
    int n_cols = static_cast<int>(headers.size());

    // Process target column (y)
    std::vector<std::string> raw_y(n_rows);
    std::set<std::string> unique_targets_set;
    for (int i = 0; i < n_rows; ++i) {
        raw_y[i] = raw_rows[i][target_col];
        unique_targets_set.insert(raw_y[i]);
    }

    y.assign(n_rows, 0.0);
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
                y[i] = std::stod(raw_y[i]);
            }
        } else {
            // Map distinct numerical target labels (e.g., -1/1 or 1/2) to 0.0 and 1.0
            std::vector<double> num_vals;
            for (const auto& val : unique_targets_set) num_vals.push_back(std::stod(val));
            std::sort(num_vals.begin(), num_vals.end());

            std::map<double, double> target_map;
            for (size_t i = 0; i < num_vals.size(); ++i) {
                target_map[num_vals[i]] = static_cast<double>(i);
            }
            for (int i = 0; i < n_rows; ++i) {
                y[i] = target_map[std::stod(raw_y[i])];
            }
        }
    } else {
        // Map categorical targets (e.g. "no"/"yes") deterministically to 0.0 and 1.0
        std::vector<std::string> unique_targets(unique_targets_set.begin(), unique_targets_set.end());
        std::map<std::string, double> target_map;
        for (size_t i = 0; i < unique_targets.size(); ++i) {
            target_map[unique_targets[i]] = static_cast<double>(i);
        }
        for (int i = 0; i < n_rows; ++i) {
            y[i] = target_map[raw_y[i]];
        }
    }

    // Process feature columns (excluding 'target')
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

    // Build feature Matrix X
    Matrix X(n_rows, total_features);

    for (int i = 0; i < n_rows; ++i) {
        int current_out_col = 0;
        for (size_t k = 0; k < feature_cols.size(); ++k) {
            int col_idx = feature_cols[k];
            const std::string& val = raw_rows[i][col_idx];

            if (col_infos[k].is_numeric) {
                X.at(i, current_out_col++) = std::stod(val);
            } else {
                // Perform One-Hot Encoding for categorical feature values
                int cat_idx = col_infos[k].cat_to_idx[val];
                int num_cats = static_cast<int>(col_infos[k].unique_categories.size());
                for (int c = 0; c < num_cats; ++c) {
                    X.at(i, current_out_col + c) = (c == cat_idx) ? 1.0 : 0.0;
                }
                current_out_col += num_cats;
            }
        }
    }

    return X;
}

class NystromKELM {
public:
    double gamma;
    double lamda;
    int n_landmarks;
    bool class_weighted;
    unsigned int seed;

    NystromKELM(double g = 0.1, double l = 1.0, int m = 500, bool cw = true, unsigned int s = 42)
        : gamma(g), lamda(l), n_landmarks(m), class_weighted(cw), seed(s), m_cached_(0) {}

    void compute_rbf_kernel(const double* A, const double* B, double* K, int n, int m, int d) {
        // K = -2 * A * B^T
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, -2.0, A, d, B, d, 0.0, K, m);

        // compute squared norm
        std::vector<double> a_sq(n), b_sq(m);
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < d; ++j) sum += A[i * d + j] * A[i * d + j];
            a_sq[i] = sum;
        }

        // compute squared norm
        #pragma omp parallel for
        for (int i = 0; i < m; ++i) {
            double sum = 0.0;
            for (int j = 0; j < d; ++j) sum += B[i * d + j] * B[i * d + j];
            b_sq[i] = sum;
        }

        // distance computation and Exp
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            double* row_k = &K[i * m];
            double asq = a_sq[i];
            for (int j = 0; j < m; ++j) {
                double dist_sq = std::max(0.0, asq + b_sq[j] + row_k[j]);
                row_k[j] = std::exp(-gamma * dist_sq);
            }
        }
    }

    void fit(const double* X, const double* y, int n, int d) {
        int m = std::min(n_landmarks, n);

        // Landmark selection (uniform random selection using configurable seed)
        std::vector<double> landmarks(m * d);
        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 rng(seed);
        std::shuffle(indices.begin(), indices.end(), rng);

        for (int i = 0; i < m; ++i) {
            std::memcpy(&landmarks[i * d], &X[indices[i] * d], d * sizeof(double));
        }

        // Kmm matrix and spectral decomposition
        Matrix Kmm(m, m);
        compute_rbf_kernel(landmarks.data(), landmarks.data(), Kmm.ptr(), m, m, d);
        for (int i = 0; i < m; ++i) Kmm.at(i, i) += 1e-8; // regularization for numerical stability

        std::vector<double> w_eig(m);
        int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', m, Kmm.ptr(), m, w_eig.data());
        if (info != 0) throw std::runtime_error("Kmm spectral decomposition failed");

        // Adaptive threshold for numerical stability of truncated pseudo-inverse
        double max_eig = 0.0;
        for (int i = 0; i < m; ++i) max_eig = std::max(max_eig, w_eig[i]);
        double tol = std::max(1e-12, 1e-10 * max_eig);

        // In LAPACKE row-major layout, the eigenvectors are the COLUMNS of Kmm (Kmm = V).
        // Compute Kmm_scaled = V * Sigma^{-1/2}, multiplying column j by lambda_j^{-1/2}
        // (zero for eigenvalues below the threshold: truncated pseudo-inverse).
        std::vector<double> inv_sqrt(m);
        for (int j = 0; j < m; ++j) inv_sqrt[j] = (w_eig[j] > tol) ? (1.0 / std::sqrt(w_eig[j])) : 0.0;
        Matrix Kmm_scaled(m, m);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                Kmm_scaled.at(i, j) = Kmm.at(i, j) * inv_sqrt[j];
            }
        }

        // Kmm^{-1/2} = V * Sigma^{-1/2} * V^T = Kmm_scaled * Kmm^T
        Matrix Kmm_inv_sqrt(m, m);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, m, m,
                    1.0, Kmm_scaled.ptr(), m, Kmm.ptr(), m, 0.0, Kmm_inv_sqrt.ptr(), m);

        // Feature map U = Knm * Kmm^{-1/2}
        Matrix Knm(n, m);
        compute_rbf_kernel(X, landmarks.data(), Knm.ptr(), n, m, d);
        Matrix U(n, m);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, m, m,
                    1.0, Knm.ptr(), m, Kmm_inv_sqrt.ptr(), m, 0.0, U.ptr(), m);

        // Ridge Regression (weighted or standard)
        std::vector<double> LHS(m * m, 0.0);
        std::vector<double> RHS(m, 0.0);

        if (class_weighted) {
            double c0 = 0, c1 = 0;
            for (int i = 0; i < n; ++i) (y[i] == 1.0) ? ++c1 : ++c0;
            double w0 = n / (2.0 * std::max(1.0, c0)), w1 = n / (2.0 * std::max(1.0, c1));

            Matrix U_w(n, m);
            std::vector<double> target_w(n);
            #pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                double weight = (y[i] == 1.0) ? w1 : w0;
                double sqrt_w = std::sqrt(weight);
                double target = (y[i] * 2.0) - 1.0;
                for (int j = 0; j < m; ++j) U_w.at(i, j) = U.at(i, j) * sqrt_w;
                target_w[i] = target * sqrt_w;
            }

            // use cblas_dsyrk in place of cblas_dgemm: optimize exploiting the symmetry of the matrix
            // by performing one half of the floating point operations required
            cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, m, n, 1.0, U_w.ptr(), m, 0.0, LHS.data(), m);
            cblas_dgemv(CblasRowMajor, CblasTrans, n, m, 1.0, U_w.ptr(), m, target_w.data(), 1, 0.0, RHS.data(), 1);
        } else {
            std::vector<double> target(n);
            #pragma omp parallel for
            for (int i = 0; i < n; ++i) target[i] = (y[i] * 2.0) - 1.0;

            // use cblas_dsyrk in place of cblas_dgemm: optimize exploiting the symmetry of the matrix
            // by performing one half of the floating point operations required
            cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, m, n, 1.0, U.ptr(), m, 0.0, LHS.data(), m);
            cblas_dgemv(CblasRowMajor, CblasTrans, n, m, 1.0, U.ptr(), m, target.data(), 1, 0.0, RHS.data(), 1);
        }

        // regularization for numerical stability
        for (int i = 0; i < m; ++i) LHS[i * m + i] += lamda;

        // solve linear system using Cholesky
        int info_solve = LAPACKE_dposv(LAPACK_ROW_MAJOR, 'U', m, 1, LHS.data(), m, RHS.data(), 1);
        if (info_solve != 0) {
            // If Cholesky fails (ill-conditioned linear system), fill-in the lower triangle and use LU decomposition instead
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < i; ++j) {
                    LHS[i * m + j] = LHS[j * m + i];
                }
            }
            std::vector<int> ipiv(m);
            info_solve = LAPACKE_dgesv(LAPACK_ROW_MAJOR, m, 1, LHS.data(), m, ipiv.data(), RHS.data(), 1);
            if (info_solve != 0) throw std::runtime_error("Linear system solution failed (Cholesky and LU)");
        }

        // precompute beta = Kmm_inv_sqrt * alpha
        std::vector<double> alpha = RHS;
        std::vector<double> beta(m, 0.0);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, m, m, 1.0, Kmm_inv_sqrt.ptr(), m, alpha.data(), 1, 0.0, beta.data(), 1);

        this->beta_cached_ = beta;
        this->landmarks_cached_ = landmarks;
        this->m_cached_ = m;
    }

    void predict_proba(const double* X_new, double* probs_out, int n_new, int d) {
        if (m_cached_ == 0) throw std::runtime_error("The model has not been trained");

        Matrix Knm_new(n_new, m_cached_);
        compute_rbf_kernel(X_new, landmarks_cached_.data(), Knm_new.ptr(), n_new, m_cached_, d);

        #pragma omp parallel for
        for (int i = 0; i < n_new; ++i) {
            double raw = 0.0;
            for (int j = 0; j < m_cached_; ++j) {
                raw += Knm_new.at(i, j) * beta_cached_[j];
            }
            probs_out[i] = 1.0 / (1.0 + std::exp(-std::clamp(raw, -500.0, 500.0)));
        }
    }

private:
    std::vector<double> beta_cached_;
    std::vector<double> landmarks_cached_;
    int m_cached_;
};

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cout << "Usage: " << argv[0] << " <landmarks> [seed]" << std::endl;
        return 1;
    }

    std::string filename = "./data/diabetes.csv";
    int n_landmarks = std::stoi(argv[1]);
    unsigned int seed = (argc == 3) ? static_cast<unsigned int>(std::stoul(argv[2])) : 42;

    try {
        std::vector<double> y;
        Matrix X = read_csv_into_matrix(filename, y);

        std::cout << "Dataset: " << filename << " (" << X.rows << "x" << X.cols << ")\n";
        std::cout << "Random Seed: " << seed << "\n";

#ifdef _OPENMP
        int max_threads = omp_get_max_threads();
        std::cout << "OpenMP maximum threads: " << max_threads << "\n";
        omp_set_num_threads(max_threads / 2);
        std::cout << "OpenMP threads in use: " << max_threads / 2 << "\n";
#else
        std::cout << "OpenMP not available\n";
#endif

        NystromKELM model(0.1, 1.0, n_landmarks, true, seed);

        auto start = std::chrono::high_resolution_clock::now();
        model.fit(X.ptr(), y.data(), X.rows, X.cols);
        auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "Training Time: " << std::chrono::duration<double, std::milli>(stop - start).count() << " ms\n";

        std::vector<double> probs(X.rows);
        start = std::chrono::high_resolution_clock::now();
        model.predict_proba(X.ptr(), probs.data(), X.rows, X.cols);
        stop = std::chrono::high_resolution_clock::now();
        std::cout << "Prediction Time: " << std::chrono::duration<double, std::milli>(stop - start).count() << " ms\n";

        for (int i = 0; i < std::min(10, (int)probs.size()); ++i)
            std::cout << "  [" << i << "] " << std::fixed << std::setprecision(7) << probs[i] << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}