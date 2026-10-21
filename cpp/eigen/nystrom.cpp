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

// October 18, 2026
// Nystrom-KELM v0.5
//
// This code implements, in C++ using the Eigen and OpenBLAS libraries,
// a Nystrom-approximated Kernel Estreme Learning Machine
// by using a standard kernel ridge regression; e.g. Williams & Seeger 2001
//
//
// This code implements, in C++ using the Eigen and OpenBLAS libraries,
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
#include <omp.h>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/Cholesky"
#include "Eigen/Eigenvalues"
#include "Eigen/QR"

// Datatypes
using MatrixXd = Eigen::MatrixXd;                         // Dynamic, ColMajor
using VectorXd = Eigen::VectorXd;                         // Dynamic column
using ArrayXd  = Eigen::ArrayXd;
using RMMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>;
using RMRMConst = Eigen::Map<const RMMatrix>;             // Zero-copy view per RowMajor

// Helper: Trim surrounding whitespace and quote characters
static std::string trim(const std::string& str) {
  size_t first = str.find_first_not_of(" \t\r\n\"'");
  if (first == std::string::npos) return "";
  size_t last = str.find_last_not_of(" \t\r\n\"'");
  return str.substr(first, (last - first + 1));
}

// Helper: Parse a single CSV line while preserving text inside quotes
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

// csv file reader supporting arbitrary 'target' position and mixed feature types
// RMMatrix is RowMajor for memory layout correctness
static void read_csv(const std::string &filename, RMMatrix &X,
                     std::vector<double> &y) {
  std::ifstream file(filename, std::ios::binary);
  if (!file)
    throw std::runtime_error("Cannot open file: " + filename);

  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  file.close();

  std::vector<std::string> lines;
  {
    size_t start = 0;
    for (size_t i = 0; i <= content.size(); ++i) {
      if (i == content.size() || content[i] == '\n') {
        size_t end = i;
        if (end > start && content[end - 1] == '\r') --end;
        if (end > start)
          lines.push_back(content.substr(start, end - start));
        start = i + 1;
      }
    }
  }
  if (lines.size() < 2)
    throw std::runtime_error("CSV file has no data rows");

  std::vector<std::string> headers = parse_csv_line(lines[0]);
  if (headers.size() < 2)
    throw std::runtime_error("Need at least 2 columns (features + target)");

  // Locate the 'target' column index (case-insensitive)
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

  const int n = static_cast<int>(lines.size()) - 1;
  const int n_cols = static_cast<int>(headers.size());

  std::vector<std::vector<std::string>> raw_rows;
  raw_rows.reserve(n);
  for (int i = 0; i < n; ++i) {
    std::vector<std::string> row = parse_csv_line(lines[i + 1]);
    if (static_cast<int>(row.size()) != n_cols) {
      throw std::runtime_error("Row " + std::to_string(i + 2) + " column mismatch");
    }
    raw_rows.push_back(row);
  }

  // Process target column
  std::vector<std::string> raw_y(n);
  std::set<std::string> unique_targets_set;
  for (int i = 0; i < n; ++i) {
    raw_y[i] = raw_rows[i][target_col];
    unique_targets_set.insert(raw_y[i]);
  }

  y.assign(n, 0.0);
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
      for (int i = 0; i < n; ++i) {
        y[i] = std::stod(raw_y[i]);
      }
    } else {
      std::vector<double> num_vals;
      for (const auto& val : unique_targets_set) num_vals.push_back(std::stod(val));
      std::sort(num_vals.begin(), num_vals.end());

      std::map<double, double> target_map;
      for (size_t i = 0; i < num_vals.size(); ++i) {
        target_map[num_vals[i]] = static_cast<double>(i);
      }
      for (int i = 0; i < n; ++i) {
        y[i] = target_map[std::stod(raw_y[i])];
      }
    }
  } else {
    std::vector<std::string> unique_targets(unique_targets_set.begin(), unique_targets_set.end());
    std::map<std::string, double> target_map;
    for (size_t i = 0; i < unique_targets.size(); ++i) {
      target_map[unique_targets[i]] = static_cast<double>(i);
    }
    for (int i = 0; i < n; ++i) {
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

    for (int i = 0; i < n; ++i) {
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

  X.resize(n, total_features);

  for (int i = 0; i < n; ++i) {
    int current_out_col = 0;
    for (size_t k = 0; k < feature_cols.size(); ++k) {
      int col_idx = feature_cols[k];
      const std::string& val = raw_rows[i][col_idx];

      if (col_infos[k].is_numeric) {
        X(i, current_out_col++) = std::stod(val);
      } else {
        int cat_idx = col_infos[k].cat_to_idx[val];
        int num_cats = static_cast<int>(col_infos[k].unique_categories.size());
        for (int c = 0; c < num_cats; ++c) {
          X(i, current_out_col + c) = (c == cat_idx) ? 1.0 : 0.0;
        }
        current_out_col += num_cats;
      }
    }
  }
}

// RBF Kernel : K(i,j) = exp( -gamma * ||a_i - b_j||^2 )
static void compute_rbf_kernel(const MatrixXd &A, const MatrixXd &B,
                               MatrixXd &K, double gamma, int n, int m) {
  K.noalias() = A * B.transpose();
  K *= -2.0;

  ArrayXd a_sq(n), b_sq(m);
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i)
    a_sq[i] = A.row(i).squaredNorm();

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < m; ++i)
    b_sq[i] = B.row(i).squaredNorm();

  #pragma omp parallel for schedule(static)
  for (int j = 0; j < m; ++j) {
    const double bsq = b_sq[j];
    double *col = K.data() + static_cast<size_t>(j) * n;
    for (int i = 0; i < n; ++i) {
      double v = col[i] + a_sq[i] + bsq;
      col[i] = std::exp(-gamma * (v > 0.0 ? v : 0.0));
    }
  }
}

class NystromKELM {
public:
  double gamma;
  double lamda;
  int n_landmarks;
  bool class_weighted;
  unsigned long seed;

  NystromKELM(double gamma_ = 0.1, double lamda_ = 1.0, int n_lm = 500,
              bool cw = true, unsigned long seed_ = 42)
      : gamma(gamma_), lamda(lamda_), n_landmarks(n_lm), class_weighted(cw), seed(seed_) {}

  void fit(const double *X_ptr, const double *y_ptr, int n, int d) {
    
    RMRMConst X_user(X_ptr, n, d);
    MatrixXd X = X_user; // converting from RowMajor to ColMajor (Eigen standard)

    const int m = std::min(n_landmarks, n);

    // Reproducible landmark selection using specified seed
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    MatrixXd Lm(m, d);
    for (int i = 0; i < m; ++i)
      Lm.row(i) = X.row(indices[i]);

    // Kmm = RBF(Lm, Lm) + regularization
    MatrixXd Kmm(m, m);
    compute_rbf_kernel(Lm, Lm, Kmm, gamma, m, m);
    Kmm.diagonal().array() += 1e-8;

    // Spectral decomposition Kmm^{-1/2}
    Eigen::SelfAdjointEigenSolver<MatrixXd> es(Kmm);
    if (es.info() != Eigen::Success)
      throw std::runtime_error("EigenSolver fallito su Kmm");

    // adaptive threshold for eigenvalue truncation
    Eigen::VectorXd eig = es.eigenvalues();
    const double max_eig = eig.maxCoeff();
    const double tol = std::max(1e-12, 1e-10 * max_eig);

    Eigen::VectorXd inv_sqrt_eig = Eigen::VectorXd::Zero(m);
    for (int i = 0; i < m; ++i) {
      if (eig(i) > tol) {
        inv_sqrt_eig(i) = 1.0 / std::sqrt(eig(i));
      }
    }

    MatrixXd Kmm_inv_sqrt =
        es.eigenvectors() * inv_sqrt_eig.asDiagonal() *
        es.eigenvectors().transpose();
    Kmm_inv_sqrt = 0.5 * (Kmm_inv_sqrt + Kmm_inv_sqrt.transpose());

    // Knm = RBF(X, Lm)
    MatrixXd Knm(n, m);
    compute_rbf_kernel(X, Lm, Knm, gamma, n, m);

    // Feature map U = Knm * Kmm^{-1/2}
    MatrixXd U = Knm * Kmm_inv_sqrt;

    // linear system for ridge regression
    MatrixXd LHS(m, m);
    VectorXd RHS(m);

    if (class_weighted) {
      int c0 = 0, c1 = 0;
      for (int i = 0; i < n; ++i)
        (y_ptr[i] == 1.0) ? ++c1 : ++c0;

      if (c0 == 0 || c1 == 0)
        throw std::runtime_error("There are no samples of both classes for class_weighted");

      const double w0 = n / (2.0 * c0);
      const double w1 = n / (2.0 * c1);

      #pragma omp parallel for schedule(static)
      for (int i = 0; i < n; ++i) {
        const double sw = std::sqrt(y_ptr[i] == 1.0 ? w1 : w0);
        for (int j = 0; j < m; ++j)
          U(i, j) *= sw;
      }

      LHS = U.transpose() * U;

      VectorXd tw(n);
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < n; ++i)
        tw[i] = (y_ptr[i] * 2.0 - 1.0) * std::sqrt(y_ptr[i] == 1.0 ? w1 : w0);
      RHS = U.transpose() * tw;
    } else {
      LHS = U.transpose() * U;

      VectorXd tw(n);
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < n; ++i)
        tw[i] = y_ptr[i] * 2.0 - 1.0;
      RHS = U.transpose() * tw;
    }

    // regularization
    LHS.diagonal().array() += lamda;

    Eigen::LLT<MatrixXd> llt(LHS);
    if (llt.info() == Eigen::Success) {
      alpha_ = llt.solve(RHS);
    } else {
      Eigen::LDLT<MatrixXd> ldlt(LHS);
      if (ldlt.info() == Eigen::Success) {
        alpha_ = ldlt.solve(RHS);
      } else {
        alpha_ = LHS.colPivHouseholderQr().solve(RHS);
      }
    }

    // Pre-computing beta = Kmm^{-1/2} * alpha
    beta_ = Kmm_inv_sqrt * alpha_;

    landmarks_ = Lm;
    m_ = m;
  }

  void predict_proba(const double *X_new, double *probs_out, int n_new, int d) {
    if (m_ == 0) throw std::runtime_error("Modello non addestrato");

    // converting input RowMajor buffer
    RMRMConst Xn(X_new, n_new, d);
    MatrixXd Xn_col = Xn;

    MatrixXd Knm(n_new, m_);
    compute_rbf_kernel(Xn_col, landmarks_, Knm, gamma, n_new, m_);

    VectorXd raw = Knm * beta_;

    VectorXd prob(n_new);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_new; ++i) {
      const double z = raw(i);
      if (z > 500.0)
        prob(i) = 1.0;
      else if (z < -500.0)
        prob(i) = 0.0;
      else
        prob(i) = 1.0 / (1.0 + std::exp(-z));
    }

    std::memcpy(probs_out, prob.data(), static_cast<size_t>(n_new) * sizeof(double));
  }

private:
  VectorXd alpha_;
  VectorXd beta_;
  MatrixXd landmarks_;
  int m_ = 0;
};

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    std::cout << "Usage: " << argv[0] << " <landmarks> [seed]\n";
    return 1;
  }

  const std::string filename = "./data/diabetes.csv";
  const int n_landmarks = std::stoi(argv[1]);
  const unsigned long seed = (argc == 3) ? std::stoul(argv[2]) : 42;

  RMMatrix X; // RowMajor order
  std::vector<double> y;
  read_csv(filename, X, y);

  const int n = static_cast<int>(X.rows());
  const int d = static_cast<int>(X.cols());

  std::cout << "File:          " << filename << '\n';
  std::cout << "X dimensions:  " << n << " x " << d << '\n';
  std::cout << "y size:        " << y.size() << '\n';

  bool all_binary = true;
  for (double v : y)
    if (v != 0.0 && v != 1.0) { all_binary = false; break; }
  std::cout << "y in {0,1}:     " << (all_binary ? "YES" : "NO") << '\n';
  if (!all_binary) return 1;

  std::cout << "Nystrom-KELM  |  n=" << n << "  d=" << d
            << "  m=" << n_landmarks << "  seed=" << seed << "\n";

#ifdef _OPENMP
  const int nt = omp_get_max_threads();
  omp_set_num_threads(nt / 2);
  std::cout << "OpenMP threads: " << nt / 2 << "\n";
#else
  std::cout << "OpenMP: not available\n";
#endif

  NystromKELM model(0.1, 1.0, n_landmarks, true, seed);

  auto t0 = std::chrono::high_resolution_clock::now();
  model.fit(X.data(), y.data(), n, d);
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "Training Time: "
            << std::chrono::duration<double, std::milli>(t1 - t0).count()
            << " ms\n";

  std::vector<double> probs(n);
  t0 = std::chrono::high_resolution_clock::now();
  model.predict_proba(X.data(), probs.data(), n, d);
  t1 = std::chrono::high_resolution_clock::now();
  std::cout << "Prediction Time: "
            << std::chrono::duration<double, std::milli>(t1 - t0).count()
            << " ms\n";

  std::cout << "First 10 probabilities:\n";
  for (int i = 0; i < std::min(10, n); ++i)
    std::cout << "  [" << i << "]  " << std::fixed << std::setprecision(7)
              << probs[i] << "\n";

  return 0;
}