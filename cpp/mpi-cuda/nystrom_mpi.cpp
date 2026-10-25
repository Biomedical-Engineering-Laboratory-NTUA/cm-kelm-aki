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
// a Nystrom-approximated Kernel Extreme Learning Machine
// by using a standard kernel ridge regression; e.g. Williams & Seeger 2001
//
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

#include <mpi.h>

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

// Helper function: trim whitespace and surrounding quote marks
static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n\"'");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n\"'");
    return str.substr(first, (last - first + 1));
}

// Helper function: parse CSV row respecting quoted fields
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

// Helper function: determine whether a string field represents a valid floating point number
static bool is_numeric_field(const std::string& s) {
    std::string t = trim(s);
    if (t.empty()) return false;
    char* endptr = nullptr;
    std::strtod(t.c_str(), &endptr);
    return endptr != t.c_str() && *endptr == '\0';
}

void read_and_scatter_csv(const std::string &filename, std::vector<double> &X_loc, std::vector<double> &y_loc,
                          std::vector<double> &X_global_out, int &n_global, int &n_loc, int &d,
                          int rank, int world_size, MPI_Comm comm) {
    std::vector<double> X_global, y_global;
    if (rank == 0) {
        std::ifstream file(filename);
        if (!file.is_open()) throw std::runtime_error("Cannot open file: " + filename);
        std::string line;
        std::vector<std::string> lines;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) lines.push_back(line);
        }

        if (lines.size() < 2) throw std::runtime_error("CSV file empty or header-only");

        std::vector<std::string> headers = parse_csv_line(lines[0]);
        if (headers.size() < 2) throw std::runtime_error("CSV file must contain at least 2 columns (features and target)");

        // Search for target column index (case-insensitive)
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

        n_global = static_cast<int>(lines.size()) - 1;
        const int n_cols = static_cast<int>(headers.size());

        std::vector<std::vector<std::string>> raw_rows;
        raw_rows.reserve(n_global);
        for (int i = 0; i < n_global; ++i) {
            std::vector<std::string> row = parse_csv_line(lines[i + 1]);
            if (static_cast<int>(row.size()) != n_cols) {
                throw std::runtime_error("Row " + std::to_string(i + 2) + " column count mismatch");
            }
            raw_rows.push_back(row);
        }

        // Process target column
        std::vector<std::string> raw_y(n_global);
        std::set<std::string> unique_targets_set;
        for (int i = 0; i < n_global; ++i) {
            raw_y[i] = raw_rows[i][target_col];
            unique_targets_set.insert(raw_y[i]);
        }

        y_global.assign(n_global, 0.0);
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
                double v = std::stod(val);
                if (v != 0.0 && v != 1.0) {
                    already_binary_01 = false;
                    break;
                }
            }

            if (already_binary_01) {
                for (int i = 0; i < n_global; ++i) {
                    y_global[i] = std::stod(raw_y[i]);
                }
            } else {
                std::vector<double> num_vals;
                for (const auto& val : unique_targets_set) num_vals.push_back(std::stod(val));
                std::sort(num_vals.begin(), num_vals.end());

                std::map<double, double> target_map;
                for (size_t i = 0; i < num_vals.size(); ++i) {
                    target_map[num_vals[i]] = static_cast<double>(i);
                }
                for (int i = 0; i < n_global; ++i) {
                    y_global[i] = target_map[std::stod(raw_y[i])];
                }
            }
        } else {
            std::vector<std::string> unique_targets(unique_targets_set.begin(), unique_targets_set.end());
            std::map<std::string, double> target_map;
            for (size_t i = 0; i < unique_targets.size(); ++i) {
                target_map[unique_targets[i]] = static_cast<double>(i);
            }
            for (int i = 0; i < n_global; ++i) {
                y_global[i] = target_map[raw_y[i]];
            }
        }

        // Process feature columns (all columns except target)
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

            for (int i = 0; i < n_global; ++i) {
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

        d = total_features;
        X_global.assign(static_cast<size_t>(n_global) * d, 0.0);

        for (int i = 0; i < n_global; ++i) {
            int current_out_col = 0;
            for (size_t k = 0; k < feature_cols.size(); ++k) {
                int col_idx = feature_cols[k];
                const std::string& val = raw_rows[i][col_idx];

                if (col_infos[k].is_numeric) {
                    X_global[static_cast<size_t>(i) * d + current_out_col++] = std::stod(val);
                } else {
                    int cat_idx = col_infos[k].cat_to_idx[val];
                    int num_cats = static_cast<int>(col_infos[k].unique_categories.size());
                    X_global[static_cast<size_t>(i) * d + current_out_col + cat_idx] = 1.0;
                    current_out_col += num_cats;
                }
            }
        }
        X_global_out = X_global;
    }

    // Broadcast dimensions among MPI processes
    MPI_Bcast(&n_global, 1, MPI_INT, 0, comm);
    MPI_Bcast(&d, 1, MPI_INT, 0, comm);

    std::vector<int> sendcounts(world_size), displs(world_size);
    std::vector<int> sendcounts_y(world_size), displs_y(world_size);

    int base_rows = n_global / world_size;
    int rem = n_global % world_size;

    int current_disp = 0;
    for (int r = 0; r < world_size; ++r) {
        int rows = base_rows + (r < rem ? 1 : 0);
        sendcounts[r] = rows * d;
        displs[r] = current_disp * d;

        sendcounts_y[r] = rows;
        displs_y[r] = current_disp;

        if (r == rank) n_loc = rows;
        current_disp += rows;
    }

    X_loc.resize(static_cast<size_t>(n_loc) * d);
    y_loc.resize(n_loc);

    // Scatter matrix X and vector y among MPI processes
    MPI_Scatterv(X_global.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, X_loc.data(), n_loc * d, MPI_DOUBLE, 0, comm);
    MPI_Scatterv(y_global.data(), sendcounts_y.data(), displs_y.data(), MPI_DOUBLE, y_loc.data(), n_loc, MPI_DOUBLE, 0, comm);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank;
    MPI_Comm_rank(local_comm, &local_rank);

    // Assign GPU device to MPI process
    int num_gpus = get_cuda_device_count();
    set_cuda_device(local_rank % num_gpus);

    if (argc < 2 || argc > 3) {
        if (rank == 0) std::cout << "Usage: mpirun -np <N> " << argv[0] << " <n_landmarks> [seed]\n";
        MPI_Finalize();
        return 1;
    }

    int n_landmarks = std::stoi(argv[1]);
    unsigned long seed = 42; // Default seed for reproducibility
    if (argc == 3) {
        seed = std::stoul(argv[2]);
    }

    std::string filename = "./data/diabetes.csv";

    int n_global = 0, n_loc = 0, d = 0;
    std::vector<double> X_loc, y_loc, X_global;

    try {
        read_and_scatter_csv(filename, X_loc, y_loc, X_global, n_global, n_loc, d, rank, world_size, MPI_COMM_WORLD);
    } catch (const std::exception &e) {
        if (rank == 0) std::cerr << "Error reading file: " << e.what() << "\n";
        MPI_Finalize();
        return 1;
    }

    int m = std::min(n_landmarks, n_global);
    std::vector<double> h_landmarks(static_cast<size_t>(m) * d);

    // Reproducible Landmark selection on process rank 0
    if (rank == 0) {
        std::vector<int> idx(n_global);
        std::iota(idx.begin(), idx.end(), 0);

        std::mt19937 rng(seed);
        std::shuffle(idx.begin(), idx.end(), rng);

        for (int i = 0; i < m; ++i) {
            int selected_row = idx[i];
            std::copy(X_global.begin() + static_cast<size_t>(selected_row) * d,
                      X_global.begin() + static_cast<size_t>(selected_row + 1) * d,
                      h_landmarks.begin() + static_cast<size_t>(i) * d);
        }
    }

    // Broadcast landmarks among MPI processes
    MPI_Bcast(h_landmarks.data(), m * d, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Distributed counting of classes for weighting
    double c0_loc = 0.0, c1_loc = 0.0;
    for (int i = 0; i < n_loc; ++i) (y_loc[i] == 1.0) ? c1_loc += 1.0 : c0_loc += 1.0;

    double c0_glob = 0.0, c1_glob = 0.0;
    MPI_Allreduce(&c0_loc, &c0_glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&c1_loc, &c1_glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Instantiate model
    NystromKELM_Cuda model(0.1, 1.0, n_landmarks, true);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Local computation on GPU device
    std::vector<double> h_LHS_loc, h_RHS_loc;
    model.compute_local_matrices(X_loc.data(), y_loc.data(), n_loc, n_global, d, h_landmarks, c0_glob, c1_glob, h_LHS_loc, h_RHS_loc);

    // Communicate and reduce (summation) m^2 elements
    std::vector<double> h_LHS_glob(m * m), h_RHS_glob(m);
    MPI_Allreduce(h_LHS_loc.data(), h_LHS_glob.data(), m * m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(h_RHS_loc.data(), h_RHS_glob.data(), m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Solve linear system on GPU device
    model.solve_global_system(h_LHS_glob, h_RHS_glob);

    double t1 = MPI_Wtime();

    if (rank == 0) {
        std::cout << " MPI + CUDA Execution\n";
        std::cout << " Global Samples (N): " << n_global << " | Features (D): " << d << "\n";
        std::cout << " Landmarks (M):     " << m << "\n";
        std::cout << " MPI Processes:     " << world_size << " | GPU Devices: " << std::min(world_size, num_gpus) << "\n";
        std::cout << " Training Time:     " << (t1 - t0) * 1000.0 << " ms\n";
    }

    std::vector<double> probs_loc;
    t0 = MPI_Wtime();
    model.predict_proba_local(X_loc.data(), n_loc, d, probs_loc);

    std::vector<double> probs_global;
    if (rank == 0) probs_global.resize(n_global);
    std::vector<int> y_sendcounts(world_size), y_displs(world_size);
    int n_local_base = n_global / world_size;
    int remainder = n_global % world_size;
    int y_offset = 0;

    for (int i = 0; i < world_size; ++i) {
        int my_n = n_local_base + (i < remainder ? 1 : 0);
        y_sendcounts[i] = my_n;
        y_displs[i] = y_offset;
        y_offset += y_sendcounts[i];
    }

    int my_n = n_local_base + (rank < remainder ? 1 : 0);
    MPI_Gatherv(probs_loc.data(), my_n, MPI_DOUBLE, probs_global.data(), y_sendcounts.data(), y_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    t1 = MPI_Wtime();

    if (rank == 0) {
        std::cout << " Prediction Time:   " << (t1 - t0) * 1000.0 << " ms\n";
        std::cout << " First 10 Probabilities:\n";
        for (int i = 0; i < std::min(10, n_global); ++i) {
            std::cout << "  [" << i << "] " << std::fixed << std::setprecision(7) << probs_global[i] << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
