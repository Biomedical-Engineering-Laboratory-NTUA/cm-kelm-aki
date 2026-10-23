# C++ backends of the Nyström KELM fit

Stand-alone implementations of one class-weighted Nyström KELM fit and prediction
(the operation inside every CM-KELM fitness evaluation), by M. Cafaro (MIT License):

| Folder | Backend | Build requirements |
|---|---|---|
| `cblas-lapack/` | CPU, CBLAS + LAPACKE, OpenMP | g++, OpenBLAS, LAPACKE (Intel MPI/MKL alternative commented in the Makefile) |
| `eigen/` | CPU, Eigen with BLAS/LAPACKE back end, OpenMP | g++, Eigen (tested with 5.0.1), OpenBLAS, LAPACKE |
| `cuda/` | one NVIDIA GPU, cuBLAS + cuSOLVER | CUDA Toolkit |
| `mpi-cuda/` | several GPUs or nodes, MPI + CUDA | CUDA Toolkit, MPI (`mpicxx`, `mpirun`) |

All four use the same algorithm: uniform landmarks drawn with `std::mt19937(seed)`, RBF kernel,
eigendecomposition of K_mm (ridge 1e-8) with a truncated inverse square root (eigenvalues below
1e-10 of the largest are dropped), class weights n / (2 n_c), and a Cholesky solve of
(U^T W U + λI) α = U^T W t. With the same seed they select the same landmarks and give the same
predictions.

## Data

The programs read `./data/diabetes.csv` (a column named `target`, all others features). To
download the public example data (CDC Diabetes Health Indicators, UCI, CC BY 4.0) and link it
into every folder:

```bash
pip install ucimlrepo
python examples/get_diabetes.py          # from the repository root
```

## Build and run

```bash
cd cpp/cblas-lapack && make && ./nystrom 500 42            # <landmarks> [seed], default seed 42
cd cpp/eigen && make EIGEN_INC=/usr/include/eigen3 && ./nystrom 500 42
cd cpp/cuda && make && ./nystrom 500 42
cd cpp/mpi-cuda && make && mpirun -np 2 ./nystrom 500 42
```

The CUDA Makefiles compile for `-arch=sm_89` (RTX 40 series / RTX 6000 Ada); set the value for your
GPU (e.g. `sm_80` for A100, `sm_86` for RTX 30 series, `sm_90` for H100). An architecture that does
not match the GPU makes the custom kernels fail to launch.

## Check

On the example data (n = 253,680, m = 500, seed 42) the four backends gave the same first ten
probabilities to the printed precision (7 decimals), which also match an independent NumPy
computation. Training took about 4 s on 8 CPU cores (CBLAS/LAPACK, Eigen) and 0.27 s on one
NVIDIA RTX 6000 Ada (CUDA, MPI + CUDA). The timings in the article (m = 1000 and 5000 on the
training set of the final INSPIRE cohort) cannot be reproduced without access to INSPIRE.
