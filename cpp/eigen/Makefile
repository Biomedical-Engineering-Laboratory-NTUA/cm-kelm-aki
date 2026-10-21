#
# Nystrom-accelerated KELM
#
# Compiled and tested on the following platform:
# Ubuntu linux 26.04.1 LTS
# CPU: Intel(R) Xeon(R) w7-2495X, 24 cores, 2 threads per core
# Compiler: gcc 15.2.0
#
# Dependencies: Eigen library, using as backend the OpenBLAS library
# The code is also parallelized using OpenMP
#
# September 23, 2026
#
# Massimo Cafaro, Ph.D.
# Associate Professor,
# Department of Engineering for Innovation
# University of Salento, Lecce, Italy
#
# Usage: nystrom <number of landmarks>


#CC=icpx
#CFLAGS=-std=c++17 -DEIGEN_USE_MKL_ALL -DNDEBUG -fiopenmp -fopenmp-targets=spir64 -I$(EIGEN_INC) -I"${MKLROOT}/include"
#LDFLAGS=-L${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

CC=g++
# Eigen is not bundled: install it (e.g. apt install libeigen3-dev, or conda install -c conda-forge eigen)
# and set EIGEN_INC if it is not in /usr/include/eigen3
EIGEN_INC ?= /usr/include/eigen3
CFLAGS=-DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE -std=c++17 -O3 -DNDEBUG -march=native -mtune=native -fopenmp -flto -funroll-loops -fprefetch-loop-arrays -I$(EIGEN_INC) -I/usr/include/x86_64-linux-gnu
LDFLAGS=-L/usr/lib/x86_64-linux-gnu -llapacke -llapack -lopenblas -lmvec


EXECUTABLE=nystrom
DEPS=nystrom.cpp

all: $(EXECUTABLE)

$(EXECUTABLE):$(DEPS)
	@echo "Compiling for " $(NAME)
	$(CC) -o $@ $(CFLAGS) $(DEPS) $(LDFLAGS)

clean:
	rm -rf $(EXECUTABLE)

