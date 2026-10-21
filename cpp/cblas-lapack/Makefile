#
# Nystrom-accelerated KELM
#
# Compiled and tested on the following platform:
# Ubuntu linux 26.04.1 LTS
# CPU: Intel(R) Xeon(R) w7-2495X, 24 cores, 2 threads per core
# Compiler: Intel(R) oneAPI DPC++/C++ Compiler 2026.1.1 (2026.1.1.20260724)
# Compiler: gcc 15.2.0

#
# Dependencies: OpenBLAS, lapack, lapacke libraries
# Note: the mvec library is included in the GNU C library (glibc)
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

CC=g++
CFLAGS=-std=c++17 -O3 -march=native -mtune=native -flto -funroll-loops -fprefetch-loop-arrays -fopenmp -I/usr/include
LDFLAGS=-L/usr/lib -llapacke -llapack -lopenblas -lmvec


#CC=icpx
#CFLAGS=-std=c++17 -O3 -fiopenmp -fopenmp-targets=spir64 -I"${MKLROOT}/include"
#LDFLAGS=-L${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl


EXECUTABLE=nystrom
DEPS=nystrom.cpp

all: $(EXECUTABLE)


$(EXECUTABLE):$(DEPS)
	@echo "Compiling for " $(NAME)
	$(CC) -o $@ $(CFLAGS) $(DEPS) $(LDFLAGS)

clean:
	rm -rf $(EXECUTABLE)

