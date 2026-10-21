#
# Nystrom-accelerated KELM
#
# Compiled and tested on the following platform:
# Ubuntu linux 26.04.1 LTS
# CPU: Intel(R) Xeon(R) w7-2495X, 24 cores, 2 threads per core
# GPU: NVIDIA GeForce RTX 4090
# Compiler: nvcc 13.4
#
# Dependencies: cublas, cusolver and cuda libraries
#
# September 23, 2026
#
# Massimo Cafaro, Ph.D.
# Associate Professor,
# Department of Engineering for Innovation
# University of Salento, Lecce, Italy
#
# Usage: nystrom <number of landmarks>


CC=nvcc
CFLAGS=-arch=sm_89 -O3 -Xptxas -O3 -std=c++17
#CFLAGS=-arch=native -O3 -Xptxas -O3 -std=c++17
LDFLAGS=-lcublas -lcusolver -lcuda


EXECUTABLE=nystrom
DEPS=nystrom_cuda.cu

all: $(EXECUTABLE)


$(EXECUTABLE):$(DEPS)
	@echo "Compiling for " $(NAME)
	$(CC) -o $@ $(CFLAGS) $(DEPS) $(LDFLAGS)

clean:
	rm -rf $(EXECUTABLE)

