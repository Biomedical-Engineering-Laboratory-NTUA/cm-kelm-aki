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

NVCC    := nvcc
MPICXX  := mpicxx

CUDA_PATH := /usr/local/cuda
CUDA_INC  := -I$(CUDA_PATH)/include
CUDA_LIB  := -L$(CUDA_PATH)/lib64

# -O3: Maximum optimization
# -arch=sm_80: Targeted at NVIDIA A100.
# Change sm_80 to:
#   sm_70 for V100
#   sm_75 for T4 / RTX 20 series
#   sm_86 for RTX 30 series
#   sm_90 for H100
NVCC_FLAGS := -O3 -arch=sm_89 $(CUDA_INC)
CXX_FLAGS  := -O3 -std=c++17 -Wall $(CUDA_INC)

# Order matters: cublas and cusolver depend on cudart
LDFLAGS    := $(CUDA_LIB) -lcudart -lcublas -lcusolver

TARGET := nystrom
OBJS   := nystrom_mpi.o nystrom_cuda.o

.PHONY: all clean

all: $(TARGET)

# Linking the final executable
$(TARGET): $(OBJS)
	$(MPICXX) $(CXX_FLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)
	@echo "--------------------------------------------------------"
	@echo "Build Successful: ./$(TARGET)"
	@echo "Execution: mpirun -np <num_procs> ./$(TARGET) <n_landmarks> [seed]"
	@echo "--------------------------------------------------------"

# Compiling the MPI C++ source
nystrom_mpi.o: nystrom_mpi.cpp nystrom.h
	$(MPICXX) $(CXX_FLAGS) -c nystrom_mpi.cpp -o nystrom_mpi.o

# Compiling the CUDA source
nystrom_cuda.o: nystrom_cuda.cu nystrom.h
	$(NVCC) $(NVCC_FLAGS) -c nystrom_cuda.cu -o nystrom_cuda.o

# Clean build artifacts
clean:
	rm -f $(OBJS) $(TARGET)

