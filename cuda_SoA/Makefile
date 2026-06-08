NVCC     ?= nvcc
CXX      ?= g++

# Only inject HDF5 paths when HDF5_ROOT is set (conda env already exposes them).
HDF5_INC := $(if $(strip $(HDF5_ROOT)),-I$(HDF5_ROOT)/include)
HDF5_LIB := $(if $(strip $(HDF5_ROOT)),-L$(HDF5_ROOT)/lib)

# sm_70 = Volta (V100).  --ptxas-options=-v prints register and shared memory
# usage per kernel, useful for occupancy tuning.
NVCCFLAGS = -O3 -arch=sm_70 -std=c++14 --ptxas-options=-v $(HDF5_INC)

# output_cuda_SoA.cc is plain C++ (no CUDA keywords) and is compiled by g++.
CXXFLAGS  = -O3 -g -Wall -Wextra -std=c++14 $(HDF5_INC)

LDFLAGS   = $(HDF5_LIB) -lhdf5 -lm

OBJS = main_cuda_SoA.o lbm_cuda_SoA.o output_cuda_SoA.o

all: lbm_cuda_SoA

# Link with nvcc so that the CUDA runtime (-lcudart) is handled automatically.
lbm_cuda_SoA: $(OBJS)
	$(NVCC) -arch=sm_70 -o $@ $(OBJS) $(LDFLAGS)

main_cuda_SoA.o: main_cuda_SoA.cu lbm_cuda_SoA.cuh output_cuda_SoA.hh
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

lbm_cuda_SoA.o: lbm_cuda_SoA.cu lbm_cuda_SoA.cuh
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

output_cuda_SoA.o: output_cuda_SoA.cc output_cuda_SoA.hh lbm_cuda_SoA.cuh
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f lbm_cuda_SoA *.o *~

distclean: clean
	rm -rf out/ probe.csv *.xdmf

.PHONY: all clean distclean
