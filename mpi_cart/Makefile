# this ?= means "assign if not defined", aka if a conda env is activated CXX will 
# already be set and this command does nothing
CXX      = mpicxx


# We use the CXX variable for both compiling and linking, which is common practice
# ":=" means "assign immediately", so that if CXX is later changed, LD won't be affected.
LD       := $(CXX)


# Only inject HDF5 include/lib paths when $HDF5_ROOT is set. Inside an
# activated conda env, the compiler driver already knows where HDF5 lives,
# so leaving HDF5_ROOT empty is fine and avoids spurious "-I/include" /
# "-L/lib" flags.
HDF5_INC := $(if $(strip $(HDF5_ROOT)),-I$(HDF5_ROOT)/include)
HDF5_LIB := $(if $(strip $(HDF5_ROOT)),-L$(HDF5_ROOT)/lib)
CXXFLAGS += -O3 -g -Wall -Wextra -Werror -pedantic -std=c++11 $(HDF5_INC)
LDFLAGS  += $(CXXFLAGS) $(HDF5_LIB) -lhdf5 -lm

# Lists the three object files that will be compiled and linked together into the final binary.
OBJS = main_mpi_cart.o lbm_mpi_cart.o output_mpi_cart.o

all: lbm_mpi_cart

lbm_mpi_cart: $(OBJS)
# $@ is a Make automatic variable meaning "the target name" — here, lbm
	$(LD) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f lbm_mpi_cart *.o *~

distclean: clean
	rm -rf out/ probe.csv *.gif

# Declares these three targets as phony — meaning they don't correspond to actual files on disk. 
# Without this, if a file named clean somehow existed, Make would think the target was already up to date
# and skip running it.
.PHONY: all clean distclean
