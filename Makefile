# TODO[Davor]: confirm SCITAS module names. Suggested:
#   module load gcc hdf5
# HDF5_ROOT must point at an HDF5 install that provides include/ and lib/.

VARIANTS = serial mpi serial_fused_AoS serial_fused_SoA

all: $(VARIANTS)

serial mpi serial_fused_AoS serial_fused_SoA:
	$(MAKE) -C $@

clean:
	for v in $(VARIANTS); do $(MAKE) -C $$v clean; done

distclean:
	for v in $(VARIANTS); do $(MAKE) -C $$v distclean; done

# $(VARIANTS) is listed as phony so 'make serial' and 'make mpi' always
# delegate to the subdirectory even if a file/dir of that name exists.
.PHONY: all clean distclean $(VARIANTS)