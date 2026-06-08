This project seeks to parallelize a code for simulating 2D flow around a cylinder using the Lattice Boltzmann Method (LBM). The project deliverables are as follows:
an MPI implementation with a weak scaling and stron scaling analysis, a CUDA version with a performance analysis, and a report describing the implementation and results. 


## File structure
- different folders for different implementations (serial, mpi, cuda) each containing their own version of 
    - lbm_{implementation}.cpp: the main LBM class initializing memory layout and a time-stepping method.
    - main_{implementation}.cpp: parsing command line arguments, initializing the LBM class, running the time-stepping loop, and writing outputs.
    - output_{implementation}.cc: functions for writing outputs using HDF5.
    output_{implementation}.cc, main_{implementation}.cpp
   - Makefile

- current implementations: 
    - serial: original two-step algorithm, least performant
    - serial_fused_AoS: one-step algorithm with an Array of Structures (AoS) memory layout, most performant on a single CPU
    - serial_fused_SoA: one-step algorithm with a Structure of Arrays (SoA) memory layout, but used for parallelization because literature indicates it runs better on GPUs.
    - mpi: parallelization of the one-step algorithm with a SoA memory layout using MPI, with domain decomposition in the x direction and halo exchange.
    - mpi_cart: parallelization of the one-step algorithm with a SoA memory layout using MPI, with domain decomposition in both x and y directions and halo exchange.
    - cuda_SoA: parallelization of the one-step algorithm with a SoA memory layout using CUDA

- experiments: each markdown file describes a slurm experiment, its setup and the decisions taken from its results.

## Patterns

- Never try to compile yourself, always ask me to compile and test myself after you have implemented code changes
- when tweaking new versions of the code, always keep old explanatory comments about the physics and code structure or change them to adapt if appropriate.

## Finished tasks
- Turned the original two-step algorithm into a one-step algorithm with pull scheme. Both SoA and AoS versions have been tested for performance and correctness. SoA version was chosen for MPI and GPU implementation.
- Implemented MPI parallelization with domain decomposition in the y direction and halo exchange. Code was debugged and now works correctly with multiple ranks. Validated on a 800x400 grid for 60000 steps with the correct Strouhal number of 16.00.
- Strong and Weak scaling analysis of the MPI version. strong scaling shows good parallel efficiency but decays sooner than Amdahl's law would predict (communication overhead). Weak scaling shows great performance when only scaling the number of rows (above 0.9 efficiency at 64 ranks). When scaling both dimensions, performance much worse, dropping below 0.4 at 64 ranks. Explanation: the rows get longer and the communication/computation ratio increases.
- MPI cart version with domain decomposition in 2D implemented and tested. Shows much better performance when scaling both dimensions.

## Current Focus

testing the cuda_SoA version