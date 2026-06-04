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

- experiments: each markdown file describes a slurm experiment, its setup and the decisions taken from its results.

## Patterns

- Never try to compile yourself, always ask me to compile and test myself after you have implemented code changes
- when tweaking new versions of the code, always keep old explanatory comments about the physics and code structure or change them to adapt if appropriate.

## Finished tasks
- Turned the original two-step algorithm into a one-step algorithm with pull scheme. Both SoA and AoS versions have been tested for performance and correctness. SoA version was chosen for MPI and GPU implementation.

## Current Focus
The MPI implementation is not working and all velocity values explode and nan out within 1000 steps. Debugging is in progress to understand the cause.