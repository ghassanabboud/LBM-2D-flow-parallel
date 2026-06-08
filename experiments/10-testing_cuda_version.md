# Experiment 10: testing cuda version
 #### **Code version:** cuda plots (76200ce11dae5339f0c82a486541cd2d21687c74)

## Results and Next Steps

Both SoA and AoS versions are validated. For the problem size analysis, both versions show a curve with increasing performance as problem size increases until a plateau is reached at 8192x4096. This is because for a fixed block size of 256, increasing problem size allows to launch more blocks to saturate the SMs and thus better hide memory latency through warp switching. For a V100, 80 SMs that can fit 8 blocks of 256 threads each. That means we need at least 163k cells to fully saturate the GPU. However, my study shows that performance keeps increasing past this 163k threshold. I do not have a clear explanation for this.

The SoA version shows better performance than the AoS version across all problem sizes. Notably, the SoA version surpasses the roofline calculated for an algorithm at this AI of 0.85. This is probalby due to cache utilization, which is not taken into account in the roofline model. 

For the 1D study, increasing the block size from 8 to 1024 threads. We see that before 32 the performance suffers. Indeed dimensions that are not multiples of 32 lead to underutilization of the warps. For example, for a block size of 8 threads, only 8/32 threads of the warp are active. We also see that even at higher block sizes, performance is better when the block size is a multiple of 32. This avoids underutilization of the warps. Within multiples of 32, there are slight variations in performance. Initial investigations using the Nividia occupancy calculator showed that this difference may be due to occupancy, with powers of two showing the best occupancy and performance.

For the 2D study, no significant performance difference is observed between different block_y values. For example for block_x=64, block_y=1 to 8 show similar performance. Since our implementation does not make use of shared memory, the 2D layout does not influence performance, it is rather the number of threads per block that matters. 



Restrictions to take into consideration on a V100:
- max 1024 threads per block
- max blocks per SM: 32
- max threads per SM: 2048
- max block dim x/y/z: 1024/1024/64
- an SM has 64K registers per SM
- the GPU has 80 SMs

These can be used to calculate occupancy of the SM to optimize block size. Occupancy is the number of max warps launched at once compared to the maximum number of warps it could run, 64 for a V100.


## Objective 

This experiment implements two cuda versions: one with a SoA layout and one with an AoS layout, and compares their performance. This experiment first verifies that the implemented cuda versions work correctly. Performance analyses are then carried out to study the influence of grid/block configuration, problem size and SoA/AoS layout. 

## Status
***COMPLETED**
- **Slurm job name and ID**: multiple jobs, see output directories below

    

## Expected outcomes
- _Deliverables_: verification of the cuda code using Strouhal, performance analysis with estimated roofline, dependence on block/grid configuration, problem size, and SoA/AoS layout.
- _output directory_: 
    - out/cuda_AoS_test for the verification of the AoS version
    - out/cuda_SoA_test for the verification of the SoA version
    - out/cuda_sweep_block_x_SoA for the 1D sweep of block_x for the SoA version
    - out/cuda_sweep_block_x_AoS for the 1D sweep of block_x for the AoS version
    - out/cuda_sweep_2D_SoA for the 2D sweep of block_x and block_y for the SoA version
    - out/cuda_sweep_2D_AoS for the 2D sweep of block_x and block_y for the AoS version
    - out/cuda_sweep_problem_size_SoA for the problem size sweep for the SoA version
    - out/cuda_sweep_problem_size_AoS for the problem size sweep for the AoS version
- _decisions to take_: Final analyses for the GPU phase of the project

## Resources required
one GPU

## Duration
08.06.2026

## Experiment description

cuda version followed the following specifications:
- one-step algorithm with both memory layouts (AoS and SoA): nothing changes between the two codes other than the data layout.
- three kernels: fused streaming and collision, apply inlet, appply outlet
- cx, cy and weights stored in constant memory, solid_ array filled on the CPU then moved to texture memory on the GPU, f_ and f_tmp arrays in global memory. 
- fetch_probe_uxy allows to query velocities at one point without needing to sync the whole grid,  used to write probe.csv.
- For coalesced memory accesses, threads that the same threadIdx.y and consecutive threadIdx.x should access consecutive memory locations. Hence the SoA data layout is suitable. for a given population direction i threads x and x+1 handle consecutive lattice points in the 9 directions. Hence their 9 direction accesses are coalesced. For streaming, consecutive threads will also access consecutive neighbours: for example, for the north direction, thread x accesses f_north at 2N + (y-1)nx + x and thread x+1 accesses f_north at 	2N + (y-1)nx + (x+1), which is also coalesced. This argument assumes convenient block dimensions and an SoA layout.

Both versions are validated by checking the Strouhal number for a 800x400 grid at Reynolds 100 and u_in=0.05 matches the expected value of around 0.16.

For performance analyses:
- 1D analysis: fix problem size at 4096x2048, keep block_y=1 and grid sizes are automatically calculated to cover the whole domain. Vary block_x. Values tried are between 8 and 1024. They include multiples and non-multiples of 32. They also show different values of occupancy. 
- 2D analysis: fix problem size at 4096x2048, vary block_x and block_y in different combinations.
- Problem size analysis: vary problem size from 2048x1024 to 32768x16384, keeping block_x=256 and block_y=1. This allows to see how performance changes with problem size.
- Problem size analysis: vary problem size from 128x64 to 8192x4096, keeping block_x=256 and block_y=1. This allows to see how performance changes with problem size for smaller sizes. Also compare the two versions between each other and with a theoretical roofline calculated for an AI of 0.85 and a memory bandwidth of 900 GB/s found online for the GPU.



### slurm script verification (change for SoA version)

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=math-454
#SBATCH --output=./out/cuda_AoS_test/log_%j.out
#SBATCH --job-name=cuda_AoS

module load gcc hdf5 cuda


NX_VALS=(800)
NY_VALS=(400)
STEPS_VALS=(60000)

# NOTE: #SBATCH --output is parsed by Slurm, not exported as a shell variable.
# Keep the directory in sync with the --output path above.
OUTPUT_DIR=./out/cuda_AoS_test
mkdir -p "${OUTPUT_DIR}"


RESULTS_CSV=${OUTPUT_DIR}/results.csv

# Create the CSV with header only if it does not exist yet.
if [ ! -f "${RESULTS_CSV}" ]; then
    echo "nx,ny,total_cells,u_in,reynolds,replicate,wall_time_s,mlups,strouhal" > ${RESULTS_CSV}
fi

U_IN=0.05
REYNOLDS=100
N_REPLICATES=1
EVERY=0
#export DEBUG_PRINTS=1

for i in "${!NX_VALS[@]}"; do
    NX=${NX_VALS[$i]}
    NY=${NY_VALS[$i]}
    STEPS=${STEPS_VALS[$i]}

    for REP in $(seq 1 ${N_REPLICATES}); do
        # Skip this (NX, NY, replicate) triple if it already has a row in the CSV.
        if awk -F',' -v nx="${NX}" -v ny="${NY}" -v rep="${REP}" \
            'NR>1 && $1==nx && $2==ny && $6==rep {found=1} END {exit !found}' "${RESULTS_CSV}"; then
            echo "Skipping NX=${NX} NY=${NY} REP=${REP} — already in ${RESULTS_CSV}"
            continue
        fi

        SPECIFIC_OUTPUT_DIR=${OUTPUT_DIR}/nx_${NX}_ny_${NY}_rep_${REP}
        RUN_LOG=${SPECIFIC_OUTPUT_DIR}/run.log
        PROBE_PATH=${SPECIFIC_OUTPUT_DIR}/probe.csv
        mkdir -p ${SPECIFIC_OUTPUT_DIR}
        srun ./cuda_AoS/lbm_cuda_AoS nx=${NX} ny=${NY} every=${EVERY} out=${SPECIFIC_OUTPUT_DIR}/lbm probe=${PROBE_PATH} re=${REYNOLDS} u_in=${U_IN} steps=${STEPS} \
            | tee ${RUN_LOG}
        WALL=$(awk '/Wall time/{print $4}' ${RUN_LOG})
        MLUPS=$(awk '/MLUPS/{print $3}'    ${RUN_LOG})

        DIAMETER=$(awk '/diameter/{print $3}' ${RUN_LOG})
        echo "Diameter: ${DIAMETER}"
        ST_LOG=${OUTPUT_DIR}/strouhal_nx_${NX}_ny_${NY}_rep_${REP}.log
        python3 viz/strouhal.py ${PROBE_PATH} --u-in ${U_IN} --diameter ${DIAMETER} \
            | tee ${ST_LOG}
        ST=$(awk '/^St/{print $3}' ${ST_LOG})

        echo "${NX},${NY},$((NX*NY)),${U_IN},${REYNOLDS},${REP},${WALL},${MLUPS},${ST}" >> ${RESULTS_CSV}
    done
done

```

### slurm script 1D sweep (change for AoS version)

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=math-454
#SBATCH --output=./out/cuda_sweep_block_x_SoA/log_%j.out
#SBATCH --job-name=cuda_multiple_sweep
module load gcc hdf5 cuda


NX=4096
NY=2048
STEPS=60000
#BLOCK_X_VALUES=(1 2 4 8 16 32 64 128 256)
#BLOCK_Y_VALUES=(1 1 1 1 1 1 1 1 1)
BLOCK_X_VALUES=(8 16 32 64 96 128 192 256 384 512 768 1024 80 144 272 400 528 784)
#BLOCK_Y_VALUES=(1 1 1 1 1 1 1 1 1 1 1 1)

# NOTE: #SBATCH --output is parsed by Slurm, not exported as a shell variable.
# Keep the directory in sync with the --output path above.
OUTPUT_DIR=./out/cuda_sweep_block_x_SoA
mkdir -p "${OUTPUT_DIR}"


RESULTS_CSV=${OUTPUT_DIR}/results.csv

# Create the CSV with header only if it does not exist yet.
if [ ! -f "${RESULTS_CSV}" ]; then
    echo "nx,ny,total_cells,block_x,block_y,u_in,reynolds,replicate,wall_time_s,mlups,strouhal" > ${RESULTS_CSV}
fi

U_IN=0.05
REYNOLDS=100
N_REPLICATES=3
EVERY=0
#export DEBUG_PRINTS=1

for i in "${!BLOCK_X_VALUES[@]}"; do
    BLOCK_X=${BLOCK_X_VALUES[$i]}
    BLOCK_Y=1

    for REP in $(seq 1 ${N_REPLICATES}); do
        # Skip this (block_x, block_y, replicate) triple if it already has a row in the CSV.
        if awk -F',' -v bx="${BLOCK_X}" -v by="${BLOCK_Y}" -v rep="${REP}" \
            'NR>1 && $4==bx && $5==by && $8==rep {found=1} END {exit !found}' "${RESULTS_CSV}"; then
            echo "Skipping BLOCK_X=${BLOCK_X} BLOCK_Y=${BLOCK_Y} REP=${REP} — already in ${RESULTS_CSV}"
            continue
        fi

        SPECIFIC_OUTPUT_DIR=${OUTPUT_DIR}/nx_${NX}_ny_${NY}_rep_${REP}_block_x_${BLOCK_X}_block_y_${BLOCK_Y}
        RUN_LOG=${SPECIFIC_OUTPUT_DIR}/run.log
        PROBE_PATH=${SPECIFIC_OUTPUT_DIR}/probe.csv
        mkdir -p ${SPECIFIC_OUTPUT_DIR}
        srun ./cuda_SoA/lbm_cuda_SoA nx=${NX} ny=${NY} every=${EVERY} out=${SPECIFIC_OUTPUT_DIR}/lbm probe=${PROBE_PATH} re=${REYNOLDS} u_in=${U_IN} steps=${STEPS} block_x=${BLOCK_X} block_y=${BLOCK_Y} \
            | tee ${RUN_LOG}
        WALL=$(awk '/Wall time/{print $4}' ${RUN_LOG})
        MLUPS=$(awk '/MLUPS/{print $3}'    ${RUN_LOG})

        DIAMETER=$(awk '/diameter/{print $3}' ${RUN_LOG})
        echo "Diameter: ${DIAMETER}"
        ST_LOG=${OUTPUT_DIR}/strouhal_nx_${NX}_ny_${NY}_rep_${REP}.log
        python3 viz/strouhal.py ${PROBE_PATH} --u-in ${U_IN} --diameter ${DIAMETER} \
            | tee ${ST_LOG}
        ST=$(awk '/^St/{print $3}' ${ST_LOG})

        echo "${NX},${NY},$((NX*NY)),${BLOCK_X},${BLOCK_Y},${U_IN},${REYNOLDS},${REP},${WALL},${MLUPS},${ST}" >> ${RESULTS_CSV}
    done
done

```

### slurm script 2D sweep (change for AoS version)

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=math-454
#SBATCH --output=./out/cuda_sweep_2D/log_%j.out
#SBATCH --job-name=cuda_multiple_sweep
module load gcc hdf5 cuda


NX=4096
NY=2048
STEPS=60000
#BLOCK_X_VALUES=(1 2 4 8 16 32 64 128 256)
#BLOCK_Y_VALUES=(1 1 1 1 1 1 1 1 1)
BLOCK_X_VALUES=(64 64 64 64 64 64 32 32 8 16)
BLOCK_Y_VALUES=(1 2 3 4 5 8 8 16 32 32)

# NOTE: #SBATCH --output is parsed by Slurm, not exported as a shell variable.
# Keep the directory in sync with the --output path above.
OUTPUT_DIR=./out/cuda_sweep_2D
mkdir -p "${OUTPUT_DIR}"


RESULTS_CSV=${OUTPUT_DIR}/results.csv

# Create the CSV with header only if it does not exist yet.
if [ ! -f "${RESULTS_CSV}" ]; then
    echo "nx,ny,total_cells,block_x,block_y,u_in,reynolds,replicate,wall_time_s,mlups,strouhal" > ${RESULTS_CSV}
fi

U_IN=0.05
REYNOLDS=100
N_REPLICATES=3
EVERY=0
#export DEBUG_PRINTS=1

for i in "${!BLOCK_X_VALUES[@]}"; do
    BLOCK_X=${BLOCK_X_VALUES[$i]}
    BLOCK_Y=${BLOCK_Y_VALUES[$i]}

    for REP in $(seq 1 ${N_REPLICATES}); do
        # Skip this (block_x, block_y, replicate) triple if it already has a row in the CSV.
        if awk -F',' -v bx="${BLOCK_X}" -v by="${BLOCK_Y}" -v rep="${REP}" \
            'NR>1 && $4==bx && $5==by && $8==rep {found=1} END {exit !found}' "${RESULTS_CSV}"; then
            echo "Skipping BLOCK_X=${BLOCK_X} BLOCK_Y=${BLOCK_Y} REP=${REP} — already in ${RESULTS_CSV}"
            continue
        fi

        SPECIFIC_OUTPUT_DIR=${OUTPUT_DIR}/nx_${NX}_ny_${NY}_rep_${REP}_block_x_${BLOCK_X}_block_y_${BLOCK_Y}
        RUN_LOG=${SPECIFIC_OUTPUT_DIR}/run.log
        PROBE_PATH=${SPECIFIC_OUTPUT_DIR}/probe.csv
        mkdir -p ${SPECIFIC_OUTPUT_DIR}
        srun ./cuda_SoA/lbm_cuda_SoA nx=${NX} ny=${NY} every=${EVERY} out=${SPECIFIC_OUTPUT_DIR}/lbm probe=${PROBE_PATH} re=${REYNOLDS} u_in=${U_IN} steps=${STEPS} block_x=${BLOCK_X} block_y=${BLOCK_Y} \
            | tee ${RUN_LOG}
        WALL=$(awk '/Wall time/{print $4}' ${RUN_LOG})
        MLUPS=$(awk '/MLUPS/{print $3}'    ${RUN_LOG})

        DIAMETER=$(awk '/diameter/{print $3}' ${RUN_LOG})
        echo "Diameter: ${DIAMETER}"
        ST_LOG=${OUTPUT_DIR}/strouhal_nx_${NX}_ny_${NY}_rep_${REP}.log
        python3 viz/strouhal.py ${PROBE_PATH} --u-in ${U_IN} --diameter ${DIAMETER} \
            | tee ${ST_LOG}
        ST=$(awk '/^St/{print $3}' ${ST_LOG})

        echo "${NX},${NY},$((NX*NY)),${BLOCK_X},${BLOCK_Y},${U_IN},${REYNOLDS},${REP},${WALL},${MLUPS},${ST}" >> ${RESULTS_CSV}
    done
done


```

### slurm script problem size sweep (change for AoS version)

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=math-454
#SBATCH --output=./out/cuda_sweep_problem_size/log_%j.out
#SBATCH --job-name=size_sweep
module load gcc hdf5 cuda


NX_VALUES=(128 256 512 1024 2048 4096 8192)
NY_VALUES=(64 128 256 512 1024 2048 4096)
STEPS=60000
BLOCK_X=256
BLOCK_Y=1

# NOTE: #SBATCH --output is parsed by Slurm, not exported as a shell variable.
# Keep the directory in sync with the --output path above.
OUTPUT_DIR=./out/cuda_sweep_problem_size
mkdir -p "${OUTPUT_DIR}"


RESULTS_CSV=${OUTPUT_DIR}/results.csv

# Create the CSV with header only if it does not exist yet.
if [ ! -f "${RESULTS_CSV}" ]; then
    echo "nx,ny,total_cells,block_x,block_y,u_in,reynolds,replicate,wall_time_s,mlups,strouhal" > ${RESULTS_CSV}
fi

U_IN=0.05
REYNOLDS=100
N_REPLICATES=3
EVERY=0
#export DEBUG_PRINTS=1

for i in "${!NX_VALUES[@]}"; do
    NX=${NX_VALUES[$i]}
    NY=${NY_VALUES[$i]}

    for REP in $(seq 1 ${N_REPLICATES}); do
        # Skip this (nx, ny, block_x, block_y, replicate) combination if it already has a row in the CSV.
        if awk -F',' -v nx="${NX}" -v ny="${NY}" -v bx="${BLOCK_X}" -v by="${BLOCK_Y}" -v rep="${REP}" \
            'NR>1 && $1==nx && $2==ny && $4==bx && $5==by && $8==rep {found=1} END {exit !found}' "${RESULTS_CSV}"; then
            echo "Skipping NX=${NX} NY=${NY} BLOCK_X=${BLOCK_X} BLOCK_Y=${BLOCK_Y} REP=${REP} — already in ${RESULTS_CSV}"
            continue
        fi

        SPECIFIC_OUTPUT_DIR=${OUTPUT_DIR}/nx_${NX}_ny_${NY}_rep_${REP}_block_x_${BLOCK_X}_block_y_${BLOCK_Y}
        RUN_LOG=${SPECIFIC_OUTPUT_DIR}/run.log
        PROBE_PATH=${SPECIFIC_OUTPUT_DIR}/probe.csv
        mkdir -p ${SPECIFIC_OUTPUT_DIR}
        srun ./cuda_SoA/lbm_cuda_SoA nx=${NX} ny=${NY} every=${EVERY} out=${SPECIFIC_OUTPUT_DIR}/lbm probe=${PROBE_PATH} re=${REYNOLDS} u_in=${U_IN} steps=${STEPS} block_x=${BLOCK_X} block_y=${BLOCK_Y} \
            | tee ${RUN_LOG}
        WALL=$(awk '/Wall time/{print $4}' ${RUN_LOG})
        MLUPS=$(awk '/MLUPS/{print $3}'    ${RUN_LOG})

        DIAMETER=$(awk '/diameter/{print $3}' ${RUN_LOG})
        echo "Diameter: ${DIAMETER}"
        ST_LOG=${OUTPUT_DIR}/strouhal_nx_${NX}_ny_${NY}_rep_${REP}.log
        python3 viz/strouhal.py ${PROBE_PATH} --u-in ${U_IN} --diameter ${DIAMETER} \
            | tee ${ST_LOG}
        ST=$(awk '/^St/{print $3}' ${ST_LOG})

        echo "${NX},${NY},$((NX*NY)),${BLOCK_X},${BLOCK_Y},${U_IN},${REYNOLDS},${REP},${WALL},${MLUPS},${ST}" >> ${RESULTS_CSV}
    done
done

```