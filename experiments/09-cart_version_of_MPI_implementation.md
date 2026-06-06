# Experiment 09: Cart version of MPI implementation for better weak scaling along 2D 
 #### **Code version:** all mpi scaling plots (86d474d229e147fc6407e3ff7a7c35c26d64fe95)

## Results and Next Steps

It was verified that the cart version produces correct Strouhal using 4 or 8 CPUs. the MPI Cart version provides impressive strong scaling with a speedup matching Amdahl's law very closely up to 64 ranks. The weak scaling results are very interesting: the cart version performs worse than the row-wise version when scaling along rows only, but performs much better when scaling along both dimensions. This confirms the hypothesis that the row-wise version suffers from high communication overhead when scaling along 2D. Could be that for x fixed at 400, it is always better to split along rows than columns for cache locality reasons and to reduce communication volume. 

## Objective 

[Experiment 08](./08-MPI_scaling_analysis_2D.md) showed that the weak scaling of the MPI version along 2D is poor because the rows get longer and longer as we scale the grid in both dimensions, which increases the communication/computation ratio. In this experiment, I will implement an MPI cart version that splits both dimensions using MPI_Cart routines and see if it improves the weak scaling along 2D.

## Status
***In Progress**
- **Slurm job name and ID**:
    - verification on 800x400 using 8 CPUs: cart_multiple_fix_wall_bigger job ID 54950141
    - verification on 800x400 using 4 CPUs: cart_multiple_fix_wall_make job ID 54950136
    - weak scaling analysis 2D: cart_2D_weak_scaling_mpi job IDs 54950166-54950170
    - weak scaling analysis along rows: cart_weak_scaling_mpi_2D job IDs 54950179-54950185
    - strong scaling analysis: cart_strong_scaling_mpi job IDs 54950196-54950201, 54950211
    

## Expected outcomes
- _Deliverables_: verification of the cart version of the MPI code using Strouhal, strong and weak scaling analysis of the cart version, comparison of the results with the previous version.
- _output directory_: `./out/cart_multiple_fix_wall_bigger` for the 8 CPU verification, `./out/cart_multiple_fix_wall_make` for the 4 CPU verification. `out/cart_weak_scaling_mpi_2D` for the weak scaling analysis along 2D, `out/cart_weak_scaling_mpi` for the weak scaling analysis along rows, `out/cart_strong_scaling_mpi` for the strong scaling analysis. figures in `./figures/scalings_mpi_cart` and comparisons with the original implementation `.figures/scalings_comparison`.
- _decisions to take_: If successful, this would be the last MPI implementation before the GPU phase.

## Resources required
up to 64 CPUs for scaling analyses.

## Duration
06.06.2026

## Experiment description

For code changes, `MPI_Cart` routines are used to split the grid along both dimensions and create a Cartesian topology. `MPI_Cart_coords` help get the coordinates of each process in the Cartesian grid and `MPI_Cart_shift` is used to determine the ranks of the neighboring processes for communication. Communication now happend along both columns and rows. 

Communication uses a 2-wave approach: first, processes exchange cell data only along rows of size `local_nx_` (not including ghost data) and then along full_columns of size `local_ny_ + 2` (including ghost data). This means that the rows copied in the first step are then communicated in the second step so that corners are communicated between diagonal neighbours. 

Verification is done by checking that the Strouhal number for a 800x400 grid at Reynolds 100 and u_in=0.05 matches the expected value of around 0.16, on both 4 and 8 CPUs. Weak scaling analysis is done in 2 ways: first, by keeping the per-rank workload constant (400x200) and scaling only along rows, and second, by scaling both dimensions so that the per-rank workload is constant (400x200 for 1 rank, 800x400 for 4 ranks, 1600x800 for 16 ranks, and 3200x1600 for 64 ranks).

Strong scaling analysis is done by running the same 1600x800 case on 1,2,4,8,16,32, and 64 CPUs and comparing the wall time and MLUPS.


### slurm script verification (change number of CPUs as needed)

```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-task 1
#SBATCH --time 16:0:0
#SBATCH --output=./out/cart_multiple_fix_wall_bigger/log_%j.out
#SBATCH --account=math-454
#SBATCH --partition=academic
#SBATCH --job-name=cart_multiple_fix_wall_bigger

module load gcc hdf5 openmpi

# Each step doubles total cell count (nx*ny) by scaling both dims by sqrt(2).
# Total points: 20K → 40K → 80K → 160K → 320K -> 640K → 1.28M
#NX_VALS=(200 283 400 566 800 1131 1600 2263 3200)
#NY_VALS=(100 141 200 283 400 566 800 1131 1600)
#STEPS_VALS=(60000 60000 60000 60000 60000 60000 60000 60000 60000)

NX_VALS=(800)
NY_VALS=(400)
STEPS_VALS=(60000)

# NOTE: #SBATCH --output is parsed by Slurm, not exported as a shell variable.
# Keep the directory in sync with the --output path above.
OUTPUT_DIR=./out/cart_multiple_fix_wall_bigger
mkdir -p "${OUTPUT_DIR}"


RESULTS_CSV=${OUTPUT_DIR}/results.csv

# Create the CSV with header only if it does not exist yet.
if [ ! -f "${RESULTS_CSV}" ]; then
    echo "nx,ny,total_cells,u_in,reynolds,replicate,wall_time_s,mlups,strouhal" > ${RESULTS_CSV}
fi

U_IN=0.05
REYNOLDS=100
N_REPLICATES=1
EVERY=500
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
        srun ./mpi_cart/lbm_mpi_cart nx=${NX} ny=${NY} every=${EVERY} out=${SPECIFIC_OUTPUT_DIR}/lbm probe=${PROBE_PATH} re=${REYNOLDS} u_in=${U_IN} steps=${STEPS} \
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

### slurm script for weak scaling along rows

```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --time 16:0:0
#SBATCH --output=./out/cart_weak_scaling_mpi/log_%j.out
#SBATCH --account=math-454
#SBATCH --partition=academic
#SBATCH --job-name=cart_weak_scaling_mpi

# Submit once per rank count, requesting exactly the cores needed:
#   sbatch --ntasks=1  weak_scaling_mpi.slurm
#   sbatch --ntasks=2  weak_scaling_mpi.slurm
#   sbatch --ntasks=4  weak_scaling_mpi.slurm
#   sbatch --ntasks=8  weak_scaling_mpi.slurm
#   sbatch --ntasks=16 weak_scaling_mpi.slurm
#   sbatch --ntasks=32 weak_scaling_mpi.slurm
#   sbatch --ntasks=64 weak_scaling_mpi.slurm
# Base domain: 400x200 with 1 rank. ny doubles with each rank doubling so each
# rank always owns a 400x200 slab (constant per-rank workload).
# All jobs append to the same results.csv in OUTPUT_DIR.

module load gcc hdf5 openmpi

NX=400
BASE_NY=200
STEPS=60000
NRANKS=${SLURM_NTASKS}
NY=$((BASE_NY * NRANKS))

# NOTE: #SBATCH --output is parsed by Slurm, not exported as a shell variable.
# Keep the directory in sync with the --output path above.
OUTPUT_DIR=./out/cart_weak_scaling_mpi
mkdir -p "${OUTPUT_DIR}"

RESULTS_CSV=${OUTPUT_DIR}/results.csv
if [ ! -f "${RESULTS_CSV}" ]; then
    echo "nx,ny,nranks,total_cells,u_in,reynolds,replicate,wall_time_s,mlups" > ${RESULTS_CSV}
fi

U_IN=0.05
REYNOLDS=100
N_REPLICATES=3
EVERY=0

for REP in $(seq 1 ${N_REPLICATES}); do
    if awk -F',' -v nx="${NX}" -v ny="${NY}" -v nr="${NRANKS}" -v rep="${REP}" \
        'NR>1 && $1==nx && $2==ny && $3==nr && $7==rep {found=1} END {exit !found}' "${RESULTS_CSV}"; then
        echo "Skipping NX=${NX} NY=${NY} NRANKS=${NRANKS} REP=${REP} — already in ${RESULTS_CSV}"
        continue
    fi

    SPECIFIC_OUTPUT_DIR=${OUTPUT_DIR}/nx_${NX}_ny_${NY}_nranks_${NRANKS}_rep_${REP}
    RUN_LOG=${SPECIFIC_OUTPUT_DIR}/run.log
    PROBE_PATH=${SPECIFIC_OUTPUT_DIR}/probe.csv
    mkdir -p ${SPECIFIC_OUTPUT_DIR}
    srun ./mpi_cart/lbm_mpi_cart nx=${NX} ny=${NY} every=${EVERY} \
        out=${SPECIFIC_OUTPUT_DIR}/lbm probe=${PROBE_PATH} re=${REYNOLDS} u_in=${U_IN} steps=${STEPS} \
        | tee ${RUN_LOG}
    WALL=$(awk '/Wall time/{print $4}' ${RUN_LOG})
    MLUPS=$(awk '/MLUPS/{print $3}'    ${RUN_LOG})

    echo "${NX},${NY},${NRANKS},$((NX*NY)),${U_IN},${REYNOLDS},${REP},${WALL},${MLUPS}" >> ${RESULTS_CSV}
done

```

### slurm script for weak scaling in 2D

```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --time 16:0:0
#SBATCH --output=./out/cart_weak_scaling_mpi_2D/log_%j.out
#SBATCH --account=math-454
#SBATCH --partition=academic
#SBATCH --job-name=cart_2D_weak_scaling_mpi

# Submit once per rank count, requesting exactly the cores needed:
#   sbatch --ntasks=1  weak_scaling_mpi_2D.slurm
#   sbatch --ntasks=4  weak_scaling_mpi_2D.slurm
#   sbatch --ntasks=16 weak_scaling_mpi_2D.slurm
#   sbatch --ntasks=64 weak_scaling_mpi_2D.slurm
# Base domain: 400x200 with 1 rank. ny and nx double with each rank quadrupling so each
# All jobs append to the same results.csv in OUTPUT_DIR.

module load gcc hdf5 openmpi

STEPS=60000
NRANKS=${SLURM_NTASKS}

# Both dimensions double each time ranks quadruple: 1→4→16→64 ranks maps to
# 400x200 → 800x400 → 1600x800 → 3200x1600 (per-rank workload constant).
case ${NRANKS} in
    1)  NX=400;  NY=200  ;;
    4)  NX=800;  NY=400  ;;
    16) NX=1600; NY=800  ;;
    64) NX=3200; NY=1600 ;;
    *)  echo "Error: unsupported NRANKS=${NRANKS}. Use 1, 4, 16, or 64."; exit 1 ;;
esac

# NOTE: #SBATCH --output is parsed by Slurm, not exported as a shell variable.
# Keep the directory in sync with the --output path above.
OUTPUT_DIR=./out/cart_weak_scaling_mpi_2D
mkdir -p "${OUTPUT_DIR}"

RESULTS_CSV=${OUTPUT_DIR}/results.csv
if [ ! -f "${RESULTS_CSV}" ]; then
    echo "nx,ny,nranks,total_cells,u_in,reynolds,replicate,wall_time_s,mlups" > ${RESULTS_CSV}
fi

U_IN=0.05
REYNOLDS=100
N_REPLICATES=3
EVERY=0

for REP in $(seq 1 ${N_REPLICATES}); do
    if awk -F',' -v nx="${NX}" -v ny="${NY}" -v nr="${NRANKS}" -v rep="${REP}" \
        'NR>1 && $1==nx && $2==ny && $3==nr && $7==rep {found=1} END {exit !found}' "${RESULTS_CSV}"; then
        echo "Skipping NX=${NX} NY=${NY} NRANKS=${NRANKS} REP=${REP} — already in ${RESULTS_CSV}"
        continue
    fi

    SPECIFIC_OUTPUT_DIR=${OUTPUT_DIR}/nx_${NX}_ny_${NY}_nranks_${NRANKS}_rep_${REP}
    RUN_LOG=${SPECIFIC_OUTPUT_DIR}/run.log
    PROBE_PATH=${SPECIFIC_OUTPUT_DIR}/probe.csv
    mkdir -p ${SPECIFIC_OUTPUT_DIR}
    srun ./mpi_cart/lbm_mpi_cart nx=${NX} ny=${NY} every=${EVERY} \
        out=${SPECIFIC_OUTPUT_DIR}/lbm probe=${PROBE_PATH} re=${REYNOLDS} u_in=${U_IN} steps=${STEPS} \
        | tee ${RUN_LOG}
    WALL=$(awk '/Wall time/{print $4}' ${RUN_LOG})
    MLUPS=$(awk '/MLUPS/{print $3}'    ${RUN_LOG})

    echo "${NX},${NY},${NRANKS},$((NX*NY)),${U_IN},${REYNOLDS},${REP},${WALL},${MLUPS}" >> ${RESULTS_CSV}
done
```

### strong scaling slurm script

```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --time 16:0:0
#SBATCH --output=./out/cart_strong_scaling_mpi/log_%j.out
#SBATCH --account=math-454
#SBATCH --partition=academic
#SBATCH --job-name=cart_strong_scaling_mpi

# Submit once per rank count, requesting exactly the cores needed to avoid oversubscription.:
#   sbatch --ntasks=1  strong_scaling_mpi.slurm
#   sbatch --ntasks=2  strong_scaling_mpi.slurm
#   sbatch --ntasks=4  strong_scaling_mpi.slurm
#   sbatch --ntasks=8  strong_scaling_mpi.slurm
#   sbatch --ntasks=16 strong_scaling_mpi.slurm
#   sbatch --ntasks=32 strong_scaling_mpi.slurm
# All jobs append to the same results.csv in OUTPUT_DIR.

module load gcc hdf5 openmpi

NX=1600
NY=800
#NX=1429
#NY=896
STEPS=60000
NRANKS=${SLURM_NTASKS}

if (( NY % NRANKS != 0 )); then
    echo "Error: ny=${NY} is not divisible by NRANKS=${NRANKS}. Aborting."
    exit 1
fi

# NOTE: #SBATCH --output is parsed by Slurm, not exported as a shell variable.
# Keep the directory in sync with the --output path above.
OUTPUT_DIR=./out/cart_strong_scaling_mpi
mkdir -p "${OUTPUT_DIR}"

RESULTS_CSV=${OUTPUT_DIR}/results.csv
if [ ! -f "${RESULTS_CSV}" ]; then
    echo "nx,ny,nranks,total_cells,u_in,reynolds,replicate,wall_time_s,mlups" > ${RESULTS_CSV}
fi

U_IN=0.05
REYNOLDS=100
N_REPLICATES=3
EVERY=0

for REP in $(seq 1 ${N_REPLICATES}); do
    if awk -F',' -v nx="${NX}" -v ny="${NY}" -v nr="${NRANKS}" -v rep="${REP}" \
        'NR>1 && $1==nx && $2==ny && $3==nr && $7==rep {found=1} END {exit !found}' "${RESULTS_CSV}"; then
        echo "Skipping NX=${NX} NY=${NY} NRANKS=${NRANKS} REP=${REP} — already in ${RESULTS_CSV}"
        continue
    fi

    SPECIFIC_OUTPUT_DIR=${OUTPUT_DIR}/nx_${NX}_ny_${NY}_nranks_${NRANKS}_rep_${REP}
    RUN_LOG=${SPECIFIC_OUTPUT_DIR}/run.log
    PROBE_PATH=${SPECIFIC_OUTPUT_DIR}/probe.csv
    mkdir -p ${SPECIFIC_OUTPUT_DIR}
    srun ./mpi_cart/lbm_mpi_cart nx=${NX} ny=${NY} every=${EVERY} \
        out=${SPECIFIC_OUTPUT_DIR}/lbm probe=${PROBE_PATH} re=${REYNOLDS} u_in=${U_IN} steps=${STEPS} \
        | tee ${RUN_LOG}
    WALL=$(awk '/Wall time/{print $4}' ${RUN_LOG})
    MLUPS=$(awk '/MLUPS/{print $3}'    ${RUN_LOG})

    echo "${NX},${NY},${NRANKS},$((NX*NY)),${U_IN},${REYNOLDS},${REP},${WALL},${MLUPS}" >> ${RESULTS_CSV}
done

```
## Links and references
TO-DO: list here publications, web pages, etc. that contain information relevant to the experiment. 