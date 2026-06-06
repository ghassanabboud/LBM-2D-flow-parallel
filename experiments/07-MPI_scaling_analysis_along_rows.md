# Experiment 07: MPI scaling analysis
#### **Code version:** reorganize scaling figures (e71eacb86ee2a1dfafdc67b0a23cb4d0c83a2b1b)
## Results and Next Steps

Strong scaling shows a linear scaleup until more or less 8 ranks then performance plateaus lower than what is predicted by Amdahl's law. This shows communication overhead: at 64 cores there are only 14 rows to compute so halo exchange becomes substantial. Weak scaling shows good scaling with efficiency over 90% until 64 ranks. There is a peak at 16 ranks. One theory is that each node has 18 cores so that using 16 cores saturates the memory bandwidth of a single node, while using 32 cores causes contention between the two nodes. I added 18 and 19 ranks in the weak scaling to test this theory, 19 ranks does not show a drop in performance but a gradual decrease, so the theory is not confirmed. 

For next steps, I will create a version of the MPI code with non-blocking communication to try to overlap communication and computation and see if it improves the strong scaling. 

LIMITATION: we only scaled the number of rows not the number of columns I want to test whether scaling both dimensions in the same ratio would lead to the same performance results or not. I would expect not because each rank would get very long rows that cannot fit in cache and would lead to more cache misses, but I want to test it.

## Objective 

The MPI code was fixed and validated in [the previous experiment](./06-fix_top_bottom_treatment.md). The next step is to run a strong and weak scaling analysis of the MPI version.


## Status
**In Progress**
- **Slurm job name and ID**: 
    - **strong scaling**: all jobs called strong_scaling_mpi, job ID 54936833, and 54937077-54937081, 54937278
    - **weak scaling**: all jobs called weak_scaling_mpi, job ID 54937101-54937106, 54937337, 54947667, 54947668 


## Expected outcomes

- _Deliverables_: results.csv with wall time, MLUPS to be turned into plots showing the strong and weak scaling of the MPI version.
- _output directory_:  `./out/strong_scaling_mpi` and `./out/weak_scaling_mpi`, `perf_evaluation_SoA` for the perf evaluation of the fused_SoA version to determine alpha. figures in `./figures/scalings_mpi`.
- _decisions to take_: Analysis of scaling to be included in the report. Decide whether it follows the appropriate scaling laws.

## Resources required

up to 32 CPUS.

## Duration

05.06.2026..

## Experiment description

Two experiments will be run:
- strong scaling: domain of 1600x800 will be run for 60000 steps with 1, 2, 4, 8, 16, and 32 ranks. I decided to run 64 ranks as well so I had to change the domain to 1429x896 to be divisible by 64 while conserving the same total number of cells. 
- weak scaling: domain of 400x200 will be run for 60000 steps with 1 rank. From there, The y dimension is multiplied by 2 while doubling the number of ranks, until 64 ranks and a domain of 400x12800. I then decided to try 18,19 ranks to resolve the superlinear scaling I first observed at 16 ranks.

the alpha parameter for strong scaling was calculated as proportion of collide_stream runtime in the fused_SoA version, determined using perf.

### script: finding alpha for strong scaling

```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --time 1:0:0
#SBATCH --output=./out/perf_evaluation_SoA/log_O3_%j.out
#SBATCH --account=math-454
#SBATCH --partition=academic

module load gcc hdf5

OUT_DIR=out/perf_evaluation_SoA
mkdir -p ${OUT_DIR}

# -F 99: 99 Hz sampling avoids lockstep with timer interrupts.
# No --call-graph: flat profile only, keeps perf.data to a few MB.
srun perf record \
    -o ${OUT_DIR}/perf_detailed_O3.data \
    ./serial_fused_SoA/lbm_serial_fused_SoA nx=800 ny=400 every=0 out=${OUT_DIR}/lbm_perf

# Flat profile with demangled C++ names.
perf report --stdio --demangle -i ${OUT_DIR}/perf_detailed_O3.data

#done
```

### script: strong scaling


```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --time 16:0:0
#SBATCH --output=./out/strong_scaling_mpi/log_%j.out
#SBATCH --account=math-454
#SBATCH --partition=academic
#SBATCH --job-name=strong_scaling_mpi

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
STEPS=60000
NRANKS=${SLURM_NTASKS}

if (( NY % NRANKS != 0 )); then
    echo "Error: ny=${NY} is not divisible by NRANKS=${NRANKS}. Aborting."
    exit 1
fi

# NOTE: #SBATCH --output is parsed by Slurm, not exported as a shell variable.
# Keep the directory in sync with the --output path above.
OUTPUT_DIR=./out/strong_scaling_mpi
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
    srun ./mpi/lbm_mpi nx=${NX} ny=${NY} every=${EVERY} \
        out=${SPECIFIC_OUTPUT_DIR}/lbm probe=${PROBE_PATH} re=${REYNOLDS} u_in=${U_IN} steps=${STEPS} \
        | tee ${RUN_LOG}
    WALL=$(awk '/Wall time/{print $4}' ${RUN_LOG})
    MLUPS=$(awk '/MLUPS/{print $3}'    ${RUN_LOG})

    echo "${NX},${NY},${NRANKS},$((NX*NY)),${U_IN},${REYNOLDS},${REP},${WALL},${MLUPS}" >> ${RESULTS_CSV}
done

```

### script: weak scaling

```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --time 16:0:0
#SBATCH --output=./out/weak_scaling_mpi/log_%j.out
#SBATCH --account=math-454
#SBATCH --partition=academic
#SBATCH --job-name=weak_scaling_mpi

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
OUTPUT_DIR=./out/weak_scaling_mpi
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
    srun ./mpi/lbm_mpi nx=${NX} ny=${NY} every=${EVERY} \
        out=${SPECIFIC_OUTPUT_DIR}/lbm probe=${PROBE_PATH} re=${REYNOLDS} u_in=${U_IN} steps=${STEPS} \
        | tee ${RUN_LOG}
    WALL=$(awk '/Wall time/{print $4}' ${RUN_LOG})
    MLUPS=$(awk '/MLUPS/{print $3}'    ${RUN_LOG})

    echo "${NX},${NY},${NRANKS},$((NX*NY)),${U_IN},${REYNOLDS},${REP},${WALL},${MLUPS}" >> ${RESULTS_CSV}
done

```


## Links and references
TO-DO: list here publications, web pages, etc. that contain information relevant to the experiment. 