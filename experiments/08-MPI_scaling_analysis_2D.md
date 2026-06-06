# Experiment 08: MPI scaling analysis along two dimensions
#### **Code version:** weak scaling along 2D (30eb115bda09dd57b47c42e98bdfb116ed1faa28)
## Results and Next Steps

Weak scaling along two dimensions shows poor performance, the effciency stays at 1 for 4 ranks but drops below 0.4 at 64 ranks. 

Explanation: we are scaling the grid in both dimensions, when p quadruples, we get two times more total rows and two times more total columns. However, our version only splits the number of rows. Hence, every time p quadruples, each rank gets two times less rows but two times more columns. Hence, the rows get longer and longer. The main mechanism is not cache misses but communication-computation ratio. When the rows get longer, the amount of communication per time step increases until it takes over.

Another way to see it:
- work = local rows * local cols = local_ny * nx
- comm = local cols = nx
- comm/work = nx / (local_ny * nx) = 1/local_ny

As local_ny gets smaller, the communication to work ratio gets worse. local_ny is fixed when we scale only along one dimension, but it gets smaller and smaller when we scale along both dimensions.  

next step: implement an MPI cart version that splits both dimensions and see if it improves the weak scaling along 2D.

## Objective 

Strong and weak scaling analysis of the MPI version are good but I wondered whether weak scaling would be similar if we scaled both dimensions instead of just the number of rows. This is closer to reality where we would want to refine more in both dimensions. My hypothesis is that scaling both dimensions would lead to worse performance because each rank would get very long rows that cannot fit in cache and would lead to more cache misses.


## Status
**Completed**
- **Slurm job name and ID**: all jobs called 2D_weak_scaling_mpi, job ID 54949460, 54949462, 54949463, 54949464


## Expected outcomes

- _Deliverables_: results.csv with wall time, MLUPS to be turned into plots showing the weak scaling of the MPI version along 2D.
- _output directory_:  `out/weak_scaling_mpi_2D` and figure in `./figures/scalings_mpi_row_wise`.
- _decisions to take_: Analysis of 2D based weak scaling to be included in the report. Decide whether it calls for an MPI cart implementation.

## Resources required

up to 32 CPUS.

## Duration

06.06.2026

## Experiment description

Weak scaling: domain of 400x200 will be run for 60000 steps with 1 rank. From there, both dimensions are multiplied by 2 while quadrupling the number of ranks:
- 1 rank: 400x200
- 4 ranks: 800x400
- 16 ranks: 1600x800
- 64 ranks: 3200x1600


### script: 2D weak scaling

```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --time 16:0:0
#SBATCH --output=./out/weak_scaling_mpi_2D/log_%j.out
#SBATCH --account=math-454
#SBATCH --partition=academic
#SBATCH --job-name=2D_weak_scaling_mpi

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
OUTPUT_DIR=./out/weak_scaling_mpi_2D
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