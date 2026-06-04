# Experiment 04: Try fixing MPI explosion by fixing solid ghost rows
#### **Code version:** fix solid ghost rows (4052909e03a90016a1864eda28bdc84518c6d140)
## Results and Next Steps

Results clearly show that the explosion is not fixed. The ghost cell stays at 0 throughout as it should but the bottom row of rank 0 still explodes.Future investigations will focus on checking the code step by step and checking if the new handling of the streaming conditions is correct.

## Objective 

small runs have shown that the MPI code contains a bug that makes all of the population values explode positively or negatively until they overflow and nan out. A first theory related to the misshandling og the solid array for ghost rows. The solid array was never updated for ghost rows, leading to fluid cells on a rank's boundary incorrectly streaming from a solid cell. The goal of this experiment was to determine whether this was the cause of the explosion.


## Status
**COMPLETED**
- **Slurm job name and ID**: first_test_mpi ID 54911984


## Expected outcomes

- _Deliverables_: log printed the values of populations on the bottom boundaries of rank 0
- _output directory_:  `./out/test_solid_fix_short_mpi_run`
- _decisions to take_: decide whether MPI code is now correct or if I need to investigate other theories.

## Resources required

4 CPUs for MPI comm

## Duration

03.06.2026

## Experiment description

800x400 grid, 1000 steps which is enough to see the explosion. Debug prints were added to visualize the population distributions on the bottom row of rank 0 and its ghost row. 

###script

`main_mpi.cpp` was modified to add debug prints of the populations on the bottom row of rank 0 and its ghost row every 10 steps. The script below was used to run the experiment.

```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 1
#SBATCH --time 16:0:0
#SBATCH --output=./out/test_solid_fix_short_mpi_run/log_%j.out
#SBATCH --account=math-454
#SBATCH --partition=academic

module load gcc hdf5 openmpi

# Each step doubles total cell count (nx*ny) by scaling both dims by sqrt(2).
# Total points: 20K → 40K → 80K → 160K → 320K -> 640K → 1.28M
#NX_VALS=(200 283 400 566 800 1131 1600 2263 3200)
#NY_VALS=(100 141 200 283 400 566 800 1131 1600)
#STEPS_VALS=(60000 60000 60000 60000 60000 60000 60000 60000 60000)

NX_VALS=(800)
NY_VALS=(400)
STEPS_VALS=(1000)

# NOTE: #SBATCH --output is parsed by Slurm, not exported as a shell variable.
# Keep the directory in sync with the --output path above.
OUTPUT_DIR=./out/test_solid_fix_short_mpi_run
mkdir -p "${OUTPUT_DIR}"


RESULTS_CSV=${OUTPUT_DIR}/results.csv

# Create the CSV with header only if it does not exist yet.
if [ ! -f "${RESULTS_CSV}" ]; then
    echo "nx,ny,total_cells,u_in,reynolds,replicate,wall_time_s,mlups,strouhal" > ${RESULTS_CSV}
fi

U_IN=0.05
REYNOLDS=100
N_REPLICATES=1
EVERY=50

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
        srun ./mpi/lbm_mpi nx=${NX} ny=${NY} every=${EVERY} out=${SPECIFIC_OUTPUT_DIR}/lbm probe=${PROBE_PATH} re=${REYNOLDS} u_in=${U_IN} steps=${STEPS} \
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



## Links and references
TO-DO: list here publications, web pages, etc. that contain information relevant to the experiment. 