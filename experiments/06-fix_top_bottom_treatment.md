# Experiment 06: fix top and bottom treatment in MPI version
#### **Code version:** fix by adding global_y check to MPI version(b4a91a89eff1162eadc08acc5a8482bdc6b4261f)
## Results and Next Steps

First, the fix showed that there was no explosion on a 200x100 grid for 300 steps. Then, running the 800x400 grid for 60000 steps with one rank showed the correct Strouhal number of 16.00. Finally, running the same 800x400 grid for 60000 steps with 4 ranks showed that the Strouhal number is still correct at 16.00. THe MLUPS for the 4 ranks version is 70.5654, showing an almost x4 speedup compared to the 1 rank version.

THE MPI CODE NOW WORKS CORRECTLY.

## Objective 
[experiment 05](./05-debug_prints_mpi_explosion.md) showed that running the MPI version even with one rank caused the explosion and that it is due to incorrect streaming of the N, NE, and NW populations at the bottom wall and S, SE, and SW populations at the top wall. These populations were getting 0 from ghost rows outside the domain instead of staying as they are. The objective of this experiment is to fix this issue and verify that the MPI version now works. It should be first tested with one rank then with multiple ranks to identify any other bugs arousing from ghost handling in the middle of the domain.


## Status
**COMPLETED**
- **Slurm job name and ID**: 
    - **check no explosion on one rank**: globaly_fix job ID 54936042
    - **check correct strouhal on one rank**: strouhal_globaly_fix job ID 54936057
    - **check correct strouhal on multiple ranks**: multi_strouhal_globaly_fix job ID 54936413


## Expected outcomes

- _Deliverables_: log prints to check the explosion, results.csv with Strouhal number
- _output directory_:  
    - check no explosion on one rank: ./out/test_globaly_fix
    - check correct strouhal on one rank: ./out/strouhal_globaly_fix
    - check correct strouhal on multiple ranks: ./out/multi_strouhal_globaly_fix
- _decisions to take_: check if the fix works and whether there are any other bugs in the multiple ranks case.

## Resources required

1 CPU for the one rank version and 4 CPUs for the multiple ranks version.

## Duration

05.06.2026

## Experiment description

In this experiment, I will first fix the bug by re-adding the y checks that were removed in the MPI version. the y check should be a check on global y that only activates on the top and bottom solid wall to avoid streaming from ghost rows. Then, to verify the fix, 
- I rerun 300 steps of the MPI version to see if there is an explosion. 
- If not, I can run the usual 800x400 grid for 60000 steps and validate using Strouhal results.
- If that version works as well, I can run the multiple ranks version to check for any other bugs.

### script: check no explosion with one rank


```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --time 16:0:0
#SBATCH --output=./out/test_globaly_fix/log_%j.out
#SBATCH --account=math-454
#SBATCH --partition=academic
#SBATCH --job-name=globaly_fix

module load gcc hdf5 openmpi

# Each step doubles total cell count (nx*ny) by scaling both dims by sqrt(2).
# Total points: 20K → 40K → 80K → 160K → 320K -> 640K → 1.28M
#NX_VALS=(200 283 400 566 800 1131 1600 2263 3200)
#NY_VALS=(100 141 200 283 400 566 800 1131 1600)
#STEPS_VALS=(60000 60000 60000 60000 60000 60000 60000 60000 60000)

NX_VALS=(200)
NY_VALS=(100)
STEPS_VALS=(300)

# NOTE: #SBATCH --output is parsed by Slurm, not exported as a shell variable.
# Keep the directory in sync with the --output path above.
OUTPUT_DIR=./out/test_globaly_fix
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
export DEBUG_PRINTS=1

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

### script: check Strouhal with one rank

```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --time 16:0:0
#SBATCH --output=./out/strouhal_globaly_fix/log_%j.out
#SBATCH --account=math-454
#SBATCH --partition=academic
#SBATCH --job-name=strouhal_globaly_fix

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
OUTPUT_DIR=./out/strouhal_globaly_fix
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

### script: check Strouhal with multiple ranks

```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 1
#SBATCH --time 16:0:0
#SBATCH --output=./out/multi_strouhal_globaly_fix/log_%j.out
#SBATCH --account=math-454
#SBATCH --partition=academic
#SBATCH --job-name=multi_strouhal_globaly_fix

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
OUTPUT_DIR=./out/multi_strouhal_globaly_fix
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