# Experiment 01: Replicate SoA and AoS runtimes for serial one-step implementations
#### **Code version:** initial commit again (f52d02dd5ed596f6c8480d1b9391adb9a285f03d)
## Results and Next Steps
TO-DO: record here the partial or final conclusions and results of the technical experiment. 

## Objective 
Independent runs have shown variations in the runtime and MLUPS performance of the serial codes. The report's `Single-CPU performance` section compares runtimes for SoA and AoS and tries to interpret them so the results should be reproducible. The objective of this experiment is to get values of performance for the three serial implementations (SoA, AoS and original) that is consistent across replicates for reproducibility in reporting.

If the values are too inconsistent across replicates, a larger number of relicates will allow to report performance on one grid with better statistics.

## Status
**Running**
- **Slurm job name and ID**:
    - SoA replicates: `rep_SoA`, job ID 54908551
    - AoS replicates: `rep_AoS`, job ID 54908556
    - Two-step original algorithm replicates: `rep_twostep`, job ID 54908549


## Expected outcomes

- _Deliverables_: results.csv files for the three versions
- _output directory_: a directory containing the output of the Slurm job, including logs and performance metrics.
- _decisions to take_: decide whether to keep results and change values in the report or to run more replicates on the 800x400 grid overnight.


## Resources required

3 CPUs

## Duration

03.06.2026: analyze results and decide whether to keep results and change values in the report or to run more replicates on the 800x400 grid overnight.


## Experiment description

Focus only on the grid sizes for which performance will be reported and against which Strouhal numbers can be checked as validation. I use three grid sizes 566x283, 800x400, and 1131x566, with the middle one used for Strouhal validation. Three replicates are performed for each grid size and no intermediate outputs are saved (every=0) to assess performance without the overhead of writing outputs. 

### SoA slurm script

```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --time 16:0:0
#SBATCH --output=./out/SoA_replicates/log_%j.out
#SBATCH --account=math-454
#SBATCH --partition=academic
#SBATCH --job-name=rep_SoA

module load gcc hdf5

# Each step doubles total cell count (nx*ny) by scaling both dims by sqrt(2).
# Total points: 20K → 40K → 80K → 160K → 320K -> 640K → 1.28M
NX_VALS=(566 800 1131)
NY_VALS=(283 400 566)
STEPS_VALS=(60000 60000 60000)

#NX_VALS=(1131 1600)
#NY_VALS=(566 800)
#STEPS_VALS=(60000 60000)

# NOTE: #SBATCH --output is parsed by Slurm, not exported as a shell variable.
# Keep the directory in sync with the --output path above.
OUTPUT_DIR=./out/SoA_replicates
mkdir -p "${OUTPUT_DIR}"


RESULTS_CSV=${OUTPUT_DIR}/results.csv

# Create the CSV with header only if it does not exist yet.
if [ ! -f "${RESULTS_CSV}" ]; then
    echo "nx,ny,total_cells,u_in,reynolds,replicate,wall_time_s,mlups,strouhal" > ${RESULTS_CSV}
fi

U_IN=0.05
REYNOLDS=100
N_REPLICATES=3
EVERY=0

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
        srun ./serial_fused_SoA/lbm_serial_fused_SoA nx=${NX} ny=${NY} every=${EVERY} out=${SPECIFIC_OUTPUT_DIR}/lbm probe=${PROBE_PATH} re=${REYNOLDS} u_in=${U_IN} steps=${STEPS} \
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

### AoS slurm script

```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --time 16:0:0
#SBATCH --output=./out/AoS_replicates/log_%j.out
#SBATCH --account=math-454
#SBATCH --partition=academic
#SBATCH --job-name=rep_AoS

module load gcc hdf5

# Each step doubles total cell count (nx*ny) by scaling both dims by sqrt(2).
# Total points: 20K → 40K → 80K → 160K → 320K -> 640K → 1.28M
NX_VALS=(566 800 1131)
NY_VALS=(283 400 566)
STEPS_VALS=(60000 60000 60000)

#NX_VALS=(1131 1600)
#NY_VALS=(566 800)
#STEPS_VALS=(60000 60000)

# NOTE: #SBATCH --output is parsed by Slurm, not exported as a shell variable.
# Keep the directory in sync with the --output path above.
OUTPUT_DIR=./out/AoS_replicates
mkdir -p "${OUTPUT_DIR}"


RESULTS_CSV=${OUTPUT_DIR}/results.csv

# Create the CSV with header only if it does not exist yet.
if [ ! -f "${RESULTS_CSV}" ]; then
    echo "nx,ny,total_cells,u_in,reynolds,replicate,wall_time_s,mlups,strouhal" > ${RESULTS_CSV}
fi

U_IN=0.05
REYNOLDS=100
N_REPLICATES=3
EVERY=0

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
        srun ./serial_fused_AoS/lbm_serial_fused_AoS nx=${NX} ny=${NY} every=${EVERY} out=${SPECIFIC_OUTPUT_DIR}/lbm probe=${PROBE_PATH} re=${REYNOLDS} u_in=${U_IN} steps=${STEPS} \
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

### Two-step original algorithm slurm script

```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --time 16:0:0
#SBATCH --output=./out/twostep_serial_replicates/log_%j.out
#SBATCH --account=math-454
#SBATCH --partition=academic
#SBATCH --job-name=rep_twostep

module load gcc hdf5

# Each step doubles total cell count (nx*ny) by scaling both dims by sqrt(2).
# Total points: 20K → 40K → 80K → 160K → 320K -> 640K → 1.28M
NX_VALS=(566 800 1131)
NY_VALS=(283 400 566)
STEPS_VALS=(60000 60000 60000)

#NX_VALS=(1131 1600)
#NY_VALS=(566 800)
#STEPS_VALS=(60000 60000)

# NOTE: #SBATCH --output is parsed by Slurm, not exported as a shell variable.
# Keep the directory in sync with the --output path above.
OUTPUT_DIR=./out/twostep_serial_replicates
mkdir -p "${OUTPUT_DIR}"


RESULTS_CSV=${OUTPUT_DIR}/results.csv

# Create the CSV with header only if it does not exist yet.
if [ ! -f "${RESULTS_CSV}" ]; then
    echo "nx,ny,total_cells,u_in,reynolds,replicate,wall_time_s,mlups,strouhal" > ${RESULTS_CSV}
fi

U_IN=0.05
REYNOLDS=100
N_REPLICATES=3
EVERY=0

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
        srun ./serial/lbm_serial nx=${NX} ny=${NY} every=${EVERY} out=${SPECIFIC_OUTPUT_DIR}/lbm probe=${PROBE_PATH} re=${REYNOLDS} u_in=${U_IN} steps=${STEPS} \
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