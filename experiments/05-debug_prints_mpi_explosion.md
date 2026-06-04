# Experiment 05: Debug explosion of populations in MPI version with debug prints
#### **Code version:** debug prints in mpi (3b467a72e559c3a2906830eb0833f827fb7a14b3)
## Results and Next Steps

The results running the MPI version showed that the streaming on the top and bottom rows are different between the fused_SoA and MPI versions. in MPI, on the bottom row for example, streaming from the ghost rows rather than doing a check on sx means that the N, NE, and NW populations are put to 0 before streaming at each timestep, this significantly impacts the populations in the entire solid row because of collision. Hence, the values that get bounce-backed to the fluid cells on the first fluid row are not the correct one. It seems like mass is consistenly leaking from the bottom and top walls, which could explain the explosion. 

Next steps: try to fix it and get it to work using one rank. This means adding an additional streaming check to make sure the solid bottom and top walls do not stream zeros from ghost rows. Once that is fixed, I can run the MPI version with more ranks and see if there are any other issues.

## Objective 

[experiment 04](./04-try_fixing_mpi_explosion.md) showed that the zero velocities I observed in the gif and the probe.csv output of the MPI code were not due to vanishing populations but rather exploding populations that nan out and then get printed as 0. It also showed that the ghost rows outside the domain at the bottom of rank 0 and top of rank p are correctly kept at 0 throughout the simulation. I also tried fixing the misshandling of the solid array for ghost rows but it did not fix the explosion. 

## Status
**In Progress**
- **Slurm job name and ID**: 
    - **fused_SoA version for validation**: debug_small_SoA job ID 54924743
    - **MPI version**: debug_small_mpi job ID 54925371


## Expected outcomes

- _Deliverables_: log prints different informative values to be compared between the fused_SoA and MPI versions.
- _output directory_:  `out/debug_prints_run_mpi` and `out/debug_prints_run_SoA`, especially the `run.log` files.
- _decisions to take_: identify the source of the explosion in the MPI.

## Resources required

1 CPUs for MPI version (one node, one task per node, one CPU per task, still using MPI implementation) and 1 CPU for fused_SoA version for validation.

## Duration

04.06.2026

## Experiment description

In this experiment, I will only use only one rank and a very small domain and only run for one step, I will introduce prints throughout the stepping loop to debug the code. These prints will be introduced in both the fused_SoA version to be used for validation and the MPI version. Using only one rank means no prallelization, so the only difference between the MPI and fused_SoA codes are the handling of the ghost rows and how that impacts the streaming at the bottom and top wall. 

- `lbm_fused_SoA.cpp`:
    - `print_solid_boundary_rows()`: print the values of the solid array for the bottom row and the top row (solid) as well as the first and last fluid rows. implementation is straightforward.
    - `print_point_populations()`: print the values of the populations for a given x,y. This is used in `main_fused_SoA.cpp` to print the populations for a point x in the middle at y=0, y=1, y=N-1, and y=N after initialization and after each step
    - `gather_streamed()`: the streaming step was put in a private method called in `collide_stream()`.
    - `print_post_stream_pre_collision()`: uses `gather_streamed()` to print the values of the populations after streaming and before collision. This is used in `main_fused_SoA.cpp` to print the populations for a point x in the middle at y=0, y=1, y=N-1, and y=N before each step. This allows to verify correct bounce-back.

- `lbm_mpi.cpp`:
    - `print_solid_boundary_rows()`: this will need the use of global indexing .
    - `print_solid_boundary_rows_ghost()`: print the values of the solid array stored in each rank's ghost rows to verify correct ghost handling of is_solid.
    - `print_point_populations()`: will need to use a global indexing logic so only the corresponding rank prints the correct point.
    - `gather_streamed()` and `print_post_stream_pre_collision()`: again need to handle global indexing. 


### SoA script


```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --time 16:0:0
#SBATCH --output=./out/debug_prints_run_SoA/log_%j.out
#SBATCH --account=math-454
#SBATCH --partition=academic
#SBATCH --job-name=debug_small_SoA

module load gcc hdf5

OVERWRITE_OUTPUT=1

NX_VALS=(10)
NY_VALS=(40)
STEPS_VALS=(3)


# NOTE: #SBATCH --output is parsed by Slurm, not exported as a shell variable.
# Keep the directory in sync with the --output path above.
OUTPUT_DIR=./out/debug_prints_run_SoA
if [ "${OVERWRITE_OUTPUT}" -eq 1 ]; then
    rm -rf "${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}"
    echo "OVERWRITE_OUTPUT is set to 1, so deleting and recreating ${OUTPUT_DIR} to avoid appending to old logs."
fi


RESULTS_CSV=${OUTPUT_DIR}/results.csv

# Create the CSV with header only if it does not exist yet.
if [ ! -f "${RESULTS_CSV}" ]; then
    echo "nx,ny,total_cells,u_in,reynolds,replicate,wall_time_s,mlups,strouhal" > ${RESULTS_CSV}
fi

U_IN=0.05
REYNOLDS=100
N_REPLICATES=1
EVERY=0
export DEBUG_PRINTS=1 # activate debugging

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

### MPI script

```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --time 16:0:0
#SBATCH --output=./out/debug_prints_run_mpi/log_%j.out
#SBATCH --account=math-454
#SBATCH --partition=academic
#SBATCH --job-name=debug_small_SoA

module load gcc hdf5 openmpi

OVERWRITE_OUTPUT=1

NX_VALS=(10)
NY_VALS=(40)
STEPS_VALS=(3)


# NOTE: #SBATCH --output is parsed by Slurm, not exported as a shell variable.
# Keep the directory in sync with the --output path above.
OUTPUT_DIR=./out/debug_prints_run_mpi
if [ "${OVERWRITE_OUTPUT}" -eq 1 ]; then
    rm -rf "${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}"
    echo "OVERWRITE_OUTPUT is set to 1, so deleting and recreating ${OUTPUT_DIR} to avoid appending to old logs."
fi


RESULTS_CSV=${OUTPUT_DIR}/results.csv

# Create the CSV with header only if it does not exist yet.
if [ ! -f "${RESULTS_CSV}" ]; then
    echo "nx,ny,total_cells,u_in,reynolds,replicate,wall_time_s,mlups,strouhal" > ${RESULTS_CSV}
fi

U_IN=0.05
REYNOLDS=100
N_REPLICATES=1
EVERY=0
export DEBUG_PRINTS=1 # activate debugging

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