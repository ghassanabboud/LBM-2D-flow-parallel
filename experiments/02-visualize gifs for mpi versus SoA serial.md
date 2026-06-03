# Experiment N: Visualize GIFs for MPI version
 #### **Code version:** mpi debug prints (d800bb03106e06a5677a0e2917598f24b8bb25bf)

## Results and Next Steps

Comparison of the MPI gif and the original serial SoA shows a big problem. The values of velocities zero out completely very early in the simulation and stay like that throughout. There is obviously a bug in the MPI code that needs to be resolved.

## Objective 

Strouhal number validation for the MPI version showed that the code cannot recover the Strouhal value on a 400x200 grid. No peak is detected by `viz/strouhal.py` and inspection of `probe.csv` shows that the velocity at the probe drops to exactly 0 and stays there from step 700. These results are shown in `out/fix_writer` This experiment aims to visualize the entire velocity field to check whether the MPI code causes this problem on the entire grid or only at the probe location. the GIF was already generated for the MPI version [in the fix_writer folder](../out/fix_writer/nx_400_ny_200_rep_1/vorticity.gif)

## Status
**Concluded**
- **Slurm job name and ID**: gif.slurm, job ID 54909471

## Expected outcomes
- _Deliverables_: GIFs
- _output directory_: [out/serial_fused_SoA_sweep/nx_800_ny_400/vorticity.gif](../out/serial_fused_SoA_sweep/nx_800_ny_400/vorticity.gif)
- _decisions to take_: next debugging steps 

## Resources required
None

## Duration
typically 10 minutes for such a grid size

## Experiment description

I used old results I had generated for a [sweep on the SoA one-step serial implementation](../out/serial_fused_SoA_sweep/) to create the GIF. This serves as baseline to compare MPI to because I validated its correctness using Strouhal. 

### slurm script

```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --time 1:0:0

#SBATCH --account=math-454
#SBATCH --partition=academic
#SBATCH --output=./plotting_log.out



XDMF_PATH=/scratch/gabboud/LBM-2D-flow-parallel/out/serial_fused_SoA_sweep/nx_800_ny_400/lbm_.xdmf

srun python viz/viz.py ${XDMF_PATH} --out "$(dirname ${XDMF_PATH})/vorticity.gif"
```
## Links and references
TO-DO: list here publications, web pages, etc. that contain information relevant to the experiment. 