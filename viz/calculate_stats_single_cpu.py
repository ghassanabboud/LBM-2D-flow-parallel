import numpy as np
import pandas as pd


dirs = {
    "one step AoS": "./out/large_scale_AoS_replicates",
    "one step SoA": "./out/large_scale_SoA_replicates",
    "two step": "./out/large_scale_twostep_serial_replicates",
}


for algorithm, dir in dirs.items():
    df = pd.read_csv(f"{dir}/results.csv")
    print(f"{algorithm}: {df['mlups'].mean():.2f} ± {df['mlups'].std():.2f} MLUPS")
