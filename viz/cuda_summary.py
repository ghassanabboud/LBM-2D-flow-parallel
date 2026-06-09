import pandas as pd

OUTPUT_DIR = "./out/"

datasets = {
    "1D sweep SoA": (pd.read_csv(OUTPUT_DIR + "cuda_sweep_block_x_SoA/results.csv"), "block_x"),
    "1D sweep AoS": (pd.read_csv(OUTPUT_DIR + "cuda_sweep_block_x_AoS/results.csv"), "block_x"),
    "2D sweep SoA": (pd.read_csv(OUTPUT_DIR + "cuda_sweep_2D_SoA/results.csv"), ["block_x", "block_y"]),
    "2D sweep AoS": (pd.read_csv(OUTPUT_DIR + "cuda_sweep_2D_AoS/results.csv"), ["block_x", "block_y"]),
    "Problem size SoA": (pd.read_csv(OUTPUT_DIR + "cuda_sweep_problem_size_SoA/results.csv"), ["total_cells", "block_x"]),
    "Problem size AoS": (pd.read_csv(OUTPUT_DIR + "cuda_sweep_problem_size_AoS/results.csv"), ["total_cells", "block_x"]),
}

for name, (df, group_col) in datasets.items():
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    overall_mean = df["mlups"].mean()
    overall_std = df["mlups"].std()
    overall_max = df["mlups"].max()
    print(f"  Overall  — mean: {overall_mean:8.2f}  std: {overall_std:7.2f}  max: {overall_max:8.2f} MLUP/s")
    print()
    summary = df.groupby(group_col)["mlups"].agg(mean="mean", std="std", max="max").reset_index()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))