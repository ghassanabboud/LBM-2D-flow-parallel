#!/usr/bin/env python3
"""Print mean and SD of MLUPS per rank count for all six scaling experiments.

Strong scaling rows also show speedup S(p) = MLUPS(p) / MLUPS(1).
Weak scaling rows also show parallel efficiency E(p) = MLUPS(p) / (p * MLUPS(1)).
"""

import pandas as pd

STRONG = [
    ("Row-wise  | Strong scaling",   "./out/strong_scaling_mpi/results.csv"),
    ("Cart MPI  | Strong scaling",   "./out/cart_strong_scaling_mpi/results.csv"),
]

WEAK = [
    ("Row-wise  | Weak scaling (1D)", "./out/weak_scaling_mpi/results.csv"),
    ("Row-wise  | Weak scaling (2D)", "./out/weak_scaling_mpi_2D/results.csv"),
    ("Cart MPI  | Weak scaling (1D)", "./out/cart_weak_scaling_mpi/results.csv"),
    ("Cart MPI  | Weak scaling (2D)", "./out/cart_weak_scaling_mpi_2D/results.csv"),
]


def print_strong(label, path):
    df = pd.read_csv(path)
    mlups_1 = df.loc[df["nranks"] == 1, "mlups"].mean()
    df["speedup"] = df["mlups"] / mlups_1

    stats = (
        df.groupby("nranks")[["mlups", "speedup"]]
        .agg(n=("mlups", "count"), mean_mlups=("mlups", "mean"), sd_mlups=("mlups", "std"),
             mean_speedup=("speedup", "mean"), sd_speedup=("speedup", "std"))
        .reset_index()
        .sort_values("nranks")
    )

    print(f"\n{'='*76}")
    print(f"  {label}")
    print(f"  {path}")
    print(f"{'='*76}")
    print(f"  {'ranks':>6}  {'n':>4}  {'mean MLUPS':>12}  {'SD MLUPS':>10}  {'mean S(p)':>10}  {'SD S(p)':>8}")
    print(f"  {'-'*6}  {'-'*4}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*8}")
    for _, r in stats.iterrows():
        sd_m = r["sd_mlups"] if not pd.isna(r["sd_mlups"]) else 0.0
        sd_s = r["sd_speedup"] if not pd.isna(r["sd_speedup"]) else 0.0
        print(f"  {int(r['nranks']):>6}  {int(r['n']):>4}  {r['mean_mlups']:>12.3f}  {sd_m:>10.3f}  {r['mean_speedup']:>10.3f}  {sd_s:>8.3f}")


def print_weak(label, path):
    df = pd.read_csv(path)
    mlups_1 = df.loc[df["nranks"] == 1, "mlups"].mean()
    df["efficiency"] = df["mlups"] / (df["nranks"] * mlups_1)

    stats = (
        df.groupby("nranks")[["mlups", "efficiency"]]
        .agg(n=("mlups", "count"), mean_mlups=("mlups", "mean"), sd_mlups=("mlups", "std"),
             mean_eff=("efficiency", "mean"), sd_eff=("efficiency", "std"))
        .reset_index()
        .sort_values("nranks")
    )

    print(f"\n{'='*76}")
    print(f"  {label}")
    print(f"  {path}")
    print(f"{'='*76}")
    print(f"  {'ranks':>6}  {'n':>4}  {'mean MLUPS':>12}  {'SD MLUPS':>10}  {'mean E(p)':>10}  {'SD E(p)':>8}")
    print(f"  {'-'*6}  {'-'*4}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*8}")
    for _, r in stats.iterrows():
        sd_m = r["sd_mlups"] if not pd.isna(r["sd_mlups"]) else 0.0
        sd_e = r["sd_eff"] if not pd.isna(r["sd_eff"]) else 0.0
        print(f"  {int(r['nranks']):>6}  {int(r['n']):>4}  {r['mean_mlups']:>12.3f}  {sd_m:>10.3f}  {r['mean_eff']:>10.3f}  {sd_e:>8.3f}")


for label, path in STRONG:
    print_strong(label, path)

for label, path in WEAK:
    print_weak(label, path)
