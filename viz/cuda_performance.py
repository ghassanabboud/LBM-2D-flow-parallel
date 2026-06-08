import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

#properties of problem
FLOPS_per_LU = 185
max_BW_in_GBs = 900
bytes_per_LU = 216

# file paths
OUTPUT_DIR = "./out/"
perf_1D_SoA = pd.read_csv(OUTPUT_DIR + "cuda_sweep_block_x_SoA/results.csv")
perf_2D_SoA = pd.read_csv(OUTPUT_DIR + "cuda_sweep_2D_SoA/results.csv")
perf_1D_AoS = pd.read_csv(OUTPUT_DIR + "cuda_sweep_block_x_AoS/results.csv")
perf_2D_AoS = pd.read_csv(OUTPUT_DIR + "cuda_sweep_2D_AoS/results.csv")
FIGS_DIR = "./figs/cuda/"
os.makedirs(FIGS_DIR, exist_ok=True)

occupancy = {8: 0.5, 16:0.5, 32:0.5, 64:1, 80: 0.984, 96:0.98, 128:1, 144:0.9375, 192:0.9375, 256:1, 272:0.984, 384:0.9375, 400:0.8125, 512:1, 528:0.797, 768:0.75, 784:0.78, 1024:1}

problem_size_SoA = pd.read_csv(OUTPUT_DIR + "cuda_sweep_problem_size_SoA/results.csv")
problem_size_AoS = pd.read_csv(OUTPUT_DIR + "cuda_sweep_problem_size_AoS/results.csv")

# problem size analysis across two block sizes

plt.figure(figsize=(10, 6))

ai = FLOPS_per_LU / bytes_per_LU
peak_flops = max_BW_in_GBs * 1e9 * ai / (FLOPS_per_LU * 1e6)  
peak_mlups = (peak_flops / FLOPS_per_LU)* 1e-6
peak_mlups = max_BW_in_GBs * ai * 1000 / FLOPS_per_LU
size_to_saturation=163840

problem_size_SoA['label'] = problem_size_SoA.apply(lambda row: f"SoA - block.x = {row['block_x']}", axis=1)
problem_size_AoS['label'] = problem_size_AoS.apply(lambda row: f"AoS - block.x = {row['block_x']}", axis=1)
sns.lineplot(data=problem_size_SoA, x="total_cells", y="mlups", estimator="mean", errorbar="sd", linewidth=1.8, hue="label", palette="Set1")
sns.lineplot(data=problem_size_AoS, x="total_cells", y="mlups", estimator="mean", errorbar="sd",  linewidth=1.8, hue="label", palette="Set2")
plt.axhline(peak_mlups, color="red", linestyle="--", linewidth=1.5, label=f"Peak performance: {peak_mlups:.2f} MLUP/s")
plt.axvline(size_to_saturation, color="blue", linestyle="--", linewidth=1.5, label=f"Size to saturate GPU: {size_to_saturation/1e3:.1f}K cells")
plt.xscale("log", base=2)
plt.xticks(problem_size_SoA['total_cells'].unique(), rotation=90, fontsize=10)
plt.xlabel("Total number of lattice cells $N$", fontsize=15)
plt.ylabel("Performance (MLUPs)", fontsize=15)
plt.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(FIGS_DIR+"problem_size_perf.png", dpi=300)

# problem size analysis only for block_x=256

plt.figure(figsize=(10, 6))

problem_size_SoA = problem_size_SoA[problem_size_SoA['block_x'] == 256]
problem_size_AoS = problem_size_AoS[problem_size_AoS['block_x'] == 256]

sns.lineplot(data=problem_size_SoA, x="total_cells", y="mlups", estimator="mean", errorbar="sd", linewidth=1.8, label="SoA", color="tab:green")
sns.lineplot(data=problem_size_AoS, x="total_cells", y="mlups", estimator="mean", errorbar="sd",  linewidth=1.8, label="AoS", color="tab:orange")
plt.axhline(peak_mlups, color="red", linestyle="--", linewidth=1.5, label=f"Peak performance: {peak_mlups:.2f} MLUP/s")
plt.axvline(size_to_saturation, color="blue", linestyle="--", linewidth=1.5, label=f"Size to saturate GPU: {size_to_saturation/1e3:.1f}K cells")
plt.xscale("log", base=2)
plt.xticks(problem_size_SoA['total_cells'].unique(), rotation=90, fontsize=10)
plt.xlabel("Total number of lattice cells $N$", fontsize=15)
plt.ylabel("Performance (MLUPs)", fontsize=15)
plt.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(FIGS_DIR+"problem_size_perf_256.png", dpi=300)

# 1D plot

block_x_values = sorted(perf_1D_SoA['block_x'].unique())


fig, ax1 = plt.subplots(figsize=(10, 6))


perf_1D_SoA['is_multiple_of_32'] = perf_1D_SoA['block_x'].apply(lambda x: x % 32 == 0)
perf_1D_SoA['label'] = perf_1D_SoA.apply(lambda row: "Multiple of 32" if row['is_multiple_of_32'] else "Not multiple of 32", axis=1)

sns.lineplot(data=perf_1D_SoA, x="block_x", y="mlups", estimator="mean",
             errorbar="sd", color="green", hue="label", style="label", linewidth=1.8, ax=ax1, palette="Set1")



ax1.set_xscale("log", base=2)
ax1.set_xticks(block_x_values)
ax1.set_xticklabels(block_x_values, rotation=90, fontsize=8)
ax1.set_xlabel("Block size in x direction", fontsize=14)
ax1.set_ylabel("Performance (MLUPs)", fontsize=14)
#ax1.set_title("Performance vs block_x (1D sweep)", fontsize=16)
ax1.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
ax1.legend(title="Block size x is:", fontsize=12)

plt.tight_layout()
plt.savefig(FIGS_DIR+"perf_1D.png", dpi=300)

# 1D plot with occupancy


block_x_values = sorted(perf_1D_SoA['block_x'].unique())


fig, ax1 = plt.subplots(figsize=(10, 6))


perf_1D_SoA['is_multiple_of_32'] = perf_1D_SoA['block_x'].apply(lambda x: x % 32 == 0)
perf_1D_SoA['label'] = perf_1D_SoA.apply(lambda row: "Multiple of 32" if row['is_multiple_of_32'] else "Not multiple of 32", axis=1)
perf_1D_SoA['occupancy'] = perf_1D_SoA['block_x'].map(occupancy)

sns.lineplot(data=perf_1D_SoA, x="block_x", y="mlups", estimator="mean",
             errorbar="sd", color="green", hue="label", style="label", linewidth=1.8, ax=ax1, palette="Set1")



ax1.set_xscale("log", base=2)
ax1.set_xticks(block_x_values)
ax1.set_xticklabels(block_x_values, rotation=90, fontsize=8)
ax1.set_xlabel("Block size in x direction", fontsize=14)
ax1.set_ylabel("Performance (MLUPs)", fontsize=14)
#ax1.set_title("Performance vs block_x (1D sweep)", fontsize=16)
ax1.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
ax1.legend(title="Block size x is:", fontsize=12)

ax2 = ax1.twinx()
sns.scatterplot(data=perf_1D_SoA, x="block_x", y="occupancy", hue="label",palette="Set1", s=15, ax=ax2, legend=False)
ax2.set_ylabel("Occupancy", fontsize=14)
ax2.set_ylim(0, 1.05)

#ax2.legend(loc="upper left", fontsize=12)

plt.tight_layout()
plt.savefig(FIGS_DIR+"perf_1D_with_occupancy.png", dpi=300)