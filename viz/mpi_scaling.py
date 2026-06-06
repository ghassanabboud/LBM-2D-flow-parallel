#!/usr/bin/env python3
"""Strong and weak scaling plots for the MPI LBM solver.

Strong scaling: fixed domain 800×400, speedup vs. rank count (log-log).
  - mean line with ±1 sd band across replicates
  - ideal linear speedup and Amdahl's law references

Weak scaling: fixed per-rank slab 400×200, efficiency vs. rank count.
  - efficiency E(p) = T(1)/T(p) = MLUPS(p) / (p × MLUPS(1)), ideal = 1
  - mean line with ±1 sd band across replicates
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

FOLDER = "./figs/scalings_mpi_cart/"
os.makedirs(FOLDER, exist_ok=True)
plt.rcParams.update({
    "font.size": 15
})

# ── Configuration ─────────────────────────────────────────────────────────────
CSV   = "./out/cart_strong_scaling_mpi/results.csv"
ALPHA = 0.9942   # parallelisable fraction for Amdahl's law  (0 < alpha <= 1)
OUT   = FOLDER + "strong_scaling.png"
# ──────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(CSV)

mlups_1 = df.loc[df["nranks"] == 1, "mlups"].mean()
df["speedup"] = df["mlups"] / mlups_1

ranks    = sorted(df["nranks"].unique())
p_smooth = np.logspace(0, np.log10(max(ranks) * 1.5), 300)

fig, ax = plt.subplots(figsize=(10, 7))

#sns.scatterplot(data=df, x="nranks", y="speedup", hue="replicate",
#                palette="deep", s=50, zorder=5, ax=ax)

sns.lineplot(data=df, x="nranks", y="speedup", estimator="mean",
             errorbar="sd", color="tab:blue", linewidth=1.8,
             label="Measured (mean ± sd)", ax=ax)

ax.plot(p_smooth, p_smooth,
        "k--", linewidth=1.0, label="Ideal (linear)")
ax.plot(p_smooth, 1.0 / ((1.0 - ALPHA) + ALPHA / p_smooth),
        "k-", linewidth=1.5, label=rf"Amdahl ($\alpha = {ALPHA}$)")

ceiling = 1.0 / (1.0 - ALPHA)
ax.axhline(ceiling, color="gray", linestyle=":", linewidth=0.8)
ax.text(max(ranks), ceiling * 1.03, f"max S = {ceiling:.0f}",
        fontsize=8, ha="right", va="bottom", color="gray")

ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_xticks(ranks)
ax.set_xticklabels(ranks)
ax.set_xlabel("Number of MPI ranks")
ax.set_ylabel(r"Speedup  $S(p) = \mathrm{MLUPS}(p)\,/\,\mathrm{MLUPS}(1)$")
#ax.set_title("Strong scaling — MPI LBM (1600x800, 60 000 steps)")
ax.set_title("")
ax.legend(title=None)
ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved {OUT}")
plt.close(fig)

# ── Weak scaling ───────────────────────────────────────────────────────────────
CSV_WEAK = "./out/cart_weak_scaling_mpi/results.csv"
CSV_WEAK_2D = "./out/cart_weak_scaling_mpi_2D/results.csv"  
OUT_WEAK = FOLDER + "weak_scaling_efficiency.png"
# ──────────────────────────────────────────────────────────────────────────────

dw = pd.read_csv(CSV_WEAK)
dw_2D = pd.read_csv(CSV_WEAK_2D)

# Efficiency: T(1)/T(p) = MLUPS(p) / (p × MLUPS(1)).
# MLUPS already accounts for total cell count, so dividing by p normalises
# back to per-rank throughput and the ideal value is 1 at every rank count.
mlups_1_weak = dw.loc[dw["nranks"] == 1, "mlups"].mean()
dw["efficiency"] = dw["mlups"] / (dw["nranks"] * mlups_1_weak)

mlups_1_weak_2D = dw_2D.loc[dw_2D["nranks"] == 1, "mlups"].mean()
dw_2D["efficiency"] = dw_2D["mlups"] / (dw_2D["nranks"] * mlups_1_weak_2D)

ranks_w = sorted(dw["nranks"].unique())
ranks_w_2D = sorted(dw_2D["nranks"].unique())

fig2, ax2 = plt.subplots(figsize=(10,7))

sns.lineplot(data=dw, x="nranks", y="efficiency", estimator="mean",
             errorbar="sd", color="tab:blue", linewidth=1.8,
             label="Weak scaling along rows", ax=ax2)

sns.lineplot(data=dw_2D, x="nranks", y="efficiency", estimator="mean",
             errorbar="sd", color="tab:orange", linewidth=1.8,
             label="Weak scaling along both dimensions", ax=ax2)


ax2.axhline(1.0, color="k", linestyle="--", linewidth=1.0, label="Ideal (efficiency = 1)")

ax2.set_xscale("log", base=2)
ax2.set_xticks([1,2,4,8,16,32,64])
ax2.set_xticklabels([1,2,4,8,16,32,64])
ax2.set_ylim(0, 1.2)
ax2.set_xlabel("Number of MPI ranks")
ax2.set_ylabel(r"Parallel efficiency  $E(p) = T_1 / T_p$")
#ax2.set_title("Weak scaling — MPI LBM (80K lattice points per rank, 60 000 steps)")
ax2.set_title("")
ax2.legend(title=None)
ax2.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)

fig2.tight_layout()
fig2.savefig(OUT_WEAK, dpi=150, bbox_inches="tight")
print(f"Saved {OUT_WEAK}")
plt.close(fig2)

# ── Weak scaling: throughput (MLUPS) ──────────────────────────────────────────
OUT_WEAK_MLUPS = FOLDER + "weak_scaling_performance.png"
# ──────────────────────────────────────────────────────────────────────────────

mlups_1 = dw.loc[dw["nranks"] == 1, "mlups"].mean()
dw["speedup"] = dw["mlups"] / mlups_1

p_smooth = np.logspace(0, np.log10(max(ranks_w) * 1.5), 300)

mlups_1_2D = dw_2D.loc[dw_2D["nranks"] == 1, "mlups"].mean()
dw_2D["speedup"] = dw_2D["mlups"] / mlups_1_2D

fig3, ax3 = plt.subplots(figsize=(10, 7))

sns.lineplot(data=dw, x="nranks", y="speedup", estimator="mean",
             errorbar="sd", color="tab:blue", linewidth=1.8,
             label="Weak scaling along rows", ax=ax3)
sns.lineplot(data=dw_2D, x="nranks", y="speedup", estimator="mean",
             errorbar="sd", color="tab:orange", linewidth=1.8,
             label="Weak scaling along both dimensions", ax=ax3)

ax3.plot(p_smooth, p_smooth,
        "k--", linewidth=1.0, label="Ideal (linear)")

ax3.set_xscale("log", base=2)
ax3.set_yscale("log", base=2)
ax3.set_xticks([1,2,4,8,16,32,64])
ax3.set_xticklabels([1,2,4,8,16,32,64])
ax3.set_xlabel("Number of MPI ranks")
ax3.set_ylabel(r"Speedup  $S(p) = \mathrm{MLUPS}(p)\,/\,\mathrm{MLUPS}(1)$")
#ax3.set_title("Weak scaling — MPI LBM (80K lattice points per rank, 60 000 steps)")
ax3.set_title("")
ax3.legend(title=None)
ax3.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)

fig3.tight_layout()
fig3.savefig(OUT_WEAK_MLUPS, dpi=150, bbox_inches="tight")
print(f"Saved {OUT_WEAK_MLUPS}")
plt.close(fig3)
