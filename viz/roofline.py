#!/usr/bin/env python3
"""Roofline plot for serial vs fused LBM variants.

Hardware limits (PLACEHOLDER — replace with measured values):
  peak_gflops  : theoretical FP64 peak in GFLOP/s
  peak_bw      : measured STREAM DRAM bandwidth in GB/s

Kernel data points:
  Each kernel is defined by (label, arithmetic_intensity, performance_gflops).
  AI  = FLOPs / bytes_to_DRAM  [FLOP/byte]
  Perf = FLOPs / wall_time      [GFLOP/s]
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Hardware limits (PLACEHOLDER) ────────────────────────────────────────────
peak_gflops = 92.1779   # FP64 peak, GFLOP/s
peak_bw     = 15.1   # DRAM bandwidth, GB/s   (e.g. from STREAM)

# ── Properties of problem ────────────────────────────────────────────
FLOPs_per_LU = 185 # how many floating point operations are performed for each lattice update (collision + streaming) 

# peak gflops was obtained from dgemm
# peak_bw was obtained from the STREAM benchmark in Triad mode
# Triad (A[i] = B[i] + scalar * C[i]) performs one read-modify-write across three arrays 
# it is the only kernel that exercises both load and store bandwidth simultaneously, which matches what our LBM kernel actually does
# The Triad number is what essentially every roofline paper in the HPC literature uses (e.g., Williams et al. 2009 "Roofline")

# ── Kernel data points (PLACEHOLDER) ─────────────────────────────────────────
# (label, memory bandwidth (Bytes/LU), performance [MLUP/s])
# these are values for a 800x400 grid over 60000 steps
kernels = [
    ("two-step",  360, 20.7587), 
    ("one-step AoS",  216, 27.929),
    ("one-step SoA",  216, 23.0893)
]

# ── Roofline curve ────────────────────────────────────────────────────────────
peak_mlups = peak_gflops * 1000 / FLOPs_per_LU  # convert GFLOP/s to MLUP/s
ridge = peak_gflops / peak_bw          # AI at ridge point [FLOP/byte]
ai    = np.logspace(-2, 3, 500)        # x-axis range
roof  = np.minimum(peak_bw * ai * 1000 / FLOPs_per_LU, peak_mlups)

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(ai, roof, "k-", linewidth=2, label="Roofline")
ax.axvline(ridge, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
#ax.text(ridge * 1.05, peak_gflops * 0.55, f"ridge = {ridge:.1f} FLOP/B",
#        fontsize=8, va="center")

# Bandwidth and compute ceilings (annotation)
ax.axhline(peak_mlups, color="gray", linestyle=":", linewidth=0.8)
ax.text(ai[-1] * 0.6, peak_mlups * 0.7,
        f"Peak compute: {peak_mlups:.3f} MLUP/s", fontsize=8, ha="right")

# Slope annotation (BW line region)
ax.text(0.03, peak_bw * 0.03 * 1000 / FLOPs_per_LU * 0.7,
        f"BW = {peak_bw} GB/s", fontsize=8, rotation=47)

# ── Plot kernels ──────────────────────────────────────────────────────────────
markers = ["s", "o", "o"]
colors  = ["tab:blue", "tab:orange", "tab:green"]
for (label, bytes_per_LU, perf), marker, color in zip(kernels, markers, colors):
    ai_val = FLOPs_per_LU / bytes_per_LU  # AI = FLOPs / bytes to DRAM
    #perf = perf * FLOPs_per_LU / 1000       # convert MLUP/s to GFLOP/s
    ax.scatter(ai_val, perf, marker=marker, color=color, s=40, zorder=5, label=label)

ax.set_xlabel("Arithmetic Intensity [FLOP/byte]")
ax.set_ylabel("Performance [MLUP/s]")
#ax.set_title("Roofline: serial vs fused LBM (D2Q9 BGK)")
ax.legend(loc="center right")
ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)

out = "roofline.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
plt.close(fig)
