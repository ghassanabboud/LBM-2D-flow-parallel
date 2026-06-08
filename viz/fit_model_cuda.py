import pandas as pd
import statsmodels.formula.api as smf


OUTPUT_DIR = "./out/"
perf_1D_SoA = pd.read_csv(OUTPUT_DIR + "cuda_sweep_block_x_multiple/results.csv")
perf_1D_AoS = pd.read_csv(OUTPUT_DIR + "cuda_sweep_block_x_AoS/results.csv")

perf_1D_SoA['layout'] = "SoA"
perf_1D_AoS['layout'] = "AoS"

perf = pd.concat([perf_1D_SoA, perf_1D_AoS], ignore_index=True)
perf["is_multiple_of_32"] = perf['block_x'] % 32 == 0

occupancy = {8: 0.5, 16:0.5, 32:0.5, 64:1, 80: 0.984, 96:0.98, 128:1, 144:0.9375, 192:0.9375, 256:1, 272:0.984, 384:0.9375, 400:0.8125, 512:1, 528:0.797, 768:0.75, 784:0.78, 1024:1}
perf["occupancy"] = perf['block_x'].map(occupancy)

# C() treats the column as categorical; AoS and False are the reference levels
perf = perf.query("block_x >=32")
model = smf.ols("mlups ~ C(layout) + C(is_multiple_of_32) + occupancy", data=perf)
result = model.fit()
print(result.summary())







