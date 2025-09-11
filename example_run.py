"""
Example pipeline to:
  1) Load or synthesize an NDVI marginal distribution
  2) Run Monte Carlo experiments with different spatial-structure generators
  3) Plot Moran's I and variance vs. coarsening factor (NumPy-only workflow)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nlm_synth.monte_carlo import run_experiments
from nlm_synth.visualize import plot_metric_by_scale

# ---------------------------------------------------------------------
# 1) NDVI distribution
# ---------------------------------------------------------------------
# Option A: load real NDVI samples from file
# samples = np.load('ndvi_samples.npy')

# Option B: synthetic bi-modal NDVI mixture (demo)
rng = np.random.default_rng(123)
veg = rng.normal(0.7, 0.08, size=50_000)
soil = rng.normal(0.2, 0.05, size=30_000)
samples = np.clip(np.hstack([veg, soil]), -0.2, 1.0)

# ---------------------------------------------------------------------
# 2) Run Monte Carlo over generators + coarsening factors
# ---------------------------------------------------------------------
generator_grid = [
    {'label': 'perlin_LF', 'method': 'perlin',
     'method_kwargs': dict(periods=(2,2), octaves=3, lacunarity=2, persistence=0.7)},
    {'label': 'perlin_MF', 'method': 'perlin',
     'method_kwargs': dict(periods=(4,4), octaves=5, lacunarity=2, persistence=0.6)},
    {'label': 'perlin_HF', 'method': 'perlin',
     'method_kwargs': dict(periods=(8,8), octaves=6, lacunarity=2, persistence=0.5)},
    {'label': 'cluster_nn', 'method': 'cluster',
     'method_kwargs': dict(p=0.55, nn_prob=0.65, periods=(6,6), octaves=2, lacunarity=2, persistence=0.4)},
]

df, meta = run_experiments(samples,
                           nrow=512, ncol=512,
                           generator_grid=generator_grid,
                           coarsen_factors=(1,2,4,8,16,32),
                           n_runs=10, semivar=False, random_seed=44)

df.to_csv('results_mc.csv', index=False)
print('Saved results to results_mc.csv')
print(df.head())

# ---------------------------------------------------------------------
# 3) Plots
# ---------------------------------------------------------------------
plot_metric_by_scale(df, metric='morans_I', by='label')
plt.savefig('moransI_vs_scale.png', dpi=150)
plot_metric_by_scale(df, metric='variance', by='label')
plt.savefig('variance_vs_scale.png', dpi=150)
print('Saved plots: moransI_vs_scale.png, variance_vs_scale.png')
