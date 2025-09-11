"""
Example:
  - Build an NDVI marginal using your ETparams.MixtureETParameter (mixture vs normal)
  - Run xarray/rioxarray-backed MC that writes per-scale GeoTIFFs with CRS/transform preserved
"""

import numpy as np
import pandas as pd
import pathlib
import sys
import importlib.util

from nlm_synth.xarray_mc import run_experiments_geotiff

# ---------------------------------------------------------------------
# (A) Bring in your ETparams to produce NDVI marginal (mixture vs normal)
# ---------------------------------------------------------------------
# Adjust path if ETparams.py lives elsewhere:
etp_path = pathlib.Path('ETparams.py')  # e.g., Path('/absolute/path/ETparams.py')
if not etp_path.exists():
    raise FileNotFoundError(f"Could not find {etp_path.resolve()}")

spec = importlib.util.spec_from_file_location('ETparams', etp_path)
ETparams = importlib.util.module_from_spec(spec)
sys.modules['ETparams'] = ETparams
spec.loader.exec_module(ETparams)

# Create NDVI parameter and sample distribution
ndvi_param = ETparams.MixtureETParameter(
    name="NDVI",
    mu1_bounds=(0.1, 0.5),
    mu2_bounds=(0.5, 0.9),
    w1_bounds=(0.2, 0.8),
    sigma1_bounds=(0.02, 0.05),
    sigma2_bounds=(0.02, 0.05),
)

# Latin Hypercube to pick a parameter set (1 draw per MC run here)
ndvi_param.lhs_sample(N_samples=1)

# Choose NDVI marginal flavor: 'mixture' or 'normal'
dist_type = 'mixture'   # switch to 'normal' to compare unimodal approx
ndvi_dist = ndvi_param.create_dist(dist_type=dist_type)

# Draw many samples to define the marginal for rank-mapping
samples = np.clip(ndvi_dist.sample(100_000), -0.2, 1.0)

# ---------------------------------------------------------------------
# (B) Georeferencing settings: CRS, transform origin, pixel size
# ---------------------------------------------------------------------
# Option 1: define transform from origin + pixel size (north-up)
x0, y0 = 500_000.0, 4_000_000.0   # upper-left corner in projected coords
pixel_size = 30.0                 # meters
crs = "EPSG:32611"                # UTM example

# Option 2 (optional): derive CRS/transform from a template GeoTIFF
# import rioxarray as rxr
# tpl = rxr.open_rasterio('template.tif').squeeze()
# crs = tpl.rio.crs.to_string()
# pixel_size = tpl.rio.resolution()[0]  # assumes square pixels
# x0 = tpl.rio.transform().c
# y0 = tpl.rio.transform().f

# ---------------------------------------------------------------------
# (C) Run Monte Carlo with GeoTIFF outputs
# ---------------------------------------------------------------------
out_dir = 'ndvi_mc_geotiff'
df, meta = run_experiments_geotiff(
    samples=samples,
    out_dir=out_dir,
    nrow=512, ncol=512,
    pixel_size=pixel_size,
    x0=x0, y0=y0,
    crs=crs,
    generator_grid=[
        {'label': 'perlin_LF', 'method': 'perlin',
         'method_kwargs': dict(periods=(2,2), octaves=3, lacunarity=2, persistence=0.7)},
        {'label': 'perlin_HF', 'method': 'perlin',
         'method_kwargs': dict(periods=(8,8), octaves=6, lacunarity=2, persistence=0.5)},
        {'label': 'cluster_nn', 'method': 'cluster',
         'method_kwargs': dict(p=0.55, nn_prob=0.65, periods=(6,6), octaves=2, lacunarity=2, persistence=0.4)},
    ],
    coarsen_factors=(1,2,4,8,16,32),
    n_runs=5,
    random_seed=123,
    write_fullres=True,
    name_prefix=f'ndvi_{dist_type}',
)

print('Saved GeoTIFFs/results to:', out_dir)
print(df.head())

# Optional: save a tidy CSV in your working directory too
df.to_csv('results_mc_geotiff_summary.csv', index=False)
print('Wrote results_mc_geotiff_summary.csv')
