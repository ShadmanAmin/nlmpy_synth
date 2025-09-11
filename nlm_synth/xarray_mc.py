import os
import numpy as np
import pandas as pd
from rasterio.transform import from_origin, Affine
from .generators import synth_ndvi_from_distribution
from .geox import to_xarray, write_geotiff, coarsen_xr_mean
from .stats import summarize_stats

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def run_experiments_geotiff(samples: np.ndarray,
                            out_dir: str,
                            nrow:int=512, ncol:int=512,
                            pixel_size: float = 30.0,
                            x0: float = 0.0, y0: float = 0.0,
                            crs: str = "EPSG:32611",
                            generator_grid=None,
                            coarsen_factors=(1,2,4,8,16,32),
                            n_runs:int=10,
                            random_seed: int = 42,
                            write_fullres: bool = True,
                            name_prefix: str = "ndvi"):
    ensure_dir(out_dir)
    transform = from_origin(x0, y0, pixel_size, pixel_size)
    if generator_grid is None:
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
    rng = np.random.default_rng(random_seed)
    rows = []
    for cfg_idx, cfg in enumerate(generator_grid):
        label = cfg.get('label', f'cfg{cfg_idx}')
        method = cfg['method']
        kwargs = cfg.get('method_kwargs', {})
        label_dir = ensure_dir(os.path.join(out_dir, label))
        for run in range(n_runs):
            seed = int(rng.integers(0, 2**31-1))
            field = synth_ndvi_from_distribution(nrow, ncol, samples, method=method, method_kwargs=kwargs, seed=seed)
            if write_fullres:
                full_name = f"{name_prefix}_{label}_run{run}_f1.tif"
                write_geotiff(os.path.join(label_dir, full_name), field, transform, crs, nodata=None, name="ndvi" )
            da = to_xarray(field, transform, crs, nodata=None, name="ndvi")
            for factor in sorted(set(int(x) for x in coarsen_factors if x >= 1)):
                da_c = coarsen_xr_mean(da, factor=factor)
                new_transform = Affine(da.rio.transform().a * factor,
                                       da.rio.transform().b,
                                       da.rio.transform().c,
                                       da.rio.transform().d,
                                       da.rio.transform().e * factor,
                                       da.rio.transform().f)
                da_c.rio.write_transform(new_transform, inplace=True)
                out_name = f"{name_prefix}_{label}_run{run}_f{factor}.tif"
                da_c.rio.to_raster(os.path.join(label_dir, out_name))
                stats = summarize_stats(da_c.values, semivar=False)
                stats.update({
                    'run': run,
                    'label': label,
                    'method': method,
                    'factor': int(factor),
                    'pixel_size': float(pixel_size * factor),
                    'nrow': int(da_c.sizes['y']),
                    'ncol': int(da_c.sizes['x'])
                })
                rows.append(stats)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "results_mc_geotiff.csv"), index=False)
    return df, {'generator_grid': generator_grid, 'coarsen_factors': list(coarsen_factors),
                'crs': crs, 'pixel_size': pixel_size, 'origin': (x0, y0)}
