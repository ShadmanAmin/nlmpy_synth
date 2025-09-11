import numpy as np
import pandas as pd
from .generators import synth_ndvi_from_distribution
from .coarsen import multi_scale_coarsen
from .stats import summarize_stats

def run_experiments(samples: np.ndarray,
                    nrow:int=512, ncol:int=512,
                    generator_grid=None,
                    coarsen_factors=(1,2,3,4,6,8,12,16,24,32),
                    n_runs:int=30, semivar=False, random_seed: int = 42):
    if generator_grid is None:
        generator_grid = [
            {'label': 'perlin_lowfreq', 'method': 'perlin',
             'method_kwargs': dict(periods=(2,2), octaves=3, lacunarity=2, persistence=0.6)},
            {'label': 'perlin_hifreq', 'method': 'perlin',
             'method_kwargs': dict(periods=(6,6), octaves=6, lacunarity=2, persistence=0.5)},
            {'label': 'cluster_nn', 'method': 'cluster',
             'method_kwargs': dict(p=0.55, nn_prob=0.65, periods=(6,6), octaves=2, lacunarity=2, persistence=0.4)},
        ]

    rng = np.random.default_rng(random_seed)
    rows = []
    for cfg_idx, cfg in enumerate(generator_grid):
        label = cfg.get('label', f'cfg{cfg_idx}')
        method = cfg['method']
        kwargs = cfg.get('method_kwargs', {})
        for run in range(n_runs):
            seed = int(rng.integers(0, 2**31-1))
            field = synth_ndvi_from_distribution(nrow, ncol, samples, method=method, method_kwargs=kwargs, seed=seed)
            for factor, arr_c in multi_scale_coarsen(field, coarsen_factors):
                stats = summarize_stats(arr_c, semivar=semivar)
                stats.update({
                    'run': run,
                    'label': label,
                    'method': method,
                    'factor': int(factor),
                    'pixel_size_rel': float(factor),
                })
                rows.append(stats)
    df = pd.DataFrame(rows)
    return df, {'generator_grid': generator_grid, 'coarsen_factors': list(coarsen_factors)}
