import numpy as np
try:
    from nlmpy import nlmpy as nlm
except Exception:
    import nlmpy as nlm

def perlin_field(nrow: int, ncol: int, periods=(4,4), octaves=4, lacunarity=2, persistence=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    arr = nlm.perlinNoise(nRow=nrow, nCol=ncol,
                          periods=periods,
                          persistence=persistence,
                          octaves=octaves,
                          lacunarity=lacunarity)
    a_min, a_max = np.nanmin(arr), np.nanmax(arr)
    if a_max - a_min < 1e-12:
        return np.zeros_like(arr) + 0.5
    return (arr - a_min) / (a_max - a_min)

def random_cluster_binary(nrow:int, ncol:int, p:float=0.5, nn_prob:float=0.58, seed=None):
    if seed is not None:
        np.random.seed(seed)
    arr = nlm.randomClusterNN(nrow, ncol, nn_prob)
    flat = arr.ravel()
    kth = int((1-p)*flat.size)
    thresh = np.partition(flat, kth)[kth]
    out = (arr >= thresh).astype(float)
    return out

def rank_map_to_distribution(field01: np.ndarray, samples: np.ndarray):
    valid = samples[~np.isnan(samples)]
    sorted_vals = np.sort(valid)
    ranks = field01.ravel()
    ranks = np.clip(ranks, 1e-9, 1-1e-9)
    idx = (ranks * (sorted_vals.size-1)).astype(int)
    mapped = sorted_vals[idx].reshape(field01.shape)
    return mapped

def synth_ndvi_from_distribution(nrow:int, ncol:int, samples: np.ndarray,
                                 method:str='perlin',
                                 method_kwargs=None, seed=None):
    method_kwargs = method_kwargs or {}
    if method == 'perlin':
        base = perlin_field(nrow, ncol, seed=seed, **method_kwargs)
    elif method == 'cluster':
        bin_map = random_cluster_binary(nrow, ncol,
                                        p=float(method_kwargs.get('p', 0.5)),
                                        nn_prob=float(method_kwargs.get('nn_prob', 0.6)),
                                        seed=seed)
        noise = perlin_field(nrow, ncol, periods=method_kwargs.get('periods', (6,6)),
                             octaves=method_kwargs.get('octaves', 2),
                             lacunarity=method_kwargs.get('lacunarity', 2),
                             persistence=method_kwargs.get('persistence', 0.4),
                             seed=None if seed is None else seed+1)
        base = (bin_map * 0.7 + noise * 0.3)
        base = (base - base.min()) / (base.max() - base.min() + 1e-12)
    else:
        raise ValueError("Unknown method. Use 'perlin' or 'cluster'.")
    ndvi = rank_map_to_distribution(base, samples)
    return ndvi
