import numpy as np

def block_reduce_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr.copy()
    r, c = arr.shape
    r2, c2 = r - (r % factor), c - (c % factor)
    trimmed = arr[:r2, :c2]
    out = trimmed.reshape(r2//factor, factor, c2//factor, factor).mean(axis=(1,3))
    return out

def multi_scale_coarsen(arr, factors):
    seen = []
    for f in sorted(set(int(x) for x in factors if x >= 1)):
        seen.append((f, block_reduce_mean(arr, f)))
    return seen
