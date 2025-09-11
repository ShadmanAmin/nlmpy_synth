import numpy as np

def _weights_rook(nrow, ncol):
    pairs = []
    for i in range(nrow):
        for j in range(ncol):
            idx = i*ncol + j
            if i+1 < nrow:
                pairs.append((idx, (i+1)*ncol + j))
            if j+1 < ncol:
                pairs.append((idx, i*ncol + (j+1)))
    return pairs

def morans_i(arr2d: np.ndarray, nan_policy='omit'):
    x = arr2d.astype(float)
    nrow, ncol = x.shape
    x_mean = np.nanmean(x)
    z = x - x_mean
    pairs = _weights_rook(nrow, ncol)
    num = 0.0
    w_sum = 0.0
    for a, b in pairs:
        ai, aj = divmod(a, ncol)
        bi, bj = divmod(b, ncol)
        va, vb = z[ai, aj], z[bi, bj]
        if nan_policy == 'omit' and (np.isnan(va) or np.isnan(vb)):
            continue
        num += va * vb * 2.0
        w_sum += 2.0
    den = np.nansum(z*z)
    n = np.count_nonzero(~np.isnan(x))
    if den == 0 or w_sum == 0 or n < 2:
        return np.nan
    I = (n / w_sum) * (num / den)
    return I

def summarize_stats(arr2d: np.ndarray, semivar=False, **semivar_kwargs):
    stats = {
        'mean': float(np.nanmean(arr2d)),
        'variance': float(np.nanvar(arr2d)),
        'std_dev': float(np.nanstd(arr2d)),
        'morans_I': float(morans_i(arr2d)),
        'n': int(np.count_nonzero(~np.isnan(arr2d))),
        'shape_r': int(arr2d.shape[0]),
        'shape_c': int(arr2d.shape[1]),
    }
    return stats
