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

def semivariogram_1d(arr2d: np.ndarray, max_lag=None, step=1,
                     samples=20000, random_state=None):
    """
    Compute an isotropic empirical semivariogram by random sampling of pixel pairs.
    Returns lags (distance) and gamma (semivariance).
    """
    rng = np.random.default_rng(random_state)
    r, c = arr2d.shape
    vals = arr2d
    mask = ~np.isnan(vals)
    valid_idx = np.argwhere(mask)
    if valid_idx.shape[0] < 2:
        return np.array([]), np.array([])

    # sample pixel pairs
    n_pairs = min(samples, valid_idx.shape[0]**2 // 10 + 1)
    i1 = rng.integers(0, valid_idx.shape[0], size=n_pairs)
    i2 = rng.integers(0, valid_idx.shape[0], size=n_pairs)
    p1 = valid_idx[i1]
    p2 = valid_idx[i2]

    d = np.linalg.norm(p1 - p2, axis=1)
    g = 0.5 * (vals[p1[:,0], p1[:,1]] - vals[p2[:,0], p2[:,1]])**2

    # bin distances
    if max_lag is None:
        max_lag = np.hypot(r, c) / 4.0
    bins = np.arange(0, max_lag + step, step)
    idx = np.digitize(d, bins) - 1
    gamma = np.array([np.nanmean(g[idx == k]) if np.any(idx == k) else np.nan
                      for k in range(len(bins)-1)])
    lags = 0.5 * (bins[:-1] + bins[1:])
    return lags, gamma


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
