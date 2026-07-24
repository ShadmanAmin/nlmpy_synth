"""Shared fixtures for the nlm_synth test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def ndvi_samples() -> np.ndarray:
    """A bimodal NDVI-like marginal: a vegetation mode and a bare-soil mode."""
    rng = np.random.default_rng(20240101)
    vegetation = rng.normal(0.70, 0.08, size=20_000)
    soil = rng.normal(0.20, 0.09, size=12_000)
    return np.clip(np.hstack([vegetation, soil]), -0.2, 1.0)


@pytest.fixture(scope="session")
def small_field() -> np.ndarray:
    """A deterministic 64x64 Perlin field in [0, 1]."""
    from nlm_synth import perlin_field

    return perlin_field(64, 64, periods=(4, 4), octaves=3, lacunarity=2, seed=12345)


def reference_morans_i(arr: np.ndarray) -> float:
    """Naive O(n) reference implementation of rook-contiguity Moran's I.

    Deliberately written as an explicit double loop, mirroring the original
    implementation, so the vectorised version in the package can be checked
    against something obviously correct rather than against itself.
    """
    x = np.asarray(arr, dtype=float)
    n_rows, n_cols = x.shape
    z = x - np.nanmean(x)

    numerator = 0.0
    weight_sum = 0.0
    for i in range(n_rows):
        for j in range(n_cols):
            for di, dj in ((1, 0), (0, 1)):
                ii, jj = i + di, j + dj
                if ii >= n_rows or jj >= n_cols:
                    continue
                a, b = z[i, j], z[ii, jj]
                if np.isnan(a) or np.isnan(b):
                    continue
                numerator += a * b * 2.0
                weight_sum += 2.0

    denominator = np.nansum(z * z)
    n = np.count_nonzero(~np.isnan(x))
    if denominator == 0 or weight_sum == 0 or n < 2:
        return float("nan")
    return (n / weight_sum) * (numerator / denominator)
