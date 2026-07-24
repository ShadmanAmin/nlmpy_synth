# Changelog

All notable changes to this project are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0]

A cleanup and correctness pass over the original research code. Several fixes
change numerical results, so figures and CSVs from 0.1.x are not comparable with
those from this release.

### Fixed

- **Quantile mapping did not reproduce the target distribution.**
  `rank_map_to_distribution` indexed the sorted samples by the field's *value*
  (`idx = value * (n - 1)`), which is only correct if the field is uniform on
  `[0, 1]`. Perlin fields are min-max rescaled and bell-shaped, so the output
  over-represented mid-range values and truncated both tails: for a bimodal NDVI
  sample the 1st percentile came out at 0.166 instead of 0.026. Now mapped by true
  rank, reproducing the target quantiles exactly. **This changes all output
  values**, and is the reason the package's central claim now holds.

- **`random_cluster_binary` ignored its own `p` argument.** nlmpy's
  `randomClusterNN(nRow, nCol, p, n)` takes a neighbourhood *name* as its fourth
  argument, but the old code passed `nn_prob` positionally into the `p` slot and
  dropped the caller's `p` entirely, so the proportion of the field in the high
  class was not controllable. The parameter is renamed `cluster_p` to match what it
  actually does, `p` is honoured, and `neighbourhood` is exposed.

- **Radial power spectrum measured radii from a meaningless origin.**
  `np.fft.fft2` puts the DC component at index `[0, 0]`, but the code binned by
  distance from the array *centre* without an `fftshift`. Low-frequency power was
  smeared into the corners: a smooth field showed 4.9% of its power in the lowest
  frequency bins instead of ~100%, so the descriptor could not distinguish coarse
  from fine structure and the parameter fit was comparing noise. The spectrum is
  now shifted before binning, and normalised unconditionally rather than only when
  every bin happened to be non-empty.

- **The parameter grid search could exhaust memory.** nlmpy allocates a square whose
  side is `lcm(p_r·L^(o-1), p_c·L^(o-1))`; the documented default grid reached
  `periods=(12,12), octaves=6, lacunarity=5`, requiring a 37500×37500 array (11 GB)
  for a 512×512 output. Candidates are now screened before allocation. The default
  limit is the field's own size, which is also the right scientific cut-off:
  beyond it a candidate shows less than one period of its coarsest octave across
  the image. A full 308-candidate search on a 200×200 scene now takes ~11 s.

- **`perlin_field` crashed on many non-square shapes.** nlmpy's
  `extractRandomArrayFromSquareArray` calls `np.random.choice(range(dim - nRow))`,
  which raises whenever the padded square equals the requested row count.
  `perlin_field` now always requests a square and crops deterministically, which
  also makes a given seed correspond to a fixed field rather than a random window.

- **`ETParameter.create_dist` raised `AttributeError` on a fresh instance.** Its
  guard tested `self.w2`, which `__init__` never defined, so the intended
  auto-sampling path was unreachable. All parameter attributes are now initialised.

- **`summarize_stats(semivar=True)` was silently ignored.** The argument was
  accepted and discarded; it now computes and returns `semivar_range` and
  `semivar_sill`.

- **`fit_perlin_parameters_*` could raise `KeyError: 'target_moran'`** when no
  candidate improved on the initial infinite score.

- `block_reduce_mean` now ignores NaNs within a block instead of propagating them,
  and raises a clear error when the factor exceeds the array; `multi_scale_coarsen`
  skips oversized factors rather than failing mid-experiment.

- The semivariogram no longer counts self-pairs, which were piling zero
  semivariance into the shortest lag.

- `run_experiments_geotiff` no longer writes the full-resolution raster twice when
  `coarsen_factors` includes 1.

### Changed

- **Moran's I is vectorised**: ~287× faster on a 512×512 field (900 ms to 3 ms),
  numerically identical to the naive implementation, which is retained as a test
  reference. This was the dominant cost of every experiment, since it runs once per
  scale per realisation.
- **Seeding is reproducible and non-invasive.** Per-realisation seeds are drawn as a
  `(generator, run)` block up front, so extending a generator grid no longer shifts
  the realisations of the existing entries. `perlin_field` saves and restores the
  global NumPy RNG state that nlmpy requires, so it no longer clobbers the caller's
  random stream.
- Coordinate construction in `to_xarray` is vectorised instead of evaluating the
  affine transform once per row and column in a Python loop.
- `create_et_parameters` returns a dict keyed by variable name, replacing an
  8-tuple whose last element duplicated the other seven.
- `find_gaussian_instances` and `find_instances` are replaced by
  `find_matching_rows(df, columns, values)`, which works on any column set.
- `fit_gaussian_mixture` sorts components by mean so component identity is stable
  across fits, and takes a `seed`.
- `semivariogram_1d` is renamed `semivariogram` (the old name remains an alias); its
  `samples` argument is renamed `n_pairs` to avoid colliding with the meaning of
  `samples` everywhere else in the package.
- `run_experiments_geotiff`'s `write_fullres` becomes `write_rasters`, which now
  suppresses raster writing entirely for statistics-only runs.
- Binning in the semivariogram and power spectrum uses `np.bincount` in a single
  pass instead of a per-bin scan over all pairs.

### Added

- `nlm-synth` command line with `mc`, `geotiff` and `fit` subcommands, so every
  workflow can be reproduced without writing Python.
- A pytest suite of ~140 tests, including a naive reference implementation of
  Moran's I, round-trip GeoTIFF georeferencing checks, and an assertion that the
  NumPy and GeoTIFF drivers agree for a given seed.
- GitHub Actions CI across Python 3.10–3.12, with a separate job that smoke-tests
  the CLI and example scripts from a clean install.
- `perlin_internal_dim` for predicting nlmpy's internal allocation before it happens.
- `plot_field_grid` and `plot_marginal`; `plot_metric_by_scale` gains across-run
  spread shading and an `ax` argument.
- Packaging via `pyproject.toml` with `geo`, `fit` and `dev` extras. Geo-dependent
  names are lazily imported so the core workflow runs without rasterio.
- Type hints and NumPy-style docstrings throughout; `CITATION.cff`.

### Removed

- ~90 lines of commented-out and superseded padding/alignment code in
  `approximations.py`, replaced by the up-front feasibility screen.
- The `lcm_align` option and its `_max_periods_multiple` helper, which computed an
  LCM across the entire grid and floored the crop to it — usually collapsing to
  the fallback branch, and unnecessary now that candidates are screened directly.
- Committed experiment outputs (`results_mc.csv`, `best_params.csv`,
  `all_param_scores.csv`, `ndvi_mc_geotiff/`) and the 40-cell scratch notebook,
  replaced by an executed quickstart notebook and regenerable outputs.

## [0.1.0]

Initial research code: Perlin and random-cluster generators, block coarsening,
Moran's I and semivariogram, NumPy and GeoTIFF Monte Carlo drivers, Perlin
parameter approximation, and mixture-parameter sampling.
