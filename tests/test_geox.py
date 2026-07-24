"""Tests for georeferencing helpers and the GeoTIFF Monte Carlo driver."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rioxarray", reason="requires the optional geo extra")

from affine import Affine  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from nlm_synth.geox import coarsen_xr_mean, scale_transform, to_xarray, write_geotiff  # noqa: E402
from nlm_synth.xarray_mc import RESULTS_FILENAME, run_experiments_geotiff  # noqa: E402

CRS = "EPSG:32611"
PIXEL = 30.0
X0, Y0 = 500_000.0, 4_000_000.0


@pytest.fixture
def transform():
    return from_origin(X0, Y0, PIXEL, PIXEL)


@pytest.fixture
def data():
    return np.arange(48.0).reshape(8, 6)


class TestToXarray:
    def test_dims_crs_and_transform(self, data, transform):
        da = to_xarray(data, transform, CRS)
        assert da.dims == ("y", "x")
        assert da.shape == data.shape
        assert da.rio.crs.to_string() == CRS
        assert da.rio.transform() == transform

    def test_coordinates_are_cell_centres(self, data, transform):
        da = to_xarray(data, transform, CRS)
        assert da.x.values[0] == pytest.approx(X0 + PIXEL / 2)
        assert da.y.values[0] == pytest.approx(Y0 - PIXEL / 2)
        assert np.diff(da.x.values)[0] == pytest.approx(PIXEL)
        assert np.diff(da.y.values)[0] == pytest.approx(-PIXEL)

    def test_matches_an_explicit_affine_evaluation(self, data, transform):
        """The vectorised coordinate build must agree with per-cell affine maths."""
        da = to_xarray(data, transform, CRS)
        expected_x = [(transform * (c + 0.5, 0.5))[0] for c in range(data.shape[1])]
        expected_y = [(transform * (0.5, r + 0.5))[1] for r in range(data.shape[0])]
        np.testing.assert_allclose(da.x.values, expected_x)
        np.testing.assert_allclose(da.y.values, expected_y)

    def test_nodata_is_recorded(self, data, transform):
        assert to_xarray(data, transform, CRS, nodata=-9999.0).rio.nodata == -9999.0

    def test_rejects_non_2d(self, transform):
        with pytest.raises(ValueError):
            to_xarray(np.zeros((2, 2, 2)), transform, CRS)


class TestScaleTransform:
    def test_pixel_size_scales_and_origin_is_fixed(self, transform):
        scaled = scale_transform(transform, 4)
        assert scaled.a == pytest.approx(PIXEL * 4)
        assert scaled.e == pytest.approx(-PIXEL * 4)
        assert (scaled.c, scaled.f) == (transform.c, transform.f)

    def test_matches_manual_affine_construction(self, transform):
        manual = Affine(
            transform.a * 3, transform.b, transform.c,
            transform.d, transform.e * 3, transform.f,
        )
        assert scale_transform(transform, 3) == manual


class TestCoarsenXrMean:
    def test_updates_shape_and_transform(self, data, transform):
        coarse = coarsen_xr_mean(to_xarray(data, transform, CRS), factor=2)
        assert coarse.shape == (4, 3)
        assert coarse.rio.transform().a == pytest.approx(PIXEL * 2)
        assert coarse.rio.crs.to_string() == CRS

    def test_factor_one_is_a_no_op(self, data, transform):
        da = to_xarray(data, transform, CRS)
        assert coarsen_xr_mean(da, 1) is da

    def test_values_match_numpy_block_mean(self, data, transform):
        from nlm_synth.coarsen import block_reduce_mean

        coarse = coarsen_xr_mean(to_xarray(data, transform, CRS), factor=2)
        np.testing.assert_allclose(coarse.values, block_reduce_mean(data, 2))


class TestWriteGeotiff:
    def test_roundtrips_through_disk(self, data, transform, tmp_path):
        import rioxarray as rxr

        path = tmp_path / "out.tif"
        write_geotiff(str(path), data, transform, CRS)

        with rxr.open_rasterio(path) as reopened:
            np.testing.assert_allclose(reopened.squeeze().values, data)
            assert reopened.rio.crs.to_string() == CRS
            assert reopened.rio.transform() == transform


@pytest.fixture(scope="module")
def run(ndvi_samples, tmp_path_factory):
    """One shared GeoTIFF Monte Carlo run, reused across the assertions below."""
    out_dir = tmp_path_factory.mktemp("geotiff_mc")
    grid = [{"label": "lf", "method": "perlin",
             "method_kwargs": dict(periods=(2, 2), octaves=2, lacunarity=2, persistence=0.7)}]
    df, meta = run_experiments_geotiff(
        ndvi_samples, out_dir=out_dir, nrow=32, ncol=32, pixel_size=PIXEL,
        x0=X0, y0=Y0, crs=CRS, generator_grid=grid,
        coarsen_factors=(1, 2, 4), n_runs=2, random_seed=3,
    )
    return df, meta, out_dir


class TestRunExperimentsGeotiff:
    def test_row_count_and_pixel_size_column(self, run):
        df, _, _ = run
        assert len(df) == 1 * 2 * 3
        assert sorted(df["pixel_size"].unique()) == [PIXEL, PIXEL * 2, PIXEL * 4]

    def test_writes_one_raster_per_run_and_scale(self, run):
        _, _, out_dir = run
        rasters = sorted((out_dir / "lf").glob("*.tif"))
        assert len(rasters) == 2 * 3, "expected exactly one raster per (run, factor)"

    def test_rasters_are_georeferenced_consistently(self, run):
        import rioxarray as rxr

        _, _, out_dir = run
        with rxr.open_rasterio(out_dir / "lf" / "ndvi_lf_run0_f4.tif") as raster:
            assert raster.rio.crs.to_string() == CRS
            assert raster.rio.transform().a == pytest.approx(PIXEL * 4)
            assert raster.rio.transform().c == pytest.approx(X0)
            assert raster.squeeze().shape == (8, 8)

    def test_results_csv_is_written(self, run):
        df, _, out_dir = run
        assert (out_dir / RESULTS_FILENAME).exists()

    def test_no_rasters_mode_writes_only_the_csv(self, ndvi_samples, tmp_path):
        df, _ = run_experiments_geotiff(
            ndvi_samples, out_dir=tmp_path, nrow=32, ncol=32,
            coarsen_factors=(1, 2), n_runs=1, write_rasters=False,
            generator_grid=[{"label": "lf", "method": "perlin",
                             "method_kwargs": dict(periods=(2, 2), octaves=2)}],
        )
        assert len(df) == 2
        assert not list(tmp_path.rglob("*.tif"))

    def test_matches_the_numpy_pipeline(self, ndvi_samples, tmp_path):
        """Both drivers must report the same statistics for the same seed."""
        from nlm_synth.monte_carlo import run_experiments

        grid = [{"label": "lf", "method": "perlin",
                 "method_kwargs": dict(periods=(2, 2), octaves=2, lacunarity=2, persistence=0.7)}]
        kwargs = dict(generator_grid=grid, coarsen_factors=(1, 2, 4), n_runs=2, random_seed=11)

        numpy_df, _ = run_experiments(ndvi_samples, nrow=32, ncol=32, **kwargs)
        geo_df, _ = run_experiments_geotiff(
            ndvi_samples, out_dir=tmp_path, nrow=32, ncol=32, write_rasters=False, **kwargs
        )
        np.testing.assert_allclose(
            numpy_df["morans_I"].to_numpy(), geo_df["morans_I"].to_numpy(), rtol=1e-9
        )
