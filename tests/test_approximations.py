"""Tests for Perlin parameter estimation from an observed field."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rioxarray", reason="requires the optional geo extra")

from nlm_synth.approximations import (  # noqa: E402
    _feasible_combos,
    fit_perlin_parameters_array,
    fit_perlin_parameters_geotiff,
    radial_power_spectrum,
    square_crop_dataarray,
)
from nlm_synth.generators import perlin_field  # noqa: E402


def spectral_centroid(power: np.ndarray) -> float:
    """Power-weighted mean bin index: low for coarse structure, high for fine."""
    idx = np.arange(power.size)
    return float(np.nansum(idx * power) / np.nansum(power))


class TestRadialPowerSpectrum:
    def test_normalised_to_unit_total(self, small_field):
        _, power = radial_power_spectrum(small_field)
        assert np.nansum(power) == pytest.approx(1.0)

    def test_frequencies_span_zero_to_one(self, small_field):
        freq, power = radial_power_spectrum(small_field, n_bins=32)
        assert freq.size == power.size == 32
        assert 0.0 <= freq[0] < freq[-1] <= 1.0

    def test_coarse_structure_concentrates_power_at_low_frequency(self):
        """Requires the fftshift: without it the DC term sits in the corners.

        The previous implementation binned radii from the array centre while
        `fft2` puts DC at [0, 0], so a smooth field appeared to have almost no
        low-frequency power and the descriptor could not tell coarse from fine.
        """
        coarse = perlin_field(128, 128, periods=(2, 2), octaves=1, seed=1)
        _, power = radial_power_spectrum(coarse, n_bins=60)
        assert np.nansum(power[:5]) > 0.9

    def test_separates_coarse_from_fine_structure(self):
        coarse = perlin_field(128, 128, periods=(2, 2), octaves=1, seed=1)
        fine = perlin_field(128, 128, periods=(16, 16), octaves=1, seed=1)
        _, coarse_power = radial_power_spectrum(coarse)
        _, fine_power = radial_power_spectrum(fine)
        assert spectral_centroid(fine_power) > spectral_centroid(coarse_power)

    def test_is_invariant_to_amplitude(self, small_field):
        _, a = radial_power_spectrum(small_field)
        _, b = radial_power_spectrum(small_field * 100.0)
        np.testing.assert_allclose(a, b, rtol=1e-9)

    def test_nans_are_tolerated(self, small_field):
        field = small_field.copy()
        field[10:20, 10:20] = np.nan
        _, power = radial_power_spectrum(field)
        assert np.nansum(power) == pytest.approx(1.0)


class TestFeasibleCombos:
    def test_screens_out_the_oversized_combination(self):
        """periods=(12,12), octaves=6, lacunarity=5 needs a 37500x37500 grid."""
        feasible, skipped = _feasible_combos(
            [(12, 12)], [6], [5], [0.5], max_internal_dim=8192
        )
        assert feasible == []
        assert len(skipped) == 1

    def test_keeps_modest_combinations(self):
        feasible, skipped = _feasible_combos(
            [(2, 2), (4, 4)], [1, 2, 3], [2], [0.5], max_internal_dim=8192
        )
        assert len(feasible) == 6 and skipped == []

    def test_auto_limit_tracks_the_field_size(self):
        """'auto' keeps only candidates whose full period structure fits the field."""
        target = perlin_field(64, 64, periods=(4, 4), octaves=2, seed=1)
        with pytest.warns(UserWarning, match="less than one period"):
            _, diagnostics = fit_perlin_parameters_array(
                target,
                periods_grid=((4, 4),),
                octaves_grid=(2, 6),
                lacunarity_grid=(4,),
                persistence_grid=(0.5,),
                verbose=False,
            )
        # octaves=6, lacunarity=4 needs 4*4**5 = 4096 > 64, so only octaves=2 survives.
        assert [d["octaves"] for d in diagnostics] == [2]


class TestFitPerlinParametersArray:
    def test_recovers_the_generating_periods(self):
        target = perlin_field(128, 128, periods=(8, 8), octaves=3, lacunarity=2,
                              persistence=0.5, seed=99)
        best, _ = fit_perlin_parameters_array(
            target, periods_grid=((2, 2), (4, 4), (8, 8), (12, 12)),
            octaves_grid=(3,), lacunarity_grid=(2,), persistence_grid=(0.5,),
            seed=5, verbose=False,
        )
        assert best["periods"] == (8, 8)

    def test_recovers_the_generating_octaves(self):
        target = perlin_field(128, 128, periods=(4, 4), octaves=5, lacunarity=2,
                              persistence=0.6, seed=17)
        best, _ = fit_perlin_parameters_array(
            target, periods_grid=((4, 4),), octaves_grid=(1, 3, 5),
            lacunarity_grid=(2,), persistence_grid=(0.6,), seed=17, verbose=False,
        )
        assert best["octaves"] == 5

    def test_diagnostics_cover_every_candidate(self):
        target = perlin_field(64, 64, periods=(4, 4), octaves=2, seed=1)
        _, diagnostics = fit_perlin_parameters_array(
            target, periods_grid=((2, 2), (4, 4)), octaves_grid=(1, 2),
            lacunarity_grid=(2,), persistence_grid=(0.5,), verbose=False,
        )
        assert len(diagnostics) == 4
        assert all(np.isfinite(d["score"]) for d in diagnostics)

    def test_best_always_carries_target_moran(self):
        """`target_moran` used to be missing whenever no candidate improved on inf."""
        target = perlin_field(64, 64, periods=(4, 4), octaves=2, seed=2)
        best, _ = fit_perlin_parameters_array(
            target, periods_grid=((4, 4),), octaves_grid=(2,),
            lacunarity_grid=(2,), persistence_grid=(0.5,), verbose=False,
        )
        assert {"periods", "octaves", "lacunarity", "persistence",
                "score", "moran", "target_moran"} <= set(best)

    def test_warns_and_continues_when_some_combos_are_too_large(self):
        target = perlin_field(64, 64, periods=(4, 4), octaves=2, seed=3)
        with pytest.warns(UserWarning, match="max_internal_dim"):
            best, _ = fit_perlin_parameters_array(
                target, periods_grid=((4, 4), (12, 12)), octaves_grid=(2, 6),
                lacunarity_grid=(5,), persistence_grid=(0.5,),
                max_internal_dim=4096, verbose=False,
            )
        assert best["periods"] == (4, 4)

    def test_raises_when_nothing_is_feasible(self):
        target = perlin_field(32, 32, periods=(2, 2), octaves=2, seed=4)
        with pytest.warns(UserWarning), pytest.raises(ValueError, match="No parameter combination"):
            fit_perlin_parameters_array(
                target, periods_grid=((12, 12),), octaves_grid=(6,),
                lacunarity_grid=(5,), persistence_grid=(0.5,),
                max_internal_dim=1024, verbose=False,
            )

    def test_is_insensitive_to_the_marginal_distribution(self):
        """Matching happens on ranks, so a monotone transform must not change the fit."""
        target = perlin_field(96, 96, periods=(8, 8), octaves=2, seed=8)
        grid = dict(periods_grid=((2, 2), (8, 8)), octaves_grid=(2,),
                    lacunarity_grid=(2,), persistence_grid=(0.5,), seed=1, verbose=False)
        plain, _ = fit_perlin_parameters_array(target, **grid)
        skewed, _ = fit_perlin_parameters_array(np.exp(5 * target), **grid)
        assert plain["periods"] == skewed["periods"]

    @pytest.mark.parametrize("bad", [np.zeros((4, 4, 4)), np.full((8, 8), np.nan)])
    def test_rejects_invalid_input(self, bad):
        with pytest.raises(ValueError):
            fit_perlin_parameters_array(bad, verbose=False)


class TestSquareCropDataArray:
    @pytest.fixture
    def rectangular(self):
        from rasterio.transform import from_origin

        from nlm_synth.geox import to_xarray

        return to_xarray(np.arange(200.0).reshape(10, 20), from_origin(0, 0, 1, 1), "EPSG:32611")

    def test_produces_a_square(self, rectangular):
        cropped = square_crop_dataarray(rectangular)
        assert cropped.sizes["y"] == cropped.sizes["x"] == 10

    @pytest.mark.parametrize("align,first_x", [("ul", 0.5), ("ur", 10.5), ("center", 5.5)])
    def test_alignment_selects_the_expected_window(self, rectangular, align, first_x):
        cropped = square_crop_dataarray(rectangular, align=align)
        assert cropped.x.values[0] == pytest.approx(first_x)

    def test_georeferencing_survives(self, rectangular):
        assert square_crop_dataarray(rectangular).rio.crs is not None

    def test_rejects_unknown_alignment(self, rectangular):
        with pytest.raises(ValueError, match="align must be"):
            square_crop_dataarray(rectangular, align="middle")


class TestFitPerlinParametersGeotiff:
    @pytest.fixture
    def raster_path(self, tmp_path):
        from rasterio.transform import from_origin

        from nlm_synth.geox import write_geotiff

        field = perlin_field(96, 128, periods=(8, 8), octaves=3, lacunarity=2,
                             persistence=0.5, seed=31)
        path = tmp_path / "scene.tif"
        write_geotiff(str(path), field, from_origin(0, 0, 30, 30), "EPSG:32611")
        return path

    def test_end_to_end_writes_both_csvs(self, raster_path, tmp_path):
        import pandas as pd

        best_csv = tmp_path / "best.csv"
        diag_csv = tmp_path / "diagnostics.csv"
        best = fit_perlin_parameters_geotiff(
            str(raster_path), out_csv=str(best_csv), save_diagnostics_csv=str(diag_csv),
            periods_grid=((2, 2), (8, 8)), octaves_grid=(3,),
            lacunarity_grid=(2,), persistence_grid=(0.5,), verbose=False,
        )
        assert best["periods"] == (8, 8)

        best_row = pd.read_csv(best_csv)
        assert len(best_row) == 1
        assert best_row.loc[0, "periods_r"] == 8
        assert best_row.loc[0, "n_rows"] == best_row.loc[0, "n_cols"] == 96
        assert len(pd.read_csv(diag_csv)) == 2

    def test_runs_without_writing_anything(self, raster_path):
        best = fit_perlin_parameters_geotiff(
            str(raster_path), periods_grid=((4, 4),), octaves_grid=(2,),
            lacunarity_grid=(2,), persistence_grid=(0.5,), verbose=False,
        )
        assert "score" in best
