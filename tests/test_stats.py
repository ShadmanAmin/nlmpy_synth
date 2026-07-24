"""Tests for the spatial statistics, including a check against a naive reference."""

from __future__ import annotations

import numpy as np
import pytest

from nlm_synth.stats import morans_i, semivariogram, summarize_stats

from .conftest import reference_morans_i


class TestMoransI:
    def test_matches_naive_reference(self, small_field):
        assert morans_i(small_field) == pytest.approx(reference_morans_i(small_field), rel=1e-12)

    def test_matches_naive_reference_with_nans(self, small_field):
        field = small_field.copy()
        field[5:12, 20:25] = np.nan
        assert morans_i(field) == pytest.approx(reference_morans_i(field), rel=1e-12)

    def test_matches_naive_reference_on_non_square(self):
        rng = np.random.default_rng(3)
        field = rng.random((17, 29))
        assert morans_i(field) == pytest.approx(reference_morans_i(field), rel=1e-12)

    def test_smooth_field_is_strongly_positive(self, small_field):
        assert morans_i(small_field) > 0.8

    def test_white_noise_is_near_zero(self):
        noise = np.random.default_rng(0).random((128, 128))
        assert abs(morans_i(noise)) < 0.05

    def test_checkerboard_is_near_minus_one(self):
        board = np.indices((32, 32)).sum(axis=0) % 2
        assert morans_i(board.astype(float)) == pytest.approx(-1.0, abs=1e-9)

    def test_constant_field_is_nan(self):
        assert np.isnan(morans_i(np.full((8, 8), 3.0)))

    def test_too_few_valid_cells_is_nan(self):
        field = np.full((4, 4), np.nan)
        field[0, 0] = 1.0
        assert np.isnan(morans_i(field))

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            morans_i(np.arange(10.0))


class TestSemivariogram:
    def test_increases_with_lag_for_a_smooth_field(self, small_field):
        lags, gamma = semivariogram(small_field, max_lag=16, step=2, random_state=0)
        assert lags.size == gamma.size > 0
        finite = np.isfinite(gamma)
        assert gamma[finite][0] < gamma[finite][-1]

    def test_is_reproducible(self, small_field):
        _, a = semivariogram(small_field, random_state=42)
        _, b = semivariogram(small_field, random_state=42)
        np.testing.assert_array_equal(a, b)

    def test_white_noise_is_flat_at_the_variance(self):
        noise = np.random.default_rng(1).random((96, 96))
        _, gamma = semivariogram(noise, max_lag=20, step=2, n_pairs=60_000, random_state=1)
        finite = gamma[np.isfinite(gamma)]
        assert np.allclose(finite, noise.var(), rtol=0.25)

    def test_too_few_valid_cells_returns_empty(self):
        assert semivariogram(np.full((4, 4), np.nan))[0].size == 0


class TestSummarizeStats:
    def test_core_keys_and_values(self, small_field):
        stats = summarize_stats(small_field)
        assert set(stats) == {
            "mean", "variance", "std_dev", "morans_I", "n", "shape_r", "shape_c",
        }
        assert stats["mean"] == pytest.approx(small_field.mean())
        assert stats["variance"] == pytest.approx(small_field.var())
        assert stats["n"] == small_field.size
        assert stats["shape_r"], stats["shape_c"] == small_field.shape

    def test_semivar_flag_adds_keys(self, small_field):
        """`semivar=True` used to be accepted and silently ignored."""
        plain = summarize_stats(small_field)
        with_semivar = summarize_stats(small_field, semivar=True, random_state=0)
        assert "semivar_range" not in plain
        assert {"semivar_range", "semivar_sill"} <= set(with_semivar)
        assert np.isfinite(with_semivar["semivar_sill"])

    def test_nans_are_excluded(self, small_field):
        field = small_field.copy()
        field[:8, :] = np.nan
        stats = summarize_stats(field)
        assert stats["n"] == field.size - 8 * field.shape[1]
        assert stats["mean"] == pytest.approx(np.nanmean(field))

    def test_all_nan_field_is_handled(self):
        stats = summarize_stats(np.full((6, 6), np.nan))
        assert stats["n"] == 0
        assert np.isnan(stats["mean"]) and np.isnan(stats["morans_I"])
