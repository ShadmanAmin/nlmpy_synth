"""Tests for block-mean coarsening."""

from __future__ import annotations

import numpy as np
import pytest

from nlm_synth.coarsen import block_reduce_mean, multi_scale_coarsen


class TestBlockReduceMean:
    def test_known_case(self):
        arr = np.arange(16, dtype=float).reshape(4, 4)
        np.testing.assert_allclose(
            block_reduce_mean(arr, 2), [[2.5, 4.5], [10.5, 12.5]]
        )

    @pytest.mark.parametrize("factor", [0, 1])
    def test_factor_at_most_one_copies(self, factor, small_field):
        out = block_reduce_mean(small_field, factor)
        np.testing.assert_array_equal(out, small_field)
        assert out is not small_field

    def test_trims_partial_blocks(self):
        arr = np.ones((7, 5))
        assert block_reduce_mean(arr, 3).shape == (2, 1)

    def test_preserves_the_mean_of_a_divisible_grid(self, small_field):
        assert block_reduce_mean(small_field, 4).mean() == pytest.approx(small_field.mean())

    def test_reduces_variance(self, small_field):
        """Averaging within blocks removes within-block variance."""
        assert block_reduce_mean(small_field, 8).var() < small_field.var()

    def test_nans_are_ignored_within_a_block(self):
        arr = np.array([[1.0, 2.0], [3.0, np.nan]])
        assert block_reduce_mean(arr, 2)[0, 0] == pytest.approx(2.0)

    def test_fully_nan_block_is_nan(self):
        assert np.isnan(block_reduce_mean(np.full((2, 2), np.nan), 2)[0, 0])

    def test_factor_larger_than_array_raises(self):
        with pytest.raises(ValueError, match="no full block fits"):
            block_reduce_mean(np.ones((3, 3)), 4)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            block_reduce_mean(np.arange(8.0), 2)


class TestMultiScaleCoarsen:
    def test_sorted_deduplicated_and_filtered(self, small_field):
        out = multi_scale_coarsen(small_field, [4, 1, 2, 4, 0, -3])
        assert [factor for factor, _ in out] == [1, 2, 4]

    def test_shapes_shrink_with_factor(self, small_field):
        for factor, arr in multi_scale_coarsen(small_field, [1, 2, 4, 8]):
            assert arr.shape == (small_field.shape[0] // factor, small_field.shape[1] // factor)

    def test_oversized_factors_are_skipped_not_raised(self, small_field):
        """A too-large factor should drop out quietly rather than kill a long MC run."""
        factors = [factor for factor, _ in multi_scale_coarsen(small_field, [2, 1000])]
        assert factors == [2]
