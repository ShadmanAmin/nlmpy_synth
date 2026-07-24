"""Tests for field generation and quantile mapping."""

from __future__ import annotations

import numpy as np
import pytest

from nlm_synth.generators import (
    perlin_field,
    perlin_internal_dim,
    random_cluster_binary,
    rank_map_to_distribution,
    synth_ndvi_from_distribution,
)


class TestPerlinField:
    def test_shape_and_range(self):
        field = perlin_field(48, 32, periods=(4, 4), octaves=3, seed=1)
        assert field.shape == (48, 32)
        assert field.min() >= 0.0 and field.max() <= 1.0

    @pytest.mark.parametrize("shape", [(48, 32), (32, 48), (100, 37), (7, 5)])
    def test_non_square_shapes_work(self, shape):
        """nlmpy crashes on some non-square requests; the wrapper works around it.

        Its extractRandomArrayFromSquareArray calls
        ``np.random.choice(range(dim - nRow))``, which raises whenever the
        padded square happens to equal the requested row count.
        """
        assert perlin_field(*shape, periods=(4, 4), octaves=3, seed=2).shape == shape

    def test_same_seed_reproduces_field(self):
        a = perlin_field(32, 32, periods=(2, 2), octaves=2, seed=7)
        b = perlin_field(32, 32, periods=(2, 2), octaves=2, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = perlin_field(32, 32, periods=(2, 2), octaves=2, seed=7)
        b = perlin_field(32, 32, periods=(2, 2), octaves=2, seed=8)
        assert not np.array_equal(a, b)

    def test_does_not_disturb_global_rng(self):
        """Seeding for nlmpy must not leak into the caller's RNG stream."""
        np.random.seed(999)
        expected = np.random.random(5)

        np.random.seed(999)
        perlin_field(32, 32, periods=(2, 2), octaves=2, seed=123)
        actual = np.random.random(5)

        np.testing.assert_array_equal(expected, actual)

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(nrow=0, ncol=8),
            dict(nrow=8, ncol=8, periods=(0, 4)),
            dict(nrow=8, ncol=8, lacunarity=0),
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs):
        with pytest.raises(ValueError):
            perlin_field(**{"nrow": 8, "ncol": 8, **kwargs})

    def test_more_periods_gives_finer_structure(self):
        """Higher spatial frequency must reduce neighbour autocorrelation."""
        from nlm_synth.stats import morans_i

        coarse = morans_i(perlin_field(128, 128, periods=(2, 2), octaves=1, seed=3))
        fine = morans_i(perlin_field(128, 128, periods=(16, 16), octaves=1, seed=3))
        assert coarse > fine


class TestPerlinInternalDim:
    def test_known_blowup_case_is_detected(self):
        """The combination that used to trigger an 11 GB allocation."""
        assert perlin_internal_dim((12, 12), octaves=6, lacunarity=5) == 37_500

    def test_single_octave_needs_only_the_period_lcm(self):
        assert perlin_internal_dim((4, 6), octaves=1, lacunarity=2) == 12


class TestRandomClusterBinary:
    def test_output_is_binary(self):
        field = random_cluster_binary(64, 64, p=0.5, cluster_p=0.55, seed=2)
        assert set(np.unique(field)).issubset({0.0, 1.0})

    @pytest.mark.parametrize("p", [0.25, 0.5, 0.75])
    def test_p_controls_the_proportion_of_ones(self, p):
        """The requested proportion is honoured; previously `p` was ignored entirely."""
        field = random_cluster_binary(96, 96, p=p, cluster_p=0.55, seed=5)
        assert field.mean() == pytest.approx(p, abs=0.06)

    def test_degenerate_proportions(self):
        assert random_cluster_binary(16, 16, p=0.0, seed=1).sum() == 0
        assert random_cluster_binary(16, 16, p=1.0, seed=1).all()

    def test_rejects_invalid_arguments(self):
        with pytest.raises(ValueError):
            random_cluster_binary(16, 16, p=1.5)
        with pytest.raises(ValueError):
            random_cluster_binary(16, 16, cluster_p=0.0)
        with pytest.raises(ValueError):
            random_cluster_binary(16, 16, neighbourhood="nonsense")


class TestRankMapToDistribution:
    def test_reproduces_the_target_marginal(self, ndvi_samples):
        """The whole point of the package: output quantiles match the input's."""
        field = perlin_field(128, 128, periods=(4, 4), octaves=4, seed=11)
        mapped = rank_map_to_distribution(field, ndvi_samples)

        for q in (1, 5, 25, 50, 75, 95, 99):
            assert np.percentile(mapped, q) == pytest.approx(
                np.percentile(ndvi_samples, q), abs=0.01
            )

    def test_preserves_spatial_ordering(self, small_field, ndvi_samples):
        mapped = rank_map_to_distribution(small_field, ndvi_samples)
        order_before = np.argsort(small_field.ravel(), kind="stable")
        order_after = np.argsort(mapped.ravel(), kind="stable")
        np.testing.assert_array_equal(order_before, order_after)

    def test_is_invariant_to_monotone_rescaling(self, small_field, ndvi_samples):
        """Only ranks matter, so an affine rescale of the field changes nothing."""
        a = rank_map_to_distribution(small_field, ndvi_samples)
        b = rank_map_to_distribution(small_field * 17.0 - 4.0, ndvi_samples)
        np.testing.assert_allclose(a, b)

    def test_nan_cells_are_preserved(self, ndvi_samples):
        field = perlin_field(32, 32, periods=(2, 2), octaves=2, seed=4)
        field[3:6, 3:6] = np.nan
        mapped = rank_map_to_distribution(field, ndvi_samples)
        assert np.isnan(mapped[3:6, 3:6]).all()
        assert np.isfinite(mapped[10:, 10:]).all()

    def test_nan_samples_are_ignored(self, small_field):
        samples = np.array([0.1, 0.5, np.nan, 0.9])
        mapped = rank_map_to_distribution(small_field, samples)
        assert np.isfinite(mapped).all()
        assert set(np.unique(mapped)).issubset({0.1, 0.5, 0.9})

    def test_rejects_empty_samples(self, small_field):
        with pytest.raises(ValueError):
            rank_map_to_distribution(small_field, np.array([np.nan, np.nan]))


class TestSynthNdviFromDistribution:
    @pytest.mark.parametrize("method", ["perlin", "cluster"])
    def test_shape_and_marginal(self, method, ndvi_samples):
        field = synth_ndvi_from_distribution(64, 64, ndvi_samples, method=method, seed=3)
        assert field.shape == (64, 64)
        assert field.mean() == pytest.approx(ndvi_samples.mean(), abs=0.02)

    def test_reproducible(self, ndvi_samples):
        a = synth_ndvi_from_distribution(48, 48, ndvi_samples, seed=21)
        b = synth_ndvi_from_distribution(48, 48, ndvi_samples, seed=21)
        np.testing.assert_array_equal(a, b)

    def test_cluster_output_is_continuous_not_two_valued(self, ndvi_samples):
        """Blending Perlin noise into the binary field must break the ties."""
        field = synth_ndvi_from_distribution(
            64, 64, ndvi_samples, method="cluster", method_kwargs={"p": 0.5}, seed=6
        )
        assert np.unique(field).size > 100

    def test_rejects_unknown_method(self, ndvi_samples):
        with pytest.raises(ValueError, match="Unknown method"):
            synth_ndvi_from_distribution(16, 16, ndvi_samples, method="banana")

    def test_rejects_unknown_cluster_kwargs(self, ndvi_samples):
        with pytest.raises(TypeError, match="unexpected method_kwargs"):
            synth_ndvi_from_distribution(
                16, 16, ndvi_samples, method="cluster", method_kwargs={"nn_prob": 0.6}
            )
