"""Tests for the mixture-parameter sampling utilities."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy", reason="requires the optional fit extra")

from nlm_synth.et_params import (  # noqa: E402
    ETParameter,
    MixtureETParameter,
    create_et_parameters,
    find_matching_rows,
    gsmax_mmol_to_ms,
)

BOUNDS = dict(
    mu1_bounds=(0.1, 0.5),
    mu2_bounds=(0.5, 0.9),
    w1_bounds=(0.2, 0.8),
    sigma1_bounds=(0.02, 0.05),
    sigma2_bounds=(0.02, 0.05),
)

_HAS_MIXTURE = hasattr(__import__("scipy.stats", fromlist=["stats"]), "Mixture")
requires_mixture = pytest.mark.skipif(
    not _HAS_MIXTURE, reason="scipy.stats.Mixture requires SciPy >= 1.15"
)


class TestLhsSample:
    def test_shape_and_storage(self):
        param = ETParameter(name="NDVI", **BOUNDS)
        sample = param.lhs_sample(n_samples=8, seed=0)
        assert sample.shape == (8, 5)
        assert param.mu1.shape == (8,)
        assert param.is_sampled

    def test_respects_bounds(self):
        param = ETParameter(name="NDVI", **BOUNDS)
        param.lhs_sample(n_samples=32, seed=1)
        assert (param.mu1 >= 0.1).all() and (param.mu1 <= 0.5).all()
        assert (param.mu2 >= 0.5).all() and (param.mu2 <= 0.9).all()
        assert (param.sigma1 > 0).all() and (param.sigma2 > 0).all()

    def test_weights_sum_to_one(self):
        param = ETParameter(name="NDVI", **BOUNDS)
        param.lhs_sample(n_samples=16, seed=2)
        np.testing.assert_allclose(param.w1 + param.w2, 1.0)

    def test_seed_makes_draws_reproducible(self):
        a = ETParameter(name="x", **BOUNDS).lhs_sample(4, seed=42)
        b = ETParameter(name="x", **BOUNDS).lhs_sample(4, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_fresh_instance_reports_unsampled(self):
        assert not ETParameter(name="x", **BOUNDS).is_sampled


@requires_mixture
class TestCreateDist:
    def test_autosamples_on_a_fresh_instance(self):
        """This used to raise AttributeError because `w2` was never initialised."""
        param = MixtureETParameter(name="NDVI", **BOUNDS)
        dist = param.create_dist(dist_type="mixture")
        assert param.is_sampled
        assert np.isfinite(dist.mean())

    def test_normal_matches_the_mixture_moments(self):
        param = MixtureETParameter(name="NDVI", **BOUNDS)
        param.lhs_sample(n_samples=1, seed=3)
        mixture = param.create_dist("mixture")
        normal = param.create_dist("normal")
        assert normal.mean() == pytest.approx(mixture.mean(), rel=1e-9)
        assert normal.standard_deviation() == pytest.approx(
            mixture.standard_deviation(), rel=1e-9
        )

    def test_samples_are_usable_as_a_target_marginal(self):
        from nlm_synth.generators import synth_ndvi_from_distribution

        param = MixtureETParameter(name="NDVI", **BOUNDS)
        param.lhs_sample(n_samples=1, seed=4)
        samples = np.clip(param.create_dist("mixture").sample(20_000), -0.2, 1.0)

        field = synth_ndvi_from_distribution(48, 48, samples, seed=1)
        assert field.mean() == pytest.approx(np.mean(samples), abs=0.03)

    def test_rejects_bad_dist_type(self):
        with pytest.raises(ValueError, match="dist_type"):
            MixtureETParameter(name="x", **BOUNDS).create_dist("uniform")

    def test_rejects_out_of_range_sample_index(self):
        param = MixtureETParameter(name="x", **BOUNDS)
        param.lhs_sample(n_samples=2, seed=0)
        with pytest.raises(IndexError):
            param.create_dist("mixture", sample_index=5)


class TestCreateEtParameters:
    def test_returns_all_variables_keyed_by_name(self):
        params = create_et_parameters()
        assert set(params) == {"Tr", "Alb", "NDVI", "P", "Ta", "Sdn", "Ldn"}
        assert all(isinstance(p, MixtureETParameter) for p in params.values())

    def test_bounds_are_ordered_low_then_high(self):
        for param in create_et_parameters().values():
            assert param.mu1_bounds[0] < param.mu1_bounds[1] <= param.mu2_bounds[1]


class TestGsmaxConversion:
    def test_known_value(self):
        # 1000 mmol/m2/s at 300 K and 100000 Pa -> R*T/P m/s
        assert gsmax_mmol_to_ms(1000.0, 300.0, 100_000.0) == pytest.approx(
            8.314472 * 300.0 / 100_000.0
        )

    def test_scales_linearly_with_conductance(self):
        a = gsmax_mmol_to_ms(500.0, 290.0, 95_000.0)
        b = gsmax_mmol_to_ms(1000.0, 290.0, 95_000.0)
        assert b == pytest.approx(2 * a)


class TestFindMatchingRows:
    @pytest.fixture
    def frame(self):
        import pandas as pd

        return pd.DataFrame({"mean": [0.1, 0.2, 0.3], "std": [0.01, 0.02, 0.03]})

    def test_selects_the_matching_row(self, frame):
        found = find_matching_rows(frame, ["mean", "std"], [0.2, 0.02])
        assert len(found) == 1 and found.index[0] == 1

    def test_returns_empty_when_nothing_matches(self, frame):
        assert len(find_matching_rows(frame, ["mean"], [9.9])) == 0

    def test_tolerance_widens_the_match(self, frame):
        assert len(find_matching_rows(frame, ["mean"], [0.2], tol=0.15)) == 3

    def test_reports_missing_columns(self, frame):
        with pytest.raises(KeyError, match="missing column"):
            find_matching_rows(frame, ["nope"], [1.0])

    def test_rejects_mismatched_lengths(self, frame):
        with pytest.raises(ValueError, match="same length"):
            find_matching_rows(frame, ["mean", "std"], [0.2])


class TestFitGaussianMixture:
    def test_recovers_two_known_modes(self):
        pytest.importorskip("sklearn", reason="requires the optional fit extra")
        if not _HAS_MIXTURE:
            pytest.skip("scipy.stats.Mixture requires SciPy >= 1.15")
        from nlm_synth.et_params import fit_gaussian_mixture

        rng = np.random.default_rng(0)
        data = np.hstack([rng.normal(0.2, 0.03, 5_000), rng.normal(0.7, 0.05, 5_000)])
        fit = fit_gaussian_mixture(data, seed=0)

        # Components are returned sorted by mean, so identity is stable.
        assert fit["means"][0] == pytest.approx(0.2, abs=0.02)
        assert fit["means"][1] == pytest.approx(0.7, abs=0.02)
        assert fit["weights"].sum() == pytest.approx(1.0)
