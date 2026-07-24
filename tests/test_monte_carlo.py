"""Tests for the Monte Carlo experiment driver."""

from __future__ import annotations

import numpy as np
import pytest

from nlm_synth.monte_carlo import default_generator_grid, run_experiments

GRID = [
    {"label": "lf", "method": "perlin",
     "method_kwargs": dict(periods=(2, 2), octaves=2, lacunarity=2, persistence=0.7)},
    {"label": "hf", "method": "perlin",
     "method_kwargs": dict(periods=(8, 8), octaves=3, lacunarity=2, persistence=0.5)},
]


@pytest.fixture(scope="module")
def result(ndvi_samples):
    return run_experiments(
        ndvi_samples, nrow=64, ncol=64, generator_grid=GRID,
        coarsen_factors=(1, 2, 4), n_runs=3, random_seed=7,
    )


class TestRunExperiments:
    def test_shape_and_columns(self, result):
        df, _ = result
        assert len(df) == len(GRID) * 3 * 3  # generators x runs x factors
        assert {"label", "method", "run", "factor", "morans_I", "variance"} <= set(df.columns)

    def test_every_combination_appears_once(self, result):
        df, _ = result
        assert not df.duplicated(subset=["label", "run", "factor"]).any()

    def test_meta_records_the_settings(self, result):
        _, meta = result
        assert meta["random_seed"] == 7
        assert meta["coarsen_factors"] == [1, 2, 4]
        assert meta["nrow"] == meta["ncol"] == 64

    def test_reproducible_across_calls(self, ndvi_samples, result):
        df, _ = result
        again, _ = run_experiments(
            ndvi_samples, nrow=64, ncol=64, generator_grid=GRID,
            coarsen_factors=(1, 2, 4), n_runs=3, random_seed=7,
        )
        np.testing.assert_allclose(df["morans_I"], again["morans_I"])

    def test_different_seeds_give_different_realisations(self, ndvi_samples, result):
        df, _ = result
        other, _ = run_experiments(
            ndvi_samples, nrow=64, ncol=64, generator_grid=GRID,
            coarsen_factors=(1, 2, 4), n_runs=3, random_seed=8,
        )
        assert not np.allclose(df["morans_I"], other["morans_I"])

    def test_seeds_are_independent_of_grid_position(self, ndvi_samples, result):
        """Adding a generator must not perturb the realisations of the others.

        Seeds are drawn as a (config, run) block up front rather than
        sequentially, so results for existing generators stay comparable when
        the grid is extended.
        """
        df, _ = result
        extended_grid = GRID + [
            {"label": "mf", "method": "perlin",
             "method_kwargs": dict(periods=(4, 4), octaves=2, lacunarity=2, persistence=0.6)},
        ]
        extended, _ = run_experiments(
            ndvi_samples, nrow=64, ncol=64, generator_grid=extended_grid,
            coarsen_factors=(1, 2, 4), n_runs=3, random_seed=7,
        )
        shared = extended[extended["label"].isin(["lf", "hf"])].reset_index(drop=True)
        np.testing.assert_allclose(df["morans_I"], shared["morans_I"])

    def test_low_frequency_is_more_autocorrelated_than_high(self, result):
        """The scientific claim the package exists to measure."""
        df, _ = result
        means = df[df["factor"] == 1].groupby("label")["morans_I"].mean()
        assert means["lf"] > means["hf"]

    def test_coarsening_reduces_variance(self, result):
        df, _ = result
        by_factor = df.groupby("factor")["variance"].mean()
        assert by_factor.is_monotonic_decreasing

    def test_marginal_is_preserved_at_full_resolution(self, result, ndvi_samples):
        df, _ = result
        full = df[df["factor"] == 1]
        assert full["mean"].mean() == pytest.approx(ndvi_samples.mean(), abs=0.02)

    def test_semivar_columns_appear_when_requested(self, ndvi_samples):
        df, _ = run_experiments(
            ndvi_samples, nrow=64, ncol=64, generator_grid=GRID[:1],
            coarsen_factors=(1,), n_runs=1, semivar=True, random_seed=1,
        )
        assert {"semivar_range", "semivar_sill"} <= set(df.columns)

    def test_rejects_empty_grid(self, ndvi_samples):
        with pytest.raises(ValueError, match="empty"):
            run_experiments(ndvi_samples, generator_grid=[])


class TestDefaultGeneratorGrid:
    def test_returns_a_fresh_mutable_copy(self):
        first = default_generator_grid()
        first[0]["label"] = "mutated"
        assert default_generator_grid()[0]["label"] == "perlin_LF"

    def test_labels_are_unique(self):
        labels = [cfg["label"] for cfg in default_generator_grid()]
        assert len(labels) == len(set(labels))

    def test_cluster_config_uses_the_current_parameter_names(self):
        """`nn_prob` was renamed to `cluster_p` when the argument bug was fixed."""
        cluster = [c for c in default_generator_grid() if c["method"] == "cluster"][0]
        assert "cluster_p" in cluster["method_kwargs"]
        assert "nn_prob" not in cluster["method_kwargs"]
