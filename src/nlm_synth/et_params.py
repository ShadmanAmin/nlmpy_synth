"""Latin-hypercube sampling of two-component Gaussian mixture parameters.

Land-surface variables such as NDVI or radiometric temperature are often
bimodal -- vegetated versus bare, sunlit versus shaded. This module describes
such a variable by the bounds of a two-component Gaussian mixture, draws
parameter sets from those bounds by Latin hypercube sampling, and turns any
draw into a frozen distribution that can be sampled to build the target
marginal for :mod:`nlm_synth.generators`.

Each parameter can also be collapsed to a single Gaussian with the mixture's
mean and standard deviation, which makes it easy to ask what is lost by
approximating a bimodal surface as unimodal.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.stats as stats
from scipy.stats import qmc

__all__ = [
    "ETParameter",
    "MixtureETParameter",
    "create_et_parameters",
    "fit_gaussian_mixture",
    "gsmax_mmol_to_ms",
    "find_matching_rows",
]

Bounds = tuple[float, float]


class ETParameter:
    """Bounds for a two-component Gaussian mixture, sampled by Latin hypercube.

    Parameters
    ----------
    name:
        Label for the variable, e.g. ``"NDVI"``.
    mu1_bounds, mu2_bounds:
        Ranges for the two component means.
    w1_bounds:
        Range for the weight of the first component; the second weight is
        ``1 - w1``.
    sigma1_bounds, sigma2_bounds:
        Ranges for the two component standard deviations.

    Attributes
    ----------
    mu1, mu2, w1, w2, sigma1, sigma2:
        Arrays of sampled values, or None before :meth:`lhs_sample` is called.
    """

    def __init__(
        self,
        name: str,
        mu1_bounds: Bounds,
        mu2_bounds: Bounds,
        w1_bounds: Bounds,
        sigma1_bounds: Bounds,
        sigma2_bounds: Bounds,
    ):
        self.name = name
        self.mu1_bounds = mu1_bounds
        self.mu2_bounds = mu2_bounds
        self.w1_bounds = w1_bounds
        self.sigma1_bounds = sigma1_bounds
        self.sigma2_bounds = sigma2_bounds

        # Declared up front so `is None` checks work on a fresh instance.
        self.mu1: np.ndarray | None = None
        self.mu2: np.ndarray | None = None
        self.w1: np.ndarray | None = None
        self.w2: np.ndarray | None = None
        self.sigma1: np.ndarray | None = None
        self.sigma2: np.ndarray | None = None

    @property
    def is_sampled(self) -> bool:
        """True once :meth:`lhs_sample` has populated the parameter arrays."""
        return self.mu1 is not None

    def lhs_sample(self, n_samples: int = 1, seed: int | None = None) -> np.ndarray:
        """Draw ``n_samples`` parameter sets by Latin hypercube sampling.

        Parameters
        ----------
        n_samples:
            Number of parameter sets to draw.
        seed:
            Seed for the sampler; pass an integer for reproducible draws.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_samples, 5)`` with columns
            ``(mu1, mu2, w1, sigma1, sigma2)``. The values are also stored on
            the instance.
        """
        sampler = qmc.LatinHypercube(d=5, seed=seed)
        unit = sampler.random(n=int(n_samples))

        lows = [
            self.mu1_bounds[0],
            self.mu2_bounds[0],
            self.w1_bounds[0],
            self.sigma1_bounds[0],
            self.sigma2_bounds[0],
        ]
        highs = [
            self.mu1_bounds[1],
            self.mu2_bounds[1],
            self.w1_bounds[1],
            self.sigma1_bounds[1],
            self.sigma2_bounds[1],
        ]
        sample = qmc.scale(unit, lows, highs)

        self.mu1, self.mu2, self.w1, self.sigma1, self.sigma2 = sample.T
        self.w2 = 1.0 - self.w1
        return sample

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} {self.name}: "
            f"mu1={self.mu1_bounds}, mu2={self.mu2_bounds}, w1={self.w1_bounds}, "
            f"sigma1={self.sigma1_bounds}, sigma2={self.sigma2_bounds}>"
        )


class MixtureETParameter(ETParameter):
    """An :class:`ETParameter` that can emit frozen distributions."""

    def create_dist(self, dist_type: str = "mixture", sample_index: int = 0):
        """Build a distribution from one sampled parameter set.

        Parameters
        ----------
        dist_type:
            ``'mixture'`` for the two-component mixture, or ``'normal'`` for a
            single Gaussian with the same mean and standard deviation -- the
            unimodal approximation to compare against.
        sample_index:
            Which sampled parameter set to use. :meth:`lhs_sample` is called
            automatically if no draw has been made yet.

        Returns
        -------
        A frozen ``scipy.stats`` distribution exposing ``.sample()``.
        """
        if dist_type not in ("normal", "mixture"):
            raise ValueError("dist_type must be 'normal' or 'mixture'")

        # Previously this guard read `self.w2`, which was never initialised in
        # __init__, so a fresh instance raised AttributeError instead of
        # auto-sampling as intended.
        if not self.is_sampled:
            self.lhs_sample(n_samples=1)

        if not 0 <= sample_index < len(self.mu1):
            raise IndexError(
                f"sample_index {sample_index} out of range for "
                f"{len(self.mu1)} sampled parameter set(s)"
            )

        component_1 = stats.Normal(mu=self.mu1[sample_index], sigma=self.sigma1[sample_index])
        component_2 = stats.Normal(mu=self.mu2[sample_index], sigma=self.sigma2[sample_index])
        mixture = stats.Mixture(
            [component_1, component_2],
            weights=[self.w1[sample_index], self.w2[sample_index]],
        )

        if dist_type == "mixture":
            return mixture
        return stats.Normal(mu=mixture.mean(), sigma=mixture.standard_deviation())


def create_et_parameters() -> dict[str, MixtureETParameter]:
    """Reference mixture bounds for the surface-energy-balance inputs.

    Returns
    -------
    dict
        Keyed by variable name: ``Tr`` (radiometric temperature, K), ``Alb``
        (albedo), ``NDVI``, ``P`` (air pressure, Pa), ``Ta`` (air temperature,
        K), ``Sdn`` (incoming shortwave, W/m2) and ``Ldn`` (incoming longwave,
        W/m2).

    Notes
    -----
    This previously returned an 8-tuple whose last element duplicated the other
    seven as a dict; callers had to unpack positionally and keep the order in
    sync. A single dict keyed by name is equivalent and harder to misuse.
    """
    specs = {
        "Tr": dict(mu1_bounds=(280, 300), mu2_bounds=(300, 320), w1_bounds=(0.3, 0.7),
                   sigma1_bounds=(1, 10), sigma2_bounds=(1, 15)),
        "Alb": dict(mu1_bounds=(0.1, 0.5), mu2_bounds=(0.5, 0.9), w1_bounds=(0.3, 0.7),
                    sigma1_bounds=(0.01, 0.03), sigma2_bounds=(0.01, 0.03)),
        "NDVI": dict(mu1_bounds=(0.1, 0.5), mu2_bounds=(0.5, 0.9), w1_bounds=(0.2, 0.8),
                     sigma1_bounds=(0.02, 0.05), sigma2_bounds=(0.02, 0.05)),
        "P": dict(mu1_bounds=(90_000, 96_000), mu2_bounds=(96_000, 110_000), w1_bounds=(0.3, 0.7),
                  sigma1_bounds=(1_000, 3_000), sigma2_bounds=(1_000, 3_000)),
        "Ta": dict(mu1_bounds=(270, 290), mu2_bounds=(290, 310), w1_bounds=(0.3, 0.7),
                   sigma1_bounds=(1, 5), sigma2_bounds=(1, 5)),
        "Sdn": dict(mu1_bounds=(200, 600), mu2_bounds=(600, 1000), w1_bounds=(0.4, 0.6),
                    sigma1_bounds=(10, 30), sigma2_bounds=(10, 30)),
        "Ldn": dict(mu1_bounds=(100, 250), mu2_bounds=(250, 500), w1_bounds=(0.3, 0.7),
                    sigma1_bounds=(5, 15), sigma2_bounds=(5, 15)),
    }
    return {name: MixtureETParameter(name=name, **spec) for name, spec in specs.items()}


def gsmax_mmol_to_ms(g_mmol: float, t_air: float, p_air: float) -> float:
    """Convert stomatal conductance from mmol/m2/s to m/s.

    Parameters
    ----------
    g_mmol:
        Conductance in mmol m-2 s-1.
    t_air:
        Air temperature in K.
    p_air:
        Air pressure in Pa.
    """
    gas_constant = 8.314472  # J / (mol K)
    return (g_mmol / 1000.0) * (gas_constant * t_air / p_air)


def fit_gaussian_mixture(data: np.ndarray, n_components: int = 2, seed: int | None = 0):
    """Fit a Gaussian mixture to 1-D data and describe the result.

    Parameters
    ----------
    data:
        1-D array; non-finite values are dropped.
    n_components:
        Number of mixture components.
    seed:
        Seed for the EM initialisation, for reproducible fits.

    Returns
    -------
    dict
        ``means``, ``variances``, ``weights`` (arrays sorted by increasing
        mean) and ``skewness`` of the fitted mixture.

    Notes
    -----
    Requires scikit-learn, which is an optional dependency of this package
    (``pip install nlm-synth[fit]``).
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "fit_gaussian_mixture requires scikit-learn; "
            "install it with `pip install nlm-synth[fit]`"
        ) from exc

    values = np.asarray(data, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size < n_components:
        raise ValueError(f"need at least {n_components} finite values to fit")

    gmm = GaussianMixture(
        n_components=n_components, covariance_type="diag", random_state=seed
    ).fit(values.reshape(-1, 1))

    means = gmm.means_.ravel()
    variances = gmm.covariances_.ravel()
    weights = gmm.weights_.ravel()

    # Sort by mean so component identity is stable across fits; EM labels are
    # otherwise arbitrary, which makes results irreproducible run to run.
    order = np.argsort(means)
    means, variances, weights = means[order], variances[order], weights[order]

    mixture = stats.Mixture(
        [stats.Normal(mu=m, sigma=np.sqrt(v)) for m, v in zip(means, variances, strict=True)],
        weights=list(weights),
    )
    return {
        "means": means,
        "variances": variances,
        "weights": weights,
        "skewness": float(mixture.skewness()),
    }


def find_matching_rows(df, columns: Sequence[str], values: Sequence[float], tol: float = 1e-6):
    """Select rows of ``df`` whose ``columns`` match ``values`` within ``tol``.

    Parameters
    ----------
    df:
        DataFrame to filter.
    columns:
        Column names to compare.
    values:
        Target value per column, in the same order.
    tol:
        Absolute tolerance passed to :func:`numpy.isclose`.

    Notes
    -----
    Replaces the earlier ``find_gaussian_instances`` / ``find_instances`` pair,
    which hard-coded two different fixed column sets and raised ``KeyError`` on
    any table that did not use those exact names.
    """
    if len(columns) != len(values):
        raise ValueError("columns and values must have the same length")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame is missing column(s): {missing}")

    mask = np.ones(len(df), dtype=bool)
    for column, value in zip(columns, values, strict=True):
        mask &= np.isclose(df[column].to_numpy(dtype=float), float(value), atol=tol)
    return df[mask]
