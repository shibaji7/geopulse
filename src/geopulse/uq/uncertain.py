"""Core uncertainty type: ``Uncertain[T]``.

Design philosophy: UQ is baked in from day one, not bolted on later. The
:class:`Uncertain` type wraps any numeric value (scalar, array, or dataclass)
and can represent either a deterministic value or a distribution of values.

When deterministic values flow through the pipeline, there is zero overhead.
When distributions flow through, Monte-Carlo propagation kicks in via
:func:`propagate_uncertainty`.

Examples
--------
>>> from geopulse.uq.uncertain import Uncertain, propagate_uncertainty
>>> b = Uncertain(nominal=1.0, distribution="gaussian", params={"std": 0.1})
>>> out = propagate_uncertainty(lambda x: 2.0 * x, b, n_samples=200)
>>> abs(out.mean - 2.0) < 0.05
True
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar

import numpy as np

__all__ = ["Uncertain", "propagate_uncertainty"]

T = TypeVar("T")


@dataclass
class Uncertain(Generic[T]):
    """A value that may carry uncertainty information.

    Parameters
    ----------
    nominal : T
        The central / best-estimate value.
    samples : list[T] | None, optional
        Monte-Carlo samples, if uncertainty has been propagated.
        Length equals the number of MC draws.
    distribution : str, optional
        Distribution family. One of ``"deterministic"``, ``"gaussian"``,
        ``"uniform"``, ``"ensemble"``. Default: ``"deterministic"``.
    params : dict, optional
        Distribution parameters. For ``gaussian``: ``{"std": ...}``.
        For ``uniform``: ``{"low": ..., "high": ...}``.
        For ``ensemble``: empty (samples ARE the distribution).

    Notes
    -----
    :class:`Uncertain` is *not* frozen: :meth:`generate_samples` is a query
    method and does not mutate the instance, but downstream propagation code
    may attach freshly-drawn ``samples`` after construction.
    """

    nominal: T
    samples: Optional[list[T]] = None
    distribution: str = "deterministic"
    params: dict = field(default_factory=dict)

    @property
    def is_deterministic(self) -> bool:
        """Whether this value carries no uncertainty."""
        return self.distribution == "deterministic" and self.samples is None

    @property
    def n_samples(self) -> int:
        """Number of Monte-Carlo samples, or 0 if deterministic."""
        return len(self.samples) if self.samples is not None else 0

    @property
    def mean(self) -> Any:
        """Mean of ``samples``, or ``nominal`` if deterministic."""
        if self.samples is None:
            return self.nominal
        return np.mean(np.asarray(self.samples), axis=0)

    @property
    def std(self) -> Any:
        """Standard deviation of ``samples``, or zero if deterministic."""
        if self.samples is None:
            if hasattr(self.nominal, "__len__"):
                return np.zeros_like(np.asarray(self.nominal))
            return 0.0
        return np.std(np.asarray(self.samples), axis=0)

    def generate_samples(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
    ) -> list:
        """Draw ``n`` Monte-Carlo samples from the declared distribution.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        rng : numpy.random.Generator, optional
            Reproducible RNG. If ``None``, a fresh default RNG is created.

        Returns
        -------
        list
            ``n`` samples, each with the same shape/type as :attr:`nominal`.

        Raises
        ------
        ValueError
            If :attr:`distribution` is not one of the supported families.
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.distribution == "deterministic":
            return [self.nominal] * n
        if self.distribution == "gaussian":
            std = self.params.get("std", 0.0)
            shape = np.shape(self.nominal)
            return [self.nominal + rng.normal(0.0, std, size=shape) for _ in range(n)]
        if self.distribution == "uniform":
            low = self.params.get("low")
            high = self.params.get("high")
            if low is None or high is None:
                raise ValueError("uniform distribution requires 'low' and 'high'")
            shape = np.shape(self.nominal)
            return [rng.uniform(low, high, size=shape) for _ in range(n)]
        if self.distribution == "ensemble":
            if self.samples is None:
                raise ValueError("ensemble distribution requires pre-populated samples")
            # Resample with replacement.
            idx = rng.integers(0, len(self.samples), size=n)
            return [self.samples[i] for i in idx]
        raise ValueError(f"Unknown distribution: {self.distribution!r}")


def propagate_uncertainty(
    func: Callable[..., Any],
    *args: Any,
    n_samples: int = 100,
    seed: int = 42,
    **kwargs: Any,
) -> Uncertain:
    """Propagate uncertainty through a function via Monte Carlo.

    Any argument that is an :class:`Uncertain` with a non-deterministic
    distribution is sampled; deterministic arguments pass through unchanged.

    Parameters
    ----------
    func : callable
        The function to propagate through.
    *args
        Positional arguments — may include :class:`Uncertain` values.
    n_samples : int, optional
        Number of Monte-Carlo samples. Default: 100.
    seed : int, optional
        Random seed for reproducibility. Default: 42.
    **kwargs
        Keyword arguments — may include :class:`Uncertain` values.

    Returns
    -------
    Uncertain
        Result with ``distribution="ensemble"`` and ``samples`` populated.

    Examples
    --------
    >>> u = Uncertain(nominal=1.0, distribution="gaussian", params={"std": 0.1})
    >>> out = propagate_uncertainty(lambda x: x * 2, u, n_samples=100)
    >>> out.n_samples
    100
    """
    rng = np.random.default_rng(seed)

    def _sample(v: Any) -> list:
        if isinstance(v, Uncertain) and not v.is_deterministic:
            return v.generate_samples(n_samples, rng)
        val = v.nominal if isinstance(v, Uncertain) else v
        return [val] * n_samples

    arg_samples = [_sample(a) for a in args]
    kwarg_samples = {k: _sample(v) for k, v in kwargs.items()}

    results: list[Any] = []
    for i in range(n_samples):
        sample_args = [s[i] for s in arg_samples]
        sample_kwargs = {k: v[i] for k, v in kwarg_samples.items()}
        results.append(func(*sample_args, **sample_kwargs))

    nominal_args = [a.nominal if isinstance(a, Uncertain) else a for a in args]
    nominal_kwargs = {k: (v.nominal if isinstance(v, Uncertain) else v) for k, v in kwargs.items()}
    nominal = func(*nominal_args, **nominal_kwargs)

    return Uncertain(nominal=nominal, samples=results, distribution="ensemble")
