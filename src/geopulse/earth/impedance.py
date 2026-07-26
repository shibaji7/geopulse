"""Surface-impedance abstraction — the polymorphism spine of GeoPulse.

The surface impedance ``Z`` relates the horizontal magnetic field ``B`` to the
geoelectric field ``E``. Its mathematical rank depends on Earth-model
complexity:

*   **1-D** — ``Z(ω)`` is a complex **SCALAR** per frequency::

        E_x =  (Z / μ₀) · B_y,    E_y = -(Z / μ₀) · B_x

*   **2-D** — ``Z̃(ω)`` is a **2×2 TENSOR** per frequency::

        E_i(ω) = Σ_j Z_ij(ω) · B_j(ω) / μ₀

*   **3-D** — ``Z̃(r, r', ω)`` is a spatial **KERNEL** (convolution)::

        E(r, ω) = ∬ Z̃(r, r', ω) · B(r', ω) d²r' / μ₀

All three implement :meth:`Impedance.apply` and return ``(Ex_f, Ey_f)``. The
:mod:`~geopulse.efield` layer calls ``apply()`` blindly.

HDF5 storage
------------
=====  =========================  ==================
Rank   Dataset shape              Recommended chunk
=====  =========================  ==================
1-D    ``(n_freq,)``              full
2-D    ``(n_freq, 2, 2)``         full
3-D    ``(n_freq, n_r, n_r, 2, 2)``  chunk on freq
=====  =========================  ==================

References
----------
.. [1] Wait, J. R. (1954). On the relation between telluric currents and the
   Earth's magnetic field. Geophysics, 19(2), 281-289.
.. [2] Boteler, D. H. (2014). Methodology for simulation of geomagnetically
   induced currents in power systems. J. Space Weather Space Clim., 4, A21.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np

from geopulse.constants import MU_0
from geopulse.exceptions import NotImplementedYetError, ShapeMismatchError

if TYPE_CHECKING:  # pragma: no cover
    import h5py

__all__ = [
    "Impedance",
    "ScalarImpedance",
    "TensorImpedance",
    "KernelImpedance",
]


class Impedance(abc.ABC):
    """Abstract base class for surface impedance.

    Parameters
    ----------
    freqs_Hz : numpy.ndarray
        Frequency array in Hz. Cast internally to ``float64``. Shape:
        ``(n_freqs,)``.

    Attributes
    ----------
    freqs_Hz : numpy.ndarray
        The stored frequency array.
    """

    def __init__(self, freqs_Hz: np.ndarray) -> None:
        self.freqs_Hz = np.asarray(freqs_Hz, dtype=np.float64)

    @abc.abstractmethod
    def apply(
        self,
        Bx_f: np.ndarray,
        By_f: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the impedance to frequency-domain B-field, returning E-field.

        Parameters
        ----------
        Bx_f : numpy.ndarray
            FFT of ``Bx`` in Tesla. Shape: ``(n_freqs,)``.
        By_f : numpy.ndarray
            FFT of ``By`` in Tesla. Shape: ``(n_freqs,)``.

        Returns
        -------
        Ex_f : numpy.ndarray
            FFT of ``Ex`` in V/m. Shape matches subclass semantics.
        Ey_f : numpy.ndarray
            FFT of ``Ey`` in V/m. Shape matches subclass semantics.
        """

    @property
    @abc.abstractmethod
    def rank(self) -> int:
        """Rank marker: 0 for scalar (1-D), 2 for tensor (2-D), 4 for kernel (3-D)."""

    @abc.abstractmethod
    def to_hdf5(self, group: "h5py.Group") -> None:
        """Serialise impedance data to an HDF5 group."""

    @classmethod
    @abc.abstractmethod
    def from_hdf5(cls, group: "h5py.Group") -> "Impedance":
        """Deserialise impedance data from an HDF5 group."""


class ScalarImpedance(Impedance):
    """1-D scalar impedance ``Z(ω)``.

    For a horizontally layered Earth under plane-wave excitation, the surface
    impedance is a single complex number per frequency. This is the standard
    magnetotelluric relation for the 1-D case.

    Parameters
    ----------
    freqs_Hz : numpy.ndarray
        Frequency array in Hz. Shape: ``(n_freqs,)``.
    Z_values : numpy.ndarray
        Complex impedance values in Ohms. Shape: ``(n_freqs,)``. Cast to
        ``complex128`` on construction.

    Raises
    ------
    ShapeMismatchError
        If ``Z_values.shape`` does not equal ``freqs_Hz.shape``.

    Notes
    -----
    The applied relation is [1]_::

        E_x(ω) =  (Z(ω) / μ₀) · B_y(ω)
        E_y(ω) = -(Z(ω) / μ₀) · B_x(ω)

    Examples
    --------
    >>> import numpy as np
    >>> freqs = np.array([1e-3, 1e-2, 1e-1])
    >>> Z = np.array([1+1j, 3+3j, 10+10j])
    >>> imp = ScalarImpedance(freqs, Z)
    >>> Ex, Ey = imp.apply(np.ones(3), np.ones(3))
    >>> Ex.shape
    (3,)
    """

    def __init__(self, freqs_Hz: np.ndarray, Z_values: np.ndarray) -> None:
        super().__init__(freqs_Hz)
        self.Z_values = np.asarray(Z_values, dtype=np.complex128)
        if self.Z_values.shape != self.freqs_Hz.shape:
            raise ShapeMismatchError(
                f"Z_values shape {self.Z_values.shape} does not match "
                f"freqs_Hz shape {self.freqs_Hz.shape}"
            )

    def apply(
        self,
        Bx_f: np.ndarray,
        By_f: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the 1-D plane-wave MT relation ``E = (Z/μ₀) · B_perp``.

        Parameters
        ----------
        Bx_f, By_f : numpy.ndarray
            FFT of the horizontal B-field components in Tesla. Shape:
            ``(n_freqs,)``.

        Returns
        -------
        Ex_f, Ey_f : numpy.ndarray
            FFT of the horizontal E-field components in V/m. Shape:
            ``(n_freqs,)``.

        Raises
        ------
        ShapeMismatchError
            If ``Bx_f`` or ``By_f`` do not have the same length as
            :attr:`freqs_Hz`.
        """
        Bx_f = np.asarray(Bx_f)
        By_f = np.asarray(By_f)
        if Bx_f.shape != self.freqs_Hz.shape or By_f.shape != self.freqs_Hz.shape:
            raise ShapeMismatchError(
                f"B-field shapes {Bx_f.shape}, {By_f.shape} do not match "
                f"impedance freqs_Hz shape {self.freqs_Hz.shape}"
            )
        Z_over_mu0 = self.Z_values / MU_0
        Ex_f = Z_over_mu0 * By_f
        Ey_f = -Z_over_mu0 * Bx_f
        return Ex_f, Ey_f

    @property
    def rank(self) -> int:  # noqa: D401
        """Return ``0`` — scalar per frequency."""
        return 0

    def to_hdf5(self, group: "h5py.Group") -> None:
        """Serialise to an HDF5 group with a schema-version attribute.

        Layout::

            group/
              attrs:
                impedance_type = "scalar_1d"
                schema_version = 1
              freqs_Hz         dataset
              Z_values         dataset (complex128)
        """
        group.attrs["impedance_type"] = "scalar_1d"
        group.attrs["schema_version"] = 1
        group.create_dataset("freqs_Hz", data=self.freqs_Hz)
        group.create_dataset("Z_values", data=self.Z_values)

    @classmethod
    def from_hdf5(cls, group: "h5py.Group") -> "ScalarImpedance":
        """Read a :class:`ScalarImpedance` from an HDF5 group."""
        return cls(
            freqs_Hz=group["freqs_Hz"][()],
            Z_values=group["Z_values"][()],
        )


class TensorImpedance(Impedance):
    r"""2-D magnetotelluric impedance tensor ``Z̃(ω)`` — 2×2 per frequency.

    Applied relation (per frequency ``ω_i``)::

        Ex(ω_i) = (Z[i,0,0] · Bx(ω_i) + Z[i,0,1] · By(ω_i)) / μ₀
        Ey(ω_i) = (Z[i,1,0] · Bx(ω_i) + Z[i,1,1] · By(ω_i)) / μ₀

    Shape convention: ``Z_tensor[i, :, :]`` is the 2×2 tensor at
    ``freqs_Hz[i]``. A 1-D isotropic Earth is the special case
    ``Z_tensor[i] = [[0, Z], [-Z, 0]]``.

    Parameters
    ----------
    freqs_Hz : numpy.ndarray
        Frequency array in Hz. Shape ``(n_freqs,)``.
    Z_tensor : numpy.ndarray
        Complex impedance tensor. Shape ``(n_freqs, 2, 2)``.

    Raises
    ------
    ShapeMismatchError
        If ``Z_tensor.shape`` does not equal ``(n_freqs, 2, 2)``.
    """

    def __init__(self, freqs_Hz: np.ndarray, Z_tensor: np.ndarray) -> None:
        super().__init__(freqs_Hz)
        self.Z_tensor = np.asarray(Z_tensor, dtype=np.complex128)
        expected = (self.freqs_Hz.size, 2, 2)
        if self.Z_tensor.shape != expected:
            raise ShapeMismatchError(
                f"Z_tensor shape {self.Z_tensor.shape} does not match expected {expected}"
            )

    def apply(
        self,
        Bx_f: np.ndarray,
        By_f: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the 2×2 tensor relation ``E(ω) = Z̃(ω) · B(ω) / μ₀``.

        Parameters
        ----------
        Bx_f, By_f : numpy.ndarray
            FFT of horizontal B-field components in Tesla. Shape ``(n_freqs,)``.

        Returns
        -------
        Ex_f, Ey_f : numpy.ndarray
            FFT of horizontal E-field components in V/m. Shape ``(n_freqs,)``.

        Raises
        ------
        ShapeMismatchError
            If either input does not match :attr:`freqs_Hz` in shape.
        """
        Bx_f = np.asarray(Bx_f)
        By_f = np.asarray(By_f)
        if Bx_f.shape != self.freqs_Hz.shape or By_f.shape != self.freqs_Hz.shape:
            raise ShapeMismatchError(
                f"B-field shapes {Bx_f.shape}, {By_f.shape} do not match "
                f"impedance freqs_Hz shape {self.freqs_Hz.shape}"
            )
        Ex_f = (self.Z_tensor[:, 0, 0] * Bx_f + self.Z_tensor[:, 0, 1] * By_f) / MU_0
        Ey_f = (self.Z_tensor[:, 1, 0] * Bx_f + self.Z_tensor[:, 1, 1] * By_f) / MU_0
        return Ex_f, Ey_f

    @property
    def rank(self) -> int:  # noqa: D401
        """Return ``2`` — 2×2 tensor per frequency."""
        return 2

    def to_hdf5(self, group: "h5py.Group") -> None:
        """Serialise to an HDF5 group with a schema-version attribute.

        Layout::

            group/
              attrs:
                impedance_type = "tensor_2d"
                schema_version = 1
              freqs_Hz      dataset
              Z_tensor      dataset (complex128, shape (n_freq, 2, 2))
        """
        group.attrs["impedance_type"] = "tensor_2d"
        group.attrs["schema_version"] = 1
        group.create_dataset("freqs_Hz", data=self.freqs_Hz)
        group.create_dataset("Z_tensor", data=self.Z_tensor)

    @classmethod
    def from_hdf5(cls, group: "h5py.Group") -> "TensorImpedance":
        """Read a :class:`TensorImpedance` from an HDF5 group."""
        return cls(
            freqs_Hz=group["freqs_Hz"][()],
            Z_tensor=group["Z_tensor"][()],
        )


class KernelImpedance(Impedance):
    """3-D volumetric-response kernel ``Z̃(r, r', ω)`` — STUB (WP3).

    Shape convention: ``(n_freqs, n_r, n_r, 2, 2)``. In practice the kernel
    will be sparse or block-structured and use chunked HDF5 storage.
    """

    def __init__(self, freqs_Hz: np.ndarray, Z_kernel: np.ndarray) -> None:
        super().__init__(freqs_Hz)
        self.Z_kernel = np.asarray(Z_kernel, dtype=np.complex128)
        if self.Z_kernel.ndim != 5 or self.Z_kernel.shape[-2:] != (2, 2):
            raise ShapeMismatchError(
                f"Z_kernel must have shape (n_freqs, n_r, n_r, 2, 2); got {self.Z_kernel.shape}"
            )

    def apply(
        self,
        Bx_f: np.ndarray,
        By_f: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Not yet implemented (WP3)."""
        raise NotImplementedYetError("KernelImpedance.apply", "WP3")

    @property
    def rank(self) -> int:  # noqa: D401
        """Return ``4`` — spatial kernel per frequency."""
        return 4

    def to_hdf5(self, group: "h5py.Group") -> None:
        """Not yet implemented (WP3)."""
        raise NotImplementedYetError("KernelImpedance.to_hdf5", "WP3")

    @classmethod
    def from_hdf5(cls, group: "h5py.Group") -> "KernelImpedance":
        """Not yet implemented (WP3)."""
        raise NotImplementedYetError("KernelImpedance.from_hdf5", "WP3")
