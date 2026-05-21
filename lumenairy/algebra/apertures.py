"""lumenairy.algebra.apertures -- aperture operators.

Aperture-class operators have identity ABCD (paraxial rays pass
through unaltered) but vignette / soft-clip the field amplitude.
They compose with the optical-ABCD operators in the usual way --
``ThinLens(f) * Aperture(D)`` produces a system with a finite
clear aperture upstream of the focusing element.

Author: Andrew Traverso
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .base import Operator, _validate_abcd

# ---------------------------------------------------------------------------
# Aperture (hard amplitude mask)
# ---------------------------------------------------------------------------


class Aperture(Operator):
    """Hard amplitude aperture with selectable shape.

    ABCD is identity (apertures vignette rays but don't bend them).

    Parameters
    ----------
    diameter : float
        Aperture full-diameter [m].  For rectangular shape this is
        interpreted as the side length (square aperture).
    shape : {'circular', 'rectangular', 'annular'}, default 'circular'
        Aperture shape:

        - ``'circular'`` -- disk of diameter ``diameter``.
        - ``'rectangular'`` -- square of side ``diameter``.
        - ``'annular'`` -- ring with outer diameter ``diameter``.
          The inner diameter defaults to ``diameter / 2``; pass
          ``inner_diameter=...`` to override.
    inner_diameter : float, optional
        For ``shape='annular'``, the inner diameter [m].

    Notes
    -----
    Delegates :meth:`_apply` to :func:`lumenairy.apply_aperture`.
    For rectangular non-square apertures (``width_x != width_y``)
    or off-axis decentered apertures, call ``apply_aperture``
    directly -- the algebraic surface deliberately exposes a
    minimal shape vocabulary.
    """

    _ALLOWED_SHAPES = ('circular', 'rectangular', 'annular')

    def __init__(
        self,
        diameter: float,
        shape: str = 'circular',
        *,
        inner_diameter: float = 0.0,
    ) -> None:
        try:
            D = float(diameter)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Aperture: diameter must be a real number, got "
                f"{diameter!r}."
            ) from e
        if not np.isfinite(D) or D <= 0.0:
            raise ValueError(
                f"Aperture: diameter must be positive and finite, got "
                f"{D}."
            )
        if shape not in self._ALLOWED_SHAPES:
            raise ValueError(
                f"Aperture: shape must be one of "
                f"{self._ALLOWED_SHAPES}, got {shape!r}."
            )
        try:
            inner = float(inner_diameter)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Aperture: inner_diameter must be a real number, got "
                f"{inner_diameter!r}."
            ) from e
        if inner < 0.0 or inner >= D:
            if shape == 'annular' and inner == 0.0:
                # Default inner = D / 2 for annular if caller did not
                # supply one.
                inner = D / 2.0
            elif inner != 0.0:
                raise ValueError(
                    f"Aperture: inner_diameter must be in [0, "
                    f"diameter), got inner={inner}, diameter={D}."
                )
        self.diameter = D
        self.shape = shape
        self.inner_diameter = inner

        I = _validate_abcd([[1.0, 0.0], [0.0, 1.0]], 'Aperture')
        self._abcd_x = I
        self._abcd_y = I.copy()

    def __repr__(self) -> str:
        if self.shape == 'annular':
            return (
                f"Aperture(diameter={self.diameter:.6g}, "
                f"shape={self.shape!r}, "
                f"inner_diameter={self.inner_diameter:.6g})"
            )
        return (
            f"Aperture(diameter={self.diameter:.6g}, "
            f"shape={self.shape!r})"
        )

    def _apply(
        self,
        E: np.ndarray,
        *,
        dx: float,
        dy: float,
        wavelength: float,
    ) -> Tuple[np.ndarray, float, float]:
        from ..elements import apply_aperture
        if self.shape == 'circular':
            params = {'diameter': self.diameter}
        elif self.shape == 'rectangular':
            params = {'width_x': self.diameter, 'width_y': self.diameter}
        else:  # annular
            params = {
                'inner_diameter': self.inner_diameter,
                'outer_diameter': self.diameter,
            }
        E_out = apply_aperture(
            E, dx=dx, dy=dy, shape=self.shape, params=params,
        )
        return E_out, dx, dy


# ---------------------------------------------------------------------------
# GaussianAperture (soft Gaussian amplitude mask)
# ---------------------------------------------------------------------------


class GaussianAperture(Operator):
    """Soft Gaussian amplitude aperture.

    ABCD is identity.  Transmission profile::

        T(r) = exp(-r^2 / (2 sigma^2))

    Parameters
    ----------
    sigma : float
        Gaussian width [m].  The 1/e amplitude radius is
        ``sigma * sqrt(2)``; the 1/e^2 intensity radius is also
        ``sigma * sqrt(2)``.

    Notes
    -----
    Delegates :meth:`_apply` to
    :func:`lumenairy.apply_gaussian_aperture`.  Useful as an
    apodizer to smooth diffraction edges before downstream
    propagation; combine with :class:`Aperture` for a hard-soft
    cascade.
    """

    def __init__(self, sigma: float) -> None:
        try:
            s = float(sigma)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"GaussianAperture: sigma must be a real number, got "
                f"{sigma!r}."
            ) from e
        if not np.isfinite(s) or s <= 0.0:
            raise ValueError(
                f"GaussianAperture: sigma must be positive and finite, "
                f"got {s}."
            )
        self.sigma = s
        I = _validate_abcd([[1.0, 0.0], [0.0, 1.0]], 'GaussianAperture')
        self._abcd_x = I
        self._abcd_y = I.copy()

    def __repr__(self) -> str:
        return f"GaussianAperture(sigma={self.sigma:.6g})"

    def _apply(
        self,
        E: np.ndarray,
        *,
        dx: float,
        dy: float,
        wavelength: float,
    ) -> Tuple[np.ndarray, float, float]:
        from ..elements import apply_gaussian_aperture
        E_out = apply_gaussian_aperture(
            E, dx=dx, sigma=self.sigma, dy=dy,
        )
        return E_out, dx, dy


__all__ = ['Aperture', 'GaussianAperture']
