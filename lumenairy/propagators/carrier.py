"""
Carrier-referenced ("pilot-beam") free-space propagation
========================================================

A **prototype** scaled-coordinate free-space propagator that carries a
strongly-diverging / converging beam's own spherical wavefront
*analytically* and propagates only the slowly-varying **envelope** on a
modest grid -- the Sziklas & Siegman (1975) coordinate transform used by
laser-resonator and beam-propagation codes, and the mechanism Zemax POP
calls "pilot-beam re-referencing".

Motivation
----------
The 2026-07-18/19 lens audits showed that a strongly diverging input's
OWN phase fringes set the grid: the fringe pitch ``lambda*R/r`` at the
beam edge forces production grids to ``N = 28672 / dx = 0.9 um``
*propagator-independently* -- ANY model that samples the full phase
``exp(i*k*r^2/(2R))`` on a grid pays that cost (hammer-audit finding H8).

The carrier-referenced propagator sidesteps it: write the field as an
envelope times a spherical carrier,

    E(x, y) = u(x, y) * exp(i*k*(x^2 + y^2)/(2*R)),

track ``R`` analytically (``R -> R + z`` under free-space propagation),
and propagate the smooth envelope ``u`` on a co-moving grid whose pitch
magnifies with the geometric beam (``dx' = dx*(R+z)/R``).  The envelope
has almost no transverse phase, so it is sampled comfortably on a grid
orders of magnitude smaller than the one the full field needs.

The transform
-------------
For a spherical carrier of signed radius ``R`` at the input plane
(``R > 0`` diverging, the ``exp(-i*omega*t)`` / ``exp(+i*k*r^2/2R)``
convention this library uses everywhere), a free-space step of distance
``z`` maps

    R_out  = R + z                                (carrier sphere grows)
    m      = R_out / R                            (geometric magnification)
    dx_out = m * dx                               (co-moving grid)
    z_eff  = z * R / R_out = z / m                (reduced envelope distance)
    u_out(x_out) = (1/m) * exp(i*k*z^2/R_out)
                   * Fresnel_{z_eff}(u_in)(x_out / m)

i.e. propagate the envelope a **reduced** distance ``z_eff`` with an
ordinary collimated Fresnel transfer-function step (no curvature
sampling), rescale the coordinate by ``m``, and divide the amplitude by
``m`` (2-D power conservation: the area element grows by ``m^2``).  The
``exp(i*k*z^2/R_out)`` factor restores the on-axis (piston) phase the
reduced envelope leg under-counts (``z - z_eff = z^2/R_out``), so the
reconstructed full field is phase-faithful and legs compose.

This is EXACT for the quadratic (carrier) part of the wavefront under the
Fresnel kernel -- a textbook coordinate transform, not an approximation
of it (Sziklas & Siegman 1975).  The only modelling approximation is the
paraxial (Fresnel) propagation of the *envelope*, which is negligible
because the envelope's residual angular content (after the carrier is
removed) is tiny -- for a matched Gaussian carrier it is essentially the
diffraction-limited spread ``lambda/(pi*w)``.

Validity (prototype envelope)
-----------------------------
* Single spherical (isotropic) carrier: one ``R`` for both axes.  General
  ASTIGMATIC carriers (separate ``R_x``, ``R_y``) are a documented
  follow-up -- the transform generalises coordinate-wise but this
  prototype ships the isotropic form.
* Cannot step THROUGH the carrier's geometric focus: as ``R_out -> 0``
  the magnification ``m -> 0`` (the co-moving grid collapses) and as it
  changes sign the frame inverts.  Stepping exactly to, or across, the
  focus raises -- split the leg so the measurement plane is on one side,
  or hand off to a full-grid propagator across the focus.
* The envelope must be sampled by the grid it is handed on (its own
  ``lambda/(pi*w)`` spread), and free of the carrier fringe -- that is the
  whole point.  APERTURE handling at element planes (the envelope frame
  does not clip the same way the full field does) is a documented
  follow-up.

References
----------
[1] A.E. Siegman & E.A. Sziklas, "Mode calculations in unstable
    resonators with flowing saturable gain. 2: Fast Fourier transform
    method," Appl. Opt. 14(8), 1874-1889 (1975).
[2] A.E. Siegman, *Lasers* (University Science Books, 1986), Ch. 20
    (Huygens integral and the collimated/coordinate-scaled form).

Author: Andrew Traverso
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from .fresnel import fresnel_tf_propagate

__all__ = [
    'CarrierReferencedField',
    'propagate_carrier_referenced',
    'carrier_referenced_reconstruct',
    'carrier_referenced_envelope',
]


class CarrierReferencedField(NamedTuple):
    """Result of a carrier-referenced propagation step.

    A 3-field named tuple -- the grid pitch changes (the co-moving grid
    magnifies with the carrier), so unlike the grid-preserving
    propagators this MUST return the new pitch alongside the envelope and
    the carrier radius.  Unpacks as ``env, R, dx``.

    Attributes
    ----------
    env : ndarray, complex
        Envelope ``u`` on the co-moving grid (same shape / dtype as the
        input envelope).  The full physical field is
        ``env * exp(i*k*(x^2+y^2)/(2*R))`` on the ``dx`` grid -- rebuild
        it with :func:`carrier_referenced_reconstruct`.
    R : float
        Carrier radius of curvature at the output plane (m); ``+inf`` for
        a collimated carrier.
    dx : float
        Grid pitch at the output plane (m).  Equals ``m * dx_in`` with
        ``m = (R_in + z)/R_in`` for a finite carrier (unchanged for a
        collimated one).  ``dy`` scales by the same ``m``.
    """

    env: np.ndarray
    R: float
    dx: float


def _radial_carrier_phase(shape, dx, dy, wavelength, R, sign):
    """``exp(sign*i*k*(x^2+y^2)/(2R))`` on the centred grid (float64
    carrier argument, cast to nothing here -- caller casts)."""
    Ny, Nx = shape
    x = (np.arange(Nx, dtype=np.float64) - Nx / 2) * dx
    y = (np.arange(Ny, dtype=np.float64) - Ny / 2) * dy
    Y, X = np.meshgrid(y, x, indexing='ij')
    r2 = X * X + Y * Y
    k = 2.0 * np.pi / wavelength
    return np.exp(sign * 1j * k * r2 / (2.0 * R))


def propagate_carrier_referenced(
    E_env: np.ndarray,
    R_carrier: float,
    z: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
) -> CarrierReferencedField:
    """Carrier-referenced ("pilot-beam") free-space propagation step.

    Propagate the slowly-varying ENVELOPE ``E_env`` of a beam whose
    spherical carrier has signed radius ``R_carrier`` a distance ``z``
    through free space, via the Sziklas-Siegman scaled-coordinate Fresnel
    transform.  The carrier radius and the grid pitch evolve
    ANALYTICALLY; the envelope is propagated a reduced distance on the
    same-size grid, so a strongly diverging beam is transported on a grid
    orders of magnitude smaller than the full field ``E_env *
    exp(i*k*r^2/(2*R))`` would need to sample its own fringe.

    See the module docstring for the transform and its validity envelope.

    Parameters
    ----------
    E_env : ndarray, complex, shape (Ny, Nx)
        Beam envelope ``u`` (the field with the spherical carrier phase
        divided out).  Obtain it from a full field with
        :func:`carrier_referenced_envelope`, or construct it directly
        (e.g. a real Gaussian amplitude when the carrier is the beam's
        own wavefront).
    R_carrier : float
        Signed carrier radius of curvature at the INPUT plane (m).
        ``R > 0`` diverging, ``R < 0`` converging, ``+/-inf`` collimated
        (the step then reduces to an ordinary Fresnel transfer-function
        propagation with no grid magnification).
    z : float
        Propagation distance (m); may be negative (back-propagation).
        ``z == 0`` returns the input unchanged.
    wavelength : float
        Wavelength (m).
    dx : float
        Input grid pitch in x (m).
    dy : float, optional
        Input grid pitch in y (m).  Defaults to ``dx``.

    Returns
    -------
    CarrierReferencedField
        Named tuple ``(env, R, dx)`` -- the output envelope, the output
        carrier radius ``R_carrier + z``, and the magnified output pitch
        ``m * dx``.  ``dy`` (if it differed from ``dx``) scales by the
        same ``m``; recover it as ``dy_out = dy_in * dx_out / dx_in``.

    Raises
    ------
    ValueError
        If ``R_carrier == 0``, or the step lands on / crosses the
        carrier's geometric focus (``R_carrier + z`` is zero or flips
        sign, so the co-moving magnification is non-positive).

    Examples
    --------
    Transport a diverging Gaussian (waist ``w0`` a distance ``z_A`` back,
    so ``R_A`` finite) a further ``z``::

        from lumenairy import (
            propagate_carrier_referenced, carrier_referenced_reconstruct)
        env_out, R_out, dx_out = propagate_carrier_referenced(
            env_in, R_A, z, wavelength, dx)
        E_full = carrier_referenced_reconstruct(
            env_out, R_out, wavelength, dx_out)
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_env, 'propagate_carrier_referenced')
    if dy is None:
        dy = dx
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError(
            f"propagate_carrier_referenced: wavelength must be finite and "
            f"positive, got {wavelength!r}.")
    if not (np.isfinite(dx) and dx > 0 and np.isfinite(dy) and dy > 0):
        raise ValueError(
            f"propagate_carrier_referenced: dx, dy must be finite and "
            f"positive, got dx={dx!r}, dy={dy!r}.")
    if not np.isfinite(z):
        raise ValueError(
            f"propagate_carrier_referenced: z must be finite, got {z!r}.")

    R = float(R_carrier)

    # z == 0: exact identity (mirror the ASM / fresnel_tf z==0 contract).
    if z == 0:
        env0 = E_env.copy() if hasattr(E_env, 'copy') else np.array(E_env)
        return CarrierReferencedField(env0, R, dx)

    # Collimated carrier: the transform degenerates to m == 1, z_eff == z;
    # an ordinary Fresnel transfer-function step, grid unchanged.
    if np.isinf(R):
        env_out = fresnel_tf_propagate(E_env, z, wavelength, dx, dy)
        return CarrierReferencedField(env_out, R, dx)

    if R == 0.0:
        raise ValueError(
            "propagate_carrier_referenced: R_carrier == 0 is the carrier's "
            "own focus (undefined magnification).  Reference the beam to a "
            "plane away from its focus.")

    R_out = R + z
    m = R_out / R
    if not np.isfinite(m) or m <= 0.0:
        raise ValueError(
            f"propagate_carrier_referenced: the step lands on or crosses the "
            f"carrier's geometric focus (R_in={R:.6g} m, z={z:.6g} m -> "
            f"R_out={R_out:.6g} m, magnification m={m:.6g}).  The "
            f"scaled-coordinate frame is singular there; split the leg so the "
            f"plane is on one side of the focus, or hand off to a full-grid "
            f"propagator across it.")

    # Reduced (collimated-frame) envelope distance.  z_eff = z / m.
    z_eff = z * R / R_out

    # Envelope leg: ordinary collimated Fresnel TF step, SAME grid.
    u_out = fresnel_tf_propagate(E_env, z_eff, wavelength, dx, dy)

    # On-axis piston the reduced leg under-counts: z - z_eff = z^2 / R_out.
    k = 2.0 * np.pi / wavelength
    piston = np.exp(1j * k * (z * z / R_out))
    # 2-D power conservation: the co-moving area element grows by m^2, so
    # the amplitude carries 1/m (each axis 1/sqrt(m)).
    scale = piston / m
    env_out = scale * u_out
    # Preserve the input complex dtype (the python-complex scalar would
    # otherwise upcast complex64 -> complex128).
    if np.iscomplexobj(E_env) and env_out.dtype != E_env.dtype:
        env_out = env_out.astype(E_env.dtype)

    return CarrierReferencedField(env_out, R_out, m * dx)


def carrier_referenced_reconstruct(
    E_env: np.ndarray,
    R_carrier: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
) -> np.ndarray:
    """Rebuild the full physical field from an envelope + spherical carrier.

    ``E_full = E_env * exp(i*k*(x^2 + y^2)/(2*R_carrier))`` on the centred
    ``dx`` grid (``exp(+i*k*r^2/2R)`` is this library's diverging-``R>0``
    convention -- the same sign as :func:`apply_fresnel_curvature` with
    ``sign=+1`` and the traced carrier eikonal).  A collimated carrier
    (``R = +/-inf``) is a no-op and the envelope is returned as the field.

    Use this to hand a carrier-referenced leg's output off to a full-grid
    element model (e.g. :func:`apply_real_lens_traced`, which additionally
    accepts the carrier ``R`` as its ``carrier=`` argument so the OPL is
    referenced exactly -- hammer-audit H6).

    Parameters
    ----------
    E_env : ndarray, complex, shape (Ny, Nx)
    R_carrier : float
        Signed carrier radius (m); ``+/-inf`` -> collimated (no-op).
    wavelength, dx : float
    dy : float, optional
        Defaults to ``dx``.

    Returns
    -------
    E_full : ndarray, complex
        Same shape / dtype as ``E_env``.
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_env, 'carrier_referenced_reconstruct')
    if dy is None:
        dy = dx
    if np.isinf(R_carrier):
        return E_env.copy() if hasattr(E_env, 'copy') else np.array(E_env)
    if R_carrier == 0.0:
        raise ValueError(
            "carrier_referenced_reconstruct: R_carrier == 0 (carrier focus).")
    phase = _radial_carrier_phase(E_env.shape, dx, dy, wavelength,
                                  float(R_carrier), +1)
    if np.iscomplexobj(E_env):
        phase = phase.astype(E_env.dtype, copy=False)
    return E_env * phase


def carrier_referenced_envelope(
    E_full: np.ndarray,
    R_carrier: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
) -> np.ndarray:
    """Extract the envelope of a full field for a given spherical carrier.

    ``E_env = E_full * exp(-i*k*(x^2 + y^2)/(2*R_carrier))`` -- the inverse
    of :func:`carrier_referenced_reconstruct`.  When ``R_carrier`` matches
    the field's actual wavefront the result is slowly varying (its
    transverse phase is only the diffraction-limited residual), which is
    what makes the modest grid sufficient.  A collimated carrier is a
    no-op.

    NOTE (prototype): extracting an envelope on a grid that does NOT
    already sample the carrier fringe aliases it -- this helper is for
    grids that resolve the full field (e.g. at an element plane during a
    hand-off), not for recovering the memory win after the fact.

    Parameters
    ----------
    E_full : ndarray, complex, shape (Ny, Nx)
    R_carrier : float
    wavelength, dx : float
    dy : float, optional

    Returns
    -------
    E_env : ndarray, complex
        Same shape / dtype as ``E_full``.
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_full, 'carrier_referenced_envelope')
    if dy is None:
        dy = dx
    if np.isinf(R_carrier):
        return E_full.copy() if hasattr(E_full, 'copy') else np.array(E_full)
    if R_carrier == 0.0:
        raise ValueError(
            "carrier_referenced_envelope: R_carrier == 0 (carrier focus).")
    phase = _radial_carrier_phase(E_full.shape, dx, dy, wavelength,
                                  float(R_carrier), -1)
    if np.iscomplexobj(E_full):
        phase = phase.astype(E_full.dtype, copy=False)
    return E_full * phase
