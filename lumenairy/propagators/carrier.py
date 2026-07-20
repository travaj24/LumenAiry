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
* Stepping THROUGH the carrier's geometric focus is handled
  transparently by an automatic split (Task 2).  As ``R_out -> 0`` the
  magnification ``m -> 0`` (the co-moving grid collapses) and the frame
  inverts on sign change, so a crossing (or a landing within a safety
  margin of the focus) is auto-split into
  ``carrier -> through-waist ASM bridge -> carrier``: see "Focus
  crossing" below.  The no-crossing fast path is byte-identical.
* The envelope must be sampled by the grid it is handed on (its own
  ``lambda/(pi*w)`` spread), and free of the carrier fringe -- that is the
  whole point.  APERTURE handling at element planes (the envelope frame
  does not clip the same way the full field does) is a documented
  follow-up.

Focus crossing (Task 2)
-----------------------
The scaled frame is singular at the focus, so a step that crosses it
(``R`` and ``R+z`` opposite in sign) -- or lands within a safety margin
of it -- is auto-split into three faithful legs:

1. **carrier -> hand-off plane a**, a few Rayleigh ranges BEFORE the
   waist (``|R_a| = 6*zR_est``, ``zR_est = wl*R^2/(pi*w_env^2)`` from the
   measured envelope width).  There the co-moving grid has magnified DOWN
   to a fine pitch that resolves the compact near-focus field while still
   holding the beam.  A fast (byte-identical) carrier step.
2. **through-waist ASM bridge a -> b**.  The full field is reconstructed
   on that fine co-moving grid and stepped across the waist with an exact
   band-limited angular-spectrum propagation.  This is sound because near
   focus the field is COMPACT and gently phased (the geometric carrier
   ``R -> 0`` but the true wavefront flattens to ``R = inf`` at the
   waist), so the fine co-moving grid Nyquist-covers it -- exactly the
   regime a single fixed grid CANNOT (it would need both the 3 mm input
   beam AND the 4 um waist).
3. **re-attach carrier + continue b -> target**.  The diverging carrier
   is re-attached by the geometric continuation ``R_b = -R_a`` (the
   carrier sphere flips sign through the waist); the envelope is
   extracted and a fast carrier step carries it to the target.  Any
   residual (Gaussian ``zR^2/Delta``) curvature rides the envelope
   exactly, so legs still compose.

If the target itself lands inside the near-focus bridge zone ``[a, b]``,
the ASM leg runs straight to it and a carrier is fitted there.  Validated
(w0=4 um, R=-30 mm, +60 mm through the waist, N=2048) to <0.1 % windowed
r2m/EE on BOTH sides of focus and at the waist vs the analytic Gaussian
(ABCD) and a resolved fine-grid ASM, provided the input grid holds the
beam to the usual ``±>2.4 w`` (a tighter grid truncates the tail through
the whole chain, the same finite-grid caveat as the non-crossing path).
Raises only for an un-bridgeable case: a zero/empty envelope, or a focus
coinciding with the start plane.

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

# --- focus-crossing (auto-split) tuning (Task 2) ---------------------------
# Only landings within this fraction of |R| of the geometric focus are even
# considered for the near-focus bridge; everything else takes the byte-
# identical fast path with NO envelope measurement.
_NEAR_FOCUS_FRACTION = 0.02
# Hand off to the through-waist ASM bridge this many Rayleigh ranges before the
# waist (deep enough in the geometric regime that the flipped-carrier
# continuation R_out = -R_in is faithful, shallow enough that the compact
# near-focus field is Nyquist-covered by the co-moving grid).
_BRIDGE_ZR_FACTOR = 6.0
# Require the co-moving half-width to exceed this multiple of the beam radius
# for the plain fast path to be trusted near focus.
_BRIDGE_FIT_MARGIN = 1.6


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
        If ``R_carrier == 0`` (the carrier's own focus, undefined
        magnification).  A step that lands on / crosses the carrier's
        geometric focus is NO LONGER an error (Task 2): it is handled
        transparently by an automatic carrier -> through-waist ASM
        bridge -> carrier split (see the module docstring).  Only a
        genuinely un-bridgeable case raises -- an empty/zero envelope,
        or a focus coinciding with the start plane (no room to hand off
        before the waist).

    Notes
    -----
    Focus crossing is auto-split; the two carrier legs plus the bridge
    are transparent to the caller, which receives a single output
    ``(env, R_out=R+z, dx_out)`` with a freshly-fitted DIVERGING carrier
    past the waist.  The no-crossing fast path is byte-identical to prior
    releases.

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
    with np.errstate(divide='ignore', invalid='ignore'):
        m = R_out / R
    # Focus crossing (R_out flips sign / lands on the focus) OR a near-focus
    # landing whose co-moving grid would no longer hold the compact waist:
    # auto-split into carrier -> through-waist ASM bridge -> carrier (Task 2).
    if ((not np.isfinite(m)) or (m <= 0.0)
            or _near_focus_needs_bridge(E_env, R, R_out, wavelength, dx, dy)):
        return _propagate_carrier_focus_crossing(
            E_env, R, z, wavelength, dx, dy)

    # No-crossing fast path -- byte-identical to prior releases (pinned).
    return _carrier_step_fast(E_env, R, z, wavelength, dx, dy)


def _carrier_step_fast(E_env, R, z, wavelength, dx, dy):
    """The Sziklas-Siegman fast step for a NON-crossing leg (``m = R_out/R > 0``,
    ``R`` finite/non-zero, ``z != 0``).  Byte-identical to the historical inline
    fast path; extracted so the focus-crossing bridge can reuse it for the two
    well-conditioned carrier legs.  Returns ``(env_out, R_out, dx_out)``."""
    R_out = R + z
    m = R_out / R
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


def _envelope_amp_radius(E_env, dx, dy):
    """1/e AMPLITUDE radius of the (assumed roughly Gaussian) envelope on the
    centred grid, from the intensity second moment (``w = sqrt(2)*r2m``).
    Returns 0.0 for an empty / zero field."""
    I = np.abs(np.asarray(E_env)) ** 2
    tot = float(I.sum())
    if not (tot > 0.0):
        return 0.0
    Ny, Nx = I.shape[-2], I.shape[-1]
    x = (np.arange(Nx, dtype=np.float64) - Nx / 2) * dx
    y = (np.arange(Ny, dtype=np.float64) - Ny / 2) * dy
    Y, X = np.meshgrid(y, x, indexing='ij')
    r2 = float((I * (X * X + Y * Y)).sum()) / tot
    return float(np.sqrt(2.0 * max(r2, 0.0)))


def _near_focus_needs_bridge(E_env, R, R_out, wavelength, dx, dy):
    """Same-sign (``m>0``) guard: is the landing so close to the geometric focus
    that the shrunken co-moving grid can no longer hold the diffraction-limited
    waist?  Returns ``False`` (cheaply, no envelope measurement) for any landing
    comfortably away from focus, so the fast path stays byte-identical."""
    if abs(R_out) >= _NEAR_FOCUS_FRACTION * abs(R):
        return False
    w_in = _envelope_amp_radius(E_env, dx, dy)
    if not (w_in > 0.0):
        return False
    Nx = np.asarray(E_env).shape[-1]
    w0 = wavelength * abs(R) / (np.pi * w_in)          # est. focus waist
    w_geom = w_in * abs(R_out) / abs(R)                # geometric beam at R_out
    w_out = max(w_geom, w0)                             # diffraction floor
    half = 0.5 * Nx * dx * abs(R_out) / abs(R)          # co-moving half-width
    return half < _BRIDGE_FIT_MARGIN * w_out


def _propagate_carrier_focus_crossing(E_env, R, z, wavelength, dx, dy):
    """Auto-split a leg that crosses (or lands within a safety margin of) the
    carrier's geometric focus (Task 2).

    Mechanism -- the scaled-coordinate frame is singular at the focus
    (``m = (R+z)/R -> 0``, the co-moving grid collapses; it inverts on the far
    side).  So we (a) carrier-step to a hand-off plane a few Rayleigh ranges
    BEFORE the waist, where the co-moving grid is fine and still holds the beam;
    (b) reconstruct the full field there and bridge THROUGH the waist with an
    exact band-limited ASM step -- the field is COMPACT and gently-phased near
    focus, so the (now fine) co-moving grid Nyquist-covers it; (c) re-attach the
    flipped diverging carrier (``R -> -R`` geometric continuation through the
    waist), extract the envelope, and continue carrier-referenced.

    See the module docstring's "Focus crossing" section for the derivation and
    validity envelope."""
    from .propagation import angular_spectrum_propagate

    R_out = R + z
    z_f = -R                                    # focus distance from start
    w_in = _envelope_amp_radius(E_env, dx, dy)
    if not (w_in > 0.0):
        raise ValueError(
            "propagate_carrier_referenced: cannot bridge the carrier focus -- "
            "the envelope has no finite width to size the near-focus grid.  "
            "Reference the beam to a plane away from its focus, or split the "
            "leg manually.")
    w0 = wavelength * abs(R) / (np.pi * w_in)   # estimated focus waist
    zR = np.pi * w0 * w0 / wavelength           # = wl*R^2/(pi*w_in^2)
    delta = _BRIDGE_ZR_FACTOR * zR              # hand-off offset from focus
    # Keep the hand-off plane STRICTLY between the start and the focus.
    delta = min(delta, 0.45 * abs(z_f))
    if not (delta > 0.0):
        raise ValueError(
            f"propagate_carrier_referenced: the carrier focus (z_f={z_f:.6g} m) "
            f"coincides with the start plane, so there is no room to hand off "
            f"before the waist.  Reference the beam a finite distance from its "
            f"focus.")
    sgn = 1.0 if z_f >= 0.0 else -1.0           # focus forward/backward
    z_a = z_f - sgn * delta                     # hand-off (start side of focus)
    z_b = z_f + sgn * delta                     # re-attach (far side)
    R_b = R + z_b                               # = +sgn*delta (flipped carrier)

    # (a) fast carrier step start -> a (well conditioned: |R_a| = delta, m>0)
    env_a, R_a, dx_a = _carrier_step_fast(E_env, R, z_a, wavelength, dx, dy)
    dy_a = dy * (dx_a / dx)

    # (b) reconstruct the full field on the (now fine) co-moving grid
    E_a = carrier_referenced_reconstruct(env_a, R_a, wavelength, dx_a, dy_a)

    beyond_b = (z - z_b) * sgn                  # >0 iff target is past b
    if beyond_b <= 0.0:
        # Target lands inside the bridge zone [a, b] (near focus): the compact
        # field is held by the fine grid, so ASM straight to it and fit R_out.
        E_t = angular_spectrum_propagate(
            E_a, z - z_a, wavelength, dx_a, dy=dy_a)
        env_t = carrier_referenced_envelope(E_t, R_out, wavelength, dx_a, dy_a)
        return CarrierReferencedField(
            _match_env_dtype(env_t, E_env), R_out, dx_a)

    # (b) bridge a -> b THROUGH the waist with an exact ASM step
    E_b = angular_spectrum_propagate(
        E_a, z_b - z_a, wavelength, dx_a, dy=dy_a)
    # (c) re-attach the flipped diverging carrier and continue
    env_b = carrier_referenced_envelope(E_b, R_b, wavelength, dx_a, dy_a)
    env_f, R_f, dx_f = _carrier_step_fast(
        env_b, R_b, z - z_b, wavelength, dx_a, dy_a)
    return CarrierReferencedField(_match_env_dtype(env_f, E_env), R_f, dx_f)


def _match_env_dtype(env, ref):
    """Cast ``env`` back to ``ref``'s complex dtype if the bridge upcast it
    (preserve complex64 through the focus-crossing path)."""
    env = np.asarray(env)
    if np.iscomplexobj(ref) and env.dtype != np.asarray(ref).dtype:
        return env.astype(np.asarray(ref).dtype)
    return env


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
