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

Validity (envelope)
-------------------
* Spherical (isotropic) carrier: one ``R`` for both axes -- the fast path
  and the documented default.
* ASTIGMATIC carrier (separate ``R_x``, ``R_y``) via ``carrier=(R_x,
  R_y)`` (Phase P6 Task 1).  The Fresnel kernel is separable, so the
  transform runs coordinate-wise: per-axis magnification ``m_x = (R_x +
  z)/R_x``, ``m_y = (R_y + z)/R_y``, per-axis reduced envelope distance,
  and per-axis grid pitch (``dx_out = m_x*dx``, ``dy_out = m_y*dy``).
  The two geometric foci fall at DIFFERENT ``z`` (``-R_x`` vs ``-R_y``),
  so the focus-crossing split/bridge below is applied INDEPENDENTLY on
  each axis (a 1-D bridge that touches only that axis).  An astigmatic
  call returns ``R`` and ``dx`` as 2-tuples ``(R_x_out, R_y_out)`` and
  ``(dx_out, dy_out)``.  ``carrier=(R, R)`` (equal radii) routes to the
  scalar path and is byte-identical to the isotropic form.
* Stepping THROUGH the carrier's geometric focus is handled
  transparently by an automatic split (Task 2).  As ``R_out -> 0`` the
  magnification ``m -> 0`` (the co-moving grid collapses) and the frame
  inverts on sign change, so a crossing (or a landing within a safety
  margin of the focus) is auto-split into
  ``carrier -> through-waist ASM bridge -> carrier``: see "Focus
  crossing" below.  The no-crossing fast path is byte-identical.
* The envelope must be sampled by the grid it is handed on (its own
  ``lambda/(pi*w)`` spread), and free of the carrier fringe -- that is the
  whole point.
* A hard APERTURE mid-chain clips the envelope in place (the carrier is a
  pure phase, so ``|env|`` is clipped exactly as the full field is) and
  the clipped power is REMOVED, never renormalised.  A hard clip does not
  change the wavefront curvature at the surviving points, so the carrier
  ``R`` is unchanged by default; :func:`carrier_referenced_aperture`
  optionally re-fits ``R`` from the apertured field (to absorb residual
  envelope curvature) or re-references it to a user-supplied conjugate
  (Phase P6 Task 2).

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
beam to the usual ``+/->2.4 w`` (a tighter grid truncates the tail through
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

from typing import NamedTuple, Optional, Tuple, Union

import numpy as np

from .fresnel import fresnel_tf_propagate

__all__ = [
    'CarrierReferencedField',
    'propagate_carrier_referenced',
    'carrier_referenced_reconstruct',
    'carrier_referenced_envelope',
    'carrier_referenced_aperture',
    'carrier_referenced_fit_radius',
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
    R_carrier: Union[float, Tuple[float, float]],
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
    R_carrier : float or (float, float)
        Signed carrier radius of curvature at the INPUT plane (m).
        ``R > 0`` diverging, ``R < 0`` converging, ``+/-inf`` collimated
        (the step then reduces to an ordinary Fresnel transfer-function
        propagation with no grid magnification).  Pass a 2-tuple
        ``(R_x, R_y)`` for an ASTIGMATIC carrier (separate x/y radii): the
        separable Sziklas-Siegman transform magnifies each axis by its own
        ``m`` and crosses each geometric focus independently.  An
        astigmatic call returns ``R`` and ``dx`` as 2-tuples (see Returns).
        ``(R, R)`` (equal radii) routes to the scalar path byte-identically.
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
        For an ASTIGMATIC carrier (``R_carrier=(R_x, R_y)``, distinct
        radii) ``R`` is the 2-tuple ``(R_x + z, R_y + z)`` and ``dx`` is
        the 2-tuple of per-axis pitches ``(m_x*dx_in, m_y*dy_in)`` (the
        axes magnify independently, so no single scalar pitch applies).
        Hand the astigmatic result to :func:`carrier_referenced_reconstruct`
        / :func:`carrier_referenced_envelope`, which accept the same
        ``(R_x, R_y)`` tuple plus ``dx``, ``dy``.

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

    # Parse a possibly-astigmatic carrier.  A 2-tuple (R_x, R_y) with
    # DISTINCT radii routes to the separable astigmatic transform; equal
    # radii (and the scalar form) route to the byte-identical scalar path.
    R_x, R_y, is_astig = _parse_carrier(R_carrier,
                                        'propagate_carrier_referenced')
    if is_astig:
        if R_x == 0.0 or R_y == 0.0:
            raise ValueError(
                "propagate_carrier_referenced: an astigmatic carrier axis "
                "radius is 0 (the carrier's own focus, undefined "
                "magnification).  Reference the beam a finite distance from "
                "its focus on each axis.")
        # z == 0: exact identity, astigmatic (R, dx) as 2-tuples.
        if z == 0:
            env0 = E_env.copy() if hasattr(E_env, 'copy') else np.array(E_env)
            return CarrierReferencedField(env0, (R_x, R_y), (dx, dy))
        return _propagate_carrier_astigmatic(
            E_env, R_x, R_y, z, wavelength, dx, dy)

    R = R_x  # scalar / isotropic path (equal-radii tuple collapses here)

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


# ===========================================================================
# Astigmatic carrier (Task 1): separable per-axis Sziklas-Siegman
# ===========================================================================
# The 2-D Fresnel kernel factorises into an x-part and a y-part, and a
# separable carrier ``exp(i k [x^2/(2 R_x) + y^2/(2 R_y)])`` references each
# axis independently, so the whole transform runs coordinate-wise: a 1-D
# Sziklas-Siegman step on axis x (magnification ``m_x``, pitch ``m_x*dx``)
# and a 1-D step on axis y.  The two geometric foci sit at DIFFERENT z
# (``-R_x`` vs ``-R_y``), so each axis carries its own focus-crossing bridge.
#
# Array convention: shape ``(Ny, Nx)``; axis 1 is x (pitch ``dx``, radius
# ``R_x``), axis 0 is y (pitch ``dy``, radius ``R_y``).  Every 1-D op returns
# an envelope that CARRIES the full on-axis piston ``exp(i k z)``; the two
# axes therefore accumulate ``exp(i 2 k z)`` and the astigmatic driver
# divides the single physical ``exp(i k z)`` back out once at the end (a
# global phase -- it does not touch any intensity, but keeps composed legs
# phase-faithful, matching the scalar path's convention).


def _parse_carrier(R_carrier, fn):
    """Return ``(R_x, R_y, is_astigmatic)``.

    A scalar maps to ``(R, R, False)``.  A length-2 tuple/list/array maps to
    ``(R_x, R_y, R_x != R_y)`` -- equal radii are treated as isotropic so the
    caller routes them to the byte-identical scalar path.
    """
    if isinstance(R_carrier, (tuple, list, np.ndarray)):
        arr = np.asarray(R_carrier, dtype=np.float64).ravel()
        if arr.size != 2:
            raise ValueError(
                f"{fn}: carrier must be a scalar radius R or a 2-tuple "
                f"(R_x, R_y); got {arr.size} values.")
        R_x, R_y = float(arr[0]), float(arr[1])
        return R_x, R_y, (R_x != R_y)
    R = float(R_carrier)
    return R, R, False


def _freq_sq_1d(N, d):
    """Centred ``(2*pi*f)^2`` float64 vector for a length-``N`` axis at pitch
    ``d`` (matches the shared ASM/Fresnel freq-grid construction)."""
    f = (np.arange(N, dtype=np.float64) - N / 2) / (N * d)
    return (2.0 * np.pi * f) ** 2


def _broadcast_axis(vec, ndim, axis):
    """Reshape a 1-D vector to broadcast along ``axis`` of an ``ndim`` array."""
    shape = [1] * ndim
    shape[axis] = vec.shape[0]
    return vec.reshape(shape)


def _fresnel_tf_axis(u, z, wavelength, d, axis):
    """1-D Fresnel transfer-function step along a single ``axis`` (the
    per-axis analogue of :func:`fresnel_tf_propagate`).  Carries the on-axis
    piston ``exp(i k z)`` (the ``k*z`` term), so composed legs stay phase-
    faithful.  ``z`` here is the (already reduced) envelope distance."""
    N = u.shape[axis]
    k = 2.0 * np.pi / wavelength
    kx_sq = np.fft.ifftshift(_freq_sq_1d(N, d))
    # phase = k*z - (z/2k) (2 pi f)^2  (== k*z - pi*lambda*z*f^2), 1-D.
    H = np.exp(1j * (k * z - (z / (2.0 * k)) * kx_sq))
    H = _broadcast_axis(H, u.ndim, axis)
    Uf = np.fft.fft(u, axis=axis)
    return np.fft.ifft(Uf * H, axis=axis)


def _asm_axis(E, z, wavelength, d, axis, bandlimit=True):
    """1-D exact (band-limited) angular-spectrum step along a single ``axis``
    -- the per-axis analogue of :func:`angular_spectrum_propagate`, used only
    inside the near-focus bridge to carry the compact field across the waist
    on the (now fine) co-moving grid.  Carries the on-axis piston
    ``exp(i k z)`` (``kz(f=0) = k``).  The other axis is a spectator (its
    slowly-varying, carrier-referenced content has little transverse
    frequency, so the neglected ``k_y`` coupling in ``sqrt(k^2 - k_x^2)`` is
    a small paraxial residual -- the regime this bridge operates in)."""
    if z == 0:
        return E.copy() if hasattr(E, 'copy') else np.array(E)
    N = E.shape[axis]
    k = 2.0 * np.pi / wavelength
    kx_sq = _freq_sq_1d(N, d)
    kz_sq = k * k - kx_sq
    prop = kz_sq > 0
    kz = np.where(prop, np.sqrt(np.maximum(kz_sq, 0.0)), 0.0)
    H = np.where(prop, np.exp(1j * z * kz), 0.0 + 0.0j)
    if bandlimit:
        f = (np.arange(N, dtype=np.float64) - N / 2) / (N * d)
        f_max = (N * d) / (2.0 * wavelength * abs(z))
        H = np.where(np.abs(f) < f_max, H, 0.0 + 0.0j)
    H = _broadcast_axis(np.fft.ifftshift(H), E.ndim, axis)
    Ef = np.fft.fft(E, axis=axis)
    return np.fft.ifft(Ef * H, axis=axis)


def _axis_carrier_phase(shape, d, wavelength, R, axis, sign):
    """``exp(sign*i*k*x_a^2/(2R))`` broadcast along a single ``axis`` (the
    per-axis carrier)."""
    N = shape[axis]
    x = (np.arange(N, dtype=np.float64) - N / 2) * d
    k = 2.0 * np.pi / wavelength
    ph = np.exp(sign * 1j * k * x * x / (2.0 * R))
    return _broadcast_axis(ph, len(shape), axis)


def _axis_amp_radius(u, d, axis):
    """1/e amplitude radius of the envelope along a single ``axis`` from the
    marginal-intensity second moment.  For a Gaussian amplitude
    ``exp(-x^2/w^2)`` the marginal intensity ``exp(-2x^2/w^2)`` has
    ``<x^2> = w^2/4``, so ``w = 2*sqrt(<x^2>)`` (the 1-D analogue of the 2-D
    ``w = sqrt(2)*r2m``, which folds in both axes).  0.0 for a zero/empty
    field."""
    inten = np.abs(np.asarray(u)) ** 2
    other = tuple(a for a in range(inten.ndim) if a != axis)
    marg = inten.sum(axis=other)
    tot = float(marg.sum())
    if not (tot > 0.0):
        return 0.0
    N = marg.shape[0]
    x = (np.arange(N, dtype=np.float64) - N / 2) * d
    r2 = float((marg * x * x).sum()) / tot
    return float(2.0 * np.sqrt(max(r2, 0.0)))


def _axis_near_focus_needs_bridge(u, R, R_out, wavelength, d, axis):
    """Per-axis analogue of :func:`_near_focus_needs_bridge`: is the same-sign
    (``m>0``) landing so close to this axis's geometric focus that its
    shrunken co-moving grid can no longer hold the diffraction-limited line
    waist?  Cheap ``False`` (no width measurement) away from focus, so the
    fast path is preserved."""
    if abs(R_out) >= _NEAR_FOCUS_FRACTION * abs(R):
        return False
    w_in = _axis_amp_radius(u, d, axis)
    if not (w_in > 0.0):
        return False
    N = np.asarray(u).shape[axis]
    w0 = wavelength * abs(R) / (np.pi * w_in)
    w_geom = w_in * abs(R_out) / abs(R)
    w_out = max(w_geom, w0)
    half = 0.5 * N * d * abs(R_out) / abs(R)
    return half < _BRIDGE_FIT_MARGIN * w_out


def _axis_step_fast(u, R, z, wavelength, d, axis):
    """1-D Sziklas-Siegman fast step on a single ``axis`` (non-crossing,
    ``m = (R+z)/R > 0``).  Returns ``(u_out, R_out, d_out)`` with the envelope
    carrying the full ``exp(i k z)`` piston and the co-moving pitch magnified
    by ``m``.  ``R`` finite/non-zero, ``z != 0``."""
    R_out = R + z
    m = R_out / R
    z_eff = z * R / R_out                        # reduced envelope distance
    u_prop = _fresnel_tf_axis(u, z_eff, wavelength, d, axis)   # carries e^{ik z_eff}
    k = 2.0 * np.pi / wavelength
    # complete the 1-D piston e^{ik z_eff} -> e^{ik z}: z - z_eff = z^2/R_out.
    piston = np.exp(1j * k * (z - z_eff))
    u_out = (piston / np.sqrt(m)) * u_prop
    return u_out, R_out, m * d


def _axis_collimated_step(u, z, wavelength, d, axis):
    """Collimated (``R = +/-inf``) axis: a plain 1-D Fresnel-TF step, grid
    unchanged, carrier stays collimated.  Carries ``exp(i k z)``."""
    return _fresnel_tf_axis(u, z, wavelength, d, axis)


def _axis_bridge(u, R, z, wavelength, d, axis):
    """Auto-split a leg that crosses (or lands within a safety margin of) this
    axis's geometric focus -- the 1-D analogue of
    :func:`_propagate_carrier_focus_crossing`.

    carrier -> (fast 1-D step to a plane a few Rayleigh ranges before the line
    waist) -> reconstruct this axis's carrier -> (exact 1-D ASM across the
    waist) -> re-attach the flipped carrier -> (fast 1-D step to target).  The
    other axis rides along untouched (its carrier is still referenced out).
    Returns ``(u_out, R_out, d_out)`` carrying ``exp(i k z)``."""
    z_f = -R                                     # this axis's focus distance
    w_in = _axis_amp_radius(u, d, axis)
    if not (w_in > 0.0):
        raise ValueError(
            "propagate_carrier_referenced: cannot bridge an astigmatic "
            "carrier focus -- the envelope has no finite width on this axis "
            "to size the near-focus grid.  Reference the beam a finite "
            "distance from its focus on each axis.")
    w0 = wavelength * abs(R) / (np.pi * w_in)
    zR = np.pi * w0 * w0 / wavelength
    delta = _BRIDGE_ZR_FACTOR * zR
    delta = min(delta, 0.45 * abs(z_f))
    if not (delta > 0.0):
        raise ValueError(
            f"propagate_carrier_referenced: an astigmatic carrier focus "
            f"(z_f={z_f:.6g} m) coincides with the start plane, so there is "
            f"no room to hand off before the line waist.  Reference the beam "
            f"a finite distance from its focus on each axis.")
    sgn = 1.0 if z_f >= 0.0 else -1.0
    z_a = z_f - sgn * delta
    z_b = z_f + sgn * delta
    R_b = R + z_b                                # = +sgn*delta (flipped)

    # (a) fast 1-D step start -> a  (well conditioned; carries e^{ik z_a})
    u_a, R_a, d_a = _axis_step_fast(u, R, z_a, wavelength, d, axis)
    # reconstruct this axis's carrier on the (now fine) co-moving grid
    E_a = u_a * _axis_carrier_phase(u_a.shape, d_a, wavelength, R_a, axis, +1)

    beyond_b = (z - z_b) * sgn                   # >0 iff target is past b
    if beyond_b <= 0.0:
        # Target lands inside the bridge zone [a, b] (near the line waist):
        # 1-D ASM straight to it.  Near the waist the true wavefront is flat,
        # so the honest carrier is collimated (R=inf) -- referencing the
        # geometric R_out (~0) would only imprint an aliased fringe and never
        # changes |env|.  (E_a carries e^{ik z_a}; the ASM adds e^{ik(z-z_a)}
        # -> the result carries e^{ik z}.)
        E_t = _asm_axis(E_a, z - z_a, wavelength, d_a, axis)
        return E_t, np.inf, d_a

    # (b) bridge a -> b THROUGH the line waist with an exact 1-D ASM step
    E_b = _asm_axis(E_a, z_b - z_a, wavelength, d_a, axis)
    # (c) re-attach the flipped diverging carrier and continue
    env_b = E_b * _axis_carrier_phase(E_b.shape, d_a, wavelength, R_b, axis, -1)
    return _axis_step_fast(env_b, R_b, z - z_b, wavelength, d_a, axis)


def _axis_step(u, R, z, wavelength, d, axis):
    """Full 1-D Sziklas-Siegman transform of a single ``axis`` over distance
    ``z``: collimated no-magnification step, fast step, or focus-crossing
    bridge, chosen exactly as the scalar driver chooses.  Returns
    ``(u_out, R_out, d_out)`` carrying ``exp(i k z)``."""
    R = float(R)
    if np.isinf(R):
        return _axis_collimated_step(u, z, wavelength, d, axis), R, d
    if R == 0.0:                                 # guarded upstream; belt+braces
        raise ValueError(
            "propagate_carrier_referenced: astigmatic carrier axis radius "
            "== 0 (the carrier's own focus).")
    R_out = R + z
    with np.errstate(divide='ignore', invalid='ignore'):
        m = R_out / R
    if ((not np.isfinite(m)) or (m <= 0.0)
            or _axis_near_focus_needs_bridge(u, R, R_out, wavelength, d, axis)):
        return _axis_bridge(u, R, z, wavelength, d, axis)
    return _axis_step_fast(u, R, z, wavelength, d, axis)


def _propagate_carrier_astigmatic(E_env, R_x, R_y, z, wavelength, dx, dy):
    """Separable astigmatic carrier step (Task 1): apply the 1-D transform on
    axis x then axis y, each with its own magnification and its own
    focus-crossing bridge.  Returns ``CarrierReferencedField`` with ``R`` =
    ``(R_x+z, R_y+z)`` and ``dx`` = ``(dx_out, dy_out)``."""
    u = np.asarray(E_env)
    # axis 1 == x (pitch dx, radius R_x); axis 0 == y (pitch dy, radius R_y).
    u, Rx_out, dx_out = _axis_step(u, R_x, z, wavelength, dx, axis=1)
    u, Ry_out, dy_out = _axis_step(u, R_y, z, wavelength, dy, axis=0)
    # Each 1-D op carries e^{ik z}; the physical field carries it once.
    k = 2.0 * np.pi / wavelength
    u = u * np.exp(-1j * k * z)
    u = _match_env_dtype(u, E_env)
    return CarrierReferencedField(u, (Rx_out, Ry_out), (dx_out, dy_out))


def _build_carrier_phase(shape, dx, dy, wavelength, R_carrier, sign, fn):
    """Carrier phase ``exp(sign*i*k*[x^2/(2R_x) + y^2/(2R_y)])`` for a scalar
    or ``(R_x, R_y)`` carrier.  Returns ``None`` when the carrier is fully
    collimated (a no-op).  Collimated axes of an astigmatic carrier drop out
    of the per-axis product."""
    R_x, R_y, is_astig = _parse_carrier(R_carrier, fn)
    if not is_astig:
        R = R_x
        if np.isinf(R):
            return None
        if R == 0.0:
            raise ValueError(f"{fn}: R_carrier == 0 (carrier focus).")
        return _radial_carrier_phase(shape, dx, dy, wavelength, float(R), sign)
    if R_x == 0.0 or R_y == 0.0:
        raise ValueError(
            f"{fn}: an astigmatic carrier axis radius == 0 (carrier focus).")
    phase = None
    if np.isfinite(R_x):                          # axis 1 == x
        phase = _axis_carrier_phase(shape, dx, wavelength, float(R_x), 1, sign)
    if np.isfinite(R_y):                          # axis 0 == y
        py = _axis_carrier_phase(shape, dy, wavelength, float(R_y), 0, sign)
        phase = py if phase is None else phase * py
    return phase


def carrier_referenced_reconstruct(
    E_env: np.ndarray,
    R_carrier: Union[float, Tuple[float, float]],
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
    R_carrier : float or (float, float)
        Signed carrier radius (m); ``+/-inf`` -> collimated (no-op).  A
        2-tuple ``(R_x, R_y)`` imprints the separable ASTIGMATIC carrier
        ``exp(i*k*[x^2/(2*R_x) + y^2/(2*R_y)])`` (a collimated axis drops
        out); pass the astigmatic ``dx``/``dy`` alongside it.
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
    phase = _build_carrier_phase(E_env.shape, dx, dy, wavelength, R_carrier,
                                 +1, 'carrier_referenced_reconstruct')
    if phase is None:
        return E_env.copy() if hasattr(E_env, 'copy') else np.array(E_env)
    if np.iscomplexobj(E_env):
        phase = phase.astype(E_env.dtype, copy=False)
    return E_env * phase


def carrier_referenced_envelope(
    E_full: np.ndarray,
    R_carrier: Union[float, Tuple[float, float]],
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
    no-op.  A 2-tuple ``(R_x, R_y)`` removes the separable astigmatic
    carrier (a collimated axis drops out).

    NOTE (prototype): extracting an envelope on a grid that does NOT
    already sample the carrier fringe aliases it -- this helper is for
    grids that resolve the full field (e.g. at an element plane during a
    hand-off), not for recovering the memory win after the fact.

    Parameters
    ----------
    E_full : ndarray, complex, shape (Ny, Nx)
    R_carrier : float or (float, float)
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
    phase = _build_carrier_phase(E_full.shape, dx, dy, wavelength, R_carrier,
                                 -1, 'carrier_referenced_envelope')
    if phase is None:
        return E_full.copy() if hasattr(E_full, 'copy') else np.array(E_full)
    if np.iscomplexobj(E_full):
        phase = phase.astype(E_full.dtype, copy=False)
    return E_full * phase


# ===========================================================================
# Apertures (Task 2): hard clip on the envelope grid + carrier hardening
# ===========================================================================


def _fit_carrier_inv(E, wavelength, dx, dy, axis=None):
    """Intensity-weighted mean wavefront inverse-curvature ``1/R`` of a field
    (0.0 for a flat/collimated wavefront).

    From ``E = A exp(i*phi)`` with ``phi = +k r^2/(2R)`` (the library's
    diverging-``R>0`` convention), ``Im[E* (x dE/dx)] = k A^2 x^2 / R``, so
    ``1/R_x = Im[sum E* x dE/dx] / (k sum |E|^2 x^2)`` -- a phase-unwrap-free,
    aperture-robust estimator.  ``axis=None`` fits the isotropic (combined
    ``x^2+y^2``) curvature; ``axis=1``/``0`` fit x/y separately."""
    E = np.asarray(E)
    Ny, Nx = E.shape[-2], E.shape[-1]
    k = 2.0 * np.pi / wavelength
    x = (np.arange(Nx, dtype=np.float64) - Nx / 2) * dx
    y = (np.arange(Ny, dtype=np.float64) - Ny / 2) * dy
    Y, X = np.meshgrid(y, x, indexing='ij')
    inten = np.abs(E) ** 2
    if axis is None:
        dE = np.gradient(E, dx, axis=1)
        num = np.imag(np.sum(np.conj(E) * X * dE))
        dE = np.gradient(E, dy, axis=0)
        num += np.imag(np.sum(np.conj(E) * Y * dE))
        den = k * float(np.sum(inten * (X * X + Y * Y)))
    elif axis == 1:
        dE = np.gradient(E, dx, axis=1)
        num = np.imag(np.sum(np.conj(E) * X * dE))
        den = k * float(np.sum(inten * X * X))
    else:
        dE = np.gradient(E, dy, axis=0)
        num = np.imag(np.sum(np.conj(E) * Y * dE))
        den = k * float(np.sum(inten * Y * Y))
    num = float(num)
    if den == 0.0 or not (np.isfinite(num) and np.isfinite(den)):
        return 0.0
    return num / den


def carrier_referenced_fit_radius(
    E_full: np.ndarray,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    astigmatic: bool = False,
) -> Union[float, Tuple[float, float]]:
    """Best-fit spherical carrier radius ``R`` of a full field's wavefront.

    Returns the intensity-weighted mean radius (``+inf`` for a collimated
    wavefront) using the phase-unwrap-free estimator in
    :func:`_fit_carrier_inv`.  With ``astigmatic=True`` returns the per-axis
    pair ``(R_x, R_y)``.  Use it to reference a measured / apertured field to
    its actual wavefront so the envelope is flat and the co-moving grid is
    well conditioned.

    Parameters
    ----------
    E_full : ndarray, complex, shape (Ny, Nx)
        Full physical field (carrier NOT divided out).
    wavelength, dx : float
    dy : float, optional
        Defaults to ``dx``.
    astigmatic : bool, default False
        If True, fit ``R_x`` and ``R_y`` independently.
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_full, 'carrier_referenced_fit_radius')
    if dy is None:
        dy = dx
    if astigmatic:
        ix = _fit_carrier_inv(E_full, wavelength, dx, dy, axis=1)
        iy = _fit_carrier_inv(E_full, wavelength, dx, dy, axis=0)
        return (np.inf if ix == 0.0 else 1.0 / ix,
                np.inf if iy == 0.0 else 1.0 / iy)
    inv = _fit_carrier_inv(E_full, wavelength, dx, dy, axis=None)
    return np.inf if inv == 0.0 else 1.0 / inv


def _rereference(env, R_old, R_new, wavelength, dx, dy):
    """Move an envelope from carrier ``R_old=(Rx,Ry)`` to ``R_new=(Rx,Ry)``:
    ``env * exp(i*k*r^2/2 * (1/R_old - 1/R_new))`` per axis."""
    Rox, Roy = R_old
    Rnx, Rny = R_new
    Ny, Nx = env.shape[-2], env.shape[-1]
    k = 2.0 * np.pi / wavelength
    x = (np.arange(Nx, dtype=np.float64) - Nx / 2) * dx
    y = (np.arange(Ny, dtype=np.float64) - Ny / 2) * dy
    Y, X = np.meshgrid(y, x, indexing='ij')

    def _inv(R):
        return 0.0 if np.isinf(R) else 1.0 / float(R)

    dphi = 0.5 * k * ((X * X) * (_inv(Rox) - _inv(Rnx))
                      + (Y * Y) * (_inv(Roy) - _inv(Rny)))
    return env * np.exp(1j * dphi)


def carrier_referenced_aperture(
    E_env: np.ndarray,
    R_carrier: Union[float, Tuple[float, float]],
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    *,
    radius: Optional[float] = None,
    half_x: Optional[float] = None,
    half_y: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
    refit_carrier: bool = False,
    new_carrier: Optional[Union[float, Tuple[float, float]]] = None,
    return_transmission: bool = False,
):
    """Apply a hard aperture to a carrier-referenced field on its co-moving
    grid, removing the clipped power exactly (Task 2).

    A hard stop clips the ENVELOPE in place: the carrier is a pure phase, so
    ``|env|`` is clipped exactly as ``|E_full|`` would be, and the same real
    mask serves both.  The clipped power is genuinely REMOVED -- the envelope
    is never renormalised, so a downstream power accounting sees the true
    transmitted energy.

    Carrier after the clip.  A hard amplitude aperture does not change the
    wavefront curvature at the surviving points, so by default the carrier
    ``R`` is returned UNCHANGED (this is the physically correct default).
    Two opt-in modes harden it:

    * ``refit_carrier=True`` -- re-fit ``R`` from the apertured envelope
      (:func:`_fit_carrier_inv`) and re-reference to it, absorbing any
      residual envelope curvature (e.g. the geometric-vs-Gaussian ``R``
      mismatch accumulated over a chain) so the downstream envelope is flat.
    * ``new_carrier=R`` (scalar or ``(R_x, R_y)``) -- re-reference to a
      caller-supplied conjugate (e.g. the aperture is a new pupil for a
      downstream relay).  NOTE: a large ``R`` change re-imprints a steep
      fringe; the grid must resolve it (same caveat as
      :func:`carrier_referenced_envelope`).

    Parameters
    ----------
    E_env : ndarray, complex, shape (Ny, Nx)
        Envelope on the co-moving grid.
    R_carrier : float or (float, float)
        The field's current carrier radius (scalar or astigmatic).
    wavelength, dx : float
    dy : float, optional
        Defaults to ``dx``.
    radius : float, optional
        Circular hard-stop radius (m): keeps ``x^2 + y^2 <= radius^2``.
    half_x, half_y : float, optional
        Rectangular hard-stop half-widths (m): keeps ``|x| <= half_x`` and
        ``|y| <= half_y`` (either may be omitted for a slit).
    mask : ndarray, optional
        Explicit real transmission mask, same shape as ``E_env`` (overrides
        ``radius`` / ``half_x`` / ``half_y``).  Values need not be 0/1 (a
        soft/graded stop is allowed), but the "hard aperture" energy contract
        holds for any real mask.
    refit_carrier : bool, default False
        Re-fit and re-reference ``R`` from the apertured envelope.
    new_carrier : float or (float, float), optional
        Re-reference to this conjugate instead (mutually exclusive with
        ``refit_carrier``).
    return_transmission : bool, default False
        If True, return ``(field, transmitted_fraction)`` instead of just the
        field.

    Returns
    -------
    CarrierReferencedField
        ``(env, R, dx)`` with the apertured envelope; ``R`` / ``dx`` are
        2-tuples when the (returned) carrier is astigmatic.  If
        ``return_transmission`` is True, a ``(field, fraction)`` pair.
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_env, 'carrier_referenced_aperture')
    if dy is None:
        dy = dx
    if refit_carrier and new_carrier is not None:
        raise ValueError(
            "carrier_referenced_aperture: pass at most one of refit_carrier "
            "and new_carrier.")
    env = np.asarray(E_env)
    Ny, Nx = env.shape[-2], env.shape[-1]

    # -- build the (real) aperture mask -------------------------------------
    if mask is not None:
        msk = np.asarray(mask)
        if msk.shape != env.shape:
            raise ValueError(
                f"carrier_referenced_aperture: mask shape {msk.shape} != "
                f"field shape {env.shape}.")
        msk = msk.astype(np.float64)
    elif radius is not None or half_x is not None or half_y is not None:
        x = (np.arange(Nx, dtype=np.float64) - Nx / 2) * dx
        y = (np.arange(Ny, dtype=np.float64) - Ny / 2) * dy
        Y, X = np.meshgrid(y, x, indexing='ij')
        if radius is not None:
            if not (radius > 0):
                raise ValueError(
                    "carrier_referenced_aperture: radius must be > 0.")
            msk = (X * X + Y * Y <= float(radius) ** 2).astype(np.float64)
        else:
            hx = np.inf if half_x is None else float(half_x)
            hy = np.inf if half_y is None else float(half_y)
            if not (hx > 0 and hy > 0):
                raise ValueError(
                    "carrier_referenced_aperture: half_x / half_y must be > 0.")
            msk = ((np.abs(X) <= hx) & (np.abs(Y) <= hy)).astype(np.float64)
    else:
        raise ValueError(
            "carrier_referenced_aperture: specify an aperture via radius=, "
            "half_x= / half_y=, or mask=.")

    # -- energy accounting (fraction; the dx*dy area element cancels) -------
    p_before = float((np.abs(env) ** 2).sum())
    env_ap = env * msk
    p_after = float((np.abs(env_ap) ** 2).sum())
    transmission = (p_after / p_before) if p_before > 0.0 else 0.0
    # NO renormalisation: the clipped power is genuinely gone.

    R_x, R_y, in_astig = _parse_carrier(R_carrier, 'carrier_referenced_aperture')

    # -- carrier: keep / re-fit / re-reference ------------------------------
    if new_carrier is not None:
        Nx_new, Ny_new, out_astig = _parse_carrier(
            new_carrier, 'carrier_referenced_aperture')
        env_out = _rereference(env_ap, (R_x, R_y), (Nx_new, Ny_new),
                               wavelength, dx, dy)
        R_out = (Nx_new, Ny_new) if out_astig else Nx_new
    elif refit_carrier:
        if in_astig:
            ivx = _fit_carrier_inv(env_ap, wavelength, dx, dy, axis=1)
            ivy = _fit_carrier_inv(env_ap, wavelength, dx, dy, axis=0)
            inx = (0.0 if np.isinf(R_x) else 1.0 / R_x) + ivx
            iny = (0.0 if np.isinf(R_y) else 1.0 / R_y) + ivy
            Rnx = np.inf if inx == 0.0 else 1.0 / inx
            Rny = np.inf if iny == 0.0 else 1.0 / iny
            env_out = _rereference(env_ap, (R_x, R_y), (Rnx, Rny),
                                   wavelength, dx, dy)
            R_out = (Rnx, Rny) if (Rnx != Rny) else Rnx
            out_astig = (Rnx != Rny)
        else:
            iv = _fit_carrier_inv(env_ap, wavelength, dx, dy, axis=None)
            inv_new = (0.0 if np.isinf(R_x) else 1.0 / R_x) + iv
            Rn = np.inf if inv_new == 0.0 else 1.0 / inv_new
            env_out = _rereference(env_ap, (R_x, R_x), (Rn, Rn),
                                   wavelength, dx, dy)
            R_out = Rn
            out_astig = False
    else:
        env_out = env_ap
        R_out = (R_x, R_y) if in_astig else R_x
        out_astig = in_astig

    env_out = _match_env_dtype(env_out, E_env)
    dx_out = (dx, dy) if out_astig else dx
    field = CarrierReferencedField(env_out, R_out, dx_out)
    if return_transmission:
        return field, transmission
    return field
