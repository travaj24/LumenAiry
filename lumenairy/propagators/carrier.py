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

from typing import Callable, NamedTuple, Optional, Tuple, Union

import numpy as np

from .fresnel import fresnel_tf_propagate

__all__ = [
    'CarrierReferencedField',
    'TracedCarrierChainResult',
    'TracedCarrierChainMultiResult',
    'propagate_carrier_referenced',
    'carrier_referenced_reconstruct',
    'carrier_referenced_envelope',
    'carrier_referenced_aperture',
    'carrier_referenced_fit_radius',
    'carrier_referenced_focus_readout',
    'carrier_referenced_exact_focus_readout',
    'propagate_traced_carrier_chain',
    'propagate_traced_carrier_chain_multi',
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


# ===========================================================================
# Backend abstraction (Phase K2, plan N14): NumPy / CuPy / JAX
# ===========================================================================
# The carrier ASM is backend-agnostic.  The backend is selected -- exactly
# like every other lumenairy propagator (RCWA / PMM / ASM / Fresnel) -- by the
# array type the caller passes in: ``array_namespace(E)`` returns ``numpy``,
# ``cupy`` or ``jax.numpy`` and the FFTs + envelope-grid ops route through it.
# The algorithm is identical; this is a backend PORT, accuracy-neutral.
#
# FIELD-INDEPENDENT float64 grids / phase screens / transfer functions are
# built on ``bld`` -- host NumPy for the JAX backend (so the large
# quadratic-phase argument ``k r^2 / 2R`` up to ~1e5 rad is not silently
# truncated to float32 when ``jax_enable_x64`` is off -- the audit S2-3
# contract the ASM / Fresnel kernels already honour) and the device namespace
# otherwise -- then moved onto the device with :func:`_to_dev`.  The NumPy path
# (``xp is bld is np``) runs the historical arithmetic verbatim and is
# byte-identical (tolerance-pinned) to prior releases.
#
# The data-dependent focus-crossing split runs EAGERLY: its branch conditions
# are host-side reductions of the field (``_envelope_amp_radius`` etc. return
# Python floats), so no ``lax.cond`` is needed -- an eager JAX array
# concretises them transparently.  The non-crossing fast leg touches no
# field-derived branch, so it is fully ``jax.jit`` / ``jax.grad`` compatible
# (a gradient through a carrier leg is validated in the K2 tests).  A jitted
# focus-crossing split would need ``lax.cond``; that is out of scope (the eager
# path is the documented one).


def _backend_of(E):
    """Return ``(xp, is_jax, bld)`` for a field ``E``: its array namespace,
    whether it is a JAX array, and the module used to build field-independent
    float64 grids (host NumPy for JAX, the device namespace otherwise)."""
    from ..backend import array_namespace, is_jax_array
    xp = array_namespace(E)
    is_jax = is_jax_array(E)
    return xp, is_jax, (np if is_jax else xp)


def _to_dev(a, xp, is_jax):
    """Move a ``bld``-built array onto the field's device.  Only JAX needs the
    explicit host->device hop (CuPy grids are built on-device via ``bld=xp``;
    NumPy is a no-op)."""
    return xp.asarray(a) if is_jax else a


def _is_complex(x):
    """Complex-dtype predicate that inspects ``x.dtype`` only (never
    materialises a device / traced array), so it is safe under ``jax.grad``."""
    return np.issubdtype(x.dtype, np.complexfloating)


def _cdtype_of(x):
    """Target complex dtype for a field: its own dtype if complex, else the
    library default complex dtype."""
    if _is_complex(x):
        return x.dtype
    from .fft_infra import DEFAULT_COMPLEX_DTYPE
    return np.dtype(DEFAULT_COMPLEX_DTYPE)


def _freq_sq_1d_bld(N, d, bld):
    """Centred ``(2*pi*f)^2`` float64 vector on backend ``bld`` -- the ``bld``
    generalisation of :func:`_freq_sq_1d` (identical values for ``bld is np``)."""
    f = (bld.arange(N, dtype=np.float64) - N / 2) / (N * d)
    return (2.0 * np.pi * f) ** 2


def _tf_phase_to_H(arg, target_cdtype, xp, is_jax, bld):
    """``exp(1j*arg)`` at ``target_cdtype`` on the field's backend.

    * NumPy (``xp is np``): the historical direct complex128 exponential
      ``np.exp(1j*arg)`` -- byte-identical to the pre-K2 code (the caller casts
      the finished field to its own dtype, exactly as before).
    * CuPy / JAX: dtype-aware.  complex64 folds the phase ``mod 2*pi`` in
      float64 BEFORE the float32 cast (the audit S2-3 mitigation, so a large
      ``arg`` does not hit the float32 floor); complex128 the direct
      exponential.  Built on ``bld`` (host f64 for JAX) then moved on-device.
    """
    if xp is np:
        return np.exp(1j * arg)
    tcd = np.dtype(target_cdtype)
    if tcd == np.complex64:
        ph = bld.mod(arg, 2.0 * np.pi)
        H = bld.empty(arg.shape, dtype=np.complex64)
        H.real[...] = bld.cos(ph).astype(np.float32)
        H.imag[...] = bld.sin(ph).astype(np.float32)
    else:
        H = bld.exp(1j * arg).astype(tcd)
    return _to_dev(H, xp, is_jax)


def _fresnel_tf_2d_xp(E, z, wavelength, dx, dy, xp, is_jax, bld):
    """Backend (CuPy / JAX) same-grid Fresnel transfer-function step -- the
    ``xp`` analogue of :func:`fresnel_tf_propagate` (the NumPy path keeps using
    the pyFFTW-backed ``fresnel_tf_propagate`` for its byte-identical fast
    FFT).  ``H`` is built natural-layout on ``bld`` in float64 and moved onto
    the device; the FFT pair runs in the field's namespace.  Carries the
    on-axis piston ``exp(i k z)`` (the ``k*z`` term), matching
    ``fresnel_tf_propagate``."""
    Ny, Nx = E.shape
    k = 2.0 * np.pi / wavelength
    kx_sq = bld.fft.ifftshift(_freq_sq_1d_bld(Nx, dx, bld))
    ky_sq = bld.fft.ifftshift(_freq_sq_1d_bld(Ny, dy, bld))
    arg = (k * z) - (z / (2.0 * k)) * (ky_sq[:, None] + kx_sq[None, :])
    H = _tf_phase_to_H(arg, _cdtype_of(E), xp, is_jax, bld)
    out = xp.fft.ifft2(xp.fft.fft2(E) * H)
    if _is_complex(E) and out.dtype != E.dtype:
        out = out.astype(E.dtype)
    return out


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


def _radial_carrier_phase(shape, dx, dy, wavelength, R, sign, bld=np,
                          centre=(0.0, 0.0)):
    """``exp(sign*i*k*(x^2+y^2)/(2R))`` on the centred grid (float64
    carrier argument, cast to nothing here -- caller casts).  Built on backend
    ``bld`` (host NumPy for JAX, the device namespace otherwise); ``bld is np``
    reproduces the historical NumPy screen byte-for-byte.

    ``centre`` (niche D1) DECENTRES the parabola to ``(x0, y0)`` -- the
    transverse position of a tilted congruence's chief ray.  The default
    ``(0, 0)`` is short-circuited, so the on-axis screen is untouched."""
    Ny, Nx = shape
    x = (bld.arange(Nx, dtype=np.float64) - Nx / 2) * dx
    y = (bld.arange(Ny, dtype=np.float64) - Ny / 2) * dy
    if centre != (0.0, 0.0):
        x = x - float(centre[0])
        y = y - float(centre[1])
    Y, X = bld.meshgrid(y, x, indexing='ij')
    r2 = X * X + Y * Y
    k = 2.0 * np.pi / wavelength
    return bld.exp(sign * 1j * k * r2 / (2.0 * R))


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
    _check_2d_scalar_field(E_env, 'propagate_carrier_referenced',
                           input_kind='field')
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
        xp, is_jax, bld = _backend_of(E_env)
        if xp is np:
            env_out = fresnel_tf_propagate(E_env, z, wavelength, dx, dy)
        else:
            env_out = _fresnel_tf_2d_xp(
                E_env, z, wavelength, dx, dy, xp, is_jax, bld)
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
    well-conditioned carrier legs.  Returns ``(env_out, R_out, dx_out)``.

    Backend-agnostic (K2): NumPy keeps the pyFFTW-backed
    :func:`fresnel_tf_propagate` for a byte-identical fast FFT; CuPy / JAX route
    the envelope leg through :func:`_fresnel_tf_2d_xp` in the field's
    namespace.  The leg is data-branch-free, so it is ``jax.jit`` / ``jax.grad``
    compatible."""
    xp, is_jax, bld = _backend_of(E_env)
    R_out = R + z
    m = R_out / R
    # Reduced (collimated-frame) envelope distance.  z_eff = z / m.
    z_eff = z * R / R_out

    # Envelope leg: ordinary collimated Fresnel TF step, SAME grid.
    if xp is np:
        u_out = fresnel_tf_propagate(E_env, z_eff, wavelength, dx, dy)
    else:
        u_out = _fresnel_tf_2d_xp(E_env, z_eff, wavelength, dx, dy,
                                  xp, is_jax, bld)

    # On-axis piston the reduced leg under-counts: z - z_eff = z^2 / R_out.
    k = 2.0 * np.pi / wavelength
    piston = np.exp(1j * k * (z * z / R_out))
    # 2-D power conservation: the co-moving area element grows by m^2, so
    # the amplitude carries 1/m (each axis 1/sqrt(m)).
    if xp is np:
        # Historical arithmetic -- pinned byte-identical.
        scale = piston / m
        env_out = scale * u_out
        # Preserve the input complex dtype (the numpy-complex scalar would
        # otherwise upcast complex64 -> complex128).
        if np.iscomplexobj(E_env) and env_out.dtype != E_env.dtype:
            env_out = env_out.astype(E_env.dtype)
    else:
        # Weak Python-complex scalar so a complex64 field is NOT upcast on a
        # backend (a numpy complex128 scalar promotes a jnp complex64 array
        # when jax_enable_x64 is on; a python complex stays weakly-typed).
        env_out = complex(piston / m) * u_out
        if _is_complex(E_env) and env_out.dtype != E_env.dtype:
            env_out = env_out.astype(E_env.dtype)

    return CarrierReferencedField(env_out, R_out, m * dx)


def _envelope_amp_radius(E_env, dx, dy, centre=(0.0, 0.0)):
    """1/e AMPLITUDE radius of the (assumed roughly Gaussian) envelope on the
    centred grid, from the intensity second moment (``w = sqrt(2)*r2m``).
    Returns 0.0 for an empty / zero field.

    ``centre`` (niche D6) takes the second moment about ``(x0, y0)`` instead of
    the grid origin -- the chief-ray position of a tilted congruence.  A
    second moment about the WRONG point reads ``sqrt(2 x_c^2 + w^2)``, so an
    off-axis beam's radius (and every window sized from it) grows with the
    DECENTRE rather than following the beam.  The default is short-circuited,
    so the on-axis reading is untouched."""
    # Host-side (Python-float) branch reduction: pull the field to host on ANY
    # backend.  ``np.asarray`` raises on a CuPy device array (implicit transfer
    # is blocked); ``to_numpy`` uses ``cupy.asnumpy`` / eager JAX copy and is a
    # byte-identical no-op for a NumPy field.
    from ..backend import to_numpy
    I = np.abs(to_numpy(E_env)) ** 2
    tot = float(I.sum())
    if not (tot > 0.0):
        return 0.0
    Ny, Nx = I.shape[-2], I.shape[-1]
    x = (np.arange(Nx, dtype=np.float64) - Nx / 2) * dx
    y = (np.arange(Ny, dtype=np.float64) - Ny / 2) * dy
    if centre != (0.0, 0.0):
        x = x - float(centre[0])
        y = y - float(centre[1])
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
    Nx = E_env.shape[-1]
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
    (preserve complex64 through the focus-crossing path).  Inspects ``.dtype``
    only, so it never concretises a CuPy / JAX ``env`` off its device."""
    if _is_complex(ref) and env.dtype != ref.dtype:
        return env.astype(ref.dtype)
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
    faithful.  ``z`` here is the (already reduced) envelope distance.

    Backend-agnostic (K2): the 1-D FFT pair runs in the field's namespace
    (``xp.fft``, which for NumPy is ``np.fft`` -- byte-identical to the prior
    code, no pyFFTW is involved on this 1-D path); the transfer function is
    built natural-layout on ``bld`` and moved on-device."""
    xp, is_jax, bld = _backend_of(u)
    N = u.shape[axis]
    k = 2.0 * np.pi / wavelength
    kx_sq = bld.fft.ifftshift(_freq_sq_1d_bld(N, d, bld))
    # phase = k*z - (z/2k) (2 pi f)^2  (== k*z - pi*lambda*z*f^2), 1-D.
    arg = k * z - (z / (2.0 * k)) * kx_sq
    H = _tf_phase_to_H(arg, _cdtype_of(u), xp, is_jax, bld)
    H = _broadcast_axis(H, u.ndim, axis)
    Uf = xp.fft.fft(u, axis=axis)
    return xp.fft.ifft(Uf * H, axis=axis)


def _asm_axis(E, z, wavelength, d, axis, bandlimit=True):
    """1-D exact (band-limited) angular-spectrum step along a single ``axis``
    -- the per-axis analogue of :func:`angular_spectrum_propagate`, used only
    inside the near-focus bridge to carry the compact field across the waist
    on the (now fine) co-moving grid.  Carries the on-axis piston
    ``exp(i k z)`` (``kz(f=0) = k``).  The other axis is a spectator (its
    slowly-varying, carrier-referenced content has little transverse
    frequency, so the neglected ``k_y`` coupling in ``sqrt(k^2 - k_x^2)`` is
    a small paraxial residual -- the regime this bridge operates in).

    Backend-agnostic (K2): the FFT pair runs in the field's namespace; the
    evanescent-masked, band-limited transfer function is built on ``bld``
    (host f64 for JAX) and moved on-device.  ``xp is bld is np`` is
    byte-identical to the historical code."""
    xp, is_jax, bld = _backend_of(E)
    if z == 0:
        return E.copy() if hasattr(E, 'copy') else np.array(E)
    N = E.shape[axis]
    k = 2.0 * np.pi / wavelength
    kx_sq = _freq_sq_1d_bld(N, d, bld)
    kz_sq = k * k - kx_sq
    prop = kz_sq > 0
    kz = bld.where(prop, bld.sqrt(bld.maximum(kz_sq, 0.0)), 0.0)
    if xp is np:
        Hc = np.exp(1j * z * kz)
    else:
        # dtype-aware exp (S2-3): complex64 folds the phase mod 2*pi in f64
        # before the float32 cast; complex128 the direct exponential.
        tcd = np.dtype(_cdtype_of(E))
        arg = z * kz
        if tcd == np.complex64:
            phf = bld.mod(arg, 2.0 * np.pi)
            Hc = bld.empty(arg.shape, dtype=np.complex64)
            Hc.real[...] = bld.cos(phf).astype(np.float32)
            Hc.imag[...] = bld.sin(phf).astype(np.float32)
        else:
            Hc = bld.exp(1j * arg).astype(tcd)
    # dtype-aware zero (v4.14.1 audit P1-NEW-4): a literal ``0.0 + 0.0j`` is
    # complex128 and would silently upcast a complex64 transfer function.
    H = bld.where(prop, Hc, bld.zeros((), dtype=Hc.dtype))
    if bandlimit:
        f = (bld.arange(N, dtype=np.float64) - N / 2) / (N * d)
        f_max = (N * d) / (2.0 * wavelength * abs(z))
        H = bld.where(bld.abs(f) < f_max, H, bld.zeros((), dtype=H.dtype))
    H = _broadcast_axis(bld.fft.ifftshift(H), E.ndim, axis)
    H = _to_dev(H, xp, is_jax)
    Ef = xp.fft.fft(E, axis=axis)
    return xp.fft.ifft(Ef * H, axis=axis)


def _axis_carrier_phase(shape, d, wavelength, R, axis, sign, bld=np):
    """``exp(sign*i*k*x_a^2/(2R))`` broadcast along a single ``axis`` (the
    per-axis carrier).  Built on backend ``bld`` (host NumPy for JAX);
    ``bld is np`` reproduces the historical NumPy screen byte-for-byte."""
    N = shape[axis]
    x = (bld.arange(N, dtype=np.float64) - N / 2) * d
    k = 2.0 * np.pi / wavelength
    ph = bld.exp(sign * 1j * k * x * x / (2.0 * R))
    return _broadcast_axis(ph, len(shape), axis)


def _axis_amp_radius(u, d, axis):
    """1/e amplitude radius of the envelope along a single ``axis`` from the
    marginal-intensity second moment.  For a Gaussian amplitude
    ``exp(-x^2/w^2)`` the marginal intensity ``exp(-2x^2/w^2)`` has
    ``<x^2> = w^2/4``, so ``w = 2*sqrt(<x^2>)`` (the 1-D analogue of the 2-D
    ``w = sqrt(2)*r2m``, which folds in both axes).  0.0 for a zero/empty
    field."""
    # Host-side reduction -> Python float; CuPy-safe pull (see
    # ``_envelope_amp_radius``).
    from ..backend import to_numpy
    inten = np.abs(to_numpy(u)) ** 2
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
    N = u.shape[axis]
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
    xp, is_jax, bld = _backend_of(u)
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
    E_a = u_a * _to_dev(
        _axis_carrier_phase(u_a.shape, d_a, wavelength, R_a, axis, +1, bld),
        xp, is_jax)

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
    env_b = E_b * _to_dev(
        _axis_carrier_phase(E_b.shape, d_a, wavelength, R_b, axis, -1, bld),
        xp, is_jax)
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
    u = E_env
    # axis 1 == x (pitch dx, radius R_x); axis 0 == y (pitch dy, radius R_y).
    u, Rx_out, dx_out = _axis_step(u, R_x, z, wavelength, dx, axis=1)
    u, Ry_out, dy_out = _axis_step(u, R_y, z, wavelength, dy, axis=0)
    # Each 1-D op carries e^{ik z}; the physical field carries it once.
    k = 2.0 * np.pi / wavelength
    u = u * np.exp(-1j * k * z)
    u = _match_env_dtype(u, E_env)
    return CarrierReferencedField(u, (Rx_out, Ry_out), (dx_out, dy_out))


def _build_carrier_phase(shape, dx, dy, wavelength, R_carrier, sign, fn,
                         bld=np):
    """Carrier phase ``exp(sign*i*k*[x^2/(2R_x) + y^2/(2R_y)])`` for a scalar
    or ``(R_x, R_y)`` carrier.  Returns ``None`` when the carrier is fully
    collimated (a no-op).  Collimated axes of an astigmatic carrier drop out
    of the per-axis product.  Built on backend ``bld`` (host NumPy for JAX);
    ``bld is np`` reproduces the historical NumPy screen byte-for-byte."""
    R_x, R_y, is_astig = _parse_carrier(R_carrier, fn)
    if not is_astig:
        R = R_x
        if np.isinf(R):
            return None
        if R == 0.0:
            raise ValueError(f"{fn}: R_carrier == 0 (carrier focus).")
        return _radial_carrier_phase(shape, dx, dy, wavelength, float(R), sign,
                                     bld)
    if R_x == 0.0 or R_y == 0.0:
        raise ValueError(
            f"{fn}: an astigmatic carrier axis radius == 0 (carrier focus).")
    phase = None
    if np.isfinite(R_x):                          # axis 1 == x
        phase = _axis_carrier_phase(shape, dx, wavelength, float(R_x), 1, sign,
                                    bld)
    if np.isfinite(R_y):                          # axis 0 == y
        py = _axis_carrier_phase(shape, dy, wavelength, float(R_y), 0, sign,
                                 bld)
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
    _check_2d_scalar_field(E_env, 'carrier_referenced_reconstruct',
                           input_kind='field')
    if dy is None:
        dy = dx
    xp, is_jax, bld = _backend_of(E_env)
    phase = _build_carrier_phase(E_env.shape, dx, dy, wavelength, R_carrier,
                                 +1, 'carrier_referenced_reconstruct', bld)
    if phase is None:
        return E_env.copy() if hasattr(E_env, 'copy') else np.array(E_env)
    if _is_complex(E_env):
        phase = phase.astype(E_env.dtype, copy=False)
    phase = _to_dev(phase, xp, is_jax)
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
    _check_2d_scalar_field(E_full, 'carrier_referenced_envelope',
                           input_kind='field')
    if dy is None:
        dy = dx
    xp, is_jax, bld = _backend_of(E_full)
    phase = _build_carrier_phase(E_full.shape, dx, dy, wavelength, R_carrier,
                                 -1, 'carrier_referenced_envelope', bld)
    if phase is None:
        return E_full.copy() if hasattr(E_full, 'copy') else np.array(E_full)
    if _is_complex(E_full):
        phase = phase.astype(E_full.dtype, copy=False)
    phase = _to_dev(phase, xp, is_jax)
    return E_full * phase


# ===========================================================================
# Apertures (Task 2): hard clip on the envelope grid + carrier hardening
# ===========================================================================


# Fraction of the grid's Nyquist per-pixel phase step (pi rad/px) at which a
# wrapping-safe increment reading counts as (approaching) ALIASED -- the same
# constant and the same reasoning as the element-side
# ``_AUTO_CARRIER_NYQUIST_FRAC`` (elements/_lens_traced.py, audit R6/F1).
_FIT_CARRIER_NYQUIST_FRAC = 0.5
# Fraction of the BRIGHT support that must read aliased before the fit's
# reliability guard fires (mirrors ``_AUTO_CARRIER_ALIAS_FRAC``).
_FIT_CARRIER_ALIAS_FRAC = 0.05


def _carrier_fit_alias_fraction(E, bright_frac=0.05):
    """Fraction of the BRIGHT support whose wrapping-safe nearest-neighbour
    phase increment reads at/above ``_FIT_CARRIER_NYQUIST_FRAC * pi`` rad/px --
    i.e. how much of the beam sits at or beyond the grid's ability to represent
    its own carrier tilt.  Returns 0.0 when it cannot be formed."""
    from ..backend import to_numpy
    E = np.asarray(to_numpy(E))
    mag = np.abs(E)
    if mag.size == 0:
        return 0.0
    mx = float(mag.max())
    if not np.isfinite(mx) or mx <= 0.0:
        return 0.0
    bright = mag > bright_frac * mx
    if not bright.any():
        return 0.0
    thr = _FIT_CARRIER_NYQUIST_FRAC * np.pi
    n_tot = 0
    n_bad = 0
    for g, mk in ((np.abs(np.angle(E[:, 1:] * np.conj(E[:, :-1]))),
                   bright[:, 1:] & bright[:, :-1]),
                  (np.abs(np.angle(E[1:, :] * np.conj(E[:-1, :]))),
                   bright[1:, :] & bright[:-1, :])):
        n_tot += int(mk.sum())
        n_bad += int((g[mk] >= thr).sum())
    if n_tot == 0:
        return 0.0
    return n_bad / float(n_tot)


def _fit_carrier_inv(E, wavelength, dx, dy, axis=None, estimator='gradient'):
    """Intensity-weighted mean wavefront inverse-curvature ``1/R`` of a field
    (0.0 for a flat/collimated wavefront).

    From ``E = A exp(i*phi)`` with ``phi = +k r^2/(2R)`` (the library's
    diverging-``R>0`` convention), ``Im[E* (x dE/dx)] = k A^2 x^2 / R``, so
    ``1/R_x = Im[sum E* x dE/dx] / (k sum |E|^2 x^2)`` -- a phase-unwrap-free,
    aperture-robust estimator.  ``axis=None`` fits the isotropic (combined
    ``x^2+y^2``) curvature; ``axis=1``/``0`` fit x/y separately.

    ``estimator``:

    * ``'gradient'`` (default, historical) -- ``dE/dx`` from
      :func:`numpy.gradient`.  On a pure carrier the central difference reads
      the per-pixel phase step ``h = k*dx*r/|R|`` as ``sin(h)`` rather than
      ``h``, so ``1/R`` comes out LOW by ``sin(h)/h`` -- a GRID-PITCH-dependent
      bias (measured table in :func:`carrier_referenced_fit_radius`).
    * ``'increment'`` -- the wrapping-safe nearest-neighbour phase increment
      ``angle(E[i+1] conj(E[i]))`` taken at the sample MIDPOINTS (the estimator
      ``carrier='auto'`` and the element-side collimation guards already use).
      For a parabolic carrier the midpoint increment is ``k*x_mid/R`` EXACTLY,
      so the fit is dx-INDEPENDENT for every ``|h| < pi``.
    """
    # Host-side least-squares curvature fit -> Python float; CuPy-safe pull
    # (see ``_envelope_amp_radius``).
    from ..backend import to_numpy
    E = to_numpy(E)
    Ny, Nx = E.shape[-2], E.shape[-1]
    k = 2.0 * np.pi / wavelength
    x = (np.arange(Nx, dtype=np.float64) - Nx / 2) * dx
    y = (np.arange(Ny, dtype=np.float64) - Ny / 2) * dy
    Y, X = np.meshgrid(y, x, indexing='ij')
    if estimator == 'increment':
        # Midpoint coordinates + midpoint |A| weights, per axis.  ``num`` sums
        # exactly the axes ``den`` does, so the isotropic form reduces to 1/R
        # on a parabola (each axis contributes ``(k/R) * sum w x_mid^2``).
        num = 0.0
        den = 0.0
        if axis is None or axis == 1:
            dphi = np.angle(E[:, 1:] * np.conj(E[:, :-1]))
            wgt = np.abs(E[:, 1:]) * np.abs(E[:, :-1])
            xm = 0.5 * (X[:, 1:] + X[:, :-1])
            num += float(np.sum(wgt * xm * (dphi / dx)))
            den += k * float(np.sum(wgt * xm * xm))
        if axis is None or axis == 0:
            dphi = np.angle(E[1:, :] * np.conj(E[:-1, :]))
            wgt = np.abs(E[1:, :]) * np.abs(E[:-1, :])
            ym = 0.5 * (Y[1:, :] + Y[:-1, :])
            num += float(np.sum(wgt * ym * (dphi / dy)))
            den += k * float(np.sum(wgt * ym * ym))
    elif estimator == 'gradient':
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
    else:
        raise ValueError(
            "_fit_carrier_inv: estimator must be 'gradient' or 'increment', "
            f"got {estimator!r}.")
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
    *,
    estimator: str = 'gradient',
    on_aliased: str = 'warn',
) -> Union[float, Tuple[float, float]]:
    """Best-fit PARABOLIC (paraxial) carrier radius ``R`` of a field's wavefront.

    Returns the intensity-weighted mean radius (``+inf`` for a collimated
    wavefront) using the phase-unwrap-free estimator in
    :func:`_fit_carrier_inv`.  With ``astigmatic=True`` returns the per-axis
    pair ``(R_x, R_y)``.  Use it to reference a measured / apertured field to
    its actual wavefront so the envelope is flat and the co-moving grid is
    well conditioned.

    Convention (audit AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24 §6.2 / §8.5).
    The fitted quantity is the coefficient of the paraxial PARABOLA
    ``phi = k r^2/(2R)``, matching :func:`carrier_referenced_reconstruct` /
    :func:`carrier_referenced_envelope`.  A field whose wavefront is the
    physical EXACT SPHERE ``S(R) = sign(R)(sqrt(r^2+R^2)-|R|)`` -- what an
    on-axis point source radiates and what ``apply_real_lens_traced``'s
    ``carrier=`` consumes -- is NOT a parabola, so the best parabolic fit reads
    FLATTER than the true sphere: for a Gaussian of 1/e amplitude radius ``w``,
    ``R_fit/R ~ 1 + w^2/(2 R^2) = 1 + NA^2/2`` (measured 1.00497 at NA 0.10 vs
    1.005 analytic, dx-INVARIANT -- see the last column below).  Do not treat
    the returned value as an exact-sphere radius at a non-trivial NA; convert
    explicitly (:func:`_sphere_parab_conversion`) if the two must agree.

    GRID-PITCH DEPENDENCE of the default estimator (measured 2026-07-25 on a
    FIXED physical field: Gaussian ``w = 200 um``, ``R = -2 mm`` (NA 0.10),
    fixed 1.6 mm window, ``lambda = 1.31 um``; only the pitch is swept).
    ``h = k*dx*w/|R|`` is the per-pixel carrier phase step at the beam edge;
    ``parab`` / ``sph`` are parabola- and exact-sphere-phased inputs:

    ====== ======== ========= ============ ============ ========== =========
    dx um  h        sin(h)/h  parab:grad   parab:incr   sph:grad   sph:incr
    ====== ======== ========= ============ ============ ========== =========
    0.391  0.187    0.9942    1.00440      1.00000      1.00931    1.00497
    0.781  0.375    0.9768    1.01771      1.00000      1.02247    1.00497
    1.562  0.749    0.9090    1.07276      1.00000      1.07682    1.00497
    3.125  1.499    0.6655    1.32438      1.00104      1.32507    1.00566
    6.250  2.998    0.0478    3.07642      1.53749      3.05100    1.54927
    12.50  5.995   -0.0473   89.57         -37.12       92.14      -33.04
    ====== ======== ========= ============ ============ ========== =========

    i.e. the ``'gradient'`` estimator reads ``sin(h)`` where the physics has
    ``h`` (a pure GRID artefact: the same physical field reads +0.4% at
    dx = 0.39 um and +32% at dx = 3.1 um), and past ``h = pi`` BOTH estimators
    alias and the answer is meaningless.  ``'increment'`` is exact to 1e-5 for
    the parabola while ``h < ~1``, leaving only the dx-invariant +NA^2/2
    convention offset on a sphere -- so it separates the two error classes
    instead of mixing them.  This is the ``carrier_referenced_*`` sibling of
    the F-B "entrance-tilt finite differencing vs the per-pixel carrier phase
    step" suspect (audit AUDIT_TRACED_CHAIN_DX_SCALING_2026_07_22).

    Parameters
    ----------
    E_full : ndarray, complex, shape (Ny, Nx)
        Full physical field (carrier NOT divided out).
    wavelength, dx : float
    dy : float, optional
        Defaults to ``dx``.
    astigmatic : bool, default False
        If True, fit ``R_x`` and ``R_y`` independently.
    estimator : {'gradient', 'increment'}, default 'gradient'
        ``'gradient'`` is the historical central-difference estimator (kept as
        the default so existing results are unchanged), with the ``sin(h)/h``
        bias tabulated above.  ``'increment'`` uses the wrapping-safe
        nearest-neighbour phase increment at the sample MIDPOINTS, where the
        parabola's increment is ``k*x_mid/R`` exactly -- dx-independent to 1e-5
        while the beam's step stays under ~1 rad/px, vs the 0.4%-32% drift of
        ``'gradient'`` over the same pitches.
    on_aliased : {'warn', 'silent'}, default 'warn'
        Emit a ``RuntimeWarning`` when more than
        ``_FIT_CARRIER_ALIAS_FRAC`` (5%) of the BRIGHT support reads a
        per-pixel phase step at/above ``_FIT_CARRIER_NYQUIST_FRAC*pi``
        (i.e. the grid cannot represent the beam's own carrier tilt, so
        NEITHER estimator can recover ``R``).  Silence it once acknowledged.
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_full, 'carrier_referenced_fit_radius',
                           input_kind='field')
    if dy is None:
        dy = dx
    if estimator not in ('gradient', 'increment'):
        raise ValueError(
            "carrier_referenced_fit_radius: estimator must be 'gradient' or "
            f"'increment', got {estimator!r}.")
    if on_aliased not in ('warn', 'silent'):
        raise ValueError(
            "carrier_referenced_fit_radius: on_aliased must be 'warn' or "
            f"'silent', got {on_aliased!r}.")
    if on_aliased == 'warn':
        _af = _carrier_fit_alias_fraction(E_full)
        if _af > _FIT_CARRIER_ALIAS_FRAC:
            import warnings
            warnings.warn(
                f"carrier_referenced_fit_radius: {100.0 * _af:.1f}% of the "
                f"bright support reads a per-pixel phase step at/above "
                f"{_FIT_CARRIER_NYQUIST_FRAC:g}*pi rad/px on this grid "
                f"(dx={dx * 1e6:.4f} um), so the field's own carrier tilt is "
                f"at or beyond the grid Nyquist and the fitted R is NOT "
                f"recoverable from it -- the default 'gradient' estimator "
                f"reads sin(h) for h and past h=pi aliases outright (measured: "
                f"R_fit/R = 92 at h ~ 6 rad/px).  Refine dx, or reference the "
                f"field to a known carrier first (carrier_referenced_envelope) "
                f"and fit only the residual.  Pass on_aliased='silent' to "
                f"acknowledge.",
                RuntimeWarning, stacklevel=2)
    if astigmatic:
        ix = _fit_carrier_inv(E_full, wavelength, dx, dy, axis=1,
                              estimator=estimator)
        iy = _fit_carrier_inv(E_full, wavelength, dx, dy, axis=0,
                              estimator=estimator)
        return (np.inf if ix == 0.0 else 1.0 / ix,
                np.inf if iy == 0.0 else 1.0 / iy)
    inv = _fit_carrier_inv(E_full, wavelength, dx, dy, axis=None,
                           estimator=estimator)
    return np.inf if inv == 0.0 else 1.0 / inv


def _rereference(env, R_old, R_new, wavelength, dx, dy,
                 bld=np, xp=np, is_jax=False):
    """Move an envelope from carrier ``R_old=(Rx,Ry)`` to ``R_new=(Rx,Ry)``:
    ``env * exp(i*k*r^2/2 * (1/R_old - 1/R_new))`` per axis.  The phase screen
    is built on ``bld`` (host NumPy for JAX) and moved on-device; NumPy
    defaults reproduce the historical screen byte-for-byte.

    CONVENTION: both radii are PARABOLIC (paraxial) carriers, matching
    :func:`carrier_referenced_reconstruct`/:func:`_radial_carrier_phase`.  The
    difference of two parabolas is exact for the re-reference, but an envelope
    stored against the EXACT SPHERE (a chain running
    ``carrier_reference='sphere'``) must be converted back to the parabola
    first (:func:`_sphere_parab_conversion` with ``sign=+1``) or the
    ``(S - r^2/2R)`` term is silently reinterpreted as envelope content -- the
    per-hand-off convention mismatch audit
    AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24 §6.2 quantifies at
    ``k r^4/(8 R^3)`` (several radians at a modest NA).  Collimated radii
    (``+/-inf``) contribute exactly 0 via ``_inv``, so a collimated
    re-reference is a no-op rather than a NaN."""
    Rox, Roy = R_old
    Rnx, Rny = R_new
    Ny, Nx = env.shape[-2], env.shape[-1]
    k = 2.0 * np.pi / wavelength
    x = (bld.arange(Nx, dtype=np.float64) - Nx / 2) * dx
    y = (bld.arange(Ny, dtype=np.float64) - Ny / 2) * dy
    Y, X = bld.meshgrid(y, x, indexing='ij')

    def _inv(R):
        return 0.0 if np.isinf(R) else 1.0 / float(R)

    dphi = 0.5 * k * ((X * X) * (_inv(Rox) - _inv(Rnx))
                      + (Y * Y) * (_inv(Roy) - _inv(Rny)))
    return env * _to_dev(bld.exp(1j * dphi), xp, is_jax)


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
    refit_estimator: str = 'gradient',
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
    refit_estimator : {'gradient', 'increment'}, default 'gradient'
        Estimator used by ``refit_carrier=True`` (see
        :func:`carrier_referenced_fit_radius`).  The default is the historical
        central-difference reading, whose ``sin(h)/h`` bias makes the refitted
        ``R`` GRID-PITCH dependent (measured +0.4% at 0.4 um pitch to +32% at
        3.1 um on one fixed NA-0.10 field); ``'increment'`` is dx-independent.
        Ignored unless ``refit_carrier=True``.

    Returns
    -------
    CarrierReferencedField
        ``(env, R, dx)`` with the apertured envelope; ``R`` / ``dx`` are
        2-tuples when the (returned) carrier is astigmatic.  If
        ``return_transmission`` is True, a ``(field, fraction)`` pair.

    Notes
    -----
    CONVENTION: ``R_carrier`` / ``new_carrier`` are PARABOLIC (paraxial) radii
    -- the same convention as :func:`carrier_referenced_reconstruct` and
    :func:`_rereference`, NOT the exact sphere ``apply_real_lens_traced``'s
    ``carrier=`` consumes.  An envelope stored against the exact sphere (a
    chain run with ``carrier_reference='sphere'``) must be converted back
    before it is apertured with ``refit_carrier`` / ``new_carrier``; the hard
    mask itself is convention-free (it touches only ``|env|``).
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_env, 'carrier_referenced_aperture',
                           input_kind='field')
    if dy is None:
        dy = dx
    if refit_carrier and new_carrier is not None:
        raise ValueError(
            "carrier_referenced_aperture: pass at most one of refit_carrier "
            "and new_carrier.")
    if refit_estimator not in ('gradient', 'increment'):
        raise ValueError(
            "carrier_referenced_aperture: refit_estimator must be 'gradient' "
            f"or 'increment', got {refit_estimator!r}.")
    xp, is_jax, bld = _backend_of(E_env)
    env = E_env
    Ny, Nx = env.shape[-2], env.shape[-1]

    # -- build the (real) aperture mask -------------------------------------
    # Built on ``bld`` (host f64 for JAX) and moved on-device with the field;
    # ``bld is np`` reproduces the historical mask byte-for-byte.
    if mask is not None:
        msk = np.asarray(mask)
        if msk.shape != tuple(env.shape):
            raise ValueError(
                f"carrier_referenced_aperture: mask shape {msk.shape} != "
                f"field shape {tuple(env.shape)}.")
        # Move the (host) mask onto the field's device on ANY backend.  The
        # mask is built host-side (``np.asarray(mask)``), NOT on ``bld``, so
        # ``_to_dev`` -- which only hops for JAX -- would leave it host-NumPy on
        # the CuPy backend; ``env * msk`` below would then mix a CuPy device
        # array with a host array and raise.  ``xp.asarray`` is the correct
        # backend-agnostic host->device move (NumPy no-op / byte-identical;
        # CuPy + JAX move on-device).
        msk = xp.asarray(msk.astype(np.float64))
    elif radius is not None or half_x is not None or half_y is not None:
        x = (bld.arange(Nx, dtype=np.float64) - Nx / 2) * dx
        y = (bld.arange(Ny, dtype=np.float64) - Ny / 2) * dy
        Y, X = bld.meshgrid(y, x, indexing='ij')
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
            msk = ((bld.abs(X) <= hx) & (bld.abs(Y) <= hy)).astype(np.float64)
        msk = _to_dev(msk, xp, is_jax)
    else:
        raise ValueError(
            "carrier_referenced_aperture: specify an aperture via radius=, "
            "half_x= / half_y=, or mask=.")

    # -- energy accounting (fraction; the dx*dy area element cancels) -------
    p_before = float((xp.abs(env) ** 2).sum())
    env_ap = env * msk
    p_after = float((xp.abs(env_ap) ** 2).sum())
    transmission = (p_after / p_before) if p_before > 0.0 else 0.0
    # NO renormalisation: the clipped power is genuinely gone.

    R_x, R_y, in_astig = _parse_carrier(R_carrier, 'carrier_referenced_aperture')

    # -- carrier: keep / re-fit / re-reference ------------------------------
    if new_carrier is not None:
        Nx_new, Ny_new, out_astig = _parse_carrier(
            new_carrier, 'carrier_referenced_aperture')
        env_out = _rereference(env_ap, (R_x, R_y), (Nx_new, Ny_new),
                               wavelength, dx, dy, bld, xp, is_jax)
        R_out = (Nx_new, Ny_new) if out_astig else Nx_new
    elif refit_carrier:
        if in_astig:
            ivx = _fit_carrier_inv(env_ap, wavelength, dx, dy, axis=1,
                                   estimator=refit_estimator)
            ivy = _fit_carrier_inv(env_ap, wavelength, dx, dy, axis=0,
                                   estimator=refit_estimator)
            inx = (0.0 if np.isinf(R_x) else 1.0 / R_x) + ivx
            iny = (0.0 if np.isinf(R_y) else 1.0 / R_y) + ivy
            Rnx = np.inf if inx == 0.0 else 1.0 / inx
            Rny = np.inf if iny == 0.0 else 1.0 / iny
            env_out = _rereference(env_ap, (R_x, R_y), (Rnx, Rny),
                                   wavelength, dx, dy, bld, xp, is_jax)
            R_out = (Rnx, Rny) if (Rnx != Rny) else Rnx
            out_astig = (Rnx != Rny)
        else:
            iv = _fit_carrier_inv(env_ap, wavelength, dx, dy, axis=None,
                                  estimator=refit_estimator)
            inv_new = (0.0 if np.isinf(R_x) else 1.0 / R_x) + iv
            Rn = np.inf if inv_new == 0.0 else 1.0 / inv_new
            env_out = _rereference(env_ap, (R_x, R_x), (Rn, Rn),
                                   wavelength, dx, dy, bld, xp, is_jax)
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


# ===========================================================================
# Near-focus landing (audit F4.2): stop-short + fine Bluestein zoom
# ===========================================================================
# Landing a carrier-referenced leg AT (or within a few um of) the carrier's
# geometric focus collapses the co-moving grid: the magnification m = R_out/R
# -> 0, so the window ``N * dx * |R_out/R|`` shrinks below the diffraction-
# limited waist AND clips the focused beam's halo, producing spurious sub-
# wavelength "spots" (window-truncation ringing masquerading as speckle
# peaks -- the exact artefact the design-121 carrier chain hit at the MSoP).
# The working pattern (validated in
# ``validation/repro_traced_carrier_121/carrier_chain_121.py``) is to stop the
# carrier leg a small ``standoff`` SHORT of the target -- where the co-moving
# grid still holds the whole beam and resolves its fringe -- reconstruct the
# full field there, and finish with a fine band-limited angular-spectrum
# Bluestein zoom (:func:`angular_spectrum_propagate_mft`) onto a user-chosen
# output grid that is BOTH fine enough to resolve the focus AND wide enough to
# hold the halo.  :func:`carrier_referenced_focus_readout` packages exactly
# that.


def carrier_referenced_focus_readout(
    env: np.ndarray,
    R_carrier: float,
    z: float,
    wavelength: float,
    dx: float,
    *,
    dx_out: float,
    N_out: int,
    standoff: Optional[float] = None,
    centre_out: Tuple[float, float] = (0.0, 0.0),
    bandlimit: bool = True,
    _period_out: Optional[dict] = None,
) -> np.ndarray:
    """Read a carrier-referenced beam at a target plane NEAR its focus without
    the co-moving-grid collapse (audit F4.2).

    Landing a carrier leg at / within a few um of the carrier's geometric
    focus shrinks the co-moving grid below the diffraction-limited waist and
    clips the beam's halo, producing spurious sub-wavelength speckle "spots".
    This helper packages the working pattern: carrier-step to a plane a small
    ``standoff`` SHORT of the target (where the co-moving grid still holds the
    whole beam and resolves its fringe), reconstruct the full field there, and
    finish with a fine band-limited angular-spectrum **Bluestein zoom**
    (:func:`angular_spectrum_propagate_mft`) onto the user-chosen
    ``(dx_out, N_out)`` grid -- fine enough to resolve the focus and wide
    enough to hold the halo, the combination a single fixed grid cannot give.

    Parameters
    ----------
    env : ndarray, complex, shape (Ny, Nx)
        Beam ENVELOPE on the co-moving grid (the carrier phase divided out).
    R_carrier : float
        Signed carrier radius at the input plane (m); ``R < 0`` converging
        toward a focus a distance ``-R`` ahead.  ``+/-inf`` (collimated) is
        allowed -- there is no geometric focus to collapse, so the step is an
        ordinary carrier leg plus the Bluestein readout.
    z : float
        Distance from the input plane to the TARGET (readout) plane (m).
    wavelength, dx : float
        Wavelength and input grid pitch (m).
    dx_out : float
        Output grid pitch (m) -- pick it fine enough to resolve the focused
        spot (``<~ lambda / (few * NA)``).
    N_out : int
        Output grid size (square).
    standoff : float, optional
        Length of the final fine Bluestein-zoom leg (m): the carrier leg
        covers ``z - standoff`` and stops that far SHORT of the target.  Must
        be ``> 0``.  Defaults to ``_BRIDGE_ZR_FACTOR`` Rayleigh ranges of the
        estimated focus (plus any distance the target sits past the focus),
        clamped so the carrier leg ``z - standoff`` does not back before the
        input plane.
    centre_out : (float, float), optional
        Physical ``(x, y)`` centre of the output grid (m).  Default on-axis.
    bandlimit : bool, default True
        Band-limit the ASM transfer function (Matsushima-Shimobaba).

    Returns
    -------
    E_out : ndarray, complex, shape (N_out, N_out)
        The full physical field at the target plane on the centred
        ``(dx_out)`` grid -- carries the absolute physical phase, same
        convention as :func:`angular_spectrum_propagate_mft`.

    Other Parameters
    ----------------
    _period_out : dict, optional
        PRIVATE.  When a dict is supplied, its ``'period'`` key is filled with
        the ``(period_x, period_y)`` spatial period (m) of the final Bluestein
        zoom -- ``N * dx`` of the CO-MOVING grid at the stop plane
        (:func:`~lumenairy.propagators.mft._asm_mft_spatial_period`).  Near a
        focus that grid has collapsed, so the period can be far smaller than
        the input window; a requested ``N_out * dx_out`` wider than it returns
        periodic REPLICAS rather than signal.
        :func:`propagate_traced_carrier_chain` reports the value as
        ``readout_period`` in its stages and
        :func:`propagate_traced_carrier_chain_multi` refuses a per-congruence
        window that exceeds it (niche D2).  Not part of the public contract.

    Notes
    -----
    The carrier leg is a fast (byte-identical) Sziklas-Siegman step because
    the stop plane is chosen to PRECEDE the focus -- the near-focus bridge is
    deliberately NOT relied upon (its output would land back on the collapsed
    co-moving grid).  Validated against the analytic Gaussian focus and a
    resolved fixed-grid reference in ``test_niche_r8_tiltaware_chain_api.py``.

    CONVENTION (audit AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24 §6.2/§8.5).
    This is the PARAXIAL path end to end: ``env`` must be referenced to the
    paraxial PARABOLA ``exp(i k r^2/2R)`` (the
    :func:`carrier_referenced_reconstruct` convention), because
    :func:`propagate_carrier_referenced` transports it and reconstructs it with
    that same parabola.  An envelope stored against the EXACT SPHERE -- what a
    chain running ``carrier_reference='sphere'`` holds between hand-offs, and
    what :func:`apply_real_lens_traced`'s ``carrier=`` consumes -- must be
    converted back first (:func:`_sphere_parab_conversion`, ``sign=+1``);
    :func:`propagate_traced_carrier_chain` does exactly that before calling
    this.  Handing a sphere-referenced envelope straight in silently
    reinterprets ``k(S - r^2/2R) ~ -k r^4/(8 R^3)`` as beam content.  For a
    genuinely high-NA leg use :func:`carrier_referenced_exact_focus_readout`,
    which references the exact sphere itself.
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(env, 'carrier_referenced_focus_readout',
                           input_kind='field')
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError(
            "carrier_referenced_focus_readout: wavelength must be finite and "
            f"positive, got {wavelength!r}.")
    if not (np.isfinite(dx) and dx > 0):
        raise ValueError(
            "carrier_referenced_focus_readout: dx must be finite and "
            f"positive, got {dx!r}.")
    if not (np.isfinite(dx_out) and dx_out > 0):
        raise ValueError(
            "carrier_referenced_focus_readout: dx_out must be finite and "
            f"positive, got {dx_out!r}.")
    if int(N_out) <= 0:
        raise ValueError(
            f"carrier_referenced_focus_readout: N_out must be > 0, got {N_out!r}.")
    if not np.isfinite(z):
        raise ValueError(
            f"carrier_referenced_focus_readout: z must be finite, got {z!r}.")

    R = float(R_carrier)
    if standoff is None:
        standoff = _default_focus_standoff(env, R, z, wavelength, dx)
    standoff = float(standoff)
    if not (standoff > 0.0):
        raise ValueError(
            "carrier_referenced_focus_readout: standoff must be > 0 (it is the "
            f"length of the fine Bluestein-zoom leg), got {standoff!r}.")
    # Keep the carrier leg (z - standoff) from backing before the input plane
    # (and from overshooting to the far side of a very close target): clamp the
    # Bluestein leg to |z|.  A leg equal to |z| reconstructs at the input plane
    # and Bluestein-zooms the whole distance (still correct; just less of the
    # compact-field win).
    if standoff > abs(z) and z != 0.0:
        standoff = abs(z)
    z_stop = z - np.copysign(standoff, z) if z != 0.0 else -standoff

    cr = propagate_carrier_referenced(env, R, z_stop, wavelength, dx)
    env_s, R_s, dx_s = cr.env, cr.R, cr.dx
    if isinstance(dx_s, tuple):
        dx_s = dx_s[0]
    if isinstance(R_s, tuple):
        R_s = R_s[0]
    E_stop = carrier_referenced_reconstruct(env_s, R_s, wavelength, dx_s)

    from .mft import _asm_mft_spatial_period, angular_spectrum_propagate_mft
    if _period_out is not None:
        _sh = np.shape(E_stop)
        _period_out['period'] = _asm_mft_spatial_period(
            _sh[-1], dx_s, _sh[-2], dx_s)
    return angular_spectrum_propagate_mft(
        E_stop, z - z_stop, wavelength, dx_s, dx_out, int(N_out),
        centre_out=centre_out, bandlimit=bandlimit)


def _default_focus_standoff(env, R, z, wavelength, dx):
    """Default fine-zoom leg length for :func:`carrier_referenced_focus_readout`:
    ``_BRIDGE_ZR_FACTOR`` Rayleigh ranges of the estimated focus, plus any
    distance the target sits PAST the focus, so the carrier leg stops safely
    before the co-moving-grid collapse.  Falls back to half the target
    distance when there is no geometric focus ahead (collimated / diverging)."""
    z_focus = np.inf if not np.isfinite(R) else -R
    w_env = _envelope_amp_radius(env, dx, dx)
    if np.isfinite(z_focus) and z_focus > 0.0 and w_env > 0.0 and abs(R) > 0.0:
        w0 = wavelength * abs(R) / (np.pi * w_env)      # estimated focus waist
        zR = np.pi * w0 * w0 / wavelength
        margin = _BRIDGE_ZR_FACTOR * zR
        # Stop `margin` before the focus; if the target is PAST the focus, the
        # zoom leg additionally spans that overshoot.
        return margin + max(0.0, abs(z) - z_focus)
    # No focus ahead: split the leg (the carrier step cannot collapse).
    return 0.5 * abs(z) if z != 0.0 else 0.0


# ===========================================================================
# Exact (non-paraxial) high-NA final leg (audit R9 / F2 end-to-end)
# ===========================================================================
# The paraxial carrier (Sziklas-Siegman) references the beam's wavefront to a
# quadratic PARABOLA ``exp(i k r^2/2R)``.  On a strongly-converging FINAL leg
# (NA -> 0.5) the true converging wavefront is the EXACT sphere
# ``S(R) = sign(R)(sqrt(r^2+R^2)-|R|)``, and ``parabola - sphere`` reaches
# HUNDREDS of radians of r^4 at the beam edge (~200 rad on the design-121
# f/1.1 image leg).  Re-enveloping such a leg with the paraxial ABCD ``R_out``
# dumps that r^4 onto the paraxial envelope, which paraxial carrier propagation
# CANNOT focus (R9: a carrier through-focus of an NA-0.46 sphere reaches only a
# few % encircled energy).  The fix is an EXACT diffraction step: reference the
# field to its own EXACT sphere (leaving a genuinely smooth envelope -- the
# aliasing of the two steep phases cancels pointwise on the shared grid),
# resample that smooth envelope onto a grid that Nyquist-samples the exact
# sphere (``dx <= lambda/(2 NA)``), reconstruct, and propagate to the image
# with the EXISTING exact band-limited angular-spectrum Bluestein zoom
# (:func:`angular_spectrum_propagate_mft`) -- no paraxial magnification /
# curvature on this leg.  Validated (R9) to focus a clean NA-0.3-0.46 exact
# sphere to the diffraction limit (Strehl ~1) where the paraxial carrier path
# gives ~5%.


def _exact_sphere_eikonal(shape, dx, dy, wavelength, R, centre=(0.0, 0.0)):
    """Exact on-axis point-source sphere eikonal ``S(R) = sign(R)(sqrt(r^2 +
    R^2) - |R|)`` in METRES on the centred grid (host NumPy float64).  This is
    the EXACT converging/diverging wavefront -- the paraxial parabola
    ``r^2/2R`` is its small-``r/R`` truncation, dropping ``-r^4/8R^3`` which is
    huge at high NA.  Same sign convention as ``apply_real_lens_traced``'s
    carrier ``W`` (R7/F2) and :func:`carrier_referenced_reconstruct`.

    A COLLIMATED carrier (``R = +/-inf``) returns the plane-wave eikonal
    ``S == 0`` -- the analytic ``|R| -> inf`` limit.  Evaluating the closed
    form there would give ``inf - inf = NaN`` over the whole grid, and an
    all-NaN eikonal silently disables every downstream guard that reduces it
    (audit AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24 §4 / this sibling
    sweep: the element-side twin of this expression made a collimated
    ``carrier=inf`` skip the R7 fit restriction and emit an "All-NaN slice
    encountered" RuntimeWarning).  ``R == 0`` (the carrier's own focus) is
    undefined and also returns zeros; callers guard it explicitly.

    ``centre`` (niche D1) DECENTRES the sphere to ``(x0, y0)``; the default
    ``(0, 0)`` is short-circuited so the on-axis eikonal is untouched."""
    Ny, Nx = int(shape[-2]), int(shape[-1])
    if not np.isfinite(R) or R == 0.0:
        return np.zeros((Ny, Nx), dtype=np.float64)
    x = (np.arange(Nx, dtype=np.float64) - Nx / 2) * dx
    y = (np.arange(Ny, dtype=np.float64) - Ny / 2) * dy
    if centre != (0.0, 0.0):
        x = x - float(centre[0])
        y = y - float(centre[1])
    Y, X = np.meshgrid(y, x, indexing='ij')
    r2 = X * X + Y * Y
    sgn = 1.0 if R > 0 else -1.0
    return sgn * (np.sqrt(r2 + R * R) - abs(R))


def _sphere_parab_conversion(shape, dx, wavelength, R, sign, w_beam=None,
                             centre=(0.0, 0.0)):
    """BAND-LIMITED parabola <-> exact-sphere carrier-convention conversion
    factor ``exp(sign*i*k*(S(R) - r^2/(2R)) * T(r))`` on the centred grid, or
    ``None`` for a collimated/degenerate carrier (nothing to convert).

    The carrier-referenced machinery references the PARAXIAL PARABOLA
    ``r^2/(2R)`` (:func:`_radial_carrier_phase`), while a traced element's ray
    launch / carrier eikonal references the EXACT sphere
    ``S(R) = sign(R)(sqrt(r^2+R^2) - |R|)`` (:func:`_exact_sphere_eikonal`).
    The two differ by ``+k r^4/(8 R^3) + O(r^6)``, which is a few RADIANS at a
    modest NA over a long carrier leg (design-121: +3.4 rad at r=w at the first
    group vertex, emitter NA 0.104 over 45.9 mm) -- i.e. the paraxial carrier
    leg leaves a spurious SPHERICAL-ABERRATION term relative to the physical
    (spherical) diverging wave.  Multiplying by this factor converts a
    parabola-referenced reconstruction into the exact-sphere-referenced field
    the element consumes (``sign=+1``) and back (``sign=-1``).

    ``T(r)`` is a ``cos^2`` roll-off from ``0.75*r_safe`` to
    ``r_safe = (|R|^3 * lambda / dx)^(1/3)`` -- the radius beyond which the
    DIFFERENCE term itself exceeds the grid's Nyquist slope, so a whole-grid
    swap would scatter aliased guard-band junk into the beam (measured: the
    tapered and whole-grid conversions agree to 4 digits on the design-121
    stages, i.e. the guard band truly carries nothing, while the untapered
    swap breaks a coarse chain).  ``w_beam`` (optional) enables a warning when
    the taper reaches into the beam (``r_safe < 2*w_beam``), where the
    representation would be mixed exactly where the amplitude matters.

    **The taper's mixed-convention skirt is a MEASURED NULL on design-121
    (S12), not a residual error budget.**  Audit
    AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24 S8.6 attributed "the residual 9 %
    of Strehl beyond r > 1.5 w" partly to this skirt; direct measurement
    refutes that.  Scaling ``r_safe`` by 1.5 and by INFINITY (T == 1, i.e. the
    whole-grid swap with no taper at all) reproduces the shipping design-121
    result to the digit -- at-plane 3.650 um / 87.3 / 99.3 and best focus
    3.550 um / EE3 89.57 / EE6 99.26 in all three runs.  Two reasons: (i) the
    conversion and its inverse are POINTWISE, so ``env = E*exp(-ikS)`` is exact
    at every grid point no matter how the phase slope compares with Nyquist --
    only FFT-based steps that see the RESULT care, and a wider taper makes the
    stored envelope smoother, not rougher; (ii) geometrically the taper barely
    reaches the beam anyway -- on design-121 the taper onset ``0.75*r_safe``
    sits at 2.73 w (first entrance), 3.60 w (S21-S22 exit) and 2.07 w (S23-S24
    exit, the worst), and ``r_safe`` exceeds the whole grid on the fine
    retrace leg (full conversion, no taper), so at most ~2e-4 of the power ever
    sees a mixed convention.  The S8.6 skirt was really the
    ``preserve_input_phase='remap'`` ray-lattice alias (see
    ``apply_real_lens_traced``'s ``remap_sampling``).  The ``r_safe < 2*w``
    warning is retained as a validity flag for OTHER geometries (a much higher
    carrier NA or coarser dx can push the onset inside the beam), not as a
    known design-121 defect.
    """
    if not np.isfinite(R) or R == 0.0:
        return None
    n = int(shape[-1])
    ny = int(shape[-2])
    x = (np.arange(n, dtype=np.float64) - n / 2) * dx
    y = (np.arange(ny, dtype=np.float64) - ny / 2) * dx
    if centre != (0.0, 0.0):
        # niche D1: a tilted congruence's sphere is centred on its CHIEF RAY,
        # so both the (S - parabola) difference and the band-limit radius must
        # be measured from there.  Short-circuited on the on-axis default.
        x = x - float(centre[0])
        y = y - float(centre[1])
    r2 = x[None, :] ** 2 + y[:, None] ** 2
    k = 2.0 * np.pi / wavelength
    diff = _exact_sphere_eikonal((ny, n), dx, dx, wavelength, R,
                                 centre=centre) - r2 / (2.0 * R)
    r_safe = (abs(R) ** 3 * wavelength / dx) ** (1.0 / 3.0)
    if w_beam is not None and w_beam > 0.0 and r_safe < 2.0 * w_beam:
        import warnings
        warnings.warn(
            f"_sphere_parab_conversion: the band-limit radius r_safe="
            f"{r_safe * 1e3:.3f} mm (= (|R|^3 lambda/dx)^(1/3) at "
            f"R={R * 1e3:.3f} mm, dx={dx * 1e6:.3f} um) reaches inside "
            f"2x the beam radius (w={w_beam * 1e3:.3f} mm), so the "
            f"parabola->sphere conversion is tapered off where the beam "
            f"still carries power: the carrier convention is MIXED over the "
            f"beam skirt.  Refine dx (or lower the carrier NA) if the exit "
            f"wavefront matters at that radius.",
            RuntimeWarning, stacklevel=3)
    t = np.clip((np.sqrt(r2) - 0.75 * r_safe) / (0.25 * r_safe), 0.0, 1.0)
    return np.exp(sign * 1j * k * diff * np.cos(0.5 * np.pi * t) ** 2)


def _fourier_upsample_crop(env, n_crop, n_fine):
    """Band-limited Fourier RESAMPLE of a SMOOTH envelope onto a new pixel
    count spanning the SAME physical window: crop the centred ``n_crop``
    sub-window, then either zero-pad its spectrum (``n_fine > n_crop`` --
    exact band-limited upsample) or truncate its spectrum to the central
    ``n_fine`` block (``n_fine < n_crop`` -- exact band-limited downsample,
    valid because the envelope is smooth/band-limited by construction: the
    carrier has already been divided out, so the discarded high frequencies
    are ~0), and inverse-transform.  Far cheaper than cubic
    ``resample_field`` at the large ``n_fine`` a high-NA focus needs.

    Returns the envelope on the ``n_fine`` grid spanning the SAME physical
    window (``n_crop * dx``), i.e. pitch ``dx * n_crop / n_fine`` -- in
    EITHER direction: ``out.shape[-1] == n_fine`` always holds (audit
    AUDIT_TRACED_CHAIN_DX_SCALING_2026_07_22 F-A: the pre-fix downsample
    branch instead returned the raw ``n_crop``-sized crop, silently
    mismatching the pitch every downstream caller assumed).

    ``n_crop`` must fit inside the input (``<= env.shape[-1]``) and be at
    least 2.  A LARGER ``n_crop`` cannot be honoured -- the sub-window does
    not exist -- and the centred slice then silently degenerates: with
    ``n_crop = n + 2`` the negative start index wraps and the "crop" is a
    single pixel, which the spectral pad then broadcasts to a full
    ``n_fine`` array that passes the shape invariant while spanning
    ``1*dx`` instead of the ``n_crop*dx`` window the caller assumes
    (measured: garbage at rel-std 9.4).  That is the same silent-wrong-pitch
    failure mode as F-A itself, so it is rejected explicitly rather than
    left to an incidental ``AssertionError``/broadcast error.  Both library
    call sites clamp ``n_crop`` to the grid, so this is a contract guard,
    not a behaviour change.
    """
    env = np.asarray(env)
    n = env.shape[-1]
    n_crop = int(n_crop)
    n_fine = int(n_fine)
    if n_crop > n or n_crop < 2 or n_fine < 2:
        raise ValueError(
            f"_fourier_upsample_crop: n_crop must satisfy 2 <= n_crop <= "
            f"env.shape[-1] and n_fine >= 2 (got n_crop={n_crop}, "
            f"n_fine={n_fine}, env.shape[-1]={n}).  A crop larger than the "
            f"input has no defined window: the centred slice degenerates and "
            f"the result would span the wrong physical extent while still "
            f"having the requested shape.")
    c0 = n // 2 - n_crop // 2
    ec = np.ascontiguousarray(env[c0:c0 + n_crop, c0:c0 + n_crop])
    if n_fine == n_crop:
        out = ec
    else:
        F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(ec)))
        if n_fine > n_crop:
            # Zero-pad the spectrum (exact band-limited upsample).
            pad = np.zeros((n_fine, n_fine), dtype=np.complex128)
            o = n_fine // 2 - n_crop // 2
            pad[o:o + n_crop, o:o + n_crop] = F
        else:
            # Truncate the spectrum to its central n_fine block (exact
            # band-limited downsample -- k-space low-pass truncation).
            o = n_crop // 2 - n_fine // 2
            pad = F[o:o + n_fine, o:o + n_fine]
        out = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(pad)))
        # Same value-preserving scale in both directions: numpy's ifft2
        # normalises by 1/(output size)^2, so re-gridding from n_crop to
        # n_fine samples over the SAME window needs (n_fine/n_crop)^2 to
        # restore the point-sample values a matched-size round trip would
        # give (derivation: ifft2(fft2(ec)) == ec exactly at equal sizes).
        out = out * (float(n_fine) / float(n_crop)) ** 2
    assert out.shape[-1] == n_fine, (
        f"_fourier_upsample_crop: internal shape invariant broken "
        f"(got {out.shape[-1]}, expected n_fine={n_fine})")
    return out


def _crop_about_centre(env, dx, x0, y0, n_crop, where):
    """``n_crop``-square crop of a SMOOTH envelope centred on ``(x0, y0)``
    metres instead of the grid origin (niche D6).

    Done as an integer-pixel slice plus a sub-pixel band-limited shift OF THE
    CROP -- so the chief-ray offset is never quantised to the grid, and the
    only Fourier pair paid is on the (small) crop rather than on the whole
    input grid.  ``(0, 0)`` reproduces :func:`_fourier_upsample_crop`'s own
    centred slice exactly, byte for byte.

    Raises when the requested window does not fit inside the input grid at
    that centre -- the crop would otherwise wrap round (a plausible-looking
    wrong answer).

    NOTE (2026-07-29 adversarial verification): that raise is a DEFENSIVE
    invariant, not the guard a caller sees.  Its only shipped caller,
    :func:`carrier_referenced_exact_focus_readout`, bounds ``n_crop`` by what
    fits at ``(x0, y0)`` BEFORE calling here, so this branch is unreachable
    from it; the user-visible guard for the same failure is that function's
    ``on_readout_window``, which measures the power the bound actually
    truncates.  An earlier revision described THIS raise as the protection,
    which it was not -- the clamp was silent."""
    e = np.asarray(env)
    n = e.shape[-1]
    n_crop = int(n_crop)
    ix = int(round(float(x0) / dx))
    iy = int(round(float(y0) / dx))
    c0 = n // 2 - n_crop // 2
    i0, j0 = c0 + ix, c0 + iy
    if i0 < 0 or j0 < 0 or i0 + n_crop > n or j0 + n_crop > n:
        raise ValueError(
            f"{where}: a {n_crop * dx * 1e3:.4f} mm window centred on the "
            f"chief ray at ({float(x0) * 1e3:+.4f}, {float(y0) * 1e3:+.4f}) mm "
            f"does not fit inside the {n * dx * 1e3:.4f} mm grid (N={n} x "
            f"dx={dx * 1e6:.4f} um).  Raise N (or dx) for this congruence, or "
            f"shrink window_factor.")
    ec = np.ascontiguousarray(e[j0:j0 + n_crop, i0:i0 + n_crop])
    rx = float(x0) - ix * dx
    ry = float(y0) - iy * dx
    if rx != 0.0 or ry != 0.0:
        ec = _shift_envelope(ec, -rx, -ry, dx)
    return ec


# ---------------------------------------------------------------------------
# Guard-rail policy plumbing (niche D3 -- roadmap
# ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27 P3 + P5)
# ---------------------------------------------------------------------------
# Every guard in this module reports through ONE disposition helper so a batch
# production run can promote the whole class to fatal with the same vocabulary
# ('error' / 'warn' / 'ignore').  The shipped defaults never change behaviour:
# a guard that warns today still warns.
_GUARD_ACTIONS = ('error', 'warn', 'ignore')


def _guard_dispose(action, msg, exc=RuntimeError, stacklevel=3):
    """Apply an ``'error'`` / ``'warn'`` / ``'ignore'`` disposition to a
    detected fault.  ``'warn'`` emits a ``RuntimeWarning``; ``'error'`` raises
    ``exc``; ``'ignore'`` is silent.

    ``stacklevel`` is counted FROM THE CALLING GUARD SITE (this helper's own
    frame is added internally), so a guard that used to call
    ``warnings.warn(..., stacklevel=3)`` inline keeps pointing at the same
    frame when it is converted to ``_guard_dispose(..., stacklevel=3)``."""
    if action == 'error':
        raise exc(msg)
    if action == 'warn':
        import warnings
        warnings.warn(msg, RuntimeWarning, stacklevel=int(stacklevel) + 1)


def _check_guard_action(name, value, fn):
    """Validate one ``on_*`` policy string, raising a ValueError that names the
    knob and the accepted vocabulary."""
    if value not in _GUARD_ACTIONS:
        raise ValueError(
            f"{fn}: {name} must be 'error', 'warn' or 'ignore', got "
            f"{value!r}.")
    return value


# ---------------------------------------------------------------------------
# P3 -- multi-congruence detection at chain entry
# ---------------------------------------------------------------------------
# The failure this exists for is a PLAUSIBLE-LOOKING WRONG ANSWER.  At v5.28
# the design-121 32-order Dammann fan was pushed through
# :func:`propagate_traced_carrier_chain` MULTIPLEXED and produced a populated,
# credible-looking frame lattice whose per-frame power was scrambled
# (0.47 +/- 0.51 % against a design 2.78 %/frame, uniformity ~0.996).  Nothing
# raised and nothing warned, even though ``apply_real_lens_traced``'s
# entrance->exit map names exactly that case -- "comparable-power beams at
# well-separated angles (post-DOE at large split)" -- as EXCLUDED.
#
# TWO measurements are needed, and each is blind to what the other catches.
# Both are the SHIPPED detectors, reused as-is (no competing estimator):
#
#   A. ANGULAR SPREAD -- ``_carrier_residual_rms`` (``_lens_traced``), the
#      wrapping-safe nearest-neighbour phase-increment rms in RADIANS, and the
#      discriminator that DEFINES the documented ``_NONCOLLIMATED_RESID_THRESH
#      = 0.02 rad`` envelope.  It catches a single congruence carried on the
#      wrong reference (a bare +-46 mrad order handed in with ``r_in=inf``
#      reads 0.046 rad -- 2.3x outside the envelope) and an unreferenced
#      diverging input.
#   B. MULTI-VALUEDNESS -- ``fga._tilt_dispersion``, the spread of the LOCAL
#      wavevector about its per-region (amplitude-weighted, sigma ~2 px
#      Gaussian) mean, i.e. the measurement
#      :func:`~lumenairy.apply_real_lens_universal` already routes on.  It is
#      called here at ``na=1`` (so the return is a raw rms direction cosine, in
#      radians) and then put on a GRID-CANONICAL scale and compared against a
#      FIXED reference NA -- see the two subsections below, both of which are
#      corrections to defects measured in this gate's first cut.
#
# Measurement A alone is NOT sufficient, and this is the caveat the
# wavefront-aware audit recorded: a wrapped nearest-neighbour gradient
# UNDER-REPORTS when the content aliases or interferes.  Measured here on a
# 512^2 / dx = 1 um / lambda = 1.31 um grid with a w = 60 um Gaussian
# (see ``tests/unit/test_niche_d3_guards.py``):
#
#   input                                        A (rad)   B (rad, na=1)
#   clean Gaussian, flat phase                    0.0000       0.000000
#   Gaussian x smooth r^4 residual (8 rad p-v)    0.0000       0.000000
#   single tilted Gaussian, L = 46 mrad           0.0460       0.000000
#   TWO coincident equal beams, split 46 mrad     0.0000       0.010450
#   2x2 order fan, split 46 mrad                  0.0000       0.014915
#   8x4 fan, random order phases, +-46 mrad       0.0354       0.010360
#   diverging Gaussian R = 2 mm (aliased carrier) 0.0365       0.000094
#
# The 2x2 fan row is the whole point: A reads EXACTLY ZERO on a 46 mrad
# multi-congruence input (two equal beams interfere to a real cosine times a
# single mean-direction carrier, so every nearest-neighbour increment reports
# the MEAN tilt and the pi jumps live in the amplitude nulls), while B reads
# 0.0149 rad -- and conversely B reads exactly zero on the single tilted beam
# that A catches.  The gate therefore fires on EITHER.
#
# B.1 -- WHY THE RAW READING IS PUT ON A GRID-CANONICAL SCALE.
# ``_tilt_dispersion``'s raw value is NOT a property of the field alone: it
# falls as sqrt(dx), so the detector gets BLINDER exactly as the grid gets more
# accurate.  The mechanism is structural, not a tuning accident.  Two
# comparable beams crossing at +-theta superpose to
#
#     E = 2 A(r) cos(k0 theta x)
#
# -- a REAL cosine times ONE mean carrier.  The wrapped nearest-neighbour phase
# increment is therefore 0 inside every lobe and +-pi across every amplitude
# null, so ALL of the multi-valuedness sits in the vanishing-amplitude pixels
# at the nulls: the wrapped increment there rises as 1/dx but the amplitude
# weight a^2 that multiplies it falls as (dx/Lambda)^2, over Lambda/dx pixels
# per fringe of period Lambda = lambda/(2 theta).  The amplitude-weighted
# variance is then ~ pi^2 lambda^2 dx / (2 Lambda^3), i.e.
#
#     raw_rms ~ 2 pi theta sqrt(theta dx / lambda)          [rad]
#
# Measured on the design-121 8x4 order fan (+-46 / +-23 mrad, equal order
# phases, w = 300 um, FIXED 2.048 mm window -- only the pitch changes):
#   dx0 (um)   4       2       1       0.5     0.25    0.125
#   raw        2.97e-2 2.28e-2 1.66e-2 1.19e-2 8.36e-3 5.92e-3
#   ratio        --    0.767   0.727   0.717   0.703   0.708   (1/sqrt2 = 0.707)
# The pre-canonical gate was therefore SILENT on design 121's own 32-order fan
# at dx0 = 0.25 um / N = 8192 -- the exact production condition roadmap P4
# names as the original F-B evidence matrix's worst row -- while the multiplexed
# answer stays 36-86 % wrong by the linearity oracle at every pitch.  Detector A
# is blind there by symmetry (residual 1.5e-16 rad).
#
# Multiplying by ``sqrt(lambda / dx)`` cancels the law exactly and, by the
# expression above, leaves ``canon ~ 2 pi theta^1.5`` -- a function of the
# CROSSING ANGLE only, independent of both the grid and the wavelength:
#   canon      1.70e-2 1.84e-2 1.90e-2 1.92e-2 1.91e-2 1.92e-2
# flat to 1 % for dx0 <= 2 um, and only 11 % low at dx0 = 4 um where the
# +-46 mrad fringe spans ~7 px and the estimator starts to alias.  Wavelength
# invariance, measured on a 2-beam +-23 mrad pair at dx = 4 um with
# lambda = 0.5 / 1.31 / 3.0 / 10.6 um: raw spreads 3.6x over
# 2.78e-2 / 2.07e-2 / 1.40e-2 / 7.65e-3 (and the 10.6 um row falls BELOW the
# cutoff) while canon holds 9.81e-3 / 1.18e-2 / 1.21e-2 / 1.25e-2.  The
# measured prefactor is canon / (2 pi theta^1.5) = 0.42 / 0.49 / 0.55 / 0.56 /
# 0.56 / 0.54 at theta = 2 / 5 / 10 / 20 / 46 / 90 mrad, i.e. canon ~ 3.5
# theta^1.5 once the crossing is above ~10 mrad.
#
# B.2 -- WHY THE REFERENCE NA IS A CONSTANT AND NOT THE FIRST GROUP'S.
# ``fga._system_na`` is ``_default_p_max / 1.6`` with ``_default_p_max =
# min(0.6, max(0.05, 1.6 * na))`` (``fga.py``), so it is SATURATED at 0.375
# above and FLOORED at 0.03125 below: a 12x band whose BOTH ends are clamp
# artifacts, and which describes the FIRST LENS rather than the field handed
# in.  Dividing the field's own dispersion by it makes the verdict a function
# of the first lens's f-number, and it fails in both directions:
#   * upper end -- an 8x8 +-23 mrad fan scores 0.164 through a 121-class first
#     group (FIRES) but 0.037 through a saturated 0.375-NA one (SILENT).  Same
#     field; 56 % wrong by the linearity oracle either way.
#   * lower end -- a single Gaussian hard-clipped at 0.7 w and ASM-propagated
#     0.02 z_R (ONE beam, one aperture, one propagation; raw 3.24e-3 rad,
#     detector A silent at 1.40e-2 rad) scores 0.104 through a FLOORED
#     0.03125-NA first group and FIRES: a false positive telling the caller to
#     split a single congruence into DOE orders.  The same field is silent at
#     0.0846 and at 0.375.  Every relay slower than about f/10 lands on the
#     floor -- measured na_sys 0.0312 for 200/300/500 mm singlets.
# Both are one defect.  The reference is therefore the CONSTANT
# ``_MULTI_CONGRUENCE_NA_REF``, so the gate is an ABSOLUTE cutoff on the
# canonical dispersion -- 0.06 * 0.15 = 9.0e-3 rad for every design, every grid
# and every wavelength -- and the prescription cannot influence it at all (the
# stats helper does not even look at ``groups``).  The score keeps fga's
# currency and its 0.06 default so ``multi_congruence_threshold`` still means
# what it means in fga.  0.15 is not a tuned number: it is fga's OWN defensive
# fallback NA for an unclassifiable prescription (``_default_p_max`` returns it)
# and this module's own ``na_exact_threshold`` default.  An absolute cutoff is
# also the CONSISTENT choice, because detector A's documented envelope
# (``_NONCOLLIMATED_RESID_THRESH = 0.02 rad``) is absolute too.
#
# MEASURED SEPARATION of the canonical score.  Fixed 2.048 mm window,
# w = 300 um, dx0 swept 16 / 8 / 4 / 2 / 1 / 0.5 / 0.25 um (7 pitches) over 12
# single-valued and 6 multi-congruence constructions; "A" = detector A already
# fires on that row, so it is not a row this detector has to carry.
#   SINGLE-VALUED, canonical rad, WORST over all pitches
#     clean / super-gaussian n=8 / hard clip (unpropagated) / 3x3 separated
#       parallel emitters                                     0.0
#     strongly converging R = -20 mm                          1.3e-04
#     coma 30 rad                                             1.6e-04
#     clip 1.0 w + ASM 0.05 z_R                               1.1e-03
#     top hat band-limited to NA 0.05                         2.1e-03
#     clip 0.7 w + ASM 0.02 z_R   <- worst A-SILENT row       4.6e-03
#     clip 0.6 w + ASM 0.01 z_R                               1.1e-02  [A]
#   MULTI-CONGRUENCE, canonical rad, RANGE over all pitches
#     2 beams +-5 mrad                          1.2e-03 - 1.5e-03   SILENT
#     2 beams +-10 mrad                         3.1e-03 - 4.3e-03   SILENT
#     2 beams +-23 mrad                         8.5e-03 - 1.24e-02
#     2 beams +-46 mrad                         1.16e-02 - 3.48e-02
#     121 8x4 fan, EQUAL order phases           9.3e-03 - 1.92e-02
#     121 8x4 fan, RANDOM order phases          5.2e-03 - 1.44e-02  [A]
# The worst A-SILENT single-valued row is 4.6e-03 and the cutoff is 9.0e-3, a
# 1.9x margin held at EVERY pitch; the 121 fan fires at every pitch including
# dx0 = 0.25 um / N = 8192.  The one single-valued row that crosses the cutoff
# (clip 0.6 w + ASM 0.01 z_R, an aperture taking 70 % of the beam) trips
# detector A on its own account wherever it does (residual 2.01e-2 - 2.06e-2
# rad, outside the documented 0.02 rad envelope), so the gate's verdict there
# is not this detector's to defend.
#
# HONEST ENVELOPE -- and it is now a floor in ANGLE, not in grid pitch.  With
# canon ~ 3.5 theta^1.5 the 9.0e-3 rad cutoff puts detector B's detection floor
# at a crossing half-angle of ~19 mrad (~38 mrad total split).  Two comparable
# beams closer than that are not caught by B at ANY pitch -- measured +-10 mrad
# reads 3.1e-3 - 4.3e-3 rad, inside the clipped-beam population, so no cutoff
# separates them -- and are caught only if they also trip detector A.  That
# floor is the same order as A's own documented 0.02 rad envelope, so the two
# detectors bottom out together rather than leaving a band neither covers.
# Closing it needs an estimator that separates crossing congruences from edge
# ringing at small angle, which no shipped estimator does.
#
# The floor is stated in the angle between INTERFERING PAIRS.  Mapping a FAN
# onto that pair scale is the part an earlier cut of this note got wrong, in
# both magnitude and DIRECTION: it claimed the score is "set by the finest
# fringes, i.e. by the nearest-neighbour order spacing", so that a dense fan
# would hide far below its span.  It does not.  Re-measured with the shipped
# helper (``_chain_entry_congruence_stats``; the harness reproduces the 8x8
# row below to 3 digits, so this is the same measurement, not a competing one):
#
#   construction                       canonical rad, dx0 = 4 / 2 / 1 um   eq. PAIR
#   8x8 fan, span +-23, NN 6.571    7.83e-3 / 8.41e-3 / 8.93e-3   17.1-18.7 mrad
#   PAIR at that NN spacing +-3.286 7.37e-4 / 6.79e-4 / 6.46e-4     3.2-3.5 mrad
#   PAIR at that span       +-23.0  1.19e-2 / 1.22e-2 / 1.22e-2    22.5-23.0 mrad
#
# The fan reads 5.3x ABOVE what the nearest-neighbour rule predicts and 0.8x of
# an equal-SPAN pair, so the TOTAL SPAN -- not the order spacing -- is what
# carries the score.  Densifying at FIXED span moves it DOWN, not up, which is
# the direction the old rule got backwards: a 1-D fan of 4 / 8 / 16 orders
# spanning +-23 (NN 15.3 / 6.6 / 3.1 mrad) reads 7.56e-3 / 5.92e-3 / 5.04e-3
# canonical at dx0 = 2 um -- an equivalent pair of 16.7 / 14.2 / 12.8 mrad,
# a 1.5x drift over a 5x change in spacing, and never anywhere near the 3 mrad
# the spacing rule would demand.
#
# OPERATIONAL RULE, corrected: score a fan by its total span, derated ~20 %.
# A fan whose SPAN clears the ~19 mrad floor is caught even when its order
# spacing is far below the floor -- the old wording told callers the opposite,
# and was over-conservative rather than unsafe.  The two concrete verdicts it
# reported still stand on the re-measurement: the 8x8 +-23 fan sits ON the
# cutoff with no margin either way and is not reliably caught (though because
# its SPAN lands there, not its spacing), while the design-121 8x4 fan at
# +-46 / +-23 mrad reads 1.65e-2 / 1.82e-2 / 1.87e-2 and clears by ~2x at every
# pitch.  That boundary is pinned by
# ``test_the_documented_detection_floor_is_a_pinned_boundary`` so a future
# cutoff change cannot move it silently.
#
# One further non-physical corner: 20 % per-pixel amplitude noise reads 0.0 down
# to dx0 = 1 um but 4.7e-3 rad canonical at dx0 = 0.25 um, because its sign
# flips are a grid-tied construction rather than band-limited physical content.
#
# The measurement runs on the ENVELOPE the caller passes -- NOT on a
# reconstruction.  The chain always operates with its carrier beyond the grid
# Nyquist, so a reconstructed field would alias by construction and feed
# measurement A the very wrapped increments the caveat warns about; the
# envelope is the un-aliased quantity, and for a ``TiltedCarrier`` input it is
# the residual with the tilt already divided out (so a legitimately-tilted
# congruence run through the D1/D2 route stays silent).
_MULTI_CONGRUENCE_MV_THRESH = 0.06     # fga's own multivalued_threshold
_MULTI_CONGRUENCE_NA_REF = 0.15        # FIXED reference NA -- see note B.2


def _chain_entry_congruence_stats(env, dx, wavelength):
    """``(resid_rad, mv_score, raw_rad, canon_rad)`` at chain entry -- the two
    shipped detectors above, measured on the ENVELOPE ``env``.

    * ``resid_rad`` -- detector A, ``_carrier_residual_rms`` in radians,
      comparable to ``_NONCOLLIMATED_RESID_THRESH``.
    * ``raw_rad`` -- detector B as fga returns it, ``_tilt_dispersion`` at
      ``na=1``: an rms local direction cosine in radians ON THIS GRID, and so
      NOT a property of the field alone (it falls as sqrt(dx); note B.1).
    * ``canon_rad`` -- ``raw_rad * sqrt(wavelength / dx)``, the grid- and
      wavelength-canonical form of the same quantity (~ 3.5 theta^1.5 for a
      crossing half-angle theta).
    * ``mv_score`` -- ``canon_rad / _MULTI_CONGRUENCE_NA_REF``, in fga's own
      NA-normalized currency so it is comparable to fga's
      ``multivalued_threshold``.  The reference NA is a CONSTANT on purpose:
      normalizing by the first group's ``fga._system_na`` made the verdict a
      function of the first lens's f-number in BOTH directions (note B.2), so
      this helper is deliberately not given the prescription at all.

    Returns zeros for the two scores rather than raising if either estimator
    cannot be formed on the given field -- a diagnostic must never be the
    thing that kills a propagation."""
    from ..elements._lens_traced import _carrier_residual_rms
    from .fga import _tilt_dispersion
    E = np.asarray(env)
    try:
        resid = float(_carrier_residual_rms(E, None, wavelength, dx))
    except (ValueError, RuntimeError, FloatingPointError, ImportError):
        resid = 0.0
    try:
        raw = float(_tilt_dispersion(E, dx, dx, wavelength, 1.0))
    except (ValueError, RuntimeError, FloatingPointError, ImportError):
        raw = 0.0
    if not np.isfinite(resid):
        resid = 0.0
    if not np.isfinite(raw):
        raw = 0.0
    dxf = abs(float(dx))
    wlf = abs(float(wavelength))
    scale = np.sqrt(wlf / dxf) if (dxf > 0.0 and wlf > 0.0) else 1.0
    canon = float(raw * scale)
    if not np.isfinite(canon):
        canon = 0.0
    return resid, canon / _MULTI_CONGRUENCE_NA_REF, raw, canon


def _check_chain_entry_congruence(env, dx, wavelength, action,
                                  mv_threshold, fn):
    """P3 gate: refuse (or shout about) a multi-congruence / out-of-envelope
    chain input, naming the multi-congruence route.

    ``action='ignore'`` skips the measurement entirely (it is two full-grid
    passes plus a Gaussian filter), so the escape hatch is also the fast path.
    Returns the measured ``(resid_rad, mv_score, raw_rad, canon_rad)`` tuple,
    or ``None`` when the measurement was skipped."""
    if action == 'ignore':
        return None
    from ..elements._lens_traced import _NONCOLLIMATED_RESID_THRESH
    resid, mv, raw, canon = _chain_entry_congruence_stats(
        env, dx, wavelength)
    hit_spread = resid > _NONCOLLIMATED_RESID_THRESH
    hit_mv = mv > float(mv_threshold)
    if not (hit_spread or hit_mv):
        return (resid, mv, raw, canon)
    why = []
    if hit_spread:
        why.append(
            f"residual transverse angular spread {resid:.4f} rad > the "
            f"documented envelope _NONCOLLIMATED_RESID_THRESH="
            f"{_NONCOLLIMATED_RESID_THRESH} rad")
    if hit_mv:
        why.append(
            f"multi-valuedness score {mv:.4f} > {float(mv_threshold)} "
            f"(_tilt_dispersion, the local-wavevector spread "
            f"apply_real_lens_universal routes on: {canon:.4e} rad "
            f"grid-canonical, from {raw:.4e} rad raw at dx={float(dx):.4e} m, "
            f"against the fixed reference NA "
            f"{_MULTI_CONGRUENCE_NA_REF} -- a crossing half-angle of about "
            f"{(max(canon, 0.0) / 3.5) ** (2.0 / 3.0) * 1e3:.0f} mrad)")
    _guard_dispose(
        action,
        f"{fn}: the input is OUTSIDE the single-congruence envelope this "
        f"chain is validated for -- " + "; ".join(why) + ".  "
        "apply_real_lens_traced's entrance->exit map assumes ONE congruence "
        "per exit pixel and explicitly excludes comparable-power beams at "
        "well-separated angles (post-DOE at large split), so a fan pushed "
        "through multiplexed returns a populated, credible-looking image "
        "whose per-frame power is scrambled (measured 0.47 +/- 0.51 % "
        "against a design 2.78 %/frame) with no other symptom.  Use the "
        "MULTI-CONGRUENCE ROUTE instead: "
        "lumenairy.propagate_traced_carrier_chain_multi(congruences, ...), "
        "one congruence per DOE order / per emitter, each with its own "
        "lumenairy.TiltedCarrier(R, L, M, x0, y0) -- it runs each through "
        "this same shipped-default chain in its own chief-ray-tracking "
        "frame (so the split angle never enters this residual) and "
        "recombines coherently on one image grid.  A single congruence "
        "carried on the wrong reference is fixed in place by passing that "
        "congruence's TiltedCarrier as r_in.  Set "
        "on_multi_congruence='ignore' to silence this (or 'error' to make "
        "it fatal in batch production).")
    return (resid, mv, raw, canon)


# ---------------------------------------------------------------------------
# P2 memory budget for the exact-readout / fine-retrace grids (audit
# AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24 §4, "Memory ceiling")
# ---------------------------------------------------------------------------
# The exact readout's internal fine grid is sized from the PHYSICS
# (``window_factor`` beam radii at ``lambda/(3 NA)``) with no memory ceiling, so
# a high-NA / wide-window request can demand a single 32768^2 complex128 array =
# 16 GiB (measured on the design-121 thin-stub case) and the process dies with a
# MemoryError mid-propagation -- or worse, swaps for minutes.  These constants
# turn that into a bounded, ANNOUNCED degradation.
#
# Peak working set is ~4 arrays of the fine grid at complex128: the
# Fourier-upsample pad + its inverse transform, then the reconstructed field
# alongside the exact-sphere phasor, then the Bluestein zoom's own workspace.
_FINE_GRID_WORK_ARRAYS = 4
# Fraction of the RAM budget the fine grid may claim.  0.5 leaves room for the
# caller's own field, the OS page cache and the FFT plan scratch.
_FINE_GRID_RAM_FRAC = 0.5
# Never degrade below this (a grid this small is useless but the caller asked
# for a readout, so return something rather than raising).
_FINE_GRID_MIN = 64


def _memory_bounded_n_fine(n_req, label, *, ram_budget=None,
                           n_work=_FINE_GRID_WORK_ARRAYS,
                           frac=_FINE_GRID_RAM_FRAC,
                           window=None, nyquist_dx=None,
                           on_ram_cap='warn'):
    """Cap a fine-grid size ``n_req`` (square, complex128) to the RAM budget.

    Returns the (possibly reduced) size, always a power of two when ``n_req``
    is, and emits a ``RuntimeWarning`` naming the un-degraded requirement when
    the cap binds -- so a memory-limited result is never silently returned as if
    it were the requested resolution.

    ``ram_budget`` (bytes) overrides :func:`lumenairy.memory.get_ram_budget`
    (which itself honours :func:`lumenairy.set_max_ram`); pass ``inf`` to
    disable the cap entirely.

    ``on_ram_cap`` (D3 / roadmap P5) disposes of the bind: ``'warn'`` (default,
    the historical behaviour) degrades and announces; ``'error'`` raises a
    ``MemoryError`` so an unattended batch run fails loudly instead of
    reporting a metric computed on a grid coarser than the physics asked for --
    a ``RuntimeWarning`` labelled "RESOLUTION-LIMITED (non-converged)" is easy
    to lose in a production log; ``'ignore'`` degrades silently."""
    n_req = int(n_req)
    from ..memory import format_bytes, get_ram_budget
    budget = float(get_ram_budget() if ram_budget is None else ram_budget)
    if not np.isfinite(budget):
        return n_req
    if not (budget > 0.0):
        budget = 0.0
    allow = frac * budget
    per_grid = float(n_work) * 16.0
    n_max = int(np.floor(np.sqrt(max(allow, 0.0) / per_grid)))
    # keep the power-of-two structure the FFT path wants
    if n_max >= 2:
        n_max = int(2 ** int(np.floor(np.log2(n_max))))
    n_max = max(n_max, _FINE_GRID_MIN)
    if n_req <= n_max:
        return n_req
    extra_msg = ''
    if window is not None and nyquist_dx is not None and n_max > 0:
        _dxc = float(window) / n_max
        extra_msg = (
            f"  At the capped size the fine pitch is {_dxc * 1e6:.4f} um vs the "
            f"exact sphere's Nyquist pitch lambda/(2*NA)="
            f"{float(nyquist_dx) * 1e6:.4f} um"
            + (", so outer-NA content is DISCARDED (the downstream "
               "bandlimit masks the aliased corner)."
               if _dxc > float(nyquist_dx) else
               ", which still Nyquist-samples the sphere."))
    _guard_dispose(
        on_ram_cap,
        f"{label}: the fine grid is MEMORY-LIMITED to {n_max}x{n_max} "
        f"({format_bytes(n_work * n_max * n_max * 16)} peak working set).  The "
        f"un-degraded requirement was {n_req}x{n_req} "
        f"({format_bytes(n_work * n_req * n_req * 16)} at {n_work} complex128 "
        f"working arrays), which exceeds {frac:.0%} of the "
        f"{format_bytes(int(budget))} RAM budget.  The result is a "
        f"RESOLUTION-LIMITED (non-converged) readout: the sampling below is "
        f"coarser than the physics asked for.{extra_msg}  Raise the budget "
        f"(lumenairy.set_max_ram), shrink window_factor, or run on a larger "
        f"box to get the un-degraded number; pass on_ram_cap='error' to make "
        f"this fatal in batch production, or 'ignore' to accept the degraded "
        f"grid silently.",
        exc=MemoryError)
    return n_max


def carrier_referenced_exact_focus_readout(
    E_full: np.ndarray,
    R_carrier: float,
    z: float,
    wavelength: float,
    dx: float,
    *,
    dx_out: float,
    N_out: int,
    dx_fine: Optional[float] = None,
    N_fine: Optional[int] = None,
    window_factor: float = 7.0,
    centre_out: Tuple[float, float] = (0.0, 0.0),
    bandlimit: bool = True,
    ram_budget: Optional[float] = None,
    on_ram_cap: str = 'warn',
    centre: Tuple[float, float] = (0.0, 0.0),
    tilt: Tuple[float, float] = (0.0, 0.0),
    on_readout_window: str = 'error',
    readout_window_tol: float = 1e-4,
    _period_out: Optional[dict] = None,
) -> np.ndarray:
    """Exact (non-paraxial) readout of a strongly-converging FINAL leg (R9).

    Propagate a high-NA converging FULL field ``E_full`` (e.g. the exit field
    of the last traced group) a distance ``z`` to the image plane WITHOUT the
    paraxial carrier re-envelope.  ``E_full`` is referenced to its own EXACT
    spherical wavefront ``S(R) = sign(R)(sqrt(r^2+R^2)-|R|)`` (not the paraxial
    parabola), the resulting SMOOTH envelope is resampled onto a grid that
    Nyquist-samples the exact sphere (``dx_fine <= lambda/(2 NA)``), the field
    is reconstructed, and it is propagated to the target with the exact
    band-limited angular-spectrum Bluestein zoom
    (:func:`angular_spectrum_propagate_mft`).  See the module's "Exact
    high-NA final leg" section for why the paraxial carrier cannot focus this
    leg.

    Unlike :func:`carrier_referenced_focus_readout` (which takes the ENVELOPE
    and carrier-steps short of the focus before an ASM zoom -- a PARAXIAL
    envelope leg), this takes the FULL field and never applies a paraxial
    magnification/curvature, so it focuses an NA-0.5 sphere to the diffraction
    limit.  For a LOW-NA leg the two agree; use the paraxial (faster) path
    there.

    Parameters
    ----------
    E_full : ndarray, complex, shape (Ny, Nx)
        The FULL physical field (carrier NOT divided out) on the ``dx`` grid --
        its wavefront must be ~sphere(R_carrier) plus a smooth residual (the
        genuine aberration).  The grid may UNDER-sample the exact sphere (the
        typical co-moving-grid case): the sphere-referencing de-aliases it
        pointwise before the resample.
    R_carrier : float
        Signed radius of the exact spherical wavefront to reference (m); the
        beam's paraxial exit radius ``R_out``.  ``R < 0`` converging.  ``+/-inf``
        (collimated) skips the sphere reference (a plain ASM-MFT).
    z : float
        Distance from ``E_full``'s plane to the target/image plane (m).
    wavelength, dx : float
        Wavelength and input grid pitch (m).
    dx_out : float
        Output (focal) grid pitch (m) -- pick it to resolve the focused spot
        (``<~ lambda/(few*NA)``).
    N_out : int
        Output grid size (square).
    dx_fine : float, optional
        Pitch of the intermediate fine reconstruction grid (m).  Default:
        ``lambda/(3*NA)`` from the measured NA (``NA = w/|R|``), so the exact
        sphere is comfortably Nyquist-sampled.
    N_fine : int, optional
        Size of the intermediate fine grid (square).  Default: the next power
        of two that spans ``window_factor`` amplitude-radii of the beam at
        ``dx_fine``.

        Either way the value is capped to the RAM budget (see ``ram_budget``):
        the physics-driven default can demand 32768^2 complex128 = 16 GiB per
        working array (measured, design-121 class), which used to die with a
        ``MemoryError`` mid-propagation.  When the cap binds, a
        ``RuntimeWarning`` names the un-degraded requirement and the result is
        flagged as resolution-limited rather than silently returned as if it
        were the requested resolution.
    window_factor : float, default 7.0
        Fine-grid physical span in units of the beam amplitude radius
        (``_envelope_amp_radius``).  7 holds the beam to <1e-6 truncation.

        DOUBLE APPLICATION when reached from
        :func:`propagate_traced_carrier_chain`'s exact final leg: the SAME
        ``window_factor`` has already cropped the pre-readout re-trace grid to
        ``window_factor * w_entrance`` in :func:`_fine_trace_group_exit`, so
        this crop is the SECOND one and it re-measures ``w`` on the field that
        survived the first.  Truncating a Gaussian at ``a = wf/2`` beam radii
        lowers its measured second moment by
        ``rho(a) = sqrt((1-(1+u)e^-u)/(1-e^-u))``, ``u = 2 a^2/w^2`` (square
        crop, so slightly above the disc value), and the second cut therefore
        lands at ``wf * rho`` beam radii, not ``wf``.  Measured 2026-07-25
        (Gaussian ``w = 200 um``, NA 0.10, ``lambda = 1.31 um``):

        ==== ============= ================ ================================
        wf   2nd cut at    win2/win1        at-focus FWHM / EE3 / window
        ==== ============= ================ ================================
        2.0  0.880 w       0.880            7.750 um / 29.8% / 79.8%
        3.0  1.480 w       0.987            5.550 um / 59.1% / 99.1%
        4.0  2.000 w       1.0000 (no-op)   5.150 um / 63.6% / 100.0%
        7.0  3.500 w       1.0000 (no-op)   5.050 um / 63.9% / 100.0%
        ==== ============= ================ ================================

        (single-crop reference at wf=2.0: 6.950 um / 38.4% / 88.0% -- the
        compounding costs 8.6 EE3 points and 0.8 um of FWHM there.)  So the
        two crops are INTENDED and independent -- the first sizes the retrace
        grid to the ENTRANCE beam, the second sizes the readout to the EXIT
        beam, which is the right window for each -- but they compound below
        ``wf ~ 3``.  At the default ``wf = 7`` (and anything >= 4) the second
        crop is a measured exact no-op.  This is the ``1 w crop applied
        twice`` confound flagged in audit
        AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24 §6.4, quantified.
    centre_out : (float, float), optional
        Physical ``(x, y)`` centre of the output grid (m).  Default on-axis.
    bandlimit : bool, default True
        Band-limit the ASM transfer function (Matsushima-Shimobaba).
    ram_budget : float, optional
        Memory budget in BYTES for the internal fine grid (P2, audit
        AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24 §4).  Default ``None`` =
        :func:`lumenairy.get_ram_budget` (which honours
        :func:`lumenairy.set_max_ram`).  ``N_fine`` is capped so the fine grid's
        peak working set (4 complex128 arrays) stays within 50% of it; when the
        cap binds a ``RuntimeWarning`` states that the readout is
        resolution-limited and what the un-degraded requirement was.  Pass
        ``float('inf')`` to disable the cap (pre-v5.29 behaviour: a hard
        ``MemoryError`` when the physics asks for more than the box has).
    on_ram_cap : {'warn', 'error', 'ignore'}, default 'warn'
        Disposition when the ``ram_budget`` cap BINDS (niche D3, roadmap
        ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27 P5).  ``'warn'`` is the
        historical behaviour: degrade the fine grid, emit the
        "RESOLUTION-LIMITED (non-converged)" ``RuntimeWarning`` and return the
        metric computed on the degraded grid.  In an unattended batch run that
        warning is easy to lose and the degraded number reads like a converged
        one, so ``'error'`` raises a ``MemoryError`` instead -- fail loudly
        rather than report a non-converged result.  ``'ignore'`` degrades
        silently.
    centre : (float, float), default (0, 0)
        Transverse position ``(x0, y0)`` (m) of the congruence's CHIEF RAY on
        the input grid -- niche D6.  Everything the readout references to "the
        beam" moves with it: the exact sphere is centred there, the beam
        radius is a second moment ABOUT it, the ``window_factor`` crop is
        taken about it (so the internal fine grid costs exactly what an
        on-axis beam of the same size costs, however far off axis it sits),
        and ``centre_out`` stays in ABSOLUTE (optical-axis) coordinates.
        The default is short-circuited, so the on-axis path is byte-identical.
    tilt : (float, float), default (0, 0)
        Direction cosines ``(L, M)`` of the congruence's uniform tilt about
        ``centre`` -- niche D6.  Referenced out with the sphere so the
        resampled envelope is the genuine aberration residual, then restored
        on the fine grid; the ASM Bluestein zoom then carries the tilt's own
        transverse advance and path piston EXACTLY (no paraxial ``L*z``
        approximation and no obliquity bookkeeping, unlike
        :func:`carrier_referenced_focus_readout`'s paraxial leg, which has to
        reimpose both by hand).  ``dx_fine`` must Nyquist-sample the tilt
        (``dx_fine <= lambda/(2*hypot(L, M))``); the sphere's own NA-driven
        default is far finer than that for any tilt a lens group can deliver.
    on_readout_window : {'error', 'warn', 'ignore'}, default 'error'
        Disposition when the ``window_factor`` crop CANNOT be taken at the
        requested size because it does not fit inside the input grid at
        ``centre``, AND the shortfall truncates measurable beam power.

        The crop is necessarily bounded by what the grid holds -- at a chief
        ray ``(cx, cy)`` only ``N*dx - 2*max(|cx|, |cy|)`` is available -- and
        until v5.32.1 that bound was applied SILENTLY, so a decentred readout
        degraded with no symptom while the beam still sat comfortably on the
        grid.  Measured (1024 x 0.5 um grid, Gaussian ``w`` = 40 um,
        ``R`` = -400 um, ``z`` = 400 um, ``window_factor`` = 6) against a plain
        :func:`~lumenairy.propagators.mft.angular_spectrum_propagate_mft` on
        the same input grid: ``cx`` = 0 and 150 um agree to 3e-5 / 3e-4 of the
        peak, ``cx`` = 200 um returns 0.919 of the power at 0.906 of the peak,
        and ``cx`` = 230 um returns **0.279 of the power at 0.435 of the
        peak** -- all three with an empty warning list.  The guard now measures
        the truncated power fraction directly from ``E_full`` and refuses
        above ``readout_window_tol``.
    readout_window_tol : float, default 1e-4
        Fraction of the field's power that the clamp above may cut before
        ``on_readout_window`` fires.  Only consulted when the clamp actually
        binds, so a call whose window fits is untouched (and byte-identical).
        The shipped chain never reaches it: its second crop lands at
        ``window_factor * rho`` beam radii with ``rho < 1`` (see
        ``window_factor``), i.e. strictly inside the grid.

    Returns
    -------
    E_out : ndarray, complex, shape (N_out, N_out)
        The full physical field at the target plane on the centred ``dx_out``
        grid -- same absolute-phase convention as
        :func:`angular_spectrum_propagate_mft`.

    Other Parameters
    ----------------
    _period_out : dict, optional
        PRIVATE.  When a dict is supplied, its ``'period'`` key is filled with
        the ``(period_x, period_y)`` spatial period (m) of the final Bluestein
        zoom (:func:`~lumenairy.propagators.mft._asm_mft_spatial_period` on
        the internal fine grid).  A requested window wider than that returns
        periodic REPLICAS rather than signal, so
        :func:`propagate_traced_carrier_chain` reports the value as
        ``readout_period`` in its stages and
        :func:`propagate_traced_carrier_chain_multi` refuses a per-congruence
        window that exceeds it (niche D2).  Not part of the public contract.
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_full, 'carrier_referenced_exact_focus_readout',
                           input_kind='field')
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError(
            "carrier_referenced_exact_focus_readout: wavelength must be finite "
            f"and positive, got {wavelength!r}.")
    if not (np.isfinite(dx) and dx > 0):
        raise ValueError(
            "carrier_referenced_exact_focus_readout: dx must be finite and "
            f"positive, got {dx!r}.")
    if not (np.isfinite(dx_out) and dx_out > 0):
        raise ValueError(
            "carrier_referenced_exact_focus_readout: dx_out must be finite and "
            f"positive, got {dx_out!r}.")
    _check_guard_action('on_ram_cap', on_ram_cap,
                        'carrier_referenced_exact_focus_readout')
    _check_guard_action('on_readout_window', on_readout_window,
                        'carrier_referenced_exact_focus_readout')
    if not (np.isfinite(readout_window_tol) and readout_window_tol >= 0.0):
        raise ValueError(
            "carrier_referenced_exact_focus_readout: readout_window_tol must "
            f"be a finite non-negative fraction, got {readout_window_tol!r}.")
    if int(N_out) <= 0:
        raise ValueError(
            "carrier_referenced_exact_focus_readout: N_out must be > 0, got "
            f"{N_out!r}.")
    if not np.isfinite(z):
        raise ValueError(
            "carrier_referenced_exact_focus_readout: z must be finite, got "
            f"{z!r}.")

    from ..backend import to_numpy
    E = np.asarray(to_numpy(E_full))
    N = E.shape[-1]
    k = 2.0 * np.pi / wavelength
    R = float(R_carrier)
    _cx, _cy = float(centre[0]), float(centre[1])
    _tL, _tM = float(tilt[0]), float(tilt[1])
    for _nm, _v in (('centre[0]', _cx), ('centre[1]', _cy),
                    ('tilt[0]', _tL), ('tilt[1]', _tM)):
        if not np.isfinite(_v):
            raise ValueError(
                f"carrier_referenced_exact_focus_readout: {_nm} must be "
                f"finite, got {_v!r}.")
    _dec = bool(_cx or _cy or _tL or _tM)
    if _dec:
        _tilt_obliquity(_tL, _tM, 'carrier_referenced_exact_focus_readout')
    w_amp = _envelope_amp_radius(E, dx, dx,
                                 centre=((_cx, _cy) if _dec else (0.0, 0.0)))

    # -- exact-sphere envelope (SMOOTH; the two steep phases alias-cancel on the
    #    shared grid, leaving only amplitude x exp(i * genuine aberration)) -----
    if np.isfinite(R) and R != 0.0:
        S = _exact_sphere_eikonal(E.shape, dx, dx, wavelength, R,
                                  centre=((_cx, _cy) if _dec else (0.0, 0.0)))
        env = E * np.exp(-1j * k * S)
    else:
        env = E
    if _tL or _tM:
        # niche D6: reference the uniform tilt out too, so what gets resampled
        # is the genuine (smooth) aberration residual rather than a residual
        # riding a linear ramp.  Restored on the fine grid below.
        _rp = _tilt_ramp(E.shape, dx, wavelength, _tL, _tM, _cx, _cy, -1)
        if _rp is not None:
            env = env * _rp

    # -- fine grid sizing -----------------------------------------------------
    if dx_fine is None:
        if np.isfinite(R) and R != 0.0 and w_amp > 0.0:
            na = min(max(w_amp / abs(R), 0.02), 0.95)
        else:
            na = 0.1
        dx_fine = wavelength / (3.0 * na)
    dx_fine = float(dx_fine)
    if not (dx_fine > 0.0):
        raise ValueError(
            "carrier_referenced_exact_focus_readout: dx_fine must be > 0, got "
            f"{dx_fine!r}.")
    # crop window (physical) that holds the beam.  Decentred (niche D6): the
    # crop is taken ABOUT THE CHIEF RAY, so the window is sized by the beam
    # alone -- the same cost as an on-axis beam of the same radius, however
    # far off axis it sits -- but it is bounded by what fits at that offset.
    _avail = N * dx - 2.0 * max(abs(_cx), abs(_cy)) if _dec else N * dx
    _win_want = float(window_factor) * w_amp if w_amp > 0.0 else _avail
    win = min(_win_want, _avail)
    # The clamp above is GEOMETRICALLY forced (nothing outside the grid can be
    # cropped), but it used to be silent -- and at a decentred ``centre`` it
    # shrinks with the offset while the beam still sits comfortably on the
    # grid, which is the exact plausible-looking-wrong-answer class this
    # campaign exists to prevent.  Report it, sized by the power it actually
    # costs rather than by the window ratio (a clamp from 7 to 5 beam radii
    # costs ~1e-11 and must stay silent).
    _clamped = _win_want > _avail * (1.0 + 1e-12)
    if not (win > 0.0):
        raise ValueError(
            f"carrier_referenced_exact_focus_readout: the chief ray at "
            f"({_cx * 1e3:+.4f}, {_cy * 1e3:+.4f}) mm is off the "
            f"{N * dx * 1e3:.4f} mm input grid (N={N} x dx={dx * 1e6:.4f} um), "
            f"so no readout window can be centred on it.")
    n_crop = int(2 * round((win / dx) / 2))
    n_crop = int(min(max(n_crop, 2), N))
    win = n_crop * dx
    if _clamped and on_readout_window != 'ignore':
        # measured, not modelled: the fraction of |E_full|^2 outside the crop
        # box actually taken.  |env| == |E_full| (the references are pure
        # phase), so this is the power the readout is about to throw away.
        _ax = (np.arange(N, dtype=np.float64) - N / 2) * dx
        _in_x = np.abs(_ax - _cx) <= 0.5 * win
        _in_y = np.abs(_ax - _cy) <= 0.5 * win
        _p = np.abs(E) ** 2
        _tot = float(_p.sum())
        _lost = (1.0 - float(_p[np.ix_(_in_y, _in_x)].sum()) / _tot
                 if _tot > 0.0 else 0.0)
        if _lost > float(readout_window_tol):
            _guard_dispose(
                on_readout_window,
                f"carrier_referenced_exact_focus_readout: the requested "
                f"window_factor={window_factor} x beam radius "
                f"{w_amp * 1e6:.4f} um = {_win_want * 1e6:.4f} um does NOT "
                f"fit on the input grid centred at the chief ray "
                f"({_cx * 1e6:+.4f}, {_cy * 1e6:+.4f}) um: only "
                f"{_avail * 1e6:.4f} um is available there "
                f"(= N*dx {N * dx * 1e6:.4f} um - 2 x "
                f"{max(abs(_cx), abs(_cy)) * 1e6:.4f} um of offset), so the "
                f"crop was clamped to {win * 1e6:.4f} um and TRUNCATES "
                f"{_lost * 100:.4f} % of the field's power (tolerance "
                f"{float(readout_window_tol) * 100:.4g} %).  That loss is not "
                f"a windowing detail: it removes real beam, so the returned "
                f"peak, power and encircled energy are all low (measured on a "
                f"Gaussian stand-in: 0.919 of the power at 0.906 of the peak "
                f"one step past the tolerance, 0.279 at 0.435 two steps "
                f"past).  Remedies: raise N (or dx) so the grid holds "
                f"window_factor beam radii AT that offset; lower "
                f"window_factor; or re-centre the input grid on the "
                f"congruence.  Pass on_readout_window='warn' to accept the "
                f"truncated window, 'ignore' to skip the measurement "
                f"entirely, or raise readout_window_tol.",
                stacklevel=2)
    if N_fine is None:
        N_fine = int(2 ** int(np.ceil(np.log2(max(win / dx_fine, n_crop)))))
    N_fine = int(N_fine)
    # P2 memory budget: cap the fine grid to what the RAM budget can hold and
    # SAY SO (the pre-v5.29 path died with a MemoryError at 32768^2 = 16 GiB per
    # array).  The Nyquist consequence of the coarser dx_fine is spelled out in
    # the warning, mirroring the F-D message in _fine_trace_group_exit.
    _na_ny = (min(max(w_amp / abs(R), 0.02), 0.95)
              if (np.isfinite(R) and R != 0.0 and w_amp > 0.0) else 0.1)
    N_fine = _memory_bounded_n_fine(
        N_fine, 'carrier_referenced_exact_focus_readout', ram_budget=ram_budget,
        window=win, nyquist_dx=wavelength / (2.0 * _na_ny),
        on_ram_cap=on_ram_cap)
    dx_fine = win / N_fine

    if _dec:
        env_f = _fourier_upsample_crop(
            _crop_about_centre(env, dx, _cx, _cy, n_crop,
                               'carrier_referenced_exact_focus_readout'),
            n_crop, N_fine)
    else:
        env_f = _fourier_upsample_crop(env, n_crop, N_fine)

    # -- reconstruct the exact sphere on the fine grid ------------------------
    # (decentred: the fine grid is CENTRED ON THE CHIEF RAY, so the sphere and
    # the tilt ramp are both referenced to its own origin.)
    if np.isfinite(R) and R != 0.0:
        S_f = _exact_sphere_eikonal((N_fine, N_fine), dx_fine, dx_fine,
                                    wavelength, R)
        E_fine = (env_f * np.exp(1j * k * S_f)).astype(np.complex128)
    else:
        E_fine = np.asarray(env_f, dtype=np.complex128)
    if _tL or _tM:
        _rp = _tilt_ramp((N_fine, N_fine), dx_fine, wavelength, _tL, _tM,
                         0.0, 0.0, +1)
        if _rp is not None:
            E_fine = E_fine * _rp

    # -- exact band-limited ASM Bluestein zoom to the target ------------------
    # The ASM is translation-covariant, so propagating on the chief-ray-centred
    # grid and asking for the window at ``centre_out - chief`` returns exactly
    # the physical field at the ABSOLUTE ``centre_out`` -- including the tilt's
    # own transverse advance and path piston, which the propagator computes
    # from the field itself rather than from a paraxial bookkeeping term.
    from .mft import _asm_mft_spatial_period, angular_spectrum_propagate_mft
    if _period_out is not None:
        _period_out['period'] = _asm_mft_spatial_period(
            N_fine, dx_fine, N_fine, dx_fine)
    _co = ((float(centre_out[0]) - _cx, float(centre_out[1]) - _cy) if _dec
           else centre_out)
    return angular_spectrum_propagate_mft(
        E_fine, z, wavelength, dx_fine, dx_out, int(N_out),
        centre_out=_co, bandlimit=bandlimit)


# ===========================================================================
# Chain orchestrator (audit F4.1): carrier-referenced traced element chain
# ===========================================================================
# The per-group hand-off -- analytic carrier leg -> reconstruct at the group
# front vertex -> apply_real_lens_traced(carrier=R_in) -> re-envelope with the
# group's exit curvature R_out -- is ~30 lines of user code per chain and needs
# the element's OWN exit curvature R_out, which in the repro script came from an
# EXTERNAL ABCD q-trace.  :func:`propagate_traced_carrier_chain` packages the
# hand-off AND SUPPLIES R_out from each group's own paraxial ABCD
# (:func:`system_abcd_prescription` applied to the incoming carrier), so the
# caller needs no external q-trace.  The final leg optionally lands near the
# image-plane focus via :func:`carrier_referenced_focus_readout`.


class TracedCarrierChainResult(NamedTuple):
    """Result of :func:`propagate_traced_carrier_chain`.

    Attributes
    ----------
    field : ndarray, complex
        Full physical field at the final (target) plane.  For a focus-readout
        landing it is on the ``(dx, N_out)`` fine grid; otherwise on the
        co-moving grid ``dx``.
    R : float or None
        Carrier radius at the final plane (m); ``None`` after a focus readout
        (the Bluestein output carries its own absolute phase, not a single
        referenced carrier).
    dx : float
        Grid pitch of ``field`` (m).
    stages : list of dict
        Per-group diagnostics, one dict per lens group, each with keys
        ``name``, ``R_in``, ``R_out`` (m), ``dx`` (m), ``w`` (1/e^2 envelope
        radius, m) and ``power`` (sum |env|^2 dx^2).

        A DOE entry in ``groups`` (niche D4) appends its own stage instead,
        marked ``'doe': True`` and carrying ``order`` (the ``(mx, my)`` this
        run used -- which is the .zmx's design order unless the entry or the
        multi orchestrator's ``doe_order`` overrode it), ``dL`` / ``dM`` (the
        grating-equation increments to the direction cosines), the resulting
        ``L_out`` / ``M_out``, the chief ray ``x_c`` / ``y_c`` AT THE DOE
        PLANE, ``gap_before`` / ``gap_after``, ``amplitude`` and ``power``
        (sum |env|^2 dx^2 ACROSS the screen, i.e. with ``amplitude``
        applied).  It has no ``w`` / ``dx``: the envelope crosses the DOE
        plane inside the neighbouring transport step (see the module note),
        so no grid state exists there to report -- but power is invariant
        along that leg, so it is well defined at the screen, and reporting it
        is what makes a TRAILING DOE's amplitude visible to the exit-power
        accounting in :func:`propagate_traced_carrier_chain_multi`.

        Under a TILTED carrier (niche D1, ``r_in=TiltedCarrier(...)``) every
        group entry additionally carries the transferred congruence --
        ``L_out``, ``M_out`` (rad) and ``x_c_out``, ``y_c_out`` (m, the chief
        ray's absolute transverse position at that group's exit vertex) --
        and ONE extra entry is appended for the target plane:
        ``{'name': '<target>', 'target': True, 'L', 'M', 'x_c', 'y_c',
        'dx'}`` (plus ``'centre_out'`` after a focus readout).  Nothing is
        added on the scalar path.

        After a ``focus_readout`` landing ONE more key appears --
        ``'readout_period'``, the ``(period_x, period_y)`` spatial period (m)
        of the readout's Bluestein reconstruction (on the ``<target>`` entry
        when the carrier is tilted, else on the last group's entry).  The
        reconstruction is PERIODIC, and near a focus the co-moving grid it
        inverts has collapsed, so the period can be far smaller than the
        input window: a requested ``N_out * dx_out`` WIDER than it fills the
        outer window with periodic REPLICAS of the beam's own spot rather
        than signal (audit P11).  Sized so an accumulating caller can tell
        the two apart -- :func:`propagate_traced_carrier_chain_multi` refuses
        such a window by default (niche D2).

        Also after a ``focus_readout`` landing, and only when ``final_leg !=
        'paraxial'``, the LAST group's entry carries ``'na_exit'`` -- the
        measured exit NA (``w_entrance / |R_out|``) that ``final_leg='auto'``
        branches on against ``na_exact_threshold`` (niche D3).  Reported so a
        consumer can see how near the route flip their design sits without
        having to catch the ``on_na_proximity`` warning.  For scale: design
        121 measures ``na_exit`` = 0.405 (w = 3.126 mm over
        ``R_out`` = -7.712 mm on its last group) against the 0.15 default --
        170 % ABOVE it, i.e. NOT a near miss, and it would take a ~2.7x
        beam-radius shrink to flip.  (The "0.152" quoted as design 121's exit
        NA elsewhere is its GEOMETRIC system NA, aperture/EFL -- a different
        quantity from the one this router branches on; ``AUDIT_TRACED_CARRIER_
        CHAIN_2026_07_21.md`` already recorded the last leg at "NA ~ 0.46,
        R_out = -7.71 mm".)  The paraxial side of the flip is ~200 rad of
        wavefront wrong at that NA, which is why the guard exists at all.

        On the EXACT final leg that entry is joined by three more (niche C1
        item 4): ``'na_exit_measured'`` -- the exit NA
        :func:`~lumenairy.elements.apply_real_lens_traced` actually measured
        from the traced exit direction cosines on the grid it used (design 121
        order (-4,-2): **0.4780**, against the 0.4052 paraxial ``'na_exit'``
        that SIZED that grid) -- plus ``'na_grid_nyquist'``
        (``lambda/(2*dx_fine)``, the NA the retrace grid can carry) and
        ``'exit_power_above_nyquist'`` (the |E_in|^2-weighted fraction of
        traced exit power above it, **7.97e-04** at the shipped
        ``n_fine_cap`` = 12288).  The last of those is what
        ``on_tilt_exact_grid`` refuses on.
    """

    field: np.ndarray
    R: Optional[float]
    dx: float
    stages: list


def _group_abcd(prescription, wavelength):
    """``(A, B, C, D)`` of the group's air-to-air paraxial ABCD
    (:func:`system_abcd_prescription`), front vertex -> back vertex."""
    from ..raytrace.seidel import system_abcd_prescription
    M, _efl, _bfl, _ffl = system_abcd_prescription(prescription, wavelength)
    return (float(M[0, 0]), float(M[0, 1]),
            float(M[1, 0]), float(M[1, 1]))


def _paraxial_group_r_out(prescription, R_in, wavelength):
    """Exit carrier radius the GROUP itself supplies: its air-to-air paraxial
    ABCD (:func:`system_abcd_prescription`) mapped onto the incoming carrier
    radius by the wavefront Moebius law ``R_out = (A R_in + B)/(C R_in + D)``.

    This reproduces the design-121 repro script's external q-trace ``R_out`` to
    full precision (the two are algebraically identical for the geometric
    wavefront radius), so the orchestrator needs no external q-trace."""
    A, B, C, D = _group_abcd(prescription, wavelength)
    if not np.isfinite(R_in):
        return np.inf if abs(C) < 1e-300 else A / C
    num = A * R_in + B
    den = C * R_in + D
    if abs(den) < 1e-300:
        return np.inf
    return num / den


# ===========================================================================
# Tilted chain carriers (niche D1 -- roadmap
# ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27 P1a)
# ===========================================================================
# The chain state is (R, L, M, x_c, y_c): the exact sphere ``R`` centred on the
# CHIEF RAY at transverse position ``(x_c, y_c)``, plus the uniform tilt
# ``(L, M)`` referenced to that point.  Two facts make it carryable:
#
#   (1) the state is CLOSED under paraxial (ABCD) transfer.  For an entrance
#       congruence W(u) = S_R(u) + L u about u = x - x_c, the ray at u has
#       slope u/R + L; through an air-to-air (A B; C D) the ray at absolute
#       height h = x_c + u leaves at
#           h'  = A(x_c + u) + B(u/R + L) = (A x_c + B L) + u (A + B/R)
#           u'  = C(x_c + u) + D(u/R + L) = (C x_c + D L) + u (C + D/R)
#       so with x_c' = A x_c + B L and L' = C x_c + D L (the CHIEF RAY,
#       transforming as an ordinary paraxial ray) the residual slope is
#           u' - L' = u (C + D/R) = (h' - x_c') (C R + D)/(A R + B)
#       i.e. exactly a sphere of radius R' = (A R + B)/(C R + D) about x_c'
#       plus the uniform tilt L'.  ``_paraxial_group_r_out`` already applies
#       that Moebius law, so only the chief ray is new.  Free space (A=1, B=z,
#       C=0, D=1) reduces to x_c' = x_c + L z, L' = L, R' = R + z -- the tilt
#       is INVARIANT in the chief-ray-tracking frame, which is why the chain
#       transports the envelope with the untouched scalar code.
#
#   (2) the envelope transport in that tracking frame is BYTE-IDENTICAL to the
#       scalar path.  Writing E_0(u) = env_0(u) exp(ik[u^2/2R + L u]) and using
#       the Fresnel tilt theorem
#           F_z[f(x) e^{ikLx}](X) = e^{ikLX} e^{-ikL^2 z/2} F_z[f](X - Lz)
#       with the frame origin advanced by exactly ``L z`` gives
#           E_z(U) = env_z(U) exp(ik[U^2/2R' + L U]) * exp(+i k L^2 z / 2)
#       where ``env_z`` is precisely what :func:`propagate_carrier_referenced`
#       returns for the tilt-free problem.  So a free leg costs one scalar
#       piston multiply and two float updates -- no resampling, no shift, and
#       no walk-off as the co-moving grid collapses toward a focus.
#
# The one place the frame has to be left is the ELEMENT: a real prescription is
# traced on the grid, so the beam must sit at its true transverse position for
# the surfaces it actually crosses to be the ones it crosses.  The hand-off
# therefore band-limit-shifts the (smooth) envelope onto the axis-centred grid,
# reconstructs against the DECENTRED sphere + tilt, calls the element with the
# matching :class:`~lumenairy.TiltedCarrier`, and shifts back.


def _parse_chain_carrier(r_in, fn):
    """Normalise a chain entrance carrier to ``(R, L, M, x0, y0, tilted)``.

    Accepts a scalar radius (the historical form), a
    :class:`~lumenairy.TiltedCarrier`, or a 3/5-sequence
    ``(R, L, M[, x0, y0])``.  ``tilted`` is False for anything expressible as
    a scalar, which is what keeps the default path byte-identical."""
    from ..elements._lens_traced import TiltedCarrier
    if isinstance(r_in, TiltedCarrier):
        spec = r_in
    elif isinstance(r_in, (tuple, list, np.ndarray)) \
            and not np.isscalar(r_in) and np.ndim(r_in) == 1:
        vals = [float(v) for v in np.asarray(r_in, dtype=np.float64).ravel()]
        if len(vals) not in (3, 5):
            raise ValueError(
                f"{fn}: a sequence carrier must be (R, L, M) or "
                f"(R, L, M, x0, y0); got {len(vals)} values.  Pass a scalar "
                f"radius for an on-axis carrier, or a TiltedCarrier.")
        spec = TiltedCarrier(*vals)
    else:
        R = float(r_in)
        return R, 0.0, 0.0, 0.0, 0.0, False
    R = float(spec.R)
    if R == 0.0:
        raise ValueError(f"{fn}: carrier R == 0 (the carrier's own focus).")
    L, M = float(spec.L), float(spec.M)
    x0, y0 = float(spec.x0), float(spec.y0)
    for nm, v in (('L', L), ('M', M), ('x0', x0), ('y0', y0)):
        if not np.isfinite(v):
            raise ValueError(
                f"{fn}: TiltedCarrier.{nm} must be finite, got {v!r}.")
    return R, L, M, x0, y0, bool(L or M or x0 or y0)


def _tilt_obliquity(L, M, fn):
    """``1 / cos(theta) = 1 / sqrt(1 - L^2 - M^2)`` for direction cosines
    ``(L, M)`` -- the EXACT free-space obliquity factor.

    Why it is not just ``1`` (i.e. why the chief ray does not advance by
    ``L z``): expanding the exact angular-spectrum propagator
    ``sqrt(k^2 - kx^2)`` about the carrier's own transverse wavenumber
    ``kx = k L`` gives, in ``q = kx - kL``,

    .. code-block:: text

        sqrt(k^2 - (kL+q)^2) = k sqrt(1-L^2) - [L/sqrt(1-L^2)] q
                               - q^2 / (2 k (1-L^2)^{3/2}) + ...

    -- a piston ``k z sqrt(1-L^2)`` (path length ``z/cos(theta)`` along the
    tilted ray, which is where the ``+`` sign of the carried piston comes
    from), a LINEAR term that translates the envelope by exactly
    ``z L / sqrt(1-L^2)``, and only then the paraxial ``q^2`` diffraction.  So
    the transverse advance and the piston are available EXACTLY at no cost,
    while the residual envelope keeps the chain's usual paraxial transport.

    Measured on the D1 synthetic relay (46 mrad, N-BK7 singlet, throw
    59.07 mm): the paraxial advance ``L z`` lands the wave centroid 2.73 um
    short of an exact meridional ray trace -- a full 0.15 of the spot FWHM --
    and the exact factor closes it to 0.09 um.  The term is ``z L^3 / 2``, so
    it grows as the CUBE of the split angle: invisible for a 5 mrad tilt,
    dominant for a 46 mrad DOE order.

    Only the ENVELOPE's own diffraction stays on-axis-paraxial (the ``q^2``
    coefficient above wants ``z/(1-L^2)^{3/2}``, a +0.32% effective-distance
    stretch at 46 mrad) -- the same paraxial-transport caveat the chain
    documents for its Sziklas-Siegman legs."""
    s = float(L) ** 2 + float(M) ** 2
    if not (s < 1.0):
        raise ValueError(
            f"{fn}: the carrier tilt (L={L!r}, M={M!r}) has "
            f"L^2 + M^2 = {s:.6g} >= 1, i.e. it is not a propagating "
            f"direction.  L and M are DIRECTION COSINES (sin of the angle), "
            f"not slopes.")
    return 1.0 / np.sqrt(1.0 - s)


def _shift_envelope(env, sx, sy, dx):
    """Band-limited translation of a SMOOTH envelope by ``(sx, sy)`` metres:
    returns ``env_out(x, y) = env(x - sx, y - sy)`` via a Fourier phase ramp.

    Exact for any band-limited envelope (the chain's envelopes are exactly
    that -- the carrier holds all the fast phase), and sub-pixel by
    construction, so the chief-ray offset is never quantised to the grid.
    Periodic: callers must check the beam still fits (``_check_tilt_fits``)."""
    e = np.asarray(env)
    if sx == 0.0 and sy == 0.0:
        return e.copy()
    ny, nx = e.shape[-2], e.shape[-1]
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    ramp = np.exp(-2j * np.pi * (fx[None, :] * sx + fy[:, None] * sy))
    out = np.fft.ifft2(np.fft.fft2(e) * ramp)
    return out.astype(e.dtype, copy=False) if np.iscomplexobj(e) else out


def _tilt_ramp(shape, dx, wavelength, L, M, x0, y0, sign):
    """``exp(sign*i*k*(L*(x-x0) + M*(y-y0)))`` on the centred grid, or ``None``
    when the tilt is identically zero (nothing to impose)."""
    if L == 0.0 and M == 0.0:
        return None
    ny, nx = int(shape[-2]), int(shape[-1])
    x = (np.arange(nx, dtype=np.float64) - nx / 2) * dx - float(x0)
    y = (np.arange(ny, dtype=np.float64) - ny / 2) * dx - float(y0)
    k = 2.0 * np.pi / wavelength
    return np.exp(sign * 1j * k * (L * x[None, :] + M * y[:, None]))


def _check_tilt_fits(env, dx, x_c, y_c, where):
    """Guard the chief-ray decentre against the element grid.

    The traced element sees the beam at its PHYSICAL transverse position, so
    the grid has to hold both the optical axis and the displaced beam.  Raise
    when the beam CENTRE plus one amplitude radius is already off the grid (the
    Fourier shift would wrap it round to the far edge -- a plausible-looking
    wrong answer, the exact failure class the roadmap's P3 is about), and warn
    while it is inside but within 2 radii of the edge."""
    n = int(np.shape(env)[-1])
    half = 0.5 * n * dx
    w = _envelope_amp_radius(env, dx, dx)
    reach = float(np.hypot(x_c, y_c)) + w
    if reach >= half:
        raise ValueError(
            f"propagate_traced_carrier_chain: the tilted carrier's chief ray "
            f"is {np.hypot(x_c, y_c) * 1e3:.4f} mm off axis at {where}, so "
            f"the beam (amplitude radius {w * 1e3:.4f} mm) does not fit on "
            f"the co-moving grid (half-extent {half * 1e3:.4f} mm = "
            f"N/2 * dx = {n // 2} * {dx * 1e6:.4f} um).  The element traces "
            f"the beam at its PHYSICAL position, so the grid must span the "
            f"axis AND the displaced beam: raise N (or dx) for this order, "
            f"shorten the gap, or run the order with its own decentred "
            f"prescription.")
    if reach + w >= half:
        import warnings
        warnings.warn(
            f"propagate_traced_carrier_chain: the tilted carrier's beam edge "
            f"reaches {reach * 1e3:.4f} mm at {where}, within one amplitude "
            f"radius of the co-moving grid's half-extent "
            f"({half * 1e3:.4f} mm).  The band-limited chief-ray shift is "
            f"periodic, so the skirt is wrapping round to the opposite edge; "
            f"raise N for this order.",
            RuntimeWarning, stacklevel=3)


# Chief-ray offset, in beam amplitude radii, above which a decentred traced
# hand-off costs measurable end-to-end accuracy (see _check_decentred_fit for
# the measured calibration).  Post-D7 the measured onset on the decentre-
# invariant conic stand-in is ~0.75 w (chain/oracle EE2 1.005 at 0.50 w, 0.977
# at 0.75 w, 0.983 at 1.0 w, 0.923 at 1.5 w); 0.5 keeps the default one step
# CONSERVATIVE of that.
_DECENTRE_FIT_FRAC_DEFAULT = 0.5

# niche C1 item 4 (2026-07-30): how much of a TILTED exact final leg's traced
# exit power may sit above its own grid Nyquist NA before
# ``on_tilt_exact_grid`` refuses.
#
# WHY IT IS A POWER FRACTION AND NOT AN NA COMPARISON.  D6 sized the guard from
# the chain's PARAXIAL ``na_exit = w_in / |R_out|``; the element measures the
# exit NA from the traced direction cosines and gets a bigger number (design
# 121, order (-4,-2), N=1024, dx0=2.0 um, rs 4, wf 4.0: paraxial 0.4053 vs
# measured 0.4780, 1.18x).  Re-pointing the guard at the measured NA alone
# REFUSES the shipped headline: at ``n_fine_cap`` 12288 the leg runs at
# ``dx_fine`` = 1.508 um, which carries NA 0.4343 -- below 0.4780 -- yet 12288
# and 16384 give IDENTICAL FWHM / EE3 / EE6 / EE12 (verified independently).
# The reason is that the measured NA is the MARGINAL ray at the e^-4 AMPLITUDE
# contour (r = 2w on a Gaussian, ~3e-4 of the power), so requiring Nyquist out
# to it demands full sampling of content that carries essentially nothing.
#
# MEASURED, on the real design 121 (post-DOE 6-group chain, order (-4,-2),
# N=1024, dx0 = 2.0 um, rs 4, ``window_factor`` 4.0, exact final leg, readout
# dx_out 0.05 um / N_out 1024, ``on_tilt_exact_grid='warn'`` so every row runs).
# ``frac`` is the |E_in|^2-weighted fraction of traced exit power above the
# grid's Nyquist NA, as reported by ``_exit_na_out``.  Every row measures
# paraxial na_exit **0.4052** against a MEASURED exit NA **0.4780**:
#
#   n_fine_cap  dx_fine um  NA carried   frac        FWHM um  EE3     EE6     EE12    s
#   16384        1.1311      0.5791      0.000e+00   4.3000   61.137  89.502  97.357  312
#   12288        1.5081      0.4343      7.967e-04   4.3000   61.135  89.501  97.356  171
#    8192        2.2622      0.2895      7.482e-03   4.3000   61.127  89.498  97.356   84
#
# Two things follow, and the second is the reason this is a POWER BUDGET rather
# than a fitted accuracy knee.
#
# (i) The shipped headline (12288) measures **7.967e-04**, so 1e-2 clears it by
#     12.5x -- the guard stays silent on the configuration the verifier proved
#     converged (12288 vs 16384 agree to 0.002 EE points here, reproducing that
#     result on this readout).  8192 -- which D6's PARAXIAL pre-check refuses,
#     and still does, unchanged -- reads 7.482e-03, so the post-check does not
#     fire there either: the two tests are complementary, and NOTHING accepted
#     today becomes refused.
#
# (ii) On BOTH geometries measured, the spot metrics are essentially INSENSITIVE
#      to the aliased fraction at any level reached: 121 moves 0.010 EE3 points
#      across 16384 -> 8192 (frac 0 -> 7.5e-3), and on the synthetic below
#      EE4 wanders 3.56-3.68 % with no monotone trend from frac 4.3e-7 to
#      1.7e-1.  So there IS no measured knee to fit, and claiming one would be
#      claiming a measurement that does not exist.  1e-2 is instead a stated
#      budget -- 1 % of the delivered power landing at the wrong radius -- in
#      the same spirit as the sibling ``readout_window_tol`` = 1e-4, whose
#      quantity (truncated beam power) is the same kind of thing.
#
# WHAT THE POST-CHECK NEWLY CATCHES is the paraxial pre-check's BLIND SPOT: a
# group whose measured exit NA far exceeds ``w_in/|R_out|``.  Measured on a
# synthetic N-BK7 singlet (R 8/-8 mm, 1.5 mm thick, aperture 3.2 mm,
# collimated w = 0.40 mm, tilt L = 0.02, N = 2048, dx0 = 2.0 um, rs 2,
# ``window_factor`` 4.0): paraxial na_exit **0.0520** against a measured
# **0.3407** (6.6x), so the pre-check is silent at EVERY grid, while frac runs
#
#   n_fine_cap  2048       1024       512      384      256      192      128
#   dx_fine um  1.5625     1.5625     3.125    4.167    6.250    8.333   12.500
#   NA carried  0.4192     0.4192     0.2096   0.1572   0.1048   0.0786   0.0524
#   frac        4.25e-07   4.25e-07   2.12e-3  3.25e-3  4.95e-3  2.11e-2  1.73e-1
#
# and the post-check fires from n_fine_cap 192 down -- a grid carrying 23 % of
# the measured exit NA, which no pre-check in the library could see.
#
# Set to ``np.inf`` to disable the measured post-check entirely and keep only
# D6's paraxial pre-check -- that is the fail-before switch the C1 tests use,
# not a supported configuration.
_TILT_EXACT_NA_POWER_TOL = 1e-2


def _check_decentred_fit(w, x_c, y_c, where, action, frac):
    """Report a traced hand-off whose chief ray is far enough off the ELEMENT
    grid centre that the run's per-order IMAGE metrics are a lower bound
    (niche D6, re-measured and re-scoped by niche D7 2026-07-29).

    What this guard is NOT, any more.  Its first revision blamed
    :func:`~lumenairy.elements.apply_real_lens_traced`'s off-centre ray fit and
    quoted a 3.7 -> 408 urad exit-slope curve from
    ``validation/repro_traced_carrier_121/decentred_fit_defect.py``.  **That
    curve was an artefact of that script's own FFT-derivative slope
    extraction**: re-run on a SYNTHETIC field whose exit-slope error is
    0.36 urad by construction, the same oracle reported 400.51 urad.  Measured
    aliasing-free (local wrapped phase differences against an order-12 fit +
    tight Newton), the element's own exit-slope error on design 121's last
    group is **1.28 urad on axis and 0.90 urad at 0.97 beam radii of decentre**
    post-D7 (7.16 urad pre-D7) -- 0.007 um of blur against a 3.5 um
    diffraction FWHM, i.e. the element's fit is no longer the limit.

    Do NOT read the 0.90 < 1.28 ordering as "decentre is now cheaper than
    being on axis".  Both rows are UNTILTED (the repro sweeps ``x0`` at
    ``tilt_L = 0``), and against the TILTED on-axis control -- the case design
    121's orders actually are -- the decentred figure sits 1.4x ABOVE, not
    below: 48.7 mrad of tilt on axis reads 0.64 urad against the same 0.90.
    The conclusion above is indifferent to which baseline is picked, because
    0.90 urad is 0.007 um of blur on a 3.5 um FWHM either way; only the
    ordering is, and an earlier revision stated it without the qualifier.

    What it IS.  A decentred hand-off still measurably costs image quality end
    to end, and the calibration is now taken on geometries whose truth is
    decentre-INVARIANT rather than through an aliasing measurement:

    * the ``K = -n^2`` conic stand-in (exact Fermat solution for the WHOLE
      collimated bundle, so every sub-aperture is stigmatic on axis and off).
      Chain / independent-oracle EE2 ratio by decentre:
      0.997 (0 w), 1.002 (0.25 w), 1.005 (0.50 w), 0.977 (0.75 w),
      0.983 (1.0 w), 0.923 (1.5 w).
    * design 121's post-DOE chain on the exact final leg, per order, against
      an independent skew-ray + Debye oracle that says every order is EQUALLY
      diffraction-limited (EE3 ~90.7 %, EE6 ~99.9 %):
      EE3 87.6 % at (0,0), 86.0 % at (-1,0), 68.1 % at (-4,0), 65.3 % at
      (-4,-2).

    So a per-order image metric taken through a decentred hand-off is still a
    LOWER BOUND on the design, not the design's performance -- but the residual
    is NOT the element's ray fit (0.90 urad), NOT the fine-retrace grid
    (``n_fine_cap`` 12288 vs 16384: EE3 65.26 vs 65.26), NOT Newton iterations
    (``newton_max_iters`` 12 vs 40: 65.26) and NOT the readout window
    (``window_factor`` 4/6/8: 65.26).  Use an independent ray trace for
    per-order image quality; the POWER bookkeeping is unaffected.
    """
    if action == 'ignore' or not (float(frac) > 0.0):
        return 0.0
    reach = float(np.hypot(x_c, y_c))
    if not (w > 0.0) or reach <= float(frac) * float(w):
        return reach / w if w > 0.0 else 0.0
    ratio = reach / float(w)
    _guard_dispose(
        action,
        f"propagate_traced_carrier_chain: at {where} the congruence's chief "
        f"ray sits {reach * 1e3:.4f} mm off the element grid centre = "
        f"{ratio:.3f} beam amplitude radii (w = {w * 1e3:.4f} mm), above "
        f"decentre_fit_frac={float(frac)}.  A decentred hand-off measurably "
        f"costs IMAGE quality end to end.  MEASURED on the K=-n^2 conic "
        f"stand-in, whose truth is decentre-INVARIANT (chain / independent "
        f"ray-trace + Kirchhoff oracle, EE2 ratio): 0.00 w -> 0.997; 0.25 w -> "
        f"1.002; 0.50 w -> 1.005; 0.75 w -> 0.977; 1.00 w -> 0.983; 1.50 w -> "
        f"0.923.  And on design 121's post-DOE chain, per order, against an "
        f"independent skew-ray + Debye oracle that says every order is EQUALLY "
        f"diffraction-limited (EE3 ~90.7 %): EE3 87.6 % on axis, 86.0 % at "
        f"(-1,0), 68.1 % at (-4,0), 65.3 % at (-4,-2).  THEREFORE: any "
        f"per-order spot size, Strehl or encircled energy this run reports is "
        f"a LOWER BOUND on the design, not the design's performance.  Use an "
        f"independent ray trace for per-order image quality; the chain's POWER "
        f"bookkeeping (per-order share, throughput, chief-ray landing) is "
        f"unaffected and still validated to 3e-4.  NOTE (niche D7, "
        f"2026-07-29): the residual is NOT apply_real_lens_traced's off-centre "
        f"ray fit any more -- that fit now carries 0.90 urad of exit slope at "
        f"0.97 w against 1.28 urad on axis UNTILTED (0.64 urad tilted, so the "
        f"decentred figure is not uniformly the smaller one; either way it is "
        f"0.007 um of blur against a 3.5 um "
        f"FWHM), and it is not the fine-retrace grid, the Newton iteration cap "
        f"or the readout window either (each moves EE3 by <= 0.01 point).  An "
        f"earlier revision of this message quoted a 3.7 -> 408 urad exit-slope "
        f"curve; that was an artefact of the repro script's FFT-derivative "
        f"slope extraction, which reports 400 urad on a synthetic field "
        f"built to be right to 0.36 urad by construction.  Pass "
        f"on_decentred_fit='error' to refuse instead, 'ignore' to silence, or "
        f"raise decentre_fit_frac if your design tolerates more.",
        stacklevel=3)
    return ratio


# ===========================================================================
# DOE chain entries (niche D4 -- roadmap
# ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27 P2)
# ===========================================================================
# Design 121 is not a bare relay: it is a crossed Dammann DOE between two
# refractive halves.  Until v5.32 the DOE could not be part of the design the
# chain sees at all -- ``DGRATING`` surfaces imported as flat optical surfaces
# with their parameters dropped -- so a consumer had to hand-build the
# grating, hand-split the chain at the DOE plane, and hand-fold the DOE's
# 51.539 mm gap into a neighbouring group's ``gap_before``.  That manual fold
# is the error-prone step: it is only correct for an UNDEFLECTED order,
# because a fold transports the chief ray over the whole folded distance at
# the PRE-DOE angle.
#
# A DOE entry in ``groups`` is therefore not new diffraction physics -- it is
# the ONE congruence's exact per-order bookkeeping, which the chain is already
# built for (niche D1's tilted carrier):
#
#   * ORDER (m, n) of a thin periodic screen multiplies the incident field by
#     ``c_mn exp(i (m Kx x + n Ky y))``, a pure linear ramp.  The grating
#     equation in DIRECTION COSINES is exact, not paraxial -- the screen
#     shifts the transverse wavevector by the grating vector, so
#     ``L -> L + m lambda / Px``, ``M -> M + n lambda / Py``.
#   * In this congruence's own CHIEF-RAY-TRACKING frame that ramp is entirely
#     absorbed into the carried tilt, so what is left acting on the ENVELOPE
#     is a COMPLEX CONSTANT: ``c_mn exp(i k (dL (x_c - ox) + dM (y_c - oy)))``,
#     where ``(x_c, y_c)`` is the chief ray at the DOE plane and ``(ox, oy)``
#     the grating's own origin.  That second factor is the phase reference: the
#     grating's ramp is referenced to ITS origin while the tracking frame's is
#     referenced to the chief ray, and the constant between them is exactly
#     what makes K orders recombine coherently at the image plane.
#   * The sphere ``R`` is untouched (a thin ramp carries no power).
#
# What the library then owns is the bookkeeping the roadmap asks for: the
# entry's ``gap_before`` is charged to the PRE-DOE angle and its ``gap_after``
# to the POST-DOE angle, while BOTH are folded into the neighbouring transport
# legs (or into ``final_distance``) automatically.  Both default to the values
# ``load_zemax_zmx`` measured from the .zmx, so
# ``{'doe': rx['diffractives'][k]}`` is a complete entry.
#
# Because the constant commutes with the transport, the DOE plane does not
# have to interrupt the carrier leg, and deliberately does not: the envelope
# crosses it inside ONE Sziklas-Siegman step.  Two measured reasons, in that
# order of importance:
#
#   1. It is what lets an order-0 DOE be BITWISE inert.  Splitting the leg is
#      *nearly* inert on an ordinary leg -- the co-moving magnification
#      telescopes exactly, ``(R+z1+z2)/R = [(R+z1)/R][(R+z1+z2)/(R+z1)]`` --
#      but the two routes still differ at the 1e-11 level through the extra
#      FFT pair (and, under carrier_reference='sphere', an extra
#      sphere<->parabola round trip at the split plane).  Deferring drives
#      that to 0, so writing the DOE into the design provably cannot move the
#      validated relay result rather than merely not moving it much.
#
#      "Bitwise" is a statement about the ARITHMETIC, and it is exact under
#      one stated condition: the entry's gaps must re-sum to the very same
#      float64 as the gap they replace.  The chain guarantees the reachable
#      half of that -- it accumulates the deferred gaps ONE LEG AT A TIME in
#      axial order (see the DOE branch), which is bit-identical to the
#      axial-order hand fold ``gb1 + ga1 + gb2 + ga2 + gap`` for ANY gaps.
#      What it cannot guarantee is a fold the consumer rounded differently
#      (a decimal literal typed for the total, say), and one ulp is not
#      small here: the traced pipeline's roundoff noise floor is ~1e-7
#      relative and a few ulp on a gap reach it (measured on a 37 mm gap:
#      +1 ulp -> 6.5e-11 on one relay, 1.4e-7 on another; +10 ulp -> 8.1e-8,
#      against a 3e-11 rad physical response to the same 6.9e-18 m).
#      That noise floor is also WHY bitwise is the property worth having:
#      it is the only statement that distinguishes "provably unchanged" from
#      "unchanged to the resolution this pipeline can report".
#   2. The telescoping FAILS when the split plane lands inside the near-focus
#      bridge zone of the carrier's own focus: there the bridge re-grids the
#      co-moving pitch, and the split route lands on a DIFFERENT grid from
#      the one-step route.  Measured (synthetic 58.5393 mm leg split at
#      51.5393 mm, pinned in tests/unit/test_niche_d4_dgrating.py): pitch
#      ratios 5.4x / 49x / 278x for a carrier focus at 51.0 / 51.6 /
#      51.55 mm, against 1.000000000000 everywhere the split plane is clear
#      of the focus -- including a focus INSIDE the leg but away from the
#      split (R = -3..-45 mm, field agreement 3e-12..6e-11).  Deferring makes
#      a DOE entry safe for a design whose screen sits near a carrier focus.
#
# NOTE, corrected 2026-07-28: design 121 is NOT such a design.  Its DOE sits
# in COLLIMATED space -- measured R = +703591.2 mm (703.6 m, diverging) at
# the pre-DOE group exit -- so one 58.5393 mm step and a 51.5393 + 7.0000 mm
# pair land on the SAME co-moving pitch (51.23386 um, ratio 1.000000) and
# agree to max|dE|/max|E| = 2.1e-11.  An earlier revision of this note cited
# a 5.5x pitch split for design 121's own leg; that number belongs to the
# near-focus corner in (2), not to this design.  For the 121 the operative
# reason is (1).

_DOE_SPEC_KEYS = frozenset({
    'type', 'period', 'order', 'angle_deg', 'origin', 'lines_per_um',
    'surf_num', 'comment', 'semi_diameter', 'gap_before', 'gap_after',
    'name', 'gap_media'})
_DOE_ENTRY_KEYS = frozenset({
    'doe', 'gap_before', 'gap_after', 'order', 'amplitude', 'name'})


def _doe_pair(value, what, where, fill):
    """``(x, y)`` from a scalar (``(value, fill)``), a 2-sequence, or None."""
    if value is None:
        return float(fill), float(fill)
    if isinstance(value, (list, tuple, np.ndarray)) and np.ndim(value) == 1:
        vals = [float(v) for v in np.asarray(value, dtype=np.float64).ravel()]
        if len(vals) != 2:
            raise ValueError(
                f"{where}: {what} must be a scalar (x axis only) or a "
                f"2-sequence (x, y); got {len(vals)} values.")
        return vals[0], vals[1]
    return float(value), float(fill)


def _doe_axes(angle_deg):
    """``(cos, sin)`` of the grating azimuth, EXACT on the quadrant multiples.

    A crossed DOE behind a Zemax +-90 deg z-roll is the common case (design
    121), and ``np.cos(np.deg2rad(270))`` is -1.8e-16 rather than 0 -- enough
    to make a pure-y grating report a spurious x deflection and drag an
    otherwise scalar congruence onto the tilted path."""
    a = float(angle_deg) % 360.0
    if a % 90.0 == 0.0:
        return ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))[int(a // 90)]
    r = np.deg2rad(a)
    return float(np.cos(r)), float(np.sin(r))


def _is_doe_entry(g):
    """True for a ``groups`` entry that is a DOE (has a ``'doe'`` key)."""
    return isinstance(g, dict) and 'doe' in g


def _normalise_doe_entry(g, gi, wavelength, fn):
    """Validate one DOE ``groups`` entry and reduce it to what the chain needs.

    Returns ``(name, gap_before, gap_after, dL, dM, amplitude, origin,
    (mx, my))``.  ``dL``/``dM`` are the exact grating-equation increments to
    the carried direction cosines; everything else is bookkeeping.  Unknown
    keys RAISE on both the entry and the spec -- a silently-dropped
    ``'orders'`` or ``'thickness'`` would move the DOE plane or the order with
    no symptom, which is the failure class this whole roadmap is about."""
    where = f"{fn}: groups[{gi}]"
    unknown = set(g) - _DOE_ENTRY_KEYS
    if unknown:
        raise ValueError(
            f"{where} is a DOE entry with unknown key(s) {sorted(unknown)!r}; "
            f"the accepted keys are {sorted(_DOE_ENTRY_KEYS)!r}.  (The axial "
            f"gaps are 'gap_before' / 'gap_after' -- both default to the "
            f"values the loader measured from the .zmx.)")
    spec = g['doe']
    if not isinstance(spec, dict):
        raise ValueError(
            f"{where}['doe'] must be a DOE spec dict -- e.g. "
            f"load_zemax_zmx(...)['diffractives'][k] -- got "
            f"{type(spec).__name__}.")
    unknown = set(spec) - _DOE_SPEC_KEYS
    if unknown:
        raise ValueError(
            f"{where}['doe'] has unknown key(s) {sorted(unknown)!r}; the "
            f"accepted keys are {sorted(_DOE_SPEC_KEYS)!r}.")
    if spec.get('type', 'grating') != 'grating':
        raise ValueError(
            f"{where}['doe']['type'] must be 'grating' (the only diffractive "
            f"the chain carries per order); got {spec['type']!r}.")
    if 'period' not in spec:
        raise ValueError(
            f"{where}['doe'] must supply 'period' -- the grating pitch in "
            f"METRES, a scalar (grating vector along the 'angle_deg' "
            f"azimuth) or (Px, Py) for a crossed grating.")
    px, py = _doe_pair(spec['period'], "'period'", where, np.inf)
    for _nm, _p in (('x', px), ('y', py)):
        if np.isnan(_p) or _p <= 0.0:
            raise ValueError(
                f"{where}['doe']['period'] ({_nm} axis) must be a positive "
                f"pitch in metres (or inf for 'no grating on this axis'); "
                f"got {_p!r}.  The grating vector's DIRECTION lives in "
                f"'angle_deg', not in the sign of the period.")
    mx, my = _doe_pair(g.get('order', spec.get('order', 0.0)), "'order'",
                       where, 0.0)
    for _nm, _m, _p in (('x', mx, px), ('y', my, py)):
        if not np.isfinite(_m):
            raise ValueError(
                f"{where}: DOE order ({_nm} axis) must be finite, got {_m!r}."
                f"  (Fractional orders are allowed -- an even-count Dammann "
                f"fan sits on the half-integer lattice.)")
        if _m != 0.0 and not np.isfinite(_p):
            raise ValueError(
                f"{where}: DOE order {_m!r} was asked for on the {_nm} axis, "
                f"but this DOE has no {_nm} period (period={_p!r}).  Give a "
                f"(Px, Py) 'period' for a crossed grating, or rotate the "
                f"grating with 'angle_deg'.")
    angle = float(spec.get('angle_deg', 0.0) or 0.0)
    if not np.isfinite(angle):
        raise ValueError(
            f"{where}['doe']['angle_deg'] must be finite, got {angle!r}.")
    ox, oy = _doe_pair(spec.get('origin', (0.0, 0.0)), "'origin'", where, 0.0)
    if not (np.isfinite(ox) and np.isfinite(oy)):
        raise ValueError(
            f"{where}['doe']['origin'] must be finite metres, got "
            f"{(ox, oy)!r}.")
    amp = complex(g.get('amplitude', 1.0))
    if not (np.isfinite(amp.real) and np.isfinite(amp.imag)):
        raise ValueError(
            f"{where}['amplitude'] must be finite, got {amp!r}.")
    gap_b = float(g.get('gap_before', spec.get('gap_before', 0.0)) or 0.0)
    gap_a = float(g.get('gap_after', spec.get('gap_after', 0.0)) or 0.0)
    for _nm, _v in (('gap_before', gap_b), ('gap_after', gap_a)):
        if not np.isfinite(_v):
            raise ValueError(
                f"{where}['{_nm}'] must be a finite distance in metres, got "
                f"{_v!r}.")
    # niche C2: the importer flags a DGRATING whose axial gap runs through
    # glass rather than air.  This chain transports gap_before / gap_after as
    # FREE-SPACE distances, so honouring such an entry would misplace the
    # grating by t - t/n per glass leg with no symptom in the output.  Refuse
    # unless the caller has overridden the offending gap explicitly, which is
    # exactly the documented "split the run at the substrate" remedy.
    _media = spec.get('gap_media') or ()
    _unhandled = [m for m in _media if m.get('gap') not in g]
    if _unhandled:
        _txt = ', '.join(
            f"{float(m.get('thickness', 0.0)) * 1e3:.4f} mm of "
            f"{m.get('glass')!r} in {m.get('gap')} (Zemax surface "
            f"{m.get('surf_num')})" for m in _unhandled)
        raise NotImplementedError(
            f"{where}['doe'] was imported from a Zemax file in which the "
            f"grating's axial gap does not lie in free space -- it traverses "
            f"{_txt}.  propagate_traced_carrier_chain transports gap_before "
            f"and gap_after through AIR, so this entry would place the grating "
            f"at the wrong optical distance (error t - t/n per glass leg; 1.0 "
            f"mm for a 3 mm N-BK7 substrate).  A grating ruled on a substrate "
            f"is not supported by the drop-in import: split the run at the "
            f"substrate, or override the offending gap on THIS entry (pass "
            f"{sorted({m.get('gap') for m in _unhandled})} yourself) once you "
            f"have reduced it to a free-space distance.")
    # the grating equation, in direction cosines and therefore EXACT
    ax = 0.0 if not np.isfinite(px) else mx * float(wavelength) / px
    ay = 0.0 if not np.isfinite(py) else my * float(wavelength) / py
    c, s = _doe_axes(angle)
    dL = ax * c - ay * s
    dM = ax * s + ay * c
    name = str(g.get('name', spec.get('name')
                     or spec.get('comment')
                     or (f"DOE S{spec['surf_num']}" if 'surf_num' in spec
                         else f'doe{gi}')) or f'doe{gi}')
    return name, gap_b, gap_a, dL, dM, amp, (ox, oy), (mx, my)


def _chain_envelope_stats(env, dx):
    """(1/e^2 radius, total power) of an envelope on the centred grid."""
    inten = np.abs(np.asarray(env)) ** 2
    tot = float(inten.sum())
    if not (tot > 0.0):
        return 0.0, 0.0
    n = inten.shape[-1]
    x = (np.arange(n, dtype=np.float64) - n / 2) * dx
    mx = inten.sum(axis=0)
    cx = float((mx * x).sum() / tot)
    vx = float((mx * (x - cx) ** 2).sum() / tot)
    return 2.0 * np.sqrt(max(vx, 0.0)), tot * dx * dx


def _fine_trace_group_exit(env, R_in, cur_dx, presc, wavelength, ray_subsample,
                           n_workers, call_kw, R_out, na_exit,
                           window_factor=5.0, n_fine_cap=16384,
                           max_fine_launch_points=4096,
                           sphere_reference=False, ram_budget=None,
                           on_ram_cap='warn', on_rs_fine_clamp='warn',
                           centre=(0.0, 0.0), tilt=(0.0, 0.0),
                           on_tilt_exact_grid='error',
                           on_decentred_fit='warn',
                           decentre_fit_frac=_DECENTRE_FIT_FRAC_DEFAULT,
                           na_diag_out=None):
    """Re-trace a HIGH-NA group on a grid that Nyquist-samples its EXIT sphere
    (R9).  The co-moving grid is sized for the group ENTRANCE curvature, so on a
    strongly-focusing group the (much steeper) exit wavefront ALIASES -- the
    exit-field build ``amp*exp(i*OPL)`` wraps > pi/pixel, and the coarse Newton/
    poly OPL fit aliases high-order aberration into defocus (the design-121
    per-group F2 residual).  Fix: upsample the SMOOTH incoming envelope (Fourier
    zero-pad) onto a grid with ``dx_fine ~ lambda/(3*NA_exit)``, reconstruct, and
    trace there.  Returns ``(E_exit_fine, dx_fine)`` -- the full exit field, now
    properly sampled, ready for :func:`carrier_referenced_exact_focus_readout`.

    Audit AUDIT_TRACED_CHAIN_DX_SCALING_2026_07_22:

    * F-A -- when the chain grid is large enough that ``n_crop`` (the window
      expressed in ``cur_dx`` pixels) exceeds ``n_fine_cap``, ``n_fine`` gets
      clamped BELOW ``n_crop`` here.  :func:`_fourier_upsample_crop` now
      handles that direction correctly (band-limited spectral truncation)
      instead of silently returning the raw, wrong-pitch ``n_crop``-sized
      crop -- the fix closes an energy-non-conservation + core-blur bug that
      triggered exactly when ``n_crop > n_fine_cap``.
    * F-C -- ``ray_subsample`` is rescaled on entry so the retrace on the
      (generally much finer) ``dx_fine`` grid keeps the CHAIN's physical ray
      pitch (``ray_subsample * cur_dx``), rather than reinterpreting the same
      integer in ``dx_fine`` pixel units (which silently densifies the
      Newton/Cheb ray-fit grid by the ``cur_dx / dx_fine`` factor and can
      exhaust memory -- observed (28, 20151, 20151) float64 = 84.7 GiB at
      ``ray_subsample=1`` on a production-scale chain).
      ``max_fine_launch_points`` is an independent backstop cap on the
      resulting fit-grid size, active only if the pitch-preserving
      ``ray_subsample`` would still exceed it.
    * F-D -- when ``n_fine_cap`` forces ``dx_fine`` coarser than the exit
      sphere's Nyquist pitch (``lambda / (2 * NA)``), a warning is emitted:
      the retrace still runs (``bandlimit=True`` in the downstream readout
      masks the aliased corner) but silently discards outer-NA content.

    ``window_factor`` is applied TWICE on this leg: here (cropping the retrace
    grid to ``window_factor`` ENTRANCE-beam radii) and again inside
    :func:`carrier_referenced_exact_focus_readout` (cropping to
    ``window_factor`` EXIT-beam radii, re-measured on the field this leg
    returns).  Each crop is the right window for its own beam, but they
    COMPOUND: the second cut lands at ``window_factor * rho`` beam radii with
    ``rho`` the truncated-second-moment shrink factor -- measured 0.880 at
    ``window_factor=2`` (worth 8.6 EE3 points on the design-121-class case),
    0.987 at 3, and an exact no-op from 4 upward.  See the ``window_factor``
    entry of :func:`carrier_referenced_exact_focus_readout` for the formula and
    the measured table; the shipping default (7.0) is in the no-op regime.

    ``ray_subsample`` pitch preservation has one degenerate corner: when the
    memory/Nyquist-capped ``dx_fine`` is itself COARSER than the chain's
    physical ray pitch ``ray_subsample * cur_dx``, the rescale rounds to 0 and
    is clamped to ``rs_fine = 1``, so the retrace's physical ray pitch is
    ``dx_fine`` -- coarser than the chain's (measured 5.25x at the N=28672 /
    ``n_fine_cap=16384`` design-121 condition: chain pitch 0.286 um vs
    ``dx_fine`` 1.5 um).  Nothing finer is representable on that grid, so the
    clamp is forced rather than wrong, but the F-C contract ("keeps the CHAIN's
    physical ray pitch") does not hold there, so a ``RuntimeWarning`` fires
    naming BOTH pitches and the remedy (raise ``n_fine_cap`` / shrink
    ``window_factor``) whenever the clamp binds (S12) -- previously this was a
    silent, docstring-only contract gap that only the F-D warning hinted at
    (and F-D names the symptom, ``dx_fine`` vs the exit Nyquist pitch, not the
    ray lattice).  ``on_rs_fine_clamp='error'`` (D3 / roadmap P5) is the
    opt-in STRICT mode for that corner: a production run that needs the F-C
    pitch-preservation contract to actually hold raises there instead of
    accepting a coarser final-leg ray lattice than the rest of the chain.

    Interaction note (F-C, reviewed 2026-07-22): the pitch-preserving
    ``ray_subsample`` rescale reduces the ray-fit density on THIS leg
    relative to the pre-fix behaviour (which accidentally over-sampled by
    reusing the chain-level integer in the finer grid's pixel units), so it
    reduces -- but for a realistic beam/aperture ratio does not eliminate --
    the safety margin against ``apply_real_lens_traced``'s own
    ``min_coarse_samples_per_aperture`` aliasing floor (default 32,
    ``on_undersample='error'`` by default).  Quantitatively: tripping it
    requires the beam radius to be sampled by fewer than
    ``16 * ray_subsample`` pixels on the co-moving grid for a
    beam-filling aperture -- e.g. < 64 px at the default
    ``ray_subsample=4`` (the design-121 R9 case samples it at 213 px, a
    3.3x margin).  Any chain coarse enough to trip this was ALREADY
    aliasing the Newton/Cheb fit silently pre-fix; post-fix it fails
    loudly instead, which is strictly safer, but if you hit it, pass
    ``traced_kwargs={'on_undersample': 'warn'}`` (or ``'silent'``) at the
    ``propagate_traced_carrier_chain`` call rather than assuming a bug here.

    TILT AWARENESS (niche D6, roadmap
    ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27 P1a).  ``centre`` is the
    congruence's CHIEF RAY at this group's entrance (metres, absolute /
    optical-axis coordinates) and ``tilt`` its direction cosines ``(L, M)``;
    both default to the on-axis case, which short-circuits to the
    byte-identical historical path.  ``env`` arrives in the chain's
    CHIEF-RAY-TRACKING frame (the beam sits on the grid centre), so the crop
    above is ALREADY chief-ray-centred in absolute terms -- what the tilted
    case adds is that the ELEMENT must see the beam at its true transverse
    position, and ``apply_real_lens_traced`` builds its grid symmetrically
    about the optical axis.  The retrace window is therefore widened to
    ``2*max(|x_c|, |y_c|) + window_factor*w`` so ONE axis-centred grid holds
    both the axis and the displaced beam, the envelope is band-limit-shifted
    onto that grid, and the reconstruction uses the DECENTRED sphere + tilt
    ramp with a matching :class:`~lumenairy.TiltedCarrier` -- exactly the
    hand-off :func:`propagate_traced_carrier_chain` already performs on its
    coarse legs.  Measured on design 121's extreme order (-4,-2) at N=1024:
    the chief ray is 3.373 mm off axis against an entrance beam radius
    3.126 mm, so the window grows 12.50 -> 18.54 mm (1.48x) and the fine
    grid has to grow with it to keep the same ``dx_fine``.

    That growth is the tilted leg's whole cost, and it is REFUSED rather
    than absorbed when it cannot be paid: if the capped ``dx_fine`` ends up
    coarser than the exit sphere's Nyquist pitch, the on-axis path warns
    (F-D above) but the tilted path routes through ``on_tilt_exact_grid``
    (default ``'error'``), naming the chief-ray offset, the beam radius, the
    window, the required ``n_fine`` and the ``n_fine_cap`` that bound it.
    Silently returning a leg whose outer NA has been discarded is exactly
    the plausible-looking wrong answer this campaign exists to prevent.

    ``on_tilt_exact_grid`` fires from TWO tests (niche C1 item 4), because D6's
    single test measured the wrong NA:

    * a cheap PRE-check, before any tracing, from the chain's PARAXIAL
      ``na_exit = w_in / |R_out|`` -- unchanged from D6, so nothing that
      refused before stops refusing;
    * the DECISIVE POST-check, from the exit NA
      :func:`~lumenairy.elements.apply_real_lens_traced` actually MEASURED on
      the grid it just used (``_exit_na_out``).  The paraxial estimate is the
      smaller of the two -- design 121's order (-4,-2) reads 0.4053 against a
      measured 0.4780 -- so at the shipped ``n_fine_cap`` = 12288 the
      pre-check passed a leg the element itself warned was under-sampled.  The
      post-check's criterion is the |E_in|^2-weighted fraction of traced exit
      power above grid Nyquist, against ``_TILT_EXACT_NA_POWER_TOL``; see that
      constant for why a bare NA comparison would refuse a configuration the
      measurements prove converged.

    ``na_diag_out`` (dict, optional) receives that measurement, so the caller
    can report it -- ``propagate_traced_carrier_chain`` puts it on the final
    stage as ``na_exit_measured`` / ``na_grid_nyquist`` /
    ``exit_power_above_nyquist``.
    """
    from ..elements import apply_real_lens_traced
    N = env.shape[-1]
    w = _envelope_amp_radius(env, cur_dx, cur_dx)
    na = min(max(na_exit, 0.02), 0.95)
    dx_fine = wavelength / (3.0 * na)
    x_c, y_c = float(centre[0]), float(centre[1])
    tL, tM = float(tilt[0]), float(tilt[1])
    _tilted = bool(x_c or y_c or tL or tM)
    if not _tilted:
        win = min(window_factor * w, N * cur_dx) if w > 0 else N * cur_dx
    else:
        # niche D6: ONE axis-centred window holding the optical axis AND the
        # chief-ray-displaced beam.  ``w`` is measured in the TRACKING frame,
        # where the beam is centred, so it is the beam's own radius.
        _reach = max(abs(x_c), abs(y_c))
        _win_beam = window_factor * w if w > 0 else N * cur_dx
        _win_want = 2.0 * _reach + _win_beam
        win = min(_win_want, N * cur_dx)
        if _win_want > N * cur_dx * (1.0 + 1e-12):
            raise ValueError(
                f"_fine_trace_group_exit: the EXACT high-NA final leg needs "
                f"an axis-centred retrace window of {_win_want * 1e3:.4f} mm "
                f"(= 2 x chief-ray offset {_reach * 1e3:.4f} mm + "
                f"window_factor={window_factor} x beam radius "
                f"{w * 1e3:.4f} mm) to hold BOTH the optical axis and the "
                f"tilted congruence's beam, but the co-moving grid spans only "
                f"{N * cur_dx * 1e3:.4f} mm (N={N} x dx="
                f"{cur_dx * 1e6:.4f} um).  The element traces the beam at its "
                f"PHYSICAL position, so the grid must span both.  Raise the "
                f"chain's N, shrink window_factor via the focus_readout dict, "
                f"or run this order with final_leg='paraxial'.")
    n_crop = int(2 * round((win / cur_dx) / 2))
    n_crop = int(min(max(n_crop, 2), N))
    win = n_crop * cur_dx
    n_fine_req = int(2 ** int(np.ceil(np.log2(max(win / dx_fine, n_crop)))))
    n_fine = int(min(n_fine_req, n_fine_cap))
    # P2 memory budget (audit AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24 §4):
    # ``n_fine_cap`` is a COUNT cap the caller has to know to set; this is the
    # box's actual RAM ceiling, applied on top of it so a large window degrades
    # (announced) instead of OOM-ing the retrace.
    n_fine = _memory_bounded_n_fine(
        n_fine, '_fine_trace_group_exit', ram_budget=ram_budget,
        window=win, nyquist_dx=wavelength / (2.0 * na),
        on_ram_cap=on_ram_cap)
    dx_fine = win / n_fine

    # F-D: n_fine_cap can force dx_fine coarser than the exit sphere's
    # Nyquist pitch.  The retrace still runs (the downstream exact readout's
    # bandlimit=True masks the aliased corner) but silently discards
    # outer-NA content -- warn so that's visible instead of silent.
    nyquist_dx = wavelength / (2.0 * na)
    if dx_fine > nyquist_dx and _tilted:
        # niche D6: the SAME shortfall the F-D warning covers, but on the
        # tilted leg it is the tilt itself that bought it (the window had to
        # grow by 2*chief-ray offset), so it is reported as a refusal naming
        # the tilt-specific quantities rather than as a warning the caller
        # would have to attribute.  Downgrade with on_tilt_exact_grid='warn'
        # to accept a leg whose outer NA is discarded.
        _n_need = int(2 ** int(np.ceil(np.log2(max(win / nyquist_dx, 2.0)))))
        _guard_dispose(
            on_tilt_exact_grid,
            f"_fine_trace_group_exit: the EXACT high-NA final leg cannot be "
            f"sampled for this TILTED congruence.  Holding both the optical "
            f"axis and a chief ray {max(abs(x_c), abs(y_c)) * 1e3:.4f} mm off "
            f"it needs an axis-centred window of {win * 1e3:.4f} mm "
            f"(entrance beam radius {w * 1e3:.4f} mm, window_factor="
            f"{window_factor}) -- {win / max(window_factor * w, 1e-300):.3f}x "
            f"the on-axis window -- so merely NYQUIST-sampling the exit sphere "
            f"at NA={na:.4f} needs n_fine={_n_need} (the 3x-oversampled target "
            f"is {n_fine_req}), but n_fine_cap={n_fine_cap} (and the RAM "
            f"budget) allow only {n_fine}, giving dx_fine="
            f"{dx_fine * 1e6:.4f} um against the exit Nyquist pitch "
            f"lambda/(2*NA)={nyquist_dx * 1e6:.4f} um.  The retrace would run "
            f"but SILENTLY DISCARD every spatial frequency above "
            f"NA={wavelength / (2.0 * dx_fine):.4f}.  Remedies, in order: "
            f"raise n_fine_cap to {_n_need} in the focus_readout dict "
            f"(cost ~{(_n_need / max(n_fine, 1)) ** 2:.1f}x this leg's memory "
            f"and time); shrink window_factor; or pass final_leg='paraxial' "
            f"for this order and accept the paraxial readout.  Pass "
            f"on_tilt_exact_grid='warn' to run anyway with the outer NA "
            f"discarded, 'ignore' to silence this entirely.",
            stacklevel=2)
    elif dx_fine > nyquist_dx:
        import warnings
        warnings.warn(
            f"_fine_trace_group_exit: n_fine_cap={n_fine_cap} forces "
            f"dx_fine={dx_fine * 1e6:.3f} um, coarser than the exit "
            f"sphere's Nyquist pitch lambda/(2*NA)={nyquist_dx * 1e6:.3f} um "
            f"at NA={na:.3f}.  The retrace will still run but silently "
            f"discards outer-NA content.  Raise n_fine_cap or shrink "
            f"window_factor (currently {window_factor}) via the "
            f"focus_readout dict if the full NA is needed.",
            RuntimeWarning, stacklevel=2)

    if not _tilted:
        env_f = _fourier_upsample_crop(env, n_crop, n_fine)
        E_full = carrier_referenced_reconstruct(env_f, R_in, wavelength,
                                                dx_fine)
        if sphere_reference:
            # The stored envelope is EXACT-SPHERE-referenced (chain
            # ``carrier_reference='sphere'``): convert the parabola-referenced
            # reconstruction so the element receives the physical wavefront.
            # No exit-side conversion here -- the exit field goes straight to
            # carrier_referenced_exact_focus_readout, which references the
            # exact sphere itself.
            _cf = _sphere_parab_conversion(np.shape(E_full), dx_fine,
                                           wavelength, R_in, +1, w_beam=w)
            if _cf is not None:
                E_full = np.asarray(E_full) * _cf
        _carrier_arg = R_in
    else:
        # niche D6 -- leave the chief-ray-tracking frame for the element, on
        # the FINE grid, exactly as the chain's coarse hand-off does on the
        # co-moving one: band-limit-shift the (smooth) envelope to its true
        # transverse position, reconstruct against the DECENTRED sphere, and
        # impose the tilt ramp referenced to the same chief ray.
        _check_tilt_fits(env, cur_dx, x_c, y_c,
                         'the EXACT final leg (fine retrace)')
        _check_decentred_fit(w, x_c, y_c,
                             'the EXACT final leg (fine retrace)',
                             on_decentred_fit, decentre_fit_frac)
        env_f = _fourier_upsample_crop(
            _shift_envelope(env, x_c, y_c, cur_dx), n_crop, n_fine)
        _sh = (n_fine, n_fine)
        _ph = _radial_carrier_phase(_sh, dx_fine, dx_fine, wavelength, R_in,
                                    +1, centre=(x_c, y_c)) \
            if np.isfinite(R_in) else None
        E_full = env_f if _ph is None else np.asarray(env_f) * _ph
        if sphere_reference:
            _cf = _sphere_parab_conversion(_sh, dx_fine, wavelength, R_in, +1,
                                           w_beam=w, centre=(x_c, y_c))
            if _cf is not None:
                E_full = np.asarray(E_full) * _cf
        _rp = _tilt_ramp(_sh, dx_fine, wavelength, tL, tM, x_c, y_c, +1)
        if _rp is not None:
            E_full = np.asarray(E_full) * _rp
        from ..elements._lens_traced import TiltedCarrier as _TC
        _carrier_arg = _TC(R_in, tL, tM, x_c, y_c)

    # F-C: preserve the CHAIN's physical ray pitch (ray_subsample * cur_dx)
    # on the fine retrace grid, rather than reinterpreting the same integer
    # ray_subsample in dx_fine pixel units.
    _rs_want = float(ray_subsample) * cur_dx / dx_fine
    _rs_round = int(round(_rs_want))
    rs_fine = max(1, _rs_round)
    if _rs_round < 1:
        # S12: the clamp BINDS -- dx_fine is coarser than the chain's physical
        # ray pitch, so no integer subsample on this grid can reproduce it and
        # the F-C contract ("keeps the CHAIN's physical ray pitch") silently
        # does not hold.  Forced by the grid, not wrong, but the retrace then
        # fits the traced OPL on a COARSER ray lattice than the chain used, so
        # say so rather than leave it to the docstring.  D3 / roadmap P5:
        # ``on_rs_fine_clamp='error'`` promotes it to a hard failure for a
        # production run that needs the F-C contract to hold.
        _guard_dispose(
            on_rs_fine_clamp,
            f"_fine_trace_group_exit: the chain's physical ray pitch "
            f"({float(ray_subsample) * cur_dx * 1e6:.4f} um = ray_subsample="
            f"{ray_subsample} x cur_dx={cur_dx * 1e6:.4f} um) CANNOT be "
            f"preserved on this retrace grid: dx_fine={dx_fine * 1e6:.4f} um "
            f"is already coarser than it, so the pitch-preserving rescale "
            f"({_rs_want:.4f}) rounds below 1 and is clamped to "
            f"ray_subsample=1, giving a retrace ray pitch of "
            f"{dx_fine * 1e6:.4f} um -- {dx_fine / max(float(ray_subsample) * cur_dx, 1e-300):.2f}x "
            f"the chain's.  The traced-OPL fit on this final leg therefore "
            f"runs on a coarser ray lattice than the rest of the chain.  "
            f"Remedy: raise n_fine_cap (currently {n_fine_cap}) or shrink "
            f"window_factor (currently {window_factor}) via the focus_readout "
            f"dict so dx_fine falls at or below the chain's ray pitch.  This "
            f"is warn-only by default; pass on_rs_fine_clamp='error' for the "
            f"STRICT mode that refuses the degenerate corner, or 'ignore' to "
            f"silence it.",
            stacklevel=2)
    # Independent backstop: cap the resulting Newton/Cheb ray-fit grid size
    # even if the pitch-preserving rs_fine would still be too dense (e.g.
    # the chain-level ray_subsample was itself already very fine relative to
    # this leg's physically large window).  Only ever RAISES rs_fine above
    # the pitch-preserving value, never lowers it.
    launch_radius_est = 0.5 * win
    if isinstance(presc, dict):
        ap = presc.get('aperture_diameter')
        if ap is not None and np.isfinite(ap) and ap > 0:
            launch_radius_est = 0.5 * float(ap) * 1.50
    n_launch_est = max(8, int(2.0 * launch_radius_est / (dx_fine * rs_fine)))
    if n_launch_est > max_fine_launch_points:
        rs_needed = int(np.ceil(
            2.0 * launch_radius_est / (dx_fine * max_fine_launch_points)))
        if rs_needed > rs_fine:
            import warnings
            warnings.warn(
                f"_fine_trace_group_exit: the physical-pitch-preserving "
                f"ray_subsample ({rs_fine}) would still give an estimated "
                f"~{n_launch_est}x{n_launch_est} ray-fit grid on the fine "
                f"retrace; capping at ~{max_fine_launch_points}x"
                f"{max_fine_launch_points} by raising ray_subsample to "
                f"{rs_needed} (chain ray_subsample={ray_subsample}).  Fit "
                f"quality on this leg is reduced; lower window_factor / "
                f"na_exact_threshold, or raise max_fine_launch_points, if "
                f"you need finer sampling and can afford the memory.",
                RuntimeWarning, stacklevel=2)
            rs_fine = rs_needed

    _na_diag: dict = {}
    E_exit = apply_real_lens_traced(
        E_full, prescription=presc, wavelength=wavelength, dx=dx_fine,
        carrier=_carrier_arg, ray_subsample=rs_fine, n_workers=n_workers,
        _exit_na_out=_na_diag, **call_kw)
    # ---- niche C1 item 4: the DECISIVE tilted-leg sampling test -------------
    # The pre-check above is sized from ``na_exit``, which is the CHAIN's
    # PARAXIAL exit NA ``w_in / |R_out|``.  That is not the element's own exit
    # NA, and the gap is not small: on design 121's order (-4,-2) the chain
    # reads 0.4053 while ``apply_real_lens_traced`` MEASURES 0.4780 from the
    # traced exit direction cosines.  At the shipped headline
    # (``n_fine_cap`` 12288) ``dx_fine`` = 1.508 um passes the paraxial test
    # (its Nyquist pitch 1.616 um) while the element itself warns "NA_exit=
    # 0.4780 ... needs dx <= 1.37 um but the grid has dx = 1.51 um" -- i.e. the
    # guard whose whole job is refusing an under-sampled tilted leg stayed
    # SILENT on a leg the element calls under-sampled.
    #
    # It cannot simply be re-pointed at the measured NA, because that
    # measurement is the MARGINAL ray at the e^-4 AMPLITUDE contour (r = 2w for
    # a Gaussian), whose content carries ~3e-4 of the power -- and the verifier
    # PROVED the 12288 result converged (12288 vs 16384: identical FWHM / EE3 /
    # EE6 / EE12).  So the criterion is the discarded exit POWER, weighted by
    # |E_in|^2 over the traced rays and measured by the element on the very
    # grid it just used.  See ``_TILT_EXACT_NA_POWER_TOL`` for the calibration.
    if na_diag_out is not None and _na_diag:
        na_diag_out.update(_na_diag)
    if _tilted and float(_na_diag.get('na_exit') or 0.0) > 0.0:
        _frac = float(_na_diag.get('power_frac_above_nyquist', 0.0))
        if _frac > _TILT_EXACT_NA_POWER_TOL:
            _guard_dispose(
                on_tilt_exact_grid,
                f"_fine_trace_group_exit: the EXACT high-NA final leg for this "
                f"TILTED congruence ran on a grid that cannot carry its own "
                f"MEASURED exit NA.  The element traced the exit direction "
                f"cosines out to NA={float(_na_diag['na_exit']):.4f} (the "
                f"chain's paraxial estimate w_in/|R_out| = {na_exit:.4f} is "
                f"the number that SIZED the grid, and it is the smaller of "
                f"the two), while dx_fine={dx_fine * 1e6:.4f} um carries only "
                f"NA={float(_na_diag['na_nyquist']):.4f} -- so "
                f"{_frac * 100:.3f} % of the traced exit power sits above grid "
                f"Nyquist and ALIASES (tolerance "
                f"{_TILT_EXACT_NA_POWER_TOL * 100:.3g} %).  This congruence is "
                f"tilted at (L, M) = ({tL:+.5f}, {tM:+.5f}) with its chief ray "
                f"{max(abs(x_c), abs(y_c)) * 1e3:.4f} mm off axis at this "
                f"group's entrance, so the axis-centred retrace window is "
                f"{win * 1e3:.4f} mm "
                f"({win / max(window_factor * w, 1e-300):.3f}x the on-axis "
                f"window), and n_fine_cap={n_fine_cap} (with the RAM budget) "
                f"allowed only n_fine={n_fine}.  Remedies, in order: raise "
                f"n_fine_cap to >= "
                f"{int(2 ** int(np.ceil(np.log2(max(win * 2.0 * float(_na_diag['na_exit']) / wavelength, 2.0)))))} "
                f"in the focus_readout dict; shrink window_factor; or pass "
                f"final_leg='paraxial' for this order.  Pass "
                f"on_tilt_exact_grid='warn' to accept the aliased outer NA, "
                f"'ignore' to silence this entirely.",
                stacklevel=2)
    return np.asarray(E_exit), float(dx_fine)


def _chain_result_metrics(res):
    """Grid-comparable scalar metrics of a chain result, for the dx self-check.

    After a focus readout both runs land on the SAME ``(dx_out, N_out)`` grid,
    so the focal metrics are directly comparable: window power, peak intensity
    and the 50%-encircled-energy radius about the peak.  Without a readout the
    landing grids differ (co-moving pitch), so the comparable quantities are the
    physical ones: envelope 1/e^2 radius, power and the carrier radius."""
    E = np.asarray(res.field)
    dxo = float(res.dx)
    I = np.abs(E) ** 2
    tot = float(I.sum())
    if not np.isfinite(tot) or tot <= 0.0:
        return {}
    if res.R is not None:
        w_env, power = _chain_envelope_stats(E, dxo)
        out = {'w_env': w_env, 'power': power}
        if np.isfinite(res.R):
            out['R'] = float(res.R)
        return out
    iy, ix = np.unravel_index(int(np.argmax(I)), I.shape)
    yy = np.arange(I.shape[0], dtype=np.float64) - float(iy)
    xx = np.arange(I.shape[1], dtype=np.float64) - float(ix)
    r_px = np.sqrt(xx[None, :] ** 2 + yy[:, None] ** 2)
    ring = np.cumsum(np.bincount(r_px.astype(np.int64).ravel(),
                                 weights=I.ravel()))
    half = 0.5 * float(ring[-1])
    j = int(np.searchsorted(ring, half))
    j = min(j, ring.size - 1)
    c0 = float(ring[j - 1]) if j > 0 else 0.0
    frac = (half - c0) / max(float(ring[j]) - c0, 1e-300)
    return {'power': tot * dxo * dxo, 'peak': float(I[iy, ix]),
            'r50': float((j + min(max(frac, 0.0), 1.0)) * dxo)}


def _run_chain_dx_self_check(kw, res, tol):
    """Re-run the chain at ``dx/sqrt(2)`` (extent-preserving) and warn when the
    focal metrics move by more than ``tol`` (relative).  The cheap "is this
    number dx-stable" flag from the production-readiness plan (P2)."""
    from ..backend import to_numpy
    E_in = np.asarray(to_numpy(kw['E_in']))
    N = int(E_in.shape[-1])
    N2 = int(2 * round(N * np.sqrt(2.0) / 2.0))
    if N2 <= N:
        N2 = N + 2
    dx0 = float(kw['dx'])
    dx2 = dx0 * N / N2                      # same physical extent N*dx
    m1 = _chain_result_metrics(res)
    if not m1:
        return
    kw2 = dict(kw)
    kw2['E_in'] = _fourier_upsample_crop(E_in, N, N2)
    kw2['dx'] = dx2
    res2 = propagate_traced_carrier_chain(**kw2)
    m2 = _chain_result_metrics(res2)
    bad = []
    for key in sorted(set(m1) & set(m2)):
        a, b = float(m1[key]), float(m2[key])
        scale = max(abs(a), abs(b), 1e-300)
        rel = abs(a - b) / scale
        if rel > tol:
            bad.append(f'{key}: {a:.6g} (dx={dx0 * 1e6:.4f} um) vs {b:.6g} '
                       f'(dx={dx2 * 1e6:.4f} um), {100.0 * rel:.1f}% apart')
    from .._logging import get_logger
    get_logger(__name__).info(
        "propagate_traced_carrier_chain self_check='dx': N %d -> %d, "
        "dx %.6g -> %.6g m, metrics %r vs %r", N, N2, dx0, dx2, m1, m2)
    if bad:
        import warnings
        warnings.warn(
            f"propagate_traced_carrier_chain self_check='dx': the result is "
            f"NOT dx-STABLE.  Refining the grid from N={N} (dx="
            f"{dx0 * 1e6:.4f} um) to N={N2} (dx={dx2 * 1e6:.4f} um) at the same "
            f"physical extent moved: " + '; '.join(bad) +
            f" (tolerance {100.0 * tol:.1f}%).  A metric that moves with dx is "
            f"not converged, so treat the returned numbers as indicative only "
            f"-- refine until they plateau (or reduce ray_subsample / raise "
            f"the focus_readout window_factor / n_fine_cap, which are the other "
            f"resolution axes) before quoting absolute EE / FWHM.",
            RuntimeWarning, stacklevel=3)


def propagate_traced_carrier_chain(
    E_in: np.ndarray,
    groups,
    wavelength: float,
    dx: float,
    *,
    r_in: float = np.inf,
    ray_subsample: int = 4,
    n_workers: Optional[int] = None,
    traced_kwargs: Optional[dict] = None,
    final_distance: float = 0.0,
    focus_readout: Optional[dict] = None,
    final_leg: str = 'auto',
    na_exact_threshold: float = 0.15,
    carrier_reference: str = 'sphere',
    self_check: Optional[str] = None,
    self_check_tol: float = 0.05,
    on_multi_congruence: str = 'warn',
    multi_congruence_threshold: float = _MULTI_CONGRUENCE_MV_THRESH,
    on_na_proximity: str = 'warn',
    na_proximity_frac: float = 0.20,
    on_ram_cap: str = 'warn',
    on_rs_fine_clamp: str = 'warn',
    on_tilt_exact_grid: str = 'error',
    on_decentred_fit: str = 'warn',
    decentre_fit_frac: float = _DECENTRE_FIT_FRAC_DEFAULT,
) -> TracedCarrierChainResult:
    """Propagate a beam ENVELOPE through a chain of real (traced) lens groups on
    a co-moving carrier-referenced grid (audit F4.1).

    Packages the per-group hand-off pattern -- analytic carrier leg ->
    reconstruct at the group front vertex -> ``apply_real_lens_traced(carrier=
    R_in)`` -> re-envelope with the group's exit curvature ``R_out`` -- into one
    call.  The element SUPPLIES ``R_out`` from its own paraxial ABCD
    (:func:`system_abcd_prescription` mapped onto the incoming carrier), so the
    caller needs no external q-trace.

    ``validation/repro_traced_carrier_121/carrier_chain_121.py`` is the hand-
    written form of that pattern.  NOTE (v5.29): the two agree only with the
    LEGACY options -- ``carrier_reference='parabola'`` plus
    ``traced_kwargs={'amplitude_model': 'screen', 'preserve_input_phase':
    True}`` -- because the chain's defaults have since flipped to the validated
    carrier-regime configuration (see ``carrier_reference``).  With the shipping
    defaults this orchestrator is a DIFFERENT (and much more accurate) model
    than that script: design-121 best-focus EE6 79.7% -> 99.3%.

    Parameters
    ----------
    E_in : ndarray, complex, shape (Ny, Nx)
        Beam ENVELOPE at the input plane (the carrier phase divided out).  A
        plain field with no pre-referenced carrier is its own envelope -- pass
        it with ``r_in=inf`` (the default).
    groups : sequence
        The lens groups in order.  Each entry is either a lens **prescription
        dict** (has a ``'surfaces'`` key; taken with ``gap_before=0``) or a
        **group-spec dict** with keys:

        * ``'prescription'`` (required) -- the group prescription for
          :func:`apply_real_lens_traced`.
        * ``'gap_before'`` (float, default 0) -- free-space air distance from
          the previous plane (the input plane for the first group, else the
          previous group's exit vertex) to this group's front vertex.
        * ``'r_in'`` (optional) -- override the carrier radius handed to this
          group (default: the propagated carrier radius).
        * ``'r_out'`` (optional) -- override the exit carrier radius (default:
          the group's paraxial ABCD, :func:`_paraxial_group_r_out`).
        * ``'traced_kwargs'`` (optional dict) -- extra per-group kwargs merged
          over the chain-wide ``traced_kwargs``.

        An entry may instead be a **DOE entry** -- a thin diffractive screen
        between refractive groups (niche D4, roadmap P2) -- with keys:

        * ``'doe'`` (required) -- the DOE spec dict.  ``load_zemax_zmx``
          builds one per ``DGRATING`` surface in ``rx['diffractives']``, so
          ``{'doe': rx['diffractives'][k]}`` is a complete entry.  Keys:
          ``'period'`` (m, a scalar or ``(Px, Py)`` for a crossed grating),
          ``'order'`` (the design order, used when the entry does not
          override it), ``'angle_deg'`` (grating-vector azimuth, default 0),
          ``'origin'`` (m, default ``(0, 0)``), and the axial gaps
          ``'gap_before'`` / ``'gap_after'``.
        * ``'gap_before'`` / ``'gap_after'`` (float) -- axial distance to the
          previous / next plane (m).  **Default to the spec's own values**,
          which is the point: design 121's 51.539 mm last-lens -> DOE gap and
          its 7 mm DOE -> next-lens gap come straight from the .zmx and are
          bookkept here instead of being folded into a neighbour by hand.
          ``gap_before`` is charged to the PRE-DOE angle and ``gap_after`` to
          the POST-DOE angle -- the distinction a manual fold loses, and the
          reason the fold is only correct for an undeflected order -- while
          the ENVELOPE still crosses the DOE inside one transport step, so an
          order-0 DOE is bitwise inert whenever the entry's gaps re-sum to
          the gap they replace, which an axial-order fold always does (the
          chain accumulates them leg by leg, left to right; see the module
          note).

          **Every leg must be declared exactly once.**  The chain transports
          ``gap_before + gap_after`` for each DOE entry PLUS the next lens
          group's own ``gap_before``, and it cannot tell a leg declared twice
          from two equal legs.  ``load_zemax_zmx`` guarantees this among the
          diffractives it hands you (a DGRATING followed by another DGRATING
          gets ``gap_after = 0``, the leg riding on the next one's
          ``gap_before``); what you own is the join to the lens groups -- give
          the group that FOLLOWS the last DOE entry ``gap_before=0``, since
          that DOE's ``gap_after`` already carries the leg.
        * ``'order'`` (optional) -- ``m`` or ``(mx, my)`` for THIS run; the
          chain propagates ONE congruence, so one order at a time.  Defaults
          to the spec's design order (for an imported .zmx that reproduces
          the order Zemax's sequential trace follows, e.g. design 121's -4).
          Fractional orders are allowed (an even-count Dammann fan sits on
          the half-integer lattice).  Run a whole fan with
          :func:`propagate_traced_carrier_chain_multi`, one order per
          congruence via its ``'doe_order'`` key.
        * ``'amplitude'`` (optional complex, default 1) -- this order's
          complex coefficient ``c_mn``.  The DOE's efficiency/uniformity
          design lives here (or in the multi orchestrator's per-congruence
          ``weight``); the chain does not compute it.

        The order's action is exact and needs no new physics: the grating
        equation in DIRECTION COSINES (``L -> L + m lambda / Px``) plus, in
        this congruence's chief-ray-tracking frame, a complex CONSTANT on the
        envelope.  A DOE therefore turns a scalar-carrier chain into a tilted
        one from that plane onward (see ``r_in``), with the usual limit: the
        exact high-NA final leg does not carry a tilted congruence, so pass
        ``final_leg='paraxial'`` for a deflected order.  An order of 0 with
        unit amplitude is a pure gap and stays on the scalar path.
    wavelength, dx : float
        Wavelength and input grid pitch (m).  Isotropic (square) grid only.

        **GRID SIZE IS NOT THE ACCURACY LEVER, and large N is not a
        supported regime.**  This is the single most-often-wrong intuition
        about this function, so it is stated here rather than in an audit.
        Under the shipped defaults the chain is dx-FLAT: the design-121
        8-group relay reads best-focus FWHM 3.4156 / 3.4266 / 3.4265 um,
        EE3 88.83 / 88.83 / 88.83 %, EE6 99.58 / 99.58 / 99.58 %, window
        99.79 / 99.80 / 99.80 % at ``N`` = 2048 / 4096 / 8192 (launch
        ``dx`` = 1.0 / 0.5 / 0.25 um at a fixed 2.048 mm extent,
        pitch-preserving rays) -- four significant figures across a 4x
        refinement, including the row that read EE6 46.5 % before the v5.29
        default flip.  Refining ``dx`` past that buys nothing; if a number
        looks wrong, the lever is the CONFIGURATION (see
        ``carrier_reference`` and ``traced_kwargs``) or ``focus_readout``'s
        ``n_fine_cap`` / ``window_factor``, not ``N``.

        One configuration lever is easy to miss and is dx-invisible by
        construction: if the LAST group is fast and its input is
        COLLIMATED, no carrier engages there and the traced OPL fit has to
        represent the whole exit sphere, so the chain-default
        ``fit_radius_beam_factor`` = 2.0 can leave ~1 rad of exit-wavefront
        error (measured 1.122 rad at r=w on a stigmatic conic singlet at exit
        NA 0.20, against 0.087 rad at 1.5 and 0.031 rad from
        :func:`~lumenairy.apply_real_lens_gbd`).  It shows up as a HALO, not
        as a broadening, and it does not move with ``N``.  Try
        ``traced_kwargs={'fit_radius_beam_factor': 1.5}`` on such a design;
        see ``AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24.md`` S0.4.

        Above ``N`` = 8192 the chain is **not validated and not
        recommended**, for a cost reason rather than an accuracy one:
        measured on a 24-thread box, the 121 chain costs 82 / 104 / 191 /
        556 s at ``N`` = 1024 / 2048 / 4096 / 8192 (~2.9x per octave) and
        just the FIRST TWO of its eight groups exceeded 600 s at
        ``N`` = 16384, i.e. > 40 min for the run -- to move a converged
        4-digit answer.  Memory scales the same way: one complex128 grid is
        1.0 GiB at ``N`` = 8192, 4.0 GiB at 16384 and 12.25 GiB at 28672, and
        the chain holds several.  Large ``N`` is only ever
        meaningful PITCH-PRESERVING (pin ``dx``; ``N`` then buys guard band,
        not resolution); extent-preserving refinement is the axis this
        paragraph says is already flat.  Pinned in CI, without the
        prescription, by ``tests/unit/test_niche_d5_dx_flatness_gate.py``.
    r_in : float or TiltedCarrier, default ``inf``
        Carrier of ``E_in``.  A **float** is the historical signed radius (m);
        ``inf`` = a plain (collimated-carrier) field.

        A :class:`~lumenairy.TiltedCarrier` ``(R, L, M, x0, y0)`` -- or the
        shorthand sequence ``(R, L, M)`` / ``(R, L, M, x0, y0)`` -- carries a
        SPHERE PLUS LINEAR TILT congruence through the whole chain instead
        (niche D1, roadmap ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27
        P1a).  ``E_in`` is then the envelope in the CHIEF-RAY-TRACKING frame:
        the grid is centred on the chief ray at ``(x0, y0)``, and the physical
        field is ``E_in * exp(i k [S_R(u,v) + L u + M v])`` with
        ``u = x - x0``, ``v = y - y0``.

        Why it exists: a post-DOE fan is K comparable-power beams at
        well-separated angles, which
        :func:`lumenairy.apply_real_lens_traced`'s entrance->exit map
        explicitly excludes.  ONE ORDER AT A TIME each beam is a single clean
        congruence -- but only if the chain can carry its tilt, which a scalar
        radius cannot, so a +-46 mrad order used to sit ~2.3x outside the
        documented ``_NONCOLLIMATED_RESID_THRESH = 0.02 rad`` residual
        envelope.  With its own ``(R, L, M)`` each order's residual is the
        same small diffraction residual the on-axis case has.

        What the chain does with it (all derived, see the "Tilted chain
        carriers" note in this module):

        * **Free legs** are the untouched scalar transport.  In the tracking
          frame the tilt is INVARIANT and the chief ray advances by
          ``(L, M) * gap``; the tilt theorem's piston
          ``exp(+i k (L^2+M^2) z/2)`` is restored on the envelope so K
          congruences still recombine coherently.  Nothing is resampled, and
          the beam cannot walk off the co-moving grid as it collapses toward
          a focus.
        * **Element hand-offs** leave the tracking frame: the (smooth)
          envelope is band-limit-shifted onto the axis-centred grid so the
          element traces the beam at its PHYSICAL transverse position, and
          the element is handed the matching analytic
          :class:`~lumenairy.TiltedCarrier`.  The grid must therefore span
          the optical axis AND the displaced beam -- a ``ValueError`` names
          the shortfall if it does not.
        * **Exit states** follow the paraxial closure exactly:
          ``R_out = (A R + B)/(C R + D)`` (unchanged) with the chief ray
          ``(x_c, L)`` transforming as an ordinary paraxial ray,
          ``x_c' = A x_c + B L``, ``L' = C x_c + D L``.

        Limits: the EXACT high-NA final leg (``final_leg='exact'``, or
        ``'auto'`` above ``na_exact_threshold``) raises -- pass
        ``final_leg='paraxial'``; and ``L = M = x0 = y0 = 0`` routes through
        the scalar path bit-for-bit, so the shipped acceptance cannot move.
    ray_subsample : int, default 4
    n_workers : int, optional
        Threaded into every :func:`apply_real_lens_traced` call.
    traced_kwargs : dict, optional
        Extra kwargs for every :func:`apply_real_lens_traced` call (e.g.
        ``parallel_amp=False``).  ``carrier`` / ``dx`` / ``wavelength`` /
        ``prescription`` / ``ray_subsample`` / ``n_workers`` are managed by the
        orchestrator and must not appear here.

        The chain applies three DEFAULTS on top of the element's own (anything
        given here wins): ``amplitude_model='ray_density'`` and
        ``preserve_input_phase='remap'`` (the validated carrier-regime
        configuration -- see ``carrier_reference``) and
        ``fit_radius_beam_factor=2.0`` (the P2 aperture:beam cliff guard, audit
        AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24 §4).  The last one ties
        each group's ray-FIT domain to its beam instead of to the
        prescription's vignetting aperture: with an aperture much larger than
        the beam, marginal rays the beam never occupies corrupt the traced OPL
        fit *inside* the beam and the focus collapses silently (E4 corrected
        relay, beam w = 2 mm: exit-wavefront Strehl 0.998 at a 6 mm aperture ->
        0.105 at 7 mm -> 0.039 at 10 mm; >= 0.999 at every aperture from 4 to
        10 mm with the guard).  It
        restricts only the fit domain -- no field energy is vignetted -- and
        leaves the design-121 acceptance unchanged.  Pass
        ``traced_kwargs={'fit_radius_beam_factor': None}`` for the pre-v5.29
        aperture-only fit domain.
    final_distance : float, default 0
        Free-space distance from the last group's exit vertex to the target
        (readout) plane (m).
    focus_readout : dict, optional
        When given, land the final leg via a focus readout.  Must supply
        ``dx_out`` and ``N_out``; may supply ``centre_out`` / ``bandlimit``
        (and ``standoff`` for the paraxial path, or ``dx_fine`` / ``N_fine`` /
        ``window_factor`` for the exact path).  Otherwise the final leg is an
        ordinary carrier step and the field is reconstructed on the co-moving
        grid.  Unknown keys RAISE (niche C1): a silently-dropped
        ``'on_readout_windo'`` would leave the hard ``'error'`` default in
        place while reading as a downgrade.  The accepted set is
        ``_FOCUS_READOUT_KEYS``.

        Under a TILTED ``r_in`` (niche D1) ``centre_out`` is the ABSOLUTE
        image-plane position (optical-axis coordinates) to centre the window
        on -- the readout itself runs in the chief-ray-tracking frame and the
        offset is translated internally.  For an on-axis chief ray the two
        meanings coincide, so nothing changes on the scalar path.  Without a
        readout the returned grid is CENTRED ON THE CHIEF RAY and its
        absolute position is reported in the final ``stages`` entry.

        ``on_readout_window`` (default ``'error'``) and ``readout_window_tol``
        (default 1e-4) are forwarded to
        :func:`carrier_referenced_exact_focus_readout` on the exact path: they
        govern what happens when the ``window_factor`` crop cannot be taken at
        the requested size because the grid does not hold it AT the chief ray
        (``N*dx - 2*|c|``), and the shortfall truncates measurable beam power.
        The default REFUSES rather than returning a quietly truncated spot.

        ``window_factor`` is consumed TWICE on the exact path -- once by the
        pre-readout re-trace (:func:`_fine_trace_group_exit`, on the ENTRANCE
        beam) and once by :func:`carrier_referenced_exact_focus_readout` (on
        the EXIT beam).  The two crops compound below ``window_factor ~ 3``
        (measured: the second cut lands at 0.880 beam radii per unit factor at
        ``window_factor=2``, costing 8.6 EE3 points; exact no-op from 4
        upward, so the default 7.0 is unaffected).  See that function's
        ``window_factor`` entry for the formula and the measured table.

        Two additional keys govern the exact path's pre-readout re-trace
        (:func:`_fine_trace_group_exit`, audit
        AUDIT_TRACED_CHAIN_DX_SCALING_2026_07_22 F-C/F-D):

        * ``n_fine_cap`` (int, default 16384) -- memory cap on the re-trace
          grid size.  On a large enough chain grid the window can require
          more pixels than this to Nyquist-sample the exit sphere; the
          re-trace is then correctly DOWNSAMPLED to this size (F-A), but the
          resulting pitch may fall below the sphere's Nyquist limit and a
          ``RuntimeWarning`` fires (F-D) with the discarded-NA magnitude.
          Raise this (RAM permitting) to resolve the full NA, or shrink
          ``window_factor`` instead.
        * ``max_fine_launch_points`` (int, default 4096) -- independent
          backstop on the re-trace's Newton/Chebyshev ray-fit grid size,
          in case the physical-pitch-preserving ``ray_subsample`` (F-C)
          would still be too dense for this leg's window; a
          ``RuntimeWarning`` fires if it has to raise ``ray_subsample``
          above the pitch-preserving value to respect the cap.
        * ``ram_budget`` (float, optional) -- memory budget in BYTES for the
          fine re-trace grid AND the exact readout's internal grid (P2, audit
          AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24 §4).  Default
          :func:`lumenairy.get_ram_budget` (honours
          :func:`lumenairy.set_max_ram`).  Both grids are capped so their peak
          working set stays inside 50% of it, with a ``RuntimeWarning`` naming
          the un-degraded requirement whenever the cap binds -- so a
          memory-limited focus metric is flagged, never silently returned.
          Pass ``float('inf')`` for the pre-v5.29 uncapped behaviour (a hard
          ``MemoryError`` when the physics asks for more than the box has).
    final_leg : {'auto', 'exact', 'paraxial'}, default 'auto'
        How to land the FINAL leg when ``focus_readout`` is given.

        * ``'paraxial'`` -- always the paraxial
          :func:`carrier_referenced_focus_readout` (carrier-step short of focus
          + Bluestein zoom).  Fast, but CANNOT focus a high-NA (NA -> 0.5)
          converging leg -- the paraxial envelope cannot hold the exact-sphere
          r^4 (R9).
        * ``'exact'`` -- always the EXACT leg: re-trace the final group on a
          grid that Nyquist-samples its exit sphere, then
          :func:`carrier_referenced_exact_focus_readout` (exact-sphere
          reference + band-limited ASM Bluestein zoom, no paraxial
          magnification).
        * ``'auto'`` (default) -- the exact leg IFF the final group's exit NA
          exceeds ``na_exact_threshold`` (a strongly-focusing final group),
          else the fast paraxial path.  For every low-NA final leg this is
          byte-identical to ``'paraxial'`` (and to prior releases).
    na_exact_threshold : float, default 0.15
        Exit-NA (``w_env / |R_out|``) above which ``final_leg='auto'`` routes the
        final leg through the exact path.
    carrier_reference : {'sphere', 'parabola'}, default 'sphere'
        Which spherical reference the per-group HAND-OFFS use (audit
        AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24 S8).  The default
        ``'sphere'`` -- together with the default per-group
        ``traced_kwargs`` (``amplitude_model='ray_density'``,
        ``preserve_input_phase='remap'``, see below) -- is the VALIDATED
        carrier-regime configuration: the chain always operates with its
        carrier beyond the grid Nyquist, where these are the correct
        physics, not options (design-121: EE6 79.7% -> 99.3% at best
        focus).  ``'parabola'`` is the pre-v5.29 historical behaviour,
        retained as the legacy escape hatch: ``carrier_reference='parabola'``
        together with ``traced_kwargs={'amplitude_model': 'screen',
        'preserve_input_phase': True}`` reproduces pre-flip chain results
        exactly.

        ``'sphere'`` band-limits the paraxial parabola out of the hand-off:
        every reconstruction handed to :func:`apply_real_lens_traced` is
        converted to the EXACT sphere the element's ray launch / carrier
        eikonal assumes, and every traced exit is re-enveloped against the
        exact sphere, via :func:`_sphere_parab_conversion`.  The stored
        envelope is then the wavefront RESIDUAL vs the exact sphere -- the
        physically meaningful carried content -- instead of the residual vs
        the parabola (which contains a spurious ``+k r^4/(8 R^3)``, several
        radians on a long carrier leg at even a modest NA).

        This only changes the result when the element CONSUMES the input
        phase, i.e. together with
        ``traced_kwargs={'preserve_input_phase': 'remap', 'amplitude_model':
        'ray_density'}``.  With the default ``preserve_input_phase=False`` the
        element re-imposes its own spherical reference and discards whatever
        wavefront the input carried, so the conversion is a measured no-op
        (design-121: identical to 3 digits) -- and the chain's exit then
        carries only the LAST group's own contribution, i.e. minus the sum of
        the correction the earlier groups applied.  The validated
        carrier-regime configuration is therefore all three together:

        >>> res = propagate_traced_carrier_chain(          # doctest: +SKIP
        ...     env0, groups, wavelength, dx, r_in=R1,
        ...     carrier_reference='sphere',
        ...     traced_kwargs={'amplitude_model': 'ray_density',
        ...                    'preserve_input_phase': 'remap'},
        ...     final_distance=z, focus_readout=fr)

        On design-121 that combination takes the exit wavefront from
        ``+3.11 rad`` of r^4 (0.347 rad rms) to the full-train ray oracle's
        design floor (r^4 ``-0.13`` rad, rms 0.015 vs the oracle's 0.018) and
        the focal metrics from FWHM 5.15 um / EE3 55.6 / EE6 79.7 to
        FWHM 3.55 um / EE3 88.4 / EE6 99.3, dx-flat over N = 1024...4096.

        Caveat (documented, measured): the inter-group Sziklas-Siegman leg
        still transports the envelope with the PARAXIAL kernel, so under
        ``'sphere'`` the ``(S - parabola)`` difference rides inside the
        transported envelope (up to ~7 rad at r=w on the design-121 final
        gap).  That is exact to the transport's own paraxial order and was
        verified against the ray oracle hand-off by hand-off (agreement
        <= 0.01 rad); a ``RuntimeWarning`` fires if the conversion's
        band-limit radius reaches inside twice the beam radius.

        **How much that paraxial order actually costs is now MEASURED
        directly (S12) and it is a null.**  An exact-ASM gap prototype
        (reconstruct against the exact sphere -> non-paraxial band-limited
        angular spectrum -> re-envelope -> band-limited re-grid onto the SS
        pitch) was run against every converging design-121 gap.  Note a
        whole-grid exact transport is NOT possible on the co-moving extent --
        the reconstructed exact field is 9.2x beyond Nyquist at the grid edge
        on the last gap even after a 2x upsample, which is precisely why the
        carrier-referenced formulation exists -- so the prototype covers the
        beam CORE (a 2-beam-radius crop upsampled until the edge slope is
        0.77 of Nyquist) and blends into the SS result outside.  Measured
        agreement with the paraxial transport, amplitude-weighted over the
        core: **0.019 rad rms** on the final 3.323 mm gap (overlap 0.999936),
        0.024 rad on the S21-S22 gap, 0.019 rad on the S18-S20 gap.  End to
        end the exact transport moves best-focus EE3 by +0.20 points (89.57 ->
        89.77) and EE6 by +0.11 (99.26 -> 99.37) -- and most of even that is
        the prototype's own window bookkeeping (window-total 99.44 % ->
        99.54 %).  So the paraxial inter-group transport is worth <= 0.2 EE3
        points on this design; an exact sphere-referenced gap transport is
        still the principled generalisation for a genuinely high-NA gap, but
        it is not the design-121 error budget.

        The design-121 residual budget that IS real lives in the element:
        ``preserve_input_phase='remap'`` sampled the transported residual
        phasor on the coarse RAY lattice, which aliases the design's own
        r^4 carried content beyond ~1.5 beam radii.  Pass
        ``traced_kwargs={'remap_sampling': 'full'}`` (see
        :func:`lumenairy.apply_real_lens_traced`) to sample it at full
        wave-grid resolution instead.  Since the S12 flip this is a CHAIN
        DEFAULT (part of the shipped validated configuration); pass
        ``'ray'`` explicitly only to reproduce the pre-S12 byte-identical
        behaviour.

    self_check : {'dx', 'off', None}, default None
        Opt-in convergence self-check (P2, audit
        AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24 §5): with ``'dx'`` the
        whole chain is run a SECOND time at ``dx/sqrt(2)`` -- ``N`` raised to
        ``round_even(N*sqrt(2))`` so the physical extent is preserved, the input
        band-limit-resampled onto it -- and a ``RuntimeWarning`` fires if the
        focal metrics of the two runs disagree by more than ``self_check_tol``.

        This is the cheap "is this number dx-stable" flag: the traced chain's
        absolute focal metrics have historically drifted by tens of points under
        grid refinement (audit §1), so a single-grid number carries no
        convergence evidence on its own.  Compared metrics: after a focus
        readout (both runs land on the same ``dx_out``/``N_out`` grid) the window
        power, the peak intensity and the 50%-encircled-energy radius; without a
        readout the envelope 1/e^2 radius, the power and the carrier radius.

        Costs roughly 3x a plain run (the refined run is ~2x the primary), which
        is why it is opt-in.  It is a NECESSARY-not-sufficient test: agreement at
        two grids is evidence of a plateau, not proof.
    self_check_tol : float, default 0.05
        Relative tolerance (5%) for the ``self_check='dx'`` metric comparison.
    on_multi_congruence : {'warn', 'error', 'ignore'}, default 'warn'
        Guard rail on a MULTI-CONGRUENCE input (niche D3, roadmap
        ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27 P3).  This chain
        propagates ONE congruence: :func:`~lumenairy.apply_real_lens_traced`'s
        entrance->exit map assumes one congruence per exit pixel and names
        "comparable-power beams at well-separated angles (post-DOE at large
        split)" as the excluded case.  Pushed through multiplexed, the
        design-121 32-order fan produced a populated, credible-looking frame
        lattice whose per-frame power was scrambled (0.47 +/- 0.51 % against a
        design 2.78 %/frame) with nothing raised and nothing warned -- a
        plausible-looking WRONG ANSWER, which is why this gate exists.

        At entry the ENVELOPE is measured with the two SHIPPED detectors, and
        the gate fires when EITHER trips:

        * ``_carrier_residual_rms`` (``_lens_traced``) -- the residual
          transverse angular spread in radians, against the documented
          ``_NONCOLLIMATED_RESID_THRESH = 0.02 rad`` envelope.  Catches a
          congruence carried on the wrong reference (a bare +-46 mrad order
          handed in with ``r_in=inf`` reads 0.046 rad).
        * ``_tilt_dispersion`` (``fga``) -- the spread of the local wavevector
          about its per-region mean, the SAME measurement
          :func:`~lumenairy.apply_real_lens_universal` routes multi-valued
          fields on, put on a grid-canonical scale and compared against
          ``multi_congruence_threshold``.

        Both are needed: a wrapped nearest-neighbour gradient UNDER-REPORTS
        when the content aliases or interferes, and on a 2x2 order fan at
        46 mrad the first detector reads EXACTLY 0.0000 rad while the second
        reads 0.0149 rad; on a single tilted beam the roles reverse.  See the
        "P3 -- multi-congruence detection" note in this module for the full
        measured table.

        When it fires the message names the route to take:
        :func:`propagate_traced_carrier_chain_multi`, one congruence per DOE
        order / per emitter.  ``'error'`` makes it fatal (batch production);
        ``'ignore'`` skips the measurement entirely and is the documented
        escape for a caller who knows better (the measurement is two
        full-grid passes plus a Gaussian filter -- 0.8 s once per call at
        N=2048, i.e. well under 1 % of a design-121-class chain run, but it
        is not free on a tiny one).  A clean single congruence stays SILENT
        and byte-identical -- the measurement never touches the field.
    multi_congruence_threshold : float, default 0.06
        Cutoff for the ``_tilt_dispersion`` multi-valuedness score above, in
        fga's own currency (NA-normalized) and defaulted to fga's own
        ``multivalued_threshold``.  Two things are done to fga's raw reading
        before the comparison, and both are corrections to measured defects in
        this gate's first cut -- see the "P3 -- multi-congruence detection"
        module note (B.1, B.2) for the derivations and the full battery:

        * it is put on a GRID-CANONICAL scale (``x sqrt(lambda / dx)``).  The
          raw reading falls as sqrt(dx) on a crossing congruence -- structural,
          because two equal beams superpose to a REAL cosine times one carrier
          so the whole signal lives in the vanishing-amplitude pixels at the
          fringe nulls -- which made the detector BLINDER as the grid got
          finer: the design-121 8x4 fan read 2.97e-2 rad at dx0 = 4 um but
          8.36e-3 (SILENT) at dx0 = 0.25 um / N = 8192, the production pitch.
          Canonically it reads 1.70e-2 - 1.92e-2 rad across that whole sweep,
          and ~ 3.5 theta^1.5 for a crossing half-angle theta, independent of
          grid AND wavelength.
        * it is divided by a FIXED reference NA of 0.15
          (``_MULTI_CONGRUENCE_NA_REF``), not by the first group's
          ``fga._system_na``.  That quantity saturates at 0.375 and floors at
          0.03125, describes the first LENS rather than the field, and made the
          verdict a function of the first lens's f-number in both directions
          (a real fan silent behind a fast group; a single clipped-and-
          propagated beam firing behind a slow one).

        So the gate is an ABSOLUTE cutoff on the canonical dispersion:
        ``0.06 x 0.15`` = 9.0e-3 rad, the same for every design, grid and
        wavelength, consistent with detector A's absolute 0.02 rad envelope.
        Measured over 7 grid pitches (dx0 16 -> 0.25 um): the worst
        single-valued input the angular-spread detector does not already catch
        reads 4.6e-3 rad canonical (a beam clipped at 0.7 w then ASM-propagated
        0.02 z_R), a 1.9x margin held at every pitch, while the design-121 8x4
        fan reads 9.3e-3 - 1.92e-2 and fires at every pitch.  HONEST ENVELOPE:
        the cutoff corresponds to a crossing half-angle of ~19 mrad BETWEEN
        INTERFERING PAIRS, so two comparable beams closer than that are not
        caught by this detector at any pitch (measured +-10 mrad: 3.1e-3 -
        4.3e-3 rad, inside the clipped-beam population) and are caught only if
        they also trip the angular-spread detector.  Score a FAN by its TOTAL
        SPAN, derated about 20 % -- NOT by its order spacing, which
        under-predicts by ~5x: an 8x8 fan spanning +-23 mrad (spacing 6.6 mrad)
        reads like a 17-19 mrad pair, against 3.2-3.5 for a pair AT that
        spacing and 22.5-23.0 for a pair at that span.  Densifying at fixed
        span moves the score DOWN, not up (4 / 8 / 16 orders across +-23 read
        like 16.7 / 14.2 / 12.8 mrad).  So a fan whose span clears the floor is
        caught even when its spacing does not: the 8x8 +-23 fan sits ON the
        cutoff with no margin, while the design-121 8x4 fan at +-46 / +-23 mrad
        clears it by ~2x.
    on_na_proximity : {'warn', 'error', 'ignore'}, default 'warn'
        Guard rail on the EXIT-NA NEAR MISS (niche D3, roadmap P5).  With
        ``final_leg='auto'`` the readout silently routes to the PARAXIAL path
        below ``na_exact_threshold`` and to the exact path above it, and the
        paraxial side is ~200 rad of wavefront wrong at a design-121-class exit
        NA -- so a design sitting near the threshold can be flipped onto a
        badly wrong readout by one beam-size change, with no symptom.  (Design
        121 itself is NOT such a design: its measured ``na_exit`` is 0.405,
        170 % above the 0.15 default.  The 0.152 sometimes quoted for it is its
        geometric system NA, aperture/EFL, not the quantity this router
        branches on.)  When the measured exit NA lands
        within ``na_proximity_frac`` of the threshold (either side), a
        ``RuntimeWarning`` names both numbers, which side it landed on and the
        consumer-side fix (``final_leg='exact'`` explicitly).  Only checked for
        ``final_leg='auto'``: an explicit ``'exact'`` / ``'paraxial'`` is the
        caller's decision, not a silent route.
    na_proximity_frac : float, default 0.20
        Half-width of the proximity band, as a fraction of
        ``na_exact_threshold`` (0.20 = "within 20 %").
    on_ram_cap : {'warn', 'error', 'ignore'}, default 'warn'
        Disposition when the exact final leg's fine grid is capped by the RAM
        budget (niche D3, roadmap P5).  ``'warn'`` (historical) degrades and
        emits the "RESOLUTION-LIMITED (non-converged)" ``RuntimeWarning``;
        ``'error'`` raises a ``MemoryError`` so an unattended production run
        fails loudly instead of reporting a metric computed on a degraded
        grid.  Forwarded to both :func:`_fine_trace_group_exit` and
        :func:`carrier_referenced_exact_focus_readout`.
    on_rs_fine_clamp : {'warn', 'error', 'ignore'}, default 'warn'
        Disposition for the ``rs_fine`` clamp degenerate corner on the exact
        final leg (niche D3, roadmap P5).  When the memory/Nyquist-capped
        ``dx_fine`` is itself COARSER than the chain's physical ray pitch
        ``ray_subsample * cur_dx``, the pitch-preserving rescale rounds below 1
        and is clamped to 1, so the F-C pitch-preservation contract stops
        holding (measured 5.25x mismatch at the N=28672 / ``n_fine_cap``=16384
        design-121 condition).  ``'warn'`` (historical) names both pitches and
        continues; ``'error'`` is the opt-in STRICT mode that refuses it.
    on_tilt_exact_grid : {'error', 'warn', 'ignore'}, default 'error'
        Disposition when a TILTED congruence's EXACT high-NA final leg
        (niche D6) cannot be sampled.  The exact leg re-traces the last group
        through :func:`apply_real_lens_traced`, which builds its grid
        symmetrically about the OPTICAL AXIS, so a chief-ray-displaced beam
        needs an axis-centred window of ``2*max(|x_c|,|y_c|) + window_factor*w``
        rather than ``window_factor*w`` -- and the fine grid has to grow with
        it to keep ``dx_fine`` at or below the exit sphere's Nyquist pitch.
        When ``n_fine_cap`` / the RAM budget cannot pay for that, the retrace
        would still run but would silently discard the outer NA, so this
        REFUSES by default, naming the chief-ray offset, the beam radius, the
        window, the required ``n_fine`` and the cap that bound it.  ``'warn'``
        accepts the degraded leg (the on-axis F-D behaviour); ``'ignore'``
        silences it.  Measured on design 121's extreme order (-4,-2) at
        N=1024: the on-axis leg needs ``n_fine_cap`` 8192, the tilted one
        16384 -- 4x the memory and time for the same ``dx_fine``.

        TWO tests dispose through this knob (niche C1 item 4).  The one above
        is a cheap PRE-check from the chain's PARAXIAL exit NA
        ``w_in / |R_out|``, which is what SIZES the grid.  That is not the
        element's own exit NA and is systematically the smaller of the two
        (design 121 order (-4,-2): 0.4052 paraxial vs 0.4780 measured from the
        traced exit direction cosines), so a leg could pass it while
        ``apply_real_lens_traced`` itself warned the grid was too coarse.  The
        DECISIVE test is therefore a POST-check on the MEASURED exit NA, and
        its criterion is the fraction of traced exit power above the grid's
        Nyquist NA -- reported on the final stage as
        ``exit_power_above_nyquist`` next to ``na_exit_measured`` /
        ``na_grid_nyquist``, so the margin is visible without catching a
        warning.  See ``_TILT_EXACT_NA_POWER_TOL`` for the budget and the
        measurements that set it; the shipped 121 headline measures 7.97e-04
        against a 1e-2 budget, so nothing that runs today stops running.
    on_decentred_fit : {'error', 'warn', 'ignore'}, default 'warn'
        Disposition when a traced hand-off's chief ray sits more than
        ``decentre_fit_frac`` beam amplitude radii off the ELEMENT grid centre.
        A decentred hand-off measurably costs IMAGE quality end to end.
        MEASURED on the ``K = -n^2`` conic stand-in, whose truth is
        decentre-INVARIANT (chain / independent ray-trace + Kirchhoff oracle,
        EE2 ratio): 0.997 at 0 w, 1.005 at 0.50 w, 0.977 at 0.75 w, 0.983 at
        1.0 w, 0.923 at 1.5 w; and on design 121's post-DOE chain per order
        against a skew-ray + Debye oracle that says every order is EQUALLY
        diffraction-limited (EE3 ~90.7 %): EE3 87.6 % on axis, 86.0 % at
        (-1,0), 68.1 % at (-4,0), 65.3 % at (-4,-2).  **Any per-order spot
        size, Strehl or encircled energy reported through such a hand-off is a
        LOWER BOUND on the design, not the design's performance.**  The
        chain's POWER bookkeeping (per-order share, throughput, chief-ray
        landing) is unaffected.  Default ``'warn'`` because every
        multi-congruence fan trips it by construction; ``'error'`` refuses,
        ``'ignore'`` silences.

        Niche D7 (2026-07-29) re-scoped this guard: the residual is NOT
        ``apply_real_lens_traced``'s off-centre ray fit -- measured
        aliasing-free, that fit carries 0.90 urad of exit slope at 0.97 beam
        radii against 1.28 urad on axis UNTILTED, and 0.64 urad on axis under
        48.7 mrad of tilt, so it is not uniformly the smaller of the two
        (0.007 um of blur against a 3.5 um FWHM on any of those readings)
        -- nor the fine-retrace grid, the Newton cap or the readout
        window, each of which moves design 121's extreme-order EE3 by
        <= 0.01 point.  The 3.7 -> 408 urad exit-slope curve this docstring
        used to quote was an artefact of the repro script's FFT-derivative
        slope extraction (it reports 400 urad on a synthetic field built to be
        right to 0.36 urad); see ``_DECENTRED_FIT_POLY_ORDER``.
    decentre_fit_frac : float, default 0.5
        Chief-ray offset, in beam amplitude radii, above which
        ``on_decentred_fit`` fires.  The measured onset on the decentre-
        invariant stand-in above is ~0.75 w; 0.5 keeps the default one step
        conservative of it.  ``0`` disables the check.

    Returns
    -------
    TracedCarrierChainResult
        ``(field, R, dx, stages)`` -- see the class docstring.  ``R`` is
        ``None`` after any focus readout (paraxial or exact).
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'propagate_traced_carrier_chain',
                           input_kind='field')
    from ..elements import apply_real_lens_traced

    if self_check not in (None, 'off', 'dx'):
        raise ValueError(
            "propagate_traced_carrier_chain: self_check must be 'dx', 'off' or "
            f"None, got {self_check!r}.")
    _fn = 'propagate_traced_carrier_chain'
    _check_guard_action('on_multi_congruence', on_multi_congruence, _fn)
    _check_guard_action('on_na_proximity', on_na_proximity, _fn)
    _check_guard_action('on_ram_cap', on_ram_cap, _fn)
    _check_guard_action('on_rs_fine_clamp', on_rs_fine_clamp, _fn)
    _check_guard_action('on_tilt_exact_grid', on_tilt_exact_grid, _fn)
    _check_guard_action('on_decentred_fit', on_decentred_fit, _fn)
    # niche C1 item 5: ``focus_readout`` had no key whitelist, so a typo was
    # silently accepted and the caller kept the DEFAULT -- e.g.
    # ``'on_readout_windo'`` left ``on_readout_window`` at the hard ``'error'``
    # while the caller believed they had downgraded it.  Refuse, naming the
    # accepted set (the same contract ``congruences`` / ``output_grid`` / the
    # DOE entries already have).
    if focus_readout is not None:
        if not isinstance(focus_readout, dict):
            raise ValueError(
                f"{_fn}: focus_readout must be a dict with 'dx_out' and "
                f"'N_out' (the image-plane readout grid); got "
                f"{type(focus_readout).__name__}.")
        _fr_unknown = set(focus_readout) - _FOCUS_READOUT_KEYS
        if _fr_unknown:
            raise ValueError(
                f"{_fn}: focus_readout has unknown key(s) "
                f"{sorted(_fr_unknown)!r}; accepted keys are "
                f"{sorted(_FOCUS_READOUT_KEYS)!r}.  (A dropped key is not "
                f"inert: 'on_readout_windo' would leave on_readout_window at "
                f"its hard 'error' default while reading as a downgrade.)")
    if not (np.isfinite(decentre_fit_frac) and decentre_fit_frac >= 0.0):
        raise ValueError(
            f"{_fn}: decentre_fit_frac must be a finite non-negative number "
            f"of beam amplitude radii, got {decentre_fit_frac!r}.")
    if not (np.isfinite(multi_congruence_threshold)
            and multi_congruence_threshold > 0.0):
        raise ValueError(
            f"{_fn}: multi_congruence_threshold must be a finite score > 0, "
            f"got {multi_congruence_threshold!r}.")
    if not (np.isfinite(na_proximity_frac) and na_proximity_frac >= 0.0):
        raise ValueError(
            f"{_fn}: na_proximity_frac must be a finite fraction >= 0, got "
            f"{na_proximity_frac!r}.")
    if self_check == 'dx':
        # Run the primary + the dx/sqrt(2) control through this same entry point
        # with the check disabled, then compare.  (Recursion, not a refactor, so
        # the validated body below stays byte-identical on the default path.)
        # ``list(groups)`` first: the two runs must see the same groups, so a
        # one-shot iterable would otherwise leave the control run empty.
        _kw = dict(
            E_in=E_in, groups=list(groups), wavelength=wavelength, dx=dx,
            r_in=r_in,
            ray_subsample=ray_subsample, n_workers=n_workers,
            traced_kwargs=traced_kwargs, final_distance=final_distance,
            focus_readout=focus_readout, final_leg=final_leg,
            na_exact_threshold=na_exact_threshold,
            carrier_reference=carrier_reference,
            on_multi_congruence=on_multi_congruence,
            multi_congruence_threshold=multi_congruence_threshold,
            on_na_proximity=on_na_proximity,
            na_proximity_frac=na_proximity_frac,
            on_ram_cap=on_ram_cap, on_rs_fine_clamp=on_rs_fine_clamp,
            on_tilt_exact_grid=on_tilt_exact_grid,
            on_decentred_fit=on_decentred_fit,
            decentre_fit_frac=decentre_fit_frac)
        _res = propagate_traced_carrier_chain(**_kw)
        _run_chain_dx_self_check(_kw, _res, float(self_check_tol))
        return _res

    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError(
            "propagate_traced_carrier_chain: wavelength must be finite and "
            f"positive, got {wavelength!r}.")
    if not (np.isfinite(dx) and dx > 0):
        raise ValueError(
            "propagate_traced_carrier_chain: dx must be finite and positive, "
            f"got {dx!r}.")

    if final_leg not in ('auto', 'exact', 'paraxial'):
        raise ValueError(
            "propagate_traced_carrier_chain: final_leg must be 'auto', 'exact' "
            f"or 'paraxial', got {final_leg!r}.")

    if carrier_reference not in ('parabola', 'sphere'):
        raise ValueError(
            "propagate_traced_carrier_chain: carrier_reference must be "
            f"'parabola' or 'sphere', got {carrier_reference!r}.")
    _sphere_ref = (carrier_reference == 'sphere')

    groups = list(groups)
    # P3 (niche D3): refuse -- or shout about -- an input that is not the ONE
    # congruence this chain propagates, naming the multi-congruence route.
    # Measured on the ENVELOPE the caller passed (never on a reconstruction:
    # the chain's carrier is beyond the grid Nyquist by construction, so a
    # reconstructed field would alias and feed the gradient estimator the very
    # wrapped increments that make it under-report).  Read-only.
    _check_chain_entry_congruence(
        E_in, dx, wavelength, on_multi_congruence,
        multi_congruence_threshold, _fn)
    # v5.29 default flip (audit AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24 S8):
    # the chain's per-group traced calls default to the validated
    # carrier-regime configuration -- the chain ALWAYS operates with its
    # carrier beyond the grid Nyquist, where the geometric (ray-density)
    # amplitude and the geometric residual carry are the correct physics,
    # not preferences.  Anything the caller passes in ``traced_kwargs`` (or a
    # group's own ``traced_kwargs``) WINS over these defaults; the standalone
    # ``apply_real_lens_traced`` element defaults are untouched.
    # P2 (audit AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24 §4): the chain also
    # defaults the APERTURE:BEAM CLIFF GUARD on -- the ray-fit domain is tied to
    # the beam, not to the (arbitrary, prescription-supplied) vignetting
    # aperture.  A chain is exactly the daily-driver case that receives
    # arbitrary apertures, and the cliff is silent: measured on the E4 corrected
    # relay, exit-wavefront Strehl 0.998 (6 mm aperture) -> 0.105 (7 mm) ->
    # 0.039 (10 mm) with no warning and no energy loss to show for it, recovered
    # to 0.9995 at every aperture by this default.  Fit-domain only: no field
    # energy is vignetted (measured identical exit power to 4 digits), and the
    # design-121 acceptance is unchanged.
    from ..elements._lens_traced import _FIT_RADIUS_BEAM_FACTOR_DEFAULT
    base_kw = {'amplitude_model': 'ray_density',
               'preserve_input_phase': 'remap',
               'fit_radius_beam_factor': _FIT_RADIUS_BEAM_FACTOR_DEFAULT,
               # S12 (audit sweep): sample the carried residual phasor at
               # wave-grid resolution -- the coarse-ray-lattice sampling
               # aliases the design's r^4 correction beyond ~1.5w and makes
               # the result ray_subsample-dependent.  'full' is dx- and
               # rs-flat and puts the 121 at the ideal-field FWHM ceiling.
               'remap_sampling': 'full'}
    if traced_kwargs:
        base_kw.update(traced_kwargs)
    R, tilt_L, tilt_M, x_c, y_c, _tilted = _parse_chain_carrier(
        r_in, 'propagate_traced_carrier_chain')
    if _tilted:                    # fail fast on a non-propagating tilt
        _tilt_obliquity(tilt_L, tilt_M, 'propagate_traced_carrier_chain')
    k0 = 2.0 * np.pi / wavelength
    cur_dx = float(dx)
    env = E_in
    stages: list = []
    # ---- niche D4 (roadmap P2): DOE entries in ``groups`` ------------------
    # ``_last_group`` is the last LENS group, not the last entry: a DOE after
    # it must not steal the final group's exact-leg routing.  ``pend_gap`` is
    # the DOE's own trailing gap, still unspent -- the chain folds it into the
    # next group's leg (or into ``final_distance``) so the consumer never
    # hand-folds it.  Both are inert without a DOE: ``_last_group`` is then
    # ``n_groups - 1`` and ``pend_gap`` stays 0.0, which keeps every leg
    # byte-identical (``0.0 + gap is gap`` for every finite gap).
    _is_doe = [_is_doe_entry(g) for g in groups]
    _last_group = max((i for i, d in enumerate(_is_doe) if not d), default=-1)
    if _last_group < 0:
        raise ValueError(
            f"{_fn}: groups contains only DOE entries and no lens group.  A "
            f"DOE is a thin screen between refractive groups, not a group -- "
            f"apply it to the field directly (its whole action on ONE order "
            f"is a tilt plus a constant) if there is no lens to run.")
    _trailing_doe = any(_is_doe[_last_group + 1:])
    # ``pend_gap`` is deferred TRANSPORT distance; ``pend_own`` is the part of
    # it still owed to the CURRENT tilt (the chief-ray advance and the
    # obliquity piston, both analytic).  See the DOE branch for why they can
    # differ.  Both stay 0.0 without a DOE, so every leg is byte-identical.
    pend_gap = 0.0
    pend_own = 0.0

    for gi, g in enumerate(groups):
        if _is_doe[gi]:
            # ---- DOE plane (see the "DOE chain entries" note above) --------
            (_dnm, _dgb, _dga, _dL, _dM, _damp, _dorg,
             _dord) = _normalise_doe_entry(g, gi, wavelength, _fn)
            # The DOE plane does NOT interrupt the carrier leg.  In this
            # congruence's tracking frame the order's entire action on the
            # ENVELOPE is a complex CONSTANT, and a constant commutes with the
            # (linear) transport exactly -- so the envelope crosses the DOE
            # inside ONE Sziklas-Siegman step, and the DOE's gaps are simply
            # added to the neighbouring legs.  That is not just tidier, it is
            # what keeps an EXPRESSED design on the same numerics as the
            # hand-folded gap it replaces: BITWISE (given the axial-order
            # gap accumulation below), where a split leg would agree only to
            # ~1e-11 (the extra FFT pair) -- and not even that when the split
            # plane lands inside the near-focus bridge zone, where the bridge
            # re-grids the co-moving pitch.  See the module note for both
            # measurements (design 121's own DOE leg is COLLIMATED,
            # R = +703.6 m, so only the first applies to it).
            # Only the chief ray, the piston and the tilt are per-segment, and
            # all three are analytic.
            _own = pend_own + _dgb
            if _tilted and _own != 0.0:
                _ob = _tilt_obliquity(tilt_L, tilt_M, _fn)
                x_c += tilt_L * _own * _ob
                y_c += tilt_M * _own * _ob
                env = np.asarray(env) * np.exp(
                    1j * k0 * _own * (_ob - 1.0))
            # In the tracking frame the order's ramp IS the tilt, so all that
            # is left on the envelope is that complex constant: the order
            # amplitude times the phase between the grating's own origin and
            # this congruence's chief ray (what makes K orders recombine).
            _c = _damp * np.exp(1j * k0 * (_dL * (x_c - _dorg[0])
                                           + _dM * (y_c - _dorg[1])))
            if _c != 1.0:
                _e = np.asarray(env)
                env = (_e * _e.dtype.type(_c) if np.iscomplexobj(_e)
                       else _e * _c)
            tilt_L += _dL
            tilt_M += _dM
            if _dL or _dM:
                _tilted = True
                _tilt_obliquity(tilt_L, tilt_M, _fn)   # fail fast, named
            # ACCUMULATE LEFT TO RIGHT, IN AXIAL ORDER -- one ``+=`` per leg,
            # NOT ``pend_gap += _dgb + _dga``.  Float addition is not
            # associative, and the hand fold a DOE entry replaces is the
            # axial-order sum ``gb1 + ga1 + gb2 + ga2 + gap``, which Python
            # evaluates left to right.  Grouping the pairs first gave
            # ``(gb1+ga1) + (gb2+ga2)``, a DIFFERENT float: gaps
            # 0.02/0.0/0.01/0.007 folded to 0.037 but accumulated to
            # 0.037000000000000005, one ulp out -- and one ulp is NOT small
            # here: the traced pipeline (ray trace -> wavefront fit ->
            # resample) has a roundoff NOISE FLOOR of ~1e-7 relative, which a
            # few ulp on a gap are enough to reach.  Measured on that 37 mm
            # gap: +1 ulp (6.9e-18 m, a k dz of 3e-11 rad) moves the output
            # field by 6.5e-11 on one relay and 1.4e-7 on a faster one, and
            # +10 ulp by 8.1e-8 -- i.e. it jumps to the floor and SATURATES
            # there rather than scaling with the perturbation.  So "an
            # order-0 DOE is bitwise inert" silently became a measurable
            # difference.  Summing leg by leg in axial order makes the
            # accumulation bit-identical to the axial-order fold for ANY
            # gaps, not just ones that happen to re-associate.
            pend_gap += _dgb
            pend_gap += _dga
            pend_own = _dga
            # POWER ACROSS THE SCREEN.  The stage reports it even though the
            # DOE plane has no grid state of its own (the transport is
            # deferred, so ``cur_dx`` still belongs to the last completed
            # plane): power is INVARIANT along the deferred leg, so it is
            # well defined here while ``w`` and ``dx`` are not.  Without it a
            # TRAILING DOE's ``amplitude`` went unaccounted -- the multi
            # orchestrator's exit power is the last stage that reports one
            # (:func:`_multi_chain_exit_power`), so ``power_exit`` /
            # ``throughput`` / ``capture`` all read the pre-DOE power against
            # a post-DOE readout, i.e. capture |a|^-2 too small and a FALSE
            # readout-clipping diagnostic at ``on_readout_clip='error'``.
            _p_doe = _chain_envelope_stats(env, cur_dx)[1]
            stages.append({
                'name': _dnm, 'doe': True, 'order': _dord,
                'dL': _dL, 'dM': _dM, 'L_out': tilt_L, 'M_out': tilt_M,
                'x_c': x_c, 'y_c': y_c, 'R': R,
                'gap_before': _dgb, 'gap_after': _dga,
                'amplitude': complex(_damp), 'power': _p_doe})
            continue
        if isinstance(g, dict) and 'prescription' in g:
            presc = g['prescription']
            gap = float(g.get('gap_before', g.get('distance', 0.0)))
            g_r_in = g.get('r_in')
            g_r_out = g.get('r_out')
            g_kw = g.get('traced_kwargs')
        elif isinstance(g, dict) and 'surfaces' in g:
            presc, gap, g_r_in, g_r_out, g_kw = g, 0.0, None, None, None
        else:
            raise ValueError(
                f"propagate_traced_carrier_chain: groups[{gi}] must be a "
                "prescription dict (with 'surfaces'), a group-spec dict "
                "(with 'prescription') or a DOE entry (with 'doe'); got "
                f"{type(g).__name__}.")

        # niche D4: spend any DOE's deferred gaps on THIS leg -- the library's
        # side of "no manual fold".  ``_own`` is the part of it travelled at
        # the CURRENT angle (everything after the last DOE plane); the segment
        # before that plane was already charged to the pre-DOE angle there.
        # Without a DOE both are this group's own ``gap_before``.
        _own = pend_own + gap
        gap = pend_gap + gap
        pend_gap = 0.0
        pend_own = 0.0

        # free-space carrier leg to the group front vertex
        if gap != 0.0:
            cr = propagate_carrier_referenced(env, R, gap, wavelength, cur_dx)
            env, R, cur_dx = cr.env, cr.R, cr.dx
            if isinstance(cur_dx, tuple):
                cur_dx = cur_dx[0]
            if isinstance(R, tuple):
                R = R[0]
        if _tilted and _own != 0.0:
            # The chief ray advances by the EXACT geometric
            # ``gap * (L, M)/cos(theta)`` and the frame picks up the exact
            # tilted-ray path piston ``exp(i k gap (1/cos(theta) - 1))``
            # -- both are closed forms of the angular-spectrum expansion
            # about the carrier wavevector (see _tilt_obliquity), so
            # nothing beyond the residual envelope stays paraxial here.
            # The tilt itself is invariant in the tracking frame.
            _ob = _tilt_obliquity(tilt_L, tilt_M,
                                  'propagate_traced_carrier_chain')
            x_c += tilt_L * _own * _ob
            y_c += tilt_M * _own * _ob
            env = np.asarray(env) * np.exp(1j * k0 * _own * (_ob - 1.0))

        if g_r_in is None:
            R_use = float(R)
        else:
            # a per-group override may itself be a TiltedCarrier (re-seed the
            # whole congruence) or a plain radius (keep the carried tilt).
            (R_use, _oL, _oM, _ox, _oy, _otilt) = _parse_chain_carrier(
                g_r_in,
                f"propagate_traced_carrier_chain: groups[{gi}]['r_in']")
            if _otilt:
                tilt_L, tilt_M, x_c, y_c = _oL, _oM, _ox, _oy
                _tilted = True
        # the element SUPPLIES R_out (its own paraxial ABCD), unless overridden
        R_out = (float(g_r_out) if g_r_out is not None
                 else _paraxial_group_r_out(presc, R_use, wavelength))
        _name = (presc.get('name', f'group{gi}') if isinstance(presc, dict)
                 else f'group{gi}')
        call_kw = dict(base_kw)
        if g_kw:
            call_kw.update(g_kw)
        # Default-flip boundary guard: the chain SUPPLIES
        # preserve_input_phase='remap', which the element rejects unless
        # amplitude_model='ray_density'.  A caller who overrides only the
        # amplitude therefore hits an element-level error naming an option they
        # never wrote.  Raise here instead, naming where 'remap' came from and
        # the two ways out.
        if (call_kw.get('preserve_input_phase') == 'remap'
                and call_kw.get('amplitude_model') != 'ray_density'
                and not (traced_kwargs
                         and 'preserve_input_phase' in traced_kwargs)
                and not (g_kw and 'preserve_input_phase' in g_kw)):
            raise ValueError(
                f"propagate_traced_carrier_chain: groups[{gi}] would run "
                f"amplitude_model={call_kw.get('amplitude_model')!r} with the "
                f"CHAIN-DEFAULT preserve_input_phase='remap', which "
                f"apply_real_lens_traced rejects ('remap' reuses the "
                f"ray-density entrance pullback).  Since v5.29 the chain "
                f"defaults to the validated triple carrier_reference='sphere' "
                f"+ amplitude_model='ray_density' + "
                f"preserve_input_phase='remap' (see carrier_reference), so "
                f"overriding the amplitude alone leaves an inconsistent pair.  "
                f"Either keep amplitude_model='ray_density', or take the whole "
                f"legacy configuration: carrier_reference='parabola' with "
                f"traced_kwargs={{'amplitude_model': 'screen', "
                f"'preserve_input_phase': True}}.")

        # ---- exact high-NA FINAL leg (R9): re-trace this (last) group on a
        # grid that Nyquist-samples its exit sphere, then exact-ASM to target.
        is_final = (gi == _last_group)
        do_exact = False
        na_exit = 0.0
        if is_final and focus_readout is not None and final_leg != 'paraxial':
            w_in = _envelope_amp_radius(env, cur_dx, cur_dx)
            if np.isfinite(R_out) and R_out != 0.0 and w_in > 0.0:
                na_exit = w_in / abs(R_out)
            do_exact = (final_leg == 'exact'
                        or (final_leg == 'auto' and na_exit > na_exact_threshold))
            # P5 (niche D3): the 'auto' route flips SILENTLY at
            # na_exact_threshold, and below it the paraxial readout is ~200 rad
            # wrong at a design-121-class exit NA.  Design 121 itself is clear
            # of the flip (measured na_exit 0.405 against the 0.15 default,
            # 170% above), but any design that lands near it gets moved across
            # by one beam-size change with no symptom.  Make the near miss
            # visible.
            if (final_leg == 'auto' and na_exit > 0.0
                    and na_proximity_frac > 0.0
                    and abs(na_exit - na_exact_threshold)
                    <= na_proximity_frac * na_exact_threshold):
                _side = ('ABOVE (routing EXACT)' if do_exact
                         else 'BELOW (routing PARAXIAL)')
                _guard_dispose(
                    on_na_proximity,
                    f"{_fn}: the final group's measured exit NA "
                    f"{na_exit:.5f} sits within "
                    f"{na_proximity_frac:.0%} of na_exact_threshold="
                    f"{na_exact_threshold} -- {_side}.  final_leg='auto' flips "
                    f"between the exact and the PARAXIAL focus readout at that "
                    f"threshold with no other symptom, and the paraxial "
                    f"readout is ~200 rad of wavefront wrong at a "
                    f"design-121-class exit NA (design 121 measures na_exit "
                    f"0.405 -- clear of the flip; the 0.152 sometimes quoted "
                    f"for it is its geometric aperture/EFL system NA, not "
                    f"this quantity).  A beam-size change of "
                    f"{abs(na_exit - na_exact_threshold) / na_exit:.1%} would "
                    f"flip it.  Pass final_leg='exact' explicitly to pin the "
                    f"route (the recommended production setting whenever the "
                    f"exit NA is anywhere near the threshold), or move "
                    f"na_exact_threshold clear of the design.  "
                    f"on_na_proximity='error' makes this fatal, 'ignore' "
                    f"silences it.",
                    stacklevel=2)
        if do_exact and _trailing_doe:
            raise NotImplementedError(
                f"{_fn}: the EXACT high-NA final leg (final_leg="
                f"{final_leg!r}, exit NA {na_exit:.4f}) lands the target "
                f"plane from inside the LAST GROUP, so it cannot also apply "
                f"the DOE entr{'ies' if sum(_is_doe[_last_group + 1:]) > 1 else 'y'}"
                f" that follow{'' if sum(_is_doe[_last_group + 1:]) > 1 else 's'}"
                f" it in groups (niche D4).  Move the trailing DOE into its "
                f"own run, or pass final_leg='paraxial'.")
        if do_exact:
            fr = dict(focus_readout)
            if 'dx_out' not in fr or 'N_out' not in fr:
                raise ValueError(
                    "propagate_traced_carrier_chain: focus_readout must supply "
                    "'dx_out' and 'N_out'.")
            _na_diag: dict = {}
            E_exit_fine, dx_fine = _fine_trace_group_exit(
                env, R_use, cur_dx, presc, wavelength, ray_subsample, n_workers,
                call_kw, R_out, na_exit,
                window_factor=float(fr.get('window_factor', 7.0)),
                n_fine_cap=int(fr.get('n_fine_cap', 16384)),
                max_fine_launch_points=int(
                    fr.get('max_fine_launch_points', 4096)),
                sphere_reference=_sphere_ref,
                ram_budget=fr.get('ram_budget'),
                on_ram_cap=on_ram_cap, on_rs_fine_clamp=on_rs_fine_clamp,
                centre=((x_c, y_c) if _tilted else (0.0, 0.0)),
                tilt=((tilt_L, tilt_M) if _tilted else (0.0, 0.0)),
                on_tilt_exact_grid=on_tilt_exact_grid,
                on_decentred_fit=on_decentred_fit,
                decentre_fit_frac=decentre_fit_frac,
                na_diag_out=_na_diag)
            w_stage, p_stage = _chain_envelope_stats(E_exit_fine, dx_fine)
            stages.append({
                'name': _name, 'R_in': R_use, 'R_out': R_out, 'dx': dx_fine,
                'w': w_stage, 'power': p_stage, 'exact_final': True,
                'na_exit': na_exit})
            # niche C1 item 4: report the element's MEASURED exit NA and the
            # exit power above this grid's Nyquist NA alongside the paraxial
            # ``na_exit`` the leg was SIZED from, so the margin is visible
            # without catching a warning (the same reason D3 put 'na_exit'
            # here).
            if _na_diag.get('na_exit') is not None:
                stages[-1].update({
                    'na_exit_measured': float(_na_diag['na_exit']),
                    'na_grid_nyquist': float(_na_diag['na_nyquist']),
                    'exit_power_above_nyquist': float(
                        _na_diag['power_frac_above_nyquist'])})
            exact_kw = {kk: fr[kk] for kk in (
                'dx_out', 'N_out', 'dx_fine', 'N_fine', 'window_factor',
                'centre_out', 'bandlimit', 'ram_budget',
                'on_readout_window', 'readout_window_tol') if kk in fr}
            if _tilted:
                # niche D6: the EXIT congruence -- the same closure the coarse
                # path uses (``(x_c, L)`` is an ordinary paraxial ray through
                # the group's air-to-air ABCD; the sphere follows the Moebius
                # law ``R_out`` already carries).  ``E_exit_fine`` is on an
                # AXIS-CENTRED grid (the element traced it there), so the
                # readout is told where the beam actually is and takes its own
                # window about that point; ``centre_out`` stays absolute.
                _A, _B, _C, _D = _group_abcd(presc, wavelength)
                _xco = _A * x_c + _B * tilt_L
                _yco = _A * y_c + _B * tilt_M
                exact_kw['centre'] = (_xco, _yco)
                exact_kw['tilt'] = (_C * x_c + _D * tilt_L,
                                    _C * y_c + _D * tilt_M)
                stages[-1].update({
                    'L_out': exact_kw['tilt'][0], 'M_out': exact_kw['tilt'][1],
                    'x_c_out': _xco, 'y_c_out': _yco})
                # the exact-leg exit power lives on the beam, not the axis:
                # measure the stage envelope about the chief ray too.
                stages[-1]['w'] = _envelope_amp_radius(
                    E_exit_fine, dx_fine, dx_fine, centre=(_xco, _yco))
            _pd = {}
            field = carrier_referenced_exact_focus_readout(
                E_exit_fine, R_out, final_distance, wavelength, dx_fine,
                on_ram_cap=on_ram_cap, _period_out=_pd, **exact_kw)
            if 'period' in _pd:
                stages[-1]['readout_period'] = _pd['period']
            if _tilted:
                _obf = _tilt_obliquity(exact_kw['tilt'][0],
                                       exact_kw['tilt'][1], _fn)
                stages.append({
                    'name': '<target>', 'target': True,
                    'L': exact_kw['tilt'][0], 'M': exact_kw['tilt'][1],
                    'x_c': _xco + exact_kw['tilt'][0] * final_distance * _obf,
                    'y_c': _yco + exact_kw['tilt'][1] * final_distance * _obf,
                    'centre_out': tuple(float(v) for v in exact_kw.get(
                        'centre_out', (0.0, 0.0))),
                    'dx': float(fr['dx_out']), 'exact_final': True,
                    'readout_period': _pd.get('period')})
            return TracedCarrierChainResult(
                np.asarray(field), None, float(fr['dx_out']), stages)

        # ---- standard coarse trace + paraxial re-envelope ------------------
        if not _tilted:
            E_full = carrier_referenced_reconstruct(env, R_use, wavelength,
                                                    cur_dx)
            if _sphere_ref:
                # hand the element the EXACT-sphere-referenced wavefront its
                # ray launch assumes (see carrier_reference)
                _cf = _sphere_parab_conversion(
                    np.shape(E_full), cur_dx, wavelength, R_use, +1,
                    w_beam=_envelope_amp_radius(env, cur_dx, cur_dx))
                if _cf is not None:
                    E_full = np.asarray(E_full) * _cf
            _carrier_arg = R_use
        else:
            # niche D1 -- leave the chief-ray-tracking frame for the element:
            # the prescription is traced on the GRID, so the beam has to sit
            # at its true transverse position for the surface zones it
            # actually crosses to be the ones the trace uses.
            _check_tilt_fits(env, cur_dx, x_c, y_c, f"groups[{gi}] ({_name})")
            _w_track = _envelope_amp_radius(env, cur_dx, cur_dx)
            _check_decentred_fit(_w_track, x_c, y_c,
                                 f"groups[{gi}] ({_name})", on_decentred_fit,
                                 decentre_fit_frac)
            env_axis = _shift_envelope(env, x_c, y_c, cur_dx)
            _ph = _radial_carrier_phase(
                np.shape(env_axis), cur_dx, cur_dx, wavelength, R_use, +1,
                centre=(x_c, y_c)) if np.isfinite(R_use) else None
            E_full = env_axis if _ph is None else np.asarray(env_axis) * _ph
            if _sphere_ref:
                _cf = _sphere_parab_conversion(
                    np.shape(E_full), cur_dx, wavelength, R_use, +1,
                    w_beam=_w_track, centre=(x_c, y_c))
                if _cf is not None:
                    E_full = np.asarray(E_full) * _cf
            _rp = _tilt_ramp(np.shape(E_full), cur_dx, wavelength,
                             tilt_L, tilt_M, x_c, y_c, +1)
            if _rp is not None:
                E_full = np.asarray(E_full) * _rp
            from ..elements._lens_traced import TiltedCarrier as _TC
            _carrier_arg = _TC(R_use, tilt_L, tilt_M, x_c, y_c)
        E_exit = apply_real_lens_traced(
            E_full, prescription=presc, wavelength=wavelength, dx=cur_dx,
            carrier=_carrier_arg, ray_subsample=ray_subsample,
            n_workers=n_workers, **call_kw)
        E_exit = np.asarray(E_exit)
        if not _tilted:
            if _sphere_ref:
                # re-envelope against the EXACT exit sphere, so the stored
                # envelope is the wavefront residual (the carried content)
                _cf = _sphere_parab_conversion(E_exit.shape, cur_dx,
                                               wavelength, R_out, -1)
                if _cf is not None:
                    E_exit = E_exit * _cf
            env = carrier_referenced_envelope(E_exit, R_out, wavelength,
                                              cur_dx)
        else:
            # exit congruence: sphere R_out about the transferred CHIEF RAY,
            # plus the transferred tilt (the closure derived in the D1 note).
            _A, _B, _C, _D = _group_abcd(presc, wavelength)
            x_c_out = _A * x_c + _B * tilt_L
            y_c_out = _A * y_c + _B * tilt_M
            L_out = _C * x_c + _D * tilt_L
            M_out = _C * y_c + _D * tilt_M
            _rp = _tilt_ramp(E_exit.shape, cur_dx, wavelength,
                             L_out, M_out, x_c_out, y_c_out, -1)
            if _rp is not None:
                E_exit = E_exit * _rp
            if _sphere_ref:
                _cf = _sphere_parab_conversion(
                    E_exit.shape, cur_dx, wavelength, R_out, -1,
                    centre=(x_c_out, y_c_out))
                if _cf is not None:
                    E_exit = E_exit * _cf
            _ph = _radial_carrier_phase(
                E_exit.shape, cur_dx, cur_dx, wavelength, R_out, -1,
                centre=(x_c_out, y_c_out)) if np.isfinite(R_out) else None
            env_axis = E_exit if _ph is None else E_exit * _ph
            env = _shift_envelope(env_axis, -x_c_out, -y_c_out, cur_dx)
            if np.iscomplexobj(E_in) and env.dtype != E_in.dtype:
                env = env.astype(E_in.dtype)     # the float64 screens upcast
            tilt_L, tilt_M, x_c, y_c = L_out, M_out, x_c_out, y_c_out
        R = R_out
        w_stage, p_stage = _chain_envelope_stats(env, cur_dx)
        stage = {
            'name': _name,
            'R_in': R_use, 'R_out': R_out, 'dx': cur_dx,
            'w': w_stage, 'power': p_stage}
        if _tilted:
            stage.update({'L_out': tilt_L, 'M_out': tilt_M,
                          'x_c_out': x_c, 'y_c_out': y_c})
        if is_final and focus_readout is not None and final_leg != 'paraxial':
            # D3 / roadmap P5: report the quantity the 'auto' final-leg route
            # actually branches on, so a consumer can see how close their
            # design sits to na_exact_threshold WITHOUT reading a warning
            # (design 121 measures 0.405 against the 0.15 default -- clear).
            stage['na_exit'] = na_exit
        stages.append(stage)

    # ---- final leg to the target plane ----
    # niche D4: a DOE after the last lens group leaves its gaps unspent; they
    # are part of the distance to the target.  ``fd_own`` is the part of that
    # distance travelled at the CURRENT angle -- everything after the last DOE
    # plane -- and drives the chief ray and the piston; the transport gets the
    # whole thing.  Both equal ``final_distance`` without a DOE.
    fd_own = pend_own + final_distance
    final_distance = pend_gap + final_distance
    pend_gap = 0.0
    pend_own = 0.0
    # Both remaining paths are PARABOLA-referenced (the paraxial focus readout
    # and the plain reconstruct), so under carrier_reference='sphere' convert
    # the stored (sphere-referenced) envelope back first: the physical field is
    # env*exp(ikS) = [env*exp(ik(S-parab))]*exp(ik*parab).
    if _sphere_ref:
        _cf = _sphere_parab_conversion(
            np.shape(env), cur_dx, wavelength, R, +1,
            w_beam=_envelope_amp_radius(env, cur_dx, cur_dx))
        if _cf is not None:
            env = np.asarray(env) * _cf
    if focus_readout is not None:
        fr = dict(focus_readout)
        if 'dx_out' not in fr or 'N_out' not in fr:
            raise ValueError(
                "propagate_traced_carrier_chain: focus_readout must supply "
                "'dx_out' and 'N_out'.")
        # Only the PARAXIAL readout's own kwargs: a caller who supplies the
        # exact-path keys (``n_fine_cap`` / ``window_factor`` /
        # ``max_fine_launch_points`` / ``dx_fine`` / ``N_fine`` /
        # ``ram_budget``) and whose final leg then turns out to be low-NA used
        # to get a bare ``TypeError`` from here.  The exact-only keys are
        # inapplicable on this path, so drop them rather than crash.
        _par_kw = {kk: fr[kk] for kk in (
            'dx_out', 'N_out', 'standoff', 'centre_out', 'bandlimit')
            if kk in fr}
        # niche D2: the readout's Bluestein reconstruction is PERIODIC (its
        # period is N*dx of the co-moving grid at the stop plane, which has
        # COLLAPSED near a focus).  Record it so a caller -- in particular
        # propagate_traced_carrier_chain_multi, which accumulates K such
        # readouts onto one lattice -- can tell signal from replica instead
        # of silently summing wrapped copies of each spot.
        _pd = {}
        if not _tilted:
            field = carrier_referenced_focus_readout(
                env, R, final_distance, wavelength, cur_dx, _period_out=_pd,
                **_par_kw)
            if stages and 'period' in _pd:
                stages[-1]['readout_period'] = _pd['period']
            return TracedCarrierChainResult(np.asarray(field), None,
                                            float(fr['dx_out']), stages)
        # niche D1: read out in the chief-ray-tracking frame (the co-moving
        # grid collapses toward the focus, so a physically-placed off-axis
        # beam could not be held there), then restore the tilt theorem's ramp
        # and piston and express the window in ABSOLUTE (optical-axis)
        # coordinates.  ``centre_out`` is therefore the physical image-plane
        # position the caller wants the window centred on -- identical to the
        # historical meaning whenever the chief ray is on axis.
        _ob = _tilt_obliquity(tilt_L, tilt_M,
                              'propagate_traced_carrier_chain')
        x_t = x_c + tilt_L * fd_own * _ob
        y_t = y_c + tilt_M * fd_own * _ob
        _c_abs = tuple(float(v) for v in _par_kw.get('centre_out', (0.0, 0.0)))
        _par_kw['centre_out'] = (_c_abs[0] - x_t, _c_abs[1] - y_t)
        field = np.asarray(carrier_referenced_focus_readout(
            env, R, final_distance, wavelength, cur_dx, _period_out=_pd,
            **_par_kw))
        _nn, _dxo = int(fr['N_out']), float(fr['dx_out'])
        _u = (np.arange(_nn, dtype=np.float64) - _nn / 2) * _dxo \
            + _par_kw['centre_out'][0]
        _v = (np.arange(_nn, dtype=np.float64) - _nn / 2) * _dxo \
            + _par_kw['centre_out'][1]
        field = field * np.exp(
            1j * k0 * (tilt_L * _u[None, :] + tilt_M * _v[:, None]))
        field = field * np.exp(1j * k0 * fd_own * (_ob - 1.0))
        stages.append({'name': '<target>', 'target': True,
                       'L': tilt_L, 'M': tilt_M, 'x_c': x_t, 'y_c': y_t,
                       'centre_out': _c_abs, 'dx': _dxo,
                       'readout_period': _pd.get('period')})
        return TracedCarrierChainResult(field, None, _dxo, stages)

    if final_distance != 0.0:
        cr = propagate_carrier_referenced(env, R, final_distance, wavelength,
                                          cur_dx)
        env, R, cur_dx = cr.env, cr.R, cr.dx
        if isinstance(cur_dx, tuple):
            cur_dx = cur_dx[0]
        if isinstance(R, tuple):
            R = R[0]
        if _tilted:
            _ob = _tilt_obliquity(tilt_L, tilt_M,
                                  'propagate_traced_carrier_chain')
            x_c += tilt_L * fd_own * _ob
            y_c += tilt_M * fd_own * _ob
            env = np.asarray(env) * np.exp(
                1j * k0 * fd_own * (_ob - 1.0))
    field = carrier_referenced_reconstruct(env, R, wavelength, cur_dx)
    if _tilted:
        # the returned grid is CENTRED ON THE CHIEF RAY at (x_c, y_c); the
        # tilt ramp is referenced to that same point.
        _rp = _tilt_ramp(np.shape(field), cur_dx, wavelength,
                         tilt_L, tilt_M, 0.0, 0.0, +1)
        if _rp is not None:
            field = np.asarray(field) * _rp
        stages.append({'name': '<target>', 'target': True,
                       'L': tilt_L, 'M': tilt_M, 'x_c': x_c, 'y_c': y_c,
                       'dx': cur_dx})
    return TracedCarrierChainResult(np.asarray(field), float(R), cur_dx, stages)


# ===========================================================================
# Per-congruence chain orchestrator (niche D2 -- roadmap
# ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27 P1b)
# ===========================================================================
# :func:`propagate_traced_carrier_chain` propagates ONE congruence.  The
# shipping design-121 device is not one congruence: it is a Dammann DOE fan
# (8x4 orders, 480 um frame pitch, +-46 mrad) from an emitter array, i.e. K
# comparable-power beams at well-separated angles.  Pushed through the chain
# MULTIPLEXED that fan produced a populated, credible-looking frame lattice
# whose per-frame power was scrambled (0.47 +/- 0.51 % against a design
# 2.78 %/frame) with nothing raised and nothing warned -- the element's
# entrance->exit map names exactly that case as excluded
# (``_lens_traced.py``, ``carrier``'s validity paragraph).
#
# The orchestrator below runs each congruence through the SHIPPED-DEFAULT
# chain -- ``carrier_reference='sphere'`` + ``preserve_input_phase='remap'``
# + ``amplitude_model='ray_density'`` + ``remap_sampling='full'``, the v5.29
# configuration that put the single-beam 121 relay at its ideal-field ceiling
# -- and sums complex amplitudes on ONE common image grid.  It deliberately
# does NOT route through :func:`~lumenairy.apply_real_lens_traced_multi`,
# which cannot express that configuration (it forces
# ``preserve_input_phase=True`` and rejects ``amplitude_model='ray_density'``
# / ``fit_radius_beam_factor`` on its default ``reuse_prepared=True``), and it
# does not inherit ``apply_real_lens_traced_segmented``'s ``max_segments=32``
# cap -- which an 8x4 fan saturates exactly, leaving no headroom for the
# zero-order leak or a stray order.  Here K is bounded only by time and by the
# memory guard below.
#
# The DECENTRE / RE-OFFSET the roadmap asks for is done INTERNALLY and is not
# a separate mechanism: niche D1's ``r_in=TiltedCarrier(R, L, M, x0, y0)``
# already runs each congruence in ITS OWN chief-ray-tracking frame, so the
# +-46 mrad never enters the residual that
# ``_NONCOLLIMATED_RESID_THRESH = 0.02 rad`` bounds -- each run sits INSIDE
# the validated envelope instead of 2.3x outside it -- while the element still
# traces the beam at its true transverse position, so genuine off-axis
# aberration is kept rather than approximated away by a shift.  The RE-OFFSET
# is the readout: D1's tilted focus readout takes ``centre_out`` in ABSOLUTE
# (optical-axis) coordinates and returns the window carrying the absolute tilt
# ramp and path piston, so every congruence lands already expressed on the
# same absolute lattice and the recombination is a plain add.
#
# THE READOUT WINDOW IS PART OF THE PHYSICS HERE, not a display choice, and it
# is guarded as such (both guards were added after an adversarial pass found
# each of them failing in the field):
#
#   * TOO WIDE -- the readout's Bluestein reconstruction is PERIODIC (its
#     period is N*dx of the co-moving grid at the stop plane, which has
#     COLLAPSED near a focus).  A common window wider than one period fills
#     itself with wrapped copies of the congruence's own spot; accumulated on
#     the lattice those land on the NEIGHBOURING frames.  Measured on the
#     design-121 geometry (period 483 um), the natural 1.638 mm "span the
#     lattice" window put 9.1x the frame's real power into one congruence and
#     turned a 25/25/25/25 % design block into 11.1/22.2/22.2/44.6 %: the
#     v5.28 failure, reproduced by the fix for it.  ``on_replica`` (default
#     'error') refuses it, from the period the chain now reports, rather than
#     leaving it to a downstream UserWarning that any upstream
#     ``filterwarnings('ignore')`` silences.  Two corrections a second
#     adversarial pass forced, both about WHOSE window is at risk:
#       - the shared 'auto' window is sized from min(period) over ALL K
#         congruences (measured in a cheap 16-px probe pass), not from
#         congruence 0.  Design-121's per-order periods span 1.8 %, so sizing
#         from the first congruence made the DEFAULT raise on the acceptance
#         config, and made "does it run at all" depend on list order.
#       - the guard is a MULTIPLEXING guard: at K = 1 there is no neighbouring
#         frame to contaminate and the answer is exactly the chain's, so it
#         downgrades to a warning and 'auto' keeps the requested field of view
#         (an earlier cut silently returned zeros over 55 % of a K=1 grid).
#   * TOO NARROW -- a tile that clips the halo makes ``power_out``
#     window-dependent, and since the halo grows with field angle the clipped
#     fraction varies across a fan and reads as vignetting.  Measured on the
#     same fan: 102 um -> 410 um tiles moved the apparent per-order
#     "throughput" spread from 6.4e-3 to 5.0e-4.  So ``throughput`` is defined
#     at the CHAIN EXIT (window-independent) and the window's share is
#     reported separately as ``capture``, watched by ``on_readout_clip``.


class TracedCarrierChainMultiResult(NamedTuple):
    """Result of :func:`propagate_traced_carrier_chain_multi`.

    Attributes
    ----------
    field : ndarray, complex, shape ``(N_out, N_out)``
        The recombined image-plane field on the common output grid.  Pixel
        ``(j, i)`` sits at the ABSOLUTE transverse position
        ``centre + ((i - N_out/2) * dx, (j - N_out/2) * dx)`` -- the same
        convention :func:`carrier_referenced_focus_readout` uses for its
        ``centre_out``.

        Under ``recombine='incoherent'`` this array is
        ``sqrt(sum_k |A_k|^2)``: its MODULUS SQUARED is the incoherent power
        sum, and its phase is identically zero and carries no meaning.
    dx : float
        Output grid pitch (m) -- ``output_grid['dx_out']``.
    centre : (float, float)
        Absolute ``(x, y)`` centre of the common output grid (m).
    congruences : list of dict
        One entry per input congruence, in input order, with keys

        ``name``, ``weight`` (complex),
        ``carrier`` (the :class:`~lumenairy.TiltedCarrier` or float actually
        run), ``chief_ray`` (predicted absolute ``(x, y)`` at the target
        plane, m), ``tile_centre`` (the absolute centre of the readout window
        this congruence used, snapped to the common lattice), ``tile_origin``
        (its ``(row, col)`` index origin in the common grid), ``tile`` (the
        readout window size actually used, in pixels), ``readout_period``
        (SCALAR: the smaller axis of that congruence's Bluestein spatial
        period, m, or ``None`` -- the chain's ``stages`` carry the full
        ``(period_x, period_y)`` pair),

        ``power_in`` / ``power_exit`` / ``power_out`` (W-equivalent
        ``sum |E|^2 dx^2`` at the chain INPUT, at the last group's EXIT and in
        the readout WINDOW, all WITH ``weight`` applied),
        ``throughput`` (``power_exit / power_in`` -- the chain's real
        vignetting, WINDOW-INDEPENDENT), ``capture``
        (``power_out / power_exit`` -- how much of the delivered power the
        readout window holds, which is where a too-small tile shows up),
        ``clipped`` (fraction of ``power_out`` that fell outside the common
        grid) and ``stages`` (that run's
        :attr:`TracedCarrierChainResult.stages`).
    """

    field: np.ndarray
    dx: float
    centre: Tuple[float, float]
    congruences: list


_CONGRUENCE_KEYS = frozenset({'field', 'carrier', 'weight', 'name',
                              'doe_order'})
# focus_readout keys propagate_traced_carrier_chain understands.  ``N_out`` /
# ``centre_out`` are OWNED by the orchestrator (they define the common grid
# and the per-congruence tile), so they are not forwarded from output_grid.
_OUTPUT_GRID_PASSTHROUGH = ('standoff', 'bandlimit', 'window_factor',
                            'n_fine_cap', 'max_fine_launch_points',
                            'ram_budget', 'dx_fine', 'N_fine',
                            # niche C1 item 5: the readout-window guard's own
                            # message prescribes ``on_readout_window='warn'``
                            # (and ``readout_window_tol``) as THE remedy, and
                            # the guard fires from inside the exact final leg
                            # -- which is exactly where the design-121 fan runs.
                            # Without these two the prescribed remedy was
                            # unreachable from this entry point: measured,
                            # ``output_grid={'on_readout_window': 'warn'}``
                            # raised "output_grid has unknown key(s)".
                            'on_readout_window', 'readout_window_tol')

# niche C1 item 5: the keys ``propagate_traced_carrier_chain`` understands in
# ``focus_readout``.  It had NO whitelist, so a typo ('on_readout_windo') was
# silently accepted and the caller kept the hard ``'error'`` default while
# believing they had downgraded it -- the same silently-dropped-key class the
# ``congruences`` / ``output_grid`` / DOE-entry validators already refuse.
# ``dx_out`` / ``N_out`` are required; ``standoff`` and ``centre_out`` are the
# paraxial readout's; the rest belong to the exact leg and are DROPPED (not
# rejected) when the leg turns out to be paraxial -- see ``_par_kw``.
_FOCUS_READOUT_KEYS = frozenset(
    {'dx_out', 'N_out', 'centre_out'} | set(_OUTPUT_GRID_PASSTHROUGH))


def _doe_groups_for_order(groups, doe_order, where):
    """``groups`` with every DOE entry's ``'order'`` set to this congruence's.

    Returns the SAME list object when ``doe_order`` is None, which is what
    keeps every pre-D4 caller byte-identical.  With one DOE in the chain the
    value is that DOE's order (``m`` or ``(mx, my)``); with several it must be
    one order spec per DOE, in chain order -- so design 121's crossed pair
    takes ``doe_order=(m, n)``, order ``m`` on the x grating and ``n`` on the
    y one."""
    if doe_order is None:
        return groups
    idx = [i for i, g in enumerate(groups) if _is_doe_entry(g)]
    if not idx:
        raise ValueError(
            f"{where}: 'doe_order' was given but ``groups`` has no DOE entry "
            f"to apply it to.  Put the DOE in the chain as "
            f"{{'doe': rx['diffractives'][k]}} (niche D4), or drop the key.")
    if len(idx) == 1:
        orders = [doe_order]
    else:
        if (not isinstance(doe_order, (list, tuple, np.ndarray))
                or len(doe_order) != len(idx)):
            raise ValueError(
                f"{where}: this chain has {len(idx)} DOE entries, so "
                f"'doe_order' must be a sequence of {len(idx)} order specs "
                f"(one per DOE, in chain order); got {doe_order!r}.")
        orders = list(doe_order)
    out = list(groups)
    for j, i in enumerate(idx):
        entry = dict(out[i])
        entry['order'] = orders[j]
        out[i] = entry
    return out


def _normalise_congruence(spec, i, fn):
    """Normalise one entry of ``congruences`` to
    ``(field, carrier, weight, name, doe_order)``.

    Accepts a bare 2-D ndarray (an on-axis, collimated-carrier, unit-weight
    congruence) or a dict with keys ``field`` (required), ``carrier``
    (a float radius or a :class:`~lumenairy.TiltedCarrier`; default ``inf``),
    ``weight`` (complex, default 1), ``name`` and ``doe_order`` (this
    congruence's order at each DOE entry of ``groups``, niche D4; default
    None = leave every DOE on its own declared order).  Unknown keys RAISE
    rather than being ignored -- a silently-dropped ``'r_in'`` typo would
    produce exactly the plausible-looking wrong answer this entry point exists
    to prevent."""
    from .._validation import _check_2d_scalar_field
    if isinstance(spec, dict):
        unknown = set(spec) - _CONGRUENCE_KEYS
        if unknown:
            raise ValueError(
                f"{fn}: congruences[{i}] has unknown key(s) "
                f"{sorted(unknown)!r}; the accepted keys are "
                f"{sorted(_CONGRUENCE_KEYS)!r}.  (The carrier key is "
                f"'carrier', not 'r_in' -- a dropped carrier would run the "
                f"congruence on axis and silently misplace its frame.)")
        if 'field' not in spec:
            raise ValueError(
                f"{fn}: congruences[{i}] must supply a 'field' entry (the "
                f"beam ENVELOPE in this congruence's own chief-ray-tracking "
                f"frame).")
        field = spec['field']
        carrier = spec.get('carrier', np.inf)
        weight = spec.get('weight', 1.0)
        name = spec.get('name', f'congruence{i}')
        doe_order = spec.get('doe_order')
    else:
        field, carrier, weight, name = spec, np.inf, 1.0, f'congruence{i}'
        doe_order = None
    _check_2d_scalar_field(field, f"{fn}: congruences[{i}]['field']",
                           input_kind='field')
    weight = complex(weight)
    if not (np.isfinite(weight.real) and np.isfinite(weight.imag)):
        raise ValueError(
            f"{fn}: congruences[{i}]['weight'] must be finite, got "
            f"{weight!r}.")
    # parse (and therefore VALIDATE) the carrier here so a bad tilt fails
    # before any propagation work is done for ANY congruence -- including the
    # propagating-direction check, which otherwise would not fire until that
    # congruence's own chain run
    _R, L, M, _x0, _y0, tilted = _parse_chain_carrier(
        carrier, f"{fn}: congruences[{i}]['carrier']")
    if tilted:
        _tilt_obliquity(L, M, f"{fn}: congruences[{i}]['carrier']")
    return field, carrier, weight, str(name), doe_order


def _chain_chief_ray_at_target(groups, wavelength, carrier, final_distance,
                               fn):
    """Absolute ``(x, y)`` of a congruence's CHIEF RAY at the chain's target
    plane (m), plus its exit direction cosines ``(L, M)``.

    Mirrors :func:`propagate_traced_carrier_chain`'s own closure exactly, and
    needs nothing but it: on a free leg the chief ray advances by
    ``gap * (L, M) / cos(theta)`` with the tilt invariant, and through a group
    the pair ``(x_c, L)`` transforms as an ordinary paraxial ray under the
    group's air-to-air ABCD (:func:`_group_abcd`).  The carrier RADIUS never
    enters -- which is why this prediction cannot drift from the chain's own
    bookkeeping under a curvature change.  A per-group ``r_in`` override that
    is itself a :class:`~lumenairy.TiltedCarrier` re-seeds the congruence,
    exactly as the chain does.

    Used to place each congruence's readout tile on the common output
    lattice; :func:`propagate_traced_carrier_chain_multi` cross-checks it
    against the value the chain reports in ``stages[-1]``."""
    _R, L, M, x_c, y_c, tilted = _parse_chain_carrier(carrier, fn)
    pend = 0.0
    for gi, g in enumerate(groups):
        if _is_doe_entry(g):
            # niche D4: same closure as the chain's own DOE branch -- the
            # entry's gap_before is travelled at the PRE-DOE angle, the order
            # shifts the direction cosines exactly, and gap_after is spent on
            # the next leg (which is precisely what a hand fold gets wrong).
            (_nm, _gb, _ga, _dL, _dM, _amp, _org,
             _ord) = _normalise_doe_entry(g, gi, wavelength, fn)
            gap = pend + _gb
            if gap != 0.0 and tilted:
                ob = _tilt_obliquity(L, M, fn)
                x_c += L * gap * ob
                y_c += M * gap * ob
            L += _dL
            M += _dM
            tilted = tilted or bool(_dL or _dM)
            pend = _ga
            continue
        if isinstance(g, dict) and 'prescription' in g:
            presc = g['prescription']
            gap = float(g.get('gap_before', g.get('distance', 0.0)))
            g_r_in = g.get('r_in')
        elif isinstance(g, dict) and 'surfaces' in g:
            presc, gap, g_r_in = g, 0.0, None
        else:
            raise ValueError(
                f"{fn}: groups[{gi}] must be a prescription dict (with "
                f"'surfaces'), a group-spec dict (with 'prescription') or a "
                f"DOE entry (with 'doe'); got {type(g).__name__}.")
        gap = pend + gap
        pend = 0.0
        if gap != 0.0 and tilted:
            ob = _tilt_obliquity(L, M, fn)
            x_c += L * gap * ob
            y_c += M * gap * ob
        if g_r_in is not None:
            (_oR, oL, oM, ox, oy, otilt) = _parse_chain_carrier(
                g_r_in, f"{fn}: groups[{gi}]['r_in']")
            if otilt:
                L, M, x_c, y_c, tilted = oL, oM, ox, oy, True
        A, B, C, D = _group_abcd(presc, wavelength)
        x_c, y_c, L, M = (A * x_c + B * L, A * y_c + B * M,
                          C * x_c + D * L, C * y_c + D * M)
        tilted = tilted or bool(L or M or x_c or y_c)
    final_distance = pend + final_distance
    if final_distance != 0.0 and tilted:
        ob = _tilt_obliquity(L, M, fn)
        x_c += L * final_distance * ob
        y_c += M * final_distance * ob
    return float(x_c), float(y_c), float(L), float(M)


def _multi_mem_budget_mb(mem_budget_mb):
    """Resolve the orchestrator's memory ceiling in MB: an explicit
    ``mem_budget_mb`` wins, then ``LUMENAIRY_MEM_BUDGET_MB``, then the
    library's RAM budget (:func:`lumenairy.memory.get_ram_budget`)."""
    if mem_budget_mb is not None:
        val = float(mem_budget_mb)
        if not (val > 0.0):
            raise ValueError(
                "propagate_traced_carrier_chain_multi: mem_budget_mb must be "
                f"> 0 (MB), got {mem_budget_mb!r}.")
        return val
    from .fga import _env_mem_budget_mb
    env = _env_mem_budget_mb()
    if env is not None:
        return float(env)
    from ..memory import get_ram_budget
    return float(get_ram_budget()) / 1e6


def _multi_readout_period(stages):
    """Smallest-axis spatial period (m) of the readout's Bluestein
    reconstruction, as the chain recorded it in ``stages``
    (``'readout_period'``), or ``None`` when no readout period was reported.

    Beyond one period the readout returns periodic REPLICAS of the beam's own
    spot rather than signal (audit P11).  For a SINGLE congruence those
    replicas sit in the far window and are visibly wrong; for K congruences
    accumulated on one lattice they land ON TOP of the neighbouring frames and
    scramble their power -- indistinguishable from physics, and precisely the
    v5.28 multiplexed-fan failure this entry point exists to prevent."""
    for s in reversed(list(stages or ())):
        if isinstance(s, dict):
            p = s.get('readout_period')
            if p is not None:
                return min(float(p[0]), float(p[1]))
    return None


def _multi_chain_exit_power(stages):
    """``sum |env|^2 dx^2`` at the LAST chain plane that reports a power, as
    the chain recorded it (``stages[...]['power']``), or ``None``.

    This is the TILE-INDEPENDENT power the chain delivered -- it is measured
    on the co-moving grid at the last group's exit vertex, so it carries the
    real aperture vignetting the traced element applied and NOTHING about the
    readout window.  The window power (``power_out``) divided by it is the
    readout's capture fraction, which is what actually varies with field angle
    when a tile is too small to hold the halo.

    "Last plane that reports one" rather than "last GROUP" on purpose: a
    TRAILING DOE entry (niche D4) sits after the last lens group and scales
    the field by its order ``amplitude``, so it reports its own post-screen
    power.  Reading the lens group's instead would compare a pre-DOE exit
    power against a post-DOE readout -- ``capture`` low by |amplitude|^-2 and
    a readout-clipping diagnostic that fires on bookkeeping rather than on a
    clipped halo."""
    for s in reversed(list(stages or ())):
        if isinstance(s, dict) and 'power' in s:
            return float(s['power'])
    return None


def _multi_dispose(action, msg, exc=RuntimeError):
    """Apply an ``'error'`` / ``'warn'`` / ``'ignore'`` disposition to a
    detected readout fault.  ``'error'`` is the production-safe default
    everywhere in this entry point: a wrong multiplexed answer looks exactly
    like a right one.

    Thin alias for :func:`_guard_dispose` (niche D3 unified the guard-policy
    vocabulary across this module); ``stacklevel=3`` reproduces the frame this
    helper reported before that merge."""
    _guard_dispose(action, msg, exc, stacklevel=3)


# ``readout_tile='auto'``'s PERIOD-PROBE window, in pixels.  The Bluestein
# period is ``N_in * d_in`` of the readout's INPUT plane
# (:func:`~lumenairy.propagators.mft._asm_mft_spatial_period`), so it is a
# property of the co-moving grid the chain arrives on and does NOT depend on
# the output window: a 16-px probe measures exactly the period a full-grid
# readout would report, for none of the memory.  That is what makes an
# order-independent probe pass affordable -- probing on the common grid
# instead would be 4.3 GB per congruence on the design-121 16384-square
# lattice.
_MULTI_AUTO_PROBE_TILE = 16
# Defensive cap on ``'auto'``'s shrink-and-restart fallback (see the real pass
# below).  The probe pass measures the same quantity the real pass checks, so
# a resize there should never trigger; the cap keeps a library bug from
# looping instead of raising.
_MULTI_AUTO_MAX_RESIZE = 2


def propagate_traced_carrier_chain_multi(
    congruences,
    groups,
    wavelength: float,
    dx: float,
    *,
    output_grid: dict,
    recombine: str = 'coherent',
    readout_tile: Union[int, str, None] = 'auto',
    on_replica: str = 'error',
    on_readout_clip: str = 'warn',
    readout_capture_tol: float = 0.01,
    final_distance: float = 0.0,
    ray_subsample: int = 4,
    n_workers: Optional[int] = None,
    traced_kwargs: Optional[dict] = None,
    final_leg: str = 'auto',
    na_exact_threshold: float = 0.15,
    carrier_reference: str = 'sphere',
    mem_budget_mb: Optional[float] = None,
    on_mem_budget: str = 'error',
    on_multi_congruence: str = 'warn',
    multi_congruence_threshold: float = _MULTI_CONGRUENCE_MV_THRESH,
    on_na_proximity: str = 'warn',
    na_proximity_frac: float = 0.20,
    on_ram_cap: str = 'warn',
    on_rs_fine_clamp: str = 'warn',
    on_tilt_exact_grid: str = 'error',
    on_decentred_fit: str = 'warn',
    decentre_fit_frac: float = _DECENTRE_FIT_FRAC_DEFAULT,
    progress: Optional[Callable] = None,
) -> TracedCarrierChainMultiResult:
    """Run K INDEPENDENT congruences through one traced lens chain and
    recombine them on a common image grid (niche D2, roadmap
    ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27 P1b).

    One implementation serves both faces of the same feature:

    * **per-ORDER** -- a DOE fan is K comparable-power beams at well-separated
      angles, the case :func:`~lumenairy.apply_real_lens_traced`'s
      entrance->exit map explicitly excludes.  Pass one congruence per order,
      each carrying that order's tilt
      (``carrier=TiltedCarrier(R, L, M)``) and complex amplitude
      (``weight=``), and each is a single clean congruence.
    * **per-EMITTER** -- a 4x4 / 8x8 array is K decentred beams.  Pass one
      congruence per emitter with ``carrier=TiltedCarrier(R, 0, 0, x0, y0)``.
      Unlike :func:`~lumenairy.apply_real_lens_traced_multi` (which fixes
      ``preserve_input_phase=True`` and rejects
      ``amplitude_model='ray_density'`` on its default
      ``reuse_prepared=True``), every run here uses the SHIPPED VALIDATED
      configuration -- the v5.29 default triple that put the single-beam
      design-121 relay at its ideal-field ceiling.

    Each congruence is run in ITS OWN chief-ray-tracking frame (niche D1's
    tilted carrier, applied internally), so the split angle never enters the
    residual the traced correction is validated against
    (``_NONCOLLIMATED_RESID_THRESH = 0.02 rad``) -- the "decentre to on-axis,
    re-offset at the image plane" the roadmap asks for, without the consumer
    having to arrange it, and without approximating away the genuine off-axis
    aberration (the element still traces each beam at its true transverse
    position).

    The K runs are **sequential**, not simultaneous: peak memory is one chain
    run plus the accumulator, independent of K.

    Parameters
    ----------
    congruences : sequence
        The congruences, in any order.  Each entry is either a bare 2-D
        complex ndarray (an on-axis collimated-carrier, unit-weight
        congruence) or a dict with

        ``field`` : ndarray, required
            Beam ENVELOPE at the chain input plane, in THIS congruence's own
            chief-ray-tracking frame -- i.e. centred on the grid with its own
            carrier phase divided out.  All congruences share ``dx`` and the
            grid shape.
        ``carrier`` : float or :class:`~lumenairy.TiltedCarrier`, default
            ``inf``
            The congruence's entrance carrier, passed straight through as the
            chain's ``r_in``.  A float is the historical on-axis signed
            radius; a ``TiltedCarrier(R, L, M, x0, y0)`` carries the sphere
            plus a uniform tilt about the chief ray at ``(x0, y0)``.
        ``weight`` : complex, default 1
            Complex amplitude applied to this congruence's image-plane
            contribution -- a DOE order's diffraction amplitude and phase, or
            an emitter's drive.  Applied ONCE, at the image plane, so it does
            not perturb any beam-radius / NA measurement inside the chain.
        ``name`` : str, optional
            Label carried into the result's ``congruences`` diagnostics.
        ``doe_order`` : optional (niche D4)
            This congruence's order at the DOE entries of ``groups`` -- ``m``
            or ``(mx, my)`` for a single DOE, or one order spec per DOE (in
            chain order) when the design has several.  With it, the DOE lives
            IN the chain (``{'doe': rx['diffractives'][k]}``) and the fan is
            one call: the tilt, the DOE's axial gaps and the frame placement
            are all bookkept from the .zmx.  Without it every DOE stays on
            its own declared design order, and ``carrier=TiltedCarrier(...)``
            remains the way to seed an already-diffracted congruence (the
            pre-D4 hand-split form).

    groups, wavelength, dx, final_distance, ray_subsample, n_workers,
    traced_kwargs, final_leg, na_exact_threshold, carrier_reference
        Forwarded unchanged to :func:`propagate_traced_carrier_chain` for
        every congruence -- see that function.  The chain's shipped default
        configuration is used unless ``traced_kwargs`` overrides it.
    output_grid : dict, required
        The COMMON image grid every congruence is recombined on.  Must supply
        ``dx_out`` (m) and ``N_out`` (even int); may supply ``centre_out``
        (absolute ``(x, y)`` centre, default on-axis) and any of
        ``standoff`` / ``bandlimit`` / ``window_factor`` / ``n_fine_cap`` /
        ``max_fine_launch_points`` / ``ram_budget`` / ``dx_fine`` /
        ``N_fine`` / ``on_readout_window`` / ``readout_window_tol``, which are
        forwarded to the chain's ``focus_readout``.  Unknown keys RAISE.

        ``on_readout_window`` / ``readout_window_tol`` (niche C1) are the
        remedy the exact readout's own window guard prescribes, and the exact
        leg is where a design-121-class fan runs -- so they have to be
        reachable from here, not only from
        :func:`propagate_traced_carrier_chain`.

        A common grid (and therefore a focus readout) is REQUIRED: without it
        each congruence would land on its own co-moving grid and "sum the
        complex amplitudes" would not be defined without a resample.
    recombine : {'coherent', 'incoherent'}, default 'coherent'
        ``'coherent'`` sums complex amplitudes -- the physical answer for a
        common-source DOE fan or a phase-locked emitter array, and the one
        that reproduces interference where congruences overlap.
        ``'incoherent'`` sums ``|A_k|^2`` instead and returns
        ``sqrt(sum |A_k|^2)`` (real, phase identically zero) -- the
        mutually-incoherent-source answer, and a useful control: where the
        two differ, interference is doing something.
    readout_tile : int or 'auto' or None, default ``'auto'``
        Read each congruence out on a ``readout_tile``-square window centred
        on its own predicted chief ray instead of on the whole common grid,
        and accumulate.  The tile centre is SNAPPED to the common lattice
        (``centre_out + integer * dx_out``), so accumulation is an exact
        index-offset add with no resampling and no sub-pixel interpolation.
        An explicit value must be even and ``<= N_out``.

        This is the memory/time lever for a wide, sparse frame lattice: a
        32-order fan on a 480 um pitch needs a common grid spanning several
        mm, but each frame only needs a window around its own spot.

        **It is also an ACCURACY lever, and for a wide lattice the accurate
        choice** -- which is why the default is ``'auto'`` rather than the
        whole grid.  The readout's final Bluestein zoom reconstructs the
        co-moving grid's spectrum, and that reconstruction is PERIODIC with
        period ``N * dx`` of the (collapsed, near-focus) co-moving grid, which
        the chain reports as ``stages[...]['readout_period']``.  A common
        window several mm wide can exceed one such period, and the samples
        beyond it are periodic REPLICAS of the congruence's own spot, not
        signal -- they land on top of the NEIGHBOURING frames and scramble
        their power, which is the v5.28 failure mode all over again.  Measured
        on the design-121 geometry (period 483 um): a naive 1.638 mm common
        window put 9.1x the frame's real power into a single congruence and
        turned a 4-frame block whose design shares are 25/25/25/25 % into
        11.1/22.2/22.2/44.6 %.

        The three modes:

        ``'auto'`` (default)
            For ``K > 1``: a cheap PROBE PASS first reads every congruence out
            on a 16-px window purely to measure its Bluestein period (the
            period is ``N_in * dx_in`` of the readout's INPUT plane, so it is
            the same number a full-size window would report), then sizes ONE
            shared window to the largest even tile that fits the SHORTEST
            period over ALL K congruences, says so with a ``RuntimeWarning``,
            and runs the real pass on it.  Sizing from the minimum over all K
            is what makes the result INDEPENDENT of the order the congruences
            were listed in -- an adversarial pass killed an earlier version
            that sized from congruence 0 alone, because design-121's per-order
            periods span 1.8 % and a later, shorter-period congruence then hit
            the replica guard (so whether the default ran at all depended on
            list order).

            Cost: ``2K`` chain runs instead of ``K``.  The warning names the
            exact ``readout_tile=<int>`` to pass to skip the probe pass; a
            common grid that already fits inside every period keeps the full
            grid (still ``2K`` runs, since only a probe can establish that).

            For ``K == 1`` there is no probe and no re-size: the single
            congruence IS ``propagate_traced_carrier_chain``'s answer on the
            requested grid, so the full requested field of view is honoured
            (see ``on_replica``).
        ``int``
            Use exactly this window.  Still period-checked per congruence, and
            one chain run per congruence.
        ``None``
            Read every congruence out on the full common grid (no lattice
            snap).  The historical behaviour, and correct only while the
            common window fits inside one period; ``on_replica`` decides what
            happens when it does not.

        ``None`` (and ``'auto'`` when it keeps the full grid) reads on axis;
        an explicit ``int`` window -- including ``int == N_out`` -- is snapped
        onto the congruence's own chief ray, so a frame that lands off the
        common grid shows up in ``congruences[k]['clipped']`` instead of being
        silently re-centred onto empty sky.

        Size an explicit tile to hold one frame's spot AND its halo -- a tile
        that is too small clips a field-angle-dependent fraction of the halo,
        which reads as physical vignetting and is not (see
        ``on_readout_clip``).  ``congruences[k]['tile']`` reports the window
        actually used.
    on_replica : {'error', 'warn', 'ignore'}, default 'error'
        What to do when a congruence's readout window exceeds one spatial
        period of its Bluestein reconstruction, i.e. when the window contains
        periodic replicas.  The default REFUSES: an accumulated replica is
        indistinguishable from signal in the output, so continuing would
        return a populated, credible-looking frame lattice with the wrong
        per-frame power.  The error names the largest tile that fits.

        **This is a MULTIPLEXING guard.**  With ``K == 1`` the recombination
        is a no-op and the replicas stay inside the one congruence's own
        window -- there is no neighbouring frame for them to land on, and the
        returned field is exactly what
        :func:`propagate_traced_carrier_chain` returns for the same readout
        (which is the contract this entry point must keep, so that the shipped
        single-beam design-121 acceptance cannot move by being routed through
        here).  So at ``K == 1`` this check is DOWNGRADED to a
        ``RuntimeWarning`` naming the period, and ``'auto'`` does not shrink
        the requested grid; the chain's own
        :func:`~lumenairy.angular_spectrum_propagate_mft` warning fires
        alongside it.  ``'ignore'`` silences both.
    on_readout_clip : {'error', 'warn', 'ignore'}, default 'warn'
        What to do when a congruence's readout window captures less (or more)
        of the power the chain delivered than ``readout_capture_tol`` allows
        -- ``congruences[k]['capture'] = power_out / power_exit``.  Under 1
        the window is clipping the halo; over 1 replicas are ADDING power.
        Either way ``power_out`` stops being a window-independent observable,
        and because the halo grows with field angle the clipped fraction
        varies across a fan and masquerades as vignetting.  Use
        ``congruences[k]['throughput']``, which is measured at the chain exit
        and is window-independent, for the real vignetting.
    readout_capture_tol : float, default 0.01
        Tolerance for the ``on_readout_clip`` check (fractional).
    mem_budget_mb : float, optional
        Ceiling (MB) for the orchestrator's OWN arrays -- the accumulator plus
        one readout tile plus one scratch copy.  Defaults to
        ``LUMENAIRY_MEM_BUDGET_MB`` when set, else the library RAM budget
        (:func:`lumenairy.memory.get_ram_budget`).  Each per-congruence chain
        run keeps its own budget via ``output_grid['ram_budget']``; because
        the runs are sequential that peak does not multiply by K.
    on_mem_budget : {'error', 'warn', 'ignore'}, default 'error'
        What to do when the estimate exceeds the budget.  The default FAILS
        LOUDLY rather than reporting a degraded number from an unattended
        batch run.
    on_multi_congruence, multi_congruence_threshold, on_na_proximity, na_proximity_frac, on_ram_cap, on_rs_fine_clamp, on_tilt_exact_grid, on_decentred_fit, decentre_fit_frac
        The niche-D3 (plus D6's ``on_tilt_exact_grid`` and
        ``on_decentred_fit``) guard rails,
        forwarded VERBATIM to every per-congruence
        :func:`propagate_traced_carrier_chain` run -- see that function for
        what each one watches and its default.  They are worth setting to
        ``'error'`` here in particular: this entry point is what a batch
        production fan run calls, and each of them guards a silent degradation
        (a congruence that is itself multi-valued, a near-miss exit-NA route
        flip, a RAM-degraded readout grid, a broken ray-pitch contract).
        ``on_multi_congruence`` here checks each congruence INDIVIDUALLY, which
        is the point: a correctly-decomposed fan has K clean single
        congruences and stays silent, and a congruence that is still a
        superposition is exactly the decomposition bug this route exists to
        prevent.  ``readout_tile='auto'``'s period-probe pass suppresses the
        two entry guards (it re-runs the identical input, so they would fire
        twice for no extra information).
    progress : callable, optional
        Called as ``progress(k, K, name)`` before each congruence's run.
        ``readout_tile='auto'``'s probe pass calls it too, with ``name``
        suffixed ``' (period probe)'``, so a 2K-run 'auto' call is not
        mistaken for a stalled K-run one.

    Returns
    -------
    TracedCarrierChainMultiResult
        ``(field, dx, centre, congruences)`` -- see the class docstring.

    Raises
    ------
    RuntimeError
        If a congruence's readout window exceeds one spatial period of its
        Bluestein reconstruction (``on_replica='error'``, the default, and
        ``K > 1`` -- see ``on_replica`` for the ``K == 1`` downgrade), or if
        its window captures the wrong fraction of the delivered power and
        ``on_readout_clip='error'``.  Both are readout-window faults that
        would otherwise return a plausible-looking wrong per-frame power.
        The default ``readout_tile='auto'`` sizes the window so that the
        first of these does not arise.
    NotImplementedError
        If ``groups`` ends in a DOE entry AND the run routes onto the exact
        high-NA final leg -- that leg lands the target plane from inside the
        last lens group, so it cannot also apply a trailing screen.  A TILTED
        congruence on the exact leg is supported since niche D6; what it can
        still refuse (a ``RuntimeError`` from ``on_tilt_exact_grid``, not this)
        is a fine grid too coarse to sample the widened, axis-centred retrace
        window -- see ``on_tilt_exact_grid``.

    Notes
    -----
    **Why not** :func:`~lumenairy.apply_real_lens_traced_segmented` --
    its ``max_segments`` default is 32, which an 8x4 fan saturates exactly
    (no headroom for the zero-order leak or a stray order), and its
    multi-segment path routes through ``apply_real_lens_traced_multi``, i.e.
    back into the contract that cannot express the validated configuration.
    Here K is capped only by time (K sequential chain runs, or 2K under the
    default ``readout_tile='auto'`` -- see that parameter) and by
    ``mem_budget_mb``.

    **Power bookkeeping** -- three DIFFERENT planes, deliberately kept apart
    (conflating them is how a readout-window artefact gets reported as
    physics):

    * ``['power_in']`` = ``|weight|^2 sum|field|^2 dx^2`` at the chain INPUT.
    * ``['power_exit']`` = the power in the last group's exit envelope, on the
      co-moving grid, with ``weight`` applied -- and, when ``groups`` ends in
      a DOE entry (niche D4), with that screen's order ``amplitude`` applied
      too, since it is what the chain actually delivered.  This is what the
      chain DELIVERED: it carries the traced element's real aperture
      vignetting and nothing about the readout.
      ``['throughput'] = power_exit / power_in`` is therefore
      WINDOW-INDEPENDENT and is the number to quote for vignetting.
    * ``['power_out']`` = ``sum|A_k|^2 dx_out^2`` over that congruence's
      readout WINDOW, and ``['capture'] = power_out / power_exit``.  This one
      DOES depend on the window: a tile smaller than the halo clips it, and
      because the halo grows with field angle the clipped fraction varies
      across a fan and looks exactly like vignetting.  Measured on the
      design-121 32-order fan at N=1024, changing only the tile
      (102 um -> 410 um) moved the apparent per-order "throughput" spread from
      6.4e-3 to 5.0e-4 -- a 13x collapse, i.e. nearly all of it was the tile.
      ``on_readout_clip`` watches ``capture`` for exactly this reason.

    ``['clipped']`` reports the fraction of ``power_out`` that landed outside
    the COMMON grid -- non-zero means the common grid is too small for the
    lattice, not that power was lost.  ``['readout_period']`` is the
    congruence's Bluestein period (m); ``['tile']`` is the window actually
    used.

    Examples
    --------
    A two-order fan through a relay, recombined coherently::

        >>> import numpy as np, lumenairy as la           # doctest: +SKIP
        >>> fan = [{'field': env, 'name': 'm=-1', 'weight': 0.5,
        ...         'carrier': la.TiltedCarrier(R, -0.046, 0.0)},
        ...        {'field': env, 'name': 'm=+1', 'weight': 0.5,
        ...         'carrier': la.TiltedCarrier(R, +0.046, 0.0)}]
        >>> res = la.propagate_traced_carrier_chain_multi(
        ...     fan, groups, 1.31e-6, dx, final_distance=fd,
        ...     output_grid={'dx_out': 0.2e-6, 'N_out': 4096},
        ...     readout_tile=512, final_leg='paraxial')   # doctest: +SKIP
    """
    fn = 'propagate_traced_carrier_chain_multi'
    if recombine not in ('coherent', 'incoherent'):
        raise ValueError(
            f"{fn}: recombine must be 'coherent' or 'incoherent', got "
            f"{recombine!r}.")
    if on_mem_budget not in ('error', 'warn', 'ignore'):
        raise ValueError(
            f"{fn}: on_mem_budget must be 'error', 'warn' or 'ignore', got "
            f"{on_mem_budget!r}.")
    if on_replica not in ('error', 'warn', 'ignore'):
        raise ValueError(
            f"{fn}: on_replica must be 'error', 'warn' or 'ignore', got "
            f"{on_replica!r}.")
    if on_readout_clip not in ('error', 'warn', 'ignore'):
        raise ValueError(
            f"{fn}: on_readout_clip must be 'error', 'warn' or 'ignore', got "
            f"{on_readout_clip!r}.")
    # niche D3 guard rails, forwarded verbatim to every per-congruence chain
    # run (see propagate_traced_carrier_chain for what each one watches).
    _check_guard_action('on_multi_congruence', on_multi_congruence, fn)
    _check_guard_action('on_na_proximity', on_na_proximity, fn)
    _check_guard_action('on_ram_cap', on_ram_cap, fn)
    _check_guard_action('on_rs_fine_clamp', on_rs_fine_clamp, fn)
    _check_guard_action('on_tilt_exact_grid', on_tilt_exact_grid, fn)
    _check_guard_action('on_decentred_fit', on_decentred_fit, fn)
    if not (np.isfinite(decentre_fit_frac) and decentre_fit_frac >= 0.0):
        raise ValueError(
            f"{fn}: decentre_fit_frac must be a finite non-negative number of "
            f"beam amplitude radii, got {decentre_fit_frac!r}.")
    readout_capture_tol = float(readout_capture_tol)
    if not (np.isfinite(readout_capture_tol) and readout_capture_tol > 0.0):
        raise ValueError(
            f"{fn}: readout_capture_tol must be a finite fraction > 0, got "
            f"{readout_capture_tol!r}.")
    if not isinstance(output_grid, dict):
        raise ValueError(
            f"{fn}: output_grid must be a dict with 'dx_out' and 'N_out' "
            f"(the COMMON image grid every congruence recombines on); got "
            f"{type(output_grid).__name__}.")
    missing = [k for k in ('dx_out', 'N_out') if k not in output_grid]
    if missing:
        raise ValueError(
            f"{fn}: output_grid is missing {missing!r}.  A common image grid "
            f"is required -- without a focus readout each congruence would "
            f"land on its own co-moving grid and the coherent sum would need "
            f"a resample.")
    unknown = set(output_grid) - ({'dx_out', 'N_out', 'centre_out'}
                                  | set(_OUTPUT_GRID_PASSTHROUGH))
    if unknown:
        raise ValueError(
            f"{fn}: output_grid has unknown key(s) {sorted(unknown)!r}; "
            f"accepted keys are ['dx_out', 'N_out', 'centre_out'] plus "
            f"{list(_OUTPUT_GRID_PASSTHROUGH)!r}.")
    dx_out = float(output_grid['dx_out'])
    N_out = int(output_grid['N_out'])
    if not (np.isfinite(dx_out) and dx_out > 0.0):
        raise ValueError(f"{fn}: output_grid['dx_out'] must be > 0, got "
                         f"{output_grid['dx_out']!r}.")
    if N_out < 2 or N_out % 2 != 0:
        raise ValueError(
            f"{fn}: output_grid['N_out'] must be an even integer >= 2 (the "
            f"grid centre convention is (i - N_out/2) * dx_out), got "
            f"{output_grid['N_out']!r}.")
    centre = tuple(float(v) for v in output_grid.get('centre_out', (0.0, 0.0)))
    if len(centre) != 2 or not all(np.isfinite(v) for v in centre):
        raise ValueError(
            f"{fn}: output_grid['centre_out'] must be a finite (x, y) pair, "
            f"got {output_grid.get('centre_out')!r}.")
    if readout_tile is None:
        tile_mode, n_tile = 'full', N_out
    elif isinstance(readout_tile, str):
        if readout_tile != 'auto':
            raise ValueError(
                f"{fn}: readout_tile must be an even integer in [2, N_out="
                f"{N_out}], 'auto' (size it from the readout's own Bluestein "
                f"period) or None (the whole common grid), got "
                f"{readout_tile!r}.")
        tile_mode, n_tile = 'auto', N_out
    else:
        tile_mode = 'fixed'
        n_tile = int(readout_tile)
        if n_tile < 2 or n_tile % 2 != 0 or n_tile > N_out:
            raise ValueError(
                f"{fn}: readout_tile must be an even integer in [2, N_out="
                f"{N_out}], got {readout_tile!r}.")
    if not np.isfinite(final_distance):
        raise ValueError(
            f"{fn}: final_distance must be finite, got {final_distance!r}.")

    groups = list(groups)
    if not groups:
        raise ValueError(f"{fn}: groups is empty -- nothing to propagate "
                         f"through.")
    specs = [_normalise_congruence(c, i, fn)
             for i, c in enumerate(congruences)]
    if not specs:
        raise ValueError(
            f"{fn}: congruences is empty.  Pass at least one congruence; K=1 "
            f"reduces to propagate_traced_carrier_chain.")
    # niche D4: each congruence's own DOE order, resolved ONCE.  Without a
    # 'doe_order' anywhere this is ``groups`` itself, K times over, so the
    # pre-D4 behaviour is untouched (identity, not a copy).
    groups_k = [_doe_groups_for_order(groups, s[4],
                                      f"{fn}: congruences[{i}] ({s[3]!r})")
                for i, s in enumerate(specs)]
    shape0 = np.shape(specs[0][0])
    for i, (fld, _c, _w, nm, _do) in enumerate(specs):
        if np.shape(fld) != shape0:
            raise ValueError(
                f"{fn}: congruences[{i}] ({nm!r}) has field shape "
                f"{np.shape(fld)} but congruences[0] has {shape0}.  All "
                f"congruences share the chain input grid (one dx, one "
                f"shape).")

    K = len(specs)
    half = 0.5 * N_out * dx_out

    # ---- chief rays: analytic, cheap, and they place every window ----------
    # Computed up front, BEFORE any propagation, so an off-grid congruence is
    # reported once rather than once per (probe, real) run below.
    chief = []
    for k, (fld, carrier, weight, name, _do) in enumerate(specs):
        x_pred, y_pred, L_out, M_out = _chain_chief_ray_at_target(
            groups_k[k], wavelength, carrier, final_distance,
            f"{fn}: congruences[{k}] ({name!r})")
        if abs(x_pred - centre[0]) > half or abs(y_pred - centre[1]) > half:
            import warnings
            warnings.warn(
                f"{fn}: congruence {name!r} lands at "
                f"({x_pred * 1e3:.4f}, {y_pred * 1e3:.4f}) mm, outside the "
                f"common output grid (centre "
                f"({centre[0] * 1e3:.4f}, {centre[1] * 1e3:.4f}) mm, "
                f"half-extent {half * 1e3:.4f} mm).  Its contribution will be "
                f"clipped away; raise N_out or move centre_out.",
                RuntimeWarning, stacklevel=2)
        chief.append((x_pred, y_pred, L_out, M_out))

    def _window(k, n_win):
        """Lattice-snapped placement of congruence ``k``'s ``n_win``-square
        readout window, plus the ``focus_readout`` it implies.  A window that
        spans the WHOLE common grid cannot also be snapped onto the chief ray
        (it would hang off the grid), so the full-grid modes keep the
        historical on-axis placement.  An EXPLICIT ``readout_tile == N_out``
        still snaps: the caller asked for that window, and a frame that then
        hangs off the common grid is reported as ``clipped`` rather than
        silently re-centred onto empty sky."""
        if n_win >= N_out and tile_mode in ('full', 'auto'):
            mx = my = 0
            tile_centre = centre
        else:
            mx = int(round((chief[k][0] - centre[0]) / dx_out))
            my = int(round((chief[k][1] - centre[1]) / dx_out))
            tile_centre = (centre[0] + mx * dx_out,
                           centre[1] + my * dx_out)
        fr = {kk: output_grid[kk] for kk in _OUTPUT_GRID_PASSTHROUGH
              if kk in output_grid}
        fr['dx_out'] = dx_out
        fr['N_out'] = int(n_win)
        fr['centre_out'] = tile_centre
        return mx, my, tile_centre, fr

    def _run(k, fr, quiet=False):
        # ``quiet`` is the 'auto' PERIOD-PROBE pass: it runs the same chain a
        # second time only to read back the Bluestein period, so the D3 entry
        # guards are suppressed there (they would fire twice per congruence,
        # and cost two extra full-grid passes each, for no extra information --
        # the REAL pass below runs them on the identical input).
        return propagate_traced_carrier_chain(
            specs[k][0], groups_k[k], wavelength, dx, r_in=specs[k][1],
            ray_subsample=ray_subsample, n_workers=n_workers,
            traced_kwargs=traced_kwargs, final_distance=final_distance,
            focus_readout=fr, final_leg=final_leg,
            na_exact_threshold=na_exact_threshold,
            carrier_reference=carrier_reference,
            on_multi_congruence=('ignore' if quiet else on_multi_congruence),
            multi_congruence_threshold=multi_congruence_threshold,
            on_na_proximity=('ignore' if quiet else on_na_proximity),
            na_proximity_frac=na_proximity_frac,
            on_ram_cap=on_ram_cap, on_rs_fine_clamp=on_rs_fine_clamp,
            on_tilt_exact_grid=on_tilt_exact_grid,
            on_decentred_fit=('ignore' if quiet else on_decentred_fit),
            decentre_fit_frac=decentre_fit_frac)

    def _fit(period):
        """Largest EVEN window (px) that fits inside one spatial period."""
        return max(2, min(N_out, int(2 * int(np.floor(period
                                                      / (2.0 * dx_out))))))

    # ---- memory guard: the orchestrator's OWN arrays -----------------------
    budget_mb = _multi_mem_budget_mb(mem_budget_mb)
    est_dtype = np.result_type(np.complex64,
                               *[np.asarray(s[0]).dtype for s in specs])
    # floor at complex128: the readout's Bluestein zoom returns double even
    # for a complex64 input, so the accumulator is sized by the OUTPUT
    itemsize = max(int(np.dtype(est_dtype).itemsize), 16)

    def _mem_check(n_win):
        est_bytes = float(N_out) ** 2 * itemsize          # accumulator
        if recombine == 'incoherent':
            est_bytes += float(N_out) ** 2 * 8.0      # + the float64 |A|^2 sum
        est_bytes += 2.0 * float(n_win) ** 2 * itemsize   # tile + one scratch
        est_mb = est_bytes / 1e6
        if est_mb <= budget_mb:
            return
        msg = (f"{fn}: the recombination arrays need ~{est_mb:.0f} MB "
               f"(N_out={N_out} accumulator + a {n_win}-square tile at "
               f"{itemsize}-byte complex) against a "
               f"{budget_mb:.0f} MB budget.  Shrink N_out or dx_out's span, "
               f"pass a smaller readout_tile, raise mem_budget_mb / "
               f"LUMENAIRY_MEM_BUDGET_MB, or set on_mem_budget='warn'.  (The "
               f"K={K} chain runs are SEQUENTIAL, so this estimate "
               f"does not scale with K.)")
        if on_mem_budget == 'error':
            raise MemoryError(msg)
        if on_mem_budget == 'warn':
            import warnings
            warnings.warn(msg, RuntimeWarning, stacklevel=3)

    # ---- 'auto' PASS 0: size ONE window from the SHORTEST period over ALL K -
    # An adversarial pass killed the first cut of this, which sized the shared
    # window from congruence 0 alone and then locked it: every LATER congruence
    # with a shorter period hit the replica guard, so whether the DEFAULT ran
    # at all depended on the order the caller happened to list the congruences
    # in (design-121's per-order periods span 1.8 %, and the 32-order fan
    # raised on its second congruence).  The window is therefore sized from
    # min(period) over ALL congruences, measured in a cheap probe pass -- the
    # period is a property of the readout's INPUT plane, so a 16-px probe
    # measures the same number the real window will.
    n_probe = min(N_out, _MULTI_AUTO_PROBE_TILE)
    probing = (tile_mode == 'auto' and K > 1)
    _mem_check(n_probe if probing else n_tile)
    if probing:
        periods = []
        for k in range(K):
            if progress is not None:
                progress(k, K, f"{specs[k][3]} (period probe)")
            periods.append(_multi_readout_period(
                _run(k, _window(k, n_probe)[3], quiet=True).stages))
        measured = [(p, k) for k, p in enumerate(periods) if p is not None]
        if measured:
            p_min, k_min = min(measured)
            n_safe = _fit(p_min)
            if n_safe < n_tile:
                import warnings
                n_over = sum(1 for p, _k in measured
                             if p * (1.0 + 1e-9) < n_tile * dx_out)
                warnings.warn(
                    f"{fn}: readout_tile='auto' -- the full common window "
                    f"({n_tile} x {dx_out * 1e6:.4f} um = "
                    f"{n_tile * dx_out * 1e3:.4f} mm) exceeds one spatial "
                    f"period of the readout's Bluestein reconstruction for "
                    f"{n_over} of the {K} congruences (shortest: "
                    f"{specs[k_min][3]!r}, {p_min * 1e3:.4f} mm, "
                    f"{n_tile * dx_out / p_min:.3f}x), so it would accumulate "
                    f"periodic REPLICAS of each congruence's spot on top of "
                    f"its neighbours.  Sizing every congruence's readout "
                    f"window to {n_safe} px ({n_safe * dx_out * 1e6:.1f} um), "
                    f"snapped to the common lattice.  The size comes from the "
                    f"MINIMUM period over ALL {K} congruences, so it does not "
                    f"depend on the order they were listed in.  Pass "
                    f"readout_tile={n_safe} to skip the probe pass (which "
                    f"halves the chain runs, {2 * K} -> {K}), or "
                    f"readout_tile=None with on_replica='ignore' for the "
                    f"historical (replica-contaminated) full-grid readout.",
                    RuntimeWarning, stacklevel=2)
                n_tile = n_safe
        if n_tile != n_probe:
            _mem_check(n_tile)

    # ---- PASS 1: the real runs ---------------------------------------------
    resizes_left = _MULTI_AUTO_MAX_RESIZE
    while True:
        acc = None
        acc_i = None
        acc_dtype = None
        infos = []
        restart = False
        for k, (fld, carrier, weight, name, _do) in enumerate(specs):
            if progress is not None:
                progress(k, K, name)
            x_pred, y_pred, L_out, M_out = chief[k]
            mx, my, tile_centre, fr = _window(k, n_tile)
            res = _run(k, fr)
            # ---- REPLICA GUARD (orchestrator-owned) ------------------------
            # The readout's Bluestein reconstruction is periodic; beyond one
            # period the window holds wrapped copies of THIS congruence's own
            # spot.  Summed onto the common lattice they land on the
            # NEIGHBOURING frames and scramble their power, which is exactly
            # the v5.28 failure.  angular_spectrum_propagate_mft warns about
            # it, but a warning is not a guard: it is silenced by any
            # ``filterwarnings('ignore')`` upstream and it does not fire at all
            # for a window that is safe for one congruence and not another.
            # Check it HERE, from the period the chain reports, every time.
            period = _multi_readout_period(res.stages)
            win = n_tile * dx_out
            if period is not None and win > period * (1.0 + 1e-9):
                n_safe = _fit(period)
                if probing and n_safe < n_tile and resizes_left > 0:
                    # DEFENSIVE ONLY.  Pass 0 measured this same period on a
                    # probe window, so this cannot fire unless the period
                    # depends on the output window after all; shrink and
                    # restart rather than raise, and keep every congruence on
                    # ONE window.
                    resizes_left -= 1
                    import warnings
                    warnings.warn(
                        f"{fn}: readout_tile='auto' -- congruence {name!r} "
                        f"reports a shorter Bluestein period "
                        f"({period * 1e3:.4f} mm) on the real "
                        f"{n_tile}-px window than the probe pass measured; "
                        f"re-sizing to {n_safe} px and re-running all {K} "
                        f"congruences.  (The period is expected to be "
                        f"window-independent, so this path indicates a "
                        f"library bug -- please report it.)",
                        RuntimeWarning, stacklevel=2)
                    n_tile = n_safe
                    restart = True
                    break
                msg = (
                    f"{fn}: congruence {name!r} was read out on a "
                    f"{n_tile} x {dx_out * 1e6:.4f} um = {win * 1e3:.4f} mm "
                    f"window, which EXCEEDS one spatial period "
                    f"({period * 1e3:.4f} mm, {win / period:.3f}x) of that "
                    f"readout's Bluestein reconstruction.  The samples beyond "
                    f"one period are periodic REPLICAS of this congruence's "
                    f"own spot, not signal")
                if K == 1:
                    _multi_dispose(
                        'ignore' if on_replica == 'ignore' else 'warn',
                        msg + f".  With a single congruence they stay inside "
                              f"this congruence's own window -- there is no "
                              f"neighbouring frame to contaminate -- so this "
                              f"IS the field propagate_traced_carrier_chain "
                              f"returns for this readout, and the guard is "
                              f"downgraded to a warning so that K=1 keeps "
                              f"reducing to the chain.  Pass "
                              f"readout_tile={n_safe} (the largest even "
                              f"window that fits one period here) for a "
                              f"replica-free readout, or "
                              f"on_replica='ignore' to silence this.")
                else:
                    _multi_dispose(
                        on_replica,
                        msg + f": accumulated onto the common lattice they "
                              f"land on the NEIGHBOURING frames and scramble "
                              f"their power -- a populated, credible-looking "
                              f"frame lattice with the wrong per-frame power, "
                              f"which is the v5.28 multiplexed-fan failure "
                              f"this entry point exists to prevent.  Pass "
                              f"readout_tile <= {n_safe} (the largest even "
                              f"window that fits one period here) or "
                              f"readout_tile='auto' (which sizes the window "
                              f"from the SHORTEST period over all "
                              f"{K} congruences); on_replica='warn'/'ignore' "
                              f"downgrades this check.")
            # ANTI-DRIFT: the tile was placed from the analytic prediction,
            # so a divergence between it and the chain's own chief-ray
            # bookkeeping would silently misplace a frame.  The chain
            # reports its tracked chief ray in the final '<target>' stage
            # whenever the carrier is tilted; cross-check it rather than
            # trust two copies of one formula.
            _last = res.stages[-1] if res.stages else {}
            if _last.get('target') and 'x_c' in _last:
                _dxc = abs(float(_last['x_c']) - x_pred)
                _dyc = abs(float(_last['y_c']) - y_pred)
                _tol = max(1e-9, 1e-6 * float(np.hypot(x_pred, y_pred)))
                if _dxc > _tol or _dyc > _tol:
                    raise RuntimeError(
                        f"{fn}: congruence {name!r} -- the chain's tracked "
                        f"chief ray ({float(_last['x_c']) * 1e6:.4f}, "
                        f"{float(_last['y_c']) * 1e6:.4f}) um disagrees "
                        f"with the tile-placement prediction "
                        f"({x_pred * 1e6:.4f}, {y_pred * 1e6:.4f}) um by "
                        f"({_dxc * 1e6:.4g}, {_dyc * 1e6:.4g}) um "
                        f"(tolerance {_tol * 1e6:.4g} um).  The readout "
                        f"window would be misplaced; this is a library "
                        f"bug, not a usage error.")
            tile = np.asarray(res.field)
            if weight != 1.0:
                tile = tile * weight
            if acc is None and acc_i is None:
                acc_dtype = np.result_type(tile.dtype, np.complex64)
                if recombine == 'incoherent':
                    acc_i = np.zeros((N_out, N_out), dtype=np.float64)
                else:
                    acc = np.zeros((N_out, N_out), dtype=acc_dtype)
            # index origin of the tile inside the common grid.  Both sizes
            # are even, so the lattice snap above makes this an EXACT
            # integer offset: tile pixel j sits at
            # centre + (j - n_tile/2 + m) * dx_out and common pixel i at
            # centre + (i - N_out/2) * dx_out.
            c0 = mx + (N_out - n_tile) // 2
            r0 = my + (N_out - n_tile) // 2
            sr0, sr1 = max(r0, 0), min(r0 + n_tile, N_out)
            sc0, sc1 = max(c0, 0), min(c0 + n_tile, N_out)
            p_tile = float((np.abs(tile) ** 2).sum()) * dx_out * dx_out
            p_kept = 0.0
            if sr1 > sr0 and sc1 > sc0:
                sub = tile[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0]
                if recombine == 'coherent':
                    acc[sr0:sr1, sc0:sc1] += sub
                else:
                    acc_i[sr0:sr1, sc0:sc1] += np.abs(sub) ** 2
                p_kept = float((np.abs(sub) ** 2).sum()) * dx_out * dx_out
            p_in = float((np.abs(np.asarray(fld)) ** 2).sum()) * dx * dx \
                * float(abs(weight) ** 2)
            # THREE planes, kept apart: power_exit is measured at the last
            # group's exit vertex on the co-moving grid, so it carries the
            # traced element's real aperture vignetting and NOTHING about
            # the readout window -- which makes 'throughput'
            # window-independent.  'capture' is where a too-small tile
            # shows up, and it is field-angle dependent (the halo grows off
            # axis), so folding it into 'throughput' is how a bookkeeping
            # artefact gets reported as vignetting.
            p_exit_env = _multi_chain_exit_power(res.stages)
            p_exit = (p_exit_env * float(abs(weight) ** 2)
                      if p_exit_env is not None else float('nan'))
            capture = ((p_tile / p_exit)
                       if (np.isfinite(p_exit) and p_exit > 0.0)
                       else float('nan'))
            if (np.isfinite(capture)
                    and abs(capture - 1.0) > readout_capture_tol):
                _multi_dispose(
                    on_readout_clip,
                    f"{fn}: congruence {name!r} -- its "
                    f"{n_tile} x {dx_out * 1e6:.4f} um readout window holds "
                    f"{capture * 100:.2f} % of the power the chain delivered "
                    f"(power_out {p_tile:.6e} vs power_exit {p_exit:.6e}, "
                    f"tolerance {readout_capture_tol * 100:.2f} %).  "
                    + ("Below 100 % the window is CLIPPING the beam's halo; "
                       "because the halo grows with field angle the clipped "
                       "fraction varies across a fan and reads as physical "
                       "vignetting when it is not.  Raise readout_tile "
                       "(bounded above by one Bluestein period"
                       + (f", {period * 1e3:.4f} mm = {_fit(period)} px, "
                          f"here" if period else "")
                       + ") or widen the common grid."
                       if capture < 1.0 else
                       "Above 100 % the window is picking up periodic "
                       "REPLICAS, which ADD power that is not there; "
                       "shrink readout_tile.")
                    + "  Use congruences[k]['throughput'] (measured at the "
                      "chain exit, window-independent) for the real "
                      "vignetting, and on_readout_clip='ignore' to silence "
                      "this check.")
            infos.append({
                'name': name,
                'weight': weight,
                'carrier': carrier,
                'chief_ray': (x_pred, y_pred),
                'exit_tilt': (L_out, M_out),
                'tile': n_tile,
                'tile_centre': tile_centre,
                'tile_origin': (r0, c0),
                'readout_period': period,
                'power_in': p_in,
                'power_exit': p_exit,
                'power_out': p_tile,
                'throughput': (p_exit / p_in) if p_in > 0.0 else float('nan'),
                'capture': capture,
                'clipped': (1.0 - p_kept / p_tile) if p_tile > 0.0 else 0.0,
                'stages': res.stages,
            })
        if not restart:
            break

    if recombine == 'incoherent':
        acc = np.sqrt(acc_i).astype(acc_dtype)
    return TracedCarrierChainMultiResult(acc, dx_out, centre, infos)
