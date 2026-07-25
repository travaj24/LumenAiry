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
    'TracedCarrierChainResult',
    'propagate_carrier_referenced',
    'carrier_referenced_reconstruct',
    'carrier_referenced_envelope',
    'carrier_referenced_aperture',
    'carrier_referenced_fit_radius',
    'carrier_referenced_focus_readout',
    'carrier_referenced_exact_focus_readout',
    'propagate_traced_carrier_chain',
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


def _radial_carrier_phase(shape, dx, dy, wavelength, R, sign, bld=np):
    """``exp(sign*i*k*(x^2+y^2)/(2R))`` on the centred grid (float64
    carrier argument, cast to nothing here -- caller casts).  Built on backend
    ``bld`` (host NumPy for JAX, the device namespace otherwise); ``bld is np``
    reproduces the historical NumPy screen byte-for-byte."""
    Ny, Nx = shape
    x = (bld.arange(Nx, dtype=np.float64) - Nx / 2) * dx
    y = (bld.arange(Ny, dtype=np.float64) - Ny / 2) * dy
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


def _envelope_amp_radius(E_env, dx, dy):
    """1/e AMPLITUDE radius of the (assumed roughly Gaussian) envelope on the
    centred grid, from the intensity second moment (``w = sqrt(2)*r2m``).
    Returns 0.0 for an empty / zero field."""
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
    _check_2d_scalar_field(E_env, 'carrier_referenced_reconstruct')
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
    _check_2d_scalar_field(E_full, 'carrier_referenced_envelope')
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


def _fit_carrier_inv(E, wavelength, dx, dy, axis=None):
    """Intensity-weighted mean wavefront inverse-curvature ``1/R`` of a field
    (0.0 for a flat/collimated wavefront).

    From ``E = A exp(i*phi)`` with ``phi = +k r^2/(2R)`` (the library's
    diverging-``R>0`` convention), ``Im[E* (x dE/dx)] = k A^2 x^2 / R``, so
    ``1/R_x = Im[sum E* x dE/dx] / (k sum |E|^2 x^2)`` -- a phase-unwrap-free,
    aperture-robust estimator.  ``axis=None`` fits the isotropic (combined
    ``x^2+y^2``) curvature; ``axis=1``/``0`` fit x/y separately."""
    # Host-side least-squares curvature fit -> Python float; CuPy-safe pull
    # (see ``_envelope_amp_radius``).
    from ..backend import to_numpy
    E = to_numpy(E)
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


def _rereference(env, R_old, R_new, wavelength, dx, dy,
                 bld=np, xp=np, is_jax=False):
    """Move an envelope from carrier ``R_old=(Rx,Ry)`` to ``R_new=(Rx,Ry)``:
    ``env * exp(i*k*r^2/2 * (1/R_old - 1/R_new))`` per axis.  The phase screen
    is built on ``bld`` (host NumPy for JAX) and moved on-device; NumPy
    defaults reproduce the historical screen byte-for-byte."""
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
            ivx = _fit_carrier_inv(env_ap, wavelength, dx, dy, axis=1)
            ivy = _fit_carrier_inv(env_ap, wavelength, dx, dy, axis=0)
            inx = (0.0 if np.isinf(R_x) else 1.0 / R_x) + ivx
            iny = (0.0 if np.isinf(R_y) else 1.0 / R_y) + ivy
            Rnx = np.inf if inx == 0.0 else 1.0 / inx
            Rny = np.inf if iny == 0.0 else 1.0 / iny
            env_out = _rereference(env_ap, (R_x, R_y), (Rnx, Rny),
                                   wavelength, dx, dy, bld, xp, is_jax)
            R_out = (Rnx, Rny) if (Rnx != Rny) else Rnx
            out_astig = (Rnx != Rny)
        else:
            iv = _fit_carrier_inv(env_ap, wavelength, dx, dy, axis=None)
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

    Notes
    -----
    The carrier leg is a fast (byte-identical) Sziklas-Siegman step because
    the stop plane is chosen to PRECEDE the focus -- the near-focus bridge is
    deliberately NOT relied upon (its output would land back on the collapsed
    co-moving grid).  Validated against the analytic Gaussian focus and a
    resolved fixed-grid reference in ``test_niche_r8_tiltaware_chain_api.py``.
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(env, 'carrier_referenced_focus_readout')
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

    from .mft import angular_spectrum_propagate_mft
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


def _exact_sphere_eikonal(shape, dx, dy, wavelength, R):
    """Exact on-axis point-source sphere eikonal ``S(R) = sign(R)(sqrt(r^2 +
    R^2) - |R|)`` in METRES on the centred grid (host NumPy float64).  This is
    the EXACT converging/diverging wavefront -- the paraxial parabola
    ``r^2/2R`` is its small-``r/R`` truncation, dropping ``-r^4/8R^3`` which is
    huge at high NA.  Same sign convention as ``apply_real_lens_traced``'s
    carrier ``W`` (R7/F2) and :func:`carrier_referenced_reconstruct`."""
    Ny, Nx = int(shape[-2]), int(shape[-1])
    x = (np.arange(Nx, dtype=np.float64) - Nx / 2) * dx
    y = (np.arange(Ny, dtype=np.float64) - Ny / 2) * dy
    Y, X = np.meshgrid(y, x, indexing='ij')
    r2 = X * X + Y * Y
    sgn = 1.0 if R > 0 else -1.0
    return sgn * (np.sqrt(r2 + R * R) - abs(R))


def _sphere_parab_conversion(shape, dx, wavelength, R, sign, w_beam=None):
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
    """
    if not np.isfinite(R) or R == 0.0:
        return None
    n = int(shape[-1])
    ny = int(shape[-2])
    x = (np.arange(n, dtype=np.float64) - n / 2) * dx
    y = (np.arange(ny, dtype=np.float64) - ny / 2) * dx
    r2 = x[None, :] ** 2 + y[:, None] ** 2
    k = 2.0 * np.pi / wavelength
    diff = _exact_sphere_eikonal((ny, n), dx, dx, wavelength, R) \
        - r2 / (2.0 * R)
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
    """
    env = np.asarray(env)
    n = env.shape[-1]
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
    window_factor : float, default 7.0
        Fine-grid physical span in units of the beam amplitude radius
        (``_envelope_amp_radius``).  7 holds the beam to <1e-6 truncation.
    centre_out : (float, float), optional
        Physical ``(x, y)`` centre of the output grid (m).  Default on-axis.
    bandlimit : bool, default True
        Band-limit the ASM transfer function (Matsushima-Shimobaba).

    Returns
    -------
    E_out : ndarray, complex, shape (N_out, N_out)
        The full physical field at the target plane on the centred ``dx_out``
        grid -- same absolute-phase convention as
        :func:`angular_spectrum_propagate_mft`.
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_full, 'carrier_referenced_exact_focus_readout')
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
    w_amp = _envelope_amp_radius(E, dx, dx)

    # -- exact-sphere envelope (SMOOTH; the two steep phases alias-cancel on the
    #    shared grid, leaving only amplitude x exp(i * genuine aberration)) -----
    if np.isfinite(R) and R != 0.0:
        S = _exact_sphere_eikonal(E.shape, dx, dx, wavelength, R)
        env = E * np.exp(-1j * k * S)
    else:
        env = E

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
    # crop window (physical) that holds the beam
    if w_amp > 0.0:
        win = min(float(window_factor) * w_amp, N * dx)
    else:
        win = N * dx
    n_crop = int(2 * round((win / dx) / 2))
    n_crop = int(min(max(n_crop, 2), N))
    win = n_crop * dx
    if N_fine is None:
        N_fine = int(2 ** int(np.ceil(np.log2(max(win / dx_fine, n_crop)))))
    N_fine = int(N_fine)
    dx_fine = win / N_fine

    env_f = _fourier_upsample_crop(env, n_crop, N_fine)

    # -- reconstruct the exact sphere on the fine grid ------------------------
    if np.isfinite(R) and R != 0.0:
        S_f = _exact_sphere_eikonal((N_fine, N_fine), dx_fine, dx_fine,
                                    wavelength, R)
        E_fine = (env_f * np.exp(1j * k * S_f)).astype(np.complex128)
    else:
        E_fine = np.asarray(env_f, dtype=np.complex128)

    # -- exact band-limited ASM Bluestein zoom to the target ------------------
    from .mft import angular_spectrum_propagate_mft
    return angular_spectrum_propagate_mft(
        E_fine, z, wavelength, dx_fine, dx_out, int(N_out),
        centre_out=centre_out, bandlimit=bandlimit)


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
    """

    field: np.ndarray
    R: Optional[float]
    dx: float
    stages: list


def _paraxial_group_r_out(prescription, R_in, wavelength):
    """Exit carrier radius the GROUP itself supplies: its air-to-air paraxial
    ABCD (:func:`system_abcd_prescription`) mapped onto the incoming carrier
    radius by the wavefront Moebius law ``R_out = (A R_in + B)/(C R_in + D)``.

    This reproduces the design-121 repro script's external q-trace ``R_out`` to
    full precision (the two are algebraically identical for the geometric
    wavefront radius), so the orchestrator needs no external q-trace."""
    from ..raytrace.seidel import system_abcd_prescription
    M, _efl, _bfl, _ffl = system_abcd_prescription(prescription, wavelength)
    A, B = float(M[0, 0]), float(M[0, 1])
    C, D = float(M[1, 0]), float(M[1, 1])
    if not np.isfinite(R_in):
        return np.inf if abs(C) < 1e-300 else A / C
    num = A * R_in + B
    den = C * R_in + D
    if abs(den) < 1e-300:
        return np.inf
    return num / den


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
                           sphere_reference=False):
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
    """
    from ..elements import apply_real_lens_traced
    N = env.shape[-1]
    w = _envelope_amp_radius(env, cur_dx, cur_dx)
    na = min(max(na_exit, 0.02), 0.95)
    dx_fine = wavelength / (3.0 * na)
    win = min(window_factor * w, N * cur_dx) if w > 0 else N * cur_dx
    n_crop = int(2 * round((win / cur_dx) / 2))
    n_crop = int(min(max(n_crop, 2), N))
    win = n_crop * cur_dx
    n_fine = int(2 ** int(np.ceil(np.log2(max(win / dx_fine, n_crop)))))
    n_fine = int(min(n_fine, n_fine_cap))
    dx_fine = win / n_fine

    # F-D: n_fine_cap can force dx_fine coarser than the exit sphere's
    # Nyquist pitch.  The retrace still runs (the downstream exact readout's
    # bandlimit=True masks the aliased corner) but silently discards
    # outer-NA content -- warn so that's visible instead of silent.
    nyquist_dx = wavelength / (2.0 * na)
    if dx_fine > nyquist_dx:
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

    env_f = _fourier_upsample_crop(env, n_crop, n_fine)
    E_full = carrier_referenced_reconstruct(env_f, R_in, wavelength, dx_fine)
    if sphere_reference:
        # The stored envelope is EXACT-SPHERE-referenced (chain
        # ``carrier_reference='sphere'``): convert the parabola-referenced
        # reconstruction so the element receives the physical wavefront.  No
        # exit-side conversion here -- the exit field goes straight to
        # carrier_referenced_exact_focus_readout, which references the exact
        # sphere itself.
        _cf = _sphere_parab_conversion(np.shape(E_full), dx_fine, wavelength,
                                       R_in, +1, w_beam=w)
        if _cf is not None:
            E_full = np.asarray(E_full) * _cf

    # F-C: preserve the CHAIN's physical ray pitch (ray_subsample * cur_dx)
    # on the fine retrace grid, rather than reinterpreting the same integer
    # ray_subsample in dx_fine pixel units.
    rs_fine = max(1, int(round(float(ray_subsample) * cur_dx / dx_fine)))
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

    E_exit = apply_real_lens_traced(
        E_full, prescription=presc, wavelength=wavelength, dx=dx_fine,
        carrier=R_in, ray_subsample=rs_fine, n_workers=n_workers,
        **call_kw)
    return np.asarray(E_exit), float(dx_fine)


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
) -> TracedCarrierChainResult:
    """Propagate a beam ENVELOPE through a chain of real (traced) lens groups on
    a co-moving carrier-referenced grid (audit F4.1).

    Packages the per-group hand-off pattern -- analytic carrier leg ->
    reconstruct at the group front vertex -> ``apply_real_lens_traced(carrier=
    R_in)`` -> re-envelope with the group's exit curvature ``R_out`` -- into one
    call.  The element SUPPLIES ``R_out`` from its own paraxial ABCD
    (:func:`system_abcd_prescription` mapped onto the incoming carrier), so the
    caller needs no external q-trace.  Reproduces
    ``validation/repro_traced_carrier_121/carrier_chain_121.py`` in a few lines.

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
    wavelength, dx : float
        Wavelength and input grid pitch (m).  Isotropic (square) grid only.
    r_in : float, default ``inf``
        Carrier radius of ``E_in`` (m).  ``inf`` = a plain (collimated-carrier)
        field.
    ray_subsample : int, default 4
    n_workers : int, optional
        Threaded into every :func:`apply_real_lens_traced` call.
    traced_kwargs : dict, optional
        Extra kwargs for every :func:`apply_real_lens_traced` call (e.g.
        ``parallel_amp=False``).  ``carrier`` / ``dx`` / ``wavelength`` /
        ``prescription`` / ``ray_subsample`` / ``n_workers`` are managed by the
        orchestrator and must not appear here.
    final_distance : float, default 0
        Free-space distance from the last group's exit vertex to the target
        (readout) plane (m).
    focus_readout : dict, optional
        When given, land the final leg via a focus readout.  Must supply
        ``dx_out`` and ``N_out``; may supply ``centre_out`` / ``bandlimit``
        (and ``standoff`` for the paraxial path, or ``dx_fine`` / ``N_fine`` /
        ``window_factor`` for the exact path).  Otherwise the final leg is an
        ordinary carrier step and the field is reconstructed on the co-moving
        grid.

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

    Returns
    -------
    TracedCarrierChainResult
        ``(field, R, dx, stages)`` -- see the class docstring.  ``R`` is
        ``None`` after any focus readout (paraxial or exact).
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'propagate_traced_carrier_chain')
    from ..elements import apply_real_lens_traced

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
    n_groups = len(groups)
    # v5.29 default flip (audit AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24 S8):
    # the chain's per-group traced calls default to the validated
    # carrier-regime configuration -- the chain ALWAYS operates with its
    # carrier beyond the grid Nyquist, where the geometric (ray-density)
    # amplitude and the geometric residual carry are the correct physics,
    # not preferences.  Anything the caller passes in ``traced_kwargs`` (or a
    # group's own ``traced_kwargs``) WINS over these defaults; the standalone
    # ``apply_real_lens_traced`` element defaults are untouched.
    base_kw = {'amplitude_model': 'ray_density',
               'preserve_input_phase': 'remap'}
    if traced_kwargs:
        base_kw.update(traced_kwargs)
    R = float(r_in)
    cur_dx = float(dx)
    env = E_in
    stages: list = []

    for gi, g in enumerate(groups):
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
                "prescription dict (with 'surfaces') or a group-spec dict "
                "(with 'prescription'); got "
                f"{type(g).__name__}.")

        # free-space carrier leg to the group front vertex
        if gap != 0.0:
            cr = propagate_carrier_referenced(env, R, gap, wavelength, cur_dx)
            env, R, cur_dx = cr.env, cr.R, cr.dx
            if isinstance(cur_dx, tuple):
                cur_dx = cur_dx[0]
            if isinstance(R, tuple):
                R = R[0]

        R_use = float(R if g_r_in is None else g_r_in)
        # the element SUPPLIES R_out (its own paraxial ABCD), unless overridden
        R_out = (float(g_r_out) if g_r_out is not None
                 else _paraxial_group_r_out(presc, R_use, wavelength))
        _name = (presc.get('name', f'group{gi}') if isinstance(presc, dict)
                 else f'group{gi}')
        call_kw = dict(base_kw)
        if g_kw:
            call_kw.update(g_kw)

        # ---- exact high-NA FINAL leg (R9): re-trace this (last) group on a
        # grid that Nyquist-samples its exit sphere, then exact-ASM to target.
        is_final = (gi == n_groups - 1)
        do_exact = False
        na_exit = 0.0
        if is_final and focus_readout is not None and final_leg != 'paraxial':
            w_in = _envelope_amp_radius(env, cur_dx, cur_dx)
            if np.isfinite(R_out) and R_out != 0.0 and w_in > 0.0:
                na_exit = w_in / abs(R_out)
            do_exact = (final_leg == 'exact'
                        or (final_leg == 'auto' and na_exit > na_exact_threshold))
        if do_exact:
            fr = dict(focus_readout)
            if 'dx_out' not in fr or 'N_out' not in fr:
                raise ValueError(
                    "propagate_traced_carrier_chain: focus_readout must supply "
                    "'dx_out' and 'N_out'.")
            E_exit_fine, dx_fine = _fine_trace_group_exit(
                env, R_use, cur_dx, presc, wavelength, ray_subsample, n_workers,
                call_kw, R_out, na_exit,
                window_factor=float(fr.get('window_factor', 7.0)),
                n_fine_cap=int(fr.get('n_fine_cap', 16384)),
                max_fine_launch_points=int(
                    fr.get('max_fine_launch_points', 4096)),
                sphere_reference=_sphere_ref)
            w_stage, p_stage = _chain_envelope_stats(E_exit_fine, dx_fine)
            stages.append({
                'name': _name, 'R_in': R_use, 'R_out': R_out, 'dx': dx_fine,
                'w': w_stage, 'power': p_stage, 'exact_final': True})
            exact_kw = {kk: fr[kk] for kk in (
                'dx_out', 'N_out', 'dx_fine', 'N_fine', 'window_factor',
                'centre_out', 'bandlimit') if kk in fr}
            field = carrier_referenced_exact_focus_readout(
                E_exit_fine, R_out, final_distance, wavelength, dx_fine,
                **exact_kw)
            return TracedCarrierChainResult(
                np.asarray(field), None, float(fr['dx_out']), stages)

        # ---- standard coarse trace + paraxial re-envelope ------------------
        E_full = carrier_referenced_reconstruct(env, R_use, wavelength, cur_dx)
        if _sphere_ref:
            # hand the element the EXACT-sphere-referenced wavefront its ray
            # launch assumes (see carrier_reference)
            _cf = _sphere_parab_conversion(
                np.shape(E_full), cur_dx, wavelength, R_use, +1,
                w_beam=_envelope_amp_radius(env, cur_dx, cur_dx))
            if _cf is not None:
                E_full = np.asarray(E_full) * _cf
        E_exit = apply_real_lens_traced(
            E_full, prescription=presc, wavelength=wavelength, dx=cur_dx,
            carrier=R_use, ray_subsample=ray_subsample, n_workers=n_workers,
            **call_kw)
        E_exit = np.asarray(E_exit)
        if _sphere_ref:
            # re-envelope against the EXACT exit sphere, so the stored
            # envelope is the wavefront residual (the carried content)
            _cf = _sphere_parab_conversion(E_exit.shape, cur_dx, wavelength,
                                           R_out, -1)
            if _cf is not None:
                E_exit = E_exit * _cf
        env = carrier_referenced_envelope(E_exit, R_out, wavelength, cur_dx)
        R = R_out
        w_stage, p_stage = _chain_envelope_stats(env, cur_dx)
        stages.append({
            'name': _name,
            'R_in': R_use, 'R_out': R_out, 'dx': cur_dx,
            'w': w_stage, 'power': p_stage})

    # ---- final leg to the target plane ----
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
        field = carrier_referenced_focus_readout(
            env, R, final_distance, wavelength, cur_dx, **fr)
        return TracedCarrierChainResult(np.asarray(field), None,
                                        float(fr['dx_out']), stages)

    if final_distance != 0.0:
        cr = propagate_carrier_referenced(env, R, final_distance, wavelength,
                                          cur_dx)
        env, R, cur_dx = cr.env, cr.R, cr.dx
        if isinstance(cur_dx, tuple):
            cur_dx = cur_dx[0]
        if isinstance(R, tuple):
            R = R[0]
    field = carrier_referenced_reconstruct(env, R, wavelength, cur_dx)
    return TracedCarrierChainResult(np.asarray(field), float(R), cur_dx, stages)
