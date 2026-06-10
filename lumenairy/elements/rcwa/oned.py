"""RCWA 1-D gratings: planar/conical efficiency + Jones, wavelength
sweeps, segment builders, ASR, and the differentiable JAX-1D entry point."""
from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np

from ...backend import (
    array_namespace,
    backend_name,
    to_numpy,
)
from ._core import (
    _C,
    _block,
    _check_energy,
    _concrete,
    _EnergyError,
    _EnergyWarning,
    _forward_flux_kz,
    _fourier_coeffs_1d,
    _grazing_safe_wavelength,
    _homogeneous_eigenmodes,
    _interface_smatrix,
    _interface_smatrix_general,
    _is_traced,
    _layer_eigenmodes,
    _layer_eigenmodes_tensor,
    _modes_to_M,
    _normalize_pol,
    _propagation_smatrix,
    _propagation_smatrix_general,
    _rcwa_xp,
    _redheffer_star,
    _reject_jax_offplane,
    _require_inplane_tensor,
    _require_jax_x64,
    _require_propagating_incidence,
    _sqrt_forward,
    _stabilize_bumps,
    _tensor_convolutions,
    _tensor_convolutions_full,
    _toeplitz_1d,
    _validate_geometry,
    _with_blas_limit,
)


def _binary_grating_convolutions(n_ridge, n_groove, duty_cycle, n_orders,
                                 n_samples=4096, use_li=True):
    """Convolution matrices for a 1-D binary grating: the Laurent ``[[eps]]``
    and the Li inverse-rule ``[[1/eps]]^{-1}``.  The ridge (index
    ``n_ridge``) occupies the fraction ``duty_cycle`` of the period.

    A closed-form Fourier series exists for a binary profile, but sampling
    + FFT keeps the path identical to the (future) arbitrary-profile and
    2-D cases and is exact to machine precision at this sampling.

    Backend-agnostic / JAX-differentiable: the hard-edge ``where`` selects
    between the (possibly traced) ridge / groove permittivities, so the
    gradient flows to the INDEX VALUES (the documented JAX design targets);
    ``duty_cycle`` is a discrete threshold and is not differentiated.

    ``use_li`` (default ``True``) controls whether the inverse-rule matrix is
    built: the Laurent / TE-dielectric path never reads it, so passing
    ``use_li=False`` skips an O(n_orders^3) matrix inverse + one FFT and returns
    ``EPS_II = None`` (byte-identical result -- the matrix was discarded anyway).
    """
    xp = array_namespace(n_ridge, n_groove)
    # The Toeplitz needs c_k for |k| up to 2*n_orders; an N-sample FFT represents
    # c_k WITHOUT ALIASING only for |k| <= N/2, so N must exceed 4*n_orders.  When
    # n_orders is large (> ~1024 at the 4096 default) the modular wrap full[k % N]
    # folds high harmonics onto low orders and SILENTLY corrupts EPS/EPS_II -- bump
    # the internal grid to the next power of two that clears the Nyquist limit.
    need = 4 * n_orders + 2
    if n_samples < need:
        n_samples = 1 << (need - 1).bit_length()
    x = (xp.arange(n_samples) + 0.5) / n_samples
    eps_r = xp.asarray(n_ridge).astype(_C) ** 2
    eps_g = xp.asarray(n_groove).astype(_C) ** 2
    eps = xp.where(x < duty_cycle, eps_r, eps_g).astype(_C)
    # The Toeplitz matrix needs coefficients c_k for k = -(N-1)..(N-1) with
    # N = 2*n_orders+1, i.e. n_coeffs = N.
    n_coeffs = 2 * n_orders + 1
    eps_coeffs = _fourier_coeffs_1d(eps, n_coeffs)
    EPS = _toeplitz_1d(eps_coeffs, n_orders)               # Laurent rule
    if not use_li:
        return EPS, None                                   # inverse rule unused
    inv_eps_coeffs = _fourier_coeffs_1d(1.0 / eps, n_coeffs)
    EPS_II = xp.linalg.inv(_toeplitz_1d(inv_eps_coeffs, n_orders))  # inverse rule
    return EPS, EPS_II



# ===========================================================================
# Adaptive Spatial Resolution (ASR / Granet 1999) -- 1-D binary grating
# ===========================================================================
#
# ASR replaces the uniform spatial sampling with a periodic coordinate map
# ``x = x(u)`` whose Jacobian ``f(u) = dx/du`` is SMALL (fine resolution) at the
# grating walls and LARGE (coarse) in the homogeneous regions, then runs the
# Fourier Modal Method in the uniform ``u``-coordinate.  Concentrating the
# harmonics where the field varies fastest gives much faster convergence for
# metals / high contrast / TM than uniform sampling (~10x fewer orders for the
# same TM error on a gold grating; ~hundreds-x for TE).  "Matched coordinates"
# (the configuration used here) places the binary walls EXACTLY on coordinate
# lines, so ``eps(x(u))`` is a clean step on the ``u``-grid; combined with the
# Li inverse rule (``use_li``) this is the matched-coordinate FFF.
#
# Two NON-OBVIOUS facts -- both proven the hard way by three independent
# prototypes, both load-bearing, both regression-tested -- govern correctness:
#
#  (1) FACTORIZATION.  Use the NON-MULTIPLIED (chain-rule) form: the metric
#      enters ONLY on the derivative (``Kx_asr = [[1/f]] @ Du``) and the
#      permittivity is the plain ``eps(x(u))`` sampled on the ``u``-grid
#      (Laurent ``[[eps]]`` tangential, inverse-rule ``[[1/eps]]^{-1}``
#      wall-normal).  Do NOT fold the metric into the permittivity (the
#      ``[[f eps]]`` / ``[[1/(f eps)]]^{-1}`` "multiply-by-f covariant" form):
#      mixing it with ``[[1/f]] Du`` converges to the WRONG value at high N
#      while STILL being bit-exact at ``eta=0`` -- internal consistency does
#      not imply the correct continuous-limit operator.
#
#  (2) BASIS BRIDGE.  The layer is solved in the ``u``-Fourier basis but the
#      homogeneous regions live in the physical-``x`` Rayleigh basis.  The
#      order-``m`` harmonics do NOT coincide between the two bases unless
#      ``f == 1``, so the layer modes must be mapped through the dense Rayleigh
#      transform ``G[m, n] = <exp(i a_n x(u)) exp(-i a_m u)>_u`` BEFORE the
#      interface match.  The direction is ``G^{-1}`` (a physical field with
#      x-coeffs ``a`` has u-coeffs ``G a``); applying ``G`` instead gives a
#      stable-but-WRONG answer.  At ``eta=0``: ``x(u)=u`` -> ``G=I``.


def _asr_metric_profile(duty_cycle, eta, n_samples):
    """Matched Granet sine-stretch coordinate map for a 1-D binary cell.

    Period is normalised to 1; the two walls sit at ``u = 0`` and
    ``u = duty_cycle`` (and the cell edge ``u = 1``), landing exactly on
    coordinate lines.  On each sub-interval of length ``L`` (local
    ``s = (u - a)/L in [0, 1)``)::

        x(u) = a + L (s - (eta / 2pi) sin(2pi s))
        f(u) = dx/du = 1 - eta cos(2pi s)

    so ``f`` is fine (``1 - eta``) at the walls, coarse (``1 + eta``) mid-cell,
    ``integral(f) = L`` over each sub-interval (``<f> = 1`` exactly, preserving
    the Floquet lattice), and ``eta = 0`` gives ``f == 1``, ``x(u) = u``.
    Returns ``(u, f_u, x_u, in_ridge)`` on the uniform grid
    ``u = (i + 0.5)/n_samples``.  Host NumPy (pure geometry).
    """
    u = (np.arange(n_samples) + 0.5) / n_samples
    u1 = float(duty_cycle)
    f = np.ones(n_samples)
    x = u.copy()
    tp = 2.0 * np.pi
    in_ridge = u < u1
    if u1 > 0.0:                                  # ridge sub-interval [0, u1)
        a = u[in_ridge]
        f[in_ridge] = 1.0 - eta * np.cos(tp * a / u1)
        x[in_ridge] = a - (eta * u1 / tp) * np.sin(tp * a / u1)
    w = 1.0 - u1
    if w > 0.0:                                   # groove sub-interval [u1, 1)
        b = u[~in_ridge] - u1
        f[~in_ridge] = 1.0 - eta * np.cos(tp * b / w)
        x[~in_ridge] = u[~in_ridge] - (eta * w / tp) * np.sin(tp * b / w)
    return u, f, x, in_ridge



def _asr_convolutions(n_ridge, n_groove, duty_cycle, n_orders, eta, xp,
                      n_samples=16384):
    """ASR convolution matrices + the u<->x basis bridge (NON-multiplied form).

    Returns ``(Fi, EPS, EPS_II, G)`` (in the namespace ``xp``):

    - ``Fi = [[1/f]]``  -- Laurent Toeplitz of ``1/f`` (``f`` is smooth, so the
      bare metric takes the DIRECT rule; never an inverse rule on ``f``).  Used
      as ``Kx_asr = Fi @ Du`` (the metric lives on the derivative).
    - ``EPS = [[eps(x(u))]]``        -- Laurent (wall-tangential ``E_y``/``E_z``).
    - ``EPS_II = [[1/eps(x(u))]]^-1`` -- Li inverse rule (wall-normal ``E_x``).
      Plain ``1/eps`` (NO metric inside): ``D_x = eps E_x`` is continuous on the
      same ``u``-line, and the metric already lives on the derivative.
    - ``G[m, n] = <exp(i a_n x(u)) exp(-i a_m u)>_u`` -- the dense Rayleigh
      transform from the layer ``u``-basis to the physical-``x`` region basis
      (see section header fact (2); applied as ``G^{-1}`` to the layer modes).

    NumPy/CuPy only (the ASR path is gated off JAX).  The permittivity is
    sampled from the ALREADY-CONJUGATED internal indices, so it shares the
    public->internal convention bridge with ``_binary_grating_convolutions``.
    """
    u, f_u, x_u, in_ridge = _asr_metric_profile(duty_cycle, eta, n_samples)
    eps_r = complex(n_ridge) ** 2
    eps_g = complex(n_groove) ** 2
    eps_u = np.where(in_ridge, eps_r, eps_g).astype(_C)   # eps(x(u)) on u-grid
    n_coeffs = 2 * n_orders + 1
    Fi = _toeplitz_1d(_fourier_coeffs_1d(1.0 / f_u, n_coeffs), n_orders)
    EPS = _toeplitz_1d(_fourier_coeffs_1d(eps_u, n_coeffs), n_orders)
    EPS_II = np.linalg.inv(
        _toeplitz_1d(_fourier_coeffs_1d(1.0 / eps_u, n_coeffs), n_orders))
    # u<->x Rayleigh bridge (period-normalised orders; normal incidence).
    orders = np.arange(-n_orders, n_orders + 1)
    twopi = 2.0 * np.pi
    ph_xn = np.exp(1j * twopi * np.outer(x_u, orders))     # (Ns, N) exp(i a_n x)
    ph_um = np.exp(-1j * twopi * np.outer(u, orders))      # (Ns, N) exp(-i a_m u)
    G = (ph_um.T @ ph_xn) / n_samples                      # (N, N)
    return (xp.asarray(Fi.astype(_C)), xp.asarray(EPS.astype(_C)),
            xp.asarray(EPS_II.astype(_C)), xp.asarray(G.astype(_C)))



# ===========================================================================
# Public 1-D entry point
# ===========================================================================

def _resolve_incidence(angle, theta):
    """Cross-dimension alias: accept ``theta`` (the polar-angle spelling used by
    ``RCWAStack`` and the ``rcwa_*_2d`` conical solvers) as a synonym for the 1-D
    classical-mount ``angle``.  ``theta`` IS ``angle`` -- the SAME number, NO
    scaling or conversion, both measured from the ``+z`` surface normal; the 1-D
    mount is planar (azimuth ``phi = 0``).  ``theta`` overrides when supplied;
    ``None`` (the default) keeps ``angle``."""
    return angle if theta is None else theta



@_with_blas_limit
def rcwa_efficiency_1d(
    period: float,
    n_ridge: complex,
    n_groove: complex,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    duty_cycle: float,
    wavelength: float,
    *,
    angle: float = 0.0,
    theta: float | None = None,
    polarization: str = "te",
    n_orders: int = 11,
    formulation: str = "auto",
    stabilize: bool = False,
    asr_eta: float = 0.0,
    asr_samples: int = 16384,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigorous diffraction efficiencies of a 1-D binary grating.

    Backend-dispatched: returns NumPy arrays by default, CuPy arrays when
    ``use_gpu=True`` (or a CuPy input is passed), and differentiable JAX
    arrays when any index / geometry argument is a JAX array -- this single
    routine is what the (now-deprecated) :func:`rcwa_efficiency_1d_jax`
    forwards to, so the NumPy and JAX results agree to eig precision.

    Parameters
    ----------
    period : float
        Grating period (metres).
    n_ridge, n_groove : complex
        Refractive INDICES of the ridge and groove regions of the patterned
        layer (``n = n + i kappa``, ``kappa > 0`` for loss).  CONVENTION WARNING:
        this scalar entry point takes the refractive index ``n``, whereas the Jones
        family (:func:`rcwa_jones_1d` etc.) takes the PERMITTIVITY ``eps = n**2`` --
        a wrong-convention value is silently accepted (e.g. ``n=2.1`` read as
        ``eps=2.1``), so pass ``n`` here and ``n**2`` to the Jones functions.
    n_substrate, n_superstrate : complex
        Indices of the transmission (below) and incidence (above)
        half-spaces.
    depth : float
        Grating (layer) thickness (metres).
    duty_cycle : float
        Ridge fraction of the period, in ``[0, 1]``.
    wavelength : float
        Vacuum wavelength (metres).
    angle : float, optional
        Incidence polar angle from the surface normal (radians), in the
        plane of periodicity (planar mounting).  Default 0.  Accepts ``theta``
        as a cross-dimension alias (the polar-angle spelling used by the 2-D /
        conical solvers and ``RCWAStack``); when given it overrides ``angle``.
    polarization : {'te', 'tm'}, optional
        ``'te'`` (s, E along grooves / y) or ``'tm'`` (p).  Default ``'te'``.
    n_orders : int, optional
        Number of retained orders per side; total harmonics ``2*n_orders+1``.
        Default 11.
    formulation : {'auto', 'laurent', 'li'}, optional
        Fourier factorization.  ``'laurent'`` (direct rule) converges fast
        for dielectrics; ``'li'`` (inverse rule) is strongly preferred for
        metals / high-contrast TM (the wall-normal ``E_x`` is discontinuous).
        ``'auto'`` (default) picks ``'li'`` for TM or when any index is
        metallic, else ``'laurent'`` -- and ``'laurent'`` for TE never touches
        the inverse rule (the wall-normal operator does not enter the TE
        problem, so the result is identical to ``'li'``).  An explicit
        ``'laurent'`` with TM is PERMITTED but converges slowly -- and for
        METALS it may not converge within reachable order counts at all
        (audit 2026-06-10: on a Ag grating ``R0`` still oscillates and the
        absorptance is ~2-3e-2 off ``'li'`` at ``n_orders=128``); treat an
        explicit ``'laurent'``+TM as a factorization-study mode only and use
        ``'auto'`` / ``'li'`` for real metal TM answers.
    asr_eta : float, optional
        Adaptive Spatial Resolution sharpness (Granet 1999), in ``[0, 1)``.
        ``0.0`` (default) is the standard uniform method, **bit-identical** to
        a call without the argument.  ``asr_eta > 0`` applies a matched
        coordinate stretch ``f(u) = 1 - asr_eta*cos(...)`` that clusters the
        Fourier harmonics at the grating walls.

        **When it helps:** cases where the uniform method is SLOWLY convergent
        -- lossy-metal / high-contrast TM, deep gratings -- reaching a given
        accuracy at far fewer orders (e.g. ~10x lower TM error and ~100x lower
        TE error at ``n_orders=12`` on a gold grating; ASR at 12 orders beats
        the uniform method at 24).  Validated sweet spot ``0.5-0.8``
        (geometry-dependent).

        **When it does NOT help:** ASR has an accuracy FLOOR (the matched
        coordinate + ``u<->x`` bridge plateau, ~1e-4 for TM), and its error is
        non-monotonic in ``n_orders`` (a low-order sweet spot, not
        machine-precision convergence).  For EASY / already-well-converged
        geometries (shallow, low-contrast, or simply enough orders) the uniform
        method is already below that floor, so ASR offers no benefit and can be
        marginally LESS accurate.  Enable it for hard metal/TM problems, not
        universally.  It is also a low-to-moderate-ORDER method: the bridge is
        increasingly ill-conditioned as ``n_orders`` grows (a conditioning
        warning is emitted at high order) -- use a low ``n_orders`` (its
        purpose) or disable ASR for high-order runs.

        Combined with the inverse rule (``formulation='li'`` / metals) this is
        the matched-coordinate FFF.  **Normal incidence only** (raises for
        ``angle != 0``); NumPy / CuPy only (raises on the JAX path).
    asr_samples : int, optional
        Uniform ``u``-grid sample count for the ASR metric / permittivity /
        bridge FFTs (default 16384).  Only used when ``asr_eta > 0``.

    Returns
    -------
    orders : (2*n_orders+1,) int ndarray
        Diffraction-order indices, ascending.
    R_eff : (2*n_orders+1,) float ndarray
        Reflected diffraction efficiency per order (evanescent orders 0).
    T_eff : (2*n_orders+1,) float ndarray
        Transmitted diffraction efficiency per order.

    Notes
    -----
    ``stabilize`` (default ``False``): the large-period / high-contrast
    instability is a near-singular layer<->region mode-match at a
    *measure-zero, erratic* set of ``n_orders`` -- the clean truncations sit
    immediately next to the bad ones (e.g. ``n_orders`` itself blows up while
    ``n_orders + 1`` conserves energy to 1e-6), and going to *higher*
    ``n_orders`` is not monotonically safer.  ``stabilize=True`` therefore
    searches the nearby truncations ``n_orders + {0, 1, 2, 3, 4, 6, 8}`` and
    returns the first energy-conserving solve (so the returned order count
    may differ from ``2*n_orders+1``); it raises only if none conserve.  With
    the default ``False`` the guard raises immediately (bit-for-bit backward
    compatible).  NumPy / CuPy only; the JAX path is unchanged.
    """
    angle = _resolve_incidence(angle, theta)
    if stabilize and not _is_traced(wavelength):
        last = None
        for bump in _stabilize_bumps(n_orders):
            try:
                with warnings.catch_warnings(record=True) as wlist:
                    warnings.simplefilter("always")
                    res = rcwa_efficiency_1d(
                        period, n_ridge, n_groove, n_substrate, n_superstrate,
                        depth, duty_cycle, wavelength, angle=angle,
                        polarization=polarization,
                        n_orders=int(n_orders) + bump,
                        formulation=formulation, stabilize=False,
                        use_gpu=use_gpu,
                        asr_eta=asr_eta, asr_samples=asr_samples)  # fwd ASR
            except _EnergyError as e:
                last = e
                continue
            # Re-emit unrelated warnings; a lossless-closure _EnergyWarning
            # means the per-order answers are wrong (audit P1: the silent
            # 1e-6..0.05 window) -> treat as a failed attempt and keep
            # laddering instead of returning the byte-identical wrong answer.
            closure = None
            for w in wlist:
                if issubclass(w.category, _EnergyWarning):
                    closure = closure or w
                else:
                    warnings.warn_explicit(w.message, w.category, w.filename,
                                           w.lineno)
            if closure is not None:
                last = _EnergyError(str(closure.message))
                continue
            return res
        raise last
    polarization = _normalize_pol("rcwa_efficiency_1d", polarization)
    if not (0.0 <= float(duty_cycle) <= 1.0):
        raise ValueError(
            f"rcwa_efficiency_1d: duty_cycle must be in [0, 1], got "
            f"{duty_cycle}.")
    if formulation not in ("auto", "laurent", "li"):
        raise ValueError(
            f"rcwa_efficiency_1d: formulation must be 'auto', 'laurent' or "
            f"'li', got {formulation!r}.")

    xp = _rcwa_xp("rcwa_efficiency_1d", use_gpu, n_ridge, n_groove,
                  n_substrate, n_superstrate, depth, angle, wavelength)
    is_jax = backend_name(xp) == "jax"
    if is_jax:
        _require_jax_x64("rcwa_efficiency_1d")
    _validate_geometry(
        "rcwa_efficiency_1d",
        **_concrete(period=period, depth=depth, wavelength=wavelength),
        n_orders=n_orders)

    # --- factorization choice (uses the PUBLIC n = n + i kappa) ----------
    def _metallic(n):
        try:
            nv = complex(n)
        except Exception:
            return False  # traced (JAX): assume dielectric; use formulation='li'
        return (nv.imag > 1e-6) or (nv.real ** 2 - nv.imag ** 2 < 0)
    is_metal = _metallic(n_ridge) or _metallic(n_groove)
    if formulation == "auto":
        use_li = (polarization == "tm") or is_metal
    else:
        use_li = (formulation == "li")

    # Convention bridge: the eigenmode/S-matrix core is derived in the
    # engineering exp(+i omega t) convention (forward wave exp(-i kz z),
    # layer propagator exp(-lam k0 L) with Re(lam) >= 0), in which a passive
    # absorber has Im(eps) < 0.  LumenAiry's public convention is
    # exp(-i omega t) with n = n + i kappa (kappa > 0), i.e. Im(eps) > 0.
    # Conjugating the indices maps a public absorber to the internal loss
    # sign.  Done in the ACTIVE namespace so the JAX path stays differentiable
    # w.r.t. the index values; lossless (real n) is bit-identical to v5.5.0.
    n_ridge = xp.conj(xp.asarray(n_ridge).astype(_C))
    n_groove = xp.conj(xp.asarray(n_groove).astype(_C))
    n_inc = xp.conj(xp.asarray(n_superstrate).astype(_C))
    eps_sup = xp.conj(xp.asarray(n_superstrate).astype(_C)) ** 2
    eps_sub = xp.conj(xp.asarray(n_substrate).astype(_C)) ** 2

    M = int(n_orders)
    N = 2 * M + 1
    orders = xp.arange(-M, M + 1)
    kx0 = xp.real(n_inc) * xp.sin(angle)

    # The grazing/non-propagating guards need concrete numbers.  On the JAX
    # path the GEOMETRY (angle, wavelength, region indices) is normally
    # concrete -- only the layer indices / depth are traced -- so the guards
    # still run against the concrete super/substrate, catching the dominant
    # region Rayleigh anomaly (and non-propagating incidence) instead of
    # silently returning NaN that poisons the gradient.  The layer-index
    # grazing term is included only when those indices are concrete.
    geom_concrete = not (_is_traced(kx0) or _is_traced(wavelength))
    if not is_jax or geom_concrete:
        _require_propagating_incidence("rcwa_efficiency_1d", complex(eps_sup),
                                       complex(kx0) ** 2)
        eps_reals = [complex(eps_sup), complex(eps_sub)]
        if not _is_traced(n_ridge):
            eps_reals += [complex(n_ridge) ** 2, complex(n_groove) ** 2]
        wl_eff = _grazing_safe_wavelength(
            float(wavelength), float(xp.real(kx0)), 0.0, np.arange(-M, M + 1),
            np.zeros(N), period, 1.0, eps_reals)
    else:
        wl_eff = wavelength

    k0 = 2.0 * np.pi / wl_eff
    # Tangential wavevector normalised by k0; planar mounting -> ky = 0.
    # Grating equation kx_m = kx0 + m * lambda/period (standard order sign:
    # order +m carries the +m'th grating vector G = 2*pi/period), matching
    # the diffraction-order labelling used across the RCWA literature.
    kx = kx0 + orders * (wl_eff / period)
    Kx = xp.diag(kx.astype(_C))
    Ky = xp.zeros((N, N), dtype=_C)

    # --- convolution matrices (+ optional ASR metric) -------------------
    # Adaptive Spatial Resolution: a coordinate stretch f(u) concentrates the
    # harmonics at the grating walls (see the section above _asr_metric_profile).
    # The metric enters ONLY on the derivative (Kx_layer = [[1/f]] @ Kx); the
    # permittivity is the plain eps(x(u)) on the u-grid; the layer modes are
    # bridged back to the physical-x region basis by G^{-1} before the
    # interface.  asr_eta == 0 is the exact uniform path (no branch taken).
    use_asr = float(asr_eta) != 0.0
    Gbridge = None
    Kx_layer = Kx
    if use_asr:
        if not (0.0 <= float(asr_eta) < 1.0):
            raise ValueError(
                f"rcwa_efficiency_1d: asr_eta must be in [0, 1) (eta>=1 makes "
                f"the coordinate Jacobian f=1-eta*cos touch zero), got "
                f"{asr_eta}.")
        if is_jax:
            raise ValueError(
                "rcwa_efficiency_1d: asr_eta>0 (Adaptive Spatial Resolution) is "
                "NumPy/CuPy only; it is not supported on the JAX path.")
        if abs(float(xp.real(kx0))) > 1e-12:
            raise ValueError(
                "rcwa_efficiency_1d: asr_eta>0 (Adaptive Spatial Resolution) is "
                "currently implemented for normal incidence only (angle=0).")
        Fi, EPS, EPS_II, Gbridge = _asr_convolutions(
            n_ridge, n_groove, duty_cycle, M, float(asr_eta), xp,
            n_samples=int(asr_samples))
        Kx_layer = Fi @ Kx                       # metric x derivative
        # The u<->x bridge G is increasingly ill-conditioned as n_orders grows
        # (high-order x-harmonics map outside the u-truncation window), so the
        # ASR benefit is a LOW-to-MODERATE-order effect and high-order ASR can
        # be LESS accurate than the uniform solver.  Warn (never silently wrong)
        # when G enters the unreliable regime; the result is still returned.
        _cond_G = float(xp.linalg.cond(Gbridge))
        if _cond_G > 1e8:
            warnings.warn(
                f"rcwa_efficiency_1d: ASR coordinate-bridge conditioning is "
                f"poor (cond={_cond_G:.1e}) at n_orders={M}, asr_eta={asr_eta}; "
                f"ASR is a low-to-moderate-order accelerator and the result "
                f"here may be less accurate than the uniform solver. Reduce "
                f"n_orders or asr_eta, or disable ASR (asr_eta=0) for high "
                f"order counts.", stacklevel=2)
    else:
        EPS, EPS_II = _binary_grating_convolutions(n_ridge, n_groove,
                                                   duty_cycle, M, use_li=use_li)
    # Wall-normal E_x uses the Li inverse rule [[1/eps]]^{-1} when requested
    # (TM / metals); E_y (tangential) always uses the Laurent [[eps]].  EPS_II is
    # None when use_li is False (the inverse rule was skipped) -- never read here.
    EPS_normal = EPS_II if use_li else EPS

    # --- region (half-space) modes (physical-x basis, UNCHANGED by ASR) -
    Wref, Vref, kz_ref = _homogeneous_eigenmodes(Kx, Ky, eps_sup)
    Wtrn, Vtrn, kz_trn = _homogeneous_eigenmodes(Kx, Ky, eps_sub)

    # --- global S = (sup|layer) * propagate(layer) * (layer|sub) --------
    Wl, Vl, lam = _layer_eigenmodes(Kx_layer, Ky, EPS, EPS_normal)
    if Gbridge is not None:
        # Map the layer's u-basis modes to the physical-x Rayleigh basis the
        # regions use (direction is G^{-1}; applying G is silently WRONG).
        Gi = xp.linalg.inv(Gbridge)
        zN = xp.zeros_like(Gi)
        Giblk = _block(xp, [[Gi, zN], [zN, Gi]])
        Wl = Giblk @ Wl
        Vl = Giblk @ Vl
    S = _interface_smatrix(Wref, Vref, Wl, Vl)
    S = _redheffer_star(S, _propagation_smatrix(lam, k0 * depth))
    S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wtrn, Vtrn))
    S11, S12, S21, S22 = S

    # --- incident field (delta on 0th order, chosen polarization) -------
    delta = (orders == 0).astype(_C)             # unit on the 0th order
    zeros_N = xp.zeros(N, dtype=_C)
    if polarization == "te":
        cinc = xp.concatenate([zeros_N, delta])   # E along y
    else:
        cinc = xp.concatenate([delta, zeros_N])   # E along x
    # Source is given in the reflection-region eigenbasis (W_ref = I).
    r = S11 @ cinc            # reflected tangential-E mode amplitudes
    t = S21 @ cinc            # transmitted

    rx, ry = r[:N], r[N:]
    tx, ty = t[:N], t[N:]
    kyv = xp.diag(Ky)
    # PUBLIC-convention forward kz (Re >= 0) for the z-FLUX, propagating mask, and
    # longitudinal field.  The internal exp(+iwt) loss bridge conjugates the region
    # eps, and _sqrt_forward (Im >= 0 branch) then returns Re(kz) < 0 for a LOSSY
    # half-space (a 4th-quadrant conjugated eps), which the Re(kz) > 0 propagating
    # mask wrongly read as evanescent -> T silently ZEROED into any absorbing exit
    # substrate (a long-standing energy-corruption bug; R unaffected).  Un-conjugate
    # the region eps here so a forward wave into a lossy medium keeps Re(kz) >= 0
    # and carries its physical z-flux; LOSSLESS eps is real so conj is identity and
    # this is byte-unchanged.  (The eigenmode kz_ref/kz_trn keep the internal
    # convention for the mode match -- that path is already correct.)
    kz_ref_f = _forward_flux_kz(eps_sup, kx, kyv)
    kz_trn_f = _forward_flux_kz(eps_sub, kx, kyv)
    safe_r = xp.where(xp.abs(kz_ref_f) < 1e-12, 1.0, kz_ref_f)
    safe_t = xp.where(xp.abs(kz_trn_f) < 1e-12, 1.0, kz_trn_f)
    # Longitudinal field from div(D) = 0 in each homogeneous region: a
    # diffracted order's full E carries Ez = -(kx Ex + ky Ey)/kz, so the
    # transverse-only |E_t|^2 understates the power by |Ez|^2.
    rz = -(kx * rx + kyv * ry) / safe_r
    tz = -(kx * tx + kyv * ty) / safe_t

    # The injected source has UNIT tangential amplitude (Ex=1 for TM, Ey=1
    # for TE) but the physical incident plane wave also carries a
    # longitudinal Ez_inc, so |E_inc|^2 = 1 + (kx0/kz_inc)^2 = sec^2(theta)
    # for TM (and exactly 1 for TE).  Normalising the diffraction
    # efficiencies by this incident |E|^2 is what keeps sum(R)+sum(T)=1 at
    # oblique TM (without it the sums scale as sec^2 theta).
    kz_inc = xp.real(_sqrt_forward(xp.conj(eps_sup) - kx0 ** 2))
    if polarization == "te":
        einc_sq = 1.0
    else:
        einc_sq = 1.0 + (kx0 / kz_inc) ** 2
    R_eff = xp.real(kz_ref_f / kz_inc) * (xp.abs(rx) ** 2 + xp.abs(ry) ** 2
                                          + xp.abs(rz) ** 2) / einc_sq
    T_eff = xp.real(kz_trn_f / kz_inc) * (xp.abs(tx) ** 2 + xp.abs(ty) ** 2
                                          + xp.abs(tz) ** 2) / einc_sq
    R_eff = xp.where(xp.real(kz_ref_f) > 0, xp.real(R_eff), 0.0)
    T_eff = xp.where(xp.real(kz_trn_f) > 0, xp.real(T_eff), 0.0)
    if not is_jax:
        # Provably lossless (every permittivity exactly real) => the closure
        # R+T = 1 is exact; _check_energy then warns in the silent window
        # 1e-6..0.05 where the per-order answers are wrong (audit P1).
        try:
            lossless = all(
                float(np.imag(complex(v))) == 0.0
                for v in (complex(eps_sup), complex(eps_sub),
                          _C(n_ridge) ** 2, _C(n_groove) ** 2))
        except TypeError:                  # traced / array-valued inputs
            lossless = False
        _check_energy("rcwa_efficiency_1d", R_eff, T_eff, lossless=lossless)
    return orders, R_eff, T_eff



@_with_blas_limit
def rcwa_efficiency_vs_wavelength(
    period: float,
    n_ridge: complex,
    n_groove: complex,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    duty_cycle: float,
    wavelengths,
    *,
    order: int = 1,
    angle: float = 0.0,
    theta: float | None = None,
    polarization: str = "te",
    n_orders: int = 11,
    formulation: str = "auto",
    quantity: str = "transmitted",
    stabilize: bool = False,
) -> np.ndarray:
    """Rigorous diffraction efficiency of a single order across a wavelength
    sweep -- the spectral companion to :func:`rcwa_efficiency_1d`, mirroring
    :func:`lumenairy.elements.thin_grating.grating_efficiency_vs_wavelength`
    but with the full vector solver (metals, TM, high contrast).

    Parameters
    ----------
    order : int, optional
        Diffraction order whose efficiency is returned (default ``+1``).
    quantity : {'transmitted', 'reflected'}, optional
        Which side's efficiency to return for ``order``.  Default
        ``'transmitted'``.
    wavelengths : float or array-like
        Vacuum wavelength(s) [m].  Scalar indices are assumed dispersionless
        across the sweep; for dispersive media call :func:`rcwa_efficiency_1d`
        per wavelength with the wavelength-specific indices.

    Returns
    -------
    eff : ndarray
        Efficiency of ``order`` at each wavelength (shape matches
        ``wavelengths``); ``0`` where the order is evanescent.

    Other parameters are as in :func:`rcwa_efficiency_1d`.
    """
    angle = _resolve_incidence(angle, theta)
    if quantity not in ("transmitted", "reflected"):
        raise ValueError(
            f"rcwa_efficiency_vs_wavelength: quantity must be 'transmitted' "
            f"or 'reflected', got {quantity!r}.")
    # Validate geometry HERE so the error carries this function's prefix
    # (otherwise the failure surfaces with the inner rcwa_efficiency_1d
    # prefix from the per-wavelength call, confusing the caller).
    _validate_geometry("rcwa_efficiency_vs_wavelength", period=period,
                       depth=depth, n_orders=n_orders)
    wl = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    if wl.size == 0:
        raise ValueError(
            "rcwa_efficiency_vs_wavelength: wavelengths is empty; pass at "
            "least one wavelength [m].")
    if not np.all(np.isfinite(wl)) or np.any(wl <= 0.0):
        raise ValueError(
            "rcwa_efficiency_vs_wavelength: every wavelength must be a finite "
            "value > 0 [m].")
    out = np.empty(wl.shape[0], dtype=float)
    n_unstable = 0
    for i, w in enumerate(wl):
        try:
            orders, R, T = rcwa_efficiency_1d(
                period, n_ridge, n_groove, n_substrate, n_superstrate, depth,
                duty_cycle, float(w), angle=angle, polarization=polarization,
                n_orders=n_orders, formulation=formulation, stabilize=stabilize)
        except _EnergyError:
            out[i] = np.nan          # one unstable wavelength must NOT abort the
            n_unstable += 1          # whole sweep (audit P2); flag it and continue
            continue
        idx = np.searchsorted(orders, order)
        if idx >= orders.shape[0] or orders[idx] != order:
            raise ValueError(
                f"rcwa_efficiency_vs_wavelength: order {order} is outside the "
                f"retained range +/-{n_orders}; increase n_orders.")
        out[i] = (T[idx] if quantity == "transmitted" else R[idx])
    if n_unstable:
        warnings.warn(
            f"rcwa_efficiency_vs_wavelength: {n_unstable}/{wl.size} wavelengths hit "
            f"a numerical instability (energy non-conservation) and were set to "
            f"NaN; pass stabilize=True or adjust n_orders / period.", stacklevel=2)
    return out if np.ndim(wavelengths) else out[0]



def rcwa_jones_vs_wavelength(
    period: float,
    eps_ridge,
    eps_groove,
    n_substrate,
    n_superstrate,
    depth: float,
    duty_cycle: float,
    wavelengths,
    *,
    angle: float = 0.0,
    theta: float | None = None,
    n_orders: int = 11,
):
    """DISPERSIVE Jones spectral sweep of the 1-D anisotropic grating -- the
    Jones companion to :func:`rcwa_efficiency_vs_wavelength` (which is scalar +
    dispersionless).

    Each of ``eps_ridge``, ``eps_groove``, ``n_substrate``, ``n_superstrate``
    may be a FIXED value or a CALLABLE ``wl -> value`` (so material dispersion is
    handled by passing ``n(lambda)`` / ``eps(lambda)`` closures -- e.g. from the
    bundled ``refractiveindex`` database).

    Parameters
    ----------
    eps_ridge, eps_groove : (3, 3) array_like or callable
        Ridge / groove permittivity tensors (PUBLIC ``Im(eps) > 0``), or a
        ``wl -> (3, 3)`` callable for a dispersive medium.
    n_substrate, n_superstrate : complex or callable
        Half-space indices, or ``wl -> complex`` callables.
    wavelengths : float or array-like
        Vacuum wavelength(s) [m].
    angle, n_orders
        As in :func:`rcwa_jones_1d`.

    Returns
    -------
    wavelengths : ndarray
        The sweep grid (scalar in -> scalar out).
    jones : (Nwl, 2, 2) complex ndarray
        Zeroth-order Jones reflection at each wavelength.
    R_total, T_total : (Nwl, 2) float ndarray
        Total reflected / transmitted efficiency (summed over orders) for each
        incident polarization (column 0 = incident ``E_x``, 1 = ``E_y``).
    """
    angle = _resolve_incidence(angle, theta)
    wl = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    if wl.size == 0 or not np.all(np.isfinite(wl)) or np.any(wl <= 0.0):
        raise ValueError(
            "rcwa_jones_vs_wavelength: every wavelength must be finite and > 0 "
            "[m] (got an empty or invalid sweep).")

    def _at(v, w):
        return v(w) if callable(v) else v

    J = np.empty((wl.size, 2, 2), dtype=_C)
    Rt = np.empty((wl.size, 2), dtype=float)
    Tt = np.empty((wl.size, 2), dtype=float)
    for i, w in enumerate(wl):
        _o, R, T, jr = rcwa_jones_1d(
            period, _at(eps_ridge, w), _at(eps_groove, w),
            _at(n_substrate, w), _at(n_superstrate, w), depth, duty_cycle,
            float(w), angle=angle, n_orders=n_orders)
        J[i] = np.asarray(to_numpy(jr))
        Rt[i] = np.asarray(to_numpy(R)).sum(axis=1)
        Tt[i] = np.asarray(to_numpy(T)).sum(axis=1)
    if np.ndim(wavelengths):
        return wl, J, Rt, Tt
    return wl[0], J[0], Rt[0], Tt[0]



def rcwa_jones_vs_wavelength_segments(
    period: float,
    segments,
    n_substrate,
    n_superstrate,
    depth: float,
    wavelengths,
    *,
    angle: float = 0.0,
    theta: float | None = None,
    n_orders: int = 11,
):
    """DISPERSIVE Jones spectral sweep of a MULTI-SEGMENT 1-D anisotropic grating
    -- the arbitrary-profile generalisation of :func:`rcwa_jones_vs_wavelength`
    (a binary ridge / groove cell), looping :func:`rcwa_jones_1d_segments` over
    ``wavelengths``.

    Each segment's ``eps`` (and ``n_substrate`` / ``n_superstrate``) may be a
    FIXED value or a CALLABLE ``wl -> value`` (so material dispersion is handled
    by passing ``n(lambda)`` / ``eps(lambda)`` closures -- e.g. from the bundled
    ``refractiveindex`` database); the per-segment ``width_fraction`` is fixed.

    Parameters
    ----------
    period : float
        Grating period (metres).
    segments : list of (width_fraction, eps_or_callable)
        Consecutive regions covering one period (the
        :func:`rcwa_jones_1d_segments` profile); each ``eps`` is a scalar /
        ``(3, 3)`` tensor, or a ``wl -> (scalar | (3, 3))`` callable for a
        dispersive region.  The ``width_fraction`` values must sum to ``1``.
    n_substrate, n_superstrate : complex or callable
        Half-space indices, or ``wl -> complex`` callables.
    depth : float
        Grating thickness (metres).
    wavelengths : float or array-like
        Vacuum wavelength(s) [m].
    angle, n_orders
        As in :func:`rcwa_jones_1d_segments`.

    Returns
    -------
    wavelengths : ndarray
        The sweep grid (scalar in -> scalar out).
    jones : (Nwl, 2, 2) complex ndarray
        Zeroth-order Jones reflection at each wavelength.
    R_total, T_total : (Nwl, 2) float ndarray
        Total reflected / transmitted efficiency (summed over orders) for each
        incident polarization (column 0 = incident ``E_x``, 1 = ``E_y``).
    """
    angle = _resolve_incidence(angle, theta)
    wl = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    if wl.size == 0 or not np.all(np.isfinite(wl)) or np.any(wl <= 0.0):
        raise ValueError(
            "rcwa_jones_vs_wavelength_segments: every wavelength must be finite "
            "and > 0 [m] (got an empty or invalid sweep).")
    seg_list = list(segments)
    if len(seg_list) == 0:
        raise ValueError(
            "rcwa_jones_vs_wavelength_segments: segments must be a non-empty "
            "list of (width_fraction, eps) pairs.")

    def _at(v, w):
        return v(w) if callable(v) else v

    J = np.empty((wl.size, 2, 2), dtype=_C)
    Rt = np.empty((wl.size, 2), dtype=float)
    Tt = np.empty((wl.size, 2), dtype=float)
    for i, w in enumerate(wl):
        segs_w = [(width, _at(eps, w)) for (width, eps) in seg_list]
        _o, R, T, jr = rcwa_jones_1d_segments(
            period, segs_w, _at(n_substrate, w), _at(n_superstrate, w),
            depth, float(w), angle=angle, n_orders=n_orders)
        J[i] = np.asarray(to_numpy(jr))
        Rt[i] = np.asarray(to_numpy(R)).sum(axis=1)
        Tt[i] = np.asarray(to_numpy(T)).sum(axis=1)
    if np.ndim(wavelengths):
        return wl, J, Rt, Tt
    return wl[0], J[0], Rt[0], Tt[0]



def _jones_1d_from_profiles(profiles, offplane, *, M, orders, Kx, Ky, kxv, k0,
                            eps_sup, eps_sub, kz_inc, depth, kx0, xp, is_jax,
                            fn_name):
    """Shared 1-D anisotropic Jones solve core (binary or multi-segment).

    Given the per-component one-period ``profiles`` (5 keys for the in-plane
    path; 9 keys when ``offplane`` is True) and the already-set-up modal grid
    (``Kx, Ky, kxv, k0`` and the half-space ``eps_sup / eps_sub / kz_inc``),
    build the convolutions, layer eigenmodes, region/layer S-matrix (the
    general full-tensor branch or the in-plane branch), then the R/T/Jones
    efficiency tail.  Returns ``(orders, R_eff, T_eff, jones_reflection)``.

    Factored out of :func:`rcwa_jones_1d` so that
    :func:`rcwa_jones_1d_segments` reuses the EXACT same core; the binary and
    multi-segment callers differ only in how they sample ``profiles``.  Keeps
    the JAX-differentiable stack-based structure (no in-place assignment).
    """
    N = 2 * M + 1
    if offplane:
        # ---- FULL-3x3 (out-of-plane) path (Li 2003) ------------------------
        # Sample all nine component profiles, build the full convolutions +
        # generator eigenmodes (explicit forward/backward), and assemble the
        # half-space regions as [W; -V] (the in-plane symmetry holds for an
        # isotropic half-space) and the layer via the GENERAL S-matrix.
        Cxx, Cxy, Cyx, Cyy, EZZ, EZX, EZY, EXZ, EYZ = \
            _tensor_convolutions_full(profiles, M)
        Wref, Vref, kz_ref = _homogeneous_eigenmodes(Kx, Ky, eps_sup)
        Wtrn, Vtrn, kz_trn = _homogeneous_eigenmodes(Kx, Ky, eps_sub)
        Wl, Vl, lam, Wlb, Vlb, lam_b = _layer_eigenmodes_tensor(
            Kx, Ky, Cxx, Cxy, Cyx, Cyy, EZZ, EZX, EZY, EXZ, EYZ)
        Mref = _modes_to_M(Wref, Vref, Wref, -Vref)
        Mtrn = _modes_to_M(Wtrn, Vtrn, Wtrn, -Vtrn)
        Ml = _modes_to_M(Wl, Vl, Wlb, Vlb)
        S = _interface_smatrix_general(Mref, Ml)
        S = _redheffer_star(
            S, _propagation_smatrix_general(lam, lam_b, k0 * depth))
        S = _redheffer_star(S, _interface_smatrix_general(Ml, Mtrn))
        S11, S12, S21, S22 = S
    else:
        Cxx, Cxy, Cyx, Cyy, EZZ = _tensor_convolutions(profiles, M)

        Wref, Vref, kz_ref = _homogeneous_eigenmodes(Kx, Ky, eps_sup)
        Wtrn, Vtrn, kz_trn = _homogeneous_eigenmodes(Kx, Ky, eps_sub)
        Wl, Vl, lam = _layer_eigenmodes_tensor(Kx, Ky, Cxx, Cxy, Cyx, Cyy, EZZ)
        S = _interface_smatrix(Wref, Vref, Wl, Vl)
        S = _redheffer_star(S, _propagation_smatrix(lam, k0 * depth))
        S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wtrn, Vtrn))
        S11, S12, S21, S22 = S

    delta = xp.asarray((orders == 0).astype(_C))
    zeros_N = xp.zeros(N, dtype=_C)
    # Build the two incident-polarization responses then STACK (no item
    # assignment, so the path is JAX-differentiable as well as GPU-ready).
    R_rows, T_rows, j_cols = [], [], []
    for pol in ("x", "y"):
        if pol == "x":
            cinc = xp.concatenate([delta, zeros_N])
            einc_sq = 1.0 + (kx0 / kz_inc) ** 2 if kz_inc != 0 else 1.0
        else:
            cinc = xp.concatenate([zeros_N, delta])
            einc_sq = 1.0
        r = S11 @ cinc
        t = S21 @ cinc
        rx, ry = r[:N], r[N:]
        tx, ty = t[:N], t[N:]
        # PUBLIC-convention forward kz for the z-flux + mask + Ez (see
        # _forward_flux_kz): the internal-conjugated lossy substrate would
        # otherwise zero the transmittance.  1-D: ky = 0.
        kz_ref_f = _forward_flux_kz(eps_sup, kxv, 0.0)
        kz_trn_f = _forward_flux_kz(eps_sub, kxv, 0.0)
        safe_r = xp.where(xp.abs(kz_ref_f) < 1e-12, 1.0, kz_ref_f)
        safe_t = xp.where(xp.abs(kz_trn_f) < 1e-12, 1.0, kz_trn_f)
        rz = -(kxv * rx) / safe_r
        tz = -(kxv * tx) / safe_t
        Re = xp.real(kz_ref_f / kz_inc) * (xp.abs(rx) ** 2 + xp.abs(ry) ** 2
                                           + xp.abs(rz) ** 2) / einc_sq
        Te = xp.real(kz_trn_f / kz_inc) * (xp.abs(tx) ** 2 + xp.abs(ty) ** 2
                                           + xp.abs(tz) ** 2) / einc_sq
        R_rows.append(xp.where(xp.real(kz_ref_f) > 0, xp.real(Re), 0.0))
        T_rows.append(xp.where(xp.real(kz_trn_f) > 0, xp.real(Te), 0.0))
        # Zeroth-order Jones column (conjugate back to public exp(-i w t)).
        j_cols.append(xp.stack([xp.conj(rx[M]), xp.conj(ry[M])]))
    R_eff = xp.stack(R_rows)                       # (2, N)
    T_eff = xp.stack(T_rows)
    jones_reflection = xp.stack(j_cols, axis=1)    # (2, 2): columns = pol
    if not is_jax:
        _check_energy(fn_name, R_eff, T_eff)
    return orders, R_eff, T_eff, jones_reflection



@_with_blas_limit
def rcwa_jones_1d(
    period: float,
    eps_ridge,
    eps_groove,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    duty_cycle: float,
    wavelength: float,
    *,
    angle: float = 0.0,
    theta: float | None = None,
    n_orders: int = 11,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rigorous 1-D anisotropic grating: a binary grating whose ridge and
    groove are full ``(3, 3)`` permittivity tensors (the liquid-crystal /
    birefringent case).  Because the in-plane tensor couples TE and TM, the
    response is a full Jones matrix.

    Backend-dispatched (NumPy / CuPy via ``use_gpu`` / differentiable JAX when
    a tensor argument is a JAX array); see :func:`rcwa_efficiency_1d`.

    Parameters
    ----------
    period : float
        Grating period (metres).
    eps_ridge, eps_groove : (3, 3) array_like of complex
        PERMITTIVITY tensors of the ridge and groove (PUBLIC convention
        ``Im(eps) > 0`` for loss).  Pass ``scalar * np.eye(3)`` for an
        isotropic region; build LC tensors with :func:`uniaxial_tensor`.
        CONVENTION WARNING: the Jones family takes PERMITTIVITY ``eps = n**2`` here,
        whereas the scalar :func:`rcwa_efficiency_1d` takes the refractive INDEX
        ``n`` -- a wrong-convention value is silently accepted, so square your index
        (``n**2``) for this function.
    n_substrate, n_superstrate : complex
        Transmission / incidence half-space (isotropic) indices.
    depth, duty_cycle, wavelength, angle, n_orders
        As in :func:`rcwa_efficiency_1d` (ridge occupies ``duty_cycle`` of
        the period; planar incidence at ``angle``).

    Returns
    -------
    orders : (2*n_orders+1,) int ndarray
        Diffraction-order indices.
    R_eff, T_eff : (2, 2*n_orders+1) float ndarray
        Reflected / transmitted diffraction efficiency per order; row 0 is
        the response to an incident ``E_x`` wave, row 1 to incident
        ``E_y``.
    jones_reflection : (2, 2) complex ndarray
        Zeroth-order Jones reflection matrix in the lab ``(x, y)`` basis
        (PUBLIC ``exp(-i w t)`` convention); columns are the responses to
        incident ``E_x`` / ``E_y``, rows are ``[E_x; E_y]`` reflected.
    """
    angle = _resolve_incidence(angle, theta)
    _validate_geometry("rcwa_jones_1d",
                       **_concrete(period=period, depth=depth,
                                   wavelength=wavelength), n_orders=n_orders)
    if not (0.0 <= float(duty_cycle) <= 1.0):
        raise ValueError(
            f"rcwa_jones_1d: duty_cycle must be in [0, 1], got {duty_cycle}.")
    # Out-of-plane (full-3x3) tensors are allowed on the 1-D path (v5.11.0);
    # the flag routes to the general full-tensor solver below.  In-plane tensors
    # keep the existing path bit-identical.
    offplane = _require_inplane_tensor("rcwa_jones_1d", eps_ridge, eps_groove,
                                       allow_offplane=True)

    xp = _rcwa_xp("rcwa_jones_1d", use_gpu, eps_ridge, eps_groove)
    is_jax = backend_name(xp) == "jax"
    if is_jax:
        _require_jax_x64("rcwa_jones_1d")
        # See _reject_jax_offplane: the full-3x3 path is non-differentiable and
        # a JAX off-plane tensor would otherwise be silently treated as in-plane.
        _reject_jax_offplane("rcwa_jones_1d", eps_ridge, eps_groove)
    M = int(n_orders)
    N = 2 * M + 1
    orders = np.arange(-M, M + 1)

    # Loss-convention bridge: conjugate the PUBLIC tensors in the active
    # namespace (differentiable for JAX); region scalars stay host complex.
    eps_ridge = xp.conj(xp.asarray(eps_ridge).astype(_C))
    eps_groove = xp.conj(xp.asarray(eps_groove).astype(_C))
    eps_sup = complex(np.conj(_C(n_superstrate) ** 2))
    eps_sub = complex(np.conj(_C(n_substrate) ** 2))

    kx0 = float(np.real(np.conj(_C(n_superstrate))) * np.sin(angle))
    # Guards run on the concrete geometry (angle/wavelength are not tensor
    # arguments here, so always concrete); the region Rayleigh anomaly and
    # non-propagating incidence are caught on JAX too.  The tensor-layer
    # diagonal permittivities are added to the nudge only when concrete.
    if not is_jax or not _is_traced(wavelength):
        _require_propagating_incidence("rcwa_jones_1d", eps_sup, kx0 ** 2)
        eps_reals = [eps_sup, eps_sub]
        if not is_jax:
            eps_reals += [complex(eps_ridge[0, 0]), complex(eps_ridge[1, 1]),
                          complex(eps_ridge[2, 2]), complex(eps_groove[0, 0]),
                          complex(eps_groove[1, 1]), complex(eps_groove[2, 2])]
        wl_eff = _grazing_safe_wavelength(
            float(wavelength), kx0, 0.0, orders, np.zeros_like(orders), period,
            1.0, eps_reals)
    else:
        wl_eff = wavelength
    k0 = 2.0 * np.pi / wl_eff
    kx = kx0 + orders * (wl_eff / period)
    Kx = xp.asarray(np.diag(kx.astype(_C)))
    Ky = xp.zeros((N, N), dtype=_C)
    kxv = xp.asarray(kx.astype(_C))

    # Sample the per-component profiles across one period (ridge over duty).
    n_samples = 4096
    xq = (xp.arange(n_samples) + 0.5) / n_samples
    inside = xq < duty_cycle
    if offplane:
        # ---- FULL-3x3 (out-of-plane) path (Li 2003): sample all nine
        # component profiles (ridge over duty), the rest is the shared core.
        profiles = {}
        for key, (ii, jj) in {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0),
                              "yy": (1, 1), "zz": (2, 2), "xz": (0, 2),
                              "zx": (2, 0), "yz": (1, 2), "zy": (2, 1)}.items():
            profiles[key] = xp.where(inside, eps_ridge[ii, jj],
                                     eps_groove[ii, jj]).astype(_C)
    else:
        profiles = {}
        for key, (ii, jj) in {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0),
                              "yy": (1, 1), "zz": (2, 2)}.items():
            profiles[key] = xp.where(inside, eps_ridge[ii, jj],
                                     eps_groove[ii, jj]).astype(_C)

    kz_inc = float(np.real(_sqrt_forward(np.conj(eps_sup) - kx0 ** 2)))
    return _jones_1d_from_profiles(
        profiles, offplane, M=M, orders=orders, Kx=Kx, Ky=Ky, kxv=kxv, k0=k0,
        eps_sup=eps_sup, eps_sub=eps_sub, kz_inc=kz_inc, depth=depth, kx0=kx0,
        xp=xp, is_jax=is_jax, fn_name="rcwa_jones_1d")



@_with_blas_limit
def rcwa_jones_1d_segments(
    period: float,
    segments,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelength: float,
    *,
    angle: float = 0.0,
    theta: float | None = None,
    n_orders: int = 11,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rigorous 1-D anisotropic grating with an ARBITRARY piecewise-constant
    profile -- the multi-region / multi-level generalisation of
    :func:`rcwa_jones_1d` (which is the 2-segment ridge/groove special case).

    A single grating layer is partitioned into consecutive regions along ``x``;
    each region carries its own (possibly anisotropic) permittivity.  This
    covers multi-level staircases (blazed-grating approximations), arbitrary
    multi-region cells, and mixed isotropic / liquid-crystal regions.  Because
    an in-plane tensor couples TE and TM the response is a full Jones matrix;
    out-of-plane (full ``3x3``) tensors route through the v5.11.0 general
    solver.  The solve shares the EXACT same core as :func:`rcwa_jones_1d`
    (:func:`_jones_1d_from_profiles`); only the profile sampling differs.

    Backend-dispatched (NumPy / CuPy via ``use_gpu`` / differentiable JAX when
    a segment tensor is a JAX array); see :func:`rcwa_efficiency_1d`.  The JAX
    path differentiates the IN-PLANE tensor subset (``exx, exy, eyx, eyy,
    ezz``); a JAX tensor with OUT-OF-PLANE coupling (``eps_xz / eps_yz /
    eps_zx / eps_zy != 0`` -- a tilted-director LC) raises
    :class:`NotImplementedError`, because the full-3x3 solver's forward-mode
    flux selection is a host ``np.where`` / ``argsort`` that breaks the autodiff
    graph (call on NumPy / CuPy for the rigorous off-plane solve).

    Parameters
    ----------
    period : float
        Grating period (metres).
    segments : list of (width_fraction, eps)
        Consecutive regions covering one period in order; the
        ``width_fraction`` values (each in ``(0, 1]``) must sum to ``1`` (within
        ``1e-6``).  Each ``eps`` is either a complex scalar (taken as
        ``scalar * I(3)``, isotropic), an in-plane ``(3, 3)`` tensor, or a full
        out-of-plane ``(3, 3)`` tensor (e.g. a tilted-director LC built with
        :func:`uniaxial_tensor`).  PUBLIC convention ``Im(eps) > 0`` for loss.
    n_substrate, n_superstrate : complex
        Transmission / incidence half-space (isotropic) indices.
    depth, wavelength, angle, n_orders
        As in :func:`rcwa_jones_1d` (planar incidence at ``angle``).

    Returns
    -------
    orders : (2*n_orders+1,) int ndarray
        Diffraction-order indices.
    R_eff, T_eff : (2, 2*n_orders+1) float ndarray
        Reflected / transmitted diffraction efficiency per order; row 0 is the
        response to an incident ``E_x`` wave, row 1 to incident ``E_y``.
    jones_reflection : (2, 2) complex ndarray
        Zeroth-order Jones reflection matrix in the lab ``(x, y)`` basis
        (PUBLIC ``exp(-i w t)`` convention); columns are the responses to
        incident ``E_x`` / ``E_y``, rows are ``[E_x; E_y]`` reflected.
    """
    angle = _resolve_incidence(angle, theta)
    _validate_geometry("rcwa_jones_1d_segments",
                       **_concrete(period=period, depth=depth,
                                   wavelength=wavelength), n_orders=n_orders)
    seg_list = list(segments)
    if len(seg_list) == 0:
        raise ValueError(
            "rcwa_jones_1d_segments: segments must be a non-empty list of "
            "(width_fraction, eps) pairs.")
    widths = []
    eps_raw = []
    for k, item in enumerate(seg_list):
        try:
            w, e = item
        except (TypeError, ValueError):
            raise ValueError(
                f"rcwa_jones_1d_segments: segment {k} must be a "
                f"(width_fraction, eps) pair, got {item!r}.") from None
        wf = float(w)
        if not np.isfinite(wf) or wf <= 0.0:
            raise ValueError(
                f"rcwa_jones_1d_segments: width_fraction of segment {k} must "
                f"be > 0, got {wf}.")
        widths.append(wf)
        eps_raw.append(e)
    total = float(np.sum(widths))
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"rcwa_jones_1d_segments: the segment width_fractions must sum to "
            f"1 (within 1e-6), got {total}.")

    # Promote scalars to scalar * I(3); leave (3, 3) tensors as-is.  Each cell
    # may independently be a host array, CuPy, or JAX array.
    eps_tensors = []
    for k, e in enumerate(eps_raw):
        arr = e
        if np.ndim(e) == 0:
            arr = _C(e) * np.eye(3, dtype=_C)
        if np.shape(arr)[-2:] != (3, 3):
            raise ValueError(
                f"rcwa_jones_1d_segments: eps of segment {k} must be a scalar "
                f"or a (3, 3) tensor, got shape {np.shape(arr)}.")
        eps_tensors.append(arr)

    # Out-of-plane present in ANY segment -> route every segment through the
    # 9-key full-tensor solver (v5.11.0).  In-plane-only stays on the legacy
    # 5-key path (bit-identical to rcwa_jones_1d for a 2-segment cell).
    offplane = _require_inplane_tensor("rcwa_jones_1d_segments", *eps_tensors,
                                       allow_offplane=True)

    xp = _rcwa_xp("rcwa_jones_1d_segments", use_gpu, *eps_tensors)
    is_jax = backend_name(xp) == "jax"
    if is_jax:
        _require_jax_x64("rcwa_jones_1d_segments")
        # The full-3x3 (out-of-plane) solver is non-differentiable (its
        # forward-mode flux split is a host np.where/argsort); the in-plane
        # router silently skips JAX, so reject a JAX off-plane tensor here.
        _reject_jax_offplane("rcwa_jones_1d_segments", *eps_tensors)
    M = int(n_orders)
    N = 2 * M + 1
    orders = np.arange(-M, M + 1)

    # Loss-convention bridge: conjugate the PUBLIC tensors in the active
    # namespace (differentiable for JAX); region scalars stay host complex.
    eps_tensors = [xp.conj(xp.asarray(t).astype(_C)) for t in eps_tensors]
    eps_sup = complex(np.conj(_C(n_superstrate) ** 2))
    eps_sub = complex(np.conj(_C(n_substrate) ** 2))

    kx0 = float(np.real(np.conj(_C(n_superstrate))) * np.sin(angle))
    # Guards run on the concrete geometry (angle/wavelength are not tensor
    # arguments here, so always concrete); the region Rayleigh anomaly and
    # non-propagating incidence are caught on JAX too.  The tensor-layer
    # diagonal permittivities are added to the nudge only when concrete.
    if not is_jax or not _is_traced(wavelength):
        _require_propagating_incidence("rcwa_jones_1d_segments", eps_sup,
                                       kx0 ** 2)
        eps_reals = [eps_sup, eps_sub]
        if not is_jax:
            for t in eps_tensors:
                eps_reals += [complex(t[0, 0]), complex(t[1, 1]),
                              complex(t[2, 2])]        # ezz too (audit P2 -- a layer
                #                          mode grazing on ezz was never nudged)
        wl_eff = _grazing_safe_wavelength(
            float(wavelength), kx0, 0.0, orders, np.zeros_like(orders), period,
            1.0, eps_reals)
    else:
        wl_eff = wavelength
    k0 = 2.0 * np.pi / wl_eff
    kx = kx0 + orders * (wl_eff / period)
    Kx = xp.asarray(np.diag(kx.astype(_C)))
    Ky = xp.zeros((N, N), dtype=_C)
    kxv = xp.asarray(kx.astype(_C))

    # Sample the per-component profiles across one period.  For sample x in
    # [0, 1) find which segment's cumulative [c_{k-1}, c_k) interval it lands
    # in and take that segment's component.  Built stack-based via nested
    # xp.where over the segments (no in-place assignment -> JAX-differentiable).
    n_samples = 4096
    xq = (xp.arange(n_samples) + 0.5) / n_samples
    cum = np.cumsum([0.0] + widths)
    cum[-1] = 1.0  # close the last interval exactly despite float roundoff

    if offplane:
        comp_map = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1),
                    "zz": (2, 2), "xz": (0, 2), "zx": (2, 0), "yz": (1, 2),
                    "zy": (2, 1)}
    else:
        comp_map = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1),
                    "zz": (2, 2)}
    profiles = {}
    for key, (ii, jj) in comp_map.items():
        # Start from the LAST segment, then fold earlier segments in REVERSE
        # order so each segment's left boundary ``xq < c_{k+1}`` is applied
        # over the wider ones -- segment k wins on [c_k, c_{k+1}) because every
        # later-applied (smaller-k) mask is the disjoint left part.
        prof = xp.full(n_samples, eps_tensors[-1][ii, jj], dtype=_C)
        for k in range(len(eps_tensors) - 2, -1, -1):
            in_seg = xq < cum[k + 1]
            prof = xp.where(in_seg, eps_tensors[k][ii, jj], prof)
        profiles[key] = prof.astype(_C)

    kz_inc = float(np.real(_sqrt_forward(np.conj(eps_sup) - kx0 ** 2)))
    return _jones_1d_from_profiles(
        profiles, offplane, M=M, orders=orders, Kx=Kx, Ky=Ky, kxv=kxv, k0=k0,
        eps_sup=eps_sup, eps_sub=eps_sub, kz_inc=kz_inc, depth=depth, kx0=kx0,
        xp=xp, is_jax=is_jax, fn_name="rcwa_jones_1d_segments")



# ===========================================================================
# W3 -- 1-D device grating builders (emit the ``segments`` list for
# rcwa_jones_1d_segments, so users don't hand-roll region masks).
# ===========================================================================
def grating_segments(widths, materials):
    """Build a ``segments`` list for :func:`rcwa_jones_1d_segments` from per-region
    widths and materials.

    ``widths`` are RELATIVE region widths (any positive units; normalized so the
    period sums to 1); ``materials`` are the matching per-region permittivities
    (a scalar for an isotropic region, or a ``(3, 3)`` tensor).  Returns
    ``[(width_fraction, eps), ...]`` -- the ``segments`` argument.  Arbitrary
    region count (2, 4, N).
    """
    widths = [float(w) for w in widths]
    materials = list(materials)
    if len(widths) != len(materials):
        raise ValueError("grating_segments: widths and materials must have the "
                         f"same length ({len(widths)} != {len(materials)}).")
    if not widths:
        raise ValueError("grating_segments: need at least one region.")
    if any(w <= 0 for w in widths):
        raise ValueError("grating_segments: all widths must be positive.")
    total = sum(widths)
    return [(w / total, eps) for w, eps in zip(widths, materials)]



def binary_grating_segments(duty_cycle, eps_ridge, eps_groove):
    """Two-region (binary) ``segments`` list: a ridge of fractional width
    ``duty_cycle`` followed by a groove -- the geometry of :func:`rcwa_jones_1d`
    expressed for :func:`rcwa_jones_1d_segments`."""
    duty = float(duty_cycle)
    if not (0.0 < duty < 1.0):
        raise ValueError(
            f"binary_grating_segments: duty_cycle must be in (0, 1), got {duty}.")
    return [(duty, eps_ridge), (1.0 - duty, eps_groove)]



def interdigitated_grating_segments(tooth_widths, gap_width, tooth_materials,
                                    gap_material):
    """``segments`` list for an interdigitated-teeth profile: each tooth (its own
    width + material) separated by a gap of width ``gap_width`` filled with
    ``gap_material`` -- the 'grounded tooth | gap | floating tooth | gap | ...'
    device pattern.  Widths are relative (normalized over the period)."""
    tooth_widths = [float(w) for w in tooth_widths]
    tooth_materials = list(tooth_materials)
    if len(tooth_widths) != len(tooth_materials):
        raise ValueError("interdigitated_grating_segments: tooth_widths and "
                         "tooth_materials must have the same length "
                         f"({len(tooth_widths)} != {len(tooth_materials)}).")
    if not tooth_widths:
        raise ValueError("interdigitated_grating_segments: need at least one tooth.")
    g = float(gap_width)
    if g <= 0 or any(w <= 0 for w in tooth_widths):
        raise ValueError("interdigitated_grating_segments: all widths must be "
                         "positive.")
    raw = []
    for w, m in zip(tooth_widths, tooth_materials):
        raw.append((w, m))
        raw.append((g, gap_material))
    total = sum(w for w, _ in raw)
    return [(w / total, eps) for w, eps in raw]



# ===========================================================================
# W2 -- reflective-Jones device helpers (metasurface-as-Jones-element).
# ===========================================================================
def _qwp_matrix(theta, xp=np):
    """Quarter-wave-plate 2x2 Jones matrix, fast axis at ``theta`` (radians), in
    the library's ``exp(-i w t)`` convention (matches ``apply_waveplate`` with
    retardance pi/2): ``R(theta) diag(1, -i) R(-theta)``.  Built in namespace
    ``xp`` (NumPy by default) so it can multiply a JAX Jones matrix and keep the
    autodiff graph; ``theta`` is a concrete host scalar (a device geometry knob,
    not a traced quantity)."""
    c, s = float(np.cos(theta)), float(np.sin(theta))
    e = -1j                                        # exp(-i * pi/2)
    return xp.asarray([[c * c + e * s * s, c * s * (1 - e)],
                       [c * s * (1 - e), s * s + e * c * c]], dtype=_C)



def reflective_outcoupling(jones_reflection, *, qwp_angle=None):
    """Cross-port (out-coupled) power fraction of the reflective-Jones device
    ``PBS -> QWP@45 -> grating -> QWP@45 -> PBS`` for a grating's zeroth-order
    ``jones_reflection`` (the 2x2 returned by :func:`rcwa_jones_1d` /
    :func:`rcwa_jones_1d_segments`).

    The in-coupling PBS launches an x-polarized wave; a quarter-wave plate (fast
    axis at ``qwp_angle``, default 45 deg) converts it, the grating reflects it
    (Jones ``J``), the wave passes back through the QWP, and the out-coupling PBS
    routes the orthogonal (y) component to the side port.  Returns
    ``|[Q J Q]_{yx}|**2``.  For a LOSSLESS grating whose retardance ``Gamma`` is
    aligned to TE/TM this equals ``cos**2(Gamma/2)`` (so tuning the grating's
    retardance -- e.g. via an LC fill -- modulates the side-port power).

    Backend-agnostic: a NumPy / CuPy Jones returns a Python float (bit-identical
    to the historical NumPy path), and a JAX Jones returns a traced scalar so
    ``jax.grad`` flows through (the device is just three 2x2 matrix products).
    """
    xp = array_namespace(jones_reflection)
    is_jax = backend_name(xp) == "jax"
    J = jones_reflection if is_jax else np.asarray(jones_reflection, dtype=_C)
    if J.shape != (2, 2):
        raise ValueError("reflective_outcoupling: expected a (2, 2) Jones "
                         f"matrix, got shape {J.shape}.")
    Q = _qwp_matrix(np.pi / 4 if qwp_angle is None else float(qwp_angle), xp=xp)
    M = Q @ J @ Q
    out = xp.abs(M[1, 0]) ** 2
    return out if is_jax else float(out)



def jones_retardance_diattenuation(jones_reflection):
    """Retardance, diattenuation, and fast-axis orientation of a 2x2 Jones matrix
    via the polar decomposition ``J = U H`` (``U`` unitary retarder, ``H``
    Hermitian diattenuator), from its SVD.

    Returns ``(retardance, diattenuation, fast_axis_rad)``: ``retardance`` is the
    phase difference between the retarder eigenpolarizations [radians, in
    ``(-pi, pi]``] (for a TE/TM-aligned grating, ``arg(r_TM) - arg(r_TE)``);
    ``diattenuation`` = ``(Tmax - Tmin)/(Tmax + Tmin)`` of the intensity
    eigentransmittances (0 = none, 1 = ideal polarizer); ``fast_axis_rad`` is the
    orientation of the maximum-transmittance input eigenpolarization."""
    J = np.asarray(jones_reflection, dtype=_C)
    if J.shape != (2, 2):
        raise ValueError("jones_retardance_diattenuation: expected a (2, 2) "
                         f"Jones matrix, got shape {J.shape}.")
    U, s, Vh = np.linalg.svd(J)
    tmax, tmin = float(s[0]) ** 2, float(s[1]) ** 2
    diatt = (tmax - tmin) / (tmax + tmin) if (tmax + tmin) > 0 else 0.0
    ev = np.linalg.eigvals(U @ Vh)                 # unitary retarder eigenphases
    retard = float(np.angle(ev[0] / ev[1]))
    v0 = np.conj(Vh[0])                            # max-T input eigenpolarization
    fast_axis = float(np.arctan2(np.real(v0[1]), np.real(v0[0])))
    return retard, diatt, fast_axis



def rcwa_efficiency_1d_jax(
    period,
    n_ridge,
    n_groove,
    n_substrate,
    n_superstrate,
    depth,
    duty_cycle,
    wavelength,
    *,
    angle=0.0,
    theta=None,
    polarization="te",
    n_orders=11,
    formulation="auto",
    n_samples=512,
):
    """JAX (differentiable) twin of :func:`rcwa_efficiency_1d`.

    .. deprecated:: 5.5.1
        Retained for backward compatibility.  This is now a thin wrapper that
        promotes its inputs to ``jax.numpy`` arrays and forwards to the
        unified :func:`rcwa_efficiency_1d`, which auto-dispatches to the
        differentiable JAX backend when given JAX inputs.  Prefer calling
        ``rcwa_efficiency_1d(...)`` with ``jax.numpy`` arguments directly.

    Returns ``(orders, R_eff, T_eff)`` as JAX arrays; the efficiencies are
    differentiable w.r.t. ``n_ridge``, ``n_groove``, ``depth`` and ``angle``
    (pass them as JAX tracers / floats), enabling ``jax.grad`` /
    ``jax.value_and_grad`` gradient-based metasurface inverse design.

    .. note::
        The non-Hermitian-eig custom VJP provides validated **first-order**
        gradients (matched to complex finite differences).  Second derivatives
        (``jax.hessian`` / forward-over-reverse) flow through the
        Lorentzian-broadened eigenvector term and are **not** validated; treat
        Hessian-based optimizers as unsupported on the eig path.

    Folding the former stand-alone JAX solver into the backend-dispatched
    core removed ~150 lines of duplicated physics (the source of the v5.5.0
    Wood-anomaly / validation drift) and switched the differentiable path
    from a soft-edge sampled profile to the SAME exact binary-grating Fourier
    coefficients the NumPy path uses -- so JAX now matches NumPy to eig
    precision rather than the old ~5e-3.  ``n_samples`` is accepted but
    ignored (the exact analytic coefficients need no sampling); ``duty_cycle``
    is a discrete threshold and is not differentiated.  Assumes no order sits
    exactly at grazing (no Wood-anomaly nudge on the differentiable path);
    choose ``wavelength`` / ``angle`` away from an exact Rayleigh anomaly.
    """
    warnings.warn(
        "rcwa_efficiency_1d_jax is deprecated since v5.5.1 and will be removed "
        "in v6.0.0 (the next major); call rcwa_efficiency_1d(...) with jax.numpy "
        "instead (it auto-dispatches to the differentiable JAX backend).",
        DeprecationWarning, stacklevel=2)
    from ...backend import JAX_AVAILABLE as _JAX_AVAILABLE
    if not _JAX_AVAILABLE:
        raise ImportError(
            "rcwa_efficiency_1d_jax requires the optional 'jax' extra; install "
            "with `pip install lumenairy[jax]` (or `pip install jax`).  Use the "
            "NumPy rcwa_efficiency_1d for non-differentiable evaluation.")
    import jax.numpy as jnp
    del n_samples  # accepted for back-compat; unused by the exact-coeff path
    # Promote the differentiable arguments to JAX arrays so the unified solver
    # dispatches to the JAX backend (gradients flow through these).
    return rcwa_efficiency_1d(
        period,
        jnp.asarray(n_ridge),
        jnp.asarray(n_groove),
        jnp.asarray(n_substrate),
        jnp.asarray(n_superstrate),
        jnp.asarray(depth),
        duty_cycle,
        jnp.asarray(wavelength),
        angle=jnp.asarray(angle),
        theta=theta,
        polarization=polarization,
        n_orders=n_orders,
        formulation=formulation,
    )


__all__ = [
    "_binary_grating_convolutions",
    "_asr_metric_profile",
    "_asr_convolutions",
    "_resolve_incidence",
    "rcwa_efficiency_1d",
    "rcwa_efficiency_vs_wavelength",
    "rcwa_jones_vs_wavelength",
    "rcwa_jones_vs_wavelength_segments",
    "_jones_1d_from_profiles",
    "rcwa_jones_1d",
    "rcwa_jones_1d_segments",
    "grating_segments",
    "binary_grating_segments",
    "interdigitated_grating_segments",
    "_qwp_matrix",
    "reflective_outcoupling",
    "jones_retardance_diattenuation",
    "rcwa_efficiency_1d_jax",
]
