"""
Rigorous Coupled-Wave Analysis (RCWA / Fourier Modal Method)
============================================================

A frequency-domain, rigorous Maxwell solver for laterally PERIODIC,
layered structures: dielectric / metallic gratings, sub-wavelength
metasurfaces, and birefringent (liquid-crystal) stacks.  RCWA fills the
gap between LumenAiry's scalar :mod:`~lumenairy.elements.thin_grating`
(thin-phase, no metal, no polarization coupling) and the laterally-uniform
isotropic TMM in :mod:`~lumenairy.elements.coatings`: it solves the full
vector Maxwell equations inside a structured periodic layer and returns
rigorous diffraction efficiencies AND the complex zeroth-order
(specular) Jones reflection matrix that bridges into the
:class:`~lumenairy.elements.polarization.JonesField` pipeline via
:meth:`RCWAResult.to_jones_field` / :meth:`RCWAResult.apply_reflection`
(the specular order only -- non-zero diffraction orders are not
reconstructed into a field).

This is a clean-room implementation derived from the published Fourier
Modal Method literature -- principally

* M. G. Moharam, E. B. Grann, D. A. Pommet, T. K. Gaylord, "Formulation
  for stable and efficient implementation of the rigorous coupled-wave
  analysis of binary gratings," JOSA A 12, 1068 (1995); and the
  enhanced transmittance-matrix companion, JOSA A 12, 1077 (1995).
* L. Li, "Use of Fourier series in the analysis of discontinuous periodic
  structures," JOSA A 13, 1870 (1996)  -- the inverse rule / correct
  factorization for TM and metals.
* L. Li, "Formulation and comparison of two recursive matrix algorithms
  for modeling layered diffraction gratings," JOSA A 13, 1024 (1996) --
  the S-matrix (Redheffer) recursion, and the 1-D anisotropic
  factorization.
* L. Li, "New formulation of the Fourier modal method for crossed
  surface-relief gratings," JOSA A 14, 2758 (1997).
* R. C. Rumpf, "Improved formulation of scattering matrices for
  semi-analytical methods that is consistent with convention," PIER B 35,
  241 (2011) -- the gap-medium scattering-matrix bookkeeping used here.

No GPL/closed RCWA source (nannos, S4, RETICOLO, inkstone, grcwa) is
copied; those are used only as independent numerical validation oracles.

Conventions
-----------
* Time convention ``exp(-i omega t)``; forward propagation ``exp(+i k z)``
  (the library standard -- see CONVENTIONS.md section 7).  Consequently a
  passive absorber has ``n = n + i kappa`` with ``kappa > 0``, hence
  ``Im(epsilon) = Im(n^2) > 0``, and the physical (decaying) z-branch is
  the one with ``Im(k_z) >= 0``.
* All wavelengths, periods, thicknesses are in METRES (vacuum wavelength).
* Angles are in RADIANS.  ``theta`` is the polar angle from the +z axis;
  ``phi`` is the azimuth (conical mounting) measured from +x.
* Diffraction-order indices are integers centred on 0.  Efficiencies are
  REAL power fractions; amplitudes are COMPLEX field coefficients.
* Energy: for a lossless stack ``sum(R) + sum(T) == 1`` to the harmonic
  truncation; with loss, ``sum(R) + sum(T) + A == 1`` where ``A`` is the
  absorptance.

Internally the solver is the full VECTORIAL scattering-matrix method
(Rumpf 2011 / Moharam 1995): the 2N-component (x, y tangential-field)
eigenproblem is assembled per layer, so planar TE/TM, conical TE/TM
coupling, and (in later layers) 2-D crossed gratings all flow through one
stable Redheffer star-product recursion -- the algorithm never forms an
exponentially-growing T-matrix.

Author: Andrew Traverso (LumenAiry RCWA module).
"""
from __future__ import annotations

import contextlib
import functools
import threading
from typing import List, Optional, Tuple

import numpy as np

from ..backend import (
    CUPY_AVAILABLE,
    JAX_AVAILABLE,
    array_namespace,
    backend_name,
    is_cupy_array,
    is_jax_array,
    to_numpy,
)

__all__ = [
    "rcwa_efficiency_1d",
    "rcwa_efficiency_vs_wavelength",
    "rcwa_efficiency_2d",
    "rcwa_efficiency_2d_shapes",
    "rcwa_extrapolate",
    "rcwa_convergence",
    "rcwa_jones_1d",
    "rcwa_jones_1d_segments",
    "grating_segments",
    "binary_grating_segments",
    "interdigitated_grating_segments",
    "reflective_outcoupling",
    "jones_retardance_diattenuation",
    "rcwa_jones_2d",
    "rcwa_jones_vs_wavelength",
    "rcwa_jones_vs_wavelength_segments",
    "rcwa_efficiency_1d_jax",
    "uniaxial_tensor",
    "RCWAStack",
    "RCWAResult",
    "set_blas_threads",
    "rcwa_blas_threads",
]

# Internal eigenproblem dtype -- always double complex regardless of the
# field-side default (RCWA conditioning is sensitive to single precision).
_C = np.complex128

# Safe ceiling on the retained-harmonic count N (the dense 2N x 2N eig is
# O((2N)^2) memory): N = 5000 -> a 10000 x 10000 complex128 matrix ~ 1.6 GB.
# Guards against an accidental huge n_orders OOM-hanging the solve; advanced
# users with more memory can raise it.
_MAX_HARMONICS = 5000

# Optional BLAS-thread cap for the NumPy/CuPy solve.  The dense non-Hermitian
# eigensolver (LAPACK zgeev, largely serial) plus the S-matrix BLAS3 thrash
# under thread oversubscription on many-core boxes, so capping the BLAS pool
# to a few threads is a MODEST, machine-dependent ~2-3x speedup at moderate N
# with ZERO numerics change.  Opt-in (None = leave the environment's threading
# untouched) because the optimum is configuration-dependent and a global
# thread change shouldn't be forced on the caller.  THREAD-LOCAL: concurrent
# solves with different caps must not leak each other's setting (the context
# manager's save/restore would otherwise race on a shared global).
_BLAS_STATE = threading.local()


def _get_blas_threads() -> Optional[int]:
    return getattr(_BLAS_STATE, "n", None)


def set_blas_threads(n: Optional[int]) -> None:
    """Cap the BLAS thread pool used by subsequent NumPy/CuPy RCWA solves on
    the CURRENT thread.

    On a thread-oversubscribed many-core box the dense ``zgeev`` eigensolver
    (largely serial) and the S-matrix BLAS3 contend, so a small cap (the
    measured optimum is ~2) gives a modest ~2-3x speedup at moderate truncation
    -- machine-dependent, with no change to the numbers.  Pass ``None`` to
    restore the default (untouched) threading.  Has no effect on the JAX path
    (XLA manages its own threads).  The setting is thread-local, so concurrent
    solves with different caps don't interfere.  For a scoped cap use
    :func:`rcwa_blas_threads`.
    """
    _BLAS_STATE.n = None if n is None else max(1, int(n))


@contextlib.contextmanager
def rcwa_blas_threads(n: Optional[int]):
    """Context manager that caps the BLAS pool for RCWA solves within the
    ``with`` block on the current thread (see :func:`set_blas_threads`);
    restores the prior setting on exit."""
    prev = _get_blas_threads()
    set_blas_threads(n)
    try:
        yield
    finally:
        _BLAS_STATE.n = prev


def _blas_limit():
    """Apply this thread's BLAS cap if one is set, else a zero-overhead no-op
    context (so the default path is untouched)."""
    n = _get_blas_threads()
    if n is None:
        return contextlib.nullcontext()
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - threadpoolctl ships with numpy
        return contextlib.nullcontext()
    return threadpool_limits(limits=n, user_api="blas")


def _with_blas_limit(fn):
    """Decorator: run an RCWA entry point under the optional BLAS-thread cap."""
    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        with _blas_limit():
            return fn(*args, **kwargs)
    return _wrapped


def _stabilize_bumps(n_orders, reach=12, floor=2):
    """``n_orders`` offsets to try when ``stabilize=True`` self-heals a
    measure-zero energy blow-up, ordered nearest-first in BOTH directions.

    The clean truncations bracket the bad ones, and on some LAPACK builds the
    nearest clean truncation sits BELOW the request -- low orders are
    generically well-conditioned (the near-singular layer<->region mode-match
    needs enough orders to appear), so the downward search is what makes the
    heal platform-robust (an upward-only schedule fails when the LAPACK-
    dependent blow-up pattern leaves no clean truncation above the request).
    Higher order is tried first at equal distance (more accurate); the floor
    keeps the retained count at ``>= floor`` per side.
    """
    n = int(n_orders)
    bumps = [0]
    for d in range(1, int(reach) + 1):
        bumps.append(d)                   # up first (more accurate when clean)
        if n - d >= floor:
            bumps.append(-d)
    return bumps


def _eig_for(xp):
    """Backend-appropriate general (non-Hermitian) eigendecomposition for the
    layer / Omega^2 solve.  NumPy and CuPy use their native ``linalg.eig``;
    JAX uses the gauge-stable custom-VJP eig (:func:`_jax_eig_stable`) so the
    whole RCWA solve stays differentiable.  Returns a callable with the
    ``eig(A) -> (eigvals, eigvecs)`` signature."""
    if JAX_AVAILABLE and backend_name(xp) == "jax":
        return _jax_eig_stable()
    return xp.linalg.eig


def _block(xp, rows):
    """Assemble a 2-level block matrix (like ``numpy.block`` on a nested list
    of 2-D blocks) using only ``concatenate`` -- portable across NumPy / CuPy
    / JAX (some CuPy builds lack ``cupy.block``)."""
    return xp.concatenate([xp.concatenate(row, axis=1) for row in rows],
                          axis=0)


def _rcwa_xp(fn_name, use_gpu, *arrays):
    """Resolve the array namespace for an RCWA entry point.

    JAX if any input is a JAX array (the differentiable path); else CuPy if
    ``use_gpu`` is set or any input is already a CuPy array (the GPU path);
    else NumPy.  Raises if ``use_gpu`` is requested without CuPy installed.
    The numerically delicate eig/solve still run in double precision (``_C``).
    """
    if any(is_jax_array(a) for a in arrays):
        # The JAX short-circuit must not silently swallow a conflicting GPU
        # request: a JAX input combined with use_gpu / a CuPy input is an
        # ambiguous mixed-backend call, so reject it explicitly.
        if use_gpu or any(is_cupy_array(a) for a in arrays):
            raise ValueError(
                f"{fn_name}: a JAX input cannot be combined with use_gpu or a "
                f"CuPy input -- pass a single backend (all-JAX for the "
                f"differentiable path, or use_gpu/CuPy for the GPU path).")
        import jax.numpy as jnp
        return jnp
    if use_gpu:
        if not CUPY_AVAILABLE:
            raise RuntimeError(
                f"{fn_name}: use_gpu=True but CuPy is not installed.  Install "
                f"the GPU stack (`pip install lumenairy[cuda]`) or call with "
                f"use_gpu=False for the NumPy path.")
        import cupy as cp
        return cp
    return array_namespace(*arrays)


def _is_traced(v):
    """True if ``v`` is an abstract JAX tracer (no concrete numeric value), so
    geometry validation / Wood-anomaly nudges that need a concrete number are
    skipped on the differentiable path.  Uses ``complex`` (not ``float``) so a
    concrete complex index doesn't emit a discard-imaginary ComplexWarning."""
    try:
        complex(v)
        return False
    except Exception:
        return True


def _concrete(**kw):
    """Subset of geometry kwargs whose values are concrete (float-able);
    used to validate only the non-traced arguments on the JAX path."""
    return {k: v for k, v in kw.items() if not _is_traced(v)}


class _EnergyError(ValueError):
    """Raised by :func:`_check_energy` when a passive solve returns
    non-physical ``sum(R)+sum(T) >> 1``.  A subclass of ``ValueError`` (so
    existing ``except ValueError`` handlers are unaffected) that the
    ``stabilize=`` retry path can catch specifically."""


def _check_energy(fn_name, R, T):
    """Raise if the total efficiency exceeds the incident power by a large
    margin.  A PASSIVE structure cannot reflect + transmit more than what
    comes in, so ``sum(R) + sum(T) >> 1`` per incident polarization signals a
    numerical instability -- the layer<->region mode-match matrix in
    :func:`_interface_smatrix` goes near-singular (cond up to ~1e13) at an
    erratic, measure-zero (period, n_orders) coincidence for high contrast,
    and its explicit inverse amplifies the noise floor into the Redheffer
    star denominators (the v5.6 root-cause analysis).  The true answer there
    is ~1.0; bumping ``n_orders`` by a few shifts the quasi-resonance away
    (see the ``stabilize=`` retry).  This was otherwise SILENTLY returning a
    non-physical answer (R+T up to 1e30+).

    Skipped on the JAX path (the sums are traced).  Lossy media give R+T < 1
    (never triggered); the tolerance leaves normal Wood-nudge residue alone.
    """
    tot = float(np.real(np.sum(np.asarray(R))) + np.real(np.sum(np.asarray(T))))
    n_states = int(R.shape[0]) if getattr(R, "ndim", 1) == 2 else 1
    if tot > n_states * 1.05:
        raise _EnergyError(
            f"{fn_name}: energy non-conservation detected (sum R+T = "
            f"{tot:.3e} exceeds {n_states}); the solve is numerically unstable "
            f"at this geometry (a near-degenerate layer<->region mode-match at "
            f"a measure-zero period / n_orders coincidence, common at very "
            f"large period / low index contrast).  Pass stabilize=True to "
            f"auto-retry at a slightly higher n_orders, or reduce n_orders, "
            f"adjust the period, or increase the index contrast.")


def _warn_if_jax_f32(fn_name):
    """Warn if JAX x64 is disabled.  RCWA's eigenproblem is ill-conditioned in
    single precision, and JAX silently truncates the requested complex128 to
    complex64 unless ``jax_enable_x64`` is set -- giving quietly inaccurate
    results.  Emitted once per call site (Python's default warning filter)."""
    import jax
    try:
        enabled = bool(jax.config.read("jax_enable_x64"))
    except Exception:
        enabled = bool(getattr(jax.config, "jax_enable_x64", False))
    if not enabled:
        import warnings
        warnings.warn(
            f"{fn_name}: JAX x64 is disabled, so the RCWA solve runs in "
            f"complex64 -- its eigenproblem is ill-conditioned in single "
            f"precision.  Enable double precision before the call with "
            f"jax.config.update('jax_enable_x64', True).",
            stacklevel=3)


def _normalize_pol(fn_name, polarization):
    """Normalise a polarization string, accepting ``'s'`` / ``'p'`` as
    aliases for ``'te'`` / ``'tm'`` (the ``coatings`` module speaks s/p while
    RCWA / ``thin_grating`` speak te/tm -- CONVENTIONS Section 7 bridge)."""
    pol = {"s": "te", "p": "tm"}.get(str(polarization).lower(),
                                     str(polarization).lower())
    if pol not in ("te", "tm"):
        raise ValueError(
            f"{fn_name}: polarization must be 'te'/'tm' (or the 's'/'p' "
            f"aliases), got {polarization!r}.")
    return pol


def _normalize_2d_formulation(fn_name, formulation):
    """Normalise the 2-D Fourier-factorization selector.  ``'laurent'`` is the
    direct rule everywhere (default, bit-for-bit backward compatible);
    ``'li'`` (alias ``'fff'``) is the dual-Laurent z-rule: the ``E_z``
    elimination uses ``[[1/eps]]`` for fast TM / metal convergence;
    ``'fff_nv'`` is the normal-vector fast Fourier factorization (Schuster 2007)
    -- the in-plane inverse rule projected on the local wall-normal field, the
    correct factorization for 2-D metallic gratings."""
    f = str(formulation).lower()
    if f == "fff":
        f = "li"
    if f not in ("laurent", "li", "fff_nv"):
        raise ValueError(
            f"{fn_name}: formulation must be 'laurent', 'li', 'fff' "
            f"(alias of 'li') or 'fff_nv', got {formulation!r}.")
    return f


# ===========================================================================
# Convention-aware square root (branch selection)
# ===========================================================================

def _sqrt_forward(x: np.ndarray) -> np.ndarray:
    """Square root on the branch that makes ``exp(+i k z)`` the forward
    (outgoing / evanescent-decaying) PLANE WAVE for the library's
    ``exp(-i omega t)`` convention: ``Im(result) >= 0``.

    Used for the homogeneous-region longitudinal wavevector ``k_z`` (where
    the field varies as ``exp(+i k_z z)``).  For a real positive argument
    this is the ordinary positive root (a propagating order); for a real
    negative argument it is ``+i|.|^{1/2}`` (an order that decays as
    ``z -> +inf``).
    """
    xp = array_namespace(x)
    x = xp.asarray(x).astype(_C)
    r = xp.sqrt(x)
    # numpy's principal branch already yields Im >= 0 except on the cut;
    # force the decaying root for any residual negative-imaginary roundoff.
    bad = (r.imag < 0) | ((xp.abs(r.imag) <= 1e-300) & (r.real < 0))
    return xp.where(bad, -r, r)


def _inv_lam(lam: np.ndarray) -> np.ndarray:
    """``1/lam`` with a floor on ``|lam|`` so a grazing mode (``kz -> 0`` so
    the modal eigenvalue ``lam -> 0``) does not produce ``inf``/``NaN`` in
    ``V = Q W diag(1/lam)``.  A ``kz = 0`` mode carries no z-directed power,
    so this regularisation never affects a physical diffraction efficiency;
    it only keeps the eigenvector matrix finite at an exact Wood anomaly.
    """
    xp = array_namespace(lam)
    safe = xp.where(xp.abs(lam) < 1e-12, 1e-12, lam)
    return 1.0 / safe


def _sqrt_decay(x: np.ndarray) -> np.ndarray:
    """Square root on the ``Re(result) >= 0`` (principal) branch, used for
    the LAYER modal eigenvalue ``lam`` that drives the propagator
    ``X = exp(-lam k0 L)``.

    Choosing ``Re(lam) >= 0`` guarantees ``|X| <= 1`` for forward thickness
    ``L > 0`` -- the unconditional-stability property of the S-matrix
    method.  This is NOT the ``Im >= 0`` branch: for a strongly evanescent
    mode the eigenvalue ``lam^2`` is a large positive real that ``eig``
    returns with a tiny spurious imaginary part, and an ``Im >= 0`` rule
    would flip ``lam`` to a NEGATIVE real, turning the decaying propagator
    into ``exp(+|gamma| k0 L)`` -- a catastrophic high-order blow-up.  The
    principal branch is immune (it never flips the sign of a positive-real
    root).  For propagating modes (``lam^2`` negative real) both branches
    agree on ``+i|kz|``, so physics is unchanged.
    """
    xp = array_namespace(x)
    x = xp.asarray(x).astype(_C)
    r = xp.sqrt(x)  # principal branch: Re(r) >= 0 by construction
    # On the cut (pure-imaginary r, i.e. lam^2 real negative) pin Im >= 0
    # so propagating modes use the outgoing root deterministically.
    on_cut = r.real == 0
    return xp.where(on_cut & (r.imag < 0), -r, r)


# ===========================================================================
# Robustness guards: non-propagating incidence + generalized Wood-anomaly
# ===========================================================================

def _require_propagating_incidence(fn_name, eps_sup, kt0_sq):
    """Raise if the incidence half-space is non-propagating, i.e. the
    incident plane wave is evanescent in the superstrate
    (``Re(eps_superstrate) <= kx0^2 + ky0^2``).  Without this guard the
    efficiency normalisation divides by ``kz_inc ~ 0`` and silently returns
    negative / NaN 'efficiencies'.  For a real lossless superstrate this can
    only trip at exactly grazing incidence (theta -> 90 deg); it fires for
    evanescent / metallic / gain incidence media."""
    if float(np.real(eps_sup)) - float(np.real(kt0_sq)) <= 1e-12:
        raise ValueError(
            f"{fn_name}: the incidence half-space is non-propagating "
            f"(Re(eps_superstrate) = {float(np.real(eps_sup)):.4g} <= "
            f"kx0^2+ky0^2 = {float(np.real(kt0_sq)):.4g}); the incident plane "
            f"wave is evanescent in the superstrate.  Use a propagating "
            f"incidence medium (real n_superstrate > n_inc*sin(theta)).")


def _grazing_safe_wavelength(wavelength, kx0, ky0, m_orders, n_orders,
                             period_x, period_y, eps_reals, max_iter=8):
    """Wavelength nudged off any EXACT Wood anomaly -- a diffracted order
    grazing (``kz = 0``) in ANY medium whose real permittivity is in
    ``eps_reals`` (the super/substrate AND the layer's constituent indices;
    omitting the layer is what let a grazing LAYER mode crash the interface
    S-matrix).  A tiny relative REAL nudge is applied only when an exact
    grazing is detected, so lossless energy stays exact, ``+/-m`` symmetry is
    preserved, and the grazing order (which carries no z-power) limits
    continuously."""
    eps_reals = [float(np.real(e)) for e in eps_reals]

    def closest(wl):
        kxg = kx0 + m_orders * (wl / period_x)
        kyg = ky0 + n_orders * (wl / period_y)
        kt2 = kxg ** 2 + kyg ** 2
        return min(float(np.min(np.abs(e - kt2))) for e in eps_reals)

    wl = wavelength
    for _ in range(max_iter):
        if closest(wl) > 1e-9:
            return wl
        wl = wl * (1.0 + 1e-7)
    return wl


def _validate_geometry(fn_name, *, period=None, period_y=None, depth=None,
                       wavelength=None, n_orders=None, n_orders_y=None):
    """Shared geometric input validation for every RCWA entry point.

    Raises :class:`ValueError` with a ``fn_name:`` prefix (CONVENTIONS Section
    2) on any non-physical geometry.  Replaces the silent-wrong-answer /
    cryptic-LinAlgError failure modes the v5.5.0 audit found: ``depth < 0``
    silently returned a wrong answer, ``period = 0`` raised ``ZeroDivision``,
    and ``n_orders < 1`` raised a bare ``zero-size array`` error."""
    def _pos(name, val):
        if val is None:
            return
        try:
            v = float(val)
        except (TypeError, ValueError):
            raise ValueError(
                f"{fn_name}: {name} must be a positive real number, got "
                f"{val!r}.") from None
        if not np.isfinite(v) or v <= 0.0:
            raise ValueError(f"{fn_name}: {name} must be > 0, got {v}.")

    _pos("period", period)
    _pos("period_y", period_y)
    _pos("depth", depth)
    _pos("wavelength", wavelength)
    for name, val in (("n_orders", n_orders), ("n_orders_y", n_orders_y)):
        if val is None:
            continue
        try:
            iv = int(val)
        except (TypeError, ValueError):
            raise ValueError(
                f"{fn_name}: {name} must be an integer >= 1, got "
                f"{val!r}.") from None
        if iv != val or iv < 1:
            raise ValueError(
                f"{fn_name}: {name} must be an integer >= 1, got {val!r}.")
    # Upper bound: the dense 2N x 2N non-Hermitian eigenproblem is O((2N)^3)
    # time and O((2N)^2) memory, where N is the retained-harmonic count (1-D:
    # 2*n_orders+1; 2-D: (2*nox+1)(2*noy+1)).  Without a ceiling a fat-finger
    # n_orders (e.g. 1e9) passes validation and OOM-hangs the solve, so cap N.
    if n_orders is not None:
        nh = (2 * int(n_orders) + 1)
        if n_orders_y is not None:
            nh *= (2 * int(n_orders_y) + 1)
        if nh > _MAX_HARMONICS:
            raise ValueError(
                f"{fn_name}: the retained-harmonic count {nh} (from "
                f"n_orders={int(n_orders)}"
                + (f", n_orders_y={int(n_orders_y)}"
                   if n_orders_y is not None else "")
                + f") exceeds the safe ceiling {_MAX_HARMONICS}; the dense "
                f"2N x 2N eigenproblem would risk OOM.  Reduce n_orders "
                f"(raise lumenairy.elements.rcwa._MAX_HARMONICS only if you "
                f"have the memory).")


def _validate_cell_sampling(fn_name, cell, n_orders_x, n_orders_y):
    """Enforce the 2-D Fourier-aliasing bound.  The Laurent convolution table
    spans difference orders ``[-2N..2N]`` per axis, so a PATTERNED cell must
    satisfy ``S >= 4*n_orders + 1`` along each axis or the ``% S`` wrap aliases
    high-frequency permittivity coefficients into the low orders -- a silent
    wrong answer.  Raises with a ``fn_name:`` prefix when undersampled.

    A spatially UNIFORM cell (every pixel identical -- a homogeneous layer
    passed as an array) has only a DC coefficient, but that DC term still
    aliases onto off-diagonal entries (corrupting the otherwise ``const*I``
    convolution into a singular matrix) once ``S <= 2*n_orders``; it is exact
    only for ``S >= 2*n_orders + 1``, the relaxed floor used here."""
    Sx, Sy = int(cell.shape[0]), int(cell.shape[1])
    Mx, My = int(n_orders_x), int(n_orders_y)
    # Uniformity is a VALUE check -- only attempt it on a concrete (non-traced)
    # array; a traced JAX cell can't be inspected, so fall back to the strict
    # patterned bound (shape is always available).  uniform <=> every component
    # is constant across the two SPATIAL axes (per-component spread over (0, 1)
    # is zero -- NOT the spread over the whole array, which a varying tensor's
    # distinct components would trip).
    uniform = False
    if not is_jax_array(cell):
        xpc = array_namespace(cell)
        arr = xpc.asarray(cell)
        spatial = arr.reshape(Sx, Sy, -1)
        spread = xpc.ptp(spatial.real, axis=(0, 1))
        if xpc.iscomplexobj(spatial):
            spread = spread + xpc.ptp(spatial.imag, axis=(0, 1))
        uniform = bool(float(xpc.max(spread)) == 0.0)
    fac = 2 if uniform else 4
    need_x, need_y = fac * Mx + 1, fac * My + 1
    if Sx < need_x or Sy < need_y:
        bound = ("2*n_orders + 1 (uniform cell)" if uniform
                 else "4*n_orders + 1")
        raise ValueError(
            f"{fn_name}: the unit-cell sampling {(Sx, Sy)} is too coarse for "
            f"n_orders_x={Mx}, n_orders_y={My}; the Fourier convolution would "
            f"alias.  Need at least ({need_x}, {need_y}) samples "
            f"(>= {bound} per axis).")


def _validate_shapes(fn_name, shapes, period_x, period_y):
    """Validate the analytic-shape list against three silent-wrong-answer
    traps, each rejected up front with a ``fn_name:`` prefix:

    * unknown kind / zero or negative size -- a non-positive dimension
      vanishes or sign-flips the shape's contribution to the permittivity
      spectrum;
    * area fraction > 1 -- a shape whose area exceeds the cell drives the
      ``G = 0`` (DC) Fourier coefficient (the cell-average permittivity)
      past the shape's own ``eps``; an average must lie between
      ``eps_background`` and ``eps_shape``, so this is physically impossible;
    * bounding extent > a period -- the shape wraps across the cell and
      self-overlaps, so even an area fraction <= 1 is mis-modelled;
    * CUMULATIVE area fraction > 1 -- the analytic factorization ADDS each
      shape's form factor over the background, so the painted area must total
      <= one cell (disjoint shapes).  Two disjoint disks at fraction 0.6 each
      (total 1.2) drive the DC permittivity past the shapes' eps just as a
      single oversized shape would; the per-shape check alone misses it.

    The solver runs without complaint on any of these (it even conserves
    energy R + T = 1) while modelling a non-physical structure, hence the
    up-front guard.  ``period_x`` / ``period_y`` are the unit-cell lattice
    periods [m]."""
    area_cell = float(period_x) * float(period_y)
    total_fraction = 0.0
    for i, sh in enumerate(shapes):
        kind = sh.get("shape")
        if kind == "rectangle":
            wx, wy = (float(v) for v in sh["size"])
            dims, ext_x, ext_y, area_shape = (wx, wy), wx, wy, wx * wy
        elif kind == "disk":
            r = float(sh["radius"])
            dims, ext_x, ext_y, area_shape = (r,), 2.0 * r, 2.0 * r, np.pi * r * r
        elif kind == "ellipse":
            ax, ay = (float(v) for v in sh["semi_axes"])
            dims = (ax, ay)
            ext_x, ext_y, area_shape = 2.0 * ax, 2.0 * ay, np.pi * ax * ay
        else:
            raise ValueError(
                f"{fn_name}: shapes[{i}] has unknown shape {kind!r} (expected "
                f"'rectangle', 'disk' or 'ellipse').")
        for d in dims:
            if not (np.isfinite(d) and d > 0.0):
                raise ValueError(
                    f"{fn_name}: shapes[{i}] ({kind}) has a non-positive "
                    f"dimension {d!r}; all sizes / radii / semi-axes must "
                    f"be > 0 metres.")
        # An exactly-tiling rectangle (fraction == 1) and an inscribed
        # disk/ellipse (extent == period) are legitimate, so compare with a
        # tiny relative slack -- far below any real overshoot.
        fraction = area_shape / area_cell
        if fraction > 1.0 + 1e-9:
            raise ValueError(
                f"{fn_name}: shapes[{i}] ({kind}) has area fraction "
                f"{fraction:.4g} > 1 of the {period_x:.4g} x {period_y:.4g} m "
                f"unit cell; its area exceeds the cell, so the average (G=0) "
                f"permittivity would overshoot the shape's own eps -- a "
                f"non-physical structure.  Shrink the shape or enlarge the "
                f"period.")
        if ext_x > period_x * (1.0 + 1e-9) or ext_y > period_y * (1.0 + 1e-9):
            raise ValueError(
                f"{fn_name}: shapes[{i}] ({kind}) bounding extent "
                f"{ext_x:.4g} x {ext_y:.4g} m exceeds the {period_x:.4g} x "
                f"{period_y:.4g} m unit cell; the shape wraps across the "
                f"period and self-overlaps.  Shrink the shape or enlarge the "
                f"period.")
        total_fraction += fraction
    if total_fraction > 1.0 + 1e-9:
        raise ValueError(
            f"{fn_name}: the shapes' CUMULATIVE area fraction "
            f"{total_fraction:.4g} exceeds 1 of the {period_x:.4g} x "
            f"{period_y:.4g} m unit cell; the analytic factorization adds each "
            f"shape's form factor, so the painted area must total <= one cell "
            f"(disjoint shapes).  An overlapping / over-painted layer drives "
            f"the average (G=0) permittivity past the shapes' eps -- a "
            f"non-physical structure.  Reduce the shapes or enlarge the "
            f"period.")


# ===========================================================================
# Fourier factorization -- convolution matrices
# ===========================================================================

def _fourier_coeffs_1d(profile, n_coeffs: int):
    """Centred Fourier coefficients ``c_k`` (``k = -(n_coeffs-1) ..
    (n_coeffs-1)``, length ``2*n_coeffs-1``) of a uniformly-sampled,
    one-period profile, with ``c_k = <f(x) exp(-i k G x)>``.

    Backend-agnostic and JAX-differentiable: vectorised fancy-indexing (no
    item assignment) so it runs unchanged on NumPy / CuPy / JAX arrays.
    """
    xp = array_namespace(profile)
    profile = xp.asarray(profile).astype(_C)
    Nx = profile.shape[0]
    full = xp.fft.fft(profile) / Nx  # full[k] holds c_k (periodic in k)
    ks = xp.arange(-(n_coeffs - 1), n_coeffs)
    return full[ks % Nx]


def _toeplitz_1d(coeffs, n_orders: int):
    """``(N, N)`` Toeplitz convolution matrix from centred Fourier
    coefficients, ``N = 2*n_orders + 1``; entry ``[m, n] = c_{m-n}``.

    Backend-agnostic / JAX-differentiable (vectorised gather)."""
    xp = array_namespace(coeffs)
    N = 2 * n_orders + 1
    centre = (coeffs.shape[0] - 1) // 2  # index of c_0
    idx = xp.arange(N)
    tidx = centre + (idx[:, None] - idx[None, :])  # (N, N) of (m - n)
    return coeffs[tidx]


def _binary_grating_convolutions(n_ridge, n_groove, duty_cycle, n_orders,
                                 n_samples=4096):
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
    """
    xp = array_namespace(n_ridge, n_groove)
    x = (xp.arange(n_samples) + 0.5) / n_samples
    eps_r = xp.asarray(n_ridge).astype(_C) ** 2
    eps_g = xp.asarray(n_groove).astype(_C) ** 2
    eps = xp.where(x < duty_cycle, eps_r, eps_g).astype(_C)
    # The Toeplitz matrix needs coefficients c_k for k = -(N-1)..(N-1) with
    # N = 2*n_orders+1, i.e. n_coeffs = N.
    n_coeffs = 2 * n_orders + 1
    eps_coeffs = _fourier_coeffs_1d(eps, n_coeffs)
    inv_eps_coeffs = _fourier_coeffs_1d(1.0 / eps, n_coeffs)
    EPS = _toeplitz_1d(eps_coeffs, n_orders)               # Laurent rule
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
# Layer eigen-solve (vectorial 2N system, Rumpf/Moharam)
# ===========================================================================

def _layer_Q_matrix(Kx, Ky, EPS, EPS_xx):
    """The ``Q`` block (``dE/dz' = Q H``) of the layer ODE system.

    ``EPS`` is the Laurent ``[[eps]]`` (used where E is tangential to the
    grating walls -- the ``E_y`` response); ``EPS_xx`` is the convolution
    for the wall-NORMAL field ``E_x``, which is ``[[eps]]`` for the Laurent
    rule but the Li inverse-rule ``[[1/eps]]^{-1}`` for fast TM / metal
    convergence.  Shared by the structured-layer eig solve and the analytic
    uniform-layer path so the magnetic eigenvectors ``V = Q W diag(1/lam)``
    use one convention everywhere (essential for evanescent-order interface
    consistency).
    """
    xp = array_namespace(Kx, Ky, EPS, EPS_xx)
    return _block(xp, [
        [Kx @ Ky,           EPS - Kx @ Kx],
        [Ky @ Ky - EPS_xx,  -Ky @ Kx],
    ])


# ===========================================================================
# Even-parity-sector RCWA solve (opt-in symmetry speed-up)
# ===========================================================================
#
# When the cell is centro-symmetric AND incidence is normal, EVERY operator in
# the solve (layer system matrix ``M = P @ Q``, region modes, interface and
# Redheffer S-matrices) commutes with the order-flip ``G = blockdiag(J, J)``
# (``J`` maps order ``(m, n) -> (-m, -n)``).  The (0, 0) incident order is the
# fixed point of ``J``, so the source is PURELY EVEN; because no operator
# couples the two parities, the odd half of the field is never excited and can
# be discarded entirely.  The whole recursion therefore runs in the
# ``(N + 1)``-dimensional EVEN subspace instead of the full ``2N`` -- every
# ``O(N^3)`` step (the eig, the interface ``inv``/``solve``, the Redheffer
# star) shrinks ~8x.  This realizes the symmetry speed-up end-to-end; folding
# only the layer eig (the obvious move) is Amdahl-capped because the interface
# and Redheffer algebra, also ``O(N^3)``, would stay full-size.
#
# The even block of a ``G``-commuting operator is ``B^H A B`` for the ``(N+1)``
# orthonormal even basis (a fixed-point column ``e_f`` plus pair columns
# ``(e_i + e_j)/sqrt(2)``).  It is assembled by an ``O(N^2)`` index FOLD --
# never a dense ``B^H A B`` matmul (itself ``O(N^3)``, which would erase the
# saving): each column has <= 2 nonzeros, so every entry is a 4-term index
# combination.  The path is GATED on the exact precondition (normal incidence +
# ``J EPS J = EPS``); if it fails (oblique -> the order set is not flip-closed;
# a non-centro-symmetric cell; or a uniform layer, whose degenerate eig wants
# the analytic path) it returns ``None`` and the caller runs the full solve, so
# the result is always correct.  Opt-in (``symmetry=True``) because the
# even-basis recursion changes the result at the ~1e-12 level -- physically
# identical, but not bit-for-bit with the default path.


def _order_flip_perm(Kx, Ky):
    """Permutation ``p`` with ``p[i]`` = index of the order whose transverse
    wavevector is ``(-kx_i, -ky_i)``, or ``None`` if the order set is not
    closed under that flip.

    Derived purely from the ``K`` diagonals, so it is self-contained (no order
    table needed) and serves any truncation and both the 1-D and 2-D cores.
    The set is closed only at NORMAL incidence: an oblique ``kx0`` offset makes
    ``-kx_i`` land off-lattice, so ``None`` is returned and symmetry is skipped.
    """
    kx = np.real(np.diagonal(to_numpy(Kx))).astype(float)
    ky = np.real(np.diagonal(to_numpy(Ky))).astype(float)
    n = kx.shape[0]
    scale = float(max(np.max(np.abs(kx)), np.max(np.abs(ky)), 1.0))
    tol = 1e-9 * scale
    # Bucket each order by a rounded (kx, ky) key, then look up its flip.
    q = 1.0 / tol
    key = lambda a, b: (int(round(a * q)), int(round(b * q)))  # noqa: E731
    lut = {}
    for i in range(n):
        lut[key(kx[i], ky[i])] = i
    perm = np.empty(n, dtype=np.intp)
    for i in range(n):
        j = lut.get(key(-kx[i], -ky[i]))
        if j is None:
            return None
        perm[i] = j
    return perm


def _flip_invariant(A, flip):
    """True if ``J A J == A`` (``J`` the order flip), i.e. ``A`` is the
    convolution of a real, origin-even permittivity -- the precondition for the
    even-sector solve.  ``O(N^2)`` host-cheap check."""
    Ah = to_numpy(A)
    resid = np.max(np.abs(Ah[np.ix_(flip, flip)] - Ah))
    scale = float(max(np.max(np.abs(np.diagonal(Ah))), 1.0))
    return resid <= 1e-10 * scale


def _recentering_phase(EPS, orders, xp):
    """Diagonal gauge ``d`` that moves a cell's symmetry centre to the FFT
    origin, or ``None`` if it cannot be inferred.

    A cell even about real position ``(x0, y0)`` -- the usual ``centred''
    feature, NOT aligned to sample 0 -- has Fourier coefficients
    ``c_{-k} = e^{i phi . k} c_k`` (a linear phase ramp).  Conjugating ``EPS``
    by ``D = diag(e^{-i phi . order / 2})`` cancels the ramp so ``D^{-1} EPS D``
    is flip-invariant (``c_{-k} = c_k``) and the even-sector machinery applies.
    ``D`` is a per-order phase, hence a gauge: it leaves every per-order
    efficiency ``|r_i|^2`` unchanged, so no back-transform is needed.  ``phi``
    is read off the first harmonics ``c_{(+/-1,0)}`` / ``c_{(0,+/-1)}``; a wrong
    guess is caught by the caller's flip-invariance check (-> full-solve
    fallback), so this is safe even when the read-off is degenerate.
    """
    om = {(int(a), int(b)): i for i, (a, b) in enumerate(np.asarray(orders))}
    i00 = om.get((0, 0))
    if i00 is None:
        return None
    Eh = to_numpy(EPS)
    ref = abs(Eh[i00, i00]) + 1e-300

    def _phi(plus, minus):
        ip, im = om.get(plus), om.get(minus)
        if ip is None or im is None:
            return 0.0
        cp = Eh[ip, i00]                       # c_{+e}
        if abs(cp) < 1e-10 * ref:              # no first-harmonic content
            return 0.0
        return float(np.angle(Eh[im, i00] * np.conj(cp)))   # angle(c_{-e}/c_{+e})

    phix = _phi((1, 0), (-1, 0))
    phiy = _phi((0, 1), (0, -1))
    m = np.asarray(orders)[:, 0].astype(float)
    n = np.asarray(orders)[:, 1].astype(float)
    return xp.asarray(np.exp(-0.5j * (phix * m + phiy * n)).astype(_C))


def _even_basis_desc(flip):
    """Descriptor of the orthonormal EVEN basis of ``G = blockdiag(J, J)`` over
    the ``2N`` field space: a fixed-point column ``e_f`` for each self-paired
    order (the two ``(0,0)`` components) and a column ``(e_i + e_j)/sqrt(2)``
    for each flip-pair.  Returns ``(i0, i1, c0, c1, n2)`` -- two support indices
    and two coefficients per even column (the fixed-point column repeats its
    index with ``c1 = 0``), plus the field-space size ``n2 = 2N``."""
    n = flip.shape[0]
    n2 = 2 * n
    flip2 = np.concatenate([flip, flip + n])          # G = blockdiag(J, J)
    ar = np.arange(n2)
    fixed = np.flatnonzero(flip2 == ar)               # self-paired (e.g. (0,0))
    pair_i = np.flatnonzero(flip2 > ar)               # canonical reps (i < j)
    pair_j = flip2[pair_i]
    inv2 = 1.0 / np.sqrt(2.0)
    i0 = np.concatenate([fixed, pair_i])
    i1 = np.concatenate([fixed, pair_j])
    c0 = np.concatenate([np.ones(fixed.size), np.full(pair_i.size, inv2)])
    c1 = np.concatenate([np.zeros(fixed.size), np.full(pair_i.size, inv2)])
    return i0, i1, c0, c1, n2


def _even_fold(A, desc, xp):
    """Even block ``B^H A B`` (``(N+1) x (N+1)``) of a ``G``-commuting ``2N``
    operator, as a 4-term index combination (no dense matmul)."""
    i0, i1, c0, c1, _ = desc
    c0x = xp.asarray(c0.astype(_C))
    c1x = xp.asarray(c1.astype(_C))
    i0n, i1n = np.asarray(i0), np.asarray(i1)
    A00 = A[i0n[:, None], i0n[None, :]]
    A01 = A[i0n[:, None], i1n[None, :]]
    A10 = A[i1n[:, None], i0n[None, :]]
    A11 = A[i1n[:, None], i1n[None, :]]
    return (c0x[:, None] * c0x[None, :] * A00
            + c0x[:, None] * c1x[None, :] * A01
            + c1x[:, None] * c0x[None, :] * A10
            + c1x[:, None] * c1x[None, :] * A11)


def _even_project(v, desc, xp):
    """``B^H v`` -- project a ``2N`` field vector onto the even basis
    (``(N+1)`` coords).  Lossless for a purely even ``v`` (e.g. the source)."""
    i0, i1, c0, c1, _ = desc
    c0x = xp.asarray(c0.astype(_C))
    c1x = xp.asarray(c1.astype(_C))
    return c0x * v[xp.asarray(i0)] + c1x * v[xp.asarray(i1)]


def _even_unfold(ve, desc, xp):
    """``B ve`` -- expand an even-basis ``(N+1)`` vector back to the full
    ``2N`` field space."""
    i0, i1, c0, c1, n2 = desc
    i0x, i1x = xp.asarray(i0), xp.asarray(i1)
    c0x = xp.asarray(c0.astype(_C))
    c1x = xp.asarray(c1.astype(_C))
    v = xp.zeros(n2, dtype=_C)
    # i0 entries are all-distinct and i1 entries are all-distinct, so the two
    # fancy-indexed adds need no scatter-accumulate (the only shared index is a
    # fixed point, where c1 == 0).
    v[i0x] = v[i0x] + c0x * ve
    v[i1x] = v[i1x] + c1x * ve
    return v


def _symmetric_solve_rt(Vref, Vtrn, Kx, Ky, EPS, EPS_xx, ez_inv,
                        orders, k0, depth, cinc, xp):
    """Even-parity-sector reflection/transmission (see section header).

    Runs the full single-layer S-matrix recursion in the ``(N+1)``-d even
    subspace and returns the full ``2N`` ``(r, t)`` so the caller's per-order
    efficiency tail is unchanged -- or ``None`` if the symmetry precondition
    fails (the caller then runs the full ``2N`` solve).  The region electric
    eigenvector block is the identity (gauge-invariant), so only the region
    magnetic blocks ``Vref`` / ``Vtrn`` are needed.
    """
    flip = _order_flip_perm(Kx, Ky)
    if flip is None:                                  # oblique / not flip-closed
        return None
    # Move an off-origin symmetry centre to the FFT origin with a diagonal gauge
    # (a centred feature is even about its geometric centre, not sample 0).
    d = _recentering_phase(EPS, orders, xp)
    if d is None:
        return None
    dinv = 1.0 / d

    def _recentre(A):
        return (dinv[:, None] * A) * d[None, :]       # D^{-1} A D (cheap O(N^2))

    EPS = _recentre(EPS)
    if not _flip_invariant(EPS, flip):                # non-centro-symmetric cell
        return None
    EPS_xx = _recentre(EPS_xx)
    ez_inv = _recentre(ez_inv) if ez_inv is not None else None
    if ez_inv is not None and not _flip_invariant(ez_inv, flip):
        return None
    offdiag = EPS - xp.diag(xp.diag(EPS))
    scale = max(1.0, float(xp.max(xp.abs(xp.diag(EPS)))))
    if float(xp.max(xp.abs(offdiag))) < 1e-12 * scale:  # uniform -> analytic path
        return None

    n = flip.shape[0]
    desc = _even_basis_desc(flip)
    # Layer system matrix M = P @ Q (built as in _structured_modes), folded to
    # its even block; the half-size eig replaces the full 2N eig.  The gauge D
    # is a per-order phase, so r/t below are returned in the recentred gauge --
    # |r_i| (hence every efficiency) is gauge-invariant, so no undo is needed.
    Imat = xp.eye(n, dtype=_C)
    EPS_inv = ez_inv if ez_inv is not None else xp.linalg.inv(EPS)
    P = _block(xp, [
        [Kx @ EPS_inv @ Ky,        Imat - Kx @ EPS_inv @ Kx],
        [Ky @ EPS_inv @ Ky - Imat, -Ky @ EPS_inv @ Kx],
    ])
    Q = _layer_Q_matrix(Kx, Ky, EPS, EPS_xx)
    Mp = _even_fold(P @ Q, desc, xp)
    lam2_e, Wl_e = _eig_for(xp)(Mp)
    lam_e = _sqrt_decay(lam2_e)
    Q_e = _even_fold(Q, desc, xp)
    Vl_e = Q_e @ Wl_e @ xp.diag(_inv_lam(lam_e))
    # Region modes folded to even.  W_region = I (2N) -> I in the even basis;
    # only the V blocks carry the half-space index.
    ne = Mp.shape[0]
    Ireg_e = xp.eye(ne, dtype=_C)
    Vref_e = _even_fold(Vref, desc, xp)
    Vtrn_e = _even_fold(Vtrn, desc, xp)
    # S-matrix recursion in the even sector (dimension-agnostic helpers).
    S = _interface_smatrix(Ireg_e, Vref_e, Wl_e, Vl_e)
    S = _redheffer_star(S, _propagation_smatrix(lam_e, k0 * depth))
    S = _redheffer_star(S, _interface_smatrix(Wl_e, Vl_e, Ireg_e, Vtrn_e))
    S11, _S12, S21, _S22 = S
    cinc_e = _even_project(cinc, desc, xp)
    r = _even_unfold(S11 @ cinc_e, desc, xp)
    t = _even_unfold(S21 @ cinc_e, desc, xp)
    return r, t


def _layer_eigenmodes(Kx, Ky, EPS, EPS_xx, ez_laurent_inv=None):
    """Eigenmodes of a single layer (structured or uniform).

    Dimension-agnostic: the harmonic count ``N`` is inferred from ``Kx`` so
    the SAME routine serves the 1-D core (``N = 2*n_orders+1``) and the 2-D
    crossed-grating path (``N = Nx*Ny``).

    The tangential field vector is ``[Sx; Sy]`` (x- and y- electric-field
    harmonics), giving a ``2N x 2N`` system.

    Factorization (Li 1996/1997): the wall-normal field ``E_x`` (and, in
    the ``P`` block, ``E_z``) is discontinuous and needs the inverse rule,
    while the wall-tangential ``E_y`` uses the Laurent rule.  Concretely the
    ``P`` block uses the Laurent inverse ``[[eps]]^{-1}`` (the ``E_z``
    elimination, which is wall-tangential -> direct rule on ``eps``, then
    inverted), and the ``Q`` block uses ``EPS_xx`` for the wall-normal
    ``E_x`` (the Li inverse-rule matrix ``[[1/eps]]^{-1}`` when requested)
    and the Laurent ``EPS`` for the tangential ``E_y``.  This placement is
    what gives the fast TM convergence; putting the inverse-rule matrix in
    the ``P`` inner term instead leaves TM converging at the slow Laurent
    rate.  Non-magnetic (``mu = 1``).

    A laterally UNIFORM layer (diagonal ``[[eps]]``) is detected and solved
    analytically: its 2N modes are doubly degenerate (TE/TM share ``kz``),
    so ``np.linalg.eig`` would return a singular eigenvector matrix.

    Returns ``(W, V, lam)`` -- the electric eigenvector block, the magnetic
    eigenvector block, and the modal eigenvalue ``lam = sqrt(-kz^2/k0^2)``
    (``Re >= 0`` branch; ``= i kz`` propagating, ``= |gamma|`` evanescent),
    which feeds the forward-decaying propagator ``X = exp(-lam k0 L)``.
    """
    xp = array_namespace(Kx, Ky, EPS, EPS_xx)
    Kx = xp.asarray(Kx).astype(_C)
    Ky = xp.asarray(Ky).astype(_C)
    N = Kx.shape[0]
    I = xp.eye(N, dtype=_C)
    Q = _layer_Q_matrix(Kx, Ky, EPS, EPS_xx)
    is_jax = backend_name(xp) == "jax"

    # A laterally UNIFORM (diagonal [[eps]]) layer has DOUBLY-DEGENERATE 2N
    # modes (TE/TM share kz); a general eig then returns an arbitrary, often
    # ill-conditioned eigenvector basis whose interface ``solve`` corrupts the
    # reflected orders.  The analytic modes (W = I, kz from eps) are the
    # well-posed answer.
    def _uniform_modes():
        eps0 = EPS[0, 0]
        kz = _sqrt_forward(eps0 - xp.diag(Kx) ** 2 - xp.diag(Ky) ** 2)
        lam = _sqrt_decay(-xp.concatenate([kz, kz]) ** 2)
        return xp.eye(2 * N, dtype=_C), Q @ xp.diag(_inv_lam(lam)), lam

    def _structured_modes():
        # E_z elimination (P block): inv([[eps]]) by default, or the supplied
        # Laurent [[1/eps]] for the dual-Laurent (analytic-FT) formulation.
        EPS_inv = (ez_laurent_inv if ez_laurent_inv is not None
                   else xp.linalg.inv(EPS))
        P = _block(xp, [
            [Kx @ EPS_inv @ Ky,        I - Kx @ EPS_inv @ Kx],
            [Ky @ EPS_inv @ Ky - I,    -Ky @ EPS_inv @ Kx],
        ])
        lam2, W = _eig_for(xp)(P @ Q)            # Omega^2 = P @ Q
        lam = _sqrt_decay(lam2)                  # = i kz (prop.) / |gamma| (evan.)
        return W, Q @ W @ xp.diag(_inv_lam(lam)), lam

    offdiag = EPS - xp.diag(xp.diag(EPS))
    if not is_jax:
        # NumPy / CuPy: a concrete value test selects the well-posed branch
        # (kept identical to v5.5.0 -- bit-for-bit on both branches).
        scale = max(1.0, float(xp.max(xp.abs(xp.diag(EPS)))))
        if xp.max(xp.abs(offdiag)) < 1e-12 * scale:
            return _uniform_modes()
        return _structured_modes()

    # JAX: a data-dependent ``if`` is illegal under tracing, so compute BOTH
    # and select with ``where`` (the unselected branch stays finite -- the
    # eig is Lorentzian-broadened -- so no NaN leaks into the gradient).  This
    # makes a uniform-isotropic array layer well-posed on the differentiable
    # path too (else its degenerate eig silently broke energy at oblique).
    Wu, Vu, lamu = _uniform_modes()
    Ws, Vs, lams = _structured_modes()
    diagmax = xp.max(xp.abs(xp.diag(EPS)))
    scale = xp.where(diagmax > 1.0, diagmax, 1.0)
    uniform = xp.max(xp.abs(offdiag)) < 1e-12 * scale
    return (xp.where(uniform, Wu, Ws),
            xp.where(uniform, Vu, Vs),
            xp.where(uniform, lamu, lams))


def _homogeneous_eigenmodes(Kx, Ky, eps):
    """Analytic eigenmodes of a UNIFORM half-space of scalar permittivity
    ``eps`` -- the reflection (superstrate) and transmission (substrate)
    regions.  Uses the SAME ``V = Q diag(1/lam)`` convention as
    :func:`_layer_eigenmodes` so propagating AND evanescent orders match at
    every interface.  Dimension-agnostic (``N`` inferred from ``Kx``).
    """
    xp = array_namespace(Kx, Ky)
    Kx = xp.asarray(Kx).astype(_C)
    Ky = xp.asarray(Ky).astype(_C)
    N = Kx.shape[0]
    kx = xp.diag(Kx)
    ky = xp.diag(Ky)
    kz = _sqrt_forward(eps - kx ** 2 - ky ** 2)   # per-order kz/k0
    lam = _sqrt_decay(-xp.concatenate([kz, kz]) ** 2)
    W = xp.eye(2 * N, dtype=_C)
    eps_I = eps * xp.eye(N, dtype=_C)             # uniform: Laurent == inverse rule
    Q = _layer_Q_matrix(Kx, Ky, eps_I, eps_I)
    V = Q @ xp.diag(_inv_lam(lam))
    return W, V, kz


# ===========================================================================
# Redheffer scattering-matrix algebra (gap-free interface + propagation)
# ===========================================================================
#
# The global S-matrix is assembled WITHOUT a free-space gap medium: each
# physical interface gets its own scattering matrix and each layer its own
# pure-propagation matrix, star-multiplied together.  The decaying
# propagator exp(-|gamma| L) <= 1 is thereby kept strictly separate from
# the interface matching, which is the textbook-stable decomposition (S4 /
# Whittaker-Culshaw).  A gap medium instead produces huge evanescent
# reflection blocks (vacuum-vs-high-index mode mismatch) that a near-
# singular star denominator then leaks into the propagating orders -- the
# observed nord=15 blow-up.

def _redheffer_star(SA, SB):
    """Redheffer star product of two block S-matrices, each a 4-tuple
    ``(S11, S12, S21, S22)`` of ``2N x 2N`` blocks."""
    A11, A12, A21, A22 = SA
    B11, B12, B21, B22 = SB
    xp = array_namespace(A11, B11)
    n = A11.shape[0]
    I = xp.eye(n, dtype=_C)
    D = xp.linalg.inv(I - B11 @ A22)
    F = xp.linalg.inv(I - A22 @ B11)
    C11 = A11 + A12 @ D @ B11 @ A21
    C12 = A12 @ D @ B12
    C21 = B21 @ F @ A21
    C22 = B22 + B21 @ F @ A22 @ B12
    return (C11, C12, C21, C22)


def _interface_smatrix(Wa, Va, Wb, Vb):
    """Scattering matrix of the interface from medium ``a`` to medium
    ``b`` (mode matrices ``W, V``), with NO propagation.

    Tangential E and H continuity across the interface gives, with
    ``a = Wb^{-1} Wa`` and ``b = Vb^{-1} Va``::

        S11 = -(a+b)^{-1}(a-b)   S12 = 2 (a+b)^{-1}
        S21 = (a+b)/2 - (a-b)(a+b)^{-1}(a-b)/2   S22 = (a-b)(a+b)^{-1}

    ``solve`` is used for the ``Wb^{-1}Wa`` / ``Vb^{-1}Va`` products so the
    deliberately tiny-columned evanescent eigenvectors do not blow up an
    explicit inverse.
    """
    xp = array_namespace(Wa, Va, Wb, Vb)
    a = xp.linalg.solve(Wb, Wa)
    b = xp.linalg.solve(Vb, Va)
    apb = a + b
    amb = a - b
    iapb = xp.linalg.inv(apb)
    S11 = -iapb @ amb
    S12 = 2.0 * iapb
    S21 = 0.5 * (apb - amb @ iapb @ amb)
    S22 = amb @ iapb
    return (S11, S12, S21, S22)


def _propagation_smatrix(lam, k0_L):
    """Pure-propagation S-matrix of a layer: forward and backward modes
    each acquire ``X = exp(-lam k0 L)`` (a phase for propagating orders, a
    decay for evanescent ones), with zero self-reflection."""
    xp = array_namespace(lam)
    n = lam.shape[0]
    X = xp.diag(xp.exp(-lam * k0_L))
    Z = xp.zeros((n, n), dtype=_C)
    return (Z, X, X, Z)


# ===========================================================================
# GENERALIZED S-matrix (explicit forward / backward modes) for full-3x3 tensors
# ===========================================================================
#
# :func:`_interface_smatrix` hardwires the backward modes as ``[W; -V]`` (the
# in-plane / isotropic symmetry ``lam -> -lam``).  A full anisotropic generator
# G breaks that symmetry, so each medium must carry its forward AND backward
# mode matrices independently.  These helpers operate on the 4N x 4N field-mode
# matrix ``M = [[Wf, Wb], [Vf, Vb]]`` and produce S-matrices in the SAME
# Redheffer block convention as :func:`_redheffer_star`, so the recursion is
# unchanged.

def _modes_to_M(Wf, Vf, Wb, Vb):
    """Assemble the 4N x 4N field-mode matrix ``[[Wf, Wb], [Vf, Vb]]``."""
    return _block(array_namespace(Wf, Vf, Wb, Vb), [[Wf, Wb], [Vf, Vb]])


def _interface_smatrix_general(Ma, Mb):
    """Interface S-matrix (medium ``a`` -> medium ``b``) from the full field-mode
    matrices ``Ma, Mb`` (each ``[[Wf, Wb], [Vf, Vb]]``).

    State partition: top 2N = forward ('+') amplitudes, bottom 2N = backward
    ('-').  Returns ``(S11, S12, S21, S22)`` in the same block convention as
    :func:`_interface_smatrix` / :func:`_redheffer_star`.  Built by solving the
    tangential-field continuity ``Ma ca = Mb cb`` for the scattering form
    (``T = inv(Mb) Ma``, re-blocked)."""
    xp = array_namespace(Ma, Mb)
    n2 = Ma.shape[0] // 2
    T = xp.linalg.solve(Mb, Ma)
    T11 = T[:n2, :n2]
    T12 = T[:n2, n2:]
    T21 = T[n2:, :n2]
    T22 = T[n2:, n2:]
    iT22 = xp.linalg.inv(T22)
    S11 = -iT22 @ T21             # a+ -> a-
    S12 = iT22                    # b- -> a-
    S21 = T11 - T12 @ iT22 @ T21  # a+ -> b+
    S22 = T12 @ iT22              # b- -> b+
    return (S11, S12, S21, S22)


def _propagation_smatrix_general(lam_f, lam_b, k0_L):
    """Pure-propagation S-matrix for a layer with explicit forward eigenvalues
    ``lam_f`` (decay ``exp(-lam_f k0 L)``) and backward eigenvalues ``lam_b``
    (the backward branch, ``Re(lam_b) <= 0``, so ``exp(+lam_b k0 L)`` decays).
    No self-reflection."""
    xp = array_namespace(lam_f, lam_b)
    Xf = xp.diag(xp.exp(-lam_f * k0_L))
    Xb = xp.diag(xp.exp(lam_b * k0_L))
    Z = xp.zeros_like(Xf)
    return (Z, Xb, Xf, Z)


# ===========================================================================
# Public 1-D entry point
# ===========================================================================

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
        Refractive indices of the ridge and groove regions of the patterned
        layer (``n = n + i kappa``, ``kappa > 0`` for loss).
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
        plane of periodicity (planar mounting).  Default 0.
    polarization : {'te', 'tm'}, optional
        ``'te'`` (s, E along grooves / y) or ``'tm'`` (p).  Default ``'te'``.
    n_orders : int, optional
        Number of retained orders per side; total harmonics ``2*n_orders+1``.
        Default 11.
    formulation : {'auto', 'laurent', 'li'}, optional
        Fourier factorization.  ``'laurent'`` (direct rule) converges fast
        for dielectrics; ``'li'`` (inverse rule) is required for metals /
        high-contrast TM.  ``'auto'`` picks ``'li'`` for TM or when any
        index is metallic, else ``'laurent'``.
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
    if stabilize and not _is_traced(wavelength):
        last = None
        for bump in _stabilize_bumps(n_orders):
            try:
                return rcwa_efficiency_1d(
                    period, n_ridge, n_groove, n_substrate, n_superstrate,
                    depth, duty_cycle, wavelength, angle=angle,
                    polarization=polarization, n_orders=int(n_orders) + bump,
                    formulation=formulation, stabilize=False, use_gpu=use_gpu)
            except _EnergyError as e:
                last = e
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
        _warn_if_jax_f32("rcwa_efficiency_1d")
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
            import warnings
            warnings.warn(
                f"rcwa_efficiency_1d: ASR coordinate-bridge conditioning is "
                f"poor (cond={_cond_G:.1e}) at n_orders={M}, asr_eta={asr_eta}; "
                f"ASR is a low-to-moderate-order accelerator and the result "
                f"here may be less accurate than the uniform solver. Reduce "
                f"n_orders or asr_eta, or disable ASR (asr_eta=0) for high "
                f"order counts.", stacklevel=2)
    else:
        EPS, EPS_II = _binary_grating_convolutions(n_ridge, n_groove,
                                                   duty_cycle, M)
    # Wall-normal E_x uses the Li inverse rule [[1/eps]]^{-1} when requested
    # (TM / metals); E_y (tangential) always uses the Laurent [[eps]].
    EPS_xx = EPS_II if use_li else EPS

    # --- region (half-space) modes (physical-x basis, UNCHANGED by ASR) -
    Wref, Vref, kz_ref = _homogeneous_eigenmodes(Kx, Ky, eps_sup)
    Wtrn, Vtrn, kz_trn = _homogeneous_eigenmodes(Kx, Ky, eps_sub)

    # --- global S = (sup|layer) * propagate(layer) * (layer|sub) --------
    Wl, Vl, lam = _layer_eigenmodes(Kx_layer, Ky, EPS, EPS_xx)
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
    safe_r = xp.where(xp.abs(kz_ref) < 1e-12, 1.0, kz_ref)
    safe_t = xp.where(xp.abs(kz_trn) < 1e-12, 1.0, kz_trn)
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
    kz_inc = xp.real(_sqrt_forward(eps_sup - kx0 ** 2))
    if polarization == "te":
        einc_sq = 1.0
    else:
        einc_sq = 1.0 + (kx0 / kz_inc) ** 2
    R_eff = xp.real(kz_ref / kz_inc) * (xp.abs(rx) ** 2 + xp.abs(ry) ** 2
                                        + xp.abs(rz) ** 2) / einc_sq
    T_eff = xp.real(kz_trn / kz_inc) * (xp.abs(tx) ** 2 + xp.abs(ty) ** 2
                                        + xp.abs(tz) ** 2) / einc_sq
    R_eff = xp.where(xp.real(kz_ref) > 0, xp.real(R_eff), 0.0)
    T_eff = xp.where(xp.real(kz_trn) > 0, xp.real(T_eff), 0.0)
    if not is_jax:
        _check_energy("rcwa_efficiency_1d", R_eff, T_eff)
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
    polarization: str = "te",
    n_orders: int = 11,
    formulation: str = "auto",
    quantity: str = "transmitted",
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
    for i, w in enumerate(wl):
        orders, R, T = rcwa_efficiency_1d(
            period, n_ridge, n_groove, n_substrate, n_superstrate, depth,
            duty_cycle, float(w), angle=angle, polarization=polarization,
            n_orders=n_orders, formulation=formulation)
        idx = np.searchsorted(orders, order)
        if idx >= orders.shape[0] or orders[idx] != order:
            raise ValueError(
                f"rcwa_efficiency_vs_wavelength: order {order} is outside the "
                f"retained range +/-{n_orders}; increase n_orders.")
        out[i] = (T[idx] if quantity == "transmitted" else R[idx])
    return out if np.ndim(wavelengths) else out[0]


def _order_key(o):
    """Hashable order key: an int for the 1-D order index, a ``(m, n)`` tuple
    for a 2-D order pair."""
    o = np.atleast_1d(np.asarray(o))
    return int(o[0]) if o.size == 1 else tuple(int(x) for x in o)


def _max_aligned_delta(o_lo, A_lo, o_hi, A_hi):
    """Max ``|A_lo - A_hi|`` over the diffraction orders present in BOTH solves
    (aligned by order index/pair).  ``A`` may be ``(N,)`` or ``(2, N)`` (Jones),
    the order axis being the last."""
    A_lo = np.asarray(to_numpy(A_lo))
    A_hi = np.asarray(to_numpy(A_hi))
    hi_map = {_order_key(o): j for j, o in enumerate(np.asarray(to_numpy(o_hi)))}
    dmax = 0.0
    for i, o in enumerate(np.asarray(to_numpy(o_lo))):
        j = hi_map.get(_order_key(o))
        if j is not None:
            dmax = max(dmax, float(np.max(np.abs(A_lo[..., i] - A_hi[..., j]))))
    return dmax


def _rcwa_convergence_stack(stack, *, bump, atol, warn):
    """``rcwa_convergence`` for a configured :class:`RCWAStack`: solve at its
    current ``n_orders`` AND a ``bump``-higher count, compare the per-order
    efficiencies, and return ``(high_result, report)`` (the bumped
    :class:`RCWAResult`).

    The stack's truncation lives on the object (``nox`` / ``noy``), not in a
    kwarg, so it is bumped in place and RESTORED in a ``finally`` (a 2-D stack
    bumps both axes).  The two solves are compared on their ``(2, N)`` per-order
    R / T, aligned by order index (the high solve retains strictly more orders).
    """
    base_nox, base_noy = stack.nox, stack.noy
    low = stack.solve()
    o_lo, R_lo, T_lo = low.efficiencies()
    try:
        stack.nox = base_nox + int(bump)
        if not stack.is_1d:
            stack.noy = base_noy + int(bump)
        high = stack.solve()
    finally:
        stack.nox, stack.noy = base_nox, base_noy
    o_hi, R_hi, T_hi = high.efficiencies()
    delta = max(_max_aligned_delta(o_lo, R_lo, o_hi, R_hi),
                _max_aligned_delta(o_lo, T_lo, o_hi, T_hi))
    dsR = abs(float(np.sum(to_numpy(R_lo))) - float(np.sum(to_numpy(R_hi))))
    dsT = abs(float(np.sum(to_numpy(T_lo))) - float(np.sum(to_numpy(T_hi))))
    converged = delta <= atol
    no_lo = {"n_orders": base_nox} if stack.is_1d else {
        "n_orders": base_nox, "n_orders_y": base_noy}
    no_hi = {"n_orders": base_nox + int(bump)} if stack.is_1d else {
        "n_orders": base_nox + int(bump), "n_orders_y": base_noy + int(bump)}
    report = dict(converged=converged, delta=delta, delta_sum_R=dsR,
                  delta_sum_T=dsT, n_orders_low=no_lo, n_orders_high=no_hi)
    if warn and not converged:
        import warnings
        warnings.warn(
            f"rcwa_convergence: RCWAStack NOT converged at {no_lo} -- the "
            f"per-order efficiency changed by {delta:.2e} (> atol={atol:.1e}) "
            f"going to {no_hi}; the lower-order result may be unreliable. "
            f"Increase n_orders.", stacklevel=3)
    return high, report


def rcwa_convergence(solver, *, order_params=("n_orders",), bump=4, atol=1e-3,
                     warn=True, **kwargs):
    """Run an RCWA ``solver`` at the requested harmonic count AND a higher one,
    and report whether the answer has converged in the retained order count.

    A too-low truncation can manufacture a *plausible but wrong* result -- most
    dangerously a spurious deep reflection null at a sharp / high-Q resonance --
    so this solves twice (each name in ``order_params`` bumped by ``bump``) and
    compares the per-order efficiencies, warning if the largest change exceeds
    ``atol``.  Cheap insurance (one extra solve) against silently
    under-resolved physics; for a converged *value* (not just a check) see
    :func:`rcwa_extrapolate`.

    Parameters
    ----------
    solver : callable or RCWAStack
        An RCWA efficiency entry point returning ``(orders, R, T, ...)`` --
        e.g. :func:`rcwa_efficiency_1d`, :func:`rcwa_efficiency_2d`,
        :func:`rcwa_jones_1d` -- OR a configured :class:`RCWAStack` (with its
        source + layers already set).  For a stack the truncation it is bumped
        is its ``n_orders`` (and ``n_orders_y`` for a 2-D stack); ``kwargs`` /
        ``order_params`` are ignored and the two ``RCWAResult`` solves are
        compared on their per-order efficiencies (the HIGHER-resolution
        ``RCWAResult`` is returned).
    order_params : tuple of str, optional
        The harmonic-count keyword(s) to bump (callable ``solver`` only).
        ``("n_orders",)`` (default) for the 1-D / Jones-1-D solvers;
        ``("n_orders_x", "n_orders_y")`` for 2-D.
    bump : int, optional
        Increment added to each order parameter for the high-resolution solve
        (default 4).
    atol : float, optional
        Convergence tolerance on the largest per-order efficiency change
        (default ``1e-3``).
    warn : bool, optional
        Emit a ``UserWarning`` when not converged (default ``True``).
    **kwargs
        Passed verbatim to a callable ``solver`` (must include the
        ``order_params`` and the geometry); ignored for an :class:`RCWAStack`.

    Returns
    -------
    result : tuple or RCWAResult
        The HIGHER-resolution solve (the more trustworthy of the two) -- the
        ``solver(**kwargs_bumped)`` tuple for a callable, or the bumped
        :class:`RCWAResult` for a stack.
    report : dict
        ``converged`` (bool), ``delta`` (max per-order efficiency change),
        ``delta_sum_R`` / ``delta_sum_T`` (change in total R / T),
        ``n_orders_low`` / ``n_orders_high`` (the two truncations).
    """
    if isinstance(solver, RCWAStack):
        return _rcwa_convergence_stack(solver, bump=bump, atol=atol, warn=warn)
    for p in order_params:
        if p not in kwargs:
            raise ValueError(
                f"rcwa_convergence: order parameter {p!r} not found in kwargs; "
                f"pass it (and set order_params to the solver's harmonic-count "
                f"argument names).")
    low = solver(**kwargs)
    hi_kwargs = dict(kwargs)
    for p in order_params:
        hi_kwargs[p] = int(kwargs[p]) + int(bump)
    high = solver(**hi_kwargs)

    o_lo, R_lo, T_lo = low[0], low[1], low[2]
    o_hi, R_hi, T_hi = high[0], high[1], high[2]
    delta = max(_max_aligned_delta(o_lo, R_lo, o_hi, R_hi),
                _max_aligned_delta(o_lo, T_lo, o_hi, T_hi))
    dsR = abs(float(np.sum(to_numpy(R_lo))) - float(np.sum(to_numpy(R_hi))))
    dsT = abs(float(np.sum(to_numpy(T_lo))) - float(np.sum(to_numpy(T_hi))))
    converged = delta <= atol
    report = dict(converged=converged, delta=delta, delta_sum_R=dsR,
                  delta_sum_T=dsT,
                  n_orders_low={p: int(kwargs[p]) for p in order_params},
                  n_orders_high={p: int(hi_kwargs[p]) for p in order_params})
    if warn and not converged:
        import warnings
        warnings.warn(
            f"rcwa_convergence: NOT converged at {report['n_orders_low']} -- the "
            f"per-order efficiency changed by {delta:.2e} (> atol={atol:.1e}) "
            f"going to {report['n_orders_high']}; the lower-order result may be "
            f"unreliable (a too-low truncation can fabricate a spurious "
            f"resonance/null). Increase the order count.", stacklevel=2)
    return high, report


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


def rcwa_extrapolate(values, *, n_orders=None, method="richardson"):
    """Extrapolate a slowly-converging RCWA quantity toward its
    ``n_orders -> infinity`` limit from a few finite-``n_orders`` samples.

    Two estimators, picked by ``method``:

    - ``'richardson'`` (default) assumes the **algebraic** tail
      ``s(N) = L + C N^{-p}`` typical of Fourier-truncated RCWA (sharp
      permittivity steps give ``p ~ 1``).  Using the last three samples it
      solves the order ``p`` from the finite-difference ratio and returns the
      intercept ``L``.  Needs ``n_orders`` (the harmonic counts the samples
      were taken at).
    - ``'shanks'`` applies the iterated Shanks (epsilon) transform
      ``S(s)_k = (s_{k+1} s_{k-1} - s_k^2)/(s_{k+1} + s_{k-1} - 2 s_k)``,
      which is exact for a **geometric** tail ``s_k = L + A r^k`` (e.g. the
      exponential convergence of a spectral / PMM solver).  Index-based, so
      ``n_orders`` is optional.

    .. important::
       Extrapolation assumes a **smoothly / monotonically** converging
       sequence.  An irregular sequence -- e.g. a metallic cell with sharp
       corners under rectangular truncation, whose tail wiggles -- can make
       either estimator overshoot; treat the result as an *estimate* and
       cross-check against a direct higher-``N`` solve.  It is most reliable
       for clean dielectric convergence and as a per-order error gauge.

    Parameters
    ----------
    values : array-like of float
        The quantity at increasing ``n_orders`` (at least 3 samples).
    n_orders : array-like of int, optional
        The harmonic counts the samples were taken at (required for
        ``method='richardson'``; ignored by ``'shanks'``).
    method : {'richardson', 'shanks'}, optional

    Returns
    -------
    estimate : float
        The extrapolated ``N -> infinity`` value.
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size < 3:
        raise ValueError(
            "rcwa_extrapolate: need at least 3 samples (the quantity at "
            "increasing n_orders) to extrapolate.")
    if method == "shanks":
        # One Aitken / Shanks pass: S_k = (s_{k+1} s_{k-1} - s_k^2) /
        # (s_{k+1} + s_{k-1} - 2 s_k), exact for a geometric tail.  Return the
        # most-converged (highest-k) accelerated estimate.  We do NOT iterate
        # to a single point: once the sequence flattens the denominator -> 0
        # and a second pass divides rounding noise, overshooting wildly.
        num = v[2:] * v[:-2] - v[1:-1] ** 2
        den = v[2:] + v[:-2] - 2.0 * v[1:-1]
        acc = np.where(np.abs(den) < 1e-15 * (np.abs(v[1:-1]) + 1e-30),
                       v[1:-1], num / den)
        return float(acc[-1])
    if method == "richardson":
        if n_orders is None:
            raise ValueError(
                "rcwa_extrapolate: method='richardson' needs n_orders (the "
                "harmonic counts the samples were taken at).")
        N = np.asarray(n_orders, dtype=float).ravel()
        if N.size != v.size:
            raise ValueError(
                "rcwa_extrapolate: n_orders and values must have the same "
                f"length, got {N.size} and {v.size}.")
        n1, n2, n3 = N[-3:]
        s1, s2, s3 = v[-3:]
        denom = s3 - s2
        if abs(denom) < 1e-300 or abs(s2 - s1) < 1e-300:
            return float(s3)                 # already converged at the tail
        target = (s2 - s1) / denom

        def _ratio(p):
            a, b, c = n1 ** -p, n2 ** -p, n3 ** -p
            d = b - c
            return (a - b) / d if abs(d) > 1e-300 else np.inf

        # bisect p in [0.2, 8] for the monotone ratio == target; fall back to
        # the dominant p = 1 rate if the bracket does not contain a root.
        lo, hi = 0.2, 8.0
        flo, fhi = _ratio(lo) - target, _ratio(hi) - target
        if not np.isfinite(flo) or not np.isfinite(fhi) or flo * fhi > 0:
            p = 1.0
        else:
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                fmid = _ratio(mid) - target
                if flo * fmid <= 0:
                    hi, fhi = mid, fmid
                else:
                    lo, flo = mid, fmid
            p = 0.5 * (lo + hi)
        # model s = L + C N^-p  ->  s2 - s1 = C (N2^-p - N1^-p)
        C = (s2 - s1) / (n2 ** -p - n1 ** -p)
        return float(s1 - C * n1 ** -p)
    raise ValueError(
        f"rcwa_extrapolate: method must be 'richardson' or 'shanks', got "
        f"{method!r}.")


# ===========================================================================
# 2-D crossed gratings (doubly periodic)
# ===========================================================================
#
# The 2-D path reuses the dimension-agnostic vectorial machinery above
# (_layer_eigenmodes, _homogeneous_eigenmodes, the gap-free interface /
# propagation / Redheffer assembly, the _sqrt_forward/_sqrt_decay branches,
# the loss-conjugation bridge and the longitudinal-field efficiency
# formula) verbatim; only the harmonic indexing (a 2-D reciprocal lattice)
# and the permittivity convolution (block-Toeplitz-of-block-Toeplitz)
# become two-dimensional.

def _harmonic_orders_2d(n_orders_x, n_orders_y, *, truncation="rectangular",
                        period_x=None, period_y=None):
    """Flat list of integer ``(m, n)`` diffraction-order pairs on the 2-D
    reciprocal lattice (``m`` slow in ``[-Mx..Mx]``, ``n`` fast in
    ``[-My..My]``).  Returns ``(orders, N)`` with ``orders`` an ``(N, 2)``
    int array.

    ``truncation='rectangular'`` (default) keeps the full Cartesian box,
    ``N = (2 Mx + 1)(2 My + 1)`` -- bit-for-bit the historical order set.

    ``truncation='circular'`` (Lalanne 1997) keeps only the orders inside the
    largest reciprocal-space circle inscribed in that box,
    ``(m / period_x)^2 + (n / period_y)^2 <= R^2`` with
    ``R = min(Mx / period_x, My / period_y)``.  This gives isotropic
    resolution and drops the wasted high-|G| corner orders, so a target
    accuracy is reached with fewer harmonics (and, since the eig is
    ``O(N^3)``, less work).  The kept set is period-dependent, so
    ``period_x`` / ``period_y`` are required.  The (0, 0) order is always
    retained.
    """
    Mx, My = int(n_orders_x), int(n_orders_y)
    m = np.repeat(np.arange(-Mx, Mx + 1), 2 * My + 1)
    n = np.tile(np.arange(-My, My + 1), 2 * Mx + 1)
    orders = np.stack([m, n], axis=1)
    if truncation == "rectangular":
        return orders, orders.shape[0]
    if truncation == "circular":
        if period_x is None or period_y is None:
            raise ValueError(
                "_harmonic_orders_2d: truncation='circular' needs period_x "
                "and period_y (the reciprocal-circle radius is "
                "period-dependent).")
        gx = m / float(period_x)
        gy = n / float(period_y)
        r2 = min(Mx / float(period_x), My / float(period_y)) ** 2
        keep = (gx ** 2 + gy ** 2) <= r2 * (1.0 + 1e-9)
        orders = orders[keep]
        return orders, orders.shape[0]
    raise ValueError(
        f"_harmonic_orders_2d: truncation must be 'rectangular' or 'circular', "
        f"got {truncation!r}.")


def _eps_convolution_2d(eps_cell, orders, n_orders_x, n_orders_y):
    """``N x N`` Laurent (direct-rule) permittivity convolution matrix from a
    one-cell sampling ``eps_cell`` (shape ``(Sx, Sy)``).

    Entry ``[p, p'] = c_{(m-m'), (n-n')}`` where ``c`` are the centred 2-D
    Fourier coefficients of ``eps``; built by vectorised fancy-indexing into
    the coefficient table (the block-Toeplitz-Toeplitz structure).
    """
    xp = array_namespace(eps_cell)
    Mx, My = int(n_orders_x), int(n_orders_y)
    eps_cell = xp.asarray(eps_cell).astype(_C)
    Sx, Sy = int(eps_cell.shape[0]), int(eps_cell.shape[1])
    full = xp.fft.fft2(eps_cell) / (Sx * Sy)  # full[k, l] = c_{k,l} (periodic)
    # Coefficient table over the difference range k in [-2Mx..2Mx], l in
    # [-2My..2My].  The index arrays are plain NumPy ints (lattice indices),
    # which index a NumPy / CuPy / JAX `full` identically via broadcasting.
    krange = np.arange(-2 * Mx, 2 * Mx + 1) % Sx
    lrange = np.arange(-2 * My, 2 * My + 1) % Sy
    table = full[krange[:, None], lrange[None, :]]  # (4Mx+1, 4My+1)
    dm = orders[:, 0][:, None] - orders[:, 0][None, :]   # (N, N)
    dn = orders[:, 1][:, None] - orders[:, 1][None, :]
    return table[dm + 2 * Mx, dn + 2 * My]


def _nv_field_2d(eps_cell, period_x, period_y, *, method="smoothed_gradient",
                 sigma_px=1.5, eps_reg_frac=1e-3):
    """Real normal-vector field ``(Nx, Ny)`` over the unit cell for the
    normal-vector FFF (Schuster 2007).  ``Nx, Ny`` have the same ``(Sx, Sy)``
    shape as ``eps_cell`` and point ACROSS every material boundary, unit-norm
    in the near-boundary band and tapered toward 0 in homogeneous regions.

    ``method='smoothed_gradient'`` (default, Goetz 2008): the normalised
    gradient of a Gaussian-smoothed material indicator ``|eps - eps_bg|``.
    Shape-agnostic -- works for numeric cells, disks, ellipses, multi-shape
    cells.  ``sigma_px`` is the smoothing width in pixels.

    ``method='xy_wedge'``: the closed-form axis-aligned field (Schuster
    Fig. 7c) -- the cell is split by the diagonals of a single centred
    rectangle and each wedge carries the wall normal it faces.  Exact for one
    axis-aligned rectangular feature; ``sigma_px`` is ignored.

    ``eps_cell`` MUST be the INTERNAL (loss-bridge-conjugated) sample so the
    indicator lives in the same namespace as the eps operators.  The field is
    built on the host (real NumPy) regardless of the backend; the (Sx, Sy)
    products feed ``_eps_convolution_2d`` which is backend-agnostic.
    """
    eps_np = to_numpy(eps_cell)
    Sx, Sy = int(eps_np.shape[0]), int(eps_np.shape[1])
    if method == "xy_wedge":
        ux = (np.arange(Sx) + 0.5) / Sx
        uy = (np.arange(Sy) + 0.5) / Sy
        xx, yy = np.meshgrid(ux, uy, indexing="ij")
        p = xx - 0.5
        q = yy - 0.5
        Nx = np.zeros((Sx, Sy))
        Ny = np.zeros((Sx, Sy))
        vert = np.abs(p) >= np.abs(q)            # left/right wedge -> normal +/- x
        Nx[vert] = np.sign(p[vert])
        Ny[~vert] = np.sign(q[~vert])
        bad = (Nx == 0) & (Ny == 0)              # exact centre row/col (measure 0)
        Nx[bad] = 1.0
        return Nx, Ny
    if method != "smoothed_gradient":
        raise ValueError(
            f"_nv_field_2d: method must be 'smoothed_gradient' or 'xy_wedge', "
            f"got {method!r}.")
    f = np.abs(eps_np - eps_np.flat[0]).astype(float)
    gx = 2.0 * np.pi * np.fft.fftfreq(Sx)        # per-pixel angular frequency
    gy = 2.0 * np.pi * np.fft.fftfreq(Sy)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    phi_hat = np.fft.fft2(f) * np.exp(-0.5 * sigma_px ** 2 * (GX ** 2 + GY ** 2))
    dphidx = np.real(np.fft.ifft2(1j * GX * phi_hat))
    dphidy = np.real(np.fft.ifft2(1j * GY * phi_hat))
    mag = np.hypot(dphidx, dphidy)
    eps_reg = eps_reg_frac * (mag.max() + 1e-300)
    # Taper N -> 0 where the gradient is tiny (homogeneous interior); the
    # softened normalisation keeps |N| <= 1 everywhere.
    Nx = dphidx / np.sqrt(mag ** 2 + eps_reg ** 2)
    Ny = dphidy / np.sqrt(mag ** 2 + eps_reg ** 2)
    # Exact unit-norm renormalisation in the near-boundary band (large |grad|).
    band = mag > 0.1 * (mag.max() + 1e-300)
    nn = np.hypot(Nx, Ny)
    nn_safe = np.where(nn < 1e-12, 1.0, nn)
    Nx[band] = (Nx / nn_safe)[band]
    Ny[band] = (Ny / nn_safe)[band]
    norm_max = float(np.hypot(Nx, Ny).max())
    if not norm_max <= 1.0 + 1e-6:
        raise AssertionError(
            f"_nv_field_2d: normal-vector field exceeds unit norm "
            f"(max |N| = {norm_max:.6f}); the projector would not be a "
            f"valid orthogonal projection.")
    return Nx, Ny


def _nv_convolutions_2d(eps_cell, Nx, Ny, orders, n_orders_x, n_orders_y, xp):
    """Normal-vector FFF in-plane tensor operators (Schuster 2007).

    Returns ``(Cxx, Cxy, Cyx, Cyy, EZZ)`` for the full-tensor layer eigensolver
    ``_layer_eigenmodes_tensor``.  With the flux split (``N_z = 0``)::

        [Dx]   [ E - D.Nxx     -D.Nxy   ] [Sx]
        [Dy] = [ -D.Nxy      E - D.Nyy  ] [Sy]

    where ``E = [[eps]]`` (Laurent / direct rule), ``Einv = [[1/eps]]^{-1}``
    (Li inverse rule), ``D = E - Einv`` (the delta operator that the
    wall-normal projection ``N N^T`` switches to the inverse rule), and
    ``Nab = [[Na Nb]]`` (Laurent convolution of the real field products).

    ``EZZ = Einv = [[1/eps]]^{-1}`` (the DUAL-LAURENT / Li ``E_z`` elimination,
    load-bearing): the tensor eigensolver uses ``inv(EZZ) = [[1/eps]]`` for the
    ``P`` block, which is the SAME ``E_z`` rule the analytic-shape solver
    :func:`rcwa_efficiency_2d_shapes` uses and the rule that matches the
    staircase-free reference on a clean dielectric (the direct-rule
    ``EZZ = [[eps]]`` is biased low by ~6e-2 there).  Pairing the in-plane
    normal-vector projection with the dual-Laurent ``E_z`` is what makes the
    factorization reduce to the rigorous answer for both dielectrics and metals.

    ``eps_cell`` MUST be the INTERNAL (loss-bridge-conjugated) sample; ``Nx, Ny``
    are real host arrays from :func:`_nv_field_2d` on the same grid.
    """
    Mx, My = int(n_orders_x), int(n_orders_y)
    E = _eps_convolution_2d(eps_cell, orders, Mx, My)
    Einv = xp.linalg.inv(_eps_convolution_2d(1.0 / eps_cell, orders, Mx, My))
    Delta = E - Einv
    Nx = xp.asarray(np.asarray(Nx).astype(_C))
    Ny = xp.asarray(np.asarray(Ny).astype(_C))
    Nxx = _eps_convolution_2d(Nx * Nx, orders, Mx, My)
    Nyy = _eps_convolution_2d(Ny * Ny, orders, Mx, My)
    Nxy = _eps_convolution_2d(Nx * Ny, orders, Mx, My)
    Cxx = E - Delta @ Nxx
    Cyy = E - Delta @ Nyy
    Cxy = -(Delta @ Nxy)
    Cyx = Cxy                                    # N N^T is symmetric
    EZZ = Einv                                   # dual-Laurent E_z: inv(EZZ)=[[1/eps]]
    return Cxx, Cxy, Cyx, Cyy, EZZ


@_with_blas_limit
def rcwa_efficiency_2d(
    period_x: float,
    period_y: float,
    eps_cell,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelength: float,
    *,
    theta: float = 0.0,
    phi: float = 0.0,
    polarization: str = "te",
    n_orders_x: int = 5,
    n_orders_y: int = 5,
    formulation: str = "laurent",
    truncation: str = "rectangular",
    stabilize: bool = False,
    symmetry: bool = False,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigorous diffraction efficiencies of a 2-D (doubly periodic) crossed
    grating: a single patterned layer of permittivity ``eps_cell`` between a
    ``n_superstrate`` half-space and a ``n_substrate`` half-space.

    Backend-dispatched (NumPy / CuPy via ``use_gpu`` / differentiable JAX when
    ``eps_cell`` is a JAX array); see :func:`rcwa_efficiency_1d`.

    Parameters
    ----------
    period_x, period_y : float
        Lattice periods along x and y (metres).
    eps_cell : (Sx, Sy) array_like of complex
        Permittivity sampled over one unit cell (PUBLIC convention
        ``Im(eps) > 0`` for loss).  ``Sx``/``Sy`` must comfortably exceed
        ``4*n_orders_{x,y}`` to avoid Fourier aliasing.
    n_substrate, n_superstrate : complex
        Transmission and incidence half-space indices.
    depth : float
        Patterned-layer thickness (metres).
    wavelength : float
        Vacuum wavelength (metres).
    theta, phi : float, optional
        Polar angle from +z and azimuth from +x (radians) of the incident
        plane wave (conical mounting).  Defaults 0.
    polarization : {'te', 'tm'}, optional
        ``'te'`` (s) / ``'tm'`` (p) relative to the plane of incidence.
    n_orders_x, n_orders_y : int, optional
        Retained orders per side along each axis (default 5 -> 11x11 = 121
        harmonics).
    formulation : {'laurent', 'li', 'fff', 'fff_nv'}, optional
        Fourier factorization of the patterned layer.

        - ``'laurent'`` (default) -- the direct rule everywhere
          (``E_z`` elimination uses ``[[eps]]^{-1}``).  Correct and
          fast-converging for low-contrast dielectrics; kept as the default
          for bit-for-bit backward compatibility.
        - ``'li'`` / ``'fff'`` -- the dual-Laurent (Li 1996 inverse-rule)
          factorization: the ``E_z`` elimination uses ``[[1/eps]]`` (the
          Toeplitz of the Fourier coefficients of ``1/eps``) instead of
          ``[[eps]]^{-1}``.  This is the convergence-accelerating rule for
          **TM / metals / high contrast** -- the same formulation the
          analytic-shape solver :func:`rcwa_efficiency_2d_shapes` uses
          unconditionally, and the rule used by mature FMM codes
          (verified to converge toward grcwa / inkstone).  Recommended for
          metallic 2-D gratings; for a *pixelated* ``eps_cell`` the in-plane
          staircase still limits the rate, so prefer the analytic-shape
          solver when the geometry is describable by disks / rectangles.

          NOTE: the 2-D ``'li'`` ``E_z`` elimination uses the ``[[1/eps]]``
          pixel route, which is itself biased on a pixelated cell (a
          y-uniform pixelated stripe converges to a value offset from the
          rigorous 1-D-Li oracle); ``'fff_nv'`` below pairs the in-plane
          projection with the unbiased direct-rule (Laurent) ``E_z``.
        - ``'fff_nv'`` -- the **normal-vector fast Fourier factorization**
          (Schuster 2007): the in-plane permittivity operator is assembled as
          a full 2x2 tensor ``[[Cxx, Cxy], [Cyx, Cyy]]`` from the local
          wall-normal field ``N(x, y)``, so the inverse rule is applied along
          the (spatially varying) wall normal and the direct rule along the
          tangent -- the rigorous Li-1996 factorization generalised to
          arbitrarily oriented walls.  The ``E_z`` elimination uses the
          dual-Laurent ``[[1/eps]]`` rule (``inv(EZZ) = [[1/eps]]``, the same
          unbiased ``E_z`` rule the analytic-shape solver uses).  This is the
          convergence-accelerating factorization for **SEPARABLE / axis-aligned
          2-D metallic gratings**: there it reaches a target accuracy in fewer
          orders than ``'li'`` and matches the rigorous 1-D-Li oracle (absorptance
          to ~6e-5 on a metal stripe).  ``N`` is built automatically from a
          Gaussian-smoothed gradient of the material indicator.

          VALIDATION SCOPE (corrected, 2026-06-07 audit): the diagonal
          normal/tangential projection (the dominant win) is validated for
          AXIS-ALIGNED / SEPARABLE features -- it matches the analytic-shape
          reference on a clean dielectric to ~1e-3 and genuinely beats 2-D
          ``'li'`` / ``'laurent'`` on a metal stripe.  The off-diagonal cross term
          ``Cxy = -Delta @ [[Nx Ny]]`` (nonzero only on CURVED boundaries --
          disks/ellipses) is **NOT validated**: on a lossy metal disk it
          **mis-splits the absorptance by ~50%** -- a lossless-trap failure
          (``R+T+A`` still closes, but the per-channel split is wrong; converged
          oracles give ``A ~ 0.094-0.096``, ``fff_nv`` gives ``A ~ 0.046-0.077``).
          A ``UserWarning`` fires when the geometry is non-separable; **use**
          ``'li'`` **or** ``'laurent'`` **for curved / non-separable walls** until
          the cross-term factorization is corrected.  ``'fff_nv'`` is NumPy / CuPy
          only and is incompatible with the ``symmetry`` even-parity fast path
          (transparently skipped).

        (``'fff'`` is accepted as an alias of ``'li'``.)
    truncation : {'rectangular', 'circular'}, optional
        Order-set shape.  ``'rectangular'`` (default) keeps the full
        ``(2 Mx + 1)(2 My + 1)`` box; ``'circular'`` (Lalanne 1997) keeps
        only the orders inside the inscribed reciprocal circle, giving
        isotropic resolution and dropping the wasted corner orders -- the
        same target accuracy at fewer harmonics (and less ``O(N^3)`` eig
        work).  The returned ``orders`` length ``N`` shrinks accordingly.
    stabilize : bool, optional
        When ``True`` and the energy guard detects the measure-zero
        large-period / high-contrast instability, search the nearby
        truncations (``n_orders +/- {1, 2, ...}``, nearest first, BOTH
        directions -- the clean truncations bracket the bad ones and on some
        LAPACK builds sit below the request) and return the first
        energy-conserving solve (its order count may then differ from the
        request).  Default ``False`` raises (bit-for-bit backward compatible).
        NumPy / CuPy only.
    symmetry : bool, optional
        When ``True`` AND the cell is centro-symmetric AND incidence is normal
        (``theta == 0``), run the WHOLE single-layer solve in the even-parity
        subspace: the ``(0, 0)`` source is even and no operator couples the
        parities, so the odd half is never excited and the eig, interface, and
        Redheffer steps all shrink from ``2N`` to ``~N`` -- a ~2-4x end-to-end
        speed-up that grows with ``n_orders``.  An off-origin symmetry centre
        is handled by a diagonal recentering gauge.  Gated on the exact
        precondition; if it does not hold (oblique, non-centro-symmetric, or a
        uniform layer) the solver transparently falls back to the full ``2N``
        solve, so the result is always correct.  The even-adapted basis differs
        from the default by a mode-wise rescale/reorder, so efficiencies match
        the ``symmetry=False`` path to ~1e-12 but NOT bit-for-bit.  Default
        ``False``.  NumPy / CuPy only (no effect under JAX tracing).

    Returns
    -------
    orders : (N, 2) int ndarray
        Diffraction-order pairs ``(m, n)``.
    R_eff, T_eff : (N,) float ndarray
        Reflected / transmitted diffraction efficiency per order.
    """
    if stabilize and not (_is_traced(wavelength) or _is_traced(theta)):
        last = None
        for bump in _stabilize_bumps(min(int(n_orders_x), int(n_orders_y))):
            try:
                return rcwa_efficiency_2d(
                    period_x, period_y, eps_cell, n_substrate, n_superstrate,
                    depth, wavelength, theta=theta, phi=phi,
                    polarization=polarization,
                    n_orders_x=int(n_orders_x) + bump,
                    n_orders_y=int(n_orders_y) + bump, formulation=formulation,
                    truncation=truncation, stabilize=False, symmetry=symmetry,
                    use_gpu=use_gpu)
            except _EnergyError as e:
                last = e
        raise last
    _validate_geometry("rcwa_efficiency_2d",
                       **_concrete(period=period_x, period_y=period_y,
                                   depth=depth, wavelength=wavelength),
                       n_orders=n_orders_x, n_orders_y=n_orders_y)
    _validate_cell_sampling("rcwa_efficiency_2d", eps_cell,
                            n_orders_x, n_orders_y)
    polarization = _normalize_pol("rcwa_efficiency_2d", polarization)
    formulation = _normalize_2d_formulation("rcwa_efficiency_2d", formulation)

    xp = _rcwa_xp("rcwa_efficiency_2d", use_gpu, eps_cell)
    is_jax = backend_name(xp) == "jax"
    if is_jax:
        _warn_if_jax_f32("rcwa_efficiency_2d")
        if formulation == "fff_nv":
            raise NotImplementedError(
                "rcwa_efficiency_2d: formulation='fff_nv' (normal-vector FFF) "
                "has no JAX/differentiable path -- its normal-vector field is "
                "built on the host (real NumPy gradient).  Use 'laurent' or "
                "'li' for gradient-based design, or call on NumPy/CuPy.")

    # Loss-convention bridge (see rcwa_efficiency_1d): conjugate the PUBLIC
    # permittivity in the active namespace (so a JAX eps_cell stays
    # differentiable); the region scalars stay host complex.
    eps_cell = xp.conj(xp.asarray(eps_cell).astype(_C))
    eps_sup = complex(np.conj(_C(n_superstrate) ** 2))
    eps_sub = complex(np.conj(_C(n_substrate) ** 2))

    orders, N = _harmonic_orders_2d(n_orders_x, n_orders_y,
                                    truncation=truncation,
                                    period_x=period_x, period_y=period_y)
    nre = float(np.real(np.sqrt(eps_sup)))
    kx0 = nre * np.sin(theta) * np.cos(phi)        # concrete host floats
    ky0 = nre * np.sin(theta) * np.sin(phi)
    # Run the grazing / non-propagating guards whenever the GEOMETRY is
    # concrete (always on NumPy/CuPy; on JAX when angle/wavelength aren't
    # differentiated), so a region Rayleigh anomaly / non-propagating incidence
    # is caught instead of poisoning the gradient with NaN.  The (traced) cell
    # permittivities are added to the nudge only on the non-JAX path.
    geom_concrete = not (_is_traced(wavelength) or _is_traced(theta)
                         or _is_traced(phi))
    if not is_jax or geom_concrete:
        _require_propagating_incidence("rcwa_efficiency_2d", eps_sup,
                                       kx0 ** 2 + ky0 ** 2)
        eps_reals = [eps_sup, eps_sub]
        if not is_jax:
            eps_reals += [float(xp.real(eps_cell).min()),
                          float(xp.real(eps_cell).max())]
        wl_eff = _grazing_safe_wavelength(
            float(wavelength), kx0, ky0, orders[:, 0], orders[:, 1], period_x,
            period_y, eps_reals)
    else:
        wl_eff = wavelength
    k0 = 2.0 * np.pi / wl_eff
    kx = kx0 + orders[:, 0] * (wl_eff / period_x)   # host float arrays
    ky = ky0 + orders[:, 1] * (wl_eff / period_y)
    # Build the (constant) K matrices on the host then move to the backend so
    # they share eps_cell's namespace (mixing backends in one op would raise).
    Kx = xp.asarray(np.diag(kx.astype(_C)))
    Ky = xp.asarray(np.diag(ky.astype(_C)))
    kxv = xp.asarray(kx.astype(_C))
    kyv = xp.asarray(ky.astype(_C))

    EPS = _eps_convolution_2d(eps_cell, orders, n_orders_x, n_orders_y)
    EPS_xx = EPS  # Laurent rule: wall-normal convolution == [[eps]]
    # Dual-Laurent (Li) z-rule: E_z elimination uses [[1/eps]] (Toeplitz of
    # the Fourier coefficients of 1/eps) rather than [[eps]]^{-1}.  This is
    # the convergence-accelerating factorization for TM / metals; gated so
    # the default 'laurent' path stays bit-for-bit unchanged.
    ez_inv = (_eps_convolution_2d(1.0 / eps_cell, orders, n_orders_x,
                                  n_orders_y)
              if formulation == "li" else None)
    # Normal-vector FFF (Schuster 2007): build the local wall-normal field from
    # the SAME internal (loss-bridge-conjugated) cell and assemble the full
    # in-plane tensor operator (the inverse rule projected on the normal, the
    # direct rule on the tangent).  Routed through the tensor eigensolver below.
    if formulation == "fff_nv":
        Nx_nv, Ny_nv = _nv_field_2d(eps_cell, period_x, period_y)
        # Non-separability gate (2026-06-07 audit P1-A).  The NV cross term
        # [[Nx*Ny]] is ~0 for axis-aligned / separable patterns (where fff_nv is
        # validated and beats 2-D li/laurent on metal stripes) but is exercised on
        # CURVED walls -- where the cross-term factorization mis-splits absorption
        # by ~50% (a lossless-trap failure: R+T+A closes but the channel split is
        # wrong).  Warn rather than silently return a wrong absorptance.
        _nv_cross = float(np.max(np.abs(np.asarray(Nx_nv) * np.asarray(Ny_nv))))
        if _nv_cross > 1e-2:
            import warnings
            warnings.warn(
                "rcwa_efficiency_2d(formulation='fff_nv'): the geometry is "
                "NON-SEPARABLE (curved / non-axis-aligned walls; NV cross term "
                f"max|Nx*Ny| = {_nv_cross:.3g}).  fff_nv's cross-term "
                "factorization is NOT validated there -- it can mis-split "
                "absorptance by ~50% (total R+T+A still closes).  Use "
                "formulation='li' or 'laurent' for curved / non-separable "
                "patterns; fff_nv is validated for separable axis-aligned "
                "features.", stacklevel=2)
        Cxx_nv, Cxy_nv, Cyx_nv, Cyy_nv, EZZ_nv = _nv_convolutions_2d(
            eps_cell, Nx_nv, Ny_nv, orders, n_orders_x, n_orders_y, xp)

    Wref, Vref, kz_ref = _homogeneous_eigenmodes(Kx, Ky, eps_sup)
    Wtrn, Vtrn, kz_trn = _homogeneous_eigenmodes(Kx, Ky, eps_sub)

    # Incident unit plane wave on the (0, 0) order, TE/TM relative to the
    # plane of incidence (built from the in-plane azimuth direction).
    delta = xp.asarray(((orders[:, 0] == 0) & (orders[:, 1] == 0)).astype(_C))
    kz_inc = float(np.real(_sqrt_forward(eps_sup - kx0 ** 2 - ky0 ** 2)))
    kt = float(np.hypot(kx0, ky0))
    if kt < 1e-12:                       # normal incidence
        ex0, ey0 = (0.0, 1.0) if polarization == "te" else (1.0, 0.0)
        einc_sq = 1.0
    else:
        ax, ay = kx0 / kt, ky0 / kt      # in-plane (rho) unit vector
        if polarization == "te":
            ex0, ey0 = -ay, ax           # s-pol: perpendicular, no z-component
            einc_sq = 1.0
        else:
            ex0, ey0 = ax, ay            # p-pol transverse part along rho
            einc_sq = 1.0 + (kt / kz_inc) ** 2
    cinc = xp.concatenate([ex0 * delta, ey0 * delta])

    # Opt-in even-parity-sector fast path: a centro-symmetric cell at normal
    # incidence excites only even modes, so the whole recursion runs in the
    # (N+1)-d even subspace (see the section header above _symmetric_solve_rt).
    # Returns None -> not applicable -> fall through to the full 2N solve.
    # GATED OFF for 'fff_nv': _symmetric_solve_rt is hard-wired to the scalar
    # (EPS, EPS_xx, ez_inv) core and cannot represent the full in-plane tensor.
    rt = None
    if symmetry and not is_jax and kt < 1e-12 and formulation != "fff_nv":
        rt = _symmetric_solve_rt(Vref, Vtrn, Kx, Ky, EPS, EPS_xx, ez_inv,
                                 orders, k0, depth, cinc, xp)
    if rt is not None:
        r, t = rt
    else:
        if formulation == "fff_nv":
            Wl, Vl, lam = _layer_eigenmodes_tensor(
                Kx, Ky, Cxx_nv, Cxy_nv, Cyx_nv, Cyy_nv, EZZ_nv)
        else:
            Wl, Vl, lam = _layer_eigenmodes(Kx, Ky, EPS, EPS_xx,
                                            ez_laurent_inv=ez_inv)
        S = _interface_smatrix(Wref, Vref, Wl, Vl)
        S = _redheffer_star(S, _propagation_smatrix(lam, k0 * depth))
        S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wtrn, Vtrn))
        S11, S12, S21, S22 = S
        r = S11 @ cinc
        t = S21 @ cinc
    rx, ry = r[:N], r[N:]
    tx, ty = t[:N], t[N:]
    safe_r = xp.where(xp.abs(kz_ref) < 1e-12, 1.0, kz_ref)
    safe_t = xp.where(xp.abs(kz_trn) < 1e-12, 1.0, kz_trn)
    rz = -(kxv * rx + kyv * ry) / safe_r
    tz = -(kxv * tx + kyv * ty) / safe_t
    R_eff = xp.real(kz_ref / kz_inc) * (xp.abs(rx) ** 2 + xp.abs(ry) ** 2
                                        + xp.abs(rz) ** 2) / einc_sq
    T_eff = xp.real(kz_trn / kz_inc) * (xp.abs(tx) ** 2 + xp.abs(ty) ** 2
                                        + xp.abs(tz) ** 2) / einc_sq
    R_eff = xp.where(xp.real(kz_ref) > 0, xp.real(R_eff), 0.0)
    T_eff = xp.where(xp.real(kz_trn) > 0, xp.real(T_eff), 0.0)
    if not is_jax:
        _check_energy("rcwa_efficiency_2d", R_eff, T_eff)
    return orders, R_eff, T_eff


# ===========================================================================
# 1-D anisotropic gratings (full in-plane permittivity tensor -- LC subset)
# ===========================================================================
#
# A layer whose permittivity is a 3x3 tensor with a full in-plane block
# [[exx, exy], [eyx, eyy]] (plus ezz; eps_xz = eps_yz = 0, the z-decoupled
# liquid-crystal subset).  The in-plane block couples TE and TM, so the
# reflection is a full 2x2 Jones matrix.  Factorization is Li 1996
# (anisotropic 1-D): the wall-normal x-direction uses the inverse rule, the
# tangential y the direct rule.

def uniaxial_tensor(n_o, n_e, theta, *, phi=0.0):
    """``(3, 3)`` rotated uniaxial permittivity tensor (the liquid-crystal /
    birefringent layer).

    The optic axis (director) starts along ``+z``
    (``eps = diag(n_o^2, n_o^2, n_e^2)``), is tilted by the polar angle
    ``theta`` from ``+z`` toward ``+x``, then rotated azimuthally by ``phi``
    about ``+z`` (``R = Rz(phi) @ Ry(theta)``).  ``theta = pi/2`` puts the
    director in the x-y plane (the in-plane-rotating LC), giving a full
    in-plane block with ``ezz = n_o^2`` -- the z-decoupled subset used by
    the 1-D / 2-D anisotropic solvers.

    Parameters
    ----------
    n_o, n_e : complex
        Ordinary and extraordinary indices (``n + i kappa`` for loss).
    theta : float
        Director polar tilt from ``+z`` (radians).
    phi : float, optional
        Director azimuth about ``+z`` (radians).  Default 0.

    Returns
    -------
    (3, 3) complex ndarray
        Permittivity tensor in the lab ``(x, y, z)`` basis.
    """
    eo = _C(n_o) ** 2
    ee = _C(n_e) ** 2
    eps_diag = np.diag([eo, eo, ee]).astype(_C)
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    Ry = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]], dtype=_C)
    Rz = np.array([[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]], dtype=_C)
    R = Rz @ Ry
    return R @ eps_diag @ R.T


def _toeplitz_of_profile(profile, n_orders):
    """Laurent (direct-rule) Toeplitz ``[[f]]`` of a sampled one-period
    profile."""
    return _toeplitz_1d(_fourier_coeffs_1d(profile, 2 * n_orders + 1), n_orders)


def _inv_toeplitz_of_profile(profile, n_orders):
    """Inverse-rule operator ``[[1/f]]^{-1}`` of a sampled one-period
    profile."""
    xp = array_namespace(profile)
    return xp.linalg.inv(
        _toeplitz_1d(_fourier_coeffs_1d(1.0 / profile, 2 * n_orders + 1),
                     n_orders))


def _tensor_convolutions(profiles, n_orders):
    """Anisotropic 1-D Fourier operators (Li 1996; wall normal along x).

    ``profiles`` holds the one-period samplings of the tensor components
    ``xx, xy, yx, yy, zz``.  Returns ``(Cxx, Cxy, Cyx, Cyy, EZZ)`` where
    ``[Dx; Dy] = [[Cxx, Cxy], [Cyx, Cyy]] [Ex; Ey]`` and ``EZZ = [[ezz]]``
    (the wall-tangential ``E_z`` uses the direct rule, inverted later in the
    ``P`` block).  Reduces to ``Cxx = Cyy = [[eps]]``, ``Cxy = Cyx = 0`` for
    a scalar (isotropic) tensor.
    """
    xp = array_namespace(profiles["xx"])
    a = xp.asarray(profiles["xx"]).astype(_C)
    b = xp.asarray(profiles["xy"]).astype(_C)
    c = xp.asarray(profiles["yx"]).astype(_C)
    d = xp.asarray(profiles["yy"]).astype(_C)
    ezz = xp.asarray(profiles["zz"]).astype(_C)
    inv_a = _inv_toeplitz_of_profile(a, n_orders)             # [[1/exx]]^{-1}
    T_b_a = _toeplitz_of_profile(b / a, n_orders)             # [[exy/exx]]
    T_c_a = _toeplitz_of_profile(c / a, n_orders)             # [[eyx/exx]]
    T_schur = _toeplitz_of_profile(d - c * b / a, n_orders)   # [[eyy - eyx exy/exx]]
    Cxx = inv_a
    Cxy = inv_a @ T_b_a
    Cyx = T_c_a @ inv_a
    Cyy = T_schur + T_c_a @ inv_a @ T_b_a
    EZZ = _toeplitz_of_profile(ezz, n_orders)
    return Cxx, Cxy, Cyx, Cyy, EZZ


def _tensor_has_offplane(profiles):
    """True if any of the off-plane profile keys (``xz, zx, yz, zy``) is present
    and above a tiny RELATIVE tolerance.  Used to branch onto the full-3x3 path
    WITHOUT perturbing the in-plane result, so the legacy 5-tuple path is
    bit-identical when the tensor is in-plane (the same ``1e-9 * scale`` cutoff
    as :func:`_tensor_offplane_present`, so the convolution helper and the
    ``rcwa_jones_1d`` routing decision agree -- roundoff-level off-plane from a
    pi/2 director rotation does NOT trip the full path)."""
    off = 0.0
    for key in ("xz", "zx", "yz", "zy"):
        p = profiles.get(key)
        if p is None:
            continue
        if is_jax_array(p):
            # JAX: cannot test concretely here -- assume present (caller decides).
            return True
        off = max(off, float(np.max(np.abs(np.asarray(to_numpy(p))))))
    if off == 0.0:
        return False
    scale = 1.0
    for key in ("xx", "yy", "zz"):
        p = profiles.get(key)
        if p is not None and not is_jax_array(p):
            scale = max(scale, float(np.max(np.abs(np.asarray(to_numpy(p))))))
    return off > 1e-9 * scale


def _tensor_convolutions_full(profiles, n_orders):
    """Full anisotropic 1-D Fourier operators with OUT-OF-PLANE coupling
    (Li 2003; wall normal along x).

    ``profiles`` holds the one-period samplings of ALL nine tensor components
    ``xx, xy, yx, yy, zz`` and (optionally) ``xz, zx, yz, zy``.  Returns the
    9-tuple ``(Cxx, Cxy, Cyx, Cyy, EZZ, EZX, EZY, EXZ, EYZ)``.

    The in-plane 2x2 block ``[[Cxx, Cxy], [Cyx, Cyy]]`` is built from the
    ``ezz``-Schur-REDUCED effective in-plane profile -- done POINTWISE in x
    (a_eff = exx - exz ezx/ezz, etc.) BEFORE the existing wall-normal-x Li
    factorization (inverse-rule-on-x + Schur), which is then run on the
    ``*_eff`` profiles.  ``EZX, EZY, EXZ, EYZ`` are the direct-rule (Laurent)
    Toeplitz operators of the off-plane components, feeding the A, B generator
    cross-blocks in :func:`_layer_eigenmodes_tensor`.

    When the off-plane keys are ABSENT or all zero, ``(Cxx, Cxy, Cyx, Cyy, EZZ)``
    is bit-identical to :func:`_tensor_convolutions` (the effective profile then
    equals the raw in-plane profile, branched on absence -- no ``-0`` subtraction
    perturbs it) and ``EZX = EZY = EXZ = EYZ = 0``.
    """
    xp = array_namespace(profiles["xx"])
    has_off = _tensor_has_offplane(profiles)
    if not has_off:
        # Bit-identical in-plane path + zero off-plane operators.
        Cxx, Cxy, Cyx, Cyy, EZZ = _tensor_convolutions(profiles, n_orders)
        N = 2 * n_orders + 1
        Z = xp.zeros((N, N), dtype=_C)
        return Cxx, Cxy, Cyx, Cyy, EZZ, Z, Z, Z, Z

    a = xp.asarray(profiles["xx"]).astype(_C)
    b = xp.asarray(profiles["xy"]).astype(_C)
    c = xp.asarray(profiles["yx"]).astype(_C)
    d = xp.asarray(profiles["yy"]).astype(_C)
    ezz = xp.asarray(profiles["zz"]).astype(_C)
    exz = xp.asarray(profiles["xz"]).astype(_C)
    ezx = xp.asarray(profiles["zx"]).astype(_C)
    eyz = xp.asarray(profiles["yz"]).astype(_C)
    ezy = xp.asarray(profiles["zy"]).astype(_C)

    # ----- ezz Schur reduction, POINTWISE in x (do NOT commute as FT ops) -----
    # Eliminate Ez = (1/ezz)(Dz - ezx Ex - ezy Ey); substituting into the Dx,Dy
    # rows gives the effective in-plane 2x2 tensor profile.
    inv_ezz = 1.0 / ezz
    a_eff = a - exz * ezx * inv_ezz
    b_eff = b - exz * ezy * inv_ezz
    c_eff = c - eyz * ezx * inv_ezz
    d_eff = d - eyz * ezy * inv_ezz

    # ----- existing wall-normal-x Li factorization on the EFFECTIVE profile ---
    inv_a = _inv_toeplitz_of_profile(a_eff, n_orders)        # [[1/a_eff]]^{-1}
    T_b_a = _toeplitz_of_profile(b_eff / a_eff, n_orders)
    T_c_a = _toeplitz_of_profile(c_eff / a_eff, n_orders)
    T_schur = _toeplitz_of_profile(d_eff - c_eff * b_eff / a_eff, n_orders)
    Cxx = inv_a
    Cxy = inv_a @ T_b_a
    Cyx = T_c_a @ inv_a
    Cyy = T_schur + T_c_a @ inv_a @ T_b_a
    EZZ = _toeplitz_of_profile(ezz, n_orders)

    # ----- direct-rule operators for the generator cross-blocks A, B ----------
    EZX = _toeplitz_of_profile(ezx, n_orders)
    EZY = _toeplitz_of_profile(ezy, n_orders)
    EXZ = _toeplitz_of_profile(exz, n_orders)
    EYZ = _toeplitz_of_profile(eyz, n_orders)
    return Cxx, Cxy, Cyx, Cyy, EZZ, EZX, EZY, EXZ, EYZ


def _select_forward_flux(gam, Vfull, N):
    """Generalized all-harmonic flux-based forward-mode selector for the full
    anisotropic generator G (Li 2003).

    Returns EXACTLY ``2N`` indices of the FORWARD (outgoing toward ``+z``) modes.
    A mode is classified by the net Poynting z-flux SUMMED OVER ALL harmonics
    (the m=0-only rule is correct ONLY at M=0)::

        Ex = v[:N], Ey = v[N:2N], Hx = v[2N:3N]/1j, Hy = v[3N:4N]/1j
        Sz = real( sum( Ex conj(Hy) - Ey conj(Hx) ) )

    Forward = (propagating, ``|Re(gam)| < tol``): ``Sz > 0``; (evanescent):
    ``Re(gam) > 0`` (decaying as ``exp(-lam k0 z)``).  This is gauge-robust and,
    unlike a +-pair search, does NOT throw on NON-RECIPROCAL media (whose
    spectrum is not +-symmetric).

    A defensive rebalance keeps the count at exactly ``2N`` if a near-zero-flux
    propagating mode (``|Sz| < flux_tol``) would otherwise tip the split: the
    excess/deficit is resolved by the signed flux (most-forward kept)."""
    xp = array_namespace(Vfull)
    n = gam.shape[0]
    gre = xp.real(gam)
    tol = 1e-7 * float(xp.maximum(xp.asarray(1.0), xp.max(xp.abs(gam))))
    Ex = Vfull[:N, :]
    Ey = Vfull[N:2 * N, :]
    Hx = Vfull[2 * N:3 * N, :] / 1j
    Hy = Vfull[3 * N:4 * N, :] / 1j
    Sz = xp.real(xp.sum(Ex * xp.conj(Hy) - Ey * xp.conj(Hx), axis=0))   # (n,)
    prop = xp.abs(gre) < tol
    fwd = xp.where(prop, Sz > 0, gre > 0)
    idx = xp.asarray(np.where(to_numpy(fwd))[0])
    if int(idx.shape[0]) == 2 * N:
        return idx
    # ---- defensive rebalance to EXACTLY 2N (near-zero-flux / cut tie) --------
    Sz_np = np.asarray(to_numpy(Sz))
    gre_np = np.asarray(to_numpy(gre))
    prop_np = np.asarray(to_numpy(prop))
    # Rank all modes by a signed "forwardness" score: propagating by Sz,
    # evanescent by Re(gam).  The 2N largest scores are forward.
    score = np.where(prop_np, Sz_np, gre_np)
    order = np.argsort(-score)
    fwd_fixed = np.zeros(n, dtype=bool)
    fwd_fixed[order[:2 * N]] = True
    return np.where(fwd_fixed)[0]


def _layer_eigenmodes_tensor(Kx, Ky, Cxx, Cxy, Cyx, Cyy, EZZ,
                             EZX=None, EZY=None, EXZ=None, EYZ=None):
    """Eigenmodes of a full-in-plane-tensor layer (dimension-agnostic).

    The anisotropic ``Q`` block (rigorously derived and locked to the
    isotropic core by ``b = c = 0`` reduction; getting the ``Cyx`` sign
    wrong silently violates energy conservation at off-axis director
    angles) is::

        Q = [[ Cyx + Kx Ky,   Cyy - Kx Kx ],
             [ Ky Ky - Cxx,   -(Cxy + Ky Kx) ]]

    The ``P`` block is the core's, with the ``E_z`` elimination ``inv(EZZ)``.

    OUT-OF-PLANE (full-3x3, Li 2003): when ``EZX, EZY, EXZ, EYZ`` are supplied
    (not None), the layer ODE picks up the off-plane cross-blocks ``A`` (from
    ``ezx, ezy``) and ``B`` (from ``exz, eyz``), so the first-order generator
    ``G = [[A, P], [Q, B]]`` is eigendecomposed directly (the in-plane symmetry
    ``[W; -V] <-> -lam`` is BROKEN, so forward AND backward modes are genuinely
    distinct).  This path returns the 6-tuple ``(W, V, lam, Wb, Vb, lam_b)`` --
    forward E/H-block + eigenvalues and backward E/H-block + eigenvalues -- for
    the GENERALIZED S-matrix (:func:`_interface_smatrix_general`).  When all four
    are None the EXACT current ``eig(P@Q)`` path runs byte-for-byte (the
    isotropic / JAX branches are untouched).
    """
    xp = array_namespace(Kx, Ky, Cxx, Cxy, Cyx, Cyy, EZZ)
    Kx = xp.asarray(Kx).astype(_C)
    Ky = xp.asarray(Ky).astype(_C)
    N = Kx.shape[0]
    I = xp.eye(N, dtype=_C)
    Ez_inv = xp.linalg.inv(EZZ)
    P = _block(xp, [
        [Kx @ Ez_inv @ Ky,        I - Kx @ Ez_inv @ Kx],
        [Ky @ Ez_inv @ Ky - I,    -Ky @ Ez_inv @ Kx],
    ])
    Q = _block(xp, [
        [Cyx + Kx @ Ky,        Cyy - Kx @ Kx],
        [Ky @ Ky - Cxx,        -(Cxy + Ky @ Kx)],
    ])
    if any(t is not None for t in (EZX, EZY, EXZ, EYZ)):
        # ---- full-3x3 (out-of-plane) generator path (Li 2003) ---------------
        Z = xp.zeros((N, N), dtype=_C)
        EZX = Z if EZX is None else xp.asarray(EZX).astype(_C)
        EZY = Z if EZY is None else xp.asarray(EZY).astype(_C)
        EXZ = Z if EXZ is None else xp.asarray(EXZ).astype(_C)
        EYZ = Z if EYZ is None else xp.asarray(EYZ).astype(_C)
        # A block: Ez = inv(EZZ)(Dz - EZX Ex - EZY Ey) feeds -Kx inv(EZZ) EZX etc.
        A = _block(xp, [
            [-Kx @ Ez_inv @ EZX,   -Kx @ Ez_inv @ EZY],
            [-Ky @ Ez_inv @ EZX,   -Ky @ Ez_inv @ EZY],
        ])
        # B block: the exz/eyz feedback into the modal-H rows (CORRECTED block,
        # validated against the Berreman 4x4 Delta).
        B = _block(xp, [
            [EYZ @ Ez_inv @ Ky,    -EYZ @ Ez_inv @ Kx],
            [-EXZ @ Ez_inv @ Ky,   EXZ @ Ez_inv @ Kx],
        ])
        G = _block(xp, [[A, P], [Q, B]])
        gam, Vfull = _eig_for(xp)(G)
        fidx = _select_forward_flux(gam, Vfull, N)
        fset = set(np.asarray(to_numpy(fidx)).tolist())
        bidx = xp.asarray(np.array(sorted(set(range(4 * N)) - fset)))
        lam = gam[fidx]
        lam_b = gam[bidx]
        Vf = Vfull[:, fidx]
        Vb = Vfull[:, bidx]
        W = Vf[:2 * N, :]
        V = Vf[2 * N:, :]
        Wb = Vb[:2 * N, :]
        Vbk = Vb[2 * N:, :]
        return W, V, lam, Wb, Vbk, lam_b
    lam2, W = _eig_for(xp)(P @ Q)
    lam = _sqrt_decay(lam2)
    V = Q @ W @ xp.diag(_inv_lam(lam))
    if backend_name(xp) != "jax":
        return W, V, lam

    # JAX: an ISOTROPIC-uniform tensor layer (Cxx = Cyy = eps0 I, Cxy = Cyx = 0)
    # is doubly degenerate, so jnp's eig returns an ill-conditioned basis that
    # corrupts the reflected orders (NumPy's eig happens to stay well-posed).
    # Blend in the analytic uniform modes (W = I) when that degeneracy is
    # detected -- tracer-safe ``where``, no NaN (the eig is broadened).
    eps0 = Cxx[0, 0]
    kz = _sqrt_forward(eps0 - xp.diag(Kx) ** 2 - xp.diag(Ky) ** 2)
    lam_u = _sqrt_decay(-xp.concatenate([kz, kz]) ** 2)
    Wu = xp.eye(2 * N, dtype=_C)
    Vu = Q @ xp.diag(_inv_lam(lam_u))
    aniso = (xp.max(xp.abs(Cxx - eps0 * I)) + xp.max(xp.abs(Cyy - eps0 * I))
             + xp.max(xp.abs(Cxy)) + xp.max(xp.abs(Cyx)))
    scale = xp.where(xp.abs(eps0) > 1.0, xp.abs(eps0), 1.0)
    iso_uniform = aniso < 1e-10 * scale
    return (xp.where(iso_uniform, Wu, W),
            xp.where(iso_uniform, Vu, V),
            xp.where(iso_uniform, lam_u, lam))


def _tensor_offplane_present(*tensors):
    """True if any concrete ``(3, 3)`` (or ``(..., 3, 3)``) tensor has
    OUT-OF-PLANE coupling (``eps_xz, eps_yz, eps_zx, eps_zy`` above a tiny
    relative tolerance).  JAX / non-3x3 inputs are skipped (treated as
    in-plane)."""
    for t in tensors:
        if t is None or is_jax_array(t):
            continue
        a = np.asarray(to_numpy(t)).astype(_C)
        if a.shape[-2:] != (3, 3):
            continue
        offz = np.maximum.reduce([np.abs(a[..., 0, 2]), np.abs(a[..., 1, 2]),
                                  np.abs(a[..., 2, 0]), np.abs(a[..., 2, 1])])
        diag = np.maximum.reduce([np.abs(a[..., 0, 0]), np.abs(a[..., 1, 1]),
                                  np.abs(a[..., 2, 2])])
        scale = max(float(np.max(diag)), 1.0)
        if float(np.max(offz)) > 1e-9 * scale:
            return True
    return False


def _reject_jax_offplane(fn_name, *tensors):
    """Raise ``NotImplementedError`` if any JAX tensor argument carries
    OUT-OF-PLANE coupling (``eps_xz / eps_yz / eps_zx / eps_zy != 0``).

    The 1-D full-3x3 (out-of-plane) solver routes through
    :func:`_select_forward_flux`, whose forward/backward mode split is a
    ``np.where`` / ``argsort`` on host-materialised flux -- a HARD, non-
    differentiable selection that breaks the autodiff graph (mirroring the
    ``rcwa_efficiency_2d`` ``formulation='fff_nv'`` JAX rejection).  Worse, the
    plain in-plane router (:func:`_tensor_offplane_present`) SKIPS JAX arrays, so
    a JAX off-plane tensor would otherwise be SILENTLY treated as in-plane and
    its z-coupling dropped -- a quietly wrong gradient.  So reject it explicitly.

    A CONCRETE JAX array is inspected for the off-plane block; a TRACER (under
    ``jax.grad`` / ``jax.jit``) cannot be materialised, so its off-plane content
    is undetectable here -- the docstring documents that the JAX path supports
    the in-plane tensor subset only."""
    for t in tensors:
        if t is None or not is_jax_array(t):
            continue
        try:                                 # concrete JAX array -> inspectable
            a = np.asarray(to_numpy(t)).astype(_C)
        except Exception:                    # tracer -> not materialisable
            continue
        if a.shape[-2:] != (3, 3):
            continue
        offz = np.maximum.reduce([np.abs(a[..., 0, 2]), np.abs(a[..., 1, 2]),
                                  np.abs(a[..., 2, 0]), np.abs(a[..., 2, 1])])
        diag = np.maximum.reduce([np.abs(a[..., 0, 0]), np.abs(a[..., 1, 1]),
                                  np.abs(a[..., 2, 2])])
        scale = max(float(np.max(diag)), 1.0)
        if float(np.max(offz)) > 1e-9 * scale:
            raise NotImplementedError(
                f"{fn_name}: a JAX (differentiable) tensor with OUT-OF-PLANE "
                f"coupling (eps_xz / eps_yz / eps_zx / eps_zy != 0 -- e.g. a "
                f"tilted-director LC) has NO differentiable path: the full-3x3 "
                f"solver's forward-mode selection (_select_forward_flux) uses a "
                f"host np.where / argsort that breaks the autodiff graph. Use an "
                f"IN-PLANE tensor (exx, exy, eyx, eyy, ezz; e.g. a theta=pi/2 "
                f"uniaxial_tensor) for gradient-based design, or call on "
                f"NumPy/CuPy for the rigorous (non-differentiable) off-plane "
                f"solve.")


def _require_inplane_tensor(fn_name, *tensors, allow_offplane=False):
    """Reject a ``(3, 3)`` permittivity tensor (or ``(..., 3, 3)`` tensor cell)
    with OUT-OF-PLANE coupling -- ``eps_xz, eps_yz, eps_zx, eps_zy``.

    The legacy anisotropic FMM is the z-DECOUPLED in-plane subset
    (``[[exx, exy], [eyx, eyy]]`` + ``ezz``; Li 1996 / 2003).  A tilted-director
    LC, a magneto-optic / gyrotropic medium, or any tensor with x/y<->z coupling
    would have those components SILENTLY DROPPED on the legacy path -- so raise
    instead of returning a quietly wrong answer.  Concrete (NumPy / CuPy) tensors
    only -- a JAX tensor is skipped (not materialisable here) and assumed
    in-plane on the differentiable path.

    The 1-D NumPy/CuPy path now has a FULL-3x3 (out-of-plane) solver (Li 2003,
    v5.11.0), so it passes ``allow_offplane=True`` and this returns whether
    out-of-plane coupling is present (the caller routes to the full path) instead
    of raising.  The 2-D (:func:`rcwa_jones_2d`) and :class:`RCWAStack` paths keep
    ``allow_offplane=False`` (1-D only; 2-D / stack out-of-plane pending)."""
    has_off = _tensor_offplane_present(*tensors)
    if has_off and not allow_offplane:
        raise ValueError(
            f"{fn_name}: the anisotropic path is the z-decoupled in-plane "
            f"tensor subset (exx, exy, eyx, eyy, ezz); the supplied tensor "
            f"has out-of-plane coupling (eps_xz / eps_yz / eps_zx / eps_zy "
            f"!= 0 -- e.g. a tilted-director LC or a magneto-optic / "
            f"gyrotropic tensor), which this solver would silently drop. "
            f"Full 3x3 (out-of-plane) tensors are supported on the 1-D path "
            f"(rcwa_jones_1d); 2-D / RCWAStack out-of-plane is pending.")
    return has_off


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
        safe_r = xp.where(xp.abs(kz_ref) < 1e-12, 1.0, kz_ref)
        safe_t = xp.where(xp.abs(kz_trn) < 1e-12, 1.0, kz_trn)
        rz = -(kxv * rx) / safe_r
        tz = -(kxv * tx) / safe_t
        Re = xp.real(kz_ref / kz_inc) * (xp.abs(rx) ** 2 + xp.abs(ry) ** 2
                                         + xp.abs(rz) ** 2) / einc_sq
        Te = xp.real(kz_trn / kz_inc) * (xp.abs(tx) ** 2 + xp.abs(ty) ** 2
                                         + xp.abs(tz) ** 2) / einc_sq
        R_rows.append(xp.where(xp.real(kz_ref) > 0, xp.real(Re), 0.0))
        T_rows.append(xp.where(xp.real(kz_trn) > 0, xp.real(Te), 0.0))
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
        Permittivity tensors of the ridge and groove (PUBLIC convention
        ``Im(eps) > 0`` for loss).  Pass ``scalar * np.eye(3)`` for an
        isotropic region; build LC tensors with :func:`uniaxial_tensor`.
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
        _warn_if_jax_f32("rcwa_jones_1d")
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
                          complex(eps_groove[0, 0]), complex(eps_groove[1, 1])]
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

    kz_inc = float(np.real(_sqrt_forward(eps_sup - kx0 ** 2)))
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
        _warn_if_jax_f32("rcwa_jones_1d_segments")
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
                eps_reals += [complex(t[0, 0]), complex(t[1, 1])]
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

    kz_inc = float(np.real(_sqrt_forward(eps_sup - kx0 ** 2)))
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


@_with_blas_limit
def rcwa_jones_2d(
    period_x: float,
    period_y: float,
    eps_tensor_cell,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelength: float,
    *,
    theta: float = 0.0,
    phi: float = 0.0,
    n_orders_x: int = 5,
    n_orders_y: int = 5,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rigorous 2-D (doubly periodic) anisotropic grating: a single layer
    whose permittivity is a full in-plane TENSOR FIELD (the z-decoupled LC
    subset; Li 2003, direct-rule factorization).  Returns diffraction
    efficiencies for both incident linear polarizations plus the 2x2
    zeroth-order Jones reflection matrix.

    Backend-dispatched (NumPy / CuPy via ``use_gpu`` / differentiable JAX when
    ``eps_tensor_cell`` is a JAX array); see :func:`rcwa_efficiency_1d`.

    Parameters
    ----------
    period_x, period_y : float
        Lattice periods (metres).
    eps_tensor_cell : (Sx, Sy, 3, 3) array_like of complex
        Per-pixel permittivity tensor over one unit cell (PUBLIC convention
        ``Im(eps) > 0`` for loss).  Only the in-plane block ``[[xx, xy],
        [yx, yy]]`` and ``zz`` are used.  ``Sx``/``Sy`` must exceed
        ``4*n_orders_{x,y}``.
    n_substrate, n_superstrate, depth, wavelength, theta, phi,
    n_orders_x, n_orders_y
        As in :func:`rcwa_efficiency_2d`.

    Returns
    -------
    orders : (N, 2) int ndarray
        Diffraction-order pairs ``(m, n)``.
    R_eff, T_eff : (2, N) float ndarray
        Diffraction efficiencies per order; row 0 is the response to an
        incident ``E_x`` plane wave, row 1 to incident ``E_y``.
    jones_reflection : (2, 2) complex ndarray
        Zeroth-order Jones reflection matrix in the lab ``(x, y)`` basis
        (columns = response to incident ``E_x`` / ``E_y``).

    Notes
    -----
    Uses the direct (Laurent) tensor factorization, which is exactly
    energy-conserving for a lossless tensor and reduces to
    :func:`rcwa_efficiency_2d` for a scalar cell; it converges fastest for
    smooth / dielectric anisotropic media (e.g. liquid-crystal cells).
    """
    _validate_geometry("rcwa_jones_2d",
                       **_concrete(period=period_x, period_y=period_y,
                                   depth=depth, wavelength=wavelength),
                       n_orders=n_orders_x, n_orders_y=n_orders_y)
    _validate_cell_sampling("rcwa_jones_2d", eps_tensor_cell,
                            n_orders_x, n_orders_y)
    _require_inplane_tensor("rcwa_jones_2d", eps_tensor_cell)

    xp = _rcwa_xp("rcwa_jones_2d", use_gpu, eps_tensor_cell)
    is_jax = backend_name(xp) == "jax"
    if is_jax:
        _warn_if_jax_f32("rcwa_jones_2d")
    eps_t = xp.conj(xp.asarray(eps_tensor_cell).astype(_C))
    eps_sup = complex(np.conj(_C(n_superstrate) ** 2))
    eps_sub = complex(np.conj(_C(n_substrate) ** 2))

    orders, N = _harmonic_orders_2d(n_orders_x, n_orders_y)
    nre = float(np.real(np.sqrt(eps_sup)))
    kx0 = nre * np.sin(theta) * np.cos(phi)
    ky0 = nre * np.sin(theta) * np.sin(phi)
    geom_concrete = not (_is_traced(wavelength) or _is_traced(theta)
                         or _is_traced(phi))
    if not is_jax or geom_concrete:
        _require_propagating_incidence("rcwa_jones_2d", eps_sup,
                                       kx0 ** 2 + ky0 ** 2)
        eps_reals = [eps_sup, eps_sub]
        if not is_jax:
            dr = xp.real(eps_t[:, :, [0, 1, 2], [0, 1, 2]])
            eps_reals += [float(dr.min()), float(dr.max())]
        wl_eff = _grazing_safe_wavelength(
            float(wavelength), kx0, ky0, orders[:, 0], orders[:, 1], period_x,
            period_y, eps_reals)
    else:
        wl_eff = wavelength
    k0 = 2.0 * np.pi / wl_eff
    kx = kx0 + orders[:, 0] * (wl_eff / period_x)
    ky = ky0 + orders[:, 1] * (wl_eff / period_y)
    Kx = xp.asarray(np.diag(kx.astype(_C)))
    Ky = xp.asarray(np.diag(ky.astype(_C)))
    kxv = xp.asarray(kx.astype(_C))
    kyv = xp.asarray(ky.astype(_C))

    # Direct-rule (Laurent) convolution of each tensor component.
    def _conv(comp):
        return _eps_convolution_2d(comp, orders, n_orders_x, n_orders_y)
    Cxx = _conv(eps_t[:, :, 0, 0])
    Cxy = _conv(eps_t[:, :, 0, 1])
    Cyx = _conv(eps_t[:, :, 1, 0])
    Cyy = _conv(eps_t[:, :, 1, 1])
    EZZ = _conv(eps_t[:, :, 2, 2])

    Wref, Vref, kz_ref = _homogeneous_eigenmodes(Kx, Ky, eps_sup)
    Wtrn, Vtrn, kz_trn = _homogeneous_eigenmodes(Kx, Ky, eps_sub)
    Wl, Vl, lam = _layer_eigenmodes_tensor(Kx, Ky, Cxx, Cxy, Cyx, Cyy, EZZ)
    S = _interface_smatrix(Wref, Vref, Wl, Vl)
    S = _redheffer_star(S, _propagation_smatrix(lam, k0 * depth))
    S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wtrn, Vtrn))
    S11, S12, S21, S22 = S

    p0 = int(np.where((orders[:, 0] == 0) & (orders[:, 1] == 0))[0][0])
    delta = xp.asarray(((orders[:, 0] == 0) & (orders[:, 1] == 0)).astype(_C))
    kz_inc = float(np.real(_sqrt_forward(eps_sup - kx0 ** 2 - ky0 ** 2)))
    safe_r = xp.where(xp.abs(kz_ref) < 1e-12, 1.0, kz_ref)
    safe_t = xp.where(xp.abs(kz_trn) < 1e-12, 1.0, kz_trn)
    R_rows, T_rows, j_cols = [], [], []
    for ex0, ey0 in ((1.0, 0.0), (0.0, 1.0)):
        # Unit tangential E along (ex0, ey0); the incident wave's longitudinal
        # Ez = -(kx0 ex + ky0 ey)/kz_inc inflates |E_inc|^2 (cf. the 1-D sec^2).
        long_inc = (kx0 * ex0 + ky0 * ey0)
        einc_sq = 1.0 + (long_inc / kz_inc) ** 2 if kz_inc != 0 else 1.0
        cinc = xp.concatenate([ex0 * delta, ey0 * delta])
        r = S11 @ cinc
        t = S21 @ cinc
        rx, ry = r[:N], r[N:]
        tx, ty = t[:N], t[N:]
        rz = -(kxv * rx + kyv * ry) / safe_r
        tz = -(kxv * tx + kyv * ty) / safe_t
        Re = xp.real(kz_ref / kz_inc) * (xp.abs(rx) ** 2 + xp.abs(ry) ** 2
                                         + xp.abs(rz) ** 2) / einc_sq
        Te = xp.real(kz_trn / kz_inc) * (xp.abs(tx) ** 2 + xp.abs(ty) ** 2
                                         + xp.abs(tz) ** 2) / einc_sq
        R_rows.append(xp.where(xp.real(kz_ref) > 0, xp.real(Re), 0.0))
        T_rows.append(xp.where(xp.real(kz_trn) > 0, xp.real(Te), 0.0))
        j_cols.append(xp.stack([xp.conj(rx[p0]), xp.conj(ry[p0])]))
    R_eff = xp.stack(R_rows)
    T_eff = xp.stack(T_rows)
    jones_reflection = xp.stack(j_cols, axis=1)
    if not is_jax:
        _check_energy("rcwa_jones_2d", R_eff, T_eff)
    return orders, R_eff, T_eff, jones_reflection


# ===========================================================================
# Analytic shape Fourier transforms + dual-Laurent 2-D factorization
# ===========================================================================
#
# For known shapes (rectangle, disk, ellipse) the permittivity Fourier
# coefficients are computed in CLOSED FORM (exact form factors) instead of
# by FFT-sampling a pixelated cell -- eliminating the aliasing / staircase
# error of pixelation, so the spectrum is exact and convergence is clean.
# Both [[eps]] and [[1/eps]] are built from the SAME analytic form factors,
# and the layer eigenproblem uses the dual-Laurent factorization (the
# in-plane Q block uses [[eps]]; the E_z elimination uses [[1/eps]]
# directly) -- the formulation used by mature analytic-FT FMM codes.

def _shape_form_factor(shape, gxv, gyv, period_x, period_y):
    """Analytic Fourier form factor ``F(G) = (1/A_cell) integral_shape
    exp(-i G.r) d^2r`` for one shape, at the (difference) reciprocal vectors
    ``(gxv, gyv)`` [1/m].  Closed form for ``rectangle`` / ``disk`` /
    ``ellipse``; the ``G = 0`` entry is the area fraction."""
    kind = shape["shape"]
    cx, cy = shape.get("center", (period_x / 2.0, period_y / 2.0))
    area = period_x * period_y
    phase = np.exp(-1j * (gxv * cx + gyv * cy))
    if kind == "rectangle":
        wx, wy = shape["size"]
        # np.sinc(z) = sin(pi z)/(pi z), so sinc(G w / (2 pi)) = sin(G w/2)/(G w/2).
        sx = np.sinc(gxv * wx / (2.0 * np.pi))
        sy = np.sinc(gyv * wy / (2.0 * np.pi))
        return (wx * wy / area) * sx * sy * phase
    if kind in ("disk", "ellipse"):
        from scipy.special import j1
        if kind == "disk":
            ax = ay = shape["radius"]
        else:
            ax, ay = shape["semi_axes"]
        q = np.sqrt((gxv * ax) ** 2 + (gyv * ay) ** 2)
        small = q < 1e-12
        qsafe = np.where(small, 1.0, q)
        bessel = np.where(small, 1.0, 2.0 * j1(qsafe) / qsafe)   # -> 1 as q -> 0
        return (np.pi * ax * ay / area) * bessel * phase
    raise ValueError(
        f"_shape_form_factor: unknown shape {kind!r} (expected 'rectangle', "
        f"'disk' or 'ellipse').")


def _analytic_convolutions_2d(eps_background, shapes, orders, n_orders_x,
                              n_orders_y, period_x, period_y):
    """Analytic ``[[eps]]`` and ``[[1/eps]]`` convolution matrices for a 2-D
    unit cell of background ``eps_background`` overlaid with ``shapes`` (each
    a dict ``{'shape', 'eps', geometry, ['center']}``).  Returns
    ``(EPS, EPS_inv_laurent)``."""
    Mx, My = int(n_orders_x), int(n_orders_y)
    ks = np.arange(-2 * Mx, 2 * Mx + 1)
    ls = np.arange(-2 * My, 2 * My + 1)
    KK, LL = np.meshgrid(ks, ls, indexing="ij")
    gxv = KK * (2.0 * np.pi / period_x)
    gyv = LL * (2.0 * np.pi / period_y)
    eps_bg = _C(eps_background)
    c_eps = np.zeros(KK.shape, dtype=_C)
    c_inv = np.zeros(KK.shape, dtype=_C)
    c_eps[2 * Mx, 2 * My] = eps_bg            # background (DC) term
    c_inv[2 * Mx, 2 * My] = 1.0 / eps_bg
    for sh in shapes:
        eps_s = _C(sh["eps"])
        F = _shape_form_factor(sh, gxv, gyv, period_x, period_y)
        c_eps = c_eps + (eps_s - eps_bg) * F
        c_inv = c_inv + (1.0 / eps_s - 1.0 / eps_bg) * F
    dm = orders[:, 0][:, None] - orders[:, 0][None, :]
    dn = orders[:, 1][:, None] - orders[:, 1][None, :]
    EPS = c_eps[dm + 2 * Mx, dn + 2 * My]
    EPS_inv = c_inv[dm + 2 * Mx, dn + 2 * My]
    return EPS, EPS_inv


@_with_blas_limit
def rcwa_efficiency_2d_shapes(
    period_x: float,
    period_y: float,
    eps_background: complex,
    shapes,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelength: float,
    *,
    theta: float = 0.0,
    phi: float = 0.0,
    polarization: str = "te",
    n_orders_x: int = 5,
    n_orders_y: int = 5,
    truncation: str = "rectangular",
    use_gpu: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigorous 2-D crossed-grating efficiencies using **analytic** shape
    Fourier transforms and the dual-Laurent factorization.

    Backend-dispatched for NumPy / CuPy (``use_gpu``).  The analytic shape
    form factors are evaluated on the host (closed form) and the resulting
    convolution matrices are moved to the backend for the eigensolve; the
    JAX (differentiable) path is not offered for the analytic-shape solver
    (its Bessel form factors are host ``scipy.special``).

    The patterned layer is a background permittivity ``eps_background``
    overlaid with analytically-described ``shapes`` (no pixelation), so the
    permittivity spectrum is exact and convergence is clean.

    Parameters
    ----------
    period_x, period_y : float
        Lattice periods (metres).
    eps_background : complex
        Permittivity of the unpatterned background (PUBLIC convention
        ``Im(eps) > 0`` lossy).
    shapes : list of dict
        Each shape is ``{'shape': 'rectangle'|'disk'|'ellipse', 'eps':
        complex, ...geometry..., 'center': (cx, cy) [m]}``; geometry is
        ``'size': (wx, wy)`` for a rectangle, ``'radius': r`` for a disk,
        ``'semi_axes': (ax, ay)`` for an ellipse (all metres).  Shapes are
        painted in order over the background.
    n_substrate, n_superstrate, depth, wavelength, theta, phi, polarization,
    n_orders_x, n_orders_y
        As in :func:`rcwa_efficiency_2d`.

    Returns
    -------
    orders : (N, 2) int ndarray
    R_eff, T_eff : (N,) float ndarray
    """
    _validate_geometry("rcwa_efficiency_2d_shapes", period=period_x,
                       period_y=period_y, depth=depth, wavelength=wavelength,
                       n_orders=n_orders_x, n_orders_y=n_orders_y)
    _validate_shapes("rcwa_efficiency_2d_shapes", shapes, period_x, period_y)
    polarization = _normalize_pol("rcwa_efficiency_2d_shapes", polarization)
    truncation = str(truncation)

    xp = _rcwa_xp("rcwa_efficiency_2d_shapes", use_gpu)
    if backend_name(xp) == "jax":
        raise NotImplementedError(
            "rcwa_efficiency_2d_shapes: the analytic-shape solver has no JAX "
            "(differentiable) path; use a JAX eps_cell with rcwa_efficiency_2d "
            "for gradient-based design, or call with use_gpu for CuPy.")
    # Loss-sign bridge: conjugate every public permittivity (host scalars).
    eps_bg = complex(np.conj(_C(eps_background)))
    shapes_c = [dict(s, eps=complex(np.conj(_C(s["eps"])))) for s in shapes]
    eps_sup = complex(np.conj(_C(n_superstrate) ** 2))
    eps_sub = complex(np.conj(_C(n_substrate) ** 2))

    orders, N = _harmonic_orders_2d(n_orders_x, n_orders_y,
                                    truncation=truncation,
                                    period_x=period_x, period_y=period_y)
    nre = float(np.real(np.sqrt(eps_sup)))
    kx0 = nre * np.sin(theta) * np.cos(phi)
    ky0 = nre * np.sin(theta) * np.sin(phi)
    _require_propagating_incidence("rcwa_efficiency_2d_shapes", eps_sup,
                                   kx0 ** 2 + ky0 ** 2)
    layer_eps = [eps_bg] + [s["eps"] for s in shapes_c]
    wl_eff = _grazing_safe_wavelength(
        wavelength, kx0, ky0, orders[:, 0], orders[:, 1], period_x, period_y,
        [eps_sup, eps_sub] + layer_eps)
    k0 = 2.0 * np.pi / wl_eff
    kx = kx0 + orders[:, 0] * (wl_eff / period_x)
    ky = ky0 + orders[:, 1] * (wl_eff / period_y)
    Kx = xp.asarray(np.diag(kx.astype(_C)))
    Ky = xp.asarray(np.diag(ky.astype(_C)))
    kxv = xp.asarray(kx.astype(_C))
    kyv = xp.asarray(ky.astype(_C))

    # Analytic (host) form factors -> move the convolution matrices to the
    # backend for the eigensolve.
    EPS_np, EPS_inv_np = _analytic_convolutions_2d(
        eps_bg, shapes_c, orders, n_orders_x, n_orders_y, period_x, period_y)
    EPS = xp.asarray(EPS_np)
    EPS_inv = xp.asarray(EPS_inv_np)

    Wref, Vref, kz_ref = _homogeneous_eigenmodes(Kx, Ky, eps_sup)
    Wtrn, Vtrn, kz_trn = _homogeneous_eigenmodes(Kx, Ky, eps_sub)
    # Dual-Laurent: in-plane uses [[eps]], E_z elimination uses [[1/eps]].
    Wl, Vl, lam = _layer_eigenmodes(Kx, Ky, EPS, EPS, ez_laurent_inv=EPS_inv)
    S = _interface_smatrix(Wref, Vref, Wl, Vl)
    S = _redheffer_star(S, _propagation_smatrix(lam, k0 * depth))
    S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wtrn, Vtrn))
    S11, S12, S21, S22 = S

    delta = xp.asarray(((orders[:, 0] == 0) & (orders[:, 1] == 0)).astype(_C))
    kz_inc = float(np.real(_sqrt_forward(eps_sup - kx0 ** 2 - ky0 ** 2)))
    kt = float(np.hypot(kx0, ky0))
    if kt < 1e-12:
        ex0, ey0 = (0.0, 1.0) if polarization == "te" else (1.0, 0.0)
        einc_sq = 1.0
    else:
        ax, ay = kx0 / kt, ky0 / kt
        if polarization == "te":
            ex0, ey0 = -ay, ax
            einc_sq = 1.0
        else:
            ex0, ey0 = ax, ay
            einc_sq = 1.0 + (kt / kz_inc) ** 2
    cinc = xp.concatenate([ex0 * delta, ey0 * delta])
    r = S11 @ cinc
    t = S21 @ cinc
    rx, ry = r[:N], r[N:]
    tx, ty = t[:N], t[N:]
    safe_r = xp.where(xp.abs(kz_ref) < 1e-12, 1.0, kz_ref)
    safe_t = xp.where(xp.abs(kz_trn) < 1e-12, 1.0, kz_trn)
    rz = -(kxv * rx + kyv * ry) / safe_r
    tz = -(kxv * tx + kyv * ty) / safe_t
    R_eff = xp.real(kz_ref / kz_inc) * (xp.abs(rx) ** 2 + xp.abs(ry) ** 2
                                        + xp.abs(rz) ** 2) / einc_sq
    T_eff = xp.real(kz_trn / kz_inc) * (xp.abs(tx) ** 2 + xp.abs(ty) ** 2
                                        + xp.abs(tz) ** 2) / einc_sq
    R_eff = xp.where(xp.real(kz_ref) > 0, xp.real(R_eff), 0.0)
    T_eff = xp.where(xp.real(kz_trn) > 0, xp.real(T_eff), 0.0)
    _check_energy("rcwa_efficiency_2d_shapes", R_eff, T_eff)
    return orders, R_eff, T_eff


# ===========================================================================
# JAX backend: differentiable 1-D RCWA for inverse design (adjoint/autodiff)
# ===========================================================================
#
# A JAX reimplementation of the 1-D isotropic core whose efficiencies are
# differentiable w.r.t. continuous design parameters (layer permittivities,
# depth, angle) -- the enabler for gradient-based metasurface inverse
# design.  JAX is an OPTIONAL dependency, imported lazily so the rest of the
# module (and CI without jax) is unaffected.
#
# The one hard gradient is through ``jnp.linalg.eig`` of the non-Hermitian
# layer matrix (reverse-mode AD of general eig is unstable near degenerate
# eigenvalues).  We register a custom VJP with the torcwa-style
# Lorentzian-broadened eigenvector gradient (``eps_broaden``) plus a
# canonical eigenvector gauge, so the decomposition is a deterministic,
# differentiable function of the matrix (validated entrywise vs complex
# finite differences to < 1e-5).

_JAX_EIG_STABLE = None


def _jax_eig_stable():
    """Lazily build (once) and return a differentiable, gauge-fixed ``eig``
    for complex non-Hermitian matrices.  Raises ImportError if JAX absent."""
    global _JAX_EIG_STABLE
    if _JAX_EIG_STABLE is not None:
        return _JAX_EIG_STABLE
    from functools import partial

    import jax
    import jax.numpy as jnp

    @partial(jax.custom_vjp, nondiff_argnums=(1,))
    def _eig_raw(A, eps_broaden=1e-10):
        return jnp.linalg.eig(A)

    def _eig_raw_fwd(A, eps_broaden):
        lam, V = jnp.linalg.eig(A)
        return (lam, V), (lam, V)

    def _eig_raw_bwd(eps_broaden, res, cot):
        lam, V = res
        lam_bar, V_bar = cot
        D = lam[None, :] - lam[:, None]
        n = lam.shape[0]
        offdiag = 1.0 - jnp.eye(n, dtype=D.dtype)
        denom = jnp.abs(D) ** 2 + eps_broaden
        F = jnp.where(offdiag != 0, D / jnp.where(denom == 0, 1.0, denom), 0.0)
        Vinv = jnp.linalg.inv(V)
        VinvH = jnp.conj(Vinv).T
        VH = jnp.conj(V).T
        Mmat = VH @ jnp.conj(V_bar)
        inner = jnp.diag(jnp.conj(lam_bar)) + F * Mmat
        # Conjugate the matrix cotangent to match JAX's Wirtinger convention
        # (dL/d conj(A)); without it the eig-path gradient is silently wrong
        # (verified by a gauge-invariant grid search over the conjugations).
        return (jnp.conj(VinvH @ inner @ VH),)

    _eig_raw.defvjp(_eig_raw_fwd, _eig_raw_bwd)

    # The physical RCWA efficiencies are GAUGE-INVARIANT (independent of the
    # eigenvector phase / ordering), so the solver uses the raw eig with the
    # Lorentzian-broadened custom VJP directly.  A canonical gauge fix
    # (unit-norm + phase pivot + sort) would only be needed if a downstream
    # quantity depended on the gauge -- and its argmax / lexsort are
    # non-differentiable, which corrupts the gradient of a gauge-invariant
    # loss (observed: a 3x error on the eig-path gradient).  So it is omitted.
    _JAX_EIG_STABLE = _eig_raw
    return _eig_raw


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
    import warnings
    warnings.warn(
        "rcwa_efficiency_1d_jax is deprecated since v5.5.1 and will be removed "
        "in v5.7.0; call rcwa_efficiency_1d(...) with jax.numpy arguments "
        "instead (it auto-dispatches to the differentiable JAX backend).",
        DeprecationWarning, stacklevel=2)
    from ..backend import JAX_AVAILABLE as _JAX_AVAILABLE
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
        polarization=polarization,
        n_orders=n_orders,
        formulation=formulation,
    )


# ===========================================================================
# Unified multi-layer API: RCWAStack / RCWAResult (+ caching, Jones bridge)
# ===========================================================================

# Cache of homogeneous half-space eigenmodes -- the most-repeated solve cost:
# the same superstrate / substrate / spacer recurs across a stack and across
# a wavelength or angle sweep.  Guarded by a Lock for thread safety; cleared
# via the library cache registry.
_HOMOG_CACHE: dict = {}
_HOMOG_LOCK = threading.Lock()


def _clear_rcwa_caches() -> None:
    """Clear the RCWA homogeneous-eigenmode cache (registered with the
    library cache registry)."""
    with _HOMOG_LOCK:
        _HOMOG_CACHE.clear()


def _cached_homogeneous_eigenmodes(eps, Kx, Ky, key):
    with _HOMOG_LOCK:
        hit = _HOMOG_CACHE.get(key)
    if hit is not None:
        return hit
    res = _homogeneous_eigenmodes(Kx, Ky, eps)
    with _HOMOG_LOCK:
        _HOMOG_CACHE[key] = res
    return res


class RCWAResult:
    """Result of an :class:`RCWAStack` solve.

    Accessors
    ---------
    efficiencies() -> (orders, R, T)
        ``R``, ``T`` are ``(2, N)`` real arrays: row 0 is the response to an
        incident ``E_x`` plane wave, row 1 to incident ``E_y``; ``orders`` is
        ``(N, 2)`` (2-D) or ``(N,)`` (1-D).
    absorptance() -> (2,) ndarray
        ``1 - sum(R) - sum(T)`` per incident polarization (``>= 0`` for
        passive media -- the loss-sign bridge guarantees the sign).
    jones_reflection() / jones_transmission() -> (2, 2) complex
        Zeroth-order Jones matrices (columns = incident ``E_x`` / ``E_y``,
        rows = ``[E_x; E_y]``).
    apply_reflection(jones_field) -> JonesField
        Apply the zeroth-order Jones reflection to a
        :class:`~lumenairy.elements.polarization.JonesField` -- the bridge
        from a rigorous metasurface reflection into the polarization
        pipeline.
    to_jones_field(nx, ny, dx, ..., order=None) -> JonesField
        Build a JonesField from one diffraction order (``order=None`` is the
        uniform specular field; a non-zero order is a tilted carrier).
    per_order_amplitudes(port) -> dict
        Per-order complex tangential amplitudes ``(2, N)`` + transverse
        k-vectors -- the modal data the solver computes.
    to_multiorder_field(nx, ny, dx, ...) -> JonesField
        Reconstruct the full diffracted field as a superposition of the
        propagating orders (the metasurface-deflector / metalens bridge).
    """

    def __init__(self, orders, R, T, jones_reflection, jones_transmission,
                 modal=None):
        self.orders = orders
        self._R = R
        self._T = T
        self._Jr = jones_reflection
        self._Jt = jones_transmission
        self._modal = modal   # per-order amplitudes + k-vectors (or None)

    def efficiencies(self):
        return self.orders, self._R, self._T

    # -- per-order modal access + multi-order field reconstruction ---------

    def _require_modal(self):
        if self._modal is None:
            raise ValueError(
                "RCWAResult: per-order modal data is unavailable on this "
                "result (it is retained only by RCWAStack.solve()).")
        return self._modal

    def _order_index(self, order):
        """Flat index of a diffraction order: ``m`` (1-D) or ``(m, n)``
        (2-D)."""
        o2 = self._require_modal()["orders2d"]
        from ..backend import to_numpy
        o2 = to_numpy(o2)
        if np.ndim(order) == 0:                      # 1-D: integer order m
            hit = np.where((o2[:, 0] == int(order)) & (o2[:, 1] == 0))[0]
        else:
            m, n = order
            hit = np.where((o2[:, 0] == int(m)) & (o2[:, 1] == int(n)))[0]
        if hit.size == 0:
            raise ValueError(
                f"RCWAResult: order {order!r} is outside the retained range; "
                f"increase n_orders.")
        return int(hit[0])

    def per_order_amplitudes(self, port="reflection"):
        """Per-order complex tangential field amplitudes (PUBLIC ``exp(-iwt)``
        convention) and transverse k-vectors -- the data the multi-order field
        bridge is built on.

        Returns a dict with ``Ex`` / ``Ey`` each ``(2, N)`` (row 0 = response
        to incident ``E_x``, row 1 to incident ``E_y``), the per-order
        ``kx`` / ``ky`` / ``kz`` normalised by ``k0 = 2*pi/wavelength``, the
        ``orders``, and the ``wavelength``.  ``port`` selects the reflection or
        transmission side.

        Note: these are the raw TANGENTIAL field amplitudes, so
        ``|Ex_m|^2 + |Ey_m|^2`` is NOT the order's efficiency -- recovering the
        efficiency needs the Poynting flux weight ``Re(kz_m/kz_inc)``, the
        longitudinal component ``|az_m|^2`` with ``az = -(kx Ex + ky Ey)/kz``,
        and the incident ``|E|^2``.  :meth:`to_multiorder_field` (with the
        default ``normalize='power'``) applies all three for you.
        """
        if port not in ("reflection", "transmission"):
            raise ValueError(
                f"RCWAResult.per_order_amplitudes: port must be 'reflection' "
                f"or 'transmission', got {port!r}.")
        from ..backend import to_numpy
        m = self._require_modal()
        ex, ey = ("rx", "ry") if port == "reflection" else ("tx", "ty")
        kz = m["kz_ref"] if port == "reflection" else m["kz_trn"]
        return dict(orders=self.orders, Ex=to_numpy(m[ex]), Ey=to_numpy(m[ey]),
                    kx=to_numpy(m["kx"]), ky=to_numpy(m["ky"]),
                    kz=to_numpy(kz), wavelength=m["wavelength"])

    def absorptance(self):
        return 1.0 - self._R.sum(axis=1) - self._T.sum(axis=1)

    def jones_reflection(self):
        return self._Jr

    def jones_transmission(self):
        return self._Jt

    def apply_reflection(self, jones_field):
        # apply_jones_matrix transforms its field IN PLACE; operate on a copy
        # so the caller's incident field is preserved.  Note: this carries
        # only the zeroth-order (specular) 2x2 Jones -- for a strongly
        # diffracting cell most power is in non-zero orders, so the returned
        # field is the specular component only (a single plane wave).
        from .polarization import apply_jones_matrix
        return apply_jones_matrix(jones_field.copy(), self._Jr)

    def _order_amplitude(self, idx, incident, port):
        """Incident-weighted PUBLIC [Ex, Ey] tangential amplitude of one order
        (flat index ``idx``)."""
        from ..backend import to_numpy
        m = self._require_modal()
        ex_a, ey_a = ("rx", "ry") if port == "reflection" else ("tx", "ty")
        Ex = to_numpy(m[ex_a])[:, idx]      # (2,) responses to incident Ex/Ey
        Ey = to_numpy(m[ey_a])[:, idx]
        inc = np.asarray(incident, dtype=_C).reshape(2)
        return inc @ Ex, inc @ Ey           # ex*resp_Ex + ey*resp_Ey

    def _order_power_scale(self, idx, ax, ay, incident, port):
        """Scale ``s`` so the deposited tangential power ``|s ax|^2 + |s ay|^2``
        equals the order's TRUE diffraction efficiency.

        A diffracted order carries power ``flux*(|ax|^2+|ay|^2+|az|^2)/einc_sq``
        with the Poynting flux weight ``flux = Re(kz_m/kz_inc)``, the
        longitudinal field ``az = -(kx ax + ky ay)/kz`` (large for steep
        orders), and the incident ``|E|^2 = einc_sq``.  Depositing the raw
        tangential ``|ax|^2+|ay|^2`` drops all three -> the reconstructed field
        violates energy conservation and can show the wrong dominant order.
        Scaling each carrier by ``s = sqrt(efficiency / (|ax|^2+|ay|^2))``
        restores per-order power (and the right dominant order)."""
        from ..backend import to_numpy
        m = self._require_modal()
        kz = complex(to_numpy(m["kz_ref"] if port == "reflection"
                              else m["kz_trn"])[idx])
        kx = complex(to_numpy(m["kx"])[idx])
        ky = complex(to_numpy(m["ky"])[idx])
        kz_inc, kx0, ky0 = m["kz_inc"], m["kx0"], m["ky0"]
        tang = float(abs(ax) ** 2 + abs(ay) ** 2)
        flux = float(np.real(kz / kz_inc)) if kz_inc != 0 else 0.0
        if tang < 1e-300 or flux <= 0.0:     # evanescent / no tangential power
            return 0.0
        az = -(kx * ax + ky * ay) / (kz if abs(kz) > 1e-12 else 1.0)
        inc = np.asarray(incident, dtype=_C).reshape(2)
        ez_inc = (-(kx0 * inc[0] + ky0 * inc[1]) / kz_inc) if kz_inc != 0 else 0.0
        einc_sq = float(abs(inc[0]) ** 2 + abs(inc[1]) ** 2 + abs(ez_inc) ** 2)
        eff = flux * (tang + float(abs(az) ** 2)) / (einc_sq if einc_sq else 1.0)
        return float(np.sqrt(eff / tang)) if eff > 0.0 else 0.0

    def to_jones_field(self, nx, ny, dx, *, incident=(1.0, 0.0),
                       port="reflection", order=None, normalize="power",
                       dy=None):
        """Build a :class:`~lumenairy.elements.polarization.JonesField` from a
        single diffraction order's plane-wave response, for the polarization /
        propagation pipeline.

        Parameters
        ----------
        nx, ny : int
            Output grid shape ``(ny, nx)``.
        dx : float
            Grid pitch [m] (``dy`` defaults to ``dx``).
        incident : (2,) complex, optional
            Incident Jones vector ``(E_x, E_y)``.  Default ``(1, 0)``.
        port : {'reflection', 'transmission'}, optional
            Reflection or transmission side.  Default reflection.
        order : int | (int, int) | None, optional
            Which diffraction order to reconstruct: ``None`` (default) is the
            specular ``(0, 0)`` order, returned as a UNIFORM field (the literal
            specular Jones response); a non-zero order is a TILTED carrier
            ``exp(i (kx_m x + ky_m y))`` ready for :func:`propagate_tilted`.
        normalize : {'power', 'field'}, optional
            For an explicit ``order``: ``'power'`` (default) scales the carrier
            so ``|Ex|^2+|Ey|^2`` equals the order's diffraction EFFICIENCY (the
            energy-correct field that propagates with the right per-order
            power); ``'field'`` deposits the raw tangential boundary amplitude
            (whose ``|.|^2`` is NOT power -- it drops the Poynting flux weight
            and the longitudinal component).  Ignored when ``order is None``.

        Notes
        -----
        Carries one order only; use :meth:`to_multiorder_field` to superpose
        several (a strongly diffracting cell spreads power across orders).
        """
        if port not in ("reflection", "transmission"):
            raise ValueError(
                f"RCWAResult.to_jones_field: port must be 'reflection' or "
                f"'transmission', got {port!r}.")
        if normalize not in ("power", "field"):
            raise ValueError(
                f"RCWAResult.to_jones_field: normalize must be 'power' or "
                f"'field', got {normalize!r}.")
        from .polarization import JonesField
        ny_i, nx_i = int(ny), int(nx)
        if order is None:
            # Specular order as a UNIFORM field (backward-compatible; the
            # literal specular Jones response applied to the incident vector).
            from ..backend import to_numpy
            J = to_numpy(self._Jr if port == "reflection" else self._Jt)
            inc = np.asarray(incident, dtype=_C).reshape(2)
            out = J @ inc
            ex = np.full((ny_i, nx_i), out[0], dtype=_C)
            ey = np.full((ny_i, nx_i), out[1], dtype=_C)
            return JonesField(ex, ey, dx=dx, dy=dy)
        idx = self._order_index(order)
        from ..backend import to_numpy
        kz_o = to_numpy(self._modal["kz_ref"] if port == "reflection"
                        else self._modal["kz_trn"])[idx]
        if np.real(kz_o) <= 1e-12:
            import warnings
            warnings.warn(
                f"RCWAResult.to_jones_field: order {order!r} is evanescent "
                f"(Re(kz) <= 0); its carrier is a non-physical fast-oscillating "
                f"plane wave (and carries no propagating power, so the "
                f"power-normalized field is zero).", stacklevel=2)
        ax, ay = self._order_amplitude(idx, incident, port)
        if normalize == "power":
            s = self._order_power_scale(idx, ax, ay, incident, port)
            ax, ay = s * ax, s * ay
        carrier = self._order_carrier(idx, nx_i, ny_i, float(dx),
                                      float(dx if dy is None else dy))
        return JonesField(ax * carrier, ay * carrier, dx=dx, dy=dy)

    def _order_carrier(self, idx, nx, ny, dx, dy):
        """Unit plane-wave carrier ``exp(i (kx_m x + ky_m y))`` of one order on
        a centred ``(ny, nx)`` grid (k-vectors are stored normalised by k0)."""
        from ..backend import to_numpy
        m = self._require_modal()
        k0 = 2.0 * np.pi / m["wavelength"]
        kx = k0 * float(np.real(to_numpy(m["kx"])[idx]))   # physical [1/m]
        ky = k0 * float(np.real(to_numpy(m["ky"])[idx]))
        xg = (np.arange(nx) - nx // 2) * dx
        yg = (np.arange(ny) - ny // 2) * dy
        X, Y = np.meshgrid(xg, yg)                         # (ny, nx)
        return np.exp(1j * (kx * X + ky * Y)).astype(_C)

    def to_multiorder_field(self, nx, ny, dx, *, incident=(1.0, 0.0),
                            port="reflection", orders=None, normalize="power",
                            filter="none", dy=None):
        """Reconstruct the full diffracted field as a superposition of the
        PROPAGATING diffraction orders -- the bridge a strongly diffracting
        metasurface (deflector / grating coupler / metalens cell) needs, where
        most power is in non-zero orders that :meth:`to_jones_field` drops.

        ``E(x, y) = sum_m A_m exp(i (kx_m x + ky_m y))`` over the requested
        ``orders`` (default: every propagating order, ``Re(kz) > 0``).  Returns
        a :class:`~lumenairy.elements.polarization.JonesField` on a centred
        ``(ny, nx)`` grid of pitch ``dx`` (``dy``).

        ``normalize='power'`` (default) scales each order so the field carries
        the correct per-order diffraction efficiency (energy-conserving:
        ``sum |A_m|^2 == sum efficiencies``, and the dominant order in the
        field matches the dominant efficiency); ``'field'`` deposits the raw
        tangential boundary amplitudes (NOT power-conserving -- see
        :meth:`to_jones_field`).

        ``filter='lanczos'`` multiplies each order by the Lanczos sigma factor
        ``sinc(m/(Mx+1)) sinc(n/(My+1))`` before deposition, suppressing the
        Gibbs ringing a truncated-order reconstruction shows at sharp
        permittivity steps (the high orders are damped smoothly rather than
        cut off).  ``'none'`` (default) leaves the orders unweighted.  Note
        the filter trades a little total power for a much smoother field, so
        it is a visualisation / post-processing aid, not energy-exact.

        Note: the reconstruction is exact only over one unit cell (the field is
        quasi-periodic); evanescent orders are excluded.
        """
        if port not in ("reflection", "transmission"):
            raise ValueError(
                f"RCWAResult.to_multiorder_field: port must be 'reflection' "
                f"or 'transmission', got {port!r}.")
        if normalize not in ("power", "field"):
            raise ValueError(
                f"RCWAResult.to_multiorder_field: normalize must be 'power' "
                f"or 'field', got {normalize!r}.")
        if filter not in ("none", "lanczos"):
            raise ValueError(
                f"RCWAResult.to_multiorder_field: filter must be 'none' or "
                f"'lanczos', got {filter!r}.")
        sigma = self._lanczos_sigma() if filter == "lanczos" else None
        from ..backend import to_numpy
        from .polarization import JonesField
        m = self._require_modal()
        kz = to_numpy(m["kz_ref"] if port == "reflection" else m["kz_trn"])
        if orders is None:
            idxs = [i for i in range(kz.shape[0]) if np.real(kz[i]) > 1e-12]
        else:
            # explicit orders: skip evanescent ones (their carrier would be a
            # bogus fast-oscillating plane wave) with a warning rather than
            # silently depositing garbage.
            idxs = []
            for o in orders:
                i = self._order_index(o)
                if np.real(kz[i]) <= 1e-12:
                    import warnings
                    warnings.warn(
                        f"RCWAResult.to_multiorder_field: order {o!r} is "
                        f"evanescent (Re(kz) <= 0) and is skipped; it carries "
                        f"no propagating power.", stacklevel=2)
                    continue
                idxs.append(i)
        ny_i, nx_i = int(ny), int(nx)
        dy_f = float(dx if dy is None else dy)
        ex = np.zeros((ny_i, nx_i), dtype=_C)
        ey = np.zeros((ny_i, nx_i), dtype=_C)
        for idx in idxs:
            ax, ay = self._order_amplitude(idx, incident, port)
            if normalize == "power":
                s = self._order_power_scale(idx, ax, ay, incident, port)
                ax, ay = s * ax, s * ay
            if sigma is not None:
                w = sigma[idx]
                ax, ay = w * ax, w * ay
            carrier = self._order_carrier(idx, nx_i, ny_i, float(dx), dy_f)
            ex += ax * carrier
            ey += ay * carrier
        return JonesField(ex, ey, dx=dx, dy=dy)

    def _lanczos_sigma(self):
        """Per-order Lanczos sigma factors ``sinc(m/(Mx+1)) sinc(n/(My+1))``
        (1-D: the y-factor is 1), indexed by flat order index.  Damps the
        high orders smoothly to suppress Gibbs ringing in the reconstructed
        real-space field."""
        o = np.asarray(self.orders)
        if o.ndim == 2:
            mx = max(1, int(np.abs(o[:, 0]).max()))
            my = max(1, int(np.abs(o[:, 1]).max()))
            return (np.sinc(o[:, 0] / (mx + 1.0))
                    * np.sinc(o[:, 1] / (my + 1.0)))
        mx = max(1, int(np.abs(o).max()))
        return np.sinc(o / (mx + 1.0))

    # -- in-structure (internal) E/H field reconstruction -----------------

    def _internal_cpm(self, info, i, cinc):
        """Forward amplitude ``c+`` at the TOP of layer ``i`` and the backward
        amplitude ``c-_bot`` at its BOTTOM (the numerically-stable reference) for
        the internal-convention source ``cinc`` (the validated gap-free S-matrix
        recovery).

        The backward field is then ``c-_bot exp(-lam k0 (L - z))`` -- a DECAYING
        exponential -- so a deep, lossy layer never forms the overflowing
        ``exp(+lam k0 z)`` (the old top-referenced ``c- exp(+lam k0 z)`` blew up
        to NaN through high-loss metal layers, silently zeroing
        :meth:`layer_absorption`).  Math-identical: ``c- = X c-_bot`` with
        ``X = exp(-lam k0 L)``."""
        N = info["N"]
        Sa = info["S_above"][i]
        Sb = info["S_below"][i]
        Sa22 = np.asarray(to_numpy(Sa[3]))
        Sa21 = np.asarray(to_numpy(Sa[2]))
        Sb11 = np.asarray(to_numpy(Sb[0]))
        denom = np.linalg.inv(np.eye(2 * N, dtype=_C) - Sa22 @ Sb11)
        cplus = denom @ (Sa21 @ cinc)              # forward amp at TOP of layer i
        # backward amp at the BOTTOM = (reflection below the bottom) @ (forward
        # propagated to the bottom).  X = exp(-lam k0 L) decays (Re(lam) >= 0
        # forward branch), so every exponential here is bounded.
        lam = np.asarray(to_numpy(info["lam"][i]))
        X = np.exp(-lam * info["k0"] * float(info["thick"][i]))
        Sbb11 = np.asarray(to_numpy(info["S_below_bot"][i][0]))
        cminus_bot = Sbb11 @ (X * cplus)
        return cplus, cminus_bot

    def internal_field(self, z, *, component="all", nx=64, ny=None, dx=None,
                       dy=None, layer=None, incident=(1.0, 0.0), filter="none"):
        """Reconstruct the real-space E and/or H field INSIDE the structure
        (audit GAP1) -- the in-layer near field that :meth:`to_multiorder_field`
        (a far-field superposition) cannot show.

        Requires the stack to have been solved with
        ``RCWAStack.solve(retain_internal=True)``.

        Parameters
        ----------
        z : float or array-like
            Depth(s) [m] measured from the TOP of the stack (the superstrate
            interface).  With ``layer=i`` given, ``z`` is instead the LOCAL depth
            inside layer ``i`` (``0`` = its top).
        component : {'E', 'H', 'all'}, optional
            Which field(s) to return (default ``'all'`` -> all six components).
        nx, ny : int, optional
            Real-space grid (``ny`` defaults to 1 for a 1-D stack, else ``nx``).
        dx, dy : float, optional
            Grid pitch [m] (default tiles one unit cell: ``period / n``);
            co-registers with :meth:`to_multiorder_field` when matched.
        layer : int, optional
            Force a specific layer index (then ``z`` is local to that layer);
            default maps each ``z`` to its layer via the cumulative thicknesses.
        incident : (complex, complex), optional
            Incident Jones vector ``(E_x, E_y)`` (default x-polarized).
        filter : {'none', 'lanczos'}, optional
            ``'lanczos'`` damps the high orders (Gibbs suppression at sharp
            permittivity steps) -- a smoothing aid, not energy-exact.

        Returns
        -------
        dict
            ``{'Ex','Ey','Ez','Hx','Hy','Hz'}`` (per ``component``) each a
            ``(nz, ny, nx)`` complex array (``ny`` axis dropped when 1), plus
            ``'z'`` (the depth samples), ``'x'``, ``'y'`` (the grid axes).
            Fields are in the PUBLIC ``exp(-i w t)`` convention.
        """
        info = self._require_modal().get("internal")
        if info is None:
            raise ValueError(
                "RCWAResult.internal_field: the stack was solved without the "
                "internal-field data; call RCWAStack.solve(retain_internal="
                "True).")
        names = {"E": ("Ex", "Ey", "Ez"), "H": ("Hx", "Hy", "Hz"),
                 "all": ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
        if component not in names:
            raise ValueError(
                f"RCWAResult.internal_field: component must be 'E', 'H' or "
                f"'all', got {component!r}.")
        if filter not in ("none", "lanczos"):
            raise ValueError(
                f"RCWAResult.internal_field: filter must be 'none' or "
                f"'lanczos', got {filter!r}.")
        want = names[component]
        N = info["N"]
        Kx = np.asarray(to_numpy(info["Kx"]))
        Ky = np.asarray(to_numpy(info["Ky"]))
        k0 = info["k0"]
        delta = np.asarray(to_numpy(info["delta"]))
        thick = info["thick"]
        z_top = np.concatenate([[0.0], np.cumsum(thick)])
        ex0, ey0 = incident
        cinc = np.concatenate([ex0 * delta, ey0 * delta]).astype(_C)
        sigma = self._lanczos_sigma() if filter == "lanczos" else None

        ny_i = (1 if info["is_1d"] else int(nx)) if ny is None else int(ny)
        nx_i = int(nx)
        dx_f = float(info["period_x"] / nx_i if dx is None else dx)
        dy_f = float(info["period_y"] / ny_i if dy is None else dy)
        xg = (np.arange(nx_i) - nx_i // 2) * dx_f
        yg = (np.arange(ny_i) - ny_i // 2) * dy_f
        # all-order carriers (internal field INCLUDES evanescent orders, unlike
        # the propagating-only far field) -- precompute once and reuse.
        carriers = np.stack([np.asarray(self._order_carrier(idx, nx_i, ny_i,
                                                            dx_f, dy_f))
                             for idx in range(N)])           # (N, ny, nx)

        # cache c+/c- per layer for this incident vector
        cpm = {}

        def _harm(i, zloc):
            if i not in cpm:
                cpm[i] = self._internal_cpm(info, i, cinc)
            cplus, cminus_bot = cpm[i]
            W = np.asarray(to_numpy(info["W"][i]))
            V = np.asarray(to_numpy(info["V"][i]))
            lam = np.asarray(to_numpy(info["lam"][i]))
            EPS = np.asarray(to_numpy(info["EPS"][i]))
            # forward referenced to the layer TOP (depth z), backward to the
            # BOTTOM (L - z): BOTH exponents are <= 0 (Re(lam) >= 0 on the forward
            # branch), so a deep/lossy layer never overflows.  Same field as the
            # old c- exp(+lam k0 z), evaluated in the numerically-stable order.
            Lz = float(thick[i]) - zloc
            fwd = cplus * np.exp(-lam * k0 * zloc)
            bwd = cminus_bot * np.exp(-lam * k0 * Lz)
            s = W @ (fwd + bwd)                              # E tangential
            u = V @ (fwd - bwd)                              # H partner
            Sx, Sy = s[:N], s[N:]
            Hx, Hy = -1j * u[:N], -1j * u[N:]                # -i*eta0 scale
            Sz = np.linalg.solve(EPS, -(Kx @ Hy - Ky @ Hx))  # curl-H z-comp
            Hz = Kx @ Sy - Ky @ Sx                           # curl-E z-comp
            return dict(Ex=Sx, Ey=Sy, Ez=Sz, Hx=Hx, Hy=Hy, Hz=Hz)

        zs = np.atleast_1d(np.asarray(z, dtype=float))
        out = {c: np.empty((zs.shape[0], ny_i, nx_i), dtype=_C) for c in want}
        for zi, zz in enumerate(zs):
            if layer is None:
                i = int(np.clip(np.searchsorted(z_top, zz, side="right") - 1,
                                0, len(thick) - 1))
                zloc = zz - z_top[i]
            else:
                i, zloc = int(layer), zz
            h = _harm(i, zloc)
            for c in want:
                amp = np.conj(h[c])                          # internal -> public
                if sigma is not None:
                    amp = amp * sigma
                out[c][zi] = np.tensordot(amp, carriers, axes=([0], [0]))
        result = {c: (out[c][:, 0, :] if ny_i == 1 else out[c]) for c in want}
        result["z"] = zs
        result["x"] = xg
        result["y"] = yg
        return result

    @staticmethod
    def _cell_grid_index(xg, yg, period_x, period_y, Sx, Sy):
        """Nearest-cell-pixel ``(ix, iy)`` index grids mapping the real-space
        field samples ``(xg, yg)`` onto a ``(Sx, Sy)`` unit-cell sampling."""
        ix = (np.floor((xg % period_x) / period_x * Sx).astype(int)) % Sx
        iy = (np.floor((yg % period_y) / period_y * Sy).astype(int)) % Sy
        return ix, iy

    @staticmethod
    def _layer_im_eps(layer, xg, yg, period_x, period_y):
        """``Im(eps_public(x, y)) >= 0`` of a SCALAR (uniform / isotropic) layer
        on the real-space field grid (the local loss density weight for
        :meth:`layer_absorption`)."""
        ny, nx = len(yg), len(xg)
        if layer.kind == "uniform":
            return np.full((ny, nx), float(np.imag(complex(layer.data))))
        if layer.kind == "iso":
            cell = np.asarray(layer.data)                 # (Sx, Sy), public
            Sx, Sy = cell.shape
            ix, iy = RCWAResult._cell_grid_index(xg, yg, period_x, period_y,
                                                 Sx, Sy)
            return np.imag(cell[np.ix_(ix, iy)]).T          # (ny, nx)
        raise NotImplementedError(
            "RCWAResult.layer_absorption: _layer_im_eps handles only uniform / "
            "isotropic-cell layers; a tensor layer uses _layer_eps_tensor_grid.")

    @staticmethod
    def _layer_eps_tensor_grid(layer, xg, yg, period_x, period_y):
        """Full ``(ny, nx, 3, 3)`` PUBLIC permittivity tensor of a TENSOR layer
        on the real-space field grid (nearest-pixel sampling of the
        ``(Sx, Sy, 3, 3)`` cell) -- the loss operator for the tensor branch of
        :meth:`layer_absorption`."""
        cell = np.asarray(layer.data)                     # (Sx, Sy, 3, 3) public
        Sx, Sy = cell.shape[0], cell.shape[1]
        ix, iy = RCWAResult._cell_grid_index(xg, yg, period_x, period_y, Sx, Sy)
        # gather (nx, ny, 3, 3) then move to (ny, nx, 3, 3)
        g = cell[np.ix_(ix, iy)]                           # (nx, ny, 3, 3)
        return np.transpose(g, (1, 0, 2, 3))               # (ny, nx, 3, 3)

    def layer_absorption(self, *, nx=64, ny=None, nz_per_layer=8):
        """Absorbed power fraction broken down PER LAYER (audit GAP6).

        The total absorptance ``A = 1 - sum(R) - sum(T)`` tells you how much
        power is lost but not WHERE; this attributes it to each layer by
        integrating the local loss density over each layer, normalised so the
        layers sum to the total absorptance (energy-conserving by construction).
        For a scalar (uniform / isotropic) layer the density is
        ``Im(eps) |E|^2``; for an ANISOTROPIC (tensor) layer it is the full
        quadratic form ``Im(E* . eps . E)`` (the rigorous power-loss operator,
        which reduces to ``Im(eps) |E|^2`` for a diagonal medium) -- the
        reconstructed ``(Ex, Ey, Ez)`` is contracted against the local
        ``Im(eps_tensor)``.

        Requires ``RCWAStack.solve(retain_internal=True)``.  Uniform,
        isotropic-cell, and TENSOR (in-plane) layers are supported (raises for
        analytic-shape layers, whose local map is not reconstructed here).

        Returns
        -------
        (2, n_layers) ndarray
            Absorbed fraction in each layer; row 0 for incident ``E_x``, row 1
            for incident ``E_y``.  ``row.sum() == absorptance()[row]``.
        """
        info = self._require_modal().get("internal")
        if info is None:
            raise ValueError(
                "RCWAResult.layer_absorption: the stack was solved without the "
                "internal-field data; call RCWAStack.solve(retain_internal="
                "True).")
        layers = info["layers"]
        thick = info["thick"]
        nlay = len(layers)
        A_tot = self.absorptance()                          # (2,)
        px, py = info["period_x"], info["period_y"]
        ny_i = (1 if info["is_1d"] else int(nx)) if ny is None else int(ny)
        nx_i = int(nx)
        dx, dy = px / nx_i, py / ny_i
        xg = (np.arange(nx_i) - nx_i // 2) * dx
        yg = (np.arange(ny_i) - ny_i // 2) * dy
        # Per-layer local loss operator: a scalar Im(eps) map for uniform /
        # isotropic layers, the full (ny, nx, 3, 3) tensor for anisotropic
        # layers (the loss density is then the quadratic form Im(E* . eps . E)).
        if any(L.kind == "shapes" for L in layers):
            raise NotImplementedError(
                "RCWAResult.layer_absorption: analytic-shape layers are not "
                "supported (their local Im(eps) map is not reconstructed here); "
                "uniform, isotropic-cell, and tensor layers are.")
        im_eps = [None if L.kind == "tensor"
                  else self._layer_im_eps(L, xg, yg, px, py) for L in layers]
        eps_t = [self._layer_eps_tensor_grid(L, xg, yg, px, py)
                 if L.kind == "tensor" else None for L in layers]
        out = np.zeros((2, nlay))
        for p, inc in enumerate(((1.0, 0.0), (0.0, 1.0))):
            raw = np.zeros(nlay)
            for i in range(nlay):
                zl = (np.arange(nz_per_layer) + 0.5) / nz_per_layer * thick[i]
                f = self.internal_field(zl, component="E", nx=nx_i, ny=ny_i,
                                        dx=dx, dy=dy, layer=i, incident=inc)
                ex, ey, ez = f["Ex"], f["Ey"], f["Ez"]
                if ex.ndim == 2:                            # 1-D: (nz, nx)
                    ex, ey, ez = ex[:, None, :], ey[:, None, :], ez[:, None, :]
                if layers[i].kind == "tensor":
                    # density = Im(E* . eps . E) with E = (Ex, Ey, Ez), the
                    # local public tensor T = eps_t[i] broadcast over (nz):
                    # sum_ab conj(E_a) T_ab E_b, then take Im (>= 0 passive).
                    E = np.stack([ex, ey, ez], axis=-1)     # (nz, ny, nx, 3)
                    T = eps_t[i][None]                      # (1, ny, nx, 3, 3)
                    TE = np.einsum("...ab,...b->...a", T, E)  # eps . E
                    dens = np.imag(np.sum(np.conj(E) * TE, axis=-1))
                else:
                    e2 = np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2
                    dens = im_eps[i][None] * e2
                # thickness-weighted cell+depth average of the loss density
                raw[i] = thick[i] * float(np.mean(dens))
            total = raw.sum()
            out[p] = A_tot[p] * raw / total if total > 0 else np.zeros(nlay)
        return out


class _RCWALayer:
    __slots__ = ("thickness", "kind", "data")

    def __init__(self, thickness, kind, data):
        # keep a traced (JAX) thickness native so layer DEPTH is differentiable
        self.thickness = thickness if is_jax_array(thickness) else float(thickness)
        self.kind = kind          # 'uniform' | 'iso' | 'tensor'
        self.data = data


# ---------------------------------------------------------------------------
# RCWAStack.solve(stabilize=True) consensus selector
# (audit AUDIT_RCWA_STACK_RESONANT_CONVERGENCE_2026_06_02).  A sharp-resonant
# metal STACK biases a SINGLE diffraction order (a reflection null)
# non-monotonically at isolated n_orders while total power stays bounded -- the
# isolated-resonance pathology fixed for pmm_efficiency_1d in v5.10.6, here in
# the multilayer S-matrix (near-singular layer<->layer / layer<->region
# mode-match at isolated truncation counts).  Scan a short upward n_orders
# window and keep the solve whose ZERO-ORDER R/T is in the consensus cluster --
# PER-ORDER, not total power, which does not move.
# ---------------------------------------------------------------------------
_STACK_STABILIZE_WINDOW = 5
_STACK_STABILIZE_TOL = 0.02          # 0-order efficiency agreement (spikes >~0.1)


def _zero_order_efficiency(result):
    """The zeroth-order ``(R, T)`` efficiencies for both incident polarizations
    of an ``RCWAResult``, as a length-4 vector -- the per-order quantity the
    stack resonance biases (the reflection null)."""
    o, R, T = result.efficiencies()
    o = np.asarray(to_numpy(o))
    R = np.asarray(to_numpy(R))
    T = np.asarray(to_numpy(T))
    if o.ndim == 1:
        p0 = int(np.where(o == 0)[0][0])
    else:
        p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    return np.concatenate([R[:, p0], T[:, p0]])


def _largest_feature_cluster(feats, tol):
    """Indices of the largest group of feature vectors mutually within ``tol``
    (L-inf) -- the resonance-free consensus; greedy around each anchor."""
    best = []
    for i, fi in enumerate(feats):
        grp = [j for j, fj in enumerate(feats)
               if float(np.max(np.abs(fi - fj))) <= tol]
        if len(grp) > len(best):
            best = grp
    return best


class RCWAStack:
    """Builder + solver for a MULTI-LAYER RCWA stack (1-D or 2-D periodic).

    Compose a stack of uniform spacers and patterned layers (isotropic or
    full-tensor / liquid-crystal) between a superstrate and substrate, set
    the incident plane wave, and solve once for the diffraction efficiencies
    of both incident polarizations plus the zeroth-order Jones reflection.

    Example
    -------
    >>> stack = RCWAStack(period=1.0e-6, n_superstrate=1.0, n_substrate=1.5,
    ...                   n_orders=11)
    >>> stack.add_layer(0.1e-6, eps=2.1 ** 2)          # uniform spacer
    >>> stack.add_layer(0.2e-6, eps_cell=cell)         # 2-D patterned layer
    >>> res = stack.set_source(0.633e-6, theta=0.2).solve()
    >>> orders, R, T = res.efficiencies()

    Parameters
    ----------
    period : float
        Period along x (metres).
    period_y : float, optional
        Period along y for a 2-D (crossed) stack.  If omitted (and
        ``n_orders_y`` is omitted) the stack is 1-D (mono-periodic).
    n_superstrate, n_substrate : complex
        Incidence / transmission half-space indices.
    n_orders : int
        Retained orders per side along x.
    n_orders_y : int, optional
        Retained orders per side along y (2-D only; default = ``n_orders``).
    """

    def __init__(self, period, *, period_y=None, n_superstrate=1.0,
                 n_substrate=1.0, n_orders=11, n_orders_y=None, use_gpu=False):
        _validate_geometry("RCWAStack", period=period, period_y=period_y,
                           n_orders=n_orders, n_orders_y=n_orders_y)
        self.period_x = float(period)
        self.is_1d = period_y is None and n_orders_y is None
        self.period_y = float(period if period_y is None else period_y)
        self.n_superstrate = n_superstrate
        self.n_substrate = n_substrate
        self.nox = int(n_orders)
        self.noy = 0 if self.is_1d else int(
            n_orders if n_orders_y is None else n_orders_y)
        self.use_gpu = bool(use_gpu)
        self._layers: List[_RCWALayer] = []
        self._source: Optional[dict] = None

    def add_layer(self, thickness, *, eps=None, eps_cell=None,
                  eps_tensor_cell=None, shapes=None, eps_background=None):
        """Append a layer.  Provide exactly one layer specification:

        * ``eps`` (scalar) -- uniform spacer;
        * ``eps_cell`` (``(Sx, Sy)``) -- isotropic patterned, FFT-sampled;
        * ``eps_tensor_cell`` (``(Sx, Sy, 3, 3)``) -- anisotropic patterned;
        * ``shapes`` (with ``eps_background``) -- isotropic patterned using
          ANALYTIC shape Fourier transforms + the dual-Laurent factorization
          (exact, no pixelation; see :func:`rcwa_efficiency_2d_shapes`).

        Permittivities are in the PUBLIC convention (``Im(eps) > 0`` lossy).
        """
        n = sum(x is not None for x in (eps, eps_cell, eps_tensor_cell, shapes))
        if n != 1:
            raise ValueError(
                "add_layer: provide exactly one of eps, eps_cell, "
                "eps_tensor_cell, shapes.")
        if not is_jax_array(thickness):       # traced depth -> skip the guard
            _validate_geometry("add_layer", depth=thickness)
        if eps is not None:
            self._layers.append(_RCWALayer(thickness, "uniform", _C(eps)))
        elif eps_cell is not None:
            # Keep a JAX cell native (a np.asarray would materialise the tracer
            # and break the differentiable RCWAStack path); NumPy/list inputs
            # are still normalised to a contiguous complex array.
            cell = (eps_cell.astype(_C) if is_jax_array(eps_cell)
                    else np.asarray(eps_cell, dtype=_C))
            if cell.ndim == 1:
                cell = cell[:, None]
            _validate_cell_sampling("add_layer", cell, self.nox, self.noy)
            self._layers.append(_RCWALayer(thickness, "iso", cell))
        elif shapes is not None:
            if eps_background is None:
                raise ValueError(
                    "add_layer: shapes requires eps_background.")
            _validate_shapes("add_layer", shapes, self.period_x, self.period_y)
            self._layers.append(
                _RCWALayer(thickness, "shapes", (_C(eps_background), shapes)))
        else:
            tcell = (eps_tensor_cell.astype(_C) if is_jax_array(eps_tensor_cell)
                     else np.asarray(eps_tensor_cell, dtype=_C))
            _validate_cell_sampling("add_layer", tcell, self.nox, self.noy)
            _require_inplane_tensor("add_layer", tcell)
            self._layers.append(_RCWALayer(thickness, "tensor", tcell))
        return self

    def add_graded_layer(self, thickness, profile, *, n_slices=8,
                         rule="midpoint"):
        """Append a continuously-graded ``eps(z)`` layer as an auto-sliced
        z-staircase of ``n_slices`` thin layers (audit GAP4 / P4).

        A continuous depth profile (a carrier-accumulation / ENZ layer, a
        thermo-optic or field gradient, a tapered etch) is approximated by a
        staircase; this centralises that slicing so the caller does not
        hand-roll many :meth:`add_layer` calls.

        Parameters
        ----------
        thickness : float
            Total layer thickness (metres).
        profile : callable
            ``profile(zeta) -> permittivity`` at fractional depth
            ``zeta in [0, 1]`` (``0`` = top, ``1`` = bottom).  The return shape
            selects the layer kind: a SCALAR -> uniform spacer; an ``(Sx, Sy)``
            array -> isotropic ``eps_cell``; an ``(Sx, Sy, 3, 3)`` array ->
            anisotropic ``eps_tensor_cell``.
        n_slices : int, optional
            Number of staircase slices (the convergence knob; default 8).
        rule : {'midpoint', 'trapezoid'}, optional
            Sample each slice at its centre (``'midpoint'``, default) or average
            its two edges (``'trapezoid'``).
        """
        n = int(n_slices)
        if n < 1:
            raise ValueError(
                f"add_graded_layer: n_slices must be >= 1, got {n_slices}.")
        if rule not in ("midpoint", "trapezoid"):
            raise ValueError(
                f"add_graded_layer: rule must be 'midpoint' or 'trapezoid', "
                f"got {rule!r}.")
        dz = float(thickness) / n
        for k in range(n):
            if rule == "midpoint":
                eps_k = np.asarray(profile((k + 0.5) / n), dtype=_C)
            else:
                eps_k = 0.5 * (np.asarray(profile(k / n), dtype=_C)
                               + np.asarray(profile((k + 1) / n), dtype=_C))
            if eps_k.ndim == 0:
                self.add_layer(dz, eps=complex(eps_k))
            elif eps_k.ndim == 2:
                self.add_layer(dz, eps_cell=eps_k)
            elif eps_k.ndim == 4:
                self.add_layer(dz, eps_tensor_cell=eps_k)
            else:
                raise ValueError(
                    f"add_graded_layer: profile must return a scalar, an "
                    f"(Sx, Sy) cell, or an (Sx, Sy, 3, 3) tensor cell; got "
                    f"shape {eps_k.shape}.")
        return self

    def add_tapered_grating(self, thickness, *, eps_ridge, eps_groove,
                            duty_bottom, duty_top=None, n_slices=12, n_x=256):
        """Append a 1-D grating with SLANTED (trapezoidal) sidewalls as an
        auto-sliced z-staircase (audit GAP4 -- fab realism).

        The centred ridge's duty cycle varies linearly with depth from
        ``duty_top`` (at the top, ``zeta = 0``) to ``duty_bottom`` (at the
        bottom, ``zeta = 1``); ``duty_top == duty_bottom`` is the usual vertical
        binary grating.  A small sidewall taper can materially change a device,
        so this makes the staircase a one-liner with a documented ``n_slices``
        convergence knob.

        Parameters
        ----------
        thickness : float
            Grating thickness (metres).
        eps_ridge, eps_groove : complex
            Ridge / groove permittivities (PUBLIC ``Im(eps) > 0``).
        duty_bottom : float
            Ridge fraction at the bottom of the grating, in ``[0, 1]``.
        duty_top : float, optional
            Ridge fraction at the top; defaults to ``duty_bottom`` (vertical).
        n_slices : int, optional
            Staircase slice count (default 12).
        n_x : int, optional
            Cell samples along x for each slice's ``eps_cell`` (default 256).
        """
        dt = float(duty_bottom if duty_top is None else duty_top)
        db = float(duty_bottom)
        for d in (db, dt):
            if not (0.0 <= d <= 1.0):
                raise ValueError(
                    f"add_tapered_grating: duty cycles must be in [0, 1], got "
                    f"duty_top={dt}, duty_bottom={db}.")
        er, eg = _C(eps_ridge), _C(eps_groove)
        x = (np.arange(int(n_x)) + 0.5) / int(n_x)
        n_y = max(1, 4 * self.noy + 1)              # grating is uniform in y

        def _profile(zeta):
            duty = dt + (db - dt) * zeta            # top (0) -> bottom (1)
            half = 0.5 * duty
            ridge = np.abs(x - 0.5) < half          # centred ridge
            col = np.where(ridge, er, eg).astype(_C)
            return np.broadcast_to(col[:, None], (int(n_x), n_y)).copy()

        return self.add_graded_layer(thickness, _profile, n_slices=n_slices,
                                     rule="midpoint")

    def set_source(self, wavelength, *, theta=0.0, phi=0.0):
        """Set the incident plane wave (vacuum ``wavelength`` [m], polar
        ``theta`` and azimuth ``phi`` [rad]).

        The stack solver always returns the full zeroth-order Jones response
        (the reaction to both incident ``E_x`` and ``E_y``), so no incident
        polarization is selected here."""
        _validate_geometry("RCWAStack.set_source", wavelength=wavelength)
        self._source = dict(wavelength=float(wavelength), theta=float(theta),
                            phi=float(phi))
        return self

    def _layer_eps_reals(self):
        """Real permittivities present in the layers (for the Wood-anomaly
        grazing nudge)."""
        vals = []
        for L in self._layers:
            if L.kind == "uniform":
                vals.append(float(np.real(_C(L.data))))
            elif L.kind == "iso":
                r = np.real(L.data)
                vals += [float(r.min()), float(r.max())]
            elif L.kind == "shapes":
                bg, shapes = L.data
                vals.append(float(np.real(_C(bg))))
                vals += [float(np.real(_C(s["eps"]))) for s in shapes]
            else:  # tensor
                d = np.real(L.data[:, :, [0, 1, 2], [0, 1, 2]])
                vals += [float(d.min()), float(d.max())]
        return vals

    def _layer_modes(self, layer, Kx, Ky, orders):
        """Layer eigenmodes ``(W, V, lam, EPS)``.  ``EPS`` is the wall-tangential
        ``[[eps]]`` convolution (``EZZ`` for a tensor layer) in the INTERNAL
        convention -- retained for the curl-H ``E_z`` solve in the internal-field
        reconstruction; ignored on the (default) far-field path."""
        xp = array_namespace(Kx)
        if layer.kind == "uniform":
            W, V, kz = _homogeneous_eigenmodes(Kx, Ky, complex(np.conj(layer.data)))
            lam = _sqrt_decay(-xp.concatenate([kz, kz]) ** 2)
            EPS = complex(np.conj(layer.data)) * xp.eye(Kx.shape[0], dtype=_C)
            return W, V, lam, EPS
        if layer.kind == "iso":
            EPS = _eps_convolution_2d(xp.conj(xp.asarray(layer.data)), orders,
                                      self.nox, self.noy)
            W, V, lam = _layer_eigenmodes(Kx, Ky, EPS, EPS)
            return W, V, lam, EPS
        if layer.kind == "shapes":
            # Analytic (host) form factors -> move the convolution to backend.
            eps_bg, shapes = layer.data
            shapes_c = [dict(s, eps=complex(np.conj(_C(s["eps"])))) for s in shapes]
            EPS_np, EPS_inv_np = _analytic_convolutions_2d(
                complex(np.conj(_C(eps_bg))), shapes_c, orders, self.nox,
                self.noy, self.period_x, self.period_y)
            EPS = xp.asarray(EPS_np)
            W, V, lam = _layer_eigenmodes(Kx, Ky, EPS, EPS,
                                          ez_laurent_inv=xp.asarray(EPS_inv_np))
            return W, V, lam, EPS
        et = xp.conj(xp.asarray(layer.data))

        def cv(comp):
            return _eps_convolution_2d(comp, orders, self.nox, self.noy)
        EZZ = cv(et[:, :, 2, 2])
        W, V, lam = _layer_eigenmodes_tensor(
            Kx, Ky, cv(et[:, :, 0, 0]), cv(et[:, :, 0, 1]),
            cv(et[:, :, 1, 0]), cv(et[:, :, 1, 1]), EZZ)
        return W, V, lam, EZZ

    @staticmethod
    def _layer_eig_key(layer):
        """Content key identifying a layer's eigenproblem (the layer's
        permittivity + kind; the layer eig is THICKNESS-INDEPENDENT and, within
        one solve, ``Kx`` / ``Ky`` are shared).  Two layers with the same key
        have identical modes, so a repeated layer (a DBR / Bragg period) is
        solved once.  ``None`` -> not dedupable (e.g. a traced JAX array)."""
        kind = layer.kind
        data = layer.data
        if kind == "uniform":
            return ("uniform", complex(data))
        if kind == "shapes":
            eps_bg, shapes = data
            return ("shapes", complex(eps_bg),
                    tuple(sorted((k, repr(v)) for s in shapes
                                 for k, v in s.items())))
        from ..backend import to_numpy
        try:                                   # iso / tensor: hash the cell
            arr = np.ascontiguousarray(to_numpy(data))
        except Exception:                      # traced array -> no dedup
            return None
        return (kind, arr.shape, str(arr.dtype), arr.tobytes())

    @_with_blas_limit
    def solve(self, *, retain_internal=False, stabilize=False) -> RCWAResult:
        """Solve the stack -> :class:`RCWAResult`.

        ``stabilize=True`` (opt-in; default ``False``) guards against the
        ISOLATED-RESONANCE spikes a sharp-resonant metal MULTILAYER can show: a
        near-singular layer<->layer / layer<->region mode-match at isolated
        ``n_orders`` biases a SINGLE diffraction order (e.g. a reflection null)
        non-monotonically while total power stays bounded -- the same pathology
        fixed for :func:`pmm_efficiency_1d` (v5.10.6).  It re-solves over a short
        ``n_orders`` window at or below the requested count (so the cell sampling,
        sized for ``n_orders``, never aliases) and returns the solve whose
        ZERO-ORDER R/T is in the consensus cluster (the spikes are the outliers;
        the consensus is PER-ORDER because total power does not move).  If no
        plateau forms (genuinely under-resolved, not an isolated spike) it warns
        and returns the requested ``n_orders``.  It costs
        ``_STACK_STABILIZE_WINDOW`` full stack solves, so it is opt-in; clean
        (non-resonant) stacks are unaffected and ``stabilize=False`` is the exact
        single solve.

        ``retain_internal=True`` additionally retains the per-layer modes and the
        cumulative partial S-matrices needed to reconstruct the in-structure E/H
        field via :meth:`RCWAResult.internal_field` (and the per-layer
        absorption via :meth:`RCWAResult.layer_absorption`).  It costs an extra
        ``O(n_layers)`` star-product sweep and keeps the per-layer ``2N x 2N``
        matrices, so it is off by default for hot efficiency sweeps.  NumPy /
        CuPy only."""
        if not stabilize:
            return self._solve_once(retain_internal=retain_internal)
        import warnings
        base_nox, base_noy = self.nox, self.noy

        def _set(nox):
            self.nox = nox
            self.noy = (base_noy if self.is_1d
                        else max(1, base_noy - (base_nox - nox)))

        # DOWNWARD window: distinct n_orders from the requested count down, so the
        # cell sampling (sized for the requested n_orders) NEVER aliases.  The
        # spikes are isolated, so the remaining counts form the consensus cluster.
        window = sorted({max(1, base_nox - b)
                         for b in range(_STACK_STABILIZE_WINDOW)}, reverse=True)
        results, feats = [], []
        try:
            for nox in window:
                _set(nox)
                res = self._solve_once(retain_internal=False)
                results.append(res)
                feats.append(_zero_order_efficiency(res))
        finally:
            self.nox, self.noy = base_nox, base_noy
        cluster = _largest_feature_cluster(feats, _STACK_STABILIZE_TOL)
        if len(cluster) < 2:
            warnings.warn(
                "RCWAStack.solve(stabilize=True): no consensus across the "
                f"n_orders window {window}; the stack is likely UNDER-RESOLVED "
                "(no convergence plateau, not just an isolated spike) -- raise "
                "n_orders / n_orders_y (and the cell sampling).  Returning the "
                "requested n_orders.", stacklevel=2)
            chosen = 0                        # results[0] = requested (highest) n
        else:
            chosen = min(cluster)             # highest-n_orders consensus solve
        if not retain_internal:
            return results[chosen]
        # re-solve the consensus n_orders WITH the internal-field data retained
        try:
            _set(window[chosen])
            return self._solve_once(retain_internal=True)
        finally:
            self.nox, self.noy = base_nox, base_noy

    def _solve_once(self, *, retain_internal=False) -> RCWAResult:
        """Inner single-``n_orders`` stack solve (the public :meth:`solve` body;
        ``stabilize`` scans a window of these).

        ``retain_internal=True`` additionally retains the per-layer modes and the
        cumulative partial S-matrices needed to reconstruct the in-structure E/H
        field via :meth:`RCWAResult.internal_field` (and the per-layer
        absorption via :meth:`RCWAResult.layer_absorption`).  It costs an extra
        ``O(n_layers)`` star-product sweep and keeps the per-layer ``2N x 2N``
        matrices, so it is off by default for hot efficiency sweeps.  NumPy /
        CuPy only."""
        if self._source is None:
            raise ValueError("RCWAStack.solve: call set_source first.")
        if not self._layers:
            raise ValueError("RCWAStack.solve: add at least one layer.")
        src = self._source
        wl, theta, phi = src["wavelength"], src["theta"], src["phi"]
        # Dispatch the backend off the patterned-layer arrays so a JAX cell
        # makes the whole stack solve differentiable (the source geometry --
        # wavelength/angle -- is always concrete here, so only the layer
        # permittivities are traced).
        layer_arrays = [L.data for L in self._layers if L.kind in ("iso",
                                                                    "tensor")]
        xp = _rcwa_xp("RCWAStack.solve", self.use_gpu, *layer_arrays)
        bname = backend_name(xp)
        is_jax = bname == "jax"
        if is_jax:
            _warn_if_jax_f32("RCWAStack.solve")
        orders, N = _harmonic_orders_2d(self.nox, self.noy)
        eps_sup = complex(np.conj(_C(self.n_superstrate) ** 2))
        eps_sub = complex(np.conj(_C(self.n_substrate) ** 2))
        nre = float(np.real(np.sqrt(eps_sup)))
        kx0 = nre * np.sin(theta) * np.cos(phi)
        ky0 = nre * np.sin(theta) * np.sin(phi)
        _require_propagating_incidence("RCWAStack.solve", eps_sup,
                                       kx0 ** 2 + ky0 ** 2)
        # The traced (JAX) layer permittivities cannot feed the concrete grazing
        # nudge, so on the differentiable path the nudge sees only the region
        # indices (the dominant region Rayleigh anomaly is still caught).
        eps_reals = ([eps_sup, eps_sub] if is_jax
                     else [eps_sup, eps_sub] + self._layer_eps_reals())
        wl = _grazing_safe_wavelength(
            wl, kx0, ky0, orders[:, 0], orders[:, 1], self.period_x,
            self.period_y, eps_reals)
        k0 = 2.0 * np.pi / wl
        kx = kx0 + orders[:, 0] * (wl / self.period_x)
        ky = ky0 + orders[:, 1] * (wl / self.period_y)
        Kx = xp.asarray(np.diag(kx.astype(_C)))
        Ky = xp.asarray(np.diag(ky.astype(_C)))
        kxv = xp.asarray(kx.astype(_C))
        kyv = xp.asarray(ky.astype(_C))
        # n_superstrate MUST be part of the key for BOTH region caches: the
        # substrate modes depend on Kx/Ky whose kx0 = Re(sqrt(eps_sup))*... ,
        # so two stacks with the same n_substrate but different n_superstrate
        # have DIFFERENT substrate modes.  Omitting it caused a cache
        # collision (33-40% spurious energy gain on oblique sweeps).  The
        # backend name is in the key too so a NumPy and a CuPy solve of the
        # same geometry never alias to each other's (wrong-device) modes.
        geom = (self.nox, self.noy, wl, theta, phi, self.period_x,
                self.period_y, self.n_superstrate, bname)
        Wref, Vref, kz_ref = _cached_homogeneous_eigenmodes(
            eps_sup, Kx, Ky, ("sup", self.n_superstrate) + geom)
        Wtrn, Vtrn, kz_trn = _cached_homogeneous_eigenmodes(
            eps_sub, Kx, Ky, ("sub", self.n_substrate) + geom)

        # Eig reuse (v5.6): the per-layer modal eig is the dominant cost and is
        # THICKNESS-INDEPENDENT, so two layers with identical permittivity (a
        # repeated DBR / Bragg period, a metamaterial supercell) share one eig
        # instead of recomputing it.  Bit-exact -- it memoises a pure function;
        # a None key (traced JAX array) falls back to per-layer solves.
        _mode_cache = {}
        modes = []
        for L in self._layers:
            key = self._layer_eig_key(L)
            cached = _mode_cache.get(key) if key is not None else None
            if cached is None:
                cached = self._layer_modes(L, Kx, Ky, orders)
                if key is not None:
                    _mode_cache[key] = cached
            modes.append(cached)
        W0, V0, lam0, _e0 = modes[0]
        S = _interface_smatrix(Wref, Vref, W0, V0)
        S = _redheffer_star(S, _propagation_smatrix(lam0, k0 * self._layers[0].thickness))
        for i in range(1, len(modes)):
            Wp, Vp, _lp, _ = modes[i - 1]
            Wc, Vc, lamc, _ = modes[i]
            S = _redheffer_star(S, _interface_smatrix(Wp, Vp, Wc, Vc))
            S = _redheffer_star(S, _propagation_smatrix(lamc, k0 * self._layers[i].thickness))
        Wl, Vl, _ll, _el = modes[-1]
        S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wtrn, Vtrn))
        S11, S12, S21, S22 = S

        p0 = int(np.where((orders[:, 0] == 0) & (orders[:, 1] == 0))[0][0])
        delta = xp.asarray(((orders[:, 0] == 0) & (orders[:, 1] == 0)).astype(_C))
        kz_inc = float(np.real(_sqrt_forward(eps_sup - kx0 ** 2 - ky0 ** 2)))
        safe_r = xp.where(xp.abs(kz_ref) < 1e-12, 1.0, kz_ref)
        safe_t = xp.where(xp.abs(kz_trn) < 1e-12, 1.0, kz_trn)
        R_rows, T_rows, jr_cols, jt_cols = [], [], [], []
        # Per-order tangential field amplitudes (PUBLIC exp(-iwt) convention =
        # conjugate of the internal), kept for the multi-order field bridge.
        rx_rows, ry_rows, tx_rows, ty_rows = [], [], [], []
        for ex0, ey0 in ((1.0, 0.0), (0.0, 1.0)):
            long_inc = kx0 * ex0 + ky0 * ey0
            einc_sq = 1.0 + (long_inc / kz_inc) ** 2 if kz_inc != 0 else 1.0
            cinc = xp.concatenate([ex0 * delta, ey0 * delta])
            r = S11 @ cinc
            t = S21 @ cinc
            rx, ry = r[:N], r[N:]
            tx, ty = t[:N], t[N:]
            rz = -(kxv * rx + kyv * ry) / safe_r
            tz = -(kxv * tx + kyv * ty) / safe_t
            Re = xp.real(kz_ref / kz_inc) * (xp.abs(rx) ** 2 + xp.abs(ry) ** 2
                                             + xp.abs(rz) ** 2) / einc_sq
            Te = xp.real(kz_trn / kz_inc) * (xp.abs(tx) ** 2 + xp.abs(ty) ** 2
                                             + xp.abs(tz) ** 2) / einc_sq
            R_rows.append(xp.where(xp.real(kz_ref) > 0, xp.real(Re), 0.0))
            T_rows.append(xp.where(xp.real(kz_trn) > 0, xp.real(Te), 0.0))
            rx_rows.append(xp.conj(rx))
            ry_rows.append(xp.conj(ry))
            tx_rows.append(xp.conj(tx))
            ty_rows.append(xp.conj(ty))
            jr_cols.append(xp.stack([xp.conj(rx[p0]), xp.conj(ry[p0])]))
            jt_cols.append(xp.stack([xp.conj(tx[p0]), xp.conj(ty[p0])]))
        R = xp.stack(R_rows)
        T = xp.stack(T_rows)
        Jr = xp.stack(jr_cols, axis=1)
        Jt = xp.stack(jt_cols, axis=1)
        out_orders = orders[:, 0].copy() if self.is_1d else orders
        # Modal data for RCWAResult's multi-order field bridge: per-order
        # amplitudes (2, N) for the two incident pols, the transverse
        # wavevectors normalised by k0 (kxv = kx/k0), and the per-order kz/k0
        # in each region (for propagation + the propagating-order mask).
        modal = dict(
            orders2d=orders, p0=p0, wavelength=float(wl),
            rx=xp.stack(rx_rows), ry=xp.stack(ry_rows),
            tx=xp.stack(tx_rows), ty=xp.stack(ty_rows),
            kx=kxv, ky=kyv, kz_ref=kz_ref, kz_trn=kz_trn,
            period_x=self.period_x, period_y=self.period_y,
            # incidence k-vectors (normalised by k0) for the per-order flux
            # weight Re(kz_m/kz_inc) and the incident |E|^2 (einc_sq) that the
            # power-calibrated field reconstruction needs.
            kx0=float(kx0), ky0=float(ky0), kz_inc=float(kz_inc))
        if retain_internal and bname != "jax":
            # Per-layer cumulative partial S-matrices that bracket the TOP plane
            # of each layer -> the c+/c- modal amplitudes for the in-structure
            # field (see RCWAResult.internal_field).  Same gap-free
            # _interface/_propagation/_redheffer sequence as the global solve;
            # star(S_above[0], S_below[0]) reproduces the global S used for r/t.
            S_above, S_below, S_below_bot = self._internal_partials(
                modes, Wref, Vref, Wtrn, Vtrn, k0)
            modal["internal"] = dict(
                W=[m[0] for m in modes], V=[m[1] for m in modes],
                lam=[m[2] for m in modes], EPS=[m[3] for m in modes],
                thick=[float(L.thickness) for L in self._layers],
                S_above=S_above, S_below=S_below, S_below_bot=S_below_bot,
                Kx=Kx, Ky=Ky, k0=float(k0),
                N=N, delta=delta, layers=list(self._layers),
                period_x=self.period_x, period_y=self.period_y,
                is_1d=self.is_1d, nox=self.nox, noy=self.noy)
        if not is_jax:                       # the guard needs concrete R/T
            _check_energy("RCWAStack.solve", R, T)
        return RCWAResult(out_orders, R, T, Jr, Jt, modal=modal)

    def _internal_partials(self, modes, Wref, Vref, Wtrn, Vtrn, k0):
        """Cumulative ``(S_above, S_below)`` partial S-matrices bracketing the
        TOP of each layer (the recovery basis for the internal field).

        ``S_above[i]`` = star product superstrate -> TOP of layer ``i`` (through
        the interface INTO layer ``i``, before its own propagation);
        ``S_below[i]`` = star product TOP of layer ``i`` -> substrate (its own
        propagation first).  ``star(S_above[0], S_below[0])`` reproduces the
        global S-matrix."""
        nlay = len(modes)
        ifc = [_interface_smatrix(Wref, Vref, modes[0][0], modes[0][1])]
        for i in range(1, nlay):
            ifc.append(_interface_smatrix(modes[i - 1][0], modes[i - 1][1],
                                          modes[i][0], modes[i][1]))
        ifc.append(_interface_smatrix(modes[-1][0], modes[-1][1], Wtrn, Vtrn))
        prop = [_propagation_smatrix(modes[i][2], k0 * self._layers[i].thickness)
                for i in range(nlay)]
        S_above = [None] * nlay
        S_above[0] = ifc[0]
        for i in range(1, nlay):
            S_above[i] = _redheffer_star(
                _redheffer_star(S_above[i - 1], prop[i - 1]), ifc[i])
        S_below = [None] * nlay
        # S_below_bot[i] = star product from the BOTTOM of layer i to the
        # substrate (i.e. S_below[i] with layer i's own propagation removed from
        # the top).  Its S11 is the reflection seen looking DOWN from the bottom
        # of layer i; it lets the internal-field recovery reference the backward
        # mode to the layer BOTTOM with a DECAYING exponential (no exp(+lam k0 z)
        # overflow through a deep/lossy layer -- see RCWAResult._internal_cpm).
        S_below_bot = [None] * nlay
        for i in range(nlay - 1, -1, -1):
            below_bot = (ifc[nlay] if i == nlay - 1
                         else _redheffer_star(ifc[i + 1], S_below[i + 1]))
            S_below_bot[i] = below_bot
            S_below[i] = _redheffer_star(prop[i], below_bot)
        return S_above, S_below, S_below_bot


# Register the RCWA caches with the library cache registry (so the global
# "clear all caches" path empties them too).  Canonical v4.16.0 enrollment
# pattern (mirrors propagators/propagation.py).
try:
    import sys as _sys

    from .._cache_registry import register_cache_clearer as _register_cache_clearer
    _this_mod = _sys.modules[__name__]
    _register_cache_clearer(
        "rcwa_homogeneous_modes",
        lambda: getattr(_this_mod, "_clear_rcwa_caches")(),
    )
except ImportError:  # pragma: no cover - registry always present in-tree
    pass
