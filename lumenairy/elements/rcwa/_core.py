"""RCWA shared core: BLAS controls, validation, S-matrix / Redheffer
algebra, eigenmodes, convolutions, and dimension-agnostic public utilities.
NOT a public import surface -- use ``lumenairy.elements.rcwa``."""
from __future__ import annotations

import contextlib
import functools
import threading
import warnings
from collections import OrderedDict
from typing import Optional

import numpy as np

from ...backend import (
    CUPY_AVAILABLE,
    JAX_AVAILABLE,
    array_namespace,
    backend_name,
    is_cupy_array,
    is_jax_array,
    to_numpy,
)

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
        # Audit P2 2026-06-10: a PARTIAL CUDA install (cupy wheel present but
        # cublas/cufft/cusolver runtime wheels missing) passes the import
        # check and then dies deep inside the first fft2 with a raw
        # 'DLL load failed while importing cufft'.  Probe a trivial device op
        # once and re-raise as the same friendly RuntimeError.
        try:
            cp.zeros(1).sum()
            from cupy.fft import fft as _probe_fft  # noqa: F401
        except (ImportError, OSError, RuntimeError) as e:
            raise RuntimeError(
                f"{fn_name}: use_gpu=True and CuPy is installed, but the CUDA "
                f"runtime libraries are unusable ({type(e).__name__}: {e}).  "
                f"Install the matching NVIDIA wheels (e.g. `pip install "
                f"nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-cusolver-cu12`) "
                f"or call with use_gpu=False for the NumPy path.") from e
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



class Efficiency2D(tuple):
    """A 2-D diffraction-efficiency result that unpacks EXACTLY like the legacy
    ``(orders, R, T)`` tuple but ALSO carries ``.dof`` -- the modal eigenproblem
    dimension (``2 * n_retained_orders`` for RCWA, ``2 * Nf`` / ``2 * q^2`` for the
    PMM hybrid / staggered solvers).  ``dof`` is the cross-suite degrees-of-freedom
    cost metric for an accuracy-vs-cost comparison (the PMM win is matched accuracy
    at smaller ``dof``).  Being a ``tuple`` subclass, ``o, R, T = result``,
    ``result[i]`` and ``isinstance(result, tuple)`` all behave as before -- only the
    extra ``.dof`` attribute is new (the cross-suite return-shape unification)."""

    def __new__(cls, orders, R, T, dof):
        self = super().__new__(cls, (orders, R, T))
        self.dof = int(dof)
        return self

    def __repr__(self):
        return (f"Efficiency2D(orders=<{len(self[0])}>, R=..., T=..., "
                f"dof={self.dof})")



class _EnergyError(ValueError):
    """Raised by :func:`_check_energy` when a passive solve returns
    non-physical ``sum(R)+sum(T) >> 1``.  A subclass of ``ValueError`` (so
    existing ``except ValueError`` handlers are unaffected) that the
    ``stabilize=`` retry path can catch specifically."""


class _EnergyWarning(UserWarning):
    """Emitted by :func:`_check_energy` when a PROVABLY LOSSLESS solve
    violates the exact closure ``sum(R)+sum(T) = 1`` beyond ~1e-6 but below
    the hard 5% tripwire (the audited 'silent window': the per-order answers
    there are wrong even though no error is raised).  The ``stabilize=``
    retry ladders treat a recorded ``_EnergyWarning`` as a failed attempt."""



def _check_energy(fn_name, R, T, lossless=False):
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
    # to_numpy (not np.asarray) so the CuPy/GPU path does not raise on the implicit
    # device->host conversion -- _check_energy is the lone site that bypassed it.
    tot = float(np.real(np.sum(to_numpy(R))) + np.real(np.sum(to_numpy(T))))
    n_states = int(R.shape[0]) if getattr(R, "ndim", 1) == 2 else 1
    # NaN/inf propagates PAST a one-sided '>' comparison (audit P3: a NaN
    # substrate index silently returned sumR = nan) -- raise loudly.
    if not np.isfinite(tot):
        raise _EnergyError(
            f"{fn_name}: non-finite total efficiency (sum R+T = {tot}); a "
            f"NaN/inf material index or permittivity reached the solve "
            f"(check the region indices and the cell values).")
    if tot > n_states * 1.05:
        raise _EnergyError(
            f"{fn_name}: energy non-conservation detected (sum R+T = "
            f"{tot:.3e} exceeds {n_states}); the solve is numerically unstable "
            f"at this geometry (a near-degenerate layer<->region mode-match at "
            f"a measure-zero period / n_orders coincidence, common at very "
            f"large period / low index contrast).  Pass stabilize=True to "
            f"auto-retry at a slightly higher n_orders, or reduce n_orders, "
            f"adjust the period, or increase the index contrast.")
    # Two-sided (audit P1 2026-06-10): a NEGATIVE total is just as
    # non-physical as an excessive one (the gain-superstrate kz_inc flip
    # returned sum T = -392 below the one-sided tripwire).
    if tot < -1e-9 * n_states:
        raise _EnergyError(
            f"{fn_name}: NEGATIVE total efficiency (sum R+T = {tot:.3e}); "
            f"the efficiency normalisation is non-physical (e.g. a gain or "
            f"non-propagating incidence medium slipped past the entry "
            f"guards).")
    # Lossless-closure tripwire (audit P1 2026-06-10): for a PROVABLY
    # lossless input (every region / structure permittivity exactly real)
    # the closure R+T = 1 is exact in this code (clean solves hold it to
    # <1e-11), so a violation in the silent window 1e-6..0.05 means the
    # per-order answers are wrong (measured: +3.3e-2 closure error carried
    # an 8% per-order error and broken +/-1 symmetry).  WARN here (raising
    # would break shipped behaviour); the stabilize= retry ladders treat
    # this warning as a failed attempt and move to the next truncation.
    if lossless and abs(tot - n_states) > 1e-6 * n_states:
        warnings.warn(_EnergyWarning(
            f"{fn_name}: lossless energy closure violated (sum R+T - "
            f"{n_states} = {tot - n_states:+.3e}, structure is provably "
            f"lossless): the truncation is numerically unstable here and "
            f"the PER-ORDER efficiencies are suspect.  Pass stabilize=True "
            f"(retries nearby truncations) or change n_orders."),
            stacklevel=3)



def _require_jax_x64(fn_name):
    """Require JAX double precision on the JAX (differentiable) path.  RCWA/PMM's
    eigenproblem is ill-conditioned in single precision (cond ~1e13), and JAX
    silently truncates the requested ``complex128`` to ``complex64`` unless
    ``jax_enable_x64`` is set -- yielding quietly WRONG efficiencies and gradients
    in the advertised differentiable regime, with the runtime energy tripwire
    (:func:`_check_energy`) skipped on the JAX path.  A suppressible warning is not
    enough for a correctness-critical precision requirement, so RAISE.

    (The rest of the library auto-promotes via ``jax.config.update`` -- e.g.
    :func:`fft_infra._resolve_jax_complex_dtype` -- but that global mutation is
    unsafe mid-trace when the caller jits the whole solve, and enabling x64 is a
    one-line caller setup.  Call before any ``complex128`` allocation.)"""
    import jax
    try:
        enabled = bool(jax.config.read("jax_enable_x64"))
    except Exception:
        enabled = bool(getattr(jax.config, "jax_enable_x64", False))
    if not enabled:
        raise RuntimeError(
            f"{fn_name}: the JAX (differentiable) path requires double precision, "
            f"but jax_enable_x64 is disabled -- JAX would silently truncate "
            f"complex128 to complex64 and the eigenproblem is ill-conditioned in "
            f"single precision (cond ~1e13), giving quietly wrong efficiencies / "
            f"gradients.  Enable it once at import: "
            f"jax.config.update('jax_enable_x64', True).")



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


def _forward_flux_kz(eps_region, kx, ky):
    """PUBLIC-convention forward ``kz`` (``Re(kz) >= 0``) for a homogeneous
    half-space, used by the z-FLUX weight, the propagating-order mask, and the
    longitudinal field ``Ez = -(kx Ex + ky Ey)/kz``.

    The internal ``exp(+i w t)`` loss bridge conjugates the region permittivity,
    and :func:`_sqrt_forward` (the ``Im(kz) >= 0`` branch) then returns
    ``Re(kz) < 0`` for a LOSSY half-space (a 4th-quadrant conjugated ``eps``).  The
    ``Re(kz) > 0`` propagating mask read that as evanescent and SILENTLY ZEROED the
    transmitted efficiency into ANY absorbing exit substrate (a long-standing
    energy-corruption bug; the reflected side and the mode-match path are
    unaffected).  Un-conjugating ``eps`` here restores ``Re(kz) >= 0`` for a forward
    wave into a lossy medium so it carries its physical z-flux; for LOSSLESS (real)
    ``eps`` the conjugate is the identity, so this is byte-for-bit unchanged.

    Use this for the FLUX/mask/Ez only -- the modal eigenmode ``kz`` keeps the
    internal convention (that path matches at every interface and is correct).

    SCOPE (adversarial verification 2026-06-09, ~526 oracle-checked configs):
    correct for a lossy EXIT substrate (``Im(eps) > 0``) -- T, R and the absorbed
    fraction all match an independent TMM to ~1e-15.  Two regimes are out of
    scope by construction: (1) a lossy INCIDENCE medium (lossy superstrate) makes
    per-wave ``R + T != 1`` because each wave is normalized by its own
    ``|amp|^2`` z-flux while the incident/reflected cross-term carries net flux --
    the per-order value still matches analytic Fresnel exactly, so R/T are only
    physically meaningful for a LOSSLESS incidence medium; (2) GAIN media
    (``Im(n) < 0``) fall in the opposite eps quadrant and trip the passive
    :func:`_check_energy` tripwire (a LOUD raise, never a silent wrong answer) --
    active media are unsupported."""
    xp = array_namespace(kx)
    return _sqrt_forward(xp.conj(xp.asarray(eps_region).astype(_C))
                         - kx ** 2 - ky ** 2)


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
    evanescent / metallic incidence media.

    Also rejects a GAIN superstrate (public ``Im(n_superstrate) < 0``, i.e.
    INTERNAL ``Im(eps_sup) > 0`` after the loss-convention bridge) -- audit
    P1 2026-06-10: even ``Im(n_sup) = -1e-9`` flips ``_sqrt_forward`` to its
    ``Re < 0`` root, so ``kz_inc < 0`` silently negated every efficiency
    (TE ``sum T = -11.7``, TM ``-392.8`` on a plain lossless grating) while
    the reflected orders were masked to zero -- a discontinuity at
    ``Im(n_sup) = 0^-`` invisible to the one-sided energy tripwire.  An
    infinitesimally LOSSY superstrate remains continuous and supported."""
    if float(np.imag(eps_sup)) > 0.0:
        raise ValueError(
            f"{fn_name}: gain incidence medium (Im(n_superstrate) < 0; "
            f"public eps_superstrate = "
            f"{complex(np.conj(complex(eps_sup))):.6g}) is not supported: "
            f"the forward-root convention flips kz_inc negative and would "
            f"silently negate every efficiency.  Use a lossless or lossy "
            f"(Im(n_superstrate) >= 0) incidence medium.")
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
        eps_sh = sh.get("eps")
        if eps_sh is not None and abs(_C(eps_sh)) < 1e-12:
            raise ValueError(
                f"{fn_name}: shapes[{i}] ({kind}) has eps ~ 0 ({eps_sh!r}); a "
                f"zero permittivity blows up the averaged-eps / inverse-rule "
                f"convolution (inf/NaN) -- use a small non-zero eps.")
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



# ===========================================================================
# Layer eigen-solve (vectorial 2N system, Rumpf/Moharam)
# ===========================================================================

def _layer_Q_matrix(Kx, Ky, EPS, EPS_normal):
    """The ``Q`` block (``dE/dz' = Q H``) of the layer ODE system.

    ``EPS`` is the Laurent ``[[eps]]`` (used where E is tangential to the
    grating walls -- the ``E_y`` response); ``EPS_normal`` is the convolution
    for the wall-NORMAL field ``E_x``, which is ``[[eps]]`` for the Laurent
    rule but the Li inverse-rule ``[[1/eps]]^{-1}`` for fast TM / metal
    convergence.  Shared by the structured-layer eig solve and the analytic
    uniform-layer path so the magnetic eigenvectors ``V = Q W diag(1/lam)``
    use one convention everywhere (essential for evanescent-order interface
    consistency).
    """
    xp = array_namespace(Kx, Ky, EPS, EPS_normal)
    return _block(xp, [
        [Kx @ Ky,           EPS - Kx @ Kx],
        [Ky @ Ky - EPS_normal,  -Ky @ Kx],
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



def _symmetric_solve_rt(Vref, Vtrn, Kx, Ky, EPS, EPS_normal, ez_inv,
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
    EPS_normal = _recentre(EPS_normal)
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
    Q = _layer_Q_matrix(Kx, Ky, EPS, EPS_normal)
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
    S = _propagation_star(S, lam_e, k0 * depth)
    S = _redheffer_star(S, _interface_smatrix(Wl_e, Vl_e, Ireg_e, Vtrn_e))
    S11, _S12, S21, _S22 = S
    cinc_e = _even_project(cinc, desc, xp)
    r = _even_unfold(S11 @ cinc_e, desc, xp)
    t = _even_unfold(S21 @ cinc_e, desc, xp)
    return r, t



def _tensor_PQ(Kx, Ky, Cxx, Cxy, Cyx, Cyy, EZZ, xp):
    """The in-plane tensor layer's first-order blocks ``(P, Q)`` -- the exact
    construction :func:`_layer_eigenmodes_tensor` eigendecomposes, exposed so
    the even-parity machinery can fold them (backlog A1, 2026-06-10)."""
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
    return P, Q


def _scalar_PQ(Kx, Ky, EPS, EPS_normal, ez_inv, xp):
    """The scalar layer's ``(P, Q)`` blocks (the :func:`_layer_eigenmodes`
    structured-branch construction)."""
    N = Kx.shape[0]
    I = xp.eye(N, dtype=_C)
    EPS_inv = ez_inv if ez_inv is not None else xp.linalg.inv(EPS)
    P = _block(xp, [
        [Kx @ EPS_inv @ Ky,        I - Kx @ EPS_inv @ Kx],
        [Ky @ EPS_inv @ Ky - I,    -Ky @ EPS_inv @ Kx],
    ])
    Q = _layer_Q_matrix(Kx, Ky, EPS, EPS_normal)
    return P, Q


def _symmetric_cascade_rt(Vref, Vtrn, Kx, Ky, layer_specs, depths, k0,
                          cincs, orders, xp):
    """GENERALIZED even-parity-sector S-matrix cascade (backlog A1,
    2026-06-10) -- the multi-layer / tensor extension of
    :func:`_symmetric_solve_rt`.

    ``layer_specs[i]`` is either ``("uniform", eps0)`` or
    ``("PQ", P, Q, probe)`` where ``probe`` is an N-space convolution matrix
    used for symmetry-centre detection (the layer's primary permittivity
    convolution).  All layers must share ONE symmetry centre (detected from
    the first structured layer; per-layer flip-invariance checks catch
    mismatches -> ``None`` -> the caller falls back to the full solve).
    ``cincs`` is a list of full-space source vectors (each must be purely
    even -- the (0, 0)-order drive is); returns ``[(r, t), ...]`` per source
    or ``None`` when any precondition fails.

    Same contract as the single-layer path: results match the full solve to
    ~1e-12 (a different, even-adapted basis -- NOT bit-identical; see
    docs/TOLERANCE_POLICY.md), and the recentering gauge is a per-order
    phase, so every |r|^2 / |t|^2 efficiency is gauge-invariant.
    """
    flip = _order_flip_perm(Kx, Ky)
    if flip is None:
        return None
    n = flip.shape[0]
    probe0 = next((sp[3] for sp in layer_specs if sp[0] == "PQ"), None)
    if probe0 is None:
        return None                       # all-uniform: analytic path is better
    d = _recentering_phase(probe0, orders, xp)
    if d is None:
        return None
    d2 = xp.concatenate([d, d])
    d2inv = 1.0 / d2
    flip2 = np.concatenate([flip, flip + n])
    desc = _even_basis_desc(flip)
    ne = desc[0].shape[0]
    Ireg_e = xp.eye(ne, dtype=_C)
    Vref_e = _even_fold(Vref, desc, xp)
    Vtrn_e = _even_fold(Vtrn, desc, xp)
    kxd = xp.diag(Kx)
    kyd = xp.diag(Ky)
    i0 = np.asarray(desc[0])

    # per-layer even modes
    modes_e = []
    for sp in layer_specs:
        if sp[0] == "uniform":
            eps0 = _C(sp[1])
            kz = _sqrt_forward(eps0 - kxd ** 2 - kyd ** 2)
            lam_full = _sqrt_decay(-xp.concatenate([kz, kz]) ** 2)
            lam_e = lam_full[xp.asarray(i0)]      # pairs share kz(k)=kz(-k)
            EPSu = eps0 * xp.eye(n, dtype=_C)
            Qu = _layer_Q_matrix(Kx, Ky, EPSu, EPSu)
            Vl_e = _even_fold(Qu, desc, xp) * _inv_lam(lam_e)[None, :]
            modes_e.append((Ireg_e, Vl_e, lam_e))
            continue
        _kind, P, Q, _probe = sp
        Pr = (d2inv[:, None] * P) * d2[None, :]
        Qr = (d2inv[:, None] * Q) * d2[None, :]
        M = Pr @ Qr
        Mh = to_numpy(M)
        resid = np.max(np.abs(Mh[np.ix_(flip2, flip2)] - Mh))
        scale = float(max(np.max(np.abs(np.diagonal(Mh))), 1.0))
        if resid > 1e-10 * scale:
            return None                   # not jointly centro-symmetric
        Mp = _even_fold(M, desc, xp)
        lam2_e, Wl_e = _eig_for(xp)(Mp)
        lam_e = _sqrt_decay(lam2_e)
        Q_e = _even_fold(Qr, desc, xp)
        Vl_e = Q_e @ Wl_e @ xp.diag(_inv_lam(lam_e))
        modes_e.append((Wl_e, Vl_e, lam_e))

    # even-sector Redheffer recursion: sup | L1 ... Ln | sub
    W0, V0, lam0 = modes_e[0]
    S = _interface_smatrix(Ireg_e, Vref_e, W0, V0)
    S = _propagation_star(S, lam0, k0 * depths[0])
    for i in range(1, len(modes_e)):
        Wp, Vp, _lp = modes_e[i - 1]
        Wc, Vc, lamc = modes_e[i]
        S = _redheffer_star(S, _interface_smatrix(Wp, Vp, Wc, Vc))
        S = _propagation_star(S, lamc, k0 * depths[i])
    Wl, Vl, _ll = modes_e[-1]
    S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Ireg_e, Vtrn_e))
    S11, _S12, S21, _S22 = S

    out = []
    for cinc in cincs:
        ce = _even_project(cinc, desc, xp)
        out.append((_even_unfold(S11 @ ce, desc, xp),
                    _even_unfold(S21 @ ce, desc, xp)))
    return out


def _layer_eigenmodes(Kx, Ky, EPS, EPS_normal, ez_laurent_inv=None):
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
    inverted), and the ``Q`` block uses ``EPS_normal`` for the wall-normal
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
    xp = array_namespace(Kx, Ky, EPS, EPS_normal)
    Kx = xp.asarray(Kx).astype(_C)
    Ky = xp.asarray(Ky).astype(_C)
    N = Kx.shape[0]
    I = xp.eye(N, dtype=_C)
    Q = _layer_Q_matrix(Kx, Ky, EPS, EPS_normal)
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

    Returns ``(W, V, kz)`` -- NOTE the 3rd slot is the per-order longitudinal
    ``kz`` (vacuum-normalised), NOT the modal eigenvalue ``lam`` that
    :func:`_layer_eigenmodes` returns in its 3rd slot.  A half-space order is its
    own eigenmode, so the caller wants ``kz`` directly for the Rayleigh phase and
    z-flux (``lam = sqrt_decay(-kz^2)`` is used only to build ``V``).
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
    # Layer-PROPAGATION S-matrices have S11 = S22 = 0 (see _propagation_smatrix),
    # and the recursion stars one at every layer -- so ~half the stars feed a zero
    # A22 or B11 here.  When either is the exact zero block, (I - B11 @ A22) and
    # (I - A22 @ B11) are both exactly I, whose inverse is byte-identically I
    # (verified: max|inv(I)-I| == 0).  Substituting the literal I is therefore
    # bit-for-bit identical to inverting it, so skip the two dense 2N inverses.
    # The JAX backend SKIPS this optimization entirely (it is a NumPy/CuPy-only
    # fast path): bool(.any()) cannot be evaluated on a jit tracer, so a JAX
    # array always takes the inverse branch below -- the SAME (unchanged) result,
    # just without the zero-block shortcut.  The is_jax_array guards are a backend
    # test, NOT the zero test; only the concrete .any() calls test for a zero
    # block.  (is_jax_array, not the scalar-only _is_traced whose complex()
    # coercion always raises on a matrix.)
    if (not is_jax_array(A22) and not is_jax_array(B11)
            and (not bool(A22.any()) or not bool(B11.any()))):
        D = I
        F = I
    else:
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



def _propagation_star(S, lam, k0_L):
    """``S`` Redheffer-starred with the pure-propagation S-matrix of
    ``(lam, k0_L)``, exploiting its structure (diagonal off-blocks, ZERO
    diagonal blocks): ``P11 = 0`` collapses both star inverses to the
    identity, so the whole product reduces to row/column scaling by
    ``X = exp(-lam k0 L)`` -- algebraically identical to
    ``_redheffer_star(S, _propagation_smatrix(lam, k0_L))`` but without the
    ~10-zgemm chain against literal identity/zero blocks (audit RCWA-LEV-2:
    463.5 ms -> 7.3 ms at 2N = 722; one of the two stars in every
    single-layer solve and half the stars in every stack solve)."""
    A11, A12, A21, A22 = S
    xp = array_namespace(A11, lam)
    X = xp.exp(-lam * k0_L)
    return (A11, A12 * X[None, :], X[:, None] * A21,
            (X[:, None] * A22) * X[None, :])



def _propagation_star_general(S, lam_f, lam_b, k0_L):
    """Diagonal-aware star against the GENERALIZED propagation S-matrix
    (explicit forward ``exp(-lam_f k0 L)`` / backward ``exp(+lam_b k0 L)``
    factors) -- the :func:`_propagation_star` of the full-3x3 cascade."""
    A11, A12, A21, A22 = S
    xp = array_namespace(A11, lam_f)
    Xf = xp.exp(-lam_f * k0_L)
    Xb = xp.exp(lam_b * k0_L)
    return (A11, A12 * Xb[None, :], Xf[:, None] * A21,
            (Xf[:, None] * A22) * Xb[None, :])



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
        warnings.warn(
            f"rcwa_convergence: RCWAStack NOT converged at {no_lo} -- the "
            f"per-order efficiency changed by {delta:.2e} (> atol={atol:.1e}) "
            f"going to {no_hi}; the lower-order result may be unreliable. "
            f"Increase n_orders.", stacklevel=3)
    return high, report



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
        # Divide ONLY where the denominator is safe (a flattened sequence drives
        # den -> 0; the mask falls back to s_k there).  Computing num/den for every
        # element first leaked a spurious "invalid value encountered in divide"
        # RuntimeWarning (audit P3) -- the returned value was already correct.
        den_safe = np.abs(den) >= 1e-15 * (np.abs(v[1:-1]) + 1e-30)
        ratio = np.divide(num, den, out=np.full_like(num, np.nan), where=den_safe)
        acc = np.where(den_safe, ratio, v[1:-1])
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

    Forward = (carries net z-flux, ``|Sz| > flux_tol``): ``Sz > 0``; (flux-null /
    evanescent): ``Re(gam) > 0`` (decaying as ``exp(-lam k0 z)``).  The split is by
    FLUX MAGNITUDE rather than ``Re(gam)`` sign, so it is gauge-robust AND correct
    for NON-RECIPROCAL / gyrotropic media (whose ``lam`` are neither purely real nor
    purely imaginary -- a ``Re(gam)`` split misclassifies them and yields a
    rank-deficient forward set / Singular matrix).  For reciprocal media the two
    criteria coincide.

    A defensive rebalance keeps the count at exactly ``2N`` if a near-zero-flux
    propagating mode (``|Sz| < flux_tol``) would otherwise tip the split: the
    excess/deficit is resolved by the signed flux (most-forward kept)."""
    xp = array_namespace(Vfull)
    n = gam.shape[0]
    gre = xp.real(gam)
    Ex = Vfull[:N, :]
    Ey = Vfull[N:2 * N, :]
    Hx = Vfull[2 * N:3 * N, :] / 1j
    Hy = Vfull[3 * N:4 * N, :] / 1j
    Sz = xp.real(xp.sum(Ex * xp.conj(Hy) - Ey * xp.conj(Hx), axis=0))   # (n,)
    # Classify by FLUX MAGNITUDE, not Re(gam) sign (audit P2-A): a mode carrying
    # significant net z-flux is forward iff Sz > 0 -- robust for NON-RECIPROCAL /
    # gyrotropic media whose lam are neither purely real (propagating) nor purely
    # imaginary (evanescent).  The old |Re(gam)|<tol split shunted such modes into
    # the Re-sign branch and produced a rank-deficient forward set (Singular matrix
    # on a uniform non-reciprocal layer).  A flux-NULL (truly evanescent) mode
    # falls back to the decay sign Re(gam) > 0.  For RECIPROCAL media the two
    # criteria coincide (propagating modes carry flux, evanescent ones are
    # flux-null), so the Berreman-validated path is unchanged.
    mx = float(xp.maximum(xp.asarray(1.0), xp.max(xp.abs(Sz))))
    flux_tol = 1e-9 * mx
    carries = xp.abs(Sz) > flux_tol
    # DEEP-DECAY override (v5.14, PMM-2D out-of-plane): in a PROJECTED
    # (non-orthogonal, truncated) modal basis the deep-evanescent modes carry
    # PROJECTION-NOISE flux far above the 1e-9 tolerance (measured up to
    # ~7e-4 of max on the 2-D PMM generator), and the noise SIGN is random --
    # one growing mode classified forward blows the cascade up by exp(+|Re
    # gam}| k0 L) ~ 1e30.  A mode whose flux is below the noise ceiling while
    # decaying decisively is physically flux-null: trust the decay sign.
    # Genuine flux carriers (lossless propagating ~O(1) rel, lossy-metal
    # forward ~0.2 rel, the gyrotropic audit-P2-A modes) sit far above the
    # ceiling and are untouched, so the Berreman-validated rcwa full-3x3
    # behavior is preserved.
    deep_noise = (xp.abs(Sz) < 3e-3 * mx) & (xp.abs(gre) > 0.1)
    # STABILITY band: a strongly-decaying mode (|Re gam| > 0.5, i.e. e^{-pi}
    # per period-depth) must be classified by its decay sign REGARDLESS of
    # flux -- keeping a growing mode in the forward set makes the cascade's
    # exp(+|Re gam| k0 L) unbounded no matter what its (possibly genuine,
    # non-reciprocal) flux says.  The gyrotropic modes the flux-first rule
    # protects (audit P2-A) are near-propagating (|Re gam| small) and stay
    # flux-classified.
    deep = xp.abs(gre) > 0.5
    carries = carries & ~deep_noise & ~deep
    fwd = xp.where(carries, Sz > 0, gre > 0)
    idx = xp.asarray(np.where(to_numpy(fwd))[0])
    if int(idx.shape[0]) == 2 * N:
        return idx
    # ---- defensive rebalance to EXACTLY 2N (near-zero-flux / cut tie) --------
    Sz_np = np.asarray(to_numpy(Sz))
    gre_np = np.asarray(to_numpy(gre))
    carries_np = np.asarray(to_numpy(carries))
    # Rank all modes by a signed "forwardness" score: flux-carrying by Sz,
    # flux-null (evanescent) by Re(gam).  The 2N largest scores are forward.
    score = np.where(carries_np, Sz_np, gre_np)
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

    Full-3x3 (out-of-plane) solvers now exist on BOTH the 1-D NumPy/CuPy path
    (:func:`rcwa_jones_1d`, Li 2003, v5.11.0) and the 2-D tensor / stack paths
    (:func:`rcwa_jones_2d`, :class:`RCWAStack`, Li 2003 6-tuple generator, GAP2
    v5.14.1) -- those callers pass ``allow_offplane=True`` and route to the full
    generalized-S-matrix path.  This guard therefore fires only on the paths that
    genuinely CANNOT represent z-coupling: the z-decoupled legacy in-plane subset
    and the scalar/isotropic efficiency solvers.  Concrete (NumPy / CuPy) tensors
    only -- a JAX tensor is skipped (not materialisable here) and assumed in-plane
    on the differentiable path."""
    has_off = _tensor_offplane_present(*tensors)
    if has_off and not allow_offplane:
        raise ValueError(
            f"{fn_name}: this path is the z-decoupled in-plane tensor subset "
            f"(exx, exy, eyx, eyy, ezz); the supplied tensor has out-of-plane "
            f"coupling (eps_xz / eps_yz / eps_zx / eps_zy != 0 -- e.g. a "
            f"tilted-director LC or a magneto-optic / gyrotropic tensor), which "
            f"it would silently drop.  Full 3x3 (out-of-plane) tensors are "
            f"supported by rcwa_jones_1d, rcwa_jones_2d, and RCWAStack (forward "
            f"R / T / Jones); use one of those for an out-of-plane tensor.")
    return has_off



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
        # Return a PLAIN tuple, not whatever ``jnp.linalg.eig`` returns: modern
        # JAX (>=0.4.x / numpy 2.0) returns an ``EigResult`` namedtuple, a custom
        # pytree node.  ``custom_vjp`` requires the primal ``f`` and its ``fwd``
        # rule to share output pytree structure; ``_eig_raw_fwd`` returns a plain
        # ``(lam, V)`` tuple, so the primal MUST too -- otherwise the structures
        # disagree and ``grad`` composed with ``vmap`` raises (plain ``grad``
        # happened to tolerate it).  Unpacking here is version-agnostic.
        lam, V = jnp.linalg.eig(A)
        return lam, V

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



# ===========================================================================
# Unified multi-layer API: RCWAStack / RCWAResult (+ caching, Jones bridge)
# ===========================================================================

# Cache of homogeneous half-space eigenmodes -- the most-repeated solve cost:
# the same superstrate / substrate / spacer recurs across a stack and across
# a wavelength or angle sweep.  Guarded by a Lock for thread safety; cleared
# via the library cache registry.
#
# v5.17.1 (audit P2-16/P2-17): bounded LRU OrderedDict (was a plain unbounded
# dict).  The key embeds (wl, theta, phi), so every point of
# ``solve_vs_wavelength`` / an angle sweep minted 2 permanent entries, each
# holding dense (2N, 2N) complex W and V (~10.7 MB/entry at nox=noy=8):
# a 500-wavelength sweep retained ~10.7 GB for the life of the process.
# Bound rationale: one ``RCWAStack.solve`` touches exactly 2 entries
# (sup + sub), so 32 keeps the last 16 (wavelength, angle) source
# configurations hot -- repeated re-solves of the same geometry (optimizer
# loops, symmetry A/B checks) still hit -- while capping worst-case
# retention at ~32 x entry (e.g. ~342 MB at nox=noy=8).  Same
# move_to_end/popitem pattern as ``_H_CACHE`` (propagators/fft_infra.py)
# and the v4.12.2 LRU conversions.  Eviction only drops the dict's
# reference; the value tuple (W, V, kz) held by any in-flight caller is
# never mutated, and a post-eviction recompute is a pure function of
# (Kx, Ky, eps) so it returns byte-identical arrays.
_HOMOG_CACHE: 'OrderedDict[tuple, tuple]' = OrderedDict()
_HOMOG_CACHE_SIZE = 32
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
            _HOMOG_CACHE.move_to_end(key)
    if hit is not None:
        return hit
    res = _homogeneous_eigenmodes(Kx, Ky, eps)
    with _HOMOG_LOCK:
        _HOMOG_CACHE[key] = res
        _HOMOG_CACHE.move_to_end(key)
        while len(_HOMOG_CACHE) > _HOMOG_CACHE_SIZE:
            _HOMOG_CACHE.popitem(last=False)
    return res


# Register the RCWA caches with the library cache registry (so the global
# "clear all caches" path empties them too).  Canonical v4.16.0 enrollment
# pattern (mirrors propagators/propagation.py).
try:
    import sys as _sys

    from ..._cache_registry import register_cache_clearer as _register_cache_clearer
    _this_mod = _sys.modules[__name__]
    _register_cache_clearer(
        "rcwa_homogeneous_modes",
        lambda: getattr(_this_mod, "_clear_rcwa_caches")(),
    )
except ImportError:  # pragma: no cover - registry always present in-tree
    pass

__all__ = [
    "_C",
    "_MAX_HARMONICS",
    "_BLAS_STATE",
    "_get_blas_threads",
    "set_blas_threads",
    "rcwa_blas_threads",
    "_blas_limit",
    "_with_blas_limit",
    "_stabilize_bumps",
    "_eig_for",
    "_block",
    "_rcwa_xp",
    "_is_traced",
    "_concrete",
    "Efficiency2D",
    "_EnergyError",
    "_EnergyWarning",
    "_check_energy",
    "_require_jax_x64",
    "_normalize_pol",
    "_sqrt_forward",
    "_forward_flux_kz",
    "_inv_lam",
    "_sqrt_decay",
    "_require_propagating_incidence",
    "_grazing_safe_wavelength",
    "_validate_geometry",
    "_validate_cell_sampling",
    "_validate_shapes",
    "_fourier_coeffs_1d",
    "_toeplitz_1d",
    "_layer_Q_matrix",
    "_order_flip_perm",
    "_flip_invariant",
    "_recentering_phase",
    "_even_basis_desc",
    "_even_fold",
    "_even_project",
    "_even_unfold",
    "_symmetric_solve_rt",
    "_symmetric_cascade_rt",
    "_tensor_PQ",
    "_scalar_PQ",
    "_layer_eigenmodes",
    "_homogeneous_eigenmodes",
    "_redheffer_star",
    "_interface_smatrix",
    "_propagation_smatrix",
    "_propagation_star",
    "_propagation_star_general",
    "_modes_to_M",
    "_interface_smatrix_general",
    "_propagation_smatrix_general",
    "_order_key",
    "_max_aligned_delta",
    "_rcwa_convergence_stack",
    "rcwa_extrapolate",
    "uniaxial_tensor",
    "_toeplitz_of_profile",
    "_inv_toeplitz_of_profile",
    "_tensor_convolutions",
    "_tensor_has_offplane",
    "_tensor_convolutions_full",
    "_select_forward_flux",
    "_layer_eigenmodes_tensor",
    "_tensor_offplane_present",
    "_reject_jax_offplane",
    "_require_inplane_tensor",
    "_JAX_EIG_STABLE",
    "_jax_eig_stable",
    "_HOMOG_CACHE",
    "_HOMOG_LOCK",
    "_clear_rcwa_caches",
    "_cached_homogeneous_eigenmodes",
]
