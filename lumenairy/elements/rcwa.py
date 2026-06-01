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
)

__all__ = [
    "rcwa_efficiency_1d",
    "rcwa_efficiency_vs_wavelength",
    "rcwa_efficiency_2d",
    "rcwa_efficiency_2d_shapes",
    "rcwa_jones_1d",
    "rcwa_jones_2d",
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


def _check_energy(fn_name, R, T):
    """Raise if the total efficiency exceeds the incident power by a large
    margin.  A PASSIVE structure cannot reflect + transmit more than what
    comes in, so ``sum(R) + sum(T) >> 1`` per incident polarization signals a
    numerical instability -- typically a near-degenerate / ill-conditioned
    layer eigenproblem at a specific large-period / low-contrast geometry
    (the blow-up is erratic: it can hit one period and not the next) -- that
    was otherwise SILENTLY returning a non-physical answer (R+T up to 1e30+).

    Skipped on the JAX path (the sums are traced).  Lossy media give R+T < 1
    (never triggered); the tolerance leaves normal Wood-nudge residue alone.
    """
    tot = float(np.real(np.sum(np.asarray(R))) + np.real(np.sum(np.asarray(T))))
    n_states = int(R.shape[0]) if getattr(R, "ndim", 1) == 2 else 1
    if tot > n_states * 1.05:
        raise ValueError(
            f"{fn_name}: energy non-conservation detected (sum R+T = "
            f"{tot:.3e} exceeds {n_states}); the solve is numerically unstable "
            f"at this geometry (a near-degenerate layer eigenproblem, common "
            f"at very large period / low index contrast).  Reduce n_orders, "
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

    Returns
    -------
    orders : (2*n_orders+1,) int ndarray
        Diffraction-order indices, ascending.
    R_eff : (2*n_orders+1,) float ndarray
        Reflected diffraction efficiency per order (evanescent orders 0).
    T_eff : (2*n_orders+1,) float ndarray
        Transmitted diffraction efficiency per order.
    """
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

    # --- convolution matrices -------------------------------------------
    EPS, EPS_II = _binary_grating_convolutions(n_ridge, n_groove, duty_cycle, M)
    # Wall-normal E_x uses the Li inverse rule [[1/eps]]^{-1} when requested
    # (TM / metals); E_y (tangential) always uses the Laurent [[eps]].
    EPS_xx = EPS_II if use_li else EPS

    # --- region (half-space) modes --------------------------------------
    Wref, Vref, kz_ref = _homogeneous_eigenmodes(Kx, Ky, eps_sup)
    Wtrn, Vtrn, kz_trn = _homogeneous_eigenmodes(Kx, Ky, eps_sub)

    # --- global S = (sup|layer) * propagate(layer) * (layer|sub) --------
    Wl, Vl, lam = _layer_eigenmodes(Kx, Ky, EPS, EPS_xx)
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

def _harmonic_orders_2d(n_orders_x, n_orders_y):
    """Flat list of integer ``(m, n)`` diffraction-order pairs on the 2-D
    reciprocal lattice (``m`` slow in ``[-Mx..Mx]``, ``n`` fast in
    ``[-My..My]``).  Returns ``(orders, N)`` with ``orders`` an ``(N, 2)``
    int array and ``N = (2 Mx + 1)(2 My + 1)``."""
    Mx, My = int(n_orders_x), int(n_orders_y)
    m = np.repeat(np.arange(-Mx, Mx + 1), 2 * My + 1)
    n = np.tile(np.arange(-My, My + 1), 2 * Mx + 1)
    orders = np.stack([m, n], axis=1)
    return orders, orders.shape[0]


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
    formulation : {'laurent'}, optional
        Fourier factorization.  ``'laurent'`` (direct rule) is correct and
        fast-converging for dielectrics; the fast-Fourier-factorization
        (normal-vector) rule for 2-D metals is provided separately.

    Returns
    -------
    orders : (N, 2) int ndarray
        Diffraction-order pairs ``(m, n)``.
    R_eff, T_eff : (N,) float ndarray
        Reflected / transmitted diffraction efficiency per order.
    """
    _validate_geometry("rcwa_efficiency_2d",
                       **_concrete(period=period_x, period_y=period_y,
                                   depth=depth, wavelength=wavelength),
                       n_orders=n_orders_x, n_orders_y=n_orders_y)
    _validate_cell_sampling("rcwa_efficiency_2d", eps_cell,
                            n_orders_x, n_orders_y)
    polarization = _normalize_pol("rcwa_efficiency_2d", polarization)
    if formulation != "laurent":
        raise ValueError(
            f"rcwa_efficiency_2d: only formulation='laurent' is available in "
            f"this build, got {formulation!r}.")

    xp = _rcwa_xp("rcwa_efficiency_2d", use_gpu, eps_cell)
    is_jax = backend_name(xp) == "jax"
    if is_jax:
        _warn_if_jax_f32("rcwa_efficiency_2d")

    # Loss-convention bridge (see rcwa_efficiency_1d): conjugate the PUBLIC
    # permittivity in the active namespace (so a JAX eps_cell stays
    # differentiable); the region scalars stay host complex.
    eps_cell = xp.conj(xp.asarray(eps_cell).astype(_C))
    eps_sup = complex(np.conj(_C(n_superstrate) ** 2))
    eps_sub = complex(np.conj(_C(n_substrate) ** 2))

    orders, N = _harmonic_orders_2d(n_orders_x, n_orders_y)
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

    Wref, Vref, kz_ref = _homogeneous_eigenmodes(Kx, Ky, eps_sup)
    Wtrn, Vtrn, kz_trn = _homogeneous_eigenmodes(Kx, Ky, eps_sub)
    Wl, Vl, lam = _layer_eigenmodes(Kx, Ky, EPS, EPS_xx)
    S = _interface_smatrix(Wref, Vref, Wl, Vl)
    S = _redheffer_star(S, _propagation_smatrix(lam, k0 * depth))
    S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wtrn, Vtrn))
    S11, S12, S21, S22 = S

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


def _layer_eigenmodes_tensor(Kx, Ky, Cxx, Cxy, Cyx, Cyy, EZZ):
    """Eigenmodes of a full-in-plane-tensor layer (dimension-agnostic).

    The anisotropic ``Q`` block (rigorously derived and locked to the
    isotropic core by ``b = c = 0`` reduction; getting the ``Cyx`` sign
    wrong silently violates energy conservation at off-axis director
    angles) is::

        Q = [[ Cyx + Kx Ky,   Cyy - Kx Kx ],
             [ Ky Ky - Cxx,   -(Cxy + Ky Kx) ]]

    The ``P`` block is the core's, with the ``E_z`` elimination ``inv(EZZ)``.
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

    xp = _rcwa_xp("rcwa_jones_1d", use_gpu, eps_ridge, eps_groove)
    is_jax = backend_name(xp) == "jax"
    if is_jax:
        _warn_if_jax_f32("rcwa_jones_1d")
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
    profiles = {}
    for key, (ii, jj) in {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0),
                          "yy": (1, 1), "zz": (2, 2)}.items():
        profiles[key] = xp.where(inside, eps_ridge[ii, jj],
                                 eps_groove[ii, jj]).astype(_C)
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
    kz_inc = float(np.real(_sqrt_forward(eps_sup - kx0 ** 2)))
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
        _check_energy("rcwa_jones_1d", R_eff, T_eff)
    return orders, R_eff, T_eff, jones_reflection


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

    orders, N = _harmonic_orders_2d(n_orders_x, n_orders_y)
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
                            dy=None):
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
            carrier = self._order_carrier(idx, nx_i, ny_i, float(dx), dy_f)
            ex += ax * carrier
            ey += ay * carrier
        return JonesField(ex, ey, dx=dx, dy=dy)


class _RCWALayer:
    __slots__ = ("thickness", "kind", "data")

    def __init__(self, thickness, kind, data):
        self.thickness = float(thickness)
        self.kind = kind          # 'uniform' | 'iso' | 'tensor'
        self.data = data


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
        _validate_geometry("add_layer", depth=thickness)
        if eps is not None:
            self._layers.append(_RCWALayer(thickness, "uniform", _C(eps)))
        elif eps_cell is not None:
            cell = np.asarray(eps_cell, dtype=_C)
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
            tcell = np.asarray(eps_tensor_cell, dtype=_C)
            _validate_cell_sampling("add_layer", tcell, self.nox, self.noy)
            self._layers.append(_RCWALayer(thickness, "tensor", tcell))
        return self

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
        xp = array_namespace(Kx)
        if layer.kind == "uniform":
            W, V, kz = _homogeneous_eigenmodes(Kx, Ky, complex(np.conj(layer.data)))
            lam = _sqrt_decay(-xp.concatenate([kz, kz]) ** 2)
            return W, V, lam
        if layer.kind == "iso":
            EPS = _eps_convolution_2d(xp.conj(xp.asarray(layer.data)), orders,
                                      self.nox, self.noy)
            return _layer_eigenmodes(Kx, Ky, EPS, EPS)
        if layer.kind == "shapes":
            # Analytic (host) form factors -> move the convolution to backend.
            eps_bg, shapes = layer.data
            shapes_c = [dict(s, eps=complex(np.conj(_C(s["eps"])))) for s in shapes]
            EPS_np, EPS_inv_np = _analytic_convolutions_2d(
                complex(np.conj(_C(eps_bg))), shapes_c, orders, self.nox,
                self.noy, self.period_x, self.period_y)
            EPS = xp.asarray(EPS_np)
            return _layer_eigenmodes(Kx, Ky, EPS, EPS,
                                     ez_laurent_inv=xp.asarray(EPS_inv_np))
        et = xp.conj(xp.asarray(layer.data))

        def cv(comp):
            return _eps_convolution_2d(comp, orders, self.nox, self.noy)
        return _layer_eigenmodes_tensor(
            Kx, Ky, cv(et[:, :, 0, 0]), cv(et[:, :, 0, 1]),
            cv(et[:, :, 1, 0]), cv(et[:, :, 1, 1]), cv(et[:, :, 2, 2]))

    @_with_blas_limit
    def solve(self) -> RCWAResult:
        """Solve the stack -> :class:`RCWAResult`."""
        if self._source is None:
            raise ValueError("RCWAStack.solve: call set_source first.")
        if not self._layers:
            raise ValueError("RCWAStack.solve: add at least one layer.")
        src = self._source
        wl, theta, phi = src["wavelength"], src["theta"], src["phi"]
        xp = _rcwa_xp("RCWAStack.solve", self.use_gpu)
        bname = backend_name(xp)
        orders, N = _harmonic_orders_2d(self.nox, self.noy)
        eps_sup = complex(np.conj(_C(self.n_superstrate) ** 2))
        eps_sub = complex(np.conj(_C(self.n_substrate) ** 2))
        nre = float(np.real(np.sqrt(eps_sup)))
        kx0 = nre * np.sin(theta) * np.cos(phi)
        ky0 = nre * np.sin(theta) * np.sin(phi)
        _require_propagating_incidence("RCWAStack.solve", eps_sup,
                                       kx0 ** 2 + ky0 ** 2)
        wl = _grazing_safe_wavelength(
            wl, kx0, ky0, orders[:, 0], orders[:, 1], self.period_x,
            self.period_y, [eps_sup, eps_sub] + self._layer_eps_reals())
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

        modes = [self._layer_modes(L, Kx, Ky, orders) for L in self._layers]
        W0, V0, lam0 = modes[0]
        S = _interface_smatrix(Wref, Vref, W0, V0)
        S = _redheffer_star(S, _propagation_smatrix(lam0, k0 * self._layers[0].thickness))
        for i in range(1, len(modes)):
            Wp, Vp, _lp = modes[i - 1]
            Wc, Vc, lamc = modes[i]
            S = _redheffer_star(S, _interface_smatrix(Wp, Vp, Wc, Vc))
            S = _redheffer_star(S, _propagation_smatrix(lamc, k0 * self._layers[i].thickness))
        Wl, Vl, _ll = modes[-1]
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
        _check_energy("RCWAStack.solve", R, T)
        return RCWAResult(out_orders, R, T, Jr, Jt, modal=modal)


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
