"""
lumenairy.gbd -- Gaussian Beamlet Decomposition propagator.

Decompose an arbitrary complex source field into a finite set of
Gaussian beamlets, propagate each beamlet's base ray + complex
beam parameter through the optical system using ABCD matrices,
then coherently recombine at the output plane.

The deterministic counterpart to Monte Carlo HFPI.  Strengths:

* **Deterministic** -- no Monte Carlo noise.
* **Fast** -- typically 100x faster than HFPI for comparable
  image-plane accuracy on smooth refractive systems.
* **Composes with raytrace** -- each beamlet's base ray is just a
  geometric ray, so the existing ``trace`` infrastructure
  propagates everything.
* **Aberration-aware (per-surface form)** --
  ``propagate_gbd_through_prescription(..., per_surface=True)`` evolves each
  beamlet's complex parameter surface-by-surface via the real per-ray
  differential ray transfer (``raytrace.ray_transfer_jacobian``), promoting
  ``Q`` to a ``(N, 2, 2)`` tensor that captures off-axis **astigmatism** and
  higher-order aberration (the paraxial whole-system-ABCD form cannot).
* **Differentiable** -- backend-dispatched (``array_namespace``), so the
  free-space / thin-lens paths run under ``jax.numpy`` and are ``jax.grad`` /
  ``jax.jit`` friendly.

Feature helpers: Husimi (direction-sampling) decomposition for tilted /
diverging sources (``direction_sampling=True``), aperture vignetting
(``apply_aperture_to_beamlets``), polychromatic (``propagate_gbd_freespace_
spectral``), vector / Jones (``propagate_gbd_freespace_vector``), and
auto-sampling (``recommend_gbd_sampling``).

Limitations:

* **Smooth aperture handling** -- a Gaussian beamlet has continuous
  edges; HFPI handles hard cutoffs better.
* **Caustic-region accuracy** -- like all paraxial complex-ray
  methods, GBD's accuracy degrades near a caustic.
* **Polarization** -- ``propagate_gbd_freespace_vector`` propagates the Jones
  components independently (exact for non-polarizing / free-space systems);
  ``propagate_gbd_vector_through_prescription`` carries the vector field through
  a real prescription applying per-surface Fresnel s/p transmission
  (polarization ray tracing).  Reflection / thin-film coatings build on the same
  base-ray trace and remain a future extension.

See ``REFERENCES.txt`` Section C for the foundational publications.

Multi-backend
-------------

Backend dispatched via :func:`lumenairy._array.array_namespace`.

Author: Andrew Traverso
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..backend import array_namespace, is_jax_array

# v5.21: default amplitude-1/e-radius window for the windowed (bounded-support)
# reconstruction the drivers use on the NumPy backend.  5.0 -> Gaussian tail
# exp(-25) ~ 1.4e-11 dropped (machine-precision identical to the dense sum, but
# O(beamlets * window_box) instead of O(beamlets * Ny * Nx)).  Set to ``None``
# for the historical dense sum.  JAX / CuPy always take the dense path.
_GBD_RECONSTRUCT_WINDOW: Optional[float] = 5.0

# audit S2-20: the public ``reconstruct_field_from_beamlets`` default
# ``window=None`` takes the dense O(beamlets * Ny * Nx) path (every INTERNAL
# caller passes ``_GBD_RECONSTRUCT_WINDOW`` = 5.0).  Warn when that dense path
# would run at a genuinely large workload so an external caller is not silently
# stuck on the slow / memory-hungry branch.  The threshold is a work estimate
# (beamlets * output pixels): 256^2 * 2048 ~ 1.3e8 (the 13 GB N=256 case).
_GBD_DENSE_WARN_WORK = 1.0e8

# audit S2-20: with ``direction_sampling=False`` (the default) an INPUT field
# that is already tilted / diverging at the source plane is not walked off
# correctly (its intensity envelope stays put, losing energy on propagation).
# Warn at the propagate entry points when the intensity-weighted MEAN input
# tilt (direction cosine) exceeds this heuristic bound.  Below it the frozen
# axial beamlets + amplitude-carried tilt reproduce a near-paraxial source
# well; above it the position-only decomposition increasingly mis-walks the
# envelope.  Conservative: it does not fire on a flat / near-collimated field
# (mean tilt ~ 0) and, being the MEAN, not the spread, does not fire on a
# symmetric diverging beam -- that subtler case is surfaced only in the docs.
_GBD_TILT_WARN_COS = 0.02


def _mean_input_tilt(E_in, dx, dy, wavelength) -> float:
    """Intensity-weighted mean transverse direction cosine (tilt magnitude) of
    a 2-D input field.  Uses the stable local-momentum expectation
    ``<k_x> = sum Im(conj(E) dE/dx) / (k sum|E|^2)`` (no division by ``E``), so
    it is robust where ``|E|`` is small.  Returns ``0.0`` for a dark field."""
    E = np.asarray(E_in)
    inten = np.abs(E) ** 2
    tot = float(inten.sum())
    if not np.isfinite(tot) or tot <= 0.0:
        return 0.0
    gx = np.gradient(E, dx, axis=-1)
    gy = np.gradient(E, dy, axis=-2)
    kmag = 2.0 * np.pi / wavelength
    lbar = float(np.imag(np.sum(np.conj(E) * gx))) / (kmag * tot)
    mbar = float(np.imag(np.sum(np.conj(E) * gy))) / (kmag * tot)
    return float(np.hypot(lbar, mbar))


# H7 (2026-07-19): ``direction_sampling='auto'`` engages the Husimi (carrier-
# normal) launch when the input carries transverse angular content -- a curved
# (diverging / converging) or tilted wavefront -- above this RMS local-tilt
# threshold (radians).  Below it the flat-wavefront (collimated) field takes the
# axial position-only decomposition, byte-identical to the historical default.
# A real / globally-phased collimated field measures exactly 0 (see
# ``_input_angular_spread``), so the threshold only guards against numerical
# phase noise: 5e-4 rad (~0.03 deg) sits far below any divergence that shifts
# the focus measurably yet far above float round-off.  Defined here (the beamlet
# layer) and re-exported from ``elements.lenses_gbd`` for backward-compatible
# imports; ``apply_real_lens_gbd`` and ``propagate_gbd_through_prescription``
# both resolve their ``'auto'`` policy through it.
_GBD_AUTO_HUSIMI_THRESH = 5e-4


def _input_angular_spread(E_in: np.ndarray, dx: float, dy: float,
                          wavelength: float) -> float:
    """RMS transverse direction cosine (local wavevector magnitude / k) of the
    input field over its bright support -- the SPREAD (not the mean) of the
    launch angles the beamlets would carry.

    Unlike the intensity-weighted MEAN tilt (:func:`_mean_input_tilt`, ~0 for a
    symmetric diverging beam), this fires on a diverging / converging source: a
    curved wavefront's local normals fan out, so their RMS magnitude ~ the beam
    NA.  Uses wrapping-safe nearest-neighbour phase increments
    (``angle(E_i * conj(E_{i-1}))``) so it is robust to fringes up to the grid
    Nyquist, and restricts to the bright support (``|E| > 0.05 max``) so
    dark-region phase noise does not inflate it.  Returns ``0.0`` exactly for a
    real / globally-phased (flat-wavefront) field, so the ``'auto'`` launch is a
    strict no-op there.  NumPy host-side (a cheap eager reduction)."""
    E = np.asarray(E_in)
    mag = np.abs(E)
    peak = float(mag.max()) if mag.size else 0.0
    if peak <= 0.0:
        return 0.0
    k = 2.0 * np.pi / wavelength
    mask = mag > 0.05 * peak
    gx = E[:, 1:] * np.conj(E[:, :-1])
    gy = E[1:, :] * np.conj(E[:-1, :])
    mxx = mask[:, 1:] & mask[:, :-1]
    myy = mask[1:, :] & mask[:-1, :]
    lx = np.angle(gx[mxx]) / (k * dx)
    my = np.angle(gy[myy]) / (k * dy)
    vals = np.concatenate([lx, my])
    if vals.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(vals ** 2)))


def _warn_if_tilted_no_dirsample(E_in, dx, dy, wavelength,
                                 direction_sampling) -> None:
    """Surface the ``direction_sampling`` energy-loss caveat (audit S2-20) at a
    propagate entry point.  When direction sampling is off and the input field
    carries a significant coherent tilt, position-only beamlets do not walk the
    envelope off correctly and ~tens-of-percent of energy can be lost.  NumPy
    backend only (the tilt probe is a cheap eager reduction)."""
    if direction_sampling:
        return
    xp = array_namespace(E_in)
    if xp is not np:
        return
    dyv = dx if dy is None else dy
    try:
        tilt = _mean_input_tilt(E_in, float(dx), float(dyv), float(wavelength))
    except (ValueError, TypeError, ZeroDivisionError, FloatingPointError):
        return
    if tilt > _GBD_TILT_WARN_COS:
        warnings.warn(
            "GBD: the input field carries a mean transverse tilt of "
            f"~{tilt:.3g} (direction cosine) but direction_sampling=False "
            "(the default). Position-only beamlets carry that tilt in the "
            "amplitude, not the launch direction, so an already-tilted / "
            "diverging source does not walk off correctly and can lose "
            "significant energy on propagation. Pass direction_sampling=True "
            "(Husimi) for a source that is already tilted or diverging at the "
            "input plane.",
            RuntimeWarning, stacklevel=3)


# v5.2 (AUDIT_V4_13_1 Part 2 P1-A closure): output-grid kwarg semantics
# disambiguation.  Pre-v5.2 the sub-propagators below interpreted
# ``output_grid`` as ``(Ny, Nx)`` (shape only) while the dispatcher's
# ``propagate(output_grid=...)`` contract advertises ``(N_out, dx_out)``.
# The mismatch silently produced wrong-shape output arrays when the
# dispatcher forwarded the kwarg.  v5.2 keeps the dispatcher contract
# as canonical and renames the sub-propagator kwarg to ``output_shape``
# for the shape-only meaning.  The legacy ``output_grid`` kwarg on
# the sub-propagators is preserved with a ``DeprecationWarning``.
def _resolve_output_shape(
    output_shape: Optional[Tuple[int, int]],
    output_grid: Optional[Any],
    *,
    fn_name: str,
    default_shape: Tuple[int, int],
) -> Tuple[int, int]:
    """Resolve the (Ny, Nx) output shape from the v5.2 ``output_shape``
    kwarg and the deprecated ``output_grid`` legacy kwarg.

    Both kwargs may be ``None`` (use the default).  Passing both raises
    ``ValueError``.  Passing ``output_grid`` emits a
    ``DeprecationWarning`` directing the caller to either
    ``output_shape=(Ny, Nx)`` (the new name for this sub-propagator's
    shape-only kwarg) or the dispatcher's ``output_grid=(N_out, dx_out)``
    convention if they actually want grid resampling.
    """
    if output_shape is not None and output_grid is not None:
        raise ValueError(
            f"{fn_name}: both ``output_shape`` and ``output_grid`` were "
            f"provided.  Pass only ``output_shape=(Ny, Nx)`` for the "
            f"shape-only kwarg (v5.2+) or the dispatcher's "
            f"``output_grid=(N_out, dx_out)`` form via "
            f"``propagate(method=...)``.")
    if output_shape is not None:
        return (int(output_shape[0]), int(output_shape[1]))
    if output_grid is not None:
        warnings.warn(
            f"{fn_name}: the ``output_grid`` kwarg now (v5.2+) means "
            f"the dispatcher's ``(N_out, dx_out)`` grid spec; on "
            f"sub-propagators it has been renamed to ``output_shape`` "
            f"for the ``(Ny, Nx)`` shape-only meaning.  Pass "
            f"``output_shape=(Ny, Nx)`` to silence this warning, or "
            f"call via ``propagate(method='gbd', output_grid=(N_out, "
            f"dx_out), ...)`` if you actually want grid resampling.",
            DeprecationWarning, stacklevel=3,
        )
        return (int(output_grid[0]), int(output_grid[1]))
    return default_shape


# ============================================================================
# Beamlet bundle
# ============================================================================

@dataclass
class BeamletBundle:
    """Coherent set of Gaussian beamlets.

    Field naming aligns with :class:`lumenairy.raytrace.RayBundle`
    and :class:`lumenairy.hfpi.PathBundle`: ``positions`` and
    ``directions`` are the per-beamlet base ray's central position
    and direction.  GBD-specific additions are ``Q`` (complex beam
    parameter), ``amplitude`` (complex on-axis scale), and
    ``waist0`` (initial waist used to evaluate the transverse
    profile).
    """

    positions: object       # (N, 3) -- base ray position per beamlet
    directions: object      # (N, 3) -- base ray direction per beamlet
    # v5.4.7 (audit AUDIT_V5_4_6 #9): Q is the ENGINEERING 1/q parameter
    # (q_code = conj(q_physics)); at the waist Q = -i/z_R.  The physics
    # (exp(-i omega t) / exp(+ikz)) field is rendered in
    # reconstruct_field_from_beamlets via exp(+0.5j k conj(Q) rho^2) =
    # exp(+i k rho^2 / (2 q_physics)) (the v5.4.6 F-1 fix).  The Q-evolution
    # (Q/(1+tQ), Q-1/f) is left in this engineering convention.
    Q: object               # (N,) complex (engineering 1/q; see note above)
    amplitude: object       # (N,) complex on-axis amplitude
    waist0: object          # (N,) initial waist (for profile)

    def __len__(self) -> int:
        try:
            return int(self.positions.shape[0])
        except (AttributeError, TypeError, IndexError):
            # positions may be None, a non-array sentinel, or 0-D.
            return 0


def _q_is_tensor(Q: Any) -> bool:
    """True if ``Q`` is a (N, 2, 2) tensor beam parameter (general astigmatic
    Gaussian), False for the (N,) scalar (rotationally-symmetric) parameter.
    The single dispatch predicate that keeps the scalar paths untouched."""
    return getattr(Q, 'ndim', 1) == 3


def _inv2x2(M: np.ndarray, xp: Any) -> np.ndarray:
    """Batched exact inverse of a stack of 2x2 matrices ``M`` (..., 2, 2).

    Closed-form adjugate / determinant -- ``inv = [[d, -b], [-c, a]] / (ad-bc)``
    -- instead of ``xp.linalg.inv``.  For 2x2 this is arithmetically exact (to
    round-off) and avoids the per-matrix LAPACK ``getrf``/``getri`` dispatch,
    which dominates when the batch is large and each matrix is tiny.  It is also
    made of only ``+ - * /`` so it is JAX-``grad`` / CuPy differentiable, unlike
    the general ``xp.linalg.inv`` VJP on a complex non-symmetric matrix.
    """
    a = M[..., 0, 0]
    b = M[..., 0, 1]
    c = M[..., 1, 0]
    d = M[..., 1, 1]
    det = a * d - b * c
    inv = xp.stack([xp.stack([d, -b], axis=-1),
                    xp.stack([-c, a], axis=-1)], axis=-2)
    return inv / det[..., None, None]


def _eigvals2x2(M: np.ndarray, xp: Any) -> np.ndarray:
    """Batched eigenvalues of a stack of 2x2 matrices ``M`` (..., 2, 2).

    Quadratic-formula roots of ``lam^2 - tr*lam + det`` -- ``lam = tr/2 +-
    sqrt((tr/2)^2 - det)`` -- returning ``(..., 2)``.  Replaces
    ``xp.linalg.eigvals`` (LAPACK ``geev`` per matrix, and no JAX VJP for a
    complex non-symmetric 2x2).  ``xp.sqrt`` takes the complex principal root,
    matching ``eigvals`` up to eigenvalue ORDER; every consumer here reduces the
    pair with an order-independent ``prod`` so the order is immaterial.
    """
    a = M[..., 0, 0]
    b = M[..., 0, 1]
    c = M[..., 1, 0]
    d = M[..., 1, 1]
    half_tr = 0.5 * (a + d)
    det = a * d - b * c
    disc = xp.sqrt(half_tr * half_tr - det)
    return xp.stack([half_tr + disc, half_tr - disc], axis=-1)


def _det2x2(M: np.ndarray) -> np.ndarray:
    """Batched determinant of a stack of 2x2 matrices -- ``ad - bc``, exact,
    no LAPACK dispatch."""
    return M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]


def _guard_tensor_freespace_branch(lam, t, xp) -> None:
    """Guard the per-eigenvalue principal-sqrt amplitude branch of the tensor
    (astigmatic / anamorphic) free-space Mobius step (audit S2-17).

    That step evaluates ``prod_i 1/sqrt(1 + t*lambda_i)`` on the PRINCIPAL
    branch of the square root.  The branch is focus-continuous as long as the
    argument ``w_i = 1 + t*lambda_i`` never sits on the principal-sqrt branch
    cut (the negative real axis).  A physical beamlet has ``Im(lambda_i) < 0``
    (``1/q`` carries ``-i*lambda/(pi w^2)``), so ``Im(w_i) = t*Im(lambda_i))``
    stays strictly off the axis and the sqrt is unambiguous through a focus.
    The audit framed the assumption as ``Im(lambda_i) < 0``; the SHARP failure
    condition, though, is ``w_i`` landing ON the cut -- ``Re(w_i) < 0`` with
    ``Im(w_i) ~ 0`` -- which needs a NEAR-REAL curvature eigenvalue driven past
    its focus (extreme aberration).  There the principal sqrt can SILENTLY FLIP
    that beamlet's amplitude sign.  Emit a warning so a caller can distrust the
    affected beamlets; the numeric result is unchanged (detection only).

    Testing the argument (not merely the sign of ``Im(lambda)``) avoids a false
    positive on a fold / periscope reframe, where a reflection legitimately
    flips ``Im(lambda)`` positive while ``w_i`` stays well off the cut.

    Only the NumPy backend is checked eagerly: a JAX / CuPy tracer cannot be
    reduced to a Python bool here without breaking ``jit`` / ``grad``.
    """
    if xp is not np:
        return
    la = np.asarray(lam)                       # (n, 2) curvature eigenvalues
    if la.size == 0:
        return
    tt = np.asarray(t, dtype=la.real.dtype).reshape(-1)[:, None]
    w = 1.0 + tt * la                          # per-eigenvalue sqrt argument
    on_cut = (w.real < 0.0) & (np.abs(w.imag) <= 1e-6 * (1.0 + np.abs(w)))
    if np.any(on_cut):
        warnings.warn(
            "GBD tensor free-space: a beamlet curvature eigenvalue is nearly "
            "real and driven past its focus (1 + t*lambda on the negative real "
            "axis). The per-eigenvalue principal-sqrt amplitude branch is "
            "ambiguous there, so the affected beamlet amplitude sign may be "
            "unreliable (extreme aberration).",
            RuntimeWarning, stacklevel=3)


def _freespace_tensor_moebius_np(Q, amp, t, k0):
    """Branch-safe free-space Mobius update of a tensor beamlet ``Q`` and its
    amplitude, shared by the NumPy image-plane reframers (audit S2-14: this
    block was written out identically at two lens-image sites).

    Applies the matrix Mobius ``Q_new = Q (I + t Q)^{-1}`` (re-symmetrized), the
    per-eigenvalue amplitude ``prod_i 1/sqrt(1 + t lambda_i)`` (each principal
    sqrt continuous through a focus, guarded by
    :func:`_guard_tensor_freespace_branch`), and the axial phase
    ``exp(i k0 t)``.  Returns ``(Q, amp)``.  Reproduces the former inline block
    operation-for-operation, so both routed sites are bit-identical."""
    I2 = np.eye(2)[None, :, :]
    lam = _eigvals2x2(Q, np)
    _guard_tensor_freespace_branch(lam, t, np)
    amp = amp * np.prod(1.0 / np.sqrt(1.0 + t[:, None] * lam), axis=1)
    amp = amp * np.exp(1j * k0 * t)
    Q = Q @ _inv2x2(I2 + t[:, None, None] * Q, np)
    Q = 0.5 * (Q + np.transpose(Q, (0, 2, 1)))
    return Q, amp


# ============================================================================
# Source-plane decomposition
# ============================================================================

def decompose_field_to_beamlets(
    E_in: np.ndarray,
    dx: float,
    *,
    wavelength: float,
    dy: Optional[float] = None,
    waist_factor: float = 1.0,
    sample_step: int = 1,
    z_input_plane: float = 0.0,
    direction_sampling: bool = False,
) -> BeamletBundle:
    """Decomposition of a 2-D complex source field into a regular grid
    of Gaussian beamlets.

    Each grid pixel becomes a beamlet centred at its physical
    coordinate, with on-axis amplitude ``E_in[i, j]`` and Gaussian
    waist ``w0 = waist_factor * dx``.

    Parameters
    ----------
    direction_sampling : bool, default ``False``
        Selects how each beamlet's launch **direction** is chosen.

        ``False`` (default, position-only) -- every beamlet is launched
        along ``(0, 0, +1)`` and the field's local tilt is carried in
        the beamlet AMPLITUDE.  Backward-compatible and byte-identical to
        the historical behaviour.  Correct for near-collimated / paraxial
        sources and for any beam that *acquires* its angles from later
        optics (the direction is updated at each lens / ABCD element), but
        a source that is *already* tilted or diverging at the input plane
        is not walked off correctly (its intensity envelope stays put).

        ``True`` ("Husimi" / Gabor decomposition) -- each beamlet is
        launched along the field's **local wavevector**, obtained from the
        transverse phase gradient
        ``k_x = d(arg E)/dx = Im((dE/dx) / E)`` (and likewise ``k_y``);
        the direction cosines are ``(k_x/k, k_y/k, sqrt(1 - ...))``.  The
        tilt then lives in the *direction*, so a tilted or diverging
        source walks off correctly on propagation.  Falls back to axial
        where ``|E|`` is negligible or the local angle exceeds the light
        cone (``k_x^2 + k_y^2 >= k^2``, i.e. under-sampled / evanescent),
        so it never produces NaNs.  The local gradient must be resolved by
        the grid (phase change ``< pi`` per pixel); beyond that use HFPI.
    """
    xp = array_namespace(E_in)
    Ny, Nx = E_in.shape[-2], E_in.shape[-1]
    # v5.21: anamorphic sampling.  dy == dx (the default) keeps the scalar-Q
    # circular-beamlet path byte-identical; dy != dx makes each beamlet an
    # ELLIPTICAL (axis-aligned) Gaussian -> a DIAGONAL tensor Q.
    if dy is None:
        dy = dx
    _aniso = abs(float(dy) - float(dx)) > 1e-12 * max(abs(float(dx)), 1.0)

    iy = xp.arange(0, Ny, sample_step)
    ix = xp.arange(0, Nx, sample_step)
    Iy, Ix = xp.meshgrid(iy, ix, indexing='ij')
    Iy = Iy.reshape(-1)
    Ix = Ix.reshape(-1)
    n = Iy.shape[0]

    # v4.12.1 (B1-10): switch from cell-centred `(arange(N) - N/2 + 0.5)*dx`
    # to pixel-centred `(arange(N) - N/2)*dx`, matching the library-wide
    # convention (ASM, Fresnel, RS, sources, ``apply_fresnel_curvature``).
    # ``reconstruct_field_from_beamlets`` (line ~264) already uses the
    # pixel-centred grid, so prior to this fix a coherent self-roundtrip
    # walked the beamlet centres half a pixel relative to the
    # reconstruction grid -- producing a `k_0 * dx / 2 * off-axis` phase
    # error that grew with NA and field angle.
    x_b = (Ix - Nx / 2) * dx
    y_b = (Iy - Ny / 2) * dy
    z_b = xp.full((n,), float(z_input_plane), dtype=x_b.dtype)
    positions = xp.stack([x_b, y_b, z_b], axis=-1)

    if direction_sampling:
        # Husimi / Gabor: launch each beamlet along the field's LOCAL
        # wavevector k_local = grad(arg E) = Im((grad E) / E).  This puts
        # the input tilt into the beamlet DIRECTION (so it walks off on
        # propagation) instead of only its amplitude.
        kmag = 2.0 * float(np.pi) / wavelength
        gx = xp.gradient(E_in, dx, axis=-1)      # dE/dx
        gy = xp.gradient(E_in, dy, axis=-2)      # dE/dy
        E_c = E_in[Iy, Ix]
        mag_c = xp.abs(E_c)
        thr = 1e-6 * xp.abs(E_in).max()
        E_safe = xp.where(mag_c > thr, E_c, xp.full_like(E_c, thr))
        kx = xp.imag(gx[Iy, Ix] / E_safe) / kmag
        ky = xp.imag(gy[Iy, Ix] / E_safe) / kmag
        # Fall back to axial where |E| is negligible or the local angle
        # leaves the light cone (aliased / evanescent) -- keeps N real.
        weak = (mag_c <= thr) | (kx * kx + ky * ky >= 1.0)
        zero = xp.zeros_like(kx)
        L = xp.where(weak, zero, kx).astype(x_b.dtype)
        M = xp.where(weak, zero, ky).astype(x_b.dtype)
        N = xp.sqrt(xp.maximum(1.0 - L * L - M * M, xp.zeros_like(L)))
    else:
        L = xp.zeros((n,), dtype=x_b.dtype)
        M = xp.zeros((n,), dtype=x_b.dtype)
        N = xp.ones((n,), dtype=x_b.dtype)
    directions = xp.stack([L, M, N], axis=-1)

    _cdt = xp.complex128 if hasattr(xp, 'complex128') else 'complex128'
    wx = waist_factor * dx
    wy = waist_factor * dy
    sample = E_in[Iy, Ix]
    if _aniso:
        # Elliptical beamlet -> diagonal tensor Q = diag(-i/zRx, -i/zRy).
        zRx = float(np.pi) * (wx ** 2) / wavelength
        zRy = float(np.pi) * (wy ** 2) / wavelength
        diag = xp.stack([xp.full((n,), -1j / zRx, dtype=_cdt),
                         xp.full((n,), -1j / zRy, dtype=_cdt)], axis=-1)
        Q = diag[:, :, None] * xp.eye(2, dtype=_cdt)[None, :, :]  # (n,2,2)
        waist0 = xp.full((n,), float(np.sqrt(wx * wy)), dtype=x_b.dtype)
        pixel_area = (sample_step ** 2) * dx * dy
        amplitude = sample * pixel_area / (float(np.pi) * wx * wy)
    else:
        w0 = wx
        z_R = float(np.pi) * (w0 ** 2) / wavelength
        Q = xp.full((n,), -1j / z_R, dtype=_cdt)
        waist0 = xp.full((n,), float(w0), dtype=x_b.dtype)
        pixel_area = (sample_step * dx) ** 2
        amplitude = sample * pixel_area / (float(np.pi) * w0 * w0)

    return BeamletBundle(
        positions=positions,
        directions=directions,
        Q=Q,
        amplitude=amplitude.astype(Q.dtype),
        waist0=waist0,
    )


def recommend_gbd_sampling(
    E_in: np.ndarray,
    dx: float,
    *,
    target_overlap: float = 1.5,
    oversample: float = 4.0,
) -> Dict[str, Any]:
    """Recommend ``sample_step`` and ``waist_factor`` for a source field.

    Picks the beamlet grid automatically instead of leaving ``sample_step``
    and ``waist_factor`` as blind manual knobs.  The spacing must resolve BOTH
    the field's amplitude structure and its local wavevector (phase gradient),
    so the beamlet grid is chosen as the finer of:

    * an **amplitude** feature scale ``||E|| / || grad E ||`` (L2), and
    * a **phase / angular** Nyquist from the max local tilt
      ``k_local = grad(arg E)`` (a beamlet spacing that resolves the tilt
      variation) --

    downsampled by ``oversample`` beamlets per feature, clamped to
    ``[1, N/4]``.  ``waist_factor`` is then set to ``target_overlap *
    sample_step`` so neighbouring beamlets overlap (``w0 = target_overlap *
    spacing``); ``target_overlap ~ 1.5`` is a good smooth-field default.

    Wavelength dependence is DATA-DRIVEN, not parametric
    ----------------------------------------------------
    Both feature scales are measured off ``E_in`` itself and neither needs
    ``wavelength``:

    * the amplitude scale ``||E|| / ||grad E||`` is a pure envelope
      property;
    * the phase scale ``2*pi / max|grad(arg E)|`` is read from the field's
      own phase gradient in **rad/m**, and for any physical field that
      gradient already carries the ``1/lambda`` scaling (a converging
      wavefront ``exp(i k r^2 / 2R)`` has ``|grad phi| = k r / R``).

    So the recommendation *does* move with wavelength -- through the
    field, which is the only place the wavelength is actually observable.
    Measured on a Gaussian with ``R = 4 mm`` curvature (N=96,
    ``dx = 1 um``): ``max|grad phi|`` = 2.159e5 -> 5.602e4 -> 8.686e3
    rad/m for ``lambda`` = 0.4 / 1.55 / 10 um (exactly ``1/lambda``), and
    ``sample_step`` = 3 -> 4 -> 4 accordingly.

    Notes on the removed ``wavelength`` keyword
    -------------------------------------------
    ``wavelength`` was **Deprecated and unused** (audit P8) and is REMOVED
    in v5.30.  See above: the wavelength reaches this function through
    ``E_in``'s phase gradient, so the body never read it.  A
    ``lambda``-dependent beamlet-divergence term was considered and
    rejected on measurement -- the sibling
    :func:`converge_gbd_sampling`, which scores real propagations
    against the exact ASM oracle, returns the SAME optimal overlap
    (1.0) at ``lambda`` = 0.633 / 1.55 / 3.0 um on a fixed field and
    ``sample_step`` (only the error magnitude rises, 1.21e-2 ->
    1.78e-2 -> 2.60e-2), so there is no wavelength-dependent optimum
    for this function's ``waist_factor`` to track.  Use
    :func:`converge_gbd_sampling` when you want the width tuned
    against a propagation at a specific wavelength and distance.

    Returns
    -------
    dict
        ``{'sample_step': int, 'waist_factor': float, 'n_beamlets': int}``
        -- splat into ``decompose_field_to_beamlets`` /
        ``propagate_gbd_*`` (``**recommend_gbd_sampling(...)`` minus
        ``n_beamlets``).

    .. versionchanged:: 5.30
        ``wavelength`` (audit P8) is **REMOVED**.  It was a required
        keyword the body never read (identical output for ``lambda`` =
        0.4 um and 10 um at fixed ``E_in``); v5.30 first made it optional
        + ``DeprecationWarning``, and the W5 shim-removal wave deletes it
        in the same release rather than carrying an inert keyword to
        v5.32.  Migration: drop the kwarg -- the returned dict was proven
        independent of it by construction, so no output changes.  For a
        width tuned against a real propagation at a given wavelength and
        distance use :func:`converge_gbd_sampling`.
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'recommend_gbd_sampling', input_kind='field')
    xp = array_namespace(E_in)
    Ny, Nx = E_in.shape[-2], E_in.shape[-1]
    gx = xp.gradient(E_in, dx, axis=-1)
    gy = xp.gradient(E_in, dx, axis=-2)
    amp = xp.abs(E_in)
    norm_E = float(xp.sqrt(xp.sum(amp ** 2)))
    grad_mag = xp.sqrt(xp.abs(gx) ** 2 + xp.abs(gy) ** 2)
    norm_grad = float(xp.sqrt(xp.sum(grad_mag ** 2)))
    # Amplitude feature scale (m); guard the smooth/flat field (grad ~ 0).
    if norm_grad > 0.0:
        feat_amp = norm_E / norm_grad
    else:
        feat_amp = float(Nx * dx)
    # Phase / angular feature: max local transverse spatial frequency of the
    # tilt.  k_local = Im(grad E / E) [rad/m]; the beamlet spacing must be
    # small vs 1/max|k_local| (in cycles) so the launch direction is locally
    # constant.
    thr = 1e-6 * float(xp.max(amp)) if amp.size else 0.0
    safe = xp.where(amp > thr, E_in, xp.full_like(E_in, thr + 1e-30))
    kx = xp.imag(gx / safe)
    ky = xp.imag(gy / safe)
    kloc_max = float(xp.max(xp.sqrt(kx ** 2 + ky ** 2)))
    feat_phase = (2.0 * float(np.pi) / kloc_max) if kloc_max > 0.0 else feat_amp
    feature = min(feat_amp, feat_phase)
    s_px = feature / (oversample * dx)
    sample_step = int(min(max(1, round(s_px)), max(1, min(Ny, Nx) // 4)))
    waist_factor = float(target_overlap * sample_step)
    n_beamlets = ((Ny + sample_step - 1) // sample_step) * \
                 ((Nx + sample_step - 1) // sample_step)
    return {
        'sample_step': sample_step,
        'waist_factor': waist_factor,
        'n_beamlets': int(n_beamlets),
    }


def converge_gbd_sampling(
    E_in: np.ndarray,
    dx: float,
    *,
    wavelength: float,
    test_distance: float,
    dy: Optional[float] = None,
    sample_step: Optional[int] = None,
    overlaps: Any = (2.5, 2.0, 1.5, 1.2, 1.0, 0.8),
    rtol: float = 1e-2,
    reference: Optional[np.ndarray] = None,
    window: float = 5.0,
    direction_sampling: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Pick the beamlet width that makes a free-space GBD propagation most
    accurate, validated against the EXACT angular-spectrum propagator.

    GBD accuracy is a genuine beamlet-width trade-off, NOT "smaller is always
    better": a **wider** beamlet (larger ``w0`` -> smaller divergence NA
    ``lambda/(pi w0)``) stays more paraxial as it spreads, but its single
    per-beamlet quadratic phase ``Q`` then follows a curved / high-NA total
    wavefront worse; a **narrower** beamlet tracks the wavefront locally but
    diverges faster than its paraxial ``Q`` describes.  The optimum is
    field-dependent: a **broad, flat plateau** for smooth / low-NA fields (where
    accuracy is set by the beamlet density and reconstruction window, not the
    width), and a **genuine interior optimum** for high-NA or finely-structured
    fields where the two errors above actually compete.  The default sweep stays
    in the proper-frame range (overlap ``>= 0.8``); much narrower beamlets stop
    forming a frame and only "work" by degenerating to near-delta samples of a
    smooth field (a low-NA artefact, not real accuracy).

    The width is swept as the **overlap** ``= w0 / spacing`` (the physically
    meaningful knob: it is decimation-independent, whereas a raw ``waist_factor``
    relative to ``dx`` under-overlaps a coarse ``sample_step`` grid and just
    trades gaps).  The beamlet width used is ``waist_factor = overlap *
    sample_step`` (so ``w0 = overlap * sample_step * dx``).  Each decomposition
    is propagated ``test_distance`` in free space, reconstructed, and scored
    against :func:`~lumenairy.propagators.asm.angular_spectrum_propagate` --
    exact scalar Helmholtz, the trusted free-space oracle -- returning the
    overlap with the smallest error plus the full error curve.

    The score reconciles the known GBD<->ASM global beamlet-Gouy phase
    convention first (via :func:`match_global_phase`), so it measures true
    shape/decomposition accuracy, not the width-dependent global-phase offset.

    Parameters
    ----------
    test_distance : float
        Representative free-space distance (m) at which to score accuracy.
    sample_step : int, optional
        Beamlet grid stride; defaults to :func:`recommend_gbd_sampling`.
    overlaps : sequence of float
        Candidate beamlet overlaps ``w0 / spacing`` to score (``~1.5`` is a
        typical smooth-field default; ``> 2`` over-smooths, ``< 1`` leaves gaps).
    rtol : float
        Target relative error; ``converged`` is True if the best overlap beats it.
    reference : ndarray, optional
        Override the ASM oracle with a supplied reference field at
        ``test_distance`` (e.g. a higher-fidelity through-optics solver, to tune
        the beamlet grid for a non-free-space problem).

    Returns
    -------
    dict
        ``{'overlap', 'waist_factor', 'sample_step', 'error', 'errors',
        'converged', 'n_beamlets', 'reference'}``.  ``errors`` maps each
        candidate overlap to its relative L2 error vs the oracle; splat
        ``waist_factor`` / ``sample_step`` into ``decompose_field_to_beamlets`` /
        ``propagate_gbd_*``.
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'converge_gbd_sampling', input_kind='field')
    from ..propagators.asm import angular_spectrum_propagate
    xp = array_namespace(E_in)
    Ny, Nx = E_in.shape[-2], E_in.shape[-1]
    if dy is None:
        dy = dx
    if sample_step is None:
        # v5.30 (audit P8): ``wavelength`` is deprecated on the
        # recommender (it never read it) -- don't forward it, or every
        # default-``sample_step`` convergence run emits a spurious
        # DeprecationWarning from inside the library.
        sample_step = int(recommend_gbd_sampling(E_in, dx)['sample_step'])
    used_asm = reference is None
    if used_asm:
        reference = angular_spectrum_propagate(E_in, test_distance,
                                               wavelength, dx)
    ref = xp.asarray(reference)
    ref_norm = float(xp.sqrt(xp.sum(xp.abs(ref) ** 2))) + 1e-300

    errors: Dict[float, float] = {}
    best_ov, best_err = None, float('inf')
    for ov in overlaps:
        wf = float(ov) * sample_step   # w0 = overlap * spacing
        b = decompose_field_to_beamlets(
            E_in, dx, wavelength=wavelength, dy=dy, sample_step=sample_step,
            waist_factor=wf, direction_sampling=direction_sampling)
        b = propagate_beamlets_freespace(b, test_distance, wavelength)
        F = reconstruct_field_from_beamlets(
            b, Ny=Ny, Nx=Nx, dx=dx, dy=dy, wavelength=wavelength, window=window)
        # GBD and ASM share coordinates/direction but differ by the global
        # beamlet-Gouy phase convention (2*arctan(z/zR_b), zR_b depends on
        # width) -- a convention, not a decomposition error.  Reconcile it so
        # the score measures true SHAPE accuracy, not the global-phase offset.
        F = match_global_phase(F, ref)
        err = float(xp.sqrt(xp.sum(xp.abs(F - ref) ** 2)) / ref_norm)
        errors[float(ov)] = err
        if verbose:
            print(f"  overlap={float(ov):.3f} (waist_factor={wf:.2f})  "
                  f"err={err:.3e}")
        if err < best_err:
            best_err, best_ov = err, float(ov)
    n_beam = (((Ny + sample_step - 1) // sample_step)
              * ((Nx + sample_step - 1) // sample_step))
    return {
        'overlap': best_ov,
        'waist_factor': float(best_ov) * int(sample_step),
        'sample_step': int(sample_step),
        'error': best_err,
        'errors': errors,
        'converged': bool(best_err < rtol),
        'n_beamlets': int(n_beam),
        'reference': 'asm' if used_asm else 'supplied',
    }


def _concat_bundles(a: BeamletBundle, b: BeamletBundle) -> BeamletBundle:
    """Concatenate two beamlet bundles (same backend, same Q rank)."""
    xp = array_namespace(a.positions)
    def _cat(x, y):
        if x is None or y is None:
            return None
        return xp.concatenate([x, y], axis=0)
    return BeamletBundle(
        positions=_cat(a.positions, b.positions),
        directions=_cat(a.directions, b.directions),
        Q=_cat(a.Q, b.Q),
        amplitude=_cat(a.amplitude, b.amplitude),
        waist0=_cat(a.waist0, b.waist0),
    )


def decompose_field_adaptive(
    E_in: np.ndarray,
    dx: float,
    *,
    wavelength: float,
    dy: Optional[float] = None,
    base_step: int = 4,
    refine_step: int = 1,
    refine_ratio: float = 0.12,
    target_overlap: float = 1.5,
    direction_sampling: bool = False,
    indicator: str = 'gradient',
    return_stats: bool = False,
) -> BeamletBundle:
    """Two-level **adaptive / edge-refined** Gaussian-beamlet decomposition.

    A uniform beamlet grid resolves a sharp feature (a hard-aperture rim, an
    amplitude step, a phase kink) only to the beamlet pitch -- the source of
    GBD's residual error at hard-edged caustics (soft Gaussian edges cannot
    reproduce the Airy ring structure when the beamlet waist is as wide as the
    ring).  This decomposition places a **coarse** grid at ``base_step`` where
    the field is smooth and a **finer, narrower** grid at ``refine_step`` only
    where a local sharpness indicator exceeds ``refine_ratio`` of its maximum.
    The refined beamlets have waist ``target_overlap * refine_step * dx`` (vs
    ``target_overlap * base_step * dx`` for the coarse ones), so they resolve
    the edge at the fine scale.  ALL coarse cells are **kept**; the fine
    beamlets are decomposed from the RESIDUAL ``E - reconstruct(coarse)`` and
    **add** a correction on top (a clean partition of unity -- no coarse/fine
    seam, no double counting), rather than being dropped and replaced by a
    mismatched-waist fine beamlet.

    Because the fine grid is confined to the (typically thin) feature set, the
    beamlet count is far below a uniform ``refine_step`` grid while the edge is
    resolved as if it were uniform-fine.  Pairs naturally with the windowed
    reconstruction (``reconstruct_field_from_beamlets(..., window=...)``), which
    makes the extra edge beamlets nearly free.

    Parameters
    ----------
    base_step, refine_step : int
        Coarse / fine beamlet pitch in pixels (``refine_step < base_step``).
    refine_ratio : float
        Flag a coarse cell when the peak indicator inside it exceeds this
        fraction of the global-max indicator.  Lower -> more refinement.
    indicator : {'gradient', 'amplitude'}
        ``'gradient'`` flags on ``|grad E|`` (edges of amplitude OR phase);
        ``'amplitude'`` flags on ``|grad |E||`` (amplitude edges only).
    return_stats : bool
        If True, also return a dict with ``n_coarse``/``n_fine``/``frac_refined``.

    Notes
    -----
    The refinement is driven by the residual ``E - reconstruct(coarse)`` -- i.e.
    it refines wherever the **coarse** grid is inadequate, which is a sharp
    feature ONLY when ``base_step`` already resolves the smooth structure.  If
    ``base_step`` is too coarse to represent even the smooth field (e.g. a
    broad Gaussian body under a very coarse grid), the residual is large
    everywhere and refinement is broad -- pick ``base_step`` fine enough for the
    smooth regions so the fine grid concentrates on the true edges.

    This is a cost-saver for fields with **localized** sharp features on an
    otherwise well-resolved background (structured sources, hard input
    apertures propagated in free space).  It is NOT a good fit for a full-pupil
    **focusing** system, where every pupil beamlet contributes to the focus and
    coarsening the interior degrades the focal field -- there, use a uniform
    fine grid + ``apply_aperture_to_beamlets(..., soft_edge=True)``.

    NumPy backend (the boolean cell masking / concatenation is host-side); for
    JAX / CuPy use the uniform :func:`decompose_field_to_beamlets`.
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'decompose_field_adaptive',
                           input_kind='field')
    xp = array_namespace(E_in)
    if xp is not np:
        raise NotImplementedError(
            'decompose_field_adaptive is NumPy-only (host-side cell masking); '
            'use decompose_field_to_beamlets for JAX / CuPy.')
    if dy is None:
        dy = dx
    if refine_step >= base_step:
        raise ValueError('refine_step must be < base_step')
    Ny, Nx = E_in.shape[-2], E_in.shape[-1]

    # --- coarse bundle: uniform base grid, ALL cells (kept in full) ---
    coarse = decompose_field_to_beamlets(
        E_in, dx, wavelength=wavelength, dy=dy,
        waist_factor=target_overlap * base_step,
        sample_step=base_step, direction_sampling=direction_sampling)

    # --- residual = E_in - reconstruct(coarse) at the input plane ---
    # The residual is large exactly where the coarse grid under-resolves the
    # field (hard-aperture rims, amplitude/phase steps) and ~0 in smooth
    # regions.  Refining it (rather than dropping+replacing coarse cells with a
    # mismatched waist) keeps a clean partition of unity: the fine beamlets ADD
    # a correction, so reconstruct(coarse ++ fine) = reconstruct(coarse) +
    # reconstruct(fine) ~ E_c + residual = E_in wherever the residual is
    # refined -- no coarse/fine seam.
    E_c = reconstruct_field_from_beamlets(
        coarse, Ny=Ny, Nx=Nx, dx=dx, dy=dy, wavelength=wavelength,
        window=_GBD_RECONSTRUCT_WINDOW)
    residual = E_in - E_c

    # Indicator on the RESIDUAL: where the coarse fit is worst.  (The
    # 'gradient'/'amplitude' selector only chooses how the residual magnitude
    # is measured; residual |.| already localizes the sharp features.)
    if indicator == 'amplitude':
        s = np.abs(np.abs(E_in) - np.abs(E_c))
    else:
        s = np.abs(residual)
    smax = float(s.max()) if s.size else 0.0
    thr = refine_ratio * smax

    # --- fine bundle: uniform refine grid over the residual, keep only cells
    #     whose residual exceeds the threshold (the feature set) ---
    fine = decompose_field_to_beamlets(
        residual, dx, wavelength=wavelength, dy=dy,
        waist_factor=target_overlap * refine_step,
        sample_step=refine_step, direction_sampling=direction_sampling)
    iy_f = np.arange(0, Ny, refine_step)
    ix_f = np.arange(0, Nx, refine_step)
    Iy_f, Ix_f = np.meshgrid(iy_f, ix_f, indexing='ij')
    keep_f = (s[Iy_f.reshape(-1), Ix_f.reshape(-1)] > thr) if smax > 0.0 \
        else np.zeros(Iy_f.size, dtype=bool)

    def _subset(bundle, mask):
        return BeamletBundle(
            positions=bundle.positions[mask],
            directions=(bundle.directions[mask]
                        if bundle.directions is not None else None),
            Q=bundle.Q[mask], amplitude=bundle.amplitude[mask],
            waist0=bundle.waist0[mask])

    out = _concat_bundles(coarse, _subset(fine, keep_f))
    if return_stats:
        flag = np.zeros((Ny, Nx), dtype=bool)
        flag[s > thr] = True
        stats = {
            'n_coarse': int(len(coarse)),
            'n_fine': int(keep_f.sum()),
            'n_total': int(len(coarse) + keep_f.sum()),
            'frac_refined': float(flag.mean()),
        }
        return out, stats
    return out


# ============================================================================
# ABCD evolution
# ============================================================================

def propagate_beamlets_freespace(
    beamlets: BeamletBundle,
    z_distance: float,
    wavelength: float,
    *,
    n_medium: float = 1.0,
) -> BeamletBundle:
    """Advance every beamlet by free-space distance ``z_distance``."""
    xp = array_namespace(beamlets.positions)

    Nz = beamlets.directions[..., 2]
    eps = 1e-30
    t = z_distance / xp.where(xp.abs(Nz) > eps, Nz, eps)

    new_positions = beamlets.positions + t[..., None] * beamlets.directions

    Q_old = beamlets.Q
    k = 2 * float(np.pi) / wavelength * n_medium
    # 4.9 fix (audit #2.2): use raw ``t`` (signed) instead of
    # ``abs(t)``.  Under the exp(-iωt) time convention forward
    # propagation by distance z imparts exp(+i·k·z) -- correct for
    # both signs of z.  Pre-4.9 ``abs(t)`` accidentally took the
    # complex conjugate of the axial phase on back-propagation,
    # giving wrong sign on the propagated wavefront.  Forward
    # propagation was unaffected (because abs(positive) == positive).
    axial_phase = xp.exp(1j * k * t)
    if _q_is_tensor(Q_old):
        # v5.21: tensor (astigmatic / anamorphic) Q free-space -- matrix
        # Mobius Q_new = Q (I + t Q)^{-1} with a per-eigenvalue branch-safe
        # amplitude prod_i 1/sqrt(1 + t lambda_i) (eigenvalues have Im<0 for a
        # physical beam, so each principal sqrt is continuous through a focus).
        I2 = xp.eye(2, dtype=Q_old.dtype)[None, :, :]
        tt = t[:, None, None].astype(Q_old.dtype)
        Q_new = Q_old @ _inv2x2(I2 + tt * Q_old, xp)
        Q_new = 0.5 * (Q_new + xp.transpose(Q_new, (0, 2, 1)))
        lam = _eigvals2x2(Q_old, xp)
        _guard_tensor_freespace_branch(lam, t, xp)
        qratio = xp.prod(
            1.0 / xp.sqrt(1.0 + t[:, None].astype(Q_old.dtype) * lam), axis=1)
    else:
        Q_new = Q_old / (1 + t.astype(Q_old.dtype) * Q_old)
        qratio = Q_new / Q_old
    new_amplitude = beamlets.amplitude * qratio * axial_phase.astype(Q_old.dtype)

    return BeamletBundle(
        positions=new_positions,
        directions=beamlets.directions,
        Q=Q_new,
        amplitude=new_amplitude,
        waist0=beamlets.waist0,
    )


def apply_thin_lens_to_beamlets(
    beamlets: BeamletBundle,
    focal_length: float,
    wavelength: float,
    *,
    centre: Tuple[float, float] = (0.0, 0.0),
) -> BeamletBundle:
    """Apply an ideal thin lens to every beamlet."""
    xp = array_namespace(beamlets.positions)
    cx, cy = centre

    if _q_is_tensor(beamlets.Q):
        # v5.21: tensor Q -- an ideal (rotationally-symmetric) lens subtracts
        # 1/f from BOTH principal curvatures, i.e. (1/f) I2.
        I2 = xp.eye(2, dtype=beamlets.Q.dtype)[None, :, :]
        Q_new = beamlets.Q - (1.0 / focal_length) * I2
    else:
        Q_new = beamlets.Q - (1.0 / focal_length)

    x_off = beamlets.positions[..., 0] - cx
    y_off = beamlets.positions[..., 1] - cy
    L_old = beamlets.directions[..., 0]
    M_old = beamlets.directions[..., 1]
    N_old = beamlets.directions[..., 2]
    # 4.10.2: the thin-lens kick acts on PARAXIAL SLOPES u = L/N, not on
    # direction cosines.  Pre-4.10.2 subtracted x/f directly from L,
    # which is only correct in the small-angle limit (N -> 1).  For
    # moderately non-paraxial bundles (N ~ 0.95-0.99) this introduces
    # a few-percent error per surface; for wide-angle fans the error
    # compounds.  Convert to slope, apply the kick, re-normalise.
    N_safe = xp.where(xp.abs(N_old) > 1e-30, N_old, 1e-30)
    u_x_old = L_old / N_safe
    u_y_old = M_old / N_safe
    u_x_new = u_x_old - x_off / focal_length
    u_y_new = u_y_old - y_off / focal_length

    norm = xp.sqrt(u_x_new ** 2 + u_y_new ** 2 + 1.0)
    L_new = u_x_new / norm
    M_new = u_y_new / norm
    N_new = 1.0 / norm
    new_direction = xp.stack([L_new, M_new, N_new], axis=-1)

    k = 2 * float(np.pi) / wavelength
    lens_phase = xp.exp(-1j * k * (x_off * x_off + y_off * y_off) / (2 * focal_length))
    new_amplitude = beamlets.amplitude * lens_phase.astype(beamlets.amplitude.dtype)

    return BeamletBundle(
        positions=beamlets.positions,
        directions=new_direction,
        Q=Q_new,
        amplitude=new_amplitude,
        waist0=beamlets.waist0,
    )


def apply_aperture_to_beamlets(
    beamlets: BeamletBundle,
    semi_diameter: float,
    *,
    centre: Tuple[float, float] = (0.0, 0.0),
    shape: str = 'circular',
    soft_edge: bool = False,
    wavelength: Optional[float] = None,
) -> BeamletBundle:
    """Vignette a beamlet bundle at an aperture stop of half-size
    ``semi_diameter``.

    Beamlets whose base-ray transverse position falls outside the aperture
    have their amplitude zeroed (geometric / chief-ray vignetting).  This is
    the *approximate* GBD aperture model: because each beamlet is a smooth
    Gaussian, a hard edge is resolved only to the beamlet-spacing scale (a
    beamlet straddling the rim is kept or dropped whole).  For hard-edged
    apertures where the diffraction ripple matters at finer than the beamlet
    pitch, use finer ``sample_step`` (more, narrower beamlets),
    :func:`decompose_field_adaptive` (edge-refined beamlets), ``soft_edge=True``
    (below), or the HFPI propagator.  ``shape='circular'`` clips on
    ``sqrt(x^2+y^2)``; ``shape='rectangular'`` clips on ``max(|x|,|y|)``
    (``semi_diameter`` is the half-width).

    ``soft_edge=True`` (v5.21) replaces the binary chief-ray keep/drop with an
    **analytic partial-vignetting weight**: each beamlet is scaled by the
    fraction of its Gaussian that passes the aperture, using the local
    straight-edge approximation ``f = 1/2 (1 + erf(d * sqrt(2) / w))`` where
    ``d = semi_diameter - r_edge`` is the signed distance from the beamlet
    centre to the rim (``> 0`` inside) and ``w`` is the beamlet amplitude-1/e
    radius *at the aperture plane*.  A beamlet straddling the rim then contributes
    partially instead of all-or-nothing, which removes the beamlet-pitch
    staircase at the edge and improves the hard-aperture / Airy-focus accuracy
    (valid when ``w`` is small vs the aperture radius, i.e. the rim is locally
    straight).  ``soft_edge=False`` (default) keeps the exact binary behaviour.

    .. versionchanged:: 5.24.4
        The soft-edge width ``w`` is now the *propagated* amplitude-1/e radius
        derived from ``Im(Q)`` (identically to :func:`reconstruct_field_from_
        beamlets`) when ``wavelength`` is supplied, rather than the launch-time
        ``waist0``.  Because the stop is applied *after* the source->lens
        free-space leg, a beamlet that has diffracted to many Rayleigh ranges
        has a true width far larger than ``waist0`` (audit S2-5); using the
        stale ``waist0`` silently collapses the soft edge back to a near-hard
        cut whenever ``z_to_lens > 0``.  ``wavelength=None`` (the default when
        the routine is called standalone) preserves the old ``waist0`` fallback
        for backward compatibility.

    Parameters
    ----------
    beamlets : BeamletBundle
    semi_diameter : float
        Aperture half-size (radius for circular, half-width for rectangular).
    centre : (float, float), optional
        Aperture centre in the transverse plane (m).
    shape : {'circular', 'rectangular'}
    soft_edge : bool, default False
        Use the analytic partial-vignetting weight instead of a binary cut.
    wavelength : float, optional
        Free-space wavelength (m).  When given (and ``soft_edge=True``), the
        soft-edge width is the propagated amplitude-1/e radius from ``Im(Q)``;
        when ``None`` it falls back to the launch-time ``beamlets.waist0``.
    """
    xp = array_namespace(beamlets.positions)
    cx, cy = centre
    x = beamlets.positions[..., 0] - cx
    y = beamlets.positions[..., 1] - cy
    if shape not in ('circular', 'rectangular'):
        raise ValueError(
            f"apply_aperture_to_beamlets: shape must be 'circular' or "
            f"'rectangular', got {shape!r}")

    if soft_edge:
        # Signed distance d from beamlet centre to the nearest rim (>0 inside).
        # Beamlet amplitude-1/e radius w AT THE APERTURE PLANE.  With a
        # wavelength, derive the *propagated* width from Im(Q) (identically to
        # _reconstruct_windowed's R_cut block): the amplitude ~ exp(-alpha rho^2)
        # with alpha = 0.5 k lam_min, lam_min = -Im(Q) (scalar) or the smallest
        # eigenvalue of -Im(Q) (tensor, the widest axis), so w = 1/sqrt(alpha).
        # This equals waist0 at the launch waist (Q = -i/z_R) and grows as the
        # beamlet diffracts, fixing the stale-waist0 collapse (audit S2-5).
        # wavelength=None keeps the legacy waist0 for backward compatibility.
        if wavelength is not None:
            Q = beamlets.Q
            if _q_is_tensor(Q):
                D = -xp.imag(Q)                       # (n,2,2) real symmetric >0
                a11 = D[..., 0, 0]
                a22 = D[..., 1, 1]
                a12 = D[..., 0, 1]
                disc = xp.sqrt(xp.maximum((a11 - a22) ** 2 + 4.0 * a12 * a12,
                                          xp.zeros_like(a11)))
                lam_min = 0.5 * (a11 + a22 - disc)   # smallest eig of -Im(Q)
            else:
                lam_min = -xp.imag(Q)                 # scalar D
            k_wid = 2.0 * float(np.pi) / wavelength
            alpha_min = xp.maximum(0.5 * k_wid * lam_min,
                                   xp.full_like(lam_min, 1e-30))
            w = 1.0 / xp.sqrt(alpha_min)              # amplitude 1/e radius (m)
        else:
            w = beamlets.waist0
        eps = 1e-30
        w_safe = xp.where(xp.abs(w) > eps, w, xp.full_like(w, eps))
        if shape == 'circular':
            r = xp.sqrt(x * x + y * y)
            d = semi_diameter - r
        else:  # rectangular: distance to the nearest of the four straight edges
            d = semi_diameter - xp.maximum(xp.abs(x), xp.abs(y))
        # erf may be absent on some backends; use the numpy/scipy-free identity
        # via xp.special if present, else fall back to a math.erf vectorization.
        frac = 0.5 * (1.0 + _erf_xp(xp, d * float(np.sqrt(2.0)) / w_safe))
        weight = frac.astype(beamlets.amplitude.dtype)
    else:
        if shape == 'rectangular':
            inside = (xp.abs(x) <= semi_diameter) & (xp.abs(y) <= semi_diameter)
        else:
            inside = (x * x + y * y) <= (semi_diameter * semi_diameter)
        weight = inside.astype(beamlets.amplitude.dtype)
    new_amplitude = beamlets.amplitude * weight

    return BeamletBundle(
        positions=beamlets.positions,
        directions=beamlets.directions,
        Q=beamlets.Q,
        amplitude=new_amplitude,
        waist0=beamlets.waist0,
    )


def _erf_xp(xp: Any, z: np.ndarray) -> np.ndarray:
    """Vectorized erf dispatched across backends.

    Uses ``scipy.special.erf`` on NumPy, ``jax.scipy.special.erf`` /
    ``cupyx.scipy.special.erf`` where available, else an Abramowitz-&-Stegun
    7.1.26 rational approximation (max abs error ~1.5e-7) as a dependency-free
    fallback -- accuracy comfortably below the GBD propagator budget.
    """
    if xp is np:
        try:
            from scipy.special import erf as _erf
            return _erf(z)
        except ImportError:
            pass
    else:
        _sp = getattr(xp, 'scipy', None)
        if _sp is not None and hasattr(_sp, 'special') and hasattr(_sp.special, 'erf'):
            return _sp.special.erf(z)
    # A&S 7.1.26 rational approximation (real argument).
    sign = xp.sign(z)
    az = xp.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * az)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
                + t * (-1.453152027 + t * 1.061405429))))
    return sign * (1.0 - poly * xp.exp(-az * az))


# ============================================================================
# Reconstruction
# ============================================================================

def frame_completeness(
    bundle: 'BeamletBundle',
    E_ref: np.ndarray,
    dx: float,
    *,
    wavelength: float,
    dy: Optional[float] = None,
    centre: Tuple[float, float] = (0.0, 0.0),
    window: Optional[float] = 5.0,
    mem_budget_mb: float = 512.0,
) -> float:
    """Fraction of a reference field's power captured by a beamlet frame.

    Reconstructs ``bundle`` on ``E_ref``'s grid and returns
    ``sum|reconstruct|^2 / sum|E_ref|^2`` -- the P4 (N3) **frame-completeness
    metric**.  A value near 1 means the frame reproduces the field (a proper
    partition of unity); ``<< 1`` means the frame UNDER-spans the field (frame
    incompleteness -- the flat-waist beamlets cannot represent the field's local
    wavefront curvature, so their coherent sum sheds power); ``> 1`` means the
    frame OVER-counts (too much overlap).

    This isolates frame quality from downstream grid truncation when evaluated at
    the decomposition plane against the (aperture-clipped) input field.  NumPy
    host-side reduction; the reconstruction backend follows ``bundle``.
    """
    E_ref = np.asarray(E_ref)
    Ny, Nx = E_ref.shape[-2], E_ref.shape[-1]
    E_rec = reconstruct_field_from_beamlets(
        bundle, Ny=Ny, Nx=Nx, dx=dx, dy=(dx if dy is None else dy),
        wavelength=wavelength, centre=centre, window=window,
        mem_budget_mb=mem_budget_mb)
    p_ref = float(np.sum(np.abs(E_ref) ** 2))
    if p_ref <= 0.0:
        return 0.0
    p_rec = float(np.sum(np.abs(np.asarray(E_rec)) ** 2))
    return p_rec / p_ref


def reconstruct_field_from_beamlets(
    beamlets: BeamletBundle,
    *,
    Ny: int,
    Nx: int,
    dx: float,
    centre: Tuple[float, float] = (0.0, 0.0),
    wavelength: float,
    dy: Optional[float] = None,
    chunk_beamlets: int = 2048,
    window: Optional[float] = None,
    mem_budget_mb: float = 512.0,
) -> np.ndarray:
    """Coherently sum every beamlet's transverse profile on a 2-D
    output grid.  ``dy`` (default ``dx``) sets the y-axis output pitch for an
    anamorphic grid.

    ``window`` (v5.21, opt-in): if not ``None``, each beamlet is summed only
    over the local pixel box where its Gaussian is non-negligible (out to
    ``window`` amplitude-1/e radii, tail ``exp(-window^2)``), via a
    ``bincount`` scatter-add instead of the dense ``(Ny, Nx, chunk)`` product.
    This makes the cost ``O(beamlets * window_box)`` instead of
    ``O(beamlets * Ny * Nx)`` -- a large speed **and** memory win whenever the
    beamlets are localized (focused spots, near the decomposition plane, or
    moderate spread).  ``window=None`` (default) keeps the historical dense
    path **byte-identical**; ``window=5.0`` is a safe accuracy-preserving value
    (tail ~1e-11).  Only the NumPy backend takes the windowed path; JAX / CuPy
    fall back to the dense sum (data-dependent scatter indices are awkward /
    non-differentiable there).

    ``mem_budget_mb`` caps the dense path's per-chunk working set: if the
    ``(Ny, Nx, chunk_beamlets)`` complex buffer would exceed the budget,
    ``chunk_beamlets`` is auto-reduced.  This prevents the multi-GB peaks the
    fixed ``chunk_beamlets=2048`` hit at large ``N`` (13 GB at N=256) without
    changing small-``N`` results (the budget only bites when a chunk is large).
    The same budget caps the windowed (``window``) path's per-tile beamlet
    stack (B3); the ``LUMENAIRY_MEM_BUDGET_MB`` environment variable, if set, is
    a HARD CEILING on it for a memory-constrained host.
    """
    xp = array_namespace(beamlets.positions)
    cx, cy = centre
    if dy is None:
        dy = dx

    # v5.21: fast reconstruction paths.
    if window is not None:
        # #9: when the bundle is uniform-Q (scalar or diagonal), uniform-
        # direction and on-grid -- exactly what decompose + free-space produces
        # -- the coherent sum is a single Gaussian-kernel CONVOLUTION, an FFT of
        # O(Ny*Nx log) INDEPENDENT of beamlet count.  Machine-precision identical
        # to the dense/windowed sum (~1e-15) and orders of magnitude faster in
        # the spread/dense regime (measured ~2000-3700x).  Backend-generic
        # (#5: NumPy / JAX / CuPy), so GPU + jax.grad get it too.
        if _fft_reconstruct_applicable(beamlets, Nx, Ny, dx, dy, centre):
            return _reconstruct_fft(
                beamlets, xp=xp, Ny=Ny, Nx=Nx, dx=dx, dy=dy, centre=centre,
                wavelength=wavelength)
        # Windowed scatter-add: NumPy fast path (per-beamlet Q, skew, off-grid,
        # tilted-per-beamlet); JAX / CuPy non-uniform bundles fall to the dense
        # sum below (data-dependent windows are awkward there).
        if xp is np:
            return _reconstruct_windowed(
                beamlets, Ny=Ny, Nx=Nx, dx=dx, dy=dy, centre=centre,
                wavelength=wavelength, n_sigma=float(window),
                mem_budget_mb=mem_budget_mb)

    # audit S2-20: warn when the dense O(beamlets * Ny * Nx) path (window=None)
    # is about to run at a large workload -- the perf / memory footgun every
    # INTERNAL caller avoids by passing window=5.0.
    if window is None:
        _n_beamlets = int(np.asarray(beamlets.positions).shape[0])
        if float(Ny) * float(Nx) * float(_n_beamlets) > _GBD_DENSE_WARN_WORK:
            warnings.warn(
                "reconstruct_field_from_beamlets is using the dense "
                f"O(beamlets * Ny * Nx) path (window=None) with {_n_beamlets} "
                f"beamlets on a {Ny}x{Nx} grid. Pass window=5.0 for the "
                "bounded-support scatter-add (accuracy-preserving, tail "
                "~1e-11) to avoid the N^2 speed / memory footgun.",
                RuntimeWarning, stacklevel=2)

    # v5.21: auto-shrink chunk_beamlets to the memory budget.  bytes per
    # beamlet-column of the dense working set ~ Ny*Nx*16 (the dX/dY/rho2/phase
    # buffers); keep chunk*Ny*Nx*16 under mem_budget.  Never grows the chunk
    # (so small-N default runs stay byte-identical); only shrinks when a chunk
    # would blow the budget.
    if mem_budget_mb and mem_budget_mb > 0:
        _bytes_per_col = Ny * Nx * 16.0
        _max_chunk = max(1, int(mem_budget_mb * 1e6 / max(1.0, _bytes_per_col)))
        chunk_beamlets = min(chunk_beamlets, _max_chunk)

    ix = xp.arange(Nx, dtype=beamlets.positions.dtype)
    iy = xp.arange(Ny, dtype=beamlets.positions.dtype)
    Xg, Yg = xp.meshgrid((ix - Nx / 2) * dx + cx,
                         (iy - Ny / 2) * dy + cy,
                         indexing='xy')

    k = 2 * float(np.pi) / wavelength
    out = xp.zeros((Ny, Nx), dtype=beamlets.amplitude.dtype)

    n = int(beamlets.positions.shape[0])
    # Per-beamlet direction cosines (paraxial tilt).  These produce a
    # linear phase ramp `exp(i k (L dx + M dy))` from each beamlet's
    # centroid -- needed for non-paraxial bundles to interfere
    # correctly off-chief-ray.  Pre-4.10 omitted this ramp; the
    # focal spot still focused correctly (the chief-ray phase is the
    # same), but off-chief-ray interference patterns and PSF wings
    # were degraded.  When the beamlets bundle was assembled with
    # ``directions = (0, 0)`` (the default for axial-input decompositions)
    # the ramp is zero so this fix is a no-op for that path.
    has_dirs = (hasattr(beamlets, 'directions')
                and beamlets.directions is not None)
    # v5.21 (per-surface GBD): a beamlet bundle may carry a (N, 2, 2) complex-
    # symmetric TENSOR Q (general astigmatic Gaussian beam) instead of the (N,)
    # scalar Q.  Detected once here; the scalar branch below is preserved
    # verbatim (byte-identical) and only the tensor case takes the new quadratic
    # form.  ``_q_is_tensor`` is the single dispatch predicate.
    is_tensor = _q_is_tensor(beamlets.Q)
    # v4.13.1 perf: fuse the two ``xp.exp`` calls into one (only on the
    # has_dirs branch, where pre-v4.13.1 evaluated ``exp(-i*k*Q*rho2/2)``
    # and ``exp(i*k*tilt)`` separately and multiplied them).  ``exp(A) *
    # exp(B) == exp(A + B)`` analytically; in complex128 the round-off
    # difference is ulp-level (<1e-15 relative) -- well within the
    # propagator accuracy budget.  This roughly halves the per-chunk
    # transcendental cost (exp dominates the inner-loop runtime on
    # moderate grids).  Also switches the per-chunk reduction from
    # ``out + sum(a_b * phase, axis=-1)`` to ``out += einsum('mnk,k->mn',
    # phase, a_b)`` -- the ``a_b * phase`` intermediate is the
    # largest 3-D buffer the loop allocates (chunk * Ny * Nx complex),
    # so dropping it shrinks the working set noticeably for the
    # default chunk_beamlets=4096 and saves one big allocation per
    # chunk on numpy.
    for start in range(0, n, chunk_beamlets):
        end = min(start + chunk_beamlets, n)
        x_b = beamlets.positions[start:end, 0]
        y_b = beamlets.positions[start:end, 1]
        Q_b = beamlets.Q[start:end]
        a_b = beamlets.amplitude[start:end]

        dX = Xg[..., None] - x_b[None, None, :]
        dY = Yg[..., None] - y_b[None, None, :]
        rho2 = dX * dX + dY * dY
        if is_tensor:
            # General astigmatic Gaussian: transverse curvature is the
            # quadratic form 0.5 * conj(Q) : (rho rho^T) with the F-1 conj
            # (gbd.py physics convention) applied ELEMENTWISE to every Q
            # component, incl. the off-diagonal Qxy -- dropping conj on Qxy
            # would silently conjugate only the skew-astigmatism phase (an
            # intensity-invisible bug).  Q is symmetric so Qxy == Qyx.
            Qxx = Q_b[:, 0, 0]
            Qyy = Q_b[:, 1, 1]
            Qxy = Q_b[:, 0, 1]
            L_b = beamlets.directions[start:end, 0]
            M_b = beamlets.directions[start:end, 1]
            arg = (0.5 * (xp.conj(Qxx)[None, None, :] * (dX * dX)
                          + xp.conj(Qyy)[None, None, :] * (dY * dY)
                          + 2.0 * xp.conj(Qxy)[None, None, :] * (dX * dY))
                   + L_b[None, None, :] * dX + M_b[None, None, :] * dY)
            phase = xp.exp(1j * k * arg)
        elif has_dirs:
            L_b = beamlets.directions[start:end, 0]
            M_b = beamlets.directions[start:end, 1]
            # Fused phase argument.  v5.4.6 (audit F-1): the stored Q uses
            # the engineering 1/q parameterisation (q_code = conj(q_physics));
            # the reconstructed FIELD must be expressed in the library's
            # exp(-i omega t) / forward exp(+ikz) convention, i.e. the
            # transverse curvature is exp(+i k rho^2 / (2 q_physics)) =
            # exp(+0.5j k conj(Q) rho^2).  The pre-fix ``-0.5*Q`` rendered the
            # complex CONJUGATE of the propagated wavefront curvature (the
            # |E| envelope and waist are sign-blind, so intensity/focus tests
            # never caught it).  conj(Q) has the same Im part, so the Gaussian
            # decay and the on-axis-waist (Re(Q)=0) reconstruction are
            # unchanged; only the off-waist phase sign is corrected.
            arg = (0.5 * xp.conj(Q_b[None, None, :]) * rho2
                   + L_b[None, None, :] * dX + M_b[None, None, :] * dY)
            phase = xp.exp(1j * k * arg)
        else:
            phase = xp.exp(0.5j * k * xp.conj(Q_b[None, None, :]) * rho2)
        # ``out += einsum('mnk,k->mn', phase, a_b)`` if numpy; the
        # operator is equivalent to ``sum(a_b * phase, axis=-1)``
        # but avoids the (Ny, Nx, chunk) ``a_b * phase`` intermediate.
        # JAX / CuPy also have einsum; fall back to the original
        # pattern when einsum is unavailable.  In-place ``+=`` on
        # numpy / cupy avoids the per-chunk (Ny, Nx) allocation that
        # ``out = out + ...`` would create; JAX arrays are immutable
        # so we keep the rebind form for that backend.
        einsum = getattr(xp, 'einsum', None)
        if einsum is not None:
            contrib_sum = einsum('mnk,k->mn', phase, a_b)
        else:
            contrib = a_b[None, None, :] * phase
            contrib_sum = xp.sum(contrib, axis=-1)
        if is_jax_array(out):
            out = out + contrib_sum
        else:
            out += contrib_sum

    return out


def _env_mem_budget_mb():
    """Read the ``LUMENAIRY_MEM_BUDGET_MB`` environment override (or ``None`` when
    unset / invalid) -- the GBD sibling of
    :func:`lumenairy.propagators.fga._env_mem_budget_mb`.  A positive value acts
    as a HARD CEILING on the windowed-reconstruct per-chunk memory budget so a
    memory-constrained / shared host can force smaller accumulation tiles than a
    caller's ``mem_budget_mb`` requested, honoring the same env knob FGA's chunk
    loop respects.  A non-positive / unparseable value is ignored with a one-line
    warning so a typo does not silently disable the guard."""
    import os
    raw = os.environ.get('LUMENAIRY_MEM_BUDGET_MB')
    if raw is None or str(raw).strip() == '':
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        warnings.warn(
            f"LUMENAIRY_MEM_BUDGET_MB={raw!r} is not a number; ignoring it for "
            "the GBD windowed-reconstruct budget.",
            RuntimeWarning, stacklevel=2)
        return None
    if not (val > 0.0):
        warnings.warn(
            f"LUMENAIRY_MEM_BUDGET_MB={raw!r} is not positive; ignoring it.",
            RuntimeWarning, stacklevel=2)
        return None
    return val


# Peak simultaneous per-cell footprint of one windowed accumulation TILE
# (a ``(n_g, by, bx)`` beamlet x window-box stack).  After the v5.28 in-place
# masking (below) the coexisting big arrays are: ``contrib`` (complex128, 16 B),
# ``idx`` (int64, 8 B), plus the transient ``valid`` / ``bad`` bool masks
# (1 B each, freed before the bincount) -- ~26 B/cell.  32 B/cell is the
# chunk-sizing estimate with margin, so peak stays UNDER ``mem_budget_mb``.
_WINDOWED_CELL_BYTES = 32.0


def _reconstruct_windowed(
    beamlets: BeamletBundle,
    *,
    Ny: int,
    Nx: int,
    dx: float,
    dy: float,
    centre: Tuple[float, float],
    wavelength: float,
    n_sigma: float = 5.0,
    mem_budget_mb: float = 512.0,
) -> np.ndarray:
    """Windowed (bounded-support) coherent reconstruction, NumPy only.

    Each beamlet is a Gaussian whose amplitude decays as
    ``exp(-0.5 k (rho^T D rho))`` with ``D = -Im(Q)`` (positive-definite for a
    physical beam).  Beyond ``R_cut = n_sigma * w_amp`` (``w_amp`` = amplitude
    1/e radius along the widest principal axis, ``= 1/sqrt(alpha_min)`` with
    ``alpha_min = 0.5 k lambda_min(D)``) the contribution is ``< exp(-n_sigma^2)``
    and is dropped.  The beamlet's field is evaluated only on the local pixel
    box ``[+-Wy, +-Wx]`` around its centre and scatter-added into the output via
    :func:`numpy.bincount` -- cost ``O(sum_b window_box_b)`` rather than
    ``O(n_beamlets * Ny * Nx)``.  Truncation error is bounded by
    ``exp(-n_sigma^2)`` (``1.4e-11`` at the default ``n_sigma=5``), far below the
    GBD propagator accuracy budget, so the result matches the dense sum to
    machine-relevant precision.

    Memory (B3): the per-bucket beamlet stack is chunked so its peak transient
    footprint stays under ``mem_budget_mb`` (mirroring the FGA position-lattice
    chunk loop).  ``LUMENAIRY_MEM_BUDGET_MB`` (if set) is a HARD CEILING on that
    budget, so a memory-constrained host caps the peak regardless of the passed
    value.  With the env unset the default (``mem_budget_mb``) chunking is
    unchanged, so the output is byte-identical to the pre-B3 path; the win comes
    purely from dropping two redundant ``(n_g, by, bx)`` ``np.where`` copies in
    favour of in-place index/contribution masking (~2x lower peak, same result).
    """
    cx, cy = centre
    # B3: env ceiling on the per-chunk memory budget (never raises it).
    _env_budget = _env_mem_budget_mb()
    if _env_budget is not None:
        mem_budget_mb = (_env_budget if mem_budget_mb is None
                         else min(float(mem_budget_mb), _env_budget))
    k = 2.0 * float(np.pi) / wavelength
    out_dtype = beamlets.amplitude.dtype
    x_b = np.asarray(beamlets.positions[:, 0], dtype=np.float64)
    y_b = np.asarray(beamlets.positions[:, 1], dtype=np.float64)
    a_b = np.asarray(beamlets.amplitude)
    is_tensor = _q_is_tensor(beamlets.Q)
    has_dirs = (getattr(beamlets, 'directions', None) is not None)
    if has_dirs:
        L_all = np.asarray(beamlets.directions[:, 0], dtype=np.float64)
        M_all = np.asarray(beamlets.directions[:, 1], dtype=np.float64)

    # Per-beamlet decay coefficient alpha_min (widest axis) -> R_cut.
    if is_tensor:
        Q_all = np.asarray(beamlets.Q)            # (n, 2, 2) complex
        D = -np.imag(Q_all)                       # (n, 2, 2) real symmetric >0
        # smallest eigenvalue of each 2x2 (closed form, avoids per-beamlet eigh)
        a11 = D[:, 0, 0]
        a22 = D[:, 1, 1]
        a12 = D[:, 0, 1]
        tr = a11 + a22
        disc = np.sqrt(np.maximum((a11 - a22) ** 2 + 4.0 * a12 * a12, 0.0))
        lam_min = 0.5 * (tr - disc)
    else:
        Q_all = np.asarray(beamlets.Q)            # (n,) complex
        lam_min = -np.imag(Q_all)                 # scalar D

    alpha_min = 0.5 * k * lam_min
    alpha_floor = 1e-30
    alpha_min = np.maximum(alpha_min, alpha_floor)
    R_cut = n_sigma / np.sqrt(alpha_min)          # metres
    Wx_f = R_cut / dx
    Wy_f = R_cut / dy

    out_r = np.zeros(Ny * Nx + 1, dtype=np.float64)
    out_i = np.zeros(Ny * Nx + 1, dtype=np.float64)
    sentinel = Ny * Nx

    # Bucket beamlets by (Wx, Wy) quantized to a geometric ladder so a handful
    # of vectorized passes cover all support sizes.  Cap at the grid extent.
    # v5.21: round each half-width UP a sqrt(2)-geometric ladder (ceil to
    # 2^(k/2)) rather than the next power of two.  Power-of-two rounding
    # inflated each axis by up to ~2x (box AREA up to ~4x the exact R_cut box);
    # the finer sqrt(2) ladder caps per-axis inflation at ~1.41x (area ~2x),
    # halving the evaluated cells (hence exp + bincount work) for the cost of a
    # few more np.unique buckets -- each bucket is one vectorized pass, so the
    # extra buckets are cheap.
    def _quantize(Wf, cap):
        W = np.ceil(Wf).astype(np.int64)
        W = np.clip(W, 1, cap)
        # ceil to the next rung of the ladder {1,2,3,4,6,8,11,16,23,32,...}
        # = ceil(2^(ceil(2*log2 W)/2)); exact-integer via the doubled exponent.
        lg2 = np.ceil(2.0 * np.log2(np.maximum(W, 1))).astype(np.int64)
        Wq = np.ceil(np.power(2.0, 0.5 * lg2)).astype(np.int64)
        Wq = np.where(W <= 1, 1, Wq)
        return np.clip(Wq, 1, cap).astype(np.int64)

    Wx_q = _quantize(Wx_f, Nx)
    Wy_q = _quantize(Wy_f, Ny)

    # Centre pixel of each beamlet.
    ix0 = np.round((x_b - cx) / dx + Nx / 2.0).astype(np.int64)
    iy0 = np.round((y_b - cy) / dy + Ny / 2.0).astype(np.int64)

    # v5.21: the beamlet Gaussian argument is axis-SEPARABLE unless the tensor Q
    # has a non-zero off-diagonal (skew astigmatism): for scalar Q and diagonal
    # tensor Q,  0.5 conj(Q):rho rho + L dX + M dY  =  [x-only](q) + [y-only](p),
    # so exp(...) = exp(x-part) (x) exp(y-part) is an OUTER PRODUCT.  That cuts
    # the complex-exp count per beamlet from ng*by*bx to ng*(bx+by) (and skips
    # the (ng,by,bx) arg buffer) on the default scalar free-space / thin-lens
    # path.  Only the true skew case (Qxy != 0, per-surface aberration) keeps
    # the full 2-D form.  exp(a+b) == exp(a) exp(b) analytically (ULP-close).
    skew = bool(is_tensor and np.any(Q_all[:, 0, 1] != 0.0))

    keys = Wy_q.astype(np.int64) * (Nx + 1) + Wx_q.astype(np.int64)
    for key in np.unique(keys):
        sel = np.nonzero(keys == key)[0]
        Wx = int(Wx_q[sel[0]])
        Wy = int(Wy_q[sel[0]])
        bx = 2 * Wx + 1
        by = 2 * Wy + 1
        # B3: chunk this bucket so the peak (n_g, by, bx) accumulation tile stays
        # under mem_budget_mb.  _WINDOWED_CELL_BYTES counts the coexisting big
        # arrays (contrib complex + idx int64 + bool masks) with margin; the
        # estimate is unchanged from the pre-B3 32 B/cell, so with the default
        # budget the chunk boundaries -- and hence the output -- are identical.
        cells = by * bx
        chunk = max(1, int(mem_budget_mb * 1e6
                           / max(1.0, cells * _WINDOWED_CELL_BYTES)))
        off_x = np.arange(-Wx, Wx + 1, dtype=np.int64)
        off_y = np.arange(-Wy, Wy + 1, dtype=np.int64)
        for cs in range(0, sel.size, chunk):
            g = sel[cs:cs + chunk]
            ixg = ix0[g][:, None] + off_x[None, :]        # (ng, bx)
            iyg = iy0[g][:, None] + off_y[None, :]        # (ng, by)
            Xloc = (ixg - Nx / 2.0) * dx + cx
            Yloc = (iyg - Ny / 2.0) * dy + cy
            dX = Xloc - x_b[g][:, None]                   # (ng, bx)
            dY = Yloc - y_b[g][:, None]                   # (ng, by)
            if skew:
                # Full 2-D form (non-separable cross term 2 Qxy dX dY).
                dX2 = dX[:, None, :] ** 2                 # (ng, 1, bx)
                dY2 = dY[:, :, None] ** 2                 # (ng, by, 1)
                Qxx = np.conj(Q_all[g, 0, 0])[:, None, None]
                Qyy = np.conj(Q_all[g, 1, 1])[:, None, None]
                Qxy = np.conj(Q_all[g, 0, 1])[:, None, None]
                arg = 0.5 * (Qxx * dX2 + Qyy * dY2
                             + 2.0 * Qxy * (dX[:, None, :] * dY[:, :, None]))
                if has_dirs:
                    arg = (arg + L_all[g][:, None, None] * dX[:, None, :]
                           + M_all[g][:, None, None] * dY[:, :, None])
                contrib = a_b[g][:, None, None] * np.exp(1j * k * arg)
                del arg          # B3: free the (n_g, by, bx) skew arg buffer
            else:
                # Separable: exp of an outer sum -> outer product of two exps.
                if is_tensor:                            # diagonal tensor Q
                    cqx = np.conj(Q_all[g, 0, 0])
                    cqy = np.conj(Q_all[g, 1, 1])
                else:                                    # scalar Q
                    cqx = np.conj(Q_all[g])
                    cqy = cqx
                argx = 0.5 * cqx[:, None] * (dX * dX)     # (ng, bx)
                argy = 0.5 * cqy[:, None] * (dY * dY)     # (ng, by)
                if has_dirs:
                    argx = argx + L_all[g][:, None] * dX
                    argy = argy + M_all[g][:, None] * dY
                gx = np.exp(1j * k * argx)                # (ng, bx)
                gy = np.exp(1j * k * argy)                # (ng, by)
                contrib = (a_b[g][:, None, None]
                           * gy[:, :, None]) * gx[:, None, :]   # (ng,by,bx)

            idx = iyg[:, :, None] * Nx + ixg[:, None, :]            # (ng,by,bx)
            valid = ((iyg[:, :, None] >= 0) & (iyg[:, :, None] < Ny)
                     & (ixg[:, None, :] >= 0) & (ixg[:, None, :] < Nx))
            # B3: mask out-of-grid cells IN PLACE (route to the sentinel bin,
            # zero their contribution) instead of building two extra
            # (ng, by, bx) np.where copies.  idx / contrib are freshly-built
            # C-contiguous arrays so .reshape(-1) is a view and the in-place
            # writes hit their own buffers -- the bincount inputs are
            # byte-identical to the np.where form, at ~2x lower peak memory.
            idx_flat = idx.reshape(-1)
            cflat = contrib.reshape(-1)
            bad = ~valid.reshape(-1)
            del valid
            idx_flat[bad] = sentinel
            cflat[bad] = 0.0
            del bad
            out_r += np.bincount(idx_flat, weights=cflat.real,
                                 minlength=Ny * Nx + 1)
            out_i += np.bincount(idx_flat, weights=cflat.imag,
                                 minlength=Ny * Nx + 1)
            # B3: free this tile's big (ng, by, bx) buffers before the next
            # chunk allocates, so old- and new-iteration stacks never coexist
            # (the cross-iteration double-buffering that dominated the peak).
            del idx, idx_flat, contrib, cflat

    out = (out_r[:Ny * Nx] + 1j * out_i[:Ny * Nx]).reshape(Ny, Nx)
    return out.astype(out_dtype)


def _fft_reconstruct_applicable(beamlets: BeamletBundle, Nx: int, Ny: int,
                                dx: float, dy: float,
                                centre: Tuple[float, float]) -> bool:
    """True when the FFT-convolution reconstruction (:func:`_reconstruct_fft`)
    applies: a UNIFORM ``Q`` (scalar, or diagonal tensor with equal entries and
    no skew), a UNIFORM launch direction, and beamlet centres that land on the
    output grid.  This is exactly what :func:`decompose_field_to_beamlets`
    followed by free-space / uniform-ABCD evolution produces (positions
    unchanged for an axial bundle, ``Q`` and amplitude evolve identically).

    Whether it applies is a data-dependent DECISION (is ``Q`` uniform? on-grid?)
    and so cannot be taken under a JAX ``jit`` trace, where the bundle arrays are
    abstract tracers -- inspecting them (``np.asarray``) raises.  In that case
    (and on any inspection failure) return ``False`` so the reconstruction falls
    back to the trace-safe dense sum; the FFT fast path still engages for
    concrete arrays (eager execution and under ``jax.grad``)."""
    try:
        return _fft_applicable_impl(beamlets, Nx, Ny, dx, dy, centre)
    except Exception:      # jax tracer (jit) / non-inspectable array -> dense
        return False


def _fft_applicable_impl(beamlets, Nx, Ny, dx, dy, centre) -> bool:
    n = int(beamlets.positions.shape[0])
    if n == 0:
        return False
    # uniform launch direction (a global tilt is fine; per-beamlet tilt is not)
    dr = getattr(beamlets, 'directions', None)
    if dr is not None:
        dr = np.asarray(dr)
        if not (np.ptp(dr[:, 0]) <= 1e-12 and np.ptp(dr[:, 1]) <= 1e-12):
            return False
    Q = np.asarray(beamlets.Q)
    if _q_is_tensor(beamlets.Q):
        if np.max(np.abs(Q[:, 0, 1])) > 1e-30:           # skew astigmatism
            return False
        if not (np.allclose(Q[:, 0, 0], Q[0, 0, 0], rtol=1e-9, atol=1e-30)
                and np.allclose(Q[:, 1, 1], Q[0, 1, 1], rtol=1e-9, atol=1e-30)):
            return False
    else:
        if not np.allclose(Q, Q[0], rtol=1e-9, atol=1e-30):
            return False
    # beamlet centres must fall on the output grid (else the scatter rounds)
    cx, cy = centre
    fx = (np.asarray(beamlets.positions[:, 0]) - cx) / dx + Nx / 2.0
    fy = (np.asarray(beamlets.positions[:, 1]) - cy) / dy + Ny / 2.0
    if (np.max(np.abs(fx - np.round(fx))) > 1e-4
            or np.max(np.abs(fy - np.round(fy))) > 1e-4):
        return False
    return True


def _fftconv_same(xp: Any, a: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Linear (zero-padded) 2-D convolution of ``a`` with ``G``, returning the
    ``mode='same'`` central slice aligned to ``a`` -- backend-generic via
    ``xp.fft`` (matches ``scipy.signal.fftconvolve(a, G, 'same')`` to ~6e-16)."""
    Ny, Nx = a.shape
    Gy, Gx = G.shape
    sy, sx = Ny + Gy - 1, Nx + Gx - 1
    F = xp.fft.ifft2(xp.fft.fft2(a, s=(sy, sx)) * xp.fft.fft2(G, s=(sy, sx)))
    sy0, sx0 = (Gy - 1) // 2, (Gx - 1) // 2
    return F[sy0:sy0 + Ny, sx0:sx0 + Nx]


def _scatter_grid(xp: Any, Ny: int, Nx: int, flat: np.ndarray,
                  vals: np.ndarray) -> np.ndarray:
    """Scatter-add complex ``vals`` into an ``(Ny, Nx)`` grid at flat indices
    ``flat`` -- ``jnp.at[].add`` for JAX (static shape), real/imag ``bincount``
    for NumPy / CuPy."""
    if is_jax_array(vals):
        g = xp.zeros(Ny * Nx, dtype=vals.dtype).at[flat].add(vals)
    else:
        gr = xp.bincount(flat, weights=vals.real, minlength=Ny * Nx)[:Ny * Nx]
        gi = xp.bincount(flat, weights=vals.imag, minlength=Ny * Nx)[:Ny * Nx]
        g = gr + 1j * gi
    return g.reshape(Ny, Nx)


def _reconstruct_fft(beamlets: BeamletBundle, *, xp: Any, Ny: int, Nx: int,
                     dx: float, dy: float, centre: Tuple[float, float],
                     wavelength: float) -> np.ndarray:
    """FFT-convolution reconstruction -- backend-generic (NumPy / JAX / CuPy).

    For a uniform-``Q``, uniform-direction, on-grid bundle the coherent sum
    ``E(p) = sum_b a_b exp(0.5j k conj(Q):(p-b)^2 + i k dir.(p-b))`` is the
    convolution of the amplitude array (``a_b`` scattered onto the grid) with a
    single Gaussian kernel ``G``.  Evaluated as one linear (zero-padded) FFT
    convolution -- ``O(Ny*Nx log(Ny*Nx))`` INDEPENDENT of the beamlet count,
    vs the ``O(n_beamlets * Ny * Nx)`` dense/windowed sum which the window path
    degrades to once the beamlets spread to fill the grid.  Machine-precision
    identical to the dense sum (~1e-15); the FFT + scatter run entirely on the
    backend (GPU under CuPy, and it is ``jax.grad`` / ``jit`` differentiable
    under JAX).  Applicability is gated by :func:`_fft_reconstruct_applicable`.
    """
    k = 2.0 * float(np.pi) / wavelength
    cx, cy = centre
    a = beamlets.amplitude
    out_dtype = a.dtype
    xb = beamlets.positions[:, 0]
    yb = beamlets.positions[:, 1]
    # Uniform-Q / uniform-direction scalars (one host element read; cheap).
    if _q_is_tensor(beamlets.Q):
        Qh = np.asarray(beamlets.Q[0])
        cqx = np.conj(complex(Qh[0, 0]))
        cqy = np.conj(complex(Qh[1, 1]))
    else:
        cqx = cqy = np.conj(complex(np.asarray(beamlets.Q).reshape(-1)[0]))
    dr = getattr(beamlets, 'directions', None)
    if dr is not None:
        drh = np.asarray(dr[0])
        L0, M0 = float(drh[0]), float(drh[1])
    else:
        L0, M0 = 0.0, 0.0

    # Scatter the beamlet amplitudes onto the grid (static-shape safe: clamp OOB
    # indices and zero their value rather than boolean-mask).
    ix = xp.round((xb - cx) / dx + Nx / 2.0).astype(xp.int32)
    iy = xp.round((yb - cy) / dy + Ny / 2.0).astype(xp.int32)
    valid = (ix >= 0) & (ix < Nx) & (iy >= 0) & (iy < Ny)
    vals = xp.where(valid, a, xp.zeros_like(a))
    flat = xp.clip(iy, 0, Ny - 1) * Nx + xp.clip(ix, 0, Nx - 1)
    a_grid = _scatter_grid(xp, Ny, Nx, flat, vals)

    # Gaussian kernel over the full linear-convolution offset range, centred.
    ux = (xp.arange(2 * Nx - 1) - (Nx - 1)) * dx
    uy = (xp.arange(2 * Ny - 1) - (Ny - 1)) * dy
    UX, UY = xp.meshgrid(ux, uy)
    G = xp.exp(1j * k * (0.5 * (cqx * UX * UX + cqy * UY * UY)
                         + L0 * UX + M0 * UY))
    out = _fftconv_same(xp, a_grid, G)
    return out.astype(out_dtype)


def _bundle_to_backend(bundle: 'BeamletBundle', xp: Any) -> 'BeamletBundle':
    """Move every array of a :class:`BeamletBundle` to the namespace ``xp``
    (numpy / cupy / jax.numpy).  Used to run the coherent reconstruction on the
    GPU after the (NumPy) per-surface evolution."""
    from ..backend.array import to_backend
    return BeamletBundle(
        positions=to_backend(bundle.positions, xp),
        directions=(to_backend(bundle.directions, xp)
                    if bundle.directions is not None else None),
        Q=to_backend(bundle.Q, xp),
        amplitude=to_backend(bundle.amplitude, xp),
        waist0=to_backend(bundle.waist0, xp),
    )


def _gpu_namespace():
    """Return the CuPy namespace, or raise a clear error if unavailable."""
    try:
        import cupy as _cp
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            'use_gpu=True requires CuPy (and a CUDA device); it is not '
            'installed.  Install cupy-cudaXX or use the default CPU path.'
        ) from exc
    return _cp


# ============================================================================
# End-to-end convenience
# ============================================================================

def propagate_gbd(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    **kwargs: Any,
) -> np.ndarray:
    """Canonical-order GBD free-space propagation.

    Argument order ``(E_in, z, wavelength, dx)`` matches
    :func:`angular_spectrum_propagate` and
    :func:`propagate_huygens_fresnel`.  This is the recommended entry
    point for new code.  Internally delegates to
    :func:`propagate_gbd_freespace` (which retains its legacy
    ``(E_in, dx, *, z, wavelength, ...)`` order for backwards
    compatibility).
    """
    return propagate_gbd_freespace(
        E_in, dx, z=z, wavelength=wavelength, **kwargs)


def propagate_gbd_freespace(
    E_in: np.ndarray,
    dx: float,
    *,
    z: float,
    wavelength: float,
    output_shape: Optional[Tuple[int, int]] = None,
    output_grid: Optional[Tuple[int, int]] = None,
    output_dx: Optional[float] = None,
    output_centre: Tuple[float, float] = (0.0, 0.0),
    dy: Optional[float] = None,
    output_dy: Optional[float] = None,
    waist_factor: float = 1.0,
    sample_step: int = 1,
    chunk_beamlets: int = 2048,
    direction_sampling: bool = False,
) -> np.ndarray:
    """End-to-end free-space GBD: source -> z -> output.

    ``direction_sampling=True`` selects the Husimi (position + direction)
    decomposition -- needed for a source that is already tilted / diverging
    at the input plane; see :func:`decompose_field_to_beamlets`.

    ``dy`` (source y-pitch, default ``dx``) and ``output_dy`` (output y-pitch,
    default ``output_dx``) support **anamorphic** grids: ``dy != dx`` makes each
    beamlet an axis-aligned ellipse (a diagonal tensor ``Q``).

    .. note::
       This function uses a non-canonical argument order
       ``(E_in, dx, *, z, wavelength, ...)``.  Prefer
       :func:`propagate_gbd` for the canonical
       ``(E_in, z, wavelength, dx)`` order shared with
       :func:`angular_spectrum_propagate` and
       :func:`propagate_huygens_fresnel`.

    .. versionchanged:: 5.2
        ``output_grid`` was renamed to ``output_shape`` for the
        ``(Ny, Nx)`` shape-only meaning -- the dispatcher's
        ``propagate(output_grid=...)`` is canonical for the
        ``(N_out, dx_out)`` grid spec (AUDIT_V4_13_1 Part 2 P1-A).
        The legacy ``output_grid`` form still works but emits a
        ``DeprecationWarning``.
    """
    Ny, Nx = _resolve_output_shape(
        output_shape, output_grid,
        fn_name='propagate_gbd_freespace',
        default_shape=(E_in.shape[-2], E_in.shape[-1]),
    )
    if output_dx is None:
        output_dx = dx
    if output_dy is None:
        output_dy = dy if dy is not None else output_dx

    _warn_if_tilted_no_dirsample(E_in, dx, dy, wavelength, direction_sampling)
    bundle = decompose_field_to_beamlets(
        E_in, dx, wavelength=wavelength, dy=dy,
        waist_factor=waist_factor,
        sample_step=sample_step,
        direction_sampling=direction_sampling,
    )
    bundle = propagate_beamlets_freespace(bundle, z_distance=z,
                                          wavelength=wavelength)
    return reconstruct_field_from_beamlets(
        bundle, Ny=Ny, Nx=Nx, dx=output_dx, dy=output_dy,
        centre=output_centre, wavelength=wavelength,
        chunk_beamlets=chunk_beamlets, window=_GBD_RECONSTRUCT_WINDOW,
    )


# ============================================================================
# GBD <-> ASM interoperability
# ============================================================================

def gbd_asm_gouy_phase(z: float, wavelength: float, dx: float,
                       waist_factor: float = 1.0) -> float:
    """Global phase offset between a GBD-reconstructed **free-space** field and
    the Angular-Spectrum (ASM) / physical-phase field of the same beam.

    A GBD field and an ASM field of the same physical beam are **the same
    field** -- same transverse coordinates, same forward (``exp(+ikz)``)
    direction, no complex conjugate and no axis flip -- differing ONLY by a
    spatially-constant phase

        ``phi0(z) = 2 * arctan(z / zR_b)``,   ``zR_b = pi (waist_factor*dx)^2 / lambda``

    i.e. **twice the beamlet Gouy phase** (``zR_b`` is the Rayleigh range of a
    single decomposition beamlet of waist ``waist_factor*dx``).  GBD's
    finite-waist beamlets each accumulate this Gouy phase on propagation and it
    survives the coherent sum as a global factor; ASM (which propagates the
    field spectrally, with no beamlet basis) does not carry it.  It is
    field-independent and, being a global phase, is physically irrelevant to the
    intensity and to any *further* linear propagation -- so a GBD free-space
    field can be handed straight to :func:`angular_spectrum_propagate` (the
    global phase merely rides along).  Reconcile it only when you need the
    ABSOLUTE phase to agree (e.g. coherently combining a GBD field with an
    independently-computed ASM field).

    Verified: the formula matches the measured overlap phase to ~1e-7 across
    waist factors and non-trivial (off-centre / tilted / astigmatic) fields, and
    after removing it the GBD field matches ``angular_spectrum_propagate`` to the
    GBD decomposition accuracy (~1e-3 here, no free phase fit).

    .. note::
       This closed form is for a **pure free-space** GBD leg
       (:func:`propagate_gbd_freespace`).  A GBD field that has passed through
       optics (``propagate_gbd_through_prescription``) carries a different,
       path-dependent global phase (the accumulated per-surface Gouy / OPL
       piston); the two fields still agree in intensity and up to a global
       phase, but this specific closed form does not apply.  ``waist_factor``
       and ``z`` must match the ``propagate_gbd_freespace`` call that produced
       the field.
    """
    zR_b = float(np.pi) * (waist_factor * dx) ** 2 / wavelength
    return 2.0 * float(np.arctan(z / zR_b))


def gbd_field_to_asm(E: np.ndarray, *, z: float, wavelength: float, dx: float,
                     waist_factor: float = 1.0) -> np.ndarray:
    """Convert a GBD free-space reconstructed field to the ASM / physical-phase
    convention by removing the beamlet-Gouy global phase
    (:func:`gbd_asm_gouy_phase`).  After conversion the field matches
    :func:`angular_spectrum_propagate` to the GBD decomposition accuracy, so the
    two propagators are interchangeable (coherent combination, GBD->ASM
    handoff).  ``z`` / ``waist_factor`` must match the ``propagate_gbd_freespace``
    call that produced ``E``."""
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E, 'gbd_field_to_asm', input_kind='field')
    xp = array_namespace(E)
    return E * xp.exp(-1j * gbd_asm_gouy_phase(z, wavelength, dx, waist_factor))


def asm_field_to_gbd(E: np.ndarray, *, z: float, wavelength: float, dx: float,
                     waist_factor: float = 1.0) -> np.ndarray:
    """Convert an ASM / physical-phase field to the GBD free-space
    reconstruction convention (add the beamlet-Gouy global phase) -- the inverse
    of :func:`gbd_field_to_asm`.  Use when feeding an ASM field into a GBD
    pipeline that expects GBD's convention for absolute-phase consistency; note
    that decomposing an ASM field into beamlets
    (:func:`decompose_field_to_beamlets`) and continuing with GBD does NOT need
    this (decomposition + reconstruction is self-consistent in GBD's
    convention)."""
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E, 'asm_field_to_gbd', input_kind='field')
    xp = array_namespace(E)
    return E * xp.exp(1j * gbd_asm_gouy_phase(z, wavelength, dx, waist_factor))


def match_global_phase(E: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Reconcile the **global** (spatially-constant) phase of ``E`` to a
    ``reference`` field of the same beam on the same grid: returns
    ``E * exp(-i phi)`` with ``phi = arg <reference, E>`` so the two share the
    same absolute phase (``<reference, E_out>`` is real, positive).

    The general, propagator-agnostic interoperability primitive.  Every coherent
    propagator in the library (ASM, GBD, Maslov, Fresnel, RS, HFPI) uses the same
    physical convention -- same transverse coordinates, forward ``exp(+ikz)``,
    NO complex conjugate and NO axis flip -- so two fields of the same beam on
    the same plane differ, to leading order, by a global phase.  Reconcile it
    only for ABSOLUTE-phase agreement (coherent superposition / phase-sensitive
    comparison); intensity and further linear propagation are unaffected.

    .. note::
       This removes only the GLOBAL phase.  It is EXACT for the GBD-free-space
       vs ASM difference (which is purely global -- and there the closed-form
       :func:`gbd_asm_gouy_phase` needs no reference).  It does NOT reconcile the
       higher-order (wavefront) difference between fields produced by
       *different-order approximations* of the same beam -- e.g. the phase-space
       ``apply_real_lens_maslov`` (default ``normalize_output='power'`` gives a
       physical-amplitude field) and the paraxial-per-surface GBD agree in the
       intensity envelope and convention family but differ at the ~10-25% level
       in the complex field for the same lens (they are different approximations,
       not the same field).  For a chief-relative vs absolute *wavefront*
       convention difference use
       :func:`lumenairy.propagators.asm.apply_fresnel_curvature`.
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E, 'match_global_phase', input_kind='field')
    xp = array_namespace(E)
    ip = xp.sum(xp.conj(reference) * E)
    return E * (xp.conj(ip) / (xp.abs(ip) + 1e-300))


def propagate_gbd_freespace_spectral(
    E_in: np.ndarray,
    dx: float,
    *,
    z: float,
    wavelengths: Any,
    weights: Optional[Any] = None,
    combine: str = 'stack',
    **freespace_kwargs: Any,
) -> np.ndarray:
    """Polychromatic free-space GBD: propagate ``E_in`` at each wavelength.

    Because the beamlet complex parameter ``Q = -i/z_R``, its Rayleigh range
    ``z_R = pi w0^2 / lambda`` and the reconstruction phase ``k`` are all
    wavelength-dependent, each wavelength is decomposed and propagated
    independently and the results are combined.

    Parameters
    ----------
    wavelengths : array-like of float
        The wavelengths (m) to propagate.
    weights : array-like, optional
        Per-wavelength spectral weights (e.g. source SED); defaults to equal
        weights.  Used only by ``combine='intensity'``.
    combine : {'stack', 'intensity'}
        ``'stack'`` returns the per-wavelength complex fields stacked as
        ``(n_lambda, Ny, Nx)``.  ``'intensity'`` returns the weighted
        **incoherent** sum ``sum_l w_l |E_l|^2`` (real ``(Ny, Nx)``) -- the
        broadband PSF / image-plane intensity.
    **freespace_kwargs
        Forwarded to :func:`propagate_gbd_freespace` (``output_shape``,
        ``output_dx``, ``waist_factor``, ``sample_step``, ``chunk_beamlets``,
        ``direction_sampling``, ...).

    Returns
    -------
    array
        ``(n_lambda, Ny, Nx)`` complex for ``combine='stack'`` or
        ``(Ny, Nx)`` real for ``combine='intensity'``.
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'propagate_gbd_freespace_spectral',
                           input_kind='field')
    xp = array_namespace(E_in)
    lams = [float(_l) for _l in wavelengths]
    if not lams:
        raise ValueError(
            "propagate_gbd_freespace_spectral: ``wavelengths`` is empty.")
    fields = [
        propagate_gbd_freespace(E_in, dx, z=z, wavelength=_l,
                                **freespace_kwargs)
        for _l in lams
    ]
    if combine == 'stack':
        return xp.stack(fields, axis=0)
    if combine == 'intensity':
        if weights is None:
            w = [1.0] * len(lams)
        else:
            w = [float(_w) for _w in weights]
            if len(w) != len(lams):
                raise ValueError(
                    "propagate_gbd_freespace_spectral: ``weights`` length "
                    f"{len(w)} != ``wavelengths`` length {len(lams)}.")
        acc = xp.zeros(fields[0].shape, dtype=xp.abs(fields[0]).dtype)
        for _w, _F in zip(w, fields):
            acc = acc + _w * xp.abs(_F) ** 2
        return acc
    raise ValueError(
        f"propagate_gbd_freespace_spectral: combine must be 'stack' or "
        f"'intensity', got {combine!r}")


def propagate_gbd_freespace_vector(
    E_vec: np.ndarray,
    dx: float,
    *,
    z: float,
    wavelength: float,
    return_longitudinal: bool = False,
    **freespace_kwargs: Any,
) -> np.ndarray:
    """Vector (polarized) free-space GBD.

    Propagates a two-component Jones field.  In free space -- and through
    ideal, **polarization-preserving** optics -- the two transverse
    polarization components ``E_x`` and ``E_y`` propagate independently (the
    paraxial beamlet geometry is polarization-agnostic), so each is decomposed
    and reconstructed on its own and the vector field is reassembled.

    ``return_longitudinal=True`` (v5.21) additionally returns the LONGITUDINAL
    ``E_z`` from transversality (``E . k = 0``): a beamlet travelling along unit
    direction ``(L, M, N)`` carrying transverse ``(E_x, E_y)`` also carries
    ``E_z = -(L*E_x + M*E_y)/N``.  This is negligible for a paraxial / near-axial
    field (``L, M ~ 0`` -> ``E_z ~ 0``) but grows with tilt / NA -- the piece a
    transverse-only (scalar-per-component) GBD misses at high NA.  It is
    populated only when the beamlets actually carry transverse direction, so it
    needs ``direction_sampling=True`` (a source already tilted / diverging at
    the input plane); ``E_x`` and ``E_y`` are decomposed on a **common** beamlet
    geometry (the ``E_x`` wavevector, valid for a common-wavefront beam) so the
    per-beamlet longitudinal weight is well defined.  The returned grid is the
    input grid (output-resampling kwargs are ignored on this path).

    .. note::
       This free-space helper models only polarization-**preserving**
       propagation (independent components).  For **polarization-changing**
       transmission through a real prescription -- per-surface Fresnel s/p
       diattenuation along each base ray -- use
       :func:`propagate_gbd_vector_through_prescription`.  Reflection at
       surfaces, waveplates, and thin-film coatings build on the same base-ray
       trace and remain future extensions.

    Parameters
    ----------
    E_vec : array ``(2, Ny, Nx)`` complex
        The ``(E_x, E_y)`` Jones components of the source field.
    dx, z, wavelength : see :func:`propagate_gbd_freespace`.
    return_longitudinal : bool, default False
        If True, return the full ``(3, Ny, Nx)`` ``(E_x, E_y, E_z)`` field.
    **freespace_kwargs
        Forwarded to :func:`propagate_gbd_freespace` for each component (and,
        on the longitudinal path, to the decomposition: ``waist_factor``,
        ``sample_step``, ``direction_sampling``, ``dy``, ``window``).

    Returns
    -------
    array ``(2, Ny, Nx)`` complex, or ``(3, Ny, Nx)`` if
    ``return_longitudinal`` -- the propagated Jones field ``(E_x, E_y[, E_z])``.
    """
    xp = array_namespace(E_vec)
    E_vec = xp.asarray(E_vec)
    if E_vec.shape[0] != 2 or E_vec.ndim != 3:
        raise ValueError(
            "propagate_gbd_freespace_vector: E_vec must be (2, Ny, Nx) "
            f"(the E_x, E_y Jones components); got shape {E_vec.shape}.")
    if not return_longitudinal:
        comps = [
            propagate_gbd_freespace(E_vec[c], dx, z=z, wavelength=wavelength,
                                    **freespace_kwargs)
            for c in range(2)
        ]
        return xp.stack(comps, axis=0)

    # Full-vector path: decompose E_x and E_y on a COMMON beamlet geometry (so a
    # single beamlet carries both transverse components), propagate, and add the
    # longitudinal E_z from transversality at reconstruction.
    Ny, Nx = E_vec.shape[-2], E_vec.shape[-1]
    dy = freespace_kwargs.get('dy', None)
    waist_factor = freespace_kwargs.get('waist_factor', 1.0)
    sample_step = freespace_kwargs.get('sample_step', 1)
    direction_sampling = freespace_kwargs.get('direction_sampling', False)
    window = freespace_kwargs.get('window', 5.0)
    _warn_if_tilted_no_dirsample(E_vec, dx, dy, wavelength, direction_sampling)
    bx = decompose_field_to_beamlets(
        E_vec[0], dx, wavelength=wavelength, dy=dy, waist_factor=waist_factor,
        sample_step=sample_step, direction_sampling=direction_sampling)
    by_raw = decompose_field_to_beamlets(
        E_vec[1], dx, wavelength=wavelength, dy=dy, waist_factor=waist_factor,
        sample_step=sample_step, direction_sampling=False)
    # E_y on E_x's geometry + directions (co-located, common wavefront).
    by = BeamletBundle(positions=bx.positions, directions=bx.directions,
                       Q=bx.Q, amplitude=by_raw.amplitude, waist0=bx.waist0)
    bx = propagate_beamlets_freespace(bx, z, wavelength)
    by = propagate_beamlets_freespace(by, z, wavelength)
    return reconstruct_vector_field_with_ez(
        bx, by, Ny=Ny, Nx=Nx, dx=dx, wavelength=wavelength, dy=dy, window=window)


def reconstruct_vector_field_with_ez(
    bundle_x: BeamletBundle,
    bundle_y: BeamletBundle,
    *,
    Ny: int,
    Nx: int,
    dx: float,
    wavelength: float,
    dy: Optional[float] = None,
    window: Optional[float] = 5.0,
) -> np.ndarray:
    """Reconstruct the full 3-component ``(E_x, E_y, E_z)`` field from two
    co-located Jones beamlet bundles, adding the LONGITUDINAL ``E_z``.

    A propagating field is transverse to its local wavevector (``E . k = 0``),
    so a beamlet along unit direction ``(L, M, N)`` carrying transverse
    ``(E_x, E_y)`` also carries ``E_z = -(L*E_x + M*E_y)/N``.  Negligible for
    paraxial / near-axial beamlets (``L, M ~ 0``), significant at high NA / large
    tilt -- the piece a transverse-only GBD misses.  The two bundles MUST be
    co-located (same beamlet centres AND directions -- e.g. the ``E_x``/``E_y``
    Jones components of a common-wavefront beam); the per-beamlet longitudinal
    weight is then ``-(L*ampx + M*ampy)/N``, which coherently sums the
    angular-spectrum longitudinal field ``E_z(k) = -(kx*Ex + ky*Ey)/kz``.

    Returns
    -------
    array ``(3, Ny, Nx)`` complex -- the ``(E_x, E_y, E_z)`` field.
    """
    xp = array_namespace(bundle_x.positions)
    L = bundle_x.directions[..., 0]
    M = bundle_x.directions[..., 1]
    N = bundle_x.directions[..., 2]
    N_safe = xp.where(xp.abs(N) > 1e-30, N, xp.full_like(N, 1e-30))
    amp_ez = -(L * bundle_x.amplitude + M * bundle_y.amplitude) / N_safe
    bundle_z = BeamletBundle(
        positions=bundle_x.positions,
        directions=bundle_x.directions,
        Q=bundle_x.Q,
        amplitude=amp_ez.astype(bundle_x.amplitude.dtype),
        waist0=bundle_x.waist0,
    )
    Ex = reconstruct_field_from_beamlets(
        bundle_x, Ny=Ny, Nx=Nx, dx=dx, dy=dy, wavelength=wavelength,
        window=window)
    Ey = reconstruct_field_from_beamlets(
        bundle_y, Ny=Ny, Nx=Nx, dx=dx, dy=dy, wavelength=wavelength,
        window=window)
    Ez = reconstruct_field_from_beamlets(
        bundle_z, Ny=Ny, Nx=Nx, dx=dx, dy=dy, wavelength=wavelength,
        window=window)
    return xp.stack([Ex, Ey, Ez], axis=0)


def csp_beamlet_field(
    Xf: np.ndarray,
    Yf: np.ndarray,
    Zf: Any,
    positions: np.ndarray,
    directions: np.ndarray,
    waist0: np.ndarray,
    wavelength: float,
    amplitude: np.ndarray,
    *,
    chunk_beamlets: int = 256,
) -> np.ndarray:
    """Coherent sum of NON-PARAXIAL complex-source-point (CSP) beamlet fields.

    Each beamlet is the EXACT scalar-Helmholtz field of a point source displaced
    into a complex location -- Deschamps' "Gaussian beam as a bundle of complex
    rays" -- so unlike the paraxial-Gaussian beamlet it stays accurate at high NA
    (it matches the angular-spectrum method to grid precision at all NA; a
    paraxial Gaussian is ~33% wrong by NA 0.45).  For a beamlet with waist at
    ``r0``, unit axis ``s_hat``, waist radius ``w0`` and complex displacement
    ``b = zR = pi w0^2 / lambda``::

        r_s = r0 + i b s_hat
        R   = sqrt((r - r_s).(r - r_s))          [complex-symmetric, branch Im R <= 0]
        u   = A0 (-i zR) exp(i k R - k zR) / R

    which reduces to the paraxial Gaussian for ``zR >> lambda`` and satisfies
    ``(grad^2 + k^2) u = 0`` exactly at every real field point.  The propagation
    is analytic in ``Zf`` (no beamlet marching), so this evaluates the field at
    any output plane directly.  Convention: ``exp(-i w t)`` / forward
    ``exp(+ikz)`` (matches the rest of the library), branch ``Im R <= 0``
    (Salamin / Kaiser) for transverse decay.

    Parameters
    ----------
    Xf, Yf : (P,) arrays -- flattened output transverse coordinates [m].
    Zf : scalar or (P,) -- output axial coordinate(s) [m].
    positions : (n, 3) beamlet waist centres.
    directions : (n, 3) unit beamlet axes.
    waist0 : (n,) beamlet waist radii [m].
    amplitude : (n,) complex beamlet weights (``u(r0) = amplitude``).

    Returns
    -------
    (P,) complex field = sum over beamlets.
    """
    xp = array_namespace(positions)
    k = 2.0 * float(np.pi) / wavelength
    P = Xf.shape[0]
    out = xp.zeros((P,), dtype=xp.complex128)
    Zf_arr = Zf if xp.ndim(Zf) else xp.full((P,), float(Zf))
    n = positions.shape[0]
    step = max(1, int(chunk_beamlets))
    for c0 in range(0, n, step):
        c1 = min(n, c0 + step)
        w0 = waist0[c0:c1]
        zR = float(np.pi) * w0 ** 2 / wavelength                 # (m,)
        pos = positions[c0:c1]
        dirs = directions[c0:c1]
        rsx = pos[:, 0] + 1j * zR * dirs[:, 0]                   # (m,)
        rsy = pos[:, 1] + 1j * zR * dirs[:, 1]
        rsz = pos[:, 2] + 1j * zR * dirs[:, 2]
        dX = Xf[None, :] - rsx[:, None]                          # (m, P)
        dY = Yf[None, :] - rsy[:, None]
        dZ = Zf_arr[None, :] - rsz[:, None]
        R2 = dX * dX + dY * dY + dZ * dZ
        r = xp.sqrt(R2)
        R = xp.where(r.imag <= 0.0, r, -r)                       # branch Im R<=0
        u = (amplitude[c0:c1][:, None] * (-1j * zR[:, None])
             * xp.exp(1j * k * R - k * zR[:, None]) / R)
        out = out + xp.sum(u, axis=0)
    return out


def _csp_field_windowed(positions, directions, waist0, wavelength, amplitude,
                        Ny, Nx, dx, dy, z, window):
    """Windowed CSP reconstruction: evaluate each beamlet only over the pixel
    box where its (Gaussian-enveloped) CSP field is non-negligible -- out to
    ``window`` beam radii at the output plane (tail ``exp(-window^2)``).  Cost is
    ``O(n_beamlets * box)`` instead of ``O(n_beamlets * N^2)``; matches the dense
    sum to the tail truncation.  NumPy-only (data-dependent boxes); callers on
    other backends use the dense path."""
    k = 2.0 * float(np.pi) / wavelength
    out = np.zeros((Ny, Nx), dtype=np.complex128)
    x0, y0, z0 = positions[:, 0], positions[:, 1], positions[:, 2]
    L, M, N = directions[:, 0], directions[:, 1], directions[:, 2]
    zR = float(np.pi) * waist0 ** 2 / wavelength
    N_safe = np.where(np.abs(N) > 1e-30, N, 1e-30)
    t = (z - z0) / N_safe                          # along-axis distance ~ zeta
    cx = x0 + L * t                                # transverse beam centres
    cy = y0 + M * t
    wz = waist0 * np.sqrt(1.0 + (t / zR) ** 2)     # beam radius at the plane
    hwx = window * wz
    hwy = window * wz
    for i in range(positions.shape[0]):
        ix0 = max(0, int(np.floor((cx[i] - hwx[i]) / dx + Nx / 2)))
        ix1 = min(Nx, int(np.ceil((cx[i] + hwx[i]) / dx + Nx / 2)) + 1)
        iy0 = max(0, int(np.floor((cy[i] - hwy[i]) / dy + Ny / 2)))
        iy1 = min(Ny, int(np.ceil((cy[i] + hwy[i]) / dy + Ny / 2)) + 1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        xs = (np.arange(ix0, ix1) - Nx / 2) * dx
        ys = (np.arange(iy0, iy1) - Ny / 2) * dy
        Xg, Yg = np.meshgrid(xs, ys)
        dX = Xg - (x0[i] + 1j * zR[i] * L[i])
        dY = Yg - (y0[i] + 1j * zR[i] * M[i])
        dZ = z - (z0[i] + 1j * zR[i] * N[i])
        R2 = dX * dX + dY * dY + dZ * dZ
        r = np.sqrt(R2)
        R = np.where(r.imag <= 0.0, r, -r)
        out[iy0:iy1, ix0:ix1] += (amplitude[i] * (-1j * zR[i])
                                  * np.exp(1j * k * R - k * zR[i]) / R)
    return out


def propagate_gbd_freespace_csp(
    E_in: np.ndarray,
    dx: float,
    *,
    z: float,
    wavelength: float,
    dy: Optional[float] = None,
    waist_factor: float = 1.5,
    sample_step: int = 1,
    direction_sampling: bool = True,
    chunk_beamlets: int = 256,
    window: Optional[float] = None,
) -> np.ndarray:
    """Non-paraxial free-space GBD using complex-source-point beamlets.

    Drop-in alternative to :func:`propagate_gbd_freespace` that rides the same
    real-ray beamlet skeleton but propagates each beamlet as an EXACT-Helmholtz
    :func:`csp_beamlet_field` instead of a paraxial Gaussian.  Each beamlet is
    then exact at any NA (a single CSP beam matches the angular-spectrum method
    to grid precision, vs ~33% error for a paraxial Gaussian at NA 0.45).

    **Honest scope (measured).**  Per-beamlet exactness does NOT automatically
    make the GBD *sum* dramatically better than the paraxial sum: the two knobs
    are coupled -- narrowing beamlets raises each beamlet's NA (where CSP wins)
    but simultaneously worsens neighbour overlap (raising the reconstruction
    floor).  At moderate NA the reconstruction floor dominates and CSP-GBD and
    paraxial-GBD are comparable (both ~4e-2 on a moderate test); the CSP sum only
    pulls ahead when the FIELD is genuinely high-NA AND the decomposition is fine
    enough that per-beamlet physics is the limiting term.  Use
    :func:`csp_beamlet_field` directly for a sparse set of exact high-NA
    beamlets (its clearest win).  Cost is ``O(n_beamlets * N^2)`` (no windowing
    ``direction_sampling`` defaults to True (Husimi).  ``window`` (opt-in, NumPy
    only) evaluates each beamlet over its local pixel box out to ``window`` beam
    radii -- ``O(n_beamlets * box)`` instead of ``O(n_beamlets * N^2)`` -- and
    matches the dense sum to the tail truncation (``window=6.0`` -> ~1e-16 tail).
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'propagate_gbd_freespace_csp',
                           input_kind='field')
    xp = array_namespace(E_in)
    if dy is None:
        dy = dx
    Ny, Nx = E_in.shape[-2], E_in.shape[-1]
    b = decompose_field_to_beamlets(
        E_in, dx, wavelength=wavelength, dy=dy, waist_factor=waist_factor,
        sample_step=sample_step, direction_sampling=direction_sampling)
    if window is not None and xp is np:
        return _csp_field_windowed(
            b.positions, b.directions, b.waist0, wavelength, b.amplitude,
            Ny, Nx, dx, dy, float(z), float(window))
    xs = (xp.arange(Nx) - Nx / 2) * dx
    ys = (xp.arange(Ny) - Ny / 2) * dy
    Xg, Yg = xp.meshgrid(xs, ys)
    Ef = csp_beamlet_field(
        Xg.reshape(-1), Yg.reshape(-1), float(z),
        b.positions, b.directions, b.waist0, wavelength, b.amplitude,
        chunk_beamlets=chunk_beamlets)
    return Ef.reshape(Ny, Nx)


def _resolve_coating_index(coating: Any, wavelength: float) -> complex:
    """Resolve a single-material ``coating`` / layer index spec to a complex
    refractive index ``n + i*kappa``.

    Accepts a material NAME (looked up via
    :func:`lumenairy.glass.get_glass_index_complex`, e.g. ``'Au'``, ``'Ag'``,
    ``'Al'``), a callable ``wavelength -> complex``, or a direct numeric
    (``complex`` / ``float``) index.
    """
    if callable(coating):
        return complex(coating(wavelength))
    if isinstance(coating, (int, float, complex)):
        return complex(coating)
    from ..glass import get_glass_index_complex
    return complex(get_glass_index_complex(str(coating), wavelength))


def _coating_stack_layers(coating: Any, wavelength: float):
    """Parse a multilayer thin-film ``coating`` into ``[(n_complex, d_m), ...]``.

    A stack is a **list/tuple of layers**, each ``(index, thickness_m)`` or
    ``{'index': ..., 'thickness': ...}`` (``index`` resolved by
    :func:`_resolve_coating_index`), OR a ``{'layers': [...], 'substrate': ...}``
    dict (the ``substrate`` -- read by :func:`_coating_substrate` -- backs the
    stack for a **reflection** coating, e.g. a dielectric HR mirror).  Returns
    ``None`` when ``coating`` is not a stack (a single metal index for a mirror,
    a name/callable, or ``None``)."""
    if coating is None or isinstance(coating, (int, float, complex)) \
            or callable(coating) or isinstance(coating, str):
        return None
    if isinstance(coating, dict) and 'layers' in coating:
        coating = coating['layers']
    if not isinstance(coating, (list, tuple)) or len(coating) == 0:
        return None
    layers = []
    for lay in coating:
        if isinstance(lay, dict):
            idx, d = lay['index'], lay['thickness']
        else:
            idx, d = lay[0], lay[1]
        layers.append((_resolve_coating_index(idx, wavelength), float(d)))
    return layers


def _coating_substrate(coating: Any, wavelength: float):
    """The substrate index of a ``{'layers':..., 'substrate':...}`` reflection
    coating, or ``None`` (caller supplies a default)."""
    if isinstance(coating, dict) and coating.get('substrate') is not None:
        return _resolve_coating_index(coating['substrate'], wavelength)
    return None


def _thin_film_coefficients(layers, n0, ns, cos0, wavelength):
    """Thin-film characteristic-matrix (TMM) amplitude coefficients for a stack
    of ``layers`` ``[(n, d), ...]`` between incident index ``n0`` and substrate
    ``ns``, at per-ray ``cos0`` (cosine of incidence in medium 0).

    Returns ``(r_s, r_p, t_s, t_p)`` as **E-field** amplitude coefficients (the
    convention the Jones matrix uses): they reduce to the bare Fresnel
    coefficients at zero layers (validated), with the standard p-reflection sign
    and the ``cos0/cos_s`` p-transmission E-field factor.  ``n0`` is real,
    ``ns`` / layer indices may be complex (absorbing).
    """
    n0 = complex(n0)
    ns = complex(ns)
    cos0 = np.asarray(cos0, dtype=np.complex128)
    sin0sq = 1.0 - cos0 ** 2
    cos_s = np.sqrt(1.0 - (n0 / ns) ** 2 * sin0sq)

    def _mat(pol):
        def eta(nn, cc):
            return nn * cc if pol == 's' else nn / cc
        M11 = np.ones_like(cos0)
        M12 = np.zeros_like(cos0)
        M21 = np.zeros_like(cos0)
        M22 = np.ones_like(cos0)
        for nj, dj in layers:
            nj = complex(nj)
            cosj = np.sqrt(1.0 - (n0 / nj) ** 2 * sin0sq)
            delta = (2.0 * np.pi / wavelength) * nj * dj * cosj
            ej = eta(nj, cosj)
            c = np.cos(delta)
            s = np.sin(delta)
            m11, m12, m21, m22 = c, 1j * s / ej, 1j * ej * s, c
            M11, M12, M21, M22 = (M11 * m11 + M12 * m21, M11 * m12 + M12 * m22,
                                  M21 * m11 + M22 * m21, M21 * m12 + M22 * m22)
        e0 = eta(n0, cos0)
        es = eta(ns, cos_s)
        B = M11 + M12 * es
        C = M21 + M22 * es
        r = (e0 * B - C) / (e0 * B + C)
        t = 2.0 * e0 / (e0 * B + C)
        return r, t
    rs, ts = _mat('s')
    rp_adm, tp_adm = _mat('p')
    rp = -rp_adm                      # standard Fresnel p-reflection sign
    tp = tp_adm * (cos0 / cos_s)      # tangential -> E-field p-transmission
    return rs, rp, ts, tp


def gbd_ghost_analysis(
    prescription: Dict[str, Any],
    wavelength: float,
    *,
    field_angle: float = 0.0,
) -> Dict[str, Any]:
    """First-order stray-light / ghost budget for a refractive prescription.

    Forward GBD carries only the transmitted beam; a fraction of the light
    Fresnel-**reflects** at every dielectric surface, and a double bounce
    (reflect at surface j on the way in, reflect again at surface i < j) forms a
    ghost that reaches the image as veiling glare.  This quantifies the budget
    without propagating the (much weaker) ghost fields:

    * ``surface_reflectance`` -- per refracting surface, the unpolarized power
      reflectance ``R = (R_s + R_p) / 2`` of the base ray (the ghost-*source*
      strength; a multilayer AR ``coating`` is honoured via the thin-film TMM).
    * ``double_bounce`` -- for each surface pair ``(i, j)`` (i < j), the relative
      ghost irradiance ``~ R_i * R_j`` (the leading double-reflection term; the
      transmission at the other surfaces is ~1 for good AR coats and is not
      double-counted here).  The largest entries are the ghosts to design out.

    A chief ray at ``field_angle`` (radians, tilt in x) samples the incidence
    angles.  This is a ray-optics budget; the coherent ghost *field* (its focus
    / defocus and PSF) needs the double-bounce field propagation, a documented
    extension (it shares the curved-fold world-frame machinery).

    Returns
    -------
    dict with ``'surface_reflectance'`` ``(n_surf,)``, ``'double_bounce'``
    ``(n_surf, n_surf)`` (upper triangular, ``R_i*R_j``), and ``'worst'`` --
    ``(i, j, relative_intensity)`` of the brightest ghost.
    """
    from ..glass import get_glass_index
    from ..raytrace.intersection import (
        _intersect_surface,
        _reflect,
        _refract,
        _surface_normal,
        _transfer,
    )
    from ..raytrace.trace import _make_bundle, surfaces_from_prescription

    surfs = surfaces_from_prescription(prescription)
    ux0 = float(np.tan(field_angle))
    inv = 1.0 / np.sqrt(1.0 + ux0 ** 2)
    rb = _make_bundle(np.array([0.0]), np.array([0.0]),
                      np.array([ux0 * inv]), np.array([0.0]), wavelength)
    refl = []
    for s in surfs:
        n1 = float(get_glass_index(getattr(s, 'glass_before', None) or 'air',
                                   wavelength))
        n2 = float(get_glass_index(getattr(s, 'glass_after', None) or 'air',
                                   wavelength))
        _intersect_surface(rb, s, n_medium=n1)
        if bool(getattr(s, 'is_mirror', False)):
            # a real mirror is the intended path, not a ghost source (R=0)
            refl.append(0.0)
            _reflect(rb, s)
            _transfer(rb, float(getattr(s, 'thickness', 0.0) or 0.0), n2)
            continue
        nx, ny, nz = _surface_normal(rb.x, rb.y, s)
        cosd = rb.L * nx + rb.M * ny + rb.N * nz
        cos_i = float(np.abs(cosd)[0])
        stack = _coating_stack_layers(getattr(s, 'coating', None), wavelength)
        if stack is not None:
            rs, rp, _ts, _tp = _thin_film_coefficients(
                stack, n1, n2, np.array([cos_i]), wavelength)
            R = 0.5 * (abs(rs[0]) ** 2 + abs(rp[0]) ** 2)
        else:
            ct = np.sqrt(max(1.0 - (n1 / n2) ** 2 * (1.0 - cos_i ** 2), 0.0))
            rs = (n1 * cos_i - n2 * ct) / (n1 * cos_i + n2 * ct)
            rp = (n2 * cos_i - n1 * ct) / (n2 * cos_i + n1 * ct)
            R = 0.5 * (rs ** 2 + rp ** 2)
        refl.append(float(R))
        _refract(rb, s, n1, n2)
        _transfer(rb, float(getattr(s, 'thickness', 0.0) or 0.0), n2)
    refl = np.asarray(refl)
    n = refl.shape[0]
    db = np.zeros((n, n))
    worst = (0, 0, 0.0)
    for i in range(n):
        for j in range(i + 1, n):
            db[i, j] = refl[i] * refl[j]
            if db[i, j] > worst[2]:
                worst = (i, j, float(db[i, j]))
    return {'surface_reflectance': refl, 'double_bounce': db, 'worst': worst}


def _fresnel_jones_matrix_per_beamlet(x, y, ux, uy, prescription, wavelength):
    """Per-beamlet 2x2 transverse Jones matrix ``(E_x, E_y)_in -> _out`` from
    per-surface Fresnel s/p along each base ray (polarization ray tracing).
    Traces surface-by-surface, dispatching each surface exactly as
    :func:`raytrace.trace` (reflect at ``is_mirror`` surfaces, refract
    otherwise), then splits the field into s (perp to plane of incidence) and p
    components and applies the surface coefficients:

    * **Refraction** -- Fresnel amplitude transmission ``t_s``, ``t_p``, rotating
      the p axis from the incident to the refracted ray.  If the surface carries
      a **multilayer** ``coating`` (a list of ``(index, thickness)`` layers --
      an AR / dichroic stack), ``t_s`` / ``t_p`` come from the thin-film
      characteristic-matrix method (:func:`_thin_film_coefficients`) instead of
      the bare single-interface Fresnel, capturing the stack's transmittance +
      retardance (reduces to bare Fresnel at zero layers).
    * **Reflection** (``is_mirror``) -- if the surface carries no ``coating``,
      an ideal reflector ``|r_s| = |r_p| = 1`` (energy-conserving, no
      diattenuation; PEC convention ``r_s = -1``, ``r_p = +1`` for the fold's
      relative s-p phase).  If the surface carries a ``coating`` (a complex
      refractive index ``n + i*kappa`` -- a metal), the full **complex Fresnel**
      ``r_s`` / ``r_p`` are used, giving the metal mirror's **diattenuation**
      (``|r_s| != |r_p|``) and **retardance** (``arg r_s != arg r_p``); these
      reduce continuously to the ideal ``r_s = -1``, ``r_p = +1`` as
      ``|n_coating| -> inf`` (perfect conductor).  Either way the geometric s/p
      frame rotation is carried by recomposing on the reflected p axis.

    Partial (Fresnel) reflection at refractive surfaces (ghost beams) is not
    modeled -- forward GBD carries only the transmitted / intended-reflected
    beam.

    Convention: the returned matrix is the **transverse** ``(E_x, E_y)`` field,
    so the s channel (always transverse) is Fresnel-exact at every angle, while
    the p channel carries the honest ``cos(theta_out)`` projection of the tilted
    output ray's p-vector (its small longitudinal component is not part of the
    transverse field).  Input rays should be launched **axially** (``ux = uy =
    0``), as they are in :func:`propagate_gbd_vector_through_prescription`, so
    that ``(x, y)`` *is* the input transverse frame; the incidence angles then
    come from the surface curvature/tilt, not from a pre-tilted input ray.

    Returns ``(P, alive)``: ``P`` is ``(N, 2, 2)`` complex; ``alive`` is
    ``(N,)`` bool (False where the base ray vignetted / TIR'd).  ``P`` is zeroed
    for dead rays so ``P @ E`` never propagates NaNs.
    """
    from ..glass import get_glass_index
    from ..raytrace.intersection import (
        _intersect_surface,
        _reflect,
        _refract,
        _surface_normal,
        _transfer,
    )
    from ..raytrace.trace import _make_bundle, surfaces_from_prescription

    surfs = surfaces_from_prescription(prescription)
    inv = 1.0 / np.sqrt(1.0 + ux ** 2 + uy ** 2)
    L0, M0 = ux * inv, uy * inv
    n = x.shape[0]

    def _trace_jones(J0):
        rb = _make_bundle(x.copy(), y.copy(), L0.copy(), M0.copy(), wavelength)
        J = J0.astype(np.complex128).copy()
        for s in surfs:
            n1 = float(get_glass_index(
                getattr(s, 'glass_before', None) or 'air', wavelength))
            n2 = float(get_glass_index(
                getattr(s, 'glass_after', None) or 'air', wavelength))
            # pass n_medium=n1 so rb.opd stays consistent with raytrace.trace
            # in immersed media (the Jones build never reads opd, but keep the
            # trace faithful).
            _intersect_surface(rb, s, n_medium=n1)
            nx, ny, nz = _surface_normal(rb.x, rb.y, s)
            d_in = np.stack([rb.L, rb.M, rb.N], axis=-1)
            nrm = np.stack([nx, ny, nz], axis=-1)
            cosd = np.sum(d_in * nrm, axis=-1)
            nrm = np.where(cosd[:, None] > 0, -nrm, nrm)
            cos_i = np.abs(cosd)
            is_mirror = bool(getattr(s, 'is_mirror', False))
            # Dispatch the base ray exactly as raytrace.trace does (reflect vs
            # refract) so the traced path stays correct through mirror / fold
            # surfaces; then build the per-surface s/p coefficients.
            if is_mirror:
                _reflect(rb, s)
                coating = getattr(s, 'coating', None)
                stack = _coating_stack_layers(coating, wavelength)
                if stack is not None:
                    # Dielectric multilayer (HR / Bragg) mirror: r_s / r_p from
                    # the thin-film characteristic matrix, incident medium n1 ->
                    # stack -> substrate (defaults to n1, a free-standing stack
                    # reflecting back into the incident medium).
                    subs = _coating_substrate(coating, wavelength)
                    if subs is None:
                        subs = complex(n1)
                    cs, cp, _ts, _tp = _thin_film_coefficients(
                        stack, n1, subs, cos_i, wavelength)
                elif coating is None:
                    # Ideal reflector: |r_s| = |r_p| = 1 (energy-conserving, no
                    # diattenuation).  PEC convention r_s = -1, r_p = +1 carries
                    # the fold's relative s-p phase; the geometric s/p frame
                    # rotation (recompose on the reflected p_out) does the rest.
                    cs = -np.ones_like(cos_i, dtype=np.complex128)   # r_s
                    cp = np.ones_like(cos_i, dtype=np.complex128)    # r_p
                else:
                    # Real metal coating: full complex Fresnel r_s / r_p from the
                    # coating's complex index n_c = n + i*kappa -> diattenuation
                    # (|r_s| != |r_p|) + retardance (arg r_s != arg r_p).
                    # Continuously reduces to the ideal r_s=-1, r_p=+1 as
                    # |n_c| -> inf (perfect conductor).
                    nc = _resolve_coating_index(coating, wavelength)
                    ci = cos_i.astype(np.complex128)
                    ct = np.sqrt(1.0 - (n1 / nc) ** 2 * (1.0 - ci ** 2))
                    cs = (n1 * ci - nc * ct) / (n1 * ci + nc * ct)   # r_s
                    cp = (nc * ci - n1 * ct) / (nc * ci + n1 * ct)   # r_p
                # at normal incidence s/p degenerate -> the physical reflection
                # coefficient is r_s (r_p = -r_s there is the p-basis-flip
                # artifact the oblique p_out recomposition otherwise absorbs).
                c_normal = cs
                n_after = n2
            else:
                _refract(rb, s, n1, n2)
                stack = _coating_stack_layers(
                    getattr(s, 'coating', None), wavelength)
                if stack is not None:
                    # Multilayer thin-film (e.g. AR / dichroic) between n1 and
                    # n2: E-field transmission t_s / t_p from the characteristic
                    # matrix, with the stack's wavelength/angle-dependent
                    # transmittance + retardance.  Reduces to bare Fresnel at 0
                    # layers.
                    _rs, _rp, cs, cp = _thin_film_coefficients(
                        stack, n1, n2, cos_i, wavelength)
                    c_normal = cs
                else:
                    mu = n1 / n2
                    cos_t = np.sqrt(np.maximum(
                        1.0 - mu ** 2 * (1.0 - cos_i ** 2), 0.0))
                    cs = (2.0 * n1 * cos_i
                          / (n1 * cos_i + n2 * cos_t)).astype(np.complex128)
                    cp = (2.0 * n1 * cos_i
                          / (n2 * cos_i + n1 * cos_t)).astype(np.complex128)
                    c_normal = cs                                # t_s == t_p
                n_after = n2
            d_out = np.stack([rb.L, rb.M, rb.N], axis=-1)
            s_vec = np.cross(d_in, nrm)
            s_norm = np.linalg.norm(s_vec, axis=-1, keepdims=True)
            at_normal = s_norm[:, 0] < 1e-9
            s_hat = np.where(s_norm > 1e-30, s_vec / np.where(s_norm > 0, s_norm, 1.0),
                             np.array([1.0, 0.0, 0.0]))
            p_in = np.cross(d_in, s_hat)
            p_out = np.cross(d_out, s_hat)
            Es = np.sum(J * s_hat, axis=-1)
            Ep = np.sum(J * p_in, axis=-1)
            J_sp = (Es * cs)[:, None] * s_hat + (Ep * cp)[:, None] * p_out
            # at normal incidence s/p is degenerate but the coefficients agree
            # (t_s == t_p for refraction; r_s == r_p = -1 for a mirror) -> scale.
            J = np.where(at_normal[:, None], c_normal[:, None] * J, J_sp)
            _transfer(rb, float(getattr(s, 'thickness', 0.0) or 0.0), n_after)
        # project onto the output transverse (x, y) plane (paraxial output ray)
        return J[:, 0], J[:, 1], rb.alive

    ex_x, ex_y, alive = _trace_jones(
        np.tile(np.array([1.0, 0.0, 0.0]), (n, 1)))
    ey_x, ey_y, _ = _trace_jones(np.tile(np.array([0.0, 1.0, 0.0]), (n, 1)))
    P = np.empty((n, 2, 2), dtype=np.complex128)
    P[:, 0, 0] = ex_x
    P[:, 1, 0] = ex_y
    P[:, 0, 1] = ey_x
    P[:, 1, 1] = ey_y
    # zero out dead / non-finite rays so ``P @ E`` never propagates NaNs into
    # live neighbouring beamlets (a vignetted base ray can leave NaN in P).
    alive = np.asarray(alive, bool) & np.isfinite(P).all(axis=(1, 2))
    P[~alive] = 0.0
    return P, alive


def _upsample_nodes_to_grid(node_vals: np.ndarray, Ny: int, Nx: int,
                            step: int) -> np.ndarray:
    """Bilinearly upsample a coarse ``(n_iy, n_ix)`` complex NODE grid (values on
    the beamlet subsample lattice -- node ``(a, b)`` at pixel ``(a*step,
    b*step)``) to the full ``(Ny, Nx)`` grid.

    Used by the vector prescription chain (A7 / N4 port) to spread the
    per-beamlet Fresnel Jones (a SMOOTH per-surface diattenuation) densely, so
    the Husimi launch reads the true input wavefront from ``E * P`` instead of
    from a near-empty zero-scatter grid whose gradient is dominated by the fill.
    At the exact nodes the ``order=1`` interpolation reproduces the node values,
    so the sampled beamlet amplitudes are byte-identical to the scatter path --
    only the launch DIRECTIONS gain the true wavevector.  Edge-clamped
    (``mode='nearest'``) beyond the last node (the ~few-pixel margin where the
    aperture field is already ~0)."""
    from scipy.ndimage import map_coordinates
    rr = np.clip(np.arange(Ny) / step, 0.0, node_vals.shape[0] - 1)
    cc = np.clip(np.arange(Nx) / step, 0.0, node_vals.shape[1] - 1)
    R, C = np.meshgrid(rr, cc, indexing='ij')
    coords = np.stack([R.ravel(), C.ravel()])
    out = (map_coordinates(node_vals.real, coords, order=1, mode='nearest')
           + 1j * map_coordinates(node_vals.imag, coords, order=1,
                                  mode='nearest'))
    return out.reshape(Ny, Nx)


def _vector_prescription_mixed_fields(
    E_vec: np.ndarray,
    dx: float,
    prescription: Dict[str, Any],
    *,
    wavelength: float,
    sample_step: int,
    direction_sampling: Any,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Resolve the SHARED carrier-launch policy on the ORIGINAL Jones field and
    build the two per-component Fresnel-mixed input fields the scalar
    prescription chain consumes.

    Returns ``(ExF, EyF, ds)`` where ``ds`` is the resolved boolean
    direction-sampling policy.  Split out so the vector chain and its
    chain-vs-sequential equivalence oracle share ONE construction (no drift).

    ``direction_sampling`` resolution (A7, mirrors N4's scalar chain):

    * ``'auto'`` measures the RMS local-tilt spread of the input via the SHARED
      :func:`_input_angular_spread` detector, on the ORIGINAL (un-mixed)
      components -- the MAX over the two, so a diverging / converging / tilted
      field fires even when only one component carries the curvature, and a
      component with no power contributes 0.  Detecting on the ORIGINAL field
      (not the P-mixed one) keeps a collimated input on the byte-identical axial
      path: the Fresnel s/p phase P carries across the aperture would otherwise
      trip ``'auto'`` on an otherwise flat wavefront.
    * ``True`` / ``False`` force Husimi / axial.

    Field construction:

    * ``ds`` (Husimi): the scalar chain's carrier-normal launch reads the
      wavevector from the DENSE field gradient, so the mixed component must be
      DENSE.  Build ``ExF = P00_dense * Ex + P01_dense * Ey`` (and likewise
      ``EyF``) with the Fresnel Jones bilinearly upsampled to the full grid
      (:func:`_upsample_nodes_to_grid`) -- P is SMOOTH so it adds no spurious
      carrier, while the DENSE original ``E_vec`` supplies the true, rapidly
      varying wavefront phase.  At the beamlet nodes the upsampled P reproduces
      P exactly, so the beamlet amplitudes are byte-identical to the scatter
      path (only the launch direction changes).
    * not ``ds`` (axial / collimated): the historical sparse scatter -- the
      mixed samples on a zero-filled grid at the ``sample_step`` lattice --
      which the axial decompose sub-samples identically.  Byte-identical to
      prior releases (the axial launch ignores the gradient, so density does not
      matter, but the exact old construction is kept so the default is provably
      unchanged)."""
    E_vec = np.asarray(E_vec)
    Ny, Nx = E_vec.shape[-2], E_vec.shape[-1]
    if direction_sampling == 'auto':
        spread = max(
            _input_angular_spread(E_vec[0], float(dx), float(dx),
                                  float(wavelength)),
            _input_angular_spread(E_vec[1], float(dx), float(dx),
                                  float(wavelength)))
        ds = spread > _GBD_AUTO_HUSIMI_THRESH
    elif direction_sampling in (True, False):
        ds = bool(direction_sampling)
    else:
        raise ValueError(
            "propagate_gbd_vector_through_prescription: direction_sampling must "
            f"be 'auto', True or False, got {direction_sampling!r}.")

    iy = np.arange(0, Ny, sample_step)
    ix = np.arange(0, Nx, sample_step)
    Iy, Ix = np.meshgrid(iy, ix, indexing='ij')
    xb = (Ix.ravel() - Nx / 2) * dx
    yb = (Iy.ravel() - Ny / 2) * dx
    zc = np.zeros_like(xb)
    P, _alive = _fresnel_jones_matrix_per_beamlet(
        xb, yb, zc, zc, prescription, wavelength)
    ExS = E_vec[0][Iy, Ix].ravel()
    EyS = E_vec[1][Iy, Ix].ravel()
    ExM = (P[:, 0, 0] * ExS + P[:, 0, 1] * EyS).reshape(Iy.shape)
    EyM = (P[:, 1, 0] * ExS + P[:, 1, 1] * EyS).reshape(Iy.shape)

    if ds:
        n_iy, n_ix = Iy.shape

        def _dense(i, j):
            return _upsample_nodes_to_grid(
                P[:, i, j].reshape(n_iy, n_ix), Ny, Nx, sample_step)

        # Build the two mixed components sequentially (each needs only its own
        # row of the dense Jones), so at most TWO N^2 dense-P arrays coexist, not
        # four -- the N^2-scale-array memory discipline (Section 0 spirit) even
        # though this path adds no persistent cache.
        ExF = _dense(0, 0) * E_vec[0] + _dense(0, 1) * E_vec[1]
        EyF = _dense(1, 0) * E_vec[0] + _dense(1, 1) * E_vec[1]
    else:
        ExF = np.zeros((Ny, Nx), dtype=np.complex128)
        EyF = np.zeros((Ny, Nx), dtype=np.complex128)
        ExF[Iy, Ix] = ExM
        EyF[Iy, Ix] = EyM
    return ExF, EyF, ds


def propagate_gbd_vector_through_prescription(
    E_vec: np.ndarray,
    dx: float,
    prescription: Dict[str, Any],
    *,
    wavelength: float,
    per_surface: bool = True,
    direction_sampling: Any = 'auto',
    **kwargs: Any,
) -> np.ndarray:
    """Vector (Jones) GBD through a prescription with **polarization ray
    tracing** -- per-surface Fresnel s/p diattenuation along each beamlet's
    base ray, mixing the ``(E_x, E_y)`` components.

    Combines the scalar/tensor envelope propagation
    (:func:`propagate_gbd_through_prescription`, ``per_surface`` selecting the
    aberration-aware form) with the per-beamlet Fresnel Jones matrix from
    :func:`_fresnel_jones_matrix_per_beamlet`: the diattenuation modifies the
    beamlet amplitudes, then each transformed component is reconstructed.

    ``direction_sampling`` (A7, 2026-07-21) defaults to ``'auto'``, porting the
    N4 carrier-normal launch to the vector chain: the SHARED
    :func:`_input_angular_spread` detector (measured on the ORIGINAL Jones
    field) selects the Husimi (carrier-normal) launch for a diverging /
    converging / tilted source and the axial launch otherwise, resolved ONCE and
    applied to BOTH components.  A collimated input measures ~0 spread and takes
    the byte-identical axial path; a diverging input then conserves power (to the
    Fresnel-transmission limit) instead of collapsing to the collimated plane and
    shedding frame energy.  ``True`` / ``False`` force Husimi / axial.  See
    :func:`_vector_prescription_mixed_fields`.

    Transmission only (no reflection / thin-film coatings -- those build on the
    same base-ray trace and are a documented extension).

    Scope note (D10): for the axial launch the Jones-mixed samples are scattered
    onto zero-filled full grids and rely on ``decompose_field_to_beamlets``
    re-sampling the IDENTICAL pixel set (same ``sample_step`` phase); for the
    Husimi launch a DENSE mixed field (smooth Fresnel-Jones upsampled x the dense
    input) is built so the carrier-normal decomposition reads the true wavevector
    (the scatter grid's gradient is dominated by the zero fill and would collapse
    the launch to axial).  Polarization itself is evaluated for the AXIAL
    congruence only (``_fresnel_jones_matrix_per_beamlet`` is axial-launch by
    contract).

    Parameters
    ----------
    E_vec : array ``(2, Ny, Nx)`` complex -- the ``(E_x, E_y)`` Jones field.
    dx, prescription, wavelength : as elsewhere.
    per_surface : bool -- passed to the envelope propagator.
    direction_sampling : ``'auto'`` (default) / ``True`` / ``False`` -- the
        shared carrier-normal launch policy (A7).
    **kwargs : forwarded to :func:`propagate_gbd_through_prescription`.

    Returns
    -------
    array ``(2, Ny, Nx)`` complex -- the ``(E_x, E_y)`` output Jones field.
    """
    E_vec = np.asarray(E_vec)
    if E_vec.shape[0] != 2 or E_vec.ndim != 3:
        raise ValueError(
            "propagate_gbd_vector_through_prescription: E_vec must be "
            f"(2, Ny, Nx); got {E_vec.shape}.")
    sample_step = int(kwargs.get('sample_step', 1))
    # A7 (2026-07-21): resolve the shared carrier launch on the ORIGINAL Jones
    # field and build the per-component Fresnel-mixed inputs (dense for Husimi,
    # sparse-scatter for axial) -- ONE construction shared with the equivalence
    # oracle.  The resolved boolean ``ds`` is passed EXPLICITLY to the scalar
    # chains so their ``'auto'`` does not re-measure on the P-modulated field.
    ExF, EyF, ds = _vector_prescription_mixed_fields(
        E_vec, dx, prescription, wavelength=wavelength,
        sample_step=sample_step, direction_sampling=direction_sampling)
    kw = dict(kwargs)
    kw['sample_step'] = sample_step
    out_x = propagate_gbd_through_prescription(
        ExF, dx, prescription, wavelength=wavelength, per_surface=per_surface,
        direction_sampling=ds, **kw)
    out_y = propagate_gbd_through_prescription(
        EyF, dx, prescription, wavelength=wavelength, per_surface=per_surface,
        direction_sampling=ds, **kw)
    return np.stack([out_x, out_y], axis=0)


def propagate_gbd_thin_lens(
    E_in: np.ndarray,
    dx: float,
    *,
    z_to_lens: float,
    focal_length: float,
    z_lens_to_output: float,
    wavelength: float,
    output_shape: Optional[Tuple[int, int]] = None,
    output_grid: Optional[Tuple[int, int]] = None,
    output_dx: Optional[float] = None,
    output_centre: Tuple[float, float] = (0.0, 0.0),
    lens_centre: Tuple[float, float] = (0.0, 0.0),
    waist_factor: float = 1.0,
    sample_step: int = 1,
    chunk_beamlets: int = 2048,
    direction_sampling: bool = False,
    aperture_semi_diameter: Optional[float] = None,
    aperture_shape: str = 'circular',
    aperture_soft_edge: bool = False,
) -> np.ndarray:
    """End-to-end three-leg GBD: source -> free space -> thin lens
    -> free space -> output (the canonical GBD validation case).

    ``direction_sampling=True`` selects the Husimi (position + direction)
    decomposition so a source already tilted / diverging at the input plane
    walks off correctly through the system; see
    :func:`decompose_field_to_beamlets`.

    ``aperture_soft_edge=True`` uses the analytic partial-vignetting weight
    (:func:`apply_aperture_to_beamlets`) at the stop instead of a binary
    chief-ray cut -- measured to improve the hard-aperture / Airy-focus
    accuracy ~1.8x (e.g. 3.3% -> 1.9% relative-intensity error at a hard
    circular stop's focus) at no extra cost.

    .. versionchanged:: 5.2
        ``output_grid`` -> ``output_shape`` rename (AUDIT_V4_13_1 Part 2
        P1-A); see :func:`propagate_gbd_freespace`.
    """
    Ny, Nx = _resolve_output_shape(
        output_shape, output_grid,
        fn_name='propagate_gbd_thin_lens',
        default_shape=(E_in.shape[-2], E_in.shape[-1]),
    )
    if output_dx is None:
        output_dx = dx

    _warn_if_tilted_no_dirsample(E_in, dx, None, wavelength, direction_sampling)
    bundle = decompose_field_to_beamlets(
        E_in, dx, wavelength=wavelength,
        waist_factor=waist_factor,
        sample_step=sample_step,
        direction_sampling=direction_sampling,
    )
    bundle = propagate_beamlets_freespace(bundle, z_distance=z_to_lens,
                                          wavelength=wavelength)
    if aperture_semi_diameter is not None:
        bundle = apply_aperture_to_beamlets(
            bundle, aperture_semi_diameter, centre=lens_centre,
            shape=aperture_shape, soft_edge=aperture_soft_edge,
            wavelength=wavelength)
    bundle = apply_thin_lens_to_beamlets(bundle, focal_length=focal_length,
                                         wavelength=wavelength,
                                         centre=lens_centre)
    bundle = propagate_beamlets_freespace(bundle, z_distance=z_lens_to_output,
                                          wavelength=wavelength)
    return reconstruct_field_from_beamlets(
        bundle, Ny=Ny, Nx=Nx, dx=output_dx,
        centre=output_centre, wavelength=wavelength,
        chunk_beamlets=chunk_beamlets, window=_GBD_RECONSTRUCT_WINDOW,
    )


# ============================================================================
# Prescription-aware GBD via system ABCD
# ============================================================================

def apply_abcd_to_beamlets(
    beamlets: BeamletBundle,
    A: float,
    B: float,
    C: float,
    D: float,
    wavelength: float,
    axial_opl: Optional[float] = None,
) -> BeamletBundle:
    """Apply a paraxial 2x2 ABCD matrix to every beamlet.

    Each beamlet's complex Q-parameter (= 1/q) transforms as

        Q_out = (C + D Q_in) / (A + B Q_in)

    and its base ray's transverse offset / slope transform as

        x_out = A x_in + B u_in
        u_out = C x_in + D u_in

    where ``u`` is the ray slope (paraxial direction cosine).  This
    is a paraxial approximation suitable for propagation through a
    sequential refractive system characterised by its system ABCD;
    see :func:`lumenairy.raytrace.system_abcd_prescription`.

    Parameters
    ----------
    beamlets : BeamletBundle
    A, B, C, D : float
        ABCD matrix elements.
    wavelength : float
        Vacuum wavelength.
    """
    xp = array_namespace(beamlets.positions)

    # Q evolution.
    Q_old = beamlets.Q
    if _q_is_tensor(Q_old):
        # v5.21: tensor Q -- generalized (matrix) Collins with scalar blocks
        # A,B,C,D -> A I2 etc.
        _I2 = xp.eye(2, dtype=Q_old.dtype)[None, :, :]
        Q_new = (C * _I2 + D * Q_old) @ _inv2x2(A * _I2 + B * Q_old, xp)
        Q_new = 0.5 * (Q_new + xp.transpose(Q_new, (0, 2, 1)))
    else:
        Q_new = (C + D * Q_old) / (A + B * Q_old)

    # Base-ray paraxial transform: ray height x and slope u.
    x_in = beamlets.positions[..., 0]
    y_in = beamlets.positions[..., 1]
    L_in = beamlets.directions[..., 0]
    M_in = beamlets.directions[..., 1]
    N_in = beamlets.directions[..., 2]
    u_x = L_in / xp.where(xp.abs(N_in) > 1e-30, N_in, 1e-30)
    u_y = M_in / xp.where(xp.abs(N_in) > 1e-30, N_in, 1e-30)

    x_out = A * x_in + B * u_x
    y_out = A * y_in + B * u_y
    u_x_out = C * x_in + D * u_x
    u_y_out = C * y_in + D * u_y

    # Re-normalise direction (paraxial slope -> direction cosines).
    norm = xp.sqrt(u_x_out ** 2 + u_y_out ** 2 + 1.0)
    L_out = u_x_out / norm
    M_out = u_y_out / norm
    N_out = 1.0 / norm
    new_directions = xp.stack([L_out, M_out, N_out], axis=-1)
    z_out = beamlets.positions[..., 2]
    new_positions = xp.stack([x_out, y_out, z_out], axis=-1)

    # Amplitude correction: Collins/Siegman on-axis Gaussian-beam factor
    #   u_out(0) = u_in(0) / (A + B/q_in) = u_in(0) / (A + B*Q_in)
    # (Siegman ch. 20 / Collins integral; both transverse dimensions of
    # the rotationally-symmetric beamlet give 1/sqrt each).
    #
    # v5.17.x (P2-30) BEHAVIOUR CHANGE: pre-fix this applied
    # ``Q_new / Q_old`` = ``q_in / q_out`` = (C*q_in + D)/(A + B*Q_in),
    # i.e. the Collins factor times a spurious ``(C*q_in + D)``.  For any
    # focusing system (C != 0) that factor has non-unit modulus, so the
    # single-ABCD path disagreed with the sequential per-leg path
    # (``propagate_beamlets_freespace`` + ``apply_thin_lens_to_beamlets``
    # compose exactly to exp(ikL)/(A + B*Q_in), verified to 1e-15) in
    # both amplitude and piston phase -- e.g. a t1=20mm -> f=50mm ->
    # t2=30mm system came out |C*q_in+D| = 0.60x low in field amplitude
    # (0.36x in intensity) with a wrong piston, defeating the v4.11.1
    # axial_opl coherent-superposition fix.  Free-space-only ABCD (C=0,
    # D=1) is unchanged.  ``Q`` here is the module's engineering 1/q
    # parameterisation (Q = 1/q_code, q_code = conj(q_physics)); ABCD
    # elements are real so the Collins factor commutes with that
    # convention.
    if _q_is_tensor(Q_old):
        _I2 = xp.eye(2, dtype=Q_old.dtype)[None, :, :]
        qratio = 1.0 / xp.sqrt(_det2x2(A * _I2 + B * Q_old))
    else:
        qratio = 1.0 / (A + B * Q_old)
    # 4.10.2: include the chief-ray axial OPL phase exp(+i*k*L_chief)
    # when supplied.  The three-leg helpers (propagate_gbd_freespace,
    # propagate_gbd_thin_lens) accumulate this leg-wise; the single-
    # ABCD path doesn't see L_chief unless the caller passes it
    # explicitly.  Missing this factor is a constant piston that
    # only matters when interfering the GBD output with another
    # (separately-propagated) reference arm.
    if axial_opl is not None:
        k0 = 2.0 * np.pi / wavelength
        axial_phase = xp.exp(1j * k0 * float(axial_opl))
        qratio = qratio * axial_phase
    new_amplitude = beamlets.amplitude * qratio

    return BeamletBundle(
        positions=new_positions,
        directions=new_directions,
        Q=Q_new,
        amplitude=new_amplitude,
        waist0=beamlets.waist0,
    )


def _frame_from_normal(n: np.ndarray) -> np.ndarray:
    """Right-handed orthonormal 3x3 whose 3rd column is the unit normal ``n``
    (the output-plane propagation direction).  The in-plane x/y axes are a
    deterministic Gram-Schmidt against world ``+x`` (or ``+y`` if ``n`` is
    nearly along ``+x``)."""
    n = np.asarray(n, dtype=np.float64)
    n = n / np.linalg.norm(n)
    up = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0,
                                                                     0.0])
    xh = np.cross(up, n)
    xh = xh / np.linalg.norm(xh)
    yh = np.cross(n, xh)
    return np.stack([xh, yh, n], axis=1)   # columns = x_hat, y_hat, n_hat


def _unfolded_equivalent_surfaces(prescription, wavelength):
    """Straight-line surface list equivalent (for Q evolution) to a folded
    prescription: each optical surface at its cumulative WORLD path distance,
    with **flat** fold mirrors replaced by transmissive identity surfaces.

    Rationale: a fold mirror makes the sequential trace reflect (the axial
    direction cosine ``N`` flips sign, so the slope ``u = L/N`` flips and the
    slope-based Q evolution is corrupted); on-axis / flat-fold systems have
    isotropic Q that is invariant to the fold, so the straight equivalent (same
    powers, same geometric gaps, no reflection) reproduces the true Q.  A
    **curved** fold mirror has optical power that a transmissive surface does
    not, so it is *not* equivalent -- that case raises ``NotImplementedError``
    (it needs the full world per-surface differential transfer).
    """
    import copy as _copy

    from ..raytrace.world import world_surfaces_from_prescription
    wsurfs = world_surfaces_from_prescription(prescription)
    out = []
    for k, s in enumerate(wsurfs):
        if bool(getattr(s, 'is_mirror', False)) and np.isfinite(
                getattr(s, 'radius', np.inf)):
            raise NotImplementedError(
                'world_output_plane: curved (powered) fold mirrors are not yet '
                'supported (a flat fold is invariant to Q, a curved one is '
                'not).  Use per-surface world differential transfer, or a flat '
                'fold.')
        s2 = _copy.copy(s)
        s2.is_mirror = False           # flat mirror -> transmissive identity
        s2.world_origin = None
        s2.world_R = None
        if k < len(wsurfs) - 1:
            gap = float(np.linalg.norm(
                np.asarray(wsurfs[k + 1].world_origin, np.float64)
                - np.asarray(wsurfs[k].world_origin, np.float64)))
            s2.thickness = gap
        else:
            s2.thickness = 0.0
        out.append(s2)
    return out


def _resolve_world_output_plane(prescription, wavelength, spec):
    """Resolve ``world_output_plane`` to ``(p0, R_out, world_surfaces)``.

    ``spec='auto'`` -> plane at the paraxial image in world coords, normal along
    the (folded) chief-ray direction.  ``spec=(p0, R_out)`` -> used directly
    (``R_out`` columns = plane x_hat, y_hat, normal).
    """
    from ..raytrace.world import (
        paraxial_focus_world,
        world_surfaces_from_prescription,
    )
    wsurfs = world_surfaces_from_prescription(prescription)
    if isinstance(spec, str) and spec == 'auto':
        p0, nrm = paraxial_focus_world(wsurfs, wavelength)
        return np.asarray(p0, np.float64), _frame_from_normal(nrm), wsurfs
    p0, R_out = spec
    return np.asarray(p0, np.float64), np.asarray(R_out, np.float64), wsurfs


def apply_prescription_persurface_to_beamlets(
    beamlets: BeamletBundle,
    prescription: Dict[str, Any],
    wavelength: float,
    *,
    z_image: Optional[float] = None,
    world_output_plane: Any = None,
    jacobian: str = 'auto',
) -> BeamletBundle:
    """Per-surface tensor-Q evolution of a beamlet bundle through a prescription.

    The per-element form of :func:`apply_abcd_to_beamlets` (which applies a
    single whole-system paraxial ABCD).  Each beamlet's complex parameter is
    evolved **surface by surface** via the real per-ray differential ray
    transfer (:func:`lumenairy.raytrace.ray_transfer_jacobian`), so the
    scalar ``Q`` promotes to a ``(N, 2, 2)`` complex-symmetric TENSOR that
    captures off-axis **astigmatism** and higher-order aberration the paraxial
    system ABCD cannot.  The beamlets are carried through the refracting
    surfaces (generalized Collins ``Q_out = (C + D Q)(A + B Q)^{-1}``, amplitude
    ``1/sqrt(det(A + B Q))`` accumulated per surface so the sqrt branch stays
    unambiguous, plus the base-ray OPL piston), then a branch-safe tensor
    free-space to the image plane.

    NumPy backend (the trace + finite-difference Jacobian are numpy); the
    returned bundle's tensor ``Q`` reconstructs via the tensor branch of
    :func:`reconstruct_field_from_beamlets`.  Dead beamlets (vignette / TIR /
    miss) are dropped.

    Parameters
    ----------
    beamlets : BeamletBundle
        Source-plane bundle (scalar or tensor Q).
    prescription : dict
    wavelength : float
    z_image : float, optional
        Axial distance from the last surface vertex to the output/image plane
        [m].  Defaults to the system back focal length
        (``system_abcd_prescription`` BFL).
    """
    import copy as _copy

    from ..raytrace import surfaces_from_prescription, system_abcd_prescription
    from ..raytrace.differential import (
        ray_transfer_jacobian,
        ray_transfer_jacobian_analytic,
    )

    # v5.21 (#3): 'auto' (the default) prefers the truncation-free analytic
    # differential ray transfer (one dual-number trace, no per-surface (N,4,4)
    # inverse) and transparently falls back to the finite-difference Jacobian
    # (9N traces) for any surface type the analytic path does not yet cover
    # (raising NotImplementedError).  The two agree to ~1e-8 where both apply;
    # analytic is exact (no FD step), so 'auto' never lowers accuracy vs 'fd'.
    if jacobian == 'auto':
        _jac_candidates = [ray_transfer_jacobian_analytic, ray_transfer_jacobian]
    elif jacobian == 'analytic':
        _jac_candidates = [ray_transfer_jacobian_analytic]
    elif jacobian == 'fd':
        _jac_candidates = [ray_transfer_jacobian]
    else:
        raise ValueError(
            f"jacobian must be 'auto', 'fd' or 'analytic', got {jacobian!r}.")

    if world_output_plane is not None:
        # Evolve Q on the UNFOLDED-EQUIVALENT straight system: a fold mirror
        # makes the local trace reflect (N flips sign -> the slope ``u = L/N``
        # flips, corrupting the slope phase-space Q evolution).  The straight
        # equivalent (flat mirrors -> transmissive identity, thicknesses = world
        # geometric gaps) has no sign flip and the true path lengths, so Q
        # evolves correctly; the fold geometry re-enters only via the world
        # trace in the reframe.  Base-ray positions then come from the world
        # trace, Q from this straight evolution.
        surfs = _unfolded_equivalent_surfaces(prescription, wavelength)
    else:
        surfs = list(surfaces_from_prescription(prescription))
        if z_image is None:
            _res = system_abcd_prescription(prescription, wavelength)
            z_image = float(_res[2])
    # Trace to the LAST surface vertex (zero its transfer); the image-side
    # free-space is done branch-safe below (its single leg crosses the focus).
    surfs[-1] = _copy.copy(surfs[-1])
    try:
        surfs[-1].thickness = 0.0
    except (AttributeError, TypeError):
        pass

    pos = np.asarray(beamlets.positions)
    dr = np.asarray(beamlets.directions)
    x = pos[:, 0].astype(np.float64)
    y = pos[:, 1].astype(np.float64)
    Nz = dr[:, 2]
    ux = (dr[:, 0] / Nz).astype(np.float64)
    uy = (dr[:, 1] / Nz).astype(np.float64)

    dt = None
    _jac_err = None
    for _jac in _jac_candidates:
        try:
            dt = _jac(x, y, ux, uy, surfs, wavelength, per_surface=True)
            break
        except NotImplementedError as _e:
            _jac_err = _e            # analytic unsupported here -> try next (FD)
    if dt is None:
        raise (_jac_err or RuntimeError(
            'apply_prescription_persurface_to_beamlets: no ray-transfer '
            'Jacobian backend succeeded.'))
    n = x.shape[0]
    I2 = np.eye(2)[None, :, :]
    Qsrc = np.asarray(beamlets.Q)
    Q = (Qsrc.astype(np.complex128) if _q_is_tensor(Qsrc)
         else Qsrc[:, None, None] * I2)
    amp = np.ones(n, dtype=np.complex128)
    k0 = 2.0 * np.pi / wavelength

    for k in range(dt.jacobian.shape[0]):
        J = dt.jacobian[k]
        A = J[:, 0:2, 0:2]
        B = J[:, 0:2, 2:4]
        C = J[:, 2:4, 0:2]
        D = J[:, 2:4, 2:4]
        ABQ = A + B @ Q
        amp = amp / np.sqrt(_det2x2(ABQ))
        Q = (C + D @ Q) @ _inv2x2(ABQ, np)
        Q = 0.5 * (Q + np.transpose(Q, (0, 2, 1)))
    # base-ray OPL piston (input plane -> last vertex): where the wavefront
    # aberration enters (different beamlets accumulate different OPL).
    amp = amp * np.exp(1j * k0 * dt.opd)

    if world_output_plane is not None:
        # Reframe onto a WORLD-frame output plane (large folds -- e.g. a
        # periscope -- reverse the propagation axis, so the fixed local x-y
        # grid is meaningless).  World-trace the base rays through the folded
        # prescription, express each at the plane's local (x, y), and free-space
        # the (isotropic-or-tensor) Q to the plane.  See _reframe_beamlets_to_
        # world_plane.  Reduces to the local +z path on an unfolded system.
        new_pos, new_dir, Q, amp2, alive = _reframe_beamlets_to_world_plane(
            beamlets, prescription, wavelength, Q, amp, k0, world_output_plane)
        new_amp = np.asarray(beamlets.amplitude) * amp2
        w0 = np.asarray(beamlets.waist0)
        return BeamletBundle(
            positions=new_pos[alive], directions=new_dir[alive], Q=Q[alive],
            amplitude=new_amp[alive], waist0=w0[alive])

    # branch-safe tensor free-space to the image plane (per-eigenvalue sqrt so
    # the focus-crossing Gouy phase is unambiguous; eigenvalues have Im<0).
    inv = 1.0 / np.sqrt(1.0 + dt.ux ** 2 + dt.uy ** 2)
    Lx = dt.ux * inv
    My = dt.uy * inv
    Nz2 = inv
    # v5.22 FIX (per-surface GBD OPL piston): the differential trace leaves the
    # bundle at the last-surface INTERSECTION (z = sag(r)), not the vertex plane
    # -- dt.opd / dt.x / dt.y are all referenced there.  The image-side leg below
    # assumes the vertex plane (z = 0), so a POWERED last surface injects a
    # spurious sag(r) of defocus into the OPL piston (measured as ~2x the exit
    # wavefront curvature for a curved exit surface; the error vanishes only when
    # the last surface happens to be planar).  Every OTHER consumer of the
    # ray-transfer OPL applies the same SIGNED (not abs) `-sag/N` vertex
    # correction -- see _lens_traced.py (`-final.z / final.N`) and _lens_jax.py;
    # this path was the sole omission.  Fold -sag(r) into the leg length so the
    # piston, transverse position and Q are all referenced to the vertex plane.
    _Rl = float(getattr(surfs[-1], 'radius', np.inf))
    _kl = float(getattr(surfs[-1], 'conic', 0.0) or 0.0)
    if np.isfinite(_Rl) and _Rl != 0.0:
        _cl = 1.0 / _Rl
        _r2 = dt.x ** 2 + dt.y ** 2
        _sag = _cl * _r2 / (1.0 + np.sqrt(np.maximum(
            1.0 - (1.0 + _kl) * _cl * _cl * _r2, 0.0)))
    else:
        _sag = np.zeros_like(dt.x)
    t = (z_image - _sag) / Nz2
    Q, amp = _freespace_tensor_moebius_np(Q, amp, t, k0)   # shared (S2-14)
    new_pos = np.stack([dt.x + t * Lx, dt.y + t * My,
                        np.zeros_like(dt.x)], axis=-1)
    new_dir = np.stack([Lx, My, Nz2], axis=-1)
    new_amp = np.asarray(beamlets.amplitude) * amp
    # Scrub any non-finite row so no NaN enters the coherent sum -- the same
    # mask the world branch applies (D2): defense-in-depth with the FD
    # backend's companion-aliveness fix for FD-fallback surfaces (aspheres /
    # freeforms) with rim-adjacent beamlets.
    alive = (np.asarray(dt.alive, dtype=bool)
             & np.isfinite(new_pos).all(axis=1) & np.isfinite(new_amp)
             & np.isfinite(Q).all(axis=(1, 2)))
    w0 = np.asarray(beamlets.waist0)
    return BeamletBundle(
        positions=new_pos[alive], directions=new_dir[alive], Q=Q[alive],
        amplitude=new_amp[alive], waist0=w0[alive])


def _reframe_beamlets_to_world_plane(beamlets, prescription, wavelength, Q, amp,
                                     k0, world_output_plane):
    """Reframe straight-evolved beamlets (tensor Q at the unfolded last vertex)
    onto a world-frame output plane.

    World-traces the base rays through the folded prescription, lifts the output
    to world coords, expresses each in the plane's local frame, propagates each
    beamlet to the plane along its own ray, and free-spaces the (branch-safe)
    tensor Q by that distance -- rotating Q into the plane's transverse frame
    first (a no-op for the isotropic Q of an on-axis / flat-fold system).  ``Q``
    was evolved on the unfolded-equivalent straight system (see
    :func:`_unfolded_equivalent_surfaces`), so the geometric world path length
    to the last vertex matches the straight one and a single ``t_geom`` leg
    (last vertex -> plane) is exact.

    Returns ``(positions, directions, Q, amp, alive)`` in the plane's local
    frame (ready for :func:`reconstruct_field_from_beamlets` at ``centre=(0,0)``).
    """
    from ..raytrace.trace import _make_bundle
    from ..raytrace.world_trace import trace_world

    p0, R_out, wsurfs = _resolve_world_output_plane(
        prescription, wavelength, world_output_plane)
    last = wsurfs[-1]

    pos = np.asarray(beamlets.positions)
    dr = np.asarray(beamlets.directions)
    x = pos[:, 0].astype(np.float64)
    y = pos[:, 1].astype(np.float64)
    Nz = dr[:, 2]
    L0 = (dr[:, 0] / Nz)
    M0 = (dr[:, 1] / Nz)
    inv0 = 1.0 / np.sqrt(1.0 + L0 ** 2 + M0 ** 2)
    rb = _make_bundle(x.copy(), y.copy(), (L0 * inv0), (M0 * inv0), wavelength)
    res = trace_world(rb, wsurfs, wavelength, output_filter='last')
    img = res.image_rays
    alive = np.asarray(img.alive, dtype=bool)

    # lift the last-surface LOCAL output state to WORLD, then into the plane
    # frame (R_out columns = x_hat, y_hat, n_hat; R_out^T @ v == v @ R_out).
    loc = np.stack([img.x, img.y, img.z], axis=-1)
    p_w = last.world_origin[None, :] + loc @ last.world_R.T
    d_w = np.stack([img.L, img.M, img.N], axis=-1) @ last.world_R.T
    d_w = d_w / np.linalg.norm(d_w, axis=1, keepdims=True)
    p_l = (p_w - p0[None, :]) @ R_out
    d_l = d_w @ R_out
    # propagate the base ray (and Q) to the plane (local z = 0)
    dz = np.where(np.abs(d_l[:, 2]) < 1e-12, 1e-12, d_l[:, 2])
    t = -p_l[:, 2] / dz
    p_hit = p_l + t[:, None] * d_l

    # rotate Q into the plane's transverse frame (isotropic Q -> no-op) then
    # branch-safe free-space by t.
    R2 = R_out[:, 0:2].T @ last.world_R[:, 0:2]        # (2,2)
    Q = R2[None] @ Q @ R2.T[None]
    Q = 0.5 * (Q + np.transpose(Q, (0, 2, 1)))
    Q, amp = _freespace_tensor_moebius_np(Q, amp, t, k0)   # shared (S2-14)

    positions = np.stack([p_hit[:, 0], p_hit[:, 1],
                          np.zeros_like(p_hit[:, 0])], axis=-1)
    directions = d_l
    # kill any non-finite (dead) rows so no NaNs enter the coherent sum.
    alive = alive & np.isfinite(positions).all(axis=1) & np.isfinite(amp) \
        & np.isfinite(Q).all(axis=(1, 2))
    return positions, directions, Q, amp, alive


def propagate_gbd_through_prescription(
    E_in: np.ndarray,
    dx: float,
    prescription: Dict[str, Any],
    *,
    wavelength: float,
    output_shape: Optional[Tuple[int, int]] = None,
    output_grid: Optional[Tuple[int, int]] = None,
    output_dx: Optional[float] = None,
    output_centre: Tuple[float, float] = (0.0, 0.0),
    waist_factor: float = 1.0,
    sample_step: int = 1,
    chunk_beamlets: int = 2048,
    direction_sampling: Any = 'auto',
    per_surface: bool = False,
    z_image: Optional[float] = None,
    world_output_plane: Any = None,
    jacobian: str = 'auto',
    use_gpu: bool = False,
) -> np.ndarray:
    """End-to-end GBD through a sequential lumenairy prescription
    via system ABCD evolution.

    ``use_gpu=True`` runs the O(N_beamlets x N_pixels) coherent reconstruction
    (the dominant cost) on the GPU via CuPy -- the bundle is moved to the device
    after the (NumPy) per-surface / ABCD evolution and the reconstruction is
    backend-generic (measured ~71x at N=128, matching the CPU result to ~1e-15).
    Returns a CuPy device array (``cupy.asnumpy`` / ``backend.to_numpy`` to pull
    to host).  Requires CuPy + a CUDA device.

    Decomposes the source field into a regular grid of Gaussian
    beamlets, transforms each beamlet's complex Q-parameter and
    base ray by the prescription's paraxial system ABCD matrix,
    and coherently reconstructs the output field.

    Two forms, selected by ``per_surface``:

    * ``per_surface=False`` (default) -- the **paraxial** form: a single
      whole-system ABCD applied beamlet-by-beamlet (the Collins integral).
      Exact for well-corrected / paraxial systems; carries no aberration.
      The ABCD is air-to-air (front vertex -> back vertex), so the field is
      reconstructed at the **exit vertex**; ``z_image`` is *not* consumed on
      this path (passing it warns).
    * ``per_surface=True`` -- the **per-element** form: each beamlet's
      complex parameter is evolved surface-by-surface via the real per-ray
      differential ray transfer, promoting ``Q`` to a ``(N, 2, 2)`` tensor
      that captures off-axis **astigmatism** (tangential/sagittal focal
      separation growing ~field^2) and higher-order aberration.  NumPy
      backend; see :func:`apply_prescription_persurface_to_beamlets` and
      :func:`lumenairy.raytrace.ray_transfer_jacobian`.  ``z_image`` sets the
      last-vertex-to-output distance (defaults to the system BFL).

    ``direction_sampling`` (N4, 2026-07-19) defaults to ``'auto'``: the
    beamlet launch direction rides the input field's local wavevector (the
    Husimi / carrier-normal decomposition) when the input carries transverse
    angular content -- a curved (diverging / converging) or tilted wavefront --
    and along the axis otherwise, so a diverging / converging source focuses at
    its true finite-conjugate image and conserves power instead of collapsing to
    the collimated focal plane and shedding frame energy.  A flat-wavefront
    (collimated) input measures ~0 spread and takes the byte-identical axial
    path.  ``True`` / ``False`` force Husimi / axial explicitly; see
    :func:`decompose_field_to_beamlets`.

    Parameters
    ----------
    E_in : array (Ny, Nx) complex
        Source-plane field.
    dx : float
        Source-grid pitch (m).
    prescription : dict
    wavelength : float
    output_grid, output_dx, output_centre : grid geometry
    waist_factor, sample_step, chunk_beamlets : decomposition tuning
    z_image : float, optional
        Last-vertex -> output-plane distance (m).  Honored **only** when
        ``per_surface=True`` (defaults there to the system BFL, i.e. the
        focus).  Ignored on the default ``per_surface=False`` path, which
        always lands at the exit vertex; passing it there emits a
        ``RuntimeWarning``.

    Returns
    -------
    array (Ny, Nx) complex
        Output-plane reconstructed field.  ``per_surface=False`` reconstructs
        at the **exit vertex**; ``per_surface=True`` at ``z_image`` (default
        BFL / focus).

    .. versionchanged:: 5.2
        ``output_grid`` -> ``output_shape`` rename (AUDIT_V4_13_1 Part 2
        P1-A); see :func:`propagate_gbd_freespace`.
    """
    from ..raytrace import system_abcd_prescription

    Ny, Nx = _resolve_output_shape(
        output_shape, output_grid,
        fn_name='propagate_gbd_through_prescription',
        default_shape=(E_in.shape[-2], E_in.shape[-1]),
    )
    if output_dx is None:
        output_dx = dx

    # N4 (2026-07-19): resolve the H7 carrier-normal ('auto') launch policy here
    # too, so the prescription-chain entry conserves power on a diverging /
    # converging / tilted input exactly as ``apply_real_lens_gbd`` does -- the
    # position-only (axial) decomposition carries the input's wavefront curvature
    # in the beamlet AMPLITUDE only, so the axial base rays focus at the
    # COLLIMATED plane and the frame sheds the diverging beam's energy (power
    # collapses to ~0.2-0.7 on the dual-oracle singlet).  'auto' (the default)
    # measures the input's RMS local-tilt spread and launches Husimi
    # (carrier-normal) beamlets when the wavefront is curved/tilted, axial
    # otherwise; a flat-wavefront (collimated) field measures exactly 0 and takes
    # the byte-identical axial path (pin: the whole existing collimated GBD
    # matrix reproduces bit-for-bit).  ``True`` / ``False`` force Husimi / axial.
    if direction_sampling == 'auto':
        _spread = _input_angular_spread(E_in, dx, dx, wavelength)
        direction_sampling = _spread > _GBD_AUTO_HUSIMI_THRESH
    elif direction_sampling not in (True, False):
        raise ValueError(
            "propagate_gbd_through_prescription: direction_sampling must be "
            f"'auto', True or False, got {direction_sampling!r}.")

    _warn_if_tilted_no_dirsample(E_in, dx, None, wavelength, direction_sampling)
    bundle = decompose_field_to_beamlets(
        E_in, dx, wavelength=wavelength,
        waist_factor=waist_factor,
        sample_step=sample_step,
        direction_sampling=direction_sampling,
    )

    if world_output_plane is not None and not per_surface:
        # The whole-system-ABCD path drops coord-breaks (is fold-blind), so a
        # world output plane requires the per-surface (world-traced) evolver.
        per_surface = True

    if z_image is not None and not per_surface:
        # S2-4: ``z_image`` is only consumed on the per_surface=True path
        # (last-vertex -> output leg, default = BFL).  The whole-system-ABCD
        # path below reconstructs at the exit vertex and never propagates the
        # image leg, so a passed ``z_image`` would be silently dropped -- the
        # output plane is the exit vertex regardless.  Warn rather than raise
        # to preserve back-compat (the returned field is unchanged).
        warnings.warn(
            "propagate_gbd_through_prescription: z_image is only honored on "
            "the per_surface=True path (last-vertex -> output leg, default = "
            "BFL); with per_surface=False the field is reconstructed at the "
            "exit vertex and z_image is ignored.  Pass per_surface=True to "
            "land at z_image.", RuntimeWarning, stacklevel=2)

    if per_surface:
        # Per-element form: evolve each beamlet's tensor Q surface-by-surface
        # via the real per-ray differential ray transfer (captures off-axis
        # astigmatism / aberration the single whole-system ABCD cannot).  With
        # ``world_output_plane`` the output is reconstructed on a world-frame
        # plane (large folds reverse the propagation axis; the fixed local x-y
        # grid is then meaningless) -- the plane is the local origin, so the
        # reconstruction centre is (0, 0).
        bundle = apply_prescription_persurface_to_beamlets(
            bundle, prescription, wavelength, z_image=z_image,
            world_output_plane=world_output_plane, jacobian=jacobian)
        centre = (0.0, 0.0) if world_output_plane is not None else output_centre
        if use_gpu:
            bundle = _bundle_to_backend(bundle, _gpu_namespace())
        return reconstruct_field_from_beamlets(
            bundle, Ny=Ny, Nx=Nx, dx=output_dx,
            centre=centre, wavelength=wavelength,
            chunk_beamlets=chunk_beamlets, window=_GBD_RECONSTRUCT_WINDOW,
        )

    # Get the system's paraxial ABCD matrix.  ``system_abcd_prescription``
    # returns ``(matrix, efl, bfl)``.
    abcd_result = system_abcd_prescription(prescription, wavelength)
    if isinstance(abcd_result, tuple):
        M = abcd_result[0]
    else:
        M = abcd_result
    A = float(M[0, 0])
    B = float(M[0, 1])
    C = float(M[1, 0])
    D = float(M[1, 1])

    # 4.11.1 (H-AS-1): compute the axial OPL = sum_k n_k * t_k across
    # every glass/air segment of the prescription, out to the EXIT VERTEX
    # (this whole-system-ABCD path reconstructs THERE -- it does NOT add a
    # BFL / image-plane leg; that leg belongs to the per_surface=True
    # z_image path).  Pre-4.11.1 ``axial_opl=`` was never populated so the
    # per-beamlet
    # complex envelope lacked the system's axial phase reference and
    # multi-prescription reconstructions had the wrong piston relative
    # to ASM / Fresnel cross-checks.
    #
    # 4.11.2: the v4.11.1 implementation here was dead-on-arrival:
    # ``surfaces_from_prescription`` returns ``List[Surface]`` (Surface
    # is a @dataclass, not a dict), so ``_s.get('thickness', 0.0)``
    # raised AttributeError on the first iteration -- silently swallowed
    # by the surrounding bare ``except Exception`` and ``axial_opl``
    # always defaulted to None.  Switched to attribute access on the
    # Surface dataclass.  Caught by AUDIT_ROUND3_2026_05_16.md (CRIT-8).
    try:
        from ..glass import get_glass_index
        from ..raytrace import surfaces_from_prescription
        _surfs = surfaces_from_prescription(prescription)
        axial_opl = 0.0
        for _s in _surfs:
            _t = float(getattr(_s, 'thickness', 0.0) or 0.0)
            _glass = (getattr(_s, 'glass_after', None)
                      or getattr(_s, 'glass_before', None)
                      or 'air')
            try:
                _n = float(get_glass_index(_glass, wavelength))
            except (KeyError, ValueError, TypeError):
                # get_glass_index can raise: KeyError on unknown
                # catalogue / glass name, ValueError on
                # outside-Sellmeier-range wavelength, TypeError on
                # a malformed (non-string) glass identifier.  Fall
                # back to n=1.0 (air); this matches the v4.11.2
                # behaviour but no longer hides AttributeError /
                # ImportError from the broader try.
                _n = 1.0
            axial_opl += _n * _t
    except (ImportError, AttributeError, TypeError, ValueError,
            KeyError) as _exc:
        # Surface the failure rather than silently fall through to a
        # missing axial-phase reference; reconstruction still proceeds
        # without the piston.  Errors we expect from the inner block:
        # ImportError (raytrace / glass modules missing),
        # AttributeError (Surface dataclass missing expected field),
        # TypeError/ValueError (thickness/glass coercion failures),
        # KeyError (catalogue lookup re-raised).
        import warnings as _w
        _w.warn(
            f"propagate_gbd_through_prescription: axial-OPL "
            f"computation failed ({type(_exc).__name__}: {_exc}); "
            f"reconstructed field will lack the absolute axial-phase "
            f"reference and may not coherently superpose with other "
            f"propagator outputs.", RuntimeWarning, stacklevel=2)
        axial_opl = None

    # Apply ABCD to every beamlet.
    bundle = apply_abcd_to_beamlets(bundle, A, B, C, D,
                                     wavelength=wavelength,
                                     axial_opl=axial_opl)

    if use_gpu:
        bundle = _bundle_to_backend(bundle, _gpu_namespace())
    return reconstruct_field_from_beamlets(
        bundle, Ny=Ny, Nx=Nx, dx=output_dx,
        centre=output_centre, wavelength=wavelength,
        chunk_beamlets=chunk_beamlets, window=_GBD_RECONSTRUCT_WINDOW,
    )


__all__ = [
    'BeamletBundle',
    'decompose_field_to_beamlets',
    'recommend_gbd_sampling',
    'propagate_beamlets_freespace',
    'apply_thin_lens_to_beamlets',
    'apply_aperture_to_beamlets',
    'apply_abcd_to_beamlets',
    'apply_prescription_persurface_to_beamlets',
    'reconstruct_field_from_beamlets',
    'frame_completeness',
    'gbd_ghost_analysis',
    'propagate_gbd_freespace',
    'propagate_gbd_freespace_spectral',
    'propagate_gbd_freespace_vector',
    'propagate_gbd_vector_through_prescription',
    'propagate_gbd_thin_lens',
    'propagate_gbd_through_prescription',
]
