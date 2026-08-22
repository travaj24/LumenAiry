"""
Angular Spectrum Method (ASM) propagators
=========================================

v5.1.0 Agent C split: extracted from ``propagation.py``.  Contains the
exact-band-limited ASM kernel, the tilted / off-axis ASM, the batched
ASM variant, the shared ``_build_asm_H_square`` helper, and the
``apply_fresnel_curvature`` curvature-convention conversion utility.

All public symbols are re-exported by ``propagation.py`` for back-
compatibility with the pre-v5.1.0 import paths.

Author:  Andrew Traverso
"""

from __future__ import annotations

import importlib.util as _importlib_util_for_ne
from typing import Optional, Tuple, Union

import numpy as np

from . import fft_infra as _state
from .fft_infra import (
    CUPY_AVAILABLE,
    _fft2,
    _fft2_nd,
    _get_or_make_bandlimit,
    _get_or_make_freq_grids,
    _h_cache_lookup,
    _h_cache_store,
    _ifft2,
    _ifft2_nd,
    _is_cupy_array,
    _validate_propagator_inputs,
)

# ---------------------------------------------------------------------------
# Optional numexpr fused-expression backend (lazy) -- S5-8b (perf, no-loss).
# ---------------------------------------------------------------------------
# The ASM transfer-function build is dominated by the elementwise complex
# ``exp(1j * kz * z)`` over the full (Ny, Nx) grid.  numexpr's fused,
# multi-threaded kernel evaluates it ~1.6x faster and BYTE-IDENTICALLY to
# ``np.exp`` (audit AUDIT_V5_24_2 S5-8: measured max |diff| = 0.0; re-verified
# on numexpr 2.14).  Gated on availability + a grid-size floor + the numpy
# backend (numexpr does not operate on CuPy / JAX arrays), mirroring the
# phase-screen path in ``elements/_lens_real.py``.
NUMEXPR_AVAILABLE = _importlib_util_for_ne.find_spec('numexpr') is not None
_ne = None


def _ensure_numexpr_loaded():
    global _ne
    if _ne is None and NUMEXPR_AVAILABLE:
        import numexpr as _n
        _ne = _n
    return _ne is not None


# Grid-size floor below which numexpr setup / thread dispatch outweighs the
# fused-kernel win; below it the plain ``np.exp`` path is used.  1<<18
# (262144) ~ a 512x512 grid; the measured 1.6x win is at 2048^2+.
_NE_MIN_SIZE = 1 << 18

#: Workspace band for the streamed transfer function (v5.40), in ELEMENTS.
#: 4 Mi elements holds the float64 kernel workspace (kz_sq, kz, the mask and
#: the complex band) to ~130 MB whatever the grid, which is what makes the
#: streamed path's saving monotone in N.  Bit-irrelevant: H is elementwise in
#: (row, column), so no band width changes a single bit of the result.
_ASM_STREAM_BAND_ELEMS = 1 << 22

__all__ = [
    'angular_spectrum_propagate',
    'angular_spectrum_propagate_tilted',
    # v5.1.0 (Wave-4 integration): ``angular_spectrum_propagate_batch``
    # is a power-user 3-D-input variant reached via
    # ``lumenairy.propagators.propagation.angular_spectrum_propagate_batch``;
    # never been in the top-level ``lumenairy.__all__`` so leaving it
    # in this submodule's ``__all__`` would create a V9-walker
    # symmetry violation.  Module-attribute-accessible.
    'apply_fresnel_curvature',
    '_build_asm_H_square',
    '_asm_H_from_kz',
]


def _asm_H_from_kz(kz, prop, z, target_cdtype, xp=np, use_numexpr=False):
    """Assemble the ASM transfer function ``exp(1j * kz * z)`` on the
    propagating set, zeroed on the evanescent set (``~prop``), at
    ``target_cdtype``.

    This is the single shared complex-exponential kernel for EVERY ASM /
    ASM-MFT transfer-function builder -- :func:`_build_asm_H_square`,
    :func:`_get_asm_H_natural` (host-JAX and chunked NumPy/CuPy branches),
    :func:`angular_spectrum_propagate_tilted`, and
    :func:`angular_spectrum_propagate_mft` (audit S2-10 consolidation).
    It carries the S2-3 complex64 mitigation **uniformly**:

    * ``complex128`` -- the direct complex exponential
      ``where(prop, exp(1j*kz*z), 0)``, numexpr-fused when
      ``use_numexpr`` is True (byte-identical to ``np.exp``; audit S5-8b).
    * ``complex64`` -- the phase ``kz*z`` is folded ``mod 2*pi`` in
      float64 and ``cos`` / ``sin`` are evaluated in float64 **before**
      the float32 cast, so a large ``kz*z`` (up to ~1e6 rad) does not
      hit the float32 precision floor and inject speckle-like noise.
      This is the mitigation the natural-layout builder
      (:func:`_get_asm_H_natural`) has always used; before v5.24.5 the
      square / tilted / MFT builders cast the complex128 ``exp`` result
      straight to complex64 and carried ~1 float32-ULP of avoidable phase
      error per bin vs the correctly-rounded value (audit S2-10 / S2-3).
      ``use_numexpr`` is ignored on this path (the mitigation is
      trigonometric, not a complex ``exp``).

    Parameters
    ----------
    kz : ndarray, float64
        Axial wavenumber ``sqrt(k^2 - kx^2 - ky^2)``; 0 on the evanescent
        set.  May be the full grid or a row-chunk of it.
    prop : ndarray of bool, same shape as ``kz``
        Propagating-mode mask (``kz_sq > 0``).
    z : float
        Propagation distance (m).  May be negative.
    target_cdtype : complex dtype
        ``complex64`` or ``complex128``.
    xp : module, default :mod:`numpy`
        Array backend (numpy / cupy).  For the host-JAX path this is
        NumPy (H is built on the host in float64, then moved to device).
    use_numexpr : bool, default False
        For the complex128 path only, fuse ``where(prop, exp(...), 0)``
        through numexpr.  The caller decides eligibility (numexpr
        available, numpy backend, grid-size floor).

    Returns
    -------
    H : ndarray, dtype ``target_cdtype``, same shape as ``kz``.
        Band-limit masking and FFT-layout shifting stay in the caller.
    """
    if np.dtype(target_cdtype) == np.complex64:
        # complex64: fold phase mod 2*pi in float64 BEFORE casting to
        # float32 so the float32 floor doesn't inject speckle-like noise.
        fdt = np.float32
        phase = xp.mod(kz * z, 2.0 * np.pi)
        c = xp.cos(phase).astype(fdt)
        s = xp.sin(phase).astype(fdt)
        H = xp.empty(kz.shape, dtype=target_cdtype)
        H.real[:] = xp.where(prop, c, fdt(0))
        H.imag[:] = xp.where(prop, s, fdt(0))
        return H
    # complex128 (the huge kernel argument stays in float64 either way).
    if use_numexpr and _ensure_numexpr_loaded():
        # S5-8b: byte-identical fused complex-exp (numexpr resolves the
        # ``kz`` / ``prop`` / ``z`` names from this frame's locals).
        H = _ne.evaluate('where(prop, exp(1j * kz * z), 0j)')
    else:
        H = xp.where(prop, xp.exp(1j * kz * z), 0)
    if H.dtype != np.dtype(target_cdtype):
        H = H.astype(target_cdtype)
    return H


def _build_asm_H_square(
    N,
    dx,
    z,
    wavelength,
    dtype=None,
    bandlimit=True,
):
    """Build a square (N x N) band-limited Angular-Spectrum transfer
    function on the canonical centered frequency grid.

    This is the single source of truth for the centered ASM ``H``
    construction used by:

    * :func:`angular_spectrum_propagate` (square-grid path / JAX path
      / one-shot fallback when chunking is not needed).
    * :func:`lumenairy.analysis.detector.shack_hartmann` (per-lenslet
      sub-aperture propagation).

    Conventions
    -----------
    * Output is **centered** (not ``fftshift``-ed), matching the
      ``E_fft = fftshift(fft2(ifftshift(E))) ; E_out = fftshift(
      ifft2(ifftshift(E_fft * H)))`` propagation idiom both call sites
      use.
    * Frequency grid is ``(arange(N) - N // 2) / (N * dx)``, i.e. the
      same centered convention as :func:`_get_or_make_freq_grids`
      with square ``dy == dx`` -- an INTEGER DC anchor, so the grid is
      exactly ``fftshift(fftfreq(N, dx))`` for both parities of N
      (audit P1).
    * Evanescent modes (``kz_sq <= 0``) are zeroed for ``z != 0``.
    * When ``bandlimit`` is True and ``z != 0`` the 1-D mask
      ``|f| < L / (2*lambda*|z|)`` is applied as the outer product of the
      per-axis masks.  v5.30 (audit P12): that cutoff is the
      **z -> infinity asymptote** of Matsushima & Shimobaba's exact
      local-frequency limit ``1/(lambda*sqrt((2z/L)^2 + 1))``, not the
      exact limit itself; it is strictly larger, so it never over-filters
      (up to 2.24x too wide at ``z = L/4``, within 0.5% for ``z >= 5 L``).
      See :func:`~lumenairy.propagators.fft_infra._get_or_make_bandlimit`
      for the derivation and the measured table.
    * ``z == 0`` short-circuits to ``H = 1`` for EVERY bin (evanescent
      bins included) -- i.e. the exact identity; both the bandlimit and
      the evanescent mask are bypassed so ASM(z=0) reproduces the input
      for any grid, sub-wavelength grids included.  (audit S2-11)

    Parameters
    ----------
    N : int
        Square grid size (Ny == Nx == N).
    dx : float
        Pixel pitch (m).  ``dy == dx`` is assumed.
    z : float
        Propagation distance (m).  May be negative for back-propagation.
    wavelength : float
        Vacuum wavelength (m).
    dtype : numpy dtype, optional
        Target complex dtype.  Defaults to ``np.complex128``.  Real
        dtypes are promoted to ``np.complex128``.
    bandlimit : bool, default True
        Apply the 1-D bandlimit mask ``|f| < L / (2*lambda*|z|)`` -- the
        z -> infinity asymptote of the Matsushima-Shimobaba cutoff (audit
        P12; see the Notes above).

    Returns
    -------
    H : ndarray, shape (N, N), dtype as requested
        The centered ASM transfer function.

    Notes
    -----
    Numerical equivalence to the inline path is bit-exact for
    matching ``N``, ``dx``, ``z``, ``wavelength``, and ``bandlimit``
    arguments (same arithmetic; no caching / chunking detour).
    """
    if dtype is None or not np.issubdtype(dtype, np.complexfloating):
        dtype = np.complex128
    N = int(N)
    if z == 0:
        # (audit S2-11) z == 0 is the EXACT identity.  The transfer
        # function exp(1j*kz*z) equals 1 for every bin at z == 0 --
        # including evanescent bins, whose decay factor exp(-kappa*|z|)
        # -> 1 as z -> 0.  Returning all-ones here keeps ASM(z=0) equal
        # to the input for ANY grid, including sub-wavelength grids
        # (dx < lambda/2) whose spectrum contains evanescent bins.  The
        # general ``kz_sq > 0`` mask below would instead zero those bins
        # and make z == 0 differ from the input by rel err ~1,
        # contradicting the documented "returns the input unchanged"
        # contract and the dispatcher's z=None copy.
        return np.ones((N, N), dtype=dtype)
    k = 2.0 * np.pi / wavelength
    # audit P1 (2026-07-25): INTEGER DC anchor ``N // 2``.  The centred
    # layout this returns is consumed as ``fftshift(fft2(ifftshift(E)))
    # * H`` by both call sites, and fftshift anchors DC at the integer
    # index ``N // 2`` for every N.  Identical to ``N / 2`` for even N
    # (bit-identical H), but for ODD N the float anchor mislabelled every
    # bin by -df/2, i.e. evaluated the kernel at ``f_true - df/2`` -- a
    # linear phase in f, i.e. a lateral walk of ``-lambda*z/(2*N*dx)``.
    # Measured via the shack_hartmann consumer path (ideal lens, Np=65,
    # dx=1 um, lambda=633 nm, f=2 mm): focal-spot centroid -8.0874 px
    # pre-fix vs -0.1535 px post-fix (Np=64: -0.1896 px, unchanged).
    fx = (np.arange(N, dtype=np.float64) - N // 2) / (N * dx)
    fy = fx  # square sub-aperture (dy == dx)
    kx_sq = (2 * np.pi * fx) ** 2
    ky_sq = (2 * np.pi * fy) ** 2
    kz_sq = k ** 2 - kx_sq[None, :] - ky_sq[:, None]
    prop = kz_sq > 0
    kz = np.where(prop, np.sqrt(np.where(prop, kz_sq, 0.0)), 0.0)
    # S2-10: single shared kernel.  complex128 uses the direct
    # (numexpr-fused, S5-8b) complex exp; complex64 folds the phase mod
    # 2*pi in float64 before the float32 cast (S2-3 mitigation, now
    # applied to EVERY H builder, not just _get_asm_H_natural).
    _use_ne = (NUMEXPR_AVAILABLE and kz.size >= _NE_MIN_SIZE
               and _ensure_numexpr_loaded())
    H = _asm_H_from_kz(kz, prop, z, dtype, np, use_numexpr=_use_ne)
    if bandlimit and z != 0:
        L = N * dx
        f_max = L / (2 * wavelength * abs(z))
        bl_x = np.abs(fx) < f_max
        bl_y = np.abs(fy) < f_max
        mask = bl_x[None, :] & bl_y[:, None]
        H = H * mask.astype(dtype)
    return H


def _get_asm_H_natural(
    Ny: int,
    Nx: int,
    dy: float,
    dx: float,
    wavelength: float,
    z: float,
    bandlimit: bool,
    target_cdtype,
    xp,
    is_jax: bool = False,
    verbose: bool = False,
):
    """Fetch (or build + cache) the plain-ASM transfer function in
    NATURAL (``ifftshift``-ed) FFT layout.

    v5.17.x (P2-27): factored out of :func:`angular_spectrum_propagate`
    verbatim so :func:`angular_spectrum_propagate_batch` can obtain
    ``H`` without running a full wasted FFT+IFFT pair on a proxy field.
    The arithmetic is byte-identical to the pre-refactor inline block
    (same cache key, same chunked construction, same verbose output).

    Returns ``H`` in the natural (un-``fftshift``-ed) spectrum layout,
    which is how the v5.5.3 2-shift propagation fold consumes it.
    """
    k = 2 * np.pi / wavelength

    if z == 0:
        # (audit S2-11) z == 0 is the EXACT identity: exp(1j*kz*z) == 1 for
        # every bin, evanescent bins included (their decay exp(-kappa*|z|)
        # -> 1 as z -> 0).  Returning all-ones H makes
        # angular_spectrum_propagate reproduce the input to FFT round-trip
        # precision for ANY grid -- including sub-wavelength grids
        # (dx < lambda/2) whose spectrum contains evanescent bins -- and
        # makes the z=0 propagator path agree with the dispatcher's
        # ``propagate(method='asm', z=None)`` copy.  The ``kz_sq > 0`` mask
        # below would instead zero the evanescent bins and break the
        # documented "returns the input unchanged" contract.  All-ones is
        # identical in natural and centred FFT layout, so no ifftshift is
        # needed; it is trivially cheap to rebuild, so it is not cached.
        return xp.ones((Ny, Nx), dtype=target_cdtype)

    # 3.2.14 H cache
    # Geometry signature.  Hits return the previously-built H without
    # re-running the chunked kernel construction (~30-50% of total
    # ASM time on 2k+ grids).  CuPy device arrays and JAX traced
    # arrays are kept out of the cache (host-side dict can't safely
    # retain device pointers / traced objects).
    h_key = None
    H = None
    if xp is np:
        # 4.10: add 'ASM' tag string to the cache key so plain-ASM
        # entries are guaranteed disjoint from ASM_TILTED / ASM_MFT /
        # RS / SAS even if those keys ever evolve to the same tuple
        # length.  Defensive future-proofing.
        h_key = (int(Ny), int(Nx), float(dy), float(dx),
                 float(wavelength), float(z), bool(bandlimit),
                 np.dtype(target_cdtype).str, 'ASM')
        H = _h_cache_lookup(h_key)

    if H is None and is_jax:
        # JAX path.  The transfer function H is FIELD-INDEPENDENT (it
        # depends only on the grid geometry, wavelength and z), so we
        # build it on the HOST in float64 -- with the SAME mod-2pi fold
        # the chunked NumPy branch below uses -- and only then move it
        # onto the JAX device, casting to the target complex dtype.
        #
        # v5.24.4 (audit S2-3): building H directly with jax.numpy
        # silently evaluated the huge kernel argument ``kz * z`` (up to
        # ~1e6 rad) in float32 whenever ``jax_enable_x64`` is off -- the
        # JAX default -- because ``jnp.arange(dtype=float64)`` truncates
        # to float32 there, and no mod-2pi fold was applied.  That cost
        # ~26 dB of phase accuracy vs the documented "float32 noise
        # floor, does not degrade with phase magnitude" contract (rel
        # err ~2e-3 at z=8 mm, N=256; ~13,000x worse than NumPy c64).
        # Host-building in float64 restores the NumPy contract and is
        # trace-safe: H does not depend on the field, so the field
        # gradient survives (only gradients w.r.t. the concrete-float
        # geometry z/dx/wavelength are foregone).
        # audit P1: integer DC anchor ``N // 2`` -- this centred H is
        # ``ifftshift``-ed below, and ifftshift anchors DC at the integer
        # index.  Bit-identical for even N; fixes the odd-N half-bin
        # frequency offset (see _get_or_make_freq_grids' layout contract).
        fx = (np.arange(Nx, dtype=np.float64) - Nx // 2) / (Nx * dx)
        fy = (np.arange(Ny, dtype=np.float64) - Ny // 2) / (Ny * dy)
        kx_sq = (2 * np.pi * fx) ** 2
        ky_sq = (2 * np.pi * fy) ** 2
        kz_sq = k ** 2 - kx_sq[None, :] - ky_sq[:, None]
        prop = kz_sq > 0
        kz = np.where(prop, np.sqrt(np.where(prop, kz_sq, 0.0)), 0.0)
        # S2-10: shared kernel (numexpr-fused complex128; mod-2*pi float64
        # fold before the float32 cast for complex64 -- the S2-3 mitigation).
        _use_ne = (NUMEXPR_AVAILABLE and kz.size >= _NE_MIN_SIZE
                   and _ensure_numexpr_loaded())
        H_np = _asm_H_from_kz(kz, prop, z, target_cdtype, np,
                              use_numexpr=_use_ne)
        if bandlimit and z != 0:
            Lx = Nx * dx
            Ly = Ny * dy
            fx_max = Lx / (2 * wavelength * abs(z))
            fy_max = Ly / (2 * wavelength * abs(z))
            bl_x = np.abs(fx) < fx_max
            bl_y = np.abs(fy) < fy_max
            mask = bl_x[None, :] & bl_y[:, None]
            H_np = H_np * mask.astype(target_cdtype)
        # v5.5.3: store H in NATURAL (un-shifted) FFT layout so the per-call
        # propagation folds away the two spectrum-domain shifts (4 -> 2 shifts).
        H_np = np.fft.ifftshift(H_np)
        H = xp.asarray(H_np)

    if H is None:
        # Spatial-frequency squared vectors (cached on numpy path).
        kx_sq, ky_sq = _get_or_make_freq_grids(Ny, Nx, dy, dx, xp is np)
        if bandlimit and z != 0:
            bl_x, bl_y = _get_or_make_bandlimit(
                Ny, Nx, dy, dx, wavelength, abs(z), xp is np)
        else:
            bl_x = bl_y = None

        # S5-8g (perf, no-loss): build H DIRECTLY in the natural (un-shifted)
        # FFT layout by ``ifftshift``-ing the four 1-D input vectors ONCE
        # (cheap, length-N each) instead of ``ifftshift``-ing the assembled
        # (Ny, Nx) H at the end (a full-grid roll, ~28 ms/cold build at 2k^2).
        # ``ifftshift`` returns a fresh array, so the shared cached
        # freq-grid / band-limit vectors are NOT mutated.  Byte-identical: the
        # elementwise kernel commutes with the index permutation, so
        # build-then-shift == shift-then-build.
        kx_sq = xp.fft.ifftshift(kx_sq)
        ky_sq = xp.fft.ifftshift(ky_sq)
        if bl_x is not None:
            bl_x = xp.fft.ifftshift(bl_x)
            bl_y = xp.fft.ifftshift(bl_y)

        # Chunked H construction, sized to fit a small slice of RAM.
        from ..memory import get_ram_budget
        ram = get_ram_budget()
        row_cost = 3 * Nx * 16   # bytes per row of workspace (complex128)
        if row_cost > 0:
            max_chunk = max(1, int(ram * 0.1 / row_cost))
        else:
            max_chunk = Ny
        chunk = min(Ny, max_chunk)

        # S5-8b (perf, no-loss): fuse the elementwise complex exp through
        # numexpr when available (numpy backend, complex128, large grid) --
        # byte-identical to ``np.exp`` (verified max |diff| = 0.0), ~1.6x.
        _use_ne = (NUMEXPR_AVAILABLE and xp is np
                   and target_cdtype == np.complex128
                   and (Ny * Nx) >= _NE_MIN_SIZE
                   and _ensure_numexpr_loaded())

        H = xp.empty((Ny, Nx), dtype=target_cdtype)
        kept_count = 0
        for j0 in range(0, Ny, chunk):
            j1 = min(Ny, j0 + chunk)
            # kz_sq is float64 regardless of target dtype to keep the
            # huge kernel argument (kz * z up to ~1e6 rad) accurate.
            kz_sq_c = k**2 - kx_sq[None, :] - ky_sq[j0:j1, None]
            prop = kz_sq_c > 0
            kz_c = xp.where(prop, xp.sqrt(xp.maximum(kz_sq_c, 0)), 0)
            # S2-10: shared kernel -- numexpr-fused complex128; complex64
            # folds phase mod 2*pi in float64 before the float32 cast.
            H_c = _asm_H_from_kz(kz_c, prop, z, target_cdtype, xp,
                                 use_numexpr=_use_ne)
            if bl_x is not None:
                bl_mask = bl_x[None, :] & bl_y[j0:j1, None]
                H_c *= bl_mask
                if verbose:
                    kept_count += int(xp.sum(bl_mask))
            H[j0:j1, :] = H_c

        if verbose and bl_x is not None:
            kept_frac = kept_count / (Nx * Ny)
            print(f"  Band-limiting: keeping {kept_frac*100:.1f}% of spectrum")
        if verbose:
            print(f"  ASM propagation: z = {z*1e3:.3f} mm  "
                  f"(H cache miss, built in {chunk}-row chunks)")
            print(f"  Grid: {Ny}x{Nx}, dx={dx*1e6:.3f} um, dy={dy*1e6:.3f} um")
            print(f"  Wavelength: {wavelength*1e9:.1f} nm")
        # v5.5.3: H is cached in NATURAL (un-shifted) FFT layout so the per-call
        # propagation folds away the two spectrum-domain shifts.  S5-8g: it is
        # already built in natural order (the input freq/band-limit vectors were
        # ``ifftshift``-ed above), so the former full-grid ``H = ifftshift(H)``
        # is dropped -- byte-identical, one fewer (Ny, Nx) roll per cold build.
        # Store under the numpy key only.  The cached H is read-only
        # in normal use; we don't deep-copy on lookup, so callers must
        # not mutate it in place.
        if h_key is not None:
            _h_cache_store(h_key, H)
    elif verbose:
        print(f"  ASM propagation: z = {z*1e3:.3f} mm  (H cache HIT)")

    return H


def _asm_apply_H_streamed(
    spec: np.ndarray,
    Ny: int,
    Nx: int,
    dy: float,
    dx: float,
    wavelength: float,
    z: float,
    bandlimit: bool,
    verbose: bool = False,
) -> np.ndarray:
    """``spec *= H``, generating ``H`` a row band at a time (v5.40).

    The plain path materialises the whole (Ny, Nx) transfer function and then
    allocates a second full grid for the product.  At N = 32768 / complex64
    that is 2 x 8.59 GB for an operation whose operands are needed one row at
    a time, and H is not even cacheable there (it exceeds
    ``_H_CACHE_MAX_BYTES_PER_ENTRY``), so the grid is rebuilt on every call
    only to be thrown away.

    **Bit-identity is structural, not measured-and-hoped.**  Two facts carry
    it, and both are properties of the code rather than of any fixture:

    1. ``_get_asm_H_natural`` ALREADY builds H in row chunks, with exactly
       this expression on exactly these operands; ``H[j0:j1]`` therefore has
       the same bytes whatever the chunking, because every element of H
       depends on its own row and column alone.  This function reuses that
       kernel (``_asm_H_from_kz``) rather than restating it.
    2. ``np.multiply(a, b, out=a)`` yields the same bits as ``a * b`` when
       ``result_type(a, b) is a.dtype`` -- numpy does not reassociate or
       reorder an elementwise ufunc on account of an ``out=``.  The caller
       guarantees the dtype precondition (H is built at ``spec.dtype``); if it
       could not, it does not take this path.

    The band-limit mask, the ``ifftshift`` of the four 1-D input vectors and
    the numexpr eligibility rule are all the plain path's, unchanged.
    """
    k = 2 * np.pi / wavelength
    if z == 0:
        # H is all-ones at z = 0 (the exact identity, evanescent bins
        # included), so the multiply is skipped rather than performed against
        # a grid of ones.  ONE EDGE CASE IS NOT BIT-IDENTICAL AND IS STATED
        # RATHER THAN GLOSSED: numpy's naive complex product makes
        # ``(inf + 0j) * (1 + 0j)`` equal ``inf + nan*j``, so a spectrum
        # holding a non-finite value would come back differently here than
        # from the materialised path.  It is unreachable from a finite input
        # -- an FFT that overflows to inf produces nan across the transform,
        # and ``nan * (1 + 0j)`` is nan either way -- but it is a difference,
        # so it is written down.
        return spec
    kx_sq, ky_sq = _get_or_make_freq_grids(Ny, Nx, dy, dx, True)
    if bandlimit:
        bl_x, bl_y = _get_or_make_bandlimit(Ny, Nx, dy, dx, wavelength,
                                            abs(z), True)
    else:
        bl_x = bl_y = None
    kx_sq = np.fft.ifftshift(kx_sq)
    ky_sq = np.fft.ifftshift(ky_sq)
    if bl_x is not None:
        bl_x = np.fft.ifftshift(bl_x)
        bl_y = np.fft.ifftshift(bl_y)

    target_cdtype = spec.dtype
    _use_ne = (NUMEXPR_AVAILABLE
               and target_cdtype == np.complex128
               and (Ny * Nx) >= _NE_MIN_SIZE
               and _ensure_numexpr_loaded())

    # The band size is a FREE CHOICE -- H is elementwise in (row, column), so
    # no band width changes a single bit -- and the plain builder's choice is
    # the wrong one here.  It sizes the chunk at 10% of the RAM budget, which
    # on a large box resolves to the WHOLE grid below N ~ 8192; that is
    # harmless when the float64 kernel workspace (kz_sq, kz, prop: ~4 grids at
    # complex64) is the only thing live, and actively counter-productive here,
    # where the spectrum is live alongside it and the streamed path would then
    # cost MORE than the grid it avoids.  Measured at N = 4096: whole-grid
    # workspace read +1.69 grids over the plain path; the capped band reads
    # below it.  Cap the workspace instead, in elements.
    chunk = max(1, min(Ny, _ASM_STREAM_BAND_ELEMS // max(Nx, 1)))

    kept_count = 0
    for j0 in range(0, Ny, chunk):
        j1 = min(Ny, j0 + chunk)
        kz_sq_c = k**2 - kx_sq[None, :] - ky_sq[j0:j1, None]
        prop = kz_sq_c > 0
        kz_c = np.where(prop, np.sqrt(np.maximum(kz_sq_c, 0)), 0)
        H_c = _asm_H_from_kz(kz_c, prop, z, target_cdtype, np,
                             use_numexpr=_use_ne)
        if bl_x is not None:
            bl_mask = bl_x[None, :] & bl_y[j0:j1, None]
            H_c *= bl_mask
            if verbose:
                kept_count += int(np.sum(bl_mask))
        spec[j0:j1] *= H_c
    if verbose:
        if bl_x is not None:
            print(f"  Band-limiting: keeping "
                  f"{kept_count / (Nx * Ny) * 100:.1f}% of spectrum")
        print(f"  ASM propagation: z = {z*1e3:.3f} mm  "
              f"(H STREAMED in {chunk}-row bands, never materialised)")
    return spec


def angular_spectrum_propagate(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    bandlimit: bool = True,
    return_transfer_function: bool = False,
    use_gpu: bool = False,
    verbose: bool = False,
    stream_transfer_function: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Propagate an optical field using the Angular Spectrum Method (ASM).

    This function propagates a 2-D complex electric field through free space
    using the exact transfer function (no paraxial approximation).

    Parameters
    ----------
    E_in : ndarray (complex)
        Input electric field, shape (Ny, Nx).  Can be a NumPy or CuPy array.

    z : float
        Propagation distance in meters.
        Positive z = forward propagation (away from source).
        Negative z = backward propagation (toward source).

    wavelength : float
        Optical wavelength in meters (e.g. 1.31e-6 for 1310 nm).

    dx : float
        Grid spacing in x-direction in meters (e.g. 1e-6 for 1 um).

    dy : float, optional
        Grid spacing in y-direction in meters.  If None, assumes dy = dx.

    bandlimit : bool, default True
        If True, applies band-limiting to suppress Fresnel aliasing.
        The band-limit cutoff per axis is:  f_max = L / (2 * lambda * |z|).
        Recommended for large propagation distances.

    return_transfer_function : bool, default False
        If True, also returns the transfer function H.

    use_gpu : bool, default False
        If True and CuPy is available, performs computation on GPU.
        If *E_in* is already a CuPy array, GPU is used automatically.

    verbose : bool, default False
        If True, prints diagnostic information.

    stream_transfer_function : bool, default False
        Opt-in memory trim (v5.40).  Generate the transfer function one row
        band at a time DURING the frequency-domain multiply, in place on the
        spectrum, instead of materialising the full ``(Ny, Nx)`` ``H`` and
        allocating a second full grid for the product.  Saves two complex
        full-grid arrays -- 17.2 GB at ``N = 32768`` / complex64.

        **Byte-identical** to the default path (see
        :func:`_asm_apply_H_streamed` for the two-line argument), but NOT
        free: the streamed H is never cached, so a caller that repeats the
        same ``(shape, dtype, wavelength, z, bandlimit)`` pays the kernel
        construction on every call instead of once.  That trade is worth
        taking exactly where the memory matters -- above
        ``_H_CACHE_MAX_BYTES_PER_ENTRY`` (2 GB, i.e. ``N >= 16384`` at
        complex64) H is not cacheable anyway, so at those sizes the streamed
        path rebuilds no more often than the plain one and simply holds less.
        Below that it is a real cost, which is why the default is off.

        NumPy backend only; ignored (with the plain path taken) on CuPy and
        JAX inputs, and when ``return_transfer_function=True`` asks for the
        very grid this avoids building.

    Returns
    -------
    E_out : ndarray (complex)
        Propagated electric field, same shape and array type as *E_in*.

    H : ndarray (complex), optional
        Transfer function (only returned when *return_transfer_function=True*).

    Notes
    -----
    Sampling requirements for accurate results:

    1. ``dx < lambda / 2`` -- Nyquist for propagating waves.
    2. ``L > 2 * lambda * z / d_min`` -- avoids Fresnel aliasing, where
       L = N * dx is the grid extent and d_min is the smallest feature size
       to be resolved.

    Memory: approximately 3x the size of the input array (E_in, E_fft, H).

    Examples
    --------
    >>> import numpy as np
    >>> from lumenairy.propagation import angular_spectrum_propagate
    >>>
    >>> N = 512
    >>> dx = 1e-6                    # 1 um grid spacing
    >>> wavelength = 1.31e-6         # 1310 nm
    >>> x = (np.arange(N) - N/2) * dx
    >>> X, Y = np.meshgrid(x, x)
    >>> sigma = 10e-6                # 10 um beam waist
    >>> E_in = np.exp(-(X**2 + Y**2) / (2 * sigma**2)).astype(complex)
    >>>
    >>> E_out = angular_spectrum_propagate(E_in, z=1e-3,
    ...                                    wavelength=wavelength, dx=dx)
    >>> print(f"Input power:  {np.sum(np.abs(E_in)**2):.4f}")
    >>> print(f"Output power: {np.sum(np.abs(E_out)**2):.4f}")

    References
    ----------
    [1] Goodman, J.W. "Introduction to Fourier Optics" (3rd ed.), Ch. 3-4.
    [2] Matsushima, K. and Shimobaba, T. (2009). "Band-limited angular
        spectrum method for numerical simulation of free-space propagation
        in far and near fields." Opt. Express 17(22): 19662-19673.
        NOTE (v5.30, audit P12): ``bandlimit=True`` applies the
        ``z -> infinity`` asymptote ``L / (2*lambda*|z|)`` of this paper's
        local-frequency limit, not the exact expression.  The asymptote is
        the larger of the two, so it never over-filters; see
        :func:`~lumenairy.propagators.fft_infra._get_or_make_bandlimit`.
    """
    # v4.15.3 (P0-NEW-F2-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper.  v4.15.2 inlined the guard
    # here; v4.15.3 routes through the helper so future entry points
    # can't be added unguarded.  Runs FIRST (before any input
    # validation or backend dispatch) so the user gets a clear,
    # actionable error rather than a downstream AttributeError or a
    # silent wrong-axis FFT.
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'angular_spectrum_propagate',
                           input_kind='field')

    _validate_propagator_inputs(E_in, z, wavelength, dx, dy,
                                fn_name='angular_spectrum_propagate')

    # -- array library selection (NumPy / CuPy / JAX) ----------------------
    # JAX arrays bypass the chunked H construction and the host cache;
    # they take a one-shot all-NxN H and stay in the input backend.
    from ..backend import is_jax_array
    is_jax = is_jax_array(E_in)
    if is_jax:
        import jax.numpy as _jnp
        xp = _jnp
    elif CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        xp = _state.cp
        if not _is_cupy_array(E_in):
            E_in = _state.cp.asarray(E_in)
    else:
        xp = np
        if _is_cupy_array(E_in):
            E_in = E_in.get()  # CuPy -> NumPy when GPU not requested

    Ny, Nx = E_in.shape

    if dy is None:
        dy = dx

    # Target complex dtype for the transfer function and the output.
    # Inferred from E_in so the caller controls precision by the dtype of
    # the field they pass in.  Non-complex input (e.g. float arrays used
    # in examples) falls back to DEFAULT_COMPLEX_DTYPE.
    if xp.iscomplexobj(E_in):
        target_cdtype = E_in.dtype
    else:
        target_cdtype = np.dtype(_state.DEFAULT_COMPLEX_DTYPE)
        # v5.17.x (P2-26): cast the real-dtype field to the target complex
        # dtype BEFORE it reaches ``_fft2`` (mirrors the batch sibling).
        # Pre-fix, a real float32/float64 E_in was fed uncast into the
        # pyFFTW dispatcher, which rejects a real->complex in-place plan
        # with ``ValueError: Invalid direction``; the failure handler then
        # permanently blacklisted the bare SHAPE for ALL dtypes (so every
        # later complex128 call at that shape silently ran on scipy) and
        # emitted a misleading 'memory pressure' warning.
        E_in = E_in.astype(target_cdtype)

    # v5.40 (LEVER 3a): the streamed transfer function.  Taken only on the
    # NumPy backend and only when the caller does not also want H handed back
    # -- the whole point is that H never exists as a grid.  Everything else
    # about the call is unchanged, including the 2-shift fold below.
    if (stream_transfer_function and xp is np and not is_jax
            and not return_transfer_function):
        spec = _fft2(np.fft.ifftshift(E_in))
        if spec.dtype != target_cdtype:          # pragma: no cover - defensive
            spec = spec.astype(target_cdtype)
        _asm_apply_H_streamed(spec, Ny, Nx, dy, dx, wavelength, z, bandlimit,
                              verbose=verbose)
        return np.fft.fftshift(_ifft2(spec))

    # v5.17.x (P2-27): the H-cache lookup / chunked construction moved
    # verbatim into :func:`_get_asm_H_natural` (shared with the batch
    # variant).  Byte-identical to the pre-refactor inline block.
    H = _get_asm_H_natural(Ny, Nx, dy, dx, wavelength, z, bandlimit,
                           target_cdtype, xp, is_jax=is_jax, verbose=verbose)

    # -- propagate: E_out = IFFT{ FFT{E_in} * H } ---------------------------
    # H is stored NATURAL-layout, so the two spectrum-domain shifts fold away:
    #   fftshift(ifft2(ifftshift( fftshift(fft2(ifftshift(E)))*H_centred )))
    #   == fftshift(ifft2(           fft2(ifftshift(E))      *H_natural   ))
    # (ifftshift distributes over the elementwise product; ifftshift.fftshift =
    # id).  Algebraically EXACT for any N, even or odd -- 4 shifts -> 2.
    if xp is np:
        E_out = np.fft.fftshift(_ifft2(_fft2(np.fft.ifftshift(E_in)) * H))
    else:
        E_out = xp.fft.fftshift(
            xp.fft.ifft2(xp.fft.fft2(xp.fft.ifftshift(E_in)) * H))

    if return_transfer_function:
        # 4.10: return a copy so a caller that does ``E_out, H = ...(
        # return_transfer_function=True)`` then ``H *= mask`` cannot mutate the
        # cached entry.  Re-centre H (fftshift) so the returned transfer
        # function keeps the historical CENTERED-spectrum contract callers use.
        H_returned = xp.fft.fftshift(H)
        H_returned = (H_returned.copy() if hasattr(H_returned, 'copy')
                      else xp.asarray(H_returned))
        return E_out, H_returned
    else:
        return E_out


def apply_fresnel_curvature(
    E: np.ndarray,
    dx: float,
    wavelength: float,
    R: float,
    sign: int = +1,
    dy: Optional[float] = None,
) -> np.ndarray:
    """Apply (or remove) a Fresnel quadratic phase ``exp(i*sign*k*r^2/(2R))``.

    Used to convert between phase conventions when comparing fields
    produced by different libraries.

    Background
    ----------
    Lumenairy's propagators (and the standard Fresnel/ASM family --
    LightPipes, prysm, diffractio, POPPy, Zemax POP) keep the **full
    physical phase** at the output plane.  Some ray-trace-rooted
    aberration-analysis tools (notably OPDPy and Zemax wavefront
    operands like ``OPDX``) instead store the **chief-relative OPD**,
    which implicitly subtracts the natural Gaussian-beam wavefront
    curvature at the image plane.

    The two conventions differ by exactly a Fresnel quadratic phase
    ``exp(i*k*r^2/(2*R))`` with ``R = v - f`` for a thin-lens
    imager (image distance minus focal length).

    Use this function to round-trip between conventions:

    .. code-block:: python

        # Convert OPDPy / Zemax-OPD output to Lumenairy / LightPipes:
        E_absolute = apply_fresnel_curvature(
            E_chief_relative, dx, wavelength, R=v - f, sign=+1)

        # Convert Lumenairy / LightPipes output to chief-relative:
        E_chief_relative = apply_fresnel_curvature(
            E_absolute, dx, wavelength, R=v - f, sign=-1)

    For multi-element systems, ``R`` is the wavefront radius of
    curvature at the image plane predicted by Gaussian-beam ABCD
    propagation -- see Saleh & Teich, *Fundamentals of Photonics*,
    Section 3.1.

    Parameters
    ----------
    E : ndarray, complex 2D
        Input field.  Grid is assumed to be centred on the chief image
        point (the centre pixel is at coordinate ``(0, 0)``, with the
        same half-pixel offset convention as
        :func:`angular_spectrum_propagate`).
    dx : float
        Pixel pitch in the x-direction (metres).
    wavelength : float
        Wavelength (metres).
    R : float
        Wavefront radius of curvature (metres).  For a thin-lens
        imager, ``R = image_distance - focal_length``.
    sign : int, default ``+1``
        ``+1`` adds the curvature (chief-relative -> absolute).
        ``-1`` removes the curvature (absolute -> chief-relative).
    dy : float, optional
        Pixel pitch in y.  Defaults to ``dx``.

    Returns
    -------
    E_out : ndarray, complex
        Same shape and dtype as ``E``, with the Fresnel curvature
        multiplied (or divided) in.

    See also
    --------
    Wiki: "Phase conventions and inter-library comparison"
    """
    if dy is None:
        dy = dx
    # R = 0 / inf / NaN is treated as a no-op so multi-element
    # prescriptions where v-f is ill-defined (e.g. an afocal section)
    # can pass through without curvature.  This is documented behaviour
    # locked in by the test suite.
    if R == 0 or not np.isfinite(R):
        return E.copy()
    if sign not in (+1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign}")
    Ny, Nx = E.shape
    # 4.10: drop the spurious +0.5 half-pixel offset.  Every other
    # propagator in this file builds coordinates as (arange(N) - N/2)*dx
    # (no +0.5).  The mismatch produced a half-pixel walk-off in the
    # curvature centre relative to the propagated field grid, visible
    # as a small coma-like residual in OPDPy cross-checks.
    ax_x = (np.arange(Nx) - Nx / 2) * dx
    ax_y = (np.arange(Ny) - Ny / 2) * dy
    Y, X = np.meshgrid(ax_y, ax_x, indexing='ij')
    r2 = X * X + Y * Y
    k = 2.0 * np.pi / wavelength
    # v5.17.x (audit P3-51): honour dtype-follows-input.  The carrier
    # argument ``k*r2/(2R)`` is accumulated at float64 (r2 is built from
    # f64 grids above) and only the FINISHED phase factor is cast to
    # E's complex dtype before the multiply -- the P2-29 f64-carrier-
    # then-cast recipe.  Pre-fix a complex64 E was silently promoted to
    # complex128 whenever R != 0 (while the R=0 early-return above kept
    # complex64), contradicting the docstring's "same shape and dtype".
    # complex128 inputs are byte-identical (astype(copy=False) no-op).
    if np.iscomplexobj(E):
        target_cdtype = E.dtype
    else:
        target_cdtype = np.dtype(np.complex128)
    phase = np.exp(sign * 1j * k * r2 / (2.0 * R)).astype(
        target_cdtype, copy=False)
    return E * phase


def angular_spectrum_propagate_batch(
    E_stack: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    bandlimit: bool = True,
    use_gpu: bool = False,
) -> np.ndarray:
    """ASM propagation of a stack of fields ``(B, Ny, Nx)`` in one
    fused FFT pair (3.2.14).

    All ``B`` fields share the same grid + wavelength + propagation
    distance, so the transfer function ``H`` is built once (reusing
    the H cache) and broadcast across the batch.  Two batched FFTs
    (forward + inverse, axes ``(-2, -1)``) replace ``2*B`` separate
    2-D FFTs.

    .. note::
       **Honest performance (v5.17.x, P2-27).**  Measured warm
       (H-cache + pyFFTW plans hot, complex128, B=2, N=1024/2048):
       batch is at PARITY with two scalar calls (median ratio
       0.98-1.02x), with a single fused (B, Ny, Nx) plan and fewer
       Python-level dispatches.  The pre-v5.17.x docstring claimed
       '30-60% faster'; in reality that version was 2x+ SLOWER than
       per-component scalar calls because it fetched ``H`` by running
       the full scalar propagator on a garbage proxy field (a wasted
       full-grid FFT+IFFT pair per call, since removed).  Prefer the
       batch form for convenience / GPU stacks, not for a wall-clock
       win at small B.

    Parameters
    ----------
    E_stack : ndarray, complex, shape (B, Ny, Nx)
        Input field stack.  ``B`` must be at least 1.
    z, wavelength, dx, dy, bandlimit, use_gpu
        Same semantics as :func:`angular_spectrum_propagate`.

    Returns
    -------
    E_out : ndarray, complex, shape (B, Ny, Nx)
        Propagated stack, same dtype + array library as input.
    """
    if E_stack.ndim != 3:
        raise ValueError(
            f"angular_spectrum_propagate_batch: input must be 3-D "
            f"(B, Ny, Nx), got shape {E_stack.shape}.")
    # Validate using a representative 2-D slice; the batched call has
    # the same (z, wavelength, dx, dy) constraints as the scalar
    # propagator, so reuse the helper.
    _validate_propagator_inputs(E_stack[0], z, wavelength, dx, dy,
                                fn_name='angular_spectrum_propagate_batch')

    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_stack)):
        xp = _state.cp
        if not _is_cupy_array(E_stack):
            E_stack = _state.cp.asarray(E_stack)
    else:
        xp = np
        if _is_cupy_array(E_stack):
            E_stack = E_stack.get()

    B, Ny, Nx = E_stack.shape
    if dy is None:
        dy = dx

    if xp.iscomplexobj(E_stack):
        target_cdtype = E_stack.dtype
    else:
        target_cdtype = np.dtype(_state.DEFAULT_COMPLEX_DTYPE)
        E_stack = E_stack.astype(target_cdtype)

    # v5.17.x (P2-27): fetch H directly through the shared cache/build
    # helper.  Pre-fix this delegated to the FULL scalar propagator on
    # an uninitialised ``xp.empty`` proxy field with
    # ``return_transfer_function=True``, paying a wasted full-grid
    # FFT+IFFT pair (plus an fftshift+copy of H) on garbage data on
    # EVERY batch call -- even on H-cache hits -- which made the batch
    # entry point measurably SLOWER than two scalar calls.
    H = _get_asm_H_natural(Ny, Nx, dy, dx, wavelength, z, bandlimit,
                           target_cdtype, xp, is_jax=False)

    # Single batched FFT pair across the last two axes.  pyFFTW's
    # multi-slot plan cache (also new in 3.2.14) keys on the full
    # shape including the batch dimension, so a 3-D plan is built on
    # the first call and reused thereafter.  The numpy / scipy
    # fallback paths handle 3-D input natively via ``fft2`` over the
    # last two axes.
    #
    # v5.17.x (P2-27): H is NATURAL-layout, so the two spectrum-domain
    # shifts fold away exactly as in the v5.5.3 scalar path (4 -> 2
    # shifts per batch call; algebraically exact for any N).
    if xp is np:
        # Use scipy.fft for ND batched (workers parameter), pyFFTW
        # plan cache picks up the (B, Ny, Nx) shape automatically via
        # ``_fft2_nd`` if the array is large enough.
        E_out = xp.fft.fftshift(
            _ifft2_nd(_fft2_nd(xp.fft.ifftshift(E_stack, axes=(-2, -1)))
                      * H[None, :, :]),
            axes=(-2, -1))
    else:
        E_out = xp.fft.fftshift(
            xp.fft.ifft2(
                xp.fft.fft2(xp.fft.ifftshift(E_stack, axes=(-2, -1)),
                            axes=(-2, -1)) * H[None, :, :],
                axes=(-2, -1)),
            axes=(-2, -1))
    return E_out


def angular_spectrum_propagate_tilted(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
    bandlimit: bool = True,
    *,
    tilt_x_deg: Optional[float] = None,
    tilt_y_deg: Optional[float] = None,
) -> np.ndarray:
    """
    ASM propagation with a carrier tilt (off-axis propagation).

    Propagates the field while accounting for a mean propagation direction
    that is tilted relative to the optical axis.  This is useful for:

    - Beams arriving at an angle
    - Propagation after a prism or wedge
    - Off-axis portions of a wide-field system

    The tilt is handled by shifting the frequency-domain transfer function,
    which is equivalent to propagating the field in a tilted reference frame.

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
        Input electric field.

    z : float
        Axial separation [m] between the parallel input and output planes.
        (This is NOT an arc length along the tilted beam axis: the
        shifted-``kz`` construction transforms plane-to-plane at axial
        distance ``z``, exactly like plain ASM.)

    wavelength : float
        Optical wavelength [m].

    dx : float
        Grid spacing in x [m].

    dy : float, optional
        Grid spacing in y [m].  Defaults to dx.

    tilt_x, tilt_y : float, default 0.0
        Tilt angles [radians] of the propagation direction relative to the
        z-axis.  The beam propagates at angle (tilt_x, tilt_y) from the
        optical axis.

    bandlimit : bool, default True
        Apply band-limiting to avoid aliasing.

    Returns
    -------
    E_out : ndarray (complex, Ny x Nx)
        Propagated electric field.

    Notes
    -----
    The method removes the carrier frequency (tilt) before propagation,
    then restores it afterwards.  This avoids *spectral* aliasing of the
    carrier (the tilted plane-wave spectrum staying on-grid), NOT physical
    envelope walk-off: the linear term of ``kz(f + f0)`` is retained (as
    it must be), so the envelope still translates by ``z * tan(theta)``
    across the output grid.  Size the grid to contain that shift.

    The carrier spatial frequencies are::

        fx0 = sin(tilt_x) / wavelength
        fy0 = sin(tilt_y) / wavelength

    The field is demodulated as::

        E_demod = E_in * exp(-i * 2*pi * (fx0*X + fy0*Y))

    propagated with a shifted transfer function, then remodulated::

        E_out = E_prop * exp(+i * 2*pi * (fx0*X + fy0*Y))

    For ``tilt_x = tilt_y = 0`` this reduces to standard ASM propagation.

    4.7+: convenience kwargs ``tilt_x_deg`` / ``tilt_y_deg`` accept the
    angle in degrees and take precedence over the radian forms when
    supplied.  These are part of the broader push toward ``_deg`` as
    the canonical user-facing angle unit (see the polish-pass note in
    :ref:`Release Notes`).
    """
    # v4.15.3 (P0-NEW-F2-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper -- siblings missed by the
    # v4.15.2 closure now share the same first-line guard.
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'angular_spectrum_propagate_tilted',
                           input_kind='field')

    _validate_propagator_inputs(E_in, z, wavelength, dx, dy,
                                fn_name='angular_spectrum_propagate_tilted')
    if tilt_x_deg is not None:
        tilt_x = float(np.radians(tilt_x_deg))
    if tilt_y_deg is not None:
        tilt_y = float(np.radians(tilt_y_deg))
    if dy is None:
        dy = dx

    Ny, Nx = E_in.shape

    # -- carrier spatial frequencies from tilt angles ------------------------
    fx0 = np.sin(tilt_x) / wavelength
    fy0 = np.sin(tilt_y) / wavelength

    # Shortcut: no tilt -> fall back to standard ASM
    if abs(fx0) < 1e-15 and abs(fy0) < 1e-15:
        return angular_spectrum_propagate(E_in, z, wavelength, dx, dy,
                                          bandlimit=bandlimit)

    # Target complex dtype (matches angular_spectrum_propagate / RS).
    if np.iscomplexobj(E_in):
        target_cdtype = E_in.dtype
    else:
        target_cdtype = np.dtype(_state.DEFAULT_COMPLEX_DTYPE)

    # -- spatial coordinate grids (carrier; per-call) ------------------------
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)

    # -- demodulate: remove carrier tilt -------------------------------------
    # v5.17.1: build the carrier AT the target dtype.  Pre-fix it was
    # unconditionally complex128 (``np.exp`` of a float64 phase), which
    # silently upcast the ENTIRE tilted pipeline (demod field, FFTs,
    # remodulated output) for complex64 inputs -- doubling the working
    # memory AND returning complex128, violating the dtype-follows-input
    # contract every other propagator honours.  For complex64 the carrier
    # phase is folded mod 2*pi in float64 BEFORE the float32 cast (the same
    # accuracy mitigation as the main ASM kernel), so the large carrier
    # argument (~1e5 rad across a big tilted grid) doesn't hit the float32
    # precision floor.  The complex128 path is bit-identical to pre-fix.
    _carrier_phase = (-2.0 * np.pi) * (fx0 * X + fy0 * Y)
    del X, Y
    if np.dtype(target_cdtype) == np.complex64:
        _ph = np.mod(_carrier_phase, 2.0 * np.pi)
        carrier = np.empty(_ph.shape, dtype=np.complex64)
        carrier.real[:] = np.cos(_ph).astype(np.float32)
        carrier.imag[:] = np.sin(_ph).astype(np.float32)
        del _ph
    else:
        carrier = np.exp(1j * _carrier_phase)
    del _carrier_phase
    E_demod = E_in.astype(target_cdtype, copy=False) * carrier

    # H cache (NumPy backend)
    # The shifted transfer function depends on (Ny, Nx, dy, dx,
    # wavelength, z, fx0, fy0, bandlimit, dtype).  fx0/fy0 are the
    # tilt-derived carrier frequencies and they fully encode the
    # propagation-direction shift, so the cache key handles arbitrary
    # tilt angles without needing tilt_x / tilt_y in the key directly.
    # 'ASM_TILTED' tag keeps these entries disjoint from plain-ASM ones.
    h_key = (int(Ny), int(Nx), float(dy), float(dx),
             float(wavelength), float(z),
             float(fx0), float(fy0),
             bool(bandlimit),
             np.dtype(target_cdtype).str, 'ASM_TILTED')
    H = _h_cache_lookup(h_key)

    if H is None:
        # -- shifted transfer function -------------------------------------
        # kz is evaluated at (fx + fx0, fy + fy0) so the baseband field
        # propagates with the correct kz for each plane-wave component.
        k = 2 * np.pi / wavelength
        dfx = 1.0 / (Nx * dx)
        dfy = 1.0 / (Ny * dy)
        # audit P1: integer DC anchor ``N // 2`` -- H is built centred and
        # ``ifftshift``-ed below (line ~1068), so the centred bin index
        # must use the fftshift anchor.  Bit-identical for even N.  The
        # SPATIAL carrier grid above keeps the ``- N/2`` convention: it
        # cancels exactly between demodulation and remodulation (a change
        # of spatial origin is a constant phase there), so only the
        # frequency lattice matters for the tilted kernel.
        fx = (np.arange(Nx) - Nx // 2) * dfx
        fy = (np.arange(Ny) - Ny // 2) * dfy
        FX, FY = np.meshgrid(fx, fy)

        FX_shifted = FX + fx0
        FY_shifted = FY + fy0
        kx = 2 * np.pi * FX_shifted
        ky = 2 * np.pi * FY_shifted

        kz_sq = k**2 - kx**2 - ky**2
        prop = kz_sq > 0
        kz = np.where(prop, np.sqrt(np.maximum(kz_sq, 0)), 0)
        # S2-10: shared kernel.  complex128 is byte-identical to the former
        # ``np.where(kz_sq > 0, np.exp(1j*kz*z), 0)``; complex64 now folds
        # the phase mod 2*pi in float64 before the float32 cast (S2-3
        # mitigation) instead of casting the complex128 exp straight down.
        H = _asm_H_from_kz(kz, prop, z, target_cdtype, np)

        # -- band-limiting on the ORIGINAL-FRAME spectrum ----------------
        # Matsushima bounds the FREQUENCY OF THE CHIRP in the angular-
        # spectrum kernel, which depends on the original (non-shifted)
        # frequency (FX + fx0, FY + fy0): that's where the chirp's
        # phase-derivative is taken.  The H built above is also
        # evaluated at the shifted arguments, so the mask must use
        # FX_shifted = FX + fx0 (and FY + fy0) -- otherwise it clips the
        # *baseband* (around FX=0) and lets through the actual aliasing-
        # prone high-(FX+fx0) bands.  Pre-4.10 used `|FX| < fx_max`,
        # which for any non-trivial tilt killed the baseband DC and
        # zeroed the propagated field.
        if bandlimit and z != 0:
            Lx = Nx * dx
            Ly = Ny * dy
            fx_max = Lx / (2 * wavelength * abs(z))
            fy_max = Ly / (2 * wavelength * abs(z))
            H = np.where((np.abs(FX_shifted) < fx_max) &
                          (np.abs(FY_shifted) < fy_max), H, 0)

        if H.dtype != target_cdtype:
            H = H.astype(target_cdtype)

        # v5.17.x (P2-28): store the tilted H in NATURAL (un-shifted) FFT
        # layout, mirroring the v5.5.3 plain-ASM fold, so the per-call
        # propagation drops the two spectrum-domain shifts (4 -> 2).
        # ``ifftshift(fftshift(X)) == X`` and shifts are permutations that
        # distribute over elementwise products, so the fold is
        # algebraically EXACT for any N (even or odd) -- verified
        # bit-identical at complex64, complex128 and odd N.
        H = np.fft.ifftshift(H)
        _h_cache_store(h_key, H)

    # -- propagate baseband with shifted transfer function -------------------
    # H is NATURAL-layout (see above): 2-shift fold of the pre-v5.17.x
    # 4-shift centered-H idiom.
    E_prop = np.fft.fftshift(_ifft2(_fft2(np.fft.ifftshift(E_demod)) * H))
    del E_demod

    # -- remodulate: restore carrier tilt ------------------------------------
    E_out = E_prop * np.conj(carrier)

    return E_out
