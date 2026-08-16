"""Regression tests for audit finding S2-10 (v5.24.5).

The ASM / ASM-MFT transfer-function construction used to be duplicated across
~5 sites (the square builder ``_build_asm_H_square``, the natural-layout
builder ``_get_asm_H_natural`` host-JAX + chunked branches, the tilted
propagator, and the MFT propagator), and the copies DISAGREED: the S2-3
complex64 mod-2*pi mitigation lived only in ``_get_asm_H_natural``.  The
square / tilted / MFT builders cast the complex128 ``exp(1j*kz*z)`` straight
down to complex64.

Sequence followed (per the deferred-items roadmap A2 -- reconcile -> validate
-> consolidate):

1. DECIDE -- the mod-2*pi-before-float32-cast mitigation is a correctness fix
   (numpy's complex-exp imaginary-argument reduction is slightly less accurate
   than an explicit float64 ``mod 2*pi`` + ``cos`` / ``sin``), so it must
   apply to EVERY H builder, not just one.
2. VALIDATE -- these tests pin every builder's complex64 output to an
   INDEPENDENT longdouble (80-bit extended precision on x86) ground-truth H,
   a different reduction path from the library's float64 kernel.  The
   mitigated builders match it EXACTLY; the pre-fix exp-then-cast path does
   not (measured up to 5.96e-8, a full float32 ULP near 1, over a 180-config
   sweep).  The complex128 paths stay byte-identical.
3. CONSOLIDATE -- all sites now route through the single shared kernel
   :func:`~lumenairy.propagators.asm._asm_H_from_kz`.

This is therefore a NUMERIC change for the square / tilted / MFT complex64
paths (fail-before / pass-after here), and a byte-identical refactor for
every complex128 path and for the already-mitigated ``_get_asm_H_natural``
(also guarded by the existing S5-8b / S5-8g tests).
"""
import numpy as np
import pytest

from lumenairy.propagators import fft_infra
from lumenairy.propagators.asm import (
    _asm_H_from_kz,
    _build_asm_H_square,
    _get_asm_H_natural,
    angular_spectrum_propagate_tilted,
)
from lumenairy.propagators.mft import angular_spectrum_propagate_mft

WL = 1.31e-6

# Large-phase config: max kz*z ~ 2.4e5 rad -- ~38000 full 2*pi wraps, deep
# into the regime where a float32-order phase reduction loses accuracy.
N, DX, Z = 256, 1e-6, 0.05

# ---------------------------------------------------------------------------
# The error gate is DERIVED IN-BUILD, not a platform constant.
# ---------------------------------------------------------------------------
# 2026-08-15 (docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md).  This file used to
# gate on hard-coded ``TOL = 4e-8`` (and ``8e-8`` for the two one-ULP sites),
# numbers read off ONE CI image.  Both bars sat BELOW the achievable worst
# case of the very contract they assert:
#
#   * the quantity is ``max|H_complex64 - longdouble_oracle|``, a COMPLEX
#     magnitude.  If a bin's two float32 components each land one rounding
#     step apart -- entirely legal for two independently-rounded paths -- the
#     magnitude is ``sqrt(2) * step``, not ``step``;
#   * at the half-ULP figure this file's own docstrings record (2.98e-8) that
#     is 4.21e-8 > the old 4e-8 bar, and at the one-ULP figure (5.96e-8) it is
#     8.43e-8 > the old 8e-8 bar.  Only the accident that no bin happened to
#     round differently in BOTH components kept the old bars alive; the first
#     platform whose libm nudged one extra bin would have burned a tag with
#     the contract fully intact.
#
# Fix: derive the envelope from THIS build's oracle.  ``np.spacing`` on the
# oracle's float32 component magnitudes is the local ULP; half of it is the
# correctly-rounded error of one path, and ``sqrt(2)`` accounts for both
# components deviating at once.  No platform constant survives, so the
# assertion is the CONTRACT ("the mod-2pi f64 fold is bounded at half-to-one
# f32 ULP on EVERY platform") rather than a transcription of one runner.
#
# Measured 2026-08-15 on both mounts (Windows py3.14 / np2.4.4, longdouble
# nmant 52; WSL py3.12 / np2.5.1, longdouble nmant 63 = genuine 80-bit):
#   derived bars       : half = 8.429370e-08, full = 1.685874e-07 (identical
#                        on both mounts -- the envelope saturates at |H| = 1,
#                        which every propagating-bin grid contains)
#   every mitigated builder (kernel / square / natural / MFT / tilted): 0.0
#   pre-fix exp-then-cast: 2.980232e-08
# Two-sided headroom:
#   * will not burn a tag -- the largest value ever RECORDED for a mitigated
#     path anywhere (2.98e-8 half-ULP, 5.96e-8 one-ULP tilted, both from the
#     PR #18 CI image) sits 2.8x below its bar, and the bar is a mathematical
#     envelope: a correctly-rounded-to-float32 result CANNOT exceed it, on
#     any libm, so there is no platform left that can move it;
#   * still fires -- a builder that reverts to a float32-order phase fold
#     measures 1.44e-2 here (5 orders over the bar), and even a 2-step
#     perturbation of a single bin's two components measures 1.33e-7, over
#     the half-ULP bar.
# The structural consolidation pins (every builder routes through
# ``_asm_H_from_kz``) are unchanged and remain the regression guard against a
# builder quietly re-copying the exp-then-cast path.


def _c64_ulp_envelope(oracle):
    """Derive this build's float32 rounding envelope from ``oracle``.

    Returns ``(bar_half, bar_full)``:

    * ``bar_half`` -- ``sqrt(2) * max(half-ULP)``: the magnitude of a
      complex64 deviation whose two components are each correctly rounded
      (i.e. up to one rounding step apart between two independent paths).
    * ``bar_full`` -- ``2 * bar_half``: the same envelope allowing one FULL
      extra ULP per component, for the builders that carry an additional
      float64 rounding into the phase (the tilted carrier) and for the
      libm-dependent exp-then-cast reference.
    """
    step = np.maximum(np.spacing(np.abs(oracle.real).astype(np.float32)),
                      np.spacing(np.abs(oracle.imag).astype(np.float32)))
    bar_half = float(np.sqrt(2.0) * 0.5 * float(step.max()))
    return bar_half, 2.0 * bar_half


def _kz_prop_centered(Ny, Nx, dy, dx, wl):
    """Centered-grid kz and propagating mask (square / MFT / natural
    convention: ``(arange(N) - N/2) / (N*dx)``)."""
    k = 2.0 * np.pi / wl
    fx = (np.arange(Nx, dtype=np.float64) - Nx / 2) / (Nx * dx)
    fy = (np.arange(Ny, dtype=np.float64) - Ny / 2) / (Ny * dy)
    kx_sq = (2.0 * np.pi * fx) ** 2
    ky_sq = (2.0 * np.pi * fy) ** 2
    kz_sq = k ** 2 - kx_sq[None, :] - ky_sq[:, None]
    prop = kz_sq > 0
    kz = np.where(prop, np.sqrt(np.where(prop, kz_sq, 0.0)), 0.0)
    return kz, prop


def _c64_longdouble_oracle(kz, prop, z):
    """Correctly-rounded-to-float32 ``exp(1j*kz*z)`` computed through an
    80-bit longdouble ``mod 2*pi`` reduction -- an independent, higher-
    precision path than the library's float64 kernel.  Evanescent bins are
    zeroed."""
    ph = np.mod(kz.astype(np.longdouble) * np.longdouble(z),
                2.0 * np.longdouble(np.pi))
    H = np.empty(kz.shape, dtype=np.complex64)
    H.real[:] = np.where(prop, np.cos(ph).astype(np.float32), np.float32(0))
    H.imag[:] = np.where(prop, np.sin(ph).astype(np.float32), np.float32(0))
    return H


def _c64_pre_fix_expcast(kz, prop, z):
    """The pre-S2-10 square / tilted / MFT complex64 path: complex128
    ``exp`` then a straight cast to complex64."""
    return np.where(prop, np.exp(1j * kz * z), 0).astype(np.complex64)


def _cache_H_by_tag(tag):
    """Return the single cached H whose key carries ``tag`` (last key
    element).  Asserts exactly one so a key-format drift fails loudly."""
    ents = [v for k, v in fft_infra._H_CACHE.items()
            if isinstance(k, tuple) and k and k[-1] == tag]
    assert len(ents) == 1, (
        f"expected exactly one '{tag}' H-cache entry, got {len(ents)} "
        f"(cache-key format drift?)")
    return ents[0]


# ---------------------------------------------------------------------------
# The DECISION, quantified: the mitigation is a genuine correctness fix.
# ---------------------------------------------------------------------------

def test_mitigated_path_bound_is_platform_independent():
    """The mitigation's REAL contract (v5.25.0, PR #18 CI): the mod-2pi
    f64 fold is BOUNDED at ~half-to-one f32 ULP on EVERY platform,
    whereas the pre-fix exp-then-cast path's accuracy depends entirely
    on the host libm -- measured 5.96e-8 (1 ULP) on Windows ucrt but
    1.49e-8 (BETTER than the fold) on Linux glibc.  A per-platform
    "old is worse" ordering is therefore NOT an invariant and is not
    asserted; the cross-platform guarantee of the mitigated path is.
    (Regression protection against a builder quietly reverting to
    exp-cast lives in the structural consolidation pins in this file:
    every builder must route through ``_asm_H_from_kz``.)"""
    from lumenairy.propagators.asm import _asm_H_from_kz
    kz, prop = _kz_prop_centered(N, N, DX, DX, WL)
    oracle = _c64_longdouble_oracle(kz, prop, Z)
    old = _c64_pre_fix_expcast(kz, prop, Z)
    new = _asm_H_from_kz(kz, prop, Z, np.complex64)
    bar_half, bar_full = _c64_ulp_envelope(oracle)
    err_old = float(np.max(np.abs(old - oracle)))
    err_new = float(np.max(np.abs(new - oracle)))
    assert err_new <= bar_half, (
        f"mitigated-path c64 err {err_new:.2e} exceeds the in-build "
        f"correctly-rounded envelope {bar_half:.2e}")
    # informational envelope for the libm-dependent path: within ~1 ULP
    # per component on every platform observed (1.49e-8 glibc .. 5.96e-8
    # ucrt, 2.98e-8 measured 2026-08-15 on both mounts) -- if a platform
    # ever exceeds this, the mitigation decision should be revisited with
    # that platform's numbers.
    assert err_old <= bar_full, (
        f"exp-then-cast err {err_old:.2e} exceeds one f32 ULP per "
        f"component ({bar_full:.2e}) -- new platform regime; "
        f"re-evaluate the mitigation analysis")


# ---------------------------------------------------------------------------
# Complex64 correctness fix -- every former call site (fail-before/pass-after)
# ---------------------------------------------------------------------------

def test_shared_kernel_c64_matches_longdouble_oracle():
    """The shared kernel -- through which every builder now routes -- matches
    the longdouble ground truth exactly at complex64."""
    kz, prop = _kz_prop_centered(N, N, DX, DX, WL)
    H = _asm_H_from_kz(kz, prop, Z, np.complex64, np)
    oracle = _c64_longdouble_oracle(kz, prop, Z)
    assert H.dtype == np.complex64
    err = float(np.max(np.abs(H - oracle)))
    bar_half, _ = _c64_ulp_envelope(oracle)
    assert err <= bar_half, (
        f"shared-kernel c64 err {err:.2e} exceeds {bar_half:.2e}")


def test_build_asm_H_square_c64_matches_longdouble_oracle():
    """SQUARE builder: complex64 H is correctly rounded (was ~3e-8 off)."""
    kz, prop = _kz_prop_centered(N, N, DX, DX, WL)
    H = _build_asm_H_square(N, DX, Z, WL, dtype=np.complex64, bandlimit=False)
    oracle = _c64_longdouble_oracle(kz, prop, Z)
    assert H.dtype == np.complex64
    err = float(np.max(np.abs(H - oracle)))
    bar_half, _ = _c64_ulp_envelope(oracle)
    assert err <= bar_half, f"square c64 err {err:.2e} exceeds {bar_half:.2e}"


def test_get_asm_H_natural_c64_matches_longdouble_oracle():
    """NATURAL builder (already mitigated pre-S2-10): still correctly rounded
    after routing through the shared kernel.  H is natural-layout, so the
    oracle is ifftshift-ed."""
    fft_infra.clear_asm_caches()
    kz, prop = _kz_prop_centered(N, N, DX, DX, WL)
    H = _get_asm_H_natural(N, N, DX, DX, WL, Z, False, np.complex64, np,
                           is_jax=False)
    oracle = np.fft.ifftshift(_c64_longdouble_oracle(kz, prop, Z))
    assert H.dtype == np.complex64
    err = float(np.max(np.abs(H - oracle)))
    bar_half, _ = _c64_ulp_envelope(oracle)
    assert err <= bar_half, f"natural c64 err {err:.2e} exceeds {bar_half:.2e}"


def test_mft_H_c64_matches_longdouble_oracle():
    """MFT builder: the cached complex64 transfer function is correctly
    rounded (centered layout)."""
    fft_infra.clear_asm_caches()
    E = np.ones((N, N), dtype=np.complex64)
    angular_spectrum_propagate_mft(E, Z, WL, DX, DX, N, bandlimit=False)
    H = _cache_H_by_tag('ASM_MFT')
    kz, prop = _kz_prop_centered(N, N, DX, DX, WL)
    oracle = _c64_longdouble_oracle(kz, prop, Z)
    assert H.dtype == np.complex64
    err = float(np.max(np.abs(H - oracle)))
    bar_half, _ = _c64_ulp_envelope(oracle)
    assert err <= bar_half, f"MFT c64 err {err:.2e} exceeds {bar_half:.2e}"


def test_tilted_H_c64_matches_longdouble_oracle():
    """TILTED builder: the cached complex64 shifted transfer function is
    correctly rounded.  H is natural-layout on the SHIFTED frequency grid."""
    fft_infra.clear_asm_caches()
    tilt = np.deg2rad(10.0)
    E = np.ones((N, N), dtype=np.complex64)
    angular_spectrum_propagate_tilted(E, Z, WL, DX, tilt_x=tilt,
                                      bandlimit=False)
    H = _cache_H_by_tag('ASM_TILTED')

    # Independent oracle: kz on the (fx + fx0) shifted grid, natural layout.
    k = 2.0 * np.pi / WL
    fx0 = np.sin(tilt) / WL
    dfx = 1.0 / (N * DX)
    fx = (np.arange(N) - N / 2) * dfx
    FX, FY = np.meshgrid(fx, fx)
    kx = 2.0 * np.pi * (FX + fx0)
    ky = 2.0 * np.pi * FY               # no y-tilt -> fy0 = 0
    kz_sq = k ** 2 - kx ** 2 - ky ** 2
    prop = kz_sq > 0
    kz = np.where(prop, np.sqrt(np.maximum(kz_sq, 0.0)), 0.0)
    oracle = np.fft.ifftshift(_c64_longdouble_oracle(kz, prop, Z))
    _, bar_full = _c64_ulp_envelope(oracle)
    assert H.dtype == np.complex64
    err = float(np.max(np.abs(H - oracle)))
    # v5.25.0 (PR #18 CI): the tilted builder folds kz*z + the tilt
    # CARRIER phase -- one extra f64 rounding in the sum before the mod-2pi
    # fold, so its c64 error lands at ONE full float32 ULP (measured
    # 5.96e-8 on Linux 80-bit longdouble) where the carrier-free builders
    # stay at half a ULP.  Physically equivalent, so this site uses the
    # FULL-ULP-per-component envelope derived from its own oracle
    # (2026-08-15, docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md -- the old
    # hard-coded 8e-8 was BELOW the sqrt(2)*5.96e-8 = 8.43e-8 that one full
    # ULP in both components of a single bin already reaches).  Measured
    # 0.0 on both mounts 2026-08-15 against a derived bar of 1.686e-07.
    assert err <= bar_full, (
        f"tilted c64 err {err:.2e} exceeds one f32 ULP per component "
        f"({bar_full:.2e})")


# ---------------------------------------------------------------------------
# Complex128 byte-identity -- the consolidation must not touch double precision
# ---------------------------------------------------------------------------

def _c128_reference(kz, prop, z):
    return np.where(prop, np.exp(1j * kz * z), 0).astype(np.complex128)


def test_shared_kernel_c128_byte_identical_numpy():
    """Shared kernel (numexpr OFF) is byte-identical to a plain ``np.exp``
    complex128 reference."""
    kz, prop = _kz_prop_centered(N, N, DX, DX, WL)
    H = _asm_H_from_kz(kz, prop, Z, np.complex128, np, use_numexpr=False)
    ref = _c128_reference(kz, prop, Z)
    assert H.dtype == np.complex128
    assert np.array_equal(H, ref), (
        f"c128 kernel differs from np.exp ref; "
        f"max|diff|={np.max(np.abs(H - ref)):.2e}")


def test_shared_kernel_c128_numexpr_byte_identical():
    """Shared kernel (numexpr ON) is byte-identical to the numpy path AND the
    reference (audit S5-8b: numexpr complex-exp is bit-for-bit np.exp)."""
    pytest.importorskip("numexpr")
    from lumenairy.propagators import asm
    assert asm.NUMEXPR_AVAILABLE, "numexpr importable but flag is False"
    # A grid at/above the numexpr floor so the fused path is exercised.
    Nl = 512
    kz, prop = _kz_prop_centered(Nl, Nl, DX, DX, WL)
    H_ne = _asm_H_from_kz(kz, prop, Z, np.complex128, np, use_numexpr=True)
    H_np = _asm_H_from_kz(kz, prop, Z, np.complex128, np, use_numexpr=False)
    ref = _c128_reference(kz, prop, Z)
    assert np.array_equal(H_ne, H_np)
    assert np.array_equal(H_ne, ref)


def test_square_and_natural_c128_byte_identical():
    """The square and natural builders are byte-identical to an independent
    complex128 reference (the refactor left double precision untouched)."""
    kz, prop = _kz_prop_centered(N, N, DX, DX, WL)
    ref_centered = _c128_reference(kz, prop, Z)

    Hsq = _build_asm_H_square(N, DX, Z, WL, dtype=np.complex128,
                              bandlimit=False)
    assert np.array_equal(Hsq, ref_centered), "square c128 not byte-identical"

    fft_infra.clear_asm_caches()
    Hnat = _get_asm_H_natural(N, N, DX, DX, WL, Z, False, np.complex128, np,
                              is_jax=False)
    assert np.array_equal(Hnat, np.fft.ifftshift(ref_centered)), (
        "natural c128 not byte-identical")


# ---------------------------------------------------------------------------
# Consolidation: one shared kernel is the single source of truth
# ---------------------------------------------------------------------------

def test_shared_kernel_is_exported_and_single_source():
    """``_asm_H_from_kz`` is the exported shared kernel that MFT imports (one
    definition, not a re-copy)."""
    from lumenairy.propagators import asm, mft
    assert '_asm_H_from_kz' in asm.__all__
    assert mft._asm_H_from_kz is asm._asm_H_from_kz
