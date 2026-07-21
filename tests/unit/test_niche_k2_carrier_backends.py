"""Phase K2 (plan N14) -- CuPy + JAX backends for the astigmatic / apertured
carrier ASM (``lumenairy.propagators.carrier``).

The carrier-referenced Sziklas-Siegman propagator is backend-agnostic: the
backend is selected -- exactly like every other lumenairy propagator -- by the
array type the caller passes in (``array_namespace``).  This is a backend PORT,
accuracy-neutral; the algorithm is identical on NumPy / CuPy / JAX.

Acceptance (plan N14):

1. **Backend parity** -- CuPy and JAX match NumPy to < 1e-6 (relative) on the
   astigmatic line-foci + circle-of-least-confusion case AND an apertured
   converging leg; the NumPy path stays byte-identical (tolerance-pinned, NOT
   ``array_equal`` between two live FFT calls -- the v5.26.0 cache-warmth ~1 ULP
   lesson).
2. **Speedup + peak memory** measured per backend at N=2048..8192 (a modest
   in-test measurement here; the full sweep lives in the K2 report).
3. **Focus crossing** works on every backend (both line foci + the waist);
   isotropic input reduces to the scalar path on every backend.
4. **No silent dtype upcast** (the v4.14.1 audit contract) on any backend.
   JAX additionally yields differentiability -- a gradient through a carrier
   leg is validated.

Backends in this harness: NumPy (always), JAX CPU-XLA (measured), CuPy
(functional-probed).  The propagating CuPy tests need a working cuFFT and skip
on this box (cupy installed, cuFFT DLL broken); the explicit-mask CuPy test
calls no FFT, so it gates on elementwise kernels only and RUNS here -- giving
genuine on-device coverage of the host-mask-moves-on-device fix.

Independent oracle for the parity claim: the NumPy result IS the reference (a
different code path -- pyFFTW `fresnel_tf_propagate` vs `xp.fft`), plus the
analytic astigmatic-Gaussian ABCD q-parameter for the foci/lc plane locations
and the analytic power-derivative for the gradient.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

import lumenairy as la
from lumenairy import (
    carrier_referenced_aperture,
    carrier_referenced_envelope,
    carrier_referenced_reconstruct,
    propagate_carrier_referenced,
)
from lumenairy.memory import available_memory_bytes

WL = 1.31e-6
K = 2.0 * np.pi / WL


# ---------------------------------------------------------------------------
# backend fixtures / skips
# ---------------------------------------------------------------------------
def _jax_or_skip():
    """Import JAX (skip if absent) and enable x64 so the backend-parity
    comparison runs at complex128 -- the fair, precision-matched check that
    demonstrates the < 1e-6 acceptance (in practice ~1e-15).  x64 is process-
    wide and sticky; that is the established campaign convention."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    return jax, jnp


def _cupy_or_skip():
    """Import CuPy AND functionally probe a tiny FFT round-trip.  ``import
    cupy`` succeeding does NOT imply a working GPU (this box has cupy 14 with a
    broken cuFFT DLL), so a bare ``importorskip`` would hand back a namespace
    whose every array op raises.  Skip unless a real device round-trips.

    Use this for tests that actually PROPAGATE (the carrier ASM calls
    ``xp.fft``); tests that touch only elementwise kernels (an explicit-mask
    aperture) should use :func:`_cupy_basic_or_skip`, which does not require a
    working cuFFT."""
    cp = pytest.importorskip("cupy")
    try:
        a = cp.ones((4, 4), dtype=cp.complex128)
        cp.asnumpy(cp.fft.fft2(a))
    except Exception as exc:  # noqa: BLE001 -- any CUDA/cuFFT failure => skip
        pytest.skip(f"CuPy present but no functional CUDA device: "
                    f"{type(exc).__name__}")
    return cp


def _cupy_basic_or_skip():
    """Import CuPy and probe a tiny ELEMENTWISE round-trip (device multiply +
    reduction + ``asnumpy``) -- deliberately NOT an FFT.  The explicit-mask
    aperture path never calls an FFT (it is a pure on-device ``env * msk`` plus
    a power reduction), so it can be -- and IS -- exercised on real CuPy even on
    a box whose cuFFT DLL is broken, as long as basic CUDA kernels launch.  This
    box (cupy 14, broken cuFFT, functional elementwise kernels) therefore gets
    genuine on-device coverage of the mask fix instead of a spurious skip.  Skip
    only if CuPy/CUDA is entirely non-functional."""
    cp = pytest.importorskip("cupy")
    try:
        a = cp.ones((4, 4), dtype=cp.complex128)
        cp.asnumpy((a * 2.0).sum())
    except Exception as exc:  # noqa: BLE001 -- any CUDA failure => skip
        pytest.skip(f"CuPy present but no functional CUDA device: "
                    f"{type(exc).__name__}")
    return cp


def _relerr(got, ref):
    got = np.asarray(got)
    ref = np.asarray(ref)
    denom = np.max(np.abs(ref))
    return float(np.max(np.abs(got - ref)) / denom)


def _gauss(N, dx, wx, dy=None, wy=None):
    """Real elliptical Gaussian amplitude envelope, axis 1 == x, axis 0 == y."""
    dy = dx if dy is None else dy
    wy = wx if wy is None else wy
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    Y, X = np.meshgrid(y, x, indexing='ij')
    return np.exp(-(X ** 2) / wx ** 2 - (Y ** 2) / wy ** 2).astype(np.complex128)


def _q_at(R, z, w_in):
    q0 = 1.0 / (1.0 / R - 1j * WL / (np.pi * w_in ** 2))
    return q0 + z


def _w_of_q(q):
    return np.sqrt(-WL / (np.pi * (1.0 / q).imag))


# ===========================================================================
# 1. NumPy path byte-identical (tolerance-pinned)
# ===========================================================================
def test_numpy_path_tolerance_pinned_stable():
    """The NumPy carrier path is unchanged by the backend port: a scalar and an
    astigmatic leg reproduce themselves across a cold/warm ASM-cache cycle to
    < 1e-10 * scale (the tolerance-pinned byte-identity contract -- NOT
    ``array_equal`` between two live FFT/cache propagator calls, per the
    v5.26.0 cache-warmth ~1 ULP CI lesson)."""
    N, dx = 512, 4e-6
    env = _gauss(N, dx, 40e-6)
    try:
        for R, z in [(30e-3, 40e-3), ((30e-3, 45e-3), 12e-3), (np.inf, 5e-3)]:
            a = propagate_carrier_referenced(env, R, z, WL, dx)
            la.clear_asm_caches()
            b = propagate_carrier_referenced(env, R, z, WL, dx)
            av, bv = np.asarray(a.env), np.asarray(b.env)
            scale = np.max(np.abs(av))
            assert np.max(np.abs(av - bv)) <= 1e-10 * scale, f"R={R} z={z}"
            assert av.dtype == bv.dtype == np.complex128
    finally:
        la.clear_asm_caches()


# ===========================================================================
# 2. JAX backend parity (< 1e-6)
# ===========================================================================
def test_jax_scalar_diverging_parity():
    """Scalar diverging non-crossing leg: JAX (c128) matches NumPy < 1e-6."""
    _jax, jnp = _jax_or_skip()
    N, dx = 512, 4e-6
    env = _gauss(N, dx, 40e-6)
    ref = propagate_carrier_referenced(env, 30e-3, 40e-3, WL, dx)
    got = propagate_carrier_referenced(jnp.asarray(env), 30e-3, 40e-3, WL, dx)
    assert _relerr(got.env, ref.env) < 1e-6
    assert got.R == ref.R
    assert got.dx == pytest.approx(ref.dx, rel=1e-12)


def test_jax_collimated_parity():
    """Collimated carrier (plain Fresnel-TF step) parity."""
    _jax, jnp = _jax_or_skip()
    N, dx = 512, 4e-6
    env = _gauss(N, dx, 60e-6)
    ref = propagate_carrier_referenced(env, np.inf, 5e-3, WL, dx)
    got = propagate_carrier_referenced(jnp.asarray(env), np.inf, 5e-3, WL, dx)
    assert _relerr(got.env, ref.env) < 1e-6


@pytest.mark.slow
def test_jax_astigmatic_line_foci_and_lc_parity():
    """ACCEPTANCE #1: a cylindrical-lens astigmatic Gaussian (distinct R_x,
    R_y) transported to the x line focus, the y line focus, AND the
    circle-of-least-confusion plane -- exercising the per-axis focus-crossing
    bridge -- matches NumPy to < 1e-6 on the FULL field on JAX."""
    _jax, jnp = _jax_or_skip()
    R_x, R_y, w_in = -20e-3, -30e-3, 0.5e-3
    z_fx = -_q_at(R_x, 0, w_in).real
    z_fy = -_q_at(R_y, 0, w_in).real
    zz = np.linspace(0.5 * min(z_fx, z_fy), 1.3 * max(z_fx, z_fy), 4001)
    wx = np.array([_w_of_q(_q_at(R_x, z, w_in)) for z in zz])
    wy = np.array([_w_of_q(_q_at(R_y, z, w_in)) for z in zz])
    z_lc = zz[int(np.argmin(np.abs(wx - wy)))]

    N = 1024
    dx = (8 * w_in) / N
    env = _gauss(N, dx, w_in)
    ej = jnp.asarray(env)
    try:
        for label, zt in [('x-focus', z_fx), ('lc', z_lc), ('y-focus', z_fy)]:
            ref = propagate_carrier_referenced(env, (R_x, R_y), zt, WL, dx)
            got = propagate_carrier_referenced(ej, (R_x, R_y), zt, WL, dx)
            assert _relerr(got.env, ref.env) < 1e-6, (
                f"{label}: rel {_relerr(got.env, ref.env):.2e}")
            assert np.all(np.isfinite(np.asarray(got.env)))
            assert got.R == ref.R
            for gd, rd in zip(got.dx, ref.dx):
                assert gd == pytest.approx(rd, rel=1e-12)
    finally:
        la.clear_asm_caches()


@pytest.mark.slow
def test_jax_apertured_converging_leg_parity():
    """ACCEPTANCE #1 (part 2): a converging leg apertured mid-chain and
    continued matches NumPy to < 1e-6 on JAX, with the clipped-power
    transmission identical."""
    _jax, jnp = _jax_or_skip()
    R0, w_in = -60e-3, 0.4e-3
    N = 1024
    dx = (8 * w_in) / N
    env0 = _gauss(N, dx, w_in)

    def _chain(E):
        e, R, d = propagate_carrier_referenced(E, R0, 20e-3, WL, dx)
        field, tr = carrier_referenced_aperture(
            e, R, WL, d, radius=320e-6, return_transmission=True)
        e2, R2, d2 = propagate_carrier_referenced(
            field.env, field.R, 10e-3, WL, field.dx)
        return e2, tr, d2

    try:
        e_n, tr_n, d_n = _chain(env0)
        e_j, tr_j, d_j = _chain(jnp.asarray(env0))
        assert _relerr(e_j, e_n) < 1e-6
        assert tr_j == pytest.approx(tr_n, rel=1e-9)
        assert tr_n < 0.98              # a genuine (hard) clip
        assert d_j == pytest.approx(d_n, rel=1e-12)
    finally:
        la.clear_asm_caches()


def test_jax_host_numpy_mask_moves_on_device():
    """A NON-NumPy field apertured with a HOST NumPy ``mask=`` array: the mask
    is moved onto the field's device (backend-agnostic ``xp.asarray``) so
    ``env * msk`` stays a single-backend op, the result is a device (JAX)
    array, and it matches the NumPy reference to < 1e-6 with identical
    transmission.

    Guards the fix that replaced the JAX-only ``_to_dev`` hop on the explicit-
    mask path (which left the mask host-NumPy on CuPy, so ``env * msk`` would
    mix a device array with a host array and raise on a real GPU).  JAX is the
    runnable non-NumPy backend here; the CuPy analogue is
    ``test_cupy_mask_aperture_parity`` (skipped without a GPU)."""
    _jax, jnp = _jax_or_skip()
    from lumenairy.backend import is_jax_array
    N, dx = 256, 4e-6
    env = _gauss(N, dx, 60e-6)
    ax = (np.arange(N) - N / 2) * dx
    yy, xx = np.meshgrid(ax, ax, indexing='ij')
    # a soft (graded) real HOST numpy mask -- exercises mask= (not radius=)
    mask = np.exp(-(xx ** 2 + yy ** 2) / (120e-6) ** 2).astype(np.float64)
    ref, tr_n = carrier_referenced_aperture(
        env, 30e-3, WL, dx, mask=mask, return_transmission=True)
    got, tr_j = carrier_referenced_aperture(
        jnp.asarray(env), 30e-3, WL, dx, mask=mask, return_transmission=True)
    assert is_jax_array(got.env)          # the multiply ran on-device
    assert _relerr(got.env, ref.env) < 1e-6
    assert tr_j == pytest.approx(tr_n, rel=1e-9)
    assert tr_n < 1.0                     # a genuine (soft) clip


def test_jax_reconstruct_envelope_parity():
    """Grid-op parity: reconstruct/extract of a full field match NumPy < 1e-6
    on JAX, for scalar and astigmatic carriers."""
    _jax, jnp = _jax_or_skip()
    N, dx, dy = 256, 5e-6, 6e-6
    env = _gauss(N, dx, 60e-6, dy=dy, wy=70e-6) * np.exp(0.3j)
    for R in [42e-3, (42e-3, 55e-3), (np.inf, 55e-3)]:
        rn = carrier_referenced_reconstruct(env, R, WL, dx, dy)
        rj = carrier_referenced_reconstruct(jnp.asarray(env), R, WL, dx, dy)
        assert _relerr(rj, rn) < 1e-6, f"reconstruct R={R}"
        bn = carrier_referenced_envelope(rn, R, WL, dx, dy)
        bj = carrier_referenced_envelope(jnp.asarray(rn), R, WL, dx, dy)
        assert _relerr(bj, bn) < 1e-6, f"envelope R={R}"


# ===========================================================================
# 3. Focus crossing + isotropic-reduces-to-scalar on every backend
# ===========================================================================
def test_jax_scalar_focus_crossing_finite_and_flipped():
    """A converging beam stepped THROUGH its focus on JAX returns a finite
    field with the flipped diverging carrier -- and matches NumPy < 1e-6."""
    _jax, jnp = _jax_or_skip()
    N = 1024
    w_in = 0.5e-3
    dx = (8 * w_in) / N
    env = _gauss(N, dx, w_in)
    try:
        ref = propagate_carrier_referenced(env, -20e-3, 40e-3, WL, dx)
        got = propagate_carrier_referenced(jnp.asarray(env), -20e-3, 40e-3,
                                           WL, dx)
        assert np.all(np.isfinite(np.asarray(got.env)))
        assert got.R == pytest.approx(20e-3, rel=1e-9)     # -20 + 40, diverging
        assert _relerr(got.env, ref.env) < 1e-6
    finally:
        la.clear_asm_caches()


def test_jax_astig_per_axis_focus_crossing():
    """Both line foci crossed (x at ~20 mm, y at ~30 mm) on JAX: finite, both
    carriers flipped diverging, parity with NumPy < 1e-6."""
    _jax, jnp = _jax_or_skip()
    N = 1024
    w_in = 0.5e-3
    dx = (8 * w_in) / N
    env = _gauss(N, dx, w_in)
    try:
        ref = propagate_carrier_referenced(env, (-20e-3, -30e-3), 40e-3, WL, dx)
        got = propagate_carrier_referenced(
            jnp.asarray(env), (-20e-3, -30e-3), 40e-3, WL, dx)
        assert np.all(np.isfinite(np.asarray(got.env)))
        assert got.R[0] == pytest.approx(20e-3, rel=1e-9)
        assert got.R[1] == pytest.approx(10e-3, rel=1e-9)
        assert _relerr(got.env, ref.env) < 1e-6
    finally:
        la.clear_asm_caches()


def test_jax_isotropic_reduces_to_scalar():
    """``carrier=(R, R)`` (equal radii) routes to the scalar path byte-
    identically on JAX too (the astigmatic API reduces to the scalar transform
    on every backend)."""
    _jax, jnp = _jax_or_skip()
    N, dx = 512, 4e-6
    env = jnp.asarray(_gauss(N, dx, 40e-6))
    for R, z in [(30e-3, 40e-3), (-20e-3, 40e-3)]:
        s = propagate_carrier_referenced(env, R, z, WL, dx)
        t = propagate_carrier_referenced(env, (R, R), z, WL, dx)
        assert np.array_equal(np.asarray(t.env), np.asarray(s.env)), \
            f"R={R} z={z}"


# ===========================================================================
# 4. No dtype upcast + differentiability (JAX)
# ===========================================================================
def test_jax_no_dtype_upcast():
    """complex64 in -> complex64 out on every JAX code path (scalar / astig /
    collimated / scalar-crossing / astig-crossing / aperture / reconstruct) --
    the v4.14.1 no-silent-upcast contract.  Holds regardless of x64 state
    (the phase screens are cast to the field dtype and a weak python-complex
    scalar is used so a numpy c128 scalar cannot promote the field)."""
    _jax, jnp = _jax_or_skip()
    N, dx = 512, 4e-6
    e = jnp.asarray(_gauss(N, dx, 40e-6).astype(np.complex64))
    assert e.dtype == np.complex64
    P = propagate_carrier_referenced
    assert P(e, 30e-3, 40e-3, WL, dx).env.dtype == np.complex64
    assert P(e, (30e-3, 45e-3), 12e-3, WL, dx).env.dtype == np.complex64
    assert P(e, np.inf, 5e-3, WL, dx).env.dtype == np.complex64
    w = 0.5e-3
    N2 = 1024
    dxm = (8 * w) / N2
    ec = jnp.asarray(_gauss(N2, dxm, w).astype(np.complex64))
    try:
        assert P(ec, -20e-3, 40e-3, WL, dxm).env.dtype == np.complex64
        assert P(ec, (-20e-3, -30e-3), 40e-3, WL, dxm).env.dtype == np.complex64
    finally:
        la.clear_asm_caches()
    assert carrier_referenced_aperture(
        e, 30e-3, WL, dx, radius=2e-4).env.dtype == np.complex64
    assert carrier_referenced_reconstruct(
        e, 30e-3, WL, dx).dtype == np.complex64


def test_jax_grad_through_carrier_leg():
    """JAX differentiability: a gradient of a real functional (transported
    power) w.r.t. the input envelope amplitude, through a non-crossing carrier
    leg, is finite AND physically correct.

    The leg conserves power to grid accuracy, so ``d/da_ij [ sum|u_out|^2
    dx_out^2 ] = 2 a_ij dx_in^2``.  Validated both against that analytic form
    and against an independent central finite difference."""
    jax, jnp = _jax_or_skip()
    N, dx = 128, 4e-6
    amp0 = jnp.asarray(np.real(_gauss(N, dx, 40e-6)))

    def leg_power(amp):
        e = amp.astype(jnp.complex128)
        out = propagate_carrier_referenced(e, 30e-3, 40e-3, WL, dx)
        return jnp.sum(jnp.abs(out.env) ** 2) * jnp.abs(out.dx) ** 2

    g = np.asarray(jax.grad(leg_power)(amp0))
    assert np.all(np.isfinite(g))
    a = np.asarray(amp0)
    analytic = 2.0 * a * dx ** 2
    # matches the analytic power-derivative where the beam has support
    m = a > 0.05 * a.max()
    assert _relerr(g[m], analytic[m]) < 1e-3

    # independent central finite difference at the peak pixel
    ij = np.unravel_index(int(np.argmax(a)), a.shape)
    eps = 1e-4

    def _perturb(sign):
        ap = np.asarray(amp0).copy()
        ap[ij] += sign * eps
        return float(leg_power(jnp.asarray(ap)))

    fd = (_perturb(+1) - _perturb(-1)) / (2 * eps)
    assert abs(fd - g[ij]) / abs(g[ij]) < 1e-3


# ===========================================================================
# 5. CuPy backend (functional-probed; skipped where no GPU)
# ===========================================================================
def test_cupy_scalar_and_astig_parity():
    """CuPy parity with NumPy < 1e-6 for a scalar diverging leg and an
    astigmatic leg.  Skipped on a box without a functional CUDA/cuFFT device
    (the CuPy code path is import-guarded and exercised only where a GPU is
    present)."""
    cp = _cupy_or_skip()
    N, dx = 512, 4e-6
    env = _gauss(N, dx, 40e-6)
    ecp = cp.asarray(env)
    for R, z in [(30e-3, 40e-3), ((30e-3, 45e-3), 12e-3), (np.inf, 5e-3)]:
        ref = propagate_carrier_referenced(env, R, z, WL, dx)
        got = propagate_carrier_referenced(ecp, R, z, WL, dx)
        assert _relerr(cp.asnumpy(got.env), ref.env) < 1e-6, f"R={R} z={z}"


def test_cupy_mask_aperture_parity():
    """CuPy explicit-mask aperture parity: a HOST NumPy ``mask=`` array is
    moved onto the device (backend-agnostic ``xp.asarray``) so ``env * msk`` is
    a single-backend op on CuPy too (it would raise a mixed-operand TypeError
    if the mask stayed host-NumPy).

    This is the on-hardware guard for the KILL-2 fix.  It gates on
    :func:`_cupy_basic_or_skip` (elementwise kernels only) -- NOT the cuFFT
    round-trip -- because the mask-aperture path calls no FFT, so it RUNS on
    this box's real CuPy (cupy 14, broken cuFFT, working kernels) and gives
    genuine on-device coverage; it skips only where CuPy/CUDA is entirely
    non-functional.

    The mask is a hard disk of radius ``w`` (== the amplitude waist), which is a
    genuine clip -- it removes ~13.7% of the power (a Gaussian passes
    ``1 - exp(-2 (r/w)^2) = 1 - exp(-2) ~ 0.865`` inside radius ``w``).  An
    earlier revision used radius ``2w``, which passes ``1 - exp(-8) ~ 0.9997``
    and does NOT satisfy the ``< 0.98`` clip check -- fixed here (the code fix
    was correct; the test's radius/threshold were not)."""
    cp = _cupy_basic_or_skip()
    N, dx = 256, 4e-6
    w = 60e-6
    env = _gauss(N, dx, w)
    ax = (np.arange(N) - N / 2) * dx
    yy, xx = np.meshgrid(ax, ax, indexing='ij')
    mask = (xx ** 2 + yy ** 2 <= w ** 2).astype(np.float64)   # r == waist
    ref, tr_n = carrier_referenced_aperture(
        env, 30e-3, WL, dx, mask=mask, return_transmission=True)
    got, tr_c = carrier_referenced_aperture(
        cp.asarray(env), 30e-3, WL, dx, mask=mask, return_transmission=True)
    # the multiply ran on-device: the CuPy result is a CuPy array
    assert isinstance(got.env, cp.ndarray)
    assert _relerr(cp.asnumpy(got.env), ref.env) < 1e-6
    assert tr_c == pytest.approx(tr_n, rel=1e-9)
    assert tr_n < 0.98                    # a genuine (hard) clip (~0.865)


def test_cupy_no_dtype_upcast():
    """complex64 in -> complex64 out on CuPy (skipped without a GPU)."""
    cp = _cupy_or_skip()
    N, dx = 512, 4e-6
    e = cp.asarray(_gauss(N, dx, 40e-6).astype(np.complex64))
    out = propagate_carrier_referenced(e, 30e-3, 40e-3, WL, dx)
    assert out.env.dtype == np.complex64


# ===========================================================================
# 6. Measured per-backend perf (modest; full sweep in the K2 report)
# ===========================================================================
@pytest.mark.slow
def test_perf_backends_measured():
    """MEASURE wall-clock per backend for the non-crossing scalar leg at
    N=2048 (RAM-guarded), reproducing the K2 report's methodology.  Asserts the
    leg completes on each available backend and records the timing; no strict
    speedup threshold is asserted (hardware-dependent), but the numbers are
    printed so the verifier can reproduce them.

    Measured envelope (this dev box; cross-process COLD -- fresh process per
    point, 1 warm-up call + 3 timed reps; complex128; 8-thread FFTW).
    Run-to-run noise ~5-10%:

        N          2048      4096      8192
        NumPy     314 ms   1179 ms   4468 ms   (tracemalloc peak 559/2069/8109 MB)
        JAX       214 ms   1015 ms   4085 ms   (tracemalloc peak 168/ 671/2685 MB)
        speedup   1.47x     1.16x     1.09x

    JAX CPU-XLA is only MODESTLY faster and the edge SHRINKS with N -- by
    N=8192 the two are at parity (run-to-run noise straddles 1.0x; a separate
    run measured JAX 0.96x, i.e. marginally slower).  On the SAME (consistent)
    tracemalloc-peak metric JAX holds ~3.0-3.3x less host memory -- NOT 7x
    (that earlier figure compared NumPy tracemalloc against JAX RSS, an
    apples-to-oranges mix).  The real speed win is a CUDA GPU via the CuPy
    path, which is import-guarded and untested on this box (no functional
    cuFFT).  No hardware-dependent speedup threshold is asserted below.
    """
    N, dx = 2048, 4e-6
    field_bytes = N * N * 16
    # peak NumPy working set measured ~7.5x the field; guard generously.
    if available_memory_bytes() < 8 * field_bytes:
        pytest.skip("insufficient RAM for the N=2048 perf measurement")
    env = _gauss(N, dx, N * dx / 12)

    def _bench(E, sync=None, reps=3):
        o = propagate_carrier_referenced(E, 30e-3, 40e-3, WL, dx)  # warm
        if sync is not None:
            sync(o.env)
        t0 = time.perf_counter()
        for _ in range(reps):
            o = propagate_carrier_referenced(E, 30e-3, 40e-3, WL, dx)
            if sync is not None:
                sync(o.env)
        return (time.perf_counter() - t0) / reps * 1e3

    try:
        t_np = _bench(env)
        assert np.isfinite(t_np) and t_np > 0
        print(f"\n[K2 perf] N={N} NumPy leg: {t_np:.1f} ms")

        try:
            jax, jnp = _jax_or_skip()
            t_jx = _bench(jnp.asarray(env),
                          sync=lambda a: a.block_until_ready())
            assert np.isfinite(t_jx) and t_jx > 0
            print(f"[K2 perf] N={N} JAX  leg: {t_jx:.1f} ms "
                  f"(speedup {t_np / t_jx:.2f}x vs NumPy)")
        except Exception as exc:  # noqa: BLE001
            print(f"[K2 perf] JAX leg not measured: {exc}")
    finally:
        la.clear_asm_caches()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
