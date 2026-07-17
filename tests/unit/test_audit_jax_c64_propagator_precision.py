"""Regression tests for audit finding S2-3 (v5.24.4).

The JAX complex64 branches of the ASM / MFT / Fresnel / Fraunhofer
propagators used to build their FIELD-INDEPENDENT transfer function /
quadratic-phase carriers directly under the tracer with
``jnp.arange(dtype=float64)``.  When ``jax_enable_x64`` is off (the JAX
default) that request is silently truncated to float32 and no mod-2pi
fold is applied, so the huge kernel argument ``kz * z`` (~1e6 rad for
ASM/MFT) is evaluated in float32 -- ~26 dB worse than the documented
"float32 noise floor, does not degrade with phase magnitude" contract
that the NumPy complex64 path already honors.

The fix host-builds those carriers in float64 (numpy) and moves the
finished, bounded result onto the device.  These tests pin the JAX
complex64 output to the full-double-precision NumPy reference (an
INDEPENDENT oracle -- a different code path) at the float32 noise floor,
and confirm the input-field gradient still flows (host-building H is
trace-safe because H does not depend on the field).

Measured (WSL venv, x64 off):
    propagator   pre-fix (emulated)   post-fix vs c128 truth
    ASM          4.4e-3               2.0e-7
    ASM-MFT      ~few e-3             4.1e-7
    Fresnel      3.7e-4               4.4e-7
    Fraunhofer   2.6e-3               7.3e-8
"""
import numpy as np
import pytest

from lumenairy.propagators.propagation import (
    angular_spectrum_propagate,
    angular_spectrum_propagate_mft,
    fraunhofer_propagate,
    fresnel_propagate,
)

WL = 1.31e-6


def _jnp_or_skip():
    """Import jax (skip if absent) and skip if x64 is on process-wide --
    the complex64 phase-fold bug only manifests with x64 OFF, the JAX
    default and what the standalone test run uses."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    if jax.config.jax_enable_x64:
        pytest.skip(
            "jax_enable_x64 is on process-wide; the complex64 phase-fold "
            "bug (audit S2-3) only reproduces with x64 OFF (JAX default)")
    return jax, jnp


def _relerr(got, ref):
    got = np.asarray(got)
    ref = np.asarray(ref)
    return float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))


def _gauss(N, dx, w0, tilt=0.0):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2) * np.exp(2j * np.pi * tilt * X)


def test_asm_jax_complex64_honors_phase_contract():
    _jax, jnp = _jnp_or_skip()
    N, dx, z = 256, 2e-6, 8e-3
    E = _gauss(N, dx, 30e-6, tilt=3000.0)
    ref = angular_spectrum_propagate(E.astype(np.complex128), z, WL, dx)
    got = angular_spectrum_propagate(
        jnp.asarray(E.astype(np.complex64)), z, WL, dx)
    rel = _relerr(got, ref)
    assert rel < 1e-4, (
        f"ASM JAX c64 rel err {rel:.2e} vs c128 truth exceeds the float32 "
        f"noise floor (pre-fix ~4.4e-3); H phase must be built in f64.")


def test_mft_jax_complex64_honors_phase_contract():
    _jax, jnp = _jnp_or_skip()
    N, dx, z = 256, 2e-6, 8e-3
    E = _gauss(N, dx, 30e-6, tilt=3000.0)
    ref = angular_spectrum_propagate_mft(
        E.astype(np.complex128), z, WL, dx, dx, N)
    got = angular_spectrum_propagate_mft(
        jnp.asarray(E.astype(np.complex64)), z, WL, dx, dx, N)
    rel = _relerr(got, ref)
    assert rel < 1e-4, (
        f"ASM-MFT JAX c64 rel err {rel:.2e} vs c128 truth exceeds the "
        f"float32 noise floor (pre-fix ~a few e-3).")


def test_fresnel_jax_complex64_honors_phase_contract():
    _jax, jnp = _jnp_or_skip()
    N, dx, z = 1024, 4e-6, 3e-3
    E = _gauss(N, dx, N * dx / 8)
    ref = fresnel_propagate(E.astype(np.complex128), z, WL, dx)[0]
    got = fresnel_propagate(jnp.asarray(E.astype(np.complex64)), z, WL, dx)[0]
    rel = _relerr(got, ref)
    assert rel < 5e-5, (
        f"Fresnel JAX c64 rel err {rel:.2e} vs c128 truth exceeds the "
        f"float32 noise floor (pre-fix ~3.7e-4).")


def test_fraunhofer_jax_complex64_honors_phase_contract():
    _jax, jnp = _jnp_or_skip()
    N, dx, z = 256, 4e-6, 100e-3
    E = _gauss(N, dx, N * dx / 8)
    ref = fraunhofer_propagate(E.astype(np.complex128), z, WL, dx)[0]
    got = fraunhofer_propagate(jnp.asarray(E.astype(np.complex64)), z, WL, dx)[0]
    rel = _relerr(got, ref)
    assert rel < 1e-4, (
        f"Fraunhofer JAX c64 rel err {rel:.2e} vs c128 truth exceeds the "
        f"float32 noise floor (pre-fix ~2.6e-3).")


def test_asm_jax_field_gradient_survives_host_built_H():
    """The host-built (numpy-f64) transfer function is a constant w.r.t.
    the traced field, so gradients of a field-dependent scalar must still
    flow.  Confirms the fix is trace-safe rather than a numpy fallback
    that silently breaks autodiff.  ASM is linear in the input field, so
    merit(a) = a^2 * sum|ASM(E)|^2 and d/da = 2a * sum|ASM(E)|^2."""
    jax, jnp = _jnp_or_skip()
    N, dx, z = 64, 2e-6, 2e-3
    Ej = jnp.asarray(_gauss(N, dx, 12e-6).astype(np.complex64))

    def merit(a):
        out = angular_spectrum_propagate(a * Ej, z, WL, dx)
        return jnp.sum(jnp.abs(out) ** 2)

    g = float(jax.grad(merit)(2.0))
    out0 = np.asarray(angular_spectrum_propagate(Ej, z, WL, dx))
    expected = 2.0 * 2.0 * float(np.sum(np.abs(out0) ** 2))
    assert np.isfinite(g)
    assert abs(g - expected) / abs(expected) < 1e-3, (
        f"field gradient {g:.6e} != expected {expected:.6e}; the host-built "
        f"H should be a trace constant, not break autodiff.")
