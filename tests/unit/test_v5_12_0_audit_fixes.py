"""Regression tests for the 2026-06-08 PMM/RCWA suite-audit fixes.

See docs/audits/PMM_RCWA_AUDIT_2026_06_08.md.  One test per confirmed P1 bug that
is unit-testable without a GPU (B6 _check_energy/CuPy is covered by the existing
RCWA suite on the NumPy path + needs a device for the device-path assertion).
"""
import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements.rcwa import _binary_grating_convolutions, _require_jax_x64


def _t3(exx, eyy, ezz):
    return np.diag([exx, eyy, ezz]).astype(complex)


# --------------------------------------------------------------------------- B2
def test_b2_binary_grating_no_alias_high_orders():
    """B2: the binary-grating Fourier coefficients must not ALIAS for
    n_orders > ~1024 -- the fixed 4096-sample FFT folds |k| > 2048 onto low
    orders.  The auto-bumped grid matches a huge-grid reference, and low orders
    stay byte-identical to the prior 4096 path (zero regression)."""
    EPS_d, II_d = _binary_grating_convolutions(1.5, 1.0, 0.5, 1100)
    EPS_r, II_r = _binary_grating_convolutions(1.5, 1.0, 0.5, 1100, n_samples=32768)
    assert np.max(np.abs(EPS_d - EPS_r)) / np.max(np.abs(EPS_r)) < 1e-3
    assert np.max(np.abs(II_d - II_r)) / np.max(np.abs(II_r)) < 1e-3
    lo = _binary_grating_convolutions(1.5, 1.0, 0.5, 200)
    lo2 = _binary_grating_convolutions(1.5, 1.0, 0.5, 200, n_samples=4096)
    assert np.array_equal(lo[0], lo2[0]) and np.array_equal(lo[1], lo2[1])


# --------------------------------------------------------------------------- B5
def test_b5_opposite_sign_slant_is_not_uniform():
    """B5: a +phi/-phi equal-magnitude stack used to pass the abs()-based uniform
    test and cascade -phi layer modes against +phi-frame half-spaces (mixed
    covariant gauges, silently wrong S-matrix).  The signed-slant gate now treats
    it as non-uniform: 'covariant' raises, 'auto' falls back to convection."""
    rd, gr = _t3(4.0, 2.25, 2.0), _t3(2.0, 2.0, 2.0)

    def build(fac):
        st = la.PMMStack(1e-6, n_substrate=1.5, n_superstrate=1.0, degree=16,
                         factorization=fac)
        st.add_layer(0.3e-6, segments=[(0.5, rd), (0.5, gr)],
                     slant_angle=np.deg2rad(30.0))
        st.add_layer(0.3e-6, segments=[(0.5, rd), (0.5, gr)],
                     slant_angle=np.deg2rad(-30.0))
        return st.set_source(0.633e-6, angle=0.0)

    with pytest.raises(NotImplementedError):
        build("covariant").solve()
    o, R, T, J = build("auto").solve()              # convection fallback, no crash
    assert float(np.sum(R) + np.sum(T)) < 2.1       # energy-sane (2 incident pols)


# ------------------------------------------------------------------- JAX (B3/B4/B7)
try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:                  # pragma: no cover - environment dependent
    jax = jnp = None
    _HAS_JAX = False

# SCOPED skip (2026-06-10): the previous MODULE-LEVEL importorskip('jax')
# silently skipped this entire file -- including every non-JAX pin in it --
# on any environment without jax (CI and the WSL proxy among them).
_requires_jax = pytest.mark.skipif(not _HAS_JAX,
                                   reason="could not import 'jax'")

if _HAS_JAX:
    jax.config.update('jax_enable_x64', True)

from lumenairy.elements.pmm import pmm_jones_1d  # noqa: E402


@_requires_jax
def test_b4_jax_jones_metal_tensor_no_nan_crash():
    """B4: the JAX-Jones order-set sizing must take Re(sqrt(complex eps)), not
    sqrt(Re(eps)).  For a metal/ENZ eps (Re < 0) the latter is sqrt(neg)=NaN and
    int(NaN) crashes _n_propagating_orders -- the exact reflective-metal grating
    the PMM advertises.  It must solve to finite values."""
    er = _t3(-5.0 + 1.0j, -5.0 + 1.0j, -5.0 + 1.0j)
    eg = _t3(2.25, 2.25, 2.25)
    o, R, T, J = pmm_jones_1d(1e-6, jnp.asarray(er), jnp.asarray(eg), 1.5, 1.0,
                              0.2e-6, 0.5, 0.633e-6, degree=12, stabilize=False)
    assert np.all(np.isfinite(np.asarray(J)))


@_requires_jax
def test_b3_jax_jones_traced_duty_raises():
    """B3: a TRACED duty_cycle was silently frozen at 0.5 (the duty=0.5 answer for
    every duty, a 0.0 duty-gradient).  It must raise NotImplementedError instead of
    returning a wrong value."""
    er, eg = _t3(4.0, 2.25, 2.0), _t3(2.0, 2.0, 2.0)

    def f(duty):
        return pmm_jones_1d(1e-6, jnp.asarray(er), jnp.asarray(eg), 1.5, 1.0,
                            0.5e-6, duty, 0.633e-6, degree=12,
                            stabilize=False)[1].sum()

    with pytest.raises(NotImplementedError, match="duty_cycle"):
        jax.grad(f)(0.5)


@_requires_jax
def test_b7_jax_path_requires_x64():
    """B7: the JAX path must RAISE (not just warn) when x64 is disabled -- JAX
    silently truncates complex128 to complex64 and the eigenproblem is
    ill-conditioned in single precision."""
    saved = bool(jax.config.read("jax_enable_x64"))
    try:
        jax.config.update("jax_enable_x64", False)
        with pytest.raises(RuntimeError, match="x64"):
            _require_jax_x64("test")
    finally:
        jax.config.update("jax_enable_x64", saved)
