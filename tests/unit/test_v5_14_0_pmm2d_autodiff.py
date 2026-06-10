"""v5.14.0 -- JAX-differentiable 2-D hybrid PMM (Phase 7).

``pmm_efficiency_2d`` auto-dispatches to a jnp twin on JAX inputs: traced
``eps_pillar`` / ``eps_host`` / half-space indices / ``depth`` / ``wavelength``
/ ``theta`` / ``phi`` on STATIC pillar bounds + degree + orders (the 1-D binary
twin's scope split).  Gates (the Phase-7 pressure-test list): forward parity
with NumPy, AD-vs-FD gradients at BOTH a C4 square pillar (mass-degenerate
eigenpairs) and a 1%-rectangular one, jit compile + reuse, lossy absorptance
parity, the symmetry-zero angle-gradient artifact bound, and the stabilize
rejection."""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from lumenairy.elements.pmm import pmm_efficiency_2d  # noqa: E402

_P = 0.6e-6
_WL = 0.55e-6
_DEP = 0.25e-6
_XB = (0.2 * _P, 0.6 * _P)
_KW = dict(degree=5, n_orders=2, polarization="te")


def _sumT(eps_pillar, yb=_XB, depth=_DEP, wl=_WL, theta=0.0):
    o, R, T = pmm_efficiency_2d(_P, _P, eps_pillar, 1.0, _XB, yb, 1.5, 1.0,
                                depth, wl, theta=theta, **_KW)
    return jnp.sum(T)


def test_gate_fwd_twin_matches_numpy():
    o1, R1, T1 = pmm_efficiency_2d(_P, _P, 6.0, 1.0, _XB, _XB, 1.5, 1.0,
                                   _DEP, _WL, **_KW)
    o2, R2, T2 = pmm_efficiency_2d(_P, _P, jnp.asarray(6.0 + 0j), 1.0, _XB,
                                   _XB, 1.5, 1.0, _DEP, _WL, **_KW)
    assert np.array_equal(o1, np.asarray(o2))
    assert np.max(np.abs(R1 - np.asarray(R2))) < 1e-11
    assert np.max(np.abs(T1 - np.asarray(T2))) < 1e-11


@pytest.mark.parametrize("yb", [_XB, (0.2 * _P, 0.604 * _P)],
                         ids=["C4-square", "1pc-rect"])
def test_gate_grad_eps_vs_fd(yb):
    g = jax.grad(lambda e: _sumT(e, yb=yb))(jnp.asarray(6.0 + 0j))
    h = 1e-5
    fd = (float(_sumT(jnp.asarray(6.0 + h + 0j), yb=yb))
          - float(_sumT(jnp.asarray(6.0 - h + 0j), yb=yb))) / (2 * h)
    assert abs(float(jnp.real(g)) - fd) < 1e-4 * max(abs(fd), 1e-6)


def test_gate_grad_depth_and_wavelength_vs_fd():
    gd = jax.grad(lambda d: _sumT(jnp.asarray(6.0 + 0j), depth=d))(
        jnp.asarray(_DEP))
    h = _DEP * 1e-6
    fd = (float(_sumT(jnp.asarray(6.0 + 0j), depth=jnp.asarray(_DEP + h)))
          - float(_sumT(jnp.asarray(6.0 + 0j),
                        depth=jnp.asarray(_DEP - h)))) / (2 * h)
    assert abs(float(gd) - fd) < 1e-4 * max(abs(fd), 1.0)
    gw = jax.grad(lambda w: _sumT(jnp.asarray(6.0 + 0j), wl=w))(
        jnp.asarray(_WL))
    hw = _WL * 1e-6
    fdw = (float(_sumT(jnp.asarray(6.0 + 0j), wl=jnp.asarray(_WL + hw)))
           - float(_sumT(jnp.asarray(6.0 + 0j),
                         wl=jnp.asarray(_WL - hw)))) / (2 * hw)
    assert abs(float(gw) - fdw) < 1e-4 * max(abs(fdw), 1.0)


def test_gate_jit_compiles_and_reuses():
    jf = jax.jit(lambda e: _sumT(e))
    v1 = float(jf(jnp.asarray(6.0 + 0j)))
    v2 = float(jf(jnp.asarray(6.5 + 0j)))
    assert np.isfinite(v1) and np.isfinite(v2) and v1 != v2


def test_gate_lossy_absorptance_matches_numpy():
    oL, RL, TL = pmm_efficiency_2d(_P, _P, jnp.asarray(6.0 + 0.8j), 1.0,
                                   _XB, _XB, 1.5, 1.0, _DEP, _WL, **_KW)
    oN, RN, TN = pmm_efficiency_2d(_P, _P, 6.0 + 0.8j, 1.0, _XB, _XB, 1.5,
                                   1.0, _DEP, _WL, **_KW)
    A_j = 1.0 - float(np.sum(np.asarray(RL))) - float(np.sum(np.asarray(TL)))
    A_n = 1.0 - float(RN.sum()) - float(TN.sum())
    assert A_n > 0.01                      # genuinely absorbing
    assert abs(A_j - A_n) < 1e-10


def test_gate_angle_grad_oblique_vs_fd():
    """The MEANINGFUL angle-gradient gate: away from the symmetry point the
    AD theta-gradient must match FD."""
    g = jax.grad(lambda th: _sumT(jnp.asarray(6.0 + 0j), theta=th))(
        jnp.asarray(0.3))
    h = 1e-6
    fd = (float(_sumT(jnp.asarray(6.0 + 0j), theta=jnp.asarray(0.3 + h)))
          - float(_sumT(jnp.asarray(6.0 + 0j),
                        theta=jnp.asarray(0.3 - h)))) / (2 * h)
    assert abs(float(g) - fd) < 1e-4 * max(abs(fd), 1e-3)


def test_gate_angle_grad_at_normal_offcenter_is_genuine():
    """The default test pillar is OFF-CENTER (no mirror symmetry), so
    dT/d(theta) at normal incidence is genuinely NONZERO -- and AD matches FD
    (measured 0.0628 both, rel ~3e-6).  This catches the trap of assuming a
    symmetry zero for any 'square' pillar."""
    g = jax.grad(lambda th: _sumT(jnp.asarray(6.0 + 0j), theta=th))(
        jnp.asarray(0.0))
    h = 1e-6
    fd = (float(_sumT(jnp.asarray(6.0 + 0j), theta=jnp.asarray(h)))
          - float(_sumT(jnp.asarray(6.0 + 0j),
                        theta=jnp.asarray(-h)))) / (2 * h)
    assert abs(fd) > 1e-3                       # genuinely nonzero
    assert abs(float(g) - fd) < 1e-4 * abs(fd)


def test_gate_degen_angle_grad_centered_square_is_clean_zero():
    """A CENTERED square at exactly normal incidence: the true derivative is a
    symmetry zero (FD ~ -8e-10) and AD returns ~ -4e-15 -- NO degenerate-gauge
    artifact (the 2-D twin is cleaner here than the 1-D's documented ~1e-3
    normal-incidence artifact)."""
    cb = (0.2 * _P, 0.8 * _P)
    def s(th):
        o, R, T = pmm_efficiency_2d(_P, _P, jnp.asarray(6.0 + 0j), 1.0, cb,
                                    cb, 1.5, 1.0, _DEP, _WL, theta=th, **_KW)
        return jnp.sum(T)
    h = 1e-6
    fd = (float(s(jnp.asarray(h))) - float(s(jnp.asarray(-h)))) / (2 * h)
    assert abs(fd) < 1e-6                       # the true symmetry zero
    g = float(jax.grad(s)(jnp.asarray(0.0)))
    assert np.isfinite(g)
    assert abs(g) < 1e-8                        # clean AD zero, no artifact


def test_stabilize_rejected_on_jax_path():
    with pytest.raises(ValueError, match="stabilize"):
        pmm_efficiency_2d(_P, _P, jnp.asarray(6.0 + 0j), 1.0, _XB, _XB,
                          1.5, 1.0, _DEP, _WL, stabilize=True, **_KW)
