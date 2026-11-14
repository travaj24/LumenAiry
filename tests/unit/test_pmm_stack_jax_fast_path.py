"""The uniform-layer fast path in the JAX twin of PMMStack (2026-11-14).

The twin runs the NumPy kernels themselves (``_core._sem_modes_uniform``,
``_uniform_cascade._cascade_vertical`` and friends) with ``xp=jax.numpy``.
References: the NumPy solve (forward), the twin with the fast path off (the
dense eig per layer, ``stack._UNIFORM_FAST_PATH = False``) and central finite
differences of the NumPy solve (gradients).  Every bar carries its
measurement (date, both builds).
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import lumenairy.elements.pmm._jax_stack as JS  # noqa: E402
import lumenairy.elements.pmm.stack as S  # noqa: E402
from lumenairy.elements.pmm import PMMStack  # noqa: E402

WL, P, F = 1300e-9, 370e-9, 175.0 / 370.0
E_AU, E_AL = -74.0 + 6.0j, -160.0 + 30.0j


def _build(e_ox=2.5 + 0.5j, t_ox=5e-9, n_sub_layers=6, angle=0.2,
           degree=8, n_el=3):
    """Metal strip over a stack of thin uniform films over a mirror; the
    lossy oxide film (``e_ox``, ``t_ox``) may be traced."""
    st = PMMStack(P, n_substrate=1.447, degree=degree, elements_per_region=n_el)
    st.add_layer(50e-9, segments=[(F, E_AU), (1.0 - F, 1.0)])
    for k in range(n_sub_layers):
        st.add_layer(1.5e-9, eps=3.05 + 0.08 * k)
    st.add_layer(t_ox, eps=e_ox)
    st.add_layer(7e-9, eps=3.53)
    st.add_layer(70e-9, eps=E_AL)
    st.set_source(WL, angle=angle)
    return st


def _J(st):
    return np.asarray(st.solve()[3])


#: Twin (fast path) vs NumPy (fast path), max over J.  MEASURED 2026-11-14:
#: 9.4e-14 (Windows/MKL), 6.8e-14 (WSL/OpenBLAS); the twin's fast and dense
#: paths differ by 1.9e-13 / 1.5e-13.  Bar = 100 x the cross-build envelope.
FWD_BAR = 1e-11


def test_twin_fast_path_forward_matches_numpy():
    J_np = _J(_build())
    J_jx = _J(_build(e_ox=jnp.asarray(2.5 + 0.5j)))
    assert np.max(np.abs(J_np - J_jx)) < FWD_BAR


def test_twin_fast_path_skips_the_uniform_layer_eigs(monkeypatch):
    """Deterministic speed claim: with the fast path the twin runs the dense
    coupled eig for the patterned layer only."""
    calls = {"n": 0}
    orig = JS._jpmm_sem_modes_tensor

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    monkeypatch.setattr(JS, "_jpmm_sem_modes_tensor", spy)
    st = _build(e_ox=jnp.asarray(2.5 + 0.5j))
    st.solve()
    assert calls["n"] == 1
    calls["n"] = 0
    monkeypatch.setattr(S, "_UNIFORM_FAST_PATH", False)
    st.solve()
    assert calls["n"] == 10                      # every layer


def _loss(e_ox, t_ox, angle):
    st = _build(e_ox=e_ox, t_ox=t_ox, angle=angle)
    o, R, T, J = st.solve()
    return jnp.real(J[0, 0]) + 0.3 * jnp.imag(J[1, 1]) + jnp.sum(R[0])


def _fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2.0 * h)


#: Gradients of the fast-path twin vs the dense twin and vs central FD of
#: the NumPy solve, w.r.t. a uniform layer's permittivity (real and
#: imaginary part), its thickness, and the incidence angle.  MEASURED
#: 2026-11-14 (thickness per nm): fast vs dense twin <= 1.5e-12 (Windows/MKL),
#: 1.6e-12 (WSL/OpenBLAS); fast twin vs central FD 2.5e-10 / 9e-12 / 6.8e-10
#: (eps real / imag / thickness) and 2.8e-7 (angle, the FD truncation of a
#: 1e-6 step) on both builds.  Speed (forward + gradient w.r.t. one uniform
#: eps, 44 layers, degree 8 x 3 el.): 68.5 s dense, 1.05 s fast (65x),
#: Windows/MKL, BLAS 2 threads.
GRAD_BAR_DENSE = 1e-10
GRAD_BAR_FD = 1e-6


def test_twin_fast_path_gradients(monkeypatch):
    e0, t0, a0 = 2.5 + 0.5j, 5e-9, 0.2
    g = jax.grad(lambda er, ei, t, a: _loss(er + 1j * ei, t, a),
                 argnums=(0, 1, 2, 3))
    gf = [float(v) for v in g(jnp.asarray(2.5), jnp.asarray(0.5),
                               jnp.asarray(t0), jnp.asarray(a0))]
    monkeypatch.setattr(S, "_UNIFORM_FAST_PATH", False)
    gd = [float(v) for v in g(jnp.asarray(2.5), jnp.asarray(0.5),
                               jnp.asarray(t0), jnp.asarray(a0))]
    monkeypatch.setattr(S, "_UNIFORM_FAST_PATH", True)
    scale = [1.0, 1.0, 1e-9, 1.0]                  # per-unit -> per-nm
    for a, b, s in zip(gf, gd, scale):
        assert abs(a - b) * s < GRAD_BAR_DENSE * max(1.0, abs(a) * s), (a, b)

    def num(f, x, h):
        return float(_fd(lambda v: float(f(v)), x, h))
    fd = [num(lambda v: _loss(v + 0.5j, t0, a0), 2.5, 1e-5),
          num(lambda v: _loss(2.5 + 1j * v, t0, a0), 0.5, 1e-5),
          num(lambda v: _loss(e0, v, a0), t0, 1e-12),
          num(lambda v: _loss(e0, t0, v), a0, 1e-6)]
    for a, b, s in zip(gf, fd, scale):
        assert abs(a - b) * s < GRAD_BAR_FD * max(1.0, abs(a) * s), (a, b)
