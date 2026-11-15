"""The uniform-layer fast path in the JAX twin of PMMStack (2026-11-14; fixture
replaced by a generic one and re-measured 2026-11-15).

The twin runs the NumPy kernels themselves (``_core._sem_modes_uniform``,
``_uniform_cascade._cascade_vertical`` and friends) with ``xp=jax.numpy``.
References: the NumPy solve (forward), the twin with the fast path off (the
dense eig per layer, ``stack._UNIFORM_FAST_PATH = False``) and central finite
differences of the NumPy solve (gradients).  Every bar carries its
measurement (date, BLAS build).
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import lumenairy.elements.pmm._jax_stack as JS  # noqa: E402
import lumenairy.elements.pmm.stack as S  # noqa: E402
from lumenairy.elements.pmm import PMMStack  # noqa: E402

#: A variant of the strip / spacer / mirror stack (a generic fixture).
WL, P, F = 1500e-9, 500e-9, 180.0 / 500.0
E_STRIP, E_MIRROR = -90.0 + 7.0j, -130.0 + 20.0j
#: the lossy film's permittivity and thickness (the traced parameters)
E0, T0 = 2.2 + 0.6j, 3e-9


def _build(e_ox=E0, t_ox=T0, n_sub_layers=6, angle=0.2,
           degree=8, n_el=3):
    """Metal strip over a stack of thin uniform films over a mirror; the
    lossy film (``e_ox``, ``t_ox``) may be traced."""
    st = PMMStack(P, n_substrate=1.5, degree=degree, elements_per_region=n_el)
    st.add_layer(40e-9, segments=[(F, E_STRIP), (1.0 - F, 1.0)])
    for k in range(n_sub_layers):
        st.add_layer(1.5e-9, eps=2.8 + 0.08 * k)
    st.add_layer(t_ox, eps=e_ox)
    st.add_layer(6e-9, eps=4.2)
    st.add_layer(80e-9, eps=E_MIRROR)
    st.set_source(WL, angle=angle)
    return st


def _J(st):
    return np.asarray(st.solve()[3])


#: Twin (fast path) vs NumPy (fast path), max over J.  Fixture replaced by a generic one; RE-MEASURED 2026-11-15: 4.7e-13
#: (Windows/MKL); the twin's fast and dense paths differ by 3.0e-13.  Bar
#: kept: 21 x the reading.
FWD_BAR = 1e-11


def test_twin_fast_path_forward_matches_numpy():
    J_np = _J(_build())
    J_jx = _J(_build(e_ox=jnp.asarray(E0)))
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
    st = _build(e_ox=jnp.asarray(E0))
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
#: imaginary part), its thickness, and the incidence angle.  Fixture
#: replaced by a generic one; RE-MEASURED 2026-11-15 (Windows/MKL;
#: thickness per nm, relative where |g| > 1): fast vs dense twin <= 9.8e-14;
#: fast twin vs central FD 1.5e-10 / 1.3e-10 / 2.9e-9 (eps real / imag /
#: thickness) and 7.6e-9 (angle).  Speed (forward + gradient w.r.t. one
#: uniform eps, 44 layers, degree 8 x 3 el.): 38.2 s dense, 1.12 s fast
#: (34x), Windows/MKL, BLAS 2 threads.
GRAD_BAR_DENSE = 1e-10
GRAD_BAR_FD = 1e-6


def test_twin_fast_path_gradients(monkeypatch):
    e0, t0, a0 = E0, T0, 0.2
    g = jax.grad(lambda er, ei, t, a: _loss(er + 1j * ei, t, a),
                 argnums=(0, 1, 2, 3))
    gf = [float(v) for v in g(jnp.asarray(e0.real), jnp.asarray(e0.imag),
                               jnp.asarray(t0), jnp.asarray(a0))]
    monkeypatch.setattr(S, "_UNIFORM_FAST_PATH", False)
    gd = [float(v) for v in g(jnp.asarray(e0.real), jnp.asarray(e0.imag),
                               jnp.asarray(t0), jnp.asarray(a0))]
    monkeypatch.setattr(S, "_UNIFORM_FAST_PATH", True)
    scale = [1.0, 1.0, 1e-9, 1.0]                  # per-unit -> per-nm
    for a, b, s in zip(gf, gd, scale):
        assert abs(a - b) * s < GRAD_BAR_DENSE * max(1.0, abs(a) * s), (a, b)

    def num(f, x, h):
        return float(_fd(lambda v: float(f(v)), x, h))
    fd = [num(lambda v: _loss(v + 1j * e0.imag, t0, a0), e0.real, 1e-5),
          num(lambda v: _loss(e0.real + 1j * v, t0, a0), e0.imag, 1e-5),
          num(lambda v: _loss(e0, v, a0), t0, 1e-12),
          num(lambda v: _loss(e0, t0, v), a0, 1e-6)]
    for a, b, s in zip(gf, fd, scale):
        assert abs(a - b) * s < GRAD_BAR_FD * max(1.0, abs(a) * s), (a, b)
