"""JAX differentiability at the STACK level: ``PMMStack`` + ``PMM2DStack``.

The multilayer twins (``pmm/_jax_stack.py``, ``pmm/_jax_stack2d.py``) compose
the validated single-layer machinery -- frozen NumPy geometry, eps-LINEAR
traced assembly, the gauge-stable custom-VJP eig, the backend-generic
Redheffer cascade -- into full stack solves.  Surface:

* 1-D ``PMMStack``: all-vertical IN-PLANE stacks; gradients w.r.t. segment
  eps (re+im, scalar or in-plane tensor), layer thickness, wavelength,
  angle, half-space indices.  Widths/walls static.
* 2-D ``PMM2DStack``: the scalar surface; traced uniform eps, thickness,
  wavelength, theta/phi, half-spaces, and traced eps_cell VALUES via
  ``add_layer(eps_cell=<jax>, region_layout=<int grid>)`` (concrete walls).
* Loud rejection of everything off-surface: slant / OOP / tensor cells /
  stabilize / retain_internal / the assemble-once sweep + prepare paths.

REQUIRES JAX double precision (``jax_enable_x64``).
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from lumenairy.elements.pmm import PMM2DStack, PMMStack  # noqa: E402

P1, WL = 0.8e-6, 0.633e-6
T1, T2 = 0.25e-6, 0.10e-6


def _fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


# =========================================================================== #
# 1-D PMMStack
# =========================================================================== #

def _stack_1d(eps_a=4.0 + 0.5j, t1=T1, wl=WL, ang=0.0, nsub=1.5, jax_on=()):
    """2-layer stack; the names in ``jax_on`` arrive as traced jnp scalars."""
    j = lambda name, v: jnp.asarray(v) if name in jax_on else v  # noqa: E731
    st = PMMStack(P1, n_substrate=j("nsub", nsub), n_superstrate=1.0,
                  degree=12)
    st.add_layer(j("t1", t1), segments=[(0.4, j("eps", eps_a)), (0.6, 1.0)])
    st.add_layer(T2, eps=2.25)
    st.set_source(j("wl", wl), angle=j("ang", ang))
    return st


@pytest.mark.parametrize("ang", [0.0, 0.3])
def test_pmm_stack_jax_forward_matches_numpy(ang):
    o0, R0, T0, J0 = _stack_1d(ang=ang).solve()
    o1, R1, T1_, J1 = _stack_1d(
        ang=ang, jax_on=("eps", "t1", "wl", "ang", "nsub")).solve()
    assert np.array_equal(np.asarray(o0), np.asarray(o1))
    assert np.max(np.abs(R0 - np.asarray(R1))) < 1e-12
    assert np.max(np.abs(T0 - np.asarray(T1_))) < 1e-12
    assert np.max(np.abs(J0 - np.asarray(J1))) < 1e-12


def test_pmm_stack_jax_grad_eps_thickness():
    def loss_re(re):
        _, R, _, _ = _stack_1d(eps_a=re + 0.5j, jax_on=("eps",)).solve()
        return R[0].sum()

    def loss_im(im):
        _, R, _, _ = _stack_1d(eps_a=4.0 + 1j * im, jax_on=("eps",)).solve()
        return R[0].sum()

    def loss_t(t1):
        _, R, _, _ = _stack_1d(t1=t1, jax_on=("t1",)).solve()
        return R[0].sum()

    for f, x, h in ((loss_re, 4.0, 1e-5), (loss_im, 0.5, 1e-5),
                    (loss_t, T1, 1e-12)):
        ad = float(jax.grad(f)(x))
        fd = _fd(lambda v, _f=f: float(_f(jnp.asarray(v))), x, h)
        assert abs(ad - fd) / max(abs(fd), 1e-12) < 1e-5, (ad, fd)


def test_pmm_stack_jax_grad_wavelength_angle_halfspace():
    def loss_wl(wl):
        _, R, _, _ = _stack_1d(wl=wl, jax_on=("wl",)).solve()
        return R[0].sum()

    def loss_ang(a):
        _, R, _, _ = _stack_1d(ang=a, jax_on=("ang",)).solve()
        return R[1].sum()

    def loss_n(n):
        _, _, T, _ = _stack_1d(nsub=n, jax_on=("nsub",)).solve()
        return T[0].sum()

    for f, x, h in ((loss_wl, WL, 1e-12), (loss_ang, 0.3, 1e-6),
                    (loss_n, 1.5, 1e-6)):
        ad = float(jax.grad(f)(x))
        fd = _fd(lambda v, _f=f: float(_f(jnp.asarray(v))), x, h)
        assert abs(ad - fd) / max(abs(fd), 1e-12) < 1e-5, (ad, fd)


def test_pmm_stack_jax_guards():
    st = _stack_1d(jax_on=("eps",))
    with pytest.raises(NotImplementedError, match="retain_internal"):
        st.solve(retain_internal=True)
    with pytest.raises(NotImplementedError, match="NumPy-only"):
        st.solve_vs_wavelength(np.array([0.6e-6, 0.65e-6]))
    with pytest.raises(NotImplementedError, match="NumPy-only"):
        st.prepare()
    st2 = PMMStack(P1, degree=10)
    st2.add_layer(0.1e-6, segments=[(0.5, jnp.asarray(4.0 + 0j)), (0.5, 1.0)],
                  slant_angle=0.2)
    st2.set_source(WL)
    with pytest.raises(NotImplementedError, match="slant"):
        st2.solve()
    oop = np.eye(3, dtype=complex) * 2.25
    oop[0, 2] = oop[2, 0] = 0.3
    st3 = PMMStack(P1, degree=10)
    st3.add_layer(0.1e-6, eps=oop)
    st3.set_source(jnp.asarray(WL))
    with pytest.raises(NotImplementedError, match="out-of-plane"):
        st3.solve()


# =========================================================================== #
# 2-D PMM2DStack
# =========================================================================== #

P2, S2 = 0.6e-6, 6
_LAYOUT = np.zeros((S2, S2), dtype=np.int64)
_LAYOUT[1:4, 1:4] = 1


def _cell(eps_p):
    c = np.full((S2, S2), 1.0 + 0j)
    c[1:4, 1:4] = eps_p
    return c


def _stack_2d(eps_p=6.0 + 1.5j, eps_u=2.25, t1=0.15e-6, wl=WL, th=0.0,
              jax_on=(), traced_cell=False):
    j = lambda name, v: jnp.asarray(v) if name in jax_on else v  # noqa: E731
    st = PMM2DStack(P2, n_substrate=1.5, n_superstrate=1.0, degree=7,
                    n_orders=3)
    if traced_cell:
        c = jnp.asarray(_cell(0j))
        c = c.at[_LAYOUT == 0].set(jnp.asarray(1.0 + 0j))
        c = c.at[_LAYOUT == 1].set(jnp.asarray(eps_p))
        st.add_layer(t1, eps_cell=c, region_layout=_LAYOUT)
    else:
        st.add_layer(j("t1", t1), eps_cell=_cell(eps_p))
    st.add_layer(0.08e-6, eps=j("eps_u", eps_u))
    st.set_source(j("wl", wl), theta=j("th", th))
    return st


def test_pmm2d_stack_jax_forward_matches_numpy():
    o0, R0, T0, J0 = _stack_2d().solve()
    for kw in (dict(jax_on=("eps_u",)), dict(jax_on=("t1", "wl")),
               dict(traced_cell=True)):
        o1, R1, T1_, J1 = _stack_2d(**kw).solve()
        assert np.array_equal(np.asarray(o0), np.asarray(o1))
        assert np.max(np.abs(R0 - np.asarray(R1))) < 1e-12, kw
        assert np.max(np.abs(T0 - np.asarray(T1_))) < 1e-12, kw
        assert np.max(np.abs(J0 - np.asarray(J1))) < 1e-12, kw


def test_pmm2d_stack_jax_forward_oblique():
    o0, R0, _T0, J0 = _stack_2d(th=0.25).solve()
    o1, R1, _T1, J1 = _stack_2d(th=0.25, jax_on=("th",)).solve()
    assert np.max(np.abs(R0 - np.asarray(R1))) < 1e-12
    assert np.max(np.abs(J0 - np.asarray(J1))) < 1e-12


def test_pmm2d_stack_jax_grads():
    def loss_u(eps_u):
        _, R, _, _ = _stack_2d(eps_u=eps_u, jax_on=("eps_u",)).solve()
        return R[0].sum()

    def loss_t(t1):
        _, R, _, _ = _stack_2d(t1=t1, jax_on=("t1",)).solve()
        return R[0].sum()

    def loss_cell(re):
        _, R, _, _ = _stack_2d(eps_p=re + 1.5j, traced_cell=True).solve()
        return R[0].sum()

    for f, x, h in ((loss_u, 2.25, 1e-6), (loss_t, 0.15e-6, 1e-12),
                    (loss_cell, 6.0, 1e-5)):
        ad = float(jax.grad(f)(x))
        fd = _fd(lambda v, _f=f: float(_f(jnp.asarray(v))), x, h)
        assert abs(ad - fd) / max(abs(fd), 1e-12) < 1e-5, (ad, fd)


def test_pmm2d_stack_jax_guards():
    st = _stack_2d(jax_on=("eps_u",))
    with pytest.raises(NotImplementedError, match="retain_internal"):
        st.solve(retain_internal=True)
    tens = np.zeros((S2, S2, 3, 3), complex)
    for i in range(3):
        tens[:, :, i, i] = 2.25
    st2 = PMM2DStack(P2, degree=7, n_orders=3)
    st2.add_layer(0.1e-6, eps_tensor_cell=tens)
    st2.add_layer(0.05e-6, eps=jnp.asarray(2.0 + 0j))
    st2.set_source(WL)
    with pytest.raises(NotImplementedError, match="tensor"):
        st2.solve()
    with pytest.raises(ValueError, match="region_layout"):
        PMM2DStack(P2, degree=7, n_orders=3).add_layer(
            0.1e-6, eps_cell=jnp.asarray(_cell(4.0)))
    with pytest.raises(NotImplementedError, match="traced eps_tensor_cell"):
        PMM2DStack(P2, degree=7, n_orders=3).add_layer(
            0.1e-6, eps_tensor_cell=jnp.asarray(tens))
