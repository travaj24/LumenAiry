"""EMT homogenization bridge + Berreman JAX twin (v5.14.5).

* **EMT bridge** (`lumenairy.elements.emt`): a sub-wavelength 1-D grating
  homogenizes (Rytov) to a uniaxial tensor; the EMT + Berreman slab Jones
  converges MONOTONICALLY onto the rigorous `rcwa_jones_1d` / `pmm_jones_1d`
  as the period shrinks.  Mixing rules (Maxwell-Garnett, Bruggeman) have
  their analytic end-point / symmetry properties.
* **Berreman JAX twin** (`_berreman_jax`): dispatched on JAX inputs;
  forward-identical to NumPy, AD-vs-FD across every parameter class, with
  the trace-safe argsort forward/backward split (where the PMM out-of-plane
  path's host argsort could not trace).
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements.berreman import berreman_jones_1d
from lumenairy.elements.emt import (
    bruggeman,
    maxwell_garnett,
    rytov_segments_tensor,
    rytov_tensor,
)

WL = 0.633e-6
_I3 = np.eye(3, dtype=complex)


# =========================================================================== #
# EMT: Rytov tensor + convergence to the rigorous solve
# =========================================================================== #

def test_rytov_tensor_structure():
    ea, eb, f = 6.25, 1.0, 0.45
    t = rytov_tensor(ea, eb, f, order=0)
    par = f * ea + (1 - f) * eb                 # arithmetic (y, z)
    perp = 1.0 / (f / ea + (1 - f) / eb)        # harmonic (x)
    assert abs(t[1, 1] - par) < 1e-12 and abs(t[2, 2] - par) < 1e-12
    assert abs(t[0, 0] - perp) < 1e-12
    assert perp < par                           # harmonic <= arithmetic
    # off-diagonals zero
    assert np.max(np.abs(t - np.diag(np.diag(t)))) < 1e-15


def test_rytov_converges_to_rigorous():
    """EMT+Berreman slab Jones -> rigorous RCWA monotonically as period->0."""
    ea, eb, f, depth = 6.25, 1.0, 0.45, 0.3e-6
    errs = []
    for u in (0.2, 0.1, 0.05, 0.025):
        period = u * WL
        o, R, T, Jr_rig = la.rcwa_jones_1d(period, ea * _I3, eb * _I3, 1.5,
                                           1.0, depth, f, WL, n_orders=40)
        eps = rytov_tensor(ea, eb, f, order=0)
        _R, _T, Jr, _Jt = berreman_jones_1d([(eps, depth)], 1.5, 1.0, WL)
        errs.append(np.max(np.abs(Jr - np.asarray(Jr_rig))))
    # strictly decreasing (monotone convergence) and small at the tight end
    assert all(errs[i + 1] < errs[i] for i in range(len(errs) - 1))
    assert errs[-1] < 1e-2


def test_rytov_matches_pmm_oracle_subwavelength():
    # deep sub-wavelength (period = lambda/100): the inherent O(period/lambda)
    # slab interface error is then ~3e-3 (measured), so EMT tracks the
    # rigorous PMM to that level.
    ea, eb, f, depth = 6.25, 1.0, 0.5, 0.25e-6
    period = 0.01 * WL
    o, R, T, Jr_rig = la.pmm_jones_1d(period, ea * _I3, eb * _I3, 1.5, 1.0,
                                      depth, f, WL, degree=20)
    eps = rytov_tensor(ea, eb, f, order=0)
    _R, _T, Jr, _Jt = berreman_jones_1d([(eps, depth)], 1.5, 1.0, WL)
    assert np.max(np.abs(Jr - np.asarray(Jr_rig))) < 5e-3


def test_rytov_segments_reduces_to_binary():
    ea, eb, f = 5.0, 1.5, 0.4
    tb = rytov_tensor(ea, eb, f, order=0)
    ts = rytov_segments_tensor([(f, ea), (1 - f, eb)])
    assert np.max(np.abs(tb - ts)) < 1e-13
    with pytest.raises(ValueError, match="sum to 1"):
        rytov_segments_tensor([(0.4, ea), (0.4, eb)])


def test_rytov_guards():
    with pytest.raises(ValueError, match="fill"):
        rytov_tensor(5.0, 1.0, 1.5)
    with pytest.raises(ValueError, match="order"):
        rytov_tensor(5.0, 1.0, 0.5, order=1)
    with pytest.raises(ValueError, match="period and wavelength"):
        rytov_tensor(5.0, 1.0, 0.5, order=2)


def test_mixing_rules_properties():
    # Maxwell-Garnett end points
    assert abs(maxwell_garnett(2.0, 5.0, 0.0) - 2.0) < 1e-12
    assert abs(maxwell_garnett(2.0, 5.0, 1.0) - 5.0) < 1e-9
    # Bruggeman is symmetric in the two phases at fill 0.5
    assert abs(bruggeman(2.0, 5.0, 0.5) - bruggeman(5.0, 2.0, 0.5)) < 1e-12
    # ... and satisfies its defining self-consistency equation (audit P1-02:
    # symmetry alone also held for a wrong linear coefficient)
    for ea, eb, f, g in [(2.0, 5.0, 0.5, 2.0), (9.0, 2.0, 0.3, 1.0)]:
        geom = "sphere" if g == 2.0 else "cylinder"
        e = bruggeman(ea, eb, f, geometry=geom)
        res = (f * (ea - e) / (ea + g * e)
               + (1 - f) * (eb - e) / (eb + g * e))
        assert abs(res) < 1e-12
    # both stay within the constituent bounds for a real dilute mix
    mg = maxwell_garnett(2.0, 5.0, 0.3, geometry="cylinder")
    assert 2.0 < mg.real < 5.0
    br = bruggeman(2.0, 5.0, 0.3)
    assert 2.0 < br.real < 5.0
    with pytest.raises(ValueError, match="geometry"):
        maxwell_garnett(2.0, 5.0, 0.3, geometry="bogus")


def test_add_effective_grating_bridge():
    from lumenairy.elements.berreman import BerremanStack
    st = BerremanStack(n_substrate=1.5, n_superstrate=1.0)
    st.add_effective_grating(0.3e-6, eps_ridge=6.25, eps_groove=1.0, fill=0.45)
    R, T, J = st.set_source(WL).solve()
    # same as building the tensor by hand
    eps = rytov_tensor(6.25, 1.0, 0.45, order=0)
    R2, T2, J2 = (BerremanStack(n_substrate=1.5, n_superstrate=1.0)
                  .add_layer(0.3e-6, eps=eps).set_source(WL).solve())
    assert np.max(np.abs(np.asarray(J) - np.asarray(J2))) < 1e-14
    # order=2 needs set_source first
    st2 = BerremanStack(n_substrate=1.5)
    with pytest.raises(ValueError, match="set_source"):
        st2.add_effective_grating(0.3e-6, eps_ridge=6.25, eps_groove=1.0,
                                  fill=0.45, period=0.1e-6, order=2)


# =========================================================================== #
# Berreman JAX twin
# =========================================================================== #

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402


def _lc(tilt=35.0, no=1.5, ne=1.75, loss=(0.0, 0.0)):
    th = np.deg2rad(tilt)
    no2, ne2 = no ** 2, ne ** 2
    c, s = np.cos(th), np.sin(th)
    M = np.array([[ne2 * c * c + no2 * s * s, (ne2 - no2) * c * s, 0],
                  [(ne2 - no2) * c * s, ne2 * s * s + no2 * c * c, 0],
                  [0, 0, no2]], complex)
    M[0, 0] += 1j * loss[0]
    M[1, 1] += 1j * loss[1]
    return M


@pytest.mark.parametrize("ang,phi", [(0.0, 0.0), (0.3, 0.0), (0.3, 0.5)])
def test_jax_forward_matches_numpy(ang, phi):
    eps = _lc(loss=(0.05, 0.03))
    Rn, Tn, Jrn, Jtn = berreman_jones_1d([(eps, 180e-9), (2.1, 100e-9)],
                                         1.5, 1.0, WL, angle=ang, phi=phi)
    Rj, Tj, Jrj, Jtj = berreman_jones_1d(
        [(jnp.asarray(eps), jnp.asarray(180e-9)),
         (jnp.asarray(2.1 + 0j), jnp.asarray(100e-9))],
        jnp.asarray(1.5 + 0j), jnp.asarray(1.0 + 0j), jnp.asarray(WL),
        angle=jnp.asarray(ang), phi=jnp.asarray(phi))
    assert np.max(np.abs(Rn - np.asarray(Rj))) < 1e-12
    assert np.max(np.abs(Jrn - np.asarray(Jrj))) < 1e-12
    assert np.max(np.abs(Jtn - np.asarray(Jtj))) < 1e-12


def test_jax_grads_match_fd():
    base = _lc(loss=(0.05, 0.03))

    def _layers(e, t, w, a, n):
        return ([(e, t), (jnp.asarray(2.1 + 0j), jnp.asarray(100e-9))],
                n, jnp.asarray(1.0 + 0j), w, dict(angle=a))

    def loss_eps_re(re):
        e = jnp.asarray(base).at[0, 0].set(re + 0.05j)
        R, T, Jr, Jt = berreman_jones_1d(
            [(e, jnp.asarray(180e-9)), (jnp.asarray(2.1 + 0j),
             jnp.asarray(100e-9))], jnp.asarray(1.5 + 0j),
            jnp.asarray(1.0 + 0j), jnp.asarray(WL), angle=jnp.asarray(0.2))
        return R[0]

    def loss_t(t1):
        R, T, Jr, Jt = berreman_jones_1d(
            [(jnp.asarray(base), t1), (jnp.asarray(2.1 + 0j),
             jnp.asarray(100e-9))], jnp.asarray(1.5 + 0j),
            jnp.asarray(1.0 + 0j), jnp.asarray(WL), angle=jnp.asarray(0.2))
        return R[0]

    def loss_ang(a):
        R, T, Jr, Jt = berreman_jones_1d(
            [(jnp.asarray(base), jnp.asarray(180e-9)),
             (jnp.asarray(2.1 + 0j), jnp.asarray(100e-9))],
            jnp.asarray(1.5 + 0j), jnp.asarray(1.0 + 0j), jnp.asarray(WL),
            angle=a)
        return T[1]

    def fd(f, x, h):
        return (f(x + h) - f(x - h)) / (2 * h)

    for f, x, h in ((loss_eps_re, base[0, 0].real, 1e-5),
                    (loss_t, 180e-9, 1e-12), (loss_ang, 0.2, 1e-6)):
        ad = float(jax.grad(f)(x))
        fdv = fd(lambda v, _f=f: float(_f(jnp.asarray(v))), x, h)
        assert abs(ad - fdv) / max(abs(fdv), 1e-12) < 1e-5, (f.__name__, ad, fdv)


def test_jax_vmap_and_jit():
    base = _lc(loss=(0.05, 0.03))

    def loss_t(t1):
        R, T, Jr, Jt = berreman_jones_1d(
            [(jnp.asarray(base), t1), (jnp.asarray(2.1 + 0j),
             jnp.asarray(100e-9))], jnp.asarray(1.5 + 0j),
            jnp.asarray(1.0 + 0j), jnp.asarray(WL), angle=jnp.asarray(0.2))
        return R[0]
    vals = jnp.linspace(170e-9, 190e-9, 4)
    vb = jax.vmap(loss_t)(vals)
    seq = jnp.stack([loss_t(v) for v in vals])
    assert float(jnp.max(jnp.abs(vb - seq))) < 1e-14
    assert bool(jnp.isfinite(jax.jit(loss_t)(jnp.asarray(180e-9))))


def test_jax_stack_dispatch_and_guard():
    """Traced-stack dispatch parity + the retain_internal contract on the
    JAX path.  STALE-GATE FIX (loose-ends round 2026-07-14): this gate
    originally pinned ``retain_internal=True`` raising for ANY traced stack;
    the v5.21 line SHIPPED differentiable retain_internal for in-plane
    stacks (``_solve_jax_retain``), so the in-plane leg now pins the
    SUPPORTED behavior (internals served, absorption budget closed) and the
    raise-leg moved to the still-unsupported OUT-OF-PLANE tensor.  (This
    gate never ran on CI -- the CI unit jobs have no jax -- so the stale
    pin only failed locally.)"""
    from lumenairy.elements.berreman import BerremanStack
    base = _lc(loss=(0.05, 0.03))
    st = BerremanStack(n_substrate=jnp.asarray(1.5 + 0j), n_superstrate=1.0)
    st.add_layer(jnp.asarray(180e-9), eps=jnp.asarray(base))
    st.add_layer(100e-9, eps=2.1)
    st.set_source(jnp.asarray(WL), theta=jnp.asarray(0.2))
    R, T, J = st.solve()
    Rn, Tn, Jn = (BerremanStack(n_substrate=1.5, n_superstrate=1.0)
                  .add_layer(180e-9, eps=base).add_layer(100e-9, eps=2.1)
                  .set_source(WL, theta=0.2).solve())
    assert np.max(np.abs(np.asarray(J) - Jn)) < 1e-12
    # in-plane traced stack: retain_internal is SUPPORTED (v5.21) -- the
    # retained internals must close the lossy absorption budget.
    R2, T2, _J2 = st.solve(retain_internal=True)
    A = st.layer_absorption()
    budget = np.abs(np.asarray(A).sum(axis=0) + np.asarray(R2)
                    + np.asarray(T2) - 1.0)
    assert float(np.max(budget)) < 1e-10
    # OUT-OF-PLANE tensor on the JAX path: still guarded (any incidence).
    oop = _lc(loss=(0.05, 0.03))
    oop[0, 2] = oop[2, 0] = 0.3
    st_oop = BerremanStack(n_substrate=1.5, n_superstrate=1.0)
    st_oop.add_layer(jnp.asarray(180e-9), eps=jnp.asarray(oop))
    st_oop.set_source(WL, theta=0.2)
    with pytest.raises(NotImplementedError, match="retain_internal"):
        st_oop.solve(retain_internal=True)
