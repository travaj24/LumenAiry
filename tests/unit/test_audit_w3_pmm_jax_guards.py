"""Audit wave-3 (2026-07): PMM JAX twins mirror their NumPy siblings' guards.

Findings covered (docs/audits/AUDIT_V5_17_0_2026_07_01_DEEP.md):

* P2-10 -- ``_pmm_stack_solve_jax`` had no grazing/evanescent-incidence guard
  (divides by ``kz_inc`` -> NaN R/T where the NumPy stack raises).
* P2-11 -- ``_scalar_jax_tail`` (the 2-D pillar/cell twins) skipped the
  propagating-incidence raise AND the Wood-anomaly wavelength nudge that both
  the NumPy 2-D core and the sibling ``_jax_stack2d`` apply for concrete
  inputs.
* P2-14 -- ``pmm_efficiency_2d``'s JAX dispatch returned before
  ``_validate_pillar_bounds`` (inverted/degenerate bounds silently gave an
  energy-conserving but WRONG answer).
* wave-1 verifier bonus -- the 1-D jax twins (``_pmm_efficiency_1d_jax`` /
  ``_pmm_jones_1d_jax``) carried no incidence-medium guard while the NumPy
  path raises (wave-1 P1-03 mirror).

The guards are HOST-SIDE and act on CONCRETE values only (a Tracer skips
them), so the helper functions are testable without jax; jax-runtime
integration tests importorskip('jax').
"""
import numpy as np
import pytest

from lumenairy.elements.pmm import (
    PMMStack,
    pmm_efficiency_1d,
    pmm_efficiency_2d,
    pmm_jones_1d,
)
from lumenairy.elements.pmm._core import _jpmm_concrete_incidence_guard
from lumenairy.elements.pmm._jax_stack import _grazing_guard_concrete
from lumenairy.elements.pmm._jax_twod import (
    _host_incidence_guard,
    _static_prep,
)
from lumenairy.elements.rcwa._core import _grazing_safe_wavelength


class _FakeTracer:
    """Non-materialisable stand-in for a jax Tracer: any host concretization
    (``complex()`` / ``float()``) raises, exactly like an abstract Tracer."""

    def __complex__(self):
        raise TypeError("abstract tracer cannot be materialised")

    def __float__(self):
        raise TypeError("abstract tracer cannot be materialised")


def _tiny_stack():
    st = PMMStack(0.8e-6, n_substrate=1.5, n_superstrate=1.0, degree=10)
    st.add_layer(0.2e-6, eps=2.1)
    st.add_layer(0.3e-6, segments=[(0.5, 4.0 + 0j), (0.5, 1.0 + 0j)])
    return st


# --------------------------------------------------------------------------
# [P2-10] stack twin grazing guard (host-side helper, no jax needed)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("angle", [np.pi / 2, np.pi / 2 - 5e-10])
def test_stack_jax_grazing_guard_raises(angle):
    with pytest.raises(ValueError, match="grazing/evanescent incidence"):
        _grazing_guard_concrete(1.0, angle)


def test_stack_grazing_and_frontside_guards_both_reject():
    # v5.24.4 (audit S1-7): the numpy stack's set_source now applies the
    # front-side GEOMETRY guard (|angle| < pi/2) BEFORE .solve() -- so a pi/2
    # grazing angle is rejected there (previously it silently aliased to the
    # supplementary front-side angle).  The solve-level grazing/evanescent
    # PHYSICS guard (_grazing_guard_concrete) is a SEPARATE sibling that the jax
    # twin's concrete path uses.  Both correctly reject a grazing pi/2
    # incidence, each with its own layer-appropriate message.
    with pytest.raises(ValueError, match="front-side illumination") as exc_np:
        _tiny_stack().set_source(1.55e-6, angle=np.pi / 2).solve()
    with pytest.raises(ValueError, match="grazing/evanescent incidence") as exc_tw:
        _grazing_guard_concrete(1.0, np.pi / 2)
    assert isinstance(exc_np.value, ValueError)
    assert isinstance(exc_tw.value, ValueError)


def test_stack_jax_grazing_guard_valid_and_tracer_skip():
    _grazing_guard_concrete(1.0, 0.2)              # valid: silent
    _grazing_guard_concrete(_FakeTracer(), np.pi / 2)   # traced n_sup: skip
    _grazing_guard_concrete(1.0, _FakeTracer())         # traced angle: skip


# --------------------------------------------------------------------------
# [P2-11] 2-D twin host guards (propagating incidence + Wood nudge)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _st2d():
    return _static_prep(1e-6, 1e-6, 0.3e-6, 0.6e-6, 0.2e-6, 0.7e-6,
                        5, 1, True, 2)


def test_twod_host_guard_nonpropagating_raises(_st2d):
    with pytest.raises(ValueError, match="non-propagating"):
        _host_incidence_guard("pmm_efficiency_2d", _st2d, 1e-6, 1e-6,
                              0.8e-6, np.pi / 2, 0.0, 1.0, 1.5,
                              (12.0 + 0j, 2.25))


def test_twod_host_guard_wood_nudge_matches_numpy(_st2d):
    wl_g = _host_incidence_guard("pmm_efficiency_2d", _st2d, 1e-6, 1e-6,
                                 1.0e-6, 0.0, 0.0, 1.0, 1.5,
                                 (12.0 + 0j, 2.25))
    eps_reals = [1.0 + 0j, (1.5 + 0j) ** 2, 12.0 + 0j, 2.25 + 0j]
    wl_ref = _grazing_safe_wavelength(1.0e-6, 0.0, 0.0, _st2d["order_x"],
                                      _st2d["order_y"], 1e-6, 1e-6,
                                      eps_reals)
    assert wl_g == wl_ref            # the numpy core's exact nudge
    assert wl_g != 1.0e-6            # and it DID nudge off the anomaly


def test_twod_host_guard_off_anomaly_identity(_st2d):
    wl = _host_incidence_guard("pmm_efficiency_2d", _st2d, 1e-6, 1e-6,
                               0.8e-6, 0.1, 0.2, 1.0, 1.5,
                               (12.0 + 0j, 2.25))
    assert wl == 0.8e-6


def test_twod_host_guard_tracer_skip(_st2d):
    tr = _FakeTracer()
    # traced wavelength: the SAME object passes through untouched, even at
    # otherwise-grazing angles (the documented concrete-only scope)
    assert _host_incidence_guard("pmm_efficiency_2d", _st2d, 1e-6, 1e-6,
                                 tr, np.pi / 2, 0.0, 1.0, 1.5,
                                 (12.0 + 0j,)) is tr
    # traced theta: skip too
    assert _host_incidence_guard("pmm_efficiency_2d", _st2d, 1e-6, 1e-6,
                                 0.8e-6, _FakeTracer(), 0.0, 1.0, 1.5,
                                 (12.0 + 0j,)) == 0.8e-6


# --------------------------------------------------------------------------
# [P2-14] pillar bounds validated BEFORE the JAX dispatch
# --------------------------------------------------------------------------

@pytest.mark.parametrize("xb", [(0.6e-6, 0.3e-6), (0.0, 0.6e-6),
                                (0.3e-6, 1.0e-6)])
def test_pillar_bounds_still_raise_on_numpy_path(xb):
    with pytest.raises(ValueError, match="x_bounds must satisfy"):
        pmm_efficiency_2d(1e-6, 1e-6, 12.0 + 0j, 2.25, xb,
                          (0.2e-6, 0.7e-6), 1.5, 1.0, 0.3e-6, 0.8e-6,
                          degree=5, n_orders=2, polarization="tm")


def test_pillar_bounds_check_precedes_jax_dispatch_in_source():
    # Structural pin (works without jax): the hoisted _validate_pillar_bounds
    # call must appear BEFORE the _jax_twod dispatch inside pmm_efficiency_2d.
    import inspect

    src = inspect.getsource(pmm_efficiency_2d)
    i_val = src.index("_validate_pillar_bounds(")
    i_jax = src.index("_pmm_efficiency_2d_jax")
    assert i_val < i_jax


# --------------------------------------------------------------------------
# [bonus] 1-D jax twins: concrete-only incidence-medium guard (wave-1 mirror)
# --------------------------------------------------------------------------

def test_1d_jax_guard_gain_matches_numpy_message():
    with pytest.raises(ValueError) as exc_np:
        pmm_efficiency_1d(2e-6, 2.0, 1.0, 1.5, 1.0 - 1e-6j, 0.3e-6, 0.5,
                          0.6e-6, polarization="te", degree=10,
                          stabilize=False)
    with pytest.raises(ValueError) as exc_tw:
        _jpmm_concrete_incidence_guard("pmm_efficiency_1d", 1.0 - 1e-6j, 0.0)
    assert str(exc_tw.value) == str(exc_np.value)
    assert "gain incidence medium" in str(exc_tw.value)


def test_1d_jax_guard_grazing_raises():
    with pytest.raises(ValueError, match="non-propagating"):
        _jpmm_concrete_incidence_guard("pmm_jones_1d", 1.0, np.pi / 2)


def test_1d_jax_guard_valid_and_tracer_skip():
    _jpmm_concrete_incidence_guard("pmm_efficiency_1d", 1.0, 0.25)
    _jpmm_concrete_incidence_guard("pmm_efficiency_1d", _FakeTracer(), 0.0)
    _jpmm_concrete_incidence_guard("pmm_efficiency_1d", 1.0, _FakeTracer())


def test_numpy_paths_still_solve_clean():
    # cheap non-regression: valid numpy solves are untouched by the hoists
    o, R, T = pmm_efficiency_1d(2e-6, 2.0, 1.0, 1.5, 1.0, 0.3e-6, 0.5,
                                0.6e-6, polarization="tm", degree=10,
                                angle=0.25, stabilize=False)
    assert abs(R.sum() + T.sum() - 1.0) < 2e-3
    o, R, T, J = pmm_jones_1d(2e-6, np.diag([4.0, 4.1, 4.2]).astype(complex),
                              np.eye(3, dtype=complex), 1.5, 1.0, 0.3e-6,
                              0.5, 0.6e-6, degree=10, angle=0.15,
                              stabilize=False)
    assert abs(R[0].sum() + T[0].sum() - 1.0) < 2e-3
    os_, Rs, Ts, Js = _tiny_stack().set_source(1.55e-6, angle=0.2).solve()
    assert abs(Rs[0].sum() + Ts[0].sum() - 1.0) < 1e-9


# --------------------------------------------------------------------------
# jax-runtime integration (skips where jax is absent; runs on the CI boxes)
# --------------------------------------------------------------------------

def _jnp():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return jax.numpy


def test_jax_pillar_inverted_bounds_raise():
    jnp = _jnp()
    with pytest.raises(ValueError, match="x_bounds must satisfy"):
        pmm_efficiency_2d(1e-6, 1e-6, jnp.asarray(12.0 + 0j), 2.25,
                          (0.6e-6, 0.3e-6), (0.2e-6, 0.7e-6), 1.5, 1.0,
                          0.3e-6, 0.8e-6, degree=5, n_orders=2,
                          polarization="tm")


def test_jax_pillar_grazing_raises():
    jnp = _jnp()
    with pytest.raises(ValueError, match="non-propagating"):
        pmm_efficiency_2d(1e-6, 1e-6, jnp.asarray(12.0 + 0j), 2.25,
                          (0.3e-6, 0.6e-6), (0.2e-6, 0.7e-6), 1.5, 1.0,
                          0.3e-6, 0.8e-6, degree=5, n_orders=2,
                          polarization="tm", theta=np.pi / 2)


def test_jax_pillar_wood_anomaly_parity_with_numpy():
    jnp = _jnp()
    args = (1e-6, 1e-6)
    kw = dict(degree=5, n_orders=2, polarization="tm")
    o_n, R_n, T_n = pmm_efficiency_2d(*args, 12.0 + 0j, 2.25,
                                      (0.3e-6, 0.6e-6), (0.2e-6, 0.7e-6),
                                      1.5, 1.0, 0.3e-6, 1.0e-6, **kw)
    o_j, R_j, T_j = pmm_efficiency_2d(*args, jnp.asarray(12.0 + 0j), 2.25,
                                      (0.3e-6, 0.6e-6), (0.2e-6, 0.7e-6),
                                      1.5, 1.0, 0.3e-6, 1.0e-6, **kw)
    # both sides now solve the SAME nudged wavelength (was ~1e-5 apart)
    assert float(np.max(np.abs(np.asarray(R_j) - R_n))) < 1e-10
    assert float(np.max(np.abs(np.asarray(T_j) - T_n))) < 1e-10


def test_jax_stack_grazing_raises():
    jnp = _jnp()
    st = PMMStack(0.8e-6, n_substrate=1.5, n_superstrate=1.0, degree=10)
    st.add_layer(0.2e-6, eps=2.1)
    st.add_layer(0.3e-6, segments=[(0.5, jnp.asarray(4.0 + 0j)),
                                   (0.5, 1.0 + 0j)])
    # v5.24.4 (audit S1-7): set_source now applies the front-side geometry
    # guard (|angle| < pi/2) FIRST, so a pi/2 grazing angle is rejected there
    # (front-side message) before it can reach the solve's grazing/evanescent
    # physics guard.  Both are valid rejections of grazing incidence.
    with pytest.raises(ValueError,
                       match="grazing/evanescent incidence|front-side illumination"):
        st.set_source(1.55e-6, angle=np.pi / 2).solve()


def test_jax_1d_gain_superstrate_raises():
    jnp = _jnp()
    with pytest.raises(ValueError, match="gain incidence medium"):
        pmm_efficiency_1d(2e-6, jnp.asarray(2.0 + 0j), 1.0, 1.5,
                          1.0 - 1e-6j, 0.3e-6, 0.5, 0.6e-6,
                          polarization="te", degree=10, stabilize=False)


def test_jax_1d_traced_superstrate_skips_guard_and_grads():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy

    def loss(nsup):
        o, R, T = pmm_efficiency_1d(2e-6, 2.0, 1.0, 1.5, nsup, 0.3e-6, 0.5,
                                    0.6e-6, polarization="te", degree=8,
                                    stabilize=False)
        return jnp.sum(R)

    g = jax.grad(loss)(jnp.asarray(1.2))   # traced n_sup: guard skipped
    assert np.isfinite(float(g))
