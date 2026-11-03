"""The degenerate-cluster rule (``lumenairy.backend.jax_cluster_rule``) at
exactly normal incidence ON and NEAR a Rayleigh anomaly -- audit L-10 of
``docs/audits/AUDIT_DYNAMETA_BRIDGE_FINDINGS_2026_11_02.md`` (plan
``PLAN_DYNAMETA_BRIDGE_FINDINGS_2026_11_02.md`` section 3, A3).

The rule makes the theta = 0 angle gradient of a mirror-symmetric grating
right (the +-m orders are an exact cluster there), but its accuracy degrades
as the +-1 orders approach grazing in a half-space, where the consumer
``sqrt(eps - mu)`` of the half-space eig has its branch point (the
``set_jax_cluster_rule`` docstring carries the full measured curve).  Error
measure: |AD dR/dtheta at theta = 0| (truth exactly 0, R(theta) is even)
relative to the slope scale |dR/dtheta| at theta = 1e-2.

Fixture (DynaMeta's F542 grating tuned onto the substrate anomaly; the same
numbers reproduce bit-for-bit through DynaMeta's ``pmm_stack_RT`` bridge):
``PMMStack``, binary layer eps 4 / 1, duty 0.5, 220 nm, n_sup 1.0,
n_sub 1.5, lambda 1.31 um, degree 8, n_orders 11, period
P = (lambda / n_sub) / (1 + delta); figure of merit the order-summed R of
the E_y-incident row (row 1).

Pinned, so a future change to the rule or to the half-space eigenproblem is
visible:

* exactly ON the anomaly (delta = 0) the rule-on theta = 0 gradient is NOT
  FINITE (measured NaN, Windows jax 0.11.0 and WSL jax 0.10.2);
* at delta = 1e-3 the rule-on error stays below 2e-9 (measured 1.87e-10 /
  1.88e-10, Windows / WSL; worst row of the fixture 4.91e-10).

NOT pinned (it costs a second ~12 s first-call trace of the rule-off path),
measured on both builds identically: rule OFF returns a FINITE but wrong
-6.15 on the anomaly and errs 4.8e-5 relative at delta = 1e-3 -- the rule is
still doing work there.

Cost: one first-call JAX warm-up of the routed PMMStack gradient (22-30 s on
a loaded Windows box when this file runs alone, shared with any routed JAX
test run before it), then ~3 s.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumenairy.backend import JAX_AVAILABLE  # noqa: E402

pytestmark = [
    pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed"),
    pytest.mark.filterwarnings("ignore:.*energy closure.*"),
]

_LAM, _NSUB = 1.31e-6, 1.5
_P_RA = _LAM / _NSUB          # the +-1 orders graze in the substrate at theta 0
_ROW = 1                      # E_y incident


@pytest.fixture(autouse=True, scope="module")
def _jax_x64_for_this_module():
    """Hold ``jax_enable_x64`` on for this file and RESTORE it (the JAX x64
    isolation contract: the flag must not leak to the next module)."""
    if not JAX_AVAILABLE:
        yield
        return
    import jax
    previous = bool(getattr(jax.config, "jax_enable_x64", False))
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _R(period, angle, xp):
    from lumenairy.elements.pmm import PMMStack
    st = PMMStack(period, n_substrate=_NSUB, n_superstrate=1.0, degree=8,
                  n_orders=11)
    st.add_layer(220e-9, segments=[(0.5, 4.0), (0.5, 1.0)])
    st.set_source(_LAM, angle=angle)
    _o, R, _T, _J = st.solve()
    return xp.real(xp.sum(R[_ROW]))


def _grad0(period):
    import jax
    import jax.numpy as jnp
    return float(jax.grad(lambda a: _R(period, a, jnp))(jnp.asarray(0.0)))


def test_theta0_gradient_is_not_finite_exactly_on_a_rayleigh_anomaly():
    """Rule on (the default), P = lambda / n_sub exactly (the +-1 orders
    graze in the substrate): the theta = 0 gradient is NaN (measured).  The
    forward value stays finite (R = 0.437299)."""
    from lumenairy.backend import jax_cluster_rule_enabled
    assert jax_cluster_rule_enabled()            # the default
    assert np.isfinite(float(_R(_P_RA, 0.0, np)))
    assert not np.isfinite(_grad0(_P_RA))


def test_theta0_gradient_error_a_relative_1e3_below_a_rayleigh_anomaly():
    """P = (lambda / n_sub) / (1 + 1e-3).  Premise first (cheap, NumPy): R
    is even in theta, so the true theta = 0 gradient is 0 (|R(h) - R(-h)|
    measured 1.2e-14 at h = 1e-6; bar 1e-11).  Slope scale: central FD at
    theta = 1e-2, measured -5.262 (bar: 4 < |slope| < 7).  Rule-on error
    relative to it: measured 1.87e-10 (Windows) / 1.88e-10 (WSL); bar 2e-9
    (x10)."""
    from lumenairy.backend import jax_cluster_rule_enabled
    assert jax_cluster_rule_enabled()            # the default
    P = _P_RA / (1.0 + 1e-3)
    h = 1e-6
    assert abs(_R(P, h, np) - _R(P, -h, np)) < 1e-11
    slope = (_R(P, 1e-2 + h, np) - _R(P, 1e-2 - h, np)) / (2 * h)
    assert 4.0 < abs(slope) < 7.0
    rel = abs(_grad0(P)) / abs(slope)
    assert rel < 2e-9, rel
