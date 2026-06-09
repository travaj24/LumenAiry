"""Cross-suite naming-alias regression tests (2026-06-08 PMM/RCWA audit, Stage 2).

The 1-D PMM suite historically spells the Rayleigh-order count ``far_field_orders``
while RCWA and the 2-D PMM use ``n_orders``.  Every public 1-D PMM entry point now
accepts ``n_orders`` as a cross-suite alias (override semantics: when supplied it
wins over ``far_field_orders``).  These tests pin the alias to BYTE-identical output
so the convenience can never silently diverge from the canonical keyword.

Stage 2b adds the cross-DIMENSION incidence alias: the 1-D entry points spell the
incidence angle ``angle`` (classical planar mount) while the 2-D / conical solvers
and ``RCWAStack`` use ``theta`` (polar) + ``phi`` (azimuth).  Every 1-D entry point
(both suites) now accepts ``theta`` as an alias for ``angle`` (and the two
``set_source`` methods accept each other's spelling), again byte-identically.

See docs/audits/PMM_RCWA_AUDIT_2026_06_08.md (naming divergence) and
docs/audits/AUDIT_EXECUTION_PLAN.md (Stage 2a/2b).
"""
import numpy as np

import lumenairy as la
from lumenairy.elements.pmm import (
    pmm_1d,
    pmm_efficiency_1d,
    pmm_efficiency_1d_segments,
    pmm_jones_1d,
    pmm_jones_1d_segments,
)
from lumenairy.elements.rcwa import rcwa_efficiency_1d, rcwa_jones_1d

_N = 15


def _eye3(v):
    return (v * np.eye(3)).astype(complex)


def test_efficiency_1d_n_orders_byte_identical():
    base = dict(period=0.8e-6, n_ridge=2.0 + 0j, n_groove=1.0 + 0j,
                n_substrate=1.5 + 0j, n_superstrate=1.0 + 0j, depth=0.5e-6,
                duty_cycle=0.5, wavelength=0.633e-6, degree=12)
    o1, R1, T1 = pmm_efficiency_1d(**base, far_field_orders=_N)
    o2, R2, T2 = pmm_efficiency_1d(**base, n_orders=_N)
    assert np.array_equal(o1, o2)
    assert np.array_equal(np.asarray(R1), np.asarray(R2))
    assert np.array_equal(np.asarray(T1), np.asarray(T2))
    # override: n_orders wins over a (deliberately different) far_field_orders
    o3, _, _ = pmm_efficiency_1d(**base, far_field_orders=99, n_orders=_N)
    assert np.array_equal(o1, o3)


def test_jones_1d_n_orders_byte_identical():
    er, eg = _eye3(2.0), _eye3(1.0)
    base = dict(period=0.8e-6, eps_ridge=er, eps_groove=eg, n_substrate=1.5,
                n_superstrate=1.0, depth=0.5e-6, duty_cycle=0.5,
                wavelength=0.633e-6, degree=12, stabilize=False)
    a = pmm_jones_1d(**base, far_field_orders=_N)
    b = pmm_jones_1d(**base, n_orders=_N)
    assert np.array_equal(a[0], b[0])          # orders
    assert np.array_equal(a[3], b[3])          # Jones 2x2


def test_efficiency_segments_n_orders_byte_identical():
    segs = [(0.5, 2.0 + 0j), (0.5, 1.0 + 0j)]
    base = dict(period=0.8e-6, segments=segs, n_substrate=1.5 + 0j,
                n_superstrate=1.0 + 0j, depth=0.5e-6, wavelength=0.633e-6,
                degree=12)
    o1, R1, T1 = pmm_efficiency_1d_segments(**base, far_field_orders=_N)
    o2, R2, T2 = pmm_efficiency_1d_segments(**base, n_orders=_N)
    assert np.array_equal(o1, o2)
    assert np.array_equal(np.asarray(R1), np.asarray(R2))


def test_jones_segments_n_orders_byte_identical():
    segs = [(0.5, _eye3(2.0)), (0.5, _eye3(1.0))]
    base = dict(period=0.8e-6, segments=segs, n_substrate=1.5, n_superstrate=1.0,
                depth=0.5e-6, wavelength=0.633e-6, degree=12, stabilize=False)
    a = pmm_jones_1d_segments(**base, far_field_orders=_N)
    b = pmm_jones_1d_segments(**base, n_orders=_N)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[3], b[3])


def test_pmm_1d_dispatcher_n_orders_byte_identical():
    # pmm_1d is the unified Jones dispatcher -> (orders, R, T, jones) 4-tuple.
    base = dict(period=0.8e-6, eps_ridge=2.0 + 0j, eps_groove=1.0 + 0j,
                n_substrate=1.5 + 0j, n_superstrate=1.0 + 0j, depth=0.5e-6,
                duty_cycle=0.5, wavelength=0.633e-6, degree=12, stabilize=False)
    o1, R1, T1, J1 = pmm_1d(**base, far_field_orders=_N)
    o2, R2, T2, J2 = pmm_1d(**base, n_orders=_N)
    assert np.array_equal(o1, o2)
    assert np.array_equal(np.asarray(R1), np.asarray(R2))
    assert np.array_equal(J1, J2)


def test_pmm_1d_scalar_eps_promoted_to_isotropic_tensor():
    """pmm_1d documents "scalar promoted to an isotropic tensor"; the underlying
    Jones solver requires an explicit (3, 3).  A scalar call must equal the
    explicit ``scalar * eye(3)`` call byte-for-byte (the documented contract was
    previously unmet -- the scalar reached pmm_jones_1d and raised)."""
    base = dict(period=0.8e-6, n_substrate=1.5 + 0j, n_superstrate=1.0 + 0j,
                depth=0.5e-6, duty_cycle=0.5, wavelength=0.633e-6, degree=12,
                stabilize=False, far_field_orders=_N)
    a = pmm_1d(eps_ridge=2.0 + 0j, eps_groove=1.0 + 0j, **base)
    b = pmm_1d(eps_ridge=_eye3(2.0), eps_groove=_eye3(1.0), **base)
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_pmmstack_n_orders_constructor_alias():
    rd, gr = _eye3(4.0), _eye3(2.0)

    def build(**kw):
        st = la.PMMStack(1e-6, n_substrate=1.5, n_superstrate=1.0, degree=14, **kw)
        st.add_layer(0.3e-6, segments=[(0.5, rd), (0.5, gr)])
        return st.set_source(0.633e-6, angle=0.0).solve()

    o1, R1, T1, J1 = build(far_field_orders=_N)
    o2, R2, T2, J2 = build(n_orders=_N)
    assert np.array_equal(o1, o2)
    assert np.array_equal(np.asarray(R1), np.asarray(R2))
    assert np.array_equal(J1, J2)


# ===================================================================== Stage 2b
# Incidence alias: theta <-> angle on 1-D entry points; angle <-> theta on
# RCWAStack.set_source.  Tested at OBLIQUE incidence so the alias actually bites.
_TH = 0.3


def test_pmm_efficiency_1d_theta_aliases_angle():
    base = dict(period=0.8e-6, n_ridge=2.0 + 0j, n_groove=1.0 + 0j,
                n_substrate=1.5 + 0j, n_superstrate=1.0 + 0j, depth=0.5e-6,
                duty_cycle=0.5, wavelength=0.633e-6, degree=12)
    o1, R1, T1 = pmm_efficiency_1d(**base, angle=_TH)
    o2, R2, T2 = pmm_efficiency_1d(**base, theta=_TH)
    assert np.array_equal(np.asarray(R1), np.asarray(R2))
    assert np.array_equal(np.asarray(T1), np.asarray(T2))
    # override: theta wins over a different angle
    _, R3, _ = pmm_efficiency_1d(**base, angle=0.9, theta=_TH)
    assert np.array_equal(np.asarray(R1), np.asarray(R3))


def test_pmm_jones_1d_theta_aliases_angle():
    er, eg = _eye3(2.0), _eye3(1.0)
    base = dict(period=0.8e-6, eps_ridge=er, eps_groove=eg, n_substrate=1.5,
                n_superstrate=1.0, depth=0.5e-6, duty_cycle=0.5,
                wavelength=0.633e-6, degree=12, stabilize=False)
    a = pmm_jones_1d(**base, angle=_TH)
    b = pmm_jones_1d(**base, theta=_TH)
    assert np.array_equal(a[3], b[3])


def test_rcwa_efficiency_1d_theta_aliases_angle():
    base = dict(period=0.8e-6, n_ridge=2.0, n_groove=1.0, n_substrate=1.5,
                n_superstrate=1.0, depth=0.5e-6, duty_cycle=0.5,
                wavelength=0.633e-6, n_orders=11)
    o1, R1, T1 = rcwa_efficiency_1d(**base, angle=_TH)
    o2, R2, T2 = rcwa_efficiency_1d(**base, theta=_TH)
    assert np.array_equal(np.asarray(R1), np.asarray(R2))
    assert np.array_equal(np.asarray(T1), np.asarray(T2))
    _, R3, _ = rcwa_efficiency_1d(**base, angle=0.9, theta=_TH)
    assert np.array_equal(np.asarray(R1), np.asarray(R3))


def test_rcwa_jones_1d_theta_aliases_angle():
    er, eg = _eye3(2.0), _eye3(1.0)
    base = dict(period=0.8e-6, eps_ridge=er, eps_groove=eg, n_substrate=1.5,
                n_superstrate=1.0, depth=0.5e-6, duty_cycle=0.5,
                wavelength=0.633e-6, n_orders=11)
    a = rcwa_jones_1d(**base, angle=_TH)
    b = rcwa_jones_1d(**base, theta=_TH)
    assert np.array_equal(a[3], b[3])


def test_rcwa_stack_set_source_angle_aliases_theta():
    grid = np.where(np.arange(64) < 32, 4.0, 1.0).astype(complex)

    def build(**kw):
        st = la.RCWAStack(1e-6, n_substrate=1.5, n_superstrate=1.0, n_orders=11)
        st.add_layer(0.3e-6, eps_cell=grid)
        return st.set_source(0.633e-6, **kw).solve().jones_reflection()

    a = build(theta=_TH)
    b = build(angle=_TH)
    assert np.array_equal(a, b)
    # override: angle wins over a different theta
    c = build(theta=0.9, angle=_TH)
    assert np.array_equal(a, c)


def test_pmm_stack_set_source_theta_aliases_angle():
    rd, gr = _eye3(4.0), _eye3(2.0)

    def build(**kw):
        st = la.PMMStack(1e-6, n_substrate=1.5, n_superstrate=1.0, degree=14)
        st.add_layer(0.3e-6, segments=[(0.5, rd), (0.5, gr)])
        return st.set_source(0.633e-6, **kw).solve()

    o1, R1, T1, J1 = build(angle=_TH)
    o2, R2, T2, J2 = build(theta=_TH)
    assert np.array_equal(np.asarray(R1), np.asarray(R2))
    assert np.array_equal(J1, J2)
