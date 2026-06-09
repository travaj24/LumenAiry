"""Cross-suite naming-alias regression tests (2026-06-08 PMM/RCWA audit, Stage 2).

The 1-D PMM suite historically spells the Rayleigh-order count ``far_field_orders``
while RCWA and the 2-D PMM use ``n_orders``.  Every public 1-D PMM entry point now
accepts ``n_orders`` as a cross-suite alias (override semantics: when supplied it
wins over ``far_field_orders``).  These tests pin the alias to BYTE-identical output
so the convenience can never silently diverge from the canonical keyword.

See docs/audits/PMM_RCWA_AUDIT_2026_06_08.md (naming divergence) and
docs/audits/AUDIT_EXECUTION_PLAN.md (Stage 2a).
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
