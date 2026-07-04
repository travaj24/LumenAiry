"""PMM/RCWA upgrade program (PMM_RCWA_AUDIT_2026_07_02 + conical audit).

Phase-1 defect regressions (B3 angle reuse, B4/B5 covered by existing suites +
the parity harness) and the bit-exact perf ports (F1/P2 covered by the parity
harness).  Later phases add their own gate files.
"""
import numpy as np
import pytest

import lumenairy as la


# --------------------------------------------------------------------------- #
# B3 -- solve_vs_wavelength reuses a previously set_source()-configured angle   #
# --------------------------------------------------------------------------- #
def _stack():
    st = la.PMM2DStack(period_x=0.6e-6, period_y=0.6e-6, n_substrate=1.5,
                       n_superstrate=1.0, degree=7, n_orders=3)
    st.add_layer(0.25e-6, eps=2.25)
    return st


def test_solve_vs_wavelength_reuses_set_source_angle():
    """After set_source(theta=T), a sweep with no explicit theta must run at T
    (audit B3), not silently reset to normal incidence."""
    wl = 0.55e-6
    # explicit-angle sweep (the intended geometry)
    a = _stack()
    a.set_source(wl, theta=0.3)
    o_e, R_e, T_e = a.solve_vs_wavelength([wl], theta=0.3)[:3]
    # reuse: set_source(theta=0.3) then sweep with NO theta -> must equal explicit
    b = _stack()
    b.set_source(wl, theta=0.3)
    o_r, R_r, T_r = b.solve_vs_wavelength([wl])[:3]
    assert np.array_equal(o_e, o_r)
    assert np.max(np.abs(R_e - R_r)) < 1e-12
    assert np.max(np.abs(T_e - T_r)) < 1e-12
    # and the reused angle genuinely differs from normal incidence
    c = _stack()
    _o0, R0, _T0 = c.solve_vs_wavelength([wl], theta=0.0)[:3]
    assert np.max(np.abs(R_r - R0)) > 1e-6, (
        "the reused oblique angle must change R vs normal incidence")


def test_solve_vs_wavelength_explicit_theta_still_wins():
    """An explicit theta overrides any set_source angle (B3 must not hijack)."""
    wl = 0.55e-6
    st = _stack()
    st.set_source(wl, theta=0.3)          # configure oblique...
    _o, R_norm, _T = st.solve_vs_wavelength([wl], theta=0.0)[:3]  # ...override to normal
    ref = _stack()
    _o2, R_ref, _T2 = ref.solve_vs_wavelength([wl], theta=0.0)[:3]
    assert np.max(np.abs(R_norm - R_ref)) < 1e-12


def test_solve_vs_wavelength_defaults_normal_without_set_source():
    """No set_source() -> the sweep still defaults to normal incidence (the
    pre-B3 behavior is preserved when nothing was configured)."""
    wl = 0.55e-6
    st = _stack()
    _o, R_a, _T = st.solve_vs_wavelength([wl])[:3]
    ref = _stack()
    _o2, R_b, _T2 = ref.solve_vs_wavelength([wl], theta=0.0)[:3]
    assert np.max(np.abs(R_a - R_b)) < 1e-12
