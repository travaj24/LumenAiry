"""Conical (out-of-plane, phi != 0) incidence for the 1-D grating via the
PMM2DStack y-invariant bridge -- Path A of AUDIT_PMM_CONICAL_OUT_OF_PLANE.

The published reduction test only covered phi = 0; per the audit this is the
"single gap" before trusting the bridge for exp12's out-of-plane cut.  These
gates validate the separable conical path at phi != 0:

  G_ymom : a y-invariant x-grating puts ZERO power in n_y != 0 orders even at
           phi != 0 (convention-free -- the core separable-in-y check).
  G1     : a uniform slab's zeroth-order reflection-Jones singular values match
           the analytic Berreman 4x4 conical oracle (basis-invariant, so it is
           robust to any s/p-vs-lab-xy convention rotation).
  G_energy : lossless energy closure per incident polarization at phi != 0.
"""
import numpy as np
import pytest

import lumenairy as la

# All v5.20.0 PMM/RCWA/conical tests are eig-heavy 2-D numerics -> run in
# the dedicated slow-tests job (keeps the fast unit gate under its 25-min
# cap, the v5.15.0 design); they still gate every push there.
pytestmark = pytest.mark.slow

_P, _WL, _DEP = 0.6e-6, 0.55e-6, 0.25e-6
_TH, _PHI = np.deg2rad(30.0), np.deg2rad(40.0)


def test_conical_y_invariant_grating_conserves_y_momentum():
    """A y-invariant (x-only) grating at phi != 0 diffracts ONLY into n_y = 0
    orders -- the separable-exact-in-y path carries machine-zero into the
    forbidden n_y != 0 orders (audit G2 gap, now at phi != 0)."""
    from lumenairy.elements.pmm.twod import pmm_efficiency_2d_cell
    S = 6
    cell = np.full((S, S), 1.0 + 0j)
    cell[1:4, :] = 6.25                       # x-grating, uniform along y
    for pol in ("te", "tm"):
        o, R, T = pmm_efficiency_2d_cell(
            _P, _P, cell, 1.5, 1.0, _DEP, _WL, degree=11, n_orders=8,
            theta=_TH, phi=_PHI, polarization=pol)
        side = o[:, 1] != 0                    # any n_y != 0 order
        assert float(R[side].sum() + T[side].sum()) < 1e-12, (
            f"pol={pol}: y-invariant grating leaked power into n_y!=0 at "
            f"phi={_PHI}")
        # energy closes to the x-truncation floor (the honest conical grating
        # convergence, not a bug -- the (m, 0) row still converges in n_orders
        # like the classical mount; TM the wall-normal channel is the slower).
        assert abs(float(R.sum() + T.sum()) - 1.0) < 1.5e-2


def test_conical_uniform_slab_matches_berreman():
    """A uniform isotropic slab through the PMM2DStack conical bridge: the
    zeroth-order reflection-Jones singular values match the Berreman 4x4
    conical oracle to Berreman-grade accuracy (audit G1)."""
    from lumenairy.elements.berreman import berreman_jones_1d
    eps = 2.25
    # PMM2DStack, y-invariant uniform layer, conical source
    st = la.PMM2DStack(period_x=_P, period_y=_P, n_substrate=1.5,
                       n_superstrate=1.0, degree=7, n_orders=3)
    st.add_layer(_DEP, eps=eps)
    o, _R, _T, J = st.set_source(_WL, theta=_TH, phi=_PHI).solve()
    # zeroth (0,0) order reflection Jones
    i0 = int(np.argmin(np.abs(o[:, 0]) + np.abs(o[:, 1])))
    Jpmm = np.asarray(J) if np.ndim(J) == 2 else np.asarray(J)[..., i0]
    # Berreman conical oracle (same uniform slab)
    _Rb, _Tb, Jr, _Jt = berreman_jones_1d(
        [(eps, _DEP)], 1.5, 1.0, _WL, angle=_TH, phi=_PHI)
    # singular values are invariant to the incident-basis (s/p vs lab-xy) choice
    s_pmm = np.sort(np.linalg.svd(Jpmm, compute_uv=False))
    s_ber = np.sort(np.linalg.svd(np.asarray(Jr), compute_uv=False))
    assert np.allclose(s_pmm, s_ber, atol=2e-3), (
        f"conical uniform-slab reflection singular values differ: "
        f"pmm={s_pmm} berreman={s_ber}")


def test_native_conical_matches_berreman_uniform_slab():
    """G1: native pmm_jones_1d_conical on a uniform slab matches the analytic
    Berreman 4x4 conical oracle (singular values, basis-invariant)."""
    import lumenairy as la
    from lumenairy.elements.berreman import berreman_jones_1d
    o, R, T, J = la.pmm_jones_1d_conical(
        _P, 2.25, 2.25, 1.5, 1.0, _DEP, 0.5, _WL, theta=_TH, phi=_PHI,
        degree=9, n_orders=3)
    _Rb, _Tb, Jr, _Jt = berreman_jones_1d([(2.25, _DEP)], 1.5, 1.0, _WL,
                                          angle=_TH, phi=_PHI)
    s_n = np.sort(np.linalg.svd(np.asarray(J), compute_uv=False))
    s_b = np.sort(np.linalg.svd(np.asarray(Jr), compute_uv=False))
    assert np.allclose(s_n, s_b, atol=3e-3), f"native {s_n} vs berreman {s_b}"
    for p in range(2):
        assert abs(float(R[p].sum() + T[p].sum()) - 1.0) < 1e-6  # lossless (BLAS-robust; conical energy floor ~2e-9)


def test_native_conical_reduces_to_classical_at_phi0():
    """G0: at phi=0 (classical mount) TE/TM decouple, so the incident-Ey (TE)
    row must reproduce classical pmm_efficiency_1d(te) -- the reduction to the
    in-plane 1-D solver (TM is the slower wall-normal channel; gated looser)."""
    import lumenairy as la
    th = np.deg2rad(25.0)
    o, R, T, J = la.pmm_jones_1d_conical(
        _P, 6.0, 1.0, 1.5, 1.0, _DEP, 0.5, _WL, theta=th, phi=0.0,
        degree=14, n_orders=6)
    o_te, Rte, Tte = la.pmm_efficiency_1d(
        _P, np.sqrt(6.0), 1.0, 1.5, 1.0, _DEP, 0.5, _WL, angle=th,
        polarization="te", degree=14, far_field_orders=13)
    assert abs(float(R[1].sum()) - float(Rte.sum())) < 5e-3   # incident Ey == TE
    # energy at the honest low-order slow-TM-channel floor (converges with orders)
    for p in range(2):
        assert abs(float(R[p].sum() + T[p].sum()) - 1.0) < 2e-2


def test_native_conical_grating_converges_to_path_a():
    """G2: the native conical grating and the Path A (PMM2DStack y-invariant)
    bridge converge TOGETHER toward the same result as n_orders grows (the
    residual is the honest slow-TM-at-conical floor, not a divergence)."""
    import lumenairy as la
    xw_ridge = 6.0
    diffs = []
    for nq in (8, 12):
        o, R, T, J = la.pmm_jones_1d_conical(
            _P, xw_ridge, 1.0, 1.5, 1.0, _DEP, 0.5, _WL, theta=_TH, phi=_PHI,
            degree=2 * nq - 1, n_orders=nq)
        S = 2 * nq + 2
        cell = np.full((S, S), 1.0 + 0j)
        cell[:S // 2, :] = xw_ridge
        st = la.PMM2DStack(period_x=_P, period_y=_P, n_substrate=1.5,
                           n_superstrate=1.0, degree=2 * nq - 1, n_orders=nq)
        st.add_layer(_DEP, eps_cell=cell)
        _o2, _R2, _T2, J2 = st.set_source(_WL, theta=_TH, phi=_PHI).solve()
        s_n = np.sort(np.linalg.svd(np.asarray(J), compute_uv=False))
        s_a = np.sort(np.linalg.svd(np.asarray(J2), compute_uv=False))
        diffs.append(float(np.max(np.abs(s_n - s_a))))
    assert diffs[1] < diffs[0], "native and Path A must converge together"
    assert diffs[1] < 8e-3


def test_conical_uniform_slab_energy_closes():
    """Lossless conical slab: R + T = 1 per incident polarization at phi != 0."""
    st = la.PMM2DStack(period_x=_P, period_y=_P, n_substrate=1.5,
                       n_superstrate=1.0, degree=7, n_orders=3)
    st.add_layer(_DEP, eps=2.25)
    _o, R, T, _J = st.set_source(_WL, theta=_TH, phi=_PHI).solve()
    R = np.asarray(R)
    T = np.asarray(T)
    for p in range(R.shape[0]):
        assert abs(float(R[p].sum() + T[p].sum()) - 1.0) < 5e-3


# ===========================================================================
# AUDIT_PMM_CONICAL_PATTERNED_TENSOR_BUG_2026_07_12 regression gates: the
# PATTERNED degenerate-limit reduction the v5.20.0 suite was missing (its
# reduction test used a uniform-tensor / scalar cell, so the projected-path
# systematic error shipped).  Fixed by the pure-nodal conical cascade
# (_conical_nodal_solve): classical machinery generalized to ky0.
# ===========================================================================

def _lc_tensor(angle_deg, no=1.5, ne=1.7):
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    no2, ne2 = no * no, ne * ne
    T = np.diag([no2, no2, no2]).astype(complex)
    T[0, 0] = ne2 * c * c + no2 * s * s
    T[1, 1] = ne2 * s * s + no2 * c * c
    T[0, 1] = T[1, 0] = (ne2 - no2) * c * s
    return T


def _stack_jones(theta, phi, segments, degree=8, ffo=15):
    st = la.PMMStack(700e-9, n_substrate=1.5, n_superstrate=1.0,
                     degree=degree, grade=True, far_field_orders=ffo)
    st.add_layer(300e-9, segments=segments)
    return np.asarray(
        st.set_source(1310e-9, theta=theta, phi=phi).solve()[3])


def test_conical_patterned_tensor_reduces_to_classical_at_ky0():
    """THE missing gate: a PATTERNED anisotropic layer at theta=0, phi=90
    (ky0 == 0) must equal the classical phi=0 solve to machine precision.
    Pre-fix the projected route left a ~3.2e-3 resolution-INDEPENDENT gap
    (~3.5 deg retardance error)."""
    segs = [(0.5, _lc_tensor(30.0)), (0.5, 1.0 + 0j)]
    Jcl = _stack_jones(0.0, 0.0, segs)
    Jco = _stack_jones(0.0, np.deg2rad(90.0), segs)
    assert np.linalg.norm(Jco - Jcl) < 1e-10


def test_conical_patterned_scalar_reduces_to_classical_at_ky0():
    """The SCALAR patterned degenerate limit (the audit under-scoped this:
    the projected-path defect hit scalar gratings too -- the old suite's 5e-3
    'slow-TM-channel' tolerance masked it)."""
    segs = [(0.5, 2.89 + 0j), (0.5, 1.0 + 0j)]
    Jcl = _stack_jones(0.0, 0.0, segs)
    Jco = _stack_jones(0.0, np.deg2rad(90.0), segs)
    assert np.linalg.norm(Jco - Jcl) < 1e-10


def test_conical_patterned_tensor_retardance_tracks_director():
    """Retardance gate: sweep the LC director; the conical (ky0=0) reflection
    retardance arg(J00) - arg(J11) must track the classical value to well
    under 0.1 deg (pre-fix: a fixed ~3.5 deg offset)."""
    for ang in (15.0, 30.0, 60.0):
        segs = [(0.5, _lc_tensor(ang)), (0.5, 1.0 + 0j)]
        Jcl = _stack_jones(0.0, 0.0, segs)
        Jco = _stack_jones(0.0, np.deg2rad(90.0), segs)
        ret_cl = np.degrees(np.angle(Jcl[0, 0]) - np.angle(Jcl[1, 1]))
        ret_co = np.degrees(np.angle(Jco[0, 0]) - np.angle(Jco[1, 1]))
        assert abs(ret_co - ret_cl) < 0.01, (
            f"director {ang} deg: retardance classical {ret_cl:.3f} vs "
            f"conical {ret_co:.3f}")


def test_conical_patterned_single_layer_entries_reduce_at_ky0():
    """The single-layer public entries take the same nodal route: scalar
    (pmm_jones_1d_conical) and tensor (pmm_jones_1d_conical_tensor) patterned
    cells at theta=0 must agree with the PMMStack classical solve."""
    P, WL, DEP = 700e-9, 1310e-9, 300e-9
    # scalar
    Jcl = _stack_jones(0.0, 0.0, [(0.5, 2.89 + 0j), (0.5, 1.0 + 0j)])
    _o, _R, _T, Js = la.pmm_jones_1d_conical(
        P, 2.89, 1.0, 1.5, 1.0, DEP, 0.5, WL, theta=0.0,
        phi=np.deg2rad(90.0), degree=8, grade=True, n_orders=7)
    assert np.linalg.norm(np.asarray(Js) - Jcl) < 1e-10
    # tensor
    Jcl_t = _stack_jones(0.0, 0.0, [(0.5, _lc_tensor(30.0)), (0.5, 1.0 + 0j)])
    S = 64
    cell = np.zeros((S, 3, 3), complex)
    cell[:] = np.eye(3)
    cell[:S // 2] = _lc_tensor(30.0)
    _o, _R, _T, Jt = la.pmm_jones_1d_conical_tensor(
        P, cell, 1.5, 1.0, DEP, WL, theta=0.0, phi=np.deg2rad(90.0),
        degree=8, grade=True, n_orders=7)
    assert np.linalg.norm(np.asarray(Jt) - Jcl_t) < 1e-10


def test_conical_patterned_energy_and_theta_continuity():
    """Physics gates at GENUINE conical: lossless energy closure to 1e-8 per
    incident polarization, and the theta -> 0 limit approaches the classical
    solve quadratically (no jump at the dispatch boundary)."""
    segs = [(0.5, _lc_tensor(30.0)), (0.5, 1.0 + 0j)]
    st = la.PMMStack(700e-9, n_substrate=1.5, n_superstrate=1.0, degree=10,
                     grade=True, far_field_orders=21)
    st.add_layer(300e-9, segments=segs)
    _o, R, T, _J = st.set_source(
        1310e-9, theta=np.deg2rad(25.0), phi=np.deg2rad(35.0)).solve()
    R, T = np.asarray(R), np.asarray(T)
    for p in range(2):
        assert abs(float(R[p].sum() + T[p].sum()) - 1.0) < 1e-8
    Jcl = _stack_jones(0.0, 0.0, segs, degree=10, ffo=21)
    Jsm = _stack_jones(1e-3, np.deg2rad(90.0), segs, degree=10, ffo=21)
    assert np.linalg.norm(Jsm - Jcl) < 1e-5      # ~theta^2 continuity


def test_conical_patterned_offplane_tensor_raises():
    """A PATTERNED cell with out-of-plane tensor coupling at conical incidence
    now fails LOUD (the old projected route returned silently-wrong
    retardance for it)."""
    tilt = _lc_tensor(30.0)
    tilt[0, 2] = tilt[2, 0] = 0.05          # xz coupling (tilted director)
    st = la.PMMStack(700e-9, n_substrate=1.5, n_superstrate=1.0, degree=8,
                     far_field_orders=15)
    st.add_layer(300e-9, segments=[(0.5, tilt), (0.5, 1.0 + 0j)])
    with pytest.raises(NotImplementedError, match="out-of-plane"):
        st.set_source(1310e-9, theta=np.deg2rad(10.0),
                      phi=np.deg2rad(45.0)).solve()


def test_conical_patterned_cross_oracle_rcwa_at_phi():
    """Converged cross-oracle at GENUINE conical (theta=25, phi=35): the
    independent rcwa_jones_2d (y-invariant cell) must CONVERGE TOWARD the
    nodal PMM answer as its order count rises (the residual is RCWA's own
    truncation, shrinking with n_orders -- pre-fix the PMM answer itself was
    off and the trend stalled)."""
    from lumenairy.elements.rcwa.twod import rcwa_jones_2d
    P, WL, DEP = 700e-9, 1310e-9, 300e-9
    th, ph = np.deg2rad(25.0), np.deg2rad(35.0)
    segs = [(0.5, _lc_tensor(30.0)), (0.5, 1.0 + 0j)]
    st = la.PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=14,
                     grade=True, far_field_orders=25)
    st.add_layer(DEP, segments=segs)
    Jp = np.asarray(st.set_source(WL, theta=th, phi=ph).solve()[3])
    S, SY = 64, 8
    cell = np.zeros((S, SY, 3, 3), complex)
    cell[:, :] = np.eye(3)
    cell[:S // 2, :] = _lc_tensor(30.0)
    dif = []
    for no in (7, 11):
        _o, _R, _T, Jr = rcwa_jones_2d(P, P, cell, 1.5, 1.0, DEP, WL,
                                       theta=th, phi=ph, n_orders_x=no,
                                       n_orders_y=1)
        sp = np.sort(np.linalg.svd(Jp, compute_uv=False))
        sr = np.sort(np.linalg.svd(np.asarray(Jr), compute_uv=False))
        dif.append(float(np.max(np.abs(sp - sr))))
    assert dif[1] < dif[0], "rcwa must converge toward the nodal PMM answer"
    assert dif[1] < 8e-3


def test_conical_mixed_multilayer_reduces_at_ky0():
    """MIXED uniform + patterned multilayer (the exp11-like shape) through the
    nodal conical cascade: the ky0=0 degenerate limit must match the classical
    solve across R, T, and Jones (uniform layers ride the same union nodal
    grid -- exact for constants)."""
    def _solve(theta, phi):
        st = la.PMMStack(700e-9, n_substrate=1.5, n_superstrate=1.0,
                         degree=10, grade=True, far_field_orders=21)
        st.add_layer(120e-9, eps=2.25 + 0j)
        st.add_layer(300e-9, segments=[(0.5, _lc_tensor(30.0)),
                                       (0.5, 1.0 + 0j)])
        st.add_layer(80e-9, segments=[(0.3, 4.0 + 0j), (0.7, 1.0 + 0j)])
        st.add_layer(150e-9, eps=1.9 + 0j)
        _o, R, T, J = st.set_source(1310e-9, theta=theta, phi=phi).solve()
        return np.asarray(R), np.asarray(T), np.asarray(J)

    Rcl, Tcl, Jcl = _solve(0.0, 0.0)
    Rco, Tco, Jco = _solve(0.0, np.deg2rad(90.0))
    assert np.linalg.norm(Jco - Jcl) < 1e-10
    assert np.max(np.abs(Rco - Rcl)) < 1e-10
    assert np.max(np.abs(Tco - Tcl)) < 1e-10
