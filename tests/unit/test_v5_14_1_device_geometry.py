"""v5.14.1 -- device-geometry roadmap (docs/audits/ROADMAP_DEVICE_GEOMETRY_
SWEEPS_2026_06_10.md): builders, geometry algebra, staircase robustness,
sweeps, PMM absorption, prepared material slots, viewers + the RCWAStack
out-of-plane promotion.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements.pmm import PMM2DStack, PMMStack
from lumenairy.elements.pmm._core import _pmm_union_grid
from lumenairy.elements.rcwa import RCWAStack, rcwa_jones_2d
from lumenairy.elements.rcwa._core import uniaxial_tensor
from lumenairy.elements.segment_geometry import SegmentStackGeometry

_P, _WL = 0.8e-6, 0.633e-6


# =========================================================================== #
# item 1 -- multi-feature tapered builders (center-anchored)
# =========================================================================== #

def test_pmm_tapered_ridges_single_equals_legacy():
    a = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=10)
    a.add_tapered_grating(0.3e-6, eps_ridge=4.0, eps_groove=1.0,
                          duty_bottom=0.6, duty_top=0.4, n_slices=4)
    b = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=10)
    b.add_tapered_ridges(0.3e-6, ridges=[(0.5 * _P, 0.4 * _P, 0.6 * _P, 4.0)],
                         eps_groove=1.0, n_slices=4)
    oa, Ra, Ta, Ja = a.set_source(_WL).solve()
    ob, Rb, Tb, Jb = b.set_source(_WL).solve()
    assert np.array_equal(Ra, Rb) and np.array_equal(Ja, Jb)


def test_pmm_tapered_ridges_centers_do_not_drift():
    """The audited geometry bug: a left-anchored taper drifts each ridge's
    center as it narrows (~3% of out-coupling).  Each slice's ridge centers
    must be exactly z-independent."""
    st = PMMStack(0.55e-6, n_substrate=1.5, degree=10)
    st.add_tapered_ridges(0.35e-6,
                          ridges=[(0.13e-6, 0.110e-6, 0.130e-6, 4.0),
                                  (0.40e-6, 0.200e-6, 0.220e-6, 6.0)],
                          eps_groove=1.0, n_slices=5)
    for _t, segs, _sl in st._layers:
        cw = np.concatenate([[0.0], np.cumsum([w for w, _ in segs])])
        centers = [0.5 * (cw[i] + cw[i + 1]) * 0.55e-6
                   for i, (w, e) in enumerate(segs)
                   if abs(complex(np.asarray(e)[0, 0]) - 1.0) > 1e-9]
        assert np.allclose(centers, [0.13e-6, 0.40e-6], atol=1e-20)


def test_pmm_tapered_ridges_overlap_and_wrap():
    with pytest.raises(ValueError, match="overlap"):
        PMMStack(_P, degree=8).add_tapered_ridges(
            0.1e-6, ridges=[(0.4 * _P, 0.3 * _P, 0.3 * _P, 4.0),
                            (0.5 * _P, 0.3 * _P, 0.3 * _P, 6.0)],
            eps_groove=1.0, n_slices=1)
    w = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=10)
    w.add_tapered_ridges(0.2e-6, ridges=[(0.0, 0.3 * _P, 0.3 * _P, 4.0)],
                         eps_groove=1.0, n_slices=2)     # wraps the edge
    o, R, T, _ = w.set_source(_WL).solve()
    assert abs(float(R[0].sum() + T[0].sum()) - 1.0) < 1e-7


def test_rcwa_tapered_ridges_single_equals_legacy():
    a = RCWAStack(_P, n_superstrate=1.0, n_substrate=1.5, n_orders=6)
    a.add_tapered_grating(0.3e-6, eps_ridge=4.0, eps_groove=1.0,
                          duty_bottom=0.6, duty_top=0.4, n_slices=4)
    b = RCWAStack(_P, n_superstrate=1.0, n_substrate=1.5, n_orders=6)
    b.add_tapered_ridges(0.3e-6, ridges=[(0.5 * _P, 0.4 * _P, 0.6 * _P, 4.0)],
                         eps_groove=1.0, n_slices=4)
    o1, R1, T1 = a.set_source(_WL).solve().efficiencies()
    o2, R2, T2 = b.set_source(_WL).solve().efficiencies()
    assert np.array_equal(np.asarray(R1), np.asarray(R2))


def test_rcwa_tapered_pillars_2d():
    st = RCWAStack(_P, period_y=_P, n_superstrate=1.0, n_substrate=1.5,
                   n_orders=3, n_orders_y=3)
    st.add_tapered_pillars(0.2e-6, pillars=[
        ((0.3 * _P, 0.3 * _P), (0.2 * _P, 0.2 * _P), (0.3 * _P, 0.3 * _P),
         6.0),
        ((0.7 * _P, 0.7 * _P), (0.15 * _P, 0.15 * _P), (0.2 * _P, 0.2 * _P),
         4.0)], eps_host=1.0, n_slices=3)
    o, R, T = st.set_source(_WL).solve().efficiencies()
    assert abs(float(np.sum(np.asarray(R)) + np.sum(np.asarray(T))) - 2.0) \
        < 1e-9


def test_pmm2d_tapered_pillars_single_equals_legacy():
    pa = PMM2DStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=9,
                    n_orders=4)
    pa.add_tapered_pillar(0.24e-6, eps_pillar=6.0, eps_host=1.0,
                          x_bounds_bottom=(0.15 * _P, 0.65 * _P),
                          y_bounds_bottom=(0.15 * _P, 0.65 * _P), n_slices=1)
    pb = PMM2DStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=9,
                    n_orders=4)
    pb.add_tapered_pillars(0.24e-6, pillars=[
        ((0.4 * _P, 0.4 * _P), (0.5 * _P, 0.5 * _P), (0.5 * _P, 0.5 * _P),
         6.0)], eps_host=1.0, n_slices=1)
    oa, Ra, Ta, _ = pa.set_source(_WL).solve()
    ob, Rb, Tb, _ = pb.set_source(_WL).solve()
    assert np.max(np.abs(Ra - Rb)) < 1e-12


# =========================================================================== #
# item 3 -- staircase robustness
# =========================================================================== #

def test_union_grid_cross_layer_snap_and_own_layer_preserved():
    segsA = [(0.30, 4.0), (0.70, 1.0)]
    segsB = [(0.300001, 4.0), (0.699999, 1.0)]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        uw, _ = _pmm_union_grid([segsA, segsB], 1e-5)
    assert len(uw) == 2                       # snapped to one wall
    assert any("snapped" in str(x.message) for x in w)
    # a 1e-6-wide liner WITHIN one layer is that layer's own feature: kept
    segsC = [(0.3, 4.0), (1e-6, 9.0), (0.699999, 1.0)]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        uw2, _ = _pmm_union_grid([segsC], 1e-5)
    assert len(uw2) == 3 and not w


def test_stack_slices_consensus():
    """RECONCILED 2026-08-01 (union-grid audit 2026-07-28, R-1).

    This pin used to assert that a stack with NO recorded taper builder got
    a ``"...the consensus check was skipped."`` warning -- the old
    behaviour, where the ENTIRE recipe-free route (hand-added layers and
    every ``SegmentStackGeometry``-built device, i.e. the documented device
    route) was silently unprotected against the passive-but-wrong staircase
    pathology.  ``PMMStack._slices_consensus_check`` now falls back to
    ``_union_grid_consensus_check``, which needs no recipe: it re-solves
    with ``min_feature`` perturbed, so the guard is finally REACHABLE
    there.  That is strictly more coverage than the warning it replaced, so
    the pin is updated to the stronger contract rather than the code
    reverted.

    Fail-before witness: on the pre-fix tree the second block below emits
    the "skipped" warning and ``assert not skipped`` fails; on this tree it
    emits NOTHING (measured: 0 warnings), because a uniform layer has no
    cross-layer walls to snap and therefore scores exactly 0 -- a clean
    stack cannot false-positive.  The same contract is pinned from the
    covariant side in ``test_audit_w3_entry_validation.py::
    TestP331CovariantKwargs::test_stabilize_slices_is_honoured``.
    """
    st = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=10)
    st.add_tapered_grating(0.3e-6, eps_ridge=4.0, eps_groove=1.0,
                           duty_bottom=0.6, duty_top=0.4, n_slices=4)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        st.set_source(_WL).solve(stabilize="slices")
    assert not [x for x in w if "PASSIVE-BUT-WRONG" in str(x.message)]
    # No recorded taper -> the n_slices probe is impossible, but the guard
    # must NOT announce that it gave up: it runs the union-grid consensus.
    st2 = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=10)
    st2.add_layer(0.2e-6, eps=2.25)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o2, R2, T2, _j2 = st2.set_source(_WL).solve(stabilize="slices")
    msgs = [str(x.message) for x in w]
    assert not any("skipped" in m or "no taper builder" in m for m in msgs), (
        "the recipe-free route must run the union-grid consensus, not skip "
        f"it; got {msgs}")
    # A uniform layer has nothing to snap, so the consensus scores 0 and
    # must stay SILENT rather than cry pathology.
    assert not any("min_feature` was perturbed" in m for m in msgs), msgs
    assert np.isfinite(float(np.real(np.sum(R2) + np.sum(T2))))
    with pytest.raises(ValueError, match="stabilize"):
        st2.set_source(_WL).solve(stabilize="bogus")


# =========================================================================== #
# item 5 -- sweeps: segments jones + dispersive stacks
# =========================================================================== #

def test_pmm_jones_segments_sweep_matches_per_wavelength():
    segs = [(0.4, lambda w: 2.25 + 5.0 * (w / 1e-6) * 1j), (0.6, 1.0)]
    wls = (0.55e-6, 0.7e-6)
    wlv, J, Rt, Tt = la.pmm_jones_1d_segments_vs_wavelength(
        _P, segs, 1.5, 1.0, 0.3e-6, wls, degree=10)
    for i, w in enumerate(wls):
        o, R, T, jr = la.pmm_jones_1d_segments(
            _P, [(0.4, 2.25 + 5.0 * (w / 1e-6) * 1j), (0.6, 1.0)],
            1.5, 1.0, 0.3e-6, w, degree=10)
        assert np.max(np.abs(J[i] - jr)) == 0.0
        assert abs(Rt[i, 0] - R[0].sum()) == 0.0


def test_pmm_stack_dispersive_sweep_and_jones():
    wls = (0.55e-6, 0.7e-6)
    std = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=10)
    std.add_layer(0.2e-6, eps=lambda w: (1.45 + 0.05 * (w / 1e-6)) ** 2)
    std.add_layer(0.3e-6, segments=[
        (0.5, lambda w: 4.0 + 8.0 * (w / 1e-6) * 1j), (0.5, 1.0)])
    with pytest.raises(ValueError, match="DISPERSIVE"):
        std.set_source(0.6e-6).solve()
    od, Rd, Td, Jd = std.solve_vs_wavelength(wls, jones=True)
    for i, w in enumerate(wls):
        ctl = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=10)
        ctl.add_layer(0.2e-6, eps=(1.45 + 0.05 * (w / 1e-6)) ** 2)
        ctl.add_layer(0.3e-6, segments=[
            (0.5, 4.0 + 8.0 * (w / 1e-6) * 1j), (0.5, 1.0)])
        o1, R1, T1, j1 = ctl.set_source(w).solve()
        assert np.max(np.abs(Jd[i] - j1)) == 0.0
    # non-dispersive 3-tuple regression
    st = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=10)
    st.add_layer(0.2e-6, eps=2.25)
    o3 = st.solve_vs_wavelength(wls)
    assert len(o3) == 3


def test_pmm2d_stack_dispersive_sweep():
    P = 0.6e-6

    def cell_at(w):
        c = np.full((6, 6), 1.0 + 0j)
        c[1:4, 1:4] = 6.0 + 4.0 * (w / 1e-6) * 1j
        return c
    st = PMM2DStack(P, n_substrate=1.5, n_superstrate=1.0, degree=9,
                    n_orders=4)
    st.add_layer(0.2e-6, eps_cell=cell_at)
    with pytest.raises(ValueError, match="DISPERSIVE"):
        st.set_source(0.55e-6).solve()
    wls = (0.5e-6, 0.6e-6)
    o, R, T, J = st.solve_vs_wavelength(wls, jones=True)
    for i, w in enumerate(wls):
        ctl = PMM2DStack(P, n_substrate=1.5, n_superstrate=1.0, degree=9,
                         n_orders=4)
        ctl.add_layer(0.2e-6, eps_cell=cell_at(w))
        o1, R1, T1, j1 = ctl.set_source(w).solve()
        assert np.max(np.abs(R[i] - R1)) == 0.0
        assert np.max(np.abs(J[i] - j1)) == 0.0


# =========================================================================== #
# item 6 -- PMM per-layer / per-material absorption
# =========================================================================== #

def _lossy_stack(degree=12):
    st = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=degree)
    st.add_layer(0.15e-6, segments=[(0.5, 4.0 + 2.0j), (0.5, 1.0)])
    st.add_layer(0.10e-6, eps=2.25)
    st.add_layer(0.12e-6, segments=[(0.3, 2.25), (0.4, -20.0 + 3.0j),
                                    (0.3, 2.25)])
    return st


def test_pmm_layer_absorption_closes_against_far_field():
    st = _lossy_stack()
    o, R, T, _ = st.set_source(_WL).solve(retain_internal=True)
    A = st.layer_absorption()
    budget = 1.0 - R.sum(axis=1) - T.sum(axis=1)
    assert np.max(np.abs(A.sum(axis=0) - budget)) < 1e-10
    assert np.max(np.abs(A[1])) < 1e-12          # lossless spacer: exactly 0


def test_pmm_material_absorption_consistent():
    st = _lossy_stack()
    st.set_source(_WL).solve(retain_internal=True)
    A, mat = st.layer_absorption(by_material=True)
    tot = np.zeros(2)
    for v in mat.values():
        tot += v
    assert np.max(np.abs(tot - (A[0] + A[2]))) < 1e-10
    assert complex(4.0 + 2.0j) in mat and complex(-20.0 + 3.0j) in mat


def test_pmm_absorption_total_matches_rcwa_single_lossy_layer():
    """With a SINGLE lossy layer the split is the total, so the two solver
    families must agree (residual = cross-solver convergence, ~5e-3)."""
    st = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=16)
    st.add_layer(0.15e-6, segments=[(0.5, 4.0 + 2.0j), (0.5, 1.0)])
    o, R, T, _ = st.set_source(_WL).solve(retain_internal=True)
    A = st.layer_absorption()
    S = 1024
    c1 = np.where(np.arange(S)[:, None] < S // 2, 4.0 + 2.0j,
                  1.0).astype(complex)
    rs = RCWAStack(_P, n_superstrate=1.0, n_substrate=1.5, n_orders=40)
    rs.add_layer(0.15e-6, eps_cell=c1, formulation="li")
    res = rs.set_source(_WL).solve(retain_internal=True)
    Ar = np.asarray(res.layer_absorption())
    assert abs(float(A[0, 0]) - float(Ar[0, 0])) < 1e-2


def test_pmm_retain_internal_guards():
    st = _lossy_stack()
    with pytest.raises(ValueError, match="retain_internal"):
        st.set_source(_WL).solve()
        st.layer_absorption()


# =========================================================================== #
# items 2 + 9 -- geometry algebra + materials
# =========================================================================== #

def test_geometry_conformal_coat_hand_exact():
    g = SegmentStackGeometry(1.0)
    g.add_ridges(0.4, ridges=[(0.5, 0.3, 0.3, "Cu")], n_slices=1)
    g.coat(0.05, "Al")
    g.fill("LC")
    bands = g.layers()
    assert len(bands) == 4
    # cap band: film over tooth + overhang [0.3, 0.7]
    t0, segs0 = bands[0]
    assert abs(t0 - 0.05) < 1e-15
    assert [m for _w, m in segs0] == ["LC", "Al", "LC"]
    assert abs(segs0[1][0] - 0.4) < 1e-12
    # wall bands: 0.05-wide strips beside the tooth
    for t, segs in bands[1:3]:
        assert [m for _w, m in segs] == ["LC", "Al", "Cu", "Al", "LC"]
        assert abs(segs[1][0] - 0.05) < 1e-12
    # floor band: film across both gaps
    t3, segs3 = bands[3]
    assert abs(t3 - 0.05) < 1e-15
    assert [m for _w, m in segs3] == ["Al", "Cu", "Al"]


def test_geometry_liners():
    h = SegmentStackGeometry(1.0)
    h.add_band(0.2, [(0.3, "SiCN"), (0.4, "Cu"), (0.3, "SiCN")])
    h.line_interface("Cu", "SiCN", t=0.02, mat="Ta", side="a")
    (_t, segs), = h.layers()
    assert [m for _w, m in segs] == ["SiCN", "Ta", "Cu", "Ta", "SiCN"]
    assert abs(segs[2][0] - 0.36) < 1e-12     # carved from the Cu side
    v = SegmentStackGeometry(1.0)
    v.add_band(0.2, [(1.0, "Cu")])
    v.add_band(0.2, [(1.0, "SiCN")])
    v.line_interface("Cu", "SiCN", t=0.05, mat="Ta", side="a")
    mats = [segs[0][1] for _t, segs in v.layers()]
    ths = [t for t, _s in v.layers()]
    assert mats == ["Cu", "Ta", "SiCN"]
    assert np.allclose(ths, [0.15, 0.05, 0.2])


def test_geometry_feeds_both_solvers():
    g = SegmentStackGeometry(_P)
    g.add_ridges(0.3e-6, ridges=[(0.4e-6, 0.25e-6, 0.30e-6, "Cu")],
                 n_slices=3)
    g.coat(20e-9, "Al")
    g.fill("LC")
    mats = {"Cu": -20.0 + 3.0j, "Al": 3.1, "LC": 2.4}
    pst = g.to_pmm_stack(materials=mats, n_substrate=1.5, n_superstrate=1.0,
                         degree=12)
    o, R, T, _ = pst.set_source(_WL).solve()
    assert 0.0 < float(R[0].sum() + T[0].sum()) < 1.0      # lossy
    rst = g.to_rcwa_stack(materials=mats, n_superstrate=1.0, n_substrate=1.5,
                          n_orders=15)
    o2, R2, T2 = rst.set_source(_WL).solve().efficiencies()
    assert 0.0 < float(np.asarray(R2)[0].sum()
                       + np.asarray(T2)[0].sum()) < 1.0
    # unresolved background must refuse to export
    g2 = SegmentStackGeometry(_P)
    g2.add_ridges(0.1e-6, ridges=[(0.4e-6, 0.2e-6, 0.2e-6, "Cu")], n_slices=1)
    with pytest.raises(ValueError, match="BACKGROUND"):
        g2.layers()


def test_material_from_csv(tmp_path):
    f = tmp_path / "cu.csv"
    f.write_text("wl_um,n,k\n1.2,0.4,8.0\n1.3,0.45,8.5\n1.4,0.5,9.0\n")
    cu = la.Material.from_csv(str(f), wl_unit=1e-6, name="Cu")
    assert cu(1.3e-6) == (0.45 + 8.5j) ** 2
    assert abs(cu.index(1.3e-6) - (0.45 + 8.5j)) < 1e-15
    with pytest.raises(ValueError, match="outside"):
        cu(2.0e-6)
    st = PMMStack(_P, n_substrate=1.5, degree=10)
    st.add_layer(0.1e-6, segments=[(0.4, cu), (0.6, 2.25)])
    o, R, T, J = st.solve_vs_wavelength((1.25e-6, 1.35e-6), jones=True)
    assert (1.0 - R[:, 0].sum(axis=1) - T[:, 0].sum(axis=1) > 0).all()


# =========================================================================== #
# item 8 -- prepared material slots
# =========================================================================== #

def test_pmm_prepare_material_swap_matches_rebuild():
    def lc(phi):
        no2, ne2 = 1.5 ** 2, 1.7 ** 2
        c, s = np.cos(phi), np.sin(phi)
        M = np.diag([no2, no2, no2]).astype(complex)
        M[0, 0] = ne2 * c * c + no2 * s * s
        M[1, 1] = ne2 * s * s + no2 * c * c
        M[0, 1] = M[1, 0] = (ne2 - no2) * c * s
        return M
    st = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=12)
    st.add_layer(0.10e-6, eps=2.25)
    st.add_layer(0.15e-6, segments=[(0.4, 4.0 + 0.5j), (0.6, "LC")])
    prep = st.prepare()
    phis = np.linspace(0, np.pi / 2, 4)
    for f in phis:
        op, Rp, Tp, Jp = prep.solve(wavelength=_WL,
                                    materials={"LC": lc(f)})
        ctl = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=12)
        ctl.add_layer(0.10e-6, eps=2.25)
        ctl.add_layer(0.15e-6, segments=[(0.4, 4.0 + 0.5j), (0.6, lc(f))])
        oc, Rc, Tc, Jc = ctl.set_source(_WL).solve()
        assert np.array_equal(Rp, Rc) and np.array_equal(Jp, Jc)
    # 1 static-layer eig + 4 LC eigs (the LC-free layer eig is shared)
    assert len(prep._eig_cache) == 1 + len(phis)
    with pytest.raises(ValueError, match="material key"):
        prep.solve(wavelength=_WL)
    # a stack with unresolved keys refuses a plain solve
    with pytest.raises(ValueError, match="unresolved material keys"):
        st.set_source(_WL).solve()


def test_pmm_prepare_rejects_out_of_plane_tensors():
    """Audit S1-1 [P1]: the prepared path assembles the in-plane 2n eig
    (_build_sem_tensor_segments reads only exx/exy/eyx/eyy/ezz) and used to
    SILENTLY drop eps_xz/yz/zx/zy -- both energy-conserving, so no tripwire
    fired.  Both entry points must now raise: a CONCRETE OOP layer at
    prepare(), and a material KEY that resolves to an OOP tensor at solve().
    Independent oracle: a tilted-uniaxial optic axis (theta=30, phi=20 deg)
    which really has out-of-plane coupling, cross-checked against
    PMMStack._is_oop; an in-plane (eps_xy-only) tensor must still solve so the
    guard does not over-reject."""
    assert PMMStack._is_oop(np.asarray(_OOP))       # sanity: really OOP

    # (A) concrete OOP tensor layer -> rejected at prepare()
    sa = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=12)
    sa.add_layer(0.10e-6, eps=2.25)
    sa.add_layer(0.15e-6, eps=_OOP)
    with pytest.raises(NotImplementedError, match="out-of-plane"):
        sa.prepare()

    # (B) material key that resolves to an OOP tensor -> rejected at solve()
    sb = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=12)
    sb.add_layer(0.10e-6, eps=2.25)
    sb.add_layer(0.15e-6, segments=[(0.4, 2.25), (0.6, "X")])
    prep = sb.prepare()                             # keyed value unknown yet
    with pytest.raises(NotImplementedError, match="out-of-plane"):
        prep.solve(wavelength=_WL, materials={"X": _OOP})

    # positive control: an in-plane (eps_xy-only) key still resolves + solves
    ip = np.diag([1.5 ** 2, 1.7 ** 2, 1.6 ** 2]).astype(complex)
    ip[0, 1] = ip[1, 0] = 0.3
    assert not PMMStack._is_oop(ip)
    _op, Rp, _Tp, _Jp = prep.solve(wavelength=_WL, materials={"X": ip})
    assert np.isfinite(np.asarray(Rp)).all()


# =========================================================================== #
# item 7 -- viewers (Agg smoke)
# =========================================================================== #

def test_plot_geometry_smoke():
    st = PMMStack(_P, n_substrate=1.5, degree=10)
    st.add_tapered_ridges(0.2e-6, ridges=[(0.4 * _P, 0.2 * _P, 0.25 * _P,
                                           4.0)], eps_groove=1.0, n_slices=2)
    assert st.plot_geometry() is not None
    rs = RCWAStack(_P, n_superstrate=1.0, n_substrate=1.5, n_orders=5)
    rs.add_layer(0.1e-6, eps=2.25)
    rs.add_layer(0.2e-6, eps_cell=np.where(np.arange(64)[:, None] < 32, 4.0,
                                           1.0).astype(complex))
    assert rs.plot_geometry() is not None
    s2 = PMM2DStack(_P, n_substrate=1.5, degree=9, n_orders=4)
    s2.add_layer(0.1e-6, eps=2.25)
    assert len(s2.plot_geometry()) == 1
    g = SegmentStackGeometry(1.0)
    g.add_band(0.1, [(0.5, "A"), (0.5, "B")])
    assert g.plot() is not None


# =========================================================================== #
# RCWAStack out-of-plane promotion (prior roadmap, GAP2 follow-through)
# =========================================================================== #

_OOP = uniaxial_tensor(1.5, 1.7, np.deg2rad(30.0), phi=np.deg2rad(20.0))


def test_rcwa_stack_oop_matches_direct():
    S = 16
    cell = np.broadcast_to(_OOP, (S, S, 3, 3)).copy()
    st = RCWAStack(0.4e-6, period_y=0.4e-6, n_superstrate=1.0,
                   n_substrate=1.5, n_orders=2, n_orders_y=2)
    st.add_layer(0.4e-6, eps_tensor_cell=cell)
    res = st.set_source(_WL, theta=np.deg2rad(20), phi=np.deg2rad(25)).solve()
    o1, R1, T1 = res.efficiencies()
    o2, R2, T2, J2 = rcwa_jones_2d(0.4e-6, 0.4e-6, cell, 1.5, 1.0, 0.4e-6,
                                   _WL, theta=np.deg2rad(20),
                                   phi=np.deg2rad(25), n_orders_x=2,
                                   n_orders_y=2)
    assert np.max(np.abs(np.asarray(R1) - R2)) < 1e-12
    assert np.max(np.abs(np.asarray(res.jones_reflection()) - J2)) < 1e-12


def test_rcwa_stack_oop_split_identity_and_mixed_energy():
    S = 16
    cell = np.broadcast_to(_OOP, (S, S, 3, 3)).copy()
    one = RCWAStack(0.4e-6, period_y=0.4e-6, n_superstrate=1.0,
                    n_substrate=1.5, n_orders=2, n_orders_y=2)
    one.add_layer(0.4e-6, eps_tensor_cell=cell)
    two = RCWAStack(0.4e-6, period_y=0.4e-6, n_superstrate=1.0,
                    n_substrate=1.5, n_orders=2, n_orders_y=2)
    two.add_layer(0.2e-6, eps_tensor_cell=cell)
    two.add_layer(0.2e-6, eps_tensor_cell=cell)
    src = dict(theta=np.deg2rad(20), phi=np.deg2rad(25))
    o1, R1, T1 = one.set_source(_WL, **src).solve().efficiencies()
    o2, R2, T2 = two.set_source(_WL, **src).solve().efficiencies()
    assert np.max(np.abs(np.asarray(R1) - np.asarray(R2))) < 1e-10
    cellp = np.full((24, 24), 2.25 + 0j)
    cellp[6:18, 6:18] = 6.0
    mx = RCWAStack(0.4e-6, period_y=0.4e-6, n_superstrate=1.0,
                   n_substrate=1.5, n_orders=2, n_orders_y=2)
    mx.add_layer(0.1e-6, eps=2.25)
    mx.add_layer(0.2e-6, eps_tensor_cell=cell)
    mx.add_layer(0.15e-6, eps_cell=cellp)
    o, R, T = mx.set_source(_WL, theta=np.deg2rad(15),
                            phi=np.deg2rad(40)).solve().efficiencies()
    for r in (0, 1):
        assert abs(float(np.asarray(R)[r].sum()
                         + np.asarray(T)[r].sum()) - 1.0) < 1e-9


def test_rcwa_stack_oop_retain_internal_and_absorption_budget():
    """v5.22.0: ``RCWAStack.solve(retain_internal=True)`` now SOLVES
    out-of-plane tensor stacks (previously ``NotImplementedError``) via the
    Berreman-C2 generalized retention (explicit asymmetric mode sets + the
    full-tensor ``E_z`` recovery).  The per-layer absorption recovered from
    the retained internal fields closes the energy budget ``R + T + A = 1``
    per incident polarization, and the lossy OOP layers actually absorb."""
    S = 8
    lossy = uniaxial_tensor(1.5 + 0.02j, 1.7 + 0.03j, np.deg2rad(30.0),
                            phi=np.deg2rad(20.0))
    cell = np.broadcast_to(lossy, (S, S, 3, 3)).copy()
    st = RCWAStack(0.4e-6, period_y=0.4e-6, n_superstrate=1.0,
                   n_substrate=1.5, n_orders=2, n_orders_y=2)
    st.add_layer(0.2e-6, eps_tensor_cell=cell)
    st.add_layer(0.15e-6, eps_tensor_cell=cell)
    res = st.set_source(_WL, theta=np.deg2rad(20),
                        phi=np.deg2rad(25)).solve(retain_internal=True)
    _o, R, T = res.efficiencies()
    R, T = np.asarray(R), np.asarray(T)
    A = np.asarray(res.layer_absorption(nx=S))          # (2, n_layers)
    for r in (0, 1):
        assert abs(R[r].sum() + T[r].sum() + A[r].sum() - 1.0) < 1e-6
    assert A.sum() > 1e-3                                # the OOP layers absorb


# =========================================================================== #
# application feedback (FEEDBACK_DEVICE_GEOMETRY_V5_14_1_2026_06_10)
# =========================================================================== #

def test_feedback_by_key_attribution_and_legend_names():
    """Twin material keys mapped to the SAME eps (the under-grounded-tooth
    Ta trick: tooth-Cu vs column-Cu) must split BOTH the absorption map and
    the viewer legend -- attribution/labels ride the key names a
    SegmentStackGeometry export carries, with raw-eps stacks falling back to
    complex-eps keys."""
    g = SegmentStackGeometry(_P)
    g.add_ridges(0.15e-6, ridges=[(0.4e-6, 0.3e-6, 0.3e-6, "Cu")],
                 n_slices=1)
    g.add_ridges(0.12e-6, ridges=[(0.4e-6, 0.2e-6, 0.2e-6, "CuCol")],
                 n_slices=1)
    g.fill("LC")
    eps_cu = -20.0 + 3.0j
    st = g.to_pmm_stack(materials={"Cu": eps_cu, "CuCol": eps_cu, "LC": 2.4},
                        n_substrate=1.5, n_superstrate=1.0, degree=12)
    o, R, T, _ = st.set_source(_WL).solve(retain_internal=True)
    A, mat = st.layer_absorption(by_material=True)
    assert "Cu" in mat and "CuCol" in mat          # twins split
    tot = sum(v for v in mat.values())
    assert np.max(np.abs(tot - A.sum(axis=0))) < 1e-12
    budget = 1.0 - R.sum(axis=1) - T.sum(axis=1)
    assert np.max(np.abs(A.sum(axis=0) - budget)) < 1e-10
    labels = [t.get_text() for t in st.plot_geometry().get_legend()
              .get_texts()]
    assert {"Cu", "CuCol", "LC"} <= set(labels)
    # raw-eps stacks keep the complex-eps fallback
    st2 = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=12)
    st2.add_layer(0.15e-6, segments=[(0.5, 4.0 + 2.0j), (0.5, 1.0)])
    st2.set_source(_WL).solve(retain_internal=True)
    _A2, mat2 = st2.layer_absorption(by_material=True)
    assert complex(4.0 + 2.0j) in mat2


def test_feedback_to_rcwa_stack_tensor_materials():
    """Application feedback ask 4: to_rcwa_stack must accept (3, 3) tensor
    materials (the LC), pixelating mixed bands into eps_tensor_cell with
    scalars promoted to eps*I3; scalar-only bands keep eps_cell.  The two
    exports of one geometry must agree (same Fourier content)."""
    no2, ne2 = 1.5 ** 2, 1.7 ** 2
    c, sn = np.cos(0.6), np.sin(0.6)
    lc = np.diag([no2, no2, no2]).astype(complex)
    lc[0, 0] = ne2 * c * c + no2 * sn * sn
    lc[1, 1] = ne2 * sn * sn + no2 * c * c
    lc[0, 1] = lc[1, 0] = (ne2 - no2) * c * sn
    g = SegmentStackGeometry(_P)
    g.add_ridges(0.2e-6, ridges=[(0.4e-6, 0.3e-6, 0.3e-6, "Cu")], n_slices=1)
    g.fill("LC")
    g.add_band(0.1e-6, [(1.0, "SiCN")])
    mats = {"Cu": -20.0 + 3.0j, "LC": lc, "SiCN": 4.84}
    st = g.to_rcwa_stack(materials=mats, n_superstrate=1.0, n_substrate=1.5,
                         n_orders=10)
    kinds = [L.kind for L in st._layers]
    assert kinds == ["tensor", "uniform"] or kinds == ["tensor", "iso"]
    res = st.set_source(_WL).solve()
    o, R, T = res.efficiencies()
    tot = float(np.asarray(R)[0].sum() + np.asarray(T)[0].sum())
    assert 0.0 < tot < 1.0                      # lossy Cu present
    # PMM export of the SAME geometry agrees at the cross-family level
    pst = g.to_pmm_stack(materials=mats, n_substrate=1.5, n_superstrate=1.0,
                         degree=14)
    o2, R2, T2, _ = pst.set_source(_WL).solve()
    tot_p = float(R2[0].sum() + T2[0].sum())
    # COARSE cross-family sanity only: RCWA tensor layers are Laurent-only
    # and Cu at n_orders=10 is the documented under-resolved regime (the
    # application's own cross-check reads low even at nh=80); the exact
    # gates above pin the EXPORT mechanics, not FMM convergence.
    assert abs(tot - tot_p) < 0.25
    # scalar tensor (eps*I3) export == plain scalar export
    g2 = SegmentStackGeometry(_P)
    g2.add_band(0.2e-6, [(0.5, "A"), (0.5, "B")])
    sa = g2.to_rcwa_stack(materials={"A": 4.0, "B": 1.0}, n_superstrate=1.0,
                          n_substrate=1.5, n_orders=8)
    sb = g2.to_rcwa_stack(materials={"A": 4.0 * np.eye(3), "B": np.eye(3)},
                          n_superstrate=1.0, n_substrate=1.5, n_orders=8)
    oa, Ra, Ta = sa.set_source(_WL).solve().efficiencies()
    ob, Rb, Tb = sb.set_source(_WL).solve().efficiencies()
    assert np.max(np.abs(np.asarray(Ra) - np.asarray(Rb))) < 1e-10
