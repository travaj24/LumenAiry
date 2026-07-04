"""Native conical (out-of-plane, phi != 0) PMMStack -- the O(N) n_y = 0 multilayer
reduction (conical audit Path B phase 4).

PMMStack.set_source(phi=...) routes phi != 0 through PMMStack._solve_conical:
each layer's coupled 2N modes come from the same 2-D machinery the native
single-layer conical uses (scalar layers via the Li inverse-rule projected path,
tensor layers via the generalized generator), with oy = [0].  Gates:

  G_grating : a single scalar grating layer reproduces pmm_jones_1d_conical
              BYTE-EXACTLY (the stack's scalar path IS the native scalar path).
  G_tensor  : a single in-plane tensor layer reproduces
              pmm_jones_1d_conical_tensor byte-exactly.
  G_slab    : a uniform slab matches the analytic Berreman 4x4 conical oracle.
  G_pathA   : a multilayer stack matches PMM2DStack Path A (y-invariant cells)
              on the (m, 0) orders.
  G_phi0    : phi = 0 is byte-identical to the classical (no-phi) solve.
  G_reject  : conical + slant / stabilize / retain_internal raise clearly.
"""
import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements.berreman import berreman_jones_1d

_P, _WL, _DEP = 0.6e-6, 0.55e-6, 0.30e-6
_TH, _PHI = np.deg2rad(30.0), np.deg2rad(40.0)


def _sv(J):
    return np.sort(np.linalg.svd(np.asarray(J), compute_uv=False))


def _inplane(n_o, n_e, az):
    eo, ee = n_o ** 2, n_e ** 2
    c, s = np.cos(az), np.sin(az)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ np.diag([ee, eo, eo]).astype(complex) @ Rz.T


def test_stack_conical_scalar_grating_matches_native():
    """G_grating: a single scalar grating layer == pmm_jones_1d_conical
    BYTE-EXACTLY (the stack's scalar path is the native scalar path)."""
    duty, epsr, epsg = 0.5, 6.0, 1.0
    st = la.PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=15,
                     far_field_orders=15)
    st.add_layer(_DEP, segments=[(duty, epsr), (1 - duty, epsg)])
    o1, R1, T1, J1 = st.set_source(_WL, angle=_TH, phi=_PHI).solve()
    o2, R2, T2, J2 = la.pmm_jones_1d_conical(
        _P, epsr, epsg, 1.5, 1.0, _DEP, duty, _WL, theta=_TH, phi=_PHI,
        degree=15, n_orders=7)
    # PMMStack returns the 1-D (m,) order array (pmm_jones_1d_segments contract);
    # the native single-layer returns (Nf, 2) (m, n_y=0) like pmm_jones_2d
    assert o1.ndim == 1 and np.array_equal(o1, o2[:, 0])
    assert np.max(np.abs(np.asarray(R1) - np.asarray(R2))) == 0.0
    assert np.max(np.abs(np.asarray(T1) - np.asarray(T2))) == 0.0
    assert np.max(np.abs(np.asarray(J1) - np.asarray(J2))) == 0.0


def test_stack_conical_tensor_layer_matches_native():
    """G_tensor: a single in-plane tensor layer == pmm_jones_1d_conical_tensor
    byte-exactly."""
    T = _inplane(1.5, 1.7, np.deg2rad(25.0))
    st = la.PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=9,
                     far_field_orders=7)
    st.add_layer(_DEP, eps=T)
    o1, R1, T1, J1 = st.set_source(_WL, angle=_TH, phi=_PHI).solve()
    cell = np.tile(T, (6, 1, 1))
    o2, R2, T2, J2 = la.pmm_jones_1d_conical_tensor(
        _P, cell, 1.5, 1.0, _DEP, _WL, theta=_TH, phi=_PHI, degree=9, n_orders=3)
    assert np.max(np.abs(np.asarray(J1) - np.asarray(J2))) == 0.0
    assert np.max(np.abs(np.asarray(R1) - np.asarray(R2))) == 0.0


def test_stack_conical_uniform_slab_matches_berreman():
    """G_slab: a uniform slab == the analytic Berreman conical oracle (sv)."""
    st = la.PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=9,
                     far_field_orders=7)
    st.add_layer(_DEP, eps=2.25)
    o, R, T, J = st.set_source(_WL, angle=_TH, phi=_PHI).solve()
    _Rb, _Tb, Jr, _Jt = berreman_jones_1d([(2.25, _DEP)], 1.5, 1.0, _WL,
                                           angle=_TH, phi=_PHI)
    assert np.allclose(_sv(J), _sv(Jr), atol=3e-3)
    for p in range(2):
        assert abs(float(R[p].sum() + T[p].sum()) - 1.0) < 1e-8


def test_stack_conical_multilayer_matches_path_a():
    """G_pathA: a (gentle) multilayer stack matches PMM2DStack Path A (the
    y-invariant-cell bridge) on the (m, 0) orders."""
    Pp, WL = 0.6e-6, 0.6e-6
    th, ph = np.deg2rad(20.0), np.deg2rad(35.0)
    nq = 8
    deg = 2 * nq + 1
    st = la.PMMStack(Pp, n_substrate=1.5, n_superstrate=1.0, degree=deg,
                     far_field_orders=2 * nq + 1)
    st.add_layer(0.2e-6, segments=[(0.5, 2.5), (0.5, 1.2)])
    st.add_layer(0.15e-6, eps=1.9)
    o1, R1, T1, J1 = st.set_source(WL, angle=th, phi=ph).solve()
    S = 2 * nq + 2
    c1 = np.full((S, S), 1.2 + 0j)
    c1[:S // 2, :] = 2.5
    st2 = la.PMM2DStack(period_x=Pp, period_y=Pp, n_substrate=1.5,
                        n_superstrate=1.0, degree=deg, n_orders=nq)
    st2.add_layer(0.2e-6, eps_cell=c1)
    st2.add_layer(0.15e-6, eps=1.9)
    o2, R2, T2, J2 = st2.set_source(WL, theta=th, phi=ph).solve()
    keep = o2[:, 1] == 0
    assert np.max(np.abs(_sv(J1) - _sv(J2))) < 3e-3
    assert np.max(np.abs(np.asarray(R1) - np.asarray(R2)[:, keep])) < 3e-3


def test_stack_conical_phi0_is_classical():
    """G_phi0: phi = 0 must NOT take the conical path -- byte-identical to the
    classical (no-phi) solve."""
    st = la.PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=13)
    st.add_layer(_DEP, segments=[(0.5, 6.0), (0.5, 1.0)])
    a = st.set_source(_WL, angle=_TH, phi=0.0).solve()
    b = st.set_source(_WL, angle=_TH).solve()
    assert np.array_equal(np.asarray(a[1]), np.asarray(b[1]))
    assert np.array_equal(np.asarray(a[2]), np.asarray(b[2]))
    assert np.array_equal(np.asarray(a[3]), np.asarray(b[3]))


def test_stack_conical_rejects_unsupported_combinations():
    """G_reject: conical + slant / stabilize / retain_internal raise clearly."""
    st = la.PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=11)
    st.add_layer(_DEP, segments=[(0.5, 6.0), (0.5, 1.0)],
                 slant_angle=np.deg2rad(10.0))
    with pytest.raises(NotImplementedError, match="SLANTED"):
        st.set_source(_WL, angle=_TH, phi=_PHI).solve()
    st2 = la.PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=11)
    st2.add_layer(_DEP, segments=[(0.5, 6.0), (0.5, 1.0)])
    st2.set_source(_WL, angle=_TH, phi=_PHI)
    with pytest.raises(NotImplementedError, match="stabilize"):
        st2.solve(stabilize="slices")
    with pytest.raises(NotImplementedError, match="retain_internal"):
        st2.solve(retain_internal=True)


def test_stack_conical_caps_orders_and_warns_on_underresolution():
    """Audit review (P2): the conical path CAPS n_orders to the spectral-element
    grid capacity (no rank-deficient garbage), RAISES when the grid cannot even
    resolve the propagating orders, and WARNS on residual energy violation --
    classical-path parity (previously it silently returned R+T up to ~6.9)."""
    import warnings
    Pp, WL, DEP = 2.0e-6, 1.0e-6, 0.6e-6
    th, ph = np.deg2rad(30.0), np.deg2rad(40.0)
    # degree too low to resolve the propagating orders -> raise, not silent garbage
    st = la.PMMStack(Pp, n_substrate=1.5, n_superstrate=1.0, degree=4,
                     far_field_orders=41)
    st.add_layer(DEP, segments=[(0.5, 6.0), (0.5, 1.0)])
    with pytest.raises(ValueError, match="propagate"):
        st.set_source(WL, angle=th, phi=ph).solve()
    # a tight-but-resolvable degree caps the output well below far_field_orders
    # AND fires the energy tripwire on the residual under-resolution
    st2 = la.PMMStack(Pp, n_substrate=1.5, n_superstrate=1.0, degree=5,
                      far_field_orders=41)
    st2.add_layer(DEP, segments=[(0.5, 6.0), (0.5, 1.0)])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        o, R, T, J = st2.set_source(WL, angle=th, phi=ph).solve()
    assert o.ndim == 1 and len(o) < 41                      # capped to capacity
    assert float((R.sum(1) + T.sum(1)).max()) < 1.1        # no rank-def blowup
    assert any("energy" in str(x.message).lower() for x in rec)   # tripwire fired
