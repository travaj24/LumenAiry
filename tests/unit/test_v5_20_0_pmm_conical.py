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
