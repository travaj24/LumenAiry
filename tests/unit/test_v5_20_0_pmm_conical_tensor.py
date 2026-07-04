"""Native full-tensor conical PMM -- pmm_jones_1d_conical_tensor (conical audit
Path B phase 2).

The y-uniform anisotropic (LC director) profile is routed through the SAME 2-D
tensor machinery (``_tensor_layer_modes``) with ``oy = [0]`` -- an O(N) reduction
of the 2-D coupled build.  Gates:

  G_inplane : a uniform IN-PLANE tensor slab at conical incidence matches the
              analytic Berreman 4x4 conical oracle (singular values, basis-
              invariant -- the common twisted-nematic / in-plane-director case).
  G_iso     : an isotropic tensor (eps*I) reduces BYTE-EXACTLY to the isotropic
              native pmm_jones_1d_conical.
  G_oop_norm: an OUT-OF-PLANE (tilted-director) tensor slab at NORMAL incidence
              matches Berreman exactly (the OOP coupling A/B vanish at kt=0).
  G_faithful: the native reduction reproduces the full 2-D pmm_jones_2d for an
              OOP tensor at conical (same generator) -- the reduction is exact;
              agreement with Berreman there is a shared-generator follow-up
              (documented), so it is NOT asserted here.
  G_energy  : lossless energy closure per incident polarization.
"""
import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements.berreman import berreman_jones_1d
from lumenairy.elements.rcwa._core import uniaxial_tensor

# All v5.20.0 PMM/RCWA/conical tests are eig-heavy 2-D numerics -> run in
# the dedicated slow-tests job (keeps the fast unit gate under its 25-min
# cap, the v5.15.0 design); they still gate every push there.
pytestmark = pytest.mark.slow

_P, _WL, _DEP = 0.6e-6, 0.55e-6, 0.30e-6
_TH, _PHI = np.deg2rad(30.0), np.deg2rad(40.0)


def _sv(J):
    return np.sort(np.linalg.svd(np.asarray(J), compute_uv=False))


def _inplane(n_o, n_e, az):
    """Exactly-in-plane uniaxial tensor (zz = n_o**2, no z-coupling)."""
    eo, ee = n_o ** 2, n_e ** 2
    c, s = np.cos(az), np.sin(az)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ np.diag([ee, eo, eo]).astype(complex) @ Rz.T


def test_native_conical_inplane_tensor_matches_berreman():
    """G_inplane: uniform in-plane tensor slab at conical == Berreman (sv)."""
    T = _inplane(1.5, 1.7, np.deg2rad(25.0))
    cell = np.tile(T, (6, 1, 1))                       # (Sx, 3, 3), uniform slab
    o, R, Tt, J = la.pmm_jones_1d_conical_tensor(
        _P, cell, 1.5, 1.0, _DEP, _WL, theta=_TH, phi=_PHI, degree=9, n_orders=3)
    _Rb, _Tb, Jr, _Jt = berreman_jones_1d([(T, _DEP)], 1.5, 1.0, _WL,
                                           angle=_TH, phi=_PHI)
    assert np.allclose(_sv(J), _sv(Jr), atol=3e-3), (
        f"in-plane conical sv {_sv(J)} vs berreman {_sv(Jr)}")
    for p in range(2):
        assert abs(float(R[p].sum() + Tt[p].sum()) - 1.0) < 1e-6


def test_native_conical_isotropic_tensor_reduces_to_scalar():
    """G_iso: an isotropic tensor (eps*I) binary grating reduces BYTE-EXACTLY to
    the isotropic native pmm_jones_1d_conical."""
    epsr, epsg = 6.0, 1.0
    cell = np.zeros((8, 3, 3), dtype=complex)
    cell[:4] = np.eye(3) * epsr
    cell[4:] = np.eye(3) * epsg
    a = la.pmm_jones_1d_conical_tensor(
        _P, cell, 1.5, 1.0, _DEP, _WL, theta=_TH, phi=_PHI, degree=15,
        n_orders=7, formulation="laurent")
    b = la.pmm_jones_1d_conical(
        _P, epsr, epsg, 1.5, 1.0, _DEP, 0.5, _WL, theta=_TH, phi=_PHI,
        degree=15, n_orders=7, formulation="laurent")
    assert np.array_equal(a[0], b[0])
    assert np.max(np.abs(a[1] - b[1])) == 0.0
    assert np.max(np.abs(a[2] - b[2])) == 0.0
    assert np.max(np.abs(np.asarray(a[3]) - np.asarray(b[3]))) == 0.0


def test_native_conical_oop_tensor_matches_berreman_at_normal():
    """G_oop_norm: an OUT-OF-PLANE (tilted-director) tensor slab matches
    Berreman EXACTLY at normal incidence (the A/B OOP blocks vanish at kt=0)."""
    T = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0), phi=np.deg2rad(20.0))
    cell = np.tile(T, (6, 1, 1))
    o, R, Tt, J = la.pmm_jones_1d_conical_tensor(
        _P, cell, 1.5, 1.0, _DEP, _WL, theta=0.0, phi=0.0, degree=9, n_orders=3)
    _Rb, _Tb, Jr, _Jt = berreman_jones_1d([(T, _DEP)], 1.5, 1.0, _WL,
                                           angle=0.0, phi=0.0)
    assert np.allclose(_sv(J), _sv(Jr), atol=3e-3), (
        f"OOP normal sv {_sv(J)} vs berreman {_sv(Jr)}")


def test_native_conical_oop_reduces_the_2d_path_faithfully():
    """G_faithful: for an OUT-OF-PLANE tensor at conical incidence the native
    reduction reproduces the full 2-D pmm_jones_2d (same generator) -- the O(N)
    reduction is exact.  (Berreman agreement at OOP+conical is a documented
    shared-generator follow-up, so it is deliberately NOT asserted here.)"""
    T = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0), phi=np.deg2rad(20.0))
    cell1d = np.tile(T, (6, 1, 1))
    cell2d = np.tile(T, (6, 6, 1, 1))                  # y-uniform 2-D tile
    o1, R1, T1, J1 = la.pmm_jones_1d_conical_tensor(
        _P, cell1d, 1.5, 1.0, _DEP, _WL, theta=_TH, phi=_PHI, degree=9, n_orders=3)
    o2, R2, T2, J2 = la.pmm_jones_2d(
        _P, _P, cell2d, 1.5, 1.0, _DEP, _WL, theta=_TH, phi=_PHI, degree=9,
        n_orders=3)
    # the native (m, 0) orders are the n_y = 0 slice of the 2-D order set
    keep = o2[:, 1] == 0
    assert np.allclose(_sv(J1), _sv(np.asarray(J2)), atol=1e-9)
    assert np.max(np.abs(R1 - R2[:, keep])) < 1e-9
    assert np.max(np.abs(T1 - T2[:, keep])) < 1e-9
    for p in range(2):
        assert abs(float(R1[p].sum() + T1[p].sum()) - 1.0) < 5e-3   # energy


def test_native_conical_tensor_rejects_non_uniform_y():
    """A genuinely doubly-periodic tensor cell (Sy > 1) is rejected -- this
    entry is the y-UNIFORM native reduction; use pmm_jones_2d otherwise."""
    cell = np.tile(np.eye(3).astype(complex), (4, 2, 1, 1))   # Sy = 2
    with pytest.raises(ValueError, match="uniform along y"):
        la.pmm_jones_1d_conical_tensor(_P, cell, 1.5, 1.0, _DEP, _WL,
                                       theta=_TH, phi=_PHI)
