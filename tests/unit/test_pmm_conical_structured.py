"""Structured conical (phi != 0) modal solves of the nodal PMM cascade, and
geometric element grading at conical incidence (2026-11-14).

The derivation is the block comment above ``_conical_khat`` in
``lumenairy/elements/pmm/_core.py``.  References: ``RCWAStack(asr=True)``
(an independent adaptive-resolution Fourier solver, <= 3e-8 self-agreement
on these stacks), hard-coded below with the setting that produced them, and the
classical (phi = 0) solve for the degenerate limit.

Fail-before: ``conical.PMM_CONICAL_STRUCTURED = False`` restores the dense
conical eig; the decision tests below run both arms.  Every physics bar
carries its measurement (date, BLAS build) in a comment.
"""
import warnings

import numpy as np
import pytest

import lumenairy.elements.pmm._core as C
import lumenairy.elements.pmm.conical as CO
import lumenairy.elements.pmm.stack as S
from lumenairy.elements.pmm import (
    PMMStack,
    pmm_jones_1d_conical,
    pmm_jones_1d_conical_tensor,
)

WL, P, F = 1500e-9, 500e-9, 180.0 / 500.0
E_STRIP, E_MIRROR, E_SP = -90.0 + 7.0j, -130.0 + 20.0j, 3.6 + 0j
E_LOSSY = 2.2 + 0.6j
N_SUB = 1.5
#: the strip / spacer / mirror stack with its single-layer spacer (a generic
#: fixture): strip (40 nm, 180 nm in 500 nm) / 19 nm spacer (eps 3.6) /
#: 80 nm mirror, air above, n = 1.5 below.
STRIP = [(40e-9, [(F, E_STRIP), (1.0 - F, 1.0 + 0j)]), (19e-9, E_SP),
         (80e-9, E_MIRROR)]
#: the default geometric floor here: 1.25e-5 x wavelength
FLOOR = 1.875e-11
#: RCWAStack(asr=True), n_orders = 150, theta 0.2, phi 0.3 (n_orders 200
#: agrees to 2.8e-8).  Fixture replaced by a generic one; re-computed
#: 2026-11-15.
R_STRIP = np.array([[-0.07463526282317982 + 0.2962747189982692j,
                     0.008789862123693952 + 0.005852793967713634j],
                    [-0.0004522747074352674 - 0.0021081952860410817j,
                     -0.8998602726186565 - 0.4095331978275855j]])
#: dielectric lamellar grating: period 1000 nm, eps 4 bars (duty 0.4,
#: 300 nm) in air on n = 1.5, 800 nm, theta 0.2, phi 0.3.  RCWAStack
#: (asr=True), n_orders 150 (200 agrees to 3e-9).  Computed 2026-11-14.
R_DIEL = np.array([[-0.16392799511553932 + 0.11327071274217976j,
                    -0.0041112678137951225 - 0.005768706294897405j],
                   [-0.004086769303139145 - 0.005713157323674781j,
                    -0.14978979721699592 + 0.13494038832533586j]])


def _strip(degree, epr, grade, phi=0.3, theta=0.2, **kw):
    st = PMMStack(P, n_substrate=N_SUB, n_superstrate=1.0, degree=degree,
                  elements_per_region=epr, grade=grade, **kw)
    for t, e in STRIP:
        if isinstance(e, list):
            st.add_layer(t, segments=e)
        else:
            st.add_layer(t, eps=e)
    st.set_source(WL, theta=theta, phi=phi)
    return st


def _diel(degree, epr, grade, phi=0.3):
    st = PMMStack(1000e-9, n_substrate=1.5, degree=degree,
                  elements_per_region=epr, grade=grade)
    st.add_layer(300e-9, segments=[(0.4, 4.0), (0.6, 1.0)])
    st.set_source(800e-9, theta=0.2, phi=phi)
    return st


def _J(st):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(st.solve()[3])


@pytest.fixture
def dense_conical(monkeypatch):
    """The pre-2026-11-14 dense conical eig, with the refusal it carried
    lifted so the decision tests can measure it."""
    monkeypatch.setattr(CO, "PMM_CONICAL_STRUCTURED", False)
    monkeypatch.setattr(S, "_refuse_geometric_conical", lambda g, w: None)
    yield


# --------------------------------------------------------------------------- #
# the kernels
# --------------------------------------------------------------------------- #
def _grid(eps_list, degree=8, epr=12, grade=("geometric", 0.17, FLOOR)):
    ts = [C._tensor3_dict(e * np.eye(3)) for e in eps_list]
    w = [F, 1.0 - F] if len(eps_list) == 2 else [1.0]
    return C._build_sem_tensor_segments(P, w, ts, degree, epr, grade)


def _pq(mats, k0, kx0, ky0):
    """The full conical factors exactly as ``_sem_modes_tensor`` builds them
    (isotropic layer): ``P0 + dP``, ``Q0 + dQ``."""
    n = mats["n_glob"]
    apply, _ = C._row_scale_apply(mats["S0"])
    cxx = 1.0 / np.diag(apply(mats["mass"]["inv_xx"]))
    k02 = k0 * k0
    stf, cnv, mss = mats["stiff"], mats["conv"], mats["mass"]

    def kxop(key):
        op = stf[key] - 1j * kx0 * (cnv[key] - cnv[key].T) + kx0 ** 2 * mss[key]
        return apply(op) / k02
    G = np.eye(n) - kxop("inv_ezz")
    Kx2 = kxop("one")
    cyy = np.diag(apply(mss["schur"]))
    Z = np.zeros((n, n), complex)
    P0 = np.block([[Z, G], [-np.eye(n), Z]])
    Q0 = np.block([[Z, np.diag(cyy) - Kx2], [-np.diag(cxx), Z]])
    dP, dQ = C._conical_dpq(mats, k0, kx0, ky0)
    return P0 + dP, Q0 + dQ


def test_uniform_conical_modes_are_exact_eigenpairs():
    """The structured uniform modes satisfy the first-order conical equations
    ``Q W = lam V`` and ``P V = lam W`` of the discrete operator, with exactly
    degenerate pairs and exactly zero H_y on the second family.  Checked on a
    Chebyshev grid: a residual needs the products ``Q @ W`` / ``P @ V``
    themselves, whose round-off on a geometric grid (~eps x ||Kx^2|| ~ 1e-4
    relative) is exactly what the structured solve avoids.  RE-MEASURED
    2026-11-15 on the generic fixture (20 slowest modes, air / lossy film /
    mirror): 3.2e-11 / 1.4e-11 / 2.4e-13 (Windows/MKL)."""
    k0 = 2 * np.pi / WL
    kx0, ky0 = 0.19 * k0, 0.06 * k0
    for eps in (1.0 + 0j, E_LOSSY, E_MIRROR):
        mats = _grid([eps, eps], degree=10, epr=3, grade=True)
        n = mats["n_glob"]
        W, V, lam, q = C._sem_modes_uniform_conical(mats, k0, kx0, ky0, eps)
        assert np.array_equal(q[:n] ** 2, q[n:] ** 2)          # exact pairs
        assert np.all(V[n:, n:] == 0.0)                         # H_y = 0
        Pm, Qm = _pq(mats, k0, kx0, ky0)
        low = np.argsort(np.abs(q))[:20]
        rq = (np.linalg.norm(Qm @ W[:, low] - V[:, low] * lam[low], axis=0)
              / np.linalg.norm(V[:, low] * lam[low], axis=0))
        rp = (np.linalg.norm(Pm @ V[:, low] - W[:, low] * lam[low], axis=0)
              / np.linalg.norm(W[:, low] * lam[low], axis=0))
        assert float(max(rq.max(), rp.max())) < 1e-8, (eps, rq.max(), rp.max())


def test_conical_kernels_reduce_to_the_classical_ones():
    """``ky0 -> 0``: the structured uniform modes tend to the shared-eig
    classical ones and the classical-basis layer solve to the classical eig,
    linearly in ky0.  RE-MEASURED 2026-11-15 on the generic fixture at
    kyn = 1e-9, 40 slowest modes: 3.5e-12 (uniform q) / 3.2e-10 (layer q)
    on Windows/MKL."""
    k0 = 2 * np.pi / WL
    kx0 = 0.19 * k0
    mats = _grid([2.25 + 0j, 2.25 + 0j])
    q0 = C._sem_modes_uniform(mats, k0, kx0, 2.25)[3]
    q1 = C._sem_modes_uniform_conical(mats, k0, kx0, 1e-9 * k0, 2.25)[3]
    low = np.argsort(np.abs(q0))[:40]
    d = max(np.min(np.abs(q1 - x)) for x in q0[low])
    assert d < 1e-8, d
    mats = _grid([E_STRIP, 1.0 + 0j])
    q0 = C._sem_modes_tensor(mats, k0, kx0, True)[3]
    q1 = C._sem_modes_conical(mats, k0, kx0, 1e-9 * k0)[3]
    low = np.argsort(np.abs(q0))[:40]
    d = max(np.min(np.abs(q1 - x)) for x in q0[low])
    assert d < 1e-8, d


# --------------------------------------------------------------------------- #
# geometric grading at conical incidence, against RCWAStack(asr=True)
# --------------------------------------------------------------------------- #
#: Geometric degree 8 x 12 el. (default floor 18.75 pm), strip stack.
#: RE-MEASURED 2026-11-15 on the generic fixture, |J - R_STRIP| per entry
#: (Windows/MKL): xx 6.0e-6 (the planar accuracy of the layout: the same
#: layout at phi = 0 is 5.4e-6 from its ASR reference), yy 1.1e-7, xy
#: 6.8e-8, yx 7.8e-9.  The dense eig read xy 1.2e-4 (round-off: it moved
#: 4.4e-4 under a 1e-10 relative change of phi).
XY_BAR = 3e-7
XX_BAR = 1e-5


def test_geometric_conical_matches_the_asr_reference():
    J = _J(_strip(8, 12, "geometric"))
    e = np.abs(J - R_STRIP)
    assert e[0, 1] < XY_BAR and e[1, 0] < XY_BAR, e
    assert e[0, 0] < XX_BAR and e[1, 1] < XX_BAR, e


def test_dense_conical_eig_fails_the_same_comparison(dense_conical):
    """Decision pair with the test above (fail-before): the dense eig of
    M = -(P Q) on the same layout.  RE-MEASURED 2026-11-15 on the generic
    fixture: xy 1.2e-4 (Windows/MKL), 4x above the bar; it is round-off
    (the dense J_xy moves 4.4e-4 under a 1e-10 relative change of phi and
    3.6e-4 between degree 8 and 12)."""
    J = _J(_strip(8, 12, "geometric"))
    assert abs(J[0, 1] - R_STRIP[0, 1]) > 3e-5


def test_geometric_conical_cross_polarization_converges_in_degree():
    """J_xy over degree 8 / 12 / 16.  RE-MEASURED 2026-11-15 on the generic
    fixture: max spread 5.5e-8 (Windows/MKL); the dense eig drifted 3.6e-4
    between degree 8 and 12."""
    js = [_J(_strip(d, 12, "geometric"))[0, 1] for d in (8, 12, 16)]
    assert max(abs(a - b) for a in js for b in js) < 2e-7, js


def test_geometric_conical_reduces_to_the_classical_solve():
    """phi = 1e-9 against phi = 0 (the classical solve) on the geometric
    layout.  The physical cross-polarized term is ~3.8e-11 here (Chebyshev
    3 el. reads 3.8e-11).  RE-MEASURED 2026-11-15 on the generic fixture:
    3.7e-11 (Windows/MKL); the dense eig read 1.6e-6."""
    a = _J(_strip(8, 12, "geometric", phi=1e-9))
    b = _J(_strip(8, 12, "geometric", phi=0.0))
    assert np.max(np.abs(a - b)) < 3e-9


def test_geometric_conical_dielectric_grating():
    """Dielectric lamellar grating, geometric degree 8 x 12 el.  MEASURED
    2026-11-14: max |J - R_DIEL| 3.62e-8 (Windows/MKL), 3.63e-8
    (WSL/OpenBLAS); Chebyshev 3 el. at degree 24 is 1.0e-6 off."""
    J = _J(_diel(8, 12, "geometric"))
    assert np.max(np.abs(J - R_DIEL)) < 4e-7


def test_single_layer_conical_entry_points_accept_geometric():
    """``pmm_jones_1d_conical`` / ``_tensor`` reach the same nodal cascade:
    the strip alone (n = 1.5 below) agrees between them and with PMMStack
    to round-off, and converges in degree."""
    args = (P, E_STRIP, 1.0, N_SUB, 1.0, 40e-9, F, WL)
    j1 = np.asarray(pmm_jones_1d_conical(
        *args, theta=0.2, phi=0.3, degree=8, elements_per_region=12,
        grade="geometric")[3])
    cell = np.array([E_STRIP * np.eye(3), np.eye(3)])
    # the tensor entry samples the cell on a pixel grid: a 2-pixel cell puts
    # the wall at 0.5, so compare it against PMMStack on the same 0.5 duty
    j2 = np.asarray(pmm_jones_1d_conical_tensor(
        P, cell, N_SUB, 1.0, 40e-9, WL, theta=0.2, phi=0.3, degree=8,
        elements_per_region=12, grade="geometric")[3])
    st = PMMStack(P, n_substrate=N_SUB, degree=8, elements_per_region=12,
                  grade="geometric")
    st.add_layer(40e-9, segments=[(F, E_STRIP), (1 - F, 1.0)])
    st.set_source(WL, theta=0.2, phi=0.3)
    j3 = _J(st)
    st2 = PMMStack(P, n_substrate=N_SUB, degree=8, elements_per_region=12,
                   grade="geometric")
    st2.add_layer(40e-9, segments=[(0.5, E_STRIP), (0.5, 1.0)])
    st2.set_source(WL, theta=0.2, phi=0.3)
    j4 = _J(st2)
    # (the entry points project onto different far-field order counts, which
    # moves the min-norm projection at round-off: measured 2.5e-11 / 3.9e-11
    # on the generic fixture, 2026-11-15; J_xy degree 12 vs 8: 2.7e-9)
    assert np.max(np.abs(j1 - j3)) < 1e-9
    assert np.max(np.abs(j2 - j4)) < 1e-9
    j12 = np.asarray(pmm_jones_1d_conical(
        *args, theta=0.2, phi=0.3, degree=12, elements_per_region=12,
        grade="geometric")[3])
    assert abs(j12[0, 1] - j1[0, 1]) < 2e-7


# --------------------------------------------------------------------------- #
# Chebyshev layouts: round-off moved toward the references
# --------------------------------------------------------------------------- #
def test_chebyshev_conical_roundoff_reduced():
    """Chebyshev 12 el. x degree 12 (fine elements): stability under a
    1e-10 relative change of phi.  RE-MEASURED 2026-11-15 on the generic
    fixture: structured 4.2e-11, dense 3.0e-9 (Windows/MKL).  Results
    otherwise move by <= 3.7e-9 against discretization errors of 7e-4."""
    a = _J(_strip(12, 12, True))
    b = _J(_strip(12, 12, True, phi=0.3 * (1 + 1e-10)))
    assert np.max(np.abs(a - b)) < 1e-9


def test_theta_zero_conical_equals_classical_exactly():
    """theta = 0 with phi != 0 has ky0 = 0: the cascade's uniform media now
    take the classical shared-eig modes, so the conical route reproduces the
    classical normal-incidence solve to round-off on a geometric grid.
    RE-MEASURED 2026-11-15 on the generic fixture: 4.7e-15 (Windows/MKL);
    the dense uniform eig read 4.0e-7."""
    a = _J(_strip(8, 12, "geometric", theta=0.0, phi=0.3))
    b = _J(_strip(8, 12, "geometric", theta=0.0, phi=0.0))
    assert np.max(np.abs(a - b)) < 1e-12


def test_structured_fallback_policy(monkeypatch):
    """Where a structured solve is undefined (FloatingPointError), Chebyshev
    falls back to the dense eig and geometric raises."""
    def boom(*a, **k):
        raise FloatingPointError("test")
    monkeypatch.setattr(C, "_sem_modes_conical", boom)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _strip(6, 3, True).solve()
        with pytest.raises(ValueError, match="structured modal solve"):
            _strip(6, 4, "geometric").solve()
