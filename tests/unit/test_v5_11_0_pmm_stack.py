"""PMMStack -- multilayer 1-D PMM (the RCWAStack analogue).

Composes anisotropic 1-D patterned layers + uniform spacers on the UNION of
every layer's walls (one shared nodal grid) and Redheffer-stacks them.  Validated
against (1) the single-layer ``pmm_jones_1d_segments`` (a 1-layer stack is
bit-identical) and (2) ``RCWAStack`` (a multilayer tensor stack matches the FMM
to the convergence tolerance, energy conserved).
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements.rcwa import uniaxial_tensor

_C = np.complex128
GR = np.eye(3, dtype=_C)
WL = 0.55e-6


def test_one_layer_stack_equals_segments():
    # a 1-layer PMMStack reproduces pmm_jones_1d_segments.  Bit-identity held
    # while both paths used the gauge-INDEPENDENT legacy Im(q)>=0 mode branch;
    # since v5.14 (the dense-resonance P1 fix) both use the Poynting-flux
    # selector, whose sign choices within DEGENERATE mode pairs depend on the
    # eig gauge -- the two paths assemble machine-different matrices, so their
    # equally-valid selections can differ, propagating ~1e-7 into J.  Pin a
    # tight tolerance instead.
    lc = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.4)
    for ang_deg in (0.0, 25.0):
        st = la.PMMStack(0.8e-6, n_substrate=1.5, n_superstrate=1.0, degree=20)
        st.add_layer(0.5e-6, segments=[(0.5, lc), (0.5, GR)])
        o, R, T, J = st.set_source(WL, angle=np.radians(ang_deg)).solve()
        o2, R2, T2, J2 = la.pmm_jones_1d_segments(
            0.8e-6, [(0.5, lc), (0.5, GR)], 1.5, 1.0, 0.5e-6, WL,
            angle=np.radians(ang_deg), degree=20)
        assert np.array_equal(o, o2)
        assert np.max(np.abs(J - J2)) < 5e-6
        assert np.max(np.abs(R - R2)) < 5e-6 and np.max(np.abs(T - T2)) < 5e-6


def _rcwa_two_layer(theta):
    S = 256
    x = (np.arange(S) + 0.5) / S
    lcB = uniaxial_tensor(1.5, 1.7, np.pi / 2, phi=0.3)
    cellA = np.empty((S, 1, 3, 3), dtype=complex)
    cellB = np.empty((S, 1, 3, 3), dtype=complex)
    for k in range(S):
        cellA[k, 0] = (2.5 * GR) if abs(x[k] - 0.5) < 0.25 else GR
        cellB[k, 0] = lcB if abs(x[k] - 0.3) < 0.2 else (2.0 * GR)
    st = la.RCWAStack(0.8e-6, n_substrate=1.5, n_superstrate=1.0, n_orders=30)
    st.add_layer(0.2e-6, eps_tensor_cell=cellA)
    st.add_layer(0.3e-6, eps_tensor_cell=cellB)
    res = st.set_source(WL, theta=theta).solve()
    return res.efficiencies(), lcB


@pytest.mark.parametrize("ang_deg", [0.0, 15.0])
def test_two_layer_stack_matches_rcwastack(ang_deg):
    theta = np.radians(ang_deg)
    (ro, rR, rT), lcB = _rcwa_two_layer(theta)
    pst = la.PMMStack(0.8e-6, n_substrate=1.5, n_superstrate=1.0, degree=22)
    pst.add_layer(0.2e-6, segments=[(0.25, GR), (0.5, 2.5 * GR), (0.25, GR)])
    pst.add_layer(0.3e-6, segments=[(0.1, 2.0 * GR), (0.4, lcB), (0.5, 2.0 * GR)])
    po, pR, pT, pJ = pst.set_source(WL, angle=theta).solve()

    roa = np.asarray(ro)
    rd = ({int(a): i for i, a in enumerate(roa)} if roa.ndim == 1
          else {(int(a), int(b)): i for i, (a, b) in enumerate(roa)})
    rkey = (lambda m: m) if roa.ndim == 1 else (lambda m: (m, 0))
    pd = {int(a): i for i, a in enumerate(po)}
    worst = 0.0
    for row in (0, 1):
        r00 = float(rR[row][rd[rkey(0)]])
        t00 = float(rT[row][rd[rkey(0)]])
        pr00 = float(pR[row][pd[0]])
        pt00 = float(pT[row][pd[0]])
        worst = max(worst, abs(r00 - pr00), abs(t00 - pt00))
    assert worst < 2e-3, f"PMMStack vs RCWAStack worst 0-order |d|={worst:.2e}"
    assert np.allclose(pR.sum(axis=1) + pT.sum(axis=1), 1.0, atol=1e-6)


def test_uniform_spacer_layer_and_energy():
    # a stack with a uniform spacer + two patterned layers conserves energy
    lc = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.4)
    st = la.PMMStack(0.8e-6, n_substrate=1.5, n_superstrate=1.0, degree=20)
    st.add_layer(0.1e-6, eps=2.1)                       # uniform spacer
    st.add_layer(0.3e-6, segments=[(0.25, GR), (0.5, 2.5 * GR), (0.25, GR)])
    st.add_layer(0.15e-6, segments=[(0.4, lc), (0.6, GR)])
    o, R, T, J = st.set_source(WL, angle=np.radians(30)).solve()
    assert np.allclose(R.sum(axis=1) + T.sum(axis=1), 1.0, atol=1e-6)


def test_all_vacuum_stack_transparent():
    # every layer vacuum (index-matched half-spaces) -> R=0, T=1
    st = la.PMMStack(0.8e-6, n_substrate=1.0, n_superstrate=1.0, degree=16)
    st.add_layer(0.2e-6, eps=1.0)
    st.add_layer(0.3e-6, segments=[(0.5, GR), (0.5, GR)])
    o, R, T, J = st.set_source(WL, angle=np.radians(20)).solve()
    assert np.allclose(R.sum(axis=1) + T.sum(axis=1), 1.0, atol=1e-9)
    assert np.max(np.abs(R)) < 1e-9


def test_isotropic_stack_decouples():
    # an all-isotropic stack has no cross-pol (diagonal Jones)
    st = la.PMMStack(0.8e-6, n_substrate=1.5, n_superstrate=1.0, degree=18)
    st.add_layer(0.2e-6, segments=[(0.5, 2.25), (0.5, 1.0)])
    o, R, T, J = st.set_source(WL, angle=np.radians(15)).solve()
    assert abs(J[0, 1]) < 1e-9 and abs(J[1, 0]) < 1e-9
    assert np.allclose(R.sum(axis=1) + T.sum(axis=1), 1.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
def test_accepts_vertical_out_of_plane_layer():
    # A VERTICAL out-of-plane layer is now accepted (see
    # test_single_out_of_plane_layer_stack_equals_binary); only a SLANTED
    # out-of-plane layer raises (test_slanted_out_of_plane_layer_raises).
    op = uniaxial_tensor(1.5, 1.7, np.pi / 4, phi=0.3)     # tilted -> off-plane
    la.PMMStack(0.8e-6).add_layer(0.1e-6, eps=op)          # no raise


def test_rejects_bad_add_layer_args():
    st = la.PMMStack(0.8e-6)
    with pytest.raises(ValueError, match="exactly one"):
        st.add_layer(0.1e-6)                               # neither eps nor segments
    with pytest.raises(ValueError, match="exactly one"):
        st.add_layer(0.1e-6, eps=2.0, segments=[(1.0, 2.0)])


def test_solve_requires_source_and_layers():
    with pytest.raises(ValueError, match="set_source"):
        la.PMMStack(0.8e-6).add_layer(0.1e-6, eps=2.0).solve()
    with pytest.raises(ValueError, match="at least one layer"):
        la.PMMStack(0.8e-6).set_source(WL).solve()


def test_pmmstack_exported():
    assert hasattr(la, "PMMStack") and "PMMStack" in la.__all__


# ---------------------------------------------------------------------------
# SLANTED layers (add_layer slant_angle) -- a stack may MIX vertical + slanted
# ---------------------------------------------------------------------------
def _eps_coupled():
    er = np.diag([2.25, 2.10, 2.25]).astype(_C)
    er[0, 1] = er[1, 0] = 0.15
    return er


@pytest.mark.parametrize("phi_deg", [30.0, 60.0])
@pytest.mark.parametrize("ang_deg", [0.0, 15.0])
def test_single_slanted_layer_stack_equals_binary(phi_deg, ang_deg):
    """A single slanted layer in a stack (vacuum/sub half-spaces) reproduces the
    binary pmm_jones_1d_slanted -- energy + zeroth-order Jones magnitude to
    ~1e-12.  (Segments reverse internally, so compare energy + |jones|, which are
    invariant to the order-label mirror.)"""
    er, eg = _eps_coupled(), GR
    phi, ang = np.radians(phi_deg), np.radians(ang_deg)
    # 'auto' routes this IN-PLANE slanted stack to the covariant generator;
    # compare LIKE-TO-LIKE against the covariant single-layer solver (both
    # spectral) -- 2-segment == binary covariant is exact, so they match tightly.
    st = la.PMMStack(1.0e-6, n_substrate=1.5, n_superstrate=1.0, degree=24)
    st.add_layer(0.5e-6, segments=[(0.5, er), (0.5, eg)], slant_angle=phi)
    o, R, T, J = st.set_source(0.633e-6, angle=ang).solve()
    ob, Rb, Tb, Jb = la.pmm_jones_1d_slanted(
        period=1.0e-6, eps_ridge=er, eps_groove=eg, n_substrate=1.5,
        n_superstrate=1.0, depth=0.5e-6, duty_cycle=0.5, wavelength=0.633e-6,
        slant_angle=phi, angle=ang, degree=24, stabilize=False,
        factorization="covariant")
    # The covariant 1-layer stack reproduces the covariant single-layer solver to
    # ~machine precision (same union-grid modes + half-spaces in the same oblique
    # frame; 2-segment == binary covariant is exact).
    assert np.max(np.abs(R - Rb)) < 1e-11
    assert np.max(np.abs(T - Tb)) < 1e-11
    assert np.max(np.abs(J - Jb)) < 1e-11


def test_covariant_stack_is_spectral_vs_convection():
    """A 2-layer UNIFORM-slant stack: 'auto' uses the covariant generator, which
    converges to the (high-degree convection) truth far faster than convection at
    matched low degree -- the spectral win in a multilayer stack.

    2026-08-15 (docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md, D3): the win used
    to be pinned as ``con_err > 2 * cov_err`` at the single working degree 20,
    with no derivation.  MEASURED cov_err 2.7033e-05 (Win py3.14/np2.4.4) /
    2.3361e-05 (WSL py3.12/np2.5.1) against con_err 7.5766e-05 on BOTH
    (bit-identical): ratios 2.803 / 3.243, i.e. 1.40x of headroom on the 2x bar
    with a 14% cross-build move, ALL of it in cov_err -- an eigendecomposition
    of a non-normal operator at degree 20, the LAPACK-sensitive half.  One
    degree, one ratio, one build-dependent number: a per-build magnitude
    asserted as a universal.

    WHAT THE LADDER ACTUALLY SHOWS -- and it is not what one might assume.  The
    obvious repair (score the two arms' convergence RATES and require the
    covariant one to fall faster per step) was tried and MEASURED FALSE: over
    degrees 12/16/20 the covariant errors are 3.4618e-05 / 2.9347e-05 /
    2.7033e-05 -- FLAT, a factor 1.28 across the whole ladder, with
    reference-free self-increments |R(12)-R(16)| = 2.778e-05 and |R(16)-R(20)|
    = 2.969e-05 that do not shrink at all -- while the convection errors are
    2.4722e-04 / 1.3103e-04 / 7.5766e-05, still contracting by ~0.55 per step.
    The covariant arm has ALREADY converged by degree 12, into the degree-40
    convection reference's own accuracy; it is the convection arm that is still
    descending.  A "covariant falls faster" assertion would therefore have
    pinned the opposite of the truth, and it is also why the matched-degree
    ratio SHRINKS with degree (7.14x at 12, 4.46x at 16, 2.80x at 20) -- the
    old 2x bar was measured at the ladder's worst point.

    THE BARS, all measured in this same process against the same reference, and
    none of them a chosen factor:

      (a) the ladder is measuring convergence at all -- the convection arm's
          error is strictly decreasing (2.4722e-04 > 1.3103e-04 > 7.5766e-05,
          steps of 1.89x and 1.73x);
      (b) the covariant arm is closer at EVERY matched degree -- margins 7.14x
          / 4.46x / 2.80x, versus 1.40x for the old bar;
      (c) the win is a DEGREE OFFSET, not a constant: covariant at the BOTTOM
          of the ladder still beats convection at the TOP, so eight extra
          degrees do not buy convection the covariant answer.  Margin
          min(con)/max(cov) = 7.5766e-05 / 3.4618e-05 = 2.19x.

    Two-sided: the binding margins (2.19x, 2.80x) are ~16x the 14% cross-build
    motion actually observed, and every one of them is a plain inequality
    between two numbers this run measured -- there is no magnitude to widen.
    Below, a covariant generator that had lost its advantage fails (b) and (c)
    outright, and a broken ladder (a solver flat in degree) fails (a)."""
    er = _eps_coupled()
    e2 = np.diag([3.0, 2.2, 2.5]).astype(_C)
    phi = np.radians(35.0)

    def mk(degree, fac):
        st = la.PMMStack(1.0e-6, n_substrate=1.5, n_superstrate=1.0,
                         degree=degree, factorization=fac)
        st.add_layer(0.3e-6, segments=[(0.5, er), (0.5, GR)], slant_angle=phi)
        st.add_layer(0.25e-6, segments=[(0.4, e2), (0.6, GR)], slant_angle=phi)
        return st.set_source(0.633e-6, angle=0.0).solve()

    ladder = (12, 16, 20)
    o_ref, R_ref, _T, _J = mk(40, "convection")          # converged truth
    cov, con = [], []
    for deg in ladder:
        o_v, R_v, Tv, _Jv = mk(deg, "auto")              # -> covariant
        o_c, R_c, _Tc, _Jc = mk(deg, "convection")
        cov.append(float(np.max(np.abs(R_v - R_ref))))
        con.append(float(np.max(np.abs(R_c - R_ref))))
        if deg == ladder[-1]:
            assert abs(R_v[0].sum() + Tv[0].sum() - 1.0) < 1e-3     # energy
    # (a) the ladder really is resolving convergence
    assert con[0] > con[1] > con[2], con
    # (b) covariant is closer at every matched degree
    assert all(a < b for a, b in zip(cov, con)), (cov, con)
    # (c) ... and by a whole degree offset: covariant at the bottom of the
    #     ladder still beats convection at the top
    assert max(cov) < min(con), (cov, con)


def test_covariant_stack_rejects_mixed_and_oop():
    """'covariant' requires a uniform non-zero slant and in-plane tensors; 'auto'
    falls back to convection for mixed-slant / vertical / out-of-plane stacks."""
    er = _eps_coupled()
    st = la.PMMStack(1.0e-6, factorization="covariant")
    st.add_layer(0.2e-6, eps=2.1)                          # vertical
    st.add_layer(0.3e-6, segments=[(0.5, er), (0.5, GR)], slant_angle=np.radians(30))
    with pytest.raises(NotImplementedError, match="UNIFORM"):
        st.set_source(0.633e-6).solve()
    # 'auto' on the same mixed-slant stack falls back to convection (energy ok)
    sta = la.PMMStack(1.0e-6, n_substrate=1.5, factorization="auto")
    sta.add_layer(0.2e-6, eps=2.1)
    sta.add_layer(0.3e-6, segments=[(0.5, er), (0.5, GR)], slant_angle=np.radians(30))
    o, R, T, J = sta.set_source(0.633e-6).solve()
    assert abs(R[0].sum() + T[0].sum() - 1.0) < 1e-3


def test_mixed_vertical_slanted_stack_conserves():
    """A stack mixing a VERTICAL spacer and a SLANTED grating (oblique incidence)
    conserves energy -- the general fwd/back cascade handles both families."""
    er, eg = _eps_coupled(), GR
    st = la.PMMStack(1.0e-6, n_substrate=1.5, n_superstrate=1.0, degree=20)
    st.add_layer(0.2e-6, eps=2.1)                                  # vertical spacer
    st.add_layer(0.4e-6, segments=[(0.5, er), (0.5, eg)],
                 slant_angle=np.radians(40.0))                     # slanted grating
    o, R, T, J = st.set_source(0.633e-6, angle=np.radians(10.0)).solve()
    assert abs(R[0].sum() + T[0].sum() - 1.0) < 1e-3
    assert abs(R[1].sum() + T[1].sum() - 1.0) < 1e-3


def test_slanted_multiregion_layer_in_stack_conserves():
    """A SLANTED 3-region layer (coupled middle) inside a stack conserves -- the
    multi-region metric generator composes in the stack cascade."""
    e1 = np.diag([2.25, 2.25, 2.25]).astype(_C)
    e2 = np.diag([4.0, 3.6, 4.0]).astype(_C)
    e2[0, 1] = e2[1, 0] = 0.3
    st = la.PMMStack(1.0e-6, n_substrate=1.5, n_superstrate=1.0, degree=22)
    st.add_layer(0.15e-6, eps=2.0)
    st.add_layer(0.4e-6, segments=[(0.2, e1), (0.5, e2), (0.3, GR)],
                 slant_angle=np.radians(35.0))
    o, R, T, J = st.set_source(0.633e-6, angle=np.radians(12.0)).solve()
    assert abs(R[0].sum() + T[0].sum() - 1.0) < 1e-3
    assert abs(R[1].sum() + T[1].sum() - 1.0) < 1e-3


# ---------------------------------------------------------------------------
# OUT-OF-PLANE layers (full 3x3, vertical) -- routed through the metric generator
# ---------------------------------------------------------------------------
def _eps_oop():
    er = np.diag([2.25, 2.10, 2.40]).astype(_C)
    er[0, 2] = er[2, 0] = 0.3
    er[1, 2] = er[2, 1] = 0.2
    return er


def test_single_out_of_plane_layer_stack_equals_binary():
    """A single VERTICAL out-of-plane layer in a stack reproduces the binary
    pmm_jones_1d (full-3x3) to ~1e-11 (energy + zeroth-order Jones magnitude)."""
    er, eg = _eps_oop(), GR
    st = la.PMMStack(1.0e-6, n_substrate=1.5, n_superstrate=1.0, degree=16,
                     elements_per_region=6)
    st.add_layer(0.5e-6, segments=[(0.5, er), (0.5, eg)])
    o, R, T, J = st.set_source(0.633e-6).solve()
    ob, Rb, Tb, Jb = la.pmm_jones_1d(1.0e-6, er, eg, 1.5, 1.0, 0.5e-6, 0.5,
                                     0.633e-6, degree=16, elements_per_region=6)
    # v5.14: cross-path consistency, not bit-identity -- the metric paths'
    # flux-split selections within degenerate pairs are BLAS-build-sensitive
    # (a CI build diverged 0.036 under the LEGACY branch; both paths now use
    # the robust branch, but the gauge can still differ at eig noise).
    assert abs((R[0].sum() + T[0].sum()) - (Rb[0].sum() + Tb[0].sum())) < 1e-6
    assert abs(abs(J[0, 0]) - abs(Jb[0, 0])) < 1e-6


def test_mixed_inplane_out_of_plane_stack_conserves():
    """A stack mixing an in-plane spacer and a VERTICAL out-of-plane grating
    conserves energy (the general cascade routes the out-of-plane layer through
    the metric generator)."""
    er, eg = _eps_oop(), GR
    st = la.PMMStack(1.0e-6, n_substrate=1.5, n_superstrate=1.0, degree=14,
                     elements_per_region=4)
    st.add_layer(0.2e-6, eps=2.1)
    st.add_layer(0.4e-6, segments=[(0.5, er), (0.5, eg)])
    o, R, T, J = st.set_source(0.633e-6).solve()
    assert abs(R[0].sum() + T[0].sum() - 1.0) < 1e-3
    assert abs(R[1].sum() + T[1].sum() - 1.0) < 1e-3


def test_slanted_out_of_plane_layer_equals_single_layer_solver():
    """A SLANTED out-of-plane layer now solves: a 1-layer stack reproduces the
    (per-order-validated) single-layer pmm_jones_1d_slanted to ~1e-12."""
    er, eg = _eps_oop(), GR
    sl = np.radians(30.0)
    st = la.PMMStack(1.0e-6, n_substrate=1.5, n_superstrate=1.0, degree=16)
    st.add_layer(0.5e-6, segments=[(0.5, er), (0.5, eg)], slant_angle=sl)
    o, R, T, J = st.set_source(0.633e-6).solve()
    oR, ReR, TeR, JR = la.pmm_jones_1d_slanted(
        1.0e-6, er, eg, 1.5, 1.0, 0.5e-6, 0.5, 0.633e-6, sl, degree=16,
        stabilize=False)
    assert np.max(np.abs(R - ReR)) < 1e-10
    assert np.max(np.abs(T - TeR)) < 1e-10
    assert np.max(np.abs(J - JR)) < 1e-10


def test_mixed_slanted_out_of_plane_stack_conserves_energy():
    """A MIXED stack (vertical isotropic + slanted out-of-plane + vertical
    out-of-plane) at oblique incidence cascades and conserves energy."""
    er, eg = _eps_oop(), GR
    st = la.PMMStack(1.0e-6, n_substrate=1.5, n_superstrate=1.0, degree=14)
    st.add_layer(0.15e-6, eps=2.1)
    st.add_layer(0.3e-6, segments=[(0.5, er), (0.5, eg)],
                 slant_angle=np.radians(30.0))
    st.add_layer(0.2e-6, segments=[(0.4, er), (0.6, eg)])
    o, R, T, J = st.set_source(0.633e-6, angle=np.radians(12.0)).solve()
    assert abs(R[0].sum() + T[0].sum() - 1.0) < 1e-3
    assert abs(R[1].sum() + T[1].sum() - 1.0) < 1e-3
    assert np.all(np.isfinite(J))
