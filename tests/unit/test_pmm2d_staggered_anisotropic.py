"""Stage A -- IN-PLANE (block-form) anisotropic permittivity for the PURE
(no-floor) staggered 2-D PMM: ``pmm_jones_2d_staggered`` and the tensor
``PMM2DStackPure``.

Gates G1-G10 of ``docs/audits/PLAN_PMM2D_STAGGERED_ANISOTROPIC_2026_09_09.md``.
EVERY bar below is DERIVED from a measurement made during this build on
2026-09-09 (Windows 11, py3.14, numpy 2.x, OMP=1); the measurement and its
table row in ``docs/audits/BUILD_PMM2D_STAGGERED_ANISOTROPIC_2026_09_09.md``
are cited in each assertion's comment.  Nothing here pins a cross-build value:
the comparisons are two-arm, same-build, and the oracle-referenced bars carry
decades of gap on both sides.

Formulation: Granet, J. Opt. Soc. Am. A 40, 652 (2023), Eqs. 23-25 with the
general block-form ``[eps_t]`` (Appendix A Eqs. 40, 41, 44).  The paper uses
``exp(+i w t)``; this module is PUBLIC ``exp(-i w t)`` end to end, so every
tensor quoted from the paper is CONJUGATED here.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumenairy.elements.berreman import berreman_jones_1d  # noqa: E402
from lumenairy.elements.pmm import (  # noqa: E402
    PMM2DStackPure,
    pmm_jones_1d,
    pmm_jones_2d,
)
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE,
    _homog_geom_cache,
    pmm_efficiency_2d_staggered,
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa import rcwa_jones_2d  # noqa: E402
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

# --------------------------------------------------------------------------- #
# fixtures (grids <= (3,3), M <= 8 -- the plan's test-cost rule)
# --------------------------------------------------------------------------- #
_WL = 0.55e-6
_P = 0.70e-6
_DEP = 0.28e-6
#: rotated in-plane uniaxial director: REAL-SYMMETRIC off-diagonal (e12 = e21)
_LC = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.55)
_LC_M = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=-0.55)
#: GYROTROPIC (Hermitian, lossless): e12 = -e21 = +0.5i in the PUBLIC gauge
_GYRO = np.array([[2.25, 0.5j, 0.0], [-0.5j, 2.25, 0.0], [0.0, 0.0, 2.0]],
                 dtype=complex)
_ISO = 4.0 * np.eye(3, dtype=complex)


def _cell(host, pillar, n=2):
    """``(n, n, 3, 3)`` cell: ``pillar`` in segment (0, 0), ``host`` elsewhere."""
    c = np.empty((n, n, 3, 3), dtype=complex)
    c[:] = host
    c[0, 0] = pillar
    return c


def _uniform(t33, n=2):
    c = np.empty((n, n, 3, 3), dtype=complex)
    c[:] = t33
    return c


def _promote(scalar_map):
    t = np.zeros(np.shape(scalar_map) + (3, 3), dtype=complex)
    for i in range(3):
        t[..., i, i] = scalar_map
    return t


def _idx(orders):
    return {(int(a), int(b)): i for i, (a, b) in enumerate(np.asarray(orders))}


# =========================================================================== #
# G1 -- REDUCTION: a scalar cell and the tensor ``e * I`` cell are the SAME
#       discretization, BIT FOR BIT (two arms, one build).
# =========================================================================== #
def test_g1_scalar_and_tensor_eye_operators_are_bit_identical():
    """The tensor dispatch must not perturb the shipped isotropic assembly.

    MEASURED 2026-09-09 (build doc table T1): every retained operator agrees
    EXACTLY (max|diff| = 0.0, identical sha256) between the scalar arm and the
    ``e * I`` tensor arm at (2,2)/M=6, oblique Bloch phases.  This is the
    strongest available statement -- stronger than the plan's fallback
    "rel. diff ~1e-14" -- and it is a decision (identity), not a reading.
    """
    cell = np.array([[6.25, 1.0], [1.0, 2.25]], dtype=complex)
    kw = dict(alpha0x=0.31, alpha0y=-0.17, k0=2 * np.pi / 0.62)
    a = Granet2DTransverseE(1.1, 1.1, 2, 2, 6, cell, **kw)
    b = Granet2DTransverseE(1.1, 1.1, 2, 2, 6, _promote(cell), **kw)
    for name in ("Lmat", "Rmat", "Stt", "Schur"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    for k in (0, 1):
        assert np.array_equal(a.Et_blocks[k], b.Et_blocks[k]), k
    # the scalar path retains NOTHING extra (audit P3-37); the tensor path
    # retains exactly the two Eq.40 mixed masses, which are EXACTLY zero here.
    assert a.Et_offdiag is None
    assert b.Et_offdiag is not None
    for blk in b.Et_offdiag:
        assert np.array_equal(blk, np.zeros_like(blk))
    for dead in ("Curl", "Kzt", "Ktz", "G3", "Meps33"):
        assert not hasattr(a, dead) and not hasattr(b, dead), dead


def test_g1_public_entries_agree_on_a_scalar_cell():
    """The scalar entry (``polarization='tm'`` = incident ``E_x`` at normal
    incidence) and the tensor Jones entry on the promoted ``e * I`` cell.

    MEASURED 2026-09-09 (build doc T1): max|dR| = max|dT| = 0.0 -- the two
    entries take bit-identical arithmetic routes on this input.  The bar is
    therefore exact equality; a regression that merely perturbs the tensor
    dispatch by 1 ULP fails it, which is the point.
    """
    cell = np.array([[6.25, 1.0], [1.0, 2.25]], dtype=complex)
    g = dict(period_x=0.8e-6, period_y=0.8e-6, depth=0.3e-6,
             wavelength=0.633e-6, n_substrate=1.5, n_superstrate=1.0)
    o1, R1, T1 = pmm_efficiency_2d_staggered(eps_cell=cell, degree=6,
                                             n_orders=3, polarization="tm",
                                             **g)
    o2, R2, T2, _J = pmm_jones_2d_staggered(eps_cell=_promote(cell), degree=6,
                                            n_orders=3, **g)
    assert np.array_equal(o1, o2)
    assert np.array_equal(R1, R2[0])
    assert np.array_equal(T1, T2[0])
    # ... and a SCALAR (Nx, Ny) map handed to the tensor entry is promoted
    # internally to the same thing.
    _o3, R3, _T3, _J3 = pmm_jones_2d_staggered(eps_cell=cell, degree=6,
                                               n_orders=3, **g)
    assert np.array_equal(R3[0], R2[0])


# =========================================================================== #
# G2 -- PUBLISHED ORACLE (Granet 2023 Table 2, Fig. 4 anisotropic grating).
#
# HONEST RECORD (2026-09-09, build doc T2).  The ABSOLUTE Table-2 efficiencies
# are NOT reproduced by this build under ANY reading tried -- 32 combinations
# of {axis assignment (2.4,1.4)/(1.4,2.4)} x {which tensor is the pillar} x
# {conjugated / not} x {eps_sub = 1+5i / 1-5i}, each under four transmitted-
# efficiency definitions, plus vacuum-host and index-vs-permittivity substrate
# readings: the best maximum deviation over the four quoted orders is 2.65e-2
# (paper (0,0) = 0.2979 vs this build 0.528 under the Poynting-flux definition,
# or 0.302 under a bare |t|^2).  Per the plan this is RECORDED, NOT TUNED, and
# the physics claim is carried by G3/G4/G5 (Berreman to ~1e-14, the 1-D
# engines to ~4e-6, and the two independent 2-D engines to ~6e-4).
#
# What the paper DOES discriminate, and what is asserted here, is the SIGN of
# the gyrotropic +/- order asymmetry: it is invisible to every energy check
# and it is exactly the observable a wrong e12/e21 sign (or a missing
# conjugation bridge) flips.
# =========================================================================== #
_GRANET_B = np.conj(np.array([[2.25, -0.5j, 0.0], [0.5j, 2.25, 0.0],
                              [0.0, 0.0, 2.0]], dtype=complex))
_GRANET_A = np.conj(_GRANET_B)
_LAM = 1.0e-6


def _granet(pillar, host):
    o, _R, T, _J = pmm_jones_2d_staggered(
        2.4 * _LAM, 1.4 * _LAM, _cell(host, pillar), np.sqrt(1.0 + 5.0j), 1.0,
        _LAM, _LAM, degree=6, n_orders=3)
    i = _idx(o)
    return float(T[0][i[(1, 1)]]), float(T[0][i[(-1, 1)]])


def test_g2_gyrotropic_order_asymmetry_sign_matches_the_paper():
    """Granet Table 2 has T(1,1) = 0.0268 > T(-1,1) = 0.0139 (SEM) and
    0.0269 > 0.0137 (FMM) -- a robust SIGN, quoted by two independent methods.

    MEASURED 2026-09-09 (build doc T2): with the paper's tensor CONJUGATED
    into this module's PUBLIC exp(-iwt) gauge, T(1,1) = 0.021328 and
    T(-1,1) = 0.016157, so the difference is +5.171e-03; with the tensor left
    UNCONJUGATED the two values swap exactly and the difference is -5.171e-03.
    Bar 1e-3: two-sided, 5.2x below the measured signal and many decades above
    the build spread of a difference of two O(0.02) numbers.  The energy
    closure is IDENTICAL on both arms (a gyrotropic sign error is invisible to
    it), so this is the only gate that sees it.
    """
    t_p, t_m = _granet(_GRANET_B, _GRANET_A)
    assert t_p - t_m > 1e-3, (t_p, t_m)
    # unconjugated (i.e. the paper's exp(+iwt) tensor used raw) REVERSES it
    u_p, u_m = _granet(np.conj(_GRANET_B), np.conj(_GRANET_A))
    assert u_m - u_p > 1e-3, (u_p, u_m)


def test_g2_control_no_gyrotropy_no_asymmetry():
    """CONTROL: the same half-filled cell with REAL, non-gyrotropic tensors is
    mirror-symmetric up to a translation, so T(1,1) == T(-1,1) exactly.

    MEASURED 2026-09-09 (build doc T2): |T(1,1) - T(-1,1)| = 1.02e-15 for an
    isotropic pillar in an isotropic host -- 12 decades under the 5.171e-03
    gyrotropic signal above.  Bar 1e-9.
    """
    t_p, t_m = _granet(_ISO, 2.25 * np.eye(3, dtype=complex))
    assert abs(t_p - t_m) < 1e-9, (t_p, t_m)


# =========================================================================== #
# G3 -- UNIFORM in-plane tensor slab vs the EXACT Berreman 4x4 oracle.
# =========================================================================== #
_G3 = dict(period=0.40e-6, wl=1.0e-6, dep=0.55e-6, nsub=1.5, nsup=1.0)


def _g3_residual(t33, nseg, M, theta, phi):
    _o, R, T, J = pmm_jones_2d_staggered(
        _G3["period"], _G3["period"], _uniform(t33, nseg), _G3["nsub"],
        _G3["nsup"], _G3["dep"], _G3["wl"], degree=M, n_orders=2, theta=theta,
        phi=phi)
    Rb, Tb, jr, _jt = berreman_jones_1d([(t33, _G3["dep"])], _G3["nsub"],
                                        _G3["nsup"], _G3["wl"], angle=theta,
                                        phi=phi)
    return max(float(np.max(np.abs(R.sum(axis=1) - Rb))),
               float(np.max(np.abs(T.sum(axis=1) - Tb))),
               float(np.max(np.abs(J - jr))))


@pytest.mark.parametrize("name,t33", [("lc", _LC), ("gyro", _GYRO)])
@pytest.mark.parametrize("theta,phi", [(0.0, 0.0),
                                       (25 * np.pi / 180, 0.0),
                                       (25 * np.pi / 180, 40 * np.pi / 180)])
def test_g3_uniform_tensor_slab_matches_berreman(name, t33, theta, phi):
    """A UNIFORM in-plane tensor region is smooth, so the staggered basis
    converges SPECTRALLY to the analytic Berreman 4x4 answer -- R, T AND the
    complex Jones, in the PUBLIC gauge with NO conjugation anywhere.

    MEASURED 2026-09-09 at M=7, (2,2) grid (build doc T3): the worst residual
    over the six (tensor, incidence) combinations is 9.33e-14 (R, T and Jones
    together; the six values are 9.33e-14 / 4.08e-14 / 5.24e-14 for the LC
    tensor and 6.54e-14 / 1.67e-14 / 5.88e-14 for the gyrotropic one).  Bar =
    1e2 x that, rounded to 1e-11: 107x above the worst measured value and
    decades below any physics-level error.
    """
    assert _g3_residual(t33, 2, 7, theta, phi) < 1e-11


def test_g3_convergence_is_two_sided_in_M():
    """TWO-SIDED: not merely "small at M=7" -- the M=5 -> M=7 residual must
    DROP, which is what makes the M=7 number a convergence statement rather
    than a coincidence.

    MEASURED 2026-09-09 (build doc T3), gyrotropic slab at theta=25 deg,
    phi=0, (2,2) grid: 9.711e-11 (M=5) -> 1.665e-14 (M=7), a factor 5.8e3.
    Bar 10x, ~580x of headroom.  (The NORMAL-incidence configurations are
    already at roundoff by M=5, so the claim is made where the ladder is
    resolvable.)
    """
    r5 = _g3_residual(_GYRO, 2, 5, 25 * np.pi / 180, 0.0)
    r7 = _g3_residual(_GYRO, 2, 7, 25 * np.pi / 180, 0.0)
    assert r7 < r5 / 10.0, (r5, r7)


def test_g3_multisegment_grid_gives_the_same_uniform_answer():
    """The uniform tensor answer must not depend on how many segments the
    (physically uniform) cell is cut into -- the tensor assembly's per-segment
    weighting is exact.

    MEASURED 2026-09-09 (build doc T3): (3,3)/M=7 conical residual 7.11e-14
    vs (2,2)/M=7 5.24e-14 -- the same roundoff plateau.  Bar 1e-11 as above.
    """
    assert _g3_residual(_LC, 3, 7, 25 * np.pi / 180, 40 * np.pi / 180) < 1e-11


# --------------------------------------------------------------------------- #
# G3 companion -- FAIL-BEFORE: each NEW term must be LOAD-BEARING.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("term", ["mass", "kzt", "lhh"])
def test_g3_each_new_tensor_term_is_load_bearing(term, monkeypatch):
    """Right-conclusion-wrong-mechanism guard.  Each of the three things the
    tensor path adds is zeroed IN TURN (at the class / module level, no source
    edit) and the G3 Berreman residual is re-measured.  A term that is not
    load-bearing would leave that residual at its 1e-14 reference.

    MEASURED 2026-09-09 (build doc T3b), (2,2) grid, M=7, conical
    theta=25 deg / phi=40 deg, residual over R, T and the complex Jones:

        arm                              LC slab      gyrotropic slab
        all terms present (reference)    5.240e-14    5.884e-14
        Eq.40 mixed masses  -> 0         4.101e-02    7.749e-02
        Eq.44 second K_zt term -> 0      5.697e-03    4.058e-02
        Eq.25 Lhh mixed blocks -> 0      1.049e-01    1.613e-01

    Bar 1e-4 on the broken arm: 1.7 decades under the smallest measured break
    (5.70e-03) and 10 decades over the reference -- so the assertion cannot be
    satisfied by round-off drift in either direction.
    """
    import lumenairy.elements.pmm.stack2d_pure as SP
    from lumenairy.elements.pmm import twod_staggered as TS

    ew, ed, rm = (TS.Granet2DTransverseE._eps_weighted,
                  TS.Granet2DTransverseE._eps_dir, TS._region_modes)

    def no_mixed_mass(self, refx_pair, refy_pair, wmap=None):
        out = ew(self, refx_pair, refy_pair, wmap)
        # the Eq.40 MIXED blocks are the only ones whose 1-D set pairs differ
        return np.zeros_like(out) if refx_pair[2] is not refx_pair[3] else out

    def no_second_kzt(self, bx, lx, opx, rx, by, ly, opy, ry, wmap=None):
        out = ed(self, bx, lx, opx, rx, by, ly, opy, ry, wmap)
        # the second Eq.44 term per column carries the derivative on the OTHER
        # axis than the shipped isotropic one
        if (opx, rx, opy, ry) in (("m", "B", "dL", "Btilde"),
                                  ("dL", "Btilde", "m", "B")):
            return np.zeros_like(out)
        return out

    def no_lhh_mixed(solver):
        saved, solver.Et_offdiag = solver.Et_offdiag, None
        try:
            return rm(solver)
        finally:
            solver.Et_offdiag = saved

    if term == "mass":
        monkeypatch.setattr(TS.Granet2DTransverseE, "_eps_weighted",
                            no_mixed_mass)
    elif term == "kzt":
        monkeypatch.setattr(TS.Granet2DTransverseE, "_eps_dir", no_second_kzt)
    else:
        monkeypatch.setattr(TS, "_region_modes", no_lhh_mixed)
        # stack2d_pure imported the name by value
        monkeypatch.setattr(SP, "_region_modes", no_lhh_mixed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        broken = max(_g3_residual(t33, 2, 7, 25 * np.pi / 180,
                                  40 * np.pi / 180)
                     for t33 in (_LC, _GYRO))
    assert broken > 1e-4, (term, broken)


# =========================================================================== #
# G4 -- 1-D REDUCTION: a y-uniform anisotropic stripe grating against the two
#       independent 1-D engines, PER ORDER, both polarizations, plus Jones.
# =========================================================================== #
_G4_P, _G4_WL, _G4_DEP = 0.90e-6, 0.55e-6, 0.30e-6
_G4_RIDGE = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.55)
_G4_GROOVE = 2.10 * np.eye(3, dtype=complex)


def _g4_stripe():
    c = np.empty((2, 2, 3, 3), dtype=complex)
    c[:] = _G4_GROOVE
    c[0, :] = _G4_RIDGE
    return c


@pytest.fixture(scope="module")
def g4_oracle():
    return pmm_jones_1d(_G4_P, _G4_RIDGE, _G4_GROOVE, 1.5, 1.0, _G4_DEP, 0.5,
                        _G4_WL, angle=0.22, degree=16, stabilize=False)


def _g4_residual(M, oracle):
    o1, R1, T1, J1 = oracle
    o, R, T, J = pmm_jones_2d_staggered(_G4_P, _G4_P, _g4_stripe(), 1.5, 1.0,
                                        _G4_DEP, _G4_WL, degree=M, n_orders=3,
                                        theta=0.22, phi=0.0)
    sel = np.asarray(o)[:, 1] == 0
    two = {int(m): i for i, m in zip(np.arange(len(o))[sel],
                                     np.asarray(o)[sel, 0])}
    one = {int(m): i for i, m in enumerate(np.asarray(o1).ravel())}
    common = [m for m in one if m in two]
    dRT = max(max(abs(R[r][two[m]] - R1[r][one[m]]),
                  abs(T[r][two[m]] - T1[r][one[m]]))
              for m in common for r in (0, 1))
    forb = float(max(np.max(np.abs(R[:, ~sel])), np.max(np.abs(T[:, ~sel]))))
    return dRT, float(np.max(np.abs(J - J1))), forb


def test_g4_stripe_grating_matches_the_1d_engines(g4_oracle):
    """A y-uniform anisotropic stripe has NO corner in the 2-D field, so the
    staggered solve converges cleanly onto the exact 1-D PMM.

    MEASURED 2026-09-09 (build doc T4), theta = 0.22 rad, ladder over M:

        M     per-order max|dR|,|dT|     max|dJones|
        5           2.180e-03             1.025e-03
        6           2.411e-04             1.196e-04
        7           2.598e-05             2.874e-05
        8           4.306e-06             9.088e-06

    The residual is DETERMINISTIC discretization error (strictly monotone down
    the ladder), not build noise, so the bar is 3x the M=8 value: 1.3e-05 on
    R/T and 2.7e-05 on the Jones.  The two 1-D oracles (pmm_jones_1d degree 16
    and rcwa_jones_1d with 81 orders) agree with EACH OTHER to 5.22e-07 on R/T
    and 1.51e-06 on the Jones, i.e. 25x / 18x under the bar, so the oracle's
    own floor is not what is being measured.
    """
    dRT, dJ, _forb = _g4_residual(8, g4_oracle)
    assert dRT < 1.3e-5, dRT
    assert dJ < 2.7e-5, dJ


def test_g4_convergence_ladder_is_monotone(g4_oracle):
    """TWO-SIDED companion to the bar above: the residual must FALL down the
    ladder (measured 2.180e-03 -> 4.306e-06 from M=5 to M=8, a factor 506).
    Bar: a drop of at least 10x between the ladder's ends, i.e. 50x of
    headroom.
    """
    d5 = _g4_residual(5, g4_oracle)[0]
    d8 = _g4_residual(8, g4_oracle)[0]
    assert d8 < d5 / 10.0, (d5, d8)


def test_g4_y_momentum_is_conserved(g4_oracle):
    """A y-uniform cell cannot scatter into ``n != 0``: those orders are
    forbidden by transverse-momentum conservation, and the staggered tensor
    assembly must not leak into them.

    MEASURED 2026-09-09 (build doc T4): max efficiency over all n != 0 orders
    = 4.86e-27 at M=8, moving only over 6.73e-29 .. 4.86e-27 down the whole M
    ladder -- i.e. a round-off floor, not a convergent quantity.  Bar 1e-20: 6
    decades above the measurement and 17 decades below the smallest physically
    meaningful order here (~1e-3).
    """
    assert _g4_residual(8, g4_oracle)[2] < 1e-20


# =========================================================================== #
# G5 -- THREE ENGINES on a genuinely 2-D anisotropic cell + the NO-FLOOR
#       property, two-sided.
# =========================================================================== #
def _g5_cell(up=1, reverse=False):
    n = 2 * up
    c = np.empty((n, n, 3, 3), dtype=complex)
    host, pill = (_ISO, _LC) if reverse else (_LC, _ISO)
    c[:] = host
    c[:up, :up] = pill
    return c


def _order0(o, A):
    i = int(np.where((np.asarray(o)[:, 0] == 0)
                     & (np.asarray(o)[:, 1] == 0))[0][0])
    return np.array([float(A[0][i]), float(A[1][i])])


def test_g5_three_engines_agree_on_a_2d_anisotropic_cell():
    """The no-floor staggered tensor solver, the FMM-floored hybrid PMM
    (``pmm_jones_2d``) and ``rcwa_jones_2d`` -- three independent
    discretizations of the same physics -- on an isotropic pillar in a
    rotated-uniaxial LC host.

    MEASURED 2026-09-09 (build doc T5), order-0 R and T, both polarizations:
    hybrid(degree 11, n_orders 13) vs staggered(M=7) 1.2e-04 (R) / 9.3e-05
    (T); rcwa(n_orders 13) vs staggered 1.1e-04 (R) / 5.8e-04 (T).  Bar
    5e-03 = ~8x the largest measured pairwise spread, and it is the FOURIER
    engines' own floor that sets it (the staggered arm closes energy to
    2.5e-08 while the hybrid closes to 2.9e-04 and both Fourier arms still
    move with n_orders).
    """
    c = _g5_cell()
    o, R, T, _J = pmm_jones_2d_staggered(_P, _P, c, 1.5, 1.0, _DEP, _WL,
                                         degree=7, n_orders=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oh, Rh, Th, _Jh = pmm_jones_2d(_P, _P, c, 1.5, 1.0, _DEP, _WL,
                                       degree=11, n_orders=13)
    orc, Rr, Tr, _Jr = rcwa_jones_2d(_P, _P, _g5_cell(32), 1.5, 1.0, _DEP,
                                     _WL, n_orders_x=13, n_orders_y=13)
    base = (_order0(o, R), _order0(o, T))
    for tag, (oo, RR, TT) in (("hybrid", (oh, Rh, Th)),
                              ("rcwa", (orc, Rr, Tr))):
        assert np.max(np.abs(_order0(oo, RR) - base[0])) < 5e-3, tag
        assert np.max(np.abs(_order0(oo, TT) - base[1])) < 5e-3, tag


def test_g5_no_fourier_floor_two_sided():
    """THE defining property of the pure engine, carried into the tensor path:
    the answer does not move with ``n_orders`` (the Rayleigh set is a
    once-only FORWARD far-field projection, not a solve basis), whereas the
    hybrid -- which projects the layer into a truncated Fourier basis BEFORE
    the eigensolve -- does.

    MEASURED 2026-09-09 (build doc T5), n_orders 4 -> 8 at fixed resolution,
    order-0 R and T on the LC-host cell: staggered 3.89e-15, hybrid 9.69e-03
    -- a ratio of 2.5e12.  Bars: staggered < 1e-10 (5 decades of headroom over
    the measurement, and above the float64 reassociation limit of the
    different-length reductions the two n_orders perform); hybrid > 1e-4 (2
    decades under its measured motion).  Both sides asserted: "no floor" is
    only a claim if the floored engine is shown to be floored.
    """
    c = _g5_cell()
    stag, hyb = {}, {}
    for nor in (4, 8):
        o, R, T, _J = pmm_jones_2d_staggered(_P, _P, c, 1.5, 1.0, _DEP, _WL,
                                             degree=7, n_orders=nor)
        stag[nor] = np.concatenate([_order0(o, R), _order0(o, T)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            oh, Rh, Th, _Jh = pmm_jones_2d(_P, _P, c, 1.5, 1.0, _DEP, _WL,
                                           degree=9, n_orders=nor)
        hyb[nor] = np.concatenate([_order0(oh, Rh), _order0(oh, Th)])
    assert np.max(np.abs(stag[4] - stag[8])) < 1e-10
    assert np.max(np.abs(hyb[4] - hyb[8])) > 1e-4


# =========================================================================== #
# G6 -- ENERGY CLOSURE, two-sided (Hermitian closes; non-Hermitian absorbs).
# =========================================================================== #
@pytest.mark.parametrize("name,host", [("lc", _LC), ("gyro", _GYRO)])
def test_g6_hermitian_tensor_cell_closes(name, host):
    """A HERMITIAN permittivity absorbs nothing -- including the GYROTROPIC
    tensor, whose ``e12 = -e21 = i b`` is anti-symmetric but Hermitian.

    MEASURED 2026-09-09 (build doc T6) at M=8, (2,2) grid, both incident
    polarizations: |sum R + sum T - 1| = 6.3e-09 (LC host) and 3.9e-08
    (gyrotropic host).  Bar = 1e2 x the worse of those, rounded to 1e-05:
    decades above the measurement and decades below the ~1e-3 scale at which
    the corner-capped modal error of this cell class shows up at low M.
    """
    _o, R, T, _J = pmm_jones_2d_staggered(_P, _P, _cell(host, _ISO), 1.5, 1.0,
                                          _DEP, _WL, degree=8, n_orders=4)
    tot = R.sum(axis=1) + T.sum(axis=1)
    assert np.max(np.abs(tot - 1.0)) < 1e-5, tot


def test_g6_closure_improves_with_M_and_a_lossy_cell_absorbs():
    """TWO-SIDED.  (a) the closure must FALL with M (measured 1.17e-05 at M=6
    -> 3.93e-08 at M=8 for the gyrotropic cell, a factor 298; bar 10x).  (b) a
    NON-Hermitian (absorbing) cell must close BELOW 1 by a real margin, and no
    unity is ever claimed for it: measured 1 - (R+T) = 0.2270 / 0.2445 for the
    two polarizations at M=8; bar 0.05, ~4.5x under the measurement and
    decades above the 1e-8 closure of the lossless arm.
    """
    kw = dict(period_x=_P, period_y=_P, n_substrate=1.5, n_superstrate=1.0,
              depth=_DEP, wavelength=_WL, n_orders=4)
    dev = {}
    for M in (6, 8):
        _o, R, T, _J = pmm_jones_2d_staggered(eps_cell=_cell(_GYRO, _ISO),
                                              degree=M, **kw)
        dev[M] = float(np.max(np.abs(R.sum(axis=1) + T.sum(axis=1) - 1.0)))
    assert dev[8] < dev[6] / 10.0, dev
    lossy = _cell(_LC, _ISO + 0.8j * np.eye(3))
    _o, R, T, _J = pmm_jones_2d_staggered(eps_cell=lossy, degree=8, **kw)
    tot = R.sum(axis=1) + T.sum(axis=1)
    assert np.max(tot) < 1.0 - 0.05, tot


# =========================================================================== #
# G7 -- DISCRETE SYMMETRIES.  These are EXACT symmetries of the
#       discretization, and they are what a swapped e12/e21 placement breaks.
# =========================================================================== #
_S3 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
_S2 = np.array([[0, 1], [1, 0]], dtype=float)
_M2 = np.diag([1.0, -1.0])
_G7 = dict(n_substrate=1.5, n_superstrate=1.0, depth=_DEP, wavelength=_WL,
           degree=6, n_orders=3)


def _transpose_cell(A):
    B = np.empty_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            B[i, j] = _S3 @ A[j, i] @ _S3
    return B


def _transpose_residual(A, B):
    """max |R_B[pol][(m,n)] - R_A[1-pol][(n,m)]| for the x<->y transposed pair
    (periods swapped with the cell), and the Jones residual."""
    oA, RA, TA, JA = pmm_jones_2d_staggered(_P, 1.3 * _P, A, **_G7)
    oB, RB, TB, JB = pmm_jones_2d_staggered(1.3 * _P, _P, B, **_G7)
    ia, ib = _idx(oA), _idx(oB)
    dev = max(max(abs(RB[r][ib[(m, n)]] - RA[1 - r][ia[(n, m)]]),
                  abs(TB[r][ib[(m, n)]] - TA[1 - r][ia[(n, m)]]))
              for (m, n) in ib if (n, m) in ia for r in (0, 1))
    return dev, float(np.max(np.abs(JB - _S2 @ JA @ _S2)))


def test_g7_xy_transpose_symmetry():
    """Transposing the cell about x = y (swap the grid axes, the periods, AND
    e11<->e22, e12<->e21) must map order (m,n) -> (n,m) and swap the Jones
    rows and columns -- exactly, on a symmetric discretization.

    MEASURED 2026-09-09 (build doc T7), LC host + isotropic pillar at M=6:
    per-order residual 2.07e-14, Jones residual 1.07e-14.  Bar = 1e2 x the
    larger, rounded up to 1e-11: 480x over the measurement, and 8 decades
    under the 3.64e-03 that the swapped-block control below produces.
    """
    A = _cell(_LC, _ISO)
    dev, dJ = _transpose_residual(A, _transpose_cell(A))
    assert dev < 1e-11 and dJ < 1e-11, (dev, dJ)


def test_g7_mixed_block_placement_control_gyrotropic():
    """THE placement control.  Interchanging ``e12`` and ``e21`` in ONE arm
    must BREAK the transpose symmetry -- and it does, but only for a tensor
    whose off-diagonal is NOT symmetric.  The rotated LC has e12 = e21 (the
    swap is a no-op there, measured 6.6e-15, which is why it cannot be the
    control); the GYROTROPIC tensor has e12 = -e21 and is the discriminator.

    MEASURED 2026-09-09 (build doc T7) at M=6: correct placement 1.30e-14,
    swapped placement 3.64e-03 (its Jones residual 8.14e-02) -- 11 decades
    apart.  Bars: correct < 1e-11 (as above), swapped > 1e-7 (4.6 decades
    under the measured break, 4 decades over the correct arm) -- a two-sided,
    fail-before demonstration.  The same swap on the LC cell measures
    2.55e-14, i.e. no break at all, which is why the gyrotropic tensor is the
    discriminator.
    """
    A = _cell(_GYRO, _ISO)
    B = _transpose_cell(A)
    ok, _ = _transpose_residual(A, B)
    Aw = A.copy()
    Aw[..., 0, 1], Aw[..., 1, 0] = A[..., 1, 0].copy(), A[..., 0, 1].copy()
    bad, _ = _transpose_residual(Aw, B)
    assert ok < 1e-11, ok
    assert bad > 1e-7, bad


def test_g7_y_mirror_symmetry():
    """Mirroring the cell in y (and the director azimuth phi -> -phi, i.e.
    e12 -> -e12, e21 -> -e21) must map order (m,n) -> (m,-n) and flip the sign
    of the OFF-DIAGONAL Jones entries.

    MEASURED 2026-09-09 (build doc T7) at M=6: per-order residual 2.85e-15,
    Jones residual 1.16e-14.  Bar 1e-11, as for the transpose.
    """
    C = _cell(_LC, _ISO)
    Mm = np.diag([1.0, -1.0, 1.0])
    D = np.empty_like(C)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            D[i, j] = Mm @ C[i, C.shape[1] - 1 - j] @ Mm
    oC, RC, _TC, JC = pmm_jones_2d_staggered(_P, _P, C, **_G7)
    oD, RD, _TD, JD = pmm_jones_2d_staggered(_P, _P, D, **_G7)
    ic, idd = _idx(oC), _idx(oD)
    dev = max(abs(RD[r][idd[(m, n)]] - RC[r][ic[(m, -n)]])
              for (m, n) in idd if (m, -n) in ic for r in (0, 1))
    assert dev < 1e-11, dev
    assert np.max(np.abs(JD - _M2 @ JC @ _M2)) < 1e-11


# =========================================================================== #
# G8 -- THE STACK: tensor layers in PMM2DStackPure.
# =========================================================================== #
def test_g8a_split_tensor_layer_equals_the_single_layer():
    """Cascading one tensor layer as TWO half-thickness layers must reproduce
    the single layer: the interface between them is the identity match, so
    this exercises the Redheffer cascade on tensor modes without changing the
    physics.

    MEASURED 2026-09-09 (build doc T8), conical incidence theta=0.15,
    phi=0.4, M=6: max|dR| 2.36e-16, max|dT| 3.33e-16, max|dJones| 8.74e-16 --
    i.e. round-off.  Bar 1e-12: ~3 decades over the measurement (room for the
    reassociation a different cascade order legitimately produces) and decades
    under any physical difference.
    """
    c = _cell(_LC, _ISO)
    st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.5, n_modes=6,
                        n_orders=3)
    st.add_layer(_DEP, eps_cell=c).add_layer(_DEP, eps_cell=c)
    st.set_source(_WL, theta=0.15, phi=0.4)
    _o2, R2, T2, J2 = st.solve()
    _o1, R1, T1, J1 = pmm_jones_2d_staggered(_P, _P, c, 1.5, 1.0, 2 * _DEP,
                                             _WL, degree=6, n_orders=3,
                                             theta=0.15, phi=0.4)
    assert np.max(np.abs(R2 - R1)) < 1e-12
    assert np.max(np.abs(T2 - T1)) < 1e-12
    assert np.max(np.abs(J2 - J1)) < 1e-12


def _g8b_residual(M):
    layers = [(_LC, 0.21e-6), (_GYRO, 0.13e-6), (_LC_M, 0.17e-6)]
    stk = PMM2DStackPure(0.4e-6, 0.4e-6, n_superstrate=1.0, n_substrate=1.5,
                         n_modes=M, n_orders=2)
    for t33, th in layers:
        stk.add_layer(th, eps=t33)
    stk.set_source(1.0e-6, theta=25 * np.pi / 180, phi=40 * np.pi / 180)
    _o, R, T, J = stk.solve()
    Rb, Tb, jr, _jt = berreman_jones_1d(layers, 1.5, 1.0, 1.0e-6,
                                        angle=25 * np.pi / 180,
                                        phi=40 * np.pi / 180)
    return max(float(np.max(np.abs(R.sum(axis=1) - Rb))),
               float(np.max(np.abs(T.sum(axis=1) - Tb))),
               float(np.max(np.abs(J - jr))))


def test_g8b_uniform_tensor_multilayer_matches_berreman():
    """Three UNIFORM anisotropic layers (LC / gyrotropic / mirrored LC) at
    conical incidence against the exact Berreman multilayer.  Each uniform
    TENSOR region takes its own region eig (it is not eps-free-separable, so
    it cannot ride the shared geometric eig), and the cascade must still be
    exact.

    MEASURED 2026-09-09 (build doc T8): 4.21e-14 at M=7, 2.40e-12 at M=5 --
    spectral, and DROPPING (factor 57).  Bars: < 1e-11 at M=7 (238x over the
    measurement) and a 10x drop from M=5 to M=7 (5.7x of headroom), which is
    what makes the M=7 number a convergence statement rather than a plateau.
    """
    r5, r7 = _g8b_residual(5), _g8b_residual(7)
    assert r7 < 1e-11, r7
    assert r7 < r5 / 10.0, (r5, r7)


def test_g9_absorption_budget_closes_for_a_lossy_tensor_stack():
    """``retain_internal`` + ``layer_absorption`` on a stack containing a
    PATTERNED lossy tensor layer and a UNIFORM gyrotropic layer.  The flux
    form is the eps-FREE block Gram, so it must keep working for tensor modes
    -- verified, not assumed.

    MEASURED 2026-09-09 (build doc T9), conical theta=0.12 phi=0.3, ladder in
    M:

        M      |sum A - (1 - R - T)|      A(lossy layer)     max|A(lossless)|
        6      2.155e-05 / 1.290e-06     0.254416/0.291326      5.80e-15
        7      3.982e-07 / 1.067e-06     0.254468/0.291394      6.22e-15
        8      8.205e-08 / 1.670e-08     0.254447/0.291294      5.59e-14

    Bar at M=8 = 1e2 x the worse of the two polarizations (8.2e-08), rounded
    to 1e-05: this is a CROSS-MACHINERY closure -- internal block-Gram flux
    against the Rayleigh far field -- so it tracks the MODAL error, not
    round-off, and the 1e2 factor is what keeps it off the convergence knife
    edge.  It sits ~4 decades under the 0.254 absorption it audits.  The M=6
    -> M=8 drop (263x) is asserted too, so the number is a convergence
    statement rather than a plateau.
    """
    def _run(M):
        st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.5,
                            n_modes=M, n_orders=3)
        st.add_layer(0.12e-6, eps_cell=_cell(_LC, _ISO))
        st.add_layer(_DEP, eps_cell=_cell(_LC, _ISO + 0.8j * np.eye(3)))
        st.add_layer(0.09e-6, eps=_GYRO)
        st.set_source(_WL, theta=0.12, phi=0.3)
        _o, R, T, _J = st.solve(retain_internal=True)
        A = st.layer_absorption()
        dev = max(abs(float(A[:, c].sum())
                      - float(1.0 - R[c].sum() - T[c].sum())) for c in (0, 1))
        return A, dev

    A8, dev8 = _run(8)
    _A6, dev6 = _run(6)
    assert A8.shape == (3, 2)
    assert dev8 < 1e-5, dev8
    assert dev8 < dev6 / 10.0, (dev6, dev8)
    assert A8[1].min() > 0.05          # the lossy layer carries it ...
    assert np.max(np.abs(A8[[0, 2]])) < 1e-9   # ... the lossless ones do not


# =========================================================================== #
# G10 -- GUARDS.  Every message must name the alternative.
# =========================================================================== #
def _oop_cell(stray):
    c = _cell(_LC, _ISO)
    c[0, 0, 0, 2] = stray
    c[0, 0, 2, 0] = stray
    return c


#: the message must NAME THE ALTERNATIVE, not merely mention the entry's own
#: (prefix) name -- ``pmm_jones_2d`` is a substring of
#: ``pmm_jones_2d_staggered``, so a bare ``match="pmm_jones_2d"`` would pass on
#: the prefix alone and assert nothing.
_NAMES_HYBRID = r"Use pmm_jones_2d \(the hybrid"


def test_g10_out_of_plane_tensor_raises_and_names_the_hybrid():
    with pytest.raises(NotImplementedError, match=_NAMES_HYBRID) as exc:
        pmm_jones_2d_staggered(_P, _P, _oop_cell(0.4), 1.5, 1.0, _DEP, _WL,
                               degree=5)
    assert "OUT-OF-PLANE" in str(exc.value)
    with pytest.raises(NotImplementedError, match=_NAMES_HYBRID):
        PMM2DStackPure(_P, _P).add_layer(_DEP, eps_cell=_oop_cell(0.4))
    # a genuinely tilted director (theta = 0.6 rad off z) is out of plane too
    with pytest.raises(NotImplementedError, match=_NAMES_HYBRID):
        PMM2DStackPure(_P, _P).add_layer(_DEP,
                                         eps=uniaxial_tensor(1.5, 1.8, 0.6))


def test_g10_offplane_test_is_relative_not_strict():
    """A physically IN-PLANE cell built by ROTATING a diagonal tensor carries
    float noise in the xz/yz slots (``uniaxial_tensor(no, ne, pi/2, phi)`` has
    ``cos(pi/2) = 6.1e-17``), so a strict ``> 0`` test would refuse every real
    LC cell.  The floor is RELATIVE (1e-12 * tensor scale), shared verbatim
    with the hybrid's ``_tile_is_offplane``.

    MEASURED 2026-09-09 (build doc T10): the largest off-plane entry of
    ``uniaxial_tensor(1.5, 1.8, pi/2, 0.55)`` is 5.17e-17 against a tensor
    scale of 2.970, i.e. a floor of 2.97e-12 -- 4.8 decades of margin -- so
    the cell solves; a 1e-16 injected stray is likewise accepted and a 1e-3
    one is refused.
    """
    off = np.abs(_LC[[0, 1, 2, 2], [2, 2, 0, 1]]).max()
    assert 0.0 < off < 1e-12 * np.abs(_LC).max()      # genuinely nonzero noise
    _o, R, T, _J = pmm_jones_2d_staggered(_P, _P, _cell(_LC, _ISO), 1.5, 1.0,
                                          _DEP, _WL, degree=5, n_orders=2)
    assert np.isfinite(R.sum() + T.sum())
    pmm_jones_2d_staggered(_P, _P, _oop_cell(1e-16), 1.5, 1.0, _DEP, _WL,
                           degree=5, n_orders=2)
    with pytest.raises(NotImplementedError):
        pmm_jones_2d_staggered(_P, _P, _oop_cell(1e-3), 1.5, 1.0, _DEP, _WL,
                               degree=5, n_orders=2)


def test_g10_zero_ezz_and_bad_shapes_raise():
    bad = _cell(_LC, _ISO)
    bad[0, 0, 2, 2] = 0.0
    with pytest.raises(ValueError, match="e_zz"):
        pmm_jones_2d_staggered(_P, _P, bad, 1.5, 1.0, _DEP, _WL, degree=5)
    with pytest.raises(ValueError, match=r"Nx, Ny, 3, 3"):
        pmm_jones_2d_staggered(_P, _P, np.ones((2, 2, 2, 2), complex), 1.5,
                               1.0, _DEP, _WL, degree=5)
    with pytest.raises(ValueError, match="eps_cell must be SQUARE"):
        pmm_jones_2d_staggered(_P, _P, _cell(_LC, _ISO)[:1], 1.5, 1.0, _DEP,
                               _WL, degree=5)
    with pytest.raises(ValueError, match=r"scalar or a \(3, 3\)"):
        PMM2DStackPure(_P, _P).add_layer(_DEP, eps=np.ones((2, 2)))


def test_g10_scalar_entry_refuses_a_tensor_cell_and_names_the_jones_entry():
    with pytest.raises(ValueError, match="pmm_jones_2d_staggered") as exc:
        pmm_efficiency_2d_staggered(_P, _P, _cell(_LC, _ISO), 1.5, 1.0, _DEP,
                                    _WL, degree=5)
    assert "pmm_efficiency_2d_staggered" in str(exc.value)


def test_g10_shared_geometric_eig_refuses_a_tensor_assembly():
    """``_homog_geom_cache``'s eps-free split needs Meps33 = eps*G3 and
    Kzt = eps*Kzt0 to cancel; for a tensor they do not, so it must RAISE
    rather than hand back a silently-wrong geometric basis."""
    sol = Granet2DTransverseE(_P, _P, 2, 2, 4, _uniform(_LC),
                              k0=2 * np.pi / _WL)
    with pytest.raises(ValueError, match="uniform SCALAR"):
        _homog_geom_cache(sol)


# =========================================================================== #
# The lossless-closure tripwire (library item 2.1.5).
# =========================================================================== #
def test_tripwire_silent_on_a_well_resolved_hermitian_tensor_stack():
    """It must NOT fire on a converged lossless tensor solve.

    MEASURED 2026-09-09 (build doc T6/T11): the closure of this configuration
    is 6.3e-09, i.e. 7 decades inside the 5e-02 window.
    """
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        pmm_jones_2d_staggered(_P, _P, _cell(_LC, _ISO), 1.5, 1.0, _DEP, _WL,
                               degree=8, n_orders=4)
    assert not any("closure violated" in str(w.message) for w in rec)


def test_tripwire_silent_on_a_non_hermitian_tensor():
    """A NON-Hermitian (absorbing) tensor legitimately gives R+T < 1, so no
    unity is claimed and the tripwire must stay silent -- this is what makes
    the guard non-tautological (measured deficit 0.227, far outside the
    window, yet silent).
    """
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _o, R, T, _J = pmm_jones_2d_staggered(
            _P, _P, _cell(_LC, _ISO + 0.8j * np.eye(3)), 1.5, 1.0, _DEP, _WL,
            degree=8, n_orders=4)
    assert not any("closure violated" in str(w.message) for w in rec)
    assert float(R[0].sum() + T[0].sum()) < 0.95      # it really does absorb


def test_tripwire_fires_on_an_engineered_lossless_violation():
    """FAIL-BEFORE demonstration, engineered through the public API rather
    than hoped for: a high-contrast LOSSLESS gyrotropic cell at the MINIMUM
    modal count (M=3, the smallest the basis admits) is far from resolved, and
    the tripwire must say so.

    MEASURED 2026-09-09 (build doc T11), ladder over (eps_pillar, M) at this
    geometry:

        eps_pillar    M=3        M=4        M=5        M=8
          4.0      4.77e-02   2.46e-04   1.16e-04   3.93e-08
         16.0      8.26e-02   2.93e-04   3.54e-04   1.07e-07
        100.0      9.08e-02   2.53e-04   3.88e-04   1.51e-07

    The M=3 closure SATURATES near 1e-01 (the staggered basis degrades
    gracefully -- there is no blow-up to exploit), so the strongest available
    violation is 9.08e-02 against the 5e-02 window: a factor 1.8.  That ratio
    is modest, but the quantity is a DETERMINISTIC discretization error whose
    cross-build spread is ~1e-06, so the decision sits ~4 decades clear of
    build noise on the deciding side; and the SAME cell at M=8 is silent
    (1.5e-07, the arm above), which is the two-sided half of the claim.
    """
    hot = _cell(_GYRO, 100.0 * np.eye(3, dtype=complex))
    with pytest.warns(UserWarning, match="closure violated"):
        _o, R, T, _J = pmm_jones_2d_staggered(_P, _P, hot, 1.5, 1.0, _DEP,
                                              _WL, degree=3, n_orders=4)
    assert abs(float(R[0].sum() + T[0].sum()) - 1.0) > 5e-2
    # ... and the identical geometry, resolved, is silent
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _o, R8, T8, _J8 = pmm_jones_2d_staggered(_P, _P, hot, 1.5, 1.0, _DEP,
                                                 _WL, degree=8, n_orders=4)
    assert not any("closure violated" in str(w.message) for w in rec)
    assert abs(float(R8[0].sum() + T8[0].sum()) - 1.0) < 1e-5
