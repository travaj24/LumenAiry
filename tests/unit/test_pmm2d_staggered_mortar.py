"""PER-LAYER ELEMENT GRIDS (L2 mortar) for the PURE staggered 2-D PMM --
gates G1-G9 of ``docs/audits/BUILD_PMM2D_STAGGERED_MORTAR_2026_09_11.md``.

Design and the GO decision:
``docs/audits/EXPERIMENT_PMM2D_STAGGERED_MORTAR_2026_09_10.md``.  The 1-D
per-layer surface this transplants is
``AUDIT_PMM_PER_LAYER_GRIDS_IMPL_2026_07_28.md`` and its conditioning lessons
are ``PMM_M1_CONDITIONING_2026_08_04.md``.

``PMM2DStackPure(..., layer_grids='per-layer')`` gives every layer its own
element grid and couples adjacent grids weakly: tangential E tested on grid
B's trace space, tangential H on grid A's (the classic mode-matching pairing,
which keeps the system square for ``q_a != q_b``), with NO union trace space --
a union mortar space would need the common refinement at every interface,
which is precisely the cost this mode exists to avoid.

THE ONE PIECE THAT IS NEW PHYSICS-OF-THE-BASIS, and the reason ``test_g3_*``
exists: in 1-D both transverse components live in the SAME nodal space, so one
mass applies blockwise as ``kron(I_2, M)``.  Here ``E1 in V1`` and ``E2 in V2``
are DIFFERENT tensor-product spaces and the Eq.-25 H partner SWAPS them, so the
H row must be tested with the V1/V2 blocks exchanged.  On a CONFORMING
interface ``C1 = G1`` and ``C2 = G2`` and the swap cancels identically -- so
G1/G2, the identity gates, pass either way.  A conforming-parity-only gate
would have shipped the bug.

EVERY BAR BELOW IS DERIVED FROM A MEASUREMENT MADE ON **TWO BUILDS**
(2026-09-11), both readings stated in the assertion's comment:

  * WIN -- Windows 11, CPython 3.14.6, numpy 2.4.4, scipy 1.17.1
    (scipy-openblas), OMP/OPENBLAS/MKL = 1;
  * WSL -- Ubuntu, CPython 3.12.3, numpy 2.4.6, scipy 1.17.1
    (scipy-openblas), OMP/OPENBLAS = 1.

Per ``docs/TESTING_STANDARDS.md``: bit-identity claims are SAME-BUILD two-arm
by sha256; the conforming-identity bar is DERIVED AT RUNTIME from the stack's
own ``eps * cond_2(G)`` (it grows with ``M`` -- a fixed constant would fail);
the equal-DOF claim is stated as "not worse", never "better", and is barred on
the RATIO, which the cross-build study found to be the most reproducible
quantity in the campaign; and every silent-capable design choice carries a
FAIL-BEFORE arm.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumenairy.elements.berreman import berreman_jones_1d  # noqa: E402
from lumenairy.elements.pmm import (  # noqa: E402
    PMM2DStackHybrid,
    PMM2DStackPure,
    PMMStack,
)
from lumenairy.elements.pmm import _core as _pmmcore  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    _C,
    Basis1D,
    Granet2DTransverseE,
    _stag_cross_mass_1d,
    _stag_kron_apply,
)
from lumenairy.elements.rcwa import _core as _rc  # noqa: E402

_P = 1.2
_WL = 0.85
_EPS_P, _EPS_H = 6.0, 2.25


def _h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _pillar(N, lo, hi, e_p=_EPS_P, e_h=_EPS_H):
    c = np.full((N, N), complex(e_h))
    c[lo:hi, lo:hi] = e_p
    return c


def _stripe(N, lo, hi, e_p=_EPS_P, e_h=_EPS_H):
    c = np.full((N, N), complex(e_h))
    c[lo:hi, :] = e_p
    return c


def _gram_cond(N, M, a0x=0.0, a0y=0.0):
    """``cond_2`` of the two V1 / V2 block field Grams on one grid -- the
    round-off scale that DERIVES the conforming-identity bar."""
    sol = Granet2DTransverseE(_P, _P, N, N, M, np.full((N, N), 2.0 + 0j),
                              alpha0x=a0x, alpha0y=a0y, k0=2 * np.pi / _WL)
    G = -sol.Rmat
    qq = sol.q ** 2
    return max(float(np.linalg.cond(G[:qq, :qq])),
               float(np.linalg.cond(G[qq:, qq:])))


# ==========================================================================
# G1 -- the identical-grid BYPASS is BIT-EXACT vs layer_grids='shared'
# ==========================================================================
def test_g1_conforming_per_layer_stack_is_bit_exact_vs_shared():
    """G1.  A per-layer stack whose grids happen to COINCIDE takes the shipped
    square interface and reproduces ``layer_grids='shared'`` byte for byte.

    This is what makes the mode safe to expose: no existing user's bits move,
    and it is the 1-D contract
    (``test_identical_wall_stack_is_bit_exact_vs_shared``) lifted to 2-D.  It
    is an EXACT claim (``== 0``), not a tolerance, because it is a BYPASS, not
    an arithmetic identity."""
    cA, cB = _pillar(2, 0, 1), _stripe(2, 1, 2)

    def _build(lg):
        st = PMM2DStackPure(_P, n_modes=5, n_orders=2, layer_grids=lg)
        extra = {"grid": 2, "n_modes": 5} if lg == "per-layer" else {}
        st.add_layer(0.30, eps_cell=cA)
        st.add_layer(0.15, eps=2.0, **extra)
        st.add_layer(0.25, eps_cell=cB)
        st.set_source(_WL, theta=0.20, phi=0.30)
        return st

    o1, R1, T1, J1 = _build("shared").solve()
    o2, R2, T2, J2 = _build("per-layer").solve()
    np.testing.assert_array_equal(o1, o2)
    for a, b in ((R1, R2), (T1, T2), (J1, J2)):
        assert _h(a) == _h(b), float(np.max(np.abs(a - b)))


# ==========================================================================
# G2 -- the conforming identity through the FORCED mortar, bar DERIVED
# ==========================================================================
@pytest.mark.parametrize("name, cells, M, th, ph", [
    ("stripe|stripe", (_stripe(2, 0, 1), _stripe(2, 1, 2)), 5, 0.0, 0.0),
    ("stripe|stripe oblique", (_stripe(2, 0, 1), _stripe(2, 1, 2)), 5, 0.20, 0.0),
    ("stripe|pillar", (_stripe(3, 0, 1), _pillar(3, 1, 2)), 6, 0.20, 0.0),
    ("pillar|pillar conical", (_pillar(3, 0, 2), _pillar(3, 1, 3)), 6, 0.25, 0.7),
    ("stripe|uniform|pillar", (_stripe(2, 0, 1), None, _pillar(2, 0, 1)), 5,
     0.15, 0.0),
])
def test_g2_forced_mortar_reduces_to_the_square_modal_match(name, cells, M,
                                                            th, ph):
    """G2.  Driven through the MORTAR algebra with the identical-grid bypass
    DISABLED, a conforming stack must reproduce the shipped union-grid cascade.

    Bit identity is not attainable and is not claimed: the mortar multiplies by
    the block Gram ``G`` on both sides of a ``solve`` where the plain interface
    does not, so the identity is exact algebraically and rounds at
    ``eps * cond_2(G)``.  THE BAR IS THEREFORE DERIVED AT RUNTIME from this
    stack's own grid -- it grows with ``M`` (MEASURED ``eps*cond`` 6.8e-13 at
    ``M = 5``, 1.6e-12 at ``M = 6``, 7.8e-12 at ``M = 8``), so a fixed 1e-12
    constant would fail at ``M = 8``.  The multiplier is 10, chosen because the
    cross-build study measured the READING's own spread at O(1) (round-off
    limited) while the bar itself is deterministic: worst reading 1.11e-13
    (WIN) / 1.36e-13 (WSL) against a derived bar of 1.6e-11 -- a 120x gap with
    the build spread inside it.

    ``_gram_cond`` is evaluated at NORMAL incidence even for the oblique and
    conical rows.  That is deliberate and MEASURED (2026-09-11): the Gram's
    ``cond_2`` moves at most 8.7 % across ``alpha0`` in
    ``(0, 0) .. (1.7, -0.9)`` on these grids, and normal incidence is the
    larger reading on two of the three, so it is a conservative proxy against
    margins of 201x-2251x.

    NOTE this gate CANNOT see the V1/V2 H-row swap (on identical grids
    ``C1 = G1``, ``C2 = G2`` and it cancels).  ``test_g3_*`` is what does --
    and the independent verification reproduced the BLINDNESS explicitly:
    with the swap DISABLED this identity still reads 1.82e-15 (stripe pair)
    and 3.65e-13 (pillar pair), i.e. it passes either way
    (``VERIFY_PMM2D_STAGGERED_MORTAR_2026_09_11.md`` S4.1)."""
    N = cells[0].shape[0]
    st_a = PMM2DStackPure(_P, n_modes=M, n_orders=2)
    st_b = PMM2DStackPure(_P, n_modes=M, n_orders=2, layer_grids="per-layer")
    for c in cells:
        if c is None:
            st_a.add_layer(0.20, eps=2.0)
            st_b.add_layer(0.20, eps=2.0, grid=N, n_modes=M)
        else:
            st_a.add_layer(0.30, eps_cell=c)
            st_b.add_layer(0.30, eps_cell=c)
    st_a.set_source(_WL, theta=th, phi=ph)
    st_b.set_source(_WL, theta=th, phi=ph)
    o1, R1, T1, J1 = st_a.solve()
    o2, R2, T2, J2 = st_b._solve_per_layer(jones=True, retain_internal=False,
                                           force_mortar=True)
    bar = 10.0 * np.finfo(float).eps * _gram_cond(
        N, M, a0x=0.0, a0y=0.0)
    scale = max(float(np.max(R1)), float(np.max(T1)))
    got = max(float(np.max(np.abs(R1 - R2))) / scale,
              float(np.max(np.abs(T1 - T2))) / scale,
              float(np.max(np.abs(J1 - J2))) / float(np.max(np.abs(J1))))
    assert got < bar, (name, got, bar)


# ==========================================================================
# G3 -- THE FAIL-BEFORE: the H-row V1/V2 swap is load-bearing
# ==========================================================================
def test_g3_h_row_v1_v2_swap_is_load_bearing():
    """G3.  Disabling the Eq.-25 V1/V2 swap on the mortar's H row must move
    the observable by ORDERS on a genuinely NON-conforming stack, while every
    conforming gate above stays green either way.

    This is the single most important gate the design produced: the naive
    1-D-looking same-order blocks are SILENT on a conforming interface, so the
    identity tests cannot see the bug.  Both arms run in the SAME process on
    the SAME build, and the claim is a RATIO, not a reading.  MEASURED on THIS
    fixture (union reference at ``M = 4``, arms at ``M_A = 7`` / ``M_B = 5``):
    swap ON 2.33e-02 from the reference with closure 3.59e-06; swap OFF
    4.02e+01 and 1.15e+02 -- a factor 1.7e+03 on the observable and 3.2e+07 on
    lossless closure.  The build doc's G3 table runs the same control one rung
    finer (``M_A = 9`` / ``M_B = 7``) on BOTH builds and reads, to every
    printed digit, 5.8396e-03 / 1.0412e-08 (on) against 8.0455e+00 (WIN) /
    8.0459e+00 (WSL) and 2.7654e+01 (off) -- both arms are
    discretisation-scale, not round-off-scale, so the fail-before reproduces
    across builds rather than merely re-occurring."""
    cA, cB = _pillar(2, 0, 1), _pillar(3, 1, 2)
    ref = PMM2DStackPure(_P, n_modes=4, n_orders=2)
    ref.add_layer(0.30, eps_cell=np.kron(cA, np.ones((3, 3))))
    ref.add_layer(0.25, eps_cell=np.kron(cB, np.ones((2, 2))))
    ref.set_source(_WL, theta=0.18, phi=0.35)
    _o, Rr, Tr, _J = ref.solve()

    def _arm():
        st = PMM2DStackPure(_P, n_modes=7, n_orders=2,
                            layer_grids="per-layer")
        st.add_layer(0.30, eps_cell=cA, n_modes=7)
        st.add_layer(0.25, eps_cell=cB, n_modes=5)
        st.set_source(_WL, theta=0.18, phi=0.35)
        return st.solve()

    _o, R_on, T_on, _J = _arm()
    assert _pmmcore.PMM2D_MORTAR_H_SWAP is True
    try:
        _pmmcore.PMM2D_MORTAR_H_SWAP = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _o, R_off, T_off, _J = _arm()
    finally:
        _pmmcore.PMM2D_MORTAR_H_SWAP = True

    def _err(R, T):
        return float(np.max(np.abs(R - Rr))) + float(np.max(np.abs(T - Tr)))

    def _clo(R, T):
        return float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0)))

    # DECISIONS, both two-sided: the correct arm is close to the reference and
    # conserves energy; the broken arm is neither, by >= 2 orders each.
    assert _err(R_off, T_off) > 100.0 * _err(R_on, T_on)
    assert _clo(R_off, T_off) > 100.0 * _clo(R_on, T_on)
    assert _err(R_on, T_on) < 5e-2 and _clo(R_on, T_on) < 1e-5


# ==========================================================================
# G4 -- the mortar's OWN error, isolated against independent oracles
# ==========================================================================
@pytest.mark.parametrize("grids", [(2, 2), (2, 3), (3, 4)])
def test_g4_transparent_interface_reproduces_the_analytic_fresnel_slab(grids):
    """G4 (scalar).  ONE uniform slab SPLIT into two sub-layers on DIFFERENT
    grids.  The interface is physically ABSENT, so the exact answer is the
    analytic Fresnel slab and there is no geometry to resolve -- nothing but
    the mortar can move the answer.

    RE-MEASURED 2026-09-11 AT THIS TEST'S OWN SETTING (``M = 6``,
    ``theta = 0.20``) -- the readings this docstring previously carried were
    the build doc's ``M = 7`` table, which is a different fixture and included
    a ``(2, 4)`` arm this test does not run:

        (2,2) conforming     2.278e-13   closure 7.11e-15
        (2,4) nested         1.641e-13   closure 3.12e-13
        (2,3) NON-conforming 1.474e-13   closure 2.83e-13
        (3,4) NON-conforming 3.197e-14   closure 2.59e-14

    A NON-conforming mortar interface is NOT measurably worse than a
    conforming one, which is what says every gap in the accuracy gates is
    RESOLUTION, not the interface.  The bar is a magnitude bar with decades on
    both sides (440x above the worst reading, and the 1e-3-class resolution
    signals of the other gates far below); the readings themselves are
    round-off-limited and have an O(1) cross-build spread, so they are
    deliberately NOT pinned."""
    n, t, th, M = 2.0, 0.30, 0.20, 6
    k0 = 2 * np.pi / _WL
    kz0 = k0 * np.cos(th)
    kz1 = k0 * np.sqrt(n ** 2 - np.sin(th) ** 2 + 0j)
    r01 = (kz0 - kz1) / (kz0 + kz1)
    ph = np.exp(2j * kz1 * t)
    R_exact = float(abs((r01 - r01 * ph) / (1 - r01 * r01 * ph)) ** 2)
    ga, gb = grids
    st = PMM2DStackPure(_P, n_modes=M, n_orders=1, layer_grids="per-layer")
    st.add_layer(0.5 * t, eps=n ** 2, grid=ga, n_modes=M)
    st.add_layer(0.5 * t, eps=n ** 2, grid=gb, n_modes=M)
    st.set_source(_WL, theta=th, phi=0.0)
    o, R, T, _J = st.solve()
    p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    assert abs(float(R[1, p0]) - R_exact) < 1e-10
    assert abs(float(R.sum(1)[1] + T.sum(1)[1] - 1.0)) < 1e-10


@pytest.mark.parametrize("grids", [(2, 2), (2, 3)])
def test_g4_generalized_mortar_reproduces_berreman_through_a_split(grids):
    """G4 (out-of-plane).  The same transparent-interface argument on the
    GENERALIZED cascade: one uniform tilted-director LC slab split across
    grids, against ``berreman_jones_1d``.

    An out-of-plane layer returns DISTINCT forward/backward sets and puts the
    whole stack on the generalized S-matrix, so this exercises
    ``_interface_smatrix_general_mortar_2d`` -- the twin with the same V1/V2
    swap.  MEASURED at ``M = 7``, 25 deg CONICAL (WIN): conforming (2,2)
    2.1e-12, NON-conforming (2,3) 3.2e-11, closure <= 1.3e-11.  (A ``(3,4)``
    reading of 3.4e-13 appears in the build doc's table; this test does not
    run that pair.)  Bar 1e-8: five decades above the worst reading and five
    below the smallest real signal (the M = 5 rung of the same fixture reads
    1.6e-06).  The independent verification re-ran the same argument on ITS
    OWN fixture over three mounts, two modal counts and three grid pairs and
    reads 3.06e-11 .. 9.99e-12 at ``M = 7`` against 1.03e-06 .. 2.52e-05 at
    ``M = 5`` -- the same two-sided separation
    (``VERIFY_PMM2D_STAGGERED_MORTAR_2026_09_11.md`` S9)."""
    def _uni(no, ne, tilt, azim):
        c, s = np.cos(tilt), np.sin(tilt)
        d = np.array([s * np.cos(azim), s * np.sin(azim), c])
        return no ** 2 * np.eye(3) + (ne ** 2 - no ** 2) * np.outer(d, d)

    eps33 = _uni(1.5, 1.7, np.deg2rad(35.0), np.deg2rad(25.0))
    th, phz, M = np.deg2rad(25.0), np.deg2rad(40.0), 7
    Rb, Tb, Jb, _Jt = berreman_jones_1d([(eps33, 0.35)], 1.5, 1.0, _WL,
                                        angle=th, phi=phz)
    ga, gb = grids
    st = PMM2DStackPure(_P, n_superstrate=1.0, n_substrate=1.5, n_modes=M,
                        n_orders=1, layer_grids="per-layer")
    st.add_layer(0.175, eps=eps33, grid=ga, n_modes=M)
    st.add_layer(0.175, eps=eps33, grid=gb, n_modes=M)
    st.set_source(_WL, theta=th, phi=phz)
    o, R, T, J = st.solve()
    assert float(np.max(np.abs(J - np.asarray(Jb)))) < 1e-8
    assert float(np.max(np.abs(R.sum(1) - np.asarray(Rb)))) < 1e-8
    assert float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0))) < 1e-8


# ==========================================================================
# G5 / G6 -- against the EXACT 1-D oracle, and at EQUAL degrees of freedom
# ==========================================================================
_PER1D, _WL1D, _TH1D = 0.9, 0.6, 0.20
_TS = (0.30, 0.22)
_DUTIES = (0.5, 1.0 / 3.0)


def _oracle_stripe(deg):
    st = PMMStack(_PER1D, degree=deg, far_field_orders=5)
    for duty, t in zip(_DUTIES, _TS):
        st.add_layer(t, segments=[(duty, _EPS_P), (1.0 - duty, _EPS_H)])
    st.set_source(_WL1D, theta=_TH1D)
    o, R, T = st.solve()[:3]
    return np.asarray(o).ravel(), np.atleast_2d(R), np.atleast_2d(T)


def _score_1d(o2d, R, T, o1d, R1d, T1d, mirror=False):
    """max |R - R_1D|, |T - T_1D| over the retained ``n = 0`` orders on the TE
    row (incident ``E_y``, row 1 in BOTH engines).  ``mirror=True`` is the
    anti-mirror tripwire: it scores against order ``-m``, the failure the
    far-field projection kernel once had."""
    best = 0.0
    for m in (-1, 0, 1):
        sel = np.where((o2d[:, 0] == m) & (o2d[:, 1] == 0))[0][0]
        j = int(np.where(o1d == (-m if mirror else m))[0][0])
        best = max(best, abs(float(R[1, sel]) - float(R1d[1, j])),
                   abs(float(T[1, sel]) - float(T1d[1, j])))
    return best


def _stripe_cell(N, num):
    c = np.full((N, N), _EPS_H + 0j)
    c[:num, :] = _EPS_P
    return c


def _mortar_stripe(MA, MB, n_orders=1):
    st = PMM2DStackPure(_PER1D, n_modes=max(MA, MB), n_orders=n_orders,
                        layer_grids="per-layer")
    st.add_layer(_TS[0], eps_cell=_stripe_cell(2, 1), n_modes=MA)
    st.add_layer(_TS[1], eps_cell=_stripe_cell(3, 1), n_modes=MB)
    st.set_source(_WL1D, theta=_TH1D)
    return st.solve(jones=False)


def test_g5_stripe_pair_per_order_vs_the_exact_1d_oracle():
    """G5.  A y-uniform stripe stack with a DIFFERENT duty per layer -- layer A
    on halves (``N = 2``), layer B on thirds (``N = 3``), genuinely
    NON-conforming, common refinement ``N = 6`` -- scored PER ORDER against the
    EXACT 1-D pure PMM, which is the truth for the whole 2-D answer because the
    stack is y-uniform.

    Two claims, both DECISIONS rather than readings:

    1. the error collapses monotonically over three rungs, ending far inside
       the coarsest and far above the oracle's own degree-12-vs-14 self-gap;
    2. the ANTI-MIRROR tripwire -- the direct order assignment must beat the
       MIRRORED one by a wide margin.  Order ``-m`` deposited into slot ``m``
       is exact at normal incidence and for reflection-symmetric cells and
       wrong everywhere else, and it is the historical A|B defect of this
       engine; a per-layer cascade must not reintroduce it.

    MEASURED (WIN) errs 1.73e-01 / 1.66e-02 / 6.44e-03 at
    ``(M_A, M_B) = (5,5) / (7,7) / (8,6)``, mirrored arm 1.02e-01 / 1.37e-01 /
    1.35e-01, closure 2.06e-04 / 9.89e-06 / 5.91e-06; the oracle's own
    degree-12-vs-14 self-gap is 8.60e-06.  The ladder continues to 2.90e-05 at
    ``M = 11`` with closure 2.49e-11 -- measured in the build doc's G5 table
    rather than here, because a ``q = 30`` region eig is an 1800-dimension
    solve.
    """
    o1d, R1d, T1d = _oracle_stripe(14)
    o12, R12, T12 = _oracle_stripe(12)
    keep = np.abs(o1d) <= 1
    self_gap = float(max(np.max(np.abs(R12[:, keep] - R1d[:, keep])),
                         np.max(np.abs(T12[:, keep] - T1d[:, keep]))))
    # the oracle is readable throughout: MEASURED self-gap 8.60e-06 (WIN)
    assert self_gap < 1e-4, self_gap

    errs, mirs = [], []
    for MA, MB in ((5, 5), (7, 7), (8, 6)):
        o, R, T = _mortar_stripe(MA, MB)
        errs.append(_score_1d(o, R, T, o1d, R1d, T1d))
        mirs.append(_score_1d(o, R, T, o1d, R1d, T1d, mirror=True))
        assert abs(float(R.sum(1)[1] + T.sum(1)[1] - 1.0)) < 1e-2, (MA, errs)
    assert errs[1] < 0.5 * errs[0] and errs[2] < 0.5 * errs[1], errs
    assert errs[-1] < 5e-2 and errs[-1] > 5.0 * self_gap, (errs, self_gap)
    # the anti-mirror tripwire, at the two RESOLVED rungs.  The coarsest is
    # not resolved enough for the two assignments to separate (it reads
    # 0.59x), which is itself why the tripwire is stated at the resolved end.
    assert mirs[1] > 4.0 * errs[1] and mirs[2] > 4.0 * errs[2], (errs, mirs)


def test_g6_equal_dof_per_layer_is_not_worse_than_the_union_grid():
    """G6.  At EQUAL degrees of freedom -- the only comparison that is not
    rigged -- the per-layer arm must be NOT WORSE than the union grid on the
    observable and not worse on lossless closure.

    The staggered basis makes the comparison exact: per axis ``q = N (M-1)``,
    so ``q_union(M) = 6(M-1)`` equals ``q_A = 2(M_A-1)`` at ``M_A = 3M-2`` and
    ``q_B = 3(M_B-1)`` at ``M_B = 2M-1``, and at those settings EVERY region
    eigenproblem in both arms has exactly the same dimension ``2 q^2``.

    THE CLAIM IS DELIBERATELY 'NOT WORSE', NEVER 'BETTER'.  The advantage is
    regime-dependent and the regime is mapped: on this stripe pair the mortar
    is 7.6x / 7.3x more accurate at ``q = 18 / 24`` and 1.48x WORSE at
    ``q = 30`` once both arms have converged; on a corner-dominated 2-D pillar
    pair the crossing moves earlier, to between ``q = 18`` and ``q = 24``.  The
    durable half is the CLOSURE, which is 170x-5142x tighter and does not
    decay.  The bar is on the RATIO because the cross-build study measured the
    ratio to 5.2e-09 while the errors themselves move at 1e-4 relative.
    MEASURED ratios (WIN / WSL) in the build doc's G6 table."""
    o1d, R1d, T1d = _oracle_stripe(14)
    for Mu in (3,):
        q = 6 * (Mu - 1)
        stu = PMM2DStackPure(_PER1D, n_modes=Mu, n_orders=1)
        stu.add_layer(_TS[0], eps_cell=_stripe_cell(6, 3))
        stu.add_layer(_TS[1], eps_cell=_stripe_cell(6, 2))
        stu.set_source(_WL1D, theta=_TH1D)
        ou, Ru, Tu = stu.solve(jones=False)
        om, Rm, Tm = _mortar_stripe(q // 2 + 1, q // 3 + 1)
        eu = _score_1d(ou, Ru, Tu, o1d, R1d, T1d)
        em = _score_1d(om, Rm, Tm, o1d, R1d, T1d)
        cu = abs(float(Ru.sum(1)[1] + Tu.sum(1)[1] - 1.0))
        cm = abs(float(Rm.sum(1)[1] + Tm.sum(1)[1] - 1.0))
        # NOT WORSE on the observable, with a factor of 2 of slack for the
        # converged regime where the ordering is a coin toss between two right
        # answers; strictly TIGHTER on the two-sided lossless closure, which
        # is the half that does not decay.
        assert em < 2.0 * eu, (q, em, eu)
        assert cm < cu, (q, cm, cu)


def test_staircase_equal_dof_per_layer_vs_the_union_lattice():
    """The USE CASE, at EQUAL degrees of freedom: a 3-slice z-staircase whose
    slice widths are 1/2, 1/3 and 1/6 of the period, so per-slice
    ``N = 2, 3, 6`` and the union lattice is their common refinement ``N = 6``.

    Made y-uniform so the exact 1-D ``PMMStack`` at degree 14 is the truth for
    the WHOLE 3-slice stack.  Parameterised by ``q = N (M - 1)``, at which every
    region eigenproblem in BOTH arms has exactly the same dimension ``2 q^2``,
    with per-slice ``M_i = q / N_i + 1``.

    Two claims, and the second is the structural one:

    1. at matched DOF the per-layer arm is NOT WORSE than the union grid --
       MEASURED 8.21e-02 against 1.29e-01 at ``q = 12`` (ratio 0.634) and
       2.41e-03 against 4.79e-03 at ``q = 18`` (ratio 0.504), the second rung
       measured in the build doc rather than here because it is a 33 s solve;
    2. the union lattice has a DOF FLOOR the per-layer arm does not.
       ``Basis1D`` requires ``M >= 3``, so the smallest ``q`` a lattice can
       carry is ``2 N``: the union grid here cannot be run below ``q = 12``,
       and on the sibling staircase with widths 1/2, 1/3, 1/4 (union ``N = 12``)
       it cannot be run below ``q = 24`` where per-layer reaches ``q = 12`` --
       64x less eigenwork, and it is a FLOOR, not a tuning choice.
    """
    per, wl, th, ts = 0.9, 0.6, 0.20, 0.12
    duties = (0.5, 1.0 / 3.0, 1.0 / 6.0)
    st = PMMStack(per, degree=14, far_field_orders=5)
    for d in duties:
        st.add_layer(ts, segments=[(d, _EPS_P), (1.0 - d, _EPS_H)])
    st.set_source(wl, theta=th)
    o1d, R1d, T1d = st.solve()[:3]
    o1d = np.asarray(o1d).ravel()
    R1d, T1d = np.atleast_2d(R1d), np.atleast_2d(T1d)

    def _stripe_cell(N, num):
        c = np.full((N, N), _EPS_H + 0j)
        c[:num, :] = _EPS_P
        return c

    q = 12
    su = PMM2DStackPure(per, n_modes=q // 6 + 1, n_orders=1)
    for num in (3, 2, 1):
        su.add_layer(ts, eps_cell=_stripe_cell(6, num))
    su.set_source(wl, theta=th)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ou, Ru, Tu = su.solve(jones=False)
    sp = PMM2DStackPure(per, n_modes=8, n_orders=1, layer_grids="per-layer")
    for N in (2, 3, 6):
        sp.add_layer(ts, eps_cell=_stripe_cell(N, 1), n_modes=q // N + 1)
    sp.set_source(wl, theta=th)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        op, Rp, Tp = sp.solve(jones=False)
    eu = _score_1d(ou, Ru, Tu, o1d, R1d, T1d)
    ep = _score_1d(op, Rp, Tp, o1d, R1d, T1d)
    # identical eigenproblem sizes in both arms -- the comparison is exact
    assert 2 * q * q == 288
    assert ep < 1.2 * eu, (ep, eu)          # NOT WORSE; measured 0.634x
    # ... and the union arm CANNOT be run below this q on this stack, because
    # M >= 3 on its N = 6 lattice already costs q = 12.
    with pytest.raises(ValueError, match="n_modes"):
        PMM2DStackPure(per, n_modes=2)


# ==========================================================================
# G7 -- two-sided lossless closure, scalar AND Hermitian tensor
# ==========================================================================
@pytest.mark.parametrize("kind", ["scalar", "gyrotropic"])
def test_g7_two_sided_lossless_closure_on_non_conforming_grids(kind):
    """G7.  On a LOSSLESS stack ``sum R + sum T = 1`` is EXACT, so the gate is
    TWO-SIDED: a deficit is as much a defect as an excess.  A HERMITIAN tensor
    (gyrotropic ``e12 = -e21 = i b``) absorbs nothing, so the same exact claim
    applies -- and it is the arm that shows the mortar carries no material
    dependence (it is a geometric projection).

    The build-free restatement of a closure ladder is CONVERGENCE, not a
    reading: each modal rung must be at least a decade tighter than the one
    below.  MEASURED (WIN) scalar 9.69e-04 -> 1.83e-04 -> 1.67e-05 at
    ``M = 4/5/6`` on the (2,3) non-conforming pillar pair."""
    cA = _pillar(2, 0, 1)
    if kind == "gyrotropic":
        t = np.zeros((2, 2, 3, 3), dtype=_C)
        t[:] = np.diag([_EPS_H, _EPS_H, _EPS_H])
        t[0, 0] = np.array([[6.0, 0.9j, 0.0], [-0.9j, 6.0, 0.0],
                            [0.0, 0.0, 5.0]])
        cA = t
    cB = _pillar(3, 1, 2)
    clos = []
    for M in (4, 5, 6):
        st = PMM2DStackPure(_P, n_modes=M, n_orders=2,
                            layer_grids="per-layer")
        st.add_layer(0.30, eps_cell=cA)
        st.add_layer(0.25, eps_cell=cB)
        st.set_source(_WL, theta=0.18, phi=0.35)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            o, R, T = st.solve(jones=False)
        clos.append(float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0))))
    # MEASURED (WIN) scalar 9.69e-04 -> 1.83e-04 -> 1.67e-05, i.e. 5.3x then
    # 11x per rung; the DECISION is 'each rung materially tighter than the
    # last', barred at 3x with the measured minimum 5.3x inside it.
    assert clos[1] < 0.3 * clos[0] and clos[2] < 0.3 * clos[1], clos
    assert clos[-1] < 1e-4, clos


# ==========================================================================
# G8 -- the far-field order cap, DERIVED from the END grids, RAISES
# ==========================================================================
def test_g8_order_cap_is_derived_from_the_end_grids_and_raises():
    """G8.  The forward Rayleigh projection has ``q = N (M-1)`` columns per
    axis, so it can carry at most ``q`` order slots: ``n_orders <=
    (q - 1) // 2`` on BOTH end grids (the half-spaces ride them).  Beyond it
    the retained slots ALIAS one another and the least-squares draw is
    build-dependent -- while conserving energy exactly, i.e. energy-invisible.
    This is the T3-3 shape, applied before it bites.

    The prototype CLAMPED; the shipped surface RAISES, because a user asking
    for orders the end grids cannot carry should be told, not quietly served
    fewer.  The message must name the cap and both grids."""
    st = PMM2DStackPure(_P, n_modes=4, n_orders=7, layer_grids="per-layer")
    st.add_layer(0.30, eps_cell=_pillar(2, 0, 1))
    st.set_source(_WL, theta=0.10)
    with pytest.raises(ValueError, match=r"n_orders <= \(q - 1\) // 2 = 2"):
        st.solve()
    # ... and exactly at the cap it solves
    ok = PMM2DStackPure(_P, n_modes=4, n_orders=2, layer_grids="per-layer")
    ok.add_layer(0.30, eps_cell=_pillar(2, 0, 1))
    ok.set_source(_WL, theta=0.10)
    o, R, T = ok.solve(jones=False)
    assert o.shape[0] == 25
    # the cap follows the END layers' own n_modes, not the stack default
    big = PMM2DStackPure(_P, n_modes=4, n_orders=5, layer_grids="per-layer")
    big.add_layer(0.30, eps_cell=_pillar(2, 0, 1), n_modes=7)
    big.set_source(_WL, theta=0.10)
    big.solve(jones=False)          # q = 12 -> cap 5: no raise


# ==========================================================================
# G9 -- the conditioning census, in M1's own instruments
# ==========================================================================
def test_g9_conditioning_census_refuses_nothing_in_the_useful_range():
    """G9.  The mortar's ONE explicit inverse (``I + BA``) goes through the
    shipped guard, so M1's census instrument reads it for free.

    M1's conclusions transfer without a change of instrument: the two mortar
    ``solve``s stay ``solve``s (LAPACK ``gesv`` is backward stable, so a
    residual screen on them measures nothing) and the compounded exposure
    lands on the explicit inverse, which is where the guard is.  MEASURED over
    the 4 solves THIS TEST RUNS (two non-conforming configurations at
    ``M = 4, 5``): worst equilibrated ``rcond`` 8.9e-05 (WIN), i.e. FOUR
    decades above M1's 1e-8 screen.  The build doc's wider census (three
    configurations at ``M = 4, 5, 6``) and the independent verification's
    (nine solves on three configurations, worst 1.505e-05) agree: nothing is
    refused in the useful range, and the census is an instrument here, not a
    gate.

    WHAT THIS DOES NOT SAY, and the verification measured it: the design
    argument above holds for the configurations sampled, NOT in general.  On a
    grid carrying two walls 1e-3 of the period apart the two UNGUARDED
    ``solve``s reach ``cond_2`` 5.9e+07 and 4.9e+10 while ``I + BA``'s guarded
    ``rcond`` still reads 2.5e-07, and below ``delta ~ 1e-7`` they raise a bare
    ``numpy.linalg.LinAlgError``.  See
    ``docs/audits/VERIFY_PMM2D_STAGGERED_MORTAR_2026_09_11.md`` S6.3-S6.4."""
    prev, _rc._INV_CENSUS = _rc._INV_CENSUS, []
    try:
        worst, seen = 1.0, set()
        for cA, cB in ((_pillar(2, 0, 1), _pillar(3, 1, 2)),
                       (_stripe(2, 0, 1), _stripe(3, 0, 1))):
            for M in (4, 5):
                st = PMM2DStackPure(_P, n_modes=M, n_orders=2,
                                    layer_grids="per-layer")
                st.add_layer(0.30, eps_cell=cA)
                st.add_layer(0.25, eps_cell=cB)
                st.set_source(_WL, theta=0.18, phi=0.35)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    st.solve(jones=False)
        for site, _n, rc, _res, _flag in _rc._INV_CENSUS:
            seen.add(site)
            if rc == rc:                       # not NaN
                worst = min(worst, float(rc))
    finally:
        _rc._INV_CENSUS = prev
    assert "pmm2d staggered mortar interface (I + BA)" in seen, sorted(seen)
    # two decades of gap to M1's screen on both sides: the screen is 1e-8, the
    # worst measured reading 8.9e-05 (WIN), and the census quantity's measured
    # cross-build spread is 10 % (one configuration, on the NEHALEM-kernel
    # arm), so the margin is real and MEASURED rather than assumed.
    assert worst > 1e-6, worst


# ==========================================================================
# The separable cross-mass -- an IDENTITY, and the reason it is never dense
# ==========================================================================
def test_cross_mass_kron_factorisation_is_an_identity_and_never_materialised():
    """The 2-D cross-mass factors EXACTLY (both field spaces are tensor
    products and both partitions are rectangular), so
    ``C1 = kron(Ctt_y, Cbb_x)`` and the separable apply is an IDENTITY -- BLAS
    reassociation only.

    Materialising it is a REJECTED design and the measurement is not close: at
    ``N = (6, 12), M = 6`` the dense ``C1`` is 51.8 MB against 0.058 MB for its
    two factors (900x) and the apply is 39x faster -- per COMPONENT per
    INTERFACE, of which a staircase has two and ``nlay + 1``.  MEASURED
    identity 2.2e-16 .. 6.6e-16 across the build doc's four grid pairs on both
    builds; this test runs the three of them that are cheap, and the
    independent verification re-ran five pairs and reads 2.73e-16 .. 5.08e-16
    with memory ratios 75x / 225x / 588x / 900x / 1600x and apply speed-ups
    0.49x / 3.04x / 15.6x / 40.4x / 66.6x.

    Note the small pairs are NOT faster factored ((2,3) at M=6 reads 0.34x on
    WIN, 0.49x on the verification's re-run): the crossover is around 100x100
    -- the (2,3) dense operator is exactly 100x225 -- and the win is the
    MEMORY, which is what makes the design work at all."""
    rng = np.random.default_rng(0)
    ratios = []
    for (Na, Nb), M in (((2, 3), 6), ((3, 6), 6), ((6, 12), 6)):
        ba, bb = Basis1D(_P, Na, M, 1.0 + 0j), Basis1D(_P, Nb, M, 1.0 + 0j)
        Cx = _stag_cross_mass_1d(ba, bb, "B")
        Cy = _stag_cross_mass_1d(ba, bb, "Btilde")
        X = rng.standard_normal((Cy.shape[1] * Cx.shape[1], 3)) + 0j
        Y1 = _stag_kron_apply(Cy, Cx, X)
        Y2 = np.kron(Cy, Cx) @ X
        rel = float(np.max(np.abs(Y1 - Y2))) / float(np.max(np.abs(Y2)))
        assert rel < 1e-13, ((Na, Nb), rel)
        ratios.append((Cx.size * Cy.size * 16) / (Cx.nbytes + Cy.nbytes))
    # the memory ratio grows as q_A q_B; the claim is the GROWTH, not a
    # pinned megabyte count.  MEASURED 75x / 225x / 900x on both builds.
    assert ratios[0] > 50 and ratios[-1] > 500 and ratios[-1] > 5 * ratios[0]


def test_cross_mass_reduces_to_the_mass_on_one_grid():
    """The cross-mass between a basis and ITSELF is that basis's mass, to
    quadrature round-off -- the smoke test that separates a mortar defect from
    a resolution gap, and the reason the mortar's E-row operator can be read
    off ``-Rmat`` instead of rebuilt."""
    for N, M in ((2, 5), (3, 6), (4, 4)):
        b = Basis1D(_P, N, M, np.exp(-1j * 0.31 * _P))
        for which in ("B", "Btilde"):
            C = _stag_cross_mass_1d(b, b, which)
            Mm = b.mass(getattr(b, which), getattr(b, which))
            rel = float(np.max(np.abs(C - Mm))) / float(np.max(np.abs(Mm)))
            assert rel < 1e-13, (N, M, which, rel)


# ==========================================================================
# retain_internal / layer_absorption on per-layer grids
# ==========================================================================
def test_layer_absorption_closes_the_budget_on_non_conforming_grids():
    """``retain_internal`` / :meth:`layer_absorption` on per-layer grids
    (open item O-3).

    Two changes and one non-change: the partial cascades become RECTANGULAR
    and go through ``_redheffer_star_rect``; ``_flux_at`` reads the LAYER's own
    block field Gram ``blkdiag(G1_i, G2_i)`` and ``qq_i`` (no new assembly --
    ``G_i`` IS ``-Rmat_i`` and its Kronecker factors are already held, so the
    quadrature is separable too); and the half-space calibration is unchanged,
    because the half-spaces are conforming by construction.

    The honest gate is the CROSS-MACHINERY closure ``sum_i A_i == 1 - sum R -
    sum T``: an internal Gram-flux budget against the Rayleigh far field, two
    independent pieces of machinery that need not agree unless both are
    right."""
    cA = _pillar(2, 0, 1, e_p=6.0 + 0.35j)
    cB = _pillar(3, 1, 2, e_p=5.0 + 0.20j)
    st = PMM2DStackPure(_P, n_modes=6, n_orders=2, layer_grids="per-layer")
    st.add_layer(0.30, eps_cell=cA)
    st.add_layer(0.25, eps_cell=cB)
    st.set_source(_WL, theta=0.18, phi=0.35)
    o, R, T = st.solve(jones=False, retain_internal=True)
    A = st.layer_absorption()
    assert A.shape == (2, 2)
    assert float(np.min(A)) > 1e-4          # both lossy layers absorb
    budget = 1.0 - R.sum(axis=1) - T.sum(axis=1)
    gap = float(np.max(np.abs(A.sum(axis=0) - budget)))
    # The gap is DISCRETISATION-limited (it tracks the solve's own resolution),
    # so it reproduces across builds and takes a real bar: MEASURED 5.27e-05 at
    # M = 5 and 2.14e-06 at M = 6 on this non-conforming lossy pair, against an
    # absorbed fraction of 0.20-0.23.  The bar sits 47x above the M = 6 reading
    # and 4 decades below the quantity it is checking.
    assert gap < 1e-4, (gap, budget, A.sum(axis=0))


# ==========================================================================
# API contracts and refusals
# ==========================================================================
def test_layer_grids_validation_and_shared_path_refuses_per_layer_keywords():
    with pytest.raises(ValueError, match="'shared' or 'per-layer'"):
        PMM2DStackPure(_P, layer_grids="per_layer")
    with pytest.raises(ValueError, match="window_halfwidth"):
        PMM2DStackPure(_P, layer_grids="per-layer", window_halfwidth=1)
    st = PMM2DStackPure(_P, n_modes=5)
    for kw in ({"x_walls": [0.4]}, {"grid": 2}, {"n_modes": 6}):
        with pytest.raises(ValueError, match="layer_grids='per-layer'"):
            st.add_layer(0.2, eps=2.0, **kw)


def test_patterned_layer_refuses_a_free_segment_count():
    """Which per-layer grids are ADMISSIBLE is a GEOMETRY question, not a
    knob: a pillar 1/2 of the period wide exists on ``N in {2,4,...}`` and one
    1/3 wide on ``N in {3,6,...}``.  Asking for the 1/2 pillar 'on N=3'
    silently changes the DEVICE, so ``grid=`` is refused for a patterned
    layer and its count comes from its own cell."""
    st = PMM2DStackPure(_P, n_modes=5, layer_grids="per-layer")
    with pytest.raises(ValueError, match="grid= is not accepted"):
        st.add_layer(0.2, eps_cell=_pillar(2, 0, 1), grid=3)


def test_per_layer_mode_lifts_the_union_grid_constraint():
    """The union-grid raise is the shared path's contract and stays; the
    per-layer path is exactly the mode that lifts it."""
    sh = PMM2DStackPure(_P, n_modes=5)
    sh.add_layer(0.2, eps_cell=_pillar(2, 0, 1))
    with pytest.raises(ValueError, match="union-grid constraint"):
        sh.add_layer(0.2, eps_cell=_pillar(3, 1, 2))
    pl = PMM2DStackPure(_P, n_modes=5, n_orders=2, layer_grids="per-layer")
    pl.add_layer(0.2, eps_cell=_pillar(2, 0, 1))
    pl.add_layer(0.2, eps_cell=_pillar(3, 1, 2))
    pl.set_source(_WL, theta=0.18, phi=0.35)
    o, R, T = pl.solve(jones=False)
    assert np.all(np.isfinite(R)) and np.all(np.isfinite(T))


def test_uniform_layer_modal_default_follows_the_measured_neighbour_rule():
    """A UNIFORM layer defaults to ``grid = 1`` -- the cheapest region the
    engine can express -- but its ``n_modes`` must NOT default to the stack's
    ``M``: inside a cascade the uniform layer must also carry the NEIGHBOURS'
    traces, and at the stack default a ``grid = 1`` layer measured 3-6x worse.
    The rule is ``M_u = max(q_prev, q_next) / N_u + 1``, MEASURED to put all
    three grid choices within 3 % of each other at matched ``q_u``."""
    st = PMM2DStackPure(_P, n_modes=5, layer_grids="per-layer")
    st.add_layer(0.2, eps_cell=_pillar(3, 1, 2), n_modes=6)     # q = 15
    st.add_layer(0.1, eps=2.0)                                   # N_u = 1
    st.add_layer(0.2, eps_cell=_pillar(2, 0, 1), n_modes=5)     # q = 8
    Ms = st._perlayer_modal_counts()
    assert Ms == [6, 16, 5], Ms                # 15 / 1 + 1
    # ... and an explicit n_modes wins
    st2 = PMM2DStackPure(_P, n_modes=5, layer_grids="per-layer")
    st2.add_layer(0.2, eps_cell=_pillar(3, 1, 2), n_modes=6)
    st2.add_layer(0.1, eps=2.0, grid=3, n_modes=7)
    assert st2._perlayer_modal_counts() == [6, 7]


def test_convergence_floor_is_a_lower_bound_on_the_stack_error():
    """The per-layer ``M`` recipe, as a surface (open item O-10).

    A per-layer solve CAN be stationary in one knob and wrong: walking one
    layer's ``M`` across four rungs measured 3.42e-01 / 2.82e-01 / 2.68e-01 /
    2.79e-01 -- stationary to 4 %, NOT monotone, and 27 % wrong, because the
    OTHER layer was the limiting error the whole time.  So the shipped recipe
    is stationarity in EVERY knob, screened by the per-layer own-residual
    FLOOR, which measured a lower bound on the pair error at 15 of 16 surface
    points.  This test asserts the floor's DEFINING property on a fixture where
    one layer is deliberately starved."""
    st = PMM2DStackPure(_P, n_modes=5, n_orders=2, layer_grids="per-layer")
    st.add_layer(0.30, eps_cell=_pillar(2, 0, 1), n_modes=3)   # starved
    st.add_layer(0.25, eps_cell=_pillar(3, 1, 2), n_modes=5)   # resolved
    st.set_source(_WL, theta=0.18, phi=0.35)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        floor, per = st.convergence_floor()
    assert len(per) == 2 and floor == max(per)
    # the starved layer is the one the screen indicts, by a wide margin --
    # which is the whole point: it is knowable BEFORE the stack is assembled.
    # MEASURED 1.76e-01 (starved) against 1.99e-03 (resolved) -- an 88x
    # separation; the bar is a DECISION ("the screen indicts the starved
    # layer"), barred at 10x with the measured 88x inside it.
    assert per[0] > 10.0 * per[1], per
    with pytest.raises(ValueError, match="per-layer"):
        PMM2DStackPure(_P).convergence_floor()


# ==========================================================================
# The USE CASE: a z-staircase taper on per-layer non-uniform grids
# ==========================================================================
_XB0 = (0.1873 * _P, 0.7241 * _P)
_XB1 = (0.2917 * _P, 0.6109 * _P)


def test_taper_requires_per_layer_grids_and_says_why():
    st = PMM2DStackPure(_P, n_modes=5)
    with pytest.raises(ValueError, match="layer_grids='per-layer'"):
        st.add_tapered_pillar(0.3, eps_pillar=6.0, eps_host=2.25,
                              x_bounds_bottom=_XB0, y_bounds_bottom=_XB0)


def test_tapered_pillar_slices_have_distinct_exact_walls():
    """Every slice sits on its OWN 3-segment NON-UNIFORM grid whose interior
    walls are the interpolated pillar bounds, so NO TWO SLICES SHARE A WALL --
    which is exactly what the shared union grid cannot express (a wall moving
    1.8 nm per slice on a 700 nm period needs ``N ~ 390``)."""
    st = PMM2DStackPure(_P, n_modes=5, n_orders=1, layer_grids="per-layer")
    st.add_tapered_pillar(0.30, eps_pillar=6.0, eps_host=2.25,
                          x_bounds_bottom=_XB0, y_bounds_bottom=_XB0,
                          x_bounds_top=_XB1, y_bounds_top=_XB1, n_slices=4)
    walls = [tuple(np.asarray(L["wx"])[1:-1]) for L in st._layers]
    assert len(walls) == 4 and len(set(walls)) == 4
    for w in walls:
        # the midpoint rule samples STRICTLY inside the two end cross-sections
        assert _XB0[0] <= w[0] <= _XB1[0] and _XB1[1] <= w[1] <= _XB0[1]
    # each slice is 3 segments -- q = 3(M-1), not 390(M-1)
    assert all(np.asarray(L["wx"]).size == 4 for L in st._layers)


def test_taper_agrees_with_the_hybrid_staircase_at_the_same_slices():
    """The taper against an INDEPENDENT engine: the hybrid's own
    ``add_tapered_pillar`` staircase with the IDENTICAL slice walls.

    The hybrid carries a Fourier floor and this fixture is corner-dominated, so
    the comparison is bounded by the ORACLE, not by the pure arm: the bar is
    the hybrid's own degree self-gap, measured in the test.  What the pure arm
    contributes is a lossless closure DECADES tighter than the oracle it is
    scored against -- so a residual at the oracle's floor is the oracle's, and
    the test says so rather than pretending otherwise.  MEASURED (WIN) on THIS
    fixture (hybrid ``n_orders = 7``): hybrid ``R(0,0)`` 0.026323 at degree 7
    and 0.026838 at degree 9, i.e. an ORACLE self-gap of 5.15e-04 with closure
    6.90e-03 / 2.61e-03; the pure arm reads 0.055159 at ``M = 4`` and 0.027188
    at ``M = 5``, so it sits 8.65e-04 from the oracle -- 1.7x its own
    uncertainty -- with a closure of 5.17e-04, 13x tighter than the oracle it
    is being scored against.  The build doc's taper table runs the same
    fixture at the hybrid's ``n_orders = 9`` and ``M = 6`` and reads 1.86e-04
    from the oracle with a closure 101x tighter."""
    kw = dict(eps_pillar=6.0, eps_host=2.25, x_bounds_bottom=_XB0,
              y_bounds_bottom=_XB0, x_bounds_top=_XB1, y_bounds_top=_XB1,
              n_slices=4)

    def _hyb(deg):
        h = PMM2DStackHybrid(_P, n_orders=7, degree=deg)
        h.add_tapered_pillar(0.30, **kw)
        h.set_source(_WL, theta=0.18, phi=0.35)
        o, R, T = h.solve()[:3]
        o = np.asarray(o)
        p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
        return float(np.atleast_2d(R)[0, p0]), float(
            np.max(np.abs(np.atleast_2d(R).sum(1)
                          + np.atleast_2d(T).sum(1) - 1.0)))

    r9, clo9 = _hyb(7)
    r11, _clo11 = _hyb(9)
    oracle_gap = abs(r9 - r11)

    def _pure(M):
        st = PMM2DStackPure(_P, n_modes=M, n_orders=1,
                            layer_grids="per-layer")
        st.add_tapered_pillar(0.30, **kw)
        st.set_source(_WL, theta=0.18, phi=0.35)
        o, R, T = st.solve(jones=False)
        p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
        return float(R[0, p0]), float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0)))

    r5, clo5 = _pure(4)
    r6, clo6 = _pure(5)
    r7, _clo7 = _pure(6)
    # 1. the pure arm CONVERGES (its own rung-to-rung gap shrinks).  RESTATED
    #    2026-09-11: the previous form was
    #       abs(r6-r5) < 0.5*max(abs(r5-r9), 1e-12) or abs(r6-r5) < 5e-2
    #    whose FIRST clause is FALSE on this fixture (|r6-r5| = 2.797e-02
    #    against 0.5*|r5-r9| = 1.442e-02), so it passed only through the
    #    escape hatch and asserted nothing.  With the third rung the claim is
    #    real: MEASURED |r(M5)-r(M4)| = 2.797e-02 and |r(M6)-r(M5)| =
    #    9.356e-04, i.e. 30x tighter, barred at 2x with 15x inside it.
    assert abs(r7 - r6) < 0.5 * abs(r6 - r5), (r5, r6, r7)
    # 2. ... it agrees with the hybrid to a few times the ORACLE's own
    #    self-gap -- the most this cross-engine comparison can assert ...
    assert abs(r6 - r9) < 20.0 * max(oracle_gap, 1e-6), (r6, r9, oracle_gap)
    # 3. ... and its lossless closure is DECADES tighter than the oracle's,
    #    which is what says the residual above is the oracle's floor showing.
    # MEASURED: hybrid degree-7 closure 8.34e-03 against the pure arm's
    # 5.17e-04 at M = 5 -- 16x tighter -- and 3.68e-03 against 3.64e-05 at
    # degree 9 / M = 6, i.e. 101x.  Barred at 4x, with the measured minimum
    # 16x inside it.
    assert clo6 < 0.25 * clo9, (clo6, clo9, clo5)


def test_tapered_pillars_multi_feature_and_its_refusals():
    """The N-feature ``add_tapered_pillars`` transplant: center-anchored,
    exact walls per slice, and the staggered basis's EQUAL-SEGMENT-COUNT
    constraint enforced loudly."""
    st = PMM2DStackPure(_P, n_modes=4, n_orders=1, layer_grids="per-layer")
    st.add_tapered_pillars(
        0.30, eps_host=2.25, n_slices=2,
        pillars=[((0.3 * _P, 0.3 * _P), (0.18 * _P, 0.18 * _P),
                  (0.26 * _P, 0.26 * _P), 6.0),
                 ((0.72 * _P, 0.72 * _P), (0.16 * _P, 0.16 * _P),
                  (0.22 * _P, 0.22 * _P), 5.0)])
    assert len(st._layers) == 2
    assert all(np.asarray(L["wx"]).size == 6 for L in st._layers)
    st.set_source(_WL, theta=0.12, phi=0.25)
    o, R, T = st.solve(jones=False)
    assert float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0))) < 5e-2
    # two pillars SHARING a y-extent collapse the y-wall set to 2 while the
    # x-wall set stays 4 -- unequal segment counts, which this tensor basis
    # cannot carry, and it says so instead of building a silently wrong grid
    bad = PMM2DStackPure(_P, n_modes=5, layer_grids="per-layer")
    with pytest.raises(ValueError, match="EQUAL SEGMENT COUNTS"):
        bad.add_tapered_pillars(
            0.30, eps_host=2.25, n_slices=1,
            pillars=[((0.3 * _P, 0.5 * _P), (0.18 * _P, 0.18 * _P),
                      (0.18 * _P, 0.18 * _P), 6.0),
                     ((0.72 * _P, 0.5 * _P), (0.16 * _P, 0.18 * _P),
                      (0.16 * _P, 0.18 * _P), 5.0)])
