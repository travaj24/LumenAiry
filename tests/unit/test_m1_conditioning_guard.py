"""M1 -- "no solver draws an arbitrary answer" (X-1, N-2, T3-3, 2026-08-04).

``docs/audits/PMM_M1_CONDITIONING_2026_08_04.md``.

Three unguarded numerical solves were hardened across BOTH rigorous solvers:

* **X-1** ``rcwa/_core.py`` -- the explicit ``inv(a+b)`` in
  :func:`_interface_smatrix` (whose ``cond ~1e13`` the module's own
  ``_check_energy`` docstring records), the two star denominators in
  :func:`_redheffer_star` (which the census found to be the DOMINANT site --
  ``cond`` 2.4e31 where the interface behind it read 3.1e16), and the ``T22``
  inverse of the 4N generalized interface;
* **N-2** ``pmm/_core.py`` -- the same three shapes on the PMM side plus the
  Rayleigh-projection least squares;
* **T3-3** ``pmm/conical.py`` -- the per-layer far-field order cap, which was
  computed from the FULL-UNION cell count on a path whose half-spaces live on
  the three-layer window grids.

The contract is: **screen for free, score on the equations, refuse what no
build agrees on, and return historical bits everywhere else.**  Every claim
below is a null control, a fail-before, or a measured separation -- there is no
absolute accuracy bar in this file that is not comparative.
"""
import warnings

import numpy as np
import pytest

import lumenairy.elements.pmm._core as _pc
import lumenairy.elements.pmm.conical as _con
import lumenairy.elements.pmm.stack as _ps
import lumenairy.elements.rcwa._core as _rc
from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.rcwa import rcwa_efficiency_1d

WL = 700e-9
P = 1.0e-6

# the library's own documented instability class (`_check_energy`'s docstring,
# and `test_rcwa_reduces_to_thin_grating_limit`'s comment naming OpenBLAS
# tripping where MKL is clean): large period, low index contrast.
THIN = dict(period=10e-6, n_ridge=1.55, n_groove=1.5, n_substrate=1.5,
            n_superstrate=1.5, depth=0.5e-6, duty_cycle=0.5)

# the audit staircase: six lossless slices whose walls shift 4 nm per slice.
STAIR = [(60e-9, [(0.5 - 0.35 / 2 - 0.002 * i, 1.0 + 0j),
                  (0.35 + 0.004 * i, 4.0 + 0j),
                  (0.5 - 0.35 / 2 - 0.002 * i, 1.0 + 0j)])
         for i in range(6)]


# O-11 (2026-09-11, docs/audits/FIX_PMMSTACK_SLIVER_WALLS_2026_09_11.md).  The
# audit staircase above is EXACTLY the geometry the sliver refusal screens for:
# six slices whose walls shift 4 nm on a 1 um period, so the union grid carries
# 2 nm cross-layer cells, 157x finer than anything the layers asked for.  This
# file's whole purpose is to DRIVE that stack past its capacity and read the
# wrong answers that come back -- several of its arms deliberately disarm
# `INTERFACE_CONDITIONING_GUARD` or `PMM_CONICAL_PERLAYER_ORDER_CAP` to harvest
# a pre-fix draw whose `R+T` then reads 2.1 to 21 -- and the sliver refusal
# would turn those harvests into raises.  Which of them trip it is a BLAS fact
# (this file's own docstrings record closure moving from 6.7e-06 to 2.1e+01
# between one and two OpenBLAS threads on the SAME cell), so the switch is
# thrown for the MODULE rather than per arm.  Nothing here asserts the sliver
# behaviour; tests/unit/test_fix_pmmstack_sliver_walls.py owns it.
@pytest.fixture(autouse=True)
def _sliver_guard_off():
    prev = _ps.PMM_SLIVER_GUARD
    _ps.PMM_SLIVER_GUARD = False
    try:
        yield
    finally:
        _ps.PMM_SLIVER_GUARD = prev


@pytest.fixture
def guard_off():
    """The M1 fail-before switch, restored after the test."""
    prev = _rc.INTERFACE_CONDITIONING_GUARD
    _rc.INTERFACE_CONDITIONING_GUARD = False
    try:
        yield
    finally:
        _rc.INTERFACE_CONDITIONING_GUARD = prev


@pytest.fixture
def cap_off():
    """T3-3's fail-before switch, restored after the test."""
    prev = _con.PMM_CONICAL_PERLAYER_ORDER_CAP
    _con.PMM_CONICAL_PERLAYER_ORDER_CAP = False
    try:
        yield
    finally:
        _con.PMM_CONICAL_PERLAYER_ORDER_CAP = prev


def _stair_stack(ffo, grids, degree=6):
    st = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=degree,
                  far_field_orders=ffo, layer_grids=grids)
    for t, segs in STAIR:
        st.add_layer(t, segments=segs)
    return st


def _solve_conical(ffo, grids, degree=6):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _stair_stack(ffo, grids, degree).set_source(
            WL, theta=0.15, phi=0.6).solve()


def _closure(R, T):
    """``max`` over incident polarizations of ``|sum(R) + sum(T) - 1|``."""
    R, T = np.asarray(R), np.asarray(T)
    if R.ndim == 1:
        return abs(float(R.sum() + np.asarray(T).sum()) - 1.0)
    return float(np.max(np.abs(R.sum(axis=1) + T.sum(axis=1) - 1.0)))


# ---------------------------------------------------------------------------
# THE PER-BUILD PARTITION OF THE THIN-GRATING LADDER
#
# Everything in this file that used to name a truncation now asks this scan.
# The reason is the four-name adjudication's S3 finding, re-confirmed on CI in
# ``docs/audits/FIX_CI_M1_T34_2026_08_06.md``: the DEVICE and the DEFECT are
# build-independent -- the library's own documented instability class always
# contains machine-precision truncations AND silently-wrong ones -- but WHICH
# ``n_orders`` lands in which class is decided by whether a near-cancelling
# star denominator's smallest singular value falls above or below the screen,
# i.e. by the BLAS reduction order and the LAPACK build.  Measured, same code,
# same geometry, ONLY the environment varying:
#
#   cell     Windows/np2.4.4 (1, 2, 24 thr)   ubuntu CI (np2.2.6 .. 2.4.6)
#   12 TE    flagged, rcond 1.3e-10           NOT flagged (census empty)
#   20 TE    flagged, rcond 2.3e-11           NOT flagged
#   19 TE    flagged                          flagged
#   21 TE    flagged; sum(R) 3.2e-2 / 6.1e-3  flagged; sum(R) 2.095e-4 (clean)
#   22 TM    closure 3.5e-13 (clean)          closure 3.760e-03 (NOT clean)
#
# A list of cells is therefore a per-build fact asserted as a universal one.
# What IS universal is asserted below, on the SCAN.
# ---------------------------------------------------------------------------

#: The truncations scanned.  Wide enough that the partition below is never
#: empty on any build measured (Windows x {1, 2, 24} threads, WSL x {1, 2, 64}
#: threads, and the five ubuntu CI images).
_THIN_LADDER = tuple(range(6, 31))

#: A truncation is CLEAN when it returns and closes to better than this.  Not
#: a calibration: the two populations are six decades apart on every build
#: measured -- clean truncations close at 0.0 .. 3.4e-10 and the next-worst
#: returning truncation closes at 3.5e-06 (Windows) / 5.1e-05 (CI).
_THIN_CLEAN_CLOSURE = 1e-9

#: A truncation is a DEFECT CANDIDATE when it returns an answer while missing
#: closure by more than this.  Classified on CLOSURE ALONE, deliberately: the
#: claim the defect set then carries -- that its ``sum(R)`` is orders outside
#: the clean population's own spread -- has to be an assertion, not a
#: definition, or the test proves itself.
_THIN_DEFECT_CLOSURE = 1e-6

_THIN_SCAN = {}


def _thin_solve(M, pol):
    """One thin-grating solve with the M1 census armed, never raising.

    ``dict(M, pol, raised, close, sumR, census, flagged, refused)``.
    """
    prev = _rc._INV_CENSUS
    _rc._INV_CENSUS = []
    raised, close, sumR = None, float("nan"), float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                o, R, T = rcwa_efficiency_1d(
                    THIN["period"], THIN["n_ridge"], THIN["n_groove"],
                    THIN["n_substrate"], THIN["n_superstrate"], THIN["depth"],
                    THIN["duty_cycle"], WL, angle=0.0, polarization=pol,
                    n_orders=M, stabilize=False)
                close, sumR = _closure(R, T), float(np.sum(R))
            except _rc._EnergyError as exc:
                raised = type(exc).__name__
        census = list(_rc._INV_CENSUS)
    finally:
        _rc._INV_CENSUS = prev
    return dict(M=M, pol=pol, raised=raised, close=close, sumR=sumR,
                census=census,
                flagged=[c for c in census
                         if np.isfinite(c[2]) and c[2] < _rc._INV_RCOND_SCREEN],
                refused=[c for c in census if c[4]])


def _thin_scan(pol):
    """The whole ladder for one polarization, solved ONCE per process."""
    if pol not in _THIN_SCAN:
        _THIN_SCAN[pol] = [_thin_solve(M, pol) for M in _THIN_LADDER]
    return _THIN_SCAN[pol]


def _thin_clean(pol):
    return [r for r in _thin_scan(pol)
            if r["raised"] is None and r["close"] < _THIN_CLEAN_CLOSURE]


def _thin_clean_sumR(pol):
    """The build's own converged ``sum(R)``: the median over its clean set."""
    clean = _thin_clean(pol)
    return float(np.median([r["sumR"] for r in clean])) if clean else float("nan")


def _thin_converged(pol):
    """The ladder's CONVERGED tail on this build: the clean truncations in the
    top third by ``n_orders``.

    The whole clean population spans 5 % in ``sum(R)`` -- that is truncation
    convergence from ``n_orders`` 6 up, not disagreement -- so it is the wrong
    yardstick for "this answer is wrong".  The tail has stopped moving:
    measured 2.0941e-04 +/- 1e-05 relative (TE) and 2.0138e-04 +/- 1e-05 (TM)
    at every thread count on both mounts.
    """
    clean = sorted(_thin_clean(pol), key=lambda r: r["M"])
    return clean[-max(3, len(clean) // 3):] if len(clean) >= 3 else clean


def _thin_converged_sumR(pol):
    tail = _thin_converged(pol)
    return float(np.median([r["sumR"] for r in tail])) if tail else float("nan")


def _thin_converged_spread(pol):
    """How much the CONVERGED tail disagrees with itself on this build -- the
    yardstick every "wrong" claim below is measured in.  Floored at 1e-3
    because that is the finest the ladder resolves anything (measured tail
    spread 1e-05); the floor, not the measurement, is what the bars see."""
    ref, tail = _thin_converged_sumR(pol), _thin_converged(pol)
    if not tail or not np.isfinite(ref):
        return float("nan")
    return max(max(abs(r["sumR"] / ref - 1.0) for r in tail), 1e-3)


def _thin_defects(pol):
    """The truncations that RETURN an answer while breaking closure."""
    return [r for r in _thin_scan(pol)
            if r["raised"] is None and np.isfinite(r["close"])
            and r["close"] > _THIN_DEFECT_CLOSURE]


def _thin_wrongness(r):
    """``|sum(R)/converged - 1|`` in units of the converged tail's own spread,
    for the cell's OWN polarization.  ``nan`` if that pol has no converged tail
    to measure against."""
    ref = _thin_converged_sumR(r["pol"])
    spread = _thin_converged_spread(r["pol"])
    if not (np.isfinite(ref) and np.isfinite(spread)
            and np.isfinite(r["sumR"])):
        return float("nan")
    return abs(r["sumR"] / ref - 1.0) / spread


def _thin_flagged(pol):
    """The truncations whose census SEES the conditioning on this build."""
    return [r for r in _thin_scan(pol) if r["flagged"]]


def _thin_table(pols=("te", "tm")):
    """One printable line per scanned cell -- the probe behind every message
    this file emits when a pinned cell has migrated."""
    out = []
    for pol in pols:
        for r in _thin_scan(pol):
            out.append("%2d %s %-14s close=%9.3e sumR=%10.4e flagged=%d"
                       % (r["M"], pol, r["raised"] or "returned", r["close"],
                          r["sumR"], len(r["flagged"])))
    return "\n    ".join(out)


def _census_instrument_is_alive():
    """PROBE: does the census still SEE a singular operator on this build?

    The only reading that separates "the X-1 defect does not manifest in this
    environment" from "the instrument was turned off" -- and the difference
    decides whether a missing defect is a skip or a failure.
    """
    n = 24
    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n))
                        + 1j * rng.standard_normal((n, n)))
    A = (Q * np.logspace(0, -20, n)) @ Q.conj().T
    prev = _rc._INV_CENSUS
    _rc._INV_CENSUS = []
    try:
        _rc._guarded_inverse(A, "probe")
        rows = list(_rc._INV_CENSUS)
    finally:
        _rc._INV_CENSUS = prev
    return bool(rows) and np.isfinite(rows[0][2]) \
        and rows[0][2] < _rc._INV_RCOND_SCREEN


# ---------------------------------------------------------------------------
# the instruments themselves
# ---------------------------------------------------------------------------

def test_rcond_1_is_the_exact_condition_number_not_an_estimate():
    """``_rcond_1`` must be ``1 / (||A||_1 ||A^-1||_1)`` exactly -- it is the
    screen, and a screen whose value drifts with the BLAS build would put two
    builds on opposite sides of it."""
    rng = np.random.default_rng(7)
    for n in (4, 17, 40):
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        X = np.linalg.inv(A)
        want = 1.0 / (np.linalg.norm(A, 1) * np.linalg.norm(X, 1))
        assert _rc._rcond_1(A, X) == pytest.approx(want, rel=0, abs=0)
    # a diagonal with a known spread: cond_1 is exactly hi/lo
    D = np.diag(np.array([1e-6, 1.0, 1e3], dtype=complex))
    assert _rc._rcond_1(D, np.linalg.inv(D)) == pytest.approx(1e-6 / 1e3,
                                                              rel=1e-12)
    # unusable input routes to the residual check (the safe direction)
    assert _rc._rcond_1(np.array([[np.nan]]), np.array([[1.0]])) == 0.0
    assert _rc._rcond_1(np.zeros((2, 3)), np.zeros((2, 3))) == 0.0


def test_inverse_residual_is_per_entry_and_scale_free_in_n():
    """The bar is a per-entry one, so the residual must not drift with the
    truncation: an exact inverse reads 0 at every ``n``, and a uniformly
    perturbed one reads the perturbation at every ``n``."""
    for n in (4, 32, 100):
        A = np.eye(n, dtype=complex)
        assert _rc._inverse_residual(A, A) == 0.0
        X = A + 1e-9 * np.eye(n, dtype=complex)
        assert _rc._inverse_residual(A, X) == pytest.approx(1e-9, rel=1e-9)


# ---------------------------------------------------------------------------
# NULL CONTROL: a well-conditioned solve is bit-identical, guard on or off
# ---------------------------------------------------------------------------

def test_guarded_inverse_is_bit_identical_above_the_screen():
    """Tolerance-at-0.0 on the max absolute difference, per the standing rule
    (never ``array_equal``)."""
    rng = np.random.default_rng(11)
    for n in (5, 33, 64):
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        assert _rc._rcond_1(A, np.linalg.inv(A)) >= _rc._INV_RCOND_SCREEN
        got = _rc._guarded_inverse(A, "probe")
        assert float(np.max(np.abs(got - np.linalg.inv(A)))) == 0.0


def test_well_conditioned_rcwa_solve_is_bit_identical_guard_on_or_off(
        guard_off):
    """The null-floor control the M1 gate names: an ordinary sub-wavelength
    high-contrast grating (the census measured ``cond(a+b)`` = 9.0 there) must
    return the SAME BITS with the guard on and off."""
    def run():
        return rcwa_efficiency_1d(0.5e-6, 2.0, 1.0, 1.5, 1.0, 0.3e-6, 0.5, WL,
                                  angle=np.deg2rad(8.0), polarization="te",
                                  n_orders=12)
    o0, R0, T0 = run()                       # guard OFF (fixture)
    _rc.INTERFACE_CONDITIONING_GUARD = True
    o1, R1, T1 = run()
    assert float(np.max(np.abs(np.asarray(R0) - np.asarray(R1)))) == 0.0
    assert float(np.max(np.abs(np.asarray(T0) - np.asarray(T1)))) == 0.0


def test_conforming_per_layer_stack_is_bit_identical_guard_on_or_off(
        guard_off):
    """The per-layer null control: on a CONFORMING stack the mortar is bypassed
    entirely, so nothing the guard touches may move a bit."""
    lay = [(220e-9, [(0.30, 4.0 + 0j), (0.70, 1.0 + 0j)]),
           (180e-9, [(0.50, 2.25 + 0j), (0.50, 1.0 + 0j)])]

    def run():
        st = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=6,
                      far_field_orders=7, layer_grids="per-layer")
        for t, segs in lay:
            st.add_layer(t, segments=segs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return st.set_source(WL, theta=np.deg2rad(8.0)).solve()
    o0, R0, T0, J0 = run()                   # guard OFF (fixture)
    _rc.INTERFACE_CONDITIONING_GUARD = True
    o1, R1, T1, J1 = run()
    assert float(np.max(np.abs(np.asarray(J0) - np.asarray(J1)))) == 0.0
    assert float(np.max(np.abs(np.asarray(R0) - np.asarray(R1)))) == 0.0
    assert float(np.max(np.abs(np.asarray(T0) - np.asarray(T1)))) == 0.0


# ---------------------------------------------------------------------------
# the refusal, and that it is a REFUSAL and not a step-down
# ---------------------------------------------------------------------------

def test_the_inverse_refusal_was_withdrawn_and_the_instruments_record_instead():
    """THE REFUTATION, pinned.

    The first cut of this guard REFUSED a numerically singular cascade inverse.
    The breadth sweep killed it: the thresholds were calibrated on two 1-D
    families, and the 2-D hybrid interface -- correct, build-stable and pinned
    since v5.14 -- reads INSIDE the 1-D broken band on BOTH instruments
    (equilibrated rcond 3.9e-14 against a 1-D broken band of 3.8e-19..1.3e-10;
    equilibrated residual 1.2e-05 against 5.3e-08..3.7e+07).  No global bar
    exists, so the refusal was withdrawn.

    What must now hold: :func:`_guarded_inverse` NEVER raises and NEVER moves a
    bit, however singular its argument."""
    n = 24
    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n))
                        + 1j * rng.standard_normal((n, n)))
    A = (Q * np.logspace(0, -20, n)) @ Q.conj().T      # cond 1e20
    got = _rc._guarded_inverse(A, "probe")             # must NOT raise
    assert float(np.max(np.abs(got - np.linalg.inv(A)))) == 0.0
    # the instruments still SEE it -- they simply do not act on it
    _rc._INV_CENSUS = []
    try:
        _rc._guarded_inverse(A, "probe")
        assert len(_rc._INV_CENSUS) == 1
        _site, _dim, rcond_eq, resid_eq, refused = _rc._INV_CENSUS[0]
        assert refused is False
        assert rcond_eq < _rc._INV_RCOND_SCREEN
        assert resid_eq > _rc._INV_RESID_REFUSE
    finally:
        _rc._INV_CENSUS = None


def test_a_nan_material_index_is_left_to_the_named_nan_guard():
    """A NaN substrate must reach ``_check_energy``'s precise diagnosis
    ("non-finite total efficiency ... a NaN/inf material index"), NOT a
    conditioning message.

    The first cut hijacked it: a NaN operand made the equilibrated rcond read
    0.0 and the residual read inf, so the conditioning guard raised first and
    told the user their truncation was singular when their material index was
    NaN -- a strictly worse diagnostic.  A non-finite operand is a propagation
    defect, not a conditioning one, and the guard now stands aside."""
    A = np.eye(4, dtype=complex)
    A[2, 2] = np.nan
    got = _rc._guarded_inverse(A, "probe")             # must NOT raise
    assert got.shape == A.shape
    _rc._INV_CENSUS = []
    try:
        _rc._guarded_inverse(A, "probe")
        assert _rc._INV_CENSUS[0][4] is False           # not refused
    finally:
        _rc._INV_CENSUS = None


def test_conditioning_error_is_an_energy_error_so_stabilize_routes_around_it():
    """The refusal is deliberately an ``_EnergyError`` subclass: every existing
    ``stabilize=`` ladder already catches that and steps ``n_orders``, so a
    singular truncation is routed around with no ladder change.  ``ValueError``
    handlers upstream are likewise unaffected."""
    assert issubclass(_rc._ConditioningError, _rc._EnergyError)
    assert issubclass(_rc._ConditioningError, ValueError)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T = rcwa_efficiency_1d(
            THIN["period"], THIN["n_ridge"], THIN["n_groove"],
            THIN["n_substrate"], THIN["n_superstrate"], THIN["depth"],
            THIN["duty_cycle"], WL, angle=0.0, polarization="te",
            n_orders=19, stabilize=True)
    # 19 TE is one of the refused truncations; the ladder must deliver a
    # CONSERVING solve from a nearby one instead of propagating the refusal.
    assert abs(float(np.sum(R) + np.sum(T)) - 1.0) < 1e-6


def test_the_step_down_is_not_a_re_solve_the_census_says_there_is_none():
    """The C13 pattern's step-down is DELIBERATELY absent here, and this pins
    the measurement that decided it: on a matrix in the refused class there is
    nothing to step DOWN TO.  Householder QR, column-pivoted QR and the SVD
    pseudo-inverse all land on the SAME ``||A X - I||`` as the LU inverse, and
    only iterative refinement lowers it -- which is why refinement is the one
    candidate the guard tried.

    **v5.33.1 -- THE ORDERING WAS A SINGLE-DRAW COIN FLIP.**  It was
    ``resid(lu) <= resid(qr)`` on ONE matrix, and the two numbers are the same
    number to within a few percent, so which side of the ``<=`` a build lands
    on is round-off.  Measured on the original draw (seed 5, n = 40,
    cond 1e14), code and geometry fixed::

        build                       resid(lu)    resid(qr)   lu/qr
        Windows np2.4.4, 1/2/24 thr 9.5257e-04   9.9054e-04  0.962   <- passes
        ubuntu CI  np2.2.6 .. 2.4.6 9.3063e-04   8.8442e-04  1.052   <- FAILS

    A 5 % excursion decided a strict inequality.  Over the 15 draws scored
    below x 6 (mount x thread-count) cells the ratio runs 0.61 .. 1.32 and
    changes SIDE with the seed: the ordering is not a fact about the
    algorithms.

    What IS a fact, in every one of those cells: no direct alternative is even
    2x better (worst measured 1.32), so no step-down buys anything; and one
    step of refinement is the one route that helps the FAMILY.

    **v5.33.2 -- "EVERY DRAW" WAS THE SAME COIN FLIP ONE LEVEL DOWN**
    (``FIX_CI_ROUND2_PMM_2026_08_08``).  The second claim was scored as
    ``resid(ir) < resid(lu)`` on each of the 15 draws (worst locally 0.942).
    On ubuntu CI py3.12 one draw read

        seed 5, n 24:  resid(ir) = 9.7681e-04   resid(lu) = 8.8175e-04
                       ratio 1.108

    -- refinement within 11 % of LU, i.e. INSIDE the same few-percent band
    that decided the ordering this test was already restructured for.  The
    docstring above says the decision-relevant statement is "no direct
    alternative is MATERIALLY better", and per-draw strict improvement is a
    stronger statement than that: it is not what the guard's design rests on,
    it is not what the family measures, and it is round-off on any draw where
    the two residuals coincide.

    So claim (2) is aligned with the claim it states, in the SAME 2x unit
    claim (1) already uses -- refinement never makes a draw materially worse
    (worst measured 1.108 against a 2.0 bar, 1.8x headroom) -- and the
    "refinement is what helps" half is kept as the FAMILY claims it always
    was: the median improves by at least 5 % (measured 0.80) and a majority
    of draws improve outright.  Nothing is relaxed that the guard's design
    rests on; the per-draw strict ordering was never that."""
    sla = pytest.importorskip("scipy.linalg")
    rows = []
    for seed in (5, 6, 7, 8, 9):
        for n in (24, 40, 64):
            rng = np.random.default_rng(seed)
            Q, _ = np.linalg.qr(rng.standard_normal((n, n))
                                + 1j * rng.standard_normal((n, n)))
            # cond 1e14: the refused class.  (1e16 is past float64's reach --
            # there BOTH the ordering AND refinement stop meaning anything.)
            A = (Q * np.logspace(0, -14, n)) @ Q.conj().T

            def resid(X):
                return _rc._inverse_residual(A, X)
            x_lu = np.linalg.inv(A)
            Qf, Rf = sla.qr(A, mode='economic', check_finite=False)
            x_qr = sla.solve_triangular(Rf, Qf.conj().T, check_finite=False)
            Qp, Rp, piv = sla.qr(A, mode='economic', pivoting=True,
                                 check_finite=False)
            Xp = sla.solve_triangular(Rp, Qp.conj().T, check_finite=False)
            x_qp = np.empty_like(Xp)
            x_qp[piv, :] = Xp                 # undo the column permutation
            x_svd = np.linalg.pinv(A, rcond=1e-15)
            x_ir = x_lu + x_lu @ (np.eye(n, dtype=complex) - A @ x_lu)
            rows.append((seed, n, resid(x_lu), min(resid(x_qr), resid(x_qp),
                                                   resid(x_svd)), resid(x_ir)))

    # (1) there is nothing to step DOWN to: the best of three direct
    #     alternatives never beats LU by even 2x.  Measured worst 1.32.
    for seed, n, lu, alt, _ir in rows:
        assert lu < 2.0 * alt, (
            f"seed {seed} n {n}: the best direct alternative reads {alt:.4e} "
            f"against LU's {lu:.4e} -- an alternative that good IS a "
            f"step-down, and the guard would have to take it")
    # ... and on the median draw the two are the same number.
    assert float(np.median([lu / alt for _s, _n, lu, alt, _i in rows])) < 1.5
    # (2) refinement is the one route that helps -- scored in the same "not
    #     MATERIALLY different" unit as (1), per draw, and as a FAMILY for the
    #     improvement itself.  See the docstring for why the per-draw strict
    #     ordering was the wrong assertion for this claim.
    for seed, n, lu, _alt, ir in rows:
        assert ir < 2.0 * lu, (
            f"seed {seed} n {n}: refinement read {ir:.4e} against LU's "
            f"{lu:.4e} -- one step of refinement is making the inverse "
            f"MATERIALLY worse, which is not round-off and not what the "
            f"guard's one candidate is supposed to do")
    ratios = [ir / lu for _s, _n, lu, _a, ir in rows]
    assert float(np.median(ratios)) < 0.95, (
        f"refinement no longer improves the MEDIAN draw of the class "
        f"(median ratio {float(np.median(ratios)):.4g}) -- the one candidate "
        f"the guard tried stopped helping.  ratios: "
        f"{[round(r, 4) for r in ratios]}")
    n_better = sum(1 for r in ratios if r < 1.0)
    assert n_better > len(rows) / 2, (
        f"refinement improved only {n_better} of {len(rows)} draws: it is no "
        f"longer the route that helps this class.  ratios: "
        f"{[round(r, 4) for r in ratios]}")


def test_equilibration_is_what_separates_scaling_from_singularity():
    """A badly SCALED but perfectly invertible operator must pass, and the
    unequilibrated screen must be shown to fail it -- this is the measurement
    that chose the instrument, kept as a test.

    A diagonal with a 1e-9 spread has raw ``cond_1`` = 1e9 and equilibrates to
    the identity."""
    n = 12
    A = np.diag(np.linspace(1.0, 1e-9, n)).astype(complex)
    X = np.linalg.inv(A)
    assert _rc._rcond_1(A, X) < 1e-8         # the raw screen would fire
    assert _rc._rcond_1_equilibrated(A, X) == pytest.approx(1.0, rel=1e-12)
    got = _rc._guarded_inverse(A, "probe")
    assert float(np.max(np.abs(got - X))) == 0.0


def test_the_equilibrated_rcond_identity_needs_no_second_factorisation():
    """``(R^-1 A C^-1)^-1 = C A^-1 R`` exactly, which is what keeps the screen
    free.  Pinned against an explicit inverse of the equilibrated matrix."""
    rng = np.random.default_rng(31)
    for n in (6, 29):
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        A = A * np.logspace(-6, 6, n)[:, None]     # wildly scaled rows
        X = np.linalg.inv(A)
        r, c, _As = _rc._equilibration(A)
        Ae = (A / r[:, None]) / c[None, :]
        want = 1.0 / (np.linalg.norm(Ae, 1)
                      * np.linalg.norm(np.linalg.inv(Ae), 1))
        assert _rc._rcond_1_equilibrated(A, X) == pytest.approx(want,
                                                                rel=1e-8)


def test_anisotropic_cascade_is_not_falsely_refused():
    """THE FALSE-POSITIVE REGRESSION GUARD, and the reason the guard scores an
    equilibrated operator rather than the raw one.

    **RESTATED 2026-09-11 -- this cell's ill-conditioning WAS the RCWA
    branch-cut defect** (`docs/audits/FIX_RCWA_EVEN_SECTOR_WSL_2026_09_11.md`):
    a propagating mode of the lossless uniaxial layer was handed the INCOMING
    root of ``lam^2`` because the outgoing-root pin tested for an EXACTLY zero
    real part, and against the groove/substrate index coincidence
    (``eps_groove`` = ``n_sub^2`` = 2.25) that mode duplicated a substrate
    backward mode, so the generalized interface's ``T22`` went near-singular.
    Everything the paragraphs below attribute to "near-cancelling
    deep-evanescent star denominators" and "BLAS reduction order" was that:
    measured on the pre-fix tree (48c8747) at ``M`` = 5 / 9 / 15 / 19 / 25 the
    RAW inverse residual read 5.2e-01 / 2.8e-01 / 1.2e-02 / 1.7e-02 / 1.1e-02,
    ``rcond`` 1.7e-03 .. 1.2e-05, and ``|R+T-2|`` 2.0e-02 / 2.1e-03 / 1.3e-04
    / 1.5e-04 / 2.0e-03 at one thread against 2.0e-02 / 2.1e-03 / 2.0e-03 /
    4.4e-16 / 9.3e-03 at four.  On the fixed tree the same rungs read raw
    residual <= 5.5e-16, equilibrated residual <= 5.4e-16, ``rcond`` >= 2.3e-02
    and ``|R+T-2|`` <= 2.1e-14 -- at 1, 4 and 8 threads and on both builds
    (Windows py3.14 / OpenBLAS Haswell, WSL py3.12 / OpenBLAS SkylakeX) --
    while ``J00`` is unchanged to six figures on every rung.  So the closure IS
    deterministic and IS a valid tripwire here, the raw and equilibrated
    instruments now AGREE, and premise (b) below is inverted: this cell no
    longer motivates the equilibration.  Whether ANY fixture still does is the
    subject of the fix's independent verification; the rationale text below is
    kept for the record and is no longer the claim.  What is asserted now:
    (a) nothing refused; (b') raw residual < 1e-3 x the refusal bar (measured
    5.5e-16: 4.3 decades under the assertion, the bar 3 decades above it),
    equilibrated residual likewise, ``rcond`` > 10 x the screen, and the two
    residual instruments within 10x of each other (measured 0.7x .. 1.3x);
    (c) and (d) unchanged; (f') the closure ladder is ASSERTED at
    ``|R+T-2| < 1e-10`` on every rung -- 6.7 decades above the measured
    2.1e-14 and 8.3 decades below the pre-fix symptom 2.0e-02, on both builds.

    ``rcwa_jones_1d`` on a uniaxial cell builds star denominators with
    ``||I - B11 A22||_1`` ~ 1e16-1e17 at EVERY truncation -- deep-evanescent
    blocks of the generalized S-matrix -- so their RAW inverse residual runs
    1e-2 to 5e-1 while the answer is right and both BLAS builds agree to twelve
    digits.  A raw-residual bar refuses all of them; the equilibrated one reads
    1e-15 .. 6e-14 and passes all of them.

    **v5.33.1 -- THE ENERGY BAR WAS NOT MEASURING THE CLAIM.**  The test also
    asserted ``|R+T-2| < 5e-3`` per rung.  On this cell that residual is NOT a
    convergence indicator: it plateaus at ~1e-3 after M = 7 and then jitters
    with the BLAS reduction order over more than two decades, because the
    cascade carries near-degenerate deep-evanescent pairs::

        rung  Win 1 thr   Win 2 thr   Win 24 thr   ubuntu CI
        M=15  1.259e-04   5.401e-05   1.244e-03    -
        M=17  1.664e-03   1.247e-03   4.480e-03    -    <- 1.1x under the bar
        M=25  2.041e-03   7.008e-05   3.811e-03    -
        worst of (9,15,19,25)                      1.312e-02   <- FAILED

    The 5e-3 constant was calibrated on two environments and the CI excursion
    walked straight through it.

    **v5.33.2 -- THE ANCHOR RUNG WAS NOT AN ANCHOR EITHER**
    (``FIX_CI_ROUND2_PMM_2026_08_08``).  The replacement scored the refined
    rungs against the ladder's own "deliberately unconverged" M = 5 rung.  On
    ubuntu CI py3.10/3.11 that rung closed at ``1.937e-05`` -- three decades
    TIGHTER than the 2.020e-02 it reads here -- so ``5 x loose`` became a
    1e-04 bar and a refined rung reading 1.312e-02 walked through that one
    too.  ``max([1.3116e-02, 1.0457e-03, 9.9920e-15, 1.1106e-03])`` is the CI
    ladder verbatim: 12 decades of spread inside one run.

    The reason no rung can be the anchor is now measured rather than supposed.
    Holding code, geometry and rung fixed and varying ONLY the BLAS pool and
    the mount [M]::

        rung   |R+T-2|                                              J00
               Win 1 thr   Win 2 thr   Win 24 thr  WSL 1 thr   WSL 2 thr
        M= 3   4.2577e-03  4.2577e-03  4.2577e-03  4.8709e-02  4.8709e-02   =
        M= 5   2.0200e-02  2.0200e-02  2.0200e-02  4.4875e-03  4.4875e-03   =
        M= 7   4.4301e-04  4.4301e-04  4.4301e-04  5.4675e-04  5.4675e-04   =
        M= 9   2.0581e-03  2.0581e-03  2.0581e-03  3.9565e-03  3.9565e-03   =
        M=15   1.2593e-04  5.4012e-05  1.2443e-03  1.7622e-03  4.4269e-04   =
        M=19   1.5227e-04  2.4035e-04  1.1724e-04  5.4986e-04  2.8244e-04   =
        M=25   2.0413e-03  7.0078e-05  3.8106e-03  1.2089e-03  2.0599e-03   =
        M=29   2.5313e-14  4.2829e-04  6.2761e-04  2.0662e-03  2.0028e-04   =

    ``J00`` in the last column is IDENTICAL to every printed digit in all ten
    cells -- across two pythons (3.14.6 / 3.12.3), two numpys (2.4.4 / 2.4.6)
    and two OpenBLAS kernels (Haswell / SkylakeX) -- while the SAME rung's
    closure moves by up to 54x with the pool and 11x with the mount, and the
    sequence is non-monotone in M on every one of them.  On a LOSSLESS cascade
    ``R + T = 2`` is very nearly tautological -- the campaign's standing
    caution -- so what this residual measures is the round-off of the
    near-cancelling deep-evanescent star denominators, not the truncation.  A
    quantity with no systematic content cannot carry a convergence claim under
    ANY bar, absolute or relative, so it is no longer asked to.

    What is asserted, all comparative or instrument-anchored:

      (a) NOTHING is refused -- the regression guard proper;
      (b) the RAW instruments would refuse every rung (raw residual 1.1e-2 ..
          2.8e-1 against ``_INV_RESID_REFUSE`` = 1e-8) and the EQUILIBRATED
          ones pass every rung by five orders (1e-15 .. 6e-14) -- the measured
          separation that chose the instrument, ~1e6 x 1e5 wide;
      (c) ``J00`` is stationary;
      (d) ``J00`` is CAUCHY -- ``|J00(M) - J00(M_finest)|`` strictly DECREASES
          rung by rung, 1.1357e-04 -> 3.6167e-05 -> 1.0151e-05 -> 4.244e-06,
          ~3x per step, IDENTICAL on both mounts at every thread count (the
          table above).  This is the claim "refining the truncation is not
          making the cascade worse", made on the quantity that actually
          converges and that does not move with the pool.  It is strictly
          stronger than the energy ratio it replaces: an energy bar cannot see
          a cascade that drifts to a wrong-but-unitary answer, which is the
          failure mode this whole campaign exists for;
      (e) the closure ladder is PRINTED, never asserted on.  The evidence
          that it is round-off is the CROSS-POOL table above, and a test
          cannot vary the BLAS pool in-process; every single-process statistic
          on it is itself a coin flip.  That was measured rather than assumed
          -- see the comment at the print;
      (f) and the only thing the closure CAN carry: the cascade has not blown
          up, ``|R+T-2| < 0.5`` on every rung.  Worst ever measured anywhere
          is 2.02e-02 (25x headroom) and the mis-assembled-cascade magnitude
          this class produces when it does go wrong is ``|R+T-1|`` = 21 (the
          ninth name, 40x the other side), so 0.5 sits inside a three-decade
          gap and is not a calibration.
    """
    from lumenairy.elements.rcwa import rcwa_jones_1d, uniaxial_tensor
    er = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=np.deg2rad(20))
    eg = (1.5 ** 2) * np.eye(3)

    # a test-local spy on the SAME operators the guard sees, so (b) is read off
    # the equations rather than asserted from the audit's prose.  It delegates
    # to the shipped function, so nothing about the solve changes.
    seen = []
    _orig = _rc._guarded_inverse

    def _spy(A, site, hint=None):
        A_np = np.asarray(A)
        if (A_np.ndim == 2 and A_np.shape[0] == A_np.shape[1]
                and np.all(np.isfinite(A_np))):
            X = np.linalg.inv(A_np)
            seen.append((_rc._inverse_residual(A_np, X),
                         _rc._equilibrated_inverse_residual(A_np),
                         _rc._rcond_1_equilibrated(A_np, X)))
        return _orig(A, site, hint)

    def run(M):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, J = rcwa_jones_1d(1.0e-6, er, eg, 1.5, 1.0, 0.4e-6, 0.5,
                                       WL, angle=np.deg2rad(10), n_orders=M)
        return (abs(float(np.sum(R) + np.sum(T)) - 2.0),
                complex(np.asarray(J)[0, 0]))

    _rc._guarded_inverse = _spy
    try:
        # M = 5 is the ladder's own coarsest rung; it is NOT part of the
        # stationarity claim (c) but IS part of the Cauchy claim (d).
        loose, j5 = run(5)
        rungs = [run(M) for M in (9, 15, 19, 25)]
    finally:                                  # (a) nothing raised, either
        _rc._guarded_inverse = _orig

    assert seen, "the spy never saw a guarded inverse: nothing was measured"
    raw = max(s[0] for s in seen)
    eq = max(s[1] for s in seen)
    eq_rc = min(s[2] for s in seen)
    # (b') RESTATED 2026-09-11: with the branch cut pinned the T22 operators
    #      on this cascade are well conditioned, so BOTH instruments sit deep
    #      under the refusal bar and agree.  The pre-fix shape (raw > 1e3 x
    #      bar, raw / eq > 1e9) is the defect's signature and would now FAIL.
    assert raw < 1.0e-3 * _rc._INV_RESID_REFUSE, (
        f"the raw residual on this cascade reads {raw:.3e}: the T22 operators "
        f"are ill-conditioned again, which on this cell means a propagating "
        f"layer mode carries the incoming root (the RCWA branch-cut defect "
        f"fixed 2026-09-11)")
    assert eq < 1.0e-3 * _rc._INV_RESID_REFUSE, (
        f"the EQUILIBRATED residual reads {eq:.3e} against the refusal bar "
        f"{_rc._INV_RESID_REFUSE:.0e} -- the guard is one round-off from "
        f"refusing a correct answer")
    # the free screen does not fire here either: measured rcond >= 2.3e-02
    # against the 1e-08 screen (pre-fix it read 1.2e-05 .. 1.7e-03).
    assert eq_rc > 1.0e1 * _rc._INV_RCOND_SCREEN
    assert 0.1 < raw / eq < 10.0, (            # measured 0.7x .. 1.3x
        f"the raw and equilibrated residuals disagree by {raw / eq:.3g}x on "
        f"a well-conditioned cascade")

    # (c) the ANSWER is stationary -- converging, not wandering.
    ref = rungs[0][1]
    for close, j00 in rungs[1:]:
        assert abs(j00 - ref) < 1e-4, (
            f"J00 moved to {j00!r} from {ref!r} across the ladder")
    # (d) THE CONVERGENCE CLAIM, on the quantity that converges: J00 is
    #     CAUCHY towards the finest rung.  Strictly decreasing, ~3x a step,
    #     and thread-count-independent to the last bit (see the docstring).
    ladder = [(5, j5)] + [(M, j) for M, (_c, j) in zip((9, 15, 19, 25), rungs)]
    finest = ladder[-1][1]
    tails = [(M, abs(j - finest)) for M, j in ladder[:-1]]
    for (Ma, da), (Mb, db) in zip(tails, tails[1:]):
        assert db < da, (
            f"J00 is not converging: |J00(M={Mb}) - J00(M=25)| = {db:.3e} is "
            f"no better than |J00(M={Ma}) - J00(M=25)| = {da:.3e}.  Refining "
            f"the truncation stopped improving the ANSWER, which is what "
            f"'making the cascade worse' means on a lossless cell where "
            f"R + T = 2 is nearly tautological.  ladder: "
            + str([(M, f'{d:.3e}') for M, d in tails]))
    # (e) the premise, PRINTED and not asserted -- and that is the finding.
    #     The evidence that |R+T-2| carries no truncation information on this
    #     cell is a CROSS-POOL measurement (the same rung's closure moving 54x
    #     with OPENBLAS_NUM_THREADS while its J00 does not move a bit), and a
    #     test cannot vary the BLAS pool in-process -- the same argument the
    #     ninth-name pin makes.  Every single-process statistic on it is
    #     itself a coin flip, which was measured the hard way: an earlier form
    #     of this line asserted the ladder spans >= 10x, and WSL at one thread
    #     read 8.16x (['4.487e-03', '3.956e-03', '1.762e-03', '5.499e-04',
    #     '1.209e-03']) while Windows reads 160x .. 173x and CI 1.3e12.
    #     Asserting a bar on a quantity this test exists to disqualify would
    #     have reproduced the disease one more level down.
    closes = [loose] + [c for c, _j in rungs]
    lo, hi = min(closes), max(closes)
    print(f"\nM1 anisotropic cascade |R+T-2| ladder (M = 5, 9, 15, 19, 25): "
          f"{[f'{c:.3e}' for c in closes]}  spread {hi / max(lo, 1e-300):.3g}x"
          f" -- round-off on the near-cancelling star denominators, NOT a "
          f"convergence indicator; the answer's own convergence is (d).")
    # (f') RESTATED 2026-09-11: the closure is deterministic on the fixed tree
    #      (<= 2.1e-14 on every rung at 1 / 4 / 8 threads, both builds) and
    #      is asserted with a derived two-sided bar: 1e-10 sits 6.7 decades
    #      above the measurement and 8.3 decades below the pre-fix symptom
    #      (2.0e-02, the branch-cut defect), inside a fourteen-decade gap.
    assert hi < 1.0e-10, (
        f"a rung closes at {hi:.3e} on a lossless cascade whose closure reads "
        f"<= 2.1e-14 on the fixed tree: either the cascade is mis-assembled "
        f"or a propagating mode carries the incoming root again "
        f"({[f'{c:.3e}' for c in closes]})")


# ---------------------------------------------------------------------------
# X-1 on the library's own documented instability class -- CLOSED 2026-09-11
# ---------------------------------------------------------------------------
#
# X-1 WAS the RCWA modal branch-cut defect, and the round-1 fix
# (`docs/audits/FIX_RCWA_EVEN_SECTOR_WSL_2026_09_11.md`) closes it.  The
# geometry says why: `THIN`'s groove index equals BOTH half-spaces' (1.5), so
# it is a permittivity coincidence on the substrate AND the superstrate side at
# once, and a propagating layer mode handed the INCOMING root is then exactly a
# half-space BACKWARD mode -- which makes the interface mode-match `a + b`,
# whose explicit inverse IS `S12`, singular.  That is the near-cancelling
# denominator this file's census was built to record.
#
# Re-measured independently on the whole ladder, both arms in one process, with
# the census armed exactly as the tests below arm it
# (`validation/probe_fix_branch_cut_round2/b3_x1.py`, 2026-09-11):
#
#   ladder / arm            raises  flagged cells  worst |R+T-1|  worst sum(R)
#   TE  WIN 1 thr  PRE         7         14          3.1956e-02     152.60x
#   TE  WIN 1 thr  POST        0          0          1.3323e-15    5.064e-02*
#   TE  WSL 1 thr  PRE         8         14          3.1956e-02     152.60x
#   TE  WSL 1 thr  POST        0          0          1.4433e-15    5.064e-02*
#   TM  WIN 1 thr  PRE         5          9          2.6165e-04       1.2998x
#   TM  WIN 1 thr  POST        0          0          9.9920e-16    1.887e-02*
#   TM  WSL 1 thr  PRE         5          9          1.5404e-04       0.7654x
#   TM  WSL 1 thr  POST        0          0          9.9920e-16    1.887e-02*
#
#   (*) the POST "worst" is the COARSEST rung, M = 6 -- ordinary truncation
#   convergence, not a defect; every finer rung is better.
#
# The four historically pinned cells, Windows one thread, sum(R):
#
#   M = 12 TE  2.016454e-04 -> 2.015824e-04
#   M = 19 TE  1.838764e-02 -> 2.053766e-04
#   M = 20 TE  2.088570e-04 -> 2.053491e-04
#   M = 21 TE  3.216567e-02 -> 2.095174e-04
#
# and M = 19 TE, the cell whose whole point was that it RETURNED 1.018 on
# Windows and RAISED on WSL, now returns 2.053766e-04 on BOTH builds.
#
# The two tests below are therefore restated as DECISIONS ABOUT THE CLOSED
# STATE.  Their fail-before is ENGINEERED rather than found -- the pre-round-1
# branch body is reinstalled in-process -- which is strictly stronger than the
# old form: it fires at every thread count on both builds instead of depending
# on which truncation a given BLAS reduction order happens to break.
# ---------------------------------------------------------------------------


def _pre_branch_cut_sqrt_decay(x, xp=None, band=1e-8):
    """The pre-round-1 body: the EXACT ``Re(r) == 0`` pin and the ``-r`` flip.

    An ``eig`` output never satisfies ``Re(sqrt(lam^2)) == 0`` -- its real part
    is the eigensolver's backward error, ~1e-16 -- so this pin fires only for
    the REGION modes, built in exact arithmetic, and never for a structured
    LAYER's.  Reinstalling it is how the tests below keep a fail-before now
    that the defect no longer occurs in the shipped tree.
    """
    from lumenairy.backend.array import array_namespace
    if xp is None:
        xp = array_namespace(x)
    x = xp.asarray(x).astype(complex)
    r = xp.sqrt(x)
    on_cut = r.real == 0
    return xp.where(on_cut & (r.imag < 0), -r, r)


class _pre_branch_cut:
    """Install :func:`_pre_branch_cut_sqrt_decay` at every module binding of
    the shared selector, for the duration of a ``with`` block."""

    _MODULES = ("lumenairy.elements.rcwa._core",
                "lumenairy.elements.rcwa.oned",
                "lumenairy.elements.rcwa.stack",
                "lumenairy.elements.pmm.twod")

    def __init__(self):
        self._saved = []

    def __enter__(self):
        import importlib
        for name in self._MODULES:
            mod = importlib.import_module(name)
            if hasattr(mod, "_sqrt_decay"):
                self._saved.append((mod, mod._sqrt_decay))
                mod._sqrt_decay = _pre_branch_cut_sqrt_decay
        return self

    def __exit__(self, *a):
        for mod, fn in self._saved:
            mod._sqrt_decay = fn
        return False


# ------------------------------------------------------------------ ROUND 3
# THE FAIL-BEFORE QUANTITY for the ladder test below.  Until 2026-09-11 its
# engineered arm closed on ``worst pre-fix sum(R) is 152x the converged
# value``.  That ratio is AMPLIFIED ROUNDING -- how far a singular mode-match
# throws the answer depends on where the singularity falls relative to the
# arithmetic -- so it is a per-kernel fact by its nature, and on CI's AMD
# runners it read 0.68x and the test failed for reproducing nothing.
#
# The two quantities below are decisions rather than readings.
#
#   THE SIGN CENSUS.  Count the modes the selector returns numerically ON THE
#   CUT (``|Re(lam)| <= _CUT_BAND_REL * max(max|lam|, 1)`` -- propagating and
#   lossless) but carrying the INCOMING root (``Im(lam) < 0``).  The S-matrix
#   recursion requires a layer's FORWARD set to carry the OUTGOING root
#   ``+i|kz|``; the incoming root IS the defect.  Post-fix the count is ZERO BY
#   CONSTRUCTION.  Pre-fix the sign is the eigensolver's backward error -- a
#   coin flip per mode -- so over 25 truncations "at least one" survives every
#   kernel.  Measured on the TE ladder, pre-arm: 413 (WIN-HASWELL) / 426
#   (WIN-PRESCOTT) / 444 (WIN-SANDYBRIDGE) / 413 (WSL-HASWELL) / 426
#   (WSL-PRESCOTT) of 2505 on-cut modes.
#
#   THE CONDITIONING.  Worst ``rcond(a+b)`` over the ladder, read from the M1
#   census this file already arms.  Measured over five (build x core-type)
#   samples:
#
#     arm     WIN-HAS    WIN-PRE    WIN-SAND   WSL-HAS    WSL-PRE
#     POST   6.2504e-02 6.2504e-02 6.2504e-02 6.2504e-02 6.2504e-02
#     PRE    3.331e-19  4.676e-21  6.217e-20  3.614e-19  2.712e-21
#
#   The POST row does not move a digit on any of the five.  The bar below sits
#   6.4 decades above the worst PRE reading and 10.2 decades below the POST
#   one -- seventeen decades of empty gap between the two populations.
_X1_PRE_RCOND = 1e-12


class _root_sign_census:
    """Wrap whatever ``_sqrt_decay`` is installed and count the on-cut modes it
    returns on the INCOMING root."""

    def __init__(self):
        self.incoming = 0
        self.oncut = 0

    def __enter__(self):
        import importlib
        self._saved = []

        def _wrap(inner):
            def wrapped(x, xp=None, band=_rc._CUT_BAND_REL):
                out = inner(x, xp, band)
                try:
                    lam = np.asarray(out)
                    if lam.size > 4 and np.all(np.isfinite(lam)):
                        scale = max(float(np.max(np.abs(lam))), 1.0)
                        on = np.abs(lam.real) <= _rc._CUT_BAND_REL * scale
                        self.oncut += int(on.sum())
                        self.incoming += int((on & (lam.imag < 0)).sum())
                except Exception:                    # instrument only
                    pass
                return out
            return wrapped

        for name in _pre_branch_cut._MODULES:
            mod = importlib.import_module(name)
            if hasattr(mod, "_sqrt_decay"):
                self._saved.append((mod, mod._sqrt_decay))
                mod._sqrt_decay = _wrap(mod._sqrt_decay)
        return self

    def __exit__(self, *a):
        for mod, fn in self._saved:
            mod._sqrt_decay = fn
        return False


def _worst_rcond(rows):
    """Worst LAPACK reciprocal condition the M1 census recorded over a scan --
    i.e. the mode-match ``a + b`` at its worst interface on the whole ladder."""
    vals = [c[2] for r in rows for c in r["census"] if np.isfinite(c[2])]
    return min(vals) if vals else float("nan")


def _thin_scan_uncached(pol):
    """The ladder, solved fresh -- never through ``_THIN_SCAN``, whose entries
    would otherwise carry an engineered arm's readings into every later test in
    the file."""
    return [_thin_solve(M, pol) for M in _THIN_LADDER]


#: The POST-fix closure envelope on the whole THIN ladder: <= 1.5e-15 over
#: 25 truncations x 2 polarizations x 2 builds.  The bar sits 6.8 decades above
#: it and 2.4 decades below the smallest PRE reading that manifests (3.5e-06,
#: 20 TE Windows), inside a gap the measurement leaves empty.
_X1_CLOSED_CLOSURE = 1e-8


@pytest.mark.parametrize("M,pol", [(19, "te"), (21, "te"), (12, "te"),
                                   (20, "te")])
def test_x1_is_closed_on_the_cell_it_was_pinned_at(M, pol):
    """X-1 IS CLOSED.  This pins the FIX, not the defect.

    Until 2026-09-11 this test asserted the opposite -- that the four cells
    still returned a wrong answer and that the census still flagged them -- and
    its own docstring asked for this restatement ("a future fix should make
    this test fail").  What closed it is the modal branch-cut fix; see the
    block comment above for the whole ladder on both builds and both arms.

    Three claims, each a decision rather than a reading:

      (a) the cell RETURNS -- nothing is refused, which is the regression guard
          the old form also carried;
      (b) it CLOSES energy to ``_X1_CLOSED_CLOSURE``, on a lossless cell where
          conservation is exact at any truncation under the Laurent rule.  The
          old form could not make this claim on any cell: 19 TE read
          ``R+T = 1.018`` on one build and RAISED on the other;
      (c) the census FLAGS NOTHING there -- the near-cancelling denominator the
          instrument exists to see is gone, not merely below a screen.

    The fail-before is the sibling test below, which reinstates the pre-fix
    branch body and re-reads the same ladder.
    """
    rows = _thin_scan(pol)
    hit = next(r for r in rows if r["M"] == M)
    assert hit["raised"] is None, (
        f"{M} {pol.upper()} raised {hit['raised']} on a cell that reads "
        f"|R+T-1| <= 1.5e-15 on both builds since the branch-cut fix.\n    "
        + _thin_table())
    assert not hit["refused"]
    assert hit["close"] < _X1_CLOSED_CLOSURE, (
        f"{M} {pol.upper()} closes at {hit['close']:.3e}: X-1 has REOPENED, "
        f"which on this cell means a propagating layer mode is carrying the "
        f"incoming root again (the groove index equals BOTH half-spaces').\n"
        f"    " + _thin_table())
    assert not hit["flagged"], (
        f"{M} {pol.upper()}: the census flags "
        f"{len(hit['flagged'])} near-singular inverse(s) (min equilibrated "
        f"rcond {min(c[2] for c in hit['flagged']):.3e}) on a cell that is "
        f"now well conditioned.\n    " + _thin_table())


def test_x1_is_closed_across_the_whole_thin_ladder_and_reopens_pre_fix():
    """The two-sided form of the claim above, on the LADDER rather than four
    cells -- because which truncation manifests was always a per-build fact
    (see the partition comment at the top of this file), and the thing that is
    now build-independent is that NONE of them does.

    POST: 0 raising cells, 0 flagged cells, worst closure <= 1.5e-15 over
    25 truncations x 2 polarizations on both builds, and every rung's
    ``sum(R)`` within 5.1e-02 of the converged 2.05e-04 (that worst case is
    the COARSEST rung, M = 6).

    PRE (engineered, this process): 7-8 raising cells and 14 flagged of 25 in
    TE, worst closure 3.196e-02 and a ``sum(R)`` 152.60x the converged value at
    M = 21 -- on both builds, at every thread count measured.

    RESTATED 2026-09-11 (branch-cut ROUND 3).  That last figure was the
    fail-before's closing assertion (``worst_ratio > 10.0``) and it is a
    per-kernel fact by its nature: it is amplified rounding, so how far the
    singular mode-match throws ``sum(R)`` depends on where the singularity
    falls relative to the arithmetic.  On CI's AMD runners it read 0.68x, and
    the test failed for reproducing nothing rather than for anything being
    wrong.  The engineered arm now closes on the two BUILD-INDEPENDENT
    quantities defined at :data:`_X1_PRE_RCOND` -- a sign census on the root
    the selector returns, and the worst ``rcond(a+b)`` over the ladder -- which
    are decisions, and which separate by seventeen decades rather than by a
    ratio that has to be lucky.  The counting claims above (``n_bad``,
    ``n_flag``) are unchanged; they held on every CI shard.
    """
    for pol in ("te", "tm"):
        rows = _thin_scan(pol)
        assert all(r["raised"] is None for r in rows), (
            f"{sum(1 for r in rows if r['raised'])} of {len(rows)} "
            f"{pol.upper()} truncations raise.\n    " + _thin_table((pol,)))
        worst = max(r["close"] for r in rows)
        assert worst < _X1_CLOSED_CLOSURE, (
            f"worst {pol.upper()} closure {worst:.3e} over the ladder.\n    "
            + _thin_table((pol,)))
        assert not any(r["flagged"] for r in rows), (
            f"{sum(1 for r in rows if r['flagged'])} {pol.upper()} cells are "
            f"flagged.\n    " + _thin_table((pol,)))

    # ---- the fail-before, ENGINEERED: put the pre-round-1 branch back.
    with _root_sign_census() as post_census:
        post = _thin_scan_uncached("te")
    with _pre_branch_cut():
        with _root_sign_census() as pre_census:
            pre = _thin_scan_uncached("te")
    n_bad = sum(1 for r in pre
                if r["raised"] is not None or r["close"] > 1e-6)
    n_flag = sum(1 for r in pre if r["flagged"])
    assert n_bad >= 5 and n_flag >= 5, (
        f"the pre-round-1 branch body does not reopen X-1 on this build "
        f"({n_bad} bad cells, {n_flag} flagged of {len(pre)}): the "
        f"fail-before has stopped demonstrating anything, so the POST claims "
        f"above are no longer two-sided")
    # ... and the CAUSE really is present there, on quantities no kernel moves.
    assert post_census.oncut > 0, (
        "no on-cut modes on this ladder -- wrong fixture")
    assert post_census.incoming == 0, (
        f"the SHIPPED selector returned {post_census.incoming} of "
        f"{post_census.oncut} on-cut modes on the INCOMING root: the branch "
        f"cut has reopened in the tree, not just in the engineered arm")
    assert pre_census.incoming >= 1, (
        f"the pre-round-1 branch body mis-rooted NOTHING on this build "
        f"({pre_census.incoming} of {pre_census.oncut} on-cut modes on the "
        f"incoming root), so the engineered arm is not reproducing the cause")
    pre_rcond = _worst_rcond(pre)
    post_rcond = _worst_rcond(post)
    assert pre_rcond <= _X1_PRE_RCOND, (
        f"the pre-round-1 arm's worst rcond(a+b) over the ladder is "
        f"{pre_rcond:.3e}, above the {_X1_PRE_RCOND:.0e} bar: the mis-rooted "
        f"modes are not collapsing the mode-match, so the engineered arm is "
        f"not reproducing the defect")
    assert post_rcond > 1e-4, (
        f"the SHIPPED tree's worst rcond(a+b) over the ladder is "
        f"{post_rcond:.3e}, so the two arms are not separated")


def test_thin_grating_clean_truncations_are_untouched():
    """The other side of the same claim: the guard must not fire on any
    truncation the census scored clean, and the numbers must not move.  This is
    the false-positive control -- a guard that refused these would be worse
    than the defect.

    **v5.33.1 -- THE CLEAN SET IS A PER-BUILD FACT TOO.**  The seven
    truncations were listed: ``8, 9, 10, 11, 14, 15, 22`` TM, each asserted to
    close under 1e-6.  On the ubuntu CI images one of them closes at
    ``5.051e-05`` with ``sum(R)`` = 2.502e-04 against the clean population's
    2.011e-04 -- i.e. that cell has migrated OUT of the clean set and into the
    X-1 defect set, which is the very migration the sibling test above
    documents in the other direction.  The list was the bug.

    So the clean set is now the truncations that ARE clean here (closure under
    ``_THIN_CLEAN_CLOSURE``, a bar sitting in a six-decade gap), and the
    claims are the two that do not depend on which cells those are: the guard
    refused none of them, and their ``sum(R)`` all agree -- with each other,
    and with the other polarization's clean population solved in the same
    process on the same pool."""
    clean = _thin_clean("tm")
    assert len(clean) >= 5, (
        f"only {len(clean)} of {len(_THIN_LADDER)} TM truncations are clean "
        f"on this build: the false-positive control has nothing left to "
        f"control.\n    " + _thin_table(("tm",)))
    # the historical set, for the record -- and a non-vacuity check that the
    # device is still the one the audit measured
    hist = sorted(r["M"] for r in clean if r["M"] in (8, 9, 10, 11, 14, 15, 22))
    assert hist, ("none of the historically clean truncations 8/9/10/11/14/15/"
                  "22 TM is clean on this build.\n    " + _thin_table(("tm",)))

    # (1) the guard fired on NONE of them.
    for r in clean:
        assert not r["refused"], (
            f"{r['M']} TM closes at {r['close']:.3e} -- machine precision -- "
            f"and the guard refused it anyway: that is the false positive the "
            f"campaign's R-1b precedent rates worse than the defect")
    # (2) the numbers did not move: R is the deep null here (~2e-4), the
    #     sensitive observable, and every clean truncation must agree on it.
    med = _thin_clean_sumR("tm")
    for r in clean:
        assert abs(r["sumR"] / med - 1.0) < 0.10, (
            f"{r['M']} TM returns sum(R)={r['sumR']:.6e} against the clean "
            f"population's median {med:.6e}: a truncation that closes to "
            f"{r['close']:.1e} is not allowed to move the answer")
    # (3) ... and the OTHER polarization's clean population agrees, which is
    #     the cross-check no single ladder can give itself.
    assert _thin_clean("te"), (
        "no TE truncation is clean on this build, so the cross-polarization "
        "check below has nothing to compare against.\n    "
        + _thin_table(("te",)))
    med_te = _thin_clean_sumR("te")
    assert abs(med / med_te - 1.0) < 0.10, (
        f"the clean TM population reads sum(R)={med:.6e} and the clean TE "
        f"population {med_te:.6e}: at normal incidence on a 1.55/1.50 "
        f"grating these are the same deep null and must agree")


def test_the_withdrawn_refusal_moves_no_bit_and_the_ladder_carries_no_silent_defect(
        guard_off):
    """The refusal switch, and the population it was calibrated on.

    **WHAT THIS TEST USED TO SAY.**  Its name was
    ``test_the_refusal_reproduces_the_prior_answer_with_the_switch``, and it
    asserted that with ``INTERFACE_CONDITIONING_GUARD`` off the ladder RETURNS
    a silently-wrong answer -- the fail-before for the withdrawn refusal.  Its
    long v5.33.0 / v5.33.1 docstring recorded the two ways that claim had to be
    weakened: the absolute bars were BLAS-thread-dependent (``sum(R)`` reading
    3.216567e-02 at one thread and 6.112765e-03 at two on BOTH builds), and
    then the two pinned CELLS were per-build too, so both ends of the ratio had
    to be chosen from a per-build scan.

    **WHY IT IS RESTATED (2026-09-11).**  Neither weakening was the whole
    story: the DEFECT itself was the RCWA modal branch cut.  ``THIN``'s groove
    index equals both half-spaces' (1.5), so a propagating layer mode handed
    the INCOMING root -- which the pre-round-1 exact ``Re(r) == 0`` pin could
    not prevent, because an ``eig`` output never satisfies it -- was exactly a
    half-space BACKWARD mode, and the interface mode-match ``a + b`` whose
    explicit inverse IS ``S12`` was then singular.  Round 1 pinned the root;
    the ladder now closes to <= 1.5e-15 at all 25 truncations x 2
    polarizations on both builds and the census flags nothing, so there is no
    silently-wrong cell left for this test to find.  It SKIPPED rather than
    failed, because its widening path was written for a per-build absence.

    **WHAT IT SAYS NOW.**  Two claims that survive the closure, plus an
    ENGINEERED fail-before that no longer depends on which truncation a given
    BLAS reduction order breaks:

      (a) the ladder carries NO silently-wrong truncation -- every cell that
          returns closes better than ``_THIN_DEFECT_CLOSURE``, with the guard
          OFF (the fixture), which is where a wrong answer would come back;
      (b) THE SWITCH ITSELF is a no-op: the inverse refusal was WITHDRAWN (see
          the note above ``_INV_RCOND_SCREEN``), so flipping
          ``INTERFACE_CONDITIONING_GUARD`` must not move a bit on the cell that
          historically carried the defect.  That was always the assertion this
          test's name was about, and it is unchanged;
      (c) with the pre-round-1 branch body reinstated the wrong answer comes
          BACK -- and comes back through the switch in both positions, which is
          the withdrawn refusal's fail-before, reproduced on demand.

    **RESTATED 2026-09-11 (CI PREMISE GATES).**  Claim (c) below is now
    PREMISE-GATED.  The CI runner arm (ubuntu, AMD EPYC 7763, unpinned BLAS,
    pip wheels of numpy 2.4.6 / scipy 1.17.1) produces CORRECT answers on the
    ill-conditioned fixtures of this campaign where every local arm -- four
    OpenBLAS kernels, one and four threads, two builds -- produces wrong ones.
    On the 5.45.0 matrix the engineered pre-round-1 arm reproduced only
    **0.267x** (py3.10 shard 3, py3.11 shard 1) and **0.677x** (py3.10 shard 3
    and the JAX lane) of the converged 2.094088e-04, against the 152x the fix
    document records: there was no gross error on that arm to withdraw a
    refusal from.  A reproduction that is a numerical reading of a pathology
    is therefore MEASURED on the running arm and SKIPPED when it does not
    hold, with the reading in the skip reason; it is never asserted, never
    relaxed into a smaller bar, and the arm is never deleted.  Claims (a) and
    (b) -- the SHIPPED ladder carries no silent defect, and the withdrawn
    refusal moves no bit -- stay unconditional on every arm.  WHY the CI arm
    differs is an OPEN item, recorded in
    ``docs/audits/CI_PREMISE_GATES_2026_09_11.md``.

    **THE M1 EQUILIBRATION INSTRUMENT: KEPT (decision, 2026-09-11).**
    ``_equilibrated_inverse_residual`` / ``_rcond_1_equilibrated`` were chosen
    over the raw instruments because a measured population of calls existed
    where the raw residual would have REFUSED a correct answer and the
    equilibrated one passed it.  That population was the branch-cut defect:
    re-measured over a 24-fixture sweep it goes from 9 of 106 guarded calls
    pre-fix (7 unpinned; 9 / 8 / 6 on WSL, i.e. moving with the BLAS pool --
    the defect's own signature one level up) to **0 of 110 post-fix, on both
    builds at every thread count**.  The instrument is KEPT anyway: it is
    behaviour-preserving, it costs nothing on the screened path, the ARMED
    ``T22`` refusal is a separate population this change does not touch (its
    minimum equilibrated ``rcond`` reads 2.792e-02 on both arms, eight decades
    above its own ``1e-10`` bar), and a user's own coincident geometry can
    still reach it.  Deleting a guard because its found population is empty
    would repeat, one level up, the mistake this campaign exists to prevent.
    No library code was removed in round 2.
    """
    # (a) with the guard OFF, nothing in the ladder returns a wrong answer.
    for pol in ("te", "tm"):
        defects = _thin_defects(pol)
        assert not defects, (
            f"{len(defects)} {pol.upper()} truncation(s) return while missing "
            f"closure by more than {_THIN_DEFECT_CLOSURE:.0e}: X-1 has "
            f"reopened with the guard off.\n    " + _thin_table((pol,)))

    # (b) THE SWITCH: bit-identical in both positions on the historical cell.
    def solve(M, pol, flag):
        prev = _rc.INTERFACE_CONDITIONING_GUARD
        _rc.INTERFACE_CONDITIONING_GUARD = flag
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _o, R, T = rcwa_efficiency_1d(
                    THIN["period"], THIN["n_ridge"], THIN["n_groove"],
                    THIN["n_substrate"], THIN["n_superstrate"], THIN["depth"],
                    THIN["duty_cycle"], WL, angle=0.0, polarization=pol,
                    n_orders=M, stabilize=False)
            return np.asarray(R).copy(), np.asarray(T).copy()
        finally:
            _rc.INTERFACE_CONDITIONING_GUARD = prev

    off, on = solve(21, "te", False), solve(21, "te", True)
    assert float(np.max(np.abs(off[0] - on[0]))) == 0.0
    assert float(np.max(np.abs(off[1] - on[1]))) == 0.0

    # (c) the fail-before, ENGINEERED and PREMISE-GATED.  See the RESTATED
    #     2026-09-11 paragraph in this test's docstring: the pre-round-1
    #     branch body's ERROR MAGNITUDE is a property of the running arm's
    #     BLAS, and neither the 5.45.0 CI matrix nor the JAX lane reproduced
    #     the documented 152x on it (0.267x on py3.10/3.11, 0.677x on the JAX
    #     job and one py3.10 shard, printed below on whatever arm runs).  The
    #     reproduction is therefore MEASURED and SKIPPED when absent; it is
    #     never asserted, no bar is relaxed, and the arm is not deleted.
    ref = _thin_converged_sumR("te")
    with _pre_branch_cut():
        rows = [r for r in _thin_scan_uncached("te")
                if r["raised"] is None and np.isfinite(r["sumR"])]
        bad = [r for r in rows if r["close"] > _THIN_DEFECT_CLOSURE]
        if not bad:
            pytest.skip(
                "premise absent on this arm: the pre-round-1 branch body "
                "returns no silently-wrong truncation (worst closure %.3e "
                "over %d returning cells against the %.0e bar), so there is "
                "no wrong answer here for the withdrawn refusal to have "
                "acted on.  The two UNCONDITIONAL claims of this test -- the "
                "shipped ladder carries no silent defect, and the switch is "
                "a no-op on the historical cell -- passed above."
                % (max((r["close"] for r in rows), default=float("nan")),
                   len(rows), _THIN_DEFECT_CLOSURE))
        worst = max(bad, key=lambda r: abs(r["sumR"] / ref - 1.0))
        pre_off = solve(worst["M"], worst["pol"], False)
        pre_on = solve(worst["M"], worst["pol"], True)
    score = abs(worst["sumR"] / ref - 1.0)
    print(f"\nX-1 engineered fail-before: {worst['M']} "
          f"{worst['pol'].upper()} sum(R)={worst['sumR']:.6e} closure="
          f"{worst['close']:.3e} against the converged {ref:.6e} "
          f"-- {score:.1f}x wrong")
    # INVARIANT on any arm that reaches here: whatever the magnitude, the
    # WITHDRAWN refusal did not act on the row -- the switch is a no-op in
    # both positions.  That is what this test's name is about.
    assert float(np.max(np.abs(pre_off[0] - pre_on[0]))) == 0.0
    assert float(np.max(np.abs(pre_off[1] - pre_on[1]))) == 0.0
    # PREMISE-GATED: the GROSS error the fix document records (152x).  A
    # decade is the gate, because the point of the arm is a defect nobody
    # could mistake for truncation; below it the arm demonstrates nothing.
    if score < 10.0:
        pytest.skip(
            "premise absent on this arm: the engineered pre-round-1 arm's "
            "worst cell is only %.3fx from the converged %.6e (M = %d %s, "
            "sum(R) = %.6e, closure %.3e) -- it does not reproduce the "
            "documented 152x error, so there is no gross defect here to "
            "withdraw a refusal from.  The switch was still measured to be a "
            "no-op on that cell, bit for bit."
            % (score, ref, worst["M"], worst["pol"].upper(), worst["sumR"],
               worst["close"]))
    # ... and on an arm that DOES reproduce it, the SHIPPED branch body closes
    # the very same cell -- the fail-before and the fix-after on one row.
    shipped = [r for r in _thin_scan_uncached(worst["pol"])
               if r["M"] == worst["M"]]
    assert len(shipped) == 1, (worst["M"], worst["pol"], len(shipped))
    assert shipped[0]["raised"] is None, (worst, shipped[0])
    assert shipped[0]["close"] <= _THIN_DEFECT_CLOSURE, (
        "the pre-round-1 body is %.1fx wrong at M = %d %s and the SHIPPED "
        "body does not close the same cell (%.3e against %.0e)"
        % (score, worst["M"], worst["pol"].upper(), shipped[0]["close"],
           _THIN_DEFECT_CLOSURE))


# ---------------------------------------------------------------------------
# N-2: the Rayleigh projection
# ---------------------------------------------------------------------------

def test_guarded_lstsq_returns_historical_bits_on_a_healthy_projection():
    rng = np.random.default_rng(19)
    A = rng.standard_normal((40, 12)) + 1j * rng.standard_normal((40, 12))
    b = A @ (rng.standard_normal(12) + 1j * rng.standard_normal(12))
    got = _pc._guarded_lstsq(A, b, "probe")
    ref, *_ = np.linalg.lstsq(A, b, rcond=None)
    assert float(np.max(np.abs(got - ref))) == 0.0


def test_guarded_lstsq_refuses_only_on_rank_deficiency_AND_residual():
    """The surviving refusal, and the truth table that earned it.

    A minimum-norm draw REQUIRES a null space, so rank deficiency is necessary;
    a rank-deficient system whose right-hand side still lies in the range is
    fine, so a residual is necessary too.  Neither alone separates the four
    measured families:

      family                            rank        relres    verdict
      2-D staggered far field           200 of 200  2.1e-07   PASS (full rank)
      shared grid, ffo 41               73 of 82    2.1e-14   PASS (in range)
      shared grid, ffo 61 (10% wrong)   78 of 122   6.5e-07   REFUSE
      shared grid, ffo 77 (J00 = 113)   82 of 154   2.0e-03   REFUSE
    """
    rng = np.random.default_rng(23)
    # (a) FULL RANK, b out of range, big residual -> must PASS
    A = rng.standard_normal((30, 6)) + 1j * rng.standard_normal((30, 6))
    b = rng.standard_normal(30) + 1j * rng.standard_normal(30)
    got = _pc._guarded_lstsq(A, b, "probe")
    ref, *_ = np.linalg.lstsq(A, b, rcond=None)
    assert float(np.max(np.abs(got - ref))) == 0.0
    # (b) RANK DEFICIENT but b IN the range -> must PASS
    Ad = A.copy()
    Ad[:, 5] = Ad[:, 4]                       # exact column duplicate
    b_in = Ad @ (rng.standard_normal(6) + 1j * rng.standard_normal(6))
    got = _pc._guarded_lstsq(Ad, b_in, "probe")
    ref, *_ = np.linalg.lstsq(Ad, b_in, rcond=None)
    assert float(np.max(np.abs(got - ref))) == 0.0
    # (c) RANK DEFICIENT and b OUT of range -> must REFUSE
    with pytest.raises(_pc._ConditioningError, match="rank-deficient"):
        _pc._guarded_lstsq(Ad, b, "probe")
    # ... and the switch gives the draw back, bit for bit
    prev = _rc.INTERFACE_CONDITIONING_GUARD
    _rc.INTERFACE_CONDITIONING_GUARD = False
    try:
        got = _pc._guarded_lstsq(Ad, b, "probe")
        ref, *_ = np.linalg.lstsq(Ad, b, rcond=None)
        assert float(np.max(np.abs(got - ref))) == 0.0
    finally:
        _rc.INTERFACE_CONDITIONING_GUARD = prev


def test_guarded_lstsq_stands_aside_on_a_non_finite_system():
    """A non-finite system must keep its PRE-M1 outcome, whichever it is.

    Two distinct routes, and only the second reaches the guard at all:
    a non-finite ``A`` makes ``gelsd`` itself raise ``LinAlgError`` before the
    guard gets control (so M1 changes nothing), while a non-finite ``b`` comes
    back as a NaN solution -- which the guard must pass through rather than
    re-badge as a conditioning failure."""
    rng = np.random.default_rng(41)
    A = rng.standard_normal((12, 4)) + 1j * rng.standard_normal((12, 4))
    b = rng.standard_normal(12) + 0j
    # (a) non-finite A: numpy's own error, unchanged by M1
    A_bad = A.copy()
    A_bad[3, 1] = np.nan
    # (one call, not two: LAPACK writes a DLASCL complaint to stderr for each
    # non-finite gelsd, and one line of CI noise is enough to make the point)
    with pytest.raises(np.linalg.LinAlgError):
        _pc._guarded_lstsq(A_bad, b, "probe")
    # (b) non-finite b: reaches the guard, must pass through untouched
    b_bad = b.copy()
    b_bad[5] = np.nan
    got = _pc._guarded_lstsq(A, b_bad, "probe")    # must NOT raise
    ref, *_ = np.linalg.lstsq(A, b_bad, rcond=None)
    assert got.shape == (4,)
    assert np.array_equal(np.isnan(got), np.isnan(ref))


def test_rcond_of_hsup_would_have_been_the_wrong_instrument():
    """The measurement that chose the residual over ``rcond(Hsup)``, kept as a
    test because a future reader will reach for the condition number first.

    On the audit staircase the SHARED path at ``far_field_orders`` = 41 has a
    numerically rank-deficient ``Hsup`` (``rcond`` ~ 8e-17) and returns the
    right answer to nine digits; the broken one at 61 reads ~7e-17.  A screen
    on ``rcond`` cannot separate them.  The residual can.

    Scored as a COMPARATIVE ENVELOPE: the rank-deficient-but-sound solve's
    deviation from the low-order reference must be orders below the deviation
    the refused one carries -- no absolute bar is asserted, because the
    numerator here is BLAS round-off.

    **v5.33.1 -- THE CONSERVATION HALF WAS MEASURED ON A DEVICE THAT DOES NOT
    CONSERVE ON EVERY BUILD.**  The second assertion compared ``|R+T-1|`` at
    41 orders against 61.  But this is the SHARED (union) grid at
    ``phi`` = 0.6 -- the oblique union-grid pathology the per-layer campaign
    exists for -- and on this staircase the union grid's own energy closure is
    a build fact, identical at EVERY truncation::

        OPENBLAS_NUM_THREADS   closure at ffo 7/11/21/31/41   ffo 61 (guard off)
        1                      6.654e-06                      1.126e+00
        2                      2.135e+01                      1.796e+01

    At two threads the CONVERGED reference itself misses closure by 21, so the
    observable carries no information about the truncation and the comparison
    reads ``21.3 < 1e-4 x 17.2``.  (The J00 half is unaffected: 3.05e-09
    against 1.96e-01, ratio 1.6e-08, at both thread counts.)  So the
    conservation claim is now anchored on the DEVICE's own baseline -- the
    rank-deficient truncation must add nothing to it -- and the "61 breaks
    conservation" half is asserted where the baseline can carry it, with the
    screen printed where it cannot."""
    o41, R41, T41, J41 = _solve_conical(41, "shared")
    o7, R7, T7, J7 = _solve_conical(7, "shared")
    good = complex(np.asarray(J7)[0, 0])
    d41 = abs(complex(np.asarray(J41)[0, 0]) - good)
    # ... while 61 is refused
    with pytest.raises(_pc._ConditioningError):
        _solve_conical(61, "shared")
    prev = _rc.INTERFACE_CONDITIONING_GUARD
    _rc.INTERFACE_CONDITIONING_GUARD = False
    try:
        o61, R61, T61, J61 = _solve_conical(61, "shared")
    finally:
        _rc.INTERFACE_CONDITIONING_GUARD = prev
    d61 = abs(complex(np.asarray(J61)[0, 0]) - good)
    assert d41 < 1e-4 * d61                  # measured ~3e-7 of it
    # and the refused one is the one that breaks conservation, not the
    # rank-deficient one.  The device's own baseline is the yardstick.
    base, c41, c61 = _closure(R7, T7), _closure(R41, T41), _closure(R61, T61)
    assert c41 < 2.0 * max(base, 1e-15), (
        f"the rank-deficient truncation closes at {c41:.3e} against the "
        f"converged reference's {base:.3e}: it is ADDING conservation error, "
        f"which is what the refused one is supposed to do")
    if base < 1e-3:
        assert c61 > 1.0e3 * max(base, 1e-15), (
            f"the refused truncation closes at {c61:.3e} against the "
            f"device's {base:.3e} baseline: it has stopped being the one "
            f"that breaks conservation")
    else:
        print(f"\nHsup: the conservation half is SCREENED OUT on this build "
              f"-- the union grid's converged reference already misses "
              f"closure by {base:.3e} at every truncation (41 orders: "
              f"{c41:.3e}, 61 orders: {c61:.3e}), so |R+T-1| separates "
              f"nothing here.  The J00 half above is unaffected "
              f"(d41={d41:.3e}, d61={d61:.3e}).")


# ---------------------------------------------------------------------------
# T3-3: the conical per-layer far-field order cap
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def uncapped_drift():
    """THIS BUILD's pre-M1 over-capacity drift: ``|J00(61) - J00(7)|`` on the
    per-layer conical stack with the cap (and the interface guard) switched
    OFF, i.e. the exact quantity T3-3 removed.

    2026-08-15, ``docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md``.  This used to
    be the hard-coded literal ``1.6e-06`` below -- a pre-fix reading from one
    build, which made ``1e-3 * 1.6e-6`` an ABSOLUTE 1.6e-09 bar wearing a
    ratio's clothing (exactly what this file's own docstring forbids), and
    which had since drifted 21x: the same quantity now measures 3.403084e-05
    on Windows py3.14/np2.4.4 and 3.403086e-05 on WSL py3.12/np2.5.1.  Note
    what that re-measurement shows: the DEFECT is build-stable to 6 digits
    (it is a truncation error of the wrong operator, not round-off), so it is
    a sound denominator -- it is only the STALE LITERAL that was per-build.
    Computed once per module (two extra solves) so both halves of the T3-3
    claim are scored against the same in-process reference."""
    prev_cap = _con.PMM_CONICAL_PERLAYER_ORDER_CAP
    prev_guard = _rc.INTERFACE_CONDITIONING_GUARD
    _con.PMM_CONICAL_PERLAYER_ORDER_CAP = False
    _rc.INTERFACE_CONDITIONING_GUARD = False
    try:
        _o7, _R7, _T7, J7 = _solve_conical(7, "per-layer")
        o61, _R61, _T61, J61 = _solve_conical(61, "per-layer")
        assert int(np.asarray(o61).shape[0]) == 61, (
            "premise moved: the pre-M1 arithmetic no longer accepts 61 orders "
            "over a capacity of 29, so there is nothing to score against")
        return abs(complex(np.asarray(J61)[0, 0])
                   - complex(np.asarray(J7)[0, 0]))
    finally:
        _con.PMM_CONICAL_PERLAYER_ORDER_CAP = prev_cap
        _rc.INTERFACE_CONDITIONING_GUARD = prev_guard


def test_t3_3_per_layer_cap_clamps_to_the_window_half_spaces(uncapped_drift):
    """The fix: the cap comes from the grids the HALF-SPACES live on, so the
    per-layer answer is STATIONARY in ``far_field_orders`` past the capacity
    instead of degrading.  The union grid has 13 cells (``n_glob`` = 78, cap 77
    orders); the END WINDOW grids have 5 (``n_glob`` = 30, cap 29).

    The spread is scored COMPARATIVELY against what the pre-M1 cap produces on
    the same stack IN THIS PROCESS (the ``uncapped_drift`` fixture): the
    un-capped 61-order solve drifts in ``J00`` while conserving energy, so a
    stationarity spread three orders under that is the claim."""
    ref = None
    spread = 0.0
    for ffo in (7, 21, 31, 41, 61, 77):
        o, R, T, J = _solve_conical(ffo, "per-layer")
        assert int(np.asarray(o).shape[0]) <= 29
        if ref is None:
            ref = complex(np.asarray(J)[0, 0])
        else:
            spread = max(spread, abs(complex(np.asarray(J)[0, 0]) - ref))
    # 2026-08-15 (docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md): denominator is
    # now DERIVED IN-BUILD, so the bar tracks the defect instead of a 2026-08-04
    # literal.  Measured ``spread`` (cap ON, max over ffo of |J00 - J00(7)|):
    # 2.843243e-11 Windows py3.14/np2.4.4, 5.249477e-11 WSL py3.12/np2.5.1 --
    # a 1.85x cross-build spread, because the CAPPED answer is stationary to
    # round-off and round-off is what differs between LAPACK builds.  Against
    # the in-build denominator (3.403e-05 on both) those read 8.355e-07 and
    # 1.543e-06, so the 1e-3 bar clears the worst build by 648x.  In the other
    # direction it refuses any regression that lets 0.1% of the over-capacity
    # draw back in -- 3.4e-08, still ~650x above the stationarity floor -- so
    # neither side is close.
    assert spread < 1e-3 * uncapped_drift, (
        f"the capped per-layer answer is not stationary past the capacity: "
        f"spread {spread:.3e} against the pre-M1 over-capacity drift "
        f"{uncapped_drift:.3e} measured on this build "
        f"(ratio {spread / uncapped_drift:.3e}, bar 1e-3)")


def test_t3_3_fail_before_reproduces_the_over_capacity_draw(cap_off,
                                                            guard_off):
    """FAIL-BEFORE, and the proof that this was never 'latent only'.

    With the cap computed from the FULL UNION (the pre-M1 arithmetic) the
    per-layer conical solve accepts 61 and 77 far-field orders on grids that
    carry 59, and the result is wrong in two DIFFERENT ways -- which is the
    whole point of scoring conservation next to accuracy:

    * at 77 orders energy closure blows to ``|R+T-1|`` ~ 15, i.e. loudly;
    * at 61 orders closure stays at the mortar's own 1.0e-04 -- CLEAN -- while
      the zero-order Jones has moved in its sixth digit.  Energy conservation
      is blind to it, because a null-space component of ``cinc`` is invisible
      to ``Hsup``.

    Both are compared against the converged, capacity-respecting answer."""
    o7, R7, T7, J7 = _solve_conical(7, "per-layer")
    good = complex(np.asarray(J7)[0, 0])

    o61, R61, T61, J61 = _solve_conical(61, "per-layer")
    clos61 = float(np.max(np.abs(np.asarray(R61).sum(axis=1)
                                 + np.asarray(T61).sum(axis=1) - 1.0)))
    o77, R77, T77, J77 = _solve_conical(77, "per-layer")
    clos77 = float(np.max(np.abs(np.asarray(R77).sum(axis=1)
                                 + np.asarray(T77).sum(axis=1) - 1.0)))
    assert int(np.asarray(o61).shape[0]) == 61     # over capacity, accepted
    assert int(np.asarray(o77).shape[0]) == 77
    # the LOUD failure
    assert clos77 > 1.0
    # the SILENT one: closure no worse than the honest solve, answer moved
    assert clos61 < 10.0 * float(np.max(np.abs(
        np.asarray(R7).sum(axis=1) + np.asarray(T7).sum(axis=1) - 1.0)))
    # 2026-08-15 (docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md): "moved" needs a
    # scale, and the only build-free one is the HONEST path's own
    # truncation-to-truncation variation.  Measure it here, in this process,
    # from BELOW-capacity solves -- where the cap is provably a no-op
    # (test_t3_3_switch_is_a_no_op_below_the_capacity pins bit-identity at
    # ffo 7 / 21 / 29), so this floor is the same with the switch either way.
    # The old absolute ``> 1e-6`` passed with 34x today, but its rationale was
    # a 2026-08-04 build reading of the numerator that has since moved 21x;
    # a constant whose justification has already drifted is a per-build fact.
    # MEASURED floor = max(|J00(21) - J00(7)|, |J00(29) - J00(7)|):
    # 2.843243e-11 Windows py3.14/np2.4.4, 5.249477e-11 WSL py3.12/np2.5.1
    # (round-off-limited, hence the 1.85x cross-build spread).  MEASURED
    # numerator |J00(61) - J00(7)| = 3.403084e-05 [W] / 3.403086e-05 [M] --
    # build-stable to 6 digits, because it is the wrong OPERATOR, not noise.
    # Ratios 1.197e+06 [W] / 6.483e+05 [M]; the 1e3 bar therefore holds with
    # 648x on the worst build, while in the other direction it cannot be
    # tripped by round-off (that would need the honest floor to grow 1000x).
    floor = max(abs(complex(np.asarray(J)[0, 0]) - good)
                for _o, _R, _T, J in (_solve_conical(21, "per-layer"),
                                      _solve_conical(29, "per-layer")))
    d61 = abs(complex(np.asarray(J61)[0, 0]) - good)
    assert d61 > 1.0e3 * floor, (
        f"the over-capacity 61-order draw moved J00 by {d61:.3e}, which is "
        f"not 1000x the honest path's own truncation floor {floor:.3e} on "
        f"this build: the SILENT half of the defect is no longer reproduced")
    # and it is the SILENT one -- strictly smaller than the loud 77-order draw
    assert d61 < abs(complex(np.asarray(J77)[0, 0]) - good)


def test_t3_3_switch_is_a_no_op_below_the_capacity(cap_off):
    """The fail-before switch must be BIT-IDENTICAL wherever the cap does not
    bind -- the clamp is a fix at high ``far_field_orders``, not a rewrite."""
    for ffo in (7, 21, 29):
        o0, R0, T0, J0 = _solve_conical(ffo, "per-layer")
        _con.PMM_CONICAL_PERLAYER_ORDER_CAP = True
        try:
            o1, R1, T1, J1 = _solve_conical(ffo, "per-layer")
        finally:                       # the cap_off fixture owns the restore
            _con.PMM_CONICAL_PERLAYER_ORDER_CAP = False
        assert float(np.max(np.abs(np.asarray(J0) - np.asarray(J1)))) == 0.0
        assert float(np.max(np.abs(np.asarray(R0) - np.asarray(R1)))) == 0.0


def test_t3_3_matches_the_sibling_paths_it_was_the_outlier_from():
    """The three siblings that already clamped correctly are the oracle: the
    conical per-layer cap must land on the same capacity as the classical
    per-layer path's ``min(n0, nN)`` from the window half-spaces."""
    st = _stair_stack(77, "per-layer")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o_cls, _R, _T, _J = st.set_source(WL, theta=0.15).solve()
        o_con, _R2, _T2, _J2 = st.set_source(WL, theta=0.15, phi=0.6).solve()
    assert int(np.asarray(o_cls).shape[0]) == int(np.asarray(o_con).shape[0])


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------

def test_the_guard_is_numpy_only_and_the_traced_path_is_unchanged():
    """The JAX path is traced -- a data-dependent branch is not expressible --
    and CuPy would pay a device sync per interface, so both keep the historical
    arithmetic.  Pinned so a future 'just call it everywhere' cannot land
    silently."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    A = jnp.asarray(np.diag(np.logspace(0, -20, 8)).astype(complex))
    got = _rc._guarded_inverse(A, "probe")   # must NOT raise
    assert not isinstance(got, np.ndarray)
