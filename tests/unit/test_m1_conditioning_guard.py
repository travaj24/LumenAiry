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
    # (b) the separation that chose the instrument.  Both halves, as ratios.
    assert raw > 1.0e3 * _rc._INV_RESID_REFUSE, (
        f"the raw residual on this cascade reads {raw:.3e}: it no longer "
        f"exceeds the refusal bar, so this cell has stopped being the "
        f"false-positive the equilibration exists for")
    assert eq < 1.0e-3 * _rc._INV_RESID_REFUSE, (
        f"the EQUILIBRATED residual reads {eq:.3e} against the refusal bar "
        f"{_rc._INV_RESID_REFUSE:.0e} -- equilibration stopped rescuing the "
        f"correct answer and the guard is one round-off from refusing it")
    # the free screen does not even fire here, so the residual is never paid
    # for: measured 1.2e-05 .. 3.1e-04 against the 1e-08 screen.
    assert eq_rc > 1.0e1 * _rc._INV_RCOND_SCREEN
    assert raw / eq > 1.0e9                   # measured 1e12 .. 1e13

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
    # (f) the one thing it CAN carry: the cascade has not blown up.
    assert hi < 0.5, (
        f"a rung closes at {hi:.3e} on a lossless cascade: |R+T-2| that large "
        f"is a MIS-ASSEMBLED cascade, not round-off "
        f"({[f'{c:.3e}' for c in closes]})")


# ---------------------------------------------------------------------------
# X-1 on the library's own documented instability class
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,pol", [(19, "te"), (21, "te"), (12, "te"),
                                   (20, "te")])
def test_x1_defect_is_reproduced_and_flagged_but_NOT_closed(M, pol):
    """X-1 IS REAL AND IS STILL OPEN.  This pins the DEFECT, not a fix.

    Measured pre-M1 and unchanged by M1, both builds:

    * 19 TE returns ``R+T`` = 1.018 on Windows/MKL and RAISES on WSL/OpenBLAS
      -- a literal build-dependent answer on the DEFAULT path;
    * 21 TE returns ``sum(R)`` = 3.2e-2 on BOTH builds against a converged
      2.0e-4 -- 160x wrong, agreed to every digit, inside ``_EnergyWarning``'s
      documented silent window;
    * 12 TE and 20 TE disagree across builds by 8.9e-04 and 8.3e-02 relative.

    The M1 census DETECTS all four.  It does not ACT, because the same readings
    occur on healthy 2-D solves -- see
    ``test_the_inverse_refusal_was_withdrawn_and_the_instruments_record_instead``.
    Closing X-1 needs a criterion that survives every method in the library.

    Pinned here: the solve still RETURNS (nothing is refused) and the
    instrument still FLAGS it.  A future fix should make this test fail.

    **v5.33.1 -- THE FOUR CELLS ARE A PER-BUILD FACT.**  ``12 TE`` and
    ``20 TE`` are flagged on Windows (equilibrated rcond 1.3e-10 and 2.3e-11,
    at 1, 2 and 24 threads) and are NOT flagged on any of the five ubuntu CI
    images -- the census comes back empty and the test read "the census must
    still see the conditioning".  ``19 TE`` and ``21 TE`` are flagged in both
    places.  Nothing about X-1 changed; the near-cancellation simply lands on
    the other side of a screen that is 1e-8 wide.

    So the parametrization is now a STARTING CELL, not the claim.  If the
    pinned cell does not manifest on this build the scan SUBSTITUTES the
    nearest cell that does and says so; only if NOTHING in the ladder
    manifests -- and the census instrument is separately PROVED alive on a
    synthetic singular operator -- is the defect declared absent here, with
    the full probe table printed.  It is never silently skipped, and a build
    on which the guard REFUSES a manifesting cell still fails, which is the
    "a future fix should make this test fail" contract."""
    rows = _thin_scan(pol)
    hit = next((r for r in rows if r["M"] == M and r["flagged"]), None)
    if hit is None:
        # WIDEN, do not skip: the nearest manifesting truncation of the same
        # polarization, then the other one -- preferring cells that RETURN,
        # because X-1 is about answers that come back rather than about the
        # ones `_check_energy` already catches.
        def _rank(r):
            return (r["raised"] is not None, abs(r["M"] - M))
        cand = sorted(_thin_flagged(pol), key=_rank)
        cand += sorted([r for p in ("te", "tm") if p != pol
                        for r in _thin_flagged(p)], key=_rank)
        if cand:
            hit = cand[0]
            print(f"\nX-1: the pinned cell {M} {pol.upper()} does not manifest "
                  f"on this build (census flags nothing); substituting "
                  f"{hit['M']} {hit['pol'].upper()}, which does.\n    "
                  + _thin_table())
    if hit is None:
        alive = _census_instrument_is_alive()
        assert alive, (
            "no truncation in the scanned ladder flags the conditioning AND "
            "the census instrument does not flag a synthetic cond-1e20 "
            "operator either: the INSTRUMENT is broken, not the defect "
            "absent.\n    " + _thin_table())
        pytest.skip(
            "X-1 does not manifest anywhere in the scanned ladder on this "
            "build, and this is probe-backed rather than assumed: the census "
            "instrument still flags a synthetic cond-1e20 operator "
            "(_census_instrument_is_alive() is True), so the screen is live "
            "and the thin-grating family simply carries no near-cancelling "
            "star denominator here.  Scan:\n    " + _thin_table())
    # the solve still RETURNED or reached the named energy guard -- either way
    # nothing was refused ...
    assert not hit["refused"], (
        f"{hit['M']} {hit['pol'].upper()}: the guard REFUSED a flagged "
        f"truncation.  X-1 may have been closed -- re-pin this test against "
        f"the fix rather than restoring the old behaviour.")
    # ... and the instrument still SEES the conditioning it does not act on.
    assert hit["flagged"], "the census must still see the conditioning"
    assert min(c[2] for c in hit["flagged"]) < _rc._INV_RCOND_SCREEN


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


def test_the_refusal_reproduces_the_prior_answer_with_the_switch(guard_off):
    """Fail-before, verified PER CONFIGURATION on the geometry that changed:
    with the guard off, 21 TE RETURNS the pre-M1 answer -- the non-conserving
    one -- instead of raising.

    **v5.33.0 -- THE BARS WERE THREAD-COUNT-DEPENDENT.**  They were absolute:
    ``sum(R)+sum(T) > 1.03`` and ``sum(R) > 3.0e-2``.  Holding the code, the
    build and the geometry fixed and varying ONLY ``OPENBLAS_NUM_THREADS``
    [M, BOTH builds]::

        threads    sum(R)             sum(R)+sum(T) - 1
              1    3.216567261e-02    +3.195613251e-02   <- bars calibrated here
        2 / 4 / 24 6.112765047e-03    +5.903262542e-03   <- BOTH bars fail

    Windows/scipy-openblas 0.3.31 and WSL/OpenBLAS agree to TEN DIGITS at each
    thread count, so this is not a build difference: it is exactly the
    near-degenerate draw M1's census exists to record, moving with the BLAS
    reduction order.  The suite pins the pool nowhere, so the test passed run
    alone (pool 1) and failed inside a breadth sweep (pool > 1).  Same trap as
    ``_PAR_TOTAL`` in the v5.20.2 PMM-JAX suite and M3's ``worst_rcond``: an
    absolute bar on a BLAS-dependent magnitude, which no re-tune can fix.

    What IS build- and thread-independent is the COMPARISON against a
    truncation the census scores clean, solved in the same process on the same
    pool: the clean solves sit at ``sum(R)`` ~ 2.0e-4 with closure ~1e-13 at
    every thread count on both builds, while the unguarded solve is 30x-160x
    that in R and eight-plus orders worse in closure.  The ratio is the claim;
    the constants never were.

    **v5.33.1 -- BOTH ENDS OF THAT RATIO WERE STILL HARD-CODED.**  The ratio
    was right and the two CELLS were not.  On the ubuntu CI images:

    * the pinned reference ``22 TM`` closes at ``3.760e-03`` -- it is not
      clean there, and the test said so ("it cannot calibrate the defect");
    * the pinned defect ``21 TE`` returns ``sum(R)`` = 2.095e-04 against the
      clean population's 2.014e-04 -- 4 % out, i.e. the defect does not
      manifest at that truncation there at all.

    Both cells are now CHOSEN from the per-build scan: the reference is a
    truncation that IS clean here, the defect is the worst silently-wrong
    truncation that IS wrong here, and every bar is a ratio between them.  The
    fail-before content -- with the guard off the wrong answer comes BACK
    instead of being refused -- is unchanged and is what the scan re-proves on
    whatever cell carries it."""
    clean = _thin_clean("tm") + _thin_clean("te")
    assert clean, ("no truncation in the scanned ladder is clean on this "
                   "build; nothing can calibrate the defect.\n    "
                   + _thin_table())
    ref_close = max(r["close"] for r in clean)

    defects = _thin_defects("te") + _thin_defects("tm")
    scored = [(r, _thin_wrongness(r)) for r in defects]
    scored = [(r, s) for r, s in scored if np.isfinite(s)]
    if not scored:
        alive = _census_instrument_is_alive()
        assert alive, (
            "no truncation returns a non-conserving answer AND the census "
            "instrument does not flag a synthetic cond-1e20 operator: the "
            "INSTRUMENT is broken, not the defect absent.\n    "
            + _thin_table())
        pytest.skip(
            "the pre-M1 X-1 answer does not reproduce on any truncation of "
            "the scanned ladder on this build: every cell that returns closes "
            f"better than {_THIN_DEFECT_CLOSURE:.0e}.  This is probe-backed "
            "rather than assumed -- the census instrument still flags a "
            "synthetic cond-1e20 operator, so the screen is live and the "
            "family simply carries no silently-wrong truncation here.  "
            "Scan:\n    " + _thin_table())
    worst, score = max(scored, key=lambda t: t[1])
    ref_R = _thin_converged_sumR(worst["pol"])
    spread = _thin_converged_spread(worst["pol"])
    tail = _thin_converged(worst["pol"])
    defect_R, defect_closure = worst["sumR"], worst["close"]
    print(f"\nX-1 fail-before: defect cell {worst['M']} "
          f"{worst['pol'].upper()} sum(R)={defect_R:.6e} closure="
          f"{defect_closure:.3e}; its polarization's converged tail "
          f"{[r['M'] for r in tail]} reads {ref_R:.6e} +/- {spread:.2%} "
          f"(worst clean closure {ref_close:.3e}); wrongness "
          f"{score:.1f} spreads")

    # (1) it RETURNED -- that is the fail-before content, and reaching this
    #     line at all is most of the assertion.
    assert worst["raised"] is None
    assert np.isfinite(defect_R) and np.isfinite(defect_closure)
    # (2) the control really is clean, so the comparison means something.
    assert ref_close < _THIN_CLEAN_CLOSURE, (
        f"the reference population is not clean on this build "
        f"(worst closure {ref_close:.3e}); it cannot calibrate the defect")
    # (3) and what came back is the WRONG answer, measured in the CONVERGED
    #     tail's own units.  The defect set was classified on CLOSURE alone,
    #     so this is a claim and not a definition.  Measured: 1.5e+05 spreads
    #     (21 TE, 1 thread) down to 170 (the mildest cell that breaks closure
    #     at all, 26 TE at 24 threads) against the bar of 5; the CI cell that
    #     broke the sibling test (sum(R) 2.502e-04 against a 2.014e-04 tail)
    #     would read 240.
    assert score > 5.0, (
        f"the worst non-conserving cell {worst['M']} "
        f"{worst['pol'].upper()} returns sum(R)={defect_R:.6e}, only "
        f"{score:.2f} spreads from the converged tail's {ref_R:.6e} "
        f"-- the pre-M1 defect is not being reproduced with the guard off, "
        f"it is truncation convergence.\n    " + _thin_table())
    # (4) ... and it is SILENT in the sense the audit means: conservation is
    #     violated by orders more than any clean truncation, yet not enough
    #     for `_check_energy` to have raised on it (assertion 1).
    assert defect_closure > 1.0e3 * max(ref_close, 1e-15), (
        f"unguarded closure {defect_closure:.3e} is not orders above the "
        f"clean population's {ref_close:.3e}")
    # (5) THE SWITCH ITSELF, on the cell that carries the defect: the inverse
    #     refusal was WITHDRAWN (see the note above `_INV_RCOND_SCREEN`), so
    #     flipping `INTERFACE_CONDITIONING_GUARD` must not move a bit of this
    #     path -- which is why the wrong answer comes back either way, and is
    #     the fail-before this test's name is about.
    got = []
    for flag in (False, True):
        prev = _rc.INTERFACE_CONDITIONING_GUARD
        _rc.INTERFACE_CONDITIONING_GUARD = flag
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                o, R, T = rcwa_efficiency_1d(
                    THIN["period"], THIN["n_ridge"], THIN["n_groove"],
                    THIN["n_substrate"], THIN["n_superstrate"], THIN["depth"],
                    THIN["duty_cycle"], WL, angle=0.0,
                    polarization=worst["pol"], n_orders=worst["M"],
                    stabilize=False)
            got.append((np.asarray(R).copy(), np.asarray(T).copy()))
        finally:
            _rc.INTERFACE_CONDITIONING_GUARD = prev
    assert float(np.max(np.abs(got[0][0] - got[1][0]))) == 0.0
    assert float(np.max(np.abs(got[0][1] - got[1][1]))) == 0.0


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

def test_t3_3_per_layer_cap_clamps_to_the_window_half_spaces():
    """The fix: the cap comes from the grids the HALF-SPACES live on, so the
    per-layer answer is STATIONARY in ``far_field_orders`` past the capacity
    instead of degrading.  The union grid has 13 cells (``n_glob`` = 78, cap 77
    orders); the END WINDOW grids have 5 (``n_glob`` = 30, cap 29).

    The spread is scored COMPARATIVELY against what the pre-M1 cap produced on
    the same stack: the sibling below measures the un-capped 61-order solve
    drifting 1.6e-06 in ``J00`` while conserving energy, so a stationarity
    spread three orders under that is the claim."""
    ref = None
    spread = 0.0
    for ffo in (7, 21, 31, 41, 61, 77):
        o, R, T, J = _solve_conical(ffo, "per-layer")
        assert int(np.asarray(o).shape[0]) <= 29
        if ref is None:
            ref = complex(np.asarray(J)[0, 0])
        else:
            spread = max(spread, abs(complex(np.asarray(J)[0, 0]) - ref))
    assert spread < 1e-3 * 1.6e-6            # the pre-M1 drift, S4 of the audit


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
    assert abs(complex(np.asarray(J61)[0, 0]) - good) > 1e-6


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
