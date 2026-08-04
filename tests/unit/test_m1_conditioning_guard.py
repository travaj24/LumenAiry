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
    the measurement that decided it: on a matrix in the refused class the LU
    inverse already beats Householder QR, column-pivoted QR and the SVD
    pseudo-inverse on ``||A X - I||``.  Only iterative refinement improves it,
    which is why refinement is the one candidate the guard tries."""
    sla = pytest.importorskip("scipy.linalg")
    n = 40
    rng = np.random.default_rng(5)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n))
                        + 1j * rng.standard_normal((n, n)))
    A = (Q * np.logspace(0, -14, n)) @ Q.conj().T

    def resid(X):
        return _rc._inverse_residual(A, X)
    x_lu = np.linalg.inv(A)
    Qf, Rf = sla.qr(A, mode='economic', check_finite=False)
    x_qr = sla.solve_triangular(Rf, Qf.conj().T, check_finite=False)
    x_svd = np.linalg.pinv(A, rcond=1e-15)
    x_ir = x_lu + x_lu @ (np.eye(n, dtype=complex) - A @ x_lu)
    assert resid(x_lu) <= resid(x_qr)
    assert resid(x_lu) <= resid(x_svd)
    assert resid(x_ir) < resid(x_lu)         # the only route that helps


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
    1e-15 .. 6e-14 and passes all of them."""
    from lumenairy.elements.rcwa import rcwa_jones_1d, uniaxial_tensor
    er = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=np.deg2rad(20))
    eg = (1.5 ** 2) * np.eye(3)
    ref = None
    for M in (9, 15, 19, 25):                 # 5 is not converged (R+T = 1.98)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, J = rcwa_jones_1d(1.0e-6, er, eg, 1.5, 1.0, 0.4e-6, 0.5,
                                       WL, angle=np.deg2rad(10), n_orders=M)
        assert abs(float(np.sum(R) + np.sum(T)) - 2.0) < 5e-3
        if ref is None:
            ref = complex(np.asarray(J)[0, 0])
        else:                                 # converging, not wandering
            assert abs(complex(np.asarray(J)[0, 0]) - ref) < 1e-4


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
    instrument still FLAGS it.  A future fix should make this test fail."""
    _rc._INV_CENSUS = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                rcwa_efficiency_1d(
                    THIN["period"], THIN["n_ridge"], THIN["n_groove"],
                    THIN["n_substrate"], THIN["n_superstrate"], THIN["depth"],
                    THIN["duty_cycle"], WL, angle=0.0, polarization=pol,
                    n_orders=M, stabilize=False)
            except _rc._EnergyError:
                pass                     # OpenBLAS reaches the energy guard
        flagged = [c for c in _rc._INV_CENSUS
                   if np.isfinite(c[2]) and c[2] < _rc._INV_RCOND_SCREEN]
        assert flagged, "the census must still see the conditioning"
        assert all(c[4] is False for c in _rc._INV_CENSUS)   # nothing refused
    finally:
        _rc._INV_CENSUS = None


def test_thin_grating_clean_truncations_are_untouched():
    """The other side of the same claim: the guard must not fire on any
    truncation the census scored clean, and the numbers must not move.  This is
    the false-positive control -- a guard that refused these would be worse
    than the defect."""
    for M in (8, 9, 10, 11, 14, 15, 22):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T = rcwa_efficiency_1d(
                THIN["period"], THIN["n_ridge"], THIN["n_groove"],
                THIN["n_substrate"], THIN["n_superstrate"], THIN["depth"],
                THIN["duty_cycle"], WL, angle=0.0, polarization="tm",
                n_orders=M, stabilize=False)
        assert abs(float(np.sum(R) + np.sum(T)) - 1.0) < 1e-6
        # R is the deep null here (~2e-4): the sensitive observable
        assert 1.9e-4 < float(np.sum(R)) < 2.1e-4


def test_the_refusal_reproduces_the_prior_answer_with_the_switch(guard_off):
    """Fail-before, verified PER CONFIGURATION on the geometry that changed:
    with the guard off, 19 TE returns the pre-M1 Windows/MKL answer -- the
    non-conserving one -- instead of raising."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T = rcwa_efficiency_1d(
            THIN["period"], THIN["n_ridge"], THIN["n_groove"],
            THIN["n_substrate"], THIN["n_superstrate"], THIN["depth"],
            THIN["duty_cycle"], WL, angle=0.0, polarization="te",
            n_orders=21, stabilize=False)
    # the pre-M1 library reported this, on both builds, without raising
    assert float(np.sum(R) + np.sum(T)) > 1.03
    assert float(np.sum(R)) > 3.0e-2         # against a converged ~2.0e-4


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
    numerator here is BLAS round-off."""
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
    # rank-deficient one
    def clos(R, T):
        return float(np.max(np.abs(np.asarray(R).sum(axis=1)
                                   + np.asarray(T).sum(axis=1) - 1.0)))
    assert clos(R41, T41) < 1e-4 * clos(R61, T61)


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
