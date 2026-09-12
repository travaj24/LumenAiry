"""WP-A12 -- the 2026-09-11 adversarial audit's PMM 1-D findings G1-G4.

Every bar below is MEASURED on the running build against something the code
under test did not produce: an analytic TMM, a re-derivation of the same
arithmetic in the test, a deterministic COUNT of solve invocations, or the
NumPy branch used as the oracle for the JAX branch.  Nothing here asserts a
wall-clock number (TESTING_STANDARDS S1) and nothing skips on a resource
precondition (S2).

Findings covered
----------------
G1 (P1)  the differentiable ``PMMStack.solve`` twin returned BEFORE every
         NumPy-branch guard: a concrete gain superstrate returned
         ``R+T = [-0.848, -0.863]`` silently (the audit-M3 defect the NumPy
         path was fixed for), and a manufactured cross-layer sliver returned
         ``T0 = 1.3417`` with ``max R+T = 8.35`` at degrees 14/16/18 while
         agreeing with NumPy at 12 and 20.
G2 (P2)  the default ``min_feature`` sat at the BOTTOM of the measured sliver
         hazard band ``s in [1, 8] * min_feature``, so it snapped away only the
         collisions that were already harmless.
G3 (P2)  the sliver arbiter paid 3 extra whole-stack solves on EVERY solve of a
         sliver-carrying stack; the exactly-diagonal GLL masses were inverted
         densely; ``Q @ W2`` was built twice; the geometric-eig LRU was keyed on
         ``n_glob^2`` complex128 of operator bytes and was not byte-budgeted.
G4 (P3)  ``internal_field(pol=...)`` took 0/1 where the family takes
         ``'te'``/``'tm'``; ``angle=A, theta=T`` with ``A != T`` resolved
         silently; the far-field order budget was copy-pasted ~13 times (one
         copy carrying NO capacity refusal at all); CONVENTIONS §7.1 named the
         wrong 1-D Jones basis.

RE-MEASURED 2026-09-12 (WP-A23; ``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A23_REPORT.md``).  The premise gate(s) in this file were
re-measured on this tree against a freshly re-recorded kernel census
(``validation/probe_ci_kernel_sweep/decisions.json``, 24 arms) and they STILL
HOLD here -- nothing below is skipping for a stale reason.  What HAS changed
is the evidence behind the phrase "the CI runner's kernel": that reading is no
longer remote.  The census now carries a MEASURED arm reproducing it BIT FOR
BIT -- ``WSL-SkylakeX-t4`` returns ``R+T`` = 1.0000010471871335 at the 1e-05
wall separation where ``WIN-SkylakeX-t1`` returns 3.6124215325 on the SAME
fixture and the SAME silicon, one build and one OpenBLAS micro-kernel apart.
Read "the CI runner's kernel" as "a Linux AVX-512 (SkylakeX) arm", which is
reproducible on this workstation and was not on the 2026-09-11 one (SIGILL).
"""
import os

# The sliver fixtures are near-degenerate eigenproblems whose ANSWER is a
# property of the BLAS reduction order (the round-4 finding); pin one thread
# before numpy is imported, as the rest of this family does.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import ast  # noqa: E402
import inspect  # noqa: E402
import pathlib  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumenairy.elements.pmm import PMMStack, pmm_jones_1d  # noqa: E402
from lumenairy.elements.pmm import _core as PC  # noqa: E402
from lumenairy.elements.pmm import stack as PS  # noqa: E402

_REPO = pathlib.Path(__file__).resolve().parents[2]


# ===========================================================================
# shared fixtures
# ===========================================================================
_PER, _WL = 1.0e-6, 1.55e-6
_EPS_HI, _EPS_LO = 3.48 ** 2, 1.444 ** 2
#: The ``min_feature`` the audit's JAX-guard reproducer used, in metres.  It is
#: the PRE-2026-09-12 library default; it has to be stated now because the
#: default moved to ``period * 1e-3`` (finding G2) and at the new default the
#: 1.5e-5 sliver below is snapped away and the fixture carries no sliver.
_MF_OLD = _PER * 1e-5
_SLIVER = 1.5e-5


def _mk(traced, *, degree, ffo=11, segs2=None, n_sup=1.0, angle=0.2,
        min_feature=_MF_OLD):
    """The audit's two-layer Si/SiO2 fixture (repro PMM-1D/p11_jax_guards.py).

    ``traced=True`` makes ONE layer's eps a ``jnp`` array and leaves every other
    input a plain Python/NumPy value, which is exactly how the audit routed the
    solve to the differentiable twin."""
    import jax.numpy as jnp
    st = PMMStack(_PER, n_substrate=1.444, n_superstrate=n_sup, degree=degree,
                  far_field_orders=ffo, min_feature=min_feature)
    eh = jnp.asarray(_EPS_HI + 0j) if traced else _EPS_HI
    st.add_layer(0.25e-6, segments=[(0.5, eh), (0.5, _EPS_LO)])
    if segs2 is not None:
        st.add_layer(0.25e-6, segments=segs2)
    st.set_source(_WL, angle=angle)
    return st


def _sliver_segs():
    return [(0.5 + _SLIVER, _EPS_HI), (0.5 - _SLIVER, _EPS_LO)]


def _run(st):
    """``(kind, R, T, J, messages)`` -- solve, recording warnings."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            o, R, T, J = st.solve()
            return ("ok", np.asarray(R), np.asarray(T), np.asarray(J),
                    [str(w.message) for w in rec])
        except Exception as exc:                      # noqa: BLE001 - reported
            return (type(exc).__name__, None, None, None, [str(exc)])


def _tot(R, T):
    return float(np.max(np.real(R).sum(-1) + np.real(T).sum(-1)))


def _jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return jax


# ===========================================================================
# G1 -- the differentiable PMMStack.solve twin now runs the guards
# ===========================================================================
def test_g1_the_jax_twin_refuses_a_gain_superstrate_exactly_as_numpy_does():
    """A CONCRETE gain superstrate must raise on BOTH branches.

    Pre-fix the JAX branch returned ``R+T = [-0.848, -0.863]`` -- NEGATIVE
    efficiencies, silently -- because ``PMMStack.solve`` dispatched to the twin
    and returned BEFORE ``_require_propagating_incidence``.  ``n_superstrate``
    is fully concrete here, so the documented "a TRACED value skips the guard"
    carve-out does not cover it.  The oracle is the NumPy branch: same stack,
    same source, only the eps dtype differs.
    """
    _jax()
    kw = dict(degree=12, n_sup=1.0 - 1e-3j)
    numpy_side = _run(_mk(False, **kw))
    jax_side = _run(_mk(True, **kw))
    assert numpy_side[0] == "ValueError", numpy_side
    assert jax_side[0] == "ValueError", (
        "the differentiable twin returned instead of refusing; pre-fix it "
        f"returned max R+T = {jax_side[1] is not None and _tot(jax_side[1], jax_side[2])}")
    # the SAME refusal, not merely some refusal
    assert "gain incidence medium" in jax_side[4][0], jax_side[4][0]
    assert jax_side[4][0] == numpy_side[4][0], (jax_side[4][0],
                                                numpy_side[4][0])


def _g1_sliver_ladder():
    """``{degree: (T0, max R+T, energy-warned)}`` over the audit's degree
    ladder, asserting the BUILD-FREE half on the way through: the geometric
    screen must warn at EVERY degree."""
    seen = {}
    for degree in (12, 14, 16, 18, 20):
        out = _run(_mk(True, degree=degree, segs2=_sliver_segs()))
        assert out[0] == "ok", out
        msgs = " || ".join(out[4])
        assert "MANUFACTURED NEAR-COINCIDENT-WALL SLIVER" in msgs, (
            f"degree={degree}: the geometric screen did not fire; it is pure "
            f"geometry and must fire at every degree.  Got: {out[4]}")
        seen[degree] = (float(out[2][1][len(out[2][1]) // 2]),
                        _tot(out[1], out[2]),
                        "energy not conserved" in msgs)
    return seen


def test_g1_the_jax_twin_screens_the_geometry_that_numpy_refuses():
    """The pure-geometry cross-layer SLIVER screen runs before the dispatch --
    the half of the G1 fix that is a fact about WALL COORDINATES and therefore
    holds on every build.

    The screen cannot become the NumPy path's ``ValueError`` (the round-4
    arbiter needs three re-solves that a trace cannot supply), so the contract
    asserted here is: it WARNS at EVERY degree of the ladder, whatever the
    answer at that degree happens to be.  Whether a given degree's answer is
    ALSO corrupt is a property of the running BLAS kernel and is scored
    separately, in the premise-gated sibling below.
    """
    _jax()
    _g1_sliver_ladder()


def test_g1_the_energy_tripwire_fires_on_the_rows_that_are_actually_corrupt():
    """The second half of the G1 fix -- ``_warn_stack_energy`` on the twin's
    CONCRETE outputs -- scored against a real corrupt solve.

    PREMISE-GATED, and the premise is a BUILD PROPERTY, not a resource
    precondition, so ``TESTING_STANDARDS`` S2 ("never ``pytest.skip`` on a
    resource check") does not apply: what is absent on some arms is the
    library's own arithmetic, not a machine the runner could have provided.
    The round-4 sibling
    (``test_fix_pmm2d_mortar_round2.py::test_the_plain_1d_interface_solve_is_
    left_unguarded_and_this_is_why``) records the measurement that forces this
    shape -- the SAME near-degenerate fixture row reads ``max R+T`` = **3.61**
    on this box's kernel and **1.000115** on the CI runner's, with the returned
    answer wrong on the first and right on the second.  A test that asserted
    corruption here would therefore be asserting the kernel.

    So: the corruption is MEASURED; where it is present the tripwire must have
    fired on exactly those rows (bar: the shipped ``_STACK_SUPERUNITY_BAR`` =
    1e-2, and the corrupt rows read ~7.35 above unity -- 2.9 decades of
    margin), and where it is absent the test skips with the readings.  The
    build-free half of the same ladder -- the geometric screen warning at every
    degree -- is asserted unconditionally in the sibling above and is NOT
    relaxed by this gate.
    """
    _jax()
    seen = _g1_sliver_ladder()
    # the audit measured T0(E_y) = 1.3417 against a correct 0.7659 with
    # max R+T = 8.35 at degrees 14/16/18 -- a 735% energy violation.  Scored
    # loosely: SOME degree of the ladder grossly super-unity.
    bad = [d for d, (_t0, tot, _e) in seen.items() if tot > 1.5]
    if not bad:
        pytest.skip(
            "premise absent on this arm: the manufactured-sliver fixture "
            "returns an UNCORRUPTED answer at every degree of the ladder on "
            "this BLAS kernel, so there is no corrupt row for the energy "
            "tripwire to have fired on.  Readings (degree: T0, max R+T, "
            "energy-warned) %r against the audit's 1.3417 / 8.35 at degrees "
            "14/16/18.  The UNCONDITIONAL half of this contract -- the "
            "geometric screen warning at every degree -- passed in "
            "test_g1_the_jax_twin_screens_the_geometry_that_numpy_refuses."
            % (seen,))
    for d in bad:
        assert seen[d][2], (d, seen[d])
        assert seen[d][1] > 1.0 + PS._STACK_SUPERUNITY_BAR, seen[d]


def test_g1_a_clean_jax_stack_is_silent_and_matches_numpy():
    """Negative control: none of the three hoisted guards may fire on a clean
    stack, and the twin's answer must still match NumPy.

    Bar 1e-12 on the per-order efficiencies: the audit measured NumPy-vs-JAX
    parity at 5.9e-14 (R) / 8.7e-14 (T) with x64 enforced, and the fixture's
    own energy closure is ~3e-5, so 1e-12 sits ~1.2 decades above the parity
    floor and ~7 decades below anything the guards could be reacting to.
    """
    _jax()
    clean2 = [(0.5, _EPS_HI), (0.5, _EPS_LO)]
    jx = _run(_mk(True, degree=12, segs2=clean2))
    npy = _run(_mk(False, degree=12, segs2=clean2))
    assert jx[0] == "ok" and npy[0] == "ok", (jx[0], npy[0])
    assert not jx[4], f"the guards fired on a clean stack: {jx[4]}"
    assert float(np.max(np.abs(jx[1] - npy[1]))) < 1e-12
    assert float(np.max(np.abs(jx[2] - npy[2]))) < 1e-12


def test_g1_the_screen_is_trace_safe_and_grad_still_flows():
    """Under ``jax.grad`` the outputs are Tracers, so the energy tripwire is
    skipped by design -- but the geometric screen is pure host geometry and
    must still fire, and the gradient must be unchanged.

    AD-vs-FD bar 1e-6 relative: the audit measured 5.8e-8 on this class and a
    central difference at ``h = 1e-6 * eps`` carries a truncation error
    ~``h^2 f'''/6`` plus a cancellation floor ~``2 eps_mach f / h`` ~ 1e-10,
    so 1e-6 is ~1.2 decades above the measured agreement and ~4 decades below
    a broken gradient (a wrong branch moves this derivative by O(1)).
    """
    jax = _jax()
    import jax.numpy as jnp

    def loss(eh, segs2):
        st = PMMStack(_PER, n_substrate=1.444, n_superstrate=1.0, degree=14,
                      far_field_orders=11, min_feature=_MF_OLD)
        st.add_layer(0.25e-6, segments=[(0.5, eh), (0.5, _EPS_LO)])
        st.add_layer(0.25e-6, segments=segs2(eh))
        st.set_source(_WL, angle=0.2)
        return jnp.real(st.solve()[2][1].sum())

    clean = lambda e: [(0.5, e), (0.5, _EPS_LO)]            # noqa: E731
    sliv = lambda e: [(0.5 + _SLIVER, e), (0.5 - _SLIVER, _EPS_LO)]  # noqa: E731
    e0 = jnp.asarray(_EPS_HI + 0j)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        g = jax.grad(lambda e: loss(e, sliv))(e0)
    assert any("MANUFACTURED NEAR-COINCIDENT-WALL SLIVER" in str(w.message)
               for w in rec), [str(w.message)[:60] for w in rec]
    assert np.all(np.isfinite(np.asarray(g)))

    # ... and the gradient on a CLEAN stack is unchanged and correct
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        gc = jax.grad(lambda e: loss(e, clean))(e0)
    assert not rec, [str(w.message)[:60] for w in rec]
    h = 1e-6 * _EPS_HI
    fd = (float(loss(jnp.asarray(_EPS_HI + h + 0j), clean))
          - float(loss(jnp.asarray(_EPS_HI - h + 0j), clean))) / (2.0 * h)
    ad = float(np.real(np.asarray(gc)))
    assert abs(ad - fd) <= 1e-6 * abs(fd), (ad, fd)


# ===========================================================================
# G2 -- the default min_feature clears the measured hazard band
# ===========================================================================
def test_g2_the_default_min_feature_is_the_measured_value():
    """The default is ``period * 1e-3``.

    Not a taste: the sliver hazard band was measured on TWO independent
    fixtures at ``s in [1, 8] * min_feature`` and the cure was measured on the
    ladder -- 6 of 11 rungs degree-scatter at ``1e-5 * P``, 1 of 11 at
    ``1e-4 * P``, 0 of 11 at ``1e-3 * P``.
    """
    assert PS._MIN_FEATURE_DEFAULT_FRAC == 1.0e-3
    for period in (0.55e-6, 1.0e-6, 1.2e-6):
        st = PMMStack(period, degree=8)
        assert st.min_feature == period * 1e-3
        assert PMMStack(period, degree=8, min_feature=7e-9).min_feature == 7e-9


def test_g2_the_default_snaps_away_the_whole_hazard_band():
    """GEOMETRIC, exact, and instant: across the measured band the default grid
    now carries NO manufactured cell, while the old default left one.

    The screen is the shipped ``_cross_layer_sliver`` -- a deterministic
    function of the wall coordinates and ``min_feature`` -- so this is the
    band's own definition, not a proxy for it.  ``s`` is swept over the band
    the audit measured at the OLD default (1x .. 8x of ``1e-5 * P``) plus the
    rungs on either side.
    """
    for s in (1.0e-5, 1.5e-5, 2.0e-5, 3.0e-5, 5.0e-5, 8.0e-5):
        segs = [[(0.5, _EPS_HI), (0.5, _EPS_LO)],
                [(0.5 + s, _EPS_HI), (0.5 - s, _EPS_LO)]]
        old = PS._cross_layer_sliver(segs, 1e-5)      # pre-fix default
        new = PS._cross_layer_sliver(segs, PS._MIN_FEATURE_DEFAULT_FRAC)
        assert old is not None, (s, "the band premise is absent")
        assert new is None, (s, new, "the default still leaves the sliver")
    # The guard is MOVED, not disarmed: a collision the new threshold does not
    # reach is still manufactured, still screened, and still arbitrated.
    for s in (1.2e-3, 2.0e-3):
        big = [[(0.5, _EPS_HI), (0.5, _EPS_LO)],
               [(0.5 + s, _EPS_HI), (0.5 - s, _EPS_LO)]]
        assert PS._cross_layer_sliver(
            big, PS._MIN_FEATURE_DEFAULT_FRAC) is not None, s


#: The audit's SECOND G2 fixture (TiO2-like 2.35/1.46, 0.55 um pitch, 0.70 um,
#: 31 deg), and the degree ladder it was measured on.  WHICH degree a given
#: rung scatters at is a property of the BLAS kernel (the round-4 finding), so
#: the ladder has to be wide enough to contain the scatter wherever this build
#: puts it.
_G2_PER, _G2_WL, _G2_ANG = 0.55e-6, 0.7e-6, np.deg2rad(31.0)
_G2_E1, _G2_E2 = 2.35 ** 2, 1.46 ** 2
_G2_DEGREES = (10, 14, 18, 22, 26)
_G2_MULTS = (1.0, 1.5, 2.0, 3.0)


def _g2_t0(s, degree, mf):
    """Order-0 ``T`` on the second fixture, with the REFUSAL disarmed so what
    is measured is the answer and not the guard."""
    was = PS.PMM_SLIVER_GUARD
    PS.PMM_SLIVER_GUARD = False
    try:
        st = PMMStack(_G2_PER, n_substrate=1.52, n_superstrate=1.0,
                      degree=degree, far_field_orders=13, min_feature=mf)
        st.add_layer(0.18e-6, segments=[(0.4, _G2_E1), (0.6, _G2_E2)])
        st.add_layer(0.22e-6, segments=[(0.4 + s, _G2_E2), (0.6 - s, _G2_E1)])
        st.set_source(_G2_WL, angle=_G2_ANG)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, _J = st.solve()
        return float(np.real(T[1, int(np.where(o == 0)[0][0])]))
    finally:
        PS.PMM_SLIVER_GUARD = was


def _g2_spreads(mf, mults):
    """Degree-scatter of order-0 ``T`` at each rung, at threshold ``mf``.  ``s``
    is a PERIOD FRACTION (it is added to a segment width), so a rung is the
    multiple of the threshold's own fraction."""
    out = []
    for mult in mults:
        vals = [_g2_t0(mult * (mf / _G2_PER), d, mf) for d in _G2_DEGREES]
        out.append(max(vals) - min(vals))
    return out


def test_g2_the_raised_default_leaves_no_degree_scatter():
    """NUMERIC, on the audit's SECOND fixture -- the UNCONDITIONAL half.

    The diagnostic is DEGREE-SCATTER at fixed ``s``, because a larger ``s`` is
    a genuinely different geometry and a smooth drift with ``s`` is correct
    physics, while an answer that jumps between branches as ``degree`` changes
    is the pathology.  At the new default every rung of the 1x..3x ladder must
    be degree-independent -- and that is a statement about the FIXED code, so
    it holds on every arm and is asserted here without any premise.

    Bar 1e-6 on the scatter.  Derivation: the quantity is an order-0
    transmittance of order 0.2; the solve's own converged-degree spread on this
    cell away from the band is < 1e-7 (the audit's 15x/30x/100x rungs agree to
    7 digits), and the pathology moves it by 5e-2 -- so 1e-6 sits ~1 decade
    above the clean floor and 4.7 decades below the defect.
    """
    new_spread = _g2_spreads(_G2_PER * PS._MIN_FEATURE_DEFAULT_FRAC, _G2_MULTS)
    assert max(new_spread) < 1e-6, new_spread


def test_g2_the_old_default_is_what_scattered():
    """The other side of the same measurement: at the OLD default this fixture
    reads T0 = 0.2645 / 0.3082 / 0.1939 against a correct 0.199230 -- up to 55%
    wrong, scattering by +-5% between ADJACENT degrees.

    PREMISE-GATED, and the premise is a BUILD PROPERTY, not a resource
    precondition, so ``TESTING_STANDARDS`` S2 does not apply: what may be
    absent is the pre-fix arithmetic's misbehaviour on this kernel, which no
    runner configuration can supply.  The round-4 sibling records the
    measurement that forces the shape -- the same near-degenerate class reads
    ``max R+T`` = **3.61** on this box's kernel and **1.000115** on the CI
    runner's.  Where the old default does scatter, it must scatter by more than
    1e-3 (3 decades above the 1e-6 bar its cure is held to); where it does not,
    the test skips with the readings, and the CURE remains asserted
    unconditionally in the sibling above.
    """
    old_spread = _g2_spreads(_G2_PER * 1e-5, _G2_MULTS)
    if not max(old_spread) > 1e-3:
        pytest.skip(
            "premise absent on this arm: at the OLD default (period*1e-5) the "
            "hazard band does not scatter on this BLAS kernel, so the cure "
            "has no defect to be scored against.  Per-rung degree spreads "
            "%r over degrees %r against the audit's ~5e-2.  The "
            "UNCONDITIONAL half -- zero scatter at the NEW default -- passed "
            "in test_g2_the_raised_default_leaves_no_degree_scatter."
            % (old_spread, _G2_DEGREES))
    assert max(old_spread) > 1e-3, old_spread


def test_g2_raising_the_knob_does_not_perturb_what_it_does_not_touch():
    """Where BOTH settings leave a cross-layer collision unsnapped they must
    return the SAME number -- otherwise the default change would be moving
    answers rather than removing manufactured cells.

    ``s`` is chosen ABOVE the new threshold so neither setting snaps it.  Bar:
    bit-identical.  The two runs differ only in a float attribute that no
    surviving code path reads for this geometry, so anything but 0.0 would be
    a real behaviour change.
    """
    P = 1.0e-6

    def solve(mf, s):
        st = PMMStack(P, n_substrate=1.444, n_superstrate=1.0, degree=12,
                      far_field_orders=11, min_feature=mf)
        st.add_layer(0.25e-6, segments=[(0.5, _EPS_HI), (0.5, _EPS_LO)])
        st.add_layer(0.25e-6,
                     segments=[(0.5 + s, _EPS_HI), (0.5 - s, _EPS_LO)])
        st.set_source(_WL, angle=0.2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, J = st.solve()
        return np.asarray(R), np.asarray(T)

    for s in (2.0e-3, 5.0e-3):
        a = solve(P * 1e-5, s)
        b = solve(P * PS._MIN_FEATURE_DEFAULT_FRAC, s)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), s


# ===========================================================================
# G3 -- cost: the arbiter's solves, and the diagonal-mass kernels
# ===========================================================================
_ARB_P, _ARB_WL, _ARB_TH = 1.2e-6, 0.85e-6, 0.15
_ARB_DZ, _ARB_EH, _ARB_EP = 0.32e-6 / 4, 2.25, 9.0
_ARB_A0, _ARB_B0 = 0.27865, 0.62505


#: The arbiter fixtures run at degree 8 and 3-8 slices: the verdict and the
#: SOLVE COUNT -- the only things these tests read -- are the same as at the
#: audit's degree 14 / 16 slices (measured: 'truncation', 2 solves, at
#: (3, 3e-4), (4, 3e-4), (8, 3e-4) and at degree 8 and 10 alike), while one
#: solve costs 0.08-1.5 s instead of minutes.  The audit's own configurations
#: are re-measured in the WP report, not here: a test file on a shared machine
#: must not spend ten minutes re-deriving a count that is degree-independent.
def _arb_stack(n_slices, s, degree=8):
    st = PMMStack(_ARB_P, n_superstrate=1.0, n_substrate=1.0, degree=degree,
                  min_feature=_ARB_P * 1e-5)
    for k in range(n_slices):
        a, b = _ARB_A0 - k * s, _ARB_B0 + k * s
        st.add_layer(_ARB_DZ, segments=[(a, _ARB_EH), (b - a, _ARB_EP),
                                        (1.0 - b, _ARB_EH)])
    st.set_source(_ARB_WL, theta=_ARB_TH)
    return st


def _count_solves(make, repeats=1):
    """Deterministic COUNT of ``PMMStack.solve`` invocations (the audit's own
    instrument -- a wall-clock A/B is unusable on a shared machine, and the
    auditor recorded three failed timing probes to say so).

    ``PMMStack.solve`` itself is wrapped, NOT a ``_core`` helper: ``stack.py``
    binds ``_core`` names at import, so patching there is invisible to the
    caller -- the instrumentation bug the auditor caught and recorded."""
    orig = PMMStack.solve
    n = [0]

    def counting(self, **kw):
        n[0] += 1
        return orig(self, **kw)

    PC._clear_pmm_caches()
    PMMStack.solve = counting
    try:
        st = make()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(repeats):
                try:
                    st.solve()
                except ValueError:
                    pass
    finally:
        PMMStack.solve = orig
    return n[0]


def test_g3_the_arbiter_costs_nothing_on_a_stack_with_no_sliver():
    """Scope control, unchanged by this work package: the cost is paid on
    exactly the stacks that carry a manufactured cross-layer sliver."""
    assert _count_solves(lambda: _arb_stack(4, 0.0)) == 1
    assert _count_solves(lambda: _arb_stack(1, 0.0)) == 1
    # ... and the sliver-carrying twin of the same stack does pay
    assert _count_solves(lambda: _arb_stack(4, 3e-4)) > 1


_ARB_ROWS = ((3, 3e-4), (4, 3e-4))


def _arb_arms(ns, s):
    """``(verdict, solves)`` for the lazy and the eager arm of the same stack,
    plus the solve itself.  The eager arm is the PRE-FIX control flow, restored
    through the library's own switch rather than quoted from the audit."""
    st = _arb_stack(ns, s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, R, T, _J = PMMStack.solve(st)
    worst = _tot(R, T)
    PC._clear_pmm_caches()
    lazy_v = PS._sliver_arbiter(st, worst, R, T, None)
    lazy_n = _count_solves(lambda: _arb_stack(ns, s))
    was = PS.PMM_SLIVER_ARBITER_LAZY
    PS.PMM_SLIVER_ARBITER_LAZY = False
    try:
        eager_n = _count_solves(lambda: _arb_stack(ns, s))
        PC._clear_pmm_caches()
        eager_v = PS._sliver_arbiter(st, worst, R, T, None)
    finally:
        PS.PMM_SLIVER_ARBITER_LAZY = was
    return (lazy_v, lazy_n), (eager_v, eager_n), (R, T)


def test_g3_the_arbiter_switch_changes_the_COST_and_nothing_else():
    """The UNCONDITIONAL half of the laziness contract, and the one that
    matters: whatever verdict this build's arithmetic reaches, the lazy and the
    eager arm must reach the SAME one, return the SAME numbers, and the lazy
    arm must never cost more.

    That is a statement about control flow, not about which fork a
    near-degenerate fixture lands on, so it holds on every BLAS kernel.  The
    only field the laziness is allowed to drop is a collapse-solve product
    (``d12`` / ``d0_over_d12`` / ``closed_super_unity``), and only on the fork
    that provably does not read it.
    """
    for ns, s in _ARB_ROWS:
        (lazy_v, lazy_n), (eager_v, eager_n), _out = _arb_arms(ns, s)
        assert (lazy_v is None) == (eager_v is None), (ns, s)
        assert lazy_n <= eager_n, (ns, s, lazy_n, eager_n)
        if lazy_v is None:
            continue
        assert lazy_v[0] == eager_v[0], (ns, s, lazy_v[0], eager_v[0])
        lz, eg = lazy_v[1], eager_v[1]
        assert set(lz) == set(eg), (sorted(lz), sorted(eg))
        dropped = {k for k, v in lz.items() if v is None and eg[k] is not None}
        assert dropped <= {"d12", "d0_over_d12", "closed_super_unity"}, dropped
        if lazy_v[0] != "truncation":
            assert not dropped, (lazy_v[0], dropped)
        for key in set(lz) - dropped:
            assert np.all(np.asarray(lz[key]) == np.asarray(eg[key])), key


def test_g3_a_truncation_verdict_pays_two_solves_not_four():
    """The COUNT the finding is about: on a ``'truncation'`` verdict the two
    COLLAPSE solves are skipped, 4 -> 2.

    ``d12`` -- the device's own sensitivity to where the contested wall sits --
    is read only on the ``'sliver'``/``'wall'`` fork, which is reached only
    after ``d0`` has cleared the geometric floor, so a stack whose answer did
    not move needs the probe solve and nothing else.

    PREMISE-GATED, and the premise is a BUILD PROPERTY, not a resource
    precondition, so ``TESTING_STANDARDS`` S2 does not apply: WHICH fork this
    near-degenerate fixture is arbitrated onto is decided by the solve's own
    arithmetic, and the round-4 sibling records that arithmetic moving across
    kernels on exactly this fixture family -- ``max R+T`` = **3.61** here
    against **1.000115** on the CI runner, right answer on one arm and wrong on
    the other.  No runner setting can put a row on the truncation fork.  Where
    a row IS arbitrated ``'truncation'`` the counts are asserted exactly
    (2 lazy, 4 eager, both re-derived in-process); where none is, the test
    skips with the verdicts it saw, and the switch's build-free contract stays
    asserted in the sibling above.
    """
    seen = {}
    for ns, s in _ARB_ROWS:
        (lazy_v, lazy_n), (eager_v, eager_n), _out = _arb_arms(ns, s)
        seen[(ns, s)] = (None if lazy_v is None else lazy_v[0], lazy_n, eager_n)
        if lazy_v is None or lazy_v[0] != "truncation":
            continue
        assert eager_n == 4, (ns, s, eager_n)
        assert lazy_n == 2, (ns, s, lazy_n)
        # the eager arm additionally measured the denominator the lazy one
        # skipped, and both agree on the numerator they share
        assert eager_v[0] == "truncation", eager_v[0]
        assert eager_v[1]["d12"] is not None and eager_v[1]["d12"] >= 0.0
        assert lazy_v[1]["d12"] is None
        assert eager_v[1]["move"] == lazy_v[1]["move"]
        return
    pytest.skip(
        "premise absent on this arm: no configuration of the O-11 fixture is "
        "arbitrated 'truncation' on this BLAS kernel, so the lazy fork is not "
        "reached and its 4 -> 2 count cannot be measured.  Verdicts and "
        "(lazy, eager) solve counts seen: %r.  The UNCONDITIONAL half -- that "
        "the switch changes the cost and nothing else -- passed in "
        "test_g3_the_arbiter_switch_changes_the_COST_and_nothing_else."
        % (seen,))


def test_g3_the_geometric_eig_cache_key_is_a_digest_and_is_byte_budgeted():
    """The key was the FULL operator bytes -- ``n_glob^2`` complex128, i.e.
    ~1.4 MB of KEY per entry at a production ``n_glob`` = 300, retained beside
    a ~1.4 MB value in a cache that, unlike its sibling next door, was not
    enrolled in ``LUMENAIRY_CACHE_BUDGET_MB``.
    """
    from lumenairy.cache import ByteBudgetedLRU, cache_report
    assert isinstance(PC._GEO_EIG_CACHE, ByteBudgetedLRU)
    rep = cache_report()["caches"]
    assert "pmm_geometric_eig" in rep, sorted(rep)
    n = 96
    B = (np.random.default_rng(0).normal(size=(n, n))
         + 1j * np.random.default_rng(1).normal(size=(n, n)))
    key = PC._geo_eig_key(b"tensor", B)
    # the digest half is 32 bytes; the whole key must be tiny next to the
    # n^2 complex128 = 147 456 bytes the operator itself occupies
    assert len(key[1]) == 32
    assert len(repr(key)) < 512, len(repr(key))
    assert B.nbytes == n * n * 16
    # ... and it still SEPARATES: a one-ULP change in one entry is a new key
    B2 = B.copy()
    B2[3, 7] = np.nextafter(B2[3, 7].real, 1.0) + 1j * B2[3, 7].imag
    assert PC._geo_eig_key(b"tensor", B2) != key
    assert PC._geo_eig_key(b"scalar", B) != key       # the tag is in the key


def test_g3_diagonal_mass_shortcuts_are_bit_identical_to_lapack():
    """``_safe_inv`` / ``_safe_solve`` / the ``inv(S0) @ X`` row scale take an
    ``O(n)`` path on the exactly-real diagonal GLL mass.

    The oracle is LAPACK itself on the same matrix, and the bar is BIT
    identity, because the substitution is arithmetically exact: this asserts
    that the shortcut fires (``_real_diagonal`` is not ``None``) AND that it
    reproduces the dense result exactly.  Complex diagonals deliberately stay
    dense -- LAPACK's complex division and NumPy's differ in the last 1-2 ULP
    (measured 1000/1000 trials, 2-4e-16 relative), and that is not worth a
    decade of speed here.
    """
    mats = PC._build_sem_tensor_segments(
        1.1e-6, [0.37, 0.28, 0.35],
        [PC._tensor3_dict(np.diag([12.1, 10.9, 9.6]).astype(complex)),
         PC._tensor3_dict(np.eye(3, dtype=complex)),
         PC._tensor3_dict(np.diag([4.0, 3.6, 3.9]).astype(complex))],
        12, 1, True)
    S0 = mats["S0"]
    d = PC._real_diagonal(S0)
    assert d is not None, "the GLL mass is no longer exactly real-diagonal"
    assert np.array_equal(PC._safe_inv(S0), np.linalg.inv(S0))
    rng = np.random.default_rng(11)
    n = S0.shape[0]
    B = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    assert np.array_equal(PC._safe_solve(S0, B), np.linalg.solve(S0, B))
    apply_fn, _ = PC._row_scale_apply(S0)
    assert np.array_equal(apply_fn(B), np.linalg.inv(S0) @ B)
    # the predicate must be capable of saying NO
    assert PC._real_diagonal(np.diag([1.0 + 1j, 2.0])) is None
    assert PC._real_diagonal(np.diag([1.0, 0.0])) is None       # 1/0
    assert PC._real_diagonal(np.array([[1.0, 1e-300], [0.0, 2.0]])) is None


def test_g3_the_whole_diagonal_and_reuse_batch_is_bit_identical_end_to_end():
    """A/B on a real multilayer solve: the fast paths OFF vs ON must agree
    bit-for-bit on R, T and the Jones.

    ``_real_diagonal`` returning ``None`` is exactly the pre-fix arithmetic
    (every caller falls back to ``_safe_inv`` + matmul), so this is a
    fail-before A/B taken in-process rather than a quoted number.
    """
    def solve():
        st = PMMStack(1.0e-6, n_substrate=1.444, n_superstrate=1.0, degree=14,
                      far_field_orders=11, min_feature=1.0e-9)
        st.add_layer(0.22e-6, segments=[(0.45, _EPS_HI), (0.55, 1.0)])
        st.add_layer(0.19e-6, segments=[(0.3, np.diag([8.4, 4.8, 4.8])
                                         .astype(complex)), (0.7, _EPS_LO)])
        st.add_layer(0.11e-6, segments=[(0.6, (1.9 + 0.05j) ** 2), (0.4, 1.0)])
        st.set_source(1.31e-6, angle=0.21)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, J = st.solve()
        return np.asarray(R), np.asarray(T), np.asarray(J)

    PC._clear_pmm_caches()
    on = solve()
    was = PC._real_diagonal
    PC._real_diagonal = lambda A: None
    try:
        PC._clear_pmm_caches()
        off = solve()
    finally:
        PC._real_diagonal = was
    for a, b, nm in zip(on, off, ("R", "T", "J")):
        assert np.array_equal(a, b), (nm, float(np.max(np.abs(a - b))))


def test_g3_the_fourier_projection_scatter_is_bit_identical():
    """``np.add.at`` replaced a per-column Python loop in
    ``_sem_fourier_projection``.  ``l2g`` repeats a global index at the
    periodic wrap, so the accumulation ORDER matters; the oracle is the loop,
    re-implemented here, and the bar is bit identity."""
    mats = PC._build_sem_segments(1.1e-6, [0.37, 0.28, 0.35],
                                  [12.1, 1.0, 4.0 + 1.2j], 12, 1, True)
    orders = np.arange(-6, 7)
    got = PC._sem_fourier_projection(orders, 1.1e-6, mats)

    l2g, elem_bnds, degree = mats["l2g"], mats["elem_bnds"], mats["degree"]
    xg, wg, Lv = PC._sem_projection_quad(
        int(degree), tuple(float(x) for x in mats["ref_nodes"]))
    G = 2.0 * np.pi / 1.1e-6
    want = np.zeros((len(orders), mats["n_glob"]), dtype=PC._C)
    for e in range(len(elem_bnds)):
        xl, xr, _eps = elem_bnds[e]
        J = 0.5 * (xr - xl)
        phase = np.exp(-1j * np.outer(orders * G,
                                      0.5 * (xr + xl) + J * xg))
        contrib = (phase * (wg * J / 1.1e-6)) @ Lv
        for a in range(degree + 1):
            want[:, l2g[e][a]] += contrib[:, a]
    assert np.array_equal(got, want)
    # the wrap really does repeat an index, so this is not a vacuous test
    assert len(set(int(i) for i in l2g[-1])) < l2g.shape[1] or \
        int(l2g[-1][-1]) == int(l2g[0][0])


def test_g3_the_oblique_geometric_eig_genuinely_changes_with_wavelength():
    """The audit proposed a dimensionless-``kx0`` rewrite that would make the
    half-space eig wavelength-independent at every angle.  It cannot exist, and
    this pins why so nobody re-attempts it.

    Scaling ``x`` by the period ``P`` and writing ``kx0 = kxn k0``, the pencil
    is ``[a^2 L~ - i a kxn (C~ - C~^T) + kxn^2 S0~] x = mu S0~ x`` with
    ``a = 1/(P k0)`` -- a QUADRATIC matrix polynomial in the wavelength.  Only
    at ``kxn = 0`` does a single power of ``a`` survive and factor out.  So:
    at NORMAL incidence two wavelengths share the operator exactly, and at
    OBLIQUE incidence they do not.
    """
    mats = PC._build_sem_tensor_segments(
        1.1e-6, [0.5, 0.5],
        [PC._tensor3_dict(np.diag([12.1, 12.1, 12.1]).astype(complex)),
         PC._tensor3_dict(np.eye(3, dtype=complex))], 10, 1, True)
    k0a = 2.0 * np.pi / 1.20e-6
    k0b = 2.0 * np.pi / 1.55e-6
    # NORMAL incidence: two wavelengths share the operator, so the second one
    # HITS the cache -- one entry for two solves.
    PC._clear_geo_eig_cache()
    PC._uniform_geo_eig(mats, k0a, 0.0)
    PC._uniform_geo_eig(mats, k0b, 0.0)
    assert len(PC._GEO_EIG_CACHE) == 1, len(PC._GEO_EIG_CACHE)
    # OBLIQUE at a FIXED angle: kx0 = n sin(theta) k0 scales with k0, so the
    # pencil moves and the sweep MISSES -- two entries for two solves.
    kxn = 1.444 * np.sin(0.3)
    PC._clear_geo_eig_cache()
    oa = PC._uniform_geo_eig(mats, k0a, kxn * k0a)[2] * (k0a * k0a)
    ob = PC._uniform_geo_eig(mats, k0b, kxn * k0b)[2] * (k0b * k0b)
    assert len(PC._GEO_EIG_CACHE) == 2, len(PC._GEO_EIG_CACHE)
    # ... and not by a rounding: the operators differ by O(1) relative, which
    # is what makes the quadratic-in-wavelength argument above concrete.
    rel = float(np.max(np.abs(oa - ob))) / float(np.max(np.abs(oa)))
    assert rel > 1e-3, rel


# ===========================================================================
# G4 -- the API and documentation corrections
# ===========================================================================
def _retained_stack():
    st = PMMStack(_PER, n_substrate=1.444, n_superstrate=1.0, degree=10,
                  far_field_orders=9)
    st.add_layer(0.25e-6, segments=[(0.5, _EPS_HI), (0.5, _EPS_LO)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st.set_source(_WL, angle=0.2).solve(retain_internal=True)
    return st


def test_g4_internal_field_takes_the_family_polarization_spellings():
    """CONVENTIONS §7 pins that ``s``/``te`` and ``p``/``tm`` are accepted
    everywhere, case-insensitively.  ``internal_field`` was the one place in the
    PMM surface that took an integer index instead -- ``pol='x'`` raised
    ``ValueError: pol must be 0 or 1``.

    The mapping is checked against the INDEX spelling it must reproduce, so a
    transposed alias table cannot pass.
    """
    st = _retained_stack()
    f0 = st.internal_field(0.1e-6, pol=0)
    f1 = st.internal_field(0.1e-6, pol=1)
    for name in ("tm", "TM", "p", " P "):
        got = st.internal_field(0.1e-6, pol=name)
        assert np.array_equal(got["Ex"], f0["Ex"]), name
    for name in ("te", "TE", "s", "S"):
        got = st.internal_field(0.1e-6, pol=name)
        assert np.array_equal(got["Ey"], f1["Ey"]), name
    # the two rows are genuinely different, so the check is not vacuous
    assert not np.array_equal(f0["Ex"], f1["Ex"])
    # and an unknown spelling still refuses, with the §2 prefix
    with pytest.raises(ValueError) as e:
        st.internal_field(0.1e-6, pol="circular")
    assert str(e.value).startswith("PMMStack.internal_field: "), str(e.value)
    with pytest.raises(ValueError):
        st.internal_field(0.1e-6, pol=2)


def test_g4_a_conflicting_angle_and_theta_is_no_longer_silent():
    """``set_source(angle=A, theta=T)`` with ``A != T`` resolved to ``T`` with
    no signal at all.  The resolution is a deliberate, test-pinned cross-suite
    contract and does NOT change; what changes is that the ambiguity is now
    audible.

    The ordinary ``theta=``-only call must stay silent -- it leaves ``angle``
    at its ``0.0`` default, which is indistinguishable here from an explicit
    zero.
    """
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = PC._resolve_incidence(0.9, 0.25)
    assert got == 0.25
    assert any("DISAGREE" in str(w.message) for w in rec), \
        [str(w.message) for w in rec]
    for args in ((0.0, 0.25), (0.25, 0.25), (0.9, None)):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            PC._resolve_incidence(*args)
        assert not rec, (args, [str(w.message) for w in rec])
    # end to end through the public setter, and theta still wins
    st = PMMStack(_PER, degree=8)
    st.add_layer(0.2e-6, segments=[(0.5, _EPS_HI), (0.5, 1.0)])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        st.set_source(_WL, angle=0.9, theta=0.25)
    assert any("DISAGREE" in str(w.message) for w in rec)
    assert st._src["angle"] == 0.25


def test_g4_there_is_one_definition_of_the_1d_far_field_order_budget():
    """The 1-D block was copy-pasted 13 times across ``_core.py``,
    ``stack.py`` and the JAX twin.  That is the exact shape this codebase's own
    audits blame for its three worst recent defects (the six-copy factor-i
    defect, the six-copy ``_sqrt_decay`` branch-cut defect, and the T3-3
    conical order-cap defect).

    DISCOVERED, not listed: the sweep counts the idiom's own signature lines in
    the source, so copy N+1 cannot ship silently.

    SCOPE -- this gate covers the 1-D idiom ONLY, and says so because the
    honest statement is narrower than "one definition of the far-field order
    budget".  ``conical.py`` implements a DIFFERENT contract and is
    deliberately not consolidated into :func:`_farfield_order_set`: its
    ``n_orders`` is a HALF-order count, there is no ``2 m + 5`` evanescent
    buffer and no odd-parity trim, and its cap is
    ``(nU * n_el * degree - 1) // 2`` on the shared path and
    ``(min(n_glob_sup, n_glob_sub) - 1) // 2`` on the per-layer one.  Including
    that file in the token sweep below is still worth doing -- it fires if
    anyone copies the 1-D idiom INTO it -- but it is not a check on conical's
    own correctness, and it must not be read as one: conical's formula, its
    T3-3 per-layer direction and its refusal are pinned directly in
    ``test_audit2609_a12_verify_pmm1d.py::
    test_g4_the_conical_order_cap_is_its_own_documented_formula`` and its two
    siblings.
    """
    files = ("lumenairy/elements/pmm/_core.py",
             "lumenairy/elements/pmm/stack.py",
             "lumenairy/elements/pmm/conical.py",
             "lumenairy/elements/pmm/oned.py")
    srcs = {rel: (_REPO / rel).read_text(encoding="utf-8") for rel in files}
    # the parity trim is the 1-D idiom's fingerprint: it must occur EXACTLY
    # once across the whole package, inside _farfield_order_set itself.
    # ``conical.py`` is swept for the SAME tokens, which it has never had --
    # that arm fires only if the 1-D idiom is copied INTO it, and is not a
    # statement about conical's own (different) budget.  See the SCOPE note.
    n = sum(src.count("if n_proj % 2 == 0:") for src in srcs.values())
    assert n == 1, (
        f"the order-budget parity trim occurs {n} times; it belongs in "
        f"_farfield_order_set and nowhere else")
    body = inspect.getsource(PC._farfield_order_set)
    assert "if n_proj % 2 == 0:" in body
    # the two sizing lines the copies carried must be gone entirely
    for token in ("n_proj = max(", "cap = n_glob if"):
        for rel, src in srcs.items():
            assert token not in src, (
                f"{rel} still carries the copied order-budget idiom "
                f"({token!r}); call _farfield_order_set instead")
    # conical's own budget is still THERE and still its own shape -- so that a
    # future "tidy-up" that silently folds it into the 1-D helper (changing a
    # half-order cap into a total-order one) fails here rather than in a user's
    # far field.  Its numeric contract is pinned in the VERIFY file.
    con = srcs["lumenairy/elements/pmm/conical.py"]
    for token in ("cap = (nU * n_el * degree - 1) // 2", "if m_prop > cap:"):
        assert token in con, (
            f"conical.py no longer carries its own half-order budget "
            f"({token!r}).  If that is deliberate, the replacement has to be "
            f"re-derived: _farfield_order_set returns a TOTAL order count and "
            f"conical's n_orders is a HALF count, so a drop-in swap silently "
            f"doubles the projector.  See test_audit2609_a12_verify_pmm1d.py::"
            f"test_g4_the_conical_order_cap_is_its_own_documented_formula.")
    # ... and the ONE definition behaves: cap, odd parity and the refusal
    orders, kx, half = PC._farfield_order_set(
        1e-6, 0.5e-6, 1.5, 21, 41, "probe", degree=8, kx0=0.0, k0=1.0)
    assert len(orders) % 2 == 1 and orders[0] == -half and orders[-1] == half
    assert len(orders) <= 41
    assert np.allclose(kx, orders * (2.0 * np.pi / 1e-6))
    # a two-grid cap takes the MINIMUM of the odd-trimmed counts
    o2, _k2, _h2 = PC._farfield_order_set(1e-6, 0.5e-6, 1.5, 999, (41, 22),
                                          "probe", degree=8)
    assert len(o2) == 21
    with pytest.raises(ValueError) as e:
        PC._farfield_order_set(1e-6, 0.05e-6, 3.5, 5, 9, "probe", degree=4)
    assert str(e.value).startswith("probe: degree=4 too low"), str(e.value)
    with pytest.raises(ValueError) as e:
        PC._farfield_order_set(1e-6, 0.05e-6, 3.5, 5, 9, "probe")
    assert "resolution too low" in str(e.value), str(e.value)


def test_g4_the_prepared_path_refuses_an_under_capacity_order_budget():
    """``_PreparedPMMStack.solve`` was the copy with NO capacity refusal: it
    clamped the projector to the nodal capacity and then returned a far field
    with propagating orders MISSING -- sub-unity power, which the one-sided
    energy tripwire cannot see.  It now refuses like every sibling path.
    """
    st = PMMStack(3.0e-6, n_substrate=3.5, n_superstrate=1.0, degree=2,
                  far_field_orders=5)
    st.add_layer(0.2e-6, segments=[(0.5, 12.1), (0.5, 1.0)])
    prep = st.prepare()
    with pytest.raises(ValueError) as e:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prep.solve(wavelength=0.4e-6, angle=0.0)
    msg = str(e.value)
    assert msg.startswith("PMMStack.prepare().solve: "), msg
    assert "propagating orders" in msg, msg


def test_g4_the_1d_jones_is_the_lab_cartesian_basis_and_jxx_is_minus_rp():
    """CONVENTIONS §7.1 said the 1-D solvers return ``te``/``tm``.  They return
    the lab Cartesian ``(E_x, E_y)`` Jones, whose ``xx`` entry is ``-r_p`` in
    the standard Fresnel ``p`` convention.

    The oracle is an analytic three-medium TMM written here, not library code.
    Bar 1e-12 on the ratio: the audit measured the agreement at 1e-15 / 2.3e-14
    and the sign error this catches is a factor of exactly -1, so the bar sits
    ~3 decades above the floor and 12 below the defect.
    """
    n_sup, n_lay, n_sub = 1.0, 2.1, 1.5
    d, wl = 0.32e-6, 0.55e-6

    def fresnel(ni, nt, ci, ct, pol):
        if pol == "s":
            return (ni * ci - nt * ct) / (ni * ci + nt * ct)
        return (nt * ci - ni * ct) / (nt * ci + ni * ct)

    def tmm(theta, pol):
        s = n_sup * np.sin(theta)
        c = [np.sqrt(1.0 - (s / n) ** 2 + 0j) for n in (n_sup, n_lay, n_sub)]
        r01 = fresnel(n_sup, n_lay, c[0], c[1], pol)
        r12 = fresnel(n_lay, n_sub, c[1], c[2], pol)
        b = 2.0 * np.pi / wl * n_lay * d * c[1]
        return (r01 + r12 * np.exp(2j * b)) / (1.0 + r01 * r12 * np.exp(2j * b))

    for deg in (0.0, 30.0, 60.0):
        th = np.deg2rad(deg)
        e_l = (n_lay ** 2) * np.eye(3, dtype=complex)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, _R, _T, J = pmm_jones_1d(
                0.4e-6, eps_ridge=e_l, eps_groove=e_l, n_substrate=n_sub,
                n_superstrate=n_sup, depth=d, duty_cycle=0.5, wavelength=wl,
                angle=th, degree=12, far_field_orders=9, stabilize=False)
        rp, rs = tmm(th, "p"), tmm(th, "s")
        assert abs(J[0, 0] / rp + 1.0) < 1e-12, (deg, J[0, 0], rp)
        assert abs(J[1, 1] / rs - 1.0) < 1e-12, (deg, J[1, 1], rs)
    # normal incidence is isotropic in the LAB basis -- which a literal
    # te/tm matrix would NOT be, so this is the discriminating statement
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, _R, _T, J0 = pmm_jones_1d(
            0.4e-6, eps_ridge=(n_lay ** 2) * np.eye(3, dtype=complex),
            eps_groove=(n_lay ** 2) * np.eye(3, dtype=complex),
            n_substrate=n_sub, n_superstrate=n_sup, depth=d, duty_cycle=0.5,
            wavelength=wl, angle=0.0, degree=12, far_field_orders=9,
            stabilize=False)
    assert abs(J0[0, 0] - J0[1, 1]) < 1e-13

    # and the SOURCE OF TRUTH now says so
    text = (_REPO / "CONVENTIONS.md").read_text(encoding="utf-8")
    assert "J[0, 0] = -r_p" in text, "CONVENTIONS.md §7.1 not corrected"
    assert "The 1-D solvers return ``te``/``tm``" not in text


def test_g4_the_tm_channel_is_algebraic_and_the_docstring_no_longer_denies_it():
    """``pmm_jones_1d`` claimed "converges SPECTRALLY in the polynomial degree
    with no accuracy floor" without qualification, and ``pmm_jones_1d`` is the
    physics ``PMMStack`` runs.

    The FACT is measured here on a lossless high-contrast cell, where the wall
    corner (not the metal) is the limit: TE accelerates, TM does not.  Oracle:
    each channel's own degree-60 value, which is 5-8 x finer than the coarsest
    rung -- self-referential in level but NOT in RATE, and the rate is the
    claim.  Bar: TE's local convergence order must exceed TM's by at least 2,
    measured over the same rungs.  (The audit measured TE rate ~9 rising and
    TM ~2.7 flat; a gap of 2 sits well inside that 6+ and far above the ~0.3
    the rung-to-rung noise moves it by.)
    """
    n_hi, n_lo = 3.48, 1.0
    kw = dict(period=0.6e-6, n_substrate=1.0, n_superstrate=1.0,
              depth=0.1e-6, duty_cycle=0.5, wavelength=0.633e-6,
              angle=np.deg2rad(10.0), far_field_orders=9, stabilize=False)

    def r0(degree, row):
        e_r = (n_hi ** 2) * np.eye(3, dtype=complex)
        e_g = (n_lo ** 2) * np.eye(3, dtype=complex)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, _T, _J = pmm_jones_1d(eps_ridge=e_r, eps_groove=e_g,
                                        degree=degree, **kw)
        return float(np.real(R[row, int(np.where(o == 0)[0][0])]))

    rate = {}
    for row, name in ((1, "TE"), (0, "TM")):
        ref = r0(48, row)
        errs = [abs(r0(d, row) - ref) for d in (8, 12, 16)]
        # local order from the first and last rung of the ladder
        rate[name] = (np.log(errs[0] / errs[-1])
                      / np.log(16.0 / 8.0)) if errs[-1] > 0 else np.inf
    assert rate["TE"] - rate["TM"] > 2.0, rate

    doc = inspect.getdoc(pmm_jones_1d)
    assert "Converges SPECTRALLY in the polynomial ``degree`` with no" not in doc
    assert "TM" in doc and "wall corner" in doc.lower()
