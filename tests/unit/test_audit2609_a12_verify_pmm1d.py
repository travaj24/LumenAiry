"""VERIFY-A12 -- independent re-verification of the WP-A12 PMM 1-D fixes.

Written by the adversarial verifier, not by the author of the fixes.  Each test
here closes a gap in ``test_audit2609_a12_pmm1d.py``: a claim that file states
but does not pin, or a claim it pins only through a premise the BLAS kernel is
entitled to withhold.

Every bar is measured against something the code under test did not produce --
a re-implementation of the PRE-CHANGE arithmetic written in the test, a
hand-written analytic TMM, or pure geometry.  No wall clock is read.

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

# The 1-D PMM sliver fixtures are near-degenerate eigenproblems whose answer is
# a property of the BLAS reduction order (the round-4 finding); pin one thread
# before numpy is imported, as the rest of this family does.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumenairy.elements.pmm import (  # noqa: E402
    PMMStack,
    pmm_efficiency_1d,
    pmm_jones_1d,
)
from lumenairy.elements.pmm import _core as PC  # noqa: E402
from lumenairy.elements.pmm import conical as PCON  # noqa: E402
from lumenairy.elements.pmm import stack as PS  # noqa: E402
from lumenairy.elements.rcwa import WoodNudgeWarning  # noqa: E402


# ===========================================================================
# G1 -- the energy tripwire on the differentiable twin, WITHOUT a premise
# ===========================================================================
def test_g1_the_concrete_energy_tripwire_is_exact_and_build_free():
    """``_warn_stack_energy_concrete`` is the second half of the G1 fix, and
    the shipped test scores it only on a fixture whose corruption is a
    property of the running BLAS kernel (the round-4 finding: the same row
    reads 1.000115 on one kernel and 3.612 on another).  Here it is scored on
    arrays constructed in the test, so the contract holds on every arm.

    The three severities are the NumPy branch's own, unchanged: non-finite ->
    raise, negative -> raise, ``R+T > 1 + _STACK_SUPERUNITY_BAR`` -> warn.
    Bars: the shipped ``_STACK_SUPERUNITY_BAR`` itself, probed one decade
    either side of it (0.1 x bar must be silent, 10 x bar must warn), so the
    assertion sits nowhere near a threshold the arithmetic could move -- these
    arrays are exact decimals, not solve output.
    """
    bar = PS._STACK_SUPERUNITY_BAR
    R = np.array([[0.25, 0.25], [0.25, 0.25]])

    def totals(extra):
        """R + T with max(R+T) = 1 + extra, exactly."""
        return R, np.full_like(R, 0.25) + 0.5 * extra

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert PS._warn_stack_energy_concrete(*totals(0.1 * bar)) is True
    assert not [w for w in rec if "energy not conserved" in str(w.message)], \
        [str(w.message)[:70] for w in rec]

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        PS._warn_stack_energy_concrete(*totals(10.0 * bar))
    assert [w for w in rec if "energy not conserved" in str(w.message)], \
        [str(w.message)[:70] for w in rec]

    # NEGATIVE total -- the gain-superstrate signature the audit measured at
    # R+T = [-0.848, -0.863] on the twin -- must RAISE, not warn.
    with pytest.raises(ValueError, match="NEGATIVE total efficiency"):
        PS._warn_stack_energy_concrete(-R, -R)
    # NON-FINITE -- the NaN-blind one-sided comparison of audit M3.
    with pytest.raises(ValueError, match="non-finite total efficiency"):
        PS._warn_stack_energy_concrete(R * np.nan, R)


def test_g1_the_tripwire_stands_down_on_a_traced_output():
    """Inside ``jit``/``grad`` the twin's outputs are Tracers and the
    comparison cannot be taken without severing the trace, so the tripwire
    must return ``False`` and warn nothing -- the documented scope limit.  The
    detection is a typed ``isinstance(x, jax.core.Tracer)``, so a CONCRETE
    ``jax.Array`` must still be tripped: that is the case the audit measured
    (an eager ``solve()`` on a stack that merely holds a ``jnp`` array).
    """
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    concrete = jnp.asarray(np.full((2, 2), -0.25))
    assert PS._is_traced_output(concrete) is False
    with pytest.raises(ValueError, match="NEGATIVE total efficiency"):
        PS._warn_stack_energy_concrete(concrete, concrete)

    seen = {}

    def probe(x):
        seen["traced"] = PS._is_traced_output(x)
        seen["ran"] = PS._warn_stack_energy_concrete(x, x)
        return jnp.sum(x)

    jax.jit(probe)(jnp.asarray(np.full((2, 2), -0.25)))
    assert seen["traced"] is True
    assert seen["ran"] is False          # skipped, and nothing raised


# ===========================================================================
# G2 -- the single-layer ownership rule under the RAISED default
# ===========================================================================
@pytest.mark.parametrize("t", [1.0e-2, 1.0e-3, 5.0e-4, 1.0e-4, 1.0e-5, 1.0e-6])
def test_g2_a_single_layer_thin_feature_survives_the_raised_default(t):
    """Raising the default snap by two decades must not start eating features
    a SINGLE layer owns -- an intentional 1 nm liner on a 1 um pitch is
    ``t = 1e-3`` and the sub-nm ones are below it.

    The bar is the union grid itself: byte-identical walls at the old and the
    new default.  That is exact geometry, not a solve, so there is no build
    freedom in it at all.  The cross-layer twin of the same collision (next
    test) is the two-sided half: the rule is ownership, not size.
    """
    liner = [(0.5 - t / 2, 12.1), (t, 2.1), (0.5 - t / 2, 12.1)]
    other = [(0.3, 2.1), (0.7, 12.1)]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        w_old, _e, _o = PC._pmm_union_grid([liner, other], 1e-5,
                                           return_owners=True)
        w_new, _e2, _o2 = PC._pmm_union_grid([liner, other],
                                            PS._MIN_FEATURE_DEFAULT_FRAC,
                                            return_owners=True)
    assert not [r for r in rec if "snapped" in str(r.message)], \
        [str(r.message)[:80] for r in rec]
    assert np.array_equal(w_old, w_new), (t, w_old, w_new)
    # the liner is still a cell of its own width (walls are cumulative sums,
    # so allow the 1-ULP-of-1.0 the sum itself carries: 2.3e-16)
    assert np.any(np.abs(np.asarray(w_new) - t) < 1e-15), (t, w_new)


@pytest.mark.parametrize("s", [5.0e-4, 1.0e-4, 1.0e-5])
def test_g2_the_cross_layer_twin_of_that_collision_is_snapped(s):
    """The two-sided half: the SAME separation, owned by two DIFFERENT layers,
    is a manufactured cell and the raised default removes it.  Measured: the
    narrowest union cell goes from ``s`` (old default, unsnapped) to ~0.5 of a
    period (new default, the pair merged), i.e. the cell is gone, not thinned.
    """
    a = [(0.5, 12.1), (0.5, 2.1)]
    b = [(0.5 + s, 2.1), (0.5 - s, 12.1)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w_old, _e, _o = PC._pmm_union_grid([a, b], 1e-5, return_owners=True)
        w_new, _e2, _o2 = PC._pmm_union_grid([a, b],
                                            PS._MIN_FEATURE_DEFAULT_FRAC,
                                            return_owners=True)
    assert abs(float(np.min(w_old)) - s) < 1e-15, (s, np.min(w_old))
    assert float(np.min(w_new)) > 0.4, (s, np.min(w_new))


# ===========================================================================
# G4 -- the order-budget helper against the PRE-CHANGE block, exhaustively
# ===========================================================================
def _pre_change_scalar(period, wl, n_max, ffo, n_glob, label, degree):
    """``_pmm_solve_core``'s order-budget block as it stood at 56a76f22^,
    transcribed verbatim.  This is the oracle: the consolidation is only
    correct if it reproduces this on every input, including the refusal AND
    its exact message."""
    m_prop = PC._n_propagating_orders(period, wl, n_max)
    n_proj = max(int(ffo), 2 * m_prop + 5)
    cap = n_glob if n_glob % 2 else n_glob - 1
    n_proj = min(n_proj, cap)
    if n_proj % 2 == 0:
        n_proj -= 1
    half = (n_proj - 1) // 2
    if 2 * m_prop + 1 > n_proj:
        raise ValueError(
            f"{label}: degree={degree} too low to resolve the "
            f"{2 * m_prop + 1} propagating orders (n_glob={n_glob}); raise "
            f"degree or elements_per_region.")
    return np.arange(-half, half + 1), half


def _pre_change_pair(period, wl, n_max, ffo, n0, nN):
    """``_solve_vertical_perlayer``'s two-half-space variant, likewise."""
    m_prop = PC._n_propagating_orders(period, wl, n_max)
    n_proj = max(int(ffo), 2 * m_prop + 5)
    cap = min(n0 if n0 % 2 else n0 - 1, nN if nN % 2 else nN - 1)
    n_proj = min(n_proj, cap)
    if n_proj % 2 == 0:
        n_proj -= 1
    if 2 * m_prop + 1 > n_proj:
        raise ValueError("refuse")
    half = (n_proj - 1) // 2
    return np.arange(-half, half + 1), half


def test_g4_the_order_budget_helper_is_the_pre_change_block_on_every_input():
    """13 near-verbatim copies became one helper.  The shipped test checks
    that no copy REMAINS (a source sweep); this checks that the survivor
    COMPUTES the same thing, on a grid that deliberately straddles every
    branch the block has: ``ffo`` below and above ``2 m + 5``, odd and even
    ``ffo``, odd and even nodal capacity, and capacities on both sides of the
    refusal.

    The oracle is the pre-change arithmetic re-implemented above, so this is a
    fail-before comparison and not a restatement.  Bar: EXACT equality of the
    order array, ``half``, whether it refused, and the refusal text.
    """
    mismatches = []
    n_cells = 0
    for period in (0.4e-6, 1.0e-6, 3.1e-6):
        for wl in (0.4e-6, 0.633e-6, 1.55e-6):
            for n_max in (1.0, 1.5, 3.48):
                for ffo in (1, 2, 5, 6, 11, 12, 21, 40, 61):
                    for n_glob in range(2, 34):
                        n_cells += 1
                        try:
                            o, h = _pre_change_scalar(period, wl, n_max, ffo,
                                                      n_glob, "L", 7)
                            old = ("ok", tuple(int(x) for x in o), h)
                        except ValueError as exc:
                            old = ("raise", str(exc))
                        try:
                            o2, _kx, h2 = PC._farfield_order_set(
                                period, wl, n_max, ffo, n_glob, "L", degree=7)
                            new = ("ok", tuple(int(x) for x in o2), h2)
                        except ValueError as exc:
                            new = ("raise", str(exc))
                        if old != new:
                            mismatches.append((period, wl, n_max, ffo, n_glob,
                                               old, new))
    assert not mismatches, (len(mismatches), n_cells, mismatches[:3])
    # both branches were genuinely exercised, so the sweep is not vacuous
    assert n_cells > 2000

    pair_mismatches = []
    for wl in (0.4e-6, 1.55e-6):
        for n_max in (1.0, 3.48):
            for ffo in (5, 11, 12, 21):
                for n0 in range(2, 26):
                    for nN in range(2, 26):
                        try:
                            o, h = _pre_change_pair(1.0e-6, wl, n_max, ffo,
                                                    n0, nN)
                            old = ("ok", tuple(int(x) for x in o), h)
                        except ValueError:
                            old = ("raise",)
                        try:
                            o2, _kx, h2 = PC._farfield_order_set(
                                1.0e-6, wl, n_max, ffo, (n0, nN), "L",
                                degree=7)
                            new = ("ok", tuple(int(x) for x in o2), h2)
                        except ValueError:
                            new = ("raise",)
                        if old != new:
                            pair_mismatches.append((wl, n_max, ffo, n0, nN))
    assert not pair_mismatches, (len(pair_mismatches), pair_mismatches[:3])

    # the kx the helper hands back is the one the copies formed themselves
    for kx0, k0, period in ((0.0, 1e7, 1e-6), (3.3e6, 9.9e6, 0.6e-6),
                            (-1.1e6, 5.0e6, 2.0e-6)):
        o, kx, _h = PC._farfield_order_set(period, 0.633e-6, 1.5, 11, 41, "L",
                                           degree=7, kx0=kx0, k0=k0)
        assert np.array_equal(kx, (kx0 + o * (2.0 * np.pi / period)) / k0)


def test_g4_the_prepared_path_returned_orders_missing_before_the_fix():
    """The fail-before for the ``_PreparedPMMStack.solve`` refusal, taken
    in-process: with the pre-change block (clamp, no refusal) restored, the
    prepared path RETURNS a far field carrying fewer orders than propagate,
    with sub-unity power and NO warning -- which the one-sided energy tripwire
    cannot see.

    Measured here on a 3 um pitch at 0.5 um into n = 1.5 (m_prop = 9, so 19
    orders propagate): degree 4 returned 7 orders at max R+T = 0.894, degree 5
    returned 9 at 0.994, degree 6 returned 11 at 0.959 -- silently.  Bars: the
    returned order count must be BELOW ``2 m + 1`` (an integer fact), and the
    fixed path must refuse (a raise, not a number), so neither side of this
    comparison is a tolerance.
    """
    def build(degree):
        st = PMMStack(3.0e-6, n_substrate=1.5, n_superstrate=1.0,
                      degree=degree, far_field_orders=41)
        st.add_layer(0.2e-6, segments=[(0.5, 6.25), (0.5, 1.0)])
        return st.prepare()

    m_prop = PC._n_propagating_orders(3.0e-6, 0.5e-6, 1.5)
    assert m_prop >= 4, m_prop
    pre_change = PS._farfield_order_set

    def clamp_only(period, wl, n_max, ffo, n_glob, label, *, degree=None,
                   kx0=0.0, k0=None, when=""):
        m = PC._n_propagating_orders(period, wl, n_max)
        n_proj = max(int(ffo), 2 * m + 5)
        cap = n_glob if n_glob % 2 else n_glob - 1
        n_proj = min(n_proj, cap)
        if n_proj % 2 == 0:
            n_proj -= 1
        half = (n_proj - 1) // 2
        orders = np.arange(-half, half + 1)
        kx = None if k0 is None else (kx0 + orders * (2.0 * np.pi / period)) / k0
        return orders, kx, half

    dropped = {}
    PS._farfield_order_set = clamp_only
    try:
        for degree in (4, 5, 6):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                o, R, T, _J = build(degree).solve(wavelength=0.5e-6, angle=0.0)
            dropped[degree] = (len(np.asarray(o)),
                               float(np.max(np.real(R).sum(-1)
                                            + np.real(T).sum(-1))),
                               len(rec))
    finally:
        PS._farfield_order_set = pre_change

    for degree, (n_ord, tot, n_warn) in dropped.items():
        assert n_ord < 2 * m_prop + 1, (degree, n_ord, m_prop)
        assert tot < 1.0, (degree, tot)          # power is MISSING, not excess
        assert n_warn == 0, (degree, n_warn)     # and nothing said so

    for degree in (4, 5, 6):
        with pytest.raises(ValueError) as exc:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                build(degree).solve(wavelength=0.5e-6, angle=0.0)
        assert str(exc.value).startswith("PMMStack.prepare().solve: "), \
            str(exc.value)


# ===========================================================================
# G4 -- CONVENTIONS 7.1 on the OTHER 1-D Jones solver, over a wider fan
# ===========================================================================
def test_g4_the_stack_solver_also_returns_the_lab_cartesian_jones():
    """CONVENTIONS §7.1's new sentence is about EVERY solver, and the shipped
    test measures only ``pmm_jones_1d`` at 0/30/60 deg.  ``PMMStack`` is the
    other 1-D Jones path (a different cascade, a different assembly), and the
    statement has to hold there too, away from the three angles that were
    fitted.

    Oracle: a three-medium characteristic-matrix TMM written here, in the
    standard Fresnel ``p`` convention ``r_p = (n2 c1 - n1 c2)/(n2 c1 + n1 c2)``
    -- so ``r_p(0) = -r_s(0)`` -- with NO library code in it.  Bar 1e-11 on the
    ratio: measured -1.0000000 / +1.0000000 to 7 printed digits at every angle
    below, the defect it separates is a factor of exactly -1 (2 in the ratio,
    11 decades above the bar), and the solve's own truncation on an unpatterned
    cell at degree 12 is ~1e-14 (audit CFC-6: 2.3e-14 in |r| and 1.7e-13 in
    phase), i.e. ~3 decades below it.
    """
    n_sup, n_lay, n_sub = 1.0, 2.1, 1.5
    d, wl, per = 0.32e-6, 0.55e-6, 0.4e-6

    def tmm(theta, pol):
        s = n_sup * np.sin(theta)
        c = [np.sqrt(1.0 - (s / n) ** 2 + 0j) for n in (n_sup, n_lay, n_sub)]

        def r(ni, nt, ci, ct):
            if pol == "s":
                return (ni * ci - nt * ct) / (ni * ci + nt * ct)
            return (nt * ci - ni * ct) / (nt * ci + ni * ct)

        r01 = r(n_sup, n_lay, c[0], c[1])
        r12 = r(n_lay, n_sub, c[1], c[2])
        ph = np.exp(2j * (2.0 * np.pi / wl) * n_lay * d * c[1])
        return (r01 + r12 * ph) / (1.0 + r01 * r12 * ph)

    eps = (n_lay ** 2) * np.eye(3, dtype=complex)
    for deg in (0.0, 20.0, 45.0, 75.0):
        th = np.deg2rad(deg)
        st = PMMStack(per, n_substrate=n_sub, n_superstrate=n_sup, degree=12,
                      far_field_orders=9)
        st.add_layer(d, segments=[(0.5, eps), (0.5, eps)])
        st.set_source(wl, angle=th)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, _R, _T, J = st.solve()
        assert abs(J[0, 0] / tmm(th, "p") + 1.0) < 1e-11, (deg, J[0, 0])
        assert abs(J[1, 1] / tmm(th, "s") - 1.0) < 1e-11, (deg, J[1, 1])
        # off-diagonal must vanish on an unpatterned cell: the lab basis is
        # the eigenbasis there, so a rotated (te/tm) matrix would NOT be
        # diagonal away from normal incidence.
        assert abs(J[0, 1]) < 1e-13 and abs(J[1, 0]) < 1e-13, (deg, J)


# ===========================================================================
# G3 -- the diagonal fast path must REFUSE a lossy (complex) 1/eps mass
# ===========================================================================
def test_g3_a_lossy_inverse_permittivity_mass_stays_on_the_dense_path():
    """``_real_diagonal`` gates on REAL, not merely diagonal, and the reason
    is arithmetic: on a complex diagonal LAPACK's division and NumPy's differ
    in the last 1-2 ULP, so taking the shortcut there would move every layer
    mode of a LOSSY cell.  This pins the gate from the physics side -- the
    ``1/eps`` mass of an absorbing segment -- rather than from a synthetic
    matrix.

    Two-sided: the lossless twin of the same cell MUST take the shortcut (or
    the gate would be vacuous), and the lossy one must not.
    """
    def mass(eps_mid):
        mats = PC._build_sem_tensor_segments(
            1.1e-6, [0.37, 0.28, 0.35],
            [PC._tensor3_dict(np.diag([12.1, 10.9, 9.6]).astype(complex)),
             PC._tensor3_dict((eps_mid * np.eye(3)).astype(complex)),
             PC._tensor3_dict(np.diag([4.0, 3.6, 3.9]).astype(complex))],
            12, 1, True)
        return mats

    lossless = mass(2.25 + 0.0j)
    lossy = mass(2.25 + 0.4j)
    # the geometric mass is real-positive whatever the material
    assert PC._real_diagonal(lossless["S0"]) is not None
    assert PC._real_diagonal(lossy["S0"]) is not None
    # the 1/eps mass follows the material
    assert PC._real_diagonal(lossless["mass"]["inv_xx"]) is not None
    assert PC._real_diagonal(lossy["mass"]["inv_xx"]) is None
    # and the shipped entry points route accordingly, bit for bit
    m = lossy["mass"]["inv_xx"]
    assert np.array_equal(PC._safe_inv(m), np.linalg.inv(m))
    ml = lossless["mass"]["inv_xx"]
    assert np.array_equal(PC._safe_inv(ml), np.linalg.inv(ml))


def test_g3_the_lazy_arbiter_leaves_every_measured_field_untouched():
    """The laziness may zero out ONLY the three fields the skipped solves
    would have produced.  Anything else moving would make it a behaviour
    change rather than a cost change.

    The comparison is field-by-field between the two arms of the shipped
    ``PMM_SLIVER_ARBITER_LAZY`` switch on the same stack and the same
    ``(worst, R, T)``, so there is no tolerance anywhere in it: every shared
    field must be EQUAL.
    """
    p, wl, th = 1.2e-6, 0.85e-6, 0.15
    dz, e_h, e_p = 0.32e-6 / 4, 2.25, 9.0
    a0, b0 = 0.27865, 0.62505

    st = PMMStack(p, n_superstrate=1.0, n_substrate=1.0, degree=8,
                  min_feature=p * 1e-5)
    for k in range(4):
        a, b = a0 - k * 3e-4, b0 + k * 3e-4
        st.add_layer(dz, segments=[(a, e_h), (b - a, e_p), (1.0 - b, e_h)])
    st.set_source(wl, theta=th)

    was_guard = PS.PMM_SLIVER_GUARD
    PS.PMM_SLIVER_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, R, T, _J = st.solve()
    finally:
        PS.PMM_SLIVER_GUARD = was_guard
    worst = float(np.max(np.real(R).sum(-1) + np.real(T).sum(-1)))

    verdicts = {}
    was = PS.PMM_SLIVER_ARBITER_LAZY
    try:
        for lazy in (False, True):
            PS.PMM_SLIVER_ARBITER_LAZY = lazy
            PC._clear_pmm_caches()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                verdicts[lazy] = PS._sliver_arbiter(st, worst, R, T, None)
    finally:
        PS.PMM_SLIVER_ARBITER_LAZY = was

    assert verdicts[False] is not None and verdicts[True] is not None
    assert verdicts[False][0] == verdicts[True][0], (verdicts[False][0],
                                                     verdicts[True][0])
    eager, lazy = verdicts[False][1], verdicts[True][1]
    assert set(eager) == set(lazy), (sorted(eager), sorted(lazy))
    lazy_none = {k for k, v in lazy.items() if v is None}
    # ONLY the collapse-solve products may be dropped, and only on the fork
    # that does not read them
    assert lazy_none <= {"d12", "d0_over_d12", "closed_super_unity"}, lazy_none
    if verdicts[True][0] != "truncation":
        assert not lazy_none, (verdicts[True][0], lazy_none)
    for key in set(eager) - lazy_none:
        assert np.all(np.asarray(eager[key]) == np.asarray(lazy[key])), key


# ===========================================================================
# G3(d) follow-up -- the Wood-anomaly nudge names the CALLER, not the helper
# ===========================================================================
_WOOD_P = 1.0e-6


def _wood_names(fn):
    """The ``fn_name`` prefixes of every ``WoodNudgeWarning`` ``fn`` raises."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn()
    return [str(w.message).split(":")[0] for w in rec
            if isinstance(w.message, WoodNudgeWarning)]


def _wood_stack_conical(wl):
    st = PMMStack(_WOOD_P, n_substrate=1.0, n_superstrate=1.0, degree=8,
                  far_field_orders=5)
    st.add_layer(0.2e-6, segments=[(0.5, 2.25), (0.5, 1.0)])
    st.set_source(wl, theta=0.0, phi=0.3)
    return st.solve()


#: ``(label, call)``.  ``wl = period`` with an air superstrate puts the
#: ``m = +-1`` order EXACTLY on its Rayleigh cut-off, which is the only
#: condition under which the nudge -- and therefore the warning -- fires.
_WOOD_CASES = [
    ("pmm_efficiency_1d",
     lambda wl: pmm_efficiency_1d(_WOOD_P, 1.5, 1.0, 1.0, 1.0, 0.2e-6, 0.5,
                                  wl, angle=0.0, degree=8,
                                  far_field_orders=5)),
    ("pmm_jones_1d",
     lambda wl: pmm_jones_1d(_WOOD_P, 2.25 * np.eye(3), np.eye(3), 1.0, 1.0,
                             0.2e-6, 0.5, wl, angle=0.0, degree=8,
                             far_field_orders=5)),
    ("PMMStack.solve (conical)", _wood_stack_conical),
]


@pytest.mark.parametrize("label,call", _WOOD_CASES,
                         ids=[c[0] for c in _WOOD_CASES])
def test_g3_the_wood_nudge_warning_names_the_entry_point_the_caller_called(
        label, call):
    """WP-A12 threaded ``fn_name=`` into WP-A14's ``WoodNudgeWarning`` so the
    nudge names the PUBLIC entry point rather than the private helper
    (``_wood_safe_wl_1d`` / ``_grazing_safe_wavelength``).  The WP verified
    that by hand and shipped no test for it; this is the test.

    Two-sided.  AT the cut-off (``wl = period`` with an air superstrate, where
    the ``m = +-1`` order has ``kz = 0`` exactly) the warning must fire AND be
    prefixed with this entry point's own name -- a regression to the helper's
    name, or to no ``fn_name`` at all, changes that prefix and fails here.
    AWAY from the cut-off the call must be SILENT, which is what makes the
    first half a statement about the nudge rather than about a noise floor:
    the nudge is an identity away from grazing, and the shipped contract is
    that ordinary solves stay byte-unchanged and quiet.
    """
    at_cut = _wood_names(lambda: call(_WOOD_P))
    assert at_cut, (
        f"{label}: no WoodNudgeWarning at wl = period, where the m = +-1 "
        f"order sits exactly on its Rayleigh cut-off")
    assert set(at_cut) == {label}, (label, at_cut)

    # ... and a wavelength nowhere near a cut-off is silent
    off_cut = _wood_names(lambda: call(_WOOD_P * 0.813))
    assert off_cut == [], (label, off_cut)


# ===========================================================================
# G4 follow-up -- conical keeps its OWN order budget: pin ITS contract
# ===========================================================================
_CON_P, _CON_WL = 1.0e-6, 1.55e-6
_CON_DEG, _CON_NEL = 6, 1
#: Three layers with three DIFFERENT interior walls, so the union grid is
#: strictly finer than any one layer's window and the two cap formulas below
#: give different numbers -- which is what makes the comparison discriminating.
_CON_LAYERS = [(0.20e-6, [0.3, 0.7], [2.25, 1.0]),
               (0.18e-6, [0.5, 0.5], [1.0, 4.0]),
               (0.15e-6, [0.7, 0.3], [4.0, 2.25])]


def _con_segs():
    return [list(zip(w, [np.asarray(e * np.eye(3), dtype=complex)
                         for e in eps]))
            for _t, w, eps in _CON_LAYERS]


def _con_specs():
    return [(t, list(w), [np.asarray(e * np.eye(3), dtype=complex)
                          for e in eps])
            for t, w, eps in _CON_LAYERS]


def _con_solve(layer_grids, n_orders, *, eps_sup=1.0, eps_sub=1.0,
               wavelength=_CON_WL):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PCON._conical_nodal_solve(
            _CON_P, _con_specs(), eps_sup, eps_sub, wavelength, 0.15, 0.3,
            _CON_DEG, _CON_NEL, True, n_orders,
            label="probe conical", layer_grids=layer_grids,
            window_halfwidth=1)


def _con_expected_caps():
    """``(shared_cap, perlayer_cap)`` from the two DOCUMENTED formulas, built
    here out of the same geometry helpers the solver uses."""
    segs = _con_segs()
    uw, _eps_u = PC._pmm_union_grid(segs, None)
    shared = (len(uw) * _CON_NEL * _CON_DEG - 1) // 2

    grid_of = PC._perlayer_window_grids(segs, None, 1)
    t_air = PC._tensor3_dict(np.eye(3, dtype=complex))
    n0 = PC._build_sem_tensor_segments(
        _CON_P, grid_of[0][0], [t_air] * len(grid_of[0][0]),
        _CON_DEG, _CON_NEL, True)["n_glob"]
    nN = PC._build_sem_tensor_segments(
        _CON_P, grid_of[-1][0], [t_air] * len(grid_of[-1][0]),
        _CON_DEG, _CON_NEL, True)["n_glob"]
    return shared, (min(int(n0), int(nN)) - 1) // 2


@pytest.mark.parametrize("layer_grids,which", [("shared", 0), ("per-layer", 1)])
def test_g4_the_conical_order_cap_is_its_own_documented_formula(layer_grids,
                                                                which):
    """``conical.py`` is NOT part of the ``_farfield_order_set``
    consolidation and must not be: its budget is a different contract -- a
    HALF-order cap (``n_orders`` is a half count there, not a total), with no
    ``2 m + 5`` evanescent buffer and no odd-parity trim, and with the shared
    path sizing from ``nU * n_el * degree`` rather than from ``n_glob``.

    The 1-D "one definition" sweep cannot see that file's correctness -- it
    checks for the ABSENCE of the 1-D idiom's tokens, which conical never had
    -- so the contract is pinned HERE, directly:

        shared    cap = (nU * n_el * degree - 1) // 2
        per-layer cap = (min(n_glob_sup, n_glob_sub) - 1) // 2     [T3-3]

    Observed by asking for far more orders than the grid can carry: the solver
    clamps with ``n_orders = max(1, min(n_orders, cap))``, so the returned
    order array is exactly ``2 cap + 1`` long.  Both expected values are
    rebuilt in the test from the same geometry helpers the solver uses, so
    this is a statement about the FORMULA, not a remembered integer.

    T3-3 is why the second row exists: that copy used to cap from the
    full-union ``nU`` on the per-layer path, over-stating the capacity of the
    WINDOW grids the half-spaces actually live on, and
    ``_sem_fourier_projection`` then built a projector with more Rayleigh
    orders than the grid had nodes -- a rank-deficient ``Hsup`` and a
    build-dependent null-space draw, invisible to ``R + T``.
    """
    expected = _con_expected_caps()[which]
    orders, _R, _T, _J = _con_solve(layer_grids, 10_000)
    assert len(np.asarray(orders)) == 2 * expected + 1, (
        layer_grids, len(np.asarray(orders)), expected)


def test_g4_the_conical_per_layer_cap_is_the_window_grids_not_the_union():
    """The T3-3 direction itself, two-sided through the library's own
    fail-before switch ``PMM_CONICAL_PERLAYER_ORDER_CAP``.

    On this fixture the window grids are strictly coarser than the union, so
    the correct per-layer cap is strictly SMALLER; with the switch off the
    pre-T3-3 union formula returns and the cap grows back to the shared one.
    Both numbers are derived above, so neither side is a remembered constant.
    """
    shared, perlayer = _con_expected_caps()
    assert perlayer < shared, (perlayer, shared)   # the fixture is meaningful

    on = len(np.asarray(_con_solve("per-layer", 10_000)[0]))
    was = PCON.PMM_CONICAL_PERLAYER_ORDER_CAP
    PCON.PMM_CONICAL_PERLAYER_ORDER_CAP = False
    try:
        off = len(np.asarray(_con_solve("per-layer", 10_000)[0]))
    finally:
        PCON.PMM_CONICAL_PERLAYER_ORDER_CAP = was
    assert on == 2 * perlayer + 1, (on, perlayer)
    assert off == 2 * shared + 1, (off, shared)


def test_g4_the_conical_cap_refuses_rather_than_dropping_propagating_orders():
    """The other half of conical's own contract: when the grid cannot carry
    the propagating orders it RAISES, in its own wording (``resolves only N
    orders but M propagate``), rather than clamping and returning a far field
    with orders missing -- the same class of defect the prepared 1-D path
    carried (finding G4).

    Engineered, not hoped for: the refusal fires on ``m_prop > cap``, so the
    test drives ``m_prop`` up by shortening the wavelength into a high-index
    substrate while holding the grid -- and therefore ``cap`` -- fixed, and
    asserts the constructed premise (``m_prop > cap``) before asking for the
    raise.  The control at the fixture's own wavelength must still return, so
    this is a two-sided statement about the gate and not about the geometry.
    """
    shared, _perlayer = _con_expected_caps()
    # control: at the fixture wavelength the grid carries the orders
    orders, _R, _T, _J = _con_solve("shared", 3)
    assert len(np.asarray(orders)) == 7

    # push m_prop past cap: a high-index substrate and a short wavelength
    n_sub = 3.5
    wl_bad = _CON_P * n_sub / (shared + 3)      # m_prop ~ n_sub P / wl > cap
    m_prop = PC._n_propagating_orders(_CON_P, wl_bad, n_sub)
    assert m_prop > shared, (m_prop, shared)    # the premise is CONSTRUCTED
    with pytest.raises(ValueError) as exc:
        _con_solve("shared", 3, eps_sub=n_sub ** 2, wavelength=wl_bad)
    msg = str(exc.value)
    assert msg.startswith("probe conical: "), msg
    assert "resolves only" in msg and "propagate" in msg, msg
