"""O-11 -- the ``PMMStack`` NEAR-COINCIDENT-WALL (sliver) refusal (2026-09-11).

Two adjacent layers whose wall sets differ by ``delta`` of the period put a
SLIVER element of exactly that width on the shared union grid.  The
spectral-element operators carry the element Jacobian, so the nodal ``Kx^2``
grows as ``1/w^2``, the layer's modal spectrum acquires spurious wavenumbers
``|q| ~ 0.65 N(N+1)/4 / (k0 J)``, the interface mode-match conditions as
``1/w^2``, and past a degree-dependent onset the cascade returns a
deterministic wrong answer.

Everything asserted here is MEASURED ON THE RUNNING BUILD -- the reference is
the ``delta -> 0`` limit (the two layers with IDENTICAL walls, an exact
reference because the structure is continuous in ``delta``), the right/wrong
classification is that continuity, and the guard's separation is re-derived in
:func:`test_the_bar_has_decades_of_gap_on_both_sides_measured_here` rather than
pinned from a prior run.

Evidence: ``docs/audits/FIX_PMMSTACK_SLIVER_WALLS_2026_09_11.md`` and
``validation/probe_pmmstack_sliver/``.
"""
import os

# The sliver fixture is a near-degenerate eigenproblem: the classification the
# tests read must not move with the BLAS reduction order, so pin one thread
# before numpy is imported (the pattern of test_v5_13_0_pmm_tapered).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumenairy.elements.pmm import PMMStack  # noqa: E402
from lumenairy.elements.pmm import stack as ps  # noqa: E402
from lumenairy.elements.pmm._core import _pmm_union_grid  # noqa: E402

# ---- the O-11 fixture, verbatim from validation/probe_pmm2d_staggered_mortar
# /f5f_attrib.py: two slices of a taper, the second's walls opened by ``delta``.
_P = 1.2e-6
_WL = 0.85e-6
_THETA = 0.15
_DZ = 0.32e-6 / 4
_EH, _EP = 2.25, 9.0
_A0, _B0 = 0.27865, 0.62505


def _frames(delta):
    return [(_A0, _B0), (_A0 - delta, _B0 + delta)]


def _solve(delta, degree=14, *, guard=True, min_feature=None, per_layer=False):
    """One stack solve.  ``guard`` toggles the shipped refusal through its own
    fail-before switch, so the pre-fix arm runs the pre-fix code path."""
    kw = dict(layer_grids="per-layer") if per_layer else {}
    if min_feature is not None:
        kw["min_feature"] = min_feature
    st = PMMStack(_P, n_superstrate=1.0, n_substrate=1.0, degree=degree, **kw)
    for (a, b) in _frames(delta):
        st.add_layer(_DZ, segments=[(a, _EH), (b - a, _EP), (1.0 - b, _EH)])
    st.set_source(_WL, theta=_THETA)
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = bool(guard)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, J = st.solve()
    finally:
        ps.PMM_SLIVER_GUARD = was
    i = np.argsort(np.asarray(o).ravel())
    return np.asarray(o).ravel()[i], R[1][i], T[1][i], np.asarray(J)


def _err(a, b):
    return float(max(np.abs(a[1] - b[1]).max(), np.abs(a[2] - b[2]).max()))


def _total(res):
    return float(res[1].sum() + res[2].sum())


# The structure is CONTINUOUS in delta, so the exact answer moves linearly with
# it (the running build measures the slope in the fail-before test below).  A
# solve further than 100x that away cannot be the physical shift; one within
# 10x provably is.  The band between is neither and is excluded from both
# populations -- it is the guard's own documented residual (S3.4 of the audit).
_WRONG_FACTOR = 100.0
_RIGHT_FACTOR = 10.0


def _classify(delta, degree, ref):
    res = _solve(delta, degree, guard=False)
    e = _err(res, ref)
    kind = ("wrong" if e > _WRONG_FACTOR * delta
            else "right" if e <= _RIGHT_FACTOR * delta else "grey")
    return kind, e, _total(res), res


# ===========================================================================
# FAIL-BEFORE: what the library did, executed on the pre-fix code path
# ===========================================================================
def test_fail_before_the_pre_fix_path_returns_a_wrong_energy_violating_answer():
    """With the guard disarmed -- which restores the pre-fix selector bit for
    bit -- ``delta = 1e-4`` returns a deterministic answer that is neither the
    physical shift nor energy-conserving.  The MAGNITUDE is derived from this
    build's own continuity slope, not pinned."""
    ref = _solve(0.0, 14, guard=False)
    # the reference closes at 3.2e-14 on both builds; 1e-6 is 8 decades of
    # headroom and is a PREMISE check, not the claim.
    assert abs(_total(ref) - 1.0) < 1e-6, "the delta -> 0 reference must close"

    # the continuity slope, measured here: three deltas far above the hazard.
    slopes = [_err(_solve(d, 14, guard=False), ref) / d
              for d in (3e-3, 1e-3, 3e-4)]
    assert max(slopes) < 2.0 and min(slopes) > 0.5, slopes

    bad = _solve(1e-4, 14, guard=False)
    e = _err(bad, ref)
    # >= three decades past what continuity allows at this delta
    assert e > 1000.0 * max(slopes) * 1e-4, e
    assert e > 0.1, e
    # and it is NOT energy-invisible in the 1-D stack (O-11 said it was; that
    # reading was the 2-D mortar arm's closure, not this one's)
    assert _total(bad) - 1.0 > 1.0, _total(bad)


def test_fail_before_the_pre_fix_path_only_warns_it_does_not_refuse():
    """The pre-fix library RETURNED the wrong answer with a UserWarning, which
    is why the defect propagated into a probe that suppressed warnings."""
    st = PMMStack(_P, n_superstrate=1.0, n_substrate=1.0, degree=14)
    for (a, b) in _frames(1e-4):
        st.add_layer(_DZ, segments=[(a, _EH), (b - a, _EP), (1.0 - b, _EH)])
    st.set_source(_WL, theta=_THETA)
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = False
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            out = st.solve()
    finally:
        ps.PMM_SLIVER_GUARD = was
    assert out is not None and len(out) == 4
    msgs = [str(w.message) for w in rec]
    assert any("energy not conserved" in m for m in msgs), msgs


# ===========================================================================
# THE GUARD, TWO-SIDED
# ===========================================================================
_LADDER = [(14, d) for d in (3e-3, 1e-3, 3e-4, 1e-4, 5e-5, 3e-5, 1e-5)] + \
          [(12, d) for d in (3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)]


def test_the_guard_refuses_every_wrong_row_and_no_right_one():
    """Two-sided over the whole ladder: the guard's decision must agree with
    the CONTINUITY classification, which is measured here and needs no bar."""
    refs = {deg: _solve(0.0, deg, guard=False) for deg in (12, 14)}
    verdicts = []
    for deg, d in _LADDER:
        kind, e, tot, pre = _classify(d, deg, refs[deg])
        try:
            post = _solve(d, deg, guard=True)
            refused = False
        except ValueError as exc:
            assert "NEAR-COINCIDENT-WALL SLIVER" in str(exc), str(exc)[:200]
            post, refused = None, True
        verdicts.append((deg, d, kind, refused))
        if kind == "wrong":
            assert refused, f"degree {deg}, delta {d:g}: wrong ({e:.3g}) and NOT refused"
        elif kind == "right":
            assert not refused, f"degree {deg}, delta {d:g}: right ({e:.3g}) and refused"
            # and untouched: bit for bit the pre-fix numbers
            for a, b in zip(pre, post):
                assert np.array_equal(a, b), (deg, d)
    kinds = [v[2] for v in verdicts]
    assert kinds.count("right") >= 4 and kinds.count("wrong") >= 4, verdicts


def test_a_delta_far_outside_the_hazard_is_bit_for_bit_untouched():
    """The explicit two-sided partner: comfortably outside, the guard changes
    nothing at all -- orders, both efficiencies AND the Jones matrix."""
    for d in (3e-3, 1e-3, 3e-4):
        off = _solve(d, 14, guard=False)
        on = _solve(d, 14, guard=True)
        for a, b in zip(off, on):
            assert np.array_equal(a, b), d


def test_the_bar_has_decades_of_gap_on_both_sides_measured_here():
    """TESTING_STANDARDS rule 5, re-derived on the running build.  The guard's
    bar is ``_STACK_SUPERUNITY_BAR``; this measures the two populations it must
    separate and requires decades either side -- so the assertion survives any
    build whose separation is still real, and fails honestly on one whose is
    not."""
    refs = {deg: _solve(0.0, deg, guard=False) for deg in (12, 14)}
    right, wrong = [], []
    for deg, d in _LADDER:
        kind, _e, tot, _res = _classify(d, deg, refs[deg])
        if kind == "right":
            right.append(abs(tot - 1.0))
        elif kind == "wrong":
            wrong.append(tot - 1.0)
    assert len(right) >= 4 and len(wrong) >= 4, (right, wrong)
    bar = ps._STACK_SUPERUNITY_BAR
    assert max(right) <= bar / 100.0, (max(right), bar)      # >= 2 decades
    assert min(wrong) >= bar * 30.0, (min(wrong), bar)       # >= 1.5 decades


def test_the_refusal_names_the_geometry_and_the_two_remedies():
    with pytest.raises(ValueError) as ei:
        _solve(1e-4, 14, guard=True)
    msg = str(ei.value)
    for token in ("NEAR-COINCIDENT-WALL SLIVER", "min_feature=", "metres",
                  "PROVABLY PASSIVE", "layer_grids='per-layer'",
                  "PMM_SLIVER_GUARD", "FIX_PMMSTACK_SLIVER_WALLS"):
        assert token in msg, token
    # the prescribed min_feature must be a number the API accepts, and larger
    # than the sliver it is meant to snap
    val = float(msg.split("min_feature=")[1].split(" ")[0])
    assert val > 1e-4 * _P, (val, 1e-4 * _P)


# ===========================================================================
# THE REMEDY, scored against the exact delta -> 0 reference
# ===========================================================================
def test_the_prescribed_min_feature_lands_within_the_derived_bar():
    """The refusal prescribes a ``min_feature``; running it must return an
    answer within the geometric perturbation the snap describes.  BAR: twice
    the sliver width, against a continuity slope this build measures at ~1.15
    -- so the bar carries 1.7x headroom and no fitted constant."""
    checked = []
    for deg in (12, 14, 16):
        ref = _solve(0.0, deg, guard=False)
        for d in (1e-4, 5e-5, 3e-5, 1e-5):
            try:
                _solve(d, deg, guard=True)
                continue            # this (degree, delta) is not in the band
            except ValueError as exc:
                assert "NEAR-COINCIDENT-WALL SLIVER" in str(exc)
                mf = float(str(exc).split("min_feature=")[1].split(" ")[0])
            fixed = _solve(d, deg, guard=True, min_feature=mf)
            assert _err(fixed, ref) <= 2.0 * d, (deg, d, _err(fixed, ref))
            assert abs(_total(fixed) - 1.0) < 1e-6, (deg, d, _total(fixed))
            checked.append((deg, d))
    assert len(checked) >= 4, checked


# ===========================================================================
# CONJUNCT (a): what the geometric screen does and does NOT call a sliver
# ===========================================================================
def _segs(walls, eps=(_EH, _EP, _EH)):
    out, prev = [], 0.0
    for w, e in zip(list(walls) + [1.0], eps + (eps[-1],)):
        out.append((w - prev, e))
        prev = w
    return [s for s in out if s[0] > 0.0]


def test_an_ordinary_non_conforming_stack_is_not_a_sliver():
    """Two layers whose walls differ by a real 5% feature: the union has
    cross-layer cells, but none is finer than the geometry itself."""
    segs = [_segs([0.30, 0.50]), _segs([0.35, 0.55])]
    assert ps._cross_layer_sliver(segs, 1e-5) is None


def test_a_thin_feature_inside_ONE_layer_is_never_flagged():
    """A 1e-4-wide liner owned by a single layer is intentional geometry, and
    the ownership rule -- the same one the snap uses -- must leave it alone."""
    segs = [_segs([0.30, 0.3001, 0.70], eps=(_EH, _EP, _EH)),
            _segs([0.30, 0.70])]
    assert ps._cross_layer_sliver(segs, 1e-9) is None


def test_a_cross_layer_sliver_is_flagged_and_reports_its_own_geometry():
    segs = [_segs([0.30, 0.70]), _segs([0.3001, 0.7001])]
    hit = ps._cross_layer_sliver(segs, 1e-9)
    assert hit is not None
    w, x_l, x_r, w_wide, own, n = hit
    assert n == 2
    assert abs(w - 1e-4) < 1e-9 and abs(w_wide - 1e-4) < 1e-9
    assert abs((x_r - x_l) - w) < 1e-12
    assert abs(own - 0.2999) < 1e-9
    assert own / w >= ps._SLIVER_OWN_SCALE_RATIO


def test_the_snap_removes_the_cell_and_the_screen_then_reads_clean():
    """``_cross_layer_sliver`` screens the grid the cascade actually ran on, so
    raising ``min_feature`` past the collision must make it answer None."""
    segs = [_segs([0.30, 0.70]), _segs([0.3001, 0.7001])]
    assert ps._cross_layer_sliver(segs, 1e-9) is not None
    assert ps._cross_layer_sliver(segs, 1e-3) is None


# ===========================================================================
# CONJUNCT (b): the theorem, and where it does NOT hold
# ===========================================================================
def test_a_gain_layer_is_not_provably_passive_so_the_guard_stays_silent():
    """Super-unity is LEGAL with gain, so the guard must not speak -- even with
    the sliver present.  Negative control for the passivity conjunct."""
    st = PMMStack(_P, n_superstrate=1.0, n_substrate=1.0, degree=14)
    for (a, b) in _frames(1e-4):
        st.add_layer(_DZ, segments=[(a, _EH), (b - a, (3.0 - 0.5j) ** 2),
                                    (1.0 - b, _EH)])
    st.set_source(_WL, theta=_THETA)
    assert ps._stack_provably_passive(st) is False
    # the conjunction cannot fire on it at ANY super-unity reading ...
    assert ps._sliver_refusal(st, 5.0) is None
    # ... and the solve therefore never returns the SLIVER message (this
    # particular gain cell diverges outright, which _warn_stack_energy's
    # non-finite arm refuses on its own -- a different, correct refusal).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out = st.solve()
            assert len(out) == 4
        except ValueError as exc:
            assert "NEAR-COINCIDENT-WALL SLIVER" not in str(exc), str(exc)[:200]
            assert "non-finite total efficiency" in str(exc), str(exc)[:200]


def test_an_absorbing_superstrate_is_exempt_from_the_theorem():
    """``_lossy_incidence`` documents that the family's flux normalization
    legitimately reads above unity there, so the conjunct must reject it."""
    st = PMMStack(_P, n_superstrate=1.0 + 0.01j, n_substrate=1.0, degree=8)
    st.add_layer(_DZ, segments=[(0.5, _EH), (0.5, _EP)])
    assert ps._stack_provably_passive(st) is False
    st2 = PMMStack(_P, n_superstrate=1.0, n_substrate=1.0, degree=8)
    st2.add_layer(_DZ, segments=[(0.5, _EH), (0.5, _EP)])
    assert ps._stack_provably_passive(st2) is True


def test_a_lossy_but_passive_stack_still_satisfies_the_theorem():
    st = PMMStack(_P, n_superstrate=1.0, n_substrate=1.5, degree=8)
    st.add_layer(_DZ, segments=[(0.5, (2.0 + 0.3j) ** 2), (0.5, _EH)])
    assert ps._stack_provably_passive(st) is True


# ===========================================================================
# THE HELPER'S OWN CONTRACT
# ===========================================================================
def test_return_owners_is_additive_and_warn_false_is_silent():
    segs = [_segs([0.30, 0.70]), _segs([0.3001, 0.7001])]
    base = _pmm_union_grid(segs, 1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rich = _pmm_union_grid(segs, 1e-3, return_owners=True)
    assert len(base) == 2 and len(rich) == 3
    assert np.array_equal(np.asarray(base[0]), np.asarray(rich[0]))
    assert base[1] == rich[1]
    assert len(rich[2]) == len(rich[0]) + 1
    assert all(isinstance(o, frozenset) for o in rich[2])
    # the period ends belong to every layer
    assert rich[2][0] == frozenset({0, 1}) == rich[2][-1]
    # warn=True reports the snap; warn=False does not
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _pmm_union_grid(segs, 1e-3)
    assert any("snapped" in str(w.message) for w in rec)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _pmm_union_grid(segs, 1e-3, warn=False)
    assert not [w for w in rec if "snapped" in str(w.message)]


# ===========================================================================
# THE MECHANISM, and the caveat the refusal names
# ===========================================================================
def test_the_spurious_wavenumber_predictor_matches_the_measured_spectrum():
    """The refusal quotes ``|q| ~ 0.65 N(N+1)/4 / (k0 J)``.  Re-derive the
    constant on this build: it must be the same number across degrees, which is
    what makes it a predictor rather than a fit."""
    from lumenairy.elements.pmm._core import _build_sem_tensor_segments, _sem_modes_tensor

    def _t3(e):
        return dict(exx=complex(e), exy=0.0, eyx=0.0, eyy=complex(e),
                    ezz=complex(e))

    k0 = 2.0 * np.pi / _WL
    kx0 = np.sin(_THETA) * k0
    segs = [_segs([_A0, _B0]), _segs([_A0 - 1e-4, _B0 + 1e-4])]
    uw, leps = _pmm_union_grid(segs, 1e-9)
    J = 0.5 * float(np.min(uw)) * _P
    consts = []
    for deg in (8, 12, 14, 16, 20):
        m = _build_sem_tensor_segments(_P, uw, [_t3(e) for e in leps[0]],
                                       deg, 1, True)
        _W, _V, _lam, q = _sem_modes_tensor(m, k0, kx0, True)
        consts.append(float(np.abs(q).max()) * k0 * J
                      / (deg * (deg + 1) / 4.0))
    # MEASURED here, at this fixture's w = 1e-4, both builds 2026-09-11:
    # 0.6736 / 0.6518 / 0.6478 / 0.6453 / 0.6424, spread 1.0485.  The band is
    # +-20% of the quoted 0.65 (3.4x the observed 5.8% span) and the spread bar
    # is 2x the observed one -- it is the CONSTANCY across degree that makes
    # this a predictor, so the spread is the load-bearing half.
    assert 0.52 < min(consts) and max(consts) < 0.78, consts
    assert max(consts) / min(consts) < 1.10, consts


def test_the_wavenumber_the_message_quotes_is_the_one_the_solve_actually_has():
    """Right-conclusion-wrong-numbers is the dangerous shape, and a refusal
    message is exactly where it hides.  The ``|q| ~ ...`` the message prints
    must agree with the spectrum the layer's own eig carries."""
    from lumenairy.elements.pmm._core import (
        _build_sem_tensor_segments,
        _sem_modes_tensor,
    )

    def _t3(e):
        return dict(exx=complex(e), exy=0.0, eyx=0.0, eyy=complex(e),
                    ezz=complex(e))

    with pytest.raises(ValueError) as ei:
        _solve(1e-4, 14, guard=True)
    quoted = float(str(ei.value).split("|q| ~ ")[1].split(" ")[0])

    uw, leps = _pmm_union_grid([_segs([_A0, _B0]),
                                _segs([_A0 - 1e-4, _B0 + 1e-4])], 1e-9)
    m = _build_sem_tensor_segments(_P, uw, [_t3(e) for e in leps[1]],
                                   14, 1, True)
    k0 = 2.0 * np.pi / _WL
    _W, _V, _lam, q = _sem_modes_tensor(m, k0, np.sin(_THETA) * k0, True)
    measured = float(np.abs(q).max())
    assert 0.9 < quoted / measured < 1.1, (quoted, measured)


def test_an_unknown_wavelength_prints_the_symbol_not_a_nan():
    """``prepare()`` never requires ``set_source``, so the refusal can be
    reached with no wavelength on the stack.  It must degrade to the SYMBOL,
    not to a ``nan`` the reader would have to interpret."""
    st = PMMStack(_P, n_superstrate=1.0, n_substrate=1.0, degree=14)
    for (a, b) in _frames(1e-4):
        st.add_layer(_DZ, segments=[(a, _EH), (b - a, _EP), (1.0 - b, _EH)])
    msg = ps._sliver_refusal(st, 2.17)          # _src is None: never sourced
    assert msg is not None
    assert "|q| ~ 0.65 N(N+1)/4 / (k0 J)" in msg, msg[:400]
    # ("quasi-resonance" contains "nan", so score the slot, not the string)
    assert "|q| ~ nan" not in msg


def test_per_layer_grids_is_not_a_second_opinion_on_a_two_layer_stack():
    """The caveat the refusal states: at ``window_halfwidth = 1`` a 2-layer
    window IS the whole union, so the per-layer path rebuilds the same grid and
    returns the same answer -- it is not an escape from this defect."""
    for d in (1e-4, 3e-5):
        a = _solve(d, 14, guard=False)
        b = _solve(d, 14, guard=False, per_layer=True)
        assert _err(a, b) < 1e-12, (d, _err(a, b))
