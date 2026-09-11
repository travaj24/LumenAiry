"""Gaps found by the INDEPENDENT VERIFICATION of ROUND 2 of the 5.45.1 BOR /
EME guards (``docs/audits/VERIFY_BOR_GUARDS_ROUND2_2026_09_12.md``).

WHAT THIS FILE IS AND IS NOT.  Round 2's own claims -- the passivity
predicate's loss ladder, the two-sided energy screen, the index ceiling's
populations, the EME band's unit invariance, the ``warn_own`` attribution --
were all RE-MEASURED on this verification's own fixtures and reproduce, so none
of them is re-asserted here (``tests/unit/test_fix_bor_guards_round2.py`` and
``tests/unit/test_verify_bor_multilayer_guards.py`` already own them).  What is
here is the five places where the re-measurement found a gate that cannot fail,
a decision the physics does not support, or a margin quoted over a family the
gate does not sweep.

THREE OF THEM ARE ``xfail(strict=True)``.  They assert the behaviour the
physics requires, which the tree does not have; the strict marker fires the
moment the gap closes, which is what it is for.  That is the convention
``test_verify_bor_multilayer_guards.py`` established in round 1 and which round
2 then cleared.

PREMISE GATING, IN ONE DIRECTION ONLY.  The claim that a pathology REPRODUCES
on the running arm is gated (the CI pool is a random per-job mix of EPYC 9V74
and 7763 whose LAPACK can return a correct answer from a solve that is
ill-conditioned on every local mount).  The claim that a guard HOLDS is never
gated.

TERMS.  *Passive* -- ``Im eps >= 0`` in this library's ``exp(-i omega t)``
convention.  *Incidence half-space* -- ``layers[0]``, the medium ``R`` is
measured in.  *Channel* -- a propagating entry of the ``R`` / ``T`` arrays;
``_orient.channel_core`` drops a mode whose ``|Im qn|`` exceeds
``_BOR_CHANNEL_IMAG_BAR`` (5e-5), which is why a lossy half-space can return NO
channels at all and is the mechanism behind GAP 1.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

import numpy as np
import pytest

import lumenairy.elements.bor.bor_solve as _bs
from lumenairy.elements.bor import _orient as _or
from lumenairy.elements.bor import _sem_contract as _sc
from lumenairy.elements.bor.bor_solve import BORNodalPassivityError, build_layer, solve
from lumenairy.elements.bor.bor_stack import BORStack

pytestmark = pytest.mark.slow          # eig-heavy BOR solves


# --------------------------------------------------------------------------- #
#  fixtures -- this verification's own, not imported from either fix file
# --------------------------------------------------------------------------- #
def _uni(eps):
    return lambda r: np.full(np.shape(r), complex(eps), dtype=complex)


def _rings(Rbig, lo, hi, n=4, duty=0.5):
    per = float(Rbig) / n

    def f(r):
        frac = np.mod(np.asarray(r, float), per) / per
        return np.where(frac < duty, complex(hi), complex(lo)).astype(complex)
    return f


def _nodal_stack(im_sup, m=1, N=200, rbl=2.0, k0=2.0, basis="nodal"):
    """A damaging ring stack between two ``eps = 2`` half-spaces, with an
    imaginary part on the INCIDENCE half-space only.

    ``im_sup`` is ``Im(eps)/Re(eps)`` there; every other layer is exactly
    lossless, so the only thing the loss can change is the screen's verdict.
    """
    Rbig = float(rbl) * 2.0 * np.pi / k0
    sup = _uni(complex(2.0, 2.0 * im_sup))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [build_layer(m, Rbig, N, sup, k0, basis=basis),
                build_layer(m, Rbig, N, _rings(Rbig, 2.0, 6.0), k0,
                            basis=basis, thickness=0.35),
                build_layer(m, Rbig, N, _uni(2.0), k0, basis=basis)]


def _disarmed(layers, k0):
    """The raw answer, with the guard off -- so a decision can be scored
    against a number the screen did not produce."""
    prev = _bs.BOR_NODAL_PASSIVITY_GUARD
    _bs.BOR_NODAL_PASSIVITY_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return solve(layers, k0)
    finally:
        _bs.BOR_NODAL_PASSIVITY_GUARD = prev


def _armed(layers, k0):
    """``(verdict, n_passivity_warnings)`` as a caller receives them."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            solve(layers, k0)
            v = "returned"
        except BORNodalPassivityError:
            v = "REFUSED"
        n = len([x for x in w if "R + T" in str(x.message)])
    return v, n


# --------------------------------------------------------------------------- #
#  GAP 1 -- the gate that protects the incidence-lossless conjunct cannot fail
# --------------------------------------------------------------------------- #
def test_the_absorbing_incidence_gate_needs_a_non_empty_channel_set():
    """``test_the_screen_disarms_on_an_absorbing_incidence_medium`` is VACUOUS
    at the loss its fixture uses, and this is the live replacement.

    THE MECHANISM.  That gate builds its superstrate at ``eps = 2 + 2e-3j``,
    i.e. ``Im/Re = 1e-3``.  ``_orient.channel_core`` drops any mode whose
    ``|Im qn|`` exceeds ``_BOR_CHANNEL_IMAG_BAR`` (5e-5), and a 1e-3 relative
    loss puts EVERY mode of that half-space past it -- so ``inc`` is empty,
    ``R + T`` is empty, and ``_check_nodal_passivity`` returns at its
    ``e.size == 0`` line WITHOUT ever calling the predicate.  Measured by
    deleting the incidence conjunct from ``_stack_is_provably_passive`` and
    re-running that gate: it still passes.

    WHAT THIS GATE DOES INSTEAD.  It runs the same geometry at ``Im/Re =
    1e-7`` -- two decades INSIDE the channel gate, so the channels survive and
    the predicate is actually reached -- and asserts the DECISION on both
    sides: silent with the absorbing superstrate, refusing with a lossless one.
    Deleting the conjunct flips the first assertion.
    """
    k0 = 2.0
    lossy = _nodal_stack(1e-7, k0=k0)
    res = _disarmed(lossy, k0)
    n_ch = int(np.size(res["R"]))
    assert n_ch > 0, (
        "the premise of this gate is gone: a 1e-7 relative loss on the "
        "incidence medium emptied the channel set, so the passivity predicate "
        "is unreachable here too and the conjunct has no live gate at all")
    assert not _bs._stack_is_provably_passive(lossy), (
        "the predicate called a stack with an ABSORBING incidence medium "
        "provably passive; R and T are formed from a unit-|z-flux| basis "
        "there, so they are not power fractions and no bar can mean anything")
    v, nw = _armed(lossy, k0)
    assert (v, nw) == ("returned", 0), (
        "the screen spoke about a stack whose incidence medium absorbs "
        "(verdict %s, %d warning(s)) on %d channels" % (v, nw, n_ch))

    # the SAME geometry with a lossless superstrate -- premise-gated, because
    # whether THIS arm's nodal cascade is damaged is an arm property
    clean = _nodal_stack(0.0, k0=k0)
    raw = _disarmed(clean, k0)
    e = np.asarray(raw["energy"], float)
    worst = float(np.max(np.abs(e - 1.0))) if e.size else 0.0
    if worst <= _bs._BOR_NODAL_SUPERUNITY_BAR:
        pytest.skip("premise absent on this arm: the lossless twin of the "
                    "fixture closes to %.4e, inside the bar, so there is no "
                    "refusal for the absorbing arm to be contrasted with"
                    % (worst,))
    v2, _ = _armed(clean, k0)
    assert v2 == "REFUSED", (
        "the lossless twin returns |R+T-1| = %.4e, above the %.0e bar, and "
        "was not refused -- so the absorbing arm's silence above proves "
        "nothing about the conjunct"
        % (worst, _bs._BOR_NODAL_SUPERUNITY_BAR))


# --------------------------------------------------------------------------- #
#  GAP 2 -- D1's hole is closed on every layer EXCEPT the one it is keyed on
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason=(
    "VERIFY round 2, GAP 2: a negligible loss on the INCIDENCE half-space "
    "still disarms both detectors, which is defect D1's own shape relocated "
    "rather than removed.  Measured: Im/Re = 1e-6 on the superstrate, the "
    "div-conforming twin closing to 8.75e-07, the nodal cascade returning "
    "R + T = 2.41297 in silence."))
def test_a_negligible_loss_on_the_incidence_medium_must_not_disarm_the_screen():
    """D1 SAID: ``R + T <= 1`` is a theorem on EVERY passive stack, so a
    negligible loss must not take the guard out.  Round 2 implemented that for
    every layer except ``layers[0]``, where a loss of ANY size -- 3e-12
    included, the exact rung D1 was named for -- still disarms the energy
    screen AND the index ceiling together.

    THE BUDGET IS MEASURABLE, WHICH IS WHY DISARMING IS NOT THE ONLY OPTION.
    The reason given for keeping the conjunct is that in an absorbing incidence
    medium a mode's ``|z-flux|`` is not a conserved power, so ``R + T`` is not
    a power fraction.  True -- but the size of that effect is not unbounded and
    it is not unmeasurable: the DIV-CONFORMING twin of the same geometry
    returns the legitimate super-unity directly, and this gate reads it rather
    than assuming it.

    THE CLAIM, derived at runtime and free of any pinned number: whatever the
    twin's own excess is, a nodal excess THREE DECADES above it is not flux
    bookkeeping, and the caller must be told.  Premise-gated on the arm
    actually producing that separation.
    """
    k0, im = 2.0, 1e-6
    lossy = _nodal_stack(im, k0=k0)
    twin = _nodal_stack(im, k0=k0, basis="staggered")
    nod = _disarmed(lossy, k0)
    stg = _disarmed(twin, k0)
    en, et = (np.asarray(nod["energy"], float),
              np.asarray(stg["energy"], float))
    if not en.size or not et.size:
        pytest.skip("premise absent on this arm: the channel set is empty at "
                    "Im/Re = %g on the incidence medium" % (im,))
    nodal_excess = float(np.max(en)) - 1.0
    twin_excess = max(float(np.max(np.abs(et - 1.0))), 1e-15)
    if nodal_excess <= 1e3 * twin_excess:
        pytest.skip("premise absent on this arm: the nodal excess %.4e is "
                    "within three decades of the div-conforming twin's own "
                    "%.4e, so there is nothing here the screen is missing"
                    % (nodal_excess, twin_excess))
    v, nw = _armed(lossy, k0)
    assert (v, nw) != ("returned", 0), (
        "the legacy nodal cascade returned R + T = %.6g on a stack whose ONLY "
        "loss is Im/Re = %g on the incidence half-space -- an excess of %.4g "
        "where the div-conforming twin on the identical geometry closes to "
        "%.4e, i.e. %.2f decades of the excess is NOT flux bookkeeping -- and "
        "the guard said nothing at all"
        % (float(np.max(en)), im, nodal_excess, twin_excess,
           np.log10(nodal_excess / twin_excess)))


# --------------------------------------------------------------------------- #
#  GAP 3 -- a NON-FINITE spectrum is the SEM contract's most benign reading
# --------------------------------------------------------------------------- #
def _liner_record(position, w_frac, Rbig=24.0, N=120, degree=8, k0=2.0):
    """One layer's mesh record for a caller-prescribed liner of relative width
    ``w_frac`` at the axis, at the outer wall, or in the interior."""
    w = w_frac * Rbig
    segs = {"outer": [(Rbig - w, 2.0), (Rbig, 6.0)],
            "axis": [(w, 6.0), (Rbig, 2.0)],
            "middle": [(6.0, 6.0), (6.0 + w, 2.0), (Rbig, 2.0)]}[position]
    st = BORStack(Rbig=Rbig, m=1, N=N, n_superstrate=1.0, n_substrate=1.5,
                  basis="sem", degree=degree)
    st.add_layer(0.5, segments=segs)
    st.set_source(k0=k0)
    prev = _sc.BOR_SEM_MESH_GUARD
    _sc.BOR_SEM_MESH_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st.solve()
        recs = list(getattr(st, "_sem_mesh_report", []) or [])
    finally:
        _sc.BOR_SEM_MESH_GUARD = prev
    assert recs, "%s liner produced no mesh report" % (position,)
    return recs[0]


@pytest.mark.xfail(strict=True, reason=(
    "VERIFY round 2, GAP 3: _sem_contract.verdict reads a NON-FINITE q_excess "
    "as not hot, so the most damaged geometry on the ladder -- an axis liner "
    "at 1e-8 of Rbig, whose spectrum blows up to inf -- is the one reading "
    "'ok'.  Measured identically on the pre-round-2 tree, so it is inherited, "
    "not introduced; the round-2 gate's two rungs (1e-7, 1e-4) both miss it.  "
    "And because inf is a backward-error outcome the VERDICT moves with the "
    "kernel: ok on Haswell / Nehalem / Katmai (q_excess = inf), warn_own on "
    "Sandybridge (8.89487e+07), identically on both builds."))
def test_a_non_finite_q_excess_is_not_a_benign_sem_verdict():
    """``verdict`` computes ``hot = np.isfinite(excess) and excess >
    _BOR_Q_EXCESS``.  A spectrum that has actually blown up gives
    ``excess = inf`` (or ``nan``), which makes ``hot`` FALSE -- so the contract
    becomes SILENT exactly where the damage is worst, and a caller who
    prescribed the liner is told nothing.

    THE SAME GEOMETRY, BOTH BUILDS, 2026-09-12:

        Haswell      q_excess = inf            verdict = ok
        Nehalem      q_excess = inf            verdict = ok
        Katmai       q_excess = inf            verdict = ok
        Sandybridge  q_excess = 8.89487e+07    verdict = warn_own

    so the contract's decision moves with the BLAS kernel -- which is what
    ``docs/TESTING_STANDARDS.md`` exists to forbid.  The consequence is bounded
    (a missing ``UserWarning``, never a wrong returned number).

    Stated as a decision, with the pathology premise-gated: IF this arm drives
    the liner's spectrum non-finite, THEN the verdict must not be ``ok``.  On
    Sandybridge the premise is genuinely absent and this gate SKIPS with the
    reading rather than reporting a defect that is not there.
    """
    rec = _liner_record("axis", 1e-8)
    if np.isfinite(rec["q_excess"]):
        pytest.skip("premise absent on this arm: the axis liner's q_excess is "
                    "finite (%.4g), so the non-finite branch is not exercised"
                    % (rec["q_excess"],))
    assert _sc.verdict(rec) != "ok", (
        "a caller-prescribed liner 1e-8 of Rbig wide at the AXIS drove "
        "|q|max / (n_max k0) to %r -- NON-FINITE, i.e. past every bar there "
        "is -- and the contract said 'ok' (w_min_own_frac %.4e against the "
        "%.0e edge)"
        % (rec["q_excess"], rec["w_min_own_frac"], _sc._BOR_MIN_ELEM_FRAC))


# --------------------------------------------------------------------------- #
#  GAP 4 -- the warn_own edge is decided by the representation of the width
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason=(
    "VERIFY round 2, GAP 4: _BOR_MIN_ELEM_FRAC is a STRICT inequality against "
    "a quantity the geometry reproduces only to round-off, so the SAME "
    "nominal liner width warns at the outer wall (9.999999999917e-07) and "
    "does not at the axis (1.0e-06 exactly).  Inherited from before round 2; "
    "the round-2 gate's ladder (1e-7 and 1e-4) straddles the tie."))
def test_the_warn_own_edge_is_not_decided_by_the_representation_of_the_width():
    """``test_a_caller_prescribed_liner_is_warned_wherever_it_sits_in_r``
    asserts that the three positions behave the SAME.  Over a ladder they do
    not, and the reason is arithmetic rather than physics: the outer liner's
    two walls are ``Rbig - w`` and ``Rbig``, whose difference lands an ULP
    BELOW ``w``, while the axis liner's are ``0`` and ``w``, whose difference
    is ``w`` exactly.  At ``w / Rbig`` exactly on the edge that decides one
    way and the other.

    Neither position is wrong about the geometry; the EDGE is wrong to be a
    strict comparison at the representation limit.  Asserted as a decision --
    the verdict must not depend on which domain end the liner sits against --
    with the widths that reach the tie derived from the constant itself rather
    than pinned.
    """
    w_frac = float(_sc._BOR_MIN_ELEM_FRAC)          # the edge itself
    got = {p: _sc.verdict(_liner_record(p, w_frac))
           for p in ("outer", "axis", "middle")}
    assert len(set(got.values())) == 1, (
        "the same %.0e-of-Rbig liner gets different verdicts depending on "
        "which end of the domain it sits against: %s" % (w_frac, got))


def test_the_liner_verdict_agrees_at_all_three_positions_away_from_the_edge():
    """The part of the three-position claim that IS true, over a ladder rather
    than two rungs -- and deliberately skipping the edge itself, which GAP 4
    owns.

    UNCONDITIONAL on every arm: at a width the contract must speak about
    (3e-7 of Rbig, three-tenths of the edge) all three positions warn; at a
    width it must not (3e-6, three times the edge) none of them does.
    """
    edge = float(_sc._BOR_MIN_ELEM_FRAC)
    for frac, expect in ((0.3 * edge, "warn_own"), (3.0 * edge, "ok")):
        got = {p: _sc.verdict(_liner_record(p, frac))
               for p in ("outer", "axis", "middle")}
        assert set(got.values()) == {expect}, (
            "at %.1e of Rbig (%.2gx the %.0e edge) the three positions should "
            "all read %r; got %s" % (frac, frac / edge, edge, expect, got))


# --------------------------------------------------------------------------- #
#  GAP 5 -- the near-cutoff bar is scoped to ONE radial cutoff index
# --------------------------------------------------------------------------- #
_CUT_RBIG, _CUT_N, _CUT_NREF = 24.0, 120, 1.41
_CUT_EPS = _CUT_NREF ** 2

#: THE FAMILY BAR the shipped ``_CUTOFF_LADDER_BAR`` is not.
#:
#: ORIGIN.  ``tests/unit/test_fix_bor_multilayer_guards._gamma_of(m, idx=2)``
#: defaults the radial cutoff index to 2 and every caller there takes the
#: default, so the round-2 envelope (1.2716e-06, and the 1e-5 bar 7.86x above
#: it) is a property of ONE cutoff.  Swept over ``m`` 0..3 x ``idx`` 0..4 on
#: the SHIPPED fixture (Rbig 24, N 120, the same floor-derived 13-rung ladder;
#: Windows / Haswell / 1 thread, 2026-09-12) the worst closure is
#: **1.787437e-04**, at ``m = 1``, ``idx = 1`` -- 17.9x above the 1e-5 bar --
#: and 8 of the 20 combinations exceed 1e-5.  The gate's own index reads
#: 1.9655e-07 / 3.6267e-08 / 5.5270e-08 / 6.9146e-09 for m = 0/1/2/3.
#:
#: 2e-3 sits **11.2x (1.05 decades)** above that measured envelope, which is
#: the margin this file can defend; it is a BOUND on the family, not a
#: statement that the family is as good as the swept index.
_CUT_FAMILY_BAR = 2.0e-3


def _cut_gamma(m, idx):
    """The ``idx``-th radial cutoff of the PEC-walled cylinder at order ``m``.

    ``tests/unit/test_fix_bor_multilayer_guards._gamma_of`` takes this same
    argument and DEFAULTS it to 2, and every caller there uses the default.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        L = build_layer(m, _CUT_RBIG, _CUT_N, _uni(_CUT_EPS), 2.0)
    q = np.asarray(L["q"])
    g = np.sqrt(2.0 ** 2 * _CUT_EPS - q ** 2)
    g = np.real(g[np.abs(g.imag) < 1e-9 * np.maximum(np.abs(g.real), 1e-300)])
    g = np.sort(g[g > 1e-6])
    return float(g[idx]) if idx < g.size else None


def _cut_stack(m, k0):
    s = BORStack(_CUT_RBIG, m, n_substrate=_CUT_NREF,
                 n_superstrate=_CUT_NREF, N=_CUT_N, basis="fd")
    s.add_layer(0.4, eps=_CUT_EPS)
    s.add_layer(0.5, rings=(3.0, 0.5, 2.45, 1.41))
    s.add_layer(0.4, eps=_CUT_EPS)
    s.set_source(k0=float(k0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return s.solve()


def _cut_rungs():
    """The shipped ladder, DECIMATED to every second rung.

    The stopping rung is still derived from ``_BOR_CHANNEL_REAL_FLOOR`` the way
    ``_cutoff_ladder_rungs`` derives it, and the decimation keeps both ends --
    including the deepest rung, which is where every envelope in the table
    above is attained.  It halves the cost of this gate, which runs TWO ladders
    (the swept radial cutoff index and an un-swept one) and would otherwise put
    the file over the 60 s shard cap on the slower kernels.
    """
    floor = _or._BOR_CHANNEL_REAL_FLOOR * 10.0
    out = []
    for e_ in range(8, 40):
        dl = 10.0 ** (-e_ / 2.0)
        if _CUT_NREF * np.sqrt(dl) < floor:
            break
        out.append(dl)
    return out[::2] if len(out) >= 8 else out


def _cut_ladder(m, idx):
    """``(sorted channel counts, worst |R+T-1|)`` over the near-cutoff ladder
    approaching the ``idx``-th radial cutoff at order ``m``."""
    g = _cut_gamma(m, idx)
    if g is None:
        return None, None
    counts, worst = set(), 0.0
    for dl in _cut_rungs():
        k0 = g / (_CUT_NREF * np.sqrt(1.0 - dl))
        res = _cut_stack(m, k0)
        counts.add(int(np.size(res["R"])))
        en = np.asarray(res["energy"])
        if en.size:
            worst = max(worst, float(np.max(np.abs(en - 1.0))))
    return sorted(counts), worst


def test_the_near_cutoff_bar_is_scoped_to_one_radial_cutoff_index():
    """``_CUTOFF_LADDER_BAR = 1e-5`` is quoted as carrying 7.86x over "the
    family", where the family swept is ``m`` in {0, 1, 2}.  The dominant axis
    is not ``m``: it is WHICH radial cutoff the ladder approaches, which
    ``_gamma_of(m, idx=2)`` fixes by an undocumented default that no caller
    overrides and the round-2 restatement does not mention.

    Measured on the SHIPPED fixture, varying only ``idx`` (Windows / Haswell /
    1 thread, 2026-09-12; 13 rungs, the same floor-derived ladder):

        m    idx=2 (the gate)   idx=1
        0    1.9655e-07         5.5489e-05
        1    3.6267e-08         **1.7874e-04**
        2    5.5270e-08         2.0690e-05
        3    6.9146e-09         8.5079e-06

    so at ``m = 1`` -- inside the gate's own family -- one rung of the same
    ladder is **17.9x ABOVE the bar** and 179x above the 1e-6 the round
    replaced.  8 of the 20 ``(m, idx)`` combinations swept exceed 1e-5.

    WHAT IS ASSERTED, and nothing here is a pinned pathology.  The CHANNEL
    COUNT -- an integer, and the property the gate exists for -- is ONE number
    over the ladder at BOTH indices, so the library is not regressing at the
    index nobody swept.  And the closure at the un-swept index is bounded by
    :data:`_CUT_FAMILY_BAR`, which is the FAMILY bar the 1e-5 one is not.
    """
    m = 1
    counts_swept, worst_swept = _cut_ladder(m, 2)      # the gate's own index
    counts_other, worst_other = _cut_ladder(m, 1)      # one it does not sweep
    assert counts_swept is not None and counts_other is not None, (
        "the cylinder has no second/third radial cutoff at m=%d" % (m,))
    for idx, counts in ((2, counts_swept), (1, counts_other)):
        assert len(counts) == 1, (
            "m=%d, radial cutoff index %d: the channel count moves over the "
            "near-cutoff ladder: %s -- the orientation band's own integer "
            "invariant fails at an index the shipped gate does not sweep"
            % (m, idx, counts))
    assert worst_other < _CUT_FAMILY_BAR, (
        "m=%d: the near-cutoff closure at radial cutoff index 1 is %.4e, past "
        "the family bar %.0e (the gate's own index 2 reads %.4e on the same "
        "arm).  Either the ladder degraded or the family bar needs "
        "re-deriving -- it was measured at 1.7874e-04 over m 0..3 x index "
        "0..4 on Windows / Haswell / 1 thread, 2026-09-12"
        % (m, worst_other, _CUT_FAMILY_BAR, worst_swept))
