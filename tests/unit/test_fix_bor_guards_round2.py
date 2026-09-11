"""ROUND 2 of the 5.45.1 BOR/EME multilayer guards -- the populations and the
margins, re-measured on whatever build runs the suite.

WHAT THIS FILE IS.  ``docs/audits/FIX_BOR_GUARDS_ROUND2_2026_09_12.md`` closes
four defects the independent verification
(``docs/audits/VERIFY_BOR_MULTILAYER_GUARDS_2026_09_12.md``) left open.  The
DECISIONS those fixes make are gated in
``tests/unit/test_verify_bor_multilayer_guards.py`` (the four gates that were
strict xfails, now passing).  What is here is the other half the standards ask
for: the two-sided POPULATIONS the new bars sit between, measured on the
running arm rather than quoted from the report.

TERMS.

*Passive medium* -- one with ``Im eps >= 0`` in this library's
``exp(-i omega t)`` convention: absorbing or lossless, but not amplifying.  For
a stack of passive media, energy conservation reads ``R + T + A = 1`` with the
absorbed fraction ``A >= 0``, so ``R + T <= 1`` is a THEOREM there, lossy or
not.  On a LOSSLESS passive stack inside a PEC wall there is no absorption and
no other exit, so ``R + T = 1`` is an EQUALITY.

*Channel* -- a propagating diffraction channel of the cascade; one entry of the
``R`` / ``T`` arrays.

*Index ceiling* -- ``Re qn <= Re sqrt(eps_max)``.  ``q^2`` is an eigenvalue of
``eps k0^2 + D`` with ``D`` the transverse operator; wherever ``D`` is negative
semi-definite the Rayleigh quotient bounds ``q^2 <= max(eps) k0^2``.  A channel
above it has ``gamma^2 < 0`` -- a transverse eigenvalue the PEC-walled cylinder
does not have.  It is a CONTRADICTION, not a magnitude, which is why it is
usable as a refusal that no BLAS kernel can move.

*Set-wrong* -- a nodal row whose channel COUNT differs from the div-conforming
staggered twin's on the same geometry.  Used below as a definition of damage
that reads no energy at all, so the energy bar can be scored against it without
assuming its own conclusion.

EVERY CLAIM HERE IS EITHER AN UNCONDITIONAL INVARIANT OR PREMISE-GATED.  The CI
pool is a random per-job mix of AMD EPYC 9V74 (Zen 4) and EPYC 7763 with older
numpy/scipy wheels, on which an ill-conditioned solve can come out correct; a
claim that a PATHOLOGY reproduces is therefore gated on measuring it, and a
claim that a GUARD holds is not.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

import numpy as np
import pytest
from scipy.special import jn_zeros, jnp_zeros

import lumenairy.elements.bor.bor_solve as _bs
from lumenairy.elements.bor.bor_solve import BORNodalPassivityError, build_layer, solve

pytestmark = pytest.mark.slow          # eig-heavy BOR solves


# --------------------------------------------------------------------------- #
#  fixtures
# --------------------------------------------------------------------------- #
def _uni(val):
    return lambda r: np.full_like(r, val, dtype=complex)


def _ring(period, e_lo, e_hi, duty=0.5):
    def f(r):
        e = np.full_like(r, e_lo, dtype=complex)
        e[(r % period) < duty * period] = e_hi
        return e
    return f


def _stack(basis, family, m, N, rbl, k0=2.0, e_out=2.0, e_mid=4.0, e_hi=6.0):
    Rbig = float(rbl) * 2.0 * np.pi / k0
    prof = (_uni(e_mid + 0j) if family == "uniform"
            else _ring(0.8, e_out + 0j, e_hi + 0j))
    th = 0.4 if family == "uniform" else 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Rbig, [build_layer(m, Rbig, N, _uni(e_out + 0j), k0,
                                  basis=basis),
                      build_layer(m, Rbig, N, prof, k0, thickness=th,
                                  basis=basis),
                      build_layer(m, Rbig, N, _uni(e_out + 0j), k0,
                                  basis=basis)]


def _row(basis, family, m, N, rbl, k0=2.0, e_out=2.0):
    """One census row, with EVERY guard disarmed, so the numbers are the ones
    the solver RETURNS rather than the ones the guard allows."""
    Rbig, layers = _stack(basis, family, m, N, rbl, k0=k0, e_out=e_out)
    prev = _bs.BOR_NODAL_PASSIVITY_GUARD
    _bs.BOR_NODAL_PASSIVITY_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = solve(layers, k0)
    finally:
        _bs.BOR_NODAL_PASSIVITY_GUARD = prev
    E = np.asarray(res["R"]) + np.asarray(res["T"])
    qn = np.asarray(res["q_inc"]) / k0
    n_inc = float(np.sqrt(float(np.real(e_out))))
    return dict(basis=basis, family=family, m=m, N=N, rbl=rbl, Rbig=Rbig,
                n_channels=int(E.size),
                abs_err=float(np.max(np.abs(E - 1.0))) if E.size else 0.0,
                excess=float(np.max(E)) - 1.0 if E.size else 0.0,
                deficit=1.0 - float(np.min(E)) if E.size else 0.0,
                ceiling_excess=(float(np.max(np.real(qn))) / n_inc - 1.0
                                if qn.size else float("-inf")))


def _bessel_count(m, n_index, k0, Rbig, nz=400):
    """``(count, basin)`` for a UNIFORM PEC-walled cylinder: TE modes satisfy
    ``J_m'(gamma Rbig) = 0``, TM modes ``J_m(gamma Rbig) = 0``, and a mode
    propagates when ``gamma < n k0``.  Depends on no solver in this library.

    ``basin`` is the relative distance from the propagation cut to the nearest
    zero on either side.  Matching by BASIN rather than by a tolerance is what
    makes the oracle build-free: the count can only move if a zero CROSSES.
    """
    x = float(np.real(n_index)) * float(np.real(k0)) * float(Rbig)
    z = np.sort(np.concatenate([jnp_zeros(m, nz), jn_zeros(m, nz)]))
    n = int(np.sum(z < x))
    below = x - z[n - 1] if n > 0 else x
    above = z[n] - x if n < z.size else float("inf")
    return n, float(min(below, above) / x)


#: The census the three population gates read.  Kept deliberately small -- 2
#: bases x 2 families x 3 ``m`` x 3 cell radii = 36 solves at ``N = 120`` --
#: because the point is the SEPARATION of two populations, which the wide census
#: in ``validation/probe_fix_bor_round2/r1_passivity_census.py`` (160 solves,
#: N = 120 and 200, Rbig/lambda to 8) establishes and this re-measures on the
#: running arm.  Both edges of the 4.90-decade gap the wide census found are
#: ``N = 120`` rows, so nothing is lost by staying there and the file fits well
#: inside the 60 s shard cap.
_CENSUS_N = 120
_CENSUS = [(f, m, rbl) for f in ("uniform", "ring")
           for m in (0, 1, 2) for rbl in (0.5, 1.0, 2.0)]


_CENSUS_CACHE = []


def _census_rows():
    """The census, solved ONCE per session.  Three gates read it and each pair
    of solves is an eig of a 400x400 generalised problem; re-solving per gate
    put the file over the 60 s shard cap for no new information."""
    if _CENSUS_CACHE:
        return _CENSUS_CACHE[0]
    out = []
    for family, m, rbl in _CENSUS:
        st = _row("staggered", family, m, _CENSUS_N, rbl)
        nd = _row("nodal", family, m, _CENSUS_N, rbl)
        nd["twin_channels"] = st["n_channels"]
        nd["set_wrong"] = bool(st["n_channels"] != nd["n_channels"])
        out.append((st, nd))
    _CENSUS_CACHE.append(out)
    return out


# --------------------------------------------------------------------------- #
#  D2 -- the energy screen's two sides, and the gap it does NOT have
# --------------------------------------------------------------------------- #
def test_the_nodal_energy_bar_has_a_two_sided_gap_on_the_population_it_owns():
    """THE BAR, RE-MEASURED, and the honest statement of what it separates.

    ``_BOR_NODAL_SUPERUNITY_BAR`` does NOT separate "accurate nodal answers"
    from "damaged" ones -- the wide census measures those two populations
    OVERLAPPING by ten decades on ``|R+T-1|`` when damage is defined by the
    channel SET (rows whose set matches the twin reach 1.95e+03; rows whose set
    is wrong come as mild as 2.08e-07).  Saying so is the point of verification
    D3.

    What the bar DOES have a two-sided gap against is the population it is the
    only detector for: rows whose channel set is RIGHT and whose index ceiling
    is silent, i.e. the rows neither deterministic conjunct can see.  There the
    accurate (uniform) family and the damaged (ring) family separate cleanly.
    Measured on the wide census: 2.7141e-07 against 2.1730e-02, **4.90 decades
    with nothing in it**, with the bar at 1e-3 sitting 3.57 decades above the
    first and 1.34 decades below the second.

    UNCONDITIONAL: the accurate family stays under the bar with margin.
    PREMISE-GATED: that a damaged row exists on this arm to bound the gap from
    above.
    """
    bar = _bs._BOR_NODAL_SUPERUNITY_BAR
    clean, dirty = [], []
    for _st, nd in _census_rows():
        if nd["set_wrong"] or nd["ceiling_excess"] > 0.0:
            continue                       # a deterministic conjunct sees it
        (clean if nd["family"] == "uniform" else dirty).append(nd)
    assert len(clean) >= 6, (
        "only %d accurate-family rows survived the conjuncts -- the census "
        "cannot bound the bar from below" % (len(clean),))
    worst_clean = max(r["abs_err"] for r in clean)
    assert worst_clean < bar / 30.0, (
        "the ACCURATE nodal family reaches |R+T-1| = %.6e, only %.3gx below "
        "the %.0e refusal bar (measured 2.7141e-07, i.e. 3.57 decades).  A bar "
        "this close to the population it must not refuse is not two-sided."
        % (worst_clean, bar / worst_clean, bar))
    if not dirty:
        pytest.skip("premise absent on this arm: no ring-family row is damaged "
                    "in a way only the energy screen can see, so the gap has "
                    "no upper edge here")
    mildest_dirty = min(r["abs_err"] for r in dirty)
    assert mildest_dirty > 30.0 * worst_clean, (
        "the two populations the energy bar separates are only %.3g decades "
        "apart on this arm (accurate worst %.6e, damaged mildest %.6e); the "
        "wide census measured 4.90"
        % (np.log10(mildest_dirty / worst_clean), worst_clean, mildest_dirty))


def test_the_deficit_half_is_what_catches_the_rows_the_excess_half_misses():
    """D2, as a MEASUREMENT of what the second half adds.

    The pre-round-2 screen tested ``max(R+T) - 1`` only.  On a provably
    LOSSLESS stack ``R + T = 1`` is an equality, so a deficit is damage of the
    same kind -- and the verification found 5 of 48 lossless nodal rows
    returning ``R + T < 1`` silently, worst 0.537349.

    UNCONDITIONAL: every row whose deficit exceeds the bar is refused by the
    armed guard, whatever its excess is.
    PREMISE-GATED: that such a row exists on this arm at all.
    """
    bar = _bs._BOR_NODAL_SUPERUNITY_BAR
    only_deficit = []
    for _st, nd in _census_rows():
        if nd["excess"] <= bar < nd["deficit"]:
            only_deficit.append(nd)
    if not only_deficit:
        pytest.skip("premise absent on this arm: no lossless nodal row has a "
                    "deficit above the bar while its excess is below it")
    for nd in only_deficit:
        _Rbig, layers = _stack("nodal", nd["family"], nd["m"], nd["N"],
                               nd["rbl"])
        with pytest.raises(BORNodalPassivityError) as ei:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                solve(layers, 2.0)
        msg = str(ei.value)
        assert "DEFICIT" in msg or "ceiling" in msg or "axial index" in msg, (
            "the row (%s m=%d rbl=%.1f) has a %.4g DEFICIT and a %.4g excess "
            "-- below the bar -- and the refusal message names neither the "
            "deficit nor the ceiling:  %s"
            % (nd["family"], nd["m"], nd["rbl"], nd["deficit"], nd["excess"],
               msg[:300]))


# --------------------------------------------------------------------------- #
#  D2 -- the DETERMINISTIC conjunct: the index ceiling
# --------------------------------------------------------------------------- #
def test_the_index_ceiling_conjunct_is_two_sided_on_this_build():
    """THE CONJUNCT THAT CARRIES NO BAR, and the two populations it sits
    between -- which are on OPPOSITE SIDES OF ZERO, not a decade apart.

    ``q^2 <= max(eps) k0^2`` is a Rayleigh-quotient bound wherever the
    transverse operator is negative semi-definite.  The div-conforming
    staggered basis satisfies it structurally; the legacy nodal FD basis, which
    is not divergence-conforming, does not, and its spurious sea returns
    channels above it.

    Measured on the wide census (160 solves): staggered worst ``Re qn / n - 1``
    = -3.0273e-05, nodal-with-the-right-set worst -1.4137e-03, and the rows
    whose set is WRONG run +5.0266e-06 .. +1.2966e-03 on 40 of 44.  In absolute
    ``qn`` the mildest violation is 4.15 decades above
    ``_BOR_INDEX_CEILING_SLACK``.

    UNCONDITIONAL: no correct answer trips it -- neither the staggered basis
    nor a nodal row whose channel set matches its twin.
    PREMISE-GATED: that it FIRES on something here.
    """
    slack = _bs._BOR_INDEX_CEILING_SLACK
    stag, ok_nodal, bad_nodal = [], [], []
    for st, nd in _census_rows():
        stag.append(st)
        (bad_nodal if nd["set_wrong"] else ok_nodal).append(nd)
    assert len(stag) >= 12 and len(ok_nodal) >= 3

    worst_ok = max([r["ceiling_excess"] for r in stag + ok_nodal])
    assert worst_ok < 0.0, (
        "a correct answer sits ABOVE its own index ceiling (Re qn / n - 1 = "
        "%+.6e).  Either the Rayleigh bound is being violated by a "
        "div-conforming basis, which would be a real defect, or the conjunct "
        "has a false-positive population and must not refuse." % (worst_ok,))
    assert -worst_ok > 100.0 * slack, (
        "the closest a correct answer comes to its index ceiling is %.3e, "
        "only %.3gx the %.0e slack -- the conjunct has no room below it"
        % (-worst_ok, -worst_ok / slack, slack))

    firing = [r for r in bad_nodal if r["ceiling_excess"] > 0.0]
    if not firing:
        pytest.skip("premise absent on this arm: no nodal row returns a "
                    "channel above its own index ceiling (%d rows have a wrong "
                    "channel set but a silent ceiling)" % (len(bad_nodal),))
    mildest = min(r["ceiling_excess"] for r in firing)
    assert mildest > 100.0 * slack, (
        "the mildest ceiling violation is %.3e, only %.3gx the %.0e slack -- "
        "re-derive the slack from the measured envelope"
        % (mildest, mildest / slack, slack))


def test_the_index_ceiling_agrees_with_the_bessel_zero_oracle():
    """THE CROSS-CHECK AGAINST A REFERENCE OUTSIDE THIS LIBRARY.

    Where the ceiling conjunct fires, the claim is that the channel SET is
    wrong.  For a UNIFORM PEC-walled half-space the number of channels is a
    closed form in Bessel zeros, so that claim is checkable without any solver
    -- and this gate checks it, restricted to the rows where the oracle is
    trustworthy: uniform profile, ``m >= 1`` (the verification measured the
    oracle's own ``m = 0`` bookkeeping off by one), and a BASIN wide enough that
    no arithmetic could move the count.

    The oracle is deliberately NOT the production predicate: it exists only for
    a uniform layer, while the index ceiling is exact on every profile.  This
    gate is what licenses the one to stand for the other.

    UNCONDITIONAL: the div-conforming basis returns the closed-form count.
    PREMISE-GATED: that the nodal basis trips the ceiling on such a row here.
    """
    k0, N, e_out = 2.0, _CENSUS_N, 2.0
    checked, agreed = 0, 0
    for m in (1, 2):
        for rbl in (1.0, 2.0):
            Rbig = rbl * 2.0 * np.pi / k0
            exact, basin = _bessel_count(m, np.sqrt(e_out), k0, Rbig)
            if basin < 1e-2:
                continue                   # too near a cutoff to pin a count
            st = _row("staggered", "uniform", m, N, rbl)
            nd = _row("nodal", "uniform", m, N, rbl)
            assert st["n_channels"] == exact, (
                "m=%d Rbig/lambda=%.1f: the div-conforming basis returned %d "
                "channels against the closed-form %d (basin %.3g) -- the "
                "oracle and the reference disagree, so neither can score the "
                "nodal basis here"
                % (m, rbl, st["n_channels"], exact, basin))
            if nd["ceiling_excess"] <= 0.0:
                continue
            checked += 1
            if nd["n_channels"] != exact:
                agreed += 1
    if checked == 0:
        pytest.skip("premise absent on this arm: the index ceiling fires on no "
                    "uniform wide-basin row at m >= 1, so there is nothing for "
                    "the closed form to corroborate")
    assert agreed == checked, (
        "the index ceiling fired on %d row(s) and the closed-form Bessel-zero "
        "count agreed that the channel set was wrong on only %d of them"
        % (checked, agreed))


def _lossy_incidence_stack(im_rel, m=1, N=200, rbl=2.0, k0=2.0, basis="nodal",
                           e_out=2.0):
    """The damaging ring stack of ``_stack``, with a RELATIVE loss
    ``Im(eps)/Re(eps) = im_rel`` on the INCIDENCE half-space ALONE.  Every
    other layer stays exactly lossless, so the only thing the loss can change
    is the screen's verdict."""
    Rbig = float(rbl) * 2.0 * np.pi / k0
    sup = complex(e_out, e_out * float(im_rel))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return k0, [build_layer(m, Rbig, N, _uni(sup), k0, basis=basis),
                    build_layer(m, Rbig, N, _ring(0.8, e_out + 0j, 6.0 + 0j),
                                k0, thickness=0.5, basis=basis),
                    build_layer(m, Rbig, N, _uni(e_out + 0j), k0,
                                basis=basis)]


def _verdict(layers, k0):
    """``(verdict, which detector, n energy warnings)`` as a caller sees it.

    The two detectors are told apart by their own messages: the ENERGY one
    quotes ``max(R + T)``, the INDEX CEILING one quotes an ``axial index``.
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            solve(layers, k0)
            v, det = "returned", ""
        except BORNodalPassivityError as exc:
            v = "REFUSED"
            det = "ceiling" if "axial index" in str(exc) else "energy"
        nw = len([x for x in w if "R + T" in str(x.message)])
    return v, det, nw


def test_the_energy_screen_disarms_on_an_absorbing_incidence_medium_but_the_ceiling_does_not():
    """THE REQUIREMENT ROUND 2 KEPT, THE SCOPE ROUND 3 GAVE IT, and a gate that
    can actually fail.

    WHAT THE REQUIREMENT IS.  ``R`` and ``T`` are formed from a basis
    normalised to unit ``|z-flux|`` per mode.  In an ABSORBING incidence medium
    a mode's flux is not a conserved power, so the sums are not power fractions
    and there is no theorem for an ENERGY bar to enforce -- the 1-D peer
    ``pmm/stack._stack_provably_passive`` excludes an absorbing superstrate for
    exactly this reason and records the 1.00026 / 1.0152 / 1.0303 super-unity
    ladder it legitimately produces.

    WHAT IT IS NOT.  That reasoning is about the ENERGY and says nothing about
    the INDEX CEILING, whose Rayleigh argument concerns one half-space's own
    ``eps``.  Until round 3 the two detectors shared one early return, so a
    loss of any size on ``layers[0]`` disarmed BOTH -- verification round 2's
    GAP 2, which is defect D1's own shape relocated rather than removed.
    Round 3 splits them: :func:`bor_solve._stack_media_are_passive` (GAIN
    disarms everything) gates both, and the incidence-lossless conjunct now
    gates the energy detector alone.

    WHY THE GATE THIS REPLACES COULD NOT FAIL.  It built its superstrate at
    ``eps = 2 + 2e-3j``, i.e. ``Im/Re = 1e-3``.  ``_orient.channel_core`` drops
    any mode whose ``|Im qn|`` exceeds ``_BOR_CHANNEL_IMAG_BAR`` (5e-5), and a
    1e-3 relative loss puts EVERY mode of that half-space past it -- so the
    incidence set was EMPTY, ``R + T`` was empty, and
    ``_check_nodal_passivity`` returned at its ``e.size == 0`` line WITHOUT
    ever calling the predicate.  Measured by the round-2 verification by
    deleting the incidence conjunct and re-running that gate: it still passed.
    Measured channel counts on its fixture: ``Im(eps_sup)`` = 2e-3 -> 0
    channels; 2e-5 .. 2e-11 -> 7-8 channels.  This gate runs at ``Im/Re =
    1e-7``, two decades INSIDE the channel gate, and asserts its own premise.

    UNCONDITIONAL on every arm, all four clauses.
    """
    k0, lossy = _lossy_incidence_stack(1e-7)
    prev = _bs.BOR_NODAL_PASSIVITY_GUARD
    _bs.BOR_NODAL_PASSIVITY_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = solve(lossy, k0)
    finally:
        _bs.BOR_NODAL_PASSIVITY_GUARD = prev
    n_ch = int(np.size(raw["R"]))
    assert n_ch > 0, (
        "the premise of this gate is gone: a 1e-7 relative loss on the "
        "incidence medium emptied the channel set, so the predicate is "
        "unreachable here too and the conjunct would have no live gate at all")

    # (a) the ENERGY theorem is off -- the predicate says so ...
    assert not _bs._stack_is_provably_passive(lossy), (
        "the predicate called a stack with an ABSORBING incidence medium "
        "provably passive; R and T come from a unit-|z-flux| basis there, so "
        "they are not power fractions and no energy bar can mean anything")
    # (b) ... while the stack's MEDIA are still passive, which is what keeps
    #     the ceiling armed.  Deleting the incidence conjunct collapses (a)
    #     into (b) and flips this gate.
    assert _bs._stack_media_are_passive(lossy), (
        "Im eps >= 0 on every layer of this stack, so the Rayleigh bound the "
        "index ceiling reads is intact; the media predicate denied it")

    # (c) the caller never receives an ENERGY verdict on this stack
    v, det, nw = _verdict(lossy, k0)
    assert nw == 0 and det != "energy", (
        "the ENERGY screen spoke about a stack whose incidence medium absorbs "
        "(verdict %s via %s, %d warning(s)) on %d channels"
        % (v, det or "-", nw, n_ch))

    # (d) and the index ceiling is still ARMED there.  Premise-gated on this
    #     arm's nodal cascade actually returning a channel above the ceiling --
    #     whether it does is an arm property, whether the guard then speaks is
    #     not.
    exc = _bs._channel_index_excess(
        lossy[-1], np.asarray(lossy[-1]["q"])[raw["out"]] / k0)
    if exc is None or exc <= _bs._BOR_INDEX_CEILING_SLACK:
        pytest.skip("premise absent on this arm: the exit half-space's worst "
                    "returned channel sits %s the index ceiling, so there is "
                    "no contradiction for the conjunct to refuse"
                    % ("at" if exc is None else "%.4e below" % (-exc,)))
    assert (v, det) == ("REFUSED", "ceiling"), (
        "the exit half-space returned a channel %.4e ABOVE its own index "
        "ceiling -- gamma^2 < 0, which this PEC-walled cylinder does not have "
        "-- and the guard answered %r (%s).  The incidence medium's loss must "
        "not reach this detector: its Rayleigh bound is about the exit "
        "half-space's own eps" % (exc, v, det or "no detector"))


# --------------------------------------------------------------------------- #
#  the SEM warn edge -- what ORDINARY geometry actually reaches
# --------------------------------------------------------------------------- #
def _graded_hp_pair(eps_seg, degree=4, N=40, Rbig=24.0, k0=2.0):
    """An ORDINARY two-layer ring pair -- real features, no coincidences, no
    sliver -- with hp refinement and GRADING on.

    ``degree`` and ``N`` are the CHEAPEST that build the mesh, deliberately:
    ``w_min_union_frac`` is mesh arithmetic and reads 1.003182e-04 identically
    at degree 3 / 4 / 6 / 8 / 12 and at N = 40 / 60 / 80 / 120 (measured), so
    nothing is lost by keeping the modal eigensolve small.  At
    ``elements_per_segment = 32`` the mesh is 64 elements per segment pair and
    the solve is the whole cost of this gate."""
    from lumenairy.elements.bor.bor_stack import BORStack
    s = BORStack(Rbig=Rbig, m=1, N=N, n_superstrate=1.0, n_substrate=1.0,
                 basis="sem", degree=degree,
                 elements_per_segment=eps_seg, grade=True)
    s.add_layer(0.5, segments=[(6.0, 4.0), (Rbig, 2.0)])
    s.add_layer(0.5, segments=[(7.0, 2.0), (Rbig, 4.0)])
    s.set_source(k0=k0)
    return s


def _mesh_report(st):
    """The stack's OWN mesh records, with the contract disarmed -- the numbers
    the guard reads, not a re-implementation of them."""
    from lumenairy.elements.bor import _sem_contract as C
    prev = C.BOR_SEM_MESH_GUARD
    C.BOR_SEM_MESH_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st.solve()
        return list(getattr(st, "_sem_mesh_report", []) or [])
    finally:
        C.BOR_SEM_MESH_GUARD = prev


def test_the_sem_warn_edge_s_binding_ordinary_margin_is_graded_hp_refinement():
    """WHAT ORDINARY GEOMETRY THE WARN EDGE ACTUALLY REACHES, and it is not the
    family the 5.45.1 census measured.

    ``_BOR_SLIVER_BAND_FRAC = 1e-4`` was fixed from a census whose binding
    ordinary geometry was a 256-slice taper staircase at 9.766e-04, quoted as a
    9.77x margin.  Round 2 measures a family that census did not sweep: hp
    refinement with GRADING on, on an ordinary two-layer ring pair.

    ``BORStack(elements_per_segment=k, grade=True)`` splits every segment
    interval into ``k`` Chebyshev-Lobatto graded sub-elements, so the narrowest
    sub-cell shrinks as ``1/k^2``; and because the ``+-1`` enrichment window
    makes the interval between two DIFFERENT layers' walls appear in both
    meshes, every sub-cell inside it is attributed to the union.  At ``k = 32``
    the narrowest ordinary cross-layer cell is **1.0032e-04 of Rbig** -- a
    **1.003x** margin, not 9.77x -- and a ``k = 64`` sweep lands inside the band
    and warns.

    THE ASSERTIONS ARE DECISIONS, AND THE QUANTITY IS KERNEL-EXACT.
    ``w_min_union_frac`` is mesh arithmetic -- it reads the same on every arm,
    at every ``N`` and every degree -- so this gate does not need a margin: it
    asserts the ORDER (the ladder is monotone in ``k``), the VERDICT (ordinary
    geometry is not warned at ``k = 32``, which is the contract), and the
    1/k^2 LAW that says where the next rung lands.  Nothing here pins a
    magnitude the library is entitled to move.
    """
    from lumenairy.elements.bor import _sem_contract as C
    edge = C._BOR_SLIVER_BAND_FRAC
    got = {}
    for k in (8, 32):
        recs = _mesh_report(_graded_hp_pair(k))
        assert recs, "the hp pair at k=%d produced no mesh report" % (k,)
        got[k] = dict(
            union=min(r["w_min_union_frac"] for r in recs),
            verdicts=sorted({C.verdict(r) for r in recs}))
        assert np.isfinite(got[k]["union"]), (
            "graded hp at elements_per_segment=%d produced NO cross-layer cell "
            "at all; this gate's premise is that the +-1 window attributes the "
            "graded sub-cells to the union, and it does not hold here" % (k,))

    # THE CONTRACT: ordinary geometry is not spoken about.
    assert got[32]["verdicts"] == ["ok"], (
        "ORDINARY graded hp refinement at elements_per_segment=32 was "
        "%s (narrowest cross-layer cell %.4e of Rbig against the %.0e edge)"
        % (got[32]["verdicts"], got[32]["union"], edge))

    # THE LAW: 1/k^2, which is what says the NEXT rung lands inside the band.
    ratio = got[8]["union"] / got[32]["union"]
    assert 12.0 < ratio < 20.0, (
        "the graded sub-cell should shrink as 1/k^2 -- 16x from k=8 to k=32 -- "
        "but it moved %.3gx (%.4e -> %.4e).  If the grading recipe changed, "
        "_BOR_SLIVER_BAND_FRAC's ordinary margin has to be re-measured with it"
        % (ratio, got[8]["union"], got[32]["union"]))

    # AND THE MARGIN, RECORDED RATHER THAN BARRED, because there is none to bar:
    # measured 1.0032e-04 against a 1.0e-04 edge.  What IS asserted is that the
    # constant's own comment has not drifted away from the measurement.
    margin = got[32]["union"] / edge
    assert 1.0 <= margin < 1.5, (
        "the binding ORDINARY margin on the warn edge reads %.4gx (cell "
        "%.4e of Rbig, edge %.0e).  _BOR_SLIVER_BAND_FRAC's comment records "
        "1.003x; if this has moved, the comment and "
        "docs/audits/FIX_BOR_GUARDS_ROUND2_2026_09_12.md have to move with it"
        % (margin, got[32]["union"], edge))


def test_a_caller_prescribed_liner_is_warned_wherever_it_sits_in_r():
    """THE COVERAGE D4'S FIX HAD TO KEEP, and nearly did not -- now stated over
    a LADDER, which is where round 2's version of it was false.

    ``warn_own`` fires on ``w_min_own_frac`` -- the narrowest cell BOTH of
    whose enclosing boundaries THIS layer asked for -- instead of on
    ``w_min_frac``, so a liner's NEIGHBOUR stops being blamed for a cell it
    never prescribed.  But a layer's segment list holds only its INTERIOR walls
    (``bor_stack``: ``rs for rs, _t in segs if rs < Rbig``), because every
    layer's list ends at ``Rbig`` and starts at the axis.  Reading ownership
    from interior walls alone therefore leaves a liner hard against EITHER
    DOMAIN END with no owner at all -- and the message that exists to tell that
    caller would never fire.  Measured while building the fix: a
    ``1e-6``-of-``Rbig`` liner at the outer wall and at the axis drove
    ``|q|max / (n_max k0)`` to **2.521e+05** and **8.895e+05** -- 1.4 and 1.9
    decades past the spectral screen -- and read ``ok``.

    ROUND 3 -- WHAT ROUND 2 CLAIMED HERE AND DID NOT HAVE.  Round 2 asserted
    two rungs (1e-7 and 1e-4) and its report generalised them to *"the three
    positions behave identically over a width ladder"*.  The verification
    refuted that: over a ladder the three positions disagreed at **4 of 9
    rungs**, for TWO reasons, both now fixed in ``_sem_contract``:

      * at ``1e-9``, ``3e-9`` and ``1e-8`` of ``Rbig`` the AXIS liner drives
        ``q_excess`` NON-FINITE, and ``verdict`` read a non-finite ratio as NOT
        hot -- so the axis read ``ok`` while the interior and the outer wall
        read ``warn_own``.  Worse, ``inf`` is a backward-error outcome, so on
        Sandybridge the same geometry gave 8.89487e+07 and read ``warn_own``:
        a KERNEL-DEPENDENT verdict (verification GAP 3);
      * at the edge itself the axis liner's walls (``0``, ``w``) difference to
        ``1.000000000000e-06`` EXACTLY while the other two lose 38,968 ULP to
        the subtraction, and a STRICT ``<`` decided them oppositely
        (verification GAP 4).

    SO THE CLAIM IS NOW THE LADDER ONE, and it is a DECISION rather than a
    magnitude: over widths derived from ``_BOR_MIN_ELEM_FRAC`` itself -- three
    decades below it, the edge, and two decades above -- the three positions
    must return the SAME verdict at every rung, and that verdict must be
    ``warn_own`` where the contract has to speak and ``ok`` where it must not.
    Re-measured after the fix: 0 of the rungs disagree, on every arm
    (``validation/probe_fix_bor_round3/g3_sem_ladder.py``).
    """
    import warnings as _w

    from lumenairy.elements.bor import _sem_contract as C
    from lumenairy.elements.bor.bor_stack import BORStack

    Rbig = 24.0

    def verdicts(w_frac):
        w = w_frac * Rbig
        out = {}
        for label, segs in (
                ("outer", [(Rbig - w, 2.0), (Rbig, 6.0)]),
                ("axis", [(w, 6.0), (Rbig, 2.0)]),
                ("middle", [(6.0, 6.0), (6.0 + w, 2.0), (Rbig, 2.0)])):
            st = BORStack(Rbig=Rbig, m=1, N=120, n_superstrate=1.0,
                          n_substrate=1.5, basis="sem", degree=8)
            st.add_layer(0.5, segments=segs)
            st.set_source(k0=2.0)
            prev = C.BOR_SEM_MESH_GUARD
            C.BOR_SEM_MESH_GUARD = False
            try:
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    st.solve()
                recs = list(getattr(st, "_sem_mesh_report", []) or [])
            finally:
                C.BOR_SEM_MESH_GUARD = prev
            assert recs, "%s liner produced no mesh report" % (label,)
            out[label] = dict(own=min(r["w_min_own_frac"] for r in recs),
                              union=min(r["w_min_union_frac"] for r in recs),
                              q=max(r["q_excess"] for r in recs),
                              finite=all(np.isfinite(r["q_excess"])
                                         for r in recs),
                              verdicts=sorted({C.verdict(r) for r in recs}))
        return out

    edge = float(C._BOR_MIN_ELEM_FRAC)
    #: the ladder, derived from the edge rather than written down: three
    #: decades below it (where GAP 3's non-finite spectra live), the edge
    #: itself (GAP 4's tie), and two decades above (where the contract must be
    #: silent).
    ladder = ((1e-3 * edge, "warn_own"), (1e-2 * edge, "warn_own"),
              (1e-1 * edge, "warn_own"), (edge, "warn_own"),
              (3.0 * edge, "ok"), (100.0 * edge, "ok"))
    for w_frac, expect in ladder:
        got = verdicts(w_frac)
        # (a) the three positions AGREE -- the ladder claim round 2 did not have
        seen = {label: g["verdicts"] for label, g in got.items()}
        assert len({tuple(v) for v in seen.values()}) == 1, (
            "the same %.3e-of-Rbig liner (%.4gx the %.0e edge) gets different "
            "verdicts depending on which end of the domain it sits against: "
            "%s.  Widths measured: %s; q_excess: %s"
            % (w_frac, w_frac / edge, edge, seen,
               {k: "%.12e" % g["own"] for k, g in got.items()},
               {k: "%.4g" % g["q"] for k, g in got.items()}))
        # (b) and it is the RIGHT verdict
        for label, g in got.items():
            assert g["verdicts"] == [expect], (
                "a %s liner %.3e of Rbig wide (%.4gx the edge) drove "
                "|q|max/(n_max k0) to %.4g and the contract said %s, not %r"
                % (label, w_frac, w_frac / edge, g["q"], g["verdicts"],
                   expect))
            # (c) the caller ASKED for this cell, so it is never the union's
            assert not np.isfinite(g["union"]), (
                "the %s liner at %.3e of Rbig was attributed to the UNION "
                "(%.4e); the library did not manufacture it, the caller asked "
                "for it, and it must never be refusable"
                % (label, w_frac, g["union"]))
        # (d) a liner the contract must speak about has an OWNER at every
        #     position -- the domain-end half of D4
        if expect == "warn_own":
            for label, g in got.items():
                assert np.isfinite(g["own"]), (
                    "the %s liner at %.3e of Rbig has NO owner "
                    "(w_min_own_frac = inf): a cell the caller prescribed "
                    "against a domain end must still be attributed to the "
                    "layer that prescribed it" % (label, w_frac))


# --------------------------------------------------------------------------- #
#  D13 -- the EME branch band, through every site that reads it
# --------------------------------------------------------------------------- #
def _eme_strip(scale, im_eps, Nx=64, wl_um=1.55):
    from lumenairy.elements.eme import eme_2d as e2
    Lx = 1.0 * scale
    k0 = 2.0 * np.pi / (wl_um * scale)
    eps = np.full(Nx, 1.0 + 0j)
    eps[Nx // 3:2 * Nx // 3] = 12.0 + im_eps
    lam = np.asarray(e2.strip_x_modes(eps, Lx, Nx, k0)[0], complex)
    fwd = np.asarray(e2._ky_forward(lam, 0.0, k0)) * scale
    return fwd[np.lexsort((np.imag(fwd), np.real(fwd)))], k0


@pytest.mark.parametrize("im_eps", [-1e-6j, -1e-7j, 0.0, +1e-6j])
def test_the_eme_scalar_branch_is_unit_invariant(im_eps):
    """D13.  The same 1 um cell at lambda = 1550 nm, stated in micrometres,
    nanometres and metres.  ``ky`` scales as 1/length, so ``ky * scale`` is the
    unit-free observable; the spectra are SORTED before differencing because
    the eigensolver's own ordering is not stable between two solves.

    Pre-round-2 the literal 1.0 floor made the nanometre band 5.2x wider in
    physical terms and put 3 of 96 roots on a different branch.
    """
    base, _k0 = _eme_strip(1.0, im_eps)
    for name, s in (("nanometres", 1e3), ("metres", 1e-6)):
        other, _ = _eme_strip(s, im_eps)
        assert other.size == base.size
        tol = 1e-6 * max(float(np.max(np.abs(base))), 1.0)
        bad = int((np.abs(base - other) > tol).sum())
        assert bad == 0, (
            "Im(eps) = %s: %d of %d forward roots differ between micrometres "
            "and %s; worst |d ky * scale| = %.6g"
            % (im_eps, bad, base.size, name,
               float(np.max(np.abs(base - other)))))


def test_the_eme_diffraction_driver_is_unit_invariant():
    """D13 through ``eme_diffraction.mode_match``, whose ``qz2`` is CALLER
    supplied -- the production surface where the floor actually decides.

    Measured pre-round-2 on this slab: ``T00`` = 0.738986606108 in micrometres
    against 0.738986551923 in nanometres, where the metre arm agreed to 1e-15.
    """
    from lumenairy.elements.eme import eme_diffraction as ed
    orders = [(0, 0), (1, 0), (0, 1), (-1, 0)]
    got = {}
    for name, s in (("um", 1.0), ("nm", 1e3), ("m", 1e-6)):
        Lx = Ly = 1.0 * s
        Nx = Ny = 16
        k0 = 2.0 * np.pi / (1.55 * s)
        rng = np.random.default_rng(20260912)
        Psi = (rng.standard_normal((Nx * Ny, 6))
               + 1j * rng.standard_normal((Nx * Ny, 6)))
        qz2 = (np.array([30.0, 12.0, 3.0, -8.0, -40.0, 5.0], complex)
               - 1e-6j) / s ** 2
        res = ed.mode_match(qz2, Psi, orders, kx0=0.0, ky0=0.0, k0=k0,
                            eps_sup=1.0, eps_sub=1.0, depth=0.4 * s,
                            Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, inc_order=(0, 0))
        got[name] = float(np.real(res["T"][res["orders"].index((0, 0))]))
    for name in ("nm", "m"):
        assert got[name] == pytest.approx(got["um"], abs=1e-12), (
            "mode_match returned T00 = %.12f in %s against %.12f in um on the "
            "SAME slab" % (got[name], name, got["um"]))


def test_every_eme_site_that_reads_the_band_passes_k0():
    """The floor is only unit-safe where ``k0`` reaches it.  This pins that the
    three production sites pass it, by CALLING them and checking the band they
    produce is the ``k0``-floored one -- not by grepping, which a renamed
    keyword would walk through.
    """
    from lumenairy.elements.eme._branch import _EME_CUT_BAND_REL, cut_band
    z = np.array([1e-3 + 0j, -2e-3 + 1e-9j])
    k0 = 6.5
    assert float(cut_band(z, k0=k0, xp=np)) == pytest.approx(
        _EME_CUT_BAND_REL * k0), "cut_band is not flooring at |k0|"
    assert float(cut_band(z, xp=np)) == pytest.approx(
        _EME_CUT_BAND_REL * 2e-3), (
        "with no k0 the scale must be the spectrum's own top, not a literal")

    # the vector site, which takes its k0 from strip_vector_modes
    from lumenairy.elements.eme import eme_2d_vector as ev
    for s in (1.0, 1e3):
        Lx, Nx = 1.0 * s, 24
        k0s = 2.0 * np.pi / (1.55 * s)
        eps = np.full(Nx, 1.0 + 0j)
        eps[Nx // 3:2 * Nx // 3] = 12.0 - 1e-6j
        kys = np.asarray(ev.strip_vector_modes(eps, Lx, Nx, k0s)[0])
        fwd = sorted(int(i) for i in ev._strip_split_forward(kys, k0s))
        if s == 1.0:
            ref = fwd
        else:
            assert fwd == ref, (
                "the vector forward SET moved between unit systems: %s vs %s"
                % (fwd, ref))
