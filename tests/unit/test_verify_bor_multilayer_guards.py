"""Gaps found by the INDEPENDENT VERIFICATION of the 5.45.1 BOR multilayer
guards (``docs/audits/VERIFY_BOR_MULTILAYER_GUARDS_2026_09_12.md``).

WHAT THIS FILE IS AND IS NOT.  It is not a second copy of
``test_fix_bor_multilayer_guards.py``: every claim that file makes was
re-measured during the verification and is not re-asserted here.  What is here
is the places where the re-measurement found the shipped guards deciding
something the physics does not support, plus one bound on the SEM warn edge
that the shipped census does not carry.  The POPULATIONS and MARGINS the round-2
fixes sit between are a separate file again --
``tests/unit/test_fix_bor_guards_round2.py`` -- so this one stays a statement
about DECISIONS.

FOUR OF THEM WERE ``xfail(strict=True)`` AND ARE NOW PASSING GATES.  They
asserted the behaviour the physics requires, which the tree did not have; the
strict marker fired the moment the gap closed, which is what it was for.  ROUND
2 (``docs/audits/FIX_BOR_GUARDS_ROUND2_2026_09_12.md``) closed all four -- GAP 1
and GAP 2 by replacing the nodal screen's predicate and making it two-sided,
GAP 5 by giving ``_sem_contract`` the quantity its ``warn_own`` message
describes, GAP 6 by flooring the EME branch band at ``k0`` -- so the markers are
gone and the gates assert the CLOSED behaviour.

EVERY GATE HERE IS PREMISE-GATED IN ONE DIRECTION ONLY.  The claim that a
pathology REPRODUCES on the running arm is gated (the 5.45.0 CI lesson -- the
release runner's LAPACK returns correct answers from solves that are
ill-conditioned on every local mount, and the CI pool is a random per-job mix of
EPYC 9V74 and 7763 with older numpy/scipy wheels).  The claim that the GUARD
holds is NOT gated: on every arm, a solve that RETURNS must return a physical
answer, and that is asserted unconditionally.

TERMS USED BELOW.

*Passive medium* -- one with ``Im eps >= 0`` in this library's
``exp(-i omega t)`` convention, i.e. absorbing or lossless but not amplifying.
For a stack of passive media energy conservation reads ``R + T + A = 1`` with
the absorbed fraction ``A >= 0``, so ``R + T <= 1`` is a THEOREM there --
whether or not the stack is lossless.  On a LOSSLESS passive stack closed by a
PEC wall there is no other exit and no absorption, so ``R + T = 1`` is an
EQUALITY.

*Channel* -- a propagating diffraction channel of the cascade; one entry of the
``R`` / ``T`` arrays.  For a UNIFORM PEC-walled half-space the number of them
is a closed form (below), which is what makes it usable as an oracle.
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
from lumenairy.elements.bor.bor_stack import BORStack

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


def _nodal_stack(e_hi, Rbig=4.0, N=200, k0=2.0, m=1, e_out=2.0 + 0j):
    """The stack the shipped gate ``test_structured_stack_energy_floor_nodal``
    built, with the ring's HIGH permittivity left free so an infinitesimal
    imaginary part can be put on it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [build_layer(m, Rbig, N, _uni(e_out), k0, basis="nodal"),
                build_layer(m, Rbig, N, _ring(0.8, e_out, e_hi), k0,
                            thickness=0.5, basis="nodal"),
                build_layer(m, Rbig, N, _uni(e_out), k0, basis="nodal")]


def _uniform_stack(Rbig, N, k0, m, e_out=2.0 + 0j, e_mid=4.0 + 0j,
                   basis="nodal", thickness=0.4):
    """Uniform half-spaces around a uniform middle layer -- the family the
    5.45.1 census calls accurate, used here as the NEVER-REFUSE control."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [build_layer(m, Rbig, N, _uni(e_out), k0, basis=basis),
                build_layer(m, Rbig, N, _uni(e_mid), k0, thickness=thickness,
                            basis=basis),
                build_layer(m, Rbig, N, _uni(e_out), k0, basis=basis)]


def _solve_disarmed(layers, k0):
    prev = _bs.BOR_NODAL_PASSIVITY_GUARD
    _bs.BOR_NODAL_PASSIVITY_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return solve(layers, k0)
    finally:
        _bs.BOR_NODAL_PASSIVITY_GUARD = prev


# --------------------------------------------------------------------------- #
#  GAP 1 -- a negligible loss disarms the passivity screen entirely
# --------------------------------------------------------------------------- #
#: The loss ladder GAP 1 walks.  ``Im(eps) / Re(eps)`` on the ring's high
#: region, from below the pre-round-2 disarm threshold (1e-12) to a genuinely
#: absorbing medium.  The physical absorption over this whole ladder, measured
#: on the div-conforming staggered twin, is 1.2e-12 .. 1.4e-01; the NODAL
#: violation is the same 2.9 % excess / 3.8 % deficit on every rung of it.
_LOSS_LADDER = (0.0, 1e-14, 1e-13, 1e-12, 3e-12, 1e-11, 1e-9, 1e-8, 1e-6,
                1e-4, 1e-3, 1e-2)


def _decide(layers, k0):
    """``('refused', msg)`` or ``('returned', R+T array, n_warnings)``."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            res = solve(layers, k0)
        except BORNodalPassivityError as exc:
            return ("refused", str(exc), 0)
    E = np.asarray(res["R"]) + np.asarray(res["T"])
    return ("returned", E, len(w))


def test_a_negligible_loss_must_not_defeat_the_nodal_passivity_screen():
    """GAP 1 (CLOSED in round 2).  ``R + T <= 1`` is a theorem on EVERY PASSIVE
    stack -- ``R + T + A = 1`` with the absorbed fraction ``A >= 0`` -- not only
    on lossless ones, so a loss cannot be a reason to stop screening for
    super-unity.

    The pre-round-2 predicate required every layer's ``max|Im eps| /
    max|Re eps|`` to be <= 1e-12 and disarmed the WHOLE screen otherwise, so a
    3e-12 relative loss -- a stack that absorbs two parts in a million million,
    measured on its staggered twin -- took the guard out and the same 2.9 %
    violation came back silently.  Measured across this ladder on the
    pre-round-2 tree: REFUSED at 0 / 1e-14 / 1e-13 / 1e-12, RETURNED from 3e-12
    out to 1e-01, with the nodal excess pinned at 2.881869e-02 throughout.

    TWO CLAIMS, ONE GATED AND ONE NOT.

    * UNCONDITIONAL, on every arm: if the solve RETURNS, the answer it returns
      is physical -- no rung may hand back a super-unity above the bar in
      silence.  This is the guard's contract and it does not depend on the
      arithmetic reproducing any pathology.
    * PREMISE-GATED: that the pathology is HERE at all, i.e. that the lossless
      arm of this very fixture is refused.  An arm whose nodal cascade does not
      blow up has no screen to defeat and nothing to report.
    """
    k0 = 2.0
    import lumenairy.elements.bor.bor_solve as bs
    bar = bs._BOR_NODAL_SUPERUNITY_BAR
    rungs = []
    for rel in _LOSS_LADDER:
        layers = _nodal_stack(complex(6.0, 6.0 * rel))
        rungs.append((rel,) + _decide(layers, k0))

    # ---- UNCONDITIONAL: nothing returns a non-physical answer in silence ----
    silent = [(rel, float(np.max(E)) - 1.0)
              for rel, kind, E, nw in rungs
              if kind == "returned" and np.asarray(E).size
              and float(np.max(E)) - 1.0 > bar and nw == 0]
    assert not silent, (
        "the passivity screen returned a super-unity above its own %.0e bar "
        "with NO warning on %d rung(s) of the loss ladder: %s"
        % (bar, len(silent),
           ", ".join("Im/Re=%.0e -> max(R+T)-1=%.6g" % (r, v)
                     for r, v in silent)))

    # ---- PREMISE: the lossless arm must exhibit the pathology on this arm ----
    if rungs[0][1] != "refused":
        pytest.skip("premise absent on this arm: the LOSSLESS nodal stack is "
                    "not refused, so there is no screen to defeat")

    # ---- the CLOSED behaviour: the loss does not disarm it ----
    returned = [rel for rel, kind, _E, _nw in rungs if kind != "refused"]
    assert not returned, (
        "a loss of %s (relative) disarmed the passivity screen on a stack "
        "whose LOSSLESS twin is refused; the physical absorption at the "
        "smallest of them is ~1e-12 of the incident power"
        % (returned,))


def test_gain_is_not_passive_and_stays_outside_the_screen():
    """The other side of GAP 1's predicate, and the reason it is SIGNED.

    A medium with ``Im eps < 0`` AMPLIFIES in this library's
    ``exp(-i omega t)`` convention.  ``R + T + A = 1`` with ``A >= 0`` is then
    false in both directions, so there is no theorem and the screen must not
    speak -- exactly as the 1-D peer ``pmm/stack._stack_provably_passive``
    refuses to judge a gain substrate.  The pre-round-2 predicate tested
    ``|Im eps|`` and therefore judged infinitesimal GAIN as though it were
    lossless; round 2 tests the SIGN.

    UNCONDITIONAL: the guard does not raise on a gain stack, on any arm.
    """
    k0 = 2.0
    for rel in (3e-12, 1e-8, 1e-4):
        kind, _payload, _nw = _decide(_nodal_stack(complex(6.0, -6.0 * rel)),
                                      k0)
        assert kind == "returned", (
            "a stack with GAIN (Im eps / Re eps = %.0e) was judged by the "
            "passivity screen, which has no theorem there" % (-rel,))


# --------------------------------------------------------------------------- #
#  GAP 2 -- the screen was one-sided on a stack it had already proven lossless
# --------------------------------------------------------------------------- #
def test_the_nodal_passivity_screen_must_be_two_sided_on_a_lossless_stack():
    """GAP 2 (CLOSED in round 2).  On a stack the guard has ALREADY established
    is lossless and passive, and which is closed by a PEC wall -- so the
    propagating channels of the two half-spaces are the only exits -- ``R + T =
    1`` is an EQUALITY, not an inequality.  There is no absorption for a deficit
    to hide in, so a DEFICIT is damage of exactly the same kind as an excess and
    the screen must look both ways.

    The pre-round-2 screen tested only ``max(R+T) - 1``.  The verification
    measured 5 of 48 provably lossless nodal rows returning ``R + T < 1``
    silently, the worst at 0.537349; this fixture is that row (``m = 2``,
    ``Rbig/lambda = 0.5``, ``N = 200``, ring layer), whose staggered twin
    returns 1 to 6.5e-12.

    UNCONDITIONAL, on every arm: a LOSSLESS nodal stack that RETURNS must
    return ``|R + T - 1|`` inside the bar, in BOTH directions.
    PREMISE-GATED: that this arm's nodal cascade under-closes here at all.
    """
    import lumenairy.elements.bor.bor_solve as bs
    bar = bs._BOR_NODAL_SUPERUNITY_BAR
    k0, m = 2.0, 2
    Rbig = 0.5 * 2.0 * np.pi / k0                 # half a vacuum wavelength

    def build():
        return _nodal_stack(6.0 + 0j, Rbig=Rbig, N=200, k0=k0, m=m)

    kind, payload, nwarn = _decide(build(), k0)

    # ---- UNCONDITIONAL ----
    if kind == "returned":
        E = np.asarray(payload)
        worst = float(np.max(np.abs(E - 1.0))) if E.size else 0.0
        assert worst <= bar or nwarn, (
            "a LOSSLESS nodal stack returned |R + T - 1| = %.6g (R + T spans "
            "[%.9g, %.9g]) above the %.0e bar with NO warning -- the screen is "
            "still one-sided" % (worst, float(np.min(E)), float(np.max(E)),
                                 bar))

    # ---- PREMISE: the deficit has to be there on this arm ----
    ref = _solve_disarmed(build(), k0)
    E = np.asarray(ref["R"]) + np.asarray(ref["T"])
    if E.size == 0 or float(np.min(E)) > 1.0 - bar:
        pytest.skip("premise absent on this arm: the nodal cascade does not "
                    "under-close on this fixture (min R+T = %.12g)"
                    % (float(np.min(E)) if E.size else float("nan")))

    # ---- the CLOSED behaviour ----
    assert kind == "refused", (
        "a LOSSLESS nodal stack returning min(R + T) = %.9g -- a %.1f %% "
        "energy DEFICIT where R + T = 1 is an equality -- was not refused"
        % (float(np.min(E)), 100.0 * (1.0 - float(np.min(E)))))
    assert "DEFICIT" in payload, (
        "the refusal fired but its message does not name the half that "
        "decided it: %s" % (payload[:400],))


def test_a_lossy_nodal_stack_keeps_its_deficit_half_disarmed():
    """The complement of GAP 2, and the reason the two halves are separate.

    On a stack with a genuinely ABSORBING layer, ``R + T < 1`` is what the
    physics says: the deficit IS the absorption.  Measured here on uniform
    half-spaces with a lossy middle layer, the deficit tracks the loss exactly
    -- 1.8e-06 at ``Im/Re`` = 1e-6 up to 1.5e-01 at 1e-1 -- so a two-sided
    screen armed there would refuse every absorbing stack in the library.

    UNCONDITIONAL on every arm: none of these is refused or warned.
    """
    k0 = 2.0
    Rbig = 1.0 * 2.0 * np.pi / k0
    bad = []
    for rel in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
        for m in (0, 1, 2):
            layers = _uniform_stack(Rbig, 200, k0, m,
                                    e_mid=complex(4.0, 4.0 * rel))
            kind, payload, nwarn = _decide(layers, k0)
            if kind == "refused":
                bad.append("Im/Re=%.0e m=%d REFUSED" % (rel, m))
            elif nwarn:
                bad.append("Im/Re=%.0e m=%d warned x%d" % (rel, m, nwarn))
    assert not bad, (
        "the deficit half of the screen fired on a legitimately ABSORBING "
        "stack, whose R + T < 1 is the absorption and not damage: "
        + "; ".join(bad))


# --------------------------------------------------------------------------- #
#  GAP 3 -- the channel COUNT, against a closed form
# --------------------------------------------------------------------------- #
def _exact_channel_count(m, n_index, k0, Rbig, nz=200):
    """The number of propagating channels of a uniform PEC-walled cylinder.

    TE modes have ``J_m'(gamma Rbig) = 0``, TM modes ``J_m(gamma Rbig) = 0``,
    and a mode propagates when ``gamma Rbig < n k0 Rbig``.  Returns the count
    and the BASIN: the distance from the cut to the nearest zero on either
    side, in units of the cut.  Matching by basin rather than by a tolerance is
    what makes this build-free -- the count can only move if a zero crosses,
    and the nearest one is decades away in relative terms.
    """
    x = float(np.real(n_index)) * float(np.real(k0)) * float(Rbig)
    z = np.sort(np.concatenate([jnp_zeros(m, nz), jn_zeros(m, nz)]))
    n = int(np.sum(z < x))
    below = x - z[n - 1] if n > 0 else x
    above = z[n] - x if n < z.size else float("inf")
    return n, float(min(below, above) / x)


def _counts(basis, m, Rbig, N, k0, e_out=2.0, e_mid=4.0):
    prev = _bs.BOR_NODAL_PASSIVITY_GUARD
    _bs.BOR_NODAL_PASSIVITY_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            layers = [build_layer(m, Rbig, N, _uni(e_out), k0, basis=basis),
                      build_layer(m, Rbig, N, _uni(e_mid), k0, thickness=0.4,
                                  basis=basis),
                      build_layer(m, Rbig, N, _uni(e_out), k0, basis=basis)]
            r = solve(layers, k0)
        R, T = np.asarray(r["R"]), np.asarray(r["T"])
        return int(R.size), float(np.max(R + T)) - 1.0 if R.size else None
    finally:
        _bs.BOR_NODAL_PASSIVITY_GUARD = prev


@pytest.mark.parametrize("m", [1, 2])
def test_the_staggered_channel_count_is_the_closed_form_count(m):
    """THE REFERENCE THE PASSIVITY BAR DOES NOT HAVE.  Before asking whether
    ``R + T`` is physical it is worth asking whether ``R`` and ``T`` are the
    right observables at all, and for a uniform half-space that has a closed
    form.  The div-conforming staggered basis returns exactly it.

    Build-free by construction: the assertion is an INTEGER count matched
    against Bessel zeros, and the basin (the relative distance from the
    propagation cut to the nearest zero) is asserted to be wide enough that no
    arithmetic could move the count.
    """
    k0, N = 2.0, 200
    Rbig = 2.0 * 2.0 * np.pi / k0                  # two vacuum wavelengths
    exact, basin = _exact_channel_count(m, np.sqrt(2.0), k0, Rbig)
    assert basin > 1e-3, (
        "this fixture sits too near a mode cutoff to pin a count (basin %.3g)"
        % basin)
    n_stag, _ = _counts("staggered", m, Rbig, N, k0)
    assert n_stag == exact, (
        "m=%d, Rbig=%.6g, k0=%.6g: the staggered basis returned %d channels "
        "against the closed-form %d (basin %.3g)"
        % (m, Rbig, k0, n_stag, exact, basin))


@pytest.mark.parametrize("m", [1, 2])
def test_the_nodal_channel_count_is_wrong_where_the_passivity_screen_says_ok(m):
    """THE GAP, stated as a decision rather than as a magnitude.

    The 5.45.1 passivity bar refuses a nodal solve when ``max(R+T) - 1`` passes
    1e-3 and warns above 1e-6.  On this fixture it reads ~1e-9, i.e. the guard
    says OK -- and the returned channel SET is nevertheless wrong against the
    closed form.  The energy excess is therefore not a detector of the nodal
    basis's defect; it only sees the part of it that happens to carry power.

    This is not marked xfail: the assertion is that the gap EXISTS and is
    measured, so the number below is a finding, not a target.  It is
    premise-gated -- an arm on which the nodal basis happens to return the
    right count has nothing to report.
    """
    k0, N = 2.0, 200
    Rbig = 2.0 * 2.0 * np.pi / k0
    exact, basin = _exact_channel_count(m, np.sqrt(2.0), k0, Rbig)
    assert basin > 1e-3
    n_nodal, excess = _counts("nodal", m, Rbig, N, k0)
    if n_nodal == exact:
        pytest.skip("premise absent on this arm: the nodal basis returned the "
                    "closed-form count (%d)" % exact)
    assert excess is not None and excess <= 1e-6, (
        "the premise of this finding is that the passivity screen is SILENT "
        "here; it reads max(R+T) - 1 = %.6g, which is above the 1e-6 warning "
        "edge, so the screen is not silent and the finding has to be "
        "re-stated" % excess)
    assert n_nodal > exact, (
        "m=%d: nodal returned %d channels against the closed-form %d while "
        "the passivity screen read %.4g and said OK"
        % (m, n_nodal, exact, excess))


# --------------------------------------------------------------------------- #
#  GAP 4 -- the SEM warn edge is entered by an ordinary shallow taper
# --------------------------------------------------------------------------- #
def _taper(n_slices, dr, Rbig=24.0, k0=2.0, degree=8, r_top=8.0, height=0.8):
    s = BORStack(Rbig=Rbig, m=1, N=120, n_superstrate=1.0, n_substrate=1.0,
                 basis="sem", degree=degree)
    for j in range(n_slices):
        r = r_top - dr * (j + 0.5) / n_slices
        s.add_layer(height / n_slices, segments=[(r, 4.0), (Rbig, 1.0)])
    s.set_source(k0=k0)
    return s


def test_the_sem_warn_edge_is_reached_by_a_shallow_taper_at_32_slices():
    """WHAT THE SHIPPED CENSUS'S MARGIN IS A PROPERTY OF.

    ``_BOR_SLIVER_BAND_FRAC = 1e-4`` was fixed from a census whose binding
    ordinary geometry was a taper staircase, and the build's note says the edge
    "can never REFUSE one, because ... a taper reaches [1e-6 of Rbig] only at
    ~6 million slices".  Both statements are about the census's own cone, which
    changes radius by 6 over its height: the manufactured cell of a taper is
    ``dr / n_slices``, so it is set by the cone's STEEPNESS as much as by the
    slice count, and a SHALLOW cone reaches the same cell at an ordinary slice
    count.

    This gate pins that as a decision: a 32-slice taper whose ring radius
    changes by 0.05 in total (0.6 % of the ring radius, 0.2 % of ``Rbig``) --
    ordinary conical geometry, the kind a taper-angle sweep walks through --
    lands inside the warn band and warns.  The band membership is DERIVED here
    from the library's own two constants, so the gate tracks them if they
    move.
    """
    from lumenairy.elements.bor import _sem_contract as C
    Rbig, n_slices = 24.0, 32
    dr = 0.05                   # a 0.6 % change of the ring radius, total
    cell = dr / n_slices / Rbig                 # the manufactured cell / Rbig
    assert C._BOR_MIN_ELEM_FRAC <= cell < C._BOR_SLIVER_BAND_FRAC, (
        "this fixture is meant to sit INSIDE the degradation band [%.0e, "
        "%.0e); it reads %.3e -- re-derive it from the current constants"
        % (C._BOR_MIN_ELEM_FRAC, C._BOR_SLIVER_BAND_FRAC, cell))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _taper(n_slices, dr, Rbig=Rbig).solve()
    msgs = [str(x.message) for x in w
            if "MANUFACTURED element" in str(x.message)]
    assert msgs, (
        "a %d-slice taper of total radius change %.4g on Rbig = %.4g "
        "manufactures a cell of %.3e of Rbig, inside the contract's own "
        "degradation band, and emitted no warning" % (n_slices, dr, Rbig, cell))
    # and the warning is emitted once per AFFECTED LAYER, not once per cause:
    # one geometric coincidence, n_slices messages.
    assert len(msgs) >= n_slices - 1, (
        "expected one message per affected layer (%d), got %d"
        % (n_slices, len(msgs)))


# --------------------------------------------------------------------------- #
#  GAP 5 -- warn_own is attributed to a layer that asked for nothing
# --------------------------------------------------------------------------- #
def test_warn_own_is_only_reported_for_the_layer_that_prescribed_the_cell():
    """GAP 5 (CLOSED in round 2).  ``_sem_contract.verdict``'s ``warn_own``
    branch used to fire on ``w_min_frac`` -- the narrowest cell of the
    POST-WINDOW mesh, whoever asked for it -- and then tell the caller that
    "the LAYER'S OWN segment list asked for" that cell.  For the NEIGHBOUR of a
    liner that is false: its own segment list is a single full-radius entry, and
    it has the cell only because the ``+-1`` enrichment window put it there.
    Measured pre-fix: TWO messages, blaming layers [0, 1], where layer 1 is
    ``add_layer(0.5, eps=1.21)``.

    Round 2 gives the branch ``w_min_own_frac``, the narrowest cell BOTH of
    whose enclosing walls this layer's own segment list asked for -- which is
    the quantity the message's own words describe.  The stated reason for never
    REFUSING a ``warn_own`` cell ("you prescribed the geometry and the library
    does not overrule it") now holds for every layer that receives one.

    PREMISE-GATED on the warning being emitted at all; the ATTRIBUTION claim is
    the assertion.
    """
    Rbig, w = 24.0, 1e-6 * 24.0
    s = BORStack(Rbig=Rbig, m=1, N=120, n_superstrate=1.0, n_substrate=1.0,
                 basis="sem", degree=8)
    s.add_layer(0.5, segments=[(9.0, 1.0), (9.0 + w, 4.0), (Rbig, 1.0)])
    s.add_layer(0.5, eps=1.21)              # asks for NO wall at all
    s.set_source(k0=2.0)
    with warnings.catch_warnings(record=True) as w_rec:
        warnings.simplefilter("always")
        s.solve()
    own = [str(x.message) for x in w_rec
           if "LAYER'S OWN segment list" in str(x.message)]
    if not own:
        pytest.skip("premise absent on this arm: no warn_own message was "
                    "emitted for this liner")
    blamed = sorted({int(msg.split("layer ")[1].split("'s")[0])
                     for msg in own})
    assert blamed == [0], (
        "warn_own blamed layers %s; only layer 0 has a segment list that "
        "mentions the %.3e-wide cell" % (blamed, w / Rbig)
    )


# --------------------------------------------------------------------------- #
#  GAP 6 -- the EME band's literal 1.0 floor is unit-dependent
# --------------------------------------------------------------------------- #
def test_the_eme_branch_band_is_unit_system_invariant():
    """GAP 6 (CLOSED in round 2).  ``elements/eme/_branch.cut_band`` floored its
    band at a LITERAL 1.0, justified by "``ky`` here is DIMENSIONLESS".  It is
    not: ``strip_x_modes`` assembles ``d2/dx2 + eps k0^2`` on a spacing
    ``Lx/Nx``, so ``lam`` carries 1/length^2 and ``ky`` carries 1/length, and
    ``k0`` is a free argument carrying units rather than a normalisation.  The
    floor engaged whenever ``max|ky| < 1`` in the caller's units -- the ordinary
    case for a sub-micron cell written in nanometres -- and the BRANCH DECISION
    moved with the unit system, which the pre-5.45.1 exact-zero pin (having no
    scale) did not.  Round 2 floors at ``|k0|``, which is what the BOR peer
    ``_orient.orient_band_scale`` does and for the same reason (audit P2-06).

    THE OBSERVABLE IS UNIT-FREE BY CONSTRUCTION: ``ky`` scales as 1/length, so
    ``ky * scale`` is invariant, and the roots are SORTED before differencing --
    the eigensolver's own ordering is not stable between two solves, so an
    elementwise difference of the raw arrays measures a permutation rather than
    a moved root.

    Measured pre-fix on this fixture: 3 of 96 modes came back on a DIFFERENT
    ROOT in nm than in um (the nm band pinned at the floor, 5.2x wider in
    physical terms), and the whole sorted spectrum then differed.  Three unit
    systems are used here, not two, so an arm on which nm happened to agree
    still has metres to answer for.
    """
    from lumenairy.elements.eme import eme_2d as _e2

    def roots(scale):
        """The same physics stated with ``scale`` units to the micrometre:
        lengths x scale, wavenumbers / scale."""
        Lx, Nx = 1.0 * scale, 96
        k0 = 2.0 * np.pi / (1.55 * scale)
        eps = np.full(Nx, 1.0 + 0j)
        eps[Nx // 3:2 * Nx // 3] = 12.0 - 1e-6j          # weak GAIN
        lam = np.asarray(_e2.strip_x_modes(eps, Lx, Nx, k0)[0], complex)
        z = np.asarray(_e2._ky_forward(lam, 0.0, k0)) * scale
        return z[np.lexsort((np.imag(z), np.real(z)))]

    base = roots(1.0)                                     # micrometres
    for name, scale in (("nanometres", 1e3), ("metres", 1e-6)):
        other = roots(scale)
        # PREMISE: the two unit systems must describe the same spectrum to
        # begin with.  If the eigensolve itself moved, there is nothing to
        # attribute to the band.
        if base.size != other.size:
            pytest.skip("premise absent on this arm: %s returned a different "
                        "spectrum size (%d, %d)"
                        % (name, base.size, other.size))
        tol = 1e-6 * max(float(np.max(np.abs(base))), 1.0)
        n_diff = int((np.abs(base - other) > tol).sum())
        assert n_diff == 0, (
            "%d of %d forward roots differ between a micrometre and a %s "
            "statement of the SAME cell; worst |d ky * scale| = %.6g"
            % (n_diff, base.size, name,
               float(np.max(np.abs(base - other)))))
