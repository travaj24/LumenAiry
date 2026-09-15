"""VERIFY-WP-B7c round 2: the gaps the pixel-halving arbiter's own gate leaves.

`fixes/VERIFY_WP-B7c_ROUND2.md` re-measured WP-B7c round 2 on eight optics of
its own (five new to the campaign, one at NA 0.3865 and one conic) against an
oracle written for the verification -- an exact sequential conic ray trace
propagated three independent ways (the Debye ``J0`` ring integral, an EXACT
azimuthal quadrature, and a band-limited angular spectrum, which makes no
azimuthal approximation at all).  The arbiter's mechanism, its bit identity,
its cost and its two headline measurements all reproduce.

What did NOT reproduce is the GATE.  A mutation matrix over THIRTEEN
plausible regressions of the arbiter, each applied to a detached tree and run
against the shipped decision tests, found two that leave the suite GREEN and
one that is held only through a premise gate that can skip:

* **the bar itself is unpinned.**  Loosening ``_PIXEL_CONTINUITY_MAX`` from
  1.06 to 1.20 -- which RETURNS VERIFY-WP-B7c's own D1 plane, the R-5
  counterexample the whole round exists to close, at oracle fidelity 0.858 --
  left ``26 passed, 2 skipped``.  Every constant assertion in the shipped
  tests is satisfied by 1.20 (``1.0 < MAX < _ENERGY_BLOWUP_FACTOR``,
  ``MIN == 1 / MAX``), and the one behavioural test that would have seen it --
  the D1 plane's -- ``pytest.skip``s on exactly the premise the regression
  removes ("this build returns the pinned plane; its continuity reads 1.1851,
  inside the band"), which is the fail-open shape
  ``docs/TESTING_STANDARDS.md`` rule 4 exists to forbid;
* **where the reading is taken is unpinned.**  Deciding on the BRANCH SUM
  instead of the returned completed field -- the one design choice round 2
  argues for at length, and the one that would refuse VERIFY-B7b's own
  fixture at z = 1761 um where the branch sum reads 1.0636 and the completion
  1.0021 at oracle fidelity 0.988 -- left ``28 passed``;
* **the entry cap is held only through a ladder that can skip.**  Lowering
  ``_ARBITER_MAX_FINE_ENTRIES`` to 2e6 -- which keeps every published fixture
  inside the cap and puts only the refined grid VERIFY-WP-B7c's D2 is about
  (N = 1280) outside it -- is caught by exactly one round-1 test, whose own
  premise gate is a ``pytest.skip``.

The first two are closed here, two-sided and derived on the running build;
the third gets an arm that cannot skip; the rest are belt and braces.

No test here pins a number this verification measured: the bar is
checked against the FIELD (through the ``caustic='wave'`` hand-off, which the
oracle scores at 0.998 on these planes), the reading's provenance against the
diagnostics the module itself publishes, and the cap against the refinement
the module's own ``l_airy`` gate tells a caller to make.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.elements import _lens_traced_uniform as _U
from lumenairy.elements._lens_traced_multibranch import (
    _ARBITER_MAX_FINE_ENTRIES,
    _PIXEL_CONTINUITY_MAX,
)
from lumenairy.elements._lens_traced import apply_real_lens_traced
from lumenairy.elements._lens_traced_uniform import (
    _MB_PIXEL_CONTINUITY_MAX,
    apply_real_lens_traced_uniform,
)

# ---------------------------------------------------------------------------
# VERIFY-WP-B7c's ``F_alt`` -- the fast N-LASF9 singlet stopped to f/2.0, the
# ONLY fixture in either study whose ladder straddles the bar with a small
# margin, and the one carrying D1.  Measured by this verification against its
# own angular-spectrum oracle (2026-09-15), which reproduces the builder's
# fidelities to four decimals:
#
#   z [um]   continuity   oracle fidelity
#   1063     1.0038       0.9891
#   1070     1.0043       0.9901
#   1073     1.0221       0.9747     <- the largest RETURNED reading
#   1074     1.0921       0.9306     <- the smallest REFUSED reading
#   1076     1.1851       0.8583     <- D1's plane
# ---------------------------------------------------------------------------
_F_WL, _F_DX, _F_N, _F_W0 = 633e-9, 1.40e-6, 640, 220e-6
#: The three planes the bar is actually derived between -- the largest
#: RETURNED reading, the smallest REFUSED one, and D1's plane -- kept to
#: three so the ``caustic='wave'`` accuracy axis stays inside the 60 s budget.
_F_GOOD_Z = (1073e-6,)
_F_BAD_Z = (1074e-6, 1076e-6)

# VERIFY-B7b's own N-BAF10 biconvex: at z = 1761 um the BRANCH SUM reads
# 1.0636 (above the bar) while the COMPLETION built on it reads 1.0021, and
# the completed field's fidelity against this verification's oracle is 0.9879
# with 0.988x its power.  That is the plane the round-2 design decision is
# argued on, and nothing in the shipped gate holds it.
_V_WL, _V_DX, _V_N, _V_W0 = 1.064e-6, 2.20e-6, 512, 330e-6
_V_SPLIT_Z = 1761e-6

#: Accuracy basins, chosen far from each other and from any bar so the claim
#: is which REGIME a plane is in, not its value.  The ladder above separates
#: 0.9747 from 0.9306, so 0.97 / 0.94 sit inside the data's own spacing.
_GOOD_FIDELITY = 0.97
_BAD_FIDELITY = 0.94


def _f_presc():
    return {'wavelength': _F_WL, 'aperture_diameter': 0.60e-3, 'surfaces': [
        {'radius': 2.2e-3, 'thickness': 0.80e-3, 'glass_before': 'air',
         'glass_after': 'N-LASF9', 'semi_diameter': 0.50e-3},
        {'radius': -2.2e-3, 'thickness': 0.0, 'glass_before': 'N-LASF9',
         'glass_after': 'air', 'semi_diameter': 0.50e-3}],
        'thicknesses': [0.80e-3], 'stop_index': 0}


def _v_presc():
    return {'wavelength': _V_WL, 'aperture_diameter': 0.90e-3, 'surfaces': [
        {'radius': 2.6e-3, 'thickness': 0.70e-3, 'glass_before': 'air',
         'glass_after': 'N-BAF10', 'semi_diameter': 0.45e-3},
        {'radius': -2.6e-3, 'thickness': 0.0, 'glass_before': 'N-BAF10',
         'glass_after': 'air', 'semi_diameter': 0.45e-3}],
        'thicknesses': [0.70e-3], 'stop_index': 0}


def _gauss(N, dx, w0):
    x = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


def _fidelity(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(abs(np.vdot(a, b)) / (na * nb))


def _uniform_unbarred(monkeypatch, E, presc, wl, dx, z, **kw):
    """The completion's READING and FIELD at one plane, with both refusal
    bars lifted, so the reading and the field it would have refused come from
    the SAME call.  The shipped bars are imported at module scope, before any
    patch, and the decision is re-derived from them below."""
    monkeypatch.setattr(_U, '_MB_PIXEL_CONTINUITY_MAX', float('inf'))
    monkeypatch.setattr(_U, '_MB_PIXEL_CONTINUITY_MIN', 0.0)
    monkeypatch.setattr(_U, '_MB_POWER_RATIO_MAX', float('inf'))
    monkeypatch.setattr(_U, '_MB_POWER_RATIO_MIN', 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return apply_real_lens_traced_uniform(
            E, prescription=presc, wavelength=wl, dx=dx,
            output_plane_distance=z, return_diagnostics=True, **kw)


def _wave(E, presc, wl, dx, z):
    """The band-limited ASM hand-off of the traced exit-vertex field -- the
    library's own member that is exact through folds, which this
    verification's independent angular-spectrum oracle scores at 0.998 at
    these very planes.  Used here as the ACCURACY axis, never as the
    quantity under test."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(apply_real_lens_traced(
            E, prescription=presc, wavelength=wl, dx=dx,
            output_plane_distance=z, caustic='wave'))


# ===========================================================================
# 1. the bar separates the FIELD, not only the reading
# ===========================================================================
def test_vb7c2_the_bar_is_derived_against_the_field_on_both_sides(monkeypatch):
    """DECISION, two-sided, derived on the running build: on a ladder that
    straddles ``_PIXEL_CONTINUITY_MAX``, every plane whose FIELD is good
    reads BELOW the bar and every plane whose field is broken reads ABOVE it.

    This is the claim ``_PIXEL_CONTINUITY_MAX``'s comment makes -- "the 67
    planes RETURNED read 0.9860 .. 1.0221 ... the 15 REFUSED read
    1.092 .. 3.998 ... the two fidelity populations do not overlap" -- and
    nothing in the shipped gate holds it.  Every shipped assertion on this
    constant is a shape (``1.0 < MAX < _ENERGY_BLOWUP_FACTOR``,
    ``MIN == 1 / MAX``), and the behavioural test that would have seen a
    loosened bar ``pytest.skip``s when the guard stops refusing.  Measured:
    loosening the bar to 1.20 RETURNS D1's plane at oracle fidelity 0.858 and
    leaves the shipped files at ``26 passed, 2 skipped``.

    Build-free: the accuracy axis is measured here, on this build, against
    the ``caustic='wave'`` hand-off at the same plane (a different propagator
    of the same traced exit field, which this verification's independent
    angular-spectrum oracle scores at 0.998 on both sides of this ladder);
    the basins 0.97 / 0.94 sit inside the spacing the ladder itself exhibits
    (0.9747 against 0.9306).  Nothing pins a continuity VALUE.
    """
    E = _gauss(_F_N, _F_DX, _F_W0)
    presc = _f_presc()
    good, bad = [], []
    for z in _F_GOOD_Z + _F_BAD_Z:
        _E, d = _uniform_unbarred(monkeypatch, E, presc, _F_WL, _F_DX, z)
        c = d.get('pixel_continuity')
        assert c is not None, (
            f'z={z * 1e6:.0f} um: the completion returned no reading at all; '
            f'the arbiter is not being asked, or its entry cap moved')
        assert d.get('pixel_continuity_of') == 'the completed fold field', (
            f"z={z * 1e6:.0f} um is no longer on the completion route "
            f"({d.get('reason')!r}); this ladder must stay on it")
        f = _fidelity(_E, _wave(E, presc, _F_WL, _F_DX, z))
        if f >= _GOOD_FIDELITY:
            good.append((z, c, f))
        elif f <= _BAD_FIDELITY:
            bad.append((z, c, f))
    assert good, (
        'no plane of this ladder has a field the hand-off agrees with; the '
        'fixture or the completion moved, and the bar cannot be derived here')
    worst_good = max(good, key=lambda t: t[1])
    assert worst_good[1] < _MB_PIXEL_CONTINUITY_MAX, (
        f'a plane whose field is GOOD (fidelity {worst_good[2]:.4f} against '
        f'the hand-off) reads {worst_good[1]:.4f}, at or above the '
        f'{_MB_PIXEL_CONTINUITY_MAX:g} bar: the guard would refuse a field '
        f'that is right')
    if not bad:
        pytest.fail(
            'no plane of this ladder produces a BROKEN field on this build, '
            'so the bar has no upper side to be derived against here; '
            'extend the ladder rather than dropping the claim')
    best_bad = min(bad, key=lambda t: t[1])
    assert best_bad[1] > _MB_PIXEL_CONTINUITY_MAX, (
        f'a plane whose field is BROKEN (fidelity {best_bad[2]:.4f} against '
        f'the hand-off, which is exact through folds) reads '
        f'{best_bad[1]:.4f}, INSIDE the {_MB_PIXEL_CONTINUITY_MAX:g} bar: '
        f'the guard returns it')
    # ...and the gap the bar sits in is the one the constant claims, with
    # both margins reported rather than asserted to a value.
    assert best_bad[1] / worst_good[1] > 1.0


# ===========================================================================
# 2. the reading is taken on the RETURNED field
# ===========================================================================
def test_vb7c2_the_reading_is_of_the_returned_field_not_of_the_branch_sum(
        monkeypatch):
    """DECISION: where the completion applies, the recorded reading is the
    COMPLETED field's and not the branch sum's -- which is the single design
    choice round 2 argues for, and which no shipped test holds.

    Measured (VERIFY round 2, 2026-09-15) on VERIFY-B7b's own fixture at
    z = 1761 um: the branch sum reads 1.0636, ABOVE the bar, while the
    completion built on it reads 1.0021 and the completed field scores 0.9879
    against this verification's angular-spectrum oracle with 0.988x its
    power.  Deciding on the branch sum would refuse that field.  Substituting
    the branch sum's reading for the completion's leaves the shipped files at
    ``28 passed``.

    Unconditional arm: the diagnostics name the completed field and carry a
    number that is not the branch sum's.  Premise-gated arm: on this build's
    own reading, the branch sum is above the bar and the completion below it,
    so the substitution is a DECISION difference and not a cosmetic one.
    """
    E = _gauss(_V_N, _V_DX, _V_W0)
    presc = _v_presc()
    _E, d = _uniform_unbarred(monkeypatch, E, presc, _V_WL, _V_DX, _V_SPLIT_Z)
    assert d.get('reason') == 'fold_ring' and not d.get('fell_back'), (
        f"z = {_V_SPLIT_Z * 1e6:.0f} um is no longer on the completion route "
        f"({d.get('reason')!r}); this claim is only about that route")
    c_ret = d.get('pixel_continuity')
    c_mb = d.get('multibranch_pixel_continuity')
    assert c_ret is not None and c_mb is not None
    assert d.get('pixel_continuity_of') == 'the completed fold field'
    assert c_ret != c_mb, (
        'the completion recorded the BRANCH SUM\'s reading as its own: '
        f'{c_ret!r} == {c_mb!r}.  The CFU swap rewrites the fold band and the '
        'whole dark side, so the two are not the same number, and the field '
        'this call returns is the completed one')
    if not (c_mb > _MB_PIXEL_CONTINUITY_MAX >= c_ret):
        pytest.skip(
            f'this build does not straddle the bar at this plane '
            f'(branch sum {c_mb:.4f}, completion {c_ret:.4f}); the '
            f'unconditional arm above ran instead')
    # the decision, with the SHIPPED bars back in place -- the reading above
    # was taken with them lifted, so a decision read there would be vacuous
    monkeypatch.undo()
    assert _U._MB_PIXEL_CONTINUITY_MAX == _MB_PIXEL_CONTINUITY_MAX
    # the field IS returned, and it is the completed one
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E_out, ud = apply_real_lens_traced_uniform(
            E, prescription=presc, wavelength=_V_WL, dx=_V_DX,
            output_plane_distance=_V_SPLIT_Z, return_diagnostics=True)
    assert ud['pixel_continuity_decision'] == 'ok'
    assert np.all(np.isfinite(np.asarray(E_out)))
    f = _fidelity(E_out, _wave(E, presc, _V_WL, _V_DX, _V_SPLIT_Z))
    assert f > 0.95, (
        f'the field the completion returns at the plane where the branch '
        f'sum would have refused scores only {f:.4f} against the hand-off; '
        f'the argument for reading the returned field rests on that field '
        f'being right')


# ===========================================================================
# 3. the entry cap covers the refinement the module tells callers to make
# ===========================================================================
def test_vb7c2_the_entry_cap_survives_the_refinement_d2_is_about(monkeypatch):
    """``_ARBITER_MAX_FINE_ENTRIES`` must not put the reading out of reach on
    the grid a caller refines TO.

    VERIFY-WP-B7c's D2 is a caller halving ``dx`` to resolve the Airy layer,
    which this module's own ``l_airy`` gate tells them to do: a 640-grid
    fixture becomes 1280.  The constant's comment says 7e6 was chosen over
    1e6/N=1024 for exactly that reason -- "a cap that stopped reporting THERE
    would reproduce the very defect the arbiter closes" -- and nothing
    asserts it directly.  Lowering the cap to 1e6 is caught loudly (it takes
    the reading away from every published fixture too); lowering it to 2e6,
    which keeps N = 512 / 640 / 768 inside and puts only D2's N = 1280
    outside, is caught by ONE round-1 test through a ladder that can itself
    skip.  This arm cannot skip.

    Derived, not pinned: the entry count is ``4 N^2`` for a caller grid
    ``N``, and the two grids named are the campaign's largest published one
    (768) and D2's refined one (1280).  Both arms are unconditional, and the
    behavioural arm reads the decision on a real call.
    """
    assert 4 * 768 ** 2 <= _ARBITER_MAX_FINE_ENTRIES, (
        "the arbiter's entry cap no longer covers the largest grid the "
        'campaign publishes (N = 768)')
    assert 4 * 1280 ** 2 <= _ARBITER_MAX_FINE_ENTRIES, (
        "the arbiter's entry cap no longer covers D2's refined grid "
        '(N = 640 halved to 1280): the guard would disappear exactly when a '
        "caller refines, which is the defect it closes")
    # behavioural: at the UNREFINED grid of that fixture the reading is taken
    E = _gauss(_F_N, _F_DX, _F_W0)
    _E, d = _uniform_unbarred(monkeypatch, E, _f_presc(), _F_WL, _F_DX,
                              _F_GOOD_Z[0])
    assert d.get('pixel_continuity_decision') not in (
        'not_measured', 'not_requested'), (
        f"the reading is {d.get('pixel_continuity_decision')!r} at N = "
        f'{_F_N}, which is inside the cap: the completion is not asking for '
        f'it, or the cap moved')
    assert d.get('pixel_continuity') is not None


# ===========================================================================
# 4. an unmeasurable reading is never silently 'ok'
# ===========================================================================
def test_vb7c2_an_unmeasurable_reading_is_never_reported_as_ok(monkeypatch):
    """Past the entry cap the decision must say ``not_measured``, never
    ``ok``: "converged" and "not asked" are different answers, and a consumer
    that cannot tell them apart has no guard at all.

    Unconditional.  The premise is ENGINEERED rather than hoped for -- the
    cap is lowered on the running build so the plane under test is past it,
    which is how this claim stays true whatever ``_ARBITER_MAX_FINE_ENTRIES``
    is set to.
    """
    from lumenairy.elements import _lens_traced_multibranch as _MB
    monkeypatch.setattr(_MB, '_ARBITER_MAX_FINE_ENTRIES', 1000)
    E = _gauss(_V_N, _V_DX, _V_W0)
    _E, d = _uniform_unbarred(monkeypatch, E, _v_presc(), _V_WL, _V_DX,
                              _V_SPLIT_Z)
    assert d.get('pixel_continuity') is None, (
        'the reading was taken although the fine grid is past the cap')
    assert d.get('pixel_continuity_decision') == 'not_measured', (
        f"past the entry cap the decision reads "
        f"{d.get('pixel_continuity_decision')!r}; a consumer cannot tell "
        f'"converged" from "not asked"')
    assert np.all(np.isfinite(np.asarray(_E))), (
        'the field must still be returned when the reading cannot be taken')


# ===========================================================================
# 5. the fallback route records a reading, and says what it is of
# ===========================================================================
def test_vb7c2_a_fallback_plane_is_arbitrated_on_the_field_it_returns(
        monkeypatch):
    """On a fallback the field RETURNED is the branch sum itself, so the
    branch sum's reading IS the returned field's -- and the diagnostics must
    say so, with the number present rather than ``None``.

    This is the one route where reading the branch sum is correct, and it is
    the route the arbiter is weakest on (the constant's own comment scopes
    it: "a plane it accepts is not thereby certified").  What is pinned here
    is only that the reading is TAKEN and correctly attributed -- dropping it
    on the fallback path leaves the shipped files green.

    Premise-gated on the ladder still falling back; the attribution arm is
    unconditional on whichever route the plane takes.
    """
    E = _gauss(_V_N, _V_DX, _V_W0)
    presc = _v_presc()
    got = None
    for z_um in (1775, 1780, 1790):
        _E, d = _uniform_unbarred(monkeypatch, E, presc, _V_WL, _V_DX,
                                  z_um * 1e-6)
        if d.get('fell_back'):
            got = (z_um, d, _E)
            break
    if got is None:
        pytest.skip('no rung of this ladder falls back on this build')
    z_um, d, _E = got
    assert d.get('pixel_continuity') is not None, (
        f'z = {z_um} um falls back and returns the branch sum, but no '
        f'reading of it was recorded')
    assert d.get('pixel_continuity_of') == 'the plain multibranch field', (
        f"a fallback plane reports its reading as "
        f"{d.get('pixel_continuity_of')!r}")
    assert d['pixel_continuity'] == d['multibranch_pixel_continuity'], (
        'on a fallback the returned field IS the branch sum, so the two '
        'readings must be the same number to the bit')
    assert d.get('pixel_continuity_decision') in (
        'ok', 'not_converged_gain', 'not_converged_loss')


# ===========================================================================
# 6. the two modules' constants stay one constant
# ===========================================================================
def test_vb7c2_the_bar_is_one_constant_and_the_band_is_its_reciprocal():
    """Unconditional, build-free.  The consumer's bar IS the producer's, and
    the reporting arm IS its reciprocal, so the two modules cannot classify
    the same field differently and the band cannot become asymmetric without
    a derivation of its own."""
    assert _MB_PIXEL_CONTINUITY_MAX == _PIXEL_CONTINUITY_MAX
    assert _U._MB_PIXEL_CONTINUITY_MIN == pytest.approx(
        1.0 / _MB_PIXEL_CONTINUITY_MAX, rel=1e-12)
    # the gain arm must sit strictly inside the launched-power tripwire it
    # was introduced to decide underneath
    assert 1.0 < _MB_PIXEL_CONTINUITY_MAX < _U._MB_POWER_RATIO_MAX
