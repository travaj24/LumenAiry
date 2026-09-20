"""VERIFY-WP-B7c round 3: the gaps the round-3 gate leaves open.

`fixes/VERIFY_WP-B7c_ROUND3.md` re-measured WP-B7c round 3 -- the bar's
derivation, the fold ring's two fidelity populations, the half-pitch lattice,
the fallback route's scope and the bit identity -- on an independent
population of twelve (prescription, grid) pairs over eight prescriptions, four
of which no round of this campaign has used, scored against an independently
written full-radius oracle.  Most of what round 3 reports reproduces.  This
file is the gate for the five things that did not have one.

* **an unnamed SCOPE must be refused with a diagnosable error.**  Round 3
  added the guard and states the property ("an unknown scope raises inside
  ``_arbitrate`` rather than reaching a caller with the key absent"), and its
  own structural pin checks the enum's CONTENTS -- not that an unnamed value
  is refused, and not that the refusal says which names are legal.  Deleting
  the guard leaves the round-3 gate green (verification's mutation `N1`).
* **the two modules' derivations must cite the SAME population.**  The bar's
  derivation lives in one module and its consumer's summary in the other, and
  both quote the population they were measured on.  Editing one of those
  counts leaves the round-3 gate green (mutation `N4`), which
  `docs/TESTING_STANDARDS.md` makes a defect: "a numeric constant in a test
  without a stated origin is a defect", and the same applies to a constant
  whose stated origin can drift.
* **the scope NOTE a caller reads must agree with the constant's own
  derivation.**  The note carries the fallback route's confusion numbers in
  words; nothing held them against the numbers the constants were derived
  from (mutation `N5`).
* **the bar has no MARGIN, because the crossing is a JUMP.**  Round 3's
  strongest structural finding is that the gap is a reading of the z ladder.
  This verification found the sharper statement: bisecting z to three
  PICOMETRES does not close the crossing -- on VERIFY-B7b's own fixture the
  reading still steps from 0.9932 to 1.3885 across the bar at
  ``dz = 2.9e-3 nm``.  The reading is DISCONTINUOUS in the plane, so a margin
  is not a property that exists to be measured.  Nothing pinned that.
* **the converged reading's residue does not follow the WINDOW.**  Round 3
  attributes it to the two lattices' Voronoi-hull mismatch at the window edge.
  Its own window ladder refutes that: once the window holds the field the
  reading stops moving at all, which an edge term cannot do.  The residue is
  the two point-sampled quadratures' INTERIOR difference and it follows the
  PITCH.  Pinned here as a two-sided decision.

Every bar below is derived on the running build from that build's own
geometry, and every premise is ASSERTED rather than skipped.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from lumenairy.elements import _lens_traced_multibranch as _MB
from lumenairy.elements import _lens_traced_uniform as _U
from lumenairy.elements._lens_traced_multibranch import _PIXEL_CONTINUITY_MAX
from lumenairy.elements._lens_traced_uniform import (
    _PIXEL_CONTINUITY_SCOPES,
    apply_real_lens_traced_uniform,
)

# ---------------------------------------------------------------------------
# VERIFY-B7b's own N-BAF10 biconvex, which the round-2 design decision is
# argued on and the shipped gate already runs, and the plano-first N-BAK4 of
# the round-2 verification.  No new prescription joins the suite's cost.
# ---------------------------------------------------------------------------
_V = dict(wavelength=1.064e-6, N=512, dx=2.20e-6, w0=330e-6,
          prescription={
              'wavelength': 1.064e-6, 'aperture_diameter': 0.90e-3,
              'surfaces': [
                  {'radius': 2.6e-3, 'thickness': 0.70e-3,
                   'glass_before': 'air', 'glass_after': 'N-BAF10',
                   'semi_diameter': 0.45e-3},
                  {'radius': -2.6e-3, 'thickness': 0.0,
                   'glass_before': 'N-BAF10', 'glass_after': 'air',
                   'semi_diameter': 0.45e-3}],
              'thicknesses': [0.70e-3], 'stop_index': 0})

_W = dict(wavelength=780e-9, N=512, dx=2.40e-6, w0=400e-6,
          prescription={
              'wavelength': 780e-9, 'aperture_diameter': 1.10e-3,
              'surfaces': [
                  {'radius': float('inf'), 'thickness': 0.80e-3,
                   'glass_before': 'air', 'glass_after': 'N-BAK4',
                   'semi_diameter': 0.55e-3},
                  {'radius': -2.9e-3, 'thickness': 0.0,
                   'glass_before': 'N-BAK4', 'glass_after': 'air',
                   'semi_diameter': 0.55e-3}],
              'thicknesses': [0.80e-3], 'stop_index': 0})

#: the straddling pair the bisection below starts from, in metres.  Measured
#: on win py3.14.6 / numpy 2.4.4 and wsl py3.12.3 / numpy 2.4.6, 2026-09-19:
#: 1761 um reads 1.00213 (returned) and 1764 um reads 3.99657 (refused), both
#: on the completion route.  BOTH premises are asserted, so a build that moves
#: the crossing fails with an instruction rather than passing vacuously.
_V_STRADDLE = (1761e-6, 1764e-6)


def _gauss(N, dx, w0):
    x = (np.arange(int(N)) - int(N) / 2.0) * float(dx)
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X * X + Y * Y) / (float(w0) ** 2)).astype(np.complex128)


def _call(fx, z, N=None, dx=None, unbarred=False, monkeypatch=None):
    """One completion call, optionally with both refusal bars lifted so the
    reading and the field the bar would have refused come from one call."""
    if unbarred:
        monkeypatch.setattr(_U, '_MB_PIXEL_CONTINUITY_MAX', float('inf'))
        monkeypatch.setattr(_U, '_MB_PIXEL_CONTINUITY_MIN', 0.0)
        monkeypatch.setattr(_U, '_MB_POWER_RATIO_MAX', float('inf'))
        monkeypatch.setattr(_U, '_MB_POWER_RATIO_MIN', 0.0)
    N = int(fx['N'] if N is None else N)
    dx = float(fx['dx'] if dx is None else dx)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return apply_real_lens_traced_uniform(
            _gauss(N, dx, fx['w0']), prescription=fx['prescription'],
            wavelength=fx['wavelength'], dx=dx,
            output_plane_distance=float(z), return_diagnostics=True)


def _paraxial_focal_distance(fx):
    """Back focal distance from the last vertex, by a paraxial ``y-nu`` trace
    of this prescription on this build's own glass data.

    Derived rather than pinned: the plane the window ladder is taken at is a
    fraction of THIS number, so a change of glass model moves the plane with
    it instead of silently moving the test off its premise.
    """
    from lumenairy.glass import get_glass_index
    wl = float(fx['wavelength'])
    y, nu = 1.0, 0.0
    n = 1.0
    for s in fx['prescription']['surfaces']:
        g = s.get('glass_after')
        n2 = (1.0 if g in (None, 'air') else float(get_glass_index(g, wl)))
        R = s.get('radius')
        if R is not None and np.isfinite(R) and R != 0.0:
            nu = nu - y * (n2 - n) / float(R)
        y = y + float(s['thickness']) * (nu / n2)
        n = n2
    assert nu < 0.0, ('this prescription is not converging paraxially '
                      f'(nu = {nu!r}); the ladder below needs a focus')
    return -y / (nu / n)


# ===========================================================================
# 1. the scope enum's GUARD, not just its contents
# ===========================================================================
def test_vb7c3_an_unnamed_continuity_scope_is_refused_with_a_diagnosable_error(
        monkeypatch):
    """DECISION, two-sided: a scope the enum does not name never reaches a
    caller, and the refusal says which names are legal.

    Round 3 states this property where the enum is defined, and its own
    structural pin asserts the enum's CONTENTS.  Contents are not the guard:
    deleting the four-line check inside ``_arbitrate`` leaves round 3's five
    files green, and what a caller then gets is either a bare ``KeyError``
    from the note lookup or -- if the note lookup is made forgiving too -- a
    diagnostics dict carrying a scope nothing defines.  Neither is a
    diagnosable refusal.

    The unnamed scope is INJECTED by removing the name the shipped completion
    route uses from the enum, which is the same state a fourth route added
    later without a scope would produce, and is constructed through the
    module's own surface rather than by editing it.

    Two-sided: with the shipped enum the same call returns, and the scope it
    records is one the enum names.
    """
    # (a) the shipped enum: the call returns and the scope is a named one
    _E, d = _call(_V, _V_STRADDLE[0])
    assert d['reason'] == 'fold_ring' and not d['fell_back'], (
        f"z = {_V_STRADDLE[0] * 1e6:g} um no longer takes the completion "
        f"route on this build (reason={d['reason']!r}); move the plane rather "
        'than dropping the arm')
    assert d['pixel_continuity_scope'] in _PIXEL_CONTINUITY_SCOPES
    assert (d['pixel_continuity_scope_note']
            == _PIXEL_CONTINUITY_SCOPES[d['pixel_continuity_scope']]), (
        'the note a caller reads is not the enum entry the scope names')
    used = d['pixel_continuity_scope']

    # (b) the same call with that name removed from the enum
    monkeypatch.setattr(_U, '_PIXEL_CONTINUITY_SCOPES',
                        {k: v for k, v in _PIXEL_CONTINUITY_SCOPES.items()
                         if k != used})
    with pytest.raises(AssertionError) as exc:
        _call(_V, _V_STRADDLE[0])
    msg = str(exc.value)
    assert repr(used) in msg, (
        f'the refusal does not name the offending scope: {msg!r}')
    assert 'apply_real_lens_traced_uniform' in msg, msg
    for legal in _PIXEL_CONTINUITY_SCOPES:
        if legal == used:
            continue
        assert legal in msg, (
            f'the refusal does not list the legal scope {legal!r}, so a '
            f'caller cannot tell what it should have passed: {msg!r}')


# ===========================================================================
# 2 / 3. the derivations must not drift apart
# ===========================================================================
def _src(module):
    return Path(module.__file__).read_text(encoding='cp1252')


def test_vb7c3_the_two_modules_derivations_cite_the_same_population():
    """The bar's derivation and its consumer's summary cite ONE population.

    ``_PIXEL_CONTINUITY_MAX`` carries the derivation; ``_MB_PIXEL_CONTINUITY_MAX``
    in the consumer carries a summary of the same measurement, and both state
    the number of oracle-scored planes it was taken on.  Nothing held them
    together, so editing either one leaves the gate green -- and
    ``docs/TESTING_STANDARDS.md`` is explicit that a constant's stated origin
    is part of the constant.

    What is asserted is AGREEMENT, not a value: the test reads both numbers
    off the running source and requires them to be equal, so a re-derivation
    on a larger population moves both or fails.

    NOTE (VERIFY round 3, defect D-1): this covers the two CONSTANT
    derivations.  A third population count, in the ``_PIXEL_CONTINUITY_SCOPES``
    block of the consumer, is stale on this branch (it cites a count from an
    intermediate pass); the requested edit makes it agree, and this assertion
    should be widened to the whole file once it does.
    """
    pat = re.compile(r'\*{0,2}([\d][\d\s,]*)\*{0,2}\s+oracle-scored planes')

    def counts(text, start_marker, end_marker):
        i = text.index(start_marker)
        j = text.index(end_marker, i)
        return [int(m.group(1).replace(' ', '').replace(',', ''))
                for m in pat.finditer(text[i:j])]

    a = counts(_src(_MB), '# RE-MEASURED (WP-B7c round 3',
               '\n_PIXEL_CONTINUITY_MAX =')
    b = counts(_src(_U), '# RE-MEASURED (WP-B7c round 3',
               '\n_MB_PIXEL_CONTINUITY_MAX =')
    assert a, 'the bar constant no longer states the population it was derived on'
    assert b, 'the consumer no longer states the population it summarises'
    assert set(a) == set(b) and len(set(a)) == 1, (
        f'the bar derivation cites {sorted(set(a))} oracle-scored planes and '
        f'its consumer cites {sorted(set(b))}; a constant whose stated origin '
        'disagrees with itself documents nothing')


def test_vb7c3_the_scope_note_a_caller_reads_agrees_with_the_constants_derivation():
    """The words a CALLER reads carry the numbers the constants were derived
    from.

    ``pixel_continuity_scope_note`` is the only part of this derivation that
    reaches a caller at run time, and it states the fallback route's
    population and the two loss arms' flag counts.  The same three numbers
    appear in the two constants' own derivation comments.  Editing the note
    alone leaves the round-3 gate green, which makes the string a place a
    measurement can rot without anything saying so.

    Asserted as AGREEMENT between the running string and the running source,
    never as a value.
    """
    note = _PIXEL_CONTINUITY_SCOPES['returned_field_quadrature_only']
    src = _src(_MB) + _src(_U)

    def one(pattern, where):
        m = re.search(pattern, where)
        assert m, f'{pattern!r} no longer appears in {"the note" if where is note else "the source"}'
        return int(m.group(1))

    n_pop = one(r'over (\d+) returned fallback planes', note)
    n_cont = one(r'flags (\d+) planes', note)
    n_pow = one(r'below 0\.889 flags (\d+)', note)

    assert re.search(rf'\*\*{n_pop} fallback planes the shipped bars', src), (
        f'the note tells a caller the fallback population is {n_pop} planes, '
        'which is not the population either constant states it was derived '
        'on')
    assert re.search(rf'flags {n_cont}, EVERY ONE of them wrong', src), (
        f'the note says the continuity loss arm flags {n_cont} planes; the '
        "constant's own derivation says something else")
    assert re.search(rf'tighter 0\.889\s+flags {n_pow}', src), (
        f'the note says the tighter power-loss bar flags {n_pow} planes; the '
        "constant's own derivation says something else")


# ===========================================================================
# 4. the bar has no margin, because the crossing is a JUMP in z
# ===========================================================================
def test_vb7c3_the_bar_has_no_margin_because_the_crossing_is_a_jump_in_z(
        monkeypatch):
    """DECISION: the reading is DISCONTINUOUS in the output plane at the bar.

    Round 3's central structural finding is that the "gap" the bar sits in is
    a reading of the z ladder: refining the ladder keeps finding readings
    closer to the bar, so no margin statement can be made.  That is a
    statement about a population of ladders.  The mechanism under it is
    sharper and is what this pins: at a single crossing the reading does not
    approach the bar continuously at all -- it STEPS across it, and the step
    survives bisecting the plane to picometres, because what changes is the
    ray map's branch count on a pixel and not a smooth field.

    Measured here on the running build.  On win py3.14.6 / numpy 2.4.4 and on
    wsl py3.12.3 / numpy 2.4.6 (2026-09-19) the bisection closes to
    ``dz = 2.9e-3 nm`` with the two sides still reading 0.99322 and 1.38848 --
    a 1.398x step at a separation of three picometres.  Sixteen halvings of
    the 3 um starting bracket leave 46 pm, and the claim asserted is a BASIN:
    the pair still straddles the bar and its ratio still clears 1.05.  That is
    decades below the 1.398x measured and decades above anything a build's
    last bits can produce, and it is the statement that a MARGIN on this bar
    is not a property waiting to be measured more carefully.

    Both refusal bars are lifted for the duration, so the READING and the
    field the bar would have refused come from the same call and the
    bisection can walk through planes the shipped library refuses.  The
    shipped bar is imported BEFORE the patch and is what every decision below
    is taken against.

    Both premises are ASSERTED: if this build stops straddling at the starting
    pair the test fails with an instruction to move it.
    """
    lo, hi = _V_STRADDLE
    _E, d_lo = _call(_V, lo, unbarred=True, monkeypatch=monkeypatch)
    _E, d_hi = _call(_V, hi, unbarred=True, monkeypatch=monkeypatch)
    c_lo, c_hi = d_lo['pixel_continuity'], d_hi['pixel_continuity']
    assert c_lo is not None and c_hi is not None, (c_lo, c_hi)
    assert d_lo['reason'] == 'fold_ring' and not d_lo['fell_back'], (
        f"z = {lo * 1e6:g} um no longer takes the completion route "
        f"(reason={d_lo['reason']!r}); move the pair rather than dropping "
        'the claim')
    assert c_lo <= _PIXEL_CONTINUITY_MAX < c_hi, (
        f'the starting pair no longer straddles the bar on this build '
        f'({c_lo!r} at {lo * 1e6:g} um, {c_hi!r} at {hi * 1e6:g} um); find '
        'a straddling pair rather than skipping')

    z_lo, z_hi = lo, hi
    for _ in range(16):
        mid = 0.5 * (z_lo + z_hi)
        _E, d = _call(_V, mid, unbarred=True, monkeypatch=monkeypatch)
        c = d['pixel_continuity']
        assert c is not None, (
            f'no reading at z = {mid * 1e6:.9f} um; the bisection needs one')
        if c > _PIXEL_CONTINUITY_MAX:
            z_hi, c_hi = mid, c
        else:
            z_lo, c_lo = mid, c
    dz = z_hi - z_lo
    assert dz < 1.0e-10, (
        f'the bracket closed to only {dz * 1e9:.3g} nm; sixteen halvings of '
        f'{(hi - lo) * 1e6:g} um should reach {(hi - lo) / 2 ** 16 * 1e9:.3g} '
        'nm')
    assert c_lo <= _PIXEL_CONTINUITY_MAX < c_hi, (c_lo, c_hi)
    assert c_hi / c_lo > 1.05, (
        f'the crossing closed to {c_hi / c_lo:.5f}x at dz = {dz * 1e12:.3g} '
        'pm, so on this build the reading IS continuous through the bar and '
        'a two-sided margin can be derived after all -- re-derive the bar '
        'from that margin instead of from its cost')


# ===========================================================================
# 5. the converged residue follows the PITCH, not the WINDOW
# ===========================================================================
def test_vb7c3_the_converged_readings_residue_does_not_follow_the_window():
    """DECISION, two-sided: growing the WINDOW at a fixed pitch stops moving
    the reading, and refining the PITCH does not.

    The reading's usable property is that it has a fixed reference -- a
    converged point-sampled quadrature deposits the same power at any pitch.
    Round 2 published that reference as exact; round 3 corrected it to a
    measured spread and attributed the residue to the two lattices' Voronoi
    HULLS differing by a quarter of a coarse pixel at the window edge.

    That attribution is refuted by its own ladder and by this one.  A term
    living at the window EDGE must shrink as the window grows past the field.
    Measured here (2026-09-19, both builds): at a fixed 3.00 um pitch the
    reading is 4.96e-4 from 1 on a window too small to hold the field and then
    does not move AT ALL -- 7.30e-5, bit for bit, at three window sizes
    spanning 2.3x.  What does move it is the PITCH: holding the window and
    refining the grid takes the same deviation from 7.85e-4 to 1.3e-5.  So the
    residue is the two point-sampled quadratures' INTERIOR difference.

    The bars are RELATIVE to the residue this build measures, never to a
    remembered number: the two large windows must agree to better than a
    twentieth of the residue itself, and the undersized one must differ by at
    least 2x.  The plane is a fraction of the paraxial focal distance this
    build's own glass data gives, and the premise that it is short of the
    caustic is asserted through the library's own branch census.
    """
    f_par = _paraxial_focal_distance(_W)
    z = 0.63 * f_par
    dx = 3.00e-6
    read = {}
    for N in (192, 384, 512):
        _E, d = _call(_W, z, N=N, dx=dx)
        c = d.get('pixel_continuity')
        assert c is not None, (
            f'no reading at N = {N}, z = {z * 1e6:.1f} um '
            f"(decision={d.get('pixel_continuity_decision')!r})")
        read[N] = float(c)
        if N >= 384:
            assert d.get('n_branch_max') == 1, (
                f'the ray map is already multi-valued at z = {z * 1e6:.1f} um '
                f"(n_branch_max={d.get('n_branch_max')!r}); this ladder needs "
                'a plane short of the caustic -- move the fraction of the '
                'paraxial focal distance, do not skip')

    residue = abs(read[512] - 1.0)
    assert residue > 0.0, (
        'the reading is exactly 1 on this build, so there is no residue to '
        'attribute; re-open the question rather than passing')
    assert residue < 2.0e-3, (
        f'the converged reference itself moves by {residue:.3e}; the bar '
        'cannot be a tolerance on a reference that loose')
    # (a) the window does not move it
    assert abs(read[512] - read[384]) < 0.05 * residue, (
        f'growing the window from {384 * dx * 1e6:g} um to '
        f'{512 * dx * 1e6:g} um at a fixed pitch moved the reading by '
        f'{abs(read[512] - read[384]):.3e}, a fifth or more of the residue '
        f'{residue:.3e} -- on this build the residue DOES follow the window, '
        'so the window-edge attribution may hold after all and the '
        "constant's derivation should be re-measured")
    # (b) but an undersized window does
    assert abs(read[192] - 1.0) > 2.0 * residue, (
        f'a window too small to hold the field ({192 * dx * 1e6:g} um) reads '
        f'{abs(read[192] - 1.0):.3e} from 1, not the >= 2x of the converged '
        f'residue {residue:.3e} this claim needs; pick a window that really '
        'clips the field rather than skipping')
    # (c) and the PITCH does: the same window, twice the pixels
    _E, d = _call(_W, z, N=1024, dx=0.5 * dx)
    c_fine = d.get('pixel_continuity')
    assert c_fine is not None
    assert abs(float(c_fine) - 1.0) < residue, (
        f'halving the pitch over the same window moved the deviation from '
        f'{residue:.3e} to {abs(float(c_fine) - 1.0):.3e}; the residue is '
        'supposed to be the interior quadrature difference, which falls with '
        'the pitch')
