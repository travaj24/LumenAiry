"""WP-B7c round 3: the follow-ups the round-2 VERIFICATION left.

`fixes/VERIFY_WP-B7c_ROUND2.md` confirmed the pixel-halving arbiter's
mechanism, its placement, its bit identity and its cost, and left five
defects.  Round 3 answers them, and this file is the gate for what it changed.
The measurements are in
`fixes/WP-B7c_ROUND3_REPORT.md`; the probes and their JSON in
`validation/probe_wp_b7c_round3/` (642 oracle-scored planes over sixteen
optics, four of which no earlier round scored at all).

What is pinned here, and why each one needed pinning:

* **E5 / R3-3 -- the half-pitch lattice has ONE definition, and the
  convention is NESTING.**  The branch sum's fine render and the completion's
  half-pitch fill were placed by two copies of the same expression, which is a
  seam the two could drift apart at; they now both call
  ``half_pitch_centres``.  The convention that definition implements is that
  fine pixel ``2j`` sits exactly on coarse pixel ``j``, which is what bounds
  the reading at ~4 in the collapse limit and keeps the ratio's sampling noise
  below either render's own.  Its price is that the two Voronoi hulls differ
  by a quarter of a coarse pixel at the edges, so a converged render reads 1
  to O(1/N) and not exactly -- round 2 published "1 exactly, on any optic, at
  any plane, at any grid".  The hull-aligned alternative was built and
  measured this round and is strictly worse (the ~4 identity becomes
  unbounded: 7343.9 where the nested lattice reads 3.998).
* **E4 / R3-2 -- the Pearcey cusp route no longer labels the branch sum's
  reading as the cusp field's.**  On that route there is no half-pitch Pearcey
  field to compare against without a second cusp trace, so the number IS the
  branch sum's -- which the module's own claim-2 measurement shows can differ
  from the returned field's by 6 % at a plane whose field is right.  The label
  now says so.
* **E7 / R3-4 -- a fallback says it is a fallback.**  On the fallback route
  the returned field is the bright-side-only branch sum and the dominant error
  is the dark tail the completion did not build, which this arm cannot see:
  round 3 measured returned fields down to oracle fidelity 0.4639 with
  ``pixel_continuity`` 1.002, the launched-power bracket 1.005, both decisions
  ``'ok'`` and no warning.  Nothing already in the diagnostics orders that
  population (section 7 of the report tries nine candidates), so the
  diagnostics now state the scope instead of implying one.
* **E1 / R3-1 -- the bar clears the converged population's own spread.**  The
  quantity's "fixed reference" is what makes the bar a tolerance rather than a
  boundary, so the bar must sit outside the spread of the reading on planes
  whose render IS converged.  That spread is measured here, on the running
  build, and the bar is required to clear it by decades of that spread rather
  than by a remembered number.

No test here pins a number round 3 measured.  Every bar is derived on the
running build from that build's own geometry, and the two-sided arms fail --
never skip -- when the premise they need is absent.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.elements import _lens_traced_multibranch as _MB
from lumenairy.elements import _lens_traced_uniform as _U
from lumenairy.elements._lens_traced_multibranch import (
    _PIXEL_CONTINUITY_MAX,
    _PIXEL_CONTINUITY_MIN,
    half_pitch_centres,
)
from lumenairy.elements._lens_traced_uniform import (
    _PIXEL_CONTINUITY_SCOPES,
    apply_real_lens_traced_uniform,
)

# ---------------------------------------------------------------------------
# VERIFY-B7b's own N-BAF10 biconvex -- the optic the round-2 design decision is
# argued on and the one the shipped gate already runs, so this file adds no
# new prescription to the suite's cost.  Its paraxial focus is at ~1876 um and
# the interior fold opens at ~1571 um (measured by round 3's own geometry
# probe, validation/probe_wp_b7c_round3/geom_win.json), which is where the
# "converged" planes below come from: every one of them is at least a quarter
# of the paraxial focal distance short of the fold, so the render is converged
# for a GEOMETRIC reason and not because the arbiter said so.
# ---------------------------------------------------------------------------
_V_WL, _V_DX, _V_N, _V_W0 = 1.064e-6, 2.20e-6, 512, 330e-6
_V_FOLD_Z = 1679e-6            # inside the two-branch band, healthy
_V_FALLBACK_Z = 1870e-6        # past it: the completion declines
_V_CONVERGED_Z = (400e-6, 700e-6, 1000e-6, 1250e-6)


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


def _uni(z, monkeypatch=None, unbarred=False, **kw):
    """One completion call at ``z``, optionally with both refusal bars
    lifted so the reading and the field it would have refused come from the
    SAME call."""
    if unbarred:
        monkeypatch.setattr(_U, '_MB_PIXEL_CONTINUITY_MAX', float('inf'))
        monkeypatch.setattr(_U, '_MB_PIXEL_CONTINUITY_MIN', 0.0)
        monkeypatch.setattr(_U, '_MB_POWER_RATIO_MAX', float('inf'))
        monkeypatch.setattr(_U, '_MB_POWER_RATIO_MIN', 0.0)
    E = _gauss(_V_N, _V_DX, _V_W0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return apply_real_lens_traced_uniform(
            E, prescription=_v_presc(), wavelength=_V_WL, dx=_V_DX,
            output_plane_distance=z, return_diagnostics=True, **kw)


# ===========================================================================
# 1. E5 / R3-3 -- the lattice
# ===========================================================================
def test_b7c3_the_half_pitch_lattice_nests_on_the_callers_own():
    """The fine lattice CONTAINS the coarse one, exactly.

    The reading is a ratio of two point-sampled estimators of the same area
    integral.  Placing the fine lattice so that pixel ``2j`` falls exactly on
    coarse pixel ``j`` is what makes those two estimators share half their
    sample points, and two consequences of the module's own follow from it and
    from nothing else: a mapped triangle that catches a coarse centre catches
    the coincident fine centre too, so the reading is BOUNDED by 4 in the
    collapse limit (the ``~4 per halving`` identity VERIFY-WP-B7c measured by
    hand on four optics), and the two estimators are positively correlated, so
    the ratio's noise is far below either one's.

    Three exact statements, on integers and halves so they are build-free:

    * every EVEN fine centre is a coarse centre, to the bit;
    * the fine pitch is exactly half the coarse one;
    * the fine lattice spans the coarse one (it starts on it and ends half a
      coarse pixel past it, which is the hull mismatch
      ``_HALF_PITCH_CENTRE_OFFSET`` documents and this round measured).

    A regression of the convention -- the hull-aligned offset, or any other --
    breaks the first outright.
    """
    for N, dx in ((8, 1.0), (16, 2.5e-6), (512, 2.20e-6), (37, 1.3e-6)):
        c = (np.arange(N) - N / 2.0) * dx
        f = half_pitch_centres(N, dx)
        assert f.shape == (2 * N,), (N, f.shape)
        # NESTED, to the bit: no tolerance, because both lattices are the
        # same expression evaluated at the same points
        assert np.array_equal(f[0::2], c), (
            N, dx, "the half-pitch lattice no longer nests on the "
            "caller's: the ~4-per-halving identity and the ratio's low "
            "variance both rest on fine pixel 2j being coarse pixel j")
        # every tolerance below is the floating-point representation error of
        # the lattice ITSELF -- a few ULP of its own extent -- and not a
        # fitted number: these identities are exact in the reals.
        tol = 8.0 * np.finfo(float).eps * (float(np.abs(f).max()) + dx)
        d = np.diff(f)
        assert np.max(np.abs(d - 0.5 * dx)) <= tol, (N, dx, d[:3])
        assert abs(f[0] - c[0]) <= tol
        assert abs(f[-1] - (c[-1] + 0.5 * dx)) <= tol


def test_b7c3_the_completion_fills_the_half_pitch_render_on_its_own_lattice(
        monkeypatch):
    """The branch sum's fine render and the completion's half-pitch fill take
    their geometry from ONE definition.

    The consumer completes ``pixel_halved_field`` with the same fold
    parameters and then compares the two powers; if it builds its own
    ``(np.arange(2N) - N) dx/2`` grid for that array it is evaluating the
    completion at radii the render does not have, and the two halves of the
    ratio are taken on different geometry.  Pinned by PROVENANCE -- the
    lattice the completion uses is the one ``half_pitch_centres`` returns,
    for this call's own ``(N, dx)`` -- because the alternative (asserting a
    number) would pin a reading rather than the property.
    """
    seen = []
    real = _MB.half_pitch_centres

    def _rec(N, dx):
        seen.append((int(N), float(dx)))
        return real(N, dx)

    monkeypatch.setattr(_U, 'half_pitch_centres', _rec)
    _E, d = _uni(_V_FOLD_Z)
    assert d['reason'] == 'fold_ring' and not d['fell_back'], (
        'this build no longer takes the completion route at the pinned '
        f"plane (reason={d['reason']!r}, fell_back={d['fell_back']!r}); the "
        'provenance arm needs a plane that does -- extend the ladder rather '
        'than removing the claim')
    assert seen == [(_V_N, _V_DX)], (
        'the completion did not take its half-pitch lattice from '
        f'half_pitch_centres(N, dx); calls seen: {seen}')


def test_b7c3_the_arbiter_lattice_never_moves_the_returned_field(monkeypatch):
    """The half-pitch lattice is the ARBITER's alone.

    It decides a refusal; it must not touch the answer.  Two calls that differ
    only in ``_HALF_PITCH_CENTRE_OFFSET`` -- the shipped value and round 2's
    -- must return the same bytes, or the offset has leaked into the coarse
    render (which would make every field in the library depend on a constant
    introduced for a diagnostic).

    Two-sided: the same pair must also produce DIFFERENT readings, otherwise
    the constant is not reaching the fine render at all and the arm above is
    vacuous.
    """
    _E0, d0 = _uni(_V_FOLD_Z)
    monkeypatch.setattr(_MB, '_HALF_PITCH_CENTRE_OFFSET', -0.5)
    _E1, d1 = _uni(_V_FOLD_Z)
    a, b = np.asarray(_E0), np.asarray(_E1)
    assert a.dtype == b.dtype and a.shape == b.shape
    assert a.tobytes() == b.tobytes(), (
        'the half-pitch lattice moved the RETURNED field; it must move only '
        'the reading')
    c0, c1 = d0.get('pixel_continuity'), d1.get('pixel_continuity')
    assert c0 is not None and c1 is not None
    assert c0 != c1, (
        f'both lattices read {c0!r}: the constant is not reaching the fine '
        'render, so the bit-identity arm above proves nothing')


# ===========================================================================
# 2. E4 / R3-2 -- the Pearcey cusp route's label
# ===========================================================================
def test_b7c3_the_cusp_route_says_its_reading_is_of_the_branch_sum(
        monkeypatch):
    """DECISION: on the Pearcey cusp route the recorded reading IS the branch
    sum's, and the diagnostics say so in both the label and the scope.

    Round 2 recorded ``pixel_continuity_of = 'the Pearcey cusp field'`` while
    the number was ``mb_diag['pixel_continuity']`` -- the branch sum's.  By
    the module's own claim-2 argument that is the reading which "would refuse
    fields that are right" (1.0636 against 1.0020 at oracle fidelity 0.9878
    on VERIFY-B7b's own fixture), so the cusp route carries exactly the
    exposure the design decision removes, under a label saying it does not.

    The route is reached through the module's own dispatcher with the fold
    trace, the cusp trace and the Pearcey build stubbed: a real cusp ring
    needs an N = 1536 grid (``test_niche_r2_pearcey_cusp``'s end-to-end
    fixture), which is past the arbiter's own entry cap, so the reading would
    be ``None`` there and the label could not be read at all.  What is pinned
    is a labelling property of that branch, and the stub reaches the branch.
    """
    N, dx = 96, 2.0e-6
    E0 = _gauss(N, dx, 60e-6)
    base = np.ones((N, N), dtype=np.complex128)
    #: a value no other quantity in the call can produce, so the recorded
    #: reading is traceable to the branch sum and to nothing else
    sentinel = 1.0123456789

    def _fake_mb(E_in, **kw):
        diag = {'input_carrier': (0.0, 0.0), 'pixel_continuity': sentinel,
                'pixel_continuity_decision': 'ok', 'n_branch_max': 3,
                'power_ratio': 1.0, 'power_ratio_triangles': 1.0,
                'pixel_halved_field': None}
        return (base, diag) if kw.get('return_diagnostics') else base

    monkeypatch.setattr(_U, '_multibranch_render', _fake_mb)
    monkeypatch.setattr(_U, '_is_rotationally_symmetric', lambda *a, **k: True)
    monkeypatch.setattr(
        _U, '_trace_meridional_fold',
        lambda *a, **k: {'ok': False, 'reason': 'cusp_or_multiple',
                         'n_turn': 2})
    monkeypatch.setattr(
        _U, '_trace_meridional_cusp',
        lambda *a, **k: {'ok': True, 'r1': 1.0e-5, 'r2': 2.0e-5, 'x0': 0.0,
                         'gamma': 1.0, 'map_resid': 0.01})
    monkeypatch.setattr(_U, '_build_pearcey_cusp_field',
                        lambda E_mb, cusp, dx_: 2.0 * np.asarray(E_mb))

    presc = {'wavelength': _V_WL, 'surfaces': [], 'thicknesses': []}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E, d = apply_real_lens_traced_uniform(
            E0, prescription=presc, wavelength=_V_WL, dx=dx,
            output_plane_distance=1.0e-3, return_diagnostics=True)
    assert d['reason'] == 'cusp_ring' and not d['fell_back'], (
        f"the stub did not reach the Pearcey route: reason={d['reason']!r}")
    assert np.all(np.isfinite(E))
    # the number IS the branch sum's, to the bit
    assert d['pixel_continuity'] == sentinel, (
        f"the cusp route recorded {d['pixel_continuity']!r}; the branch sum's "
        f'reading was {sentinel!r}')
    # ... and the label and the scope both say which field it is of
    assert 'Pearcey' not in str(d['pixel_continuity_of']) or 'branch sum' in \
        str(d['pixel_continuity_of']), d['pixel_continuity_of']
    assert 'branch sum' in str(d['pixel_continuity_of']), (
        f"pixel_continuity_of reads {d['pixel_continuity_of']!r}; the number "
        'is the branch sum -- the label must not name the returned field')
    assert d['pixel_continuity_scope'] == 'underlying_branch_sum', (
        d['pixel_continuity_scope'])
    assert 'NOT of the field this call returns' in str(
        d['pixel_continuity_scope_note'])


# ===========================================================================
# 3. E7 / R3-4 -- a fallback says it is one
# ===========================================================================
def test_b7c3_a_fallback_reports_that_the_dark_tail_is_not_arbitrated(
        monkeypatch):
    """DECISION, two-sided on the SAME optic: the completion route reports
    that the reading arbitrates the returned field, and the fallback route
    reports that it arbitrates only that field's quadrature -- because the
    dark tail the completion did not build is absent and this arm cannot see
    it.

    Round 3 measured the consequence on 433 fallback planes: returned fields
    down to oracle fidelity 0.4639 with ``pixel_continuity`` 1.00201, the
    bracketed launched-power ratio 1.005, both decisions ``'ok'`` and no
    warning from either arm; and nine candidate readings already in these
    diagnostics, none of which orders that population.  Saying so is what the
    module can do honestly, and it is what this pins.

    Both premises are ASSERTED rather than skipped: if this build no longer
    puts one plane on each route the test fails with an instruction, because
    a premise gate that skips is the fail-open shape
    ``docs/TESTING_STANDARDS.md`` rule 4 forbids.
    """
    _Ef, df = _uni(_V_FOLD_Z)
    assert df['reason'] == 'fold_ring' and not df['fell_back'], (
        f"z = {_V_FOLD_Z * 1e6:g} um no longer takes the completion route "
        f"(reason={df['reason']!r}); move the pinned plane rather than "
        'dropping the arm')
    assert df['pixel_continuity_scope'] == 'returned_field', (
        df['pixel_continuity_scope'])
    assert df['pixel_continuity_of'] == 'the completed fold field'

    _Eb, db = _uni(_V_FALLBACK_Z, monkeypatch, unbarred=True)
    assert db['fell_back'], (
        f"z = {_V_FALLBACK_Z * 1e6:g} um no longer falls back "
        f"(reason={db['reason']!r}); the fallback arm needs a plane that "
        'does -- extend the ladder rather than removing the claim')
    assert db['pixel_continuity_scope'] == 'returned_field_quadrature_only', (
        db['pixel_continuity_scope'])
    assert db['pixel_continuity_of'] == 'the plain multibranch field'
    note = str(db['pixel_continuity_scope_note'])
    assert 'FALLBACK' in note and 'NOT arbitrated' in note, note
    # the two routes must not report the same scope, or the key says nothing
    assert db['pixel_continuity_scope'] != df['pixel_continuity_scope']


def test_b7c3_every_route_declares_a_scope_and_only_a_named_one():
    """The scope is an enum a consumer can branch on, every value is
    documented, and the three are distinct.

    Unconditional and build-free.  It is the structural half of the pin above:
    a fourth route added later without a scope raises inside ``_arbitrate``
    rather than reaching a caller with the key absent.
    """
    assert set(_PIXEL_CONTINUITY_SCOPES) == {
        'returned_field', 'returned_field_quadrature_only',
        'underlying_branch_sum'}
    notes = [str(v) for v in _PIXEL_CONTINUITY_SCOPES.values()]
    assert all(len(n) > 60 for n in notes)
    assert len(set(notes)) == 3


# ===========================================================================
# 4. E1 / R3-1 -- the bar against the converged population's own spread
# ===========================================================================
def test_b7c3_the_bar_clears_the_converged_readings_own_spread():
    """DERIVED on the running build: the bar sits decades of the CONVERGED
    population's own spread above 1.

    What makes this quantity usable as a tolerance rather than as a boundary
    between two moving populations is that it has a fixed reference -- a
    converged point-sampled quadrature deposits the same power at any pitch.
    Round 2 stated that reference as "**1 exactly**, on any optic, at any
    plane, at any grid"; it is not exact (VERIFY round 2, E5, and section 4 of
    the round-3 report), so what the bar has to clear is the reference's
    measured spread, not the reference.

    Measured here, on this build, on planes that are converged for a GEOMETRIC
    reason -- at least a quarter of the paraxial focal distance short of the
    interior fold, where the landing map is single-valued and far from its
    turning point -- so no reading selects the planes its own spread is then
    measured on.

    THE FACTOR IS DERIVED, not chosen for comfort.  On this ladder the four
    planes read 4.08e-05, 1.16e-04, 1.63e-04 and 1.82e-04 from 1 -- identical
    to the printed digit on win py3.14.6 / numpy 2.4.4 and on wsl py3.12.3 /
    numpy 2.4.6, measured 2026-09-19 -- so the shipped bar clears the worst of
    them by **330x**.  Requiring 50x asserts that the bar is decades above the
    reference's own spread, which is what makes it a tolerance on a known
    value rather than noise, while leaving 6.6x of headroom to the shipped
    value: a build whose spread were six times larger would still pass.
    Round 3 measured the same spread on 102 planes over seventeen optics
    (0.99941 .. 1.00044, rms 1.5e-4), so this ladder is not an outlier.

    The premise that the spread is small enough for the claim to mean anything
    is ASSERTED, not skipped.
    """
    devs = []
    for z in _V_CONVERGED_Z:
        _E, d = _uni(z)
        c = d.get('pixel_continuity')
        assert c is not None, (
            f'no reading at z = {z * 1e6:g} um '
            f"(decision={d.get('pixel_continuity_decision')!r}); the "
            'converged control needs one')
        assert d['pixel_continuity_scope'] in _PIXEL_CONTINUITY_SCOPES
        devs.append(abs(float(c) - 1.0))
    spread = max(devs)
    # the premise: these planes really are converged.  A spread of 2 % would
    # mean the "fixed reference" argument has failed on its own control, and
    # that is a failure of the claim, not a reason to skip.
    assert spread < 0.02, (
        f'the converged control itself spreads by {spread:.2e} about 1 '
        f'({[f"{d:.2e}" for d in devs]}); the bar cannot be a tolerance on a '
        'reference that moves by that much -- re-derive the reference')
    assert _PIXEL_CONTINUITY_MAX > 1.0 + 50.0 * spread, (
        f'the bar {_PIXEL_CONTINUITY_MAX} is within fifty times the converged '
        f'population\'s own spread ({spread:.2e} measured here, so the bar '
        f'clears it by {(_PIXEL_CONTINUITY_MAX - 1.0) / spread:.0f}x against '
        'the 330x this ladder measured), so it is no longer decades above the '
        'reference it is a tolerance on -- it is noise')
    # and two-sided: it is far BELOW the regime it exists to catch.  A
    # quadrature that has stopped being unbiased deposits a power proportional
    # to the pixel area and reads ~4 per halving, which is the mechanism; a
    # bar at half of that is not a classifier of anything.
    assert _PIXEL_CONTINUITY_MAX < 2.0, _PIXEL_CONTINUITY_MAX
    assert _PIXEL_CONTINUITY_MIN == 1.0 / _PIXEL_CONTINUITY_MAX


def test_b7c3_the_bar_is_above_the_converged_and_below_the_blown_up(
        monkeypatch):
    """The same claim as a DECISION on two real planes of one optic: the
    converged plane is returned and the blown-up one is refused, with the two
    readings separated by more than the bar's own distance to either.

    Premise-gated by ASSERTION: if this build's blow-up plane stops blowing
    up the test fails with an instruction to move the plane, because a skip
    here is exactly the shape that let a bar of 1.20 read green through the
    round-2 gate.
    """
    _E0, d0 = _uni(_V_CONVERGED_Z[-1])
    c0 = d0['pixel_continuity']
    assert c0 is not None and d0['pixel_continuity_decision'] == 'ok', (
        c0, d0['pixel_continuity_decision'])
    _E1, d1 = _uni(_V_FALLBACK_Z, monkeypatch, unbarred=True)
    c1 = d1['pixel_continuity']
    assert c1 is not None
    # the blow-up regime, as a basin and not a value
    assert c1 > 1.5 * _PIXEL_CONTINUITY_MAX, (
        f'the pinned blow-up plane reads {c1:.4f}, not clear of the bar '
        f'{_PIXEL_CONTINUITY_MAX} by the 1.5x this claim needs; move the '
        'plane to one that still blows up on this build rather than skipping')
    assert c0 < _PIXEL_CONTINUITY_MAX < c1
    assert (c1 / _PIXEL_CONTINUITY_MAX) > (_PIXEL_CONTINUITY_MAX / c0)
