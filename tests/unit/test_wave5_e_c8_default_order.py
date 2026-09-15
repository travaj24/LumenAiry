"""WAVE5-E item E2 -- a durable fail-before for the C8 inverse-support bound at
the SHIPPED decentred-fit order.

WHY THIS FILE EXISTS.  ``test_niche_c7_ray_density_halo_check.py`` and
``test_niche_c8_inverse_support_bound.py`` state
``decentred_fit_poly_order=10`` -- the pre-WP-A26 default -- because that is the
order their 768^2 ``_GHOST`` geometry manufactures its lobe at.  That is the
right layer for those files (the stimulus is stated, not inherited), but it left
nothing exercising the order the library actually ships.  VERIFY-WP-B14 F1
measured that the defect class IS still reachable at the shipped default (ratio
301x at ``alpha=3.0, cx=1.5 mm, z=12 mm`` and 3164x at
``alpha=3.5, cx=1.35 mm, z=13 mm``, 768^2) and declined to turn it into a test,
because a 27-cell neighbourhood around the strongest cell reads ratio 1.00 in 23
of 27 -- the S1/S5 knife edge ``docs/TESTING_STANDARDS.md`` forbids pinning as a
single cell -- and the sweep costs ~14 min.

WHAT WAS SEARCHED (2026-09-15, Windows py3.14.6 / numpy 2.4.4,
``validation/probe_wave5_e/probe_e2_search.py``, per-arm JSON beside it).  A
432-cell sweep over ``alpha`` x ``cx`` x ``z`` x ``fit_radius_beam_factor`` at
``n = 256`` and ``n = 512``, plus four fine scans, asked whether any GEOMETRY
parameter the trip is monotone in exists:

===========================  ============================================
``fit_radius_beam_factor``   1.25 / 1.50 / 2.00 / 2.50 at one cell reads
                             ratio 1.0 / 1793.9 / 1.0 / 1.3
``cx`` (the decentre)        1.30 .. 1.70 mm in 0.10 mm steps reads
                             3.5 / 1.0 / 1793.9 / 1.0 / 170.7
``alpha`` (the aberration)   3.45 / 3.50 / 3.55 reads 1.0 / 1793.9 / 1.0
``n`` (the sampling)         128 / 192 / 256 / 512 / 768 at the cell below
                             reads 1.0 / 1.0 / 12.9 / 490.9 / 1.0
===========================  ============================================

-- none of them is.  Whether the order-16 fit's extrapolated inverse folds back
into the bright beam is a chaotic function of which traced samples the ray grid
happens to contain, which is exactly why F1 refused a single cell.

THE PARAMETER THE EFFECT IS MONOTONE IN is the HALO ANNULUS RADIUS, and it is
monotone for a reason rather than by luck: the C8 bound zeroes exit pixels with
no traced ray behind them, so the further out the annulus sits the larger the
fraction of it outside the traced footprint and the more completely the bound
must empty it.

THE CELL.  Of 216 cells at ``n = 256``, 47 trip in three or more annuli; of
those, the one below is the one that ALSO trips at ``n = 512`` -- i.e. the
cheapest sampling at which the stimulus survives a doubling.  Measured
suppression ``on/off`` per annulus (``validation/probe_wave5_e/
e2_radius_nstable_win32_314.json``):

    r >     2.0 w   2.5 w   3.0 w   3.5 w   4.0 w   4.5 w   5.0 w
    n=256   0.596   0.596   7.7e-2  9.0e-3  9.0e-3  2.4e-5  1.8e-7
    n=512   0.216   2.3e-2  2.0e-3  1.7e-4  1.3e-5  1.0e-6  7.5e-8

so five of the seven rungs clear 10x at 256 and seven of seven do at 512, both
monotone, and the 256^2 pair costs ~2 s against the ~14 min the 768^2 sweep
took.  The geometry is the ``_GHOST`` optic re-sampled onto 256 cells over the
same 19.2 mm physical extent.

NOTHING HERE STATES A FIT ORDER.  The calls run at whatever
``lumenairy.elements._lens_traced._DECENTRED_FIT_POLY_ORDER`` ships, which is
the point of the file.  If a future re-derivation of that default makes the
class unreachable, the premise gate says so with every candidate's reading --
AND first proves the bound is not simply dead, by re-running the published
order-10 ``_GHOST`` control, which must still read 51.5x.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_traced as LT

_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL

#: The C8 ``_GHOST`` optic at its own 768-cell sampling.  Used ONLY by the
#: order-10 control on the skip path.
_GHOST = dict(n=768, dx=25e-6, w=1.5e-3, rc=-0.15, alpha=5.0,
              r1=150e-3, r2=-150e-3, th=4e-3, z=6e-3, ap=18e-3)

#: The same optic re-sampled onto ``_N`` cells over the SAME 19.2 mm extent --
#: the same optic and the same halo annulus, one third of the sampling.
_N = 256
_BASE = dict(_GHOST)
_BASE['n'] = _N
_BASE['dx'] = (_GHOST['n'] * _GHOST['dx']) / _N

#: Candidate stimuli, each measured to trip at the SHIPPED default order on
#: 2026-09-15.  The list is a LADDER, not a pin: the first candidate that
#: reaches the bar is used and the rest are not evaluated.  ``n_stable`` marks
#: the one whose stimulus survives the ``n`` doubling to 512 (the only one of
#: the 47 tripping cells at 256 that does), which is why it is first.
_CANDIDATES = (
    dict(alpha=3.0, cx=1.25e-3, z=12e-3, frbf=2.0, n_stable=True),   # 12.9x
    dict(alpha=3.5, cx=1.50e-3, z=12e-3, frbf=1.5, n_stable=False),  # 1793.9x
    dict(alpha=3.5, cx=1.70e-3, z=12e-3, frbf=1.5, n_stable=False),  # 170.7x
    dict(alpha=3.5, cx=1.50e-3, z=12e-3, frbf=1.6, n_stable=False),  # 158.1x
)

#: The ladder's free parameter: the inner radius of the halo annulus, in input
#: beam widths.  Seven rungs; >= 3 must trip for the pin to be a ladder.
_FACTORS = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)

#: THE BAR.  A rung TRIPS when the bound removes a factor of 10 of amplitude
#: from the annulus.  Gap below: where the bound is inert the two fields are
#: BIT-IDENTICAL, so the inert reading is exactly 1.0 and not 1.0000001 -- every
#: one of the 385 inert cells of the 432-cell sweep reads exactly 1.0.  Gap
#: above: the weakest tripping rung of the cell this file selects reads 12.9x at
#: n = 256 and 490.9x at n = 512.  So the bar sits one decade above an exact
#: identity and 1.3x to 49x below the signal.
_TRIP = 10.0

#: The published order-10 control on the ``_GHOST`` geometry: the bound removes
#: 4.594672922387141e-02 -> 8.913411901995892e-04, ratio 51.5478.  Reproduced
#: bit-for-bit by VERIFY-WP-B14 in three separate ``git archive`` trees and
#: again here on 2026-09-15 (4.595e-02 / 8.913e-04, 51.5).  Run only on the
#: skip path, to tell "the stimulus moved" from "the bound is dead".
_CONTROL_TRIP = 10.0


def _flat():
    return {'radius': np.inf, 'glass_before': 'air', 'glass_after': 'air',
            'conic': 0.0, 'radius_y': None, 'conic_y': None,
            'aspheric_coeffs': None, 'aspheric_coeffs_y': None}


def _surf(r, gb, ga):
    d = _flat()
    d['radius'], d['glass_before'], d['glass_after'] = r, gb, ga
    return d


def _singlet(s, glass='N-BK7'):
    return {'name': 'wave5e_c8', 'aperture_diameter': s['ap'],
            'surfaces': [_surf(s['r1'], 'air', glass),
                         _surf(s['r2'], glass, 'air'), _flat()],
            'thicknesses': [s['th'], s['z']]}


def _field(s, cx):
    """Gaussian on a converging carrier sphere plus an ``alpha (r/w)^4``
    residual -- the construction the C6, C7 and C8 fixtures all use."""
    n, dx, w, rc, alpha = s['n'], s['dx'], s['w'], s['rc'], s['alpha']
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    sg = 1.0 if rc > 0 else -1.0
    rho = np.sqrt((X - cx) ** 2 + Y ** 2 + rc * rc)
    Wc = sg * (rho - abs(rc))
    r2 = (X - cx) ** 2 + Y ** 2
    a = (alpha / _K0) * (r2 / (w * w)) ** 2
    return (np.exp(-r2 / (w * w))
            * np.exp(1j * _K0 * (Wc + a))).astype(np.complex128)


def _call(s, *, bound, cx, frbf, order=None):
    """One element call, at the SHIPPED default fit order unless ``order`` is
    given (only the control passes one)."""
    E = _field(s, cx)
    presc = _singlet(s)
    old = (LT.REMAP_STATIONARY_PHASE_LAUNCH,
           LT.REMAP_STATIONARY_PHASE_FIT_GUARD,
           LT.REMAP_INVERSE_SUPPORT_BOUND)
    LT.REMAP_STATIONARY_PHASE_LAUNCH = True
    LT.REMAP_STATIONARY_PHASE_FIT_GUARD = True
    LT.REMAP_INVERSE_SUPPORT_BOUND = bool(bound)
    kw = dict(wavelength=_WL, amplitude_model='ray_density',
              preserve_input_phase='remap', remap_sampling='full',
              parallel_amp=False, on_undersample='silent',
              on_noncollimated='silent', on_aperture_beam='silent',
              ray_subsample=4, fit_radius_beam_factor=float(frbf),
              dx=s['dx'])
    if order is not None:
        kw['decentred_fit_poly_order'] = int(order)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.asarray(la.apply_real_lens_traced(
                E, prescription=presc, carrier=s['rc'], **kw))
    finally:
        (LT.REMAP_STATIONARY_PHASE_LAUNCH,
         LT.REMAP_STATIONARY_PHASE_FIT_GUARD,
         LT.REMAP_INVERSE_SUPPORT_BOUND) = old


def _radii(s, cx):
    ax = (np.arange(s['n']) - s['n'] // 2) * s['dx']
    X, Y = np.meshgrid(ax, ax)
    return np.hypot(X - cx, Y)


def _ladder(Foff, Fon, s, cx):
    """``(factor, off, on, suppression)`` per annulus, each amplitude
    normalised by its OWN field's peak so the ladder reads a shape, not a
    scale."""
    R = _radii(s, cx)
    aoff, aon = np.abs(Foff), np.abs(Fon)
    poff = float(aoff.max()) or 1.0
    pon = float(aon.max()) or 1.0
    rows = []
    for f in _FACTORS:
        m = R > f * s['w']
        if not m.any():
            continue
        o = float(aoff[m].max()) / poff
        b = float(aon[m].max()) / pon
        rows.append((f, o, b, (b / o) if o > 0.0 else 1.0))
    return rows


def _evaluate(cand, n=_N):
    s = dict(_BASE)
    s['n'] = int(n)
    s['dx'] = (_GHOST['n'] * _GHOST['dx']) / float(n)
    s['alpha'], s['z'] = cand['alpha'], cand['z']
    Foff = _call(s, bound=False, cx=cand['cx'], frbf=cand['frbf'])
    Fon = _call(s, bound=True, cx=cand['cx'], frbf=cand['frbf'])
    rows = _ladder(Foff, Fon, s, cand['cx'])
    trips = [r for r in rows if r[3] > 0.0 and (1.0 / r[3]) >= _TRIP]
    return s, Foff, Fon, rows, trips


def _fmt(rows):
    return '\n  '.join(
        'r > %.1f w: off %.4e  on %.4e  ratio %.4g'
        % (f, o, b, (1.0 / sup) if sup > 0 else float('inf'))
        for f, o, b, sup in rows)


@pytest.fixture(scope='module')
def stimulus():
    """The first candidate that reaches the bar, or the reason none did.

    Module-scoped so the ladder is traced ONCE for the whole file: the pair of
    element calls at 256^2 is the entire cost of the first three tests (~2 s on
    the reference build, against ~14 min for the 768^2 sweep F1 ran).
    """
    readings = []
    for cand in _CANDIDATES:
        s, Foff, Fon, rows, trips = _evaluate(cand)
        readings.append((cand, rows, len(trips)))
        if len(trips) >= 3:
            return {'cand': cand, 's': s, 'Foff': Foff, 'Fon': Fon,
                    'rows': rows, 'trips': trips, 'readings': readings}
    return {'cand': None, 'readings': readings}


def _require(stim):
    """Premise gate.  Before skipping, PROVE the bound is not simply dead."""
    if stim['cand'] is not None:
        return
    Coff = _call(_GHOST, bound=False, cx=0.0, frbf=2.0, order=10)
    Con = _call(_GHOST, bound=True, cx=0.0, frbf=2.0, order=10)
    crows = _ladder(Coff, Con, _GHOST, 0.0)
    cbest = max((1.0 / r[3]) if r[3] > 0 else float('inf') for r in crows)
    assert cbest >= _CONTROL_TRIP, (
        'the C8 support bound removes NOTHING anywhere: the default-order '
        'ladder found no stimulus on %d candidates AND the published order-10 '
        'control on the _GHOST geometry -- which reads 51.5x here and in three '
        'of VERIFY-WP-B14\'s git-archive trees -- reads only %.4g.  That is '
        'the bound being dead, not the stimulus moving, so this is a hard '
        'failure and not a skip.\n  %s'
        % (len(_CANDIDATES), cbest, _fmt(crows)))
    lines = []
    for cand, rows, ntrip in stim['readings']:
        best = max(((1.0 / r[3]) if r[3] > 0 else 1.0) for r in rows) \
            if rows else 1.0
        lines.append(
            'alpha=%.2f cx=%.3f mm z=%.1f mm frbf=%.2f -> %d rungs at >= %gx, '
            'best ratio %.4g'
            % (cand['alpha'], cand['cx'] * 1e3, cand['z'] * 1e3,
               cand['frbf'], ntrip, _TRIP, best))
    pytest.skip(
        'premise absent: the C8 defect class is not reachable at the shipped '
        'decentred_fit_poly_order (%d) on any of the %d candidate stimuli, so '
        'there is no manufactured lobe for the bound to remove and this ladder '
        'would assert nothing.  The bound itself is ALIVE -- the order-10 '
        '_GHOST control reads %.4g against a bar of %g -- so this is the fit '
        'default having moved, which is a real and welcome outcome.  Every '
        'candidate\'s reading:\n  %s\nRe-run '
        'validation/probe_wave5_e/probe_e2_search.py to find the new reachable '
        'cells and replace _CANDIDATES.'
        % (int(LT._DECENTRED_FIT_POLY_ORDER), len(_CANDIDATES), cbest,
           _CONTROL_TRIP, '\n  '.join(lines)))


def test_the_bound_removes_manufactured_light_at_the_shipped_fit_order(
        stimulus):
    """THE FAIL-BEFORE, at the order the library ships, as a LADDER.

    At least three annuli must trip -- the bound must remove a factor of
    :data:`_TRIP` of amplitude from each -- so the claim does not rest on the
    single cell VERIFY-WP-B14 F1 rightly refused to pin.  Measured on the
    reference build at ``n = 256``: 5 of 7 rungs trip, at 12.9x to 5.6e+06x.

    Every number in the message is read at run time, so a build whose ladder is
    shorter says exactly how much shorter.
    """
    _require(stimulus)
    rows, trips, cand = stimulus['rows'], stimulus['trips'], stimulus['cand']
    assert len(trips) >= 3, (
        'the default-order ladder must trip in at least 3 annuli, not 1 -- a '
        'single cell is the shape TESTING_STANDARDS forbids.  Got %d of %d at '
        'alpha=%.2f cx=%.3f mm z=%.1f mm frbf=%.2f, n=%d:\n  %s'
        % (len(trips), len(rows), cand['alpha'], cand['cx'] * 1e3,
           cand['z'] * 1e3, cand['frbf'], stimulus['s']['n'], _fmt(rows)))


def test_the_suppression_is_monotone_in_the_annulus_radius(stimulus):
    """THE LADDER'S OWN SHAPE, which is what makes it durable.

    The C8 bound zeroes exit pixels with no traced ray behind them, so the
    further out the annulus the larger the fraction of it outside the traced
    footprint and the more completely the bound must empty it.  The suppression
    ``on/off`` must therefore be non-increasing in the radius -- a property of
    the mechanism, not of the cell.  Measured on the reference build:
    5.96e-01 -> 1.79e-07 across seven rungs at ``n = 256`` and
    2.16e-01 -> 7.52e-08 at ``n = 512``.

    A build that tripped the rungs in a scattered order would pass the
    fail-before above and fail here, which is the point: that would be a
    coincidence, not the bound doing its job.
    """
    _require(stimulus)
    rows = stimulus['rows']
    sup = [r[3] for r in rows]
    assert len(sup) >= 3, f'the ladder needs >= 3 rungs; got {len(sup)}'
    # One part in 1e9 of slack -- the round-off of a ratio of two amplitudes
    # read from the same pair of fields, far below the decade a rung moves by.
    bad = [i for i in range(len(sup) - 1) if sup[i + 1] > sup[i] * (1 + 1e-9)]
    assert not bad, (
        'the bound\'s suppression must be non-increasing in the annulus '
        'radius; it rises at rung(s) %s:\n  %s' % (bad, _fmt(rows)))


def test_the_bound_is_subtractive_and_leaves_the_traced_core_alone(stimulus):
    """THE OTHER SIDE, on the same two fields, at no extra cost.

    The bound is a SUBTRACTION and nothing else: it must never raise total
    power, and it must not touch a pixel inside the beam, where every exit cell
    has traced data behind it.  Without this arm the ladder above would also
    pass for a bound that simply zeroed the whole field outside some radius,
    which is not what C8 claims to do.
    """
    _require(stimulus)
    Foff, Fon, s, cand = (stimulus['Foff'], stimulus['Fon'], stimulus['s'],
                          stimulus['cand'])
    p_off = float((np.abs(Foff) ** 2).sum())
    p_on = float((np.abs(Fon) ** 2).sum())
    assert p_on <= p_off * (1.0 + 1e-12), (
        f'the support bound must be power-subtractive; power went '
        f'{p_off!r} -> {p_on!r} ({p_on / p_off - 1.0:.3e} relative)')
    core = _radii(s, cand['cx']) <= s['w']
    assert core.any(), 'premise: the core mask must select pixels'
    d = float(np.max(np.abs(Foff[core] - Fon[core])))
    scale = float(np.abs(Foff[core]).max()) or 1.0
    assert d / scale < 1e-12, (
        f'the support bound must not touch a pixel inside one beam width, '
        f'where every exit cell has traced data behind it; it moved '
        f'{d / scale:.3e} of the core peak')


def test_the_stimulus_survives_a_doubling_of_the_sampling(stimulus):
    """WHY ``n = 256`` AND NOT SOMETHING SMALLER OR LUCKIER.

    The cell this file selects was chosen out of the 47 cells (of 216 swept at
    ``n = 256``) that trip in three or more annuli, because it is the one that
    ALSO trips at ``n = 512``.  That doubling is the durability statement: the
    stimulus is a property of the optic and not of one sampling.  Measured:
    ratio 12.9x at 256 and 490.9x at 512 beyond three beam widths, with 5 and 7
    monotone rungs.  Below 256 (128, 192) and at 768 the same cell is inert,
    which is the chaotic ``n`` dependence this file's header documents -- so
    256 is the SMALLEST sampling that survives a doubling, not merely a
    sampling that works.

    Costs one extra 512^2 pair (~5 s).  When a FALLBACK candidate had to be
    used -- none of which is n-stable, by measurement -- this reports that
    instead of asserting it.
    """
    _require(stimulus)
    cand = stimulus['cand']
    if not cand.get('n_stable'):
        pytest.skip(
            'the n-stable candidate did not reach the bar on this build, so a '
            'fallback was selected (alpha=%.2f cx=%.3f mm z=%.1f mm '
            'frbf=%.2f); none of the fallbacks trips at 2N, by measurement, so '
            'there is nothing to assert here.  The first candidate\'s own '
            'reading is in the fail-before test\'s message.'
            % (cand['alpha'], cand['cx'] * 1e3, cand['z'] * 1e3,
               cand['frbf']))
    _s2, _off2, _on2, rows2, trips2 = _evaluate(cand, n=2 * _N)
    assert len(trips2) >= 3, (
        'the stimulus must survive the sampling doubling %d -> %d: at least 3 '
        'annuli must still trip at >= %gx.  Got %d of %d:\n  %s'
        % (_N, 2 * _N, _TRIP, len(trips2), len(rows2), _fmt(rows2)))
    sup2 = [r[3] for r in rows2]
    bad = [i for i in range(len(sup2) - 1)
           if sup2[i + 1] > sup2[i] * (1 + 1e-9)]
    assert not bad, (
        'the suppression must stay monotone in the annulus radius at 2N too; '
        'it rises at rung(s) %s:\n  %s' % (bad, _fmt(rows2)))
