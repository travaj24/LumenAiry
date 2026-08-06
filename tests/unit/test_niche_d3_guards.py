"""Chain guard rails -- niche D3 (roadmap
ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27 P3 + P5).

Why they exist.  Every failure guarded here is a SILENT one -- the run
completes, returns a populated array, and the number looks like an answer:

* **P3, multi-congruence input.**  ``propagate_traced_carrier_chain``
  propagates ONE congruence; ``apply_real_lens_traced``'s entrance->exit map
  assumes one congruence per exit pixel and names "comparable-power beams at
  well-separated angles (post-DOE at large split)" as the excluded case.  At
  v5.28 the design-121 32-order fan was pushed through MULTIPLEXED anyway and
  produced a populated, credible-looking frame lattice whose per-frame power
  was scrambled (0.47 +/- 0.51 % against a design 2.78 %/frame).  Nothing
  raised and nothing warned.
* **P5a, exit-NA near miss.**  ``final_leg='auto'`` flips between the exact
  and the PARAXIAL focus readout at ``na_exact_threshold``, and the paraxial
  side is ~200 rad of wavefront wrong at a design-121-class exit NA, so a
  design sitting near the threshold gets moved across it by one beam-size
  change with no symptom.  Design 121 itself is NOT such a design: its
  measured ``na_exit`` (``w_entrance`` / ``|R_out|``, the quantity the router
  actually branches on) is **0.405**, 170 % above the 0.15 default -- matching
  ``AUDIT_TRACED_CARRIER_CHAIN_2026_07_21.md``'s "last leg at NA ~ 0.46,
  R_out = -7.71 mm".  The 0.152 quoted for design 121 elsewhere is its
  geometric aperture/EFL system NA, a different quantity.  These pins are
  therefore expressed RELATIVE to the synthetic relay's own measured
  ``na_exit`` and encode no design-121 margin.
* **P5b, RAM-capped readout.**  The cap degrades the fine grid and labels the
  result "RESOLUTION-LIMITED (non-converged)" in a ``RuntimeWarning`` -- easy
  to lose in an unattended batch log, and the degraded number reads like a
  converged one.
* **P5c, ``rs_fine`` clamp.**  When the capped ``dx_fine`` is coarser than the
  chain's physical ray pitch, the F-C pitch-preservation contract stops
  holding (measured 5.25x at the N=28672 / ``n_fine_cap``=16384 design-121
  condition) and the final leg fits its OPL on a coarser ray lattice than the
  rest of the chain.

WHICH DETECTOR P3 REUSES, and why it takes TWO.  No new estimator is written
here.  The gate calls the two SHIPPED ones:

* ``_carrier_residual_rms`` (``elements/_lens_traced.py``) -- the wrapping-safe
  nearest-neighbour phase-increment rms in RADIANS, the discriminator that
  DEFINES the documented ``_NONCOLLIMATED_RESID_THRESH = 0.02 rad`` envelope;
* ``_tilt_dispersion`` (``propagators/fga.py``) -- the spread of the local
  wavevector about its per-region (amplitude-weighted, sigma ~2 px Gaussian)
  mean, i.e. the measurement ``apply_real_lens_universal`` already routes
  multi-valued fields on, compared against fga's own
  ``multivalued_threshold = 0.06``.

The wavefront-aware audit's caveat is that a wrapped nearest-neighbour gradient
UNDER-REPORTS when the content aliases or interferes -- so the gate must not be
built on it alone.  ``test_first_detector_alone_under_reports_a_resolved_fan``
is the measured proof of exactly that, and is the FAIL-BEFORE for the second
detector: on a 2x2 order fan at +-23 mrad with the fringes well resolved,
``_carrier_residual_rms`` reads 0.000000 rad (it would pass a 0.02 rad gate
untouched) while ``_tilt_dispersion`` reads 0.0149 rad.  On a single tilted
beam the roles reverse exactly.

HOW ``_tilt_dispersion``'S READING IS USED, and why neither adjustment is
cosmetic (both are fixes to measured defects in this gate's first cut; the
derivations and the full battery live in the "P3" note in
``propagators/carrier.py``):

* it is taken at ``na=1`` (raw rms direction cosine, radians) and then
  multiplied by ``sqrt(lambda / dx)``.  The raw reading falls as sqrt(dx) on a
  crossing congruence -- two equal beams superpose to a REAL cosine times one
  carrier, so the entire signal sits in the vanishing-amplitude pixels at the
  fringe nulls -- and the detector therefore went BLINDER as the grid got
  finer.  ``test_the_detector_does_not_go_blind_as_the_grid_gets_finer`` is the
  FAIL-BEFORE: a 121-class 8x4 fan goes SILENT on the pre-canonical score at
  dx0 = 0.25 um, the production pitch roadmap P4 names.
* it is divided by a FIXED reference NA (0.15), NOT by the first group's
  ``fga._system_na``.  That quantity saturates at 0.375 and floors at 0.03125,
  and it describes the first LENS, not the field handed in -- so it made the
  verdict a function of the first lens's f-number in BOTH directions.
  ``test_the_verdict_does_not_depend_on_the_first_groups_f_number`` is the
  two-sided FAIL-BEFORE.

Everything here is SELF-CONTAINED: synthetic N-BK7 singlets built inline, no
prescription asset, no ``.zmx``.  The analytic oracle for the multi-congruence
inputs is the closed-form two-beam superposition (whose crossing half-angle is
known exactly by construction) and, for the wrongness claim, an INDEPENDENT
per-congruence recombination through ``propagate_traced_carrier_chain_multi``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
import lumenairy.elements._lens_traced as _lens_traced
from lumenairy.elements._lens_traced import (
    _NONCOLLIMATED_RESID_THRESH,
    _carrier_residual_rms,
)
from lumenairy.propagators.carrier import (
    _MULTI_CONGRUENCE_MV_THRESH,
    _MULTI_CONGRUENCE_NA_REF,
    _chain_entry_congruence_stats,
    _fine_trace_group_exit,
    _memory_bounded_n_fine,
    _paraxial_group_r_out,
)
from lumenairy.propagators.fga import _system_na, _tilt_dispersion

_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL


# ---------------------------------------------------------------------------
# synthetic optics (shared with the D1/D2 pins' construction, written inline)
# ---------------------------------------------------------------------------
def _singlet(R1, R2, d, glass, ap, name='s'):
    surfaces = [
        {'radius': R1, 'glass_before': 'air', 'glass_after': glass,
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
        {'radius': R2, 'glass_before': glass, 'glass_after': 'air',
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]
    return {'name': name, 'aperture_diameter': ap,
            'surfaces': surfaces, 'thicknesses': [d]}


_GAP = 25e-3


def _relay_groups(ap=10e-3):
    gA = _singlet(60e-3, -60e-3, 3.0e-3, 'N-BK7', ap, 'gA')
    gB = _singlet(60e-3, -60e-3, 3.0e-3, 'N-BK7', ap, 'gB')
    return [{'prescription': gA, 'gap_before': 0.0},
            {'prescription': gB, 'gap_before': _GAP}]


def _grid(n, dx):
    x = (np.arange(n) - n // 2) * dx
    return np.meshgrid(x, x, indexing='xy')


def _gauss(n, dx, w):
    X, Y = _grid(n, dx)
    return np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)


def _runtime_msgs(rec):
    return [str(r.message) for r in rec
            if issubclass(r.category, RuntimeWarning)]


def _p3_msgs(rec):
    """Only the P3 gate's own messages (the element emits its own
    non-collimated diagnostics on the same inputs -- those are pre-existing
    and are not what this module pins)."""
    return [m for m in _runtime_msgs(rec)
            if 'single-congruence envelope' in m]


# ===========================================================================
# P3 -- the DETECTOR.  Pure measurement, no propagation: fast, and it is the
# fail-before evidence for "why two detectors".
# ===========================================================================
_DN, _DDX, _DW = 512, 1.0e-6, 60e-6      # fringes at +-23 mrad span ~28 px


def _detector_field(kind):
    X, Y = _grid(_DN, _DDX)
    G = _gauss(_DN, _DDX, _DW)
    if kind == 'clean':
        return G
    if kind == 'aberrated':                     # smooth r^4, ~8 rad p-v
        return G * np.exp(1j * _K0 * (X ** 2 + Y ** 2) ** 2 / (8 * 0.05 ** 3))
    if kind == 'tilted':                        # ONE congruence at 46 mrad
        return G * np.exp(1j * _K0 * 0.046 * X)
    if kind == 'diverging':                     # aliased spherical carrier
        return G * np.exp(1j * _K0 * (X ** 2 + Y ** 2) / (2 * 2e-3))
    if kind == 'two_beam':                      # TWO congruences, +-23 mrad
        return (G * np.exp(1j * _K0 * 0.023 * X)
                + G * np.exp(-1j * _K0 * 0.023 * X))
    if kind == 'fan_2x2':                       # FOUR congruences, +-23 mrad
        out = np.zeros_like(G)
        for sx in (-1, 1):
            for sy in (-1, 1):
                out = out + G * np.exp(1j * _K0 * 0.023 * (sx * X + sy * Y))
        return out
    raise AssertionError(kind)


@pytest.mark.parametrize('kind', ['clean', 'aberrated', 'tilted', 'diverging'])
def test_single_valued_inputs_sit_far_below_the_multivalued_cutoff(kind):
    """Every SINGLE-valued input -- flat, strongly aberrated, tilted, and a
    strongly diverging beam whose carrier ALIASES on the grid -- must read
    essentially zero on the multi-valuedness detector.  This is the false-
    positive side of the gate: it is what keeps a clean congruence silent."""
    mv_rad = _tilt_dispersion(_detector_field(kind), _DDX, _DDX, _WL, 1.0)
    assert mv_rad < 1e-3, (kind, mv_rad)


@pytest.mark.parametrize('kind,expect_rad', [('two_beam', 1.0e-2),
                                             ('fan_2x2', 1.4e-2)])
def test_multi_congruence_inputs_are_an_order_above_that(kind, expect_rad):
    """A genuine superposition of comparable-power beams at +-23 mrad reads
    >= 1e-2 rad on the SAME detector -- a >10x separation from the
    single-valued ceiling above, which is what makes the cutoff insensitive."""
    mv_rad = _tilt_dispersion(_detector_field(kind), _DDX, _DDX, _WL, 1.0)
    assert mv_rad > expect_rad, (kind, mv_rad)


def test_first_detector_alone_under_reports_a_resolved_fan():
    """FAIL-BEFORE for the second detector, and the measured form of the
    wavefront-aware audit's caveat.

    Two equal beams crossing at +-23 mrad interfere to a REAL cosine times a
    single mean-direction carrier, so every nearest-neighbour phase increment
    reports the MEAN tilt and the pi jumps sit in the amplitude nulls: the
    wrapped-gradient angular-spread detector reads EXACTLY zero and would sail
    through a 0.02 rad gate.  A gate built on it alone -- the obvious
    single-detector design -- therefore MISSES the exact input class P3
    exists for.  ``_tilt_dispersion`` reads 0.0149 rad on the same field.

    The converse is pinned too: on a single 46 mrad congruence the tilt
    dispersion is ~0 and only the angular-spread detector fires.  Neither
    detector is sufficient; the gate fires on EITHER."""
    fan = _detector_field('fan_2x2')
    resid = _carrier_residual_rms(fan, None, _WL, _DDX)
    mv_rad = _tilt_dispersion(fan, _DDX, _DDX, _WL, 1.0)
    assert resid < 1e-6, ('the fail-before premise moved: the angular-spread '
                          f'detector now reads {resid} rad on the fan')
    assert resid < _NONCOLLIMATED_RESID_THRESH
    assert mv_rad > 1.4e-2, mv_rad

    tilt = _detector_field('tilted')
    assert _carrier_residual_rms(tilt, None, _WL, _DDX) > \
        _NONCOLLIMATED_RESID_THRESH
    assert _tilt_dispersion(tilt, _DDX, _DDX, _WL, 1.0) < 1e-5


def test_entry_stats_use_a_fixed_reference_na_not_the_prescription():
    """The chain-entry helper reports the tilt dispersion in fga's OWN
    currency, but against a CONSTANT reference NA -- and it is not given the
    prescription at all, so the first group cannot influence the verdict.

    The four returned numbers are pinned against their definitions: detector A
    verbatim, fga's raw ``na=1`` reading verbatim, the grid-canonical form
    ``raw * sqrt(lambda / dx)``, and the score ``canon /
    _MULTI_CONGRUENCE_NA_REF``."""
    import inspect
    params = list(inspect.signature(_chain_entry_congruence_stats)
                  .parameters)
    assert params == ['env', 'dx', 'wavelength'], (
        f'the entry-stats helper took {params}: if it is handed the '
        f'prescription again, the first lens can steer the verdict again')

    fan = _detector_field('fan_2x2')
    resid, mv, raw, canon = _chain_entry_congruence_stats(fan, _DDX, _WL)
    assert resid == pytest.approx(
        _carrier_residual_rms(fan, None, _WL, _DDX), rel=1e-12)
    assert raw == pytest.approx(
        _tilt_dispersion(fan, _DDX, _DDX, _WL, 1.0), rel=1e-12)
    assert canon == pytest.approx(raw * np.sqrt(_WL / _DDX), rel=1e-12)
    assert mv == pytest.approx(canon / _MULTI_CONGRUENCE_NA_REF, rel=1e-12)
    assert mv > _MULTI_CONGRUENCE_MV_THRESH, (mv, canon)


def _fast_relay_groups():
    """A first group fast enough to SATURATE ``fga._system_na`` at its 0.375
    cap (``_default_p_max`` = ``min(0.6, 1.6 * NA)``) -- the condition that
    used to divide a real fan's score BELOW the cutoff."""
    gA = _singlet(12e-3, -12e-3, 4.0e-3, 'N-BK7', 14e-3, 'fastA')
    gB = _singlet(12e-3, -12e-3, 4.0e-3, 'N-BK7', 14e-3, 'fastB')
    return [{'prescription': gA, 'gap_before': 0.0},
            {'prescription': gB, 'gap_before': _GAP}]


def _slow_relay_groups():
    """A first group slow enough to sit on ``_default_p_max``'s 0.05 FLOOR, so
    ``fga._system_na`` reports its minimum 0.03125 -- the condition that used
    to multiply an ordinary clipped beam's score ABOVE the cutoff.  Every
    relay slower than about f/10 lands here."""
    gA = _singlet(500e-3, -500e-3, 3.0e-3, 'N-BK7', 10e-3, 'slowA')
    gB = _singlet(500e-3, -500e-3, 3.0e-3, 'N-BK7', 10e-3, 'slowB')
    return [{'prescription': gA, 'gap_before': 0.0},
            {'prescription': gB, 'gap_before': _GAP}]


def _sym_fan_8x8(n=None, dx=None, w=None):
    """8x8 SYMMETRIC +-23 mrad fan -- mean tilt exactly 0, so detector A is
    blind by construction and only the multi-valuedness detector can fire.
    This one sits ON the detection floor; see
    ``test_the_documented_detection_floor_is_a_pinned_boundary``."""
    n = _CN if n is None else n
    dx = _CDX if dx is None else dx
    w = _CW if w is None else w
    X, Y = _grid(n, dx)
    G = _gauss(n, dx, w)
    ix = (np.arange(8) - 3.5) / 3.5
    return sum(G * np.exp(1j * _K0 * 0.023 * (a * X + b * Y))
               for a in ix for b in ix)


def _clipped_and_propagated(n=None, dx=None, w=None, clip=0.7, zfrac=0.02):
    """ONE congruence: a Gaussian hard-clipped by an aperture and
    ASM-propagated a short distance, so it carries real Fresnel edge ringing.
    A completely ordinary chain input that must stay silent."""
    n = _CN if n is None else n
    dx = _CDX if dx is None else dx
    w = _CW if w is None else w
    X, Y = _grid(n, dx)
    r = np.hypot(X, Y)
    clipped = (_gauss(n, dx, w) * (r < clip * w)).astype(np.complex128)
    return la.angular_spectrum_propagate(
        clipped, zfrac * np.pi * w ** 2 / _WL, _WL, dx)


def test_the_verdict_does_not_depend_on_the_first_groups_f_number():
    """TWO-SIDED FAIL-BEFORE (both D3 adversarial kills on the reference NA),
    fixed by making the reference a CONSTANT.

    ``fga._system_na`` is ``_default_p_max / 1.6`` with ``_default_p_max =
    min(0.6, max(0.05, 1.6 * NA))`` -- SATURATED at 0.375 above, FLOORED at
    0.03125 below, and a description of the first LENS rather than of the
    field handed in.  Dividing the field's own dispersion by it broke the gate
    in both directions on fields that never changed:

    * a genuine 8x8 +-23 mrad fan (56 % wrong by the linearity oracle) scored
      BELOW the cutoff behind a saturated first group -- a MISS;
    * an ordinary single beam, hard-clipped and propagated 0.02 z_R, scored
      ABOVE the cutoff behind a floored one -- a FALSE POSITIVE telling the
      caller to split one congruence into DOE orders.

    Both are pinned here as the pre-fix scores, so removing the constant
    reference fails this test."""
    slow, fast = _slow_relay_groups(), _fast_relay_groups()
    na_slow = _system_na(slow[0]['prescription'], _WL)
    na_fast = _system_na(fast[0]['prescription'], _WL)
    na_121 = _system_na(_relay_groups()[0]['prescription'], _WL)
    assert na_fast > 0.37, na_fast              # fga's saturation cap
    assert na_slow < 0.032, na_slow             # fga's floor
    assert 0.05 < na_121 < 0.10, na_121         # a 121-class relay group

    # -- MISS side: design 121's own 8x4 order fan.  Detector A is blind
    #    (symmetric orders, residual ~1e-17 rad), so only B can fire, and the
    #    pre-fix score behind the fast group was below cutoff.
    fan = _fan_121_8x4(_FANDX)
    resid, mv, raw, _canon = _chain_entry_congruence_stats(fan, _FANDX, _WL)
    assert resid < _NONCOLLIMATED_RESID_THRESH, resid
    assert mv > _MULTI_CONGRUENCE_MV_THRESH, (mv, raw)
    assert raw / na_fast < _MULTI_CONGRUENCE_MV_THRESH, (
        'the fail-before premise moved: a saturated system NA no longer '
        f'normalises this fan below the cutoff (raw {raw:.4e})')

    # -- FALSE-POSITIVE side: one clipped, propagated beam.  Detector A is
    #    silent on it too, so the pre-fix gate fired on B alone.
    one = _clipped_and_propagated()
    resid1, mv1, raw1, _c1 = _chain_entry_congruence_stats(one, _CDX, _WL)
    assert resid1 < _NONCOLLIMATED_RESID_THRESH, resid1
    assert raw1 > 1e-4, ('the fixture stopped exercising edge ringing at all; '
                         f'raw dispersion {raw1:.3e}')
    assert raw1 / na_slow > _MULTI_CONGRUENCE_MV_THRESH, (
        'the fail-before premise moved: a floored system NA no longer '
        f'normalises this clipped beam above the cutoff (raw {raw1:.4e})')
    assert mv1 < _MULTI_CONGRUENCE_MV_THRESH, (
        f'the gate warns on a clipped-and-propagated SINGLE beam '
        f'(score {mv1:.4f}) -- crying wolf')


def test_the_verdict_is_identical_through_a_slow_and_a_fast_chain():
    """The same two fields through the LIVE chain, at the two f-numbers that
    used to decide the verdict: the fan must warn behind BOTH, and the clipped
    single beam must stay silent behind BOTH."""
    cases = ((_fan_121_8x4(_FANDX), _FANDX, 1),
             (_clipped_and_propagated(), _CDX, 0))
    for groups in (_slow_relay_groups(), _fast_relay_groups()):
        for field, dx, want in cases:
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter('always')
                la.propagate_traced_carrier_chain(
                    np.ascontiguousarray(field), groups, _WL, dx,
                    r_in=float('inf'), ray_subsample=8,
                    final_distance=_CFD, traced_kwargs=dict(_TKW))
            msgs = _p3_msgs(rec)
            assert len(msgs) == want, (
                groups[0]['prescription']['name'], want, msgs)
            if want:
                assert 'propagate_traced_carrier_chain_multi' in msgs[0]
                # the message must expose the grid dependence it removed
                assert 'grid-canonical' in msgs[0]
                assert 'crossing half-angle' in msgs[0]


def test_the_documented_detection_floor_is_a_pinned_boundary():
    """The honest envelope, pinned rather than merely prosed.

    The absolute cutoff (0.06 x 0.15 = 9.0e-3 rad canonical) corresponds, via
    ``canon ~ 3.5 theta^1.5``, to a crossing half-angle of ~19 mrad.  An 8x8
    fan spanning +-23 mrad reads like a ~17-19 mrad PAIR and sits ON the
    boundary: it is NOT reliably caught, at any pitch.  That is a real gap and
    this pin is where it is recorded -- if a future cutoff change moves it,
    this test says so instead of the gap silently becoming a surprise.

    WHY it reads that way (corrected, niche C2 2026-07-30 -- an earlier
    revision of this docstring said a dense fan "is scored by its finest
    fringes, not by its total span", and that is measurably backwards).  The
    score tracks the TOTAL SPAN, derated ~20 %: this fan's nearest-neighbour
    spacing is 6.571 mrad, and a PAIR at that spacing reads only 3.2-3.5 mrad
    equivalent -- 5.3x below the fan -- while a pair at its +-23 mrad span
    reads 22.5-23.0.  Densifying at fixed span moves the score DOWN, not up
    (4 / 8 / 16 orders across +-23 read like 16.7 / 14.2 / 12.8 mrad).  The
    boundary asserted below is unchanged; only the reason for it is."""
    for dx in (4.0e-6, 2.0e-6):
        fan = _sym_fan_8x8(_CN if dx == _CDX else int(round(2.048e-3 / dx)),
                           dx, _CW)
        resid, mv, _raw, canon = _chain_entry_congruence_stats(fan, dx, _WL)
        assert resid < _NONCOLLIMATED_RESID_THRESH, resid    # A is blind
        equiv = (canon / 3.5) ** (2.0 / 3.0)
        assert 0.015 < equiv < 0.021, (dx, equiv)
        assert 0.5 < mv / _MULTI_CONGRUENCE_MV_THRESH < 2.0, (
            f'dx0={dx * 1e6:.1f} um: the +-23 mrad 8x8 fan now scores '
            f'{mv:.4f} against a {_MULTI_CONGRUENCE_MV_THRESH} cutoff, i.e. '
            f'it left the documented no-margin boundary -- the envelope note '
            f'in propagators/carrier.py needs re-measuring')


# dx0 = 0.25 um at a 0.512 mm window is N = 2048 (67 MB complex128); the
# detector allocates ~6 same-shape float64 temporaries on top.
_DXSWEEP_WINDOW, _DXSWEEP_W = 0.512e-3, 60e-6
_DXSWEEP = (4.0e-6, 2.0e-6, 1.0e-6, 0.5e-6, 0.25e-6)
_FANDX = 1.0e-6                      # the N=512 row of that sweep


def _fan_121_8x4(dx):
    """The design-121 order fan -- 8x4 at +-46 / +-23 mrad with EQUAL order
    phases -- on a FIXED 0.512 mm physical window, so only the grid pitch
    changes between calls."""
    n = int(round(_DXSWEEP_WINDOW / dx))
    X, Y = _grid(n, dx)
    G = _gauss(n, dx, _DXSWEEP_W)
    out = np.zeros_like(G)
    for a in (np.arange(8) - 3.5) / 3.5 * 0.046:
        for b in (np.arange(4) - 1.5) / 1.5 * 0.023:
            out = out + G * np.exp(1j * _K0 * (a * X + b * Y))
    return out


def test_the_detector_does_not_go_blind_as_the_grid_gets_finer():
    """FAIL-BEFORE (the D3 adversarial kill on grid dependence), fixed by the
    ``sqrt(lambda / dx)`` canonicalization.

    ``_tilt_dispersion``'s raw reading is not a property of the field: two
    equal beams crossing at +-theta superpose to a REAL cosine times ONE mean
    carrier, so the wrapped nearest-neighbour increment is zero inside every
    lobe and +-pi only across the amplitude NULLS -- where the amplitude weight
    that multiplies it falls as (dx / fringe)^2.  The raw rms therefore scales
    as sqrt(dx) and the detector goes BLINDER as the grid gets more accurate.

    On ONE unchanged physical field -- design 121's own 8x4 order fan on a
    fixed 0.512 mm window -- the pre-canonical score falls monotonically with
    the pitch and drops below the cutoff, while detector A is blind by
    symmetry.  The canonical score is flat and fires at every pitch."""
    import psutil
    if psutil.virtual_memory().available < 3 * 1024 ** 3:
        pytest.skip('needs ~3 GB free for the N=2048 pitch')

    raws, canons, silent_pre = [], [], []
    for dx in _DXSWEEP:
        fan = _fan_121_8x4(dx)
        resid, mv, raw, canon = _chain_entry_congruence_stats(fan, dx, _WL)
        raws.append(raw)
        canons.append(canon)
        pre = raw / _MULTI_CONGRUENCE_NA_REF        # the pre-fix score
        if pre <= _MULTI_CONGRUENCE_MV_THRESH:
            silent_pre.append(dx)
        # PASS-AFTER: fires at EVERY pitch
        assert mv > _MULTI_CONGRUENCE_MV_THRESH, (
            f'dx0={dx * 1e6:.3f} um: the 121 order fan scored {mv:.4f}, at or '
            f'below the {_MULTI_CONGRUENCE_MV_THRESH} cutoff')
        if dx <= 2e-6:
            # ...and it is not detector A doing the work at the fine pitches
            assert resid < _NONCOLLIMATED_RESID_THRESH, (dx, resid)

    # FAIL-BEFORE: the raw reading really does decay with the pitch, and the
    # pre-fix gate really did go silent at the fine end.
    assert raws[-1] < 0.5 * raws[0], raws
    for a, b in zip(raws, raws[1:]):
        assert b < a, ('the fail-before premise moved: the raw dispersion no '
                       f'longer decays with the grid pitch -- {raws}')
    assert silent_pre, (
        'the fail-before premise moved: the pre-canonical score no longer '
        f'goes silent on the 121 fan at any pitch down to '
        f'{_DXSWEEP[-1] * 1e6:.3f} um -- {raws}')
    # PASS-AFTER, quantitatively: the canonical score is grid-STABLE.
    assert max(canons) < 2.0 * min(canons), canons


def test_canonical_score_tracks_the_closed_form_crossing_angle():
    """The canonical dispersion has an ANALYTIC oracle, which is what makes the
    absolute cutoff meaningful rather than fitted.

    For two comparable beams at +-theta the null-pixel argument above gives
    ``raw ~ 2 pi theta sqrt(theta dx / lambda)``, so the canonical form
    ``raw * sqrt(lambda / dx)`` is a function of the CROSSING ANGLE alone.
    Measured prefactor ~0.55 of ``2 pi``, i.e. ``canon ~ 3.5 theta^1.5`` -- and
    that is what turns the 9.0e-3 rad cutoff into a stated ~19 mrad half-angle
    detection floor."""
    n, dx, w = 512, 1.0e-6, 60e-6
    X, _Y = _grid(n, dx)
    G = _gauss(n, dx, w)
    for half in (0.010, 0.023, 0.046):
        E = (G * np.exp(1j * _K0 * half * X)
             + G * np.exp(-1j * _K0 * half * X))
        _resid, _mv, _raw, canon = _chain_entry_congruence_stats(E, dx, _WL)
        assert canon == pytest.approx(3.5 * half ** 1.5, rel=0.10), (
            half, canon, 3.5 * half ** 1.5)
    # the stated floor: the cutoff really does land near 19 mrad half-angle
    floor = (_MULTI_CONGRUENCE_MV_THRESH * _MULTI_CONGRUENCE_NA_REF
             / 3.5) ** (2.0 / 3.0)
    assert 0.015 < floor < 0.025, floor


def test_entry_stats_survive_a_degenerate_field():
    """A diagnostic must never be the thing that kills a propagation: an
    all-zero field yields zeros, not a divide-by-zero or a raise."""
    zeros = np.zeros((32, 32), dtype=np.complex128)
    resid, mv, raw, canon = _chain_entry_congruence_stats(zeros, _DDX, _WL)
    assert resid == 0.0 and mv == 0.0 and raw == 0.0 and canon == 0.0


# ===========================================================================
# P3 -- the GATE, end to end through the chain.
# ===========================================================================
_CN, _CDX, _CW = 512, 4.0e-6, 300e-6     # 2.048 mm window, fringes ~7 px
_CFD = 0.05
_TKW = dict(on_undersample='silent', on_noncollimated='silent')
# D3 (2026-08-06): ``on_replica='ignore'`` -- this file's fixtures read a
# 358 um window against a ~77 um Bluestein period, which they always did.
# Nothing here reads the outer window: every assertion in this file is about
# a GUARD firing or not firing, not about field values.  Waived at the fixture
# so the guards under test are the ones this file is about.
_RO = dict(dx_out=1.4e-6, N_out=256, on_replica='ignore')


def _chain(field, *, groups=None, quiet=False, **kw):
    kw.setdefault('ray_subsample', 8)
    kw.setdefault('n_workers', 4)
    kw.setdefault('final_distance', _CFD)
    kw.setdefault('traced_kwargs', _TKW)
    kw.setdefault('final_leg', 'paraxial')
    kw.setdefault('focus_readout', dict(_RO))

    def _go():
        return la.propagate_traced_carrier_chain(
            field, groups if groups is not None else _relay_groups(),
            _WL, _CDX, **kw)
    if quiet:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return _go()
    return _go()


@pytest.fixture(scope='module')
def _clean_input():
    return _gauss(_CN, _CDX, _CW)


@pytest.fixture(scope='module')
def _fan_input():
    """A 2x2 order fan at +-23 mrad -- four comparable-power congruences at
    well-separated angles, i.e. the excluded case, on a grid that RESOLVES
    the interference (fringe pitch lambda/0.046 = 28.5 um = 7.1 px)."""
    X, Y = _grid(_CN, _CDX)
    G = _gauss(_CN, _CDX, _CW)
    out = np.zeros_like(G)
    for sx in (-1, 1):
        for sy in (-1, 1):
            out = out + G * np.exp(1j * _K0 * 0.023 * (sx * X + sy * Y))
    return out


def test_clean_single_congruence_stays_silent(_clean_input):
    """The default must NOT break a currently-passing single-congruence run:
    no P3 warning at all."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _chain(_clean_input)
    assert _p3_msgs(rec) == []


def test_clean_single_congruence_is_byte_identical_under_the_guard(
        _clean_input):
    """The gate is READ-ONLY: the guarded default and the pre-D3 behaviour
    (``on_multi_congruence='ignore'``, which skips the measurement entirely)
    must return the same field to within 1e-10 of scale.  Compared with a
    TOLERANCE, not ``array_equal``: both sides are live FFT/cache calls."""
    a = _chain(_clean_input, quiet=True)
    b = _chain(_clean_input, quiet=True, on_multi_congruence='ignore')
    scale = float(np.max(np.abs(a.field)))
    assert scale > 0.0
    assert float(np.max(np.abs(a.field - b.field))) <= 1e-10 * scale


def test_multi_congruence_input_warns_and_names_the_route(_fan_input):
    """PASS-AFTER: the fan trips the gate, and the message NAMES the
    multi-congruence route (the D2 orchestrator) rather than merely
    complaining."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _chain(_fan_input)
    msgs = _p3_msgs(rec)
    assert len(msgs) == 1, _runtime_msgs(rec)
    assert 'propagate_traced_carrier_chain_multi' in msgs[0]
    assert 'TiltedCarrier' in msgs[0]


def test_multi_congruence_error_policy_raises(_fan_input):
    """Batch production can make it fatal."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(RuntimeError,
                           match='single-congruence envelope'):
            _chain(_fan_input, on_multi_congruence='error')


def test_multi_congruence_ignore_reproduces_the_pre_d3_silence(_fan_input):
    """FAIL-BEFORE, structurally: ``'ignore'`` is exactly the v5.31 behaviour
    -- the fan propagates to completion, returns a populated field, and NO
    warning names the multi-congruence route.  That is the silent wrong
    answer the default now catches."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        res = _chain(_fan_input, on_multi_congruence='ignore')
    assert _p3_msgs(rec) == []
    assert np.all(np.isfinite(res.field))
    assert float(np.max(np.abs(res.field))) > 0.0


def _linearity_error(tilt):
    """Relative L2 LINEARITY VIOLATION of the chain at a 2x2 fan of
    half-angle ``tilt``.

    Physical propagation through a passive optic is a LINEAR operator on the
    field, so the exact answer for a superposition is the superposition of the
    answers: ``chain(sum_k E_k) == sum_k chain(E_k)``.  The traced
    entrance->exit map is NOT linear on a multi-valued input -- it launches one
    ray per pixel along the local phase gradient, i.e. along the
    amplitude-weighted MEAN of the crossing directions, and applies that one
    angle's OPD to all of them.  Any gross departure here is therefore direct,
    oracle-free proof that the multiplexed route returned something that is not
    the physical field.

    Both sides run with ``on_multi_congruence='ignore'`` -- the comparison is
    the point, not the warning.  Read out on the CO-MOVING grid (no focus
    readout) so nothing is window-clipped and all five runs land on the same
    lattice."""
    X, Y = _grid(_CN, _CDX)
    G = _gauss(_CN, _CDX, _CW)
    parts = [G * np.exp(1j * _K0 * tilt * (sx * X + sy * Y))
             for sx in (-1, 1) for sy in (-1, 1)]
    kw = dict(focus_readout=None, on_multi_congruence='ignore')
    ref = None
    for p in parts:
        f = _chain(p, quiet=True, **kw).field
        ref = f if ref is None else ref + f
    mux = _chain(sum(parts), quiet=True, **kw).field
    return float(np.linalg.norm(mux - ref) / np.linalg.norm(ref))


def test_the_guarded_input_really_is_the_wrong_answer():
    """The gate must fire where the answer is actually WRONG, and stay quiet
    where it is right -- otherwise it is crying wolf.

    Measured on this synthetic relay: the +-23 mrad fan the gate refuses
    violates linearity by 62 % in relative L2, while a +-0.5 mrad pair the
    gate passes violates it by 0.2 % -- a ~280x separation that lands on the
    same side of the cutoff as the detector does.

    ERA-PINNED at ``_REMAP_RESID_EIKONAL_DEGREE = 4`` (niche C11 CI
    reconciliation, 2026-08-03).  The assertions are the originals, word for
    word, and nothing here is relaxed -- what moved is the library under them,
    and it moved for a reason this file cannot argue away.  Measured on
    Linux/OpenBLAS py3.12 (the CI build; Windows/MKL passes at the shipped
    default, which is why this only ever went red on CI):

        state                          bad (23 mrad)   good (0.5 mrad)  ratio
        C6 launch OFF (pre-niche-C6)        0.6236           0.003766   165.6
        C6 on, resid degree 4               14.2337          0.032148   442.8
        C6 on, resid degree 6 (SHIPPED)      0.9863          0.057458    17.2

    The docstring's own "62 %" is the C6-OFF row exactly, so this case was
    calibrated before the niche-C6 stationary-phase launch existed.  That
    launch makes ``apply_real_lens_traced`` deliberately NONLINEAR in its input
    (it fits the input's residual eikonal and launches along
    ``grad(W + a_fit)``), so a MULTIPLEXED input -- which carries beat fringes
    no single-valued residual model represents -- diverges further from the
    linear superposition the better that model gets.  Niche C10's degree raise
    is a better model, and it moves ``good`` 0.032 -> 0.057, across the 0.05
    bar.  ``SPHERE_PARAB_CONVERSION_EXACT`` (niche C9) is NOT involved: its two
    rows are identical to every printed digit.

    This is therefore a real, attributed increase in the error of the
    MULTIPLEXED route -- the route this very guard exists to refuse -- and not
    a regression on any supported single-congruence input, where C10 improved
    every measured number.  The shipped-era statement is kept live and
    comparative in the sibling below.
    """
    _deg = _lens_traced._REMAP_RESID_EIKONAL_DEGREE
    _lens_traced._REMAP_RESID_EIKONAL_DEGREE = 4
    try:
        bad = _linearity_error(0.023)
        good = _linearity_error(0.0005)
        assert bad > 0.30, (
            f'the multiplexed fan reproduced the linear superposition to '
            f'{bad:.4f}; the P3 gate would then be crying wolf')
        assert good < 0.05, (
            f'a near-collinear pair the gate PASSES violated linearity by '
            f'{good:.4f}; the gate would then be missing a real failure')
        assert bad > 20.0 * good, (bad, good)
    finally:
        _lens_traced._REMAP_RESID_EIKONAL_DEGREE = _deg
    # ...and the detector agrees with the oracle on which is which.  This half
    # reads only the congruence statistics of the INPUT, so it is independent
    # of the residual model and stays on the shipped default.
    X, Y = _grid(_CN, _CDX)
    G = _gauss(_CN, _CDX, _CW)
    for tilt, fires in ((0.023, True), (0.0005, False)):
        fld = sum(G * np.exp(1j * _K0 * tilt * (sx * X + sy * Y))
                  for sx in (-1, 1) for sy in (-1, 1))
        resid, mv, _raw, _canon = _chain_entry_congruence_stats(
            fld, _CDX, _WL)
        hit = (resid > _NONCOLLIMATED_RESID_THRESH
               or mv > _MULTI_CONGRUENCE_MV_THRESH)
        assert hit is fires, (tilt, resid, mv)


def test_the_separation_survives_the_c10_residual_degree_and_is_caused_by_it():
    """The SHIPPED-era half of the case above, stated comparatively so no
    absolute bar on a BLAS-dependent magnitude can rot again.

    Two claims, both ratios between arms measured in this process:

    1. the gate still SEPARATES -- the fan it refuses violates linearity by an
       order of magnitude more than the pair it passes;
    2. the thing that moves the multiplexed route is
       ``_REMAP_RESID_EIKONAL_DEGREE``, which is the reason the sibling above
       is era-pinned rather than relaxed.

    **Claim 2 is measured on the REFUSED fan, not the passed pair, and that is
    the whole point of this revision.**  Both quantities carry the same
    mechanism, but not with the same signal-to-noise across BLAS builds:

        deg 4 -> deg 6         Windows/MKL   WSL/OpenBLAS   CI/OpenBLAS
        good (0.5 mrad)          1.19x          1.79x          1.04x
        bad  (23 mrad)          19.2x          14.4x            --

    A bar on ``good`` has to live inside a 1.04-1.79x spread and there is no
    value that is both meaningful and safe -- a 1.10x bar passed two builds and
    failed the third by 6 %.  The SAME mechanism read on ``bad`` is 14-19x on
    every build measured, so the bar sits at 5x with 3x of headroom on the
    weakest.  Nothing was weakened: the claim moved to where it is large.

    ERA-PINNED to ``LSTSQ_CONDITIONING_STEPDOWN = False`` (niche C13,
    2026-08-03), and the 3x of headroom above is exactly what C13 consumed.
    ``bad6`` is a degree-6 linearity error computed through the traced fits,
    and before C13 those fits were solved through a numerically singular Gram:
    the degree-6 arm's error was being SUPPRESSED by a lucky null-space draw
    that differed per build.  Measured on this fixture:

        bad6            step-down OFF      step-down ON
        Windows            1.2135             3.3113
        Linux              0.9863             3.3246
        build spread        23 %               0.4 %

    ``bad4`` barely moves (23.24 -> 22.46, 14.23 -> 10.64), so the ratio this
    test bars falls to 6.78x / 3.20x purely because the DENOMINATOR became
    honest.  The mechanism claim is intact and still large; the 5x bar was a
    pre-C13 calibration.  The shipped era is asserted -- more strongly than
    this test ever did -- in the sibling below.
    """
    _step = _lens_traced.LSTSQ_CONDITIONING_STEPDOWN
    _lens_traced.LSTSQ_CONDITIONING_STEPDOWN = False
    try:
        _run_the_era_pinned_body()
    finally:
        _lens_traced.LSTSQ_CONDITIONING_STEPDOWN = _step


def _run_the_era_pinned_body():
    bad6 = _linearity_error(0.023)
    good6 = _linearity_error(0.0005)
    _deg = _lens_traced._REMAP_RESID_EIKONAL_DEGREE
    _lens_traced._REMAP_RESID_EIKONAL_DEGREE = 4
    try:
        bad4 = _linearity_error(0.023)
    finally:
        _lens_traced._REMAP_RESID_EIKONAL_DEGREE = _deg
    # 1. the gate still separates (measured 17x-92x across builds)
    assert bad6 > 5.0 * good6, (bad6, good6)
    # 2. the residual degree is what moves the multiplexed route (14-19x)
    assert bad4 > 5.0 * bad6, (bad4, bad6)


def test_c13_makes_the_d3_separation_build_independent():
    """The SHIPPED-era statement, and it is strictly stronger than the
    era-pinned sibling's claim 1.

    Niche C13 gave the traced fits a backward-stable solve, and the degree-6
    linearity error stopped being a per-build lottery: ``bad6`` reads 3.3113
    (Windows) against 3.3246 (Linux), a **0.4 %** spread where the pre-C13
    solver gave 1.2135 against 0.9863 (**23 %**).

    The gate's own separation ``bad6 / good6`` follows it: **80.5x / 17.2x**
    before, **258.9x / 259.9x** after -- one number on both builds instead of
    a 4.7x disagreement about it.  That is what a bar can finally be placed on,
    so this test places one where the sibling could not.
    """
    bad6 = _linearity_error(0.023)
    good6 = _linearity_error(0.0005)
    # 100x, against 258.9 / 259.9 measured -- and against 17.2 pre-C13 on the
    # weaker build, so this bar is one C13 alone can carry.
    assert bad6 > 100.0 * good6, (bad6, good6)


def test_multi_congruence_policy_is_validated():
    with pytest.raises(ValueError, match='on_multi_congruence'):
        _chain(_gauss(64, _CDX, _CW), on_multi_congruence='shout')
    with pytest.raises(ValueError, match='multi_congruence_threshold'):
        _chain(_gauss(64, _CDX, _CW), multi_congruence_threshold=0.0)


def test_a_correctly_decomposed_congruence_of_the_fan_stays_silent():
    """The route the gate POINTS AT must not itself trip the gate: an order
    carried as its own ``TiltedCarrier`` congruence is a clean single
    congruence and stays silent -- the tilt is divided out of the envelope
    the gate measures.

    (The tilt here is 4 mrad rather than the fan's 23: at 23 mrad this
    synthetic relay's chief ray walks off its own co-moving grid, which is a
    separate, already-guarded D1 condition and not what this pin is about.)"""
    G = _gauss(_CN, _CDX, _CW)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _chain(G, r_in=la.TiltedCarrier(np.inf, 0.004, 0.004))
    assert _p3_msgs(rec) == []


# ===========================================================================
# P5a -- exit-NA proximity to the 'auto' route flip.
# ===========================================================================
@pytest.fixture(scope='module')
def _measured_na_exit(_clean_input):
    """The exit NA this synthetic relay actually presents, read from the
    chain's own ``stages`` (D3 also reports it there).  Everything below is
    expressed RELATIVE to it, so the pins do not encode a magic number."""
    res = _chain(_clean_input, quiet=True, final_leg='auto',
                 na_exact_threshold=10.0, on_na_proximity='ignore')
    na = [s.get('na_exit') for s in res.stages if s.get('na_exit') is not None]
    assert len(na) == 1, res.stages
    assert 0.0 < na[0] < 1.0
    return float(na[0])


def _na_msgs(rec):
    return [m for m in _runtime_msgs(rec) if 'na_exact_threshold' in m]


def test_exit_na_just_under_the_threshold_warns(_clean_input,
                                                _measured_na_exit):
    """na_exit 10 % BELOW the threshold -> inside the default 20 % band ->
    the near miss is announced, and the message says which side it fell on.
    (The band is placed relative to this relay's OWN measured na_exit, so the
    pin carries no design-121 margin -- design 121 sits at na_exit 0.405,
    nowhere near the 0.15 default.)"""
    thr = _measured_na_exit / 0.90
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _chain(_clean_input, final_leg='auto', na_exact_threshold=thr)
    msgs = _na_msgs(rec)
    assert len(msgs) == 1, _runtime_msgs(rec)
    assert 'BELOW (routing PARAXIAL)' in msgs[0]
    assert "final_leg='exact'" in msgs[0]


def test_exit_na_far_below_the_threshold_is_silent(_clean_input,
                                                   _measured_na_exit):
    """na_exit 50 % below the threshold is not a near miss -- no warning."""
    thr = _measured_na_exit / 0.50
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _chain(_clean_input, final_leg='auto', na_exact_threshold=thr)
    assert _na_msgs(rec) == []


def test_exit_na_proximity_error_policy_raises(_clean_input,
                                               _measured_na_exit):
    thr = _measured_na_exit / 0.90
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(RuntimeError, match='na_exact_threshold'):
            _chain(_clean_input, final_leg='auto', na_exact_threshold=thr,
                   on_na_proximity='error')


def test_exit_na_proximity_is_not_checked_for_an_explicit_final_leg(
        _clean_input, _measured_na_exit):
    """An explicit ``final_leg`` is the caller's decision, not a silent
    route: the proximity guard is an 'auto'-only diagnostic."""
    thr = _measured_na_exit / 0.90
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _chain(_clean_input, final_leg='paraxial', na_exact_threshold=thr)
    assert _na_msgs(rec) == []


def test_na_proximity_policy_is_validated():
    with pytest.raises(ValueError, match='on_na_proximity'):
        _chain(_gauss(64, _CDX, _CW), on_na_proximity='shout')
    with pytest.raises(ValueError, match='na_proximity_frac'):
        _chain(_gauss(64, _CDX, _CW), na_proximity_frac=-1.0)


# ===========================================================================
# P5b -- RAM-capped readout: 'warn' degrades, 'error' refuses.
# ===========================================================================
_TIGHT_RAM = 64 * 1024 ** 2      # 64 MiB budget: caps a 32768^2 request hard
_TINY_RAM = 4 * 1024 ** 2        # 4 MiB: binds even on this small synthetic leg


def test_ram_cap_warn_degrades_and_announces():
    """Historical behaviour, unchanged by D3: the grid degrades, the warning
    names the un-degraded requirement and labels the result non-converged."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        n = _memory_bounded_n_fine(1 << 15, 'probe', ram_budget=_TIGHT_RAM)
    assert n < (1 << 15)
    msgs = _runtime_msgs(rec)
    assert len(msgs) == 1
    assert 'RESOLUTION-LIMITED (non-converged)' in msgs[0]
    assert 'on_ram_cap' in msgs[0]


def test_ram_cap_error_policy_refuses():
    """PASS-AFTER: an unattended production run can fail loudly instead of
    reporting a metric computed on a degraded grid."""
    with pytest.raises(MemoryError, match='RESOLUTION-LIMITED'):
        _memory_bounded_n_fine(1 << 15, 'probe', ram_budget=_TIGHT_RAM,
                               on_ram_cap='error')


def test_ram_cap_ignore_is_silent():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        n = _memory_bounded_n_fine(1 << 15, 'probe', ram_budget=_TIGHT_RAM,
                                   on_ram_cap='ignore')
    assert n < (1 << 15)
    assert _runtime_msgs(rec) == []


def test_ram_cap_that_does_not_bind_never_fires():
    """The policy must be inert when the cap does not bind -- otherwise
    'error' would be unusable in production."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        n = _memory_bounded_n_fine(256, 'probe', ram_budget=float('inf'),
                                   on_ram_cap='error')
    assert n == 256
    assert _runtime_msgs(rec) == []


def test_ram_cap_policy_reaches_the_exact_readout(_clean_input):
    """End-to-end: the chain forwards ``on_ram_cap`` into the exact final
    leg, so a production run really can make the degradation fatal."""
    fr = dict(_RO)
    fr['ram_budget'] = _TINY_RAM
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(MemoryError, match='RESOLUTION-LIMITED'):
            _chain(_clean_input, final_leg='exact', focus_readout=fr,
                   on_ram_cap='error')
    # ...and the SAME call under the shipped default completes, degraded.
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        res = _chain(_clean_input, final_leg='exact', focus_readout=fr)
    assert np.all(np.isfinite(res.field))
    assert any('RESOLUTION-LIMITED' in m for m in _runtime_msgs(rec))


def test_ram_cap_policy_is_validated():
    with pytest.raises(ValueError, match='on_ram_cap'):
        _chain(_gauss(64, _CDX, _CW), on_ram_cap='shout')
    with pytest.raises(ValueError, match='on_ram_cap'):
        la.carrier_referenced_exact_focus_readout(
            _gauss(64, _CDX, _CW), 0.1, 0.01, _WL, _CDX,
            dx_out=1e-6, N_out=32, on_ram_cap='shout')


# ===========================================================================
# P5c -- the rs_fine clamp degenerate corner: warn-only today, opt-in STRICT.
# ===========================================================================
@pytest.fixture(scope='module')
def _clamp_case():
    """A setup that forces the corner: ``n_fine_cap`` small enough that
    ``dx_fine`` lands COARSER than the chain's physical ray pitch
    ``ray_subsample * cur_dx``, so the pitch-preserving rescale rounds below
    1 and is clamped to 1 (the N=28672 / n_fine_cap=16384 design-121
    condition, reproduced small)."""
    presc = _singlet(30e-3, -30e-3, 2.0e-3, 'N-BK7', 2.0e-3, 'strong')
    R_out = _paraxial_group_r_out(presc, np.inf, _WL)
    N, cur_dx, w = 256, 10e-6, 0.6e-3
    env = _gauss(N, cur_dx, w)
    na_exit = w / abs(R_out)
    call_kw = dict(parallel_amp=False, on_undersample='silent',
                   on_noncollimated='silent')
    # the clamp binds when dx_fine = win / n_fine_cap > ray_subsample * cur_dx
    return presc, R_out, env, cur_dx, na_exit, call_kw


def _clamp_msgs(rec):
    return [m for m in _runtime_msgs(rec) if 'CANNOT be' in m]


def test_rs_fine_clamp_warns_by_default(_clamp_case):
    """Today's behaviour: the corner is announced (naming BOTH pitches) and
    the retrace continues on the coarser ray lattice."""
    presc, R_out, env, cur_dx, na_exit, call_kw = _clamp_case
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _fine_trace_group_exit(
            env, np.inf, cur_dx, presc, _WL, 4, 1, call_kw, R_out, na_exit,
            window_factor=7.0, n_fine_cap=32)
    msgs = _clamp_msgs(rec)
    assert len(msgs) == 1, _runtime_msgs(rec)
    assert 'ray pitch' in msgs[0] and 'on_rs_fine_clamp' in msgs[0]


def test_rs_fine_clamp_strict_mode_raises(_clamp_case):
    """PASS-AFTER: the opt-in strict mode refuses the corner, so a run that
    needs the F-C pitch-preservation contract to hold cannot silently lose
    it."""
    presc, R_out, env, cur_dx, na_exit, call_kw = _clamp_case
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(RuntimeError, match='physical ray pitch'):
            _fine_trace_group_exit(
                env, np.inf, cur_dx, presc, _WL, 4, 1, call_kw, R_out,
                na_exit, window_factor=7.0, n_fine_cap=32,
                on_rs_fine_clamp='error')


def test_rs_fine_strict_mode_is_inert_when_the_contract_holds(_clamp_case):
    """The strict mode must not fire on the normal condition (dx_fine finer
    than the chain's ray pitch), or nothing could enable it."""
    presc, R_out, env, cur_dx, na_exit, call_kw = _clamp_case
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _fine_trace_group_exit(
            env, np.inf, cur_dx, presc, _WL, 4, 1, call_kw, R_out, na_exit,
            window_factor=7.0, n_fine_cap=16384, on_rs_fine_clamp='error')
    assert _clamp_msgs(rec) == []


def test_rs_fine_clamp_policy_is_validated():
    with pytest.raises(ValueError, match='on_rs_fine_clamp'):
        _chain(_gauss(64, _CDX, _CW), on_rs_fine_clamp='shout')


# ===========================================================================
# The D2 orchestrator forwards every D3 policy (a batch fan run is exactly
# what needs them fatal).
# ===========================================================================
def test_multi_forwards_the_guard_policies():
    import inspect
    sig = inspect.signature(la.propagate_traced_carrier_chain_multi)
    for k in ('on_multi_congruence', 'multi_congruence_threshold',
              'on_na_proximity', 'na_proximity_frac', 'on_ram_cap',
              'on_rs_fine_clamp'):
        assert k in sig.parameters, k
    G = _gauss(64, _CDX, _CW)
    with pytest.raises(ValueError, match='on_multi_congruence'):
        la.propagate_traced_carrier_chain_multi(
            [G], _relay_groups(), _WL, _CDX,
            output_grid=dict(_RO), on_multi_congruence='shout')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
