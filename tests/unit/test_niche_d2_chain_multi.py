"""Per-congruence chain orchestrator -- niche D2 (roadmap
ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27 P1b).

Why it exists.  The shipping design-121 device is a Dammann DOE fan (8x4
orders, 480 um frame pitch, +-46 mrad) from an emitter array -- K comparable-
power beams at well-separated angles, which is precisely the case
``apply_real_lens_traced``'s entrance->exit map excludes.  Pushed through the
chain MULTIPLEXED at v5.28 that fan produced a populated, credible-looking
frame lattice whose per-frame power was scrambled, with nothing raised.  The
two existing routes cannot fix it: ``apply_real_lens_traced_multi`` fixes
``preserve_input_phase=True`` and rejects ``amplitude_model='ray_density'`` on
its default ``reuse_prepared=True`` (so it cannot express the v5.29 validated
configuration at all), and ``apply_real_lens_traced_segmented``'s
``max_segments=32`` is saturated exactly by an 8x4 fan.

``propagate_traced_carrier_chain_multi`` runs each congruence through the
SHIPPED-DEFAULT chain in its own chief-ray-tracking frame (niche D1's tilted
carrier, applied internally -- the roadmap's "decentre to on-axis, re-offset at
the image plane", done for the consumer) and recombines complex amplitudes on
one common image grid.

Everything here is SELF-CONTAINED: synthetic N-BK7 singlets built inline, no
prescription asset, every physical claim checked against an inline oracle that
does not share code with the orchestrator (the paraxial ABCD of the groups, an
exact meridional ray trace, and the analytic two-beam fringe period).

Pins, in order:

* K=1 reduces to ``propagate_traced_carrier_chain`` to <= 1e-10 * scale (the
  shipped design-121 acceptance cannot move through this entry point);
* K=2 / K=4 well-separated congruences equal the hand-summed INDEPENDENT
  single-congruence runs -- both on the full common grid and through the
  lattice-snapped readout tiles (the orchestrator is only doing that
  bookkeeping);
* the chief-ray placement matches an EXACT meridional ray trace, and the
  chain's own tracked chief ray;
* BOTH faces are served by the one implementation: a per-ORDER fan (tilts)
  and a per-EMITTER array (pure decentres) both image onto their exact ray
  traces;
* ``readout_tile`` is an ACCURACY lever, not just a memory one -- the naive
  full-grid readout of an off-axis congruence exceeds one spatial period of
  the near-focus Bluestein reconstruction and fills the lattice with periodic
  REPLICAS of its own spot.  That regime is REFUSED by default now
  (``on_replica='error'``) and sized away by the default
  ``readout_tile='auto'``; an adversarial pass found the first cut of this
  entry point reproducing the very v5.28 scramble it exists to fix, at its
  own default setting.  A SECOND adversarial pass then found two faults in
  that fix, both pinned here:

  - ``'auto'`` sized the shared window from congruence 0 alone, so a later
    congruence with a shorter period raised -- i.e. whether the DEFAULT ran
    depended on list ORDER, and it could not run the roadmap's own 32-order
    acceptance (measured per-order period spread 1.8 %).  It now sizes from
    ``min(period)`` over ALL K, measured in a cheap 16-px probe pass
    (``test_auto_window_is_independent_of_congruence_order``);
  - the guard fired at ``K == 1`` too, where there is no neighbouring frame
    to contaminate, silently shrinking a clean single-congruence call's
    requested field of view and breaking the "K=1 reduces to the chain"
    contract (``test_k1_keeps_the_requested_field_of_view``);
* per-congruence power is conserved and reported ACROSS THREE PLANES --
  ``throughput`` is measured at the chain exit and is window-independent,
  while ``power_out`` / ``capture`` are window-dependent.  The same
  adversarial pass showed a ~1.5% per-order "vignetting" spread to be the
  readout tile clipping a field-angle-dependent halo, so the two are now
  reported separately and the clipping is warned about
  (``on_readout_clip``);
* INTERFERENCE is real: two congruences that overlap at an image conjugate
  produce the analytically-correct fringe period, and ``recombine='incoherent'``
  produces none;
* K is NOT capped at 32 (the ``max_segments`` cap is not inherited);
* the guards fire before any propagation work.
"""
from __future__ import annotations

import dataclasses
import os
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.propagators.carrier import (
    _chain_chief_ray_at_target,
    _group_abcd,
)
from lumenairy.raytrace import Surface, make_ray, trace
from lumenairy.raytrace.trace import surfaces_from_prescription

_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL

# --- synthetic 2-group relay (shared with the D1 pins) ----------------------
_N, _DX = 1024, 12e-6          # 12.288 mm window
_W_IN = 1.0e-3                 # 1/e amplitude radius
_GAP = 25e-3
_TILT = 0.030                  # 30 mrad: OUTSIDE the 0.02 rad residual
_TKW = dict(on_undersample='silent', on_noncollimated='silent')
# _TILE must fit inside ONE Bluestein period of each congruence's readout.
#
# RESTORED 256 (fix D3, 2026-08-06).  It had been cut to 120 because the tile
# was left to track whatever the shipped standoff happened to give: at 6.0 zR
# a 256 tile fitted, at 0.8 zR the period was 0.1739 mm and only <= 124 did.
# That is a 2.1x reduction in the area over which "tiled equals hand-placed"
# is checked, bought by a default flip -- so it is reversed here, and the tile
# is PAID FOR instead, with the same physically-justified remedy this file
# already applies three times below: a wide readout window REQUIRES a long
# Bluestein leg, so ask for the leg that holds the window rather than shrink
# the window to whatever leg the accuracy resolver picked.  The tile size is
# now independent of the shipped default, which is the point.
_DXO, _NOUT, _TILE = 1.4e-6, 2048, 256
_RS, _NW = 8, 4


_LEG_CAL = {}


def _leg_for_window(groups, field, carrier, fd, window, margin=1.3):
    """Shortest fine-zoom leg whose ONE Bluestein period holds ``window``.

    The reconstruction's period is ``N * d`` of the CO-MOVING grid at the
    hand-off plane, whose pitch is proportional to the standoff -- so the
    period is LINEAR in the leg and one calibration point fixes the whole
    relation.  The constant of proportionality is the chain's grid scaling AT
    THE FINAL LEG, which is NOT the input grid's: the closed form
    ``period = _N * _DX * standoff / fd`` (used by the three inline fixtures
    below) reads 1.73x high on this relay, because the groups rescale the
    co-moving pitch on the way.  So it is MEASURED once per ``fd`` from the
    chain's own ``readout_period`` stage report and cached, rather than
    assumed.

    Note this deliberately couples the LEG to the requested WINDOW, which the
    library's own resolver refuses to do by default (it would make the
    propagated field depend on how wide a window it is viewed through,
    breaking the K=1 contract).  A CALLER that genuinely needs a wide window
    is exactly who should make that trade, knowingly -- which is what these
    fixtures do.
    """
    key = round(float(fd), 12)
    if key not in _LEG_CAL:
        s0 = float(window) * float(fd) / (_N * _DX)      # first guess
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            probe = la.propagate_traced_carrier_chain(
                field, groups, _WL, _DX, r_in=carrier, ray_subsample=32,
                n_workers=1, final_distance=fd, traced_kwargs=_TKW,
                final_leg='paraxial',
                focus_readout=dict(dx_out=_DXO, N_out=8, standoff=s0,
                                   on_replica='ignore'))
        per = float(probe.stages[-1]['readout_period'][0])
        _LEG_CAL[key] = per / s0                # period per metre of leg
    return margin * float(window) / _LEG_CAL[key]


def _singlet(R1, R2, d, glass, ap, name):
    surfaces = [
        {'radius': R1, 'glass_before': 'air', 'glass_after': glass,
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
        {'radius': R2, 'glass_before': glass, 'glass_after': 'air',
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]
    return {'name': name, 'aperture_diameter': ap,
            'surfaces': surfaces, 'thicknesses': [d]}


def _gauss(n=_N, dx=_DX, w=_W_IN):
    x = (np.arange(n) - n // 2) * dx
    return np.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / w ** 2
                  ).astype(np.complex128)


def _relay_groups(ap=10e-3, d0=0.0):
    gA = _singlet(60e-3, -60e-3, 3.0e-3, 'N-BK7', ap, 'gA')
    gB = _singlet(60e-3, -60e-3, 3.0e-3, 'N-BK7', ap, 'gB')
    return gA, gB, [{'prescription': gA, 'gap_before': d0},
                    {'prescription': gB, 'gap_before': _GAP}]


def _total_abcd(gA, gB, d0):
    """Input plane -> last group's exit vertex, by hand from the group ABCDs
    (the independent paraxial oracle for every placement claim below)."""
    MA = np.array(_group_abcd(gA, _WL)).reshape(2, 2)
    MB = np.array(_group_abcd(gB, _WL)).reshape(2, 2)
    return (MB @ np.array([[1.0, _GAP], [0.0, 1.0]])
            @ MA @ np.array([[1.0, d0], [0.0, 1.0]]))


def _exact_group_step(presc, x, y, L, M):
    """One group, front vertex -> back vertex, by an EXACT skew ray trace.

    Written out here rather than imported so this oracle keeps sharing no
    code with the chain's own ``_group_chief_transfer`` (niche C3).
    """
    sf = [dataclasses.replace(s, semi_diameter=np.inf)
          for s in surfaces_from_prescription(presc)]
    sf[-1] = dataclasses.replace(sf[-1], thickness=0.0)
    sf = sf + [Surface(radius=np.inf, conic=0.0, semi_diameter=np.inf,
                       glass_before='air', glass_after='air',
                       is_mirror=False, thickness=0.0, label='vtx')]
    im = trace(make_ray(float(x), float(y), float(L), float(M),
                        wavelength=_WL), sf, _WL,
               output_filter='last').image_rays
    return (float(np.asarray(im.x).ravel()[0]),
            float(np.asarray(im.y).ravel()[0]),
            float(np.asarray(im.L).ravel()[0]),
            float(np.asarray(im.M).ravel()[0]))


def _hand_chief_ray(gA, gB, L, M, fd, d0=0.0, groups_paraxial=False):
    """Chief ray at the target plane, computed BY HAND: each GROUP by an exact
    skew ray trace through its real surfaces, each FREE LEG by the exact
    ``1/cos(theta)`` obliquity -- the same closure the chain uses, written out
    independently here so the placement bookkeeping is checked against
    something that shares no code with it.

    ``groups_paraxial=True`` selects the OLD lumped-ABCD group step, kept as
    the fail-before witness: niche C3 replaced it because a group ABCD is
    neither a sine nor a tangent convention (Snell is linear in sines, free
    transfer in tangents), which left a measurable error at large tilt.
    """
    def _step(presc, x, y, l, m):
        if groups_paraxial:
            A, B, C, D = _group_abcd(presc, _WL)
            return A * x + B * l, A * y + B * m, C * x + D * l, C * y + D * m
        return _exact_group_step(presc, x, y, l, m)

    ob0 = 1.0 / np.sqrt(1.0 - L ** 2 - M ** 2) if (L or M) else 1.0
    x0, y0 = L * d0 * ob0, M * d0 * ob0
    xA, yA, LA, MA = _step(gA, x0, y0, L, M)
    obA = 1.0 / np.sqrt(1.0 - LA ** 2 - MA ** 2) if (LA or MA) else 1.0
    xg, yg = xA + LA * _GAP * obA, yA + MA * _GAP * obA
    xB, yB, LB, MB = _step(gB, xg, yg, LA, MA)
    obB = 1.0 / np.sqrt(1.0 - LB ** 2 - MB ** 2) if (LB or MB) else 1.0
    return xB + LB * fd * obB, yB + MB * fd * obB, LB, MB


def _hand_tile(x, y):
    """Lattice snap + index origin of a readout tile, by hand."""
    mx, my = int(round(x / _DXO)), int(round(y / _DXO))
    return ((mx * _DXO, my * _DXO),
            (my + (_NOUT - _TILE) // 2, mx + (_NOUT - _TILE) // 2))


def _exact_chief_height(gA, gB, tilt, image_distance, d0=0.0):
    """EXACT meridional trace of the tilted chief ray through the REAL
    spherical surfaces -- no paraxial approximation anywhere."""
    sa = surfaces_from_prescription(gA)
    sb = surfaces_from_prescription(gB)
    sa[-1] = dataclasses.replace(sa[-1], thickness=_GAP)
    sb[-1] = dataclasses.replace(sb[-1], thickness=image_distance)
    surfs = sa + sb + [Surface(
        radius=np.inf, conic=0.0, semi_diameter=np.inf, glass_before='air',
        glass_after='air', is_mirror=False, thickness=0.0, label='img')]
    res = trace(make_ray(float(tilt * d0), 0.0, float(tilt), 0.0,
                         wavelength=_WL), surfs, _WL)
    return float(res.image_rays.x[0])


def _run_chain(groups, field, carrier, fd, centre, n_out=_NOUT, quiet=True):
    """One INDEPENDENT single-congruence chain run, read out on the common
    grid -- the hand-written form of what the orchestrator does per
    congruence.

    D3 (2026-08-06): the leg is sized for the TILE, not for the whole common
    grid, so this run is bit-comparable with the orchestrated one inside the
    tile -- which is the only region any caller of this helper compares.  The
    common grid is 2.867 mm against a tile-sized period, so the outer part of
    THIS array carries replicas; ``on_replica='ignore'`` acknowledges that
    rather than hiding it, and no assertion in this file reads outside the
    tile.
    """
    def _go():
        return la.propagate_traced_carrier_chain(
            field, groups, _WL, _DX, r_in=carrier, ray_subsample=_RS,
            n_workers=_NW, final_distance=fd, traced_kwargs=_TKW,
            final_leg='paraxial',
            focus_readout=dict(
                dx_out=_DXO, N_out=n_out, centre_out=centre,
                standoff=_leg_for_window(groups, field, carrier, fd,
                                         _TILE * _DXO),
                on_replica='ignore'))
    if not quiet:
        return _go()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return _go()


def _run_multi(groups, specs, fd, **kw):
    # D3: same leg as _run_chain, so the tiled and hand-placed arms are
    # comparing the same physics rather than two different hand-off planes.
    kw.setdefault('output_grid',
                  dict(dx_out=_DXO, N_out=_NOUT,
                       standoff=_leg_for_window(
                           groups, specs[0]['field'], specs[0].get('carrier'),
                           fd, _TILE * _DXO)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return la.propagate_traced_carrier_chain_multi(
            specs, groups, _WL, _DX, final_distance=fd, ray_subsample=_RS,
            n_workers=_NW, traced_kwargs=_TKW, final_leg='paraxial', **kw)


# ===========================================================================
# 1.  K = 1 reduces to the single-congruence chain
# ===========================================================================

def test_k1_reduces_to_the_single_congruence_chain():
    """THE regression pin: with one unit-weight congruence the orchestrator
    must reproduce ``propagate_traced_carrier_chain`` -- so the shipped
    design-121 acceptance (FWHM 3.450 / EE3 88.8 / EE6 99.6) cannot move by
    being routed through this entry point.

    Compared with a TOLERANCE (<= 1e-10 * scale) rather than array_equal:
    both arms are live FFT work.  (Measured margin here: exactly 0.)"""
    n, dx, w, R_in = 512, 30e-6, 4.5e-3, 60e-3
    presc = _singlet(60e-3, -60e-3, 3e-3, 'N-BK7', 14e-3, 'p')
    env = _gauss(n, dx, w)
    groups = [{'prescription': presc, 'gap_before': 20e-3},
              {'prescription': presc, 'gap_before': 10e-3}]
    fr = dict(dx_out=0.5e-6, N_out=256)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        single = la.propagate_traced_carrier_chain(
            env, groups, _WL, dx, r_in=R_in, ray_subsample=_RS, n_workers=1,
            final_distance=8e-3, traced_kwargs=_TKW, final_leg='paraxial',
            focus_readout=fr)
        multi = la.propagate_traced_carrier_chain_multi(
            [{'field': env, 'carrier': R_in}], groups, _WL, dx,
            output_grid=fr, final_distance=8e-3, ray_subsample=_RS,
            n_workers=1, traced_kwargs=_TKW, final_leg='paraxial')
    A, B = np.asarray(single.field), np.asarray(multi.field)
    assert A.shape == B.shape and A.dtype == B.dtype
    margin = float(np.abs(A - B).max())
    assert margin <= 1e-10 * float(np.abs(A).max()), margin
    assert multi.dx == pytest.approx(0.5e-6)
    assert multi.centre == (0.0, 0.0)
    assert len(multi.congruences) == 1
    assert multi.congruences[0]['stages'] == single.stages


# ===========================================================================
# 2.  the fan: hand-summed independent runs, full grid and tiled
# ===========================================================================

@pytest.fixture(scope='module')
def _fan():
    gA, gB, groups = _relay_groups()
    M = _total_abcd(gA, gB, 0.0)
    A2, B2, C2, D2 = _group_abcd(gB, _WL)
    R_A = _group_abcd(gA, _WL)[0] / _group_abcd(gA, _WL)[2]
    R_g = R_A + _GAP
    R_B = (A2 * R_g + B2) / (C2 * R_g + D2)
    fd = -R_B                                   # land on the geometric focus
    env = _gauss()
    tilts = [(-_TILT, 0.0), (+_TILT, 0.0), (0.0, -_TILT), (0.0, +_TILT)]
    weights = [0.5, 0.5j, -0.25, 0.75]
    specs = [{'field': env, 'name': f'c{i}', 'weight': wt,
              'carrier': la.TiltedCarrier(np.inf, L, Mt)}
             for i, ((L, Mt), wt) in enumerate(zip(tilts, weights))]
    # BY HAND: predict each chief ray, snap its tile onto the common lattice,
    # run that congruence ALONE on that tile, and record where it belongs.
    hand_state = [_hand_chief_ray(gA, gB, L, Mt, fd) for (L, Mt) in tilts]
    hand_tiles = []
    for s, st in zip(specs, hand_state):
        cen, org = _hand_tile(st[0], st[1])
        run = _run_chain(groups, env, s['carrier'], fd, cen, n_out=_TILE)
        hand_tiles.append((np.asarray(run.field), org))
    # ... and the same congruences on the FULL common grid (the naive route)
    hand_full = [_run_chain(groups, env, s['carrier'], fd, (0.0, 0.0))
                 for s in specs[:2]]
    return dict(gA=gA, gB=gB, groups=groups, M=M, fd=fd, env=env,
                specs=specs, weights=weights, tilts=tilts,
                hand_state=hand_state, hand_tiles=hand_tiles,
                hand_full=hand_full)


def _hand_sum_full(_fan, k):
    out = np.zeros_like(np.asarray(_fan['hand_full'][0].field))
    for i in range(k):
        out += _fan['weights'][i] * np.asarray(_fan['hand_full'][i].field)
    return out


def _hand_sum_tiled(_fan, k):
    """Place the K hand-run tiles into the common grid at the hand-computed
    integer index origins and sum -- the bookkeeping the orchestrator claims
    to be doing, written out longhand."""
    out = np.zeros((_NOUT, _NOUT), dtype=np.complex128)
    for i in range(k):
        tile, (r0, c0) = _fan['hand_tiles'][i]
        out[r0:r0 + _TILE, c0:c0 + _TILE] += _fan['weights'][i] * tile
    return out


@pytest.fixture(scope='module')
def _multi2_full(_fan):
    # readout_tile=None is the HISTORICAL full-grid readout, and on this
    # fixture it sits in the periodic-replica regime (see
    # test_default_refuses_the_periodic_replica_regime, which pins that the
    # DEFAULT now refuses it).  It is kept here ONLY as a bookkeeping
    # equivalence: whatever the readout returns, the orchestrator must place
    # and sum it exactly as K independent runs would.  on_replica='ignore' is
    # therefore deliberate and is not a physics claim.
    return _run_multi(_fan['groups'], _fan['specs'][:2], _fan['fd'],
                      readout_tile=None, on_replica='ignore',
                      on_readout_clip='ignore')


@pytest.fixture(scope='module')
def _multi2_tiled(_fan):
    return _run_multi(_fan['groups'], _fan['specs'][:2], _fan['fd'],
                      readout_tile=_TILE)


@pytest.fixture(scope='module')
def _multi4_tiled(_fan):
    return _run_multi(_fan['groups'], _fan['specs'], _fan['fd'],
                      readout_tile=_TILE)


def test_k2_full_grid_equals_hand_summed_independent_runs(_fan, _multi2_full):
    """The orchestrator on the full common grid IS the hand-written sum of two
    independent single-congruence chain runs -- weights included, nothing else
    applied.

    BOOKKEEPING ONLY.  This fixture runs ``readout_tile=None`` with
    ``on_replica='ignore'``, i.e. deliberately inside the periodic-replica
    regime, so that the placement/accumulation arithmetic is checked against
    the naive route on identical (replica-contaminated) inputs.  The physics
    claim about that regime is the opposite one and lives in
    ``test_default_refuses_the_periodic_replica_regime``."""
    ref = _hand_sum_full(_fan, 2)
    got = np.asarray(_multi2_full.field)
    scale = float(np.abs(ref).max())
    assert scale > 0.0
    assert float(np.abs(got - ref).max()) <= 1e-10 * scale


def test_k2_tiled_equals_hand_placed_independent_runs(_fan, _multi2_tiled):
    """THE bookkeeping test.  Each congruence is read out on its own
    256-square window centred on its own chief ray and accumulated at an
    integer index offset; the answer must equal the same two runs placed and
    summed BY HAND from an independent ABCD + obliquity prediction."""
    ref = _hand_sum_tiled(_fan, 2)
    got = np.asarray(_multi2_tiled.field)
    scale = float(np.abs(ref).max())
    assert scale > 0.0
    covered = np.abs(ref) > 0.0
    assert covered.sum() == 2 * _TILE ** 2          # disjoint tiles, no overlap
    assert float(np.abs(got - ref).max()) <= 1e-10 * scale
    # the orchestrator painted exactly the same pixels
    assert np.array_equal(np.abs(got) > 0.0, covered)


def test_k4_tiled_equals_hand_placed_independent_runs(_fan, _multi4_tiled):
    """K=4 (a 2x2 fan, two of them on the y axis) through the tiles."""
    ref = _hand_sum_tiled(_fan, 4)
    got = np.asarray(_multi4_tiled.field)
    scale = float(np.abs(ref).max())
    assert float(np.abs(got - ref).max()) <= 1e-10 * scale
    assert np.array_equal(np.abs(got) > 0.0, np.abs(ref) > 0.0)
    assert (np.abs(got) > 0.0).sum() == 4 * _TILE ** 2
    assert np.isfinite(got).all()


def test_tiles_avoid_the_periodic_replica_regime(_fan, _multi2_tiled):
    """A finding worth pinning, and the reason ``readout_tile`` is not only a
    memory lever.  Near a focus the co-moving grid COLLAPSES, so a common
    output window several mm wide can exceed one spatial period of the
    Bluestein reconstruction: the naive full-grid readout of an off-axis
    congruence then fills the rest of the lattice with periodic REPLICAS of
    its own spot -- a populated, credible-looking frame lattice with the wrong
    per-frame power, exactly the v5.28 failure class.

    Measured here: the full-grid window holds ~6x the power that is really in
    the frame, and ``angular_spectrum_propagate_mft`` says so; the tile does
    not trip it, and inside the tile the two agree to ~1e-12 relative."""
    tile, (r0, c0) = _fan['hand_tiles'][0]
    full = np.asarray(_fan['hand_full'][0].field)
    inside = full[r0:r0 + _TILE, c0:c0 + _TILE]
    scale = float(np.abs(tile).max())
    assert float(np.abs(tile - inside).max()) <= 1e-10 * scale
    p_tile = float((np.abs(tile) ** 2).sum())
    p_full = float((np.abs(full) ** 2).sum())
    assert p_full > 3.0 * p_tile, (p_full, p_tile)      # replicas, not signal
    # ... and the library says so on the full-grid route
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        run = _run_chain(_fan['groups'], _fan['env'],
                         _fan['specs'][0]['carrier'], _fan['fd'], (0.0, 0.0),
                         quiet=False)
    assert any('exceeds one spatial period' in str(w.message) for w in rec)
    # ... and it REPORTS the period, so an accumulating caller can act on it
    # without depending on a warning that any upstream filterwarnings()
    # silences (this is what the orchestrator's guard reads).
    period = run.stages[-1]['readout_period']
    assert period is not None
    px, py = float(period[0]), float(period[1])
    assert px == pytest.approx(py, rel=1e-12)
    assert _NOUT * _DXO > px > _TILE * _DXO, (px, _NOUT * _DXO, _TILE * _DXO)


def test_default_refuses_the_periodic_replica_regime(_fan):
    """FAIL-BEFORE / PASS-AFTER for the adversarial kill: the orchestrator's
    DEFAULT used to reproduce the very v5.28 failure it exists to fix -- a
    naive full-grid readout of an off-axis congruence wraps periodic REPLICAS
    of its own spot onto the neighbouring frames, and nothing in the
    orchestrator said so (the only signal was a downstream
    ``angular_spectrum_propagate_mft`` UserWarning, which the campaign's own
    acceptance runner silenced with ``filterwarnings('ignore')``).

    Now, for K > 1 (the MULTIPLEXED case the guard is about -- see
    ``test_k1_keeps_the_requested_field_of_view`` for the K=1 contract):

    * ``readout_tile=None`` in that regime RAISES by default, naming the
      period and the largest window that fits it;
    * the DEFAULT ``readout_tile='auto'`` sizes the window down for you, says
      so, and lands within one period."""
    fn = la.propagate_traced_carrier_chain_multi
    kw = dict(output_grid=dict(dx_out=_DXO, N_out=_NOUT),
              final_distance=_fan['fd'], ray_subsample=_RS, n_workers=_NW,
              traced_kwargs=_TKW, final_leg='paraxial')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(RuntimeError, match='REPLICAS'):
            fn(_fan['specs'][:2], _fan['groups'], _WL, _DX,
               readout_tile=None, **kw)
        # the same window as an EXPLICIT tile is refused too -- the guard is
        # on the window, not on the mode
        with pytest.raises(RuntimeError, match='spatial period'):
            fn(_fan['specs'][:2], _fan['groups'], _WL, _DX,
               readout_tile=_NOUT, **kw)
    # ... and 'auto' resolves it, loudly.  D3 (2026-08-06): 'auto' sizes the
    # tile to ONE PERIOD, so the readout's own self-replica guard is silent on
    # this arm by construction -- which is the composition the two guards are
    # meant to have, and is asserted by the fact that this call succeeds at
    # the shipped on_replica default.
    seen = []
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        res = fn(_fan['specs'][:2], _fan['groups'], _WL, _DX,
                 on_readout_clip='ignore',
                 progress=lambda k, K, nm: seen.append((k, K, nm)), **kw)
    assert any("readout_tile='auto'" in str(w.message) for w in rec), [
        str(w.message)[:80] for w in rec]
    # 'auto' costs 2K chain runs: a labelled PROBE pass (which is what makes
    # the sizing independent of congruence order) plus the real pass
    assert len(seen) == 4, seen
    assert sum('(period probe)' in nm for _k, _K, nm in seen) == 2, seen
    p_min = min(info['readout_period'] for info in res.congruences)
    for info in res.congruences:
        assert info['tile'] < _NOUT
        assert info['tile'] % 2 == 0
        # the orchestrator reports the period as a SCALAR (the smaller axis),
        # and EVERY congruence's window fits inside its OWN period
        assert info['tile'] * _DXO <= info['readout_period']
    # ... sized from the SHORTEST period over all K, and maximal against it
    assert (res.congruences[0]['tile'] + 2) * _DXO > p_min
    # every congruence used the SAME window (they must be commensurable)
    assert len({info['tile'] for info in res.congruences}) == 1
    assert np.isfinite(np.asarray(res.field)).all()


def test_auto_window_is_independent_of_congruence_order(_fan):
    """SECOND adversarial kill, fail-before/pass-after: ``'auto'`` used to
    size the shared window from congruence 0 alone and then lock it, so every
    LATER congruence with a SHORTER period hit ``on_replica='error'`` -- i.e.
    whether the library's own DEFAULT ran at all depended on the order the
    caller happened to list the congruences in.  On the real design-121 fan
    (per-order periods spanning 1.8 %) that made the default RAISE on the
    roadmap's own acceptance configuration.

    Two congruences with deliberately DIFFERENT periods (different beam
    widths -> different focus standoff -> different co-moving grid at the
    readout).  Both orders must run, pick the SAME window -- the one set by
    the SHORTER period -- and give the same field."""
    fn = la.propagate_traced_carrier_chain_multi
    kw = dict(output_grid=dict(dx_out=_DXO, N_out=_NOUT),
              final_distance=_fan['fd'], ray_subsample=_RS, n_workers=_NW,
              traced_kwargs=_TKW, final_leg='paraxial',
              on_readout_clip='ignore')
    specs = [{'field': _gauss(w=1.0e-3), 'name': 'wide',
              'carrier': la.TiltedCarrier(np.inf, _TILT, 0.0)},
             {'field': _gauss(w=0.62e-3), 'name': 'narrow',
              'carrier': la.TiltedCarrier(np.inf, 0.012, 0.0)}]
    out = {}
    for label, order in (('as-listed', [0, 1]), ('reversed', [1, 0])):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out[label] = fn([specs[i] for i in order], _fan['groups'], _WL,
                            _DX, **kw)
    fwd, rev = out['as-listed'], out['reversed']
    periods = {c['name']: c['readout_period'] for c in fwd.congruences}
    # the premise: the two congruences really do have different periods
    assert max(periods.values()) > 1.2 * min(periods.values()), periods
    tiles = ({c['tile'] for c in fwd.congruences}
             | {c['tile'] for c in rev.congruences})
    assert len(tiles) == 1, (tiles, periods)
    n = tiles.pop()
    assert n * _DXO <= min(periods.values())
    assert (n + 2) * _DXO > min(periods.values())
    A, B = np.asarray(fwd.field), np.asarray(rev.field)
    scale = float(np.abs(A).max())
    assert scale > 0.0
    assert float(np.abs(A - B).max()) <= 1e-10 * scale


def test_k1_keeps_the_requested_field_of_view(_fan):
    """THIRD adversarial kill, fail-before/pass-after.  The replica guard is a
    MULTIPLEXING guard: with one congruence the recombination is a no-op, the
    replicas stay inside that congruence's own window, and the answer IS
    ``propagate_traced_carrier_chain``'s.  An earlier cut applied the guard
    anyway, so a clean K=1 call at pure defaults silently shrank the requested
    2.87 mm window to 1.29 mm and returned ZEROS over the rest -- while
    ``readout_tile=None`` was refused outright, leaving no default route to
    the field of view the caller asked for.

    Now K=1 reproduces the chain over the WHOLE requested grid, and says why
    (a warning naming the period, plus ``capture`` > 1 because the replicas
    add power) instead of quietly re-sizing.

    D3 (2026-08-06): the READOUT now has its own SELF-replica guard, and that
    one is NOT a multiplexing guard -- a spot wrapping onto itself needs no
    neighbour -- so it fires at K = 1 too and the pure-default call is now
    REFUSED.  That is deliberate: this window is 24.5x the period, i.e. 96 % of
    what it returns is copies.  The K = 1 CONTRACT is unchanged and is what
    this test still pins: the multi at K = 1 must do exactly what the single
    chain does, refusal included, and with the self-replica guard waived it
    must return the same field over the same whole window.  Both halves are
    asserted below.
    """
    fn = la.propagate_traced_carrier_chain_multi
    # the SAME leg the hand-placed reference uses, or the two would be
    # comparing different hand-off planes rather than the same physics.
    _leg = _leg_for_window(_fan['groups'], _fan['env'],
                           _fan['specs'][0]['carrier'], _fan['fd'],
                           _TILE * _DXO)
    kw = dict(output_grid=dict(dx_out=_DXO, N_out=_NOUT, standoff=_leg),
              final_distance=_fan['fd'], ray_subsample=_RS, n_workers=_NW,
              traced_kwargs=_TKW, final_leg='paraxial')
    # (a) the default REFUSES, at K = 1 as at K > 1 ...
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(RuntimeError, match='REPLICAS'):
            fn(_fan['specs'][:1], _fan['groups'], _WL, _DX, **kw)
    # (b) ... and with the SELF-replica guard waived the historical K = 1
    #     field-of-view contract holds verbatim.
    kw['output_grid'] = dict(kw['output_grid'], on_replica='warn')
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        res = fn(_fan['specs'][:1], _fan['groups'], _WL, _DX, **kw)
    ref = _hand_sum_full(_fan, 1)               # the chain, weight applied
    got = np.asarray(res.field)
    scale = float(np.abs(ref).max())
    assert scale > 0.0
    assert got.shape == (_NOUT, _NOUT)
    assert float(np.abs(got - ref).max()) <= 1e-10 * scale
    # the whole requested window is live -- nothing was zeroed away
    assert res.congruences[0]['tile'] == _NOUT
    assert float(np.abs(got[:_TILE, :_TILE]).max()) > 0.0
    # ... and it is NOT silent: the period is named, and capture > 1 says the
    # extra power is replicas
    msgs = [str(w.message) for w in rec]
    assert any('spatial period' in m and 'REPLICAS' in m for m in msgs), [
        m[:90] for m in msgs]
    assert not any("readout_tile='auto'" in m for m in msgs), [
        m[:90] for m in msgs]
    assert res.congruences[0]['capture'] > 1.5
    # 'ignore' silences the orchestrator's copy of the notice
    with warnings.catch_warnings(record=True) as rec2:
        warnings.simplefilter('always')
        fn(_fan['specs'][:1], _fan['groups'], _WL, _DX, on_replica='ignore',
           on_readout_clip='ignore', **kw)
    assert not any('REPLICAS' in str(w.message) and 'congruence' in
                   str(w.message) for w in rec2)


def test_auto_tile_equals_the_same_tile_asked_for_explicitly(_fan):
    """``readout_tile='auto'`` is a SIZING convenience and nothing else: the
    field it produces is bit-for-bit (to <= 1e-10 * scale) what the same tile
    asked for explicitly produces."""
    fn = la.propagate_traced_carrier_chain_multi
    kw = dict(output_grid=dict(dx_out=_DXO, N_out=_NOUT),
              final_distance=_fan['fd'], ray_subsample=_RS, n_workers=_NW,
              traced_kwargs=_TKW, final_leg='paraxial',
              on_readout_clip='ignore')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        auto = fn(_fan['specs'][:2], _fan['groups'], _WL, _DX, **kw)
        n = auto.congruences[0]['tile']
        fixed = fn(_fan['specs'][:2], _fan['groups'], _WL, _DX,
                   readout_tile=n, **kw)
    A, B = np.asarray(auto.field), np.asarray(fixed.field)
    scale = float(np.abs(A).max())
    assert scale > 0.0
    assert float(np.abs(A - B).max()) <= 1e-10 * scale
    assert [c['tile_origin'] for c in auto.congruences] == \
        [c['tile_origin'] for c in fixed.congruences]


def test_tiles_land_on_the_exact_ray_trace(_fan, _multi4_tiled):
    """Placement oracle.  Each congruence's readout window must be centred on
    where an EXACT meridional ray trace through the real spherical surfaces
    puts that chief ray -- not on a paraxial stand-in, and not on the grid
    centre."""
    for info, (L, Mt) in zip(_multi4_tiled.congruences, _fan['tilts']):
        tilt = L if L else Mt
        x_exact = _exact_chief_height(_fan['gA'], _fan['gB'], tilt, _fan['fd'])
        axis = 0 if L else 1
        got = info['chief_ray'][axis]
        assert got == pytest.approx(x_exact, abs=0.5e-6), (info['name'], got,
                                                           x_exact)
        # the OTHER axis is exactly on axis
        assert info['chief_ray'][1 - axis] == 0.0
        # tile centre is the lattice snap of the prediction (sub-pixel)
        assert abs(info['tile_centre'][axis] - got) <= 0.5 * _DXO
        # ... and the image height is big enough that this is a real test
        assert abs(x_exact) > 1.0e-3


def test_measured_lobes_sit_at_the_predicted_frame_centres(_fan,
                                                           _multi4_tiled):
    """End to end: the recombined field's four lobes are AT the four predicted
    frame centres, and each is a clean single peak."""
    I = np.abs(np.asarray(_multi4_tiled.field)) ** 2
    ax = (np.arange(_NOUT) - _NOUT / 2) * _DXO
    peaks = {}
    for info in _multi4_tiled.congruences:
        cx, cy = info['chief_ray']
        m = ((np.abs(ax[None, :] - cx) < 30e-6)
             & (np.abs(ax[:, None] - cy) < 30e-6))
        sub = np.where(m, I, 0.0)
        iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
        assert abs(ax[ix] - cx) < 3.0e-6 and abs(ax[iy] - cy) < 3.0e-6
        peaks[info['name']] = sub.max()
    # each lobe's peak scales as |weight|^2 -- the weights are applied once,
    # at the image plane, and nowhere else
    ref = None
    for info, wt in zip(_multi4_tiled.congruences, _fan['weights']):
        r = peaks[info['name']] / abs(wt) ** 2
        if ref is None:
            ref = r
        assert r == pytest.approx(ref, rel=0.02), (info['name'], r, ref)


def test_per_congruence_power_is_conserved_and_reported(_fan, _multi4_tiled):
    """Per-congruence power bookkeeping across THREE planes: ``power_in``
    (chain input, carrying the weight), ``power_exit`` (last group's exit --
    what the chain delivered) and ``power_out`` (this congruence's readout
    window).  No energy is created anywhere along the way."""
    p_env = float((np.abs(_fan['env']) ** 2).sum()) * _DX * _DX
    for info, wt in zip(_multi4_tiled.congruences, _fan['weights']):
        assert info['power_in'] == pytest.approx(p_env * abs(wt) ** 2,
                                                 rel=1e-12)
        assert info['power_exit'] <= info['power_in'] * (1.0 + 1e-9)
        assert info['throughput'] == pytest.approx(
            info['power_exit'] / info['power_in'], rel=1e-12)
        assert info['throughput'] <= 1.0 + 1e-9
        assert info['throughput'] > 0.98
        assert info['capture'] == pytest.approx(
            info['power_out'] / info['power_exit'], rel=1e-12)
        assert info['clipped'] == 0.0
        assert info['weight'] == complex(wt)
        assert np.isfinite(info['power_out'])
    # ... and the coherent total is bounded by the incoherent total (the
    # frames are disjoint here, so they must in fact be equal)
    tot = float((np.abs(np.asarray(_multi4_tiled.field)) ** 2).sum()) \
        * _DXO * _DXO
    assert tot == pytest.approx(
        sum(i['power_out'] for i in _multi4_tiled.congruences), rel=1e-9)


def test_throughput_is_window_independent_but_power_out_is_not(_fan):
    """THE kill-1 pin.  ``power_out`` is measured in the readout WINDOW, so it
    grows with the tile; the campaign's first report read its ~1.5% spread
    across a 32-order fan as "field-angle-dependent vignetting" when it was
    the tile clipping a field-angle-dependent halo -- bookkeeping, not
    physics.

    So ``throughput`` is now measured at the CHAIN EXIT and must be identical
    across tile sizes to ~1e-12, while ``power_out`` / ``capture`` must
    visibly move.  Two congruences at DIFFERENT field angles (on-axis-ish and
    the 30 mrad edge) so the field-angle dependence is in the sample."""
    fn = la.propagate_traced_carrier_chain_multi
    specs = [{'field': _fan['env'], 'name': 'axis',
              'carrier': la.TiltedCarrier(np.inf, 2e-3, 0.0)},
             {'field': _fan['env'], 'name': 'edge',
              'carrier': la.TiltedCarrier(np.inf, _TILT, 0.0)}]
    kw = dict(output_grid=dict(dx_out=_DXO, N_out=_NOUT),
              final_distance=_fan['fd'], ray_subsample=_RS, n_workers=_NW,
              traced_kwargs=_TKW, final_leg='paraxial',
              on_readout_clip='ignore')
    out = {}
    for n in (16, 64):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out[n] = fn(specs, _fan['groups'], _WL, _DX, readout_tile=n, **kw)
    a, b = out[16].congruences, out[64].congruences
    for ia, ib in zip(a, b):
        # window-INDEPENDENT: the real vignetting
        assert ia['throughput'] == pytest.approx(ib['throughput'], rel=1e-12)
        # window-DEPENDENT: the tile captures strictly more halo when bigger
        assert ib['power_out'] > ia['power_out']
        assert ib['capture'] > ia['capture']
    assert a[0]['capture'] < 0.8 and b[0]['capture'] > 0.999   # real clipping
    # THE point: the APPARENT per-frame power ratio between the two field
    # angles is a function of the tile, and only converges onto the true
    # (window-independent) throughput ratio once the tile stops clipping.
    # Measured on this fixture: 1.0051 at a 22 um tile vs a true 0.99988.
    thr_ratio = a[1]['throughput'] / a[0]['throughput']
    assert b[1]['throughput'] / b[0]['throughput'] == pytest.approx(
        thr_ratio, rel=1e-12)
    apparent = {n: r.congruences[1]['power_out'] / r.congruences[0]['power_out']
                for n, r in out.items()}
    assert abs(apparent[16] / thr_ratio - 1.0) > 1e-3          # fools you ...
    assert abs(apparent[64] / thr_ratio - 1.0) < 1e-5          # ... then does not
    assert abs(apparent[64] / thr_ratio - 1.0) \
        < 0.02 * abs(apparent[16] / thr_ratio - 1.0)


def test_a_clipping_readout_window_is_reported(_fan):
    """``on_readout_clip`` (default 'warn') names the window that is throwing
    away part of the beam, because a silently-clipped halo turns ``power_out``
    into a field-angle-dependent number that reads as vignetting."""
    fn = la.propagate_traced_carrier_chain_multi
    kw = dict(output_grid=dict(dx_out=_DXO, N_out=_NOUT),
              final_distance=_fan['fd'], ray_subsample=_RS, n_workers=_NW,
              traced_kwargs=_TKW, final_leg='paraxial')
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        res = fn(_fan['specs'][:1], _fan['groups'], _WL, _DX,
                 readout_tile=16, readout_capture_tol=1e-3, **kw)
    assert any('CLIPPING' in str(w.message) for w in rec), [
        str(w.message)[:80] for w in rec]
    assert res.congruences[0]['capture'] < 1.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(RuntimeError, match='readout window holds'):
            fn(_fan['specs'][:1], _fan['groups'], _WL, _DX, readout_tile=16,
               readout_capture_tol=1e-3, on_readout_clip='error', **kw)


def test_chief_ray_prediction_matches_the_chain_bookkeeping(_fan,
                                                            _multi4_tiled):
    """The analytic tile placement and the chain's OWN tracked chief ray are
    two independent copies of the D1 closure; the orchestrator raises if they
    drift, and here they must agree to sub-nanometre."""
    for info in _multi4_tiled.congruences:
        tgt = info['stages'][-1]
        assert tgt.get('target') is True
        assert float(tgt['x_c']) == pytest.approx(info['chief_ray'][0],
                                                  abs=1e-12)
        assert float(tgt['y_c']) == pytest.approx(info['chief_ray'][1],
                                                  abs=1e-12)


def test_exit_tilt_matches_the_exact_chief_trace(_fan, _multi4_tiled):
    """The reported exit direction cosines are those of the chief ray TRACED
    exactly through both groups (niche C3), not the lumped-ABCD composition
    ``L_out = C x_c + D L`` this test used to pin.  The paraxial stand-in is
    kept below as the discriminator: it is close but measurably not equal."""
    M = _fan['M']
    for info, (L, Mt), st in zip(_multi4_tiled.congruences, _fan['tilts'],
                                 _fan['hand_state']):
        # exact (with the free-leg 1/cos(theta) obliquity)
        assert info['exit_tilt'][0] == pytest.approx(st[2], rel=1e-12)
        assert info['exit_tilt'][1] == pytest.approx(st[3], rel=1e-12)
        # ... and the fully-paraxial D*L stand-in is CLOSE but not equal, so
        # the pin above genuinely discriminates the obliquity term
        par = M[1, 1] * (L or Mt)
        exact = st[2] or st[3]
        assert exact == pytest.approx(par, rel=1e-3)
        assert exact != pytest.approx(par, rel=1e-6)


# ===========================================================================
# 3.  interference is real
# ===========================================================================

_D0_CONJ = 200e-3
_T_INT = 0.010
_DXO_I, _NOUT_I = 1.0e-6, 1024


@pytest.fixture(scope='module')
def _conjugate():
    """A relay run at the INPUT-PLANE CONJUGATE (``B_total = 0``): two
    congruences tilted +-t then land at the SAME image point with exit angles
    ``+-D_t t``, so their overlap is a two-beam fringe of period
    ``lambda / (2 D_t t) = lambda |A_t| / (2 t)`` -- an exact analytic oracle
    that shares no code with the orchestrator."""
    gA, gB, groups = _relay_groups(ap=14e-3, d0=_D0_CONJ)
    M = _total_abcd(gA, gB, _D0_CONJ)
    A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    fd = -B / D                              # B_total(fd) = 0 -> conjugate
    A_t = A + fd * C
    assert fd > 0.0 and abs(A_t * D - 1.0) < 1e-9
    period = _WL * abs(A_t) / (2.0 * _T_INT)
    env = _gauss()
    specs = [{'field': env, 'name': f'a{s:+d}',
              'carrier': la.TiltedCarrier(np.inf, s * _T_INT, 0.0)}
             for s in (-1, +1)]
    # A WIDE readout window physically requires a LONGER Bluestein leg: the
    # reconstruction's spatial period is N_in * d_in and the co-moving pitch at
    # the hand-off plane scales with the standoff, so
    #     period = _N * _DX * standoff / z_focus.
    # The shipped default sizes the standoff for READOUT ACCURACY (0.8 zR), which
    # gives a 171.7 um period -- ample for a spot, far too small for the +-150 um
    # fringe analysis these tests do.  So invert the relation and ask for the
    # shortest leg that still holds the requested window (30% margin).
    # Downgrading the replica guard instead is NOT a valid alternative here: it
    # was measured to corrupt the recombined power by 2.4%.
    _need = 1.3 * (_NOUT_I * _DXO_I) * fd / (_N * _DX)
    og = dict(dx_out=_DXO_I, N_out=_NOUT_I, standoff=_need)
    kw = dict(output_grid=og, final_distance=fd, ray_subsample=_RS,
              n_workers=_NW, traced_kwargs=_TKW, final_leg='paraxial')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        coh = la.propagate_traced_carrier_chain_multi(
            specs, groups, _WL, _DX, **kw)
        inc = la.propagate_traced_carrier_chain_multi(
            specs, groups, _WL, _DX, recombine='incoherent', **kw)
    return dict(coh=coh, inc=inc, period=period, A_t=A_t, fd=fd)


def _fringe_period(field, dx):
    """Dominant fringe period (m) of the central row, from a zero-padded
    rFFT restricted to the FRINGE band (periods below 60 um) so the smooth
    image envelope cannot win the argmax."""
    row = np.abs(np.asarray(field)[np.shape(field)[0] // 2]) ** 2
    n = row.size
    ax = (np.arange(n) - n / 2) * dx
    m = np.abs(ax) < 150e-6
    sub = row[m] * np.hanning(int(m.sum()))
    npad = 16 * sub.size
    sp = np.abs(np.fft.rfft(sub, n=npad))
    f = np.fft.rfftfreq(npad, d=dx)
    lo = int(np.searchsorted(f, 1.0 / 60e-6))
    k = lo + int(np.argmax(sp[lo:]))
    return 1.0 / f[k], float(sp[k]), float(sp[1:lo].max())


def test_coherent_recombination_makes_the_analytic_fringe(_conjugate):
    """THE interference acceptance.  Two congruences that overlap at the image
    conjugate must produce the two-beam fringe of the analytically-predicted
    period -- not an incoherent power sum."""
    got, amp, env_amp = _fringe_period(_conjugate['coh'].field, _DXO_I)
    assert got == pytest.approx(_conjugate['period'], rel=0.02), (
        got, _conjugate['period'])
    assert amp > 0.2 * env_amp                    # the fringe is not a ripple
    # full contrast: two equal beams -> the nulls go to zero
    I = np.abs(np.asarray(_conjugate['coh'].field)) ** 2
    row = I[_NOUT_I // 2]
    # +-20 um: the smooth image envelope (216 um radius) varies < 2% there, so
    # anything deeper is the fringe
    core = row[np.abs((np.arange(_NOUT_I) - _NOUT_I / 2) * _DXO_I) < 20e-6]
    assert core.min() < 0.02 * core.max()


def test_incoherent_recombination_has_no_fringe(_conjugate):
    """The control.  ``recombine='incoherent'`` sums |A|^2, so the same two
    congruences give a smooth lobe -- and the fringe-band amplitude collapses
    by orders of magnitude."""
    _p_c, amp_c, _e = _fringe_period(_conjugate['coh'].field, _DXO_I)
    _p_i, amp_i, env_i = _fringe_period(_conjugate['inc'].field, _DXO_I)
    # measured margins: amp_i / amp_c = 1.7e-3, amp_i / env_i = 1.5e-3
    assert amp_i < 1e-2 * amp_c, (amp_i, amp_c)
    assert amp_i < 1e-2 * env_i, (amp_i, env_i)
    I = np.abs(np.asarray(_conjugate['inc'].field)) ** 2
    row = I[_NOUT_I // 2]
    core = row[np.abs((np.arange(_NOUT_I) - _NOUT_I / 2) * _DXO_I) < 20e-6]
    assert core.min() > 0.97 * core.max()      # the Gaussian envelope alone


def test_coherent_and_incoherent_carry_the_same_total_power(_conjugate):
    """Interference redistributes power, it does not create or destroy it."""
    pc = float((np.abs(np.asarray(_conjugate['coh'].field)) ** 2).sum())
    pi = float((np.abs(np.asarray(_conjugate['inc'].field)) ** 2).sum())
    assert pc == pytest.approx(pi, rel=2e-3)
    # ... but the PEAK differs by exactly the two-beam factor 2
    peak_c = float((np.abs(np.asarray(_conjugate['coh'].field)) ** 2).max())
    peak_i = float((np.abs(np.asarray(_conjugate['inc'].field)) ** 2).max())
    assert peak_c / peak_i == pytest.approx(2.0, rel=0.02)
    # the incoherent result is a real, non-negative amplitude by contract
    assert np.allclose(np.asarray(_conjugate['inc'].field).imag, 0.0)


# ===========================================================================
# 3b. the OTHER face -- a per-EMITTER array (pure decentre, no tilt)
# ===========================================================================

def test_per_emitter_array_images_onto_the_exact_ray_trace():
    """The second face of the same feature.  An emitter array is K congruences
    that differ only by a transverse DECENTRE
    (``TiltedCarrier(R, 0, 0, x0, y0)``) -- the case
    ``apply_real_lens_traced_multi`` exists for but cannot run in the
    validated configuration (it fixes ``preserve_input_phase=True`` and
    rejects ``amplitude_model='ray_density'`` on its default
    ``reuse_prepared=True``).

    Run at the input-plane CONJUGATE, where the array images at magnification
    ``A_t = 1 / (C d0 + D)``.  A CROSS layout keeps each emitter meridional so
    an EXACT meridional ray trace through the real spherical surfaces is a
    valid oracle for each one."""
    gA, gB, groups = _relay_groups(ap=14e-3, d0=_D0_CONJ)
    M = _total_abcd(gA, gB, _D0_CONJ)
    fd = -M[0, 1] / M[1, 1]
    A_t = M[0, 0] + fd * M[1, 0]
    s = 2.0e-3                                   # emitter offset at the TX
    env = _gauss()
    layout = [(-1, 0), (+1, 0), (0, -1), (0, +1)]
    specs = [{'field': env, 'name': f'e{i}',
              'carrier': la.TiltedCarrier(np.inf, 0.0, 0.0, sx * s, sy * s)}
             for i, (sx, sy) in enumerate(layout)]
    dxo, nout, tile = 1.0e-6, 2048, 1024
    # same window-vs-leg relation as the conjugate fixture above: a 1.024 mm
    # tile needs a period at least that wide, so ask for the leg that gives it
    _need = 1.3 * (tile * dxo) * fd / (_N * _DX)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = la.propagate_traced_carrier_chain_multi(
            specs, groups, _WL, _DX,
            output_grid=dict(dx_out=dxo, N_out=nout, standoff=_need),
            readout_tile=tile,
            final_distance=fd, ray_subsample=_RS, n_workers=_NW,
            traced_kwargs=_TKW, final_leg='paraxial')

    sa = surfaces_from_prescription(gA)
    sb = surfaces_from_prescription(gB)
    sa[-1] = dataclasses.replace(sa[-1], thickness=_GAP)
    sb[-1] = dataclasses.replace(sb[-1], thickness=fd)
    surfs = sa + sb + [Surface(
        radius=np.inf, conic=0.0, semi_diameter=np.inf, glass_before='air',
        glass_after='air', is_mirror=False, thickness=0.0, label='img')]

    I = np.abs(np.asarray(res.field)) ** 2
    ax = (np.arange(nout) - nout / 2) * dxo
    seen = []
    for info, (sx, sy) in zip(res.congruences, layout):
        axis = 0 if sx else 1
        off = (sx or sy) * s
        got = info['chief_ray'][axis]
        x_exact = float(trace(make_ray(off, 0.0, 0.0, 0.0, wavelength=_WL),
                              surfs, _WL).image_rays.x[0])
        x_abcd = A_t * off
        assert abs(x_abcd) > 0.4e-3           # a real, resolvable decentre
        assert info['chief_ray'][1 - axis] == 0.0
        # the orchestrator's placement lands on the EXACT ray trace ...
        assert got == pytest.approx(x_exact, abs=4.0e-6), (got, x_exact)
        # ... and closer to it than the fully-paraxial ABCD stand-in, i.e.
        # the exact free-leg obliquity is doing real work (measured: 2.4 um
        # vs 4.6 um on this fixture)
        assert abs(got - x_exact) < abs(x_abcd - x_exact)
        # the measured lobe is at the predicted spot
        m = ((np.abs(ax[None, :] - info['chief_ray'][0]) < 300e-6)
             & (np.abs(ax[:, None] - info['chief_ray'][1]) < 300e-6))
        sub = np.where(m, I, 0.0)
        tot = float(sub.sum())
        cx = float((sub.sum(axis=0) * ax).sum() / tot)
        cy = float((sub.sum(axis=1) * ax).sum() / tot)
        assert abs(cx - info['chief_ray'][0]) < 20e-6, (cx, info['chief_ray'])
        assert abs(cy - info['chief_ray'][1]) < 20e-6, (cy, info['chief_ray'])
        assert info['throughput'] > 0.90
        seen.append((round(cx, 8), round(cy, 8)))
    assert len(set(seen)) == 4                # four DISTINCT imaged emitters
    assert np.isfinite(np.asarray(res.field)).all()


# ===========================================================================
# 4.  K is not capped at 32 (the max_segments cap is not inherited)
# ===========================================================================

def test_k_is_not_capped_at_thirty_two():
    """``apply_real_lens_traced_segmented``'s ``max_segments`` default is 32,
    which an 8x4 fan saturates EXACTLY.  The orchestrator must not inherit
    it: K=40 runs (cheaply, on a small grid) and every congruence lands."""
    import inspect
    sig = inspect.signature(la.apply_real_lens_traced_segmented)
    assert sig.parameters['max_segments'].default == 32
    sig2 = inspect.signature(la.propagate_traced_carrier_chain_multi)
    assert not any('segment' in p for p in sig2.parameters)

    n, dx, w = 256, 30e-6, 1.2e-3
    presc = _singlet(80e-3, -80e-3, 3e-3, 'N-BK7', 12e-3, 'g')
    groups = [{'prescription': presc, 'gap_before': 0.0}]
    A, _B, C, _D = _group_abcd(presc, _WL)
    fd = -A / C
    env = _gauss(n, dx, w)
    K = 40
    specs = [{'field': env, 'name': f'o{i}',
              'carrier': la.TiltedCarrier(np.inf, (i - K // 2) * 1e-3, 0.0)}
             for i in range(K)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = la.propagate_traced_carrier_chain_multi(
            specs, groups, _WL, dx, output_grid=dict(dx_out=2e-6, N_out=2048),
            final_distance=fd, ray_subsample=8, n_workers=_NW,
            traced_kwargs=_TKW, final_leg='paraxial', readout_tile=64)
    assert len(res.congruences) == K
    assert np.isfinite(np.asarray(res.field)).all()
    xs = [c['chief_ray'][0] for c in res.congruences]
    assert np.all(np.diff(xs) > 0)                  # 40 distinct frames
    assert all(c['throughput'] > 0.5 for c in res.congruences), [
        round(c['throughput'], 3) for c in res.congruences]
    assert all(c['clipped'] == 0.0 for c in res.congruences)


# ===========================================================================
# 5.  guards -- they must fire BEFORE any propagation work
# ===========================================================================

def _tiny():
    presc = _singlet(80e-3, -80e-3, 3e-3, 'N-BK7', 12e-3, 'g')
    return ([{'prescription': presc, 'gap_before': 0.0}], _gauss(64, 120e-6,
                                                                2e-3),
            dict(dx_out=2e-6, N_out=64))


@pytest.mark.parametrize('kw, match', [
    (dict(recombine='average'), 'recombine'),
    (dict(on_mem_budget='shrug'), 'on_mem_budget'),
    (dict(on_replica='shrug'), 'on_replica'),
    (dict(on_readout_clip='shrug'), 'on_readout_clip'),
    (dict(readout_capture_tol=0.0), 'readout_capture_tol'),
    (dict(readout_capture_tol=np.nan), 'readout_capture_tol'),
    (dict(readout_tile=63), 'readout_tile'),
    (dict(readout_tile=128), 'readout_tile'),
    (dict(readout_tile='biggest'), 'readout_tile'),
    (dict(final_distance=np.inf), 'final_distance'),
])
def test_scalar_option_guards(kw, match):
    groups, env, og = _tiny()
    with pytest.raises(ValueError, match=match):
        la.propagate_traced_carrier_chain_multi(
            [env], groups, _WL, 120e-6, output_grid=og, **kw)


def test_output_grid_guards():
    groups, env, _og = _tiny()
    fn = la.propagate_traced_carrier_chain_multi
    with pytest.raises(ValueError, match='N_out'):
        fn([env], groups, _WL, 120e-6, output_grid=dict(dx_out=2e-6))
    with pytest.raises(ValueError, match='even integer'):
        fn([env], groups, _WL, 120e-6,
           output_grid=dict(dx_out=2e-6, N_out=63))
    with pytest.raises(ValueError, match='dx_out'):
        fn([env], groups, _WL, 120e-6,
           output_grid=dict(dx_out=-1.0, N_out=64))
    with pytest.raises(ValueError, match='unknown key'):
        fn([env], groups, _WL, 120e-6,
           output_grid=dict(dx_out=2e-6, N_out=64, N_ouy=1))
    with pytest.raises(ValueError, match='must be a dict'):
        fn([env], groups, _WL, 120e-6, output_grid=(2e-6, 64))


def test_congruence_guards():
    groups, env, og = _tiny()
    fn = la.propagate_traced_carrier_chain_multi

    def call(cs):
        return fn(cs, groups, _WL, 120e-6, output_grid=og)

    with pytest.raises(ValueError, match='empty'):
        call([])
    # the exact trap this entry point exists to prevent: a mis-spelled
    # carrier key would silently run the congruence ON AXIS
    with pytest.raises(ValueError, match="unknown key"):
        call([{'field': env, 'r_in': 1.0}])
    with pytest.raises(ValueError, match="'field'"):
        call([{'carrier': np.inf}])
    with pytest.raises(ValueError, match='shape'):
        call([env, _gauss(32, 240e-6, 2e-3)])
    with pytest.raises(ValueError, match='finite'):
        call([{'field': env, 'weight': np.nan}])
    # a non-propagating tilt fails before any work
    with pytest.raises(ValueError, match='not a propagating direction'):
        call([{'field': env, 'carrier': la.TiltedCarrier(np.inf, 1.5, 0.0)}])
    with pytest.raises(ValueError, match='groups is empty'):
        fn([env], [], _WL, 120e-6, output_grid=og)


def test_memory_budget_is_honoured():
    """``on_mem_budget='error'`` FAILS LOUDLY rather than reporting a number
    from a degraded unattended run; 'warn' downgrades it; and
    ``LUMENAIRY_MEM_BUDGET_MB`` is honoured when no explicit budget is set."""
    groups, env, _og = _tiny()
    fn = la.propagate_traced_carrier_chain_multi
    og = dict(dx_out=2e-6, N_out=8192)          # 8192^2 * 16 B = 1.07 GB
    with pytest.raises(MemoryError, match='SEQUENTIAL'):
        fn([env], groups, _WL, 120e-6, output_grid=og, mem_budget_mb=64.0)
    with pytest.raises(ValueError, match='mem_budget_mb'):
        fn([env], groups, _WL, 120e-6, output_grid=og, mem_budget_mb=0.0)
    old = os.environ.get('LUMENAIRY_MEM_BUDGET_MB')
    os.environ['LUMENAIRY_MEM_BUDGET_MB'] = '64'
    try:
        with pytest.raises(MemoryError):
            fn([env], groups, _WL, 120e-6, output_grid=og)
    finally:
        if old is None:
            os.environ.pop('LUMENAIRY_MEM_BUDGET_MB', None)
        else:
            os.environ['LUMENAIRY_MEM_BUDGET_MB'] = old
    # 'warn' downgrades to a RuntimeWarning and proceeds (K=1, tiny tile)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        res = fn([env], groups, _WL, 120e-6,
                 output_grid=dict(dx_out=2e-6, N_out=1024),
                 mem_budget_mb=1e-3, on_mem_budget='warn',
                 final_distance=40e-3, ray_subsample=8, n_workers=1,
                 traced_kwargs=_TKW, final_leg='paraxial', readout_tile=64)
    assert any('mem_budget_mb' in str(w.message) for w in rec)
    assert np.isfinite(np.asarray(res.field)).all()


def test_off_grid_congruence_warns_and_reports_the_clip():
    """A congruence whose frame lands outside the common grid must SAY SO --
    the silent-scramble failure mode is exactly what this entry point is
    for."""
    gA, gB, groups = _relay_groups()
    A2, B2, C2, D2 = _group_abcd(gB, _WL)
    R_A = _group_abcd(gA, _WL)[0] / _group_abcd(gA, _WL)[2]
    R_B = (A2 * (R_A + _GAP) + B2) / (C2 * (R_A + _GAP) + D2)
    env = _gauss()
    specs = [{'field': env, 'name': 'far',
              'carrier': la.TiltedCarrier(np.inf, _TILT, 0.0)}]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        res = la.propagate_traced_carrier_chain_multi(
            specs, groups, _WL, _DX, output_grid=dict(dx_out=_DXO, N_out=64),
            final_distance=-R_B, ray_subsample=_RS, n_workers=_NW,
            traced_kwargs=_TKW, final_leg='paraxial', readout_tile=64)
    assert any('outside the common output grid' in str(w.message)
               for w in rec)
    assert res.congruences[0]['clipped'] == pytest.approx(1.0)
    assert float(np.abs(np.asarray(res.field)).max()) == 0.0


def test_tilted_congruence_takes_the_exact_final_leg():
    """The inherited D1 limit is CLOSED by niche D6: a tilted congruence now
    routes onto the EXACT high-NA final leg through this orchestrator, and the
    guard it can still hit (``on_tilt_exact_grid``) is forwarded per
    congruence.  Physics coverage is in ``test_niche_d6_exact_tilted_leg.py``;
    what is pinned here is that the ORCHESTRATOR reaches it and still fills in
    the per-congruence bookkeeping a fan run reads."""
    gA, gB, groups = _relay_groups()
    env = _gauss()
    specs = [{'field': env, 'carrier': la.TiltedCarrier(np.inf, _TILT, 0.0)}]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = la.propagate_traced_carrier_chain_multi(
            specs, groups, _WL, _DX,
            output_grid=dict(dx_out=_DXO, N_out=_NOUT),
            final_distance=20e-3, ray_subsample=_RS, n_workers=_NW,
            traced_kwargs=_TKW, final_leg='exact', readout_tile=_TILE,
            on_tilt_exact_grid='warn')
    c = res.congruences[0]
    assert np.isfinite(np.asarray(res.field)).all()
    assert c['power_exit'] > 0.0 and 0.0 < c['throughput'] <= 1.0 + 1e-9
    assert c['readout_period'] is not None
    assert abs(c['chief_ray'][0]) > 0.0


def test_result_grid_convention_is_the_readout_convention():
    """``field[j, i]`` sits at ``centre + ((i - N/2) dx, (j - N/2) dx)`` --
    the same convention ``carrier_referenced_focus_readout``'s ``centre_out``
    uses, so a consumer can index frames without re-deriving it."""
    presc = _singlet(80e-3, -80e-3, 3e-3, 'N-BK7', 12e-3, 'g')
    groups = [{'prescription': presc, 'gap_before': 0.0}]
    env = _gauss(256, 40e-6, 1.5e-3)
    A, B, C, D = _group_abcd(presc, _WL)
    fd = -A / C                              # the group's own back focus
    cen = (11.0e-6, -5.0e-6)
    tilt = 5.0e-4
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = la.propagate_traced_carrier_chain_multi(
            [{'field': env, 'carrier': la.TiltedCarrier(np.inf, tilt, 0.0)}],
            groups, _WL, 40e-6,
            # This measures the intensity CENTROID, which is wing-sensitive: if
            # the 256 um readout window exceeds one Bluestein period the outer
            # window fills with replicas and the centroid walks (measured 58.8 um
            # at the 0.8 zR accuracy standoff).  At K = 1 the replica guard is
            # deliberately permissive -- it is a MULTIPLEXING guard and warns
            # rather than raising -- so nothing refuses the run for us here.
            # period = N * dx * standoff / z_focus, so ask for the leg that
            # holds the requested window (30% margin), as the wide-window
            # fixtures above do.
            output_grid=dict(dx_out=0.5e-6, N_out=512, centre_out=cen,
                             standoff=1.3 * (512 * 0.5e-6) * fd / (256 * 40e-6)),
            final_distance=fd, ray_subsample=8, n_workers=1,
            traced_kwargs=_TKW, final_leg='paraxial')
    assert res.centre == cen
    I = np.abs(np.asarray(res.field)) ** 2
    ax = (np.arange(512) - 512 / 2) * 0.5e-6
    x = cen[0] + float((I.sum(axis=0) * ax).sum() / I.sum())
    y = cen[1] + float((I.sum(axis=1) * ax).sum() / I.sum())
    xc, yc = res.congruences[0]['chief_ray']
    assert abs(x - xc) < 1.5e-6 and abs(y - yc) < 1.5e-6, (x, y, xc, yc)
    # the offset centre_out is genuinely exercised (not a no-op)
    assert abs(xc) > 10.0 * 0.5e-6


def test_chief_ray_predictor_matches_an_exact_ray_trace():
    """The analytic predictor that places the tiles, checked directly against
    an exact meridional ray trace through the real spherical surfaces (no
    propagation run needed)."""
    gA, gB, groups = _relay_groups()
    A2, B2, C2, D2 = _group_abcd(gB, _WL)
    R_A = _group_abcd(gA, _WL)[0] / _group_abcd(gA, _WL)[2]
    R_B = (A2 * (R_A + _GAP) + B2) / (C2 * (R_A + _GAP) + D2)
    fd = -R_B
    for tilt in (0.0, 5e-3, 0.02, _TILT, 0.046):
        x, y, L, _M = _chain_chief_ray_at_target(
            groups, _WL, la.TiltedCarrier(np.inf, tilt, 0.0), fd, 'probe')
        x_exact = _exact_chief_height(gA, gB, tilt, fd)
        assert y == 0.0
        assert x == pytest.approx(x_exact, abs=0.4e-6), (tilt, x, x_exact)
