"""Every SHIPPED pipeline spec must still satisfy the guards it will meet.

THE DEFECT THIS EXISTS FOR (design-121 campaign audit S9.1 #2, 2026-08-17).
``validation/pipeline/specs/d121_32order.json`` shipped with
``aggregate.dx_common = 1.2292 um`` against a band-aware Nyquist requirement
of 1.108184 um -- a margin of 0.9015, i.e. REFUSED.  The guard was tightened
in ``0f46efb`` (the envelope's own half-band was folded into the binding
pitch); the spec was last touched at ``087d151``, before it.  Nothing
connected the two, so a reference spec that cannot complete on the shipped
library sat in the tree looking authoritative.

Re-measuring found the audit under-counted: ALL FOUR shipped specs carried
that same stranded pitch, not one.

THE CLASS, not the instance.  A spec is stranded when the guard moves under
it, so the test has to re-run the GUARD, not re-read a number:

* ``test_every_shipped_spec_loads`` -- the schema / enum / positivity guards
  in ``validation/pipeline/spec.py`` run on every shipped spec.
* ``test_every_shipped_spec_has_a_nyquist_witness_or_a_stated_exemption`` --
  a spec with no measurement cannot ship silently.
* ``test_the_live_nyquist_guard_admits_every_certified_spec`` -- THE ARM THAT
  CATCHES A TIGHTENING.  It calls ``carrier_difference_nyquist`` itself, on
  the measured carriers, so any future change to that arithmetic (a new
  additive term, a different ``_BAND_HEADROOM``, a different choice of which
  candidate binds) re-prices every spec here rather than at hour seven of a
  run.
* ``test_the_live_containment_arithmetic_admits_every_certified_spec`` -- the
  other guard the same call makes, and the one that decides ``n_common``.

THE WITNESS.  ``specs/_measured_nyquist.json`` carries the guard's INPUTS --
both carriers, each beam's support radius and its envelope half-band --
measured on the exp29 run of record.  Everything but ``env_band`` is read
straight out of that run's own artifacts; ``env_band`` is recovered by
inverting the guard at the pitch that run used, which is exact because it is
the only free quantity in that arithmetic.  All 32 beams reproduce their
recorded ``nyquist_margin`` to better than 1e-12 relative on 5.38.1, and
``test_the_witness_reproduces_the_run_it_was_measured_on`` asserts exactly
that, so the witness cannot rot into fiction.

SCOPE, stated rather than implied.  The witness bounds what this file can
claim.  It does NOT re-measure ``_enclosed_band_radius`` (that needs the 6.7
GB of chain fields), so a change to how the ENVELOPE BAND ITSELF is measured
would pass here.  What it does cover is every other way the binding pitch can
move, which is the way it moved in ``0f46efb``.
"""
import json
import math
import os

import pytest

from lumenairy.propagators.carrier_field import CarrierSpec, carrier_difference_nyquist

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPECS = os.path.normpath(os.path.join(
    _HERE, '..', '..', 'validation', 'pipeline', 'specs'))
_WITNESS = os.path.join(_SPECS, '_measured_nyquist.json')


def _spec_files():
    return sorted(f for f in os.listdir(_SPECS)
                  if f.endswith('.json') and not f.startswith('_'))


def _load_raw(name):
    with open(os.path.join(_SPECS, name)) as fh:
        return json.load(fh)


@pytest.fixture(scope='module')
def witness():
    with open(_WITNESS) as fh:
        return json.load(fh)


#: Specs whose chains are NOT the ones the witness was measured on, with the
#: reason.  An entry here is a spec this file CANNOT certify -- that is a
#: finding, not a waiver, and the tests below keep it visible.
UNCERTIFIED = {
    'd121_3order_probe.json': (
        "chain.kind='cached_aperture' at ray_subsample=4: it replays an "
        "ARCHIVED arm-A field (_sumap_ap_*.npy) rather than running the "
        "traced chain, so its beams' support radii and envelope bands are "
        "the archive's and not the ones the witness measured at rs=1.  The "
        "archive is not in the tree (it is ~1 GB per order and was never "
        "committed), so the guard cannot be evaluated for this spec here.  "
        "Its dx_common is therefore left at the probe's configuration of "
        "record and is NOT claimed to satisfy the shipped guard."),
}


def _spec_chain_signature(raw):
    """The chain fields that determine a beam's exit carrier, support radius
    and envelope band -- i.e. everything the Nyquist guard reads.  Two specs
    with the same signature share a witness; two that differ must not."""
    c = raw['chain']
    return (c.get('kind'), c.get('plane'), c.get('ray_subsample'),
            c.get('n_fine_cap'), c.get('window_factor'))


def _witness_signature(w):
    c = w['chain_signature']
    return (c['kind'], c['plane'], c['ray_subsample'], c['n_fine_cap'],
            c['window_factor'])


def _beams_for(raw, witness):
    """The witness rows this spec's aggregate will actually include."""
    orders = raw['decompose'].get('params', {}).get('orders')
    if orders in (None, 'all'):
        return dict(witness['beams'])
    keys = [witness['order_key_for'][f'({int(m)},{int(n)})']
            for m, n in orders]
    return {k: witness['beams'][k] for k in keys}


def _carriers(witness, row):
    src = CarrierSpec(R=row['R_out'], centre=tuple(row['chief_exit']),
                      tilt=tuple(row['tilt_exit']), piston=0.0)
    c = witness['common_carrier']
    dst = CarrierSpec(R=c['R'], centre=tuple(c['centre']),
                      tilt=tuple(c['tilt']), piston=c['piston'])
    return src, dst


# ---------------------------------------------------------------------------
def test_every_shipped_spec_loads():
    """The schema guards run on every shipped spec, so a spec cannot ship
    with an unknown key, a bad enum or a non-positive pitch."""
    from validation.pipeline.spec import PipelineSpec
    names = _spec_files()
    assert names, f'no specs found under {_SPECS}'
    for name in names:
        PipelineSpec.from_dict(_load_raw(name))


def test_every_shipped_spec_has_a_nyquist_witness_or_a_stated_exemption(
        witness):
    """A new spec cannot ship UNMEASURED.  Either the witness covers its
    chain configuration, or it is named in UNCERTIFIED with a reason."""
    wsig = _witness_signature(witness)
    for name in _spec_files():
        raw = _load_raw(name)
        sig = _spec_chain_signature(raw)
        if sig == wsig:
            assert name not in UNCERTIFIED, (
                f'{name} matches the witness chain signature {sig} but is '
                f'listed as UNCERTIFIED; remove the exemption')
        else:
            assert name in UNCERTIFIED, (
                f'{name} has chain signature {sig}, which the witness '
                f'({wsig}) was NOT measured on, and it is not listed in '
                f'UNCERTIFIED.  Either measure it or state why it cannot be '
                f'-- a spec whose Nyquist requirement nobody has measured is '
                f'exactly what stranded d121_32order.')


def test_the_uncertified_registry_does_not_rot():
    """Every exemption names a spec that still exists, with a reason."""
    names = set(_spec_files())
    for name, why in UNCERTIFIED.items():
        assert name in names, (
            f'UNCERTIFIED names {name}, which is no longer shipped; drop the '
            f'entry')
        assert len(why) > 80, f'{name}: the exemption needs a real reason'


def test_the_witness_reproduces_the_run_it_was_measured_on(witness):
    """The witness is only worth anything if the LIVE guard, fed it, returns
    the run of record's own numbers.  Re-measured here rather than trusted:
    this is what stops the file rotting into fiction after a library change.

    The bar is 1e-12 RELATIVE on the margin.  It is not a tolerance on
    physics -- the guard is closed-form arithmetic on doubles, so agreement
    is at round-off -- and the failure it must catch (a changed term, a
    changed headroom, a different binding candidate) moves the margin by
    percent, ten orders of magnitude above it.  Measured 2026-08-17 on
    lumenairy 5.38.1: all 32 beams reproduce to better than 1e-12.
    """
    lam = witness['wavelength']
    dx_used = witness['provenance']['dx_common_used_by_that_run']
    worst = math.inf
    for key, row in witness['beams'].items():
        src, dst = _carriers(witness, row)
        rep = carrier_difference_nyquist(
            src, dst, lam, row['support_radius'], dx_target=dx_used,
            env_band=row['env_band'])
        assert rep.binding_term == row['measured_binding_term'], key
        got = rep.dx_binding
        want = row['measured_dx_binding']
        assert abs(got / want - 1.0) < 1e-12, (
            f'{key}: the live guard now binds at {got * 1e6:.6f} um where '
            f'the run of record measured {want * 1e6:.6f} um.  If the guard '
            f'was INTENTIONALLY changed, re-measure the witness against a '
            f'fresh run and date it; do not relax this bar.')
        worst = min(worst, got)
    assert math.isclose(worst,
                        witness['provenance']['worst_nyquist_margin_recorded']
                        * dx_used, rel_tol=1e-12)


@pytest.mark.parametrize('name', [n for n in _spec_files()
                                  if n not in UNCERTIFIED])
def test_the_live_nyquist_guard_admits_every_certified_spec(name, witness):
    """THE CLASS FIX.  Re-evaluate the guard, live, at each spec's own
    dx_common, and require the margin the spec itself asks for.

    A future tightening of ``carrier_difference_nyquist`` fails HERE, in
    seconds, instead of at the aggregate stage of a seven-hour run."""
    raw = _load_raw(name)
    ags = raw['aggregate']
    lam = raw['wavelength']
    dx = float(ags['dx_common'])
    want_margin = float(ags['nyquist_margin'])
    worst_key, worst_margin = None, math.inf
    for key, row in _beams_for(raw, witness).items():
        src, dst = _carriers(witness, row)
        rep = carrier_difference_nyquist(
            src, dst, lam, row['support_radius'], dx_target=dx,
            env_band=row['env_band'])
        if rep.margin < worst_margin:
            worst_key, worst_margin = key, rep.margin
    assert worst_margin >= want_margin, (
        f"{name}: aggregate.dx_common = {dx * 1e6:.4f} um scores a Nyquist "
        f"margin of {worst_margin:.4f} on beam {worst_key!r}, below the "
        f"{want_margin:.4f} the spec asks for -- on_nyquist="
        f"{ags.get('on_nyquist')!r} means this spec CANNOT COMPLETE.  The "
        f"binding pitch over its beams is "
        f"{worst_margin * dx * 1e6:.6f} um.")


@pytest.mark.parametrize('name', [n for n in _spec_files()
                                  if n not in UNCERTIFIED])
def test_the_live_containment_arithmetic_admits_every_certified_spec(
        name, witness):
    """The other guard ``re_reference`` makes on the same call, reproduced
    from its source: ``reach = |chief - origin| + support`` must fit the
    window's half extent.  This is what decides ``n_common``, and lowering
    dx_common without raising n_common trades a Nyquist refusal for a
    containment one."""
    raw = _load_raw(name)
    ags = raw['aggregate']
    dx, n = float(ags['dx_common']), int(ags['n_common'])
    ox, oy = (float(v) for v in ags['origin'])
    half = 0.5 * n * dx
    worst_key, worst = None, math.inf
    for key, row in _beams_for(raw, witness).items():
        cx, cy = row['chief_exit']
        reach = max(abs(cx - ox), abs(cy - oy)) + row['support_radius']
        if half - reach < worst:
            worst_key, worst = key, half - reach
    assert worst >= 0.0, (
        f"{name}: n_common x dx_common = {n} x {dx * 1e6:.4f} um is a "
        f"{n * dx * 1e3:.4f} mm window (half extent {half * 1e3:.4f} mm), "
        f"but beam {worst_key!r} reaches {(half - worst) * 1e3:.4f} mm -- "
        f"short by {-worst * 1e3:.4f} mm.  The band-limited resample is "
        f"PERIODIC, so the overhang wraps rather than vanishing.")


def test_no_shipped_spec_still_carries_the_stranded_pitch():
    """The instance, pinned so it cannot come back by copy-paste.  1.2292 um
    was the pre-0f46efb pitch and it is refused by the shipped guard on every
    design-121 chain measured; it appeared in all four shipped specs."""
    for name in _spec_files():
        if name in UNCERTIFIED:
            continue
        dx = float(_load_raw(name)['aggregate']['dx_common'])
        assert abs(dx - 1.2292e-06) > 1e-12, (
            f'{name} still carries dx_common = 1.2292 um, the pitch the '
            f'band-aware guard refuses (margin 0.9015).')
