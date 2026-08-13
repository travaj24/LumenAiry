"""Unit tests for the staged field-aggregation pipeline
(``validation/pipeline``).

WHAT IS AND IS NOT TESTED HERE.  This file tests the PLUMBING -- config
validation, the artifact key discipline, checkpoint round trips, resume, and
the one claim that makes the whole staging trustworthy: a single-beam run
through the pipeline is BIT-IDENTICAL to calling the primitives directly.  It
does not test physics: the physics lives in
``lumenairy.propagators.carrier_field`` and is pinned by
``tests/unit/test_carrier_field.py``, and the design-121 reproduction is
scored by ``validation/repro_traced_carrier_121/pipeline_accept_121.py``
against PROBE_SUM_AT_APERTURE's printed numbers.

The fixtures are SYNTHETIC and small (192-256 square) so this file runs in
seconds on a fresh checkout with no Zemax file, no design-121 caches and no
optional dependency.  The pipeline's own plug-in registries are what make that
possible, which is itself part of the design being tested.

No xfail, no skip.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VALIDATION = str(_REPO_ROOT / 'validation')
if _VALIDATION not in sys.path:
    sys.path.insert(0, _VALIDATION)

import pipeline as P  # noqa: E402
from pipeline import artifacts as PA  # noqa: E402
from pipeline import driver as PD  # noqa: E402
from pipeline import sources as PS  # noqa: E402
from pipeline.spec import SpecError  # noqa: E402

from lumenairy.propagators.carrier import carrier_referenced_exact_focus_readout  # noqa: E402
from lumenairy.propagators.carrier_field import (  # noqa: E402
    CarrierField,
    CarrierSpec,
    FieldGrid,
    re_reference,
)

LAM = 1.31e-6
N_AP = 256
DX_AP = 2.0e-6
R_AP = -1.0e-3
W_AP = 6.0e-5
N_COM = 192
DX_COM = 2.0e-6
Z_LEG = 1.0e-3
DX_OUT = 2.0e-7
N_OUT = 128

#: How many times the synthetic chain runner has actually run, per beam key.
#: The resume tests assert on THIS rather than on a wall-clock, because
#: "resumed" has to mean "did not recompute", not "was fast".
_CHAIN_CALLS: dict = {}


# ---------------------------------------------------------------------------
# synthetic plug-ins
# ---------------------------------------------------------------------------
def _synth_full_field(centre, origin, tilt=(0.0, 0.0)):
    grid = FieldGrid((N_AP, N_AP), DX_AP, origin=tuple(origin))
    car = CarrierSpec(R=R_AP, centre=tuple(centre), tilt=tuple(tilt),
                      piston=0.0)
    x, y = grid.axes()
    u = x[None, :] - car.centre[0]
    v = y[:, None] - car.centre[1]
    env = np.exp(-(u * u + v * v) / (W_AP * W_AP)).astype(np.complex128)
    return env * car.phasor_on(grid, LAM, sign=+1), grid, car


def _synth_decompose(params, spec):
    beams = []
    for i, b in enumerate(params['beams']):
        beams.append(PS.Beam(
            key=b['key'], index=i,
            weight=complex(b['weight'][0], b['weight'][1]),
            frame_centre=tuple(b['frame_centre']), label=b['key'],
            payload={'centre': list(b['centre']), 'origin': list(b['origin']),
                     'tilt': list(b.get('tilt', [0.0, 0.0]))}))
    return PS.DecomposeResult(beams=beams, context={'synthetic': True})


def _synth_chain(beam, context, spec):
    _CHAIN_CALLS[beam.key] = _CHAIN_CALLS.get(beam.key, 0) + 1
    E, grid, car = _synth_full_field(beam.payload['centre'],
                                     beam.payload['origin'],
                                     beam.payload['tilt'])
    meta = PS._aperture_meta(E, DX_AP, grid.origin, R_AP, car.centre, car.tilt,
                             power_exit=None)
    meta.update(runner='synthetic', order=None, ray_subsample=1,
                n_fine_cap=N_AP, window_factor=4.0, t_retrace=0.0,
                t_readout=0.0, readout_skipped=True, wall=0.0, warnings=[],
                na_exit=None, na_exit_measured=None)
    fld = PS.wrap_aperture_field(E, meta, LAM, {'beam': beam.key})
    return PS.ChainOutput(field=fld, meta=meta, reference_tile=None)


PS.register_decomposer('test_synth', _synth_decompose)
PS.register_chain_runner('test_synth', _synth_chain)


def _beam_cfg(key, centre, frame=None, weight=(1.0, 0.0)):
    # The frame centre is where a beam referenced to a sphere of radius R
    # about ``centre`` actually images: the sphere's own centre of curvature,
    # i.e. ``centre`` again.  Spelling it that way keeps the synthetic fixture
    # physically consistent, which is what makes the crosstalk test mean
    # something.
    frame = centre if frame is None else frame
    return {'key': key, 'centre': list(centre), 'origin': list(centre),
            'frame_centre': list(frame), 'weight': list(weight)}


def _spec_dict(workdir, beams, **over):
    d = {
        'schema': 1, 'name': 'synth', 'workdir': str(workdir),
        'wavelength': LAM, 'coherent_sum': False,
        'decompose': {'kind': 'test_synth', 'params': {'beams': beams}},
        'chain': {'kind': 'test_synth', 'plane': 'fine_retrace_exit',
                  'ray_subsample': 1, 'n_workers': 1, 'n_fine_cap': N_AP,
                  'window_factor': 4.0, 'final_leg': 'auto',
                  'ram_budget': 'inf', 'on_box_budget': 'warn',
                  'capture_reference_tile': False, 'params': {}},
        'aggregate': {'dx_common': DX_COM, 'n_common': N_COM,
                      'origin': [0.0, 0.0], 'reference_beam': None,
                      'include': 'all', 'weights': 'decompose',
                      'batch_size': 8, 'support_frac': 0.99999,
                      'nyquist_margin': 1.0, 'on_nyquist': 'error',
                      'on_window': 'warn', 'bandlimit': False},
        'leg': {'mode': 'full', 'distance': Z_LEG, 'n_fine_cap': N_AP,
                'window_factor': 4.0, 'ram_budget': 'inf',
                'on_replica': 'warn', 'on_readout_window': 'warn',
                'on_ram_cap': 'error', 'on_n_fine_cap': 'error'},
        'readout': {'dx_out': DX_OUT, 'n_out': N_OUT, 'frames': 'all',
                    'save_frames': True, 'metrics': True,
                    'ee_radii': [5e-6, 1e-5]},
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = {**d[k], **v}
        else:
            d[k] = v
    return d


def _run(workdir, beams, quiet=True, **over):
    spec = P.PipelineSpec.from_dict(_spec_dict(workdir, beams, **over))
    return PD.run(spec, log=(lambda *a, **k: None) if quiet else print)


# ===========================================================================
# (a) CONFIG VALIDATION
# ===========================================================================
def test_spec_round_trip_is_exact(tmp_path):
    spec = P.PipelineSpec.from_dict(
        _spec_dict(tmp_path, [_beam_cfg('a', (0.0, 0.0), (0.0, 0.0))]))
    again = P.PipelineSpec.from_json(spec.to_json())
    assert again == spec, "a spec must be a fixed point of to_json/from_json"
    # and the floats survive bit-exactly, which is what makes the stored spec
    # beside an artifact THE spec rather than a rendering of it
    assert again.aggregate.dx_common == DX_COM
    assert again.wavelength == LAM


def test_spec_refuses_an_unknown_key(tmp_path):
    d = _spec_dict(tmp_path, [_beam_cfg('a', (0.0, 0.0), (0.0, 0.0))])
    d['aggregate']['nyquist_margins'] = 2.0          # plausible typo
    with pytest.raises(SpecError) as e:
        P.PipelineSpec.from_dict(d)
    assert 'nyquist_margins' in str(e.value)
    # the whole point: the typo would otherwise have run at the DEFAULT 1.0
    # while its author believed it ran at 2.0, and nothing downstream could
    # tell the two runs apart
    assert 'unknown key' in str(e.value)


def test_spec_refuses_an_unknown_top_level_key(tmp_path):
    d = _spec_dict(tmp_path, [_beam_cfg('a', (0.0, 0.0), (0.0, 0.0))])
    d['coherent_summ'] = True
    with pytest.raises(SpecError, match='unknown top-level key'):
        P.PipelineSpec.from_dict(d)


def test_spec_refuses_a_missing_section(tmp_path):
    d = _spec_dict(tmp_path, [_beam_cfg('a', (0.0, 0.0), (0.0, 0.0))])
    del d['leg']
    with pytest.raises(SpecError, match='missing required section'):
        P.PipelineSpec.from_dict(d)


def test_spec_refuses_a_schema_mismatch(tmp_path):
    d = _spec_dict(tmp_path, [_beam_cfg('a', (0.0, 0.0), (0.0, 0.0))])
    d['schema'] = 99
    with pytest.raises(SpecError, match='schema'):
        P.PipelineSpec.from_dict(d)


@pytest.mark.parametrize('section,key,bad', [
    ('aggregate', 'on_nyquist', 'raise'),
    ('aggregate', 'weights', 'dammann'),
    ('leg', 'mode', 'cropped'),
    ('chain', 'plane', 'coarse_exit'),
    ('chain', 'on_box_budget', 'ignore'),
])
def test_spec_refuses_an_unknown_enum(tmp_path, section, key, bad):
    d = _spec_dict(tmp_path, [_beam_cfg('a', (0.0, 0.0), (0.0, 0.0))])
    d[section][key] = bad
    with pytest.raises(SpecError, match=key):
        P.PipelineSpec.from_dict(d)


def test_spec_refuses_a_nyquist_margin_below_one(tmp_path):
    """A margin under 1.0 is not a loosened tolerance -- it is a request to
    accept a grid the sampling theorem says cannot carry the answer.  The
    refusal names the honest alternative (a recorded disposition)."""
    d = _spec_dict(tmp_path, [_beam_cfg('a', (0.0, 0.0), (0.0, 0.0))])
    d['aggregate']['nyquist_margin'] = 0.5
    with pytest.raises(SpecError) as e:
        P.PipelineSpec.from_dict(d)
    assert 'BELOW bare Nyquist' in str(e.value)
    assert "on_nyquist='warn'" in str(e.value)


def test_spec_refuses_a_support_fraction_written_as_a_percentage(tmp_path):
    d = _spec_dict(tmp_path, [_beam_cfg('a', (0.0, 0.0), (0.0, 0.0))])
    d['aggregate']['support_frac'] = 99.999
    with pytest.raises(SpecError, match='ENCLOSED-POWER fraction'):
        P.PipelineSpec.from_dict(d)


def test_beam_key_must_be_filesystem_safe():
    """Beam keys name files, and a key with a separator in it writes outside
    the workdir."""
    with pytest.raises(SpecError, match='names files'):
        PS.Beam(key='../escape', index=0, weight=1.0, frame_centre=(0.0, 0.0))
    with pytest.raises(SpecError):
        PS.Beam(key='', index=0, weight=1.0, frame_centre=(0.0, 0.0))


def test_resolve_selection_refuses_an_unknown_beam_and_keeps_order():
    avail = ['a', 'b', 'c']
    assert P.resolve_selection('all', avail, 'x') == avail
    # ORDER comes from the beam list, not from the spelling of the subset --
    # so the accumulation order of a sum is a property of the decomposition
    assert P.resolve_selection(['c', 'a'], avail, 'x') == ['a', 'c']
    with pytest.raises(SpecError, match="'d'"):
        P.resolve_selection(['a', 'd'], avail, 'x')


def test_unknown_registry_name_names_what_is_registered():
    with pytest.raises(SpecError) as e:
        PS.get_decomposer('no_such_thing')
    assert 'test_synth' in str(e.value)
    with pytest.raises(SpecError) as e:
        PS.get_chain_runner('no_such_thing')
    assert 'traced' in str(e.value)


# ===========================================================================
# (b) THE KEY DISCIPLINE
# ===========================================================================
def test_stage_key_is_deterministic_and_covers_the_library(tmp_path,
                                                           monkeypatch):
    spec = P.PipelineSpec.from_dict(
        _spec_dict(tmp_path, [_beam_cfg('a', (0.0, 0.0), (0.0, 0.0))]))
    _k1, d1 = PA.stage_key(spec, 'chains', upstream='u0')
    _k2, d2 = PA.stage_key(spec, 'chains', upstream='u0')
    assert d1 == d2
    # A LIBRARY EDIT ORPHANS THE ARTIFACT.  This is the defect-D6 lesson and
    # the PROBE_SUM_AT_APERTURE S1.2 hazard in one assertion: an edit under
    # lumenairy/** moved a chain's absolute output phase by 2.88 rad while
    # every intensity metric held to 0.001 EE points, and the cache key was
    # the only thing that noticed.
    monkeypatch.setattr(PA, '_LIB_SHA', 'deadbeef' * 8)
    _k3, d3 = PA.stage_key(spec, 'chains', upstream='u0')
    assert d3 != d1
    # ...and so does an edit to this driver, which is as much an input to its
    # artifacts as the library is.
    monkeypatch.setattr(PA, '_LIB_SHA', None)
    monkeypatch.setattr(PA, '_PKG_SHA', 'cafe' * 16)
    _k4, d4 = PA.stage_key(spec, 'chains', upstream='u0')
    assert d4 != d1


def test_stage_key_isolates_stages_from_irrelevant_spec_changes(tmp_path):
    """Keying every stage on the WHOLE spec would orphan 32 aperture fields --
    each a multi-minute chain -- every time a frame window changed, which is
    how a caching layer trains its users to delete it."""
    base = _spec_dict(tmp_path, [_beam_cfg('a', (0.0, 0.0), (0.0, 0.0))])
    s0 = P.PipelineSpec.from_dict(base)
    moved = dict(base)
    moved['readout'] = {**base['readout'], 'n_out': 256}
    s1 = P.PipelineSpec.from_dict(moved)
    assert PA.stage_key(s0, 'chains')[1] == PA.stage_key(s1, 'chains')[1]
    assert PA.stage_key(s0, 'readout')[1] != PA.stage_key(s1, 'readout')[1]
    # and a change UPSTREAM orphans everything downstream through the digest
    a = PA.stage_key(s0, 'aggregate', upstream='chains-A')[1]
    b = PA.stage_key(s0, 'aggregate', upstream='chains-B')[1]
    assert a != b


def test_json_artifact_refuses_a_foreign_key(tmp_path):
    key_a, _ = PA.stage_key(
        P.PipelineSpec.from_dict(
            _spec_dict(tmp_path, [_beam_cfg('a', (0, 0), (0, 0))])),
        'decompose')
    p = str(tmp_path / 'x.json')
    PA.write_json(p, {'v': 1}, key_a)
    assert PA.read_json(p, key_a) == {'v': 1}
    assert PA.read_json(p, {**key_a, 'stage': 'chains'}) is None, (
        "a stored artifact whose key does not match must read as MISSING, "
        "not as a near-enough hit")


def test_json_artifact_write_is_atomic(tmp_path):
    """An interrupted run must leave the previous artifact or none -- never a
    half-written one whose key was flushed first."""
    p = str(tmp_path / 'y.json')
    PA.write_json(p, {'v': 1}, {'k': 1})
    assert not os.path.exists(p + '.tmp')
    PA.write_json(p, {'v': 2}, {'k': 1})
    assert PA.read_json(p, {'k': 1}) == {'v': 2}


def test_checkpoint_field_round_trip_and_key_check(tmp_path):
    """A field checkpoint round-trips its ENVELOPE bit-for-bit and its
    absolute geometry exactly, and refuses a foreign key.

    The geometry half matters as much as the array: a mis-read grid ORIGIN or
    carrier CENTRE relocates a field in absolute coordinates without changing
    a single sample of it, which no intensity metric can see."""
    E, grid, car = _synth_full_field((3e-5, -1e-5), (3e-5, -1e-5))
    f = CarrierField.from_full_field(E, grid, car, LAM,
                                     provenance={'beam': 'a'})
    p = str(tmp_path / 'chk.zarr')
    PA.save_field(p, f, {'k': 'v'}, extra_provenance={'stage': 'chains'})
    assert PA.field_exists(p)
    back = PA.load_field(p, {'k': 'v'})
    assert back is not None
    assert back.envelope.tobytes() == f.envelope.tobytes()
    assert back.grid == f.grid and back.carrier == f.carrier
    assert back.wavelength == f.wavelength
    assert back.provenance['stage'] == 'chains'
    assert PA.load_field(p, {'k': 'other'}) is None


def test_selection_tag_is_readable_while_short_and_hashed_when_long():
    assert PA.selection_tag(['p0_p0', 'm2_p0']) == 'sum_p0_p0_m2_p0'
    long = PA.selection_tag([f'm{i}_p0' for i in range(32)])
    assert long.startswith('sum_32beams_') and len(long) < 40
    # deterministic, and different sets get different tags
    assert long == PA.selection_tag([f'm{i}_p0' for i in range(32)])
    assert long != PA.selection_tag([f'm{i}_p1' for i in range(32)])


# ===========================================================================
# (c) THE ADMISSIBILITY BANNER
# ===========================================================================
def test_admissibility_warns_loudly_without_the_piston_fix(tmp_path,
                                                           monkeypatch):
    spec = P.PipelineSpec.from_dict(
        _spec_dict(tmp_path, [_beam_cfg('a', (0, 0), (0, 0))],
                   coherent_sum=True))
    monkeypatch.setattr(PD, 'piston_fix_status',
                        lambda: ('absent', '/x/_lens_traced.py'))
    with pytest.warns(RuntimeWarning) as rec:
        out = PD.admissibility_banner(spec, log=lambda *a: None)
    msg = str(rec[0].message)
    assert 'FIX_TILT_QUADRATIC_OPL' in msg
    assert '0.050-0.416 waves' in msg
    assert out['piston_columns_valid'] is False


def test_admissibility_is_silent_with_the_piston_fix(tmp_path, monkeypatch):
    spec = P.PipelineSpec.from_dict(
        _spec_dict(tmp_path, [_beam_cfg('a', (0, 0), (0, 0))],
                   coherent_sum=True))
    monkeypatch.setattr(PD, 'piston_fix_status',
                        lambda: ('present', '/x/_lens_traced.py'))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = PD.admissibility_banner(spec, log=lambda *a: None)
    assert out['piston_columns_valid'] is True


def test_piston_fix_status_reads_the_module_that_is_imported():
    """The probe is blunt by design (the fix adds no public symbol), so pin
    what it actually inspects: the source file of the imported module."""
    status, detail = PD.piston_fix_status()
    assert status in ('present', 'absent', 'unknown')
    if status != 'unknown':
        assert detail.endswith('_lens_traced.py') and os.path.exists(detail)


# ===========================================================================
# (d) END TO END, RESUME, AND THE NULL CONTROL
# ===========================================================================
def test_pipeline_runs_end_to_end_and_writes_every_checkpoint(tmp_path):
    beams = [_beam_cfg('a', (0.0, 0.0), (0.0, 0.0)),
             _beam_cfg('b', (3e-5, 0.0))]
    st = _run(tmp_path / 'wd', beams)
    wd = st.workdir
    assert os.path.exists(os.path.join(wd, 'decompose', 'beams.json'))
    for k in ('a', 'b'):
        assert PA.field_exists(os.path.join(wd, 'chains', f'{k}.zarr'))
        assert os.path.exists(os.path.join(wd, 'chains', f'{k}.json'))
    tag = st.cache['agg_tag']
    assert PA.field_exists(os.path.join(wd, 'aggregate', tag, 'summed.zarr'))
    assert os.path.exists(os.path.join(wd, 'aggregate', tag, 'ledger.json'))
    assert os.path.exists(os.path.join(wd, 'aggregate', tag, 'leg_full.json'))
    ro = os.path.join(wd, 'readout', f'{tag}__full')
    assert os.path.exists(os.path.join(ro, 'metrics.json'))
    assert os.path.exists(os.path.join(ro, 'frames.npz'))
    # the manifest records every stage under its own digest
    with open(os.path.join(wd, '_manifest.json'), encoding='cp1252') as fh:
        man = json.load(fh)
    assert set(man['stages']) == set(P.STAGES)
    assert man['run']['peak_wset_gb'] > 0.0
    # the energy ledger is per beam and accounts for the window
    led = st.report['aggregate']
    assert len(led['rows']) == 2
    for r in led['rows']:
        assert r['power_source'] > 0.0
        assert abs(r['frac_out_of_window']) < 1e-6
        assert r['nyquist_margin'] > 1.0
        assert r['binding_term'] in ('ramp', 'reconstruct')


def test_resume_skips_stages_whose_key_still_matches(tmp_path):
    beams = [_beam_cfg('a', (0.0, 0.0), (0.0, 0.0))]
    _CHAIN_CALLS.clear()
    _run(tmp_path / 'wd', beams)
    assert _CHAIN_CALLS['a'] == 1
    st = _run(tmp_path / 'wd', beams)
    assert _CHAIN_CALLS['a'] == 1, ("the second run must RESUME the chain "
                                    "checkpoint, not recompute it")
    assert st.report['chains']['resumed'] == 1
    assert st.report['chains']['ran'] == 0
    assert st.report['aggregate']['resumed'] is True
    assert st.report['readout']['resumed'] is True


def test_resume_from_a_stage_recomputes_it_and_everything_after(tmp_path):
    beams = [_beam_cfg('a', (0.0, 0.0), (0.0, 0.0))]
    _CHAIN_CALLS.clear()
    _run(tmp_path / 'wd', beams)
    st = _run(tmp_path / 'wd', beams, quiet=True)
    assert _CHAIN_CALLS['a'] == 1
    spec = P.PipelineSpec.from_dict(_spec_dict(tmp_path / 'wd', beams))
    st = PD.run(spec, from_stage='aggregate', log=lambda *a, **k: None)
    assert _CHAIN_CALLS['a'] == 1, ("--from aggregate must not re-run the "
                                    "chains: that is the point of a stage "
                                    "boundary")
    assert st.report['aggregate']['resumed'] is False
    assert st.report['readout']['resumed'] is False


def test_a_changed_chain_spec_orphans_the_chain_checkpoint(tmp_path):
    beams = [_beam_cfg('a', (0.0, 0.0), (0.0, 0.0))]
    _CHAIN_CALLS.clear()
    _run(tmp_path / 'wd', beams)
    assert _CHAIN_CALLS['a'] == 1
    _run(tmp_path / 'wd', beams, chain={'ray_subsample': 4})
    assert _CHAIN_CALLS['a'] == 2, ("a chain-stage spec change must orphan "
                                    "the per-beam artifact rather than serve "
                                    "a field computed under other settings")


def test_a_changed_readout_does_not_orphan_the_chains(tmp_path):
    beams = [_beam_cfg('a', (0.0, 0.0), (0.0, 0.0))]
    _CHAIN_CALLS.clear()
    _run(tmp_path / 'wd', beams)
    _run(tmp_path / 'wd', beams, readout={'n_out': 256})
    assert _CHAIN_CALLS['a'] == 1


def test_only_runs_the_named_stage_but_still_verifies_upstream(tmp_path):
    beams = [_beam_cfg('a', (0.0, 0.0), (0.0, 0.0))]
    _CHAIN_CALLS.clear()
    _run(tmp_path / 'wd', beams)
    spec = P.PipelineSpec.from_dict(_spec_dict(tmp_path / 'wd', beams))
    st = PD.run(spec, only=['readout'], log=lambda *a, **k: None)
    assert _CHAIN_CALLS['a'] == 1
    assert st.report['chains']['resumed'] == 1
    # ``only`` selects WHICH stages may run; it does not force them.  A stage
    # whose key still matches is still resumed, because "run exactly these"
    # and "recompute" are different requests and conflating them would make
    # every farmed-out stage rebuild what it was handed.
    assert st.report['readout']['resumed'] is True
    st = PD.run(spec, only=['readout'], from_stage='readout',
                log=lambda *a, **k: None)
    assert _CHAIN_CALLS['a'] == 1
    assert st.report['readout']['resumed'] is False


def test_single_beam_null_control_is_bit_identical_to_the_direct_call(
        tmp_path):
    """THE NULL CONTROL.  One beam, unit weight: the pipeline's frame must be
    the SAME BYTES as calling ``re_reference`` and the exact readout by hand.

    ``aggregate`` of one field with unit weight is the ``re_reference`` it
    delegates to (``0 + x == x`` exactly), so anything the staging adds --
    checkpoint round trips, the leg plan, the frame bookkeeping -- shows up
    here as a difference.  ``np.array_equal``, not ``allclose``: a staging
    layer that perturbs the answer at all has done something it was not asked
    to do.

    The beam is DECENTRED (30 um off the common grid's centre, on its own
    origin), so the resample and the analytic carrier difference are both
    live; a beam on the common lattice would take the identity short-circuit
    and prove nothing."""
    centre = (3.0e-5, 0.0)
    frame = centre
    beams = [_beam_cfg('a', centre, weight=(1.0, 0.0))]
    st = _run(tmp_path / 'wd', beams)
    got = np.load(os.path.join(st.workdir, 'readout',
                               f"{st.cache['agg_tag']}__full",
                               'frames.npz'))['f_a']

    E, grid_k, car_k = _synth_full_field(centre, centre)
    f = CarrierField.from_full_field(E, grid_k, car_k, LAM)
    car_c = CarrierSpec(R=R_AP, centre=(0.0, 0.0), tilt=(0.0, 0.0),
                        piston=0.0)
    grid_c = FieldGrid((N_COM, N_COM), DX_COM, origin=(0.0, 0.0))
    rr = re_reference(f, car_c, grid_c, support_frac=0.99999,
                      nyquist_margin=1.0, on_nyquist='error',
                      on_window='warn', bandlimit=False)
    want = carrier_referenced_exact_focus_readout(
        rr.full_field(), R_AP, Z_LEG, LAM, DX_COM, dx_out=DX_OUT,
        N_out=N_OUT, tilt=(0.0, 0.0), centre=(0.0, 0.0), centre_out=frame,
        N_fine=N_COM, window_factor=1e6, ram_budget=float('inf'),
        on_ram_cap='error', on_replica='warn', on_readout_window='warn',
        on_n_fine_cap='error')
    assert got.shape == want.shape
    assert np.array_equal(got, want), (
        "the single-beam pipeline is not bit-identical to the direct "
        "primitive call: max |d| = %.3e"
        % float(np.max(np.abs(got - want))))


def test_readout_reads_frames_that_were_not_summed(tmp_path):
    """The crosstalk census, structurally: sum ONE beam and read ALL frames.

    The shipped per-order path cannot express this at all -- its off-diagonal
    is zero by construction, not by physics (PROBE_SUM_AT_APERTURE S6.4)."""
    beams = [_beam_cfg('a', (0.0, 0.0), (0.0, 0.0)),
             _beam_cfg('b', (3e-5, 0.0))]
    st = _run(tmp_path / 'wd', beams, aggregate={'include': ['a']})
    rows = st.report['readout']['rows']
    assert sorted(rows) == ['a', 'b']
    assert st.report['readout']['included'] == ['a']
    # the diagonal carries essentially all of the power; the off-diagonal is
    # what the architecture can see and the shipped path cannot
    assert rows['a']['spot']['power'] > 1e3 * rows['b']['spot']['power']


def test_batched_aggregation_matches_one_call_to_float64(tmp_path):
    """``batch_size >= K`` is ONE ``aggregate`` call; a smaller batch changes
    only the ORDER of a float64 summation.  Both are asserted, because the
    first is what makes the acceptance runs reproduce a primitive exactly and
    the second is what makes a 32-beam run affordable."""
    beams = [_beam_cfg('a', (0.0, 0.0), (0.0, 0.0)),
             _beam_cfg('b', (3e-5, 0.0)),
             _beam_cfg('c', (-3e-5, 2e-5))]
    st1 = _run(tmp_path / 'w1', beams, aggregate={'batch_size': 8})
    st2 = _run(tmp_path / 'w2', beams, aggregate={'batch_size': 1})
    assert st1.report['aggregate']['n_batches'] == 1
    assert st2.report['aggregate']['n_batches'] == 3
    f1 = PA.load_field(os.path.join(st1.workdir, 'aggregate',
                                    st1.cache['agg_tag'], 'summed.zarr'),
                       st1.keys['aggregate'])
    f2 = PA.load_field(os.path.join(st2.workdir, 'aggregate',
                                    st2.cache['agg_tag'], 'summed.zarr'),
                       st2.keys['aggregate'])
    d = np.linalg.norm(f1.envelope - f2.envelope) / np.linalg.norm(f1.envelope)
    assert d < 1e-15, f"batching moved the sum by {d:.3e}"


def test_the_nyquist_guard_is_live_through_the_pipeline(tmp_path):
    """The guard is a PIPELINE-level refusal, not a primitive-level detail.

    A common grid too coarse for the carrier difference returns a populated,
    everywhere-finite, credible-looking field that is wrong by O(1)
    (BUILD_CARRIER_FIELD S2.1 measured 1.414 = sqrt(2), i.e. uncorrelated with
    the truth).  A staged driver that swallowed the refusal would put that
    back."""
    beams = [_beam_cfg('a', (3e-5, 0.0), (4e-6, 0.0))]
    with pytest.raises(ValueError) as e:
        _run(tmp_path / 'wd', beams,
             aggregate={'n_common': 12, 'dx_common': 3.2e-5})
    assert 'cannot hold the carrier difference' in str(e.value)
    assert 'reconstruct' in str(e.value)
    # and it is dispositionable, per the library's own on_* vocabulary
    st = _run(tmp_path / 'wd2', beams,
              aggregate={'n_common': 12, 'dx_common': 3.2e-5,
                         'on_nyquist': 'ignore', 'on_window': 'ignore'})
    assert st.report['aggregate']['worst_nyquist_margin'] < 1.0


def test_aggregate_names_a_chain_checkpoint_that_vanished(tmp_path):
    """A field artifact deleted BETWEEN the chains stage and the aggregation
    that consumes it must be named, not filled in.

    (Re-running the pipeline would simply rebuild it -- that is what resume is
    for.  This is the in-flight case, which is real on a shared box: the
    aperture checkpoints are gigabytes and get cleaned up under running
    jobs.)"""
    import shutil
    beams = [_beam_cfg('a', (0.0, 0.0))]
    spec = P.PipelineSpec.from_dict(_spec_dict(tmp_path / 'wd', beams))
    st = PD.run(spec, through='chains', log=lambda *a, **k: None)
    z = os.path.join(st.workdir, 'chains', 'a.zarr')
    if os.path.isdir(z):
        shutil.rmtree(z)
    if os.path.exists(z + '.npz'):
        os.remove(z + '.npz')
    with pytest.raises(PD.PipelineError, match='chain checkpoint'):
        PD.stage_aggregate(st, force=True)


def test_run_refuses_an_unknown_stage_name(tmp_path):
    spec = P.PipelineSpec.from_dict(
        _spec_dict(tmp_path, [_beam_cfg('a', (0, 0), (0, 0))]))
    with pytest.raises(SpecError, match='from_stage'):
        PD.run(spec, from_stage='aggregation')
    with pytest.raises(SpecError, match='are not stages'):
        PD.run(spec, only=['aggregation'])


# ===========================================================================
# (e) LINEAGE -- a per-beam artifact must not be orphaned by its siblings
# ===========================================================================
def test_adding_a_beam_does_not_orphan_its_siblings(tmp_path):
    """Dropping a 32-order fan to 8, or adding one order to a 3-order study,
    must not throw away 32 aperture fields.  A per-beam artifact depends on
    how that beam is DEFINED, not on which siblings were listed with it."""
    _CHAIN_CALLS.clear()
    _run(tmp_path / 'wd', [_beam_cfg('a', (0.0, 0.0))])
    _run(tmp_path / 'wd', [_beam_cfg('a', (0.0, 0.0)),
                           _beam_cfg('b', (3e-5, 0.0))])
    assert _CHAIN_CALLS['a'] == 1, "beam 'a' was recomputed when 'b' appeared"
    assert _CHAIN_CALLS['b'] == 1
    # ...and dropping back to one beam still reuses it
    _run(tmp_path / 'wd', [_beam_cfg('a', (0.0, 0.0))])
    assert _CHAIN_CALLS['a'] == 1


def test_changing_one_beam_definition_orphans_only_that_beam(tmp_path):
    _CHAIN_CALLS.clear()
    both = [_beam_cfg('a', (0.0, 0.0)), _beam_cfg('b', (3e-5, 0.0))]
    _run(tmp_path / 'wd', both)
    moved = [_beam_cfg('a', (0.0, 0.0)), _beam_cfg('b', (2e-5, 0.0))]
    _run(tmp_path / 'wd', moved)
    assert _CHAIN_CALLS['a'] == 1
    assert _CHAIN_CALLS['b'] == 2, ("a beam whose own definition moved must "
                                    "be recomputed")


def test_a_non_selection_decompose_param_orphans_every_beam(tmp_path):
    """The selection keys are the ONLY ones excluded from the lineage.  A
    parameter that changes how beams are BUILT -- a launch chain, a cell
    resolution, a cache directory -- still orphans all of them."""
    _CHAIN_CALLS.clear()
    beams = [_beam_cfg('a', (0.0, 0.0))]
    d = _spec_dict(tmp_path / 'wd', beams)
    PD.run(P.PipelineSpec.from_dict(d), log=lambda *a, **k: None)
    d2 = _spec_dict(tmp_path / 'wd', beams)
    d2['decompose']['params']['context'] = {'launch': 'other'}
    PD.run(P.PipelineSpec.from_dict(d2), log=lambda *a, **k: None)
    assert _CHAIN_CALLS['a'] == 2


def test_the_decompose_artifact_itself_is_keyed_on_the_selection(tmp_path):
    """The lineage exemption must not leak into ``beams.json``: a 1-beam beam
    list served to a 2-beam run would silently drop a beam, which is a wrong
    answer with no signature."""
    one = P.PipelineSpec.from_dict(
        _spec_dict(tmp_path, [_beam_cfg('a', (0.0, 0.0))]))
    two = P.PipelineSpec.from_dict(
        _spec_dict(tmp_path, [_beam_cfg('a', (0.0, 0.0)),
                              _beam_cfg('b', (3e-5, 0.0))]))
    assert PA.stage_key(one, 'decompose')[1] != PA.stage_key(two,
                                                             'decompose')[1]
    assert PA.decompose_lineage_digest(one) == PA.decompose_lineage_digest(two)


# ---------------------------------------------------------------------------
# RESUME INTEGRITY -- presence is not completeness
# ---------------------------------------------------------------------------
# A Zarr store with a chunk file missing is a perfectly VALID store: the
# reader fills the hole with the array's fill_value and hands back an array of
# the right shape and dtype without a word.  So an `os.path.exists` resume
# gate accepted a partial checkpoint as whole -- rel L2 7.071e-01 on the
# summed field, frame power 1.634e-09 -> 4.78e-20 (eleven decades), and a
# CLEAN energy ledger, because a hole reads as zeros and zeros conserve
# energy perfectly (VERIFY_ARCHITECTURE P0-3).  The number that catches it was
# already computed, already stored, and never compared.


def _chunk_files(zpath):
    out = []
    for root, _d, files in os.walk(zpath):
        for f in files:
            if f != 'zarr.json':
                out.append(os.path.join(root, f))
    return sorted(out)


def _hole_the_checkpoint(path):
    """Punch a hole in WHICHEVER container ``save_field`` produced, in the way
    that container hides it, and return the container's name.

    ``save_field`` dispatches on ``have_zarr()``, and ``zarr`` is an optional
    extra keyed on ``python_version >= "3.11"`` (pyproject: zarr v3 itself
    requires 3.11+), so the Python 3.10 leg of CI legitimately checkpoints to
    ``.npz`` and has NO chunk files at all.  A test that can only hole a Zarr
    store therefore stops testing on that leg instead of failing -- which is
    how this arm met the runner.

    The defect is the same in both containers and so is the hole: a container
    that is PRESENT, VALID, and returns an array of exactly the right shape
    and dtype whose content is short.  Zarr does it by deleting a chunk file
    (the reader substitutes ``fill_value``); the npz does it by writing a
    block of the envelope back as zeros.  Either way the power on disk no
    longer matches the ``power_field`` the metadata recorded, which is the
    number the resume gate compares."""
    if os.path.isdir(path):
        chunks = _chunk_files(path)
        assert chunks, f'{path} is a store directory but holds no chunks'
        os.remove(chunks[len(chunks) // 2])
        return 'zarr'
    npz = path + '.npz'
    assert os.path.exists(npz), (
        f'no checkpoint container at {path} or {npz} -- save_field produced '
        f'a third format this test does not know how to hole')
    with np.load(npz, allow_pickle=False) as d:
        env = np.array(d['envelope'])
        meta = np.array(d['meta'])
    ny = env.shape[-2]
    lo = ny // 2
    env[lo:lo + max(1, ny // 8), :] = 0.0
    np.savez_compressed(npz, envelope=env, meta=meta)
    return 'npz'


def _chain_meta_payload(wd, key='a'):
    with open(os.path.join(str(wd), 'chains', f'{key}.json'),
              encoding='cp1252') as fh:
        return json.load(fh)['payload']


def test_the_stored_power_matches_the_power_on_disk(tmp_path):
    """The comparison the resume gate makes, on a healthy run: the metadata's
    ``power_field`` and the power re-integrated straight off the checkpoint
    are the same sum of the same doubles."""
    beams = [_beam_cfg('a', (0.0, 0.0), (0.0, 0.0))]
    _CHAIN_CALLS.clear()
    _run(tmp_path / 'wd', beams)
    fz = os.path.join(str(tmp_path / 'wd'), 'chains', 'a.zarr')
    stored = _chain_meta_payload(tmp_path / 'wd')['power_field']
    on_disk = PA.field_power_on_disk(fz)
    assert on_disk == pytest.approx(stored, rel=1e-15)
    ok, why = PA.field_is_complete(fz, stored)
    assert ok, why


def test_a_partial_field_checkpoint_is_rejected_not_resumed(tmp_path):
    """FAIL-BEFORE: punch a hole in a completed checkpoint.

    ``field_exists`` said yes and the stage printed RESUMED.  The content
    check must say no, and the beam must be recomputed.

    Run against WHICHEVER container this interpreter's ``save_field``
    produced -- see ``_hole_the_checkpoint``.  The claim is about the resume
    GATE, which is container-independent (``field_is_complete`` reads both),
    so pinning it to the Zarr store would have retired the arm on every
    Python where the zarr extra does not resolve."""
    beams = [_beam_cfg('a', (0.0, 0.0)), _beam_cfg('b', (3e-5, 0.0))]
    _CHAIN_CALLS.clear()
    _run(tmp_path / 'wd', beams)
    assert _CHAIN_CALLS == {'a': 1, 'b': 1}

    fz = os.path.join(str(tmp_path / 'wd'), 'chains', 'a.zarr')
    stored = _chain_meta_payload(tmp_path / 'wd')['power_field']
    container = _hole_the_checkpoint(fz)
    assert container in ('zarr', 'npz'), container

    # presence still says yes -- that is the whole defect
    assert PA.field_exists(fz), container
    # ... and the power off the holed store is not the stored power
    holed = PA.field_power_on_disk(fz)
    assert holed != pytest.approx(stored, rel=1e-9)
    ok, why = PA.field_is_complete(fz, stored)
    assert not ok
    assert 'not COMPLETE' in why

    _CHAIN_CALLS.clear()
    st = _run(tmp_path / 'wd', beams)
    assert _CHAIN_CALLS.get('a', 0) == 1, (
        'the beam whose checkpoint was holed must be recomputed')
    assert _CHAIN_CALLS.get('b', 0) == 0, (
        'the intact sibling must still resume -- the check is per artifact')
    assert st.report['chains']['ran'] == 1
    assert st.report['chains']['resumed'] == 1
    # ... and the artifact is whole again
    ok, _why = PA.field_is_complete(
        fz, _chain_meta_payload(tmp_path / 'wd')['power_field'])
    assert ok


def test_an_older_artifact_without_a_recorded_power_still_resumes(tmp_path):
    """Backward compatibility: nothing to compare against is reported as
    'presence only' rather than silently implying a check happened."""
    beams = [_beam_cfg('a', (0.0, 0.0), (0.0, 0.0))]
    _CHAIN_CALLS.clear()
    _run(tmp_path / 'wd', beams)
    fz = os.path.join(str(tmp_path / 'wd'), 'chains', 'a.zarr')
    ok, why = PA.field_is_complete(fz, None)
    assert ok and 'presence only' in why


# ---------------------------------------------------------------------------
# THE READOUT TERMS IN THE CHAINS / DECOMPOSE KEYS
# ---------------------------------------------------------------------------
# `decompose` quantises every frame_centre to dx_out, and a chain that
# captures a reference tile emits one sampled at dx_out on an n_out window.
# Neither was in the key, so changing dx_out 1.5e-7 -> 2.0e-7 resumed, served
# the tile produced at the OLD pitch, and moved reference_spot['power'] by
# exactly (2.0/1.5)**2 (VERIFY_ARCHITECTURE P0-4).


def test_a_changed_readout_reruns_a_chain_that_captured_a_tile(tmp_path):
    """The artifact CONTAINS a readout-sampled product, so the readout is in
    its key."""
    beams = [_beam_cfg('a', (0.0, 0.0), (0.0, 0.0))]
    _CHAIN_CALLS.clear()
    _run(tmp_path / 'wd', beams, chain={'capture_reference_tile': True})
    assert _CHAIN_CALLS['a'] == 1
    _run(tmp_path / 'wd', beams, chain={'capture_reference_tile': True},
         readout={'dx_out': DX_OUT * 4.0 / 3.0})
    assert _CHAIN_CALLS['a'] == 2, (
        'a chain that captured a tile at the old pitch must not serve it at '
        'a new one')
    _run(tmp_path / 'wd', beams, chain={'capture_reference_tile': True},
         readout={'n_out': N_OUT // 2})
    assert _CHAIN_CALLS['a'] == 3, 'n_out sizes the tile window too'


def test_the_readout_terms_are_in_the_keys_that_consume_them(tmp_path):
    """Directly on the key slices, so the reason is legible without a run."""
    beams = [_beam_cfg('a', (0.0, 0.0), (0.0, 0.0))]
    base = P.PipelineSpec.from_dict(_spec_dict(tmp_path / 'wd', beams))
    moved = P.PipelineSpec.from_dict(
        _spec_dict(tmp_path / 'wd', beams,
                   readout={'dx_out': DX_OUT * 4.0 / 3.0}))
    # decompose quantises frame_centre to dx_out -> always keyed on it
    assert (PA.stage_key(base, 'decompose')[1]
            != PA.stage_key(moved, 'decompose')[1])
    # chains: keyed on the readout only when it captures a readout product
    tiled = P.PipelineSpec.from_dict(
        _spec_dict(tmp_path / 'wd', beams,
                   chain={'capture_reference_tile': True}))
    tiled_moved = P.PipelineSpec.from_dict(
        _spec_dict(tmp_path / 'wd', beams,
                   chain={'capture_reference_tile': True},
                   readout={'dx_out': DX_OUT * 4.0 / 3.0}))
    assert (PA.stage_key(tiled, 'chains')[1]
            != PA.stage_key(tiled_moved, 'chains')[1])
    assert (PA.stage_key(base, 'chains')[1]
            == PA.stage_key(moved, 'chains')[1]), (
        'with no tile the artifact is the aperture field, whose pitch comes '
        'from n_fine_cap -- the readout cannot reach it, so orphaning 32 '
        'two-minute chains on a frame change would be a false positive')


# ---------------------------------------------------------------------------
# ATOMIC PAYLOAD WRITES
# ---------------------------------------------------------------------------
# ``write_json`` was already atomic; ``save_field`` was not, and
# ``save_carrier_field_zarr`` deletes the existing group BEFORE it rewrites
# it.  A kill 0.35 s in therefore left a headless store beside a still
# key-matching .json, and every later run reported "0 run, N resumed" then
# died on "missing or was produced by a different configuration".  It never
# healed, because re-running resumed (VERIFY_ARCHITECTURE P1-2).


def test_an_interrupted_field_write_leaves_the_previous_artifact_intact(
        tmp_path, monkeypatch):
    beams = [_beam_cfg('a', (0.0, 0.0), (0.0, 0.0))]
    _CHAIN_CALLS.clear()
    _run(tmp_path / 'wd', beams)
    fz = os.path.join(str(tmp_path / 'wd'), 'chains', 'a.zarr')
    good = PA.field_power_on_disk(fz)
    assert good and good > 0.0

    import lumenairy.propagators.carrier_field as _cf

    reached = []

    def _boom(*a, **kw):
        reached.append(1)
        raise KeyboardInterrupt('simulated taskkill mid-write')

    # ``save_field`` dispatches its PAYLOAD write on ``have_zarr()``, and the
    # zarr extra is keyed on ``python_version >= "3.11"``, so the 3.10 leg of
    # CI takes the npz writer and the zarr writer is never called.  Inject at
    # BOTH payload writers, and assert below that the injection was actually
    # reached -- otherwise this arm passes vacuously the moment the dispatch
    # grows a branch, which is exactly how it met the runner.
    monkeypatch.setattr(_cf, 'save_carrier_field_zarr', _boom)
    monkeypatch.setattr(PA, '_save_field_npz', _boom)
    g = FieldGrid((32, 32), 1e-6)
    victim = CarrierField(np.ones((32, 32), dtype=np.complex128), g,
                          CarrierSpec(R=-1.0), LAM)
    with pytest.raises(KeyboardInterrupt):
        PA.save_field(fz, victim, {'k': 1})
    monkeypatch.undo()
    assert reached, (
        'the interrupt was never injected -- save_field wrote the payload '
        'through neither save_carrier_field_zarr nor _save_field_npz, so '
        'this test proved nothing about the atomic path')

    assert PA.field_power_on_disk(fz) == good, (
        'the interrupted write destroyed the previous good artifact')
    leftovers = [p for p in os.listdir(os.path.dirname(fz)) if '.tmp-' in p]
    assert not leftovers, f'temp artifacts survived: {leftovers}'

    # and the run still resumes rather than deadlocking
    _CHAIN_CALLS.clear()
    st = _run(tmp_path / 'wd', beams)
    assert _CHAIN_CALLS.get('a', 0) == 0
    assert st.report['chains']['resumed'] == 1


def test_save_field_replaces_atomically_and_leaves_no_temp(tmp_path):
    """A successful overwrite goes through a temp path and lands whole."""
    g = FieldGrid((64, 64), 1e-6)
    x, y = g.axes()
    env = np.exp(-((x[None, :] ** 2 + y[:, None] ** 2) / 20e-6 ** 2)
                 ).astype(np.complex128)
    f1 = CarrierField(env, g, CarrierSpec(R=-1.0), LAM)
    f2 = CarrierField(2.0 * env, g, CarrierSpec(R=-1.0), LAM)
    p = str(tmp_path / 'f.zarr')
    PA.save_field(p, f1, {'k': 1})
    assert PA.field_power_on_disk(p) == pytest.approx(f1.power(), rel=1e-15)
    PA.save_field(p, f2, {'k': 2})
    assert PA.field_power_on_disk(p) == pytest.approx(f2.power(), rel=1e-15)
    assert not [q for q in os.listdir(str(tmp_path)) if '.tmp-' in q]
    assert PA.load_field(p, {'k': 2}) is not None
    assert PA.load_field(p, {'k': 1}) is None


def test_the_reference_tile_is_written_atomically(tmp_path):
    """``np.save`` in place is the same hazard on the tile the readout stage
    loads on presence alone."""
    p = str(tmp_path / 'ref.npy')
    a = np.arange(64, dtype=np.complex128).reshape(8, 8)
    PA.save_npy_atomic(p, a)
    assert np.array_equal(np.load(p), a)
    b = a * 3.0
    PA.save_npy_atomic(p, b)
    assert np.array_equal(np.load(p), b)
    assert not [q for q in os.listdir(str(tmp_path)) if '.tmp-' in q]
