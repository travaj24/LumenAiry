"""Structure and round trips of the lens configuration objects (audit 2026-09-11, WP-A16).

WHY THIS FILE EXISTS.  The audit's section-14 item 13 asks for config objects
over the ``apply_real_lens`` family and names three concrete wins, of which the
third is the one a test can enforce: *"the 'knob silently discarded' class of
bug becomes a ``dataclasses.fields()`` round-trip assertion"*.  This file is
that assertion, plus the structural gates that keep the field tables honest.

WHAT IT ASSERTS, and why each one is not vacuous:

1. **Every table entry is real.**  For each (entry point, config field) pair in
   ``lens_config``'s four tables, the keyword it maps to must EXIST on that
   entry point and its signature default must EQUAL the dataclass default.
   That equality is not decoration -- the whole precedence rule ("a field that
   differs from its default is a request") is only well defined while it
   holds, and a sibling that quietly changed a default would otherwise turn a
   config into a silent behaviour change.
2. **Every parameter is classified.**  Every keyword-only parameter of every
   entry point must be either a config field, one of the four contract
   parameters, or listed in ``KWARG_ONLY`` with a written reason.  A parameter
   can therefore be *excluded* but not *forgotten* -- which is exactly the
   failure the audit found (a knob nobody noticed had stopped being read).
3. **The round trips close.**  ``from_kwargs(**to_kwargs()) == cfg`` for a
   config that sets a field in each of the four groups, and
   ``to_kwargs(entry_point=...)`` renames to that entry point's spelling.
4. **The precedence rule holds**, including the two refusals: a keyword that
   disagrees with a set field, and a set field the entry point has no
   parameter for.
5. **Validation runs in ``__post_init__``** -- the audit's second win -- with
   no field, no prescription and no call.

NO NUMERIC BARS.  Every comparison here is exact: identity of defaults,
set equality of parameter names, dataclass equality.  The field-level
bit-identity of a configured call against the equivalent keyword call is a
separate file (``test_audit2609_a16_lens_config_bit_identity.py``) because it
needs the covering-array fixture and costs seconds rather than milliseconds.
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

from lumenairy.elements import lens_config as lc
from lumenairy.elements._lens_real import apply_real_lens
from lumenairy.elements._lens_traced import (
    apply_real_lens_traced,
    prepare_real_lens_traced,
)
from lumenairy.elements._lens_traced_multibranch import (
    apply_real_lens_traced_multibranch,
)
from lumenairy.elements.lens_config import (
    LensConfig,
    LensGeometry,
    LensNumerics,
    LensPhysics,
    LensResources,
)
from lumenairy.elements.lenses_gbd import apply_real_lens_gbd
from lumenairy.elements.lenses_maslov import apply_real_lens_maslov
from lumenairy.propagators.fga import apply_real_lens_fga

#: name -> the live function, so the walkers below read SIGNATURES rather than
#: a transcription of them.
ENTRY_POINTS = {
    'apply_real_lens': apply_real_lens,
    'apply_real_lens_traced': apply_real_lens_traced,
    'prepare_real_lens_traced': prepare_real_lens_traced,
    'apply_real_lens_maslov': apply_real_lens_maslov,
    'apply_real_lens_gbd': apply_real_lens_gbd,
    'apply_real_lens_fga': apply_real_lens_fga,
    'apply_real_lens_traced_multibranch': apply_real_lens_traced_multibranch,
}

CONFIG_PARAMS = ('geometry', 'numerics', 'resources', 'config')

#: The FIFTH config parameter.  It is not in ``CONFIG_PARAMS`` because only the
#: entry points that own at least one field of a group need that group's
#: parameter, and ``LensPhysics``'s fields exist on ``apply_real_lens`` alone
#: (the traced / Maslov / GBD / FGA models build their screens from a ray trace,
#: so none of the analytic screen's model-term switches has a counterpart
#: there).  ``test_the_physics_parameter_is_declared_exactly_where_it_applies``
#: below is the gate that keeps those two facts in step, in BOTH directions.
PHYSICS_PARAM = 'physics'


# A tiny biconvex singlet whose aperture FITS the 8 x 8 grid (0.8 mm across at
# dx = 1e-4), so the entry point's aperture-vs-grid advisory does not fire and
# the precedence tests below assert on the refusal they are about rather than
# on a warning they did not ask for.
_SMALL_RX = dict(
    surfaces=[dict(radius=0.05, glass_before='AIR', glass_after='N-BK7'),
              dict(radius=-0.05, glass_before='N-BK7', glass_after='AIR')],
    thicknesses=[3.0e-3], aperture_diameter=4.0e-4)


def _small_case():
    """``(E_in, base kwargs)`` for the precedence tests."""
    return (np.ones((8, 8), dtype=np.complex128),
            dict(prescription=_SMALL_RX, wavelength=633e-9, dx=1e-4))


def _kwonly(fn):
    return {n: p for n, p in inspect.signature(fn).parameters.items()
            if p.kind is inspect.Parameter.KEYWORD_ONLY}


def _dataclass_defaults(dcls):
    import dataclasses
    return {f.name: f.default for f in dataclasses.fields(dcls)}


# ---------------------------------------------------------------------------
# 1. The tables describe the real signatures
# ---------------------------------------------------------------------------

def test_the_entry_point_set_is_the_one_lens_config_declares():
    """Counter-pin on this file: if ``lens_config`` grows or loses an entry
    point and this module's table is not updated, every walker below would
    quietly test a smaller set and still pass."""
    assert set(lc.ENTRY_POINTS) == set(ENTRY_POINTS), (
        f'lens_config declares {sorted(lc.ENTRY_POINTS)} but this test file '
        f'knows {sorted(ENTRY_POINTS)}.  Add the new entry point here (and to '
        f'the bit-identity file) rather than letting the walkers shrink.')


@pytest.mark.parametrize('ep', sorted(ENTRY_POINTS))
def test_every_entry_point_accepts_the_four_config_parameters(ep):
    params = _kwonly(ENTRY_POINTS[ep])
    missing = [p for p in CONFIG_PARAMS if p not in params]
    assert not missing, (
        f'{ep} does not accept {missing}; a config object handed to it would '
        f'be a TypeError instead of a configuration.')
    for p in CONFIG_PARAMS:
        assert params[p].default is None, (
            f'{ep}: {p}= must default to None -- the "was a config passed?" '
            f'test is `is not None`, and a non-None default would send every '
            f'call down the re-dispatch path.')


@pytest.mark.parametrize('ep', sorted(ENTRY_POINTS))
def test_every_table_entry_names_a_real_keyword_with_the_same_default(ep):
    """The load-bearing structural gate.

    A mapping ``field -> keyword`` is only meaningful if the keyword exists and
    carries the SAME default; the precedence rule reads both sides as "is this
    a request?" and compares them against their respective defaults.  If a
    sibling's default drifted (say ``caustic_band`` became ``'plain'`` on one
    of the two functions that take it), a config that set nothing would start
    changing behaviour on one of them and not the other -- silently.
    """
    params = _kwonly(ENTRY_POINTS[ep])
    problems = []
    for _, dcls, table in lc._GROUPS:
        dflt = _dataclass_defaults(dcls)
        for fieldname, kw in table[ep].items():
            if fieldname not in dflt:
                problems.append(
                    f'{dcls.__name__}.{fieldname} is not a field of that '
                    f'dataclass')
                continue
            if kw not in params:
                problems.append(
                    f'{dcls.__name__}.{fieldname} -> {ep}({kw}=...) but {ep} '
                    f'has no such keyword-only parameter')
                continue
            sig_default = params[kw].default
            if sig_default is inspect.Parameter.empty:
                problems.append(
                    f'{ep}({kw}=...) is REQUIRED, so it cannot carry a '
                    f'config default')
                continue
            if not lc._same(sig_default, dflt[fieldname]):
                problems.append(
                    f'{dcls.__name__}.{fieldname} defaults to '
                    f'{dflt[fieldname]!r} but {ep}({kw}=...) defaults to '
                    f'{sig_default!r}')
    assert not problems, (
        f'{ep}: the lens_config field tables disagree with the signature:\n  '
        + '\n  '.join(problems)
        + '\n\nEither the signature changed (update the table AND the field '
          'docstring) or the field belongs in KWARG_ONLY with a reason.')


@pytest.mark.parametrize('ep', sorted(ENTRY_POINTS))
def test_every_keyword_is_classified_as_field_contract_or_documented_exclusion(ep):
    """The "silently ignored kwarg" detector the audit asked for.

    Walks the LIVE signature.  A parameter that is neither a config field, nor
    one of the contract names, nor an entry in ``KWARG_ONLY`` with a written
    reason, fails here -- so a new keyword cannot be added to any of these
    seven functions without someone deciding, in writing, whether it is
    configurable.
    """
    params = set(_kwonly(ENTRY_POINTS[ep]))
    fielded = set()
    for _, _, table in lc._GROUPS:
        fielded |= set(table[ep].values())
    documented = set(lc.KWARG_ONLY.get(ep, {}))
    unclassified = sorted(
        params - fielded - documented - set(lc.CONTRACT_PARAMETERS))
    assert not unclassified, (
        f'{ep}: {unclassified} are keyword parameters that lens_config '
        f'neither carries as a field nor documents as deliberately '
        f'keyword-only.  Add each to a field table (with its docstring) or to '
        f'lens_config.KWARG_ONLY with the reason it is excluded.')
    # and the reverse: nothing documented as excluded may have vanished
    stale = sorted(documented - params)
    assert not stale, (
        f'{ep}: KWARG_ONLY documents {stale}, which {ep} no longer accepts.  '
        f'Delete the entry -- a stale exclusion is a hiding place.')
    # every reason is a real sentence, not a placeholder
    thin = sorted(k for k, v in lc.KWARG_ONLY.get(ep, {}).items()
                  if not isinstance(v, str) or len(v) < 30)
    assert not thin, (
        f'{ep}: KWARG_ONLY entries {thin} carry no usable reason.')


def test_the_classification_walker_is_not_vacuous():
    """Counter-pin: a fake entry point with an unclassified keyword must FAIL.

    Without this, a bug that made ``_kwonly`` return ``{}`` would turn the
    three walkers above into unconditional passes.
    """
    params = set(_kwonly(apply_real_lens))
    assert len(params) >= 25, (
        f'the keyword-only walker sees only {len(params)} parameters on '
        f'apply_real_lens; it is broken, and the classification tests above '
        f'mean nothing.')
    fielded = set()
    for _, _, table in lc._GROUPS:
        fielded |= set(table['apply_real_lens'].values())
    documented = set(lc.KWARG_ONLY['apply_real_lens'])
    pretend = params | {'a_keyword_nobody_classified'}
    left = pretend - fielded - documented - set(lc.CONTRACT_PARAMETERS)
    assert left == {'a_keyword_nobody_classified'}, (
        'injecting an unclassified keyword did not survive the set '
        'difference, so the real test could not have caught one either.')


def test_every_config_field_is_used_by_at_least_one_entry_point():
    """A field nothing accepts is dead weight that would RAISE for any caller
    who set it -- the worst of both worlds."""
    orphans = []
    for _, dcls, table in lc._GROUPS:
        used_fields = set()
        for m in table.values():
            used_fields |= set(m)
        for f in _dataclass_defaults(dcls):
            if f not in used_fields:
                orphans.append(f'{dcls.__name__}.{f}')
    assert not orphans, (
        f'{orphans} are config fields no entry point accepts.  Setting one '
        f'would raise on every call; delete it or wire it.')


# ---------------------------------------------------------------------------
# 2. Round trips
# ---------------------------------------------------------------------------

def test_from_kwargs_to_kwargs_round_trips_on_all_four_groups():
    cfg = LensConfig.from_kwargs(
        output_plane_distance=2.0e-3,      # geometry
        newton_poly_order=8,               # numerics
        n_workers=3,                       # resources
        slant_correction=True,             # physics
    )
    assert cfg.geometry.output_plane_distance == 2.0e-3
    assert cfg.numerics.newton_poly_order == 8
    assert cfg.resources.n_workers == 3
    assert cfg.physics.slant_correction is True
    flat = cfg.to_kwargs()
    assert flat == {'output_plane_distance': 2.0e-3,
                    'newton_poly_order': 8, 'n_workers': 3,
                    'slant_correction': True}, flat
    assert LensConfig.from_kwargs(**flat) == cfg


def test_to_kwargs_emits_only_requests_unless_asked_for_defaults():
    cfg = LensConfig.from_kwargs(newton_poly_order=8)
    assert cfg.to_kwargs() == {'newton_poly_order': 8}
    everything = cfg.to_kwargs(include_defaults=True)
    n_fields = sum(len(_dataclass_defaults(d)) for _, d, _ in lc._GROUPS)
    # 40 across geometry/numerics/resources + LensPhysics's 9.
    assert len(everything) == n_fields == 49, (
        f'to_kwargs(include_defaults=True) emitted {len(everything)} of '
        f'{n_fields} fields.')
    assert everything['newton_poly_order'] == 8


@pytest.mark.parametrize('ep', sorted(ENTRY_POINTS))
def test_to_kwargs_for_an_entry_point_is_exactly_what_it_accepts(ep):
    """Every field at a non-default value, narrowed to one entry point, must
    come back spelled the way that entry point spells it -- and nothing else.
    """
    everything = LensConfig(
        geometry=LensGeometry(dy=1e-6, output_plane_distance=1e-3,
                              output_plane_n=1.5, conjugate=-0.1,
                              surface_model='displaced', clip_aperture=False,
                              carrier=0.25, origin=(1e-6, 2e-6),
                              beam_centre=(0.0, 1e-6), roi=(0, 4, 0, 4)),
        numerics=LensNumerics(bandlimit=False, wave_propagator='rs',
                              ray_subsample=4, output_subsample=2,
                              remap_order=5, displaced_n_side=513,
                              min_coarse_samples_per_aperture=8,
                              fit_radius_beam_factor=1.5,
                              newton_fit='spline', newton_poly_order=8,
                              fit_basis='zernike',
                              newton_max_iters=40, inversion_method='fit',
                              amplitude_model='ray_density',
                              caustic='multibranch', caustic_band='plain',
                              caustic_ray_subsample=3,
                              caustic_min_area_ratio=1e-5, inverse_map=True),
        resources=LensResources(use_gpu=True, amp_use_gpu=True, n_workers=2,
                                parallel_amp=False,
                                parallel_amp_min_free_gb=1.0,
                                sag_dtype=np.float32, sag_chunk_rows=16,
                                accumulator_store='memmap',
                                scratch_dir='.', progress=print,
                                verbose=True),
        physics=LensPhysics(fresnel=True, slant_correction=True,
                            absorption=True, seidel_correction=True,
                            seidel_poly_order=8, surface_frame=True,
                            displaced_mode='remap',
                            displaced_obliquity='pointwise',
                            screen_obliquity=True))
    kw = everything.to_kwargs(entry_point=ep)
    accepted = set(_kwonly(ENTRY_POINTS[ep]))
    assert set(kw) <= accepted, (
        f'{ep}: to_kwargs(entry_point=...) emitted {sorted(set(kw)-accepted)}, '
        f'which {ep} does not accept.')
    expected = set()
    for _, _, table in lc._GROUPS:
        expected |= set(table[ep].values())
    assert set(kw) == expected, (
        f'{ep}: to_kwargs emitted {sorted(kw)} but the tables say {ep} takes '
        f'{sorted(expected)}.')
    # narrowed_to + to_kwargs must agree
    assert everything.narrowed_to(ep).to_kwargs(entry_point=ep) == kw


def test_from_kwargs_accepts_an_entry_points_own_spelling():
    """``apply_real_lens_traced_multibranch`` spells two shared settings
    differently.  The rename must work in BOTH directions or the config object
    is a trap for exactly the caller who needs it."""
    cfg = LensConfig.from_kwargs(
        entry_point='apply_real_lens_traced_multibranch',
        ray_subsample=3, min_area_ratio=1e-5)
    assert cfg.numerics.caustic_ray_subsample == 3
    assert cfg.numerics.caustic_min_area_ratio == 1e-5
    # ...and the traced OPL spacing is UNTOUCHED (this is the trap)
    assert cfg.numerics.ray_subsample == 8
    back = cfg.to_kwargs(entry_point='apply_real_lens_traced_multibranch')
    assert back == {'ray_subsample': 3, 'min_area_ratio': 1e-5}


def test_from_kwargs_refuses_an_unknown_name():
    with pytest.raises(ValueError) as exc:
        LensConfig.from_kwargs(levin_tol=1e-3)
    assert 'LensConfig.from_kwargs:' in str(exc.value)
    assert 'levin_tol' in str(exc.value)


def test_from_kwargs_refuses_a_field_the_named_entry_point_does_not_take():
    with pytest.raises(ValueError) as exc:
        LensConfig.from_kwargs(entry_point='apply_real_lens',
                               newton_poly_order=8)
    assert 'apply_real_lens' in str(exc.value)


def test_narrowed_to_drops_exactly_the_inapplicable_requests():
    cfg = LensConfig.from_kwargs(newton_poly_order=8, bandlimit=False,
                                 n_workers=2)
    narrowed = cfg.narrowed_to('apply_real_lens')
    assert narrowed.to_kwargs() == {'bandlimit': False}, narrowed.to_kwargs()
    # and it is idempotent
    assert narrowed.narrowed_to('apply_real_lens') == narrowed


def test_requests_is_to_kwargs():
    cfg = LensConfig.from_kwargs(use_gpu=True)
    assert cfg.requests() == cfg.to_kwargs() == {'use_gpu': True}


# ---------------------------------------------------------------------------
# 3. Validation in __post_init__
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cls,kwargs,needle', [
    (LensGeometry, dict(dy=0.0), 'dy'),
    (LensGeometry, dict(dy=-1.0), 'dy'),
    (LensGeometry, dict(output_plane_distance=np.nan), 'finite'),
    (LensGeometry, dict(output_plane_n=0.0), 'output_plane_n'),
    (LensGeometry, dict(surface_model='thick'), 'surface_model'),
    (LensGeometry, dict(clip_aperture='yes'), 'clip_aperture'),
    (LensGeometry, dict(origin=(0.0,)), 'origin'),
    (LensNumerics, dict(bandlimit=1), 'bandlimit'),
    (LensNumerics, dict(wave_propagator='asmm'), 'wave_propagator'),
    (LensNumerics, dict(ray_subsample=0), 'ray_subsample'),
    (LensNumerics, dict(ray_subsample=2.5), 'ray_subsample'),
    (LensNumerics, dict(output_subsample=0), 'output_subsample'),
    (LensNumerics, dict(remap_order=2), 'remap_order'),
    (LensNumerics, dict(newton_poly_order=0), 'newton_poly_order'),
    (LensNumerics, dict(newton_max_iters=0), 'newton_max_iters'),
    (LensNumerics, dict(caustic_min_area_ratio=0.0), 'caustic_min_area_ratio'),
    (LensNumerics, dict(caustic_min_area_ratio=2.0), 'caustic_min_area_ratio'),
    (LensNumerics, dict(fit_radius_beam_factor=0.0),
     'fit_radius_beam_factor'),
    (LensNumerics, dict(inverse_map='yes'), 'inverse_map'),
    (LensResources, dict(use_gpu='yes'), 'use_gpu'),
    (LensResources, dict(n_workers=0), 'n_workers'),
    (LensResources, dict(parallel_amp_min_free_gb=-1.0),
     'parallel_amp_min_free_gb'),
    (LensResources, dict(sag_dtype=np.float16), 'sag_dtype'),
    (LensResources, dict(sag_dtype=np.complex128), 'sag_dtype'),
    (LensResources, dict(sag_chunk_rows=-1), 'sag_chunk_rows'),
    (LensResources, dict(accumulator_store='disk'), 'accumulator_store'),
    (LensResources, dict(scratch_dir=3), 'scratch_dir'),
    (LensPhysics, dict(fresnel=1), 'fresnel'),
    (LensPhysics, dict(absorption='yes'), 'absorption'),
    (LensPhysics, dict(slant_correction=None), 'slant_correction'),
    (LensPhysics, dict(seidel_correction=1.0), 'seidel_correction'),
    (LensPhysics, dict(seidel_poly_order=0), 'seidel_poly_order'),
    (LensPhysics, dict(seidel_poly_order=6.0), 'seidel_poly_order'),
    (LensPhysics, dict(surface_frame='global'), 'surface_frame'),
    (LensPhysics, dict(displaced_mode='thin'), 'displaced_mode'),
    (LensPhysics, dict(displaced_obliquity='sagittal'),
     'displaced_obliquity'),
    # 1 and 0 must NOT masquerade as True / False -- the same identity rule
    # ``_lens_real._check_screen_obliquity_support`` applies, so a config
    # cannot accept a value the call would refuse.
    (LensPhysics, dict(screen_obliquity=1), 'screen_obliquity'),
    (LensPhysics, dict(screen_obliquity=0), 'screen_obliquity'),
    (LensPhysics, dict(screen_obliquity='on'), 'screen_obliquity'),
])
def test_post_init_refuses_bad_values_with_the_conventions_prefix(
        cls, kwargs, needle):
    """Validation runs with no field, no prescription and no call -- the
    audit's win #2.  Every message must name the class (CONVENTIONS.md
    section 2) and the offending field, because a config is often built far
    from the call that will use it."""
    with pytest.raises(ValueError) as exc:
        cls(**kwargs)
    msg = str(exc.value)
    assert msg.startswith(cls.__name__ + ':'), msg[:200]
    assert needle in msg, msg[:200]


def test_post_init_accepts_every_documented_legal_value():
    """Counter-pin to the refusal table: the checks must not be so eager that
    a legal config is refused.  A validator that raised on everything would
    make every test above pass."""
    LensGeometry(dy=1e-6, output_plane_distance=-1e-3, output_plane_n=1.5,
                 surface_model='tangent_facet_remap', clip_aperture=False,
                 origin=(1e-3, -1e-3), conjugate=-0.12,
                 carrier=np.zeros((4, 4)), beam_centre=None, roi=None)
    LensNumerics(bandlimit=False, wave_propagator='rayleigh_sommerfeld',
                 ray_subsample=1, output_subsample=4, remap_order=5,
                 min_coarse_samples_per_aperture=0,
                 fit_radius_beam_factor=1.5, newton_fit='spline',
                 newton_poly_order=12, newton_max_iters=1,
                 inversion_method='backward_trace',
                 amplitude_model='ray_density', caustic='wave',
                 caustic_band='plain', caustic_ray_subsample=1,
                 caustic_min_area_ratio=1.0, inverse_map=True)
    LensResources(use_gpu=True, amp_use_gpu=True, n_workers=1,
                  parallel_amp=False, parallel_amp_min_free_gb=0.0,
                  sag_dtype=np.float32, sag_chunk_rows=0,
                  accumulator_store='memmap', scratch_dir='.',
                  progress=None, verbose=True)
    LensResources(sag_dtype=np.float64)
    LensPhysics(fresnel=True, slant_correction=True, absorption=True,
                seidel_correction=True, seidel_poly_order=12,
                surface_frame=True, displaced_mode='split',
                displaced_obliquity='meridional', screen_obliquity=False)
    LensPhysics(screen_obliquity=True)
    # a caller-built (non-interned) 'auto' is the documented default value
    LensPhysics(screen_obliquity=''.join(['au', 'to']))


def test_the_dataclasses_are_frozen_and_compare_by_value():
    a = LensNumerics(newton_poly_order=8)
    b = LensNumerics(newton_poly_order=8)
    assert a == b and a is not b
    with pytest.raises(Exception):
        a.newton_poly_order = 9          # FrozenInstanceError


def test_lens_config_refuses_a_wrong_component_type():
    with pytest.raises(TypeError) as exc:
        LensConfig(geometry=LensNumerics())
    assert 'LensConfig:' in str(exc.value)


# ---------------------------------------------------------------------------
# 4. The precedence rule at a real entry point
# ---------------------------------------------------------------------------

def test_a_disagreeing_keyword_and_field_raise_naming_both():
    """The audit's rule: they must agree or the call raises."""
    e, kwargs = _small_case()
    with pytest.raises(ValueError) as exc:
        apply_real_lens(e, remap_order=5,
                        numerics=LensNumerics(remap_order=1), **kwargs)
    msg = str(exc.value)
    assert msg.startswith('apply_real_lens:'), msg[:200]
    assert 'remap_order' in msg and 'must agree' in msg, (
        'the refusal does not name the setting and say that the two spellings '
        'must agree, so a caller cannot tell which of the two to drop.  '
        'Message was: ' + msg[:300])


def test_an_agreeing_keyword_and_field_are_accepted():
    e, kwargs = _small_case()
    a = apply_real_lens(e, wave_propagator='rs',
                        numerics=LensNumerics(wave_propagator='rs'), **kwargs)
    b = apply_real_lens(e, wave_propagator='rs', **kwargs)
    assert np.array_equal(a, b)


def test_an_inapplicable_field_raises_and_names_the_owner():
    e, kwargs = _small_case()
    with pytest.raises(ValueError) as exc:
        apply_real_lens(e, numerics=LensNumerics(newton_poly_order=8),
                        **kwargs)
    msg = str(exc.value)
    assert msg.startswith('apply_real_lens:'), msg[:200]
    assert 'newton_poly_order' in msg
    assert 'apply_real_lens_traced' in msg, (
        'the refusal must say which entry point DOES take the setting, or the '
        'caller has no way forward')
    assert 'narrowed_to' in msg


def test_a_field_left_at_its_default_is_not_a_request():
    """The documented consequence of "default == unset", pinned so it cannot
    drift into a surprise.

    A config whose fields are all at their defaults is indistinguishable from
    passing no config -- that is what makes the configured path bit-identical
    for free.  It also means an explicitly-written default cannot CONFLICT
    with a keyword: the keyword simply wins.  That is a deliberate trade and
    this test is where it is recorded.
    """
    e, kwargs = _small_case()
    explicit = apply_real_lens(e, wave_propagator='rs', **kwargs)
    with_default_config = apply_real_lens(
        e, wave_propagator='rs',
        numerics=LensNumerics(wave_propagator=None), **kwargs)
    assert np.array_equal(explicit, with_default_config)


def test_config_and_an_explicit_component_compose_by_replacement():
    cfg = LensConfig(numerics=LensNumerics(wave_propagator='rs'))
    e, kwargs = _small_case()
    # The explicit component REPLACES the config's, whole: 'fresnel' wins.
    # 'fresnel' is the witness because it is the propagator most unlike 'rs' on
    # this geometry -- and the library correctly says so (the 3 mm in-glass hop
    # under-samples the Fresnel chirp on an 8 x 8 grid).  That advisory is the
    # library working, not the subject of this test, so it is silenced here.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got = apply_real_lens(e, config=cfg,
                              numerics=LensNumerics(wave_propagator='fresnel'),
                              **kwargs)
        want = apply_real_lens(e, wave_propagator='fresnel', **kwargs)
        rs_only = apply_real_lens(e, wave_propagator='rs', **kwargs)
    assert np.array_equal(got, want)
    assert not np.array_equal(got, rs_only), (
        "the two propagators produce the same field on this fixture, so this "
        "test cannot tell replacement from merging.  Pick another witness.")


def test_a_non_config_object_is_refused_by_type():
    e, kwargs = _small_case()
    with pytest.raises(TypeError) as exc:
        apply_real_lens(e, numerics=LensGeometry(), **kwargs)
    assert 'apply_real_lens:' in str(exc.value)


# ---------------------------------------------------------------------------
# 5. The comparison helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('a,b,want', [
    (None, None, True),
    (1, 1.0, True),
    ('auto', 'auto', True),
    ((0.0, 0.0), (0.0, 0.0), True),
    ((0.0, 0.0), (0.0, 1.0), False),
    (np.float32, np.float32, True),
    (np.dtype('float64'), np.float64, True),
    (None, 0, False),
])
def test_same_on_scalars(a, b, want):
    assert lc._same(a, b) is want


def test_same_on_arrays_does_not_raise_or_return_an_array():
    x = np.zeros((3, 3))
    assert lc._same(x, x.copy()) is True
    assert lc._same(x, np.ones((3, 3))) is False
    assert lc._same(x, np.zeros((2, 2))) is False
    assert lc._same(x, None) is False
    assert lc._same(None, x) is False


def test_same_treats_an_unorderable_comparison_as_different():
    """A setting whose ``__eq__`` REFUSES the other operand (``TypeError``) or
    whose result is not a scalar truth (``ValueError``) counts as "different"
    -- which costs at most a spurious conflict message naming both values, and
    never a silent overwrite.  Anything else propagates: an ``__eq__`` that
    raises ``RuntimeError`` is broken, not incomparable, and swallowing that
    is how a real defect hides."""
    class Refuses:
        def __eq__(self, other):
            raise TypeError('not comparable with that')

    class Ambiguous:
        def __eq__(self, other):
            raise ValueError('truth value is ambiguous')

    class Broken:
        def __eq__(self, other):
            raise RuntimeError('this is a bug, not a comparison')

    assert lc._same(Refuses(), 1) is False
    assert lc._same(Ambiguous(), 1) is False
    r = Refuses()
    assert lc._same(r, r) is True        # identity short-circuits first
    with pytest.raises(RuntimeError):
        lc._same(Broken(), 1)


def test_the_group_set_is_the_one_lens_config_declares():
    """Counter-pin, the same shape as the entry-point one: every walker in
    this file iterates ``lc._GROUPS``, so a group added there and not here
    would be tested by the walkers but by none of the explicit tables (the
    refusal parametrisation, the legal-value counter-pin, the 'everything'
    fixture), and the file would still pass."""
    assert [a for a, _, _ in lc._GROUPS] == [
        'geometry', 'numerics', 'resources', 'physics']
    assert [d for _, d, _ in lc._GROUPS] == [
        LensGeometry, LensNumerics, LensResources, LensPhysics]
    assert set(LensConfig().__dataclass_fields__) == {
        a for a, _, _ in lc._GROUPS}


def test_the_physics_parameter_is_declared_exactly_where_it_applies():
    """``physics=`` is the one config parameter that is NOT on every entry
    point, so the rule has to be checkable rather than remembered: an entry
    point declares it iff ``_PHYSICS_FOR`` gives it at least one field.

    Both directions matter.  Missing where it applies is a ``TypeError``
    instead of a configuration; present where it does not is a parameter that
    can only ever raise.
    """
    for ep, fn in sorted(ENTRY_POINTS.items()):
        params = _kwonly(fn)
        owns_fields = bool(lc._PHYSICS_FOR[ep])
        assert (PHYSICS_PARAM in params) is owns_fields, (
            f'{ep}: _PHYSICS_FOR gives it {sorted(lc._PHYSICS_FOR[ep])} but '
            f'the signature {"has" if PHYSICS_PARAM in params else "has no"} '
            f'{PHYSICS_PARAM}=.  Declare the parameter when you add the first '
            f'field, and not before.')
        if owns_fields:
            assert params[PHYSICS_PARAM].default is None, (
                f'{ep}: physics= must default to None -- the "was a config '
                f'passed?" test is `is not None`.')
    # not vacuous: at least one entry point on each side of the split
    owners = [ep for ep in ENTRY_POINTS if lc._PHYSICS_FOR[ep]]
    assert owners == ['apply_real_lens'], owners
    assert len(ENTRY_POINTS) > 1


def test_a_physics_request_handed_to_a_sibling_raises_and_names_the_owner():
    """The six entry points with no ``physics=`` parameter still refuse a
    physics request that reaches them through ``config=``: the resolver walks
    every group whatever the caller passed, so the empty ``_PHYSICS_FOR``
    entry is what produces the refusal."""
    from lumenairy.elements._lens_traced import apply_real_lens_traced
    e, kwargs = _small_case()
    with pytest.raises(ValueError) as exc:
        apply_real_lens_traced(
            e, config=LensConfig(physics=LensPhysics(fresnel=True)), **kwargs)
    msg = str(exc.value)
    assert msg.startswith('apply_real_lens_traced:'), msg[:200]
    assert 'physics.fresnel' in msg
    assert 'apply_real_lens' in msg
    assert 'narrowed_to' in msg


def test_a_physics_field_and_its_keyword_are_the_same_setting():
    """The precedence rule on the new group, both arms: agreeing is accepted
    and produces the keyword call's own answer; disagreeing raises naming
    both.  ``fresnel`` is the witness because it multiplies in the interface
    transmittances, so the two answers are visibly different."""
    e, kwargs = _small_case()
    configured = apply_real_lens(e, physics=LensPhysics(fresnel=True),
                                 **kwargs)
    keyword = apply_real_lens(e, fresnel=True, **kwargs)
    plain = apply_real_lens(e, **kwargs)
    assert np.array_equal(configured, keyword)
    assert not np.array_equal(configured, plain), (
        'fresnel=True does not move this fixture, so it cannot witness the '
        'precedence rule here.  Pick another field.')
    agreeing = apply_real_lens(e, fresnel=True,
                               physics=LensPhysics(fresnel=True), **kwargs)
    assert np.array_equal(agreeing, keyword)
    with pytest.raises(ValueError) as exc:
        apply_real_lens(e, seidel_poly_order=8,
                        physics=LensPhysics(seidel_poly_order=10), **kwargs)
    msg = str(exc.value)
    assert msg.startswith('apply_real_lens:'), msg[:200]
    assert 'seidel_poly_order' in msg and 'must agree' in msg


def test_a_default_physics_object_is_indistinguishable_from_none():
    """What makes the configured path bit-identical for free: a field left at
    its default is not a request, so ``physics=LensPhysics()`` changes no
    argument at all."""
    e, kwargs = _small_case()
    assert np.array_equal(apply_real_lens(e, physics=LensPhysics(), **kwargs),
                          apply_real_lens(e, **kwargs))
    assert LensConfig(physics=LensPhysics()).to_kwargs() == {}
