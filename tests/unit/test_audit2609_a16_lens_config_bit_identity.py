"""Bit-identity of the configured call vs the keyword call (audit 2026-09-11, WP-A16).

WHY THIS FILE EXISTS.  The config objects of ``lumenairy.elements.lens_config``
are ADDITIVE: no default changed, nothing was deprecated, and every existing
call site must keep producing the byte-identical field it produced before they
existed.  Two properties carry that promise, and both are asserted here for
EVERY entry point:

  **P1 -- the configured path is the keyword path.**  ``fn(E, ..., config=cfg)``
  and ``fn(E, ..., geometry=g, numerics=n, resources=r)`` must equal
  ``fn(E, ..., **cfg.to_kwargs(entry_point=fn.__name__))`` with
  ``np.array_equal`` -- not ``allclose``.  A re-dispatch that merges keywords
  and re-enters the same function has no numerical content of its own, so any
  difference at all is a defect, and a tolerance would only hide one.
  (This is also why there is no derived numeric bar in this file: there is no
  quantity to bound.  The comparison is exact by construction.)

  **P2 -- an unconfigured call is untouched.**  ``fn(E, ...)`` with none of the
  four new parameters must equal the same call made through a config whose
  fields are all at their defaults, and both must equal the pre-change result.
  The third leg of that is what the surrounding suite already pins (the WP-A2 /
  A3 / A4 files and the WP-A15a covering array all call these entry points with
  keywords only); this file pins the first two.

NON-VACUITY.  Every parametrised case also asserts that the settings it
configures actually MOVE the field -- ``configured != all-default`` -- so a bug
that made the config a no-op would fail here rather than pass quietly.  As
MEASURED 2026-09-12 all eight rows move, so ``_NO_MOVE_EXPECTED`` is empty; it
exists (and is asserted in both directions) so that a row which genuinely
exercises only byte-identical knobs can be exempted explicitly rather than by
deleting the check.

THE FIXTURE is WP-A15a's (``test_audit2609_a15a_lens_covering_array
.lens_covering_array_fixture``), imported rather than rebuilt: a curved-rear
AC254-ish cemented doublet under a diverging spherical wave at N = 64.  The two
work packages then agree on one geometry rather than two, and a fixture change
moves both gates together.

RUNTIME.  MEASURED 2026-09-12 on this box: the whole file is ~13 s, dominated
by the two traced arms (a warm traced call is 20-45 ms and the caustic arm is
~0.4 s) and by ``apply_real_lens_fga``'s first call, which compiles its numba
kernels.  No wall-clock is asserted anywhere (TESTING_STANDARDS S1).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.elements._lens_real import apply_real_lens
from lumenairy.elements._lens_traced import (
    apply_real_lens_traced,
    prepare_real_lens_traced,
)
from lumenairy.elements._lens_traced_multibranch import (
    apply_real_lens_traced_multibranch,
)
from lumenairy.elements import lens_config as lc
from lumenairy.elements.lens_config import LensConfig  # noqa: I001
from lumenairy.elements.lenses_gbd import apply_real_lens_gbd
from lumenairy.elements.lenses_maslov import apply_real_lens_maslov
from lumenairy.propagators.fga import apply_real_lens_fga

from .test_audit2609_a15a_lens_covering_array import (
    lens_covering_array_fixture,
)

E_IN, DX, RX, LAM = lens_covering_array_fixture()
BASE = dict(prescription=RX, wavelength=LAM, dx=DX)
#: ``ray_subsample=1`` for the traced arms: at N = 64 the shipped default of 8
#: leaves 5 coarse samples across the 6 mm aperture and the undersample guard
#: correctly refuses the call (WP-A15a measured the message).
TRACED_BASE = dict(BASE, ray_subsample=1)

#: ``(id, fn, base kwargs, settings to configure)``.  The settings are spelled
#: the way the ENTRY POINT spells them, so ``LensConfig.from_kwargs(
#: entry_point=...)`` has to perform its rename to build the config -- the
#: multibranch row exercises exactly that.
CASES = [
    ('analytic-displaced', apply_real_lens, BASE, dict(
        dy=DX, surface_model='displaced', conjugate=-120.0e-3,
        wave_propagator='asm', remap_order=3, bandlimit=True,
        sag_chunk_rows=0, accumulator_store='ram', use_gpu=False)),
    ('analytic-rs-float32sag', apply_real_lens, BASE, dict(
        wave_propagator='rs', sag_dtype=np.float32)),
    ('traced-caustic', apply_real_lens_traced, TRACED_BASE, dict(
        newton_fit='spline', newton_poly_order=4, newton_max_iters=40,
        amplitude_model='ray_density', caustic='multibranch',
        output_plane_distance=2.0e-3, caustic_band='plain',
        caustic_ray_subsample=1, fit_radius_beam_factor=1.5,
        parallel_amp=False, n_workers=1)),
    ('traced-resources', apply_real_lens_traced, TRACED_BASE, dict(
        inversion_method='newton', inverse_map=False, sag_chunk_rows=0,
        origin=(0.0, 0.0), min_coarse_samples_per_aperture=0,
        wave_propagator='rs')),
    ('maslov', apply_real_lens_maslov, BASE, dict(
        dy=DX, output_plane_distance=1.0e-3, output_plane_n=1.0,
        output_subsample=1, use_gpu=False, verbose=False)),
    ('gbd', apply_real_lens_gbd, BASE, dict(
        dy=DX, output_plane_distance=1.0e-3, output_plane_n=1.0,
        clip_aperture=False, output_subsample=1, verbose=False)),
    ('fga', apply_real_lens_fga, BASE, dict(
        dy=DX, output_plane_distance=1.0e-3)),
    ('multibranch', apply_real_lens_traced_multibranch, BASE, dict(
        output_plane_distance=2.0e-3, output_plane_n=1.0, ray_subsample=2,
        min_area_ratio=1e-6, caustic_band='plain')),
]

#: Cases whose configured settings are all BYTE-IDENTICAL knobs, so "configured
#: differs from all-default" cannot be asserted.  They still carry P1 (the
#: identity) and are listed here so the non-vacuity check below is an explicit,
#: justified exemption rather than a silent skip.  The membership is asserted
#: BOTH ways -- a row listed here that starts moving the field fails too, so
#: the list cannot become a hiding place.
_NO_MOVE_EXPECTED: dict = {}


def _signature_default(fn, name):
    import inspect
    return inspect.signature(fn).parameters[name].default


def _as_field(out):
    return np.asarray(out[0] if isinstance(out, tuple) else out)


def _call(fn, base, **extra):
    with warnings.catch_warnings():
        # The fixture is deliberately non-collimated and deliberately clipped,
        # so several arms emit the library's (correct) diagnostics.  They are
        # not the subject of this file.
        warnings.simplefilter('ignore')
        return _as_field(fn(E_IN, **base, **extra))


@pytest.mark.parametrize('name,fn,base,settings', CASES,
                         ids=[c[0] for c in CASES])
def test_config_call_is_bit_identical_to_the_keyword_call(name, fn, base,
                                                          settings):
    """P1, for one entry point, through all three spellings."""
    ep = fn.__name__
    cfg = LensConfig.from_kwargs(entry_point=ep, **settings)
    # The config must round-trip to exactly the keywords we were given that
    # are REQUESTS.  A setting written at its own signature default is not a
    # request (see the module docstring of ``lens_config``), so it correctly
    # does not come back -- and asserting the requests explicitly is what
    # pins the rename table (multibranch's two) at the same time.
    requests = {k: v for k, v in settings.items()
                if not lc._same(v, _signature_default(fn, k))}
    assert cfg.to_kwargs(entry_point=ep) == requests, (
        f'{ep}: to_kwargs(entry_point=...) gave '
        f'{cfg.to_kwargs(entry_point=ep)}, not the requests {requests} it was '
        f'built from -- the rename table or the defaults disagree.')

    by_kwargs = _call(fn, base, **settings)
    by_config = _call(fn, base, config=cfg)
    by_triple = _call(fn, base, geometry=cfg.geometry, numerics=cfg.numerics,
                      resources=cfg.resources)

    for label, got in (('config=', by_config), ('triple', by_triple)):
        assert got.shape == by_kwargs.shape, (
            f'{ep} [{label}]: shape {got.shape} != keyword call '
            f'{by_kwargs.shape}.')
        assert np.array_equal(got, by_kwargs), (
            f'{ep} [{label}]: the configured call differs from the identical '
            f'keyword call by max |diff| = '
            f'{float(np.max(np.abs(got - by_kwargs))):.6e}.  The merge is '
            f'supposed to produce the SAME call, so any difference is a '
            f'resolver defect, not a tolerance question.')


@pytest.mark.parametrize('name,fn,base,settings', CASES,
                         ids=[c[0] for c in CASES])
def test_the_configured_settings_actually_move_the_field(name, fn, base,
                                                         settings):
    """Non-vacuity for the case above.

    If the resolver silently dropped every request, the identity test would
    still pass (both sides would be the all-default call).  This asserts the
    configured field differs from the all-default one, for every row whose
    settings are not themselves the defaults.
    """
    baseline = _call(fn, base)
    configured = _call(fn, base, **settings)
    moved = (configured.shape != baseline.shape
             or not np.array_equal(configured, baseline))
    if name in _NO_MOVE_EXPECTED:
        assert not moved, (
            f'{name} is listed as a settings-at-their-defaults row but the '
            f'field MOVED.  Either a default changed or the row now carries a '
            f'real setting -- remove it from _NO_MOVE_EXPECTED.')
        return
    assert moved, (
        f'{fn.__name__}: {settings} produced the byte-identical field to the '
        f'all-default call, so the bit-identity test above cannot distinguish '
        f'an honoured config from a discarded one.  Pick settings that move '
        f'this fixture.')


@pytest.mark.parametrize('name,fn,base,settings', CASES,
                         ids=[c[0] for c in CASES])
def test_an_all_default_config_is_the_unconfigured_call(name, fn, base,
                                                        settings):
    """P2.  ``LensConfig()`` must be indistinguishable from passing nothing --
    that is the property that makes every existing call site safe."""
    bare = _call(fn, base)
    via_empty_config = _call(fn, base, config=LensConfig())
    assert np.array_equal(bare, via_empty_config), (
        f'{fn.__name__}: an all-default LensConfig changed the result by max '
        f'|diff| = {float(np.max(np.abs(bare - via_empty_config))):.6e}.  '
        f'A config that requests nothing must take the same path as no config '
        f'at all.')


def test_prepare_real_lens_traced_screen_is_bit_identical():
    """The seventh entry point has no ``E_in``, so it gets its own case: the
    prepared SCREEN (the whole point of the object) must be byte-identical."""
    settings = dict(ray_subsample=1, newton_poly_order=4, bandlimit=True,
                    min_coarse_samples_per_aperture=0)
    cfg = LensConfig.from_kwargs(entry_point='prepare_real_lens_traced',
                                 **settings)
    common = dict(prescription=RX, wavelength=LAM, dx=DX, N=E_IN.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        by_kwargs = prepare_real_lens_traced(**common, **settings)
        by_config = prepare_real_lens_traced(**common, config=cfg)
        by_triple = prepare_real_lens_traced(
            **common, geometry=cfg.geometry, numerics=cfg.numerics,
            resources=cfg.resources)
        different = prepare_real_lens_traced(
            **common, ray_subsample=1, newton_poly_order=8,
            min_coarse_samples_per_aperture=0)
    ref = np.asarray(by_kwargs.screen)
    assert np.array_equal(np.asarray(by_config.screen), ref)
    assert np.array_equal(np.asarray(by_triple.screen), ref)
    # non-vacuity: newton_poly_order really is load-bearing on this fixture
    assert not np.array_equal(np.asarray(different.screen), ref), (
        'newton_poly_order=4 and =8 give the byte-identical prepared screen, '
        'so this test cannot tell an honoured config from a discarded one.')


def test_a_shared_config_reaches_four_engines_through_narrowed_to():
    """The cross-engine use case the objects exist for.

    ONE :class:`LensConfig` describing "evaluate 2 mm past the exit vertex, in
    air, and if you have a caustic model use it" is narrowed per engine and
    must reproduce each engine's own keyword call.  Note how different the four
    narrowings are -- FGA has no ``output_plane_n`` at all, and only the traced
    engine takes the Newton and caustic settings.  Doing this by hand across
    28/29/26/21-parameter signatures is exactly what the audit measured nobody
    doing (68 % of 673 call sites pass zero or one optional keyword).
    """
    cfg = LensConfig.from_kwargs(
        output_plane_distance=2.0e-3, output_plane_n=1.0,
        newton_poly_order=4, amplitude_model='ray_density',
        caustic='multibranch', caustic_ray_subsample=1, caustic_band='plain')
    expected = {
        'apply_real_lens_maslov': dict(output_plane_distance=2.0e-3),
        'apply_real_lens_gbd': dict(output_plane_distance=2.0e-3),
        'apply_real_lens_fga': dict(output_plane_distance=2.0e-3),
        'apply_real_lens_traced': dict(
            output_plane_distance=2.0e-3, newton_poly_order=4,
            amplitude_model='ray_density', caustic='multibranch',
            caustic_ray_subsample=1, caustic_band='plain'),
    }
    for fn, base in ((apply_real_lens_maslov, BASE),
                     (apply_real_lens_gbd, BASE),
                     (apply_real_lens_fga, BASE),
                     (apply_real_lens_traced, TRACED_BASE)):
        ep = fn.__name__
        narrowed = cfg.narrowed_to(ep)
        assert narrowed.to_kwargs(entry_point=ep) == expected[ep], (
            f'{ep}: narrowing the shared config gave '
            f'{narrowed.to_kwargs(entry_point=ep)}, not {expected[ep]}.')
        want = _call(fn, base, **expected[ep])
        got = _call(fn, base, config=narrowed)
        assert np.array_equal(got, want), (
            f'{ep}: the narrowed shared config did not reproduce the keyword '
            f'call (max |diff| = {float(np.max(np.abs(got - want))):.3e}).')


def test_the_unnarrowed_shared_config_refuses_rather_than_dropping():
    """...and the counter-pin: without ``narrowed_to`` the same config RAISES.

    This is the audit's "silently discarded knob" closed: ``newton_poly_order``
    means nothing to the Maslov engine, and the library says so instead of
    quietly ignoring it.
    """
    cfg = LensConfig.from_kwargs(output_plane_distance=1.0e-3,
                                 newton_poly_order=4)
    with pytest.raises(ValueError) as exc:
        _call(apply_real_lens_maslov, BASE, config=cfg)
    msg = str(exc.value)
    assert msg.startswith('apply_real_lens_maslov:'), msg[:200]
    assert 'newton_poly_order' in msg
