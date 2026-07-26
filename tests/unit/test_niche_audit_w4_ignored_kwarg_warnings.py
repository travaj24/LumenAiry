"""w4 -- two accepted-but-inert kwargs, made audible / made quiet.

Both findings are about a kwarg and a warning disagreeing with reality.

**T2 (flagged in 3f22778) -- ``system.py`` legacy ``'propagate_tilted'``
elements silently ignore ``elem['method']``.**  MEASURED pre-fix on a
tilted 1 mm leg (N=64, dx=4 um, lambda=633 nm): ``method='fresnel'``,
``method='sas'`` and even ``method='not_a_method'`` all produced output
BIT-IDENTICAL to omitting the key (0.0 relative difference) with ZERO
warnings -- so the key was neither honoured nor validated, while the modern
``'propagate'`` element raises ``ValueError`` on the same junk.  The handler is
structurally ASM-only (it calls ``angular_spectrum_propagate_tilted``
directly; there is no tilted Fresnel/SAS kernel), so the fix is to say so:
a ``UserWarning`` naming the limitation.  Raising was rejected -- it would be
a new breakage class for a legacy alias that has accepted the key for many
releases.  The physics is deliberately unchanged, which is why the
bit-identity assertions below are *kept* rather than inverted.

**T3 (flagged in bba1bc4) -- ``lenses_gbd`` tripped the callee's S2-4
``z_image`` ``RuntimeWarning`` on EVERY paraxial call.**
``apply_real_lens_gbd`` forwarded ``z_image=float(output_plane_distance)``,
and ``output_plane_distance`` defaults to ``0.0``, which is ``not None`` --
the exact condition the callee's guard tests.  MEASURED pre-fix: 1 such
RuntimeWarning on a bare ``per_surface=False`` call that passed no distance
at all.  Fixed by mapping ``0.0 -> None`` (behaviour-free: the paraxial path
ignores ``z_image`` either way, pinned below by bit-identity), so the warning
survives only where a distance really was requested and really is dropped.

Pre-fix (worktree at 865e922): every ``test_t2_warns_*`` fails (no warning
existed) and ``test_t3_default_paraxial_call_is_silent`` fails (the warning
fired).  The counter-pins pass in both states by construction.

Author: audit wave 4
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.propagators.system import propagate_through_system

LAM = 632.8e-9


def _squash(text):
    return ' '.join((text or '').split())


# ======================================================================
# T2 -- 'propagate_tilted' + method
# ======================================================================

def _tilt_field():
    N, dx = 64, 4e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-((X ** 2 + Y ** 2) / (20e-6) ** 2)).astype(np.complex128), dx


Z_LEG = 1e-3


def _run_system(elements):
    E, dx = _tilt_field()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out, _inter = propagate_through_system(
            E, elements, wavelength=LAM, dx=dx)
    return np.asarray(out), rec


def _method_warnings(rec):
    return [r for r in rec
            if issubclass(r.category, UserWarning)
            and 'propagate_tilted' in str(r.message)
            and 'IGNORED' in str(r.message)]


class TestT2LegacyTiltedElementMethodKey:

    @pytest.mark.parametrize('method',
                             ['fresnel', 'sas', 'asm', 'not_a_method'])
    def test_t2_warns_for_any_method_value(self, method):
        """Every value warns -- including ``'asm'`` (which happens to match
        what runs) and junk (which the modern element rejects).  The warning
        is about the key being inert here, not about its value."""
        _out, rec = _run_system([
            {'type': 'propagate_tilted', 'z': Z_LEG, 'tilt_x': 1e-3,
             'method': method},
        ])
        hits = _method_warnings(rec)
        assert len(hits) == 1, (
            f'expected one propagate_tilted/method UserWarning for '
            f'method={method!r}, got {[str(r.message)[:70] for r in rec]}')
        assert repr(method) in str(hits[0].message)

    def test_t2_no_method_key_is_silent(self):
        """Counter-pin: the ordinary legacy call must stay quiet."""
        _out, rec = _run_system([
            {'type': 'propagate_tilted', 'z': Z_LEG, 'tilt_x': 1e-3},
        ])
        assert not _method_warnings(rec)
        assert not [r for r in rec if issubclass(r.category, UserWarning)], (
            f'unexpected UserWarning on a plain propagate_tilted element: '
            f'{[str(r.message)[:70] for r in rec]}')

    def test_t2_message_names_the_limitation_and_the_fix(self):
        _out, rec = _run_system([
            {'type': 'propagate_tilted', 'z': Z_LEG, 'tilt_x': 1e-3,
             'method': 'fresnel'},
        ])
        msg = _squash(str(_method_warnings(rec)[0].message))
        assert 'IGNORED' in msg
        assert 'ASM-only' in msg
        assert 'angular_spectrum_propagate_tilted' in msg
        assert "'type': 'propagate'" in msg, (
            'the message must name the unified element as the migration')
        assert 'tilt_x' in msg and 'tilt_y' in msg
        assert 'silence' in msg

    def test_t2_message_names_the_offending_element_index(self):
        """The chain can be long; the message must point at the element the
        caller wrote, not just at the type."""
        _out, rec = _run_system([
            {'type': 'propagate', 'z': Z_LEG},
            {'type': 'propagate_tilted', 'z': Z_LEG, 'tilt_x': 1e-3,
             'method': 'sas'},
        ])
        msg = str(_method_warnings(rec)[0].message)
        assert 'elements[1]' in msg

    @pytest.mark.parametrize('method', ['fresnel', 'sas', 'not_a_method'])
    def test_t2_physics_is_deliberately_unchanged(self, method):
        """The fix adds a diagnostic, not a behaviour change: the field must
        still be bit-identical to the no-``method`` call.  This is the
        independent probe of the "silently ignored" mechanism -- it is what
        made the finding real, and it must stay true after the warning
        lands."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            base, _ = _run_system([
                {'type': 'propagate_tilted', 'z': Z_LEG, 'tilt_x': 1e-3},
            ])
            got, _ = _run_system([
                {'type': 'propagate_tilted', 'z': Z_LEG, 'tilt_x': 1e-3,
                 'method': method},
            ])
        assert np.array_equal(base, got), (
            f"method={method!r} changed the propagate_tilted output; the "
            f"element is ASM-only and the fix must not alter physics")

    def test_t2_counter_pin_modern_element_still_raises_on_junk(self):
        """Guard that warning-not-raising here did not weaken the v5.30
        validation on the unified ``'propagate'`` element."""
        E, dx = _tilt_field()
        with pytest.raises(ValueError, match='not a recognised free-space'):
            propagate_through_system(
                E, [{'type': 'propagate', 'z': Z_LEG, 'tilt_x': 1e-3,
                     'method': 'not_a_method'}],
                wavelength=LAM, dx=dx)

    def test_t2_limitation_is_documented(self):
        doc = _squash(propagate_through_system.__doc__)
        assert 'ASM-only and ignores ``method``' in doc
        assert 'UserWarning' in doc
        assert 'neither honoured nor validated' in doc


# ======================================================================
# T3 -- lenses_gbd default paraxial call vs the S2-4 z_image warning
# ======================================================================

def _gbd_presc():
    return {'surfaces': [
        {'radius': 25e-3, 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': 'air', 'glass_after': 'N-BK7'},
        {'radius': np.inf, 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': 'N-BK7', 'glass_after': 'air'}],
        'thicknesses': [3e-3], 'aperture_diameter': 4e-3}


def _gbd_field():
    N, dx = 64, 100e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-((X ** 2 + Y ** 2) / (1.2e-3) ** 2)).astype(np.complex128), dx


_GBD_KW = dict(sample_step=8, beamlets_per_aperture=8)


def _run_gbd(**kw):
    E, dx = _gbd_field()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = la.apply_real_lens_gbd(
            E, prescription=_gbd_presc(), wavelength=LAM, dx=dx,
            **_GBD_KW, **kw)
    zi = [r for r in rec if 'z_image is only honored' in str(r.message)]
    return np.asarray(out), zi


class TestT3ZImageWarningOnlyWhereItApplies:

    def test_t3_default_paraxial_call_is_silent(self):
        """THE bug: a caller who passed no ``output_plane_distance`` at all
        got a warning about ``z_image`` being dropped."""
        _out, zi = _run_gbd(per_surface=False)
        assert not zi, (
            f'the S2-4 z_image RuntimeWarning fired on a default '
            f'per_surface=False call: {[str(r.message)[:80] for r in zi]}')

    def test_t3_explicit_zero_distance_is_also_silent(self):
        """``0.0`` means "no extra leg" whether defaulted or written out."""
        _out, zi = _run_gbd(per_surface=False, output_plane_distance=0.0)
        assert not zi

    def test_t3_default_per_surface_call_is_silent(self):
        _out, zi = _run_gbd()
        assert not zi

    def test_t3_real_distance_on_the_paraxial_path_still_warns(self):
        """Counter-pin -- the load-bearing half.  A genuinely requested
        distance IS silently dropped by the paraxial path, so that warning
        must survive; over-quieting it would hide real data loss."""
        _out, zi = _run_gbd(per_surface=False, output_plane_distance=1e-3)
        assert len(zi) == 1, (
            'a non-zero output_plane_distance on the paraxial path must '
            'still raise the S2-4 RuntimeWarning')
        assert issubclass(zi[0].category, RuntimeWarning)

    def test_t3_pytest_warns_contract_for_the_real_distance(self):
        E, dx = _gbd_field()
        with pytest.warns(RuntimeWarning, match='z_image is only honored'):
            la.apply_real_lens_gbd(
                E, prescription=_gbd_presc(), wavelength=LAM, dx=dx,
                per_surface=False, output_plane_distance=1e-3, **_GBD_KW)

    def test_t3_zero_to_none_mapping_is_behaviour_free(self):
        """Why the fix is safe: on the paraxial path ``z_image`` is dropped,
        so ``0.0`` and a real distance already produce the SAME field.  Both
        must therefore also equal what ``None`` produces -- i.e. mapping
        ``0.0 -> None`` cannot have moved the output plane."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a, _ = _run_gbd(per_surface=False)
            b, _ = _run_gbd(per_surface=False, output_plane_distance=1e-3)
        assert np.array_equal(a, b), (
            'the paraxial path is supposed to ignore z_image entirely; if '
            'this differs, the 0.0 -> None mapping is NOT behaviour-free')

    def test_t3_per_surface_path_still_consumes_the_distance(self):
        """Guard the other direction: the fix must not have mapped the
        distance to ``None`` on the branch that actually honours it (there
        ``None`` means "use the system BFL", a different plane entirely)."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a, _ = _run_gbd(per_surface=True)
            b, _ = _run_gbd(per_surface=True, output_plane_distance=1e-3)
        assert not np.array_equal(a, b), (
            'per_surface=True must land on a different plane when a real '
            'output_plane_distance is requested')
