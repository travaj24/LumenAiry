"""Audit v5.17.0 Wave-3 raytrace-parity cluster: P2-34 / P2-35.

P2-34: trace_jax_with_params bypassed _build_jax_prescription (perf) and
therefore missed the v4.10/v4.13.2 unsupported-surface fail-loud guard,
silently tracing mirrors / coord-breaks / biconics / freeforms as flat
refractives.  The guard loop is now hoisted to the module-level helper
_reject_unsupported_jax_surfaces and called from BOTH entry paths with
the caller's name in the message.  The guard inspects only static
prescription-dict fields (never differentiable leaves), so it is
jit/grad trace-safe by construction.

P2-35: the per-surface 'semi_diameter' prescription key was honored by
the JAX backend (jax_trace resolves a finite positive value INSTEAD of
aperture_diameter/2) but silently ignored by the NumPy
surfaces_from_prescription, so the identical prescription vignetted
differently under the two backends (audit probe: 25/25 vs 1/25 alive).
surfaces_from_prescription now resolves ps['semi_diameter'] with the
same replace-semantics and the same validity condition
(None / non-finite / <= 0 -> fall back to the default).

Pre-fix (verified via side-by-side A/B against the pre-edit modules):
test_per_surface_semi_diameter_resolved, test_trace_vignettes_like_jax,
test_semi_diameter_matches_jax_resolution and both with_params guard
tests FAIL; all non-semi_diameter prescriptions remain byte-identical.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.raytrace.jax_trace import _reject_unsupported_jax_surfaces
from lumenairy.raytrace.trace import (
    make_grid,
    surfaces_from_prescription,
    trace,
)

WV = 633e-9


def _flat(**extra):
    s = {'radius': float('inf'), 'glass_before': 'air', 'glass_after': 'air'}
    s.update(extra)
    return s


# ---------------------------------------------------------------------------
# P2-34: unsupported-surface guard (helper level -- runs everywhere)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field,kind", [
    ({'is_mirror': True}, 'is_mirror'),
    ({'glass_after': 'MIRROR'}, 'is_mirror'),
    ({'glass_after': 'mirror'}, 'is_mirror'),          # case-insensitive
    ({'is_coordbrk': True}, 'is_coordbrk'),
    ({'radius_y': 0.1}, 'radius_y (biconic)'),
    ({'conic_y': -1.0}, 'conic_y (biconic)'),
    ({'aspheric_coeffs_y': {4: 1e-3}}, 'aspheric_coeffs_y (biconic)'),
    ({'freeform': {'freeform_type': 'xy_poly'}}, 'freeform'),
])
def test_guard_rejects_each_unsupported_kind(field, kind):
    """Every unsupported surface kind raises NotImplementedError naming the
    calling function, the surface index, and the offending field."""
    with pytest.raises(NotImplementedError) as exc:
        _reject_unsupported_jax_surfaces([_flat(**field)],
                                         'trace_jax_with_params')
    msg = str(exc.value)
    assert 'trace_jax_with_params does not yet support surface 0 with' in msg
    assert kind in msg


def test_guard_message_family_matches_trace_jax():
    """Same message family as the sibling trace_jax builder guard (only the
    function name differs)."""
    def _msg(fn_name):
        with pytest.raises(NotImplementedError) as exc:
            _reject_unsupported_jax_surfaces(
                [_flat(is_mirror=True)], fn_name)
        return str(exc.value)

    m_builder = _msg('trace_jax')
    m_params = _msg('trace_jax_with_params')
    assert m_builder.startswith('trace_jax does not yet support')
    assert m_params.replace('trace_jax_with_params', 'trace_jax',
                            1) == m_builder


def test_guard_reports_correct_surface_index():
    with pytest.raises(NotImplementedError, match='surface 2 with'):
        _reject_unsupported_jax_surfaces(
            [_flat(), _flat(), _flat(is_coordbrk=True)], 'trace_jax')


def test_guard_passes_supported_surfaces():
    """Refractive spherical / conic / aspheric surfaces pass untouched."""
    _reject_unsupported_jax_surfaces([
        _flat(),
        {'radius': 0.05, 'conic': -1.0, 'glass_before': 'air',
         'glass_after': 'N-BK7', 'aspheric_coeffs': {4: 1e2}},
        {'radius': -0.05, 'glass_before': 'N-BK7', 'glass_after': 'air'},
    ], 'trace_jax_with_params')   # must not raise


# ---------------------------------------------------------------------------
# P2-34: entry-point behavior (requires a jax runtime)
# ---------------------------------------------------------------------------

MIRROR_RX = {
    'surfaces': [{'radius': -0.2, 'glass_before': 'air',
                  'glass_after': 'air', 'is_mirror': True}],
    'thicknesses': [0.1],
    'aperture_diameter': 10e-3,
}


def test_trace_jax_with_params_rejects_mirror():
    """The audit's exact failure scenario: pre-fix this returned unreflected
    rays (N=+1.0) with no error; post-fix it fails loud like trace_jax."""
    pytest.importorskip('jax')
    from lumenairy.raytrace.jax_trace import make_jax_ray_state, trace_jax_with_params
    state = make_jax_ray_state(
        np.zeros(3), np.zeros(3), np.zeros(3),
        np.zeros(3), np.zeros(3), np.ones(3))
    with pytest.raises(NotImplementedError,
                       match='trace_jax_with_params does not yet support '
                             'surface 0 with is_mirror'):
        trace_jax_with_params(state, MIRROR_RX, WV)


def test_trace_jax_with_params_parity_on_supported_rx():
    """Guarded entry point still matches trace_jax on a supported singlet
    (guard is a pass-through no-op for valid prescriptions)."""
    pytest.importorskip('jax')
    from lumenairy.raytrace.jax_trace import make_jax_ray_state, trace_jax, trace_jax_with_params
    rx = {
        'surfaces': [
            {'radius': 0.05, 'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'radius': -0.05, 'glass_before': 'N-BK7', 'glass_after': 'air'},
        ],
        'thicknesses': [0.003, 0.095],
        'aperture_diameter': 10e-3,
    }
    y = np.linspace(-3e-3, 3e-3, 7)
    state = make_jax_ray_state(
        np.zeros(7), y, np.zeros(7), np.zeros(7), np.zeros(7), np.ones(7))
    out_a = trace_jax(state, rx, WV)
    out_b = trace_jax_with_params(state, rx, WV)
    np.testing.assert_allclose(np.asarray(out_a.y), np.asarray(out_b.y),
                               rtol=0, atol=1e-12)
    assert np.array_equal(np.asarray(out_a.alive), np.asarray(out_b.alive))


# ---------------------------------------------------------------------------
# P2-35: per-surface semi_diameter honored by the NumPy backend
# ---------------------------------------------------------------------------

def _sd_rx(sd0):
    return {
        'surfaces': [_flat(semi_diameter=sd0), _flat()],
        'thicknesses': [0.01, 0.01],
        'aperture_diameter': 10e-3,
    }


def test_per_surface_semi_diameter_resolved():
    """surfaces_from_prescription resolves ps['semi_diameter'] exactly like
    the JAX builder: (0.001, 0.005) for the audit's prescription."""
    surfs = surfaces_from_prescription(_sd_rx(1e-3))
    assert [s.semi_diameter for s in surfs] == [1e-3, 5e-3]


@pytest.mark.parametrize("bad_sd", [None, float('nan'), float('inf'),
                                    0.0, -2e-3])
def test_invalid_semi_diameter_falls_back_to_default(bad_sd):
    """None / non-finite / <= 0 fall back to aperture_diameter/2 -- the
    identical validity condition the JAX backend applies."""
    surfs = surfaces_from_prescription(_sd_rx(bad_sd))
    assert [s.semi_diameter for s in surfs] == [5e-3, 5e-3]


def test_semi_diameter_replaces_not_mins():
    """JAX-parity REPLACE semantics: a per-surface value LARGER than
    aperture/2 wins (the JAX backend never mins with the default)."""
    surfs = surfaces_from_prescription(_sd_rx(7e-3))
    assert surfs[0].semi_diameter == 7e-3


def test_trace_vignettes_like_jax():
    """The audit's failure scenario end-to-end: sd0=1e-3 must vignette the
    5x5 grid down to the single on-axis ray (JAX kept 1/25; pre-fix NumPy
    kept them all inside the 5 mm default)."""
    surfs = surfaces_from_prescription(_sd_rx(1e-3))
    rays = make_grid(4.5e-3, 5, 0.0, WV, pattern='square')
    res = trace(rays, surfs, WV)
    assert int(res.image_rays.alive.sum()) == 1

    surfs_open = surfaces_from_prescription(_sd_rx(None))
    res_open = trace(make_grid(4.5e-3, 5, 0.0, WV, pattern='square'),
                     surfs_open, WV)
    assert int(res_open.image_rays.alive.sum()) > 1


def test_elements_tightening_still_applies():
    """Non-regression: the pre-existing 'elements' min()-tightening path is
    unchanged for prescriptions without per-surface semi_diameter."""
    rx = {
        'surfaces': [_flat(), _flat()],
        'thicknesses': [0.01, 0.01],
        'aperture_diameter': 10e-3,
        'elements': [
            {'element_type': 'surface', 'semi_diameter': 3e-3},
            {'element_type': 'surface', 'semi_diameter': 4e-3},
        ],
    }
    surfs = surfaces_from_prescription(rx)
    assert [s.semi_diameter for s in surfs] == [3e-3, 4e-3]


def test_semi_diameter_matches_jax_resolution():
    """Field-for-field parity with the JAX resolution rule for a mixed
    prescription (valid, missing, invalid, larger-than-default)."""
    sds = [1e-3, None, -1.0, 7e-3]
    rx = {
        'surfaces': [_flat(semi_diameter=sd) if sd is not None else _flat()
                     for sd in sds],
        'thicknesses': [0.01] * 4,
        'aperture_diameter': 10e-3,
    }

    def jax_rule(s, default):
        sd = s.get('semi_diameter')
        if sd is None or not np.isfinite(sd) or sd <= 0:
            sd = default
        return float(sd)

    expected = [jax_rule(s, 5e-3) for s in rx['surfaces']]
    got = [s.semi_diameter for s in surfaces_from_prescription(rx)]
    assert got == expected


def test_numpy_jax_alive_parity_with_semi_diameter():
    """Cross-backend vignetting parity on the audit prescription (requires
    a jax runtime): both backends must kill the same rays."""
    pytest.importorskip('jax')
    from lumenairy.raytrace.jax_trace import make_jax_ray_state, trace_jax
    rx = _sd_rx(1e-3)
    rays = make_grid(4.5e-3, 5, 0.0, WV, pattern='square')
    res_np = trace(rays, surfaces_from_prescription(rx), WV)
    state = make_jax_ray_state(
        rays.x, rays.y, rays.z, rays.L, rays.M, rays.N)
    res_jx = trace_jax(state, rx, WV)
    assert np.array_equal(np.asarray(res_np.image_rays.alive),
                          np.asarray(res_jx.alive))
