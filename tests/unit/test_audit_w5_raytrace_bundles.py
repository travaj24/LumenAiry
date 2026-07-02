"""Audit v5.17.0 Wave-5 raytrace-bundles cluster: P2-33 + P2-35 residual.

P2-33: ``ray_to_beamlet`` silently discarded ``RayBundle.opd`` (and
``alive``): ``BeamletBundle`` has no opl/alive fields -- its complex
``amplitude`` is the ONLY phase/weight carrier that
``reconstruct_field_from_beamlets`` sums coherently -- yet the default
amplitude was all-ones.  Every beamlet got piston phase 0 (a focused
beam's spherical wavefront reconstructed flat) and dead/TIR rays
contributed full amplitude.  The default now folds both in as
``exp(+1j * k0 * opd) * alive``, the sign pinned by the library's
``exp(-i omega t)`` / ``exp(+i k z)`` convention
(gbd.propagate_beamlets_freespace accumulates ``exp(+1j k t)``) and by
``rays_from_field`` seeding ``opd = angle(E)/k0`` (which the new
default inverts exactly).  An explicitly passed ``amplitude`` stays
verbatim (the pre-fix escape hatch).

P2-35 residual: wave 3 made the NumPy backend honour the per-surface
'semi_diameter' key, but the JAX builders still never consulted
``prescription['elements']`` (where Zemax-loaded apertures live), so
the identical prescription vignetted under NumPy ``trace`` but not
under ``trace_jax`` / ``trace_jax_with_params``.  Both JAX entry
points now resolve apertures through the shared
``_resolve_semi_diameters`` helper mirroring
``surfaces_from_prescription``: per-surface key REPLACES the
aperture_diameter/2 default, then an 'elements' entry TIGHTENS via
min().

Pre-fix: the P2-33 amplitude/interference/dead-ray tests and every
elements-side parity test below FAIL (probe: two co-located rays with
opd=[0, lambda/2] reconstructed |E|=1.99 fully constructive instead of
~0; elements rx resolved numpy [0.001, 0.001] vs jax (0.005, 0.005)).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import types

import numpy as np
import pytest

from lumenairy.raytrace.bundles import ray_to_beamlet, ray_to_path
from lumenairy.raytrace.core import RayBundle
from lumenairy.raytrace.jax_trace import _resolve_semi_diameters
from lumenairy.raytrace.trace import (
    make_grid,
    surfaces_from_prescription,
    trace,
)

WV = 633e-9
K0 = 2.0 * np.pi / WV


# ---------------------------------------------------------------------------
# P2-33: ray_to_beamlet carries opd + alive into the beamlet amplitude
# ---------------------------------------------------------------------------

def _rb(opd, alive=None, x=None):
    n = len(opd)
    if alive is None:
        alive = np.ones(n, dtype=bool)
    if x is None:
        x = np.zeros(n)
    return RayBundle(
        x=np.asarray(x, dtype=float), y=np.zeros(n), z=np.zeros(n),
        L=np.zeros(n), M=np.zeros(n), N=np.ones(n),
        wavelength=WV,
        alive=np.asarray(alive, dtype=bool),
        opd=np.asarray(opd, dtype=float),
    )


def test_default_amplitude_is_exp_ik_opd_times_alive():
    """The default amplitude folds the piston phase exp(+1j*k0*opd) and
    the alive mask -- the exact quantity coherent recombination
    interferes on."""
    opd = np.array([0.0, WV / 4.0, WV / 2.0, 1.5e-3])
    alive = np.array([True, True, True, False])
    bb = ray_to_beamlet(_rb(opd, alive), wavelength=WV, waist0=50e-6)
    np.testing.assert_allclose(
        np.asarray(bb.amplitude), np.exp(1j * K0 * opd) * alive,
        rtol=0, atol=1e-12)


def test_opd_sign_inverts_rays_from_field_seed():
    """rays_from_field seeds opd = angle(E)/k0; the default amplitude
    must restore exactly that wave-optical phase (sign pinned by the
    library's exp(+ikz) convention)."""
    phi = np.array([-2.5, -0.5, 0.0, 1.0, 3.0])
    bb = ray_to_beamlet(_rb(phi / K0), wavelength=WV, waist0=50e-6)
    np.testing.assert_allclose(np.angle(np.asarray(bb.amplitude)), phi,
                               rtol=0, atol=1e-12)


def test_half_wave_piston_destructive_interference():
    """Audit probe end-to-end: two co-located beamlets with a lambda/2
    path difference must cancel (pre-fix |E| = 1.99, fully
    constructive)."""
    from lumenairy.propagators.gbd import reconstruct_field_from_beamlets
    bb = ray_to_beamlet(_rb([0.0, WV / 2.0]), wavelength=WV, waist0=50e-6)
    E = reconstruct_field_from_beamlets(bb, Ny=9, Nx=9, dx=5e-6,
                                        wavelength=WV)
    assert abs(E[4, 4]) < 1e-12

    bb0 = ray_to_beamlet(_rb([0.0, 0.0]), wavelength=WV, waist0=50e-6)
    E0 = reconstruct_field_from_beamlets(bb0, Ny=9, Nx=9, dx=5e-6,
                                         wavelength=WV)
    assert abs(E0[4, 4]) > 1.9   # in-phase pair stays constructive


def test_dead_ray_contributes_nothing():
    """Dead/TIR rays must not radiate (pre-fix they contributed full
    amplitude 1)."""
    from lumenairy.propagators.gbd import reconstruct_field_from_beamlets
    bb = ray_to_beamlet(_rb([0.0], alive=[False]),
                        wavelength=WV, waist0=50e-6)
    assert np.all(np.asarray(bb.amplitude) == 0.0)
    E = reconstruct_field_from_beamlets(bb, Ny=9, Nx=9, dx=5e-6,
                                        wavelength=WV)
    assert np.all(np.abs(E) == 0.0)


def test_explicit_amplitude_used_verbatim():
    """The escape hatch is preserved: an explicitly passed amplitude is
    NOT re-multiplied by exp(ik*opd) or alive (callers who already fold
    the phase themselves would otherwise double-count)."""
    amp = np.array([0.5 + 0.0j, 2.0 - 1.0j])
    bb = ray_to_beamlet(_rb([0.0, WV / 2.0], alive=[True, False]),
                        wavelength=WV, waist0=50e-6, amplitude=amp)
    np.testing.assert_array_equal(np.asarray(bb.amplitude), amp)


def test_zero_opd_all_alive_matches_old_default():
    """Non-regression: opd == 0 with all rays alive reproduces the old
    all-ones default exactly."""
    bb = ray_to_beamlet(_rb([0.0, 0.0, 0.0]), wavelength=WV, waist0=50e-6)
    np.testing.assert_array_equal(np.asarray(bb.amplitude),
                                  np.ones(3, dtype=np.complex128))


def test_schema_less_bundle_falls_back_to_ones():
    """Duck-typed bundles without opd/alive keep the old unit default --
    the same getattr fallbacks ray_to_path uses."""
    n = 3
    duck = types.SimpleNamespace(
        x=np.zeros(n), y=np.zeros(n), z=np.zeros(n),
        L=np.zeros(n), M=np.zeros(n), N=np.ones(n))
    bb = ray_to_beamlet(duck, wavelength=WV, waist0=50e-6)
    np.testing.assert_array_equal(np.asarray(bb.amplitude),
                                  np.ones(n, dtype=np.complex128))
    # sanity: same fallback contract as ray_to_path
    pb = ray_to_path(duck)
    np.testing.assert_array_equal(np.asarray(pb.opl), np.zeros(n))


def test_geometry_and_q_unchanged():
    """Non-regression: positions / directions / Q / waist0 are untouched
    by the amplitude fix."""
    rb = _rb([0.0, 1e-3], x=[-1e-3, 1e-3])
    bb = ray_to_beamlet(rb, wavelength=WV, waist0=1e-3)
    assert np.asarray(bb.positions).shape == (2, 3)
    np.testing.assert_allclose(np.asarray(bb.positions)[:, 0], rb.x)
    z_R = np.pi * (1e-3 ** 2) / WV
    np.testing.assert_allclose(np.asarray(bb.Q), -1j / z_R)
    np.testing.assert_allclose(np.asarray(bb.waist0), 1e-3)


# ---------------------------------------------------------------------------
# P2-35 residual: JAX builders resolve apertures like the NumPy backend
# ---------------------------------------------------------------------------

def _flat(**extra):
    s = {'radius': float('inf'), 'glass_before': 'air', 'glass_after': 'air'}
    s.update(extra)
    return s


def _elements_rx():
    """Zemax-loader layout: apertures live in 'elements', not on the
    per-surface dicts."""
    return {
        'surfaces': [_flat(), _flat()],
        'thicknesses': [0.01, 0.01],
        'aperture_diameter': 10e-3,
        'elements': [
            {'element_type': 'surface', 'semi_diameter': 1e-3},
            {'element_type': 'surface', 'semi_diameter': 4e-3},
        ],
    }


def _numpy_sds(rx):
    return [s.semi_diameter for s in surfaces_from_prescription(rx)]


def test_resolver_matches_numpy_elements_only():
    """Resolver-level parity (no jax runtime needed): elements-based
    apertures resolve identically (pre-fix jax: (0.005, 0.005))."""
    rx = _elements_rx()
    assert _resolve_semi_diameters(rx) == _numpy_sds(rx) == [1e-3, 4e-3]


def test_resolver_precedence_per_surface_then_elements_min():
    """Precedence parity: the per-surface key REPLACES the default, then
    a tighter 'elements' entry mins it down -- both backends, same
    answer."""
    rx = {
        'surfaces': [_flat(semi_diameter=2e-3), _flat(semi_diameter=1e-3)],
        'thicknesses': [0.01, 0.01],
        'aperture_diameter': 10e-3,
        'elements': [
            {'element_type': 'surface', 'semi_diameter': 1e-3},   # tightens
            {'element_type': 'surface', 'semi_diameter': 3e-3},   # looser: no-op
        ],
    }
    assert _resolve_semi_diameters(rx) == _numpy_sds(rx) == [1e-3, 1e-3]


def test_resolver_matches_numpy_invalid_and_missing():
    """Validity-condition parity: None / non-finite / <= 0 per-surface
    values and non-'surface' / missing elements entries behave
    identically."""
    rx = {
        'surfaces': [_flat(semi_diameter=-1.0), _flat(),
                     _flat(semi_diameter=float('nan'))],
        'thicknesses': [0.01, 0.01, 0.01],
        'aperture_diameter': 10e-3,
        'elements': [
            {'element_type': 'gap'},                       # skipped
            {'element_type': 'surface'},                   # no sd -> no-op
            {'element_type': 'surface', 'semi_diameter': 2e-3},
        ],
    }
    assert _resolve_semi_diameters(rx) == _numpy_sds(rx) == [5e-3, 2e-3, 5e-3]


def test_resolver_no_aperture_keys_stays_open():
    """Non-regression: without aperture_diameter / semi_diameter /
    elements every surface stays unclipped (inf)."""
    rx = {'surfaces': [_flat(), _flat()], 'thicknesses': [0.01, 0.01]}
    got = _resolve_semi_diameters(rx)
    assert got == [float('inf'), float('inf')]


def test_build_jax_prescription_honours_elements():
    """Builder-level parity (jax runtime): _build_jax_prescription's
    static semi-diameter aux matches the NumPy Surface list."""
    pytest.importorskip('jax')
    from lumenairy.raytrace.jax_trace import _build_jax_prescription
    rx = _elements_rx()
    jp = _build_jax_prescription(rx, WV)
    assert list(jp.aux[2]) == _numpy_sds(rx)


def test_trace_jax_alive_parity_elements_rx():
    """End-to-end cross-backend vignetting parity for an elements-based
    prescription: both backends must kill the same rays (pre-fix JAX
    kept all 25 alive vs NumPy vignetting to the 1 mm aperture)."""
    pytest.importorskip('jax')
    from lumenairy.raytrace.jax_trace import make_jax_ray_state, trace_jax
    rx = _elements_rx()
    rays = make_grid(4.5e-3, 5, 0.0, WV, pattern='square')
    res_np = trace(rays, surfaces_from_prescription(rx), WV)
    state = make_jax_ray_state(rays.x, rays.y, rays.z,
                               rays.L, rays.M, rays.N)
    res_jx = trace_jax(state, rx, WV)
    assert int(np.asarray(res_np.image_rays.alive).sum()) < 25
    assert np.array_equal(np.asarray(res_np.image_rays.alive),
                          np.asarray(res_jx.alive))


def test_trace_jax_with_params_alive_parity_elements_rx():
    """Same parity through the with_params entry point (which bypasses
    _build_jax_prescription and resolves apertures inline)."""
    pytest.importorskip('jax')
    from lumenairy.raytrace.jax_trace import (
        make_jax_ray_state,
        trace_jax_with_params,
    )
    rx = _elements_rx()
    rays = make_grid(4.5e-3, 5, 0.0, WV, pattern='square')
    res_np = trace(rays, surfaces_from_prescription(rx), WV)
    state = make_jax_ray_state(rays.x, rays.y, rays.z,
                               rays.L, rays.M, rays.N)
    res_jx = trace_jax_with_params(state, rx, WV)
    assert np.array_equal(np.asarray(res_np.image_rays.alive),
                          np.asarray(res_jx.alive))


def test_per_surface_sd_parity_unchanged():
    """Non-regression on the wave-3 direction: per-surface semi_diameter
    (no elements) still resolves identically in both backends."""
    rx = {
        'surfaces': [_flat(semi_diameter=1e-3), _flat()],
        'thicknesses': [0.01, 0.01],
        'aperture_diameter': 10e-3,
    }
    assert _resolve_semi_diameters(rx) == _numpy_sds(rx) == [1e-3, 5e-3]
