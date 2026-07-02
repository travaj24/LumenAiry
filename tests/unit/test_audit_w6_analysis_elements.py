"""Wave-6 audit fixes -- analysis-elements cluster (v5.17 deep audit).

Discriminating tests for the CODE fixes:

* P3-01 make_shack_hartmann_wfs: seeded RNG hoisted to the factory --
  reproducible noise SEQUENCE, not a frozen per-frame realisation.
* P3-03 through_focus_scan_jax: host copy streamed one plane at a time
  (no second full (n_z, Ny, Nx) host stack).
* P3-04 _lens_jax: duplicate _jax_available definition removed.
* P3-15 GaussianBSDF: incidence-aware normalization -- hemisphere
  integral of evaluate()*cos(theta_s) equals scattered_fraction at
  oblique incidence too.
* P3-24 rytov_tensor / rytov_segments_tensor: validity-domain
  UserWarning outside the homogenization regime.
* P3-39 segment_geometry.line_interface: UserWarning when the
  carved-side band is thinner than the liner thickness (previously a
  silent no-op).

Doc-only fixes (P3-02, P3-07, P3-25) need no test.
"""
import ast
import inspect
import warnings

import numpy as np
import pytest

# --------------------------------------------------------------------------
# P3-01
# --------------------------------------------------------------------------


def _disk_phase(N=64):
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    mask = (X ** 2 + Y ** 2) <= 1.0
    return 0.05 * X * mask


def test_p3_01_seeded_noise_is_fresh_per_frame():
    """A fixed rng_seed must give DIFFERENT noise on successive frames
    (reproducible sequence), not the identical frozen realisation."""
    from lumenairy.analysis.ao import make_shack_hartmann_wfs
    phase = _disk_phase()
    kw = dict(subaperture_grid=8, noise_sigma_pixels=1.0, rng_seed=7,
              n_modes=15)
    wfs = make_shack_hartmann_wfs(**kw)
    a, b = wfs(phase), wfs(phase)
    assert not np.array_equal(a, b), (
        "seeded closure returned the IDENTICAL noise realisation on "
        "two frames -- P3-01 frozen-noise regression")


def test_p3_01_seeded_noise_sequence_reproducible():
    """Two factories built with the same seed reproduce the same
    per-call noise sequence."""
    from lumenairy.analysis.ao import make_shack_hartmann_wfs
    phase = _disk_phase()
    kw = dict(subaperture_grid=8, noise_sigma_pixels=1.0, rng_seed=7,
              n_modes=15)
    w1 = make_shack_hartmann_wfs(**kw)
    w2 = make_shack_hartmann_wfs(**kw)
    a1, b1 = w1(phase), w1(phase)
    a2, b2 = w2(phase), w2(phase)
    assert np.array_equal(a1, a2)
    assert np.array_equal(b1, b2)


# --------------------------------------------------------------------------
# P3-03
# --------------------------------------------------------------------------


def test_p3_03_jax_scan_host_copy_is_streamed():
    """The JAX scan must NOT materialise a second full host stack via
    np.asarray(fields); the per-plane copy np.asarray(fields[i]) is the
    streamed contract."""
    from lumenairy.analysis import through_focus
    src = inspect.getsource(through_focus.through_focus_scan_jax)
    assert 'np.asarray(fields[i])' in src, (
        "per-plane streamed host copy missing")
    assert 'fields_np = np.asarray(fields)' not in src, (
        "full-stack host copy re-introduced -- P3-03 regression")


def test_p3_03_jax_scan_still_correct():
    """Streaming change is metric-transparent: JAX scan matches the
    NumPy backend on a small scan (loose tol: jax may run f32)."""
    pytest.importorskip('jax')
    from lumenairy.analysis.through_focus import (
        through_focus_scan,
        through_focus_scan_jax,
    )
    N = 64
    x = (np.arange(N) - N / 2) * 5e-6
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X ** 2 + Y ** 2) / (60e-6) ** 2)
    z = np.linspace(0.01, 0.03, 5)
    r_np = through_focus_scan(E, 5e-6, 633e-9, z)
    r_jx = through_focus_scan_jax(E, 5e-6, 633e-9, z)
    assert np.all(np.isfinite(r_jx.peak_I))
    # rtol accommodates JAX-without-x64 (float32 ASM kernel).
    np.testing.assert_allclose(r_jx.peak_I, r_np.peak_I, rtol=3e-2)


# --------------------------------------------------------------------------
# P3-04
# --------------------------------------------------------------------------


def test_p3_04_single_jax_available_definition():
    import lumenairy.elements._lens_jax as lj
    tree = ast.parse(open(lj.__file__, encoding='utf-8').read())
    defs = [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and n.name == '_jax_available']
    assert len(defs) == 1, (
        f"expected exactly one _jax_available definition, got lines "
        f"{defs} -- P3-04 duplicate-shadow regression")


# --------------------------------------------------------------------------
# P3-15
# --------------------------------------------------------------------------


def _hemi_tis(g, theta_i, nth=512, nph=256):
    th = (np.arange(nth) + 0.5) * (np.pi / 2) / nth
    ph = (np.arange(nph) + 0.5) * (2 * np.pi) / nph
    TH, PH = np.meshgrid(th, ph, indexing='ij')
    sd = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH),
                   np.cos(TH)], axis=-1).reshape(-1, 3)
    inc = np.array([np.sin(theta_i), 0.0, -np.cos(theta_i)])
    val = g.evaluate(np.broadcast_to(inc, sd.shape), sd).reshape(nth, nph)
    w = (np.sin(TH) * np.cos(TH))
    return float(np.sum(val * w) * (np.pi / 2 / nth) * (2 * np.pi / nph))


@pytest.mark.parametrize('deg', [0.0, 45.0, 60.0])
def test_p3_15_oblique_tis_matches_scattered_fraction(deg):
    """Hemisphere integral of evaluate()*cos(theta_s) equals
    scattered_fraction (== total_integrated_scatter()) at ALL
    incidence angles, not just normal."""
    from lumenairy.elements.bsdf import GaussianBSDF
    g = GaussianBSDF(sigma_rad=0.02, scattered_fraction=0.01)
    tis = _hemi_tis(g, np.deg2rad(deg))
    assert tis == pytest.approx(g.total_integrated_scatter(), rel=0.02), (
        f"TIS at {deg} deg = {tis} != scattered_fraction -- P3-15 "
        f"cos(theta_i) inconsistency")


def test_p3_15_scalar_incident_dir_still_works():
    from lumenairy.elements.bsdf import GaussianBSDF
    g = GaussianBSDF(sigma_rad=0.02, scattered_fraction=0.01)
    v = g.evaluate(np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0]))
    assert np.isfinite(float(v)) and float(v) > 0


# --------------------------------------------------------------------------
# P3-24
# --------------------------------------------------------------------------


def test_p3_24_order2_beyond_validity_warns():
    from lumenairy.elements.emt import rytov_tensor
    with pytest.warns(UserWarning, match='period/wavelength'):
        rytov_tensor(12 + 0.1j, 1.0, 0.5, period=800e-9,
                     wavelength=633e-9, order=2)


def test_p3_24_order2_deep_subwavelength_silent():
    from lumenairy.elements.emt import rytov_tensor
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        rytov_tensor(4.0, 1.0, 0.5, period=50e-9, wavelength=633e-9,
                     order=2)


def test_p3_24_order0_diffractive_period_warns():
    from lumenairy.elements.emt import rytov_tensor
    with pytest.warns(UserWarning, match='diffraction'):
        rytov_tensor(4.0, 1.0, 0.5, period=5e-6, wavelength=633e-9,
                     order=0)


def test_p3_24_order0_no_period_silent():
    from lumenairy.elements.emt import rytov_tensor
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        rytov_tensor(4.0, 1.0, 0.5, order=0)


def test_p3_24_segments_diffractive_period_warns():
    from lumenairy.elements.emt import rytov_segments_tensor
    segs = [(0.5, 4.0), (0.5, 1.0)]
    with pytest.warns(UserWarning, match='diffraction'):
        rytov_segments_tensor(segs, period=5e-6, wavelength=633e-9)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        rytov_segments_tensor(segs)  # no period -> silent


# --------------------------------------------------------------------------
# P3-39
# --------------------------------------------------------------------------


def _mats(g):
    return {m for _t, ivs in g._bands for _l, _h, m in ivs}


def test_p3_39_thin_carved_band_warns():
    from lumenairy.elements.segment_geometry import SegmentStackGeometry
    g = SegmentStackGeometry(period=500e-9)
    g.add_band(0.8e-9, [(1.0, 'Cu')])
    g.add_band(10e-9, [(1.0, 'SiO2')])
    with pytest.warns(UserWarning, match='thinner than the liner'):
        g.line_interface('Cu', 'SiO2', t=1e-9, mat='Ta', side='a')
    # Liner still (correctly) omitted -- warn-not-raise contract.
    assert 'Ta' not in _mats(g)


def test_p3_39_thick_carved_band_no_warning_liner_inserted():
    from lumenairy.elements.segment_geometry import SegmentStackGeometry
    g = SegmentStackGeometry(period=500e-9)
    g.add_band(5e-9, [(1.0, 'Cu')])
    g.add_band(10e-9, [(1.0, 'SiO2')])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        g.line_interface('Cu', 'SiO2', t=1e-9, mat='Ta', side='a')
    assert 'Ta' in _mats(g)
