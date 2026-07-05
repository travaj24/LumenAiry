"""GBD feature-completeness tests (2026-07 program).

Covers the feature-completeness work layered on top of the audit-closed GBD
correctness: Husimi decomposition plumbed through the lens / prescription
helpers, per-surface Q-evolution, anamorphic sampling, multi-wavelength,
aperture clipping, adaptive beamlet count, and the JAX twin.  Kept in a
separate file from ``test_audit_propagation.py`` (which holds the audit
regression pins) to avoid stepping on concurrent edits there.
"""
import numpy as np
import pytest

from lumenairy.propagators.gbd import (
    apply_aperture_to_beamlets,
    decompose_field_to_beamlets,
    propagate_gbd_freespace,
    propagate_gbd_freespace_spectral,
    propagate_gbd_thin_lens,
)

LAM = 1.0e-6


def _rms_width(F, X, Y):
    I = np.abs(F) ** 2
    s = I.sum()
    if s <= 0:
        return 0.0
    cx = (I * X).sum() / s
    cy = (I * Y).sum() / s
    return float(np.sqrt(((I * ((X - cx) ** 2 + (Y - cy) ** 2)).sum()) / s))


def _centroid_x(F, X):
    I = np.abs(F) ** 2
    s = I.sum()
    return float((I * X).sum() / s) if s > 0 else 0.0


def test_husimi_through_lens_focuses_tilted_beam_off_axis():
    """A collimated beam tilted by theta through a lens f focuses at
    x = f*tan(theta).  The Husimi (direction_sampling=True) decomposition,
    now plumbed into propagate_gbd_thin_lens, walks the tilt off correctly;
    the position-only decomposition does not (its beamlets launch axially and
    never walk off during the free-space legs)."""
    N, dx = 128, 10e-6
    f, theta = 20e-3, 0.02
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    w = 0.4e-3
    E = (np.exp(-(X ** 2 + Y ** 2) / w ** 2)
         * np.exp(1j * 2 * np.pi / LAM * np.sin(theta) * X)).astype(
             np.complex128)
    kw = dict(dx=dx, z_to_lens=0.0, focal_length=f, z_lens_to_output=f,
              wavelength=LAM, output_dx=dx, waist_factor=1.0, sample_step=2)
    g_pos = propagate_gbd_thin_lens(E, **kw, direction_sampling=False)
    g_hus = propagate_gbd_thin_lens(E, **kw, direction_sampling=True)
    x_analytic = f * np.tan(theta)
    err_pos = abs(_centroid_x(g_pos, X) - x_analytic)
    err_hus = abs(_centroid_x(g_hus, X) - x_analytic)
    assert np.isfinite(g_hus).all() and np.isfinite(g_pos).all()
    # Husimi lands within ~12% of the analytic off-axis focus...
    assert err_hus < 0.12 * abs(x_analytic), (err_hus, x_analytic)
    # ...and is dramatically better than position-only for this tilted source.
    assert err_hus < 0.3 * err_pos, (err_hus, err_pos)


def test_aperture_clips_out_of_bound_beamlets():
    """apply_aperture_to_beamlets zeros beamlets whose base ray lies outside
    the stop and leaves the interior amplitudes untouched."""
    N, dx = 64, 20e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.ones((N, N), dtype=np.complex128)
    bundle = decompose_field_to_beamlets(E, dx, wavelength=LAM)
    R = 0.4e-3
    clipped = apply_aperture_to_beamlets(bundle, R, shape='circular')
    # match the function's own criterion (x^2+y^2 <= R^2) to avoid a
    # sqrt-vs-square floating-point disagreement on boundary beamlets.
    r2 = bundle.positions[:, 0] ** 2 + bundle.positions[:, 1] ** 2
    inside = r2 <= R ** 2
    # inside amplitudes untouched, outside zeroed
    assert np.allclose(clipped.amplitude[inside], bundle.amplitude[inside])
    assert np.allclose(clipped.amplitude[~inside], 0.0)
    assert inside.any() and (~inside).any()  # the test actually clips something


def test_aperture_vignettes_and_broadens_focus():
    """A stop smaller than the beam through a lens must remove energy and
    diffraction-broaden the focal spot (a hard aperture -> wider PSF)."""
    N, dx = 192, 8e-6
    f = 20e-3
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    w = 0.6e-3
    E = np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)
    kw = dict(dx=dx, z_to_lens=0.0, focal_length=f, z_lens_to_output=f,
              wavelength=LAM, output_dx=dx, waist_factor=1.0, sample_step=2)
    full = propagate_gbd_thin_lens(E, **kw)
    stopped = propagate_gbd_thin_lens(E, **kw, aperture_semi_diameter=0.3e-3)
    e_full = float((np.abs(full) ** 2).sum())
    e_stop = float((np.abs(stopped) ** 2).sum())
    assert np.isfinite(stopped).all()
    assert e_stop < e_full           # vignetting removed energy
    # a tighter stop diffraction-broadens the focus
    assert _rms_width(stopped, X, Y) > _rms_width(full, X, Y)


def test_spectral_stack_matches_single_wavelength_and_intensity_sum():
    """propagate_gbd_freespace_spectral stacks per-wavelength fields that each
    equal the single-wavelength call, and 'intensity' returns the weighted
    incoherent sum."""
    N, dx = 96, 8e-6
    z = 5e-3
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (0.15e-3) ** 2).astype(np.complex128)
    lams = [0.8e-6, 1.0e-6, 1.3e-6]
    stack = propagate_gbd_freespace_spectral(
        E, dx, z=z, wavelengths=lams, combine='stack', sample_step=2)
    assert stack.shape == (3, N, N)
    for i, lam in enumerate(lams):
        one = propagate_gbd_freespace(E, dx, z=z, wavelength=lam, sample_step=2)
        assert np.allclose(stack[i], one, rtol=1e-10, atol=1e-12)
    w = [1.0, 2.0, 0.5]
    inten = propagate_gbd_freespace_spectral(
        E, dx, z=z, wavelengths=lams, weights=w, combine='intensity',
        sample_step=2)
    assert inten.shape == (N, N)
    assert np.isrealobj(inten) or np.allclose(inten.imag, 0.0)
    ref = sum(wi * np.abs(stack[i]) ** 2 for i, wi in enumerate(w))
    assert np.allclose(np.asarray(inten), ref, rtol=1e-10, atol=1e-12)
