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
    recommend_gbd_sampling,
    reconstruct_field_from_beamlets,
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


def test_recommend_sampling_scales_with_field_structure_and_roundtrips():
    """recommend_gbd_sampling picks a coarser grid for a smooth field than for
    a finely-structured one, sets overlapping waists, and the recommended
    sampling reconstructs the source (z=0 round-trip) to a few percent."""
    N, dx = 128, 8e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    smooth = np.exp(-(X ** 2 + Y ** 2) / (0.25e-3) ** 2).astype(np.complex128)
    fine = (smooth * np.exp(1j * 2 * np.pi / LAM * np.sin(0.05) * X)).astype(
        np.complex128)  # steep tilt -> fast phase gradient
    rs = recommend_gbd_sampling(smooth, dx, wavelength=LAM)
    rf = recommend_gbd_sampling(fine, dx, wavelength=LAM)
    assert rs['sample_step'] >= 1 and rf['sample_step'] >= 1
    # the steeply-tilted (finer) field must not be sampled MORE coarsely
    assert rf['sample_step'] <= rs['sample_step']
    # overlapping waists (w0 > spacing)
    assert rs['waist_factor'] > rs['sample_step'] * 0.9
    # z=0 round-trip with the recommendation reproduces the smooth source
    kw = {k: rs[k] for k in ('sample_step', 'waist_factor')}
    b = decompose_field_to_beamlets(smooth, dx, wavelength=LAM, **kw)
    recon = reconstruct_field_from_beamlets(
        b, Ny=N, Nx=N, dx=dx, wavelength=LAM)
    m = np.abs(smooth) > 0.05 * np.abs(smooth).max()
    rel = np.linalg.norm((np.abs(recon) - np.abs(smooth))[m]) / \
        np.linalg.norm(np.abs(smooth)[m])
    assert rel < 0.1, rel


# --------------------------------------------------------------------------
# Per-surface tensor-Q GBD + the differential-ray-transfer primitive
# --------------------------------------------------------------------------
def _singlet(last=0.0):
    return {'name': 's', 'aperture_diameter': 12e-3,
            'surfaces': [
                {'radius': 51.5e-3, 'conic': 0., 'glass_before': 'air',
                 'glass_after': 'N-BK7', 'semi_diameter': 6e-3},
                {'radius': -51.5e-3, 'conic': 0., 'glass_before': 'N-BK7',
                 'glass_after': 'air', 'semi_diameter': 6e-3}],
            'thicknesses': [4e-3, last]}


def test_ray_transfer_jacobian_matches_paraxial_abcd():
    """The differential-ray-transfer primitive's on-axis 2x2 meridional block
    reproduces the analytic paraxial system ABCD -- i.e. system_abcd is exactly
    its paraxial limit."""
    from lumenairy.raytrace import (
        ray_transfer_jacobian,
        surfaces_from_prescription,
        system_abcd_prescription,
    )
    surfs = surfaces_from_prescription(_singlet())
    Msys = np.asarray(system_abcd_prescription(_singlet(), LAM)[0])
    z = np.zeros(1)
    dt = ray_transfer_jacobian(z, z, z, z, surfs, LAM)
    J = dt.jacobian[0]
    xblk = np.array([[J[0, 0], J[0, 2]], [J[2, 0], J[2, 2]]])
    assert np.max(np.abs(xblk - Msys)) < 1e-6


def test_persurface_reduces_isotropic_on_axis():
    """per_surface=True on an on-axis field gives an isotropic (round) focus --
    it reduces to the rotationally-symmetric result, no regression."""
    from lumenairy.propagators.gbd import propagate_gbd_through_prescription
    N, dx = 160, 18e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (1.2e-3) ** 2).astype(np.complex128)
    F = propagate_gbd_through_prescription(
        E, dx, _singlet(), wavelength=LAM, per_surface=True,
        output_dx=3e-6, output_shape=(96, 96), sample_step=2, waist_factor=2.0)
    assert np.isfinite(F).all()
    xf = (np.arange(96) - 48) * 3e-6
    Xf, Yf = np.meshgrid(xf, xf)
    I = np.abs(F) ** 2
    s = I.sum()
    sx = np.sqrt((I * (Xf - (I * Xf).sum() / s) ** 2).sum() / s)
    sy = np.sqrt((I * (Yf - (I * Yf).sum() / s) ** 2).sum() / s)
    assert abs(sx / sy - 1.0) < 0.05, (sx, sy)


def test_persurface_captures_off_axis_astigmatism():
    """per_surface=True at an off-axis field produces a strongly astigmatic
    (line-like) focus -- the tangential and sagittal foci separate -- which the
    paraxial single-ABCD form cannot.  Measured as focal-spot ellipticity far
    from 1 off-axis vs ~1 on-axis."""
    from lumenairy.propagators.gbd import propagate_gbd_through_prescription
    from lumenairy.raytrace import system_abcd_prescription
    N, dx = 160, 18e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    efl = float(system_abcd_prescription(_singlet(), LAM)[1])
    th = np.radians(6.0)
    E = (np.exp(-(X ** 2 + Y ** 2) / (1.2e-3) ** 2)
         * np.exp(1j * 2 * np.pi / LAM * np.sin(th) * X)).astype(np.complex128)
    F = propagate_gbd_through_prescription(
        E, dx, _singlet(), wavelength=LAM, per_surface=True,
        output_dx=3e-6, output_shape=(96, 96), sample_step=2, waist_factor=2.0,
        direction_sampling=True, output_centre=(efl * np.tan(th), 0.0))
    assert np.isfinite(F).all()
    xf = (np.arange(96) - 48) * 3e-6
    Xf, Yf = np.meshgrid(xf, xf)
    I = np.abs(F) ** 2
    s = I.sum()
    assert s > 0
    sx = np.sqrt((I * (Xf - (I * Xf).sum() / s) ** 2).sum() / s)
    sy = np.sqrt((I * (Yf - (I * Yf).sum() / s) ** 2).sum() / s)
    ellip = sx / sy
    # strongly astigmatic: one axis focuses much tighter than the other
    assert (ellip < 0.5) or (ellip > 2.0), ellip


# --------------------------------------------------------------------------
# JAX differentiability (the free-space / thin-lens paths are xp-dispatched)
# --------------------------------------------------------------------------
def _jax_ok():
    try:
        import jax  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _jax_ok(), reason='jax not installed')
def test_gbd_freespace_is_jax_differentiable_and_matches_numpy():
    """GBD's free-space path is backend-dispatched (array_namespace), so it
    runs under jax.numpy, matches the numpy result, and is differentiable /
    jittable -- the differentiable GBD twin for source-field / free-space /
    lens design optimisation (the per-surface prescription path is numpy-only)."""
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    N, dx = 64, 10e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (0.15e-3) ** 2).astype(np.complex128)
    F_np = propagate_gbd_freespace(E, dx, z=5e-3, wavelength=LAM, sample_step=2)
    F_j = propagate_gbd_freespace(jnp.asarray(E), dx, z=5e-3, wavelength=LAM,
                                  sample_step=2)
    assert float(jnp.linalg.norm(jnp.asarray(F_np) - F_j)
                 / jnp.linalg.norm(F_j)) < 1e-10

    def loss(a):
        F = propagate_gbd_freespace(a * jnp.asarray(E), dx, z=5e-3,
                                    wavelength=LAM, sample_step=2)
        return jnp.sum(jnp.abs(F) ** 2)

    g = float(jax.grad(loss)(1.0))
    assert np.isfinite(g) and g != 0.0
    assert np.isfinite(float(jax.jit(loss)(1.0)))


def test_vector_gbd_propagates_jones_components_independently():
    """propagate_gbd_freespace_vector propagates each Jones component through
    free space and each equals the scalar propagate_gbd_freespace of that
    component (independent-component propagation is exact in free space)."""
    from lumenairy.propagators.gbd import propagate_gbd_freespace_vector
    N, dx = 96, 8e-6
    z = 4e-3
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    Ex = np.exp(-(X ** 2 + Y ** 2) / (0.15e-3) ** 2).astype(np.complex128)
    Ey = (X / 0.15e-3) * Ex     # a distinct (linearly-polarized-varying) comp
    Evec = np.stack([Ex, Ey], axis=0)
    out = propagate_gbd_freespace_vector(
        Evec, dx, z=z, wavelength=LAM, sample_step=2)
    assert out.shape == (2, N, N)
    ox = propagate_gbd_freespace(Ex, dx, z=z, wavelength=LAM, sample_step=2)
    oy = propagate_gbd_freespace(Ey, dx, z=z, wavelength=LAM, sample_step=2)
    assert np.allclose(out[0], ox, rtol=1e-10, atol=1e-12)
    assert np.allclose(out[1], oy, rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------
# Anamorphic (dy != dx) grid sampling -> elliptical / diagonal-tensor-Q beamlets
# --------------------------------------------------------------------------
def test_anamorphic_scalar_path_byte_identical():
    """dy=None (and dy=dx) keep the scalar circular-beamlet path
    byte-identical -- no regression from the tensor-Q generalization."""
    N, dx = 96, 8e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (0.15e-3) ** 2).astype(np.complex128)
    F0 = propagate_gbd_freespace(E, dx, z=5e-3, wavelength=LAM, sample_step=2)
    F1 = propagate_gbd_freespace(E, dx, z=5e-3, wavelength=LAM, dy=dx,
                                 sample_step=2)
    assert np.array_equal(F0, F1)


def test_anamorphic_circular_beam_stays_circular():
    """On an anamorphic grid (dy = 2*dx) a physically circular Gaussian is
    decomposed into elliptical (diagonal-tensor-Q) beamlets; after free-space
    propagation it must stay physically circular and finite -- exercising the
    tensor free-space Q evolution."""
    dxa, dya, N = 6e-6, 12e-6, 160
    xa = (np.arange(N) - N // 2) * dxa
    ya = (np.arange(N) - N // 2) * dya
    Xa, Ya = np.meshgrid(xa, ya)
    E = np.exp(-(Xa ** 2 + Ya ** 2) / (0.12e-3) ** 2).astype(np.complex128)
    F = propagate_gbd_freespace(E, dxa, z=8e-3, wavelength=LAM, dy=dya,
                                sample_step=2, waist_factor=1.5)
    assert np.isfinite(F).all()
    I = np.abs(F) ** 2
    s = I.sum()
    sx = np.sqrt((I * (Xa - (I * Xa).sum() / s) ** 2).sum() / s)
    sy = np.sqrt((I * (Ya - (I * Ya).sum() / s) ** 2).sum() / s)
    assert abs(sx / sy - 1.0) < 0.05, (sx, sy)   # physically circular


@pytest.mark.skipif(not _jax_ok(), reason='jax not installed')
def test_ray_transfer_jacobian_jax_matches_fd_and_differentiable():
    """The JAX differential-ray-transfer (jacfwd around trace_jax) matches the
    NumPy finite-difference primitive at low NA and is jax.grad-differentiable
    -- the differentiable foundation for per-surface GBD design optimization."""
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    from lumenairy.raytrace import (
        ray_transfer_jacobian,
        ray_transfer_jacobian_jax,
        surfaces_from_prescription,
    )
    p = _singlet()
    surfs = surfaces_from_prescription(p)
    x = np.linspace(-1e-3, 1e-3, 5)
    z = np.zeros(5)
    Jfd = ray_transfer_jacobian(x, z, z, z, surfs, 0.633e-6).jacobian
    Jjax = np.asarray(ray_transfer_jacobian_jax(x, z, z, z, p, 0.633e-6))
    assert np.max(np.abs(Jjax - Jfd)) < 1e-6

    def loss(xv):
        J = ray_transfer_jacobian_jax(xv, jnp.zeros(5), jnp.zeros(5),
                                      jnp.zeros(5), p, 0.633e-6)
        return jnp.sum(J[:, 0, 0] ** 2)

    g = np.asarray(jax.grad(loss)(jnp.asarray(x)))
    assert np.isfinite(g).all() and g.shape == (5,)
