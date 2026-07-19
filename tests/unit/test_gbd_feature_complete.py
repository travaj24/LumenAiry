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


def test_z_image_ignored_on_paraxial_path_warns():
    """S2-4: ``z_image`` is consumed only on the per_surface=True path
    (last-vertex -> output leg, default BFL).  On the default per_surface=False
    (whole-system-ABCD) path the field is reconstructed at the exit vertex and
    a passed ``z_image`` is silently dropped -- so it must (a) emit a
    RuntimeWarning and (b) leave the output BYTE-IDENTICAL to not passing it.

    The byte-identity is the independent probe (the accepted-but-dropped-kwarg
    mechanism), not a tautology on the warning.  The default-``z_image`` False
    call must NOT warn (no false positive), and the per_surface=True call must
    consume ``z_image`` silently and land on a genuinely different plane.
    """
    import warnings

    from lumenairy.propagators.gbd import propagate_gbd_through_prescription

    N, dx = 96, 18e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (1.2e-3) ** 2).astype(np.complex128)
    kw = dict(wavelength=LAM, output_dx=6e-6, output_shape=(64, 64),
              sample_step=2, waist_factor=2.0)

    # Baseline paraxial-path field (no z_image) -- the exit-vertex plane.
    F_no_zimage = np.asarray(propagate_gbd_through_prescription(
        E, dx, _singlet(), per_surface=False, **kw))

    # Passing z_image on the False path warns AND is dropped (byte-identical).
    with pytest.warns(RuntimeWarning, match="z_image is only honored"):
        F_zimage = np.asarray(propagate_gbd_through_prescription(
            E, dx, _singlet(), per_surface=False, z_image=45e-3, **kw))
    assert np.array_equal(F_no_zimage, F_zimage), \
        "z_image changed the per_surface=False output despite being dropped"

    # Default (no z_image) on the False path must NOT raise the S2-4 warning.
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        propagate_gbd_through_prescription(
            E, dx, _singlet(), per_surface=False, **kw)
    assert not any('z_image is only honored' in str(w.message) for w in rec), \
        "S2-4 warning fired on the default (no z_image) path"

    # per_surface=True consumes z_image silently and lands on a different plane.
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        F_true = np.asarray(propagate_gbd_through_prescription(
            E, dx, _singlet(), per_surface=True, z_image=45e-3, **kw))
    assert not any('z_image is only honored' in str(w.message) for w in rec), \
        "S2-4 warning fired on the per_surface=True path that honors z_image"
    assert not np.array_equal(F_true, F_no_zimage), \
        "per_surface=True with z_image should not match the exit-vertex field"


def test_persurface_captures_off_axis_astigmatism():
    """per_surface=True at an off-axis field produces an ASTIGMATIC beam -- the
    tensor Q's two principal 1/q curvatures separate (tangential vs sagittal),
    which the paraxial single-ABCD (scalar, rotationally-symmetric Q) cannot
    represent.  Measured directly on the reconstructed tensor Q's eigenvalue
    split, so the check is fast and does not depend on resolving the astigmatic
    line focus in a tiny reconstruction window.

    NOTE the input grid MUST resolve the field tilt (Nyquist): a 6 deg tilt at
    633 nm has ``sin(theta)*dx/lambda`` cycles/pixel, which must stay well below
    0.5, i.e. ``dx << lambda/(2 sin theta) ~ 3 um``.  On a coarser grid the phase
    ramp ALIASES -- ``np.gradient`` then recovers the aliased (near-zero) angle
    and NO propagator can walk the beam off-axis.  (An earlier 18 um-grid form of
    this test aliased the tilt; the code faithfully propagates the aliased input,
    so that form was ill-posed.)  A single 6 deg chief ray traces to
    ``efl*tan(6 deg)`` exactly, confirming the ray physics is correct.
    """
    from lumenairy.propagators.gbd import (
        _eigvals2x2,
        apply_prescription_persurface_to_beamlets,
        decompose_field_to_beamlets,
    )
    N, dx = 160, 1.4e-6            # 6 deg -> 0.23 cyc/px, safely resolved
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    th = np.radians(6.0)

    def astig_index(tilted):
        ramp = (np.exp(1j * 2 * np.pi / LAM * np.sin(th) * X)
                if tilted else np.ones_like(X))
        E = (np.exp(-(X ** 2 + Y ** 2) / (0.07e-3) ** 2)
             * ramp).astype(np.complex128)
        b = decompose_field_to_beamlets(E, dx, wavelength=LAM, sample_step=3,
                                        waist_factor=3.0,
                                        direction_sampling=True)
        bundle = apply_prescription_persurface_to_beamlets(b, _singlet(), LAM)
        Q = np.asarray(bundle.Q)
        assert np.isfinite(Q).all()
        lam = _eigvals2x2(Q, np)         # (n, 2) principal 1/q eigenvalues
        w = np.abs(bundle.amplitude) ** 2
        idx = (np.abs(lam[:, 0] - lam[:, 1])
               / (np.abs(lam[:, 0] + lam[:, 1]) + 1e-30))
        return float(np.average(idx, weights=w))

    a_on = astig_index(False)
    a_off = astig_index(True)
    # on-axis stays rotationally symmetric (Qxx==Qyy, Qxy==0 -> zero split);
    # off-axis the tensor Q is astigmatic (a split the paraxial form can't hold).
    assert a_on < 1e-6, a_on
    assert a_off > 1e-5, a_off
    assert a_off > 20.0 * (a_on + 1e-12)


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


# --------------------------------------------------------------------------
# Polarization ray tracing (per-surface Fresnel s/p Jones along each base ray)
# --------------------------------------------------------------------------
def _single_sphere(R=20e-3):
    """One air->glass spherical refracting surface (clean analytic Fresnel)."""
    return {'name': 'sph', 'aperture_diameter': 30e-3,
            'surfaces': [{'radius': R, 'conic': 0., 'glass_before': 'air',
                          'glass_after': 'N-BK7', 'semi_diameter': 15e-3}],
            'thicknesses': [0.0]}


def test_fresnel_jones_onaxis_isotropic_and_s_channel_exact():
    """The per-beamlet Fresnel Jones matrix is isotropic on-axis (= the
    normal-incidence Fresnel amplitude, no diattenuation, no s-p mixing), and
    for an off-axis meridional ray its s channel (E_y here) is transverse-exact
    -- it equals the analytic Fresnel t_s at the surface's local incidence
    angle -- with no s-p cross terms."""
    from lumenairy.glass import get_glass_index
    from lumenairy.propagators.gbd import _fresnel_jones_matrix_per_beamlet
    R = 20e-3
    presc = _single_sphere(R)
    ng = float(get_glass_index('N-BK7', 0.633e-6))
    z = np.zeros(1)
    # on-axis: P = diag(t0, t0), t0 = 2/(1+ng), off-diagonals zero
    P0, a0 = _fresnel_jones_matrix_per_beamlet(z, z, z, z, presc, 0.633e-6)
    t0 = 2.0 / (1.0 + ng)
    assert a0[0]
    assert abs(P0[0, 0, 0] - t0) < 1e-9 and abs(P0[0, 1, 1] - t0) < 1e-9
    assert abs(P0[0, 0, 1]) < 1e-12 and abs(P0[0, 1, 0]) < 1e-12
    # off-axis meridional (x-offset) ray: incidence i = asin(h/R); s = E_y
    for h in (4e-3, 8e-3):
        x = np.array([h])
        P, al = _fresnel_jones_matrix_per_beamlet(x, z, z, z, presc, 0.633e-6)
        i = np.arcsin(h / R)
        ci = np.cos(i)
        ct = np.sqrt(1.0 - (np.sin(i) / ng) ** 2)
        ts = 2.0 * ci / (ci + ng * ct)
        assert al[0]
        assert abs(abs(P[0, 1, 1]) - ts) < 1e-6, (h, abs(P[0, 1, 1]), ts)
        # no s-p mixing for a purely meridional ray
        assert abs(P[0, 0, 1]) < 1e-9 and abs(P[0, 1, 0]) < 1e-9


def test_fresnel_jones_dead_ray_is_zeroed_not_nan():
    """A base ray that vignettes (outside the aperture) comes back alive=False
    with a zeroed (finite, not NaN) Jones matrix, so P @ E never propagates
    NaNs into live neighbouring beamlets."""
    from lumenairy.propagators.gbd import _fresnel_jones_matrix_per_beamlet
    presc = _single_sphere(20e-3)
    z = np.zeros(1)
    x = np.array([20e-3])   # outside the 15 mm semi-diameter stop
    P, alive = _fresnel_jones_matrix_per_beamlet(x, z, z, z, presc, 0.633e-6)
    assert not alive[0]
    assert np.isfinite(P).all()
    assert np.allclose(P, 0.0)


def test_vector_through_prescription_applies_fresnel_transmission():
    """propagate_gbd_vector_through_prescription carries an x-polarized beam
    through a singlet applying per-surface Fresnel s/p transmission: the output
    (2, Ny, Nx) Jones field is finite, stays predominantly x-polarized (cross-
    pol negligible by rotational symmetry), and its x-channel power is the
    scalar (no-Fresnel) result scaled by the two-surface Fresnel power
    transmission T1*T2 (near-axis)."""
    from lumenairy.glass import get_glass_index
    from lumenairy.propagators.gbd import (
        propagate_gbd_through_prescription,
        propagate_gbd_vector_through_prescription,
    )
    N, dx = 96, 10e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E0 = np.exp(-(X ** 2 + Y ** 2) / (2.0e-3) ** 2).astype(np.complex128)
    kw = dict(wavelength=LAM, per_surface=True, output_dx=3e-6,
              output_shape=(64, 64), sample_step=2, waist_factor=2.0)
    scal = propagate_gbd_through_prescription(E0, dx, _singlet(49.3e-3), **kw)
    vec = propagate_gbd_vector_through_prescription(
        np.stack([E0, np.zeros_like(E0)]), dx, _singlet(49.3e-3), **kw)
    assert vec.shape == (2, 64, 64) and np.isfinite(vec).all()
    Ps = float(np.sum(np.abs(scal) ** 2))
    Pvx = float(np.sum(np.abs(vec[0]) ** 2))
    Pvy = float(np.sum(np.abs(vec[1]) ** 2))
    # cross-pol negligible by symmetry
    assert Pvy < 1e-6 * Pvx
    # x-channel carries the two-surface near-axis Fresnel power transmission
    ng = float(get_glass_index('N-BK7', LAM))
    T1 = ng * (2.0 / (1.0 + ng)) ** 2                    # air -> glass
    T2 = (1.0 / ng) * (2.0 * ng / (ng + 1.0)) ** 2       # glass -> air
    assert abs(Pvx / Ps - T1 * T2) < 5e-3, (Pvx / Ps, T1 * T2)


def test_fresnel_jones_mirror_reflects_and_conserves_energy():
    """A surface flagged ``is_mirror`` is REFLECTED (not refracted) along the
    base ray -- matching raytrace.trace -- and gets ideal-reflector coefficients
    ``|r_s| = |r_p| = 1`` (energy-conserving, no diattenuation), unlike a
    refractive surface which is Fresnel-lossy.  Off-axis the p (x) column carries
    the honest ``cos(2i)`` transverse projection of the beam folded by ``2i``
    while the s (y) column stays exactly unit."""
    from lumenairy.propagators.gbd import _fresnel_jones_matrix_per_beamlet
    from lumenairy.raytrace.intersection import _intersect_surface, _reflect
    from lumenairy.raytrace.trace import _make_bundle, surfaces_from_prescription
    R = 40e-3
    mirror = {'name': 'm', 'aperture_diameter': 40e-3,
              'surfaces': [{'radius': -R, 'conic': 0., 'glass_before': 'air',
                            'glass_after': 'air', 'is_mirror': True,
                            'semi_diameter': 18e-3}],
              'thicknesses': [0.0]}
    surfs = surfaces_from_prescription(mirror)
    assert getattr(surfs[0], 'is_mirror', False) is True
    hs = np.array([0.0, 5e-3, 10e-3, 15e-3])
    z = np.zeros_like(hs)
    # (1) base ray really reflects: my traced output dir == a direct _reflect
    rb = _make_bundle(hs.copy(), z.copy(), z.copy(), z.copy(), 0.633e-6)
    _intersect_surface(rb, surfs[0])
    _reflect(rb, surfs[0])
    ref_N = rb.N.copy()
    assert np.all(ref_N < 0)   # reflected back toward -z (a real reflection)
    P, alive = _fresnel_jones_matrix_per_beamlet(hs, z, z, z, mirror, 0.633e-6)
    assert alive.all() and np.isfinite(P).all()
    # (2) on-axis: unitary / energy-conserving, no diattenuation
    assert abs(abs(np.linalg.det(P[0])) - 1.0) < 1e-9
    assert abs(np.linalg.norm(P[0, :, 0]) - 1.0) < 1e-9
    assert abs(np.linalg.norm(P[0, :, 1]) - 1.0) < 1e-9
    # (3) off-axis: s (y) column stays unit; p (x) column == cos(2i) projection
    for h, Pi in zip(hs[1:], P[1:]):
        i = np.arcsin(h / R)
        assert abs(np.linalg.norm(Pi[:, 1]) - 1.0) < 1e-6          # s exact
        assert abs(np.linalg.norm(Pi[:, 0]) - np.cos(2 * i)) < 1e-6  # p proj
    # (4) contrast: same geometry made refractive is Fresnel-LOSSY (< 1)
    refr = {'name': 'r', 'aperture_diameter': 40e-3,
            'surfaces': [{'radius': -R, 'conic': 0., 'glass_before': 'air',
                          'glass_after': 'N-BK7', 'semi_diameter': 18e-3}],
            'thicknesses': [0.0]}
    Pr, _ = _fresnel_jones_matrix_per_beamlet(hs, z, z, z, refr, 0.633e-6)
    assert np.linalg.norm(Pr[0, :, 0]) < 0.99   # normal-incidence Fresnel loss


def test_fresnel_jones_metal_coating_diattenuation_and_retardance():
    """A mirror carrying a complex-index ``coating`` (metal) gets the full
    complex Fresnel r_s / r_p: normal-incidence reflectance matches the analytic
    metal value, off-axis it shows diattenuation (|r_s| != |r_p|, matching
    analytic) and retardance, and as the coating -> a perfect conductor it
    reduces continuously to the ideal (no-coating) reflector."""
    from lumenairy.propagators.gbd import _fresnel_jones_matrix_per_beamlet
    nAl = 1.374 + 7.620j          # aluminum @ 633 nm (n + i*kappa)
    R = 40e-3

    def _mirror(coat):
        return {'name': 'm', 'aperture_diameter': 40e-3,
                'surfaces': [{'radius': -R, 'conic': 0., 'glass_before': 'air',
                              'glass_after': 'air', 'is_mirror': True,
                              'coating': coat, 'semi_diameter': 18e-3}],
                'thicknesses': [0.0]}
    hs = np.array([0.0, 5e-3, 10e-3, 15e-3])
    z = np.zeros_like(hs)
    P, _ = _fresnel_jones_matrix_per_beamlet(
        hs, z, z, z, _mirror(lambda w: nAl), 0.633e-6)
    # (1) normal-incidence reflectance == analytic |(1-n)/(1+n)|^2 (~0.914 Al)
    Rn = abs((1 - nAl) / (1 + nAl)) ** 2
    assert abs(abs(P[0, 0, 0]) ** 2 - Rn) < 1e-6
    assert abs(abs(P[0, 1, 1]) ** 2 - Rn) < 1e-6
    assert 0.90 < Rn < 0.93
    # (2) off-axis: |r_s| (= |Pyy|) and |r_p| (= |Pxx|/cos 2i) match analytic
    #     Fresnel, and the metal is diattenuating (|r_s| > |r_p|, growing).
    prev = -1.0
    for h, Pi in zip(hs[1:], P[1:]):
        i = np.arcsin(h / R)
        ci = np.cos(i) + 0j
        ct = np.sqrt(1 - (1 / nAl) ** 2 * (1 - ci ** 2))
        rs = (ci - nAl * ct) / (ci + nAl * ct)
        rp = (nAl * ci - ct) / (nAl * ci + ct)
        assert abs(abs(Pi[1, 1]) - abs(rs)) < 1e-6
        assert abs(abs(Pi[0, 0]) / np.cos(2 * i) - abs(rp)) < 1e-6
        d = abs(rs) - abs(rp)
        assert d > prev and d > 0        # diattenuation grows with angle
        prev = d
    # (3) retardance: the metal's s-p relative phase differs from an ideal
    #     mirror's (PEC is real -1 / +1 -> relative phase pi; the metal is not).
    Pideal, _ = _fresnel_jones_matrix_per_beamlet(hs, z, z, z, _mirror(None),
                                                  0.633e-6)
    rel_metal = np.angle(P[2, 1, 1]) - np.angle(P[2, 0, 0])
    rel_ideal = np.angle(Pideal[2, 1, 1]) - np.angle(Pideal[2, 0, 0])
    assert abs(rel_metal - rel_ideal) > 1e-3     # a real retardance signature
    # (4) perfect-conductor limit reduces continuously to the ideal reflector
    Pbig, _ = _fresnel_jones_matrix_per_beamlet(
        hs, z, z, z, _mirror(lambda w: 1 + 1e7j), 0.633e-6)
    assert np.max(np.abs(Pbig - Pideal)) < 1e-5
    # (5) a direct complex coating value resolves the same as a callable
    Pc, _ = _fresnel_jones_matrix_per_beamlet(
        hs, z, z, z, _mirror(nAl), 0.633e-6)
    assert np.allclose(Pc, P)


# --------------------------------------------------------------------------
# World-frame output plane (large folds -- reconstruct on the physical plane
# perpendicular to the folded beam, not the fixed +z x-y grid)
# --------------------------------------------------------------------------
def _periscope():
    """Plano-convex N-BK7 singlet + two 45-deg flat folds (90-deg periscope)
    to +y; detector 50 mm past the fold."""
    return {'surfaces': [
        {'radius': 50e-3, 'glass_before': 'air', 'glass_after': 'N-BK7',
         'surf_num': 1},
        {'radius': float('inf'), 'glass_before': 'N-BK7', 'glass_after': 'air',
         'surf_num': 2},
        {'radius': float('inf'), 'glass_before': 'air', 'glass_after': 'MIRROR',
         'surf_num': 15},
        {'radius': float('inf'), 'glass_before': 'air', 'glass_after': 'air',
         'surf_num': 25}],
        'thicknesses': [3e-3, 0.05, 0.0, 0.0], 'aperture_diameter': 0.010,
        'coord_breaks': [
            {'surf_num': 10, 'tilt_x_deg': 45.0, 'order': 0, 'thickness_m': 0.0},
            {'surf_num': 20, 'tilt_x_deg': 45.0, 'order': 0,
             'thickness_m': -0.05}]}


def _peak_and_rms(F, dxo):
    I = np.abs(F) ** 2
    s = I.sum()
    if not np.isfinite(s) or s <= 0:
        return float('nan'), float('nan')
    n = F.shape[0]
    xf = (np.arange(F.shape[1]) - F.shape[1] // 2) * dxo
    yf = (np.arange(n) - n // 2) * dxo
    Xf, Yf = np.meshgrid(xf, yf)
    cx = (I * Xf).sum() / s
    cy = (I * Yf).sum() / s
    rms = np.sqrt((I * ((Xf - cx) ** 2 + (Yf - cy) ** 2)).sum() / s)
    return float(I.max()), float(rms)


def test_world_output_plane_focuses_folded_periscope():
    """A 90-deg periscope folds the beam onto +y; the fixed +z x-y
    reconstruction is meaningless (blows up), but world_output_plane='auto'
    reconstructs on the physical (folded) focal plane and gives a finite,
    focused spot."""
    from lumenairy.propagators.gbd import propagate_gbd_through_prescription
    N, dx = 160, 12e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (3.0e-3) ** 2).astype(np.complex128)
    kw = dict(wavelength=0.633e-6, per_surface=True, output_dx=1.5e-6,
              output_shape=(128, 128), sample_step=3, waist_factor=2.0)
    Fw = propagate_gbd_through_prescription(
        E, dx, _periscope(), world_output_plane='auto', **kw)
    peak_w, rms_w = _peak_and_rms(Fw, 1.5e-6)
    assert np.isfinite(Fw).all()
    assert peak_w > 0 and np.isfinite(rms_w)
    assert rms_w < 60e-6            # a real focus (not a smear across the grid)
    # the fold-blind fixed x-y path does NOT focus (reflects to -z, huge tilt)
    Fx = propagate_gbd_through_prescription(E, dx, _periscope(), **kw)
    with np.errstate(over='ignore', invalid='ignore'):
        energy_x = float(np.sum(np.abs(Fx) ** 2))
        _, rms_x = _peak_and_rms(Fx, 1.5e-6)
    assert (not np.isfinite(energy_x)) or (rms_x > 3 * rms_w) \
        or (not np.isfinite(rms_x))


def test_world_output_plane_matches_baseline_on_unfolded_system():
    """On a straight (unfolded) singlet, world_output_plane='auto' reproduces
    the default (fixed x-y) per-surface reconstruction's focused spot -- same
    peak intensity and rms -- so the world reframing is a faithful no-op when
    there is no fold."""
    from lumenairy.propagators.gbd import propagate_gbd_through_prescription
    straight = {'surfaces': [
        {'radius': 50e-3, 'glass_before': 'air', 'glass_after': 'N-BK7',
         'surf_num': 1},
        {'radius': float('inf'), 'glass_before': 'N-BK7', 'glass_after': 'air',
         'surf_num': 2},
        {'radius': float('inf'), 'glass_before': 'air', 'glass_after': 'air',
         'surf_num': 25}],
        'thicknesses': [3e-3, 0.0971, 0.0], 'aperture_diameter': 0.010,
        'coord_breaks': [{'surf_num': 1, 'tilt_x_deg': 0.0, 'order': 0,
                          'thickness_m': 0.0}]}
    N, dx = 160, 12e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (3.0e-3) ** 2).astype(np.complex128)
    kw = dict(wavelength=0.633e-6, per_surface=True, output_dx=1.5e-6,
              output_shape=(128, 128), sample_step=3, waist_factor=2.0)
    Fn = propagate_gbd_through_prescription(E, dx, straight, **kw)
    Fa = propagate_gbd_through_prescription(
        E, dx, straight, world_output_plane='auto', **kw)
    pk_n, rms_n = _peak_and_rms(Fn, 1.5e-6)
    pk_a, rms_a = _peak_and_rms(Fa, 1.5e-6)
    assert abs(pk_a - pk_n) < 0.02 * pk_n       # same focused intensity
    assert abs(rms_a - rms_n) < 0.02 * rms_n + 1e-7


def test_world_output_plane_rejects_curved_fold():
    """A curved (powered) fold mirror is not Q-invariant, so the unfolded-
    equivalent shortcut is invalid and the call raises a clear error."""
    from lumenairy.propagators.gbd import propagate_gbd_through_prescription
    curved = {'surfaces': [
        {'radius': -0.2, 'glass_before': 'air', 'glass_after': 'MIRROR',
         'surf_num': 15},
        {'radius': float('inf'), 'glass_before': 'air', 'glass_after': 'air',
         'surf_num': 25}],
        'thicknesses': [0.0, 0.0], 'aperture_diameter': 0.020,
        'coord_breaks': [
            {'surf_num': 1, 'tilt_x_deg': 0.0, 'order': 0, 'thickness_m': 0.1},
            {'surf_num': 10, 'tilt_x_deg': 45.0, 'order': 0, 'thickness_m': 0.0},
            {'surf_num': 20, 'tilt_x_deg': 45.0, 'order': 0,
             'thickness_m': -0.1}]}
    N, dx = 96, 12e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (3.0e-3) ** 2).astype(np.complex128)
    with pytest.raises(NotImplementedError):
        propagate_gbd_through_prescription(
            E, dx, curved, wavelength=0.633e-6, per_surface=True,
            world_output_plane='auto', output_dx=2e-6, output_shape=(64, 64),
            sample_step=4)


def test_transmit_thin_film_ar_coating():
    """A multilayer (AR) ``coating`` on a refracting surface routes t_s / t_p
    through the thin-film characteristic matrix: a quarter-wave AR layer nearly
    eliminates the single-interface Fresnel reflection loss (transmittance -> 1),
    while ``coating=None`` stays byte-identical to bare Fresnel; the TMM reduces
    to bare Fresnel at zero layers."""
    from lumenairy.propagators.gbd import (
        _fresnel_jones_matrix_per_beamlet,
        _thin_film_coefficients,
    )
    ng = 1.515
    n_ar = np.sqrt(ng)
    d_ar = 0.633e-6 / (4 * n_ar)      # quarter-wave at 633 nm

    def _surf(coat):
        return {'name': 's', 'aperture_diameter': 30e-3, 'surfaces': [
            {'radius': 40e-3, 'conic': 0., 'glass_before': 'air',
             'glass_after': 'N-BK7', 'coating': coat, 'semi_diameter': 14e-3}],
            'thicknesses': [0.0]}
    hs = np.array([0.0, 4e-3, 8e-3])
    z = np.zeros(3)
    Pu, _ = _fresnel_jones_matrix_per_beamlet(hs, z, z, z, _surf(None), 0.633e-6)
    Pa, _ = _fresnel_jones_matrix_per_beamlet(
        hs, z, z, z, _surf([(n_ar, d_ar)]), 0.633e-6)
    R = 40e-3
    for i, h in enumerate(hs):
        ai = np.arcsin(h / R)
        ci = np.cos(ai)
        ct = np.sqrt(1 - (np.sin(ai) / ng) ** 2)
        Tu = (ng * ct) / ci * abs(Pu[i, 1, 1]) ** 2
        Ta = (ng * ct) / ci * abs(Pa[i, 1, 1]) ** 2
        assert Tu < 0.97                 # uncoated has a Fresnel loss
        assert Ta > 0.999                # AR nearly eliminates it
    # coating=None is byte-identical (bare-Fresnel path untouched)
    assert np.array_equal(
        Pu, _fresnel_jones_matrix_per_beamlet(hs, z, z, z, _surf(None),
                                              0.633e-6)[0])
    # zero-layer TMM == bare Fresnel; dict layer form == tuple form
    rs, rp, ts, tp = _thin_film_coefficients([], 1.0, ng, np.array([1.0]),
                                             0.633e-6)
    assert abs(abs(ts[0]) - 2.0 / (1.0 + ng)) < 1e-12
    Pd, _ = _fresnel_jones_matrix_per_beamlet(
        hs, z, z, z, _surf([{'index': n_ar, 'thickness': d_ar}]), 0.633e-6)
    assert np.allclose(Pd, Pa)


def test_reflection_multilayer_hr_stack():
    """A dielectric multilayer (quarter-wave Bragg) ``coating`` on a mirror
    gives r_s / r_p from the thin-film characteristic matrix: reflectance climbs
    toward 1 with more pairs (matching the analytic Bragg formula), is energy-
    conserving, and leaves the single-metal-index and PEC paths unchanged."""
    from lumenairy.propagators.gbd import (
        _fresnel_jones_matrix_per_beamlet,
        _thin_film_coefficients,
    )
    nH, nL, ns = 2.30, 1.46, 1.515
    dH, dL = 0.633e-6 / (4 * nH), 0.633e-6 / (4 * nL)

    def _hr(npairs):
        return {'layers': [(nH, dH), (nL, dL)] * npairs, 'substrate': ns}

    def _mirror(coat):
        return {'name': 'm', 'aperture_diameter': 30e-3, 'surfaces': [
            {'radius': float('inf'), 'conic': 0., 'glass_before': 'air',
             'glass_after': 'air', 'is_mirror': True, 'coating': coat,
             'semi_diameter': 14e-3}], 'thicknesses': [0.0]}
    z = np.zeros(1)
    Rs = []
    for npairs in (2, 4, 8):
        P, _ = _fresnel_jones_matrix_per_beamlet(
            z, z, z, z, _mirror(_hr(npairs)), 0.633e-6)
        Rs.append(abs(P[0, 0, 0]) ** 2)
    assert Rs[0] < Rs[1] < Rs[2]              # reflectance climbs with pairs
    assert Rs[2] > 0.99                        # 8-pair HR is a good mirror
    N = 8
    Rb = ((1.0 * nL ** (2 * N) - ns * nH ** (2 * N))
          / (1.0 * nL ** (2 * N) + ns * nH ** (2 * N))) ** 2
    assert abs(Rs[2] - Rb) < 1e-4              # matches analytic Bragg formula
    # energy R + T = 1 (lossless) at oblique
    rs, rp, ts, tp = _thin_film_coefficients(
        [(nH, dH), (nL, dL)] * 6, 1.0, ns,
        np.array([np.cos(np.radians(30))]), 0.633e-6)
    cts = np.sqrt(1 - (np.sin(np.radians(30)) / ns) ** 2)
    Ts = (ns * cts) / np.cos(np.radians(30)) * abs(ts[0]) ** 2
    assert abs(abs(rs[0]) ** 2 + Ts - 1.0) < 1e-9
    # metal single-index + PEC unchanged
    Pm, _ = _fresnel_jones_matrix_per_beamlet(
        z, z, z, z, _mirror(1.374 + 7.62j), 0.633e-6)
    assert abs(abs(Pm[0, 0, 0]) ** 2 - 0.9137) < 1e-3
    Pp, _ = _fresnel_jones_matrix_per_beamlet(z, z, z, z, _mirror(None), 0.633e-6)
    assert abs(abs(Pp[0, 0, 0]) - 1.0) < 1e-9


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


def test_gbd_ghost_analysis_stray_light_budget():
    """gbd_ghost_analysis returns the first-order ghost budget: per-surface
    Fresnel reflectance (~4% uncoated N-BK7) and the double-bounce relative
    intensity (~R_i*R_j); a quarter-wave AR coating suppresses both."""
    from lumenairy.propagators.gbd import gbd_ghost_analysis
    lam = 0.633e-6
    ng = 1.515
    n_ar = np.sqrt(ng)
    d_ar = lam / (4 * n_ar)

    def _singlet(coats):
        return {'aperture_diameter': 12e-3, 'surfaces': [
            {'radius': 51.5e-3, 'glass_before': 'air', 'glass_after': 'N-BK7',
             'coating': coats[0]},
            {'radius': -51.5e-3, 'glass_before': 'N-BK7', 'glass_after': 'air',
             'coating': coats[1]}], 'thicknesses': [4e-3, 40e-3]}
    g = gbd_ghost_analysis(_singlet([None, None]), lam)
    R = g['surface_reflectance']
    assert R.shape == (2,)
    assert np.all(np.abs(R - 0.0419) < 2e-3)          # ~4% uncoated glass
    i, j, gi = g['worst']
    assert (i, j) == (0, 1)
    assert abs(gi - R[0] * R[1]) < 1e-9               # double-bounce = R_i*R_j
    # AR coating suppresses both surfaces -> the ghost by many orders
    ga = gbd_ghost_analysis(
        _singlet([[(n_ar, d_ar)], [(n_ar, d_ar)]]), lam)
    assert np.all(ga['surface_reflectance'] < 1e-4)
    assert ga['worst'][2] < 1e-6 * gi
