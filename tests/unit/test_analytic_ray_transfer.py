"""Analytic differential ray transfer (``ray_transfer_jacobian_analytic``).

Forward-mode AD over the exact conic trace == the analytic differential ray
tracing of Stone & Forbes (Volatier 2017); the exact / autodiff twin of the
finite-difference ``ray_transfer_jacobian``.  These pin: agreement with the FD
primitive (composite + per-surface + opd + alive) for refraction, reflection and
conics; that the analytic value is the ``h -> 0`` limit of the FD; the paraxial
(system ABCD) limit on axis; and the JAX path (exact + ``jax.grad``-able, the
high-NA gradient the paraxial ``ray_transfer_jacobian_jax`` cannot give).
"""
import numpy as np
import pytest

from lumenairy.raytrace import (
    ray_transfer_jacobian,
    ray_transfer_jacobian_analytic,
    surfaces_from_prescription,
    system_abcd_prescription,
)

LAM = 0.633e-6

# A spread of rays across NA / off-axis / skew.
_X = np.array([0.0, 3e-3, 5e-3, -4e-3, 2e-3])
_Y = np.array([0.0, -2e-3, 1e-3, 3e-3, -1e-3])
_UX = np.array([0.0, 0.01, 0.0, 0.0, 0.03])
_UY = np.array([0.0, 0.0, -0.02, 0.02, -0.01])


def _singlet():
    return {'name': 's', 'aperture_diameter': 12e-3, 'surfaces': [
        {'radius': 51.5e-3, 'conic': 0., 'glass_before': 'air',
         'glass_after': 'N-BK7', 'semi_diameter': 6e-3},
        {'radius': -51.5e-3, 'conic': 0., 'glass_before': 'N-BK7',
         'glass_after': 'air', 'semi_diameter': 6e-3}],
        'thicknesses': [4e-3, 40e-3]}


def _jax_ok():
    try:
        import jax  # noqa: F401
        return True
    except Exception:
        return False


def test_analytic_matches_fd_composite_refraction():
    """The composite (x,y,ux,uy) Jacobian, output state, OPL and alive mask all
    equal the finite-difference primitive for a refractive singlet."""
    surfs = surfaces_from_prescription(_singlet())
    a = ray_transfer_jacobian_analytic(_X, _Y, _UX, _UY, surfs, LAM)
    f = ray_transfer_jacobian(_X, _Y, _UX, _UY, surfs, LAM)
    assert np.max(np.abs(a.jacobian - f.jacobian)) < 1e-6      # FD floor
    assert np.max(np.abs(np.stack(
        [a.x - f.x, a.y - f.y, a.ux - f.ux, a.uy - f.uy]))) < 1e-10
    assert np.max(np.abs(a.opd - f.opd)) < 1e-12
    assert np.array_equal(a.alive, f.alive)


def test_analytic_matches_fd_reflection():
    """A curved mirror (is_mirror) reflects the base ray and gives the same
    Jacobian as the FD primitive."""
    mir = {'name': 'm', 'aperture_diameter': 40e-3, 'surfaces': [
        {'radius': -40e-3, 'conic': 0., 'glass_before': 'air',
         'glass_after': 'air', 'is_mirror': True, 'semi_diameter': 18e-3},
        {'radius': float('inf'), 'conic': 0., 'glass_before': 'air',
         'glass_after': 'air', 'semi_diameter': 18e-3}],
        'thicknesses': [-20e-3, 0.0]}
    surfs = surfaces_from_prescription(mir)
    a = ray_transfer_jacobian_analytic(_X, _Y, _UX, _UY, surfs, LAM)
    f = ray_transfer_jacobian(_X, _Y, _UX, _UY, surfs, LAM)
    assert np.max(np.abs(a.jacobian - f.jacobian)) < 1e-6
    assert np.max(np.abs(a.x - f.x)) < 1e-10 and np.max(np.abs(a.y - f.y)) < 1e-10


def test_analytic_matches_fd_conic():
    """Conic surfaces (parabola / ellipse) match the FD primitive."""
    con = {'name': 'c', 'aperture_diameter': 20e-3, 'surfaces': [
        {'radius': 30e-3, 'conic': -1.0, 'glass_before': 'air',
         'glass_after': 'N-BK7', 'semi_diameter': 9e-3},
        {'radius': -80e-3, 'conic': -0.5, 'glass_before': 'N-BK7',
         'glass_after': 'air', 'semi_diameter': 9e-3}],
        'thicknesses': [5e-3, 20e-3]}
    surfs = surfaces_from_prescription(con)
    a = ray_transfer_jacobian_analytic(_X, _Y, _UX, _UY, surfs, LAM)
    f = ray_transfer_jacobian(_X, _Y, _UX, _UY, surfs, LAM)
    assert np.max(np.abs(a.jacobian - f.jacobian)) < 1e-6


def test_analytic_is_the_h_to_zero_limit_of_fd():
    """The analytic Jacobian is EXACT: the FD error (vs the analytic) shrinks
    quadratically as the FD step h is reduced -- i.e. the analytic is the h->0
    limit, not merely 'close to' the FD."""
    surfs = surfaces_from_prescription(_singlet())
    x = np.array([5e-3])
    z = np.zeros(1)
    Ja = ray_transfer_jacobian_analytic(x, z, z, z, surfs, LAM).jacobian
    errs = []
    for hp in (1e-6, 5e-7, 2.5e-7):
        d = ray_transfer_jacobian(x, z, z, z, surfs, LAM,
                                  h_pos=hp, h_slope=hp * 50)
        errs.append(np.max(np.abs(d.jacobian - Ja)))
    # each halving of h reduces the FD-vs-analytic error (converging to analytic)
    assert errs[1] < errs[0] and errs[2] < errs[1]
    assert errs[2] < 1e-8


def test_analytic_per_surface_composes_to_composite():
    """The per-surface local Jacobians multiply back to the composite (a valid,
    self-consistent decomposition)."""
    surfs = surfaces_from_prescription(_singlet())
    comp = ray_transfer_jacobian_analytic(_X, _Y, _UX, _UY, surfs, LAM).jacobian
    per = ray_transfer_jacobian_analytic(
        _X, _Y, _UX, _UY, surfs, LAM, per_surface=True).jacobian
    assert per.shape[0] == len(surfs)
    prod = np.einsum('nij,njk->nik', per[1], per[0])
    assert np.max(np.abs(prod - comp)) < 1e-12


def test_analytic_onaxis_equals_system_abcd():
    """On axis the 2x2 meridional block is exactly the paraxial system ABCD
    (system_abcd is the paraxial limit of the analytic differential transfer)."""
    p = _singlet()
    surfs = surfaces_from_prescription(p)
    z = np.zeros(1)
    J = ray_transfer_jacobian_analytic(z, z, z, z, surfs, LAM).jacobian[0]
    Msys = np.asarray(system_abcd_prescription(p, LAM)[0])
    xblk = np.array([[J[0, 0], J[0, 2]], [J[2, 0], J[2, 2]]])
    assert np.max(np.abs(xblk - Msys)) < 1e-9


def test_analytic_gives_the_same_gbd_field_as_fd():
    """Selected via ``jacobian='analytic'`` in the per-surface GBD, the analytic
    transfer reconstructs the same field as the default FD primitive."""
    from lumenairy.propagators.gbd import propagate_gbd_through_prescription
    p = _singlet()
    N, dx = 96, 10e-6
    xg = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xg, xg)
    E = np.exp(-(X ** 2 + Y ** 2) / (1.2e-3) ** 2).astype(np.complex128)
    kw = dict(wavelength=LAM, per_surface=True, output_dx=3e-6,
              output_shape=(64, 64), sample_step=2, waist_factor=2.0,
              z_image=40e-3)
    F_fd = propagate_gbd_through_prescription(E, dx, p, jacobian='fd', **kw)
    F_an = propagate_gbd_through_prescription(E, dx, p, jacobian='analytic',
                                              **kw)
    rel = np.linalg.norm(F_an - F_fd) / max(np.linalg.norm(F_fd), 1e-30)
    assert rel < 1e-6, rel


def test_analytic_rejects_biconic_and_asphere():
    """The analytic path reads only ``radius`` / ``conic`` for a
    refracting/reflecting surface (rotationally symmetric), so it must REJECT
    biconic (``radius_y``) and aspheric-polynomial surfaces rather than silently
    trace them rotationally-symmetric with a wrong off-axis power.  (Coordinate
    breaks ARE handled -- a separate frame-transform branch.)"""
    biconic = {'name': 'b', 'aperture_diameter': 16e-3, 'surfaces': [
        {'radius': 50e-3, 'radius_y': 30e-3, 'conic': 0., 'conic_y': 0.,
         'glass_before': 'air', 'glass_after': 'N-BK7', 'semi_diameter': 8e-3}],
        'thicknesses': [0.0]}
    surfs = surfaces_from_prescription(biconic)
    z = np.zeros(1)
    with pytest.raises(NotImplementedError):
        ray_transfer_jacobian_analytic(z, z, z, z, surfs, LAM)
    asph = {'name': 'a', 'aperture_diameter': 16e-3, 'surfaces': [
        {'radius': 50e-3, 'conic': 0., 'aspheric_coeffs': {4: 1e3},
         'glass_before': 'air', 'glass_after': 'N-BK7', 'semi_diameter': 8e-3}],
        'thicknesses': [0.0]}
    with pytest.raises(NotImplementedError):
        ray_transfer_jacobian_analytic(
            z, z, z, z, surfaces_from_prescription(asph), LAM)


def test_analytic_dead_ray_is_finite_and_masked():
    """A base ray that TIRs / misses comes back alive=False with a FINITE
    (zeroed) Jacobian -- not a NaN that would contaminate array-wide ops -- and
    the live rays are unaffected (still match the FD primitive)."""
    p = {'name': 's', 'aperture_diameter': 20e-3, 'surfaces': [
        {'radius': 12e-3, 'conic': 0., 'glass_before': 'air',
         'glass_after': 'N-BK7', 'semi_diameter': 10e-3},
        {'radius': -12e-3, 'conic': 0., 'glass_before': 'N-BK7',
         'glass_after': 'air', 'semi_diameter': 10e-3}],
        'thicknesses': [6e-3, 15e-3]}
    surfs = surfaces_from_prescription(p)
    x = np.array([0.0, 3e-3, 6e-3])
    z = np.zeros(3)
    ux = np.array([0.0, 0.5, 1.0])       # the last ray TIRs at the steep surface
    a = ray_transfer_jacobian_analytic(x, z, ux, z, surfs, LAM)
    f = ray_transfer_jacobian(x, z, ux, z, surfs, LAM)
    assert np.isfinite(a.jacobian).all()          # no NaN/inf anywhere
    assert not a.alive.all() and np.array_equal(a.alive, f.alive)
    live = a.alive
    # looser tol: the FD truncation floor grows at this steep (R=12mm) high-NA
    # geometry; the analytic is the exact value, the FD is the noisy one.
    assert np.max(np.abs((a.jacobian - f.jacobian)[live])) < 1e-5


def test_analytic_coddington_sagittal_tangential():
    """Single-surface physics cross-check: at oblique incidence the C-block
    (surface power) tangential/sagittal ratio tracks Coddington's
    1/(cos_i * cos_t), and reduces to the isotropic surface power on axis."""
    from lumenairy.glass import get_glass_index
    ng = float(get_glass_index('N-BK7', LAM))
    R = 40e-3
    surf = {'name': 'p', 'aperture_diameter': 40e-3, 'surfaces': [
        {'radius': R, 'conic': 0., 'glass_before': 'air',
         'glass_after': 'N-BK7', 'semi_diameter': 18e-3}],
        'thicknesses': [0.0]}
    surfs = surfaces_from_prescription(surf)
    for h in (2e-3, 5e-3, 9e-3):
        x = np.array([h])
        z = np.zeros(1)
        J = ray_transfer_jacobian_analytic(x, z, z, z, surfs, LAM).jacobian[0]
        Cxx = J[2, 0]      # tangential (x in the plane of incidence)
        Cyy = J[3, 1]      # sagittal
        i = np.arcsin(h / R)
        ct = np.sqrt(1.0 - (np.sin(i) / ng) ** 2)
        ratio = Cxx / Cyy
        assert abs(ratio - 1.0 / (np.cos(i) * ct)) < 0.01, (h, ratio)
    # on axis: isotropic (tangential == sagittal), the paraxial surface power
    z = np.zeros(1)
    J0 = ray_transfer_jacobian_analytic(z, z, z, z, surfs, LAM).jacobian[0]
    assert abs(J0[2, 0] - J0[3, 1]) < 1e-9


@pytest.mark.skipif(not _jax_ok(), reason='jax not installed')
def test_analytic_jax_differentiates_mirror_system():
    """Gradient-based design on a REFLECTIVE (is_mirror) system: the analytic
    JAX path differentiates through the reflection (the shared _adrt_step
    reflects on both backends), matching NumPy and giving finite jax.grad --
    the capability the paraxial trace_jax path lacks (it rejects mirrors).
    (Coord-break tilt folds remain a separate, world-frame extension.)"""
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    mir = {'name': 'm', 'aperture_diameter': 40e-3, 'surfaces': [
        {'radius': -40e-3, 'conic': 0., 'glass_before': 'air',
         'glass_after': 'air', 'is_mirror': True, 'semi_diameter': 18e-3},
        {'radius': float('inf'), 'conic': 0., 'glass_before': 'air',
         'glass_after': 'air', 'semi_diameter': 18e-3}],
        'thicknesses': [-20e-3, 0.0]}
    surfs = surfaces_from_prescription(mir)
    x = np.array([0.0, 5e-3, 10e-3])
    z = np.zeros(3)
    Jn = ray_transfer_jacobian_analytic(x, z, z, z, surfs, LAM).jacobian
    Jj = np.asarray(ray_transfer_jacobian_analytic(
        jnp.asarray(x), jnp.zeros(3), jnp.zeros(3), jnp.zeros(3),
        surfs, LAM).jacobian)
    assert np.max(np.abs(Jn - Jj)) < 1e-9

    def loss(xv):
        J = ray_transfer_jacobian_analytic(
            xv, jnp.zeros(3), jnp.zeros(3), jnp.zeros(3), surfs, LAM).jacobian
        return jnp.sum(J[:, 0, 0] ** 2)

    g = np.asarray(jax.grad(loss)(jnp.asarray(x)))
    assert np.isfinite(g).all() and g.shape == (3,)


@pytest.mark.skipif(not _jax_ok(), reason='jax not installed')
def test_analytic_jax_exact_and_differentiable():
    """On the JAX backend the analytic Jacobian equals the NumPy one (exact),
    and it is jax.grad / jit differentiable (a NumPy-native, truncation-free,
    conic/reflection differential transfer)."""
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    surfs = surfaces_from_prescription(_singlet())
    Jn = ray_transfer_jacobian_analytic(_X, _Y, _UX, _UY, surfs, LAM).jacobian
    Jj = ray_transfer_jacobian_analytic(
        jnp.asarray(_X), jnp.asarray(_Y), jnp.asarray(_UX), jnp.asarray(_UY),
        surfs, LAM).jacobian
    assert float(jnp.max(jnp.abs(jnp.asarray(Jn) - Jj))) < 1e-9

    def loss(xv):
        J = ray_transfer_jacobian_analytic(
            xv, jnp.zeros(3), jnp.zeros(3), jnp.zeros(3), surfs, LAM).jacobian
        return jnp.sum(J[:, 0, 0] ** 2)

    g = np.asarray(jax.grad(loss)(jnp.array([1e-3, 3e-3, 5e-3])))
    assert np.isfinite(g).all() and g.shape == (3,)
    assert np.isfinite(float(jax.jit(loss)(jnp.array([1e-3, 3e-3, 5e-3]))))


def test_analytic_coordbreak_tilt_decenter_matches_fd_and_differentiable():
    """A coordinate break (decenter + X/Y/Z tilts) in the surface list is a
    smooth frame transform: the analytic ray-transfer Jacobian through a
    tilted+decentered system matches the finite-difference primitive to its
    truncation floor, and is jax.grad-differentiable (the differentiable
    ray transfer through a fold -- alignment / tolerancing sensitivity)."""
    from lumenairy.raytrace.surface import Surface
    base = surfaces_from_prescription({
        'name': 's', 'aperture_diameter': 20e-3, 'surfaces': [
            {'radius': 51.5e-3, 'conic': 0., 'glass_before': 'air',
             'glass_after': 'N-BK7', 'semi_diameter': 10e-3},
            {'radius': -51.5e-3, 'conic': 0., 'glass_before': 'N-BK7',
             'glass_after': 'air', 'semi_diameter': 10e-3}],
        'thicknesses': [4e-3, 10e-3]})
    cb = Surface(is_coordbrk=True, tilt_x_deg=2.0, tilt_y_deg=1.0,
                 tilt_z_deg=0.5, decenter_x_m=0.3e-3, decenter_y_m=-0.2e-3,
                 thickness=15e-3, glass_before='air', glass_after='air')
    det = Surface(radius=np.inf, glass_before='air', glass_after='air',
                  thickness=0.0)
    surfs = [base[0], base[1], cb, det]
    x = np.array([0.0, 3e-3, -4e-3])
    y = np.array([0.0, 2e-3, 1e-3])
    z = np.zeros(3)
    a = ray_transfer_jacobian_analytic(x, y, z, z, surfs, LAM)
    f = ray_transfer_jacobian(x, y, z, z, surfs, LAM)
    assert np.max(np.abs(a.jacobian - f.jacobian)) < 1e-6      # FD floor
    assert np.max(np.abs(np.stack(
        [a.x - f.x, a.y - f.y, a.ux - f.ux, a.uy - f.uy]))) < 1e-10
    assert np.max(np.abs(a.opd - f.opd)) < 1e-12
    if _jax_ok():
        import jax
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        Jj = np.asarray(ray_transfer_jacobian_analytic(
            jnp.asarray(x), jnp.asarray(y), jnp.zeros(3), jnp.zeros(3),
            surfs, LAM).jacobian)
        assert np.max(np.abs(a.jacobian - Jj)) < 1e-9

        def loss(xv):
            J = ray_transfer_jacobian_analytic(
                xv, jnp.zeros(3), jnp.zeros(3), jnp.zeros(3), surfs,
                LAM).jacobian
            return jnp.sum(J[:, 0, 0] ** 2)
        g = np.asarray(jax.grad(loss)(jnp.asarray(x)))
        assert np.isfinite(g).all() and g.shape == (3,)
