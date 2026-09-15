"""WAVE5-E item E5 -- VERIFY-WP-B12's two open items on the exit-vertex plane.

**O-1 -- the projection moved DEAD rays; ``exit_vertex_transfer`` freezes them.**
``lumenairy.raytrace.exit_vertex.exit_vertex_transfer`` is explicit that a
vignetted ray keeps its position, direction and OPL "exactly", because it never
reached the vertex plane.  ``differential._project_to_exit_vertex_plane``
applied its arithmetic to every row.  Measured on a fan clipped at the last
surface, over the eight VERIFY-WP-B12 fixture classes x three backends
(``validation/probe_wave5_e/e5_prepost_*.json``, 2026-09-15, both builds):

    dead-row ``opd`` drift   1.37e-05 .. 1.87e-04 m   (17 of 30 cells)
    dead-row Jacobian drift  1.37e-05 .. 1.53e+300    (the analytic backend's
                                                       dead rows extrapolate)
    ALIVE rows               byte-identical before and after the fix,
                             30 of 30 cells, both builds

It was unobservable through ``fga.py`` only because all four consumers zero the
dead beamlets first -- a divergence between the module's two vertex-plane
operators, which the next consumer would not know about.

**O-3 -- the FGA image leg carries no exit index.**  Each FGA transport adds
``opd += z_image * sqrt(1 + ux^2 + uy^2)`` by hand after asking the projection
for ``reference='exit_vertex'``.  The projection resolves ``n_exit`` and weights
its sag term with it; the leg does not, so it assumes the exit medium is air.
Every FGA fixture in the suite ends in air, so it is unreachable today.  It is
now REFUSED rather than served silently, at the four sites, with the tolerance
derived from the wavefront error the omission costs.

Both claims are two-sided: the freeze does not touch a live bit (and the live
rows still move, so the projection is still doing its job), and the guard does
not refuse an air-terminated prescription.
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy import raytrace as rt
from lumenairy.propagators import fga as _fga
from lumenairy.raytrace import surfaces_from_prescription
from lumenairy.raytrace.differential import (
    DifferentialTransfer,
    _project_to_exit_vertex_plane,
    ray_transfer_jacobian,
    ray_transfer_jacobian_analytic,
)
from lumenairy.raytrace.exit_vertex import resolve_exit_index

_LAM = 1.03e-6
_SEMI = 0.15e-3
_T = 0.55e-3
_GLASS = 'N-SSK8'
#: The fan runs this far past the clear aperture, so most of it vignettes.
#: 3.0 is not a tuned number: at 1.0 the rim rays are exactly at the aperture
#: and whether they survive is a round-off decision, and past ~2 every cell
#: already vignettes two thirds of the fan (measured 134 dead of 201 at 3.0 on
#: all eight fixtures, both builds).
_OVER = 3.0
_N_RAYS = 121


def _presc(last, glass_after='air', glass=_GLASS, t=_T, semi=_SEMI):
    """A plano-first singlet whose LAST surface is whatever ``last`` says."""
    s1 = {'radius': np.inf, 'conic': 0.0, 'thickness': t,
          'glass_before': 'air', 'glass_after': glass, 'semi_diameter': semi}
    s2 = {'conic': 0.0, 'thickness': 0.0, 'glass_before': glass,
          'glass_after': glass_after, 'semi_diameter': semi}
    s2.update(last)
    return {'name': 'wave5e', 'aperture_diameter': 2 * semi,
            'surfaces': [s1, s2], 'thicknesses': [t], 'stop_index': 0}


#: The surface classes VERIFY-WP-B12 built its oracle around.  ``flat`` is the
#: control: its sag vanishes, so the projection short-circuits and the freeze
#: is a no-op there by construction -- which is itself worth pinning.
_CLASSES = {
    'conic': {'radius': -0.90e-3, 'conic': -0.60},
    'asphere': {'radius': -0.911e-3, 'conic': -0.60,
                'aspheric_coeffs': {4: 2.0e9, 6: -4.0e16}},
    'biconic': {'radius': -0.858e-3, 'radius_y': -1.17e-3,
                'conic': 0.0, 'conic_y': 0.0},
    'mirror': {'radius': -2.5e-3, 'conic': 0.0, 'glass_after': 'MIRROR'},
    'oblique': {'radius': -1.40e-3, 'conic': 0.0},     # traced at a tilt below
    'flat': {'radius': np.inf, 'conic': 0.0},
}

#: The classes the ANALYTIC backend supports (it refuses biconic by contract).
_ANALYTIC_CLASSES = ('conic', 'asphere', 'mirror', 'oblique', 'flat')


def _class_presc(name):
    last = dict(_CLASSES[name])
    ga = last.pop('glass_after', 'air')
    return _presc(last, glass_after=ga)


def _surfs(presc):
    s = [copy.copy(x) for x in surfaces_from_prescription(presc)]
    s[-1].thickness = 0.0
    return s


def _fan(name):
    """A clipped fan; the ``oblique`` class is launched at a 6 degree tilt."""
    h = np.linspace(-_OVER * _SEMI, _OVER * _SEMI, _N_RAYS)
    z = np.zeros(_N_RAYS)
    ux = (np.full(_N_RAYS, np.tan(np.deg2rad(6.0))) if name == 'oblique'
          else z.copy())
    return h, z.copy(), ux, z.copy()


def _transfers(name, backend):
    """``(surface-referenced, vertex-referenced)`` on one backend."""
    surfs = _surfs(_class_presc(name))
    h, y, ux, uy = _fan(name)
    fn = (ray_transfer_jacobian if backend == 'fd'
          else ray_transfer_jacobian_analytic)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        srf = fn(h.copy(), y.copy(), ux.copy(), uy.copy(), surfs, _LAM,
                 reference='surface')
        vtx = fn(h.copy(), y.copy(), ux.copy(), uy.copy(), surfs, _LAM,
                 reference='exit_vertex')
    return surfs, srf, vtx


def _arr(v):
    return np.asarray(v, dtype=np.float64)


# ===========================================================================
# O-1.  The projection is the identity on dead rows
# ===========================================================================
@pytest.mark.parametrize('backend', ['fd', 'analytic'])
@pytest.mark.parametrize('name', sorted(_CLASSES))
def test_the_projection_freezes_dead_rays_and_moves_live_ones(name, backend):
    """TWO-SIDED.  On a fan clipped at the last surface the projection must
    leave every DEAD row exactly where it was -- ``x``, ``y``, ``opd`` and the
    Jacobian, bit for bit -- and must still move the LIVE ones, which is what
    makes this a freeze and not a disabled projection.

    The premise (that the fan really vignettes) is ASSERTED, not skipped: a fan
    that stopped clipping would make this test vacuous rather than red, which
    is the failure mode ``docs/TESTING_STANDARDS.md`` rule 4 is about.

    ``flat`` is the control.  Its sag vanishes identically, so the projection
    short-circuits and returns its input object; dead AND live rows are frozen
    there, and the live-motion arm is therefore not asserted for it.
    """
    if backend == 'analytic' and name not in _ANALYTIC_CLASSES:
        with pytest.raises(NotImplementedError, match='biconic'):
            _transfers(name, backend)
        return
    _surfaces, srf, vtx = _transfers(name, backend)
    alive = np.asarray(srf.alive, dtype=bool)
    dead = ~alive

    assert dead.sum() >= _N_RAYS // 4, (
        f'premise: the {name} fan at {_OVER}x the clear aperture must vignette '
        f'at least a quarter of its {_N_RAYS} rays for this test to say '
        f'anything; only {int(dead.sum())} died.  Widen the fan or shrink the '
        f'aperture -- do not skip.')
    assert alive.sum() >= 8, (
        f'premise: the {name} fan must also keep live rays (got '
        f'{int(alive.sum())}), else the live-motion arm is vacuous.')
    assert np.array_equal(alive, np.asarray(vtx.alive, dtype=bool)), (
        'the projection must not change any ray\'s alive flag')

    for field in ('x', 'y', 'ux', 'uy', 'opd'):
        a = _arr(getattr(srf, field))[dead]
        b = _arr(getattr(vtx, field))[dead]
        same = np.array_equal(a.view(np.uint64), b.view(np.uint64))
        assert same, (
            f'{name}/{backend}: the projection moved DEAD rays\' {field} by up '
            f'to {float(np.nanmax(np.abs(a - b))):.3e}.  A vignetted ray never '
            f'reached the last surface, so exit_vertex_transfer freezes it '
            f'"exactly" and this operator must agree.')
    ja = _arr(srf.jacobian)[dead]
    jb = _arr(vtx.jacobian)[dead]
    assert np.array_equal(ja.view(np.uint64), jb.view(np.uint64)), (
        f'{name}/{backend}: the projection moved DEAD rays\' Jacobian rows by '
        f'up to {float(np.nanmax(np.abs(ja - jb))):.3e}')

    # -- the other side: the LIVE rows must still be projected ---------------
    if name == 'flat':
        assert np.array_equal(_arr(srf.opd)[alive], _arr(vtx.opd)[alive]), (
            'a FLAT last surface has no sag, so the projection must be the '
            'identity on live rows too (the short-circuit)')
        return
    d_opd = float(np.nanmax(np.abs(_arr(srf.opd)[alive]
                                   - _arr(vtx.opd)[alive])))
    d_x = float(np.nanmax(np.abs(_arr(srf.x)[alive] - _arr(vtx.x)[alive])))
    assert d_opd > 100.0 * _LAM * np.finfo(float).eps, (
        f'{name}/{backend}: the projection must still MOVE live rays; their '
        f'opd moved only {d_opd:.3e} m.  A freeze that froze everything would '
        f'pass the arms above and be a disabled projection.')
    assert d_x > 0.0, (
        f'{name}/{backend}: live rays\' x did not move ({d_x:.3e} m)')


@pytest.mark.parametrize('name', sorted(_CLASSES))
def test_the_frozen_dead_state_is_the_ray_bundle_operators_own(name):
    """The operator this is being brought into line with.

    ``TraceResult.at_exit_vertex()`` runs ``exit_vertex_transfer``, which
    freezes a dead ray's position and OPL.  The differential projection's dead
    rows must therefore equal the SURFACE-referenced state -- exactly what the
    bundle operator leaves behind -- on the same clipped fan.  Measured on both
    operators here rather than assumed of either.
    """
    surfs, srf, vtx = _transfers(name, 'fd')
    h, y, ux, uy = _fan(name)
    n = _N_RAYS
    nz = 1.0 / np.sqrt(1.0 + ux ** 2 + uy ** 2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        b = rt.RayBundle(x=h.copy(), y=y.copy(), z=np.zeros(n),
                         L=ux * nz, M=uy * nz, N=nz, wavelength=_LAM,
                         alive=np.ones(n, bool), opd=np.zeros(n))
        res = rt.trace(b, surfs, _LAM)
        ev = res.at_exit_vertex()
    img = res.image_rays
    bdead = ~np.asarray(img.alive, dtype=bool)
    assert bdead.sum() > 0, (
        f'premise: the {name} bundle trace must vignette some rays; none did')
    for field in ('x', 'y', 'opd'):
        a = _arr(getattr(img, field))[bdead]
        c = _arr(getattr(ev, field))[bdead]
        assert np.array_equal(a.view(np.uint64), c.view(np.uint64)), (
            f'{name}: at_exit_vertex moved a DEAD ray\'s {field} -- the '
            f'reference this file holds the differential projection to is '
            f'itself broken')
    dead = ~np.asarray(srf.alive, dtype=bool)
    assert np.array_equal(_arr(vtx.opd)[dead].view(np.uint64),
                          _arr(srf.opd)[dead].view(np.uint64)), (
        f'{name}: the differential projection and the bundle operator must '
        f'freeze dead rays the same way')


def test_the_jax_twin_of_the_projection_is_the_same_map():
    """ONE implementation, two array namespaces.

    ``_project_to_exit_vertex_plane`` branches on whether its input is a NumPy
    array; the mask must reach both arms identically.  The JAX path of
    ``ray_transfer_jacobian_analytic`` returns an all-alive mask on these
    fixtures (measured: 0 dead of 201 where the FD bundle tracer vignettes 134),
    so the twin is exercised where it is meaningful -- on the primitive itself,
    with ONE transfer that carries real vignetting projected twice, once as
    NumPy and once as ``jax.numpy``.

    Measured 2026-09-15 on both builds: the state is bit-identical on all eight
    VERIFY-WP-B12 fixtures and the Jacobians agree to 0 ULP, with x64 enabled
    (JAX's float32 default alone moves the state by ~1e-12, and 3e-5 on the
    mirror -- which is a dtype statement, not a mask statement).
    """
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    for name in sorted(_CLASSES):
        surfs, srf, _vtx = _transfers(name, 'fd')
        alive = np.asarray(srf.alive, dtype=bool)
        assert (~alive).sum() > 0, f'premise: {name} vignetted nothing'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            np_out = _project_to_exit_vertex_plane(
                srf, surfs, _LAM, None, 'wave5e_twin')
            jx_out = _project_to_exit_vertex_plane(
                DifferentialTransfer(
                    jacobian=jnp.asarray(srf.jacobian),
                    x=jnp.asarray(srf.x), y=jnp.asarray(srf.y),
                    ux=jnp.asarray(srf.ux), uy=jnp.asarray(srf.uy),
                    opd=jnp.asarray(srf.opd), alive=jnp.asarray(alive)),
                surfs, _LAM, None, 'wave5e_twin')
        for field in ('x', 'y', 'ux', 'uy', 'opd'):
            a = _arr(getattr(np_out, field))
            b = _arr(getattr(jx_out, field))
            assert np.array_equal(a.view(np.uint64), b.view(np.uint64)), (
                f'{name}: the JAX twin\'s {field} differs from NumPy\'s by '
                f'{float(np.nanmax(np.abs(a - b))):.3e}')
        # and the freeze itself, on the JAX arm
        dead = ~alive
        for field in ('x', 'y', 'opd'):
            a = _arr(getattr(srf, field))[dead]
            b = _arr(getattr(jx_out, field))[dead]
            assert np.array_equal(a.view(np.uint64), b.view(np.uint64)), (
                f'{name}: the JAX arm did not freeze dead rays\' {field}')


# ===========================================================================
# O-3.  The FGA image leg refuses an immersed exit
# ===========================================================================
_IMMERSED_LAST = {'radius': -0.90e-3, 'conic': -0.60}


def _immersed_presc():
    """A prescription whose exit medium is GLASS, not air.

    Built through the same ``_presc`` helper as every other fixture here, with
    ``glass_after`` naming a catalogue glass on the LAST surface -- the state
    ``resolve_exit_index`` reads.
    """
    return _presc(_IMMERSED_LAST, glass_after='N-SSK8')


def _air_presc():
    return _presc(_IMMERSED_LAST, glass_after='air')


def _E(n=64):
    xs = (np.arange(n) - n // 2) * 6.0e-6
    X, Y = np.meshgrid(xs, xs)
    return np.exp(-(X ** 2 + Y ** 2) / (40e-6) ** 2).astype(np.complex128)


def test_the_immersed_exit_premise_is_reachable():
    """PREMISE for the four guard arms, measured rather than assumed.

    The guard can only be reached if the fixture's exit index really is not 1.
    ``get_glass_index('air', lambda)`` returns EXACTLY 1.0 on this registry at
    every wavelength measured, so the air control is exactly at the tolerance's
    origin and the glass arm is 0.5-ish away from it.
    """
    surfs_air = _surfs(_air_presc())
    surfs_imm = _surfs(_immersed_presc())
    n_air = float(resolve_exit_index(surfs_air, _LAM, fn_name='wave5e'))
    n_imm = float(resolve_exit_index(surfs_imm, _LAM, fn_name='wave5e'))
    assert n_air == 1.0, (
        f'premise: the air control must resolve to exactly 1.0 so the guard '
        f'cannot refuse it on rounding; got {n_air!r}')
    assert n_imm > 1.4, (
        f'premise: the immersed fixture must resolve to a real glass index; '
        f'got {n_imm!r} -- the guard would be unreachable and every arm below '
        f'vacuous')


@pytest.mark.parametrize('site', ['through_lens', 'coarse', 'vector',
                                  'caustic_zone'])
def test_every_fga_site_refuses_an_immersed_exit(site):
    """FAIL-BEFORE, at each of the four sites VERIFY-WP-B12 enumerates.

    The image leg is ``opd += z_image * sqrt(1 + ux^2 + uy^2)`` -- no exit
    index -- while the projection that produced ``dt.opd`` resolved one.  In a
    medium of index ``n`` that costs ``|n-1| * z_image * sec`` of optical path,
    at least ``|n-1| * |z_image| / lambda`` waves.  Serving it silently would
    hand back a wavefront wrong by 2.9e+02 waves on this fixture
    (n = 1.617, z_image = 0.30 mm, lambda = 1.03 um); it is refused instead.

    ``_caustic_zone`` adds no leg of its own, but the distance it returns IS
    the ``output_plane_distance`` the other three then run their index-free leg
    over, so it is refused too -- at the tolerance's zero-leg floor.
    """
    presc = _immersed_presc()
    E = _E()
    kw = dict(prescription=presc, wavelength=_LAM, dx=6.0e-6,
              output_plane_distance=0.30e-3)
    with pytest.raises(NotImplementedError, match='IMMERSED'):
        if site == 'through_lens':
            la.apply_real_lens_fga(E, **kw)
        elif site == 'coarse':
            la.apply_real_lens_fga(E, coarse_stride=3, **kw)
        elif site == 'vector':
            la.apply_real_lens_fga_vector(
                    np.stack([E, E * 0.5]), **kw)
        else:
            _fga._caustic_zone(E, 6.0e-6, presc, _LAM)


@pytest.mark.parametrize('site', ['through_lens', 'coarse', 'vector',
                                  'caustic_zone'])
def test_an_air_terminated_prescription_is_not_refused(site):
    """THE OTHER SIDE.  The guard must not fire on the prescriptions the
    library actually serves -- every FGA fixture in the suite ends in air.
    Only the guard is under test here, so the call's own numerical outcome is
    not asserted; what is asserted is that no ``NotImplementedError`` naming
    an immersed exit escapes it.
    """
    presc = _air_presc()
    E = _E()
    kw = dict(prescription=presc, wavelength=_LAM, dx=6.0e-6,
              output_plane_distance=0.30e-3)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if site == 'through_lens':
                la.apply_real_lens_fga(E, **kw)
            elif site == 'coarse':
                la.apply_real_lens_fga(E, coarse_stride=3, **kw)
            elif site == 'vector':
                la.apply_real_lens_fga_vector(
                    np.stack([E, E * 0.5]), **kw)
            else:
                _fga._caustic_zone(E, 6.0e-6, presc, _LAM)
    except NotImplementedError as exc:                      # pragma: no cover
        if 'IMMERSED' in str(exc):
            raise AssertionError(
                f'the immersed-exit guard refused an AIR-terminated '
                f'prescription at the {site} site: {exc}') from exc
        raise


def test_the_guard_tolerance_is_the_wavefront_it_protects():
    """The tolerance is DERIVED, and this pins the derivation rather than a
    number.

    ``_require_non_immersed_exit`` refuses when
    ``|n - 1| > waves * lambda / max(|z_image|, lambda)``, i.e. exactly when
    the OPL the leg omits reaches ``waves`` of wavefront (``sec >= 1`` makes
    that a lower bound).  Two consequences are asserted:

      * a longer leg tightens the tolerance in proportion -- the same index
        error costs more waves over more distance;
      * at a zero-length leg the tolerance floors at the budget itself, which
        is 3.6x the air-vs-vacuum index difference at STP (2.77e-4) and ~500x
        below any immersion medium, so neither a caller who registers a real
        air index nor ``_caustic_zone``'s zero leg is decided by round-off.
    """
    waves = _fga._FGA_IMAGE_LEG_WAVE_BUDGET
    lam = _LAM

    def tol(z):
        return max(_fga._FGA_EXIT_INDEX_NOISE_FLOOR,
                   waves * lam / max(abs(z), lam))

    assert tol(0.0) == pytest.approx(waves), (
        'at a zero-length leg the tolerance must floor at the wave budget')
    assert tol(1.0e-3) == pytest.approx(waves * lam / 1.0e-3), (
        'the tolerance must scale as lambda / z_image')
    assert tol(2.0e-3) < tol(1.0e-3), (
        'a longer leg must tighten the tolerance')
    stp_air_minus_vacuum = 2.77e-4
    assert tol(0.0) > 3.0 * stp_air_minus_vacuum, (
        f'the zero-leg floor ({tol(0.0):.3e}) must sit above the air-vs-vacuum '
        f'index difference at STP ({stp_air_minus_vacuum:.3e}), else a caller '
        f'who registers a real air index is refused')
    assert tol(0.0) < 0.1 * (1.33 - 1.0), (
        f'the zero-leg floor ({tol(0.0):.3e}) must sit far below the weakest '
        f'immersion medium (water, n = 1.33), else the guard misses what it '
        f'exists for')
