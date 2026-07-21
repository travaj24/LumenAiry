"""Audit v5.17.0 Wave-6 raytrace cluster: P3-58 / P3-59 / P3-60.

P3-58: the JAX intersect kernels marked disc == 0 (exact tangency) as a
miss while the NumPy path accepts it (v5.4.6 audit P3-3), and the JAX
DOE kick used ``sumsq < 1.0`` where the NumPy trace loop uses the
strict evanescence test ``sumsq > 1.0``.  Both boundaries are now
aligned to the NumPy semantics (miss only on disc < 0; propagating on
sumsq <= 1.0); the sqrt double-where gradient guard keeps its strict
``disc > 0`` mask.

P3-59: the JAX aspheric Newton runs a FIXED 8 iterations (jit/grad
requirement) with no convergence tracking, silently accepting a
finite-but-unconverged t where the NumPy path kills the ray as
RAY_MISSED_SURFACE.  Both JAX kernels now evaluate the residual
F(t) = z - sag(x, y) once after the loop and fold
``|F| > _newton_residual_tol(dtype)`` into the miss mask (1e-12 in
float64, mirroring the NumPy stuck-with-residual criterion; scaled by
the dtype eps ratio under default float32 so converged rays are not
over-killed by the f32 rounding floor).

P3-60: ``_surface_copy_with`` dropped is_coordbrk / tilt_* / decenter_*
/ coordbrk_order / world_origin / world_R despite its docstring
claiming it propagates all optional fields; the clone now copies every
Surface field (pinned exhaustively via ``dataclasses.fields``).

Pre-fix: the P3-58/P3-59 parity tests and the P3-60 propagation test
FAIL (verified by the reproduction probe C:/tmp/w6_raytrace_probe.py);
off-boundary rays are byte-identical before/after.
"""
import dataclasses
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.raytrace.intersection import _intersect_surface
from lumenairy.raytrace.surface import (
    RAY_MISSED_SURFACE,
    RayBundle,
    Surface,
    _surface_copy_with,
)
from lumenairy.raytrace.trace import trace

WV = 633e-9


def _bundle(x, y, z, L, M, N, wl=WV):
    x = np.atleast_1d(np.asarray(x, float))
    n = len(x)

    def arr(v):
        a = np.atleast_1d(np.asarray(v, float))
        return np.full(n, a[0]) if len(a) == 1 else a

    return RayBundle(x=x, y=arr(y), z=arr(z), L=arr(L), M=arr(M),
                     N=arr(N), wavelength=wl,
                     alive=np.ones(n, bool), opd=np.zeros(n))


def _jax():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    return jax


# ---------------------------------------------------------------------------
# P3-58: disc == 0 tangency parity (both JAX intersect kernels)
# ---------------------------------------------------------------------------

def test_tangent_ray_survives_jax_intersect():
    """A ray grazing a sphere exactly (disc computes to 0.0 in float64)
    is accepted by the NumPy path since v5.4.6; the static JAX kernel
    must agree on BOTH alive and the landing point t = -b/2."""
    _jax()
    from lumenairy.raytrace.jax_trace import (
        _intersect_jax,
        make_jax_ray_state,
    )
    # x0 = R, direction +z: b = -2R, c = 0 -> disc = 4R^2 - 4R^2 == 0.0
    rb = _bundle(0.5, 0.0, 0.0, 0.0, 0.0, 1.0)
    _intersect_surface(rb, Surface(radius=0.5), n_medium=1.0)
    assert rb.alive[0] and rb.error_code[0] == 0

    st = make_jax_ray_state(np.array([0.5]), np.zeros(1), np.zeros(1),
                            np.zeros(1), np.zeros(1), np.ones(1))
    out = _intersect_jax(st, R=0.5, conic=0.0, asph_items=(), n_medium=1.0)
    assert bool(np.asarray(out.alive)[0]), \
        "JAX killed the disc==0 tangent ray the NumPy path accepts"
    # Same intersection: t = -b/2 lands at z = R.
    assert np.asarray(out.z)[0] == pytest.approx(rb.z[0], abs=1e-15)


def test_tangent_ray_survives_jax_intersect_param():
    _jax()
    import jax.numpy as jnp

    from lumenairy.raytrace.jax_trace import (
        _intersect_jax_param,
        make_jax_ray_state,
    )
    st = make_jax_ray_state(np.array([0.5]), np.zeros(1), np.zeros(1),
                            np.zeros(1), np.zeros(1), np.ones(1))
    out = _intersect_jax_param(st, jnp.asarray(0.5), jnp.asarray(0.0),
                               (), jnp.zeros(0), 1.0)
    assert bool(np.asarray(out.alive)[0])


def test_true_miss_still_dies_on_both_backends():
    """disc < 0 (ray passes outside the sphere) must still be a miss --
    the >= boundary change must not resurrect genuine misses."""
    _jax()
    from lumenairy.raytrace.jax_trace import (
        _intersect_jax,
        make_jax_ray_state,
    )
    rb = _bundle(0.6, 0.0, 0.0, 0.0, 0.0, 1.0)
    _intersect_surface(rb, Surface(radius=0.5), n_medium=1.0)
    assert not rb.alive[0] and rb.error_code[0] == RAY_MISSED_SURFACE

    st = make_jax_ray_state(np.array([0.6]), np.zeros(1), np.zeros(1),
                            np.zeros(1), np.zeros(1), np.ones(1))
    out = _intersect_jax(st, R=0.5, conic=0.0, asph_items=(), n_medium=1.0)
    assert not bool(np.asarray(out.alive)[0])


# ---------------------------------------------------------------------------
# P3-58: DOE-kick propagating boundary (sumsq == 1.0 exactly)
# ---------------------------------------------------------------------------

def _doe_case(period_scale):
    """order-1 grating with period = period_scale * wavelength:
    dL = 1/period_scale exactly."""
    surfaces = [Surface(radius=np.inf, thickness=0.0)]
    diff = {0: (1.0, 0.0, period_scale * WV, np.inf)}
    rb = _bundle(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    return trace(rb, surfaces, WV, surface_diffraction=diff), diff


def test_doe_boundary_order_parity_end_to_end():
    """Grating with period == wavelength -> L' = 1.0, sumsq == 1.0
    exactly: the grazing order propagates (N = 0) on the NumPy path and
    must now propagate on both JAX entry points too."""
    _jax()
    from lumenairy.raytrace.jax_trace import (
        make_jax_ray_state,
        trace_jax,
        trace_jax_with_params,
    )
    res_np, diff = _doe_case(1.0)
    assert res_np.image_rays.alive[0]
    assert res_np.image_rays.L[0] == 1.0

    rx = {'surfaces': [{'radius': float('inf'), 'glass_before': 'air',
                        'glass_after': 'air'}],
          'thicknesses': [0.0], 'aperture_diameter': 1.0}
    st = make_jax_ray_state(np.zeros(1), np.zeros(1), np.zeros(1),
                            np.zeros(1), np.zeros(1), np.ones(1))
    for fn in (trace_jax, trace_jax_with_params):
        out = fn(st, rx, WV, surface_diffraction=diff)
        assert bool(np.asarray(out.alive)[0]), \
            f"{fn.__name__} killed the sumsq==1.0 grazing order"
        assert np.asarray(out.L)[0] == pytest.approx(1.0, abs=0.0)
        assert np.asarray(out.N)[0] == pytest.approx(0.0, abs=0.0)


def test_doe_evanescent_order_still_dies_on_both_backends():
    """sumsq > 1.0 (period slightly below wavelength) remains evanescent
    on both backends -- the <= boundary must not admit true evanescence."""
    _jax()
    from lumenairy.raytrace.jax_trace import make_jax_ray_state, trace_jax
    res_np, diff = _doe_case(0.99)
    assert not res_np.image_rays.alive[0]

    rx = {'surfaces': [{'radius': float('inf'), 'glass_before': 'air',
                        'glass_after': 'air'}],
          'thicknesses': [0.0], 'aperture_diameter': 1.0}
    st = make_jax_ray_state(np.zeros(1), np.zeros(1), np.zeros(1),
                            np.zeros(1), np.zeros(1), np.ones(1))
    out = trace_jax(st, rx, WV, surface_diffraction=diff)
    assert not bool(np.asarray(out.alive)[0])


# ---------------------------------------------------------------------------
# P3-59: unconverged aspheric Newton must kill, converged must survive
# ---------------------------------------------------------------------------

def _steep_fan(n=8):
    """Oblique fan on a steep quartic asphere: NumPy Newton fails on
    every ray (RAY_MISSED_SURFACE); pre-fix JAX kept them all alive
    with up to ~3e4 m landing residual."""
    h = np.linspace(1e-3, 20e-3, n)
    L = 0.9
    N = np.sqrt(1.0 - L * L)
    return h, L, N


def test_unconverged_newton_killed_static_kernel():
    _jax()
    from lumenairy.raytrace.jax_trace import (
        _intersect_jax,
        make_jax_ray_state,
    )
    h, L, N = _steep_fan()
    n = len(h)
    rb = _bundle(h, 0.0, 0.0, L, 0.0, N)
    surf = Surface(radius=np.inf, aspheric_coeffs={4: 1e8})
    _intersect_surface(rb, surf, n_medium=1.0)
    assert not rb.alive.any()
    assert (rb.error_code == RAY_MISSED_SURFACE).all()

    st = make_jax_ray_state(h, np.zeros(n), np.zeros(n),
                            np.full(n, L), np.zeros(n), np.full(n, N))
    out = _intersect_jax(st, R=np.inf, conic=0.0,
                         asph_items=((4, 1e8),), n_medium=1.0)
    assert not np.asarray(out.alive).any(), \
        "JAX accepted unconverged Newton rays the NumPy path kills"


def test_unconverged_newton_killed_param_kernel():
    _jax()
    import jax.numpy as jnp

    from lumenairy.raytrace.jax_trace import (
        _intersect_jax_param,
        make_jax_ray_state,
    )
    h, L, N = _steep_fan()
    n = len(h)
    st = make_jax_ray_state(h, np.zeros(n), np.zeros(n),
                            np.full(n, L), np.zeros(n), np.full(n, N))
    out = _intersect_jax_param(st, jnp.asarray(np.inf), jnp.asarray(0.0),
                               (4,), jnp.asarray([1e8]), 1.0)
    assert not np.asarray(out.alive).any()


def test_convergent_asphere_unaffected_and_matches_numpy():
    """A typical optical asphere converges well within 8 iterations:
    the new residual kill must not fire, and landing points must match
    the NumPy backend."""
    _jax()
    from lumenairy.raytrace.jax_trace import (
        _intersect_jax,
        make_jax_ray_state,
    )
    n = 9
    h = np.linspace(-4e-3, 4e-3, n)
    rb = _bundle(h, 0.0, 0.0, 0.0, 0.0, 1.0)
    surf = Surface(radius=0.05, conic=-0.6, aspheric_coeffs={4: 1e2, 6: 1e5})
    _intersect_surface(rb, surf, n_medium=1.0)
    assert rb.alive.all()

    st = make_jax_ray_state(h, np.zeros(n), np.zeros(n),
                            np.zeros(n), np.zeros(n), np.ones(n))
    out = _intersect_jax(st, R=0.05, conic=-0.6,
                         asph_items=((4, 1e2), (6, 1e5)), n_medium=1.0)
    assert np.asarray(out.alive).all()
    np.testing.assert_allclose(np.asarray(out.z), rb.z, rtol=0, atol=1e-12)


def test_residual_check_is_grad_safe():
    """The post-loop residual evaluation only feeds a boolean mask;
    jax.grad through an aspheric trace must stay finite (trace-safe)."""
    jax = _jax()
    import jax.numpy as jnp

    from lumenairy.raytrace.jax_trace import (
        _intersect_jax_param,
        make_jax_ray_state,
    )

    st = make_jax_ray_state(np.array([2e-3]), np.zeros(1), np.zeros(1),
                            np.zeros(1), np.zeros(1), np.ones(1))

    def landing_z(radius):
        out = _intersect_jax_param(st, radius, jnp.asarray(-0.5),
                                   (4,), jnp.asarray([1e2]), 1.0)
        return out.z[0]

    g = jax.grad(landing_z)(jnp.asarray(0.05))
    assert np.isfinite(float(g))


# ---------------------------------------------------------------------------
# P3-60: _surface_copy_with propagates EVERY Surface field
# ---------------------------------------------------------------------------

# One distinct, non-default value per Surface field.  The exhaustiveness
# assertion below makes this test fail loudly if Surface ever grows a
# field that is not listed here -- update BOTH this dict and
# _surface_copy_with when that happens.
def _W6_FIELD_SAG(xs, ys):
    # Stable module-level callable so copy_with propagation is identity-checkable.
    return 1e-6 * (xs + ys)


_NON_DEFAULT = {
    'radius': 0.123,
    'conic': -0.5,
    'aspheric_coeffs': {4: 1.5e2},
    'semi_diameter': 7e-3,
    'glass_before': 'N-BK7',
    'glass_after': 'N-SF11',
    'is_mirror': True,
    'is_stop': True,
    'thickness': 0.025,
    'label': 'w6-clone-probe',
    'surf_num': 17,
    'is_coordbrk': True,
    'tilt_x_deg': 5.0,
    'tilt_y_deg': -2.0,
    'tilt_z_deg': 1.25,
    'decenter_x_m': 1e-3,
    'decenter_y_m': -2e-3,
    'coordbrk_order': 1,
    'radius_y': 0.456,
    'conic_y': -1.0,
    'aspheric_coeffs_y': {6: -3.0e4},
    'freeform': {'kind': 'xy_polynomial', 'coefficients': {(2, 1): 1e-4}},
    'field_decenter': (1e-3, -2e-3),
    'field_tilt': (5e-4, -3e-4),
    'field_sag_callable': _W6_FIELD_SAG,
    'bsdf': object(),
    'coating': (1.374 + 7.62j),
    'world_origin': np.array([0.1, -0.2, 1.0]),
    'world_R': np.array([[0.0, -1.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0]]),
}


def _field_equal(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    return a == b


def test_surface_copy_with_propagates_every_field():
    field_names = {f.name for f in dataclasses.fields(Surface)}
    assert field_names == set(_NON_DEFAULT), (
        "Surface grew/lost fields -- update _NON_DEFAULT and "
        "_surface_copy_with together")
    surf = Surface(**_NON_DEFAULT)
    clone = _surface_copy_with(surf)
    dropped = [name for name in field_names
               if not _field_equal(getattr(clone, name), getattr(surf, name))]
    assert not dropped, f"_surface_copy_with dropped fields: {dropped}"


def test_surface_copy_with_overrides_still_apply():
    surf = Surface(**_NON_DEFAULT)
    clone = _surface_copy_with(surf, thickness=0.5, is_coordbrk=False,
                               world_origin=None)
    assert clone.thickness == 0.5
    assert clone.is_coordbrk is False
    assert clone.world_origin is None
    # ...without disturbing the untouched fields.
    assert clone.tilt_x_deg == surf.tilt_x_deg
    assert clone.coordbrk_order == surf.coordbrk_order
    assert np.array_equal(clone.world_R, surf.world_R)


def test_cloned_coordbrk_stays_a_coordbrk_in_trace():
    """The audit's latent tripwire: a cloned coord-break must still be
    dispatched as a frame transform (skip intersect/refract), not as a
    refracting flat."""
    cb = Surface(is_coordbrk=True, tilt_x_deg=10.0, thickness=0.1)
    img = Surface(radius=np.inf, thickness=0.0)
    rb = _bundle(0.0, 1e-3, 0.0, 0.0, 0.0, 1.0)
    res_orig = trace(rb, [cb, img], WV)
    res_clone = trace(rb, [_surface_copy_with(cb), img], WV)
    np.testing.assert_array_equal(res_clone.image_rays.y,
                                  res_orig.image_rays.y)
    np.testing.assert_array_equal(res_clone.image_rays.N,
                                  res_orig.image_rays.N)
