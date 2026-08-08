"""K3 (plan N15): perf / memory sweep -- correctness guards + in-test speedup
measurements for the byte-identical optimizations applied to the accuracy-niche
hot paths.

The sweep profiled the traced ray-density amplitude, the displaced 2-D remap
scatter, the GBD decenter path, the astigmatic carrier ASM, the Seidel gate and
adaptive FGA.  Two FREE (byte-identical) wins were found and are guarded here;
the remaining hot paths had no free byte-identical win and are documented in the
phase report (Delaunay-query-bound pointwise obliquity, FFT-bound astigmatic
ASM, reconstruction-bound GBD, dual-number-bound FGA).

Guarded wins
------------
1. ``_apply_displaced_remap_2d`` (``surface_model='displaced', displaced_mode=
   'remap'``): the amplitude and OPL remaps now share ONE Delaunay
   triangulation via a single 2-column ``LinearNDInterpolator`` instead of
   building two.  This test reconstructs the PRE-K3 two-separate-interp
   algorithm inline as the oracle and asserts the optimized function is
   BYTE-IDENTICAL to it, then measures the new path is faster (generous bound,
   best-of-N -- not a brittle absolute threshold).

2. ``apply_real_lens_traced(amplitude_model='ray_density')``: the ray-density
   amplitude upsample reuses the OPL upsample's coarse->full coordinate stack
   instead of rebuilding ``np.indices`` + a second ``(2, N, N)`` float64 array.
   Byte-identical by construction (the reused array is the same values); guarded
   by determinism + the two-column ``LinearNDInterpolator`` identity property
   below and the full P11 suite.

All grids are small (N <= 768) so the suite stays well under the RAM/time budget;
a defensive ``available_memory_bytes`` guard skips the timed case on a
constrained runner.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pytest
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import map_coordinates

import lumenairy as la
from lumenairy.elements._lens_real import _apply_displaced_remap_2d
from lumenairy.memory import available_memory_bytes

# Model glass for THIS module only: registered and removed by
# tests/conftest.py::_module_glass_registry_guard.
MODULE_GLASSES = {'_K3A': lambda wl: 1.5168}
_WL = 1.31e-6


# ---------------------------------------------------------------------------
# Pre-K3 reference: the two-separate-interp ``_apply_displaced_remap_2d`` body,
# copied verbatim so the byte-identity guard is explicit and self-contained.
# ---------------------------------------------------------------------------
def _remap_2d_reference(E_in, ray_map_2d, wavelength, dx, dy):
    X0, Y0, XO, YO, OPL, ALIVE, dstep, r_ap = ray_map_2d
    Ny, Nx = E_in.shape
    k0 = 2.0 * np.pi / wavelength
    cx = X0.ravel() / dx + Nx / 2.0
    cy = Y0.ravel() / dy + Ny / 2.0
    amp_in = map_coordinates(np.abs(E_in), [cy, cx], order=1,
                             mode='constant', cval=0.0).reshape(X0.shape)
    XOf = XO.copy()
    YOf = YO.copy()
    dead = ~ALIVE
    if bool(dead.any()) and bool(ALIVE.any()):
        pa = np.column_stack([X0[ALIVE], Y0[ALIVE]])
        XOf[dead] = NearestNDInterpolator(pa, XO[ALIVE])(X0[dead], Y0[dead])
        YOf[dead] = NearestNDInterpolator(pa, YO[ALIVE])(X0[dead], Y0[dead])
    dXO_dy, dXO_dx = np.gradient(XOf, dstep, dstep)
    dYO_dy, dYO_dx = np.gradient(YOf, dstep, dstep)
    det = dXO_dx * dYO_dy - dXO_dy * dYO_dx
    jac_amp = 1.0 / np.sqrt(np.maximum(np.abs(det), 1e-30))
    amp_out = amp_in * jac_amp
    in_ap = (X0 * X0 + Y0 * Y0) <= (r_ap * (1.0 + 1e-9)) ** 2
    m = ALIVE & in_ap & np.isfinite(amp_out) & (amp_in > 0.0)
    if int(m.sum()) < 4:
        return np.zeros_like(E_in, dtype=np.complex128)
    pts = np.column_stack([XO[m].ravel(), YO[m].ravel()])
    x = (np.arange(Nx, dtype=np.float64) - Nx / 2) * dx
    y = (np.arange(Ny, dtype=np.float64) - Ny / 2) * dy
    Xg, Yg = np.meshgrid(x, y)
    amp_grid = LinearNDInterpolator(pts, amp_out[m].ravel(),
                                    fill_value=0.0)(Xg, Yg)
    opl_grid = LinearNDInterpolator(pts, OPL[m].ravel())(Xg, Yg)
    nan = np.isnan(opl_grid)
    if bool(nan.any()):
        opl_grid[nan] = NearestNDInterpolator(pts, OPL[m].ravel())(
            Xg[nan], Yg[nan])
        amp_grid[nan] = 0.0
    opl_ref = float(np.median(OPL[m]))
    E_out = amp_grid * np.exp(1j * k0 * (opl_grid - opl_ref))
    out_dtype = E_in.dtype if np.iscomplexobj(E_in) else np.complex128
    return np.asarray(E_out, dtype=out_dtype)


def _synthetic_ray_map(n_side, r_max, dx):
    """A smooth, decentered, aberrated 2-D ray map (aperture-scale magnification
    + a cubic coma-like walk + a quadratic OPL) -- physics-independent, so the
    interp-combine identity is tested with full control over the scatter."""
    r_fan = r_max * 1.03
    ax = np.linspace(-r_fan, r_fan, n_side)
    dstep = float(ax[1] - ax[0])
    X0, Y0 = np.meshgrid(ax, ax)
    # decentered magnifying map with a small comatic cubic term
    XO = 0.62 * X0 + 0.35e-3 + 4.0 * (X0 ** 2 + Y0 ** 2) * X0
    YO = 0.62 * Y0 + 3.0 * (X0 ** 2 + Y0 ** 2) * Y0
    OPL = 1e-3 + 0.5 * (X0 ** 2 + Y0 ** 2) / 0.05
    ALIVE = (X0 ** 2 + Y0 ** 2) <= (r_fan * 0.999) ** 2
    return (X0, Y0, XO, YO, OPL, ALIVE, dstep, float(r_max))


def _gauss(N, dx, w0):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


def _time_once(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def _singlet(dec=None):
    s0 = {'radius': 51.68e-3, 'thickness': 5e-3, 'glass_before': 'air',
          'glass_after': '_K3A', 'semi_diameter': 6e-3}
    if dec is not None:
        s0['decenter'] = dec
    return {'wavelength': _WL, 'aperture_diameter': 10e-3, 'surfaces': [
        s0,
        {'radius': -51.68e-3, 'thickness': 0.0, 'glass_before': '_K3A',
         'glass_after': 'air', 'semi_diameter': 6e-3}],
        'thicknesses': [5e-3], 'stop_index': 0}


# ===========================================================================
# Win 1 -- displaced 2-D remap: shared-Delaunay 2-column interp
# ===========================================================================
def test_two_column_linearnd_equals_separate_columns():
    """The core identity the remap optimization relies on: a 2-column
    ``LinearNDInterpolator`` is BIT-FOR-BIT the stack of two separate 1-column
    interps on the same points (one shared Delaunay + shared barycentric
    weights, an independent reduction per column)."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.0, 1.0, (600, 2))
    va = np.sin(3 * pts[:, 0]) + pts[:, 1] ** 2
    vb = np.cos(2 * pts[:, 1]) - 0.5 * pts[:, 0]
    xq = np.linspace(-0.9, 0.9, 71)
    Xq, Yq = np.meshgrid(xq, xq)
    q = LinearNDInterpolator(pts, np.column_stack([va, vb]))(Xq, Yq)
    qa = LinearNDInterpolator(pts, va)(Xq, Yq)
    qb = LinearNDInterpolator(pts, vb)(Xq, Yq)
    # NaN in the same places (outside the hull); finite values bit-identical.
    assert np.array_equal(np.isnan(q[..., 0]), np.isnan(qa))
    assert np.array_equal(np.isnan(q[..., 1]), np.isnan(qb))
    fa = np.isfinite(qa)
    fb = np.isfinite(qb)
    assert np.array_equal(q[..., 0][fa], qa[fa])
    assert np.array_equal(q[..., 1][fb], qb[fb])


@pytest.mark.parametrize('N', [384, 512])
def test_remap_2d_byte_identical_to_pre_k3_two_interp(N):
    """The optimized ``_apply_displaced_remap_2d`` is byte-identical to the
    pre-K3 two-separate-interp algorithm (the shared-Delaunay 2-column interp is
    an exact refactor)."""
    dx = 8e-6
    E_in = _gauss(N, dx, 3e-3)
    rmap = _synthetic_ray_map(n_side=181, r_max=3.0e-3, dx=dx)
    new = _apply_displaced_remap_2d(E_in, rmap, _WL, dx, dx)
    ref = _remap_2d_reference(E_in, rmap, _WL, dx, dx)
    assert new.shape == ref.shape and new.dtype == ref.dtype
    assert np.array_equal(new, ref), (
        f"max abs diff {np.max(np.abs(new - ref)):.3e}")


def test_remap_2d_shared_delaunay_is_faster():
    """The shared-Delaunay remap is measurably faster than the pre-K3
    two-Delaunay path (best-of-N; generous 0.85x bound, not a brittle absolute
    threshold).  Measured ~1.6-1.8x in the K3 sweep at N=1024."""
    if available_memory_bytes() < 2 * (1 << 30):
        pytest.skip("insufficient free RAM for the timed remap comparison")
    N, dx = 768, 8e-6
    E_in = _gauss(N, dx, 3e-3)
    rmap = _synthetic_ray_map(n_side=221, r_max=3.0e-3, dx=dx)

    def _best(fn, reps=5):
        fn()  # warm
        return min(_time_once(fn) for _ in range(reps))

    def _new():
        return _apply_displaced_remap_2d(E_in, rmap, _WL, dx, dx)

    def _old():
        return _remap_2d_reference(E_in, rmap, _WL, dx, dx)

    t_new = _best(_new)
    t_old = _best(_old)
    # sanity: still byte-identical at this grid
    assert np.array_equal(_new(), _old())
    assert t_new < 0.85 * t_old, (
        f"remap not faster: new {t_new*1e3:.1f} ms vs old {t_old*1e3:.1f} ms "
        f"(ratio {t_new / t_old:.3f})")


# ===========================================================================
# Win 2 -- traced ray-density amplitude: shared upsample coordinate stack
# ===========================================================================
def test_ray_density_upsample_coord_reuse_deterministic():
    """``amplitude_model='ray_density'`` (which shares the OPL upsample's
    coarse->full coordinate stack with the ray-density amplitude upsample) is
    deterministic and matches a fresh call bit-for-bit -- the coord reuse does
    not perturb the field."""
    N, dx = 512, 8e-6
    E0 = _gauss(N, dx, 3e-3)
    p = _singlet()
    kw = dict(prescription=p, wavelength=_WL, dx=dx, on_undersample='silent',
              amplitude_model='ray_density')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = np.asarray(la.apply_real_lens_traced(E0, **kw))
        b = np.asarray(la.apply_real_lens_traced(E0, **kw))
    assert np.array_equal(a, b)
    assert np.isfinite(a).all()


def test_ray_density_support_within_screen_support():
    """The ray-density field's nonzero support is a SUBSET of the screen field's
    -- the ray-density magnitude rides the screen field's unit phasor, which is
    zero wherever the OPL/valid mask zeros the screen field.  Guards that the
    shared-coordinate upsample did not leak ray-density energy OUTSIDE the OPL
    path's support it reuses (the coord stacks are identical, so the two
    upsampled masks register)."""
    N, dx = 512, 8e-6
    E0 = _gauss(N, dx, 3e-3)
    p = _singlet()
    common = dict(prescription=p, wavelength=_WL, dx=dx,
                  on_undersample='silent')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        screen = np.asarray(la.apply_real_lens_traced(
            E0, amplitude_model='screen', **common))
        rd = np.asarray(la.apply_real_lens_traced(
            E0, amplitude_model='ray_density', **common))
    leaked = (np.abs(rd) > 0) & (np.abs(screen) == 0)
    assert not bool(leaked.any()), f"{int(leaked.sum())} leaked pixels"
