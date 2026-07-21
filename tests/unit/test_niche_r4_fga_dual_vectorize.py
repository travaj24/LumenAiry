"""R4 / roadmap B4 (2026-07-21): numba-vectorized batched-dual acceleration of
the adaptive-FGA analytic ray-transfer Jacobian.

K3 found the adaptive-FGA ``exact_jacobian`` hot spot is the scalar-object
dual-number arithmetic in ``raytrace/differential.py`` (``_AdrtDual.__mul__``
~178 k calls dominates -- FGA's largest single cost).  The fix replaces the
per-elementary-op NumPy dispatch with a numba kernel that runs the IDENTICAL
forward-mode-AD ray transfer per-ray in one compiled loop (value + 4-vector
tangent carried as a 5-tuple through inlined AD primitives).

These tests pin:

* EXACTNESS -- the numba kernel reproduces the ``_AdrtDual`` (``_adrt_numpy``)
  Jacobian / state / OPL / alive-mask element-wise on a ray batch, to the last
  ULP (the analytic transfer is exact; it is pinned hard).  Cross-checked
  against the independent finite-difference primitive too.
* Refraction, reflection, conic, and dead-ray (miss / TIR) coverage.
* DISPATCH -- ``ray_transfer_jacobian_analytic`` routes the composite all-conic
  path through the numba kernel and falls back to the dual path for coordinate
  breaks / ``per_surface=True`` (which the kernel does not cover).
* The adaptive-FGA end-to-end output is UNCHANGED (byte-identical-to-tolerance)
  with the kernel vs the forced-dual path.
* A MEASURED speedup on the ``__mul__``-dominated ray-transfer batch.

The kernel is opt-out-safe: numba unavailable / a build failure / a coordinate
break all fall through to the pure-NumPy ``_AdrtDual`` implementation, so the
default answer is never wrong.
"""
from __future__ import annotations

import copy as _copy
import time
import warnings

import numpy as np
import pytest

from lumenairy.glass import GLASS_REGISTRY
from lumenairy.raytrace import differential as D
from lumenairy.raytrace import (
    ray_transfer_jacobian,
    ray_transfer_jacobian_analytic,
    surfaces_from_prescription,
)

LAM = 1.31e-6
GLASS_REGISTRY['ZF_R4'] = lambda wl: 1.5168


# --------------------------------------------------------------------------
# prescriptions + ray batch
# --------------------------------------------------------------------------
def _singlet():
    return {'name': 's', 'aperture_diameter': 24e-3, 'surfaces': [
        {'radius': 51.68e-3, 'conic': 0., 'thickness': 5e-3,
         'glass_before': 'air', 'glass_after': 'ZF_R4', 'semi_diameter': 12e-3},
        {'radius': -51.68e-3, 'conic': 0., 'thickness': 0.0,
         'glass_before': 'ZF_R4', 'glass_after': 'air', 'semi_diameter': 12e-3}],
        'thicknesses': [5e-3, 0.0], 'stop_index': 0}


def _conic_doublet():
    return {'name': 'c', 'aperture_diameter': 20e-3, 'surfaces': [
        {'radius': 30e-3, 'conic': -1.0, 'glass_before': 'air',
         'glass_after': 'N-BK7', 'semi_diameter': 9e-3},
        {'radius': -80e-3, 'conic': -0.5, 'glass_before': 'N-BK7',
         'glass_after': 'air', 'semi_diameter': 9e-3}],
        'thicknesses': [5e-3, 20e-3]}


def _mirror():
    return {'name': 'm', 'aperture_diameter': 40e-3, 'surfaces': [
        {'radius': -40e-3, 'conic': 0., 'glass_before': 'air',
         'glass_after': 'air', 'is_mirror': True, 'semi_diameter': 18e-3},
        {'radius': float('inf'), 'conic': 0., 'glass_before': 'air',
         'glass_after': 'air', 'semi_diameter': 18e-3}],
        'thicknesses': [-20e-3, 0.0]}


def _ray_batch(n=800, seed=7, spread=18e-3, uspread=0.03):
    rng = np.random.RandomState(seed)
    x = (rng.rand(n) - 0.5) * spread
    y = (rng.rand(n) - 0.5) * spread
    ux = (rng.rand(n) - 0.5) * uspread
    uy = (rng.rand(n) - 0.5) * uspread
    return x, y, ux, uy


def _assert_dual_numba_match(surfs, x, y, ux, uy, jtol=1e-11, stol=1e-11):
    """The numba kernel reproduces the _AdrtDual composite result element-wise
    on the alive rays; the alive mask is bit-identical."""
    dual = D._adrt_numpy(x.copy(), y.copy(), ux.copy(), uy.copy(), surfs, LAM,
                         False)
    nb = D._adrt_numba(x.copy(), y.copy(), ux.copy(), uy.copy(), surfs, LAM)
    assert nb is not None, "numba kernel unexpectedly unavailable"
    # alive masks identical (the discrete miss/vignette/TIR decisions agree)
    assert np.array_equal(dual.alive, nb.alive)
    a = dual.alive
    jscale = max(float(np.abs(dual.jacobian[a]).max()), 1.0)
    assert np.max(np.abs(dual.jacobian[a] - nb.jacobian[a])) <= jtol * jscale
    for da, nba in ((dual.x, nb.x), (dual.y, nb.y), (dual.ux, nb.ux),
                    (dual.uy, nb.uy), (dual.opd, nb.opd)):
        sc = max(float(np.abs(da[a]).max()), 1.0)
        assert np.max(np.abs(da[a] - nba[a])) <= stol * sc
    return dual, nb


# ==========================================================================
# EXACTNESS -- numba kernel vs the _AdrtDual reference (refraction / conic /
# reflection), cross-checked against the FD primitive.
# ==========================================================================
def test_numba_matches_dual_refractive_singlet():
    pytest.importorskip("numba")
    surfs = [_copy.copy(s) for s in surfaces_from_prescription(_singlet())]
    x, y, ux, uy = _ray_batch()
    dual, nb = _assert_dual_numba_match(surfs, x, y, ux, uy)
    # cross-check against the independent finite-difference oracle
    f = ray_transfer_jacobian(x, y, ux, uy, surfs, LAM)
    live = nb.alive & f.alive
    assert np.max(np.abs(nb.jacobian[live] - f.jacobian[live])) < 1e-6


def test_numba_matches_dual_conic():
    pytest.importorskip("numba")
    surfs = surfaces_from_prescription(_conic_doublet())
    x, y, ux, uy = _ray_batch(spread=14e-3)
    _assert_dual_numba_match(surfs, x, y, ux, uy)


def test_numba_matches_dual_reflection():
    pytest.importorskip("numba")
    surfs = surfaces_from_prescription(_mirror())
    x, y, ux, uy = _ray_batch(spread=28e-3, uspread=0.02)
    _assert_dual_numba_match(surfs, x, y, ux, uy)


def test_numba_dead_rays_finite_masked_and_match_dual():
    """A batch that includes rays which miss / TIR at a steep surface: the numba
    result is FINITE (dead rows scrubbed to zero), the alive mask is identical to
    both the dual and FD primitives, and the live rays match the dual exactly."""
    pytest.importorskip("numba")
    p = {'name': 's', 'aperture_diameter': 20e-3, 'surfaces': [
        {'radius': 12e-3, 'conic': 0., 'glass_before': 'air',
         'glass_after': 'N-BK7', 'semi_diameter': 10e-3},
        {'radius': -12e-3, 'conic': 0., 'glass_before': 'N-BK7',
         'glass_after': 'air', 'semi_diameter': 10e-3}],
        'thicknesses': [6e-3, 15e-3]}
    surfs = surfaces_from_prescription(p)
    x = np.array([0.0, 3e-3, 6e-3, 9e-3])
    z = np.zeros(4)
    ux = np.array([0.0, 0.3, 0.7, 1.0])       # steep rays TIR / miss
    nb = D._adrt_numba(x.copy(), z.copy(), ux.copy(), z.copy(), surfs, LAM)
    dual = D._adrt_numpy(x.copy(), z.copy(), ux.copy(), z.copy(), surfs, LAM,
                         False)
    f = ray_transfer_jacobian(x, z, ux, z, surfs, LAM)
    assert np.isfinite(nb.jacobian).all()                 # no NaN/inf anywhere
    assert not nb.alive.all()                             # some ray died
    assert np.array_equal(nb.alive, dual.alive)
    assert np.array_equal(nb.alive, f.alive)
    a = nb.alive
    assert np.max(np.abs(nb.jacobian[a] - dual.jacobian[a])) < 1e-11 * max(
        float(np.abs(dual.jacobian[a]).max()), 1.0)


def test_numba_is_ulp_identical_to_dual_on_axis():
    """On the flagship (on-axis + moderate off-axis) batch the numba kernel is
    not merely close but BIT-identical to the dual reference -- the forward-AD
    arithmetic is replicated op-for-op with no fastmath reassociation."""
    pytest.importorskip("numba")
    surfs = [_copy.copy(s) for s in surfaces_from_prescription(_singlet())]
    x, y, ux, uy = _ray_batch(n=500)
    dual = D._adrt_numpy(x.copy(), y.copy(), ux.copy(), uy.copy(), surfs, LAM,
                         False)
    nb = D._adrt_numba(x.copy(), y.copy(), ux.copy(), uy.copy(), surfs, LAM)
    a = dual.alive & nb.alive
    assert np.array_equal(dual.jacobian[a], nb.jacobian[a])
    assert np.array_equal(dual.x[a], nb.x[a])
    assert np.array_equal(dual.opd[a], nb.opd[a])


# ==========================================================================
# DISPATCH -- the public entry point routes the composite all-conic path
# through the kernel and falls back for the cases it does not cover.
# ==========================================================================
def test_public_entry_uses_numba_for_composite_all_conic():
    """``ray_transfer_jacobian_analytic`` (composite, all-conic) returns exactly
    the numba kernel result AND still equals the dual reference -- i.e. the fast
    path is wired in without changing the answer."""
    pytest.importorskip("numba")
    surfs = [_copy.copy(s) for s in surfaces_from_prescription(_singlet())]
    x, y, ux, uy = _ray_batch()
    pub = ray_transfer_jacobian_analytic(x, y, ux, uy, surfs, LAM)
    nb = D._adrt_numba(x.copy(), y.copy(), ux.copy(), uy.copy(), surfs, LAM)
    dual = D._adrt_numpy(x.copy(), y.copy(), ux.copy(), uy.copy(), surfs, LAM,
                         False)
    assert np.array_equal(pub.jacobian, nb.jacobian)
    assert np.array_equal(pub.alive, nb.alive)
    a = pub.alive
    assert np.max(np.abs(pub.jacobian[a] - dual.jacobian[a])) <= 1e-11 * max(
        float(np.abs(dual.jacobian[a]).max()), 1.0)


def test_eligibility_gate_excludes_coordbreak():
    """A coordinate break in the surface list makes the batch numba-INELIGIBLE
    (the smooth frame transform is dual-only), so the public entry point falls
    back to the dual path and still matches the FD primitive."""
    pytest.importorskip("numba")
    from lumenairy.raytrace.surface import Surface
    base = surfaces_from_prescription({
        'name': 's', 'aperture_diameter': 20e-3, 'surfaces': [
            {'radius': 51.5e-3, 'conic': 0., 'glass_before': 'air',
             'glass_after': 'N-BK7', 'semi_diameter': 10e-3},
            {'radius': -51.5e-3, 'conic': 0., 'glass_before': 'N-BK7',
             'glass_after': 'air', 'semi_diameter': 10e-3}],
        'thicknesses': [4e-3, 10e-3]})
    cb = Surface(is_coordbrk=True, tilt_x_deg=2.0, decenter_x_m=0.3e-3,
                 thickness=15e-3, glass_before='air', glass_after='air')
    det = Surface(radius=np.inf, glass_before='air', glass_after='air',
                  thickness=0.0)
    surfs = [base[0], base[1], cb, det]
    assert D._adrt_surfaces_numba_eligible(base) is True
    assert D._adrt_surfaces_numba_eligible(surfs) is False
    x = np.array([0.0, 3e-3, -4e-3])
    y = np.array([0.0, 2e-3, 1e-3])
    z = np.zeros(3)
    a = ray_transfer_jacobian_analytic(x, y, z, z, surfs, LAM)
    f = ray_transfer_jacobian(x, y, z, z, surfs, LAM)
    assert np.max(np.abs(a.jacobian - f.jacobian)) < 1e-6


def test_per_surface_still_uses_dual_path():
    """``per_surface=True`` is not covered by the composite kernel; it must fall
    back to the dual path and still compose to the (numba) composite Jacobian."""
    pytest.importorskip("numba")
    surfs = [_copy.copy(s) for s in surfaces_from_prescription(_singlet())]
    x, y, ux, uy = _ray_batch(n=200)
    comp = ray_transfer_jacobian_analytic(x, y, ux, uy, surfs, LAM).jacobian
    per = ray_transfer_jacobian_analytic(x, y, ux, uy, surfs, LAM,
                                         per_surface=True).jacobian
    assert per.shape[0] == len(surfs)
    prod = np.einsum('nij,njk->nik', per[1], per[0])
    a = ray_transfer_jacobian_analytic(x, y, ux, uy, surfs, LAM).alive
    assert np.max(np.abs(prod[a] - comp[a])) < 1e-10


# ==========================================================================
# END-TO-END -- adaptive-FGA output is byte-identical-to-tolerance with the
# kernel vs the forced-dual path (the analytic transfer feeds exact_jacobian).
# ==========================================================================
@pytest.mark.slow
def test_fga_end_to_end_unchanged_numba_vs_dual():
    pytest.importorskip("numba")
    from lumenairy.memory import available_memory_bytes
    if available_memory_bytes() < 3 * 1024 ** 3:
        pytest.skip("insufficient RAM for the N=256 FGA end-to-end comparison")
    from lumenairy.propagators.fga import apply_real_lens_fga
    N, dx = 256, 24e-6
    xg = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(xg, xg)
    E0 = np.exp(-(X ** 2 + Y ** 2) / (5e-3) ** 2).astype(np.complex128)
    kw = dict(prescription=_singlet(), wavelength=LAM, dx=dx,
              output_plane_distance=49.163e-3, coarse_stride=1,
              exact_jacobian=True, mem_budget_mb=2000)

    # ensure a clean kernel state, then run the numba path
    D._ADRT_NUMBA_STATE = None
    D._ADRT_NUMBA_KERNEL = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out_numba = apply_real_lens_fga(E0, **kw)
    # force the dual path and re-run
    saved_state, saved_kern = D._ADRT_NUMBA_STATE, D._ADRT_NUMBA_KERNEL
    D._ADRT_NUMBA_STATE = False
    D._ADRT_NUMBA_KERNEL = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out_dual = apply_real_lens_fga(E0, **kw)
    finally:
        D._ADRT_NUMBA_STATE, D._ADRT_NUMBA_KERNEL = saved_state, saved_kern
    scale = float(np.abs(out_dual).max()) + 1e-300
    assert np.max(np.abs(out_numba - out_dual)) <= 1e-10 * scale


# ==========================================================================
# PERF -- measured speedup on the __mul__-dominated ray-transfer batch.
# ==========================================================================
@pytest.mark.slow
def test_measured_speedup_on_ray_transfer_batch():
    """The numba kernel is measurably faster than the scalar-object dual path on
    the per-momentum ray-transfer batch that dominated adaptive FGA.  Measured
    ~12x on the dev box (dq_step lattice, 49 momenta x 1600 rays); the assertion
    uses a conservative 2x floor to stay robust on a loaded runner."""
    pytest.importorskip("numba")
    surfs = [_copy.copy(s) for s in surfaces_from_prescription(_singlet())]
    nqc = 1600
    rng = np.random.RandomState(3)
    xs = (rng.rand(nqc) - 0.5) * 20e-3
    ys = (rng.rand(nqc) - 0.5) * 20e-3
    pv = np.linspace(-0.05, 0.05, 7)
    PX, PY = np.meshgrid(pv, pv)
    pxs, pys = PX.ravel(), PY.ravel()
    Np = pxs.size

    def _loop(fn):
        for j in range(Np):
            pxi, pyi = float(pxs[j]), float(pys[j])
            pz = np.sqrt(max(1.0 - pxi * pxi - pyi * pyi, 1e-12))
            uxin = np.full(nqc, pxi / pz)
            uyin = np.full(nqc, pyi / pz)
            fn(xs.copy(), ys.copy(), uxin, uyin, surfs, LAM)

    # warm the compile
    _loop(lambda *a: D._adrt_numba(*a))

    reps = 4
    t0 = time.perf_counter()
    for _ in range(reps):
        _loop(lambda x, y, ux, uy, s, w:
              D._adrt_numpy(x, y, ux, uy, s, w, False))
    dual_s = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        _loop(lambda *a: D._adrt_numba(*a))
    numba_s = (time.perf_counter() - t0) / reps
    speedup = dual_s / numba_s
    print(f"\nR4 ray-transfer speedup: dual {dual_s*1e3:.1f} ms/loop, "
          f"numba {numba_s*1e3:.1f} ms/loop -> {speedup:.1f}x "
          f"({Np} momenta x {nqc} rays)")
    assert speedup > 2.0, (
        f"expected >2x speedup on the ray-transfer batch, got {speedup:.2f}x")
