"""Oracle-backed regression tests for the FGA hammer findings H4 (auto-sampler
memory wall) and H5 (auto-sampler p_max mis-sizing = focused-halo garbage).

Both were RESOLVED in-library (2026-07-19); these tests pin the FIX (NOT the
former broken behavior):

  H4  ``apply_real_lens_fga`` now
        (a) DEFAULTS ``mem_budget_mb`` to ~25% of the available RAM (floored at
            1000 MB) when unset, so the byte-identical position-lattice chunk loop
            ENGAGES for a large aperture instead of one huge (Nq, Np) allocation
            that OOMs;
        (b) sizes the chunk model with the ACTUAL differential ray-trace bundle
            (finite-difference 9-ray vs analytic single-ray), so the budget is
            honored on the FD path -- the old flat 500 B trace term under-counted
            the FD bundle ~6x (measured 6000 MB budget -> ~60 GB); and
        (c) DEFAULTS ``exact_jacobian`` to the analytic single-ray Jacobian for an
            all-conic prescription (~5x lighter trace), making a large-N full
            trace feasible where the FD path OOMs.

  H5  For a NEAR-COLLIMATED input the auto sampler used to floor ``p_max`` at the
      beamlet completeness width 3/(k*w0) -- 32-130x the field's real angular
      content -- and a strong-focusing lens sprayed those excess-momentum beamlets
      into a halo of radius ~efl*p_max (measured f/5 r2m 187/304 um vs oracle 65).
      The sampler now DETECTS this (floor/content > 10), sizes ``p_max`` to the
      field's own content instead, and power-normalizes the output to restore the
      beamlet-completeness energy -- so the DEFAULT call reconstructs a sane spot.

Measured (f/5 biconvex singlet, lambda 1.31um, w0=5mm collimated input, image
49.163 mm behind exit vertex; dual-oracle EE50=15.17 EE80=55.22 r2m=64.98 um):
  GBD b120 @dx6um:                 15.12 / 54.00 / 63.97   (nails it)
  FGA auto (PRE-FIX) @N2048/dx12:  171.5 / 231.4 / 187.5   (3x wrong halo)
  FGA auto (FIXED)  @N2048/dx12:   r2m within ~1.5x of the oracle
"""
import numpy as np
import pytest

from lumenairy.glass import GLASS_REGISTRY
from lumenairy.propagators import fga
from lumenairy.propagators.fga import apply_real_lens_fga

WL = 1.31e-6


@pytest.fixture(scope="module")
def presc():
    # dispersionless model glass (fixed index 1.5168), as the hammer oracle used.
    GLASS_REGISTRY['ZF'] = lambda wl: 1.5168
    return {'wavelength': WL, 'aperture_diameter': 24e-3, 'surfaces': [
        {'radius': 51.68e-3, 'thickness': 5e-3, 'glass_before': 'air',
         'glass_after': 'ZF', 'semi_diameter': 12e-3},
        {'radius': -51.68e-3, 'thickness': 0.0, 'glass_before': 'ZF',
         'glass_after': 'air', 'semi_diameter': 12e-3}],
        'thicknesses': [5e-3], 'stop_index': 0}


def _gauss(N, dx, w0m=5e-3):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w0m ** 2).astype(np.complex128), X, Y, x


def _metrics(E, X, Y, x):
    I = np.abs(np.asarray(E)) ** 2
    j, i = np.unravel_index(np.argmax(I), I.shape)
    r = np.sqrt((X - x[i]) ** 2 + (Y - x[j]) ** 2)
    msk = r <= 500e-6
    r2m = np.sqrt((I * r ** 2)[msk].sum() / I[msk].sum()) * 1e6
    rb = np.linspace(0, 500e-6, 126)
    cum = np.array([I[r <= rr].sum() for rr in rb]) / I.sum()
    return (float(np.interp(0.5, cum, rb)) * 1e6,
            float(np.interp(0.8, cum, rb)) * 1e6, r2m)


ORACLE = dict(EE50=15.17, EE80=55.22, r2m=64.98)


# --------------------------------------------------------------------------
# H5 fix -- FAST (no propagation): for a near-collimated input (floor/content
# far above 1) the auto sampler now sizes p_max to the field CONTENT, not the
# beamlet completeness floor, and flags the power-normalization compensation.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("N,dx,exp_ratio", [(1024, 24e-6, 32.6),
                                            (2048, 12e-6, 65.2),
                                            (4096, 6e-6, 130.4)])
def test_h5_auto_pmax_content_sized_for_near_collimated(presc, N, dx, exp_ratio):
    E0, X, Y, x = _gauss(N, dx)
    w0 = 5.0 * dx
    content = fga._field_angular_content(E0, dx, dx, WL)
    k = 2.0 * np.pi / WL
    floor = fga._PMAX_BEAMLET_C / (k * w0)
    # the regime: the completeness floor is >>10x the field's angular content
    ratio = floor / content
    assert ratio == pytest.approx(exp_ratio, rel=0.05)
    assert ratio > fga._PMAX_CONTENT_OVERRIDE_RATIO       # override regime

    info = {}
    with pytest.warns(RuntimeWarning, match="near-collimated"):
        p_max, n_p = fga._resolve_sampling(E0, dx, dx, WL, w0, presc, None, None,
                                           info=info)
    # FIXED: p_max follows the field content (x tail), NOT the floor.
    assert info['content_override'] is True
    assert p_max == pytest.approx(fga._PMAX_CONTENT_TAIL * content, rel=1e-6)
    assert p_max < 0.2 * floor                            # far below the old floor
    # the excess-momentum halo radius ~efl*p_max collapses by the same factor
    assert n_p >= 7                                       # auto n_p clamps up to 7


# --------------------------------------------------------------------------
# H5 guard -- FAST: the identity / diverging regimes (floor/content <= ~7) are
# UNCHANGED -- p_max stays the completeness floor and no override fires, so the
# S2-2 identity-power and diverging-beam contracts are untouched at the sampler.
# --------------------------------------------------------------------------
def test_h5_identity_and_diverging_keep_floor():
    N, dx = 256, 0.633e-6
    xs = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(xs, xs)
    flat = {'name': 'flat', 'aperture_diameter': N * dx,
            'surfaces': [{'radius': np.inf, 'conic': 0.0, 'glass_before': 'air',
                          'glass_after': 'air', 'semi_diameter': N * dx / 2}],
            'thicknesses': []}
    w0 = 5.0 * dx
    k = 2.0 * np.pi / 0.633e-6
    floor = fga._PMAX_BEAMLET_C / (k * w0)
    # collimated Gaussians: content well within an order of magnitude of the floor
    for w in (12e-6, 18e-6, 30e-6):
        u0 = np.exp(-(Xg ** 2 + Yg ** 2) / w ** 2).astype(np.complex128)
        info = {}
        p_max, _n_p = fga._resolve_sampling(u0, dx, dx, 0.633e-6, w0, flat,
                                            None, None, info=info)
        assert info['content_override'] is False
        assert p_max == pytest.approx(floor, rel=1e-9)     # floor path preserved
    # diverging beam (~0.09 rad content > floor): also the floor/content path
    r2 = Xg ** 2 + Yg ** 2
    div = (np.exp(-r2 / (28e-6) ** 2)
           * np.exp(1j * k * r2 / (2 * 0.6e-3))).astype(np.complex128)
    info = {}
    fga._resolve_sampling(div, dx, dx, 0.633e-6, w0, flat, None, None, info=info)
    assert info['content_override'] is False


# --------------------------------------------------------------------------
# H4a/H4b fix -- FAST: the default path is NO LONGER unbounded, and the memory
# model COUNTS the FD ray-transfer bundle (so a budget is honored on the FD path).
# --------------------------------------------------------------------------
def test_h4_default_budget_engages_chunk_loop():
    # A concrete default budget (was None -> unbounded single (Nq, Np) alloc).
    budget = fga._default_mem_budget_mb()
    assert budget >= fga._DEFAULT_MEM_BUDGET_FLOOR_MB     # floored, non-None
    # a big grid-filling lattice: no budget = the OLD unbounded default; a
    # (modest) budget now engages the position-lattice chunk loop.
    Np = 49
    Nq = len(range(0, 4096, 2)) ** 2
    assert fga._resolve_nq_chunk(Nq, Np, True, Np, None, 't') is None   # unbounded
    ch = fga._resolve_nq_chunk(Nq, Np, True, Np, 3000.0, 't')
    assert ch is not None and 1 <= ch < Nq               # chunk loop engaged


def test_h4_memory_model_counts_fd_bundle():
    # The chunk model must include the finite-difference 9-ray bundle: the FD
    # path costs more per lattice point than the analytic single-ray path, so for
    # the SAME budget it gets a SMALLER chunk (was byte-blind: both used a flat
    # 500 B trace term that under-counted the FD bundle ~6x -> budget not honored).
    Np, Nq, budget = 49, 4_000_000, 3000.0
    fd = fga._resolve_nq_chunk(Nq, Np, True, Np, budget, 't', fd_bundle=True)
    an = fga._resolve_nq_chunk(Nq, Np, True, Np, budget, 't', fd_bundle=False)
    assert fd is not None and an is not None
    assert fd < an                                        # FD bundle costs more
    # the FD trace term is now a real per-point cost, not the old flat 500 B.
    trace_fd = fga._FD_BUNDLE_RAYS * fga._TRACE_STATE_BYTES
    assert trace_fd > 5 * 500.0                           # >5x the old under-count
    assert fga._FD_BUNDLE_RAYS == 9                       # base + +/-h in x,y,ux,uy


# --------------------------------------------------------------------------
# H4c fix -- FAST: exact_jacobian DEFAULTS (None) to the analytic single-ray
# Jacobian for an all-conic prescription (the 9N-ray FD bundle removed); explicit
# True/False preserved; aspheric falls back to FD.
# --------------------------------------------------------------------------
def test_h4_exact_jacobian_default_analytic_for_conic(presc):
    from lumenairy.raytrace import surfaces_from_prescription
    from lumenairy.raytrace.differential import (
        ray_transfer_jacobian,
        ray_transfer_jacobian_analytic,
    )
    surfs = surfaces_from_prescription(presc)
    assert fga._is_all_conic(surfs)
    # DEFAULT (None) auto-selects the analytic Jacobian for the conic singlet.
    assert fga._pick_ray_transfer(surfs, None) is ray_transfer_jacobian_analytic
    assert fga._pick_ray_transfer(surfs, True) is ray_transfer_jacobian_analytic
    assert fga._pick_ray_transfer(surfs, False) is ray_transfer_jacobian

    # an aspheric surface is NOT all-conic -> analytic unavailable -> FD, even at
    # the True/None default (the analytic form does not handle aspheres).
    class _Stub:
        aspheric_coeffs = [1e-3]
    assert not fga._is_all_conic([_Stub()])
    assert fga._pick_ray_transfer([_Stub()], None) is ray_transfer_jacobian
    assert fga._pick_ray_transfer([_Stub()], True) is ray_transfer_jacobian


# --------------------------------------------------------------------------
# H4/H5 fix -- SLOW (end-to-end): the DEFAULT call now completes (no OOM) AND
# reconstructs a SANE spot (r2m within ~1.5x of the oracle), not the pre-fix
# focused-halo garbage (r2m ~187 um).  coarse_stride=2 keeps the trace cheap.
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_h4_h5_default_call_is_sane_not_garbage(presc):
    pytest.importorskip("numba")
    N, dx = 2048, 12e-6
    E0, X, Y, x = _gauss(N, dx)
    Z = 49.163e-3
    pin = float(np.sum(np.abs(E0) ** 2))
    # DEFAULT sampling / budget / jacobian (only coarse_stride + dq_step + a
    # pinned budget for test speed/determinism -- all pure performance knobs, not
    # the H4/H5 fix).  The near-collimated override fires + auto power-normalizes,
    # so this must warn.  Measured r2m ~77 um (dq_step=3), ~69 um (dq_step=2).
    with pytest.warns(RuntimeWarning, match="near-collimated"):
        out = apply_real_lens_fga(E0, prescription=presc, wavelength=WL, dx=dx,
                                  output_plane_distance=Z, coarse_stride=2,
                                  dq_step=3, mem_budget_mb=3000)
    e50, e80, r2m = _metrics(out, X, Y, x)
    # the H5 fix: a sane spot within ~1.5x of the dual-oracle r2m ...
    assert r2m < 1.5 * ORACLE['r2m']
    assert r2m > 0.5 * ORACLE['r2m']                      # and not collapsed
    # ... FAR below the pre-fix halo (r2m ~187 um at this N/dx)
    assert r2m < 0.6 * 187.5
    # auto power-normalization restored the absolute scale
    pr = float(np.sum(np.abs(out) ** 2)) / pin
    assert abs(pr - 1.0) < 1e-6


# --------------------------------------------------------------------------
# H5 exoneration -- SLOW: coarse_stride=2 is byte-faithful to the full trace on a
# grid that CONTAINS the vignetting boundary (aperture r=12mm in-grid) -- cs=2 was
# NOT the H5 "garbage" (the auto sampler was), so cs1(exact)==cs2 to the FGA floor.
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_h5_coarse_stride_faithful_to_full_trace(presc):
    pytest.importorskip("numba")
    N, dx = 1024, 24e-6            # +-12.29 mm -> aperture edge in grid; N=1024 is
    assert N * dx / 2 >= 12e-3     # the coarse-interp resolution the <1e-3 needs
    E0, X, Y, x = _gauss(N, dx)
    Z = 49.163e-3
    full = apply_real_lens_fga(E0, prescription=presc, wavelength=WL, dx=dx,
                               output_plane_distance=Z, coarse_stride=1,
                               exact_jacobian=True, mem_budget_mb=3000)
    coarse = apply_real_lens_fga(E0, prescription=presc, wavelength=WL, dx=dx,
                                 output_plane_distance=Z, coarse_stride=2,
                                 mem_budget_mb=3000)
    rel = np.linalg.norm(coarse - full) / np.linalg.norm(full)
    assert rel < 1e-3            # measured 2.7e-4 (N=1024), 2.4e-4 (N=2048)


# --------------------------------------------------------------------------
# H4 -- SLOW: mem_budget chunking (position lattice only) is a lossless memory
# lever -- the chunked run reproduces the unchunked one to float round-off.
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_h4_mem_budget_chunk_lossless(presc):
    pytest.importorskip("numba")
    N, dx = 256, 6e-6
    E0, X, Y, x = _gauss(N, dx)
    Z = 49.163e-3
    # explicit large chunk pins OFF momentum chunking so only the position
    # lattice is chunked (a strict reorder of the outer additive sum).
    big = dict(prescription=presc, wavelength=WL, dx=dx, output_plane_distance=Z,
               chunk=10_000)
    full = apply_real_lens_fga(E0, mem_budget_mb=1e7, **big)      # unchunked
    scale = float(np.abs(full).max()) + 1e-300
    for mb in (8.0, 64.0):
        chunked = apply_real_lens_fga(E0, mem_budget_mb=mb, **big)
        # position-lattice chunking is a strict reorder of the outer additive
        # sum (row-parallel scatter preserves per-pixel order) -> lossless.
        assert np.max(np.abs(chunked - full)) <= 1e-9 * scale
