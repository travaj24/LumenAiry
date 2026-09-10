"""complex64 THROUGH the carrier chain (AUDIT_TRACED_MEMORY_2026_08_09 sec 3,
ranked row 12 -- closed in v5.44).

The audit measured that requesting complex64 saved 0.0 GB because six helpers
in ``propagators/carrier.py`` returned complex128 unconditionally, NumPy
promoted the product, and the dtype survived exactly one chain leg.  Its
prescription (sec 3.4): complex64 STORAGE of the smooth envelope, float64
CONSTRUCTION of every reference-phase ARGUMENT, and the phasor narrowed only
after ``exp``.  ``carrier._phasor_rows`` is that boundary; the four phase
helpers take ``dtype=``, ``_fourier_upsample_crop`` pads in the envelope's
dtype, and the exact focus readout keeps the envelope's dtype.

Pins, all build-free by construction:

* the DEFAULT path is untouched: ``dtype=None`` and ``dtype=complex128``
  return ``np.array_equal`` arrays (both take the shipped whole-grid ``exp``);
* a complex64 phasor is ONE float32 rounding from the complex128 one, and
  that error does NOT grow with the phase argument -- while the naive
  float32-ARGUMENT build (the control) does grow with it;
* ``_fourier_upsample_crop``, the exact focus readout and a one-group
  ``propagate_traced_carrier_chain`` keep a complex64 input complex64 end to
  end, within a bound derived from the float32 floor.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.propagators import carrier as C

_WL = 1.31e-6
_K = 2 * np.pi / _WL

#: max |c64 - c128| of a UNIT phasor narrowed from a float64 build: each
#: component rounds by at most eps32/2 = 5.96e-8, so |dz| <= sqrt(2) * 5.96e-8
#: = 8.4e-8.  Measured 4.2e-8 on all four helpers (validate_mixed_precision.py,
#: 2026-09-09).  Bar 2.5e-7: 3x above the analytic ceiling, 60x below the
#: naive control at the SMALLEST argument tested.
_C64_PHASOR_TOL = 2.5e-7


@pytest.fixture(autouse=True)
def _deterministic_fft():
    prev = la.get_fft_auto_promote()
    la.set_fft_auto_promote(False)
    yield
    la.set_fft_auto_promote(prev)


def _helpers(N, dx, R):
    """The four phase helpers at one (R, tilt, centre) configuration, each as
    ``fn(dtype) -> array``."""
    shape = (N, N)
    L, M = 0.0515, -0.02
    x0, y0 = 1.3e-3, -0.4e-3
    return {
        '_radial_carrier_phase': lambda dt: C._radial_carrier_phase(
            shape, dx, dx, _WL, R, +1, dtype=dt),
        '_tilt_ramp': lambda dt: C._tilt_ramp(
            shape, dx, _WL, L, M, x0, y0, -1, dtype=dt),
        '_tilt_exactness_phase': lambda dt: C._tilt_exactness_phase(
            shape, dx, dx, _WL, R, L, M, +1, centre=(x0, y0), dtype=dt),
        '_sphere_parab_conversion': lambda dt: C._sphere_parab_conversion(
            shape, dx, _WL, R, +1, centre=(x0, y0), dtype=dt),
    }


# ===========================================================================
# 1.  the default path is the shipped path
# ===========================================================================
@pytest.mark.parametrize('name', ['_radial_carrier_phase', '_tilt_ramp',
                                  '_tilt_exactness_phase',
                                  '_sphere_parab_conversion'])
def test_dtype_none_and_complex128_are_byte_identical(name):
    fn = _helpers(256, 1.806e-6, 45.9e-3)[name]
    a = fn(None)
    b = fn(np.complex128)
    if a is None:
        pytest.skip('helper returned None (flag off)')
    assert a.dtype == np.complex128
    assert np.array_equal(a, b)


# ===========================================================================
# 2.  complex64: one float32 rounding, flat in the phase argument
# ===========================================================================
@pytest.mark.parametrize('R, N, dx', [
    (45.9e-3, 512, 1.806e-6),       # max |k r^2 / 2R| ~ 3.6e+02 rad
    (5.0e-3, 1024, 1.806e-6),       # ~1.3e+04 rad
])
def test_complex64_phasors_are_one_float32_rounding_from_complex128(R, N, dx):
    for name, fn in _helpers(N, dx, R).items():
        ref = fn(np.complex128)
        if ref is None:
            continue
        got = fn(np.complex64)
        assert got.dtype == np.complex64, name
        err = float(np.abs(ref - got.astype(np.complex128)).max())
        assert err <= _C64_PHASOR_TOL, (name, R, err)


def test_the_float32_argument_control_grows_with_the_argument():
    """WHY the boundary sits after ``exp``: building the ARGUMENT in float32
    loses ~7 digits of ``k r^2 / 2R``, so its phasor error scales with the
    argument (measured 1.5e-05 at 3.6e+02 rad, 4.9e-04 at 1.3e+04 rad,
    2026-09-09) while the float64-argument build stays at the float32
    phasor floor.  Bar: the control must exceed the shipped complex64 error
    by at least 10x at the SMALL argument and 100x at the large one --
    both decades below the measured ratios (360x and 11 600x)."""
    for R, N, dx, ratio in ((45.9e-3, 512, 1.806e-6, 10.0),
                            (5.0e-3, 1024, 1.806e-6, 100.0)):
        x = (np.arange(N) - N / 2) * dx
        r2 = x[None, :] ** 2 + x[:, None] ** 2
        ref = np.exp(1j * _K * r2 / (2.0 * R))
        good = C._radial_carrier_phase((N, N), dx, dx, _WL, R, +1,
                                       dtype=np.complex64)
        naive = np.exp(1j * (_K * r2 / (2.0 * R)).astype(np.float32))
        e_good = float(np.abs(ref - good.astype(np.complex128)).max())
        e_naive = float(np.abs(ref - naive.astype(np.complex128)).max())
        assert e_naive >= ratio * e_good, (R, e_good, e_naive)


# ===========================================================================
# 2b.  the SEVENTH reference phase: ``_build_carrier_phase`` (v5.44.1, D2)
# ===========================================================================
# VERIFY_LENS_BANDED_COMPLEX64_2026_09_10 D2: the 5.44.0 change gave ``dtype=``
# to four helpers and missed this one, which
# ``carrier_referenced_envelope`` / ``carrier_referenced_reconstruct`` reach --
# so a complex64 chain still built ONE FULL-GRID complex128 phasor per call
# (measured 5 per two-group chain; 4.29 GB each at N=16384).  Both sides are
# pinned here: the complex64 arm builds no complex128 grid, and the complex128
# arm is byte-identical.


def _carrier_grids(N, dx, dtype):
    x = (np.arange(N) - N / 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    return np.exp(-r2 / (0.28 * N * dx) ** 2).astype(dtype)


def _log_phase_helpers(monkeypatch, log):
    """Wrap every reference-phase helper ``_build_carrier_phase`` can reach and
    record ``(name, dtype, size)`` of what it returns -- the V7 caller probe
    the verification recommended, as a test."""
    for nm in ('_radial_carrier_phase', '_axis_carrier_phase',
               '_phasor_rows', '_narrow_rows'):
        real = getattr(C, nm)

        def mk(real=real, nm=nm):
            def wrapper(*a, **kw):
                out = real(*a, **kw)
                if out is not None:
                    log.append((nm, np.dtype(out.dtype), int(out.size)))
                return out
            return wrapper
        monkeypatch.setattr(C, nm, mk())


@pytest.mark.parametrize('R', [55e-3, (55e-3, -70e-3), (55e-3, np.inf)],
                         ids=['scalar', 'astigmatic', 'one-axis'])
@pytest.mark.parametrize('fn_name', ['carrier_referenced_envelope',
                                     'carrier_referenced_reconstruct'])
def test_build_carrier_phase_builds_no_complex128_grid_on_a_complex64_field(
        monkeypatch, R, fn_name):
    """complex64 in -> the phase factor is complex64 and NO helper returns a
    full-grid complex128 array (pre-fix: exactly one per call, on every
    branch).  The VALUE pin is exact and two-arm: the complex64 factor is the
    whole-grid complex128 factor narrowed ONCE, bit for bit -- on the radial
    branch because ``_phasor_rows`` stores ``exp`` of the same float64
    argument, on the astigmatic branch because ``_narrow_rows`` stores the
    same elementwise per-axis product."""
    N, dx = 256, 4.0e-6
    fn = getattr(C, fn_name)
    sign = -1 if fn_name == 'carrier_referenced_envelope' else +1
    E64 = _carrier_grids(N, dx, np.complex64)
    log = []
    _log_phase_helpers(monkeypatch, log)
    out = fn(E64, R, _WL, dx)
    assert out.dtype == np.complex64
    wide = [d for d in log if d[1] == np.dtype(np.complex128) and d[2] >= N * N]
    assert wide == [], wide
    # the factor itself, against the shipped whole-grid build narrowed once
    ref = C._build_carrier_phase((N, N), dx, dx, _WL, R, sign, 'ref')
    got = C._build_carrier_phase((N, N), dx, dx, _WL, R, sign, 'got',
                                 dtype=np.complex64)
    assert got.dtype == np.complex64
    assert np.array_equal(got, ref.astype(np.complex64))
    assert np.array_equal(np.asarray(out),
                          (E64 * ref.astype(np.complex64)))


@pytest.mark.parametrize('R', [55e-3, (55e-3, -70e-3), (55e-3, np.inf)],
                         ids=['scalar', 'astigmatic', 'one-axis'])
def test_build_carrier_phase_complex128_is_the_shipped_whole_grid_build(R):
    """complex128 in -> byte-identical.  ``dtype=None`` (every pre-v5.44.1
    call), ``dtype=complex128`` and the public helpers all take the shipped
    whole-grid ``np.exp`` and return the same bits."""
    N, dx = 256, 4.0e-6
    a = C._build_carrier_phase((N, N), dx, dx, _WL, R, +1, 'a')
    b = C._build_carrier_phase((N, N), dx, dx, _WL, R, +1, 'b',
                               dtype=np.complex128)
    assert a.dtype == np.complex128 and b.dtype == np.complex128
    assert np.array_equal(a, b)
    E128 = _carrier_grids(N, dx, np.complex128)
    got = C.carrier_referenced_reconstruct(E128, R, _WL, dx)
    assert got.dtype == np.complex128
    assert np.array_equal(np.asarray(got), E128 * a)


def test_the_complex64_carrier_call_no_longer_pays_a_full_grid_complex128():
    """TEETH for the memory half, at the first N where the band is smaller
    than the grid.

    ``_phasor_rows`` / ``_narrow_rows`` band at ``_PHASOR_BAND_BYTES`` = 32 MB
    of complex128 scratch, so the transient is a full grid until
    ``16 N^2 > 32e6`` (N > 1414); N=2048 is the first power of two past it.
    Measured (2026-09-11, numpy 2.4 / py 3.14, whole-call tracemalloc peak,
    warm): complex128 arm 224.03 MiB both before and after; complex64 arm
    224.03 MiB before the fix (the leak: the SAME peak, hence the audit's
    "requesting complex64 saved 0.0 GB"), 189.03 MiB after -- a gap of
    35.0 MiB = 8.75 * N^2 bytes, where the removed complex128 phasor is
    16 N^2 = 64 MiB minus the 32 MB band that replaces it.  The bar is
    4 * N^2 bytes (16.8 MiB), 2.2x under the measurement and infinitely above
    the pre-fix gap of exactly 0."""
    import tracemalloc

    N, dx, R = 2048, 3.0e-6, 55e-3

    def _peak(dtype):
        E = _carrier_grids(N, dx, dtype)
        C.carrier_referenced_envelope(E, R, _WL, dx)          # warm
        tracemalloc.start()
        tracemalloc.reset_peak()
        out = C.carrier_referenced_envelope(E, R, _WL, dx)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert out.dtype == dtype
        return peak

    p128 = _peak(np.complex128)
    p64 = _peak(np.complex64)
    assert p128 - p64 >= 4 * N * N, (p128 / 2 ** 20, p64 / 2 ** 20,
                                     (p128 - p64) / 2 ** 20)


# ===========================================================================
# 3.  the transform pair, the readout and the chain keep complex64
# ===========================================================================
def test_fourier_upsample_crop_keeps_complex64():
    """Bar derivation: the c64 round trip differs from the c128 one by the
    float32 rounding of the envelope samples propagated through two FFTs --
    ~N * eps32 relative at worst, measured 2.0e-07 rel L2 at N=2048
    (2026-09-09).  Bar 1e-5, 50x above the measurement."""
    N, dx = 512, 1.806e-6
    rng = np.random.default_rng(1)
    x = (np.arange(N) - N / 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    env = (np.exp(-r2 / (0.15 * N * dx) ** 2)
           * (1 + 0.05 * rng.standard_normal((N, N)))).astype(np.complex128)
    for nc, nf in ((N // 2, N), (N, N // 2)):
        ref = C._fourier_upsample_crop(env, nc, nf)
        same = C._fourier_upsample_crop(env.copy(), nc, nf)
        assert np.array_equal(ref, same)                 # shipped path
        got = C._fourier_upsample_crop(env.astype(np.complex64), nc, nf)
        assert got.dtype == np.complex64
        rel = np.linalg.norm(ref - got) / np.linalg.norm(ref)
        assert rel <= 1e-5, (nc, nf, rel)


def _readout_field(N, dx, R, dtype):
    x = (np.arange(N) - N // 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    E = (np.exp(-r2 / (0.25e-3 ** 2))
         * np.exp(1j * _K * (-(np.sqrt(r2 + R * R) - abs(R)))))
    return E.astype(dtype)


def test_exact_focus_readout_keeps_complex64():
    """The readout's two ``.astype(np.complex128)`` casts were the leak on
    the memory-dominant stage; now the field keeps the envelope's dtype.

    Bar derivation: the complex64 arm differs from the complex128 arm by the
    float32 rounding of the input and of each phasor (~1e-7 relative per
    stage, four stages) plus the Bluestein zoom in complex64 -- measured
    end to end on this fixture (recorded in the assertion message on first
    run, 2026-09-09).  Bar 1e-4 on the relative L2 of the readout field."""
    N, dx, R = 256, 2.0e-6, -2.0e-3
    kw = dict(dx_out=0.1e-6, N_out=64, window_factor=4.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = np.asarray(C.carrier_referenced_exact_focus_readout(
            _readout_field(N, dx, R, np.complex128), R, -R, _WL, dx, **kw))
        b = np.asarray(C.carrier_referenced_exact_focus_readout(
            _readout_field(N, dx, R, np.complex64), R, -R, _WL, dx, **kw))
    assert a.dtype == np.complex128
    assert b.dtype == np.complex64, b.dtype
    rel = float(np.linalg.norm(a - b) / np.linalg.norm(a))
    assert rel <= 1e-4, rel


def _singlet(R1, R2, d, glass, ap, name='s'):
    return {'name': name, 'aperture_diameter': ap, 'thicknesses': [d],
            'surfaces': [
                {'radius': R1, 'glass_before': 'air', 'glass_after': glass,
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
                {'radius': R2, 'glass_before': glass, 'glass_after': 'air',
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]}


def test_one_group_chain_keeps_complex64_end_to_end():
    """The S8 chain fixture (one group, sphere reference, ray-density +
    remap, the shipped defaults) run once at complex128 and once with the
    SAME envelope in complex64.

    The dtype claim is exact: every stage the audit listed as a leak now
    follows the field, so the returned field IS complex64.  The accuracy bar
    is the float32 floor compounded over the chain's ~8 phasor multiplies
    and one element (measured on the synthetic 6-leg prototype chain:
    2.8e-7 relative per leg, 1.1e-6 after six; validate_mixed_precision.py
    2026-09-09) -- 1e-4 on relative L2 leaves two decades."""
    N, dx = 512, 30e-6
    w, R_in = 4.5e-3, 60e-3
    presc = _singlet(60.0e-3, -60.0e-3, 3.0e-3, 'N-BK7', 14.0e-3, 'sph')
    x = (np.arange(N) - N // 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    env = np.exp(-r2 / w ** 2).astype(np.complex128)

    def _run(e):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = la.propagate_traced_carrier_chain(
                e, [{'prescription': presc, 'gap_before': 0.0}], _WL, dx,
                r_in=R_in, ray_subsample=8, n_workers=1,
                traced_kwargs=dict(on_undersample='silent',
                                   on_noncollimated='silent'))
        return np.asarray(res.field)

    a = _run(env)
    b = _run(env.astype(np.complex64))
    assert a.dtype == np.complex128
    assert b.dtype == np.complex64, b.dtype
    rel = float(np.linalg.norm(a - b) / np.linalg.norm(a))
    assert rel <= 1e-4, rel
