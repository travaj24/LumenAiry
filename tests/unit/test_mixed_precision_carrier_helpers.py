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
#: = 8.43e-8.  Bar 2.5e-7 = 2.97x that analytic ceiling.
#:
#: RE-MEASURED 2026-09-11 on the two fixtures below, on BOTH builds -- Windows
#: py 3.14.6 / numpy 2.4.4 and WSL py 3.12.3 / numpy 2.4.6, IDENTICAL to every
#: figure: 4.2032e-08 at R=45.9 mm and 4.2117e-08 at R=5 mm, so the bar sits
#: 5.94x above the measurement.  TWO-SIDED: the naive float32-ARGUMENT control
#: on the SMALLER of the two arguments reads 9.8719e-07, which is 3.95x ABOVE
#: this bar -- so the bar separates the two builds of the phasor at both
#: fixtures.  (This line used to claim "60x below the naive control at the
#: SMALLEST argument tested"; it is 3.95x.
#: VERIFY_LENS_BANDED_COMPLEX64_2026_09_10 D4.)
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
    # the TRUE max |k r^2 / (2R)| of each fixture, re-measured 2026-09-11 on
    # both builds (D4: these comments read 3.6e+02 and 1.3e+04 rad, 16x off).
    (45.9e-3, 512, 1.806e-6),       # max |k r^2 / (2R)| = 22.34 rad
    (5.0e-3, 1024, 1.806e-6),       # 820.2 rad
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
    loses ~7 digits of ``k r^2 / (2R)``, so its phasor error scales with the
    argument, while the float64-argument build stays flat at the float32
    phasor floor.

    RE-MEASURED 2026-09-11 on BOTH builds (Windows py 3.14.6 / numpy 2.4.4 and
    WSL py 3.12.3 / numpy 2.4.6 -- identical to every figure).  The numbers
    this docstring used to carry -- "1.5e-05 at 3.6e+02 rad, 4.9e-04 at
    1.3e+04 rad", ratios "360x and 11 600x" -- do not reproduce on these
    fixtures and are corrected here
    (VERIFY_LENS_BANDED_COMPLEX64_2026_09_10 D4)::

        fixture             max argument   control     shipped c64   ratio
        R=45.9 mm, N=512      22.34 rad    9.872e-07    4.203e-08     23.5x
        R=5 mm,    N=1024    820.19 rad    3.052e-05    4.198e-08    727.0x

    The bars stay 10x and 100x, and the margins they actually carry are 2.35x
    and 7.27x -- thin at the small argument, and that is the honest reading:
    the control only SEPARATES above ~2e+03 rad (an independent ladder over
    2.0e+02 .. 2.0e+05 rad reads ratios 1.0, 1.0, 1.0, 5.8e+03, 2.3e+04,
    4.6e+04, 1.9e+05 -- verification V5), and these two shipped fixtures sit
    BELOW that, at 22.34 and 820.19 rad.  What the test pins two-sidedly at
    these arguments is the DIRECTION and the decade: a float32 argument is
    already 23x worse at 22 rad and 727x at 820, while the shipped build is
    flat at the 4.2e-08 float32 phasor floor either way."""
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
    ~sqrt(log2 N) * eps32 relative, since BOTH transforms of the pair run in
    single precision on numpy >= 2.0 (D3, and the docstring of
    ``test_niche_perf_round2_2026_08_10::
    test_upsample_crop_keeps_the_envelope_dtype``).

    RE-MEASURED 2026-09-11 on THIS test's own fixture -- N=512, the 256->512
    branch -- on both builds: 1.7151e-07 rel L2 (Windows and WSL agree to 12
    figures).  The number this docstring used to record, "2.0e-07 at N=2048",
    was measured on a grid the test does not run
    (VERIFY_LENS_BANDED_COMPLEX64_2026_09_10 D4).  Bar 1e-5 = 58.3x above the
    measurement."""
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
    stage, four stages) plus the Bluestein zoom in complex64.

    MEASURED end to end on this fixture, 2026-09-11 -- the docstring used to
    record NO number at all ("recorded in the assertion message on first run"
    is not a derivation; VERIFY_LENS_BANDED_COMPLEX64_2026_09_10 D5):
    **9.8995e-08** on Windows (py 3.14.6 / numpy 2.4.4) and **9.8841e-08** on
    WSL (py 3.12.3 / numpy 2.4.6) -- the two builds agree to three figures.
    Bar **1e-6**, TIGHTENED from the 1e-4 that shipped, which sat three
    decades above the measurement and would have passed a 1000x precision
    regression.  1e-6 is 10.1x above BOTH builds' readings, so it is
    two-sided: it fails on any real widening of the complex64 path and passes
    on the build spread (0.16 % between the two)."""
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
    assert rel <= 1e-6, rel


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
    is the float32 floor compounded over the chain's ~8 phasor multiplies and
    one element.

    MEASURED on this fixture, 2026-09-11: **9.4427e-08** on Windows (py 3.14.6
    / numpy 2.4.4) and **9.4427e-08** on WSL (py 3.12.3 / numpy 2.4.6) -- the
    two builds agree to 10 figures.  Bar **1e-6**, TIGHTENED from the 1e-4
    that shipped, which sat three decades above the measurement, so a 1000x
    precision regression passed it
    (VERIFY_LENS_BANDED_COMPLEX64_2026_09_10 D5).  1e-6 is 10.6x above both
    builds: it fails on any real widening and passes on the build spread.
    (An independent six-leg synthetic chain grows LINEARLY at 4.58e-08 per
    leg, 2.75e-07 after six -- verification V6 -- which is the shape this
    single-group bar is set against; the "2.8e-7 per leg, 1.1e-6 after six"
    this docstring used to cite is the prototypes' different chain.)"""
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
    assert rel <= 1e-6, rel
