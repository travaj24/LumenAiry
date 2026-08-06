"""The Newton process pool must serve BOTH ``newton_fit`` backends.

Before v5.30.1 the pool only knew how to rebuild a ``RectBivariateSpline``, so
``_invert_newton_parallel`` force-returned the serial path whenever
``newton_fit='polynomial'`` -- which is the DEFAULT.  The consequences were:

  * ``n_workers`` was a silent no-op in the default configuration.  Measured:
    n_workers=1/4/8 gave 11.2 / 11.3 / 11.4 s, i.e. 0.98-0.99x -- indistinguish-
    able from noise, with no warning that the knob did nothing.
  * whether you got parallel Newton depended on a knob (``newton_fit``) that is
    about FIT ACCURACY, not about parallelism, and that most callers never set.
    Choosing the default fit silently cost the pool speed-up.

The fix teaches the worker to rebuild whichever fit the caller chose, from the
same pickled grids.  The thing that makes it safe -- and the thing this file
guards -- is that it must remain BIT-IDENTICAL to serial for both backends: the
Chebyshev fit is a deterministic lstsq on identical data so every worker
recovers the same coefficients, evaluation is elementwise so chunking cannot
change it, and the worker mirrors the serial path's choice of the combined
value+gradient call (using separate ``.ev`` calls instead would reorder the
floating-point work and break identity).

These tests deliberately size the grid past ``_POOL_MIN_PIXELS`` so the pool
actually engages -- below that threshold everything runs serial and a
pool/serial comparison would be vacuously true.
"""
import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_traced as LT

_WL = 1.31e-6
# ray_subsample=2 on a 1024 grid gives (1024/2)^2 = 262144 Newton points,
# comfortably past the 200k pool threshold.
_N, _RS = 1024, 2
_W0 = 1.0e-3


def _skip_if_low_ram():
    try:
        import psutil
        if psutil.virtual_memory().available < 3 * (1 << 30):
            pytest.skip('insufficient free RAM for the pool test')
    except ImportError:
        pass


def _doublet(ap):
    """Cemented doublet, same shape the design battery uses."""
    surfs, before = [], 'air'
    for R, g in ((61.5e-3, 'N-BK7'), (-45.0e-3, 'N-SF5'), (-128.0e-3, 'air')):
        surfs.append({'radius': R, 'glass_before': before, 'glass_after': g,
                      'conic': 0.0, 'radius_y': None, 'conic_y': None,
                      'aspheric_coeffs': None, 'aspheric_coeffs_y': None})
        before = g
    return {'name': 'doublet', 'aperture_diameter': ap,
            'thicknesses': [4.0e-3, 2.5e-3], 'surfaces': surfs}


def _run(fit, n_workers):
    ap = 1.2 * 2.0 * _W0
    dx = float(2.2 * max(ap, 3.0 * _W0) / _N)
    x = (np.arange(_N) - _N / 2) * dx
    X, Y = np.meshgrid(x, x)
    env0 = np.exp(-(X ** 2 + Y ** 2) / _W0 ** 2).astype(np.complex128)
    res = la.propagate_traced_carrier_chain(
        env0, [{'prescription': _doublet(ap), 'gap_before': 0.0}],
        _WL, dx, r_in=np.inf, ray_subsample=_RS, n_workers=n_workers,
        final_distance=0.0,
        traced_kwargs=dict(parallel_amp=False, on_undersample='silent',
                           newton_fit=fit))
    return np.asarray(res.field)


def test_the_pool_threshold_is_actually_exceeded():
    """Guard the premise: if the grid ever shrinks below the pool threshold,
    the identity tests below stop testing the pool and start passing for free.

    The threshold is two-tier since v5.30.2 -- a COLD pool must amortise Windows
    spawn (measured crossover ~200k points; at 16k a cold pool is 1.62x SLOWER),
    while a WARM one only has per-chunk pickling to cover and wins down to 1k.
    This grid clears the COLD bar, so it engages the pool either way."""
    assert _N // _RS > 0
    pts = (_N // _RS) ** 2
    assert pts >= LT._POOL_MIN_PIXELS, (
        f'{pts} Newton points is below the cold threshold '
        f'_POOL_MIN_PIXELS={LT._POOL_MIN_PIXELS}: the pool may not engage and '
        'these tests would be vacuous')
    assert LT._POOL_MIN_PIXELS_WARM <= LT._POOL_MIN_PIXELS, (
        'the warm threshold must not exceed the cold one -- a live pool has '
        'strictly less overhead left to amortise')


def test_a_warm_pool_engages_sooner_than_a_cold_one():
    """The two-tier rule is the whole point: a multi-group chain calls
    apply_real_lens_traced once per group, so only the FIRST group is cold.  A
    single threshold would keep every later group serial too -- which is what
    happened to design-121-class chains at ray_subsample=4 (65k points/group,
    below the 200k cold bar) before this split."""
    assert LT._POOL_MIN_PIXELS_WARM < LT._POOL_MIN_PIXELS, (
        'warm and cold thresholds are equal, so the split does nothing')
    # 65k is the measured design-121-class per-group count at rs=4: it must sit
    # below the warm bar (so later groups parallelise) and below the cold bar
    # (so the first group does not pay a losing spawn)
    assert LT._POOL_MIN_PIXELS_WARM < 65_536 < LT._POOL_MIN_PIXELS, (
        f'65536 points no longer straddles the two thresholds '
        f'({LT._POOL_MIN_PIXELS_WARM}, {LT._POOL_MIN_PIXELS}) -- the measured '
        'cold/warm tables that justified the split need redoing')


@pytest.mark.parametrize('fit', ['polynomial', 'spline'])
def test_pool_result_is_bit_identical_to_serial(fit):
    """The whole safety argument for parallel Newton: identical bits, not
    merely 'close'.  Runs for the DEFAULT polynomial backend too, which the
    pool refused to serve before v5.30.1."""
    _skip_if_low_ram()
    serial = _run(fit, 1)
    pooled = _run(fit, 4)
    assert np.array_equal(serial, pooled), (
        f'newton_fit={fit!r}: pool result differs from serial, '
        f'max|delta| = {np.abs(pooled - serial).max():.3e}')


def test_polynomial_is_no_longer_force_routed_to_serial():
    """Pin the specific defect: ``_invert_newton_parallel`` used to contain an
    early ``if newton_fit == 'polynomial': return _invert_newton(...)``.  A
    future refactor that reinstates it would silently halve throughput on the
    default path, and the bit-identity tests above would still pass -- so
    assert on the mechanism, not just the numbers."""
    import inspect
    src = inspect.getsource(LT.apply_real_lens_traced)
    i = src.find('def _invert_newton_parallel')
    assert i != -1, 'could not locate _invert_newton_parallel'
    body = src[i:i + 2600]
    assert "if newton_fit == 'polynomial':" not in body, (
        'the polynomial serial-force gate is back in '
        '_invert_newton_parallel: n_workers is a silent no-op again on the '
        'default fit')
    # the GPU bail-out is legitimate and must stay
    assert 'if use_gpu:' in body, (
        'the use_gpu in-process bail-out disappeared; CuPy arrays would be '
        'host-copied through the pool')


def test_worker_payload_carries_what_the_polynomial_fit_needs():
    """The worker can only rebuild the Chebyshev evaluators if the pickled
    payload actually carries the fit choice and its parameters."""
    import inspect
    src = inspect.getsource(LT.apply_real_lens_traced)
    for key in ("'newton_fit':", "'fit_poly_order':", "'fit_weights':"):
        assert key in src, f'worker payload is missing {key}'
    wsrc = inspect.getsource(LT._newton_invert_chunk)
    assert '_Cheb2DEvaluator' in wsrc, (
        'the pool worker cannot rebuild the polynomial fit')
    # backwards compatibility: an older payload with no key must still run
    assert "get('newton_fit', 'spline')" in wsrc, (
        'worker must default to spline for payloads written before the '
        'polynomial worker path existed')
    # and it must mirror the serial combined value+gradient route
    assert 'ev_value_and_grad' in wsrc, (
        'worker does not use the combined value+gradient call, so its '
        'floating-point order differs from serial')
