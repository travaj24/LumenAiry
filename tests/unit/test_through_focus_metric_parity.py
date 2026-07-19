"""AUDIT_V5_24_2 S3-19 -- through-focus single-plane-metric parity pin.

``through_focus_scan`` (NumPy) routes every plane through
``single_plane_metrics``; ``through_focus_scan_jax`` hand-inlines the same
metrics in a device loop.  The audit flagged the two as a duplicated pair
that once diverged (the all-NaN / zero-field bug).  The inline twin is NOT
byte-identical by design (power_in_bucket uses a strict ``<`` vs the NumPy
``<=`` band edge; rms_radius is an independent second moment), so this test
pins the parity CONTRACT on a SMOOTH Gaussian -- where those edge cases are
never exercised and the two agree to machine precision -- rather than
asserting byte-equality.  A gross future divergence re-breaks it.

Guarded by ``pytest.importorskip('jax')``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip('jax')


def test_numpy_jax_single_plane_metric_parity_smooth_gaussian():
    from lumenairy.analysis.through_focus import through_focus_scan, through_focus_scan_jax

    N, dx, wl = 96, 4e-6, 1.55e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    w0 = 60e-6
    E = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)

    z_values = np.linspace(-2e-3, 2e-3, 7)
    bucket_radius = 80e-6

    numpy_scan = through_focus_scan(
        E, dx, wl, z_values, bucket_radius=bucket_radius, ideal_peak=1.0)
    with warnings.catch_warnings():
        # x64 auto-enable RuntimeWarning is expected on the JAX path.
        warnings.simplefilter('ignore', RuntimeWarning)
        jax_scan = through_focus_scan_jax(
            E, dx, wl, z_values, bucket_radius=bucket_radius, ideal_peak=1.0)

    for name in ('peak_I', 'strehl', 'd4sigma_x', 'd4sigma_y',
                 'rms_radius', 'power_in_bucket'):
        a = np.asarray(getattr(numpy_scan, name), dtype=float)
        b = np.asarray(getattr(jax_scan, name), dtype=float)
        rel = np.max(np.abs(a - b) / (np.abs(a) + 1e-30))
        assert rel < 1e-9, f'{name} parity broke: max rel {rel:.2e}'
