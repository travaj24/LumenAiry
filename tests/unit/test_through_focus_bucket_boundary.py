"""AUDIT_V5_24_2 S3-19 -- reconcile the through-focus single-plane metrics.

``through_focus_scan_jax`` used to hand-inline a twin of
``single_plane_metrics`` that diverged from the NumPy backend in two
documented, band-edge ways:

1. **power_in_bucket boundary** -- the inline twin masked with a STRICT
   ``r < bucket_radius`` while ``single_plane_metrics`` ->
   ``radial_power_bands`` uses ``R2 <= r*r`` (``<=``).  A pixel sitting
   EXACTLY on the bucket radius was dropped on the JAX backend but kept on
   the NumPy backend.
2. **zero-field rms_radius** -- the inline twin left ``rms_radius`` at NaN
   on an all-zero plane (guarded by ``I_sum > 0``), whereas the NumPy path
   derives it from ``beam_d4sigma`` which returns 0 on a zero field.

v5.24.6 routes the JAX backend through the same ``single_plane_metrics``,
adopting the canonical ``<=`` boundary and the d4sigma-derived rms.  These
tests pin the chosen conventions with INDEPENDENT hand oracles (analytic
pixel energy / analytic second moment), not the code's own output.

The JAX-backend assertions are guarded by ``pytest.importorskip('jax')``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest


# --------------------------------------------------------------------------
# Convention pin (no JAX needed): the shared source of truth uses ``<=``.
# --------------------------------------------------------------------------
def test_bucket_boundary_pixel_included_on_radius():
    """A pixel whose distance from the centroid is EXACTLY the bucket radius
    is INCLUDED (``R2 <= r*r``).

    Independent oracle: two unit-amplitude pixels placed symmetrically about
    the grid centre at x = +/- k*dx.  The centroid is the grid centre
    (symmetry), so each lit pixel lies at distance exactly ``k*dx`` from it.
    With the canonical ``<=`` boundary and a bucket radius of ``k*dx`` the
    encircled power must be the analytic ``2 * |E|^2 * dx * dy`` = ``2*dx*dx``
    (both rings counted); one epsilon smaller it must be exactly 0.
    """
    from lumenairy.analysis.through_focus import single_plane_metrics

    N, dx = 32, 1e-6
    E = np.zeros((N, N), dtype=np.complex128)
    k = 5
    E[N // 2, N // 2 + k] = 1.0
    E[N // 2, N // 2 - k] = 1.0
    R = k * dx  # bucket radius lands EXACTLY on both lit pixels

    m_on = single_plane_metrics(E, dx, wavelength=1.55e-6, bucket_radius=R)
    # Hand oracle: 2 pixels * |1|^2 * dx * dy.
    assert m_on['power_in_bucket'] == pytest.approx(2.0 * dx * dx, rel=0, abs=0)

    # One ULP-scale smaller radius must EXCLUDE the on-radius ring -> 0.
    m_in = single_plane_metrics(
        E, dx, wavelength=1.55e-6, bucket_radius=R * (1.0 - 1e-9))
    assert m_in['power_in_bucket'] == 0.0


def test_bucket_boundary_matches_radial_power_bands():
    """``single_plane_metrics``' bucket is exactly ``radial_power_bands`` at
    the intensity centroid -- the canonical ``<=`` implementation, verified
    against an independent direct evaluation of the same mask."""
    from lumenairy.analysis.core import beam_centroid, radial_power_bands
    from lumenairy.analysis.through_focus import single_plane_metrics

    N, dx = 48, 2e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    w0 = 30e-6
    E = np.exp(-((X - 4e-6) ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)
    R = 25e-6

    m = single_plane_metrics(E, dx, wavelength=1.55e-6, bucket_radius=R)

    # Independent recomputation of the <= mask about the intensity centroid.
    cx, cy = beam_centroid(E, dx, dx)
    R2 = (X - cx) ** 2 + (Y - cy) ** 2
    I = np.abs(E) ** 2
    oracle = float(np.sum(I[R2 <= R * R]) * dx * dx)
    assert m['power_in_bucket'] == pytest.approx(oracle, rel=0, abs=0)
    # And it agrees with the public radial_power_bands entry point.
    band = radial_power_bands(E, dx, [R], dy=dx, center=(cx, cy))[0]
    assert m['power_in_bucket'] == pytest.approx(band, rel=0, abs=0)


# --------------------------------------------------------------------------
# Backend parity (JAX): the reconciled JAX path now == the NumPy path.
# --------------------------------------------------------------------------
def test_zero_field_rms_backend_parity():
    """A dark (all-zero) scan reports rms 0 and best_focus_spot 0.0 on BOTH
    backends (was NaN on the JAX inline twin before v5.24.6).

    Fail-before/pass-after: the pre-consolidation JAX path guarded its rms
    with ``I_sum > 0`` and left NaN on every zero plane, so
    ``best_focus_spot`` was NaN while the NumPy backend returned 0.0.
    Independent oracle: the second moment of a zero intensity distribution
    is identically 0, so rms == 0 on every plane.
    """
    pytest.importorskip('jax')
    from lumenairy.analysis.through_focus import (
        through_focus_scan,
        through_focus_scan_jax,
    )

    N, dx, wl = 48, 4e-6, 1.55e-6
    Z = np.zeros((N, N), dtype=np.complex128)
    z = np.linspace(-1e-3, 1e-3, 5)
    R = 40e-6

    ref = through_focus_scan(Z, dx, wl, z, bucket_radius=R)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        got = through_focus_scan_jax(Z, dx, wl, z, bucket_radius=R)

    # Independent oracle: zero field -> zero moments everywhere.
    assert np.allclose(np.nan_to_num(got.rms_radius, nan=-1.0), 0.0)
    assert np.array_equal(ref.rms_radius, got.rms_radius)
    assert ref.best_focus_spot == got.best_focus_spot == 0.0
    # Bucket power of a dark field is 0 on both backends.
    assert np.array_equal(ref.power_in_bucket, got.power_in_bucket)
    assert np.all(got.power_in_bucket == 0.0)


def test_numpy_jax_bucket_parity_smooth():
    """On a smooth Gaussian (no boundary/zero edge cases exercised) the two
    backends' bucket and rms agree tightly -- the shared metric path means
    the only residual difference is the propagated field itself (numpy vs
    jax FFT), which is ~1e-9, not a formula divergence."""
    pytest.importorskip('jax')
    from lumenairy.analysis.through_focus import (
        through_focus_scan,
        through_focus_scan_jax,
    )

    N, dx, wl = 96, 4e-6, 1.55e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    w0 = 60e-6
    E = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)
    z = np.linspace(-2e-3, 2e-3, 7)
    R = 80e-6

    ref = through_focus_scan(E, dx, wl, z, bucket_radius=R, ideal_peak=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        got = through_focus_scan_jax(E, dx, wl, z, bucket_radius=R,
                                     ideal_peak=1.0)

    for name in ('power_in_bucket', 'rms_radius'):
        a = np.asarray(getattr(ref, name), float)
        b = np.asarray(getattr(got, name), float)
        rel = np.max(np.abs(a - b) / (np.abs(a) + 1e-30))
        assert rel < 1e-9, f'{name} parity broke: max rel {rel:.2e}'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
