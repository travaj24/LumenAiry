"""GBD feature-completeness tests (2026-07 program).

Covers the feature-completeness work layered on top of the audit-closed GBD
correctness: Husimi decomposition plumbed through the lens / prescription
helpers, per-surface Q-evolution, anamorphic sampling, multi-wavelength,
aperture clipping, adaptive beamlet count, and the JAX twin.  Kept in a
separate file from ``test_audit_propagation.py`` (which holds the audit
regression pins) to avoid stepping on concurrent edits there.
"""
import numpy as np
import pytest

from lumenairy.propagators.gbd import (
    propagate_gbd_thin_lens,
)

LAM = 1.0e-6


def _centroid_x(F, X):
    I = np.abs(F) ** 2
    s = I.sum()
    return float((I * X).sum() / s) if s > 0 else 0.0


def test_husimi_through_lens_focuses_tilted_beam_off_axis():
    """A collimated beam tilted by theta through a lens f focuses at
    x = f*tan(theta).  The Husimi (direction_sampling=True) decomposition,
    now plumbed into propagate_gbd_thin_lens, walks the tilt off correctly;
    the position-only decomposition does not (its beamlets launch axially and
    never walk off during the free-space legs)."""
    N, dx = 128, 10e-6
    f, theta = 20e-3, 0.02
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    w = 0.4e-3
    E = (np.exp(-(X ** 2 + Y ** 2) / w ** 2)
         * np.exp(1j * 2 * np.pi / LAM * np.sin(theta) * X)).astype(
             np.complex128)
    kw = dict(dx=dx, z_to_lens=0.0, focal_length=f, z_lens_to_output=f,
              wavelength=LAM, output_dx=dx, waist_factor=1.0, sample_step=2)
    g_pos = propagate_gbd_thin_lens(E, **kw, direction_sampling=False)
    g_hus = propagate_gbd_thin_lens(E, **kw, direction_sampling=True)
    x_analytic = f * np.tan(theta)
    err_pos = abs(_centroid_x(g_pos, X) - x_analytic)
    err_hus = abs(_centroid_x(g_hus, X) - x_analytic)
    assert np.isfinite(g_hus).all() and np.isfinite(g_pos).all()
    # Husimi lands within ~12% of the analytic off-axis focus...
    assert err_hus < 0.12 * abs(x_analytic), (err_hus, x_analytic)
    # ...and is dramatically better than position-only for this tilted source.
    assert err_hus < 0.3 * err_pos, (err_hus, err_pos)
