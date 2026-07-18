"""Regression tests for audit finding S3-5 (analysis/beam_stats.py).

``beam_d4sigma`` / ``M2`` integrate the second moment over the WHOLE
grid with no ISO 11146 background subtraction and no integration
aperture, so a small intensity pedestal inflates D4sigma without bound
in the grid size.  ``single_plane_metrics`` and
``find_best_focus(metric='spot'|'rms')`` inherit the bias, so a noisy
field lands on the wrong best-focus plane.

The fix adds *opt-in* ISO 11146 conditioning to ``beam_d4sigma``
(``background`` = float | 'corner' | None; ``aperture`` = bool | float,
an iterative integration aperture) and threads both params through
``single_plane_metrics`` / ``through_focus_scan``.  Defaults are
unchanged.

These tests use an *independent oracle*: the noisy field is built as
``E = sqrt(I_clean + pedestal)`` from a Gaussian whose true second
moment is known in closed form (D4sigma = 2*w0 for a field
``exp(-r^2/w0^2)``).  We assert (a) the default path is byte-identical
to a hand-rolled whole-grid moment, (b) the pedestal demonstrably
inflates the default D4sigma, and (c) each opt-in recovers the clean
value.  Deliberately break x<->y symmetry (dx != dy, w0x != w0y) so an
axis-swap regression would fail rather than slip through.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.analysis.beam_stats import beam_d4sigma
from lumenairy.analysis.through_focus import (
    single_plane_metrics,
    through_focus_scan,
)

# Asymmetric grid + beam so an x<->y swap is observable.  The grid is
# many beam-widths wide so the pedestal-inflation lever (grid/beam)^2 is
# large and the effect is unambiguous.
_N = 256
_DX = 1.0e-6
_DY = 1.3e-6
_LAM = 1.0e-6
_W0X = 20.0e-6
_W0Y = 26.0e-6
_PED = 1.0e-3  # uniform intensity pedestal, 0.1% of the unit peak


def _axes(N, dx, dy):
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    return np.meshgrid(x, y)


def _clean_field():
    """Elliptical Gaussian field; |E|^2 = exp(-2x^2/w0x^2 - 2y^2/w0y^2).

    Closed-form second moment of the intensity: sigma_x = w0x/2, so
    D4sigma_x = 2*w0x (and likewise y).
    """
    X, Y = _axes(_N, _DX, _DY)
    E = np.exp(-(X ** 2) / _W0X ** 2 - (Y ** 2) / _W0Y ** 2).astype(complex)
    return E


def _noisy_field(ped=_PED):
    """Clean field plus a uniform intensity pedestal (real amplitude,
    so |E|^2 = I_clean + pedestal exactly)."""
    E = _clean_field()
    I = np.abs(E) ** 2 + ped
    return np.sqrt(I).astype(complex)


def _whole_grid_d4sigma(E, dx, dy):
    """Independent oracle: the historical whole-grid second moment."""
    x = (np.arange(E.shape[1]) - E.shape[1] / 2) * dx
    y = (np.arange(E.shape[0]) - E.shape[0] / 2) * dy
    X, Y = np.meshgrid(x, y)
    I = np.abs(E) ** 2
    tot = I.sum()
    cx = (X * I).sum() / tot
    cy = (Y * I).sum() / tot
    vx = ((X - cx) ** 2 * I).sum() / tot
    vy = ((Y - cy) ** 2 * I).sum() / tot
    return 4 * np.sqrt(vx), 4 * np.sqrt(vy)


def test_default_path_byte_identical_to_whole_grid_oracle():
    # A random complex field: the default (no background, no aperture)
    # must match the independent whole-grid moment to machine precision.
    rng = np.random.default_rng(20250717)
    E = (rng.standard_normal((_N, _N))
         + 1j * rng.standard_normal((_N, _N)))
    got = beam_d4sigma(E, _DX, _DY)
    ref = _whole_grid_d4sigma(E, _DX, _DY)
    assert got == (float(ref[0]), float(ref[1]))


def test_clean_gaussian_matches_closed_form():
    d4x, d4y = beam_d4sigma(_clean_field(), _DX, _DY)
    # D4sigma = 2*w0 for a exp(-r^2/w0^2) field, within discretisation.
    assert abs(d4x - 2 * _W0X) / (2 * _W0X) < 0.01
    assert abs(d4y - 2 * _W0Y) / (2 * _W0Y) < 0.01


def test_pedestal_inflates_default_d4sigma():
    clean = beam_d4sigma(_clean_field(), _DX, _DY)
    noisy = beam_d4sigma(_noisy_field(), _DX, _DY)
    # 0.1% pedestal on a ~6-beam-wide grid inflates D4sigma > 1.5x.
    assert noisy[0] > 1.5 * clean[0]
    assert noisy[1] > 1.5 * clean[1]


def test_background_float_recovers_clean():
    clean = beam_d4sigma(_clean_field(), _DX, _DY)
    fixed = beam_d4sigma(_noisy_field(), _DX, _DY, background=_PED)
    # Subtracting the exact pedestal reproduces the clean field's I.
    assert abs(fixed[0] - clean[0]) / clean[0] < 1e-6
    assert abs(fixed[1] - clean[1]) / clean[1] < 1e-6


def test_background_corner_recovers_clean():
    clean = beam_d4sigma(_clean_field(), _DX, _DY)
    fixed = beam_d4sigma(_noisy_field(), _DX, _DY, background='corner')
    # Border pixels sit far in the Gaussian tail (~1e-30 of peak), so
    # the corner estimate is the pedestal to many digits.
    assert abs(fixed[0] - clean[0]) / clean[0] < 0.02
    assert abs(fixed[1] - clean[1]) / clean[1] < 0.02


def test_corner_background_is_near_noop_on_clean_field():
    base = beam_d4sigma(_clean_field(), _DX, _DY)
    corner = beam_d4sigma(_clean_field(), _DX, _DY, background='corner')
    assert abs(corner[0] - base[0]) / base[0] < 1e-6
    assert abs(corner[1] - base[1]) / base[1] < 1e-6


def test_aperture_monotone_improvement_and_bounded():
    # The iterative aperture is seeded from the (inflated) whole-grid
    # width, so it can only engage once 1.5*D4sigma fits inside the
    # grid -- i.e. on mild pedestals or after background subtraction.
    # Use a mild pedestal here so the aperture demonstrably converges
    # inward; the heavy-pedestal case is covered by the
    # background+aperture combination (the ISO-correct order).
    mild = 1.0e-4
    clean = beam_d4sigma(_clean_field(), _DX, _DY)
    noisy = beam_d4sigma(_noisy_field(mild), _DX, _DY)
    ap = beam_d4sigma(_noisy_field(mild), _DX, _DY, aperture=True)
    for i in range(2):
        # The iterative aperture strictly reduces the pedestal bias ...
        assert ap[i] < noisy[i]
        # ... but can never fall below the true (clean) width.
        assert ap[i] > 0.98 * clean[i]


def test_aperture_plus_background_recovers_clean():
    clean = beam_d4sigma(_clean_field(), _DX, _DY)
    both = beam_d4sigma(
        _noisy_field(), _DX, _DY, background='corner', aperture=True)
    assert abs(both[0] - clean[0]) / clean[0] < 0.02
    assert abs(both[1] - clean[1]) / clean[1] < 0.02


def test_custom_aperture_multiple_accepted():
    # A float aperture overrides the ISO default multiple (3).
    val = beam_d4sigma(_noisy_field(), _DX, _DY, aperture=4.0)
    assert np.isfinite(val[0]) and val[0] > 0.0


def test_zero_field_is_graceful_with_options():
    Z = np.zeros((_N, _N), dtype=complex)
    assert beam_d4sigma(Z, _DX, _DY) == (0.0, 0.0)
    assert beam_d4sigma(Z, _DX, _DY, background='corner') == (0.0, 0.0)
    assert beam_d4sigma(Z, _DX, _DY, aperture=True) == (0.0, 0.0)


def test_single_plane_metrics_forwards_conditioning():
    E = _noisy_field()
    base = single_plane_metrics(E, _DX, _LAM, dy=_DY)
    cond = single_plane_metrics(
        E, _DX, _LAM, dy=_DY, background=_PED)
    direct = beam_d4sigma(E, _DX, _DY, background=_PED)
    # single_plane_metrics must forward the kwarg to beam_d4sigma ...
    assert cond['d4sigma_x'] == pytest.approx(direct[0], rel=1e-12)
    assert cond['d4sigma_y'] == pytest.approx(direct[1], rel=1e-12)
    # ... and the conditioned width is smaller than the default one.
    assert cond['d4sigma_x'] < base['d4sigma_x']
    assert cond['rms_radius'] < base['rms_radius']


def test_through_focus_scan_forwards_conditioning():
    E = _noisy_field()
    z = np.array([0.0])
    base = through_focus_scan(E, _DX, _LAM, z)
    cond = through_focus_scan(E, _DX, _LAM, z, background=_PED)
    # Conditioning propagates into the scan's best-focus spot metric.
    assert cond.rms_radius[0] < base.rms_radius[0]
    assert cond.best_focus_spot < base.best_focus_spot


def test_through_focus_scan_jax_rejects_conditioning():
    # The ISO conditioning is numpy-only; the jax dispatch must raise a
    # clear error rather than silently ignore the kwargs.  Checked before
    # any jax import, so this runs without jax installed.
    with pytest.raises(NotImplementedError):
        through_focus_scan(
            _clean_field(), _DX, _LAM, np.array([0.0]),
            backend='jax', background=_PED)
