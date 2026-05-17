"""Benchmarks for the v4.12.0 asymptotic-propagator perf work.

Two workloads are pinned here.  The pre-v4.12.0 baselines are not
captured as separate fixture files because the pre-fix
:func:`propagate_hf_chebyshev_quadrature` had a pre-existing
shape-mismatch bug when called with a 2-D ``S1X`` input grid and
scalar ``s2`` outputs (the inner ``fit.eval_phi(S1X, S1Y, s2x, s2y)``
call would crash before the chunk loop completed -- see
``tests/unit/test_audit_fixes_v4_11_2_hfpi_hf.py:371`` for the
upstream note on this latent bug).  Wall times reported here are
therefore the v4.12.0 post-fix numbers; the speedup factor reported
in the audit (3-10x for #5, 10-50x for #6) is derived from the
Python-loop count that the fix eliminates.

Workloads pinned here:

- propagate_modal_asymptotic 64x64 output, 3 modes  (perf #5)
- propagate_modal_asymptotic 64x64 output, 10 modes (perf #5, shows mode-count scaling)
- propagate_hf_chebyshev_quadrature 64x64 output, 256x256 input (perf #6)
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as lm
from lumenairy.propagators.asymptotic import (
    clear_lg_polynomial_cache,
    fit_canonical_polynomials,
    fit_hf_polynomials,
    propagate_hf_chebyshev_quadrature,
    propagate_modal_asymptotic,
)


def _make_singlet():
    rx = lm.make_singlet(
        R1=20e-3, R2=-20e-3, d=2e-3,
        glass='N-BK7', aperture=10e-3,
    )
    rx['object_distance'] = 0.1
    return rx


@pytest.fixture(scope='module')
def canonical_fit():
    rx = _make_singlet()
    return fit_canonical_polynomials(
        rx, wavelength=1.31e-6,
        source_box_half=20e-6, pupil_box_half=0.02,
        n_field=8, n_pupil=8, poly_order=6,
    )


@pytest.fixture(scope='module')
def hf_fit():
    rx = _make_singlet()
    return fit_hf_polynomials(
        rx, wavelength=1.31e-6,
        source_box_half=20e-6, pupil_box_half=0.02,
        n_field=8, n_pupil=8, poly_order=4,
    )


# ---------------------------------------------------------------------
# Perf #5  -- modal asymptotic propagator (LG polynomial hoist & cache)
# ---------------------------------------------------------------------


def test_bench_modal_asymptotic_64_3_modes(benchmark, canonical_fit):
    """64x64 output, 3 source modes x 1 pupil mode = 3 inner products
    per pixel.  Pre-v4.12.0 this was 3 * 64 * 64 = 12,288 redundant
    rebuilds of ``lg_polynomial``; v4.12.0 hoists to 3 builds total."""
    fit = canonical_fit
    L = fit.s2x_halfrange * 0.3
    n = 64
    ax = np.linspace(-L, L, n) + fit.s2x_centre
    ay = np.linspace(-L, L, n) + fit.s2y_centre
    X, Y = np.meshgrid(ax, ay, indexing='xy')
    source_amps = {
        (0, 0): 1.0 + 0.0j,
        (1, 0): 0.3 - 0.1j,
        (0, 1): 0.2 + 0.4j,
    }
    pupil_amps = {(0, 0): 1.0 + 0.0j}

    def run():
        clear_lg_polynomial_cache()
        return propagate_modal_asymptotic(
            fit,
            source_point=(0.0, 0.0),
            source_amplitudes=source_amps,
            pupil_amplitudes=pupil_amps,
            w_s=20e-6, w_p=0.02,
            v2_centre=(fit.v2x_centre, fit.v2y_centre),
            s2_grid_x=X, s2_grid_y=Y,
        )

    benchmark(run)


def test_bench_modal_asymptotic_64_10_modes(benchmark, canonical_fit):
    """64x64 output, 5 source x 2 pupil = 10 inner products per pixel.
    Mode-count is doubled vs. the 3-mode benchmark, so the per-pixel
    cost grows by 10/3 ~ 3.3x but the rebuild-per-pixel overhead
    grows by 10/3 too -- the hoist payoff scales linearly with mode
    count, so the relative speedup is constant across mode counts."""
    fit = canonical_fit
    L = fit.s2x_halfrange * 0.3
    n = 64
    ax = np.linspace(-L, L, n) + fit.s2x_centre
    ay = np.linspace(-L, L, n) + fit.s2y_centre
    X, Y = np.meshgrid(ax, ay, indexing='xy')
    source_amps = {
        (0, 0): 1.0 + 0.0j,
        (1, 0): 0.3 - 0.1j,
        (0, 1): 0.2 + 0.4j,
        (1, 1): 0.15 + 0.05j,
        (0, 2): 0.1 + 0.0j,
    }
    pupil_amps = {
        (0, 0): 1.0 + 0.0j,
        (0, 1): 0.1 + 0.2j,
    }

    def run():
        clear_lg_polynomial_cache()
        return propagate_modal_asymptotic(
            fit,
            source_point=(0.0, 0.0),
            source_amplitudes=source_amps,
            pupil_amplitudes=pupil_amps,
            w_s=20e-6, w_p=0.02,
            v2_centre=(fit.v2x_centre, fit.v2y_centre),
            s2_grid_x=X, s2_grid_y=Y,
        )

    benchmark(run)


# ---------------------------------------------------------------------
# Perf #6  -- HF Chebyshev-quadrature vectorised chunk loop
# ---------------------------------------------------------------------


def test_bench_hf_chebyshev_quadrature_64_256(benchmark, hf_fit):
    """64x64 output on 256x256 input -- 4096 outputs over 65,536
    inputs, ~2.7e8 phase evaluations total.  Pre-v4.12.0 ran one
    output at a time (4096 Python iterations); v4.12.0 broadcasts
    over a configurable chunk."""
    fit = hf_fit
    N_in = 256
    N_out = 64
    ax = np.linspace(-fit.s1x_halfrange * 0.5,
                      fit.s1x_halfrange * 0.5, N_in) + fit.s1x_centre
    ay = np.linspace(-fit.s1y_halfrange * 0.5,
                      fit.s1y_halfrange * 0.5, N_in) + fit.s1y_centre
    bx = np.linspace(-fit.s2x_halfrange * 0.3,
                      fit.s2x_halfrange * 0.3, N_out) + fit.s2x_centre
    by = np.linspace(-fit.s2y_halfrange * 0.3,
                      fit.s2y_halfrange * 0.3, N_out) + fit.s2y_centre
    X, Y = np.meshgrid(ax, ay, indexing='xy')
    w = 1.5e-5
    E = np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)

    benchmark(
        propagate_hf_chebyshev_quadrature,
        fit, E, ax, ay, bx, by,
        apply_van_vleck=True, chunk_output=64,
    )


def test_bench_hf_chebyshev_quadrature_64_256_no_vanvleck(benchmark, hf_fit):
    """As above without the Van Vleck factor (faster path -- isolates
    the bare ``exp(2pi i phi) * E`` reduction cost)."""
    fit = hf_fit
    N_in = 256
    N_out = 64
    ax = np.linspace(-fit.s1x_halfrange * 0.5,
                      fit.s1x_halfrange * 0.5, N_in) + fit.s1x_centre
    ay = np.linspace(-fit.s1y_halfrange * 0.5,
                      fit.s1y_halfrange * 0.5, N_in) + fit.s1y_centre
    bx = np.linspace(-fit.s2x_halfrange * 0.3,
                      fit.s2x_halfrange * 0.3, N_out) + fit.s2x_centre
    by = np.linspace(-fit.s2y_halfrange * 0.3,
                      fit.s2y_halfrange * 0.3, N_out) + fit.s2y_centre
    X, Y = np.meshgrid(ax, ay, indexing='xy')
    w = 1.5e-5
    E = np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)

    benchmark(
        propagate_hf_chebyshev_quadrature,
        fit, E, ax, ay, bx, by,
        apply_van_vleck=False, chunk_output=64,
    )
