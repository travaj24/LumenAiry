"""Audit v5.17.0 P1-06: LG/HG mode-stack cache key must include the grid ORIGIN.

Pre-fix, ``_lg_mode_conj_stack`` / ``_hg_mode_conj_stack`` keyed the cache
on (orders, shape, waist, centre, pitch, dtype) but NOT the grid origin,
so two same-shape/same-pitch grids at different offsets (e.g. a shifted
ROI) collided: the second call silently received modes evaluated at the
FIRST grid's physical coordinates.  Repro at N=64, dx=2e-6, w=20e-6,
grid B = grid A shifted by +8*dx in x: fresh-cache a00 = 0.9999995 but
cache-hit a00 = 0.7261490 (~27% wrong, no warning).  Every shifted-grid
test below FAILS on the pre-fix code (stash A/B verified) because the
collision returns the grid-A stack for the grid-B call.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

import lumenairy.propagators.asymptotic_modes as am
from lumenairy.propagators.asymptotic_modes import (
    clear_lg_mode_stack_cache,
    decompose_hg,
    decompose_lg,
)

_N = 64
_DX = 2e-6
_W = 20e-6


def _axis(shift_samples: float = 0.0) -> np.ndarray:
    return (np.arange(_N) - _N // 2 + shift_samples) * _DX


def _gauss00(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """On-axis LG00 == HG00 fundamental Gaussian, unit L2 norm."""
    X, Y = np.meshgrid(x, y, indexing='xy')
    return np.sqrt(2.0 / np.pi) / _W * np.exp(-(X ** 2 + Y ** 2) / _W ** 2)


class TestAuditP1_06_LgHgCacheOrigin:
    """Shifted same-shape/same-pitch grids must NOT share a cache entry."""

    def test_decompose_lg_shifted_x_grid_no_collision(self):
        xA = _axis()
        xB = _axis(8.0)  # +8*dx shift: same N, same pitch, new origin
        clear_lg_mode_stack_cache()
        a00_fresh = decompose_lg(_gauss00(xB, xA), xB, xA, _W, 0, 0)[(0, 0)]
        clear_lg_mode_stack_cache()
        decompose_lg(_gauss00(xA, xA), xA, xA, _W, 0, 0)  # populate grid A
        a00 = decompose_lg(_gauss00(xB, xA), xB, xA, _W, 0, 0)[(0, 0)]
        # Pre-fix the cache collision gives |a00| = 0.7261490 here.
        assert abs(a00 - a00_fresh) < 1e-12, (
            f'shifted-grid LG overlap {a00} != fresh-cache {a00_fresh}'
        )
        assert abs(a00) > 0.999
        assert len(am._LG_MODE_STACK_CACHE) == 2  # two distinct entries

    def test_decompose_lg_shifted_y_grid_no_collision(self):
        """Y origin is captured separately from X (asymmetric shift)."""
        xA = _axis()
        yB = _axis(-5.0)  # shift ONLY the y axis
        clear_lg_mode_stack_cache()
        a00_fresh = decompose_lg(_gauss00(xA, yB), xA, yB, _W, 0, 0)[(0, 0)]
        clear_lg_mode_stack_cache()
        decompose_lg(_gauss00(xA, xA), xA, xA, _W, 0, 0)
        a00 = decompose_lg(_gauss00(xA, yB), xA, yB, _W, 0, 0)[(0, 0)]
        assert abs(a00 - a00_fresh) < 1e-12, (
            f'shifted-y LG overlap {a00} != fresh-cache {a00_fresh}'
        )
        assert len(am._LG_MODE_STACK_CACHE) == 2

    def test_decompose_hg_shifted_grid_no_collision(self):
        xA = _axis()
        xB = _axis(8.0)
        clear_lg_mode_stack_cache()
        a00_fresh = decompose_hg(
            _gauss00(xB, xA), xB, xA, _W, _W, 0, 0)[(0, 0)]
        clear_lg_mode_stack_cache()
        decompose_hg(_gauss00(xA, xA), xA, xA, _W, _W, 0, 0)
        a00 = decompose_hg(_gauss00(xB, xA), xB, xA, _W, _W, 0, 0)[(0, 0)]
        assert abs(a00 - a00_fresh) < 1e-12, (
            f'shifted-grid HG overlap {a00} != fresh-cache {a00_fresh}'
        )
        assert abs(a00) > 0.999
        assert len(am._HG_MODE_STACK_CACHE) == 2

    def test_same_grid_repeat_still_cache_hits(self):
        """Non-regression: identical-grid repeats reuse one entry and
        return bit-identical values (the v4.14.0 perf contract)."""
        xA = _axis()
        field = _gauss00(xA, xA)
        clear_lg_mode_stack_cache()
        out1 = decompose_lg(field, xA, xA, _W, 1, 1)
        assert len(am._LG_MODE_STACK_CACHE) == 1
        out2 = decompose_lg(field, xA, xA, _W, 1, 1)
        assert len(am._LG_MODE_STACK_CACHE) == 1  # hit, not a new entry
        assert set(out1) == set(out2)
        for k in out1:
            assert out1[k] == out2[k]
