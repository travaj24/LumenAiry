"""Pinning tests for the v4.13.2 audit (Agent D scope, six items).

Covers six audit items handled by Agent D in the v4.13.1 -> v4.13.2
pass.  Each test pins exactly one finding so a regression points
straight at the responsible fix:

* **D.1 (C-P0-3)** -- :func:`lumenairy.io.storage.load_plane_slice`
  documented to return ``np.ndarray``; actually returns
  ``(arr, attrs)`` in both backends.  v4.13.2 pinned the docstring /
  annotation to the tuple shape that callers have always destructured.
* **D.2 (C-P0-4)** -- CODE V ``.seq`` reader dropped the final
  surface's THI (BFL) on read.  v4.13.2 surfaces the BFL via the
  ``back_focal_length`` key on the returned prescription dict.
* **D.3 (C-P0-5)** -- Quadoa ``.qos`` reader dropped the final
  surface's THI on read; the round-trip with :func:`export_quadoa_qos`
  was lossy for BFL.  v4.13.2 reads the top-level ``back_focal_length``
  JSON field plus the trailing surface as a fallback.
* **D.4** -- :meth:`Source.fiber_mode` forwarded a user-supplied
  ``dy=`` to :func:`create_fiber_mode`, which previously rejected it.
  v4.13.2 widens ``create_fiber_mode`` to accept ``dy=`` and threads
  it through the underlying Gaussian helper.
* **D.5 (C-P1-3)** -- :func:`propagate_through_system` only threaded
  ``dy`` through the free-space ``propagate_*`` steps; every other
  element handler passed the outer function's ``dx``/``dy`` literals
  (squaring anamorphic grids to ``dx``).  v4.13.2 routes every handler
  through ``current_dx`` / ``current_dy``.

D.6 (broken wiki anchor + ``__all__`` reshuffles + duplicate
``reset_fft_backend`` import) are verified by ``python -c "import
lumenairy"`` succeeding plus a smoke test asserting the moved names
are still importable.

Author: Andrew Traverso
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

import lumenairy
from lumenairy import (
    Source,
    propagate_through_system,
)
from lumenairy.io.prescriptions import (
    export_codev_seq,
    export_quadoa_qos,
    load_codev_seq,
    load_quadoa_qos,
)
from lumenairy.io.storage import append_plane, load_plane_slice
from lumenairy.sources.core import create_fiber_mode


# ====================================================================
# D.1 -- load_plane_slice return-type matches docstring (tuple)
# ====================================================================

def test_d1_load_plane_slice_returns_tuple_h5(tmp_path):
    """load_plane_slice must return ``(arr, attrs)`` per the v4.13.2
    docstring -- a bare ``np.ndarray`` would silently break every
    caller that destructures."""
    h5py = pytest.importorskip('h5py')

    # Build a small HDF5 store with one plane.
    fpath = str(tmp_path / 'one_plane.h5')
    E = np.arange(64, dtype=np.complex64).reshape(8, 8)
    append_plane(fpath, E, dx=1e-6, label='disk-aperture')

    result = load_plane_slice(
        fpath, plane_index=0,
        y_slice=slice(0, 4), x_slice=slice(0, 4))

    # Documented return type is (ndarray, dict).
    assert isinstance(result, tuple), (
        f"load_plane_slice must return a tuple, got {type(result).__name__}")
    assert len(result) == 2, (
        f"load_plane_slice must return a 2-tuple, got len={len(result)}")
    arr, attrs = result
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (4, 4)
    assert isinstance(attrs, dict)
    assert 'label' in attrs


# ====================================================================
# D.2 -- CODE V .seq BFL round-trip
# ====================================================================

def test_d2_codev_seq_roundtrip_preserves_bfl(tmp_path):
    """A `.seq` file whose final surface encodes the BFL must
    round-trip through ``load_codev_seq`` -- the BFL must be
    accessible (v4.13.2 surfaces it via the ``back_focal_length`` key
    on the returned prescription)."""
    # Hand-written .seq with an SI image plane THI = 0.012345 m.
    seq_text = '\n'.join([
        '! Hand-written CODE V test file',
        'LEN NEW',
        'DIM M',
        'WL 1550.0',
        'REF 1',
        'APE F1 CIR R 0.005',
        '',
        '! Object surface',
        'SO',
        '  RDY INFINITY',
        '  THI INFINITY',
        '',
        '! Surface 1 (stop)',
        'S1',
        '  STO',
        '  RDY 0.025000',
        '  THI 0.003000',
        '  GLA BK7',
        '',
        '! Surface 2',
        'S2',
        '  RDY -0.025000',
        '  THI 0.012345',
        '',
        '! Image surface',
        'SI',
        '  RDY INFINITY',
        '  THI 0.012345',
        '',
        'GO',
        'END',
    ])

    fpath = str(tmp_path / 'roundtrip_bfl.seq')
    Path(fpath).write_text(seq_text, encoding='utf-8')

    pres = load_codev_seq(fpath)

    # The library's traditional thicknesses convention has len ==
    # len(surfaces) - 1; the trailing BFL gap must be surfaced via
    # ``back_focal_length`` rather than silently dropped.
    assert 'back_focal_length' in pres, (
        "v4.13.2 must surface the trailing-surface THI as a BFL key.")
    assert pres['back_focal_length'] == pytest.approx(0.012345, rel=1e-6)
    assert len(pres['thicknesses']) == len(pres['surfaces']) - 1


def test_d2_codev_seq_export_roundtrip_preserves_bfl(tmp_path):
    """Round-trip a real lumenairy prescription through
    ``export_codev_seq`` -> ``load_codev_seq`` and confirm the BFL is
    preserved end-to-end."""
    pres = {
        'name': 'test_lens',
        'aperture_diameter': 0.010,
        'surfaces': [
            {'radius': 0.025, 'conic': 0.0,
             'glass_before': 'air', 'glass_after': 'BK7'},
            {'radius': -0.025, 'conic': 0.0,
             'glass_before': 'BK7', 'glass_after': 'air'},
        ],
        'thicknesses': [0.003],  # only inter-surface gap
    }
    bfl = 0.04321
    fpath = str(tmp_path / 'roundtrip.seq')
    export_codev_seq(
        pres, fpath, wavelength=1.55e-6, stop_surface=0,
        back_focal_length=bfl)
    pres_back = load_codev_seq(fpath)
    assert 'back_focal_length' in pres_back
    assert pres_back['back_focal_length'] == pytest.approx(bfl, rel=1e-5)


# ====================================================================
# D.3 -- Quadoa .qos BFL round-trip
# ====================================================================

def test_d3_quadoa_qos_roundtrip_preserves_bfl(tmp_path):
    """``export_quadoa_qos`` writes ``back_focal_length`` at JSON top
    level; ``load_quadoa_qos`` must read it back into the returned
    prescription dict."""
    pres = {
        'name': 'doublet',
        'aperture_diameter': 0.010,
        'surfaces': [
            {'radius': 0.025, 'conic': 0.0,
             'glass_before': 'air', 'glass_after': 'BK7'},
            {'radius': -0.025, 'conic': 0.0,
             'glass_before': 'BK7', 'glass_after': 'air'},
        ],
        'thicknesses': [0.003],
    }
    bfl = 0.03579
    fpath = str(tmp_path / 'roundtrip.qos')
    export_quadoa_qos(
        pres, fpath, wavelength=1.55e-6, stop_surface=0,
        back_focal_length=bfl)
    pres_back = load_quadoa_qos(fpath)
    assert 'back_focal_length' in pres_back, (
        "v4.13.2 must surface ``back_focal_length`` on `.qos` round-trip.")
    assert pres_back['back_focal_length'] == pytest.approx(bfl, rel=1e-5)
    # thicknesses length convention preserved.
    assert len(pres_back['thicknesses']) == len(pres_back['surfaces']) - 1


# ====================================================================
# D.4 -- Source.fiber_mode accepts dy=
# ====================================================================

def test_d4_fiber_mode_accepts_dy_kwarg():
    """``Source.fiber_mode(..., dy=2*dx, ...)`` must build a Source
    with ``dy`` distinct from ``dx`` (v4.13.2 widens
    ``create_fiber_mode`` to accept ``dy=`` so the wrap-level
    ``Source(..., dy=...)`` actually fires)."""
    dx = 1e-6
    dy_in = 2e-6
    src = Source.fiber_mode(
        mode_field_diameter=10e-6,
        N=64, dx=dx, dy=dy_in,
        wavelength=1.55e-6, na=0.12)
    assert src.dx == pytest.approx(dx)
    assert src.dy == pytest.approx(dy_in), (
        f"Source.fiber_mode must propagate dy={dy_in}, got {src.dy}")


def test_d4_create_fiber_mode_accepts_dy_directly():
    """``create_fiber_mode`` itself accepts ``dy=`` after v4.13.2
    (option-(a) fix)."""
    dx = 1e-6
    dy_in = 1.5e-6
    E, x, y = create_fiber_mode(
        N=64, dx=dx, dy=dy_in,
        wavelength=1.55e-6, mode_field_diameter=10e-6)
    # y-axis step must reflect dy, not dx.
    assert (y[1] - y[0]) == pytest.approx(dy_in)
    assert (x[1] - x[0]) == pytest.approx(dx)


# ====================================================================
# D.5 -- propagate_through_system threads dy through every element
# ====================================================================

def test_d5_propagate_through_system_threads_dy_aperture():
    """An anamorphic source through an aperture must keep dy == dy_in
    (pre-fix the aperture handler called ``apply_aperture(E, dx, ...)``
    only, squaring dy -> dx)."""
    N = 64
    dx = 1e-6
    dy = 2e-6
    wavelength = 1.55e-6
    # Uniform plane-wave input on an anamorphic grid.
    E_in = np.ones((N, N), dtype=np.complex64)

    elements = [
        {'type': 'aperture',
         'shape': 'circular',
         'params': {'diameter': 20e-6}},
    ]
    result = propagate_through_system(
        E_in, elements, wavelength, dx=dx, dy=dy,
        return_result=True)
    # The aperture is an in-place mask; result.dx / result.dy are the
    # final grid pitches that the system maintained throughout.
    assert result.dx == pytest.approx(dx)
    assert result.dy == pytest.approx(dy), (
        f"propagate_through_system must preserve anamorphic dy through "
        f"element handlers; got dy={result.dy}, expected {dy}.")


def test_d5_propagate_through_system_threads_dy_mask():
    """Same threading guarantee for ``mask`` element."""
    N = 64
    dx = 1e-6
    dy = 3e-6
    wavelength = 1.55e-6
    E_in = np.ones((N, N), dtype=np.complex64)
    mask = np.ones((N, N), dtype=np.complex64)  # transparent mask
    elements = [{'type': 'mask', 'mask': mask}]
    result = propagate_through_system(
        E_in, elements, wavelength, dx=dx, dy=dy,
        return_result=True)
    assert result.dx == pytest.approx(dx)
    assert result.dy == pytest.approx(dy)


# ====================================================================
# D.6 -- smoke checks for __all__ retiering and import dedup
# ====================================================================

def test_d6_tier_moves_keep_names_importable():
    """The four names moved between Tier 6 -> Tier 1/4 must remain
    importable from the top-level package after the reshuffle."""
    # Tier 1 (lens models)
    assert hasattr(lumenairy, 'apply_real_lens_traced_jax')
    assert hasattr(lumenairy, 'apply_real_lens_maslov_jax')
    # Tier 4 (tolerancing)
    assert hasattr(lumenairy, 'monte_carlo_tolerancing_jax')
    assert hasattr(lumenairy, 'monte_carlo_tolerancing_linearized')
    assert hasattr(lumenairy, 'tolerancing_report')


def test_d6_reset_fft_backend_still_callable():
    """The duplicate ``reset_fft_backend`` import was removed; the
    name must remain callable through the package namespace."""
    assert callable(lumenairy.reset_fft_backend)
