"""Tests for the v5.16.2 opt-in lens memory modes:

* ``sag_chunk_rows`` -- row-band phase screens (apply_real_lens) + row-band
  OPL-upsample/assembly (apply_real_lens_traced).  BYTE-IDENTICAL to the
  whole-grid path: every banded op is pointwise (or, for the order-1
  ``map_coordinates`` upsample, pointwise in the output), so the values are
  bit-equal -- pinned here with ``np.array_equal``.
* ``sag_dtype`` -- float32 geometry (coordinate/sag/opd lineage).  Default
  ``None`` -> float64 is byte-identical to prior releases; float32 is an
  accuracy trade validated by ``lens_sag_float32_opd_error``.

Determinism note: pyFFTW's ESTIMATE->MEASURE auto-promotion rebuilds plans
after a few calls at one shape, and MEASURE plans may round differently at
the ULP level *between calls*.  Byte-equality across separate lens calls is
therefore only guaranteed with auto-promote off, which the fixture pins
(matching the production runner configuration).
"""
import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements._lens_real import (
    apply_real_lens,
    get_lens_sag_dtype,
    lens_sag_float32_opd_error,
    set_lens_sag_dtype,
)
from lumenairy.elements._lens_traced import apply_real_lens_traced


@pytest.fixture(autouse=True)
def _deterministic_fft():
    """Pin FFT plan determinism across calls for byte-equality asserts."""
    la.set_fft_auto_promote(False)
    yield
    la.set_fft_auto_promote(True)


def _presc(decenter=None):
    s0 = {'radius': 12e-3, 'conic': -0.4,
          'aspheric_coeffs': {4: 2e4, 6: 5e6},
          'glass_before': 'air', 'glass_after': 'N-BK7'}
    if decenter is not None:
        s0 = dict(s0, decenter=decenter, aspheric_coeffs=None, conic=0.0)
    return {
        'name': 't', 'aperture_diameter': 3e-3,
        'surfaces': [
            s0,
            {'radius': -15e-3, 'conic': 0.0, 'aspheric_coeffs': {4: -1e4},
             'glass_before': 'N-BK7', 'glass_after': 'air'},
        ],
        'thicknesses': [2.5e-3],
    }


def _field(N, dx, w=1.2e-3, dtype=np.complex64):
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    return np.exp(-(X**2 + Y**2) / w**2).astype(dtype)


LAM = 1.31e-6


@pytest.mark.parametrize("N,dx", [(1024, 3e-6),   # numexpr band path (>=1Mi px)
                                  (512, 6e-6)])   # plain-numpy band path
def test_chunked_real_lens_byte_identical(N, dx):
    E = _field(N, dx)
    kw = dict(prescription=_presc(), wavelength=LAM, dx=dx)
    Ew = apply_real_lens(E.copy(), **kw)
    for cr in (N // 8, 111):   # divisor and non-divisor band sizes
        Ec = apply_real_lens(E.copy(), sag_chunk_rows=cr, **kw)
        assert np.array_equal(Ew, Ec), f"chunk_rows={cr} not byte-identical"


@pytest.mark.parametrize("preserve_input_phase", [True, False])
def test_chunked_traced_byte_identical(preserve_input_phase):
    N, dx = 1024, 3e-6
    E = _field(N, dx)
    kw = dict(prescription=_presc(), wavelength=LAM, dx=dx,
              ray_subsample=4, parallel_amp=False,
              preserve_input_phase=preserve_input_phase)
    Tw = apply_real_lens_traced(E.copy(), **kw)
    for cr in (N // 8, 111):
        Tc = apply_real_lens_traced(E.copy(), sag_chunk_rows=cr, **kw)
        assert np.array_equal(Tw, Tc), (
            f"pip={preserve_input_phase} chunk_rows={cr} not byte-identical")


def test_chunked_fallback_surface_byte_identical():
    """A decentered surface falls through to the whole-grid path (lazy grids
    built on demand); the result must still equal the unchunked run."""
    N, dx = 1024, 3e-6
    E = _field(N, dx)
    kw = dict(prescription=_presc(decenter=(30e-6, -20e-6)),
              wavelength=LAM, dx=dx)
    Ew = apply_real_lens(E.copy(), **kw)
    Ec = apply_real_lens(E.copy(), sag_chunk_rows=128, **kw)
    assert np.array_equal(Ew, Ec)


def test_chunked_sub1_falls_back_whole_grid():
    """sag_chunk_rows with ray_subsample=1 (full-grid Newton) must not break;
    the assembly chunking is guarded to the sub>1 path."""
    N, dx = 256, 8e-6
    E = _field(N, dx, w=0.4e-3)
    kw = dict(prescription=_presc(), wavelength=LAM, dx=dx,
              ray_subsample=1, parallel_amp=False)
    Tw = apply_real_lens_traced(E.copy(), **kw)
    Tc = apply_real_lens_traced(E.copy(), sag_chunk_rows=64, **kw)
    assert np.array_equal(Tw, Tc)


def test_sag_dtype_default_byte_identical():
    """Default (None) and explicit float64 produce identical bytes -- the
    shipped behaviour is unchanged."""
    N, dx = 1024, 3e-6
    E = _field(N, dx)
    kw = dict(prescription=_presc(), wavelength=LAM, dx=dx)
    E_def = apply_real_lens(E.copy(), **kw)
    E_f64 = apply_real_lens(E.copy(), sag_dtype=np.float64, **kw)
    assert np.array_equal(E_def, E_f64)


def test_sag_dtype_float32_close_and_composes_with_chunking():
    """float32 geometry error scales with TOTAL sag depth (per-pixel phase
    quantisation ~ k0 * OPD * eps_f32), so this deep singlet measures ~2e-3
    max field error -- large enough that the validator must flag it (see
    the validator test) but bounded.  A gentle lens sits at ~1e-5."""
    N, dx = 1024, 3e-6
    E = _field(N, dx)
    kw = dict(prescription=_presc(), wavelength=LAM, dx=dx)
    E64 = apply_real_lens(E.copy(), **kw)
    E32 = apply_real_lens(E.copy(), sag_dtype=np.float32, **kw)
    m = np.abs(E64).max()
    assert np.abs(E32 - E64).max() / m < 1e-2
    # float32 + chunking == float32 whole-grid (bit-equal)
    E32c = apply_real_lens(E.copy(), sag_dtype=np.float32,
                           sag_chunk_rows=128, **kw)
    assert np.array_equal(E32, E32c)


def test_set_lens_sag_dtype_roundtrip_and_validation():
    assert get_lens_sag_dtype() is np.float64
    set_lens_sag_dtype(np.float32)
    try:
        assert get_lens_sag_dtype() is np.float32
    finally:
        set_lens_sag_dtype(None)
    assert get_lens_sag_dtype() is np.float64
    with pytest.raises(ValueError):
        set_lens_sag_dtype(np.int32)


def test_lens_sag_float32_opd_error_validator():
    r = lens_sag_float32_opd_error(_presc(), LAM)
    assert set(['max_opd_error_waves', 'rms_opd_error_waves',
                'max_opd_error_nm', 'max_field_rel_error',
                'aperture_m', 'ok']).issubset(r)
    assert r['max_opd_error_waves'] < 0.02
    # The float32 field error is CONFIG-dependent (the f32 phase
    # perturbation interferes through the in-glass diffraction), so the
    # production-sampling A/B (dx=3e-6, N=1024 -- the regime where the
    # direct comparison measures ~2e-3) must veto this deep lens even
    # though the coarse default check and the radial OPD scan pass it.
    rp = lens_sag_float32_opd_error(_presc(), LAM,
                                    field_check_n=1024,
                                    field_check_dx=3e-6)
    assert rp['max_field_rel_error'] > 1e-3          # field A/B catches it
    assert rp['ok'] is False                          # and vetoes
    # A gentle (shallow-sag) lens passes both gates.
    gentle = {
        'name': 'g', 'aperture_diameter': 1.5e-3,
        'surfaces': [
            {'radius': 50e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'radius': -50e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'N-BK7', 'glass_after': 'air'},
        ],
        'thicknesses': [1.5e-3],
    }
    rg = lens_sag_float32_opd_error(gentle, LAM)
    assert rg['ok'] is True, rg
    with pytest.raises(ValueError):
        lens_sag_float32_opd_error(
            {'surfaces': _presc()['surfaces']}, LAM)  # no aperture


def test_v5_17_0_auto_default_byte_identical_at_threshold():
    """At N >= 4096 the DEFAULT (sag_chunk_rows=None -> auto row-band) must
    produce byte-identical results to the forced whole-grid path
    (sag_chunk_rows=0) -- the guarantee that justified flipping the
    default."""
    N, dx = 4096, 1.5e-6
    E = _field(N, dx)
    kw = dict(prescription=_presc(), wavelength=LAM, dx=dx)
    E_auto = apply_real_lens(E.copy(), **kw)                    # default: auto
    E_whole = apply_real_lens(E.copy(), sag_chunk_rows=0, **kw)  # forced whole
    assert np.array_equal(E_auto, E_whole)
