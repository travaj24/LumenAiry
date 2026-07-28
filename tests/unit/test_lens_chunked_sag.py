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
    """Pin FFT plan determinism across calls for byte-equality asserts.

    audit W9: restore the PRIOR value, not a hardcoded True.  Since
    v5.30.1 the shipped default is False, so a hardcoded True teardown
    would leak the non-reproducible planner into the rest of the worker.
    """
    prev = la.get_fft_auto_promote()
    la.set_fft_auto_promote(False)
    yield
    la.set_fft_auto_promote(prev)


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


# ---------------------------------------------------------------------------
# v5.17.1 audit wave-2 fixes (AUDIT_V5_17_0_2026_07_01_DEEP, lens-chunked
# cluster): P2-04 / P2-05 / P3-08 / P3-09 / P3-06 / P3-05.
# ---------------------------------------------------------------------------

def _zernike_presc():
    p = _presc()
    p['surfaces'][0] = dict(p['surfaces'][0], freeform_type='zernike',
                            zernike_coeffs={8: 2e-8}, norm_radius=1.5e-3)
    return p


def test_chunked_freeform_warns_and_falls_through():
    """P2-04: a non-Q freeform surface (zernike) must fall through to the
    whole-grid path on the chunked run so the 'freeform departure is NOT
    included' RuntimeWarning keeps firing (pre-fix the band loop silently
    dropped the departure with no diagnostic).  Outputs stay byte-identical
    (the departure was dropped on both paths)."""
    N, dx = 256, 8e-6
    E = _field(N, dx, w=0.6e-3)
    kw = dict(prescription=_zernike_presc(), wavelength=LAM, dx=dx)
    with pytest.warns(RuntimeWarning, match='freeform departure'):
        Ew = apply_real_lens(E.copy(), sag_chunk_rows=0, **kw)
    with pytest.warns(RuntimeWarning, match='freeform departure'):
        Ec = apply_real_lens(E.copy(), sag_chunk_rows=64, **kw)
    assert np.array_equal(Ew, Ec)


def test_traced_forwards_raw_sag_chunk_rows_to_amp_legs(monkeypatch):
    """P2-05: sag_chunk_rows=0 (documented force-whole-grid) must survive
    into the internal apply_real_lens amp legs.  Pre-fix the traced function
    forwarded the RESOLVED value (0 -> None), which the inner call
    re-resolved to AUTO banding at N >= 4096."""
    from lumenairy.elements import _lens_traced as lt
    from lumenairy.elements._lens_real import _resolve_sag_chunk_rows

    N, dx = 256, 8e-6
    E = _field(N, dx, w=0.6e-3)
    kw = dict(prescription=_presc(), wavelength=LAM, dx=dx,
              ray_subsample=4, parallel_amp=False)

    seen = []
    orig = lt.apply_real_lens

    def spy(*a, **k):
        seen.append(k['sag_chunk_rows'])
        return orig(*a, **k)

    monkeypatch.setattr(lt, 'apply_real_lens', spy)
    apply_real_lens_traced(E.copy(), sag_chunk_rows=0, **kw)
    assert seen, "amp legs never called"
    # The forwarded value must force whole-grid at ANY N when re-resolved
    # by the inner apply_real_lens (0 -> whole grid; the pre-fix None
    # re-resolved to a 1024-row band at N=16384).
    assert all(_resolve_sag_chunk_rows(v, 16384) is None for v in seen), seen
    # Explicit positive ints round-trip untouched.
    seen.clear()
    apply_real_lens_traced(E.copy(), sag_chunk_rows=64, **kw)
    assert all(v == 64 for v in seen), seen


def test_undersample_floor_enforced_without_aperture():
    """P3-08: the min_coarse_samples_per_aperture floor must also fire for
    prescriptions WITHOUT aperture_diameter (pre-fix it was silently
    skipped), using the largest per-surface clear_aperture when present,
    else the launch diameter (grid extent)."""
    N, dx = 256, 1.5e-5
    E = _field(N, dx, w=1.5e-3)
    kw = dict(wavelength=LAM, dx=dx, parallel_amp=False)
    p_noap = _presc()
    p_noap.pop('aperture_diameter')
    # Grid-extent fallback: N/sub = 16 coarse samples < 32 -> raises.
    with pytest.raises(ValueError, match='coarse samples'):
        apply_real_lens_traced(E.copy(), prescription=p_noap,
                               ray_subsample=16, **kw)
    # clear_aperture fallback: 1 mm pupil / (8 * 15 um) = 8.3 < 32 even
    # though the grid extent would pass (N/sub = 32).
    p_ca = _presc()
    p_ca.pop('aperture_diameter')
    p_ca['surfaces'][0] = dict(p_ca['surfaces'][0], clear_aperture=1.0e-3)
    with pytest.raises(ValueError, match='clear_aperture'):
        apply_real_lens_traced(E.copy(), prescription=p_ca,
                               ray_subsample=8, **kw)
    # A well-sampled apertureless run still completes (N/sub = 64 >= 32).
    out = apply_real_lens_traced(E.copy(), prescription=p_noap,
                                 ray_subsample=4, **kw)
    assert np.all(np.isfinite(out))


def test_amp_freed_before_assembly_on_preserve_path():
    """P3-09: on the sub>1 preserve_input_phase=True path the full-grid
    |E_analytic| array is dead after the coarse subsample; it must be freed
    (del'd) before the band assembly.  Byte-identity with the whole-grid
    path must survive the eager free."""
    import sys

    N, dx = 256, 8e-6
    E = _field(N, dx, w=0.6e-3)
    seen = {}

    def prog(stage, frac, msg=''):
        if 'assembling' in msg:
            f = sys._getframe(1)
            while f is not None and \
                    f.f_code.co_name != 'apply_real_lens_traced':
                f = f.f_back
            assert f is not None, "traced frame not found"
            seen['amp_alive'] = 'amp' in f.f_locals

    kw = dict(prescription=_presc(), wavelength=LAM, dx=dx,
              ray_subsample=4, parallel_amp=False, progress=prog)
    T_chunk = apply_real_lens_traced(E.copy(), sag_chunk_rows=64, **kw)
    assert seen.get('amp_alive') is False, (
        "full-grid amp still resident at band assembly")
    T_whole = apply_real_lens_traced(E.copy(), sag_chunk_rows=0, **kw)
    assert np.array_equal(T_chunk, T_whole)


def test_public_docstrings_document_memory_knobs():
    """P3-06 + P3-05: the v5.17.0 knobs must be discoverable via help() on
    both NumPy propagators, and the JAX twins must document the ABSENCE of
    the row-band mode."""
    for fn in (apply_real_lens, apply_real_lens_traced):
        doc = fn.__doc__ or ''
        assert 'sag_dtype' in doc, fn.__name__
        assert 'sag_chunk_rows' in doc, fn.__name__
        assert 'lens_sag_float32_opd_error' in doc, fn.__name__
    from lumenairy.elements import _lens_jax
    for fn in (_lens_jax.apply_real_lens_traced_jax,
               _lens_jax.apply_real_lens_maslov_jax):
        assert 'sag_chunk_rows' in (fn.__doc__ or ''), fn.__name__
        assert 'monolithic' in (fn.__doc__ or ''), fn.__name__


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
