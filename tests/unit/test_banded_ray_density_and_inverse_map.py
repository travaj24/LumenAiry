"""Row-banded assembly for ``amplitude_model='ray_density'`` and for the
inverse-characteristic route (AUDIT_TRACED_MEMORY_2026_08_09 row 3, closed).

Before this change ``sag_chunk_rows`` (AUTO at ``N >= 4096``) had two named
exclusions in ``apply_real_lens_traced``:

* ``amplitude_model='ray_density'`` forced the band path OFF, because the
  magnitude swap ran on the whole-grid exit field;
* the band path refused the inverse-characteristic evaluator
  (``_imap_domain_gate`` carried ``not _chunk_assembly``), so a banded call
  silently selected the incumbent coarse-Newton inversion instead -- a
  DIFFERENT answer (measured 2.19e-02 relative on the S10 carrier fixture).

Both are lifted.  The magnitude swap, the residual multiply, the ray-density
upsample and its NaN pass are pointwise in the exit pixel and now run per
band; the inverse map is evaluated per band, with the one non-pointwise piece
-- the caustic census (median / min / max / adjacent-pixel sign scan of
``|det J|``) -- taken in a first pass over ``|det J|`` alone, exactly as the
whole-grid closure takes it.

The pins here are BYTE-IDENTITY (``np.array_equal``) of the banded field
against the whole-grid field AT THE SAME INVERSION, plus equality of the
diagnostics the two paths report (``n_out_of_domain``, the fold-caustic
warning).  These are decisions the band decomposition is entitled to move by
exactly nothing, so the bar is exact -- the same bar
``test_lens_chunked_sag.py`` has always used for the screen path.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_imap as IM

_WL = 1.31e-6
_N, _DX, _W = 384, 16e-6, 1.4e-3
_SUB = 4
_RC = -0.06                      # carrier conjugate (converging), metres


@pytest.fixture(autouse=True)
def _deterministic_fft():
    prev = la.get_fft_auto_promote()
    la.set_fft_auto_promote(False)
    yield
    la.set_fft_auto_promote(prev)


def _surf(radius, gb, ga):
    return {'radius': radius, 'glass_before': gb, 'glass_after': ga,
            'conic': 0.0, 'radius_y': None, 'conic_y': None,
            'aspheric_coeffs': None, 'aspheric_coeffs_y': None}


def _presc():
    return {'name': 'band_rd_singlet', 'aperture_diameter': 9e-3,
            'surfaces': [_surf(0.030, 'air', 'N-BK7'),
                         _surf(-0.030, 'N-BK7', 'air')],
            'thicknesses': [3e-3]}


def _field(n=_N, dx=_DX, w=_W):
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    return np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)


def _base_kw(**over):
    kw = dict(prescription=_presc(), wavelength=_WL, dx=_DX,
              ray_subsample=_SUB, n_workers=1, on_undersample='silent',
              on_noncollimated='off', on_aperture_beam='silent',
              parallel_amp=False, carrier=_RC)
    kw.update(over)
    return kw


_MODES = [
    dict(amplitude_model='ray_density', preserve_input_phase=True),
    dict(amplitude_model='ray_density', preserve_input_phase='remap',
         remap_sampling='lattice'),
    dict(amplitude_model='ray_density', preserve_input_phase='remap',
         remap_sampling='full'),
]


def _run(rows, **kw):
    """One call with the inverse-map cache cold, returning
    ``(field, guard record, warning messages)``."""
    IM.inverse_map_cache_clear()
    rec = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = la.apply_real_lens_traced(_field(), sag_chunk_rows=rows,
                                        _imap_out=rec, **kw)
    IM.inverse_map_cache_clear()
    return np.asarray(out), rec, sorted(str(w.message)[:80] for w in caught)


# ===========================================================================
# 1.  ray-density on the coarse-Newton path (evaluator off)
# ===========================================================================
@pytest.mark.parametrize('mode', _MODES, ids=['rd', 'rd+remap.lattice',
                                              'rd+remap.full'])
def test_ray_density_band_path_is_byte_identical_on_coarse_newton(mode):
    kw = _base_kw(inverse_map=False, **mode)
    whole, _, w_whole = _run(0, **kw)
    assert np.isfinite(whole).all() and np.abs(whole).max() > 0
    for rows in (32, 128, 7):                     # divisor and non-divisor
        band, _, w_band = _run(rows, **kw)
        assert np.array_equal(whole, band), rows
        # the two paths must reach the same DECISIONS as well: the fold
        # census and the three ray-density self-checks warn identically.
        assert w_band == w_whole, (rows, w_band, w_whole)


# ===========================================================================
# 2.  the inverse-characteristic evaluator ENGAGES on the band path
# ===========================================================================
@pytest.mark.parametrize('mode', [dict(amplitude_model='screen')] + _MODES,
                         ids=['screen', 'rd', 'rd+remap.lattice',
                              'rd+remap.full'])
def test_inverse_map_engages_on_band_path_and_is_byte_identical(mode):
    """At the shipped default (evaluator on) the banded and whole-grid calls
    are now the SAME inversion: both open the gate, both engage, and the
    fields are bit-equal.  Pre-change the banded arm reported
    ``gate_open=False`` and returned the incumbent's different answer."""
    kw = _base_kw(**mode)
    whole, rec_w, w_whole = _run(0, **kw)
    assert rec_w['gate_open'] and rec_w['engaged'], rec_w
    for rows in (32, 128, 7):
        band, rec_b, w_band = _run(rows, **kw)
        assert rec_b['gate_open'] and rec_b['engaged'], (rows, rec_b)
        assert rec_b['n_out_of_domain'] == rec_w['n_out_of_domain'], rows
        assert np.array_equal(whole, band), rows
        assert w_band == w_whole, (rows, w_band, w_whole)


def test_band_path_result_differs_from_the_incumbent_it_used_to_select():
    """The reason the change matters: with the evaluator on, the banded
    field is the model's answer, NOT the coarse-Newton incumbent's.  Measured
    2.19e-02 relative on the S10 fixture; here the bar is only "not equal",
    which no decomposition can produce by accident (the two inversions
    differ at every interior pixel)."""
    kw = _base_kw(amplitude_model='ray_density', preserve_input_phase='remap',
                  remap_sampling='full')
    model, rec_m, _ = _run(64, **kw)
    incumbent, rec_i, _ = _run(64, inverse_map=False, **kw)
    assert rec_m['engaged'] and not rec_i['engaged']
    assert not np.array_equal(model, incumbent)


# ===========================================================================
# 2b.  the self-check warnings are attributed to the CALLER (v5.44.1, D1)
# ===========================================================================
def _run_recording_filenames(rows, **kw):
    """Like ``_run`` but keeps each warning's ``filename`` -- the field
    ``warnings.filterwarnings(module=...)`` and the default filter's
    per-location dedup registry key on."""
    IM.inverse_map_cache_clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = la.apply_real_lens_traced(_field(), sag_chunk_rows=rows, **kw)
    IM.inverse_map_cache_clear()
    return np.asarray(out), [(os.path.basename(str(w.filename)),
                              str(w.message)) for w in caught]


@pytest.mark.parametrize('rows', [0, 32, 7], ids=['whole', 'band32', 'band7'])
def test_ray_density_self_check_warnings_name_the_caller(monkeypatch, rows):
    """The three self-checks live in a nested closure shared by the banded and
    the whole-grid path, so their ``stacklevel`` has to be 3, not the 2 they
    were written with when they sat in the function body (v5.44.1,
    VERIFY_LENS_BANDED_COMPLEX64_2026_09_10 D1).

    MEASURED before the fix, on this box: 2 of the 3 notices the probe fires
    were attributed to ``_lens_traced.py:12236`` on the banded path and
    ``:12350`` on the whole-grid one; v5.43.0 attributed all three to the
    caller.  That file/line is what ``warnings.filterwarnings(module=...)``
    selects on and what the default filter's per-location dedup registry is
    keyed by, so a library attribution also stops a second call from a
    different caller module re-warning.

    Every threshold is driven over so the energy notice FIRES; the assertion
    is that THIS file is what it names, on both routes, and that no notice
    ``apply_real_lens_traced`` itself raises names the library.  (The
    ``apply_real_lens:`` notices of the analytic sub-call are a different
    function with a genuine library caller and are excluded by prefix.  The
    halo and support-band notices need a fixture this one is not -- they are
    measured on the caller in ``validation/probe_fix_lens_5440/p2_d1_attr.py``,
    which fires the support-band check on both routes.)  Byte-identity of the
    field is asserted alongside: a stacklevel must not be able to move a
    value."""
    import lumenairy.elements._lens_traced as LT
    monkeypatch.setattr(LT, '_RD_ENERGY_GAIN_TOL', -1.0)
    monkeypatch.setattr(LT, '_RD_ENERGY_DEFICIT_BASE', -1.0)
    monkeypatch.setattr(LT, '_RD_ENERGY_DEFICIT_PER_SUB', 0.0)
    monkeypatch.setattr(LT, '_RD_HALO_AMAX_TOL', 0.0)
    monkeypatch.setattr(LT, '_SUPPORT_BAND_PEAK_RATIO_TOL', 0.0)
    kw = _base_kw(amplitude_model='ray_density', preserve_input_phase=True)
    whole, w_whole = _run_recording_filenames(0, **kw)
    got, ws = _run_recording_filenames(rows, **kw)
    assert np.array_equal(whole, got), rows
    here = os.path.basename(__file__)
    for tag, got_ws in (('band', ws), ('whole', w_whole)):
        fired = [(f, m) for f, m in got_ws
                 if "'ray_density' energy self-check FAILED" in m]
        assert len(fired) == 1, (tag, got_ws)      # or this pins nothing
        assert fired[0][0] == here, (tag, fired)
        mine = [(f, m) for f, m in got_ws
                if m.startswith('apply_real_lens_traced:')]
        assert [f for f, _ in mine if f != here] == [], (tag, mine)


# ===========================================================================
# 3.  the memory lever is real on the routes that used to refuse it
# ===========================================================================
def test_band_path_lowers_the_peak_on_the_ray_density_inverse_map_route():
    """The whole-call tracemalloc peak drops by at least 6 full-grid float64
    equivalents when the ray-density + inverse-map route bands.

    THE STATE IS ENGINEERED, not hoped for -- three things can hide the
    assembly behind another stage at test-sized N, and each is excluded:

    * the inverse-map FIT's design matrix (``n_launch^2 x P``) scales with
      the launch lattice ``aperture / (sub * dx)``, not with N; a fine
      lattice makes it the peak on both arms and the difference reads ~0
      (measured 2026-09-09: N=384 / sub=4 gave 60.8 vs 60.8 MB).  Hence the
      COARSE lattice here (sub=16, dx=12 um -> n_launch ~ 70).
    * ``preserve_input_phase='remap'``'s residual builder de-chirps by the
      niche-C6 eikonal in bands of ``4194304 // N`` rows -- the WHOLE grid
      below N=2048 -- and ``_ResidualEikonal.value`` on a full 512^2 grid
      is a 23-grid transient (measured, both arms); it is the peak of every
      remap arm at this N and it is not this change's.  Hence NO remap here
      (the remap arms' byte-identity is pinned above).
    * cold caches: the arm that runs first pays for the FFT plans / H cache
      / glass tables.  Hence one identical warm-up call per arm.

    Bar derivation (2026-09-10, numpy 2.4 / scipy 1.17, this fixture, warm):
    whole-grid peak 22.6 grids, banded 10.2 grids (32 rows) -- a saving of
    12.4 grids, all of it the whole-grid arm's four full-grid model channels
    (4), its coordinate stack (2), and the swap / mask transients (~6) that
    the band loop never materialises.  The bar of 6 grids is 2x below the
    measurement and above anything a build can move: tracemalloc counts
    REQUESTED sizes, fixed by shapes and dtypes, and the resident grids
    common to both arms cancel in the difference."""
    import tracemalloc

    N, dx, sub = 512, 12e-6, 16

    def _peak(rows):
        E = _field(N, dx)
        kw = _base_kw(dx=dx, ray_subsample=sub, amplitude_model='ray_density',
                      preserve_input_phase=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            IM.inverse_map_cache_clear()
            la.apply_real_lens_traced(E, sag_chunk_rows=rows, **kw)   # warm
            IM.inverse_map_cache_clear()
            rec = {}
            tracemalloc.start()
            tracemalloc.reset_peak()
            la.apply_real_lens_traced(E, sag_chunk_rows=rows, _imap_out=rec,
                                      **kw)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            IM.inverse_map_cache_clear()
        assert rec['engaged'], rec          # the route under test IS the route
        return peak

    grid = 8 * N * N
    p_whole = _peak(0)
    p_band = _peak(32)
    assert p_whole - p_band >= 6 * grid, (
        p_whole / grid, p_band / grid, (p_whole - p_band) / grid)
