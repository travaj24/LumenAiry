"""Pins for AUDIT_ADVERSARIAL_CODEBASE_2026_07_25 Territory A (MEDIUM/LOW).

Covered findings, each measured before the fix and re-measured after:

* **A-4** ``io/storage.py`` -- ``append_plane_h5(metadata=)`` handed raw
  values to h5py attrs.  Measured over the module's own 19-type probe set:
  4 hard raises (nested dict, heterogeneous list, arbitrary object), 1
  silent DROP (``None``), and 4 silent type coercions -- while the module's
  own ``_meta_dumps`` codec round-trips 19/19 and was wired only to
  ``write_sim_metadata``.  Post-fix 19/19, and files written by the old
  raw-attr scheme still read.
* **A-5** ``cache.deep_nbytes`` -- charged a numpy VIEW its slice size
  (16 B for a 16-byte window on a 4 MiB base) while double-counting
  repeated arrays.  Measured: a view-heavy cache under a 1 MiB ceiling
  accounted 256 B and retained 67108864 B (64x over cap, zero evictions).
* **A-6** ``memory.estimate_asm_memory`` -- est/measured first-call peak
  0.53 (N=512) / 0.96 (1024) / 1.22 (2048): neither a bound nor a
  steady-state figure.  Re-derived from fresh-interpreter tracemalloc
  profiles; now 1.02-1.09 (always a bound) over the eight points
  N=256/512/1024/2048 x {complex64, complex128}.
* **A-7** ``analysis/psf_mtf_otf`` -- OTF/MTF docstrings claimed
  ``otf[0, 0]`` was DC=1; the output is fftshifted so DC is
  ``[N//2, N//2]`` and ``otf[0, 0]`` measures -1.43e-17.  Doc-only fix,
  behaviour deliberately unchanged (pinned both ways).
* **A-8** ``user_library.load_all_materials`` -- corrupted saved materials
  skipped by ``except ...: pass`` twice with no warning; user glass
  vanished silently.
* **A-9..A-14 (LOW)** dead ``backend/fft._jnp_or_none``; undocumented
  ping-pong buffer ownership on public ``backend.fft.fft2/ifft2``;
  ``zernike_basis_matrix`` returning the mutable cached array; ``io/``
  modules with no ``__all__``; ``memory`` accepting negative costs.

Every assertion here was verified to FAIL against the pre-fix tree
(git worktree of 7ea2eb9).
"""
from __future__ import annotations

import subprocess
import sys
import warnings

import numpy as np
import pytest

import lumenairy as la

# =========================================================================
# A-4 -- append_plane_h5 metadata fidelity
# =========================================================================

def _probe_metadata():
    """The 19 metadata value kinds the module's own codec round-trips."""
    return {
        'v_none': None,
        'v_bool': True,
        'v_int': 7,
        'v_float': 2.5,
        'v_str': 'hello',
        'v_complex': 1 + 2j,
        'v_bytes': b'\x01\x02',
        'v_np_scalar_f': np.float32(1.5),
        'v_np_scalar_c': np.complex128(3 - 4j),
        'v_ndarray_f': np.arange(4, dtype=np.float64),
        'v_ndarray_c': np.arange(3, dtype=np.complex128) * 1j,
        'v_ndarray_2d': np.eye(2),
        'v_list_int': [1, 2, 3],
        'v_list_mixed': [1, 'a', 2.5],
        'v_list_empty': [],
        'v_tuple': (1, 2),
        'v_dict_flat': {'a': 1},
        'v_dict_nested': {'a': {'b': [1, 2]}},
        'v_obj': object(),
    }


class TestA4AppendPlaneMetadataFidelity:

    def test_all_19_probe_kinds_round_trip(self, tmp_path):
        """Pre-fix: 4 raises + 1 silent drop + 4 type coercions."""
        pytest.importorskip('h5py')
        from lumenairy.io.storage import append_plane_h5, load_planes_h5
        p = str(tmp_path / 'planes.h5')
        meta = _probe_metadata()
        # Pre-fix this call itself raised TypeError on the nested dict.
        append_plane_h5(p, np.ones((8, 8), dtype=np.complex128), dx=1e-6,
                        metadata=meta, swmr=False)
        got = load_planes_h5(p)[0][0]

        missing = [k for k in meta if k not in got]
        assert not missing, (
            f"append_plane_h5 dropped metadata keys {missing} -- A-4 "
            f"regression (the ``None`` value was the pre-fix casualty).")

        # Exact type + value fidelity for everything the codec claims.
        assert got['v_none'] is None
        assert got['v_bool'] is True
        assert got['v_int'] == 7 and isinstance(got['v_int'], int)
        assert got['v_float'] == 2.5
        assert got['v_str'] == 'hello'
        assert got['v_complex'] == 1 + 2j
        assert got['v_bytes'] == b'\x01\x02'
        assert got['v_np_scalar_f'] == 1.5
        assert got['v_np_scalar_c'] == 3 - 4j
        np.testing.assert_array_equal(got['v_ndarray_f'], meta['v_ndarray_f'])
        np.testing.assert_array_equal(got['v_ndarray_c'], meta['v_ndarray_c'])
        np.testing.assert_array_equal(got['v_ndarray_2d'], meta['v_ndarray_2d'])
        # list stays a list (NOT coerced to ndarray as the raw-attr path did)
        assert got['v_list_int'] == [1, 2, 3]
        assert isinstance(got['v_list_int'], list)
        assert got['v_list_mixed'] == [1, 'a', 2.5]
        assert got['v_list_empty'] == []
        assert got['v_tuple'] == (1, 2) and isinstance(got['v_tuple'], tuple)
        assert got['v_dict_flat'] == {'a': 1}
        assert got['v_dict_nested'] == {'a': {'b': [1, 2]}}
        assert isinstance(got['v_obj'], str)   # repr fallback, as documented

        # The native per-plane attributes still work untouched.
        assert got['dx'] == pytest.approx(1e-6)
        assert got['dy'] == pytest.approx(1e-6)
        assert got['field'].shape == (8, 8)

    def test_reserved_blob_key_never_surfaces(self, tmp_path):
        """The serialization carrier must not leak into user-visible keys,
        and the lossy flattened shadow copies must not double-report."""
        pytest.importorskip('h5py')
        from lumenairy.io import storage as st
        p = str(tmp_path / 'blob.h5')
        st.append_plane_h5(p, np.ones((4, 4), dtype=np.complex128), dx=1e-6,
                           metadata={'run': {'id': 3, 'tag': 'x'}},
                           swmr=False)
        got = st.load_planes_h5(p)[0][0]
        assert st._META_BLOB_KEY not in got
        assert got['run'] == {'id': 3, 'tag': 'x'}
        # The dotted flat shadow keys the writer leaves for HDFView are
        # suppressed on read in favour of the nested structure.
        assert 'run.id' not in got and 'run.tag' not in got
        # ... but they ARE on disk for external inspectors.
        import h5py
        with h5py.File(p, 'r') as f:
            raw = set(f['planes/plane_00'].attrs.keys())
        assert 'run.id' in raw and st._META_BLOB_KEY in raw

    def test_every_h5_plane_reader_agrees(self, tmp_path):
        """All four plane read paths must decode the blob identically."""
        pytest.importorskip('h5py')
        from lumenairy.io import storage as st
        p = str(tmp_path / 'readers.h5')
        meta = {'nested': {'a': [1, 2]}, 'nothing': None, 'tup': (1, 2)}
        st.append_plane_h5(p, np.ones((4, 4), dtype=np.complex128), dx=1e-6,
                           label='L0', metadata=meta, swmr=False)
        by_load = st.load_planes_h5(p)[0][0]
        by_list = st._h5_list_planes(p)[0][0]
        by_label = st._h5_load_plane_by_label(p, 'L0')
        by_slice = st._h5_load_plane_slice(p, 0, slice(0, 2), slice(0, 2))[1]
        for name, d in (('load_planes_h5', by_load),
                        ('_h5_list_planes', by_list),
                        ('_h5_load_plane_by_label', by_label),
                        ('_h5_load_plane_slice', by_slice)):
            assert d['nested'] == {'a': [1, 2]}, name
            assert d['nothing'] is None, name
            assert d['tup'] == (1, 2), name
            assert st._META_BLOB_KEY not in d, name

    def test_legacy_raw_attr_files_still_read(self, tmp_path):
        """BACK-COMPAT: a plane whose metadata was written by the pre-A-4
        raw-attr scheme (no blob) must read exactly as it always did."""
        pytest.importorskip('h5py')
        import h5py

        from lumenairy.io import storage as st
        p = str(tmp_path / 'legacy.h5')
        st.append_plane_h5(p, np.ones((4, 4), dtype=np.complex128), dx=1e-6,
                           label='leg', swmr=False)
        # Emulate the pre-fix writer: raw values straight onto attrs.
        with h5py.File(p, 'a') as f:
            dset = f['planes/plane_00']
            dset.attrs['legacy_scalar'] = 3.5
            dset.attrs['legacy_str'] = 'raw'
            dset.attrs['legacy_arr'] = np.arange(3)
            assert st._META_BLOB_KEY not in dset.attrs
        got = st.load_planes_h5(p)[0][0]
        assert got['legacy_scalar'] == pytest.approx(3.5)
        assert got['legacy_str'] == 'raw'
        np.testing.assert_array_equal(got['legacy_arr'], np.arange(3))
        assert got['label'] == 'leg'

    def test_no_metadata_writes_no_blob(self, tmp_path):
        """A metadata-free append must not grow a blob attribute (keeps
        old readers and byte-level file diffs clean)."""
        pytest.importorskip('h5py')
        import h5py

        from lumenairy.io import storage as st
        p = str(tmp_path / 'plain.h5')
        st.append_plane_h5(p, np.ones((4, 4), dtype=np.complex128), dx=1e-6,
                           swmr=False)
        with h5py.File(p, 'r') as f:
            assert st._META_BLOB_KEY not in f['planes/plane_00'].attrs

    def test_flat_shadow_suppression_edge_cases(self, tmp_path):
        """The read path suppresses the writer's dotted flat copies; make
        sure that cannot eat a legitimate user key."""
        pytest.importorskip('h5py')
        from lumenairy.io import storage as st

        def rt(meta, name):
            p = str(tmp_path / name)
            st.append_plane_h5(p, np.ones((4, 4), dtype=np.complex128),
                               dx=1e-6, metadata=meta, swmr=False)
            return st.load_planes_h5(p)[0][0]

        # A nested key and a literal dotted key that flatten to the same
        # dotted name must both survive, distinctly.
        got = rt({'a': {'b': 1}, 'a.b': 2}, 'dotted.h5')
        assert got['a'] == {'b': 1}
        assert got['a.b'] == 2
        # A user key equal to the codec's own type tag must not be
        # mis-read as a tagged wrapper.
        got = rt({st._META_TAG: 'user'}, 'tagkey.h5')
        assert got[st._META_TAG] == 'user'
        # Arbitrary depth, with None inside a list inside a dict.
        deep = {'x': {'y': {'z': [1, {'w': None}]}}}
        assert rt({'deep': deep}, 'deep.h5')['deep'] == deep

    def test_write_sim_metadata_contract_unchanged(self, tmp_path):
        """A-4 must not disturb the S4-19 root-metadata contract."""
        pytest.importorskip('h5py')
        from lumenairy.io.storage import read_sim_metadata, write_sim_metadata
        p = str(tmp_path / 'sim.h5')
        meta = _probe_metadata()
        write_sim_metadata(p, meta)
        back = read_sim_metadata(p)
        assert set(back) == set(meta)
        assert back['v_none'] is None
        assert back['v_dict_nested'] == {'a': {'b': [1, 2]}}
        assert isinstance(back['v_tuple'], tuple)


# =========================================================================
# A-5 -- deep_nbytes base-buffer accounting
# =========================================================================

def _true_retained(cache):
    """Unique base-buffer bytes the cache genuinely keeps alive."""
    owners = {}
    for entry in cache._store.values():
        root = entry.value
        while getattr(root, 'base', None) is not None:
            root = root.base
        owners[id(root)] = int(root.nbytes)
    return sum(owners.values())


class TestA5DeepNbytesViewAccounting:

    def test_view_charged_its_base_buffer(self):
        """Pre-fix: 16 B for a window on a 4 MiB base."""
        from lumenairy.cache import deep_nbytes
        base = np.zeros((512, 512), dtype=np.complex128)
        assert base.nbytes == 4 * 1024 * 1024
        window = base[0:1, 0:1]
        assert window.nbytes == 16                     # what pre-fix charged
        assert deep_nbytes(window) == base.nbytes, (
            "A-5: a view must be charged the buffer it keeps alive, not "
            "its own slice size.")
        strided = base[::2, ::2]
        assert deep_nbytes(strided) == base.nbytes

    def test_owning_array_unchanged(self):
        """The common case must be byte-identical to the old behaviour."""
        from lumenairy.cache import deep_nbytes
        a = np.zeros((64, 64), dtype=np.complex128)
        assert a.base is None
        assert deep_nbytes(a) == a.nbytes
        assert deep_nbytes(None) == 0

    def test_repeated_array_counted_once(self):
        """Pre-fix: the ``nbytes`` shortcut returned before the ``_seen``
        check, so ``(a, a)`` charged the same buffer twice."""
        from lumenairy.cache import deep_nbytes
        a = np.zeros(1000, dtype=np.float64)
        assert a.nbytes == 8000
        one = deep_nbytes((a,))
        two = deep_nbytes((a, a))
        three = deep_nbytes((a, a, a))
        # Only the tuple's own getsizeof grows; the buffer is charged once.
        assert two - one < 100, (
            f"A-5: repeated array double-counted ({one} -> {two}).")
        assert three - one < 200
        assert two >= a.nbytes

    def test_two_views_of_one_base_counted_once(self):
        from lumenairy.cache import deep_nbytes
        base = np.zeros((256, 256), dtype=np.complex128)
        pair = (base[0:2], base[4:6])
        assert deep_nbytes(pair) < 1.5 * base.nbytes
        assert deep_nbytes(pair) >= base.nbytes

    def test_same_size_views_do_not_inflate(self):
        """Guard the ONE live library consumer.  The only in-library
        ``ByteBudgetedLRU`` instance (``_lens_real._DISPLACED_COS_GRID_CACHE``)
        stores ``rgi(pq).reshape(Xg.shape)`` results -- reshape / transpose
        views whose base is the same-size flat array.  Base-buffer
        accounting must leave those unchanged, or an opt-in cache would
        silently start declining its own entries."""
        from lumenairy.cache import deep_nbytes
        flat = np.arange(4096, dtype=np.float64)
        reshaped = flat.reshape(64, 64)
        assert reshaped.base is flat
        assert deep_nbytes(reshaped) == flat.nbytes
        assert deep_nbytes(reshaped.T) == flat.nbytes
        # np.frombuffer bases are memoryviews, also same-size.
        fb = np.frombuffer(bytearray(800), dtype=np.float64)
        assert deep_nbytes(fb) == fb.nbytes

    def test_live_library_cache_still_accepts_its_entries(self):
        """End-to-end: enable the opt-in cos-grid cache with a realistic
        budget and check a cos-grid pair is still stored + retrievable."""
        from lumenairy.elements import _lens_real as lr
        old = lr.get_pointwise_cos_grid_cache_budget()
        try:
            lr.set_pointwise_cos_grid_cache_budget(64)   # 64 MB
            grids = [(np.zeros(4096).reshape(64, 64),
                      np.zeros(4096).reshape(64, 64))]
            stored = lr._DISPLACED_COS_GRID_CACHE.put(('probe',), grids)
            assert stored is True
            assert lr._DISPLACED_COS_GRID_CACHE.get(('probe',)) is grids
            rep = lr._DISPLACED_COS_GRID_CACHE.report()
            assert rep['retained_bytes'] == pytest.approx(2 * 4096 * 8,
                                                          rel=0.05)
        finally:
            lr._DISPLACED_COS_GRID_CACHE.clear()
            lr.set_pointwise_cos_grid_cache_budget(
                0 if old in (None, 0) else old / (1024 * 1024))

    def test_view_heavy_cache_respects_its_ceiling(self):
        """The finding that matters: retention vs the byte budget.

        Pre-fix, 16 views of 256 KiB bases under a 1 MiB ceiling accounted
        256 B, evicted nothing, and genuinely retained 4 MiB.
        """
        from lumenairy.cache import ByteBudgetedLRU
        cap = 1024 * 1024
        c = ByteBudgetedLRU('a5_pin', max_bytes=cap, register=False)
        for i in range(16):
            b = np.zeros((256, 128), dtype=np.complex128)   # 256 KiB
            c.put((i,), b[0:1, 0:1])
            del b
        rep = c.report()
        true_bytes = _true_retained(c)
        assert true_bytes <= cap, (
            f"A-5: cache retains {true_bytes} B against a {cap} B "
            f"ceiling ({true_bytes / cap:.1f}x over).")
        assert rep['retained_bytes'] == true_bytes, (
            "A-5: accounted bytes must equal the bytes actually retained.")
        assert rep['evictions'] > 0, (
            "A-5: eviction never fired -- the budget is not being enforced.")

    def test_oversized_view_is_skipped_not_silently_retained(self):
        """A view whose BASE exceeds the hard ceiling can never be
        retained under budget, so ``put`` must decline it."""
        from lumenairy.cache import ByteBudgetedLRU
        cap = 1024 * 1024
        c = ByteBudgetedLRU('a5_pin2', max_bytes=cap, register=False)
        base = np.zeros((512, 512), dtype=np.complex128)    # 4 MiB
        stored = c.put(('k',), base[0:1, 0:1])
        assert stored is False
        assert len(c) == 0
        assert _true_retained(c) == 0


# =========================================================================
# A-6 -- estimate_asm_memory re-derivation
# =========================================================================

_A6_CHILD = r"""
import sys, tracemalloc
import numpy as np
N = int(sys.argv[1]); dt = sys.argv[2]
import lumenairy as la
from lumenairy.propagators.asm import angular_spectrum_propagate
E = np.ones((N, N), dtype=np.dtype(dt))
tracemalloc.start()
tracemalloc.reset_peak()
b = tracemalloc.get_traced_memory()[0]
out = angular_spectrum_propagate(E, z=0.05, wavelength=1.31e-6, dx=2e-6)
cold = tracemalloc.get_traced_memory()[1] - b
del out
peaks = []
for _ in range(3):
    tracemalloc.reset_peak()
    b = tracemalloc.get_traced_memory()[0]
    out = angular_spectrum_propagate(E, z=0.05, wavelength=1.31e-6, dx=2e-6)
    peaks.append(tracemalloc.get_traced_memory()[1] - b)
    del out
tracemalloc.stop()
print(cold, max(peaks), la.estimate_asm_memory(N, dt))
"""


def _measure_asm_peak(n_grid, dtype):
    """Fresh-interpreter (cold, first-call) and steady-state ASM peaks."""
    proc = subprocess.run(
        [sys.executable, '-c', _A6_CHILD, str(n_grid), dtype],
        capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    cold, steady, est = (int(v) for v in proc.stdout.split())
    return cold, steady, est


class TestA6EstimateAsmMemory:

    def test_formula_carries_a_fixed_first_call_term(self):
        """Pre-fix the estimate was purely ``k * N^2 * itemsize``, so it
        collapsed to 128 B at N=1 and under-read the first-call peak by
        ~2x at N=512.  The measured one-time lazy FFT-backend import cost
        is 38.17-38.50 MB, constant over N=64..2048 and both dtypes."""
        from lumenairy import memory as m
        est_tiny = m.estimate_asm_memory(1, 'complex128')
        assert est_tiny >= 32 * 1024 * 1024, (
            f"A-6: no N-independent first-call term (est(N=1) = "
            f"{est_tiny} B); the estimate cannot bound a fresh-process "
            f"first call.")
        assert est_tiny == pytest.approx(m._ASM_FIRST_CALL_FIXED_BYTES,
                                         rel=1e-3)

    def test_shape_term_matches_the_measured_allocation_profile(self):
        """The per-pixel slope must be the re-derived
        ``2 complex + 0.7 float64 + 2*plan_cache_keys`` profile
        (101.6 B/px at complex128, keys=2), not the old flat 8x
        (128 B/px)."""
        from lumenairy import memory as m
        n1, n2 = 1024, 2048
        d1 = m.estimate_asm_memory(n2) - m.estimate_asm_memory(n1)
        slope = d1 / (n2 * n2 - n1 * n1)
        expected = (m._ASM_COMPLEX_ARRAYS * 16
                    + m._ASM_F64_GRID_ARRAYS * 8
                    + 2 * 2 * 16)
        assert slope == pytest.approx(expected, rel=1e-6)
        assert slope == pytest.approx(101.6, rel=1e-3)
        # complex64 keeps the dtype-independent float64 grid term.
        d64 = (m.estimate_asm_memory(n2, 'complex64')
               - m.estimate_asm_memory(n1, 'complex64'))
        slope64 = d64 / (n2 * n2 - n1 * n1)
        assert slope64 == pytest.approx(8 * 2 + 0.7 * 8 + 2 * 2 * 8,
                                        rel=1e-6)
        assert slope64 > 8 * 6, (
            "A-6: the float64 frequency-grid term must keep complex64 "
            "bounded (measured 6.63x itemsize, vs 6.00x at complex128).")

    def test_documented_band_vs_steady_state(self):
        """The docstring states the estimate runs ~6.4x the steady-state
        per-call transient asymptotically, and more at small N where the
        fixed import term dominates (16x at N=512 complex128).  Pin both
        so the docstring and the formula cannot drift apart."""
        from lumenairy import memory as m
        ratios = {n: m.estimate_asm_memory(n) / (n * n * 16)
                  for n in (512, 1024, 2048, 16384)}
        assert ratios[512] == pytest.approx(16.35, rel=0.02)
        assert ratios[16384] == pytest.approx(6.35, rel=0.01)
        # Monotonically approaching the asymptote from above.
        assert (ratios[512] > ratios[1024] > ratios[2048]
                > ratios[16384] > 101.6 / 16 - 1e-6)

    @pytest.mark.parametrize('n_grid,dtype', [(512, 'complex128'),
                                              (1024, 'complex128')])
    def test_est_bounds_measured_first_call_peak(self, n_grid, dtype):
        """MEASURED pin (fresh interpreter + tracemalloc, the audit's own
        method).  Pre-fix est/measured was 0.53 at N=512 and 0.96 at
        N=1024 -- not a bound.  Post-fix: 1.02-1.09 over the eight points
        N=256/512/1024/2048 x {complex64, complex128} on the reference
        box."""
        cold, steady, est = _measure_asm_peak(n_grid, dtype)
        ratio = est / cold
        assert ratio >= 1.0, (
            f"A-6: estimate_asm_memory({n_grid}, {dtype!r}) = {est} B does "
            f"NOT bound the measured fresh-interpreter first-call peak "
            f"{cold} B (est/measured = {ratio:.3f}).")
        # Tightness ceiling is PLATFORM-SCOPED: the 40 MiB first-call
        # fixed term was calibrated on Windows (this box); on CI Linux
        # the allocator retains a much smaller cold peak, so the same
        # estimate reads est/measured = 1.46 (N=1024) to 2.63 (N=512)
        # there (measured, CI run on e1fd64a) -- still a BOUND, the
        # fail-safe direction.  The >= 1.0 bound assertion above is the
        # A-6 contract and runs everywhere (pre-fix 0.53 fails it);
        # cross-platform we only fence absurd looseness.
        import importlib.util
        import sys
        if importlib.util.find_spec('pyfftw') is not None:
            if sys.platform == 'win32':
                assert ratio <= 1.35, (
                    f"A-6: estimate has drifted loose (est/measured = "
                    f"{ratio:.3f}); the documented band on the Windows "
                    f"calibration platform is 1.02-1.09.")
            else:
                assert ratio <= 4.0, (
                    f"A-6: estimate absurdly loose (est/measured = "
                    f"{ratio:.3f}); CI-Linux measured 1.46-2.63.")
        # Steady state is ~1 output field; the estimate is not that.
        assert steady == pytest.approx(n_grid * n_grid * 16, rel=0.05)

    def test_junk_sizing_inputs_raise(self):
        from lumenairy import memory as m
        with pytest.raises(ValueError, match='n_grid must be positive'):
            m.estimate_asm_memory(0)
        with pytest.raises(ValueError, match='n_grid must be positive'):
            m.estimate_asm_memory(-8)
        with pytest.raises(ValueError, match='plan_cache_keys must be'):
            m.estimate_asm_memory(512, plan_cache_keys=-1)
        # plan_cache_keys=0 is legitimate (plan cache disabled).
        assert m.estimate_asm_memory(512, plan_cache_keys=0) > 0

    def test_estimate_sim_memory_still_lens_driven(self):
        """The re-derivation must not flip the documented driving step of
        the design-119 reference point."""
        d = la.estimate_sim_memory(16384, 'complex64', ray_subsample=8,
                                   itemized=True)
        assert d['driving_step'] == 'lens'


# =========================================================================
# A-7 -- OTF / MTF DC-location docstrings (documentation-only)
# =========================================================================

class TestA7OtfDcConvention:

    def test_dc_is_at_the_centre_not_the_corner(self):
        """The behavioural truth the docstrings must describe."""
        from lumenairy.analysis.psf_mtf_otf import compute_mtf, compute_otf
        n = 64
        yy, xx = np.mgrid[0:n, 0:n] - n // 2
        psf = np.exp(-(xx ** 2 + yy ** 2) / (2 * 3.0 ** 2))
        otf = compute_otf(psf)
        mtf = compute_mtf(psf)
        assert otf[n // 2, n // 2] == pytest.approx(1.0, abs=1e-12)
        assert abs(otf[0, 0]) < 1e-12
        assert mtf[n // 2, n // 2] == pytest.approx(1.0, abs=1e-12)
        assert mtf[0, 0] < 1e-12
        assert np.unravel_index(np.argmax(np.abs(otf)), otf.shape) == \
            (n // 2, n // 2)

    def test_docstrings_no_longer_claim_the_corner_is_dc(self):
        from lumenairy.analysis.psf_mtf_otf import compute_mtf, compute_otf
        for fn, arr in ((compute_otf, 'otf'), (compute_mtf, 'mtf')):
            doc = fn.__doc__ or ''
            assert f'{arr}[0, 0]`` (DC) = 1' not in doc, (
                f"A-7: {fn.__name__} still documents the corner as DC.")
            assert f'{arr}[0, 0]`` = 1 at DC' not in doc, (
                f"A-7: {fn.__name__} still documents the corner as DC.")
            assert 'fftshift' in doc, (
                f"A-7: {fn.__name__} must state the output is fftshifted.")
            assert 'N // 2, N // 2' in doc, (
                f"A-7: {fn.__name__} must name the actual DC index.")


# =========================================================================
# A-8 -- user_library corrupted-material diagnostics
# =========================================================================

class TestA8UserLibraryCorruptMaterialWarns:

    @pytest.fixture()
    def isolated_library(self, tmp_path, monkeypatch):
        import lumenairy.user_library as ul
        monkeypatch.setattr(ul, '_library_path', str(tmp_path))
        for sub in ('materials', 'lenses', 'phase_masks'):
            (tmp_path / sub).mkdir(exist_ok=True)
        return tmp_path

    def test_corrupt_entry_warns_and_names_file_and_exception(
            self, isolated_library):
        """Pre-fix: two ``except ...: pass`` sites, zero diagnostics --
        the user's glass just was not in GLASS_REGISTRY any more."""
        import lumenairy.user_library as ul
        mats = isolated_library / 'materials'
        (mats / 'a8_bad.json').write_text('{ not json', encoding='utf-8')
        ul.save_material('a8_good', n=1.5)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            ul.load_all_materials()
        msgs = [str(w.message) for w in caught
                if issubclass(w.category, UserWarning)]
        hits = [msg for msg in msgs if 'a8_bad' in msg]
        assert hits, (
            f"A-8: corrupted material skipped with no warning "
            f"(warnings seen: {msgs}).")
        msg = hits[0]
        assert 'a8_bad.json' in msg, "must name the file on disk"
        assert 'JSONDecodeError' in msg or 'ValueError' in msg, \
            "must name the exception type"
        # The skip is preserved: the healthy sibling still registered.
        from lumenairy.glass import GLASS_REGISTRY
        assert 'a8_good' in GLASS_REGISTRY

    def test_healthy_library_is_silent(self, isolated_library):
        import lumenairy.user_library as ul
        ul.save_material('a8_quiet', n=1.6)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            ul.load_all_materials()
        assert [str(w.message) for w in caught
                if 'user_library' in str(w.message)] == []


# =========================================================================
# A-9..A-14 (LOW) -- dead code, ownership docs, cache mutability, __all__
# =========================================================================

class TestLowHygiene:

    def test_jnp_or_none_is_gone(self):
        """Dead helper, zero references repo-wide."""
        import importlib
        mod = importlib.import_module('lumenairy.backend.fft')
        assert not hasattr(mod, '_jnp_or_none'), (
            "A-9..A-14: dead ``_jnp_or_none`` is back.")
        assert hasattr(mod, '_jnp_required')

    def test_jnp_or_none_has_no_references_in_the_package(self):
        import os

        import lumenairy
        root = os.path.dirname(os.path.abspath(lumenairy.__file__))
        hits = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != '__pycache__']
            for fn in filenames:
                if not fn.endswith(('.py', '.pyi')):
                    continue
                p = os.path.join(dirpath, fn)
                with open(p, encoding='utf-8', errors='ignore') as f:
                    for i, line in enumerate(f, 1):
                        if '_jnp_or_none' in line and not line.lstrip(
                                ).startswith('#'):
                            hits.append(f'{p}:{i}')
        assert hits == [], f"A-9..A-14: live references remain: {hits}"

    def test_public_fft_documents_buffer_ownership(self):
        """P13-P16: ``backend.fft.fft2/ifft2`` forward the pyFFTW
        ping-pong buffer with no copy; the ownership contract has to be in
        the docstring (measured clobber: max|delta| 3.66e3 on a 2.45e3-peak
        field after the 2nd subsequent call)."""
        import importlib
        mod = importlib.import_module('lumenairy.backend.fft')
        for name in ('fft2', 'ifft2'):
            doc = getattr(mod, name).__doc__ or ''
            low = doc.lower()
            assert 'ping-pong' in low, (
                f"A-9..A-14: {name} does not document the ping-pong "
                f"buffer it returns.")
            assert '.copy()' in doc, (
                f"A-9..A-14: {name} does not tell retaining callers to "
                f"copy.")
            assert 'one' in low, (
                f"A-9..A-14: {name} must state the one-call stability "
                f"window.")

    def test_public_fft2_really_returns_the_shared_buffer(self):
        """Counter-pin: prove the documented hazard is real, so the
        docstring is not describing a condition that no longer exists."""
        import importlib

        import lumenairy.propagators.fft_infra as fi
        if not (fi.PYFFTW_AVAILABLE and fi.USE_PYFFTW
                and fi.get_fft_double_buffer()):
            pytest.skip('pyFFTW double-buffer path not active')
        mod = importlib.import_module('lumenairy.backend.fft')
        n = max(512, int(fi.FFTW_MIN_SIZE))
        rng = np.random.default_rng(0)
        ins = [rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
               for _ in range(3)]
        first = mod.fft2(ins[0])
        assert first.flags.owndata is False
        snapshot = first.copy()
        mod.fft2(ins[1])
        assert np.array_equal(first, snapshot), (
            "documented window is ONE subsequent call; it broke earlier")
        mod.fft2(ins[2])
        assert not np.array_equal(first, snapshot), (
            "the second subsequent call must clobber the buffer -- if it "
            "no longer does, the fft2 ownership docstring is now wrong.")

    def test_zernike_cache_returns_readonly_arrays(self):
        """A-13ish: the cached basis came back writable, so one stray
        in-place write silently poisoned every later consumer."""
        from lumenairy.analysis.zernike import (
            clear_zernike_basis_cache,
            zernike_basis_matrix,
        )
        clear_zernike_basis_cache()
        try:
            g = np.linspace(-1, 1, 32)
            X, Y = np.meshgrid(g, g)
            basis, mask = zernike_basis_matrix(6, X, Y, 1.0)
            assert basis.flags.writeable is False
            assert mask.flags.writeable is False
            with pytest.raises(ValueError):
                basis[0, 0] = 12345.0
            # The contract must be identical on the cache-HIT path.
            basis2, mask2 = zernike_basis_matrix(6, X, Y, 1.0)
            assert basis2 is basis and mask2 is mask
            assert basis2.flags.writeable is False
            assert mask2.flags.writeable is False
            # Copies are writable, as documented.
            assert basis.copy().flags.writeable is True
        finally:
            clear_zernike_basis_cache()

    def test_zernike_readonly_changes_no_numbers(self):
        """The freeze must be numerically invisible: the cached basis has
        to stay bit-identical to the uncached build, and the public
        decompose path must still work end-to-end (incl. the WLS branch
        that rescales the basis out-of-place)."""
        from lumenairy.analysis.core import _zernike_basis_matrix_build
        from lumenairy.analysis.zernike import (
            clear_zernike_basis_cache,
            zernike_basis_matrix,
            zernike_decompose,
        )
        clear_zernike_basis_cache()
        try:
            g = np.linspace(-1, 1, 48)
            X, Y = np.meshgrid(g, g)
            cached, mask_c = zernike_basis_matrix(10, X, Y, 0.9)
            raw, mask_r = _zernike_basis_matrix_build(10, X, Y, 0.9)
            assert np.array_equal(cached, raw)
            assert np.array_equal(mask_c, mask_r)
            assert raw.flags.writeable is True   # uncached build unfrozen

            n, dx, ap = 48, 1e-5, 48 * 1e-5 * 0.8
            yy, xx = np.mgrid[0:n, 0:n] - n / 2
            opd = 1e-7 * ((xx * dx) ** 2 + (yy * dx) ** 2) / ap ** 2
            c_plain, _ = zernike_decompose(opd, dx, ap, n_modes=6)
            w = np.ones_like(opd)
            c_weighted, _ = zernike_decompose(opd, dx, ap, n_modes=6,
                                              weighting=w)
            assert np.all(np.isfinite(c_plain))
            np.testing.assert_allclose(c_plain, c_weighted, rtol=1e-8,
                                       atol=1e-14)
        finally:
            clear_zernike_basis_cache()

    @pytest.mark.parametrize('mod_name', [
        'storage', 'codegen', 'prescriptions', 'prescriptions_builders',
        'prescriptions_code_v', 'prescriptions_quadoa',
        'prescriptions_transforms', 'prescriptions_zemax',
    ])
    def test_io_modules_declare_all(self, mod_name):
        """``analysis/`` declares ``__all__`` in all 20 modules; ``io/``
        declared it in 1 of 8."""
        import importlib
        mod = importlib.import_module(f'lumenairy.io.{mod_name}')
        assert hasattr(mod, '__all__'), (
            f"A-9..A-14: lumenairy.io.{mod_name} has no __all__.")
        assert isinstance(mod.__all__, list) and mod.__all__
        missing = [n for n in mod.__all__ if not hasattr(mod, n)]
        assert not missing, (
            f"lumenairy.io.{mod_name}.__all__ lists unresolvable "
            f"{missing}")
        assert len(set(mod.__all__)) == len(mod.__all__)

    @pytest.mark.parametrize('mod_name', [
        'storage', 'codegen', 'prescriptions_builders',
        'prescriptions_code_v', 'prescriptions_quadoa',
        'prescriptions_transforms', 'prescriptions_zemax',
    ])
    def test_io_all_entries_reach_the_facade(self, mod_name):
        """Every declared public name must actually be reachable from
        ``lumenairy.io`` or the top-level facade -- otherwise ``__all__``
        is just a second, drifting manifest."""
        import importlib

        import lumenairy.io as lio
        mod = importlib.import_module(f'lumenairy.io.{mod_name}')
        orphans = [n for n in mod.__all__
                   if not (hasattr(lio, n) or hasattr(la, n))]
        assert not orphans, (
            f"lumenairy.io.{mod_name} exports {orphans} that neither "
            f"lumenairy.io nor lumenairy re-exports.")

    def test_negative_memory_costs_raise(self):
        """A negative byte cost was silently read as 'free' (max batch) /
        'fits' (no split) -- the worst possible answer for a sign slip."""
        from lumenairy import memory as m
        with pytest.raises(ValueError, match='cost_per_item must be'):
            m.pick_batch_size(10, -1)
        with pytest.raises(ValueError, match='total_cost must be'):
            m.should_split(-1)
        # Zero stays legitimate ("free"), and the honoured-override
        # behaviour of the P2-21 pins is untouched.
        assert m.pick_batch_size(10, 0) == 10
        assert m.should_split(0) is False
        # 512 KiB budget / 1 KiB per item = 512, under the 1000-item total.
        assert m.pick_batch_size(1000, 1024, available=1024 * 1024,
                                 safety=0.5) == 512
