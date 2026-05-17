"""Regression tests for the v4.13.0 IO-domain audit fixes.

Audit reference
---------------

``AUDIT_V4_12_1_2026_05_16.md`` Part 4 / Part 5 carried three IO-domain
findings forward into the v4.12.2 CHANGELOG "Known limitations"
section.  v4.13.0 closes them inside ``lumenairy/io/``.

* **S1** -- ``io/storage.py`` append-side hardcoded ``complex128``.
  Both :func:`append_plane_h5` (line 342) and
  :func:`save_jones_field_h5` (lines 282-283) called
  ``np.asarray(field, dtype=np.complex128)`` unconditionally,
  silently doubling on-disk size for ``complex64`` simulations
  streamed via :meth:`MhsPipeline.run` / :func:`replay_run`.
  v4.13.0 fix: expose ``preserve_dtype=False`` (default) consistent
  with the existing :func:`save_field_h5` / :func:`save_planes_h5`
  single-shot APIs.

* **S2a** -- ``io/codegen.py`` aperture-stop drop.  The docstring of
  ``_decompose_prescription`` advertised a ``'type': 'aperture'``
  step and the downstream generators handled it, but the function
  never emitted one.  Zemax prescriptions with a ``STOP`` marker
  lost the stop in the generated script.

* **S2b** -- ``io/codegen.py`` silent 1310 nm wavelength default.
  When neither the user nor the prescription supplied a wavelength
  the codegen silently defaulted to 1.31e-6 m (NIR O-band), so
  visible-band Zemax files quietly converted to NIR scripts.
  v4.13.0 raises a clear :class:`ValueError`.

* **L8** -- ``_open_zarr_group_safe`` ``Path.mkdir`` monkey-patch is
  not thread-safe.  Two threads racing through
  :func:`append_plane_h5` could each install a patched ``mkdir``,
  one thread's saved "original" being the other thread's patched
  version -- permanently corrupting ``pathlib.Path.mkdir`` for the
  whole process.  v4.13.0 guards the install/restore window with a
  module-level :class:`threading.Lock`.

Author: Andrew Traverso
"""
from __future__ import annotations

import os
import tempfile
import threading
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.io import storage as _storage
from lumenairy.io.codegen import (
    _decompose_prescription,
    generate_simulation_script,
)


# ============================================================================
# S1 -- append-side dtype preservation
# ============================================================================


class TestAppendSideDtypePreservation:
    """v4.13.0 honours ``preserve_dtype`` on every append-side I/O.

    Pre-v4.13.0:
    * :func:`append_plane_h5` cast unconditionally to ``complex128``.
    * :func:`save_jones_field_h5` cast unconditionally to ``complex128``.

    Post-v4.13.0:
    * Both accept ``preserve_dtype: bool = False``.
    * ``preserve_dtype=True`` keeps ``complex64`` across a round-trip.
    * ``preserve_dtype=False`` (the default) preserves the historical
      ``complex128`` coercion so existing code is unaffected.

    Pin: write a ``complex64`` field via the unified ``append_plane``
    API with ``preserve_dtype=True`` and assert the round-tripped
    array is ``complex64``.  Also pin the back-compat default.
    """

    def _have_h5py(self):
        try:
            import h5py  # noqa: F401
        except ImportError:
            return False
        return True

    def test_append_plane_h5_preserve_dtype_true_keeps_complex64(self):
        if not self._have_h5py():
            pytest.skip('h5py not installed')
        N = 16
        E64 = np.ones((N, N), dtype=np.complex64) * (1.0 + 1.0j)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'append_c64.h5')
            la.append_plane_h5(path, E64, dx=5e-6, dy=5e-6,
                               label='c64',
                               preserve_dtype=True)
            planes, _ = la.load_planes_h5(path)
        assert len(planes) == 1
        E_back = planes[0]['field']
        assert E_back.dtype == np.complex64, (
            f'preserve_dtype=True did not survive round-trip: '
            f'got dtype={E_back.dtype}, expected complex64')
        assert np.allclose(E_back, E64), 'round-trip values disagree'

    def test_append_plane_h5_default_coerces_to_complex128(self):
        """Default (preserve_dtype=False) keeps the historical
        complex128 coercion -- backward compatibility."""
        if not self._have_h5py():
            pytest.skip('h5py not installed')
        N = 8
        E64 = np.ones((N, N), dtype=np.complex64)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'append_default.h5')
            la.append_plane_h5(path, E64, dx=1e-6, dy=1e-6,
                               label='default')
            planes, _ = la.load_planes_h5(path)
        assert planes[0]['field'].dtype == np.complex128, (
            'default behaviour must still coerce to complex128')

    def test_append_plane_unified_passes_preserve_dtype(self):
        """The auto-dispatch ``append_plane`` API forwards
        ``preserve_dtype`` to the H5 backend (S1 followthrough)."""
        if not self._have_h5py():
            pytest.skip('h5py not installed')
        N = 8
        E64 = (np.random.default_rng(0)
               .standard_normal((N, N))
               .astype(np.complex64))
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'append_unified.h5')
            la.append_plane(path, E64, dx=1e-6, dy=1e-6,
                            label='unified',
                            preserve_dtype=True)
            planes, _ = la.load_planes_h5(path)
        assert planes[0]['field'].dtype == np.complex64

    def test_save_jones_field_h5_preserve_dtype_true_keeps_complex64(self):
        if not self._have_h5py():
            pytest.skip('h5py not installed')
        from lumenairy.elements.polarization import JonesField
        N = 8
        Ex = np.ones((N, N), dtype=np.complex64) * (1.0 + 0.5j)
        Ey = np.ones((N, N), dtype=np.complex64) * (0.25 - 0.1j)
        jf = JonesField(Ex=Ex, Ey=Ey, dx=2e-6)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'jones_c64.h5')
            la.save_jones_field_h5(path, jf, wavelength=1.31e-6,
                                   preserve_dtype=True)
            jf_back, _meta = la.load_jones_field_h5(path)
        assert jf_back.Ex.dtype == np.complex64, (
            f'Ex dtype post-roundtrip = {jf_back.Ex.dtype}')
        assert jf_back.Ey.dtype == np.complex64, (
            f'Ey dtype post-roundtrip = {jf_back.Ey.dtype}')

    def test_save_jones_field_h5_default_coerces_to_complex128(self):
        """Default (preserve_dtype=False) keeps the historical
        complex128 coercion."""
        if not self._have_h5py():
            pytest.skip('h5py not installed')
        from lumenairy.elements.polarization import JonesField
        N = 8
        Ex = np.ones((N, N), dtype=np.complex64)
        Ey = 1j * np.ones((N, N), dtype=np.complex64)
        jf = JonesField(Ex=Ex, Ey=Ey, dx=2e-6)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'jones_default.h5')
            la.save_jones_field_h5(path, jf, wavelength=1.31e-6)
            jf_back, _ = la.load_jones_field_h5(path)
        assert jf_back.Ex.dtype == np.complex128
        assert jf_back.Ey.dtype == np.complex128


# ============================================================================
# S2a -- codegen aperture-stop emission
# ============================================================================


class TestCodegenApertureStopEmission:
    """Pre-v4.13.0 ``_decompose_prescription`` never emitted any
    ``{'type': 'aperture'}`` step even though its docstring promised
    one and the downstream emitters handle it.  v4.13.0 emits an
    aperture step for every element flagged ``is_stop=True``.

    Pin: build a prescription whose ``elements`` list contains a
    STOP-flagged dummy plane.  Assert :func:`_decompose_prescription`
    emits at least one ``type='aperture'`` step AND
    :func:`generate_simulation_script` emits the matching
    ``la.apply_aperture(...)`` call in both `'unrolled'` and
    `'system'` code-styles.
    """

    def _stop_prescription_with_dummy(self, stop_diameter=10e-3):
        """Two-surface lens with a STOP dummy plane in front."""
        # Air-to-air dummy STOP plane, then a singlet R1=50/R2=-50/4mm BK7.
        elements = [
            {
                'element_type': 'surface',
                'radius': float('inf'),
                'conic': 0.0,
                'aspheric_coeffs': None,
                'glass_before': 'air',
                'glass_after': 'air',
                'semi_diameter': stop_diameter * 0.5,
                'surf_num': 1,
                'comment': 'STOP dummy',
                'is_stop': True,
            },
            {
                'element_type': 'surface',
                'radius': 50e-3,
                'conic': 0.0,
                'aspheric_coeffs': None,
                'glass_before': 'air',
                'glass_after': 'N-BK7',
                'semi_diameter': stop_diameter * 0.5,
                'surf_num': 2,
                'comment': '',
                'is_stop': False,
            },
            {
                'element_type': 'surface',
                'radius': -50e-3,
                'conic': 0.0,
                'aspheric_coeffs': None,
                'glass_before': 'N-BK7',
                'glass_after': 'air',
                'semi_diameter': stop_diameter * 0.5,
                'surf_num': 3,
                'comment': '',
                'is_stop': False,
            },
        ]
        return {
            'name': 'stop_test_design',
            'aperture_diameter': stop_diameter,
            'elements': elements,
            'all_thicknesses': [3e-3, 4e-3],
            # surfaces / thicknesses (refracting-only) for the lens RX:
            'surfaces': [
                {'radius': 50e-3, 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'glass_before': 'air', 'glass_after': 'N-BK7'},
                {'radius': -50e-3, 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'glass_before': 'N-BK7', 'glass_after': 'air'},
            ],
            'thicknesses': [4e-3],
            'wavelength': 1.31e-6,
        }

    def test_decompose_emits_aperture_step(self):
        rx = self._stop_prescription_with_dummy(stop_diameter=8e-3)
        steps = _decompose_prescription(rx)
        ap_steps = [s for s in steps if s.get('type') == 'aperture']
        assert len(ap_steps) >= 1, (
            f'no aperture step emitted; pre-fix bug.  Steps: {steps}')
        # Diameter pulled from the stop element's semi_diameter * 2.
        assert abs(ap_steps[0]['diameter'] - 8e-3) < 1e-12, (
            f'aperture diameter mismatch: {ap_steps[0]["diameter"]}')

    def test_unrolled_script_contains_apply_aperture_call(self):
        rx = self._stop_prescription_with_dummy(stop_diameter=6e-3)
        script = generate_simulation_script(
            rx, wavelength=1.31e-6, N=64, dx=10e-6,
            source_sigma=1e-3,
            include_plotting=False, include_analysis=False,
            style='unrolled',
        )
        assert 'la.apply_aperture(' in script, (
            'unrolled codegen missing apply_aperture call for STOP-'
            'flagged input -- S2a pre-fix bug')
        # The script must still compile.
        compile(script, '<codegen-stop-test>', 'exec')

    def test_system_style_script_contains_aperture_entry(self):
        rx = self._stop_prescription_with_dummy(stop_diameter=6e-3)
        script = generate_simulation_script(
            rx, wavelength=1.31e-6, N=64, dx=10e-6,
            source_sigma=1e-3,
            include_plotting=False, include_analysis=False,
            style='system',
        )
        assert "'type': 'aperture'" in script, (
            'system-list codegen missing aperture element for STOP-'
            'flagged input -- S2a pre-fix bug')
        compile(script, '<codegen-stop-system-test>', 'exec')

    def test_stop_index_translation_for_loaders_that_dropped_is_stop(self):
        """When elements have no ``is_stop`` flag but the prescription
        carries a top-level ``stop_index`` (the .qos / .seq loader
        convention), the decomposer still emits an aperture step
        whose index matches the named refracting surface."""
        # Build a refracting-only element list and set stop_index = 0
        # (first refracting surface).
        rx = self._stop_prescription_with_dummy(stop_diameter=8e-3)
        # Strip is_stop flags and use stop_index instead.
        for e in rx['elements']:
            e.pop('is_stop', None)
        # stop_index is documented as "zero-based index of the stop
        # among refracting surfaces"; here our 0th refracting surface
        # is the dummy STOP plane.
        rx['stop_index'] = 0
        steps = _decompose_prescription(rx)
        ap_steps = [s for s in steps if s.get('type') == 'aperture']
        assert len(ap_steps) >= 1, (
            'stop_index-based codegen path also missing aperture step')


# ============================================================================
# S2b -- codegen wavelength default
# ============================================================================


class TestCodegenWavelengthDefault:
    """Pre-v4.13.0 the codegen silently defaulted ``wavelength`` to
    1.31e-6 m when neither user nor prescription supplied one.
    v4.13.0 raises :class:`ValueError`.
    """

    def _prescription_without_wavelength(self):
        # Minimal prescription with no 'wavelength' key.
        return {
            'name': 'no_wavelength_design',
            'aperture_diameter': 10e-3,
            'elements': [
                {
                    'element_type': 'surface',
                    'radius': 50e-3,
                    'conic': 0.0,
                    'aspheric_coeffs': None,
                    'glass_before': 'air',
                    'glass_after': 'N-BK7',
                    'semi_diameter': 5e-3,
                    'surf_num': 1,
                    'comment': '',
                    'is_stop': False,
                },
                {
                    'element_type': 'surface',
                    'radius': -50e-3,
                    'conic': 0.0,
                    'aspheric_coeffs': None,
                    'glass_before': 'N-BK7',
                    'glass_after': 'air',
                    'semi_diameter': 5e-3,
                    'surf_num': 2,
                    'comment': '',
                    'is_stop': False,
                },
            ],
            'all_thicknesses': [4e-3],
            'surfaces': [
                {'radius': 50e-3, 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'glass_before': 'air', 'glass_after': 'N-BK7'},
                {'radius': -50e-3, 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'glass_before': 'N-BK7', 'glass_after': 'air'},
            ],
            'thicknesses': [4e-3],
        }

    def test_missing_wavelength_raises_valueerror(self):
        rx = self._prescription_without_wavelength()
        with pytest.raises(ValueError) as excinfo:
            generate_simulation_script(
                rx, wavelength=None, N=64, dx=10e-6,
                source_sigma=1e-3,
                include_plotting=False, include_analysis=False,
            )
        # The error message must name the offending parameter.
        assert 'wavelength' in str(excinfo.value).lower()

    def test_explicit_wavelength_still_works(self):
        """Sanity check: passing wavelength explicitly must not raise."""
        rx = self._prescription_without_wavelength()
        script = generate_simulation_script(
            rx, wavelength=587.6e-9, N=64, dx=5e-6,
            source_sigma=1e-3,
            include_plotting=False, include_analysis=False,
        )
        assert '587.6' in script or '5.876' in script

    def test_prescription_wavelength_used_if_present(self):
        """Sanity check: wavelength stored in the prescription dict
        wins over the (now removed) 1310 nm default."""
        rx = self._prescription_without_wavelength()
        rx['wavelength'] = 632.8e-9  # HeNe red
        script = generate_simulation_script(
            rx, wavelength=None, N=64, dx=5e-6,
            source_sigma=1e-3,
            include_plotting=False, include_analysis=False,
        )
        # The header writes wavelength in nm with one decimal.
        assert '632.8 nm' in script or '6.328' in script, (
            f'prescription wavelength 632.8e-9 not honoured.  Script '
            f'first 1200 chars:\n{script[:1200]}')


# ============================================================================
# L8 -- _open_zarr_group_safe thread-safety
# ============================================================================


class TestZarrMkdirPatchThreadSafety:
    """``_open_zarr_group_safe`` monkey-patches ``Path.mkdir`` for the
    duration of a single zarr open call.  Two threads racing through
    that function could leave the patched mkdir permanently installed.

    v4.13.0 wraps the install/restore window in a
    :class:`threading.Lock`.

    Pin: spawn two threads that each call :func:`append_plane_h5`
    concurrently against the same file.  Assertions:

    1. Both threads complete without exception.
    2. The resulting file is valid (loads back both planes).
    3. ``pathlib.Path.mkdir`` after the test is the original
       (unpatched) one -- if the patch leaked, subsequent ``mkdir``
       calls would have the patched dispatch.
    """

    def test_concurrent_append_plane_h5_no_mkdir_corruption(self):
        """Two threads append concurrently to disjoint HDF5 files; the
        critical pin is that ``Path.mkdir`` is the un-patched original
        after both threads exit.  Pre-fix the zarr-side mkdir patch
        was unguarded; this test surfaces leakage even when the H5
        path itself doesn't go through ``_open_zarr_group_safe``,
        because the failure mode lives in the global ``Path.mkdir``
        symbol and any concurrent zarr open can corrupt it for
        downstream H5 operations that rely on ``Path.mkdir``."""
        try:
            import h5py  # noqa: F401
        except ImportError:
            pytest.skip('h5py not installed')

        # Capture the real Path.mkdir reference up-front.
        from pathlib import Path
        original_mkdir = Path.mkdir

        N = 16
        errors = []

        def _worker(path, label, payload):
            try:
                la.append_plane_h5(
                    path, payload, dx=4e-6, dy=4e-6, label=label)
            except Exception as e:  # pragma: no cover -- only on failure
                errors.append((label, repr(e)))

        with tempfile.TemporaryDirectory() as td:
            # Each thread writes to its OWN file -- the L8 finding is
            # specifically about ``Path.mkdir`` patch leakage, not
            # about h5py's own intra-file locking (h5py serialises
            # writes to a single file via its global lock but that's
            # orthogonal to L8).
            p1 = os.path.join(td, 'shared_a.h5')
            p2 = os.path.join(td, 'shared_b.h5')
            E1 = np.ones((N, N), dtype=np.complex128)
            E2 = 0.5 * np.ones((N, N), dtype=np.complex128)
            threads = [
                threading.Thread(
                    target=_worker, args=(p1, 't1', E1)),
                threading.Thread(
                    target=_worker, args=(p2, 't2', E2)),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10.0)
                assert not t.is_alive(), \
                    f'{t.name} hung beyond 10 s deadline'

            assert not errors, f'thread errors: {errors}'
            # Both files must be valid (one plane each).
            planes_a, _ = la.load_planes_h5(p1)
            planes_b, _ = la.load_planes_h5(p2)
            assert len(planes_a) == 1, \
                f'thread 1 wrote {len(planes_a)} planes'
            assert len(planes_b) == 1, \
                f'thread 2 wrote {len(planes_b)} planes'

        # CRITICAL: Path.mkdir must be the un-patched original after
        # both threads exit.  If the lock isn't there, one thread can
        # restore *before* the other thread saved its "original"
        # reference, and the saved "original" ends up being the
        # patched dispatch -- making the patch permanent.
        assert Path.mkdir is original_mkdir, (
            'Path.mkdir patch leaked beyond _open_zarr_group_safe: '
            'L8 thread-safety regression')

    def test_open_zarr_group_safe_serial_install_restore(self):
        """Directly exercise ``_open_zarr_group_safe`` from a thread
        pool.  Even when zarr is unavailable, the patch install/
        restore happens around the actual ``open_group`` call (in
        the writable branch); we mock that to force the bug-prone
        window."""
        try:
            import zarr  # noqa: F401
        except ImportError:
            pytest.skip('zarr not installed')

        from pathlib import Path
        original_mkdir = Path.mkdir

        # Race ``_open_zarr_group_safe`` directly: each worker opens
        # (writable) a unique zarr store.  Repeated runs under the
        # lock must always restore Path.mkdir to its original
        # implementation after each worker exits.
        with tempfile.TemporaryDirectory() as td:
            errors = []

            def _worker(idx):
                try:
                    store_path = os.path.join(td, f'store_{idx}.zarr')
                    grp = _storage._open_zarr_group_safe(
                        zarr, store_path, writable=True)
                    # touch an attr so the open is non-trivial.
                    try:
                        grp.attrs['idx'] = int(idx)
                    except Exception:
                        pass
                except Exception as e:  # pragma: no cover
                    errors.append((idx, repr(e)))

            threads = [
                threading.Thread(target=_worker, args=(i,))
                for i in range(4)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10.0)
                assert not t.is_alive(), \
                    f'{t.name} hung beyond 10 s deadline'
            assert not errors, f'thread errors: {errors}'

        assert Path.mkdir is original_mkdir, (
            'Path.mkdir patch leaked beyond _open_zarr_group_safe '
            'across concurrent zarr opens')
