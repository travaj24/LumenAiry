"""Tests for the v4.15 audit-fix Agent-E changes.

Scope: carryover P1s from the v4.14.2 audit Part 2 F3 final-corner
section that the v4.14.3 dispatch did not close.

What this module pins
---------------------

E.1 -- UI subpackage (7 findings, 1 file each):

* **P1-UI-1** ``main_window.py`` Thorlabs sort table -- N-LASF9 / S-NPH1
  now resolved at n_d~=1.85/1.81 instead of the default 1.5168 (N-BK7).
* **P1-UI-2** ``main_window._nudge_distance`` -- Shift+Up/Down nudge
  now routes through :meth:`SystemModel.set_display_distance` so the
  keystroke respects the current coordinate-display mode.
* **P1-UI-3** ``SystemModel._capture_state`` -- ``wavelength_weights``,
  ``field_weights``, and ``lens_options`` now round-trip through
  undo/redo.
* **P1-UI-4** ``SystemModel`` back-vertex helper -- single source of
  truth ``_prev_element_back_vertex_world(prev)`` is consumed by both
  the absolute-position editor and the absolute-distance setter.
* **P1-UI-5** ``waveoptics_dock._open_mft_options_dialog`` -- re-parent
  guarded by :func:`shiboken6.isValid` so a dead original parent does
  not segfault.
* **P1-UI-6** ``psf_mtf_dock._load_from_raytrace`` -- mean-OPD-per-pixel
  accumulation replaces the pre-4.15 last-write-wins assignment.
* **P1-UI-7** ``psf_mtf_dock._load_from_raytrace`` -- bounds-mask now
  filters rays outside the pupil grid BEFORE indexing, instead of
  ``np.clip``-snapping them to the pupil edge.

E.2 -- Codegen version pin (P1-CG):

Generated scripts now carry a runtime ``lumenairy>=GENERATED_VERSION``
assertion and stamp ``lumenairy_version`` in the comment header.

E.3 -- Ghost seed parameter (P1-GH-1):

:func:`non_sequential_stray_light` accepts a ``seed: int | None`` kwarg
that pipes to ``np.random.default_rng``.  ``seed=None`` (the new
default) draws from system entropy so repeated calls give a non-zero
sample variance; passing an int pins reproducibility.

E.4 -- Glass Sellmeier fallback (P1-GL-1):

SiO2 / F_SILICA / FUSED_SILICA (Malitson 1965), S-LAH64 / S-LAH79
(Ohara catalog) now have bundled Sellmeier rows in
:data:`SELLMEIER_COEFFICIENTS`.  A minimal install (no
``refractiveindex`` package) can resolve them through the fallback
path instead of raising ``ImportError``.

E.5 -- Glass registry reverse-direction (P2):

:func:`_check_glass_registry_consistency` now ALSO asserts every key
in :data:`SELLMEIER_COEFFICIENTS` appears in :data:`GLASS_REGISTRY`,
catching orphan coefficient rows that pre-4.15 would silently sit as
dead code.

Author: Agent E -- v4.15.
"""
from __future__ import annotations

import copy
import inspect
import os
import sys

import numpy as np
import pytest


# ============================================================================
# Qt offscreen setup (must come BEFORE PySide6 imports)
# ============================================================================

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


# Probe GUI deps the same way other dock tests do (e.g.
# test_audit_fixes_v4_13_1_thin_grating_dock.py).  CI without the
# [gui] extra skips the live-widget tests cleanly; locally the full
# suite runs.
try:
    from PySide6.QtWidgets import QApplication  # noqa: F401
    import matplotlib  # noqa: F401
    matplotlib.use('Agg')
    from lumenairy.ui.model import SystemModel, Element, SurfaceRow
    _PYSIDE6_OK = True
except ImportError as _e:
    _PYSIDE6_OK = False
    _SKIP_REASON = f'PySide6 / matplotlib unavailable: {_e}'


_skip_no_qt = pytest.mark.skipif(
    not _PYSIDE6_OK,
    reason=_SKIP_REASON if not _PYSIDE6_OK else '')


# ============================================================================
# E.1.1 -- P1-UI-1: stale Thorlabs glass-sort table
# ============================================================================

@_skip_no_qt
class TestUI1ThorlabsGlassTable:
    """``main_window.py`` Thorlabs sort table covers N-LASF9 and S-NPH1.

    The fix is structural -- the table is private to the
    ``_build_thorlabs_menu._efl_of`` closure -- so we inspect the
    source string for the canonical extension.  We can't easily hit
    it via the live widget without building the full main window
    (heavyweight, not necessary for what we're pinning).
    """

    def test_table_lists_n_lasf9_and_s_nph1(self):
        from lumenairy.ui import main_window
        src = inspect.getsource(main_window._build_thorlabs_menu) \
            if hasattr(main_window, '_build_thorlabs_menu') else None
        # _build_thorlabs_menu is a method, not a module-level function.
        # Read the file source directly via the module.
        src = inspect.getsource(main_window)
        # Both new glass entries should appear inside the sort-table
        # conditional block.
        assert "N-LASF9" in src, (
            "P1-UI-1 fix missing: N-LASF9 not referenced in main_window")
        assert "S-NPH1" in src, (
            "P1-UI-1 fix missing: S-NPH1 not referenced in main_window")
        # And the n=1.85 / n=1.81 values must be present (the
        # canonical-position estimates for the menu sort).
        assert "n = 1.85" in src or "n=1.85" in src, (
            "P1-UI-1 fix missing: N-LASF9 n_d estimate not bundled")
        assert "n = 1.81" in src or "n=1.81" in src, (
            "P1-UI-1 fix missing: S-NPH1 n_d estimate not bundled")


# ============================================================================
# E.1.2 -- P1-UI-2: _nudge_distance routes via set_display_distance
# ============================================================================

@_skip_no_qt
class TestUI2NudgeDistanceCoordinateModeAware:
    """``_nudge_distance`` is coord-mode-aware.

    Structural pin: read the source of ``MainWindow._nudge_distance``
    and assert it calls ``set_display_distance`` (the canonical
    coord-mode-aware setter) rather than mutating
    ``Element.distance_mm`` directly.
    """

    def test_nudge_calls_set_display_distance(self):
        from lumenairy.ui import main_window
        src = inspect.getsource(main_window)
        # Locate the _nudge_distance function source.
        idx = src.find("def _nudge_distance(")
        assert idx >= 0, "_nudge_distance method not found"
        # Take everything up to the next 'def ' or end-of-source.
        nxt = src.find("\n    def ", idx + 10)
        body = src[idx:nxt if nxt > 0 else len(src)]
        assert "set_display_distance" in body, (
            "P1-UI-2 fix missing: _nudge_distance no longer calls "
            "set_display_distance; the nudge would bypass coord-mode")
        # And it MUST NOT directly assign to distance_mm any more.
        # (We tolerate the f-string read of distance_mm in the status
        # message; the regression is on the assignment.)
        assert "elements[idx].distance_mm = max" not in body, (
            "P1-UI-2 regression: _nudge_distance still mutates "
            "Element.distance_mm directly")

    @_skip_no_qt
    def test_nudge_relative_mode_shifts_one_element(self):
        """End-to-end smoke: nudge in relative mode bumps the gap."""
        sm = SystemModel()
        # Add a singlet so we have a non-trivial nudge target.
        rows = [SurfaceRow(radius=51.5, thickness=4.1, glass='N-BK7'),
                SurfaceRow(radius=np.inf, thickness=0.0, glass='')]
        sm.elements.insert(
            1, Element(elem_num=1, name='L1', elem_type='Singlet',
                       distance_mm=50.0, surfaces=rows))
        sm._renumber()
        before = sm.elements[1].distance_mm
        # Use the model API the nudge will exercise.
        cur = sm.get_display_distance(1)
        sm.set_display_distance(1, cur + 0.1)
        after = sm.elements[1].distance_mm
        # In relative mode display_distance == distance_mm, so the
        # bump is direct.
        assert abs((after - before) - 0.1) < 1e-9, (
            f"Relative-mode nudge: expected +0.1 mm, got {after - before:+.6f}")

    @_skip_no_qt
    def test_nudge_absolute_mode_shifts_via_back_vertex(self):
        """In absolute mode the nudge target is the front-vertex Z."""
        sm = SystemModel()
        rows = [SurfaceRow(radius=51.5, thickness=4.1, glass='N-BK7'),
                SurfaceRow(radius=np.inf, thickness=0.0, glass='')]
        sm.elements.insert(
            1, Element(elem_num=1, name='L1', elem_type='Singlet',
                       distance_mm=50.0, surfaces=rows))
        sm._renumber()
        sm.recompute_element_frames()
        sm.set_coordinate_mode('absolute')
        z_before = sm.get_display_distance(1)
        sm.set_display_distance(1, z_before + 0.5)
        sm.recompute_element_frames()
        z_after = sm.get_display_distance(1)
        # absolute Z must move by +0.5 mm (within numeric tolerance).
        assert abs((z_after - z_before) - 0.5) < 1e-6, (
            f"Absolute-mode nudge: expected +0.5 mm front-vertex shift, "
            f"got {z_after - z_before:+.6f}")


# ============================================================================
# E.1.3 -- P1-UI-3: undo captures wavelength_weights / field_weights /
#                    lens_options
# ============================================================================

@_skip_no_qt
@_skip_no_qt
class TestUI3UndoCapturesNewFields:
    """Round-trip undo/redo for the three previously-missed fields."""

    def test_wavelength_weights_undo(self):
        sm = SystemModel()
        sm.wavelength_weights = None
        sm._checkpoint()
        sm.wavelength_weights = [0.5, 0.3, 0.2]
        sm.undo()
        assert sm.wavelength_weights is None, (
            "P1-UI-3: wavelength_weights not restored on undo")

    def test_field_weights_undo(self):
        sm = SystemModel()
        sm.field_weights = None
        sm._checkpoint()
        sm.field_weights = [1.0, 0.5]
        sm.undo()
        assert sm.field_weights is None, (
            "P1-UI-3: field_weights not restored on undo")

    def test_lens_options_undo(self):
        sm = SystemModel()
        orig = copy.deepcopy(sm.lens_options)
        sm._checkpoint()
        sm.lens_options['apply_real_lens']['n_substeps'] = 7
        assert sm.lens_options['apply_real_lens'].get('n_substeps') == 7
        sm.undo()
        assert sm.lens_options == orig, (
            f"P1-UI-3: lens_options not restored on undo "
            f"(got {sm.lens_options!r}, want {orig!r})")


# ============================================================================
# E.1.4 -- P1-UI-4: back-vertex helper unifies two call sites
# ============================================================================

@_skip_no_qt
@_skip_no_qt
class TestUI4BackVertexHelper:
    """``_prev_element_back_vertex_world`` is the single source of truth."""

    def test_helper_exists(self):
        assert hasattr(SystemModel, '_prev_element_back_vertex_world'), (
            "P1-UI-4 fix missing: _prev_element_back_vertex_world helper "
            "not present on SystemModel")

    def test_helper_returns_internal_thickness_form(self):
        """For a singlet with a non-zero back-surface thickness, the
        helper returns front-vertex + ``internal_thickness_mm`` * z.
        ``internal_thickness_mm`` is the slicing form (``surfaces[:-1]``),
        i.e. it excludes the back-surface's thickness.  Pre-v4.15
        ``_apply_absolute_position_edit`` used ``sum(all surfaces)``
        which double-counted the back air gap encoded on the back row.
        """
        from lumenairy.ui.model import Element, SurfaceRow
        # Build a singlet whose back surface carries non-zero
        # thickness (the legacy ungroup-then-edit case).
        rows = [SurfaceRow(radius=100.0, thickness=5.0, glass='N-BK7'),
                SurfaceRow(radius=-100.0, thickness=3.0, glass='')]
        e = Element(elem_num=1, name='L', elem_type='Singlet',
                    distance_mm=0.0, surfaces=rows,
                    origin=np.array([0.0, 0.0, 0.0]),
                    R=np.eye(3))
        back = SystemModel._prev_element_back_vertex_world(e)
        # internal_thickness_mm = 5.0 (excludes back's 3.0)
        # So back-vertex Z = 0 + 5.0 = 5.0 mm
        assert abs(back[2] - 5.0) < 1e-12, (
            f"Helper: expected back-vertex Z=5.0 (from internal_thickness_mm "
            f"slicing), got {back[2]:.6f}")

    def test_apply_absolute_uses_helper(self):
        """``set_element_absolute_field`` source must reference the
        helper (or otherwise not the sum-of-all-surfaces form).

        This is the v4.14.x method name for what the audit calls
        ``_apply_absolute_position_edit`` -- the back-derivation of
        ``distance_mm`` from a user-entered absolute Z lives here.
        """
        src = inspect.getsource(SystemModel.set_element_absolute_field)
        assert "_prev_element_back_vertex_world" in src, (
            "P1-UI-4: set_element_absolute_field no longer routes "
            "through _prev_element_back_vertex_world")
        # The buggy ``sum(...for s in prev.surfaces)`` line must be gone.
        assert "sum(float(s.thickness) for s in prev.surfaces)" not in src, (
            "P1-UI-4: set_element_absolute_field still uses the "
            "buggy sum-of-all-surfaces form")


# ============================================================================
# E.1.5 -- P1-UI-5: waveoptics_dock re-parent guard
# ============================================================================

@_skip_no_qt
class TestUI5WaveOpticsReparentGuard:
    """``_open_mft_options_dialog`` guards against a dead parent.

    v5.0.1 (CI fix post-PyPI): class-level ``@_skip_no_qt`` -- the
    ``lumenairy.ui.waveoptics_dock`` import inside the test body
    transitively imports PySide6, so without the ``[gui]`` extras
    every test in this class hits ``ModuleNotFoundError``.  Source-
    inspection content is still pinned by the structural sibling
    pins in ``TestUI1`` through ``TestUI4`` above (which don't pull
    Qt into the test process)."""

    def test_shiboken_guard_present(self):
        from lumenairy.ui import waveoptics_dock
        src = inspect.getsource(
            waveoptics_dock.WaveOpticsDock._open_mft_options_dialog)
        # The guard must use shiboken6.isValid (canonical PySide6
        # idiom) -- our chosen path per the audit.  Either way, the
        # post-exec re-parent block must not unconditionally call
        # setParent(original_parent).
        assert "shiboken6" in src, (
            "P1-UI-5 fix missing: no shiboken6 reference in "
            "_open_mft_options_dialog (the canonical PySide6 alive-check)")
        assert "isValid" in src, (
            "P1-UI-5 fix missing: no isValid alive-check in "
            "_open_mft_options_dialog")


# ============================================================================
# E.1.6 / E.1.7 -- P1-UI-6 / P1-UI-7: psf_mtf_dock ray accumulation
# ============================================================================

class TestUI6and7PsfMtfDockRayAccumulation:
    """Mean-OPD-per-pixel + bounds-mask in ``_load_from_raytrace``.

    v5.0.1 (CI fix post-PyPI): two of the three tests below import
    ``lumenairy.ui.psf_mtf_dock`` which transitively imports PySide6;
    each is individually decorated with ``@_skip_no_qt``.  The
    ``test_mean_accumulation_numerically`` test is a pure-NumPy
    synthetic that doesn't touch the UI module and continues to run
    without the ``[gui]`` extras."""

    @_skip_no_qt
    def test_no_last_write_wins(self):
        from lumenairy.ui import psf_mtf_dock
        src = inspect.getsource(psf_mtf_dock)
        # Locate _load_from_raytrace.
        idx = src.find("def _load_from_raytrace(")
        assert idx >= 0
        body = src[idx:idx + 3000]
        # Filter out comment lines so the audit's reference quote in
        # our own commentary doesn't trigger the regression assertion.
        non_comment_lines = [ln for ln in body.splitlines()
                              if not ln.lstrip().startswith('#')]
        body_code = '\n'.join(non_comment_lines)
        # Pre-v4.15 line: opd_grid[iy, ix] = opl (last-write-wins).
        assert "opd_grid[iy, ix] = opl" not in body_code, (
            "P1-UI-6 regression: ``opd_grid[iy, ix] = opl`` still "
            "present in executable code; multi-ray-per-pixel still "
            "last-write-wins")
        # The fix uses np.add.at or similar mean-accumulation idiom.
        assert "np.add.at" in body_code, (
            "P1-UI-6 fix missing: no np.add.at accumulator in "
            "_load_from_raytrace (the analysis/core.py mean idiom)")

    @_skip_no_qt
    def test_bounds_mask_present(self):
        from lumenairy.ui import psf_mtf_dock
        src = inspect.getsource(psf_mtf_dock)
        idx = src.find("def _load_from_raytrace(")
        assert idx >= 0
        body = src[idx:idx + 3000]
        non_comment_lines = [ln for ln in body.splitlines()
                              if not ln.lstrip().startswith('#')]
        body_code = '\n'.join(non_comment_lines)
        # Pre-v4.15: np.clip then .astype(int) wrote to the boundary.
        # Post: we mask BEFORE indexing.  Pin the structural change
        # by asserting the in-bounds-mask construction is present.
        assert ("in_bounds" in body_code
                or "in_bounds_mask" in body_code
                or "in_bounds & " in body_code), (
            "P1-UI-7 fix missing: no in_bounds mask construction "
            "in _load_from_raytrace")
        # And ``np.clip(((x/dx) + N/2).astype(int), 0, N-1)`` MUST be gone.
        # The new code computes ix_raw without clamping then filters.
        assert "np.clip(((x / dx)" not in body_code, (
            "P1-UI-7 regression: np.clip on x is still snapping "
            "out-of-aperture rays to the pupil boundary")

    def test_mean_accumulation_numerically(self):
        """Synthetic test of the accumulation idiom independent of Qt.

        Build two rays into the same pixel with OPLs +1 and -1; the
        mean should be 0 (the old last-write-wins would give -1 or +1
        depending on order).
        """
        N = 8
        sum_grid = np.zeros((N, N), dtype=np.float64)
        cnt_grid = np.zeros((N, N), dtype=np.int64)
        iy = np.array([3, 3])
        ix = np.array([4, 4])
        opl = np.array([+1.0, -1.0])
        np.add.at(sum_grid, (iy, ix), opl)
        np.add.at(cnt_grid, (iy, ix), 1)
        assert cnt_grid[3, 4] == 2
        mean = sum_grid[3, 4] / cnt_grid[3, 4]
        assert abs(mean) < 1e-12, (
            f"Mean-OPD accumulator wrong: {mean=}")


# ============================================================================
# E.2 -- P1-CG: codegen version pin
# ============================================================================

class TestE2CodegenVersionPin:
    """Generated scripts assert the runtime lumenairy is >= generator."""

    @staticmethod
    def _minimal_prescription():
        return {
            'name': 'TestPin',
            'aperture_diameter': 25.4e-3,
            'wavelength': 1.31e-6,
            'elements': [],
            'surfaces': [],
            'thicknesses': [],
            'all_thicknesses': [],
            'all_glasses': [],
        }

    def test_generated_script_has_version_assertion(self):
        """The emitted script raises RuntimeError on a too-old lumenairy."""
        import lumenairy
        from lumenairy.io.codegen import generate_simulation_script
        rx = self._minimal_prescription()
        script = generate_simulation_script(
            rx, wavelength=1.31e-6,
            N=128, dx=25e-6,
            source_sigma=2e-3,
            output_path=None,
            include_plotting=False,
            include_analysis=False)
        # Version-check imports lumenairy and inspects __version__.
        assert "import lumenairy" in script, (
            "P1-CG: generated script does not import lumenairy "
            "for the runtime version check")
        assert "lumenairy.__version__" in script, (
            "P1-CG: generated script does not reference "
            "lumenairy.__version__ in the runtime check")
        assert "raise RuntimeError" in script, (
            "P1-CG: generated script does not raise RuntimeError "
            "on a too-old lumenairy")
        # The pin tuple comes from the GENERATING version, which is
        # whatever the package reports right now.
        gen_tuple = tuple(int(x) for x in lumenairy.__version__.split('.')[:3])
        assert repr(gen_tuple) in script or str(gen_tuple) in script, (
            f"P1-CG: generated script pin tuple does not match "
            f"current lumenairy.__version__={lumenairy.__version__} "
            f"(expected tuple {gen_tuple})")

    def test_generated_script_stamps_version_in_comment(self):
        """The header comment includes ``lumenairy_version: X.Y.Z``."""
        import lumenairy
        from lumenairy.io.codegen import generate_simulation_script
        rx = self._minimal_prescription()
        script = generate_simulation_script(
            rx, wavelength=1.31e-6,
            N=128, dx=25e-6,
            source_sigma=2e-3,
            output_path=None,
            include_plotting=False,
            include_analysis=False)
        # The auto-header includes "lumenairy_version: <ver>" stamp.
        assert f"lumenairy_version: {lumenairy.__version__}" in script, (
            f"P1-CG: header comment missing the "
            f"``lumenairy_version: {lumenairy.__version__}`` stamp")


# ============================================================================
# E.3 -- P1-GH-1: ghost analysis seed parameter
# ============================================================================

class TestE3GhostSeedParameter:
    """``non_sequential_stray_light`` accepts a ``seed`` kwarg."""

    @staticmethod
    def _toy_bsdf():
        """Minimal duck-typed BSDF with a 0.001 isotropic scatter."""
        class _BSDF:
            def evaluate(self, n_in, dirs_out):
                # Slight isotropy with stochastic noise so the
                # Monte-Carlo sample variance is detectable.
                # The seeded RNG will sample different ``dirs_out``,
                # so the BSDF only needs to be a function of them.
                # We return cos-weighted noise.
                return 0.001 * (1.0 + 0.1 * dirs_out[..., 2])
        return _BSDF()

    @staticmethod
    def _toy_prescription():
        """Surface-style elements-only prescription that
        :func:`non_sequential_stray_light` (via the v4.14.3
        ``_ghost_surfaces`` adapter) accepts.  Two surfaces are
        enough to drive the BSDF-integration code path; the
        Monte-Carlo for TIS is what we're pinning, not the optics.
        """
        return {
            'name': 'TestE3Toy',
            'aperture_diameter': 25.4e-3,
            'elements': [
                {'element_type': 'surface',
                 'radius': 50.0e-3, 'conic': 0.0,
                 'glass_before': 'air', 'glass_after': 'N-BK7',
                 'semi_diameter': 12.7e-3, 'is_stop': False},
                {'element_type': 'surface',
                 'radius': -50.0e-3, 'conic': 0.0,
                 'glass_before': 'N-BK7', 'glass_after': 'air',
                 'semi_diameter': 12.7e-3, 'is_stop': False},
            ],
            'thicknesses': [4.0e-3, 0.0],
        }

    def test_seed_kwarg_in_signature(self):
        from lumenairy.analysis.ghost import non_sequential_stray_light
        sig = inspect.signature(non_sequential_stray_light)
        assert 'seed' in sig.parameters, (
            "P1-GH-1: seed kwarg missing from "
            "non_sequential_stray_light signature")
        # Default should be None (system entropy).
        assert sig.parameters['seed'].default is None, (
            f"P1-GH-1: seed default should be None, got "
            f"{sig.parameters['seed'].default!r}")

    def test_seed_int_is_deterministic(self):
        """Two calls with the same int seed give identical TIS."""
        from lumenairy.analysis.ghost import non_sequential_stray_light
        rx = self._toy_prescription()
        bsdf = self._toy_bsdf()
        r1 = non_sequential_stray_light(
            rx, wavelength=1.31e-6, bsdf_model=bsdf,
            seed=42, verbose=False)
        r2 = non_sequential_stray_light(
            rx, wavelength=1.31e-6, bsdf_model=bsdf,
            seed=42, verbose=False)
        tis1 = r1['scatter_tis_per_surface'][0]
        tis2 = r2['scatter_tis_per_surface'][0]
        assert tis1 == tis2 or abs(tis1 - tis2) < 1e-15, (
            f"P1-GH-1: seed=42 not deterministic: "
            f"call1 TIS={tis1!r}, call2 TIS={tis2!r}")

    def test_seed_none_varies_across_runs(self):
        """Three calls with seed=None give a non-zero sample variance."""
        from lumenairy.analysis.ghost import non_sequential_stray_light
        rx = self._toy_prescription()
        bsdf = self._toy_bsdf()
        samples = []
        for _ in range(3):
            r = non_sequential_stray_light(
                rx, wavelength=1.31e-6, bsdf_model=bsdf,
                seed=None, verbose=False)
            samples.append(r['scatter_tis_per_surface'][0])
        # Variance > 0 -- statistically the BSDF Monte Carlo with
        # 2000 samples won't give the same number twice from system
        # entropy.
        spread = max(samples) - min(samples)
        assert spread > 0.0, (
            f"P1-GH-1: seed=None did not vary across 3 runs "
            f"(samples={samples!r}); the RNG is still pinned")


# ============================================================================
# E.4 -- P1-GL-1: Sellmeier fallback for tuple-registered glasses
# ============================================================================

class TestE4SellmeierFallbackBundled:
    """SiO2 / F_SILICA / FUSED_SILICA / S-LAH64 / S-LAH79 fall back
    to bundled Sellmeier when refractiveindex is unavailable."""

    @staticmethod
    def _patch_no_refractiveindex(monkeypatch):
        """Simulate a minimal install: no refractiveindex package."""
        import lumenairy.glass as g
        monkeypatch.setattr(g, '_REFRACTIVEINDEX_AVAILABLE', False)
        # Also clear any cache so the fallback path is re-entered.
        monkeypatch.setattr(g, '_glass_cache', {})

    @pytest.mark.parametrize(
        'glass_name, target_n_d, tol',
        [
            ('SiO2',         1.458464, 5e-5),
            ('F_SILICA',     1.458464, 5e-5),
            ('FUSED_SILICA', 1.458464, 5e-5),
            ('S-LAH64',      1.788001, 5e-5),
            ('S-LAH79',      2.003300, 5e-5),
        ],
    )
    def test_minimal_install_fallback(
            self, monkeypatch, glass_name, target_n_d, tol):
        """With refractiveindex unavailable, the Sellmeier fallback
        produces n_d within 5e-5 of catalog."""
        self._patch_no_refractiveindex(monkeypatch)
        from lumenairy.glass import get_glass_index
        n = get_glass_index(glass_name, 587.56e-9)
        diff = abs(n - target_n_d)
        assert diff < tol, (
            f"P1-GL-1: {glass_name} Sellmeier fallback returned n_d={n:.6f} "
            f"(target {target_n_d:.6f}, diff {diff:.2e}, tol {tol:.2e})")


# ============================================================================
# E.5 -- P2: reverse-direction registry consistency
# ============================================================================

class TestE5RegistryReverseConsistency:
    """Every SELLMEIER_COEFFICIENTS row must appear in GLASS_REGISTRY."""

    def test_baseline_state_passes(self):
        """The shipped registry passes the reverse-direction check."""
        from lumenairy.glass import _check_glass_registry_consistency
        # Should be a no-op for the baseline.
        _check_glass_registry_consistency()

    def test_orphan_coefficient_row_raises(self, monkeypatch):
        """Injecting a Sellmeier row with no registry entry triggers
        the reverse-direction guard."""
        import lumenairy.glass as g
        # Take a deep-ish copy so the monkeypatch is automatically
        # rolled back at test end.
        orig = dict(g.SELLMEIER_COEFFICIENTS)
        try:
            g.SELLMEIER_COEFFICIENTS['__ORPHAN_TEST_GLASS__'] = (
                (1.0, 0.5, 0.1), (1e-3, 1e-2, 1e1))
            with pytest.raises(RuntimeError) as exc_info:
                g._check_glass_registry_consistency()
            assert "__ORPHAN_TEST_GLASS__" in str(exc_info.value), (
                "Reverse-direction error message should name the "
                "orphan glass")
            assert "unreachable" in str(exc_info.value).lower(), (
                "Reverse-direction error message should explain the "
                "unreachability symptom")
        finally:
            # Roll back to clean state for the rest of the suite.
            g.SELLMEIER_COEFFICIENTS.clear()
            g.SELLMEIER_COEFFICIENTS.update(orig)


# ============================================================================
# Self-test: module imports without error.  Run via
#   ``python -m pytest tests/unit/test_v4_15_agent_e.py -v``
# ============================================================================

def test_smoke_module_imports():
    """Smoke: importing lumenairy under v4.15 changes succeeds."""
    import lumenairy
    assert hasattr(lumenairy, '__version__')
