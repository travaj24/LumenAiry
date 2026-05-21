"""Unit tests for the 4.7 input-validation helpers.

Covers:

* ``_validate_propagator_inputs`` -- catches bad ``(E, z, lambda, dx)``
  signatures with clear error messages.
* ``validate_prescription`` -- catches malformed prescription dicts.
* Glass-registry enhancements: callable entries, Sellmeier fallback,
  ``list_glasses`` / ``search_glasses`` helpers, typo suggestions.
* Field analyses: dataclass returns + dict-style back-compat.
* ``apply_real_lens`` trio: ``dy=None`` kwarg present.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

import lumenairy as la
from lumenairy.propagators.propagation import _validate_propagator_inputs

# ---------------------------------------------------------------------------
# A.1 -- _validate_propagator_inputs
# ---------------------------------------------------------------------------

class TestValidatePropagatorInputs:

    def test_valid_call_returns_none(self):
        E = np.zeros((64, 64), dtype=complex)
        # Should not raise
        _validate_propagator_inputs(E, 0.01, 1.31e-6, 1e-6)

    @pytest.mark.parametrize('lam', [0.0, -1e-6, float('nan')])
    def test_bad_wavelength_raises(self, lam):
        E = np.zeros((64, 64), dtype=complex)
        with pytest.raises(ValueError, match='wavelength'):
            _validate_propagator_inputs(E, 0.01, lam, 1e-6)

    def test_wavelength_in_microns_raises(self):
        E = np.zeros((64, 64), dtype=complex)
        with pytest.raises(ValueError, match='suspicious'):
            _validate_propagator_inputs(E, 0.01, 1.31, 1e-6)  # forgot e-6

    @pytest.mark.parametrize('dx', [0.0, -2e-6, float('nan')])
    def test_bad_dx_raises(self, dx):
        E = np.zeros((64, 64), dtype=complex)
        with pytest.raises(ValueError, match='dx'):
            _validate_propagator_inputs(E, 0.01, 1.31e-6, dx)

    def test_dx_in_meters_raises(self):
        E = np.zeros((64, 64), dtype=complex)
        with pytest.raises(ValueError, match='suspicious'):
            _validate_propagator_inputs(E, 0.01, 1.31e-6, 2.0)

    def test_three_d_input_raises(self):
        E = np.zeros((1, 64, 64), dtype=complex)
        with pytest.raises(ValueError, match='2-D'):
            _validate_propagator_inputs(E, 0.01, 1.31e-6, 1e-6)

    def test_small_input_raises(self):
        E = np.zeros((2, 2), dtype=complex)
        with pytest.raises(ValueError, match='too small'):
            _validate_propagator_inputs(E, 0.01, 1.31e-6, 1e-6)

    def test_nan_z_raises(self):
        E = np.zeros((64, 64), dtype=complex)
        with pytest.raises(ValueError, match='z'):
            _validate_propagator_inputs(E, float('nan'), 1.31e-6, 1e-6)

    def test_function_name_in_error(self):
        E = np.zeros((64, 64), dtype=complex)
        try:
            _validate_propagator_inputs(E, 0.01, 0.0, 1e-6,
                                        fn_name='my_propagator')
        except ValueError as e:
            assert 'my_propagator' in str(e)


class TestPropagatorValidationIntegration:
    """The validator is wired into every propagator entry point."""

    def test_asm_rejects_zero_wavelength(self):
        E = np.zeros((64, 64), dtype=complex)
        with pytest.raises(ValueError):
            la.angular_spectrum_propagate(E, 0.01, 0.0, 1e-6)

    def test_fresnel_rejects_negative_dx(self):
        E = np.zeros((64, 64), dtype=complex)
        with pytest.raises(ValueError):
            la.fresnel_propagate(E, 0.01, 1.31e-6, -1.0)

    def test_fraunhofer_rejects_unit_confused_wavelength(self):
        E = np.zeros((64, 64), dtype=complex)
        with pytest.raises(ValueError):
            la.fraunhofer_propagate(E, 0.01, 1.31, 1e-6)


# ---------------------------------------------------------------------------
# A.2 -- validate_prescription
# ---------------------------------------------------------------------------

class TestValidatePrescription:

    def test_valid_singlet(self, singlet_prescription):
        # Should not raise
        la.validate_prescription(singlet_prescription)

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match="'surfaces'"):
            la.validate_prescription({})

    def test_missing_thicknesses(self, singlet_prescription):
        p = dict(singlet_prescription)
        del p['thicknesses']
        with pytest.raises(ValueError, match='thicknesses'):
            la.validate_prescription(p)

    def test_thickness_length_mismatch(self, singlet_prescription):
        p = dict(singlet_prescription)
        p['thicknesses'] = []
        with pytest.raises(ValueError, match='length mismatch'):
            la.validate_prescription(p)

    def test_nan_radius_caught(self, singlet_prescription):
        p = dict(singlet_prescription)
        bad_surf = dict(p['surfaces'][0], radius=float('nan'))
        p = dict(p, surfaces=[bad_surf, p['surfaces'][1]])
        with pytest.raises(ValueError, match='radius'):
            la.validate_prescription(p)

    def test_strict_false_returns_issues_list(self):
        issues = la.validate_prescription({}, strict=False)
        assert isinstance(issues, list)
        assert len(issues) > 0
        assert all(isinstance(t, tuple) and len(t) == 2 for t in issues)

    def test_surfaces_from_prescription_validates(self):
        with pytest.raises(ValueError):
            la.surfaces_from_prescription({})

    def test_negative_aperture_raises(self, singlet_prescription):
        p = dict(singlet_prescription, aperture_diameter=-1.0)
        with pytest.raises(ValueError, match='aperture_diameter'):
            la.validate_prescription(p)


# ---------------------------------------------------------------------------
# B.3 -- dy=None on apply_real_lens trio
# ---------------------------------------------------------------------------

class TestRealLensTrioDyKwarg:

    @pytest.mark.parametrize('fn', [
        la.apply_real_lens,
        la.apply_real_lens_traced,
        la.apply_real_lens_maslov,
    ])
    def test_has_dy_kwarg(self, fn):
        sig = inspect.signature(fn)
        assert 'dy' in sig.parameters
        assert sig.parameters['dy'].default is None

    def test_apply_real_lens_anamorphic_runs(self, singlet_prescription,
                                              wavelength_m):
        # 64x64 plane wave on dx=5um, dy=4um
        E = np.ones((64, 64), dtype=complex)
        out = la.apply_real_lens(E, prescription=singlet_prescription, wavelength=wavelength_m,
                                  dx=5e-6, dy=4e-6)
        assert out.shape == E.shape
        assert np.isfinite(np.abs(out)).all()

    def test_apply_real_lens_traced_rejects_dy_ne_dx(self,
                                                     singlet_prescription,
                                                     wavelength_m):
        E = np.ones((64, 64), dtype=complex)
        with pytest.raises(ValueError, match='square pixels'):
            la.apply_real_lens_traced(E, prescription=singlet_prescription,
                                       wavelength=wavelength_m, dx=5e-6, dy=4e-6)


# ---------------------------------------------------------------------------
# B.9 -- dataclass returns + dict-style back-compat
# ---------------------------------------------------------------------------

class TestFieldAnalysisDataclasses:

    def test_distortion_grid_dataclass_attrs(self, singlet_prescription,
                                              wavelength_m):
        g = la.distortion_grid(singlet_prescription, wavelength_m,
                                max_field_deg=1.0, n_grid=3)
        assert isinstance(g, la.DistortionGrid)
        assert g.actual_x.shape == (3, 3)
        # dict-style access
        assert g['actual_x'].shape == (3, 3)
        assert 'actual_x' in g
        assert 'nope' not in g

    def test_footprint_dataclass_attrs(self, singlet_prescription,
                                        wavelength_m):
        fp = la.footprint_per_surface(singlet_prescription, wavelength_m,
                                       semi_aperture=2e-3,
                                       fields_deg=(0.0,),
                                       num_rings=2, rays_per_ring=4)
        assert isinstance(fp, list)
        assert isinstance(fp[0], la.SurfaceFootprint)
        # dict-style nested access still works
        x = fp[0]['fields'][0]['x']
        assert x.ndim == 1

    def test_spot_diagram_dataclass_attrs(self, singlet_prescription,
                                           wavelength_m):
        sd = la.spot_diagram_vs_field(singlet_prescription, wavelength_m,
                                       fields_deg=[0.0, 1.0],
                                       semi_aperture=2e-3,
                                       num_rings=2, rays_per_ring=4)
        assert all(isinstance(s, la.SpotDiagramField) for s in sd)
        assert sd[0].field_deg == 0.0
        assert sd[0]['field_deg'] == 0.0


# ---------------------------------------------------------------------------
# I -- Glass registry callable + bundled Sellmeier + discovery helpers
# ---------------------------------------------------------------------------

class TestGlassRegistryEnhancements:

    def test_list_glasses_returns_sorted_list(self):
        g = la.list_glasses()
        assert isinstance(g, list)
        assert g == sorted(g)
        assert 'N-BK7' in g
        # 4.7 added ~30 new bundled glasses
        assert len(g) >= 40

    def test_search_glasses_substring(self):
        sf = la.search_glasses('LASF')
        assert all('LASF' in name for name in sf)
        assert len(sf) >= 5  # N-LASF31A, 40, 41, 44, 45, 46A, 46B, N-LASF9

    def test_callable_entry(self):
        # Add a custom dispersion callable
        la.GLASS_REGISTRY['_TEST_CB'] = lambda wl: 1.5 + 1e6 * wl
        try:
            n = la.get_glass_index('_TEST_CB', 1.31e-6)
            assert abs(n - (1.5 + 1.31)) < 1e-12
        finally:
            del la.GLASS_REGISTRY['_TEST_CB']

    def test_callable_returning_complex(self):
        la.GLASS_REGISTRY['_TEST_ABS'] = lambda wl: complex(1.5, 0.01)
        try:
            nc = la.get_glass_index_complex('_TEST_ABS', 1.31e-6)
            assert nc == complex(1.5, 0.01)
            # The real-only accessor strips the imaginary part
            n_re = la.get_glass_index('_TEST_ABS', 1.31e-6)
            assert n_re == 1.5
        finally:
            del la.GLASS_REGISTRY['_TEST_ABS']

    def test_bundled_sellmeier_matches_database(self):
        """N-BK7 is also in SELLMEIER_COEFFICIENTS; compare against
        the live refractiveindex.info lookup at the d-line."""
        n_db = la.get_glass_index('N-BK7', 587.6e-9)
        # Synthesize a Sellmeier-only entry pointing at the same coeffs
        la.GLASS_REGISTRY['_TEST_BK7_SM'] = '__sellmeier__'
        la.SELLMEIER_COEFFICIENTS['_TEST_BK7_SM'] = (
            la.SELLMEIER_COEFFICIENTS['N-BK7'])
        try:
            n_sm = la.get_glass_index('_TEST_BK7_SM', 587.6e-9)
            assert abs(n_sm - n_db) < 1e-4
        finally:
            del la.GLASS_REGISTRY['_TEST_BK7_SM']
            del la.SELLMEIER_COEFFICIENTS['_TEST_BK7_SM']

    def test_typo_suggests_close_match(self):
        with pytest.raises(ValueError, match='Did you mean'):
            la.get_glass_index('N-BL7', 1.31e-6)

    def test_sf57_bundled_sellmeier_evaluates(self):
        # N-SF57 is bundled-only (no refractiveindex.info path)
        n = la.get_glass_index('N-SF57', 587.6e-9)
        # SF57 d-line is around 1.847; allow generous tolerance for
        # the 3-term Sellmeier evaluation
        assert 1.8 < n < 1.9


# ---------------------------------------------------------------------------
# B.11 / C.1 -- __all__ + module rename
# ---------------------------------------------------------------------------

class TestPackagingHygiene:

    def test_analysis_submodules_define_all(self):
        import lumenairy.analysis.coherence
        import lumenairy.analysis.core
        import lumenairy.analysis.detector
        import lumenairy.analysis.ghost
        import lumenairy.analysis.image_plane_wfe
        import lumenairy.analysis.interferometry
        import lumenairy.analysis.phase_retrieval
        import lumenairy.analysis.plotting
        import lumenairy.analysis.through_focus
        for mod in (lumenairy.analysis.core,
                    lumenairy.analysis.coherence,
                    lumenairy.analysis.detector,
                    lumenairy.analysis.ghost,
                    lumenairy.analysis.image_plane_wfe,
                    lumenairy.analysis.through_focus,
                    lumenairy.analysis.interferometry,
                    lumenairy.analysis.phase_retrieval,
                    lumenairy.analysis.plotting):
            assert hasattr(mod, '__all__'), f'{mod.__name__} missing __all__'
            for name in mod.__all__:
                assert hasattr(mod, name), \
                    f'{mod.__name__}.__all__ lists {name!r} but module has none'

    def test_analysis_dot_analysis_shim_removed_in_v5_0(self):
        """v5.0 (honest break): the v4.7 back-compat shim
        ``lumenairy.analysis.analysis`` was removed.  Importing from
        the old path must raise ``ImportError`` / ``ModuleNotFoundError``
        with a clear pointer to the new home."""
        import pytest
        with pytest.raises((ImportError, ModuleNotFoundError)):
            from lumenairy.analysis.analysis import beam_centroid  # noqa: F401

    def test_validate_prescription_exported_at_top_level(self):
        assert hasattr(la, 'validate_prescription')
        assert hasattr(la, 'list_glasses')
        assert hasattr(la, 'search_glasses')
        assert hasattr(la, 'SELLMEIER_COEFFICIENTS')


# ---------------------------------------------------------------------------
# v5.0.1 -- shim-removal anti-regression pin parity
#
# The v5.0 audit (AUDIT_V5_0_0_2026_05_20.md, P2-NEW-F2-1) found that only
# one of five v5.0 honest-break closures had an anti-regression pin: the
# ``test_analysis_dot_analysis_shim_removed_in_v5_0`` test above.  The
# remaining four removals (``lumenairy.ao``, ``lumenairy.io.hdf5``,
# top-level ``lumenairy.system``, JAX aperture legacy schema,
# ``apply_detector(cosmic_ray_rate=)`` kwarg) had no pin -- a v5.1
# maintainer could accidentally re-add any of them without a test failure.
#
# This class closes that sibling-gap with one pin per removal, each
# modeled structurally on the existing analysis-shim pin.  The
# ``match=`` patterns lock in the migration recipe so the error message
# itself is regression-tested.
# audit closure: P2-NEW-F2-1 (shim-removal anti-regression pin asymmetry)
# ---------------------------------------------------------------------------

class TestV5_0_ShimRemovalAntiRegression:
    """Pins for the v5.0 honest-break closures that lacked
    anti-regression coverage at v5.0 ship.

    Each test asserts the removal still raises the documented error
    class with the documented migration-recipe text.  If a future
    maintainer accidentally re-introduces a removed shim or kwarg,
    the corresponding test fails immediately.
    """

    # v5.1.1 (audit P2-NEW-V2): content-lock helper.
    # The bare ``ImportError`` pin only verifies the OLD path fails;
    # it does NOT verify the Migration Guide still documents the new
    # path.  A future maintainer could silently delete the
    # corresponding section from Migration-Guide.md and these pins
    # would still pass -- callers would be left with a raise and no
    # migration recipe.  Each test below now additionally reads
    # Migration-Guide.md and asserts the removal line + the new
    # import path are both present.  Parallel to the V11 doc-
    # consistency walker but anchored inline at the source of the
    # break, so a maintainer editing the shim pin sees the doc
    # dependency directly.
    #
    # v5.2 (AUDIT_V5_1_1 Part 6 -- P2 tighten Migration-Guide locks):
    # the v5.1.1 form asserted ``shim-name``
    # AND ``**Removed.**`` as two independent global substrings.  The
    # file has 5 ``**Removed.**`` instances; F1 demonstrated that
    # deleting one removal section while keeping the shim-name token
    # alive elsewhere (a CHANGELOG quote, a recap table) would keep
    # the pin green.  v5.2 anchors ``**Removed.**`` to within
    # ``_MIGRATION_REMOVED_WINDOW_LINES`` lines AFTER the shim-name
    # match (the recipe pattern is: shim-name on bullet line, then
    # ``**Removed.**`` on the same line or the next 1-2 lines).
    _MIGRATION_REMOVED_WINDOW_LINES = 3

    @staticmethod
    def _migration_guide_text():
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[2]
        guide = repo_root / 'Migration-Guide.md'
        assert guide.is_file(), (
            f'Migration-Guide.md not found at {guide}; the v5.0 '
            f'shim-removal recipes have nowhere to live.')
        return guide.read_text(encoding='utf-8')

    @classmethod
    def _assert_removal_recipe_anchored(cls, guide_text, shim_name,
                                        new_import, kind='shim'):
        """Verify ``shim_name`` appears in Migration-Guide.md with
        ``**Removed.**`` within ``_MIGRATION_REMOVED_WINDOW_LINES``
        lines AND ``new_import`` appears within
        ``_MIGRATION_REMOVED_WINDOW_LINES * 3`` lines of the same
        match (slightly larger window because the new-import recipe
        usually sits 1-4 lines below the removal token).

        Raises AssertionError naming the failure mode + the line
        numbers where the partial match was found so a maintainer
        gets an actionable diagnostic.
        """
        lines = guide_text.splitlines()
        removed_window = cls._MIGRATION_REMOVED_WINDOW_LINES
        recipe_window = removed_window * 3

        anchor_hits = [
            i for i, ln in enumerate(lines) if shim_name in ln
        ]
        assert anchor_hits, (
            f'Migration-Guide.md does not mention `{shim_name}` '
            f'anywhere; the {kind}-removal recipe is missing.  '
            f'A maintainer must restore the removal entry before '
            f'this pin can pass.')

        for idx in anchor_hits:
            window = lines[idx: idx + 1 + removed_window]
            if any('**Removed.**' in ln for ln in window):
                # Found removal token close to the anchor; now check
                # the new-import recipe is within the larger window.
                recipe_band = lines[idx: idx + 1 + recipe_window]
                if any(new_import in ln for ln in recipe_band):
                    return
                # removal token near anchor but new-import isn't.
                raise AssertionError(
                    f'Migration-Guide.md anchors `{shim_name}` + '
                    f'``**Removed.**`` near line {idx + 1} but the '
                    f'new-import recipe ``{new_import}`` is missing '
                    f'within {recipe_window} lines.  The {kind} '
                    f'removal entry exists but its migration recipe '
                    f'has been deleted or moved out of band.')

        raise AssertionError(
            f'Migration-Guide.md mentions `{shim_name}` (lines '
            f'{[i + 1 for i in anchor_hits]}) but ``**Removed.**`` '
            f'is not within {removed_window} lines of any of them.  '
            f'The {kind} appears in the guide as prose (CHANGELOG '
            f'quote, recap table, etc) but its dedicated removal '
            f'section is missing.')

    def test_lumenairy_ao_shim_removed_in_v5_0(self):
        """v5.0 (honest break): the v4.x back-compat shim
        ``lumenairy.ao`` was removed.  Importing
        ``DeformableMirror`` from the old top-level path must raise
        ``ImportError`` / ``ModuleNotFoundError``; callers must use
        ``lumenairy.analysis.ao`` (or the top-level re-export).

        v5.1.1 + v5.2: also content-lock the Migration-Guide entry,
        with v5.2 anchoring ``**Removed.**`` near the shim name.
        """
        # audit closure: P2-NEW-F2-1 (shim-removal anti-regression pin asymmetry)
        with pytest.raises((ImportError, ModuleNotFoundError)):
            from lumenairy.ao import DeformableMirror  # noqa: F401
        guide = self._migration_guide_text()
        self._assert_removal_recipe_anchored(
            guide, shim_name='`lumenairy.ao`',
            new_import='lumenairy.analysis.ao')

    def test_lumenairy_io_hdf5_shim_removed_in_v5_0(self):
        """v5.0 (honest break): the v4.x back-compat shim
        ``lumenairy.io.hdf5`` was removed.  Importing
        ``save_field_h5`` from the old path must raise
        ``ImportError`` / ``ModuleNotFoundError``; callers must use
        ``lumenairy.io.storage`` (or the top-level re-export).

        v5.1.1 + v5.2: also content-lock the Migration-Guide entry,
        with v5.2 anchoring ``**Removed.**`` near the shim name.
        """
        # audit closure: P2-NEW-F2-1 (shim-removal anti-regression pin asymmetry)
        with pytest.raises((ImportError, ModuleNotFoundError)):
            from lumenairy.io.hdf5 import save_field_h5  # noqa: F401
        guide = self._migration_guide_text()
        self._assert_removal_recipe_anchored(
            guide, shim_name='`lumenairy.io.hdf5`',
            new_import='lumenairy.io.storage')

    def test_lumenairy_system_top_level_removed_in_v5_0(self):
        """v5.0 (honest break): the top-level ``lumenairy.system``
        module was moved to ``lumenairy.propagators.system``.  The
        old top-level path must raise ``ImportError`` /
        ``ModuleNotFoundError``; the canonical
        ``lumenairy.propagate_through_system`` top-level facade
        still works (verified separately via the public-API smoke
        test).

        v5.1.1 + v5.2: also content-lock the move recipe; the
        ``system.py`` move is a heading-driven section in
        Migration-Guide.md rather than a ``**Removed.**`` bullet,
        so the v5.2 anchor uses the heading + import-line proximity
        check directly (no general-helper call).
        """
        # audit closure: P2-NEW-F2-1 (shim-removal anti-regression pin asymmetry)
        with pytest.raises((ImportError, ModuleNotFoundError)):
            from lumenairy.system import propagate_through_system  # noqa: F401
        guide = self._migration_guide_text()
        lines = guide.splitlines()
        heading_hits = [
            i for i, ln in enumerate(lines)
            if 'lumenairy/system.py' in ln
            and 'lumenairy/propagators/system.py' in ln
        ]
        assert heading_hits, (
            'Migration-Guide.md is missing the '
            '``lumenairy/system.py -> lumenairy/propagators/system.py`` '
            'move heading; the raise is wired but callers have no '
            'documented migration recipe.')
        # The recipe sits ~21 lines below the section heading
        # (prose + code-block fences); allow 30-line proximity.
        window = self._MIGRATION_REMOVED_WINDOW_LINES * 10
        recipe = 'from lumenairy.propagators.system import propagate_through_system'
        found_recipe_near_heading = any(
            any(recipe in ln for ln in lines[i: i + 1 + window])
            for i in heading_hits)
        assert found_recipe_near_heading, (
            f'Migration-Guide.md has the ``system.py`` move heading '
            f'(lines {[i + 1 for i in heading_hits]}) but the recipe '
            f'``{recipe}`` is not within {window} lines of it.  The '
            f'heading exists but the migration code block has been '
            f'moved or deleted.')

    def test_jax_aperture_legacy_schema_removed_in_v5_0(self):
        """v5.0 (honest break): the legacy pre-v4.12 JAX-only
        aperture schema (``params={'radius': ...}``) was deprecated
        in v4.12 and is REMOVED in v5.0.  ``propagate_through_system_jax``
        must raise ``ValueError`` whose message names both ``legacy``
        and ``removed in v5.0`` so the migration recipe is regression-
        tested alongside the raise."""
        # audit closure: P2-NEW-F2-1 (shim-removal anti-regression pin asymmetry)
        jax = pytest.importorskip('jax')  # noqa: F841
        from lumenairy.propagators.system import propagate_through_system_jax
        N, dx, wl = 64, 5e-6, 1.55e-6
        E = np.ones((N, N), dtype=np.complex64)
        elements = [
            {'type': 'aperture', 'shape': 'circular',
             'params': {'radius': 50e-6}},
        ]
        with pytest.raises(ValueError, match=r"legacy.*removed in v5\.0"):
            propagate_through_system_jax(E, elements, wl, dx)

    def test_apply_detector_cosmic_ray_rate_kwarg_removed_in_v5_0(self):
        """v5.0 (honest break): the v4.9-deprecated
        ``cosmic_ray_rate=`` kwarg on ``apply_detector`` was removed
        in v5.0.  Callers must pass ``cosmic_ray_rate_per_m2_per_s=``
        instead.  Passing the legacy kwarg must raise ``TypeError``
        (unexpected keyword argument) -- this is the kwarg-removal
        analog of a shim removal."""
        # audit closure: P2-NEW-F2-1 (shim-removal anti-regression pin asymmetry)
        E = np.ones((64, 64), dtype=np.complex128)
        with pytest.raises(TypeError, match="cosmic_ray_rate"):
            la.apply_detector(
                E, dx_field=5e-6, pixel_pitch=5e-6,
                exposure_time=1.0, cosmic_ray_rate=5.0,
                seed=0,
            )
