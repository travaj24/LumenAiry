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
        import lumenairy.analysis.core
        import lumenairy.analysis.coherence
        import lumenairy.analysis.detector
        import lumenairy.analysis.ghost
        import lumenairy.analysis.image_plane_wfe
        import lumenairy.analysis.through_focus
        import lumenairy.analysis.interferometry
        import lumenairy.analysis.phase_retrieval
        import lumenairy.analysis.plotting
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
