"""v5.1.0 Agent B -- regression tests for the lumenairy.raytrace.core split.

The v5.1.0 release split the 4443-line ``lumenairy/raytrace/core.py``
into seven submodules:

* ``lumenairy/raytrace/surface.py``        -- data classes + sag / normal
* ``lumenairy/raytrace/intersection.py``   -- intersection / refract /
                                              reflect / transfer / coord-break
* ``lumenairy/raytrace/trace.py``          -- trace + prescription conversion
                                              + ray generators + DOE bridge
* ``lumenairy/raytrace/world_trace.py``    -- world-frame trace + helpers
* ``lumenairy/raytrace/seidel.py``         -- paraxial + ABCD + Lens/Pupil /
                                              first-order + Seidel
* ``lumenairy/raytrace/ray_fan.py``        -- spot / ray-fan / OPD /
                                              refocus / through-focus
* ``lumenairy/raytrace/layout.py``         -- trace_summary +
                                              prescription_summary

Public-API contract: every name that pre-v5.1.0 was importable from
``lumenairy.raytrace.core`` (or via ``lumenairy.raytrace``) must
continue to resolve identically.  This module is the anti-regression
pin for that promise.

These tests are pure import-shape / object-identity checks; they
intentionally don't exercise the numerics.  The raytrace numeric
behaviour is covered by ``test_raytrace.py``, ``test_audit_fixes_v4_*``,
etc.; if any of those start failing after the v5.1.0 split, the split
isn't bit-for-bit and the release must be held back.
"""

from __future__ import annotations

import numpy as np
import pytest

# Names that pre-v5.1.0 ``lumenairy/raytrace/__init__.py`` explicitly
# imported from ``.core`` and re-exposed (the public surface).  The
# split MUST preserve every one of these as importable from
# ``lumenairy.raytrace.core`` AND ``lumenairy.raytrace``.
_PUBLIC_NAMES_FROM_CORE = [
    # Data structures
    'RayBundle', 'Surface', 'TraceResult',
    # Trace engine
    'trace', 'trace_world',
    # Prescription conversion
    'surfaces_from_prescription', 'validate_prescription',
    # Ray generators
    'make_ray', 'make_fan', 'make_ring', 'make_grid', 'make_rings',
    # DOE / grating
    'apply_doe_phase_traced',
    # High-level convenience
    'trace_prescription',
    # ABCD
    'system_abcd', 'system_abcd_prescription',
    # Seidel
    'seidel_coefficients', 'seidel_prescription',
    # Spot / fan / OPD
    'spot_rms', 'spot_geo_radius', 'spot_diagram',
    'ray_fan_data', 'ray_fan_plot', 'ray_fan_plot_prescription',
    'opd_fan_data', 'ray_fan_data_world', 'opd_fan_data_world',
    # Through-focus / refocus
    'through_focus_rms', 'refocus',
    # Stop / pupils / first-order
    'find_stop', 'compute_pupils', 'lens_abcd', 'find_lenses',
    'LensInfo', 'PupilInfo', 'FirstOrderData', 'first_order_data',
    # Error codes
    'RAY_OK', 'RAY_TIR', 'RAY_APERTURE',
    'RAY_MISSED_SURFACE', 'RAY_NAN', 'RAY_EVANESCENT',
    # Misc / utility
    'find_paraxial_focus', 'trace_summary', 'prescription_summary',
    # system.py element-list bridge
    'surfaces_from_elements', 'raytrace_system',
    # Private helper exposed for cross-subpackage use
    '_make_bundle',
]


class TestSplitPublicSurfaceFromCore:
    """Every legacy public name still imports from
    ``lumenairy.raytrace.core``."""

    @pytest.mark.parametrize('name', _PUBLIC_NAMES_FROM_CORE)
    def test_name_resolves_from_core(self, name):
        """``from lumenairy.raytrace.core import <name>`` succeeds."""
        import lumenairy.raytrace.core as core
        assert hasattr(core, name), (
            f"v5.1.0 split regression: {name!r} is no longer "
            f"importable from lumenairy.raytrace.core")

    @pytest.mark.parametrize('name', _PUBLIC_NAMES_FROM_CORE)
    def test_name_resolves_from_raytrace_package(self, name):
        """``from lumenairy.raytrace import <name>`` succeeds (the
        public package namespace forwards to core)."""
        import lumenairy.raytrace as rt
        assert hasattr(rt, name), (
            f"v5.1.0 split regression: {name!r} is no longer "
            f"importable from lumenairy.raytrace")

    def test_star_import_from_core_picks_up_every_legacy_name(self):
        """``from lumenairy.raytrace.core import *`` keeps working
        for every legacy public name.  Pre-v5.1.0 ``core.py`` had
        no ``__all__`` so star-import fell back to "every module-
        level non-underscore name" semantics; the v5.1.0 shell
        re-exports every submodule's ``__all__`` so the same
        star-import surface survives.  Verified by exec-ing the
        actual ``from ... import *`` statement in a controlled
        namespace and checking each legacy name landed."""
        ns: dict = {}
        exec('from lumenairy.raytrace.core import *', ns)
        missing = [n for n in _PUBLIC_NAMES_FROM_CORE
                   if n not in ns and not n.startswith('_')]
        assert missing == [], (
            f"v5.1.0 split regression: ``from "
            f"lumenairy.raytrace.core import *`` no longer "
            f"surfaces {missing}")


class TestSplitSubmoduleResidency:
    """Each public name resides in the expected new submodule.  This
    pins the topology of the split so a later refactor doesn't
    silently shuffle ownership."""

    def test_surface_classes_live_in_surface_module(self):
        from lumenairy.raytrace import surface
        for n in ('RayBundle', 'Surface', 'TraceResult',
                  'RAY_OK', 'RAY_TIR', 'RAY_APERTURE',
                  'RAY_MISSED_SURFACE', 'RAY_NAN', 'RAY_EVANESCENT'):
            assert hasattr(surface, n), \
                f"{n!r} should live in raytrace.surface"

    def test_intersection_helpers_live_in_intersection_module(self):
        from lumenairy.raytrace import intersection
        for n in ('_intersect_surface', '_refract', '_reflect',
                  '_transfer', '_apply_coord_break'):
            assert hasattr(intersection, n), \
                f"{n!r} should live in raytrace.intersection"

    def test_trace_engine_lives_in_trace_module(self):
        # NOTE: ``from lumenairy.raytrace import trace`` would resolve
        # to the *function* (the re-export shadows the submodule on
        # the package namespace by design -- the public API is the
        # function).  Use ``importlib`` to grab the actual submodule.
        import importlib
        trace_mod = importlib.import_module('lumenairy.raytrace.trace')
        for n in ('trace', 'surfaces_from_prescription',
                  'validate_prescription', 'find_stop',
                  'make_ray', 'make_fan', 'make_ring',
                  'make_grid', 'make_rings',
                  'apply_doe_phase_traced',
                  'trace_prescription',
                  'surfaces_from_elements', 'raytrace_system',
                  '_make_bundle'):
            assert hasattr(trace_mod, n), \
                f"{n!r} should live in raytrace.trace"

    def test_world_trace_lives_in_world_trace_module(self):
        from lumenairy.raytrace import world_trace
        for n in ('trace_world',
                  '_world_to_local_state', '_local_to_world_state'):
            assert hasattr(world_trace, n), \
                f"{n!r} should live in raytrace.world_trace"

    def test_seidel_paraxial_lives_in_seidel_module(self):
        from lumenairy.raytrace import seidel
        for n in ('system_abcd', 'system_abcd_prescription',
                  'seidel_coefficients', 'seidel_prescription',
                  'LensInfo', 'lens_abcd', 'find_lenses',
                  'PupilInfo', 'FirstOrderData', 'first_order_data',
                  'compute_pupils', 'find_paraxial_focus'):
            assert hasattr(seidel, n), \
                f"{n!r} should live in raytrace.seidel"

    def test_ray_fan_lives_in_ray_fan_module(self):
        from lumenairy.raytrace import ray_fan
        for n in ('spot_rms', 'spot_geo_radius', 'spot_diagram',
                  'ray_fan_data', 'ray_fan_data_world',
                  'ray_fan_plot', 'ray_fan_plot_prescription',
                  'opd_fan_data', 'opd_fan_data_world',
                  'refocus', 'through_focus_rms'):
            assert hasattr(ray_fan, n), \
                f"{n!r} should live in raytrace.ray_fan"

    def test_layout_summaries_live_in_layout_module(self):
        from lumenairy.raytrace import layout
        for n in ('trace_summary', 'prescription_summary'):
            assert hasattr(layout, n), \
                f"{n!r} should live in raytrace.layout"


class TestSplitObjectIdentity:
    """Each name resolves to the *same* Python object regardless of
    whether the caller imports it from ``raytrace.core`` or the new
    submodule.  Object identity (not just equality) catches the
    common "I accidentally duplicated the class definition" mistake
    after a split."""

    def _assert_identical(self, name, submod_path):
        import importlib
        core_obj = getattr(importlib.import_module(
            'lumenairy.raytrace.core'), name)
        sub_obj = getattr(importlib.import_module(submod_path), name)
        pkg_obj = getattr(importlib.import_module(
            'lumenairy.raytrace'), name)
        assert core_obj is sub_obj, (
            f"{name!r}: core re-export and submodule disagree; "
            f"split is not back-compat (duplicated definition).")
        assert core_obj is pkg_obj, (
            f"{name!r}: core re-export and package namespace "
            f"disagree; split is not back-compat.")

    @pytest.mark.parametrize('name,submod', [
        ('RayBundle', 'lumenairy.raytrace.surface'),
        ('Surface', 'lumenairy.raytrace.surface'),
        ('TraceResult', 'lumenairy.raytrace.surface'),
        ('RAY_OK', 'lumenairy.raytrace.surface'),
        ('trace', 'lumenairy.raytrace.trace'),
        ('trace_world', 'lumenairy.raytrace.world_trace'),
        ('surfaces_from_prescription', 'lumenairy.raytrace.trace'),
        ('validate_prescription', 'lumenairy.raytrace.trace'),
        ('find_stop', 'lumenairy.raytrace.trace'),
        ('make_ray', 'lumenairy.raytrace.trace'),
        ('make_fan', 'lumenairy.raytrace.trace'),
        ('make_rings', 'lumenairy.raytrace.trace'),
        ('_make_bundle', 'lumenairy.raytrace.trace'),
        ('apply_doe_phase_traced', 'lumenairy.raytrace.trace'),
        ('trace_prescription', 'lumenairy.raytrace.trace'),
        ('system_abcd', 'lumenairy.raytrace.seidel'),
        ('seidel_coefficients', 'lumenairy.raytrace.seidel'),
        ('LensInfo', 'lumenairy.raytrace.seidel'),
        ('PupilInfo', 'lumenairy.raytrace.seidel'),
        ('FirstOrderData', 'lumenairy.raytrace.seidel'),
        ('first_order_data', 'lumenairy.raytrace.seidel'),
        ('compute_pupils', 'lumenairy.raytrace.seidel'),
        ('find_paraxial_focus', 'lumenairy.raytrace.seidel'),
        ('spot_rms', 'lumenairy.raytrace.ray_fan'),
        ('ray_fan_data', 'lumenairy.raytrace.ray_fan'),
        ('opd_fan_data', 'lumenairy.raytrace.ray_fan'),
        ('refocus', 'lumenairy.raytrace.ray_fan'),
        ('through_focus_rms', 'lumenairy.raytrace.ray_fan'),
        ('trace_summary', 'lumenairy.raytrace.layout'),
        ('prescription_summary', 'lumenairy.raytrace.layout'),
        ('surfaces_from_elements', 'lumenairy.raytrace.trace'),
        ('raytrace_system', 'lumenairy.raytrace.trace'),
    ])
    def test_object_identity_across_paths(self, name, submod):
        self._assert_identical(name, submod)


class TestSplitSmokeTrace:
    """A minimal end-to-end trace exercising the cross-module
    pipeline.  If the split broke any wiring (e.g. ray_fan can't see
    surfaces_from_prescription, world_trace can't see _refract),
    this smoke test fails before the numeric audit suite even
    starts.  The numbers don't matter -- only that the chain
    completes."""

    def test_singlet_trace_through_split_engine(self):
        from lumenairy.raytrace.core import (
            Surface,
            make_ray,
            system_abcd,
            trace,
        )

        # Hand-built single-surface "lens" (one refracting interface).
        surfaces = [
            Surface(radius=0.05, conic=0.0,
                     semi_diameter=10e-3,
                     glass_before='air', glass_after='N-BK7',
                     thickness=5e-3),
            Surface(radius=-0.05, conic=0.0,
                     semi_diameter=10e-3,
                     glass_before='N-BK7', glass_after='air',
                     thickness=100e-3),
        ]
        ray = make_ray(0.0, 1e-3, 0.0, 0.0, wavelength=1.31e-6)
        result = trace(ray, surfaces, 1.31e-6)

        # The image-plane ray height differs from the input height
        # for any non-zero-power lens; sanity-check both that the
        # bundle survives and that the bundle landed somewhere
        # non-trivial.  We don't pin the exact numbers because the
        # split is mechanical -- only that the trace ran end-to-end.
        assert result.image_rays.n_rays == 1
        assert bool(result.image_rays.alive[0])
        assert np.isfinite(result.image_rays.y[0])

        # ABCD should be a 2x2 numerically finite matrix.
        M, efl, bfl, ffl = system_abcd(surfaces, 1.31e-6)
        assert M.shape == (2, 2)
        assert np.all(np.isfinite(M))
        assert np.isfinite(efl)


class TestSplitNoTopologyDrift:
    """Belt-and-braces: pin which submodule each name originated
    from so a v5.2 maintainer can't accidentally move e.g.
    ``system_abcd`` out of ``seidel.py`` without updating this
    test (and consequently re-thinking the dependency graph).
    """

    def test_surface_module_only_owns_static_data(self):
        """``surface.py`` must NOT depend on intersection / trace / etc.
        Pin this by asserting it doesn't import them at module level."""
        import lumenairy.raytrace.surface as s
        # If surface.py imported any of these, the module reference
        # would be on the module.  This rules out a future drift
        # where someone moves the trace engine "back" into surface.
        for forbidden in ('_intersect_surface', '_refract',
                          '_reflect', '_transfer', '_apply_coord_break',
                          'trace', 'trace_world', 'system_abcd',
                          'seidel_coefficients'):
            assert not hasattr(s, forbidden), (
                f"surface.py drift: {forbidden!r} should not be "
                f"a module-level name in surface.py")

    def test_intersection_module_doesnt_pull_trace_engine(self):
        """``intersection.py`` is a leaf module -- only ``surface``
        in its dependency closure (no trace / no world / no
        seidel)."""
        import lumenairy.raytrace.intersection as inter
        for forbidden in ('trace', 'trace_world', 'system_abcd',
                          'seidel_coefficients', 'spot_rms',
                          'ray_fan_data'):
            assert not hasattr(inter, forbidden), (
                f"intersection.py drift: {forbidden!r} should not be "
                f"a module-level name in intersection.py")
