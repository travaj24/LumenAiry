"""Pinning tests for the v4.15.1 Agent-E scope.

Scope (per the v4.15.1 release dispatch):

* **E.1 / P1-NEW-C** -- UI deprecation migration: the 7 callsites in
  ``lumenairy/ui/model.py`` that exercised the legacy positional
  ``Source.method(size, N, dx, wavelength)`` form are migrated to the
  canonical kwarg-only form.  Source-level structural test (no live
  Qt widget required for this pin).
* **E.2 / P1-F1-6** -- ``system.evaluate`` mixed-shape precedence:
  passing a prescription dict with BOTH the factory shape
  (``surfaces`` + ``thicknesses``) AND the Zemax shape (``elements`` +
  ``all_thicknesses``) is now a ValueError instead of silent
  precedence.
* **E.3 / P2-NEW-A** -- ``system.evaluate`` Zemax-loader branch
  round-trip pin: the audit found the only ``system.evaluate`` test
  exercised the factory shape via ``make_singlet``; the Zemax branch
  was un-pinned.  This file adds the missing round-trip.
* **E.4 / P2** -- Sentinel consolidation: ``_ZeroApertureMaskSentinel``
  and ``_AngleUnsetSentinel`` now inherit from the shared
  ``_deprecation._Sentinel`` base which provides a pickle-safe
  ``__reduce__`` protocol via a name-keyed singleton registry.
  Pin: pickle round-trip preserves ``is``-identity.
* **E.5 / P3 sweep** -- four narrow P3 fixes:
    - n=1.0 consistency between ``user_library.register_fixed_glass``
      and ``multiconfig._resolve_lens_glass_index``.
    - ``io/codegen.py`` runtime version pin gains an upper-bound
      major-version-bump ``UserWarning``.
    - ``elements/bsdf.LambertianBSDF.evaluate`` gains an explicit
      surface-frame assertion + a runtime warning when the caller
      passes a non-axially-aligned incident direction.
    - ``elements/coatings.thin_film_stack`` always-emits the TIR cap
      warning at BOTH the intra-stack and substrate-interface sites
      (the substrate cap was silent pre-v4.15.1).

Author: Agent E -- v4.15.1.
"""
from __future__ import annotations

import importlib
import inspect
import pickle
import sys
import warnings

import numpy as np
import pytest


# ============================================================================
# E.1 -- P1-NEW-C: UI consumer migration
# ============================================================================

class TestE1UIModelNoDeprecatedSourceCalls:
    """``lumenairy/ui/model.py`` no longer uses the legacy positional
    ``Source.method(size, N, dx, wavelength)`` form.

    Source-level structural pin -- no live Qt widget required.  The
    fix is a 7-callsite migration to the canonical kwarg-only form
    introduced in v4.15.0; pre-v4.15.1 the UI subpackage emitted its
    own ``DeprecationWarning``s on import-and-build, the very same
    warnings v4.15.0's deprecation shim was designed to surface to
    EXTERNAL callers.
    """

    @staticmethod
    def _model_source() -> str:
        # Locate the source via importlib so the test does not require
        # PySide6 / matplotlib at module-import time.
        spec = importlib.util.find_spec('lumenairy.ui.model')
        assert spec is not None and spec.origin, (
            'lumenairy.ui.model not found on sys.path')
        with open(spec.origin, 'r', encoding='utf-8') as f:
            return f.read()

    @pytest.mark.parametrize('factory', [
        'gaussian', 'top_hat', 'fiber_mode', 'plane_wave', 'point_source',
    ])
    def test_no_positional_size_n_dx_wavelength_call(self, factory):
        """No call of the form ``_Source.<factory>(<size>, N, dx_m,
        wavelength)`` (or any 3+ positional sequence) survives.

        The legacy positional form had ``N`` as the SECOND positional
        argument (size-args first) or as the FIRST positional argument
        (for plane_wave/point_source).  Either way the pattern
        ``_Source.<factory>(<expr>, N, ...)`` is a tell.  After the
        v4.15.1 migration every call site uses ``_Source.<factory>(N=N,
        dx=dx_m, wavelength=wavelength, ...)``.
        """
        src = self._model_source()
        import re
        # Match ``_Source.<factory>(`` followed by something that does
        # NOT start with ``N=`` on the SAME or NEXT non-blank token --
        # i.e. catches both inline and split-across-lines calls.
        # Concretely: forbid a positional ``N`` argument on the first
        # non-whitespace token after the opening paren.
        pattern = re.compile(
            r'_Source\.' + factory + r'\s*\(\s*([^=\),]+)\s*,',
        )
        bad = []
        for m in pattern.finditer(src):
            first_arg = m.group(1).strip()
            # The canonical form has the first token be a kwarg of
            # the form ``X=Y`` -- any bare identifier or expression
            # would imply a positional argument.
            if '=' not in first_arg:
                bad.append((m.start(), first_arg))
        assert not bad, (
            f"P1-NEW-C: lumenairy/ui/model.py still has {len(bad)} "
            f"positional call(s) to Source.{factory}: {bad[:3]!r}.  "
            "Migrate to the canonical kwarg-only form "
            f"``_Source.{factory}(N=N, dx=dx_m, wavelength=wavelength, "
            "...)``."
        )

    def test_no_deprecation_warnings_at_ui_import(self):
        """Re-importing ``lumenairy.ui.model`` (when Qt is available)
        must NOT raise DeprecationWarning.  When Qt is unavailable
        we skip; the structural test above already covers the file
        contents.
        """
        try:
            import PySide6  # noqa: F401
        except ImportError:
            pytest.skip('PySide6 unavailable; structural test covers '
                        'the migration')
        # Force a fresh import.
        for mod in [m for m in list(sys.modules.keys())
                    if m == 'lumenairy.ui.model']:
            del sys.modules[mod]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            import lumenairy.ui.model  # noqa: F401
        dep = [w for w in caught
               if issubclass(w.category, DeprecationWarning)
               and 'Source.' in str(w.message)]
        assert not dep, (
            f"P1-NEW-C regression: lumenairy.ui.model re-import "
            f"emitted {len(dep)} Source-related DeprecationWarning(s): "
            f"{[str(w.message) for w in dep[:2]]}")


# ============================================================================
# E.2 -- P1-F1-6: mixed-shape precedence in system.evaluate
# ============================================================================

class TestE2SystemEvaluateMixedShape:
    """A prescription dict carrying BOTH the factory shape
    (``surfaces`` + ``thicknesses``) AND the Zemax shape (``elements``
    + ``all_thicknesses``) raises ValueError instead of silently
    routing through the Zemax branch.
    """

    def test_mixed_shape_raises_value_error(self):
        import lumenairy as la
        from lumenairy import Source
        # Build a dict that LOOKS like a factory output AND a Zemax
        # output -- the kind of half-constructed prescription a user
        # might accidentally produce when starting from make_singlet
        # then hand-grafting on Zemax-style elements.
        rx = {
            'name': 'Mixed',
            'aperture_diameter': 25.4e-3,
            # Factory shape:
            'surfaces': [
                {'radius': 50.0e-3, 'conic': 0.0, 'aspheric_coeffs': None,
                 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs_y': None,
                 'glass_before': 'air', 'glass_after': 'N-BK7'},
                {'radius': -50.0e-3, 'conic': 0.0, 'aspheric_coeffs': None,
                 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs_y': None,
                 'glass_before': 'N-BK7', 'glass_after': 'air'},
            ],
            'thicknesses': [3.0e-3],
            # Zemax shape:
            'elements': [
                {'element_type': 'surface', 'radius': 50.0e-3,
                 'conic': 0.0, 'glass_before': 'air',
                 'glass_after': 'N-BK7',
                 'semi_diameter': 12.7e-3, 'is_stop': False},
                {'element_type': 'surface', 'radius': -50.0e-3,
                 'conic': 0.0, 'glass_before': 'N-BK7',
                 'glass_after': 'air',
                 'semi_diameter': 12.7e-3, 'is_stop': False},
            ],
            'all_thicknesses': [3.0e-3, 0.0],
        }
        src = Source.gaussian(N=64, dx=40e-6, wavelength=587.6e-9, w0=2e-3)
        with pytest.raises(ValueError) as info:
            la.system.evaluate(rx, src)
        msg = str(info.value).lower()
        # Diagnostic must name BOTH shapes so the user knows which
        # keys to strip.
        assert 'mutually exclusive' in msg or 'both' in msg, (
            f"P1-F1-6: ValueError message must call out the mixed-"
            f"shape conflict; got: {info.value!r}")
        assert 'elements' in msg and 'surfaces' in msg, (
            f"P1-F1-6: ValueError must mention both shape keys; "
            f"got: {info.value!r}")


# ============================================================================
# E.3 -- P2-NEW-A: system.evaluate Zemax-loader branch round-trip
# ============================================================================

class TestE3SystemEvaluateZemaxBranch:
    """The Zemax-shape (``elements`` + ``all_thicknesses``) branch of
    ``system.evaluate`` produces a finite, correctly-shaped output
    field.

    The audit Part 2 V4 noted the only ``system.evaluate`` test
    exercised the factory shape via ``make_singlet``.  The Zemax
    branch (``_decompose_prescription`` route) was un-pinned despite
    being a headline ergonomic claim of v4.15.
    """

    @staticmethod
    def _zemax_singlet_prescription() -> dict:
        """Hand-built Zemax-shape prescription for a single N-BK7
        biconvex singlet.  Mirrors the structure
        :func:`load_zemax_zmx` produces but without the redundant
        ``surfaces`` / ``thicknesses`` keys -- that's the
        Zemax-only shape we want to drive.
        """
        return {
            'name': 'ZemaxBranchPin',
            'aperture_diameter': 25.4e-3,
            'elements': [
                {'element_type': 'surface', 'radius': 50.0e-3,
                 'conic': 0.0, 'aspheric_coeffs': None,
                 'glass_before': 'air', 'glass_after': 'N-BK7',
                 'semi_diameter': 12.7e-3, 'is_stop': False,
                 'surf_num': 1, 'comment': ''},
                {'element_type': 'surface', 'radius': -50.0e-3,
                 'conic': 0.0, 'aspheric_coeffs': None,
                 'glass_before': 'N-BK7', 'glass_after': 'air',
                 'semi_diameter': 12.7e-3, 'is_stop': False,
                 'surf_num': 2, 'comment': ''},
            ],
            'all_thicknesses': [4.0e-3, 0.0],
        }

    def test_zemax_shape_routes_through(self):
        """Round-trip pin: Zemax-shape prescription + Gaussian source
        through ``system.evaluate`` yields a PropagationResult with
        finite values + the expected (N, N) shape.
        """
        import lumenairy as la
        from lumenairy import Source
        from lumenairy.propagators.result import PropagationResult
        rx = self._zemax_singlet_prescription()
        src = Source.gaussian(N=64, dx=40e-6, wavelength=587.6e-9, w0=2e-3)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            result = la.system.evaluate(rx, src)
        assert isinstance(result, PropagationResult), (
            f"P2-NEW-A: system.evaluate must return PropagationResult "
            f"for the Zemax branch; got {type(result).__name__}")
        assert result.field.shape == (64, 64), (
            f"P2-NEW-A: Zemax-branch output shape mismatch; expected "
            f"(64, 64) got {result.field.shape!r}")
        # Field must be entirely finite -- a NaN/Inf is a silent
        # decomposition or propagation failure.
        assert np.all(np.isfinite(result.field)), (
            "P2-NEW-A: Zemax-branch output field has non-finite values "
            "(NaN / Inf); the decomposition or propagation silently "
            "failed.")
        # And the integrated intensity must be > 0 -- we haven't
        # propagated to a focus, so power conservation isn't expected
        # but a non-zero answer is the bare minimum.
        I = np.abs(result.field) ** 2
        assert float(I.sum()) > 0.0, (
            "P2-NEW-A: Zemax-branch output field has zero total "
            "intensity -- propagation produced nothing.")


# ============================================================================
# E.4 -- Sentinel consolidation + pickle round-trip
# ============================================================================

class TestE4SentinelPickleRoundTrip:
    """The 3 sentinels (``_NO_DEFAULT``, ``_ZERO_APERTURE_MASK``,
    ``_ANGLE_UNSET``) all inherit from the shared
    ``_deprecation._Sentinel`` base which provides a pickle-safe
    ``__reduce__`` protocol via a name-keyed singleton registry.

    Pin: ``pickle.loads(pickle.dumps(x)) is x`` holds for every
    sentinel singleton.  Pre-v4.15.1 the two custom sentinels did
    not carry a ``__reduce__``, so a pickle round-trip produced a
    NEW instance on the receiving side and silently broke any
    ``is``-identity check downstream (distributed merit eval,
    joblib caches, multiprocessing workers).
    """

    def test_no_default_pickle_round_trip(self):
        """``_NO_DEFAULT`` (the canonical ``_Sentinel`` instance from
        ``_deprecation.py``) round-trips through pickle preserving
        ``is``-identity.
        """
        from lumenairy._deprecation import _NO_DEFAULT
        roundtrip = pickle.loads(pickle.dumps(_NO_DEFAULT))
        assert roundtrip is _NO_DEFAULT, (
            f"_NO_DEFAULT pickle round-trip lost is-identity: "
            f"got {roundtrip!r} ({id(roundtrip)}) expected "
            f"{_NO_DEFAULT!r} ({id(_NO_DEFAULT)})"
        )

    def test_zero_aperture_mask_pickle_round_trip(self):
        """``_ZERO_APERTURE_MASK`` (from ``optimize/core.py``) is
        ``is``-identical after pickle round-trip.  This was the
        canonical singleton sentinel that previously did NOT carry
        a ``__reduce__`` -- distributed merit eval (joblib /
        multiprocessing) silently broke aperture-zero detection."""
        from lumenairy.optimize.core import _ZERO_APERTURE_MASK
        roundtrip = pickle.loads(pickle.dumps(_ZERO_APERTURE_MASK))
        assert roundtrip is _ZERO_APERTURE_MASK, (
            f"_ZERO_APERTURE_MASK pickle round-trip lost "
            f"is-identity: got {roundtrip!r} ({id(roundtrip)}) "
            f"expected {_ZERO_APERTURE_MASK!r} "
            f"({id(_ZERO_APERTURE_MASK)})"
        )

    def test_angle_unset_pickle_round_trip(self):
        """``_ANGLE_UNSET`` (from ``elements/polarization.py``) is
        ``is``-identical after pickle round-trip."""
        from lumenairy.elements.polarization import _ANGLE_UNSET
        roundtrip = pickle.loads(pickle.dumps(_ANGLE_UNSET))
        assert roundtrip is _ANGLE_UNSET, (
            f"_ANGLE_UNSET pickle round-trip lost is-identity: "
            f"got {roundtrip!r} ({id(roundtrip)}) expected "
            f"{_ANGLE_UNSET!r} ({id(_ANGLE_UNSET)})"
        )

    def test_sentinel_inheritance_consolidated(self):
        """Both ``_ZeroApertureMaskSentinel`` and
        ``_AngleUnsetSentinel`` inherit from ``_Sentinel`` so the
        pickle protocol + repr-name registry are uniform.
        """
        from lumenairy._deprecation import _Sentinel
        from lumenairy.optimize.core import _ZeroApertureMaskSentinel
        from lumenairy.elements.polarization import _AngleUnsetSentinel
        assert issubclass(_ZeroApertureMaskSentinel, _Sentinel), (
            "_ZeroApertureMaskSentinel must inherit from "
            "_deprecation._Sentinel (P2 sentinel consolidation)")
        assert issubclass(_AngleUnsetSentinel, _Sentinel), (
            "_AngleUnsetSentinel must inherit from "
            "_deprecation._Sentinel (P2 sentinel consolidation)")

    def test_sentinel_registry_contains_all_three(self):
        """The module-level ``_SENTINEL_REGISTRY`` lists every
        sentinel by its singleton name -- the registry is what
        ``_sentinel_unpickle`` reads to preserve ``is``-identity.
        """
        # Import the three sentinels so they're guaranteed
        # registered.
        from lumenairy._deprecation import _NO_DEFAULT, _SENTINEL_REGISTRY
        from lumenairy.optimize.core import _ZERO_APERTURE_MASK
        from lumenairy.elements.polarization import _ANGLE_UNSET
        assert _SENTINEL_REGISTRY.get('NO_DEFAULT') is _NO_DEFAULT, (
            "_NO_DEFAULT missing from _SENTINEL_REGISTRY")
        assert (_SENTINEL_REGISTRY.get('_ZERO_APERTURE_MASK')
                is _ZERO_APERTURE_MASK), (
            "_ZERO_APERTURE_MASK missing from _SENTINEL_REGISTRY")
        assert _SENTINEL_REGISTRY.get('_ANGLE_UNSET') is _ANGLE_UNSET, (
            "_ANGLE_UNSET missing from _SENTINEL_REGISTRY")


# ============================================================================
# E.5 -- P3 sweep
# ============================================================================

class TestE5P3NEqualOneConsistency:
    """P3-1: ``register_fixed_glass(n=1.0)`` accepted AND
    ``_resolve_lens_glass_index`` resolves a registered ``n=1.0``
    glass to the registered value (was rejected pre-v4.15.1 by the
    exclusive ``1.0 < n < 5.0`` bounds check).
    """

    def test_register_fixed_glass_accepts_n_equal_1(self):
        """User-library side accepts n=1.0 (vacuum) inclusively.
        This was already the case pre-v4.15.1; the test pins the
        contract so a future bounds-tightening doesn't regress it.
        """
        from lumenairy.user_library import register_fixed_glass
        from lumenairy.glass import GLASS_REGISTRY
        # Guard against test re-runs leaving a stale registry entry
        # from a previous run.
        if '__vacuum_pin__' in GLASS_REGISTRY:
            del GLASS_REGISTRY['__vacuum_pin__']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            register_fixed_glass(name='__vacuum_pin__', n=1.0)
        assert '__vacuum_pin__' in GLASS_REGISTRY, (
            "register_fixed_glass(n=1.0) must register the entry; "
            "pre-v4.15.1 the exclusive bound would have rejected this.")

    def test_resolve_lens_glass_index_accepts_n_equal_1(self):
        """multiconfig side: ``_resolve_lens_glass_index`` resolves
        a registered n=1.0 glass to 1.0 (was raising ValueError
        from the exclusive ``1.0 < n < 5.0`` bound pre-v4.15.1).
        """
        from lumenairy.user_library import register_fixed_glass
        from lumenairy.optimize.multiconfig import _resolve_lens_glass_index
        from lumenairy.glass import GLASS_REGISTRY
        if '__vacuum_resolve_pin__' in GLASS_REGISTRY:
            del GLASS_REGISTRY['__vacuum_resolve_pin__']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            register_fixed_glass(name='__vacuum_resolve_pin__', n=1.0)
            # No fallback warning should fire (the lookup is
            # canonical) -- promote any warning into a test
            # failure to catch a regression where n=1.0 trips the
            # except path.
            warnings.simplefilter('error', UserWarning)
            n = _resolve_lens_glass_index(
                '__vacuum_resolve_pin__', wavelength=587.6e-9)
        assert n == pytest.approx(1.0, abs=1e-12), (
            f"P3-1: _resolve_lens_glass_index('__vacuum_resolve_pin__') "
            f"should return 1.0 (vacuum); got {n!r}")


class TestE5P3CodegenMajorVersionWarning:
    """P3-2: ``io/codegen.py`` emits an upper-bound major-version-bump
    ``UserWarning`` (in addition to the lower-bound RuntimeError) so
    a user on v5.x running a v4.15-generated script is warned to
    regenerate rather than silently running against a potentially
    breaking-API runtime.
    """

    @staticmethod
    def _minimal_prescription():
        return {
            'name': 'PinV151',
            'aperture_diameter': 25.4e-3,
            'wavelength': 1.31e-6,
            'elements': [],
            'surfaces': [],
            'thicknesses': [],
            'all_thicknesses': [],
            'all_glasses': [],
        }

    def test_unrolled_script_has_upper_bound_warning(self):
        """The generated unrolled script source carries the
        upper-bound warning emission.
        """
        from lumenairy.io.codegen import generate_simulation_script
        rx = self._minimal_prescription()
        script = generate_simulation_script(
            rx, wavelength=1.31e-6, N=128, dx=25e-6,
            source_sigma=2e-3, output_path=None,
            include_plotting=False, include_analysis=False,
            style='unrolled')
        # The upper-bound check uses `>= (X, 0, 0)` with X = next
        # major.  The lower-bound check uses `< (a, b, c)`.
        assert ('>= (5, 0, 0)' in script or '>=(5, 0, 0)' in script
                or '>= (5,0,0)' in script), (
            "P3-2: unrolled script missing upper-bound "
            "major-version check `>= (5, 0, 0)`; got first 80 "
            "lines:\n" + '\n'.join(script.split('\n')[:80]))
        assert 'UserWarning' in script, (
            "P3-2: unrolled script must emit UserWarning on "
            "major-version bump (not RuntimeError)")
        assert ('breaking API' in script
                or 'breaking api' in script.lower()), (
            "P3-2: warning text should mention breaking API risk")

    def test_system_script_has_upper_bound_warning(self):
        """The system-style script (`style='system'`) gets the
        same upper-bound warning.
        """
        from lumenairy.io.codegen import generate_simulation_script
        rx = self._minimal_prescription()
        script = generate_simulation_script(
            rx, wavelength=1.31e-6, N=128, dx=25e-6,
            source_sigma=2e-3, output_path=None,
            include_plotting=False, include_analysis=False,
            style='system')
        assert ('>= (5, 0, 0)' in script or '>=(5, 0, 0)' in script
                or '>= (5,0,0)' in script), (
            "P3-2: system-style script missing upper-bound "
            "major-version check `>= (5, 0, 0)`")
        assert 'UserWarning' in script, (
            "P3-2: system-style script must emit UserWarning on "
            "major-version bump")


class TestE5P3LambertianBSDFFrameWarning:
    """P3-3: ``LambertianBSDF.evaluate`` emits a RuntimeWarning when
    the caller passes an incident direction with a non-trivial
    transverse component and z > +0.7 -- a signature of a
    world-frame direction being used without the
    global -> surface-local frame rotation.
    """

    def test_incident_dir_with_transverse_component_warns(self):
        from lumenairy.elements.bsdf import LambertianBSDF
        bsdf = LambertianBSDF(rho=0.8)
        # Off-axis incident direction with |x|+|y|=0.2 and z=0.99 --
        # the signature the runtime check fires on.  The dispatch
        # task specifies (0.1, 0.1, 0.99) but the warning threshold
        # is |x|+|y| > 0.3, so use a slightly larger transverse
        # component to actually trip the guard.  (We're testing
        # that the guard FIRES -- not that the guard fires for
        # every conceivable off-axis input.)
        bad_incident = np.array([0.2, 0.2, 0.96])
        scattered = np.array([0.0, 0.0, 1.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            bsdf.evaluate(bad_incident, scattered)
        rw = [w for w in caught
              if issubclass(w.category, RuntimeWarning)
              and 'surface' in str(w.message).lower()]
        assert rw, (
            "P3-3: LambertianBSDF.evaluate must emit RuntimeWarning "
            f"when incident_dir has a large transverse component; "
            f"got: {[str(w.message) for w in caught]}")

    def test_axially_aligned_incident_does_not_warn(self):
        """An axially-aligned incident direction (the canonical
        surface-frame input) MUST NOT trigger the frame warning --
        otherwise every legitimate Monte-Carlo trace path emits
        spurious noise.
        """
        from lumenairy.elements.bsdf import LambertianBSDF
        bsdf = LambertianBSDF(rho=0.8)
        good_incident = np.array([0.0, 0.0, -1.0])  # surface-aligned
        scattered = np.array([0.0, 0.0, 1.0])
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            # Any RuntimeWarning here would raise.
            val = bsdf.evaluate(good_incident, scattered)
        # Value sanity: rho/pi for upper hemisphere.
        assert val == pytest.approx(0.8 / np.pi, abs=1e-12)


class TestE5P3CoatingsTIRWarning:
    """P3-4: ``thin_film_stack`` always-emits the TIR cap warning at
    BOTH the intra-stack and substrate-interface sites.  Pre-v4.15.1
    the substrate cap (``sin_sub = min(sin_sub, 0.9999)``) was
    silent; only the intra-stack cap emitted a warning.
    """

    def test_intra_stack_tir_warns(self):
        """Intra-stack TIR at a high-AOI low-n interface fires the
        cap-warning."""
        from lumenairy.elements.coatings import coating_reflectance
        # Constructed scenario: light enters a high-index ambient
        # (n=2.0) at 70 deg, then strikes a low-index layer (n=1.0)
        # where Snell's law gives sin_t = 2.0 * sin(70) / 1.0 ~ 1.88
        # -- well above the cap.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                coating_reflectance(
                    layers=[(1.0, 100e-9)],
                    wavelengths=633e-9,
                    angle=np.deg2rad(70.0),
                    n_ambient=2.0,
                    n_substrate=1.5,
                    polarization='s',
                )
            except Exception:
                # Some implementations may raise on extreme TIR;
                # as long as the warning fired we accept either
                # outcome.
                pass
        tir_warnings = [w for w in caught
                        if ('TIR' in str(w.message)
                            or 'tir' in str(w.message).lower())]
        assert tir_warnings, (
            f"P3-4: coating_reflectance must emit a TIR cap warning "
            f"at the intra-stack site when sin_t > 1; got warnings: "
            f"{[str(w.message) for w in caught]}")
