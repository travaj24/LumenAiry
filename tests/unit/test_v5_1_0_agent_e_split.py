"""
Regression tests for the v5.1.0 (Agent E) ``optimize/core.py`` split.

The v5.1.0 mechanical reorganisation broke up the 4538-LOC
``lumenairy/optimize/core.py`` monolith into six topical submodules:

* :mod:`lumenairy.optimize.context` -- :class:`MeritTerm`,
  :class:`EvaluationContext`, :class:`DesignResult`,
  :class:`Constraint`, sentinel classes.
* :mod:`lumenairy.optimize.parameterizations` -- the two
  parameterization dataclasses.
* :mod:`lumenairy.optimize.merit_terms` -- the leaf-level merit
  classes.
* :mod:`lumenairy.optimize.wrapper_merits` -- the three wrapper
  merits (``MultiWavelengthMerit`` / ``MultiFieldMerit`` /
  ``ToleranceAwareMerit``) plus the meshgrid cache.
* :mod:`lumenairy.optimize.jax_merits` -- the JAX merit family.
* :mod:`lumenairy.optimize.driver` -- the wave-propagator registry +
  :func:`design_optimize`.

The split is purely mechanical -- public API is preserved bit-for-bit.
This module pins three contracts:

1. Every name previously importable as ``lumenairy.optimize.core.X``
   is still importable post-split (the shell re-exports each).
2. The three pickle-safe sentinel singletons round-trip across
   ``pickle.dumps`` / ``pickle.loads`` via the name-keyed registry,
   independent of where their class definitions live.
3. The meshgrid-build counter ``_WRAPPER_MERIT_MESHGRID_BUILDS``
   reads live from the canonical submodule (mutated via the
   ``global`` statement inside ``_get_wrapper_merit_cache``), so test
   fixtures that check ``core._WRAPPER_MERIT_MESHGRID_BUILDS`` see
   the up-to-date value.

Author: Agent E -- v5.1.0
"""
from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 1.  Public-API preservation
# ---------------------------------------------------------------------------

class TestPublicApiPreserved:
    """The v5.1.0 split MUST preserve every previously-importable name
    at ``lumenairy.optimize.core``.  We exercise three concentric
    import shapes:

    * ``from lumenairy.optimize import X``
    * ``from lumenairy.optimize.core import X``
    * ``import lumenairy as la; la.X``

    The first two are the canonical user-facing forms; the third is
    the top-level convenience namespace.
    """

    def test_top_level_optimize_imports(self):
        """The names listed in ``lumenairy/optimize/__init__.py``'s
        ``__all__`` are all importable from
        ``lumenairy.optimize`` (the top-level public namespace)."""
        from lumenairy.optimize import (
            WAVE_PROPAGATOR_REGISTRY,
            BackFocalLengthMerit,
            CallableMerit,
            ChromaticFocalShiftMerit,
            CompositeMerit,
            Constraint,
            DesignParameterization,
            DesignResult,
            EvaluationContext,
            FocalLengthMerit,
            JaxMeritTerm,
            LGAberrationMerit,
            MatchIdealSystemMerit,
            MatchIdealThinLensMerit,
            MatchTargetOPDMerit,
            MaxFNumberMerit,
            MaxThicknessMerit,
            MeritTerm,
            MinBackFocalLengthMerit,
            MinThicknessMerit,
            MultiFieldMerit,
            MultiPrescriptionParameterization,
            MultiWavelengthMerit,
            RMSWavefrontMerit,
            SphericalSeidelMerit,
            SpotSizeMerit,
            StrehlMerit,
            ToleranceAwareMerit,
            ZernikeCoefficientMerit,
            design_optimize,
            make_lg_aberration_merit_jax,
            register_wave_propagator,
            unregister_wave_propagator,
        )
        # Smoke: each is non-None.
        for name, obj in locals().items():
            if name == 'self':
                continue
            assert obj is not None, (
                f"public name {name!r} is None -- shell re-export "
                f"missing?")

    def test_core_shell_imports(self):
        """The names that historical user code imports DIRECTLY from
        ``lumenairy.optimize.core`` (the pre-v5.1.0 monolith) all
        still resolve."""
        from lumenairy.optimize.core import (
            WAVE_PROPAGATOR_REGISTRY,
            BackFocalLengthMerit,
            CallableMerit,
            ChromaticFocalShiftMerit,
            CompositeMerit,
            Constraint,
            DesignParameterization,
            DesignResult,
            EvaluationContext,
            FocalLengthMerit,
            JaxMeritTerm,
            LGAberrationMerit,
            MatchIdealSystemMerit,
            MatchIdealThinLensMerit,
            MatchTargetOPDMerit,
            MaxFNumberMerit,
            MaxThicknessMerit,
            MeritTerm,
            MinBackFocalLengthMerit,
            MinThicknessMerit,
            MultiFieldMerit,
            MultiPrescriptionParameterization,
            MultiWavelengthMerit,
            RMSWavefrontMerit,
            SphericalSeidelMerit,
            SpotSizeMerit,
            StrehlMerit,
            ToleranceAwareMerit,
            ZernikeCoefficientMerit,
            ctx_is_valid,
            design_optimize,
            make_lg_aberration_merit_jax,
            register_wave_propagator,
            unregister_wave_propagator,
        )
        for name, obj in locals().items():
            if name == 'self':
                continue
            assert obj is not None, (
                f"core-shell name {name!r} resolves to None")

    def test_private_helper_imports(self):
        """Private helpers historically importable from
        ``lumenairy.optimize.core`` MUST keep resolving (tests + the
        propagation cache-clearer registry reference them by this
        path)."""
        from lumenairy.optimize.core import (
            _FAILED_SCAN_STREHL_SENTINEL_OBJ,
            _INVALID_FL_SENTINEL,
            _INVALID_FL_SENTINEL_OBJ,
            _METHODS_SUPPORTING_CONSTRAINTS,
            _WRAPPER_MERIT_CACHE,
            _WRAPPER_MERIT_CACHE_LOCK,
            _WRAPPER_MERIT_CACHE_SIZE,
            _ZERO_APERTURE_MASK,
            _clear_wrapper_merit_cache,
            _FailedScanStrehlSentinel,
            _fd_grad_pure,
            _get_wrapper_merit_cache,
            _InvalidFocalLengthSentinel,
            _read_path,
            _sum_merits,
            _wrapper_merit_aperture_key,
            _write_path,
            _ZeroApertureMaskSentinel,
        )
        for name, obj in locals().items():
            if name == 'self':
                continue
            assert obj is not None, (
                f"private helper {name!r} is None")

    def test_module_level_dependency_aliases(self):
        """The split lifted ``system_abcd``, ``through_focus_scan``,
        etc. into submodules, but tests that use
        ``mock.patch('lumenairy.optimize.core.X', ...)`` need the
        binding to remain at this module path.  Verify each."""
        import lumenairy.optimize.core as oc
        # Raytrace
        assert callable(oc.system_abcd)
        assert callable(oc.surfaces_from_prescription)
        assert callable(oc.seidel_coefficients)
        assert callable(oc.trace)
        # Through-focus / wave-OPD / Strehl helpers
        assert callable(oc.through_focus_scan)
        assert callable(oc.find_best_focus)
        assert callable(oc.diffraction_limited_peak)
        assert callable(oc.wave_opd_2d)
        assert callable(oc.zernike_decompose)
        # apply_real_lens family
        assert callable(oc.apply_real_lens)
        assert callable(oc.apply_real_lens_traced)
        # Precision helpers
        assert callable(oc.get_default_complex_dtype)
        # State-file helpers (json / os, used by ``_state_save``)
        assert oc._json is not None
        assert oc._os is not None


# ---------------------------------------------------------------------------
# 2.  Sentinel pickle round-trip preservation
# ---------------------------------------------------------------------------

class TestSentinelPickleRoundTrip:
    """The three pickle-safe sentinel singletons round-trip via the
    name-keyed registry in :mod:`lumenairy._deprecation`.  Moving a
    sentinel class to a new submodule (``context.py``) changes
    ``__qualname__`` and ``__module__`` but MUST NOT break pickle
    round-trip identity.
    """

    @pytest.mark.parametrize('singleton_name', [
        '_ZERO_APERTURE_MASK',
        '_INVALID_FL_SENTINEL_OBJ',
        '_FAILED_SCAN_STREHL_SENTINEL_OBJ',
    ])
    def test_sentinel_pickle_round_trip(self, singleton_name):
        """``pickle.loads(pickle.dumps(sentinel)) is sentinel``."""
        import lumenairy.optimize.core as oc
        sentinel = getattr(oc, singleton_name)
        roundtrip = pickle.loads(pickle.dumps(sentinel))
        assert roundtrip is sentinel, (
            f"Pickle round-trip lost identity for {singleton_name}: "
            f"got {roundtrip!r} ({id(roundtrip)}), expected "
            f"{sentinel!r} ({id(sentinel)}).  The name-keyed "
            f"_SENTINEL_REGISTRY in _deprecation.py must preserve "
            f"singleton identity independent of which submodule "
            f"declares the class.")

    def test_sentinel_class_inheritance_preserved(self):
        """Each new singleton's class still inherits from
        :class:`_Sentinel`, regardless of submodule location."""
        from lumenairy._deprecation import _Sentinel
        from lumenairy.optimize.core import (
            _FailedScanStrehlSentinel,
            _InvalidFocalLengthSentinel,
            _ZeroApertureMaskSentinel,
        )
        for cls in (_ZeroApertureMaskSentinel,
                    _InvalidFocalLengthSentinel,
                    _FailedScanStrehlSentinel):
            assert issubclass(cls, _Sentinel), (
                f"{cls.__name__} no longer inherits from _Sentinel "
                f"-- the pickle protocol depends on the base class.")

    def test_sentinel_registry_lists_all_three(self):
        """The module-level ``_SENTINEL_REGISTRY`` carries an entry
        for each of the three optimize-domain sentinels."""
        # Importing optimize.core triggers all three sentinel
        # registrations via their class ``__init__`` side effects.
        import lumenairy.optimize.core  # noqa: F401
        from lumenairy._deprecation import _SENTINEL_REGISTRY
        for name in ('_ZERO_APERTURE_MASK',
                     '_INVALID_FL_SENTINEL_OBJ',
                     '_FAILED_SCAN_STREHL_SENTINEL_OBJ'):
            assert name in _SENTINEL_REGISTRY, (
                f"_SENTINEL_REGISTRY missing entry for {name!r} -- "
                f"v5.1.0 split must preserve the registry entries.")

    def test_sentinel_scalar_fallbacks_preserved(self):
        """The numeric sentinels still expose ``float(s)`` for
        downstream arithmetic compatibility."""
        from lumenairy.optimize.core import (
            _FAILED_SCAN_STREHL_SENTINEL_OBJ,
            _INVALID_FL_SENTINEL_OBJ,
        )
        assert float(_INVALID_FL_SENTINEL_OBJ) == pytest.approx(1e9)
        assert float(_FAILED_SCAN_STREHL_SENTINEL_OBJ) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3.  Meshgrid-build counter live-read
# ---------------------------------------------------------------------------

class TestMeshgridBuildCounterLiveRead:
    """The wrapper-merit cache live counter
    ``_WRAPPER_MERIT_MESHGRID_BUILDS`` is mutated via ``global`` in
    ``_get_wrapper_merit_cache`` (inside ``wrapper_merits.py``).  The
    re-export shell uses PEP 562 ``__getattr__`` to forward live
    reads, so a test that reads the counter via
    ``lumenairy.optimize.core._WRAPPER_MERIT_MESHGRID_BUILDS`` sees
    the up-to-date value rather than a snapshot bound at import time.
    """

    def setup_method(self):
        from lumenairy.optimize.core import _clear_wrapper_merit_cache
        _clear_wrapper_merit_cache()

    def test_counter_increments_through_shell(self):
        """Cache miss increments the counter; cache hit does not.
        Both increments and reads happen through the shell module."""
        import lumenairy.optimize.core as oc
        before = oc._WRAPPER_MERIT_MESHGRID_BUILDS
        oc._get_wrapper_merit_cache(64, 1e-6, 50e-6, np.complex128)
        after_first = oc._WRAPPER_MERIT_MESHGRID_BUILDS
        assert after_first == before + 1, (
            f"meshgrid build did not increment counter: "
            f"before={before}, after={after_first}.  The shell's "
            f"PEP 562 __getattr__ may have lost the live-read path.")
        # Cache hit: no increment.
        oc._get_wrapper_merit_cache(64, 1e-6, 50e-6, np.complex128)
        after_hit = oc._WRAPPER_MERIT_MESHGRID_BUILDS
        assert after_hit == after_first, (
            f"cache hit incorrectly incremented counter: "
            f"after_first={after_first}, after_hit={after_hit}")

    def test_counter_visible_from_both_modules(self):
        """The counter is in sync between
        ``lumenairy.optimize.core`` (via __getattr__) and the
        canonical home ``lumenairy.optimize.wrapper_merits``."""
        import lumenairy.optimize.core as oc
        from lumenairy.optimize import wrapper_merits as wm
        oc._clear_wrapper_merit_cache()
        oc._get_wrapper_merit_cache(128, 2e-6, 100e-6, np.complex128)
        assert (oc._WRAPPER_MERIT_MESHGRID_BUILDS
                == wm._WRAPPER_MERIT_MESHGRID_BUILDS), (
            "Counter desync between shell ({oc-value}) and submodule "
            "({wm-value}).  The PEP 562 __getattr__ must forward the "
            "live read.")


# ---------------------------------------------------------------------------
# 4.  Driver functional sanity (smoke test)
# ---------------------------------------------------------------------------

def test_design_optimize_smoke():
    """A short ``design_optimize`` run completes without raising.
    Exercises the driver -> wrapper-merits -> merit-terms chain.
    """
    import lumenairy as la
    from lumenairy.optimize import (
        DesignParameterization,
        FocalLengthMerit,
        design_optimize,
    )
    pres = la.make_singlet(R1=50e-3, R2=-50e-3, d=3e-3,
                           glass='N-BK7', aperture=10e-3)
    param = DesignParameterization(
        template=pres,
        free_vars=[('surfaces', 0, 'radius'),
                   ('surfaces', 1, 'radius')],
        bounds=[(20e-3, 200e-3), (-200e-3, -20e-3)])
    merit = [FocalLengthMerit(target=50e-3, weight=1.0)]
    # Suppress the v4.16.3 first-Constraint DeprecationWarning if any
    # earlier test in the suite reset the latch.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        result = design_optimize(
            param, merit, wavelength=1.31e-6,
            N=32, dx=10e-6, method='L-BFGS-B',
            max_iter=3, verbose=False)
    # design_optimize returns a DesignResult dataclass with the
    # expected shape (Bug 0 sanity: the orchestration didn't blow up).
    assert result.x.shape == (2,)
    assert result.prescription is not None
    assert np.isfinite(result.merit)


def test_wave_propagator_registry_preserved():
    """The five default wave propagators are registered after the
    v5.1.0 split (they live in ``driver.py`` and are re-exported)."""
    from lumenairy.optimize.core import WAVE_PROPAGATOR_REGISTRY
    for name in ('real_lens', 'gbd', 'hf', 'hfpi', 'asymptotic'):
        assert name in WAVE_PROPAGATOR_REGISTRY, (
            f"wave propagator {name!r} missing from registry after "
            f"v5.1.0 split")
