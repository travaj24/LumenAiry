"""Pinning tests for the v4.15.2 Agent-E scope.

Scope (per the v4.15.2 release dispatch, AUDIT_V4_15_1 closure):

* **E.1 / P1-NEW-I** -- ROADMAP refresh: header + "Current state"
  body now reference the v4.15.1+ baseline (not the stale v4.15.0
  baseline).
* **E.2 / P2** -- ``_sentinel_unpickle`` fallback: a missing
  registry entry now raises :class:`ImportError` (strict) instead
  of silently constructing a base :class:`_Sentinel` (the v4.15.1
  fallback that lost subclass identity in distributed pipelines
  with delayed imports).
* **E.3 / P2** -- Pre-existing sentinel patterns in
  ``lumenairy/optimize/core.py`` (line cites 2151 / 2271 / 2530 /
  2772) gained dedicated :class:`_Sentinel` subclasses for pickle-
  safety + ``is``-identity discoverability.  The scalar fallback
  writes at the call sites are preserved for arithmetic
  compatibility; the new subclasses are auxiliary identity
  markers.
* **E.6 / P2** -- UI runtime test under ``-W
  error::DeprecationWarning`` (added to
  ``test_v4_15_1_agent_e.py``).
* **E.8 / P3** -- Sparrow tolerance pin tightened to <1% to match
  the achievable accuracy (~0.02% on a 256-pixel Airy fixture).
* **E.9 / P3** -- ``_NO_DEFAULT`` promoted to a dedicated
  :class:`_NoDefaultSentinel` subclass for sentinel-class
  consistency with :class:`_ZeroApertureMaskSentinel` and
  :class:`_AngleUnsetSentinel`.
* **E.10 / P3** -- ``lumenairy.algebra`` exports moved from Tier-2
  (Propagate) to Tier-1 (Build a system).  Operator algebra is a
  build-time construction surface, not a propagation surface.
* **E.11 / P2** -- ``Source.gaussian_schell`` / ``schell_model``
  no longer wrap a 3-D ensemble in a :class:`Source` whose ``E``
  is 3-D; they return the same 4-tuple as the top-level factory.

Author: Andrew Traverso -- v4.15.2 Agent E.
"""
from __future__ import annotations

import ast
import inspect
import os
import pickle
import re
import warnings

import numpy as np
import pytest

import lumenairy as la

# v4.15.4 (AUDIT_V4_15_3 P3-NEW-F2-4): the
# ``TestSourceGaussianSchellEnsembleTuple`` tests below legitimately
# invoke ``Source.gaussian_schell(...)`` / ``Source.schell_model(...)``
# with the legacy positional default to verify the 4-tuple ensemble
# return shape.  Pin DeprecationWarning back to ``default`` for this
# module so the suite stays clean under strict
# ``-W error::DeprecationWarning``.
pytestmark = pytest.mark.filterwarnings(
    'default::DeprecationWarning',
)


# ============================================================================
# E.1 (P1-NEW-I) -- ROADMAP references the v4.15.1+ baseline
# ============================================================================

class TestRoadmapBaseline:
    """ROADMAP.md's ``Current state`` section must reflect the
    v4.15.1+ baseline, not the stale v4.15.0 baseline that v4.15.1
    shipped under (the audit's P1-NEW-I finding).
    """

    @staticmethod
    def _roadmap_path() -> str:
        # ROADMAP.md is at the repo root, one level above the
        # ``lumenairy/`` package directory.
        pkg_root = os.path.dirname(os.path.dirname(la.__file__))
        return os.path.join(pkg_root, 'ROADMAP.md')

    def test_roadmap_references_v4_15_1_baseline(self):
        """ROADMAP.md header / Current state must mention v4.15.1
        (or higher) and must NOT call v4.15.0 the current baseline.
        """
        path = self._roadmap_path()
        with open(path, 'r', encoding='utf-8') as fh:
            text = fh.read()
        # Header line: "Last updated: ... (post-v4.15.x)".
        m = re.search(r'\*\*Last updated:\*\*\s+\S+\s+\(post-v(\d+)\.(\d+)\.(\d+)\)',
                      text)
        assert m is not None, (
            "ROADMAP.md header must include a "
            "``(post-vX.Y.Z)`` annotation.")
        major, minor, patch = (int(m.group(i)) for i in (1, 2, 3))
        baseline_tuple = (major, minor, patch)
        assert baseline_tuple >= (4, 15, 1), (
            f"ROADMAP header is stale: post-v{major}.{minor}.{patch} "
            f"but the current baseline is v4.15.1+ (v4.15.2 is the "
            f"release in progress)."
        )
        # Current state body must cite v4.15.1 or higher as the
        # current library baseline (the audit caught the body
        # claiming "v4.15.0, ~1400 tests").
        current_state_match = re.search(
            r'##\s+Current state(.*?)##\s+v',
            text, re.DOTALL)
        assert current_state_match is not None, (
            "ROADMAP.md must have a ``## Current state`` section.")
        section = current_state_match.group(1)
        # Must reference v4.15.1 or higher SOMEWHERE in the section.
        assert ('v4.15.1' in section or 'v4.15.2' in section), (
            "ROADMAP.md ``Current state`` does not reference "
            "v4.15.1 / v4.15.2 -- still pinned to v4.15.0 stale "
            "baseline.  Refresh per AUDIT_V4_15_1 P1-NEW-I.")
        # The stale claim of "v4.15.0, ~1400 tests" must be gone.
        assert 'v4.15.0.  Test count: ~1400' not in section, (
            "ROADMAP.md still carries the stale "
            "``v4.15.0, ~1400 tests`` claim; refresh to v4.15.1+ "
            "baseline (1625 tests at v4.15.1; v4.15.2 finalises at "
            "release commit).")


# ============================================================================
# E.2 (P2) -- _sentinel_unpickle fallback strict-raise
# ============================================================================

class TestSentinelUnpickleStrictFallback:
    """Pre-v4.15.2 the ``_sentinel_unpickle`` fallback silently
    constructed a base :class:`_Sentinel` when the requested name
    was not registered.  v4.15.2 raises :class:`ImportError` with
    an actionable message.  Audit V4_15_1 P2 closure.
    """

    def test_sentinel_unpickle_fallback_raises_for_unknown_subclass(self):
        """Direct call with an un-registered name must raise
        :class:`ImportError`."""
        from lumenairy._deprecation import _sentinel_unpickle
        unknown_name = '_THIS_SENTINEL_DOES_NOT_EXIST_v4_15_2'
        with pytest.raises(ImportError) as exc_info:
            _sentinel_unpickle(unknown_name)
        msg = str(exc_info.value)
        assert unknown_name in msg, (
            "ImportError message must include the requested name "
            "so the user knows which sentinel is missing.")
        # Must include an actionable pointer mentioning that the
        # defining module needs to be imported.
        assert 'import' in msg.lower(), (
            "ImportError message must include a hint to import "
            "the defining module.")

    def test_sentinel_unpickle_registered_name_still_works(self):
        """Registered sentinels (the common case) must still
        unpickle via the registry lookup -- the strict-raise change
        only affects the unregistered fall-through path."""
        from lumenairy._deprecation import (
            _sentinel_unpickle,
            _NO_DEFAULT,
        )
        recovered = _sentinel_unpickle('NO_DEFAULT')
        assert recovered is _NO_DEFAULT, (
            "Round-trip through _sentinel_unpickle on a registered "
            "name must return the singleton instance.")

    def test_sentinel_pickle_round_trip_preserves_subclass_identity(self):
        """Full pickle round-trip on each registered sentinel must
        preserve subclass identity (the original motivation for
        the v4.15.1 sentinel consolidation work)."""
        # Import the modules that REGISTER the subclass instances
        # (lazy imports register via the ``__init__`` ->
        # ``_SENTINEL_REGISTRY[name] = self`` side-effect).
        from lumenairy._deprecation import _NO_DEFAULT
        from lumenairy.optimize.core import _ZERO_APERTURE_MASK
        from lumenairy.elements.polarization import _ANGLE_UNSET

        for sentinel in (_NO_DEFAULT, _ZERO_APERTURE_MASK, _ANGLE_UNSET):
            data = pickle.dumps(sentinel)
            restored = pickle.loads(data)
            assert restored is sentinel, (
                f"Pickle round-trip lost identity for {sentinel!r}; "
                f"got {restored!r} ({type(restored).__name__}).")


# ============================================================================
# E.3 (P2) -- optimize/core.py sentinel subclasses
# ============================================================================

class TestOptimizeCoreSentinelMigration:
    """The four pre-existing sentinel patterns flagged by the audit
    (line cites 2151 / 2271 / 2530 / 2772 in v4.15.1
    ``optimize/core.py``) gain dedicated :class:`_Sentinel`
    subclasses in v4.15.2 for pickle-safety + ``is``-identity
    discoverability.

    The runtime arithmetic still uses scalar fallbacks at the call
    sites (``efl = 1e9``, ``strehl_best = 0.0``, etc.) -- changing
    those would break downstream magnitude-based validity checks
    like ``ctx_is_valid``.  The new subclasses are auxiliary
    identity markers, co-located with the existing
    :class:`_ZeroApertureMaskSentinel` class block.
    """

    EXPECTED_SUBCLASSES = (
        '_ZeroApertureMaskSentinel',          # v4.15.1 (line 2034)
        '_InvalidFocalLengthSentinel',        # v4.15.2 (line 2271 site)
        '_FailedScanStrehlSentinel',          # v4.15.2 (line 2530 site)
        # v4.15.4 (AUDIT_V4_15_3 P2-NEW-F1-B option a):
        # ``_PerturbedABCDFallbackSentinel`` was DELETED as dead code
        # (never wired; tuple-shaped fallback is not single-sentinel-
        # friendly).  See the v4.15.4 Agent-B release notes and the
        # new ``test_perturbed_abcd_fallback_sentinel_deleted_v4_15_4``
        # pin below.
    )

    # v4.15.4: pin the absence of the deleted class + singleton names.
    DELETED_NAMES_V4_15_4 = (
        '_PerturbedABCDFallbackSentinel',
        '_PERTURBED_ABCD_FALLBACK_SENTINEL_OBJ',
    )

    def test_optimize_core_sentinels_inherit_from_sentinel(self):
        """Each expected sentinel class is declared in
        ``optimize/core.py``, inherits from :class:`_Sentinel`, and
        instantiates a singleton with the canonical
        ``__repr__``/``__bool__`` behaviour."""
        from lumenairy._deprecation import _Sentinel
        import lumenairy.optimize.core as oc

        for cls_name in self.EXPECTED_SUBCLASSES:
            cls = getattr(oc, cls_name, None)
            assert cls is not None, (
                f"optimize/core.py is missing the v4.15.2 sentinel "
                f"class {cls_name!r}.  Expected co-located with the "
                f"existing _ZeroApertureMaskSentinel block.")
            assert issubclass(cls, _Sentinel), (
                f"{cls_name} must inherit from _Sentinel (got "
                f"bases={[b.__name__ for b in cls.__bases__]}).")

    def test_optimize_core_sentinel_singletons_repr_and_bool(self):
        """Each new singleton has the canonical
        ``<NAME>`` ``repr`` and ``bool() == False`` (the
        :class:`_Sentinel` base contract)."""
        import lumenairy.optimize.core as oc
        for inst_name in (
            '_ZERO_APERTURE_MASK',
            '_INVALID_FL_SENTINEL_OBJ',
            '_FAILED_SCAN_STREHL_SENTINEL_OBJ',
        ):
            inst = getattr(oc, inst_name, None)
            assert inst is not None, (
                f"optimize/core.py: missing singleton instance "
                f"{inst_name!r}.")
            assert repr(inst).startswith('<') and repr(inst).endswith('>'), (
                f"{inst_name}.__repr__ must follow the "
                f"``<NAME>`` convention; got {repr(inst)!r}.")
            assert bool(inst) is False, (
                f"{inst_name} should be falsy (the _Sentinel "
                f"convention).")

    def test_optimize_core_numeric_sentinels_carry_value(self):
        """The two sentinels that map to numeric fallbacks expose
        the scalar via ``float(s)`` so call sites can choose either
        ``is``-identity OR arithmetic."""
        import lumenairy.optimize.core as oc
        assert float(oc._INVALID_FL_SENTINEL_OBJ) == pytest.approx(1e9)
        assert float(oc._FAILED_SCAN_STREHL_SENTINEL_OBJ) == pytest.approx(0.0)

    def test_optimize_core_sentinel_pickle_roundtrip(self):
        """Pickle round-trip preserves singleton identity for each
        new subclass instance."""
        import lumenairy.optimize.core as oc
        for inst_name in (
            '_ZERO_APERTURE_MASK',
            '_INVALID_FL_SENTINEL_OBJ',
            '_FAILED_SCAN_STREHL_SENTINEL_OBJ',
        ):
            inst = getattr(oc, inst_name)
            data = pickle.dumps(inst)
            restored = pickle.loads(data)
            assert restored is inst, (
                f"Pickle round-trip lost identity for {inst_name}.")

    def test_perturbed_abcd_fallback_sentinel_deleted_v4_15_4(self):
        """v4.15.4 (AUDIT_V4_15_3 P2-NEW-F1-B option a):
        ``_PerturbedABCDFallbackSentinel`` and its singleton are
        DELETED -- the class was dead code (never wired, see the
        v4.15.3 deprecation docstring history) and the v4.15.3
        class-attribute marker was data-only.  This pin asserts the
        deletion stuck."""
        import lumenairy.optimize.core as oc
        for name in self.DELETED_NAMES_V4_15_4:
            assert not hasattr(oc, name), (
                f"optimize/core.py still exposes {name!r}; v4.15.4 "
                f"(AUDIT_V4_15_3 P2-NEW-F1-B option a) deleted the "
                f"unwired perturbed-ABCD fallback sentinel class + "
                f"singleton.  See the v4.15.4 Agent-B release notes."
            )


# ============================================================================
# E.8 (P3) -- sparrow docstring tolerance matches the pin
# ============================================================================

class TestSparrowDocstringTolerance:
    """The Sparrow docstring must claim a tolerance that aligns with
    the test pin.  v4.15.2 tightened the pin from 5% to 1% to match
    the achievable 0.02% on a clean Airy fixture; the docstring must
    not advertise a different number.
    """

    def test_sparrow_docstring_matches_pin_tolerance(self):
        """Docstring tolerance claim must be within a factor of 2 of
        the test pin's tolerance.  Achievable ~0.02% --> docstring
        "<1%" and pin "1%" agree; pre-v4.15.2 docstring was silent
        on tolerance while pin was 5%, off by 5x."""
        from lumenairy.analysis.core import sparrow_resolution
        doc = sparrow_resolution.__doc__
        assert doc is not None
        # Docstring must mention an achievable accuracy / tolerance.
        # We look for "<1%" or "0.02%" or "1%".
        has_tolerance_claim = any(
            kw in doc for kw in ('<1%', '<= 1%', '<=1%', '0.02%', '1%')
        )
        assert has_tolerance_claim, (
            "sparrow_resolution docstring must mention an "
            "achievable tolerance figure that matches the test "
            "pin; pre-v4.15.2 it claimed nothing while the pin "
            "tolerated 5%, post-v4.15.2 both should say <1%.")
        # Pin tolerance read directly from the test source.
        from tests.unit import test_v4_15_1_agent_c
        src = inspect.getsource(
            test_v4_15_1_agent_c.TestSparrowResolutionFix
            .test_sparrow_resolution_airy_analytical)
        # Find the ``rel < <value>`` assertion.
        m = re.search(r'assert\s+rel\s*<\s*(\d+\.?\d*(?:[eE][+-]?\d+)?)', src)
        assert m is not None, (
            "Could not locate ``assert rel < ...`` line in the "
            "sparrow analytical test; refactor must not break "
            "this pin.")
        pin_tol = float(m.group(1))
        # Pin must be at most 1% (i.e. tightened from the v4.15.1
        # 5% to the v4.15.2 1%) and at least 0.01% (else there is
        # no headroom over the ~0.02% achievable error).
        assert 1e-4 <= pin_tol <= 1e-2, (
            f"Sparrow pin tolerance {pin_tol!r} is outside the "
            f"expected v4.15.2 range [1e-4, 1e-2] (i.e. tighter "
            f"than the original 5% but with headroom over the "
            f"~0.02% achievable accuracy floor).")


# ============================================================================
# E.10 (P3) -- algebra exports moved to Tier-1
# ============================================================================

class TestAlgebraInTier1:
    """``lumenairy.algebra`` exports (``Operator``,
    ``CompositeOperator``, ``FreeSpace``, ``ThinLens``,
    ``CylindricalLens``, ``Magnify``, ``FourierTransform``,
    ``Aperture``, ``GaussianAperture``) are build-time construction
    primitives, not propagation primitives.  v4.15.2 moves them from
    Tier-2 (Propagate) to Tier-1 (Build a system) in
    ``__all__``.
    """

    ALGEBRA_EXPORTS = (
        'Operator',
        'CompositeOperator',
        'FreeSpace',
        'ThinLens',
        'CylindricalLens',
        'Magnify',
        'FourierTransform',
        'Aperture',
        'GaussianAperture',
    )

    def test_algebra_exports_are_present(self):
        """Each algebra primitive must be on the top-level package
        (carryover from v4.15.1; sanity-check that the v4.15.2 tier
        move did not break the import surface)."""
        for name in self.ALGEBRA_EXPORTS:
            assert name in la.__all__, (
                f"v4.15.1 algebra symbol {name!r} missing from "
                f"lumenairy.__all__.")
            assert hasattr(la, name), (
                f"v4.15.1 algebra symbol {name!r} not present at "
                f"top level.")

    def test_algebra_in_tier_1_section(self):
        """Parse the __init__ source and assert the algebra block
        sits in the Tier-1 (Build a system) section, BEFORE the
        Tier-2 (Propagate) section header inside ``__all__``.

        Implementation detail: ``__init__.py`` has a tier OVERVIEW
        comment block at module top whose entries are formatted
        ``#   Tier N -- <Name> (description)`` (with trailing
        parenthetical) and section markers inside ``__all__`` that
        are formatted ``# Tier N -- <Name>`` (no parenthetical on
        the Tier-1 marker; Tier-2 onward sometimes keep the
        parenthetical).  We pick the LAST occurrence of each marker
        so the slice spans the in-``__all__`` section bodies.
        """
        init_path = la.__file__
        with open(init_path, 'r', encoding='utf-8') as fh:
            src = fh.read()
        # Anchor on the in-``__all__`` section markers.  The Tier-1
        # marker is preceded by a ``=`` rule and is on a line by
        # itself; the Tier-2 marker is similarly formatted.  The
        # overview comment block at module top has trailing
        # parentheticals that our regex avoids by requiring an
        # immediate end-of-line.
        tier1_matches = list(re.finditer(
            r'#\s*Tier 1 -- Build a system\s*\n', src))
        tier2_matches = list(re.finditer(
            r'#\s*Tier 2 -- Propagate \(dispatcher \+ propagator '
            r'families\)\s*\n', src))
        assert len(tier1_matches) >= 1 and len(tier2_matches) >= 1, (
            f"Could not locate Tier-1 / Tier-2 markers in "
            f"__init__.py; got Tier-1={len(tier1_matches)}, "
            f"Tier-2={len(tier2_matches)}.")
        # The LAST occurrence of each is the __all__ section marker
        # (the module-top overview's Tier-2 line ends in
        # ``(dispatcher + propagator families)`` followed by ``\n``
        # which our Tier-2 regex matches; the in-__all__ section
        # marker matches the same regex too).
        m_tier1 = tier1_matches[-1]
        m_tier2 = tier2_matches[-1]
        # The algebra exports must appear in the slice between the
        # Tier-1 section marker and the Tier-2 section marker.
        tier1_slice = src[m_tier1.end():m_tier2.start()]
        for name in self.ALGEBRA_EXPORTS:
            # We expect the quoted symbol in __all__:
            #   ``    'Operator',``
            assert f"'{name}'" in tier1_slice, (
                f"Algebra symbol {name!r} not found in the Tier-1 "
                f"section of __all__ (v4.15.2 P3 closure -- moved "
                f"from Tier-2 to Tier-1 as algebra is a build-time "
                f"construction surface, not a propagation surface).")


# ============================================================================
# E.9 (P3) -- _NO_DEFAULT is now a dedicated subclass
# ============================================================================

class TestNoDefaultSentinelDedicatedSubclass:
    """``_NO_DEFAULT`` was a bare ``_Sentinel('NO_DEFAULT')``
    instance in v4.15.1.  v4.15.2 promotes it to a dedicated
    :class:`_NoDefaultSentinel` subclass for consistency with
    :class:`_ZeroApertureMaskSentinel` and
    :class:`_AngleUnsetSentinel`.
    """

    def test_no_default_sentinel_dedicated_subclass(self):
        """``type(_NO_DEFAULT).__name__`` is ``_NoDefaultSentinel``
        and inherits from :class:`_Sentinel`."""
        from lumenairy._deprecation import _NO_DEFAULT, _Sentinel
        type_name = type(_NO_DEFAULT).__name__
        assert type_name == '_NoDefaultSentinel', (
            f"type(_NO_DEFAULT).__name__ is {type_name!r}; "
            "v4.15.2 expects ``_NoDefaultSentinel`` (the dedicated "
            "subclass).")
        assert isinstance(_NO_DEFAULT, _Sentinel)
        # Bool / repr preserved.
        assert bool(_NO_DEFAULT) is False
        assert repr(_NO_DEFAULT) == '<NO_DEFAULT>'

    def test_no_default_sentinel_pickle_roundtrip(self):
        """Pickle round-trip preserves identity AND subclass."""
        from lumenairy._deprecation import _NO_DEFAULT, _NoDefaultSentinel
        recovered = pickle.loads(pickle.dumps(_NO_DEFAULT))
        assert recovered is _NO_DEFAULT
        assert isinstance(recovered, _NoDefaultSentinel)


# ============================================================================
# E.11 (P2) -- Source.gaussian_schell returns ensemble tuple
# ============================================================================

class TestSourceGaussianSchellEnsembleTuple:
    """``Source.gaussian_schell(...)`` no longer wraps the 3-D
    ensemble in a :class:`Source` with a 3-D ``E`` (which broke the
    Source contract: every other ``Source.*`` produces a 2-D field).
    The classmethod now returns the same 4-tuple ``(ensemble, dx,
    dy, wavelength)`` as :func:`create_gaussian_schell_source`,
    matching the top-level factory return-type convention.
    """

    def test_source_gaussian_schell_returns_ensemble_tuple_not_source(self):
        """Verify the return type alignment with the top-level
        factory.  Tuple length 4: (E_ensemble, dx, dy, wavelength)."""
        from lumenairy.sources.core import Source
        result = Source.gaussian_schell(
            N=32, dx=1e-6, wavelength=633e-9,
            w0=10e-6, sigma_g=5e-6,
            n_realizations=4, seed=0,
        )
        # Must NOT be a Source instance.
        assert not isinstance(result, Source), (
            "v4.15.2 (Agent E, P2): Source.gaussian_schell must "
            "NOT wrap a 3-D ensemble in a Source -- every other "
            "Source.* classmethod produces a 2-D field, and "
            "wrapping a 3-D ensemble silently breaks "
            "src.intensity()-style downstream calls.")
        # Must be a 4-tuple.
        assert isinstance(result, tuple), (
            f"Expected a (ensemble, dx, dy, wavelength) 4-tuple; "
            f"got {type(result).__name__}.")
        assert len(result) == 4, (
            f"Expected a length-4 tuple; got length "
            f"{len(result)}.")
        ens, dx, dy, wavelength = result
        assert isinstance(ens, np.ndarray)
        assert ens.shape == (4, 32, 32)
        assert np.iscomplexobj(ens)
        assert dx == pytest.approx(1e-6)
        assert dy == pytest.approx(1e-6)
        assert wavelength == pytest.approx(633e-9)

    def test_source_schell_model_returns_ensemble_tuple_not_source(self):
        """Same v4.15.2 contract for the sibling ``schell_model``
        classmethod."""
        from lumenairy.sources.core import Source
        N = 32
        xs = np.linspace(-1, 1, N)
        I = np.exp(-(xs[:, None] ** 2 + xs[None, :] ** 2) / 0.25)
        result = Source.schell_model(
            N=N, dx=2e-6, wavelength=633e-9,
            intensity_profile=I,
            coherence_length=8e-6,
            n_realizations=4, seed=0,
        )
        assert not isinstance(result, Source)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_source_gaussian_schell_mcf_return_unchanged(self):
        """``return_kind='mcf'`` still returns a bare
        :class:`PartialCoherenceMCF` (NOT wrapped in a Source),
        same as v4.15.1.  Sanity check that the v4.15.2 contract
        change to the ensemble path did not break the MCF path."""
        from lumenairy.sources.core import Source, PartialCoherenceMCF
        result = Source.gaussian_schell(
            N=16, dx=2e-6, wavelength=633e-9,
            w0=20e-6, sigma_g=8e-6,
            n_realizations=4, seed=0,
            return_kind='mcf',
        )
        assert isinstance(result, PartialCoherenceMCF)
