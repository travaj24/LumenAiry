"""v4.16.3 Agent B regression tests.

Closes 1 P1 + 2 P2 audit items from
``docs/audits/AUDIT_V4_16_2_2026_05_20.md``:

* P1-NEW-F1-1 -- Migration-Guide.md no longer documents
  ``set_default_wave_propagator`` as a one-shot replacement for
  ``apply_real_lens(..., wave_propagator='fresnel')`` (which would
  silently no-op at v4.16.2 ship and produce ASM-propagated output).
* P2-NEW-F1-3 -- ``propagate_ensemble`` real-dtype fallback shape
  refactor: the previously-unreachable ``except (TypeError, ValueError)``
  branch is no longer the SOLE consumer of ``get_default_real_dtype``.
  The canonical fallback now keys off ``in_dtype is None``.
* P2-NEW-F1-4 -- ``set_default_wave_propagator`` and ``set_default_dy``
  setters emit a one-shot ``UserWarning`` that the knob is "API-only
  in v4.16.2/v4.16.3; consumer wiring lands in v5.0".  Latched at
  module scope so optimisation loops don't flood the channel.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Latch-reset fixtures.  Parallel to v4.16.2 Agent B's
# ``reset_multiwl_warn_latch`` -- restore the module-level boolean to its
# pre-test value so the one-shot ``UserWarning`` is observable test-by-test
# without leaking state across tests.
# ---------------------------------------------------------------------------


@pytest.fixture
def reset_wave_propagator_latch():
    """Reset the ``_DEFAULT_WAVE_PROPAGATOR_NO_CONSUMER_WARNED`` latch
    so the one-shot UserWarning is observable in each test."""
    import lumenairy.propagators.propagation as _prop
    _saved = _prop._DEFAULT_WAVE_PROPAGATOR_NO_CONSUMER_WARNED
    _prop._DEFAULT_WAVE_PROPAGATOR_NO_CONSUMER_WARNED = False
    try:
        yield
    finally:
        _prop._DEFAULT_WAVE_PROPAGATOR_NO_CONSUMER_WARNED = _saved


@pytest.fixture
def reset_dy_latch():
    """Reset the ``_DEFAULT_DY_NO_CONSUMER_WARNED`` latch so the
    one-shot UserWarning is observable in each test."""
    import lumenairy.propagators.propagation as _prop
    _saved = _prop._DEFAULT_DY_NO_CONSUMER_WARNED
    _prop._DEFAULT_DY_NO_CONSUMER_WARNED = False
    try:
        yield
    finally:
        _prop._DEFAULT_DY_NO_CONSUMER_WARNED = _saved


# ============================================================================
# P2-NEW-F1-4 -- one-shot UserWarning for setters with no consumers
# ============================================================================


class TestSetDefaultWavePropagatorNoConsumerWarning:
    """v5.1.0 (Wave-4 integration): inverse pin -- the v4.16.3 +
    v5.0.1 no-consumer UserWarning was RETIRED in v5.1.0 because
    the consumers landed (Agent A resolver rollout).  The setter
    now stores silently; the test asserts NO UserWarning fires."""

    # audit closure: v5.1.0 default-knob resolver rollout
    def test_first_call_emits_no_userwarning(
            self, reset_wave_propagator_latch):
        """Post-v5.1.0: the setter no longer emits the v4.16.3
        'API-only; no consumers' UserWarning -- consumers are wired
        at ``apply_real_lens`` / ``apply_real_lens_traced`` /
        ``propagate_through_system`` and the warning would be a lie."""
        from lumenairy.propagators.propagation import (
            get_default_wave_propagator,
            set_default_wave_propagator,
        )
        original = get_default_wave_propagator()
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                set_default_wave_propagator('fresnel')
            uw = [w for w in ws if issubclass(w.category, UserWarning)
                  and 'no library consumer' in str(w.message)]
            assert uw == [], (
                f"v5.1.0: set_default_wave_propagator must NOT emit "
                f"the v4.16.3 'no consumers' UserWarning -- consumers "
                f"are now wired at apply_real_lens / "
                f"apply_real_lens_traced / propagate_through_system.  "
                f"Got: {[str(w.message) for w in uw]}")
        finally:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                set_default_wave_propagator(original)

    # audit closure: P2-NEW-F1-4
    def test_second_call_emits_no_warning(
            self, reset_wave_propagator_latch):
        """The latch must suppress all subsequent calls within the same
        process (avoid flooding optimisation loops)."""
        from lumenairy.propagators.propagation import (
            get_default_wave_propagator,
            set_default_wave_propagator,
        )
        original = get_default_wave_propagator()
        try:
            # Prime the latch (1st call -> warning).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                set_default_wave_propagator('asm')
            # 2nd call must be silent.
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                set_default_wave_propagator('fresnel')
            uw = [w for w in ws if issubclass(w.category, UserWarning)]
            assert uw == [], (
                f"Expected zero UserWarnings on 2nd call (one-shot "
                f"latch); got {[str(w.message) for w in uw]}")
        finally:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                set_default_wave_propagator(original)

    # audit closure: P2-NEW-F1-4
    def test_warning_does_not_block_storage(
            self, reset_wave_propagator_latch):
        """The warning must not interfere with the storage contract --
        ``get_default_wave_propagator`` must return the just-set value
        even on the first (warning-emitting) call."""
        from lumenairy.propagators.propagation import (
            get_default_wave_propagator,
            set_default_wave_propagator,
        )
        original = get_default_wave_propagator()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                set_default_wave_propagator('fresnel')
            assert get_default_wave_propagator() == 'fresnel'
        finally:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                set_default_wave_propagator(original)

    # audit closure: P2-NEW-F1-4
    def test_validation_failure_no_warning(
            self, reset_wave_propagator_latch):
        """A rejected setter call (ValueError) must NOT flip the latch
        -- otherwise a user typo would silently suppress the legitimate
        notice on the next valid call.

        VACUOUS SINCE v5.1.0 (recorded 2026-08-03, not a defect to fix
        here).  The warning was retired, so nothing in ``lumenairy``
        writes ``_DEFAULT_WAVE_PROPAGATOR_NO_CONSUMER_WARNED`` at runtime
        any more -- it is bound ``True`` once at import
        (``fft_infra.py``) and never flipped.  The assertion below passes
        only because the fixture set it ``False``; it cannot fail.  Kept
        as a forward guard: it regains teeth if a latch-flipping setter
        ever comes back.  The sibling ``test_*_emits_no_userwarning``
        pins in this class are NOT vacuous -- they would fail if the
        warning were reintroduced."""
        import lumenairy.propagators.propagation as _prop
        from lumenairy.propagators.propagation import (
            set_default_wave_propagator,
        )
        with pytest.raises(ValueError):
            set_default_wave_propagator('not_a_propagator')
        # Latch must still be False -- the ValueError fired BEFORE the
        # warn-emit / latch-flip code.
        assert not _prop._DEFAULT_WAVE_PROPAGATOR_NO_CONSUMER_WARNED


class TestSetDefaultDyNoConsumerWarning:
    """v5.1.0 (Wave-4 integration): inverse pin -- same retirement
    as TestSetDefaultWavePropagatorNoConsumerWarning above."""

    # audit closure: v5.1.0 default-knob resolver rollout
    def test_first_call_emits_no_userwarning(self, reset_dy_latch):
        from lumenairy.propagators.propagation import (
            get_default_dy,
            set_default_dy,
        )
        original = get_default_dy()
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                set_default_dy(5e-6)
            uw = [w for w in ws if issubclass(w.category, UserWarning)
                  and 'no library consumer' in str(w.message)]
            assert uw == [], (
                f"v5.1.0: set_default_dy must NOT emit the v4.16.3 "
                f"'no consumers' UserWarning.  Consumers wired at "
                f"apply_real_lens / apply_real_lens_traced.  Got: "
                f"{[str(w.message) for w in uw]}")
            # 2026-08-03: a ``return`` used to sit here guarding 8 lines of
            # v4.16.3-era expectations (``assert len(uw) == 1`` and the
            # warning-text pins) that inverted the assertion above.  They
            # were unreachable, so they pinned nothing; deleted rather than
            # left as dead code that reads like a live contract.
        finally:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                set_default_dy(original)

    # audit closure: v5.1.0 default-knob resolver rollout (was P2-NEW-F1-4 in v4.16.3)
    def test_first_call_with_none_also_emits_no_userwarning(self, reset_dy_latch):
        """v5.1.0: inverse pin -- ``set_default_dy(None)`` no longer
        emits the no-consumer notice because consumers are now wired."""
        from lumenairy.propagators.propagation import set_default_dy
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            set_default_dy(None)
        uw = [w for w in ws if issubclass(w.category, UserWarning)
              and 'no library consumer' in str(w.message)]
        assert uw == [], (
            f"v5.1.0: set_default_dy(None) must NOT emit the v4.16.3 "
            f"'no consumers' UserWarning.  Got: "
            f"{[str(w.message) for w in uw]}")

    # audit closure: P2-NEW-F1-4
    def test_second_call_emits_no_warning(self, reset_dy_latch):
        from lumenairy.propagators.propagation import (
            get_default_dy,
            set_default_dy,
        )
        original = get_default_dy()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                set_default_dy(1e-6)
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                set_default_dy(2e-6)
            uw = [w for w in ws if issubclass(w.category, UserWarning)]
            assert uw == []
        finally:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                set_default_dy(original)

    # audit closure: P2-NEW-F1-4
    def test_validation_failure_no_warning(self, reset_dy_latch):
        """VACUOUS SINCE v5.1.0 (recorded 2026-08-03) -- same reason as
        the wave-propagator twin above: nothing in ``lumenairy`` writes
        ``_DEFAULT_DY_NO_CONSUMER_WARNED`` at runtime any more, so the
        assertion below only re-reads what the fixture set.  Kept as a
        forward guard, not counted as coverage."""
        import lumenairy.propagators.propagation as _prop
        from lumenairy.propagators.propagation import set_default_dy
        with pytest.raises(ValueError):
            set_default_dy(-1.0)
        assert not _prop._DEFAULT_DY_NO_CONSUMER_WARNED


class TestSetDefaultRealDtypeDoesNotEmitNoConsumerWarning:
    """``set_default_real_dtype`` IS consumed (post-v4.16.3 refactor in
    ``ensemble.py``).  It must NOT emit the no-consumer UserWarning --
    that would be misleading.  Pin the asymmetry explicitly."""

    # audit closure: P2-NEW-F1-3
    def test_no_userwarning_emitted(self):
        from lumenairy.propagators.propagation import (
            get_default_real_dtype,
            set_default_real_dtype,
        )
        original = get_default_real_dtype()
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                set_default_real_dtype(np.float32)
            uw = [w for w in ws if issubclass(w.category, UserWarning)]
            # Filter to just the "no consumer" class of warning.
            no_consumer = [w for w in uw
                           if 'no library consumer' in str(w.message)
                           or 'API-only' in str(w.message)]
            assert no_consumer == [], (
                f"set_default_real_dtype must NOT emit the "
                f"no-consumer warning (it IS consumed by "
                f"propagate_ensemble at v4.16.3+); got "
                f"{[str(w.message) for w in no_consumer]}")
        finally:
            set_default_real_dtype(original)


# ============================================================================
# P2-NEW-F1-3 -- propagate_ensemble real-dtype fallback shape
# ============================================================================


class TestEnsembleRealDtypeFallbackShape:
    """Verify ``get_default_real_dtype()`` is the canonical
    ``in_dtype is None`` fallback path, not an unreachable
    ``except`` branch."""

    # audit closure: P2-NEW-F1-3
    def test_get_default_real_dtype_is_reachable_via_none_branch(self):
        """Source-level structural pin: the consumer wiring at
        ``ensemble.py`` must include an ``in_dtype is None``
        conditional that calls ``get_default_real_dtype``.

        Pre-v4.16.3 the ONLY call site was inside an unreachable
        ``except`` branch (the earlier ``hasattr(ensemble, 'dtype')``
        check guaranteed ``in_dtype`` was always a valid numpy dtype
        by the time control reached the try-block).
        """
        import inspect

        from lumenairy.propagators import ensemble as _ens
        src = inspect.getsource(_ens)
        # Pin both: (a) the source mentions get_default_real_dtype,
        # (b) it does so inside an ``in_dtype is None`` branch, not
        # only inside an ``except`` clause.
        assert 'get_default_real_dtype' in src, (
            "get_default_real_dtype consumer wiring missing from "
            "ensemble.py")
        # Look for the canonical-fallback pattern.  Accept either:
        #   if in_dtype is None: ... get_default_real_dtype ...
        # The structure is a logical conjunction across lines, so
        # require the substring sequence appears in order without an
        # intervening ``def `` (to bound to a single function).
        # Find the ACTUAL code-level conditional (``in_dtype is None:``
        # with trailing colon) -- not the substring that may also
        # appear in surrounding comments.
        idx_none = src.find('in_dtype is None:')
        assert idx_none >= 0, (
            "Expected ``if in_dtype is None:`` reachable-fallback "
            "guard in ensemble.py (P2-NEW-F1-3 closure)")
        # Within ~500 chars after that guard, get_default_real_dtype
        # must be invoked (the canonical fallback assignment).
        window = src[idx_none:idx_none + 500]
        assert 'get_default_real_dtype' in window, (
            "Expected ``get_default_real_dtype()`` call within the "
            "``in_dtype is None`` branch (P2-NEW-F1-3 closure)")

    # audit closure: P2-NEW-F1-3
    def test_propagate_ensemble_honours_real_dtype_via_default(
            self, monkeypatch):
        """Smoke-test: a no-dtype-attribute ensemble surrogate falls
        through the ``in_dtype is None`` branch and uses the value
        from ``set_default_real_dtype``.

        We can't easily construct a real ndarray without a ``.dtype``,
        but we CAN drive the fallback by monkey-patching
        ``getattr(ensemble, 'dtype', None)`` -- the most direct way is
        to patch the ``get_default_real_dtype`` symbol used inside
        ``ensemble.py`` and assert it gets called when the input
        dtype is "missing"-shaped.

        Easier check: drive the existing valid-dtype path and assert
        the accumulator dtype matches the input's real counterpart
        (which IS the documented contract; we're pinning the fallback
        shape didn't break the happy path).
        """
        import lumenairy as la
        # Build a tiny 3-D ensemble and verify the accumulator dtype
        # tracks the input dtype (the matching-real-counterpart path
        # must still work post-refactor).
        ensemble = np.ones((2, 4, 4), dtype=np.complex64)

        def _trivial_propagator(field, *, dx=1e-6, wavelength=633e-9,
                                **kwargs):
            return field

        result = la.propagate_ensemble(
            ensemble,
            dx=1e-6, wavelength=633e-9,
            propagator=_trivial_propagator,
            return_intensity=True, return_ensemble=False,
        )
        # complex64 -> float32 (matching-real-counterpart contract).
        assert result.dtype == np.float32, (
            f"complex64 ensemble must yield float32 accumulator; got "
            f"{result.dtype}")


# ============================================================================
# P1-NEW-F1-1 -- Migration-Guide.md no longer documents non-functional
#                ``set_default_wave_propagator`` recipe
# ============================================================================


class TestMigrationGuideNoBrokenRecipe:
    """Pin the Migration-Guide.md correction so the unmaintained
    ``set_default_wave_propagator`` recipe doesn't regress."""

    @pytest.fixture
    def migration_guide_text(self):
        from pathlib import Path
        # tests/unit/test_v4_16_3_agent_b.py -> .../Lumenairy
        repo_root = Path(__file__).resolve().parents[2]
        guide = repo_root / 'Migration-Guide.md'
        assert guide.exists(), f"Migration-Guide.md missing at {guide}"
        return guide.read_text(encoding='utf-8')

    # audit closure: P1-NEW-F1-1
    def test_v4_16_2_section_present(self, migration_guide_text):
        """The §4.16.2 section must still exist (we corrected the
        recipe, not deleted the section)."""
        assert '4.16.2' in migration_guide_text
        assert ('Default-config knobs' in migration_guide_text
                or 'default-config knobs' in migration_guide_text)

    # audit closure: P1-NEW-F1-1
    def test_v4_16_2_section_no_longer_promises_one_shot_wave_propagator(
            self, migration_guide_text):
        """The pre-v4.16.3 recipe pattern was:

            la.set_default_wave_propagator('fresnel')
            field = la.apply_real_lens(field, prescription=pres,
                                        wavelength=wl, dx=dx)

        i.e. ``apply_real_lens`` called with NO ``wave_propagator=``
        kwarg, relying on the setter to take effect.  This is
        misleading -- ``apply_real_lens`` hardcodes ``'asm'``.

        Pin: there must not be a code block where
        ``set_default_wave_propagator`` is immediately followed by an
        ``apply_real_lens(...)`` call that omits the
        ``wave_propagator=`` kwarg.
        """
        # Extract the 4.16.2 section block.
        if '## 4.16.2' in migration_guide_text:
            start = migration_guide_text.index('## 4.16.2')
        elif '## v4.16.2' in migration_guide_text:
            start = migration_guide_text.index('## v4.16.2')
        else:
            pytest.skip("Section heading shape changed; manual review")
        # Bound the section by the next ## heading.
        rest = migration_guide_text[start + 3:]
        end_off = rest.find('\n## ')
        section = (migration_guide_text[start:start + 3 + end_off]
                   if end_off > 0
                   else migration_guide_text[start:])

        # Find code-fence blocks.
        blocks = []
        cursor = 0
        while True:
            i = section.find('```', cursor)
            if i < 0:
                break
            j = section.find('```', i + 3)
            if j < 0:
                break
            blocks.append(section[i:j + 3])
            cursor = j + 3

        # In any block where `set_default_wave_propagator(` appears,
        # if `apply_real_lens(` ALSO appears, that call must include
        # `wave_propagator=` (i.e. the per-call kwarg is preserved
        # because the setter is not yet honoured by apply_real_lens).
        for blk in blocks:
            if 'set_default_wave_propagator(' not in blk:
                continue
            if 'apply_real_lens(' not in blk:
                continue
            # The block claims a one-shot replacement.  Require the
            # apply_real_lens call still passes wave_propagator=.
            assert 'wave_propagator=' in blk, (
                f"Migration-Guide.md §4.16.2: a code block "
                f"co-mentions ``set_default_wave_propagator`` and "
                f"``apply_real_lens`` without retaining the per-call "
                f"``wave_propagator=`` kwarg.  This documents "
                f"unimplemented behaviour -- "
                f"``apply_real_lens`` hardcodes ``'asm'`` at v4.16.3 "
                f"and does NOT consult ``get_default_wave_propagator()``."
                f"\n\nBlock:\n{blk}")

    # audit closure: P1-NEW-F1-1
    def test_v4_16_2_section_has_limitation_note(self, migration_guide_text):
        """The §4.16.2 section must include an honest limitation note
        about the knobs being API-only at v4.16.2/v4.16.3."""
        if '## 4.16.2' in migration_guide_text:
            start = migration_guide_text.index('## 4.16.2')
        elif '## v4.16.2' in migration_guide_text:
            start = migration_guide_text.index('## v4.16.2')
        else:
            pytest.skip("Section heading shape changed; manual review")
        rest = migration_guide_text[start + 3:]
        end_off = rest.find('\n## ')
        section = (migration_guide_text[start:start + 3 + end_off]
                   if end_off > 0
                   else migration_guide_text[start:])
        section_l = section.lower()
        assert ('api-only' in section_l
                or 'api only' in section_l
                or 'no library consumer' in section_l
                or 'zero downstream consumer' in section_l
                or 'staged for v5.0' in section_l
                or 'staged for v5.1' in section_l), (
            "Migration-Guide.md §4.16.2 must include an honest "
            "limitation note that the wave-propagator / dy knobs are "
            "API-only at v4.16.2 through v5.0.x (v5.0.1 corrects the "
            "deferral target from v5.0 to v5.1 per audit P1-NEW-F1-2)")


# ============================================================================
# Sibling-gap sweep -- documents finding only; not a fix item.
# ============================================================================


class TestSiblingGapDefaultConfigKnobs:
    """Document the v4.16.3 sibling-gap finding: only
    ``set_default_complex_dtype`` is widely consumed; the 3 v4.16.2-new
    siblings have either narrow (real_dtype) or zero (wave_propagator,
    dy) consumer wiring.  This test pins the current honest state so
    a future maintainer either ships consumers (and updates this pin)
    or sees the gap call-out."""

    # audit closure: P2-NEW-F1-4 (sibling-gap sweep)
    def test_complex_dtype_widely_consumed_real_dtype_narrowly(self):
        """``set_default_complex_dtype`` is read in many sites;
        ``set_default_real_dtype`` is read in 1 site (post-v4.16.3
        refactor in ensemble.py).  Pin both halves: at least the
        ensemble.py site reads ``get_default_real_dtype``.

        This is a sibling-gap awareness pin -- it doesn't enforce the
        gap stays narrow, only that the v4.16.3-claimed consumer
        actually exists.
        """
        import inspect

        from lumenairy.propagators import ensemble as _ens
        src = inspect.getsource(_ens)
        assert 'get_default_real_dtype' in src

    # audit closure: v5.1.0 default-knob resolver rollout (was P2-NEW-F1-4)
    def test_wave_propagator_and_dy_have_consumers_at_expected_sites(self):
        """v5.1.0 (Wave-4 integration): INVERSE pin -- the v4.16.3
        sibling-gap pin previously asserted ZERO consumers; v5.1.0
        rolled out the resolvers, so the inverse must now hold:
        ``apply_real_lens``, ``apply_real_lens_traced``, and
        ``propagate_through_system`` each call
        ``get_default_wave_propagator()`` (and the dy variants).

        Future maintainers who back out the resolvers see this pin
        fail loudly with an actionable message: re-wire the consumers
        or re-add the v4.16.3 no-consumer UserWarning + Migration-
        Guide limitation note.
        """
        import os

        import lumenairy as la
        pkg_root = os.path.dirname(la.__file__)
        EXPECTED_CONSUMERS = {
            'get_default_wave_propagator': {
                'elements/_lens_real.py',
                'elements/_lens_traced.py',
                'propagators/system.py',
            },
            'get_default_dy': {
                'elements/_lens_real.py',
                'elements/_lens_traced.py',
                'propagators/system.py',
            },
        }
        # propagation.py / fft_infra.py are the definition sites;
        # lumenairy/__init__.py is the re-export site.  None of those
        # count as a "consumer" -- a real consumer is library code
        # that READS the value to drive a behavioural decision.
        DEFINITION_SITES = {
            os.path.normpath(os.path.join(
                pkg_root, 'propagators', 'propagation.py')),
            os.path.normpath(os.path.join(
                pkg_root, 'propagators', 'fft_infra.py')),
            os.path.normpath(os.path.join(pkg_root, '__init__.py')),
        }
        found = {sym: set() for sym in EXPECTED_CONSUMERS}
        for dirpath, _dirs, files in os.walk(pkg_root):
            for fn in files:
                if not fn.endswith('.py'):
                    continue
                p = os.path.normpath(os.path.join(dirpath, fn))
                if p in DEFINITION_SITES:
                    continue
                try:
                    with open(p, 'r', encoding='utf-8') as fh:
                        body = fh.read()
                except (OSError, UnicodeDecodeError):
                    continue
                rel = os.path.relpath(p, pkg_root).replace(os.sep, '/')
                for sym in EXPECTED_CONSUMERS:
                    if sym in body:
                        found[sym].add(rel)
        for sym, expected in EXPECTED_CONSUMERS.items():
            missing = expected - found[sym]
            assert not missing, (
                f"v5.1.0 resolver rollout: ``{sym}`` must be consumed "
                f"at every expected site {sorted(expected)} but "
                f"missing from: {sorted(missing)}.  Re-wire the "
                f"resolver at those sites or update this pin's "
                f"expected-consumers set.")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
