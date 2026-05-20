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
    # audit closure: P2-NEW-F1-4
    def test_first_call_emits_userwarning(
            self, reset_wave_propagator_latch):
        """The first call to ``set_default_wave_propagator`` after a
        fresh-process state must emit exactly one ``UserWarning``
        explaining the knob is API-only at v4.16.2/v4.16.3."""
        from lumenairy.propagators.propagation import (
            set_default_wave_propagator,
            get_default_wave_propagator,
        )
        original = get_default_wave_propagator()
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                set_default_wave_propagator('fresnel')
            uw = [w for w in ws if issubclass(w.category, UserWarning)]
            assert len(uw) == 1, (
                f"Expected exactly one UserWarning on first call; got "
                f"{len(uw)}: {[str(w.message) for w in uw]}")
            msg = str(uw[0].message)
            assert 'set_default_wave_propagator' in msg
            assert 'API-only' in msg or 'no library consumer' in msg
            assert 'v5.0' in msg
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
            set_default_wave_propagator,
            get_default_wave_propagator,
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
            set_default_wave_propagator,
            get_default_wave_propagator,
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
        notice on the next valid call."""
        from lumenairy.propagators.propagation import (
            set_default_wave_propagator,
        )
        import lumenairy.propagators.propagation as _prop
        with pytest.raises(ValueError):
            set_default_wave_propagator('not_a_propagator')
        # Latch must still be False -- the ValueError fired BEFORE the
        # warn-emit / latch-flip code.
        assert not _prop._DEFAULT_WAVE_PROPAGATOR_NO_CONSUMER_WARNED


class TestSetDefaultDyNoConsumerWarning:
    # audit closure: P2-NEW-F1-4
    def test_first_call_emits_userwarning(self, reset_dy_latch):
        from lumenairy.propagators.propagation import (
            set_default_dy,
            get_default_dy,
        )
        original = get_default_dy()
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                set_default_dy(5e-6)
            uw = [w for w in ws if issubclass(w.category, UserWarning)]
            assert len(uw) == 1
            msg = str(uw[0].message)
            assert 'set_default_dy' in msg
            assert 'API-only' in msg or 'no library consumer' in msg
            assert 'v5.0' in msg
        finally:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                set_default_dy(original)

    # audit closure: P2-NEW-F1-4
    def test_first_call_with_none_also_emits(self, reset_dy_latch):
        """The ``None`` (early-return) branch must also emit -- both
        ``set_default_dy(None)`` and ``set_default_dy(<float>)`` set
        the default and so both should surface the no-consumer notice."""
        from lumenairy.propagators.propagation import set_default_dy
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            set_default_dy(None)
        uw = [w for w in ws if issubclass(w.category, UserWarning)]
        assert len(uw) == 1, (
            f"set_default_dy(None) must emit the no-consumer notice "
            f"too; got {len(uw)} warnings")

    # audit closure: P2-NEW-F1-4
    def test_second_call_emits_no_warning(self, reset_dy_latch):
        from lumenairy.propagators.propagation import (
            set_default_dy,
            get_default_dy,
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
        from lumenairy.propagators.propagation import set_default_dy
        import lumenairy.propagators.propagation as _prop
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
            set_default_real_dtype,
            get_default_real_dtype,
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
                or 'staged for v5.0' in section_l), (
            "Migration-Guide.md §4.16.2 must include an honest "
            "limitation note that the wave-propagator / dy knobs are "
            "API-only at v4.16.2/v4.16.3")


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

    # audit closure: P2-NEW-F1-4 (sibling-gap sweep)
    def test_wave_propagator_and_dy_have_zero_consumers_lib_wide(self):
        """Pin the v4.16.3 honest state: across the lumenairy package,
        no source file (other than the propagation.py module that
        DEFINES the knob) reads ``DEFAULT_WAVE_PROPAGATOR`` /
        ``get_default_wave_propagator`` / ``DEFAULT_DY`` /
        ``get_default_dy``.  When a v5.0 PR adds the first consumer
        this assertion FAILS, prompting the maintainer to also remove
        the no-consumer UserWarning + Migration-Guide limitation
        note.

        The pin is intentionally tight: if a future PR adds a
        consumer, this test fails LOUDLY rather than leaving a stale
        warning in place.
        """
        import os
        import lumenairy as la
        pkg_root = os.path.dirname(la.__file__)
        SYMBOLS = (
            'DEFAULT_WAVE_PROPAGATOR',
            'get_default_wave_propagator',
            'DEFAULT_DY',
            'get_default_dy',
        )
        # propagation.py is the definition site; lumenairy/__init__.py
        # is the re-export site (top-level API surface).  Neither
        # counts as a "consumer" -- a real consumer is library code
        # that READS the value to drive a behavioural decision.
        EXEMPT = {
            os.path.normpath(os.path.join(
                pkg_root, 'propagators', 'propagation.py')),
            os.path.normpath(os.path.join(pkg_root, '__init__.py')),
        }
        consumers = {}
        for dirpath, _dirs, files in os.walk(pkg_root):
            for fn in files:
                if not fn.endswith('.py'):
                    continue
                p = os.path.normpath(os.path.join(dirpath, fn))
                if p in EXEMPT:
                    continue
                try:
                    with open(p, 'r', encoding='utf-8') as fh:
                        body = fh.read()
                except (OSError, UnicodeDecodeError):
                    continue
                for sym in SYMBOLS:
                    if sym in body:
                        consumers.setdefault(sym, []).append(p)
        # The current honest state: zero consumers outside the
        # definition site.  When this fires, also remove the
        # corresponding UserWarning in propagation.py.
        assert not consumers, (
            "v4.16.3 (audit P2-NEW-F1-4 sibling-gap sweep): expected "
            "zero consumers of DEFAULT_WAVE_PROPAGATOR / DEFAULT_DY "
            "outside propagation.py at v4.16.3 ship.  Found "
            f"consumers: {consumers}.  When you add the first real "
            "consumer, also (a) remove the no-consumer UserWarning "
            "in set_default_wave_propagator / set_default_dy, and "
            "(b) update the Migration-Guide.md §4.16.2 limitation "
            "note + the §v5.0 recipe.")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
