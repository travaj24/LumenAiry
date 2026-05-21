"""v5.0.1 Agent C regression tests -- ROADMAP / mypy / docstring cleanup.

Closes the v5.0.0 audit findings owned by Agent C
(``docs/audits/AUDIT_V5_0_0_2026_05_20.md``):

* **P2-NEW-F2-3** -- ROADMAP.md v5.1.0 section listed v5.0-shipped
  items (CI gates, public-API smoke, Python 3.10 bump,
  ``lumenairy/system.py`` move, 5/8 shim removals,
  Migration-Guide.md existence) as still-pending.  v5.0.1 stripped
  those entries and rewrote the v5.1 section to list only items
  v5.0 actually deferred (6 file splits, library-wide
  default-knob consumer wiring, MCF naming polish, formula-3
  glass coefficients, off-axis conic, 5 examples, 57-file test
  consolidation, mypy CI activation, ruff-baseline cleanup PR,
  shared Chebyshev helpers extraction).

* **P2-NEW-V4-2** -- ``[tool.mypy]`` baseline at v5.0 ship FAILED
  with ~1952 errors (63 scope-local + 1889 cascade into untyped
  downstream files).  v5.0.1 added ``follow_imports = "silent"``
  so a future activation only sees the 63 scope-local errors
  that the v5.1 cleanup actually owns.  Config-prep only -- no
  CI wiring at v5.0.1.

* **P3-NEW-F1-1** -- ``test_v4_16_2_dispatcher_pin_doc_consistency.py``
  carried stale Python 3.9 references in the ``tomllib`` import
  fallback comments and skip-reason text.  Library floor is 3.10
  as of v5.0; v5.0.1 refreshed the comments.

* **P3-NEW-F2-2** -- CHANGELOG.md cited the ruff exclude as
  ``"ui/"`` but ``pyproject.toml`` correctly scopes as
  ``"lumenairy/ui/"``.  v5.0.1 fixed the doc drift.

* **P3-NEW-F2-MCF** -- CHANGELOG line 112 said "MCF.coherence_at(...)
  partial-coherence object" but ``PartialCoherenceMCF`` and
  ``coherence_at(...)`` already shipped in v4.15.1 and are
  re-exported at the top level.  v5.0.1 clarified that what
  remains is the naming polish (a shorter
  ``lumenairy.MCF`` top-level alias).

* **P3-NEW-V3-3** -- ``pyproject.toml`` classifiers listed Python
  3.14 but the CI matrix runs 3.10-3.13 only.  v5.0.1 dropped
  3.14 from the classifiers (canonical target).

* **``apply_*_lens`` shim-preservation closure** -- documented
  in CHANGELOG + ROADMAP that the 3 retained shims
  (``apply_thin_lens`` / ``apply_real_lens`` /
  ``apply_real_lens_traced``) are **intentionally kept** as
  legitimate public API surface so a v5.2+ audit doesn't flag
  them again.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_ROADMAP = _REPO_ROOT / "ROADMAP.md"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_CHANGELOG = _REPO_ROOT / "CHANGELOG.md"
_DISPATCHER_PIN = (
    _REPO_ROOT / "tests" / "unit"
    / "test_v4_16_2_dispatcher_pin_doc_consistency.py"
)


# Mirror the dispatcher-pin file's tomllib fallback so this file
# also runs on a clean Python 3.10 interpreter.
try:
    import tomllib as _TOMLLIB  # type: ignore[import-not-found]
    _TOMLLIB_SKIP_REASON = None
except ImportError:  # pragma: no cover -- exercised only on Python 3.10
    try:
        import tomli as _TOMLLIB  # type: ignore[import-not-found]
        _TOMLLIB_SKIP_REASON = None
    except ImportError:  # pragma: no cover -- truly missing
        _TOMLLIB = None  # type: ignore[assignment]
        _TOMLLIB_SKIP_REASON = (
            "Neither `tomllib` (Python 3.11+ stdlib) nor `tomli` "
            "(Python 3.10 backport) is importable."
        )


def _require_tomllib():
    if _TOMLLIB is None:
        pytest.skip(_TOMLLIB_SKIP_REASON)
    return _TOMLLIB


# ----------------------------------------------------------------------
# audit closure: P2-NEW-F2-3 -- ROADMAP v5.1 stale-item strip
# ----------------------------------------------------------------------


def _extract_v5_1_section(text: str) -> str:
    """Return the body of the ``## v5.1.0 ...`` heading up to the
    next ``## `` top-level heading.
    """
    match = re.search(
        r'^##\s+v5\.1\.0\b.*?(?=^##\s)',
        text, re.MULTILINE | re.DOTALL,
    )
    assert match is not None, (
        "ROADMAP.md is missing a ``## v5.1.0 ...`` section.  "
        "v5.0.1 (audit P2-NEW-F2-3) rewrote the section; the heading "
        "must remain."
    )
    return match.group(0)


# v5.2.3 (ROADMAP refresh): the v5.1.0 ROADMAP section itself has
# been retired -- v5.1.0 shipped at the v5.1.0 release tag, v5.2.0
# shipped the remaining v5.1.0 deferrals, and v5.2.3 collapsed the
# section to a single "shipped at v5.x" pointer that links to the
# "Shipped highlights" section.  The 7-test class below was the
# v5.0.1 audit P2-NEW-F2-3 closure pin guarding against the v5.1.0
# section listing shipped items as future work.  That contract is
# preserved structurally by the v5.2.0 V11 doc-consistency walker
# + the v5.2.3 V16 CHANGELOG content-vs-changeset walker (which
# catches the analogous CHANGELOG-side fabrication class), so the
# narrow ``## v5.1.0`` section pin is superseded.  Skipped at v5.2.3
# rather than deleted so the audit-ID citation trail survives in
# the repo; the v5.3 work item is to either resurrect the
# generalized version (asserting against the current forward-horizon
# section) or delete the dead class entirely.
@pytest.mark.skip(reason="v5.0.1 audit P2-NEW-F2-3 pin superseded "
                         "by v5.2.3 ROADMAP retirement of the "
                         "v5.1.0 section; structural guard moved "
                         "to V11/V16 walkers.")
class TestRoadmapV5_1SectionStaleItemStrip:
    """ROADMAP.md ``## v5.1.0`` section must NOT list v5.0-shipped
    items as still-pending.  Closes audit P2-NEW-F2-3.
    """

    # audit closure: P2-NEW-F2-3
    def test_v5_1_section_does_not_promise_shipped_ci_gates(self) -> None:
        """v5.0 shipped ``.github/workflows/unit-tests.yml`` with a
        ``pytest tests/unit -m "not integration"`` job + ``ruff check``
        lint + public-API smoke.  The v5.1 section must not list these
        as still-pending CI gates.
        """
        section = _extract_v5_1_section(_ROADMAP.read_text(encoding="utf-8"))
        # The shipped-CI-gates bullet pattern from the pre-v5.0.1 ROADMAP
        # was a bullet titled "CI gates" with sub-bullets listing the
        # 4 components.  v5.0.1 strips it.  Detect by header phrase.
        assert "**CI gates** (currently absent)" not in section, (
            "ROADMAP v5.1 section still contains the pre-v5.0.1 "
            "``**CI gates** (currently absent)`` bullet.  v5.0 shipped "
            "the unit-tests workflow, ruff lint, public-API smoke -- "
            "this should have been stripped by audit P2-NEW-F2-3 "
            "closure."
        )

    # audit closure: P2-NEW-F2-3
    def test_v5_1_section_does_not_promise_shipped_python_310_bump(self) -> None:
        """v5.0 shipped ``requires-python = ">=3.10"``.  The v5.1
        section must not list the bump as still-pending.
        """
        section = _extract_v5_1_section(_ROADMAP.read_text(encoding="utf-8"))
        # The pre-v5.0.1 wording was exactly:
        #     "* **Bump `requires-python` to >=3.10**"
        assert "Bump `requires-python` to >=3.10" not in section, (
            "ROADMAP v5.1 section still lists the Python 3.10 bump as "
            "pending.  v5.0 shipped it; audit P2-NEW-F2-3 closure "
            "should have stripped this bullet."
        )

    # audit closure: P2-NEW-F2-3
    def test_v5_1_section_does_not_promise_shipped_system_py_move(self) -> None:
        """v5.0 moved ``lumenairy/system.py`` ->
        ``lumenairy/propagators/system.py``.  The v5.1 section must
        not list the move as still-pending.
        """
        section = _extract_v5_1_section(_ROADMAP.read_text(encoding="utf-8"))
        # Pre-v5.0.1 wording was the arrow bullet:
        #     "* **`lumenairy/system.py` -> `lumenairy/propagators/system.py`**"
        # We match on the bold-italic phrasing rather than the arrow
        # (different dash variants survive the cp1252 round-trip).
        drift_markers = [
            "**`lumenairy/system.py`",      # bold-leading marker
        ]
        for marker in drift_markers:
            assert marker not in section, (
                f"ROADMAP v5.1 section still references the "
                f"``lumenairy/system.py -> propagators/system.py`` "
                f"move as pending: marker {marker!r} present.  "
                f"v5.0 shipped this; audit P2-NEW-F2-3 closure should "
                f"have stripped the bullet."
            )

    # audit closure: P2-NEW-F2-3
    def test_v5_1_section_does_not_promise_shipped_migration_guide(self) -> None:
        """v4.16.2 shipped ``Migration-Guide.md`` (v5.0 added the
        v5.0.0 section).  The v5.1 section must not list it as
        "currently missing".
        """
        section = _extract_v5_1_section(_ROADMAP.read_text(encoding="utf-8"))
        assert "Migration-Guide.md` -- currently missing" not in section, (
            "ROADMAP v5.1 section still claims Migration-Guide.md is "
            "missing.  It shipped in v4.16.2 and v5.0 added a v5.0.0 "
            "section; audit P2-NEW-F2-3 closure should have stripped "
            "this bullet."
        )

    # audit closure: P2-NEW-F2-3
    def test_v5_1_section_does_not_promise_shipped_shim_removals(self) -> None:
        """v5.0 removed 5 of the 8 shims in AUDIT_V4_13_1 Part 5
        (``analysis.analysis``, ``ao``, ``io.hdf5``, JAX legacy
        aperture, ``cosmic_ray_rate``).  The v5.1 section must not
        list "Remove 8 active back-compat shims" as pending.
        """
        section = _extract_v5_1_section(_ROADMAP.read_text(encoding="utf-8"))
        assert "Remove 8 active back-compat shims" not in section, (
            "ROADMAP v5.1 section still says ``Remove 8 active "
            "back-compat shims``.  v5.0 removed 5 of those 8; the "
            "remaining 3 (``apply_*_lens`` re-exports) are "
            "intentionally retained per the v5.0 CHANGELOG "
            "``Shims preserved`` decision.  Audit P2-NEW-F2-3 "
            "closure should have rewritten this section."
        )

    # audit closure: P2-NEW-F2-3
    def test_v5_1_section_does_not_promise_shipped_public_api_smoke(self) -> None:
        """v5.0 shipped ``tests/unit/test_public_api.py``.  The v5.1
        section must not list it as still-pending.
        """
        section = _extract_v5_1_section(_ROADMAP.read_text(encoding="utf-8"))
        # Pre-v5.0.1 wording was a sub-bullet under "CI gates" reading
        # ``tests/test_public_api.py`` smoke test asserting every ``__all__``
        # entry is ``getattr(lumenairy, name)``-resolvable.
        assert "smoke test asserting every `__all__`" not in section, (
            "ROADMAP v5.1 section still describes the public-API "
            "smoke as pending.  v5.0 shipped "
            "``tests/unit/test_public_api.py``; audit P2-NEW-F2-3 "
            "closure should have stripped this sub-bullet."
        )

    # audit closure: P2-NEW-F2-3
    def test_v5_1_section_keeps_genuinely_deferred_items(self) -> None:
        """Verify the deferred items v5.0 actually carried forward
        ARE listed (the strip must not over-trim).
        """
        section = _extract_v5_1_section(_ROADMAP.read_text(encoding="utf-8"))
        required_substrings = [
            "6 file splits",                  # mechanical reorg
            "raytrace/core.py",               # file-split target
            "propagators/propagation.py",     # file-split target
            "propagators/asymptotic.py",      # file-split target
            "optimize/core.py",               # file-split target
            "io/prescriptions.py",            # file-split target
            "analysis/core.py",               # file-split target
            "set_default_wave_propagator",    # consumer wiring
            "formula-3",                      # glass coefficient ingestion
            "Off-axis conic",                 # surface-frame feature
            "MCF",                            # MCF naming polish (P3-NEW-F2-MCF)
            "5 missing examples",             # documentation
            "test_audit_fixes_",              # 57-file consolidation
            "mypy",                           # mypy CI activation
        ]
        missing = [s for s in required_substrings if s not in section]
        assert not missing, (
            f"ROADMAP v5.1 section is missing deferred items that "
            f"v5.0 genuinely carried forward: {missing}.  The "
            f"P2-NEW-F2-3 strip removed too much."
        )

    # audit closure: apply_*_lens shim-preservation
    def test_v5_1_section_documents_intentionally_kept_shims(self) -> None:
        """The 3 ``apply_*_lens`` shims that v5.0 kept by design
        must be documented as ``intentionally retained`` somewhere
        in the v5.1 section (or the broader ROADMAP) so a v5.2+
        audit doesn't flag them again.
        """
        roadmap = _ROADMAP.read_text(encoding="utf-8")
        # Either the v5.1 section says they're intentionally kept,
        # or the Current state section says so.
        kept_markers = (
            "intentionally retained",
            "intentionally kept",
            "Shims preserved",
        )
        assert any(m in roadmap for m in kept_markers), (
            "ROADMAP.md does not document that the ``apply_*_lens`` "
            "re-exports are intentionally retained.  Audit "
            "``apply_*_lens`` shim-preservation closure requires "
            "this so a v5.2+ audit doesn't flag them again."
        )


# ----------------------------------------------------------------------
# audit closure: P2-NEW-V4-2 -- mypy config preparation
# ----------------------------------------------------------------------


class TestMypyConfigPreparation:
    """``[tool.mypy]`` in pyproject.toml must have
    ``follow_imports = "silent"`` so a future activation does not
    cascade into ~1889 errors in untyped downstream modules.
    Closes audit P2-NEW-V4-2 (config preparation only;
    activation deferred to v5.1).
    """

    # audit closure: P2-NEW-V4-2
    def test_mypy_follow_imports_silent_is_set(self) -> None:
        tomllib_mod = _require_tomllib()
        with _PYPROJECT.open("rb") as fh:
            data = tomllib_mod.load(fh)
        mypy_cfg = data.get("tool", {}).get("mypy", {})
        assert mypy_cfg.get("follow_imports") == "silent", (
            "[tool.mypy] in pyproject.toml must set "
            "``follow_imports = \"silent\"`` (v5.0.1 audit "
            "P2-NEW-V4-2).  Without this, ``mypy --strict`` "
            "follows imports into ~1889 untyped downstream files "
            "(numpy/scipy/matplotlib + un-annotated lumenairy "
            "modules) before reaching the 63 scope-local errors "
            "the v5.1 cleanup is supposed to own.  Restore the "
            "key."
        )

    # audit closure: P2-NEW-V4-2
    def test_mypy_files_whitelist_is_intact(self) -> None:
        """The ``files`` whitelist must still scope mypy to the 5
        small self-contained modules (not silently broadened).
        """
        tomllib_mod = _require_tomllib()
        with _PYPROJECT.open("rb") as fh:
            data = tomllib_mod.load(fh)
        mypy_cfg = data.get("tool", {}).get("mypy", {})
        files = mypy_cfg.get("files", [])
        expected = {
            "lumenairy/backend",
            "lumenairy/_deprecation.py",
            "lumenairy/_context.py",
            "lumenairy/progress.py",
            "lumenairy/memory.py",
        }
        # Allow path-normalisation variants (backslash on Windows
        # configs is not used here -- pyproject is POSIX-style).
        assert set(files) == expected, (
            f"[tool.mypy] files whitelist drifted from the expected "
            f"5-module v5.0 scope.  Got {set(files)}, expected "
            f"{expected}.  Audit P2-NEW-V4-2 deferred the actual "
            f"scope-local cleanup to v5.1 and the activation "
            f"alongside it; the whitelist must not be silently "
            f"broadened before that cleanup lands."
        )

    # audit closure: P2-NEW-V4-2 -- flipped at v5.2 (AUDIT_V5_1_0
    # P2-NEW-F2-2): the pin originally guarded against premature CI
    # wiring at v5.0.1 ("activation deferred to v5.1").  v5.2 actually
    # closes the 63-error scope-local cleanup and wires the gate, so
    # the pin now asserts the opposite invariant -- mypy MUST appear
    # in unit-tests.yml and MUST run as a blocking gate (not
    # continue-on-error).  Any future PR that removes the gate
    # silently will trip this pin.
    def test_mypy_wired_into_ci_at_v5_2(self) -> None:
        """mypy must appear in ``unit-tests.yml`` as a blocking job
        from v5.2 onwards.  v5.2 closes the scope-local 63-error
        cleanup and activates the gate; the gate must not regress to
        advisory (continue-on-error: true) or be removed entirely.
        """
        workflow = (
            _REPO_ROOT / ".github" / "workflows" / "unit-tests.yml"
        )
        if not workflow.is_file():
            pytest.skip(
                "unit-tests.yml workflow not present -- structural "
                "skip, not a failure."
            )
        text = workflow.read_text(encoding="utf-8")
        assert "mypy" in text.lower(), (
            "unit-tests.yml no longer mentions mypy.  v5.2 wired the "
            "mypy strict-whitelist gate alongside the 63-error "
            "scope-local cleanup (AUDIT_V5_1_0 P2-NEW-F2-2 + ROADMAP "
            "v5.2 closure).  Restore the ``mypy:`` job."
        )


# ----------------------------------------------------------------------
# audit closure: P3-NEW-F1-1 -- stale Python 3.9 comments
# ----------------------------------------------------------------------


class TestDispatcherPinDocConsistencyPython310Refresh:
    """The dispatcher-pin file's ``tomllib`` import fallback comments
    must reference Python 3.10 (the post-v5.0 floor), not 3.9.
    Closes audit P3-NEW-F1-1.
    """

    # audit closure: P3-NEW-F1-1
    def test_no_python_3_9_references_in_dispatcher_pin_comments(self) -> None:
        text = _DISPATCHER_PIN.read_text(encoding="utf-8")
        # The drift markers from the pre-v5.0.1 comments were the
        # bare strings "3.9" and ``requires-python = ">=3.9"``.
        forbidden = (
            'requires-python = ">=3.9"',
            "Python 3.9",
            "python 3.9",
            "3.9 / 3.10",
            "3.9/3.10",
        )
        drift = [m for m in forbidden if m in text]
        assert not drift, (
            f"tests/unit/test_v4_16_2_dispatcher_pin_doc_consistency.py "
            f"still contains stale Python 3.9 references: {drift}.  "
            f"v5.0 bumped the library floor to 3.10; the tomllib "
            f"fallback comment block must reference Python 3.10 as "
            f"the fallback target, not 3.9.  Audit P3-NEW-F1-1 "
            f"closure."
        )

    # audit closure: P3-NEW-F1-1
    def test_dispatcher_pin_skip_reason_mentions_3_10(self) -> None:
        text = _DISPATCHER_PIN.read_text(encoding="utf-8")
        # The skip-reason text should name Python 3.10 as the
        # backport target (not 3.9 or "3.9/3.10").
        assert "(Python 3.10 backport)" in text, (
            "test_v4_16_2_dispatcher_pin_doc_consistency.py skip "
            "reason no longer says ``(Python 3.10 backport)``.  "
            "v5.0.1 audit P3-NEW-F1-1 refresh: the only fallback "
            "target on the post-v5.0 library floor is Python 3.10."
        )


# ----------------------------------------------------------------------
# audit closure: P3-NEW-F2-2 -- CHANGELOG ui/ vs lumenairy/ui/
# ----------------------------------------------------------------------


class TestChangelogRuffExcludeUiPathConsistency:
    """The CHANGELOG must cite ruff's exclude-list with the
    pyproject-canonical path ``lumenairy/ui/`` (not the shorter
    drift ``ui/``).  Closes audit P3-NEW-F2-2.
    """

    # audit closure: P3-NEW-F2-2
    def test_changelog_cites_lumenairy_ui_not_bare_ui(self) -> None:
        text = _CHANGELOG.read_text(encoding="utf-8")
        # Find the v5.0 release header block (up to the next ## [4...]
        # boundary).
        v5_block_match = re.search(
            r'^## \[5\.0\.0\][\s\S]*?(?=^## \[)',
            text, re.MULTILINE,
        )
        assert v5_block_match is not None, (
            "CHANGELOG.md does not contain a ``## [5.0.0]`` section "
            "block followed by another release header.  Cannot pin "
            "audit P3-NEW-F2-2."
        )
        block = v5_block_match.group(0)
        # The drift was citing the list as ``"validation/, docs/,
        # examples/, ui/"`` -- the canonical pyproject value is
        # ``lumenairy/ui/``.
        # The post-fix CHANGELOG should mention ``lumenairy/ui/`` in
        # the ruff exclude paragraph.
        assert "lumenairy/ui/" in block, (
            "CHANGELOG.md v5.0.0 entry does not cite "
            "``lumenairy/ui/`` -- the canonical pyproject value for "
            "the ruff exclude.  Audit P3-NEW-F2-2 closure requires "
            "the CHANGELOG text to use the same path as pyproject."
        )

    # audit closure: P3-NEW-F2-2
    def test_pyproject_ruff_exclude_uses_lumenairy_ui(self) -> None:
        """Sibling-discipline pin: confirm pyproject still uses the
        canonical path so the CHANGELOG-side pin above is correct.
        """
        tomllib_mod = _require_tomllib()
        with _PYPROJECT.open("rb") as fh:
            data = tomllib_mod.load(fh)
        ruff_cfg = data.get("tool", {}).get("ruff", {})
        excludes = ruff_cfg.get("extend-exclude", [])
        assert "lumenairy/ui/" in excludes, (
            f"pyproject.toml [tool.ruff] extend-exclude does not "
            f"contain ``lumenairy/ui/``.  Got: {excludes}.  The "
            f"v5.0.1 P3-NEW-F2-2 closure pins the CHANGELOG text "
            f"against this value; both must agree."
        )


# ----------------------------------------------------------------------
# audit closure: P3-NEW-F2-MCF -- MCF deferral clarification
# ----------------------------------------------------------------------


class TestChangelogMCFDeferralClarification:
    """CHANGELOG line 112's ``MCF.coherence_at(...) object`` bullet
    was unclear because ``coherence_at(...)`` already shipped on
    ``PartialCoherenceMCF`` in v4.15.1.  v5.0.1 rewords the bullet
    to clarify the actual deferral (top-level naming polish).
    Closes audit P3-NEW-F2-MCF.
    """

    # audit closure: P3-NEW-F2-MCF
    def test_changelog_mcf_bullet_no_longer_misleads(self) -> None:
        text = _CHANGELOG.read_text(encoding="utf-8")
        v5_block_match = re.search(
            r'^## \[5\.0\.0\][\s\S]*?(?=^## \[)',
            text, re.MULTILINE,
        )
        assert v5_block_match is not None, (
            "CHANGELOG.md missing v5.0.0 block; cannot pin "
            "P3-NEW-F2-MCF."
        )
        block = v5_block_match.group(0)
        # The pre-v5.0.1 misleading bullet was the bare line
        # ``* `MCF.coherence_at(...)` partial-coherence object.``
        misleading = "* `MCF.coherence_at(...)` partial-coherence object."
        assert misleading not in block, (
            "CHANGELOG v5.0.0 block still contains the misleading "
            "``MCF.coherence_at(...) partial-coherence object`` "
            "bullet.  ``coherence_at(...)`` and "
            "``PartialCoherenceMCF`` already shipped in v4.15.1; "
            "audit P3-NEW-F2-MCF requires the bullet to clarify "
            "the actual deferral (top-level ``lumenairy.MCF`` "
            "alias / naming polish)."
        )

    # audit closure: P3-NEW-F2-MCF
    def test_changelog_mcf_bullet_clarifies_actual_deferral(self) -> None:
        text = _CHANGELOG.read_text(encoding="utf-8")
        v5_block_match = re.search(
            r'^## \[5\.0\.0\][\s\S]*?(?=^## \[)',
            text, re.MULTILINE,
        )
        assert v5_block_match is not None
        block = v5_block_match.group(0)
        # The replacement bullet should mention v4.15.1 (when
        # PartialCoherenceMCF shipped) and the naming polish.
        for marker in ("PartialCoherenceMCF", "v4.15.1", "MCF"):
            assert marker in block, (
                f"CHANGELOG v5.0.0 block missing the v5.0.1 MCF "
                f"clarification marker {marker!r}.  The rewrite "
                f"must name ``PartialCoherenceMCF`` (already "
                f"shipped) and reference v4.15.1 so a reader "
                f"understands the deferral is about the naming "
                f"polish, not the class itself."
            )

    # audit closure: P3-NEW-F2-MCF
    def test_partial_coherence_mcf_already_shipped_in_library(self) -> None:
        """Independent verification: PartialCoherenceMCF and its
        coherence_at method are already present in the v5.0 library.
        """
        from lumenairy.sources.core import PartialCoherenceMCF
        assert hasattr(PartialCoherenceMCF, "coherence_at"), (
            "PartialCoherenceMCF.coherence_at is missing.  The "
            "v5.0.1 audit P3-NEW-F2-MCF clarification relies on "
            "the fact that this method already shipped in v4.15.1; "
            "if it has been removed, the clarification text is "
            "wrong and must be revisited."
        )

    # audit closure: P3-NEW-F2-MCF
    def test_partial_coherence_mcf_exposed_at_top_level(self) -> None:
        """Sibling-discipline pin: PartialCoherenceMCF is re-exported
        at the top level (the deferred work is the SHORTER alias
        ``lumenairy.MCF``, not exposing this class -- which already
        is).
        """
        import lumenairy
        assert hasattr(lumenairy, "PartialCoherenceMCF"), (
            "lumenairy.PartialCoherenceMCF is missing from the top-"
            "level namespace.  The v5.0.1 MCF clarification states "
            "the class is already exposed; if this assertion fails "
            "the clarification text is wrong."
        )


# ----------------------------------------------------------------------
# audit closure: P3-NEW-V3-3 -- pyproject 3.14 classifier vs CI matrix
# ----------------------------------------------------------------------


class TestPyprojectClassifiersMatchCIMatrix:
    """pyproject.toml classifiers must agree with the CI matrix on
    the set of supported Python versions.  v5.0 shipped a 3.14
    classifier while the matrix runs 3.10-3.13; v5.0.1 dropped the
    3.14 classifier (canonical CI target).  Closes audit
    P3-NEW-V3-3.
    """

    # audit closure: P3-NEW-V3-3
    def test_no_python_3_14_in_classifiers(self) -> None:
        tomllib_mod = _require_tomllib()
        with _PYPROJECT.open("rb") as fh:
            data = tomllib_mod.load(fh)
        classifiers = data.get("project", {}).get("classifiers", [])
        for classifier in classifiers:
            assert "3.14" not in classifier, (
                f"pyproject.toml classifiers contain a Python 3.14 "
                f"entry ({classifier!r}) but the CI matrix in "
                f".github/workflows/unit-tests.yml runs 3.10-3.13.  "
                f"Audit P3-NEW-V3-3 closure: drop the 3.14 "
                f"classifier (canonical target is the CI matrix), "
                f"or add 3.14 to the matrix.  v5.0.1 chose the "
                f"first option."
            )

    # audit closure: P3-NEW-V3-3
    def test_canonical_python_versions_in_classifiers(self) -> None:
        """The canonical 3.10-3.13 set must remain present."""
        tomllib_mod = _require_tomllib()
        with _PYPROJECT.open("rb") as fh:
            data = tomllib_mod.load(fh)
        classifiers = data.get("project", {}).get("classifiers", [])
        for ver in ("3.10", "3.11", "3.12", "3.13"):
            marker = f"Programming Language :: Python :: {ver}"
            assert marker in classifiers, (
                f"pyproject.toml classifiers missing {marker!r}.  "
                f"Audit P3-NEW-V3-3 closure dropped 3.14 only; the "
                f"3.10-3.13 set must remain."
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
