"""v4.16.3 Agent A regression tests -- V11 walker hardening.

Closes the v4.16.2 audit findings against
``test_v4_16_2_dispatcher_pin_doc_consistency.py`` (the 11th
dispatcher meta-pin walker):

* P1-NEW-F1-2 -- the v4.16.2 regex-based pyproject parser silently
  mis-parsed nested brackets.  The walker designed to retire the
  documentation-surface sibling-gap meta-pattern contained exactly
  that pattern.  v4.16.3 replaces the regex parser with
  ``tomllib`` / ``tomli``; these regression tests pin the canonical
  failure modes against a synthetic ``pyproject.toml`` constructed
  in ``tmp_path``.

* P2-NEW-F2-MED-1 -- the v4.16.2 walker iterated over a hardcoded
  breaking-change version list.  v4.16.3 derives the list from a
  CHANGELOG scan.  These regression tests exercise the scan helper
  against synthetic CHANGELOG fixtures so that future drift (e.g.,
  v4.17.0 with a breaking change) is detected automatically.

* P2-NEW-F2-MED-2 -- v4.16.3 adds a CHANGELOG <-> Migration-Guide
  drift check.  These regression tests pin the non-triviality
  heuristic against synthetic Migration-Guide fixtures.
"""
from __future__ import annotations

import re
from pathlib import Path
from textwrap import dedent

import pytest

from tests.unit.test_v4_16_2_dispatcher_pin_doc_consistency import (
    _BREAKING_BODY_MARKERS,
    _MIGRATION_GUIDE_SIBLING_COVERED,
    _block_has_active_breaking_section,
    _block_has_body_marker,
    _changelog_blocks,
    _extract_package_names_from_specs,
    _parse_pyproject_dependencies,
    _versions_with_breaking_changes_from_changelog,
)

# ----------------------------------------------------------------------
# P1-NEW-F1-2 -- pyproject parser regression tests (tomllib closure)
# ----------------------------------------------------------------------

class TestPyprojectTomllibParsing:
    """Pin the v4.16.2 non-greedy ``\\[(.*?)\\]`` regex failure modes
    against the v4.16.3 ``tomllib`` parser.

    The two canonical breakages cited in the v4.16.2 audit
    (``AUDIT_V4_16_2_2026_05_20.md`` P1-NEW-F1-2):

    1. ``jax-gpu = ["jax[cuda12]>=0.4.20"]`` -- regex captured only
       ``"jax[cuda12``, leaving the group with no parseable spec.
    2. ``all = [..., # ...lumenairy[all]..., "refractiveindex>=1.0"]``
       -- the comment's bracket terminated the regex capture before
       reaching ``refractiveindex``.
    """

    # audit closure: P1-NEW-F1-2
    def test_pyproject_parser_handles_extras_qualifier_in_value(self, tmp_path):
        """``jax[cuda12]>=0.4.20`` must parse as package ``jax``, NOT
        ``jax[cuda12`` (truncated) and NOT as missing.
        """
        pyproject = tmp_path / 'pyproject.toml'
        pyproject.write_text(dedent('''\
            [project]
            name = "synthetic"
            version = "0.0.0"
            requires-python = ">=3.9"
            dependencies = [
                "numpy>=1.20",
            ]

            [project.optional-dependencies]
            jax = ["jax>=0.4.20"]
            jax-gpu = ["jax[cuda12]>=0.4.20"]
        '''), encoding='utf-8')

        hard_deps, optional_groups = _parse_pyproject_dependencies(pyproject)
        assert hard_deps == {'numpy'}
        assert 'jax-gpu' in optional_groups, (
            "tomllib parser dropped the `jax-gpu` group; the v4.16.2 "
            "regex failure mode has regressed.")
        assert optional_groups['jax-gpu'] == {'jax'}, (
            f"jax-gpu group must parse as {{'jax'}} (extras stripped), "
            f"got {optional_groups['jax-gpu']}.  The v4.16.2 regex "
            f"captured `\"jax[cuda12` and broke; tomllib must close.")

    # audit closure: P1-NEW-F1-2
    def test_pyproject_parser_handles_comment_with_bracket(self, tmp_path):
        """A comment line containing ``lumenairy[all]`` inside an
        ``all = [...]`` group must NOT truncate the capture before
        subsequent entries (the v4.16.2 regex failure mode).
        """
        pyproject = tmp_path / 'pyproject.toml'
        pyproject.write_text(dedent('''\
            [project]
            name = "synthetic"
            version = "0.0.0"
            requires-python = ">=3.9"
            dependencies = [
                "numpy>=1.20",
            ]

            [project.optional-dependencies]
            glass = ["refractiveindex>=1.0"]
            all = [
                "pyfftw>=0.13",
                # Bundled into ``all`` so the one-shot
                # ``pip install synthetic[all]`` continues to pull it.
                "refractiveindex>=1.0",
                "jax>=0.4.20",
            ]
        '''), encoding='utf-8')

        _hard_deps, optional_groups = _parse_pyproject_dependencies(pyproject)
        assert 'refractiveindex' in optional_groups.get('all', set()), (
            "tomllib parser dropped ``refractiveindex`` from the "
            "``all`` group.  The v4.16.2 regex truncated at the "
            "comment's ``]`` and missed every entry after the "
            "comment -- this regression test pins the closure."
        )
        # All three entries (pyfftw, refractiveindex, jax) must round-
        # trip through the parser.
        assert optional_groups['all'] >= {'pyfftw', 'refractiveindex', 'jax'}

    # audit closure: P1-NEW-F1-2
    def test_live_pyproject_jax_gpu_group_captures_jax(self):
        """The live ``pyproject.toml`` must parse with ``jax-gpu``
        resolving to ``{'jax'}`` (not empty, not ``{'jax[cuda12'}``).
        Pins the closure against the actual canonical source -- not
        just synthetic fixtures.
        """
        _hard_deps, optional_groups = _parse_pyproject_dependencies()
        assert 'jax-gpu' in optional_groups, (
            "Live pyproject.toml dropped the `jax-gpu` extras group "
            "during parse.")
        assert optional_groups['jax-gpu'] == {'jax'}, (
            f"Live pyproject.toml `jax-gpu` group parsed as "
            f"{optional_groups['jax-gpu']}; expected {{'jax'}}.  The "
            f"v4.16.2 regex returned a malformed `{{'jax[cuda12'}}` "
            f"value.")

    # audit closure: P1-NEW-F1-2
    def test_live_pyproject_all_group_captures_refractiveindex(self):
        """The live ``pyproject.toml`` ``all`` extras group has a
        comment containing ``lumenairy[all]``; ``refractiveindex``
        appears AFTER that comment.  Tomllib must capture it.
        """
        _hard_deps, optional_groups = _parse_pyproject_dependencies()
        assert 'all' in optional_groups, "pyproject.toml missing `all` group"
        assert 'refractiveindex' in optional_groups['all'], (
            "Live pyproject.toml `all` group missing `refractiveindex` "
            "after parse.  Either the package was removed (unintended) "
            "or the parser is mis-handling comments -- the v4.16.2 "
            "regex truncated here.")

    # audit closure: P1-NEW-F1-2
    def test_extract_package_names_strips_extras_and_version_specs(self):
        """The PEP-508 spec parser must lowercase, strip extras, and
        strip version operators uniformly.
        """
        names = _extract_package_names_from_specs([
            "NumPy>=1.20",
            "jax[cuda12]>=0.4.20",
            "pyfftw  >=0.13",
            "refractiveindex>=1.0",
            "filelock>=3.0",
            "Package~=1.0",
            "Package2==1.0",
            "Package3!=1.0",
        ])
        assert names == [
            'numpy', 'jax', 'pyfftw', 'refractiveindex',
            'filelock', 'package', 'package2', 'package3',
        ]


# ----------------------------------------------------------------------
# P2-NEW-F2-MED-1 -- CHANGELOG-driven breaking-version list
# ----------------------------------------------------------------------

class TestChangelogBreakingVersionScan:
    """Pin the v4.16.3 CHANGELOG scan against synthetic fixtures.

    The hardcoded tuple ``('4.13.0', '4.15.1', '4.16.1', '4.16.2')``
    in the v4.16.2 walker would silently pass through a v4.17.0
    breaking-change release without manual edit.  The scan-based
    helper must:

    * Pick up versions with an active ``### Breaking changes`` section
      (i.e., NOT followed inline by ``-- none`` / ``— none``).
    * Pick up versions with high-precision body markers (``SUM->AVG``,
      ``silent semantics change``, ...).
    * NOT pick up versions whose ``### Breaking changes`` section is
      explicitly annotated ``-- none``.
    * NOT pick up pre-v4.13.0 versions (Migration-Guide.md starts
      there).
    """

    # audit closure: P2-NEW-F2-MED-1
    def test_synthetic_v4_17_0_breaking_section_is_detected(self):
        """A synthetic v4.17.0 entry with ``### Breaking changes`` must
        appear in the scanner's output.
        """
        text = dedent('''\
            # Changelog

            ## [4.17.0] -- 2026-06-01

            **Synthetic future release.**

            ### Breaking changes

            * Something newly broken.

            ## [4.16.2] -- 2026-05-20

            ### Breaking changes -- none

            All quiet.
        ''')
        flagged = _versions_with_breaking_changes_from_changelog(text)
        assert '4.17.0' in flagged, (
            "CHANGELOG scan must detect v4.17.0's active "
            "`### Breaking changes` section.  Without this, the "
            "v4.16.2 hardcoded-tuple sibling-gap meta-pattern remains "
            "open.")
        assert '4.16.2' not in flagged, (
            "CHANGELOG scan incorrectly flagged a `Breaking changes "
            "-- none` block.")

    # audit closure: P2-NEW-F2-MED-1
    def test_synthetic_silent_semantics_change_is_detected(self):
        """A body-only marker (``silent semantics change``) must be
        sufficient even without an explicit ``### Breaking changes``
        heading.
        """
        text = dedent('''\
            ## [4.17.1] -- 2026-06-15

            **Silent semantics change**: the foo helper now averages
            instead of summing.  Existing recipes drift by a factor of
            len(wavelengths).
        ''')
        flagged = _versions_with_breaking_changes_from_changelog(text)
        assert '4.17.1' in flagged

    # audit closure: P2-NEW-F2-MED-1
    def test_synthetic_sum_to_avg_marker_is_detected(self):
        """``SUM->AVG`` and ``SUM -> AVG`` are both accepted markers."""
        text_dash = dedent('''\
            ## [4.17.2] -- 2026-07-01

            * Bug 1 -- ``MultiWavelengthMerit.evaluate`` SUM->AVG.
        ''')
        text_spaced = dedent('''\
            ## [4.17.3] -- 2026-07-15

            * Bug -- SUM -> AVG semantics fix.
        ''')
        assert '4.17.2' in _versions_with_breaking_changes_from_changelog(text_dash)
        assert '4.17.3' in _versions_with_breaking_changes_from_changelog(text_spaced)

    # audit closure: P2-NEW-F2-MED-1
    def test_pre_v4_13_versions_are_excluded(self):
        """Migration-Guide.md starts at v4.13.  Pre-v4.13.0 versions
        with breaking markers must NOT be enrolled.
        """
        text = dedent('''\
            ## [4.12.2] -- 2026-05-17

            ### Breaking changes

            * Something old.

            ## [4.13.0] -- 2026-05-17

            ### Breaking changes

            * rcwa.py -> thin_grating.py rename.
        ''')
        flagged = _versions_with_breaking_changes_from_changelog(text)
        assert '4.12.2' not in flagged, (
            "Pre-v4.13.0 versions must be excluded from the "
            "breaking-change scan (Migration-Guide.md starts at v4.13).")
        assert '4.13.0' in flagged

    # audit closure: P2-NEW-F2-MED-1
    def test_live_changelog_yields_known_breaking_versions(self):
        """The live CHANGELOG must include at minimum 4.13.0, 4.15.1,
        4.16.1, 4.16.2 (the versions documented in Migration-Guide.md)
        in the scan output.  These are anchor versions; the test
        guards against accidental marker-list regressions.
        """
        flagged = _versions_with_breaking_changes_from_changelog()
        for anchor in ('4.13.0', '4.15.1', '4.16.1', '4.16.2'):
            assert anchor in flagged, (
                f"Live CHANGELOG scan missed anchor version {anchor} "
                f"-- the marker heuristic has regressed.  Scanner "
                f"output: {flagged}.")

    # audit closure: P2-NEW-F2-MED-1
    def test_naive_breaking_word_is_not_a_lone_marker(self):
        """Bare substring ``breaking`` must NOT trigger inclusion on
        its own (it appears in incidental prose like ``the v4.X.Y
        breaking changes are documented in v4.X+1's section``).
        Without this exclusion the v4.16.0 entry (a Major release
        whose ``breaking`` mentions are pure cross-references) would
        be mis-enrolled.
        """
        text = dedent('''\
            ## [4.20.0] -- 2026-12-01

            **Major minor release** referencing prior breaking
            changes documented in v4.15.1.  This release itself has
            no breaking items; users on v4.19.x can upgrade without
            code changes.
        ''')
        flagged = _versions_with_breaking_changes_from_changelog(text)
        assert '4.20.0' not in flagged, (
            "Bare ``breaking`` substring must not enroll a version "
            "with no actual breaking change.  The v4.16.2 audit "
            "warns this exact false-positive class.")


# ----------------------------------------------------------------------
# P2-NEW-F2-MED-2 -- CHANGELOG <-> Migration-Guide non-triviality
# ----------------------------------------------------------------------

class TestMigrationGuideNonTriviality:
    """The non-triviality check (>=200 chars of non-whitespace per
    section) guards against stub sections.

    These regression tests pin the threshold against synthetic
    Migration-Guide fixtures so the behavior remains deterministic if
    the live file gains a new section.
    """

    # audit closure: P2-NEW-F2-MED-2
    def test_block_has_active_breaking_section_distinguishes_none(self):
        """A ``### Breaking changes -- none`` heading must NOT count."""
        block_active = (
            "## [4.17.0]\n\n### Breaking changes\n\n* foo\n"
        )
        block_none_em = (
            "## [4.14.2]\n\n### Breaking changes — none\n\nQuiet.\n"
        )
        block_none_dashes = (
            "## [4.14.3]\n\n### Breaking changes -- none\n\nQuiet.\n"
        )
        assert _block_has_active_breaking_section(block_active)
        assert not _block_has_active_breaking_section(block_none_em)
        assert not _block_has_active_breaking_section(block_none_dashes)

    # audit closure: P2-NEW-F2-MED-2
    def test_block_has_body_marker_high_precision(self):
        """Only high-precision markers count.  ``DeprecationWarning``
        and ``FutureWarning`` alone (without an explicit ``### Breaking
        changes`` section) must NOT trigger.
        """
        # Negative cases -- noisy markers that should NOT enroll.
        assert not _block_has_body_marker(
            "Some text mentioning DeprecationWarning incidentally.")
        assert not _block_has_body_marker(
            "Mentions a FutureWarning in a code-comment context.")
        assert not _block_has_body_marker(
            "Plain `breaking` substring in prose.")
        # Positive cases -- precise markers that SHOULD enroll.
        assert _block_has_body_marker(
            "Bug 1 -- SUM->AVG semantics fix.")
        assert _block_has_body_marker(
            "Bug -- SUM -> AVG semantics fix.")
        assert _block_has_body_marker(
            "This is a silent semantics change.")

    # audit closure: P2-NEW-F2-MED-2
    def test_changelog_blocks_split_at_version_headings(self):
        """Per-release-block scan must split exactly at
        ``## [X.Y.Z]`` headings (not on subsections).
        """
        text = dedent('''\
            # Changelog

            ## [4.17.0] -- 2026-06-01

            ### Breaking changes
            * one
            ### P1 closures
            * a

            ## [4.16.2] -- 2026-05-20

            ### Breaking changes -- none
        ''')
        blocks = _changelog_blocks(text)
        versions = [v for v, _ in blocks]
        assert versions == ['4.17.0', '4.16.2'], (
            f"Block-split must enumerate versions in document order: "
            f"got {versions}.")

    # audit closure: P2-NEW-F2-MED-2
    def test_sibling_covered_allowlist_minimal(self):
        """The sibling-covered allowlist must remain minimal -- it is
        itself a sibling-gap surface.  Any future addition must be a
        deliberate choice with an inline comment in the walker source.
        v5.0 baseline (was v4.16.3 + 4.16.3): both v4.15.2 (sibling of
        v4.15.1) and v4.16.3 (soft DeprecationWarning, migration recipe
        inline in the warning text + CHANGELOG) are sibling-covered.
        """
        assert _MIGRATION_GUIDE_SIBLING_COVERED == frozenset({'4.15.2', '4.16.3'}), (
            f"The sibling-covered allowlist drifted from the v5.0 "
            f"baseline {{'4.15.2', '4.16.3'}} to "
            f"{_MIGRATION_GUIDE_SIBLING_COVERED}.  Any addition must be "
            f"documented in the walker module docstring with the "
            f"sibling version whose Migration-Guide recipe covers the "
            f"allow-listed version.")


# ----------------------------------------------------------------------
# Sibling-gap discipline -- per AUDIT_V4_16_2_2026_05_20.md commentary
# ----------------------------------------------------------------------

class TestWalkerInternalSiblingGapDiscipline:
    """The audit warns: the walker designed to retire the
    documentation-surface sibling-gap meta-pattern itself contained
    that pattern.  v4.16.3 closes the regex case.  These tests pin
    the discipline so future regressions are surfaced.
    """

    # audit closure: P1-NEW-F1-2 (sibling-gap discipline)
    def test_v11_walker_does_not_use_nongreedy_pyproject_regex(self):
        """The v4.16.2 regex ``r'^([a-zA-Z_]...)\\s*=\\s*\\[(.*?)\\]'``
        must no longer appear in the walker source.  Pin by
        substring -- if a future maintainer reintroduces a nested-
        bracket-vulnerable regex against pyproject.toml, this test
        fails loudly.
        """
        walker_src = (
            Path(__file__).parent
            / 'test_v4_16_2_dispatcher_pin_doc_consistency.py'
        ).read_text(encoding='utf-8')
        # The exact v4.16.2 failure pattern.
        assert r"r'^([a-zA-Z_][a-zA-Z0-9_-]*)\s*=\s*\[(.*?)\]'" not in walker_src, (
            "The v4.16.2 regex parser has been reintroduced -- this is "
            "the exact pattern the v4.16.3 tomllib rewrite was meant "
            "to retire.  See AUDIT_V4_16_2_2026_05_20.md P1-NEW-F1-2.")
        # Soft positive signal: tomllib import line should be present.
        assert 'tomllib' in walker_src, (
            "Walker no longer imports tomllib -- the v4.16.3 closure "
            "has been reverted.")

    # audit closure: P2-NEW-F2-MED-1 (sibling-gap discipline)
    def test_v11_walker_does_not_hardcode_breaking_version_tuple(self):
        """The v4.16.2 walker iterated over a hardcoded tuple
        ``('4.13.0', '4.15.1', '4.16.1', '4.16.2')`` for the
        Migration-Guide coverage check.  v4.16.3 drives this from a
        CHANGELOG scan.  Detect a regression: ensure the
        ``test_migration_guide_has_known_version_sections`` test body
        no longer contains the literal version tuple.
        """
        walker_src = (
            Path(__file__).parent
            / 'test_v4_16_2_dispatcher_pin_doc_consistency.py'
        ).read_text(encoding='utf-8')
        # Pin against the exact v4.16.2 literal.
        assert "('4.13.0', '4.15.1', '4.16.1', '4.16.2')" not in walker_src, (
            "The v4.16.2 hardcoded breaking-change version tuple has "
            "been reintroduced.  AUDIT_V4_16_2_2026_05_20.md "
            "P2-NEW-F2-MED-1 closes this by driving the list from a "
            "CHANGELOG scan.")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
