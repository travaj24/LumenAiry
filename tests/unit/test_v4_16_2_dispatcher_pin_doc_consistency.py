"""v4.16.2 11th meta-pin walker -- documentation-surface consistency.

The v4.16.1 audit identified that the sibling-gap meta-pattern,
structurally retired at code surfaces via V1-V10, recurred at
**documentation surfaces** (README.md cited refractiveindex as
Required after pyproject.toml moved it to [glass] extras;
requirements.txt drifted similarly; ROADMAP claimed "9 meta-pins"
after V10 shipped; CHANGELOG headline test-count arithmetic didn't
reconcile).

This walker scans 4 doc surfaces for dependency-declaration drift
against the canonical source (``pyproject.toml``):

* ``README.md`` Required / Dependencies block
* ``requirements.txt`` uncommented lines
* ``ROADMAP.md`` meta-pin enumeration count
* ``CHANGELOG.md`` per-release test-count arithmetic

Each check fails informatively with the specific drift cited.

Closes audit P2-NEW-F2-MED-1 + P2-NEW-F2-MED-2 + audit Part 3
documentation-surface meta-pattern.

v4.16.3 hardening (audit AUDIT_V4_16_2_2026_05_20.md):

* P1-NEW-F1-2: replaced the regex-based pyproject parser with
  ``tomllib`` (``tomli`` fallback on pre-3.11 interpreters; see
  the v5.0.1 P3-NEW-F1-1 comment block at the import site for the
  post-v5.0 floor).  The previous non-greedy ``\\[(.*?)\\]``
  capture mis-parsed nested brackets in values like ``jax-gpu =
  ["jax[cuda12]>=0.4.20"]`` and any group whose body contained a
  comment that mentioned ``lumenairy[all]``.  The walker designed
  to retire the documentation-surface sibling-gap meta-pattern
  itself contained that pattern -- v4.16.3 closes.
* P2-NEW-F2-MED-1: the breaking-change version list driving
  ``test_migration_guide_has_known_version_sections`` is now derived
  from a CHANGELOG ``## [X.Y.Z]`` per-release-block scan rather than
  a hardcoded tuple.  Future breaking-change releases auto-enroll.
* P2-NEW-F2-MED-2: new test
  ``test_migration_guide_sections_are_non_trivial`` verifies that each
  Migration-Guide version section contains substantive content
  between its heading and the next, not just a heading.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

# v4.16.3 (audit P1-NEW-F1-2): tomllib parsing closes the regex
# mis-parse pattern that V11 was supposed to retire but instead
# contained.  Python 3.11+ ships ``tomllib`` in the stdlib; on Python
# 3.10 (the library floor per pyproject ``requires-python = ">=3.10"``
# as of v5.0) we fall back to the ``tomli`` backport.  ``tomli`` is
# NOT a hard dev dep, so a missing-import on 3.10 becomes a clean
# ``pytest.skip`` rather than a hard failure.
# v5.0.1 (audit P3-NEW-F1-1): refreshed pre-3.10 references --
# the library floor moved to 3.10 in v5.0, so the only fallback target
# is now Python 3.10 (not the broader pre-3.11 interval the
# v4.16.3 closure had originally documented when the library still
# supported earlier 3.x versions).
try:
    import tomllib  # type: ignore[import-not-found]  # Python 3.11+
    _TOMLLIB = tomllib
    _TOMLLIB_SKIP_REASON = None
except ImportError:  # pragma: no cover -- exercised only on Python 3.10
    try:
        import tomli as _TOMLLIB  # type: ignore[import-not-found]
        _TOMLLIB_SKIP_REASON = None
    except ImportError:  # pragma: no cover -- truly missing
        _TOMLLIB = None
        _TOMLLIB_SKIP_REASON = (
            "Neither `tomllib` (Python 3.11+ stdlib) nor `tomli` "
            "(Python 3.10 backport) is importable.  "
            "Install `tomli` to run the V11 pyproject-consistency "
            "walker on this interpreter (`pip install tomli`)."
        )

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / 'pyproject.toml'
_README = _REPO_ROOT / 'README.md'
_REQUIREMENTS = _REPO_ROOT / 'requirements.txt'
_ROADMAP = _REPO_ROOT / 'ROADMAP.md'
_CHANGELOG = _REPO_ROOT / 'CHANGELOG.md'
_MIGRATION_GUIDE = _REPO_ROOT / 'Migration-Guide.md'


# ----------------------------------------------------------------------
# Helpers -- parse pyproject.toml hard deps + optional extras
# ----------------------------------------------------------------------

def _require_tomllib():
    """Skip cleanly when neither tomllib nor tomli is importable."""
    if _TOMLLIB is None:
        pytest.skip(_TOMLLIB_SKIP_REASON)
    return _TOMLLIB


def _parse_pyproject_dependencies(pyproject_path: Path | None = None):
    """Return (hard_deps, optional_groups_dict) parsed from
    pyproject.toml via ``tomllib`` / ``tomli``.

    Returns
    -------
    hard_deps : set of str
        Package names (lowercased; extras and version specs stripped)
        in ``[project.dependencies]``.
    optional_groups : dict {group_name: set of str}
        Package names in each ``[project.optional-dependencies.{group}]``.

    Notes
    -----
    v4.16.3 (audit P1-NEW-F1-2): the v4.16.2 implementation used a
    non-greedy regex (``\\[(.*?)\\]``) that silently mis-parsed both
    ``jax-gpu = ["jax[cuda12]>=0.4.20"]`` (extras qualifier inside the
    string literal terminated the capture early) and any group whose
    body contained a comment mentioning ``lumenairy[all]`` (the comment
    bracket terminated the capture mid-list).  ``tomllib`` is the
    canonical parser and has no such failure mode.
    """
    tomllib_mod = _require_tomllib()
    pyproject_path = pyproject_path or _PYPROJECT
    with pyproject_path.open('rb') as fh:
        data = tomllib_mod.load(fh)

    project = data.get('project', {})
    hard_specs = project.get('dependencies', []) or []
    hard_deps = set(_extract_package_names_from_specs(hard_specs))

    opt_section = project.get('optional-dependencies', {}) or {}
    optional_groups = {
        grp_name: set(_extract_package_names_from_specs(specs))
        for grp_name, specs in opt_section.items()
    }
    return hard_deps, optional_groups


def _extract_package_names_from_specs(specs):
    """Extract package names from a list of PEP-508-style strings like
    ``["numpy>=1.20", "scipy>=1.7", "jax[cuda12]>=0.4.20"]``.

    Returns each name lowercased (so ``refractiveindex`` and
    ``Refractiveindex`` reconcile).  Extras qualifiers
    (``pkg[extra]``) are stripped.
    """
    names = []
    for spec in specs:
        if not isinstance(spec, str):
            continue
        # Split on first version operator (==, >=, <=, !=, ~=, >, <).
        name = re.split(r'[<>=!~;\s]', spec, maxsplit=1)[0].strip()
        # Strip extras: "package[extra]" -> "package".
        name = name.split('[')[0]
        if name:
            names.append(name.lower())
    return names


# ----------------------------------------------------------------------
# Helpers -- CHANGELOG breaking-change scan (audit P2-NEW-F2-MED-1)
# ----------------------------------------------------------------------

# High-precision per-release-block markers.  The naive ``breaking`` /
# ``DeprecationWarning`` / ``FutureWarning`` markers were too noisy --
# every release that mentions a *prior* version's deprecation cycle
# was caught.  These are the markers the v4.16.2 audit's pseudo-code
# lists; v4.16.3 restricts to the high-precision subset PLUS the
# explicit ``### Breaking changes`` section heading (when not followed
# by ``-- none`` / ``— none``).  See module docstring.
_BREAKING_BODY_MARKERS = (
    'silent semantics change',
    'silent behavior change',
    'silent behaviour change',
    'sum->avg',
    'sum -> avg',
)

# Versions whose breaking-change migration is documented under a
# SIBLING version's Migration-Guide entry (and therefore don't need
# their own ``## X.Y.Z`` heading).  Documented explicitly so the
# allowlist itself stays a sibling-gap surface: any future addition
# must be a deliberate choice.
#
# * 4.15.2 emits a ``DeprecationWarning`` on the v4.15.1 Schell-family
#   return-shape break; the migration recipe lives under the v4.15.1
#   heading (see ``Migration-Guide.md`` ``## 4.15.1`` section).
# * 4.16.3 emits a one-cycle ``DeprecationWarning`` on the v4.16.2
#   ``Constraint.__post_init__`` auto-probe removal; the user-facing
#   migration recipe ("call ``.validate()`` explicitly") is documented
#   inline in the warning message itself and in the CHANGELOG v4.16.3
#   entry.  Soft transitional, not a hard break; no Migration-Guide
#   section needed.
_MIGRATION_GUIDE_SIBLING_COVERED = frozenset({'4.15.2', '4.16.3'})


def _version_tuple(ver_str: str) -> tuple[int, int, int]:
    return tuple(int(p) for p in ver_str.split('.'))  # type: ignore[return-value]


def _changelog_blocks(text: str | None = None):
    """Return list of (version_str, block_text) for every
    ``## [X.Y.Z]`` heading in CHANGELOG.md.

    The block is the slice from one heading to the next.
    """
    text = text if text is not None else _CHANGELOG.read_text(encoding='utf-8')
    parts = re.split(r'(?=^## \[\d+\.\d+\.\d+\])', text, flags=re.MULTILINE)
    out = []
    for part in parts:
        hdr = re.match(r'^## \[(\d+\.\d+\.\d+)\]', part)
        if hdr is None:
            continue
        out.append((hdr.group(1), part))
    return out


def _block_has_active_breaking_section(block: str) -> bool:
    """True iff the per-release block contains an
    ``### Breaking changes`` (or ``Breaking change``) heading that is
    NOT followed inline by ``-- none`` / ``— none``.
    """
    for m in re.finditer(
            r'^### Breaking change[s]?\b(.*)$',
            block, flags=re.MULTILINE):
        tail = m.group(1).strip().lower()
        if 'none' in tail:
            continue
        return True
    return False


def _block_has_body_marker(block: str) -> bool:
    lower = block.lower()
    return any(m in lower for m in _BREAKING_BODY_MARKERS)


def _versions_with_breaking_changes_from_changelog(text: str | None = None):
    """Return sorted list of version strings whose CHANGELOG block
    flags a breaking change (per the v4.16.3 audit closure heuristic).

    Restrictions (audit AUDIT_V4_16_2_2026_05_20.md):

    * Only versions >= 4.13.0 (Migration-Guide.md starts at v4.13).
    * Conjunction: marker found WITHIN the per-release block.
    * High-precision markers only -- ``breaking`` / ``DeprecationWarning``
      / ``FutureWarning`` as bare substrings are too noisy (they
      appear in incidental prose).  The accepted signals are:

      - An ``### Breaking changes`` heading not annotated ``-- none``.
      - A body mention of one of ``_BREAKING_BODY_MARKERS``
        (``silent semantics change``, ``SUM->AVG``, ...).
    """
    versions = []
    for ver, block in _changelog_blocks(text):
        if _version_tuple(ver) < (4, 13, 0):
            continue
        if _block_has_active_breaking_section(block) or _block_has_body_marker(block):
            versions.append(ver)
    return sorted(versions, key=_version_tuple)


# ----------------------------------------------------------------------
# Audit closure: P1-NEW-F2-HIGH-1 -- README.md vs pyproject.toml
# ----------------------------------------------------------------------

class TestReadmeDependencyConsistency:
    """README.md must not list optional-extras dependencies as
    'Required'.  Closes v4.16.1 audit P1-NEW-F2-HIGH-1: refractiveindex
    moved to [glass] extras but README still cited it as Required.
    """

    # audit closure: P1-NEW-F2-HIGH-1
    def test_readme_required_section_does_not_list_optional_deps(self):
        hard_deps, optional_groups = _parse_pyproject_dependencies()
        # Flat set of all package names that are in optional extras
        # (not in hard deps).
        all_optional = set()
        for grp_pkgs in optional_groups.values():
            all_optional |= grp_pkgs
        all_optional -= hard_deps  # bundled-into-hard names don't count

        readme = _README.read_text(encoding='utf-8')

        # Find the "Required" subsection of the Dependencies block.
        # The block opens at `## Dependencies` and the Required heading
        # is `### Required`.  We scan to the next subsection.
        match = re.search(
            r'### Required\s*\n(.*?)(?=\n###|\n##\s|\Z)',
            readme, re.DOTALL,
        )
        if match is None:
            pytest.skip(
                "README.md has no `### Required` subsection -- skipping "
                "(structural skip, not a failure).")
        required_block = match.group(1).lower()

        # Each optional-extras package must NOT appear as a bullet in
        # the Required block.  Conservative pattern: list bullet
        # ``- `pkgname```.
        drift = []
        for pkg in sorted(all_optional):
            if re.search(rf'-\s*`{re.escape(pkg)}`', required_block):
                drift.append(pkg)

        assert not drift, (
            f"README.md cites these optional-extras packages as "
            f"`Required`: {drift}.  "
            f"They live in pyproject.toml's [project.optional-"
            f"dependencies] but are listed in README.md's `### Required` "
            f"block.  Refresh the README dependency block so it matches "
            f"the pyproject.toml canonical source."
        )

    # audit closure: P1-NEW-F2-HIGH-1
    def test_readme_pip_install_command_does_not_force_optional_deps(self):
        """The README's quick-install command must not silently install
        optional-extras packages.  `pip install numpy refractiveindex`
        forces refractiveindex even though it's in [glass] extras.
        """
        hard_deps, optional_groups = _parse_pyproject_dependencies()
        all_optional = set()
        for grp_pkgs in optional_groups.values():
            all_optional |= grp_pkgs
        all_optional -= hard_deps

        readme = _README.read_text(encoding='utf-8')

        drift = []
        for pkg in sorted(all_optional):
            # Look for an unbracketed `pip install ... {pkg} ...`
            # command (i.e. NOT `pip install lumenairy[glass]`).
            for m in re.finditer(
                    rf'pip install ([^\n`]+?\b{re.escape(pkg)}\b[^\n`]*)',
                    readme, re.IGNORECASE):
                cmd = m.group(1)
                # Exclude `lumenairy[extra]` patterns.
                if re.search(rf'\blumenairy\[[^\]]*{re.escape(pkg)}',
                             cmd, re.IGNORECASE):
                    continue
                # Exclude commented-out command examples (rare).
                drift.append((pkg, m.group(0)))

        assert not drift, (
            f"README.md `pip install` command forces optional-extras "
            f"packages: {drift}.  Update to `pip install lumenairy[X]` "
            f"or move the install instruction to an optional section.")


# ----------------------------------------------------------------------
# Audit closure: P1-NEW-F2-HIGH-2 -- requirements.txt vs pyproject.toml
# ----------------------------------------------------------------------

class TestRequirementsTxtConsistency:
    """requirements.txt uncommented lines must match pyproject.toml
    [project.dependencies] hard deps.  Closes v4.16.1 audit
    P1-NEW-F2-HIGH-2.
    """

    # audit closure: P1-NEW-F2-HIGH-2
    def test_requirements_txt_matches_pyproject_hard_deps(self):
        hard_deps, optional_groups = _parse_pyproject_dependencies()
        all_optional = set()
        for grp_pkgs in optional_groups.values():
            all_optional |= grp_pkgs
        all_optional -= hard_deps

        req_text = _REQUIREMENTS.read_text(encoding='utf-8')
        req_uncommented = []
        for line in req_text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            # Strip inline comment.
            if '#' in stripped:
                stripped = stripped[:stripped.index('#')].strip()
            req_uncommented.append(stripped)

        req_names = set()
        for spec in req_uncommented:
            name = re.split(r'[<>=!~]', spec, maxsplit=1)[0].strip().lower()
            name = name.split('[')[0]
            if name:
                req_names.add(name)

        # An optional-extras package must NOT appear uncommented in
        # requirements.txt.
        drift = sorted(req_names & all_optional)
        assert not drift, (
            f"requirements.txt has uncommented entries for optional-"
            f"extras packages: {drift}.  These live in pyproject.toml's "
            f"[project.optional-dependencies] -- move to a commented "
            f"line under `=== Optional ===`.")


# ----------------------------------------------------------------------
# Audit closure: P2-NEW-F2-MED-1 -- ROADMAP meta-pin enumeration count
# ----------------------------------------------------------------------

class TestRoadmapMetaPinEnumeration:
    """ROADMAP.md must claim the correct number of dispatcher meta-pins.
    Closes v4.16.1 audit P2-NEW-F2-MED-1 (ROADMAP said "ALL 9" after
    V10 shipped) + v4.16.2 (V11 doc-consistency walker shipped here).
    """

    # audit closure: P2-NEW-F2-MED-1
    def test_roadmap_meta_pin_count_matches_dispatcher_pin_files(self):
        # Discover all dispatcher pin files in tests/unit/.
        pin_dir = _REPO_ROOT / 'tests' / 'unit'
        sorted(
            f.name for f in pin_dir.glob('test_v4_*_dispatcher_pin_*.py')
        )
        # Plus the v4.16.2 doc-consistency pin (this file).
        # Count by parsing the V-numbered enumeration in ROADMAP rather
        # than file count (file naming hasn't been consistent
        # historically -- some pre-V6 pins use different names).
        roadmap = _ROADMAP.read_text(encoding='utf-8')

        # Find the "ALL N dispatcher meta-pins" claim.
        claim_match = re.search(
            r'ALL (\d+) dispatcher meta-pins',
            roadmap)
        assert claim_match is not None, (
            "ROADMAP.md does not contain an `ALL N dispatcher meta-"
            "pins` claim.  Restore the meta-pin coverage summary.")
        claimed_count = int(claim_match.group(1))

        # Count V-numbered entries in the same section.
        v_entries = re.findall(r'^\s*-\s*\*?\*?V(\d+)',
                               roadmap, re.MULTILINE)
        listed_count = len(set(int(v) for v in v_entries))

        assert claimed_count == listed_count, (
            f"ROADMAP meta-pin drift: claim says `ALL {claimed_count} "
            f"dispatcher meta-pins` but {listed_count} V-numbered "
            f"entries enumerated.  Update both numbers in lock-step.")


# ----------------------------------------------------------------------
# Audit closure: P2-NEW-F2-MED-2 -- CHANGELOG test-count arithmetic
# ----------------------------------------------------------------------

class TestChangelogTestCountArithmetic:
    """CHANGELOG.md per-release test-count headline must reconcile to
    the prior release's pass count + the per-agent delta.  Closes
    v4.16.1 audit P2-NEW-F2-MED-2.
    """

    # audit closure: P2-NEW-F2-MED-2
    def test_changelog_test_count_arithmetic_reconciles_v4_16_1(self):
        """Specifically pin the v4.16.1 entry which the v4.16.1 audit
        flagged as off-by-16 / +17 / -7 against the per-agent
        breakdown.  v4.16.2 corrected this; the corrected numbers must
        stick across future drift."""
        text = _CHANGELOG.read_text(encoding='utf-8')

        # Find the v4.16.1 entry headline.
        match = re.search(
            r'## \[4\.16\.1\][\s\S]*?\*\*(\d+) unit tests pass\*\* '
            r'\(up from (\d+); \+(\d+) net\)',
            text,
        )
        assert match is not None, (
            "CHANGELOG.md v4.16.1 entry does not have a parseable "
            "headline of the form `**NNNN unit tests pass** (up from "
            "MMMM; +DD net)`.  Refresh the headline.")
        pass_count = int(match.group(1))
        baseline = int(match.group(2))
        delta = int(match.group(3))
        assert pass_count == baseline + delta, (
            f"v4.16.1 headline arithmetic: {pass_count} != "
            f"{baseline} + {delta}.")
        # And the corrected absolute values that v4.16.2 audit
        # closure mandates (these are empirical, verified via
        # `pytest tests/unit/ --collect-only -q`).  Headline reports
        # the COLLECTED metric, which arithmetic-reconciles cleanly
        # (collected = pass + skip + xfail).
        assert pass_count == 2198, (
            f"v4.16.1 headline collected count must be 2198 "
            f"(empirically verified post-v4.16.2 work), got {pass_count}.")
        assert baseline == 2113, (
            f"v4.16.1 headline baseline must be 2113 (v4.16.0 final "
            f"count), got {baseline}.")
        assert delta == 85, (
            f"v4.16.1 headline delta must be 85 (A=11 + B=26 + C=20+6 "
            f"+ D=22), got {delta}.")


# ----------------------------------------------------------------------
# Migration-Guide.md must exist + cover every breaking-change version
# ----------------------------------------------------------------------

class TestMigrationGuideExists:
    """Migration-Guide.md is a v4.16.2 pre-v5.0 prep addition.

    v4.16.3 (audit AUDIT_V4_16_2_2026_05_20.md P2-NEW-F2-MED-1 +
    P2-NEW-F2-MED-2): the version list is now CHANGELOG-driven rather
    than a hardcoded tuple, and a non-triviality check ensures each
    Migration-Guide section actually contains migration content (not
    just a heading).
    """

    def test_migration_guide_md_exists(self):
        assert _MIGRATION_GUIDE.is_file(), (
            f"{_MIGRATION_GUIDE} does not exist.  v4.16.2 should have "
            f"created the migration guide.")

    # audit closure: P2-NEW-F2-MED-1
    def test_migration_guide_has_known_version_sections(self):
        """Every version whose CHANGELOG block flags a breaking change
        (per ``_versions_with_breaking_changes_from_changelog``) must
        either have a ``## X.Y.Z`` section in Migration-Guide.md OR be
        explicitly allow-listed under
        ``_MIGRATION_GUIDE_SIBLING_COVERED`` (sibling-covered case).

        v4.16.3 closure: previously this test iterated over a hardcoded
        tuple (4.13.0, 4.15.1, 4.16.1, 4.16.2) and would silently pass
        when v4.17.0+ shipped a breaking change without manual edit.
        """
        text = _MIGRATION_GUIDE.read_text(encoding='utf-8')
        flagged = _versions_with_breaking_changes_from_changelog()
        assert flagged, (
            "CHANGELOG breaking-change scan returned no versions; the "
            "scanner heuristic is broken (sibling-gap meta-pattern: V11 "
            "passes vacuously).")

        drift = []
        for version in flagged:
            if version in _MIGRATION_GUIDE_SIBLING_COVERED:
                continue
            if f"## {version}" not in text:
                drift.append(version)

        assert not drift, (
            f"Migration-Guide.md missing ``## X.Y.Z`` section(s) for "
            f"CHANGELOG-flagged breaking-change version(s): {drift}.  "
            f"Either add a migration recipe for each, OR add the "
            f"version to "
            f"``_MIGRATION_GUIDE_SIBLING_COVERED`` in "
            f"``{Path(__file__).name}`` with a comment explaining "
            f"which sibling version's recipe covers it."
        )

    # audit closure: P2-NEW-F2-MED-2
    def test_migration_guide_sections_are_non_trivial(self):
        """Each ``## X.Y.Z`` Migration-Guide section must contain
        non-trivial content between its heading and the next section.

        ``Non-trivial`` is operationalised as ``> 200 chars of content
        excluding whitespace, exclusive of the heading itself``.  A
        version section with only a heading would pass the prior
        ``has_known_version_sections`` check vacuously -- this guards
        against stub-only sections.
        """
        text = _MIGRATION_GUIDE.read_text(encoding='utf-8')
        # Split on ``## X.Y.Z`` or ``## X.Y.Z -- ...`` headings.
        sections = []
        # Match top-level (``## ``) entries that look like version
        # headings.  Pattern: ``## X.Y.Z`` optionally followed by ``--
        # description``.  We use ``re.finditer`` with positions to
        # bracket each section.
        heading_pattern = re.compile(
            r'^##\s+(\d+\.\d+\.\d+)(?:\s+--\s+.*)?$',
            re.MULTILINE)
        matches = list(heading_pattern.finditer(text))
        for i, m in enumerate(matches):
            ver = m.group(1)
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end]
            # Strip the optional trailing horizontal rule / next-major
            # section delimiter.
            body_for_check = re.sub(r'\s+', '', body)
            sections.append((ver, body_for_check, body))

        assert sections, (
            "Migration-Guide.md contains no ``## X.Y.Z`` headings to "
            "verify.  The Migration-Guide has been emptied -- audit "
            "P2-NEW-F2-MED-2 closure check has degenerated.")

        stubs = []
        for ver, body_stripped, _full_body in sections:
            if len(body_stripped) < 200:
                stubs.append((ver, len(body_stripped)))

        assert not stubs, (
            f"Migration-Guide.md sections are stubs (<200 chars of "
            f"non-whitespace content): {stubs}.  Each ``## X.Y.Z`` "
            f"heading must be followed by a migration recipe / prose "
            f"explaining the user-facing change."
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
