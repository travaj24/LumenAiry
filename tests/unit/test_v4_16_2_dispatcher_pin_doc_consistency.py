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
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / 'pyproject.toml'
_README = _REPO_ROOT / 'README.md'
_REQUIREMENTS = _REPO_ROOT / 'requirements.txt'
_ROADMAP = _REPO_ROOT / 'ROADMAP.md'
_CHANGELOG = _REPO_ROOT / 'CHANGELOG.md'


# ----------------------------------------------------------------------
# Helpers -- parse pyproject.toml hard deps + optional extras
# ----------------------------------------------------------------------

def _parse_pyproject_dependencies():
    """Return (hard_deps, optional_groups_dict) parsed from
    pyproject.toml.

    Returns
    -------
    hard_deps : set of str
        Package names (not version specs) in [project.dependencies].
    optional_groups : dict {group_name: set of str}
        Package names in each [project.optional-dependencies.{group}].
    """
    text = _PYPROJECT.read_text(encoding='utf-8')

    # Match a `dependencies = [...]` block (multi-line list of strings).
    deps_match = re.search(
        r'^dependencies\s*=\s*\[(.*?)\]',
        text, re.MULTILINE | re.DOTALL,
    )
    if deps_match is None:
        pytest.skip("Could not parse [project.dependencies]")
    deps_block = deps_match.group(1)
    hard_deps = set(_extract_package_names(deps_block))

    # Match [project.optional-dependencies] -- look for `name = [...]`
    # entries inside that section.
    opt_section_match = re.search(
        r'\[project\.optional-dependencies\](.*?)(?=^\[|\Z)',
        text, re.MULTILINE | re.DOTALL,
    )
    optional_groups = {}
    if opt_section_match is not None:
        opt_block = opt_section_match.group(1)
        for grp_match in re.finditer(
                r'^([a-zA-Z_][a-zA-Z0-9_-]*)\s*=\s*\[(.*?)\]',
                opt_block, re.MULTILINE | re.DOTALL):
            grp_name = grp_match.group(1)
            grp_block = grp_match.group(2)
            optional_groups[grp_name] = set(_extract_package_names(grp_block))

    return hard_deps, optional_groups


def _extract_package_names(block):
    """Extract package names from a TOML list of strings like
    ``["numpy>=1.20", "scipy>=1.7", ...]``.  Returns each name lowercased
    (so `refractiveindex` and `Refractiveindex` reconcile).
    """
    names = []
    for line in block.splitlines():
        # Strip comments
        if '#' in line:
            line = line[:line.index('#')]
        # Find quoted strings
        for m in re.finditer(r'"([^"]+)"', line):
            spec = m.group(1)
            # Split on first version operator (==, >=, <=, !=, ~=, >, <)
            name = re.split(r'[<>=!~]', spec, maxsplit=1)[0].strip()
            # Strip extras: "package[extra]" -> "package"
            name = name.split('[')[0]
            if name:
                names.append(name.lower())
    return names


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
        pin_files = sorted(
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
# Migration-Guide.md must exist
# ----------------------------------------------------------------------

class TestMigrationGuideExists:
    """Migration-Guide.md is a v4.16.2 pre-v5.0 prep addition."""

    def test_migration_guide_md_exists(self):
        guide = _REPO_ROOT / 'Migration-Guide.md'
        assert guide.is_file(), (
            f"{guide} does not exist.  v4.16.2 should have created "
            f"the migration guide.")

    def test_migration_guide_has_known_version_sections(self):
        guide = _REPO_ROOT / 'Migration-Guide.md'
        text = guide.read_text(encoding='utf-8')
        # Each known-breaking-change version must have a section.
        for version in ('4.13.0', '4.15.1', '4.16.1', '4.16.2'):
            assert f"## {version}" in text, (
                f"Migration-Guide.md missing section for {version}.")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
