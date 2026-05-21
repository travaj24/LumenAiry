#!/usr/bin/env python3
# v5.2.3 (ROADMAP v5.2.x dep-metadata drift check; v5.2.2 incident retrospective):
# periodic check that catches the v5.2.2-class regression automatically.
#
# Between v5.1.1 (which re-added Python 3.10 to CI) and v5.2.1, two optional
# deps published 2025 releases that bumped their ``requires-python`` to
# ``>=3.11``, silently breaking ``pip install lumenairy[fft,zarr,...]`` on
# 3.10:
#
# - pyfftw 0.15.1 (2025): ``requires-python = ">=3.11"``
# - zarr 3.1.6 (2025):    ``requires-python = ">=3.11"``
#
# v5.2.2 fixed via PEP 508 environment markers, but this class of failure
# WILL recur: any dep we list with no upper cap may, in any future release,
# bump its floor and silently drop a supported Python.  This script catches
# such drift PROACTIVELY by walking pyproject.toml, fetching each dep's
# PyPI metadata, and flagging cases where the latest release of a dep does
# not support every LumenAiry-supported Python (3.10 / 3.11 / 3.12 / 3.13
# per the ``requires-python = ">=3.10"`` floor in pyproject.toml).
#
# Usage
# -----
#
#     python scripts/check_dep_metadata.py
#     python scripts/check_dep_metadata.py --help
#     python scripts/check_dep_metadata.py --pyproject /path/to/pyproject.toml
#     python scripts/check_dep_metadata.py --timeout 20
#
# Exit codes:
#
# - 0 -- no drift detected (every dep's latest version supports all of
#        LumenAiry's supported Python versions, OR an existing PEP 508
#        env marker already constrains the dep to the supported subset).
# - 1 -- drift detected (at least one dep's latest version drops a
#        LumenAiry-supported Python and is NOT guarded by an env marker).
#
# Network errors (PyPI unreachable, JSON malformed, etc.) are reported as
# WARN lines and skipped, NOT counted as drift -- the script must be safe
# to run on an offline CI runner.
#
# When drift IS flagged the report includes:
#
# - Package name
# - Latest version on PyPI + its ``requires-python``
# - The LumenAiry-supported Python versions for which the latest dep
#   version does NOT resolve
# - The latest dep version that DOES resolve on each affected Python
# - A suggested PEP 508 env-marker fix, matching the v5.2.2 pattern
#
# This script is invoked by ``.github/workflows/dep-drift.yml`` on a weekly
# cron (Mondays 06:00 UTC) and on manual dispatch.  It does NOT gate PRs;
# it surfaces a red weekly badge that the maintainer investigates.

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

# v5.2.3 (ROADMAP v5.2.x dep-metadata drift check; v5.2.2 incident retrospective):
# Python 3.11+ ships ``tomllib`` in the stdlib; on 3.10 (the documented
# library floor) we fall back to the ``tomli`` backport.  This mirrors the
# V11 walker idiom in tests/unit/test_v4_16_2_dispatcher_pin_doc_consistency.py.
try:
    import tomllib as _tomllib  # type: ignore[import-not-found]  # Python 3.11+
except ImportError:  # pragma: no cover -- exercised only on Python 3.10
    try:
        import tomli as _tomllib  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover -- truly missing
        _tomllib = None  # type: ignore[assignment]


# ----------------------------------------------------------------------
# LumenAiry-side constants
# ----------------------------------------------------------------------

# The Python versions LumenAiry advertises support for, per the
# ``requires-python = ">=3.10"`` line in pyproject.toml AND the
# Programming-Language classifiers (3.10, 3.11, 3.12, 3.13).
# v5.2.3: 3.14 is NOT in this list -- 3.14 was dropped from classifiers
# at v5.0.1 because optional accelerators (numba, jax, pyfftw) lack
# wheels; the canonical matrix is 3.10-3.13.
LUMENAIRY_SUPPORTED_PYTHONS = ("3.10", "3.11", "3.12", "3.13")


# ----------------------------------------------------------------------
# pyproject.toml parsing
# ----------------------------------------------------------------------

# PEP 508 requirement parsing -- minimal: split off ``; marker`` and
# strip version specifiers / extras to recover the bare package name.
_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+")


def _split_requirement(req: str) -> tuple[str, str]:
    """Split a PEP 508 requirement into ``(spec_part, marker_part)``.

    The marker_part is the substring after the first ``;`` (with the
    ``;`` stripped); empty string when no marker is present.
    """
    if ";" in req:
        spec, marker = req.split(";", 1)
        return spec.strip(), marker.strip()
    return req.strip(), ""


def _extract_name_and_specifier(spec: str) -> tuple[Optional[str], str]:
    """Extract ``(name, version_specifier_set)`` from a PEP 508 spec.

    The version specifier set is the substring AFTER the name (and any
    ``[extras]``) up to the marker, e.g. ``>=0.13,<0.15.1``.  Empty
    string when the requirement is bare (just a name).  Returns
    ``(None, "")`` for unparseable lines.
    """
    spec = spec.strip()
    if not spec:
        return None, ""
    # Strip extras: ``jax[cuda12]>=0.4.20`` -> ``jax>=0.4.20``
    if "[" in spec and "]" in spec:
        spec = spec.split("[", 1)[0] + spec.split("]", 1)[1]
    elif "[" in spec:
        spec = spec.split("[", 1)[0]
    m = _NAME_RE.match(spec)
    if m is None:
        return None, ""
    name = m.group(0).lower()
    rest = spec[m.end():].strip()
    return name, rest


def parse_pyproject_deps(pyproject_path: Path) -> dict[str, list[tuple[str, str]]]:
    """Parse pyproject.toml and return ``{package_name: [(spec, marker), ...]}``.

    The mapping is keyed by lowercased canonical distribution name (PEP
    503 normalisation kept light: just ``.lower()`` here, which is
    sufficient for the deps LumenAiry actually declares).  The value is
    the list of ``(version_specifier_set, marker_str)`` pairs observed
    for that package across every block it appears in (hard
    ``dependencies`` + every ``optional-dependencies`` group).  Each
    pair is one PEP 508 line: ``spec`` is the version constraint after
    the name (``">=0.13,<0.15.1"`` etc., or ``""`` for unbounded) and
    ``marker`` is the post-``;`` PEP 508 environment marker (``""`` if
    unconditional).
    """
    if _tomllib is None:
        raise RuntimeError(
            "Neither ``tomllib`` (Python 3.11+ stdlib) nor ``tomli`` "
            "(Python 3.10 backport) is importable.  Install ``tomli`` "
            "to run the dep-metadata drift check on this interpreter "
            "(``pip install tomli``)."
        )
    with pyproject_path.open("rb") as f:
        data = _tomllib.load(f)

    out: dict[str, list[tuple[str, str]]] = {}

    project = data.get("project", {})
    hard_deps = project.get("dependencies", []) or []
    optional = project.get("optional-dependencies", {}) or {}

    blocks: list[list[str]] = [hard_deps]
    for _group, deps in optional.items():
        blocks.append(deps or [])

    for block in blocks:
        for raw in block:
            spec_part, marker = _split_requirement(raw)
            name, version_spec = _extract_name_and_specifier(spec_part)
            if name is None:
                continue
            out.setdefault(name, []).append((version_spec, marker))

    return out


# ----------------------------------------------------------------------
# PEP 440 / PEP 508 helpers for requires-python matching
# ----------------------------------------------------------------------

_VER_PART_RE = re.compile(r"(\d+)")
# Matches PEP 440 pre-release / dev / post-release suffixes such as
# ``3.0.0a1``, ``3.0.0rc2``, ``1.2.dev3``, ``2.0.post1``.  Used to skip
# pre-release versions when modelling pip's default (stable-only)
# resolver behaviour.
_PRE_RE = re.compile(r"(?:a|b|c|rc|alpha|beta|dev|pre|preview)\d*", re.IGNORECASE)


def _ver_tuple(ver: str) -> tuple[int, ...]:
    """Return a tuple of leading integer components for sortable Python
    version strings, e.g. ``"3.10"`` -> ``(3, 10)``.  Non-digit suffixes
    are dropped so ``"3.10.0rc1"`` -> ``(3, 10, 0)``.
    """
    parts: list[int] = []
    for chunk in ver.split("."):
        m = _VER_PART_RE.match(chunk)
        if m is None:
            break
        parts.append(int(m.group(1)))
    return tuple(parts)


def _is_pre_release(ver: str) -> bool:
    """Return True for PEP 440 pre-release / dev versions.

    We exclude these when modelling pip's resolver because pip by
    default only picks pre-releases when the user explicitly opts in
    (or when ONLY pre-releases satisfy the requirement); LumenAiry's
    pyproject lines never set ``--pre`` so the resolver target is
    always the latest stable release.

    Detection: match the standard PEP 440 pre-release suffixes
    (``aN``, ``bN``, ``cN``, ``rcN``, ``dev``, etc.) ANYWHERE in the
    version string.  Post-releases (``postN``) are NOT considered
    pre-releases.
    """
    return bool(_PRE_RE.search(ver))


def _cmp_versions(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    """Return -1/0/+1 comparing two version tuples component-wise."""
    if a < b:
        return -1
    if a > b:
        return 1
    return 0


def _spec_set_accepts(spec_set: Optional[str], probe_ver: str) -> bool:
    """Return True if a PEP 440 version-specifier set admits ``probe_ver``.

    ``spec_set`` is e.g. ``">=3.11"``, ``">=3.8,<4"``, ``">=0.13,<0.15.1"``,
    ``""``, or None.  ``probe_ver`` is a version string like ``"3.10"``
    or ``"0.15.1"``.  We implement the common operators (``>=``, ``>``,
    ``<=``, ``<``, ``==``, ``!=``, ``~=``) directly so we don't need the
    ``packaging`` package as a hard dep.

    Used for two distinct purposes:
    - matching ``requires-python`` strings against a Python version
    - matching a dep's ``[project]`` version-spec set against a candidate
      release version (so we can ask "would the resolver, given this
      pyproject line, accept release X?").
    """
    if not spec_set:
        return True

    probe_t = _ver_tuple(probe_ver)

    for clause in spec_set.split(","):
        clause = clause.strip()
        if not clause:
            continue
        # Match longest operators first so ``>=`` doesn't lose to ``>``.
        op: Optional[str] = None
        rhs: str = ""
        for candidate in (">=", "<=", "==", "!=", "~=", ">", "<"):
            if clause.startswith(candidate):
                op = candidate
                rhs = clause[len(candidate):].strip()
                break
        if op is None:
            # Unknown operator -- conservative: treat as "accepts".
            continue

        rhs_t = _ver_tuple(rhs)
        c = _cmp_versions(probe_t, rhs_t)

        if op == ">=" and c < 0:
            return False
        if op == ">" and c <= 0:
            return False
        if op == "<=" and c > 0:
            return False
        if op == "<" and c >= 0:
            return False
        if op == "==" and c != 0:
            return False
        if op == "!=" and c == 0:
            return False
        if op == "~=":
            # ``~=3.11`` means ``>=3.11, ==3.*``.  Approximate by
            # requiring same major and >=rhs.
            if c < 0 or (probe_t and rhs_t and probe_t[0] != rhs_t[0]):
                return False

    return True


def _requires_python_accepts(requires_python: Optional[str], py_ver: str) -> bool:
    """Back-compat alias for ``_spec_set_accepts`` used in the
    requires-python matching context (kept distinct for readability).
    """
    return _spec_set_accepts(requires_python, py_ver)


# ----------------------------------------------------------------------
# PEP 508 environment-marker inspection
# ----------------------------------------------------------------------

# v5.2.3: we don't fully evaluate PEP 508 markers (that needs the
# ``packaging`` package).  Instead we recognise the SHAPE of the markers
# LumenAiry uses in pyproject.toml -- specifically ``python_version <
# "3.11"`` / ``python_version >= "3.11"`` (the v5.2.2 idiom).  Anything
# more exotic falls back to "marker present, treat as guarded" so we
# don't false-positive against an existing fix.

_PY_MARKER_RE = re.compile(
    r"""python_version\s*(?P<op>==|!=|>=|<=|>|<)\s*['"](?P<ver>\d+(?:\.\d+)*)['"]"""
)


def _marker_admits(marker: str, py_ver: str) -> bool:
    """Return True if ``marker`` admits ``py_ver``.

    Empty marker = unconditional = always admits.  When the marker
    contains a ``python_version`` clause we apply it; any OTHER clauses
    (``sys_platform``, etc.) are conservatively treated as "True" so
    we don't false-positive against platform-only markers.  Multiple
    ``python_version`` clauses joined by ``and`` are all enforced; we
    don't currently parse ``or``.
    """
    if not marker:
        return True

    # If there is no python_version clause at all, the marker is about
    # something else (platform, implementation, ...).  Treat as admitting
    # py_ver for our purposes (we only model Python-version drift).
    if "python_version" not in marker:
        return True

    py_t = _ver_tuple(py_ver)
    # All python_version clauses must admit py_ver (we approximate ``and``).
    for match in _PY_MARKER_RE.finditer(marker):
        op = match.group("op")
        rhs_t = _ver_tuple(match.group("ver"))
        c = _cmp_versions(py_t, rhs_t)
        if op == ">=" and c < 0:
            return False
        if op == ">" and c <= 0:
            return False
        if op == "<=" and c > 0:
            return False
        if op == "<" and c >= 0:
            return False
        if op == "==" and c != 0:
            return False
        if op == "!=" and c == 0:
            return False

    return True


def _has_marker_coverage(entries: list[tuple[str, str]], py_ver: str) -> bool:
    """Return True if ANY pyproject entry for the package admits py_ver.

    This is the union semantics pip's resolver uses across multiple
    entries in the same extras group: e.g. the v5.2.2 ``fft`` pair
        'pyfftw>=0.13,<0.15.1; python_version < "3.11"',
        'pyfftw>=0.13; python_version >= "3.11"',
    covers BOTH 3.10 (first line) AND 3.11+ (second line), so the
    package IS installable on every supported Python.  A package whose
    only entry is unconditional (empty marker) trivially admits every
    Python.

    ``entries`` is the ``[(spec, marker), ...]`` list from
    ``parse_pyproject_deps``; we only consult the marker here.
    """
    if not entries:
        # No entries at all -- the package isn't actually a dep, which
        # shouldn't happen because we only call this for parsed deps.
        # Defensive: treat as unconditional.
        return True
    for _spec, marker in entries:
        if _marker_admits(marker, py_ver):
            return True
    return False


def _applicable_specs_for(entries: list[tuple[str, str]], py_ver: str) -> list[str]:
    """Return the list of version-specifier-set strings from pyproject
    entries that apply on ``py_ver`` (i.e. whose env marker admits it).

    The resolver's behaviour on a given Python is to UNION the
    applicable lines: a candidate release version is acceptable if it
    satisfies AT LEAST ONE applicable specifier set.  (In practice
    pyproject doesn't put two competing lines for the same package on
    the same Python; the v5.2.2 idiom partitions by ``python_version``
    so each Python gets exactly one line.  Either way, union is the
    correct semantics.)
    """
    return [spec for spec, marker in entries if _marker_admits(marker, py_ver)]


def _any_spec_admits(specs: list[str], probe_ver: Optional[str]) -> bool:
    """Return True if at least one of ``specs`` admits ``probe_ver``.

    Used to ask "would the resolver, on this Python, accept release X
    of the dep?" -- where X is typically the dep's PyPI ``latest``.
    """
    if probe_ver is None:
        return False
    if not specs:
        # No applicable spec lines on this Python = dep isn't installed
        # there, so the question is moot; caller filters this case out.
        return False
    for s in specs:
        if _spec_set_accepts(s, probe_ver):
            return True
    return False


# ----------------------------------------------------------------------
# PyPI fetch
# ----------------------------------------------------------------------


def fetch_pypi_metadata(name: str, timeout: float) -> Optional[dict[str, Any]]:
    """Return the parsed JSON for ``https://pypi.org/pypi/<name>/json``,
    or None on any network / parse failure.  Errors are logged via
    a returned ``__error__`` sentinel inside the dict.
    """
    url = f"https://pypi.org/pypi/{name}/json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"__error__": f"network: {type(e).__name__}: {e}"}
    try:
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return {"__error__": f"parse: {type(e).__name__}: {e}"}


def _latest_version_admitting(releases: dict[str, list[dict[str, Any]]], py_ver: str) -> Optional[str]:
    """Walk releases (a ``{version: [file_dicts]}`` mapping) and return
    the highest-versioned STABLE release whose per-file ``requires_python``
    admits ``py_ver``.

    Releases with NO files (yanked-only / pre-release noise) are
    skipped, as are PEP 440 pre-releases (``aN``, ``bN``, ``rcN``,
    ``devN``) since pip's default resolver does not pick them.  When
    files within a release disagree on ``requires_python`` we use the
    value from the first non-yanked file that has one.
    """
    best: Optional[tuple[tuple[int, ...], str]] = None
    for ver, files in releases.items():
        if not files:
            continue
        if _is_pre_release(ver):
            continue
        # Pick the requires_python from the first non-yanked file.
        rp: Optional[str] = None
        all_yanked = True
        for f in files:
            if f.get("yanked"):
                continue
            all_yanked = False
            cand = f.get("requires_python")
            if cand is not None:
                rp = cand
                break
        if all_yanked:
            continue
        if not _requires_python_accepts(rp, py_ver):
            continue
        ver_t = _ver_tuple(ver)
        if not ver_t:
            continue
        if best is None or ver_t > best[0]:
            best = (ver_t, ver)
    return best[1] if best is not None else None


# ----------------------------------------------------------------------
# Drift analysis
# ----------------------------------------------------------------------


class DriftReport:
    """Per-package drift summary.

    Attributes
    ----------
    name : str
        Lowercased PyPI distribution name.
    latest : Optional[str]
        Latest version string from ``info.version``.
    latest_requires_python : Optional[str]
        ``info.requires_python`` for the latest release.
    unsupported_pythons : list[str]
        Subset of LUMENAIRY_SUPPORTED_PYTHONS for which the latest
        release's ``requires_python`` does NOT admit, AND for which no
        pyproject env marker already excludes that Python.
    last_supporting : dict[str, Optional[str]]
        For each Python in ``unsupported_pythons``, the highest version
        of this dep that DOES admit it (or None if no historical release
        does).
    error : Optional[str]
        Set when PyPI was unreachable / metadata was unparseable.  When
        non-None this dep is reported as "could not verify" and does NOT
        count toward the script's exit code.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.latest: Optional[str] = None
        self.latest_requires_python: Optional[str] = None
        self.unsupported_pythons: list[str] = []
        self.last_supporting: dict[str, Optional[str]] = {}
        self.error: Optional[str] = None

    @property
    def has_drift(self) -> bool:
        return self.error is None and bool(self.unsupported_pythons)


def _latest_version_in_specs_admitting(
    releases: dict[str, list[dict[str, Any]]],
    specs: list[str],
    py_ver: str,
) -> Optional[str]:
    """Return the highest release version that BOTH satisfies at least
    one of ``specs`` (pyproject version constraints active on this
    Python) AND has a ``requires_python`` admitting ``py_ver``.

    This models what pip's resolver actually does: it picks the highest
    version matching the pyproject specifier set, then verifies it
    supports the current interpreter.
    """
    best: Optional[tuple[tuple[int, ...], str]] = None
    for ver, files in releases.items():
        if not files:
            continue
        if _is_pre_release(ver):
            # pip's default resolver skips pre-releases; we model that
            # so e.g. ``zarr 3.0.0a0`` (which kept >=3.10 support) does
            # NOT mask the fact that every STABLE ``zarr>=3.0`` release
            # requires >=3.11.
            continue
        ver_t = _ver_tuple(ver)
        if not ver_t:
            continue
        # Must satisfy the pyproject version constraints.
        if not _any_spec_admits(specs, ver):
            continue
        # Per-release requires_python.  Use the first non-yanked file
        # that exposes it.
        rp: Optional[str] = None
        all_yanked = True
        for f in files:
            if f.get("yanked"):
                continue
            all_yanked = False
            cand = f.get("requires_python")
            if cand is not None:
                rp = cand
                break
        if all_yanked:
            continue
        if not _requires_python_accepts(rp, py_ver):
            continue
        if best is None or ver_t > best[0]:
            best = (ver_t, ver)
    return best[1] if best is not None else None


def analyse_package(
    name: str,
    entries: list[tuple[str, str]],
    timeout: float,
) -> DriftReport:
    """Fetch PyPI metadata for one package and produce a DriftReport.

    ``entries`` is the ``[(version_specifier_set, marker), ...]`` list
    from ``parse_pyproject_deps``.  Drift on Python ``py_ver`` is
    flagged when:

    1. The pyproject markers admit py_ver (the dep IS supposed to be
       installable there).
    2. The highest release version that the pyproject's version-spec
       set allows on py_ver has a ``requires_python`` that does NOT
       admit py_ver.

    (1) without (2) means the pyproject correctly caps to a working
    version; (2) without (1) means the dep simply isn't installed on
    that Python -- neither is drift.
    """
    rep = DriftReport(name)
    meta = fetch_pypi_metadata(name, timeout=timeout)
    if meta is None or "__error__" in (meta or {}):
        rep.error = (meta or {}).get("__error__", "unknown error")
        return rep

    info = meta.get("info", {}) or {}
    releases = meta.get("releases", {}) or {}
    rep.latest = info.get("version")
    rep.latest_requires_python = info.get("requires_python")

    for py_ver in LUMENAIRY_SUPPORTED_PYTHONS:
        # (1) If the existing markers already exclude this Python, no
        # drift is possible -- the dep simply isn't installed there.
        if not _has_marker_coverage(entries, py_ver):
            continue
        # The version-spec sets active on py_ver (the lines whose
        # markers admit py_ver).
        active_specs = _applicable_specs_for(entries, py_ver)
        # (2) Find the highest release version the resolver would pick
        # on py_ver under those specs, AND whose requires_python admits
        # py_ver.  If such a version exists, the pyproject is correctly
        # capped (e.g. the v5.2.2 ``pyfftw<0.15.1; python_version < "3.11"``
        # picks 0.15.0 on 3.10, which supports 3.10).  No drift.
        resolved = _latest_version_in_specs_admitting(
            releases, active_specs, py_ver
        )
        if resolved is not None:
            continue
        # No release satisfies BOTH the pyproject constraints AND
        # supports py_ver -- this Python install will fail.  Drift.
        rep.unsupported_pythons.append(py_ver)
        # For the suggested-fix block, record the highest release that
        # supports py_ver (ignoring pyproject constraints, since the
        # whole point of the fix is to rewrite them).
        rep.last_supporting[py_ver] = _latest_version_admitting(
            releases, py_ver
        )

    return rep


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def _format_marker_fix(rep: DriftReport) -> str:
    """Build a suggested PEP 508 env-marker pair matching the v5.2.2
    pattern.  We pick the lowest unsupported Python as the split point
    and suggest:

        '<name>>=<floor>,<<latest>; python_version < "<split>"',
        '<name>>=<floor>; python_version >= "<split>"',

    where ``<floor>`` is left as a placeholder (the existing pyproject
    floor) and ``<split>`` is the lowest Python the latest version
    REJECTS.  The maintainer adapts the floor in the serial pyproject
    edit phase.
    """
    if not rep.unsupported_pythons:
        return "(no fix needed)"
    split = rep.unsupported_pythons[0]
    # The "next" Python that IS supported by the latest version: the
    # first LumenAiry-supported Python AFTER the highest unsupported one.
    last_unsupported = rep.unsupported_pythons[-1]
    next_supported: Optional[str] = None
    for py in LUMENAIRY_SUPPORTED_PYTHONS:
        if _ver_tuple(py) > _ver_tuple(last_unsupported):
            next_supported = py
            break
    cap = rep.latest or "<latest>"
    legacy_ver = rep.last_supporting.get(split) or "<historical-version>"
    if next_supported is None:
        # All LumenAiry pythons are unsupported by the latest -- the dep
        # has truly dropped them all.  Pin to legacy_ver everywhere.
        return (
            f"    '{rep.name}>=<floor>,<={legacy_ver}',  "
            f"# v5.2.3: {rep.name} {cap} dropped Python "
            f"{', '.join(rep.unsupported_pythons)}; pin to last compatible."
        )
    return (
        f"    '{rep.name}>=<floor>,<{cap}; python_version < \"{next_supported}\"',\n"
        f"    '{rep.name}>=<floor>; python_version >= \"{next_supported}\"',  "
        f"# v5.2.3 (env-marker drift fix; mirrors v5.2.2 pyfftw/zarr pattern)"
    )


def print_report(reports: list[DriftReport], stream=sys.stdout) -> None:
    """Print a grep-able per-package table + a drift-detail block."""
    # Column widths.
    name_w = max(4, max((len(r.name) for r in reports), default=4))
    latest_w = max(6, max((len(r.latest or "-") for r in reports), default=6))
    rp_w = max(16, max((len(r.latest_requires_python or "-") for r in reports), default=16))
    rp_w = min(rp_w, 32)  # cap so the table doesn't run away

    header = (
        f"{'name':<{name_w}}  {'latest':<{latest_w}}  "
        f"{'requires-python':<{rp_w}}  DRIFT?"
    )
    sep = "-" * len(header)
    print(header, file=stream)
    print(sep, file=stream)

    for r in sorted(reports, key=lambda x: x.name):
        if r.error is not None:
            flag = "SKIP"
            latest = "-"
            rp = "-"
        elif r.has_drift:
            flag = "YES"
            latest = r.latest or "-"
            rp = (r.latest_requires_python or "-")[:rp_w]
        else:
            flag = "no"
            latest = r.latest or "-"
            rp = (r.latest_requires_python or "-")[:rp_w]
        print(
            f"{r.name:<{name_w}}  {latest:<{latest_w}}  {rp:<{rp_w}}  {flag}",
            file=stream,
        )

    drifted = [r for r in reports if r.has_drift]
    skipped = [r for r in reports if r.error is not None]

    if skipped:
        print("", file=stream)
        print(
            f"WARN: could not verify {len(skipped)} package(s) "
            f"(network / parse error -- NOT counted as drift):",
            file=stream,
        )
        for r in skipped:
            print(f"  - {r.name}: {r.error}", file=stream)

    if drifted:
        print("", file=stream)
        print("=" * 72, file=stream)
        print(
            f"WARNING: {len(drifted)} package(s) show metadata drift "
            f"against LumenAiry's documented Python support "
            f"({', '.join(LUMENAIRY_SUPPORTED_PYTHONS)}):",
            file=stream,
        )
        print("=" * 72, file=stream)
        for r in drifted:
            print("", file=stream)
            print(f"  Package:        {r.name}", file=stream)
            print(f"  Latest version: {r.latest}", file=stream)
            print(
                f"  requires-python: {r.latest_requires_python!r}",
                file=stream,
            )
            print(
                f"  Drops Python:   {', '.join(r.unsupported_pythons)}",
                file=stream,
            )
            print("  Last compatible:", file=stream)
            for py in r.unsupported_pythons:
                last = r.last_supporting.get(py) or "(none found)"
                print(f"    Python {py}: {last}", file=stream)
            print("  Suggested PEP 508 env-marker fix in pyproject.toml:", file=stream)
            print(_format_marker_fix(r), file=stream)
    else:
        print("", file=stream)
        print(
            "OK: no dep-metadata drift detected against LumenAiry's "
            f"supported Python versions ({', '.join(LUMENAIRY_SUPPORTED_PYTHONS)}).",
            file=stream,
        )


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="check_dep_metadata.py",
        description=(
            "v5.2.3 dep-metadata drift check.  Walks pyproject.toml, "
            "fetches each dep's PyPI metadata, and flags any case where "
            "the latest release of a dep does NOT support one of "
            "LumenAiry's advertised Python versions ("
            + ", ".join(LUMENAIRY_SUPPORTED_PYTHONS)
            + ").  See the module docstring for the v5.2.2 incident "
            "retrospective."
        ),
        epilog=(
            "Exit codes: 0 = no drift; 1 = drift detected.  Network "
            "errors are reported as WARN and skipped (NOT counted as "
            "drift) so the script is safe to run on offline runners."
        ),
    )
    p.add_argument(
        "--pyproject",
        type=Path,
        default=None,
        help=(
            "Path to pyproject.toml (default: the pyproject.toml that "
            "ships with this script's repo, found via parent dirs)."
        ),
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Per-request PyPI fetch timeout in seconds (default: 15).",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Skip the PyPI fetch entirely; only validate that pyproject "
            "parses and the package list is non-empty.  Useful for the "
            "smoke test."
        ),
    )
    return p


def _default_pyproject() -> Path:
    here = Path(__file__).resolve()
    # scripts/check_dep_metadata.py -> repo root is parents[1].
    return here.parents[1] / "pyproject.toml"


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    pyproject_path = args.pyproject or _default_pyproject()
    if not pyproject_path.is_file():
        print(
            f"ERROR: pyproject.toml not found at {pyproject_path}",
            file=sys.stderr,
        )
        return 2

    try:
        deps = parse_pyproject_deps(pyproject_path)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if not deps:
        print(
            f"ERROR: no dependencies parsed from {pyproject_path} "
            f"(is the [project] block present?)",
            file=sys.stderr,
        )
        return 2

    print(
        f"Checking {len(deps)} unique dep(s) against LumenAiry-supported "
        f"Pythons {LUMENAIRY_SUPPORTED_PYTHONS} ...",
        file=sys.stdout,
    )

    if args.offline:
        print(
            "  (--offline: skipping PyPI fetch; pyproject parse OK).",
            file=sys.stdout,
        )
        return 0

    reports: list[DriftReport] = []
    for name in sorted(deps.keys()):
        rep = analyse_package(name, deps[name], timeout=args.timeout)
        reports.append(rep)

    print("", file=sys.stdout)
    print_report(reports, stream=sys.stdout)

    drifted = [r for r in reports if r.has_drift]
    return 1 if drifted else 0


if __name__ == "__main__":
    raise SystemExit(main())
