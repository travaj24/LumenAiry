"""Public-API smoke test (v5.0 ROADMAP).

Asserts that every name listed in ``lumenairy.__all__`` is resolvable
via ``getattr(lumenairy, name)``.  Catches the recurring "exported but
not actually imported" / "imported but not exported" sibling-gap at
the top-level facade.

Note that V9 (the `__all__` symmetry walker) covers SUBMODULE
``__all__`` <-> top-level re-export.  This file covers the
complementary check: every top-level ``__all__`` entry must actually
resolve at the top-level.

COLLECTION SHAPE, 2026-09-12 (audit 2026-09-11, V4/P1-9).  This file used to
``@pytest.mark.parametrize`` over ``lumenairy.__all__``, so a 104-line file
generated **726 collected ids** -- 4.9 % of the whole suite (14 666) -- for
what is one property.  The headline test count is used as a proxy for physics
coverage, and 726 ``hasattr`` ids inflate it without adding a single
behaviour.  The parametrisation is now a loop that reports EVERY failing name
at once (strictly more useful than 726 ids, each of which reported one), so
the file collects **7 ids, one per property**.  The resolvability coverage is
unchanged: all 708 names are still checked on every run.
"""
from __future__ import annotations

import importlib.metadata
import pathlib
import re

import lumenairy as la


def test_dunder_all_exists_and_is_a_list():
    assert hasattr(la, '__all__'), (
        "lumenairy.__all__ must be defined as the canonical public-API "
        "manifest")
    assert isinstance(la.__all__, list), (
        f"lumenairy.__all__ must be a list, got {type(la.__all__).__name__}")
    assert len(la.__all__) > 100, (
        f"lumenairy.__all__ has only {len(la.__all__)} entries; expected "
        f">100 at v5.0 baseline.  Something has truncated the public-API "
        f"surface.")


def test_every_all_entry_is_resolvable():
    """Every entry in ``lumenairy.__all__`` must be importable via getattr.

    One assertion over all of ``__all__`` rather than one parametrised id per
    name: a broken facade is almost never one name (it is a whole import block
    or a whole re-export tier), and the loop names every casualty in one
    message instead of one per id.
    """
    missing = [name for name in la.__all__ if not hasattr(la, name)]
    assert not missing, (
        f"lumenairy.__all__ lists {len(missing)} name(s) that "
        f"`getattr(lumenairy, name)` cannot resolve: {missing}.  Either "
        f"remove them from __all__ or add the missing import at the top of "
        f"lumenairy/__init__.py.  ({len(la.__all__)} names checked.)")


def test_dunder_all_has_no_duplicates():
    seen = set()
    dupes = []
    for name in la.__all__:
        if name in seen:
            dupes.append(name)
        seen.add(name)
    assert not dupes, (
        f"lumenairy.__all__ contains duplicates: {dupes}")


def test_dunder_all_entries_are_identifier_strings():
    """The manifest is consumed by ``getattr`` and by the V9 symmetry walker,
    both of which assume plain identifier strings.  A tuple, a dotted path or
    a stray empty string would make ``hasattr`` silently False and read as a
    missing export."""
    bad = [entry for entry in la.__all__
           if not (isinstance(entry, str) and entry.isidentifier())]
    assert not bad, (
        f"lumenairy.__all__ must hold plain identifier strings; these are "
        f"not: {bad!r}")


def test_version_string_format():
    """__version__ must be a PEP 440 release identifier."""
    assert hasattr(la, '__version__')
    assert re.match(r'^\d+\.\d+\.\d+(a\d+|b\d+|rc\d+)?$', la.__version__), (
        f"__version__ = {la.__version__!r} is not PEP 440 conformant")


def test_installed_metadata_version_matches_source_version():
    """The installed distribution's version must equal ``__version__``.

    Audit 2026-09-11 (V4/P1-7) MEASURED the two apart on the dev box:
    ``lumenairy.__version__`` read 5.45.1 while
    ``importlib.metadata.version('lumenairy')`` read 5.43.0, and the editable
    finder in the ``-X importtime`` trace was generated at 5.21.2 -- i.e. the
    ``.pth`` shim was 24 releases stale.  Anything reading installed metadata
    (packaging checks, a user's ``pip show``, a plugin version gate) saw the
    wrong number, and nothing compared them.

    Deliberately NOT skipped when the distribution is absent (TESTING_STANDARDS
    S3: "never ``pytest.skip`` on a resource check").  Every environment that
    runs this suite installs the package -- every CI job's install line is
    ``pip install -e ".[...]"`` -- so a missing distribution is a broken
    environment, and reporting it as a skip is how a gate silently stops
    gating.  The failure message says exactly what to run.
    """
    try:
        installed = importlib.metadata.version('lumenairy')
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        raise AssertionError(
            "lumenairy is importable but has no installed distribution "
            "metadata, so `importlib.metadata.version('lumenairy')` raises "
            "PackageNotFoundError.  The suite is running against a bare "
            "source tree (or a stale editable finder).  Run "
            "`pip install -e .` from the repository root.")
    assert installed == la.__version__, (
        f"Installed distribution metadata says lumenairy=={installed} but the "
        f"source says __version__=={la.__version__}.  The editable install is "
        f"stale: re-run `pip install -e .` from the repository root.  (This "
        f"drift is invisible to every other test -- anything reading "
        f"installed metadata, including `pip show` and packaging checks, "
        f"reports {installed}.)")


def test_phantom_name_in_dunder_all_would_be_caught():
    """Negative counter-pin: prove the public-API smoke assertion is
    NOT vacuous.

    Injects a synthetic phantom name into ``la.__all__`` that has no
    backing attribute, then asserts ``hasattr(la, phantom) is False``.
    The contract under test (``test_every_all_entry_is_resolvable``)
    would observe this failure path and FAIL -- this test proves the
    machinery works without leaving the phantom in place.

    Pattern mirrors V11's ``_DISCOVERED_CALLSITES`` counter-pin work
    from the v4.16.x walker series: every positive smoke assertion
    deserves a counter-pin proving the assertion isn't trivially
    satisfied.
    """
    # audit closure: P3-NEW-F1-4 (public-API smoke negative counter-pin)
    phantom = '__lumenairy_phantom_for_audit_counter_pin__'
    # Sanity: the phantom does NOT collide with a real attribute, and
    # it is NOT already in __all__ (counter-pin must be idempotent).
    assert not hasattr(la, phantom), (
        f"Test invariant violated: {phantom!r} unexpectedly resolves "
        f"on lumenairy.  Pick a more obscure name.")
    assert phantom not in la.__all__

    la.__all__.append(phantom)
    try:
        # The exact assertion that test_every_all_entry_is_resolvable
        # would make for ``name=phantom`` -- it MUST observe False
        # here.  If this ever flips True, the smoke assertion is
        # vacuous (e.g. someone changed `hasattr` to `name in dir`,
        # which would silently pass for any string).
        assert hasattr(la, phantom) is False, (
            f"Counter-pin failed: {phantom!r} is somehow resolvable on "
            f"lumenairy -- the public-API smoke test would silently "
            f"pass a broken __all__.  Investigate before trusting "
            f"test_every_all_entry_is_resolvable.")
        # ... and the collapsed loop assertion must SEE it.  This is the
        # half the 726-id parametrisation could not express: with one id
        # per name, nothing proved the AGGREGATE reported the failure.
        assert [n for n in la.__all__ if not hasattr(la, n)] == [phantom], (
            "The collapsed resolvability loop did not pick up the injected "
            "phantom -- it is not scanning all of __all__.")
    finally:
        # Always clean up so subsequent parametrized smoke runs aren't
        # contaminated.  Remove only OUR phantom; assert it was there.
        assert phantom in la.__all__, (
            "Phantom vanished from la.__all__ mid-test -- another "
            "test or fixture is mutating __all__ behind our back.")
        la.__all__.remove(phantom)
        assert phantom not in la.__all__


# ===========================================================================
# No library docstring may claim a version the package has not shipped
# (VERIFY-WAVE5-HYGIENE2 V-D15, 2026-09-19)
# ===========================================================================

#: Contexts in which a FORWARD version is legitimate: a deprecation horizon is
#: a scheduled future, not a claimed shipping history.
_FORWARD_VERSION_CONTEXTS = (
    'version_removed', 'version_added', 'NEXT_REMOVAL_VERSION',
    '_FROZEN_IN', 'resolve_removal_version', 'removal', 'horizon',
)

#: Files whose whole job is versioning.
_VERSION_OWNING_FILES = ('__init__.py', '_deprecation.py')

#: WHAT COUNTS AS A VERSION TOKEN, and why it is not just ``\d+\.\d+``.
#: The library is full of ordinary decimals on the same major number -- a
#: 5.92 inner-scale coefficient, a 5.53 GB memory reading, a cond() of 5.196 --
#: and a bare two-component match reads every one of them as a release.  So a
#: token qualifies only when it CANNOT be a float: three dotted components
#: (``5.48.0``), or a ``v`` prefix (``v5.48``).  MEASURED on this tree: 1181
#: tokens qualify across ``lumenairy/``, 0 of them forward -- the instrument
#: is live, not vacuous.  The one shape it cannot see is a bare two-component
#: ``5.48`` with no ``v``, which is why the seven sites this gate was written
#: for were reworded rather than renumbered.
_VER3 = re.compile(r'(?<![\w.])v?(\d+)\.(\d+)\.(\d+)(?![\w.])')
_VER2 = re.compile(r'(?<![\w.])v(\d+)\.(\d+)(?![\w.])')


def _version_tuple(text):
    m = re.match(r'^(\d+)\.(\d+)(?:\.(\d+))?', str(text))
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))


def _forward_version_tokens(line, here):
    """Every version token on ``line`` that is LATER than ``here``."""
    out = []
    for rx in (_VER3, _VER2):
        for m in rx.finditer(line):
            g = m.groups()
            tok = (int(g[0]), int(g[1]), int(g[2]) if len(g) > 2 and g[2]
                   else 0)
            if tok[0] == here[0] and tok > here:
                out.append(m.group(0))
    return out


def test_no_shipped_source_claims_a_version_the_package_has_not_reached():
    """A library docstring may record history; it may not predict a release.

    MEASURED 2026-09-19 (VERIFY-WAVE5-HYGIENE2 V-D15): seven shipped
    docstrings in ``propagators/`` read ``5.48.0`` -- "shipped since 5.48.0 as
    the opt-in", "BACKENDS (5.48.0)" and four ``(5.48.0)`` tails -- while
    ``lumenairy.__init__`` read ``5.47.0`` and the four CHANGELOG entries sat
    under ``## [Unreleased]``.  Nothing caught it, and the number was not even
    safe: ``PLAN_WAVE5_LEFTOVERS_2026_09_14.md`` already showed the next
    removal version slipping 5.48 -> 5.50, so the docstrings could have
    shipped naming a release that never existed.

    THE REPOSITORY'S OWN PRACTICE IS THE BAR, and it was measured rather than
    assumed: for each of the last five releases, mentions of the version being
    released inside ``lumenairy/**/*.py`` at the release commit's PARENT were
    0, 0, 0, 1 (a numeric table cell, a false positive) and 0, and the release
    commit itself touches exactly ONE library file -- ``lumenairy/__init__.py``
    -- to bump ``__version__``.  So the neutral form is to describe the change
    and let the CHANGELOG carry the number; the number is stamped afterwards,
    at the commit that makes it true.

    DEPRECATION HORIZONS ARE EXEMPT, and that is the whole reason the check is
    contextual rather than a grep: ``version_removed='5.48'`` and
    ``_CARRIER_FIELD_FROZEN_IN = '5.48'`` are scheduled futures that the
    deprecation machinery reads, not claims about what has shipped.
    """
    root = pathlib.Path(la.__file__).parent
    here = _version_tuple(la.__version__)
    offenders = []
    for path in sorted(root.rglob('*.py')):
        if '__pycache__' in path.parts:
            continue
        if path.name in _VERSION_OWNING_FILES:
            continue
        for lineno, line in enumerate(
                path.read_text(encoding='cp1252', errors='replace')
                .splitlines(), start=1):
            if any(ctx in line for ctx in _FORWARD_VERSION_CONTEXTS):
                continue
            for tok in _forward_version_tokens(line, here):
                offenders.append(
                    f"{path.relative_to(root.parent).as_posix()}:{lineno} "
                    f"names {tok} -- {line.strip()[:70]!r}")
    assert not offenders, (
        f"{len(offenders)} shipped source line(s) name a version later than "
        f"lumenairy.__version__ = {la.__version__}.  A docstring that says a "
        f"feature 'shipped since X' before X exists is a claim the tree "
        f"cannot support, and the number is not safe either -- release "
        f"numbering moves.  Describe the change and let the CHANGELOG carry "
        f"the version; stamp the number at the release commit, the way "
        f"__init__.py is stamped.  Deprecation horizons "
        f"({', '.join(_FORWARD_VERSION_CONTEXTS[:3])}, ...) are exempt.\n"
        + '\n'.join('  ' + o for o in offenders[:20]))


def test_the_version_gate_would_see_a_forward_claim():
    """Falsification arm: the scanner is not passing because it stopped
    looking.

    A synthetic source line naming ``<major>.<minor+9>`` must be reported, and
    the same line inside a deprecation-horizon context must NOT be.  Both
    directions, so neither the detection nor the exemption can rot silently.
    """
    here = _version_tuple(la.__version__)
    forward = f"{here[0]}.{here[1] + 9}.0"
    for line in (f'    """Shipped since {forward} as the opt-in."""',
                 f'    BACKENDS ({forward}).  The transport runs on',
                 f'    # the v{here[0]}.{here[1] + 9} dispatch'):
        assert _forward_version_tokens(line, here), (
            f"the version scanner does not see a forward claim in {line!r}")
    exempt = f"        version_removed='{forward}',"
    assert any(ctx in exempt for ctx in _FORWARD_VERSION_CONTEXTS), (
        "a deprecation horizon is no longer exempt; every scheduled removal "
        "in the library would be reported as a false claim")
    # ... a PAST version is history, not a claim ...
    past = f"    # RE-MEASURED in {here[0]}.{max(here[1] - 1, 0)}.0"
    assert not _forward_version_tokens(past, here), (
        f"the scanner reads the historical {past!r} as a forward claim")
    # ... and an ordinary decimal on the same major number is not a version.
    for benign in ('peak 5.529e+03 -> 5.486e+03',
                   'max|dc| 5.484e-15',
                   'kappa_m = 5.92/l0, kappa = 2*pi*f',
                   'cond(Hsup) = 5.196 on'):
        assert not _forward_version_tokens(benign, here), (
            f"the scanner reads {benign!r} as a forward version claim; it "
            f"would fire on ordinary numerics")
