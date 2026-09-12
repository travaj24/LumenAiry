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
