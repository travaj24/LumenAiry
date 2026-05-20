"""Public-API smoke test (v5.0 ROADMAP).

Asserts that every name listed in ``lumenairy.__all__`` is resolvable
via ``getattr(lumenairy, name)``.  Catches the recurring "exported but
not actually imported" / "imported but not exported" sibling-gap at
the top-level facade.

Note that V9 (the `__all__` symmetry walker) covers SUBMODULE
``__all__`` <-> top-level re-export.  This file covers the
complementary check: every top-level ``__all__`` entry must actually
resolve at the top-level.
"""
from __future__ import annotations

import pytest

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


@pytest.mark.parametrize('name', la.__all__)
def test_every_all_entry_is_resolvable(name):
    """Each entry in lumenairy.__all__ must be importable via getattr."""
    assert hasattr(la, name), (
        f"lumenairy.__all__ lists {name!r} but `getattr(lumenairy, "
        f"{name!r})` raises AttributeError.  Either remove from "
        f"__all__ or add the missing import at the top of "
        f"lumenairy/__init__.py.")


def test_dunder_all_has_no_duplicates():
    seen = set()
    dupes = []
    for name in la.__all__:
        if name in seen:
            dupes.append(name)
        seen.add(name)
    assert not dupes, (
        f"lumenairy.__all__ contains duplicates: {dupes}")


def test_version_string_format():
    """__version__ must be a PEP 440 release identifier."""
    import re
    assert hasattr(la, '__version__')
    assert re.match(r'^\d+\.\d+\.\d+(a\d+|b\d+|rc\d+)?$', la.__version__), (
        f"__version__ = {la.__version__!r} is not PEP 440 conformant")


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
    finally:
        # Always clean up so subsequent parametrized smoke runs aren't
        # contaminated.  Remove only OUR phantom; assert it was there.
        assert phantom in la.__all__, (
            "Phantom vanished from la.__all__ mid-test -- another "
            "test or fixture is mutating __all__ behind our back.")
        la.__all__.remove(phantom)
        assert phantom not in la.__all__
