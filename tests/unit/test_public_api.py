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
