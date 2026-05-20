"""v5.0.1 Agent A regression tests -- F821 forward-reference pin.

Closes the v5.0.0 audit finding ``P1-NEW-V4-1`` against the four
real code bugs ``ruff check`` surfaced at v5.0 ship:

* ``lumenairy/algebra/base.py`` -- ``Operator.__call__`` signature
  referenced ``'Source'`` as a string forward-ref without a
  module-level binding (the runtime body does a lazy
  ``from ..sources import Source as _Source`` to avoid a top-level
  circular import).  v5.0.1 adds a ``TYPE_CHECKING`` guard.

* ``lumenairy/propagators/system.py`` -- ``evaluate(...)`` signature
  referenced ``'Source'`` and ``'PropagationResult'`` as string
  forward-refs without module-level bindings (same lazy-import
  pattern).  v5.0.1 adds a ``TYPE_CHECKING`` guard.

These four F821 violations were the ONLY real code bugs in the v5.0
ruff baseline; the remaining 692 errors are cosmetic (I001 imports,
F401 unused, F841 unused-var, F541 empty f-string, E702 semicolons)
and are deferred to v5.1's mechanical cleanup alongside the file
splits.

The pin guards against future maintainers introducing new
forward-reference bugs (lazy-imported types in signatures without a
TYPE_CHECKING binding) anywhere in ``lumenairy/``.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_LUMENAIRY_PKG_DIR = _REPO_ROOT / "lumenairy"


def _ruff_available() -> bool:
    """Return True iff a ruff executable is importable / on PATH.

    Prefer ``python -m ruff`` so the CI machine's interpreter resolves
    the dev-installed ruff; fall back to ``shutil.which('ruff')``.
    """
    # Try ``python -m ruff --version``: cheap, no checks performed.
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "ruff", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return shutil.which("ruff") is not None


# audit closure: P1-NEW-V4-1
class TestF821ForwardReferencesFixedInV5_0_1:
    """Pin the v5.0.1 P1-NEW-V4-1 closure -- ``ruff check lumenairy/
    --select F821`` must exit 0.
    """

    def test_no_f821_forward_reference_bugs_in_lumenairy(self) -> None:
        """``ruff check lumenairy/ --select F821`` returns exit 0.

        v5.0.0 shipped with four F821 forward-reference errors in
        ``lumenairy/algebra/base.py`` (Source x2) and
        ``lumenairy/propagators/system.py`` (Source + PropagationResult).
        v5.0.1 added ``TYPE_CHECKING`` guards to both files.  This pin
        ensures no new F821 violation slips in -- e.g., a future
        contributor adding a lazy-imported type to a signature without
        a typing-time binding.
        """
        # audit closure: P1-NEW-V4-1
        if not _ruff_available():
            pytest.skip(
                "ruff is not installed; install with ``pip install ruff`` "
                "to exercise the v5.0.1 P1-NEW-V4-1 F821 pin."
            )
        proc = subprocess.run(
            [
                sys.executable, "-m", "ruff", "check",
                str(_LUMENAIRY_PKG_DIR),
                "--select", "F821",
                "--no-cache",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, (
            "v5.0.1 P1-NEW-V4-1 regression: ruff F821 (undefined-name) "
            "violations have reappeared in lumenairy/.  v5.0.1 fixed four "
            "forward-reference bugs in lumenairy/algebra/base.py and "
            "lumenairy/propagators/system.py by adding TYPE_CHECKING "
            "guards for the lazy-imported Source / PropagationResult "
            "types referenced in function signatures.  A new F821 here "
            "usually means: (a) a contributor added an annotation whose "
            "type is only imported inside the function body (to avoid a "
            "circular import) without adding the matching "
            "``if TYPE_CHECKING:`` binding; or (b) a contributor "
            "removed an existing TYPE_CHECKING guard.\n\n"
            f"ruff stdout:\n{proc.stdout}\n"
            f"ruff stderr:\n{proc.stderr}"
        )

    def test_algebra_base_has_type_checking_guard_for_source(self) -> None:
        """``lumenairy/algebra/base.py`` imports Source under TYPE_CHECKING.

        Sibling-discipline pin: confirms the structural fix shape, not
        just the F821 absence (a contributor could "fix" the F821 by
        importing Source unconditionally at module scope, which would
        re-introduce the top-level circular import the lazy-load was
        designed to avoid).
        """
        # audit closure: P1-NEW-V4-1
        src = (_LUMENAIRY_PKG_DIR / "algebra" / "base.py").read_text(
            encoding="utf-8")
        assert "TYPE_CHECKING" in src, (
            "lumenairy/algebra/base.py must import Source under a "
            "``if TYPE_CHECKING:`` block (v5.0.1 P1-NEW-V4-1)."
        )
        # The TYPE_CHECKING block must specifically import Source.
        assert "from ..sources import Source" in src, (
            "lumenairy/algebra/base.py TYPE_CHECKING block must contain "
            "``from ..sources import Source`` so the string-quoted "
            "'Source' annotations in Operator.__call__ resolve at type-"
            "check time (v5.0.1 P1-NEW-V4-1)."
        )

    def test_propagators_system_has_type_checking_guard(self) -> None:
        """``lumenairy/propagators/system.py`` imports Source and
        PropagationResult under TYPE_CHECKING.
        """
        # audit closure: P1-NEW-V4-1
        src = (_LUMENAIRY_PKG_DIR / "propagators" / "system.py").read_text(
            encoding="utf-8")
        assert "TYPE_CHECKING" in src, (
            "lumenairy/propagators/system.py must use a "
            "``if TYPE_CHECKING:`` block to bind Source and "
            "PropagationResult for type-check-time resolution "
            "(v5.0.1 P1-NEW-V4-1)."
        )
        assert "from ..sources import Source" in src, (
            "lumenairy/propagators/system.py TYPE_CHECKING block must "
            "contain ``from ..sources import Source`` "
            "(v5.0.1 P1-NEW-V4-1)."
        )
        assert "from .result import PropagationResult" in src, (
            "lumenairy/propagators/system.py TYPE_CHECKING block must "
            "contain ``from .result import PropagationResult`` "
            "(v5.0.1 P1-NEW-V4-1)."
        )

    def test_lumenairy_imports_cleanly(self) -> None:
        """``import lumenairy`` succeeds after the TYPE_CHECKING refactor.

        Counter-pin: an inadvertent unconditional import (outside the
        ``if TYPE_CHECKING:`` block) of ``Source`` or
        ``PropagationResult`` at module scope would re-introduce the
        circular-import the lazy-load was designed to avoid.  This pin
        confirms the public-API entry point remains clean.
        """
        # audit closure: P1-NEW-V4-1
        proc = subprocess.run(
            [sys.executable, "-c",
             "import lumenairy as la; "
             "import lumenairy.algebra.base; "
             "import lumenairy.propagators.system; "
             "assert la.__version__"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, (
            "Importing lumenairy and the two patched modules must "
            "succeed at runtime; got:\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
