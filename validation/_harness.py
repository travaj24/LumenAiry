"""Shared test harness for the lumenairy validation suite.

Every topic-based test file imports `Harness` and creates one instance
at module import time (or inside a `main()`), then uses:

    H = Harness('propagation')
    H.section('ASM basics')
    H.run('plane-wave phase accumulation', lambda: (cond, detail))
    H.check('something simple', cond, optional_msg)
    sys.exit(H.summary())

The two patterns (`run` for thunked lazy tests that may raise, `check`
for inline assertions) are both supported so we can port test files
without rewriting their body logic.

Run the whole suite via `run_all.py`, which discovers every
`test_*.py` and invokes each file's `main()` in its own subprocess so
global state (matplotlib, registered glasses, etc.) does not leak
between suites.
"""
from __future__ import annotations

import os
import sys
import traceback
import warnings


_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_ROOT = os.path.normpath(os.path.join(_HERE, '..'))
if _LIB_ROOT not in sys.path:
    sys.path.insert(0, _LIB_ROOT)
warnings.simplefilter('ignore')


class Harness:
    """Lightweight test harness.

    Does NOT abort on first failure — collects results, prints a
    summary, and returns an exit-code suitable for `sys.exit`.
    """

    def __init__(self, name: str, verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.results: list[tuple[str, bool, str]] = []
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  {name}")
            print(f"{'=' * 60}")

    def section(self, title: str) -> None:
        if self.verbose:
            print(f"\n-- {title} --")

    def _record(self, label: str, ok: bool, detail: str) -> None:
        self.results.append((label, ok, detail))
        if ok:
            self.passed += 1
            if self.verbose:
                line = f"  [OK  ] {label}"
                if detail:
                    line += f"   -- {detail}"
                print(line)
        else:
            self.failed += 1
            line = f"  [FAIL] {label}"
            if detail:
                line += f"   -- {detail}"
            print(line)

    def run(self, label: str, fn) -> None:
        """Run a thunked test. `fn` returns bool or (bool, detail_str)."""
        try:
            result = fn()
            if isinstance(result, tuple):
                ok, detail = bool(result[0]), str(result[1])
            else:
                ok, detail = bool(result), ''
        except Exception as exc:
            ok = False
            tb = traceback.format_exc(limit=1).splitlines()
            tail = tb[-1] if tb else repr(exc)
            detail = f"EXCEPTION: {tail}"
        self._record(label, ok, detail)

    def check(self, label: str, ok: bool, detail: str = '') -> None:
        """Record an inline boolean assertion."""
        self._record(label, bool(ok), detail)

    def summary(self) -> int:
        total = self.passed + self.failed
        print(f"\n{'=' * 60}")
        print(f"RESULT [{self.name}]: {self.passed}/{total} passed "
              f"({self.failed} failed)")
        print(f"{'=' * 60}")
        return 0 if self.failed == 0 else 1
