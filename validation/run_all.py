"""Top-level entry point for the lumenairy validation suite.

Discovers every `test_*.py` in this folder and runs each in a fresh
subprocess (so matplotlib, registered glasses, cupy state, etc. don't
leak between files).  Prints a per-file result line and an aggregate
summary, exits 0 iff every file passed.

Usage::

    python validation/run_all.py                 # run everything
    python validation/run_all.py --quiet         # only per-file + summary
    python validation/run_all.py test_lenses     # run a subset
    python validation/run_all.py test_lenses.py  # same; extension optional
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time


_HERE = pathlib.Path(__file__).resolve().parent


def discover_tests() -> list[pathlib.Path]:
    files = sorted(_HERE.glob("test_*.py"))
    return [f for f in files if f.name != "_harness.py"]


def run_file(path: pathlib.Path, quiet: bool) -> tuple[int, float, str]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(_HERE),
        capture_output=True,
        text=True,
        timeout=600,
    )
    dt = time.perf_counter() - t0
    out = (proc.stdout or '') + (proc.stderr or '')
    if proc.returncode != 0 and not quiet:
        # On failure always show output so the user can diagnose
        print(out, end='' if out.endswith('\n') else '\n')
    elif not quiet:
        print(out, end='' if out.endswith('\n') else '\n')
    return proc.returncode, dt, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('selection', nargs='*',
                    help='Optional list of test files to run '
                    '(e.g. test_lenses test_raytrace.py)')
    ap.add_argument('--quiet', action='store_true',
                    help='Only print per-file pass/fail + summary')
    args = ap.parse_args()

    all_files = discover_tests()
    if args.selection:
        wanted = set()
        for s in args.selection:
            name = s if s.endswith('.py') else f'{s}.py'
            if not name.startswith('test_'):
                name = f'test_{name}'
            wanted.add(name)
        files = [f for f in all_files if f.name in wanted]
        missing = wanted - {f.name for f in files}
        if missing:
            print(f"error: no such test file(s): {sorted(missing)}")
            print(f"available: {[f.name for f in all_files]}")
            return 2
    else:
        files = all_files

    summary = []
    for f in files:
        rc, dt, _ = run_file(f, quiet=args.quiet)
        summary.append((f.name, rc, dt))

    print(f"\n{'=' * 60}")
    print(f"  run_all summary")
    print(f"{'=' * 60}")
    total_failed = 0
    for name, rc, dt in summary:
        mark = 'PASS' if rc == 0 else 'FAIL'
        total_failed += (rc != 0)
        print(f"  [{mark}] {name:35s} ({dt:5.1f}s)")
    print(f"{'=' * 60}")
    if total_failed == 0:
        print(f"  ALL {len(summary)} files passed.")
        return 0
    print(f"  {total_failed}/{len(summary)} files FAILED.")
    return 1


if __name__ == '__main__':
    sys.exit(main())
