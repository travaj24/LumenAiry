"""Pin the lumenairy this probe measures, and REFUSE to run if the import
resolves anywhere else.

A script run as ``python validation/probe_.../x.py`` gets the SCRIPT's
directory as ``sys.path[0]``, not the working directory, so ``import
lumenairy`` silently picks up whatever copy is installed in site-packages --
a whole probe run's worth of numbers measured against the wrong library,
reported without a symptom.  Import this module FIRST.

THIS verification needs to run the SAME probe file against TWO trees (the
round-4 tip and its parent ``15af675``), so the tree is an ENV VAR
``VMORTAR4_TREE`` rather than always the script's own worktree; when it is
unset the script's own worktree is used, exactly like the fix probe's
``_path``.  Either way the resolved ``lumenairy.__file__`` is ASSERTED to sit
under the requested tree and is recorded in every JSON.
"""
from __future__ import annotations

import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
OWN_ROOT = HERE.parents[1]
_env = os.environ.get("VMORTAR4_TREE", "").strip()
ROOT = pathlib.Path(_env).resolve() if _env else OWN_ROOT

if not (ROOT / "lumenairy" / "__init__.py").is_file():
    raise SystemExit(
        f"REFUSING TO MEASURE: {ROOT} carries no lumenairy/__init__.py.")

for p in (str(HERE), str(OWN_ROOT), str(ROOT)):
    while p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import lumenairy  # noqa: E402

_got = pathlib.Path(lumenairy.__file__).resolve()
if ROOT not in _got.parents:
    raise SystemExit(
        f"REFUSING TO MEASURE: lumenairy resolved to {_got}, which is not "
        f"inside the requested tree {ROOT}.  Every number this probe would "
        f"print belongs to another library.")

TREE = str(ROOT)
LUMENAIRY_FILE = str(_got)
