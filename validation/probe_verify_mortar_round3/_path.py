"""Put THIS WORKTREE'S ``lumenairy`` first on ``sys.path``, and REFUSE to run
if the import resolves anywhere else.

A script run as ``python validation/probe_.../x.py`` gets the SCRIPT's
directory as ``sys.path[0]``, not the working directory, so ``import
lumenairy`` silently picks up whatever copy is installed in site-packages (on
this box, a checkout on ``D:``) -- a whole probe run's worth of numbers
measured against the wrong library, reported without a symptom.  Import this
module FIRST, before anything that touches ``lumenairy``.
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]

for p in (str(HERE), str(ROOT)):
    if p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import lumenairy  # noqa: E402

_got = pathlib.Path(lumenairy.__file__).resolve()
if ROOT.resolve() not in _got.parents:
    raise SystemExit(
        f"REFUSING TO MEASURE: lumenairy resolved to {_got}, which is not "
        f"inside this worktree {ROOT}.  Every number this probe would print "
        f"belongs to another library.")
