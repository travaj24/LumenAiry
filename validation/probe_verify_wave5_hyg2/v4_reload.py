"""V4/A5 -- ``importlib.reload`` of the facade and of the leaf, plus the
MECHANISM (module-body ``STORE_NAME`` vs ``setattr``) confirmed by counting.

    python v4_reload.py <tree> <out.json>

The counting is done by wrapping the LIVE facade class's ``__setattr__``, so
every attribute assignment that goes through the descriptor protocol during
the reload is recorded by name.  A module body's ``STORE_NAME`` writes into
``module.__dict__`` directly and therefore does NOT appear; anything
``importlib`` itself sets on the module object DOES.  If the eight ever
appeared in that list the reload would raise, so the list IS the proof.
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
_OUT = os.path.abspath(sys.argv[2])
sys.path.insert(0, _TREE)

import vlib  # noqa: E402

vlib.anchor(_TREE)

from lumenairy.elements import _lens_kernels, lenses  # noqa: E402

EIGHT = ("CUPY_AVAILABLE", "_is_cupy_array", "_ensure_cupy_loaded",
         "_load_numba", "_get_aspheric_sag_accum_numba",
         "_ensure_numexpr_loaded", "_collect_semi_diameters",
         "_warn_if_aperture_exceeds_grid")


def _attempt(fn, *a):
    try:
        return ("ok", repr(fn(*a))[:120])
    except BaseException as exc:                  # noqa: BLE001 -- recorded
        return ("raised", type(exc).__name__, str(exc)[:300])


def main():
    out = {"build": vlib.build_tag(), "tree": _TREE}
    cls = type(lenses)
    out["class_before"] = cls.__name__
    seen = []
    orig = cls.__setattr__

    def counting(self, name, value):
        seen.append(name)
        return orig(self, name, value)

    cls.__setattr__ = counting
    try:
        r = _attempt(importlib.reload, lenses)
    finally:
        cls.__setattr__ = orig
    out["reload_lenses"] = r
    out["setattr_names_during_reload"] = list(seen)
    out["setattr_count_during_reload"] = len(seen)
    out["eight_seen_during_reload"] = [n for n in seen if n in EIGHT]
    out["class_after_reload"] = type(lenses).__name__
    out["class_object_is_same"] = type(lenses) is cls
    out["eight_readable_after_reload"] = {
        n: _attempt(getattr, lenses, n)[0] for n in EIGHT}
    out["eight_refused_after_reload"] = {
        n: _attempt(setattr, lenses, n, "X") for n in EIGHT}
    out["dir_len_after_reload"] = len(dir(lenses))

    # the leaf
    out["reload_kernels"] = _attempt(importlib.reload, _lens_kernels)
    out["lenses_read_after_leaf_reload"] = {
        n: _attempt(getattr, lenses, n)[0] for n in EIGHT}
    out["lenses_is_leaf_after_leaf_reload"] = {
        n: getattr(lenses, n, None) is getattr(_lens_kernels, n, None)
        for n in EIGHT}
    out["reload_lenses_again_after_leaf_reload"] = _attempt(
        importlib.reload, lenses)
    out["refusal_still_live"] = _attempt(
        setattr, lenses, "CUPY_AVAILABLE", "X")

    # a FRESH interpreter, importing the module from scratch after all of it
    code = (
        "import sys;"
        f"sys.path.insert(0, {_TREE!r});"
        "import lumenairy.elements.lenses as L;"
        "import lumenairy.elements._lens_kernels as K;"
        "print('IMPORT-OK', type(L).__name__, len(dir(L)),"
        " L.CUPY_AVAILABLE is K.CUPY_AVAILABLE);"
        "\ntry:\n L.CUPY_AVAILABLE = 1\n print('WRITE-OK')\n"
        "except AttributeError as e:\n print('WRITE-REFUSED')\n")
    env = dict(os.environ, PYTHONPATH=_TREE, PYTHONHASHSEED="0")
    pr = subprocess.run([sys.executable, "-c", code], capture_output=True,
                        text=True, env=env)
    out["fresh_subprocess"] = {"rc": pr.returncode,
                               "stdout": pr.stdout.strip(),
                               "stderr": pr.stderr.strip()[-600:]}
    vlib.write_json(out, _OUT)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
