"""V4 -- does the patch REACH the code, measured behaviourally rather than by
identity?  This is the claim under D3 and under the refusal message's "the
same statement against _lens_kernels does what you mean".

    python v4_reach.py <tree> <out.json>

Three measurements, all on ``surface_sag_general``, which reads
``_is_cupy_array`` out of ``_lens_kernels``'s globals at call time
(``_lens_kernels.py:715``):

  facade_patch  : substitute on ``lenses``          -> was the fake CALLED?
  leaf_patch    : substitute on ``_lens_kernels``   -> was the fake CALLED?
  restore_cycle : the try/finally restore a test would write

On the BASE tree the facade patch SUCCEEDS and the fake is never called (the
silent no-op).  On the BRANCH the facade patch RAISES.  Both trees agree that
the leaf patch reaches the code.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
_OUT = os.path.abspath(sys.argv[2])
sys.path.insert(0, _TREE)

import numpy as np  # noqa: E402

import vlib  # noqa: E402

vlib.anchor(_TREE)

from lumenairy.elements import _lens_kernels, lenses  # noqa: E402

H = np.linspace(0.0, 0.01, 33) ** 2


def _attempt(fn, *a):
    try:
        return ("ok", repr(fn(*a))[:120])
    except BaseException as exc:                  # noqa: BLE001 -- recorded
        return ("raised", type(exc).__name__, str(exc)[:160])


def main():
    real = _lens_kernels._is_cupy_array
    out = {"build": vlib.build_tag(), "tree": _TREE}

    calls = []

    def fake(x):
        calls.append(1)
        return False

    # --- 1. patch on the FACADE
    saved_shell = (("_is_cupy_array" in vars(lenses)),
                   vars(lenses).get("_is_cupy_array"))
    calls.clear()
    res = _attempt(setattr, lenses, "_is_cupy_array", fake)
    sag = _attempt(lenses.surface_sag_general, H, 0.05, -0.5, None)
    out["facade_patch"] = {
        "setattr": res,
        "fake_calls_during_sag": len(calls),
        "sag_ran": sag[0],
        "leaf_still_real": _lens_kernels._is_cupy_array is real,
        "shell_dict_entry_now": repr(
            vars(lenses).get("_is_cupy_array"))[:80],
    }
    if saved_shell[0]:
        vars(lenses)["_is_cupy_array"] = saved_shell[1]
    else:
        vars(lenses).pop("_is_cupy_array", None)

    # --- 2. patch on the LEAF (the redirect the message names)
    calls.clear()
    vars(_lens_kernels)["_is_cupy_array"] = fake
    sag2 = _attempt(lenses.surface_sag_general, H, 0.05, -0.5, None)
    out["leaf_patch"] = {
        "fake_calls_during_sag": len(calls),
        "sag_ran": sag2[0],
        "facade_read_during_patch_is_fake":
            getattr(lenses, "_is_cupy_array") is fake,
        "facade_read_during_patch_is_real":
            getattr(lenses, "_is_cupy_array") is real,
    }
    vars(_lens_kernels)["_is_cupy_array"] = real

    # --- 3. the try/finally restore cycle a test would write, THROUGH the
    #        facade: save, set, restore.  All three are writes.
    saved = getattr(lenses, "_is_cupy_array")
    r_set = _attempt(setattr, lenses, "_is_cupy_array", fake)
    r_restore = _attempt(setattr, lenses, "_is_cupy_array", saved)
    out["restore_cycle_on_facade"] = {"set": r_set, "restore": r_restore}
    if saved_shell[0]:
        vars(lenses)["_is_cupy_array"] = saved_shell[1]

    # --- 4. the IDEMPOTENT self-assignment
    out["idempotent_self_assign"] = {
        n: _attempt(setattr, lenses, n, getattr(lenses, n))
        for n in ("CUPY_AVAILABLE", "_is_cupy_array",
                  "_warn_if_aperture_exceeds_grid")}
    for n in ("CUPY_AVAILABLE", "_is_cupy_array",
              "_warn_if_aperture_exceeds_grid"):
        vars(lenses)[n] = getattr(_lens_kernels, n)

    vlib.write_json(out, _OUT)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
