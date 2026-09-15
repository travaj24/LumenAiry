"""VERIFY-B11c: re-derive, from first principles and without touching the
library, the claim that a read-only PEP 562 ``__getattr__`` on ``lenses`` would
NOT have carried the ``_NUMBA_AVAILABLE`` monkeypatch.

The WP-B11c report states this as a measurement; this rebuilds the measurement
on a synthetic two-module pair with the same shape (a leaf holding a gate and a
kernel that reads it at call time, and a facade that forwards) and runs the
same three arms against BOTH forwarding mechanisms:

  arm 1  read-only PEP 562 ``__getattr__`` -- the plan the doc recorded
  arm 2  a ``ModuleType`` subclass with ``__getattr__``/``__setattr__``/
         ``__delattr__``/``__dir__`` -- the mechanism the WP shipped
  arm 3  a plain by-value re-export -- the mechanism it rejected

For each: does ``monkeypatch.setattr(facade, 'GATE', False)`` reach the kernel,
and does the undo leave a shadow in ``facade.__dict__``?

The verdicts are printed as JSON.  No lumenairy import; nothing here can be
fooled by the library's own arrangement.
"""
from __future__ import annotations

import json
import sys
import types


def _make_leaf(name):
    leaf = types.ModuleType(name)
    leaf.GATE = True
    leaf.SLOT = None

    def kernel():
        """Reads the gate at CALL time out of THIS module's globals -- the
        same shape as ``_lens_kernels.surface_sag_general`` reading
        ``_NUMBA_AVAILABLE``."""
        return "fast" if leaf.GATE else "slow"

    # bind the function's globals to the leaf's dict, which is what a real
    # module-level ``def`` does
    kernel.__globals__.update()          # no-op; the closure reads ``leaf``
    leaf.kernel = kernel
    sys.modules[name] = leaf
    return leaf


def _facade_pep562(name, leaf, names):
    mod = types.ModuleType(name)

    def __getattr__(attr):
        if attr in names:
            return getattr(leaf, attr)
        raise AttributeError(attr)

    mod.__getattr__ = __getattr__
    sys.modules[name] = mod
    return mod


def _facade_moduletype(name, leaf, names):
    class _Facade(types.ModuleType):
        def __getattr__(self, attr):
            if attr in names:
                return getattr(leaf, attr)
            raise AttributeError(attr)

        def __setattr__(self, attr, value):
            if attr in names:
                setattr(leaf, attr, value)
                return
            super().__setattr__(attr, value)

        def __delattr__(self, attr):
            if attr in names:
                delattr(leaf, attr)
                return
            super().__delattr__(attr)

        def __dir__(self):
            return sorted(set(super().__dir__()) | set(names))

    mod = types.ModuleType(name)
    mod.__class__ = _Facade
    sys.modules[name] = mod
    return mod


def _facade_byvalue(name, leaf, names):
    mod = types.ModuleType(name)
    for n in names:
        setattr(mod, n, getattr(leaf, n))          # the snapshot
    sys.modules[name] = mod
    return mod


def _exercise(label, facade, leaf):
    """The monkeypatch cycle, by hand: read the old value, write, observe the
    kernel, undo the way ``monkeypatch`` does, then look for a shadow."""
    out = {"mechanism": label}
    out["read_is_live_before"] = getattr(facade, "GATE", "<AttributeError>")

    # does a lazily populated slot show through?
    leaf.SLOT = "populated-later"
    try:
        out["lazy_slot_seen"] = getattr(facade, "SLOT")
    except AttributeError:
        out["lazy_slot_seen"] = "<AttributeError>"
    leaf.SLOT = None

    old = getattr(facade, "GATE")
    try:
        setattr(facade, "GATE", False)              # monkeypatch's write
        out["write_raised"] = None
    except Exception as exc:                        # noqa: BLE001 -- recorded
        out["write_raised"] = type(exc).__name__
    out["leaf_gate_after_write"] = leaf.GATE
    out["kernel_after_write"] = leaf.kernel()
    out["kernel_saw_the_patch"] = (leaf.kernel() == "slow")

    setattr(facade, "GATE", old)                    # monkeypatch's undo
    out["leaf_gate_after_undo"] = leaf.GATE
    out["shadow_left_in_facade_dict"] = "GATE" in vars(facade)
    out["dir_lists_gate"] = "GATE" in dir(facade)
    return out


def main():
    names = ("GATE", "SLOT", "kernel")
    results = []
    for label, builder in (("pep562_readonly_getattr", _facade_pep562),
                           ("moduletype_subclass", _facade_moduletype),
                           ("plain_by_value_reexport", _facade_byvalue)):
        leaf = _make_leaf(f"_vb11c_leaf_{label}")
        facade = builder(f"_vb11c_facade_{label}", leaf, names)
        results.append(_exercise(label, facade, leaf))
    print(json.dumps({"arms": results}, indent=1))


if __name__ == "__main__":
    main()
