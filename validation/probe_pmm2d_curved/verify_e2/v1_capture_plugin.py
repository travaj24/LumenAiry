"""E2 verifier item 1 -- CAPTURE plugin: hash EVERY solve output of the
shipped per-layer mortar suites, exactly as the shipped tests construct their
stacks, so PRE (eae470d9) and POST can be compared byte for byte without
re-transcribing a single fixture.

Load with ``-p v1_capture_plugin`` (PYTHONPATH = <tree>;<verify_e2>).  Wraps
(at plugin import, before any test module is collected):

* ``PMM2DStackPure.solve / _solve_per_layer / layer_absorption /
  convergence_floor``
* ``PMMStack.solve / layer_absorption`` (the 1-D per-layer surface)
* ``_interface_smatrix_mortar_2d`` / ``_interface_smatrix_general_mortar_2d``
  in BOTH ``_core`` and the ``stack2d_pure`` namespace (the S-matrices of
  every mortared interface)

Each call is keyed ``<nodeid>::<what>#<k>`` (k = per-test call counter) and
stores the SHA-256 of the output (or of the exception type + message).
Output: ``$V1_CAPTURE_OUT`` (a JSON path) at session end.
"""
import hashlib
import json
import os
import sys

import numpy as np

import lumenairy
from lumenairy.elements.pmm import PMM2DStackPure, PMMStack, _core as _CORE, stack2d_pure as _SP

_REC = {}
_CUR = {"node": "<collect>"}
_CNT = {}
_OUT = {}


def _upd(h, v):
    if v is None:
        h.update(b"None")
    elif isinstance(v, (tuple, list)):
        h.update(f"seq{len(v)}".encode())
        for x in v:
            _upd(h, x)
    elif isinstance(v, dict):
        h.update(f"dict{len(v)}".encode())
        for k in sorted(v, key=repr):
            h.update(repr(k).encode())
            _upd(h, v[k])
    elif isinstance(v, (np.ndarray, np.generic, int, float, complex, bool)):
        a = np.ascontiguousarray(np.asarray(v))
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        if a.dtype == object:
            for x in a.ravel():
                _upd(h, x)
        else:
            h.update(a.tobytes())
    elif isinstance(v, str):
        h.update(v.encode())
    else:
        h.update(("obj:" + type(v).__name__).encode())


def _sha(v):
    h = hashlib.sha256()
    _upd(h, v)
    return h.hexdigest()


def _record(what, val):
    node = _CUR["node"]
    k = _CNT.get((node, what), 0)
    _CNT[(node, what)] = k + 1
    _REC[f"{node}::{what}#{k}"] = val


def _wrap(owner, name, label):
    orig = getattr(owner, name)

    def w(*a, **kw):
        try:
            out = orig(*a, **kw)
        except Exception as e:  # noqa: BLE001 -- recorded and re-raised
            _record(label, "EXC:" + type(e).__name__ + ":" + _sha(str(e)))
            raise
        _record(label, _sha(out))
        return out

    w.__wrapped__ = orig
    setattr(owner, name, w)


for _n in ("solve", "_solve_per_layer", "layer_absorption",
           "convergence_floor"):
    _wrap(PMM2DStackPure, _n, "PMM2DStackPure." + _n)
for _n in ("solve", "layer_absorption"):
    _wrap(PMMStack, _n, "PMMStack." + _n)
for _mod, _tag in ((_CORE, "core"), (_SP, "sp")):
    for _n in ("_interface_smatrix_mortar_2d",
               "_interface_smatrix_general_mortar_2d"):
        if hasattr(_mod, _n):
            _wrap(_mod, _n, f"{_tag}.{_n}")


def pytest_runtest_setup(item):
    _CUR["node"] = item.nodeid


def pytest_runtest_logreport(report):
    if report.when == "call" or (report.when == "setup" and report.outcome
                                 != "passed"):
        _OUT[report.nodeid] = report.outcome


def pytest_sessionfinish(session, exitstatus):
    out = os.environ.get("V1_CAPTURE_OUT")
    if not out:
        return
    env = {"python": sys.version.split()[0], "numpy": np.__version__,
           "lumenairy": lumenairy.__file__,
           "threads": {k: os.environ.get(k) for k in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS")}}
    with open(out, "w") as f:
        json.dump({"env": env, "sha": _REC, "outcomes": _OUT,
                   "exitstatus": int(exitstatus)}, f, indent=1)
