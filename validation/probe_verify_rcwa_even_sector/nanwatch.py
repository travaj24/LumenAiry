"""Pytest plugin: attribute a LAPACK ``XERBLA`` complaint to a test.

WHY.  The WSL suite run behind
``docs/audits/FIX_RCWA_EVEN_SECTOR_WSL_2026_09_11.md`` printed

    ** On entry to DLASCL parameter number  4 had an illegal value

and the audit leaves its origin open (its section 8 says a targeted hunt "would
wrap XERBLA or bisect the suite").  ``DLASCL(TYPE, KL, KU, CFROM, CTO, M, N, A,
LDA, INFO)`` sets ``INFO = -4`` when ``CFROM`` is ZERO or NaN, and ``D``LASCL is
the DOUBLE-REAL routine, so the emitting call is a REAL-valued LAPACK driver
handed a non-finite (or zero) scale -- typically a NaN that reached a norm.

Fortran unit-6 output is buffered and flushed at interpreter exit, so the line's
position in a log attributes nothing.  This plugin instead records, per test
nodeid, every ``numpy.linalg`` / ``scipy.linalg`` call whose INPUT or OUTPUT is
non-finite, so a suite run names the tests that could be responsible.  Combine
with per-shard stderr capture: the shard whose stderr carries the line, crossed
with the nodeids this plugin flagged in it, is the answer.

Enable with ``-p nanwatch`` (the file's directory must be importable, e.g.
``PYTHONPATH=validation/probe_verify_rcwa_even_sector``).  Writes
``nanwatch_report.json`` in the working directory at session end.
"""
from __future__ import annotations

import json
import os

import numpy as np

_CURRENT = {"nodeid": "<collection>"}
_HITS = {}
_PATCHED = []


def _check(tag, args, kwargs, out):
    bad_in, bad_out, real_in = [], [], False
    for i, a in enumerate(list(args) + list(kwargs.values())):
        try:
            arr = np.asarray(a)
        except Exception:                                     # pragma: no cover
            continue
        if arr.dtype.kind not in "fc" or arr.size == 0:
            continue
        if arr.dtype.kind == "f":
            real_in = True
        if not np.all(np.isfinite(arr)):
            bad_in.append("arg%d(%s)" % (i, arr.dtype))
    outs = out if isinstance(out, tuple) else (out,)
    for j, o in enumerate(outs):
        try:
            arr = np.asarray(o)
        except Exception:                                     # pragma: no cover
            continue
        if arr.dtype.kind not in "fc" or arr.size == 0:
            continue
        if not np.all(np.isfinite(arr)):
            bad_out.append("out%d(%s)" % (j, arr.dtype))
    if bad_in or bad_out:
        rec = _HITS.setdefault(_CURRENT["nodeid"], [])
        entry = dict(fn=tag, bad_inputs=bad_in, bad_outputs=bad_out,
                     real_dtype_input=real_in)
        if entry not in rec:
            rec.append(entry)


def _wrap(mod, name, tag):
    fn = getattr(mod, name, None)
    if fn is None or not callable(fn):
        return

    def wrapped(*a, **kw):
        try:
            out = fn(*a, **kw)
        except Exception:
            _check(tag + "(RAISED)", a, kw, None)
            raise
        _check(tag, a, kw, out)
        return out

    wrapped.__name__ = name
    setattr(mod, name, wrapped)
    _PATCHED.append((mod, name, fn))


def pytest_configure(config):
    import numpy.linalg as nl
    names = ("eig", "eigh", "eigvals", "eigvalsh", "svd", "solve", "inv",
             "lstsq", "qr", "cholesky", "det", "slogdet", "pinv", "cond",
             "norm", "matrix_rank")
    for n in names:
        _wrap(nl, n, "numpy.linalg." + n)
    try:
        import scipy.linalg as sl
        for n in ("eig", "eigh", "eigvals", "svd", "svdvals", "solve", "inv",
                  "lu", "qr", "schur", "expm", "lstsq", "cholesky"):
            _wrap(sl, n, "scipy.linalg." + n)
    except Exception:                                         # pragma: no cover
        pass


def pytest_runtest_setup(item):
    _CURRENT["nodeid"] = item.nodeid


def pytest_sessionfinish(session, exitstatus):
    path = os.environ.get("NANWATCH_OUT", "nanwatch_report.json")
    payload = dict(hits=_HITS, n_tests_with_hits=len(_HITS),
                   exitstatus=int(exitstatus))
    with open(path, "w", encoding="cp1252", errors="replace") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    print("\n[nanwatch] %d tests handed a non-finite array to LAPACK -> %s"
          % (len(_HITS), path))
