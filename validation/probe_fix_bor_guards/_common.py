"""Shared harness for the 5.45.1 BOR multilayer-guard BUILD probes (2026-09-12).

Companion to ``validation/probe_scope_bor_guards/`` (the SCOPING harness): these
probes measure the SAME quantities on the tree that is being CHANGED, so every
one of them pins the tree from ``lumenairy.__file__`` and asserts it, and every
one records the build tag and the thread / kernel environment it ran under.

Three instruments:

* :func:`rt_fixtures` / :func:`rt_hash` -- the BIT-IDENTITY population.  A
  >= 30-fixture battery of ordinary BOR solves over both bases, ``m`` and
  ``k0``, hashed to a SHA-256 of the exact IEEE-754 bytes of ``R`` and ``T``.
  Equality of the hash is the step-1 and step-2 contract; a single moved digit
  changes it.
* :func:`inv_census` -- the scoping's explicit-inverse census, re-used verbatim
  so the populations are directly comparable.
* :func:`kernel_tag` -- what OpenBLAS kernel ACTUALLY loaded, read back from
  ``threadpoolctl`` rather than inferred from ``OPENBLAS_CORETYPE`` (on this
  host ``ZEN`` silently aliases to Haswell and ``SKYLAKEX`` dies on the first
  LAPACK call).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import traceback

TREE = r"C:\tmp\lum_bor1"
_WSL = "/mnt/c/tmp/lum_bor1"
if TREE not in sys.path:
    sys.path.insert(0, TREE)
if os.path.isdir(_WSL) and _WSL not in sys.path:
    sys.path.insert(0, _WSL)

import numpy as np  # noqa: E402


def pin_tree():
    """Refuse to run against any other checkout.  A probe that silently
    measured an installed lumenairy would report the UNFIXED numbers."""
    import lumenairy
    p = os.path.abspath(lumenairy.__file__)
    norm = p.replace("\\", "/").lower()
    if "/tmp/lum_bor1/lumenairy/" not in norm:
        raise SystemExit("WRONG TREE: " + p)
    return p


def kernel_tag():
    """The kernel that actually loaded, plus what was REQUESTED."""
    try:
        import threadpoolctl
        info = [dict(api=d.get("internal_api"), arch=d.get("architecture"),
                     nthreads=d.get("num_threads"), version=d.get("version"))
                for d in threadpoolctl.threadpool_info()]
    except Exception as exc:                       # noqa: BLE001
        info = [dict(error=str(exc))]
    return dict(requested=os.environ.get("OPENBLAS_CORETYPE"), loaded=info)


def build_tag():
    import platform

    import numpy as _np
    return dict(
        python=sys.version.split()[0],
        platform=platform.system(),
        numpy=_np.__version__,
        threads_env={k: os.environ.get(k) for k in
                     ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "MKL_NUM_THREADS")},
        kernel=kernel_tag(),
    )


def outdir():
    return os.path.dirname(os.path.abspath(__file__))


def dump(name, payload):
    payload = dict(payload)
    payload["_build"] = build_tag()
    payload["_tree"] = pin_tree()
    p = os.path.join(outdir(), name)
    with open(p, "w", encoding="cp1252", errors="replace") as fh:
        json.dump(payload, fh, indent=1, default=_jsonable)
    print("WROTE", p)
    return p


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (complex, np.complexfloating)):
        return [float(o.real), float(o.imag)]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


# --------------------------------------------------------------------------- #
#  Instrument 1: the ORDINARY-GEOMETRY bit-identity battery                    #
# --------------------------------------------------------------------------- #
def hash_arrays(*arrs):
    """SHA-256 of the exact IEEE-754 bytes.  Not a tolerance -- a hash: it is
    the only comparison that cannot silently accept a 1-ulp move."""
    h = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:32]


def build_stack(basis, m, k0, *, kind="ring", degree=8, N=160, Rbig=24.0,
                n_sub=1.5, n_sup=1.0):
    """One ordinary BOR stack.  ``kind`` selects the geometry family."""
    from lumenairy.elements.bor import BORStack
    st = BORStack(Rbig, m, basis=basis, N=N, degree=degree,
                  n_substrate=n_sub, n_superstrate=n_sup)
    st.set_source(k0=k0)
    if kind == "ring":
        st.add_layer(0.6, segments=[(6.0, 4.0), (Rbig, 2.25)])
        st.add_layer(0.4, segments=[(9.0, 2.25), (Rbig, 4.0)])
    elif kind == "uniform":
        st.add_layer(0.8, eps=2.25)
    elif kind == "three":
        st.add_layer(0.5, segments=[(5.0, 4.0), (Rbig, 2.25)])
        st.add_layer(0.3, eps=2.25)
        st.add_layer(0.5, segments=[(8.0, 2.25), (Rbig, 4.0)])
    elif kind == "lossy":
        st.add_layer(0.6, segments=[(6.0, 4.0 + 0.2j), (Rbig, 2.25)])
    else:
        raise ValueError(kind)
    return st


def rt_fixtures():
    """The >= 30-fixture ordinary battery: both bases x m x k0 x geometry.

    Chosen to contain NO near-cutoff order (the population step 2 deliberately
    moves) and every ordinary geometry family the step-4 guard must leave
    alone.  Each entry is ``(name, (basis, m, k0, kind))``.
    """
    out = []
    for basis in ("fd", "sem"):
        for m in (0, 1, 2, 5):
            for k0 in (0.8, 2.0, 3.5):
                out.append(("%s|ring|m%d|k%g" % (basis, m, k0),
                            (basis, m, k0, "ring")))
        for kind in ("uniform", "three", "lossy"):
            out.append(("%s|%s|m1|k2" % (basis, kind),
                        (basis, 1, 2.0, kind)))
    return out


def rt_hash(verbose=False):
    """Run the battery and return ``{name: {hash, n, sumR, sumT}}``."""
    pin_tree()
    res = {}
    for name, (basis, m, k0, kind) in rt_fixtures():
        try:
            st = build_stack(basis, m, k0, kind=kind)
            o = st.solve()
            R, T = np.asarray(o["R"]), np.asarray(o["T"])
            res[name] = dict(hash=hash_arrays(R, T), n=int(R.size),
                             sumR=float(np.sum(R)), sumT=float(np.sum(T)))
        except Exception as exc:                   # noqa: BLE001
            res[name] = dict(error="%s: %s" % (type(exc).__name__, exc))
        if verbose:
            print(name, res[name])
    return res


# --------------------------------------------------------------------------- #
#  Instrument 2: explicit-inverse census (the scoping harness, verbatim)       #
# --------------------------------------------------------------------------- #
def _instruments():
    from lumenairy.elements.rcwa._core import (
        _equilibrated_inverse_residual,
        _rcond_1_equilibrated,
    )
    return _rcond_1_equilibrated, _equilibrated_inverse_residual


def _site_of(depth_skip=2):
    st = traceback.extract_stack()
    for fr in reversed(st[:-depth_skip]):
        fn = fr.filename.replace("\\", "/")
        if "_common.py" in fn:
            continue
        short = fn.split("/lumenairy/")[-1] if "/lumenairy/" in fn else fn
        return "%s:%d:%s" % (short, fr.lineno, fr.name)
    return "?"


@contextlib.contextmanager
def inv_census(records, *, residual=True, only_bor=True):
    """Record every ``np.linalg.inv`` / ``np.linalg.solve`` call made from
    inside ``lumenairy/elements/bor/``."""
    rcond_f, resid_f = _instruments()
    real_inv = np.linalg.inv
    real_solve = np.linalg.solve

    def _rec(kind, A, X):
        site = _site_of()
        if only_bor and "elements/bor/" not in site.replace("\\", "/"):
            return
        A = np.asarray(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return
        try:
            Xi = X if X is not None else real_inv(A)
            rc = float(rcond_f(A, Xi))
        except Exception:                          # noqa: BLE001
            rc = float("nan")
        rs = None
        if residual:
            try:
                rs = float(resid_f(A))
            except Exception:                      # noqa: BLE001
                rs = float("nan")
        records.append(dict(site=site, kind=kind, n=int(A.shape[0]),
                            rcond=rc, resid=rs))

    def inv(A):
        X = real_inv(A)
        try:
            _rec("inv", A, X)
        except Exception:                          # noqa: BLE001
            pass
        return X

    def solve(A, B):
        X = real_solve(A, B)
        try:
            _rec("solve", A, None)
        except Exception:                          # noqa: BLE001
            pass
        return X

    np.linalg.inv = inv
    np.linalg.solve = solve
    try:
        yield records
    finally:
        np.linalg.inv = real_inv
        np.linalg.solve = real_solve
