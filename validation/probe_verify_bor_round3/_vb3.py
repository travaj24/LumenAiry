"""Shared fixtures and arm bookkeeping for the INDEPENDENT verification of
ROUND 3 of the 5.45.1 BOR/EME guards.

Nothing here is imported from ``validation/probe_fix_bor_round3`` or from
either round-2 probe directory.  The geometries are deliberately DIFFERENT from
the fix round's -- different ``k0`` (3.0 vs 2.0), different half-space index
(``eps`` = 2.25, i.e. n = 1.5, vs 2.0), different middle-layer families,
different thickness -- so a number that reproduces is corroboration rather than
an echo of the same arithmetic.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import warnings

import numpy as np

K0 = 3.0
EPS_HALF = 2.25
THICK = 0.42


# ------------------------------------------------------------------ arm/JSON #
def arm():
    """The build, the LOADED OpenBLAS kernel (read back from threadpoolctl,
    never inferred from the request), and the thread count."""
    info = []
    try:
        import threadpoolctl
        info = threadpoolctl.threadpool_info()
    except Exception:                                    # pragma: no cover
        pass
    blas = [d for d in info if d.get("user_api") == "blas"]
    import scipy

    import lumenairy
    return dict(
        build=("wsl" if sys.platform.startswith("linux") else "win"),
        python=sys.version.split()[0],
        numpy=np.__version__,
        scipy=scipy.__version__,
        lumenairy_file=lumenairy.__file__,
        lumenairy_version=getattr(lumenairy, "__version__", "?"),
        requested_coretype=os.environ.get("OPENBLAS_CORETYPE", ""),
        loaded_kernel=(blas[0].get("architecture") if blas else None),
        threads=(blas[0].get("num_threads") if blas else None),
        omp=os.environ.get("OMP_NUM_THREADS", ""),
        openblas=os.environ.get("OPENBLAS_NUM_THREADS", ""),
        mkl=os.environ.get("MKL_NUM_THREADS", ""),
        machine=platform.machine(),
    )


def require_tree(a=None):
    """Refuse to measure the WRONG TREE.

    This host carries an editable install of ``lumenairy`` elsewhere on
    ``sys.path``.  Run as ``python script.py`` the script's own directory is
    ``sys.path[0]`` and the working directory is NOT on the path, so a probe
    launched from inside a worktree without ``PYTHONPATH`` imports the OTHER
    tree -- silently, with a module whose private names differ.  ``python -c``
    hides it (the cwd is ``sys.path[0]`` there), which is how it nearly
    survived.  Set ``LUM_EXPECT_ROOT`` and this hard-fails instead."""
    root = os.environ.get("LUM_EXPECT_ROOT", "")
    if not root:
        return
    a = a or arm()
    got = os.path.normcase(os.path.abspath(a["lumenairy_file"]))
    want = os.path.normcase(os.path.abspath(root))
    if not got.startswith(want):
        raise SystemExit(
            "WRONG TREE: lumenairy resolved to %s, expected it under %s -- set "
            "PYTHONPATH to the worktree" % (a["lumenairy_file"], root))


def tag(a=None):
    a = a or arm()
    pre = os.environ.get("LUM_PROBE_TAG", "")
    return "%s%s_%s_t%s" % (pre + "_" if pre else "", a["build"],
                            a["loaded_kernel"], a["threads"])


def dump(name, payload, a=None):
    a = a or arm()
    payload = dict(payload)
    payload["arm"] = a
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "%s_%s.json" % (name, tag(a)))
    with open(path, "w", encoding="cp1252", errors="replace") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True, default=_jsonable)
    print("WROTE", path)
    return path


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return [o.real, o.imag]
    return str(o)


# ---------------------------------------------------------------- geometries #
def uni(eps):
    return lambda r: np.full(np.shape(r), complex(eps), dtype=complex)


def para(Rbig, lo=2.2, hi=5.2):
    """A radially GRADED middle layer -- eps rises quadratically to the wall."""
    def f(r):
        x = np.asarray(r, float) / float(Rbig)
        return complex(lo) + (complex(hi) - complex(lo)) * x ** 2
    return f


def grate(Rbig, n=3, duty=0.35, lo=1.8, hi=5.5):
    per = float(Rbig) / n

    def f(r):
        frac = np.mod(np.asarray(r, float), per) / per
        return np.where(frac < duty, complex(hi), complex(lo)).astype(complex)
    return f


def core(Rbig, rc=0.25, inside=6.0, outside=2.1):
    def f(r):
        r = np.asarray(r, float)
        return np.where(r < rc * float(Rbig), complex(inside),
                        complex(outside)).astype(complex)
    return f


def bilayer(Rbig, split=0.4, a=2.6, b=5.8):
    def f(r):
        r = np.asarray(r, float)
        return np.where(r < split * float(Rbig), complex(a),
                        complex(b)).astype(complex)
    return f


#: DAMAGING families (structured, high index contrast) and BENIGN ones.
FAMILIES = ("para", "grate", "core", "bilayer", "unif35", "unif23")


def mid_profile(family, Rbig):
    if family == "para":
        return para(Rbig)
    if family == "grate":
        return grate(Rbig)
    if family == "core":
        return core(Rbig)
    if family == "bilayer":
        return bilayer(Rbig)
    if family == "unif35":
        return uni(3.5)
    if family == "unif23":
        return uni(2.3)
    raise ValueError(family)


def _lossy(fn, im_rel, on):
    if not on or im_rel == 0.0:
        return fn

    def g(r):
        e = np.asarray(fn(r), dtype=complex)
        return e + 1j * float(im_rel) * np.real(e)
    return g


def stack(basis, family, m, N, rbl, im_rel=0.0, where="inc", k0=K0,
          e_half=EPS_HALF, thickness=THICK):
    """Three layers: two ``e_half`` half-spaces around one structured middle
    layer.  ``im_rel`` = ``Im(eps)/Re(eps)`` applied at ``where``: ``inc`` =
    ``layers[0]``, ``exit`` = ``layers[-1]``, ``both`` = the two half-spaces,
    ``mid`` = the structured layer, ``all`` = everywhere, ``none`` = nowhere."""
    from lumenairy.elements.bor.bor_solve import build_layer
    Rbig = float(rbl) * 2.0 * np.pi / float(k0)
    h0 = _lossy(uni(e_half), im_rel, where in ("inc", "both", "all"))
    h1 = _lossy(uni(e_half), im_rel, where in ("exit", "both", "all"))
    mid = _lossy(mid_profile(family, Rbig), im_rel, where in ("mid", "all"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [build_layer(m, Rbig, N, h0, k0, basis=basis),
                build_layer(m, Rbig, N, mid, k0, basis=basis,
                            thickness=thickness),
                build_layer(m, Rbig, N, h1, k0, basis=basis)]


# ------------------------------------------------------------------- solving #
def disarmed(layers, k0=K0):
    """Solve with the nodal passivity guard OFF, so a decision can be scored
    against a number the screen did not produce."""
    import lumenairy.elements.bor.bor_solve as bs
    prev = bs.BOR_NODAL_PASSIVITY_GUARD
    bs.BOR_NODAL_PASSIVITY_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return bs.solve(layers, k0)
    finally:
        bs.BOR_NODAL_PASSIVITY_GUARD = prev


def armed(layers, k0=K0):
    """``(verdict, detector, n_energy_warnings)`` as a caller sees it."""
    import lumenairy.elements.bor.bor_solve as bs
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            bs.solve(layers, k0)
            v, det = "returned", ""
        except bs.BORNodalPassivityError as exc:
            v = "REFUSED"
            det = ("ceiling" if "axial index" in str(exc) else "energy")
        nw = len([x for x in w if "R + T" in str(x.message)])
    return v, det, nw


def ceiling_excess(layers, res, k0=K0):
    """``(inc, exit)`` index-ceiling excess over the RETURNED channels --
    exactly the quantity ``_check_nodal_passivity`` reads, evaluated here
    regardless of whether the half-space is lossless."""
    import lumenairy.elements.bor.bor_solve as bs
    out = []
    for L, idx in ((layers[0], res["inc"]), (layers[-1], res["out"])):
        qn = np.asarray(L["q"])[np.asarray(idx, int)] / k0
        out.append(bs._channel_index_excess(L, qn))
    return out[0], out[1]


def answer_hash(res):
    """SHA-256 over the exact IEEE-754 bytes of a solve's answer."""
    h = hashlib.sha256()
    for key in ("R", "T", "energy", "inc", "out", "q_inc"):
        a = np.ascontiguousarray(np.asarray(res[key]))
        h.update(key.encode("ascii"))
        h.update(str(a.dtype).encode("ascii"))
        h.update(str(a.shape).encode("ascii"))
        h.update(a.tobytes())
    for blk in res["S"]:
        h.update(np.ascontiguousarray(np.asarray(blk)).tobytes())
    return h.hexdigest()
