"""Shared fixtures and arm bookkeeping for the ROUND 3 BOR-guard probes.

Nothing here is imported from ``validation/probe_fix_bor_round2`` or
``validation/probe_verify_bor_round2``: round 3 re-measures on its own
geometries so a number that reproduces is evidence rather than an echo.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import warnings

import numpy as np


def arm():
    """The build, the LOADED OpenBLAS kernel (read back, never inferred from
    the request), and the thread count -- recorded in every JSON."""
    info = []
    try:
        import threadpoolctl
        info = threadpoolctl.threadpool_info()
    except Exception:                                  # pragma: no cover
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


def tag(a=None):
    """``<build>_<loaded kernel>_t<threads>``, optionally PREFIXED by
    ``LUM_PROBE_TAG`` -- which is how the same probe is run against the
    pre-round-3 tree without overwriting the post-round-3 JSON."""
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


def rings(Rbig, lo, hi, n=4, duty=0.5):
    per = float(Rbig) / n

    def f(r):
        frac = np.mod(np.asarray(r, float), per) / per
        return np.where(frac < duty, complex(hi), complex(lo)).astype(complex)
    return f


def segs4(Rbig, a=2.0, b=5.0, c=3.0, d=6.0):
    e = [complex(a), complex(b), complex(c), complex(d)]

    def f(r):
        r = np.asarray(r, float)
        j = np.clip((r / (float(Rbig) / 4.0)).astype(int), 0, 3)
        return np.asarray(e, dtype=complex)[j]
    return f


def mid_profile(family, Rbig):
    if family == "uniform":
        return uni(4.0)
    if family == "ring":
        return rings(Rbig, 2.0, 6.0)
    if family == "seg":
        return segs4(Rbig)
    raise ValueError(family)


def stack(basis, family, m, N, rbl, im_rel=0.0, where="inc", k0=2.0,
          e_half=2.0, thickness=0.35):
    """Three layers: two ``eps = e_half`` half-spaces around one structured
    middle layer.  ``im_rel`` is ``Im(eps)/Re(eps)`` applied at ``where``:
    ``inc`` = ``layers[0]``, ``exit`` = ``layers[-1]``, ``both`` = the two
    half-spaces, ``mid`` = the structured layer, ``all`` = everywhere."""
    from lumenairy.elements.bor.bor_solve import build_layer
    Rbig = float(rbl) * 2.0 * np.pi / float(k0)

    def lossy(fn, on):
        if not on or im_rel == 0.0:
            return fn
        return lambda r: (lambda e: e + 1j * im_rel * np.real(e))(
            np.asarray(fn(r), dtype=complex))

    h0 = lossy(uni(e_half), where in ("inc", "both", "all"))
    h1 = lossy(uni(e_half), where in ("exit", "both", "all"))
    mid = lossy(mid_profile(family, Rbig), where in ("mid", "all"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [build_layer(m, Rbig, N, h0, k0, basis=basis),
                build_layer(m, Rbig, N, mid, k0, basis=basis,
                            thickness=thickness),
                build_layer(m, Rbig, N, h1, k0, basis=basis)]


def disarmed(layers, k0):
    """Solve with BOTH BOR guards off, so a decision can be scored against a
    number the screen did not produce."""
    import lumenairy.elements.bor.bor_solve as bs
    prev = bs.BOR_NODAL_PASSIVITY_GUARD
    bs.BOR_NODAL_PASSIVITY_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return bs.solve(layers, k0)
    finally:
        bs.BOR_NODAL_PASSIVITY_GUARD = prev


def armed(layers, k0):
    """``(verdict, which_detector, n_energy_warnings)`` as a caller sees it."""
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


def ceiling_excess(layers, res, k0):
    """``(inc, exit)`` index-ceiling excess over the RETURNED channels --
    exactly what ``_check_nodal_passivity`` reads, evaluated here regardless of
    whether the half-space is lossless."""
    import lumenairy.elements.bor.bor_solve as bs
    out = []
    for L, idx in ((layers[0], res["inc"]), (layers[-1], res["out"])):
        qn = np.asarray(L["q"])[np.asarray(idx, int)] / k0
        out.append(bs._channel_index_excess(L, qn))
    return out[0], out[1]
