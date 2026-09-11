"""V1 -- the nodal/staggered census, guard DISARMED, on THIS verification's own
battery.  Re-derives every population the round-2 report's section 4 quotes:
the energy excess/deficit, the index-ceiling excess, the channel count against
the div-conforming twin, and the two predicates' decisions.

Usage:  python v1_census.py [--tag TAG] [--quick]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vfix as F  # noqa: E402

import lumenairy  # noqa: E402
from lumenairy.elements.bor import bor_solve as BS  # noqa: E402


def _prov(name):
    import scipy
    import threadpoolctl
    arch = sorted({str(d.get("architecture"))
                   for d in threadpoolctl.threadpool_info()})
    thr = sorted({int(d.get("num_threads", -1))
                  for d in threadpoolctl.threadpool_info()})
    return dict(tag=name, lumenairy_file=lumenairy.__file__,
                lumenairy_version=lumenairy.__version__,
                python=sys.version.split()[0], numpy=np.__version__,
                scipy=scipy.__version__, blas_arch=arch, blas_threads=thr,
                env={k: os.environ.get(k) for k in
                     ("OPENBLAS_CORETYPE", "OMP_NUM_THREADS",
                      "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})


def geometries(quick=False):
    """The battery.  Each entry is (name, kind, m, N, rbl, k0, build(basis)).

    ``build`` returns the layer list for one basis: [superstrate, slab,
    substrate], the slab carrying the structure.  Half-spaces are uniform
    eps = 2 unless the name says otherwise.
    """
    out = []
    rbls = [0.5, 1.0, 2.0] if quick else [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    Ns = [200] if quick else [120, 200]
    ms = [0, 1, 2]
    k0 = 2.0
    for kind in ("uniform", "ring", "segment"):
        for m in ms:
            for N in Ns:
                for rbl in rbls:
                    Rbig = F.rbig_of(rbl, k0)

                    def build(basis, kind=kind, m=m, N=N, Rbig=Rbig, k0=k0):
                        if kind == "uniform":
                            mid = F.uniform(6.0)
                        elif kind == "ring":
                            mid = F.ring(2.0, 6.0, Rbig, n_rings=4)
                        else:
                            mid = F.segments([2.0, 6.0, 3.0, 9.0], Rbig)
                        return [
                            BS.build_layer(m, Rbig, N, F.uniform(2.0), k0,
                                           basis=basis),
                            BS.build_layer(m, Rbig, N, mid, k0, basis=basis,
                                           thickness=0.35),
                            BS.build_layer(m, Rbig, N, F.uniform(2.0), k0,
                                           basis=basis),
                        ]
                    out.append(dict(
                        name="%s_m%d_N%d_rbl%g" % (kind, m, N, rbl),
                        kind=kind, m=m, N=N, rbl=rbl, k0=k0, build=build))
    return out


def measure(layers, k0):
    """Raw facts with the guard disarmed."""
    old = BS.BOR_NODAL_PASSIVITY_GUARD
    BS.BOR_NODAL_PASSIVITY_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = BS.solve(layers, k0)
    finally:
        BS.BOR_NODAL_PASSIVITY_GUARD = old
    e = np.asarray(res["energy"], dtype=float)
    qn_in = layers[0]["q"][res["inc"]] / k0
    qn_out = layers[-1]["q"][res["out"]] / k0
    # the ceiling excess is computed HERE from the public facts (the layer's
    # own eps ceiling and the returned channels) so the same probe runs on the
    # PRE tree, which has no ``_channel_index_excess``.
    ce = []
    for L, qn in ((layers[0], qn_in), (layers[-1], qn_out)):
        eps = L.get("eps_ceiling")
        if eps is None or np.asarray(qn).size == 0:
            ce.append(-np.inf)
            continue
        n_max = float(np.real(np.sqrt(complex(eps))))
        ce.append(float(np.max(np.real(qn))) - n_max)
    return dict(
        n_inc=int(len(res["inc"])), n_out=int(len(res["out"])),
        emax=float(np.max(e)) if e.size else float("nan"),
        emin=float(np.min(e)) if e.size else float("nan"),
        excess=float(np.max(e)) - 1.0 if e.size else float("nan"),
        deficit=1.0 - float(np.min(e)) if e.size else float("nan"),
        ceiling_excess=float(max(ce)),
        max_qn=float(np.max(np.real(qn_in))) if qn_in.size else float("nan"),
    )


def decide(layers, k0):
    """The guard's ARMED decision: what the caller actually receives."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            BS.solve(layers, k0)
            verdict = "returned"
            msg = ""
        except BS.BORNodalPassivityError as exc:
            verdict = "REFUSED"
            msg = str(exc)
        wmsgs = [str(x.message) for x in w
                 if issubclass(x.category, UserWarning)]
    detector = ""
    if verdict == "REFUSED":
        detector = ("ceiling" if "gamma^2 < 0" in msg else "energy")
    return dict(verdict=verdict, detector=detector,
                message=msg[:400],
                warned=any("R + T" in s or "max(R + T)" in s for s in wmsgs),
                n_warn=len(wmsgs),
                warn_msgs=[s[:220] for s in wmsgs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="local")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    rows = []
    for g in geometries(a.quick):
        ref = None
        for basis in ("staggered", "nodal"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                layers = g["build"](basis)
            row = dict(name=g["name"], kind=g["kind"], m=g["m"], N=g["N"],
                       rbl=g["rbl"], k0=g["k0"], basis=basis)
            row.update(measure(layers, g["k0"]))
            row.update(decide(layers, g["k0"]))
            row["provably_passive"] = bool(
                BS._stack_is_provably_passive(layers))
            _pl = getattr(BS, "_stack_is_provably_lossless", None)
            row["provably_lossless"] = (bool(_pl(layers)) if _pl is not None
                                        else None)
            row["min_rel_im"] = [L.get("min_rel_im_eps") for L in layers]
            row["max_rel_im"] = [L.get("max_rel_im_eps") for L in layers]
            if basis == "staggered":
                ref = (row["n_inc"], row["n_out"])
                row["set_wrong"] = False
            else:
                row["set_wrong"] = bool((row["n_inc"], row["n_out"]) != ref)
                row["ref_counts"] = list(ref)
            rows.append(row)
            print("%-26s %-10s n=%2d/%-2d  emax=%-14.6g emin=%-14.6g "
                  "ceil=%-12.5g %s %s"
                  % (row["name"], basis, row["n_inc"], row["n_out"],
                     row["emax"], row["emin"], row["ceiling_excess"],
                     row["verdict"], row.get("detector", "")), flush=True)
    out = dict(provenance=_prov(a.tag),
               constants={k: getattr(BS, v, None) for k, v in (
                   ("BAR", "_BOR_NODAL_SUPERUNITY_BAR"),
                   ("WARN", "_BOR_NODAL_SUPERUNITY_WARN"),
                   ("LOSSLESS_REL_IM", "_BOR_LOSSLESS_REL_IM"),
                   ("PASSIVE_DEADBAND", "_BOR_PASSIVE_DEADBAND"),
                   ("CEILING_SLACK", "_BOR_INDEX_CEILING_SLACK"))},
               rows=rows)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "v1_census_%s.json" % (a.tag,))
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\nwrote", p, "rows", len(rows))


if __name__ == "__main__":
    main()
