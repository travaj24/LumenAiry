"""V3 -- D1/D2 ladders on THIS verification's own stacks.

Six ladders, each run with the guard ARMED (the decision the caller receives)
and with it DISARMED (the raw numbers, so the decision can be scored against
something the screen did not produce):

  A  LOSS on a DAMAGING ring stack        -- the D1 claim (4/13 -> 13/13)
  B  LOSS on a HEALTHY uniform stack      -- the false-positive side (0/39)
  C  GAIN                                 -- outside the screen (1/5 -> 0/5)
  D  absorbing INCIDENCE half-space       -- the screen must disarm
  E  absorbing EXIT half-space            -- armed one-sided; and the ceiling
                                             conjunct is gated off on that side
  F  LOSSLESS DEFICIT                     -- the two-sided claim

Runs unchanged on the PRE tree (f2d331c5), which has neither
``_stack_is_provably_lossless`` nor the ceiling conjunct.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vfix as F  # noqa: E402

import lumenairy  # noqa: E402
from lumenairy.elements.bor import bor_solve as BS  # noqa: E402

LOSS = [0.0, 1e-14, 1e-13, 1e-12, 3e-12, 1e-11, 1e-9, 1e-8,
        1e-6, 1e-4, 1e-3, 1e-2, 1e-1]
GAIN = [-1e-14, -1e-12, -1e-9, -1e-6, -1e-3]


def _prov(name):
    import scipy
    import threadpoolctl
    return dict(tag=name, lumenairy_file=lumenairy.__file__,
                python=sys.version.split()[0], numpy=np.__version__,
                scipy=scipy.__version__,
                blas_arch=sorted({str(d.get("architecture"))
                                  for d in threadpoolctl.threadpool_info()}),
                blas_threads=sorted({int(d.get("num_threads", -1))
                                     for d in threadpoolctl.threadpool_info()}),
                env={k: os.environ.get(k) for k in
                     ("OPENBLAS_CORETYPE", "OMP_NUM_THREADS",
                      "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})


def raw(layers, k0):
    old = BS.BOR_NODAL_PASSIVITY_GUARD
    BS.BOR_NODAL_PASSIVITY_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = BS.solve(layers, k0)
    finally:
        BS.BOR_NODAL_PASSIVITY_GUARD = old
    e = np.asarray(r["energy"], dtype=float)
    qn_in = layers[0]["q"][r["inc"]] / k0
    qn_out = layers[-1]["q"][r["out"]] / k0
    ce = []
    for L, qn in ((layers[0], qn_in), (layers[-1], qn_out)):
        eps = L.get("eps_ceiling")
        if eps is None or np.asarray(qn).size == 0:
            ce.append(-np.inf)
        else:
            ce.append(float(np.max(np.real(qn)))
                      - float(np.real(np.sqrt(complex(eps)))))
    blob = np.ascontiguousarray(
        np.concatenate([np.asarray(r["R"], float),
                        np.asarray(r["T"], float)])).tobytes()
    nan = float("nan")
    return dict(n_inc=int(len(r["inc"])), n_out=int(len(r["out"])),
                emax=float(np.max(e)) if e.size else nan,
                emin=float(np.min(e)) if e.size else nan,
                excess=(float(np.max(e)) - 1.0) if e.size else nan,
                deficit=(1.0 - float(np.min(e))) if e.size else nan,
                ceiling_excess=float(max(ce)),
                # a hash of the exact IEEE-754 bytes of the answer, so
                # "the switch restores the PRE behaviour exactly" is checkable
                sha=hashlib.sha256(blob).hexdigest()[:24])


def armed(layers, k0):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            BS.solve(layers, k0)
            v, msg = "returned", ""
        except BS.BORNodalPassivityError as exc:
            v, msg = "REFUSED", str(exc)
        ws = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
    det = ""
    if v == "REFUSED":
        det = "ceiling" if "gamma^2 < 0" in msg else "energy"
    pw = [s for s in ws if "R + T" in s]
    return dict(verdict=v, detector=det, message=msg[:600],
                n_passivity_warnings=len(pw),
                warn=pw[0][:300] if pw else "")


def _profiles(kind, im_rel, Rbig, im_where):
    sup = F.uniform(2.0, im_rel if im_where == "inc" else 0.0)
    sub = F.uniform(2.0, im_rel if im_where == "exit" else 0.0)
    mid_im = im_rel if im_where == "mid" else 0.0
    if kind == "ring":
        mid = F.ring(2.0, 6.0, Rbig, n_rings=4, im_rel=mid_im, im_on="hi")
    elif kind == "uniform":
        mid = F.uniform(6.0, mid_im)
    else:
        mid = F.segments([2.0, 6.0, 3.0, 9.0], Rbig, im_rel=mid_im, im_on=1)
    return sup, mid, sub


def stack(kind, im_rel, m, N, rbl, k0, im_where, basis):
    Rbig = F.rbig_of(rbl, k0)
    sup, mid, sub = _profiles(kind, im_rel, Rbig, im_where)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [BS.build_layer(m, Rbig, N, sup, k0, basis=basis),
                BS.build_layer(m, Rbig, N, mid, k0, basis=basis,
                               thickness=0.35),
                BS.build_layer(m, Rbig, N, sub, k0, basis=basis)]


def rung(kind, im_rel, m, N, rbl, k0, im_where, with_twin=True):
    L = stack(kind, im_rel, m, N, rbl, k0, im_where, "nodal")
    row = dict(kind=kind, im_rel=im_rel, m=m, N=N, rbl=rbl, im_where=im_where)
    row.update(raw(L, k0))
    row.update(armed(L, k0))
    row["provably_passive"] = bool(BS._stack_is_provably_passive(L))
    _pl = getattr(BS, "_stack_is_provably_lossless", None)
    row["provably_lossless"] = bool(_pl(L)) if _pl else None
    row["min_rel_im"] = [Li.get("min_rel_im_eps") for Li in L]
    row["max_rel_im"] = [Li.get("max_rel_im_eps") for Li in L]
    if with_twin:
        T = stack(kind, im_rel, m, N, rbl, k0, im_where, "staggered")
        t = raw(T, k0)
        row["twin_n_inc"] = t["n_inc"]
        row["twin_emin"] = t["emin"]
        row["twin_emax"] = t["emax"]
        row["set_wrong"] = bool(row["n_inc"] != t["n_inc"]
                                or row["n_out"] != t["n_out"])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="local")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    out = dict(provenance=_prov(a.tag), ladders={})

    out["ladders"]["A_loss_damaging_ring"] = [
        rung("ring", x, 1, 200, 2.0, 2.0, "mid") for x in LOSS]
    out["ladders"]["C_gain"] = [
        rung("ring", x, 1, 200, 2.0, 2.0, "mid") for x in GAIN]
    out["ladders"]["D_absorbing_incidence"] = [
        rung("ring", x, 1, 200, 2.0, 2.0, "inc") for x in LOSS]
    out["ladders"]["E_absorbing_exit"] = [
        rung("ring", x, 1, 200, 2.0, 2.0, "exit") for x in LOSS]
    if not a.fast:
        out["ladders"]["B_loss_healthy_uniform"] = [
            rung("uniform", x, m, 200, 1.0, 2.0, "mid")
            for m in (0, 1, 2) for x in LOSS]
        out["ladders"]["F_lossless_deficit"] = [
            rung("segment", 0.0, m, N, rbl, 2.0, "mid")
            for m in (0, 1, 2) for N in (120, 200)
            for rbl in (0.5, 1.0, 2.0)]

    for name, rows in out["ladders"].items():
        nref = sum(1 for r in rows if r["verdict"] == "REFUSED")
        print("%-26s refused %2d / %2d" % (name, nref, len(rows)))
        for r in rows:
            print("   im=%-9.3g pass=%-5s lossless=%-5s  emax=%-12.6g "
                  "emin=%-12.6g ceil=%-11.4g  %s %s"
                  % (r["im_rel"], r["provably_passive"],
                     r["provably_lossless"], r["emax"], r["emin"],
                     r["ceiling_excess"], r["verdict"], r["detector"]),
                  flush=True)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "v3_ladders_%s.json" % (a.tag,))
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("wrote", p)


if __name__ == "__main__":
    main()
