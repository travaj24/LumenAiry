"""V11 -- D9.  THE SHIPPED near-cutoff fixture, varying ONLY the radial cutoff
index that ``tests/unit/test_fix_bor_multilayer_guards._gamma_of(m, idx=2)``
fixes by an undocumented default.

`v4_cutoff.py` widens the geometry; this one does not touch it.  It rebuilds
the shipped `_cutoff_stack` (`Rbig` 24, `N` 120, `n` 1.41, the same
floor-derived ladder) and sweeps ``m`` 0..3 x ``idx`` 0..4, so the only thing
that changes between a row and the gate's own reading is WHICH cutoff the
ladder approaches.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

import lumenairy
from lumenairy.elements.bor import _orient as _or
from lumenairy.elements.bor.bor_solve import build_layer
from lumenairy.elements.bor.bor_stack import BORStack

RBIG, NFD, NREF = 24.0, 120, 1.41
EPS = NREF ** 2
BAR = 1.0e-5                      # tests/unit/..._CUTOFF_LADDER_BAR


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


def _gamma(m, idx):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        L = build_layer(m, RBIG, NFD,
                        lambda r: np.full(np.shape(r), complex(EPS)), 2.0)
    q = np.asarray(L["q"])
    g = np.sqrt(2.0 ** 2 * EPS - q ** 2)
    g = np.real(g[np.abs(g.imag) < 1e-9 * np.maximum(np.abs(g.real), 1e-300)])
    g = np.sort(g[g > 1e-6])
    return float(g[idx]) if idx < g.size else None


def _stack(m, k0):
    s = BORStack(RBIG, m, n_substrate=NREF, n_superstrate=NREF, N=NFD,
                 basis="fd")
    s.add_layer(0.4, eps=EPS)
    s.add_layer(0.5, rings=(3.0, 0.5, 2.45, 1.41))
    s.add_layer(0.4, eps=EPS)
    s.set_source(k0=float(k0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return s.solve()


def _rungs():
    floor = _or._BOR_CHANNEL_REAL_FLOOR * 10.0
    out = []
    for e_ in range(8, 40):
        dl = 10.0 ** (-e_ / 2.0)
        if NREF * np.sqrt(dl) < floor:
            break
        out.append(dl)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="local")
    a = ap.parse_args()
    rr = _rungs()
    rows = []
    print("%-4s %-4s %-12s %-10s %-14s" % ("m", "idx", "gamma", "counts",
                                           "worst |R+T-1|"))
    for m in (0, 1, 2, 3):
        for idx in (0, 1, 2, 3, 4):
            g = _gamma(m, idx)
            if g is None:
                continue
            counts, worst = set(), 0.0
            for dl in rr:
                res = _stack(m, g / (NREF * np.sqrt(1.0 - dl)))
                counts.add(int(np.size(res["R"])))
                en = np.asarray(res["energy"])
                if en.size:
                    worst = max(worst, float(np.max(np.abs(en - 1.0))))
            rows.append(dict(m=m, idx=idx, gamma=g, counts=sorted(counts),
                             count_stable=len(counts) == 1,
                             worst_closure=worst, over_bar=worst > BAR))
            print("%-4d %-4d %-12.6g %-10s %-14.6e%s"
                  % (m, idx, g, sorted(counts), worst,
                     "  <<< ABOVE the 1e-5 bar" if worst > BAR else ""),
                  flush=True)
    swept = [r for r in rows if r["idx"] == 2]
    env_all = max(r["worst_closure"] for r in rows)
    env_swept = max(r["worst_closure"] for r in swept)
    worst_row = max(rows, key=lambda r: r["worst_closure"])
    print("\nENVELOPE, the gate's own index (idx=2), m 0..3 : %.6e"
          % (env_swept,))
    print("ENVELOPE, m 0..3 x idx 0..4                    : %.6e "
          "at m=%d idx=%d (%.2fx the %.0e bar)"
          % (env_all, worst_row["m"], worst_row["idx"], env_all / BAR, BAR))
    print("rows over the bar: %d of %d; count-unstable: %d"
          % (sum(1 for r in rows if r["over_bar"]), len(rows),
             sum(1 for r in rows if not r["count_stable"])))
    out = dict(provenance=_prov(a.tag), n_rungs=len(rr), bar=BAR,
               envelope_swept_index=env_swept, envelope_all=env_all,
               rows=rows)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "v11_cutoff_index_%s.json" % (a.tag,))
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("wrote", p)


if __name__ == "__main__":
    main()
