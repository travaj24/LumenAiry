"""V4 -- D9.  THE NEAR-CUTOFF CHANNEL-COUNT AND CLOSURE ENVELOPE, re-derived.

The round-2 gate sweeps ``m`` in {0, 1, 2} on ONE stack and bars the worst
lossless closure at ``_CUTOFF_LADDER_BAR = 1e-5``, justified by a measured
ten-arm envelope of 1.2716e-06 (7.86x).  This probe re-derives the envelope on:

  * ``m`` in {0, 1, 2, **3**} -- the gate's family plus one order it does not
    sweep, because "the whole near-cutoff ladder" is a family claim;
  * THREE stack geometries, not one -- the shipped coincident-superstrate
    stack, a NON-coincident variant, and a finer radial grid -- because a bar
    measured on one fixture is a SAMPLE property (``docs/TESTING_STANDARDS.md``
    rule 5);
  * the radial order index ``idx`` in {1, 2, 3}, which selects WHICH cutoff the
    ladder approaches.

The ladder's stopping rung is taken from the library's own channel floor, the
way the gate now does, so nothing here pins a literal.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

import lumenairy  # noqa: E402
from lumenairy.elements.bor import _orient as _or  # noqa: E402
from lumenairy.elements.bor.bor_solve import build_layer  # noqa: E402
from lumenairy.elements.bor.bor_stack import BORStack  # noqa: E402

EPS = 1.41 ** 2
NREF = 1.41
FLOOR_MULT = 10.0


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


GEOM = {
    # name: (Rbig, N, layers, coincident superstrate?)
    "G1_coincident": (3.0, 240, True),
    "G2_noncoincident": (3.0, 240, False),
    "G3_fine_grid": (3.5, 320, True),
}


def _fd_modes(m, k0, Rbig, N):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_layer(m, Rbig, N, lambda r: np.full(np.shape(r),
                                                         complex(EPS)), k0)


def gamma_of(m, Rbig, N, idx):
    """The ``idx``-th radial cutoff of the PEC-walled cylinder at order ``m``."""
    L = _fd_modes(m, 2.0, Rbig, N)
    q = np.asarray(L["q"])
    g = np.sqrt(2.0 ** 2 * EPS - q ** 2)
    g = np.real(g[np.abs(g.imag) < 1e-9 * np.maximum(np.abs(g.real), 1e-300)])
    g = np.sort(g[g > 1e-6])
    if idx >= g.size:
        return None
    return float(g[idx])


def stack_of(geom, m, k0):
    Rbig, N, coincident = GEOM[geom]
    s = BORStack(Rbig, m, n_substrate=NREF, n_superstrate=NREF, N=N,
                 basis="fd")
    if coincident:
        s.add_layer(0.4, eps=EPS)
    else:
        s.add_layer(0.4, eps=EPS * 1.21)
    s.add_layer(0.5, rings=(3.0 if Rbig <= 3.0 else 3.4, 0.5, 2.45, 1.41))
    s.add_layer(0.4, eps=EPS)
    s.set_source(k0=float(k0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return s.solve()


def rungs():
    floor = _or._BOR_CHANNEL_REAL_FLOOR * FLOOR_MULT
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
    rr = rungs()
    rows = []
    for geom in GEOM:
        for m in (0, 1, 2, 3):
            for idx in (1, 2, 3):
                g = gamma_of(m, GEOM[geom][0], GEOM[geom][1], idx)
                if g is None:
                    continue
                counts, worst, worst_dl = set(), 0.0, None
                for dl in rr:
                    k0 = g / (NREF * np.sqrt(1.0 - dl))
                    res = stack_of(geom, m, k0)
                    counts.add(int(np.size(res["R"])))
                    en = np.asarray(res["energy"])
                    if en.size:
                        w = float(np.max(np.abs(en - 1.0)))
                        if w > worst:
                            worst, worst_dl = w, dl
                rows.append(dict(geom=geom, m=m, idx=idx, gamma=g,
                                 n_rungs=len(rr), counts=sorted(counts),
                                 count_stable=len(counts) == 1,
                                 worst_closure=worst, worst_delta=worst_dl))
                print("%-18s m=%d idx=%d  counts=%-10s worst=%.6e (delta=%s)"
                      % (geom, m, idx, sorted(counts), worst, worst_dl),
                      flush=True)
    env = max(r["worst_closure"] for r in rows)
    env_gate = max(r["worst_closure"] for r in rows
                   if r["geom"] == "G1_coincident" and r["m"] in (0, 1, 2)
                   and r["idx"] == 2)
    unstable = [r for r in rows if not r["count_stable"]]
    print("\nENVELOPE over everything          : %.6e" % (env,))
    print("ENVELOPE over the GATE's own scope: %.6e" % (env_gate,))
    print("count-unstable rows               : %d of %d"
          % (len(unstable), len(rows)))
    out = dict(provenance=_prov(a.tag), n_rungs=len(rr),
               floor=_or._BOR_CHANNEL_REAL_FLOOR, floor_mult=FLOOR_MULT,
               envelope_all=env, envelope_gate_scope=env_gate, rows=rows)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "v4_cutoff_%s.json" % (a.tag,))
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("wrote", p)


if __name__ == "__main__":
    main()
