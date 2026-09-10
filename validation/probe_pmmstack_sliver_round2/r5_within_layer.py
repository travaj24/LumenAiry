"""ROUND 2, probe 5 -- the WITHIN-LAYER liner (verification defect V-6).

A thin feature owned by ONE layer is the geometry the caller ASKED for, so it
is never a manufactured cell and must never be refused.  The verification
measured that the SAME 1/w^2 mechanism is already catastrophic there:

    liner 1e-6 of a period -> err 1.06e-03, nothing fires
    liner 1e-7             -> err 1.05, R+T from 0.571 (sub-unity, silent)
                              to 4.19 (warn-and-return)

Round 2 does not refuse it -- it WARNS with the mechanism when the trigger
super-unity is met and the element's spurious-wavenumber predictor is far past
the stack's physical index ceiling.  This probe measures both populations of
that predictor so the bar is read off the data:

    q_excess = [0.65 N(N+1)/4 / (k0 J)] / n_max ,   J = w P / 2

on the liner ladder (the dangerous population) and on the widths a healthy
stack's narrowest OWNED cell actually has (the safe population).

    python validation/probe_pmmstack_sliver_round2/r5_within_layer.py [out.json]
"""
import json
import os
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import lumenairy
from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm import stack as ps
from lumenairy.elements.pmm._core import _pmm_union_grid

HERE = os.path.dirname(os.path.abspath(__file__))
P, WL, TH = 1.2e-6, 0.85e-6, 0.15
DZ = 0.32e-6 / 4
EH, EP = 2.25, 9.0
DEGREES = (8, 12, 14, 16)


def q_excess(w_frac, degree, n_max, period=P, wl=WL):
    """The refusal's own predictor, in index units, over the physical ceiling."""
    q = 0.65 * (degree * (degree + 1) / 4.0) * wl / (np.pi * w_frac * period)
    return q / n_max


def run(segs, deg, guard=True):
    st = PMMStack(P, n_superstrate=1.0, n_substrate=1.0, degree=deg,
                  min_feature=P * 1e-12)
    st.add_layer(DZ, segments=segs)
    st.add_layer(DZ, segments=segs)
    st.set_source(WL, theta=TH)
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = bool(guard)
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            o, R, T, _J = st.solve()
        o = np.asarray(o).ravel()
        i = np.argsort(o)
        tot = np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)
        return dict(ok=True, m=o[i], R=np.asarray(R)[1][i],
                    T=np.asarray(T)[1][i], worst=float(np.max(tot)),
                    least=float(np.min(tot)),
                    warned=[str(w.message)[:70] for w in rec])
    except (ValueError, np.linalg.LinAlgError) as exc:
        return dict(ok=False, msg=str(exc)[:160])
    finally:
        ps.PMM_SLIVER_GUARD = was


def gap(a, b):
    c = np.intersect1d(a["m"], b["m"])
    ia, ib = np.searchsorted(a["m"], c), np.searchsorted(b["m"], c)
    return float(max(np.abs(a["R"][ia] - b["R"][ib]).max(),
                     np.abs(a["T"][ia] - b["T"][ib]).max()))


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "r5_within_layer.json"))
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib, lumenairy.__version__)
    assert "lum_sliver2" in lib.replace("\\", "/"), lib
    t0 = time.time()
    n_max = float(np.sqrt(EP))                      # the stack's index ceiling
    refs = {d: run([(0.30, EH), (0.70, EH)], d, guard=False) for d in DEGREES}
    rows = []
    for d in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        segs = [(0.30, EH), (d, EP), (0.70 - d, EH)]
        uw, _le = _pmm_union_grid([segs, segs], 1e-12)
        w_nar = float(np.min(uw))
        screened = ps._cross_layer_sliver([segs, segs], 1e-12)
        for deg in DEGREES:
            r = run(segs, deg, guard=True)
            row = dict(liner=d, degree=deg, w_narrow=w_nar,
                       q_excess=q_excess(w_nar, deg, n_max),
                       manufactured=screened is not None)
            if r["ok"]:
                row.update(ok=True, worst=r["worst"], least=r["least"],
                           err=gap(r, refs[deg]),
                           err_over_d=gap(r, refs[deg]) / d,
                           n_warn=len(r["warned"]),
                           warned=r["warned"][:1])
            else:
                row.update(ok=False, msg=r["msg"])
            rows.append(row)
        good = [x for x in rows[-len(DEGREES):] if x.get("ok")]
        print(f"  liner {d:.0e} (w_narrow {w_nar:.3e}, q_excess "
              f"{q_excess(w_nar, 14, n_max):.3e} @deg14, manufactured="
              f"{screened is not None}): "
              + " ".join(f"deg{x['degree']}:err {x['err']:.2e} "
                         f"R+T {x['worst']:.4g}/{x['least']:.4g}"
                         for x in good))
        for x in rows[-len(DEGREES):]:
            if not x.get("ok"):
                print(f"    deg {x['degree']} RAISED: {x['msg'][:90]}")

    # the SAFE population: the narrowest OWNED cell of ordinary geometries
    safe = []
    for walls in ([(0.30, EH), (0.70, EH)],
                  [(0.10, EH), (0.10, EP), (0.80, EH)],
                  [(0.05, EH), (0.02, EP), (0.93, EH)],
                  [(0.4, EH), (0.005, EP), (0.595, EH)],
                  [(0.4, EH), (0.001, EP), (0.599, EH)]):
        uw, _le = _pmm_union_grid([walls, walls], 1e-12)
        w = float(np.min(uw))
        for deg in DEGREES:
            safe.append(dict(w=w, degree=deg, q_excess=q_excess(w, deg, n_max)))
    print(f"\n  SAFE owned cells (0.001..0.30 of a period, degrees "
          f"{DEGREES}): q_excess "
          f"{min(s['q_excess'] for s in safe):.3e} .. "
          f"{max(s['q_excess'] for s in safe):.3e}")
    bad = [r for r in rows if r["liner"] <= 1e-6]
    print(f"  DANGEROUS liners (1e-6, 1e-7): q_excess "
          f"{min(r['q_excess'] for r in bad):.3e} .. "
          f"{max(r['q_excess'] for r in bad):.3e}")
    okc = [r for r in rows if r["liner"] >= 1e-5]
    print(f"  BENIGN liners (1e-2..1e-5):   q_excess "
          f"{min(r['q_excess'] for r in okc):.3e} .. "
          f"{max(r['q_excess'] for r in okc):.3e}, max err/d "
          f"{max(r['err_over_d'] for r in okc if r.get('ok')):.3g}")
    with open(out_path, "w") as f:
        json.dump(dict(meta=dict(lumenairy=lib, python=sys.version.split()[0],
                                 numpy=np.__version__, n_max=n_max),
                       rows=rows, safe=safe, wall_s=time.time() - t0), f,
                  indent=1, default=str)
    print("wrote", out_path, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
