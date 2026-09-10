"""CLASS C on BOR SEM -- (iii) the TAPER STAIRCASE.

THE SURFACE.  A conical (tapered) body of revolution is meshed as a stack of
``n_slices`` layers, each a ring of its own radius.  Adjacent slices' walls
then differ by exactly ``delta = (r_top - r_bot) / n_slices``, and the
neighbour-enrichment window puts an element of THAT width into every slice's
mesh.  So a taper walks INTO the sliver family as it is refined -- more
slices means a NARROWER manufactured element, and the answer is supposed to
get BETTER.  This is precisely the surface the 2-D mortar round-4 identified
as "walking toward the contract": a taper whose narrowest sampled width is
``~ w_bottom / (2 n_slices)``, measured 8.0e-03 at 32 slices and 4.1e-03 at
64, entering the degradation band at NINE slices.

MEASURED at 8 / 16 / 32 / 64 slices (and 4 as the coarse anchor): the
narrowest element the mesh builder produced, in absolute units, as a fraction
of ``Rbig`` and as a fraction of the local wavelength; the closure; the
super-unity against the 1-D guard's bars; the spurious ``max |q|`` against the
physical ceiling; and the answer's own slice-refinement convergence, which is
the thing a user reads as "converged".

A staircase that is CONVERGING and a staircase that has hit a numerical FLOOR
look the same from one rung, so the FD arm -- which has no union grid and
therefore no manufactured element -- is run at every rung as the control.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import closure, dump, pin_tree  # noqa: E402

print("TREE", pin_tree())

RBIG = 24.0
K0 = 2.0
E_HI = 2.45 ** 2
E_LO = 1.41 ** 2
R_TOP = 8.0          # cone radius at the top
R_BOT = 2.0          # cone radius at the bottom
HEIGHT = 1.2
M = 1


def build(n_slices, *, basis, degree, N=200):
    from lumenairy import BORStack
    s = BORStack(RBIG, M, n_substrate=1.41, n_superstrate=1.41, N=N,
                 basis=basis, degree=degree)
    s.add_layer(0.4, eps=E_LO)
    t = HEIGHT / n_slices
    for j in range(n_slices):
        frac = (j + 0.5) / n_slices
        r = R_TOP + (R_BOT - R_TOP) * frac
        s.add_layer(t, segments=[(r, E_HI), (RBIG, E_LO)])
    s.add_layer(0.4, eps=E_LO)
    s.set_source(k0=K0)
    return s


def one(n_slices, *, basis, degree):
    s = build(n_slices, basis=basis, degree=degree)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = s.solve()
    d = s._last
    qmax, wmin = 0.0, np.inf
    for _t, L in ([("s", d["sup"]), ("b", d["sub"])]
                  + [("m", LL) for _tt, LL in d["mids"]]):
        qmax = max(qmax, float(np.max(np.abs(np.asarray(L["q"])))))
        mesh = L.get("mesh")
        if mesh is not None:
            wmin = min(wmin, float(np.diff(mesh.b).min()))
    R = np.asarray(res["R"], float)
    lam_hi = 2 * np.pi / (K0 * 2.45)
    return dict(n_slices=n_slices, basis=basis, degree=degree,
                delta_walls=float((R_TOP - R_BOT) / n_slices),
                n_orders=int(R.size), R=R.tolist()[:12],
                T=np.asarray(res["T"], float).tolist()[:12],
                closure=closure(res),
                superunity=float(np.max(np.asarray(res["energy"])) - 1.0)
                if R.size else None,
                qmax=qmax, qmax_over_ceiling=qmax / (2.45 * K0),
                min_element=None if not np.isfinite(wmin) else wmin,
                min_elem_frac_Rbig=(None if not np.isfinite(wmin)
                                    else float(wmin / RBIG)),
                min_elem_frac_lam=(None if not np.isfinite(wmin)
                                   else float(wmin / lam_hi)),
                warnings=[str(x.message)[:140] for x in w][:2])


def errR(a, b):
    x = np.asarray(a["R"], float)
    y = np.asarray(b["R"], float)
    n = min(x.size, y.size)
    return float(np.max(np.abs(x[:n] - y[:n]))) if n else None


def main():
    tag = os.environ.get("PROBE_TAG", "win")
    thr = os.environ.get("OPENBLAS_NUM_THREADS", "?")
    slices = (4, 8, 16, 32, 64)
    payload = dict(threads=thr, ladders=[])
    for degree in (6, 8, 12):
        sem = [one(n, basis="sem", degree=degree) for n in slices]
        fd = [one(n, basis="fd", degree=degree) for n in slices]
        for i in range(len(slices)):
            sem[i]["errR_vs_finest_sem"] = errR(sem[-1], sem[i])
            fd[i]["errR_vs_finest_fd"] = errR(fd[-1], fd[i])
            sem[i]["errR_vs_fd_same_rung"] = errR(fd[i], sem[i])
        payload["ladders"].append(dict(degree=degree, sem=sem, fd=fd))
        print(f"\n== TAPER STAIRCASE  degree={degree} ==")
        print("  slices  wall delta   min_elem     min/Rbig    min/lam    "
              "closure     R+T-1        |q|/ceil    errR vs 64   "
              "errR vs FD   warn")
        for r in sem:
            print(f"  {r['n_slices']:>6}  {r['delta_walls']:.4e}  "
                  f"{r['min_element']:.4e}  {r['min_elem_frac_Rbig']:.3e}  "
                  f"{r['min_elem_frac_lam']:.3e}  {r['closure']:.3e}  "
                  f"{(r['superunity'] if r['superunity'] is not None else float('nan')):+.3e}  "
                  f"{r['qmax_over_ceiling']:.3e}  "
                  f"{r['errR_vs_finest_sem']:.4e}   "
                  f"{r['errR_vs_fd_same_rung']:.4e}   "
                  f"{len(r['warnings'])}")
        print("  FD control:  " + "  ".join(
            f"{n}:{r['closure']:.2e}" for n, r in zip(slices, fd)))
    dump(f"c3_taper_{tag}_t{thr}.json", payload)

    print("\n== HEADLINE ==")
    for L in payload["ladders"]:
        s64 = L["sem"][-1]
        print(f"  degree {L['degree']}: at 64 slices the narrowest element is "
              f"{s64['min_elem_frac_Rbig']:.3e} of Rbig "
              f"({s64['min_elem_frac_lam']:.3e} of lambda_core); closure "
              f"{s64['closure']:.3e}; warnings {len(s64['warnings'])}")
    nw = sum(len(r["warnings"]) for L in payload["ladders"] for r in L["sem"])
    print(f"  UserWarnings over the whole taper ladder: {nw}")


if __name__ == "__main__":
    main()
