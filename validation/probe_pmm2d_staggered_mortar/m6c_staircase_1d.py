"""M6c -- the STAIRCASE use case with an EXACT INDEPENDENT ORACLE, at EQUAL
DEGREES OF FREEDOM.

The staircase is y-UNIFORM (stripes), which makes the exact 1-D pure PMM
(``PMMStack``, degree 14) the oracle for the whole 3-slice stack, so both arms
are scored against TRUTH rather than against each other.

The comparison is parameterised by the per-axis DOF ``q = N (M-1)``, not by the
modal count, because the region eig is ``2 q^2`` and depends on nothing else
(experiment doc S1.2).  At a given ``q`` EVERY region eigenproblem in BOTH arms
has exactly the same dimension, so the two arms do the same work and whatever
separates them is the h-versus-p question alone: the union grid spends its DOF
h-refining on the LCM lattice, the per-layer arm spends them p-refining on each
slice's own lattice.

(A) widths 1/2, 1/3, 1/6  -> per-layer N = 2, 3, 6;  LCM = 6
(B) widths 1/2, 1/3, 1/4  -> per-layer N = 2, 3, 4;  LCM = 12.  Note the union
    arm has a FLOOR here: M >= 3 forces q >= 2 * N_LCM = 24, while the
    per-layer arm can run the whole stack at q = 12."""
import json
import time
import warnings

import numpy as np
from mortar2d import MortarStack2D, guard

print("lumenairy:", guard(), flush=True)
from lumenairy import PMMStack
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure

warnings.simplefilter("ignore")
PX = PY = 1.2e-6
WL = 0.85e-6
EPS_H, EPS_P = 2.25, 9.0
TH, NORD = 0.18, 2
T = 0.12e-6


def stripe(N, k):
    """y-uniform stripe: the first ``k`` of ``N`` x-segments are the pillar."""
    c = np.full((N, N), EPS_H + 0j)
    c[:k, :] = EPS_P
    return c


def segs(k, N):
    return [(k / N, EPS_P), (1.0 - k / N, EPS_H)]


CASES = {
    "A": dict(per=[(2, 1), (3, 1), (6, 1)], uni=[(6, 3), (6, 2), (6, 1)],
              lcm=6, qs=[12, 18, 24]),
    "B": dict(per=[(2, 1), (3, 1), (4, 1)], uni=[(12, 6), (12, 4), (12, 3)],
              lcm=12, qs=[12, 24, 36]),
}

rows = []
for tag, cfg in CASES.items():
    o1 = PMMStack(PX, n_superstrate=1.0, n_substrate=1.0, degree=14)
    for (N, k) in cfg["per"]:
        o1.add_layer(T, segments=segs(k, N))
    o1.set_source(WL, theta=TH)
    oo, RR, TT = o1.solve()[:3]
    oo = np.asarray(oo).ravel()
    j = np.argsort(oo)
    MO, RO, TO = oo[j], RR[1][j], TT[1][j]

    def score(orders, R, T):
        a = np.asarray(orders)
        sel = a[:, 1] == 0
        m = a[sel, 0]
        i = np.argsort(m)
        m, r, t = m[i], R[1][sel][i], T[1][sel][i]
        keep = np.isin(MO, m)
        return max(float(np.abs(r - RO[keep]).max()),
                   float(np.abs(t - TO[keep]).max()))

    print(f"\n=== case {tag}: per-layer N = {[N for N, _k in cfg['per']]}, "
          f"LCM = {cfg['lcm']} ===", flush=True)
    for q in cfg["qs"]:
        dim = 2 * q * q
        # per-layer arm: M_i chosen so every slice carries the same q
        Ms = []
        ok = True
        for (N, _k) in cfg["per"]:
            if q % N:
                ok = False
                break
            Ms.append(q // N + 1)
        if ok and min(Ms) >= 3:
            t0 = time.perf_counter()
            s = MortarStack2D(PX, PY, n_modes=max(Ms), n_orders=NORD)
            for (N, k), M in zip(cfg["per"], Ms):
                s.add_layer(T, eps_cell=stripe(N, k), n_modes=M)
            s.set_source(WL, theta=TH, phi=0.0)
            o, R, Tt = s.solve(jones=False)
            dt = time.perf_counter() - t0
            e = score(o, R, Tt)
            c = max(abs(R[p].sum() + Tt[p].sum() - 1.0) for p in (0, 1))
            rows.append(dict(case=tag, arm="perlayer", q=q, dim=dim, Ms=Ms,
                             err=e, closure=float(c), t=dt))
            print(f"  q={q:3d} (dim {dim}) PER-LAYER  M_i={Ms}  err {e:9.2e}  "
                  f"closure {c:8.1e}  {dt:7.1f}s", flush=True)
        else:
            print(f"  q={q:3d} PER-LAYER: not attainable at M >= 3 for "
                  f"N = {[N for N, _k in cfg['per']]}", flush=True)
        # union arm: one M on the LCM lattice
        if q % cfg["lcm"] == 0 and q // cfg["lcm"] + 1 >= 3:
            MU = q // cfg["lcm"] + 1
            t0 = time.perf_counter()
            s = PMM2DStackPure(PX, PY, n_modes=MU, n_orders=NORD)
            for (N, k) in cfg["uni"]:
                s.add_layer(T, eps_cell=stripe(N, k))
            s.set_source(WL, theta=TH, phi=0.0)
            o, R, Tt = s.solve(jones=False)
            dt = time.perf_counter() - t0
            e = score(o, R, Tt)
            c = max(abs(R[p].sum() + Tt[p].sum() - 1.0) for p in (0, 1))
            rows.append(dict(case=tag, arm="union", q=q, dim=dim, M=MU, err=e,
                             closure=float(c), t=dt))
            print(f"  q={q:3d} (dim {dim}) UNION      M={MU}       err {e:9.2e}"
                  f"  closure {c:8.1e}  {dt:7.1f}s", flush=True)
        else:
            print(f"  q={q:3d} UNION: UNREACHABLE -- M >= 3 forces "
                  f"q >= {2*cfg['lcm']} on the N={cfg['lcm']} lattice",
                  flush=True)
        json.dump(rows, open(
            "validation/probe_pmm2d_staggered_mortar/m6c_staircase_1d.json",
            "w"), indent=1)
