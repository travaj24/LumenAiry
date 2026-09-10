"""V5 -- the EQUAL-DOF claims, re-measured with MY OWN reference ladders.

The build's G6 quotes mortar/union error ratios 0.1769 at ``q = 12`` and
0.0587 at ``q = 18`` on a stripe pair, 0.634 / 0.504 on a 3-slice staircase,
and the experiment's F2 quotes 0.529 / 0.183 on a corner-dominated 2-D pillar
pair with the advantage GONE by ``q ~ 24``.  This file re-runs all three
against references built here, and it also runs the LOSING rungs, because the
claim "not worse" is only honest if the regime where it stops being true is
measured rather than avoided.

``python v5_equal_dof.py [stripe staircase pillar]``

Fixtures are the build's own (this is a verification of ITS numbers, so the
fixture has to be the same); the REFERENCES are rebuilt here.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
import warnings

import numpy as np

import lumenairy
from lumenairy.elements.pmm.stack import PMMStack
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT)
print(f"[arm] lumenairy = {lumenairy.__file__}", flush=True)

_C = complex
RES = {}
EPS_P, EPS_H = 6.0, 2.25


def solve(st, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return st.solve(**kw)


def stripe_cell(N, num):
    c = np.full((N, N), EPS_H + 0j)
    c[:num, :] = EPS_P
    return c


def score_1d(o2d, R, T, o1d, R1d, T1d, mirror=False, orders=(-1, 0, 1)):
    best = 0.0
    for m in orders:
        sel = np.where((o2d[:, 0] == m) & (o2d[:, 1] == 0))[0]
        if not sel.size:
            continue
        k = int(sel[0])
        jj = np.where(o1d == (-m if mirror else m))[0]
        if not jj.size:
            continue
        j = int(jj[0])
        best = max(best, abs(float(R[1, k]) - float(R1d[1, j])),
                   abs(float(T[1, k]) - float(T1d[1, j])))
    return best


# ==========================================================================
_PER, _WL, _TH = 0.9, 0.6, 0.20
_TS = (0.30, 0.22)


def _oracle_stripe(deg):
    st = PMMStack(_PER, degree=deg, far_field_orders=5)
    for duty, t in zip((0.5, 1 / 3), _TS):
        st.add_layer(t, segments=[(duty, EPS_P), (1 - duty, EPS_H)])
    st.set_source(_WL, theta=_TH)
    o, R, T = st.solve()[:3]
    return np.asarray(o).ravel(), np.atleast_2d(R), np.atleast_2d(T)


def sec_stripe():
    out = {}
    ref = {d: _oracle_stripe(d) for d in (12, 14, 16)}
    o1d, R1d, T1d = ref[16]

    def gap(a, b):
        return float(max(np.max(np.abs(a[1] - b[1])),
                         np.max(np.abs(a[2] - b[2]))))
    out["oracle_selfgap_12_14"] = gap(ref[12], ref[14])
    out["oracle_selfgap_14_16"] = gap(ref[14], ref[16])
    print(f"[stripe] exact 1-D oracle self-gap 12-14 "
          f"{out['oracle_selfgap_12_14']:.3e}, 14-16 "
          f"{out['oracle_selfgap_14_16']:.3e}  (reference = degree 16)",
          flush=True)
    rows = {}
    for Mu in (3, 4, 5, 6):
        q = 6 * (Mu - 1)
        t0 = time.time()
        su = PMM2DStackPure(_PER, n_modes=Mu, n_orders=1)
        su.add_layer(_TS[0], eps_cell=stripe_cell(6, 3))
        su.add_layer(_TS[1], eps_cell=stripe_cell(6, 2))
        su.set_source(_WL, theta=_TH)
        ou, Ru, Tu = solve(su, jones=False)
        wu = time.time() - t0
        MA, MB = 3 * Mu - 2, 2 * Mu - 1
        t0 = time.time()
        sp = PMM2DStackPure(_PER, n_modes=max(MA, MB), n_orders=1,
                            layer_grids="per-layer")
        sp.add_layer(_TS[0], eps_cell=stripe_cell(2, 1), n_modes=MA)
        sp.add_layer(_TS[1], eps_cell=stripe_cell(3, 1), n_modes=MB)
        sp.set_source(_WL, theta=_TH)
        om, Rm, Tm = solve(sp, jones=False)
        wm = time.time() - t0
        eu = score_1d(ou, Ru, Tu, o1d, R1d, T1d)
        em = score_1d(om, Rm, Tm, o1d, R1d, T1d)
        cu = abs(float(Ru.sum(1)[1] + Tu.sum(1)[1] - 1.0))
        cm = abs(float(Rm.sum(1)[1] + Tm.sum(1)[1] - 1.0))
        rows[q] = {"Mu": Mu, "MA": MA, "MB": MB, "eig_dim": 2 * q * q,
                   "err_union": eu, "err_mortar": em, "ratio": em / eu,
                   "closure_union": cu, "closure_mortar": cm,
                   "closure_ratio": cm / max(cu, 1e-300),
                   "wall_union": wu, "wall_mortar": wm}
        print(f"[stripe] q={q:2d} dim={2*q*q:4d}: union(M={Mu}) {eu:.4e} "
              f"| mortar(M_A={MA},M_B={MB}) {em:.4e}  RATIO {em/eu:.4f}  "
              f"| closure {cu:.2e} -> {cm:.2e} ({cm/max(cu,1e-300):.2e}x)  "
              f"wall {wu:.1f}s / {wm:.1f}s", flush=True)
    out["rows"] = rows
    RES["stripe"] = out


# ==========================================================================
def sec_staircase():
    out = {}
    per, wl, th, ts = 0.9, 0.6, 0.20, 0.12
    duties = (0.5, 1 / 3, 1 / 6)
    ref = {}
    for deg in (12, 14, 16):
        st = PMMStack(per, degree=deg, far_field_orders=5)
        for d in duties:
            st.add_layer(ts, segments=[(d, EPS_P), (1 - d, EPS_H)])
        st.set_source(wl, theta=th)
        o, R, T = st.solve()[:3]
        ref[deg] = (np.asarray(o).ravel(), np.atleast_2d(R),
                    np.atleast_2d(T))
    o1d, R1d, T1d = ref[16]
    out["oracle_selfgap_12_14"] = float(max(
        np.max(np.abs(ref[12][1] - ref[14][1])),
        np.max(np.abs(ref[12][2] - ref[14][2]))))
    out["oracle_selfgap_14_16"] = float(max(
        np.max(np.abs(ref[14][1] - ref[16][1])),
        np.max(np.abs(ref[14][2] - ref[16][2]))))
    print(f"[staircase] oracle self-gap 12-14 "
          f"{out['oracle_selfgap_12_14']:.3e}, 14-16 "
          f"{out['oracle_selfgap_14_16']:.3e}", flush=True)
    rows = {}
    for q in (12, 18, 24):
        t0 = time.time()
        su = PMM2DStackPure(per, n_modes=q // 6 + 1, n_orders=1)
        for num in (3, 2, 1):
            su.add_layer(ts, eps_cell=stripe_cell(6, num))
        su.set_source(wl, theta=th)
        ou, Ru, Tu = solve(su, jones=False)
        wu = time.time() - t0
        t0 = time.time()
        sp = PMM2DStackPure(per, n_modes=8, n_orders=1,
                            layer_grids="per-layer")
        for N in (2, 3, 6):
            sp.add_layer(ts, eps_cell=stripe_cell(N, 1), n_modes=q // N + 1)
        sp.set_source(wl, theta=th)
        op, Rp, Tp = solve(sp, jones=False)
        wp = time.time() - t0
        eu = score_1d(ou, Ru, Tu, o1d, R1d, T1d)
        ep = score_1d(op, Rp, Tp, o1d, R1d, T1d)
        cu = abs(float(Ru.sum(1)[1] + Tu.sum(1)[1] - 1.0))
        cp = abs(float(Rp.sum(1)[1] + Tp.sum(1)[1] - 1.0))
        rows[q] = {"err_union": eu, "err_mortar": ep, "ratio": ep / eu,
                   "closure_union": cu, "closure_mortar": cp,
                   "wall_union": wu, "wall_mortar": wp,
                   "M_union": q // 6 + 1,
                   "M_i": [q // N + 1 for N in (2, 3, 6)]}
        print(f"[staircase] q={q}: union(M={q//6+1}) {eu:.4e} | per-layer"
              f"(M_i={[q//N+1 for N in (2,3,6)]}) {ep:.4e}  RATIO "
              f"{ep/eu:.4f}  closure {cu:.2e} -> {cp:.2e}  wall {wu:.1f}s /"
              f" {wp:.1f}s", flush=True)
    out["rows"] = rows
    # the DOF FLOOR claim, checked directly
    floor = {}
    try:
        PMM2DStackPure(per, n_modes=2)
        floor["M2_accepted"] = True
    except ValueError as exc:
        floor["M2_accepted"] = False
        floor["msg"] = str(exc)[:160]
    floor["q_min_union_N6"] = 6 * (3 - 1)
    floor["q_min_perlayer_N2"] = 2 * (3 - 1)
    floor["q_min_union_N12_sibling"] = 12 * (3 - 1)
    print(f"[staircase] DOF floor: M>=3 enforced ({not floor['M2_accepted']})"
          f"; union N=6 floor q={floor['q_min_union_N6']}, sibling union "
          f"N=12 floor q={floor['q_min_union_N12_sibling']}, per-layer N=2 "
          f"floor q={floor['q_min_perlayer_N2']}", flush=True)
    out["dof_floor"] = floor
    RES["staircase"] = out


# ==========================================================================
def sec_pillar():
    """The CORNER-DOMINATED 2-D pillar pair (F2): no exact oracle, so the
    reference is the union grid at the top of its OWN ladder and the ladder's
    last gap is the reference's uncertainty."""
    out = {}
    P, WL, th, ph = 1.2, 0.85, 0.18, 0.35

    def pil(N, lo, hi):
        c = np.full((N, N), EPS_H + 0j)
        c[lo:hi, lo:hi] = EPS_P
        return c

    cA6 = np.repeat(np.repeat(pil(2, 0, 1), 3, 0), 3, 1)
    cB6 = np.repeat(np.repeat(pil(3, 1, 2), 2, 0), 2, 1)

    def union(M):
        st = PMM2DStackPure(P, n_modes=M, n_orders=1)
        st.add_layer(0.30, eps_cell=cA6)
        st.add_layer(0.22, eps_cell=cB6)
        st.set_source(WL, theta=th, phi=ph)
        return solve(st, jones=False)

    lad, prev = {}, None
    for M in (3, 4, 5, 6):   # M = 6 is q = 30, an 1800-dimension region eig
        t0 = time.time()
        o, R, T = union(M)
        p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
        r00 = float(R[0, p0])
        g = None if prev is None else abs(r00 - prev) / abs(r00)
        lad[M] = {"q": 6 * (M - 1), "R00": r00, "gap_vs_prev": g,
                  "closure": float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0))),
                  "wall_s": time.time() - t0}
        prev = r00
        print(f"[pillar] union M={M} q={6*(M-1)}: R(0,0)={r00:.9f}  rel gap "
              f"{g if g is None else f'{g:.3e}'}  closure "
              f"{lad[M]['closure']:.2e}  {lad[M]['wall_s']:.1f}s", flush=True)
        RES["pillar"] = {"union_ladder": lad}
    ref_R00 = lad[6]["R00"]
    ref_unc = lad[6]["gap_vs_prev"]
    out["union_ladder"] = lad
    out["reference_R00"] = ref_R00
    out["reference_uncertainty"] = ref_unc
    rows = {}
    for Mu in (3, 4, 5):
        q = 6 * (Mu - 1)
        MA, MB = 3 * Mu - 2, 2 * Mu - 1
        t0 = time.time()
        sp = PMM2DStackPure(P, n_modes=max(MA, MB), n_orders=1,
                            layer_grids="per-layer")
        sp.add_layer(0.30, eps_cell=pil(2, 0, 1), n_modes=MA)
        sp.add_layer(0.22, eps_cell=pil(3, 1, 2), n_modes=MB)
        sp.set_source(WL, theta=th, phi=ph)
        o, R, T = solve(sp, jones=False)
        p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
        em = abs(float(R[0, p0]) - ref_R00) / abs(ref_R00)
        eu = abs(lad[Mu]["R00"] - ref_R00) / abs(ref_R00)
        cm = float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0)))
        rows[q] = {"err_union": eu, "err_mortar": em,
                   "ratio": em / max(eu, 1e-300),
                   "closure_union": lad[Mu]["closure"], "closure_mortar": cm,
                   "readable": bool(eu > 2 * ref_unc and em > 2 * ref_unc),
                   "wall_mortar": time.time() - t0}
        print(f"[pillar] q={q}: union {eu:.3e} | mortar(M_A={MA},M_B={MB}) "
              f"{em:.3e}  RATIO {em/max(eu,1e-300):.3f}  closure "
              f"{lad[Mu]['closure']:.2e} -> {cm:.2e}  READABLE="
              f"{rows[q]['readable']} (reference uncertainty "
              f"{ref_unc:.3e})", flush=True)
    out["rows"] = rows
    RES["pillar"] = out


SECTIONS = {"stripe": sec_stripe, "staircase": sec_staircase,
            "pillar": sec_pillar}


def main():
    for w in (sys.argv[1:] or list(SECTIONS)):
        t0 = time.time()
        SECTIONS[w]()
        print(f"--- {w} done in {time.time()-t0:.1f}s ---", flush=True)
    path = os.path.join(HERE, "v5_equal_dof.json")
    old = {}
    if os.path.exists(path):
        try:
            old = json.load(open(path))
        except Exception:          # a partial write from a crashed run
            old = {}
    old.update(RES)
    with open(path, "w") as f:
        json.dump(old, f, indent=1, sort_keys=True, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
