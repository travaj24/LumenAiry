"""ROUND 2, probe 4 -- the FALSE-POSITIVE census (V-1), before and after.

The verification measured 110 of 648 realistic staircase configurations
(lossy substrate, theta 1.2-1.45, degree 6-10, wall steps 0.36-3.6 nm) that
round 1 REFUSES although the answer tracks the exact ``delta -> 0`` limit to
within the campaign's own CORRECT rule -- the super-unity there is ordinary
TRUNCATION, not the sliver, and round 1's first-named remedy silences the
refusal without moving the number.

This re-runs that exact product and scores three decisions per row:

  round 1   refuse iff screen(a) + passive + ``R+T-1`` > 1e-2
  round 2   the same trigger at 1e-3, then ONE re-solve on the prescribed
            ``min_feature`` grid: attributed (refuse) iff the super-unity
            VANISHES below the trigger, else RETURN with the warning
  cost      how often the arbiter runs, and what the extra solve costs

    python validation/probe_pmmstack_sliver_round2/r4_falsepos.py [out.json]
"""
import itertools
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

HERE = os.path.dirname(os.path.abspath(__file__))
P, WL = 1.2e-6, 0.85e-6
EH, EP = 2.25, 9.0
A0, B0 = 0.27865, 0.62505
DZ = 0.32e-6 / 4
NO_SNAP = P * 1e-12
TRIG = 1e-3


def build(d, deg, nsub, nsup, th, nl=2, eps=EP, mf=NO_SNAP):
    st = PMMStack(P, n_superstrate=nsup, n_substrate=nsub, degree=deg,
                  min_feature=mf, far_field_orders=31)
    for k in range(nl):
        dd = d * k / max(nl - 1, 1)
        st.add_layer(DZ, segments=[(A0 - dd, EH),
                                   (B0 + dd - (A0 - dd), eps),
                                   (1.0 - (B0 + dd), EH)])
    st.set_source(WL, theta=th)
    return st


def raw(st):
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, _J = st.solve()
    except (ValueError, NotImplementedError, RuntimeError):
        return None
    finally:
        ps.PMM_SLIVER_GUARD = was
    o = np.asarray(o).ravel()
    i = np.argsort(o)
    tot = np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)
    return (o[i], np.asarray(R)[1][i], np.asarray(T)[1][i], float(np.max(tot)),
            np.asarray(R)[:, i], np.asarray(T)[:, i])


def err(a, b):
    c = np.intersect1d(a[0], b[0])
    ia, ib = np.searchsorted(a[0], c), np.searchsorted(b[0], c)
    return float(max(np.abs(a[1][ia] - b[1][ib]).max(),
                     np.abs(a[2][ia] - b[2][ib]).max()))


def move_both(a, b):
    """The move the LIBRARY computes: BOTH incident polarizations."""
    c = np.intersect1d(a[0], b[0])
    ia, ib = np.searchsorted(a[0], c), np.searchsorted(b[0], c)
    return float(max(np.abs(a[4][:, ia] - b[4][:, ib]).max(),
                     np.abs(a[5][:, ia] - b[5][:, ib]).max()))


def err0(a, b):
    ia = int(np.argmin(np.abs(a[0])))
    ib = int(np.argmin(np.abs(b[0])))
    return float(max(abs(a[1][ia] - b[1][ib]), abs(a[2][ia] - b[2][ib])))


def prescribed(st):
    hit = ps._cross_layer_sliver([L[1] for L in st._layers],
                                 float(st.min_feature) / float(st.period))
    return None if hit is None else 2.0 * hit[3] * float(st.period)


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "r4_falsepos.json"))
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib, lumenairy.__version__)
    assert "lum_sliver2" in lib.replace("\\", "/"), lib
    t0 = time.time()
    rows, arb_times = [], []
    tried = 0
    for nsub, nsup, th, deg, eps, nl in itertools.product(
            (1.5 + 0.05j, 1.5 + 0.2j, 3.0 + 2.0j), (2.5, 3.5),
            (1.2, 1.3, 1.45), (6, 8, 10), (12.0, EP), (2, 4)):
        for d in (3e-3, 1e-3, 3e-4):
            tried += 1
            st = build(d, deg, nsub, nsup, th, nl, eps)
            ref = raw(build(0.0, deg, nsub, nsup, th, nl, eps))
            cur = raw(st)
            if ref is None or cur is None:
                continue
            e = err(cur, ref)
            kind = ("wrong" if e > 100.0 * d else
                    "right" if e <= 10.0 * d else "grey")
            passive = bool(ps._stack_provably_passive(st))
            mf = prescribed(st)
            row = dict(nsub=str(nsub), nsup=nsup, theta=th, degree=deg,
                       eps=str(eps), n_layers=nl, delta=d, err=e,
                       err_over_delta=e / d, RplusT=cur[3], kind=kind,
                       passive=passive, screen_hit=mf is not None)
            row["round1_refuse"] = bool(passive and mf is not None
                                        and cur[3] > 1.0 + 1e-2)
            row["triggers"] = bool(passive and mf is not None
                                   and cur[3] > 1.0 + TRIG)
            if row["triggers"]:
                ta = time.perf_counter()
                snap = raw(build(d, deg, nsub, nsup, th, nl, eps, mf=mf))
                arb_times.append(time.perf_counter() - ta)
                row["snapped_RplusT"] = None if snap is None else snap[3]
                row["w_wide"] = mf / (2.0 * P)
                row["move"] = None if snap is None else err(cur, snap)
                row["move_both"] = (None if snap is None
                                    else move_both(cur, snap))
                row["move0"] = None if snap is None else err0(cur, snap)
                row["same_shape"] = (None if snap is None
                                     else bool(len(cur[0]) == len(snap[0])))
                row["attributed"] = bool(snap is not None
                                         and snap[3] <= 1.0 + TRIG)
            else:
                row["attributed"] = False
            rows.append(row)
        if tried % 162 == 0:
            print(f"  {tried} configs ({time.time() - t0:.0f} s)", flush=True)

    right = [r for r in rows if r["kind"] == "right"]
    fp1 = [r for r in right if r["round1_refuse"]]
    fp2 = [r for r in right if r["attributed"]]
    trig_right = [r for r in right if r["triggers"]]
    print(f"\n  configs {tried}, scored {len(rows)}, CORRECT-by-continuity "
          f"{len(right)}")
    print(f"  ROUND 1 false positives : {len(fp1)} "
          f"({100.0 * len(fp1) / max(len(rows), 1):.1f}% of scored, "
          f"{100.0 * len(fp1) / max(len(right), 1):.1f}% of correct)")
    print(f"  ROUND 2 false positives : {len(fp2)}")
    print(f"  arbiter FIRED on        : {len(trig_right)} of {len(right)} "
          f"correct rows ({100.0 * len(trig_right) / max(len(right), 1):.1f}%)"
          f", {sum(1 for r in rows if r['triggers'])} of {len(rows)} total")
    if arb_times:
        print(f"  arbiter cost            : {np.mean(arb_times) * 1e3:.0f} ms "
              f"mean, {np.max(arb_times) * 1e3:.0f} ms max, "
              f"{np.sum(arb_times):.1f} s over {len(arb_times)} runs")
    if fp1:
        print(f"  round-1 FP err/delta {min(r['err_over_delta'] for r in fp1):.2f}"
              f" .. {max(r['err_over_delta'] for r in fp1):.2f};  R+T "
              f"{min(r['RplusT'] for r in fp1):.5f} .. "
              f"{max(r['RplusT'] for r in fp1):.5f}")
    wrongs = [r for r in rows if r["kind"] == "wrong"]
    print(f"  wrong rows in this box: {len(wrongs)}; refused round1 "
          f"{sum(1 for r in wrongs if r['round1_refuse'])}, round2 "
          f"{sum(1 for r in wrongs if r['attributed'])}")
    with open(out_path, "w") as f:
        json.dump(dict(meta=dict(lumenairy=lib, python=sys.version.split()[0],
                                 numpy=np.__version__, trigger=TRIG),
                       rows=rows, tried=tried,
                       n_right=len(right), n_fp_round1=len(fp1),
                       n_fp_round2=len(fp2),
                       arbiter_fired_on_correct=len(trig_right),
                       arbiter_mean_s=float(np.mean(arb_times))
                       if arb_times else None,
                       arbiter_n=len(arb_times),
                       wall_s=time.time() - t0), f, indent=1, default=str)
    print("wrote", out_path, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
