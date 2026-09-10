"""V5 -- INTERLEAVED timings, ON vs OFF.

Two instruments, both required:

* ``--whole`` runs ONE (grid, M, layers, symmetry) configuration in a FRESH
  subprocess and prints its wall time, so the two arms differ only in the flag
  and never share a warmed allocator or a cache; the driver alternates them
  round-robin and takes the MIN over the rounds on both sides.
* ``--region`` times the region solve alone, in-process, after a warm-up.

Usage:  python v5_speed.py                  (driver -- runs everything)
        python v5_speed.py --whole <grid> <M> <layers> <sym>   (one arm)
"""
import json
import os
import subprocess
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import vfix as F  # noqa: E402

TIP = "C:/tmp/lum_vacc"
ROUNDS = 4


def build_cell(n):
    t = F.tensors()
    if n == 2:
        return F.centro_pair(t["lc"], 2)
    return F.centro_interior(t["lc"], t["lossy"])


def whole_arm(n, M, layers, sym):
    F.assert_arm(TIP)
    from lumenairy.elements.pmm import PMM2DStackPure
    base = build_cell(n)
    cells = [base * (1.0 + 0.017 * i) for i in range(layers)]
    t0 = time.perf_counter()
    st = PMM2DStackPure(F.PX, F.PY, n_superstrate=F.NSUP, n_substrate=F.NSUB,
                        n_modes=M, n_orders=5, symmetry=sym)
    for c in cells:
        st.add_layer(F.DEP / layers, eps_cell=c)
    st.set_source(F.WL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = st.solve(jones=True)
    dt = time.perf_counter() - t0
    print(json.dumps(dict(seconds=dt, sha=F.sha(*res))))


def region_timings():
    F.assert_arm(TIP)
    from lumenairy.elements.pmm.twod_staggered import (
        Granet2DTransverseE,
        _region_modes_oop,
    )
    K0 = 2.0 * np.pi / F.WL
    rows = []
    print("\nB. REGION solve, in-process, min over 3 timed calls after a "
          "warm-up")
    print(f"{'grid':6s} {'M':2s} {'4q^2':>6s} {'OFF [s]':>9s} {'ON [s]':>9s} "
          f"{'ratio':>6s}")
    for n in (2, 3):
        for M in (6, 7, 8):
            cell = build_cell(n)
            sol = Granet2DTransverseE(F.PX, F.PY, n, n, M, cell, k0=K0)
            qq = sol.q * sol.q
            for sym in (False, True):        # warm-up
                _region_modes_oop(sol, symmetry=sym)
            best = {}
            for sym in (False, True):
                ts = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    _region_modes_oop(sol, symmetry=sym)
                    ts.append(time.perf_counter() - t0)
                best[sym] = min(ts)
            rows.append(dict(grid=n, M=M, q4=4 * qq, off=best[False],
                             on=best[True], ratio=best[False] / best[True]))
            print(f"({n},{n})  {M:2d} {4 * qq:6d} {best[False]:9.3f} "
                  f"{best[True]:9.3f} {best[False] / best[True]:6.2f}")
    return rows


def drive():
    env = dict(os.environ)
    env.update(PYTHONPATH=TIP, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    configs = [(2, 6, 1), (2, 7, 1), (2, 8, 1), (3, 6, 1), (3, 7, 1),
               (3, 8, 1), (2, 7, 3), (3, 7, 3)]
    times = {c: {"off": [], "on": []} for c in configs}
    shas = {}
    for rnd in range(ROUNDS):
        for c in configs:
            for sym, key in ((0, "off"), (1, "on")):
                cmd = [sys.executable, os.path.abspath(__file__), "--whole",
                       str(c[0]), str(c[1]), str(c[2]), str(sym)]
                out = subprocess.run(cmd, capture_output=True, text=True,
                                     env=env, cwd=TIP)
                if out.returncode != 0:
                    raise SystemExit(out.stdout + out.stderr)
                rec = json.loads(out.stdout.strip().splitlines()[-1])
                times[c][key].append(rec["seconds"])
                shas.setdefault((c, key), rec["sha"])
        print(f"  round {rnd + 1}/{ROUNDS} done", flush=True)
    print()
    print("A. WHOLE solve, SEPARATE subprocesses, alternating, MIN over "
          f"{ROUNDS} rounds")
    print(f"{'grid':6s} {'M':2s} {'layers':6s} {'dense [s]':>10s} "
          f"{'factored [s]':>12s} {'speedup':>8s} {'spread off':>11s} "
          f"{'spread on':>10s}")
    rows = []
    for c in configs:
        off = min(times[c]["off"])
        on = min(times[c]["on"])
        so = (max(times[c]["off"]) - off) / off
        sn = (max(times[c]["on"]) - on) / on
        rows.append(dict(grid=c[0], M=c[1], layers=c[2], off=off, on=on,
                         speedup=off / on, spread_off=so, spread_on=sn,
                         rounds_off=times[c]["off"], rounds_on=times[c]["on"]))
        print(f"({c[0]},{c[0]})  {c[1]:2d} {c[2]:6d} {off:10.3f} {on:12.3f} "
              f"{off / on:7.2f}x {so * 100:10.1f}% {sn * 100:9.1f}%")
    single = [r for r in rows if r["layers"] == 1]
    multi = [r for r in rows if r["layers"] == 3]
    print()
    print(f"single-layer speedup {min(r['speedup'] for r in single):.2f} .. "
          f"{max(r['speedup'] for r in single):.2f}x   "
          f"(build doc claims 1.49 - 1.55x)")
    print(f"three-layer speedup  {min(r['speedup'] for r in multi):.2f} .. "
          f"{max(r['speedup'] for r in multi):.2f}x   "
          f"(build doc claims 1.80 - 1.87x)")
    reg = region_timings()
    print()
    print(f"region-solve ratio   {min(r['ratio'] for r in reg):.2f} .. "
          f"{max(r['ratio'] for r in reg):.2f}x   "
          f"(build doc claims 3.3 - 4.2x)")
    with open(os.path.join(HERE, "results", "v5_speed.json"), "w") as fh:
        json.dump(dict(whole=rows, region=reg), fh, indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--whole":
        whole_arm(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
                  bool(int(sys.argv[5])))
    else:
        drive()
