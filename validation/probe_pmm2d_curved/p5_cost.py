"""P5 -- COST of a curved cell against the same-grid rectangular cell.

One REGION solve (assembly + the modal eig + the Eq.-25 H partner), each
configuration in its OWN child process so the peak working set is that
configuration's alone.  Degrees 6, 8, 10.

  rect3_shipped   : lumenairy Granet2DTransverseE (3x3 pillar) + _region_modes
                    (the shipped piecewise-constant kron assembly + QZ)
  rect3_scratch   : scratch CurvedGranet, identity map, same 3x3 walls
  circle3         : scratch CurvedGranet, transfinite circle map, 3x3 walls
  rect5_scratch   : scratch identity map on the fillet's 5x5 walls
  fillet5         : scratch fillet map (r/side 0.1), 5x5 walls
The scratch rows run the eig both ways: QZ (what the shipped in-plane path
pays) and Cholesky-whitened standard eig (the pencil's right-hand matrix -R is
Hermitian positive definite in the mapped frame too).

Run:  cd /c/tmp/lum_curved && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_curved \
        python validation/probe_pmm2d_curved/p5_cost.py
Output: p5_cost.json.  Numbers are UPPER BOUNDS when the box is loaded (the
load average at run time is recorded).
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def child(kind, M, whiten):
    os.environ["CURVED_PROBE_WHITEN"] = "1" if whiten else "0"
    sys.path.insert(0, HERE)
    import _curved_scratch as cs
    import numpy as np
    P = 1.2
    rec = {"kind": kind, "M": M, "whiten": whiten}
    t0 = time.perf_counter()
    if kind == "rect3_shipped":
        from lumenairy.elements.pmm.twod_staggered import (
            Granet2DTransverseE,
            _region_modes,
        )
        eps = np.ones((3, 3), complex)
        eps[1, 1] = 4.0
        sol = Granet2DTransverseE(P, P, np.array([0, 0.3, 0.9, P]),
                                  np.array([0, 0.3, 0.9, P]), M, eps,
                                  k0=2 * np.pi)
        rec["t_assemble"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        _region_modes(sol)
        rec["t_modes"] = time.perf_counter() - t1
        rec["dof"] = 2 * sol.q * sol.q
    else:
        if kind == "rect3_scratch":
            cmap, w = None, np.array([0, 0.3, 0.9, P])
            cells = [(1, 1)]
        elif kind == "circle3":
            cmap, w = cs.circle_map_3x3(P, 0.36)
            cells = [(1, 1)]
        elif kind in ("rect5_scratch", "fillet5"):
            fm, w = cs.fillet_map_5x5(P, 0.3, 0.06)
            cmap = None if kind == "rect5_scratch" else fm
            cells = [(i, j) for i in (1, 2, 3) for j in (1, 2, 3)]
        n = len(w) - 1
        eps = np.ones((n, n), complex)
        for c in cells:
            eps[c] = 4.0
        sol = cs.CurvedGranet(P, P, w, w, M, eps, cmap, k0=2 * np.pi)
        rec["t_assemble"] = sol.t_assemble
        t1 = time.perf_counter()
        cs.curved_region_modes(sol)
        rec["t_modes"] = time.perf_counter() - t1
        rec["dof"] = 2 * sol.qq
    rec["peak_rss_mb"] = cs.peak_rss_mb()
    print("RESULT " + json.dumps(rec), flush=True)


def main():
    import psutil
    rows = []
    plan = []
    for M in (6, 8, 10):
        plan.append(("rect3_shipped", M, False))
        for kind in ("rect3_scratch", "circle3"):
            plan.append((kind, M, False))
            plan.append((kind, M, True))
    for M in (6, 8):
        for kind in ("rect5_scratch", "fillet5"):
            plan.append((kind, M, True))
    for kind, M, wh in plan:
        load = psutil.cpu_percent(interval=1.0)
        avail = psutil.virtual_memory().available / 2**30
        cmd = [sys.executable, os.path.abspath(__file__), "child", kind, str(M),
               "1" if wh else "0"]
        p = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
        line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
        if not line:
            print(p.stdout, p.stderr)
            continue
        rec = json.loads(line[0][7:])
        rec["box_cpu_percent_before"] = load
        rec["box_avail_gb_before"] = avail
        rows.append(rec)
        print(json.dumps(rec), flush=True)
        with open(os.path.join(HERE, "p5_cost.json"), "w") as f:
            json.dump({"note": "one region solve per child process; BLAS "
                       "pinned to 1 thread; timings are upper bounds on a "
                       "loaded box", "rows": rows}, f, indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1")
    else:
        main()
