# SUM-AT-APERTURE probe -- run-to-run reproducibility of the SHIPPED path,
# measured because the probe tripped over it.
#
# Re-running arm A with identical inputs (``sumap_probe_121.py arma --force``)
# reproduced every INTENSITY metric to ~1e-6 but moved the returned field's
# GLOBAL PHASE by up to 2.88 rad.  Any arm-A/arm-B phase comparison is
# meaningless until that is characterised, so this script measures it directly:
# the same chain, same order, twice IN ONE PROCESS, and against whatever the
# probe has cached from an earlier process.
#
#   python sumap_repro_121.py [--order 0,0] [--reps 2]
#
# Reports, for each pair: the global piston (arg of the complex overlap), the
# field rel L2 before and after removing it, and the rel L2 of |E| -- which
# separates "a different answer" from "the same answer with a different
# absolute phase".
import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import _d121_common as C  # noqa: E402
import sumap_probe_121 as P  # noqa: E402

import lumenairy as la  # noqa: E402
import lumenairy.propagators.carrier as CAR  # noqa: E402


def one(S, m, n):
    (cx, cy), _pred, car = P.frame_centre(S, m, n)
    fr = dict(dx_out=P.DXO, N_out=P.TILE, centre_out=(cx, cy),
              n_fine_cap=P.NFC, window_factor=P.WF, ram_budget=P.RAMB)
    t0 = time.perf_counter()
    res = CAR.propagate_traced_carrier_chain(
        S['env_doe'], S['groups_post'], C.LAM, S['dx_doe'], r_in=car,
        ray_subsample=P.RS, n_workers=P.NW, final_distance=C.TRAILING,
        focus_readout=fr, final_leg=P.LEG)
    return np.asarray(res.field) * S['table'][(m, n)], time.perf_counter() - t0


def cmp(A, B, label):
    ip = complex(np.vdot(A, B))
    pist = float(np.angle(ip))
    nA = float(np.linalg.norm(A))
    print(f"  {label:28s} piston {pist:+.6f} rad   relL2 "
          f"{float(np.linalg.norm(B - A) / nA):.3e}   after piston "
          f"{float(np.linalg.norm(B - np.exp(1j * pist) * A) / nA):.3e}   "
          f"|E| relL2 "
          f"{float(np.linalg.norm(np.abs(B) - np.abs(A)) / nA):.3e}   "
          f"power ratio {float((np.abs(B) ** 2).sum() / (np.abs(A) ** 2).sum()):.9f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--order', default='0,0')
    ap.add_argument('--reps', type=int, default=2)
    a = ap.parse_args()
    m, n = (int(v) for v in a.order.split(','))
    S = P.setup()
    runs = []
    for i in range(a.reps):
        f, w = one(S, m, n)
        runs.append(f)
        print(f"rep {i}: {w:.1f} s", flush=True)
    print(f"ORDER ({m:+d},{n:+d}) -- same process, identical inputs")
    for i in range(1, len(runs)):
        cmp(runs[0], runs[i], f"rep 0 vs rep {i}")
    try:
        cached = P.load_meta(m, n)[1]
        print("versus the tile cached by an EARLIER process:")
        cmp(np.asarray(cached), runs[0], "cache vs rep 0")
    except Exception as e:                                  # noqa: BLE001
        print(f"  (no cached tile: {e})")
