"""V8 -- reciprocity of a tensor disk at conical incidence, and the
NON-reciprocity of a gyrotropic one: is the builder's 1.6e-2 physics or
discretisation?  Two independent deciders:

1. ONSAGER with the magnetization reversed: for ANY linear medium the
   scattering matrix obeys S(eps) = S(eps^T)^T, so the forward channel of the
   GYRO disk and the REVERSED channel of the GYRO^T disk must have the same
   singular values -- to the discretisation level, NOT to 1.6e-2.
2. The UNMAPPED staircase limit: the shipped solver (no map) on staircase
   disks of k = 1, 2, 4 steps per quadrant (cells filled where more than
   half their area is inside the disk), converged in M per staircase: the
   gyro non-reciprocity of each staircase device is a physical number of
   THAT device; the sequence must approach the curved value.

Fixture: the builder's D4 / D8 pillar (r 0.36, P 1.2, depth 0.5, lambda 1,
air over n 1.45), channel (25 deg, phi 40 deg) -> reflected (-1, 0) and its
reversal, the builder's GYRO tensor and the LC30 control.

usage: python v8_recip.py rung KIND M MAT DIR
   KIND c3 | c5 | st1 | st2 | st4 ; MAT lc | gyro | gyroT ; DIR fwd | rev
       python v8_recip.py summary
"""
import glob
import json
import os
import sys
import time
import warnings

import numpy as np
from _vdcommon import HERE, PMM2DStackPure, circle, dump, lc

P, R0, DEP, WL, NSUB = 1.2, 0.36, 0.5, 1.0, 1.45
CTR = P / 2
GYRO = np.array([[2.25, 0.5j, 0.0], [-0.5j, 2.25, 0.0], [0.0, 0.0, 2.0]],
                dtype=complex)
MATS = {"lc": lc(30), "gyro": GYRO, "gyroT": GYRO.T.copy()}
TH, PH = 25.0, 40.0
CH = (-1, 0)


def reverse_angles(th_deg, ph_deg, m, n):
    s = np.sin(np.deg2rad(th_deg))
    kx = -(s * np.cos(np.deg2rad(ph_deg)) + m * WL / P)
    ky = -(s * np.sin(np.deg2rad(ph_deg)) + n * WL / P)
    return (float(np.rad2deg(np.arcsin(np.hypot(kx, ky)))),
            float(np.rad2deg(np.arctan2(ky, kx))))


def staircase(k):
    w = np.concatenate([[0.0], CTR + R0 * np.arange(-k, k + 1) / k, [P]])
    N = w.size - 1
    fill = np.zeros((N, N), bool)
    for i in range(N):
        for j in range(N):
            xs = np.linspace(w[i], w[i + 1], 41)
            ys = np.linspace(w[j], w[j + 1], 41)
            X, Y = np.meshgrid(0.5 * (xs[1:] + xs[:-1]),
                               0.5 * (ys[1:] + ys[:-1]), indexing="ij")
            fill[i, j] = np.mean((X - CTR) ** 2 + (Y - CTR) ** 2 < R0 ** 2) > 0.5
    return w, fill


def rung(kind, M, mat, d):
    t33 = MATS[mat]
    if kind in ("c3", "c5"):
        cm = circle(P, kind, R0 / P)
        N = cm.shape[0]
        eps = np.broadcast_to(np.eye(3, dtype=complex), (N, N, 3, 3)).copy()
        for c in ([(1, 1)] if N == 3 else [(i, j) for i in (1, 2, 3)
                                           for j in (1, 2, 3)]):
            eps[c] = t33
        st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=NSUB,
                            n_modes=M, n_orders=3, cmap=cm)
        st.add_layer(DEP, eps_cell=eps)
    else:
        w, fill = staircase(int(kind[2:]))
        N = w.size - 1
        eps = np.broadcast_to(np.eye(3, dtype=complex), (N, N, 3, 3)).copy()
        eps[fill] = t33
        st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=NSUB,
                            n_modes=M, n_orders=3, layer_grids="per-layer")
        st.add_layer(DEP, eps_cell=eps, x_walls=w, y_walls=w)
    th, ph = (TH, PH) if d == "fwd" else reverse_angles(TH, PH, *CH)
    t0 = time.perf_counter()
    st.set_source(WL, theta=np.deg2rad(th), phi=np.deg2rad(ph))
    o, R, T, _J = st.solve()
    o = np.asarray(o)
    md = st._modal
    want = [CH, (0, 0)]
    idx = [int(np.nonzero((o[:, 0] == m) & (o[:, 1] == n))[0][0])
           for m, n in want]
    res = {"kind": kind, "M": M, "mat": mat, "dir": d, "theta": th, "phi": ph,
           "closure": float(np.max(np.abs(np.asarray(R).sum(1)
                                          + np.asarray(T).sum(1) - 1))),
           "kx": [float(md["kx"][i]) for i in idx],
           "ky": [float(md["ky"][i]) for i in idx],
           "kz_ref": [complex(md["kz_ref"][i]) for i in idx],
           "kz_inc": float(md["kz_inc"]), "kx0": float(md["kx0"]),
           "ky0": float(md["ky0"]),
           "rx": np.asarray(md["rx"])[:, idx], "ry": np.asarray(md["ry"])[:, idx],
           "wall_s": time.perf_counter() - t0}
    dump(f"v8_{kind}_M{M}_{mat}_{d}.json", res)
    print(f"v8_{kind}_M{M}_{mat}_{d} clo {res['closure']:.1e} "
          f"wall {res['wall_s']:.0f}s")


def _c(z):
    if isinstance(z, dict):
        return np.array(z["re"]) + 1j * np.array(z["im"])
    return complex(*z) if isinstance(z, list) else z


def jones(r, k):
    rx, ry = _c(r["rx"]), _c(r["ry"])
    A = np.array([[rx[0][k], rx[1][k]], [ry[0][k], ry[1][k]]])
    kx0, ky0, kzi = r["kx0"], r["ky0"], r["kz_inc"]
    kzo = complex(*r["kz_ref"][k]).real
    Gin = np.eye(2) + np.outer([kx0, ky0], [kx0, ky0]) / kzi ** 2
    kxo, kyo = r["kx"][k], r["ky"][k]
    Wout = (kzo / kzi) * (np.eye(2) + np.outer([kxo, kyo], [kxo, kyo])
                          / kzo ** 2)

    def msqrt(S, p):
        w, V = np.linalg.eigh(S)
        return (V * w ** p) @ V.conj().T
    return msqrt(Wout, 0.5) @ A @ msqrt(Gin, -0.5)


def summary():
    runs = {}
    for f in glob.glob(os.path.join(HERE, "v8_*_M*_*.json")):
        r = json.load(open(f))
        runs[(r["kind"], r["M"], r["mat"], r["dir"])] = r
    out = {}
    for (kind, M, mat, d), r in sorted(runs.items()):
        if d != "fwd":
            continue
        sf = np.linalg.svd(jones(r, 0), compute_uv=False)
        row = {"closure_fwd": r["closure"], "sv_fwd": sf.tolist()}
        for rm in ("lc", "gyro", "gyroT"):
            rr = runs.get((kind, M, rm, "rev"))
            if rr is None:
                continue
            sr = np.linalg.svd(jones(rr, 0), compute_uv=False)
            sw = np.linalg.svd(jones(rr, 1), compute_uv=False)
            row[f"rev_{rm}"] = {"recip": float(np.max(np.abs(sf - sr))),
                                "signed_dsv": (sf - sr).tolist(),
                                "wrong_pair": float(np.max(np.abs(sf - sw)))}
        out[f"{kind}_M{M}_{mat}"] = row
    dump("v8_summary.json", {"rows": out})
    for k, v in out.items():
        s = " ".join(f"{kk}:{vv['recip']:.2e}" for kk, vv in v.items()
                     if kk.startswith("rev_"))
        print(k, f"clo {v['closure_fwd']:.1e}", s)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    if sys.argv[1] == "rung":
        rung(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
    else:
        summary()
