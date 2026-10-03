"""D8 -- oblique (25 deg, phi 0) and conical (25 deg, phi 40 deg) incidence
on the TENSOR circular pillar of D4 (LC30 disk, r 0.36, P 1.2, n_sub 1.45).

Reciprocity (an independent physical identity the solver does not impose):
for a RECIPROCAL medium (eps = eps^T -- the LC is real symmetric) the
power-normalized 2 x 2 Jones block of the channel (incidence -> reflected
order (m, n)) and of its reversal (incidence along -k_mn -> order (m, n))
have the same singular values.  Two-sided:
* wrong pairing -- the reversed run's SPECULAR channel in place of the
  reciprocal one (shows the resolution of the comparison);
* non-reciprocal control -- the GYROTROPIC disk (eps != eps^T) must break
  it at the physics level (this is not a defect, it is what the identity
  is for).
Also the lossless closure, and the y-mirror at phi = 0 (the LC30 director
breaks the y-mirror, so it is recorded, not gated).

usage: python d8_oblique.py rung <c3|c5> <M> <theta> <phi> [gyro]
       python d8_oblique.py summary
"""
import json
import os
import sys
import time

import _common as C
import _dcommon as D
import numpy as np

P, R0, DEP, WL, NSUB = 1.2, 0.36, 0.5, 1.0, 1.45
ORDS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]


def tag(th, ph):
    return f"t{th:.4f}_p{ph:.4f}"


def reverse_angles(th_deg, ph_deg, m, n):
    st = np.sin(np.deg2rad(th_deg))
    kx = st * np.cos(np.deg2rad(ph_deg)) + m * WL / P
    ky = st * np.sin(np.deg2rad(ph_deg)) + n * WL / P
    kx, ky = -kx, -ky
    return (float(np.rad2deg(np.arcsin(np.hypot(kx, ky)))) + 0.0,
            float(np.rad2deg(np.arctan2(ky, kx))) + 0.0)


def rung(kind, M, th, ph, mat="lc"):
    cm = D.make_map(kind, P)
    N = cm.shape[0]
    eps = np.broadcast_to(np.eye(3, dtype=complex), (N, N, 3, 3)).copy()
    t33 = D.LC30 if mat == "lc" else D.GYRO
    for c in ([(1, 1)] if N == 3 else [(i, j) for i in (1, 2, 3)
                                       for j in (1, 2, 3)]):
        eps[c] = t33
    t0 = time.perf_counter()
    st = D.PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=NSUB,
                          n_modes=M, n_orders=3, cmap=cm)
    st.add_layer(DEP, eps_cell=eps)
    st.set_source(WL, theta=np.deg2rad(th), phi=np.deg2rad(ph))
    o, R, T, _J = st.solve()
    o = np.asarray(o)
    md = st._modal
    idx = C.idx(o, ORDS)
    res = {"map": kind, "M": M, "theta": th, "phi": ph, "mat": mat,
           "t": time.perf_counter() - t0,
           "R": np.asarray(R)[:, idx].tolist(),
           "T": np.asarray(T)[:, idx].tolist(),
           "closure": float(np.max(np.abs(np.asarray(R).sum(1)
                                          + np.asarray(T).sum(1) - 1))),
           "kx": [float(md["kx"][i]) for i in idx],
           "ky": [float(md["ky"][i]) for i in idx],
           "kz_ref": [[float(np.real(md["kz_ref"][i])),
                       float(np.imag(md["kz_ref"][i]))] for i in idx],
           "kz_inc": float(md["kz_inc"]), "kx0": md["kx0"], "ky0": md["ky0"]}
    for k in ("rx", "ry"):
        a = np.asarray(md[k])[:, idx]
        res[k] = [[[float(z.real), float(z.imag)] for z in row] for row in a]
    D.dump(f"d8_{mat}_{kind}_{tag(th, ph)}_M{M}.json", res)
    print(kind, mat, M, th, ph, f"clo={res['closure']:.1e} "
          f"t={res['t']:.0f}s", flush=True)


def jones_block(r, k):
    kx0, ky0, kzi = r["kx0"], r["ky0"], r["kz_inc"]
    A = np.array([[complex(*r["rx"][c][k]) for c in (0, 1)],
                  [complex(*r["ry"][c][k]) for c in (0, 1)]])
    kxo, kyo = r["kx"][k], r["ky"][k]
    kzo = complex(*r["kz_ref"][k])
    if abs(kzo.imag) > 1e-12 or kzo.real <= 0:
        return None
    kzo = kzo.real
    Gin = np.eye(2) + np.outer([kx0, ky0], [kx0, ky0]) / kzi ** 2
    Wout = (kzo / kzi) * (np.eye(2) + np.outer([kxo, kyo], [kxo, kyo])
                          / kzo ** 2)

    def msqrt(S, inv=False):
        w, V = np.linalg.eigh(S)
        d = w ** (-0.5 if inv else 0.5)
        return (V * d) @ V.conj().T
    return msqrt(Wout) @ A @ msqrt(Gin, inv=True)


def summary():
    here = os.path.dirname(os.path.abspath(__file__))
    groups = {}
    for fn in sorted(os.listdir(here)):
        if (fn.startswith("d8_") and fn.endswith(".json")
                and "summary" not in fn):
            r = json.load(open(os.path.join(here, fn)))
            groups.setdefault((r["mat"], r["map"], tag(r["theta"],
                                                       r["phi"])),
                              {})[r["M"]] = r
    out = {"closure": {}, "reciprocity": {}}
    for (mat, kind, tg), rs in groups.items():
        out["closure"][f"{mat}_{kind}_{tg}"] = {
            str(M): r["closure"] for M, r in sorted(rs.items())}
    pairs = [((25.0, 0.0), (-1, 0)), ((25.0, 40.0), (-1, 0)),
             ((25.0, 40.0), (0, -1))]
    for mat in ("lc", "gyro"):
        for kind in ("c3", "c5"):
            for (th, ph), (m, n) in pairs:
                tr, pr = reverse_angles(th, ph, m, n)
                fwd = groups.get((mat, kind, tag(th, ph)), {})
                rev = groups.get((mat, kind, tag(tr, pr)), {})
                k = ORDS.index((m, n))
                kw = ORDS.index((0, 0))
                rows = []
                for M in sorted(set(fwd) & set(rev)):
                    sf = np.linalg.svd(jones_block(fwd[M], k),
                                       compute_uv=False)
                    sr = np.linalg.svd(jones_block(rev[M], k),
                                       compute_uv=False)
                    sw = np.linalg.svd(jones_block(rev[M], kw),
                                       compute_uv=False)
                    rows.append({"M": M,
                                 "recip": float(np.max(np.abs(sf - sr))),
                                 "wrong_pair": float(np.max(np.abs(sf
                                                                   - sw)))})
                if rows:
                    out["reciprocity"][f"{mat}_{kind}_{tg_(th, ph)}_"
                                       f"({m},{n})"] = {
                        "reverse_deg": [tr, pr], "rows": rows}
    D.dump("d8_summary.json", out)
    print(json.dumps(out, indent=1)[:5000])


def tg_(th, ph):
    return tag(th, ph)


if __name__ == "__main__":
    if sys.argv[1] == "rung":
        rung(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]),
             float(sys.argv[5]), *(sys.argv[6:7] or ["lc"]))
    elif sys.argv[1] == "summary":
        summary()
