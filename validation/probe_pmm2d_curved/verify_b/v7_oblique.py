"""V7 -- oblique and conical incidence on THIS verifier's curved maps.

  momentum <M> [map]   y-MOMENTUM: a y-UNIFORM stripe (eps 4 on 0.5 < x <
                       0.9) represented on a genuinely curved map (a
                       fictitious sinusoidal grid line x = 0.22 + 0.1 sin(2 pi
                       y / p) inside the air), conical incidence (25, 40) deg:
                       the structure has no y-dependence, so every order with
                       n != 0 must carry ZERO power; measured per order, both
                       inputs, against the unmapped stripe on straight walls.
                       map = 'curved' (default) or 'flat' (the same 4 x 4 grid
                       with the curve's amplitude 0)
  film <kind> <M> <th> <ph>
                       a uniform eps-4 film under a map vs the Airy slab, for
                       kind in c3, c5, c3r48, c3r24, flat3 (UNMAPPED 3 x 3
                       walls of the c3 map), flatwide (unmapped walls with a
                       0.72-wide middle cell = the disk's physical diameter),
                       c3id (identity transfinite map on c3's walls)
  rung <c3|c5> <M> <th> <ph>
                       the circle r = 0.36 at oblique / conical incidence:
                       R / T, closure, and the modal amplitudes for the
                       reciprocity check (stored)
Output: v7_<mode>_....json
"""
import sys
import time

import _vcommon as C
import numpy as np

CM = C.CM


def airy_rows(theta, phi, n2=2.0):
    k0 = C.K0
    ns = (C.N_SUP, n2, C.N_SUB)
    st = C.N_SUP * np.sin(theta)
    kz = [np.sqrt(complex(n * n - st * st)) for n in ns]

    def slab(r12, r23):
        ph = np.exp(2j * kz[1] * k0 * C.DEPTH)
        return abs((r12 + r23 * ph) / (1 + r12 * r23 * ph)) ** 2
    rs = slab((kz[0] - kz[1]) / (kz[0] + kz[1]),
              (kz[1] - kz[2]) / (kz[1] + kz[2]))
    e = [n * n for n in ns]
    rp = slab((e[1] * kz[0] - e[0] * kz[1]) / (e[1] * kz[0] + e[0] * kz[1]),
              (e[2] * kz[1] - e[1] * kz[2]) / (e[2] * kz[1] + e[1] * kz[2]))
    rows = []
    for et in ((1.0, 0.0), (0.0, 1.0)):
        # incident transverse E_t = et; s / p split (s along (-sin, cos))
        a = -np.sin(phi) * et[0] + np.cos(phi) * et[1]
        b = (np.cos(phi) * et[0] + np.sin(phi) * et[1]) / np.cos(theta)
        rows.append((a * a * rs + b * b * rp) / (a * a + b * b))
    return np.array(rows)


def film(kind, M, th, ph):
    th_r, ph_r = np.deg2rad(th), np.deg2rad(ph)
    t0 = time.time()
    if kind in ("flat3", "flatwide"):
        if kind == "flat3":
            h = 0.36 / np.sqrt(2)
            w = np.array([0.6 - h, 0.6 + h])
        else:
            w = np.array([0.24, 0.96])
        o, R, T = C.solve_walls(w, w, np.full((3, 3), 4.0 + 0j), M, th_r, ph_r)
    else:
        cm = {"c3": lambda: C.vcircle3(0.36)[0],
              "c5": lambda: C.vcircle5(0.36, 0.6)[0],
              "c3r48": lambda: C.vcircle3(0.48)[0],
              "c3r24": lambda: C.vcircle3(0.24)[0],
              "c3id": lambda: CM.TransfiniteMap(C.vcircle3(0.36)[0].u_walls,
                                                C.vcircle3(0.36)[0].v_walls)
              }[kind]()
        nx, ny = cm.shape
        o, R, T, _s, _w = C.solve_map(cm, np.full((nx, ny), 4.0 + 0j), M,
                                      th_r, ph_r)
    Rx = airy_rows(th_r, ph_r)
    i0 = C.idx(o, [(0, 0)])[0]
    R = R.copy()
    T = T.copy()
    R[:, i0] -= Rx
    T[:, i0] -= 1 - Rx
    err = float(max(np.abs(R).max(), np.abs(T).max()))
    C.dump(f"v7_film_{kind}_M{M}_t{th}_p{ph}.json",
           {"kind": kind, "M": M, "theta": th, "phi": ph, "err": err,
            "wall_s": time.time() - t0})
    print(kind, M, th, ph, f"{err:.2e}", flush=True)


def momentum(M, which="curved", n_orders=3):
    th, ph = np.deg2rad(25.0), np.deg2rad(40.0)
    A = 0.1 if which == "curved" else 0.0
    cm, eps = C.vyuniform_curved(0.5, 0.9, 0.22, A)
    o, R, T, st, w = C.solve_map(cm, eps, M, th, ph, n_orders=n_orders)
    off = o[:, 1] != 0
    leak = (R[:, off] + T[:, off])
    vw = cm.v_bounds
    o0, R0, T0 = C.solve_walls(np.array([0.22, 0.5, 0.9]), vw[1:-1],
                               np.array([[1, 1, 1, 1], [1, 1, 1, 1],
                                         [4, 4, 4, 4], [1, 1, 1, 1]],
                                        complex), M, th, ph,
                               min(n_orders, (4 * (M - 1) - 1) // 2))
    off0 = o0[:, 1] != 0
    on = ~off
    on = on & (np.abs(o[:, 0]) <= np.max(o0[:, 0]))
    i_on = [int(np.nonzero((o0[:, 0] == m) & (o0[:, 1] == 0))[0][0])
            for m in o[on, 0]]
    res = {"M": M, "map": which, "n_orders": n_orders,
           "leak_max_per_order": float(np.max(np.abs(leak))),
           "leak_total_per_input": np.abs(leak).sum(1),
           "leak_unmapped_max_per_order": float(np.max(np.abs(
               R0[:, off0] + T0[:, off0]))),
           "n0_orders_vs_unmapped": float(max(
               np.max(np.abs(R[:, on] - R0[:, i_on])),
               np.max(np.abs(T[:, on] - T0[:, i_on])))),
           "closure": float(np.max(np.abs(R.sum(1) + T.sum(1) - 1))),
           "warnings": w}
    sfx = "" if n_orders == 3 else f"_n{n_orders}"
    C.dump(f"v7_momentum_{which}_M{M}{sfx}.json", res)
    print(res, flush=True)


def rung(kind, M, th, ph):
    t0 = time.time()
    cm, eps = (C.vcircle3(0.36) if kind == "c3" else C.vcircle5(0.36, 0.6))
    o, R, T, st, w = C.solve_map(cm, eps, M, np.deg2rad(th), np.deg2rad(ph))
    md = st._modal
    keep = {k: np.asarray(md[k]).tolist() if not np.iscomplexobj(md[k])
            else [np.real(md[k]).tolist(), np.imag(md[k]).tolist()]
            for k in ("orders", "kx", "ky", "kz_ref", "kz_trn")}
    for k in ("rx", "ry"):
        keep[k] = [np.real(md[k]).tolist(), np.imag(md[k]).tolist()]
    keep["kx0"], keep["ky0"] = float(md["kx0"]), float(md["ky0"])
    kzi = md["kz_inc"]
    keep["kz_inc"] = float(np.real(kzi))
    C.dump(f"v7_rung_{kind}_M{M}_t{th:.4f}_p{ph:.4f}.json",
           {"kind": kind, "M": M, "theta": th, "phi": ph,
            "vec": C.vec(o, R, T), "R": R, "T": T, "o": o,
            "closure": float(np.max(np.abs(R.sum(1) + T.sum(1) - 1))),
            "modal": keep, "warnings": w, "wall_s": time.time() - t0})
    print(kind, M, th, ph, "done", flush=True)


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "momentum":
        momentum(int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3
                 else "curved", int(sys.argv[4]) if len(sys.argv) > 4 else 3)
    elif m == "film":
        film(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]),
             float(sys.argv[5]))
    else:
        rung(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]),
             float(sys.argv[5]))
