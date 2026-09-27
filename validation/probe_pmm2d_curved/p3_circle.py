"""P3 -- a genuinely CURVED cell: a circular dielectric pillar.

Fixture (lambda = 1; lengths in wavelengths): square lattice 1.2 x 1.2, ONE
circular pillar radius r = 0.36 (= 0.3 P) centred, eps 4 (n = 2) in air,
height 0.5, superstrate air, substrate n = 1.45, normal incidence, both
polarizations ('te' = E along y, 'tm' = E along x -- the library's
single-layer convention at normal incidence).  9 orders |m|,|n| <= 1 reported.

Modes (argv[1]):
  curved3 Mlo Mhi  -- transfinite (Gordon-Hall) map on a 3x3 (u,v) wall grid,
                      middle cell = the disk (det J = 0 at its 4 corners)
  curved5 Mlo Mhi  -- 5x5 wall grid, disk = inner 3x3 block (straight-edged
                      centre cell); same 4 singular loop corners
  quad M           -- quadrature sensitivity of the curved3 solve at degree M
  stair k Mlo Mhi  -- SHIPPED PMM2DStackPure (per-layer, explicit walls) on the
                      staircase with walls at c +- r i/k (i = 1..k), a cell
                      filled when its CENTRE is inside the circle: 4k steps
  film Mlo Mhi     -- the circle map applied to a UNIFORM eps-4 film (no
                      material boundary at all): isolates what the map's own
                      det J = 0 corners cost on a SMOOTH field; exact oracle =
                      the Fresnel slab (Airy sum)
  rcwa             -- SHIPPED rcwa_efficiency_2d_shapes (exact disk form
                      factor, Laurent) over n_orders, and rcwa_efficiency_2d on
                      pixel maps ('li') at increasing resolution
Output: p3_circle_<mode>[...].json

Run:  cd /c/tmp/lum_curved && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_curved \
        python validation/probe_pmm2d_curved/p3_circle.py curved3 4 12
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _curved_scratch as cs  # noqa: E402
import numpy as np  # noqa: E402

P = 1.2
R_CIRC = 0.36
EPS_P = 4.0
WL = 1.0
DEPTH = 0.5
N_SUP, N_SUB = 1.0, 1.45
FIX = {"period": P, "radius": R_CIRC, "eps_pillar": EPS_P, "eps_host": 1.0,
       "wl": WL, "depth": DEPTH, "n_sup": N_SUP, "n_sub": N_SUB,
       "incidence": "normal"}


def dump(name, res):
    with open(os.path.join(cs.HERE, name), "w") as f:
        json.dump(res, f, indent=1)


def run_curved(kind, m_lo, m_hi, nq_mult=None):
    if kind == "curved3":
        cmap, w = cs.circle_map_3x3(P, R_CIRC)
        disk = [(1, 1)]
    else:
        cmap, w = cs.circle_map_5x5(P, R_CIRC)
        disk = [(i, j) for i in (1, 2, 3) for j in (1, 2, 3)]
    n = len(w) - 1
    eps = np.ones((n, n), complex)
    for (i, j) in disk:
        eps[i, j] = EPS_P
    area = cs.mapped_area(cmap, w, w, disk, nq=80)
    total = cs.mapped_area(cmap, w, w, [(i, j) for i in range(n) for j in range(n)], nq=80)
    dj = cs.detJ_range(cmap, w, w)
    res = {"env": cs.env_record(), "fixture": FIX, "map": kind,
           "walls": w.tolist(),
           "disk_area_mapped": area, "disk_area_exact": np.pi * R_CIRC ** 2,
           "cell_area_mapped": total, "detJ_min_max_incl_corners": dj,
           "runs": []}
    tag = kind if nq_mult is None else f"{kind}_nq"
    for M in range(m_lo, m_hi + 1):
        nq = None if nq_mult is None else nq_mult * (M + 4)
        t0 = time.perf_counter()
        out = cs.solve_curved(P, P, w, w, M, eps, N_SUP, N_SUB, DEPTH, WL,
                              cmap=cmap, n_orders=3, nq=nq)
        row = {"M": M, "nq": nq, "dof": out["dof"],
               "t_total": time.perf_counter() - t0,
               "t_assemble": out["t_assemble"], "t_eig": out["t_eig"],
               "t_far": out["t_far"], "diag": out["diag"],
               "te": cs.table(out, "te"), "tm": cs.table(out, "tm"),
               "vec_te": cs.vec(out, "te").tolist(),
               "vec_tm": cs.vec(out, "tm").tolist(),
               "peak_rss_mb": cs.peak_rss_mb()}
        for pol in ("te", "tm"):
            row[f"closure_{pol}"] = abs(row[pol]["sumR"] + row[pol]["sumT"] - 1)
        res["runs"].append(row)
        print(f"{kind} M={M} nq={nq} dof={out['dof']} t={row['t_total']:.1f}s "
              f"R00te={row['te']['0,0'][0]:.10f} T00te={row['te']['0,0'][1]:.10f} "
              f"T10te={row['te']['1,0'][1]:.10f} clo={row['closure_te']:.1e}",
              flush=True)
        dump(f"p3_circle_{tag}.json", res)


def run_quad(M):
    cmap, w = cs.circle_map_3x3(P, R_CIRC)
    eps = np.ones((3, 3), complex)
    eps[1, 1] = EPS_P
    res = {"env": cs.env_record(), "fixture": FIX, "map": "curved3", "M": M,
           "runs": []}
    for nq in (2 * M + 8, 4 * M + 16, 8 * M + 32):
        out = cs.solve_curved(P, P, w, w, M, eps, N_SUP, N_SUB, DEPTH, WL,
                              cmap=cmap, n_orders=3, nq=nq)
        row = {"nq": nq, "te": cs.table(out, "te"), "tm": cs.table(out, "tm"),
               "vec_te": cs.vec(out, "te").tolist(),
               "vec_tm": cs.vec(out, "tm").tolist(), "t_total": out["t_total"]}
        res["runs"].append(row)
        print(f"quad M={M} nq={nq} R00te={row['te']['0,0'][0]:.12f}", flush=True)
        dump(f"p3_circle_quad_M{M}.json", res)


def stair_walls(k):
    c = P / 2
    # walls at c +- r i/k, i = 1..k (the centre line is NOT a wall: the two
    # cells it would split always share one fill by mirror symmetry, so
    # dropping it is exact and saves a segment per axis)
    inner = sorted([c - R_CIRC * i / k for i in range(1, k + 1)]
                   + [c + R_CIRC * i / k for i in range(1, k + 1)])
    w = np.array([0.0] + inner + [P])
    n = len(w) - 1
    mid = 0.5 * (w[:-1] + w[1:])
    eps = np.ones((n, n), complex)
    for i in range(n):
        for j in range(n):
            if (mid[i] - c) ** 2 + (mid[j] - c) ** 2 < R_CIRC ** 2:
                eps[i, j] = EPS_P
    area = 0.0
    for i in range(n):
        for j in range(n):
            if eps[i, j] != 1:
                area += (w[i + 1] - w[i]) * (w[j + 1] - w[j])
    return w, eps, area


def run_stair(k, m_lo, m_hi):
    from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
    w, eps, area = stair_walls(k)
    res = {"env": cs.env_record(), "fixture": FIX, "staircase_k": k,
           "steps": 4 * k, "walls": w.tolist(), "fill": eps.real.tolist(),
           "stair_area": area, "disk_area": np.pi * R_CIRC ** 2, "runs": []}
    for M in range(m_lo, m_hi + 1):
        t0 = time.perf_counter()
        st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB,
                            n_modes=M, n_orders=3, layer_grids="per-layer")
        st.add_layer(DEPTH, eps_cell=eps, x_walls=w[1:-1], y_walls=w[1:-1],
                     max_pencil_dof=20000)
        st.set_source(WL)
        o, R, T = st.solve(jones=False)
        o = np.asarray(o)
        R = np.asarray(R)
        T = np.asarray(T)
        out = {"orders": o, "R": {"tm": R[0], "te": R[1]},
               "T": {"tm": T[0], "te": T[1]}}
        n = len(w) - 1
        row = {"M": M, "dof": 2 * (n * (M - 1)) ** 2,
               "t_total": time.perf_counter() - t0,
               "te": cs.table(out, "te"), "tm": cs.table(out, "tm"),
               "vec_te": cs.vec(out, "te").tolist(),
               "vec_tm": cs.vec(out, "tm").tolist(),
               "peak_rss_mb": cs.peak_rss_mb()}
        res["runs"].append(row)
        print(f"stair k={k} M={M} dof={row['dof']} t={row['t_total']:.1f}s "
              f"R00te={row['te']['0,0'][0]:.10f} T10te={row['te']['1,0'][1]:.10f}",
              flush=True)
        dump(f"p3_circle_stair_k{k}.json", res)


def fresnel_slab(n1, n2, n3, d, wl):
    k0 = 2 * np.pi / wl
    r12 = (n1 - n2) / (n1 + n2)
    r23 = (n2 - n3) / (n2 + n3)
    t12 = 2 * n1 / (n1 + n2)
    t23 = 2 * n2 / (n2 + n3)
    ph = np.exp(1j * n2 * k0 * d)
    r = (r12 + r23 * ph ** 2) / (1 + r12 * r23 * ph ** 2)
    t = t12 * t23 * ph / (1 + r12 * r23 * ph ** 2)
    return float(abs(r) ** 2), float(abs(t) ** 2 * n3 / n1)


def run_film(m_lo, m_hi):
    cmap, w = cs.circle_map_3x3(P, R_CIRC)
    eps = np.full((3, 3), EPS_P, complex)
    Rex, Tex = fresnel_slab(N_SUP, np.sqrt(EPS_P), N_SUB, DEPTH, WL)
    res = {"env": cs.env_record(), "fixture": FIX, "map": "curved3 on a UNIFORM film",
           "fresnel_R": Rex, "fresnel_T": Tex, "runs": []}
    for M in range(m_lo, m_hi + 1):
        out = cs.solve_curved(P, P, w, w, M, eps, N_SUP, N_SUB, DEPTH, WL,
                              cmap=cmap, n_orders=3)
        row = {"M": M, "dof": out["dof"], "t_total": out["t_total"]}
        err = 0.0
        for pol in ("te", "tm"):
            vR = out["R"][pol].copy()
            vT = out["T"][pol].copy()
            o = out["orders"]
            i0 = int(np.nonzero((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
            vR[i0] -= Rex
            vT[i0] -= Tex
            e = float(max(np.max(np.abs(vR)), np.max(np.abs(vT))))
            row[f"err_{pol}"] = e
            err = max(err, e)
        row["err"] = err
        res["runs"].append(row)
        print(f"film M={M} dof={out['dof']} err={err:.3e}", flush=True)
        dump("p3_circle_film.json", res)


def run_rcwa():
    from lumenairy.elements.rcwa.twod import (
        rcwa_efficiency_2d,
        rcwa_efficiency_2d_shapes,
    )
    res = {"env": cs.env_record(), "fixture": FIX, "shapes": [], "pixel": []}
    shp = [{"shape": "disk", "eps": EPS_P, "radius": R_CIRC, "center": (P / 2, P / 2)}]
    for n in (4, 6, 8, 10, 12, 14):
        row = {"n_orders": n}
        t0 = time.perf_counter()
        for pol in ("te", "tm"):
            o, R, T = rcwa_efficiency_2d_shapes(P, P, 1.0, shp, N_SUB, N_SUP,
                                                DEPTH, WL, polarization=pol,
                                                n_orders_x=n, n_orders_y=n)
            out = {"orders": np.asarray(o), "R": {pol: np.asarray(R)},
                   "T": {pol: np.asarray(T)}}
            row[pol] = cs.table(out, pol)
            row[f"vec_{pol}"] = cs.vec(out, pol).tolist()
        row["t"] = time.perf_counter() - t0
        res["shapes"].append(row)
        print(f"shapes n={n} R00te={row['te']['0,0'][0]:.8f} t={row['t']:.1f}", flush=True)
        dump("p3_circle_rcwa.json", res)
    for S in (128, 256, 512):
        xs = (np.arange(S)) * P / S
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        cell = np.where((X - P / 2) ** 2 + (Y - P / 2) ** 2 < R_CIRC ** 2, EPS_P, 1.0).astype(complex)
        for n in (10, 12):
            row = {"S": S, "n_orders": n, "formulation": "li"}
            t0 = time.perf_counter()
            for pol in ("te", "tm"):
                o, R, T = rcwa_efficiency_2d(P, P, cell, N_SUB, N_SUP, DEPTH, WL,
                                             polarization=pol, n_orders_x=n,
                                             n_orders_y=n, formulation="li")
                out = {"orders": np.asarray(o), "R": {pol: np.asarray(R)},
                       "T": {pol: np.asarray(T)}}
                row[pol] = cs.table(out, pol)
                row[f"vec_{pol}"] = cs.vec(out, pol).tolist()
            row["t"] = time.perf_counter() - t0
            res["pixel"].append(row)
            print(f"pixel S={S} n={n} R00te={row['te']['0,0'][0]:.8f} t={row['t']:.1f}",
                  flush=True)
            dump("p3_circle_rcwa.json", res)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode in ("curved3", "curved5"):
        run_curved(mode, int(sys.argv[2]), int(sys.argv[3]))
    elif mode == "quad":
        run_quad(int(sys.argv[2]))
    elif mode == "stair":
        run_stair(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif mode == "film":
        run_film(int(sys.argv[2]), int(sys.argv[3]))
    elif mode == "rcwa":
        run_rcwa()
    else:
        raise SystemExit(mode)
