"""P2 -- PURE-STRETCH SELF-CONSISTENCY (the decisive feasibility gate).

A 1-D nonlinear periodic stretch x = u + a sin(2 pi u / px), y = v is applied
to an ORDINARY rectangular cell whose (u, v) walls are the PREIMAGES of the
physical walls.  The physical structure is unchanged, so the mapped solve must
converge to the SAME per-order efficiencies as the unmapped one.  This
exercises the effective tensors (eps'_t = eps diag(1/f', f'), eps'_33 = eps f',
chi_t = diag(f', 1/f'), chi33 = 1/f'), the variable-coefficient masses, the
mapped half-space eig and the pulled-back far-field projector -- with no
curved geometry at all.

Fixtures (lambda = 1, period 1.2 x 1.2, depth 0.5, air over n = 1.45,
eps 4 in air, normal incidence, BOTH polarizations):
  stripe : y-uniform ridge x in [0.25, 0.85]  (corner-free -> spectral; the
           1-D PMM pmm_efficiency_1d at degree 40 is an EXACT per-order oracle)
  pillar : x in [0.25, 0.85], y in [0.30, 0.90]  (corner-capped -> algebraic;
           oracle = the unmapped a = 0 solve at the top of the ladder)
a in {0, 0.05, 0.15} x px   (0.15 px: f' spans 0.058 .. 1.94, a 33:1 stretch)

Run:  cd /c/tmp/lum_curved && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_curved \
        python validation/probe_pmm2d_curved/p2_stretch.py stripe 4 11
        (... p2_stretch.py film   -> the uniform-film gate, p2_stretch_film.json)
Output: p2_stretch_<fixture>.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _curved_scratch as cs  # noqa: E402
import numpy as np  # noqa: E402

PX = PY = 1.2
WL = 1.0
DEPTH = 0.5
N_SUP, N_SUB = 1.0, 1.45
XW = np.array([0.0, 0.25, 0.85, PX])
YW_PILLAR = np.array([0.0, 0.30, 0.90, PY])
YW_STRIPE = np.array([0.0, 0.4, 0.8, PY])
A_LIST = (0.0, 0.05, 0.15)


def oracle_1d():
    from lumenairy.elements.pmm.oned import pmm_efficiency_1d
    out = {}
    for pol in ("te", "tm"):
        vals = {}
        for deg in (30, 40):
            o, R, T = pmm_efficiency_1d(PX, 2.0, 1.0, N_SUB, N_SUP, DEPTH,
                                        0.6 / PX, WL, polarization=pol,
                                        degree=deg, far_field_orders=7)
            o = np.asarray(o)
            vals[deg] = {int(m): (float(R[i]), float(T[i])) for i, m in enumerate(o)}
        out[pol] = vals
    return out


def film():
    """A UNIFORM eps-4 film under the stretch -- exact oracle = the Airy slab.
    Isolates the mapped half-spaces + mapped layer + cofactor projector from
    any material boundary (the Phase-A gate A3)."""
    k0 = 2 * np.pi / WL
    n2 = 2.0
    r12 = (N_SUP - n2) / (N_SUP + n2)
    r23 = (n2 - N_SUB) / (n2 + N_SUB)
    t12 = 2 * N_SUP / (N_SUP + n2)
    t23 = 2 * n2 / (n2 + N_SUB)
    ph = np.exp(1j * n2 * k0 * DEPTH)
    Rex = abs((r12 + r23 * ph ** 2) / (1 + r12 * r23 * ph ** 2)) ** 2
    Tex = abs(t12 * t23 * ph / (1 + r12 * r23 * ph ** 2)) ** 2 * N_SUB / N_SUP
    res = {"env": cs.env_record(), "fresnel_R": Rex, "fresnel_T": Tex, "runs": []}
    for a_frac in (0.05, 0.15):
        cmap = cs.SineStretchX(a_frac * PX, PX)
        for M in range(4, 8):
            out = cs.solve_curved(PX, PY, 3, 3, M, np.full((3, 3), 4.0 + 0j),
                                  N_SUP, N_SUB, DEPTH, WL, cmap=cmap, n_orders=3)
            row = {"a_over_px": a_frac, "M": M}
            o = out["orders"]
            i0 = int(np.nonzero((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
            for pol in ("te", "tm"):
                R = out["R"][pol].copy()
                T = out["T"][pol].copy()
                R[i0] -= Rex
                T[i0] -= Tex
                row[f"err_{pol}"] = float(max(np.max(np.abs(R)), np.max(np.abs(T))))
            res["runs"].append(row)
            print(json.dumps(row), flush=True)
    with open(os.path.join(cs.HERE, "p2_stretch_film.json"), "w") as f:
        json.dump(res, f, indent=1)


def main():
    if sys.argv[1] == "film":
        film()
        return
    fixture = sys.argv[1]
    m_lo, m_hi = int(sys.argv[2]), int(sys.argv[3])
    a_list = A_LIST if len(sys.argv) < 5 else tuple(float(x) for x in sys.argv[4].split(","))
    yw = YW_STRIPE if fixture == "stripe" else YW_PILLAR
    eps = np.ones((3, 3), complex)
    if fixture == "stripe":
        eps[1, :] = 4.0
    else:
        eps[1, 1] = 4.0
    res = {"env": cs.env_record(), "fixture": {
        "name": fixture, "period": [PX, PY], "wl": WL, "depth": DEPTH,
        "n_sup": N_SUP, "n_sub": N_SUB, "x_walls_phys": XW.tolist(),
        "y_walls": yw.tolist(), "eps_cell": eps.real.tolist(),
        "map": "x = u + a sin(2 pi u / px), y = v", "a_over_px": list(a_list)},
        "runs": []}
    if fixture == "stripe":
        t0 = time.perf_counter()
        res["oracle_1d"] = {pol: {str(d): v for d, v in dd.items()}
                            for pol, dd in oracle_1d().items()}
        res["oracle_1d_time"] = time.perf_counter() - t0
    for a_frac in a_list:
        a = a_frac * PX
        cmap = cs.SineStretchX(a, PX) if a != 0 else None
        uw = XW.copy() if a == 0 else cmap.u_of_x(XW)
        uw[0], uw[-1] = 0.0, PX
        if a != 0:
            assert np.max(np.abs(cmap.x_of_u(uw) - XW)) < 1e-14
        for M in range(m_lo, m_hi + 1):
            t0 = time.perf_counter()
            out = cs.solve_curved(PX, PY, uw, yw, M, eps, N_SUP, N_SUB, DEPTH,
                                  WL, cmap=cmap, n_orders=3)
            row = {"a_over_px": a_frac, "M": M, "dof": out["dof"],
                   "u_walls": uw.tolist(),
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
            print(f"a={a_frac} M={M} dof={out['dof']} t={row['t_total']:.1f}s "
                  f"R00te={row['te']['0,0'][0]:.10f} T00tm={row['tm']['0,0'][1]:.10f} "
                  f"clo={row['closure_te']:.1e} split={out['diag'].get('geom_split_rel')}",
                  flush=True)
            with open(os.path.join(cs.HERE, f"p2_stretch_{fixture}.json"), "w") as f:
                json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
