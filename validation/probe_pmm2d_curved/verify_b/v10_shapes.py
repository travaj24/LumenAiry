"""V10 -- B9 on THIS verifier's own shapes: an ellipse 0.42 x 0.24 (aspect
1.75, the builder used 0.40 x 0.28) and a sinusoidal ridge A = 0.08 on 0.35 <
x < 0.85 with v walls (0, 0.36, 0.9, 1.2) (the builder: A = 0.12, 0.3 .. 0.9,
thirds), against the shipped 2-D RCWA with the EXACT form factor.

  curved <ellipse|sine> <M>      the curved map ladder
  rcwa <ellipse|sine> <n>        RCWA, n orders per side (N = 2 n + 1)
  ffcheck                        the sinusoidal-ridge form factor (derived
                                 here by Jacobi-Anger) vs a 2048^2 pixel FFT
  summary                        per-entry fits r(N) = r_inf + a / N + b / N^2
                                 over N >= 13, the observed rate from
                                 successive differences, and the distance of
                                 r_inf (and of the builder-style 9 / 13
                                 Richardson pair) to the curved top rung
Output: v10_<...>.json
"""
import glob
import sys

import _vcommon as C
import numpy as np

ELL = (0.42, 0.24)
SX1, SX2, SA = 0.35, 0.85, 0.08
SVW = np.array([0.0, 0.36, 0.9, C.P])


def shape_map(kind):
    if kind == "ellipse":
        return C.vellipse3(*ELL)
    return C.vsine_ridge(SX1, SX2, SA, SVW)


def sine_ff(gxv, gyv, px, py):
    from scipy.special import jv
    n = np.rint(gyv / (2 * np.pi / py)).astype(int)
    m0 = np.abs(gxv) < 1e-12
    g = np.where(m0, 1.0, gxv)
    X = (np.exp(-1j * g * SX2) - np.exp(-1j * g * SX1)) / (-1j * g)
    out = np.where(m0, 0.0, jv(-n, g * SA) * X / px).astype(complex)
    out[m0 & (n == 0)] = (SX2 - SX1) / px
    return out


def rcwa(kind, n):
    from lumenairy.elements.rcwa import twod
    if kind == "ellipse":
        shapes = [{"shape": "ellipse", "eps": C.EPS_P, "semi_axes": ELL,
                   "center": (C.P / 2, C.P / 2)}]
        o_ff, o_val = None, None
    else:
        shapes = [{"shape": "sinstripe", "eps": C.EPS_P}]
        o_ff, o_val = twod._shape_form_factor, twod._validate_shapes

        def ff(shape, gxv, gyv, px, py):
            if shape["shape"] == "sinstripe":
                return sine_ff(gxv, gyv, px, py)
            return o_ff(shape, gxv, gyv, px, py)
        twod._shape_form_factor = ff
        twod._validate_shapes = lambda *a, **k: None
    try:
        rows = {}
        for pol in ("tm", "te"):
            o, R, T = twod.rcwa_efficiency_2d_shapes(
                C.P, C.P, 1.0, shapes, C.N_SUB, C.N_SUP, C.DEPTH, C.WL,
                polarization=pol, n_orders_x=n, n_orders_y=n)
            o = np.asarray(o)
            i = C.idx(o)
            rows[pol] = (np.asarray(R)[i], np.asarray(T)[i])
    finally:
        if o_ff is not None:
            twod._shape_form_factor, twod._validate_shapes = o_ff, o_val
    v = np.concatenate([rows["tm"][0], rows["te"][0], rows["tm"][1],
                        rows["te"][1]])
    C.dump(f"v10_rcwa_{kind}_n{n}.json", {"kind": kind, "n": n, "N": 2 * n + 1,
                                          "vec": v})
    print(kind, n, "done", flush=True)


def curved(kind, M):
    cm, eps = shape_map(kind)
    o, R, T, st, w = C.solve_map(cm, eps, M)
    sym = 0.0
    for (m, n) in C.ORD9:
        a, b = C.idx(o, [(m, n), (n, m)])
        sym = max(sym, abs(R[1, a] - R[0, b]), abs(T[1, a] - T[0, b]))
    C.dump(f"v10_curved_{kind}_M{M}.json",
           {"kind": kind, "M": M, "vec": C.vec(o, R, T),
            "closure": float(np.max(np.abs(R.sum(1) + T.sum(1) - 1))),
            "sym_te_tm": float(sym), "warnings": w})
    print(kind, M, "done", flush=True)


def ffcheck():
    Npx = 2048
    x = (np.arange(Npx) + 0.5) * C.P / Npx
    X, Y = np.meshgrid(x, x, indexing="ij")
    k = 2 * np.pi / C.P
    ind = ((X > SX1 + SA * np.sin(k * Y)) & (X < SX2 + SA * np.sin(k * Y)))
    F = np.fft.fft2(ind.astype(float)) / Npx ** 2
    worst = 0.0
    G = 2 * np.pi / C.P
    for m in range(-6, 7):
        for n in range(-6, 7):
            # pixel-centre phase: exp(-i G (m x + n y)) with x = (j + 1/2) h
            ph = np.exp(-1j * G * (m + n) * 0.5 * C.P / Npx)
            fp = F[m % Npx, n % Npx] * ph
            fa = sine_ff(np.array([G * m]), np.array([G * n]), C.P, C.P)[0]
            worst = max(worst, abs(fp - fa))
    C.dump("v10_ffcheck.json", {"pixels": Npx, "max_abs_diff_|m|,|n|<=6":
                                worst})
    print("ff check", worst)


def summary():
    out = {}
    for kind in ("ellipse", "sine"):
        cur = {}
        for f in glob.glob(C.HERE + f"/v10_curved_{kind}_M*.json"):
            d = C.load(f.split("\\")[-1].split("/")[-1])
            cur[d["M"]] = (np.array(d["vec"]), d["closure"], d["sym_te_tm"])
        Ms = sorted(cur)
        top = cur[Ms[-1]][0]
        rungs = {f"{a}->{b}": float(np.max(np.abs(cur[b][0] - cur[a][0])))
                 for a, b in zip(Ms, Ms[1:])}
        rc = {}
        for f in glob.glob(C.HERE + f"/v10_rcwa_{kind}_n*.json"):
            d = C.load(f.split("\\")[-1].split("/")[-1])
            rc[d["N"]] = np.array(d["vec"])
        Ns = sorted(rc)
        dist = {N: float(np.max(np.abs(rc[N] - top))) for N in Ns}
        # observed rate from successive differences (max over entries of
        # the per-entry log-ratio is noisy; use the norm of the differences)
        diffs = [np.linalg.norm(rc[b] - rc[a]) for a, b in zip(Ns, Ns[1:])]
        rates = []
        for i in range(len(diffs) - 1):
            Na, Nb, Nc = Ns[i], Ns[i + 1], Ns[i + 2]
            # for e ~ C N^-p: d_i ~ C p N^-(p+1) dN; dN equal (4) =>
            # log(d_i / d_{i+1}) / log(N_mid_{i+1} / N_mid_i) = p + 1
            ma, mb = 0.5 * (Na + Nb), 0.5 * (Nb + Nc)
            rates.append(float(np.log(diffs[i] / diffs[i + 1])
                               / np.log(mb / ma) - 1))
        # per-entry LSQ fits over N >= 13
        sel = [N for N in Ns if N >= 13]
        A = np.array([[1, 1 / N, 1 / N ** 2] for N in sel])
        Y = np.array([rc[N] for N in sel])
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        rinf = coef[0]
        A1 = np.array([[1, 1 / N] for N in sel])
        c1, *_ = np.linalg.lstsq(A1, Y, rcond=None)
        # leave-one-out spread of the 3-term fit (an error bar)
        loo = []
        for drop in range(len(sel)):
            keep = [i for i in range(len(sel)) if i != drop]
            cc, *_ = np.linalg.lstsq(A[keep], Y[keep], rcond=None)
            loo.append(np.max(np.abs(cc[0] - rinf)))
        rich = (13 * rc[13] - 9 * rc[9]) / 4 if 9 in rc and 13 in rc else None
        out[kind] = {
            "curved_Ms": Ms, "curved_rungs": rungs,
            "curved_top_closure": cur[Ms[-1]][1],
            "curved_sym_te_tm": {M: cur[M][2] for M in Ms},
            "rcwa_N": Ns, "rcwa_dist_to_curved_top": dist,
            "rcwa_dist_times_N": {N: dist[N] * N for N in Ns},
            "observed_rate_p": rates,
            "fit3_rinf_to_curved_top": float(np.max(np.abs(rinf - top))),
            "fit3_loo_spread": float(max(loo)),
            "fit2_rinf_to_curved_top": float(np.max(np.abs(c1[0] - top))),
            "rich_9_13_to_curved_top": (None if rich is None else
                                        float(np.max(np.abs(rich - top))))}
        print(kind, out[kind], flush=True)
    C.dump("v10_summary.json", out)


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "curved":
        curved(sys.argv[2], int(sys.argv[3]))
    elif m == "rcwa":
        rcwa(sys.argv[2], int(sys.argv[3]))
    elif m == "ffcheck":
        ffcheck()
    else:
        summary()
