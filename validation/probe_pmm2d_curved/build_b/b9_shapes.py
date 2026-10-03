"""B9 -- two NEW curved shapes (no planner numbers): an ELLIPSE and a
SINUSOIDAL wall, each against the shipped 2-D RCWA with the EXACT form
factor at high order (the Fourier-floored reference: Laurent, algebraic in
the order count) and against the staircase limit of the shipped staggered
PMM.  Fixture otherwise plan 3.3 (period 1.2, eps 4 in air, height 0.5, air
over n = 1.45, lambda 1, normal incidence).

  ellipse  : axis-aligned, semi-axes (0.40, 0.28), centred; the 3 x 3
             transfinite map (walls through the parametric 45-degree points;
             four singular vertices)
  sine     : a constant-width ridge between x = 0.3 + A sin(2 pi y / p) and
             x = 0.9 + A sin(2 pi y / p), A = 0.12; the 3 x 3 map whose two
             interior vertical grid lines ARE the sinusoids (no singular
             vertex)

  curved S M     -> b9_<S>_M<M>.json      (S = ellipse | sine)
  rcwa S n       -> b9_<S>_rcwa_n<n>.json (2n + 1 orders per axis; te, tm)
  stair S k M    -> b9_<S>_stair_k<k>_M<M>.json
  summary        -> b9_shapes.json

The sinusoid has no closed-form shape in the shipped RCWA, so this probe
adds one (probe-only): the form factor of the ridge is
    F(m, n) = (1 / p_x) J_{-n}(m G_x A) (exp(-i m G_x x2) - exp(-i m G_x x1))
              / (-i m G_x),   F(0, n) = (x2 - x1) / p_x delta_{n0}
(Jacobi-Anger on exp(-i m G_x A sin(G_y y))), checked here against a
4096 x 4096 pixel FFT of the same ridge.
"""
import os
import sys
import time

import _common as C
import numpy as np

ELL = (0.40, 0.28)
X1, X2, AMP = 0.3, 0.9, 0.12


def shape_map(S):
    if S == "ellipse":
        cm, _w = C.CM._ellipse_map_3x3(C.P, ELL)
        eps = np.ones((3, 3), complex)
        eps[1, 1] = C.EPS_P
    else:
        cm, _w = C.CM._sine_stripe_map_3x3(C.P, X1, X2, AMP)
        eps = np.ones((3, 3), complex)
        eps[1, :] = C.EPS_P
    return cm, eps


def curved(S, M):
    cm, eps = shape_map(S)
    t0 = time.perf_counter()
    o, R, T, _J = C.solve(cm, eps, M)
    sym = 0.0
    for m in (-1, 0, 1):
        for n in (-1, 0, 1):
            i = C.idx_of(o, [(m, n)])[0]
            j = C.idx_of(o, [(n, m)])[0]
            sym = max(sym, abs(R[1, i] - R[0, j]), abs(T[1, i] - T[0, j]))
    res = {"env": C.env_record(), "shape": S, "M": M,
           "t": time.perf_counter() - t0, "vec": C.vec(o, R, T).tolist(),
           "te": C.table(o, R, T, 1), "tm": C.table(o, R, T, 0),
           "closure": float(np.max(np.abs(R.sum(1) + T.sum(1) - 1))),
           "sym_te_tm_transpose": float(sym),
           "singular_vertices": cm.singular_vertices}
    print(S, M, res["te"]["0,0"], f"clo={res['closure']:.1e} sym={sym:.1e}"
          f" t={res['t']:.0f}", flush=True)
    C.dump(f"b9_{S}_M{M}.json", res)


def sine_form_factor(gxv, gyv, px, py):
    from scipy.special import jv
    Gy = 2 * np.pi / py
    n = np.rint(gyv / Gy).astype(int)
    out = np.zeros(np.shape(gxv), dtype=complex)
    zero = np.abs(gxv) < 1e-12
    out[zero & (n == 0)] = (X2 - X1) / px
    g = np.where(zero, 1.0, gxv)
    X = (np.exp(-1j * g * X2) - np.exp(-1j * g * X1)) / (-1j * g)
    val = jv(-n, g * AMP) * X / px
    out[~zero] = val[~zero]
    return out


def patched_rcwa():
    """The shipped rcwa_efficiency_2d_shapes with a probe-only 'sinstripe'
    shape kind (form factor above); every other kind untouched."""
    from lumenairy.elements.rcwa import twod
    orig_ff = twod._shape_form_factor
    orig_val = twod._validate_shapes

    def ff(shape, gxv, gyv, px, py):
        if shape["shape"] == "sinstripe":
            return sine_form_factor(gxv, gyv, px, py)
        return orig_ff(shape, gxv, gyv, px, py)

    def val(fn, shapes, *a, **k):
        if any(s.get("shape") == "sinstripe" for s in shapes):
            return None
        return orig_val(fn, shapes, *a, **k)
    twod._shape_form_factor = ff
    twod._validate_shapes = val
    return twod


def check_ff():
    """The analytic sinusoid form factor against a pixel FFT."""
    S = 4096
    xs = (np.arange(S) + 0.5) * C.P / S
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    s = AMP * np.sin(2 * np.pi * Y / C.P)
    ind = ((X > X1 + s) & (X < X2 + s)).astype(float)
    F = np.fft.fft2(ind) / S ** 2
    G = 2 * np.pi / C.P
    err = 0.0
    for m in range(-4, 5):
        for n in range(-4, 5):
            # pixel-centre sampling phase of the DFT
            ph = np.exp(-1j * (m + n) * np.pi / S)
            a = sine_form_factor(np.array([m * G]), np.array([n * G]), C.P,
                                 C.P)[0]
            err = max(err, abs(F[m % S, n % S] * ph - a))
    return float(err)


def rcwa(S, n):
    twod = patched_rcwa()
    if S == "ellipse":
        shp = [{"shape": "ellipse", "eps": C.EPS_P, "semi_axes": ELL,
                "center": (C.P / 2, C.P / 2)}]
    else:
        shp = [{"shape": "sinstripe", "eps": C.EPS_P}]
    res = {"env": C.env_record(), "shape": S, "n": n}
    if S == "sine":
        res["ff_vs_pixel_fft"] = check_ff()
    t0 = time.perf_counter()
    for pol in ("te", "tm"):
        o, R, T = twod.rcwa_efficiency_2d_shapes(
            C.P, C.P, 1.0, shp, C.N_SUB, C.N_SUP, C.DEPTH, C.WL,
            polarization=pol, n_orders_x=n, n_orders_y=n)
        o = np.asarray(o)
        idx = C.idx_of(o)
        res[pol] = np.concatenate([np.asarray(R)[idx],
                                   np.asarray(T)[idx]]).tolist()
    res["t"] = time.perf_counter() - t0
    print(S, "rcwa", n, res.get("ff_vs_pixel_fft"), f"t={res['t']:.0f}",
          flush=True)
    C.dump(f"b9_{S}_rcwa_n{n}.json", res)


def stair_grid(S, k):
    c = C.P / 2
    if S == "ellipse":
        xw = sorted([c - ELL[0] * i / k for i in range(1, k + 1)]
                    + [c + ELL[0] * i / k for i in range(1, k + 1)])
        yw = sorted([c - ELL[1] * i / k for i in range(1, k + 1)]
                    + [c + ELL[1] * i / k for i in range(1, k + 1)])
        xw = np.array([0.0] + xw + [C.P])
        yw = np.array([0.0] + yw + [C.P])
        mx = 0.5 * (xw[:-1] + xw[1:])
        my = 0.5 * (yw[:-1] + yw[1:])
        eps = np.ones((mx.size, my.size), complex)
        for i in range(mx.size):
            for j in range(my.size):
                if ((mx[i] - c) / ELL[0]) ** 2 + ((my[j] - c) / ELL[1]) ** 2 \
                        < 1:
                    eps[i, j] = C.EPS_P
        return xw, yw, eps
    # sinusoid: R = 2k rows, each row's ridge at its centre's offset
    R = 2 * k
    rows = np.linspace(0.0, C.P, R + 1)
    yc = 0.5 * (rows[:-1] + rows[1:])
    off = AMP * np.sin(2 * np.pi * yc / C.P)
    xw = np.unique(np.round(np.concatenate([X1 + off, X2 + off]), 15))
    xw = np.concatenate([[0.0], xw, [C.P]])
    Nx = xw.size - 1
    # split rows (exactly, same fill) until Ny == Nx
    yw = list(rows)
    while len(yw) - 1 < Nx:
        d = np.diff(yw)
        j = int(np.argmax(d))
        yw.insert(j + 1, 0.5 * (yw[j] + yw[j + 1]))
    yw = np.array(yw)
    mx = 0.5 * (xw[:-1] + xw[1:])
    my = 0.5 * (yw[:-1] + yw[1:])
    eps = np.ones((Nx, my.size), complex)
    for j in range(my.size):
        r = min(int(my[j] / (C.P / R)), R - 1)
        lo, hi = X1 + off[r], X2 + off[r]
        for i in range(Nx):
            if lo < mx[i] < hi:
                eps[i, j] = C.EPS_P
    return xw, yw, eps


def stair(S, k, M):
    xw, yw, eps = stair_grid(S, k)
    t0 = time.perf_counter()
    o, R, T = C.solve_walls(xw[1:-1], yw[1:-1], eps, M,
                            max_pencil_dof=20000)
    res = {"env": C.env_record(), "shape": S, "k": k, "M": M,
           "x_walls": xw.tolist(), "y_walls": yw.tolist(),
           "t": time.perf_counter() - t0, "vec": C.vec(o, R, T).tolist()}
    print(S, "stair", k, M, f"t={res['t']:.0f}", flush=True)
    C.dump(f"b9_{S}_stair_k{k}_M{M}.json", res)


def summary():
    res = {"env": C.env_record(), "shapes": {}}
    for S in ("ellipse", "sine"):
        rows = []
        for M in range(4, 14):
            try:
                rows.append(C.load(f"b9_{S}_M{M}.json"))
            except FileNotFoundError:
                pass
        top = np.array(rows[-1]["vec"])
        lad = []
        for a, b in zip(rows, rows[1:] + [None]):
            lad.append({"M": a["M"], "closure": a["closure"],
                        "sym": a["sym_te_tm_transpose"], "t": a["t"],
                        "d_next": (float(np.max(np.abs(np.array(a["vec"])
                                                       - np.array(b["vec"]))))
                                   if b is not None else None)})
        rc = []
        for fn in sorted(os.listdir(C.HERE)):
            if fn.startswith(f"b9_{S}_rcwa_n"):
                r = C.load(fn)
                v = np.array(r["tm"][:9] + r["te"][:9] + r["tm"][9:]
                             + r["te"][9:])
                rc.append((r["n"], v, r.get("ff_vs_pixel_fft")))
        rc.sort(key=lambda z: z[0])
        rcw = []
        for i, (n, v, ffc) in enumerate(rc):
            row = {"n": n, "orders_per_axis": 2 * n + 1,
                   "dist_curved_top": float(np.max(np.abs(v - top)))}
            if ffc is not None:
                row["ff_vs_pixel_fft"] = ffc
            if i > 0:
                # Richardson assuming error ~ 1 / N, N = 2n + 1
                n0, v0, _ = rc[i - 1]
                N0, N1 = 2 * n0 + 1, 2 * n + 1
                ext = (N1 * v - N0 * v0) / (N1 - N0)
                row["richardson_dist_curved_top"] = float(
                    np.max(np.abs(ext - top)))
            rcw.append(row)
        st = []
        for fn in sorted(os.listdir(C.HERE)):
            if fn.startswith(f"b9_{S}_stair_k"):
                r = C.load(fn)
                st.append({"k": r["k"], "M": r["M"],
                           "dist_curved_top": float(np.max(np.abs(
                               np.array(r["vec"]) - top)))})
        res["shapes"][S] = {"top_M": rows[-1]["M"], "ladder": lad,
                            "rcwa": rcw, "stair": st}
    C.dump("b9_shapes.json", res)
    print(res)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "curved":
        curved(sys.argv[2], int(sys.argv[3]))
    elif mode == "rcwa":
        rcwa(sys.argv[2], int(sys.argv[3]))
    elif mode == "stair":
        stair(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif mode == "summary":
        summary()
    else:
        raise SystemExit(mode)
