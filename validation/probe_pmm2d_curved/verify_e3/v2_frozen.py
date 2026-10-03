"""V2: THE FROZEN-GRID FINDING, re-measured on the verifier's own fixtures.

    python v2_frozen.py FAM M [twin] [numpy]

FAM: circle | fillet | sine.  Quantities: R00 (E_x), T00 (E_x), T00 (E_y).

twin : AD (jit jacrev) at x0; the twin's forward at x0 + d for the FD ladder
       steps and at the two "far" points x0 -+ 0.025 P (the frozen grid
       re-parametrised).
numpy: the shipped NumPy solve (its grid moves with x) at the same points ->
       FD(numpy) (Richardson, h^2 premise recorded) and the value at x0.
The analysis (v2_analyse.py) compares AD_M and FD(numpy)_M against the
highest-M FD(numpy) ("truth") and the offset AD_M - FD(numpy)_M.
"""
import sys

from _ve3 import WL, P, PMM2DStackPure, dump, jax, jnp, ladder, np, tic

from lumenairy.elements.pmm import Circle, Ellipse, FilletRect, SinusoidalWall

FAM, M = sys.argv[1], int(sys.argv[2])
DO_TW = "twin" in sys.argv[3:]
DO_NP = "numpy" in sys.argv[3:]
FAMS = {
    # x0, shapes(x), background, depth
    "circle": (0.33, lambda x: [Circle(0.6, 0.6, x, 3.5)], 1.0, 0.45),
    "fillet": (0.12, lambda x: [FilletRect(0.6, 0.6, 0.62, 0.5, x, 3.5)],
               1.0, 0.45),
    "sine": (0.1, lambda x: [SinusoidalWall("x", 0.6, x, eps=3.0)], 1.0,
             0.45),
    "ellipse": (0.33, lambda x: [Ellipse(0.6, 0.6, x, 0.22, 3.0)], 1.0,
                0.4),
    # the wall POSITION: moves the (u, v) wall (the amplitude does not)
    "sinex": (0.6, lambda x: [SinusoidalWall("x", x, 0.1, eps=3.0)], 1.0,
              0.45),
}
x0, shp, bg, depth = FAMS[FAM]
STEPS = [3e-4, 1e-4] if "short" in sys.argv[3:] else [1e-3, 3e-4, 1e-4]
FAR = [] if "nofar" in sys.argv[3:] else [-0.025, 0.025]
pts = sorted({round(x0 + s * k * P, 12) for s in STEPS for k in (-1, 1)}
             | {round(x0 + f * P, 12) for f in FAR} | {x0})
ONLY = [a for a in sys.argv[3:] if a.startswith("pt=")]
if ONLY:
    # ONE point of the NumPy set (split across jobs; v2_analyse merges)
    pts = [pts[int(ONLY[0][3:])]]


def build(x, backend="numpy"):
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=2, backend=backend)
    st.add_layer(depth, shapes=shp(x), background_eps=bg)
    st.set_source(WL)
    return st


def q_of(R, T, p0, xp):
    return xp.stack([xp.asarray(R[0, p0]), xp.asarray(T[0, p0]),
                     xp.asarray(T[1, p0])])


out = {"fam": FAM, "M": M, "x0": x0, "steps": STEPS, "far": FAR,
       "points": pts}
if DO_TW:
    t = tic()
    st = build(x0, "jax")
    tw = st.jax_twin()
    p0 = tw.p0
    out["tw_template_s"] = tic() - t

    def f(x):
        p = tw.params()
        p["layers"][0]["shapes"] = shp(x)
        _o, R, T, J = st.solve(params=p)
        return q_of(R, T, p0, jnp)
    fj = jax.jit(f)
    gj = jax.jit(jax.jacrev(f))
    t = tic()
    out["tw_value"] = np.asarray(fj(x0)).tolist()
    out["tw_fwd_compile_s"] = tic() - t
    t = tic()
    out["AD"] = np.asarray(gj(x0)).tolist()
    out["tw_grad_compile_s"] = tic() - t
    out["tw_at"] = {repr(x): np.asarray(fj(x)).tolist() for x in pts}
    rows, rich, ch, rat = ladder(fj, x0, STEPS, P)
    out["FD_twin"] = rich.tolist()
    out["FD_twin_ratios"] = rat.tolist()
if DO_NP:
    out["np_at"] = {}
    tt = []
    for x in pts:
        t = tic()
        o, R, T, J = build(x).solve()
        tt.append(tic() - t)
        p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
        out["np_at"][repr(x)] = q_of(R, T, p0, np).tolist()
        print(x, tt[-1], flush=True)
    out["np_solve_s"] = tt
    at = out["np_at"]
    if ONLY:
        print(dump(f"v2_frozen_{FAM}_M{M}_{ONLY[0].replace('=', '')}.json",
                   out))
        sys.exit(0)

    def fnp(x):
        return np.asarray(at[repr(round(x, 12))])
    rows, rich, ch, rat = ladder(fnp, x0, STEPS, P)
    out["FD_numpy"] = rich.tolist()
    out["FD_numpy_rows"] = rows.tolist()
    out["FD_numpy_ratios"] = rat.tolist() if rat.size else None
    out["np_value"] = at[repr(x0)]
tag = "".join(a[0] for a in sys.argv[3:] if not a.startswith("pt="))
print(dump(f"v2_frozen_{FAM}_M{M}_{tag}.json", out))
