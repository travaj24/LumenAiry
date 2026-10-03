"""V6b (E3-5): a SYMMETRY-BREAKING shape gradient at an exactly four-fold
symmetric reference -- the canonical shape-optimisation start (a square
pillar's width alone, a circle deformed into an ellipse, a square fillet's
width).  The degenerate Bloch-mode pairs of the symmetric cell split to
FIRST order under such a perturbation, and the eigenvector-VJP must carry
the degenerate block's contribution.

    python v6b_symbreak.py M CASE

CASE: square_w | ellipse_a | fillet_sq_w | rect_nonsq_w (control: no
degeneracy) | square_wh (control: the symmetry-PRESERVING direction).
Per case: AD (jacrev) of (R00, T00 E_x, T00 E_y) vs FD(twin) and vs
FD(numpy) (the h^2 premise recorded), for tau_rel in (default 1e-12, 0,
1e-14, 1e-8); and the smallest eigenvalue gaps of the reference pencil.
"""
import sys

import scipy.linalg as sla
from _ve3 import WL, P, PMM2DStackPure, dump, jax, jnp, ladder, np

import lumenairy.elements.pmm._jax_twod_staggered as JT
from lumenairy.elements.pmm import Ellipse, FilletRect, Rect

M, CASE = int(sys.argv[1]), sys.argv[2]
CASES = {
    "square_w": (0.5, lambda x: [Rect(0.6, 0.6, x, 0.5, 3.5)]),
    "square_wh": (0.5, lambda x: [Rect(0.6, 0.6, x, x, 3.5)]),
    "rect_nonsq_w": (0.5, lambda x: [Rect(0.6, 0.6, x, 0.4, 3.5)]),
    "ellipse_a": (0.33, lambda x: [Ellipse(0.6, 0.6, x, 0.33, 3.5)]),
    "fillet_sq_w": (0.6, lambda x: [FilletRect(0.6, 0.6, x, 0.6, 0.1,
                                               3.5)]),
}
x0, shp = CASES[CASE]
out = {"M": M, "case": CASE, "x0": x0}


def build(x, backend="numpy"):
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=2, backend=backend)
    st.add_layer(0.45, shapes=shp(x), background_eps=1.0)
    st.set_source(WL)
    return st


def q(R, T, p0, xp):
    return xp.stack([R[0, p0], T[0, p0], T[1, p0]])


st = build(x0, "jax")
tw = st.jax_twin()
ref = tw.layers[0]["ref"]
g2 = sla.eig(ref.Lmat, -ref.Rmat, right=False)
s = np.sort_complex(g2)
gaps = np.abs(np.diff(s)) / np.max(np.abs(g2))
out["n_gaps_below_1e-12"] = int(np.sum(gaps < 1e-12))
out["n_gaps_below_1e-8"] = int(np.sum(gaps < 1e-8))


def f(x):
    p = tw.params()
    p["layers"][0]["shapes"] = shp(x)
    _o, R, T, J = st.solve(params=p)
    return q(R, T, tw.p0, jnp)


fj = jax.jit(f)
STEPS = [1e-3, 3e-4, 1e-4]
_r, fdt, _c, ratt = ladder(fj, x0, STEPS, P)


def fn(x):
    o, R, T, J = build(x).solve()
    return q(R, T, 12, np)


_r, fdn, _c, ratn = ladder(fn, x0, STEPS, P)
out["FD_twin"] = fdt.tolist()
out["FD_numpy"] = fdn.tolist()
out["ratios_twin"] = np.asarray(ratt).ravel().round(3).tolist()
out["ratios_numpy"] = np.asarray(ratn).ravel().round(3).tolist()
sc = float(np.max(np.abs(fdt)))
for tau in (None, 0.0, 1e-14, 1e-8):
    JT._E3_EIG_TAU_REL = tau
    try:
        g = np.asarray(jax.jit(jax.jacrev(lambda x: f(x)))(x0))
    finally:
        JT._E3_EIG_TAU_REL = None
    out[f"tau={tau}"] = {"AD": g.tolist(),
                         "AD_vs_FDtwin_rel": float(np.max(np.abs(g - fdt)))
                         / sc,
                         "AD_vs_FDnumpy_rel": float(np.max(np.abs(g - fdn)))
                         / sc}
    print(CASE, M, "tau", tau, out[f"tau={tau}"]["AD_vs_FDtwin_rel"],
          out[f"tau={tau}"]["AD_vs_FDnumpy_rel"], flush=True)
print("premise", out["ratios_twin"], out["ratios_numpy"],
      "FDtwin-FDnumpy", float(np.max(np.abs(fdt - fdn))) / sc)
print(dump(f"v6b_symbreak_{CASE}_M{M}.json", out))
