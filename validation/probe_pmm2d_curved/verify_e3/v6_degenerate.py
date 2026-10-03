"""V6 (E3-5): NEARLY-degenerate Bloch modes.  The four-fold symmetry of a
centred circle makes exactly degenerate pairs; an ellipse with semi-axes
(r, r (1 + delta)) splits them by ~delta.  As the splitting passes through
the eigenvector-VJP broadening (tau_rel * max|lam|, default 1e-12), is the
gradient still right?

    python v6_degenerate.py M

Per delta (1e-14 .. 1e-3): the twin built AT that ellipse; the smallest
relative eigenvalue gap of its layer pencil (NumPy QZ on the reference
solver); AD of (R00, T00) w.r.t. (a, b, cx) vs the Richardson FD(twin)
(the FD of the same smooth forward -- T is analytic through a degeneracy);
for tau_rel in (default, 0, 1e-8, 1e-6).
"""
import sys

import scipy.linalg as sla
from _ve3 import WL, P, PMM2DStackPure, dump, jax, jnp, ladder, np

import lumenairy.elements.pmm._jax_twod_staggered as JT
from lumenairy.elements.pmm import Ellipse

M = int(sys.argv[1])
R0 = 0.33
DELTAS = ([float(a) for a in sys.argv[2].split(",")] if len(sys.argv) > 2
          else [0.0, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-8, 1e-6, 1e-4,
                1e-3])
TAUS = [None, 0.0, 1e-8, 1e-6]
PART = sys.argv[3] if len(sys.argv) > 3 else ""
out = {"M": M, "r": R0, "rows": []}
for dl in DELTAS:
    b0 = R0 * (1.0 + dl)
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=2, backend="jax")
    st.add_layer(0.45, shapes=[Ellipse(0.6, 0.6, R0, b0, 3.5)],
                 background_eps=1.0)
    st.set_source(WL)
    tw = st.jax_twin()
    ref = tw.layers[0]["ref"]
    g2 = sla.eig(ref.Lmat, -ref.Rmat, right=False)
    s = np.sort_complex(g2)
    gaps = np.abs(np.diff(s)) / np.max(np.abs(g2))
    # pairs counted on the SORTED spectrum (nearest neighbours in the sort)
    row = {"delta": dl, "min_rel_gap": float(np.min(gaps)),
           "n_pairs_below_1e-10": int(np.sum(gaps < 1e-10)),
           "n_pairs_below_1e-6": int(np.sum(gaps < 1e-6))}

    def f(v, tw=tw, st=st):
        p = tw.params()
        p["layers"][0]["shapes"] = [Ellipse(v[2], 0.6, v[0], v[1], 3.5)]
        _o, R, T, J = st.solve(params=p)
        return jnp.stack([R[0, tw.p0], T[0, tw.p0]])
    v0 = np.array([R0, b0, 0.6])
    fj = jax.jit(f)
    fd = []
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1.0
        _r, d, _c, _rat = ladder(lambda t: np.asarray(fj(v0 + t * e)), 0.0,
                                 [1e-3, 3e-4, 1e-4], P)
        fd.append(d)
    fd = np.stack(fd, axis=1)                    # (2 quantities, 3 params)
    row["FD"] = fd.tolist()
    sc = float(np.max(np.abs(fd[:, :2])))
    for tau in TAUS:
        JT._E3_EIG_TAU_REL = tau
        try:
            g = np.asarray(jax.jit(jax.jacrev(lambda v: f(v)))(v0))
            row[f"tau={tau}"] = {
                "AD": g.tolist(),
                "err_ab_rel": float(np.max(np.abs(g[:, :2] - fd[:, :2])))
                / sc,
                "d_dcx_abs": float(np.max(np.abs(g[:, 2]))),
                "fd_dcx_abs": float(np.max(np.abs(fd[:, 2])))}
        finally:
            JT._E3_EIG_TAU_REL = None
    out["rows"].append(row)
    print(f"delta {dl:.0e} gap {row['min_rel_gap']:.1e} "
          + " | ".join(f"{t}: {row[f'tau={t}']['err_ab_rel']:.1e}/"
                       f"{row[f'tau={t}']['d_dcx_abs']:.1e}" for t in TAUS),
          flush=True)
print(dump(f"v6_degenerate_M{M}{PART}.json", out))
