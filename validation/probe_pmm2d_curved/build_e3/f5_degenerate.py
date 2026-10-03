"""E3-5 DEGENERATE EIGENVALUES through rcwa._jax_eig_stable.

    python f5_degenerate.py M

The CENTRED circle at normal incidence is four-fold symmetric: its layer
pencil and the half-spaces' geometric pencil carry exactly degenerate pairs
(E-type Bloch modes).  Measured here:
  * the degeneracy census (pairs of eigenvalues closer than 1e-10 relative)
    of the layer and of the homogeneous geometric pencil;
  * d/dr (symmetry-PRESERVING: the perturbation is diagonal inside each
    degenerate pair) and d/dcx (symmetry-BREAKING: it splits the pairs) of
    R00 / T00, AD vs FD(twin), for the eigenvector-VJP regularisation
    tau_rel in {1e-14, 1e-12 (default), 1e-10, 1e-8, 1e-6};  at the centred
    circle d/dcx is ZERO by the x-mirror (an even function of the shift),
    which the FD reproduces to its round-off;
  * the same at an OFF-CENTRE circle (cx = 0.55, no degeneracy of that
    kind): AD vs FD;
  * the mutation arm: _jax_eig_stable replaced by the PLAIN jnp.linalg.eig
    (JAX's own eig derivative, unregularised 1 / (lam_j - lam_i)).
Output f5_degenerate_M<M>.json.
"""
import sys

from _e3common import TS, WL, P, PMM2DStackPure, dump, jax, jnp, np  # noqa

import lumenairy.elements.pmm._jax_twod_staggered as JT
from lumenairy.elements.pmm import Circle

M = int(sys.argv[1])
out = {"M": M}


def mk(cx):
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=M,
                        n_orders=2, backend="jax")
    st.add_layer(0.5, shapes=[Circle(cx, 0.6, 0.36, 4.0)],
                 background_eps=1.0)
    st.set_source(WL)
    return st


def census(vals):
    v = np.sort_complex(np.asarray(vals))
    sc = np.max(np.abs(v))
    d = np.abs(np.diff(v)) / sc
    return int(np.sum(d < 1e-10)), float(np.min(d))


st = mk(0.6)
tw = st.jax_twin()
ref = [r for r in tw.layers if "ref" in r][0]["ref"]
import scipy.linalg as sla  # noqa: E402

g2l = sla.eig(ref.Lmat, -ref.Rmat, right=False)
out["layer_degenerate_pairs"], out["layer_min_gap_rel"] = census(g2l)
out["homog_degenerate_pairs"], out["homog_min_gap_rel"] = census(
    tw.geom_ref[1])
print("census", out["layer_degenerate_pairs"], out["homog_degenerate_pairs"],
      flush=True)


def runner(cx0):
    st = mk(cx0)
    tw = st.jax_twin()
    p0 = tw.p0

    def f(v):
        r, cx = v[0], v[1]
        p = tw.params()
        p["layers"][0]["shapes"] = [Circle(cx, 0.6, r, 4.0)]
        _o, R, T, J = st.solve(params=p)
        return jnp.stack([R[0, p0], T[0, p0]])
    return f


def measure(cx0, tag):
    f = runner(cx0)
    x0 = np.array([0.36, cx0])
    fj = jax.jit(f)
    fd = {}
    for k in (0, 1):
        rows = []
        for hs in (3e-4, 1e-4):
            e = np.zeros(2)
            e[k] = hs * P
            rows.append((np.asarray(fj(x0 + e)) - np.asarray(fj(x0 - e)))
                        / (2 * hs * P))
        fd[k] = ((9 * rows[1] - rows[0]) / 8, rows[1])
    res = {"FD_r": fd[0][0].tolist(), "FD_cx": fd[1][0].tolist(),
           "FD_cx_lastrung": fd[1][1].tolist()}
    for tau in (1e-14, None, 1e-10, 1e-8, 1e-6):
        JT._E3_EIG_TAU_REL = tau
        try:
            g = np.asarray(jax.jit(jax.jacrev(f))(x0))      # (2 out, 2 in)
        finally:
            JT._E3_EIG_TAU_REL = None
        res[f"tau_{tau}"] = {"AD_r": g[:, 0].tolist(),
                             "AD_cx": g[:, 1].tolist(),
                             "rel_r": (np.abs(g[:, 0] - fd[0][0])
                                       / np.abs(fd[0][0])).tolist(),
                             "abs_cx": np.abs(g[:, 1] - fd[1][0]).tolist()}
        print(tag, tau, res[f"tau_{tau}"], flush=True)
    # mutation: the plain JAX eig
    orig = JT._stag_geneig_jax

    def plain(L, G, tau_rel=None):
        return jnp.linalg.eig(jnp.linalg.solve(G, L))
    JT._stag_geneig_jax = plain
    try:
        jax.jit(jax.jacrev(f))(x0)
        res["mutation_plain_eig_raises"] = False
    except NotImplementedError as exc:
        res["mutation_plain_eig_raises"] = repr(exc)[:120]
    finally:
        JT._stag_geneig_jax = orig

    def plain(L, G, tau_rel=None):        # noqa: F811
        return jax.lax.linalg.eig(jnp.linalg.solve(G, L),
                                  compute_left_eigenvectors=False,
                                  enable_eigvec_derivs=True)
    JT._stag_geneig_jax = plain
    try:
        g = np.asarray(jax.jit(jax.jacrev(f))(x0))
        res["mutation_plain_eig"] = {
            "AD_r": g[:, 0].tolist(), "AD_cx": g[:, 1].tolist(),
            "finite": bool(np.all(np.isfinite(g))),
            "rel_r": (np.abs(g[:, 0] - fd[0][0])
                      / np.abs(fd[0][0])).tolist(),
            "abs_cx": np.abs(g[:, 1] - fd[1][0]).tolist()}
    except Exception as exc:                               # noqa: BLE001
        res["mutation_plain_eig"] = {"error": repr(exc)[:200]}
    finally:
        JT._stag_geneig_jax = orig
    print(tag, "plain", res["mutation_plain_eig"], flush=True)
    out[tag] = res


measure(0.6, "centred")
measure(0.55, "offcentre")
dump(f"f5_degenerate_M{M}.json", out)
