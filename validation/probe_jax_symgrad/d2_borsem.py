"""D2: BOR-SEM basis JAX twin -- ``BORStack(basis='sem').solve()`` on a
traced segment -> ``bor._jax_sem._jax_sem_stack_solve``.  Is there an exactly
degenerate eig cluster at a natural symmetric point, and is AD exact there?

    python d2_borsem.py

Configuration: BORStack(Rbig 3, m in {0, 1}, n_sup = n_sub = 1.4142,
basis 'sem', degree 6, n_mesh_cap 2.6 (pins the mesh for AD and FD)), k0 = 2,
one layer (thk 0.5) segments=[(3.0, (6 + d, 6 - d, 6))].  Reference d = 0 =>
an ISOTROPIC homogeneous layer.  Parameter: the in-plane anisotropy d
(eps_rr - eps_phiphi = 2 d, zero at the reference; the LC-director knob).
Control: the isotropic eps x (all three components).

1. spectrum of every eig call (sup, layer) at the reference, m = 0, 1.
2. AD (jit(jacrev)) vs a Richardson central FD of the twin's own concrete
   forward: sum R, sum T, R and T of the first two propagating incident
   modes.  Also the NumPy-forward parity of sum R / sum T.
"""
import warnings

from _h import dump, jax, jnp, ladder, min_rel_gap, n_pairs_below, np, rel

import lumenairy.elements.rcwa as RCpkg
from lumenairy.elements.bor.bor_stack import BORStack

WL, E0, THK = 2 * np.pi / 2.0, 6.0, 0.5
_orig = RCpkg._jax_eig_stable
out = {"config": dict(Rbig=3.0, degree=6, n_mesh_cap=2.6, n_hs=1.4142,
                      k0=2.0, eps=E0, thk=THK), "m": {}}


def solve(m, tri):
    s = BORStack(3.0, m, n_superstrate=1.4142, n_substrate=1.4142,
                 basis="sem", degree=6, n_mesh_cap=2.6)
    s.add_layer(THK, segments=[(3.0, tri)])
    s.set_source(wavelength=WL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return s.solve()


def tri_aniso(d):
    d = jnp.asarray(d).astype(jnp.complex128)
    return jnp.stack([E0 + d, E0 - d, E0 + 0.0 * d])


def tri_iso(x):
    x = jnp.asarray(x).astype(jnp.complex128)
    return jnp.stack([x, x, x])


def spectra(m):
    got = []

    def factory():
        e = _orig()

        def eig(A, tau_rel=1e-12):
            lam, V = e(A, tau_rel)
            if not isinstance(lam, jax.core.Tracer):
                got.append(np.asarray(lam))
            return lam, V
        return eig
    RCpkg._jax_eig_stable = factory
    try:
        r = solve(m, tri_aniso(0.0))
    finally:
        RCpkg._jax_eig_stable = _orig
    return r, got


for m in (0, 1):
    r0, got = spectra(m)
    rec = {"spectrum": {}}
    for site, lam in zip(("sup", "layer"), got):
        rec["spectrum"][site] = {
            "n": int(lam.size), "max_abs": float(np.max(np.abs(lam))),
            "min_rel_gap": min_rel_gap(lam),
            "members_below_1e-12": n_pairs_below(lam, 1e-12),
            "members_below_1e-8": n_pairs_below(lam, 1e-8)}
    print("m", m, "spectrum", rec["spectrum"], flush=True)
    inc = np.nonzero(np.asarray(r0["inc_mask"]) > 0.5)[0][:2]
    rec["incident_idx"] = inc.tolist()

    def y(tri, m=m, inc=inc):
        r = solve(m, tri)
        R, T = jnp.asarray(r["R"]), jnp.asarray(r["T"])
        return jnp.concatenate([jnp.stack([jnp.sum(R), jnp.sum(T)]),
                                R[inc], T[inc]])

    rn = solve(m, (E0, E0, E0))
    rec["numpy_parity_sumR_sumT"] = [
        abs(float(np.sum(rn["R"])) - float(np.sum(np.asarray(r0["R"])))),
        abs(float(np.sum(rn["T"])) - float(np.sum(np.asarray(r0["T"]))))]
    print("m", m, "numpy parity", rec["numpy_parity_sumR_sumT"], flush=True)
    for pname, fun, x0 in (
            ("aniso_d", lambda x, y=y: y(tri_aniso(x)), 0.0),
            ("iso_eps_control", lambda x, y=y: y(tri_iso(x)), E0)):
        g = np.asarray(jax.jit(jax.jacrev(fun))(x0))
        _rows, fd, rat = ladder(lambda x, fun=fun: np.asarray(
            fun(jnp.asarray(x))), x0)
        rec[pname] = {"AD": g.tolist(), "FD": fd.tolist(),
                      "premise": np.asarray(rat).ravel().round(3).tolist(),
                      "rel_err": rel(g, fd)}
        print("m", m, pname, "rel %.2e" % rec[pname]["rel_err"], "premise",
              rec[pname]["premise"][:6], flush=True)
    out["m"][str(m)] = rec
print(dump("d2_borsem.json", out))
