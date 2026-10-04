"""D1: BOR-PMM (FD staggered basis) JAX twin -- ``BORStack.solve()`` on a
traced layer -> ``bor._jax_bor._jax_bor_stack_solve``.  Is there an exactly
degenerate eig cluster at a natural symmetric point, and is AD exact there?

    python d1_bor.py

Configuration: BORStack(Rbig 3, m in {0, 1}, N 56, n_sup = n_sub = 1.4142),
k0 = 2, one layer (thk 0.5) rings=(period 0.8, duty 0.5, n_r, n_g = 2.449).
Reference n_r = 2.449 => a radially HOMOGENEOUS layer (the most symmetric
point a BOR layer has; the azimuthal symmetry is already reduced to the fixed
m).  Parameter: n_r (breaks the radial homogeneity).  Control: thickness.
At m = 0 the TE (E_phi) and TM (E_r, E_z) families decouple for EVERY
isotropic eps, so no admissible parameter can couple them.

1. spectrum of every eig call (sup, layer) at the reference and m = 0, 1.
2. AD (jit(jacrev)) vs a Richardson central FD of the twin's own concrete
   forward: sum R, sum T, R and T of the first two propagating incident
   modes.  Also the NumPy-forward parity of sum R / sum T.
"""
import warnings

from _h import dump, jax, jnp, ladder, min_rel_gap, n_pairs_below, np, rel

import lumenairy.elements.rcwa as RCpkg
from lumenairy.elements.bor.bor_stack import BORStack

WL, NG, THK = 2 * np.pi / 2.0, 2.449, 0.5
_orig = RCpkg._jax_eig_stable
out = {"config": dict(Rbig=3.0, N=56, n_hs=1.4142, k0=2.0, ring=(0.8, 0.5),
                      n_g=NG, thk=THK), "m": {}}


def solve(m, nr, thk=THK):
    s = BORStack(3.0, m, N=56, n_superstrate=1.4142, n_substrate=1.4142)
    s.add_layer(thk, rings=(0.8, 0.5, nr, NG))
    s.set_source(wavelength=WL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return s.solve()


def spectra(m, nr):
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
        r = solve(m, jnp.asarray(nr))
    finally:
        RCpkg._jax_eig_stable = _orig
    return r, got


for m in (0, 1):
    r0, got = spectra(m, NG)
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

    def y(nr, thk=THK, m=m, inc=inc):
        r = solve(m, nr, thk)
        R, T = jnp.asarray(r["R"]), jnp.asarray(r["T"])
        return jnp.concatenate([jnp.stack([jnp.sum(R), jnp.sum(T)]),
                                R[inc], T[inc]])

    rn = solve(m, NG)
    rec["numpy_parity_sumR_sumT"] = [
        abs(float(np.sum(rn["R"])) - float(np.sum(np.asarray(r0["R"])))),
        abs(float(np.sum(rn["T"])) - float(np.sum(np.asarray(r0["T"]))))]
    for pname, fun, x0 in (
            ("n_r", lambda x, y=y: y(x), NG),
            ("thk_control", lambda x, y=y: y(jnp.asarray(NG), x), THK)):
        g = np.asarray(jax.jit(jax.jacrev(fun))(x0))
        _rows, fd, rat = ladder(lambda x, fun=fun: np.asarray(
            fun(jnp.asarray(x))), x0)
        rec[pname] = {"AD": g.tolist(), "FD": fd.tolist(),
                      "premise": np.asarray(rat).ravel().round(3).tolist(),
                      "rel_err": rel(g, fd)}
        print("m", m, pname, "rel %.2e" % rec[pname]["rel_err"], "premise",
              rec[pname]["premise"][:6], flush=True)
    out["m"][str(m)] = rec
print(dump("d1_bor.json", out))
