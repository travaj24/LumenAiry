"""V3: NEAR-degenerate clusters that are NOT (only) symmetry-forced, and
clusters with more than two members, on weakly modulated lossless cells
(eps 2.25 + delta x pattern; holographic-grating contrast):

  c1_conical_1e-2 : no symmetry (fixed random pattern), theta 0.2 phi 0.3,
                    delta 1e-2 -> 23 pairs inside gap_rel, NONE exact,
                    7 of them propagating (lam^2 < 0, on the sqrt cut)
  c1_normal_1e-3  : no symmetry, normal incidence, delta 1e-3 -> clusters
                    of 2, 2, 3 members, none exact, 3 propagating
  c4v_1e-2        : C4v post, normal, delta 1e-2 -> exact pairs + one
                    3-member cluster merging distinct eigenvalues
  c4v_1e-4        : C4v post, normal, delta 1e-4 -> clusters of up to 8
                    members (exact pairs chained with distinct eigenvalues)

For each: AD (rule on / off) vs a premise-checked NumPy FD, for a C1 pixel
direction and (C4v cells) the x-widening direction; the error is reported
over all outputs and over the DIFFRACTED orders alone (whose efficiencies
are O(delta^2) and would hide inside a max over the zero order).

MECHANISM: the lift the rule applies (_eig_cluster_lift) at the exact
point, the eigenvalues at A + t dN (t = +-1, +-2), and whether any lifted
member's modal root sqrt_decay(lam^2) changed BRANCH (|r_t + r| < |r_t - r|)
-- the failure mode the build record measured for the folded 1-D operator.
"""
import sys

from _fix import DIRS, RAND, post, rcwa_f
from _v import dump, fd_halving, jax, jnp, np

import lumenairy.elements.rcwa._core as RC
import lumenairy.elements.rcwa.twod as RT
from lumenairy.elements.rcwa import rcwa_efficiency_2d

GAP0 = RC._EIG_CLUSTER_GAP_REL
FIX = {
    "c1_conical_1e-2": (2.25 + 1e-2 * RAND, 0.2, 0.3),
    "c1_normal_1e-3": (2.25 + 1e-3 * RAND, 0.0, 0.0),
    "c4v_1e-2": (2.25 + 1e-2 * post, 0.0, 0.0),
    "c4v_1e-4": (2.25 + 1e-4 * post, 0.0, 0.0),
}


def mechanism(cell, theta, phi, pol):
    rec_calls = []
    orig = RT._jax_eig_cluster_adjoint

    def spy(eig_fn, problems, consumer, **kw):
        rec_calls.append((eig_fn, problems, kw))
        return orig(eig_fn, problems, consumer, **kw)
    RT._jax_eig_cluster_adjoint = spy
    try:
        rcwa_efficiency_2d(0.9, 0.9, jnp.asarray(cell.astype(complex)), 1.52,
                           1.0, 0.37, 1.0, theta=theta, phi=phi,
                           polarization=pol, n_orders_x=2, n_orders_y=2)
    finally:
        RT._jax_eig_cluster_adjoint = orig
    eig_fn, problems, kw = rec_calls[-1]
    (A, G), = problems
    lam, V = eig_fn(A, G)
    dN, anyc = RC._eig_cluster_lift(lam, V, G, GAP0, RC._EIG_CLUSTER_SPLIT_REL,
                                    kw["anchors"][0])
    lam = np.asarray(lam)
    s = np.max(np.abs(lam))
    d = RC._EIG_CLUSTER_SPLIT_REL * s
    r0 = np.asarray(RC._sqrt_decay(jnp.asarray(lam), xp=jnp))
    close = np.abs(lam[:, None] - lam[None, :]) <= GAP0 * s
    np.fill_diagonal(close, False)
    members = np.nonzero(np.any(close, axis=1))[0]
    jumps, im_ratio, re_ratio = 0, 0.0, 0.0
    for t in (1.0, -1.0, 2.0, -2.0):
        lt, _ = eig_fn(A + t * dN, G)
        lt = np.asarray(lt)
        rt = np.asarray(RC._sqrt_decay(jnp.asarray(lt), xp=jnp))
        for i in members:
            j = int(np.argmin(np.abs(lt - lam[i])))
            sh = lt[j] - lam[i]
            im_ratio = max(im_ratio, abs(sh.imag) / d)
            re_ratio = max(re_ratio, abs(sh.real) / d)
            if abs(rt[j] + r0[i]) < abs(rt[j] - r0[i]):
                jumps += 1
    prop = int(np.sum(lam[members].real < 0))
    return {"any_cluster": bool(anyc), "members": int(members.size),
            "propagating_members": prop,
            "max_member_spread_rel": float(max(
                (np.max(np.abs(lam[i] - lam[np.nonzero(close[i])[0]])) / s
                 for i in members), default=0.0)),
            "lift_shift_im_over_d": im_ratio,
            "lift_shift_re_over_d": re_ratio, "branch_jumps": jumps}


out = {}
only = set(sys.argv[1:])
for name, (cell, th, ph) in FIX.items():
    dirs = ["corner"] + (["x", "keep"] if name.startswith("c4v") else [])
    o = np.asarray(rcwa_efficiency_2d(0.9, 0.9, cell.astype(complex), 1.52,
                                      1.0, 0.37, 1.0, theta=th, phi=ph,
                                      n_orders_x=2, n_orders_y=2)[0])
    zero = int(np.nonzero((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    nz = np.ones(2 * len(o), bool)
    nz[[zero, len(o) + zero]] = False
    for pol in ("te", "tm"):
        mech = mechanism(cell, th, ph, pol)
        print(name, pol, "MECH", mech, flush=True)
        out[f"{name}_{pol}_mechanism"] = mech
        for dname in dirs:
            key = f"{name}_{pol}_{dname}"
            if only and key not in only:
                continue
            D = {"keep": post}.get(dname, DIRS[dname])
            f = rcwa_f(D, pol, n_orders=2, theta=th, phi=ph, base=cell)
            fd, prem = fd_halving(lambda t: f(t, np), 0.0, h0=2e-2)
            rec = {"premise": prem, "fd_max": float(np.max(np.abs(fd))),
                   "fd_max_diffracted": float(np.max(np.abs(fd[nz])))}
            for rule, gap in (("on", GAP0), ("off", 0.0)):
                RC._EIG_CLUSTER_GAP_REL = gap
                try:
                    g = np.asarray(jax.jit(jax.jacrev(
                        lambda t: f(t, jnp)))(0.0))
                finally:
                    RC._EIG_CLUSTER_GAP_REL = GAP0
                rec[f"rel_{rule}"] = float(np.max(np.abs(g - fd))
                                           / np.max(np.abs(fd)))
                rec[f"rel_diffracted_{rule}"] = float(
                    np.max(np.abs(g[nz] - fd[nz])) / np.max(np.abs(fd[nz])))
            out[key] = rec
            print(key, {k: (f"{v:.3g}" if isinstance(v, float) else v)
                        for k, v in rec.items()}, flush=True)
dump("v3_neardeg", out)
