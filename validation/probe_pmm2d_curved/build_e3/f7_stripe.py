"""E3-7 A 1-D SANITY: the twin's gradients on a y-UNIFORM stripe against an
INDEPENDENT 1-D solver.

    python f7_stripe.py M

The stripe Rect(1.2 - w / 2, 0.6, w, 1.2) spans the whole period in y and
ends on the cell edge (w = 0.6, eps 4, depth 0.4, n 1.0 / 1.45, wl 1.0): a
1-D lamellar grating on a 2 x 2 grid (its LEFT wall moves with w).  Incident
E_x (perpendicular to the ridges) is the 1-D TM case, E_y the TE case.
  * d R00 / d eps and d R00 / d depth: the twin's AD against the AD of the
    1-D PMM's JAX twin (pmm_efficiency_1d with JAX inputs, degree 40) --
    two independent differentiable solvers;
  * d R00 / d w: neither 1-D twin traces a wall (the duty cycle is a static
    argument of both), so the twin's AD is compared with the converged
    central difference of the NumPy 1-D PMM (degree 40, Richardson on the
    h / P = 3e-4 / 1e-4 rungs) -- an independent spectral solver.
The 2-D discretisation level is recorded alongside: the twin's forward R00
against the 1-D PMM's.
Output f7_stripe_M<M>.json.
"""
import sys

from _e3common import WL, P, PMM2DStackPure, dump, jax, jnp, np  # noqa

from lumenairy.elements.pmm import Rect, pmm_efficiency_1d

M = int(sys.argv[1])
W0, EPS0, D0 = 0.6, 4.0, 0.4
CX = 1.2 - W0 / 2      # the ridge [0.6, 1.2]: its right wall IS the cell edge
out = {"M": M}

st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=M,
                    n_orders=2, backend="jax")
st.add_layer(D0, shapes=[Rect(CX, 0.6, W0, P, EPS0)], background_eps=1.0)
st.set_source(WL)
tw = st.jax_twin()
p0 = tw.p0


def f2(v):
    w, e, d = v[0], v[1], v[2]
    p = tw.params()
    p["layers"][0]["shapes"] = [Rect(1.2 - w / 2, 0.6, w, P, e)]
    p["layers"][0]["thickness"] = d
    _o, R, T, J = st.solve(params=p)
    return jnp.stack([R[0, p0], R[1, p0]])        # E_x (TM), E_y (TE)


x0 = np.array([W0, EPS0, D0])
v2 = np.asarray(jax.jit(f2)(x0))
g2 = np.asarray(jax.jit(jax.jacrev(f2))(x0))      # (2 pol, 3 params)


def r1(pol, n_ridge, depth, duty):
    o, R, T = pmm_efficiency_1d(P, n_ridge, 1.0, 1.45, 1.0, depth, duty, WL,
                                polarization=pol, degree=40, stabilize=False)
    o = np.asarray(o)
    return R[int(np.where(o == 0)[0][0])]


out["value_2d"] = v2.tolist()
out["AD_2d"] = g2.tolist()
rows = {}
for k, pol in enumerate(("tm", "te")):
    v1 = float(r1(pol, np.sqrt(EPS0), D0, W0 / P))

    def g_eps(e):
        return r1(pol, jnp.sqrt(e), D0, W0 / P)

    def g_dep(d):
        return r1(pol, np.sqrt(EPS0), d, W0 / P)
    a_eps = float(jax.grad(g_eps)(jnp.asarray(EPS0 + 0j).real))
    a_dep = float(jax.grad(g_dep)(jnp.asarray(D0)))
    fd = []
    for hs in (3e-4, 1e-4):
        h = hs * P
        fd.append((r1(pol, np.sqrt(EPS0), D0, (W0 + h) / P)
                   - r1(pol, np.sqrt(EPS0), D0, (W0 - h) / P)) / (2 * h))
    fd_w = (9 * fd[1] - fd[0]) / 8
    rows[pol] = {"value_1d": v1, "AD_eps_1d": a_eps, "AD_depth_1d": a_dep,
                 "FD_w_1d": float(fd_w),
                 "value_2d": float(v2[k]),
                 "AD_w_2d": float(g2[k, 0]), "AD_eps_2d": float(g2[k, 1]),
                 "AD_depth_2d": float(g2[k, 2]),
                 "rel_w": abs(g2[k, 0] - fd_w) / abs(fd_w),
                 "rel_eps": abs(g2[k, 1] - a_eps) / abs(a_eps),
                 "rel_depth": abs(g2[k, 2] - a_dep) / abs(a_dep),
                 "value_diff": abs(v2[k] - v1)}
    print(pol, rows[pol], flush=True)
out["pol"] = rows
dump(f"f7_stripe_M{M}.json", out)
