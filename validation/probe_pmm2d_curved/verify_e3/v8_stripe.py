"""V8 (E3-7 and F-E3-5): the y-uniform stripe.

    python v8_stripe.py ladder      -- NumPy 2-D R00 (TM = E_x, TE = E_y) at
                                       M = 3 .. 10 on: the builder's stripe
                                       (ridge [0.6, 1.2], cells centred on
                                       the mirror planes) at normal, oblique
                                       in x, oblique in y; and an OFF-CENTRE
                                       split (ridge [0.5, 1.1], the gap cut
                                       by the seam) at normal; vs the 1-D PMM
    python v8_stripe.py grad M      -- d R00 / d eps (and / d depth): the 2-D
                                       NumPy solver's OWN FD at M vs the 1-D
                                       PMM's JAX AD; the twin's AD vs the 2-D
                                       NumPy FD at the same M
"""
import sys

from _ve3 import WL, P, PMM2DStackPure, dump, jax, jnp, ladder, np

from lumenairy.elements.pmm import Rect, pmm_efficiency_1d

MODE = sys.argv[1]
EPS0, D0 = 4.0, 0.4


def r1(pol, eps, depth, duty, theta=0.0):
    o, R, T = pmm_efficiency_1d(P, jnp.sqrt(eps) if hasattr(eps, "aval")
                                else np.sqrt(eps), 1.0, 1.45, 1.0, depth,
                                duty, WL, polarization=pol, degree=40,
                                stabilize=False, theta=theta)
    o = np.asarray(o)
    return R[int(np.where(o == 0)[0][0])]


def stack2(M, cx, w, theta=0.0, phi=0.0, eps=EPS0, depth=D0,
           backend="numpy"):
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=2, backend=backend)
    st.add_layer(depth, shapes=[Rect(cx, 0.6, w, P, eps)],
                 background_eps=1.0)
    st.set_source(WL, theta=theta, phi=phi)
    return st


def r2(M, cx, w, **kw):
    o, R, T, J = stack2(M, cx, w, **kw).solve()
    p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    return np.array([R[0, p0], R[1, p0]])


out = {"mode": MODE}
if MODE == "ladder":
    cases = {
        "centred_normal": dict(cx=0.9, w=0.6),
        "centred_oblique_x": dict(cx=0.9, w=0.6, theta=0.2, phi=0.0),
        "centred_oblique_y": dict(cx=0.9, w=0.6, theta=0.2,
                                  phi=np.pi / 2),
        "offcentre_normal": dict(cx=0.8, w=0.6),
    }
    for name, kw in cases.items():
        vals = {}
        for M in range(3, 11):
            vals[M] = r2(M, **kw).tolist()
        Ms = sorted(vals)
        diffs = {f"{a}->{b}": np.abs(np.asarray(vals[b])
                                     - np.asarray(vals[a])).tolist()
                 for a, b in zip(Ms[:-1], Ms[1:])}
        rec = {"values": vals, "consecutive_diffs": diffs}
        if name.endswith("normal"):
            duty = kw["w"] / P
            rec["one_d"] = [float(r1("tm", EPS0, D0, duty)),
                            float(r1("te", EPS0, D0, duty))]
        out[name] = rec
        print(name, {k: ["%.1e" % x for x in v] for k, v in diffs.items()},
              flush=True)
elif MODE == "grad":
    M = int(sys.argv[2])
    out["M"] = M
    st = stack2(M, 0.9, 0.6, backend="jax")
    tw = st.jax_twin()

    def f2(v):
        p = tw.params()
        p["layers"][0]["shapes"] = [Rect(0.9, 0.6, 0.6, P, v[0])]
        p["layers"][0]["thickness"] = v[1]
        _o, R, T, J = st.solve(params=p)
        return jnp.stack([R[0, tw.p0], R[1, tw.p0]])
    v0 = np.array([EPS0, D0])
    g = np.asarray(jax.jit(jax.jacrev(f2))(v0))     # (pol, param)
    fdn = []
    for k in range(2):
        e = np.zeros(2)
        e[k] = 1.0
        _r, d, _c, rat = ladder(
            lambda t: r2(M, 0.9, 0.6, eps=v0[0] + t * e[0],
                         depth=v0[1] + t * e[1]), 0.0, [1e-3, 3e-4, 1e-4])
        fdn.append(d)
    fdn = np.stack(fdn, axis=1)
    one = np.zeros((2, 2))
    for i, pol in enumerate(("tm", "te")):
        one[i, 0] = float(jax.grad(lambda e: r1(pol, e, D0, 0.5))(
            jnp.asarray(EPS0)))
        one[i, 1] = float(jax.grad(lambda d: r1(pol, EPS0, d, 0.5))(
            jnp.asarray(D0)))
    out.update(AD_twin=g.tolist(), FD_numpy2d=fdn.tolist(), AD_1d=one.tolist(),
               twin_vs_numpy2d_rel=(np.abs(g - fdn) / np.abs(fdn)).tolist(),
               numpy2d_vs_1d_rel=(np.abs(fdn - one) / np.abs(one)).tolist(),
               twin_vs_1d_rel=(np.abs(g - one) / np.abs(one)).tolist())
    print(out, flush=True)
print(dump(f"v8_stripe_{MODE}{'_M' + sys.argv[2] if MODE == 'grad' else ''}"
           ".json", out))
