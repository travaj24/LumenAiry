"""V4 (E3-3): gradients on the verifier's fixtures.

    python v4_grads.py film M      -- LC director angle: the Jones gradient of
                                      a homogeneous LC FILM through three twin
                                      routes vs the Berreman oracle (its JAX
                                      AD and its Richardson FD), normal and
                                      conical incidence
    python v4_grads.py second M    -- d2/dx2 by nested grad vs FD of the
                                      gradient: rectangle width, circle radius
                                      (and jax.hessian, forward-over-reverse)
    python v4_grads.py jac M       -- jacobian of (R00, T00) w.r.t. (r, Re eps)
                                      vs FD(twin) and FD(numpy)
    python v4_grads.py ladder M    -- FD ladders with the h^2 premise: rect
                                      width at conical, Im eps, n_substrate,
                                      ellipse semi-axis, two-layer depth
"""
import sys

from _ve3 import WL, P, PMM2DStackPure, dump, jax, jnp, ladder, lc, np, tic

from lumenairy.elements.berreman import berreman_jones_1d
from lumenairy.elements.pmm import Circle, Ellipse, Rect

MODE, M = sys.argv[1], int(sys.argv[2])
out = {"mode": MODE, "M": M}
STEPS = [1e-3, 3e-4, 1e-4]


def stack(layers, theta=0.0, phi=0.0, backend="jax", n_sub=1.45):
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=n_sub,
                        n_modes=M, n_orders=2, backend=backend)
    for L in layers:
        st.add_layer(**L)
    st.set_source(WL, theta=theta, phi=phi)
    return st


def premise(rat, r1=1e-3 / 3e-4, r2=3.0):
    """h^2 law for the ladder 1e-3, 3e-4, 1e-4: the ratio of successive
    rung changes is (r1^2 - 1) r2^2 / (r2^2 - 1) = 11.375 exactly."""
    want = (r1 * r1 - 1.0) * r2 * r2 / (r2 * r2 - 1.0)
    rat = np.asarray(rat, dtype=float).ravel()
    return float(np.max(np.abs(rat / want - 1.0))) if rat.size else None


if MODE == "film":
    D = 0.37
    res = {}
    for tag, (th, ph) in (("normal", (0.0, 0.0)), ("conical", (0.3, 0.4))):
        def ber(a, th=th, ph=ph):
            R, T, jr, jt = berreman_jones_1d([(lc(a), D)], 1.45, 1.0, WL,
                                             angle=th, phi=ph)
            return jnp.stack([jnp.real(jr).ravel(), jnp.imag(jr).ravel()])
        a0 = 0.55
        jb = np.asarray(ber(jnp.asarray(a0)))
        gb = np.asarray(jax.jacrev(ber)(jnp.asarray(a0)))
        _r, fdb, _c, ratb = ladder(lambda a: np.asarray(ber(jnp.asarray(a))),
                                   a0, STEPS)
        routes = {
            "uniform_tensor": ([dict(thickness=D, eps=lc(a0))],
                               lambda p, a: p["layers"][0].update(eps=lc(a))),
            "eps_cell": ([dict(thickness=D, eps_cell=np.broadcast_to(
                lc(a0), (2, 2, 3, 3)).copy())],
                lambda p, a: p["layers"][0].update(eps=jnp.broadcast_to(
                    lc(a), (2, 2, 3, 3)))),
            "circle_mapped": ([dict(thickness=D, shapes=[Circle(
                0.6, 0.6, 0.36, lc(a0))], background_eps=lc(a0))],
                lambda p, a: p["layers"][0].update(
                    shapes=[Circle(0.6, 0.6, 0.36, lc(a))],
                    background_eps=lc(a))),
        }
        rr = {"berreman_jones": jb.tolist(), "berreman_AD": gb.tolist(),
              "berreman_FD": fdb.tolist(),
              "berreman_AD_vs_FD": float(np.max(np.abs(gb - fdb)))}
        for name, (layers, setp) in routes.items():
            t = tic()
            st = stack(layers, th, ph)
            tw = st.jax_twin()

            def f(a, tw=tw, setp=setp, st=st):
                p = tw.params()
                setp(p, a)
                _o, R, T, J = st.solve(params=p)
                return jnp.stack([jnp.real(J).ravel(), jnp.imag(J).ravel()])
            fj = jax.jit(f)
            jt = np.asarray(fj(a0))
            g = np.asarray(jax.jit(jax.jacrev(f))(a0))
            rr[name] = {
                "jones_vs_berreman": float(np.max(np.abs(jt - jb))),
                "AD_vs_berreman_AD": float(np.max(np.abs(g - gb))),
                "AD_vs_berreman_AD_rel": float(np.max(np.abs(g - gb))
                                               / np.max(np.abs(gb))),
                "AD": g.tolist(), "s": tic() - t}
            print(tag, name, rr[name]["jones_vs_berreman"],
                  rr[name]["AD_vs_berreman_AD_rel"], flush=True)
        res[tag] = rr
    out["film"] = res

elif MODE == "second":
    res = {}
    cases = {
        "rect_w": (0.5, lambda x: [Rect(0.6, 0.6, x, 0.4, 4.0)], 0.4),
        "circle_r": (0.34, lambda x: [Circle(0.6, 0.6, x, 3.2)], 0.45),
    }
    for name, (x0, shp, dep) in cases.items():
        st = stack([dict(thickness=dep, shapes=shp(x0), background_eps=1.0)])
        tw = st.jax_twin()

        def f(x, tw=tw, st=st, shp=shp):
            p = tw.params()
            p["layers"][0]["shapes"] = shp(x)
            _o, R, T, J = st.solve(params=p)
            return T[0, tw.p0]
        g1 = jax.jit(jax.grad(f))
        r = {}
        try:
            t = tic()
            g2 = float(jax.jit(jax.grad(jax.grad(f)))(x0))
            r["nested_grad"] = g2
            r["nested_s"] = tic() - t
            _rows, fd, _c, rat = ladder(lambda x: float(g1(x)), x0, STEPS, P)
            r["FD_of_grad"] = float(fd)
            r["premise_dev"] = premise(rat)
            r["rel"] = abs(g2 - float(fd)) / abs(float(fd))
        except Exception as exc:  # noqa: BLE001 -- probe records the outcome
            r["nested_grad_error"] = repr(exc)[:300]
        try:
            h = float(jax.hessian(f)(x0))
            r["hessian_fwd_over_rev"] = h
        except Exception as exc:  # noqa: BLE001
            r["hessian_error"] = type(exc).__name__ + ": " + str(exc)[:200]
        try:
            r["jvp"] = float(jax.jvp(f, (x0,), (1.0,))[1])
        except Exception as exc:  # noqa: BLE001
            r["jvp_error"] = type(exc).__name__ + ": " + str(exc)[:200]
        res[name] = r
        print(name, r, flush=True)
    out["second"] = res

elif MODE == "jac":
    x0 = np.array([0.34, 3.2])
    st = stack([dict(thickness=0.45, shapes=[Circle(0.6, 0.6, 0.34, 3.2)],
                     background_eps=1.0)])
    tw = st.jax_twin()

    def f(v):
        p = tw.params()
        p["layers"][0]["shapes"] = [Circle(0.6, 0.6, v[0], v[1] + 0.0j)]
        _o, R, T, J = st.solve(params=p)
        return jnp.stack([R[0, tw.p0], T[0, tw.p0]])
    Jad = np.asarray(jax.jit(jax.jacrev(f))(jnp.asarray(x0)))
    Jfw = None
    try:
        Jfw = np.asarray(jax.jacfwd(f)(jnp.asarray(x0)))
    except Exception as exc:  # noqa: BLE001
        out["jacfwd_error"] = type(exc).__name__ + ": " + str(exc)[:200]
    fj = jax.jit(f)
    cols, colsn, prem = [], [], []

    def fnp(v):
        s = stack([dict(thickness=0.45, shapes=[Circle(0.6, 0.6, v[0],
                                                       v[1])],
                        background_eps=1.0)], backend="numpy")
        o, R, T, J = s.solve()
        return np.array([R[0, 12], T[0, 12]])
    for k, sc in ((0, P), (1, 1.0)):
        e = np.zeros(2)
        e[k] = 1.0
        _r, fd, _c, rat = ladder(lambda s: np.asarray(fj(jnp.asarray(
            x0 + s * e))), 0.0, STEPS, sc)
        _r, fdn, _c, ratn = ladder(lambda s: fnp(x0 + s * e), 0.0, STEPS, sc)
        cols.append(fd)
        colsn.append(fdn)
        prem.append([premise(rat), premise(ratn)])
    Jfd = np.stack(cols, axis=1)
    Jfdn = np.stack(colsn, axis=1)
    out.update(AD=Jad.tolist(), FD_twin=Jfd.tolist(), FD_numpy=Jfdn.tolist(),
               premise_dev=prem,
               AD_vs_FDtwin_rel=(np.abs(Jad - Jfd) / np.abs(Jfd)).tolist(),
               AD_vs_FDnumpy_rel=(np.abs(Jad - Jfdn) / np.abs(Jfdn)).tolist())
    if Jfw is not None:
        out["jacfwd_vs_jacrev"] = float(np.max(np.abs(Jfw - Jad)))
    print(out, flush=True)

elif MODE == "ladder":
    res = {}
    cases = {
        # name: (x0, layers(x), setp(p, x), scale, src, numpy-comparable)
        "rect_w_conical": (0.47, lambda x: [dict(
            thickness=0.4, shapes=[Rect(0.6, 0.55, x, 0.42, 3.6)],
            background_eps=1.0)], lambda p, x: p["layers"][0].update(
                shapes=[Rect(0.6, 0.55, x, 0.42, 3.6)]), P,
            dict(theta=0.3, phi=0.45)),
        "im_eps_circle": (0.15, lambda x: [dict(
            thickness=0.45, shapes=[Circle(0.6, 0.6, 0.34, 3.2 + 1j * x)],
            background_eps=1.0)], lambda p, x: p["layers"][0].update(
                shapes=[Circle(0.6, 0.6, 0.34, 3.2 + 1j * x)]), 1.0, {}),
        "n_substrate": (1.45, None, lambda p, x: p.update(n_substrate=x),
                        1.0, {}),
        "ellipse_a": (0.33, lambda x: [dict(
            thickness=0.4, shapes=[Ellipse(0.6, 0.6, x, 0.22, 3.0)],
            background_eps=1.0)], lambda p, x: p["layers"][0].update(
                shapes=[Ellipse(0.6, 0.6, x, 0.22, 3.0)]), P, {}),
        "two_layer_depth2": (0.2, lambda x: [
            dict(thickness=0.3, shapes=[Circle(0.6, 0.6, 0.3, 3.3)],
                 background_eps=1.0),
            dict(thickness=x, eps=lc(0.7))],
            lambda p, x: p["layers"][1].update(thickness=x), 1.0,
            dict(theta=0.2)),
    }
    for name, (x0, lays, setp, sc, src) in cases.items():
        L0 = lays(x0) if lays is not None else [dict(
            thickness=0.45, shapes=[Circle(0.6, 0.6, 0.34, 3.2)],
            background_eps=1.0)]
        st = stack(L0, **src)
        tw = st.jax_twin()

        def f(x, tw=tw, st=st, setp=setp):
            p = tw.params()
            setp(p, x)
            _o, R, T, J = st.solve(params=p)
            return jnp.stack([R[0, tw.p0], T[0, tw.p0], T[1, tw.p0]])
        g = np.asarray(jax.jit(jax.jacrev(f))(x0))
        _r, fdt, _c, ratt = ladder(jax.jit(f), x0, STEPS, sc)

        def fnp(x, lays=lays, src=src, name=name):
            if lays is None:
                s = stack(L0, backend="numpy", n_sub=x, **src)
            else:
                s = stack(lays(x), backend="numpy", **src)
            o, R, T, J = s.solve()
            return np.array([R[0, 12], T[0, 12], T[1, 12]])
        _r, fdn, _c, ratn = ladder(fnp, x0, STEPS, sc)
        den = np.maximum(np.abs(fdn), 1e-3 * np.max(np.abs(fdn)))
        res[name] = {"AD": g.tolist(), "FD_twin": fdt.tolist(),
                     "FD_numpy": fdn.tolist(),
                     "premise_dev_twin": premise(ratt),
                     "premise_dev_numpy": premise(ratn),
                     "AD_vs_FDtwin": float(np.max(np.abs(g - fdt) / den)),
                     "AD_vs_FDnumpy": float(np.max(np.abs(g - fdn) / den))}
        print(name, {k: v for k, v in res[name].items()
                     if not isinstance(v, list)}, flush=True)
    out["ladder"] = res
print(dump(f"v4_{MODE}_M{M}.json", out))
