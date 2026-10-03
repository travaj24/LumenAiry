"""E3-4 NON-DIFFERENTIABLE EVENTS: refused (concrete) or NaN (traced), and
two-sided: a parameter path inside ONE topology is smooth.

    python f4_events.py M

Part A (smooth side): the twin frozen at r0 = 0.36, the circle radius
r in {0.30, 0.33, 0.36, 0.39, 0.42}: value, AD and the Richardson FD(twin)
at each point (h = 1e-4 P and 3e-4 P), plus the NumPy solve built at r (the
frozen-vs-moving-grid offset).
Part B (events), each through jax.jit (traced: NaN expected) and eagerly
with a CONCRETE value (refusal expected), with a control value just inside
the topology (finite / accepted):
  fold      -- the circle bulging past the cell edge (r = 0.62 vs 0.58);
  tangency  -- a circle (layer 1) and a rectangle (layer 2) whose x-walls
               coincide with the circle's 45-degree walls at the reference
               (one merged wall); moving r separates them (r = 0.37 vs the
               reference 0.36);
  sliver    -- a rectangle widening until its outer segment is below the
               sliver contract (w = 1.199 vs 1.19 in a 1.2 cell);
  duffy     -- a fillet radius shrinking to 0 (the four singular vertices
               vanish; r = 1e-5 traced / r = 0 concrete vs r = 0.05).
Output f4_events_M<M>.json.
"""
import sys

from _e3common import WL, P, PMM2DStackPure, absd, dump, jax, jnp, np  # noqa

from lumenairy.elements.pmm import Circle, FilletRect, Rect

M = int(sys.argv[1])
out = {"M": M}


def mk(layers, backend="jax"):
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=M,
                        n_orders=2, backend=backend)
    for t, shapes in layers:
        st.add_layer(t, shapes=shapes, background_eps=1.0)
    st.set_source(WL)
    return st


def runner(st, layers_of):
    tw = st.jax_twin()
    p0 = tw.p0

    def f(x):
        p = tw.params()
        for k, shp in layers_of(x).items():
            p["layers"][k]["shapes"] = shp
        _o, R, T, J = st.solve(params=p)
        return jnp.stack([R[0, p0], T[0, p0]])
    return tw, f


# ---- Part A: smooth inside one topology ---------------------------------
circ = lambda r: {0: [Circle(0.6, 0.6, r, 4.0)]}  # noqa: E731
stA = mk([(0.5, circ(0.36)[0])])
twA, fA = runner(stA, circ)
fj, gj = jax.jit(fA), jax.jit(jax.jacrev(fA))
rows = []
for r in (0.30, 0.33, 0.36, 0.39, 0.42):
    v = np.asarray(fj(r))
    g = np.asarray(gj(r))
    fd = []
    for hs in (3e-4, 1e-4):
        h = hs * P
        fd.append((np.asarray(fj(r + h)) - np.asarray(fj(r - h))) / (2 * h))
    rich = (9 * fd[1] - fd[0]) / 8
    _o, Rn, Tn, _J = mk([(0.5, circ(r)[0])], "numpy").solve()
    vn = np.array([Rn[0, twA.p0], Tn[0, twA.p0]])
    rows.append({"r": r, "value": v.tolist(), "AD": g.tolist(),
                 "FD_rich": rich.tolist(),
                 "AD_vs_FD_rel": (np.abs(g - rich) / np.abs(rich)).tolist(),
                 "twin_vs_numpy_at_r": absd(v, vn)})
    print("A", rows[-1], flush=True)
out["smooth_path"] = rows


# ---- Part B: events --------------------------------------------------------
def event(name, layers_ref, layers_of, bad_traced, bad_concrete, good):
    st = mk(layers_ref)
    tw, f = runner(st, layers_of)
    fj = jax.jit(f)
    gj = jax.jit(jax.grad(lambda x: f(x)[1]))
    rec = {"good_value": np.asarray(fj(good)).tolist(),
           "good_grad": float(gj(good)),
           "bad_traced_value": np.asarray(fj(bad_traced)).tolist(),
           "bad_traced_grad": float(gj(bad_traced))}
    try:
        p = tw.params()
        for k, shp in layers_of(bad_concrete).items():
            p["layers"][k]["shapes"] = shp
        st.solve(params=p)
        rec["concrete"] = "ACCEPTED"
    except ValueError as exc:
        rec["concrete"] = "refused: " + str(exc)[:160]
    try:
        p = tw.params()
        for k, shp in layers_of(good).items():
            p["layers"][k]["shapes"] = shp
        st.solve(params=p)
        rec["concrete_good"] = "accepted"
    except ValueError as exc:
        rec["concrete_good"] = "REFUSED: " + str(exc)[:160]
    print("B", name, rec, flush=True)
    out[name] = rec


event("fold", [(0.5, [Circle(0.6, 0.6, 0.36, 4.0)])], circ, 0.62, 0.62,
      0.58)
h0 = 0.36 / np.sqrt(2.0)
tang = lambda r: {0: [Circle(0.6, 0.6, r, 4.0)],  # noqa: E731
                  1: [Rect(0.6, 0.11, 2 * h0, 0.18, 2.25)]}
event("tangency", [(0.3, tang(0.36)[0]), (0.2, tang(0.36)[1])], tang, 0.37,
      0.37, 0.36)
rect = lambda w: {0: [Rect(0.6, 0.6, w, 0.4, 4.0)]}  # noqa: E731
event("sliver", [(0.4, rect(0.5)[0])], rect, 1.199, 1.199, 1.19)
fil = lambda r: {0: [FilletRect(0.6, 0.6, 0.6, 0.5, r, 4.0)]}  # noqa: E731
event("duffy", [(0.4, fil(0.1)[0])], fil, 1e-5, 0.0, 0.05)
dump(f"f4_events_M{M}.json", out)
