"""V5 (E3-4): the non-differentiable events on the verifier's boundary cases
(several from the Phase C verifier's primitive-limit table), and the
airtightness of the multiplicative NaN poison.

    python v5_events.py M

Per case: the twin frozen at a reference x0; under jax.jit the value and the
gradient at an EVENT point (traced) and at a CONTROL point inside the
topology; the same event point CONCRETE (eager) -> refused or not.  For the
event points: is every entry of R / T / J NaN, do sum(R) + sum(T), the
gradient of sum(T) and the gradient w.r.t. a MATERIAL parameter come out NaN.
"""
import sys

from _ve3 import WL, P, PMM2DStackPure, dump, jax, jnp, np

from lumenairy.elements.pmm import Circle, FilletRect, Rect, SinusoidalWall

M = int(sys.argv[1])

CASES = {
    # name: (x0, shapes(x, e), event x, control x, note)
    "circle_edge_sliver": (0.5, lambda x, e: [Circle(0.6, 0.6, x, e)],
                           0.5995, 0.59, "outline within the sliver margin "
                           "of the cell edge (concrete merge refuses)"),
    "circle_bulge": (0.5, lambda x, e: [Circle(0.6, 0.6, x, e)], 0.61, 0.58,
                     "outline past the cell edge (fold)"),
    "fillet_sliver": (0.05, lambda x, e: [FilletRect(0.6, 0.6, 0.6, 0.5, x,
                                                     e)],
                      1.3e-3 * P, 1.6e-3 * P, "fillet below sqrt(2) 1e-3 p"),
    "sine_edge_sliver": (0.1, lambda x, e: [SinusoidalWall("x", 0.6, x,
                                                           eps=e)],
                         0.5995, 0.55, "wiggle within the sliver margin of "
                         "the cell edge"),
    "sine_past_edge": (0.1, lambda x, e: [SinusoidalWall("x", 0.6, x,
                                                         eps=e)],
                       0.62, 0.55, "wiggle past the cell edge"),
    "circle_in_rect_cross": (0.3, lambda x, e: [Rect(0.6, 0.6, 0.8, 0.8, 2.0),
                                                Circle(0.6, 0.6, x, e)],
                             0.42, 0.38, "the circle's arc crosses the "
                             "rectangle's edge, walls still ordered"),
    "two_circles_approach": (0.85, lambda x, e: [Circle(0.3, 0.3, 0.15, e),
                                                 Circle(x, x, 0.2, 2.2)],
                             0.53, 0.65, "two circles approach diagonally: "
                             "their 45-degree walls reorder"),
    "two_rects_reorder": (0.8, lambda x, e: [Rect(0.3, 0.6, 0.2, 0.3, e),
                                             Rect(x, 0.6, 0.2, 0.3, 2.2)],
                          0.45, 0.62, "walls of two rectangles reorder"),
}
E0 = 3.5
out = {"M": M, "cases": {}}


def run_case(name, x0, shp, xe, xc, note):
    r = {"note": note, "x0": x0, "event": xe, "control": xc}
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=2, backend="jax")
    st.add_layer(0.4, shapes=shp(x0, E0), background_eps=1.0)
    st.set_source(WL)
    tw = st.jax_twin()
    p0 = tw.p0

    def full(x, e, st=st, tw=tw, shp=shp):
        p = tw.params()
        p["layers"][0]["shapes"] = shp(x, e)
        _o, R, T, J = st.solve(params=p)
        return R, T, J

    fj = jax.jit(full)
    gT = jax.jit(jax.grad(lambda x: full(x, E0)[1][0, p0]))
    gsum = jax.jit(jax.grad(lambda x: jnp.sum(full(x, E0)[1])))
    gE = jax.jit(jax.grad(lambda e, x: full(x, e)[1][0, p0]))
    pw = jax.jit(lambda x: jnp.sum(full(x, E0)[0]) + jnp.sum(full(x, E0)[1]))
    for tag, x in (("event", xe), ("control", xc)):
        R, T, J = (np.asarray(a) for a in fj(x, E0))
        r[tag + "_all_nan"] = bool(np.all(np.isnan(R)) and np.all(np.isnan(T))
                                   and np.all(np.isnan(J)))
        r[tag + "_any_nan"] = bool(np.any(np.isnan(R)) or np.any(np.isnan(T))
                                   or np.any(np.isnan(J)))
        r[tag + "_T00"] = float(T[0, p0])
        r[tag + "_grad_T00"] = float(gT(x))
        r[tag + "_grad_sumT"] = float(gsum(x))
        r[tag + "_power_sum"] = float(pw(x))
    r["event_grad_wrt_eps"] = float(gE(E0, xe))
    # the CONCRETE event value: refused?
    try:
        p = tw.params()
        p["layers"][0]["shapes"] = shp(xe, E0)
        _o, R, T, J = st.solve(params=p)
        r["concrete_event"] = ("ACCEPTED, T00 = %r" % float(np.asarray(T)[0,
                                                                         p0]))
    except ValueError as exc:
        r["concrete_event"] = "refused: " + str(exc)[:140]
    # the NumPy stack at the event value (independent of the twin)
    try:
        sn = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                            n_modes=M, n_orders=2)
        sn.add_layer(0.4, shapes=shp(xe, E0), background_eps=1.0)
        sn.set_source(WL)
        o, R, T, J = sn.solve()
        r["numpy_event"] = "ACCEPTED, T00 = %r" % float(T[0, 12])
    except ValueError as exc:
        r["numpy_event"] = "refused: " + str(exc)[:140]
    ok_event = r["event_all_nan"] and np.isnan(r["event_grad_T00"]) and \
        np.isnan(r["event_grad_sumT"]) and np.isnan(r["event_power_sum"]) \
        and np.isnan(r["event_grad_wrt_eps"])
    ok_ctrl = not r["control_any_nan"] and np.isfinite(r["control_grad_T00"])
    r["verdict"] = ("NaN at the event, finite at the control" if ok_event
                    and ok_ctrl else "CHECK")
    return r


for name, (x0, shp, xe, xc, note) in CASES.items():
    try:
        r = run_case(name, x0, shp, xe, xc, note)
    except Exception as exc:  # noqa: BLE001 -- the probe records it
        r = {"note": note, "error": repr(exc)[:300], "verdict": "ERROR",
             "concrete_event": "", "numpy_event": "", "event_T00": None}
    out["cases"][name] = r
    print(name, r["verdict"], "| concrete:", r["concrete_event"][:60],
          "| numpy:", r["numpy_event"][:40], "| event T00",
          r["event_T00"], flush=True)
print(dump(f"v5_events_M{M}.json", out))
