"""E3-4 supplement: two boundary cases from the Phase C verifier's report
(VERIFY_PMM2D_CURVED_C, branch verify/pmm2d-curved-c: V-D2 and V-D1).

    python f4b_verifier_cases.py M

  ellipse_fold -- a rotated Ellipse(0.6, 0.6, 0.35, 0.2) frozen at angle
                  0.2; the NumPy merge REFUSES it as a fold from ~0.34 rad
                  (scan below); the angle traced to 0.36 must give NaN, a
                  concrete 0.36 must be refused, 0.28 must be finite;
  roundoff_rects -- two rectangles sharing one wall, computed two ways
                  ((0.1 + 0.2) + 0.1 and 0.5 - 0.1; aimed at V-D1, but on
                  this layout the two sums agree and the merge reaches the
                  identity -- recorded as numpy_mapped): the twin at the
                  reference equals the NumPy solve; a traced width of ONE
                  rectangle separates the shared wall (NaN) and a concrete
                  one is refused (a 5e-7 sliver).
Output f4b_verifier_cases_M<M>.json.
"""
import sys

import numpy as np
from _e3common import WL, P, PMM2DStackPure, absd, dump, jax

from lumenairy.elements.pmm import Ellipse, Rect, compile_shapes

M = int(sys.argv[1])
out = {"M": M}
scan = {}
for a in np.arange(0.28, 0.37, 0.02):
    try:
        compile_shapes(P, P, [Ellipse(0.6, 0.6, 0.35, 0.2, 4.0, angle=a)], 1.0)
        scan[f"{a:.2f}"] = "ok"
    except ValueError as exc:
        scan[f"{a:.2f}"] = "REFUSED: " + str(exc)[:60]
out["ellipse_fold_scan"] = scan


def mk(shapes, backend="jax"):
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=M,
                        n_orders=2, backend=backend)
    st.add_layer(0.4, shapes=shapes, background_eps=1.0)
    st.set_source(WL)
    return st


def case(name, ref_shapes, shapes_of, bad, good):
    st = mk(ref_shapes)
    tw = st.jax_twin()
    p0 = tw.p0

    def f(x):
        p = tw.params()
        p["layers"][0]["shapes"] = shapes_of(x)
        return st.solve(params=p)[2][0, p0]
    fj = jax.jit(f)
    gj = jax.jit(jax.grad(f))
    o, R, T, J = mk(ref_shapes, "numpy").solve()
    _o, Rt, Tt, Jt = st.solve()
    rec = {"numpy_mapped": mk(ref_shapes, "numpy").cmap is not None,
           "ref_parity": [absd(Rt, R), absd(Tt, T), absd(Jt, J)],
           "bad_traced": [float(fj(bad)), float(gj(bad))],
           "good_traced": [float(fj(good)), float(gj(good))]}
    try:
        p = tw.params()
        p["layers"][0]["shapes"] = shapes_of(bad)
        st.solve(params=p)
        rec["bad_concrete"] = "ACCEPTED"
    except ValueError as exc:
        rec["bad_concrete"] = "refused: " + str(exc)[:120]
    print(name, rec, flush=True)
    out[name] = rec


case("ellipse_fold", [Ellipse(0.6, 0.6, 0.35, 0.2, 4.0, angle=0.2)],
     lambda a: [Ellipse(0.6, 0.6, 0.35, 0.2, 4.0, angle=a)], 0.36, 0.28)
cxa = 0.1 + 0.2                          # 0.30000000000000004
case("roundoff_rects", [Rect(cxa, 0.6, 0.2, 0.4, 4.0),
                        Rect(0.5, 0.6, 0.2, 0.4, 2.25)],
     lambda w: [Rect(cxa, 0.6, w, 0.4, 4.0), Rect(0.5, 0.6, 0.2, 0.4, 2.25)],
     0.2 + 1e-6, 0.2)
dump(f"f4b_verifier_cases_M{M}.json", out)
