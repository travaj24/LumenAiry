"""E3-3 supplement: the circle radius at CONICAL incidence (theta 0.3, phi
0.4), M = 4, whose standard ladder (f3_circle_r_conical_M4.json) never
reached its asymptotic range: T00(r) has a sharp feature near r0 (the NumPy
solve shows it identically).  A finer ladder h / P = 1e-4 .. 1e-7 on the
twin's forward, and the value of T00 across +-3e-4 P.
Output f3b_fine_ladder_M4.json.
"""
import sys

from _e3common import WL, P, PMM2DStackPure, dump, jax, np

from lumenairy.elements.pmm import Circle

M = int(sys.argv[1]) if len(sys.argv) > 1 else 4
st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=M,
                    n_orders=2, backend="jax")
st.add_layer(0.5, shapes=[Circle(0.6, 0.6, 0.36, 4.0)], background_eps=1.0)
st.set_source(WL, theta=0.3, phi=0.4)
tw = st.jax_twin()
p0 = tw.p0


def f(r):
    p = tw.params()
    p["layers"][0]["shapes"] = [Circle(0.6, 0.6, r, 4.0)]
    return st.solve(params=p)[2][0, p0]


fj = jax.jit(f)
g = float(jax.jit(jax.grad(f))(0.36))
steps = [1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7]
fd = [(float(fj(0.36 + h * P)) - float(fj(0.36 - h * P))) / (2 * h * P)
      for h in steps]
scan = np.linspace(-3e-4, 3e-4, 25) * P
vals = [float(fj(0.36 + d)) for d in scan]
out = {"M": M, "AD": g, "steps": steps, "FD": fd,
       "rel": [abs(g - v) / abs(g) for v in fd],
       "scan_offsets_over_P": (scan / P).tolist(), "scan_T00": vals}
dump(f"f3b_fine_ladder_M{M}.json", out)
print(out["AD"], out["FD"], out["rel"])
