"""V4c: locate the STEP in the NumPy (unmapped) rectangle solve at oblique
incidence that breaks its FD premise (v4b: second differences 15x the
median at M = 3), and name the discrete decision behind it.

    python v4c_unmapped_step.py M
"""
import sys

from _ve3 import TS, WL, P, PMM2DStackPure, dump, np

from lumenairy.elements.pmm import Rect

M = int(sys.argv[1])
W0 = 0.47
calls = []
orig = TS._stag_quad_order


def spy(Mq, omega):
    n = orig(Mq, omega)
    calls.append(n)
    return n


TS._stag_quad_order = spy


def solve(w):
    calls.clear()
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=2)
    st.add_layer(0.4, shapes=[Rect(0.6, 0.55, w, 0.42, 3.6)],
                 background_eps=1.0)
    st.set_source(WL, theta=0.3)
    o, R, T, J = st.solve()
    return T[0, 12], tuple(calls)


ws = W0 + np.linspace(-2e-3, 2e-3, 81) * P
vals, nqs = zip(*[solve(w) for w in ws])
vals = np.array(vals)
d2 = np.abs(np.diff(vals, 2))
k = int(np.argmax(d2))
out = {"M": M, "d2_max": float(d2[k]), "d2_median": float(np.median(d2)),
       "at_w": [float(ws[k]), float(ws[k + 2])],
       "nq_change_between": [list(nqs[k]), list(nqs[k + 2])]
       if nqs[k] != nqs[k + 2] else "no nq change",
       "n_nq_changes_in_scan": int(sum(nqs[i] != nqs[i + 1]
                                       for i in range(len(nqs) - 1)))}
# bisect the step and measure its size
a, b = ws[k], ws[k + 2]
for _ in range(40):
    m = 0.5 * (a + b)
    if solve(m)[1] == solve(a)[1]:
        a = m
    else:
        b = m
if solve(a)[1] != solve(b)[1]:
    out["step_at"] = float(a)
    out["step_size"] = float(abs(solve(b)[0] - solve(a)[0]))
    out["nq_a_b"] = [list(solve(a)[1]), list(solve(b)[1])]
print(out)
dump(f"v4c_unmapped_step_M{M}.json", out)
