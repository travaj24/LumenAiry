"""PROBE 3d: FALSE-NEGATIVE CENSUS of the sliver guard.

For every (sliver width s, degree) cell: run the SHIPPED default path (guard
armed) and record
  * the returned answer vs the continuous physical trend,
  * whether it RAISED, WARNED, or was SILENT.
A cell that is WRONG and SILENT is a false negative.
"""
import sys, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm import stack as pst

per, wl = 1.0e-6, 1.55e-6
ang = np.deg2rad(12.0)
eps_hi, eps_lo = 3.48 ** 2, 1.444 ** 2


def run(s, degree, guard=True):
    old = pst.PMM_SLIVER_GUARD
    pst.PMM_SLIVER_GUARD = guard
    try:
        st = PMMStack(per, n_substrate=1.444, n_superstrate=1.0,
                      degree=degree, far_field_orders=11)
        st.add_layer(0.25e-6, segments=[(0.5, eps_hi), (0.5, eps_lo)])
        st.add_layer(0.25e-6, segments=[(0.5 + s, eps_hi), (0.5 - s, eps_lo)])
        st.set_source(wl, angle=ang)
        with warnings.catch_warnings(record=True) as W:
            warnings.simplefilter("always")
            try:
                o, R, T, J = st.solve()
            except Exception as e:
                return ("RAISE", None, None)
            m0 = int(np.where(o == 0)[0][0])
            txt = " | ".join(str(x.message)[:200] for x in W)
            sev = "SILENT"
            if "SLIVER" in txt or "sliver" in txt:
                sev = "warnSLIV"
            elif "energy not conserved" in txt:
                sev = "warnENER"
            elif txt:
                sev = "warnOTHR"
            return (sev, T[1, m0], float(np.max(R.sum(1) + T.sum(1))))
    finally:
        pst.PMM_SLIVER_GUARD = old


# continuous reference
_, T0, _ = run(0.0, 20, guard=False)
_, T1, _ = run(1e-3, 20, guard=False)
slope = (T1 - T0) / 1e-3
print(f"reference T0 = {T0:.12f}, slope = {slope:.6g}")
print()
print("cells marked ** are WRONG (|dT| > 1e-3) ; a WRONG+SILENT cell is a "
      "FALSE NEGATIVE")
degs = (10, 12, 14, 16, 18, 20, 22, 24)
print(f"{'s':>9} | " + " ".join(f"{'d'+str(d):>13}" for d in degs))
fn = []
for s in (1.0e-5, 1.2e-5, 1.5e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6.5e-5, 8e-5,
          1e-4, 1.5e-4, 2e-4):
    exp = T0 + slope * s
    cells = []
    for d in degs:
        sev, v, tot = run(s, d)
        if sev == "RAISE":
            cells.append("       RAISE ")
            continue
        wrong = abs(v - exp) > 1e-3
        tag = {"SILENT": ".", "warnSLIV": "S", "warnENER": "E",
               "warnOTHR": "o"}[sev]
        cells.append(f"{v:11.7f}{'*' if wrong else ' '}{tag}")
        if wrong and sev == "SILENT":
            fn.append((s, d, v, exp, tot))
    print(f"{s:9.1e} | " + " ".join(cells))

print()
print("FALSE NEGATIVES (wrong AND silent):")
if not fn:
    print("  none")
for s, d, v, exp, tot in fn:
    print(f"  s={s:.1e} degree={d:3d}: got {v:.9f}, expected {exp:.9f} "
          f"(err {abs(v-exp):.3g}), max R+T = {tot:.9f}")
