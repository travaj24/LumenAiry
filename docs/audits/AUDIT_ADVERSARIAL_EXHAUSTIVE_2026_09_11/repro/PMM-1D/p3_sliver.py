"""PROBE 3: sliver ladder -- is the answer CONTINUOUS in the sliver width, and
does the guard trigger where it should?

Two layers whose walls collide at a controlled offset `s` (a fraction of the
period).  As s -> 0 the union grid manufactures a cell of width s.  The
PHYSICS is continuous in s (the geometry tends to the coincident-wall limit),
so the SOLVED answer must be too.
"""
import sys, warnings, traceback
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm import stack as pst

per = 1.0e-6
wl = 1.55e-6
ang = np.deg2rad(12.0)
eps_hi, eps_lo = 3.48 ** 2, 1.444 ** 2


def solve(s, degree=16, min_feature=None, guard=True):
    """layer 1 ridge [0, 0.5]; layer 2 ridge [0, 0.5 + s]."""
    old = pst.PMM_SLIVER_GUARD
    pst.PMM_SLIVER_GUARD = guard
    try:
        st = PMMStack(per, n_substrate=1.444, n_superstrate=1.0,
                      degree=degree, far_field_orders=11,
                      min_feature=min_feature)
        st.add_layer(0.25e-6, segments=[(0.5, eps_hi), (0.5, eps_lo)])
        st.add_layer(0.25e-6, segments=[(0.5 + s, eps_hi), (0.5 - s, eps_lo)])
        st.set_source(wl, angle=ang)
        msgs = []
        with warnings.catch_warnings(record=True) as W:
            warnings.simplefilter("always")
            try:
                o, R, T, J = st.solve()
                err = None
            except Exception as e:
                return None, None, None, [str(w.message)[:90] for w in W], \
                    f"{type(e).__name__}: {str(e)[:110]}"
            msgs = [str(w.message)[:90] for w in W]
        return o, R, T, msgs, None
    finally:
        pst.PMM_SLIVER_GUARD = old


print("=== 3a: sliver ladder s = 1e-3 .. 1e-12 (guard ON, default mf) ===")
print(f"{'s':>10} {'tot0':>14} {'tot1':>14} {'R0(Ey)':>16} {'T0(Ey)':>16} note")
base = None
rows = []
for s in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11,
          1e-12, 0.0):
    o, R, T, msgs, err = solve(s)
    if err:
        print(f"{s:10.0e} {'--':>14} {'--':>14} {'--':>16} {'--':>16} RAISED "
              f"{err}")
        rows.append((s, None, None))
        continue
    m0 = int(np.where(o == 0)[0][0])
    tot = R.sum(axis=1) + T.sum(axis=1)
    note = ("; ".join(m.split(":")[0] for m in msgs))[:60]
    print(f"{s:10.0e} {tot[0]:14.10f} {tot[1]:14.10f} {R[1,m0]:16.12f} "
          f"{T[1,m0]:16.12f} {note}")
    rows.append((s, R[1, m0], T[1, m0]))

print()
print("--- continuity: |answer(s) - answer(0)| vs s ---")
ref = rows[-1]
for s, r, t in rows:
    if r is None or ref[1] is None:
        print(f"{s:10.0e}  (missing)")
    else:
        print(f"{s:10.0e}  dR0={abs(r-ref[1]):.3e}  dT0={abs(t-ref[2]):.3e}")

print()
print("=== 3b: same ladder with the guard OFF (does it change any number?) ===")
for s in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
    o1, R1, T1, m1, e1 = solve(s, guard=True)
    o2, R2, T2, m2, e2 = solve(s, guard=False)
    if e1 or e2:
        print(f"{s:10.0e}  guardON={e1}  guardOFF={e2}")
        continue
    print(f"{s:10.0e}  max|dR|={np.max(np.abs(R1-R2)):.3e} "
          f"max|dT|={np.max(np.abs(T1-T2)):.3e}")

print()
print("=== 3c: degree sensitivity at a fixed sliver (is it 'passive but "
      "wrong'?) ===")
for s in (1e-3, 1e-5, 1e-7, 1e-9):
    line = []
    for deg in (10, 14, 18, 22):
        o, R, T, msgs, err = solve(s, degree=deg, guard=False)
        if err:
            line.append("RAISE")
        else:
            m0 = int(np.where(o == 0)[0][0])
            line.append(f"{T[1,m0]:.9f}")
    print(f"s={s:.0e}: " + "  ".join(line))
