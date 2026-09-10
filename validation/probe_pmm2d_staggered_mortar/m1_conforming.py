"""M1 -- CONFORMING IDENTITY.  Two layers on the SAME grid, forced through the
MORTAR path, against the shipped union-grid PMM2DStackPure cascade.  The mortar
must reduce to the plain square modal match, so the two must agree to solver
round-off (~1e-14 relative); bit identity is NOT required (the mortar
associates the same algebra differently)."""
import json

import numpy as np
from mortar2d import MortarStack2D, guard

print("lumenairy:", guard())
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure

PX = PY = 0.9e-6
WL = 0.60e-6
A = np.array([[2.0] * 3, [4.0] * 3, [9.0] * 3], np.complex128)          # stripe
B = np.array([[9.0, 2.0, 4.0]] * 3, np.complex128).T.copy()             # crossed
P = np.array([[6.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 3.5]],
             np.complex128)                                             # pillar
tA, tB = 0.20e-6, 0.15e-6

CASES = [
    ("stripe|stripe  M=5 th=0.00", [A, B], 5, 0.00, 0.0),
    ("stripe|stripe  M=5 th=0.20", [A, B], 5, 0.20, 0.0),
    ("stripe|pillar  M=6 th=0.20", [A, P], 6, 0.20, 0.0),
    ("pillar|pillar  M=6 conical", [P, B], 6, 0.25, 0.7),
    ("3-layer + unif M=5 th=0.15", [A, None, P], 5, 0.15, 0.4),
]

rows = []
for name, cells, M, th, ph in CASES:
    ref = PMM2DStackPure(PX, PY, n_modes=M, n_orders=5)
    mor = MortarStack2D(PX, PY, n_modes=M, n_orders=5)
    for c in cells:
        if c is None:
            ref.add_layer(0.08e-6, eps=2.25)
            mor.add_layer(0.08e-6, eps=2.25, grid=3)
        else:
            ref.add_layer(tA if c is cells[0] else tB, eps_cell=c)
            mor.add_layer(tA if c is cells[0] else tB, eps_cell=c)
    ref.set_source(WL, theta=th, phi=ph)
    mor.set_source(WL, theta=th, phi=ph)
    o0, R0, T0, J0 = ref.solve()
    o1, R1, T1, J1 = mor.solve(force_mortar=True)
    assert np.array_equal(o0, o1)
    scale = max(float(np.max(R0)), float(np.max(T0)))
    dR = float(np.max(np.abs(R0 - R1))) / scale
    dT = float(np.max(np.abs(T0 - T1))) / scale
    dJ = float(np.max(np.abs(J0 - J1))) / float(np.max(np.abs(J0)))
    clo0 = float(abs(R0.sum(1)[1] + T0.sum(1)[1] - 1.0))
    clo1 = float(abs(R1.sum(1)[1] + T1.sum(1)[1] - 1.0))
    rows.append(dict(case=name, dR=dR, dT=dT, dJ=dJ,
                     closure_union=clo0, closure_mortar=clo1))
    print(f"{name:28s} dR {dR:9.2e}  dT {dT:9.2e}  dJ {dJ:9.2e} "
          f"| closure union {clo0:8.1e} mortar {clo1:8.1e}")

worst = max(max(r["dR"], r["dT"], r["dJ"]) for r in rows)
print(f"\nWORST relative disagreement over {len(rows)} cases: {worst:.3e}")
json.dump(dict(rows=rows, worst=worst),
          open("validation/probe_pmm2d_staggered_mortar/m1_conforming.json", "w"),
          indent=1)
