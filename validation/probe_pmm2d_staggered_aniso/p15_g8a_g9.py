"""Probe 15 -- G8a / G9 / G5-closure at the EXACT parameters the test file
uses (M = 6 for the stack gates)."""
import numpy as np

import lumenairy

assert lumenairy.__file__.replace("\\", "/").startswith("C:/tmp/lum_aniso"), \
    lumenairy.__file__

import tests.unit.test_pmm2d_staggered_anisotropic as t  # noqa: E402

c = t._cell(t._LC, t._ISO)
st = t.PMM2DStackPure(t._P, t._P, n_superstrate=1.0, n_substrate=1.5,
                      n_modes=6, n_orders=3)
st.add_layer(t._DEP, eps_cell=c).add_layer(t._DEP, eps_cell=c)
st.set_source(t._WL, theta=0.15, phi=0.4)
_o2, R2, T2, J2 = st.solve()
_o1, R1, T1, J1 = t.pmm_jones_2d_staggered(t._P, t._P, c, 1.5, 1.0,
                                           2 * t._DEP, t._WL, degree=6,
                                           n_orders=3, theta=0.15, phi=0.4)
print("G8a split-vs-single (M=6): dR=%.3e dT=%.3e dJ=%.3e"
      % (np.max(np.abs(R2 - R1)), np.max(np.abs(T2 - T1)),
         np.max(np.abs(J2 - J1))))

st = t.PMM2DStackPure(t._P, t._P, n_superstrate=1.0, n_substrate=1.5,
                      n_modes=6, n_orders=3)
st.add_layer(0.12e-6, eps_cell=t._cell(t._LC, t._ISO))
st.add_layer(t._DEP, eps_cell=t._cell(t._LC, t._ISO + 0.8j * np.eye(3)))
st.add_layer(0.09e-6, eps=t._GYRO)
st.set_source(t._WL, theta=0.12, phi=0.3)
_o, R, T, _J = st.solve(retain_internal=True)
A = st.layer_absorption()
for col in (0, 1):
    lhs, rhs = float(A[:, col].sum()), float(1 - R[col].sum() - T[col].sum())
    print(f"G9 pol {col}: sumA={lhs:.10f} 1-R-T={rhs:.10f} dev={abs(lhs-rhs):.3e}"
          f"  per-layer={np.round(A[:, col], 8)}")

cg = t._g5_cell()
o, R, T, _J = t.pmm_jones_2d_staggered(t._P, t._P, cg, 1.5, 1.0, t._DEP,
                                       t._WL, degree=7, n_orders=5)
print("G5 staggered closure:", np.max(np.abs(R.sum(1) + T.sum(1) - 1)))
