"""Probe 16 -- G9 absorption-budget closure ladder in M (tensor stack)."""
import time

import numpy as np

import lumenairy

assert lumenairy.__file__.replace("\\", "/").startswith("C:/tmp/lum_aniso"), \
    lumenairy.__file__

import tests.unit.test_pmm2d_staggered_anisotropic as t  # noqa: E402

for M in (6, 7, 8):
    t0 = time.perf_counter()
    st = t.PMM2DStackPure(t._P, t._P, n_superstrate=1.0, n_substrate=1.5,
                          n_modes=M, n_orders=3)
    st.add_layer(0.12e-6, eps_cell=t._cell(t._LC, t._ISO))
    st.add_layer(t._DEP, eps_cell=t._cell(t._LC, t._ISO + 0.8j * np.eye(3)))
    st.add_layer(0.09e-6, eps=t._GYRO)
    st.set_source(t._WL, theta=0.12, phi=0.3)
    _o, R, T, _J = st.solve(retain_internal=True)
    A = st.layer_absorption()
    dev = [abs(float(A[:, c].sum()) - float(1 - R[c].sum() - T[c].sum()))
           for c in (0, 1)]
    print(f"M={M}  dev = {dev[0]:.3e} / {dev[1]:.3e}   "
          f"A_lossy = {A[1, 0]:.6f} / {A[1, 1]:.6f}   "
          f"max|A_lossless| = {np.max(np.abs(A[[0, 2]])):.2e}   "
          f"{time.perf_counter()-t0:.2f} s")
