"""V3 -- re-measure the BUILDER's own G.1 / G.4 / G.7 numbers on their exact
geometry, on whichever arm is passed.

Rows re-measured (all normal incidence, n_sup = 1.0, n_sub = 1.5, M = 5,
n_orders = 3, depth 0.28 um, px = 0.5 um, wl = 1.0 um -- the cut-off of a
uniform eps = 4 layer at order m = 1):

  1. uniform eps = 4 cell: `pmm_efficiency_2d_staggered` vs
     `pmm_jones_2d_staggered` on its `e*I` promotion       (claimed 4.591e-08)
  2. patterned {4, 1} cell, `PMM2DStackPure` both arms      (claimed 7.561e-09)
  3. uniform SCALAR layer `add_layer(eps=4)` vs `add_layer(eps=4*I)`
                                          (claimed 4.977e-08 pre / 1.381e-15 post)
  4. the CONSEQUENCE of the nudge: solve(nudged) vs solve(WL*(1-1e-9)), and
     max|R|                                     (claimed 7.637e-09 / 0.0115)
  5. the dedup COST: guard time per call, 64x64 cell, n_orders = 7
                                              (claimed 46.92 ms vs 0.02 ms)

    python v3_builder_table.py <lumenairy-root>
"""
import os
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

_ROOT = os.path.abspath(sys.argv[1])
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT), lumenairy.__file__

from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_efficiency_2d_staggered,
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa._core import (  # noqa: E402
    _grazing_safe_wavelength,
)

PX, WL, D = 0.5e-6, 1.0e-6, 0.28e-6
M, NO = 5, 3
UNI = np.array([[4.0, 4.0], [4.0, 4.0]], dtype=complex)
PAT = np.array([[4.0, 1.0], [1.0, 1.0]], dtype=complex)   # the builder's quarter-fill cell


def promote(m):
    m = np.asarray(m, dtype=complex)
    t = np.zeros(m.shape + (3, 3), dtype=complex)
    for i in range(3):
        t[..., i, i] = m
    return t


def q():
    return warnings.catch_warnings()


print(f"(wl/px)**2 - 4 = {(WL / PX) ** 2 - 4.0!r}")

# 1 -----------------------------------------------------------------------
with q():
    warnings.simplefilter("ignore")
    _o, R1, T1 = pmm_efficiency_2d_staggered(PX, PX, UNI, 1.5, 1.0, D, WL,
                                             degree=M, n_orders=NO,
                                             polarization="tm")
    _o, R2, T2, _J = pmm_jones_2d_staggered(PX, PX, promote(UNI), 1.5, 1.0, D,
                                            WL, degree=M, n_orders=NO)
print(f"1. uniform eps=4 cell, eff vs jones-promotion: "
      f"max|dR| = {np.max(np.abs(R1 - R2[0])):.4e}  "
      f"identical = {np.array_equal(R1, R2[0])}")


def stack(cell, wl=WL):
    st = PMM2DStackPure(PX, PX, n_superstrate=1.0, n_substrate=1.5,
                        n_modes=M, n_orders=NO)
    st.add_layer(D, eps_cell=cell)
    st.set_source(wl, theta=0.0, phi=0.0)
    with q():
        warnings.simplefilter("ignore")
        return st.solve(jones=True)


def stack_uniform(eps, wl=WL):
    st = PMM2DStackPure(PX, PX, n_superstrate=1.0, n_substrate=1.5,
                        n_modes=M, n_orders=NO)
    st.add_layer(D, eps=eps)
    st.set_source(wl, theta=0.0, phi=0.0)
    with q():
        warnings.simplefilter("ignore")
        return st.solve(jones=True)


# 2 -----------------------------------------------------------------------
_o, Ra, Ta, Ja = stack(PAT)
_o, Rb, Tb, Jb = stack(promote(PAT))
print(f"2. patterned {{4,1}} cell, stack scalar vs e*I: "
      f"max|dR| = {np.max(np.abs(Ra - Rb)):.4e}  "
      f"identical = {np.array_equal(Ra, Rb)}")

# 3 -----------------------------------------------------------------------
_o, Rc, Tc, Jc = stack_uniform(4.0)
_o, Rd, Td, Jd = stack_uniform(np.diag([4.0] * 3).astype(complex))
print(f"3. uniform SCALAR layer eps=4 vs eps=4*I: "
      f"max|dR| = {np.max(np.abs(Rc - Rd)):.4e}  "
      f"identical = {np.array_equal(Rc, Rd)}")

# 4 -----------------------------------------------------------------------
_o, Re_, Te_, _J = stack(PAT, wl=WL * (1 - 1e-9))
print(f"4. nudged vs WL*(1-1e-9), patterned {{4,1}}: "
      f"max|dR| = {np.max(np.abs(Ra - Re_)):.4e}   "
      f"max|R| = {np.max(np.abs(Ra)):.6g}   "
      f"1000-ULP floor = {1e3 * np.max(np.abs(Ra)) * np.finfo(float).eps:.4e}")

# 5 -----------------------------------------------------------------------
big = np.where(np.add.outer(np.arange(64), np.arange(64)) % 2 == 0,
               4.0, 1.0).astype(complex)   # 2 materials, 64x64 segments
mo = np.arange(-7, 8)
mx, my = np.tile(mo, len(mo)), np.repeat(mo, len(mo))
raw = list(np.real(np.concatenate([[1.0, 2.25], np.repeat(big.ravel(), 3)])))
ded = [float(v) for v in np.unique(np.real(np.concatenate(
    [[1.0, 2.25], np.repeat(big.ravel(), 3)])))]
for nm, lst in (("raw", raw), ("dedup", ded)):
    t0 = time.perf_counter()
    for _ in range(3):
        out = _grazing_safe_wavelength(0.633e-6, 0.0, 0.0, mx, my, 0.7e-6,
                                       0.7e-6, lst)
    dt = (time.perf_counter() - t0) / 3
    print(f"5. guard {nm:5s} len={len(lst):6d}  {dt * 1e3:8.3f} ms/call  "
          f"wl -> {out!r}")
