import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

import lumenairy

_ROOT = os.path.abspath(sys.argv[1]).replace("\\", "/").lower()
assert os.path.abspath(lumenairy.__file__).replace("\\", "/").lower().startswith(_ROOT)
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
from lumenairy.elements.rcwa._core import _grazing_safe_wavelength

PX = 0.5e-6
EPS_L = 4.0
CELL = np.array([[EPS_L, 1.0], [1.0, 1.0]], dtype=complex)
NSUP, NSUB = 1.0, 1.5
WL_CUT = PX * np.sqrt(EPS_L)          # m = 1 grazes the eps = 4 layer
mo = np.arange(-3, 4)
mx, my = np.tile(mo, len(mo)), np.repeat(mo, len(mo))
old = _grazing_safe_wavelength(WL_CUT, 0.0, 0.0, mx, my, PX, PX,
                               [NSUP ** 2, NSUB ** 2])
new = _grazing_safe_wavelength(WL_CUT, 0.0, 0.0, mx, my, PX, PX,
                               [NSUP ** 2, NSUB ** 2, EPS_L])
print(f"WL_CUT = {WL_CUT!r}  old-rule -> {old!r} (moved: {old != WL_CUT})  "
      f"new-rule -> {new!r} (moved: {new != WL_CUT}, rel {(new-WL_CUT)/WL_CUT:.3e})")

def solve(wl):
    s = PMM2DStackPure(PX, PX, n_superstrate=NSUP, n_substrate=NSUB,
                       n_modes=5, n_orders=3)
    s.add_layer(0.28e-6, eps_cell=CELL)
    s.set_source(wl)
    o, R, T, J = s.solve(jones=True)
    return R, T

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Rc, Tc = solve(WL_CUT)
    Rn, Tn = solve(new)
    Rb, Tb = solve(WL_CUT * (1 - 1e-9))
    Rb2, Tb2 = solve(WL_CUT * (1 - 1e-8))
print(f"solve(WL_CUT) vs solve(new): max|dR| = {np.max(np.abs(Rc-Rn)):.3e}  "
      f"bit-identical = {np.array_equal(Rc, Rn) and np.array_equal(Tc, Tn)}")
print(f"solve(new) vs solve(WL_CUT*(1-1e-9)): max|dR| = {np.max(np.abs(Rn-Rb)):.3e}")
print(f"solve(new) vs solve(WL_CUT*(1-1e-8)): max|dR| = {np.max(np.abs(Rn-Rb2)):.3e}")
print(f"max|R| = {np.max(np.abs(Rn)):.6f}  sumRT = {float(np.sum(Rn[0])+np.sum(Tn[0])):.12f}")
