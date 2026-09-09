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
import lumenairy.elements.rcwa._core as _rc
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure

_O = _rc._grazing_safe_wavelength
_S = []
def spy(wl, *a, **kw):
    out = _O(wl, *a, **kw)
    _S.append((float(wl), float(out), len(a[-1])))
    return out
_rc._grazing_safe_wavelength = spy

# uniform SCALAR layer eps = 4, half-spaces 1.0 / 1.5 (eps 1.0 / 2.25);
# px = 0.5 um, wl = 1.0 um -> order m=1 sits at kt^2 = 4 = the LAYER's eps.
def run(tensor):
    s = PMM2DStackPure(0.5e-6, 0.5e-6, n_superstrate=1.0, n_substrate=1.5,
                       n_modes=5, n_orders=3)
    e = 4.0 + 0j
    s.add_layer(0.28e-6, eps=(np.eye(3) * e if tensor else e))
    s.add_layer(0.10e-6, eps_cell=np.full((2, 2), 2.0 + 0j))
    s.set_source(1.0e-6)
    o, R, T, J = s.solve(jones=True)
    return R, T

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Rs, Ts = run(False)
    ws = _S[-1]
    Rt, Tt = run(True)
    wt = _S[-1]
print("uniform SCALAR layer eps=4 ON its own cutoff:")
print(f"  scalar arm: list_len={ws[2]} wl {ws[0]:.17e} -> {ws[1]:.17e} rel {(ws[1]-ws[0])/ws[0]:+.3e}")
print(f"  tensor arm: list_len={wt[2]} wl {wt[0]:.17e} -> {wt[1]:.17e} rel {(wt[1]-wt[0])/wt[0]:+.3e}")
print(f"  max|dR| scalar-vs-tensor = {np.max(np.abs(Rs-Rt)):.3e}   sumR+T scalar = {float(np.sum(Rs[0])+np.sum(Ts[0])):.12f}")
