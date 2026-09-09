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
from lumenairy.elements.pmm.twod_staggered import pmm_efficiency_2d_staggered

_O = _rc._grazing_safe_wavelength
_S = []
def spy(wl, *a, **kw):
    out = _O(wl, *a, **kw)
    _S.append((float(wl), float(out), len(a[-1]) if isinstance(a[-1], list) else -1))
    return out
_rc._grazing_safe_wavelength = spy

def show(tag):
    wl0, wl1, n = _S[-1]
    print(f"  {tag}: list_len={n}  wl {wl0:.17e} -> {wl1:.17e}  rel {(wl1-wl0)/wl0:+.3e}")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    s = PMM2DStackPure(1.0e-6, 1.0e-6, n_superstrate=1.0, n_substrate=1.5, n_modes=5, n_orders=3)
    s.add_layer(0.15e-6, eps=2.25)
    s.add_layer(0.28e-6, eps_cell=np.ones((2, 2), dtype=complex))
    s.set_source(1.5e-6)
    s.solve(jones=True)
    show("stack: uniform SCALAR eps=2.25 on its own cutoff (kt=1.5)")

    pmm_efficiency_2d_staggered(0.5e-6, 0.5e-6, np.full((2, 2), 4.0 + 0j),
                                1.5, 1.0, 0.28e-6, 1.0e-6, degree=5,
                                n_orders=3, polarization="tm")
    show("single-layer scalar entry, uniform eps=4 cell on its cutoff")

    pmm_efficiency_2d_staggered(0.8e-6, 0.8e-6,
                                np.array([[2.25, 1.0], [1.0, 1.0]], dtype=complex),
                                1.45, 1.0, 0.3e-6, 0.633e-6, degree=5,
                                n_orders=3, polarization="te")
    show("single-layer scalar entry, ordinary fixture (no cutoff)")
