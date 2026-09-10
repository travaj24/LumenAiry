"""Gate 3 -- the Rayleigh-cutoff WARNING behaviour, fixture by fixture."""
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
from lumenairy.elements.pmm.twod_staggered import pmm_efficiency_2d_staggered

PILLAR = np.array([[2.25, 1.0], [1.0, 1.0]], dtype=complex)
UNIF4 = np.full((2, 2), 4.0 + 0j)

def cap(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fn()
    return [f"{x.category.__name__}: {str(x.message)[:70]}" for x in w]

CASES = {
 "ordinary 0.633/0.8": lambda: pmm_efficiency_2d_staggered(
     0.8e-6, 0.8e-6, PILLAR, 1.45, 1.0, 0.3e-6, 0.633e-6, degree=5,
     n_orders=3, polarization="te"),
 "half-space cutoff wl=px (m=1 grazes sup)": lambda: pmm_efficiency_2d_staggered(
     0.8e-6, 0.8e-6, PILLAR, 1.0, 1.0, 0.3e-6, 0.8e-6, degree=5,
     n_orders=3, polarization="te"),
 "inside the 1e-4 band (wl = 0.9999 px)": lambda: pmm_efficiency_2d_staggered(
     0.8e-6, 0.8e-6, PILLAR, 1.0, 1.0, 0.3e-6, 0.8e-6 * 0.99999, degree=5,
     n_orders=3, polarization="te"),
 "LAYER cutoff only (eps=4 cell, wl/px=2)": lambda: pmm_efficiency_2d_staggered(
     0.5e-6, 0.5e-6, UNIF4, 1.5, 1.0, 0.28e-6, 1.0e-6, degree=5,
     n_orders=3, polarization="tm"),
 "oblique conical ordinary": lambda: pmm_efficiency_2d_staggered(
     0.8e-6, 0.8e-6, PILLAR, 1.45, 1.0, 0.3e-6, 0.633e-6, degree=5,
     n_orders=3, polarization="te", theta=0.25, phi=0.4),
}
def _stack():
    s = PMM2DStackPure(0.8e-6, 0.8e-6, n_superstrate=1.0, n_substrate=1.45,
                       n_modes=5, n_orders=3)
    s.add_layer(0.2e-6, eps=2.1)
    s.add_layer(0.3e-6, eps_cell=PILLAR)
    s.set_source(0.633e-6)
    s.solve(jones=False)
CASES["stack ordinary"] = _stack
def _stack_cut():
    s = PMM2DStackPure(0.5e-6, 0.5e-6, n_superstrate=1.0, n_substrate=1.5,
                       n_modes=5, n_orders=3)
    s.add_layer(0.28e-6, eps=4.0)
    s.set_source(1.0e-6)
    s.solve(jones=True)
CASES["stack, uniform eps=4 layer ON its cutoff"] = _stack_cut

for name, fn in CASES.items():
    print(f"{name}: {cap(fn)}")
