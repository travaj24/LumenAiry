"""V4 -- the Rayleigh cut-off WARNING, fixture by fixture, on whichever arm is
passed (Task G.d), plus an ADVERSARIAL search for a geometry where the new
nudge moves the warning boundary.

The warning keys on `_gap = min over the two HALF-SPACES of |Re eps - kt^2|`
computed from the NUDGED wavelength, and fires below 1e-4.  Unifying the eps
list cannot change which half-spaces are looked at -- but it CAN change `wl`,
hence `kt^2`, hence `_gap`, on a geometry that sits on a LAYER cut-off.  The
last block constructs exactly that borderline case.

    python v4_warn.py <lumenairy-root>
"""
import os
import sys
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
)

M, NO = 5, 3
CELL = np.array([[4.0, 1.0], [1.0, 1.0]], dtype=complex)


def cap(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fn()
    return [str(x.message) for x in w if "Rayleigh cut" in str(x.message)]


def eff(px, wl, n_sup=1.0, n_sub=1.5, theta=0.0, phi=0.0, cell=CELL, py=None):
    def go():
        pmm_efficiency_2d_staggered(px, px if py is None else py, cell, n_sub,
                                    n_sup, 0.28e-6, wl, degree=M, n_orders=NO,
                                    polarization="te", theta=theta, phi=phi)
    return go


def stk(px, wl, n_sup=1.0, n_sub=1.5, theta=0.0, phi=0.0, cell=CELL,
        uniform=None):
    def go():
        s = PMM2DStackPure(px, px, n_superstrate=n_sup, n_substrate=n_sub,
                           n_modes=M, n_orders=NO)
        if uniform is not None:
            s.add_layer(0.15e-6, eps=uniform)
        s.add_layer(0.28e-6, eps_cell=cell)
        s.set_source(wl, theta=theta, phi=phi)
        s.solve(jones=True)
    return go


PX = 0.5e-6
WL_LAYER = PX * float(np.sqrt(4.0))          # LAYER cut-off (eps 4)
WL_SUP = 0.8e-6                              # wl = px on a px = 0.8 grating

FIX = [
    ("w01 eff ordinary", eff(0.8e-6, 0.633e-6)),
    ("w02 eff conical ordinary", eff(0.8e-6, 0.633e-6, theta=0.25, phi=0.4)),
    ("w03 eff wl = px (SUPERSTRATE cut-off)", eff(0.8e-6, WL_SUP, n_sup=1.0)),
    ("w04 eff wl = 0.99999 px (inside the band)",
     eff(0.8e-6, 0.8e-6 * 0.99999)),
    ("w05 eff LAYER-only cut-off", eff(PX, WL_LAYER)),
    ("w06 eff SUBSTRATE cut-off (n_sub = 1.5)",
     eff(PX, PX * 1.5, n_sub=1.5)),
    ("w07 eff rect periods ordinary", eff(0.8e-6, 0.55e-6, py=0.6e-6)),
    ("w08 stack ordinary", stk(0.8e-6, 0.633e-6)),
    ("w09 stack on a LAYER cut-off", stk(PX, WL_LAYER)),
    ("w10 stack on the SUBSTRATE cut-off", stk(PX, PX * 1.5)),
    ("w11 stack uniform+patterned, uniform on ITS cut-off",
     stk(PX, PX * float(np.sqrt(6.25)), uniform=6.25 + 0j, n_sub=1.5)),
    ("w12 stack oblique ordinary", stk(0.8e-6, 0.633e-6, theta=0.2, phi=0.5)),
]

for name, fn in FIX:
    msgs = cap(fn)
    short = [m.split("; ")[0].split(": ", 1)[-1] for m in msgs]
    print(f"{name:48s} n={len(msgs)}  {short}")

# --- adversarial: does the nudge move the 1e-4 warning boundary? ----------
# Put order (1,0) exactly on the LAYER cut-off (eps_L = 4 -> wl/px = 2) AND
# choose n_sub so that a half-space gap sits just inside 1e-4.  The nudge
# shifts kt^2 by ~2e-7 * kt^2 = 8e-7, so a half-space gap in
# [1e-4 - 8e-7, 1e-4] flips OFF when the nudge fires.
print()
print("adversarial: layer cut-off AND a half-space gap at the 1e-4 boundary")
for target in (1.0e-4, 0.999e-4, 0.9996e-4, 1.0004e-4, 1.001e-4):
    # kt^2 for order (1,0) at the un-nudged wl is exactly 4.0; want
    # |eps_sub - kt^2| = target -> eps_sub = 4 + target (kt^2 grows on nudge,
    # so the gap SHRINKS ... choose eps_sub above kt^2 so the nudge moves kt^2
    # toward it)
    eps_sub = 4.0 + target
    n_sub = float(np.sqrt(eps_sub))
    msgs = cap(eff(PX, WL_LAYER, n_sub=n_sub))
    print(f"  eps_sub = 4 + {target:.4e}  n_sub = {n_sub!r}  "
          f"warnings = {len(msgs)}  "
          f"{[m.split('within ')[1].split(' (')[0] for m in msgs]}")

print()
print("adversarial, other side: eps_sub BELOW kt^2 (nudge widens the gap)")
for target in (0.999e-4, 0.9996e-4, 1.0e-4, 1.0004e-4):
    eps_sub = 4.0 - target
    n_sub = float(np.sqrt(eps_sub))
    msgs = cap(eff(PX, WL_LAYER, n_sub=n_sub))
    print(f"  eps_sub = 4 - {target:.4e}  n_sub = {n_sub!r}  "
          f"warnings = {len(msgs)}  "
          f"{[m.split('within ')[1].split(' (')[0] for m in msgs]}")
