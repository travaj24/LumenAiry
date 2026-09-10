"""V1b -- LOCALISATION control for V1.

V1 reports the two arms BIT-IDENTICAL on 12 off-cut-off fixtures.  That is only
evidence if the same comparison CAN see the one change in the range that is
allowed to move a nonmagnetic number: the Wood-anomaly nudge list now carries a
SCALAR layer's permittivities (``_wood_eps_reals``), so an order sitting
EXACTLY on a layer cut-off is nudged in the worktree and was not in main.

This probe puts a scalar layer exactly on the ``(2, 0)`` cut-off at normal
incidence (``eps_layer = (2 wl / px)^2`` to the last bit) and prints the same
hashes.  Expected: main != worktree HERE, and identical on the detuned twin --
which localises any V1 difference to the Wood list rather than the magnetic
build.
"""
import hashlib
import sys

import numpy as np

import lumenairy
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure

_ARMS = {"main": "D:", "wt": "C:\\tmp\\lum_vmag"}


def _h(a):
    arr = np.ascontiguousarray(a)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:24]


def run(on_cut):
    wl, px = 0.55e-6, 0.90e-6
    kt2 = (2.0 * (wl / px)) ** 2          # order (2, 0) at normal incidence
    eps_layer = kt2 if on_cut else kt2 * 1.02
    cell = np.full((2, 2), 2.10 + 0.0j)
    cell[0, 0] = eps_layer                # ON the (2,0) cut-off, to the bit
    st = PMM2DStackPure(px, px, n_superstrate=1.0, n_substrate=1.0,
                        n_modes=5, n_orders=4)
    st.add_layer(0.30e-6, eps_cell=cell)
    st.set_source(wl, theta=0.0, phi=0.0)
    o, r, t, j = st.solve(jones=True)
    return {"gap": f"{abs(eps_layer - kt2):.3e}", "R": _h(r), "T": _h(t),
            "J": _h(j), "sumRT": repr(float(np.sum(r) + np.sum(t)))}


def main():
    arm = sys.argv[1]
    assert lumenairy.__file__.startswith(_ARMS[arm]), lumenairy.__file__
    for tag, on_cut in (("ON_CUTOFF", True), ("DETUNED_2pct", False)):
        print(arm, tag, run(on_cut))


if __name__ == "__main__":
    main()
