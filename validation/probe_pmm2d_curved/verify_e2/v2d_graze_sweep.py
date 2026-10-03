"""V2d -- a GRAZING cut that the kernel's cell-piece search can MISS (Phase
E2 verifier, item 2).  The sinusoid x-wall map a (x = 0.55 + 0.12 sin(2 pi
y / 1.2), crest 0.67 at y = 0.3; no singular vertices) over an unmapped grid
whose x-wall sits at 0.67 - delta: the wall cuts a sliver of width delta and
length ~ 2 sqrt(2 delta / kappa) off the crest.  Kernel (adaptive) against
the physical brute force (v2_brute, converged to ~1e-15 on this family), per
delta; and the same with the wall's pull-back sampled finer (the
_CURVE_MORTAR_SAMPLES knob x 8) to show the cause.
"""
import warnings

import numpy as np
from _ve import dump
from v2_brute import blocks_err, brute_cross, grid, sin_map

from lumenairy.elements.pmm import _curvemortar as CMOR

rows = []
for d in (0.0, 1e-7, 1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3):
    ga = grid(sin_map(0.55, 0.12), 4)
    gb = grid(None, 4, walls=([0.0, 0.67 - d, 1.2], [0.0, 0.5, 1.2]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xk, nk, ck = CMOR.curved_cross_mass_adaptive(ga, gb)
        Xb = brute_cross(ga, gb, n=24, verbose=False)[0]
        s0 = CMOR._CURVE_MORTAR_SAMPLES
        CMOR._CURVE_MORTAR_SAMPLES = 8 * (s0 - 1) + 1
        try:
            Xf = CMOR.curved_cross_mass_adaptive(ga, gb)[0]
        finally:
            CMOR._CURVE_MORTAR_SAMPLES = s0
    e = blocks_err(Xk, Xb, ga.qq, gb.qq)["all"]
    ef = blocks_err(Xf, Xb, ga.qq, gb.qq)["all"]
    sliver = 2.0 * np.sqrt(2.0 * d / (0.12 * (2 * np.pi / 1.2) ** 2))
    rows.append(dict(delta=d, kernel_vs_brute=e, kernel_n=nk,
                     kernel_change=ck, samples_x8_vs_brute=ef,
                     sliver_length=float(sliver)))
    print(f"delta {d:.0e}: sliver {sliver:.2e}  kernel {e:.2e} (n {nk}, "
          f"change {ck:.0e})  samples x8 {ef:.2e}")
dump("v2d_graze_sweep", dict(rows=rows))
