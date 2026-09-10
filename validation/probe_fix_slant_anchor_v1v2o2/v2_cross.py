"""V2 -- the CROSS-ENGINE sign proof, on a y-uniform cell.

The 1-D staircase (``v2_derive.py``) is the same engine as the thing under
test, so it can settle the sign but not the CONVENTION: if the 1-D shear law
and the 1-D anchor were both wrong in the same direction the staircase would
still agree.  Two INDEPENDENT engines, whose transmitted amplitudes are
already LAB-referenced (the 2-D hybrid since the 2026-09-11 anchor, the 2-D
pure since its own ``solve``-level walk sum), settle it:

  * ``PMM2DStackHybrid`` -- FMM-floored Fourier hybrid, 4N generator;
  * ``PMM2DStackPure``   -- the no-floor staggered nodal engine.

Both are handed the SAME parallelogram as a y-uniform 2-D cell.  The
comparison is the ZEROTH-order transmitted Jones, which is INVARIANT under a
lateral translation of the whole structure (order ``m`` picks up
``exp(i(alpha_0 - alpha_m) delta)``, which is 1 at ``m = 0``), so the engines'
differing ``centre`` conventions cannot contaminate it.

The 1-D fixture is centred so its TOP face -- the frame anchor -- puts the
ridge at 0.50 of the period, which is exactly where the 2-D cells put it.
Both 1-D grid routes are run: ``layer_grids='shared'`` (the general shared-grid
cascade) and ``'per-layer'`` (the general per-layer cascade), which store their
modal dict at two DIFFERENT sites.
"""
from __future__ import annotations

import math
import sys
import time
import warnings

import _lib as L
import numpy as np

P = 0.80e-6
WL = 0.55e-6
D = 0.40e-6
ER, EG = 4.20, 1.45
DUTY, SHEAR = 0.50, 0.25
CENTRE = 0.625                 # -> TOP-face ridge centre 0.500
NSUP, NSUB = 1.0, 1.6
TH = math.radians(25.0)
TX = SHEAR * P / D             # tan(slant_angle)
W = SHEAR * P                  # the walk, metres


def oned(*, layer_grids="shared", degree=12, n_orders=7):
    from lumenairy.elements.pmm.stack import PMMStack
    st = PMMStack(P, n_superstrate=NSUP, n_substrate=NSUB, degree=degree,
                  n_orders=n_orders, factorization="convection",
                  layer_grids=layer_grids)
    st.add_sheared_grating(D, eps_ridge=ER, eps_groove=EG, duty=DUTY,
                           shear=SHEAR, centre=CENTRE)
    st.set_source(WL, theta=TH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st.solve()
    return st


def _stripe(nx, ny=None):
    ny = ny or nx
    c = np.full((nx, ny), EG, dtype=complex)
    lo = int(round((0.5 - DUTY / 2) * nx))
    hi = int(round((0.5 + DUTY / 2) * nx))
    c[lo:hi, :] = ER
    return c


def hybrid(n_orders=5, nx=40, ny=4):
    from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid
    st = PMM2DStackHybrid(P, P, n_superstrate=NSUP, n_substrate=NSUB,
                          n_orders=n_orders)
    st.add_layer(D, eps_cell=_stripe(nx, ny), slant=(TX, 0.0))
    st.set_source(WL, theta=TH, phi=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st.solve()
    return st


def pure(n_modes=6, n_orders=3, nx=4):
    from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
    st = PMM2DStackPure(P, P, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=n_modes, n_orders=n_orders)
    st.add_layer(D, eps_cell=_stripe(nx), slant=(TX, 0.0))
    st.set_source(WL, theta=TH, phi=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st.solve()
    return st


def main():
    t0 = time.time()
    k0 = 2.0 * np.pi / WL
    a0 = float(np.real(NSUP) * math.sin(TH))
    P0 = complex(np.exp(1j * k0 * a0 * W))
    res = dict(fixture=dict(period_um=P * 1e6, wl_um=WL * 1e6, d_um=D * 1e6,
                            eps=(ER, EG), duty=DUTY, shear=SHEAR,
                            centre=CENTRE, top_centre=CENTRE - SHEAR / 2,
                            tan_slant=TX, walk_um=W * 1e6, theta_deg=25.0),
               P0=dict(re=P0.real, im=P0.imag, arg=float(np.angle(P0))))

    refs = {}
    try:
        refs["hybrid_n_orders7"] = np.asarray(hybrid(7).jones_transmission())
        refs["hybrid_n_orders9"] = np.asarray(hybrid(9).jones_transmission())
    except Exception as e:                                # noqa: BLE001
        res["hybrid_error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
    try:
        p = pure()
        refs["pure_n_modes6"] = np.asarray(p.jones_transmission())
        p8 = pure(n_modes=8)
        refs["pure_n_modes8"] = np.asarray(p8.jones_transmission())
    except Exception as e:                                # noqa: BLE001
        res["pure_error"] = "%s: %s" % (type(e).__name__, str(e)[:200])

    if "hybrid_n_orders7" in refs and "hybrid_n_orders9" in refs:
        res["hybrid_own_n_orders_step"] = L.resid(refs["hybrid_n_orders7"],
                                                  refs["hybrid_n_orders9"])
    if "pure_n_modes6" in refs and "pure_n_modes8" in refs:
        res["pure_own_n_modes_step"] = L.resid(refs["pure_n_modes6"],
                                               refs["pure_n_modes8"])

    arms = {}
    for grids in ("shared", "per-layer"):
        try:
            st = oned(layer_grids=grids)
            J = L.jt(st)
        except Exception as e:                            # noqa: BLE001
            arms["oned_%s" % grids] = "%s: %s" % (type(e).__name__,
                                                  str(e)[:180])
            continue
        row = {}
        for rname, R in refs.items():
            row[rname] = dict(as_returned=L.resid(J, R),
                              x_P0=L.resid(J * P0, R),
                              x_conj_P0=L.resid(J * np.conj(P0), R),
                              div_P0=L.resid(J / P0, R))
        arms["oned_%s" % grids] = row
    res["arms"] = arms
    res["seconds"] = round(time.time() - t0, 1)
    for k, v in arms.items():
        print("==", k)
        if isinstance(v, str):
            print("   ", v)
            continue
        for rn, r in v.items():
            print("   %-20s as-ret %.5e  x P0 %.5e  x conj %.5e  / P0 %.5e"
                  % (rn, r["as_returned"], r["x_P0"], r["x_conj_P0"],
                     r["div_P0"]))
    print("hybrid own step:", res.get("hybrid_own_n_orders_step"))
    print("pure own step  :", res.get("pure_own_n_modes_step"))
    L.dump("v2_cross", res, suffix=(sys.argv[1] if len(sys.argv) > 1 else ""))


if __name__ == "__main__":
    main()
