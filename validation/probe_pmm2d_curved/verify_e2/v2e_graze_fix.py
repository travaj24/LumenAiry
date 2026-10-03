"""V2e -- the PROPOSED fix of V-E2-D1 (a grazing cut lost between two
samples of the pulled-back wall), applied IN-PROCESS to a copy of
``_curvemortar._cell_pieces`` (the library is not edited): between samples,
every local minimum of the curve's distance outside the cell that is small
is refined on a 257-point sub-grid, and a parameter that dips INSIDE the
cell is added to the sample set.  Measured on the v2d graze family (kernel
vs the physical brute force) and on the circle / crossing sinusoid pair
(the fix must not move a pair it does not concern).
"""
import inspect
import textwrap
import warnings

import numpy as np
from _ve import dump
from v2_brute import blocks_err, brute_cross, circle_map, grid, sin_map

from lumenairy.elements.pmm import _curvemortar as CMOR


def _grazing_refine(Pm, sx, sy, Om, e, tk):
    """``tk`` plus the parameter of every between-sample dip of the curve
    into cell (sx, sy), and every between-sample excursion out of it, that
    no sample sees."""
    U, V, _a, _b, ok = CMOR._pullback(Pm, sx, sy, Om, e, tk)
    hu = Pm.ub[sx + 1] - Pm.ub[sx]
    hv = Pm.vb[sy + 1] - Pm.vb[sy]

    def dout(U, V):
        return np.maximum.reduce([(Pm.ub[sx] - U) / hu,
                                  (U - Pm.ub[sx + 1]) / hu,
                                  (Pm.vb[sy] - V) / hv,
                                  (V - Pm.vb[sy + 1]) / hv])
    d = np.where(ok, dout(U, V), np.inf)
    extra = []
    for k in range(1, tk.size - 1):
        if 0.0 < d[k] < 0.05 and d[k] <= d[k - 1] and d[k] <= d[k + 1]:
            t = np.linspace(tk[k - 1], tk[k + 1], 257)
            u, v, _a, _b, okk = CMOR._pullback(Pm, sx, sy, Om, e, t)
            dd = np.where(okk, dout(u, v), np.inf)
            j = int(np.argmin(dd))
            if dd[j] < 0.0:
                extra.append(float(t[j]))
        elif -0.05 < d[k] <= 0.0 and d[k] >= d[k - 1] and d[k] >= d[k + 1]:
            # the mirror case: a brief EXCURSION out of the cell between two
            # inside samples (the run must be split there)
            t = np.linspace(tk[k - 1], tk[k + 1], 257)
            u, v, _a, _b, okk = CMOR._pullback(Pm, sx, sy, Om, e, t)
            dd = np.where(okk, dout(u, v), -np.inf)
            j = int(np.argmax(dd))
            if dd[j] > 0.0:
                extra.append(float(t[j]))
    return np.unique(np.r_[tk, extra]) if extra else tk


src = textwrap.dedent(inspect.getsource(CMOR._cell_pieces))
old = """        U, V, _du, _dv, ok = _pullback(Pm, sx, sy, Om, e, tk)
        ins = ok & Pm.inside(sx, sy, U, V)"""
new = """        tk = _grazing_refine(Pm, sx, sy, Om, e, tk0)
        K = tk.size
        U, V, _du, _dv, ok = _pullback(Pm, sx, sy, Om, e, tk)
        ins = ok & Pm.inside(sx, sy, U, V)"""
assert old in src
src = src.replace(old, new).replace("    out = []\n    for e, ebox in edges:",
                                    "    out = []\n    tk0 = tk\n    for e, ebox in edges:")
ns = dict(vars(CMOR))
ns["_grazing_refine"] = _grazing_refine
exec(src, ns)
fixed = ns["_cell_pieces"]
orig = CMOR._cell_pieces

rows = []
cases = [("sx", d) for d in (0.0, 1e-7, 1e-6, 3e-5)] + [("circ", d) for d in
                                                         (1e-6,)]
for kind, d in cases:
    if kind == "sx":
        ga = grid(sin_map(0.55, 0.12), 4)
        gb = grid(None, 4, walls=([0.0, 0.67 - d, 1.2], [0.0, 0.5, 1.2]))
    else:
        ga = grid(circle_map(), 4)
        gb = grid(None, 4, walls=([0.0, 0.45, 1.2], [0.0, 0.24 + d, 1.2]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xs = CMOR.curved_cross_mass_adaptive(ga, gb)[0]
        CMOR._cell_pieces = fixed
        try:
            Xf, nf, cf = CMOR.curved_cross_mass_adaptive(ga, gb)
        finally:
            CMOR._cell_pieces = orig
        Xb = brute_cross(ga, gb, n=20, verbose=False)[0]
    r = dict(case=kind, delta=d,
             shipped=blocks_err(Xs, Xb, ga.qq, gb.qq)["all"],
             fixed=blocks_err(Xf, Xb, ga.qq, gb.qq)["all"], fixed_n=nf,
             fixed_change=cf, shipped_vs_fixed=float(
                 np.max(np.abs(Xs - Xf)) / np.max(np.abs(Xs))))
    rows.append(r)
    print(r)
# a pair the fix does not concern must not move
ga, gb = grid(circle_map(), 4), grid(sin_map(), 4)
Xs = CMOR.curved_cross_mass(ga, gb, 18)
CMOR._cell_pieces = fixed
Xf = CMOR.curved_cross_mass(ga, gb, 18)
CMOR._cell_pieces = orig
unmoved = float(np.max(np.abs(Xs - Xf)))
print("circle / crossing sinusoid moved by", unmoved)
dump("v2e_graze_fix", dict(rows=rows, circ_sin_moved=unmoved))
