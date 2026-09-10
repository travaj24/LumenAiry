"""F5 follow-up -- ATTRIBUTION of the near-coincident-wall blow-up.

``f5d_diag.py`` [D] and ``f5e_nearwall.py`` read an error EXPLOSION when the
two slices' walls differ by 1e-4 .. 1e-5 of the period, energy-invisible.  The
candidate is the non-uniform mortar's integration mesh -- but the ORACLE is a
1-D ``PMMStack`` on its DEFAULT ``layer_grids='shared'`` path, which unions the
two layers' walls onto ONE nodal-SEM grid and therefore grows a SLIVER element
of exactly that width.  Sliver elements are the documented 1-D pathology.

So this measures the oracle's OWN self-gap (degree 12 vs 14) alongside the
comparison, which is the only honest way to read any of it: an oracle whose
self-gap exceeds the measurement is not an oracle at that point.
"""
import json
import os
import warnings

import numpy as np
from mortar2d import guard
print("lumenairy:", guard(), flush=True)
from nonuniform import MortarStackNU
from lumenairy import PMMStack

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PX = PY = 1.2e-6
WL = 0.85e-6
THETA, NORD = 0.15, 2
EPS_H, EPS_P = 2.25, 9.0
dz = 0.32e-6 / 4
XB0, XB1 = (0.1873, 0.7241), (0.2917, 0.6109)
zf = 1.0 - 0.5 / 4
a0 = XB0[0] + (XB1[0] - XB0[0]) * zf
b0 = XB0[1] + (XB1[1] - XB0[1]) * zf
print(f"slice-1 walls: {a0:.10f} {b0:.10f}", flush=True)
TILE = np.empty((3, 3), complex)
TILE[0, :] = EPS_H
TILE[1, :] = EPS_P
TILE[2, :] = EPS_H


def orc(fr, deg=14, per_layer=False):
    kw = dict(layer_grids="per-layer") if per_layer else {}
    s = PMMStack(PX, n_superstrate=1.0, n_substrate=1.0, degree=deg, **kw)
    for (a, b) in fr:
        s.add_layer(dz, segments=[(a, EPS_H), (b - a, EPS_P), (1.0 - b, EPS_H)])
    s.set_source(WL, theta=THETA)
    o, R, T = s.solve()[:3]
    o = np.asarray(o).ravel()
    i = np.argsort(o)
    return o[i], R[1][i], T[1][i]


def pure(fr, M):
    s = MortarStackNU(PX, PY, n_modes=M, n_orders=NORD)
    for (a, b) in fr:
        s.add_layer(dz, eps_cell=TILE, x_walls=[a * PX, b * PX],
                    y_walls=[a * PY, b * PY])
    s.set_source(WL, theta=THETA, phi=0.0)
    o, R, T = s.solve(jones=False)
    oo = np.asarray(o)
    sel = oo[:, 1] == 0
    m = oo[sel, 0]
    i = np.argsort(m)
    return m[i], R[1][sel][i], T[1][sel][i]


def gap(m, R, T, MO, RO, TO):
    k = np.isin(MO, m)
    return float(max(np.abs(R - RO[k]).max(), np.abs(T - TO[k]).max()))


M0, R0, T0 = orc([(a0, b0), (a0, b0)])
rows = []
hdr = (f"{'delta':>10} {'orc SHARED':>11} {'orc PER-LAYER':>13} "
       f"{'orc(d)-orc(0)':>14} {'pure M7 - orc(d)':>17} "
       f"{'pure M7 - orcPL(d)':>19} {'pure M7 - orc(0)':>17}")
print("\n  (the two 'orc' columns are the ORACLE's own degree-12-vs-14 "
      "self-gap on its\n   shared-grid and per-layer-grid paths)", flush=True)
print(hdr, flush=True)
for d in (1e-2, 2.6e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 0.0):
    fr = [(a0, b0), (a0 - d, b0 + d)]
    MO, RO, TO = orc(fr)
    _m, R12, T12 = orc(fr, 12)
    sg = float(max(np.abs(RO - R12).max(), np.abs(TO - T12).max()))
    try:
        MP, RP, TP = orc(fr, 14, per_layer=True)
        _m, RP12, TP12 = orc(fr, 12, per_layer=True)
        sgp = float(max(np.abs(RP - RP12).max(), np.abs(TP - TP12).max()))
    except Exception as exc:                                  # noqa: BLE001
        MP, RP, TP, sgp = None, None, None, float("nan")
        print("   per-layer oracle refused:", type(exc).__name__, exc)
    dorc = float(max(np.abs(RO - R0).max(), np.abs(TO - T0).max()))
    m7, r7, t7 = pure(fr, 7)
    e_sh = gap(m7, r7, t7, MO, RO, TO)
    e_pl = (float("nan") if MP is None else gap(m7, r7, t7, MP, RP, TP))
    e_0 = gap(m7, r7, t7, M0, R0, T0)
    rows.append(dict(delta=d, oracle_selfgap_shared=sg,
                     oracle_selfgap_perlayer=sgp, oracle_shift=dorc,
                     pure_vs_oracle_shared=e_sh,
                     pure_vs_oracle_perlayer=e_pl, pure_vs_oracle_delta0=e_0))
    print(f"{d:10.2e} {sg:11.2e} {sgp:13.2e} {dorc:14.2e} {e_sh:17.2e} "
          f"{e_pl:19.2e} {e_0:17.2e}", flush=True)
    json.dump(rows, open(os.path.join(HERE, "f5f_attrib.json"), "w"), indent=1)
print("\nwrote f5f_attrib.json", flush=True)
