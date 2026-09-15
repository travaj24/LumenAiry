"""D3 fixture-bar probe: two-sided placement of the "the mirror-symmetrisation
of the fixture tensor is ROUND-OFF, not structure" guard.

Measures
  * the raw x<->y mirror defect of ``uniaxial_tensor(2.0, 2.6, 40, phi)`` as
    ``phi`` walks away from 45 deg -- the magnitude a REAL fixture change
    (a director no longer in the x = y plane) puts on the same reading;
  * the same for a wrong component permutation;
  * the SENSITIVITY of the quantity the D3 test then asserts,
    ``|Jxx - Jyy|`` at ``_D3_SYM_BAR = 1e-12``, to a fixture mirror defect of
    size delta -- so the fixture bar can be placed below the fixture defect
    that would move the Jones reading by 1e-12.

Usage:  python probe_d3_sensitivity.py <out.json>
"""
import json
import os
import sys
import warnings

import numpy as np

import lumenairy
from lumenairy.elements.rcwa import rcwa_jones_2d, uniaxial_tensor

assert "lum_reds" in lumenairy.__file__, lumenairy.__file__

_PERM = [1, 0, 2]
S = 96
D3 = dict(period=0.5e-6, depth=0.3e-6, wl=0.633e-6)


def _mirror3(t):
    return t[_PERM, :][:, _PERM]


def _mirror(cell):
    return np.transpose(cell, (1, 0, 2, 3))[:, :, _PERM, :][:, :, :, _PERM]


def _cell(kind, tilt):
    ii, jj = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    c = np.zeros((S, S, 3, 3), dtype=complex)
    c[:] = np.eye(3)
    if kind == "square":
        m = (np.abs(ii - S // 2) < S // 5) & (np.abs(jj - S // 2) < S // 5)
    else:
        m = ((ii - (S - 1) / 2.0) ** 2
             + (jj - (S - 1) / 2.0) ** 2) <= (0.25 * S) ** 2
    c[m] = tilt
    return c


def jones(cell, M=3):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rcwa_jones_2d(D3["period"], D3["period"], cell, 1.5, 1.0,
                             D3["depth"], D3["wl"], theta=0.0, phi=0.0,
                             n_orders_x=M, n_orders_y=M,
                             formulation="fff_nv")[3]


out = {"python": sys.version.split()[0], "numpy": np.__version__,
       "env": {k: os.environ.get(k) for k in
               ("OPENBLAS_CORETYPE", "OMP_NUM_THREADS")}}

D45 = uniaxial_tensor(2.0, 2.6, np.deg2rad(40.0), phi=np.deg2rad(45.0))
out["raw_defect_at_45deg"] = float(np.max(np.abs(D45 - _mirror3(D45))))
out["tensor_scale"] = float(np.max(np.abs(D45)))

# ---- 1. how big is the defect when the DIRECTOR really leaves x = y? -------
walk = {}
for dphi in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 5.0):
    D = uniaxial_tensor(2.0, 2.6, np.deg2rad(40.0),
                        phi=np.deg2rad(45.0 + dphi))
    walk[f"+{dphi}deg"] = float(np.max(np.abs(D - _mirror3(D))))
out["defect_vs_dphi_deg"] = walk

# ---- 2. a wrong component permutation -------------------------------------
out["defect_with_identity_perm"] = float(np.max(np.abs(D45 - D45)))   # 0 by defn
out["defect_xz_vs_zx(non-symmetric tensor)"] = float(
    np.max(np.abs(D45 - D45.T)))

# ---- 3. Jones sensitivity to a fixture mirror defect of size delta ---------
Dsym = 0.5 * (D45 + _mirror3(D45))
sens = {}
for kind in ("square", "disk"):
    base = jones(_cell(kind, Dsym))
    row = {"base_abs_J00_minus_J11": float(abs(base[0, 0] - base[1, 1])),
           "base_abs_J01_minus_J10": float(abs(base[0, 1] - base[1, 0])),
           "jones_scale": float(np.max(np.abs(base)))}
    for delta in (1e-14, 1e-12, 1e-10, 1e-8, 1e-6):
        D = Dsym.copy()
        D[0, 0] += delta                     # break the mirror by exactly delta
        J = jones(_cell(kind, D))
        row[f"delta={delta:g}"] = dict(
            fixture_defect=float(np.max(np.abs(D - _mirror3(D)))),
            dJ_diag=float(abs(J[0, 0] - J[1, 1])),
            dJ_offdiag=float(abs(J[0, 1] - J[1, 0])))
    sens[kind] = row
out["jones_sensitivity"] = sens

json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))
