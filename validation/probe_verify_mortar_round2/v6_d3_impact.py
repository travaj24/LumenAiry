"""VERIFY round 2 -- how much does D3 MOVE an ordinary per-layer answer?

The shipped bit-identity claim is scoped to INTEGER lattices.  This measures
what the per-segment quadrature rule does to the numbers a user of
``layer_grids='per-layer'`` with NON-UNIFORM walls actually gets, at the
DEFAULT ``n_orders``, on both arms.

    V6_TAG=with PYTHONPATH=/c/tmp/lum_vmortar2 python v6_d3_impact.py
    V6_TAG=pre  PYTHONPATH=/c/tmp/lum_prem2 V6_EXPECT_ROOT=/c/tmp/lum_prem2 \
                python v6_d3_impact.py
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = os.environ.get("V6_TAG", "with")
EXPECT = os.environ.get("V6_EXPECT_ROOT")
if EXPECT:
    assert os.path.abspath(lumenairy.__file__).lower().startswith(
        os.path.abspath(EXPECT).lower()), lumenairy.__file__
print(f"[{TAG}] lumenairy = {lumenairy.__file__}", flush=True)
_C = complex
P = 1.0e-6
WL = 0.62e-6


def _tile(eh=2.25, ep=6.0):
    c = np.full((3, 3), _C(eh))
    c[1, 1] = _C(ep)
    return c


def _solve(xa, xb, M, n_orders, uniform_array=False, integer=False):
    s = PMM2DStackPure(P, n_modes=M, n_orders=n_orders, n_substrate=1.5,
                       layer_grids="per-layer")
    if integer:
        s.add_layer(0.14e-6, eps_cell=_tile())
        s.add_layer(0.11e-6, eps_cell=_tile(ep=4.0))
    elif uniform_array:
        u = [P / 3.0, 2.0 * P / 3.0]
        s.add_layer(0.14e-6, eps_cell=_tile(), x_walls=u, y_walls=u)
        s.add_layer(0.11e-6, eps_cell=_tile(ep=4.0), x_walls=u, y_walls=u)
    else:
        s.add_layer(0.14e-6, eps_cell=_tile(), x_walls=xa, y_walls=xa)
        s.add_layer(0.11e-6, eps_cell=_tile(ep=4.0), x_walls=xb, y_walls=xb)
    s.set_source(WL, theta=0.19, phi=0.4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T = s.solve(jones=False)
    o = np.asarray(o)
    z = np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0]
    R, T = np.atleast_2d(R), np.atleast_2d(T)
    return dict(R00=[float(R[0, z]), float(R[1, z])],
                T00=[float(T[0, z]), float(T[1, z])],
                Rsum=[float(R[0].sum()), float(R[1].sum())],
                Tsum=[float(T[0].sum()), float(T[1].sum())])


XA = [0.2371 * P, 0.6183 * P]
XB = [0.3117 * P, 0.7402 * P]
LONG_A = [0.02 * P, 0.98 * P]          # a 0.96 d segment
LONG_B = [0.045 * P, 0.955 * P]

OUT = {}
for name, kw in (
        ("nonconf_M4_cap4", dict(xa=XA, xb=XB, M=4, n_orders=4)),
        ("nonconf_M6_cap7", dict(xa=XA, xb=XB, M=6, n_orders=7)),
        ("nonconf_M4_no2", dict(xa=XA, xb=XB, M=4, n_orders=2)),
        ("longseg_M4_no2", dict(xa=LONG_A, xb=LONG_B, M=4, n_orders=2)),
        ("longseg_M6_no3", dict(xa=LONG_A, xb=LONG_B, M=6, n_orders=3)),
        ("longseg_M8_no4", dict(xa=LONG_A, xb=LONG_B, M=8, n_orders=4)),
        ("taperslice_M6_cap7", dict(xa=[0.22 * P, 0.70 * P],
                                    xb=[0.24 * P, 0.68 * P], M=6,
                                    n_orders=7)),
        ("uniformarray_M4_cap4", dict(xa=XA, xb=XB, M=4, n_orders=4,
                                      uniform_array=True)),
        ("integer_M4_cap4", dict(xa=XA, xb=XB, M=4, n_orders=4,
                                 integer=True)),
):
    try:
        OUT[name] = _solve(**kw)
        print(f"  [{TAG}] {name}: R00 = {OUT[name]['R00']}", flush=True)
    except Exception as exc:                            # noqa: BLE001
        OUT[name] = {"error": f"{type(exc).__name__}: {str(exc)[:70]}"}
        print(f"  [{TAG}] {name}: {OUT[name]['error']}", flush=True)

p = os.path.join(HERE, f"v6_d3_impact_{TAG}.json")
with open(p, "w") as fh:
    json.dump({"tag": TAG, "lumenairy": lumenairy.__file__, "rows": OUT}, fh,
              indent=1)
print(f"[{TAG}] wrote {p}")

if len(sys.argv) > 1 and sys.argv[1] == "compare":
    a = json.load(open(os.path.join(HERE, "v6_d3_impact_with.json")))["rows"]
    b = json.load(open(os.path.join(HERE, "v6_d3_impact_pre.json")))["rows"]
    for k in a:
        if "error" in a[k] or "error" in b[k]:
            print(f"{k:24s} ERROR {a[k].get('error')} / {b[k].get('error')}")
            continue
        d = max(abs(x - y) for f in ("R00", "T00", "Rsum", "Tsum")
                for x, y in zip(a[k][f], b[k][f]))
        rel = d / max(abs(a[k]["R00"][0]), 1e-30)
        print(f"{k:24s} max |with - pre| = {d:.3e}  (rel to R00 {rel:.2e})")
