"""VERIFY round 2 -- DEFECT V1: the D2 rcond bar refuses ORDINARY out-of-plane
per-layer stacks.

The bar ``_MORTAR_RCOND_REFUSE = 1e-12`` was calibrated on the IN-PLANE pair
(``MassE_B W_B`` / ``MassH_A V_A``, healthy population 2.6e-07 .. 3.8e-04) and
applied unchanged to the THIRD site,
``_interface_smatrix_general_mortar_2d`` -- the ``4 qq x 4 qq`` block solve an
OUT-OF-PLANE tensor or a SLANTED per-layer layer takes.  That site's HEALTHY
population sits several decades lower, so ordinary geometries fall through it.

This probe proves the refusal is a FALSE POSITIVE, not the guard working:

  * the SAME device is built two ways -- per-layer NON-CONFORMING (which takes
    the generalized mortar) and on the UNION grid (which does not) -- so the
    union arm is an oracle for the mortar arm;
  * on the PRE-round-2 tree the mortar arm RETURNS, and its answer is compared
    with the union arm;
  * on the round-2 tree the same call RAISES.

    V8_TAG=with PYTHONPATH=/c/tmp/lum_vmortar2 python v8_oop_regression.py
    V8_TAG=pre  PYTHONPATH=/c/tmp/lum_prem2 V8_EXPECT_ROOT=/c/tmp/lum_prem2 \
                python v8_oop_regression.py
    python v8_oop_regression.py compare
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = os.environ.get("V8_TAG", "with")
EXPECT = os.environ.get("V8_EXPECT_ROOT")
if EXPECT:
    assert os.path.abspath(lumenairy.__file__).lower().startswith(
        os.path.abspath(EXPECT).lower()), lumenairy.__file__
print(f"[{TAG}] lumenairy = {lumenairy.__file__}", flush=True)
T0 = time.time()
_C = complex
P = 1.0e-6
WL = 0.62e-6
WA = (0.2371, 0.6183)     # layer A's pillar, as fractions of the period
WB = (0.3117, 0.7402)     # layer B's pillar
UNION = (0.2371, 0.3117, 0.6183, 0.7402)
EPS_H = 2.25
EPS_B = 6.0
E_OOP = np.array([[4.0, 0.0, 0.8], [0.0, 3.4, 0.0], [0.75, 0.0, 3.2]],
                 dtype=_C)


def _log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def _mid(w):
    b = (0.0,) + tuple(w) + (1.0,)
    return [0.5 * (b[i] + b[i + 1]) for i in range(len(b) - 1)]


def _tensor_cell(walls, lo, hi, eps_in, eps_host):
    m = _mid(walls)
    n = len(m)
    c = np.empty((n, n, 3, 3), dtype=_C)
    for i in range(n):
        for j in range(n):
            inside = (lo < m[i] < hi) and (lo < m[j] < hi)
            c[i, j] = eps_in if inside else np.eye(3) * eps_host
    return c


def _scalar_cell(walls, lo, hi, eps_in, eps_host):
    m = _mid(walls)
    n = len(m)
    c = np.full((n, n), _C(eps_host))
    for i in range(n):
        for j in range(n):
            if (lo < m[i] < hi) and (lo < m[j] < hi):
                c[i, j] = _C(eps_in)
    return c


def _solve(mode, M, n_orders=2):
    """``mode`` = 'perlayer' (NON-CONFORMING: takes the generalized mortar) or
    'union' (both layers on the common refinement: no mortar)."""
    s = PMM2DStackPure(P, n_modes=M, n_orders=n_orders, n_substrate=1.5,
                       layer_grids="per-layer")
    if mode == "perlayer":
        s.add_layer(0.13e-6,
                    eps_cell=_tensor_cell(WA, WA[0], WA[1], E_OOP, EPS_H),
                    x_walls=[w * P for w in WA], y_walls=[w * P for w in WA])
        s.add_layer(0.10e-6,
                    eps_cell=_scalar_cell(WB, WB[0], WB[1], EPS_B, EPS_H),
                    x_walls=[w * P for w in WB], y_walls=[w * P for w in WB])
    else:
        s.add_layer(0.13e-6,
                    eps_cell=_tensor_cell(UNION, WA[0], WA[1], E_OOP, EPS_H),
                    x_walls=[w * P for w in UNION],
                    y_walls=[w * P for w in UNION])
        s.add_layer(0.10e-6,
                    eps_cell=_scalar_cell(UNION, WB[0], WB[1], EPS_B, EPS_H),
                    x_walls=[w * P for w in UNION],
                    y_walls=[w * P for w in UNION])
    s.set_source(WL, theta=0.09, phi=0.3)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        o, R, T = s.solve(jones=False)
    o = np.asarray(o)
    R, T = np.atleast_2d(R), np.atleast_2d(T)
    keep = (np.abs(o[:, 0]) <= 1) & (np.abs(o[:, 1]) <= 1)
    return dict(R=R[:, keep].tolist(), T=T[:, keep].tolist(),
                closure=float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0))),
                n_warnings=len(ws))


OUT = {}
for M in (4, 5, 6, 7):
    for mode in ("perlayer", "union"):
        key = f"{mode}_M{M}"
        try:
            OUT[key] = _solve(mode, M)
            _log(f"[{TAG}] {key:16s} closure {OUT[key]['closure']:.3e}  "
                 f"R00 {OUT[key]['R'][1][4]:.8f}")
        except Exception as exc:                        # noqa: BLE001
            OUT[key] = {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}
            _log(f"[{TAG}] {key:16s} {OUT[key]['error'][:110]}")

p = os.path.join(HERE, f"v8_oop_{TAG}.json")
with open(p, "w") as fh:
    json.dump({"tag": TAG, "lumenairy": lumenairy.__file__, "rows": OUT}, fh,
              indent=1)
_log(f"wrote {p}")

if len(sys.argv) > 1 and sys.argv[1] == "compare":
    a = json.load(open(os.path.join(HERE, "v8_oop_with.json")))["rows"]
    b = json.load(open(os.path.join(HERE, "v8_oop_pre.json")))["rows"]
    print(f"\n{'case':16s} {'round-2':>34s}   {'pre-round-2':>34s}")
    for k in a:
        for tagn, r in (("with", a[k]), ("pre", b[k])):
            pass
    for M in (4, 5, 6, 7):
        for mode in ("perlayer", "union"):
            k = f"{mode}_M{M}"
            wa, wb = a[k], b[k]
            sa = wa.get("error", f"closure {wa.get('closure', 0):.2e}")
            sb = wb.get("error", f"closure {wb.get('closure', 0):.2e}")
            print(f"{k:16s} {sa[:34]:>34s}   {sb[:34]:>34s}")
    print("\nPER-LAYER (mortar) vs UNION (no mortar), same device, PRE tree:")
    for M in (4, 5, 6, 7):
        pl, un = b[f"perlayer_M{M}"], b[f"union_M{M}"]
        if "error" in pl or "error" in un:
            print(f"  M={M}: {pl.get('error') or un.get('error')}")
            continue
        d = max(abs(x - y) for f in ("R", "T")
                for ra, rb in zip(pl[f], un[f])
                for x, y in zip(ra, rb))
        print(f"  M={M}: max |per-layer - union| over |m|,|n| <= 1 = "
              f"{d:.3e}   (per-layer closure {pl['closure']:.2e}, union "
              f"closure {un['closure']:.2e})")
