"""ROUND 3, DEFECT V2 (P3) -- can the width contract be lifted for a
CONFORMING per-layer stack, at ``solve()``, where the neighbours ARE known?

The VERIFY audit's S5.2 case 3: a per-layer stack whose layers all carry the
SAME wall array has NO cross-grid projection anywhere -- every interface is the
plain square modal match -- so the mechanism the contract guards cannot form.
``Basis1D`` cannot know that, so it refuses anyway.

This probe measures the two things that decide whether the exemption is a CLEAN
change:

  1. WHERE the refusal is raised from.  ``StagGridOps`` builds ``Basis1D``
     (twice), but so does ``Granet2DTransverseE.__init__`` -- the REGION
     EIGENSOLVER, a public class the hybrid engine, the tests and the probes
     construct directly.  If both refuse independently, an exemption reached
     from ``PMM2DStackPure.solve`` has to be threaded through BOTH
     constructors, on the hot path.
  2. HOW MUCH the exemption would be worth: with the contract LIFTED, how
     ``delta``-INDEPENDENT is a conforming stack's answer really, over three
     decades of wall separation, against the mortared arm's known 4.9x floor.

``python r6_v2_conforming.py [win|wsl]``
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as _ts  # noqa: E402
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Basis1D,
    Granet2DTransverseE,
    StagGridOps,
)

assert os.path.abspath(lumenairy.__file__).lower().startswith(_ROOT.lower()), (
    lumenairy.__file__)
TAG = (sys.argv[1] if len(sys.argv) > 1 else "win")
T0 = time.time()
_C = complex
P, WL, TH, PH = 1.0, 0.62, 0.09, 0.3
EPS_H, EPS_B = 2.25, 6.0
print(f"[arm {TAG}] lumenairy = {lumenairy.__file__} v{lumenairy.__version__}",
      flush=True)


def _log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


# ------------------------------------------------------------- MY fixture
# The SAME shape the VERIFY audit's ``nomortar_fp`` uses and the same shape
# r5_degradation_band.py scores: a y-uniform 3-layer stack whose middle layer
# is ALL HOST.  Every layer is carried on the SLIVER's own wall array, and each
# layer's cell is the MIDPOINT SAMPLE of a FIXED physical pattern, so the
# sampled DEVICE is independent of ``delta`` to O(delta) and every movement in
# the answer is numerical.
PER, WLEN, THETA = 0.93, 0.66, 0.19
EPSP, EPSHOST, TT = 6.25, 2.1, 0.14
W0, W2, YW = (0.155, 0.585), (0.315, 0.795), (0.22, 0.68)
CENTRE = 0.41


def _oracle(deg):
    from lumenairy.elements.pmm.stack import PMMStack  # noqa: PLC0415
    s = PMMStack(PER, degree=deg, far_field_orders=5)
    s.add_layer(TT, segments=[(W0[0], EPSHOST), (W0[1] - W0[0], EPSP),
                              (1 - W0[1], EPSHOST)])
    s.add_layer(TT, segments=[(1.0, EPSHOST)])
    s.add_layer(TT, segments=[(W2[0], EPSHOST), (W2[1] - W2[0], EPSP),
                              (1 - W2[1], EPSHOST)])
    s.set_source(WLEN, theta=THETA)
    return s.solve(stabilize=None)


def _conforming(delta, M):
    """EVERY layer on the SAME (sliver-carrying) wall array -> every interface
    is the plain square modal match and NO mortar forms."""
    st = PMM2DStackPure(PER, n_modes=M, n_orders=1, layer_grids="per-layer")
    sw = [(CENTRE - delta / 2) * PER, (CENTRE + delta / 2) * PER]
    ywl = [YW[0] * PER, YW[1] * PER]
    bx = [0.0] + list(sw) + [PER]
    for w in (W0, None, W2):
        if w is None:
            cell = np.full((3, 3), _C(EPSHOST))
        else:
            cell = np.zeros((3, 3), dtype=_C)
            for i in range(3):
                mid = 0.5 * (bx[i] + bx[i + 1]) / PER
                cell[i, :] = EPSP if w[0] < mid < w[1] else EPSHOST
        st.add_layer(TT, eps_cell=cell, x_walls=sw, y_walls=ywl)
    st.set_source(WLEN, theta=THETA, phi=0.0)
    return st


RES = {}

# ---- 1. WHERE the refusal comes from ----------------------------------
walls = np.array([0.0, 0.44, 0.4405, 1.0])          # 5e-4 -> under the bar
sites = {}
for name, fn in (
        ("Basis1D", lambda: Basis1D(P, walls, 4)),
        ("StagGridOps", lambda: StagGridOps(P, P, walls, walls, 4, 1.0, 1.0)),
        ("Granet2DTransverseE",
         lambda: Granet2DTransverseE(P, P, walls, walls, 4,
                                     np.full((3, 3), _C(EPS_H)))),
):
    try:
        fn()
        sites[name] = "accepted"
    except Exception as exc:                            # noqa: BLE001
        sites[name] = f"{type(exc).__name__}: {str(exc)[:60]}"
    _log(f"  {name:22s} {sites[name]}")
RES["construction_sites"] = sites

# ---- 2. what the exemption would be worth -----------------------------
prev = _ts.PMM2D_STAG_MIN_SEG_GUARD
_ts.PMM2D_STAG_MIN_SEG_GUARD = False
rows = {}
try:
    o14, R14, T14 = _oracle(14)[:3]
    R14, T14 = np.atleast_2d(R14), np.atleast_2d(T14)
    o14 = np.asarray(o14)
    for M in (4, 5, 6):
        vals = {}
        for delta in (3e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
            try:
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter("always")
                    o, R, T = _conforming(delta, M).solve(jones=False)
                o = np.asarray(o)
                R2, T2 = np.atleast_2d(R), np.atleast_2d(T)
                worst = 0.0
                for m in (-1, 0, 1):
                    sel = int(np.where((o[:, 0] == m) & (o[:, 1] == 0))[0][0])
                    j = int(np.where(o14 == m)[0][0])
                    worst = max(worst, abs(float(R2[1, sel]) - float(R14[1, j])),
                                abs(float(T2[1, sel]) - float(T14[1, j])))
                k = int(np.argmin(np.abs(o[:, 0]) + np.abs(o[:, 1])))
                vals[f"{delta:.0e}"] = dict(
                    R00=float(R2[1, k]), err_vs_oracle=worst,
                    closure=float(np.max(np.abs(R2.sum(1) + T2.sum(1) - 1.0))),
                    warnings=[w.category.__name__ for w in ws])
            except Exception as exc:                    # noqa: BLE001
                vals[f"{delta:.0e}"] = {
                    "error": f"{type(exc).__name__}: {str(exc)[:70]}"}
        good = {k: v["R00"] for k, v in vals.items() if "R00" in v}
        errs = {k: v["err_vs_oracle"] for k, v in vals.items()
                if "err_vs_oracle" in v}
        if good:
            spread = max(good.values()) - min(good.values())
            base = good.get("3e-01")
            _log(f"  M={M}: R00 spread over 3e-01..1e-06 = {spread:.3e} "
                 f"(relative {spread / abs(base):.3e}); err vs the exact 1-D "
                 f"oracle {min(errs.values()):.3e} .. {max(errs.values()):.3e} "
                 f"(ratio {max(errs.values()) / min(errs.values()):.3f}); "
                 f"warnings "
                 f"{sum(len(v.get('warnings', [])) for v in vals.values())}")
            rows[M] = dict(values=vals, spread=spread,
                           relative=spread / abs(base),
                           err_ratio=max(errs.values()) / min(errs.values()))
        else:
            rows[M] = dict(values=vals)
finally:
    _ts.PMM2D_STAG_MIN_SEG_GUARD = prev
RES["conforming_sensitivity"] = {str(k): v for k, v in rows.items()}

# ---- 3. and the refusal a user actually meets --------------------------
try:
    _conforming(5e-4, 4).solve(jones=False)
    RES["shipped_refusal"] = "accepted"
except Exception as exc:                                # noqa: BLE001
    RES["shipped_refusal"] = f"{type(exc).__name__}: {str(exc)[:120]}"
_log(f"  shipped, guard ARMED: {RES['shipped_refusal'][:110]}")

p = os.path.join(_HERE, f"r6_v2_conforming_{TAG}.json")
with open(p, "w") as fh:
    json.dump({"tag": TAG, "lumenairy": lumenairy.__file__, **RES}, fh,
              indent=1, default=str)
_log(f"wrote {p}")
