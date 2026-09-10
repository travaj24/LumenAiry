"""ROUND 3, S5.4 -- the DEGRADATION BAND above the width contract, re-measured
so the new ``UserWarning``'s upper edge is a DERIVED bar.

The VERIFY audit's S5.4 measured, on ITS fixture, a SILENT 1.00 / 2.09 / 4.02 /
5.31 / 5.69 / 5.82 / 5.86x accuracy cost at narrowest segments 3e-1 / 1e-1 /
3e-2 / 1e-2 / 4.1e-3 / 2e-3 / 1e-3 of the period, not removable by ``n_modes``
(the ladder FLATTENS, so a user converging in ``n_modes`` reads the flattening
as convergence).  This probe re-measures that curve on a DIFFERENT fixture --
different period, wavelength, angle, contrast, wall positions and sliver centre
-- with a finer ladder, so the width at which the cost crosses **2x** can be
read off rather than assumed, and reports the margin from that width to the
narrowest segment any ORDINARY shipped geometry asks for.

The device is chosen so it CANNOT depend on the wall separation at all: a
y-uniform 3-layer stack whose MIDDLE layer is ALL HOST and carries the two
walls.  The oracle is the exact 1-D ``PMMStack`` at degree 14, bounded by its
own 12->14 self-gap.

``python r5_degradation_band.py [win|wsl] [--fast]``
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
from lumenairy.elements.pmm.stack import PMMStack  # noqa: E402
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure  # noqa: E402

assert os.path.abspath(lumenairy.__file__).lower().startswith(_ROOT.lower()), (
    lumenairy.__file__)
ARGS = sys.argv[1:]
FAST = "--fast" in ARGS
TAG = next((a for a in ARGS if not a.startswith("-")), "win")
T0 = time.time()
_C = complex
print(f"[arm {TAG}] lumenairy = {lumenairy.__file__} v{lumenairy.__version__}",
      flush=True)

# --------------------------------------------------------- MY OWN fixture
PER = 0.93          # um  (builder 1.2, verifier 1.05)
WL = 0.66           #     (builder 0.85, verifier 0.71)
TH, PH = 0.19, 0.0
EPSP, EPSH = 6.25, 2.1
TT = 0.14
W0 = (0.155, 0.585)   # patterned neighbour ABOVE
W2 = (0.315, 0.795)   # patterned neighbour BELOW
YW = (0.22, 0.68)     # identical on every layer: the ONLY non-conformity is x
SLIVER_C = 0.41       # OFF-CENTRE

DELTAS = (3e-1, 2e-1, 1.5e-1, 1e-1, 7e-2, 5e-2, 3e-2, 2e-2, 1e-2, 3e-3, 1e-3)
MODES = (6, 7) if FAST else (6, 7, 8)


def _log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def _oracle(deg):
    st = PMMStack(PER, degree=deg, far_field_orders=5)
    st.add_layer(TT, segments=[(W0[0], EPSH), (W0[1] - W0[0], EPSP),
                               (1.0 - W0[1], EPSH)])
    st.add_layer(TT, segments=[(1.0, EPSH)])
    st.add_layer(TT, segments=[(W2[0], EPSH), (W2[1] - W2[0], EPSP),
                               (1.0 - W2[1], EPSH)])
    st.set_source(WL, theta=TH)
    return st.solve(stabilize=None)


def _oracle_pair():
    o14, R14, T14 = _oracle(14)[:3]
    o12, R12, T12 = _oracle(12)[:3]
    R14, T14 = np.atleast_2d(R14), np.atleast_2d(T14)
    R12, T12 = np.atleast_2d(R12), np.atleast_2d(T12)
    keep = np.abs(np.asarray(o14)) <= 1
    gap = float(max(np.max(np.abs(R12[:, keep] - R14[:, keep])),
                    np.max(np.abs(T12[:, keep] - T14[:, keep]))))
    return np.asarray(o14), R14, T14, gap


def _score(o2d, R, T, o1d, R1d, T1d):
    worst = 0.0
    for m in (-1, 0, 1):
        sel = np.where((o2d[:, 0] == m) & (o2d[:, 1] == 0))[0][0]
        j = int(np.where(o1d == m)[0][0])
        worst = max(worst, abs(float(R[1, sel]) - float(R1d[1, j])),
                    abs(float(T[1, sel]) - float(T1d[1, j])))
    return worst


def _tile(w):
    c = np.full((3, 3), _C(EPSH))
    c[1, :] = _C(EPSP)
    return c


def _stack(delta, M):
    yw = [YW[0] * PER, YW[1] * PER]
    sw = [(SLIVER_C - delta / 2) * PER, (SLIVER_C + delta / 2) * PER]
    st = PMM2DStackPure(PER, n_modes=M, n_orders=1, layer_grids="per-layer")
    st.add_layer(TT, eps_cell=_tile(W0),
                 x_walls=[W0[0] * PER, W0[1] * PER], y_walls=yw)
    st.add_layer(TT, eps_cell=np.full((3, 3), _C(EPSH)), x_walls=sw,
                 y_walls=yw)
    st.add_layer(TT, eps_cell=_tile(W2),
                 x_walls=[W2[0] * PER, W2[1] * PER], y_walls=yw)
    st.set_source(WL, theta=TH, phi=PH)
    return st


o1d, R1d, T1d, selfgap = _oracle_pair()
_log(f"oracle degree-14; its own 12->14 self-gap = {selfgap:.3e}")

ERR, WARN, CLO = {}, {}, {}
for M in MODES:
    for delta in DELTAS:
        st = _stack(delta, M)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            o, R, T = st.solve(jones=False)
        o, R, T = np.asarray(o), np.atleast_2d(R), np.atleast_2d(T)
        e = _score(o, R, T, o1d, R1d, T1d)
        ERR[f"M{M}_d{delta:.3e}"] = e
        WARN[f"M{M}_d{delta:.3e}"] = [
            f"{w.category.__name__}: {str(w.message)[:70]}" for w in ws]
        CLO[f"M{M}_d{delta:.3e}"] = float(
            np.max(np.abs(R.sum(1) + T.sum(1) - 1.0)))
        _log(f"  M={M} delta={delta:.3e}  err {e:.4e}  closure "
             f"{CLO[f'M{M}_d{delta:.3e}']:.2e}  warnings {len(ws)}")

# ---- the degradation ratio, normalised to the ORDINARY end of the ladder
RATIO = {}
for M in MODES:
    base = ERR[f"M{M}_d{DELTAS[0]:.3e}"]
    RATIO[M] = {f"{d:.3e}": ERR[f"M{M}_d{d:.3e}"] / base for d in DELTAS}
_log("degradation ratio err(delta) / err(3e-01):")
hdr = "  delta      " + "".join(f"  M={M:<7d}" for M in MODES)
_log(hdr)
for d in DELTAS:
    _log(f"  {d:.3e}" + "".join(f"  {RATIO[M][f'{d:.3e}']:8.3f}"
                                for M in MODES))

# ---- the 2x crossing, per M and worst-case ----------------------------
CROSS = {}
for M in MODES:
    hit = None
    for d in DELTAS:                       # widest first
        if RATIO[M][f"{d:.3e}"] > 2.0:
            hit = d
            break
    CROSS[M] = hit
_log(f"first width (widest first) whose ratio exceeds 2.0, per M: {CROSS}")
worst = [d for d in DELTAS
         if max(RATIO[M][f"{d:.3e}"] for M in MODES) > 2.0]
edge = max(worst) if worst else None
_log(f"WORST-CASE over M: the widest ladder rung above 2.0x is {edge}")

# ---- the ORDINARY census: what the library's own geometries ask for ----
ORD = {}
try:
    sys.path.insert(0, os.path.join(_ROOT, "tests", "unit"))
    import test_fix_pmm2d_mortar_round2 as _r2  # noqa: E402

    for name, st in _r2._shipped_geometry_battery().items():
        ORD[name] = _r2._narrowest(st)
    _log(f"ordinary census: {len(ORD)} geometries, narrowest "
         f"{min(ORD.values()):.4e} ({min(ORD, key=ORD.get)})")
    for k in sorted(ORD, key=ORD.get)[:6]:
        _log(f"    {k:24s} {ORD[k]:.4e}")
except Exception as exc:                                    # noqa: BLE001
    _log(f"ordinary census unavailable: {type(exc).__name__}: {exc}")

RES = dict(fixture=dict(period=PER, wl=WL, theta=TH, epsp=EPSP, epsh=EPSH,
                        t=TT, w0=W0, w2=W2, yw=YW, sliver_centre=SLIVER_C),
           oracle_selfgap=selfgap, err=ERR, ratio={str(k): v
                                                   for k, v in RATIO.items()},
           warnings=WARN, closure=CLO, cross={str(k): v
                                              for k, v in CROSS.items()},
           edge=edge, ordinary=ORD)
p = os.path.join(_HERE, f"r5_degradation_band_{TAG}.json")
with open(p, "w") as fh:
    json.dump({"tag": TAG, "lumenairy": lumenairy.__file__, **RES}, fh,
              indent=1, default=str)
_log(f"wrote {p}")
