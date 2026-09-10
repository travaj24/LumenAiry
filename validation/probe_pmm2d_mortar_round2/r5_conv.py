"""R5 -- the DECIDING measurement for D1: does the M ladder STOP CONVERGING?

``python r5_conv.py [conv nomortar]``

R3 shows the y-uniform stack's error against the exact 1-D oracle growing
smoothly (and saturating) as the intra-layer sliver narrows -- roughly 2x over
four decades of ``delta`` at ``M = 6``.  That alone does not separate "a
slightly worse discretisation" from "a discretisation that has stopped
converging", because at ``M = 6`` the healthy arm's own error is the same size.

So this probe pushes the SAME fixture up the ``M`` ladder to 8 at a handful of
``delta``.  A sound scheme's error falls at every rung whatever the wall
positions; the defect is the ``delta`` at which the ladder FLATTENS.

``nomortar`` is the attribution control: the identical sliver layer with BOTH
neighbours on ITS OWN grid, so both its interfaces take the plain square modal
match and no mortar exists.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
import warnings

import numpy as np

import lumenairy
from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT), lumenairy.__file__
print(f"[arm] lumenairy = {lumenairy.__file__}", flush=True)

_C = complex
RES = {}
T0 = time.time()
PER, WL, TH = 0.9, 0.6, 0.20
EPSP, EPSH, TT = 6.0, 2.25, 0.10
W0, W2, YW = (0.21, 0.68), (0.33, 0.79), (0.30, 0.70)


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
    return st.solve()


def _oracle_pair():
    o14, R14, T14 = _oracle(14)[:3]
    o12, R12, T12 = _oracle(12)[:3]
    R14, T14 = np.atleast_2d(R14), np.atleast_2d(T14)
    keep = np.abs(np.asarray(o14)) <= 1
    gap = float(max(np.max(np.abs(np.atleast_2d(R12)[:, keep] - R14[:, keep])),
                    np.max(np.abs(np.atleast_2d(T12)[:, keep] - T14[:, keep]))))
    return np.asarray(o14), R14, T14, gap


def _score(o2d, R, T, o1d, R1d, T1d):
    best = 0.0
    for m in (-1, 0, 1):
        sel = np.where((o2d[:, 0] == m) & (o2d[:, 1] == 0))[0][0]
        j = int(np.where(o1d == m)[0][0])
        best = max(best, abs(float(R[1, sel]) - float(R1d[1, j])),
                   abs(float(T[1, sel]) - float(T1d[1, j])))
    return best


def _stack(delta, M, *, mortar=True):
    yw = [YW[0] * PER, YW[1] * PER]
    sw = [(0.5 - delta / 2) * PER, (0.5 + delta / 2) * PER]
    st = PMM2DStackPure(PER, n_modes=M, n_orders=1, layer_grids="per-layer")
    tl0 = np.array([[EPSH] * 3, [EPSP] * 3, [EPSH] * 3], dtype=_C)
    host = np.full((3, 3), _C(EPSH))
    # OUTER layers: on their own grids (mortar) or on the SLIVER's grid
    # (no mortar -- the attribution control, same eps profile either way).
    if mortar:
        st.add_layer(TT, eps_cell=tl0,
                     x_walls=[W0[0] * PER, W0[1] * PER], y_walls=yw)
        st.add_layer(TT, eps_cell=host, x_walls=sw, y_walls=yw)
        st.add_layer(TT, eps_cell=tl0,
                     x_walls=[W2[0] * PER, W2[1] * PER], y_walls=yw)
    else:
        # the SAME physical device carried entirely on the sliver's own grid:
        # three x-segments whose eps is set by which of W0/W2 they fall in.
        for w in (W0, None, W2):
            if w is None:
                cell = host
                xw = sw
            else:
                cell = np.zeros((3, 3), dtype=_C)
                bx = [0.0] + list(sw) + [PER]
                for i in range(3):
                    mid = 0.5 * (bx[i] + bx[i + 1]) / PER
                    cell[i, :] = EPSP if w[0] < mid < w[1] else EPSH
                xw = sw
            st.add_layer(TT, eps_cell=cell, x_walls=xw, y_walls=yw)
    st.set_source(WL, theta=TH)
    return st


def _solve(st):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o, R, T = st.solve(jones=False)
    return o, R, T, [str(x.message)[:80] for x in w]


def sec_conv():
    o1d, R1d, T1d, gap = _oracle_pair()
    _log(f"oracle self-gap {gap:.3e}")
    out = {"oracle_selfgap": gap}
    for delta in (0.30, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        rec = {}
        for M in (4, 5, 6, 7, 8):
            t0 = time.time()
            try:
                o, R, T, w = _solve(_stack(delta, M))
                rec[str(M)] = {"err": _score(o, R, T, o1d, R1d, T1d),
                               "closure": float(np.max(np.abs(
                                   R.sum(1) + T.sum(1) - 1.0))),
                               "warnings": w, "wall": time.time() - t0}
            except Exception as exc:                        # noqa: BLE001
                rec[str(M)] = {"REFUSED":
                               f"{type(exc).__name__}: {str(exc)[:110]}",
                               "wall": time.time() - t0}
            _log(f"  delta={delta:.0e} M={M}: "
                 + (rec[str(M)]["REFUSED"] if "REFUSED" in rec[str(M)]
                    else f"err={rec[str(M)]['err']:.4e} "
                         f"clo={rec[str(M)]['closure']:.2e} "
                         f"({rec[str(M)]['wall']:.0f}s)"))
        out[f"{delta:g}"] = rec
        errs = [rec[str(M)].get("err") for M in (4, 5, 6, 7, 8)]
        if all(e is not None for e in errs):
            _log(f"delta={delta:.0e}  LADDER " + " ".join(f"{e:.3e}"
                                                          for e in errs)
                 + f"   6->7 {errs[2] / errs[3]:.2f}x  7->8 "
                   f"{errs[3] / errs[4]:.2f}x")
    RES["conv"] = out


def sec_nomortar():
    """The attribution control: the SAME sliver, NO mortar."""
    o1d, R1d, T1d, gap = _oracle_pair()
    out = {"oracle_selfgap": gap}
    for delta in (0.30, 1e-3, 1e-5, 1e-6):
        rec = {}
        for M in (4, 6, 7):
            try:
                o, R, T, w = _solve(_stack(delta, M, mortar=False))
                rec[str(M)] = {"err": _score(o, R, T, o1d, R1d, T1d),
                               "closure": float(np.max(np.abs(
                                   R.sum(1) + T.sum(1) - 1.0))),
                               "warnings": w}
            except Exception as exc:                        # noqa: BLE001
                rec[str(M)] = {"REFUSED":
                               f"{type(exc).__name__}: {str(exc)[:110]}"}
        out[f"{delta:g}"] = rec
        _log(f"NO-MORTAR delta={delta:.0e}: "
             + "  ".join(f"M{M}=" + ("REFUSED" if "REFUSED" in rec[str(M)]
                                     else f"{rec[str(M)]['err']:.3e}")
                         for M in (4, 6, 7)))
    RES["nomortar"] = out


SECTIONS = {"conv": sec_conv, "nomortar": sec_nomortar}


def main():
    for w in (sys.argv[1:] or list(SECTIONS)):
        SECTIONS[w]()
    tag = os.environ.get("R_TAG", "")
    path = os.path.join(HERE, f"r5_conv{('_' + tag) if tag else ''}.json")
    with open(path, "w") as fh:
        json.dump(RES, fh, indent=1, sort_keys=True, default=float)
    _log(f"wrote {path}")


if __name__ == "__main__":
    main()
