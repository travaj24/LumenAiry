"""R3 -- the D1 ONSET, scored against an EXACT oracle.

``python r3_ladder.py [ladder healthy taper]``

The instrument R1 first reached for -- "compare the sliver stack with the same
stack whose middle layer sits on one segment" -- is not usable: the reference's
own ``M``-ladder self-gap on a 2-D pillar device reads 9.2e-04, which is the
size of the effect being measured.  So the device is made **y-UNIFORM**: three
x-strip layers, constant along ``y``.  The EXACT 1-D pure PMM (``PMMStack`` at
degree 14, whose own degree self-gap is measured here) is then the truth for
the WHOLE 2-D answer, exactly as the shipped ``test_n4_...`` gate uses it.

The middle layer is ALL HOST, so its two walls are element boundaries in a
CONTINUOUS medium and the device does not depend on their separation ``delta``
at all.  A sound discretisation must therefore converge to the oracle at EVERY
``delta``; the onset is the ``delta`` at which the ``M`` ladder stops
converging.  That is a DECISION, not a reading.

The y grids are IDENTICAL on all three layers, so the ONLY non-conformity is in
``x`` and the attribution is clean.
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


def _log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


# --------------------------------------------------------------- the device
FIXTURES = {
    # (period, wl, theta, eps_p, eps_h, thickness, walls of layer 0 / layer 2)
    "A": dict(per=0.9, wl=0.6, th=0.20, epsp=6.0, epsh=2.25, t=0.10,
              w0=(0.21, 0.68), w2=(0.33, 0.79), yw=(0.30, 0.70)),
    "B": dict(per=1.2, wl=0.85, th=0.15, epsp=9.0, epsh=2.25, t=0.06,
              w0=(0.2371, 0.6183), w2=(0.3117, 0.7402), yw=(0.27, 0.61)),
    "C": dict(per=1.4, wl=1.05, th=0.0, epsp=12.25, epsh=2.25, t=0.13,
              w0=(0.17, 0.55), w2=(0.41, 0.83), yw=(0.35, 0.80)),
}
DELTAS = (3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
MS = (4, 5, 6)


def _oracle(fx, deg):
    per = fx["per"]
    st = PMMStack(per, degree=deg, far_field_orders=5)
    for w in (fx["w0"], fx["w2"]):
        st.add_layer(fx["t"], segments=[(w[0], fx["epsh"]),
                                        (w[1] - w[0], fx["epsp"]),
                                        (1.0 - w[1], fx["epsh"])])
        if w is fx["w0"]:
            st.add_layer(fx["t"], segments=[(1.0, fx["epsh"])])
    st.set_source(fx["wl"], theta=fx["th"])
    return st.solve()


def _oracle_pair(fx):
    o14, R14, T14 = _oracle(fx, 14)[:3]
    o12, R12, T12 = _oracle(fx, 12)[:3]
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


def _stack(fx, delta, M):
    per = fx["per"]
    yw = [fx["yw"][0] * per, fx["yw"][1] * per]
    st = PMM2DStackPure(per, n_modes=M, n_orders=1, layer_grids="per-layer")
    tl0 = np.array([[fx["epsh"]] * 3, [fx["epsp"]] * 3, [fx["epsh"]] * 3],
                   dtype=_C)
    host = np.full((3, 3), _C(fx["epsh"]))
    st.add_layer(fx["t"], eps_cell=tl0,
                 x_walls=[fx["w0"][0] * per, fx["w0"][1] * per], y_walls=yw)
    st.add_layer(fx["t"], eps_cell=host,
                 x_walls=[(0.5 - delta / 2) * per, (0.5 + delta / 2) * per],
                 y_walls=yw)
    st.add_layer(fx["t"], eps_cell=tl0,
                 x_walls=[fx["w2"][0] * per, fx["w2"][1] * per], y_walls=yw)
    st.set_source(fx["wl"], theta=fx["th"])
    return st


def _solve(st):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o, R, T = st.solve(jones=False)
    return o, R, T, [str(x.message)[:80] for x in w]


def sec_ladder():
    out = {}
    for name, fx in FIXTURES.items():
        o1d, R1d, T1d, selfgap = _oracle_pair(fx)
        _log(f"{name}: oracle self-gap (deg 12 vs 14) = {selfgap:.3e}")
        rows = {"oracle_selfgap": selfgap}
        grid = {}
        for delta in DELTAS:
            rec = {}
            for M in MS:
                t0 = time.time()
                try:
                    o, R, T, w = _solve(_stack(fx, delta, M))
                    rec[str(M)] = {
                        "err": _score(o, R, T, o1d, R1d, T1d),
                        "closure": float(np.max(np.abs(R.sum(1) + T.sum(1)
                                                       - 1.0))),
                        "warnings": w, "wall": time.time() - t0}
                except Exception as exc:                     # noqa: BLE001
                    rec[str(M)] = {
                        "REFUSED": f"{type(exc).__name__}: {str(exc)[:110]}",
                        "wall": time.time() - t0}
            errs = [rec[str(M)].get("err") for M in MS]
            ok = [e for e in errs if e is not None]
            rec["converging"] = bool(
                len(ok) == len(MS)
                and all(ok[i + 1] < ok[i] for i in range(len(ok) - 1)))
            rec["ratio_last"] = (ok[-2] / ok[-1]) if len(ok) == len(MS) else None
            grid[f"{delta:g}"] = rec
            _log(f"{name}: delta={delta:.0e}  err(M=4..6) = "
                 + " ".join("REF" if e is None else f"{e:.3e}" for e in errs)
                 + f"   conv={rec['converging']}  clo "
                 + " ".join(f"{rec[str(M)].get('closure', float('nan')):.1e}"
                            for M in MS))
        rows["grid"] = grid
        out[name] = rows
    RES["ladder"] = out


def sec_healthy():
    """The CONTROL: the SAME stack with the middle layer's walls at ORDINARY
    spacings, and the same stack with the middle layer on ONE segment -- both
    must converge, and their errors bound what 'healthy' means."""
    out = {}
    for name, fx in FIXTURES.items():
        o1d, R1d, T1d, selfgap = _oracle_pair(fx)
        rec = {"oracle_selfgap": selfgap}
        for lab, delta in (("w0.30", 0.30), ("w0.50", 0.50), ("w0.20", 0.20),
                           ("w0.12", 0.12)):
            errs = []
            for M in MS:
                o, R, T, _w = _solve(_stack(fx, delta, M))
                errs.append(_score(o, R, T, o1d, R1d, T1d))
            rec[lab] = {"errs": errs,
                        "converging": all(errs[i + 1] < errs[i]
                                          for i in range(len(errs) - 1))}
            _log(f"{name} healthy {lab}: " + " ".join(f"{e:.3e}" for e in errs)
                 + f"  conv={rec[lab]['converging']}")
        out[name] = rec
    RES["healthy"] = out


def sec_taper():
    """The REALISTIC API path: ``add_tapered_pillar`` on a taper that closes.
    The midpoint rule's narrowest sampled width is ``w_top/2 + ...``; measure
    what the narrowest segment actually is and whether the answer converges."""
    fx = FIXTURES["B"]
    per = fx["per"]
    out = {}
    for wtop in (0.40, 0.20, 0.02, 2e-3, 2e-4, 2e-5):
        rec = {}
        for M in (4, 5, 6):
            try:
                st = PMM2DStackPure(per, n_modes=M, n_orders=1,
                                    layer_grids="per-layer")
                st.add_tapered_pillar(
                    0.24, eps_pillar=fx["epsp"], eps_host=fx["epsh"],
                    x_bounds_bottom=[0.25 * per, 0.75 * per],
                    y_bounds_bottom=[0.25 * per, 0.75 * per],
                    x_bounds_top=[(0.5 - wtop / 2) * per,
                                  (0.5 + wtop / 2) * per],
                    y_bounds_top=[(0.5 - wtop / 2) * per,
                                  (0.5 + wtop / 2) * per],
                    n_slices=4)
                st.set_source(fx["wl"], theta=fx["th"], phi=0.35)
                o, R, T, w = _solve(st)
                p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
                narrow = min(float(np.min(np.diff(np.asarray(L["wx"]))))
                             for L in st._layers) / per
                rec[str(M)] = {"R00": float(R[0, p0]),
                               "closure": float(np.max(np.abs(
                                   R.sum(1) + T.sum(1) - 1.0))),
                               "narrowest_seg_frac": narrow, "warnings": w}
            except Exception as exc:                        # noqa: BLE001
                rec[str(M)] = {"REFUSED":
                               f"{type(exc).__name__}: {str(exc)[:110]}"}
        out[f"{wtop:g}"] = rec
        _log(f"taper w_top={wtop:.0e}: "
             + "  ".join(f"M{M}=" + ("REFUSED" if "REFUSED" in rec[str(M)]
                                     else f"{rec[str(M)]['R00']:.6f}"
                                          f"(seg {rec[str(M)]['narrowest_seg_frac']:.1e})")
                         for M in (4, 5, 6)))
    RES["taper"] = out


SECTIONS = {"healthy": sec_healthy, "ladder": sec_ladder, "taper": sec_taper}


def main():
    for w in (sys.argv[1:] or list(SECTIONS)):
        SECTIONS[w]()
    tag = os.environ.get("R_TAG", "")
    path = os.path.join(HERE, f"r3_ladder{('_' + tag) if tag else ''}.json")
    with open(path, "w") as fh:
        json.dump(RES, fh, indent=1, sort_keys=True, default=float)
    _log(f"wrote {path}")


if __name__ == "__main__":
    main()
