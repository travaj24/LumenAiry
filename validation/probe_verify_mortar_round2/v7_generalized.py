"""VERIFY round 2 -- the GENERALIZED (out-of-plane / slanted) mortar site.

``_guarded_mortar_solve`` is wired at THREE sites.  The fix doc's healthy
population (106 solves, ``rcond`` 2.61e-07 .. 3.77e-04) is measured on the
in-plane pair (``MassE_B W_B`` / ``MassH_A V_A``).  This probe measures the
THIRD one -- ``_interface_smatrix_general_mortar_2d``, the site an OUT-OF-PLANE
tensor or a SLANTED per-layer stack takes -- over ORDINARY, healthy geometries,
and asks how close its healthy population comes to the 1e-12 refusal.

``python v7_generalized.py``
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402
from lumenairy.elements.pmm import _core as _pc  # noqa: E402
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
print(f"[arm] lumenairy = {lumenairy.__file__} v{lumenairy.__version__}",
      flush=True)
TAG = os.environ.get("V7_TAG", "win")
T0 = time.time()
_C = complex
P = 1.0e-6
WL = 0.62e-6
RES = {}


def _log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def _tile(eh=2.25, ep=6.0):
    c = np.full((3, 3), _C(eh))
    c[1, 1] = _C(ep)
    return c


def _oop_cell(exz=0.8, eh=2.25):
    e = np.array([[4.0, 0.0, exz], [0.0, 3.4, 0.0], [exz - 0.05, 0.0, 3.2]],
                 dtype=_C)
    c = np.empty((3, 3, 3, 3), dtype=_C)
    c[...] = np.eye(3) * eh
    c[1, 1] = e
    return c


def _run(kind, M, xa, xb, theta=0.09, phi=0.3, exz=0.8):
    s = PMM2DStackPure(P, n_modes=M, n_orders=2, n_substrate=1.5,
                       layer_grids="per-layer")
    if kind == "oop":
        s.add_layer(0.13e-6, eps_cell=_oop_cell(exz), x_walls=xa, y_walls=xa)
        s.add_layer(0.10e-6, eps_cell=_tile(), x_walls=xb, y_walls=xb)
    elif kind == "oop_both":
        s.add_layer(0.13e-6, eps_cell=_oop_cell(exz), x_walls=xa, y_walls=xa)
        s.add_layer(0.10e-6, eps_cell=_oop_cell(exz * 0.7), x_walls=xb,
                    y_walls=xb)
    elif kind == "slant":
        s.add_layer(0.13e-6, eps_cell=_tile(), x_walls=xa, y_walls=xa,
                    slant=(0.08, 0.03))
        s.add_layer(0.10e-6, eps_cell=_tile(ep=4.0), x_walls=xb, y_walls=xb,
                    slant=(0.08, 0.03))
    else:
        raise ValueError(kind)
    s.set_source(WL, theta=theta, phi=phi)
    _pc._MORTAR_SOLVE_CENSUS = []
    try:
        err = None
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                o, R, T = s.solve(jones=False)
            R, T = np.atleast_2d(R), np.atleast_2d(T)
            clo = float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0)))
            nw = len(ws)
        except Exception as exc:                        # noqa: BLE001
            err = f"{type(exc).__name__}: {str(exc)[:90]}"
            clo, nw = float("nan"), -1
        rows = list(_pc._MORTAR_SOLVE_CENSUS)
    finally:
        _pc._MORTAR_SOLVE_CENSUS = None
    gen = [r[2] for r in rows if "GENERALIZED" in r[0]]
    plain = [r[2] for r in rows if "GENERALIZED" not in r[0]]
    return dict(error=err, closure=clo, n_warnings=nw,
                gen_rcond_min=(min(gen) if gen else None), n_gen=len(gen),
                plain_rcond_min=(min(plain) if plain else None),
                n_plain=len(plain),
                n=int(rows[0][1]) if rows else None)


XA = [0.2371 * P, 0.6183 * P]
XB = [0.3117 * P, 0.7402 * P]
XC = [0.21 * P, 0.55 * P]
XD = [0.30 * P, 0.70 * P]

out = {}
for kind in ("oop", "oop_both", "slant"):
    for M in (4, 5, 6, 7):
        for tag, (xa, xb) in (("nonconf", (XA, XB)), ("conf", (XA, XA)),
                              ("wide", (XC, XD))):
            key = f"{kind}_M{M}_{tag}"
            try:
                r = _run(kind, M, xa, xb)
            except Exception as exc:                    # noqa: BLE001
                r = {"error": f"{type(exc).__name__}: {str(exc)[:80]}"}
            out[key] = r
            gr = r.get("gen_rcond_min")
            txt = "  --  " if gr is None else f"{gr:.4e}"
            xb_ = "  --  " if gr is None else f"{gr / 1e-12:.2e}"
            _log(f"{key:24s} n={r.get('n')}  GEN rcond {txt}  (x bar {xb_})"
                 f"  closure {r.get('closure', float('nan')):.2e}"
                 f"  {r.get('error') or ''}")
g = [v["gen_rcond_min"] for v in out.values()
     if v.get("gen_rcond_min") is not None]
if g:
    _log(f"GENERALIZED healthy population: {len(g)} stacks, rcond "
         f"{min(g):.4e} .. {max(g):.4e};  closest approach to the 1e-12 bar "
         f"= {np.log10(min(g) / 1e-12):.2f} decades")
RES["generalized"] = out
RES["summary"] = dict(n=len(g), rcond_min=float(min(g)) if g else None,
                      rcond_max=float(max(g)) if g else None,
                      decades_to_bar=(float(np.log10(min(g) / 1e-12))
                                      if g else None))
p = os.path.join(HERE, f"v7_generalized_{TAG}.json")
with open(p, "w") as fh:
    json.dump({"lumenairy": lumenairy.__file__, **RES}, fh, indent=1,
              default=str)
_log(f"wrote {p}")
