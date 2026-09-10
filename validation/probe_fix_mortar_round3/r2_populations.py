"""ROUND 3, DEFECT V1 -- the two populations at the GENERALIZED mortar site.

Three instruments are read at EVERY generalized-site solve:

  * ``rcond``   -- the LAPACK ``gecon`` reciprocal 1-condition the shipped
                   guard screens on (a WORST-CASE-over-all-B quantity);
  * ``residual``-- ``||A X - B||_F / ||B||_F`` of the answer actually returned;
  * ``s_min/s_max`` and ``range_defect`` (``||u_min^H B|| / ||B||``), on the
    smaller operands only, for the mechanism table.

and three populations are built:

  ``healthy``  every ORDINARY per-layer stack that reaches the site: the v7
               census grid (out-of-plane / both-out-of-plane / both-slanted x
               M = 4..7 x nonconforming / conforming / wide), the v8 fixture,
               and the OOP-next-to-a-UNIFORM-SPACER variants.  Narrowest
               segment 0.237 of the period = 237x the width contract.
  ``sliver``   the SAME stacks with an intra-layer SLIVER, the width contract
               LIFTED (``PMM2D_STAG_MIN_SEG_GUARD = False``) -- the population
               the round-2 backstop exists for.  Scored against a device that
               CANNOT depend on the sliver's wall separation.
  ``singular`` synthetic operands of the site's own shape driven to EXACT
               singularity and to a deliberately INCONSISTENT right-hand side.

``python r2_populations.py [win|wsl]``
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

import _fixtures as F  # noqa: E402
import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402

import lumenairy  # noqa: E402
from lumenairy.elements.pmm import _core as _pc  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as _ts  # noqa: E402
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure  # noqa: E402

assert os.path.abspath(lumenairy.__file__).lower().startswith(_ROOT.lower()), (
    lumenairy.__file__)
TAG = (sys.argv[1] if len(sys.argv) > 1 else "win")
T0 = time.time()
_C = complex
print(f"[arm {TAG}] lumenairy = {lumenairy.__file__} v{lumenairy.__version__}",
      flush=True)


def _log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def _instr(A, B, svd_max=700):
    """rcond / residual / (optionally) the SVD pair, on ONE operand."""
    A = np.asarray(A)
    n = int(A.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            lu, piv = sla.lu_factor(A)
        except (ValueError, sla.LinAlgError, np.linalg.LinAlgError,
                sla.LinAlgWarning):
            return dict(n=n, rcond=0.0, residual=float("inf"), s_ratio=None,
                        range_defect=None, lu_failed=True)
        X = sla.lu_solve((lu, piv), B)
    anorm = float(np.max(np.sum(np.abs(A), axis=0)))
    gecon = sla.get_lapack_funcs("gecon", (A,))
    rcv, info = gecon(lu, anorm)
    rc = float(rcv) if int(info) == 0 else 0.0
    nB = float(np.linalg.norm(B))
    res = float(np.linalg.norm(A @ X - B) / nB) if nB > 0 else float("nan")
    row = dict(n=n, rcond=rc, residual=res, lu_failed=False)
    if n <= svd_max:
        U, s, _Vh = np.linalg.svd(A)
        row["s_ratio"] = float(s[-1] / s[0])
        row["range_defect"] = float(
            np.linalg.norm(U[:, -1].conj() @ B) / nB)
    else:
        row["s_ratio"] = None
        row["range_defect"] = None
    return row


class _Recorder:
    """Replaces ``_guarded_mortar_solve`` for the duration of one solve:
    records the instruments at the GENERALIZED site and NEVER refuses."""

    def __init__(self, svd_max=700):
        self.rows = []
        self.svd_max = svd_max
        self._orig = None

    def __enter__(self):
        self._orig = _pc._guarded_mortar_solve

        def solve(A, B, site, ga=None, gb=None, hint=None, screen="rcond"):
            if "GENERALIZED" not in site:
                return self._orig(A, B, site, ga, gb, hint, screen)
            self.rows.append(_instr(A, B, self.svd_max))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lu, piv = sla.lu_factor(np.asarray(A))
                return sla.lu_solve((lu, piv), B)

        _pc._guarded_mortar_solve = solve
        return self

    def __exit__(self, *a):
        _pc._guarded_mortar_solve = self._orig
        return False


def _solve(st, svd_max=700):
    with _Recorder(svd_max) as rec, warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        o, R, T = st.solve(jones=False)
    R2, T2 = np.atleast_2d(R), np.atleast_2d(T)
    return dict(rows=rec.rows,
                closure=float(np.max(np.abs(R2.sum(1) + T2.sum(1) - 1.0))),
                R00=F.zeroth_R(o, R), n_warnings=len(ws))


# ======================================================================
# 1.  HEALTHY -- the ordinary stacks that reach the site
# ======================================================================
def healthy():
    out = {}
    XC = (0.21, 0.55)
    XD = (0.30, 0.70)
    for kind in ("oop_scalar", "oop_spacer", "oop_both", "slant_both"):
        for M in (4, 5, 6, 7):
            for tag, wb in (("nonconf", F.WB), ("conf", F.WA), ("wide", XD)):
                if kind == "oop_spacer" and tag != "nonconf":
                    continue
                key = f"{kind}_M{M}_{tag}"
                try:
                    st = F.build(kind, M, walls_b=wb)
                    if tag == "wide":
                        # the v7 'wide' variant moves BOTH wall arrays
                        st = None
                        st = PMM2DStackPure(
                            F.P, n_modes=M, n_orders=2, n_substrate=1.5,
                            layer_grids="per-layer")
                        _build_wide(st, kind, XC, XD)
                        st.set_source(F.WL, theta=F.THETA, phi=F.PHI)
                    r = _solve(st)
                except Exception as exc:                    # noqa: BLE001
                    out[key] = {"error": f"{type(exc).__name__}: "
                                         f"{str(exc)[:110]}"}
                    _log(f"  {key:26s} {out[key]['error'][:90]}")
                    continue
                out[key] = r
                g = r["rows"]
                if not g:
                    # CONFORMING grids: the identical-grid bypass means this
                    # stack never reaches the generalized mortar at all.
                    _log(f"  {key:26s} no generalized mortar (conforming) "
                         f"closure {r['closure']:.2e}")
                    continue
                _log(f"  {key:26s} n={g[0]['n']:5d} rcond {g[0]['rcond']:.4e} "
                     f"resid {g[0]['residual']:.3e} closure {r['closure']:.2e}")
    return out


def _build_wide(st, kind, xa, xb):
    sa, sb = [w * F.P for w in xa], [w * F.P for w in xb]
    if kind == "oop_scalar":
        st.add_layer(0.13e-6, eps_cell=F.tensor_cell(xa, xa[0], xa[1],
                                                     F.E_OOP, F.EPS_H),
                     x_walls=sa, y_walls=sa)
        st.add_layer(0.10e-6, eps_cell=F.scalar_cell(xb, xb[0], xb[1],
                                                     F.EPS_B, F.EPS_H),
                     x_walls=sb, y_walls=sb)
    elif kind == "oop_both":
        st.add_layer(0.13e-6, eps_cell=F.tensor_cell(xa, xa[0], xa[1],
                                                     F.E_OOP, F.EPS_H),
                     x_walls=sa, y_walls=sa)
        st.add_layer(0.10e-6, eps_cell=F.tensor_cell(xb, xb[0], xb[1],
                                                     0.7 * F.E_OOP, F.EPS_H),
                     x_walls=sb, y_walls=sb)
    elif kind == "slant_both":
        st.add_layer(0.13e-6, eps_cell=F.scalar_cell(xa, xa[0], xa[1],
                                                     F.EPS_B, F.EPS_H),
                     x_walls=sa, y_walls=sa, slant=(0.08, 0.03))
        st.add_layer(0.10e-6, eps_cell=F.scalar_cell(xb, xb[0], xb[1],
                                                     4.0, F.EPS_H),
                     x_walls=sb, y_walls=sb, slant=(0.08, 0.03))
    else:
        raise ValueError(kind)


# ======================================================================
# 2.  SLIVER -- the width contract LIFTED
# ======================================================================
def _sliver_stack(delta, M, centre=0.44):
    """Three layers on the GENERALIZED cascade whose MIDDLE layer is ALL HOST
    and carries the sliver, so the DEVICE cannot depend on ``delta`` at all.

    Layer 1 is an OUT-OF-PLANE patterned layer (this is what puts the stack on
    the generalized cascade); layer 3 is an ordinary scalar pattern on other
    walls.  Every interface is a generalized MORTAR."""
    st = PMM2DStackPure(F.P, n_modes=M, n_orders=2, n_substrate=1.5,
                        layer_grids="per-layer")
    st.add_layer(0.13e-6,
                 eps_cell=F.tensor_cell(F.WA, F.WA[0], F.WA[1], F.E_OOP,
                                        F.EPS_H),
                 x_walls=[w * F.P for w in F.WA],
                 y_walls=[w * F.P for w in F.WA])
    sw = (centre, centre + delta)
    st.add_layer(0.09e-6,
                 eps_cell=np.full((3, 3), _C(F.EPS_H)),
                 x_walls=[w * F.P for w in sw],
                 y_walls=[w * F.P for w in F.WA])
    st.add_layer(0.10e-6,
                 eps_cell=F.scalar_cell(F.WB, F.WB[0], F.WB[1], F.EPS_B,
                                        F.EPS_H),
                 x_walls=[w * F.P for w in F.WB],
                 y_walls=[w * F.P for w in F.WB])
    st.set_source(F.WL, theta=F.THETA, phi=F.PHI)
    return st


def sliver():
    out = {}
    prev = _ts.PMM2D_STAG_MIN_SEG_GUARD
    _ts.PMM2D_STAG_MIN_SEG_GUARD = False
    try:
        for M in (4, 5):
            ref = None
            for delta in (3e-1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7,
                          1e-9):
                key = f"sliver_M{M}_d{delta:.0e}"
                try:
                    r = _solve(_sliver_stack(delta, M))
                except Exception as exc:                    # noqa: BLE001
                    out[key] = {"error": f"{type(exc).__name__}: "
                                         f"{str(exc)[:110]}"}
                    _log(f"  {key:24s} {out[key]['error'][:90]}")
                    continue
                if ref is None:
                    ref = r["R00"]
                r["delta"] = delta
                r["dev_from_ordinary"] = abs(r["R00"] - ref) / abs(ref)
                out[key] = r
                g = r["rows"]
                _log(f"  {key:24s} {len(g)} solves  worst rcond "
                     f"{min(x['rcond'] for x in g):.4e}  worst resid "
                     f"{max(x['residual'] for x in g):.3e}  R00 {r['R00']:.10f}"
                     f"  dev {r['dev_from_ordinary']:.3e}  closure "
                     f"{r['closure']:.2e}  warn {r['n_warnings']}")
    finally:
        _ts.PMM2D_STAG_MIN_SEG_GUARD = prev
    return out


# ======================================================================
# 3.  SINGULAR / INCONSISTENT -- synthetic, on the site's own operands
# ======================================================================
def singular():
    """Take a REAL generalized-site operand and break it two ways."""
    grab = {}
    orig = _pc._guarded_mortar_solve

    def solve(A, B, site, ga=None, gb=None, hint=None, screen="rcond"):
        if "GENERALIZED" in site and "op" not in grab:
            grab["op"] = (np.array(A), np.array(B))
        return orig(A, B, site, ga, gb, hint, screen)

    _pc._guarded_mortar_solve = solve
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            F.build("oop_both", 4).solve(jones=False)
    finally:
        _pc._guarded_mortar_solve = orig
    A, B = grab["op"]
    out = {}
    out["real"] = _instr(A, B)
    # (a) EXACTLY singular: a repeated column (the shape that raised a bare
    #     LinAlgError in round 2)
    A2 = A.copy()
    A2[:, 3] = A2[:, 7]
    out["exact_singular"] = _instr(A2, B)
    # (b) RANK-DEFICIENT but CONSISTENT: same operand, RHS drawn from its own
    #     range (A @ Z), which is what the mixed in-plane/out-of-plane site
    #     builds
    rng = np.random.default_rng(20260911)
    Z = (rng.standard_normal(B.shape) + 1j * rng.standard_normal(B.shape))
    out["rank_def_consistent"] = _instr(A2, A2 @ Z)
    # (c) RANK-DEFICIENT and INCONSISTENT: a RHS with a component OUTSIDE the
    #     range (the left null vector of the repeated-column operand)
    U, s, Vh = np.linalg.svd(A2)
    bad = (A2 @ Z) + float(np.linalg.norm(A2 @ Z)) * np.outer(
        U[:, -1], np.ones(B.shape[1]) / np.sqrt(B.shape[1]))
    out["rank_def_inconsistent"] = _instr(A2, bad)
    for k, v in out.items():
        _log(f"  {k:22s} rcond {v['rcond']:.4e}  resid {v['residual']:.4e}"
             f"  s_ratio {v['s_ratio']}")
    return out


RES = {}
_log("HEALTHY population")
RES["healthy"] = healthy()
_log("SLIVER population (width contract LIFTED)")
RES["sliver"] = sliver()
_log("SINGULAR / INCONSISTENT synthetics")
RES["singular"] = singular()

# --- summary -----------------------------------------------------------
h = [r["rows"][0] for r in RES["healthy"].values() if r.get("rows")]
_log(f"HEALTHY: {len(h)} stacks  rcond {min(x['rcond'] for x in h):.4e} .. "
     f"{max(x['rcond'] for x in h):.4e}   residual "
     f"{min(x['residual'] for x in h):.3e} .. "
     f"{max(x['residual'] for x in h):.3e}")
sl = [x for r in RES["sliver"].values() if "rows" in r for x in r["rows"]]
if sl:
    _log(f"SLIVER : {len(sl)} solves  rcond {min(x['rcond'] for x in sl):.4e}"
         f" .. {max(x['rcond'] for x in sl):.4e}   residual "
         f"{min(x['residual'] for x in sl):.3e} .. "
         f"{max(x['residual'] for x in sl):.3e}")
RES["summary"] = dict(
    healthy_n=len(h),
    healthy_rcond=[min(x["rcond"] for x in h), max(x["rcond"] for x in h)],
    healthy_resid=[min(x["residual"] for x in h),
                   max(x["residual"] for x in h)],
    sliver_n=len(sl),
    sliver_rcond=([min(x["rcond"] for x in sl),
                   max(x["rcond"] for x in sl)] if sl else None),
    sliver_resid=([min(x["residual"] for x in sl),
                   max(x["residual"] for x in sl)] if sl else None))
p = os.path.join(_HERE, f"r2_populations_{TAG}.json")
with open(p, "w") as fh:
    json.dump({"tag": TAG, "lumenairy": lumenairy.__file__, **RES}, fh,
              indent=1, default=str)
_log(f"wrote {p}")
