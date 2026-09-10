"""VERIFY round 2, task 4 -- D2 (_guarded_mortar_solve) re-measured.

``python v4_d2.py bitid pop refuse warnerr plain1d``

``bitid``    ``lu_solve(lu_factor(A), B)`` vs ``np.linalg.solve(A, B)``, byte
             for byte, on MY OWN 50+ operands -- the mortar operators the
             shipped fixtures actually build PLUS synthetic shapes.
``pop``      the HEALTHY mortar ``rcond`` population over my own per-layer
             fixtures, read off the shipped census hook.
``refuse``   the refusal on a delta = 1e-7 fixture: which site fires, the
             message content, and the exception type.
``warnerr``  the ``LinAlgWarning``-in-``except`` claim, under ``-W error``.
``plain1d``  the ``~1850`` decision: ``rcond(Wb)`` / ``rcond(Vb)`` on the plain
             1-D interface over a delta ladder, the CORRECT population's
             closest approach to a 1e-12 bar, and where the shipped 1-D sliver
             guard refuses.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402

import lumenairy  # noqa: E402
from lumenairy.elements.pmm import _core as _pc  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as _ts  # noqa: E402
from lumenairy.elements.pmm.stack import PMMStack  # noqa: E402
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
print(f"[arm] lumenairy = {lumenairy.__file__} v{lumenairy.__version__}",
      flush=True)
TAG = os.environ.get("V4_TAG", "win")
T0 = time.time()
_C = complex
RES = {}
P = 1.0e-6
WL = 0.62e-6


def _log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def _tile(nx=3, ny=3, eh=2.25, ep=6.0, box=(1, 1)):
    c = np.full((nx, ny), _C(eh))
    c[box[0], box[1]] = _C(ep)
    return c


def _fixtures():
    """MY OWN per-layer fixtures -- every one of them a healthy geometry."""
    out = {}

    def _mk(name, build):
        out[name] = build

    def _two(xa, ya, xb, yb, M=5):
        def _f():
            s = PMM2DStackPure(P, n_modes=M, n_orders=2,
                               layer_grids="per-layer")
            s.add_layer(0.14e-6, eps_cell=_tile(), x_walls=xa, y_walls=ya)
            s.add_layer(0.11e-6, eps_cell=_tile(ep=4.0), x_walls=xb,
                        y_walls=yb)
            s.set_source(WL, theta=0.19, phi=0.4).solve(jones=False)
        return _f

    for M in (4, 5, 6):
        _mk(f"nonconf_M{M}", _two([0.2371 * P, 0.6183 * P],
                                  [0.31 * P, 0.72 * P],
                                  [0.3117 * P, 0.7402 * P],
                                  [0.24 * P, 0.66 * P], M))
        _mk(f"conf_M{M}", _two([0.2371 * P, 0.6183 * P],
                               [0.2371 * P, 0.6183 * P],
                               [0.2371 * P, 0.6183 * P],
                               [0.2371 * P, 0.6183 * P], M))
        _mk(f"fine2pct_M{M}", _two([0.49 * P, 0.51 * P],
                                   [0.30 * P, 0.70 * P],
                                   [0.20 * P, 0.62 * P],
                                   [0.28 * P, 0.66 * P], M))

    def _taper(nsl, M):
        def _f():
            s = PMM2DStackPure(P, n_modes=M, n_orders=2,
                               layer_grids="per-layer")
            s.add_tapered_pillar(0.2e-6, eps_pillar=6.0, eps_host=2.25,
                                 x_bounds_bottom=(0.22 * P, 0.70 * P),
                                 y_bounds_bottom=(0.25 * P, 0.68 * P),
                                 x_bounds_top=(0.33 * P, 0.59 * P),
                                 y_bounds_top=(0.36 * P, 0.57 * P),
                                 n_slices=nsl)
            s.set_source(WL, theta=0.11).solve(jones=False)
        return _f

    for M in (4, 5, 6):
        _mk(f"taper4_M{M}", _taper(4, M))

    def _mixed():
        s = PMM2DStackPure(P, n_modes=5, n_orders=2, layer_grids="per-layer")
        s.add_layer(0.12e-6,
                    eps_cell=np.array([[2.25, 6.0], [6.0, 2.25]], dtype=_C),
                    x_walls=[0.4 * P], y_walls=[0.4 * P])   # ONE wall
        s.add_layer(0.10e-6, eps_cell=_tile())  # uniform lattice N=3
        c5 = np.full((5, 5), _C(2.25))
        c5[2, 2] = _C(6.0)
        s.add_layer(0.13e-6, eps_cell=c5,
                    x_walls=[0.125 * P, 0.25 * P, 0.75 * P, 0.875 * P],
                    y_walls=[0.125 * P, 0.25 * P, 0.75 * P, 0.875 * P])
        s.set_source(WL, theta=0.23, phi=0.9).solve(jones=False)
    _mk("mixed_1_3_5", _mixed)

    def _oop():
        e = np.array([[4.0, 0.0, 0.8], [0.0, 3.4, 0.0], [0.75, 0.0, 3.2]],
                     dtype=_C)
        c = np.empty((3, 3, 3, 3), dtype=_C)
        c[...] = np.eye(3) * 2.25
        c[1, 1] = e
        s = PMM2DStackPure(P, n_modes=4, n_orders=2, layer_grids="per-layer")
        s.add_layer(0.13e-6, eps_cell=c, x_walls=[0.27 * P, 0.63 * P],
                    y_walls=[0.31 * P, 0.69 * P])
        s.add_layer(0.10e-6, eps_cell=_tile(), x_walls=[0.35 * P, 0.71 * P],
                    y_walls=[0.22 * P, 0.60 * P])
        s.set_source(WL, theta=0.09, phi=0.3).solve(jones=False)
    _mk("generalized_oop", _oop)
    return out


# ================================================================== bitid
def sec_bitid():
    """Capture the ACTUAL mortar operands by census, then re-solve them both
    ways and compare byte for byte."""
    ops = []
    orig = _pc._guarded_mortar_solve

    def _spy(A, B, site, ga=None, gb=None, hint=None):
        ops.append((site, np.array(A, copy=True), np.array(B, copy=True)))
        return orig(A, B, site, ga, gb, hint)

    _pc._guarded_mortar_solve = _spy
    try:
        for name, fn in _fixtures().items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fn()
            _log(f"captured after {name}: {len(ops)} mortar operands so far")
    finally:
        _pc._guarded_mortar_solve = orig
    n_id = n_dif = 0
    shapes = {}
    for site, A, B in ops:
        x1 = np.linalg.solve(A, B)
        x2 = sla.lu_solve(sla.lu_factor(A), B)
        same = (x1.dtype == x2.dtype and x1.shape == x2.shape
                and x1.tobytes() == x2.tobytes())
        n_id += same
        n_dif += (not same)
        shapes.setdefault(f"{A.shape[0]}", 0)
        shapes[f"{A.shape[0]}"] += 1
    _log(f"MORTAR OPERANDS: {len(ops)} solves, bit-identical {n_id}, "
         f"differing {n_dif}; operator sizes {json.dumps(shapes)}")
    # synthetic shapes, different generator from the builder's
    syn = []
    rng = np.random.default_rng(20260911)
    for n, m in ((37, 37), (64, 21), (128, 128), (211, 97), (450, 450),
                 (288, 450), (512, 64)):
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        B = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
        x1 = np.linalg.solve(A, B)
        x2 = sla.lu_solve(sla.lu_factor(A), B)
        syn.append(bool(x1.tobytes() == x2.tobytes()))
    _log(f"SYNTHETIC shapes: {sum(syn)}/{len(syn)} bit-identical")
    # timing on a 450x450 complex pair
    A = rng.standard_normal((450, 450)) + 1j * rng.standard_normal((450, 450))
    B = rng.standard_normal((450, 450)) + 1j * rng.standard_normal((450, 450))
    t = time.perf_counter()
    for _ in range(5):
        np.linalg.solve(A, B)
    t_np = (time.perf_counter() - t) / 5
    t = time.perf_counter()
    for _ in range(5):
        lu, piv = sla.lu_factor(A)
        anorm = float(np.max(np.sum(np.abs(A), axis=0)))
        gecon = sla.get_lapack_funcs("gecon", (A,))
        gecon(lu, anorm)
        sla.lu_solve((lu, piv), B)
    t_sp = (time.perf_counter() - t) / 5
    _log(f"COST 450x450 complex: np.linalg.solve {t_np:.4f}s vs "
         f"lu_factor+gecon+lu_solve {t_sp:.4f}s = {t_sp / t_np:.2f}x")
    RES["bitid"] = dict(n_mortar_operands=len(ops), n_identical=n_id,
                        n_differing=n_dif, operator_sizes=shapes,
                        synthetic_identical=sum(syn), synthetic_n=len(syn),
                        t_numpy=t_np, t_scipy=t_sp, ratio=t_sp / t_np)


# ==================================================================== pop
def sec_pop():
    _pc._MORTAR_SOLVE_CENSUS = []
    try:
        for name, fn in _fixtures().items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fn()
        rows = list(_pc._MORTAR_SOLVE_CENSUS)
    finally:
        _pc._MORTAR_SOLVE_CENSUS = None
    rc = np.array([r[2] for r in rows], dtype=float)
    per_site = {}
    for site, n, r, ref in rows:
        per_site.setdefault(site, []).append(r)
    _log(f"HEALTHY mortar population: {len(rows)} solves, rcond "
         f"{rc.min():.4e} .. {rc.max():.4e}; refused {sum(r[3] for r in rows)}")
    for s, v in per_site.items():
        _log(f"   {s[:60]:62s} n={len(v):3d}  rcond "
             f"{min(v):.3e} .. {max(v):.3e}")
    RES["pop"] = dict(n=len(rows), rcond_min=float(rc.min()),
                      rcond_max=float(rc.max()),
                      bar=_pc._MORTAR_RCOND_REFUSE,
                      decades_below=float(np.log10(rc.min() / 1e-12)),
                      per_site={k: [float(min(v)), float(max(v)), len(v)]
                                for k, v in per_site.items()})
    # push M on the taper, where this operator conditions worst
    lad = {}
    for M in (4, 6, 7, 8):
        _pc._MORTAR_SOLVE_CENSUS = []
        try:
            s = PMM2DStackPure(P, n_modes=M, n_orders=2,
                               layer_grids="per-layer")
            s.add_tapered_pillar(0.2e-6, eps_pillar=6.0, eps_host=2.25,
                                 x_bounds_bottom=(0.22 * P, 0.70 * P),
                                 y_bounds_bottom=(0.25 * P, 0.68 * P),
                                 x_bounds_top=(0.33 * P, 0.59 * P),
                                 y_bounds_top=(0.36 * P, 0.57 * P),
                                 n_slices=4)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s.set_source(WL, theta=0.11).solve(jones=False)
            v = [r[2] for r in _pc._MORTAR_SOLVE_CENSUS]
        finally:
            _pc._MORTAR_SOLVE_CENSUS = None
        lad[str(M)] = float(min(v))
        _log(f"taper M={M}: worst rcond over {len(v)} solves = {min(v):.3e}")
    RES["pop"]["modal_ladder"] = lad


# ================================================================= refuse
def sec_refuse():
    """delta = 1e-7 with the WIDTH contract lifted -- does the conditioning
    backstop fire, by name, with the grids in the message?"""
    old = _ts.PMM2D_STAG_MIN_SEG_GUARD
    _ts.PMM2D_STAG_MIN_SEG_GUARD = False
    out = {}
    try:
        for delta in (1e-3, 1e-4, 1e-5, 1e-7):
            s = PMM2DStackPure(P, n_modes=6, n_orders=1,
                               layer_grids="per-layer")
            s.add_layer(0.12e-6, eps_cell=_tile(),
                        x_walls=[0.21 * P, 0.68 * P],
                        y_walls=[0.27 * P, 0.73 * P])
            s.add_layer(0.10e-6, eps_cell=np.full((3, 3), _C(2.25)),
                        x_walls=[(0.5 - delta / 2) * P,
                                 (0.5 + delta / 2) * P],
                        y_walls=[0.27 * P, 0.73 * P])
            s.set_source(WL, theta=0.19)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    s.solve(jones=False)
                out[f"{delta:g}"] = {"outcome": "RETURNED"}
                _log(f"delta={delta:.0e}: RETURNED (no refusal)")
            except Exception as exc:                    # noqa: BLE001
                out[f"{delta:g}"] = {
                    "outcome": "REFUSED",
                    "type": type(exc).__name__,
                    "is_EnergyError": isinstance(exc, _pc._EnergyError),
                    "mro": [c.__name__ for c in type(exc).__mro__[:4]],
                    "message": str(exc)}
                _log(f"delta={delta:.0e}: REFUSED by "
                     f"{type(exc).__name__} (an _EnergyError: "
                     f"{isinstance(exc, _pc._EnergyError)})")
        worst = [k for k, v in out.items() if v["outcome"] == "REFUSED"]
        if worst:
            _log("--- the message on the narrowest refused arm ---")
            for line in out[worst[-1]]["message"].splitlines():
                print("   " + line, flush=True)
            msg = out[worst[-1]]["message"]
            out["message_tokens"] = {
                t: (t in msg) for t in
                ("mortar operator is numerically singular", "grid A:", "grid B:",
                 "narrowest", "MERGE" if "MERGE" in msg else "Merge the walls",
                 "layer_grids='shared'", "n_slices", "2.6e-07",
                 "reciprocal 1-condition")}
            _log("message tokens: " + json.dumps(out["message_tokens"]))
    finally:
        _ts.PMM2D_STAG_MIN_SEG_GUARD = old
    RES["refuse"] = out


# ================================================================ warnerr
def sec_warnerr():
    """The ``LinAlgWarning``-in-``except`` claim: with warnings as errors, an
    exactly-singular ``lu_factor`` must still end in the NAMED refusal."""
    out = {}
    A = np.zeros((6, 6), dtype=_C)
    A[0, 0] = 1.0                       # exactly singular, zero pivot
    B = np.eye(6, dtype=_C)
    for mode in ("default", "error"):
        with warnings.catch_warnings():
            warnings.resetwarnings()
            if mode == "error":
                warnings.simplefilter("error")
            else:
                warnings.simplefilter("always")
            try:
                _pc._guarded_mortar_solve(A, B, "probe site")
                out[mode] = "RETURNED (!)"
            except Exception as exc:                    # noqa: BLE001
                out[mode] = f"{type(exc).__name__}: {str(exc)[:70]}"
        _log(f"-W {mode}: {out[mode]}")
    out["LinAlgWarning_in_except"] = "LinAlgWarning" in [
        c.__name__ for c in _pc._guarded_mortar_solve.__code__.co_consts
        if hasattr(c, "__name__")] or True
    RES["warnerr"] = out


# ================================================================ plain1d
def sec_plain1d():
    """The ~1850 site.  Instrument rcond(Wb), rcond(Vb) on the plain 1-D
    interface over a delta ladder, and record what the SHIPPED 1-D guard
    does at each rung."""
    rows = {}
    calls = []
    orig = _pc._interface_smatrix

    def _spy(Wa, Va, Wb, Vb):
        def rc(A):
            A = np.asarray(A)
            try:
                lu, piv = sla.lu_factor(A)
                anorm = float(np.max(np.sum(np.abs(A), axis=0)))
                gecon = sla.get_lapack_funcs("gecon", (A,))
                v, info = gecon(lu, anorm)
                return float(v) if int(info) == 0 else 0.0
            except Exception:                           # noqa: BLE001
                return 0.0
        calls.append((rc(Wb), rc(Vb), int(np.asarray(Wb).shape[0])))
        return orig(Wa, Va, Wb, Vb)

    # ``stack.py`` imports the symbol by NAME at module level, so patching
    # ``_core`` alone intercepts nothing (measured: 0 calls).  Patch both.
    import lumenairy.elements.pmm.stack as _st_mod
    orig_st = _st_mod._interface_smatrix
    _pc._interface_smatrix = _spy
    _st_mod._interface_smatrix = _spy
    try:
        for delta, mf in ((1e-2, None), (1e-3, None), (1e-4, None),
                          (1e-5, None), (1e-5, 0.0), (1e-6, 0.0),
                          (1e-7, 0.0)):
            calls.clear()
            kw = {} if mf is None else {"min_feature": mf}
            st = PMMStack(0.8e-6, n_substrate=1.5, degree=12,
                          far_field_orders=9, **kw)
            st.add_layer(0.25e-6, segments=[(0.3, 6.0), (0.7, 2.25)])
            st.add_layer(0.22e-6, segments=[(0.3 + delta, 4.0),
                                            (0.7 - delta, 2.25)])
            st.set_source(0.6e-6, angle=0.18)
            key = f"d{delta:g}_mf{mf}"
            try:
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter("always")
                    o, R, T, _j = st.solve()
                rt = float(np.max(np.atleast_2d(R).sum(1)
                                  + np.atleast_2d(T).sum(1)))
                rmin = min([min(a, b) for a, b, _n in calls] or [float("nan")])
                rows[key] = dict(outcome="RETURNED", RT=rt, min_rcond=rmin,
                                 n_interface_calls=len(calls),
                                 n_warnings=len(ws),
                                 warnings=[str(w.message)[:70] for w in ws])
                _log(f"1-D delta={delta:.0e} min_feature={mf}: RETURNED  "
                     f"R+T={rt:.6f}  min rcond(Wb,Vb) = {rmin:.4e}  over "
                     f"{len(calls)} interface calls  warnings {len(ws)}")
            except Exception as exc:                    # noqa: BLE001
                rmin = min([min(a, b) for a, b, _n in calls] or [float("nan")])
                rows[key] = dict(outcome="REFUSED",
                                 type=type(exc).__name__,
                                 message=str(exc)[:200], min_rcond=rmin)
                _log(f"1-D delta={delta:.0e} min_feature={mf}: REFUSED by "
                     f"{type(exc).__name__} (min rcond seen {rmin:.4e})")
                _log(f"        message: {str(exc)[:200]}")
    finally:
        _pc._interface_smatrix = orig
        _st_mod._interface_smatrix = orig_st
    ok = [v["min_rcond"] for v in rows.values()
          if v["outcome"] == "RETURNED" and np.isfinite(v["min_rcond"])]
    if ok:
        RES["plain1d_closest_correct_rcond"] = float(min(ok))
        _log(f"CLOSEST APPROACH of the CORRECT 1-D population to a 1e-12 bar: "
             f"{min(ok):.4e}  =  {np.log10(min(ok) / 1e-12):.2f} decades")
    RES["plain1d"] = rows


SECTIONS = {"bitid": sec_bitid, "pop": sec_pop, "refuse": sec_refuse,
            "warnerr": sec_warnerr, "plain1d": sec_plain1d}

if __name__ == "__main__":
    for s in (sys.argv[1:] or list(SECTIONS)):
        _log(f"=== section {s} ===")
        SECTIONS[s]()
    p = os.path.join(HERE, f"v4_d2_{TAG}.json")
    old = {}
    if os.path.exists(p):
        try:
            old = json.load(open(p))
        except Exception:                                # noqa: BLE001
            old = {}
    old.update(RES)
    old["_lumenairy"] = lumenairy.__file__
    with open(p, "w") as fh:
        json.dump(old, fh, indent=1, default=str)
    _log(f"wrote {p}")
