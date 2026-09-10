"""VERIFY round 2, task 3b -- BREAK THE WIDTH GUARD, both ways.

``python v3_guard.py census boundary exempt shared falsepos degraded``

``census``    MY OWN set of ordinary per-layer / taper geometries, with the
              narrowest segment fraction read off the census hook -- i.e. off
              what ``Basis1D`` actually receives, not off arithmetic.  Each
              geometry is SOLVED, because ``add_layer`` only records the wall
              array: the contract is reached at ``solve`` time.
``boundary``  the 1e-9 relative slack: exactly 1e-3 accepted, 1e-3 - 2e-9
              refused, and the shipped sliver-verify fixture that sat 1.8e-16
              under the bar.
``exempt``    the INTEGER-lattice exemption and the N > 1000 arithmetic.
``shared``    the shared path refuses ``x_walls`` (so the guard cannot apply).
``falsepos``  legitimate geometries the bar REFUSES -- closing tapers, thin
              ridges -- and what the message tells the user.
``degraded``  the load-bearing question: a user at 2e-3, just ABOVE the bar.
              Is the returned answer within the physical shift, or silently
              degraded?  Scored against the exact 1-D oracle on a device that
              CANNOT depend on the wall separation.
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
from lumenairy.elements.pmm import twod_staggered as _ts  # noqa: E402
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import Basis1D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
print(f"[arm] lumenairy = {lumenairy.__file__} v{lumenairy.__version__}",
      flush=True)
TAG = os.environ.get("V3_TAG", "win")
T0 = time.time()
_C = complex
RES = {}
BAR = _ts._STAG_MIN_SEG_FRAC
WL = 0.62e-6
P = 1.0e-6


def _log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def _census(build):
    """Run ``build`` with the census hook armed; return the narrowest segment
    fraction ``Basis1D`` actually SAW, and whether anything was refused."""
    _ts._STAG_SEG_CENSUS = []
    try:
        err = None
        try:
            build()
        except Exception as exc:                        # noqa: BLE001
            err = f"{type(exc).__name__}: {str(exc)[:90]}"
        rows = list(_ts._STAG_SEG_CENSUS)
    finally:
        _ts._STAG_SEG_CENSUS = None
    fr = [r[3] for r in rows]
    return dict(n_nonuniform_bases=len(rows),
                narrowest=(min(fr) if fr else None),
                refused=any(r[4] for r in rows), error=err)


def _st(**kw):
    kw.setdefault("n_modes", 3)
    kw.setdefault("n_orders", 1)
    kw.setdefault("layer_grids", "per-layer")
    return PMM2DStackPure(P, **kw)


def _tile3(eh=2.25, ep=6.0):
    c = np.full((3, 3), _C(eh))
    c[1, 1] = _C(ep)
    return c


def _go(s):
    s.set_source(WL, theta=0.1).solve(jones=False)


# ================================================================= census
def sec_census():
    geoms = {}

    def g(name, fn):
        geoms[name] = _census(fn)
        r = geoms[name]
        w = r["narrowest"]
        _log(f"{name:50s} narrowest {('  --  ' if w is None else f'{w:.4e}')}"
             f"  x bar {('  --  ' if w is None else f'{w / BAR:8.1f}')}"
             f"  bases {r['n_nonuniform_bases']:4d}  refused={r['refused']}"
             f"  {(r['error'] or '')[:40]}")

    def _walls(xw, yw=None):
        def _f():
            s = _st()
            s.add_layer(0.1e-6, eps_cell=_tile3(), x_walls=xw,
                        y_walls=(yw if yw is not None else xw))
            _go(s)
        return _f

    def _one_wall():
        s = _st()
        s.add_layer(0.1e-6,
                    eps_cell=np.array([[2.25, 6.0], [6.0, 2.25]], dtype=_C),
                    x_walls=[0.4 * P], y_walls=[0.4 * P])
        _go(s)

    def _nested():
        s = _st()
        c = np.full((5, 5), _C(2.25))
        c[2, 2] = _C(6.0)
        s.add_layer(0.1e-6, eps_cell=c,
                    x_walls=[0.125 * P, 0.25 * P, 0.75 * P, 0.875 * P],
                    y_walls=[0.125 * P, 0.25 * P, 0.75 * P, 0.875 * P])
        _go(s)

    g("single interior wall at 0.4", _one_wall)
    g("duty-1/3 pair", _walls([P / 3, 2 * P / 3]))
    g("conforming 0.2371/0.6183", _walls([0.2371 * P, 0.6183 * P]))
    g("non-conforming 0.3117/0.7402", _walls([0.3117 * P, 0.7402 * P]))
    g("axes carrying different walls",
      _walls([0.21 * P, 0.55 * P], [0.33 * P, 0.78 * P]))
    g("nested refinement 0.125/0.25/0.75/0.875", _nested)

    def _taper(nsl, xb0=(0.22, 0.70), xb1=(0.33, 0.59)):
        def _f():
            s = _st()
            s.add_tapered_pillar(0.2e-6, eps_pillar=6.0, eps_host=2.25,
                                 x_bounds_bottom=(xb0[0] * P, xb0[1] * P),
                                 y_bounds_bottom=(xb0[0] * P, xb0[1] * P),
                                 x_bounds_top=(xb1[0] * P, xb1[1] * P),
                                 y_bounds_top=(xb1[0] * P, xb1[1] * P),
                                 n_slices=nsl)
            _go(s)
        return _f

    for nsl in (4, 8, 16, 32, 64):
        g(f"straight-ish taper n_slices={nsl}", _taper(nsl))

    def _tapers(nsl, wtop=0.10):
        def _f():
            s = _st()
            s.add_tapered_pillars(
                0.18e-6,
                pillars=[((0.5 * P, 0.5 * P), (wtop * P, wtop * P),
                          (0.40 * P, 0.40 * P), 7.0)],
                eps_host=2.25, n_slices=nsl)
            _go(s)
        return _f

    for nsl in (4, 6, 16):
        g(f"add_tapered_pillars n_slices={nsl}", _tapers(nsl))

    for edge in (0.05, 0.01, 2e-3, 1.1e-3, 1.0e-3, 9e-4, 1e-4):
        g(f"pillar edge {edge:g} from the period end",
          _walls([edge * P, (1.0 - edge) * P]))

    RES["census"] = geoms


# =============================================================== boundary
def sec_boundary():
    out = {}
    base = 0.2371

    def _try(frac, note):
        w = [base * P, (base + frac) * P]
        try:
            Basis1D(P, np.array([0.0, w[0], w[1], P]), 4)
            got, msg = "ACCEPTED", ""
        except ValueError as exc:
            got, msg = "REFUSED", str(exc)[:110]
        actual = float((w[1] - w[0]) / P)
        out[note] = dict(requested=frac, actual_fraction=actual,
                         actual_minus_bar=actual - BAR, outcome=got, msg=msg)
        _log(f"{note:36s} actual {actual:.16e} ({actual - BAR:+.3e} vs bar) "
             f" {got}")
        return got

    _try(1.0e-3, "exactly 1e-3")
    _try(1.0e-3 - 2e-9, "1e-3 - 2e-9  (absolute)")
    _try(1.0e-3 * (1 - 5e-10), "1e-3 * (1 - 5e-10)  inside slack")
    _try(1.0e-3 * (1 - 1e-9), "1e-3 * (1 - 1e-9)   ON the slack")
    _try(1.0e-3 * (1 - 2e-9), "1e-3 * (1 - 2e-9)   OUTSIDE slack")
    _try(1.0e-3 * (1 - 1e-6), "1e-3 * (1 - 1e-6)")
    _try(1.001e-3, "1.001e-3")

    a, b = 0.28452, 0.28572
    frac = (b - a) / 1.2
    out["shipped_fixture_arithmetic"] = dict(
        expr="(0.28572 - 0.28452) / 1.2", value=frac,
        minus_bar=frac - 1e-3, below_bar=bool(frac < 1e-3))
    _log(f"SHIPPED fixture arithmetic (0.28572-0.28452)/1.2 = {frac:.17e} "
         f"({frac - 1e-3:+.3e} vs 1e-3) -> strictly below the bar: "
         f"{frac < 1e-3}")
    try:
        Basis1D(1.2e-6, np.array([0.0, a * 1.2e-6, b * 1.2e-6, 1.2e-6]), 4)
        out["shipped_fixture_outcome"] = "ACCEPTED"
    except ValueError as exc:
        out["shipped_fixture_outcome"] = f"REFUSED: {str(exc)[:90]}"
    _log(f"SHIPPED fixture geometry -> {out['shipped_fixture_outcome']}")

    # the SWITCH
    sw = {}
    walls = np.array([0.0, 0.4999 * P, 0.5001 * P, P])   # 2e-4 of the period
    for state in (True, False):
        _ts.PMM2D_STAG_MIN_SEG_GUARD = state
        try:
            b1 = Basis1D(P, walls, 4)
            sw[str(state)] = f"ACCEPTED (N={b1.N})"
        except ValueError as exc:
            sw[str(state)] = f"REFUSED: {str(exc)[:60]}"
    _ts.PMM2D_STAG_MIN_SEG_GUARD = True
    out["switch"] = sw
    _log(f"SWITCH PMM2D_STAG_MIN_SEG_GUARD=True  -> {sw['True'][:70]}")
    _log(f"SWITCH PMM2D_STAG_MIN_SEG_GUARD=False -> {sw['False'][:70]}")

    # PUBLIC surface: where does the refusal actually surface?
    pub = {}
    for label, frac in (("at the bar 1e-3", 1.0e-3),
                        ("under the bar 1e-4", 1.0e-4)):
        s = PMM2DStackPure(P, n_modes=3, n_orders=1, layer_grids="per-layer")
        try:
            s.add_layer(0.1e-6, eps_cell=_tile3(),
                        x_walls=[(0.5 - frac / 2) * P, (0.5 + frac / 2) * P],
                        y_walls=[0.3 * P, 0.7 * P])
            added = "add_layer ACCEPTED"
        except ValueError as exc:
            added = f"add_layer REFUSED: {str(exc)[:50]}"
        try:
            _go(s)
            solved = "solve ACCEPTED"
        except ValueError as exc:
            solved = f"solve REFUSED: {str(exc)[:50]}"
        pub[label] = dict(add_layer=added, solve=solved)
        _log(f"PUBLIC SURFACE, {label}: {added} | {solved}")
    out["public_surface"] = pub
    RES["boundary"] = out


# ================================================================= exempt
def sec_exempt():
    out = {"bar": BAR, "N_needed": int(np.ceil(1.0 / BAR))}
    out["uniform_lattice_fractions"] = {
        str(N): dict(seg_fraction=1.0 / N, under_bar=bool(1.0 / N < BAR))
        for N in (2, 3, 12, 60, 1000, 1001, 5000)}
    _log(f"integer lattice: segments = d/N, so the bar needs N > "
         f"{out['N_needed']} (N=1000 -> {1.0 / 1000:.1e}, N=1001 -> "
         f"{1.0 / 1001:.4e})")
    b = Basis1D(1.0, 2000, 3)
    out["integer_N2000_built"] = dict(N=int(b.N), uniform=bool(b.uniform),
                                      seg_fraction=1.0 / 2000)
    _log(f"Basis1D(1.0, 2000, 3): uniform={b.uniform}, segments "
         f"{1.0 / 2000:.1e} of the period -- 0.3 decades UNDER the bar, "
         f"ACCEPTED (integer path exempt)")
    try:
        Basis1D(1.0, np.linspace(0.0, 1.0, 2001), 3)
        out["array_N2000"] = "ACCEPTED"
    except ValueError as exc:
        out["array_N2000"] = f"REFUSED: {str(exc)[:80]}"
    _log(f"the SAME lattice spelled as an ARRAY -> {out['array_N2000'][:90]}")
    q = 1001 * 3
    out["N1001_M4_region_eig_dim"] = int(2 * q * q)
    _log(f"cost of reaching the bar on the integer path: N=1001, M=4 -> "
         f"q={q}, a 2 q^2 = {2 * q * q:.3e}-dimension region eig")
    RES["exempt"] = out


# ================================================================= shared
def sec_shared():
    out = {}
    for kw in ({"x_walls": [0.4 * P]}, {"y_walls": [0.4 * P]}):
        key = str(list(kw))
        try:
            s = PMM2DStackPure(P, n_modes=4, n_orders=1)
            s.add_layer(0.1e-6,
                        eps_cell=np.array([[2.25, 6.0], [6.0, 2.25]],
                                          dtype=_C), **kw)
            out[key] = "ACCEPTED (!)"
        except ValueError as exc:
            out[key] = f"REFUSED: {str(exc)[:130]}"
        _log(f"shared path with {key} -> {out[key][:110]}")
    ar = {}
    for N in (3, 5, 8, 13, 24, 100):
        bb = Basis1D(1.0, N, 4)
        ar[str(N)] = dict(uniform=bool(bb.uniform), seg=1.0 / N)
    out["shared_lattice"] = ar
    _log("shared lattice = linspace(0, d, N+1): every segment d/N, aspect "
         "ratio exactly 1.0 at N = 3/5/8/13/24/100")
    RES["shared"] = out


# =============================================================== falsepos
def sec_falsepos():
    out = {}

    def _closing(nsl, wb=0.5, wt=0.0):
        def _f():
            s = _st()
            lo_b, hi_b = 0.5 - wb / 2, 0.5 + wb / 2
            lo_t, hi_t = 0.5 - wt / 2, 0.5 + wt / 2
            if wt == 0.0:
                lo_t, hi_t = 0.5 - 1e-9, 0.5 + 1e-9
            s.add_tapered_pillar(0.2e-6, eps_pillar=6.0, eps_host=2.25,
                                 x_bounds_bottom=(lo_b * P, hi_b * P),
                                 y_bounds_bottom=(lo_b * P, hi_b * P),
                                 x_bounds_top=(lo_t * P, hi_t * P),
                                 y_bounds_top=(lo_t * P, hi_t * P),
                                 n_slices=nsl)
            _go(s)
        return _f

    rows = {}
    for nsl in (4, 8, 32, 64, 128, 200, 250, 256, 300, 512):
        r = _census(_closing(nsl))
        rows[str(nsl)] = r
        w = r["narrowest"]
        _log(f"CLOSING taper (0.5 -> 0) n_slices={nsl:4d}: narrowest "
             f"{w:.4e}  x bar {w / BAR:7.2f}  refused={r['refused']}")
    out["closing_taper"] = rows

    try:
        _closing(300)()
        out["message"] = None
    except ValueError as exc:
        out["message"] = str(exc)
    if out["message"]:
        _log("--- the message a 300-slice closing taper produces ---")
        for line in out["message"].splitlines():
            print("   " + line, flush=True)

    thin = {}
    for frac in (5e-3, 2e-3, 1.2e-3, 1e-3, 8e-4, 5e-4, 1e-4):
        def _ridge(f=frac):
            s = _st()
            s.add_layer(0.1e-6, eps_cell=_tile3(),
                        x_walls=[(0.5 - f / 2) * P, (0.5 + f / 2) * P],
                        y_walls=[0.3 * P, 0.7 * P])
            _go(s)
        r = _census(_ridge)
        thin[f"{frac:g}"] = r
        _log(f"THIN RIDGE {frac:.1e} of the period ({frac * 1e9 * 1e-3:.2f} nm "
             f"on a 1 um period): refused={r['refused']}")
    out["thin_ridge"] = thin

    out["remedy_shared_N_needed"] = {f"{f:g}": int(np.ceil(1.0 / f))
                                     for f in (2e-3, 1e-3, 5e-4, 1e-4)}
    _log("remedy (2) 'carry the fine feature on the SHARED lattice with an N "
         "that resolves it' needs N >= "
         + json.dumps(out["remedy_shared_N_needed"]))
    RES["falsepos"] = out


# =============================================================== degraded
def sec_degraded():
    from lumenairy.elements.pmm.stack import PMMStack
    per, wl, th = 1.05e-6, 0.71e-6, 0.26
    epsp, epsh, tt = 7.5, 2.0, 0.12e-6
    w0, w2, yw = (0.18, 0.61), (0.29, 0.74), (0.27, 0.73)

    def _oracle(deg):
        s = PMMStack(per, degree=deg, far_field_orders=5)
        s.add_layer(tt, segments=[(w0[0], epsh), (w0[1] - w0[0], epsp),
                                  (1 - w0[1], epsh)])
        s.add_layer(tt, segments=[(1.0, epsh)])
        s.add_layer(tt, segments=[(w2[0], epsh), (w2[1] - w2[0], epsp),
                                  (1 - w2[1], epsh)])
        s.set_source(wl, theta=th)
        return s.solve()

    o14, R14, T14 = _oracle(14)[:3]
    o12, R12, T12 = _oracle(12)[:3]
    R14, T14 = np.atleast_2d(R14), np.atleast_2d(T14)
    keep = np.abs(np.asarray(o14)) <= 1
    gap = float(max(np.max(np.abs(np.atleast_2d(R12)[:, keep] - R14[:, keep])),
                    np.max(np.abs(np.atleast_2d(T12)[:, keep] - T14[:, keep]))))
    _log(f"oracle degree-14, self-gap {gap:.3e}")

    def _err(delta, M):
        st = PMM2DStackPure(per, n_modes=M, n_orders=1,
                            layer_grids="per-layer")
        host = np.full((3, 3), _C(epsh))
        tl = np.full((3, 3), _C(epsh))
        tl[1, :] = _C(epsp)
        ywl = [yw[0] * per, yw[1] * per]
        sw = [(0.44 - delta / 2) * per, (0.44 + delta / 2) * per]
        st.add_layer(tt, eps_cell=tl, x_walls=[w0[0] * per, w0[1] * per],
                     y_walls=ywl)
        st.add_layer(tt, eps_cell=host, x_walls=sw, y_walls=ywl)
        st.add_layer(tt, eps_cell=tl, x_walls=[w2[0] * per, w2[1] * per],
                     y_walls=ywl)
        st.set_source(wl, theta=th)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            o, R, T = st.solve(jones=False)
        o = np.asarray(o)
        R, T = np.atleast_2d(R), np.atleast_2d(T)
        worst = 0.0
        for m in (-1, 0, 1):
            sel = np.where((o[:, 0] == m) & (o[:, 1] == 0))[0][0]
            j = int(np.where(np.asarray(o14) == m)[0][0])
            worst = max(worst, abs(float(R[1, sel]) - float(R14[1, j])),
                        abs(float(T[1, sel]) - float(T14[1, j])))
        return worst, float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0))), len(ws)

    out = {"oracle_selfgap": gap, "band": {}}
    for delta in (3e-1, 1e-1, 3e-2, 1e-2, 4.1e-3, 3e-3, 2e-3, 1.5e-3, 1.1e-3,
                  1.0e-3):
        rec = {}
        for M in (6, 8):
            e, c, nw = _err(delta, M)
            rec[str(M)] = dict(err=e, closure=c, n_warnings=nw)
        out["band"][f"{delta:g}"] = rec
        _log(f"ACCEPTED delta={delta:.2e}: err M=6 {rec['6']['err']:.4e} "
             f"M=8 {rec['8']['err']:.4e}  closure {rec['8']['closure']:.2e}  "
             f"warnings {rec['8']['n_warnings']}")
    RES["degraded"] = out


# ============================================================ nomortar_fp
def sec_nomortar_fp():
    """THE SHARPEST FALSE-POSITIVE CLASS.

    The fix's own attribution (S2.4) is that the mortar's ALGEBRA is exact and
    what a degenerate grid loses is a PATTERNED NEIGHBOUR's trace across a
    CROSS-GRID projection.  A per-layer stack whose layers all carry the SAME
    wall array has no cross-grid projection at all -- every interface is the
    plain square modal match, the same one the shared path uses.  ``Basis1D``
    is built before it knows that, so the contract refuses it anyway.

    Measured here: is such a stack actually accurate?  If it is, the refusal
    is a FALSE POSITIVE on a whole class, not just on a closing taper.
    """
    from lumenairy.elements.pmm import _core as _pc
    from lumenairy.elements.pmm.stack import PMMStack
    per, wl, th = 1.05e-6, 0.71e-6, 0.26
    epsp, epsh, tt = 7.5, 2.0, 0.12e-6
    w0, w2, yw = (0.18, 0.61), (0.29, 0.74), (0.27, 0.73)

    def _oracle(deg):
        s = PMMStack(per, degree=deg, far_field_orders=5)
        s.add_layer(tt, segments=[(w0[0], epsh), (w0[1] - w0[0], epsp),
                                  (1 - w0[1], epsh)])
        s.add_layer(tt, segments=[(1.0, epsh)])
        s.add_layer(tt, segments=[(w2[0], epsh), (w2[1] - w2[0], epsp),
                                  (1 - w2[1], epsh)])
        s.set_source(wl, theta=th)
        return s.solve()

    o14, R14, T14 = _oracle(14)[:3]
    R14, T14 = np.atleast_2d(R14), np.atleast_2d(T14)

    def _conforming(delta, M):
        """EVERY layer on the SAME (sliver-carrying) wall array."""
        st = PMM2DStackPure(per, n_modes=M, n_orders=1,
                            layer_grids="per-layer")
        sw = [(0.44 - delta / 2) * per, (0.44 + delta / 2) * per]
        ywl = [yw[0] * per, yw[1] * per]
        bx = [0.0] + list(sw) + [per]
        for w in (w0, None, w2):
            if w is None:
                cellx = np.full((3, 3), _C(epsh))
            else:
                cellx = np.zeros((3, 3), dtype=_C)
                for i in range(3):
                    mid = 0.5 * (bx[i] + bx[i + 1]) / per
                    cellx[i, :] = epsp if w[0] < mid < w[1] else epsh
            st.add_layer(tt, eps_cell=cellx, x_walls=sw, y_walls=ywl)
        st.set_source(wl, theta=th)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            o, R, T = st.solve(jones=False)
        o = np.asarray(o)
        R, T = np.atleast_2d(R), np.atleast_2d(T)
        worst = 0.0
        for m in (-1, 0, 1):
            sel = np.where((o[:, 0] == m) & (o[:, 1] == 0))[0][0]
            j = int(np.where(np.asarray(o14) == m)[0][0])
            worst = max(worst, abs(float(R[1, sel]) - float(R14[1, j])),
                        abs(float(T[1, sel]) - float(T14[1, j])))
        return worst, float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0))), len(ws)

    out = {"refused_by_the_contract": {}, "accuracy_with_guard_lifted": {}}
    for delta in (1e-4, 1e-6):
        try:
            _conforming(delta, 5)
            out["refused_by_the_contract"][f"{delta:g}"] = "ACCEPTED (!)"
        except ValueError as exc:
            out["refused_by_the_contract"][f"{delta:g}"] = \
                f"REFUSED: {str(exc)[:70]}"
        _log(f"CONFORMING per-layer stack, sliver {delta:.0e}: "
             f"{out['refused_by_the_contract'][f'{delta:g}'][:80]}")
    ol = _ts.PMM2D_STAG_MIN_SEG_GUARD
    orc = _pc._MORTAR_RCOND_REFUSE
    _ts.PMM2D_STAG_MIN_SEG_GUARD = False
    _pc._MORTAR_RCOND_REFUSE = 0.0
    try:
        for delta in (3e-1, 1e-2, 1e-4, 1e-6):
            rec = {}
            for M in (4, 5, 6):
                e, c, nw = _conforming(delta, M)
                rec[str(M)] = dict(err=e, closure=c, n_warnings=nw)
            out["accuracy_with_guard_lifted"][f"{delta:g}"] = rec
            _log(f"CONFORMING accuracy delta={delta:.0e}: err M=4/5/6 "
                 + " ".join(f"{rec[str(M)]['err']:.3e}" for M in (4, 5, 6))
                 + f"   closure {rec['6']['closure']:.2e}")
    finally:
        _ts.PMM2D_STAG_MIN_SEG_GUARD = ol
        _pc._MORTAR_RCOND_REFUSE = orc
    RES["nomortar_fp"] = out


SECTIONS = {"census": sec_census, "boundary": sec_boundary,
            "exempt": sec_exempt, "shared": sec_shared,
            "falsepos": sec_falsepos, "degraded": sec_degraded,
            "nomortar_fp": sec_nomortar_fp}

if __name__ == "__main__":
    for s in (sys.argv[1:] or list(SECTIONS)):
        _log(f"=== section {s} ===")
        SECTIONS[s]()
    p = os.path.join(HERE, f"v3_guard_{TAG}.json")
    old = {}
    if os.path.exists(p):
        try:
            old = json.load(open(p))
        except Exception:                                # noqa: BLE001
            old = {}
    old.update(RES)
    old["_lumenairy"] = lumenairy.__file__
    old["_bar"] = BAR
    with open(p, "w") as fh:
        json.dump(old, fh, indent=1, default=str)
    _log(f"wrote {p}")
