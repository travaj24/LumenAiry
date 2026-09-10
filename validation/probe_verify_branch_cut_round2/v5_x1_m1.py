"""TASKS 4 and 5 -- X-1 and the M1 equilibration instrument, re-measured.

X-1's fail-before is ENGINEERED (the pre-ROUND-1 branch body reinstalled in
process), because round 1 already closed it: BOTH trees this verification
mounts carry round 1, so the tree cannot be the arm here.  The engineered arm
is installed at EVERY module binding of the selector that exists on the tree,
which on the pre-round-2 tree includes the five private PMM copies.

Measured, on the ``THIN`` family (period 10 um, ridge 1.55, groove 1.5,
substrate 1.5, SUPERSTRATE 1.5 -- a permittivity coincidence on BOTH sides at
once), ``n_orders`` 6..30, both polarizations, with the library's own
``_INV_CENSUS`` armed:

  * raising cells, flagged cells, worst ``|R + T - 1|``;
  * worst relative ``sum(R)`` against the ladder's own converged value;
  * minimum equilibrated ``rcond``.

M1: the MOTIVATING population of the equilibration instrument -- calls where
the RAW residual would refuse (> 1e-8) and the equilibrated one rescues
(<= 1e-8) -- over a sweep of my own, plus a REACHABILITY check that the
instrument's code path is still live (a synthetic operand that exercises it),
so "kept" is not "dead".

Run:  PYTHONPATH=. python validation/probe_verify_branch_cut_round2/v5_x1_m1.py out.json
"""
from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _vcommon as VC  # noqa: E402

VC.pin_tree()

OUT = sys.argv[1] if len(sys.argv) > 1 else "v5.json"
#: The X-1 ``THIN`` family is the M1 file's device; this is ITS
#: wavelength (tests/unit/test_m1_conditioning_guard.py, WL = 700 nm),
#: so the ladder below is directly comparable with the round-2
#: document's table rather than being a different device.
WL = 700e-9

THIN = dict(period=10e-6, n_ridge=1.55, n_groove=1.5, n_substrate=1.5,
            n_superstrate=1.5, depth=0.5e-6, duty_cycle=0.5)
LADDER = tuple(range(6, 31))


def pre_round1_sqrt_decay(x, xp=None, band=1e-8):
    """The pre-ROUND-1 body: the EXACT ``Re(r) == 0`` pin and the ``-r`` flip.
    Transcribed from ``rcwa/_core.py`` as it stood before round 1."""
    from lumenairy.backend.array import array_namespace
    if xp is None:
        xp = array_namespace(x)
    x = xp.asarray(x).astype(complex)
    r = xp.sqrt(x)
    on_cut = r.real == 0
    return xp.where(on_cut & (r.imag < 0), -r, r)


class PreArm:
    """Install the pre-round-1 body at EVERY binding of ``_sqrt_decay`` under
    ``lumenairy.elements`` -- discovered, not listed, so the pre-round-2 tree's
    five private PMM copies are covered too."""

    def __init__(self):
        self._saved = []

    def __enter__(self):
        import lumenairy.elements as EL
        base = Path(EL.__file__).parent
        for p in sorted(base.rglob("*.py")):
            rel = p.relative_to(base).with_suffix("")
            name = "lumenairy.elements." + ".".join(rel.parts)
            if name.endswith(".__init__"):
                name = name[: -len(".__init__")]
            try:
                mod = importlib.import_module(name)
            except Exception:
                continue
            fn = getattr(mod, "_sqrt_decay", None)
            if fn is None or not callable(fn):
                continue
            self._saved.append((mod, fn))
            mod._sqrt_decay = pre_round1_sqrt_decay
        return self

    def n_bindings(self):
        return len(self._saved)

    def __exit__(self, *a):
        for mod, fn in self._saved:
            mod._sqrt_decay = fn
        self._saved = []
        return False


# ---------------------------------------------------------------------------
# X-1
# ---------------------------------------------------------------------------
def thin_solve(M, pol):
    import lumenairy.elements.rcwa._core as rc
    from lumenairy.elements.rcwa import rcwa_efficiency_1d
    prev = rc._INV_CENSUS
    rc._INV_CENSUS = []
    raised = None
    close = float("nan")
    sumR = float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T = rcwa_efficiency_1d(
                THIN["period"], THIN["n_ridge"], THIN["n_groove"],
                THIN["n_substrate"], THIN["n_superstrate"], THIN["depth"],
                THIN["duty_cycle"], WL, angle=0.0, polarization=pol,
                n_orders=M)
        R, T = np.asarray(R), np.asarray(T)
        close = float(abs(R.sum() + T.sum() - 1.0))
        sumR = float(R.sum())
    except Exception as exc:
        raised = f"{type(exc).__name__}: {exc}"
    finally:
        census = list(rc._INV_CENSUS)
        rc._INV_CENSUS = prev
    # FLAGGED uses the library's own free screen bar, the same predicate
    # tests/unit/test_m1_conditioning_guard.py uses.
    flagged = [c for c in census
               if c[2] is not None and np.isfinite(c[2])
               and c[2] < rc._INV_RCOND_SCREEN]
    refused = [c for c in census if c[4]]
    rconds = [c[2] for c in census
              if c[2] is not None and np.isfinite(c[2])]
    return {"M": M, "pol": pol, "raised": raised, "close": close,
            "sumR": sumR, "n_census": len(census),
            "n_flagged": len(flagged), "n_refused": len(refused),
            "min_rcond": (min(rconds) if rconds else None)}


def x1_ladder(arm):
    out = {}
    for pol in ("te", "tm"):
        rows = [thin_solve(M, pol) for M in LADDER]
        clean = [r["sumR"] for r in rows
                 if r["raised"] is None and r["close"] < 1e-9
                 and np.isfinite(r["sumR"])]
        ref = float(np.median(clean)) if clean else float("nan")
        worst_ratio, worst_M = 0.0, None
        for r in rows:
            if np.isfinite(r["sumR"]) and np.isfinite(ref) and ref:
                v = abs(r["sumR"] / ref - 1.0)
                if v > worst_ratio:
                    worst_ratio, worst_M = v, r["M"]
        rc = [r["min_rcond"] for r in rows if r["min_rcond"] is not None]
        out[pol] = {
            "arm": arm,
            "n_raising": sum(1 for r in rows if r["raised"]),
            "n_flagged_cells": sum(1 for r in rows if r["n_flagged"]),
            "worst_closure": max((r["close"] for r in rows
                                  if np.isfinite(r["close"])), default=None),
            "converged_sumR": ref,
            "worst_relative_sumR": worst_ratio,
            "worst_relative_sumR_at_M": worst_M,
            "min_equilibrated_rcond": (min(rc) if rc else None),
            "rows": rows,
            "pinned_cells": {str(M): next(
                (r["sumR"] for r in rows if r["M"] == M), None)
                for M in (12, 19, 20, 21)},
            "pinned_raised": {str(M): next(
                (r["raised"] for r in rows if r["M"] == M), None)
                for M in (12, 19, 20, 21)},
        }
    return out


# ---------------------------------------------------------------------------
# M1
# ---------------------------------------------------------------------------
def m1_sweep(arm):
    """Count MOTIVATING guarded inverses: raw residual would refuse
    (> _INV_RESID_REFUSE) and the equilibrated one rescues."""
    import lumenairy.elements.rcwa._core as rc
    rows = []
    saved = rc._guarded_inverse

    def tapped(A, site, *a, **kw):
        try:
            A_np = np.asarray(A)
            if A_np.ndim == 2 and A_np.shape[0] == A_np.shape[1] \
                    and np.all(np.isfinite(A_np)):
                X = np.linalg.inv(A_np)
                n = A_np.shape[0]
                raw = float(rc._inverse_residual(A_np, X))
                eq = float(rc._equilibrated_inverse_residual(A_np))
                rows.append({"site": str(site), "n": int(n), "raw": raw,
                             "eq": eq,
                             "rcond_eq": float(rc._rcond_1_equilibrated(A_np, X)),
                             "cond": float(np.linalg.cond(A_np))})
        except Exception:
            pass
        return saved(A, site, *a, **kw)

    import lumenairy.elements.pmm._core as pc
    saved_p = pc._guarded_inverse
    rc._guarded_inverse = tapped
    pc._guarded_inverse = tapped
    errs = []
    try:
        for name, fn in m1_fixtures():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fn()
            except Exception as exc:
                errs.append(f"{name}: {type(exc).__name__}: {exc}")
    finally:
        rc._guarded_inverse = saved
        pc._guarded_inverse = saved_p
    bar = float(rc._INV_RESID_REFUSE)
    motiv = [r for r in rows if r["raw"] > bar and r["eq"] <= bar]
    return {"arm": arm, "bar": bar, "n_calls": len(rows),
            "n_motivating": len(motiv),
            "max_raw": max((r["raw"] for r in rows), default=None),
            "max_eq": max((r["eq"] for r in rows), default=None),
            "max_ratio": max((r["raw"] / max(r["eq"], 1e-300) for r in rows),
                             default=None),
            "motivating": motiv[:20],
            "sites": sorted({r["site"] for r in rows}),
            "errors": errs}


def m1_fixtures():
    from lumenairy.elements.pmm import PMM2DStackHybrid, pmm_efficiency_2d_cell
    from lumenairy.elements.rcwa import RCWAStack, rcwa_efficiency_1d, rcwa_jones_2d
    F = []
    for M in LADDER[::3]:
        F.append((f"thin_{M}", lambda m=M: rcwa_efficiency_1d(
            THIN["period"], THIN["n_ridge"], THIN["n_groove"],
            THIN["n_substrate"], THIN["n_superstrate"], THIN["depth"],
            THIN["duty_cycle"], WL, angle=0.0, polarization="te",
            n_orders=m)))
    # the anisotropic coincidence cell and its lossy sibling
    tens = np.zeros((6, 6, 3, 3), dtype=complex)
    base = np.full((6, 6), 2.25, dtype=complex)
    base[1:3, 2:4] = 2.25 * (1 + 1e-6)
    for i in range(3):
        tens[..., i, i] = base
    F.append(("aniso_coinc", lambda t=tens: rcwa_jones_2d(
        0.62e-6, 0.58e-6, t, 1.5, 1.5, 0.23e-6, WL, n_orders_x=3,
        n_orders_y=3)))
    F.append(("aniso_coinc_lossy", lambda t=tens + 1e-3j: rcwa_jones_2d(
        0.62e-6, 0.58e-6, t, 1.5, 1.5, 0.23e-6, WL, n_orders_x=3,
        n_orders_y=3)))
    # the uniform-spacer RCWA stack
    for M in (3, 4, 5):
        def rst(m=M):
            c = np.full((8, 8), 2.25, dtype=complex)
            c[1:4, 2:4] = 2.25 * (1 + 1e-6)
            st = RCWAStack(0.62e-6, period_y=0.58e-6, n_substrate=1.63,
                           n_orders=m)
            st.add_layer(0.12e-6, eps=2.25)
            st.add_layer(0.23e-6, eps_cell=np.kron(c, np.ones((4, 4))))
            st.add_layer(0.09e-6, eps=2.25)
            st.set_source(WL)
            return st.solve().efficiencies()
        F.append((f"rcwa_spacer_{M}", rst))
    # the hybrid PMM uniform-spacer stack
    for M in (3, 4):
        def hst(m=M):
            c = np.full((8, 8), 2.25, dtype=complex)
            c[1:4, 2:4] = 2.25 * (1 + 1e-6)
            st = PMM2DStackHybrid(0.62e-6, 0.58e-6, n_substrate=1.63,
                                  degree=7, n_orders=m, symmetry=False)
            st.add_layer(0.12e-6, eps=2.25)
            st.add_layer(0.23e-6, eps_cell=c)
            st.add_layer(0.09e-6, eps=2.25)
            st.set_source(WL)
            return st.solve()
        F.append((f"hyb_spacer_{M}", hst))
    # a loss ladder, a metal, an ordinary high-contrast cell
    for im in (1e-2, 1e-6, 1e-10):
        F.append((f"loss_{im}", lambda v=im: rcwa_efficiency_1d(
            1.0e-6, np.sqrt(4.41 + 1j * v), 1.0, 1.5, 1.0, 0.45e-6, 0.5, WL,
            n_orders=11)))
    F.append(("metal", lambda: rcwa_efficiency_1d(
        1.0e-6, np.sqrt(-40 + 2.5j), 1.0, 1.5, 1.0, 0.12e-6, 0.5, WL,
        n_orders=11)))
    F.append(("pmm_cell", lambda: pmm_efficiency_2d_cell(
        0.62e-6, 0.58e-6, base, 1.5, 1.0, 0.23e-6, WL, n_orders=4, degree=7,
        symmetry=False)))
    return F


def m1_reachability():
    """Is the equilibration instrument's code path still LIVE?  Construct a
    matrix whose RAW inverse residual is large purely from row/column SCALING
    while the equilibrated one is small -- the shape the instrument exists to
    rescue -- and confirm the shipped screen scores it that way."""
    import lumenairy.elements.rcwa._core as rc
    rng = np.random.default_rng(4242)
    n = 24
    Q = np.linalg.qr(rng.normal(size=(n, n))
                     + 1j * rng.normal(size=(n, n)))[0]
    out = {}
    for name, spread in (("scaled_1e10", 1e10), ("scaled_1e14", 1e14),
                         ("scaled_1e16", 1e16)):
        d = np.logspace(0, np.log10(spread), n)
        A = (Q * d) @ Q.conj().T * 1.0
        A = np.diag(d) @ Q @ np.diag(1.0 / d[::-1])
        X = np.linalg.inv(A)
        raw = float(rc._inverse_residual(A, X))
        eq = float(rc._equilibrated_inverse_residual(A))
        rcond_eq = float(rc._rcond_1_equilibrated(A, X))
        rcond_raw = float(1.0 / np.linalg.cond(A, 1))
        out[name] = {"raw_residual": raw, "equilibrated_residual": eq,
                     "ratio": raw / max(eq, 1e-300),
                     "rcond_equilibrated": rcond_eq, "rcond_raw": rcond_raw,
                     "raw_would_refuse": raw > float(rc._INV_RESID_REFUSE),
                     "equilibrated_rescues": eq <= float(rc._INV_RESID_REFUSE)}
    # and that _guarded_inverse actually routes such an operand through the
    # equilibrated instrument (the census records the equilibrated numbers)
    prev = rc._INV_CENSUS
    rc._INV_CENSUS = []
    try:
        d = np.logspace(0, 14, n)
        A = np.diag(d) @ Q @ np.diag(1.0 / d[::-1])
        rc._guarded_inverse(A, "verification synthetic operand")
        rows = list(rc._INV_CENSUS)
    finally:
        rc._INV_CENSUS = prev
    # ... and one whose EQUILIBRATED rcond is BELOW the free screen, so the
    # equilibrated-RESIDUAL branch (the deeper half of the instrument) runs.
    prev2 = rc._INV_CENSUS
    rc._INV_CENSUS = []
    try:
        sv = np.ones(n)
        sv[-1] = 1e-13                    # genuinely near-singular
        B = Q @ np.diag(sv) @ Q.conj().T
        d2 = np.logspace(0, 6, n)
        Bs = np.diag(d2) @ B @ np.diag(1.0 / d2)
        rc._guarded_inverse(Bs, "verification synthetic near-singular operand")
        rows2 = list(rc._INV_CENSUS)
    finally:
        rc._INV_CENSUS = prev2
    out["census_rows_from_near_singular_operand"] = [
        [str(r[0]), int(r[1]),
         (None if r[2] is None else float(r[2])),
         (None if r[3] is None else float(r[3])), bool(r[4])]
        for r in rows2]
    out["screen_bar"] = float(rc._INV_RCOND_SCREEN)
    out["census_rows_from_synthetic_operand"] = [
        [str(r[0]), int(r[1]),
         (None if r[2] is None else float(r[2])),
         (None if r[3] is None else float(r[3])), bool(r[4])]
        for r in rows]
    return out


def run():
    payload = {}
    payload["x1_post"] = x1_ladder("shipped")
    with PreArm() as arm:
        payload["n_pre_arm_bindings"] = arm.n_bindings()
        payload["x1_pre"] = x1_ladder("pre-round-1 engineered")
    payload["m1_post"] = m1_sweep("shipped")
    with PreArm():
        payload["m1_pre"] = m1_sweep("pre-round-1 engineered")
    payload["m1_reachability"] = m1_reachability()
    VC.dump(OUT, payload)

    st = VC.stamp()
    print(f"ARM(tree) = {st['arm']}   pre-arm bindings patched: "
          f"{payload['n_pre_arm_bindings']}")
    print(f"\n{'ladder/arm':22s} {'raises':>7s} {'flagged':>8s} "
          f"{'worst|R+T-1|':>13s} {'worst rel sumR':>15s} {'min rcond':>11s}")
    for arm, key in (("PRE", "x1_pre"), ("POST", "x1_post")):
        for pol in ("te", "tm"):
            r = payload[key][pol]
            print(f"{pol.upper() + ' ' + arm:22s} {r['n_raising']:7d} "
                  f"{r['n_flagged_cells']:8d} "
                  f"{(r['worst_closure'] or float('nan')):13.4e} "
                  f"{r['worst_relative_sumR']:15.4f} "
                  f"(M={r['worst_relative_sumR_at_M']}) "
                  f"{(r['min_equilibrated_rcond'] or float('nan')):11.3e}")
    print("\npinned TE cells sum(R):")
    for M in ("12", "19", "20", "21"):
        print(f"   M={M:>2s}  PRE {payload['x1_pre']['te']['pinned_cells'][M]}"
              f"  raised={payload['x1_pre']['te']['pinned_raised'][M]}"
              f"   POST {payload['x1_post']['te']['pinned_cells'][M]}")
    print("\nM1 instrument:")
    for arm in ("m1_pre", "m1_post"):
        r = payload[arm]
        print(f"   {arm:8s} calls {r['n_calls']:4d}  motivating "
              f"{r['n_motivating']:3d}  max raw {r['max_raw']:.3e}  max eq "
              f"{r['max_eq']:.3e}  max ratio {r['max_ratio']:.3e}")
        if r["errors"]:
            print(f"      errors: {r['errors'][:3]}")
    print("\nM1 reachability (synthetic scaled operands):")
    for k, v in payload["m1_reachability"].items():
        print(f"   {k}: {v}")


if __name__ == "__main__":
    run()
