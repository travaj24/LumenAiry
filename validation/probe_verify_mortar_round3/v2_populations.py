"""V2 -- re-derive the generalized site's REFUSE bar two-sidedly on
INDEPENDENT populations.

Three populations, all measured at the generalized mortar site's own operand:

* **HEALTHY** -- ordinary per-layer stacks that reach the site, spanning
  ``M`` = 4..8, ``n_orders`` = 1..3, mixed in-plane/out-of-plane, both
  out-of-plane, both slanted, magnetic, three-layer, and several DEVICE
  variants (period, wavelength, angle, contrast, wall array).  Nothing here has
  a segment narrower than an ordinary geometry.
* **SLIVER** -- the same shape with an intra-layer sliver of relative width
  ``delta``, the width contract LIFTED (``PMM2D_STAG_MIN_SEG_GUARD = False``),
  swept 3e-1 down to 1e-9.  The two extra walls sit INSIDE a region of
  CONSTANT permittivity, so the DEVICE is identical at every ``delta`` and any
  movement of the answer is numerical damage.
* **BROKEN** -- synthetic operands built FROM a real operand of this site: an
  exactly singular one (a repeated column) with the site's own right-hand
  side, the same one with a right-hand side drawn from its RANGE (rank
  deficient but CONSISTENT), one with a component OUTSIDE the range
  (inconsistent), and a zero column.

Both the EXACT (Frobenius) residual and the SHIPPED probe residual are
recorded for every operand, so the estimator can be scored separately from the
bar.

Run: ``python v2_populations.py <tag>``
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _path                                     # noqa: E402,F401,I001

import json                                                 # noqa: E402
import os                                                   # noqa: E402
import time                                                 # noqa: E402
import warnings                                             # noqa: E402

import numpy as np                                          # noqa: E402
import scipy.linalg as sla                                  # noqa: E402

import _vfix as F                                           # noqa: E402
from _capture import capture, rcond_of                      # noqa: E402
from lumenairy.elements.pmm import PMM2DStackPure           # noqa: E402
from lumenairy.elements.pmm import _core as _pc             # noqa: E402
from lumenairy.elements.pmm import twod_staggered as _ts    # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
BAR = _pc._MORTAR_RESID_REFUSE


def _res(A, X, B, probe):
    return float(_pc._mortar_residual(A, X, B, probe=probe))


def _shipped_answer(A, B):
    """The answer the SITE would residuate: ``lu_factor`` + ``lu_solve``, the
    same ``getrf`` + ``getrs`` pair ``_guarded_mortar_solve`` uses.

    NOT ``np.linalg.solve``: on an EXACTLY singular operand ``gesv`` raises
    ``LinAlgError`` while ``getrf`` warns and returns a usable-looking factor
    whose answer is not finite -- and the second is what the shipped code
    path actually produces, so it is what the residual has to be measured on.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            lu, piv = sla.lu_factor(np.array(A, dtype=complex))
        except (ValueError, sla.LinAlgError, np.linalg.LinAlgError):
            return None
        return sla.lu_solve((lu, piv), B)


def _facts(A, B, label, **kw):
    X = _shipped_answer(A, B)
    if X is None:                       # the factorisation itself failed
        ex = pr = float("inf")
    else:
        ex = _res(A, X, B, False)
        pr = _res(A, X, B, True)
    s = np.linalg.svd(A, compute_uv=False)
    d = {"label": label, "n": int(A.shape[0]),
         "lu_failed": X is None,
         "residual_exact": ex, "residual_probe": pr,
         "probe_over_exact": (pr / ex if ex > 0 and np.isfinite(ex)
                              and np.isfinite(pr) else float("nan")),
         "rcond": rcond_of(A),
         "s_ratio": float(s[-1] / s[0]) if s[0] else float("nan"),
         # the SHIPPED decision, in the shipped order: the cheap probe first,
         # the exact residual only when the probe exceeds the bar
         "accepted_by_shipped_bar": bool(pr <= BAR or ex <= BAR)}
    d.update(kw)
    return d


# --------------------------------------------------------------- HEALTHY
#: DEVICE variants: each changes the physics, not just the modal count, so the
#: healthy population is not one stack sampled many times.
VARIANTS = [
    # (name, period, wl, theta, phi, eps_in, wallsA, wallsB)
    ("base", 0.87e-6, 0.73e-6, 0.17, 1.1, 8.41, (0.1873, 0.5412),
     (0.3106, 0.8039)),
    ("longwave", 1.40e-6, 1.55e-6, 0.42, 0.2, 12.25, (0.2210, 0.6640),
     (0.1150, 0.4930)),
    ("shortwave", 0.52e-6, 0.41e-6, 0.05, 2.4, 4.00, (0.3330, 0.7710),
     (0.2050, 0.5090)),
    ("normal_incidence", 0.95e-6, 0.63e-6, 0.0, 0.0, 6.25, (0.1500, 0.6250),
     (0.3750, 0.8750)),
    ("high_contrast", 0.87e-6, 0.73e-6, 0.30, 0.9, 30.0, (0.2400, 0.5100),
     (0.3900, 0.7300)),
    ("wide_angle", 1.10e-6, 0.85e-6, 0.95, 1.7, 5.76, (0.1200, 0.4400),
     (0.2900, 0.9100)),
]


def _healthy_stack(var, kind, M, n_orders):
    (_nm, per, wl, th, ph, eps_in, WA, WB) = var
    st = PMM2DStackPure(per, n_modes=M, n_orders=n_orders, n_substrate=1.45,
                        layer_grids="per-layer")
    tA = F.tensor_cell(WA, *WA)
    if kind == "spacer":
        st.add_layer(0.118e-6, eps_cell=tA, x_walls=F.sc(WA, per),
                     y_walls=F.sc(WA, per))
        st.add_layer(0.094e-6, eps=2.56)
    elif kind == "pattern":
        st.add_layer(0.118e-6, eps_cell=tA, x_walls=F.sc(WA, per),
                     y_walls=F.sc(WA, per))
        st.add_layer(0.094e-6, eps_cell=F.scalar_cell(WB, *WB, eps_in=eps_in),
                     x_walls=F.sc(WB, per), y_walls=F.sc(WB, per))
    elif kind == "magnetic":
        st.add_layer(0.118e-6, eps_cell=tA, x_walls=F.sc(WA, per),
                     y_walls=F.sc(WA, per))
        st.add_layer(0.094e-6, eps_cell=F.scalar_cell(WB, *WB, eps_in=eps_in),
                     mu_cell=F.mu_cell(WB, *WB),
                     x_walls=F.sc(WB, per), y_walls=F.sc(WB, per))
    elif kind == "oop_both":
        st.add_layer(0.118e-6, eps_cell=tA, x_walls=F.sc(WA, per),
                     y_walls=F.sc(WA, per))
        st.add_layer(0.094e-6, eps_cell=F.tensor_cell(WB, *WB,
                                                      eps_in=F.E_OOP2),
                     x_walls=F.sc(WB, per), y_walls=F.sc(WB, per))
    elif kind == "slant_both":
        st.add_layer(0.118e-6, eps_cell=F.scalar_cell(WA, *WA),
                     x_walls=F.sc(WA, per), y_walls=F.sc(WA, per),
                     slant=(0.11, 0.05))
        st.add_layer(0.094e-6, eps_cell=F.scalar_cell(WB, *WB, eps_in=eps_in),
                     x_walls=F.sc(WB, per), y_walls=F.sc(WB, per),
                     slant=(0.11, 0.05))
    elif kind == "slant_spacer":
        st.add_layer(0.118e-6, eps_cell=F.scalar_cell(WA, *WA),
                     x_walls=F.sc(WA, per), y_walls=F.sc(WA, per),
                     slant=(0.11, 0.05))
        st.add_layer(0.094e-6, eps=2.56, grid=3)
    elif kind == "deep":
        # a SIX-layer stack, three distinct grids, five interior interfaces
        WC = (0.2415, 0.6688)
        for i, (w, e) in enumerate((
                (WA, None), (WB, eps_in), (WC, 3.5), (WA, 5.0),
                (WC, 2.9), (WB, 7.1))):
            if e is None:
                st.add_layer(0.061e-6, eps_cell=tA, x_walls=F.sc(w, per),
                             y_walls=F.sc(w, per))
            else:
                st.add_layer(0.049e-6,
                             eps_cell=F.scalar_cell(w, *w, eps_in=e),
                             x_walls=F.sc(w, per), y_walls=F.sc(w, per))
    else:
        raise ValueError(kind)
    st.set_source(wl, theta=th, phi=ph)
    return st


HEALTHY_CASES = [
    ("base", "spacer", 4, 2), ("base", "spacer", 5, 2),
    ("base", "spacer", 6, 2), ("base", "spacer", 7, 2),
    ("base", "spacer", 8, 2),
    ("base", "pattern", 4, 1), ("base", "pattern", 5, 3),
    ("base", "pattern", 6, 2), ("base", "pattern", 7, 2),
    ("base", "magnetic", 4, 2), ("base", "magnetic", 6, 2),
    ("base", "oop_both", 4, 2), ("base", "oop_both", 6, 2),
    ("base", "slant_both", 4, 2), ("base", "slant_both", 6, 2),
    ("base", "slant_spacer", 5, 2),
    ("base", "deep", 4, 2), ("base", "deep", 5, 2),
    ("longwave", "pattern", 4, 2), ("longwave", "spacer", 5, 2),
    ("longwave", "oop_both", 5, 1),
    ("shortwave", "pattern", 5, 2), ("shortwave", "spacer", 4, 3),
    ("shortwave", "magnetic", 5, 2),
    ("normal_incidence", "pattern", 4, 2),
    ("normal_incidence", "spacer", 6, 2),
    ("normal_incidence", "slant_both", 4, 2),
    ("high_contrast", "pattern", 4, 2), ("high_contrast", "pattern", 6, 2),
    ("high_contrast", "spacer", 5, 2), ("high_contrast", "oop_both", 5, 2),
    ("wide_angle", "pattern", 4, 2), ("wide_angle", "pattern", 7, 2),
    ("wide_angle", "spacer", 5, 2), ("wide_angle", "magnetic", 4, 2),
]
VAR = {v[0]: v for v in VARIANTS}


def healthy(rows):
    keep = []
    for vname, kind, M, no in HEALTHY_CASES:
        t0 = time.time()
        try:
            with capture() as recs:
                st = _healthy_stack(VAR[vname], kind, M, no)
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter("always")
                    o, R, T = st.solve(jones=False)
            r00 = F.R00(o, R)
            clo = abs(float(R.sum(axis=1)[1] + T.sum(axis=1)[1]) - 1.0)
        except Exception as e:                              # noqa: BLE001
            rows.append({"pop": "healthy", "label": f"{vname}/{kind}/M{M}",
                         "error": f"{type(e).__name__}: {e}"[:300]})
            print(f"  HEALTHY {vname}/{kind}/M{M} ERROR {type(e).__name__}",
                  flush=True)
            continue
        dt = time.time() - t0
        for i, r in enumerate(recs):
            d = _facts(r["A"], r["B"], f"{vname}/{kind}/M{M}/no{no}/ifc{i}",
                       pop="healthy", variant=vname, kind=kind, M=M,
                       n_orders=no, ifc=i, prom_a=r.get("prom_a"),
                       prom_b=r.get("prom_b"), R00=r00, closure=clo,
                       seconds=round(dt, 2),
                       n_warn=len(ws))
            rows.append(d)
            keep.append(d)
        print(f"  HEALTHY {vname}/{kind}/M{M}/no{no}: {len(recs)} operand(s) "
              f"{dt:5.1f}s", flush=True)
    return keep


# ---------------------------------------------------------------- SLIVER
def _sliver_stack(delta, M=5, per=0.87e-6):
    """An OUT-OF-PLANE patterned layer whose grid carries TWO EXTRA WALLS a
    relative distance ``delta`` apart, both INSIDE the host region, plus a
    uniform spacer on its own grid (so the interface is a generalized mortar).

    The extra walls do not touch the permittivity map, so the DEVICE is the
    SAME at every ``delta`` and any movement of the answer is damage.
    """
    a, b = 0.1873, 0.5412
    c = 0.3000                      # INSIDE the pillar, so eps never changes
    walls = (a, c, c + delta, b)
    st = PMM2DStackPure(per, n_modes=M, n_orders=2, n_substrate=1.45,
                        layer_grids="per-layer")
    st.add_layer(0.118e-6, eps_cell=F.tensor_cell(walls, a, b),
                 x_walls=F.sc(walls, per), y_walls=F.sc(walls, per))
    st.add_layer(0.094e-6, eps=2.56)
    st.set_source(0.73e-6, theta=0.17, phi=1.1)
    return st


#: the widest rung is limited by the geometry: the sliver pair sits between
#: 0.30 and 0.5412, so ``delta`` must stay under 0.24 for the walls to remain
#: increasing.  1e-1 is the ORDINARY end of the ladder (its narrowest segment
#: is 0.1 of the period, well outside the warning band).
DELTAS = (1e-1, 5e-2, 3e-2, 1e-2, 3e-3, 1.05e-3, 1e-3, 3e-4, 1e-4, 1e-5,
          1e-6, 1e-7, 1e-8, 1e-9)


def sliver(rows):
    prev = _ts.PMM2D_STAG_MIN_SEG_GUARD
    refs = {}
    try:
        # the sliver stack's OUT-OF-PLANE layer carries FIVE segments, so its
        # region eig is 4 q^2 = 4 (5 (M-1))^2 -- 2500 at M = 6, minutes a
        # solve.  V2_SLIVER_MS trims the ladder for a second build.
        Ms = tuple(int(x) for x in
                   os.environ.get("V2_SLIVER_MS", "5,6").split(","))
        for delta in DELTAS:
            for M in Ms:
                _ts.PMM2D_STAG_MIN_SEG_GUARD = False
                t0 = time.time()
                try:
                    with capture() as recs:
                        st = _sliver_stack(delta, M=M)
                        with warnings.catch_warnings(record=True) as ws:
                            warnings.simplefilter("always")
                            o, R, T = st.solve(jones=False)
                    r00 = F.R00(o, R)
                    clo = abs(float(R.sum(axis=1)[1] + T.sum(axis=1)[1]) - 1.0)
                except Exception as e:                      # noqa: BLE001
                    rows.append({"pop": "sliver", "delta": delta, "M": M,
                                 "error": f"{type(e).__name__}: {e}"[:300]})
                    print(f"  SLIVER d={delta:.1e} M={M} ERROR "
                          f"{type(e).__name__}", flush=True)
                    continue
                # is the SHIPPED library (guard armed, band warning armed)
                # willing to run this geometry at all?
                _ts.PMM2D_STAG_MIN_SEG_GUARD = prev
                try:
                    with warnings.catch_warnings(record=True) as ws2:
                        warnings.simplefilter("always")
                        _sliver_stack(delta, M=M).solve(jones=False)
                    shipped = "accepted"
                    # NOT truncated: "degradation band" sits ~400 chars
                    # into the message, and a [:120] slice made every
                    # band_warn_fired read False (measured: it fires at
                    # delta = 1e-2, 3e-3 and 1.05e-3 on this fixture)
                    shipped_warn = [str(w.message) for w in ws2]
                except Exception as e:                      # noqa: BLE001
                    shipped = f"refused: {type(e).__name__}"
                    shipped_warn = []
                if delta == DELTAS[0]:
                    refs[M] = r00
                ref = refs.get(M)
                rel = (abs(r00 - ref) / abs(ref)) if ref is not None                     else float("nan")
                for i, r in enumerate(recs):
                    rows.append(_facts(
                        r["A"], r["B"], f"sliver/d{delta:.0e}/M{M}/ifc{i}",
                        pop="sliver", delta=delta, M=M, ifc=i,
                        prom_a=r.get("prom_a"), prom_b=r.get("prom_b"),
                        R00=r00, closure=clo,
                        rel_move_vs_widest=rel,
                        shipped_outcome=shipped,
                        shipped_warnings=shipped_warn,
                        band_warn_fired=any("degradation band" in w
                                            for w in shipped_warn),
                        n_warn=len(ws), seconds=round(time.time() - t0, 2)))
                print(f"  SLIVER d={delta:.1e} M={M} R00={r00:.10f} "
                      f"move={rel:.2e} {shipped}", flush=True)
    finally:
        _ts.PMM2D_STAG_MIN_SEG_GUARD = prev


# ---------------------------------------------------------------- BROKEN
def broken(rows, A, B):
    n = A.shape[0]
    rng = np.random.default_rng(20260910)

    # (1) EXACTLY singular: a repeated column, with the site's REAL rhs
    A1 = np.array(A)
    A1[:, 7] = A1[:, 3]
    rows.append(_facts(A1, B, "broken/exactly_singular_real_rhs",
                       pop="broken"))

    # (2) the SAME rank-deficient operand with a CONSISTENT rhs (in range)
    Z = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Bc = A1 @ Z
    rows.append(_facts(A1, Bc, "broken/rank_deficient_but_consistent",
                       pop="broken"))

    # (3) INCONSISTENT: a component OUTSIDE the range.  The left null vector of
    # A1 is exactly computable, so the injection is exact rather than hopeful.
    U, s, Vh = np.linalg.svd(A1)
    u = U[:, -1]
    Bi = Bc + float(np.linalg.norm(Bc)) * np.outer(
        u, rng.standard_normal(n) / np.sqrt(n))
    rows.append(_facts(A1, Bi, "broken/inconsistent_rhs", pop="broken"))

    # (4) a ZERO COLUMN -- factorises, and the answer is not finite
    A2 = np.array(A)
    A2[:, 11] = 0.0
    rows.append(_facts(A2, B, "broken/zero_column", pop="broken"))

    # (5) a zero ROW (the transpose failure: no answer can hit a nonzero rhs)
    A3 = np.array(A)
    A3[13, :] = 0.0
    rows.append(_facts(A3, B, "broken/zero_row", pop="broken"))

    # (6) a NEARLY repeated column at a ladder of separations -- the shape a
    # real degeneracy has, to see where the residual crosses the bar
    for eps_c in (1e-4, 1e-8, 1e-12, 1e-14, 1e-16):
        A4 = np.array(A)
        A4[:, 7] = A4[:, 3] * (1.0 + eps_c)
        rows.append(_facts(A4, B, f"broken/near_repeated_col_{eps_c:.0e}",
                           pop="broken_ladder", sep=eps_c))
    print("  BROKEN: 10 synthetic operands", flush=True)


def main(tag, part="all"):
    rows = []
    if part in ("all", "healthy"):
        print("HEALTHY", flush=True)
        healthy(rows)
    if part in ("all", "sliver"):
        print("SLIVER", flush=True)
        sliver(rows)
    if part in ("all", "broken"):
        print("BROKEN", flush=True)
        # build the synthetic operands FROM a real operand of this site
        with capture() as recs:
            F.build("mix_pattern", 4).solve(jones=False)
        A, B = recs[0]["A"], recs[0]["B"]
        broken(rows, A, B)

    hh = [r for r in rows if r.get("pop") == "healthy"
          and "error" not in r]
    ex = [r["residual_exact"] for r in hh]
    summary = {
        "bar": BAR,
        "n_healthy_operands": len(hh),
        "healthy_exact_min": min(ex) if ex else None,
        "healthy_exact_max": max(ex) if ex else None,
        "healthy_decades_below_bar": (float(np.log10(BAR / max(ex)))
                                      if ex else None),
        "healthy_probe_max": max(r["residual_probe"] for r in hh) if hh
        else None,
    }
    out = {"env": F.env(), "tag": tag, "part": part, "summary": summary,
           "rows": rows}
    (HERE / f"v2_populations_{tag}.json").write_text(
        json.dumps(out, indent=1), encoding="cp1252")
    print(json.dumps(summary, indent=1))
    print(f"wrote v2_populations_{tag}.json ({len(rows)} rows, "
          f"{len(hh)} healthy operands)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "win",
         sys.argv[2] if len(sys.argv) > 2 else "all")
