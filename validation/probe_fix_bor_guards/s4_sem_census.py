"""STEP 4 instrument: the FALSE-POSITIVE census, the delta ladders, and the
within-layer liner ladder for the SEM manufactured-element contract.

THE CENSUS IS MEASURED FIRST AND THE WARN EDGE IS FIXED BY IT.  That order is
the round-4 correction the 2-D Cartesian contract needed: a census margin
measured on four geometry families is a SAMPLE property, not a library one
("the census margin is sample-scoped, and it is 1.67x, not 3.6x").  So this
probe walks EVERY BOR geometry family the shipped test suite builds --
uniform, uniform-anisotropic, multi-segment, ring gratings at several periods
and duties, nm-unit fixtures, graded and hp-refined meshes, coincident and
detuned neighbours -- plus a taper staircase at 4 / 8 / 16 / 32 / 64 / 128 /
256 slices, and reports the narrowest ORDINARY element per family and the
verdict the shipped contract reaches on it.

It reads the SAME measurement function the solver runs
(``lumenairy.elements.bor._sem_contract.measure_layer`` / ``verdict``), so the
census is a statement about the shipped behaviour and not about a parallel
re-implementation of it.

Usage: ``python s4_sem_census.py <tag> [--fast]``.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402


def _report(st):
    """Solve with the guard DISARMED and return the per-layer measurement."""
    from lumenairy.elements.bor import _sem_contract as SC
    prev = SC.BOR_SEM_MESH_GUARD
    SC.BOR_SEM_MESH_GUARD = False
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = st.solve()
        recs = list(getattr(st, "_sem_mesh_report", []) or [])
        for r in recs:
            r["verdict"] = SC.verdict(r)
        e = np.asarray(res["energy"])
        return recs, dict(
            n_orders=int(np.size(res["R"])),
            closure=float(np.max(np.abs(e - 1.0))) if e.size else None,
            R0=float(np.asarray(res["R"])[0]) if np.size(res["R"]) else None,
            n_warn=len(w))
    finally:
        SC.BOR_SEM_MESH_GUARD = prev


def _stack(Rbig, m, k0, degree=8, eps=1.0, eps_sub=1.5, N=160,
           elements_per_segment=1, grade=False):
    from lumenairy.elements.bor import BORStack
    st = BORStack(Rbig, m, basis="sem", degree=degree, N=N,
                  n_superstrate=eps, n_substrate=eps_sub,
                  elements_per_segment=elements_per_segment, grade=grade)
    st.set_source(k0=k0)
    return st


# --------------------------------------------------------------------------- #
#  1. the ORDINARY-GEOMETRY false-positive census                              #
# --------------------------------------------------------------------------- #
def ordinary_families(fast=False):
    rows = []

    def add(label, st, **meta):
        try:
            recs, obs = _report(st)
        except Exception as exc:                       # noqa: BLE001
            rows.append(dict(label=label, error="%s: %s"
                             % (type(exc).__name__, exc), **meta))
            return
        if not recs:
            rows.append(dict(label=label, error="no layers", **meta))
            return
        worst = min(recs, key=lambda r: r["w_min_frac"])
        hottest = max(recs, key=lambda r: (r["q_excess"]
                                           if np.isfinite(r["q_excess"])
                                           else -1.0))
        rows.append(dict(
            label=label, n_layers=len(recs),
            w_min_frac=worst["w_min_frac"],
            w_min_union_frac=min(r["w_min_union_frac"] for r in recs),
            q_excess_max=hottest["q_excess"],
            verdicts=sorted({r["verdict"] for r in recs}),
            **obs, **meta))

    degrees = (8,) if fast else (6, 8, 12, 16)
    for deg in degrees:
        # --- uniform, the simplest ordinary layer ---------------------------
        st = _stack(24.0, 1, 2.0, degree=deg)
        st.add_layer(0.5, eps=2.25)
        add("uniform", st, degree=deg)
        # --- uniform ANISOTROPIC (the eps_tensor family) --------------------
        st = _stack(24.0, 1, 2.0, degree=deg)
        st.add_layer(0.5, eps_tensor=(2.25, 2.25, 3.24))
        add("uniform_uniaxial", st, degree=deg)
        # --- multi-segment ---------------------------------------------------
        st = _stack(24.0, 1, 2.0, degree=deg)
        st.add_layer(0.5, segments=[(6.0, 6.0), (12.0, 2.25), (24.0, 2.0)])
        add("segments_3", st, degree=deg)
        # --- two ring layers, COINCIDENT walls (delta exactly zero) ---------
        st = _stack(24.0, 1, 2.0, degree=deg)
        st.add_layer(0.5, segments=[(6.0, 4.0), (24.0, 2.25)])
        st.add_layer(0.4, segments=[(6.0, 2.25), (24.0, 4.0)])
        add("coincident_walls", st, degree=deg)
        # --- ring GRATINGS at several periods and duties --------------------
        for period, duty in ((3.0, 0.5), (1.5, 0.3), (0.8, 0.5)):
            st = _stack(24.0, 1, 2.0, degree=deg)
            st.add_layer(0.5, rings=(period, duty, 2.449, 1.414))
            add("ring_grating_p%g_d%g" % (period, duty), st, degree=deg)
        # --- hp refinement and grading --------------------------------------
        st = _stack(24.0, 1, 2.0, degree=deg, elements_per_segment=3)
        st.add_layer(0.5, segments=[(6.0, 6.0), (24.0, 2.0)])
        add("eps_per_seg_3", st, degree=deg)
        st = _stack(24.0, 1, 2.0, degree=deg, elements_per_segment=3,
                    grade=True)
        st.add_layer(0.5, segments=[(6.0, 6.0), (24.0, 2.0)])
        add("eps_per_seg_3_graded", st, degree=deg)
        # --- a three-layer stack with a wall-free spacer between rings -------
        st = _stack(24.0, 1, 2.0, degree=deg)
        st.add_layer(0.4, segments=[(6.0, 4.0), (24.0, 2.25)])
        st.add_layer(0.3, eps=2.25)
        st.add_layer(0.4, segments=[(9.0, 2.25), (24.0, 4.0)])
        add("spacer_between_rings", st, degree=deg)
        # --- higher m, and small / large k0 ---------------------------------
        for m in (0, 2, 5):
            st = _stack(24.0, m, 2.0, degree=deg)
            st.add_layer(0.5, segments=[(6.0, 4.0), (24.0, 2.25)])
            add("ring_m%d" % m, st, degree=deg)
        for k0 in (0.8, 3.5, 8.0):
            st = _stack(24.0, 1, k0, degree=deg)
            st.add_layer(0.5, segments=[(6.0, 4.0), (24.0, 2.25)])
            add("ring_k%g" % k0, st, degree=deg)
        # --- nm-unit fixture (Rbig and lambda 833x / 338x apart) -------------
        sc = 1e-6                       # Rbig = 20 um, wavelength = 1550 nm
        st = _stack(20.0 * sc, 1, 2.0 * np.pi / (1.55 * sc), degree=deg)
        st.add_layer(0.5 * sc, rings=(3.0 * sc, 0.5, 2.45, 1.41))
        add("nm_units_ring", st, degree=deg)
        # --- lossy metal ring ------------------------------------------------
        st = _stack(24.0, 1, 2.0, degree=deg)
        st.add_layer(0.5, segments=[(6.0, -20.0 + 2.0j), (24.0, 2.25)])
        add("lossy_metal_ring", st, degree=deg)
    return rows


def taper_staircase(fast=False):
    """A cone sliced into N layers, each carrying its own ring radius, so
    ADJACENT slices' walls differ by ``(r_top - r_bot) / n_slices``.

    This is the family that WALKS into the contract: the wall delta halves with
    every doubling of the slice count while ``|q|max / ceiling`` doubles, so it
    is the binding ordinary geometry and the warn edge must be set by it.
    """
    rows = []
    slices = (4, 8, 16, 32) if fast else (4, 8, 16, 32, 64, 128, 256)
    for ns in slices:
        for deg in ((8,) if fast else (8, 12)):
            st = _stack(24.0, 1, 2.0, degree=deg)
            r_top, r_bot, H = 8.0, 2.0, 1.2
            for i in range(ns):
                r = r_top + (r_bot - r_top) * (i + 0.5) / ns
                st.add_layer(H / ns, segments=[(r, 6.0), (24.0, 2.0)])
            try:
                recs, obs = _report(st)
            except Exception as exc:                   # noqa: BLE001
                rows.append(dict(label="taper", n_slices=ns, degree=deg,
                                 error="%s: %s" % (type(exc).__name__, exc)))
                continue
            from lumenairy.elements.bor import _sem_contract as SC
            rows.append(dict(
                label="taper", n_slices=ns, degree=deg, n_layers=len(recs),
                wall_delta=float((r_top - r_bot) / ns),
                w_min_frac=min(r["w_min_frac"] for r in recs),
                w_min_union_frac=min(r["w_min_union_frac"] for r in recs),
                q_excess_max=max(r["q_excess"] for r in recs
                                 if np.isfinite(r["q_excess"])),
                verdicts=sorted({SC.verdict(r) for r in recs}),
                **obs))
    return rows


# --------------------------------------------------------------------------- #
#  2. the DELTA ladder (two adjacent ring layers, walls delta apart)           #
# --------------------------------------------------------------------------- #
def delta_ladder(fast=False, separated=False):
    """``separated=False``: the two walls in ADJACENT layers, so the ``+-1``
    window unions both.  ``separated=True``: the same two walls THREE layers
    apart with TWO wall-free spacers between them, so the window never spans
    both -- the attribution control.  ONE spacer is not enough: a wall-free
    layer BETWEEN two ring layers inherits both wall sets.
    """
    rows = []
    Rbig = 24.0
    degs = (8,) if fast else (6, 8, 12)
    ladder = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    for deg in degs:
        for dl in ladder:
            d = dl * Rbig
            st = _stack(Rbig, 1, 2.0, degree=deg, N=200)
            st.add_layer(0.5, segments=[(6.0, 6.0), (Rbig, 2.0)])
            if separated:
                st.add_layer(0.3, eps=2.0)
                st.add_layer(0.3, eps=2.0)
            st.add_layer(0.5, segments=[(6.0 + d, 2.0), (Rbig, 6.0)])
            try:
                recs, obs = _report(st)
            except Exception as exc:                   # noqa: BLE001
                rows.append(dict(delta_frac=dl, degree=deg, separated=separated,
                                 error="%s: %s" % (type(exc).__name__, exc)))
                continue
            from lumenairy.elements.bor import _sem_contract as SC
            rows.append(dict(
                delta_frac=dl, delta=float(d), degree=deg,
                separated=bool(separated), n_layers=len(recs),
                w_min_frac=min(r["w_min_frac"] for r in recs),
                w_min_union_frac=min(r["w_min_union_frac"] for r in recs),
                q_excess_max=max(r["q_excess"] for r in recs
                                 if np.isfinite(r["q_excess"])),
                verdicts=sorted({SC.verdict(r) for r in recs}),
                **obs))
    return rows


# --------------------------------------------------------------------------- #
#  3. the WITHIN-LAYER liner ladder                                            #
# --------------------------------------------------------------------------- #
def within_layer_ladder(fast=False):
    """An annulus of width ``w`` inside ONE layer's OWN segment list, with the
    liner's permittivity set EQUAL to the material it replaces -- so the device
    is independent of ``w`` by construction and every deviation is numerical
    damage with no physics to subtract.

    The narrow cell here is the CALLER's, not the library's, so the contract
    never refuses it; the question this ladder answers is whether the
    spurious-wavenumber screen still SEES it.
    """
    rows = []
    Rbig = 24.0
    degs = (8,) if fast else (6, 8, 12, 16)
    for deg in degs:
        for wf in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
            w = wf * Rbig
            st = _stack(Rbig, 1, 2.0, degree=deg, N=200)
            st.add_layer(0.5, segments=[(6.0, 6.0), (6.0 + w, 2.0),
                                        (Rbig, 2.0)])
            try:
                recs, obs = _report(st)
            except Exception as exc:                   # noqa: BLE001
                rows.append(dict(w_frac=wf, degree=deg,
                                 error="%s: %s" % (type(exc).__name__, exc)))
                continue
            from lumenairy.elements.bor import _sem_contract as SC
            rows.append(dict(
                w_frac=wf, w=float(w), degree=deg, n_layers=len(recs),
                w_min_frac=min(r["w_min_frac"] for r in recs),
                w_min_union_frac=min(r["w_min_union_frac"] for r in recs),
                q_excess_max=max(r["q_excess"] for r in recs
                                 if np.isfinite(r["q_excess"])),
                verdicts=sorted({SC.verdict(r) for r in recs}),
                **obs))
    return rows


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    fast = "--fast" in sys.argv
    print("TREE", C.pin_tree())
    print("KERNEL", C.kernel_tag())
    payload = dict(
        ordinary=ordinary_families(fast), taper=taper_staircase(fast),
        adjacent=delta_ladder(fast, separated=False),
        separated=delta_ladder(fast, separated=True),
        within_layer=within_layer_ladder(fast))

    ok = [r for r in payload["ordinary"] + payload["taper"]
          if "w_min_frac" in r]
    payload["summary"] = dict(
        n_ordinary=len(ok),
        ordinary_narrowest_w_min_frac=min(r["w_min_frac"] for r in ok),
        ordinary_narrowest_union_frac=min(r["w_min_union_frac"] for r in ok
                                          if np.isfinite(
                                              r["w_min_union_frac"])),
        ordinary_worst_q_excess=max(r["q_excess_max"] for r in ok),
        ordinary_non_ok=[dict(label=r.get("label"),
                              n_slices=r.get("n_slices"),
                              degree=r.get("degree"),
                              verdicts=r["verdicts"],
                              w_min_union_frac=r["w_min_union_frac"],
                              q_excess=r["q_excess_max"])
                         for r in ok if r["verdicts"] != ["ok"]],
        errors=[r for r in payload["ordinary"] + payload["taper"]
                if "error" in r],
    )
    print("CENSUS", payload["summary"]["n_ordinary"], "families;",
          "narrowest element %.4e of Rbig; narrowest UNION cell %.4e; "
          "worst q_excess %.4g"
          % (payload["summary"]["ordinary_narrowest_w_min_frac"],
             payload["summary"]["ordinary_narrowest_union_frac"],
             payload["summary"]["ordinary_worst_q_excess"]))
    print("NON-OK ORDINARY:", payload["summary"]["ordinary_non_ok"])
    print("ERRORS:", len(payload["summary"]["errors"]))
    for r in payload["taper"]:
        if "w_min_frac" in r:
            print("  taper %4d slices deg %2d: delta=%.4e w_min=%.4e "
                  "union=%.4e q_exc=%.4g -> %s"
                  % (r["n_slices"], r["degree"], r["wall_delta"],
                     r["w_min_frac"], r["w_min_union_frac"],
                     r["q_excess_max"], r["verdicts"]))
    for key in ("adjacent", "separated", "within_layer"):
        print("--", key)
        for r in payload[key]:
            if "w_min_frac" not in r:
                continue
            k = r.get("delta_frac", r.get("w_frac"))
            print("   %.0e deg %2d: w_min=%.4e union=%.4e q_exc=%.5g "
                  "closure=%.3e -> %s"
                  % (k, r["degree"], r["w_min_frac"], r["w_min_union_frac"],
                     r["q_excess_max"],
                     r["closure"] if r["closure"] is not None else float("nan"),
                     r["verdicts"]))
    C.dump("s4_census_%s.json" % (tag,), payload)


if __name__ == "__main__":
    main()
