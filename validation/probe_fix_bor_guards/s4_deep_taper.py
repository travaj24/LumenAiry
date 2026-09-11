"""STEP 4, the BINDING ARM of the false-positive census: the taper staircase
walked out to 256 slices, at several ``(degree, k0)``.

WHY THIS IS ITS OWN PROBE.  A cone sliced into ``N`` layers, each carrying its
own ring radius, makes ADJACENT slices' walls differ by
``(r_top - r_bot) / N`` -- so the MANUFACTURED cell halves with every doubling
of the slice count while ``|q|max / (n_max k0)`` doubles.  It is therefore the
ordinary geometry that BINDS both bars, and a margin measured at 64 slices (the
scoping's deepest) says nothing about 256.  Running it separately keeps the arm
cheap enough to re-run on both builds and under both ladders.

The cost is one SEM modal eigensolve per slice, so the deep arms are swept at a
lower ``degree`` and ``k0`` as well as at the census settings.  ``degree`` and
``k0`` do NOT move the geometric conjunct at all -- ``w/Rbig`` is
``(r_top - r_bot) / (N Rbig)`` exactly -- and they move the spectral one in the
direction that matters: a LOWER ``k0`` RAISES ``|q|max / (n_max k0)``, so the
low-``k0`` arms are the demanding ones for the screen, not the lenient ones.

Usage: ``python s4_deep_taper.py <tag>``.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

RBIG = 24.0
R_TOP, R_BOT, H = 8.0, 2.0, 1.2


def taper(n_slices, degree, k0):
    from lumenairy.elements.bor import BORStack
    st = BORStack(RBIG, 1, basis="sem", degree=degree, N=160,
                  n_superstrate=1.0, n_substrate=1.5)
    st.set_source(k0=k0)
    for i in range(n_slices):
        r = R_TOP + (R_BOT - R_TOP) * (i + 0.5) / n_slices
        st.add_layer(H / n_slices, segments=[(r, 6.0), (RBIG, 2.0)])
    return st


def run(n_slices, degree, k0):
    from lumenairy.elements.bor import _sem_contract as SC
    prev = SC.BOR_SEM_MESH_GUARD
    SC.BOR_SEM_MESH_GUARD = False
    t = time.time()
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            st = taper(n_slices, degree, k0)
            res = st.solve()
        recs = st._sem_mesh_report
        e = np.asarray(res["energy"])
        return dict(
            n_slices=n_slices, degree=degree, k0=k0,
            wall_delta=float((R_TOP - R_BOT) / n_slices),
            w_min_frac=min(r["w_min_frac"] for r in recs),
            w_min_union_frac=min(r["w_min_union_frac"] for r in recs),
            q_excess_max=max(r["q_excess"] for r in recs
                             if np.isfinite(r["q_excess"])),
            verdicts=sorted({SC.verdict(r) for r in recs}),
            n_orders=int(np.size(res["R"])),
            closure=float(np.max(np.abs(e - 1.0))) if e.size else None,
            n_warn=len(w), secs=round(time.time() - t, 1))
    except Exception as exc:                           # noqa: BLE001
        return dict(n_slices=n_slices, degree=degree, k0=k0,
                    error="%s: %s" % (type(exc).__name__, exc),
                    secs=round(time.time() - t, 1))
    finally:
        SC.BOR_SEM_MESH_GUARD = prev


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    print("TREE", C.pin_tree())
    print("KERNEL", C.kernel_tag())
    from lumenairy.elements.bor import _sem_contract as SC
    rows = []
    arms = [(128, 6, 0.8), (256, 6, 0.8), (128, 8, 0.8), (256, 8, 0.8),
            (256, 6, 2.0)]
    if "--full" in sys.argv:
        arms += [(128, 8, 2.0), (256, 8, 2.0), (256, 12, 2.0),
                 (512, 6, 0.8)]
    for ns, deg, k0 in arms:
        r = run(ns, deg, k0)
        rows.append(r)
        if "error" in r:
            print("  %4d slices deg %2d k0 %.1f: ERROR %s (%.1f s)"
                  % (ns, deg, k0, r["error"], r["secs"]))
        else:
            print("  %4d slices deg %2d k0 %.1f: w_union=%.4e q_exc=%8.4g "
                  "closure=%.3e -> %s   (%.1f s)"
                  % (ns, deg, k0, r["w_min_union_frac"], r["q_excess_max"],
                     r["closure"] if r["closure"] is not None else float("nan"),
                     r["verdicts"], r["secs"]))
    ok = [r for r in rows if "error" not in r]
    summary = dict(
        n_arms=len(ok),
        narrowest_union_frac=min(r["w_min_union_frac"] for r in ok),
        worst_q_excess=max(r["q_excess_max"] for r in ok),
        all_ok=all(r["verdicts"] == ["ok"] for r in ok),
        warn_edge=SC._BOR_SLIVER_BAND_FRAC, q_bar=SC._BOR_Q_EXCESS)
    summary["width_margin_x"] = (summary["narrowest_union_frac"]
                                 / summary["warn_edge"])
    summary["q_margin_x"] = summary["q_bar"] / summary["worst_q_excess"]
    print("SUMMARY", summary)
    C.dump("s4_deep_taper_%s.json" % (tag,), dict(rows=rows, summary=summary))


if __name__ == "__main__":
    main()
