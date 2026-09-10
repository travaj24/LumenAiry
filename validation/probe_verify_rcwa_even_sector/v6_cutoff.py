"""V6 -- the LAYER-CUTOFF corner: does the fix change what the library REFUSES?

Why this corner exists.  For a lossless propagating mode ``lam^2 = -s + i eta``
the principal root's real part is ``eta / (2 sqrt(s))``, so the band ratio the
fix thresholds GROWS as a layer mode approaches cutoff (``s -> 0``) at fixed
backward error.  ``v4_hunt.py`` walks a 1-D mount onto such a cutoff and finds
the ratio climbing to within a factor of 1.5 of ``_CUT_BAND_REL``; it also finds
the lossless closure degrading in BOTH arms there.  This probe asks the question
that matters for shipping: at those mounts, is the library still LOUD?

For each rung it records, on the POST arm and on the transcribed PRE arm in the
same process:
  * whether the call RAISED (``_EnergyError``: the library refuses), WARNED
    (``_EnergyWarning`` / ``UserWarning``: the library returns but says so), or
    returned SILENTLY;
  * the lossless closure defect ``sum R + T - 1``.

A rung that RAISES pre-fix and returns SILENTLY post-fix would be a severity-
raising regression; a rung that is loud on both arms is a pre-existing accuracy
limit that this change neither creates nor cures.

Usage: python v6_cutoff.py <out.json>
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v_fixtures as V  # noqa: E402


def _call(period, pol):
    from lumenairy.elements.rcwa import rcwa_efficiency_1d
    return rcwa_efficiency_1d(period, 2.1, 1.5, 1.5, 1.0, V._DEPTH, 0.5,
                              V._WL, polarization=pol, n_orders=11)


def observe(period, pol):
    """RAISED / WARNED / SILENT, plus the closure when there is one."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            res = _call(period, pol)
            clo = V.closure_defect_eff(res)
            err = None
        except Exception as exc:
            clo = None
            err = type(exc).__name__
    cats = sorted({w.category.__name__ for w in caught})
    if err:
        verdict = "RAISED"
    elif cats:
        verdict = "WARNED"
    else:
        verdict = "SILENT"
    return dict(verdict=verdict, error=err, warnings=cats, closure=clo,
                first_warning=(str(caught[0].message)[:160] if caught
                               else None))


def min_abs_lam2(period, pol):
    eig = V.EigSpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with eig:
            try:
                _call(period, pol)
            except Exception:
                pass
    if not eig.seen:
        return float("inf")
    return min(float(np.min(np.abs(w))) for w in eig.seen)


def ladder(pol):
    """Trisect the period toward a layer cutoff, keeping one period per decade
    of ``min |lam^2|`` -- the state is ENGINEERED, not sampled."""
    grid = np.linspace(0.20e-6, 1.60e-6, 401)
    vals = [min_abs_lam2(float(p), pol) for p in grid]
    k = int(np.argmin(vals))
    a0, b0 = float(grid[max(k - 1, 0)]), float(grid[min(k + 1, 400)])
    rows, seen = [], set()
    for _ in range(120):
        m1 = a0 + (b0 - a0) / 3.0
        m2 = b0 - (b0 - a0) / 3.0
        v1, v2 = min_abs_lam2(m1, pol), min_abs_lam2(m2, pol)
        p_here, v_here = (m1, v1) if v1 < v2 else (m2, v2)
        dec = (int(np.floor(np.log10(v_here)))
               if np.isfinite(v_here) and v_here > 0 else -400)
        if dec not in seen:
            seen.add(dec)
            post = observe(p_here, pol)
            with V.PreSqrtDecay():
                pre = observe(p_here, pol)
            rows.append(dict(pol=pol, period=p_here, min_abs_lam2=v_here,
                             post=post, pre=pre))
        if v1 < v2:
            b0 = m2
        else:
            a0 = m1
        if b0 - a0 < 1e-22:
            break
    rows.sort(key=lambda r: r["min_abs_lam2"], reverse=True)
    return rows


def main():
    V.require_local_tree()
    out = sys.argv[1]
    rows = []
    for pol in ("te", "tm"):
        rows.extend(ladder(pol))
    sev = []
    for r in rows:
        pre, post = r["pre"], r["post"]
        note = ""
        if pre["verdict"] == "RAISED" and post["verdict"] == "SILENT":
            note = "  <== SEVERITY DROP: refused pre-fix, silent post-fix"
            sev.append(r)
        elif pre["verdict"] != "SILENT" and post["verdict"] == "SILENT":
            note = "  <== severity drop"
            sev.append(r)
        print("%-3s |lam2|min=%-10.3e pre=%-7s(%s) post=%-7s(%s)%s" % (
            r["pol"], r["min_abs_lam2"], pre["verdict"],
            ("%+.2e" % pre["closure"]) if pre["closure"] is not None else "-",
            post["verdict"],
            ("%+.2e" % post["closure"]) if post["closure"] is not None else "-",
            note))
    worst = max((r for r in rows if r["post"]["closure"] is not None),
                key=lambda r: abs(r["post"]["closure"]))
    print("\nworst POST closure at a cutoff mount: %+.4e (%s, |lam^2| = %.3e), "
          "verdict %s" % (worst["post"]["closure"], worst["pol"],
                          worst["min_abs_lam2"], worst["post"]["verdict"]))
    print("severity drops: %d" % len(sev))
    V.dump(out, dict(rows=rows, n_severity_drops=len(sev)))


if __name__ == "__main__":
    main()
