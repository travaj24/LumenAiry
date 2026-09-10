"""V11 -- the band is relative to the LARGEST mode, not to the mode it judges.

``_sqrt_decay`` fires when ``|Re(r)| <= _CUT_BAND_REL * max(max|r|, 1)``.  The
scale is the LARGEST root in the array, so whether a given mode is treated as
"numerically on the cut" depends on the OTHER modes of the same layer.  A mode
whose real part is a substantial fraction of ITS OWN magnitude is therefore
conjugated whenever the layer's spectrum is wide enough:

    fires  <=>  (|Re(r)| / |r|) * (|r| / max|r|)  <=  _CUT_BAND_REL

so a mode at relative real part ``rho`` is caught once its magnitude falls
below ``_CUT_BAND_REL / rho`` of the spectrum's top.  At ``rho = 1e-3`` that is
five decades of dynamic range in ``|lam|`` -- reachable in a real solve when a
layer mode approaches CUTOFF while high evanescent orders sit at ``|lam| ~ 20``.

This probe measures, over every fixture of ``v3_band.py`` plus a ladder of
mounts driven onto a layer cutoff, the quantity the band SHOULD arguably be
thresholding -- each acted-on mode's OWN ``|Re(r)| / |r|`` -- and reports the
worst one.  It also states the same thing as a dynamic-range reading, so the
result can be compared against any layer spectrum.

Usage: python v11_scale_relativity.py <out.json>
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v3_band as B  # noqa: E402
import v_fixtures as V  # noqa: E402

BAND = 1e-8


def _score(lam2):
    r = V.principal_root(lam2)
    scale = max(float(np.max(np.abs(r))), 1.0) if r.size else 1.0
    acted = (np.abs(r.real) <= BAND * scale) & (r.imag < 0)
    out = dict(n_modes=int(r.size), scale=scale, n_acted=int(acted.sum()),
               dynamic_range=(float(np.max(np.abs(r))
                                    / max(np.min(np.abs(r)), 1e-300))
                              if r.size else 1.0))
    if acted.any():
        own = np.abs(r.real[acted]) / np.maximum(np.abs(r[acted]), 1e-300)
        k = int(np.argmax(own))
        idx = np.where(acted)[0][k]
        out.update(worst_own_relative_real=float(own[k]),
                   worst_abs_r=float(np.abs(r[idx])),
                   worst_lam2=[float(lam2[idx].real), float(lam2[idx].imag)])
    return out


def census(call):
    eig = V.EigSpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with eig:
            try:
                call()
            except Exception:
                pass
    if not eig.seen:
        return None
    best = None
    for w in eig.seen:
        s = _score(np.asarray(w))
        if best is None or s.get("worst_own_relative_real", -1.0) > \
                best.get("worst_own_relative_real", -1.0):
            best = s
    return best


def main():
    V.require_local_tree()
    out = sys.argv[1]
    rows = []
    for name, cls, call in B.fixtures():
        s = census(call)
        if s is None:
            continue
        s.update(name=name, cls=cls)
        rows.append(s)
    acted = [r for r in rows if "worst_own_relative_real" in r]
    acted.sort(key=lambda r: -r["worst_own_relative_real"])
    print("worst OWN relative real part among modes the band ACTS on")
    for r in acted[:15]:
        print("  %-28s %-9s own|Re r|/|r|=%.4e  |r|=%.3e  scale=%.3e  "
              "dynRange=%.2e  nActed=%d" % (
                  r["name"], r["cls"], r["worst_own_relative_real"],
                  r["worst_abs_r"], r["scale"], r["dynamic_range"],
                  r["n_acted"]))
    worst = acted[0] if acted else None
    if worst:
        print("\nWORST: %s -- a mode whose real part is %.3e of its OWN "
              "magnitude was conjugated, because it is %.3e of the spectrum's "
              "largest root."
              % (worst["name"], worst["worst_own_relative_real"],
                 worst["worst_abs_r"] / worst["scale"]))
    print("\nDynamic range needed for a mode at relative real part rho:")
    for rho in (1e-1, 1e-3, 1e-6, 1e-8, 1e-10):
        print("   rho=%.0e -> |r|/max|r| <= %.1e  (%.1f decades)"
              % (rho, BAND / rho, np.log10(rho / BAND)))
    V.dump(out, dict(rows=rows,
                     worst_own_relative_real=(worst["worst_own_relative_real"]
                                              if worst else None)))


if __name__ == "__main__":
    main()
