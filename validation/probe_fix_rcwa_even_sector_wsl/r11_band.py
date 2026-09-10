"""R11 -- the two-sided derivation of ``_CUT_BAND_REL``.

The pin in :func:`_sqrt_decay` acts on exactly ONE population: modes whose
``r = sqrt(lam^2)`` has ``Im(r) < 0``.  Everything else is returned untouched
whatever the bar, so the bar only has to separate that population in two:

* the ones whose ``Im(r) < 0`` is the EIGENSOLVER'S BACKWARD ERROR -- a
  lossless cell's propagating modes, whose ``lam^2`` is exactly real negative
  in exact arithmetic.  Their ``|Re(r)|`` is rounding noise and the outgoing
  root has to be forced.  (NOISE side.)
* the ones whose ``Im(r) < 0`` is PHYSICS -- a mode with a real decay rate,
  where ``sqrt`` already chose the right root.  (SIGNAL side.)

The discriminating quantity is the one the code thresholds,
``|Re(r)| / max(max|r|, 1)``.  This probe collects it over EVERY ``Im(r) < 0``
mode of every fixture -- lossless cells and a LOSS LADDER down to a loss so
weak a caller might not think of it as lossy -- and reads the gap off the
resulting population instead of assuming one.  A bar is admissible only with
decades on both sides.
"""
from __future__ import annotations

import itertools
import json
import warnings

import _lib as L
import numpy as np

import lumenairy.elements.rcwa._core as C
import lumenairy.elements.rcwa.twod as TW

REC = []
SPLIT = 1e-4          # a probe-side split, decades away from either side


def install():
    orig = C._eig_for

    def eig_for(xp):
        base = orig(xp)

        def wrapped(A):
            w, v = base(A)
            wn = np.asarray(w).astype(complex)
            r = np.sqrt(wn)
            scale = max(float(np.max(np.abs(r))), 1.0)
            neg = r.imag < 0
            REC.append(dict(rel=(np.abs(r.real) / scale)[neg].tolist(),
                            imrel=(np.abs(r.imag) / scale)[neg].tolist(),
                            prop=(wn.real < 0)[neg].tolist()))
            return w, v
        return wrapped
    C._eig_for = eig_for


def run(loss, twist, nord, n_sub):
    tc = L.even_sector_cell(twist=twist)
    if loss:
        for i in range(3):
            tc[:, :, i, i] = tc[:, :, i, i] + 1j * loss
    REC.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        TW.rcwa_jones_2d(L.P_DEFAULT, L.P_DEFAULT, tc, n_sub, 1.0, 0.2e-6,
                         L.WL_DEFAULT, n_orders_x=nord, n_orders_y=nord,
                         symmetry=False)
    rel, prop = [], []
    for rec in REC:
        rel += rec["rel"]
        prop += rec["prop"]
    return rel, prop


def main():
    a = L.arm()
    install()
    allrel, allprop, alltag = [], [], []
    print("### build=%s tree=%s" % (a["build"], a["tree"]), flush=True)
    grid = list(itertools.product((0.0, 0.7, 1.2), (3, 4, 5), (1.5, 1.0, 1.8)))
    for twist, nord, n_sub in grid:
        rel, prop = run(0.0, twist, nord, n_sub)
        allrel += rel
        allprop += prop
        alltag += ["lossless"] * len(rel)
    for loss in (1e-8, 1e-6, 1e-4, 1e-2, 5e-2, 5e-1):
        for twist, nord in itertools.product((0.0, 0.7), (4, 5)):
            rel, prop = run(loss, twist, nord, 1.5)
            allrel += rel
            allprop += prop
            alltag += ["loss%.0e" % loss] * len(rel)
    rel = np.asarray(allrel)
    prop = np.asarray(allprop, dtype=bool)
    tag = np.asarray(alltag)
    lo, hi = rel[rel < SPLIT], rel[rel >= SPLIT]
    out = dict(n_total=int(rel.size), split=SPLIT,
               n_below=int(lo.size), below_max=float(lo.max()) if lo.size
               else None,
               n_above=int(hi.size), above_min=float(hi.min()) if hi.size
               else None,
               below_all_lossless_propagating=bool(
                   np.all(prop[rel < SPLIT]) and
                   np.all(np.char.startswith(tag[rel < SPLIT], "lossless"))),
               above_by_tag={t: int(np.sum(tag[rel >= SPLIT] == t))
                             for t in sorted(set(alltag))},
               below_by_tag={t: int(np.sum(tag[rel < SPLIT] == t))
                             for t in sorted(set(alltag))},
               bar=C._CUT_BAND_REL)
    print("Im(r) < 0 population: n=%d over %d fixtures" % (rel.size,
                                                           len(grid) + 24))
    print("  below %.0e : n=%-5d max=%.3e   (all propagating & lossless: %s)"
          % (SPLIT, out["n_below"], out["below_max"] or -1,
             out["below_all_lossless_propagating"]))
    print("  above %.0e : n=%-5d min=%.3e" % (SPLIT, out["n_above"],
                                              out["above_min"] or -1))
    print("  below by fixture class: %s" % out["below_by_tag"])
    print("\nBAR %.1e :  %.1f decades above the noise side (%.3e), "
          "%.1f decades below the signal side (%.3e)"
          % (out["bar"], np.log10(out["bar"] / out["below_max"]),
             out["below_max"], np.log10(out["above_min"] / out["bar"]),
             out["above_min"]))
    print("R11JSON " + json.dumps(out))
    L.dump("r11_band", out)


if __name__ == "__main__":
    main()
