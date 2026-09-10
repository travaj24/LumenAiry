"""R12 -- the SAME defect on the 1-D core, on the case the warning names.

`_check_energy`'s text calls out "a groove ... equal to `n_substrate^2`".  That
is a 1-D binary grating whose LOW index equals the substrate's -- the 1-D
instance of the coincidence, reached through `rcwa_efficiency_1d`, which shares
`_sqrt_decay` but none of the 2-D even-parity machinery.  If the branch cut is
the cause, this must show the same lossless-closure defect before the fix and
none after, with no even-sector fold anywhere in the picture.

Walks a relative detune of the substrate index away from the coincidence, both
polarizations, several truncations.  Run with `PYTHONPATH` pointing at the
pre-fix archive and at the branch tip.
"""
from __future__ import annotations

import json
import warnings

import _lib as L
import numpy as np

from lumenairy.elements.rcwa import rcwa_efficiency_1d

PERIOD, WL, DEPTH = 0.5e-6, 0.6e-6, 0.35e-6
N_SUB, N_SUP = 1.5, 1.0
N_HIGH = 2.1                       # ridge
N_LOW = 1.5                        # groove -- EXACTLY n_substrate


def defect(n_sub, n_low, nord, pol):
    """``sum R + T - 1`` for the binary grating.  ``rcwa_efficiency_1d`` takes
    refractive INDICES (not permittivities) and a duty cycle."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = rcwa_efficiency_1d(PERIOD, N_HIGH, n_low, n_sub, N_SUP, DEPTH,
                                 0.5, WL, n_orders=nord, polarization=pol)
    R, T = np.asarray(out[1]), np.asarray(out[2])
    return float(np.sum(R) + np.sum(T) - 1.0)


def spectra(nord, pol):
    """The layer's PROPAGATING eigenvalues against the substrate's, plus how
    many layer modes are numerically on the cut and how many of those the
    PRE-fix root rule would have sent to the incoming branch.

    This is the measurement that separates "the permittivities coincide" from
    "a layer MODE coincides with a region MODE" -- only the second makes
    ``a + b`` singular."""
    import lumenairy.elements.rcwa.oned as O
    seen = []
    orig = O._eig_for

    def eig_for(xp):
        base = orig(xp)

        def wrapped(A):
            v, e = base(A)
            seen.append(np.asarray(v).astype(complex).copy())
            return v, e
        return wrapped
    O._eig_for = eig_for
    try:
        defect(N_SUB, N_LOW, nord, pol)
    finally:
        O._eig_for = orig
    lam2 = seen[0]
    r = np.sqrt(lam2)
    scale = max(float(np.max(np.abs(r))), 1.0)
    rel = np.abs(r.real) / scale
    on_cut = rel < 1e-8
    sub = np.array([-(N_SUB ** 2 - (m * WL / PERIOD) ** 2)
                    for m in range(-nord, nord + 1)])
    sub = sub[sub < 0]
    prop = np.sort(lam2[lam2.real < 0].real)
    gap = min(abs(p - q) for p in prop for q in sub) if prop.size else None
    return dict(pol=pol, nord=nord,
                layer_propagating=[float(v) for v in prop],
                substrate=[float(v) for v in np.unique(sub)],
                closest_layer_region_gap=float(gap) if gap is not None else None,
                n_on_cut=int(on_cut.sum()),
                n_prefix_incoming=int(np.sum(on_cut & (r.imag < 0))))


def main():
    a = L.arm()
    rows = []
    print("### build=%s tree=%s" % (a["build"], a["tree"]), flush=True)
    for pol in ("te", "tm"):
        for nord in (6, 10, 14):
            for d in (0.0, 1e-9, 1e-6, 1e-3):
                v = defect(N_SUB * (1.0 + d), N_LOW, nord, pol)
                rows.append(dict(pol=pol, nord=nord, detune=d, defect=v))
            line = "  %-3s n=%-3d " % (pol, nord) + "  ".join(
                "d=%-6.0e %+.3e" % (r["detune"], r["defect"])
                for r in rows[-4:])
            print(line, flush=True)
    coin = [r["defect"] for r in rows if r["detune"] == 0.0]
    off = [r["defect"] for r in rows if r["detune"] >= 1e-6]
    sp = [spectra(10, p) for p in ("te", "tm")]
    for x in sp:
        print("  %-3s layer propagating lam^2 %s vs substrate %s -> closest "
              "gap %.4f ; on-cut %d, of which the PRE-fix rule sent %d incoming"
              % (x["pol"],
                 ["%.4f" % v for v in x["layer_propagating"]],
                 ["%.4f" % v for v in x["substrate"]],
                 x["closest_layer_region_gap"], x["n_on_cut"],
                 x["n_prefix_incoming"]), flush=True)
    out = dict(rows=rows, spectra=sp,
               coincident_max=float(np.max(np.abs(coin))),
               detuned_max=float(np.max(np.abs(off))))
    print("  coincident |defect| max %.3e | detuned (>=1e-6) max %.3e"
          % (out["coincident_max"], out["detuned_max"]))
    print("R12JSON " + json.dumps({k: v for k, v in out.items()
                                   if k != "rows"}))
    L.dump("r12_oned_groove", out)


if __name__ == "__main__":
    main()
