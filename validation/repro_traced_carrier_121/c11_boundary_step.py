# C11 -- THE STEP AT THE BOUNDARY.
#
# Niche C1's whole finding was a DISCONTINUITY: the branch flipped at the first
# ulp of decentre and the returned field jumped by 8.32e-6 of peak at 1e-9
# PIXELS, "100x the pipeline's ~1e-7 roundoff floor and bought by nothing".
# Any replacement selector has to be held to the same standard at ITS boundary,
# so this measures the same quantity there.
#
# Two things are reported per geometry:
#
#   * the NULL contract -- at 1e-9 px, 0.4 px, 0.02 w and 0.049 w the returned
#     field must be BYTE-identical between the two flag states and to the
#     origin-referenced arm (the C1 pins, re-measured through the arbiter);
#   * the BOUNDARY step -- bisect for the decentre at which the arbiter
#     changes its mind, then report ``max|dE| / max|E|`` across it, next to
#     the two candidates' own OPL residuals there.
#
# The structural claim being tested: at the arbiter's boundary the two
# candidates have EQUAL residual against the traced map BY CONSTRUCTION, so
# the step is bounded by the accuracy they share -- which a fixed gate has
# only by luck.
#
# usage:  GEOMS=f3,f6,f6w python c11_boundary_step.py
import os
import sys

import numpy as np

os.environ.setdefault('LUMEN_PIN', '0')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c11_synth_sweep as S                                    # noqa: E402
import lumenairy.elements._lens_traced as LT                   # noqa: E402


def field(g, c, arbiter=True, tell=True):
    presc, n, dx, w, sub, frbf = S.GEOMS[g]
    old = LT.DECENTRED_FIT_ARBITER
    LT.DECENTRED_FIT_ARBITER = bool(arbiter)
    try:
        E, _ = S.run(presc, n, dx, w, sub, frbf, c, 'ship')
        if tell:
            return E
        # the ORIGIN-referenced arm: same physical field, element told the
        # beam is on the grid centre
        import warnings
        import lumenairy as la
        kw = dict(prescription=presc, wavelength=S.WL, dx=dx,
                  ray_subsample=sub, n_workers=1,
                  fit_radius_beam_factor=frbf, carrier=np.inf,
                  beam_centre=(0.0, 0.0), **S.TKW)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.asarray(la.apply_real_lens_traced(
                S.gauss(n, dx, w, c, 0.0), **kw))
    finally:
        LT.DECENTRED_FIT_ARBITER = old


def step(A, B):
    pk = float(np.abs(A).max())
    return float(np.abs(A - B).max()) / pk if pk > 0 else float('nan')


def picks_conc(g, c):
    """Did the arbiter take the concentric candidate at this decentre?  Read
    off the library's own scores, not guessed."""
    presc, n, dx, w, sub, frbf = S.GEOMS[g]
    rec = []
    orig = LT._decentred_fit_score

    def spy(xs, opl, wg, disc, wts, order):
        v = orig(xs, opl, wg, disc, wts, order)
        rec.append(v)
        return v

    LT._decentred_fit_score = spy
    try:
        S.run(presc, n, dx, w, sub, frbf, c, 'ship')
    finally:
        LT._decentred_fit_score = orig
    if len(rec) < 2:
        # the arbiter did not run at all -- below the C1 null gate the branch
        # IS the concentric one, byte-identically
        return True, None, None
    s_off, s_conc = rec[0], rec[1]
    return (s_conc <= s_off), s_conc, s_off


def main():
    geoms = os.environ.get('GEOMS', 'f3,f6,f6w').split(',')
    print(f"lumenairy {__import__('lumenairy').__version__}   "
          f"DECENTRED_FIT_ARBITER default = {LT.DECENTRED_FIT_ARBITER}   "
          f"C1 gate = max({LT._DECENTRE_GATE_PIXELS} dx, "
          f"{LT._DECENTRE_GATE_W_FRAC} w)", flush=True)
    for g in geoms:
        presc, n, dx, w, sub, frbf = S.GEOMS[g]
        print(f"\n########## {g}: {presc['name']}  w={w*1e3:.3f} mm "
              f"dx={dx*1e6:.1f} um ##########", flush=True)
        print("  NULL contract (C1's own offsets), arbiter ON:")
        for lab, c in (('1e-9 px', 1e-9 * dx), ('0.4 px', 0.4 * dx),
                       ('1 px', 1.0 * dx), ('0.02 w', 0.02 * w),
                       ('0.049 w', 0.049 * w)):
            a = field(g, c, arbiter=True)
            b = field(g, c, arbiter=True, tell=False)
            d = field(g, c, arbiter=False)
            print(f"    {lab:>9}  vs origin-referenced: "
                  f"{'BYTE-IDENTICAL' if np.array_equal(a, b) else f'{step(a, b):.3e}'}"
                  f"   vs arbiter OFF: "
                  f"{'BYTE-IDENTICAL' if np.array_equal(a, d) else f'{step(a, d):.3e}'}",
                  flush=True)
        # ---- the boundary -------------------------------------------------
        lo, hi = 0.0, 2.0
        p0, _s0c, _s0o = picks_conc(g, 1e-6 * w)
        p1, _s1c, _s1o = picks_conc(g, 2.0 * w)
        if p0 == p1:
            print(f"  BOUNDARY: the arbiter picks "
                  f"{'concentric' if p0 else 'off-centre'} across the whole "
                  f"0 - 2 w range on this geometry -- no boundary to cross, "
                  f"so no step to measure.", flush=True)
            continue
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            pm, _a, _b = picks_conc(g, mid * w)
            if pm == p0:
                lo = mid
            else:
                hi = mid
        print(f"  BOUNDARY at |c|/w = {0.5*(lo+hi):.5f}  "
              f"(bracketed to {hi-lo:.2e} w)", flush=True)
        for rel in (1e-6, 1e-4, 1e-3, 1e-2):

            cA = (0.5 * (lo + hi)) * (1.0 - rel) * w
            cB = (0.5 * (lo + hi)) * (1.0 + rel) * w
            A, B = field(g, cA), field(g, cB)
            pa, sac, sao = picks_conc(g, cA)
            pb, sbc, sbo = picks_conc(g, cB)
            print(f"    +-{rel:8.1e} of the boundary: max|dE|/max|E| = "
                  f"{step(A, B):.3e}   picks {'C' if pa else 'O'}->"
                  f"{'C' if pb else 'O'}   candidate OPL residuals "
                  + ('(below the C1 gate)' if sac is None else
                     f"{sac:.3e}/{sao:.3e} m"), flush=True)
        # and the same step measured for the SHIPPED 0.05 w gate on the same
        # geometry, i.e. what the constant would have cost had it been placed
        # at the physical crossover instead
        cb = 0.5 * (lo + hi) * w
        A = field(g, cb * 0.999, arbiter=False)
        B = field(g, cb * 1.001, arbiter=False)
        print(f"    (arbiter OFF, same +-0.1 % about the same point: "
              f"{step(A, B):.3e} -- the two branches do not swap there, so "
              f"this is the geometry's own smoothness)", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
