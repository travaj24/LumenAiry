# niche C8 on the SIX SYNTHETIC FIXTURES that keep the D1 fit guard opt-in.
#
# The fit guard removes design 121's on-axis ghost outright but REGRESSES two of
# these six (one from exactly clean to P/Pin = 1.00697 with 8.7e-03 of the input
# power beyond the exact-ray exit support at 88 % of peak).  Any structural
# replacement has to clear the same bar, so this reruns
# ``probe_ghost_synthetic``'s fixtures VERBATIM -- same singlets, same carriers,
# same r^4 residual, same metrics -- with three extra arms:
#
#     C8 f0     the support bound as a HARD cut
#     C8 f1     the shipped feather (1 exit-lattice cell)
#     C8 f2     twice that
#
# Reported exactly as probe_ghost_synthetic reports: total power over input
# power, the power beyond 3 and 5 beam radii of the (concentric) chief ray, and
# the largest |E| beyond 3 w over the peak.  An encircled-energy metric cannot
# see any of it.
#
# usage:  python probe_c8_synthetic.py
#         FEATHERS='0,1,2,4' python probe_c8_synthetic.py
import os
import sys
import warnings

import numpy as np
from probe_ghost_synthetic import BASE, CASES, field, singlet

import lumenairy as la
from lumenairy.elements import _lens_traced as LT


def run(E, dx, presc, carrier, launch, guard, bound=False, feather=None,
        **over):
    old = (LT.REMAP_STATIONARY_PHASE_LAUNCH,
           LT.REMAP_STATIONARY_PHASE_FIT_GUARD,
           LT.REMAP_INVERSE_SUPPORT_BOUND,
           LT._SUPPORT_BOUND_FEATHER_CELLS)
    LT.REMAP_STATIONARY_PHASE_LAUNCH = bool(launch)
    LT.REMAP_STATIONARY_PHASE_FIT_GUARD = bool(guard)
    LT.REMAP_INVERSE_SUPPORT_BOUND = bool(bound)
    if feather is not None:
        LT._SUPPORT_BOUND_FEATHER_CELLS = float(feather)
    try:
        kw = dict(BASE)
        kw['dx'] = dx
        kw.update(over)
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            out = np.asarray(la.apply_real_lens_traced(
                E, prescription=presc, carrier=carrier, **kw))
        nh = sum(1 for w in wl if 'HALO self-check FAILED' in str(w.message))
        return out, nh
    finally:
        (LT.REMAP_STATIONARY_PHASE_LAUNCH,
         LT.REMAP_STATIONARY_PHASE_FIT_GUARD,
         LT.REMAP_INVERSE_SUPPORT_BOUND,
         LT._SUPPORT_BOUND_FEATHER_CELLS) = old


def main():
    warnings.filterwarnings('ignore')
    feathers = [float(v) for v in
                os.environ.get('FEATHERS', '0,1,2').split(',')]
    import hashlib
    print("   lib sha256 %s" % hashlib.sha256(
        open(LT.__file__, 'rb').read()).hexdigest()[:16])
    hdr = ("%-32s %-10s %9s %11s %11s %10s %5s" %
           ('fixture', 'branch', 'P/Pin', 'P>3w', 'P>5w', 'amax3w', 'halo'))
    print(hdr)
    print('-' * len(hdr))
    for (lbl, n, dx, w, rc, al, r1, r2, th, z, ap) in CASES:
        E, X, Y = field(n, dx, w, rc, al)
        presc = singlet(r1, r2, th, z, ap)
        p_in = float((np.abs(E) ** 2).sum())
        R = np.hypot(X, Y)
        # NULL intervention first -- two identical runs, before any delta
        n1, _ = run(E, dx, presc, rc, False, False)
        n2, _ = run(E, dx, presc, rc, False, False)
        arms = [('C6 off', dict(launch=False, guard=False)),
                ('mask', dict(launch=True, guard=False)),
                ('weighted', dict(launch=True, guard=True))]
        for f in feathers:
            arms.append((f'C8 f{f:g}',
                         dict(launch=True, guard=False, bound=True,
                              feather=f)))
        arms.append(('C6off+C8', dict(launch=False, guard=False, bound=True,
                                      feather=1.0)))
        # the two interventions TOGETHER, measured rather than assumed
        arms.append(('wght+C8', dict(launch=True, guard=True, bound=True,
                                     feather=1.0)))
        for blab, kw in arms:
            F, nh = run(E, dx, presc, rc, **kw)
            pw = np.abs(F) ** 2
            pk = float(np.abs(F).max())
            m3 = R > 3.0 * w
            print("%-32s %-10s %9.5f %11.3e %11.3e %10.3e %5d"
                  % (lbl if blab == 'C6 off' else '', blab,
                     float(pw.sum()) / p_in, float(pw[m3].sum()) / p_in,
                     float(pw[R > 5.0 * w].sum()) / p_in,
                     float(np.abs(F)[m3].max() / max(pk, 1e-300)), nh))
        print("%-32s [null: array_equal=%s]" % ('', np.array_equal(n1, n2)))
    print()
    print("P>Nw = returned power beyond N beam radii from the (concentric) "
          "chief ray, over input power.")
    print("halo = firings of the v5.32 ray-density HALO self-check on that "
          "call.")


if __name__ == '__main__':
    sys.exit(main())
