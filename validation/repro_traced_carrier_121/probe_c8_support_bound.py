# niche C8 -- BOUND THE NEWTON INVERSE TO THE TRACED SAMPLES' OWN SUPPORT.
#
# THE DEFECT (established in ENERGY_CONSERVATION_AUDIT_2026_07_31.md and
# C6_FIT_GUARD_DECISION_2026_07_31.md): the C6 stationary-phase launch augments
# every ray direction by grad(a_fit) of a NON-radial polynomial, so the
# entrance->exit map loses the radial symmetry the CONCENTRIC hard-mask fit
# branch's safety argument rests on.  The global Chebyshev forward-map fit then
# EXTRAPOLATES outside its own data, the Newton inverse of that extrapolation
# folds far exit pixels back into the bright beam, and those pixels are handed
# real ray-density amplitude -- 4.7e-03 of Pin at 83 % of peak, 4-8 mm out,
# where the exact ray trace permits 3.6e-10.
#
# THE FIX UNDER TEST: an exit pixel outside the region the traced rays actually
# REACHED has no data behind it, so it gets zero amplitude.  The support is the
# convex hull of the exit landing points of the alive traced rays the entrance
# stop passes (the same rays the ray-density amplitude already keeps), taken
# BEFORE the fit-domain restriction; the bound tapers to zero with a raised
# cosine across a feather band just outside it.
#
# This probe is ELEMENT-LEVEL and differential: the chain's own group-5 call is
# replayed from cache under {C6 off, C6 on, C6 on + bound at several feather
# widths, C6 on + fit guard}, and every configuration is scored on the energy
# audit's halo family about the TRACED EXIT CHIEF RAY.
#
# usage:  ORDERS='0,0' python probe_c8_support_bound.py
#         ORDERS='0,0 -2,0 -4,-2' FEATHERS='0,0.5,1,2,4' \
#             python probe_c8_support_bound.py
import hashlib
import os
import sys
import time
import warnings

import _d121_common as C
import numpy as np
import probe_c6_element as E6
import wfe_probe_common as P

import lumenairy.elements._lens_traced as LT
from lumenairy.elements import apply_real_lens_traced
from lumenairy.elements._lens_traced import TiltedCarrier, _input_beam_amp_radius

LAM = P.LAM
SHELLS = (1.0e-3, 2.0e-3, 4.0e-3, 8.0e-3)


def element(E_in, presc, dx, car, rs, flag, guard=False, bound=False,
            feather=None, **over):
    """One element call with C6 / fit-guard / C8-bound all controlled."""
    old = (LT.REMAP_STATIONARY_PHASE_LAUNCH,
           LT.REMAP_STATIONARY_PHASE_FIT_GUARD,
           LT.REMAP_INVERSE_SUPPORT_BOUND,
           LT._SUPPORT_BOUND_FEATHER_CELLS)
    LT.REMAP_STATIONARY_PHASE_LAUNCH = bool(flag)
    LT.REMAP_STATIONARY_PHASE_FIT_GUARD = bool(guard)
    LT.REMAP_INVERSE_SUPPORT_BOUND = bool(bound)
    if feather is not None:
        LT._SUPPORT_BOUND_FEATHER_CELLS = float(feather)
    try:
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            opts = dict(E6.OPTS)
            opts.update(over)
            out = np.asarray(apply_real_lens_traced(
                E_in, prescription=presc, wavelength=LAM, dx=dx, carrier=car,
                ray_subsample=rs, **opts))
    finally:
        (LT.REMAP_STATIONARY_PHASE_LAUNCH,
         LT.REMAP_STATIONARY_PHASE_FIT_GUARD,
         LT.REMAP_INVERSE_SUPPORT_BOUND,
         LT._SUPPORT_BOUND_FEATHER_CELLS) = old
    fl = {'fold': 0, 'energy': 0, 'halo': 0, 'under': 0}
    for w in wl:
        t = str(w.message)
        # ORDER IS LOAD-BEARING: the halo message itself contains the words
        # "the energy self-check CANNOT see this", so an 'energy' test placed
        # first swallows every halo warning (measured: it read 0 firings on a
        # field whose halo is 33 % of peak).
        if 'HALO self-check FAILED' in t:
            fl['halo'] += 1
        elif 'fold caustic' in t:
            fl['fold'] += 1
        elif 'energy self-check FAILED' in t:
            fl['energy'] += 1
        elif 'NA_exit' in t:
            fl['under'] += 1
    return out, fl


def halo(F, p_in, Rr):
    pw = np.abs(F) ** 2
    tot = float(pw.sum())
    out = {'P': tot / p_in}
    for r in SHELLS:
        out['g%g' % (r * 1e3)] = float(pw[Rr > r].sum()) / p_in
    far = Rr > 4.0e-3
    amp = np.abs(F)
    pk = float(amp.max())
    out['amax4'] = (float(amp[far].max()) / pk) if far.any() and pk > 0 else 0.0
    out['r_rms'] = float(np.sqrt((pw * Rr ** 2).sum() / max(tot, 1e-300)))
    return out


def main():
    orders = [tuple(int(v) for v in o.split(','))
              for o in os.environ.get('ORDERS', '0,0').split()]
    feathers = [float(v) for v in
                os.environ.get('FEATHERS', '0,0.5,1,2,4').split(',')]
    rs = int(os.environ.get('RS', '4'))
    rn = int(os.environ.get('RN', '1024'))

    _pre, post, _g, _p = C.geometry()
    presc = post[-1]['prescription']
    surfs = P.element_surfaces(presc)

    print("   lib %s\n       sha256 %s" % (
        LT.__file__,
        hashlib.sha256(open(LT.__file__, 'rb').read()).hexdigest()[:16]))
    print("niche C8: the EXIT-SUPPORT BOUND, element level, design 121 "
          "group 5.")
    print("  gN = power beyond N mm of the traced exit chief ray / input "
          "power; amax4 = max|E| beyond 4 mm / peak.")
    print()
    for (m, n) in orders:
        E_in, _Eo, carv, dx = E6.get_call(m, n, rn=rn, rs=rs)
        car = TiltedCarrier(*carv)
        N = E_in.shape[0]
        w = float(_input_beam_amp_radius(E_in, dx, dx,
                                         centre=(car.x0, car.y0)))
        p_in = float((np.abs(E_in) ** 2).sum())
        ax = (np.arange(N) - N / 2) * dx
        Xg, Yg = np.meshgrid(ax, ax)
        xo, yo, _a, _b, _c, _d = P.trace_forward([car.x0], [car.y0], car,
                                                 surfs)
        Rr = np.hypot(Xg - float(xo[0]), Yg - float(yo[0]))

        cases = [('C6 off', dict(flag=False, bound=False)),
                 ('C6 off + C8', dict(flag=False, bound=True)),
                 ('C6 on  (HEAD)', dict(flag=True, bound=False)),
                 ('C6 on + guard', dict(flag=True, guard=True, bound=False))]
        for f in feathers:
            cases.append((f'C6 on + C8 f{f:g}',
                          dict(flag=True, bound=True, feather=f)))
        # the two interventions TOGETHER -- measured rather than assumed, since
        # the flag note now says C8 supersedes the guard.
        cases.append(('C6 on + C8 + gd',
                      dict(flag=True, guard=True, bound=True, feather=1.0)))

        # NULL INTERVENTION -- two identical HEAD runs, before any delta.
        a1, _ = element(E_in, presc, dx, car, rs, True)
        a2, _ = element(E_in, presc, dx, car, rs, True)
        print(f"order ({m},{n})  w {w*1e3:.4f} mm  dx {dx*1e6:.3f} um  N {N}"
              f"  exit chief ({float(xo[0])*1e3:+.4f},"
              f"{float(yo[0])*1e3:+.4f}) mm")
        print(f"   [NULL: array_equal={np.array_equal(a1, a2)} "
              f"max|dE|={float(np.abs(a1 - a2).max()):.3e}]")
        ref = None
        hdr = (f"{'config':>18} {'P/Pin':>9} {'g1':>10} {'g2':>10} {'g4':>10} "
               f"{'g8':>10} {'amax4':>9} {'r_rms/mm':>9} {'dP vs HEAD':>11} "
               f"{'r_cut/mm':>9} {'fold':>4} {'halo':>4}")
        print(hdr)
        print('-' * len(hdr))
        for lbl, kw in cases:
            t0 = time.time()
            F, fl = element(E_in, presc, dx, car, rs, **kw)
            if lbl == 'C6 on  (HEAD)':
                ref = F
            h = halo(F, p_in, Rr)
            if ref is not None and F.shape == ref.shape:
                d = np.abs(F - ref)
                dP = float(((np.abs(F) ** 2 - np.abs(ref) ** 2)).sum() / p_in)
                cut = Rr[d > 1e-3 * float(np.abs(ref).max())]
                r_cut = float(cut.min()) * 1e3 if cut.size else float('nan')
            else:
                dP, r_cut = float('nan'), float('nan')
            print(f"{lbl:>18} {h['P']:>9.6f} {h['g1']:>10.3e} "
                  f"{h['g2']:>10.3e} {h['g4']:>10.3e} {h['g8']:>10.3e} "
                  f"{h['amax4']:>9.2e} {h['r_rms']*1e3:>9.4f} "
                  f"{dP:>11.3e} {r_cut:>9.4f} {fl['fold']:>4d} "
                  f"{fl['halo']:>4d}   [{time.time()-t0:.0f}s]", flush=True)
        print()
    print("dP vs HEAD = (P_out - P_out(HEAD)) / P_in;  r_cut = the SMALLEST "
          "radius at which the field differs")
    print("             from HEAD by more than 1e-3 of peak (where the bound "
          "first bites).")


if __name__ == '__main__':
    sys.exit(main())
