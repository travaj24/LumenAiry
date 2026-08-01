# Niche C6 ON-AXIS GHOST -- mechanism and fix.
#
# THE OBSERVATION (recorded in _lens_traced.py's ``_REMAP_RESID_EIKONAL_DEGREE``
# docstring): with the C6 stationary-phase launch at residual degree 4 the
# ON-AXIS design-121 last-group call GAINS ~0.10 % of the input power as a far
# ghost lobe (exit power 0.9959 -> 0.9970).  Degree 3 gains none.  It appears
# only on the CONCENTRIC fit branch; the off-centre branch (weighted fit,
# _DECENTRED_FIT_POLY_ORDER = 10) is clean at every degree.
#
# THE PRIOR.  This is the D1 failure mode, and the file already documents it at
# ``_FIT_DISC_OUTSIDE_WEIGHT_REL`` (lines 1190-1263):
#
#     "[the hard sample mask] is safe only while the retained disc is
#      CONCENTRIC with the tensor-Chebyshev basis's own domain.  Then the
#      unconstrained directions of the fit inherit the map's RADIAL SYMMETRY,
#      the extrapolation outside the disc stays MONOTONE, and the Newton
#      inversion cannot find a second root."
#
# The C6 launch augments every ray's direction by grad(a_fit) of a degree-4
# polynomial fitted to the measured input residual.  That residual is NOT
# radially symmetric, so the augmented entrance->exit map is not either -- i.e.
# C6 invalidates the precondition the concentric hard-mask branch's safety
# argument rests on.  If the prior is right the ghost is a FOLD of the FITTED
# map outside the fit disc, and the two things that should kill it are the two
# things D1/D7 already ship on the off-centre branch: the WEIGHTED restriction
# instead of the hard NaN mask, and MORE TERMS.
#
# THE EXPERIMENT separates those two, on axis, by forcing the decentred branch
# with a NULL decentre (the C1 gate constants are set negative, so a zero
# offset counts as off centre; the fit disc is then still concentric -- only
# the RESTRICTION METHOD and the ORDER change):
#
#     mask   order 6    the shipped concentric path
#     mask   order 10   newton_poly_order raised, hard mask kept
#     weight order 6    gates forced + decentred_fit_poly_order=6
#     weight order 10   gates forced, D7 default
#
# Metrics are HALO metrics, not EE3: the ghost does not move the spot, so an
# encircled-energy number at 3 um cannot see it.
#
# usage:  ORDERS='0,0' DEG=3,4 python probe_ghost_c6.py
#         ORDERS='0,0 -4,-2' DEG=4 python probe_ghost_c6.py
import os
import sys
import time
import warnings

import numpy as np

import wfe_probe_common as P
import _d121_common as C
import probe_c6_element as E6
import lumenairy.elements._lens_traced as LT
from lumenairy.elements import apply_real_lens_traced
from lumenairy.elements._lens_traced import (TiltedCarrier,
                                             _input_beam_amp_radius)

LAM = P.LAM
K0 = P.K0

# radii (m) of the halo shells, measured from the traced EXIT CHIEF RAY
SHELLS = (1.0e-3, 2.0e-3, 4.0e-3, 8.0e-3)


def element(E_in, presc, dx, car, rs, flag, deg=None, force_dec=False,
            wrel=None, **over):
    """One element call with every C6 / fit-branch constant controlled.

    ``force_dec`` sets the niche-C1 decentre gates negative so a NULL decentre
    selects the OFF-CENTRE branch.  The fit disc stays concentric (it is built
    about ``(_bcx, _bcy) = (0, 0)``); what changes is the RESTRICTION METHOD
    (weights instead of a hard NaN mask) and, unless overridden, the order."""
    old = (LT.REMAP_STATIONARY_PHASE_LAUNCH, LT._REMAP_RESID_EIKONAL_DEGREE,
           LT._REMAP_RESID_DEGREE_CAP, LT._DECENTRE_GATE_PIXELS,
           LT._DECENTRE_GATE_W_FRAC, LT._FIT_DISC_OUTSIDE_WEIGHT_REL)
    LT.REMAP_STATIONARY_PHASE_LAUNCH = bool(flag)
    if deg is not None:
        LT._REMAP_RESID_EIKONAL_DEGREE = int(deg)
        LT._REMAP_RESID_DEGREE_CAP = max(int(deg), LT._REMAP_RESID_DEGREE_CAP)
    if force_dec:
        LT._DECENTRE_GATE_PIXELS = -1.0
        LT._DECENTRE_GATE_W_FRAC = -1.0
    if wrel is not None:
        LT._FIT_DISC_OUTSIDE_WEIGHT_REL = float(wrel)
    d = {}
    try:
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            opts = dict(E6.OPTS)
            opts.update(over)          # ``over`` may RESTATE an OPTS key
            out = np.asarray(apply_real_lens_traced(
                E_in, prescription=presc, wavelength=LAM, dx=dx, carrier=car,
                ray_subsample=rs, _remap_launch_out=d, **opts))
    finally:
        (LT.REMAP_STATIONARY_PHASE_LAUNCH, LT._REMAP_RESID_EIKONAL_DEGREE,
         LT._REMAP_RESID_DEGREE_CAP, LT._DECENTRE_GATE_PIXELS,
         LT._DECENTRE_GATE_W_FRAC, LT._FIT_DISC_OUTSIDE_WEIGHT_REL) = old
    fl = {'fold': 0, 'energy': 0, 'undersample': 0}
    for w in wl:
        t = str(w.message)
        if 'fold caustic' in t:
            fl['fold'] += 1
        elif 'energy self-check' in t:
            fl['energy'] += 1
        elif 'under-sampled' in t or 'undersample' in t:
            fl['undersample'] += 1
    return out, d, fl


def halo(F, p_in, Rr):
    """Halo / second-moment metrics about the exit chief ray.

    ``r_rms`` is the power-weighted second moment of the WHOLE returned field;
    ``r_rms_core`` restricts it to r < 1 mm.  A ghost carrying 1e-3 of the
    power at 8 mm moves ``r_rms`` by ~0.25 mm while leaving EE3 untouched --
    that ratio is the point of this metric."""
    pw = np.abs(F) ** 2
    tot = float(pw.sum())
    out = {'P': tot / p_in}
    for r in SHELLS:
        m = Rr > r
        out['g%g' % (r * 1e3)] = float(pw[m].sum()) / p_in
    far = Rr > 4.0e-3
    amp = np.abs(F)
    pk = float(amp.max())
    out['amax4'] = (float(amp[far].max()) / pk) if far.any() and pk > 0 else 0.0
    out['r_rms'] = float(np.sqrt((pw * Rr ** 2).sum() / max(tot, 1e-300)))
    core = Rr < 1.0e-3
    pc = float(pw[core].sum())
    out['r_rms_core'] = float(np.sqrt((pw[core] * Rr[core] ** 2).sum()
                                      / max(pc, 1e-300)))
    return out


def main():
    orders = [tuple(int(v) for v in o.split(','))
              for o in os.environ.get('ORDERS', '0,0').split()]
    degs = [int(v) for v in os.environ.get('DEG', '3,4').split(',')]
    rs = int(os.environ.get('RS', '4'))
    rn = int(os.environ.get('RN', '1024'))

    _pre, post, _g, _p = C.geometry()
    presc = post[-1]['prescription']
    surfs = P.element_surfaces(presc)

    import hashlib
    print("   lib %s\n       sha256 %s" % (
        LT.__file__,
        hashlib.sha256(open(LT.__file__, 'rb').read()).hexdigest()[:16]))
    print("niche C6 ON-AXIS GHOST: halo metrics of the last-group element "
          "call.")
    print("  mask/weight = the fit-disc RESTRICTION method; order = the "
          "forward-map fit order.")
    print("  gN = power fraction beyond N mm from the traced exit chief ray; "
          "amax4 = max |E|/peak beyond 4 mm.")
    print()
    for (m, n) in orders:
        E_in, _Eo, carv, dx = E6.get_call(m, n, rn=rn, rs=rs)
        car = TiltedCarrier(*carv)
        N = E_in.shape[0]
        w = float(_input_beam_amp_radius(E_in, dx, dx, centre=(car.x0, car.y0)))
        p_in = float((np.abs(E_in) ** 2).sum())
        ax = (np.arange(N) - N / 2) * dx
        Xg, Yg = np.meshgrid(ax, ax)
        xo, yo, _a, _b, _c, _d = P.trace_forward([car.x0], [car.y0], car, surfs)
        Rr = np.hypot(Xg - float(xo[0]), Yg - float(yo[0]))

        mode = os.environ.get('MODE', 'branch')
        if mode == 'branch':
            cases = [('C6 OFF          mask  o6', dict(flag=False))]
            for dg in degs:
                cases += [
                    (f'C6 deg{dg}         mask  o6', dict(flag=True, deg=dg)),
                    (f'C6 deg{dg}         mask  o10',
                     dict(flag=True, deg=dg, newton_poly_order=10)),
                    (f'C6 deg{dg}         mask  o14',
                     dict(flag=True, deg=dg, newton_poly_order=14)),
                    (f'C6 deg{dg}         wght  o6',
                     dict(flag=True, deg=dg, force_dec=True,
                          decentred_fit_poly_order=6)),
                    (f'C6 deg{dg}         wght  o10',
                     dict(flag=True, deg=dg, force_dec=True)),
                ]
            cases.append(('C6 OFF          wght  o10',
                          dict(flag=False, force_dec=True)))
        elif mode == 'sweep':
            # MECHANISM sweep #1 -- the FIT DOMAIN.  If the ghost is an
            # unconstrained-EXTRAPOLATION fold, widening the disc that
            # constrains the fit must weaken it monotonically and narrowing it
            # must strengthen it; if the ghost were the augmentation itself the
            # fit domain would be beside the point.
            cases = [('C6 OFF          mask  o6', dict(flag=False))]
            for f in (1.5, 2.0, 2.5, 3.0, 4.0):
                for dg in degs:
                    cases.append(
                        (f'deg{dg} mask o6  frbf {f}',
                         dict(flag=True, deg=dg, fit_radius_beam_factor=f)))
            # MECHANISM sweep #2 -- the D1 constant itself, on the forced
            # weighted branch.  D1 measured "ANY nonzero value removes the
            # fold"; 0.0 is documented as the exact restoration of the hard
            # NaN mask, i.e. the fail-before switch.
            for wr in (0.0, 1e-14, 1e-10, 1e-8, 1e-6, 1e-4):
                for dg in degs:
                    cases.append(
                        (f'deg{dg} wght o10 wrel {wr:g}',
                         dict(flag=True, deg=dg, force_dec=True, wrel=wr)))
        else:
            raise SystemExit(f"unknown MODE {mode!r}")

        # NULL INTERVENTION -- two identical shipped runs, before any delta.
        a1, _d1, _f1 = element(E_in, presc, dx, car, rs, True)
        a2, _d2, _f2 = element(E_in, presc, dx, car, rs, True)
        print(f"order ({m},{n})  w {w*1e3:.4f} mm  dx {dx*1e6:.3f} um  "
              f"exit chief ({float(xo[0])*1e3:+.4f},{float(yo[0])*1e3:+.4f}) mm"
              f"   [NULL: array_equal={np.array_equal(a1, a2)} "
              f"max|dE|={float(np.abs(a1 - a2).max()):.3e}]")
        hdr = (f"{'config':>26} {'deg':>4} {'P/Pin':>8} {'g1':>10} {'g2':>10} "
               f"{'g4':>10} {'g8':>10} {'amax4':>9} {'r_rms/mm':>9} "
               f"{'core/mm':>8} {'fold':>5}")
        print(hdr)
        print('-' * len(hdr))
        for lbl, kw in cases:
            t0 = time.time()
            F, dg, fl = element(E_in, presc, dx, car, rs, **kw)
            h = halo(F, p_in, Rr)
            print(f"{lbl:>26} {str(dg.get('degree','-')):>4} "
                  f"{h['P']:>8.5f} {h['g1']:>10.3e} {h['g2']:>10.3e} "
                  f"{h['g4']:>10.3e} {h['g8']:>10.3e} {h['amax4']:>9.2e} "
                  f"{h['r_rms']*1e3:>9.4f} {h['r_rms_core']*1e3:>8.4f} "
                  f"{fl['fold']:>5d}   [{time.time()-t0:.0f}s]", flush=True)
        print()


if __name__ == '__main__':
    sys.exit(main())
