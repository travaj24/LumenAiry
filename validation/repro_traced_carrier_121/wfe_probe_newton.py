# WHERE are the 81.2 % of Newton pixels that never converge, and do they
# matter?   (DIAG_LAST_GROUP_DECENTRE_2026_07_30, section 5 item 2 of the
# scope doc: "81 % of the grid is a lot of pixels to be out of domain and the
# claim has not been verified here".)
#
# NEEDS the TEMPORARY ``_DIAG_NEWTON`` sink in
# lumenairy/elements/_lens_traced.py (a dict populated inside ``_invert_newton``
# right after the iteration loop).  That instrumentation is REVERTED at the end
# of the study; this script only runs while it is present.
#
# usage:  ORD=-4,-2 python wfe_probe_newton.py
import os
import sys
import warnings

import numpy as np

import wfe_probe_common as P
import _d121_common as C
import wfe_probe_orders as OR
from lumenairy.elements import apply_real_lens_traced
from lumenairy.elements import _lens_traced as _LT
from lumenairy.elements._lens_traced import TiltedCarrier

LAM = P.LAM
K0 = P.K0


def main():
    m, n = (int(v) for v in os.environ.get('ORD', '-4,-2').split(','))
    iters = [int(v) for v in os.environ.get('ITERS', '12,60').split(',')]
    if not hasattr(_LT, '_DIAG_NEWTON'):
        print("_lens_traced._DIAG_NEWTON is absent -- the temporary "
              "instrumentation has been reverted.  Nothing to do.")
        return 1
    E_in, E_out_ship, carv, dx = OR.get_call(m, n)
    car = TiltedCarrier(*carv)
    N = E_in.shape[0]
    _pre, post, _g, _p = C.geometry()
    presc = post[-1]['prescription']
    surfs = P.element_surfaces(presc)
    ap = presc.get('aperture_diameter')
    print(f"design 121 LAST GROUP, order ({m},{n}):  Newton non-convergence "
          f"map")
    print(f"  grid {N}^2 @ {dx*1e6:.4f} um, aperture {ap*1e3:.4f} mm, "
          f"launch_radius {0.75*ap*1e3:.4f} mm, carrier centre "
          f"({car.x0*1e3:+.4f},{car.y0*1e3:+.4f}) mm")
    for it in iters:
        _LT._DIAG_NEWTON.clear()
        _LT._DIAG_NEWTON['on'] = True
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            E_out = np.asarray(apply_real_lens_traced(
                E_in, prescription=presc, wavelength=LAM, dx=dx, carrier=car,
                amplitude_model='ray_density', preserve_input_phase='remap',
                fit_radius_beam_factor=2.0, remap_sampling='full',
                ray_subsample=4, n_workers=8, newton_max_iters=it))
        runs = _LT._DIAG_NEWTON.get('runs', [])
        msg = [str(w.message) for w in wl if 'Newton inversion' in
               str(w.message)]
        print(f"\n--- newton_max_iters = {it}  ({len(runs)} Newton calls "
              f"recorded)")
        if msg:
            print(f"    warning: {msg[0][:120]}")
        r = runs[0]
        act = r['active']
        Xw, Yw = r['Xw'], r['Yw']
        xe, ye = r['xe'], r['ye']
        lr = r['launch_radius']
        bnd = r['bound']
        tot = act.size
        nun = int(act.sum())
        print(f"    coarse Newton grid {r['shape']} = {tot} px, sub={r['sub']},"
              f" n_launch={r['n_launch']}, launch_radius {lr*1e3:.4f} mm, "
              f"bound {bnd*1e3:.4f} mm, tol {r['tol']*1e6:.4f} um")
        print(f"    unconverged {nun}/{tot} = {100.0*nun/tot:.1f} %")
        # 1. are they clipped at the Newton bound?
        rr = np.hypot(xe, ye)
        clipped = (np.abs(np.abs(xe) - bnd) < 1e-12) | \
                  (np.abs(np.abs(ye) - bnd) < 1e-12)
        ood = rr > 0.99 * lr
        print(f"    of the unconverged: {100.0*float(clipped[act].mean()):.2f}"
              f" % sit ON the Newton clip bound, "
              f"{100.0*float(ood[act].mean()):.2f} % end outside "
              f"0.99*launch_radius (so their OPL is set to NaN and the field "
              f"is ZEROED there)")
        print(f"    of the CONVERGED:   "
              f"{100.0*float(ood[~act].mean()):.2f} % end outside "
              f"0.99*launch_radius")
        # 2. where are they, relative to the beam?
        de = np.hypot(Xw - 0.0, Yw - 0.0)
        print(f"    exit-plane radius of the unconverged pixels: min "
              f"{de[act].min()*1e3:.4f} mm, 1st pct "
              f"{np.percentile(de[act], 1)*1e3:.4f} mm, median "
              f"{np.median(de[act])*1e3:.4f} mm")
        # 3. what fraction of the RETURNED field's power do they carry?
        sub = r['sub']
        amp2 = np.abs(E_out) ** 2
        c2 = amp2[::sub, ::sub][:act.shape[0], :act.shape[1]]
        ptot = float(c2.sum())
        _pfrac = 100.0 * float(c2[act].sum()) / max(ptot, 1e-300)
        print(f"    |E_out|^2 on the coarse lattice carried by the "
              f"UNCONVERGED pixels: {_pfrac:.6f} %")
        nz = c2 > 0
        print(f"    of the pixels with NON-ZERO returned field, "
              f"{100.0*float(act[nz].mean()):.4f} % are unconverged "
              f"({int((act & nz).sum())} of {int(nz.sum())})")
        # 4. the returned field is zero exactly where?
        print(f"    returned field is EXACTLY ZERO on "
              f"{100.0*float((~nz).mean()):.2f} % of the coarse lattice; "
              f"unconverged fraction there {100.0*float(act[~nz].mean()):.2f} %")
        # 5. Is the exit chief ray's neighbourhood converged?
        xo, yo, _q, _q2, _q3, _q4 = P.trace_forward([car.x0], [car.y0], car,
                                                    surfs)
        ic = int(round(float(xo[0]) / dx + N / 2)) // sub
        jc = int(round(float(yo[0]) / dx + N / 2)) // sub
        h = 12
        blk = act[max(jc-h, 0):jc+h+1, max(ic-h, 0):ic+h+1]
        print(f"    in the +-{h*sub*dx*1e3:.3f} mm block centred on the exit "
              f"chief ray: {int(blk.sum())}/{blk.size} unconverged")
    print()
    print("VERDICT DATA: the fraction of the RETURNED field's power that sits "
          "on unconverged pixels is the number that decides whether the "
          "warning matters.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
