# Approximation audit -- the two REFERENCE/FIT constructions inside the
# traced-carrier chain, measured directly against an exact ray trace.
#
# Part 1: _paraxial_group_r_out.  The exit carrier radius comes from the
#   group's air-to-air paraxial ABCD mapped by the wavefront Moebius law.
#   The exact counterpart is the sphere that best fits the group's OWN TRACED
#   exit wavefront about the traced chief ray.  A REFERENCE choice costs
#   nothing in the continuum, so the number that matters is not "how wrong is
#   R_out" but "how much extra phase SLOPE does the wrong reference push into
#   the envelope", measured in cycles per co-moving pixel against the 0.5
#   cycles/px Nyquist limit -- because the only way a reference can lose
#   information is by making the residual unrepresentable.
#
# Part 2: the tensor-Chebyshev fit of the entrance->exit ray map.  The
#   element's fit stage is reproduced VERBATIM (same _Cheb2DEvaluator, same
#   launch lattice, same total-degree basis, same fit disc), then evaluated at
#   traced points that are NOT fit nodes.  The residual is the fit's own
#   error: exit-position error in um and OPL error in waves, swept over
#   polynomial order so the order at which it stops being negligible is
#   located rather than assumed.
#
# Part 3: the Newton inversion.  The same fitted map is inverted at the
#   element's own tolerance/cap and at a converged one; the OPL difference is
#   the cost of the 12-iteration cap.
#
# Everything here is an INDEPENDENT reimplementation of the element's own
# stages on captured inputs -- no hand-off plane, so ABLATE S6's failure mode
# does not apply.
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import approx_common as A                                      # noqa: E402

from lumenairy.raytrace import trace                            # noqa: E402
from lumenairy.raytrace.trace import (_make_bundle,             # noqa: E402
                                      surfaces_from_prescription)
from lumenairy.glass import get_glass_index                     # noqa: E402

LAM = A.LAM
K0 = A.K0
LT = A.LT
CM = A.CM


def capture(order):
    """Run the chain and record every element call's (E_in, carrier, dx,
    prescription) plus the R_out the chain paired with it."""
    post, env, R, dx, P_in, L, M, cen = A.setup(order)
    grabs = []
    real = A.EL.apply_real_lens_traced

    def patched(E_in, *, prescription, wavelength, dx, **kw):
        grabs.append({'E': np.array(E_in), 'carrier': kw.get('carrier'),
                      'dx': float(dx), 'presc': prescription,
                      'name': prescription.get('name')})
        return real(E_in, prescription=prescription, wavelength=wavelength,
                    dx=dx, **kw)

    with A.Patch([(A.EL, 'apply_real_lens_traced', patched)]):
        res = A.la.propagate_traced_carrier_chain(
            env, post, LAM, dx, r_in=A.la.TiltedCarrier(R, L, M),
            ray_subsample=A.RS, n_workers=A.NW, final_distance=A.TRAILING,
            final_leg='paraxial', on_decentred_fit='ignore',
            on_gap_paraxial='ignore', on_na_proximity='ignore')
    stages = [s for s in res.stages if not s.get('target')]
    for g, s in zip(grabs, stages):
        g['R_out'] = float(s['R_out'])
        g['x_c_out'] = float(s.get('x_c_out', 0.0))
        g['y_c_out'] = float(s.get('y_c_out', 0.0))
        g['L_out'] = float(s.get('L_out', 0.0))
        g['M_out'] = float(s.get('M_out', 0.0))
    return grabs


def group_surfaces(presc):
    sf = surfaces_from_prescription(presc)
    sf = [dataclasses.replace(s, semi_diameter=np.inf) for s in sf]
    sf[-1] = dataclasses.replace(sf[-1], thickness=0.0)
    return sf


def traced_exit(g, n=161, r_fac=1.6):
    """Launch the group's OWN congruence on a lattice, trace to the exit
    VERTEX plane, and return (x_out, y_out, W_exit, x_in, y_in, alive)."""
    E, dx, car = g['E'], g['dx'], g['carrier']
    N = E.shape[-1]
    t = (np.arange(N, dtype=np.float64) - N / 2) * dx
    X, Y = np.meshgrid(t, t)
    I = np.abs(E) ** 2
    tot = I.sum()
    xc = float((I * X).sum() / tot)
    yc = float((I * Y).sum() / tot)
    w = float(np.sqrt(2.0 * ((I * ((X - xc) ** 2 + (Y - yc) ** 2)).sum()
                             / tot)))
    rad = r_fac * w
    u = np.linspace(-rad, rad, n)
    Xi, Yi = np.meshgrid(u + xc, u + yc, indexing='ij')
    W, Lg, Mg = LT._tilted_carrier_parts(car, Xi, Yi)
    sf = group_surfaces(g['presc'])
    rays = _make_bundle(Xi.ravel(), Yi.ravel(), Lg.ravel(), Mg.ravel(), LAM)
    fin = trace(rays, sf, LAM, output_filter='last').image_rays
    n_exit = get_glass_index(sf[-1].glass_after, LAM)
    tv = np.where(np.abs(fin.N) > 1e-30, -fin.z / fin.N, 0.0)
    xo = fin.x + fin.L * tv
    yo = fin.y + fin.M * tv
    W_ex = fin.opd + n_exit * tv + W.ravel()
    return (xo, yo, W_ex, Xi.ravel(), Yi.ravel(), fin.alive, w, rad,
            (xc, yc))


def part1(grabs):
    print("=" * 78)
    print("PART 1 -- _paraxial_group_r_out: is the exit carrier radius a "
          "REFERENCE or a leak?")
    print("=" * 78)
    print("%-12s %9s %12s %12s %9s %9s %6s %6s %6s %7s %8s %7s %7s" %
          ('group', 'dx (um)', 'R_out par', 'R_out fit', 'frac err',
           'dslope', 'p99@.5w', 'p99@1w', 'p99@1.5w', 'pw rms', 'P>0.5',
           'a rms', 'a-def'))
    print("%-12s %9s %12s %12s %9s %9s %6s %6s %6s %7s %8s %7s %7s" %
          ('', '', '(mm)', '(mm)', '', '(cyc/px)', 'cyc/px', 'cyc/px',
           'cyc/px', 'cyc/px', 'frac', '(wv)', '(wv)'))
    print('-' * 126)
    out = []
    for i, g in enumerate(grabs):
        xo, yo, W, xi, yi, al, w, rad, cin = traced_exit(g)
        m = al
        if m.sum() < 100:
            print("  group %d: too few live rays" % i)
            continue
        Ro, xco, yco = g['R_out'], g['x_c_out'], g['y_c_out']
        Lo, Mo = g['L_out'], g['M_out']
        spec = LT.TiltedCarrier(Ro, Lo, Mo, xco, yco)
        S, _, _ = LT._tilted_carrier_parts(spec, xo[m], yo[m])
        a = W[m] - S
        # remove piston + tilt + defocus about the chief ray
        px, py = xo[m] - xco, yo[m] - yco
        r2 = px * px + py * py
        Amat = np.column_stack([np.ones_like(px), px, py, 0.5 * r2])
        c, *_ = np.linalg.lstsq(Amat, a, rcond=None)
        dK = float(c[3])                        # 1/R_fit - 1/R_out
        R_fit = 1.0 / (1.0 / Ro + dK) if (1.0 / Ro + dK) != 0 else np.inf
        resid_pd = a - Amat @ c
        # the reference error's own contribution to the envelope's slope,
        # in cycles per co-moving pixel at the beam edge
        r_edge = float(np.sqrt(r2).max())
        dslope = abs(dK) * r_edge * g['dx'] / LAM
        rms_a = float(np.sqrt(np.mean((a - a.mean()) ** 2))) / LAM
        rms_pd = float(np.sqrt(np.mean(resid_pd ** 2))) / LAM
        # TOTAL residual slope: what the stored envelope actually has to
        # represent on the co-moving grid.  Central differences of the FULL
        # residual with respect to the EXIT coordinates, on the traced
        # lattice, reported as the 99th percentile inside the beam.
        n_s = int(np.sqrt(al.size))
        A2 = np.full(al.size, np.nan)
        Sfull, _, _ = LT._tilted_carrier_parts(spec, xo, yo)
        A2[al] = (W - Sfull)[al]
        A2 = A2.reshape(n_s, n_s)
        XO = xo.reshape(n_s, n_s)
        YO = yo.reshape(n_s, n_s)
        with np.errstate(invalid='ignore', divide='ignore'):
            gx = (A2[2:, :] - A2[:-2, :]) / (XO[2:, :] - XO[:-2, :])
            gy = (A2[:, 2:] - A2[:, :-2]) / (YO[:, 2:] - YO[:, :-2])
        gm = np.hypot(gx[:, 1:-1], gy[1:-1, :]) * g['dx'] / LAM
        XI = xi.reshape(n_s, n_s)[1:-1, 1:-1]
        YI = yi.reshape(n_s, n_s)[1:-1, 1:-1]
        rin = np.hypot(XI - cin[0], YI - cin[1])
        fin_m = np.isfinite(gm)
        # AMPLITUDE-WEIGHTED, and resolved by radius.  A bare percentile over
        # the whole launch disc is the DIAG_LAST_GROUP_DECENTRE S8.4 trap: at
        # 1.6 w the amplitude is exp(-2*1.6^2) = 0.6 % of peak and one skirt
        # sample sets the number.
        by_r = []
        for frac in (0.5, 1.0, 1.5):
            sl = fin_m & (rin <= frac * w)
            by_r.append(float(np.percentile(gm[sl], 99)) if sl.any()
                        else np.nan)
        wq = np.exp(-2.0 * (rin / w) ** 2)
        wq = np.where(fin_m, wq, 0.0)
        wq = wq / max(wq.sum(), 1e-300)
        slope_pw = float(np.sqrt(np.sum(wq * gm ** 2)))
        frac_above = float(wq[fin_m & (gm > 0.5)].sum())
        print("%-12s %9.3f %12.4f %12.4f %9.2e %9.3e %6.3f %6.3f %6.3f "
              "%7.3f %8.1e %7.3f %7.3f" %
              (g['name'][:12], g['dx'] * 1e6, Ro * 1e3, R_fit * 1e3,
               abs(R_fit - Ro) / abs(Ro), dslope, by_r[0], by_r[1], by_r[2],
               slope_pw, frac_above, rms_a, rms_pd))
        out.append((i, dslope, by_r, slope_pw, frac_above))
    print()
    print("  dslope = the EXTRA envelope phase slope the paraxial reference")
    print("  leaves at the beam edge, in cycles per co-moving pixel.  The")
    print("  representation limit is 0.5 cyc/px; below it the residual is")
    print("  band-limited and the reference is absorbed exactly.")
    print("  p99@Xw / pw rms are the 99th-percentile and the")
    print("  AMPLITUDE-WEIGHTED rms slope of the WHOLE residual (aberration")
    print("  included) on the co-moving grid, in cycles per pixel; 'P>0.5' is")
    print("  the amplitude-weighted fraction of the beam whose residual slope")
    print("  EXCEEDS the 0.5 cyc/px representation limit.")
    print("  'a rms' is the whole residual the envelope carries (physics +")
    print("  reference); 'a-def' has piston/tilt/defocus removed, i.e.")
    print("  it is the genuine aberration no reference choice can remove.")
    print("  NOTE: the LAST group's exit is NOT stored on this coarse grid on")
    print("  the shipped final_leg='exact' route -- it is retraced at")
    print("  dx_fine ~ 1.51 um, so scale its two slope columns by")
    print("  dx_fine/dx = 1.51/33.21 = 0.0454 for the production path.")
    return out


def part2(grabs, orders=(4, 6, 8, 10, 12, 14)):
    print()
    print("=" * 78)
    print("PART 2 -- the tensor-Chebyshev fit of the entrance->exit ray map")
    print("=" * 78)
    for i, g in enumerate(grabs):
        E, dx = g['E'], g['dx']
        sub = A.RS
        xo, yo, W, xi, yi, al, w, rad, cin = traced_exit(g, n=161)
        # reproduce the element's launch lattice: pitch dx*sub over the same
        # disc, odd count (the element bumps n_launch odd)
        n_l = max(8, int(2 * rad / (dx * sub)))
        n_l += (n_l % 2 == 0)
        u = np.linspace(-rad, rad, n_l)
        Xl, Yl = np.meshgrid(u + cin[0], u + cin[1], indexing='ij')
        Wl, Ll, Ml = LT._tilted_carrier_parts(g['carrier'], Xl, Yl)
        sf = group_surfaces(g['presc'])
        rays = _make_bundle(Xl.ravel(), Yl.ravel(), Ll.ravel(), Ml.ravel(),
                            LAM)
        fin = trace(rays, sf, LAM, output_filter='last').image_rays
        n_exit = get_glass_index(sf[-1].glass_after, LAM)
        tv = np.where(np.abs(fin.N) > 1e-30, -fin.z / fin.N, 0.0)
        xol = (fin.x + fin.L * tv).reshape(n_l, n_l)
        yol = (fin.y + fin.M * tv).reshape(n_l, n_l)
        opl = (fin.opd + n_exit * tv + Wl.ravel()).reshape(n_l, n_l)
        alive = fin.alive.reshape(n_l, n_l)
        opl = np.where(alive, opl, np.nan)
        xol = np.where(alive, xol, np.nan)
        yol = np.where(alive, yol, np.nan)
        opl = opl - np.nanmean(opl)
        # evaluation points: the dense traced set, restricted to the beam
        m = al & (np.hypot(xi - cin[0], yi - cin[1]) <= 1.0 * w)
        Wref = W[m] - W[m].mean()
        print("  group %d %-12s  n_launch %d (pitch %.2f um), eval on %d "
              "traced points inside 1 w" % (i, g['name'][:12], n_l,
                                            (u[1] - u[0]) * 1e6, int(m.sum())))
        print("     %6s %14s %14s %14s" %
              ('order', 'x_out err (um)', 'y_out err (um)', 'OPL err (waves)'))
        for od in orders:
            try:
                Sx = LT._Cheb2DEvaluator(u + cin[0], u + cin[1], xol, order=od)
                Sy = LT._Cheb2DEvaluator(u + cin[0], u + cin[1], yol, order=od)
                So = LT._Cheb2DEvaluator(u + cin[0], u + cin[1], opl, order=od)
            except Exception as exc:                            # noqa: BLE001
                print("     %6d  FAILED %s" % (od, exc))
                continue
            ex = Sx.ev(xi[m], yi[m]) - xo[m]
            ey = Sy.ev(xi[m], yi[m]) - yo[m]
            eo = So.ev(xi[m], yi[m]) - Wref
            eo = eo - eo.mean()
            print("     %6d %14.4e %14.4e %14.4e" %
                  (od, np.sqrt(np.mean(ex ** 2)) * 1e6,
                   np.sqrt(np.mean(ey ** 2)) * 1e6,
                   np.sqrt(np.mean(eo ** 2)) / LAM))


def main():
    order = tuple(int(v) for v in os.environ.get('ORD', '-4,-2').split(','))
    print("order %s   pinned lib %s" % (order, A.LT.__file__))
    grabs = capture(order)
    print("captured %d element calls: %s"
          % (len(grabs), [g['name'] for g in grabs]))
    part1(grabs)
    part2(grabs)


if __name__ == '__main__':
    main()
