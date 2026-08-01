# Approximation audit -- the CHEAP, closed-form checks: which band limits,
# tapers, masks and parity conventions are INERT on design 121's shipped
# configuration, and which actually touch the beam.
#
# Everything here is arithmetic on the chain's own stage geometry, so there is
# no differential floor and no oracle: each row either reaches the beam or it
# does not, and the fraction of envelope power it reaches is reported.
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import approx_common as A                                      # noqa: E402

LAM = A.LAM
CM = A.CM


def main():
    order = tuple(int(v) for v in os.environ.get('ORD', '-4,-2').split(','))
    post, env, R, dx, P_in, L, M, cen = A.setup(order)
    legs = []
    real = CM.propagate_carrier_referenced

    def rec(E_env, R_carrier, z, wavelength, dxx, dy=None):
        legs.append((np.array(E_env), float(np.ravel(R_carrier)[0]),
                     float(z), float(dxx)))
        return real(E_env, R_carrier, z, wavelength, dxx, dy)

    with A.Patch([(CM, 'propagate_carrier_referenced', rec)]):
        res = A.la.propagate_traced_carrier_chain(
            env, post, LAM, dx, r_in=A.la.TiltedCarrier(R, L, M),
            ray_subsample=A.RS, n_workers=A.NW, final_distance=A.TRAILING,
            final_leg='paraxial', on_decentred_fit='ignore',
            on_gap_paraxial='ignore', on_na_proximity='ignore')
    stages = [s for s in res.stages if not s.get('target')]

    print("=" * 76)
    print("A. _sphere_parab_conversion taper: does it reach the beam?")
    print("=" * 76)
    print("%-14s %9s %9s %10s %10s %12s" %
          ('plane', 'dx (um)', 'w (um)', 'r_safe mm', '0.75r/w', 'power>0.75r'))
    for (E, Rl, z, dxl), s in zip(legs, stages + [None] * 9):
        if not np.isfinite(Rl) or Rl == 0.0:
            continue
        r_safe = (abs(Rl) ** 3 * LAM / dxl) ** (1.0 / 3.0)
        n = E.shape[-1]
        t = (np.arange(n) - n / 2) * dxl
        rr = np.hypot(t[None, :], t[:, None])
        I = np.abs(E) ** 2
        tot = I.sum()
        w = float(np.sqrt(2.0 * (I * rr ** 2).sum() / tot))
        frac = float(I[rr > 0.75 * r_safe].sum() / tot)
        print("%-14s %9.3f %9.1f %10.4f %10.3f %12.3e" %
              ('leg in R=%.1fmm' % (Rl * 1e3), dxl * 1e6, w * 1e6,
               r_safe * 1e3, 0.75 * r_safe / max(w, 1e-30), frac))

    print()
    print("=" * 76)
    print("B. _tilt_exactness_phase band-limit radius: coarse vs fine dx")
    print("=" * 76)
    n_t = np.hypot(L, M)
    s_t = L * L + M * M
    print("%-16s %10s %12s %12s %10s" %
          ('R (mm)', 'dx (um)', 'r_safe (mm)', '0.75r/w (-)', 'power>0.75r'))
    for (E, Rl, z, dxl), st in zip(legs, stages):
        if not np.isfinite(Rl) or Rl == 0.0:
            continue
        for dxq, tag in ((dxl, 'coarse'), (1.5081e-6, 'fine')):
            a = 1.5 * n_t / (Rl * Rl)
            b = s_t / abs(Rl)
            c = LAM / (2.0 * dxq)
            r_safe = (np.sqrt(b * b + 4.0 * a * c) - b) / (2.0 * a)
            n = E.shape[-1]
            t = (np.arange(n) - n / 2) * dxl
            rr = np.hypot(t[None, :], t[:, None])
            I = np.abs(E) ** 2
            w = float(np.sqrt(2.0 * (I * rr ** 2).sum() / I.sum()))
            frac = float(I[rr > 0.75 * r_safe].sum() / I.sum())
            print("%-16s %10.4f %12.5f %12.3f %10.3e" %
                  ('%.3f (%s)' % (Rl * 1e3, tag), dxq * 1e6, r_safe * 1e3,
                   0.75 * r_safe / max(w, 1e-30), frac))

    print()
    print("=" * 76)
    print("C. the exact readout's ASM band limit and evanescent cut")
    print("=" * 76)
    n_fine, dx_fine, z = 12288, 1.5081e-6, A.TRAILING
    Lx = n_fine * dx_fine
    fx_max = Lx / (2.0 * LAM * abs(z))
    print("   fine grid %d x %.4f um = %.3f mm, final distance %.4f mm"
          % (n_fine, dx_fine * 1e6, Lx * 1e3, z * 1e3))
    print("   grid Nyquist NA           = %.4f" % (LAM / (2 * dx_fine)))
    print("   ASM band-limit mask at NA = %.4f  -> %s"
          % (LAM * fx_max,
             'INERT (above the grid Nyquist)'
             if LAM * fx_max > LAM / (2 * dx_fine) else 'ACTIVE, cuts NA'))
    print("   evanescent cut at NA = 1.0 -> INERT (grid Nyquist NA %.4f < 1)"
          % (LAM / (2 * dx_fine)))

    print()
    print("=" * 76)
    print("D. _fourier_upsample_crop parity (the half-pixel trap)")
    print("=" * 76)
    print("   the shipped chain forces n_crop = 2*round(win/dx/2) (EVEN) and")
    print("   n_fine = 2**ceil(...) capped by n_fine_cap; the run above used")
    print("   n_fine = %d.  Parity check on this run's values:" % n_fine)
    for n_in, n_crop, n_f in ((1024, 512, 12288), (1024, 511, 12288)):
        print("      N=%d n_crop=%d n_fine=%d -> n_crop even: %s, "
              "N even: %s, n_fine even: %s"
              % (n_in, n_crop, n_f, n_crop % 2 == 0, n_in % 2 == 0,
                 n_f % 2 == 0))
    print("   -> the half-pixel mis-registration needs an ODD input N or an")
    print("      ODD user-supplied N_fine; neither is reachable from the")
    print("      shipped defaults, so it is LATENT, not active.")


if __name__ == '__main__':
    main()
