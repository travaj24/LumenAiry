# Approximation audit -- EXACT accounting of what the paraxial (Fresnel /
# Sziklas-Siegman) envelope transport drops on EVERY coarse leg of design 121.
#
# This is deliberately NOT a substitution experiment.  The chain's leg is
#
#     env_out = FresnelTF(env, z_eff),   z_eff = z * R / (R + z)
#
# whose transfer phase is ``k z_eff (1 - (p^2+q^2)/2)`` in direction sines
# ``(p, q) = lambda * (fx, fy)``.  The exact free-space transfer phase for the
# same reduced leg is ``k z_eff sqrt(1 - p^2 - q^2)``, and for a leg carried at
# a mean tilt ``(L, M)`` the exact phase is ``k z_eff sqrt(1 - (L+p)^2 -
# (M+q)^2)`` -- whose ZEROTH and FIRST order terms the chain already handles
# exactly (``_tilt_obliquity`` piston + chief-ray advance) but whose SECOND
# order is anisotropic and is NOT implemented.
#
# So the two dropped terms are computed in closed form and integrated against
# the envelope's OWN measured angular spectrum at that leg -- no oracle, no
# hand-off, no differential floor.  The only modelling choice is the power
# quantile used to define "the occupied band", and three are reported.
#
# Env knobs: ORD (DOE order), RN, RS, NW.
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import approx_common as A                                      # noqa: E402

LAM = A.LAM
K0 = A.K0


def spectrum_terms(env, dx, z_eff, L, M):
    """Return the dropped-phase maps (rad) and the spectral power weight."""
    n = env.shape[-1]
    F = np.fft.fft2(env)
    P = np.abs(F) ** 2
    P = P / max(P.sum(), 1e-300)
    f = np.fft.fftfreq(n, d=dx)
    p = (LAM * f)[None, :] * np.ones((n, 1))
    q = (LAM * f)[:, None] * np.ones((1, n))
    s = p * p + q * q
    prop = s < 1.0
    # 1. the paraxial-kernel term: exact sqrt vs the 2nd-order truncation
    ex = np.where(prop, np.sqrt(np.clip(1.0 - s, 0.0, None)), 0.0)
    d_fresnel = K0 * z_eff * (ex - (1.0 - 0.5 * s))
    d_fresnel = np.where(prop, d_fresnel, 0.0)
    # 2. the tilt-anisotropy term: the exact 2nd-order form about (L, M)
    #    minus the isotropic one the chain applies.
    nz = np.sqrt(max(1.0 - L * L - M * M, 1e-300))
    axx = (1.0 - M * M) / nz ** 3
    ayy = (1.0 - L * L) / nz ** 3
    axy = (L * M) / nz ** 3
    d_tilt = -0.5 * K0 * z_eff * ((axx - 1.0) * p * p
                                  + 2.0 * axy * p * q
                                  + (ayy - 1.0) * q * q)
    return P, s, d_fresnel, d_tilt


def stat(P, d):
    """Power-weighted rms and the max over the band holding 99.9 % of the
    envelope's spectral power (both in WAVES)."""
    rms = float(np.sqrt(np.sum(P * d * d))) / (2 * np.pi)
    fl = np.sort(P.ravel())[::-1]
    cs = np.cumsum(fl)
    thr = fl[np.searchsorted(cs, 0.999)] if cs[-1] > 0 else 0.0
    m = P >= thr
    mx = float(np.abs(d[m]).max()) / (2 * np.pi) if m.any() else 0.0
    return rms, mx


def band(P, s):
    """99.0 / 99.9 / 99.99 % power radii of the envelope spectrum, in NA."""
    o = np.argsort(s.ravel())
    cs = np.cumsum(P.ravel()[o])
    sv = np.sqrt(s.ravel()[o])
    return tuple(float(sv[min(np.searchsorted(cs, f), sv.size - 1)])
                 for f in (0.99, 0.999, 0.9999))


def main():
    order = tuple(int(v) for v in os.environ.get('ORD', '-4,-2').split(','))
    post, env, R, dx, P_in, L, M, cen = A.setup(order)
    legs = []

    real = A.CM.propagate_carrier_referenced

    def rec(E_env, R_carrier, z, wavelength, dxx, dy=None):
        legs.append((np.array(E_env), float(np.ravel(R_carrier)[0]), float(z),
                     float(dxx)))
        return real(E_env, R_carrier, z, wavelength, dxx, dy)

    # a PARAXIAL final leg keeps this cheap; the legs it measures are the
    # coarse ones, which are identical either way (only the last group's
    # readout differs).
    with A.Patch([(A.CM, 'propagate_carrier_referenced', rec)]):
        A.la.propagate_traced_carrier_chain(
            env, post, LAM, dx, r_in=A.la.TiltedCarrier(R, L, M),
            ray_subsample=A.RS, n_workers=A.NW, final_distance=A.TRAILING,
            final_leg='paraxial', on_decentred_fit='ignore',
            on_gap_paraxial='ignore', on_na_proximity='ignore')

    print("order %s   tilt (L,M) = (%.6f, %.6f), theta = %.2f mrad"
          % (order, L, M, 1e3 * np.hypot(L, M)))
    print("legs recorded: %d   (grid N=%d)" % (len(legs), env.shape[-1]))
    print()
    hdr = ("%-4s %10s %10s %10s %9s %9s %9s | %10s %10s | %10s %10s" %
           ('leg', 'z (mm)', 'z_eff(mm)', 'R_in (mm)', 'dx (um)',
            'NA99.9', 'NA99.99', 'Fres rms', 'Fres max', 'tilt rms',
            'tilt max'))
    print(hdr)
    print('-' * len(hdr))
    tot_f = tot_t = 0.0
    for i, (E, Rl, z, dxl) in enumerate(legs):
        if z == 0.0:
            continue
        z_eff = z if not np.isfinite(Rl) else z * Rl / (Rl + z)
        P, s, df, dt = spectrum_terms(E, dxl, z_eff, L, M)
        fr, fm = stat(P, df)
        tr, tm = stat(P, dt)
        b = band(P, s)
        tot_f += fm
        tot_t += tm
        print("%-4d %10.4f %10.4f %10.3f %9.4f %9.5f %9.5f | %10.3e %10.3e "
              "| %10.3e %10.3e" %
              (i, z * 1e3, z_eff * 1e3, Rl * 1e3, dxl * 1e6, b[1], b[2],
               fr, fm, tr, tm))
    print()
    print("SUM over legs of the 99.9%%-band max, in waves:  "
          "Fresnel-kernel %.3e   tilt-anisotropy %.3e" % (tot_f, tot_t))
    print("(both are UPPER bounds on the coherent sum -- the terms have")
    print(" signs and partly cancel across legs; the rms columns are the")
    print(" power-weighted figure that maps onto a Strehl.)")


if __name__ == '__main__':
    main()
