# Is the C6 on-axis ghost fix a FIX, or a lucky point in a conditioning
# lottery?  SYNTHETIC fixtures -- no design-121 asset, no Zemax.
#
# probe_ghost_c6.py establishes on design 121 that the concentric hard-mask ray
# fit ghosts under the C6 launch (1.03e-03 of Pin beyond 4 mm at 33 % of peak)
# and that the D1 WEIGHTED restriction plus D7's order raise takes it to
# exactly 0.  Before that is shipped as a default it has to survive the obvious
# adversarial question: does the weighted branch ghost anywhere the hard mask
# does not?
#
# The fixtures are singlets with a converging carrier and the same r^4 residual
# the C6 unit tests use, swept over the grid / beam / carrier / aperture scales
# that matter, INCLUDING one built at design 121's own last-group scale
# (N=1024, dx=33 um, w=3.1 mm, R_c=-21 mm, 20 mm aperture).  Everything is
# reported as halo metrics about the ORIGIN (these fixtures are concentric), as
# a fraction of input power -- an encircled-energy metric cannot see any of it.
#
# usage:  python probe_ghost_synthetic.py
import sys
import warnings

import numpy as np

import lumenairy as la
from lumenairy.elements import _lens_traced as LT

WL = 1.31e-6
K0 = 2.0 * np.pi / WL


def _flat():
    return {'radius': np.inf, 'glass_before': 'air', 'glass_after': 'air',
            'conic': 0.0, 'radius_y': None, 'conic_y': None,
            'aspheric_coeffs': None, 'aspheric_coeffs_y': None}


def _surf(r, gb, ga):
    d = _flat()
    d['radius'], d['glass_before'], d['glass_after'] = r, gb, ga
    return d


def singlet(r1, r2, th, z, ap, glass='N-BK7'):
    """Biconvex singlet followed by a free leg ``z`` to the exit plane."""
    return {'name': 'ghost_singlet', 'aperture_diameter': ap,
            'surfaces': [_surf(r1, 'air', glass), _surf(r2, glass, 'air'),
                         _flat()],
            'thicknesses': [th, z]}


def carrier_WLM(x, y, s):
    sg = 1.0 if s > 0 else -1.0
    rho = np.sqrt(x * x + y * y + s * s)
    return sg * (rho - abs(s)), sg * x / rho, sg * y / rho


def field(n, dx, w, rc, alpha):
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    Wc, _l, _m = carrier_WLM(X, Y, rc)
    r2 = X * X + Y * Y
    a = (alpha / K0) * (r2 / (w * w)) ** 2
    return (np.exp(-r2 / (w * w))
            * np.exp(1j * K0 * (Wc + a))).astype(np.complex128), X, Y


BASE = dict(wavelength=WL, amplitude_model='ray_density',
            preserve_input_phase='remap', remap_sampling='full',
            parallel_amp=False, on_undersample='silent',
            on_noncollimated='silent', on_aperture_beam='silent',
            ray_subsample=4, fit_radius_beam_factor=2.0)


def run(E, dx, presc, carrier, launch, guard, **over):
    old = (LT.REMAP_STATIONARY_PHASE_LAUNCH,
           LT.REMAP_STATIONARY_PHASE_FIT_GUARD)
    LT.REMAP_STATIONARY_PHASE_LAUNCH = bool(launch)
    LT.REMAP_STATIONARY_PHASE_FIT_GUARD = bool(guard)
    try:
        kw = dict(BASE)
        kw['dx'] = dx
        kw.update(over)
        return np.asarray(la.apply_real_lens_traced(
            E, prescription=presc, carrier=carrier, **kw))
    finally:
        (LT.REMAP_STATIONARY_PHASE_LAUNCH,
         LT.REMAP_STATIONARY_PHASE_FIT_GUARD) = old


CASES = [
    # (label, N, dx, w, R_carrier, alpha, R1, R2, thick, z, aperture)
    ('weak f/70, exit far from focus', 512, 30e-6, 1.5e-3, -0.20, 3.0,
     200e-3, -200e-3, 4e-3, 5e-3, 14e-3),
    ('weak, 2x residual', 512, 30e-6, 1.5e-3, -0.20, 6.0,
     200e-3, -200e-3, 4e-3, 5e-3, 14e-3),
    ('medium f/40', 512, 30e-6, 1.5e-3, -0.10, 4.0,
     120e-3, -120e-3, 4e-3, 5e-3, 14e-3),
    ('medium, finer grid', 768, 25e-6, 1.5e-3, -0.15, 5.0,
     150e-3, -150e-3, 4e-3, 6e-3, 18e-3),
    ('collimated carrier (C6 inert)', 512, 30e-6, 1.5e-3, 1e9, 4.0,
     300e-3, -300e-3, 4e-3, 5e-3, 14e-3),
    ('DESIGN-121 SCALE stand-in', 1024, 33e-6, 3.1e-3, -0.021, 4.0,
     200e-3, -200e-3, 5e-3, 5e-3, 20e-3),
]


def main():
    warnings.filterwarnings('ignore')
    hdr = ("%-32s %-10s %9s %11s %11s %10s" %
           ('fixture', 'branch', 'P/Pin', 'P>3w', 'P>5w', 'amax3w'))
    print(hdr)
    print('-' * len(hdr))
    for (lbl, n, dx, w, rc, al, r1, r2, th, z, ap) in CASES:
        E, X, Y = field(n, dx, w, rc, al)
        presc = singlet(r1, r2, th, z, ap)
        p_in = float((np.abs(E) ** 2).sum())
        R = np.hypot(X, Y)
        # NULL intervention first -- two identical runs, before any delta
        n1 = run(E, dx, presc, rc, False, False)
        n2 = run(E, dx, presc, rc, False, False)
        for blab, launch, guard in (('C6 off', False, False),
                                    ('mask', True, False),
                                    ('weighted', True, True)):
            F = run(E, dx, presc, rc, launch, guard)
            pw = np.abs(F) ** 2
            pk = float(np.abs(F).max())
            m3 = R > 3.0 * w
            print("%-32s %-10s %9.5f %11.3e %11.3e %10.3e"
                  % (lbl if blab == 'C6 off' else '', blab,
                     float(pw.sum()) / p_in, float(pw[m3].sum()) / p_in,
                     float(pw[R > 5.0 * w].sum()) / p_in,
                     float(np.abs(F)[m3].max() / max(pk, 1e-300))))
        print("%-32s [null: array_equal=%s]" % ('', np.array_equal(n1, n2)))
    print()
    print("P>Nw = returned power beyond N beam radii from the (concentric) "
          "chief ray, over input power.")
    print("The DIRECTION of the weighted-vs-mask difference reverses between "
          "fixtures -- that is the finding.")


if __name__ == '__main__':
    sys.exit(main())
