# WHICH ``_sphere_parab_conversion`` call carries design 121's taper cost?
#
# ``fc_taper_locality.py`` shows the whole effect appears at group 6 and 99.7 %
# of it lies outside the FINAL conversion's onset.  There are three conversions
# in play there -- the group's ENTRANCE (+1, hands the traced element an
# exact-sphere-referenced field), its EXIT (-1, re-references the element's
# output for storage) and the chain's FINAL (+1, hands the paraxial readout a
# parabola-referenced envelope).  This script (a) censuses every call with its
# arguments and its own band-limit geometry, and (b) ablates the taper on ONE
# call signature at a time, scoring each through the same readout
# ``fc_instrument_121`` uses.
#
# usage:  ORD=0,0 python fc_taper_census.py
#         ORD=0,0 MODE=census python fc_taper_census.py
import os
import sys
import warnings

import numpy as np

os.environ.setdefault('LUMEN_PIN', '0')

import _d121_common as C                                       # noqa: E402
import fc_instrument_121 as FI                                 # noqa: E402
import hybrid_localize_121 as H                                # noqa: E402
import lumenairy.propagators.carrier as CM                     # noqa: E402
from approx_common import Patch                                # noqa: E402
from lumenairy.raytrace import RayBundle, trace                # noqa: E402

LAM = C.LAM
_ORIG = CM._sphere_parab_conversion


def _key(R, dx, sign):
    return (int(sign), round(float(R) * 1e6, 3), round(float(dx) * 1e9, 3))


def make_patch(log=None, off_keys=(), off_all=False):
    """``_sphere_parab_conversion`` that records every call and can drop the
    cos^2 taper (T == 1) on a chosen set of call signatures."""
    def q(shape, dx, wavelength, R, sign, w_beam=None, centre=(0.0, 0.0)):
        if not np.isfinite(R) or R == 0.0:
            return None
        k_ = _key(R, dx, sign)
        r_safe = (abs(R) ** 3 * wavelength / dx) ** (1.0 / 3.0)
        if log is not None:
            log.append({'key': k_, 'R': float(R), 'dx': float(dx),
                        'sign': int(sign), 'w': w_beam, 'shape': tuple(shape),
                        'r_safe': r_safe, 'centre': tuple(centre)})
        if off_all or k_ in off_keys:
            n, ny = int(shape[-1]), int(shape[-2])
            x = (np.arange(n, dtype=np.float64) - n / 2) * dx - centre[0]
            y = (np.arange(ny, dtype=np.float64) - ny / 2) * dx - centre[1]
            r2 = x[None, :] ** 2 + y[:, None] ** 2
            kk = 2.0 * np.pi / wavelength
            diff = CM._exact_sphere_eikonal((ny, n), dx, dx, wavelength, R,
                                            centre=centre) - r2 / (2.0 * R)
            return np.exp(sign * 1j * kk * diff)
        return _ORIG(shape, dx, wavelength, R, sign, w_beam=w_beam,
                     centre=centre)
    return [(CM, '_sphere_parab_conversion', q)]


def score(post, env, R, dx, L, M, rs, patch_items, xci, yci, split='exact'):
    with Patch(patch_items):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = C.la.propagate_traced_carrier_chain(
                env, [dict(g) for g in post], LAM, dx,
                r_in=C.la.TiltedCarrier(R, L, M), ray_subsample=rs,
                n_workers=8, final_distance=0.0, final_leg='paraxial',
                on_decentred_fit='ignore')
    (x0, y0, amp, ph0, p, q, surfs, nl, h, w, st) = FI.chain_launch(
        res, L, M, 9999, 3.0, 1, post, 5.0e-3, split)
    r = H.rs_spot(x0, y0, amp, ph0, p, q, surfs, 5.0e-3, xci, yci,
                  dx_out=0.4e-6, n_out=61, nl=nl)
    return r, st


def main():
    m, n = (int(v) for v in os.environ.get('ORD', '0,0').split(','))
    rn = int(os.environ.get('RN', '1024'))
    rs = int(os.environ.get('RS', '4'))
    split = os.environ.get('SPLIT', 'exact')
    print(FI._provenance())
    _pre, post, _g, period = C.geometry()
    env, R, dx, _P = C.chain_a(n=rn, rs=rs)
    L, M = m * LAM / period, n * LAM / period
    ch = trace(RayBundle(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
                         L=np.array([L]), M=np.array([M]),
                         N=np.array([np.sqrt(1 - L * L - M * M)]),
                         wavelength=LAM, alive=np.ones(1, bool),
                         opd=np.zeros(1)),
               C.post_surfaces(post), LAM, output_filter='last').image_rays
    xci, yci = float(ch.x[0]), float(ch.y[0])

    log = []
    r0, st0 = score(post, env, R, dx, L, M, rs, make_patch(log=log), xci, yci,
                    split)
    print(f"\nBASELINE (taper as shipped, SPLIT={split}): "
          f"EE3 {r0['ee3'] * 100:.4f}  FWHM {r0['fwhm'] * 1e6:.3f} um  "
          f"envelope step {st0:.4f} rad")

    seen = []
    print(f"\n{'#':>3} {'sign':>5} {'R mm':>12} {'dx um':>9} {'r_safe mm':>10} "
          f"{'onset/w':>9} {'w_beam mm':>10} {'centre mm':>18} {'n':>4}")
    for i, e in enumerate(log):
        ow = (0.75 * e['r_safe'] / e['w']) if e['w'] else np.nan
        cs = f"({e['centre'][0] * 1e3:+.3f},{e['centre'][1] * 1e3:+.3f})"
        print(f"{i:3d} {e['sign']:+5d} {e['R'] * 1e3:12.4f} "
              f"{e['dx'] * 1e6:9.4f} {e['r_safe'] * 1e3:10.4f} "
              f"{ow:9.3f} "
              f"{(e['w'] * 1e3 if e['w'] else float('nan')):10.4f} "
              f"{cs:>18} {e['shape'][-1]:4d}")
        if e['key'] not in seen:
            seen.append(e['key'])

    print(f"\n{len(seen)} distinct call signatures.  ABLATION (taper off on "
          f"ONE signature at a time):")
    print(f"{'signature (sign, R um, dx nm)':>36} {'EE3':>9} {'dEE3':>8} "
          f"{'FWHM um':>8} {'env step':>9}")
    for kk in seen:
        r, st = score(post, env, R, dx, L, M, rs,
                      make_patch(off_keys={kk}), xci, yci, split)
        print(f"{str(kk):>36} {r['ee3'] * 100:9.4f} "
              f"{(r['ee3'] - r0['ee3']) * 100:+8.4f} "
              f"{r['fwhm'] * 1e6:8.3f} {st:9.4f}")
    r, st = score(post, env, R, dx, L, M, rs, make_patch(off_all=True),
                  xci, yci, split)
    print(f"{'ALL OFF':>36} {r['ee3'] * 100:9.4f} "
          f"{(r['ee3'] - r0['ee3']) * 100:+8.4f} {r['fwhm'] * 1e6:8.3f} "
          f"{st:9.4f}")
    # the element-facing PAIR only (every call whose R is a group entrance or
    # exit, i.e. everything except the chain's FINAL +1 -- which is the last
    # (+1) entry in the log)
    finals = [e['key'] for e in log if e['sign'] == +1]
    pair = {kk for kk in seen if kk != finals[-1]}
    r, st = score(post, env, R, dx, L, M, rs, make_patch(off_keys=pair),
                  xci, yci, split)
    print(f"{'ELEMENT-FACING ONLY (final kept)':>36} {r['ee3'] * 100:9.4f} "
          f"{(r['ee3'] - r0['ee3']) * 100:+8.4f} {r['fwhm'] * 1e6:8.3f} "
          f"{st:9.4f}")
    r, st = score(post, env, R, dx, L, M, rs,
                  make_patch(off_keys={finals[-1]}), xci, yci, split)
    print(f"{'FINAL ONLY':>36} {r['ee3'] * 100:9.4f} "
          f"{(r['ee3'] - r0['ee3']) * 100:+8.4f} {r['fwhm'] * 1e6:8.3f} "
          f"{st:9.4f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
