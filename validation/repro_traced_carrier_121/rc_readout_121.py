# D121 RESIDUAL CLOSURE -- is the per-order residual a FIELD difference or an
# EE-METRIC difference?
#
# ``rs_spot`` scores EE3 as
#
#     EE3 = I[ hypot(Xg - cx, Yg - cy) <= 3 um ].sum() / I.sum()
#
# on a ``dx_out``-pitched lattice, with ``(cx, cy)`` each ARM's OWN intensity
# centroid.  That is a HARD BINARY PIXEL MASK whose boundary ring, at the
# campaign's ``dx_out = 0.4 um``, is ~47 pixels long, and each of those pixels
# carries a few hundredths of a point.  Moving the mask centre by a fraction of
# one pixel flips a subset of them in or out.  The two arms have DIFFERENT
# centroids, so the quantisation does NOT cancel in the difference.
#
# This script measures that directly, on the SAME intensity arrays the campaign
# scores, and separates
#
#   GENUINE   how much EE3 a real off-centre circle costs      (smooth, even)
#   QUANTUM   how much the pixel mask alone adds               (sawtooth)
#
# by scoring every mask both HARD (as shipped) and AREA-EXACT (each pixel
# weighted by the fraction of its area inside the circle, 16x16 supersampled).
# An area-exact mask is the same physical measurement with the quantisation
# removed; the difference between the two on one intensity array is pure
# instrument.
#
# Arms are ``fc_table_121.py``'s, verbatim, through ``fc_instrument_121``:
#   * ``oracle CARRY=1``                   -- the true ceiling
#   * ``chain taper=off split=exact``      -- the converged chain readout
# and the shipped-resolution EE3 of each MUST reproduce ``_fc_table.txt``.
#
# usage:
#   ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' python rc_readout_121.py
#   ORDERS='-1,0' DXO=0.2 NOUT=121 python rc_readout_121.py     (finer readout)
import hashlib
import os
import sys
import time

import numpy as np

os.environ.setdefault('LUMEN_PIN', '0')

import _d121_common as C                                       # noqa: E402
import fc_instrument_121 as FI                                 # noqa: E402
import hybrid_localize_121 as H                                # noqa: E402
from lumenairy.raytrace import RayBundle, trace                # noqa: E402

LAM = C.LAM
_HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------
def area_frac(ax, cx, cy, rad, ss=32):
    """Fraction of each pixel's AREA inside the disc of radius ``rad`` about
    ``(cx, cy)``, by ``ss x ss`` supersampling.  ``ss=32`` puts the residual
    quantisation of this estimator at ~1/1024 of a boundary pixel, i.e. three
    orders below the effect it is measuring (checked by ss=8 vs ss=32 below)."""
    dxo = float(ax[1] - ax[0])
    off = (np.arange(ss) + 0.5) / ss - 0.5
    xs = (ax[:, None] + off[None, :] * dxo).ravel()
    Xs, Ys = np.meshgrid(xs, xs)
    ins = ((Xs - cx) ** 2 + (Ys - cy) ** 2) <= rad * rad
    n = ax.size
    return ins.reshape(n, ss, n, ss).mean(axis=(1, 3))


def hard_frac(ax, cx, cy, rad):
    """The SHIPPED mask: pixel centre inside the circle or not."""
    Xg, Yg = np.meshgrid(ax, ax)
    return (np.hypot(Xg - cx, Yg - cy) <= rad).astype(np.float64)


def ee(I, ax, cx, cy, rad, mode='hard', ss=32):
    f = (hard_frac(ax, cx, cy, rad) if mode == 'hard'
         else area_frac(ax, cx, cy, rad, ss))
    return float((I * f).sum()) / float(I.sum())


def centroid(I, ax):
    Xg, Yg = np.meshgrid(ax, ax)
    t = float(I.sum())
    return float((I * Xg).sum() / t), float((I * Yg).sum() / t)


def peak_subpixel(I, ax):
    """Sub-pixel intensity maximum by an independent quadratic fit in each
    axis through the 3 samples about the discrete argmax."""
    j, i = np.unravel_index(int(np.argmax(I)), I.shape)
    dxo = float(ax[1] - ax[0])

    def _q(a, b, c):
        d = a - 2.0 * b + c
        return 0.0 if abs(d) < 1e-300 else 0.5 * (a - c) / d

    fx = (_q(I[j, i - 1], I[j, i], I[j, i + 1])
          if 0 < i < I.shape[1] - 1 else 0.0)
    fy = (_q(I[j - 1, i], I[j, i], I[j + 1, i])
          if 0 < j < I.shape[0] - 1 else 0.0)
    return ax[i] + fx * dxo, ax[j] + fy * dxo


# ---------------------------------------------------------------------------
# the two arms, verbatim
# ---------------------------------------------------------------------------
def arm_launch(which, post, env, R, dx, L, M, rs, clip, strip=0):
    """``strip=0`` is the campaign's pairing: the oracle CARRIES the DOE-plane
    residual phase (the true ceiling) and the chain gets the same field.
    ``strip=1`` is the MATCHED-FIELD control: BOTH arms are handed
    ``|env_doe|``, i.e. the DOE-plane residual phase is removed before either
    sees it.  The difference between the two pairings is the whole cost of
    transporting that residual, which is the only thing the chain does with
    diffraction and the oracle does with rays."""
    if which == 'oracle':
        return FI.oracle_launch(env, R, dx, L, M,
                                int(os.environ.get('NLO', '321')), clip,
                                0 if strip else 1, post, BACK)
    res, _w = FI.run_chain(post, env, R, dx, L, M, rs, 'off', strip)
    return FI.chain_launch(res, L, M, 9999, clip, 1, post, BACK, 'exact')


BACK = 5.0e-3


def main():
    orders = os.environ.get(
        'ORDERS', '0,0 -1,0 -2,0 -3,0 -4,0 -4,-2').split()
    rn = int(os.environ.get('RN', '1024'))
    rs = int(os.environ.get('RS', '4'))
    dxo = float(os.environ.get('DXO', '0.4')) * 1e-6
    nout = int(os.environ.get('NOUT', '61'))
    clip = float(os.environ.get('CLIP', '3.0'))
    strip = int(os.environ.get('STRIP', '0'))
    tag = os.environ.get('TAG', f"{dxo * 1e6:g}um_{nout}"
                         + ('_strip' if strip else ''))
    print(FI._provenance(), flush=True)
    print(f"readout lattice dx_out {dxo * 1e6:g} um  n_out {nout}  "
          f"window +-{(nout - 1) / 2 * dxo * 1e6:.1f} um   RN={rn} rs={rs} "
          f"CLIP={clip}  STRIP={strip}", flush=True)

    _pre, post, _g, period = C.geometry()
    env, R, dx, _P = C.chain_a(n=rn, rs=rs)
    store = {}
    for o in orders:
        m, n = (int(v) for v in o.split(','))
        L, M = m * LAM / period, n * LAM / period
        ch = trace(RayBundle(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
                             L=np.array([L]), M=np.array([M]),
                             N=np.array([np.sqrt(1 - L * L - M * M)]),
                             wavelength=LAM, alive=np.ones(1, bool),
                             opd=np.zeros(1)),
                   C.post_surfaces(post), LAM,
                   output_filter='last').image_rays
        xci, yci = float(ch.x[0]), float(ch.y[0])
        print(f"\n########## ORDER ({m:+d},{n:+d}) ##########", flush=True)
        for which in ('oracle', 'chain'):
            t0 = time.time()
            (x0, y0, amp, ph0, p, q, surfs, nl, h, w,
             st) = arm_launch(which, post, env, R, dx, L, M, rs, clip,
                              strip)
            r = H.rs_spot(x0, y0, amp, ph0, p, q, surfs, BACK, xci, yci,
                          dx_out=dxo, n_out=nout, nl=nl)
            I = np.ascontiguousarray(r['I'])
            dig = hashlib.sha256(I.tobytes()).hexdigest()[:16]
            store[f"I_{o}_{which}"] = I
            store[f"ax_{o}_{which}"] = r['ax']
            cx, cy = centroid(I, r['ax'])
            px, py = peak_subpixel(I, r['ax'])
            store[f"c_{o}_{which}"] = np.array([cx, cy, px, py])
            print(f"  {which:6s} EE3 {r['ee3'] * 100:8.4f}  "
                  f"EE6 {r['ee6'] * 100:8.4f}  FWHM {r['fwhm'] * 1e6:6.3f}  "
                  f"centroid ({cx * 1e6:+.4f},{cy * 1e6:+.4f}) um  "
                  f"peak ({px * 1e6:+.4f},{py * 1e6:+.4f}) um  "
                  f"sha {dig}  [{time.time() - t0:.0f}s]", flush=True)
    fn = os.path.join(_HERE, f"_rc_readout_{tag}.npz")
    np.savez_compressed(fn, **store)
    print(f"\nsaved {fn}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
