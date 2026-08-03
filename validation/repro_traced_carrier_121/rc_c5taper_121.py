# D121 RESIDUAL CLOSURE -- the OTHER cos^2 taper.
#
# Niche C9 removed the ``cos^2`` band-limit taper from
# ``_sphere_parab_conversion`` because it broke an identity PAIR: the entrance
# conversion is undone by the element's own exact reference, so a taper leaves
# a spurious aliased quartic in the annulus where the beam still has power.
#
# ``_tilt_exactness_phase`` (niche C5) has the SAME structure -- the entrance
# ``+1`` call adds ``D * T`` and the element then removes the WHOLE of ``D``
# (its ``TiltedCarrier`` evaluates the exact eikonal, untapered), so what
# survives is ``exp(-i k D (1 - T))``.  Its own docstring records the taper as
# converged ("removing it ENTIRELY moves the result by 2.3e-5"), but that was
# measured at ONE order on the pre-C9 tree, and C9's lesson is precisely that
# a taper's onset MOVES as the rest of the chain improves.
#
# This does two things:
#   1. a per-call CENSUS of every ``_tilt_exactness_phase`` call the chain
#      makes, with its ``r_safe`` in beam radii -- the C9 census device;
#   2. the ablation: T == 1 on every call, EE3 re-scored area-exact.
#
# SCRIPT-SIDE ONLY -- ``carrier.py`` is not edited by this probe.
#
# usage:  ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' python rc_c5taper_121.py
import os
import sys
import time

import numpy as np

os.environ.setdefault('LUMEN_PIN', '0')

import _d121_common as C                                       # noqa: E402
import fc_instrument_121 as FI                                 # noqa: E402
import hybrid_localize_121 as H                                # noqa: E402
import rc_readout_121 as RD                                    # noqa: E402
import lumenairy.propagators.carrier as _CM                    # noqa: E402
from approx_common import Patch                                # noqa: E402
from lumenairy.raytrace import RayBundle, trace                # noqa: E402

LAM = C.LAM
BACK = 5.0e-3
_ORIG = _CM._tilt_exactness_phase
CENSUS = []


def _rsafe(wavelength, dx, dy, R, L, M):
    s = float(L) ** 2 + float(M) ** 2
    n_t = np.sqrt(s)
    if n_t == 0.0 or not np.isfinite(R) or R == 0.0:
        return np.inf
    a = 1.5 * n_t / (R * R)
    b = s / abs(R)
    c = wavelength / (2.0 * min(float(dx), float(dy)))
    return float((np.sqrt(b * b + 4.0 * a * c) - b) / (2.0 * a))


def _census_wrap(shape, dx, dy, wavelength, R, L, M, sign, centre=(0.0, 0.0)):
    CENSUS.append(dict(shape=tuple(shape[-2:]), dx=float(dx), R=float(R),
                       L=float(L), M=float(M), sign=int(sign),
                       centre=(float(centre[0]), float(centre[1])),
                       r_safe=_rsafe(wavelength, dx, dy, R, L, M)))
    return _ORIG(shape, dx, dy, wavelength, R, L, M, sign, centre=centre)


def _exact_wrap(shape, dx, dy, wavelength, R, L, M, sign, centre=(0.0, 0.0)):
    """``_tilt_exactness_phase`` with the cos^2 taper replaced by T == 1 --
    the C9 intervention, applied to the C5 term."""
    from lumenairy.elements._lens_traced import TILTED_CARRIER_EXACT_EIKONAL
    if not TILTED_CARRIER_EXACT_EIKONAL:
        return None
    L = float(L)
    M = float(M)
    if (L == 0.0 and M == 0.0) or not np.isfinite(R) or R == 0.0:
        return None
    Ny, Nx = int(shape[-2]), int(shape[-1])
    x = (np.arange(Nx, dtype=np.float64) - Nx / 2) * dx - float(centre[0])
    y = (np.arange(Ny, dtype=np.float64) - Ny / 2) * dy - float(centre[1])
    X, Y = x[None, :], y[:, None]
    sgn = 1.0 if R > 0 else -1.0
    s = L * L + M * M
    n_par = np.sqrt(1.0 - s)
    uu = X + R * L / n_par
    vv = Y + R * M / n_par
    r2 = X * X + Y * Y
    D = (sgn * (np.sqrt(uu * uu + vv * vv + R * R) - abs(R) / n_par)
         - sgn * (np.sqrt(r2 + R * R) - abs(R)) - (L * X + M * Y))
    k = 2.0 * np.pi / wavelength
    return np.exp(sign * 1j * k * D)


def run(post, env, R, dx, L, M, rs, patch, xci, yci):
    with Patch(patch):
        res, _w = FI.run_chain(post, env, R, dx, L, M, rs, 'off', 0)
        a = FI.chain_launch(res, L, M, 9999, 3.0, 1, post, BACK, 'exact')
    r = H.rs_spot(*a[:7], BACK, xci, yci, dx_out=0.4e-6, n_out=61, nl=a[7])
    I = np.ascontiguousarray(r['I'])
    cx, cy = RD.centroid(I, r['ax'])
    return (RD.ee(I, r['ax'], cx, cy, 3e-6, 'area') * 100,
            RD.ee(I, r['ax'], cx, cy, 6e-6, 'area') * 100, r['fwhm'] * 1e6, I)


def main():
    orders = os.environ.get(
        'ORDERS', '0,0 -1,0 -2,0 -3,0 -4,0 -4,-2').split()
    rn, rs = 1024, 4
    print(FI._provenance(), flush=True)
    _pre, post, _g, period = C.geometry()
    env, R, dx, _P = C.chain_a(n=rn, rs=rs)
    print(f"\n{'order':>8} {'T as shipped':>13} {'T == 1':>10} {'delta':>8} "
          f"{'EE6 ship':>9} {'EE6 T=1':>9} {'d EE6':>8} {'byte-equal':>10}",
          flush=True)
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
        t0 = time.time()
        CENSUS.clear()
        a = run(post, env, R, dx, L, M, rs,
                [(_CM, '_tilt_exactness_phase', _census_wrap)], xci, yci)
        cen = list(CENSUS)
        b = run(post, env, R, dx, L, M, rs,
                [(_CM, '_tilt_exactness_phase', _exact_wrap)], xci, yci)
        print(f"{o:>8} {a[0]:13.4f} {b[0]:10.4f} {b[0] - a[0]:+8.4f} "
              f"{a[1]:9.4f} {b[1]:9.4f} {b[1] - a[1]:+8.4f} "
              f"{str(np.array_equal(a[3], b[3])):>10}   "
              f"[{time.time() - t0:.0f}s]", flush=True)
        if cen:
            print(f"     per-call census ({len(cen)} live calls; "
                  f"r_safe in mm, onset = 0.75 r_safe):", flush=True)
            for i, cc in enumerate(cen):
                print(f"       {i:>2} sign{cc['sign']:+d} R={cc['R'] * 1e3:9.4f} mm "
                      f"dx={cc['dx'] * 1e6:7.3f} um  |L,M|="
                      f"{np.hypot(cc['L'], cc['M']) * 1e3:8.4f} mrad  "
                      f"r_safe={cc['r_safe'] * 1e3:10.4f} mm", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
