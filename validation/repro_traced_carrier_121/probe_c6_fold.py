# Niche C6 adversarial: WHY does a degree-6 residual-eikonal model manufacture
# a ghost lobe when a degree-4 one does not?
#
# Rebuilds the element's OWN launch (same launch_radius, same n_launch, same
# grad(W) + grad(a_fit) directions, same surfaces) with the library's own
# ``_fit_residual_eikonal``, and reports, per degree:
#
#   * the discrete Jacobian of the TRACED forward map on the launch lattice --
#     does it change sign (a genuine fold of the physical map)?
#   * the same for the CHEBYSHEV FIT of that map, which is what Newton
#     actually inverts -- a fit can fold where the samples do not.
#   * the launch-ray slope the model adds, by radius.
#
# usage:  ORD=-4,-2 DEG=2,4,6,8 python probe_c6_fold.py
import os
import sys
import warnings

import numpy as np

import wfe_probe_common as P
import _d121_common as C
import probe_c6_element as E6
import lumenairy.elements._lens_traced as LT
from lumenairy.elements._lens_traced import (TiltedCarrier, _Cheb2DEvaluator,
                                             _input_beam_amp_radius)
from lumenairy.raytrace import trace
from lumenairy.raytrace.trace import _make_bundle
from lumenairy.glass import get_glass_index

LAM = P.LAM
K0 = P.K0


def main():
    warnings.filterwarnings('ignore')
    m, n = (int(v) for v in os.environ.get('ORD', '-4,-2').split(','))
    rs = int(os.environ.get('RS', '4'))
    degs = [int(v) for v in os.environ.get('DEG', '0,2,4,6,8').split(',')]
    E_in, _Eo, carv, dx = E6.get_call(m, n, rs=rs)
    car = TiltedCarrier(*carv)
    N = E_in.shape[0]
    _pre, post, _g, _p = C.geometry()
    presc = post[-1]['prescription']
    surfs = P.element_surfaces(presc)
    ax = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(ax, ax)
    W_grid, _l, _mm = P.carrier_parts(car, Xg, Yg)
    w = float(_input_beam_amp_radius(E_in, dx, dx, centre=(car.x0, car.y0)))

    # the element's own launch geometry
    ap = presc.get('aperture_diameter')
    launch_radius = (0.5 * float(ap) * 1.50 if ap is not None
                     else 0.5 * N * dx)
    n_launch = max(8, int(2 * launch_radius / (dx * rs)))
    if n_launch % 2 == 0:
        n_launch += 1
    xs_in = np.linspace(-launch_radius, launch_radius, n_launch)
    Xs, Ys = np.meshgrid(xs_in, xs_in, indexing='ij')
    hx, hy = Xs.ravel(), Ys.ravel()
    _Wq, L0, M0 = LT._tilted_carrier_parts(car, hx, hy)
    n_exit = get_glass_index(surfs[-1].glass_after, LAM)
    # the beam-relative fit disc the element uses for the FORWARD-MAP fit
    fit_r = 2.0 * w
    print(f"order ({m},{n})  w {w*1e3:.4f} mm  launch_radius "
          f"{launch_radius*1e3:.4f} mm = {launch_radius/w:.2f} w  "
          f"n_launch {n_launch}  map-fit disc {fit_r*1e3:.4f} mm")
    print()
    hdr = (f"{'deg':>4} {'gadd rms':>9} {'gadd max':>9} {'r@max/w':>8} "
           f"{'detJ<0 samples':>15} {'detJ<0 in disc':>15} "
           f"{'FIT detJ<0 (launch sq)':>23} {'FIT detJ<0 (disc)':>18} "
           f"{'min|detJ|/med r<3w':>19} {'detJ<0 r<3w':>12}")
    print(hdr)
    print('-' * len(hdr))
    for deg in degs:
        if deg == 0:
            eik = None
            gL = np.zeros_like(hx)
            gM = np.zeros_like(hx)
        else:
            old = LT._REMAP_RESID_EIKONAL_DEGREE
            LT._REMAP_RESID_EIKONAL_DEGREE = deg
            try:
                # ray_fit_radius: the element's own ray-fit disc
                # (fit_radius_beam_factor=2.0), which since 2026-07-31 sets the
                # model's RADIAL FREEZE circle -- see
                # ``_REMAP_RESID_FREEZE_MARGIN``.  Omitting it rebuilds a
                # DIFFERENT launch congruence from the element's.
                eik = LT._fit_residual_eikonal(
                    E_in, W_grid, LAM, dx, dx, (car.x0, car.y0), w,
                    stride=rs, ray_fit_radius=min(2.0 * w, launch_radius))
            finally:
                LT._REMAP_RESID_EIKONAL_DEGREE = old
            gL, gM = eik.grad(hx, hy)
        rays = _make_bundle(x=hx.copy(), y=hy.copy(), L=(L0 + gL).copy(),
                            M=(M0 + gM).copy(), wavelength=LAM)
        fin = trace(rays, surfs, LAM, output_filter='last').image_rays
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(fin.alive & (np.abs(fin.N) > 1e-30),
                         -fin.z / fin.N, 0.0)
        xo = (np.asarray(fin.x) + np.asarray(fin.L) * t).reshape(n_launch, -1)
        yo = (np.asarray(fin.y) + np.asarray(fin.M) * t).reshape(n_launch, -1)
        opl = (np.asarray(fin.opd) + n_exit * t).reshape(n_launch, -1)
        opl = opl + _Wq.reshape(n_launch, -1)
        if eik is not None:
            opl = opl + eik.value(hx, hy).reshape(n_launch, -1)
        h = float(xs_in[1] - xs_in[0])
        jxx = np.gradient(xo, h, axis=0)
        jxy = np.gradient(xo, h, axis=1)
        jyx = np.gradient(yo, h, axis=0)
        jyy = np.gradient(yo, h, axis=1)
        det = jxx * jyy - jxy * jyx
        sgn = np.sign(np.nanmedian(det))
        bad = np.isfinite(det) & (det * sgn < 0)
        r2 = ((Xs - car.x0) ** 2 + (Ys - car.y0) ** 2)
        in_disc = r2 <= fit_r ** 2
        # the FITTED map (what Newton inverts): same Chebyshev order/weights
        _n_in = int(in_disc.sum())
        _n_out = int(in_disc.size) - _n_in
        wout = float(np.sqrt(1e-4 * _n_in / max(_n_out, 1)))
        fw = np.where(in_disc, 1.0, wout)
        Sx = _Cheb2DEvaluator(xs_in, xs_in, xo, order=10, weights=fw)
        Sy = _Cheb2DEvaluator(xs_in, xs_in, yo, order=10, weights=fw)
        _vx, fxx, fxy = Sx.ev_value_and_grad(hx, hy)
        _vy, fyx, fyy = Sy.ev_value_and_grad(hx, hy)
        fdet = (fxx * fyy - fxy * fyx)
        fs = np.sign(np.nanmedian(fdet))
        fbad = np.isfinite(fdet) & (fdet * fs < 0)
        gmag = np.hypot(gL, gM)
        i_mx = int(np.nanargmax(gmag))
        sup = r2 <= (3.0 * w) ** 2
        dsup = np.abs(det)[sup & np.isfinite(det)]
        rel = (float(dsup.min() / np.median(dsup)) if dsup.size else np.nan)
        print(f"{deg:>4} "
              f"{float(np.sqrt(np.nanmean(gmag[in_disc.ravel()] ** 2))):>9.2e} "
              f"{float(np.nanmax(gmag)):>9.2e} "
              f"{float(np.hypot(hx[i_mx] - car.x0, hy[i_mx] - car.y0) / w):>8.2f} "
              f"{int(bad.sum()):>15d} {int((bad & in_disc).sum()):>15d} "
              f"{int(fbad.sum()):>23d} "
              f"{int((fbad & in_disc.ravel()).sum()):>18d} "
              f"{rel:>19.4f} {int((bad & sup).sum()):>12d}")
    print()
    print("detJ<0 = samples whose discrete forward-map Jacobian has the "
          "MINORITY sign, i.e. the map folds there.")
    print("'FIT' columns evaluate the same test on the order-10 weighted "
          "Chebyshev fit Newton actually inverts.")


if __name__ == '__main__':
    sys.exit(main())
