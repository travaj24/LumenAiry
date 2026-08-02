# PURE GEOMETRY / WAVEFRONT check of design 121's post-DOE relay per DOE order.
# No propagator, no carrier, no FFT, no diffraction integral: just
# ``lumenairy.raytrace`` (exact skew trace, Zemax-validated).
#
# Reports, per order:
#   * the geometric spot rms radius at the MSoP plane,
#   * the rms and PV WAVEFRONT ERROR over the beam, measured as the OPL to a
#     best-fit reference sphere centred on the diffraction focus, weighted by
#     the actual |E|^2 illumination at the DOE,
#   * the Marechal Strehl estimate exp(-(2 pi rms_waves)^2),
#   * the surviving (unvignetted) power fraction.
#
# Sampling: the launch grid is swept (NL) and the numbers must be stationary.
# The wavefront error is a POINTWISE quantity (no quadrature), so it converges
# with far fewer rays than any diffraction integral.
import os
import sys

import numpy as np

import _d121_common as C
from lumenairy.raytrace import RayBundle, trace


def _bundle(x, y, L, M, lam):
    n = np.sqrt(np.maximum(1.0 - L * L - M * M, 0.0))
    return RayBundle(x=np.asarray(x, float).copy(), y=np.asarray(y, float).copy(),
                     z=np.zeros_like(np.asarray(x, float)),
                     L=np.asarray(L, float).copy(), M=np.asarray(M, float).copy(),
                     N=n, wavelength=lam,
                     alive=np.ones(np.size(x), bool),
                     opd=np.zeros(np.size(x)))


def run(env, R, dx, post, L, M, nl=81, clip=2.6, back=5e-3, label=''):
    lam = C.LAM
    from lumenairy.propagators.carrier import _envelope_amp_radius
    w = _envelope_amp_radius(env, dx, dx)
    h = 2 * clip * w / (nl - 1)
    t = (np.arange(nl) - (nl - 1) / 2.0) * h
    X, Y = np.meshgrid(t, t)
    x0, y0 = X.ravel(), Y.ravel()
    from exact_ray_oracle_121 import _bilinear
    amp = _bilinear(np.abs(env), dx, x0, y0)
    den = np.sqrt(R ** 2 + x0 ** 2 + y0 ** 2)
    p = x0 / den * np.sign(R) + L
    q = y0 / den * np.sign(R) + M
    # LAUNCH-PLANE OPL OFFSET.  ``trace`` starts every ray at opd = 0 on the
    # z = 0 plane, but the incident CONSTANT-PHASE surface is not that plane:
    # it is the carrier sphere TILTED by the DOE order.  Omitting the tilt
    # piston k(L x + M y) -- 562 waves at the beam edge for m = -4 -- makes the
    # wavefront appear to converge on the optical axis instead of on the
    # order's own image point.  (First cut of this script did exactly that and
    # read a 27.8-wave "WFE" for a 0.40 um geometric spot.)
    opl0 = L * x0 + M * y0 + np.sign(R) * (den - abs(R))

    # (a) image-plane geometric spot
    surf_img = C.post_surfaces(post)
    ti = trace(_bundle(x0, y0, p, q, lam), surf_img, lam, output_filter='last')
    ri = ti.image_rays
    ok = np.asarray(ri.alive, bool) & (amp > 0)
    wgt = (amp ** 2)[ok]
    xi = np.asarray(ri.x)[ok]
    yi = np.asarray(ri.y)[ok]
    cx = float((wgt * xi).sum() / wgt.sum())
    cy = float((wgt * yi).sum() / wgt.sum())
    rms_geo = float(np.sqrt((wgt * ((xi - cx) ** 2 + (yi - cy) ** 2)).sum()
                            / wgt.sum()))
    p_all = float((amp ** 2).sum())
    p_ok = float((amp ** 2)[ok].sum())

    # (b) wavefront error on an exit reference plane, about the DIFFRACTION
    #     focus (the point that minimises the illumination-weighted variance of
    #     opd + |P - r|).  Solve for P by Nelder-Mead on 3 dof (x, y, z).
    surf_ref = C.post_surfaces(post, back_off=back)
    tr = trace(_bundle(x0, y0, p, q, lam), surf_ref, lam, output_filter='last')
    rr = tr.image_rays
    okr = np.asarray(rr.alive, bool) & (amp > 0)
    xe = np.asarray(rr.x)[okr]
    ye = np.asarray(rr.y)[okr]
    op = np.asarray(rr.opd)[okr] + opl0[okr]
    wr = (amp ** 2)[okr]
    wr = wr / wr.sum()

    def _wfe(P):
        """Illumination-weighted rms/PV of (opd + |P - r|) about its mean."""
        rho = np.sqrt((P[0] - xe) ** 2 + (P[1] - ye) ** 2
                      + (back + P[2]) ** 2)
        s = op + rho
        s = s - float((wr * s).sum())
        return s, rho

    # (b1) reference point FIXED at the geometric centroid on the image plane
    P_fix = np.array([cx, cy, 0.0])
    s_fix, _ = _wfe(P_fix)
    rms_fix = float(np.sqrt((wr * s_fix ** 2).sum())) / lam

    # (b2) BEST-FOCUS reference: Gauss-Newton on the 3-dof point.  ds/dP is the
    # unit vector from the exit point to P, so each step is a weighted linear
    # least-squares removal of piston + the three direction-cosine modes.  A
    # scale-free, well-conditioned update -- unlike a Nelder-Mead in metres,
    # which walked off to a 1.7-wave "solution" inconsistent with a 0.30 um
    # geometric spot (the first cut of this script; kept as a cautionary note).
    P = P_fix.copy()
    for _it in range(40):
        s, rho = _wfe(P)
        U = np.stack([(P[0] - xe) / rho, (P[1] - ye) / rho,
                      (back + P[2]) / rho, np.ones_like(rho)], axis=1)
        A = (U * wr[:, None]).T @ U
        b = (U * wr[:, None]).T @ (-s)
        try:
            d = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
        P = P + d[:3]
        if np.max(np.abs(d[:3])) < 1e-12:
            break
    s, rho = _wfe(P)
    rms_w = float(np.sqrt((wr * s ** 2).sum())) / lam
    s = s / lam
    core = wr > 0.01 * wr.max()
    pv = float(s[core].max() - s[core].min())
    strehl = float(np.exp(-(2 * np.pi * rms_w) ** 2))
    res = type('R', (), {'x': np.array([P[0], P[1], P[2]])})()
    print(f"{label}  WFE rms about the FIXED geometric centroid: "
          f"{rms_fix:.5f} waves")
    print(f"{label}L,M = {L * 1e3:+7.3f},{M * 1e3:+7.3f} mrad  NL={nl} "
          f"clip={clip}")
    print(f"{label}  centroid ({cx * 1e6:+11.3f},{cy * 1e6:+11.3f}) um   "
          f"geo rms radius {rms_geo * 1e6:7.4f} um")
    print(f"{label}  unvignetted power {p_ok / p_all * 100:.6f} %   live rays "
          f"{int(ok.sum())}/{ok.size}")
    print(f"{label}  WFE rms {rms_w:.5f} waves   PV(core) {pv:.4f} waves   "
          f"Marechal Strehl {strehl:.4f}")
    print(f"{label}  best focus shift dz {res.x[2] * 1e6:+.3f} um, "
          f"dx {(res.x[0] - cx) * 1e6:+.3f} um, dy {(res.x[1] - cy) * 1e6:+.3f}"
          f" um")
    return dict(rms_geo=rms_geo, rms_w=rms_w, rms_fix=rms_fix, pv=pv,
                strehl=strehl, centroid=(cx, cy), focus=res.x,
                live=p_ok / p_all)


def main():
    orders = os.environ.get('ORD', '0,0;-1,0;-4,0;-4,-2')
    want = [tuple(int(v) for v in s.split(',')) for s in orders.split(';') if s]
    nls = [int(v) for v in os.environ.get('NL', '81').split(',')]
    clip = float(os.environ.get('CLIP', '2.6'))
    back = float(os.environ.get('BACK', '5.0')) * 1e-3
    _pre, post, _g, period = C.geometry()
    env, R, dx, _P = C.chain_a(n=int(os.environ.get('RN', '1024')))
    print(f"DOE plane: R = {R * 1e3:.4f} mm dx = {dx * 1e6:.4f} um "
          f"N = {env.shape[0]}")
    rows = {}
    for nl in nls:
        for (m, n) in want:
            r = run(env, R, dx, post, m * C.LAM / period, n * C.LAM / period,
                    nl=nl, clip=clip, back=back,
                    label=f"({m:+d},{n:+d}) ")
            rows[(m, n, nl)] = r
            print()
    print("SUMMARY  order   NL   geo-rms um  WFE@centroid  WFE@focus  "
          "Strehl   live%")
    for (m, n, nl), r in rows.items():
        print(f"  ({m:+d},{n:+d})  {nl:5d}   {r['rms_geo'] * 1e6:8.4f}   "
              f"{r['rms_fix']:10.5f}  {r['rms_w']:9.5f}  {r['strehl']:6.4f}  "
              f"{r['live'] * 100:9.5f}")


if __name__ == '__main__':
    sys.exit(main())
