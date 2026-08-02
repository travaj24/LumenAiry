# INDEPENDENT exact-ray oracle for design 121's per-order spot quality.
# (SCOPE_TILTED_COARSE_LEG_TRANSPORT_2026_07_30, headline test.)
#
# THE QUESTION.  ``propagate_traced_carrier_chain_multi`` reports design 121's
# DOE fan losing ~27 EE3 points between the (0,0) order and the (-4,-2) order
# (46.1 + 23.0 mrad).  Real optics DO degrade off axis, so before hunting a
# library defect one has to know whether the loss is PHYSICAL.
#
# THE METHOD.  Nothing here touches a propagator, a carrier, an FFT, a window
# factor or a Newton inversion.  It is:
#
#   1. launch a uniform grid of rays from the DOE plane, with the EXACT local
#      wavefront normal of the field that actually arrives there (an exact
#      sphere of radius R_doe, plus the order's grating tilt (L, M), plus --
#      optionally -- the measured residual envelope-phase gradient),
#   2. trace them through the post-DOE surfaces with ``lumenairy.raytrace``
#      (exact skew ray trace, Zemax-validated) to a flat EXIT REFERENCE plane
#      a chosen distance short of the MSoP plane, keeping the accumulated OPL,
#   3. evaluate the RAYLEIGH-SOMMERFELD (first Rayleigh-Sommerfeld, exact
#      spherical-wave kernel, no Fresnel/Fraunhofer expansion) integral from
#      that plane to the image plane, with an ENERGY-CONSERVING ray-density
#      amplitude: the flux |E_doe|^2 * cos(theta_launch) * h^2 of each launch
#      cell is spread over the cell's traced exit-plane area, so
#
#         E_exit,j * dA_exit,j = |E_doe,j| * sqrt(N0_j * |J_j| / N_j) * h^2,
#         J_j = |d(x_exit, y_exit) / d(x_launch, y_launch)|,
#
#      i.e. the integrand needs the exit-plane area Jacobian, taken by central
#      differences on the launch grid.
#
# WHAT WOULD MAKE THIS A MEASUREMENT ARTEFACT, and what is checked:
#   * launch-grid undersampling -- swept (``NL``); the phase step between
#     adjacent rays at the FARTHEST evaluated image point is printed as
#     ``max d(phase)/2pi``, and must be << 0.5 for the sum to be a quadrature.
#   * launch-aperture truncation -- swept (``CLIP``, in envelope amplitude
#     radii).
#   * exit-reference-plane placement -- swept (``BACK``); the RS integral is
#     exact for any placement, so a dependence on it is a bug.
#   * image-grid pixelation of the EE circles -- swept (``DXO``), and the EE
#     is reported about the CENTROID and about the PEAK.
#   * vignetting / dead rays -- counted and reported.
#
# Env knobs: ORD='m,n;m,n;...', NL, CLIP, BACK (mm), DXO (um), NOUT,
#            PHASE=0/1 (carry the chain-A residual envelope phase), RN, RS.
import os
import sys
import time

import numpy as np

import _d121_common as C
from lumenairy.raytrace import RayBundle, trace


def _bilinear(field, dx, xq, yq):
    """Bilinear sample of a centred (N, N) grid at scattered (xq, yq)."""
    n = field.shape[0]
    fx = xq / dx + n / 2.0
    fy = yq / dx + n / 2.0
    i0 = np.floor(fx).astype(int)
    j0 = np.floor(fy).astype(int)
    tx = fx - i0
    ty = fy - j0
    ok = (i0 >= 0) & (i0 < n - 1) & (j0 >= 0) & (j0 < n - 1)
    i0c = np.clip(i0, 0, n - 2)
    j0c = np.clip(j0, 0, n - 2)
    v = ((1 - tx) * (1 - ty) * field[j0c, i0c]
         + tx * (1 - ty) * field[j0c, i0c + 1]
         + (1 - tx) * ty * field[j0c + 1, i0c]
         + tx * ty * field[j0c + 1, i0c + 1])
    return np.where(ok, v, 0.0)


def _phase_gradient(env, dx):
    """Aliasing-free local phase gradient of a SMOOTH envelope [rad/m].

    Wrapped nearest-neighbour differences (the D7 aliasing-free recipe): the
    per-pixel step must be << pi or the derivative is meaningless.  Returns
    ``(gx, gy, max_step)`` with ``max_step`` in radians over the bright core."""
    ph = np.angle(env)
    d = np.abs(env)
    bright = d > 0.05 * d.max()
    def _w(a):
        return (a + np.pi) % (2 * np.pi) - np.pi
    gx = np.zeros_like(ph)
    gy = np.zeros_like(ph)
    gx[:, 1:-1] = _w(ph[:, 2:] - ph[:, :-2]) / (2 * dx)
    gy[1:-1, :] = _w(ph[2:, :] - ph[:-2, :]) / (2 * dx)
    step = np.abs(_w(ph[:, 1:] - ph[:, :-1]))[bright[:, 1:]]
    return gx, gy, (float(step.max()) if step.size else 0.0)


def oracle_spot(env_doe, R_doe, dx_doe, groups_post, L, M, *,
                n_launch=161, clip=3.0, back=5.0e-3, dx_out=0.1e-6,
                n_out=261, carry_phase=False, trailing=C.TRAILING,
                verbose=True, prefix='', flux=True):
    """Exact-ray + Rayleigh-Sommerfeld PSF for one DOE order.

    Returns a dict with the image intensity and the spot metrics.

    ABSOLUTE ENERGY (fixed 2026-08-01, ORACLE_ENERGY_AND_D6_HALO).  Until then
    the RS kernel below omitted its ``1/(i lambda)`` prefactor, so ``I`` was a
    field intensity times ``1/lambda^2`` = 5.827e11 and NO absolute
    ``P_out/P_in`` could be read off the oracle (only ratios between its own
    runs).  The prefactor is now applied, which makes ``I`` a physical
    irradiance for a launch field ``env_doe`` in sqrt(W)/m, and the returned
    dict carries the absolute bookkeeping:

    ``launch_power``   SUM |E|^2 cos(theta) h^2 -- the flux the launch lattice
                       carries.  This ALREADY included the launch-cell area
                       ``h^2``; the claim in POP_CROSSCHECK_121 S9.2 that it
                       does not is wrong, and the double-count is in that
                       audit's own harness (``pop_ours_121.py``), fixed there.
    ``P_window_sq``    SUM |E_img|^2 dx_out^2 -- by Parseval this is
                       INT |A(k)|^2 d2k, i.e. it OMITS the obliquity kz/k.
    ``P_window_flux``  (1/k) Im INT E* dE/dz dA -- the TRUE z-directed power,
                       INT |A(k)|^2 (kz/k) d2k.  This is the one that must come
                       back as the launched flux; ``P_window_sq`` overshoots it
                       by 1/<cos theta> over the beam's own cone, 2.9 % at this
                       design's exit NA.  Validated against an unaberrated
                       converging sphere in ``oracleE_rs_control.py``:
                       P_flux/P_in = 1 to 1.2e-7.

    ``flux=False`` skips the second accumulation (about 40 % of the runtime)
    and leaves ``P_window_flux`` NaN.  EE / FWHM / centroid are unaffected by
    the prefactor to within double-precision rounding (measured 4.4e-16
    relative, ``oracleE_absolute_121.py --null``)."""
    lam = C.LAM
    k = 2.0 * np.pi / lam
    from lumenairy.propagators.carrier import _envelope_amp_radius
    w = _envelope_amp_radius(env_doe, dx_doe, dx_doe)

    # --- launch grid -------------------------------------------------------
    h = 2.0 * clip * w / (n_launch - 1)
    t = (np.arange(n_launch) - (n_launch - 1) / 2.0) * h
    X0, Y0 = np.meshgrid(t, t)
    x0 = X0.ravel()
    y0 = Y0.ravel()
    amp = _bilinear(np.abs(env_doe), dx_doe, x0, y0)

    # --- launch directions: EXACT sphere normal + the order's tilt ---------
    rr = np.hypot(x0, y0)
    den = np.sqrt(R_doe ** 2 + rr ** 2)
    p = x0 / den * np.sign(R_doe) + L
    q = y0 / den * np.sign(R_doe) + M
    # LAUNCH-PLANE OPL.  ``trace`` starts every ray at opd = 0 on z = 0, but
    # the CONSTANT-PHASE surface at launch is the carrier sphere tilted by the
    # DOE order, not that plane.  The tilt piston k(Lx+My) reaches 562 waves at
    # the beam edge for m = -4; omitting it makes the wavefront look as if it
    # converged on the optical axis.
    ph0 = (2.0 * np.pi / lam) * (L * x0 + M * y0
                                 + np.sign(R_doe) * (den - abs(R_doe)))
    max_step = 0.0
    if carry_phase:
        gx, gy, max_step = _phase_gradient(env_doe, dx_doe)
        p = p + _bilinear(gx, dx_doe, x0, y0) / k
        q = q + _bilinear(gy, dx_doe, x0, y0) / k
        # the residual envelope phase itself is an extra OPL offset at launch
        ph = np.unwrap(np.unwrap(np.angle(env_doe), axis=1), axis=0)
        ph0 = ph0 + _bilinear(ph, dx_doe, x0, y0)
    n0 = np.sqrt(np.maximum(1.0 - p * p - q * q, 0.0))

    # --- exact skew trace to the EXIT REFERENCE plane ----------------------
    surfs = C.post_surfaces(groups_post, trailing=trailing, back_off=back)
    rb = RayBundle(x=x0.copy(), y=y0.copy(), z=np.zeros_like(x0),
                   L=p.copy(), M=q.copy(), N=n0.copy(), wavelength=lam,
                   alive=np.ones(x0.size, dtype=bool),
                   opd=np.zeros_like(x0))
    tr = trace(rb, surfs, lam, output_filter='last')
    ir = tr.image_rays
    alive = np.asarray(ir.alive, dtype=bool)
    xe = np.asarray(ir.x)
    ye = np.asarray(ir.y)
    opd = np.asarray(ir.opd)
    Ne = np.asarray(ir.N)

    # --- exit-plane area Jacobian (central differences on the LAUNCH grid) --
    XE = xe.reshape(n_launch, n_launch)
    YE = ye.reshape(n_launch, n_launch)
    dxe_dx = np.gradient(XE, h, axis=1)
    dxe_dy = np.gradient(XE, h, axis=0)
    dye_dx = np.gradient(YE, h, axis=1)
    dye_dy = np.gradient(YE, h, axis=0)
    J = np.abs(dxe_dx * dye_dy - dxe_dy * dye_dx).ravel()

    good = alive & (amp > 0) & np.isfinite(opd) & (J > 0) & (Ne > 0)
    p_launch = float(np.sum((amp ** 2 * n0)[amp > 0])) * h * h
    p_live = float(np.sum((amp ** 2 * n0)[good])) * h * h
    Wj = (amp[good] * np.sqrt(n0[good] * J[good] / Ne[good]) * h * h)
    ph_exit = k * opd[good] + ph0[good]
    xe = xe[good]
    ye = ye[good]

    # --- Rayleigh-Sommerfeld I to the image plane --------------------------
    # centre the image grid on the geometric CHIEF ray (the (0,0)-launch ray)
    ic = np.argmin(x0 ** 2 + y0 ** 2)
    cr = trace(RayBundle(x=np.array([0.0]), y=np.array([0.0]),
                         z=np.array([0.0]), L=np.array([L]), M=np.array([M]),
                         N=np.array([np.sqrt(1 - L * L - M * M)]),
                         wavelength=lam, alive=np.ones(1, bool),
                         opd=np.zeros(1)),
               C.post_surfaces(groups_post, trailing=trailing), lam,
               output_filter='last').image_rays
    xc, yc = float(cr.x[0]), float(cr.y[0])
    ax = (np.arange(n_out) - (n_out - 1) / 2.0) * dx_out
    PX = xc + ax
    PY = yc + ax

    # SAMPLING ADEQUACY.  The relevant criterion is NOT the exit-plane fringe
    # pitch (the integrand is stationary-phase: ph_exit and k*rho very nearly
    # cancel).  It is the largest step of the ACTUAL integrand phase
    # ``ph_exit + k|P - r|`` between adjacent LAUNCH rays, evaluated at the
    # FARTHEST image point the metrics use -- measure it, do not estimate it.
    na_eff = float(np.sqrt(np.maximum(1.0 - Ne[good] ** 2, 0.0)).max())
    _Pw = (xc + ax[0], yc + ax[0])          # the corner of the image patch
    _phi = np.full(x0.size, np.nan)
    _phi[good] = ph_exit + k * np.sqrt((_Pw[0] - xe) ** 2
                                       + (_Pw[1] - ye) ** 2 + back ** 2)
    _PH = _phi.reshape(n_launch, n_launch)
    _st = np.concatenate([np.abs(np.diff(_PH, axis=0)).ravel(),
                          np.abs(np.diff(_PH, axis=1)).ravel()])
    _st = _st[np.isfinite(_st)]
    step_frac = float(np.nanmax(_st)) / (2 * np.pi) if _st.size else np.nan
    # amplitude-weighted version: steps where the integrand carries no power
    # are irrelevant.
    _wg = np.zeros(x0.size)
    _wg[good] = Wj
    _WG = _wg.reshape(n_launch, n_launch)
    _wst = np.concatenate([
        np.abs(np.diff(_PH, axis=0)).ravel()
        * np.minimum(_WG[1:], _WG[:-1]).ravel() / max(Wj.max(), 1e-300),
        np.abs(np.diff(_PH, axis=1)).ravel()
        * np.minimum(_WG[:, 1:], _WG[:, :-1]).ravel() / max(Wj.max(), 1e-300)])
    _wst = _wst[np.isfinite(_wst)]
    step_w = float(np.nanmax(_wst)) / (2 * np.pi) if _wst.size else np.nan
    # p99.9 as well as the max: a single max is dominated by a handful of rays
    # at the vignetting edge / ray-map fold where sqrt(J) spikes.  This is the
    # statistic the campaign's other instruments quote (hybrid_localize's
    # ``step_w``), so the two are directly comparable.
    step_w999 = (float(np.percentile(_wst, 99.9)) / (2 * np.pi)
                 if _wst.size else np.nan)

    I = np.zeros((n_out, n_out))
    E = np.zeros((n_out, n_out), dtype=np.complex128)
    Ez = np.zeros((n_out, n_out), dtype=np.complex128)
    # first Rayleigh-Sommerfeld prefactor.  |1/(i lambda)| = 1/lambda; the
    # phase of the constant cancels out of |E|^2 and out of Im(E* dE/dz).
    c_rs = 1.0 / (1j * lam)
    chunk = max(1, int(6e7 // max(xe.size, 1)))
    t0 = time.time()
    for j0 in range(0, n_out, chunk):
        j1 = min(j0 + chunk, n_out)
        dyv = PY[j0:j1, None] - ye[None, :]
        for i, px in enumerate(PX):
            dxv = px - xe
            rho = np.sqrt(dxv[None, :] ** 2 + dyv ** 2 + back ** 2)
            ker = np.exp(1j * (ph_exit[None, :] + k * rho))
            E[j0:j1, i] = c_rs * np.sum(
                (Wj * back) / (rho * rho) * ker, axis=1)
            if flux:
                # d/dz of  z/rho^2 exp(i k rho)  at fixed transverse offset
                Ez[j0:j1, i] = c_rs * np.sum(
                    Wj * (1.0 / (rho * rho)
                          - 2.0 * back * back / rho ** 4
                          + 1j * k * back * back / rho ** 3) * ker, axis=1)
    I = np.abs(E) ** 2
    dt = time.time() - t0
    p_win_sq = float(I.sum()) * dx_out * dx_out
    p_win_flux = (float(np.imag(np.conj(E) * Ez).sum()) / k * dx_out * dx_out
                  if flux else float('nan'))

    # --- metrics ------------------------------------------------------------
    tot = float(I.sum())
    Xg, Yg = np.meshgrid(ax, ax)
    cx = float((I * Xg).sum() / tot)
    cy = float((I * Yg).sum() / tot)
    out = {'I': I, 'E': E, 'ax': ax, 'centre': (xc, yc), 'centroid': (cx, cy),
           'time': dt, 'w_doe': w, 'launch_power': p_launch,
           'live_power': p_live,
           'live_frac': p_live / p_launch if p_launch else 0.0,
           'n_dead': int((~good).sum()), 'ray_step_frac': step_frac,
           'ray_step_weighted': step_w,
           'ray_step_w_p999': step_w999,
           'P_window_sq': p_win_sq, 'P_window_flux': p_win_flux,
           'P_ratio_sq': p_win_sq / p_launch if p_launch else np.nan,
           'P_ratio_flux': p_win_flux / p_launch if p_launch else np.nan,
           'P_ratio_flux_live': (p_win_flux / p_live if p_live else np.nan),
           'phase_step': max_step, 'na_eff': na_eff}
    for tag, _c in (('cen', (cx, cy)), ('peak', None)):
        if _c is None:
            iy, ix = np.unravel_index(np.argmax(I), I.shape)
            ox, oy = float(ax[ix]), float(ax[iy])
        else:
            ox, oy = _c
        r = np.hypot(Xg - ox, Yg - oy)
        for rad in (3e-6, 6e-6, 12e-6):
            out[f'ee{int(rad * 1e6)}_{tag}'] = float(I[r <= rad].sum()) / tot
        # FWHM from an azimuthal profile about that point
        nb = int(min(n_out // 2, 12e-6 / dx_out))
        ring = np.clip((r / dx_out).astype(int), 0, nb)
        s = np.bincount(ring.ravel(), weights=I.ravel(), minlength=nb + 1)
        cn = np.bincount(ring.ravel(), minlength=nb + 1)
        prof = (s[:nb] / np.maximum(cn[:nb], 1))
        prof = prof / prof[0]
        rb_ = (np.arange(nb) + 0.5) * dx_out
        idx = np.where(prof < 0.5)[0]
        if len(idx) and idx[0] > 0:
            i1 = idx[0]
            f = (prof[i1 - 1] - 0.5) / (prof[i1 - 1] - prof[i1])
            out[f'fwhm_{tag}'] = 2 * (rb_[i1 - 1] + f * dx_out)
        else:
            out[f'fwhm_{tag}'] = np.nan
    if verbose:
        print(f"{prefix}L,M = {L * 1e3:+8.3f},{M * 1e3:+8.3f} mrad  "
              f"NL={n_launch} clip={clip} back={back * 1e3:.2f}mm "
              f"dxo={dx_out * 1e6:.3f}um N={n_out}  ({dt:.1f}s)")
        print(f"{prefix}  chief (x,y) = ({xc * 1e6:+.3f},{yc * 1e6:+.3f}) um, "
              f"centroid offset ({cx * 1e6:+.3f},{cy * 1e6:+.3f}) um")
        print(f"{prefix}  live power {out['live_frac'] * 100:.4f} %, dead rays "
              f"{out['n_dead']}, exit NA {na_eff:.4f}")
        print(f"{prefix}  ABSOLUTE  P_in(launch flux) = {p_launch:.6e}   "
              f"P_win(flux) = {p_win_flux:.6e}   P_win(|E|^2) = {p_win_sq:.6e}")
        print(f"{prefix}            P_win(flux)/P_in = "
              f"{out['P_ratio_flux'] * 100:.4f} %   "
              f"P_win(|E|^2)/P_in = {out['P_ratio_sq'] * 100:.4f} %  "
              f"(the second omits the image-plane obliquity kz/k)")
        print(f"{prefix}  integrand phase step between adjacent launch rays "
              f"at the patch corner: {step_frac:.4f} cycles (max), "
              f"{step_w:.4f} amplitude-weighted "
              f"({'OK' if step_w < 0.25 else 'UNDERSAMPLED'})")
        print(f"{prefix}  FWHM {out['fwhm_cen'] * 1e6:6.3f} um   "
              f"EE3 {out['ee3_cen'] * 100:6.2f}  EE6 {out['ee6_cen'] * 100:6.2f}"
              f"  EE12 {out['ee12_cen'] * 100:6.2f}   (about centroid)")
        print(f"{prefix}  FWHM {out['fwhm_peak'] * 1e6:6.3f} um   "
              f"EE3 {out['ee3_peak'] * 100:6.2f}  "
              f"EE6 {out['ee6_peak'] * 100:6.2f}"
              f"  EE12 {out['ee12_peak'] * 100:6.2f}   (about peak)")
    return out


def main():
    orders = os.environ.get('ORD', '0,0;-1,0;-4,0;-4,-2')
    want = [tuple(int(v) for v in s.split(',')) for s in orders.split(';') if s]
    nl = int(os.environ.get('NL', '161'))
    clip = float(os.environ.get('CLIP', '3.0'))
    back = float(os.environ.get('BACK', '5.0')) * 1e-3
    dxo = float(os.environ.get('DXO', '0.1')) * 1e-6
    nout = int(os.environ.get('NOUT', '261'))
    carry = os.environ.get('PHASE', '0') == '1'
    rn = int(os.environ.get('RN', '1024'))
    rs = int(os.environ.get('RS', '4'))

    _pre, post, _g, period = C.geometry()
    env, R, dx, _P = C.chain_a(n=rn, rs=rs)
    print(f"DOE plane: R = {R * 1e3:.4f} mm, dx = {dx * 1e6:.4f} um, "
          f"N = {env.shape[0]}, carry_phase={carry}")
    rows = []
    for (m, n) in want:
        L = m * C.LAM / period
        M = n * C.LAM / period
        r = oracle_spot(env, R, dx, post, L, M, n_launch=nl, clip=clip,
                        back=back, dx_out=dxo, n_out=nout, carry_phase=carry,
                        prefix=f"({m:+d},{n:+d}) ")
        rows.append(((m, n), r))
        print()
    print("ORACLE SUMMARY (about centroid)")
    print("  order      FWHM um   EE3 %   EE6 %   EE12 %")
    for (mn, r) in rows:
        print(f"  ({mn[0]:+d},{mn[1]:+d})   {r['fwhm_cen'] * 1e6:7.3f}  "
              f"{r['ee3_cen'] * 100:6.2f}  {r['ee6_cen'] * 100:6.2f}  "
              f"{r['ee12_cen'] * 100:6.2f}")


if __name__ == '__main__':
    sys.exit(main())
