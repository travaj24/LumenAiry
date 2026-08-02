# ABSOLUTE-ENERGY control for the exact-ray + Rayleigh-Sommerfeld oracle.
# (ORACLE_ENERGY_AND_D6_HALO_2026_08_01, task A.)
#
# THE QUESTION.  ``exact_ray_oracle_121.oracle_spot`` builds an integrand
#
#     E(P) = SUM_j  W_j * back / rho_j^2 * exp(i (ph_j + k rho_j)),
#     W_j  = |E_j| sqrt(n0_j J_j / N_j) h^2      (= E_exit,j * dA_exit,j)
#
# which is the first Rayleigh-Sommerfeld integral WITHOUT its ``1/(i lambda)``
# prefactor.  Ratios (EE, FWHM, relative profiles) are untouched by that; any
# ABSOLUTE statement -- P_out / P_in -- is off by 1/lambda^2 = 5.83e11.
#
# THIS SCRIPT IS THE CONTROL, and it is deliberately free of design 121: an
# UNABERRATED CONVERGING SPHERE in vacuum, whose answer is known exactly
# (P_out/P_in = 1), driven through the SAME machinery the oracle uses --
# ``lumenairy.raytrace.trace`` for the transport, central-difference exit
# Jacobian, ray-density amplitude, the same RS quadrature.  Nothing analytic is
# substituted for a step the oracle performs numerically.
#
# TWO DISTINCT POWERS AT THE IMAGE PLANE, and the difference is not a bug:
#   * ``P_sq``   = SUM |U|^2 dA          -- the naive one, and by Parseval this
#                  equals INT |A(k)|^2 d^2k, i.e. it OMITS the obliquity kz/k;
#   * ``P_flux`` = (1/k) Im INT U* dU/dz dA  -- the true z-directed power,
#                  = INT |A(k)|^2 (kz/k) d^2k.
# The launch-side ``P_in = SUM |E|^2 cos(theta) h^2`` IS a flux, so ``P_flux``
# is the one that must come back as 1.0000; ``P_sq`` overshoots by 1/<cos>,
# which at design 121's exit NA is a couple of percent -- measured below, not
# asserted.
#
# Env knobs: NA (list), NL (list), F_MM, DXO_UM, NOUT, CLIP, BACK_MM.
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))

from lumenairy.raytrace import RayBundle, Surface, trace  # noqa: E402

LAM = 1.31e-6
K0 = 2.0 * np.pi / LAM


def _free_surfaces(dist):
    """Two flat air surfaces separated by ``dist`` -- the oracle's transport
    machinery with the optics removed."""
    return [Surface(radius=np.inf, conic=0.0, semi_diameter=np.inf,
                    glass_before='air', glass_after='air', is_mirror=False,
                    thickness=float(dist), label='launch'),
            Surface(radius=np.inf, conic=0.0, semi_diameter=np.inf,
                    glass_before='air', glass_after='air', is_mirror=False,
                    thickness=0.0, label='img')]


def rs_control(w=1.0e-3, f=3.0e-3, n_launch=161, clip=3.0, back=0.5e-3,
               dx_out=0.1e-6, n_out=201, prefactor=True, flux=True):
    """One converging-sphere case.  Returns the power bookkeeping.

    ``w``     launch Gaussian amplitude radius (E = exp(-r^2/w^2)),
    ``f``     distance from the launch plane to the geometric focus,
    ``back``  where the RS exit reference plane sits, short of the focus.
    """
    h = 2.0 * clip * w / (n_launch - 1)
    t = (np.arange(n_launch) - (n_launch - 1) / 2.0) * h
    X0, Y0 = np.meshgrid(t, t)
    x0 = X0.ravel()
    y0 = Y0.ravel()
    amp = np.exp(-(x0 ** 2 + y0 ** 2) / (w * w))

    # EXACT converging sphere: every ray aimed at (0, 0, f).
    den = np.sqrt(x0 ** 2 + y0 ** 2 + f * f)
    p = -x0 / den
    q = -y0 / den
    n0 = f / den
    # launch OPL so that every ray is in phase AT THE FOCUS
    ph0 = -K0 * den

    surfs = _free_surfaces(f - back)
    rb = RayBundle(x=x0.copy(), y=y0.copy(), z=np.zeros_like(x0),
                   L=p.copy(), M=q.copy(), N=n0.copy(), wavelength=LAM,
                   alive=np.ones(x0.size, dtype=bool), opd=np.zeros_like(x0))
    ir = trace(rb, surfs, LAM, output_filter='last').image_rays
    alive = np.asarray(ir.alive, dtype=bool)
    xe = np.asarray(ir.x)
    ye = np.asarray(ir.y)
    opd = np.asarray(ir.opd)
    Ne = np.asarray(ir.N)

    XE = xe.reshape(n_launch, n_launch)
    YE = ye.reshape(n_launch, n_launch)
    J = np.abs(np.gradient(XE, h, axis=1) * np.gradient(YE, h, axis=0)
               - np.gradient(XE, h, axis=0) * np.gradient(YE, h, axis=1)
               ).ravel()

    good = alive & (amp > 0) & np.isfinite(opd) & (J > 0) & (Ne > 0)
    p_launch = float(np.sum((amp ** 2 * n0)[amp > 0])) * h * h
    p_live = float(np.sum((amp ** 2 * n0)[good])) * h * h
    Wj = amp[good] * np.sqrt(n0[good] * J[good] / Ne[good]) * h * h
    ph_exit = K0 * opd[good] + ph0[good]
    xe, ye = xe[good], ye[good]

    ax = (np.arange(n_out) - (n_out - 1) / 2.0) * dx_out
    c0 = (1.0 / (1j * LAM)) if prefactor else 1.0

    E = np.zeros((n_out, n_out), dtype=np.complex128)
    Ez = np.zeros((n_out, n_out), dtype=np.complex128)
    chunk = max(1, int(6e7 // max(xe.size, 1)))
    t0 = time.time()
    for j0 in range(0, n_out, chunk):
        j1 = min(j0 + chunk, n_out)
        dyv = ax[j0:j1, None] - ye[None, :]
        for i, px in enumerate(ax):
            rho = np.sqrt((px - xe)[None, :] ** 2 + dyv ** 2 + back ** 2)
            ker = np.exp(1j * (ph_exit[None, :] + K0 * rho))
            E[j0:j1, i] = c0 * np.sum(Wj * back / (rho * rho) * ker, axis=1)
            if flux:
                # d/dz of  z/rho^2 exp(i k rho)  at fixed transverse offset
                Ez[j0:j1, i] = c0 * np.sum(
                    Wj * (1.0 / (rho * rho)
                          - 2.0 * back * back / rho ** 4
                          + 1j * K0 * back * back / rho ** 3) * ker, axis=1)
    dt = time.time() - t0

    I = np.abs(E) ** 2
    p_sq = float(I.sum()) * dx_out * dx_out
    p_flux = (float(np.imag(np.conj(E) * Ez).sum()) / K0 * dx_out * dx_out
              if flux else float('nan'))

    # analytic paraxial Gaussian focus, for orientation only
    w_f = LAM * f / (np.pi * w)
    na = w / np.hypot(w, f)
    # power-weighted <cos> over the launch cone (the P_sq / P_flux ratio)
    ww = (amp ** 2 * n0)[amp > 0]
    cos_bar = float(np.sum(ww) / np.sum(ww / n0[amp > 0]))
    return {'E': E, 'ax': ax, 'P_in': p_launch, 'P_live': p_live,
            'P_sq': p_sq, 'P_flux': p_flux, 'w_f': w_f, 'NA': na,
            'cos_bar': cos_bar, 'h': h, 'n_dead': int((~good).sum()),
            'time': dt, 'n_launch': n_launch}


def _w_of_na(na, f):
    return na * f / np.sqrt(max(1.0 - na * na, 1e-12))


def _auto_window(w, f, half_in_wf=6.0, pts_per_wf=6.0):
    """Readout pitch / extent scaled to the case's own focal spot, so every NA
    is scored on the SAME fraction of its own Gaussian rather than on a fixed
    micron window (which at low NA holds only a sliver of the spot)."""
    w_f = LAM * f / (np.pi * w)
    dx = w_f / pts_per_wf
    n = int(2 * round(half_in_wf * pts_per_wf) + 1)
    return dx, n


def main():
    f = float(os.environ.get('F_MM', '3.0')) * 1e-3
    back = float(os.environ.get('BACK_MM', '0.5')) * 1e-3
    clip = float(os.environ.get('CLIP', '3.0'))
    nas = [float(v) for v in os.environ.get(
        'NA', '0.010,0.050,0.150,0.333').split(',')]
    nls = [int(v) for v in os.environ.get('NL', '81,121,161,241').split(',')]

    print("RS ABSOLUTE-ENERGY CONTROL -- unaberrated converging sphere")
    print(f"  lambda {LAM * 1e6:.3f} um   f {f * 1e3:.3f} mm   "
          f"back {back * 1e3:.3f} mm   clip {clip:g} w   "
          f"readout auto-sized to +/-6 w_f at 6 pts per w_f")

    # ---- 0. bit-exact NULL floor -----------------------------------------
    w0 = _w_of_na(0.15, f)
    dx0, n0w = _auto_window(w0, f)
    a = rs_control(w=w0, f=f, n_launch=121, clip=clip, back=back,
                   dx_out=dx0, n_out=n0w)
    b = rs_control(w=w0, f=f, n_launch=121, clip=clip, back=back,
                   dx_out=dx0, n_out=n0w)
    same = bool(np.array_equal(a['E'], b['E']))
    print(f"\n[NULL] two identical runs: array_equal = {same}, "
          f"max|dE| = {float(np.abs(a['E'] - b['E']).max()):.3e}")

    # ---- 1. the prefactor, stated as a number ----------------------------
    nop = rs_control(w=w0, f=f, n_launch=121, clip=clip, back=back,
                     dx_out=dx0, n_out=n0w, prefactor=False, flux=False)
    print("\n[PREFACTOR] same case with / without 1/(i lambda):")
    print(f"  P_sq  with = {a['P_sq']:.6e}   without = {nop['P_sq']:.6e}   "
          f"ratio = {nop['P_sq'] / a['P_sq']:.6e}   "
          f"lambda^2 = {LAM ** 2:.6e}")
    print(f"  P/Pin  with = {a['P_sq'] / a['P_in']:.6f}   "
          f"without = {nop['P_sq'] / nop['P_in']:.6e}")

    # ---- 2. convergence vs launch density, over NA -----------------------
    print("\n[CONVERGENCE] NA sweep x launch density.  P_flux/P_in must -> 1.")
    print(f"  {'NA':>7} {'w [mm]':>8} {'NL':>5} {'w_f[um]':>9} "
          f"{'dxo[um]':>8} {'Nout':>5} "
          f"{'P_flux/P_in':>13} {'P_sq/P_in':>11} {'1/<cos>':>9} "
          f"{'P_sq/P_flux':>12} {'t[s]':>6}")
    for na in nas:
        w = _w_of_na(na, f)
        dxo, nout = _auto_window(w, f)
        for nl in nls:
            r = rs_control(w=w, f=f, n_launch=nl, clip=clip, back=back,
                           dx_out=dxo, n_out=nout)
            print(f"  {na:7.3f} {w * 1e3:8.4f} {nl:5d} {r['w_f'] * 1e6:9.4f} "
                  f"{dxo * 1e6:8.4f} {nout:5d} "
                  f"{r['P_flux'] / r['P_in']:13.8f} "
                  f"{r['P_sq'] / r['P_in']:11.6f} "
                  f"{1.0 / r['cos_bar']:9.6f} "
                  f"{r['P_sq'] / r['P_flux']:12.6f} {r['time']:6.1f}",
                  flush=True)
        print()

    # ---- 3. window sweep at the production NA ----------------------------
    na = nas[-1]
    w = _w_of_na(na, f)
    dxo, _ = _auto_window(w, f)
    print(f"[WINDOW] NA {na:.3f}, NL 241, dxo {dxo * 1e6:.4f} um; how much of "
          f"the answer is outside the readout patch")
    for hw in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
        nn = int(2 * round(hw * 6.0) + 1)
        r = rs_control(w=w, f=f, n_launch=241, clip=clip, back=back,
                       dx_out=dxo, n_out=nn)
        print(f"  half-width {hw:4.1f} w_f  (N_out {nn:4d}, "
              f"+/-{(nn - 1) / 2 * dxo * 1e6:7.3f} um)  "
              f"P_flux/P_in = {r['P_flux'] / r['P_in']:.8f}   "
              f"P_sq/P_in = {r['P_sq'] / r['P_in']:.8f}", flush=True)

    # ---- 4. exit-plane placement (the RS integral is exact for any) ------
    dxo, nout = _auto_window(w, f)
    print(f"\n[BACK-OFF] NA {na:.3f}, NL 241, N_out {nout}")
    for bk in (0.2e-3, 0.5e-3, 1.0e-3, 2.0e-3):
        r = rs_control(w=w, f=f, n_launch=241, clip=clip, back=bk,
                       dx_out=dxo, n_out=nout)
        print(f"  back {bk * 1e3:5.2f} mm   "
              f"P_flux/P_in = {r['P_flux'] / r['P_in']:.8f}   "
              f"P_sq/P_in = {r['P_sq'] / r['P_in']:.8f}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
