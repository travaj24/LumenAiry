# THE ABSOLUTE P/Pin OF THE EXACT-RAY ORACLE ON DESIGN 121, after the
# Rayleigh-Sommerfeld bookkeeping fix.
# (ORACLE_ENERGY_AND_D6_HALO_2026_08_01, task A.)
#
# POP_CROSSCHECK_121_2026_07_31 S9.2 could not determine this: the RS kernel
# omitted ``1/(i lambda)`` and the POP harness double-counted the launch cell,
# so the measured ratio was 3.1e-05 and meaningless.  Both are fixed
# (``exact_ray_oracle_121.oracle_spot``, ``pop_ours_121.py``) and the machinery
# is validated on an analytic case in ``oracleE_rs_control.py``.
#
# WHAT IS MEASURED, and against what:
#   NULL     two identical runs, array_equal on the image field;
#   SCALE    the prefactor is a pure constant, so every EE / FWHM the campaign
#            published must be unchanged -- checked against a bypassed run;
#   ABS      P_window_flux / P_launch for (0,0) and (-4,-2), swept over the
#            readout window and the launch density;
#   S1.5     the energy audit's own conservation-reference command, re-run, so
#            its ``live power 100.0000 %`` can be checked against the fix.
#
# Env knobs: ORD, NL, CLIP, BACK, RN, RS, PART=null|abs|s15|all
import os
import sys
import time

import _d121_common as C
import numpy as np
from exact_ray_oracle_121 import oracle_spot


def _setup(rn, rs):
    _pre, post, _g, period = C.geometry()
    env, R, dx, _P = C.chain_a(n=rn, rs=rs)
    return post, period, env, R, dx


def part_null(post, period, env, R, dx, nl, clip, back):
    print("\n=== [NULL] bit-exact floor + the prefactor is a pure scale ===")
    L = M = 0.0
    a = oracle_spot(env, R, dx, post, L, M, n_launch=nl, clip=clip, back=back,
                    dx_out=0.4e-6, n_out=61, verbose=False)
    b = oracle_spot(env, R, dx, post, L, M, n_launch=nl, clip=clip, back=back,
                    dx_out=0.4e-6, n_out=61, verbose=False)
    print(f"  two identical runs: array_equal(E) = "
          f"{bool(np.array_equal(a['E'], b['E']))}   "
          f"max|dI| = {float(np.abs(a['I'] - b['I']).max()):.3e}")
    # bypass: undo the prefactor and re-derive the metrics the campaign quotes
    lam = C.LAM
    E0 = a['E'] * (1j * lam)          # exactly what the old kernel returned
    I0 = np.abs(E0) ** 2
    ax = a['ax']
    Xg, Yg = np.meshgrid(ax, ax)
    def _ee(I):
        tot = float(I.sum())
        cx = float((I * Xg).sum() / tot)
        cy = float((I * Yg).sum() / tot)
        r = np.hypot(Xg - cx, Yg - cy)
        return [float(I[r <= rad].sum()) / tot for rad in (3e-6, 6e-6, 12e-6)]
    new = _ee(a['I'])
    old = _ee(I0)
    print("  EE (centroid) with prefactor  : "
          + '  '.join(f'{v * 100:.10f}' for v in new))
    print("  EE (centroid) prefactor undone: "
          + '  '.join(f'{v * 100:.10f}' for v in old))
    rel = max(abs(n - o) / max(o, 1e-300) for n, o in zip(new, old))
    print(f"  worst relative EE change from the fix: {rel:.3e}")


def part_abs(post, period, env, R, dx, nl, clip, back, orders):
    print("\n=== [ABS] absolute P/Pin, window and launch-density sweeps ===")
    lam = C.LAM
    for (m, n) in orders:
        L = m * lam / period
        M = n * lam / period
        print(f"\n--- order ({m:+d},{n:+d})  tilt ({L * 1e3:+.3f},"
              f"{M * 1e3:+.3f}) mrad ---")
        print(f"  {'dxo[um]':>8} {'Nout':>5} {'halfwin[um]':>11} "
              f"{'NL':>5} {'live%':>9} {'P_flux/P_in':>13} {'P_sq/P_in':>11} "
              f"{'step p99.9':>11} {'t[s]':>6}")
        # window sweep at a pitch that still resolves the spot (Nyquist for
        # NA 0.37 is 1.78 um; 0.8 um is 2.2x inside it)
        for (dxo, nout) in ((0.8e-6, 41), (0.8e-6, 81), (0.8e-6, 151),
                            (0.8e-6, 251), (0.1e-6, 261)):
            t0 = time.time()
            r = oracle_spot(env, R, dx, post, L, M, n_launch=nl, clip=clip,
                            back=back, dx_out=dxo, n_out=nout, verbose=False)
            print(f"  {dxo * 1e6:8.2f} {nout:5d} "
                  f"{(nout - 1) / 2 * dxo * 1e6:11.2f} {nl:5d} "
                  f"{r['live_frac'] * 100:9.4f} "
                  f"{r['P_ratio_flux'] * 100:13.6f} "
                  f"{r['P_ratio_sq'] * 100:11.4f} "
                  f"{r['ray_step_w_p999']:11.5f} {time.time() - t0:6.1f}",
                  flush=True)
        # launch-density sweep on the widest window
        for nl2 in (81, 121, 201, 241):
            t0 = time.time()
            r = oracle_spot(env, R, dx, post, L, M, n_launch=nl2, clip=clip,
                            back=back, dx_out=0.8e-6, n_out=151, verbose=False)
            print(f"  {0.8:8.2f} {151:5d} {60.0:11.2f} {nl2:5d} "
                  f"{r['live_frac'] * 100:9.4f} "
                  f"{r['P_ratio_flux'] * 100:13.6f} "
                  f"{r['P_ratio_sq'] * 100:11.4f} "
                  f"{r['ray_step_w_p999']:11.5f} {time.time() - t0:6.1f}",
                  flush=True)
        # exit-reference-plane placement: the RS integral is exact for any
        for bk in (2.0e-3, 5.0e-3, 10.0e-3):
            t0 = time.time()
            r = oracle_spot(env, R, dx, post, L, M, n_launch=nl, clip=clip,
                            back=bk, dx_out=0.8e-6, n_out=151, verbose=False)
            print(f"  back {bk * 1e3:5.1f} mm -> P_flux/P_in = "
                  f"{r['P_ratio_flux'] * 100:.6f} %   P_sq/P_in = "
                  f"{r['P_ratio_sq'] * 100:.4f} %   [{time.time() - t0:.0f}s]",
                  flush=True)


def part_s15(post, period, env, R, dx):
    """The energy audit's OWN conservation-reference command, verbatim:
    ORD='0,0;-1,0;-4,0;-4,-2' NL=161 NOUT=61 DXO=0.2 CLIP=3.0."""
    print("\n=== [S1.5] the energy audit's conservation reference, re-run ===")
    lam = C.LAM
    print(f"  {'order':>9} {'live power %':>14} {'dead':>7} {'exit NA':>9} "
          f"{'step(max)':>10} {'EE3 %':>8} {'EE6 %':>8} "
          f"{'P_flux/P_in %':>14}")
    for (m, n) in ((0, 0), (-1, 0), (-4, 0), (-4, -2)):
        L = m * lam / period
        M = n * lam / period
        r = oracle_spot(env, R, dx, post, L, M, n_launch=161, clip=3.0,
                        back=5.0e-3, dx_out=0.2e-6, n_out=61, verbose=False)
        print(f"  ({m:+d},{n:+d})  {r['live_frac'] * 100:14.4f} "
              f"{r['n_dead']:7d} {r['na_eff']:9.4f} "
              f"{r['ray_step_weighted']:10.4f} "
              f"{r['ee3_cen'] * 100:8.2f} {r['ee6_cen'] * 100:8.2f} "
              f"{r['P_ratio_flux'] * 100:14.4f}", flush=True)
    print("  (the last column is NEW; every other column must reproduce "
          "ENERGY_CONSERVATION_AUDIT_2026_07_31 S1.5 exactly, because "
          "``live_frac`` is p_live/p_launch -- a ratio of two sums that BOTH "
          "carry h^2 and NEITHER touches the RS kernel.)")


def main():
    part = os.environ.get('PART', 'all')
    nl = int(os.environ.get('NL', '161'))
    clip = float(os.environ.get('CLIP', '3.0'))
    back = float(os.environ.get('BACK', '5.0')) * 1e-3
    rn = int(os.environ.get('RN', '1024'))
    rs = int(os.environ.get('RS', '4'))
    orders = [tuple(int(v) for v in s.split(','))
              for s in os.environ.get('ORD', '0,0;-4,-2').split(';') if s]
    post, period, env, R, dx = _setup(rn, rs)
    print(f"DOE plane: R = {R * 1e3:.4f} mm, dx = {dx * 1e6:.4f} um, "
          f"N = {env.shape[0]}   NL={nl} clip={clip} back={back * 1e3:.1f} mm")
    P_doe = float(np.sum(np.abs(env) ** 2)) * dx * dx
    print(f"DOE-plane grid power SUM|env|^2 dx^2 = {P_doe:.6e} "
          f"(the launch lattice re-quadratures the same field)")
    if part in ('null', 'all'):
        part_null(post, period, env, R, dx, nl, clip, back)
    if part in ('abs', 'all'):
        part_abs(post, period, env, R, dx, nl, clip, back, orders)
    if part in ('s15', 'all'):
        part_s15(post, period, env, R, dx)
    return 0


if __name__ == '__main__':
    sys.exit(main())
