# Approximation audit -- the decisive END-TO-END test of
# ``_paraxial_group_r_out``.
#
# approx_reference_fit_121.py PART 1 measures that the paraxial-Moebius exit
# carrier radius is wrong by 0.6-6 % on design 121's powered groups, and that
# the defocus this leaves in the stored envelope carries 0.10-0.25 cycles per
# co-moving pixel -- a large fraction of the 0.5 cyc/px representation limit,
# on top of the group's genuine aberration.
#
# A REFERENCE choice cannot cost anything in the continuum.  It CAN cost in a
# finite grid, by pushing the residual past Nyquist.  This script settles
# which case design 121 is in, by replacing the reference with the EXACT one
# (the sphere fitted to the group's own traced exit wavefront, iterated to
# self-consistency because each group's R_out is the next group's R_in) and
# re-running the complete shipped chain through the exact readout.
#
# The override goes through the PUBLIC per-group ``'r_out'`` key -- no
# monkeypatch, no library edit.
#
# Env: ORD, ITERS (default 3), plus approx_common's grid knobs.
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import approx_common as A                                      # noqa: E402
import approx_reference_fit_121 as RF                          # noqa: E402

LAM = A.LAM
LT = A.LT


def fit_rout(g, w_weight=True):
    """Best-fit exit sphere radius about the chain's own chief ray, from the
    group's traced exit wavefront.  Amplitude-weighted over the beam."""
    xo, yo, W, xi, yi, al, w, rad, cin = RF.traced_exit(g)
    m = al & (np.hypot(xi - cin[0], yi - cin[1]) <= 1.3 * w)
    if m.sum() < 100:
        return g['R_out'], np.nan
    spec = LT.TiltedCarrier(g['R_out'], g['L_out'], g['M_out'],
                            g['x_c_out'], g['y_c_out'])
    S, _, _ = LT._tilted_carrier_parts(spec, xo[m], yo[m])
    a = W[m] - S
    px = xo[m] - g['x_c_out']
    py = yo[m] - g['y_c_out']
    r2 = px * px + py * py
    wt = (np.exp(-2.0 * (np.hypot(xi[m] - cin[0], yi[m] - cin[1]) / w) ** 2)
          if w_weight else np.ones_like(px))
    Amat = np.column_stack([np.ones_like(px), px, py, 0.5 * r2]) * wt[:, None]
    c, *_ = np.linalg.lstsq(Amat, a * wt, rcond=None)
    dK = float(c[3])
    inv = 1.0 / g['R_out'] + dK
    return (1.0 / inv if inv != 0.0 else np.inf), dK


def capture_with(post, env, R, dx, L, M, r_outs, leg='paraxial'):
    """Run the chain with per-group r_out overrides and capture the element
    calls + the stage table."""
    groups = []
    k = 0
    for g in post:
        gg = dict(g)
        if r_outs is not None and k < len(r_outs) and r_outs[k] is not None:
            gg['r_out'] = float(r_outs[k])
        groups.append(gg)
        k += 1
    grabs = []
    real = A.EL.apply_real_lens_traced

    def patched(E_in, *, prescription, wavelength, dx, **kw):
        grabs.append({'E': np.array(E_in), 'carrier': kw.get('carrier'),
                      'dx': float(dx), 'presc': prescription,
                      'name': prescription.get('name')})
        return real(E_in, prescription=prescription, wavelength=wavelength,
                    dx=dx, **kw)

    with A.Patch([(A.EL, 'apply_real_lens_traced', patched)]):
        res = A.la.propagate_traced_carrier_chain(
            env, post if r_outs is None else groups, LAM, dx,
            r_in=A.la.TiltedCarrier(R, L, M), ray_subsample=A.RS,
            n_workers=A.NW, final_distance=A.TRAILING, final_leg=leg,
            on_decentred_fit='ignore', on_gap_paraxial='ignore',
            on_na_proximity='ignore')
    stages = [s for s in res.stages if not s.get('target')]
    for gr, s in zip(grabs, stages):
        gr['R_out'] = float(s['R_out'])
        gr['x_c_out'] = float(s.get('x_c_out', 0.0))
        gr['y_c_out'] = float(s.get('y_c_out', 0.0))
        gr['L_out'] = float(s.get('L_out', 0.0))
        gr['M_out'] = float(s.get('M_out', 0.0))
    return grabs, groups


def main():
    order = tuple(int(v) for v in os.environ.get('ORD', '-4,-2').split(','))
    iters = int(os.environ.get('ITERS', 3))
    post, env, R, dx, P_in, L, M, cen = A.setup(order)
    print("order %s   pinned lib %s" % (order, A.LT.__file__))

    r_outs = None
    for it in range(iters):
        grabs, groups = capture_with(post, env, R, dx, L, M, r_outs)
        new = []
        for g in grabs:
            rf, dK = fit_rout(g)
            new.append(rf)
        print("iter %d: R_out used -> traced-fit (mm)" % it)
        for g, rf in zip(grabs, new):
            print("   %-12s %14.6f -> %14.6f   (frac %+.3e)"
                  % (g['name'][:12], g['R_out'] * 1e3, rf * 1e3,
                     (rf - g['R_out']) / abs(g['R_out'])))
        r_outs = new
    print()
    print("converged traced r_out overrides (mm): %s"
          % ["%.6f" % (v * 1e3) for v in r_outs])

    # --- end-to-end, exact readout, baseline vs traced reference -----------
    E0, st0, s0 = A.run_chain(post, env, R, dx, L, M, cen)
    m0 = A.metrics(E0, P_in)
    groups = []
    for g, ro in zip(post, r_outs):
        gg = dict(g)
        gg['r_out'] = float(ro)
        groups.append(gg)
    E1, st1, s1 = A.run_chain(groups, env, R, dx, L, M, cen)
    m1 = A.metrics(E1, P_in)
    d, ph = A.field_diff(E1, E0)
    print()
    print("%-34s %8s %8s %8s %9s" % ('', 'EE3%', 'EE6%', 'EE12%', 'Ptile%'))
    print("%-34s %8.4f %8.4f %8.4f %9.5f" %
          ('baseline (paraxial Moebius R_out)', m0['EE3'] * 100,
           m0['EE6'] * 100, m0['EE12'] * 100, m0['P_tile'] * 100))
    print("%-34s %8.4f %8.4f %8.4f %9.5f" %
          ('traced best-fit exit sphere', m1['EE3'] * 100, m1['EE6'] * 100,
           m1['EE12'] * 100, m1['P_tile'] * 100))
    print("%-34s %+8.4f %+8.4f %+8.4f %+9.5f   relL2 %.3e  dphi %.2e"
          % ('DELTA (points)', (m1['EE3'] - m0['EE3']) * 100,
             (m1['EE6'] - m0['EE6']) * 100, (m1['EE12'] - m0['EE12']) * 100,
             (m1['P_tile'] - m0['P_tile']) * 100, d, ph))
    for tag, st in (('baseline', st0), ('traced-ref', st1)):
        last = [s for s in st if not s.get('target')][-1]
        print("  %-11s na_par %.4f na_meas %.4f na_nyq %.4f  P>nyq %.3e"
              % (tag, last['na_exit'], last.get('na_exit_measured', np.nan),
                 last.get('na_grid_nyquist', np.nan),
                 last.get('exit_power_above_nyquist', np.nan)))


if __name__ == '__main__':
    main()
