# POST-C6 RE-MEASUREMENT of the non-null rows of
# docs/audits/APPROXIMATION_AUDIT_TRACED_2026_07_30.md.
#
# WHY THIS EXISTS.  Every non-null row of that audit's ranked table was
# measured against pinned HEAD d2e60ca, i.e. against a tree carrying the OPEN
# niche-C6 defect (``remap`` launching along grad(W) instead of the stationary
# point grad(W + a)).  The one row that was re-controlled against C6 --
# ``_paraxial_group_r_out`` -- COLLAPSED by 97 % and changed sign (+5.97 ->
# -0.16 EE3 pts): the apparent gain was the open defect being fed a
# badly-conditioned carrier and seen through a proxy.  Every other non-null row
# is therefore an UPPER BOUND until re-measured here.
#
# This runner is deliberately a SEPARATE file from approx_ablate_121.py so the
# audit's own reproduction stays byte-identical.  It reuses that harness
# verbatim (approx_common.Patch / run_chain / metrics / field_diff) and differs
# only in (a) the row list, (b) per-row WARNING capture -- the fold-caustic
# loose end needs a controlled pair, and Python de-duplicates warnings per
# location, so every row runs under ``simplefilter('always')`` -- and (c) the
# per-row sampling-adequacy print.
#
# usage (one batch at a time; ~7 min per row on an idle 24-core / 128 GB box,
# and three concurrent 12288-point fine retraces exhaust 128 GB):
#
#   LUMEN_PIN=<pin_c6>   ORD=-4,-2 SET=p1 python approx_post_c6.py
#   LUMEN_PIN=<pin_head> ORD=-4,-2 SET=fold python approx_post_c6.py
import hashlib
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import approx_common as A                                        # noqa: E402
import approx_ablate_121 as AB                                   # noqa: E402

CM = A.CM
LT = A.LT


def p_c6_off():
    """POSITIVE CONTROL #2 -- revert the niche-C6 stationary-phase launch.

    Unlike the C5 control this is the defect the whole re-measurement is
    about, so its end-to-end value is quoted in the same session, on the same
    lattice, as every row it contaminated."""
    return [(LT, 'REMAP_STATIONARY_PHASE_LAUNCH', False)]


def p_resid_degree(d):
    return [(LT, '_REMAP_RESID_EIKONAL_DEGREE', int(d)),
            (LT, '_REMAP_RESID_DEGREE_CAP', max(int(d), 6))]


# name -> (chain_kw, traced_kw, patches)
SETS = {
    # The re-measurement proper.  NULL first (differential floor), then the
    # two positive controls, then every non-null row of the audit table that
    # was never re-controlled.
    'p1': [
        ('NULL identity patch', {}, {},
         [(CM, '_sphere_parab_conversion', CM._sphere_parab_conversion)]),
        ('CONTROL tilt eikonal OFF (C5)', {}, {}, AB.p_tilt_eikonal_off()),
        ('CONTROL C6 launch OFF', {}, {}, p_c6_off()),
        ('carrier_reference sphere -> parabola',
         {'carrier_reference': 'parabola'}, {}, []),
        ('remap_sampling full -> lattice', {},
         {'remap_sampling': 'lattice'}, []),
        ('sphere<->parab taper OFF', {}, {}, AB.p_sphere_taper_off()),
        ('fit_radius_beam_factor -> 3.0', {},
         {'fit_radius_beam_factor': 3.0}, []),
        ('ray_subsample 4 -> 2', {'ray_subsample': 2}, {}, []),
    ],
    # Baseline only -- the controlled HEAD-vs-C6 pair for the fold-caustic
    # loose end (S7.1 of the audit).  Run under each pin in a FRESH process.
    'fold': [],
    # Taper follow-ups (S6 contradiction): is the taper's cost the taper or
    # its RADIUS?  r_safe is scaled rather than removed.
    'taper': [
        ('NULL identity patch', {}, {},
         [(CM, '_sphere_parab_conversion', CM._sphere_parab_conversion)]),
        ('sphere<->parab taper OFF', {}, {}, AB.p_sphere_taper_off()),
        ('sphere<->parab r_safe x2', {}, {}, None),   # filled in main()
        ('sphere<->parab r_safe x0.5', {}, {}, None),
        ('tilt-exactness taper OFF', {}, {}, AB.p_tilt_taper_off()),
    ],
    # Residual-eikonal degree, end to end, post-C6.
    'deg': [
        ('NULL identity patch', {}, {},
         [(CM, '_sphere_parab_conversion', CM._sphere_parab_conversion)]),
        ('resid degree 3', {}, {}, p_resid_degree(3)),
        ('resid degree 4 (shipped)', {}, {}, p_resid_degree(4)),
        ('resid degree 6', {}, {}, p_resid_degree(6)),
    ],
}


def p_sphere_taper_scale(f):
    """`_sphere_parab_conversion` with ``r_safe`` scaled by ``f`` and NOTHING
    else changed (the grid, the difference term and the cos^2 shape are the
    shipped ones -- only the roll-off radius moves).

    The audit removed the taper entirely (T == 1).  This asks the weaker
    question the S6 contradiction actually poses: is the off-axis cost the
    CONVENTION SWAP the taper performs, or only WHERE it performs it?  A pure
    radius effect converges monotonically as f grows; a convention effect does
    not.  f=0.5 is included because an earlier probe found HALVING the radius
    moves 0.13 of the field energy into a ring caustic -- it is the fail-loud
    end of the same axis."""
    def q(shape, dx, wavelength, R, sign, w_beam=None, centre=(0.0, 0.0)):
        if not np.isfinite(R) or R == 0.0:
            return None
        n, ny = int(shape[-1]), int(shape[-2])
        x = (np.arange(n, dtype=np.float64) - n / 2) * dx - float(centre[0])
        y = (np.arange(ny, dtype=np.float64) - ny / 2) * dx - float(centre[1])
        r2 = x[None, :] ** 2 + y[:, None] ** 2
        k = 2.0 * np.pi / wavelength
        diff = CM._exact_sphere_eikonal((ny, n), dx, dx, wavelength, R,
                                        centre=centre) - r2 / (2.0 * R)
        r_safe = f * (abs(R) ** 3 * wavelength / dx) ** (1.0 / 3.0)
        t = np.clip((np.sqrt(r2) - 0.75 * r_safe) / (0.25 * r_safe), 0.0, 1.0)
        return np.exp(sign * 1j * k * diff * np.cos(0.5 * np.pi * t) ** 2)
    return [(CM, '_sphere_parab_conversion', q)]


def run_row(post, env, R, dx, P_in, L, M, cen, ck, tk, pt):
    """One row, with warnings recorded rather than de-duplicated."""
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        E, stages, secs = A.run_chain(post, env, R, dx, L, M, cen,
                                      chain_kw=dict(ck), traced_kw=dict(tk),
                                      patches=pt)
    tally = {}
    for w in wl:
        t = str(w.message)
        key = ('FOLD' if 'fold caustic' in t else
               'ENERGY' if 'energy self-check' in t else
               'NYQ' if 'above the retrace grid' in t or 'nyquist' in t.lower()
               else None)
        if key:
            tally[key] = tally.get(key, 0) + 1
    last = [s for s in stages if not s.get('target')][-1]
    samp = {k: float(last.get(k, np.nan)) for k in
            ('na_exit', 'na_exit_measured', 'na_grid_nyquist',
             'exit_power_above_nyquist')}
    return E, secs, tally, samp


def main():
    order = tuple(int(v) for v in os.environ.get('ORD', '-4,-2').split(','))
    which = os.environ.get('SET', 'p1')
    rows_spec = list(SETS[which])
    if which == 'taper':
        rows_spec[2] = (rows_spec[2][0], {}, {}, p_sphere_taper_scale(2.0))
        rows_spec[3] = (rows_spec[3][0], {}, {}, p_sphere_taper_scale(0.5))

    print("=== POST-C6 re-measurement, order %s, set %s ===" % (order, which))
    for mod in (A.LT, A.CM):
        print("   lib %s\n       sha256 %s" % (
            mod.__file__,
            hashlib.sha256(open(mod.__file__, 'rb').read()).hexdigest()[:16]))
    print("   C6 flag REMAP_STATIONARY_PHASE_LAUNCH = %r  resid degree %r"
          % (getattr(LT, 'REMAP_STATIONARY_PHASE_LAUNCH', 'ABSENT'),
             getattr(LT, '_REMAP_RESID_EIKONAL_DEGREE', 'ABSENT')))
    post, env, R, dx, P_in, L, M, cen = A.setup(order)
    print("grid: RN=%d dx=%.4f um RS=%d NFC=%d WF=%.1f NOUT=%d DXO=%.3f um"
          % (A.RN, dx * 1e6, A.RS, A.NFC, A.WF, A.NOUT, A.DXO * 1e6),
          flush=True)

    E0, s0, tally0, samp0 = run_row(post, env, R, dx, P_in, L, M, cen,
                                    {}, {}, [])
    m0 = A.metrics(E0, P_in)
    print("BASELINE sampling: na_par %.4f  na_measured %.4f  na_nyquist %.4f  "
          "exit power above nyquist %.4e" %
          (samp0['na_exit'], samp0['na_exit_measured'],
           samp0['na_grid_nyquist'], samp0['exit_power_above_nyquist']))
    print("BASELINE warnings: %s" % (tally0 or 'none'))
    rows = [dict(name='BASELINE (shipping defaults)', relL2=0.0, dphi=0.0,
                 secs=s0, tally=tally0, samp=samp0, **m0)]
    for name, ck, tk, pt in rows_spec:
        try:
            E, secs, tally, samp = run_row(post, env, R, dx, P_in, L, M, cen,
                                           ck, tk, pt)
            d, ph = A.field_diff(E, E0)
            rows.append(dict(name=name, relL2=d, dphi=ph, secs=secs,
                             tally=tally, samp=samp, **A.metrics(E, P_in)))
        except Exception as exc:                                # noqa: BLE001
            rows.append(dict(name=name, err='%s: %s' % (type(exc).__name__,
                                                        exc)))
        print("  done: %-42s" % name, flush=True)

    print()
    A.report(rows)
    print()
    print("dEE3 (points) vs baseline, with warnings and exit-nyquist power:")
    for r in rows[1:]:
        if r.get('err'):
            print("  %-42s FAILED %s" % (r['name'], r['err'][:80]))
            continue
        print("  %-42s %+8.4f  (dEE6 %+8.4f, dPtile %+9.5f)  "
              "P>nyq %.4e  warn %s"
              % (r['name'], (r['EE3'] - m0['EE3']) * 100,
                 (r['EE6'] - m0['EE6']) * 100,
                 (r['P_tile'] - m0['P_tile']) * 100,
                 r['samp']['exit_power_above_nyquist'], r['tally'] or '-'))


if __name__ == '__main__':
    main()
