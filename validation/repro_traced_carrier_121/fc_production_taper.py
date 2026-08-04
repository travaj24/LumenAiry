# The taper, re-measured on v5.32.0 through the LIBRARY'S OWN production
# readout -- no ray launch of this study's anywhere.
#
# ``fc_instrument_121.py`` scores the taper through a ray + Rayleigh-Sommerfeld
# readout that this study also changed (the launch-phase split), so on its own
# it cannot separate "the field got better" from "my readout got better at
# reading THIS field".  This runner removes that objection: it is
# ``approx_post_c6.py``'s row machinery reduced to two rows, on the COMPLETE
# shipped path (chain A -> TiltedCarrier(order) -> six post-DOE groups -> the
# 7.7058 mm trailing leg -> final_leg='exact' -> the exact Bluestein readout),
# fixed output lattice on the order's exact chief ray, NULL row first.
#
# APPROXIMATION_AUDIT_POST_C6 S2 measured `taper OFF` at **+1.4147 EE3** on
# order (-4,-2) against the C6 tree; that predates niche C8, so the brief asks
# for it again here.
#
# usage:  ORDERS='0,0 -4,-2' python fc_production_taper.py
import os
import sys
import warnings

os.environ.setdefault('LUMEN_PIN', '0')

import approx_ablate_121 as AB                                 # noqa: E402
import approx_common as A                                      # noqa: E402
import fc_instrument_121 as FI                                 # noqa: E402
import lumenairy.propagators.carrier as CM                     # noqa: E402


def main():
    orders = os.environ.get('ORDERS', '0,0 -4,-2').split()
    print(FI._provenance(), flush=True)
    print(f"config: RN={A.RN} RS={A.RS} NFC={A.NFC} WF={A.WF} "
          f"NOUT={A.NOUT} DXO={A.DXO * 1e6} um  final_leg='exact'", flush=True)
    for o in orders:
        m, n = (int(v) for v in o.split(','))
        post, env, R, dx, P_in, L, M, cen = A.setup((m, n))
        print(f"\n########## ORDER ({m:+d},{n:+d}) ##########", flush=True)
        rows = []
        # Pinned through the C9 FLAG, not through the library default: since
        # niche C9 landed the default IS the exact conversion, so "no patch"
        # no longer means "the taper".  Row 1 is v5.32.0, row 2 is its own
        # bit-exact null, row 3 is the shipped C9 state -- and row 4 reaches
        # the same state through the pre-flag MONKEYPATCH, so the two routes
        # are checked against each other on the production path too.
        spec = [
            ('BASELINE = v5.32.0 (C9 off, tapered)',
             ((CM, 'SPHERE_PARAB_CONVERSION_EXACT', False),)),
            ('NULL identity patch',
             ((CM, 'SPHERE_PARAB_CONVERSION_EXACT', False),
              (CM, '_sphere_parab_conversion', CM._sphere_parab_conversion))),
            ('C9 ON (exact conversion, shipped)',
             ((CM, 'SPHERE_PARAB_CONVERSION_EXACT', True),)),
            ('same via the pre-flag monkeypatch',
             tuple(AB.p_sphere_taper_off())
             + ((CM, 'SPHERE_PARAB_CONVERSION_EXACT', False),)),
        ]
        base = None
        for name, patches in spec:
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter('always')
                E, stages, secs = A.run_chain(post, env, R, dx, L, M, cen,
                                              patches=patches)
            mt = A.metrics(E, P_in)
            if base is None:
                base = E
            d, dphi = A.field_diff(E, base)
            nw = sum(1 for w in wl if 'fold caustic' in str(w.message))
            rows.append({'name': name, 'secs': secs, 'relL2': d,
                         'dphi': dphi, **mt})
            e0 = mt['EE3'] * 100
            b0 = rows[0]['EE3'] * 100
            pn = stages[-1].get('exit_power_above_nyquist')
            print(f"  {name:32s} EE3 {e0:8.4f}  dEE3 {e0 - b0:+8.4f}  "
                  f"EE6 {mt['EE6'] * 100:8.4f}  Ptile {mt['P_tile'] * 100:8.4f}"
                  f"  relL2 {d:9.3e}  fold {nw}  "
                  f"P>nyq {pn if pn is None else f'{pn:.4e}'}  "
                  f"[{secs:.0f}s]", flush=True)
        A.report(rows)
    return 0


if __name__ == '__main__':
    sys.exit(main())
