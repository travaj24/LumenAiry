# Is ``_sphere_parab_conversion``'s cos^2 taper a PHYSICS defect or only a
# readout hazard?
#
# The taper is the identity inside its onset ``0.75*r_safe``, so the CHAIN's
# returned field is expressed in the SAME convention there whether the taper is
# on or off.  Any difference inside the onset therefore cannot be a convention
# difference -- it is the transport having carried something different.  This
# script partitions ||E_on - E_off||^2 by radius against the FINAL conversion's
# own onset, and reports how much of the chain's power sits in each band.
#
# It also runs the ONLY unambiguous control available at the field level: the
# chain truncated after group k, for k = 1..6, so the leg at which the two
# diverge is named rather than inferred.
#
# usage:  ORD=0,0 python fc_taper_locality.py
import os
import sys
import warnings

import numpy as np

os.environ.setdefault('LUMEN_PIN', '0')

import _d121_common as C                                       # noqa: E402
import approx_ablate_121 as AB                                 # noqa: E402
from approx_common import Patch                                # noqa: E402
from lumenairy.propagators.carrier import (                    # noqa: E402
    _envelope_amp_radius, carrier_referenced_envelope)

LAM = C.LAM
K0 = 2 * np.pi / LAM


def run(post, env, R, dx, L, M, rs, taper, ngroups):
    items = ([] if taper == 'on' else AB.p_sphere_taper_off())
    with Patch(items):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return C.la.propagate_traced_carrier_chain(
                env, [dict(g) for g in post[:ngroups]], LAM, dx,
                r_in=C.la.TiltedCarrier(R, L, M), ray_subsample=rs,
                n_workers=8, final_distance=0.0, final_leg='paraxial',
                on_decentred_fit='ignore')


def main():
    m, n = (int(v) for v in os.environ.get('ORD', '0,0').split(','))
    rn = int(os.environ.get('RN', '1024'))
    rs = int(os.environ.get('RS', '4'))
    _pre, post, _g, period = C.geometry()
    env, R, dx, _P = C.chain_a(n=rn, rs=rs)
    L, M = m * LAM / period, n * LAM / period
    print(f"order ({m:+d},{n:+d})  RN={rn} RS={rs}")
    print(f"{'k':>2} {'R_out mm':>10} {'dx um':>8} {'w um':>9} "
          f"{'r_safe mm':>10} {'onset/w':>8} {'dP/P':>11} "
          f"{'IN onset':>11} {'OUT':>11} {'P in onset':>10}")
    for k in range(1, len(post) + 1):
        a = run(post, env, R, dx, L, M, rs, 'on', k)
        b = run(post, env, R, dx, L, M, rs, 'off', k)
        Ea, Eb = np.asarray(a.field), np.asarray(b.field)
        dxk, Rk = float(a.dx), float(a.R)
        assert Ea.shape == Eb.shape and abs(float(b.dx) - dxk) < 1e-18
        st = [s for s in a.stages if not s.get('target')][-1]
        xck, yck = st.get('x_c_out', 0.0), st.get('y_c_out', 0.0)
        envk = carrier_referenced_envelope(Ea, Rk, LAM, dxk)
        w = _envelope_amp_radius(envk, dxk, dxk)
        nn = Ea.shape[0]
        u = (np.arange(nn) - nn / 2) * dxk
        # the co-moving grid is the CHIEF-RAY-TRACKING frame at the exit, and
        # the final conversion is applied about the grid centre there, so the
        # onset is measured from the grid centre -- not from (xck, yck).
        rr = np.hypot(u[None, :], u[:, None])
        r_safe = (abs(Rk) ** 3 * LAM / dxk) ** (1.0 / 3.0)
        onset = 0.75 * r_safe
        inn = rr <= onset
        d2 = np.abs(Ea - Eb) ** 2
        pa = np.abs(Ea) ** 2
        tot = float(pa.sum())
        print(f"{k:2d} {Rk * 1e3:10.4f} {dxk * 1e6:8.3f} {w * 1e6:9.1f} "
              f"{r_safe * 1e3:10.4f} {onset / w:8.3f} "
              f"{float(d2.sum()) / tot:11.3e} "
              f"{float(d2[inn].sum()) / tot:11.3e} "
              f"{float(d2[~inn].sum()) / tot:11.3e} "
              f"{float(pa[inn].sum()) / tot:10.6f}   "
              f"(chief {xck * 1e3:+.3f},{yck * 1e3:+.3f} mm)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
