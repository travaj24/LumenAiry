# Split the "last group" step into its TWO halves: the 3.3233 mm free-space
# COARSE LEG to the group-5 front vertex, and the ELEMENT PASS itself.
#
# ``hybrid_localize_121.py`` bisects by GROUP, and a group's step contains both
# its gap leg and its element pass.  The scope doc attributes the whole
# 24.05-point step at group 5 to "the element pass", but the leg is inside that
# step too -- and group 5's leg is the one that re-grids the co-moving pitch
# (38.432 -> 33.211 um) on a beam that is already converging at R = -24.5 mm.
#
# Three hand-off planes, ONE readout:
#   n5    chain does groups 0..4, oracle does [gap + group 5]      (= hybrid n=5)
#   gap   chain does groups 0..4 AND the gap leg, oracle does group 5 alone
#   n6    chain does everything                                    (= hybrid n=6)
#
# Env: ORD, RN, RS, NL, NOUT, DXO, BACK, WHICH ('n5,gap,n6').
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
import _d121_common as C                                        # noqa: E402
from exact_ray_oracle_121 import _phase_gradient                # noqa: E402
from hybrid_localize_121 import _fourier_up, _free_surfaces, rs_spot  # noqa: E402,E501
from lumenairy.propagators.carrier import (                     # noqa: E402
    _envelope_amp_radius, carrier_referenced_envelope)
from lumenairy.raytrace import RayBundle, trace                 # noqa: E402

LAM = C.LAM
K0 = 2 * np.pi / LAM


def main():
    m, n = (int(v) for v in os.environ.get('ORD', '-4,-2').split(','))
    rn = int(os.environ.get('RN', '1024'))
    rs = int(os.environ.get('RS', '4'))
    up = int(os.environ.get('UP', '1'))
    nout = int(os.environ.get('NOUT', '61'))
    dxo = float(os.environ.get('DXO', '0.4')) * 1e-6
    back = float(os.environ.get('BACK', '5.0')) * 1e-3
    clip = float(os.environ.get('CLIP', '3.0'))
    nl0 = int(os.environ.get('NL', '121'))
    which = os.environ.get('WHICH', 'n5,gap,n6').split(',')

    _pre, post, _g, period = C.geometry()
    env_doe, R_doe, dx_doe, _P = C.chain_a(n=rn, rs=rs)
    L = m * LAM / period
    M = n * LAM / period
    # GRP: which group's ENTRANCE LEG is split (default the last).  The whole
    # experiment is only meaningful if splitting an INNOCENT leg (the scope
    # doc bounds groups 0..4 at 0.31 EE3 points in total) returns the n=GRP
    # reading -- that is the control for the method itself.
    G = int(os.environ.get('GRP', '5'))
    gap5 = float(post[G]['gap_before'])
    print(f"order ({m:+d},{n:+d})  gap before group {G} = {gap5 * 1e3:.4f} mm  "
          f"NOUT={nout} DXO={dxo * 1e6} NL={nl0} BACK={back * 1e3} mm")

    ch = trace(RayBundle(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
                         L=np.array([L]), M=np.array([M]),
                         N=np.array([np.sqrt(1 - L * L - M * M)]),
                         wavelength=LAM, alive=np.ones(1, bool),
                         opd=np.zeros(1)),
               C.post_surfaces(post), LAM, output_filter='last').image_rays
    xci, yci = float(ch.x[0]), float(ch.y[0])

    rows = []
    for tag in which:
        t0 = time.time()
        if tag == 'n5':
            grp, fd = post[:G], 0.0
            surfs = C.post_surfaces(post[G:], back_off=back)
        elif tag.startswith('gap'):
            # ``gap`` == the whole leg; ``gap:f`` == the chain does the first
            # fraction f of it and the oracle does the rest.  f -> 0 MUST
            # return continuously to the n5 reading, or the hand-off
            # bookkeeping for ``final_distance != 0`` is the thing being
            # measured rather than the leg.
            _f = float(tag.split(':', 1)[1]) if ':' in tag else 1.0
            grp, fd = post[:G], _f * gap5
            _rest = [dict(g) for g in post[G:]]
            _rest[0]['gap_before'] = (1.0 - _f) * gap5
            surfs = C.post_surfaces(_rest, back_off=back)
        elif tag == 'n6':
            grp, fd = post[:G + 1], 0.0
            surfs = (_free_surfaces(C.TRAILING, back) if G + 1 == len(post)
                     else C.post_surfaces(post[G + 1:], back_off=back))
        else:
            raise ValueError(tag)
        res = C.la.propagate_traced_carrier_chain(
            env_doe, grp, LAM, dx_doe, r_in=C.la.TiltedCarrier(R_doe, L, M),
            ray_subsample=rs, n_workers=8, final_distance=fd,
            final_leg='paraxial', on_decentred_fit='ignore')
        fld, Rk, dxk = np.asarray(res.field), float(res.R), float(res.dx)
        _tg = [s for s in res.stages if s.get('target')]
        if fd != 0.0 and _tg:
            st = _tg[-1]
            Lk, Mk = st['L'], st['M']
            xck, yck = st['x_c'], st['y_c']
        elif fd != 0.0:                    # untilted chain: no <target> stage
            Lk = Mk = 0.0
            xck = yck = 0.0
        else:
            st = [s for s in res.stages if not s.get('target')][-1]
            Lk, Mk = st.get('L_out', L), st.get('M_out', M)
            xck, yck = st.get('x_c_out', 0.0), st.get('y_c_out', 0.0)
        envk = carrier_referenced_envelope(fld, Rk, LAM, dxk)
        nn = envk.shape[0]
        u = (np.arange(nn) - nn / 2) * dxk
        envk = envk * np.exp(-1j * K0 * (Lk * u[None, :] + Mk * u[:, None]))
        envk = _fourier_up(envk, up)
        dxk_u = dxk / up
        nn = envk.shape[0]
        wk = _envelope_amp_radius(envk, dxk_u, dxk_u)
        half = int(np.ceil(clip * wk / dxk_u))
        c = nn // 2
        i0 = max(c - half, 0)
        i1 = min(c + half + 1, nn)
        envk = envk[i0:i1, i0:i1]
        nn = envk.shape[0]
        u = (np.arange(nn) - (c - i0)) * dxk_u
        gx, gy, step_ph = _phase_gradient(envk, dxk_u)
        phres = np.unwrap(np.unwrap(np.angle(envk), axis=1), axis=0)
        sd = max(1, int(np.floor(nn / nl0)))
        envk = envk[::sd, ::sd]
        gx, gy, phres = gx[::sd, ::sd], gy[::sd, ::sd], phres[::sd, ::sd]
        u = u[::sd]
        nn = envk.shape[0]
        U, V = np.meshgrid(u, u)
        x0 = (U + xck).ravel()
        y0 = (V + yck).ravel()
        amp = np.abs(envk).ravel()
        p = (U / Rk).ravel() + Lk + gx.ravel() / K0
        q = (V / Rk).ravel() + Mk + gy.ravel() / K0
        ph0 = K0 * ((U ** 2 + V ** 2).ravel() / (2 * Rk)
                    + Lk * U.ravel() + Mk * V.ravel()) + phres.ravel()
        r = rs_spot(x0, y0, amp, ph0, p, q, surfs, back, xci, yci,
                    dx_out=dxo, n_out=nout, nl=nn)
        rows.append((tag, r))
        print(f"  {tag:4s}  [{time.time() - t0:.0f}s]  grid {nn}^2 @ "
              f"{dxk_u * sd * 1e6:.3f} um  w {wk * 1e6:.1f} um  chief "
              f"({xck * 1e3:+.4f},{yck * 1e3:+.4f}) mm  R {Rk * 1e3:+.4f} mm")
        print(f"        envelope per-pixel phase step {step_ph:.4f} rad "
              f"({'OK' if step_ph < 0.5 else 'ALIASED'});  integrand step "
              f"p50/p99.9 {r['step_md']:.4f}/{r['step_w']:.4f} cycles;  live "
              f"{r['live'] * 100:.4f} %")
        print(f"        FWHM {r['fwhm'] * 1e6:6.3f} um   EE3 "
              f"{r['ee3'] * 100:6.2f}   EE6 {r['ee6'] * 100:6.2f}   EE12 "
              f"{r['ee12'] * 100:6.2f}")
    print()
    print(f"GAP-LEG SPLIT, order ({m:+d},{n:+d}):")
    for tag, r in rows:
        print(f"  {tag:4s}  EE3 {r['ee3'] * 100:6.2f}  EE6 "
              f"{r['ee6'] * 100:6.2f}  FWHM {r['fwhm'] * 1e6:6.3f} um")


if __name__ == '__main__':
    sys.exit(main())
