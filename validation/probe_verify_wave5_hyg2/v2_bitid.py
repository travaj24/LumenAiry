"""(1) INDEPENDENT NumPy-path bit-identity key set for H2-2.

Built from scratch for this verification (the author's own probe has 84 keys;
this one has its own geometries, its own helper spellings and its own
refusal arms).  Run with PYTHONPATH pinned to ONE tree; writes {key: sha256}.

    python v2_bitid.py <tree> <out.json>
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                            # noqa: E402
from vlib import Probe, anchor, build_tag                      # noqa: E402


def gauss(n, dx, w, dtype=np.complex128, cx=0.0, cy=0.0):
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (w * w)).astype(dtype)


def main():
    tree, out = sys.argv[1], sys.argv[2]
    anchor(tree)
    import lumenairy.propagators.carrier as CA
    P = Probe()

    WL = 633e-9
    N = 48
    DX = 7.5e-6
    env = gauss(N, DX, 55e-6)
    env_odd = gauss(33, DX, 55e-6)
    env_c64 = gauss(N, DX, 55e-6, dtype=np.complex64)
    env_dec = gauss(N, DX, 40e-6, cx=60e-6, cy=-30e-6)
    env_wide = gauss(N, 40e-6, 460e-6)

    # ---- A. public entry, transport='collins' -----------------------------
    GEOMS = {
        'g1': dict(R_carrier=-0.037, z=3.1e-3),
        'g2': dict(R_carrier=np.inf, z=1.7e-2),
        'g3': dict(R_carrier=+0.062, z=8.5e-3),
        'g4': dict(R_carrier=-0.21, z=0.19),
    }
    for gname, geom in GEOMS.items():
        for gk in ('auto', 'fresnel', 'exact'):
            P.call('A.pcr.' + gname + '.' + gk,
                   CA.propagate_carrier_referenced,
                   env, geom['R_carrier'], geom['z'], WL, DX,
                   transport='collins', gap_kernel=gk,
                   on_collins_sampling='ignore')

    # output-reference spellings x 2 geometries
    SPELL = {
        'default': {},
        'flat': dict(carrier_out=np.inf),
        'explicit': dict(carrier_out=-0.0295),
        'pitch': dict(dx_out=1.3e-5),
        'pitchflat': dict(dx_out=1.3e-5, carrier_out=np.inf),
    }
    for gname in ('g1', 'g4'):
        for sname, kw in SPELL.items():
            P.call('A.ref.' + gname + '.' + sname,
                   CA.propagate_carrier_referenced,
                   env, GEOMS[gname]['R_carrier'], GEOMS[gname]['z'], WL, DX,
                   transport='collins', gap_kernel='fresnel',
                   on_collins_sampling='ignore', **kw)

    # astigmatic arm and its refusal
    for gk in ('auto', 'fresnel', 'exact'):
        P.call('A.astig.' + gk, CA.propagate_carrier_referenced,
               env, (-0.037, -0.058), 3.1e-3, WL, DX,
               transport='collins', gap_kernel=gk,
               on_collins_sampling='ignore')
    P.call('A.astig.out', CA.propagate_carrier_referenced,
           env, (-0.037, -0.058), 3.1e-3, WL, DX, transport='collins',
           gap_kernel='fresnel', carrier_out=(-0.03, -0.05),
           on_collins_sampling='ignore')

    # complex64, tilt, second wavelength, sziklas control
    P.call('A.c64', CA.propagate_carrier_referenced, env_c64, -0.037, 3.1e-3,
           WL, DX, transport='collins', gap_kernel='fresnel',
           on_collins_sampling='ignore')
    P.call('A.c64.auto', CA.propagate_carrier_referenced, env_c64, -0.037,
           3.1e-3, WL, DX, transport='collins', gap_kernel='auto',
           on_collins_sampling='ignore')
    for gk in ('fresnel', 'exact'):
        P.call('A.tilt.' + gk, CA.propagate_carrier_referenced, env, -0.037,
               3.1e-3, WL, DX, transport='collins', gap_kernel=gk,
               tilt=(0.09, -0.04), on_collins_sampling='ignore')
    P.call('A.lam2', CA.propagate_carrier_referenced, env_wide, -0.30, 0.29,
           1.55e-6, 40e-6, transport='collins', gap_kernel='auto',
           on_collins_sampling='ignore')
    P.call('A.sziklas', CA.propagate_carrier_referenced, env, -0.037, 3.1e-3,
           WL, DX, transport='sziklas', gap_kernel='auto')
    P.call('A.decentred', CA.propagate_carrier_referenced, env_dec, -0.037,
           3.1e-3, WL, DX, transport='collins', gap_kernel='auto',
           on_collins_sampling='ignore')
    P.call('A.odd', CA.propagate_carrier_referenced, env_odd, -0.037, 3.1e-3,
           WL, DX, transport='collins', gap_kernel='fresnel',
           on_collins_sampling='ignore')

    # the Kelly guard's three dispositions on a leg that violates it
    import warnings

    def guarded(action):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                r = CA._collins_transport(
                    env_wide, -0.30, 0.29, 1.55e-6, 40e-6, 40e-6,
                    dx_out=40e-6 * 16, dy_out=40e-6 * 16, N_out_x=N,
                    N_out_y=N, R_ref=np.inf, gap_kernel='auto',
                    on_collins_sampling=action)
                v = ('ok', np.asarray(r))
            except BaseException as exc:                       # noqa: BLE001
                v = ('raised', type(exc).__name__, str(exc))
        return (v, [(w.category.__name__, str(w.message)) for w in caught])

    for action in ('error', 'warn', 'ignore'):
        P.call('A.guard.' + action, guarded, action)

    # ---- B. the six helpers, called directly ------------------------------
    CH = [
        dict(n=48, d=DX, wavelength=WL, R=-0.02),
        dict(n=48, d=DX, wavelength=WL, R=+0.02),
        dict(n=33, d=DX, wavelength=WL, R=-0.02),
        dict(n=48, d=DX, wavelength=WL, R=-0.02, offset=3.3e-5),
        dict(n=48, d=DX, wavelength=WL, R=-0.02, dtype=np.complex64),
        dict(n=48, d=DX, wavelength=WL, R=-1e6, offset=-1.1e-5,
             dtype=np.complex128),
        dict(n=1, d=DX, wavelength=WL, R=-0.02),
        dict(n=48, d=DX, wavelength=1.55e-6, R=2.5e-4),
    ]
    for i, kw in enumerate(CH):
        P.call('B.chirp.%d' % i, CA._collins_axis_chirp, **kw)

    from lumenairy.propagators.fft_infra import _fft2
    S = _fft2(np.ascontiguousarray(env, dtype=np.complex128))
    S_odd = _fft2(np.ascontiguousarray(env_odd, dtype=np.complex128))
    EK = [
        dict(spectrum=S, z_eff=2.2e-3, wavelength=WL, dx=DX, dy=DX,
             tilt=(0.0, 0.0)),
        dict(spectrum=S, z_eff=-2.2e-3, wavelength=WL, dx=DX, dy=DX,
             tilt=(0.0, 0.0)),
        dict(spectrum=S, z_eff=2.2e-3, wavelength=WL, dx=DX, dy=DX,
             tilt=(0.11, -0.05)),
        dict(spectrum=S_odd, z_eff=2.2e-3, wavelength=WL, dx=DX, dy=DX,
             tilt=(0.0, 0.0)),
        dict(spectrum=S, z_eff=2.2e-3, wavelength=WL, dx=DX, dy=1.4 * DX,
             tilt=(0.02, 0.0)),
        dict(spectrum=S, z_eff=2.2e-3, wavelength=WL, dx=DX, dy=DX,
             tilt=(0.8, 0.7)),          # |s|^2 >= 1 -> evanescent refusal
    ]
    for i, kw in enumerate(EK):
        P.call('B.exactcorr.%d' % i, CA._collins_exact_kernel_correction,
               **kw)

    P.call('B.space.0', CA._collins_space_support, env, DX, DX, 1e-3)
    P.call('B.space.1', CA._collins_space_support, env_dec, DX, 1.4 * DX,
           1e-2)
    P.call('B.angle.0', CA._collins_angle_support, S, DX, DX, WL, 1e-3)
    P.call('B.angle.1', CA._collins_angle_support, S_odd, DX, 1.4 * DX, WL,
           1e-2)
    P.call('B.marg.0', CA._collins_power_marginals, env)
    P.call('B.marg.1', CA._collins_power_marginals, env_c64)
    P.call('B.box.0', CA._collins_input_box, env, DX, DX, WL, 1e-3)
    P.call('B.box.1', CA._collins_input_box, env, DX, DX, WL, 1e-3,
           spectrum=S)

    P.call('B.stats.0', CA._collins_sampling_stats, 0.84, 3.1e-3, -27.0, 1.0,
           DX, DX, 1.6e-4, 1.6e-4, 1.1e-2, 1.1e-2, 8.0e-6, 8.0e-6, 48, 48,
           (0.0, 0.0), WL)
    P.call('B.stats.1', CA._collins_sampling_stats, 0.1, -3.1e-3, 5.0, -2.0,
           DX, 1.4 * DX, 1.6e-4, 2.0e-4, 1.1e-2, 3.0e-2, 8.0e-6, 9.0e-6, 48,
           64, (1.0e-4, -2.0e-4), WL)

    TKW = dict(dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N, R_ref=-0.0295,
               on_collins_sampling='ignore')
    for gk in ('auto', 'fresnel', 'exact'):
        P.call('B.tr.' + gk, CA._collins_transport, env, -0.037, 3.1e-3, WL,
               DX, DX, gap_kernel=gk, **TKW)
    P.call('B.tr.flatref', CA._collins_transport, env, -0.037, 3.1e-3, WL,
           DX, DX, gap_kernel='fresnel', **dict(TKW, R_ref=np.inf))
    P.call('B.tr.centre', CA._collins_transport, env, -0.037, 3.1e-3, WL, DX,
           DX, gap_kernel='fresnel', centre_out=(2.0e-5, -1.0e-5), **TKW)
    P.call('B.tr.astig', CA._collins_transport, env, (-0.037, -0.058),
           3.1e-3, WL, DX, DX, gap_kernel='fresnel',
           **dict(TKW, R_ref=(-0.0295, -0.041)))
    P.call('B.tr.astig.exact', CA._collins_transport, env, (-0.037, -0.058),
           3.1e-3, WL, DX, DX, gap_kernel='exact',
           **dict(TKW, R_ref=(-0.0295, -0.041)))
    P.call('B.tr.c64', CA._collins_transport, env_c64, -0.037, 3.1e-3, WL,
           DX, DX, gap_kernel='fresnel', **TKW)
    P.call('B.tr.B0', CA._collins_transport, env, -0.037, 0.0, WL, DX, DX,
           gap_kernel='fresnel', **TKW)
    P.call('B.tr.Rzero', CA._collins_transport, env, 0.0, 3.1e-3, WL, DX, DX,
           gap_kernel='fresnel', **TKW)
    P.call('B.tr.badkernel', CA._collins_transport, env, -0.037, 3.1e-3, WL,
           DX, DX, gap_kernel='Fresnel', **TKW)
    P.call('B.tr.tilt.exact', CA._collins_transport, env, -0.037, 3.1e-3, WL,
           DX, DX, gap_kernel='exact', tilt=(0.09, -0.04), **TKW)
    P.call('B.tr.tilt.evan', CA._collins_transport, env, -0.037, 3.1e-3, WL,
           DX, DX, gap_kernel='exact', tilt=(0.8, 0.7), **TKW)
    P.call('B.tr.period', CA._collins_transport, env_wide, -0.30, 0.29,
           1.55e-6, 40e-6, 40e-6, dx_out=40e-6 * 16, dy_out=40e-6 * 16,
           N_out_x=N, N_out_y=N, R_ref=np.inf, gap_kernel='fresnel',
           on_collins_sampling='ignore', check_period=True)

    def with_stats(**kw):
        st = {}
        r = CA._collins_transport(stats_out=st, **kw)
        return (np.asarray(r), st)

    P.call('B.tr.stats_out', with_stats, env=env, R_in=-0.037, z=3.1e-3,
           wavelength=WL, dx=DX, dy=DX, gap_kernel='auto', **TKW)
    P.call('B.tr.stats_out.exact', with_stats, env=env_wide, R_in=-0.30,
           z=0.29, wavelength=1.55e-6, dx=40e-6, dy=40e-6,
           dx_out=40e-6 * 16, dy_out=40e-6 * 16, N_out_x=N, N_out_y=N,
           R_ref=np.inf, gap_kernel='auto', on_collins_sampling='ignore')

    P.call('B.leg.0', CA._collins_carrier_leg, env, -0.037, 3.1e-3, WL, DX,
           DX, gap_kernel='auto', on_collins_sampling='ignore')
    P.call('B.leg.1', CA._collins_carrier_leg, env, (-0.037, -0.058), 3.1e-3,
           WL, DX, DX, gap_kernel='fresnel', on_collins_sampling='ignore',
           dx_out=1.1e-5, dy_out=1.2e-5)
    P.call('B.ro.0', CA._collins_focus_readout, env, -0.037, 3.1e-3, WL, DX,
           DX, dx_out=2.0e-6, N_out=48, gap_kernel='auto',
           on_replica='ignore', on_collins_sampling='ignore')
    P.call('B.ro.1', CA._collins_focus_readout, env, -0.037, 3.1e-3, WL, DX,
           DX, dx_out=2.0e-6, N_out=48, centre_out=(1.0e-5, 0.0),
           gap_kernel='fresnel', on_replica='ignore',
           on_collins_sampling='ignore')

    for i, (Ri, z, Rr) in enumerate(((-0.037, 3.1e-3, -0.0295),
                                     (np.inf, 1.7e-2, np.inf),
                                     (-0.037, 0.037, np.inf),
                                     (0.0, 1.0e-3, np.inf))):
        P.call('B.abcd.%d' % i, CA._collins_envelope_abcd, Ri, z, Rr)
    for i, (ze, th, span) in enumerate(((2.2e-3, 1.1e-2, 3.6e-4),
                                        (2.2e-1, 3.0e-1, 3.6e-4),
                                        (np.inf, 1.1e-2, 3.6e-4),
                                        (2.2e-3, 1.5, 3.6e-4))):
        P.call('B.k4.%d' % i, CA._collins_kernel_wrap_ratio, ze, th, span)
    P.call('B.outaxis.0', CA._collins_leg_output_axis, 0.84, 3.1e-3, -0.0295,
           DX, 48, 1.6e-4, 1.1e-2, WL)
    P.call('B.outaxis.1', CA._collins_leg_output_axis, 1e-3, 3.1e-3, np.inf,
           DX, 48, 1.6e-4, 1.1e-2, WL)
    P.call('B.contain.0', CA._collins_containment_radius,
           np.abs(env[24]) ** 2, (np.arange(48) - 24.0) * DX, 0.0, 1e-3)

    P.add('meta.build', build_tag())
    P.write(out)


if __name__ == '__main__':
    main()
