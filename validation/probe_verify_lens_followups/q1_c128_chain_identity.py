"""Q1 -- BIT IDENTITY of every complex128 carrier chain / readout / crop.

Task 1, first half.  D2 threads ``dtype=`` into ``_build_carrier_phase``; on
the complex128 path that must be the shipped whole-grid ``np.exp`` byte for
byte, and D3 changes only comments.  So EVERY quantity here must be identical
between v5.44.0 and the follow-up branch.

Own fixtures: 2- and 3-group chains on my own meniscus / doublet
prescriptions at 1.55 um, both carrier references (sphere and parab), both
final legs (paraxial and exact -- the exact one is the only route that reaches
``_fourier_upsample_crop``), scalar AND astigmatic carriers through the two
public helpers, and both crop branches (upsample n_crop<n_fine and downsample
n_crop>n_fine) on a speckled envelope.

Usage: python q1_c128_chain_identity.py <out.json> --tree <arm tree>
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vf  # noqa: E402

N, DX = 288, 11e-6
W = 0.95e-3
SUB = 4


def _tk():
    return dict(on_undersample='silent', on_noncollimated='silent',
                on_aperture_beam='silent', parallel_amp=False,
                ray_subsample=SUB, n_workers=1)


def _chain(la, groups, E, **over):
    kw = dict(r_in=0.075, ray_subsample=SUB, n_workers=1,
              traced_kwargs=dict(on_undersample='silent',
                                 on_noncollimated='silent',
                                 on_aperture_beam='silent',
                                 parallel_amp=False),
              on_multi_congruence='ignore', on_na_proximity='ignore',
              on_decentred_fit='ignore', on_gap_paraxial='ignore',
              on_gap_frame='ignore', on_rs_fine_clamp='ignore',
              on_ram_cap='ignore')
    kw.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return la.propagate_traced_carrier_chain(E, groups, _vf.WL, DX, **kw)


def main():
    args = _vf.argp(__doc__).parse_args()
    la = _vf.banner(args.tree)
    from lumenairy.propagators import carrier as C
    res = {}

    g1 = {'prescription': _vf.presc_meniscus(), 'gap_before': 0.0}
    g2 = {'prescription': _vf.presc_doublet(), 'gap_before': 22e-3}
    g3 = {'prescription': _vf.presc_meniscus(), 'gap_before': 18e-3}

    E = _vf.gauss(N, DX, W)
    Esp = _vf.speckled(N, DX, W)

    cases = [
        ('chain2_sphere_par', [g1, g2], E, dict(carrier_reference='sphere',
                                                final_leg='paraxial')),
        ('chain2_parab_par', [g1, g2], E, dict(carrier_reference='parabola',
                                               final_leg='paraxial')),
        ('chain2_sphere_speckled', [g1, g2], Esp,
         dict(carrier_reference='sphere', final_leg='paraxial')),
        ('chain3_sphere_par', [g1, g2, g3], E,
         dict(carrier_reference='sphere', final_leg='paraxial')),
        ('chain2_exact_readout', [g1, g2], E,
         dict(carrier_reference='sphere', final_leg='exact',
              final_distance=4.0e-3)),
        ('chain3_exact_readout', [g1, g2, g3], E,
         dict(carrier_reference='sphere', final_leg='exact',
              final_distance=3.0e-3)),
    ]
    for name, groups, Ein, over in cases:
        t0 = time.perf_counter()
        r = _chain(la, groups, Ein, **over)
        res[name] = {
            'field': _vf.field_record(r.field),
            'dx': float(r.dx), 'R': (None if r.R is None else float(r.R)),
            'stages': [{k: (float(v) if isinstance(v, (int, float,
                                                       np.floating))
                            else repr(v)[:80])
                        for k, v in sorted(st.items())}
                       if isinstance(st, dict) else repr(st)[:200]
                       for st in (r.stages or [])],
            'secs': round(time.perf_counter() - t0, 3),
        }
        print(f"  {name}: {res[name]['field']['hash']} "
              f"({res[name]['secs']} s)", flush=True)

    # ---- the two public carrier helpers, complex128, scalar + astigmatic ---
    for tag, R in (('scalar', 62e-3), ('astig', (62e-3, -48e-3)),
                   ('one_axis', (62e-3, np.inf)),
                   ('one_axis_y', (np.inf, -48e-3))):
        E128 = _vf.speckled(320, 3.7e-6, 0.28 * 320 * 3.7e-6, seed=11)
        a = np.asarray(C.carrier_referenced_envelope(E128, R, _vf.WL, 3.7e-6))
        b = np.asarray(C.carrier_referenced_reconstruct(E128, R, _vf.WL,
                                                        3.7e-6))
        f = C._build_carrier_phase((320, 320), 3.7e-6, 3.7e-6, _vf.WL, R,
                                   -1, 'q1')
        res[f'helper128_{tag}'] = {
            'envelope': _vf.field_record(a),
            'reconstruct': _vf.field_record(b),
            'factor': _vf.field_record(f),
        }
        print(f"  helper128_{tag}: {_vf.h(a)} / {_vf.h(b)} / {_vf.h(f)}",
              flush=True)

    # ---- the exact focus readout, complex128 -------------------------------
    for tag, (Nr, dxr, Rr) in (('readout_a', (256, 2.0e-6, -2.0e-3)),
                               ('readout_b', (320, 1.4e-6, -3.3e-3))):
        x = (np.arange(Nr) - Nr // 2) * dxr
        r2 = x[None, :] ** 2 + x[:, None] ** 2
        k = 2 * np.pi / _vf.WL
        E = (np.exp(-r2 / (0.22e-3 ** 2))
             * np.exp(1j * k * (-(np.sqrt(r2 + Rr * Rr) - abs(Rr)))))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = np.asarray(C.carrier_referenced_exact_focus_readout(
                E.astype(np.complex128), Rr, -Rr, _vf.WL, dxr,
                dx_out=0.12e-6, N_out=72, window_factor=4.0))
        res[tag] = _vf.field_record(out)
        print(f"  {tag}: {res[tag]['hash']}", flush=True)

    # ---- both crop branches, complex128 and complex64 ----------------------
    for nn in (256, 512):
        env = _vf.speckled(nn, 1.9e-6, 0.15 * nn * 1.9e-6, seed=3)
        for nc, nf in ((nn // 2, nn), (nn, nn // 2), (nn, nn)):
            for dt in (np.complex128, np.complex64):
                o = C._fourier_upsample_crop(env.astype(dt), nc, nf)
                res[f'crop_{nn}_{nc}_{nf}_{np.dtype(dt).name}'] = \
                    _vf.field_record(o)
    print(f"  crops: {sum(1 for k in res if k.startswith('crop_'))} cases",
          flush=True)

    _vf.dump(args, {'cases': res, 'N': N, 'DX': DX, 'SUB': SUB,
                    'free_gb': _vf.free_gb()})


if __name__ == '__main__':
    main()
