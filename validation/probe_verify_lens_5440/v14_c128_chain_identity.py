"""V14 -- WITHOUT-ARM BIT IDENTITY of the complex128 CARRIER CHAIN.

The complex64 change touches six helpers plus ``_fourier_upsample_crop`` and
``carrier_referenced_exact_focus_readout``, and every call site in the exact
readout, ``_fine_trace_group_exit`` and the chain's group hand-offs.  The
claim is that the complex128 path is BYTE-IDENTICAL.  This hashes:

  * the returned field of a two-group and a three-group
    ``propagate_traced_carrier_chain`` (sphere and parabola carrier
    references, with and without a final free-space leg),
  * the same chains with an exact focus readout attached,
  * ``carrier_referenced_exact_focus_readout`` called directly,
  * ``_fourier_upsample_crop`` on both branches,
  * every per-stage dx / R the chain reports.

Run on v5.43.0 and on 50824e9 and diff the JSON.

Usage:  python v14_c128_chain_identity.py <out.json>
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fix  # noqa: E402

WL = _fix.WL
K = 2 * np.pi / WL


def _singlet(R1, R2, d, glass, ap, name):
    def s(r, gb, ga):
        return {'radius': r, 'glass_before': gb, 'glass_after': ga,
                'conic': 0.0, 'radius_y': None, 'conic_y': None,
                'aspheric_coeffs': None, 'aspheric_coeffs_y': None}
    return {'name': name, 'aperture_diameter': ap, 'thicknesses': [d],
            'surfaces': [s(R1, 'air', glass), s(R2, glass, 'air')]}


G1 = _singlet(55.0e-3, -55.0e-3, 3.0e-3, 'N-BK7', 13.0e-3, 'g1')
G2 = _singlet(40.0e-3, -1e12, 2.5e-3, 'N-SF11', 13.0e-3, 'g2')
G3 = _singlet(90.0e-3, -90.0e-3, 2.0e-3, 'N-BK7', 13.0e-3, 'g3')


def main():
    la = _fix.banner()
    from lumenairy.propagators import carrier as C
    prev = la.get_fft_auto_promote()
    la.set_fft_auto_promote(False)
    out = {'version': la.__version__, 'file': la.__file__, 'cases': {}}
    N, dx, w = 512, 26e-6, 3.6e-3
    x = (np.arange(N) - N // 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    env = np.exp(-r2 / w ** 2).astype(np.complex128)

    def _chain(groups, **over):
        kw = dict(r_in=55e-3, ray_subsample=8, n_workers=1,
                  traced_kwargs=dict(on_undersample='silent',
                                     on_noncollimated='silent'))
        kw.update(over)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return la.propagate_traced_carrier_chain(env, groups, WL, dx, **kw)

    cases = {
        'chain2_sphere': (
            [{'prescription': G1, 'gap_before': 0.0},
             {'prescription': G2, 'gap_before': 12.0e-3}], {}),
        'chain2_parab': (
            [{'prescription': G1, 'gap_before': 0.0},
             {'prescription': G2, 'gap_before': 12.0e-3}],
            dict(carrier_reference='parabola')),
        'chain3_sphere': (
            [{'prescription': G1, 'gap_before': 0.0},
             {'prescription': G2, 'gap_before': 12.0e-3},
             {'prescription': G3, 'gap_before': 9.0e-3}], {}),
        'chain2_final_leg': (
            [{'prescription': G1, 'gap_before': 0.0},
             {'prescription': G2, 'gap_before': 12.0e-3}],
            dict(final_distance=6.0e-3)),
        'chain2_readout': (
            [{'prescription': G1, 'gap_before': 0.0},
             {'prescription': G2, 'gap_before': 12.0e-3}],
            dict(focus_readout=dict(dx_out=0.30e-6, N_out=96,
                                    window_factor=4.0))),
    }
    for name, (groups, over) in cases.items():
        t = time.time()
        try:
            res = _chain(groups, **over)
        except Exception as e:
            out['cases'][name] = {'error': '%s: %s' % (type(e).__name__,
                                                       str(e)[:180])}
            print('%-20s ERROR %s' % (name, type(e).__name__), flush=True)
            continue
        f = np.asarray(res.field)
        stg = []
        for s in (res.stages or []):
            stg.append({k: (float(v) if isinstance(v, (int, float,
                                                       np.floating)) else None)
                        for k, v in s.items()
                        if k in ('dx', 'R', 'R_out', 'z', 'dx_out')})
        out['cases'][name] = {
            'hash': _fix.h(f), 'dtype': str(f.dtype), 'shape': list(f.shape),
            'sum_abs2': float(np.sum(np.abs(f) ** 2)),
            'stage_scalars': stg, 'secs': round(time.time() - t, 2)}
        print('%-20s %s  dtype=%s  P=%.12e  %.1fs'
              % (name, _fix.h(f), f.dtype,
                 out['cases'][name]['sum_abs2'], time.time() - t), flush=True)

    # ---- the readout, called directly ---------------------------------
    Nr, dxr, R = 256, 2.0e-6, -2.0e-3
    xr = (np.arange(Nr) - Nr // 2) * dxr
    r2r = xr[None, :] ** 2 + xr[:, None] ** 2
    Er = (np.exp(-r2r / (0.25e-3 ** 2))
          * np.exp(1j * K * (-(np.sqrt(r2r + R * R) - abs(R))))
          ).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = np.asarray(C.carrier_referenced_exact_focus_readout(
            Er, R, -R, WL, dxr, dx_out=0.1e-6, N_out=64, window_factor=4.0))
    out['cases']['readout_direct'] = {'hash': _fix.h(a), 'dtype': str(a.dtype),
                                      'sum_abs2': float(
                                          np.sum(np.abs(a) ** 2))}
    print('%-20s %s  dtype=%s' % ('readout_direct', _fix.h(a), a.dtype),
          flush=True)

    # ---- the crop, both branches --------------------------------------
    rng = np.random.default_rng(5)
    Nc = 256
    xc = (np.arange(Nc) - Nc / 2) / Nc
    r2c = xc[None, :] ** 2 + xc[:, None] ** 2
    ec = (np.exp(-r2c / 0.05) * (1 + 0.05 * rng.standard_normal((Nc, Nc)))
          ).astype(np.complex128)
    for nc, nf in ((Nc // 2, Nc), (Nc, Nc // 2), (Nc, Nc)):
        g = np.asarray(C._fourier_upsample_crop(ec.copy(), nc, nf))
        key = 'crop_%d_%d' % (nc, nf)
        out['cases'][key] = {'hash': _fix.h(g), 'dtype': str(g.dtype)}
        print('%-20s %s  dtype=%s' % (key, _fix.h(g), g.dtype), flush=True)

    la.set_fft_auto_promote(prev)
    with open(sys.argv[1], 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', sys.argv[1])


if __name__ == '__main__':
    main()
