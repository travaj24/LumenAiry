"""VERIFY-WP-C3 CLAIM 11e -- the ADVERSARIAL hunt for a leg where WP-C3's
Sziklas fallback BYPASSES WP-C5's near-focus tau rule.

The two rules are taken in this order on a chain gap leg:

    _collins_carrier_leg  ->  tf_available and (k1 > 1 or k3 > 1)?
                              YES -> propagate_carrier_referenced(transport=
                                     'sziklas'), which NEVER evaluates
                                     _GAP_KERNEL_ACCURACY_TAU (the tau
                                     condition lives inside
                                     _collins_transport)
                              NO  -> _collins_transport, which DOES

So a leg with ``form == 'tf'`` AND ``departure > tau`` is a leg where the
5.49.0 tree keeps the exact kernel that C5 decided to drop.  This sweeps a
two-parameter family (carrier radius x distance short of the geometric focus)
looking for one, and prints the joint distribution either way.
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np

import lumenairy.propagators.carrier as CA

LAM = 1.064e-6
TAU = 1e-4


def gauss(n, dx, w):
    x = (np.arange(n) - n / 2) * dx
    return np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
                  / w ** 2).astype(np.complex128)


def theta_env_of(env, dx, lam):
    S = np.fft.fft2(np.ascontiguousarray(env, dtype=np.complex128))
    return max(CA._collins_envelope_half_angle(S, dx, dx, lam))


def probe(env, R, z, dx, lam=LAM):
    diag = {}
    err = None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            CA._collins_carrier_leg(env, R, z, lam, dx, dx, gap_kernel='auto',
                                    on_collins_sampling='ignore', diag=diag)
        except Exception as exc:                          # noqa: BLE001
            err = '%s' % type(exc).__name__
    A, B, _, _ = CA._collins_envelope_abcd(R, z, np.inf)
    z_eff = (B / A) if A != 0.0 else float('inf')
    th = theta_env_of(env, dx, lam)
    dep = (CA._collins_exact_kernel_departure(abs(z_eff), th, lam)
           if np.isfinite(z_eff) else float('inf'))
    return (diag.get('collins_form'), diag.get('collins_k1'),
            diag.get('collins_k3'), z_eff, dep, err)


def main(out):
    rows = []
    # (N, dx, w) grids; R from a long relay to a very short one
    for (N, dx, w) in ((1024, 4e-6, 0.30e-3), (512, 8e-6, 0.5e-3),
                       (256, 6e-6, 0.10e-3), (2048, 2e-6, 0.30e-3)):
        env = gauss(N, dx, w)
        for R_mm in (-100.0, -40.0, -20.0, -10.0, -5.0, -2.0, -1.0, -0.5):
            R = R_mm * 1e-3
            for dz in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
                z = -R - dz
                form, k1, k3, z_eff, dep, err = probe(env, R, z, dx)
                rows.append(dict(N=N, dx=dx, w=w, R=R, dz=dz, form=form,
                                 k1=(max(k1) if k1 else None),
                                 k3=(max(k3) if k3 else None),
                                 z_eff=z_eff, departure=dep, err=err,
                                 bypass=bool(form == 'tf' and dep > TAU),
                                 fires_on_collins=bool(form == 'chirp-z'
                                                       and dep > TAU)))
    byp = [r for r in rows if r['bypass']]
    fire = [r for r in rows if r['fires_on_collins']]
    print('legs swept            :', len(rows))
    print('form==tf              :', sum(r['form'] == 'tf' for r in rows))
    print('form==chirp-z         :', sum(r['form'] == 'chirp-z' for r in rows))
    print('errored               :', sum(r['err'] is not None for r in rows))
    print('tau WOULD fire        :', sum(r['departure'] > TAU for r in rows))
    print('  ... and reaches tau :', len(fire))
    print('  ... BYPASSED (tf)   :', len(byp))
    for r in byp[:25]:
        print('   BYPASS  N=%d dx=%.1eum w=%.2fmm R=%.1fmm dz=%.0e  k1=%.3g '
              'k3=%.3g z_eff=%.4g dep=%.4g'
              % (r['N'], r['dx'] * 1e6, r['w'] * 1e3, r['R'] * 1e3, r['dz'],
                 r['k1'] or -1, r['k3'] or -1, r['z_eff'], r['departure']))
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump({'tree': CA.__file__, 'tau_in_tree':
                   CA._GAP_KERNEL_ACCURACY_TAU, 'rows': rows}, fh,
                  indent=1, default=str)
    print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'vc3_bypass_hunt.json')
