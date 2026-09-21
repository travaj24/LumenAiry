"""VERIFY-WP-C3 CLAIM 11e -- which chain legs change ROUTE when BOTH the
WP-C3 default (``transport='collins'``) and the WP-C5 default
(``_GAP_KERNEL_ACCURACY_TAU = 1e-4``) land in 5.49.0.

For each representative leg this records, from the RUNNING tree:

  * ``form``       -- what ``_collins_carrier_leg`` resolved: ``'chirp-z'``
                      (the Collins quadrature) or ``'tf'`` (the WP-C3 Sziklas
                      fallback).  On the Sziklas transport there is no such
                      resolution at all.
  * ``k1``/``k3``  -- the Kelly readings the resolution is taken on.
  * ``z_eff``      -- ``B/A``, the reduced frame the exact-kernel refinement
                      lives in.
  * ``departure``  -- the WP-C5 law ``sqrt(3/2) k |z_eff| theta_env^4 / 8``,
                      i.e. what C5's ``tau`` is compared against.
  * ``tau_fires``  -- departure > 1e-4, i.e. whether C5's rule WOULD change
                      the kernel IF the leg reached ``_collins_transport``.
  * ``tau_reached``-- whether the leg actually reaches the tau condition at
                      all under C3's routing (it does NOT on the ``'tf'``
                      fallback, which is the whole point of this probe).

Usage::

    PYTHONPATH=<tree> python vc3_route_c5.py OUT.json
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

import lumenairy.propagators.carrier as CA

LAM = 1.064e-6


def axis(n, d):
    return (np.arange(int(n), dtype=np.float64) - int(n) / 2) * float(d)


def gauss(n, dx, w):
    x = axis(n, dx)
    return np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
                  / w ** 2).astype(np.complex128)


def q_field(n, dx, q_in, z, lam):
    k = 2.0 * np.pi / lam
    xo = axis(n, dx)
    xx, yy = np.meshgrid(xo, xo, indexing='xy')
    r2 = xx ** 2 + yy ** 2
    return (np.exp(1j * k * z) / (1.0 + z / q_in)
            * np.exp(1j * k * r2 / (2.0 * (q_in + z))))


def theta_env_of(env, dx, lam):
    S = np.fft.fft2(np.ascontiguousarray(env, dtype=np.complex128))
    return max(CA._collins_envelope_half_angle(S, dx, dx, lam))


def measure(name, env, R, z, dx, n, lam=LAM, dx_out=None, carrier_out=None):
    row = {'leg': name, 'R': R, 'z': z, 'dx': dx, 'N': n}
    diag = {}
    err = None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try:
            cr = CA._collins_carrier_leg(
                env, R, z, lam, dx, dx, gap_kernel='auto',
                on_collins_sampling='ignore', dx_out=dx_out,
                carrier_out=carrier_out, diag=diag)
            row['out_dx'] = (cr.dx if not isinstance(cr.dx, tuple)
                             else list(cr.dx))
            row['out_R'] = (cr.R if not isinstance(cr.R, tuple)
                            else list(cr.R))
            row['nan'] = bool(np.isnan(np.asarray(cr.env)).any())
            row['digest'] = float(np.linalg.norm(np.asarray(cr.env)))
        except Exception as exc:                      # noqa: BLE001
            err = '%s: %s' % (type(exc).__name__, str(exc)[:160])
        row['warnings'] = [str(x.message)[:90] for x in w]
    row['error'] = err
    row['form'] = diag.get('collins_form')
    row['k1'] = diag.get('collins_k1')
    row['k3'] = diag.get('collins_k3')

    # --- the C5 quantities, computed independently of the route taken -----
    Rx, Ry, _ = CA._parse_carrier(R, 'probe')
    A, B, _, _ = CA._collins_envelope_abcd(Rx, z, np.inf)
    z_eff = (B / A) if A != 0.0 else float('inf')
    th = theta_env_of(env, dx, lam)
    dep = (CA._collins_exact_kernel_departure(abs(z_eff), th, lam)
           if np.isfinite(z_eff) else float('inf'))
    row.update(A=A, B=B, z_eff=z_eff, theta_env=th, departure=dep,
               tau_fires=bool(dep > 1e-4),
               tau_reached=(row['form'] == 'chirp-z'),
               tau_in_tree=CA._GAP_KERNEL_ACCURACY_TAU)
    return row


def main(out):
    rows = []

    # F3: the fixture WP-C5's whole blast radius is measured on.
    N, DX, W, R = 1024, 4e-6, 0.30e-3, -40e-3
    env = gauss(N, DX, W)
    for dz, tag in ((1e-6, '1um'), (10e-6, '10um'), (100e-6, '100um'),
                    (1e-3, '1mm'), (5e-3, '5mm')):
        rows.append(measure('F3-gap-leg-%s-short-of-focus' % tag,
                            env, R, -R - dz, DX, N))
    # the SAME rungs with the lattice NAMED, which is how C5's probe calls it
    for dz, tag in ((1e-6, '1um'), (10e-6, '10um')):
        rows.append(measure('F3-NAMED-lattice-%s' % tag, env, R, -R - dz,
                            DX, N, dx_out=5.6447e-06))

    # a leg PAST the focus (A < 0) -- the WP-C3 exclusion that was dropped
    N2, DX2, W2, R2 = 2048, 12.1e-6, 4e-6, None
    # the focus-crossing oracle: w0 = 4 um at 30 mm
    zR = np.pi * (4e-6) ** 2 / LAM
    q0 = complex(-30e-3, zR)
    w_in = float(np.sqrt(LAM / (np.pi * np.imag(-1.0 / q0))))
    R_in = 1.0 / float(np.real(1.0 / q0))
    env2 = gauss(N2, DX2, w_in)
    for z, tag in ((45e-3, 'A=-0.5'), (60e-3, 'A=-1.0'), (15e-3, 'A=+0.5')):
        rows.append(measure('focus-crossing-%s' % tag, env2, R_in, z,
                            DX2, N2))

    # a COLLIMATED relay leg -- the commonest chain input
    env3 = gauss(512, 8e-6, 0.5e-3)
    for z in (5e-3, 50e-3, 500e-3):
        rows.append(measure('collimated-relay-%.0fmm' % (z * 1e3),
                            env3, float('inf'), z, 8e-6, 512))

    # an ASTIGMATIC leg -- the other dropped exclusion
    env4 = gauss(1024, 4e-6, 0.30e-3)
    rows.append(measure('astigmatic-R=(-40,-55)mm-z=5mm', env4,
                        (-40e-3, -55e-3), 5e-3, 4e-6, 1024))

    # a leg with the OUTPUT LATTICE named (never eligible for the fallback)
    rows.append(measure('named-lattice-relay', env3, float('inf'), 50e-3,
                        8e-6, 512, dx_out=9e-6))

    res = {'tree': CA.__file__, 'tau': CA._GAP_KERNEL_ACCURACY_TAU,
           'transport_default': CA.propagate_carrier_referenced.__defaults__
           is not None, 'rows': rows}
    import inspect
    res['sig_transport'] = str(
        inspect.signature(CA.propagate_carrier_referenced)
        .parameters['transport'].default)
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1, default=str)
    hdr = ('%-40s %-8s %-9s %-11s %-11s %-6s %-6s' %
           ('leg', 'form', 'k1max', 'z_eff', 'departure', 'fires', 'reach'))
    print(hdr)
    for r in rows:
        k1 = r['k1']
        k1s = ('%.4g' % max(k1)) if isinstance(k1, (list, tuple)) else '-'
        print('%-40s %-8s %-9s %-11.4g %-11.4g %-6s %-6s %s' %
              (r['leg'][:40], r['form'], k1s, r['z_eff'], r['departure'],
               r['tau_fires'], r['tau_reached'], r['error'] or ''))
    print('tau in tree =', CA._GAP_KERNEL_ACCURACY_TAU,
          ' transport default =', res['sig_transport'])
    print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'vc3_route_c5.json')
