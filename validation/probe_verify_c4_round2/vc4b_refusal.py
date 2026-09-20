"""VERIFY-WP-C4 ROUND 2, item 5 -- is the ``mft_method=`` refusal two-sided,
and is the contract applied uniformly?

The branch refuses ``mft_method=`` at three entry points where no transform is
reached (``compute_psf`` with a non-MFT sampler, ``propagate`` with a method /
grid combination that never promotes to an MFT leg, and
``propagate_carrier_referenced`` on the Sziklas transport).  Two things are
measured here rather than read:

1.  **Two-sidedness.**  At each of the three, the keyword must be ACCEPTED
    where a transform IS reached and REFUSED where it is not, and the refusal
    must be a ``ValueError`` that names the keyword and says why.
2.  **Uniformity.**  Every other exported entry point that takes
    ``mft_method=`` is put in the state where it reaches no transform, and the
    outcome is recorded.  A contract that refuses at three doors and silently
    swallows at a fourth is one contract with a hole in it, not two contracts.

    PYTHONPATH=<tree> python vc4b_refusal.py <tree>

Author:  Andrew Traverso
"""
from __future__ import annotations

import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vc4blib as L                                              # noqa: E402

WL = 1.31e-6
N = 512
M = 16


def _gauss(n, dx, w):
    import numpy as np
    g = (np.arange(n) - n / 2.0) * dx
    return np.exp(-((g[None, :] ** 2 + g[:, None] ** 2) / w ** 2)).astype(
        np.complex128)


def _try(fn):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fn()
        return {'outcome': 'accepted', 'exc': None, 'message': None}
    except Exception as exc:                                  # noqa: BLE001
        return {'outcome': 'raised', 'exc': type(exc).__name__,
                'message': str(exc)}


def main(tree):
    import numpy as np
    la = L.anchor(tree)
    L.single_thread_ffts()
    from lumenairy.propagators import carrier as C
    from lumenairy.propagators.carrier_field import (
        CarrierField, CarrierSpec, FieldGrid, re_reference)

    dx = 2e-6
    E = _gauss(N, dx, N * dx / 6.0)
    pup = _gauss(N, 4e-6, N * 4e-6 / 6.0)
    cases = []

    def case(name, kind, fn):
        rec = _try(fn)
        rec.update({'case': name, 'kind': kind})
        cases.append(rec)
        print("  %-62s %-9s %s %s"
              % (name, kind, rec['outcome'],
                 ('(' + (rec['exc'] or '') + ') '
                  + (rec['message'] or '')[:100]) if rec['exc'] else ''),
              flush=True)

    # ---- 1) compute_psf -----------------------------------------------
    case('compute_psf(method="mft", mft_method="bluestein")', 'REACHES',
         lambda: la.compute_psf(pup, WL, 50e-3, 4e-6, N_psf=M, method='mft',
                                dx_psf=1e-6, mft_method='bluestein'))
    case('compute_psf(method="fft", mft_method="bluestein")', 'NO-TRANSFORM',
         lambda: la.compute_psf(pup, WL, 50e-3, 4e-6, N_psf=N, method='fft',
                                mft_method='bluestein'))
    # ---- 2) propagate --------------------------------------------------
    case('propagate(method="asm", output_grid=..., mft_method=...)',
         'REACHES',
         lambda: la.propagate(E, z=2e-3, wavelength=WL, dx=dx, method='asm',
                              output_grid=(M, 0.4e-6),
                              mft_method='bluestein'))
    case('propagate(method="asm", NO output_grid, mft_method=...)',
         'NO-TRANSFORM',
         lambda: la.propagate(E, z=2e-3, wavelength=WL, dx=dx, method='asm',
                              mft_method='bluestein'))
    case('propagate(method="rs", output_grid=..., mft_method=...)',
         'NO-TRANSFORM',
         lambda: la.propagate(E, z=2e-3, wavelength=WL, dx=dx, method='rs',
                              output_grid=(M, 0.4e-6),
                              mft_method='bluestein'))
    # ---- 3) propagate_carrier_referenced -------------------------------
    case('propagate_carrier_referenced(transport="collins", mft_method=...)',
         'REACHES',
         lambda: C.propagate_carrier_referenced(
             _gauss(N, 0.4e-6, 25e-6), -6.0e-4, 3.0e-4, WL, 0.4e-6,
             transport='collins', on_collins_sampling='ignore',
             mft_method='bluestein'))
    case('propagate_carrier_referenced(transport="sziklas", mft_method=...)',
         'NO-TRANSFORM',
         lambda: C.propagate_carrier_referenced(
             _gauss(N, 0.4e-6, 25e-6), -6.0e-4, 3.0e-4, WL, 0.4e-6,
             transport='sziklas', mft_method='bluestein'))
    # ---- 4) UNIFORMITY: the other keyword-bearing entry points ---------
    case('resample_field(method="chirpz", mft_method=...)', 'REACHES',
         lambda: la.resample_field(E, dx, dx * N / M, N_out=M,
                                   method='chirpz',
                                   mft_method='bluestein'))
    case('resample_field(method="spline", mft_method=...)', 'NO-TRANSFORM',
         lambda: la.resample_field(E, dx, dx * N / M, N_out=M,
                                   method='spline',
                                   mft_method='bluestein'))

    def _chain(readout, kw):
        presc = {'name': 'p', 'aperture_diameter': 14e-3,
                 'thicknesses': [3e-3], 'surfaces': [
                     {'radius': 60e-3, 'glass_before': 'air',
                      'glass_after': 'N-BK7', 'conic': 0.0, 'radius_y': None,
                      'conic_y': None, 'aspheric_coeffs': None,
                      'aspheric_coeffs_y': None},
                     {'radius': -60e-3, 'glass_before': 'N-BK7',
                      'glass_after': 'air', 'conic': 0.0, 'radius_y': None,
                      'conic_y': None, 'aspheric_coeffs': None,
                      'aspheric_coeffs_y': None}]}
        extra = dict(focus_readout=readout) if readout else {}
        return C.propagate_traced_carrier_chain(
            _gauss(N, 30e-6, 4.5e-3),
            [{'prescription': presc, 'gap_before': 20e-3}], WL, 30e-6,
            r_in=60e-3, ray_subsample=16, n_workers=1, final_distance=8e-3,
            final_leg='paraxial', transport='sziklas',
            traced_kwargs=dict(on_undersample='silent',
                               on_noncollimated='silent'),
            **extra, **kw)

    case('propagate_traced_carrier_chain(focus_readout=..., mft_method=...)',
         'REACHES',
         lambda: _chain(dict(dx_out=0.4e-6, N_out=M, on_replica='ignore'),
                        {'mft_method': 'bluestein'}))
    case('propagate_traced_carrier_chain(NO focus_readout, mft_method=...)',
         'NO-TRANSFORM',
         lambda: _chain(None, {'mft_method': 'bluestein'}))

    fa = CarrierField(_gauss(N, 0.3e-6, 20e-6), FieldGrid((N, N), 0.3e-6),
                      CarrierSpec(R=-5.0e-4), WL)
    case('re_reference(coarser target grid, mft_method=...)', 'REACHES',
         lambda: re_reference(fa, CarrierSpec(R=-5.5e-4),
                              FieldGrid((M, M), N * 0.3e-6 / M),
                              on_nyquist='ignore', on_window='ignore',
                              mft_method='bluestein'))

    out = {'build': L.build(), 'python': sys.version.split()[0],
           'lumenairy_version': la.__version__, 'numpy': np.__version__,
           'cases': cases}
    reaches_refused = [c['case'] for c in cases
                       if c['kind'] == 'REACHES' and c['outcome'] == 'raised']
    silently_accepted = [c['case'] for c in cases
                         if c['kind'] == 'NO-TRANSFORM'
                         and c['outcome'] == 'accepted']
    bad_exc = [c['case'] for c in cases
               if c['kind'] == 'NO-TRANSFORM' and c['outcome'] == 'raised'
               and (c['exc'] != 'ValueError'
                    or 'mft_method' not in (c['message'] or ''))]
    out['REACHES_but_refused'] = reaches_refused
    out['NO_TRANSFORM_but_silently_accepted'] = silently_accepted
    out['NO_TRANSFORM_refused_but_not_about_the_keyword'] = bad_exc
    print()
    for k in ('REACHES_but_refused', 'NO_TRANSFORM_but_silently_accepted',
              'NO_TRANSFORM_refused_but_not_about_the_keyword'):
        print("%-48s %s" % (k, out[k]))
    L.write(out, os.path.join(HERE, "vc4b_refusal_%s.json" % L.tag()))


if __name__ == '__main__':
    main(sys.argv[1])
