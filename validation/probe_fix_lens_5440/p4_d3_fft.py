"""P4 (D3) -- is the SINGLE-PRECISION transform pair acceptable on a REAL chain?

VERIFY_LENS_BANDED_COMPLEX64_2026_09_10 D3: on numpy >= 2.0
``np.fft.fft2(complex64)`` returns complex64, so for a complex64 envelope BOTH
transforms of ``_fourier_upsample_crop``'s pair run in single precision -- not
"in complex128, narrowed back on return" as the test docstring and the
CHANGELOG say.  Measured 2.6 x eps32 x peak on synthetic samples.

The DECISION needs the number on a real chain, against the campaign's 4e-05
relative energy bar (field side 2e-05, since power is quadratic).  Three arms
on the SAME two-group traced carrier chain WITH an exact focus readout (the
stage that actually calls the crop -- ``_fine_trace_group_exit`` and
``carrier_referenced_exact_focus_readout``):

  A  complex64 envelope, SHIPPED crop (single-precision transform pair)
  B  complex64 envelope, crop forced to transform in complex128 and NARROWED
     ONCE on return (what the docstring claimed)
  C  complex128 envelope -- the reference both are scored against

Reported: rel L2 and rel total power of A and B against C, and A against B
(the transform's own contribution), plus the same on the readout-free chain
for control.

Usage:  python p4_d3_fft.py <out.json>
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import _fixp                                                   # noqa: E402

WL = _fixp.WL


def _singlet(R1, R2, d, glass, ap, name):
    def s(r, gb, ga):
        return {'radius': r, 'glass_before': gb, 'glass_after': ga,
                'conic': 0.0, 'radius_y': None, 'conic_y': None,
                'aspheric_coeffs': None, 'aspheric_coeffs_y': None}
    return {'name': name, 'aperture_diameter': ap, 'thicknesses': [d],
            'surfaces': [s(R1, 'air', glass), s(R2, glass, 'air')]}


G1 = _singlet(55.0e-3, -55.0e-3, 3.0e-3, 'N-BK7', 13.0e-3, 'g1')
G2 = _singlet(40.0e-3, -1e12, 2.5e-3, 'N-SF11', 13.0e-3, 'g2')


def rel(a, b):
    a = np.asarray(a).astype(np.complex128)
    b = np.asarray(b).astype(np.complex128)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def relp(a, b):
    pa = float(np.sum(np.abs(np.asarray(a).astype(np.complex128)) ** 2))
    pb = float(np.sum(np.abs(np.asarray(b).astype(np.complex128)) ** 2))
    return abs(pa - pb) / pb


def main():
    la = _fixp.banner()
    from lumenairy.propagators import carrier as C
    prev = la.get_fft_auto_promote()
    la.set_fft_auto_promote(False)
    N, dx, w = 512, 26e-6, 3.6e-3
    x = (np.arange(N) - N // 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    env = np.exp(-r2 / w ** 2).astype(np.complex128)
    groups = [{'prescription': G1, 'gap_before': 0.0},
              {'prescription': G2, 'gap_before': 12.0e-3}]
    real_crop = C._fourier_upsample_crop
    n_crop_calls = [0]

    def crop_c128(e, n_crop, n_fine):
        """The docstring's claim, implemented: transform in complex128 and
        narrow ONCE on return."""
        n_crop_calls[0] += 1
        a = np.asarray(e)
        out = real_crop(a.astype(np.complex128), n_crop, n_fine)
        return np.asarray(out).astype(a.dtype, copy=False)

    def counted(e, n_crop, n_fine):
        n_crop_calls[0] += 1
        return real_crop(e, n_crop, n_fine)

    def run(e, patch, **over):
        n_crop_calls[0] = 0
        C._fourier_upsample_crop = patch
        try:
            kw = dict(r_in=55e-3, ray_subsample=8, n_workers=1,
                      traced_kwargs=dict(on_undersample='silent',
                                         on_noncollimated='silent'))
            kw.update(over)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = la.propagate_traced_carrier_chain(e, groups, WL, dx,
                                                        **kw)
        finally:
            C._fourier_upsample_crop = real_crop
        return np.asarray(res.field), n_crop_calls[0]

    out = {'version': la.__version__, 'file': la.__file__, 'chains': {}}
    ro = dict(focus_readout=dict(dx_out=0.30e-6, N_out=96,
                                 window_factor=4.0, n_fine_cap=4096),
              final_leg='exact', on_tilt_exact_grid='warn',
              on_ram_cap='ignore')
    for tag, over in (('chain2_exact_readout', ro), ('chain2_plain', {})):
        C_ref, n_c = run(env, counted, **over)
        A, n_a = run(env.astype(np.complex64), counted, **over)
        B, n_b = run(env.astype(np.complex64), crop_c128, **over)
        d = {'crop_calls': n_a, 'dtype_A': str(A.dtype),
             'dtype_C': str(C_ref.dtype),
             'A_vs_C_rel_l2': rel(A, C_ref),
             'A_vs_C_rel_power': relp(A, C_ref),
             'B_vs_C_rel_l2': rel(B, C_ref),
             'B_vs_C_rel_power': relp(B, C_ref),
             'A_vs_B_rel_l2': rel(A, B),
             'A_equals_B': bool(np.array_equal(A, B)),
             'field_bar_2e_5_met_A': bool(rel(A, C_ref) <= 2e-5),
             'energy_bar_4e_5_met_A': bool(relp(A, C_ref) <= 4e-5)}
        out['chains'][tag] = d
        print('%-16s crop calls %d   A(shipped c64) vs C(c128): rel L2 %.3e  '
              'rel P %.3e' % (tag, n_a, d['A_vs_C_rel_l2'],
                              d['A_vs_C_rel_power']), flush=True)
        print('%-16s               B(c128 transform, narrow once) vs C: '
              'rel L2 %.3e  rel P %.3e' % ('', d['B_vs_C_rel_l2'],
                                           d['B_vs_C_rel_power']), flush=True)
        print('%-16s               A vs B (the transform itself): rel L2 %.3e'
              '   identical=%s' % ('', d['A_vs_B_rel_l2'], d['A_equals_B']),
              flush=True)
    # the direct crop, at four sizes, so the N growth is on record
    direct = {}
    rng = np.random.default_rng(5)
    for Nc in (256, 512, 1024, 2048):
        xc = (np.arange(Nc) - Nc / 2) / Nc
        r2c = xc[None, :] ** 2 + xc[:, None] ** 2
        ec = (np.exp(-r2c / 0.05)
              * (1 + 0.05 * rng.standard_normal((Nc, Nc)))).astype(
                  np.complex128)
        a = real_crop(ec.astype(np.complex64), Nc // 2, Nc)
        b = real_crop(ec.astype(np.complex64).astype(np.complex128),
                      Nc // 2, Nc).astype(np.complex64)
        c = real_crop(ec, Nc // 2, Nc)
        direct[str(Nc)] = {
            'max_abs_diff_vs_narrow_once': float(np.abs(
                a.astype(np.complex128) - b.astype(np.complex128)).max()),
            'eps32_x_peak': float(np.finfo(np.float32).eps
                                  * np.abs(c).max()),
            'rel_l2_vs_c128_input': rel(a, c)}
        print('crop %5d->%5d : max|A-B| %.3e  eps32*peak %.3e  '
              'rel L2 vs c128 in %.3e'
              % (Nc // 2, Nc, direct[str(Nc)]['max_abs_diff_vs_narrow_once'],
                 direct[str(Nc)]['eps32_x_peak'],
                 direct[str(Nc)]['rel_l2_vs_c128_input']), flush=True)
    out['direct_crop'] = direct
    la.set_fft_auto_promote(prev)
    with open(sys.argv[1], 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', sys.argv[1])


if __name__ == '__main__':
    main()
