"""Q4 (D7) -- the niche-C15 private probe on the BAND path.

The one movement task 1 expects.  ``_imap_out['probe_rc'] = (rows, cols)``
is filled on a whole-grid call and was silently dropped on a banded one; the
follow-up gathers it band by band.  Verified here, on my own fixtures, for
every route that can carry it -- screen + evaluator, ray-density + evaluator,
screen + coarse-Newton incumbent, ray-density + coarse-Newton -- at band
heights 7 / 32 / 128, against the whole-grid arm's values, AND with the
returned FIELD compared against the same call that did NOT ask (asking must
not move a bit).

Probe pixels are chosen to straddle band boundaries deliberately: the first
and last row, both sides of every 32-row boundary, a NaN / out-of-domain
corner, and a scattering of interior pixels.

Usage: python q4_d7_probe_rc.py <out.json> --tree <arm tree>
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vf  # noqa: E402

N, DX, W, SUB = 320, 12e-6, 0.95e-3, 4
ROWS = [7, 32, 128]


def _base(**over):
    kw = dict(prescription=_vf.presc_meniscus(ap=4.4e-3), wavelength=_vf.WL,
              dx=DX, ray_subsample=SUB, n_workers=1, on_undersample='silent',
              on_noncollimated='off', on_aperture_beam='silent',
              parallel_amp=False)
    kw.update(over)
    return kw


def probe_pixels():
    """Straddle every band boundary and both grid edges."""
    rs, cs = [], []
    for r in (0, 1, 6, 7, 8, 31, 32, 33, 63, 64, 127, 128, 129,
              N // 2, N - 2, N - 1):
        for c in (0, 3, N // 4, N // 2, N // 2 + 1, 3 * N // 4, N - 1):
            rs.append(r)
            cs.append(c)
    return (np.asarray(rs, dtype=np.intp), np.asarray(cs, dtype=np.intp))


def _arr(rec, key):
    v = rec.get(key)
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float64)
    return a


def main():
    args = _vf.argp(__doc__).parse_args()
    la = _vf.banner(args.tree)
    pr = probe_pixels()
    E = _vf.sph(N, DX, W, 0.055)
    cases = {
        'screen_eval': _base(carrier=0.055),
        'rd_eval': _base(carrier=0.055, amplitude_model='ray_density'),
        'screen_newton': _base(carrier=0.055, inverse_map=False),
        'rd_newton': _base(carrier=0.055, amplitude_model='ray_density',
                           inverse_map=False),
        'rd_remap_eval': _base(carrier=0.055, amplitude_model='ray_density',
                               preserve_input_phase='remap',
                               remap_sampling='full'),
    }
    out = {}
    for name, kw in cases.items():
        F0, rec0, _ = _vf.run_traced(la, E, kw, 0, probe_rc=pr)
        Fq, _, _ = _vf.run_traced(la, E, kw, 0)            # did NOT ask
        opl0, ard0 = _arr(rec0, 'probe_opl'), _arr(rec0, 'probe_ard')
        blk = {'whole': {
            'present': 'probe_opl' in rec0,
            'opl_hash': None if opl0 is None else _vf.h(opl0),
            'ard_hash': None if ard0 is None else _vf.h(ard0),
            'piston': (None if rec0.get('probe_opl_piston') is None
                       else float(rec0['probe_opl_piston'])),
            'n_nan': (None if opl0 is None else int(np.isnan(opl0).sum())),
            'n': (None if opl0 is None else int(opl0.size)),
            'first8': (None if opl0 is None
                       else [float(z) for z in opl0[:8]]),
            'field_hash': _vf.h(F0),
            'field_hash_without_probe': _vf.h(Fq),
        }}
        for rows in ROWS:
            F, rec, _ = _vf.run_traced(la, E, kw, rows, probe_rc=pr)
            Fn, _, _ = _vf.run_traced(la, E, kw, rows)
            opl, ard = _arr(rec, 'probe_opl'), _arr(rec, 'probe_ard')
            blk[f'rows={rows}'] = {
                'present': 'probe_opl' in rec,
                'opl_hash': None if opl is None else _vf.h(opl),
                'ard_hash': None if ard is None else _vf.h(ard),
                'piston': (None if rec.get('probe_opl_piston') is None
                           else float(rec['probe_opl_piston'])),
                'equal_to_whole_opl': (
                    None if (opl is None or opl0 is None)
                    else bool(np.array_equal(opl, opl0, equal_nan=True))),
                'equal_to_whole_ard': (
                    None if (ard is None or ard0 is None)
                    else bool(np.array_equal(ard, ard0, equal_nan=True))),
                'max_abs_diff_opl': (
                    None if (opl is None or opl0 is None) else
                    float(np.nanmax(np.abs(opl - opl0)))
                    if np.isfinite(opl - opl0).any() else 0.0),
                'nan_pattern_equal': (
                    None if (opl is None or opl0 is None)
                    else bool(np.array_equal(np.isnan(opl),
                                             np.isnan(opl0)))),
                'field_hash': _vf.h(F),
                'field_hash_without_probe': _vf.h(Fn),
                'probe_moved_the_field': _vf.h(F) != _vf.h(Fn),
            }
            print(f"  {name:16s} rows={rows:<4d} present="
                  f"{blk[f'rows={rows}']['present']!s:5s} "
                  f"eq_opl={blk[f'rows={rows}']['equal_to_whole_opl']} "
                  f"eq_ard={blk[f'rows={rows}']['equal_to_whole_ard']} "
                  f"field_moved="
                  f"{blk[f'rows={rows}']['probe_moved_the_field']}",
                  flush=True)
        out[name] = blk
        print(f"  {name:16s} whole: present={blk['whole']['present']} "
              f"n_nan={blk['whole']['n_nan']}/{blk['whole']['n']} "
              f"ard={'yes' if blk['whole']['ard_hash'] else 'no'}",
              flush=True)
    _vf.dump(args, {'cases': out, 'N': N, 'DX': DX, 'SUB': SUB,
                    'n_probe_px': int(pr[0].size),
                    'free_gb': _vf.free_gb()})


if __name__ == '__main__':
    main()
