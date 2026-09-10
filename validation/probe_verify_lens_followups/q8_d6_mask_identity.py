"""Q8 (D6) -- the cached pass-2 domain mask IS the mask it replaced.

The D6 fix hands pass 1's mask to pass 2 instead of recomputing it, on the
argument that ``eval_into`` writes each channel independently so pass 2's
CH_X_IN / CH_Y_IN are bit for bit pass 1's.  Field identity (q2) is evidence
for that; this is the direct test.

``InverseCharacteristic.domain_mask`` is wrapped to record a hash of every
mask it returns, in order, with the band it was asked about.  On v5.44.0 the
banded ray-density + evaluator route calls it 2 x n_bands times; on the
follow-up branch, n_bands times.  The claim is exact and checkable:

* v5.44.0's pass-1 sequence == v5.44.0's pass-2 sequence (the duplicate), and
* the follow-up branch's single sequence == BOTH of them, and
* the whole-grid arm's ONE mask == the concatenation of the bands.

Also recorded: the call count and the number of pixels tested, in whole
grids -- the load-free half of the D6 claim.

Usage: python q8_d6_mask_identity.py <out.json> --tree <arm tree>
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vf  # noqa: E402

N, DX, W, SUB = 384, 12e-6, 1.0e-3, 4


def _kw(model, inv=True):
    return dict(prescription=_vf.presc_meniscus(ap=4.4e-3),
                wavelength=_vf.WL, dx=DX, ray_subsample=SUB, n_workers=1,
                on_undersample='silent', on_noncollimated='off',
                on_aperture_beam='silent', parallel_amp=False,
                carrier=0.055, amplitude_model=model, inverse_map=inv)


def main():
    args = _vf.argp(__doc__).parse_args()
    la = _vf.banner(args.tree)
    from lumenairy.elements import _lens_imap as IM
    IC = IM.InverseCharacteristic
    real = IC.domain_mask
    log = []

    def wrapped(slf, Xg, Yg, x_in=None, y_in=None, axes=None, relax=0.0):
        r = real(slf, Xg, Yg, x_in=x_in, y_in=y_in, axes=axes, relax=relax)
        a = np.ascontiguousarray(np.asarray(r))
        log.append({'shape': list(a.shape),
                    'hash': hashlib.sha256(a.tobytes()).hexdigest()[:24],
                    'n_true': int(a.sum()), 'n': int(a.size)})
        return r

    IC.domain_mask = wrapped
    out = {}
    try:
        E = _vf.sph(N, DX, W, 0.055)
        for model in ('ray_density', 'screen'):
            for rows in (0, 32, 7):
                log.clear()
                F, rec, _ = _vf.run_traced(la, E, _kw(model), rows)
                g = float(N) * N
                seq = [d['hash'] for d in log]
                blk = {'n_calls': len(log),
                       'pixels_grids': round(sum(d['n'] for d in log) / g, 4),
                       'n_true_total': sum(d['n_true'] for d in log),
                       'seq': seq,
                       'field_hash': _vf.h(F),
                       'engaged': bool(rec.get('engaged')),
                       'n_out_of_domain': rec.get('n_out_of_domain')}
                if rows:
                    nb = int(np.ceil(N / rows))
                    blk['n_bands'] = nb
                    blk['calls_per_band'] = round(len(log) / nb, 3)
                    if len(log) == 2 * nb:
                        blk['pass1_eq_pass2'] = bool(seq[:nb] == seq[nb:])
                    else:
                        blk['pass1_eq_pass2'] = None
                    # the concatenated band masks, as ONE hash over the grid
                    blk['first_pass_concat_hash'] = hashlib.sha256(
                        ''.join(seq[:nb]).encode()).hexdigest()[:24]
                out[f'{model}|rows={rows}'] = blk
                print(f"  {model:12s} rows={rows:<4d} calls={len(log):<4d} "
                      f"pixels={blk['pixels_grids']:.3f} grids  "
                      f"true={blk['n_true_total']}  "
                      f"p1==p2:{blk.get('pass1_eq_pass2')}  "
                      f"field={blk['field_hash']}", flush=True)
    finally:
        IC.domain_mask = real
    _vf.dump(args, {'cases': out, 'N': N, 'DX': DX, 'SUB': SUB,
                    'free_gb': _vf.free_gb()})


if __name__ == '__main__':
    main()
