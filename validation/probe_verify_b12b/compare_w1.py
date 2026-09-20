"""VERIFY-WP-B12b -- pair probe W1's ``pre`` and ``post`` JSONs for one build.

Reads the two arm files and closes the ray-level claim end to end:

* POSITION.  the ``pre`` tree's returned base-ray position against my own
  tracer's exit-vertex state, and against the predicted miss
  ``(sag_true - sag_inline) * u``; the ``post`` tree's against the same
  state, where it must sit at the trace's own floor.
* OPTICAL PATH.  ``arg(amp_pre * conj(amp_post) * exp(-i k0 * pred))`` is a
  GLOBAL PISTON when the prediction is right, so the residual is reported
  after a circular mean is removed -- no unwrapping, and a wrong prediction
  cannot hide inside a wrap.  ``pred`` is
  ``(n_exit * sign(N) * sag_true - sag_inline) * sec``, taken from my own 3-D
  trace and the transcribed in-line copy, never from the library.

Usage::

    python validation/probe_verify_b12b/compare_w1.py <build_tag>

Author: VERIFY-WP-B12b
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def load(arm, tag):
    p = os.path.join(_HERE, f'probe_w1_mechanism_{arm}_{tag}.json')
    with open(p, encoding='cp1252') as fh:
        return json.load(fh)


def main(tag):
    pre = load('pre', tag)
    post = load('post', tag)
    assert pre['arm'] == 'pre' and post['arm'] == 'post'
    out = dict(tag=tag, pre_file=pre['env']['lumenairy_file'],
               post_file=post['env']['lumenairy_file'], rows=[])
    print(f"PRE  tree: {out['pre_file']}")
    print(f"POST tree: {out['post_file']}")
    print(f"{'fixture':15s} {'sagerr/wv':>10s} {'frac':>6s} {'OPLerr/wv':>10s} "
          f"{'pos pre-vs-oracle':>18s} {'pred':>11s} {'OPL resid/wv':>13s} "
          f"{'pos post-vs-oracle':>19s}")
    for a, b in zip(pre['rows'], post['rows']):
        assert a['key'] == b['key']
        k0 = 2.0 * np.pi / a['lam']
        row = dict(
            key=a['key'], note=a['note'],
            sag_true_waves=a['sag_true_waves'],
            sag_inline_err_waves=a['sag_inline_err_waves'],
            sag_inline_err_frac=a['sag_inline_err_frac'],
            inline_opl_err_waves=a['inline_opl_err_waves'],
            n_exit=a['n_exit'], exit_sign=a['exit_sign'],
            pos_pre_vs_oracle=a.get('pos_vs_oracle_max'),
            pos_post_vs_oracle=b.get('pos_vs_oracle_max'),
            pos_pred=a.get('pred_pos_err_m'))
        if 'amp_re' in a and 'amp_re' in b:
            ampp = np.array(a['amp_re']) + 1j * np.array(a['amp_im'])
            ampq = np.array(b['amp_re']) + 1j * np.array(b['amp_im'])
            pred = np.array(a['pred_opl_err'])
            ph = np.angle(ampp * np.conj(ampq) * np.exp(-1j * k0 * pred))
            piston = np.angle(np.mean(np.exp(1j * ph)))
            resid = np.angle(np.exp(1j * (ph - piston)))
            row['opl_resid_waves'] = float(np.nanmax(np.abs(resid))
                                           / (2.0 * np.pi))
            row['opl_pred_pv_waves'] = float((np.nanmax(pred)
                                              - np.nanmin(pred)) / a['lam'])
            row['piston_waves'] = float(piston / (2.0 * np.pi))
            # the no-prediction control: the same residual with pred = 0,
            # i.e. how big the phase move IS.
            ph0 = np.angle(ampp * np.conj(ampq))
            p0 = np.angle(np.mean(np.exp(1j * ph0)))
            row['opl_move_waves_nopred'] = float(
                np.nanmax(np.abs(np.angle(np.exp(1j * (ph0 - p0)))))
                / (2.0 * np.pi))
        out['rows'].append(row)
        print(f"{row['key']:15s} {row['sag_inline_err_waves']:10.4f} "
              f"{row['sag_inline_err_frac']:6.3f} "
              f"{row['inline_opl_err_waves']:10.4f} "
              f"{row['pos_pre_vs_oracle']:18.4e} {row['pos_pred']:11.4e} "
              f"{row.get('opl_resid_waves', float('nan')):13.3e} "
              f"{row['pos_post_vs_oracle']:19.4e}")
    p = os.path.join(_HERE, f'compare_w1_{tag}.json')
    with open(p, 'w', encoding='cp1252') as fh:
        json.dump(out, fh, indent=1)
    print('WROTE', p)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'win32_314')
