"""WP-C5 item 3 -- the replica fill, and the proof that the zeroing is confined.

What this measures, in the order the claims depend on each other:

  * ``faithful``   -- a window at or inside one period: the two fills must be
                      the SAME OBJECT, not merely equal arrays;
  * ``oversized``  -- a window past one period: inside the period the two
                      fills must be byte-identical, outside it ``'zero'`` must
                      be exactly zero, and the boundary must be the period the
                      transform itself reports through ``_period_out``;
  * ``keys``       -- what the readout publishes about the window it returned;
  * ``refusal``    -- a matrix of windows x fills x dispositions, so "the
                      refusal is unchanged" is a comparison of two censuses
                      rather than an assertion about code;
  * ``digests``    -- the archive-to-archive reading over both public readouts,
                      the chain and the multi entry point.

Usage (BLAS pinned on the COMMAND LINE)::

    PYTHONPATH=<tree> python validation/probe_c5_three_defaults/c5_item3_replica.py OUT.json
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                             # noqa: E402

from _digest import dig, write                                 # noqa: E402

import lumenairy as la                                         # noqa: E402
from lumenairy.propagators import carrier as C                 # noqa: E402

WL = 1.064e-6
RMAG = -20.0e-3


def gauss_pupil(n=512, dx=4.0e-6, w=0.5e-3):
    x = (np.arange(n) - n / 2) * dx
    E = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
               / w ** 2).astype(np.complex128)
    return E, dx, w


def read(env, dx, dx_out, n_out, **kw):
    pd = {}
    kw.setdefault('on_replica', 'error')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        F = C.carrier_referenced_focus_readout(
            env, RMAG, -RMAG, WL, dx, dx_out=dx_out, N_out=n_out,
            _period_out=pd, **kw)
    return np.asarray(F), pd


def one_period(env, dx, w):
    _F, pd = read(env, dx, (WL * abs(RMAG) / (np.pi * w)) / 8.0, 16)
    return min(pd['period'])


def confinement(F_rep, F_zero, period, dx_out, n, centre=(0.0, 0.0)):
    """The geometry the claim is about, computed from the REPORTED period."""
    u = (np.arange(n) - n / 2.0) * float(dx_out)
    inx = np.abs(u + centre[0]) <= 0.5 * period[0] * (1 + 1e-9)
    iny = np.abs(u + centre[1]) <= 0.5 * period[1] * (1 + 1e-9)
    mask = np.logical_and(iny[:, None], inx[None, :])
    inside_equal = bool(np.array_equal(F_rep[mask], F_zero[mask]))
    outside_zero = float(np.max(np.abs(F_zero[~mask]))) if (~mask).any() \
        else None
    outside_rep = float(np.max(np.abs(F_rep[~mask]))) if (~mask).any() else None
    return {'n_inside': int(mask.sum()), 'n_outside': int((~mask).sum()),
            'faithful_x': int(inx.sum()), 'faithful_y': int(iny.sum()),
            'inside_byte_identical': inside_equal,
            'outside_max_abs_zero': outside_zero,
            'outside_max_abs_repeat': outside_rep}


def main(out):
    import inspect
    res = {'signature_defaults': {
        fn.__name__: inspect.signature(fn).parameters['replica_fill'].default
        for fn in (C.carrier_referenced_focus_readout,
                   C.carrier_referenced_exact_focus_readout)}}
    res['signature_defaults']['_collins_focus_readout'] = (
        inspect.signature(C._collins_focus_readout)
        .parameters['replica_fill'].default)
    digests = {}

    env, dx, w = gauss_pupil()
    per = one_period(env, dx, w)

    # --- (a) a faithful window -------------------------------------------
    dxo_f = (WL * abs(RMAG) / (np.pi * w)) / 8.0
    fa = {}
    for fill in ('repeat', 'zero'):
        F, pd = read(env, dx, dxo_f, 32, replica_fill=fill)
        fa[fill] = {'faithful_samples': list(pd['faithful_samples']),
                    'published_fill': pd.get('replica_fill'),
                    'digest': dig(F)}
        digests['faithful/%s' % fill] = dig(F)
    F0, pd0 = read(env, dx, dxo_f, 32)
    fa['default'] = {'digest': dig(F0),
                     'published_fill': pd0.get('replica_fill')}
    digests['faithful/default'] = dig(F0)
    fa['window_over_period'] = 32 * dxo_f / per
    # identity, not equality: the same object comes back
    fa['identity_short_circuit'] = bool(
        C._fill_readout_replicas(F0, pd0['period'], dxo_f, 32,
                                 (0.0, 0.0), 'zero') is F0)
    res['faithful'] = fa

    # --- (b) an oversized window -----------------------------------------
    ov = {}
    for ratio in (1.10, 1.60, 2.20):
        n = 256
        dxo = per * ratio / n
        got = {}
        for fill in ('repeat', 'zero'):
            F, pd = read(env, dx, dxo, n, on_replica='ignore',
                         replica_fill=fill)
            got[fill] = (F, pd)
            digests['oversized/%.2f/%s' % (ratio, fill)] = dig(F)
        Fd, pdd = read(env, dx, dxo, n, on_replica='ignore')
        digests['oversized/%.2f/default' % ratio] = dig(Fd)
        row = confinement(got['repeat'][0], got['zero'][0],
                          got['repeat'][1]['period'], dxo, n)
        row['default_equals_zero'] = bool(np.array_equal(Fd, got['zero'][0]))
        row['default_equals_repeat'] = bool(
            np.array_equal(Fd, got['repeat'][0]))
        row['reported_period'] = list(got['repeat'][1]['period'])
        row['faithful_samples'] = list(got['zero'][1]['faithful_samples'])
        row['published_fill'] = {
            f: got[f][1].get('replica_fill') for f in ('repeat', 'zero')}
        # the boundary as the arithmetic of the REPORTED period
        row['expected_faithful_x'] = int(
            2 * int(np.floor(0.5 * row['reported_period'][0] / dxo
                             * (1 + 1e-9))) + 1)
        ov['%.2f' % ratio] = row
    res['oversized'] = ov

    # --- (c) an OFF-AXIS window: the zone is centred on the FIELD's origin
    n, ratio = 256, 1.60
    dxo = per * ratio / n
    cen = (0.30 * per, 0.0)
    off = {}
    for fill in ('repeat', 'zero'):
        F, pd = read(env, dx, dxo, n, on_replica='ignore',
                     replica_fill=fill, centre_out=cen)
        off[fill] = (F, pd)
        digests['offaxis/%s' % fill] = dig(F)
    res['off_axis'] = confinement(off['repeat'][0], off['zero'][0],
                                  off['repeat'][1]['period'], dxo, n,
                                  centre=cen)
    res['off_axis']['faithful_samples'] = list(
        off['zero'][1]['faithful_samples'])

    # --- (d) the refusal census -------------------------------------------
    cen_matrix = []
    for ratio in (0.50, 0.98, 1.00, 1.02, 1.60, 2.20):
        n = 128
        dxo = per * ratio / n
        for fill in ('repeat', 'zero'):
            try:
                read(env, dx, dxo, n, on_replica='error', replica_fill=fill)
                verdict = 'served'
            except RuntimeError as exc:
                verdict = 'refused:' + ('ALIASES' in str(exc) and 'ALIASES'
                                        or 'other')
            cen_matrix.append({'ratio': ratio, 'fill': fill,
                               'verdict': verdict})
    res['refusal_matrix'] = cen_matrix

    # --- (e) the exact readout --------------------------------------------
    n_e, dx_e, w_e, R_e = 512, 0.5e-6, 30e-6, -0.2e-3
    xe = (np.arange(n_e) - n_e // 2) * dx_e
    r2 = xe[:, None] ** 2 + xe[None, :] ** 2
    S = np.sign(R_e) * (np.sqrt(r2 + R_e * R_e) - abs(R_e))
    Ee = (np.exp(-r2 / w_e ** 2)
          * np.exp(1j * 2.0 * np.pi / WL * S)).astype(np.complex128)
    exact = {}
    for fill in ('repeat', 'zero'):
        pd = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            F = la.carrier_referenced_exact_focus_readout(
                Ee, R_e, -R_e, WL, dx_e, dx_out=0.05e-6, N_out=4096,
                window_factor=4.0, on_replica='ignore', replica_fill=fill,
                _period_out=pd)
        exact[fill] = (np.asarray(F), pd)
        digests['exact/%s' % fill] = dig(F)
    pdd = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Fd = la.carrier_referenced_exact_focus_readout(
            Ee, R_e, -R_e, WL, dx_e, dx_out=0.05e-6, N_out=4096,
            window_factor=4.0, on_replica='ignore', _period_out=pdd)
    digests['exact/default'] = dig(Fd)
    res['exact_readout'] = confinement(
        exact['repeat'][0], exact['zero'][0], exact['repeat'][1]['period'],
        0.05e-6, 4096)
    res['exact_readout']['default_equals_zero'] = bool(
        np.array_equal(np.asarray(Fd), exact['zero'][0]))
    res['exact_readout']['published_fill'] = pdd.get('replica_fill')
    res['exact_readout']['faithful_samples'] = list(
        pdd.get('faithful_samples', ()))

    # --- (f) a faithful window on the EXACT readout ------------------------
    for fill in ('repeat', 'zero'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            F = la.carrier_referenced_exact_focus_readout(
                Ee, R_e, -R_e, WL, dx_e, dx_out=0.05e-6, N_out=512,
                window_factor=4.0, replica_fill=fill)
        digests['exact_faithful/%s' % fill] = dig(F)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        F = la.carrier_referenced_exact_focus_readout(
            Ee, R_e, -R_e, WL, dx_e, dx_out=0.05e-6, N_out=512,
            window_factor=4.0)
    digests['exact_faithful/default'] = dig(F)

    res['digests'] = digests
    write(out, res)


if __name__ == '__main__':
    main(sys.argv[1])
