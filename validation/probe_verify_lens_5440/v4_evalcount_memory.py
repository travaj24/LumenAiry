"""V4 -- the 7/4 EVALUATION count and the MEMORY claim.

(a) Counts ``InverseCharacteristic.eval_into`` calls, channel-evaluations and
    channel-PIXELS on the banded ray-density branch vs the whole-grid branch
    (and on the banded SCREEN branch, which the CHANGELOG says is one pass).
    The claim under test is "the model is evaluated 7/4 times" on the
    ray-density branch.

(b) Whole-call ``tracemalloc`` peak of a banded ray-density call vs the
    whole-grid path -- on 5.44.0 and, for the whole-grid arm, on v5.43.0.

Usage:  python v4_evalcount_memory.py <out.json> [N] [sub] [dx_um]
"""
from __future__ import annotations

import json
import os
import sys
import time
import tracemalloc
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fix  # noqa: E402


def _kw(dx, sub, model='ray_density', **over):
    kw = dict(prescription=_fix.presc_singlet(ap=4.4e-3), wavelength=_fix.WL,
              dx=dx, ray_subsample=sub, n_workers=1, on_undersample='silent',
              on_noncollimated='off', on_aperture_beam='silent',
              parallel_amp=False, amplitude_model=model)
    kw.update(over)
    return kw


def count_evals(la, N, dx, sub):
    from lumenairy.elements import _lens_imap as IM
    IC = IM.InverseCharacteristic
    orig = IC.eval_into
    log = []

    def counted(self, Xg, Yg, out, channels=None, chunk=None):
        ch = list(range(4)) if channels is None else list(channels)
        log.append((int(np.size(Xg)), len(ch), tuple(ch)))
        return orig(self, Xg, Yg, out, channels=channels, chunk=chunk)

    IC.eval_into = counted
    res = {}
    try:
        E = _fix.gauss(N, dx, 1.0e-3)
        for tag, rows, model in (('rd_whole', 0, 'ray_density'),
                                 ('rd_band32', 32, 'ray_density'),
                                 ('rd_band7', 7, 'ray_density'),
                                 ('screen_whole', 0, 'screen'),
                                 ('screen_band32', 32, 'screen')):
            log.clear()
            f, rec, _ = _fix.run(la, E, _kw(dx, sub, model), rows)
            npix = N * N
            chan_px = sum(n * c for n, c, _ in log)
            res[tag] = {'calls': len(log),
                        'channel_evals_per_pixel': chan_px / npix,
                        'total_pixels_visited': sum(n for n, _, _ in log)
                                                / npix,
                        'channel_sets': sorted(set(t for _, _, t in log)),
                        'engaged': bool(rec.get('engaged', False)),
                        'hash': _fix.h(f)}
            print(f"  {tag:14s} calls={len(log):4d} "
                  f"chan-evals/pixel={res[tag]['channel_evals_per_pixel']:.3f} "
                  f"sets={res[tag]['channel_sets']}", flush=True)
    finally:
        IC.eval_into = orig
    return res


def peak(la, N, dx, sub, rows, model='ray_density', **over):
    from lumenairy.elements import _lens_imap as IM
    E = _fix.gauss(N, dx, 1.0e-3)
    kw = _kw(dx, sub, model, **over)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        IM.inverse_map_cache_clear()
        la.apply_real_lens_traced(E, sag_chunk_rows=rows, **kw)      # warm
        IM.inverse_map_cache_clear()
        rec = {}
        t = time.time()
        tracemalloc.start()
        tracemalloc.reset_peak()
        la.apply_real_lens_traced(E, sag_chunk_rows=rows, _imap_out=rec, **kw)
        _, pk = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        secs = time.time() - t
        IM.inverse_map_cache_clear()
    return pk, bool(rec.get('engaged', False)), secs


def main():
    la = _fix.banner()
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    sub = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    dx = (float(sys.argv[4]) if len(sys.argv) > 4 else 6.0) * 1e-6
    try:
        import psutil
        print(f"# free RAM {psutil.virtual_memory().available / 2**30:.1f} GB",
              flush=True)
    except ImportError:
        pass
    out = {'version': la.__version__, 'N': N, 'sub': sub, 'dx': dx}
    if '--peaks-only' not in sys.argv:
        print(f'--- eval counts (N={N}, sub={sub}) ---', flush=True)
        out['eval_counts'] = count_evals(la, N, dx, sub)
    grid = 8.0 * N * N
    print(f'--- tracemalloc peaks (1 grid = {grid / 2**20:.1f} MiB) ---',
          flush=True)
    out['peaks'] = {}
    for tag, rows, model in (('rd_whole', 0, 'ray_density'),
                             ('rd_band', 32, 'ray_density'),
                             ('screen_whole', 0, 'screen'),
                             ('screen_band', 32, 'screen')):
        try:
            pk, eng, secs = peak(la, N, dx, sub, rows, model)
        except Exception as e:                       # v5.43.0 has no banded rd
            out['peaks'][tag] = {'error': f'{type(e).__name__}: {e}'[:200]}
            print(f"  {tag:14s} ERROR {type(e).__name__}", flush=True)
            continue
        out['peaks'][tag] = {'peak_bytes': pk, 'grids': pk / grid,
                             'engaged': eng, 'secs': round(secs, 2)}
        print(f"  {tag:14s} peak={pk / 2**20:8.1f} MiB = "
              f"{pk / grid:6.2f} grids  eng={eng}  {secs:.1f}s", flush=True)
    with open(sys.argv[1], 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', sys.argv[1])


if __name__ == '__main__':
    main()
