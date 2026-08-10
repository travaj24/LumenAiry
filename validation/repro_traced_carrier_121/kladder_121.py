# THE K-LADDER -- congruence_workers = 1, 2, 3, 4 on the design-121 fan.
# docs/audits/FIX_PERF_PARALLEL_2026_08_10.md sec 4 (audit item #5).
#
# Runs ``fan_multi_121.py`` UNMODIFIED (it is import-safe now, which is what
# makes ``CW>1`` reachable at all) and measures, for one k:
#
#   * wall per order.  ONLY MEANINGFUL AT k=1: under ``congruence_workers``
#     the ``progress`` hook fires in COMPLETION order in the parent (carrier.py
#     passes ``done``, not ``k``), so the per-row gap is an inter-completion
#     interval, not that order's chain time.  ``chainB_per_order_s`` -- total
#     chain-B wall over the order count -- is the k-independent number the
#     ladder is read from, and it is what the table below reports.
#   * peak RSS of the whole process TREE, and the largest single CHILD's peak
#     -- the child peak is the per-worker number the clamp has to price, and
#     it is only observable when k > 1;
#   * the per-order scalars the runner itself computed, plus a sha256 of every
#     order's readout tile and of the whole accumulated field.
#
# THE ACCEPTANCE IS BIT-IDENTITY, NOT SPEED.  niche D8 accumulates in the
# PARENT in ascending k precisely so the complex sum is FP-identical to serial
# (carrier.py).  ``kladder_compare.py`` checks that claim on the artifacts this
# script writes: same field hash, same tile hashes, same scalars, at every k.
# A speed-up bought by a different answer is not a speed-up.
#
# Env consumed HERE (everything else is passed through to the runner):
#   KL_K     int, default 1     -> exported as CW
#   KL_TAG   label for the artifacts, default 'k<K>'
#   KL_OUT   output directory, default <this dir>/_kladder
#   KL_DUMP  'm,n;m,n' orders whose full tile is stored (default: first two)
import hashlib
import json
import os
import sys
import threading
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import numpy as np  # noqa: E402


# --- IMPORT-SAFE BOUNDARY --------------------------------------------------
# THIS DRIVER needs the guard as much as the runner does, and finding out
# cost a k=2 arm.  sys.modules['__main__'] is THIS file -- the fan is
# exec'd under an owned namespace, which does not change that -- so a spawn
# child re-executes THIS body.  Unguarded, the k=2 arm died with the
# library's own bootstrap message (correctly named, correctly attributed to
# the caller) and then HUNG in the executor's shutdown rather than exiting,
# which is a second reason not to rely on the refusal as a workflow.
# Transcript: scratchpad/par/kl8192_k2_UNGUARDED_DRIVER.log.
if __name__ == '__main__':
    K = int(os.environ.get('KL_K', '1'))
    os.environ['CW'] = str(K)
    TAG = os.environ.get('KL_TAG', 'k%d' % K)
    OUT = os.environ.get('KL_OUT', os.path.join(_HERE, '_kladder'))
    os.makedirs(OUT, exist_ok=True)

    _T0 = time.time()

    # --------------------------------------------------------------------------
    # 1. progress marks -- per-order wall.  Wrapping the entry point rather than
    #    the runner's own lambda keeps the runner unmodified AND still lets its
    #    printout through.
    # --------------------------------------------------------------------------
    import lumenairy as la  # noqa: E402

    _MARKS = []
    _STASH = {}
    _orig_multi = la.propagate_traced_carrier_chain_multi


    def _multi(*a, **kw):
        user_prog = kw.get('progress')

        def _prog(k, K_, nm):
            _MARKS.append({'k': int(k), 'K': int(K_), 'name': str(nm),
                           't': time.time() - _T0})
            if user_prog is not None:
                user_prog(k, K_, nm)

        kw['progress'] = _prog
        _STASH['congruence_workers'] = kw.get('congruence_workers')
        _STASH['output_grid'] = dict(kw.get('output_grid') or {})
        t0 = time.time()
        out = _orig_multi(*a, **kw)
        _STASH['chainB_s'] = time.time() - t0
        _MARKS.append({'k': -1, 'K': -1, 'name': '<end>', 't': time.time() - _T0})
        return out


    la.propagate_traced_carrier_chain_multi = _multi

    # --------------------------------------------------------------------------
    # 2. RSS sampler.  TREE total and per-CHILD, because they answer different
    #    questions: the tree total is what the box has to hold, the largest child
    #    is what ``_multi_resolve_workers`` has to price per worker.
    # --------------------------------------------------------------------------
    _peak = {'tree': 0.0, 'tree_t': 0.0, 'parent': 0.0, 'child': 0.0,
             'nchild': 0, 'min_avail': 1e9}
    _stop = threading.Event()


    def _sampler():
        try:
            import psutil
        except ImportError:
            return
        me = psutil.Process()
        while not _stop.is_set():
            try:
                try:
                    mine = me.memory_info().rss
                except psutil.Error:
                    mine = 0
                tree = float(mine)
                biggest = 0.0
                kids = me.children(recursive=True)
                for p in kids:
                    try:
                        r = float(p.memory_info().rss)
                    except psutil.Error:
                        continue
                    tree += r
                    biggest = max(biggest, r)
                g = 2.0 ** 30
                if tree / g > _peak['tree']:
                    _peak['tree'] = tree / g
                    _peak['tree_t'] = time.time() - _T0
                _peak['parent'] = max(_peak['parent'], mine / g)
                _peak['child'] = max(_peak['child'], biggest / g)
                _peak['nchild'] = max(_peak['nchild'], len(kids))
                _peak['min_avail'] = min(_peak['min_avail'],
                                         psutil.virtual_memory().available / g)
            except Exception:
                pass
            _stop.wait(1.0)


    threading.Thread(target=_sampler, daemon=True).start()

    # --------------------------------------------------------------------------
    # 3. run the runner, keeping its namespace (the adjudication's pattern: every
    #    number below is the runner's OWN, not a reimplementation)
    # --------------------------------------------------------------------------
    FAN = os.path.join(_HERE, 'fan_multi_121.py')
    G = {'__name__': '__main__', '__file__': FAN, '__builtins__': __builtins__}
    code, err = 0, ''
    print('=' * 74, flush=True)
    print('K-LADDER %s : CW=%d  NFC=%s  RAMB=%s  KEEP=%s  NOUT=%s'
          % (TAG, K, os.environ.get('NFC', '16384'),
             os.environ.get('RAMB', 'auto'), os.environ.get('KEEP', '<all>'),
             os.environ.get('NOUT', '<auto>')), flush=True)
    print('=' * 74, flush=True)
    try:
        with open(FAN, 'r', encoding='utf-8') as fh:
            _src = fh.read()
        exec(compile(_src, FAN, 'exec'), G)
    except SystemExit as exc:
        code = int(exc.code or 0)
    except BaseException:
        code = 99
        err = traceback.format_exc()
        print(err, flush=True)
    finally:
        _stop.set()
    WALL = time.time() - _T0

    # --------------------------------------------------------------------------
    # 4. artifacts
    # --------------------------------------------------------------------------
    rows, dump = [], {}
    if 'res' in G and 'fwhms' in G:
        F = np.asarray(G['res'].field)
        DXO_ = float(G['DXO'])
        tmark = {m['k']: m['t'] for m in _MARKS if m['k'] >= 0}
        tend = max([m['t'] for m in _MARKS], default=WALL)
        want = os.environ.get('KL_DUMP', '')
        want = ([tuple(int(v) for v in p.split(',')) for p in want.split(';') if p]
                if want else None)
        for k, c in enumerate(G['res'].congruences):
            NT = int(c['tile'])
            r0, c0 = c['tile_origin']
            tile = np.ascontiguousarray(F[max(r0, 0):r0 + NT, max(c0, 0):c0 + NT])
            mx_, my_ = int(G['mx'][k]), int(G['my'][k])
            wall = ((tmark[k + 1] - tmark[k]) if (k + 1) in tmark
                    else (tend - tmark[k]))
            rows.append({
                'k': k, 'order': str(c['name']), 'mx': mx_, 'my': my_,
                'wall_s': round(float(wall), 2),
                'tile_sha256': hashlib.sha256(tile.tobytes()).hexdigest(),
                'tile_origin': [int(r0), int(c0)], 'tile_px': NT,
                'power_in': float(c['power_in']),
                'power_exit': float(c['power_exit']),
                'power_out': float(c['power_out']),
                'throughput': float(c['throughput']),
                'capture': float(c['capture']),
                'cellP': float(G['cellP'][k]),
                'field_pct': float(G['cell_share'][k]) * 100.0,
                'fwhm_um': float(G['fwhms'][k]) * 1e6,
                'ee3': float(G['ee3s'][k]) * 100.0,
                'ee6': float(G['ee6s'][k]) * 100.0,
                'ee12': float(G['ee12s'][k]) * 100.0,
                'chief_x_um': float(c['chief_ray'][0]) * 1e6,
                'chief_y_um': float(c['chief_ray'][1]) * 1e6})
            if (want is None and k < 2) or (want is not None
                                            and (mx_, my_) in want):
                dump['tile_%d_%d' % (mx_, my_)] = tile
        doc = {
            'tag': TAG, 'CW': K, 'exit': code, 'err': err,
            'nfc': int(os.environ.get('NFC', '16384')),
            'ramb': os.environ.get('RAMB', 'auto'),
            'keep': os.environ.get('KEEP', '<all>'),
            'nout': int(G['NOUT']), 'dxo': DXO_,
            'orders': len(rows),
            'wall_s': round(WALL, 2),
            'chainB_s': round(float(_STASH.get('chainB_s', float('nan'))), 2),
            'chainB_per_order_s': round(float(_STASH.get('chainB_s', 0.0))
                                        / max(1, len(rows)), 2),
            'congruence_workers_passed': _STASH.get('congruence_workers'),
            'output_grid': {kk: (None if vv == float('inf') else vv)
                            for kk, vv in _STASH.get('output_grid', {}).items()},
            'peak_tree_gb': round(_peak['tree'], 3),
            'peak_tree_t_s': round(_peak['tree_t'], 1),
            'peak_parent_gb': round(_peak['parent'], 3),
            'peak_child_gb': round(_peak['child'], 3),
            'max_children': _peak['nchild'],
            'min_avail_gb': round(_peak['min_avail'], 3),
            'field_sha256': hashlib.sha256(
                np.ascontiguousarray(F).tobytes()).hexdigest(),
            'field_shape': list(F.shape),
            'marks': _MARKS, 'rows': rows}
        with open(os.path.join(OUT, '%s.json' % TAG), 'w',
                  encoding='cp1252', errors='replace') as fh:
            json.dump(doc, fh, indent=1, default=str)
        if dump:
            np.savez(os.path.join(OUT, '%s_tiles.npz' % TAG), **dump)
        print("\nK-LADDER %s: CW=%d  orders=%d  chainB %.1f s (%.1f s/order)  "
              "wall %.1f s\n  peak tree %.2f GB (parent %.2f, largest child %.2f, "
              "children %d)  min avail %.2f GB  exit=%d"
              % (TAG, K, len(rows), doc['chainB_s'], doc['chainB_per_order_s'],
                 WALL, doc['peak_tree_gb'], doc['peak_parent_gb'],
                 doc['peak_child_gb'], doc['max_children'], doc['min_avail_gb'],
                 code), flush=True)
    else:
        print("K-LADDER %s: NO ROWS (exit=%d) -- the runner did not reach its "
              "metrics" % (TAG, code), flush=True)
    sys.exit(code)
