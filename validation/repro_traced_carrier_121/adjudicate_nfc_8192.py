# ADJUDICATION driver -- n_fine_cap 8192 (arm A) vs 16384 (arm B) on the
# design-121 32-order fan.  Runs ``fan_multi_121.py`` UNMODIFIED and answers
# ONE question: does n_fine_cap=8192 lower accuracy on ANY of the 32 orders?
#
# WHY A WRAPPER AND NOT AN EDIT.  fan_multi_121.py is the acceptance runner;
# it is not touched.  Everything here is applied from the outside:
#
#   1. ``output_grid['ram_budget'] = inf`` is injected into the chain-multi
#      call.  This is remedy #1 of FIX_PERF_CACHES_BLUESTEIN_2026_08_09.md
#      sec 2.4 and it is MANDATORY for arm B: with _FINE_GRID_WORK_ARRAYS=16
#      and frac=0.5, approving n_fine=16384 needs 137.4 GB of budget, and this
#      box reports get_ram_budget() = 102.9 GB -- so WITHOUT the override the
#      shipped NFC=16384 degrades to 8192 and arm B silently becomes arm A.
#      MEASURED on this box before any run:
#          _memory_bounded_n_fine(16384) -> 8192 (box budget), 16384 (inf).
#      It is injected into BOTH arms so the ONLY difference between them is
#      the COUNT cap n_fine_cap.  ``ram_budget`` is an accepted
#      ``_OUTPUT_GRID_PASSTHROUGH`` key (carrier.py), reaching both
#      ``_fine_trace_group_exit`` and ``carrier_referenced_exact_focus_readout``.
#      The un-overridden clamp value is still recorded per order as a SHADOW
#      column, so the production-default behaviour stays visible.
#
#   2. ``carrier._memory_bounded_n_fine`` is wrapped by a RECORDER (it changes
#      nothing; it forwards and logs).  That is how the actual per-order
#      n_fine, retrace window, dx_fine and exit-sphere Nyquist pitch are
#      asserted rather than assumed -- the audit's own honesty requirement.
#
#   3. The chain-multi ``progress`` hook is wrapped for per-order wall times.
#
#   4. The runner is exec'd with ``__name__ = '__main__'`` in a namespace THIS
#      script owns, so its own arrays (res, exact_c, cellP, fwhms, ee3s, ...)
#      survive its terminal ``raise SystemExit`` and the CSV is written from
#      the runner's OWN numbers -- not from a reimplementation of them.
#
# Env consumed HERE:
#   ADJ_ARM   'A' | 'B'        label only; NFC is what actually differs
#   ADJ_CSV   path             per-order rows are APPENDED as each shard ends
#   ADJ_JSON  path             per-shard full dump
#   ADJ_LOG   path             full stdout tee
#   ADJ_NW    int (default 1)  Newton pool size (capstone: 8 workers OOMs)
#   ADJ_RAMB  'inf' (default) | float GB | 'box' (no override)
# Env passed through to fan_multi_121.py: RN RS DXO NOUT TILE KEEP LEG NFC WF
#   DPX OTEG   (NW is set here from ADJ_NW).
import json
import os
import sys
import threading
import time
import traceback
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import numpy as np  # noqa: E402

import lumenairy as la  # noqa: E402
from lumenairy.memory import get_ram_budget  # noqa: E402
from lumenairy.propagators import carrier as _C  # noqa: E402

ARM = os.environ.get('ADJ_ARM', 'A')
NFC = int(os.environ.get('NFC', '8192'))
NW = int(os.environ.get('ADJ_NW', '1'))
os.environ['NW'] = str(NW)
CSV = os.environ.get('ADJ_CSV', os.path.join(_HERE, '_adj_nfc_rows.csv'))
JSONP = os.environ.get('ADJ_JSON', '')
RAMB = os.environ.get('ADJ_RAMB', 'inf')
KEEP = os.environ.get('KEEP', '<all>')

_T0 = time.time()

# --------------------------------------------------------------------------
# 1. recorder on the RAM clamp -- forwards, records, changes nothing
# --------------------------------------------------------------------------
_CUR = {'k': -1, 'name': '<setup>', 't': 0.0}
_CLAMP: list = []
_orig_mbnf = _C._memory_bounded_n_fine


def _mbnf(n_req, label, **kw):
    out = _orig_mbnf(n_req, label, **kw)
    win = kw.get('window')
    nyq = kw.get('nyquist_dx')
    rb = kw.get('ram_budget')
    # SHADOW: what the un-overridden production clamp would have returned at
    # this instant, on this box, for this request.  Recorded, never applied.
    try:
        shadow = _orig_mbnf(n_req, label, on_ram_cap='ignore')
    except Exception:
        shadow = None
    _CLAMP.append({
        'k': _CUR['k'], 'order': _CUR['name'], 'label': label,
        'n_req': int(n_req), 'n_out': int(out),
        'ram_budget': (None if rb is None else float(rb)),
        'ram_budget_box_gb': get_ram_budget() / 1e9,
        'shadow_n': (None if shadow is None else int(shadow)),
        'window_m': (None if win is None else float(win)),
        'nyquist_dx_m': (None if nyq is None else float(nyq)),
        'dx_fine_m': (None if win is None else float(win) / float(out)),
        't': time.time() - _T0})
    return out


_C._memory_bounded_n_fine = _mbnf

# --------------------------------------------------------------------------
# 2. warning capture -- every guard message, tagged with the order it fired on
# --------------------------------------------------------------------------
warnings.simplefilter('always')     # the runner prepends its 3 'ignore's after
_WARNS: list = []
_orig_showwarning = warnings.showwarning


def _showwarning(message, category, filename, lineno, file=None, line=None):
    _WARNS.append({'k': _CUR['k'], 'order': _CUR['name'],
                   'category': category.__name__,
                   'where': '%s:%d' % (os.path.basename(filename), lineno),
                   'msg': str(message)[:1200],
                   't': time.time() - _T0})
    return _orig_showwarning(message, category, filename, lineno, file, line)


warnings.showwarning = _showwarning

# --------------------------------------------------------------------------
# 3. chain wrappers -- Newton pool size, ram_budget injection, progress stamps
# --------------------------------------------------------------------------
_orig_chain = la.propagate_traced_carrier_chain
_orig_multi = la.propagate_traced_carrier_chain_multi
_STASH: dict = {}
_MARKS: list = []


def _ram_override():
    if RAMB == 'box':
        return None
    if RAMB == 'inf':
        return float('inf')
    return float(RAMB) * 1e9


def _chain(*a, **kw):
    kw['n_workers'] = NW
    return _orig_chain(*a, **kw)


def _multi(*a, **kw):
    kw['n_workers'] = NW
    rb = _ram_override()
    og = dict(kw.get('output_grid') or {})
    if rb is not None:
        og['ram_budget'] = rb
    kw['output_grid'] = og
    _STASH['output_grid'] = dict(og)
    user_prog = kw.get('progress')

    def _prog(k, K, nm):
        _CUR.update(k=int(k), name=str(nm), t=time.time() - _T0)
        _MARKS.append({'k': int(k), 'K': int(K), 'name': str(nm),
                       't': time.time() - _T0})
        if user_prog is not None:
            user_prog(k, K, nm)

    kw['progress'] = _prog
    t0 = time.time()
    out = _orig_multi(*a, **kw)
    _STASH['res'] = out
    _STASH['chainB_s'] = time.time() - t0
    _MARKS.append({'k': -1, 'K': -1, 'name': '<end>', 't': time.time() - _T0})
    _CUR.update(k=-1, name='<post>')
    return out


la.propagate_traced_carrier_chain = _chain
la.propagate_traced_carrier_chain_multi = _multi

# --------------------------------------------------------------------------
# 4. memory sampler
# --------------------------------------------------------------------------
_peak = {'rss': 0.0, 't': 0.0, 'min_avail': 1e9}
_stop = threading.Event()


def _sampler():
    try:
        import psutil
    except ImportError:
        return
    me = psutil.Process()
    while not _stop.is_set():
        try:
            rss = 0.0
            for p in [me] + me.children(recursive=True):
                try:
                    rss += p.memory_info().rss
                except psutil.Error:
                    pass
            rss /= 2 ** 30
            av = psutil.virtual_memory().available / 2 ** 30
            if rss > _peak['rss']:
                _peak.update(rss=rss, t=time.time() - _T0)
            _peak['min_avail'] = min(_peak['min_avail'], av)
        except Exception:
            pass
        _stop.wait(3.0)


threading.Thread(target=_sampler, daemon=True).start()


# --------------------------------------------------------------------------
# 5. stdout tee
# --------------------------------------------------------------------------
class _Tee:
    def __init__(self, raw, fh):
        self._raw, self._fh = raw, fh
        self._bol = True

    def write(self, s):
        for part in s.splitlines(keepends=True):
            if self._bol and part.strip():
                stamp = '[%9.2fs] ' % (time.time() - _T0)
                self._raw.write(stamp)
                self._fh.write(stamp)
            self._raw.write(part)
            self._fh.write(part)
            self._bol = part.endswith('\n')
        return len(s)

    def flush(self):
        self._raw.flush()
        self._fh.flush()

    def __getattr__(self, k):
        return getattr(self._raw, k)


LOGP = os.environ.get('ADJ_LOG',
                      os.path.join(_HERE, '_adj_%s_%s.log' % (ARM, os.getpid())))
_logfh = open(LOGP, 'w', encoding='cp1252', errors='replace')

_raw = sys.stdout
print('=' * 78, flush=True)
print("ADJUDICATION n_fine_cap -- ARM %s (NFC=%d)" % (ARM, NFC), flush=True)
print("  KEEP=%s  NOUT=%s  DXO=%s  TILE=%s  WF=%s  LEG=%s  OTEG=%s  NW=%d"
      % (KEEP, os.environ.get('NOUT', '<auto>'),
         os.environ.get('DXO', '0.2e-6'), os.environ.get('TILE', '1024'),
         os.environ.get('WF', '4.0'), os.environ.get('LEG', 'auto'),
         os.environ.get('OTEG', 'error'), NW), flush=True)
print("  ram_budget override = %s   (box get_ram_budget() = %.3f GB)"
      % (RAMB, get_ram_budget() / 1e9), flush=True)
print('=' * 78, flush=True)
sys.stdout = _Tee(_raw, _logfh)

# --------------------------------------------------------------------------
# 6. run the runner, keeping its namespace
# --------------------------------------------------------------------------
FAN = os.path.join(_HERE, 'fan_multi_121.py')
G = {'__name__': '__main__', '__file__': FAN, '__builtins__': __builtins__}
code = 0
err = ''
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
    sys.stdout = _raw
    _stop.set()

WALL = time.time() - _T0


# --------------------------------------------------------------------------
# 7. per-order rows, from the RUNNER'S OWN arrays
# --------------------------------------------------------------------------
def _clamp_for(k, label):
    hits = [c for c in _CLAMP if c['k'] == k and c['label'] == label]
    return hits[-1] if hits else None


def _f(v, d=None):
    try:
        return float(v)
    except Exception:
        return d


rows = []
res = _STASH.get('res')


def _build_rows():
    """Per-order rows from the runner's OWN arrays; never fatal."""
    F = np.asarray(G['res'].field)
    DXO_ = float(G['DXO'])
    exact_c = np.asarray(G['exact_c'])
    cellP = np.asarray(G['cellP'])
    share_des = np.asarray(G['share_des'])
    cell_share = np.asarray(G['cell_share'])
    cell_in = np.asarray(G['cell_in'])
    fwhms = np.asarray(G['fwhms'])
    ee3s = np.asarray(G['ee3s'])
    ee6s = np.asarray(G['ee6s'])
    ee12s = np.asarray(G['ee12s'])
    cerr = np.asarray(G['cerr'])
    P_in = float(G['P_in'])
    P_tot = float(G['P_tot'])
    mxs, mys = np.asarray(G['mx']), np.asarray(G['my'])
    # per-order wall times from the progress marks
    tmark = {m['k']: m['t'] for m in _MARKS if m['k'] >= 0}
    tend = max([m['t'] for m in _MARKS], default=WALL)
    for k, c in enumerate(G['res'].congruences):
        NT = int(c['tile'])
        r0, c0 = c['tile_origin']
        tile = F[max(r0, 0):r0 + NT, max(c0, 0):c0 + NT]
        I = np.abs(tile) ** 2
        tot = float(I.sum())
        iy, ix = np.unravel_index(np.argmax(I), I.shape)
        rr = np.hypot((np.arange(I.shape[1]) - ix)[None, :] * DXO_,
                      (np.arange(I.shape[0]) - iy)[:, None] * DXO_)
        pk = float(I[iy, ix])
        # halo amax: max AMPLITUDE beyond a radius, as a fraction of the peak
        # amplitude (the library's own amax_halo convention, applied to the
        # readout tile about its peak).
        _h = {}
        for _rad in (12e-6, 20e-6, 40e-6):
            _m = rr > _rad
            _h[_rad] = (float(np.sqrt(I[_m].max() / pk))
                        if (_m.any() and pk > 0) else 0.0)
        wall = (tend - tmark[k]) if (k + 1) not in tmark else \
            (tmark[k + 1] - tmark[k])
        cl = _clamp_for(k, '_fine_trace_group_exit')
        cr = _clamp_for(k, 'carrier_referenced_exact_focus_readout')
        wr = [w for w in _WARNS if w['k'] == k]
        # niche C1 item 4: the EXACT leg reports its own MEASURED exit NA next
        # to the paraxial na_exit it was SIZED from, and the |E|^2-weighted
        # fraction of exit power above the grid's Nyquist NA.  That fraction is
        # the direct answer to "what does the coarser grid throw away", so it
        # is scored rather than inferred from the leg's sizing NA.
        # NB there are TWO stages flagged ``exact_final``: the leg itself
        # (carrier.py:7783, dx = dx_fine, carries the NA diagnostics) and the
        # '<target>' bookkeeping stage appended after it (carrier.py:7871,
        # dx = dx_out, no diagnostics).  Select by the diagnostic, not by
        # position -- taking [-1] silently picks the target stage and returns
        # None for every NA column.
        _st = [s for s in (c.get('stages') or [])
               if s.get('exact_final') and not s.get('target')
               and s.get('na_exit_measured') is not None]
        if not _st:
            _st = [s for s in (c.get('stages') or [])
                   if s.get('exact_final') and not s.get('target')]
        _st = _st[-1] if _st else {}
        row = {
            'arm': ARM, 'nfc': NFC, 'keep': KEEP, 'k': k,
            'order': str(c['name']), 'mx': int(mxs[k]), 'my': int(mys[k]),
            'wall_s': round(float(wall), 2),
            'design_pct': float(share_des[k]) * 100.0,
            'field_pct': float(cell_share[k]) * 100.0,
            'ratio': float(cell_share[k] / share_des[k]),
            'cell_in_pct': float(cell_in[k]) * 100.0,
            'cellP': float(cellP[k]),
            'exact_x_um': float(exact_c[k, 0]) * 1e6,
            'exact_y_um': float(exact_c[k, 1]) * 1e6,
            'power_in': _f(c['power_in']), 'power_exit': _f(c['power_exit']),
            'power_out': _f(c['power_out']),
            'throughput': _f(c['throughput']), 'capture': _f(c['capture']),
            'cell_vs_powerout_m1': float(cellP[k] / float(c['power_out']) - 1.0),
            'fwhm_um': float(fwhms[k]) * 1e6,
            'ee3': float(ee3s[k]) * 100.0, 'ee6': float(ee6s[k]) * 100.0,
            'ee12': float(ee12s[k]) * 100.0,
            'peak_I': pk, 'tile_power': tot,
            'halo_amax_12um': _h[12e-6], 'halo_amax_20um': _h[20e-6],
            'halo_amax_40um': _h[40e-6],
            'centroid_err_um': float(cerr[k]) * 1e6,
            'chief_x_um': float(c['chief_ray'][0]) * 1e6,
            'chief_y_um': float(c['chief_ray'][1]) * 1e6,
            'tile_px': NT, 'readout_period_um': _f(c['readout_period']) * 1e6,
            'clipped': _f(c['clipped']),
            'leg_n_req': (cl or {}).get('n_req'),
            'leg_n_fine': (cl or {}).get('n_out'),
            'leg_win_mm': (None if cl is None else cl['window_m'] * 1e3),
            'leg_dx_fine_um': (None if cl is None else cl['dx_fine_m'] * 1e6),
            'leg_nyq_dx_um': (None if cl is None else cl['nyquist_dx_m'] * 1e6),
            'leg_nyq_margin': (None if cl is None else
                               cl['nyquist_dx_m'] / cl['dx_fine_m']),
            'leg_shadow_n': (cl or {}).get('shadow_n'),
            'ro_n_req': (cr or {}).get('n_req'),
            'ro_n_fine': (cr or {}).get('n_out'),
            'ro_win_mm': (None if cr is None else cr['window_m'] * 1e3),
            'ro_dx_fine_um': (None if cr is None else cr['dx_fine_m'] * 1e6),
            'ro_nyq_dx_um': (None if cr is None else cr['nyquist_dx_m'] * 1e6),
            'ro_nyq_margin': (None if cr is None else
                              cr['nyquist_dx_m'] / cr['dx_fine_m']),
            'ro_shadow_n': (cr or {}).get('shadow_n'),
            'na_exit_sized': _f(_st.get('na_exit')),
            'na_exit_measured': _f(_st.get('na_exit_measured')),
            'na_grid_nyquist': _f(_st.get('na_grid_nyquist')),
            'exit_power_above_nyquist': _f(
                _st.get('exit_power_above_nyquist')),
            'leg_stage_dx_um': (None if _st.get('dx') is None
                                else float(_st['dx']) * 1e6),
            'ram_budget_box_gb': round(get_ram_budget() / 1e9, 3),
            'n_warn': len(wr),
            'warn_tags': '|'.join(sorted({w['where'] for w in wr})),
            'P_tot_over_P_in': P_tot / P_in,
            'verdict_pass': int(code == 0),
            'peak_rss_gb': round(_peak['rss'], 3),
            'min_avail_gb': round(_peak['min_avail'], 3),
            'shard_wall_s': round(WALL, 1),
            'exit': code}
        rows.append(row)


if res is not None and 'fwhms' in G:
    try:
        _build_rows()
    except BaseException:
        err = (err + '\n--- row build ---\n' + traceback.format_exc())
        print(err, flush=True)
        if code == 0:
            code = 98

# --------------------------------------------------------------------------
# 8. append CSV (resumable: header written once, rows appended per shard)
# --------------------------------------------------------------------------
if rows:
    import csv as _csv
    cols = list(rows[0].keys())
    new = not os.path.exists(CSV)
    with open(CSV, 'a', encoding='cp1252', errors='replace', newline='') as fh:
        # csv.writer, NOT a manual join: 'order' is '(-4,-2)' and 'keep' is
        # '-4,-2;-3,-2', both of which carry commas.  A manual join silently
        # shifts every column right of them -- a wrong TABLE, not a crash.
        w = _csv.writer(fh, quoting=_csv.QUOTE_MINIMAL, lineterminator='\n')
        if new:
            w.writerow(cols)
        for r in rows:
            w.writerow(['' if r[c] is None else
                        ('%.12g' % r[c] if isinstance(r[c], float) else r[c])
                        for c in cols])
    print("ADJ: appended %d rows to %s" % (len(rows), CSV), flush=True)
else:
    print("ADJ: NO ROWS (exit=%d) -- runner did not reach its metrics" % code,
          flush=True)

if JSONP:
    with open(JSONP, 'w', encoding='cp1252', errors='replace') as fh:
        json.dump({'arm': ARM, 'nfc': NFC, 'keep': KEEP, 'exit': code,
                   'err': err, 'wall_s': WALL, 'rows': rows,
                   'clamp': _CLAMP, 'warns': _WARNS, 'marks': _MARKS,
                   'output_grid': {k: (None if v == float('inf') else v)
                                   for k, v in
                                   _STASH.get('output_grid', {}).items()},
                   'ram_budget_override': RAMB,
                   'peak_rss_gb': _peak['rss'],
                   'min_avail_gb': _peak['min_avail']},
                  fh, indent=1, default=str)

print("\nADJ ARM %s NFC=%d KEEP=%s  WALL %.1f s (%.2f min)  peak RSS %.2f GB "
      "at t=%.0f s  min avail %.2f GB  exit=%d"
      % (ARM, NFC, KEEP, WALL, WALL / 60.0, _peak['rss'], _peak['t'],
         _peak['min_avail'], code), flush=True)
# the leg n_fine actually used, asserted rather than assumed
_legs = sorted({c['n_out'] for c in _CLAMP
                if c['label'] == '_fine_trace_group_exit'})
_ros = sorted({c['n_out'] for c in _CLAMP
               if c['label'] == 'carrier_referenced_exact_focus_readout'})
print("ADJ leg n_fine used: %s   readout n_fine used: %s   "
      "(ram_budget override %s)" % (_legs, _ros, RAMB), flush=True)
_logfh.write("\nADJ WALL %.1f s peak RSS %.2f GB exit=%d leg n_fine %s "
             "readout n_fine %s\n" % (WALL, _peak['rss'], code, _legs, _ros))
_logfh.close()
sys.exit(code)
