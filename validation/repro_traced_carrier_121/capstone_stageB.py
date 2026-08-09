# CAPSTONE Stage B driver -- runs focus_scan_121.py UNMODIFIED (via runpy, so
# the construction, the metrics and the acceptance banner are byte-for-byte the
# shipped acceptance runner's) with two things added and NOTHING removed:
#
#   1. a peak-memory sampler over the whole process tree (RSS + commit), and
#   2. a LOUD, printed override of the chain's ``n_workers`` only.
#
# WHY THE n_workers OVERRIDE EXISTS.  focus_scan_121.py hard-codes
# ``n_workers=8``.  MEASURED on this box (2026-08-06, 128 GB): the shipped
# 8-worker Newton pool committed 22.1 GB PER WORKER during the exact final
# leg's fine retrace (n_fine_cap 8192 -> a 8192^2 = 6.7e7-point Newton grid,
# chunked 1/8 per worker), i.e. ~177 GB across the pool, taking system commit
# to 205.7 / 227.5 GB and free physical RAM to 0.0 GB at 9.7 min elapsed.  The
# run was stopped there.  That is ~20x more per worker than the chunk itself
# accounts for, so it is per-PROCESS overhead, not the physics.
#
# n_workers is a SPEED knob: ``_newton_invert_chunk`` rebuilds the same fit and
# the pool path is documented (and tested) to preserve the serial path's
# numerical behaviour exactly -- same iteration cap, same tolerance, same
# out-of-domain policy.  The chain-A cache keys it anyway, out of caution, but
# nothing here depends on that cache.  So this override changes RUNTIME, not
# the field.  It is printed, and the capstone report states it.
#
# Env: CAPSTONE_NW (default 1 = serial Newton), CAPSTONE_MEMLOG.
import os
import runpy
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))

import lumenairy as la  # noqa: E402

NW = int(os.environ.get('CAPSTONE_NW', '1'))
MEMLOG = os.environ.get('CAPSTONE_MEMLOG', os.path.join(_HERE, '_capstone_mem.txt'))

_orig_chain = la.propagate_traced_carrier_chain


def _chain(*a, **kw):
    kw['n_workers'] = NW
    return _orig_chain(*a, **kw)


la.propagate_traced_carrier_chain = _chain

# ---- CAPSTONE_WARN=1: run the SAME acceptance with warnings VISIBLE ---------
# focus_scan_121.py line 34 is a blanket ``warnings.filterwarnings('ignore')``.
# Every 'warn'-disposed guard on the chain -- on_ram_cap (a fine grid degraded
# by the RAM ceiling), on_rs_fine_clamp, on_na_proximity, on_decentred_fit,
# on_gap_paraxial, on_gap_frame -- is therefore INVISIBLE in a default run, and
# so is apply_real_lens_traced's under-sampling / Newton-non-convergence
# warning.  "No warnings" in that log is not evidence; it is a filter.  With
# this flag the blanket call is neutralised (only that ONE call: the module's
# own three targeted ignores in _d121_common are untouched) so the acceptance
# can be re-run and the guard surface actually read.
if os.environ.get('CAPSTONE_WARN') == '1':
    import warnings as _w

    _real_fw = _w.filterwarnings

    def _fw(action, message='', *a, **kw):
        if action == 'ignore' and message == '' and not a and not kw:
            print("  [capstone] neutralised a BLANKET "
                  "warnings.filterwarnings('ignore')", flush=True)
            return
        return _real_fw(action, message, *a, **kw)

    _w.filterwarnings = _fw
    _w.simplefilter('always')

_peak = {'rss': 0.0, 'commit': 0.0, 't': 0.0}
_stop = threading.Event()


def _sampler():
    import psutil
    me = psutil.Process()
    t0 = time.time()
    with open(MEMLOG, 'w') as fh:
        fh.write('t_s\trss_gb\tcommit_gb\tsys_avail_gb\n')
        while not _stop.is_set():
            try:
                procs = [me] + me.children(recursive=True)
                rss = 0.0
                com = 0.0
                for p in procs:
                    try:
                        mi = p.memory_info()
                        rss += mi.rss
                        com += getattr(mi, 'vms', 0) or 0
                    except psutil.Error:
                        pass
                rss /= 2 ** 30
                com /= 2 ** 30
                av = psutil.virtual_memory().available / 2 ** 30
                t = time.time() - t0
                if rss > _peak['rss']:
                    _peak.update(rss=rss, t=t)
                _peak['commit'] = max(_peak['commit'], com)
                fh.write(f'{t:.1f}\t{rss:.3f}\t{com:.3f}\t{av:.3f}\n')
                fh.flush()
            except Exception:
                pass
            _stop.wait(2.0)


th = threading.Thread(target=_sampler, daemon=True)
th.start()

print('=' * 74, flush=True)
print(f"CAPSTONE STAGE B -- focus_scan_121.py, unmodified, via runpy", flush=True)
print(f"  OVERRIDE (loud, deliberate): chain n_workers 8 -> {NW}", flush=True)
print(f"  reason: the 8-worker Newton pool committed 22.1 GB/worker (177 GB)",
      flush=True)
print(f"          in the exact final leg and drove this box to 0 GB free.",
      flush=True)
print(f"  n_workers is a speed knob; the pool path is documented + tested to",
      flush=True)
print(f"          preserve the serial numerical behaviour exactly.", flush=True)
print(f"  memory log: {MEMLOG}", flush=True)
print('=' * 74, flush=True)

t0 = time.time()
code = 0
try:
    runpy.run_path(os.path.join(_HERE, 'focus_scan_121.py'),
                   run_name='__main__')
except SystemExit as exc:
    code = int(exc.code or 0)
finally:
    wall = time.time() - t0
    _stop.set()
    th.join(timeout=5)
    print(f"\nCAPSTONE STAGE B WALL {wall:.1f} s "
          f"({wall / 60.0:.2f} min)  peak RSS {_peak['rss']:.2f} GB "
          f"at t={_peak['t']:.0f} s", flush=True)
sys.exit(code)
