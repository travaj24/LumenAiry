# CAPSTONE Stage C driver -- runs fan_multi_121.py UNMODIFIED (via runpy) with
# per-order wall-clock stamping, a peak-memory sampler, and the SAME loud
# Newton-pool override Stage B needed (see capstone_stageB.py for the measured
# justification: 22.1 GB committed per pool worker x 8 = 177 GB, box driven to
# 0 GB free).
#
# Everything else is fan_multi_121.py's own: its order table, its exact skew
# ray trace for the frame centres, its acceptance checks and its verdict.
#
# Env passed straight through to fan_multi_121.py: RN, RS, DXO, NOUT, TILE,
# KEEP, LEG, NFC, WF, DPX, OTEG.  ``NW`` is consumed HERE (it is fan_multi's
# own name for the Newton pool size) and re-exported so the runner still
# prints what it used.
#
# Env consumed here: CAPSTONE_NW (Newton pool size, default 1),
# CAPSTONE_MEMLOG.
import os
import runpy
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import lumenairy as la  # noqa: E402

NW = int(os.environ.get('CAPSTONE_NW', '1'))
os.environ['NW'] = str(NW)
MEMLOG = os.environ.get('CAPSTONE_MEMLOG',
                        os.path.join(_HERE, '_capstone_mem_C.txt'))

_orig_chain = la.propagate_traced_carrier_chain
_orig_multi = la.propagate_traced_carrier_chain_multi


def _chain(*a, **kw):
    kw['n_workers'] = NW
    return _orig_chain(*a, **kw)


def _multi(*a, **kw):
    kw['n_workers'] = NW
    return _orig_multi(*a, **kw)


la.propagate_traced_carrier_chain = _chain
la.propagate_traced_carrier_chain_multi = _multi

# ---- per-order wall stamping ------------------------------------------------
# fan_multi passes ``progress=lambda k, K, nm: print(...)``.  Wrap the
# orchestrator's progress hook by stamping every line the script prints, which
# is simpler and cannot change what the script does: replace sys.stdout with a
# line-stamping writer.
_T0 = time.time()


class _Stamped:
    def __init__(self, raw):
        self._raw = raw
        self._at_line_start = True

    def write(self, s):
        for part in s.splitlines(keepends=True):
            if self._at_line_start and part.strip():
                self._raw.write(f"[{time.time() - _T0:9.2f}s] ")
            self._raw.write(part)
            self._at_line_start = part.endswith('\n')
        return len(s)

    def flush(self):
        self._raw.flush()

    def __getattr__(self, k):
        return getattr(self._raw, k)


_peak = {'rss': 0.0, 't': 0.0}
_stop = threading.Event()


def _sampler():
    import psutil
    me = psutil.Process()
    with open(MEMLOG, 'w') as fh:
        fh.write('t_s\trss_gb\tsys_avail_gb\n')
        while not _stop.is_set():
            try:
                procs = [me] + me.children(recursive=True)
                rss = 0.0
                for p in procs:
                    try:
                        rss += p.memory_info().rss
                    except psutil.Error:
                        pass
                rss /= 2 ** 30
                av = psutil.virtual_memory().available / 2 ** 30
                t = time.time() - _T0
                if rss > _peak['rss']:
                    _peak.update(rss=rss, t=t)
                fh.write(f'{t:.1f}\t{rss:.3f}\t{av:.3f}\n')
                fh.flush()
            except Exception:
                pass
            _stop.wait(2.0)


threading.Thread(target=_sampler, daemon=True).start()

_raw = sys.stdout
print('=' * 74, flush=True)
print("CAPSTONE STAGE C -- fan_multi_121.py, unmodified, via runpy", flush=True)
print(f"  OVERRIDE (loud, deliberate): Newton pool n_workers -> {NW}",
      flush=True)
print(f"  env: RN={os.environ.get('RN', '1024')} "
      f"RS={os.environ.get('RS', '4')} LEG={os.environ.get('LEG', 'paraxial')} "
      f"NFC={os.environ.get('NFC', '16384')} WF={os.environ.get('WF', '4.0')} "
      f"DXO={os.environ.get('DXO', '0.2e-6')} "
      f"TILE={os.environ.get('TILE', '1024')} "
      f"KEEP={os.environ.get('KEEP', '<all 32>')}", flush=True)
print(f"  every line below is stamped with seconds since Stage C start",
      flush=True)
print('=' * 74, flush=True)

sys.stdout = _Stamped(_raw)
code = 0
try:
    runpy.run_path(os.path.join(_HERE, 'fan_multi_121.py'),
                   run_name='__main__')
except SystemExit as exc:
    code = int(exc.code or 0)
finally:
    sys.stdout = _raw
    wall = time.time() - _T0
    _stop.set()
    print(f"\nCAPSTONE STAGE C WALL {wall:.1f} s ({wall / 60.0:.2f} min)  "
          f"peak RSS {_peak['rss']:.2f} GB at t={_peak['t']:.0f} s  "
          f"exit={code}", flush=True)
sys.exit(code)
