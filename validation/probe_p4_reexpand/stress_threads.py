"""S14.6 stressor -- IN-PROCESS THREADS.

The GBD path has no threading of its own, but its module-level state is shared:
the glass registry, the traced-kwarg default cache and its lock, the warning
registry, and every module-level constant the solver reads at CALL time
(``DETERMINISTIC_NORMAL_EQUATIONS``, ``LSTSQ_CONDITIONING_STEPDOWN``).  If any
of that were mutated per call, N threads running diagnosed/undiagnosed pairs
concurrently is the arm that finds it -- NumPy releases the GIL, so these run
genuinely concurrently.

``warnings.catch_warnings`` is process-global and NOT thread-safe, so this arm
sets the filter ONCE at module level and never enters that context manager
(otherwise the arm would be measuring CPython's warning-filter race, not the
library).

Usage:  python stress_threads.py <n_threads> <iters_per_thread> <out.jsonl>
"""
import json
import sys
import threading
import warnings

import numpy as np

import common as c
from lumenairy.elements.lenses_gbd import apply_real_lens_gbd

warnings.simplefilter('ignore')

_WL = c._WL
_DX = c._DX
_SS = c._SS


def _gbd_nolock(E, presc, **kw):
    return np.asarray(apply_real_lens_gbd(
        E, prescription=presc, wavelength=_WL, dx=_DX, sample_step=_SS, **kw))


def main():
    nthreads = int(sys.argv[1])
    iters = int(sys.argv[2])
    out = sys.argv[3]
    E = c.conv_input()
    presc = c.m5_biconcave()
    lock = threading.Lock()
    recs = []
    barrier = threading.Barrier(nthreads)

    def worker(tid):
        barrier.wait()                     # maximise overlap
        for it in range(iters):
            rec = {'tid': tid, 'it': it}
            try:
                diag = {}
                E_a = _gbd_nolock(E, presc, reexpand='auto', diagnostics=diag)
                E_b = _gbd_nolock(E, presc, reexpand='auto')
                eq = bool(np.array_equal(E_a, E_b))
                rec.update(equal=eq, h_a=c.field_hash(E_a),
                           h_b=c.field_hash(E_b),
                           comp=diag.get('frame_completeness'),
                           keys_ok=all(k in diag for k in (
                               'frame_completeness', 'reexpanded',
                               'n_beamlets', 'frame_completeness_input',
                               'frame_completeness_reexpanded')),
                           maxabs=(0.0 if eq
                                   else float(np.max(np.abs(E_a - E_b)))))
            except BaseException as exc:          # noqa: BLE001
                rec.update(error=f"{type(exc).__name__}: {exc}")
            with lock:
                recs.append(rec)

    ths = [threading.Thread(target=worker, args=(t,)) for t in range(nthreads)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    with open(out, 'w', encoding='ascii') as fh:
        for r in recs:
            fh.write(json.dumps(r) + '\n')
    bad = [r for r in recs if not r.get('equal', False)]
    hs = {r.get('h_a') for r in recs if 'h_a' in r}
    print(f"threads={nthreads} iters={iters} total={len(recs)} "
          f"mismatches={len(bad)} distinct_field_hashes={len(hs)}")
    for r in bad[:5]:
        print('  ', r)


if __name__ == '__main__':
    main()
