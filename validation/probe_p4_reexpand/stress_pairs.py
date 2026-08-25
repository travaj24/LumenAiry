"""S14.6 stressor -- diagnosed / undiagnosed pairs, byte-identity per iteration.

One arm = one process.  Each iteration runs EXACTLY what
``test_frame_completeness_metric_published`` runs (diagnosed call, then
undiagnosed call, same input array) and records:

* ``equal``   -- ``np.array_equal(E_a, E_b)``, the assertion under adjudication;
* ``h_a/h_b`` -- sha256 of each field, so a drift that is not a pair-mismatch
                 (iteration-to-iteration) is still caught;
* every bar the test asserts, as a NUMBER, so a failure arrives with its
  margin instead of a bare ``FAILED`` line (the S14.6 instrumentation lesson).

No retries, no supervision: it runs its iteration count and reports.

Usage:  python stress_pairs.py <n_iters> <arm_id> <out.jsonl>
"""
import json
import os
import sys
import time

import common as c
import numpy as np


def main():
    n = int(sys.argv[1])
    arm = sys.argv[2]
    out = sys.argv[3]
    E = c.conv_input()
    presc = c.m5_biconcave()
    h0 = None
    n_bad_pair = 0
    n_bad_drift = 0
    with open(out, 'w', encoding='ascii') as fh:
        for it in range(n):
            t0 = time.perf_counter()
            rec = {'arm': arm, 'it': it, 'pid': os.getpid(),
                   'omp': os.environ.get('OMP_NUM_THREADS', '(unset)')}
            try:
                diag = {}
                E_a = c.gbd(E, presc, reexpand='auto', diagnostics=diag)
                E_b = c.gbd(E, presc, reexpand='auto')
                eq = bool(np.array_equal(E_a, E_b))
                ha, hb = c.field_hash(E_a), c.field_hash(E_b)
                if h0 is None:
                    h0 = ha
                rec.update(
                    equal=eq, h_a=ha, h_b=hb, drift=bool(ha != h0),
                    maxabs=(0.0 if eq else float(np.max(np.abs(E_a - E_b)))),
                    keys_ok=all(k in diag for k in (
                        'frame_completeness', 'reexpanded', 'n_beamlets',
                        'frame_completeness_input',
                        'frame_completeness_reexpanded')),
                    comp=diag.get('frame_completeness'),
                    comp_in=diag.get('frame_completeness_input'),
                    comp_re=diag.get('frame_completeness_reexpanded'),
                    reexpanded=diag.get('reexpanded'),
                    nb=diag.get('n_beamlets'),
                    dt=round(time.perf_counter() - t0, 3))
                if not eq:
                    n_bad_pair += 1
                if rec['drift']:
                    n_bad_drift += 1
            except BaseException as exc:            # noqa: BLE001 -- record it
                rec.update(error=f"{type(exc).__name__}: {exc}")
            fh.write(json.dumps(rec) + '\n')
            fh.flush()
    print(f"arm {arm} pid {os.getpid()}: {n} iterations, "
          f"{n_bad_pair} pair mismatches, {n_bad_drift} cross-iteration drifts")


if __name__ == '__main__':
    main()
