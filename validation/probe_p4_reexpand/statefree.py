"""Is the DIAGNOSED call state-free?

S14.6's suspect is ``np.array_equal(E_a, E_b)`` between a diagnosed and an
undiagnosed call -- "a returned field that depends on whether a diagnostics
dict was passed".  For that to be possible the diagnosed call has to leave
something behind that the undiagnosed one reads.  So enumerate what it could
leave behind and check: snapshot every module-level binding of every lumenairy
module already imported, plus ``os.environ``, plus the warning filters, plus
NumPy's error state, run the diagnosed call, and diff.

An empty diff is the by-construction half of the adjudication; the stressor is
the empirical half.
"""
import os
import sys
import warnings

import numpy as np

import common as c


def _snapshot():
    snap = {}
    for name, mod in list(sys.modules.items()):
        if not name.startswith('lumenairy') or mod is None:
            continue
        for k, v in list(vars(mod).items()):
            if k.startswith('__') or callable(v):
                continue
            if isinstance(v, type(sys)):          # module aliases
                continue
            try:
                if isinstance(v, dict):
                    snap[f'{name}.{k}'] = ('dict', len(v),
                                           tuple(sorted(map(str, v.keys()))))
                elif isinstance(v, (list, tuple, set)):
                    snap[f'{name}.{k}'] = (type(v).__name__, len(v),
                                           repr(v)[:400])
                elif isinstance(v, np.ndarray):
                    snap[f'{name}.{k}'] = ('ndarray', v.shape,
                                           float(np.sum(np.abs(v))))
                else:
                    snap[f'{name}.{k}'] = ('scalar', repr(v)[:400])
            except Exception as exc:              # noqa: BLE001
                snap[f'{name}.{k}'] = ('unreadable', type(exc).__name__)
    snap['#env'] = tuple(sorted(os.environ.items()))
    snap['#warnfilters'] = repr(warnings.filters)
    snap['#seterr'] = repr(np.geterr())
    return snap


def main():
    E = c.conv_input()
    presc = c.m5_biconcave()
    # one warm call first, so import-time and first-call lazy initialisation
    # (which IS a legitimate state change) is not what the diff reports
    c.gbd(E, presc, reexpand='auto')
    before = _snapshot()
    diag = {}
    E_a = c.gbd(E, presc, reexpand='auto', diagnostics=diag)
    after = _snapshot()
    keys = set(before) | set(after)
    diffs = [k for k in sorted(keys)
             if before.get(k, '<absent>') != after.get(k, '<absent>')]
    print(f"tracked bindings: {len(keys)} over "
          f"{len({k.rsplit('.', 1)[0] for k in keys if not k.startswith('#')})}"
          f" lumenairy modules")
    if not diffs:
        print("DIFF: none -- the diagnosed call left NO module-level state, "
              "no environment change, no warning-filter change and no NumPy "
              "error-state change behind.")
    for k in diffs:
        print(f"  CHANGED {k}: {before.get(k, '<absent>')!r} -> "
              f"{after.get(k, '<absent>')!r}")
    E_b = c.gbd(E, presc, reexpand='auto')
    print(f"pair equal={np.array_equal(E_a, E_b)} "
          f"h_a={c.field_hash(E_a)} h_b={c.field_hash(E_b)}")
    # and the reverse order: undiagnosed FIRST, diagnosed second
    E_c = c.gbd(E, presc, reexpand='auto')
    diag2 = {}
    E_d = c.gbd(E, presc, reexpand='auto', diagnostics=diag2)
    print(f"reverse-order equal={np.array_equal(E_c, E_d)} "
          f"h_c={c.field_hash(E_c)} h_d={c.field_hash(E_d)}")
    print(f"input array untouched by either call: "
          f"{c.field_hash(E) == c.field_hash(c.conv_input())}")


if __name__ == '__main__':
    main()
