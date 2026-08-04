# P2 DID-NOT-WARN diagnosis -- the PRODUCTION side of the dedup question.
#
# The brief asks whether warning dedup can silence a PRODUCTION warning, which
# would be a real library defect rather than a test artefact.  The TEST side is
# closed (p2diag_capture.py probes 1/2/5: ``pytest.warns`` resets the filters on
# entry, which bumps ``_filters_version`` and invalidates every module
# ``__warningregistry__``, so nothing a previous test emitted can suppress it).
#
# Production has no pytest.  Under CPython's stock filters an unmatched
# ``RuntimeWarning`` takes the ``"default"`` action = **once per
# (text, category, module, lineno)**.  The chain's guard warns at
# ``stacklevel=3``, so the location it dedups against is the CALLER's line --
# a batch loop that calls ``propagate_traced_carrier_chain`` from ONE line.
#
# This measures the real thing: the same chain call, twice, from one line, in a
# plain interpreter with untouched filters -- counting how many
# ``NOT dx-STABLE`` warnings actually reach a handler.
#
# usage:  python p2diag_prod_dedup.py
import os
import sys
import warnings

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lumenairy as la                                          # noqa: E402
from p2diag_ram_axis import _slow_singlet_chain, _chain_kw, _WL  # noqa: E402

_SEEN = []
_ORIG = warnings.showwarning


def _count(message, category, filename, lineno, file=None, line=None):
    """Count WITHOUT touching warnings.filters -- catch_warnings would mutate
    them and invalidate exactly the registry state under test."""
    _SEEN.append((category.__name__, str(message), filename, lineno))
    return _ORIG(message, category, filename, lineno, file, line)


def main():
    env0, groups, dx = _slow_singlet_chain(N=768, dx=4e-6)
    kw = dict(_chain_kw(r_in=3e-3), self_check='dx')
    warnings.showwarning = _count
    print('stock filters:', warnings.filters[:3], '...')
    try:
        for i in range(2):                       # <-- ONE call site, twice
            la.propagate_traced_carrier_chain(env0, groups, _WL, dx, **kw)
    finally:
        warnings.showwarning = _ORIG
    dx_hits = [s for s in _SEEN if 'NOT dx-STABLE' in s[1]]
    print()
    print('total warnings delivered : %d' % len(_SEEN))
    for s in _SEEN:
        print('   %-16s %s:%d  %s' % (s[0], os.path.basename(s[2]), s[3],
                                      ' '.join(s[1].split())[:64]))
    print('NOT dx-STABLE delivered  : %d  (2 chain calls from ONE line)'
          % len(dx_hits))
    for s in dx_hits:
        print('   %s %s:%d' % (s[0], os.path.basename(s[2]), s[3]))
    if len(dx_hits) < 2:
        print()
        print('CONFIRMED: the second call is SILENT.  Python\'s "default" action '
              'dedups the\nconvergence flag per (text, category, caller line) -- '
              'so a batch loop that\ncalls the chain from one line is told '
              '"NOT dx-STABLE" ONCE, and every later\nnon-converged result in '
              'that loop returns unflagged.')
    else:
        print('both calls warned -- no production dedup on this path')


if __name__ == '__main__':
    main()
