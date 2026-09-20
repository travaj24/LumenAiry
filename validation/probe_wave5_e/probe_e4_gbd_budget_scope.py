"""WAVE5-E item E4 (VERIFY-B14 D3) -- re-measure the two numbers the scope
sentence for ``DENSE_MEM_BUDGET_ACCOUNTING`` is about to state.

VERIFY-B14 D3 records that "``'measured'`` makes the budget a bound" is true
only ABOVE one beamlet column: at ``N = 256`` a 4 MB budget reads **2.39x** and
a 1 MB budget **9.58x**, because the chunk floors at 1 and the fixed ~48 B/cell
term is outside the chunk arithmetic.  Those two numbers go into ``gbd.py``'s
module note, so under the house rule they are re-measured here on the running
build before being written down.

The measurement kernel is NOT re-implemented: ``_run`` / ``_bundle`` /
``_effective_chunk`` are imported from
``validation/probe_verify_b14/probe_v4_gbd_budget.py``, which is where the
ladder those numbers came from lives.

Usage:  python probe_e4_gbd_budget_scope.py <out.json>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'probe_verify_b14'))

from probe_v4_gbd_budget import (  # noqa: E402
    _bundle,
    _effective_chunk,
    _run,
)

import lumenairy as la  # noqa: E402
from lumenairy.propagators import gbd as G  # noqa: E402


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'e4_gbd_scope.json'
    old = G.DENSE_MEM_BUDGET_ACCOUNTING
    N, nb = 256, 1024
    res = {'lumenairy_file': la.__file__, 'version': la.__version__,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'platform': sys.platform, 'shipped_default': old,
           'N': N, 'n_beamlets': nb,
           'legacy_const': G._DENSE_CELL_BYTES_LEGACY,
           'measured_const': G._DENSE_CELL_BYTES_MEASURED,
           'fixed_term_bytes_per_cell': 48.0,
           'rows': []}
    print('lumenairy.__file__ =', la.__file__, flush=True)
    #: The loop's own floor: one beamlet column plus the fixed arrays the chunk
    #: arithmetic does not model.  Derived from the constants, not read back.
    floor_mb = N * N * (48.0 + G._DENSE_CELL_BYTES_MEASURED) / 1e6
    res['one_column_floor_mb'] = floor_mb
    print('one-column floor at N=%d: %.2f MB' % (N, floor_mb), flush=True)
    try:
        bundle = _bundle(nb)
        for budget in (512.0, 16.0, 4.0, 1.0):
            _f, peak = _run('measured', bundle, N, budget)
            ch = _effective_chunk(N, budget, 'measured', n=nb)
            row = {'budget_mb': budget, 'chunk': ch,
                   'peak_bytes': peak, 'peak_mb': peak / 1e6,
                   'ratio': peak / (budget * 1e6),
                   'above_one_column': bool(budget > floor_mb)}
            res['rows'].append(row)
            print('budget %7.1f MB -> chunk %3d, peak %8.2f MB, %6.2fx  %s'
                  % (budget, ch, peak / 1e6, row['ratio'],
                     'bounded' if row['ratio'] < 1.0 else 'NOT bounded'),
                  flush=True)
    finally:
        G.DENSE_MEM_BUDGET_ACCOUNTING = old
    by_budget = {r['budget_mb']: r['ratio'] for r in res['rows']}
    res['ratio_4mb'] = by_budget.get(4.0)
    res['ratio_1mb'] = by_budget.get(1.0)
    res['bound_only_above_one_column'] = bool(
        by_budget.get(512.0, 9e9) < 1.0
        and by_budget.get(4.0, 0.0) > 1.0
        and by_budget.get(1.0, 0.0) > 1.0)
    print('SCOPE SENTENCE HOLDS ON THIS BUILD:',
          res['bound_only_above_one_column'],
          ' 4 MB -> %.2fx, 1 MB -> %.2fx'
          % (res['ratio_4mb'], res['ratio_1mb']), flush=True)
    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(res, fh, indent=1)
    print('wrote', out_path, flush=True)


if __name__ == '__main__':
    main()
