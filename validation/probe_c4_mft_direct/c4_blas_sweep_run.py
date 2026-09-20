"""WP-C4 -- drive ``c4_blas_sweep.py`` once per (kernel, threads) cell.

``OPENBLAS_CORETYPE`` is read when OpenBLAS is first loaded, so the kernel
cannot be changed inside a live interpreter; each cell is a fresh child with the
variable set on ITS environment.  The child's own reading of the variable is
recorded in its JSON, so a cell that silently fell back to the default kernel is
visible rather than counted.

    python c4_blas_sweep_run.py <root> <python-exe> OUT.json

``<root>`` is the tree ROOT -- the directory that CONTAINS ``lumenairy/``
-- and it is what each child gets as its ``PYTHONPATH``.  MEASURED
2026-09-20: passing the PACKAGE directory instead gives every child a
``PYTHONPATH`` naming ``.../lum_c4/lumenairy``, which contains no
importable ``lumenairy``, and all eight fall through to the dev install on
``D:``.  Eight cells reported ``WRONG TREE`` and none ran: the anchor
turned a silently-wrong measurement into a refusal, which is what it is
for.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c4lib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CORETYPES = ('HASWELL', 'NEHALEM', 'KATMAI', 'SANDYBRIDGE')
THREADS = ('1', '4')


def main(root, exe, out_path):
    root = os.path.abspath(root)
    tree = os.path.join(root, 'lumenairy')
    cells = {}
    for ct in CORETYPES:
        for th in THREADS:
            label = f"{ct}x{th}"
            cell_json = os.path.join(HERE, f"_blas_cell_{label}.json")
            env = dict(os.environ)
            env['OPENBLAS_CORETYPE'] = ct
            env['OPENBLAS_NUM_THREADS'] = th
            env['OMP_NUM_THREADS'] = th
            env['MKL_NUM_THREADS'] = th
            env['PYTHONPATH'] = root
            proc = subprocess.run(
                [exe, os.path.join(HERE, 'c4_blas_sweep.py'), tree, label,
                 cell_json],
                capture_output=True, text=True, env=env, timeout=1800)
            if proc.returncode != 0 or not os.path.exists(cell_json):
                cells[label] = {'failed': True,
                                'returncode': proc.returncode,
                                'stderr': proc.stderr[-800:]}
                print(f"[{label}] FAILED rc={proc.returncode}: "
                      f"{proc.stderr.strip().splitlines()[-1:]}")
                continue
            with open(cell_json, encoding='cp1252') as fh:
                cells[label] = json.load(fh)
            os.remove(cell_json)
            print(proc.stdout.rstrip())

    ok = [c for c in cells.values() if not c.get('failed')]
    summary = {'n_cells': len(cells), 'n_ran': len(ok)}
    if ok:
        by_shape = {}
        for c in ok:
            for r in c['rows']:
                by_shape.setdefault((r['N'], r['M']), []).append((c, r))
        rows = []
        for (N, M), pairs in by_shape.items():
            digests = {p[1]['dense_digest'] for p in pairs}
            routes = {p[1]['route'] for p in pairs}
            agrees = all(p[1]['route_agrees_with_rule'] for p in pairs)
            inside = all(p[1]['dense_inside_bar'] for p in pairs)
            rows.append({
                'N': N, 'M': M,
                'rule_says_direct': pairs[0][1]['rule_says_direct'],
                'distinct_dense_digests': len(digests),
                'distinct_routes': sorted(routes),
                'route_agrees_with_rule_in_every_cell': agrees,
                'dense_inside_derived_bar_in_every_cell': inside,
                'dense_vs_exact_min': min(p[1]['dense_vs_exact_max_abs']
                                          for p in pairs),
                'dense_vs_exact_max': max(p[1]['dense_vs_exact_max_abs']
                                          for p in pairs)})
        summary['per_shape'] = rows
        summary['kernels_seen'] = sorted(
            {str(c['env'].get('OPENBLAS_CORETYPE')) for c in ok})
    out = {'build': c4lib.build_tag(), 'cells': cells, 'summary': summary}
    c4lib.write_json(out, out_path)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
