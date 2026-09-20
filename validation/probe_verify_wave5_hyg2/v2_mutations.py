"""(9) Run the author's test file against each mutant tree and record.

Each mutant is a SELF-CONTAINED copy under the scratchpad (its own
``lumenairy/``, ``pyproject.toml`` and a copy of the author's test file with
``tests/__init__.py`` / ``tests/unit/__init__.py``).  The self-containment is
required: ``tests/`` is a PACKAGE here, so pytest's prepend import mode
inserts the tree that holds it at ``sys.path[0]`` -- running the worktree's
test file against a mutant on ``PYTHONPATH`` silently measures the WORKTREE
(all four mutants "passed" that way; the finding is recorded).

    python v2_mutations.py <mutant_root> <out.json>
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlib import build_tag, write_json                         # noqa: E402

ARMS = ('m0', 'm1', 'm2', 'm3', 'm4')
DESC = {
    'm0': 'control -- unmutated copy',
    'm1': '_collins_axis_chirp: bld hardcoded to np inside the body',
    'm2': 'traced refusal removed (falls into the measuring branch)',
    'm3': '_fft2_pair returns wrappers (breaks the `is` identity)',
    'm4': '_as_c_order always xp.asarray (drops NumPy contiguity)',
}


def main():
    root, out_path = sys.argv[1], sys.argv[2]
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', LUMENAIRY_MEM_BUDGET_MB='8192',
               PYTHONHASHSEED='0')
    res = {'build': build_tag(), 'mutant_root': root, 'arms': {}}
    for m in ARMS:
        cwd = os.path.join(root, m)
        p = subprocess.run(
            [sys.executable, '-m', 'pytest',
             'tests/unit/test_wave5_h2_collins_jax.py',
             '--capture=sys', '-q', '-p', 'no:cacheprovider'],
            cwd=cwd, env=env, capture_output=True, text=True)
        tail = (p.stdout or '') + (p.stderr or '')
        summary = [ln for ln in tail.splitlines()
                   if re.search(r'\d+ (passed|failed|error)|no tests ran',
                                ln)]
        failed = sorted(set(re.findall(r'^FAILED (\S+)', tail, re.M)))
        errored = sorted(set(re.findall(r'^ERROR (\S+)', tail, re.M)))
        res['arms'][m] = {
            'description': DESC[m],
            'returncode': p.returncode,
            'summary': summary[-1] if summary else '(none)',
            'n_failed': len(failed), 'failed_ids': failed,
            'n_errors': len(errored), 'error_ids': errored,
            'detects_mutation': bool(failed or errored),
        }
        print(m, res['arms'][m]['summary'], failed)
    res['undetected_arms'] = sorted(
        m for m in ARMS if m != 'm0' and not res['arms'][m]['detects_mutation'])
    res['control_green'] = (res['arms']['m0']['n_failed'] == 0
                            and res['arms']['m0']['returncode'] == 0)
    write_json(res, out_path)
    print(json.dumps({'control_green': res['control_green'],
                      'undetected': res['undetected_arms']}, indent=1))


if __name__ == '__main__':
    main()
