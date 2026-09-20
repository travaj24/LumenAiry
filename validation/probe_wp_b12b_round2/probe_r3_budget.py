"""R3 -- VERIFY-WP-B12b D-6, re-taken: a GBD field's DIGEST depends on
``LUMENAIRY_MEM_BUDGET_MB`` through the PUBLIC entry point.

Why this matters for evidence rather than for physics: ``_reconstruct_windowed``
chunks each bucket of the coherent beamlet sum to stay under the memory
budget, and the chunk boundaries change the GROUPING of a scatter-add.  The
field agrees to ~1e-15, but the BYTES do not -- so every byte-identity
reading in this package (and in WP-B12b's own) is only reproducible with the
budget pinned on the command line, exactly as VERIFY-WP-B12 recorded for FGA
(its D-4).

Run as ONE parent process that spawns one CHILD PER BUDGET, so the variable
is read at the value the child was started with and nothing caches across
arms.  The field agreement between arms is measured too, so the finding is
"the bytes move and the physics does not" rather than only the first half.

Author: Andrew Traverso
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import r2_common as C

_BUDGETS = ('unset', '4096', '2048', '512', '64', '8')


def _child():
    """One arm: build the field at the budget this process was started with
    and print a one-line JSON record."""
    C.assert_tree()
    presc = C.r2_prescription('air')
    F = C.gbd_field(presc, C.r2_input_field(), C.R2_DX, C.R2_LAM, 2.0e-3)
    rec = dict(budget=os.environ.get('LUMENAIRY_MEM_BUDGET_MB', 'unset'),
               digest=C.sha(F),
               peak=repr(float(np.abs(F).max())),
               energy=repr(float((np.abs(F) ** 2).sum())),
               real_sum=repr(float(F.real.sum())))
    np.save(Path(os.environ['R3_OUT']), F)
    print('R3JSON ' + json.dumps(rec))


def main():
    if os.environ.get('R3_CHILD'):
        _child()
        return
    C.assert_tree()
    arm, tokens = C.detect_arm()
    out = dict(env=C.env_block(), arm=arm, arm_tokens=tokens, rows=[])
    tmp = Path(tempfile.gettempdir()) / 'r3_budget_arms'
    tmp.mkdir(parents=True, exist_ok=True)
    fields = {}
    for b in _BUDGETS:
        env = dict(os.environ)
        env['R3_CHILD'] = '1'
        env['R3_OUT'] = str(tmp / f'field_{b}.npy')
        env.pop('LUMENAIRY_MEM_BUDGET_MB', None)
        if b != 'unset':
            env['LUMENAIRY_MEM_BUDGET_MB'] = b
        r = subprocess.run([sys.executable, os.path.abspath(__file__)],
                           env=env, capture_output=True, text=True,
                           cwd=os.path.dirname(os.path.abspath(__file__)))
        line = [ln for ln in r.stdout.splitlines()
                if ln.startswith('R3JSON ')]
        if not line:
            raise RuntimeError(f'budget {b}: child produced no record\n'
                               f'{r.stdout[-2000:]}\n{r.stderr[-2000:]}')
        rec = json.loads(line[-1][len('R3JSON '):])
        fields[b] = np.load(tmp / f'field_{b}.npy')
        out['rows'].append(rec)
        print(f'  budget {b:>6s}  {rec["digest"]}  peak {rec["peak"]}')

    ref = fields['unset']
    scale = float(np.abs(ref).max())
    agree = {}
    for b, F in fields.items():
        agree[b] = dict(
            max_abs_diff=float(np.abs(F - ref).max()),
            rel=float(np.abs(F - ref).max() / scale) if scale else 0.0,
            byte_identical=bool(C.sha(F) == C.sha(ref)))
    out['agreement_vs_unset'] = agree
    out['distinct_digests'] = sorted({r['digest'] for r in out['rows']})
    for b, a in agree.items():
        print(f'  agree  {b:>6s}  rel {a["rel"]:.3e}  '
              f'byte_identical={a["byte_identical"]}')
    C.dump(out, 'probe_r3_budget_' + arm)


if __name__ == '__main__':
    main()
