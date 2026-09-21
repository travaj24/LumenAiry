"""VERIFY-WP-C3 CLAIM 8e part 2 -- is a MISSED mutation a REAL demotion?

A census miss only matters if the mutation actually breaks a device run.  For
each mutation this injects it into ``_collins_transport`` in the mutation tree
and drives ``_collins_transport`` with a CuPy array in a CHILD process.  On
this box the pristine call dies with ``ImportError ... cufft`` AT the device
transform, which is downstream of the injection point -- so any mutation that
demotes shows up as a DIFFERENT failure (the implicit-conversion TypeError, or
another TypeError) raised EARLIER.

    python v_mut_device.py <mutation-tree> <out.json>
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v_mutate as MUT  # noqa: E402  -- reuse the mutation table only

TREE = os.path.abspath(sys.argv[1])
OUT = sys.argv[2]
CARRIER = os.path.join(TREE, 'lumenairy', 'propagators', 'carrier.py')
PRISTINE = CARRIER + '.pristine'

CHILD = r'''
import sys, os, traceback
sys.path.insert(0, sys.argv[1])
import numpy as np, cupy as cp
import lumenairy.propagators.carrier as CA
assert os.path.abspath(CA.__file__).lower().startswith(
    os.path.abspath(sys.argv[1]).lower()), CA.__file__
n, dx = 32, 8e-6
ax = (np.arange(n) - n // 2) * dx
X, Y = np.meshgrid(ax, ax)
E = np.exp(-(X**2 + Y**2) / (60e-6**2)).astype(np.complex128)
Ed = cp.asarray(E)
try:
    CA._collins_transport(Ed, -0.05, 5e-3, 1.064e-6, dx, dx, dx_out=dx,
                          dy_out=dx, N_out_x=n, N_out_y=n, R_ref=np.inf,
                          on_collins_sampling='ignore')
    print('OUTCOME|ran|')
except Exception as exc:
    tb = traceback.extract_tb(exc.__traceback__)
    where = '%s:%d %s' % (os.path.basename(tb[-1].filename), tb[-1].lineno,
                          tb[-1].name)
    print('OUTCOME|%s|%s|%s' % (type(exc).__name__, str(exc)[:160], where))
'''


def run_child():
    env = dict(os.environ)
    env.update(PYTHONPATH=TREE, OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    p = subprocess.run([sys.executable, '-c', CHILD, TREE], cwd=TREE,
                       env=env, capture_output=True, text=True, timeout=600)
    for ln in (p.stdout + p.stderr).splitlines():
        if ln.startswith('OUTCOME|'):
            return ln
    return 'OUTCOME|NO-OUTPUT|' + (p.stdout + p.stderr)[-300:]


def main():
    if not os.path.exists(PRISTINE):
        shutil.copy2(CARRIER, PRISTINE)
    shutil.copy2(PRISTINE, CARRIER)
    rows = [{'mutation': 'BASELINE (pristine)', 'outcome': run_child()}]
    print('[dev] BASELINE  %s' % rows[0]['outcome'])
    for name, line, why in MUT.MUTATIONS:
        MUT.inject(line)
        o = run_child()
        rows.append({'mutation': name, 'why': why, 'injected': line,
                     'outcome': o})
        print('[dev] %-32s %s' % (name, o))
    shutil.copy2(PRISTINE, CARRIER)
    with open(OUT, 'w', encoding='cp1252', errors='replace') as fh:
        json.dump({'tree': TREE, 'rows': rows}, fh, indent=1, default=str)
    print('[wrote] %s' % OUT)


if __name__ == '__main__':
    main()
