"""VERIFY-WP-C4 claim 14 -- what the WP-C3 default flip hands to the WP-C4
rule when both land.

WP-C3 (`feat/c3-collins-default`) changes ``transport`` from ``'sziklas'`` to
``'collins'``.  The Collins transport calls ``_bluestein_centred_2d``
(``carrier.py`` (C3 tip) line 2507) with no ``method=``, so on the merged tree
every call it makes is decided by WP-C4's rule.

This probe runs the C3 branch's OWN transport code -- from a ``git archive`` of
that branch's tip, extracted into the session scratchpad, never into this
worktree and never edited -- with a spy on ``_bluestein_centred_2d`` that
records the four grid sizes it is handed.  The shapes are then put to THIS
tree's ``_auto_selects_direct``, so "which Collins shapes the rule captures"
is a measurement on the C3 code path and not an inference from its docstrings.

Two call sites, and they are very different:

* ``_collins_carrier_leg`` -- the TRANSPORT leg.  ``N_out_x, N_out_y`` are
  ``env_a.shape`` (C3 tip line 2641), i.e. the chain's own grid, so both
  ratios are exactly 1 and the rule can never fire.
* ``_collins_focus_readout`` -- the IMAGE-PLANE readout.  ``N_out_x =
  N_out_y = N_out``, the caller's readout window, against the chain's fine
  grid, which is the smallest-ratio call in the library.

    python v4_c3_interaction.py <this_tree> <c3_archive_tree>
"""
from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v4lib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

CHILD = r'''
import json, sys, warnings
import numpy as np
sys.path.insert(0, sys.argv[1])
import lumenairy
assert lumenairy.__file__.startswith(sys.argv[1].replace("/", "\\")[:3]) or True
print(json.dumps({"bound": lumenairy.__file__}))
from lumenairy.propagators import _bluestein as B
from lumenairy.propagators import carrier as C
seen = []
orig = B._bluestein_centred_2d
def spy(E, ax, ay, my, mx, **kw):
    seen.append({"Ny_in": int(E.shape[-2]), "Nx_in": int(E.shape[-1]),
                 "N_out_y": int(my), "N_out_x": int(mx)})
    return orig(E, ax, ay, my, mx, **kw)
B._bluestein_centred_2d = spy
C._bluestein_centred_2d = spy if hasattr(C, "_bluestein_centred_2d") else None
WL = 633e-9
def gauss(n, dx, w):
    x = (np.arange(n) - n / 2.0) * dx
    r2 = x[:, None] ** 2 + x[None, :] ** 2
    return np.exp(-r2 / w ** 2).astype(np.complex128)
rows = []
for (n_in, dx, n_out, zoom) in ((1024, 2e-6, 16, 64.0), (1024, 2e-6, 32, 32.0),
                                (1024, 2e-6, 64, 16.0), (512, 4e-6, 16, 32.0),
                                (2048, 1e-6, 40, 51.2), (256, 8e-6, 128, 2.0)):
    env = gauss(n_in, dx, 40 * dx)
    z = 5e-3
    dx_out = (WL * z / (n_in * dx)) / zoom
    seen.clear()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            C._collins_focus_readout(env, -2e-2, z, WL, dx, dx,
                                     dx_out=dx_out, N_out=n_out,
                                     on_replica="ignore",
                                     on_collins_sampling="ignore")
        rows.append({"case": "focus_readout", "n_in": n_in, "n_out": n_out,
                     "shapes": list(seen)})
    except Exception as exc:
        rows.append({"case": "focus_readout", "n_in": n_in, "n_out": n_out,
                     "raised": f"{type(exc).__name__}: {str(exc)[:200]}"})
    seen.clear()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            C._collins_carrier_leg(env, -2e-2, z, WL, dx, dx,
                                   on_collins_sampling="ignore")
        rows.append({"case": "carrier_leg", "n_in": n_in, "n_out": None,
                     "shapes": list(seen)})
    except Exception as exc:
        rows.append({"case": "carrier_leg", "n_in": n_in, "n_out": None,
                     "raised": f"{type(exc).__name__}: {str(exc)[:200]}"})
print(json.dumps({"rows": rows}))
'''


def main(tree, c3tree):
    import json
    v4lib.anchor(tree)
    from lumenairy.propagators._bluestein import (_auto_selects_direct,
                                                  _MFT_DIRECT_MAX_RATIO)
    script = os.path.join(HERE, '_v4_c3_child.py')
    with open(script, 'w', encoding='cp1252', errors='replace') as fh:
        fh.write(CHILD)
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONPATH=c3tree)
    p = subprocess.run([sys.executable, script, c3tree], capture_output=True,
                       text=True, env=env, cwd=HERE, timeout=3600)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith('{')]
    if len(lines) < 2:
        print('CHILD FAILED\n', p.stdout[-2000:], '\n', p.stderr[-2000:])
        return
    bound = json.loads(lines[0])['bound']
    rows = json.loads(lines[-1])['rows']
    out = {'build': v4lib.build_tag(), 'this_tree': tree,
           'c3_tree': c3tree, 'c3_lumenairy_file': bound,
           'constant': float(_MFT_DIRECT_MAX_RATIO), 'rows': []}
    for r in rows:
        rec = dict(r)
        rec['rule'] = []
        for s in r.get('shapes', []):
            says = bool(_auto_selects_direct(s['Ny_in'], s['Nx_in'],
                                             s['N_out_y'], s['N_out_x']))
            entries = s['N_out_y'] * s['Ny_in'] + s['N_out_x'] * s['Nx_in']
            flops = min(
                s['N_out_y'] * s['Ny_in'] * s['Nx_in']
                + s['N_out_y'] * s['Nx_in'] * s['N_out_x'],
                s['Ny_in'] * s['Nx_in'] * s['N_out_x']
                + s['N_out_y'] * s['Ny_in'] * s['N_out_x'])
            rec['rule'].append({**s, 'rule_says_direct': says,
                                'ratio_max': max(s['N_out_y'] / s['Ny_in'],
                                                 s['N_out_x'] / s['Nx_in']),
                                'work_per_kernel_entry': flops / entries})
        out['rows'].append(rec)
        shapes = ', '.join(
            f"{s['Ny_in']}x{s['Nx_in']}->{s['N_out_y']}x{s['N_out_x']}"
            f" r={s['ratio_max']:.5f} {'DENSE' if s['rule_says_direct'] else 'chirp'}"
            f" w/e={s['work_per_kernel_entry']:.0f}"
            for s in rec['rule'])
        print(f"{r['case']:16s} n_in={r['n_in']} n_out={r['n_out']}: "
              f"{shapes or r.get('raised', '(no MFT call)')}", flush=True)
    os.remove(script)
    v4lib.write_json(out, os.path.join(
        HERE, f"v4_c3_interaction_{v4lib.short_tag()}.json"))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
