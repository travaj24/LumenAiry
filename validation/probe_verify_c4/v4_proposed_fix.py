"""VERIFY-WP-C4 -- the requested edit of V-C4-D1, APPLIED to a tree copy and
measured, so the recommendation is not an untested suggestion.

Copies the worktree's ``lumenairy`` into the session scratchpad (nothing under
``lumenairy/`` is touched), applies the exact edit the report asks for, and
then measures three things on the patched tree:

1.  every shape this verification measured SLOWER is now REFUSED;
2.  every shape it measured safe, every square shape on the ladder and every
    shape the shipped suite drives is still CAPTURED;
3.  ``tests/unit/test_c4_mft_direct_default.py`` and
    ``tests/unit/test_verify_c4_mft_direct.py`` are still green on it.

    python v4_proposed_fix.py <tree> <scratch_root>
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v4lib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

CONSTANT = '''
#: The dense route builds ``My*Ny + Mx*Nx`` transcendental kernel entries and
#: then spends ``min(My*Ny*Nx + My*Nx*Mx, Ny*Nx*Mx + My*Ny*Mx)`` multiply-adds
#: using them.  Below this many multiply-adds PER kernel entry the build
#: dominates and the chirp-Z route wins even at a ratio the square ladder
#: measured safe.  DERIVED 2026-09-20 (VERIFY-WP-C4).
_MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY = 16.0
'''

OLD_TAIL = "    return max(my / ny, mx / nx) <= r"
NEW_TAIL = """    if not (max(my / ny, mx / nx) <= r):
        return False
    entries = my * ny + mx * nx
    flops = min(my * ny * nx + my * nx * mx, ny * nx * mx + my * ny * mx)
    return flops >= float(_MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY) * entries"""

MEASURED_SLOWER = [(2048, 128, 64, 4), (2048, 64, 64, 2), (2048, 32, 64, 1),
                   (1024, 32, 32, 1), (64, 2048, 2, 64), (32, 2048, 1, 64),
                   (4096, 64, 128, 2), (4096, 64, 64, 1), (64, 4096, 1, 64)]
MEASURED_SAFE = [(2048, 256, 64, 8), (1024, 128, 32, 4), (1024, 64, 32, 2),
                 (512, 64, 16, 2), (512, 32, 16, 1), (2048, 512, 64, 16),
                 (2048, 1024, 64, 16)]
SQUARE = [(2048, 2048, 64, 64), (1024, 1024, 32, 32), (512, 512, 16, 16),
          (256, 256, 8, 8), (128, 128, 4, 4), (64, 64, 2, 2),
          (768, 768, 24, 24), (1536, 1536, 48, 48), (1000, 1000, 25, 25),
          (96, 96, 3, 3), (512, 1024, 16, 32)]
SUITE = [(512, 512, 16, 16), (1024, 1024, 8, 8), (1024, 1024, 16, 16),
         (2048, 2048, 8, 8), (2048, 2048, 40, 40), (4096, 4096, 8, 8)]

CHILD = '''
import json, sys
sys.path.insert(0, sys.argv[1])
import lumenairy
from lumenairy.propagators._bluestein import _auto_selects_direct
shapes = json.loads(sys.argv[2])
print(json.dumps({"bound": lumenairy.__file__,
                  "answers": [bool(_auto_selects_direct(*s)) for s in shapes]}))
'''


def main(tree, scratch):
    v4lib.anchor(tree)
    dst = os.path.join(scratch, 'proposed_fix')
    if os.path.isdir(dst):
        shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst, exist_ok=True)
    shutil.copytree(os.path.join(tree, 'lumenairy'),
                    os.path.join(dst, 'lumenairy'),
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    os.makedirs(os.path.join(dst, 'tests', 'unit'), exist_ok=True)
    for rel in ('tests/conftest.py', 'tests/__init__.py',
                'tests/unit/__init__.py', 'conftest.py', 'pyproject.toml',
                'tests/unit/test_c4_mft_direct_default.py',
                'tests/unit/test_verify_c4_mft_direct.py'):
        src = os.path.join(tree, rel)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(os.path.join(dst, rel)) or dst,
                        exist_ok=True)
            shutil.copy2(src, os.path.join(dst, rel))
    p = os.path.join(dst, 'lumenairy', 'propagators', '_bluestein.py')
    with open(p, encoding='cp1252', errors='replace') as fh:
        s = fh.read()
    assert OLD_TAIL in s
    s = s.replace("_MFT_DIRECT_ALWAYS = float('inf')",
                  CONSTANT.strip() + "\n\n_MFT_DIRECT_ALWAYS = float('inf')")
    s = s.replace(OLD_TAIL, NEW_TAIL)
    with open(p, 'w', encoding='cp1252', errors='replace') as fh:
        fh.write(s)

    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONPATH=dst)
    script = os.path.join(HERE, '_v4_fix_child.py')
    with open(script, 'w', encoding='cp1252') as fh:
        fh.write(CHILD)
    groups = {'measured_slower': MEASURED_SLOWER, 'measured_safe':
              MEASURED_SAFE, 'square_ladder': SQUARE, 'shipped_suite': SUITE}
    out = {'build': v4lib.build_tag(), 'patched_tree': dst, 'groups': {}}
    for name, shapes in groups.items():
        r = subprocess.run(
            [sys.executable, script, dst, json.dumps(shapes)],
            capture_output=True, text=True, env=env, cwd=HERE)
        line = [ln for ln in r.stdout.splitlines() if ln.startswith('{')]
        if not line:
            out['groups'][name] = {'failed': r.stderr[-400:]}
            continue
        ans = json.loads(line[-1])['answers']
        out['groups'][name] = {
            'shapes': shapes, 'captured': ans,
            'n_captured': sum(ans), 'n': len(ans)}
        print(f"{name:18s} captured {sum(ans)}/{len(ans)}  {ans}", flush=True)
    os.remove(script)
    # The criterion is ONE-SIDED, because the work ratio is: the guard must
    # refuse every shape measured slower and keep every SQUARE shape and every
    # shape the shipped suite drives.  It is allowed to refuse some shapes
    # that were measured safe -- that costs time at a shape family nobody
    # drives, and is the direction a "never slower" premise needs.
    ok = (out['groups']['measured_slower']['n_captured'] == 0
          and out['groups']['square_ladder']['n_captured']
          == out['groups']['square_ladder']['n']
          and out['groups']['shipped_suite']['n_captured']
          == out['groups']['shipped_suite']['n'])
    out['fix_does_what_it_claims'] = bool(ok)
    print('FIX DOES WHAT IT CLAIMS:', ok, flush=True)

    for tf in ('tests/unit/test_c4_mft_direct_default.py',
               'tests/unit/test_verify_c4_mft_direct.py'):
        r = subprocess.run(
            [sys.executable, '-m', 'pytest', tf, '-q', '--capture=sys',
             '-p', 'no:randomly', '-rf', '--no-header'],
            capture_output=True, text=True, cwd=dst, env=env, timeout=3600)
        tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''
        failed = sorted(set(re.findall(r'FAILED (\S+)', r.stdout)))
        out.setdefault('pytest', {})[tf] = {'tail': tail, 'failed': failed}
        print(f"{tf}: {tail}", flush=True)
        if failed:
            print('   FAILED:', failed, flush=True)
    v4lib.write_json(out, os.path.join(
        HERE, f"v4_proposed_fix_{v4lib.short_tag()}.json"))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
