"""VERIFY-WP-C4 claim 11 -- the mutation matrix, reproduced and EXTENDED.

Every mutation is applied to a COPY of the tree under the session scratchpad
(nothing in ``lumenairy/`` is edited), and the SHIPPED test file
``tests/unit/test_c4_mft_direct_default.py`` is then run against that copy.
What is recorded is which ids fail -- so "caught by a NAMED id" is a
measurement.

The branch's three:

* ``rule_inverted``        -- the rule answers the opposite of what it measured
* ``constant_silently_zero`` -- the constant is 0.0 with the name unchanged
* ``dense_arm_is_separable`` -- the dense arm returns the separable answer

Four of mine, each a thing a maintainer could plausibly do to this rule:

* ``conjunction_max_to_min`` -- ``min`` instead of ``max`` over the two axes,
  which captures every shape with ONE decimating axis; this is the mutation
  the shipped boundary claim's third assertion exists for, and it is worth
  measuring because the shipped ladder contains no anisotropic shape;
* ``boundary_lt_not_le``    -- ``<`` instead of ``<=``, so the boundary ratio
  itself falls out of the dense region;
* ``swapped_axis_arguments`` -- the two CALL SITES pass ``(Ny, Nx, N_out_x,
  N_out_y)``, a y/x transposition that is invisible at a square shape;
* ``constant_1_over_31``    -- the constant retuned to a nearby value with no
  re-measurement, which must NOT be caught by anything (the claims are written
  to be relation-based, so a retune is legal) -- the control for the matrix's
  own sensitivity.

    python v4_mutations.py <tree> <scratch_root>
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
TESTFILE = 'tests/unit/test_c4_mft_direct_default.py'


def _sub(text, old, new, count=0):
    assert old in text, f"anchor not found: {old[:80]!r}"
    return text.replace(old, new) if not count else text.replace(old, new,
                                                                 count)


MUTATIONS = {
    'control': lambda s: s,
    'rule_inverted': lambda s: _sub(
        s, "    return max(my / ny, mx / nx) <= r",
        "    return not (max(my / ny, mx / nx) <= r)"),
    'constant_silently_zero': lambda s: _sub(
        s, "_MFT_DIRECT_MAX_RATIO = 1.0 / 32.0",
        "_MFT_DIRECT_MAX_RATIO = 0.0"),
    'dense_arm_is_separable': lambda s: _sub(
        s, """    if cost_y_first <= cost_x_first:
        F = xp.matmul(xp.matmul(Wy, A), Wx.T)
    else:
        F = xp.matmul(Wy, xp.matmul(A, Wx.T))""",
        """    F = _bluestein_2d_separable(
        E, alpha_x, alpha_y, N_out_y, N_out_x, sign=sign,
        target_cdtype=target_cdtype)
    if False:
        F = xp.matmul(xp.matmul(Wy, A), Wx.T)"""),
    'warning_message_lies': lambda s: _sub(
        s, """    _warn_phase_budget(alpha_x, alpha_y, Nx_in, Ny_in, N_out_x, N_out_y,
                       on_dense=auto_direct, stacklevel=3)""",
        """    _warn_phase_budget(alpha_x, alpha_y, Nx_in, Ny_in, N_out_x, N_out_y,
                       on_dense=False, stacklevel=3)"""),
    'conjunction_max_to_min': lambda s: _sub(
        s, "    return max(my / ny, mx / nx) <= r",
        "    return min(my / ny, mx / nx) <= r"),
    'boundary_lt_not_le': lambda s: _sub(
        s, "    return max(my / ny, mx / nx) <= r",
        "    return max(my / ny, mx / nx) < r"),
    # NOTE the ``and `` prefix: without it the pattern also matches the
    # function's own ``def`` line, which renames its PARAMETERS in the same
    # order and makes the whole mutation a no-op.  (It did, on the first
    # attempt, and the matrix duly reported "not caught" about nothing.)
    'swapped_axis_arguments': lambda s: _sub(
        s, "and _auto_selects_direct(Ny_in, Nx_in, N_out_y, N_out_x))",
        "and _auto_selects_direct(Ny_in, Nx_in, N_out_x, N_out_y))"),
    'constant_1_over_31': lambda s: _sub(
        s, "_MFT_DIRECT_MAX_RATIO = 1.0 / 32.0",
        "_MFT_DIRECT_MAX_RATIO = 1.0 / 31.0"),
}


def run_one(tree, scratch, name, fn):
    dst = os.path.join(scratch, f"mut_{name}")
    if os.path.isdir(dst):
        shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst, exist_ok=True)
    shutil.copytree(os.path.join(tree, 'lumenairy'),
                    os.path.join(dst, 'lumenairy'),
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    os.makedirs(os.path.join(dst, 'tests', 'unit'), exist_ok=True)
    for rel in ('tests/conftest.py', 'tests/__init__.py',
                'tests/unit/__init__.py', TESTFILE, 'conftest.py',
                'pytest.ini', 'setup.cfg', 'pyproject.toml'):
        src = os.path.join(tree, rel)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(os.path.join(dst, rel)) or dst,
                        exist_ok=True)
            shutil.copy2(src, os.path.join(dst, rel))
    p = os.path.join(dst, 'lumenairy', 'propagators', '_bluestein.py')
    with open(p, encoding='cp1252', errors='replace') as fh:
        s = fh.read()
    s2 = fn(s)
    assert (s2 == s) == (name == 'control'), \
        f"{name}: the mutation changed nothing"
    with open(p, 'w', encoding='cp1252', errors='replace') as fh:
        fh.write(s2)
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONPATH=dst)
    r = subprocess.run(
        [sys.executable, '-m', 'pytest', TESTFILE, '-q', '--capture=sys',
         '-p', 'no:randomly', '-rf', '--no-header', '-x' if False else '-q'],
        capture_output=True, text=True, cwd=dst, env=env, timeout=3600)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''
    failed = sorted(set(re.findall(r'FAILED (\S+)', r.stdout)
                        + re.findall(r'ERROR (\S+)', r.stdout)))
    short = sorted({f.split('::')[-1].split('[')[0] for f in failed})
    return {'mutation': name, 'tail': tail, 'returncode': r.returncode,
            'failed_ids': failed, 'failed_id_names': short,
            'n_failed': len(failed), 'tree': dst}


def main(tree, scratch):
    v4lib.anchor(tree)
    out = {'build': v4lib.build_tag(), 'testfile': TESTFILE, 'rows': []}
    only = os.environ.get('V4_MUT_ONLY', '')
    for name, fn in MUTATIONS.items():
        if only and name not in only.split(','):
            continue
        rec = run_one(tree, scratch, name, fn)
        out['rows'].append(rec)
        print(f"{name:26s} {rec['tail']}", flush=True)
        print(f"{'':26s} caught by: {rec['failed_id_names']}", flush=True)
    v4lib.write_json(out, os.path.join(
        HERE, f"v4_mutations_{v4lib.short_tag()}.json"))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
