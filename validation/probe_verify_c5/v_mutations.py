"""VERIFY-WP-C5 -- the mutation matrix driver.

Applies ONE source mutation at a time to a scratch copy of the branch tip
(never to the worktree), runs the two C5 test files against it, and records
which ids caught it.  A mutation no id catches is a hole in the gate; an id
that ERRORS on every mutation is not discriminating either, so the unmutated
run is recorded as the control.

    python validation/probe_verify_c5/v_mutations.py <tree> <python> OUT.json
"""
import json
import os
import subprocess
import sys

MUTATIONS = {
    # item 1 -----------------------------------------------------------
    'item1_tau_disarmed': (
        'lumenairy/propagators/carrier.py',
        '_GAP_KERNEL_ACCURACY_TAU = 1e-4',
        '_GAP_KERNEL_ACCURACY_TAU = None'),
    'item1_wrong_angle': (
        'lumenairy/propagators/carrier.py',
        'th_ex, th_ey = _collins_envelope_half_angle(S, dx, dy, wavelength)',
        'th_ex, th_ey = th_x, th_y  # MUTANT: the CONTAINMENT half-angle'),
    # item 2 -----------------------------------------------------------
    'item2_floor_forgets_fixed_term': (
        'lumenairy/propagators/gbd.py',
        'return float(Ny) * float(Nx) * (_DENSE_FIXED_CELL_BYTES\n'
        '                                    + _DENSE_CELL_BYTES_MEASURED)',
        'return float(Ny) * float(Nx) * _DENSE_CELL_BYTES_MEASURED'),
    'item2_notice_suppressed': (
        'lumenairy/propagators/gbd.py',
        'if mem_budget_mb * 1e6 < _floor:',
        'if False and mem_budget_mb * 1e6 < _floor:'),
    'item2_unknown_mode_falls_through': (
        'lumenairy/propagators/gbd.py',
        'if mode not in _DENSE_MEM_BUDGET_ACCOUNTINGS:',
        'if False:  # MUTANT: silent fall-through to legacy'),
    'item2_default_reverted': (
        'lumenairy/propagators/gbd.py',
        "DENSE_MEM_BUDGET_ACCOUNTING = 'measured'",
        "DENSE_MEM_BUDGET_ACCOUNTING = 'legacy'"),
    # item 3 -----------------------------------------------------------
    'item3_fill_keyed_on_window_centre': (
        'lumenairy/propagators/carrier.py',
        '    keep_x = np.abs(u + cx) <= 0.5 * px * (1.0 + 1e-9)\n'
        '    keep_y = np.abs(u + cy) <= 0.5 * py * (1.0 + 1e-9)',
        '    keep_x = np.abs(u) <= 0.5 * px * (1.0 + 1e-9)\n'
        '    keep_y = np.abs(u) <= 0.5 * py * (1.0 + 1e-9)'),
    'item3_fill_keyed_on_half_the_period': (
        'lumenairy/propagators/carrier.py',
        '    keep_x = np.abs(u + cx) <= 0.5 * px * (1.0 + 1e-9)\n'
        '    keep_y = np.abs(u + cy) <= 0.5 * py * (1.0 + 1e-9)',
        '    keep_x = np.abs(u + cx) <= 0.25 * px * (1.0 + 1e-9)\n'
        '    keep_y = np.abs(u + cy) <= 0.25 * py * (1.0 + 1e-9)'),
    'item3_default_reverted': (
        'lumenairy/propagators/carrier.py',
        "replica_fill: str = 'zero'",
        "replica_fill: str = 'repeat'"),
    'item3_fill_on_the_refusal_path_collins': (
        'lumenairy/propagators/carrier.py',
        "    _check_readout_replica(\n"
        "        fn, period, dx_out, N_out, on_replica, centre_out=centre_out,",
        "    _check_readout_replica(\n"
        "        fn, period, dx_out, N_out, 'ignore', centre_out=centre_out,"),
    'item3_fill_on_the_refusal_path_sziklas': (
        'lumenairy/propagators/carrier.py',
        "        'carrier_referenced_focus_readout', _period, dx_out, N_out,\n"
        "        on_replica, centre_out=centre_out,",
        "        'carrier_referenced_focus_readout', _period, dx_out, N_out,\n"
        "        'ignore', centre_out=centre_out,"),
    'item3_fill_on_the_refusal_path_exact': (
        'lumenairy/propagators/carrier.py',
        "        'carrier_referenced_exact_focus_readout', _period, dx_out, "
        "N_out,\n        on_replica, centre_out=_co,",
        "        'carrier_referenced_exact_focus_readout', _period, dx_out, "
        "N_out,\n        'ignore', centre_out=_co,"),
}

FILES = ('tests/unit/test_c5_three_defaults.py',
         'tests/unit/test_verify_c5_three_defaults.py')


def run(tree, py, extra_env=None):
    env = dict(os.environ)
    env.update(dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                    MKL_NUM_THREADS='1', LUMENAIRY_MEM_BUDGET_MB='2048',
                    PYTHONPATH=tree))
    if extra_env:
        env.update(extra_env)
    p = subprocess.run(
        [py, '-m', 'pytest', *FILES, '--capture=sys', '-p', 'no:randomly',
         '-q', '--no-header', '-rf'],
        cwd=tree, env=env, capture_output=True, text=True, timeout=7200)
    tail = (p.stdout or '')[-9000:]
    caught = sorted({ln.split('::', 1)[1].split(' ')[0]
                     for ln in tail.splitlines()
                     if ln.startswith('FAILED') and '::' in ln})
    last = [ln for ln in (p.stdout or '').splitlines()
            if ('passed' in ln or 'failed' in ln or 'error' in ln
                or 'no tests ran' in ln)]
    return dict(rc=p.returncode, summary=(last[-1] if last else '??'),
                n_caught=len(caught), caught=caught)


def main(tree, py, out_path):
    res = {'tree': tree, 'python': py, 'mutations': {}}
    res['control'] = run(tree, py)
    print('CONTROL:', res['control']['summary'])
    for name, (rel, old, new) in MUTATIONS.items():
        path = os.path.join(tree, rel)
        with open(path, encoding='utf-8') as fh:
            src = fh.read()
        n = src.count(old)
        if n == 0:
            res['mutations'][name] = {'error': 'pattern not found', 'count': 0}
            print('SKIP (no pattern):', name)
            continue
        try:
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(src.replace(old, new))
            r = run(tree, py)
            r['n_sites_mutated'] = n
            res['mutations'][name] = r
            print('%-40s %-3d sites  %s  caught %d' %
                  (name, n, r['summary'], r['n_caught']))
        finally:
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(src)
    with open(out_path, 'w', encoding='cp1252', errors='replace') as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print('WROTE', out_path)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
