"""VERIFY-WP-C3 -- the mutation harness for ``tests/unit/test_c3_collins_default.py``.

INDEPENDENT of the branch's own ``vh3_mutate.py``: this one drives a SEPARATE
tree (``C:/tmp/vc3_mutE``) whose ``lumenairy/propagators/carrier.py`` is
restored from a pristine copy before every arm, so no arm can see another's
edit.  Each arm is an EXACT string substitution (no regex), asserted to have
fired the expected number of times -- a mutation that did not apply would
otherwise read as "the suite caught nothing", which is the wrong conclusion.

Run:  python vc3_mutate.py [arm ...]
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

TREE = os.environ.get('MUT_TREE', r'C:/tmp/vc3_mutE')
SRC = os.path.join(TREE, 'lumenairy', 'propagators', 'carrier.py')
PRISTINE = os.path.join(TREE, 'carrier.py.pristine')
TESTFILE = 'tests/unit/test_c3_collins_default.py'

# arm -> list of (old, new, expected_count)
ARMS = {
    # ---- the branch's OWN named matrix -------------------------------
    'B1_default_reverted': [
        ("    transport: str = 'collins',", "    transport: str = 'sziklas',", 3),
    ],
    'B5_route_resolution_deleted': [
        ("        if _collins_ro and _ro_named:\n"
         "            _collins_ro = False\n"
         "        elif _collins_ro:\n"
         "            _ro_k1 = _collins_readout_k1(env, R, final_distance, wavelength,\n"
         "                                         cur_dx, cur_dy)\n"
         "            if not (_ro_k1 <= 1.0):\n"
         "                _collins_ro = False\n",
         "        if False:\n"
         "            _collins_ro = False\n", 1),
    ],
    'B6_internal_caller_rides_default': [
        ("                    env, R, gap, wavelength, cur_dx, gap_kernel=gap_kernel,\n"
         "                    tilt=_leg_tilt, transport='sziklas')",
         "                    env, R, gap, wavelength, cur_dx, gap_kernel=gap_kernel,\n"
         "                    tilt=_leg_tilt)", 1),
    ],
    # ---- MY OWN arms --------------------------------------------------
    'V1_readout_bar_2': [
        ("            if not (_ro_k1 <= 1.0):", "            if not (_ro_k1 <= 2.0):", 1),
    ],
    'V2_reason_always_representable': [
        ("    stage['readout_route_reason'] = (\n"
         "        'representable' if took_collins\n"
         "        else ('k1' if k1 is not None else 'stop_plane_key'))",
         "    stage['readout_route_reason'] = 'representable'", 1),
    ],
    'V2b_reason_swapped': [
        ("    stage['readout_route_reason'] = (\n"
         "        'representable' if took_collins\n"
         "        else ('k1' if k1 is not None else 'stop_plane_key'))",
         "    stage['readout_route_reason'] = (\n"
         "        'k1' if took_collins\n"
         "        else ('representable' if k1 is not None else 'stop_plane_key'))", 1),
    ],
    'V3_fallback_is_step_fast_again': [
        ("        cr = propagate_carrier_referenced(\n"
         "            env_a, ((R_x, R_y) if is_astig else R_x), z, wavelength, dx, dy,\n"
         "            gap_kernel=gap_kernel, tilt=tilt, transport='sziklas')",
         "        cr = _carrier_step_fast(env_a, R_x, z, wavelength, dx, dy,\n"
         "                                gap_kernel=gap_kernel, tilt=tilt)", 1),
    ],
    'V4_cupy_branch_takes_numpy_fft': [
        ("    fft2, ifft2 = _fft2_pair(xp, is_jax)",
         "    fft2, ifft2 = np.fft.fft2, np.fft.ifft2", 1),
    ],
    'V5_standoff_leg_unpinned': [
        ("    cr = propagate_carrier_referenced(env, R, z_stop, wavelength, dx,\n"
         "                                      gap_kernel=gap_kernel, tilt=tilt,\n"
         "                                      transport='sziklas')",
         "    cr = propagate_carrier_referenced(env, R, z_stop, wavelength, dx,\n"
         "                                      gap_kernel=gap_kernel, tilt=tilt)", 1),
    ],
    'V6_route_keys_on_sziklas_too': [
        ("    if transport != 'collins':\n        return\n", "    if False:\n        return\n", 1),
    ],
    'V7_stop_plane_keys_refused_again': [
        ("        if _collins_ro and _ro_named:\n"
         "            _collins_ro = False\n",
         "        if _collins_ro and _ro_named:\n"
         "            raise ValueError(f\"{_fn}: stop-plane keys are refused on collins\")\n", 1),
    ],
    # ---- the near-focus / C5 interaction arm -------------------------
    'X1_c5_tau_armed': [
        ("_GAP_KERNEL_ACCURACY_TAU = None", "_GAP_KERNEL_ACCURACY_TAU = 1e-4", 1),
    ],
}


def _restore():
    shutil.copyfile(PRISTINE, SRC)


def _apply(arm):
    subs = ARMS[arm]
    with open(SRC, encoding='utf-8', newline='') as fh:
        src = fh.read()
    nl = '\r\n' if '\r\n' in src else '\n'
    flat = src.replace('\r\n', '\n')
    for old, new, want in subs:
        got = flat.count(old)
        assert got == want, (arm, 'expected %d occurrences, found %d' % (want, got),
                             old[:80])
        flat = flat.replace(old, new)
    with open(SRC, 'w', encoding='utf-8', newline='') as fh:
        fh.write(flat.replace('\n', nl) if nl == '\r\n' else flat)


def _run(node=None):
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONPATH=TREE)
    nodes = (node or os.environ.get('MUT_NODES') or TESTFILE).split(',')
    cmd = [sys.executable, '-m', 'pytest', *nodes,
           '--capture=sys', '-p', 'no:randomly', '-q',
           '--no-header', '-rf']
    p = subprocess.run(cmd, cwd=TREE, env=env, capture_output=True, text=True)
    out = p.stdout + p.stderr
    failed = sorted(set(re.findall(r'^(?:FAILED|ERROR) (\S+?)(?: - |$)', out, re.M)))
    tail = [l for l in out.splitlines() if l.strip()][-1] if out.strip() else ''
    return {'failed': failed, 'tail': tail, 'rc': p.returncode}


def _locate_v4(pristine_text):
    """Find the Collins path's FFT dispatch call and hard-code numpy."""
    m = re.search(r'^(\s*)S = (\S+)\(env_a\)', pristine_text, re.M)
    return m


def main(argv):
    arms = argv[1:] or [a for a in ARMS if ARMS[a] and ARMS[a][0]]
    res = {}
    for arm in arms:
        _restore()
        try:
            _apply(arm)
        except AssertionError as exc:
            res[arm] = {'apply_error': str(exc)}
            print(arm, 'APPLY FAILED', exc)
            continue
        r = _run()
        res[arm] = r
        print('%-38s rc=%d  %s' % (arm, r['rc'], r['tail']))
        for f in r['failed']:
            print('        FAILED', f)
    _restore()
    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        os.environ.get('MUT_OUT', 'mutation_matrix_win.json'))
    with open(outp, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1)
    print('wrote', outp)


if __name__ == '__main__':
    main(sys.argv)
