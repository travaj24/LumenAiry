"""The MUTATION MATRIX for the shipped decision tests (claim 8 / claim 9).

Every mutation is a plausible regression of the arbiter, applied to a
SEPARATE detached worktree (``C:/tmp/lum_vmb2_mut``) -- ``lumenairy/`` on the
verification branch is never touched.  For each mutation the pinned decision
tests are run against that tree with ``PYTHONPATH`` pinning it, and the
result recorded.  A mutation that leaves the suite GREEN is a gap in the
gate.

Usage:
    python vmutate.py <out.json> [mutation-id ...]
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

MUT_TREE = os.environ.get('MUT_TREE', r'C:\tmp\lum_vmb2_mut')
MB = os.path.join(MUT_TREE, 'lumenairy', 'elements',
                  '_lens_traced_multibranch.py')
UN = os.path.join(MUT_TREE, 'lumenairy', 'elements', '_lens_traced_uniform.py')

CORE = ['tests/unit/test_audit2609_b7c2_pixel_halving_arbiter.py',
        'tests/unit/test_verify_b7c_multibranch.py',
        'tests/unit/test_audit2609_b7c_multibranch_envelope.py']
#: the same three PLUS this verification's own file, used to prove that the
#: two gaps M2 and M3 open are actually closed by it.
CORE_PLUS = CORE + ['tests/unit/test_verify_b7c_round2.py']
SEAM = ['tests/unit/test_niche_r2_pearcey_cusp.py',
        'tests/unit/test_niche_r5_gbd_vector_catastrophe.py']


def sub(path, old, new, count=1):
    with open(path, encoding='cp1252') as f:
        s = f.read()
    n = s.count(old)
    if n < 1:
        raise SystemExit(f'anchor not found in {path}: {old[:70]!r}')
    s = s.replace(old, new, count)
    with open(path, 'w', encoding='cp1252') as f:
        f.write(s)
    return n


# --------------------------------------------------------------------------
# The mutations.  Each is (id, description, apply-fn, test files).
# --------------------------------------------------------------------------
def m_alias():
    """M1 -- an ALIAS that bypasses ``_multibranch_render``: the completion
    reaches the branch sum through the PUBLIC entry point again, so the
    arbiter is never asked and the reading is never taken."""
    sub(UN, '    _multibranch_render,\n',
        '    apply_real_lens_traced_multibranch'
        ' as apply_real_lens_traced_multibranch,\n')
    sub(UN, '''    E_mb, mb_diag = _multibranch_render(''',
        '''    def _multibranch_render(*a, **kw):
        kw.pop('pixel_halving_arbiter', None)
        return apply_real_lens_traced_multibranch(*a, **kw)

    E_mb, mb_diag = _multibranch_render(''')


def m_branch_sum():
    """M2 -- the arbiter reads the BRANCH SUM instead of the returned
    completed field."""
    sub(UN, "    _arbitrate(_uni_cont, 'the completed fold field')",
        "    _arbitrate(_mb_cont, 'the completed fold field')")


def m_bar_120():
    """M3 -- the bar is loosened from 1.06 to 1.20."""
    sub(MB, '_PIXEL_CONTINUITY_MAX = 1.06', '_PIXEL_CONTINUITY_MAX = 1.20')


def m_same_grid():
    """M4 -- the second render is taken at ``(N, dx)`` instead of
    ``(2 N, dx / 2)``, so the reading is identically 1 and nothing is ever
    refused."""
    sub(MB, '_E_half, _nb_half, _ = _render(2 * N, 0.5 * dx)',
        '_E_half, _nb_half, _ = _render(N, dx)')
    sub(MB, '''            _p_out_half = float(np.sum(np.abs(_E_half) ** 2)) * (
                0.25 * dx * dx)''',
        '''            _p_out_half = float(np.sum(np.abs(_E_half) ** 2)) * (
                dx * dx)''')
    # the completion's own half-pitch fill must use the same sampling
    sub(UN, '        _N_h, _dx_h = 2 * N, 0.5 * dx',
        '        _N_h, _dx_h = N, dx')


def m_loss_refuses():
    """M5 -- the LOSS arm is made to refuse as well (the asymmetry the report
    justifies physically is removed)."""
    sub(UN, "        if d != 'not_converged_gain':\n            return",
        "        if d not in ('not_converged_gain', 'not_converged_loss'):\n"
        "            return")


def m_band_asymmetric():
    """M6 -- the band stops being symmetric in the log of the ratio."""
    sub(MB, '_PIXEL_CONTINUITY_MIN = 1.0 / _PIXEL_CONTINUITY_MAX',
        '_PIXEL_CONTINUITY_MIN = 0.5')


def m_cap_1e6():
    """M7 -- the entry cap is lowered to 1e6 entries (fine grid N = 500), so
    the reading disappears on exactly the refined grid D2 is about."""
    sub(MB, '_ARBITER_MAX_FINE_ENTRIES = 7_000_000',
        '_ARBITER_MAX_FINE_ENTRIES = 1_000_000')


def m_not_requested():
    """M8 -- the completion stops asking for the reading."""
    sub(UN, '        pixel_halving_arbiter=True)',
        '        pixel_halving_arbiter=False)')


def m_inverted():
    """M9 -- the ratio is inverted (half-pitch over coarse)."""
    sub(MB, '                _continuity = p_out / _p_out_half',
        '                _continuity = _p_out_half / p_out')
    sub(UN, '            _uni_cont = _p_coarse / _p_half',
        '            _uni_cont = _p_half / _p_coarse')


def m_fallback_silent():
    """M10 -- the FALLBACK path stops arbitrating at all (the returned branch
    sum is no longer read)."""
    sub(UN, "        _arbitrate(_mb_cont, 'the plain multibranch field')",
        "        _arbitrate(None, 'the plain multibranch field')")


def m_none_is_ok():
    """M11 -- an unmeasurable reading silently becomes 'ok' instead of
    'not_measured'."""
    sub(UN, """            d = mb_diag.get('pixel_continuity_decision') or 'not_measured'
            return d if d in ('not_requested', 'not_measured')                 else 'not_measured'""",
        """            return 'ok'""")


def m_cap_2e6():
    """M13 -- the entry cap is lowered to 2e6 entries, which keeps every
    PUBLISHED fixture (N = 512 / 640 / 768) inside it but puts D2's refined
    grid (N = 1280 -> 6.55e6 entries) out of reach.  The subtler half of
    M7."""
    sub(MB, '_ARBITER_MAX_FINE_ENTRIES = 7_000_000',
        '_ARBITER_MAX_FINE_ENTRIES = 2_000_000')


def m_power_bar_off():
    """M12 -- the cheap launched-power tripwire is disabled, leaving only the
    continuity arm (does anything still pin the first arm?)."""
    sub(UN, '_MB_POWER_RATIO_MAX = _ENERGY_BLOWUP_FACTOR',
        '_MB_POWER_RATIO_MAX = 1e9')


MUTATIONS = [
    ('M1_alias_bypasses_seam', m_alias, CORE + SEAM),
    ('M2_reads_branch_sum', m_branch_sum, CORE),
    ('M2plus_reads_branch_sum', m_branch_sum, CORE_PLUS),
    ('M3_bar_1_20', m_bar_120, CORE),
    ('M3plus_bar_1_20', m_bar_120, CORE_PLUS),
    ('M4plus_reading_at_N_dx', m_same_grid, CORE_PLUS),
    ('M13plus_entry_cap_2e6', m_cap_2e6, CORE_PLUS),
    ('M4_reading_at_N_dx', m_same_grid, CORE),
    ('M5_loss_arm_refuses', m_loss_refuses, CORE),
    ('M6_band_asymmetric', m_band_asymmetric, CORE),
    ('M7_entry_cap_1e6', m_cap_1e6, CORE),
    ('M8_arbiter_not_requested', m_not_requested, CORE),
    ('M9_ratio_inverted', m_inverted, CORE),
    ('M10_fallback_not_arbitrated', m_fallback_silent, CORE),
    ('M11_unmeasured_becomes_ok', m_none_is_ok, CORE),
    ('M12_power_tripwire_off', m_power_bar_off, CORE),
    ('M13_entry_cap_2e6', m_cap_2e6, CORE),
]

TAIL = re.compile(r'(\d+ (?:passed|failed|error)[^\n]*)')


def restore(tries=6):
    import time as _t
    for i in range(tries):
        p = subprocess.run(['git', '-C', MUT_TREE, 'checkout', '--',
                            'lumenairy/'], capture_output=True, text=True)
        if p.returncode == 0:
            return
        _t.sleep(5 * (i + 1))
    raise SystemExit(f'restore failed: {p.stderr[-400:]}')


def run_tests(files, timeout=1800):
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONPATH=MUT_TREE)
    p = subprocess.run(
        [sys.executable, '-m', 'pytest', *files, '-q', '--capture=sys',
         '-p', 'no:randomly', '--no-header', '-x' if False else '--tb=line'],
        cwd=MUT_TREE, env=env, capture_output=True, text=True,
        timeout=timeout)
    out = p.stdout + p.stderr
    tail = [ln for ln in out.splitlines()
            if re.search(r'\b(passed|failed|error|no tests ran)\b', ln)]
    fails = sorted({ln.split('::')[1].split()[0].split('[')[0]
                    for ln in out.splitlines()
                    if '::' in ln and ('FAILED' in ln or 'ERROR' in ln)
                    and len(ln.split('::')) > 1})
    return dict(returncode=p.returncode, tail=tail[-3:], failed_ids=fails,
                nbytes=len(out)), out


def main():
    out_path = sys.argv[1]
    want = set(sys.argv[2:])
    rows = []
    restore()
    for mid, fn, files in MUTATIONS:
        if want and mid not in want:
            continue
        restore()
        t0 = time.time()
        try:
            fn()
        except SystemExit as e:
            rows.append(dict(id=mid, error=str(e)))
            print(mid, 'ANCHOR-FAIL', e, flush=True)
            continue
        try:
            res, raw = run_tests(files)
        except subprocess.TimeoutExpired:
            res, raw = dict(returncode=-9, tail=['TIMEOUT'], failed_ids=[]), ''
        res.update(id=mid, doc=(fn.__doc__ or '').strip().split('\n')[0],
                   files=files, seconds=round(time.time() - t0, 1))
        res['caught'] = bool(res['returncode'] != 0)
        rows.append(res)
        with open(os.path.join(os.path.dirname(os.path.abspath(out_path)),
                               f'mut_{mid}_win.txt'), 'w',
                  encoding='cp1252') as f:
            f.write(raw)
        print(json.dumps({k: res[k] for k in
                          ('id', 'caught', 'tail', 'failed_ids', 'seconds')}),
              flush=True)
        with open(out_path, 'w', encoding='cp1252') as f:
            json.dump(rows, f, indent=1)
    restore()
    print('DONE', out_path, flush=True)


if __name__ == '__main__':
    main()
