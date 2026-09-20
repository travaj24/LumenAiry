"""The round-3 MUTATION MATRIX.

The round-2 verification ran thirteen plausible regressions of the arbiter
against the three shipped decision files and found two that left the suite
green (M2 -- read the branch sum; M3 -- loosen the bar to 1.20) plus one held
only through a premise gate that can skip (M13).  Its own file closed M2 and
M3.  This round adds four pins of its own, so it re-runs the WHOLE matrix --
the verifier's thirteen and four new -- against the whole gate, on both
builds, and requires every NEW pin to FAIL (not skip) under the regression it
exists to catch.

Every mutation is applied to a SEPARATE detached worktree; ``lumenairy/`` on
the round-3 branch is never touched.  ``MUT_TREE`` names that worktree.

Usage:
    MUT_TREE=C:/tmp/lum_mb3_mut python r3mutate.py <out.json> [id ...]
    MUT_TREE=... R3_PY=<python> python r3mutate.py ...     (the wsl build)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

MUT_TREE = os.environ.get('MUT_TREE', r'C:\tmp\lum_mb3_mut')
PY = os.environ.get('R3_PY', sys.executable)
MB = os.path.join(MUT_TREE, 'lumenairy', 'elements',
                  '_lens_traced_multibranch.py')
UN = os.path.join(MUT_TREE, 'lumenairy', 'elements', '_lens_traced_uniform.py')

#: the WHOLE gate for this module: the three files round 2 shipped, the
#: verification's file, and this round's.
CORE = ['tests/unit/test_audit2609_b7c2_pixel_halving_arbiter.py',
        'tests/unit/test_verify_b7c_multibranch.py',
        'tests/unit/test_audit2609_b7c_multibranch_envelope.py',
        'tests/unit/test_verify_b7c_round2.py',
        'tests/unit/test_wp_b7c_round3.py']
SEAM = ['tests/unit/test_niche_r2_pearcey_cusp.py',
        'tests/unit/test_niche_r5_gbd_vector_catastrophe.py']


def sub(path, old, new, count=1):
    with open(path, encoding='cp1252') as f:
        s = f.read()
    if s.count(old) < 1:
        raise SystemExit(f'anchor not found in {path}: {old[:80]!r}')
    with open(path, 'w', encoding='cp1252') as f:
        f.write(s.replace(old, new, count))


# --------------------------------------------------------------------------
# the round-2 verifier's thirteen, re-applied to this tree
# --------------------------------------------------------------------------
def m_alias():
    """M1 -- an ALIAS that bypasses ``_multibranch_render``."""
    sub(UN, '    _multibranch_render,\n',
        '    apply_real_lens_traced_multibranch'
        ' as apply_real_lens_traced_multibranch,\n')
    sub(UN, '''    E_mb, mb_diag = _multibranch_render(''',
        '''    def _multibranch_render(*a, **kw):
        kw.pop('pixel_halving_arbiter', None)
        return apply_real_lens_traced_multibranch(*a, **kw)

    E_mb, mb_diag = _multibranch_render(''')


def m_branch_sum():
    """M2 -- the arbiter reads the BRANCH SUM instead of the returned field."""
    sub(UN, "    _arbitrate(_uni_cont, 'the completed fold field', ",
        "    _arbitrate(_mb_cont, 'the completed fold field', ")


def m_bar_120():
    """M3 -- the bar is loosened to 1.20."""
    sub(MB, '_PIXEL_CONTINUITY_MAX = ', '_PIXEL_CONTINUITY_MAX = 1.20  # ')


def m_same_grid():
    """M4 -- the second render is taken at ``(N, dx)``."""
    sub(MB, '_E_half, _nb_half, _ = _render(2 * N, 0.5 * dx)',
        '_E_half, _nb_half, _ = _render(N, dx)')
    sub(MB, '''            _p_out_half = float(np.sum(np.abs(_E_half) ** 2)) * (
                0.25 * dx * dx)''',
        '''            _p_out_half = float(np.sum(np.abs(_E_half) ** 2)) * (
                dx * dx)''')
    sub(UN, '        _N_h, _dx_h = 2 * N, 0.5 * dx',
        '        _N_h, _dx_h = N, dx')


def m_loss_refuses():
    """M5 -- the LOSS arm is made to refuse too."""
    sub(UN, "        if d != 'not_converged_gain':\n            return",
        "        if d not in ('not_converged_gain', 'not_converged_loss'):\n"
        "            return")


def m_band_asymmetric():
    """M6 -- the band stops being symmetric in the log of the ratio."""
    sub(MB, '_PIXEL_CONTINUITY_MIN = 1.0 / _PIXEL_CONTINUITY_MAX',
        '_PIXEL_CONTINUITY_MIN = 0.5')


def m_cap_1e6():
    """M7 -- the entry cap is lowered to 1e6 entries."""
    sub(MB, '_ARBITER_MAX_FINE_ENTRIES = 7_000_000',
        '_ARBITER_MAX_FINE_ENTRIES = 1_000_000')


def m_not_requested():
    """M8 -- the completion stops asking for the reading."""
    sub(UN, '        pixel_halving_arbiter=True)',
        '        pixel_halving_arbiter=False)')


def m_inverted():
    """M9 -- the ratio is inverted."""
    sub(MB, '                _continuity = p_out / _p_out_half',
        '                _continuity = _p_out_half / p_out')
    sub(UN, '            _uni_cont = _p_coarse / _p_half',
        '            _uni_cont = _p_half / _p_coarse')


def m_fallback_silent():
    """M10 -- the FALLBACK path stops arbitrating."""
    sub(UN, "        _arbitrate(_mb_cont, 'the plain multibranch field',",
        "        _arbitrate(None, 'the plain multibranch field',")


def m_none_is_ok():
    """M11 -- an unmeasurable reading silently becomes 'ok'."""
    sub(UN, """            d = mb_diag.get('pixel_continuity_decision') or 'not_measured'
            return d if d in ('not_requested', 'not_measured')                 else 'not_measured'""",
        """            return 'ok'""")


def m_power_bar_off():
    """M12 -- the launched-power tripwire is disabled."""
    sub(UN, '_MB_POWER_RATIO_MAX = _ENERGY_BLOWUP_FACTOR',
        '_MB_POWER_RATIO_MAX = 1e9')


def m_cap_2e6():
    """M13 -- the entry cap is lowered to 2e6 entries."""
    sub(MB, '_ARBITER_MAX_FINE_ENTRIES = 7_000_000',
        '_ARBITER_MAX_FINE_ENTRIES = 2_000_000')


# --------------------------------------------------------------------------
# round 3's own four
# --------------------------------------------------------------------------
def m_unnested_centres():
    """M14 (R3-3) -- the half-pitch lattice stops NESTING on the caller's: the
    hull-aligned alternative, which makes the two Voronoi hulls coincide and
    the ~4-per-halving identity unbounded."""
    sub(MB, '_HALF_PITCH_CENTRE_OFFSET = 0.0',
        '_HALF_PITCH_CENTRE_OFFSET = -0.5')


def m_lattice_rebuilt_inline():
    """M18 (R3-3) -- the completion rebuilds the half-pitch lattice inline
    instead of taking it from the one definition, which is the seam the two
    renders can drift apart at."""
    sub(UN, '        _xh = half_pitch_centres(N, dx)',
        '        _xh = (np.arange(_N_h) - _N_h / 2.0) * _dx_h')


def m_pearcey_label():
    """M15 (R3-2) -- the Pearcey cusp route goes back to labelling the branch
    sum's reading as the Pearcey field's own."""
    sub(UN, "_arbitrate(_mb_cont, _PEARCEY_READING_OF,",
        "_arbitrate(_mb_cont, 'the Pearcey cusp field',")
    sub(UN, "                               'underlying_branch_sum')",
        "                               'returned_field')")


def m_scope_always_returned():
    """M16 (R3-4) -- every route claims the reading arbitrates the field it
    returns, so a fallback plane's missing dark tail is reported as
    arbitrated."""
    sub(UN, "        d = _continuity_verdict(c)\n"
            "        _mb_energy['pixel_continuity'] =",
        "        scope = 'returned_field'\n"
        "        d = _continuity_verdict(c)\n"
        "        _mb_energy['pixel_continuity'] =")


def m_bar_inside_spread():
    """M17 (R3-1) -- the bar is put inside the converged reading's own
    measured spread, where it is noise rather than a tolerance."""
    sub(MB, '_PIXEL_CONTINUITY_MAX = ', '_PIXEL_CONTINUITY_MAX = 1.004  # ')


MUTATIONS = [
    ('M1_alias_bypasses_seam', m_alias, CORE + SEAM),
    ('M2_reads_branch_sum', m_branch_sum, CORE),
    ('M3_bar_1_20', m_bar_120, CORE),
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
    ('M14_half_pitch_lattice_stops_nesting', m_unnested_centres, CORE),
    ('M15_pearcey_label_claims_the_cusp_field', m_pearcey_label, CORE),
    ('M16_scope_always_claims_returned_field', m_scope_always_returned, CORE),
    ('M17_bar_inside_the_converged_spread', m_bar_inside_spread, CORE),
    ('M18_lattice_rebuilt_inline', m_lattice_rebuilt_inline, CORE),
]


#: the pristine text of the two files every mutation edits, read ONCE at
#: import.  Restoring by copy rather than by ``git checkout`` is not a
#: convenience: the mutation tree is a plain export on both builds, and a git
#: worktree's ``.git`` file carries a WINDOWS gitdir path that a WSL git
#: cannot resolve -- which is how the wsl half of this matrix failed on its
#: first attempt, with every mutation reporting an unrestored tree.
_PRISTINE = {}


def _snapshot():
    for path in (MB, UN):
        with open(path, encoding='cp1252') as f:
            _PRISTINE[path] = f.read()


def restore():
    if not _PRISTINE:
        _snapshot()
        return
    for path, text in _PRISTINE.items():
        with open(path, encoding='cp1252') as f:
            if f.read() == text:
                continue
        with open(path, 'w', encoding='cp1252') as f:
            f.write(text)


def run_tests(files, timeout=5400):
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONPATH=MUT_TREE)
    p = subprocess.run(
        [PY, '-m', 'pytest', *files, '-q', '--capture=sys', '-p',
         'no:randomly', '--no-header', '--tb=line', '-rs'],
        cwd=MUT_TREE, env=env, capture_output=True, text=True, timeout=timeout)
    out = p.stdout + p.stderr
    tail = [ln for ln in out.splitlines()
            if re.search(r'\b(passed|failed|error|no tests ran)\b', ln)]
    fails = sorted({ln.split('::')[1].split()[0].split('[')[0]
                    for ln in out.splitlines()
                    if '::' in ln and ('FAILED' in ln or 'ERROR' in ln)
                    and len(ln.split('::')) > 1})
    skips = sorted({ln.split(':')[0].split('/')[-1] + ':' + ln.split(':')[1]
                    for ln in out.splitlines() if ln.startswith('SKIPPED')})
    return dict(returncode=p.returncode, tail=tail[-3:], failed_ids=fails,
                skipped=skips), out


def main():
    out_path = sys.argv[1]
    want = set(sys.argv[2:])
    suffix = os.environ.get('R3_TAG', 'win')
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
            res, raw = dict(returncode=-9, tail=['TIMEOUT'], failed_ids=[],
                            skipped=[]), ''
        res.update(id=mid, doc=(fn.__doc__ or '').strip().split('\n')[0],
                   files=files, seconds=round(time.time() - t0, 1))
        res['caught'] = bool(res['returncode'] != 0)
        rows.append(res)
        with open(os.path.join(os.path.dirname(os.path.abspath(out_path)),
                               f'mut_{mid}_{suffix}.txt'), 'w',
                  encoding='cp1252') as f:
            f.write(raw)
        print(json.dumps({k: res[k] for k in
                          ('id', 'caught', 'tail', 'failed_ids', 'skipped',
                           'seconds')}), flush=True)
        with open(out_path, 'w', encoding='cp1252') as f:
            json.dump(rows, f, indent=1)
    restore()
    print('DONE', out_path, flush=True)


if __name__ == '__main__':
    main()
