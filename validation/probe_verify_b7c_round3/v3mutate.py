"""VERIFY-WP-B7c round 3 -- the MUTATION MATRIX, re-run and extended.

Round 3 claims 18 of 18 caught on both builds.  This driver re-runs those
eighteen against the same gate and adds FIVE of its own, chosen to probe the
places round 3's own pins do NOT obviously reach:

``N1``  the scope enum silently accepts an unknown value -- the guard round 3
        added inside ``_arbitrate`` is deleted.  Round 3's structural pin
        checks the enum's CONTENTS, not that an unnamed scope raises, so this
        is the shape that would let a fourth route ship with a scope nobody
        defined;
``N2``  ``half_pitch_centres`` is off by ONE FINE PIXEL -- the seam E5 is
        about, in the direction a copy-paste actually drifts;
``N3``  the nesting breaks on ODD ``N`` only -- the fine lattice is offset by
        half a fine pixel when ``N`` is odd, which is invisible on every
        power-of-two grid the suite runs end to end;
``N4``  the derivation comment's POPULATION COUNT is edited (``1304`` becomes
        ``204``) -- the constant's derivation is a documented measurement, and
        ``docs/TESTING_STANDARDS.md`` makes an undocumented constant a defect,
        so the question is whether anything holds the documentation;
``N5``  the scope NOTE contradicts the constant it summarises (its
        ``flags 192 planes`` becomes ``flags 0 planes``) -- the same question
        for the string a CALLER reads.

Every mutation is applied to a SEPARATE tree named by ``MUT_TREE``;
``lumenairy/`` on the verification branch is never touched.  The tree is a
plain EXPORT, and the two edited files are restored BY COPY from a snapshot
taken at import -- not by ``git checkout``, because a git worktree's ``.git``
is a Windows gitdir pointer that a WSL git cannot resolve.

Usage:
    MUT_TREE=C:/tmp/lum_vmb3_mut python v3mutate.py <out.json> [id ...]
    MUT_TREE=... V3_PY=<python> V3_TAG=wsl python v3mutate.py ...
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

MUT_TREE = os.environ.get('MUT_TREE', r'C:\tmp\lum_vmb3_mut')
PY = os.environ.get('V3_PY', sys.executable)
MB = os.path.join(MUT_TREE, 'lumenairy', 'elements',
                  '_lens_traced_multibranch.py')
UN = os.path.join(MUT_TREE, 'lumenairy', 'elements', '_lens_traced_uniform.py')

#: round 3's own gate -- the five files its matrix ran against.  THIS
#: verification's file joins it only when it is present in the mutation tree,
#: so the matrix can be run twice: once against round 3's gate as shipped (to
#: reproduce "18 of 18"), and once with this verification's file added (to
#: show which of the new mutations it is what closes).
CORE = [f for f in
        ['tests/unit/test_audit2609_b7c2_pixel_halving_arbiter.py',
         'tests/unit/test_verify_b7c_multibranch.py',
         'tests/unit/test_audit2609_b7c_multibranch_envelope.py',
         'tests/unit/test_verify_b7c_round2.py',
         'tests/unit/test_wp_b7c_round3.py',
         'tests/unit/test_verify_b7c_round3.py']
        if os.path.exists(os.path.join(MUT_TREE, f))]
SEAM = ['tests/unit/test_niche_r2_pearcey_cusp.py',
        'tests/unit/test_niche_r5_gbd_vector_catastrophe.py']


def sub(path, old, new, count=1):
    with open(path, encoding='cp1252') as f:
        s = f.read()
    if s.count(old) < 1:
        raise SystemExit(f'anchor not found in {os.path.basename(path)}: '
                         f'{old[:80]!r}')
    with open(path, 'w', encoding='cp1252') as f:
        f.write(s.replace(old, new, count))


# --------------------------------------------------------------------------
# round 2's thirteen and round 3's five, re-applied
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


def m_unnested_centres():
    """M14 -- the half-pitch lattice stops NESTING (the hull-aligned
    alternative E5 asked for)."""
    sub(MB, '_HALF_PITCH_CENTRE_OFFSET = 0.0',
        '_HALF_PITCH_CENTRE_OFFSET = -0.5')


def m_pearcey_label():
    """M15 -- the Pearcey cusp route labels the branch sum's reading as the
    cusp field's own."""
    sub(UN, "_arbitrate(_mb_cont, _PEARCEY_READING_OF,",
        "_arbitrate(_mb_cont, 'the Pearcey cusp field',")
    sub(UN, "                               'underlying_branch_sum')",
        "                               'returned_field')")


def m_scope_always_returned():
    """M16 -- every route claims the reading arbitrates the field it
    returns."""
    sub(UN, "        d = _continuity_verdict(c)\n"
            "        _mb_energy['pixel_continuity'] =",
        "        scope = 'returned_field'\n"
        "        d = _continuity_verdict(c)\n"
        "        _mb_energy['pixel_continuity'] =")


def m_bar_inside_spread():
    """M17 -- the bar is put inside the converged reading's own spread."""
    sub(MB, '_PIXEL_CONTINUITY_MAX = ', '_PIXEL_CONTINUITY_MAX = 1.004  # ')


def m_lattice_rebuilt_inline():
    """M18 -- the completion rebuilds the half-pitch lattice inline."""
    sub(UN, '        _xh = half_pitch_centres(N, dx)',
        '        _xh = (np.arange(_N_h) - _N_h / 2.0) * _dx_h')


# --------------------------------------------------------------------------
# this verification's own five
# --------------------------------------------------------------------------
def n_scope_guard_removed():
    """N1 -- the unknown-scope guard is deleted, so the enum accepts
    anything."""
    sub(UN, """        if scope not in _PIXEL_CONTINUITY_SCOPES:
            raise AssertionError(
                f'apply_real_lens_traced_uniform: unknown continuity scope '
                f'{scope!r}; expected one of '
                f'{sorted(_PIXEL_CONTINUITY_SCOPES)}')
""", '')


def n_half_pitch_off_by_one():
    """N2 -- ``half_pitch_centres`` is off by ONE FINE PIXEL."""
    sub(MB, "    return ((np.arange(2 * int(N)) - "
            "_render_centre_origin(2 * int(N), int(N)))",
        "    return ((np.arange(2 * int(N)) + 1 - "
        "_render_centre_origin(2 * int(N), int(N)))")


def n_nesting_breaks_on_odd_n():
    """N3 -- the nesting breaks on ODD ``N`` only."""
    sub(MB, "    return ((np.arange(2 * int(N)) - "
            "_render_centre_origin(2 * int(N), int(N)))",
        "    return ((np.arange(2 * int(N)) - 0.5 * (int(N) % 2) - "
        "_render_centre_origin(2 * int(N), int(N)))")


def n_population_count_edited():
    """N4 -- the derivation comment's POPULATION COUNT is edited."""
    sub(MB, '# RE-MEASURED (WP-B7c round 3, 2026-09-19) on **1304 '
            'oracle-scored planes**',
        '# RE-MEASURED (WP-B7c round 3, 2026-09-19) on **204 '
        'oracle-scored planes**')


def n_scope_note_contradicts():
    """N5 -- the scope NOTE contradicts the constant it summarises."""
    sub(UN, "'1/1.06 flags 192 planes and every one scores under oracle "
            "fidelity '", "'1/1.06 flags 0 planes and every one scores under "
            "oracle fidelity '")


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
    ('N1_scope_guard_removed', n_scope_guard_removed, CORE),
    ('N2_half_pitch_off_by_one_pixel', n_half_pitch_off_by_one, CORE),
    ('N3_nesting_breaks_on_odd_N', n_nesting_breaks_on_odd_n, CORE),
    ('N4_derivation_population_count_edited', n_population_count_edited, CORE),
    ('N5_scope_note_contradicts_the_constant', n_scope_note_contradicts, CORE),
]

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
    skips = sorted({ln for ln in out.splitlines() if ln.startswith('SKIPPED')})
    return dict(returncode=p.returncode, tail=tail[-3:], failed_ids=fails,
                skipped=skips), out


def main():
    out_path = sys.argv[1]
    want = set(sys.argv[2:])
    suffix = os.environ.get('V3_TAG', 'win')
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
            rows.append(dict(id=mid, error=str(e), caught=None))
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
