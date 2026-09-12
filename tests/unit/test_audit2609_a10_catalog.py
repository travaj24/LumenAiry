"""WP-A10 / AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11 finding I6 (data half) --
the Thorlabs catalogue must reproduce its own part numbers.

``THORLABS_CATALOG['LA1509-C']`` carried ``R1 = 103.29 mm`` -- the radius of a
200 mm lens -- under a part number that IS a 100 mm lens (Thorlabs LA1509:
R = 51.5 mm, tc = 3.6 mm, N-BK7, f = 100.0 mm).  A caller got a 2x
focal-length error with no diagnostic, and the validation fixture
``validation/real_lens_opd/zemax_prescriptions/LA1509_C.zmx`` carried the same
wrong radius (``CURV 0.0096814793`` = 1/103.29 mm), so it could not catch it.

The header comment on the table claims "Surface data from Thorlabs Zemax
files", which makes the claim falsifiable -- this file falsifies or confirms
it, row by row, on every run.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest

import lumenairy as la
from lumenairy.io.prescriptions_builders import (
    _THORLABS_EFL_RTOL,
    _THORLABS_NOMINAL_EFL,
    THORLABS_CATALOG,
)


def _paraxial_efl_oracle(entry, wavelength=587.6e-9):
    """Independent thick-lens paraxial EFL [m] -- a surface-by-surface
    reduced-slope trace written here, not imported from the library.

    This is deliberately NOT ``system_abcd`` (the library's own answer) and
    NOT ``_paraxial_efl_from_entry`` (the code under test): a self-referential
    oracle is exactly the failure shape §15.2 of the audit calls out.
    """
    from lumenairy.glass import get_glass_index
    if entry['type'] == 'singlet':
        radii = [entry['R1'], entry['R2']]
        gaps = [entry['d']]
        media = ['air', entry['glass'], 'air']
    else:
        radii = [entry['R1'], entry['R2'], entry['R3']]
        gaps = [entry['d1'], entry['d2']]
        media = ['air', entry['glass1'], entry['glass2'], 'air']
    n = [1.0 if m == 'air' else float(get_glass_index(m, wavelength))
         for m in media]
    y, u = 1.0, 0.0
    for i, R in enumerate(radii):
        power = 0.0 if not math.isfinite(R) else (n[i + 1] - n[i]) / R
        u -= y * power
        if i < len(gaps):
            y += gaps[i] * u / n[i + 1]
    return float('inf') if u == 0.0 else -1.0 / u


def test_i6_la1509c_is_a_100mm_lens():
    """The row itself, against the vendor specification.

    Ground truth (audit brief): LA1509 = R 51.5 mm, tc 3.6 mm, N-BK7,
    f = 100.0 mm.  Pre-fix the row carried R1 = 103.29 mm and measured
    EFL = 199.68 mm at 587.6 nm (205.11 mm at 1310 nm) -- a factor 2.004.
    """
    e = THORLABS_CATALOG['LA1509-C']
    assert e['R1'] == pytest.approx(51.5e-3, rel=1e-12)
    assert math.isinf(e['R2'])
    assert e['d'] == pytest.approx(3.6e-3, rel=1e-12)
    assert e['glass'] == 'N-BK7'


@pytest.mark.parametrize('part', sorted(_THORLABS_NOMINAL_EFL))
def test_i6_catalog_row_efl_ledger(part):
    """Every row's measured EFL, against the focal length its part number
    states, with the three known-bad doublets pinned as known-bad.

    Bar derivation: the nominal figure is the vendor's thin-lens d-line spec
    and the measurement is the exact thick-lens paraxial EFL at 587.6 nm, so
    a few tenths of a percent of legitimate spread is expected.  Measured
    deviations on 2026-09-12: LA1050-C -0.348 %, LA1509-C -0.348 %,
    LA1301-C +0.0004 %; AC254-050-C -10.88 %, AC254-100-C -16.83 %,
    AC254-200-C -31.30 %.  The 3 % gate therefore sits ~8.6x above the
    largest legitimate deviation and ~3.6x below the smallest real defect.

    The three ``AC254-*-C`` rows are xfail-free deliberate assertions of the
    CURRENT state: no vendor surface data was available to correct them in
    this pass, so they are pinned as "known wrong, warned about" -- if a
    future change fixes (or worsens) them, this test says so.
    """
    entry = THORLABS_CATALOG[part]
    nominal = _THORLABS_NOMINAL_EFL[part]
    efl = _paraxial_efl_oracle(entry)
    rel = abs(efl / nominal - 1.0)
    known_bad = {'AC254-050-C': 0.1088,
                 'AC254-100-C': 0.1683,
                 'AC254-200-C': 0.3130}
    if part in known_bad:
        assert rel == pytest.approx(known_bad[part], abs=5e-4), (
            f'{part}: EFL {efl * 1e3:.3f} mm vs nominal {nominal * 1e3:.1f} '
            f'mm -- the catalogue data still does not match the part number, '
            f'and the deviation itself has MOVED from the 2026-09-12 '
            f'measurement.  Update this ledger with the new measurement and '
            f'say why in the changelog.')
        # ... and the user is told, on every call.
        with pytest.warns(UserWarning, match='part number specifies'):
            la.io.prescriptions_builders._THORLABS_EFL_WARNED.discard(part)
            la.thorlabs_lens(part)
    else:
        assert rel < _THORLABS_EFL_RTOL, (
            f'{part}: EFL {efl * 1e3:.4f} mm vs nominal '
            f'{nominal * 1e3:.1f} mm ({rel * 100:.2f} % off)')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            la.io.prescriptions_builders._THORLABS_EFL_WARNED.discard(part)
            la.thorlabs_lens(part)
        assert not [x for x in w if 'part number specifies' in str(x.message)]


def test_i6_independent_oracle_agrees_with_system_abcd():
    """Cross-check the oracle against the library's own paraxial trace.

    If these two disagree, one of them is wrong and the ledger above means
    nothing -- so the agreement is asserted rather than assumed.  Measured
    max relative difference over the six rows on 2026-09-12: < 1e-12
    (both are exact closed forms of the same paraxial algebra in float64,
    so the floor is accumulation of ~10 roundings, ~1e-15).
    """
    from lumenairy.raytrace import surfaces_from_prescription, system_abcd
    worst = 0.0
    for part, entry in THORLABS_CATALOG.items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rx = la.thorlabs_lens(part)
        _, efl_lib, _, _ = system_abcd(
            surfaces_from_prescription(rx), 587.6e-9)
        efl_oracle = _paraxial_efl_oracle(entry)
        worst = max(worst, abs(efl_lib / efl_oracle - 1.0))
    assert worst < 1e-12, f'oracle vs system_abcd disagree by {worst:.3e}'


def test_i6_la1509c_validation_fixture_matches_the_catalogue():
    """The ``.zmx`` fixture carried the same wrong radius, so it could not
    catch the catalogue defect.  Pin the two together.
    """
    p = Path('validation/real_lens_opd/zemax_prescriptions/LA1509_C.zmx')
    if not p.exists():                                # pragma: no cover
        pytest.fail(f'fixture missing: {p}')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rx = la.load_zemax_zmx(str(p))
    R1 = rx['surfaces'][0]['radius']
    assert R1 == pytest.approx(THORLABS_CATALOG['LA1509-C']['R1'], rel=1e-6), (
        f'fixture R1 = {R1 * 1e3:.4f} mm; catalogue says '
        f'{THORLABS_CATALOG["LA1509-C"]["R1"] * 1e3:.4f} mm.  Regenerate '
        f'with validation/real_lens_opd/export_all_zemax.py.')
    assert np.isinf(rx['surfaces'][1]['radius'])
    # And the human-readable twin.
    txt = Path('validation/real_lens_opd/zemax_prescriptions/'
               'LA1509_C.txt').read_text(encoding='utf-8')
    assert 'f=100 mm' in txt      # pre-fix: 'f=200 mm'
    assert '51.500000' in txt     # pre-fix: '103.290000'
