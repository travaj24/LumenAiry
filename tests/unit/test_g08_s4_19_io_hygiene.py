"""G08 / S4-19 (AUDIT_V5_24_2) -- IO hygiene.

* ``.txt`` prescription loader warns on a genuinely malformed numeric
  (pre-fix it silently substituted a default, shifting downstream
  surfaces with no diagnostic).
* Quadoa export unit-scales aspheric coefficients so a non-meter file is
  physically self-consistent (radius + asphere in the same units), and
  the loader inverse-scales them for an exact round-trip.
* ``replay_run`` reports the STORED per-plane wavelength when the caller
  omits it (pre-fix it always reported 0.0).
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from lumenairy.io.prescriptions_quadoa import (
    _quadoa_deserialize_aspheric,
    _quadoa_serialize_aspheric,
    export_quadoa_qos,
    load_quadoa_qos,
)
from lumenairy.io.prescriptions_zemax import load_zemax_prescription_data_txt

# =========================================================================
# .txt loader: warn on malformed numerics
# =========================================================================

def _txt_rows(radius_stop='50'):
    """A minimal Zemax prescription-data text report (tab-separated)."""
    diam = '20'
    return '\n'.join([
        'System/Prescription Data',
        'Lens Units              :   Millimeters',
        '',
        'SURFACE DATA SUMMARY:',
        '',
        'Surf\tType\tRadius\tThickness\tGlass\tClear Diam\tChip Zone'
        '\tMech Diam\tConic\tComment',
        f'OBJ\tSTANDARD\tInfinity\t100\t\t{diam}\t0\t{diam}\t0\t',
        f'1\tSTANDARD\t{radius_stop}\t5\tN-BK7\t{diam}\t0\t{diam}\t-1\tfront',
        f'2\tSTANDARD\t-50\t95\t\t{diam}\t0\t{diam}\t0\t',
        'IMA\tSTANDARD\tInfinity\t0\t\t2\t0\t2\t0\t',
    ]) + '\n'


def _write(path, text):
    path.write_text(text, encoding='utf-8')
    return str(path)


def test_txt_malformed_numeric_warns(tmp_path):
    """A non-empty, non-placeholder token that fails float parse must
    raise a RuntimeWarning naming the parse failure -- pre-fix it was
    silently replaced by the default and every downstream surface
    shifted with no diagnostic."""
    p = _write(tmp_path / 'bad.txt', _txt_rows(radius_stop='NOT_A_NUMBER'))
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        load_zemax_prescription_data_txt(p)
    hits = [w for w in ws
            if issubclass(w.category, RuntimeWarning)
            and 'could not parse' in str(w.message)]
    assert len(hits) >= 1, (
        'S4-19: malformed .txt numeric did not warn. '
        f'Got: {[str(w.message) for w in ws]}')


def test_txt_clean_numeric_does_not_warn(tmp_path):
    """No false positive: a clean report emits no parse warning (the '-'
    / empty / Infinity placeholders stay silent)."""
    p = _write(tmp_path / 'clean.txt', _txt_rows(radius_stop='50'))
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        load_zemax_prescription_data_txt(p)
    assert not [w for w in ws if 'could not parse' in str(w.message)]


# =========================================================================
# Quadoa aspheric unit scaling
# =========================================================================

def test_quadoa_serialize_meters_is_byte_identical():
    """units='M' path (scale=1.0) must leave coefficients untouched
    (byte-for-byte legacy behaviour)."""
    coeffs = {4: -3.7e5, 6: 1.1e9}
    out = _quadoa_serialize_aspheric(coeffs)  # default scale=1.0
    assert out == {'4': -3.7e5, '6': 1.1e9}


def test_quadoa_aspheric_mm_export_is_physically_consistent(tmp_path):
    """Export in MM: the written coefficient of power p must scale by
    scale**(1-p) (scale=1e3), so the file's asphere sag (in mm) equals
    the library sag (in m) times 1e3 -- verified against an independent
    sag oracle, not the code's own scaling formula."""
    a4 = 3.7e4          # 1/m^3 (an r^4 asphere coefficient, in meters)
    rx = {
        'name': 'asph_mm',
        'surfaces': [
            {'radius': 60e-3, 'conic': 0.0, 'aspheric_coeffs': {4: a4},
             'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'radius': float('inf'), 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'N-BK7', 'glass_after': 'air'},
        ],
        'thicknesses': [4e-3, 0.0],
        'aperture_diameter': 12e-3,
    }
    path = str(tmp_path / 'asph.qos')
    export_quadoa_qos(rx, path, wavelength=1.31e-6, units='MM')

    with open(path, 'r', encoding='utf-8') as fh:
        doc = json.load(fh)
    file_a4 = doc['surfaces'][0]['aspheric_coeffs']['4']

    # Independent physical-sag oracle at r = 2 mm.
    r_m = 2e-3
    sag_m = a4 * r_m ** 4                    # library units (meters)
    r_mm = r_m * 1e3
    sag_mm = file_a4 * r_mm ** 4             # file units (millimeters)
    assert sag_mm == pytest.approx(sag_m * 1e3, rel=1e-12), (
        'S4-19: MM-exported asphere sag is not consistent with the '
        'library (meter) sag -- coefficient was left unscaled.')

    # And the round-trip through the loader recovers the coefficient.
    rx_back = load_quadoa_qos(path)
    a4_back = rx_back['surfaces'][0]['aspheric_coeffs'][4]
    assert a4_back == pytest.approx(a4, rel=1e-12)


def test_quadoa_aspheric_roundtrip_lossless_all_units():
    """serialize(scale) then deserialize(1/scale) is the identity for
    every power, for MM and IN scales alike."""
    coeffs = {4: -5.0e4, 6: 2.3e9, 8: -7.1e13}
    for scale, inv_scale in ((1e3, 1e-3), (1.0 / 0.0254, 0.0254)):
        ser = _quadoa_serialize_aspheric(coeffs, scale)
        deser = _quadoa_deserialize_aspheric(ser, inv_scale)
        for p in coeffs:
            assert deser[p] == pytest.approx(coeffs[p], rel=1e-12)


# =========================================================================
# replay_run: honour the stored per-plane wavelength
# =========================================================================

def test_replay_run_uses_stored_wavelength(tmp_path):
    h5py = pytest.importorskip('h5py')  # noqa: F841
    from lumenairy.io.storage import append_plane, replay_run

    path = str(tmp_path / 'run.h5')
    field = np.ones((8, 8), dtype=np.complex128)
    stored_wl = 1.31e-6
    append_plane(path, field, dx=2e-6, label='p0',
                 metadata={'wavelength': stored_wl})
    append_plane(path, field * 2, dx=2e-6, label='p1',
                 metadata={'wavelength': stored_wl})

    # Caller omits wavelength -> stored value must be reported.
    res = replay_run(path)
    assert res.wavelength == pytest.approx(stored_wl), (
        'S4-19: replay_run ignored the stored per-plane wavelength '
        f'(got {res.wavelength}).')

    # Explicit override still wins.
    res2 = replay_run(path, wavelength=0.633e-6)
    assert res2.wavelength == pytest.approx(0.633e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
