"""Wave-6 audit fixes for lumenairy/io/prescriptions_zemax.py.

Pins the v5.17.1 fixes for AUDIT_V5_17_0_2026_07_01_DEEP findings:

* P3-41 -- malformed/truncated .zmx lines raise a clear ValueError with
  file / line-number / line-text context instead of a raw IndexError.
* P3-42 -- auto-detect no longer appends the surface AFTER a terminal
  mirror (image plane / dummy) as a bogus refractive element; the +1
  extension applies only to a refractive last surface (exit surface).
* P3-43 -- load_zemax_prescription_data_txt warns per surface when a
  non-STANDARD Type (EVENASPH, QBFS, TOROIDAL, ...) is imported, since
  the summary table carries no aspheric/freeform coefficients.

(P3-44 is a comment-only fix in export_zemax_zmx; no test needed.)
"""

import warnings

import pytest

from lumenairy.io.prescriptions_zemax import (
    load_zemax_prescription_data_txt,
    load_zemax_zmx,
)

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _write(path, text):
    path.write_text(text, encoding='utf-8')
    return str(path)


_SINGLE_MIRROR_ZMX = """UNIT MM
SURF 0
  TYPE STANDARD
  DISZ 100.0
  DIAM 5.0 0 0 0 1 ""
SURF 1
  TYPE STANDARD
  CURV 0.0
  DISZ -100.0
  GLAS MIRROR 0 0 1.5 50.0
  DIAM 5.0 0 0 0 1 ""
SURF 2
  TYPE STANDARD
  DISZ 0.0
  DIAM 55.0 0 0 0 1 ""
"""

_SINGLET_ZMX = """UNIT MM
SURF 0
  TYPE STANDARD
  DISZ 10.0
SURF 1
  TYPE STANDARD
  CURV 0.02
  DISZ 5.0
  GLAS N-BK7 0 0 1.5 50.0
  DIAM 12.0 0 0 0 1 ""
SURF 2
  TYPE STANDARD
  CURV -0.02
  DISZ 95.0
  DIAM 12.0 0 0 0 1 ""
SURF 3
  TYPE STANDARD
  DISZ 0.0
  DIAM 1.0 0 0 0 1 ""
"""


def _prescription_txt(surface_type='STANDARD'):
    rows = [
        'SURFACE DATA SUMMARY:',
        '',
        'Surf\tType\tRadius\tThickness\tGlass\tClear Diam\tChip Zone'
        '\tMech Diam\tConic\tComment',
        'OBJ\tSTANDARD\tInfinity\t100\t\t10\t0\t10\t0\t',
        f'1\t{surface_type}\t50\t5\tN-BK7\t20\t0\t20\t-1\tfront',
        '2\tSTANDARD\t-50\t95\t\t20\t0\t20\t0\t',
        'IMA\tSTANDARD\tInfinity\t0\t\t2\t0\t2\t0\t',
    ]
    return '\n'.join(rows) + '\n'


# ---------------------------------------------------------------------------
# P3-41: malformed lines -> clear ValueError with context
# ---------------------------------------------------------------------------

class TestP341MalformedLines:

    def test_truncated_curv_raises_valueerror_with_context(self, tmp_path):
        p = _write(tmp_path / 'trunc.zmx',
                   'UNIT MM\nSURF 0\n  TYPE STANDARD\n  CURV\n')
        with pytest.raises(ValueError) as exc_info:
            load_zemax_zmx(p)
        msg = str(exc_info.value)
        assert 'trunc.zmx' in msg
        assert 'line 4' in msg
        assert 'CURV' in msg

    def test_nonnumeric_disz_raises_valueerror_with_context(self, tmp_path):
        p = _write(tmp_path / 'bad_disz.zmx',
                   'UNIT MM\nSURF 0\n  DISZ oops\n')
        with pytest.raises(ValueError) as exc_info:
            load_zemax_zmx(p)
        msg = str(exc_info.value)
        assert 'bad_disz.zmx' in msg and 'line 3' in msg

    def test_truncated_parm_raises_valueerror_not_indexerror(self, tmp_path):
        p = _write(tmp_path / 'bad_parm.zmx',
                   'UNIT MM\nSURF 1\n  TYPE EVENASPH\n  PARM 2\n')
        with pytest.raises(ValueError):
            load_zemax_zmx(p)

    def test_well_formed_file_still_loads(self, tmp_path):
        p = _write(tmp_path / 'singlet.zmx', _SINGLET_ZMX)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rx = load_zemax_zmx(p)
        assert [e['surf_num'] for e in rx['elements']] == [1, 2]


# ---------------------------------------------------------------------------
# P3-42: terminal-mirror auto-detect boundary
# ---------------------------------------------------------------------------

class TestP342TerminalMirror:

    def test_zmx_terminal_mirror_excludes_image_plane(self, tmp_path):
        p = _write(tmp_path / 'mirror.zmx', _SINGLE_MIRROR_ZMX)
        rx = load_zemax_zmx(p)
        kinds = [(e['element_type'], e['surf_num']) for e in rx['elements']]
        assert kinds == [('mirror', 1)]
        # Aperture comes from the mirror's DIAM 5 (semi-dia) -> 10 mm,
        # not from the image plane's DIAM 55 -> 110 mm.
        assert rx['aperture_diameter'] == pytest.approx(0.01)
        assert rx['surfaces'] == []
        assert rx['all_thicknesses'] == []

    def test_zmx_refractive_last_surface_keeps_exit_surface(self, tmp_path):
        p = _write(tmp_path / 'singlet.zmx', _SINGLET_ZMX)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rx = load_zemax_zmx(p)
        # Glass on SURF 1 -> exit surface SURF 2 must still be captured
        # by the +1 extension; SURF 3 (image) must not.
        assert [e['surf_num'] for e in rx['elements']] == [1, 2]
        assert len(rx['surfaces']) == 2

    def test_txt_terminal_mirror_excludes_image_plane(self, tmp_path):
        rows = [
            'SURFACE DATA SUMMARY:',
            '',
            'Surf\tType\tRadius\tThickness\tGlass\tClear Diam\tChip Zone'
            '\tMech Diam\tConic\tComment',
            'OBJ\tSTANDARD\tInfinity\t100\t\t10\t0\t10\t0\t',
            '1\tSTANDARD\tInfinity\t-100\tMIRROR\t10\t0\t10\t0\tfold',
            'IMA\tSTANDARD\tInfinity\t0\t\t110\t0\t110\t0\t',
        ]
        p = _write(tmp_path / 'mirror.txt', '\n'.join(rows) + '\n')
        rx = load_zemax_prescription_data_txt(p)
        kinds = [(e['element_type'], e['surf_num']) for e in rx['elements']]
        assert kinds == [('mirror', 1)]
        assert rx['aperture_diameter'] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# P3-43: txt loader warns when aspheric/freeform shape data is dropped
# ---------------------------------------------------------------------------

class TestP343TxtAsphericWarning:

    def test_evenasph_row_warns_and_names_surface(self, tmp_path):
        p = _write(tmp_path / 'asph.txt', _prescription_txt('EVENASPH'))
        with pytest.warns(UserWarning, match='EVENASPH') as record:
            rx = load_zemax_prescription_data_txt(p)
        msgs = [str(w.message) for w in record]
        assert any('Surface 1' in m and 'BASE CONIC' in m for m in msgs)
        # Structure is unchanged: coefficients are still None (the
        # summary table has none to parse).
        assert all(e['aspheric_coeffs'] is None for e in rx['elements'])

    def test_all_standard_rows_do_not_warn(self, tmp_path):
        p = _write(tmp_path / 'std.txt', _prescription_txt('STANDARD'))
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            load_zemax_prescription_data_txt(p)
        drop_warnings = [w for w in ws if 'BASE CONIC' in str(w.message)]
        assert drop_warnings == []
