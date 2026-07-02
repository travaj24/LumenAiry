"""Wave-5 audit fixes -- Zemax .zmx loader/exporter trust boundary.

Covers AUDIT_V5_17_0_2026_07_01_DEEP findings:

* P2-19 -- unknown Zemax SURFTYPEs (TOROIDAL, ODDASPHE, BICONICX,
  DGRATING, ...) were silently parsed AS EVENASPH, turning their PARM
  values into huge bogus aspheric coefficients (TOROIDAL ``PARM 1
  100.0`` -> a_2 = 1e5 1/m -> 0.625 m of fake sag at r = 2.5 mm).
  Fixed: explicit known-type dispatch; unknown types import as the
  plain base conic with a loud per-surface UserWarning and their PARM
  table DROPPED.

* P2-20 -- the .zmx exporters silently dropped ``aspheric_coeffs`` on
  MIRROR elements (load->export->load was not identity for aspheric
  mirrors) and silently dropped all Forbes Q-type freeform keys.
  Fixed: the mirror branch of the full writer now emits TYPE EVENASPH
  + PARM lines with the same v5.16.1 ``parm_idx = power // 2`` mapping
  as refractives; Q-type freeforms (which neither writer can emit)
  now WARN loudly instead of vanishing.
"""

import os
import tempfile
import warnings

import numpy as np
import pytest

from lumenairy.io.prescriptions_zemax import (
    export_zemax_zmx,
    load_zemax_zmx,
)

# ============================================================================
# Helpers -- minimal in-memory .zmx files (same style as test_audit_io.py)
# ============================================================================

_HEADER = [
    'VERS 210000 0 123 0 0',
    'MODE SEQ',
    'NAME w5_zemax_test',
    'UNIT MM X W X CM MR CPMM',
    'ENPD 10.0',
    'WAVM 1 1.310000 1.0',
    'PWAV 1',
]


def _write_zmx(lines):
    fd, path = tempfile.mkstemp(suffix='.zmx', text=True)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return path


def _load(lines):
    path = _write_zmx(lines)
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter('always')
            rx = load_zemax_zmx(path)
    finally:
        os.unlink(path)
    return rx, wlist


def _unknown_type_zmx(stype, parm_lines=('  PARM 1 100.0',)):
    """3-optical-surface file whose middle surface has SURFTYPE ``stype``."""
    return _HEADER + [
        'SURF 0',
        '  TYPE STANDARD', '  CURV 0.0', '  DISZ INFINITY', '  DIAM 5.0',
        'SURF 1',
        f'  TYPE {stype}', '  STOP', '  CURV 0.01 0 0 0 0 ""',
        '  DISZ 3.0', '  GLAS SILICA 0 0 1.5 50.0',
        *parm_lines,
        '  DIAM 5.0',
        'SURF 2',
        '  TYPE STANDARD', '  CURV -0.005 0 0 0 0 ""', '  DISZ 50.0',
        '  DIAM 5.0',
        'SURF 3',
        '  TYPE STANDARD', '  CURV 0.0', '  DISZ 0.0', '  DIAM 5.0',
        'BLNK',
    ]


def _aspheric_mirror_zmx():
    """OBJ + EVENASPH parabolic mirror + image surface."""
    return _HEADER + [
        'SURF 0',
        '  TYPE STANDARD', '  CURV 0.0', '  DISZ INFINITY', '  DIAM 5.0',
        'SURF 1',
        '  TYPE EVENASPH', '  STOP', '  CURV 0.002 0 0 0 0 ""',
        '  CONI -1.000000', '  DISZ -50.0',
        '  GLAS MIRROR 0 0 1.5 50.0',
        '  PARM 2 1.5e-05',      # alpha_2 (r^4 term) [1/mm^3]
        '  PARM 3 2.0e-09',      # alpha_3 (r^6 term) [1/mm^5]
        '  DIAM 5.0',
        'SURF 2',
        '  TYPE STANDARD', '  CURV 0.0', '  DISZ 0.0', '  DIAM 5.0',
        'BLNK',
    ]


def _export_roundtrip(rx):
    """export_zemax_zmx -> (file text, reloaded prescription, warnings)."""
    fd, out = tempfile.mkstemp(suffix='.zmx', text=True)
    os.close(fd)
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter('always')
            export_zemax_zmx(rx, out, wavelength=1.31e-6)
        with open(out, encoding='utf-8') as f:
            text = f.read()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rx2 = load_zemax_zmx(out)
    finally:
        os.unlink(out)
    return text, rx2, wlist


def _msgs(wlist):
    return [str(w.message) for w in wlist]


# ============================================================================
# P2-19 -- unknown SURFTYPEs must not be parsed as EVENASPH
# ============================================================================


class TestP219UnknownSurftype:

    @pytest.mark.parametrize('stype', [
        'TOROIDAL', 'ODDASPHE', 'BICONICX', 'DGRATING', 'PARAXIAL',
    ])
    def test_unknown_type_parms_are_not_aspheric_coeffs(self, stype):
        """PARM 1 = 100.0 on an unknown type must NOT become a_2 = 1e5 1/m
        (pre-fix: 0.625 m of fake sag at r = 2.5 mm)."""
        rx, _ = _load(_unknown_type_zmx(stype))
        assert rx['surfaces'][0]['aspheric_coeffs'] is None, (
            f"SURFTYPE {stype} PARM table was interpreted as EVENASPH "
            f"coefficients: {rx['surfaces'][0]['aspheric_coeffs']}")
        # And no Q-type keys either.
        assert rx['surfaces'][0].get('freeform_type') is None

    def test_unknown_type_warns_loudly_per_surface(self):
        rx, wlist = _load(_unknown_type_zmx('TOROIDAL'))
        hits = [m for m in _msgs(wlist)
                if 'TOROIDAL' in m and 'SURFTYPE' in m]
        assert len(hits) == 1, (
            f"expected exactly one unknown-SURFTYPE warning, got "
            f"{_msgs(wlist)}")
        # The warning must identify the surface and say the PARMs dropped.
        assert 'surface 1' in hits[0]
        assert 'DROPPED' in hits[0]

    def test_unknown_type_without_parms_still_warns(self):
        """A PARM-free unknown type (shape still not representable) warns
        too, but without the dropped-PARM clause."""
        rx, wlist = _load(_unknown_type_zmx('IRREGULA', parm_lines=()))
        hits = [m for m in _msgs(wlist) if 'IRREGULA' in m]
        assert len(hits) == 1
        assert 'DROPPED' not in hits[0]
        assert rx['surfaces'][0]['aspheric_coeffs'] is None

    def test_unknown_type_base_conic_is_preserved(self):
        """The fix imports unknown types as plain conic: CURV/CONI honored."""
        lines = _unknown_type_zmx('TOROIDAL')
        lines.insert(lines.index('  DISZ 3.0'), '  CONI -0.5')
        rx, _ = _load(lines)
        s = rx['surfaces'][0]
        assert np.isclose(s['radius'], (1.0 / 0.01) * 1e-3)
        assert np.isclose(s['conic'], -0.5)

    def test_known_types_do_not_warn(self):
        """Discriminator: STANDARD/EVENASPH (and the EVENASPH coefficient
        path) are untouched -- no unknown-SURFTYPE warning, coefficients
        still extracted with the v5.16.1 power = 2*parm_num mapping."""
        rx, wlist = _load(_unknown_type_zmx(
            'EVENASPH', parm_lines=('  PARM 2 1.5e-05',)))
        assert not [m for m in _msgs(wlist) if 'SURFTYPE' in m]
        ac = rx['surfaces'][0]['aspheric_coeffs']
        assert ac is not None and 4 in ac
        assert np.isclose(ac[4], 1.5e-05 * 1e9, rtol=1e-9)

    def test_coordbrk_does_not_warn(self):
        """COORDBRK is a supported (skipped) type -- must not trip the
        unknown-SURFTYPE warning."""
        lines = _HEADER + [
            'SURF 0',
            '  TYPE STANDARD', '  CURV 0.0', '  DISZ INFINITY', '  DIAM 5.0',
            'SURF 1',
            '  TYPE COORDBRK', '  PARM 1 1.0', '  PARM 3 5.0', '  DISZ 10.0',
            '  DIAM 0.0',
            'SURF 2',
            '  TYPE STANDARD', '  STOP', '  CURV 0.01 0 0 0 0 ""',
            '  DISZ 3.0', '  GLAS SILICA 0 0 1.5 50.0', '  DIAM 5.0',
            'SURF 3',
            '  TYPE STANDARD', '  CURV -0.005 0 0 0 0 ""', '  DISZ 50.0',
            '  DIAM 5.0',
            'BLNK',
        ]
        _, wlist = _load(lines)
        assert not [m for m in _msgs(wlist) if 'SURFTYPE' in m]


# ============================================================================
# P2-20 -- aspheric mirrors must round-trip; Q-types must warn, not vanish
# ============================================================================


class TestP220AsphericMirrorExport:

    def test_aspheric_mirror_load_export_load_identity(self):
        """The headline round-trip: aspheric mirror coefficients survive
        load -> export -> load (pre-fix: reloaded aspheric_coeffs is None)."""
        rx, _ = _load(_aspheric_mirror_zmx())
        mirr = [e for e in rx['elements']
                if e['element_type'] == 'mirror'][0]
        # Loaded per v5.16.1 mapping: PARM 2 -> power 4, PARM 3 -> power 6.
        assert set(mirr['aspheric_coeffs'].keys()) == {4, 6}
        assert np.isclose(mirr['aspheric_coeffs'][4], 1.5e-05 * 1e9,
                          rtol=1e-9)

        text, rx2, wlist = _export_roundtrip(rx)
        assert 'TYPE EVENASPH' in text, (
            "exported aspheric mirror is not TYPE EVENASPH")
        mirr2 = [e for e in rx2['elements']
                 if e['element_type'] == 'mirror'][0]
        assert mirr2['aspheric_coeffs'] is not None, (
            "aspheric_coeffs dropped on mirror export (P2-20 regression)")
        assert set(mirr2['aspheric_coeffs']) == set(mirr['aspheric_coeffs'])
        for power, val in mirr['aspheric_coeffs'].items():
            assert np.isclose(mirr2['aspheric_coeffs'][power], val,
                              rtol=1e-9), (
                f"power {power}: {mirr2['aspheric_coeffs'][power]} "
                f"!= {val}")
        # Base geometry also round-trips.
        assert np.isclose(mirr2['radius'], mirr['radius'], rtol=1e-9)
        assert np.isclose(mirr2['conic'], mirr['conic'], rtol=1e-9)
        # No Q-type-drop warning for a plain even-aspheric mirror.
        assert not [m for m in _msgs(wlist) if 'freeform' in m.lower()]

    def test_mirror_parm_mapping_is_power_over_two(self):
        """The mirror branch must use the same v5.16.1 mapping as the
        refractive branch: parm_idx = power // 2 (power 4 -> PARM 2), with
        the 1/m^(power-1) -> 1/mm^(power-1) unit conversion."""
        rx, _ = _load(_aspheric_mirror_zmx())
        text, _, _ = _export_roundtrip(rx)
        parm_lines = [ln.strip() for ln in text.splitlines()
                      if ln.strip().startswith('PARM')]
        parms = {int(ln.split()[1]): float(ln.split()[2])
                 for ln in parm_lines}
        assert set(parms) == {2, 3}, f"PARM indices wrong: {parms}"
        assert np.isclose(parms[2], 1.5e-05, rtol=1e-9)   # 1/mm^3
        assert np.isclose(parms[3], 2.0e-09, rtol=1e-9)   # 1/mm^5

    def test_conic_only_mirror_stays_standard(self):
        """Non-regression: a mirror WITHOUT aspheric_coeffs still exports
        as TYPE STANDARD with no PARM lines."""
        lines = _aspheric_mirror_zmx()
        lines = [ln for ln in lines if not ln.startswith('  PARM')]
        rx, _ = _load(lines)
        text, rx2, _ = _export_roundtrip(rx)
        assert 'TYPE EVENASPH' not in text
        assert not [ln for ln in text.splitlines()
                    if ln.strip().startswith('PARM')]
        mirr2 = [e for e in rx2['elements']
                 if e['element_type'] == 'mirror'][0]
        assert mirr2['aspheric_coeffs'] is None


class TestP220QTypeExportWarns:

    def _qbfs_refractive_rx(self):
        lines = _HEADER + [
            'SURF 0',
            '  TYPE STANDARD', '  CURV 0.0', '  DISZ INFINITY', '  DIAM 5.0',
            'SURF 1',
            '  TYPE QBFS', '  STOP', '  CURV 0.01 0 0 0 0 ""',
            '  DISZ 3.0', '  GLAS SILICA 0 0 1.5 50.0',
            '  PARM 0 4.0', '  PARM 1 1.0e-03', '  PARM 2 2.0e-03',
            '  DIAM 5.0',
            'SURF 2',
            '  TYPE STANDARD', '  CURV -0.005 0 0 0 0 ""', '  DISZ 50.0',
            '  DIAM 5.0',
            'SURF 3',
            '  TYPE STANDARD', '  CURV 0.0', '  DISZ 0.0', '  DIAM 5.0',
            'BLNK',
        ]
        rx, _ = _load(lines)
        assert rx['surfaces'][0].get('freeform_type') == 'q_bfs'
        return rx

    def test_qbfs_refractive_export_warns_lens_only_writer(self):
        """Lens-only .zmx writer path (no mirrors/coord-breaks): the Q-bfs
        surface must trigger a loud drop warning (pre-fix: silent)."""
        rx = self._qbfs_refractive_rx()
        _, _, wlist = _export_roundtrip(rx)
        hits = [m for m in _msgs(wlist)
                if 'q_bfs' in m and 'BASE CONIC' in m]
        assert len(hits) == 1, (
            f"expected one Q-type drop warning, got {_msgs(wlist)}")

    def test_qbfs_mirror_export_warns_full_writer(self):
        """Full (mirror-aware) writer path: a Q-bfs MIRROR must trigger the
        same loud drop warning."""
        lines = _HEADER + [
            'SURF 0',
            '  TYPE STANDARD', '  CURV 0.0', '  DISZ INFINITY', '  DIAM 5.0',
            'SURF 1',
            '  TYPE QBFS', '  STOP', '  CURV 0.002 0 0 0 0 ""',
            '  DISZ -50.0', '  GLAS MIRROR 0 0 1.5 50.0',
            '  PARM 0 4.0', '  PARM 1 1.0e-03',
            '  DIAM 5.0',
            'SURF 2',
            '  TYPE STANDARD', '  CURV 0.0', '  DISZ 0.0', '  DIAM 5.0',
            'BLNK',
        ]
        rx, _ = _load(lines)
        mirr = [e for e in rx['elements']
                if e['element_type'] == 'mirror'][0]
        assert mirr.get('freeform_type') == 'q_bfs'
        _, _, wlist = _export_roundtrip(rx)
        hits = [m for m in _msgs(wlist)
                if 'q_bfs' in m and 'BASE CONIC' in m]
        assert len(hits) == 1, (
            f"expected one Q-type drop warning, got {_msgs(wlist)}")

    def test_plain_prescription_export_does_not_warn(self):
        """Non-regression: exporting a plain conic prescription emits no
        Q-type / SURFTYPE warnings."""
        lines = _unknown_type_zmx('STANDARD', parm_lines=())
        rx, _ = _load(lines)
        _, _, wlist = _export_roundtrip(rx)
        assert not [m for m in _msgs(wlist)
                    if 'BASE CONIC' in m or 'SURFTYPE' in m]
