"""S4-9 (AUDIT_V5_24_2) -- IO silent-fallback bundle regressions.

Four independent robustness defects at the IO boundary let bad or
lossy input pass without any diagnostic (or crash on one backend but
not the other):

(4) ``append_plane_h5`` / ``save_jones_field_h5`` wrote every metadata
    value straight into an HDF5 attribute.  A ``None`` value raises
    deep in h5py's C layer -- yet the SAME unified-API call stores
    ``None`` fine on the zarr backend, so the behaviour was
    backend-divergent.  ``save_field_h5`` already skips ``None``; the
    two siblings did not.  Fix: mirror the ``if value is None:
    continue`` guard.  (Highest value / lowest risk.)

(2) An unrecognised ``UNIT`` (``.zmx``) / ``Lens Units`` (``.txt``)
    token silently defaulted to millimeters -- a potential
    order-of-magnitude mis-scale with no warning.  Fix: warn before the
    mm fallback.

(3) A prescription with no STOP surface and no semi-diameter data
    yields ``aperture_diameter == 0.0`` -- a silently fully-clipped
    downstream field.  Fix: warn on the 0.0 aperture.

(1) The ``.txt`` loader dropped ``is_stop`` / ``stop_index`` (the
    ``.zmx`` twin carries them via the ZX-3 fix), so a declared stop
    relocated to surface 0 on re-export.  Fix: mirror the ZX-3
    per-surface ``is_stop`` + top-level ``stop_index`` onto the ``.txt``
    return dict.  (The ``coord_breaks`` half is deferred to S4-8's
    shared ``_finalize_surfaces`` extraction -- see caveats.)

Each test pins the observable contract against an independent probe /
pre-fix oracle rather than a tautology.  The HDF5 tests importorskip
``h5py`` (sibling to the storage suite's own guard); the zemax tests
are pure-NumPy.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.io.prescriptions_zemax import (
    load_zemax_prescription_data_txt,
    load_zemax_zmx,
)


def _write(path, text):
    path.write_text(text, encoding='utf-8')
    return str(path)


# =========================================================================
# (4) HDF5 metadata None-skip guard -- kills the backend-divergent crash.
# =========================================================================

class TestS4_9HDF5NoneMetadataGuard:

    def test_pre_fix_oracle_raw_h5py_none_attr_raises(self, tmp_path):
        """Independent probe proving the defect is real: writing a raw
        ``None`` into an h5py attribute raises (deep in the C layer).
        This is exactly what the unguarded metadata loop did -- and what
        the zarr backend tolerated, hence the divergence."""
        h5py = pytest.importorskip('h5py')
        p = str(tmp_path / 'oracle.h5')
        with h5py.File(p, 'w') as f:
            grp = f.create_group('g')
            with pytest.raises(Exception):
                grp.attrs['bad'] = None

    def test_append_plane_h5_skips_none_metadata(self, tmp_path):
        """POST-fix: a ``None`` metadata value no longer crashes the HDF5
        backend; it is dropped (reads back as absent) while sibling
        non-None keys survive -- matching the zarr backend and
        ``save_field_h5``."""
        pytest.importorskip('h5py')
        from lumenairy.io.storage import append_plane_h5, load_planes_h5
        p = str(tmp_path / 'planes.h5')
        field = np.ones((8, 8), dtype=np.complex128)
        # Would raise pre-fix on the ``note=None`` entry.
        append_plane_h5(p, field, dx=1e-6,
                        metadata={'source': 'probe', 'note': None},
                        swmr=False)
        planes, _ = load_planes_h5(p)
        assert len(planes) == 1
        # Non-None key survives; None key was dropped (absent), not stored.
        assert planes[0]['source'] == 'probe'
        assert 'note' not in planes[0]

    def test_save_jones_field_h5_skips_none_metadata(self, tmp_path):
        """POST-fix: same guard on the Jones-field writer."""
        pytest.importorskip('h5py')
        from lumenairy.elements.polarization import JonesField
        from lumenairy.io.storage import (
            load_jones_field_h5,
            save_jones_field_h5,
        )
        p = str(tmp_path / 'jones.h5')
        Ex = np.ones((6, 6), dtype=np.complex128)
        Ey = np.zeros((6, 6), dtype=np.complex128)
        jf = JonesField(Ex, Ey, 1e-6, 1e-6)
        # Would raise pre-fix on the ``pol=None`` entry.
        save_jones_field_h5(p, jf, metadata={'run': 7, 'pol': None})
        _, meta = load_jones_field_h5(p)
        assert meta['run'] == 7
        assert 'pol' not in meta


# =========================================================================
# (2) Unrecognised UNIT / Lens Units token -> warn before mm fallback.
# =========================================================================

_UNKNOWN_UNIT_ZMX = """UNIT FURLONGS
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

_KNOWN_UNIT_ZMX = _UNKNOWN_UNIT_ZMX.replace('UNIT FURLONGS', 'UNIT MM')


def _txt_rows(unit_line='Lens Units              :   Millimeters',
              stop_label='1', diam='20'):
    return '\n'.join([
        'System/Prescription Data',
        unit_line,
        '',
        'SURFACE DATA SUMMARY:',
        '',
        'Surf\tType\tRadius\tThickness\tGlass\tClear Diam\tChip Zone'
        '\tMech Diam\tConic\tComment',
        f'OBJ\tSTANDARD\tInfinity\t100\t\t{diam}\t0\t{diam}\t0\t',
        f'{stop_label}\tSTANDARD\t50\t5\tN-BK7\t{diam}\t0\t{diam}\t-1\tfront',
        f'2\tSTANDARD\t-50\t95\t\t{diam}\t0\t{diam}\t0\t',
        'IMA\tSTANDARD\tInfinity\t0\t\t2\t0\t2\t0\t',
    ]) + '\n'


class TestS4_9UnknownUnitWarns:

    def test_zmx_unknown_unit_warns(self, tmp_path):
        p = _write(tmp_path / 'furlong.zmx', _UNKNOWN_UNIT_ZMX)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            load_zemax_zmx(p)
        hits = [w for w in ws if issubclass(w.category, UserWarning)
                and 'FURLONGS' in str(w.message)]
        assert len(hits) == 1, [str(w.message) for w in ws]

    def test_zmx_known_unit_does_not_warn(self, tmp_path):
        """The guard must not fire on a recognised token (no false
        positive on clean input)."""
        p = _write(tmp_path / 'mm.zmx', _KNOWN_UNIT_ZMX)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            load_zemax_zmx(p)
        assert not [w for w in ws if 'UNIT token' in str(w.message)]

    def test_txt_unknown_unit_warns(self, tmp_path):
        p = _write(tmp_path / 'furlong.txt',
                   _txt_rows(unit_line='Lens Units          :   Furlongs'))
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            load_zemax_prescription_data_txt(p)
        hits = [w for w in ws if issubclass(w.category, UserWarning)
                and 'Furlongs' in str(w.message)]
        assert len(hits) == 1, [str(w.message) for w in ws]

    def test_txt_known_unit_does_not_warn(self, tmp_path):
        p = _write(tmp_path / 'mm.txt', _txt_rows())
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            load_zemax_prescription_data_txt(p)
        assert not [w for w in ws if 'Lens Units' in str(w.message)]


# =========================================================================
# (3) No-STOP + no-DIAM prescription -> aperture 0.0 must warn.
# =========================================================================

_NO_APERTURE_ZMX = """UNIT MM
SURF 0
  TYPE STANDARD
  DISZ 10.0
SURF 1
  TYPE STANDARD
  CURV 0.02
  DISZ 5.0
  GLAS N-BK7 0 0 1.5 50.0
SURF 2
  TYPE STANDARD
  CURV -0.02
  DISZ 95.0
SURF 3
  TYPE STANDARD
  DISZ 0.0
"""


class TestS4_9ZeroApertureWarns:

    def test_zmx_zero_aperture_warns_and_is_zero(self, tmp_path):
        """No STOP and no DIAM -> aperture 0.0 (fully clips downstream);
        the loader must warn."""
        p = _write(tmp_path / 'noap.zmx', _NO_APERTURE_ZMX)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            rx = load_zemax_zmx(p)
        assert rx['aperture_diameter'] == 0.0
        hits = [w for w in ws if issubclass(w.category, UserWarning)
                and 'aperture_diameter is 0.0' in str(w.message)]
        assert len(hits) == 1, [str(w.message) for w in ws]

    def test_zmx_nonzero_aperture_does_not_warn(self, tmp_path):
        """A prescription with DIAM data yields a non-zero aperture and
        must not emit the zero-aperture warning (no false positive)."""
        p = _write(tmp_path / 'ap.zmx', _KNOWN_UNIT_ZMX)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            rx = load_zemax_zmx(p)
        assert rx['aperture_diameter'] > 0.0
        assert not [w for w in ws
                    if 'aperture_diameter is 0.0' in str(w.message)]

    def test_txt_zero_aperture_warns(self, tmp_path):
        """.txt twin: zero Clear Diam on every surface -> aperture 0.0
        must warn."""
        p = _write(tmp_path / 'noap.txt', _txt_rows(diam='0'))
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            rx = load_zemax_prescription_data_txt(p)
        assert rx['aperture_diameter'] == 0.0
        hits = [w for w in ws if issubclass(w.category, UserWarning)
                and 'aperture_diameter is 0.0' in str(w.message)]
        assert len(hits) == 1, [str(w.message) for w in ws]


# =========================================================================
# (1) .txt loader must carry is_stop + stop_index (ZX-3 parity).
# =========================================================================

class TestS4_9TxtStopParity:

    def test_txt_return_dict_exposes_stop_index(self, tmp_path):
        """A declared 'STO' surface must surface as a top-level
        ``stop_index`` (index into the lens-only ``surfaces`` list),
        mirroring the .zmx twin -- so a re-export keys on the DECLARED
        stop instead of relocating it to surface 0."""
        # Label the SECOND lens surface (radius -50) as the stop.
        rows = '\n'.join([
            'SURFACE DATA SUMMARY:',
            '',
            'Surf\tType\tRadius\tThickness\tGlass\tClear Diam\tChip Zone'
            '\tMech Diam\tConic\tComment',
            'OBJ\tSTANDARD\tInfinity\t100\t\t20\t0\t20\t0\t',
            '1\tSTANDARD\t50\t5\tN-BK7\t20\t0\t20\t-1\tfront',
            'STO\tSTANDARD\t-50\t95\t\t20\t0\t20\t0\t',
            'IMA\tSTANDARD\tInfinity\t0\t\t2\t0\t2\t0\t',
        ]) + '\n'
        p = _write(tmp_path / 'stop.txt', rows)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rx = load_zemax_prescription_data_txt(p)
        # stop_index key now present (pre-fix: absent entirely).
        assert 'stop_index' in rx
        surfaces = rx['surfaces']
        assert rx['stop_index'] is not None
        # It indexes the lens-only surface flagged is_stop.
        assert surfaces[rx['stop_index']]['is_stop'] is True
        # Exactly one surface is the stop; it is the -50-radius exit
        # surface (in metres, mm-scaled), not surface 0.
        stops = [i for i, s in enumerate(surfaces) if s.get('is_stop')]
        assert stops == [rx['stop_index']]
        assert surfaces[rx['stop_index']]['radius'] == pytest.approx(-50e-3)
        assert rx['stop_index'] != 0

    def test_txt_no_stop_yields_none_stop_index(self, tmp_path):
        """With no 'STO' surface the key must be present but ``None``
        (matching the .zmx twin) -- not silently absent, not 0."""
        p = _write(tmp_path / 'nostop.txt', _txt_rows(stop_label='1'))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rx = load_zemax_prescription_data_txt(p)
        assert 'stop_index' in rx
        assert rx['stop_index'] is None
        assert all(not s.get('is_stop') for s in rx['surfaces'])
