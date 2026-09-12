"""WP-A10 / AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11 -- CODE V ``.seq`` I/O.

Covers findings I1 (P0, ``DIM`` lens units + writer default + migration
marker) and I3 (``REFL`` / ``RMD REFL`` mirrors, ``K`` conic, ``A``..``J``
aspheric coefficients, the dead ``radius`` default).

Every bar here is an EXACT identity (a unit conversion, a round-trip, a
closed-form sag), so the "measured value vs error floor" discipline reduces
to: the quantity is exact in IEEE-754 double up to the decimal-formatting
width the writer uses, and the tolerances below are set from that width, not
from one run's residual.  Each assertion names the pre-fix number it would
have produced, measured on the pre-fix HEAD with
``docs/audits/.../repro/IO-OPTIMIZE/p2_codev.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

import lumenairy as la
from lumenairy.io.prescriptions_code_v import (
    _CV_ASPH_LETTER_TO_POWER,
    _CV_DIM_TO_METERS,
    _CV_FORMAT_MARKER,
    _CV_WRITER_BANNER,
)

# ---------------------------------------------------------------------------
# Fixtures: hand-written CODE V sequences (the format CODE V itself writes)
# ---------------------------------------------------------------------------

_DOUBLET_SEQ = """! AC254-100-style doublet
LEN NEW
DIM {dim}
WL 587.6
REF 1
SO
  RDY INFINITY
  THI INFINITY
S1
  STO
  RDY 62.75
  THI 4.0
  GLA N-BAF10
S2
  RDY -45.71
  THI 2.5
  GLA N-SF6HT
S3
  RDY -128.23
  THI 95.0
SI
  RDY INFINITY
  THI 0.0
GO
END
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding='utf-8')
    return str(p)


# ---------------------------------------------------------------------------
# I1 -- DIM lens units
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('token, metres_per_unit', [
    ('M', 1e-3),        # CODE V: M == MILLIMETRES.  Pre-fix: 1.0 (metres).
    ('C', 1e-2),        # Pre-fix: token unrecognised -> fell to the M=1.0
    ('I', 0.0254),      #   default, so C and I ALSO read as metres.
    ('MM', 1e-3),       # tolerated aliases (not CODE V tokens)
    ('CM', 1e-2),
    ('IN', 0.0254),
])
def test_i1_dim_token_scales_lengths(tmp_path, token, metres_per_unit):
    """``DIM M`` is millimetres; ``C`` / ``I`` are recognised at all.

    Oracle: the unit definition itself (1 mm = 1e-3 m exactly, 1 in =
    0.0254 m exactly by definition).  The file says ``RDY 62.75``, so the
    only correct answer is ``62.75 * metres_per_unit`` -- an exact product of
    two double-representable numbers up to one rounding, hence rel=1e-15
    (~4.5 ULP at this magnitude).  Pre-fix measured values (p2_codev.py on
    HEAD): DIM M -> 62.75 m (x1000 too large), DIM C -> 62.75 m (x100),
    DIM I -> 62.75 m (x39.37).
    """
    p = _write(tmp_path, f'dim_{token}.seq',
               _DOUBLET_SEQ.format(dim=token))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rx = la.load_codev_seq(p)
    assert rx['surfaces'][0]['radius'] == pytest.approx(
        62.75 * metres_per_unit, rel=1e-15)
    assert rx['surfaces'][1]['radius'] == pytest.approx(
        -45.71 * metres_per_unit, rel=1e-15)
    assert rx['thicknesses'][0] == pytest.approx(
        4.0 * metres_per_unit, rel=1e-15)
    assert rx['back_focal_length'] == pytest.approx(
        95.0 * metres_per_unit, rel=1e-15)


def test_i1_dim_m_gives_millimetre_scale_efl(tmp_path):
    """End-to-end: the doublet's EFL lands in millimetres, not metres.

    Independent oracle: ``system_abcd_prescription`` is the library's paraxial
    ABCD trace, which the audit cross-checked against a hand-rolled
    surface-by-surface ray transfer (84.1429 mm on AC254_100_C.zmx, exact).
    Here it is used only to confirm the SCALE of the loaded prescription, and
    the bar is three orders of magnitude wide: pre-fix this file gave
    EFL = 72.2154 **m**, post-fix 0.0722154 m.  A bar of "< 1 m" therefore
    sits ~72x below the pre-fix value and ~14x above the correct one.
    """
    from lumenairy.raytrace.seidel import system_abcd_prescription
    p = _write(tmp_path, 'dim_m.seq', _DOUBLET_SEQ.format(dim='M'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rx = la.load_codev_seq(p)
    _, efl, _, _ = system_abcd_prescription(rx, 587.6e-9)
    assert 0.05 < efl < 0.10, (
        f'EFL {efl} m -- pre-fix this file loaded as 72.2154 m')
    assert efl == pytest.approx(0.0722154, rel=2e-5)


def test_i1_unknown_dim_token_warns(tmp_path):
    """An unrecognised ``DIM`` token must warn, not fall through silently.

    Pre-fix: ``if unit_tok in ('M','MM','IN')`` -- anything else left the
    default in force with no diagnostic, i.e. a silent order-of-magnitude
    mis-scale.  The Zemax loader has warned in exactly this situation since
    audit S4-9 (``Unrecognised UNIT token``); this is the CODE V twin.
    """
    p = _write(tmp_path, 'dim_bogus.seq', _DOUBLET_SEQ.format(dim='BOGUS'))
    with pytest.warns(UserWarning, match='DIM'):
        rx = la.load_codev_seq(p)
    # Falls back to CODE V's own default lens unit (millimetres).
    assert rx['surfaces'][0]['radius'] == pytest.approx(62.75e-3, rel=1e-15)


def test_i1_writer_emits_codev_millimetres(tmp_path):
    """The writer's ``units='M'`` default must mean CODE V millimetres.

    Pre-fix the writer emitted ``DIM M`` with SI-METRE numbers (scale 1.0),
    so every exported file was 1000x too large when CODE V opened it -- while
    the docstring advertised the unit kwarg as "useful for handing files to
    CODE V users".  Exact check: a 50 mm radius must appear as the NUMBER 50.
    """
    rx = la.make_singlet(50e-3, -30e-3, 3e-3, 'N-BK7', aperture=25.4e-3)
    p = str(tmp_path / 'w.seq')
    la.export_codev_seq(rx, p, wavelength=1.31e-6)
    txt = Path(p).read_text(encoding='utf-8')
    assert '\nDIM M\n' in txt
    assert 'RDY 50.00000000' in txt      # pre-fix: 'RDY 0.05000000'
    assert 'THI 3.00000000' in txt
    # CODE V has no 'MM' / 'CM' / 'IN' DIM token; aliases must normalise.
    for alias, expect in (('MM', 'DIM M'), ('CM', 'DIM C'), ('IN', 'DIM I')):
        q = str(tmp_path / f'w_{alias}.seq')
        la.export_codev_seq(rx, q, wavelength=1.31e-6, units=alias)
        body = Path(q).read_text(encoding='utf-8')
        assert f'\n{expect}\n' in body
        assert 'DIM MM' not in body and 'DIM CM' not in body


def test_i1_writer_load_roundtrip_is_exact(tmp_path):
    """export -> load is a lossless identity on radii / thicknesses / BFL.

    The writer formats with ``%.8f`` in lens units; at millimetre scale that
    is 1e-8 mm = 1e-11 m of quantisation on a ~5e-2 m radius, i.e. a relative
    floor of 2e-10.  The bar below (rel=1e-9) sits one decade above that floor
    and ~7 decades below any real defect.
    """
    rx = la.make_doublet(R1=50e-3, R2=-30e-3, R3=-80e-3, d1=4e-3, d2=2e-3,
                         glass1='N-BK7', glass2='N-SF6HT', aperture=10e-3)
    p = str(tmp_path / 'rt.seq')
    la.export_codev_seq(rx, p, wavelength=1.31e-6, stop_surface=0,
                        back_focal_length=43.21e-3)
    back = la.load_codev_seq(p)
    for a, b in zip(rx['surfaces'], back['surfaces']):
        assert b['radius'] == pytest.approx(a['radius'], rel=1e-9)
        assert b['glass_after'] == a['glass_after']
    for a, b in zip(rx['thicknesses'], back['thicknesses']):
        assert b == pytest.approx(a, rel=1e-9)
    assert back['back_focal_length'] == pytest.approx(43.21e-3, rel=1e-9)
    assert back['stop_index'] == 0


def test_i1_legacy_lumenairy_file_is_detected_and_read_as_metres(tmp_path):
    """A pre-v5.46 lumenairy ``.seq`` keeps its values, loudly.

    Migration contract: those files carry ``export_codev_seq``'s banner and
    NOT the format marker, and their ``DIM M`` numbers are SI metres.  That
    pairing is the only way to tell them from a genuine CODE V file, so the
    loader applies the legacy metre scale and says so.  ``dim_units='M'``
    forces the CODE V reading (x1e-3), which is the escape hatch for a file
    that was hand-edited into real CODE V units.
    """
    legacy = '\n'.join([
        _CV_WRITER_BANNER,
        '! test_lens',
        'LEN NEW',
        'DIM M',
        'WL 1550.0000',
        'S1',
        '  STO',
        '  RDY 0.02500000',
        '  THI 0.00300000',
        '  GLA N-BK7',
        'S2',
        '  RDY -0.02500000',
        '  THI 0.04321000',
        'SI',
        '  RDY INFINITY',
        '  THI 0.04321000',
        'END',
    ])
    p = _write(tmp_path, 'legacy.seq', legacy)
    with pytest.warns(UserWarning, match='before v5.46'):
        rx = la.load_codev_seq(p)
    assert rx['surfaces'][0]['radius'] == pytest.approx(0.025, rel=1e-12)
    assert rx['back_focal_length'] == pytest.approx(0.04321, rel=1e-12)
    # Forced CODE V reading of the same bytes: 1000x smaller.
    forced = la.load_codev_seq(p, dim_units='M')
    assert forced['surfaces'][0]['radius'] == pytest.approx(25e-6, rel=1e-12)
    # A file the FIXED writer produced carries the marker, so it is never
    # mistaken for a legacy one.
    rx2 = la.make_singlet(25e-3, -25e-3, 3e-3, 'N-BK7')
    q = str(tmp_path / 'new.seq')
    la.export_codev_seq(rx2, q, wavelength=1.55e-6)
    body = Path(q).read_text(encoding='utf-8')
    assert _CV_FORMAT_MARKER in body and _CV_WRITER_BANNER in body
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        back = la.load_codev_seq(q)          # must NOT warn legacy
    assert back['surfaces'][0]['radius'] == pytest.approx(25e-3, rel=1e-9)


def test_i1_dim_table_matches_codev_definition():
    """The unit table itself, pinned against the definitions."""
    assert _CV_DIM_TO_METERS['M'] == 1e-3
    assert _CV_DIM_TO_METERS['C'] == 1e-2
    assert _CV_DIM_TO_METERS['I'] == 0.0254


# ---------------------------------------------------------------------------
# I3 -- REFL / RMD REFL, K, A..J, radius default
# ---------------------------------------------------------------------------

_ASPH_REFL_SEQ = """LEN NEW
DIM M
WL 587.6
S1
  STO
  RDY 62.75
  THI 4.0
  GLA N-BK7
  ASP
  K -1.0
  A 1.234E-07
  B -5.0E-11
S2
  RDY -50.0
  THI 30.0
  REFL
S3
  RDY INFINITY
  THI 50.0
  RMD REFL
SI
  RDY INFINITY
  THI 0.0
END
"""


def test_i3_conic_and_aspheric_coefficients_are_parsed(tmp_path):
    """``K`` and ``A``/``B`` must reach the prescription.

    Oracle for the coefficient scale: the lens-unit -> SI rule
    ``a_p = a_file / L**(p-1)`` with L = 1e-3 m per lens unit, the same rule
    the Zemax loader applies (verified exact in the audit's "checked and found
    correct" list).  ``A 1.234E-07`` [1/mm^3] -> 1.234e-7 * 1e9 = 123.4
    [1/m^3]; ``B -5.0E-11`` [1/mm^5] -> -5e-11 * 1e15 = -5e4 [1/m^5].  Both
    are exact products of powers of ten, so rel=1e-12 is ~4 decades above the
    double floor.  Pre-fix measured: conics [0,0,0], asph [None,None,None] --
    the K vanished with no mention even though an ``ASP`` warning fired.
    """
    p = _write(tmp_path, 'asph.seq', _ASPH_REFL_SEQ)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rx = la.load_codev_seq(p)
    s0 = rx['surfaces'][0]
    assert s0['conic'] == pytest.approx(-1.0, rel=1e-15)
    assert s0['aspheric_coeffs'] == {
        4: pytest.approx(123.4, rel=1e-12),
        6: pytest.approx(-5.0e4, rel=1e-12),
    }


def test_i3_refl_and_rmd_refl_import_as_mirrors(tmp_path):
    """A CODE V mirror must not import as an air-to-air dummy.

    Pre-fix measured: ``glasses [('air','N-BK7'), ('N-BK7','air'),
    ('air','air')]`` -- S2's ``REFL`` became a refractive air->air surface --
    ``'elements' in rx`` False and ``has_mirrors`` False, so the design
    silently un-folded AND lost the mirror's power.
    """
    p = _write(tmp_path, 'refl.seq', _ASPH_REFL_SEQ)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rx = la.load_codev_seq(p)
    assert 'elements' in rx and 'all_thicknesses' in rx
    kinds = [e['element_type'] for e in rx['elements']]
    assert kinds == ['surface', 'mirror', 'mirror']
    assert la.has_mirrors(rx) is True
    m1 = rx['elements'][1]
    assert m1['radius'] == pytest.approx(-50.0e-3, rel=1e-15)
    # The fold legs survive on the element list; the refractive-only view
    # collapses them (the same flattening load_zemax_zmx performs).
    assert rx['all_thicknesses'] == pytest.approx([4.0e-3, 30.0e-3], rel=1e-15)


def test_i3_surface_without_rdy_is_flat_not_none(tmp_path):
    """A legal CODE V dummy surface (THI only) must read as ``inf``.

    Pre-fix: ``current`` pre-seeded ``'radius': None``, so the documented
    ``s.get('radius', float('inf'))`` default was dead code and the
    prescription came back with ``radius=None`` -- which
    ``validate_prescription`` rejects ("is None (use np.inf for a flat
    surface)"), making the file unusable.
    """
    seq = """LEN NEW
DIM M
WL 587.6
S1
  RDY 62.75
  THI 4.0
  GLA N-BK7
S2
  THI 10.0
S3
  RDY -50.0
  THI 90.0
SI
  RDY INFINITY
  THI 0.0
END
"""
    p = _write(tmp_path, 'no_rdy.seq', seq)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rx = la.load_codev_seq(p)
    assert np.isinf(rx['surfaces'][1]['radius'])
    # The whole point: it now survives validation / the paraxial trace.
    from lumenairy.raytrace.seidel import system_abcd_prescription
    system_abcd_prescription(rx, 587.6e-9)


def test_i3_mirror_and_asphere_survive_the_writer_roundtrip(tmp_path):
    """load -> export -> load is an identity for mirrors, conics, aspheres."""
    p = _write(tmp_path, 'asph_refl.seq', _ASPH_REFL_SEQ)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rx = la.load_codev_seq(p)
    q = str(tmp_path / 'rt2.seq')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        la.export_codev_seq(rx, q, wavelength=587.6e-9)
        back = la.load_codev_seq(q)
    assert [e['element_type'] for e in back['elements']] == \
        [e['element_type'] for e in rx['elements']]
    for a, b in zip(rx['elements'], back['elements']):
        if np.isinf(a['radius']):
            assert np.isinf(b['radius'])
        else:
            assert b['radius'] == pytest.approx(a['radius'], rel=1e-8)
        assert b['conic'] == pytest.approx(a['conic'], rel=1e-8)
    a0 = rx['elements'][0]['aspheric_coeffs']
    b0 = back['elements'][0]['aspheric_coeffs']
    assert set(a0) == set(b0)
    for k in a0:
        assert b0[k] == pytest.approx(a0[k], rel=1e-8)


def test_i3_codev_aspheric_letter_map_skips_i():
    """CODE V's asphere letters run A,B,C,D,E,F,G,H,J (no ``I``)."""
    assert _CV_ASPH_LETTER_TO_POWER['A'] == 4
    assert _CV_ASPH_LETTER_TO_POWER['B'] == 6
    assert _CV_ASPH_LETTER_TO_POWER['H'] == 18
    assert _CV_ASPH_LETTER_TO_POWER['J'] == 20
    assert 'I' not in _CV_ASPH_LETTER_TO_POWER


def test_i7_codev_loader_warns_about_unknown_glasses(tmp_path):
    """Parity with the two Zemax loaders' unknown-glass block.

    ``BK7`` is deliberately NOT in ``GLASS_REGISTRY`` (the audit confirmed
    glass names do not silently alias to the ``N-`` variants), and the repo's
    own CODE V fixture used to carry it -- so the failure surfaced much later
    as a ``ValueError`` from ``get_glass_index`` inside a propagation.
    """
    seq = _DOUBLET_SEQ.format(dim='M').replace('N-BAF10', 'BK7')
    p = _write(tmp_path, 'unknown_glass.seq', seq)
    with pytest.warns(UserWarning, match='GLASS_REGISTRY'):
        la.load_codev_seq(p)


def test_i4_codev_writer_warns_on_anamorphic_surface(tmp_path):
    """A cylindrical lens must not export as a sphere in silence.

    Pre-fix measured: ``cyl S0: radius=0.05 radius_y=inf`` -> ``.seq``
    ``radius_y preserved=False  warnings=[]``, i.e. a one-axis focusing
    element became a two-axis one with no diagnostic.  The Quadoa writer
    handles it, which makes the gap an inconsistency, not a format limit.
    """
    rx = la.make_cylindrical(50e-3, 3e-3, 'N-BK7', axis='x', aperture=25e-3)
    p = str(tmp_path / 'cyl.seq')
    with pytest.warns(UserWarning, match='ANAMORPHIC'):
        la.export_codev_seq(rx, p, wavelength=1.31e-6)
