"""WP-A10 / AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11 -- Zemax / codegen /
storage / transform findings (I2, I4, I5, I6, I7, I8).

Each test names the pre-fix measurement it would have produced; those numbers
come from the audit's own repro scripts
(``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/IO-OPTIMIZE/``)
re-run on the pre-fix HEAD.
"""

from __future__ import annotations

import ast
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

import lumenairy as la
from lumenairy.io.codegen import generate_simulation_script

# ---------------------------------------------------------------------------
# Zemax fixtures
# ---------------------------------------------------------------------------

_ZMX_HEAD = """VERS 210000 0 123 0 0
MODE SEQ
UNIT MM X W X CM MR CPMM
"""


def _zmx(tmp_path, name, body, head=_ZMX_HEAD):
    p = tmp_path / name
    p.write_text(head + body, encoding='utf-8')
    return str(p)


_PARAXIAL_PLUS_GLASS = """SURF 0
  TYPE STANDARD
  CURV 0
  DISZ INFINITY
SURF 1
  TYPE PARAXIAL
  STOP
  PARM 1 100.0
  DISZ 100.0
  DIAM 12.7
SURF 2
  TYPE STANDARD
  CURV 0.02
  DISZ 5.0
  GLAS N-BK7 0 0 1.5 50 0 0 0 0 0 0
  DIAM 12.7
SURF 3
  TYPE STANDARD
  CURV 0
  DISZ 50.0
  DIAM 12.7
SURF 4
  TYPE STANDARD
  CURV 0
  DISZ 0
  DIAM 12.7
"""


# ---------------------------------------------------------------------------
# I2 -- the window auto-detect deleted powered air-to-air surfaces
# ---------------------------------------------------------------------------

def test_i2_paraxial_surface_and_its_stop_are_not_deleted(tmp_path):
    """A ``TYPE PARAXIAL`` ideal lens ahead of the glass must survive import.

    Pre-fix measured (repro p1c_zemax2.py, case f2): ``elements surf_nums =
    [2, 3]``, ``stop_index = None``, ``warnings = []`` -- the 100 mm ideal
    lens AND the declared STOP were both gone, and only the lens's 100 mm
    DISZ survived as free space, because the window predicate admitted only
    glass / mirror / DGRATING.  The v5.32 DGRATING fix was this same repair
    applied to one surface type.

    Post-fix the surface is inside the window, so it reaches the
    unsupported-SURFTYPE branch and warns loudly instead of vanishing.
    """
    p = _zmx(tmp_path, 'paraxial.zmx', _PARAXIAL_PLUS_GLASS)
    with pytest.warns(UserWarning, match='PARAXIAL'):
        rx = la.load_zemax_zmx(p)
    surf_nums = [e.get('surf_num') for e in rx['elements']]
    assert 1 in surf_nums, (
        f'PARAXIAL surface 1 was dropped; elements = {surf_nums}')
    # Its declared STOP comes with it (pre-fix: stop_index None, aperture fell
    # back to the max DIAM over the window).
    assert rx['stop_index'] == 0


def test_i2_excluded_powered_surface_is_named(tmp_path):
    """Any optical surface the window still drops must be named once.

    Belt and braces for the SURFTYPEs the predicate cannot know: an explicit
    ``surface_range`` that cuts out a curved surface now says so.
    """
    body = """SURF 0
  TYPE STANDARD
  CURV 0
  DISZ INFINITY
SURF 1
  TYPE STANDARD
  CURV 0.05
  DISZ 10.0
  DIAM 12.7
SURF 2
  TYPE STANDARD
  CURV 0.02
  DISZ 5.0
  GLAS N-BK7 0 0 1.5 50 0 0 0 0 0 0
  DIAM 12.7
SURF 3
  TYPE STANDARD
  CURV 0
  DISZ 50.0
  DIAM 12.7
"""
    p = _zmx(tmp_path, 'cut.zmx', body)
    with pytest.warns(UserWarning, match='EXCLUDES'):
        la.load_zemax_zmx(p, surface_range=(2, 3))


def test_i2_ordinary_file_gains_no_new_warning(tmp_path):
    """The widened predicate must not fire on a plain doublet.

    Regression guard on the fix itself: a file whose only optical surfaces
    are the glass span must import exactly as before, with no diagnostic.
    """
    rx_path = Path('validation/real_lens_opd/zemax_prescriptions/'
                   'AC254_100_C.zmx')
    if not rx_path.exists():                      # pragma: no cover
        pytest.fail(f'fixture missing: {rx_path}')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        rx = la.load_zemax_zmx(str(rx_path))
    msgs = [str(x.message) for x in w]
    assert not [m for m in msgs if 'EXCLUDES' in m], msgs
    assert len(rx['surfaces']) == 3


# ---------------------------------------------------------------------------
# I4 -- anamorphic keys dropped by the Zemax writers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('exporter, suffix', [
    (la.export_zemax_zmx, '.zmx'),
    (la.export_zemax_lens_data, '.txt'),
])
def test_i4_zemax_writers_warn_on_anamorphic_surface(tmp_path, exporter,
                                                     suffix):
    """Pre-fix measured: ``.zmx radius_y preserved=False  warnings=[]``.

    The writers emit ``CURV`` from ``radius`` only, and the one
    anamorphic-adjacent guard (``_warn_dropped_qtype``) covers Q-type keys
    and nothing else -- so ``make_cylindrical`` exported as a sphere of
    revolution with no diagnostic.  Same loudness as the P2-20 precedent.
    """
    rx = la.make_cylindrical(50e-3, 3e-3, 'N-BK7', axis='x', aperture=25e-3)
    p = str(tmp_path / f'cyl{suffix}')
    with pytest.warns(UserWarning, match='ANAMORPHIC'):
        exporter(rx, p, wavelength=1.31e-6)


def test_i4_rotationally_symmetric_surface_does_not_warn(tmp_path):
    """``radius_y == radius`` (or both inf) loses nothing -- stay quiet."""
    rx = la.make_singlet(50e-3, -50e-3, 3e-3, 'N-BK7', aperture=25e-3)
    for s in rx['surfaces']:
        s['radius_y'] = s['radius']
    p = str(tmp_path / 'sym.zmx')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        la.export_zemax_zmx(rx, p, wavelength=1.31e-6)
    assert not [x for x in w if 'ANAMORPHIC' in str(x.message)]


# ---------------------------------------------------------------------------
# I5 -- codegen arbitrary-code injection
# ---------------------------------------------------------------------------

_INJECTION_PAYLOADS = [
    # The audit's proof of concept: closes the registry-key string, runs a
    # statement, comments out the tail.  Whitespace-free, so it survives the
    # loader's whitespace-split tokenizer.
    "X'];print(0x50574e4544);#",
    # Same shape with a REAL registry key as the prefix, so the leading
    # lookup also succeeds at runtime.
    "N-BK7'];__import__('builtins').print('PWNED');#",
]


@pytest.mark.parametrize('payload', _INJECTION_PAYLOADS)
def test_i5_hostile_glas_token_cannot_reach_a_code_position(tmp_path,
                                                            payload):
    """A doctored ``GLAS`` token must never become executable code.

    Pre-fix measured (repro p7b.py): the emitted line was
    ``la.GLASS_REGISTRY['X'];print(0x50574e4544);#'] = (...)`` and the FULL
    generated script parsed as valid Python -- i.e. the payload was live code
    that ran the moment the user executed the generated file.

    Two independent properties are asserted, because either one alone would
    be a single point of failure:
      1. the payload text never appears in a code position (the token is
         reduced to a plain name at the boundary), and
      2. every ``GLASS_REGISTRY`` line the generator emits is a lone
         subscript assignment -- checked by PARSING the emitted line and
         inspecting the AST, not by string matching.
    """
    body = f"""SURF 0
  TYPE STANDARD
  CURV 0
  DISZ INFINITY
SURF 1
  TYPE STANDARD
  STOP
  CURV 0.02
  DISZ 5.0
  GLAS {payload} 0 0 1.5 50 0 0 0 0 0 0
  DIAM 12.7
SURF 2
  TYPE STANDARD
  CURV 0
  DISZ 50.0
  DIAM 12.7
SURF 3
  TYPE STANDARD
  CURV 0
  DISZ 0
  DIAM 12.7
"""
    p = _zmx(tmp_path, 'evil.zmx', body)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rx = la.load_zemax_zmx(p)
        code = generate_simulation_script(
            rx, wavelength=1.31e-6, N=64,
            include_plotting=False, include_analysis=False)

    tree = ast.parse(code)                      # must still be valid Python
    reg_lines = [ln for ln in code.splitlines() if 'GLASS_REGISTRY' in ln]
    assert reg_lines, 'expected the unknown-glass registry block'
    for ln in reg_lines:
        stmts = ast.parse(ln.strip()).body
        assert len(stmts) == 1, f'injected extra statement(s): {ln!r}'
        assert isinstance(stmts[0], ast.Assign), ln
        tgt = stmts[0].targets[0]
        assert isinstance(tgt, ast.Subscript), ln
        # The subscript key must be a plain string constant.
        assert isinstance(tgt.slice, ast.Constant), ln
        assert isinstance(tgt.slice.value, str), ln

    # The decisive property: EXECUTING the emitted registry block against a
    # stub registry must have no side effect at all.  Pre-fix the payload ran
    # here (the audit's proof of concept printed from inside the generated
    # file).  Residual alphanumerics from the payload may survive INSIDE the
    # quoted key -- that is inert, and the AST checks above prove it is a
    # string constant, not code.
    import contextlib
    import io as _io
    stub = type('X', (object,), {'GLASS_REGISTRY': {}})()
    buf = _io.StringIO()
    reg_block = chr(10).join(reg_lines)
    with contextlib.redirect_stdout(buf):
        exec(compile(reg_block, 'gen', 'exec'), {'la': stub})
    assert buf.getvalue() == '', (
        f'the generated registry block executed a payload: {buf.getvalue()!r}')
    # No call to __import__ / print / exec anywhere in the emitted script's
    # module-level statements beyond the ones codegen itself writes.
    bad_calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name)
                 and n.func.id in ('__import__', 'eval', 'exec', 'compile')]
    assert not bad_calls, 'injected __import__/eval/exec call'


def test_i5_hostile_name_cannot_escape_the_docstring_or_print(tmp_path):
    """The system name is the file stem by default -- also untrusted."""
    body = _PARAXIAL_PLUS_GLASS
    p = _zmx(tmp_path, 'ok.zmx', body)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rx = la.load_zemax_zmx(p)
    rx['name'] = 'bad"""\nimport os\nos.system("echo pwned")\n"""'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        code = generate_simulation_script(rx, wavelength=1.31e-6, N=64,
                                          include_plotting=True,
                                          include_analysis=False)
    tree = ast.parse(code)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == 'system']
    assert not calls, 'os.system injected via the prescription name'
    # The text may survive inside the module docstring (inert); what must not
    # survive is an Import of ``os`` at statement level.
    imports = [n for n in ast.walk(tree)
               if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert 'os' not in {a.name for n in imports
                        if isinstance(n, ast.Import) for a in n.names}


# ---------------------------------------------------------------------------
# I7 -- codegen literal emission + normalize_prescription contract
# ---------------------------------------------------------------------------

def test_i7_codegen_emits_parseable_inf_nan_and_keeps_the_sign():
    """Pre-fix measured (repro p7b.py):

        GEN2> {"radius": 5.00000000000000028e-02, "conic": inf,
        GEN2> "aspheric_coeffs": {4: nan, 6: 100000.0},
        GEN2> {"radius": float('inf'), "conic": 0.0,     # this surface was -inf
          -> prescription block FAILS: NameError name 'inf' is not defined

    i.e. bare ``inf`` / ``nan`` (not Python literals) AND a sign flip on a
    ``-inf`` radius, because ``np.isinf`` is sign-blind.
    """
    rx = la.make_singlet(50e-3, float('-inf'), 4e-3, 'N-BK7', aperture=25e-3)
    for s in rx['surfaces']:
        s['element_type'] = 'surface'
        s['semi_diameter'] = 12.5e-3
    rx['surfaces'][0]['conic'] = float('inf')
    rx['surfaces'][0]['aspheric_coeffs'] = {4: float('nan'), 6: 1e5}
    rx['elements'] = rx['surfaces']
    rx['all_thicknesses'] = rx['thicknesses']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        code = generate_simulation_script(rx, wavelength=1.31e-6, N=64,
                                          include_plotting=False)
    head = code.split('# SIMULATION')[0]
    ns = {}
    exec(compile(head, 'gen', 'exec'), ns)      # pre-fix: NameError 'inf'
    lens = next(v for k, v in ns.items() if k.endswith('_RX'))
    assert math.isinf(lens['surfaces'][1]['radius'])
    assert lens['surfaces'][1]['radius'] < 0, (
        'sign of a -inf radius was flipped (pre-fix behaviour)')
    assert math.isinf(lens['surfaces'][0]['conic'])
    assert math.isnan(lens['surfaces'][0]['aspheric_coeffs'][4])
    assert lens['surfaces'][0]['aspheric_coeffs'][6] == 1e5


def test_i7_normalize_prescription_output_is_codegen_shaped():
    """Pre-fix: ``KeyError: 'element_type'``.

    ``normalize_prescription`` is documented as "the canonical superset ...
    the recommended idiom", and when ``elements`` is absent it set
    ``rx['elements'] = list(surfs)`` -- plain surface dicts with no
    ``element_type`` -- which codegen subscripted unguarded.  Both halves are
    fixed: the helper stamps the key, and codegen defaults it.
    """
    rx = la.normalize_prescription(
        la.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7'))
    rx['wavelength'] = 1.31e-6
    assert all(e.get('element_type') == 'surface' for e in rx['elements'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        code = generate_simulation_script(rx, N=64, include_plotting=False)
    ast.parse(code)
    # The stamp is applied in place on the shared dicts, so the documented
    # ``q['elements'] == q['surfaces']`` identity (and the aliasing it rests
    # on) survives -- both views now carry the canonical discriminator rather
    # than neither.
    assert rx['elements'] == rx['surfaces']
    assert rx['surfaces'][0]['element_type'] == 'surface'


# ---------------------------------------------------------------------------
# I7 -- scale_prescription self-similarity
# ---------------------------------------------------------------------------

def test_i7_scale_prescription_covers_q_type_diffractives_and_bfl():
    """Pre-fix measured at s = 0.25 (repro p7b.py):

        scaled r_max              : 0.0075     expected 0.001875
        scaled q_bfs_coeffs       : [1e-06, 2e-07]  expected [2.5e-07, 5e-08]
        scaled back_focal_length  : 0.084      expected 0.021
        scaled diffractive period : 2e-06      expected 5e-07
        scaled diffractive gap_before : 0.01   expected 0.0025

    All five are LENGTHS, so linear scaling is the definition of geometric
    self-similarity; the expected values are exact products, hence rel=1e-15.
    """
    rx = la.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7', aperture=25e-3)
    rx['surfaces'][0]['freeform_type'] = 'q_bfs'
    rx['surfaces'][0]['q_bfs_coeffs'] = [1e-6, 2e-7]
    rx['surfaces'][0]['r_max'] = 7.5e-3
    rx['back_focal_length'] = 0.084
    rx['diffractives'] = [{'type': 'grating', 'period': 2e-6, 'order': 1,
                           'origin': (1e-3, 0.0), 'gap_before': 1e-2,
                           'gap_after': 2e-2, 'semi_diameter': 6e-3,
                           'angle_deg': 0.0}]
    s = 0.25
    a = la.scale_prescription(rx, s)
    assert a['surfaces'][0]['r_max'] == pytest.approx(7.5e-3 * s, rel=1e-15)
    assert a['surfaces'][0]['q_bfs_coeffs'] == pytest.approx(
        [1e-6 * s, 2e-7 * s], rel=1e-15)
    assert a['back_focal_length'] == pytest.approx(0.084 * s, rel=1e-15)
    dg = a['diffractives'][0]
    assert dg['period'] == pytest.approx(2e-6 * s, rel=1e-15)
    assert dg['gap_before'] == pytest.approx(1e-2 * s, rel=1e-15)
    assert dg['gap_after'] == pytest.approx(2e-2 * s, rel=1e-15)
    assert dg['semi_diameter'] == pytest.approx(6e-3 * s, rel=1e-15)
    assert dg['origin'] == pytest.approx((1e-3 * s, 0.0), rel=1e-15, abs=1e-18)
    assert dg['order'] == 1            # dimensionless: must NOT scale
    # Round trip stays exact (the audit verified this identity pre-fix too --
    # it must not regress).
    b = la.scale_prescription(a, 1.0 / s)
    assert b['surfaces'][0]['r_max'] == pytest.approx(7.5e-3, rel=1e-14)
    assert b['back_focal_length'] == pytest.approx(0.084, rel=1e-14)
    assert b['diffractives'][0]['period'] == pytest.approx(2e-6, rel=1e-14)


def test_i7_scale_prescription_warns_on_unknown_length_key():
    """An unrecognised length-like key must not be left unscaled in silence."""
    rx = la.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7', aperture=25e-3)
    rx['surfaces'][0]['groove_depth_m'] = 1e-6
    with pytest.warns(UserWarning, match='length-like'):
        la.scale_prescription(rx, 0.5)


# ---------------------------------------------------------------------------
# I7 -- Zemax MNUM / MCON, Quadoa elements + stop
# ---------------------------------------------------------------------------

def test_i7_multiconfig_records_are_surfaced(tmp_path):
    """Pre-fix measured: multi-config file imported with no ``configurations``
    key and ``warnings: []`` -- the user got no signal that N-1 zoom positions
    existed, while ``optimize/multiconfig.py`` makes the feature look
    supported end to end.
    """
    body = """SURF 0
  TYPE STANDARD
  CURV 0
  DISZ INFINITY
SURF 1
  TYPE STANDARD
  STOP
  CURV 0.02
  DISZ 5.0
  GLAS N-BK7 0 0 1.5 50 0 0 0 0 0 0
  DIAM 12.7
SURF 2
  TYPE STANDARD
  CURV 0
  DISZ 20.0
  DIAM 12.7
MNUM 3 1
MCON THIC 2 2 20.0 0 0
MCON THIC 3 2 45.0 0 0
"""
    p = _zmx(tmp_path, 'zoom.zmx', body)
    with pytest.warns(UserWarning, match='MULTI-CONFIGURATION'):
        rx = la.load_zemax_zmx(p)
    cfg = rx['configurations']
    assert cfg is not None
    assert cfg['n_configs'] == 3
    assert len(cfg['operands']) == 2
    assert cfg['operands'][0]['operand'] == 'THIC'
    # VERIFY-A10 (V6): ``raw`` is the contract -- the row is kept verbatim.
    # The trailing fields are an UNDECODED positional split under
    # ``fields_provisional`` because the MCON layout after the operand token
    # is version-dependent and unconfirmed against an OpticStudio file; the
    # earlier named decode ('config' / 'surface') assigned the same
    # configuration number to every row of a 3-configuration fixture.
    assert cfg['operands'][0]['raw'] == 'MCON THIC 2 2 20.0 0 0'
    assert cfg['operands'][1]['raw'] == 'MCON THIC 3 2 45.0 0 0'
    assert cfg['operands'][0]['fields_provisional'] == [2.0, 2.0, 20.0, 0.0, 0.0]
    for row in cfg['operands']:
        assert 'config' not in row and 'surface' not in row, (
            'the unconfirmed named decode must not be advertised as a '
            'contract; read row["raw"]')


def test_i7_single_config_file_has_no_configurations_key_content(tmp_path):
    """A plain file must keep ``configurations = None`` and stay quiet."""
    p = _zmx(tmp_path, 'plain.zmx', _PARAXIAL_PLUS_GLASS)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        rx = la.load_zemax_zmx(p)
    assert rx['configurations'] is None
    assert not [x for x in w if 'MULTI-CONFIGURATION' in str(x.message)]


def test_i8_quadoa_writer_does_not_invent_a_stop(tmp_path):
    """Pre-fix: ``stop_surface`` defaulted to 0 and ``is_stop = (i == 0)`` was
    written on every surface, so re-loading ALWAYS yielded ``stop_index = 0``
    -- inventing an aperture stop the design never declared.
    """
    rx = la.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7', aperture=25e-3)
    assert 'stop_index' not in rx
    p = str(tmp_path / 'nostop.qos')
    la.export_quadoa_qos(rx, p, wavelength=1.31e-6)
    back = la.load_quadoa_qos(p)
    assert back.get('stop_index') is None
    # A declared stop still round-trips.
    rx2 = dict(rx)
    rx2['stop_index'] = 1
    la.export_quadoa_qos(rx2, p, wavelength=1.31e-6)
    assert la.load_quadoa_qos(p)['stop_index'] == 1


def test_i7_quadoa_loader_emits_elements_and_all_thicknesses(tmp_path):
    """Three docstrings promised this schema; the loader did not deliver it."""
    rx = la.make_doublet(R1=50e-3, R2=-30e-3, R3=-80e-3, d1=4e-3, d2=2e-3,
                         glass1='N-BK7', glass2='N-SF6HT', aperture=10e-3)
    p = str(tmp_path / 'd.qos')
    la.export_quadoa_qos(rx, p, wavelength=1.31e-6)
    back = la.load_quadoa_qos(p)
    assert len(back['elements']) == len(back['surfaces']) == 3
    assert all(e['element_type'] == 'surface' for e in back['elements'])
    assert back['all_thicknesses'] == back['thicknesses']
    # split_prescription_at_mirrors no longer takes the silent early return.
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        legs = la.split_prescription_at_mirrors(back)
    assert len(legs) == 1 and legs[0]['kind'] == 'refractive'


def test_i7_split_at_mirrors_warns_on_the_schema_free_fallback():
    """A ``surfaces``-only prescription cannot be inspected for folds."""
    rx = la.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7')
    with pytest.warns(UserWarning, match='no elements'):
        legs = la.split_prescription_at_mirrors(rx)
    assert len(legs) == 1


# ---------------------------------------------------------------------------
# I8 -- .zmx record injection, UTF-16-BE
# ---------------------------------------------------------------------------

def test_i8_zmx_name_newline_cannot_inject_records(tmp_path):
    """Pre-fix measured: a name of ``'bad\\nSURF 99\\n  CURV 0.5\\nNAME x'``
    produced extra ``SURF`` rows in the exported file, i.e. the file reloaded
    as a different system.
    """
    rx = la.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7', aperture=25e-3)
    p = str(tmp_path / 'inj.zmx')
    with pytest.warns(UserWarning, match='NAME'):
        la.export_zemax_zmx(rx, p, wavelength=1.31e-6,
                            name='bad\nSURF 99\n  CURV 0.5\nNAME x')
    txt = Path(p).read_text(encoding='utf-8')
    surf_rows = [ln for ln in txt.splitlines() if ln.startswith('SURF ')]
    # object + 2 refractive + image = 4.  Pre-fix: 5 (the injected SURF 99).
    assert len(surf_rows) == 4, surf_rows
    # The text survives INSIDE the single NAME record (inert); what must not
    # exist is a RECORD of its own.
    assert not [ln for ln in txt.splitlines()
                if ln.strip().startswith(('SURF 99', 'CURV 0.5'))]
    back = la.load_zemax_zmx(p)
    assert len(back['surfaces']) == 2


def test_i8_utf16be_zmx_is_read_not_misdiagnosed(tmp_path):
    """Pre-fix: a UTF-16-BE file decoded under latin-1 without a readable
    ``SURF`` and raised "does not appear to be a Zemax .zmx lens file" --
    pointing at the wrong cause.  BOM-sniffing ``utf-16`` handles both byte
    orders.
    """
    body = _ZMX_HEAD + """SURF 0
  TYPE STANDARD
  CURV 0
  DISZ INFINITY
SURF 1
  TYPE STANDARD
  STOP
  CURV 0.02
  DISZ 5.0
  GLAS N-BK7 0 0 1.5 50 0 0 0 0 0 0
  DIAM 12.7
SURF 2
  TYPE STANDARD
  CURV 0
  DISZ 50.0
  DIAM 12.7
"""
    p = tmp_path / 'be.zmx'
    p.write_bytes(b'\xfe\xff' + body.encode('utf-16-be'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rx = la.load_zemax_zmx(str(p))
    assert rx['surfaces'][0]['radius'] == pytest.approx(50e-3, rel=1e-12)


# ---------------------------------------------------------------------------
# I6 -- Zarr per-plane metadata fidelity
# ---------------------------------------------------------------------------

_META_PROBE = {
    'none': None, 'true': True, 'int': 7, 'float': 1.5, 'str': 'hi',
    'nan': float('nan'), 'inf': float('inf'), 'ninf': float('-inf'),
    'complex': 1 + 2j, 'bytes': b'\x00\x01ab',
    'tuple': (1, 2.5, 'x'), 'list': [1, 2, 3], 'emptylist': [],
    'hetlist': [1, 'a', None],
    'ndarray_f': np.arange(6.0).reshape(2, 3),
    'ndarray_c': np.array([1 + 1j, 2 - 2j], dtype=np.complex64),
    'nested': {'a': {'b': [1, 2]}, 'c': 3},
}


def _meta_faithful(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (type(a) is type(b) and a.dtype == b.dtype
                and np.array_equal(a, b, equal_nan=np.isrealobj(a)))
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return type(a) is type(b) and a == b


def test_i6_zarr_append_plane_metadata_matches_hdf5(tmp_path):
    """Pre-fix measured over this same 19-type probe set (repro p6_storage.py):

        HDF5 append_plane metadata: 18/19 faithful   (only np.float32 -> float)
        ZARR append_plane metadata: 13/19 faithful
               complex   -> str '(1+2j)'
               bytes     -> str "b'\\x00\\x01ab'"
               tuple     -> list
               ndarray_f -> str '[[0. 1. 2.]\\n [3. 4. 5.]]'   <-- irrecoverable
               ndarray_c -> str '[1.+1.j 2.-2.j]'
               np_scalar -> str '2.5'

    The ndarray case is not merely type-changed: ``str()`` of an array past
    numpy's 1000-element print threshold inserts ``...``, so the values are
    GONE.  ``_zarr_write_sim_metadata`` already used the canonical codec, so
    the same call with the same arguments had different fidelity depending on
    a global backend switch.
    """
    zarr = pytest.importorskip('zarr')
    del zarr
    from lumenairy.io import storage as S
    E = (np.random.default_rng(0).random((16, 16))
         + 1j * np.random.default_rng(1).random((16, 16))).astype(np.complex64)
    store = str(tmp_path / 'p.zarr')
    prev = S.get_storage_backend() if hasattr(S, 'get_storage_backend') else None
    S.set_storage_backend('zarr')
    try:
        S.append_plane(store, E, dx=1e-6, label='L0',
                       metadata=dict(_META_PROBE), preserve_dtype=True)
        planes, _ = S.load_planes(store)
    finally:
        S.set_storage_backend(prev or 'hdf5')
    back = planes[0]
    bad = [k for k, v in _META_PROBE.items()
           if not _meta_faithful(v, back.get(k, '<<MISSING>>'))]
    assert not bad, f'zarr metadata round-trip lost: {bad}'
    # And the big-array case the stringifier destroyed outright.
    big = np.arange(4096.0)
    store2 = str(tmp_path / 'q.zarr')
    S.set_storage_backend('zarr')
    try:
        S.append_plane(store2, E, dx=1e-6, label='L0',
                       metadata={'big': big}, preserve_dtype=True)
        planes2, _ = S.load_planes(store2)
    finally:
        S.set_storage_backend(prev or 'hdf5')
    assert np.array_equal(planes2[0]['big'], big)


# ---------------------------------------------------------------------------
# I7 -- HDF5 compression / chunk defaults
# ---------------------------------------------------------------------------

def test_i7_auto_compression_is_off_for_complex_and_gzip_otherwise():
    """The 'auto' rule itself, as a pure function -- no timing assertions.

    Rationale for the rule (measured, 1024^2 complex128, medians of 7
    interleaved runs on this workstation, 2026-09-12): gzip-4 costs
    **x57.9 write / x7.1 read for -5.6 % on disk** (0.4274 s vs 0.0074 s;
    15.12 vs 16.01 MiB).  Complex float mantissas are incompressible.  Real
    data keeps the historical gzip-4.
    """
    from lumenairy.io.storage import _resolve_chunk_edge, _resolve_compression
    assert _resolve_compression('auto', None, np.complex128) == (None, None)
    assert _resolve_compression('auto', None, np.complex64) == (None, None)
    assert _resolve_compression('auto', None, np.float64) == ('gzip', 4)
    assert _resolve_compression('auto', None, np.uint8) == ('gzip', 4)
    # Explicit values are honoured unchanged, in both directions.
    assert _resolve_compression('gzip', 4, np.complex128) == ('gzip', 4)
    assert _resolve_compression(None, None, np.float64) == (None, None)
    # Chunk budget: <= 1 MiB per chunk, power-of-two edge, clipped to shape.
    for dtype in (np.complex128, np.complex64, np.float64):
        edge = _resolve_chunk_edge('auto', (4096, 4096), dtype)
        nbytes = edge * edge * np.dtype(dtype).itemsize
        assert nbytes <= (1 << 20), (dtype, edge, nbytes)
        assert edge & (edge - 1) == 0
    assert _resolve_chunk_edge('auto', (100, 100), np.complex128) == 100
    assert _resolve_chunk_edge(1024, (4096, 4096), np.complex128) == 1024


def test_i7_complex_field_writes_uncompressed_by_default(tmp_path):
    """Structural check on the written file -- no wall-clock assertion.

    Reads the dataset's HDF5 filter pipeline back, so the property is the
    stored layout, not a timing that a shared box could move.
    """
    h5py = pytest.importorskip('h5py')
    from lumenairy.io import storage as S
    E = (np.random.default_rng(2).random((64, 64))
         + 1j * np.random.default_rng(3).random((64, 64)))
    p = str(tmp_path / 'f.h5')
    S.save_field_h5(p, E, dx=1e-6, preserve_dtype=True)
    with h5py.File(p, 'r') as f:
        assert f['field'].compression is None
    # Real data keeps gzip.
    q = str(tmp_path / 'r.h5')
    S.save_field_h5(q, np.abs(E), dx=1e-6, preserve_dtype=True)
    with h5py.File(q, 'r') as f:
        assert f['field'].compression == 'gzip'
    # Explicit request still wins.
    r = str(tmp_path / 'g.h5')
    S.save_field_h5(r, E, dx=1e-6, preserve_dtype=True, compression='gzip',
                    compression_opts=4)
    with h5py.File(r, 'r') as f:
        assert f['field'].compression == 'gzip'
    # Values are unchanged either way (this is a storage-layout change only).
    back, _ = S.load_field_h5(p)
    assert np.array_equal(back, E)


def test_i7_append_plane_chunk_is_about_one_mib(tmp_path):
    """``chunk_size=1024`` made a 16 MiB chunk for complex128 -- 16x HDF5's
    1 MiB default chunk cache, so every partial read touched a whole chunk.
    """
    h5py = pytest.importorskip('h5py')
    from lumenairy.io import storage as S
    E = np.zeros((1024, 1024), dtype=np.complex128)
    p = str(tmp_path / 'a.h5')
    S.append_plane_h5(p, E, dx=1e-6, label='L0', preserve_dtype=True)
    with h5py.File(p, 'r') as f:
        ch = f['planes/plane_00'].chunks
    nbytes = ch[0] * ch[1] * 16
    assert nbytes <= (1 << 20), (ch, nbytes)     # pre-fix: 1024*1024*16 = 16 MiB
