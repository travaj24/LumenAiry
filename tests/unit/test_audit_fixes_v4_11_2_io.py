"""Regression tests for the v4.11.2 IO-domain audit fixes.

Each test pins one of the round-3 IO findings that v4.11.2 addresses.
The tests are small (<1 s each); their purpose is to fail loudly if a
future refactor reintroduces the specific bug, not to exercise the
broader physics.

Findings covered (per ``AUDIT_ROUND3_2026_05_16.md`` "IO / Zemax /
prescriptions / storage / codegen" section):

* C-IO-1  EVENASPH PARM off-by-one (loader dropped PARM 1 = a_4 and
          mis-labelled every higher coefficient by one slot).
* C-IO-2  Quadoa aspheric serializer iterates dict keys instead of
          values (wrote ``[4.0, 6.0, ...]`` powers instead of the
          coefficients).
* C-IO-3  Zemax exporter STOP marker on wrong surface in folded
          designs (compared global surf_counter, not refractive index).
* C-IO-4  ``normalize_prescription`` mirror filter checked
          ``e.get('mirror')`` but library uses
          ``element_type='mirror'``; the filter was a no-op.
* C-IO-5  Mirror DISZ lost on Zemax round-trip (exporter applied a
          spurious mirror-parity sign-flip on top of an already
          Zemax-signed thickness).
* HIGH    codegen emitted ``op.GLASS_REGISTRY`` but imported
          ``lumenairy as la`` -> NameError on script execution.
* HIGH    codegen system-list style dropped ``aperture_diameter``
          for mirror elements.
* HIGH    ``aspheric_coeffs`` type harmonisation: every loader now
          produces dict-or-None (Quadoa loader was returning a list).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

import lumenairy as la
from lumenairy.io.prescriptions import (
    _export_zemax_zmx_full,
    _quadoa_deserialize_aspheric,
    _quadoa_serialize_aspheric,
    export_quadoa_qos,
    export_zemax_zmx,
    load_quadoa_qos,
    load_zemax_zmx,
    normalize_prescription,
)


# ============================================================================
# C-IO-1 -- EVENASPH PARM off-by-one (round-trip preserves alpha_4)
# ============================================================================


def _write_zmx(text, suffix='.zmx'):
    """Write a Zemax .zmx-style text file to a temp path and return it."""
    fd, path = tempfile.mkstemp(suffix=suffix, text=True)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(text)
    return path


def _minimal_evenasph_zmx(alpha4_per_mm3, alpha6_per_mm5=0.0):
    """A minimal 2-surface .zmx with one EVENASPH surface.

    PARM 1 (= alpha_4) is set to ``alpha4_per_mm3`` and PARM 2
    (= alpha_6) to ``alpha6_per_mm5``.  Returns the .zmx text.
    """
    lines = [
        'VERS 210000 0 123 0 0',
        'MODE SEQ',
        'NAME test_evenasph',
        'UNIT MM X W X CM MR CPMM',
        'ENPD 10.0',
        'WAVM 1 0.587600 1.0',
        'PWAV 1',
        'SURF 0',
        '  TYPE STANDARD',
        '  CURV 0.0',
        '  DISZ INFINITY',
        '  DIAM 5.0',
        'SURF 1',
        '  TYPE EVENASPH',
        '  STOP',
        '  CURV 0.01 0 0 0 0 ""',
        '  DISZ 3.0',
        '  GLAS BK7 0 0 1.5 50.0',
        f'  PARM 1 {alpha4_per_mm3:.10e}',
        f'  PARM 2 {alpha6_per_mm5:.10e}',
        '  DIAM 5.0',
        'SURF 2',
        '  TYPE STANDARD',
        '  CURV -0.005 0 0 0 0 ""',
        '  DISZ 50.0',
        '  DIAM 5.0',
        'SURF 3',
        '  TYPE STANDARD',
        '  CURV 0.0 0 0 0 0 ""',
        '  DISZ 0.0',
        '  DIAM 5.0',
        'BLNK',
    ]
    return '\n'.join(lines) + '\n'


class TestEvenAsphParm:

    def test_parm1_is_alpha4(self):
        """Pre-v4.11.2: loader's ``if parm_num >= 2`` filter dropped
        PARM 1 (alpha_4) entirely.  Post-fix: PARM 1 lands on power=4
        with correct unit conversion."""
        # Use a recognisably-large alpha_4 so the test fails
        # immediately if the filter regresses.
        alpha4_per_mm3 = 1.5e-5      # 1/mm^3
        zmx_text = _minimal_evenasph_zmx(alpha4_per_mm3, alpha6_per_mm5=0.0)
        path = _write_zmx(zmx_text)
        try:
            rx = load_zemax_zmx(path)
        finally:
            os.unlink(path)
        # First refracting surface is the EVENASPH.
        ac = rx['surfaces'][0]['aspheric_coeffs']
        assert ac is not None, (
            "alpha_4 (PARM 1) was dropped -- regression of the v4.11.2 "
            "off-by-one fix.")
        assert 4 in ac, (
            f"Expected power=4 key in aspheric_coeffs, got {list(ac.keys())}. "
            "The loader is mis-labelling PARM 1.")
        # Unit conversion: input is 1/mm^3, library stores 1/m^3.
        # 1 mm^-3 = 1e9 m^-3.
        expected_per_m3 = alpha4_per_mm3 * 1e9
        assert np.isclose(ac[4], expected_per_m3, rtol=1e-6), (
            f"alpha_4 unit conversion wrong: got {ac[4]}, "
            f"expected {expected_per_m3}.")

    def test_parm2_is_alpha6(self):
        """PARM 2 = alpha_6 in Zemax EVENASPH convention.  Pre-fix it
        was loaded as power=4."""
        alpha6_per_mm5 = 2.0e-9      # 1/mm^5
        zmx_text = _minimal_evenasph_zmx(
            alpha4_per_mm3=0.0, alpha6_per_mm5=alpha6_per_mm5)
        path = _write_zmx(zmx_text)
        try:
            rx = load_zemax_zmx(path)
        finally:
            os.unlink(path)
        ac = rx['surfaces'][0]['aspheric_coeffs']
        assert ac is not None
        # Only PARM 2 was set -> only power=6 should be present.
        assert 6 in ac, (
            f"Expected power=6 key, got {list(ac.keys())}. "
            "PARM 2 -> alpha_6 (power=6), not alpha_4 (power=4).")
        # 1 mm^-5 = 1e15 m^-5.
        expected = alpha6_per_mm5 * 1e15
        assert np.isclose(ac[6], expected, rtol=1e-6)

    def test_evenasph_full_round_trip_preserves_alpha4(self):
        """Synthetic prescription -> export .zmx -> reload -> assert
        alpha_4 survives.  This pins the loader-exporter pair against
        future drift."""
        alpha4 = 1.234e6   # 1/m^3 (library SI convention)
        rx_in = {
            'name': 'test_lens',
            'aperture_diameter': 10e-3,
            'surfaces': [
                {'radius': 100e-3, 'conic': 0.0,
                 'aspheric_coeffs': {4: alpha4},
                 'glass_before': 'air', 'glass_after': 'N-BK7'},
                {'radius': -100e-3, 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'glass_before': 'N-BK7', 'glass_after': 'air'},
            ],
            'thicknesses': [3e-3],
        }
        with tempfile.NamedTemporaryFile(
                suffix='.zmx', delete=False) as f:
            path = f.name
        try:
            export_zemax_zmx(rx_in, path, wavelength=587.6e-9)
            rx_out = load_zemax_zmx(path)
        finally:
            os.unlink(path)
        ac = rx_out['surfaces'][0]['aspheric_coeffs']
        assert ac is not None, (
            "alpha_4 lost on Zemax round-trip (.zmx export -> load).")
        assert 4 in ac, (
            f"Expected power=4 key after round-trip; got {list(ac.keys())}.")
        assert np.isclose(ac[4], alpha4, rtol=1e-6), (
            f"alpha_4 value drifted: in={alpha4}, out={ac[4]}.")


# ============================================================================
# C-IO-2  Quadoa aspheric serializer (iterate values, not keys)
# ============================================================================


class TestQuadoaAsphericSerializer:

    def test_serialize_writes_coefficient_values_not_powers(self):
        """Pre-fix the serializer did ``[float(c) for c in coeffs]`` on
        a dict, which iterates KEYS in Python -> wrote [4.0, 6.0, ...]
        (the powers) instead of [a4, a6, ...]."""
        coeffs = {4: 1.5e-3, 6: -2.7e-5, 8: 3.1e-7}
        out = _quadoa_serialize_aspheric(coeffs)
        # The serializer is allowed to choose dict-keyed or
        # list-of-pairs JSON; what matters is that the VALUES survive.
        if isinstance(out, dict):
            vals = sorted(out.values())
        elif isinstance(out, list):
            # If list, must be [a4, a6, a8] (values), not powers.
            vals = sorted(float(v) for v in out)
        else:  # pragma: no cover -- defensive
            raise AssertionError(f"Unexpected serializer output: {type(out)}")
        expected = sorted(coeffs.values())
        assert np.allclose(vals, expected), (
            "Serializer wrote the wrong numbers.  Pre-v4.11.2 it iterated "
            f"dict keys and wrote [4.0, 6.0, 8.0]; got {out}.")
        # Spot-check none of the dict keys (the powers) leaked into the
        # output as values.
        powers = {4.0, 6.0, 8.0}
        assert not (set(vals) <= powers), (
            "Serialized values look like dict keys (the powers), not the "
            f"coefficients: {vals}.")

    def test_deserialize_round_trip_preserves_dict(self):
        coeffs = {4: 1.5e-3, 6: -2.7e-5, 8: 3.1e-7}
        out = _quadoa_serialize_aspheric(coeffs)
        back = _quadoa_deserialize_aspheric(out)
        assert isinstance(back, dict)
        assert set(back.keys()) == set(coeffs.keys())
        for k, v in coeffs.items():
            assert np.isclose(back[k], v, rtol=1e-12)

    def test_quadoa_file_round_trip_preserves_aspheric_values(self):
        """Round-trip a prescription through export_quadoa_qos /
        load_quadoa_qos and assert the aspheric coefficient values
        survive (not the powers)."""
        coeffs = {4: 1.5e-3, 6: -2.7e-5}
        rx_in = {
            'name': 'test_quadoa',
            'aperture_diameter': 10e-3,
            'surfaces': [
                {'radius': 100e-3, 'conic': 0.0,
                 'aspheric_coeffs': coeffs,
                 'glass_before': 'air', 'glass_after': 'air'},
                {'radius': -100e-3, 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'glass_before': 'air', 'glass_after': 'air'},
            ],
            'thicknesses': [3e-3],
        }
        with tempfile.NamedTemporaryFile(
                suffix='.qos', delete=False) as f:
            path = f.name
        try:
            export_quadoa_qos(rx_in, path, wavelength=587.6e-9)
            rx_out = load_quadoa_qos(path)
        finally:
            os.unlink(path)
        ac = rx_out['surfaces'][0]['aspheric_coeffs']
        # C-IO-9: harmonise type -- canonical form is dict.
        assert isinstance(ac, dict), (
            "Quadoa loader should produce dict-form aspheric_coeffs "
            f"(canonical for the library); got {type(ac).__name__}.")
        assert set(ac.keys()) == set(coeffs.keys())
        for k, v in coeffs.items():
            assert np.isclose(ac[k], v, rtol=1e-9), (
                f"Quadoa round-trip lost alpha_{k}: in={v}, out={ac[k]}.")


# ============================================================================
# C-IO-3  Zemax exporter STOP marker on the correct refracting surface
# ============================================================================


class TestZemaxStopMarkerFolded:
    """In a folded design with coord-breaks/mirrors before the stop,
    ``stop_surface=k`` should mark the k-th refracting surface, NOT
    the k-th global SURF (which the pre-v4.11.2 code did)."""

    def test_stop_on_second_refractive_surface_in_folded_system(self):
        # Two refracting surfaces; ask for STOP on the SECOND one
        # (stop_surface=1).  Place a mirror element before them so
        # the global SURF index of the second refractive is 3, not 2.
        rx = {
            'name': 'stop_test',
            'aperture_diameter': 10e-3,
            'surfaces': [
                {'radius': 100e-3, 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'glass_before': 'air', 'glass_after': 'air'},
                {'radius': -100e-3, 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'glass_before': 'air', 'glass_after': 'air'},
            ],
            'thicknesses': [50e-3],
            'elements': [
                {'element_type': 'mirror', 'radius': float('inf'),
                 'conic': 0.0, 'aspheric_coeffs': None,
                 'semi_diameter': 5e-3, 'surf_num': 1},
                {'element_type': 'surface', 'radius': 100e-3, 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'glass_before': 'air', 'glass_after': 'air',
                 'semi_diameter': 5e-3, 'surf_num': 2},
                {'element_type': 'surface', 'radius': -100e-3, 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'glass_before': 'air', 'glass_after': 'air',
                 'semi_diameter': 5e-3, 'surf_num': 3},
            ],
            'all_thicknesses': [20e-3, 3e-3],
            'coord_breaks': [],
        }
        with tempfile.NamedTemporaryFile(
                suffix='.zmx', delete=False) as f:
            path = f.name
        try:
            # stop_surface=1 means "second refractive surface".
            export_zemax_zmx(rx, path, wavelength=587.6e-9,
                             stop_surface=1)
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        finally:
            os.unlink(path)
        # Find which SURF block contains '  STOP'.
        stop_surf_idx = None
        current = None
        for line in text.splitlines():
            s = line.strip()
            if s.startswith('SURF '):
                try:
                    current = int(s.split()[1])
                except (IndexError, ValueError):
                    current = None
            elif s == 'STOP':
                stop_surf_idx = current
                break
        assert stop_surf_idx is not None, "No STOP marker emitted."
        # The mirror is SURF 1, first refractive is SURF 2, second
        # refractive (the requested stop) is SURF 3.
        assert stop_surf_idx == 3, (
            f"STOP marker landed on SURF {stop_surf_idx} (mirror is SURF 1, "
            "first refractive is SURF 2, second refractive should be SURF 3). "
            "Pre-v4.11.2 the comparison used global surf_counter and placed "
            "STOP on the wrong row.")


# ============================================================================
# C-IO-4  normalize_prescription mirror filter uses element_type
# ============================================================================


class TestNormalizePrescriptionMirrorFilter:

    def test_element_type_mirror_is_filtered_from_surfaces(self):
        """If a prescription only has ``elements`` (no ``surfaces``),
        the canonical form drops mirror entries when synthesising
        ``surfaces`` for apply_real_lens.  Pre-fix this checked
        ``e.get('mirror')`` -- never set by any loader -- so the
        filter was a no-op and mirrors leaked through."""
        elems = [
            {'element_type': 'surface', 'radius': 100e-3, 'conic': 0.0,
             'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'BK7'},
            {'element_type': 'mirror', 'radius': float('inf'),
             'conic': 0.0, 'aspheric_coeffs': None},
            {'element_type': 'surface', 'radius': -100e-3, 'conic': 0.0,
             'aspheric_coeffs': None,
             'glass_before': 'BK7', 'glass_after': 'air'},
        ]
        rx = {
            'name': 'folded',
            'aperture_diameter': 10e-3,
            'elements': elems,
            'all_thicknesses': [3e-3, 50e-3],
        }
        norm = normalize_prescription(rx)
        surfs = norm['surfaces']
        assert all(s.get('element_type') != 'mirror' for s in surfs), (
            "Mirror entries leaked into the synthesised 'surfaces' "
            "list -- normalize_prescription mirror filter is still a "
            "no-op.")
        # Sanity: refractive surfaces survive.
        assert len(surfs) == 2, (
            f"Expected 2 refractive surfaces after filter, got {len(surfs)}.")

    def test_legacy_mirror_flag_still_filtered(self):
        """Belt-and-braces: an entry with the legacy ``'mirror': True``
        flag (not used by current loaders, but historically present)
        should also be filtered."""
        elems = [
            {'element_type': 'surface', 'radius': 100e-3,
             'glass_before': 'air', 'glass_after': 'BK7'},
            {'mirror': True, 'radius': float('inf')},
        ]
        rx = {
            'aperture_diameter': 10e-3,
            'elements': elems,
            'all_thicknesses': [3e-3],
        }
        norm = normalize_prescription(rx)
        assert len(norm['surfaces']) == 1


# ============================================================================
# C-IO-5  Mirror DISZ preserved on Zemax round-trip
# ============================================================================


class TestMirrorDISZRoundTrip:

    def test_zemax_signed_mirror_thickness_unchanged_on_export(self):
        """Pre-fix: ``_export_zemax_zmx_full`` applied a mirror-parity
        sign flip on top of an already-Zemax-signed thickness.  For a
        mirror with DISZ=-10mm the exporter emitted DISZ=+10mm.

        Build a prescription whose ``all_thicknesses`` carries the
        Zemax-signed convention (negative after the mirror) and verify
        the exported file contains the same magnitudes and signs."""
        rx = {
            'name': 'mirror_disz',
            'aperture_diameter': 10e-3,
            'surfaces': [],
            'thicknesses': [],
            'elements': [
                {'element_type': 'mirror', 'radius': 200e-3,
                 'conic': 0.0, 'aspheric_coeffs': None,
                 'semi_diameter': 5e-3, 'surf_num': 1},
                {'element_type': 'mirror', 'radius': -200e-3,
                 'conic': 0.0, 'aspheric_coeffs': None,
                 'semi_diameter': 5e-3, 'surf_num': 2},
            ],
            # Zemax-signed: after the first mirror, distances are
            # negative.  ``all_thicknesses[0]`` = -50 mm = -0.05 m.
            'all_thicknesses': [-0.050, 0.020],
            'coord_breaks': [],
        }
        with tempfile.NamedTemporaryFile(
                suffix='.zmx', delete=False) as f:
            path = f.name
        try:
            _export_zemax_zmx_full(rx, path, wavelength=587.6e-9)
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        finally:
            os.unlink(path)
        # Extract DISZ from the first mirror's SURF block.  The first
        # mirror lives at SURF 1 (SURF 0 is the object plane).
        in_surf_1 = False
        disz_1 = None
        for line in text.splitlines():
            s = line.strip()
            if s.startswith('SURF '):
                in_surf_1 = (s.split()[1] == '1')
                continue
            if in_surf_1 and s.startswith('DISZ'):
                disz_1 = float(s.split()[1])
                break
        assert disz_1 is not None, "Could not find DISZ for SURF 1."
        # all_thicknesses[0] = -0.050 m = -50 mm.  The exporter should
        # NOT flip the sign (pre-fix it would produce +50.0).
        assert np.isclose(disz_1, -50.0, atol=1e-6), (
            f"Mirror DISZ on export = {disz_1} mm, expected -50.0 mm. "
            "Pre-v4.11.2 the mirror-parity flip negated this back to +50.")

    def test_coord_break_disz_not_flipped_after_mirror(self):
        """A coord-break that sits AFTER a mirror should have its
        Zemax-signed DISZ preserved verbatim.  Pre-fix the mirror_count
        parity inverted it on every export."""
        rx = {
            'name': 'cb_after_mirror',
            'aperture_diameter': 10e-3,
            'surfaces': [],
            'thicknesses': [],
            'elements': [
                {'element_type': 'mirror', 'radius': float('inf'),
                 'conic': 0.0, 'aspheric_coeffs': None,
                 'semi_diameter': 5e-3, 'surf_num': 1},
                {'element_type': 'surface', 'radius': float('inf'),
                 'conic': 0.0, 'aspheric_coeffs': None,
                 'glass_before': 'air', 'glass_after': 'air',
                 'semi_diameter': 5e-3, 'surf_num': 3},
            ],
            'all_thicknesses': [-0.030, 0.0],
            'coord_breaks': [
                {'surf_num': 2, 'decenter_x_m': 0.0, 'decenter_y_m': 0.0,
                 'tilt_x_deg': 0.0, 'tilt_y_deg': 45.0,
                 'tilt_z_deg': 0.0, 'order': 0,
                 'thickness_m': -0.015},
            ],
        }
        with tempfile.NamedTemporaryFile(
                suffix='.zmx', delete=False) as f:
            path = f.name
        try:
            _export_zemax_zmx_full(rx, path, wavelength=587.6e-9)
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        finally:
            os.unlink(path)
        # Find COORDBRK surface and its DISZ.
        in_coordbrk = False
        cb_disz = None
        for line in text.splitlines():
            s = line.strip()
            if s.startswith('SURF '):
                in_coordbrk = False
                continue
            if s == 'TYPE COORDBRK':
                in_coordbrk = True
                continue
            if in_coordbrk and s.startswith('DISZ'):
                cb_disz = float(s.split()[1])
                break
        assert cb_disz is not None, "No COORDBRK DISZ found."
        # thickness_m = -0.015 -> -15 mm.  Pre-fix the mirror-parity
        # flip would produce +15.
        assert np.isclose(cb_disz, -15.0, atol=1e-6), (
            f"Coord-break DISZ after mirror = {cb_disz} mm; expected -15. "
            "Pre-v4.11.2 the parity flip negated this.")


# ============================================================================
# HIGH  codegen op -> la (script must import and exec without NameError)
# ============================================================================


class TestCodegenLaAlias:

    def _fake_glass_rx(self):
        """Build a prescription dict with ``elements`` / ``all_thicknesses``
        keys that codegen requires (it expects the loader's schema, not
        the make_singlet schema)."""
        rx = {
            'name': 'codegen_alias_test',
            'aperture_diameter': 10e-3,
            'surfaces': [
                {'radius': 100e-3, 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'glass_before': 'air',
                 'glass_after': '__FAKE_GLASS_FOR_CODEGEN_TEST__'},
                {'radius': -100e-3, 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'glass_before': '__FAKE_GLASS_FOR_CODEGEN_TEST__',
                 'glass_after': 'air'},
            ],
            'thicknesses': [3e-3],
            'elements': [
                {'element_type': 'surface', 'radius': 100e-3, 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'glass_before': 'air',
                 'glass_after': '__FAKE_GLASS_FOR_CODEGEN_TEST__',
                 'semi_diameter': 5e-3, 'surf_num': 1},
                {'element_type': 'surface', 'radius': -100e-3, 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'glass_before': '__FAKE_GLASS_FOR_CODEGEN_TEST__',
                 'glass_after': 'air',
                 'semi_diameter': 5e-3, 'surf_num': 2},
            ],
            'all_thicknesses': [3e-3, 0.0],
            'coord_breaks': [],
        }
        return rx

    def test_generated_script_with_unknown_glass_uses_la_namespace(self):
        """``GLASS_REGISTRY`` is exposed as ``lumenairy.GLASS_REGISTRY``;
        the codegen emits ``import lumenairy as la`` so the right alias
        is ``la.GLASS_REGISTRY`` (pre-v4.11.2 wrote ``op.GLASS_REGISTRY``
        which raises ``NameError`` on exec)."""
        rx = self._fake_glass_rx()
        # generate_simulation_script accepts a prescription directly.
        script = la.io.generate_simulation_script(
            prescription=rx,
            wavelength=1.3e-6, N=64, dx=10e-6,
            source_sigma=2e-3,
            include_plotting=False, include_analysis=False,
        )
        # The fake-glass line must use la., not op.
        assert 'op.GLASS_REGISTRY' not in script, (
            "Codegen still emits ``op.GLASS_REGISTRY`` -- this raises "
            "NameError because the import is ``lumenairy as la``.")
        assert "la.GLASS_REGISTRY['__FAKE_GLASS_FOR_CODEGEN_TEST__']" in script

    def test_generated_script_compiles(self):
        """The generated script must at least parse / compile."""
        rx = self._fake_glass_rx()
        script = la.io.generate_simulation_script(
            prescription=rx,
            wavelength=1.3e-6, N=64, dx=10e-6,
            source_sigma=2e-3,
            include_plotting=False, include_analysis=False,
        )
        compile(script, '<codegen-test>', 'exec')

    def test_generated_script_system_style_uses_la_namespace(self):
        """Both style branches share the same op/la mismatch.  Verify
        the 'system' style emits la.GLASS_REGISTRY as well."""
        rx = self._fake_glass_rx()
        script = la.io.generate_simulation_script(
            prescription=rx, style='system',
            wavelength=1.3e-6, N=64, dx=10e-6,
            source_sigma=2e-3,
            include_plotting=False, include_analysis=False,
        )
        assert 'op.GLASS_REGISTRY' not in script
        assert "la.GLASS_REGISTRY['__FAKE_GLASS_FOR_CODEGEN_TEST__']" in script
        compile(script, '<codegen-test-system>', 'exec')


# ============================================================================
# HIGH  codegen system-list style emits aperture_diameter on mirror
# ============================================================================


class TestCodegenSystemListMirrorAperture:

    def test_mirror_step_includes_aperture_diameter(self):
        """``_generate_system_style`` previously omitted
        ``aperture_diameter`` for ``'mirror'`` elements, so any mirror
        clipping configured upstream silently dropped on script
        emission."""
        from lumenairy.io.codegen import _generate_system_style
        steps = [
            {'type': 'mirror', 'radius': 0.2,
             'conic': 0.0, 'aperture_diameter': 0.025,
             'comment': 'fold mirror', 'surf_num': 1},
        ]
        script = _generate_system_style(
            steps=steps, wavelength=1.3e-6, N=64, dx=10e-6,
            source_sigma=2e-3, aperture=25e-3,
            sys_name='m_test', glasses_used=set(),
            include_plotting=False, include_analysis=False,
            header_comment=None,
        )
        assert "'aperture_diameter'" in script, (
            "Mirror element in system-list style is missing the "
            "aperture_diameter key (regression of v4.11.2 fix).")
        # Spot-check the value is in there.
        assert '2.5' in script or '2.50000' in script or '0.025' in script


# ============================================================================
# HIGH  aspheric_coeffs type harmonisation (loaders return dict or None)
# ============================================================================


class TestAsphericCoeffsTypeHarmonisation:

    def test_zemax_loader_returns_dict(self):
        zmx_text = _minimal_evenasph_zmx(1e-5, 0.0)
        path = _write_zmx(zmx_text)
        try:
            rx = load_zemax_zmx(path)
        finally:
            os.unlink(path)
        ac = rx['surfaces'][0]['aspheric_coeffs']
        assert isinstance(ac, dict), (
            f"Zemax loader should return dict aspheric_coeffs, "
            f"got {type(ac).__name__}.")

    def test_quadoa_loader_returns_dict(self):
        """Pre-fix the Quadoa loader returned a list."""
        coeffs = {4: 1.5e-3, 6: -2.7e-5}
        rx_in = {
            'name': 't', 'aperture_diameter': 10e-3,
            'surfaces': [
                {'radius': 100e-3, 'conic': 0.0,
                 'aspheric_coeffs': coeffs,
                 'glass_before': 'air', 'glass_after': 'air'},
                {'radius': -100e-3, 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'glass_before': 'air', 'glass_after': 'air'},
            ],
            'thicknesses': [3e-3],
        }
        with tempfile.NamedTemporaryFile(
                suffix='.qos', delete=False) as f:
            path = f.name
        try:
            export_quadoa_qos(rx_in, path, wavelength=587.6e-9)
            rx_out = load_quadoa_qos(path)
        finally:
            os.unlink(path)
        ac = rx_out['surfaces'][0]['aspheric_coeffs']
        assert isinstance(ac, dict), (
            "Quadoa loader should produce dict (canonical), "
            f"got {type(ac).__name__}.")

    def test_loader_no_aspheric_returns_none(self):
        """Make-singlet style: no aspheric -> aspheric_coeffs is None."""
        rx = la.make_singlet(
            R1=100e-3, R2=-100e-3, d=3e-3, glass='N-BK7',
            aperture=10e-3,
        )
        for s in rx['surfaces']:
            assert s['aspheric_coeffs'] is None
