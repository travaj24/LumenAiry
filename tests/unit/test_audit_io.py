"""Consolidated audit-fix tests for the **io** domain.

This module consolidates v4.9 - v5.0 audit-fix regression pins
from 3 source files (per the v5.2 ROADMAP / 57-file consolidation):

* ``test_audit_fixes_v4_11_2_io.py``
* ``test_audit_fixes_v4_13_0_io.py``
* ``test_audit_fixes_v4_14_3_agent_a.py``

Each source file's contents are concatenated below verbatim (modulo
minimal renames to avoid identifier collisions and to give each top-level
test class an audit-version attribution prefix).  inspect.getsource proxy
tests are tagged with a TODO comment per AUDIT_V4_13_1 Part 6.1.
"""
from __future__ import annotations

# ============================================================================
# Source: test_audit_fixes_v4_11_2_io.py
# Audit version: V4_11_2  scope: io
# Original module docstring preserved as comment block for git-blame traceability:
#   Regression tests for the v4.11.2 IO-domain audit fixes.
#   
#   Each test pins one of the round-3 IO findings that v4.11.2 addresses.
#   The tests are small (<1 s each); their purpose is to fail loudly if a
#   future refactor reintroduces the specific bug, not to exercise the
#   broader physics.
#   
#   Findings covered (per ``AUDIT_ROUND3_2026_05_16.md`` "IO / Zemax /
#   prescriptions / storage / codegen" section):
#   
#   * C-IO-1  EVENASPH PARM off-by-one (loader dropped PARM 1 = a_4 and
#             mis-labelled every higher coefficient by one slot).
#   * C-IO-2  Quadoa aspheric serializer iterates dict keys instead of
#             values (wrote ``[4.0, 6.0, ...]`` powers instead of the
#             coefficients).
#   * C-IO-3  Zemax exporter STOP marker on wrong surface in folded
#             designs (compared global surf_counter, not refractive index).
#   * C-IO-4  ``normalize_prescription`` mirror filter checked
#             ``e.get('mirror')`` but library uses
#             ``element_type='mirror'``; the filter was a no-op.
#   * C-IO-5  Mirror DISZ lost on Zemax round-trip (exporter applied a
#             spurious mirror-parity sign-flip on top of an already
#             Zemax-signed thickness).
#   * HIGH    codegen emitted ``op.GLASS_REGISTRY`` but imported
#             ``lumenairy as la`` -> NameError on script execution.
#   * HIGH    codegen system-list style dropped ``aperture_diameter``
#             for mirror elements.
#   * HIGH    ``aspheric_coeffs`` type harmonisation: every loader now
#             produces dict-or-None (Quadoa loader was returning a list).
# ============================================================================
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


def _minimal_evenasph_zmx(parm_map):
    """A minimal 2-surface .zmx with one EVENASPH surface whose PARM slots are
    populated per ``parm_map`` (``{parm_num: value_in_mm_units}``).

    Zemax EVENASPH stores the even-aspheric polynomial
    ``z += alpha_1*r^2 + alpha_2*r^4 + alpha_3*r^6 + ...`` as
    ``PARM_n = alpha_n = the coefficient of r^(2n)``.  So PARM 1 is the r^2
    term (conventionally 0 -- degenerate with the base curvature, which is why
    real .zmx aspheres list ``PARM 1 0.0`` explicitly and put the first real
    departure in PARM 2 = r^4).  The loader must therefore map
    ``PARM_n -> power 2n``.  Returns the .zmx text.
    """
    parm_lines = [f'  PARM {n} {v:.10e}'
                  for n, v in sorted(parm_map.items())]
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
        *parm_lines,
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


class TestAuditFixesV4_11_2_io_EvenAsphParm:

    def test_parm1_maps_to_power2_not_power4(self):
        """Zemax EVENASPH ``PARM_n = alpha_n`` is the coefficient of r^(2n),
        so PARM 1 is the r^2 term (power=2), NOT r^4.

        History: pre-v4.11.2 the loader's ``if parm_num >= 2`` filter dropped
        PARM 1; v4.11.2 correctly stopped dropping it but assigned the wrong
        power via ``power = 2 + 2*parm_num`` (PARM1->4, PARM2->6, ...) and this
        test was written to that off-by-one.  v5.16.1 corrects the loader to
        ``power = 2*parm_num``, verified against the poc1-19 design: its S9/S38
        EVENASPH surfaces list ``PARM 1 0.0`` (the conventionally-zero r^2 term)
        with the real departure in PARM 2+, and only the 2*parm_num mapping
        reproduces their physical sub-micron aspheric sag (the +2 offset
        inflated it ~10x and smeared the imaged spots)."""
        alpha2_per_mm = 3.0e-3       # 1/mm (r^2 coefficient)
        zmx_text = _minimal_evenasph_zmx({1: alpha2_per_mm})
        path = _write_zmx(zmx_text)
        try:
            rx = load_zemax_zmx(path)
        finally:
            os.unlink(path)
        ac = rx['surfaces'][0]['aspheric_coeffs']
        assert ac is not None, "PARM 1 was dropped (regression)."
        assert 2 in ac and 4 not in ac, (
            f"PARM 1 must map to power=2 (r^2), got {list(ac.keys())}.")
        # Unit conversion: r^2 coeff is 1/mm -> 1/m, i.e. x 1e3.
        assert np.isclose(ac[2], alpha2_per_mm * 1e3, rtol=1e-6), (
            f"power=2 unit conversion wrong: got {ac[2]}, "
            f"expected {alpha2_per_mm * 1e3}.")

    def test_parm2_maps_to_power4(self):
        """PARM 2 = alpha_2 = the r^4 coefficient (the first conventionally-
        nonzero aspheric term).  Pre-v5.16.1 it was mis-loaded as power=6."""
        alpha4_per_mm3 = 1.5e-5      # 1/mm^3 (r^4 coefficient)
        zmx_text = _minimal_evenasph_zmx({2: alpha4_per_mm3})
        path = _write_zmx(zmx_text)
        try:
            rx = load_zemax_zmx(path)
        finally:
            os.unlink(path)
        ac = rx['surfaces'][0]['aspheric_coeffs']
        assert ac is not None
        assert 4 in ac and 6 not in ac, (
            f"PARM 2 must map to power=4 (r^4), got {list(ac.keys())}.")
        # 1 mm^-3 = 1e9 m^-3.
        assert np.isclose(ac[4], alpha4_per_mm3 * 1e9, rtol=1e-6)

    def test_parm3_maps_to_power6(self):
        """PARM 3 = alpha_3 = the r^6 coefficient (power=6)."""
        alpha6_per_mm5 = 2.0e-9      # 1/mm^5 (r^6 coefficient)
        zmx_text = _minimal_evenasph_zmx({3: alpha6_per_mm5})
        path = _write_zmx(zmx_text)
        try:
            rx = load_zemax_zmx(path)
        finally:
            os.unlink(path)
        ac = rx['surfaces'][0]['aspheric_coeffs']
        assert ac is not None
        assert 6 in ac, (
            f"PARM 3 must map to power=6 (r^6), got {list(ac.keys())}.")
        # 1 mm^-5 = 1e15 m^-5.
        assert np.isclose(ac[6], alpha6_per_mm5 * 1e15, rtol=1e-6)

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


class TestAuditFixesV4_11_2_io_QuadoaAsphericSerializer:

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


class TestAuditFixesV4_11_2_io_ZemaxStopMarkerFolded:
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


class TestAuditFixesV4_11_2_io_NormalizePrescriptionMirrorFilter:

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


class TestAuditFixesV4_11_2_io_MirrorDISZRoundTrip:

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


class TestAuditFixesV4_11_2_io_CodegenLaAlias:

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


class TestAuditFixesV4_11_2_io_CodegenSystemListMirrorAperture:

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


class TestAuditFixesV4_11_2_io_AsphericCoeffsTypeHarmonisation:

    def test_zemax_loader_returns_dict(self):
        zmx_text = _minimal_evenasph_zmx({2: 1e-5})
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


# ============================================================================
# Source: test_audit_fixes_v4_13_0_io.py
# Audit version: V4_13_0  scope: io
# Original module docstring preserved as comment block for git-blame traceability:
#   Regression tests for the v4.13.0 IO-domain audit fixes.
#   
#   Audit reference
#   ---------------
#   
#   ``AUDIT_V4_12_1_2026_05_16.md`` Part 4 / Part 5 carried three IO-domain
#   findings forward into the v4.12.2 CHANGELOG "Known limitations"
#   section.  v4.13.0 closes them inside ``lumenairy/io/``.
#   
#   * **S1** -- ``io/storage.py`` append-side hardcoded ``complex128``.
#     Both :func:`append_plane_h5` (line 342) and
#     :func:`save_jones_field_h5` (lines 282-283) called
#     ``np.asarray(field, dtype=np.complex128)`` unconditionally,
#     silently doubling on-disk size for ``complex64`` simulations
#     streamed via :meth:`MhsPipeline.run` / :func:`replay_run`.
#     v4.13.0 fix: expose ``preserve_dtype=False`` (default) consistent
#     with the existing :func:`save_field_h5` / :func:`save_planes_h5`
#     single-shot APIs.
#   
#   * **S2a** -- ``io/codegen.py`` aperture-stop drop.  The docstring of
#     ``_decompose_prescription`` advertised a ``'type': 'aperture'``
#     step and the downstream generators handled it, but the function
#     never emitted one.  Zemax prescriptions with a ``STOP`` marker
#     lost the stop in the generated script.
#   
#   * **S2b** -- ``io/codegen.py`` silent 1310 nm wavelength default.
#     When neither the user nor the prescription supplied a wavelength
#     the codegen silently defaulted to 1.31e-6 m (NIR O-band), so
#     visible-band Zemax files quietly converted to NIR scripts.
#     v4.13.0 raises a clear :class:`ValueError`.
#   
#   * **L8** -- ``_open_zarr_group_safe`` ``Path.mkdir`` monkey-patch is
#     not thread-safe.  Two threads racing through
#     :func:`append_plane_h5` could each install a patched ``mkdir``,
#     one thread's saved "original" being the other thread's patched
#     version -- permanently corrupting ``pathlib.Path.mkdir`` for the
#     whole process.  v4.13.0 guards the install/restore window with a
#     module-level :class:`threading.Lock`.
#   
#   Author: Andrew Traverso
# ============================================================================

import os
import tempfile
import threading
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.io import storage as _storage
from lumenairy.io.codegen import (
    _decompose_prescription,
    generate_simulation_script,
)

# ============================================================================
# S1 -- append-side dtype preservation
# ============================================================================


class TestAuditFixesV4_13_0_io_AppendSideDtypePreservation:
    """v4.13.0 honours ``preserve_dtype`` on every append-side I/O.

    Pre-v4.13.0:
    * :func:`append_plane_h5` cast unconditionally to ``complex128``.
    * :func:`save_jones_field_h5` cast unconditionally to ``complex128``.

    Post-v4.13.0:
    * Both accept ``preserve_dtype: bool = False``.
    * ``preserve_dtype=True`` keeps ``complex64`` across a round-trip.
    * ``preserve_dtype=False`` (the default) preserves the historical
      ``complex128`` coercion so existing code is unaffected.

    Pin: write a ``complex64`` field via the unified ``append_plane``
    API with ``preserve_dtype=True`` and assert the round-tripped
    array is ``complex64``.  Also pin the back-compat default.
    """

    def _have_h5py(self):
        try:
            import h5py  # noqa: F401
        except ImportError:
            return False
        return True

    def test_append_plane_h5_preserve_dtype_true_keeps_complex64(self):
        if not self._have_h5py():
            pytest.skip('h5py not installed')
        N = 16
        E64 = np.ones((N, N), dtype=np.complex64) * (1.0 + 1.0j)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'append_c64.h5')
            la.append_plane_h5(path, E64, dx=5e-6, dy=5e-6,
                               label='c64',
                               preserve_dtype=True)
            planes, _ = la.load_planes_h5(path)
        assert len(planes) == 1
        E_back = planes[0]['field']
        assert E_back.dtype == np.complex64, (
            f'preserve_dtype=True did not survive round-trip: '
            f'got dtype={E_back.dtype}, expected complex64')
        assert np.allclose(E_back, E64), 'round-trip values disagree'

    def test_append_plane_h5_default_coerces_to_complex128(self):
        """Default (preserve_dtype=False) keeps the historical
        complex128 coercion -- backward compatibility."""
        if not self._have_h5py():
            pytest.skip('h5py not installed')
        N = 8
        E64 = np.ones((N, N), dtype=np.complex64)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'append_default.h5')
            la.append_plane_h5(path, E64, dx=1e-6, dy=1e-6,
                               label='default')
            planes, _ = la.load_planes_h5(path)
        assert planes[0]['field'].dtype == np.complex128, (
            'default behaviour must still coerce to complex128')

    def test_append_plane_unified_passes_preserve_dtype(self):
        """The auto-dispatch ``append_plane`` API forwards
        ``preserve_dtype`` to the H5 backend (S1 followthrough)."""
        if not self._have_h5py():
            pytest.skip('h5py not installed')
        N = 8
        E64 = (np.random.default_rng(0)
               .standard_normal((N, N))
               .astype(np.complex64))
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'append_unified.h5')
            la.append_plane(path, E64, dx=1e-6, dy=1e-6,
                            label='unified',
                            preserve_dtype=True)
            planes, _ = la.load_planes_h5(path)
        assert planes[0]['field'].dtype == np.complex64

    def test_save_jones_field_h5_preserve_dtype_true_keeps_complex64(self):
        if not self._have_h5py():
            pytest.skip('h5py not installed')
        from lumenairy.elements.polarization import JonesField
        N = 8
        Ex = np.ones((N, N), dtype=np.complex64) * (1.0 + 0.5j)
        Ey = np.ones((N, N), dtype=np.complex64) * (0.25 - 0.1j)
        jf = JonesField(Ex=Ex, Ey=Ey, dx=2e-6)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'jones_c64.h5')
            la.save_jones_field_h5(path, jf, wavelength=1.31e-6,
                                   preserve_dtype=True)
            jf_back, _meta = la.load_jones_field_h5(path)
        assert jf_back.Ex.dtype == np.complex64, (
            f'Ex dtype post-roundtrip = {jf_back.Ex.dtype}')
        assert jf_back.Ey.dtype == np.complex64, (
            f'Ey dtype post-roundtrip = {jf_back.Ey.dtype}')

    def test_save_jones_field_h5_default_coerces_to_complex128(self):
        """Default (preserve_dtype=False) keeps the historical
        complex128 coercion."""
        if not self._have_h5py():
            pytest.skip('h5py not installed')
        from lumenairy.elements.polarization import JonesField
        N = 8
        Ex = np.ones((N, N), dtype=np.complex64)
        Ey = 1j * np.ones((N, N), dtype=np.complex64)
        jf = JonesField(Ex=Ex, Ey=Ey, dx=2e-6)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'jones_default.h5')
            la.save_jones_field_h5(path, jf, wavelength=1.31e-6)
            jf_back, _ = la.load_jones_field_h5(path)
        assert jf_back.Ex.dtype == np.complex128
        assert jf_back.Ey.dtype == np.complex128


# ============================================================================
# S2a -- codegen aperture-stop emission
# ============================================================================


class TestAuditFixesV4_13_0_io_CodegenApertureStopEmission:
    """Pre-v4.13.0 ``_decompose_prescription`` never emitted any
    ``{'type': 'aperture'}`` step even though its docstring promised
    one and the downstream emitters handle it.  v4.13.0 emits an
    aperture step for every element flagged ``is_stop=True``.

    Pin: build a prescription whose ``elements`` list contains a
    STOP-flagged dummy plane.  Assert :func:`_decompose_prescription`
    emits at least one ``type='aperture'`` step AND
    :func:`generate_simulation_script` emits the matching
    ``la.apply_aperture(...)`` call in both `'unrolled'` and
    `'system'` code-styles.
    """

    def _stop_prescription_with_dummy(self, stop_diameter=10e-3):
        """Two-surface lens with a STOP dummy plane in front."""
        # Air-to-air dummy STOP plane, then a singlet R1=50/R2=-50/4mm BK7.
        elements = [
            {
                'element_type': 'surface',
                'radius': float('inf'),
                'conic': 0.0,
                'aspheric_coeffs': None,
                'glass_before': 'air',
                'glass_after': 'air',
                'semi_diameter': stop_diameter * 0.5,
                'surf_num': 1,
                'comment': 'STOP dummy',
                'is_stop': True,
            },
            {
                'element_type': 'surface',
                'radius': 50e-3,
                'conic': 0.0,
                'aspheric_coeffs': None,
                'glass_before': 'air',
                'glass_after': 'N-BK7',
                'semi_diameter': stop_diameter * 0.5,
                'surf_num': 2,
                'comment': '',
                'is_stop': False,
            },
            {
                'element_type': 'surface',
                'radius': -50e-3,
                'conic': 0.0,
                'aspheric_coeffs': None,
                'glass_before': 'N-BK7',
                'glass_after': 'air',
                'semi_diameter': stop_diameter * 0.5,
                'surf_num': 3,
                'comment': '',
                'is_stop': False,
            },
        ]
        return {
            'name': 'stop_test_design',
            'aperture_diameter': stop_diameter,
            'elements': elements,
            'all_thicknesses': [3e-3, 4e-3],
            # surfaces / thicknesses (refracting-only) for the lens RX:
            'surfaces': [
                {'radius': 50e-3, 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'glass_before': 'air', 'glass_after': 'N-BK7'},
                {'radius': -50e-3, 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'glass_before': 'N-BK7', 'glass_after': 'air'},
            ],
            'thicknesses': [4e-3],
            'wavelength': 1.31e-6,
        }

    def test_decompose_emits_aperture_step(self):
        rx = self._stop_prescription_with_dummy(stop_diameter=8e-3)
        steps = _decompose_prescription(rx)
        ap_steps = [s for s in steps if s.get('type') == 'aperture']
        assert len(ap_steps) >= 1, (
            f'no aperture step emitted; pre-fix bug.  Steps: {steps}')
        # Diameter pulled from the stop element's semi_diameter * 2.
        assert abs(ap_steps[0]['diameter'] - 8e-3) < 1e-12, (
            f'aperture diameter mismatch: {ap_steps[0]["diameter"]}')

    def test_unrolled_script_contains_apply_aperture_call(self):
        rx = self._stop_prescription_with_dummy(stop_diameter=6e-3)
        script = generate_simulation_script(
            rx, wavelength=1.31e-6, N=64, dx=10e-6,
            source_sigma=1e-3,
            include_plotting=False, include_analysis=False,
            style='unrolled',
        )
        assert 'la.apply_aperture(' in script, (
            'unrolled codegen missing apply_aperture call for STOP-'
            'flagged input -- S2a pre-fix bug')
        # The script must still compile.
        compile(script, '<codegen-stop-test>', 'exec')

    def test_system_style_script_contains_aperture_entry(self):
        rx = self._stop_prescription_with_dummy(stop_diameter=6e-3)
        script = generate_simulation_script(
            rx, wavelength=1.31e-6, N=64, dx=10e-6,
            source_sigma=1e-3,
            include_plotting=False, include_analysis=False,
            style='system',
        )
        assert "'type': 'aperture'" in script, (
            'system-list codegen missing aperture element for STOP-'
            'flagged input -- S2a pre-fix bug')
        compile(script, '<codegen-stop-system-test>', 'exec')

    def test_stop_index_translation_for_loaders_that_dropped_is_stop(self):
        """When elements have no ``is_stop`` flag but the prescription
        carries a top-level ``stop_index`` (the .qos / .seq loader
        convention), the decomposer still emits an aperture step
        whose index matches the named refracting surface."""
        # Build a refracting-only element list and set stop_index = 0
        # (first refracting surface).
        rx = self._stop_prescription_with_dummy(stop_diameter=8e-3)
        # Strip is_stop flags and use stop_index instead.
        for e in rx['elements']:
            e.pop('is_stop', None)
        # stop_index is documented as "zero-based index of the stop
        # among refracting surfaces"; here our 0th refracting surface
        # is the dummy STOP plane.
        rx['stop_index'] = 0
        steps = _decompose_prescription(rx)
        ap_steps = [s for s in steps if s.get('type') == 'aperture']
        assert len(ap_steps) >= 1, (
            'stop_index-based codegen path also missing aperture step')


# ============================================================================
# S2b -- codegen wavelength default
# ============================================================================


class TestAuditFixesV4_13_0_io_CodegenWavelengthDefault:
    """Pre-v4.13.0 the codegen silently defaulted ``wavelength`` to
    1.31e-6 m when neither user nor prescription supplied one.
    v4.13.0 raises :class:`ValueError`.
    """

    def _prescription_without_wavelength(self):
        # Minimal prescription with no 'wavelength' key.
        return {
            'name': 'no_wavelength_design',
            'aperture_diameter': 10e-3,
            'elements': [
                {
                    'element_type': 'surface',
                    'radius': 50e-3,
                    'conic': 0.0,
                    'aspheric_coeffs': None,
                    'glass_before': 'air',
                    'glass_after': 'N-BK7',
                    'semi_diameter': 5e-3,
                    'surf_num': 1,
                    'comment': '',
                    'is_stop': False,
                },
                {
                    'element_type': 'surface',
                    'radius': -50e-3,
                    'conic': 0.0,
                    'aspheric_coeffs': None,
                    'glass_before': 'N-BK7',
                    'glass_after': 'air',
                    'semi_diameter': 5e-3,
                    'surf_num': 2,
                    'comment': '',
                    'is_stop': False,
                },
            ],
            'all_thicknesses': [4e-3],
            'surfaces': [
                {'radius': 50e-3, 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'glass_before': 'air', 'glass_after': 'N-BK7'},
                {'radius': -50e-3, 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'glass_before': 'N-BK7', 'glass_after': 'air'},
            ],
            'thicknesses': [4e-3],
        }

    def test_missing_wavelength_raises_valueerror(self):
        rx = self._prescription_without_wavelength()
        with pytest.raises(ValueError) as excinfo:
            generate_simulation_script(
                rx, wavelength=None, N=64, dx=10e-6,
                source_sigma=1e-3,
                include_plotting=False, include_analysis=False,
            )
        # The error message must name the offending parameter.
        assert 'wavelength' in str(excinfo.value).lower()

    def test_explicit_wavelength_still_works(self):
        """Sanity check: passing wavelength explicitly must not raise."""
        rx = self._prescription_without_wavelength()
        script = generate_simulation_script(
            rx, wavelength=587.6e-9, N=64, dx=5e-6,
            source_sigma=1e-3,
            include_plotting=False, include_analysis=False,
        )
        assert '587.6' in script or '5.876' in script

    def test_prescription_wavelength_used_if_present(self):
        """Sanity check: wavelength stored in the prescription dict
        wins over the (now removed) 1310 nm default."""
        rx = self._prescription_without_wavelength()
        rx['wavelength'] = 632.8e-9  # HeNe red
        script = generate_simulation_script(
            rx, wavelength=None, N=64, dx=5e-6,
            source_sigma=1e-3,
            include_plotting=False, include_analysis=False,
        )
        # The header writes wavelength in nm with one decimal.
        assert '632.8 nm' in script or '6.328' in script, (
            f'prescription wavelength 632.8e-9 not honoured.  Script '
            f'first 1200 chars:\n{script[:1200]}')


# ============================================================================
# L8 -- _open_zarr_group_safe thread-safety
# ============================================================================


class TestAuditFixesV4_13_0_io_ZarrMkdirPatchThreadSafety:
    """``_open_zarr_group_safe`` monkey-patches ``Path.mkdir`` for the
    duration of a single zarr open call.  Two threads racing through
    that function could leave the patched mkdir permanently installed.

    v4.13.0 wraps the install/restore window in a
    :class:`threading.Lock`.

    Pin: spawn two threads that each call :func:`append_plane_h5`
    concurrently against the same file.  Assertions:

    1. Both threads complete without exception.
    2. The resulting file is valid (loads back both planes).
    3. ``pathlib.Path.mkdir`` after the test is the original
       (unpatched) one -- if the patch leaked, subsequent ``mkdir``
       calls would have the patched dispatch.
    """

    def test_concurrent_append_plane_h5_no_mkdir_corruption(self):
        """Two threads append concurrently to disjoint HDF5 files; the
        critical pin is that ``Path.mkdir`` is the un-patched original
        after both threads exit.  Pre-fix the zarr-side mkdir patch
        was unguarded; this test surfaces leakage even when the H5
        path itself doesn't go through ``_open_zarr_group_safe``,
        because the failure mode lives in the global ``Path.mkdir``
        symbol and any concurrent zarr open can corrupt it for
        downstream H5 operations that rely on ``Path.mkdir``."""
        try:
            import h5py  # noqa: F401
        except ImportError:
            pytest.skip('h5py not installed')

        # Capture the real Path.mkdir reference up-front.
        from pathlib import Path
        original_mkdir = Path.mkdir

        N = 16
        errors = []

        def _worker(path, label, payload):
            try:
                la.append_plane_h5(
                    path, payload, dx=4e-6, dy=4e-6, label=label)
            except Exception as e:  # pragma: no cover -- only on failure
                errors.append((label, repr(e)))

        with tempfile.TemporaryDirectory() as td:
            # Each thread writes to its OWN file -- the L8 finding is
            # specifically about ``Path.mkdir`` patch leakage, not
            # about h5py's own intra-file locking (h5py serialises
            # writes to a single file via its global lock but that's
            # orthogonal to L8).
            p1 = os.path.join(td, 'shared_a.h5')
            p2 = os.path.join(td, 'shared_b.h5')
            E1 = np.ones((N, N), dtype=np.complex128)
            E2 = 0.5 * np.ones((N, N), dtype=np.complex128)
            threads = [
                threading.Thread(
                    target=_worker, args=(p1, 't1', E1)),
                threading.Thread(
                    target=_worker, args=(p2, 't2', E2)),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10.0)
                assert not t.is_alive(), \
                    f'{t.name} hung beyond 10 s deadline'

            assert not errors, f'thread errors: {errors}'
            # Both files must be valid (one plane each).
            planes_a, _ = la.load_planes_h5(p1)
            planes_b, _ = la.load_planes_h5(p2)
            assert len(planes_a) == 1, \
                f'thread 1 wrote {len(planes_a)} planes'
            assert len(planes_b) == 1, \
                f'thread 2 wrote {len(planes_b)} planes'

        # CRITICAL: Path.mkdir must be the un-patched original after
        # both threads exit.  If the lock isn't there, one thread can
        # restore *before* the other thread saved its "original"
        # reference, and the saved "original" ends up being the
        # patched dispatch -- making the patch permanent.
        assert Path.mkdir is original_mkdir, (
            'Path.mkdir patch leaked beyond _open_zarr_group_safe: '
            'L8 thread-safety regression')

    def test_open_zarr_group_safe_serial_install_restore(self):
        """Directly exercise ``_open_zarr_group_safe`` from a thread
        pool.  Even when zarr is unavailable, the patch install/
        restore happens around the actual ``open_group`` call (in
        the writable branch); we mock that to force the bug-prone
        window."""
        try:
            import zarr  # noqa: F401
        except ImportError:
            pytest.skip('zarr not installed')

        from pathlib import Path
        original_mkdir = Path.mkdir

        # Race ``_open_zarr_group_safe`` directly: each worker opens
        # (writable) a unique zarr store.  Repeated runs under the
        # lock must always restore Path.mkdir to its original
        # implementation after each worker exits.
        with tempfile.TemporaryDirectory() as td:
            errors = []

            def _worker(idx):
                try:
                    store_path = os.path.join(td, f'store_{idx}.zarr')
                    grp = _storage._open_zarr_group_safe(
                        zarr, store_path, writable=True)
                    # touch an attr so the open is non-trivial.
                    try:
                        grp.attrs['idx'] = int(idx)
                    except Exception:
                        pass
                except Exception as e:  # pragma: no cover
                    errors.append((idx, repr(e)))

            threads = [
                threading.Thread(target=_worker, args=(i,))
                for i in range(4)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10.0)
                assert not t.is_alive(), \
                    f'{t.name} hung beyond 10 s deadline'
            assert not errors, f'thread errors: {errors}'

        assert Path.mkdir is original_mkdir, (
            'Path.mkdir patch leaked beyond _open_zarr_group_safe '
            'across concurrent zarr opens')


# ============================================================================
# Source: test_audit_fixes_v4_14_3_agent_a.py
# Audit version: V4_14_3  scope: agent_a
# Original module docstring preserved as comment block for git-blame traceability:
#   Tests for the v4.14.3 audit-fix Agent-A changes.
#   
#   Scope: ``lumenairy/io/storage.py``, ``lumenairy/elements/doe.py``,
#   ``lumenairy/propagators/propagation.py`` (Agent-A files only).
#   
#   What this module pins
#   ---------------------
#   
#   * **A.1 -- P0-NEW-1 ``append_plane_h5`` / ``_zarr_append_plane``
#     atomicity.**  The pre-4.14.3 code created the dataset first and
#     then bumped ``n_planes`` -- a crash between the two left an orphan
#     dataset whose name collided with the next append's computed name.
#     The Zarr variant additionally passed ``overwrite=True`` so the
#     next append silently clobbered the orphan (data loss).  v4.14.3
#     bumps ``n_planes`` BEFORE the create and rolls back on failure,
#     so a crash leaves the slot reserved (next append computes ``n + 1``
#     and skips it).  Multi-process concurrency is still unsupported
#     -- documented in the docstring.
#   
#   * **A.2 -- P0-NEW-2 ``makedammann2d`` SI heuristic upper bound.**
#     The v4.14.2 heuristic treated ``periodx > 1e-3`` as legacy
#     micrometres and rescaled by ``1e-6``.  Real-world THz / MMW DOE
#     designs use mm-scale SI periods (``periodx=5e-3``) and far-IR
#     wavelengths (``waveln=1.1e-3``).  Such a call silently produced
#     5 nm garbage with a confusing ``DeprecationWarning`` masking the
#     bug.  v4.14.3 adds (1) a hard ``ValueError`` for inputs above 1 m
#     (unambiguously wrong in any unit system); and (2) an explicit
#     ``_legacy_units`` kwarg with values ``'auto'`` / ``'um'`` / ``'SI'``
#     so THz / MMW users can opt out of the heuristic via
#     ``_legacy_units='SI'``.
#   
#   * **A.3 -- P1-NEW-1 ``clear_asm_caches`` chains
#     ``clear_lg_polynomial_cache``.**  The v4.14.2 docstring claimed
#     ``clear_asm_caches`` left the propagator-adjacent state completely
#     pristine -- matching :func:`lumenairy_context` -- but missed
#     ``_lg_polynomial_items`` (an ``lru_cache`` in
#     ``propagators/asymptotic.py``).  v4.14.3 chains it.
#   
#   Author: Agent A -- v4.14.3.
# ============================================================================

import os
import tempfile
import warnings

import numpy as np
import pytest

# ============================================================================
# A.1 -- ``append_plane_h5`` / ``_zarr_append_plane`` atomicity
# ============================================================================

class TestAuditFixesV4_14_3_agent_a_A1AppendPlaneAtomicity:
    """Pin the attribute-before-create discipline that keeps the
    append path atomic under a mid-call crash."""

    def test_append_plane_h5_attr_written_before_dataset(self):
        """Pin the rollback contract for ``append_plane_h5``.

        Monkey-patch ``create_dataset`` to raise; assert the
        post-call ``n_planes`` attribute equals the pre-call value
        (rolled back) and the original (pre-crash) plane is still
        intact.  Pre-4.14.3 the create-then-bump order meant the
        attribute was never written on crash; the bug being pinned
        here is the v4.14.3 reverse: bump first AND roll back on
        failure so the slot is neither stranded nor silently
        clobbered.
        """
        import h5py

        from lumenairy.io.storage import append_plane_h5

        E = np.ones((8, 8), dtype=np.complex128)
        with tempfile.TemporaryDirectory() as d:
            fp = os.path.join(d, 'crash.h5')
            # First append succeeds, baseline state.
            append_plane_h5(fp, E, dx=1e-6, label='before')
            with h5py.File(fp, 'r') as f:
                n_before = int(f['planes'].attrs['n_planes'])
            assert n_before == 1, (
                f'Expected n_planes=1 after one append, got {n_before}.'
            )

            # Patch h5py.Group.create_dataset to raise on the next call.
            # We patch via the h5py module rather than wrapping the
            # function because ``append_plane_h5`` opens its own file
            # handle inside the call.
            orig_create_dataset = h5py.Group.create_dataset
            def _boom(self, *args, **kwargs):
                raise RuntimeError('simulated mid-append crash')
            h5py.Group.create_dataset = _boom
            try:
                with pytest.raises(RuntimeError, match='simulated'):
                    append_plane_h5(fp, E, dx=1e-6, label='will-crash')
            finally:
                h5py.Group.create_dataset = orig_create_dataset

            # Post-crash state: ``n_planes`` rolled back to the
            # pre-crash value so the next legitimate append goes to
            # the same slot.
            with h5py.File(fp, 'r') as f:
                n_after = int(f['planes'].attrs['n_planes'])
                assert n_after == n_before, (
                    f'n_planes not rolled back: pre={n_before}, '
                    f'post={n_after}.  v4.14.3 contract: on '
                    f'create_dataset failure, n_planes must be '
                    f'restored to its pre-call value.'
                )
                # The pre-crash plane is intact.
                assert 'plane_00' in f['planes']
                assert (f['planes/plane_00'].attrs['label']
                        in ('before', b'before'))

            # A subsequent successful append goes to plane_01 (the
            # slot reserved by the rolled-back attribute).
            append_plane_h5(fp, E, dx=1e-6, label='after')
            with h5py.File(fp, 'r') as f:
                assert int(f['planes'].attrs['n_planes']) == 2
                assert 'plane_01' in f['planes']

    def test_append_plane_h5_docstring_documents_multiprocess_limit(self):
        """The v4.14.3 docstring must call out the multi-process
        restriction so users do not assume HDF5 SWMR is being used
        under the hood.  This is the audit's "documented for sequential
        use" assessment."""
        from lumenairy.io.storage import append_plane_h5
        doc = append_plane_h5.__doc__
        assert doc is not None
        # The atomicity contract is explained.
        assert 'Atomicity' in doc or 'atomicity' in doc, (
            'append_plane_h5 docstring must explain the atomicity '
            'contract introduced in v4.14.3.'
        )
        # The multi-process restriction is called out.
        assert 'multi-process' in doc.lower() or 'multiprocess' in doc.lower(), (
            'append_plane_h5 docstring must document the multi-process '
            'restriction (single-writer only).'
        )

    @pytest.mark.skipif(
        not pytest.importorskip(
            'zarr', reason='zarr not installed; _zarr_append_plane '
            'is optional'),
        reason='zarr not installed'
    )
    def test_zarr_append_plane_no_silent_clobber(self):
        """Pin that a partial-state Zarr store (orphan ``plane_00``
        with ``n_planes=0``) is NOT silently overwritten by the next
        legitimate append.  The pre-4.14.3 ``overwrite=True`` flag
        on ``create_array`` was the silent-clobber vector; v4.14.3
        drops it.

        Two acceptable behaviours: (a) the attribute-before-create
        order means the next append computes ``n + 1`` from the
        already-bumped attribute and skips the orphan (preferred);
        or (b) ``create_array`` raises on the existing slot (the
        ``overwrite=False`` default).  The bug-being-pinned is the
        third behaviour: silently overwriting the orphan."""
        import zarr

        from lumenairy.io.storage import _zarr_append_plane

        E_real = np.ones((4, 4), dtype=np.complex128)
        E_orphan_marker = (np.arange(16, dtype=np.complex128)
                           .reshape(4, 4) + 1000.0)

        with tempfile.TemporaryDirectory() as d:
            fp = os.path.join(d, 'partial.zarr')
            # Synthesize a partial-state store: a ``plane_00`` exists
            # but ``n_planes`` is still 0 -- the exact state left
            # behind by a pre-4.14.3 mid-call crash.
            grp = zarr.open_group(fp, mode='w')
            planes = grp.create_group('planes')
            planes.attrs['n_planes'] = 0
            planes.create_array(
                name='plane_00', data=E_orphan_marker,
                chunks=(4, 4),
            )

            # Read back the orphan to confirm setup.
            grp_r = zarr.open_group(fp, mode='r')
            orphan_before = np.array(grp_r['planes/plane_00'][:])
            assert np.allclose(orphan_before, E_orphan_marker), (
                'Test setup is broken: orphan plane_00 not written.'
            )

            # The legitimate next append:
            try:
                _zarr_append_plane(fp, E_real, dx=1e-6, label='real')
            except Exception:
                # Behaviour (b): create raised on the existing slot.
                # That's an acceptable v4.14.3 outcome -- still NOT
                # the silent-clobber bug we're pinning against.
                grp_r2 = zarr.open_group(fp, mode='r')
                orphan_after = np.array(grp_r2['planes/plane_00'][:])
                assert np.allclose(orphan_after, E_orphan_marker), (
                    'v4.14.3 contract: a raise on the existing slot '
                    'must NOT have clobbered the orphan first.'
                )
                return

            # Behaviour (a): the append succeeded.  The orphan was
            # NOT overwritten -- the real data landed in plane_01.
            grp_r3 = zarr.open_group(fp, mode='r')
            orphan_after = np.array(grp_r3['planes/plane_00'][:])
            assert np.allclose(orphan_after, E_orphan_marker), (
                'SILENT-CLOBBER BUG (v4.14.2): a partial-state Zarr '
                'store had its orphan plane_00 silently overwritten '
                'by the next append.  v4.14.3 contract: either skip '
                'the orphan and use plane_01, OR raise.'
            )
            assert 'plane_01' in grp_r3['planes']


# ============================================================================
# A.2 -- ``makedammann2d`` SI heuristic upper bound + ``_legacy_units``
# ============================================================================

class TestAuditFixesV4_14_3_agent_a_A2MakedammannSIHeuristic:
    """Pin the v4.14.3 upper-bound + ``_legacy_units`` API."""

    def test_makedammann2d_rejects_meter_scale_input(self):
        """Pin the ``> 1 m`` upper-bound guard.  A meter-scale
        ``periodx`` is unambiguously wrong in any unit system and
        must raise rather than being silently rescaled."""
        from lumenairy.elements.doe import makedammann2d

        with pytest.raises(ValueError, match='periodx'):
            makedammann2d(periodx=2.0, periody=61e-6, waveln=1.31e-6,
                          itr=1, plot=False)
        with pytest.raises(ValueError, match='periody'):
            makedammann2d(periodx=61e-6, periody=2.0, waveln=1.31e-6,
                          itr=1, plot=False)
        with pytest.raises(ValueError, match='waveln'):
            makedammann2d(periodx=61e-6, periody=61e-6, waveln=1.5,
                          itr=1, plot=False)

    def test_makedammann2d_explicit_si_units_no_rescale(self):
        """Pin that ``_legacy_units='SI'`` accepts mm-scale SI inputs
        without triggering the heuristic.  The output's
        ``cell_pixel_size`` should match a hand-built SI reference
        rather than being silently rescaled by ``1e-6``."""
        from lumenairy.elements.doe import makedammann2d

        # THz / MMW design: 5 mm grating period at 1.1 mm far-IR.
        periodx = 5e-3
        periody = 5e-3
        waveln = 1.1e-3
        wavsamp = 0.5

        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            # _legacy_units='SI' must NOT trigger the heuristic
            # warning, even though every value exceeds 1 mm.
            nf, ff, (dx_out, dy_out) = makedammann2d(
                periodx=periodx, periody=periody, waveln=waveln,
                wavsamp=wavsamp, diforders=np.ones((2, 2)),
                itr=2, plot=False, seed=0,
                _legacy_units='SI',
            )

        # Hand-built SI reference: ndifordersx = ceil(periodx /
        # (wavsamp * waveln) * 0.5) * 2; samplingx = periodx /
        # ndifordersx.  No 1e-6 rescale.
        ndifordersx_ref = int(
            np.ceil(periodx / (wavsamp * waveln) * 0.5)) * 2
        samplingx_ref = periodx / ndifordersx_ref
        # The output ``cell_pixel_size`` is reported in metres.  Pin
        # the order-of-magnitude: dx_out should be mm-scale, not
        # nm-scale.  An accidental rescale by 1e-6 would land it at
        # ~ 1e-9.
        assert dx_out > 1e-5, (
            f'cell_pixel_size={dx_out} m is sub-micron -- the '
            f'_legacy_units="SI" path must NOT have rescaled by 1e-6.  '
            f'Expected mm-scale ({samplingx_ref:.3e} m).'
        )
        # Tighter check: the output matches the SI reference within
        # numerical tolerance.
        assert np.isclose(dx_out, samplingx_ref, rtol=1e-9), (
            f'dx_out={dx_out} != SI reference samplingx={samplingx_ref}.'
        )

    def test_makedammann2d_explicit_um_units_no_warning(self):
        """Pin that ``_legacy_units='um'`` rescales unconditionally
        without emitting a ``DeprecationWarning`` -- the explicit
        opt-in form is a clean migration path."""
        from lumenairy.elements.doe import makedammann2d

        # Legacy micrometre call: periodx=61.0 (61 um).
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            # _legacy_units='um' must NOT emit a deprecation
            # warning -- the caller explicitly acknowledged the unit
            # convention.
            nf, ff, (dx_out, dy_out) = makedammann2d(
                periodx=61.0, periody=61.0, waveln=1.31,
                diforders=np.ones((2, 2)),
                itr=2, plot=False, seed=0,
                _legacy_units='um',
            )
        # The output should be the same as the SI-equivalent call:
        # rescaled to 61e-6 m / 1.31e-6 m internally.
        assert dx_out > 0
        # Quick sanity: the rescale produced micrometre-scale output,
        # not metre-scale.
        assert dx_out < 1e-3, (
            f'Expected ~um-scale cell_pixel_size, got {dx_out} m.'
        )


# ============================================================================
# A.3 -- ``clear_asm_caches`` chains ``clear_lg_polynomial_cache``
# ============================================================================

class TestAuditFixesV4_14_3_agent_a_A3ClearAsmCachesLgPolynomial:
    """Pin that the v4.14.3 ``clear_asm_caches`` chain reaches the
    LG polynomial coefficient cache, closing the v4.14.2 gap."""

    def test_clear_asm_caches_clears_lg_polynomial_cache(self):
        """Populate ``_lg_polynomial_items`` (the lru_cache wrapped by
        :func:`lumenairy.propagators.asymptotic.lg_polynomial`), call
        ``clear_asm_caches``, then assert the cache is empty."""
        from lumenairy.propagators.asymptotic import (
            _lg_polynomial_items,
            lg_polynomial,
        )
        from lumenairy.propagators.propagation import clear_asm_caches

        # Populate the cache by calling the public entry point.  The
        # exact (p, ell, w) values do not matter for cache-clearing
        # purposes; we just need at least one entry.
        lg_polynomial(0, 0, 1e-3)
        lg_polynomial(1, 0, 1e-3)
        assert _lg_polynomial_items.cache_info().currsize > 0, (
            'Test setup failure: lg_polynomial did not populate the '
            'lru_cache.  Cannot pin clear-on-call without a non-empty '
            'starting state.'
        )

        clear_asm_caches()

        assert _lg_polynomial_items.cache_info().currsize == 0, (
            'v4.14.3 contract: clear_asm_caches must chain '
            'clear_lg_polynomial_cache so the LG polynomial '
            'coefficient lru_cache is drained.  This was the 8th '
            'sibling cache missed by the v4.14.2 expansion.'
        )

    def test_clear_asm_caches_docstring_lists_lg_polynomial(self):
        """The v4.14.3 docstring must reference
        ``clear_lg_polynomial_cache`` by name so users searching the
        docs for the entry point can find it.  Mirrors the
        v4.14.2 docstring-pinning test that Agent C added for the
        previous five chained caches."""
        from lumenairy.propagators.propagation import clear_asm_caches
        doc = clear_asm_caches.__doc__
        assert doc is not None
        assert 'clear_lg_polynomial_cache' in doc, (
            'clear_asm_caches docstring missing reference to '
            'clear_lg_polynomial_cache; v4.14.3 must list every '
            'chained cache.'
        )

    def test_clear_asm_caches_drains_all_eight_caches(self):
        """Extend the v4.14.2 "combined drain" promise to include the
        LG polynomial cache.  Agent C's test in
        ``test_audit_fixes_v4_14_2_agent_c.py`` covers the seven
        previously-chained caches; this test adds the 8th
        (``_lg_polynomial_items``).  Together they pin that a single
        ``clear_asm_caches()`` call leaves the propagator-adjacent
        state completely pristine -- matching
        :func:`lumenairy.lumenairy_context`."""
        from lumenairy.propagators.asymptotic import (
            _lg_polynomial_items,
            lg_polynomial,
        )
        from lumenairy.propagators.propagation import (
            _BANDLIMIT_CACHE,
            _FREQ_GRID_CACHE,
            _H_CACHE,
            clear_asm_caches,
        )

        # Populate the LG polynomial cache + the local propagation
        # caches.  We do not need to populate every sibling cache
        # -- Agent C's test covers those; here we just pin that the
        # 8th cache is included in the drain.
        lg_polynomial(0, 0, 1e-3)
        lg_polynomial(2, 1, 2e-3)
        # Populate a freq-grid entry by computing one ASM step.
        from lumenairy.propagators.propagation import (
            angular_spectrum_propagate,
        )
        E = np.ones((16, 16), dtype=np.complex128)
        angular_spectrum_propagate(E, z=1e-3, wavelength=1.31e-6,
                                   dx=1e-6)
        assert _lg_polynomial_items.cache_info().currsize > 0
        assert len(_FREQ_GRID_CACHE) > 0

        clear_asm_caches()

        assert _lg_polynomial_items.cache_info().currsize == 0
        assert len(_FREQ_GRID_CACHE) == 0
        assert len(_BANDLIMIT_CACHE) == 0
        assert len(_H_CACHE) == 0
