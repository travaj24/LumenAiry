"""v5.4.7 regression pins for the AUDIT_V5_4_6 remediation backlog.

- Gap #1 (P2): the v5.4.6 F-14 mirror-radius negation only patched the
  ``unrolled`` codegen style; the ``system`` style emitted the raytrace-
  convention radius un-negated -> a >5000x focus-vs-defocus inversion on
  any curved fold mirror.  Pin BOTH styles.
- Gap #2 (P3): the biconic per-axis sag derivative (``_axis_deriv``)
  returned 0.0 (a bogus flat normal) out of the conic domain, where the
  sag itself is already NaN -- finish the F-19 NaN-consistency fix.

(Gap #3 -- the F-11 piston test -- is fixed in test_v5_4_6_wave6_analysis.py;
 gap #4 -- create_led_source validation visibility -- in
 test_v4_15_dispatcher_pin_validate_grid_params.py.)
"""
from __future__ import annotations

import inspect
import re
from types import SimpleNamespace

import numpy as np

import lumenairy as la
from lumenairy.raytrace.surface import _surface_sag_derivatives_xy

# ---- #11: complete the physics-sign pin coverage (F-12 tolerancing) ---

def test_tolerancing_sweep_uses_fixed_nominal_strehl_denominator():
    """F-12 (audit AUDIT_V5_4_6 Part 4 / #11): tolerancing_sweep must
    compute the Strehl denominator (diffraction-limited peak) ONCE from the
    nominal pupil and thread it to every perturbation via
    ``ideal_peak_fixed=``, NOT recompute it from each perturbed pupil --
    the v5.2.5 nominal-pupil bug (which let per-perturbation Strehls exceed
    1).  Guards the fix structure so a revert to per-perturbation recompute
    fails CI.  (The numerical behaviour was verified in AUDIT_V5_4_6.)"""
    from lumenairy.analysis.through_focus import tolerancing_sweep
    src = inspect.getsource(tolerancing_sweep)
    assert src.count('nominal_ideal_peak = diffraction_limited_peak') == 1, (
        "the nominal Strehl denominator must be computed exactly once")
    assert 'ideal_peak_fixed=nominal_ideal_peak' in src, (
        "the fixed nominal denominator must be threaded to the perturbation "
        "runs via ideal_peak_fixed=")

_MIRROR_PRESCRIPTION = {
    'surfaces': [{'radius': -0.5, 'conic': 0.0, 'aspheric_coeffs': None,
                  'glass_before': 'air', 'glass_after': 'air'}],
    'thicknesses': [0.0],
    'elements': [{'element_type': 'mirror', 'radius': -0.5, 'conic': 0.0,
                  'glass_before': 'air', 'glass_after': 'air',
                  'semi_diameter': 5e-3, 'surf_num': 1}],
    'all_thicknesses': [0.0],
    'coord_breaks': [],
}


def _emit(style):
    return la.io.generate_simulation_script(
        _MIRROR_PRESCRIPTION, wavelength=633e-9, N=64, dx=10e-6,
        source_sigma=2e-3, include_plotting=False, include_analysis=False,
        style=style)


# ---- Gap #1: codegen system style negates the mirror radius -----------

def test_codegen_system_style_negates_mirror_radius():
    """A concave prescription mirror (R<0) must be emitted with the
    wave-side sign (R>0) in the SYSTEM codegen style too, not just the
    unrolled style.  Pre-v5.4.7 the system style emitted radius=-0.5
    (diverging) -> a >5000x focus inversion vs the unrolled +0.5."""
    script = _emit('system')
    # System style emits a step dict: {'type': 'mirror', 'radius': X, ...}
    m = re.search(r"'type':\s*'mirror',\s*'radius':\s*([-\d.eE+]+)", script)
    assert m, f"no system-style mirror dict found in:\n{script}"
    emitted = float(m.group(1))
    assert np.isclose(emitted, 0.5), (
        f"system-style codegen emitted radius={emitted}; expected +0.5 "
        f"(negated from the -0.5 raytrace-convention prescription radius)")


def test_codegen_styles_agree_on_mirror_sign():
    """The unrolled and system styles must emit the SAME (wave-side)
    mirror radius -- a curved fold mirror must focus identically
    regardless of codegen style."""
    unrolled = _emit('unrolled')
    system = _emit('system')
    mu = re.search(r'apply_mirror\([^)]*radius=([-\d.eE+]+)', unrolled)
    ms = re.search(r"'type':\s*'mirror',\s*'radius':\s*([-\d.eE+]+)", system)
    assert mu and ms
    assert np.isclose(float(mu.group(1)), float(ms.group(1))), (
        "unrolled and system codegen styles disagree on the emitted "
        "mirror radius sign")


# ---- Gap #2: biconic per-axis derivative is NaN out of domain ---------

def test_biconic_axis_derivative_nan_out_of_domain():
    """The biconic per-axis sag derivative must be NaN (not 0.0) where the
    conic is out of domain, matching the NaN sag (v5.4.6 F-19 / P3-2).  A
    silent 0.0 there gives a bogus axial (flat) surface normal where the
    surface does not actually exist."""
    # Oblate biconic: R_y = 5 mm, K_y = 3.  At y = 3 mm,
    # (1+K_y) y^2 / R_y^2 = 4 * 9e-6 / 25e-6 = 1.44 >= 1 -> out of domain.
    surf = SimpleNamespace(
        radius=5e-3, radius_y=5e-3, conic=0.0, conic_y=3.0,
        aspheric_coeffs=None, aspheric_coeffs_y=None, freeform=None)
    dz_dx, dz_dy = _surface_sag_derivatives_xy(0.0, 3e-3, surf)
    assert np.isnan(float(dz_dy)), (
        "dz/dy must be NaN out of the conic domain (was a silent 0.0)")
    # x = 0 is in-domain along x -> finite (zero-slope) derivative.
    assert np.isfinite(float(dz_dx))


def test_biconic_axis_derivative_finite_in_domain():
    """Inside the conic domain the biconic derivative stays finite (the
    NaN fix must not perturb valid geometry)."""
    surf = SimpleNamespace(
        radius=5e-3, radius_y=5e-3, conic=0.0, conic_y=3.0,
        aspheric_coeffs=None, aspheric_coeffs_y=None, freeform=None)
    dz_dx, dz_dy = _surface_sag_derivatives_xy(1e-3, 1e-3, surf)
    assert np.isfinite(float(dz_dx)) and np.isfinite(float(dz_dy))


# ---- #6: binary .zmx writer resolves the aperture stop ----------------

def _folded_internal_stop_prescription():
    return {
        'surfaces': [
            {'radius': 0.05, 'conic': 0.0, 'glass_before': 'air',
             'glass_after': 'N-BK7', 'semi_diameter': 0.01, 'is_stop': False},
            {'radius': -0.05, 'conic': 0.0, 'glass_before': 'N-BK7',
             'glass_after': 'air', 'semi_diameter': 0.01, 'is_stop': True},
        ],
        'thicknesses': [0.005, 0.02],
        'stop_index': 1,
        'elements': [
            {'element_type': 'refractive', 'radius': 0.05, 'conic': 0.0,
             'glass_before': 'air', 'glass_after': 'N-BK7',
             'semi_diameter': 0.01, 'surf_num': 1, 'is_stop': False},
            {'element_type': 'mirror', 'radius': -0.3, 'conic': 0.0,
             'glass_before': 'air', 'glass_after': 'air',
             'semi_diameter': 0.01, 'surf_num': 2, 'is_stop': False},
            {'element_type': 'refractive', 'radius': -0.05, 'conic': 0.0,
             'glass_before': 'air', 'glass_after': 'N-BK7',
             'semi_diameter': 0.01, 'surf_num': 3, 'is_stop': True},
        ],
        'all_thicknesses': [0.005, 0.02, 0.0],
        'coord_breaks': [],
    }


def _zmx_stop_surf_blocks(text):
    blocks = re.split(r'\nSURF ', text)
    return [i for i, b in enumerate(blocks) if 'STOP' in b]


def test_zmx_full_writer_preserves_internal_stop(tmp_path):
    """A folded (mirror-bearing) prescription whose aperture stop is the
    SECOND refractive surface must export the STOP marker on that surface,
    not the first.  v5.4.7 (audit AUDIT_V5_4_6 #6): the cb/mirror-aware
    .zmx writer must resolve the stop from the prescription (it always did
    via the public ``export_zemax_zmx`` F-29 path; this also pins the
    internal default)."""
    from lumenairy.io.prescriptions_zemax import (
        _export_zemax_zmx_full,
        export_zemax_zmx,
    )
    rx = _folded_internal_stop_prescription()

    p_pub = tmp_path / 'pub.zmx'
    export_zemax_zmx(rx, str(p_pub), wavelength=633e-9)
    pub_stop = _zmx_stop_surf_blocks(p_pub.read_text())

    # Direct internal call with stop_surface unset must self-resolve, not
    # fall back to the historical hardcoded surface 0.
    p_dir = tmp_path / 'dir.zmx'
    _export_zemax_zmx_full(rx, str(p_dir), wavelength=633e-9)
    dir_stop = _zmx_stop_surf_blocks(p_dir.read_text())

    assert pub_stop, "exported .zmx lost the aperture stop entirely"
    assert pub_stop == dir_stop, (
        "public and direct .zmx writers disagree on the stop surface")
    # The first refractive surface is the very first STANDARD SURF block
    # (block 1); the stop is the SECOND refractive surface, so STOP must
    # NOT sit on block 1.
    assert 1 not in pub_stop, (
        f"STOP marked on the first surface (blocks={pub_stop}); expected the "
        f"second refractive surface (internal-stop resolution failed)")
