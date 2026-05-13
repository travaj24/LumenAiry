"""Validation tests for ``lumenairy.analysis.field`` (4.4.0).

Covers the eight analyses lifted from ui/*_dock.py plus the three
new helpers, and the two new Strehl alternatives.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from _harness import Harness

import numpy as np

import lumenairy as la


H = Harness('field')


def _make_test_singlet():
    return la.make_singlet(R1=50e-3, R2=np.inf, d=3e-3,
                              glass='N-BK7', aperture=10e-3)


# ----------------------------------------------------------------------
# distortion_vs_field
# ----------------------------------------------------------------------

def t_distortion_zero_on_axis():
    d = la.distortion_vs_field(_make_test_singlet(), 1.31e-6,
                                  max_field_deg=2.0, n_points=11)
    return d.distortion_pct[0] == 0.0, \
        f'distortion at theta=0 = {d.distortion_pct[0]}'


H.run('distortion_vs_field: zero distortion on-axis',
      t_distortion_zero_on_axis)


def t_distortion_returns_dataclass():
    d = la.distortion_vs_field(_make_test_singlet(), 1.31e-6,
                                  max_field_deg=2.0, n_points=11)
    ok = (isinstance(d, la.DistortionVsField)
          and d.theta_deg.shape == (11,)
          and d.efl > 0 and d.bfl > 0
          and d.sign in ('pincushion', 'barrel', 'mixed', 'unknown'))
    return ok, (f'sign={d.sign}, EFL={d.efl*1e3:.2f}mm, '
                f'BFL={d.bfl*1e3:.2f}mm')


H.run('distortion_vs_field: dataclass and shape',
      t_distortion_returns_dataclass)


def t_distortion_singlet_sign_is_barrel():
    """A simple plano-convex singlet has barrel distortion (R1
    pre-stop)."""
    d = la.distortion_vs_field(_make_test_singlet(), 1.31e-6,
                                  max_field_deg=5.0, n_points=11)
    return d.sign in ('barrel', 'unknown'), \
        f'sign = {d.sign} for plano-convex (expected barrel)'


H.run('distortion_vs_field: plano-convex singlet shows barrel',
      t_distortion_singlet_sign_is_barrel)


# ----------------------------------------------------------------------
# distortion_grid
# ----------------------------------------------------------------------

def t_distortion_grid_shapes():
    g = la.distortion_grid(_make_test_singlet(), 1.31e-6,
                              max_field_deg=2.0, n_grid=5)
    ok = (g['actual_x'].shape == (5, 5)
          and g['paraxial_x'].shape == (5, 5)
          and g['theta_x_deg'].shape == (5,))
    return ok, f"shapes ok: {g['actual_x'].shape}"


H.run('distortion_grid: returns (N, N) grids', t_distortion_grid_shapes)


# ----------------------------------------------------------------------
# footprint_per_surface
# ----------------------------------------------------------------------

def t_footprint_returns_one_dict_per_surface():
    fp = la.footprint_per_surface(_make_test_singlet(), 1.31e-6,
                                       semi_aperture=5e-3,
                                       fields_deg=(0.0, 1.0),
                                       num_rings=3, rays_per_ring=8)
    # Singlet: 2 lens surfaces + 1 image plane = 3
    return len(fp) == 3, f'len(fp) = {len(fp)} (expected 3)'


H.run('footprint_per_surface: surface count', t_footprint_returns_one_dict_per_surface)


def t_footprint_on_axis_rays_centered():
    fp = la.footprint_per_surface(_make_test_singlet(), 1.31e-6,
                                       semi_aperture=5e-3,
                                       fields_deg=(0.0,),
                                       num_rings=3, rays_per_ring=8)
    img = fp[-1]['fields'][0]
    alive = img['alive']
    if not alive.any():
        return False, 'no alive rays at image plane'
    cx = float(np.mean(img['x'][alive]))
    cy = float(np.mean(img['y'][alive]))
    return abs(cx) < 1e-6 and abs(cy) < 1e-6, \
        f'on-axis image centroid = ({cx*1e6:.2f}, {cy*1e6:.2f}) um'


H.run('footprint_per_surface: on-axis image rays centered',
      t_footprint_on_axis_rays_centered)


# ----------------------------------------------------------------------
# spot_diagram_vs_field
# ----------------------------------------------------------------------

def t_spot_diagram_returns_dict_per_field():
    sd = la.spot_diagram_vs_field(_make_test_singlet(), 1.31e-6,
                                       fields_deg=[0.0, 1.0, 2.0],
                                       semi_aperture=5e-3,
                                       num_rings=3, rays_per_ring=8)
    fields = [d['field_deg'] for d in sd]
    return fields == [0.0, 1.0, 2.0], f'fields = {fields}'


H.run('spot_diagram_vs_field: one dict per field',
      t_spot_diagram_returns_dict_per_field)


def t_spot_diagram_spot_grows_off_axis():
    sd = la.spot_diagram_vs_field(_make_test_singlet(), 1.31e-6,
                                       fields_deg=[0.0, 2.0],
                                       semi_aperture=5e-3,
                                       num_rings=3, rays_per_ring=8)
    return sd[1]['rms_radius'] > sd[0]['rms_radius'], \
        (f"RMS at 0deg = {sd[0]['rms_radius']*1e6:.2f}um, "
         f"at 2deg = {sd[1]['rms_radius']*1e6:.2f}um")


H.run('spot_diagram_vs_field: spot grows off-axis',
      t_spot_diagram_spot_grows_off_axis)


# ----------------------------------------------------------------------
# relative_illumination
# ----------------------------------------------------------------------

def t_relative_illumination_on_axis_unity():
    ri = la.relative_illumination(_make_test_singlet(), 1.31e-6,
                                       fields_deg=[0.0, 1.0, 2.0],
                                       semi_aperture=5e-3,
                                       num_rings=4, rays_per_ring=12)
    return abs(ri.relative_illumination[0] - 1.0) < 1e-9, \
        f'on-axis RI = {ri.relative_illumination[0]}'


H.run('relative_illumination: on-axis == 1.0',
      t_relative_illumination_on_axis_unity)


def t_relative_illumination_decreases_off_axis():
    ri = la.relative_illumination(_make_test_singlet(), 1.31e-6,
                                       fields_deg=[0.0, 3.0],
                                       semi_aperture=5e-3,
                                       num_rings=4, rays_per_ring=12)
    return ri.relative_illumination[-1] <= ri.relative_illumination[0], \
        f'RI at 3deg = {ri.relative_illumination[-1]:.4f}'


H.run('relative_illumination: weak monotone decrease',
      t_relative_illumination_decreases_off_axis)


# ----------------------------------------------------------------------
# field_aberration_sweep
# ----------------------------------------------------------------------

def t_field_aberration_sweep_shapes():
    fa = la.field_aberration_sweep(_make_test_singlet(), 1.31e-6,
                                       fields_deg=[0.0, 1.0],
                                       semi_aperture=5e-3,
                                       n_z=15, n_fan=7)
    ok = (fa.sagittal_focus_shift.shape == (2,)
          and fa.tangential_focus_shift.shape == (2,)
          and fa.astigmatism.shape == (2,)
          and np.isfinite(fa.petzval_radius))
    return ok, (f'sag={fa.sagittal_focus_shift}, '
                f'tan={fa.tangential_focus_shift}, '
                f'petzval={fa.petzval_radius*1e3:.2f}mm')


H.run('field_aberration_sweep: shapes + finite Petzval',
      t_field_aberration_sweep_shapes)


# ----------------------------------------------------------------------
# petzval_radius
# ----------------------------------------------------------------------

def t_petzval_singlet_positive_finite():
    R_p = la.petzval_radius(_make_test_singlet(), 1.31e-6)
    return np.isfinite(R_p) and R_p != 0, \
        f'R_p = {R_p*1e3:.4f} mm (expected finite, nonzero)'


H.run('petzval_radius: finite nonzero for singlet',
      t_petzval_singlet_positive_finite)


def t_petzval_flat_for_zero_power():
    """A glass plate (R1 = R2 = inf) has no Petzval contribution."""
    flat = la.make_singlet(R1=np.inf, R2=np.inf, d=3e-3,
                              glass='N-BK7', aperture=10e-3)
    R_p = la.petzval_radius(flat, 1.31e-6)
    return R_p == float('inf'), f'R_p = {R_p} (expected inf)'


H.run('petzval_radius: flat plate -> infinite R_p',
      t_petzval_flat_for_zero_power)


# ----------------------------------------------------------------------
# sensitivity_ranking
# ----------------------------------------------------------------------

def t_sensitivity_orders_correctly():
    def merit(x):
        return float(2 * x[0] ** 2 + 0.5 * x[1] ** 2 + 0.01 * x[2] ** 2)
    sr = la.sensitivity_ranking(merit, [1.0, 1.0, 1.0],
                                  labels=['a', 'b', 'c'])
    return sr.order[0] == 0 and sr.order[-1] == 2, \
        f'order = {list(sr.order)} (expected [0, 1, 2])'


H.run('sensitivity_ranking: orders quadratic merit correctly',
      t_sensitivity_orders_correctly)


def t_sensitivity_handles_zero_x():
    def merit(x): return float(x[0] ** 2)
    sr = la.sensitivity_ranking(merit, [0.0], labels=['a'])
    return np.isfinite(sr.derivative[0]), \
        f'deriv at x=0 = {sr.derivative[0]}'


H.run('sensitivity_ranking: nonzero step at x=0',
      t_sensitivity_handles_zero_x)


# ----------------------------------------------------------------------
# Strehl alternatives (in analysis.analysis, re-exported at top level)
# ----------------------------------------------------------------------

def t_marechal_at_diffraction_limited():
    """1/14-wave RMS -> S ~ 0.82, the standard diffraction-limited
    threshold."""
    S = float(la.strehl_marechal(1.0 / 14.0))
    return 0.80 < S < 0.83, f'S = {S:.4f} (expected ~0.82)'


H.run('strehl_marechal: 1/14 wave -> ~0.82',
      t_marechal_at_diffraction_limited)


def t_marechal_zero_input_unity():
    return float(la.strehl_marechal(0.0)) == 1.0, 'S(0) != 1'


H.run('strehl_marechal: zero RMS -> unity',
      t_marechal_zero_input_unity)


def t_phase_integral_flat_pupil_unity():
    """Constant-amplitude flat-phase pupil should give S = 1."""
    N = 64
    x = (np.arange(N) - N / 2) / (N / 2)
    X, Y = np.meshgrid(x, x)
    aperture = (X ** 2 + Y ** 2) <= 1.0
    pupil = aperture.astype(complex)
    S = la.strehl_phase_integral(pupil)
    return abs(S - 1.0) < 1e-12, f'S = {S}'


H.run('strehl_phase_integral: flat-phase pupil -> 1.0',
      t_phase_integral_flat_pupil_unity)


def t_phase_integral_piston_only_unity():
    """A pupil with uniform piston phase is still S = 1."""
    N = 64
    x = (np.arange(N) - N / 2) / (N / 2)
    X, Y = np.meshgrid(x, x)
    aperture = (X ** 2 + Y ** 2) <= 1.0
    pupil = aperture * np.exp(1j * 1.234)
    S = la.strehl_phase_integral(pupil)
    return abs(S - 1.0) < 1e-12, f'S with piston = {S}'


H.run('strehl_phase_integral: piston-only -> 1.0',
      t_phase_integral_piston_only_unity)


def t_phase_integral_defocus_reduces_strehl():
    """Adding any defocus must REDUCE the phase-integral Strehl below
    1.0 (by Cauchy-Schwarz)."""
    N = 128
    x = (np.arange(N) - N / 2) / (N / 2)
    X, Y = np.meshgrid(x, x)
    aperture = (X ** 2 + Y ** 2) <= 1.0
    defocus_phase = 2 * np.pi * 0.3 * (X ** 2 + Y ** 2)
    pupil = aperture * np.exp(1j * defocus_phase)
    S = la.strehl_phase_integral(pupil)
    return 0.0 < S < 1.0, f'S with 0.3-wave defocus = {S}'


H.run('strehl_phase_integral: defocus reduces Strehl',
      t_phase_integral_defocus_reduces_strehl)


def t_phase_integral_empty_pupil_zero():
    return la.strehl_phase_integral(np.zeros((16, 16), dtype=complex)) == 0.0, \
        'empty pupil should return 0'


H.run('strehl_phase_integral: empty pupil -> 0',
      t_phase_integral_empty_pupil_zero)


# ----------------------------------------------------------------------
# Glass-catalog warning (Section D #1)
# ----------------------------------------------------------------------

def t_aberration_summary_warns_on_missing_glass():
    """aberration_summary should issue a UserWarning when a prescription
    references a glass not in the registry, instead of silently
    returning zero-filled Seidel coefficients."""
    import warnings
    presc = la.make_singlet(R1=50e-3, R2=np.inf, d=3e-3,
                              glass='__BOGUS_GLASS_DOES_NOT_EXIST__',
                              aperture=10e-3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = la.aberration_summary(presc, wavelength=1.31e-6,
                                          include_lg_tensor=False)
    glass_warnings = [w for w in caught
                      if 'glass not in registry' in str(w.message)]
    return len(glass_warnings) >= 1, \
        f'caught {len(glass_warnings)} glass warning(s); ' \
        f'all msgs = {[str(w.message)[:60] for w in caught]}'


H.run('aberration_summary: warns on unknown glass',
      t_aberration_summary_warns_on_missing_glass)


if __name__ == '__main__':
    import sys
    sys.exit(H.summary())
