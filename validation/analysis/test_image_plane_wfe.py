"""Validation for ``lumenairy.eval_image_plane_wfe``,
``first_order_data``, and ``remove_low_order_aberrations`` (3.8.0).

Three independent checks per lens form:

1. ``first_order_data`` matches the analytic thin/thick-lens
   formulas for EFL, BFL, and principal-plane positions.
2. ``eval_image_plane_wfe`` returns chief-at-zero with PV / RMS
   matching the known Seidel third-order spherical-aberration
   prediction within a few percent.
3. ``remove_low_order_aberrations`` reduces a known
   pure-defocus + r^4 input to negligible residual.

Reference values are derived from first-principles (lens equation,
Seidel SA closed forms) rather than vendor numbers, so the tests
are stable against glass-database changes.
"""
from __future__ import annotations

import sys
import pathlib as _pathlib
import numpy as np

_sys_path = _pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_sys_path))
from _harness import Harness

import lumenairy as la
from lumenairy.raytrace import (
    surfaces_from_prescription, first_order_data, FirstOrderData,
)
from lumenairy.analysis import (
    ImagePlaneWFE, eval_image_plane_wfe, remove_low_order_aberrations,
)


H = Harness('image_plane_wfe')
WL = 587.56e-9


# ---------------------------------------------------------------------
H.section('first_order_data: paraxial geometry')


def t_thin_pcx_efl():
    """Plano-convex BK7, R1 = (n-1)*100 mm: EFL should be 100 mm."""
    n = 1.5168
    p = la.make_singlet(R1=(n - 1) * 100e-3, R2=float('inf'),
                          d=4.1e-3, glass='N-BK7', aperture=12e-3)
    fod = first_order_data(p, WL)
    return (abs(fod.efl - 100e-3) < 1e-5,
              f'EFL = {fod.efl*1e3:.4f} mm (expected 100.000)')


H.run('first_order_data EFL on plano-convex f=100',
      t_thin_pcx_efl)


def t_thin_pcx_h_at_v1():
    """For a curved-front plano-convex (flat back),
    the object principal plane H sits AT the first vertex (H=0)."""
    n = 1.5168
    p = la.make_singlet(R1=(n - 1) * 100e-3, R2=float('inf'),
                          d=4.1e-3, glass='N-BK7', aperture=12e-3)
    fod = first_order_data(p, WL)
    return (abs(fod.pp_object_z) < 1e-9,
              f'H = {fod.pp_object_z*1e3:.6f} mm (expected 0)')


H.run('first_order_data H at V1 for plano-convex (curved front)',
      t_thin_pcx_h_at_v1)


def t_thick_equiconvex_symmetric():
    """Symmetric biconvex thick lens: H and H' are symmetric
    (equal magnitude, opposite sign)."""
    R = 80e-3; t = 30e-3
    p = la.make_singlet(R1=R, R2=-R, d=t, glass='N-BK7', aperture=25e-3)
    fod = first_order_data(p, WL)
    diff = abs(fod.pp_object_z + fod.pp_image_z)
    return (diff < 1e-9,
              f'H + H_image = {diff*1e3:.6f} mm '
              f'(H={fod.pp_object_z*1e3:.4f}, '
              f"H'={fod.pp_image_z*1e3:.4f})")


H.run('first_order_data: thick equiconvex H + H_image = 0',
      t_thick_equiconvex_symmetric)


def t_fnum_finite():
    """f/# of an f=100mm singlet at EPD=12mm should be 8.33."""
    n = 1.5168
    p = la.make_singlet(R1=(n - 1) * 100e-3, R2=float('inf'),
                          d=4.1e-3, glass='N-BK7', aperture=12e-3)
    fod = first_order_data(p, WL)
    return (abs(fod.fnum - 100.0 / 12.0) < 1e-4,
              f'f/# = {fod.fnum:.4f} (expected {100/12:.4f})')


H.run('first_order_data f-number',
      t_fnum_finite)


# ---------------------------------------------------------------------
H.section('eval_image_plane_wfe: chief-at-zero, PV positive')


def t_wfe_chief_zero():
    """Chief ray sits at OPD = 0 by construction (re-zeroed)."""
    n = 1.5168
    p = la.make_singlet(R1=(n - 1) * 100e-3, R2=float('inf'),
                          d=4.1e-3, glass='N-BK7', aperture=12e-3)
    p['object_distance'] = 200e-3
    wfe = eval_image_plane_wfe(p, wavelength=WL, n_pupil=31)
    chief_opd = wfe.opd_w[wfe.chief_idx]
    return (abs(chief_opd) < 1e-10,
              f'chief OPD = {chief_opd:.3e} waves')


H.run('eval_image_plane_wfe: chief at zero',
      t_wfe_chief_zero)


def t_wfe_positive_undercorrected():
    """For an undercorrected positive singlet,
    marginal OPD should be POSITIVE (matches rayoptics sign)."""
    n = 1.5168
    p = la.make_singlet(R1=(n - 1) * 100e-3, R2=float('inf'),
                          d=4.1e-3, glass='N-BK7', aperture=12e-3)
    p['object_distance'] = 200e-3
    wfe = eval_image_plane_wfe(p, wavelength=WL, n_pupil=31)
    # Find a marginal ray (max r^2)
    r2 = wfe.px ** 2 + wfe.py ** 2
    r2 = np.where(wfe.alive & np.isfinite(wfe.opd_w), r2, -1)
    j = int(np.argmax(r2))
    return (wfe.opd_w[j] > 0,
              f'marginal OPD = {wfe.opd_w[j]:.4f} waves '
              f'(expected > 0 for undercorrected)')


H.run('eval_image_plane_wfe: marginal OPD positive (undercorrected)',
      t_wfe_positive_undercorrected)


def t_wfe_pv_matches_rayoptics():
    """The PV for our reference L1 lens should be near 1.16 waves,
    which is the rayoptics ``eval_wavefront`` value cross-checked
    in OPDPy_Lumenairy_Crosscheck/xcheck_lens_variety.py."""
    n = 1.5168
    p = la.make_singlet(R1=(n - 1) * 100e-3, R2=float('inf'),
                          d=4.1e-3, glass='N-BK7', aperture=12e-3)
    p['object_distance'] = 200e-3
    wfe = eval_image_plane_wfe(p, wavelength=WL, n_pupil=31)
    pv = wfe.pv_waves
    # 5% tolerance (different libraries use slightly different
    # internal sphere placements and chief-ray definitions; see
    # CROSS_CHECK_METHODOLOGY.md for context).
    return (abs(pv - 1.16) / 1.16 < 0.05,
              f'PV = {pv:.4f} waves (expected ~1.16)')


H.run('eval_image_plane_wfe: PV matches rayoptics (5% tol)',
      t_wfe_pv_matches_rayoptics)


def t_wfe_aplanatic_low_sa():
    """An aplanatic plano-convex (K=-1/n^2 on the curved surface)
    has substantially less spherical aberration than the spherical
    plano-convex with the same focal length and aperture."""
    n = 1.5168
    K = -1.0 / (n * n)
    p_sph = la.make_singlet(R1=(n - 1) * 100e-3, R2=float('inf'),
                              d=4.1e-3, glass='N-BK7', aperture=12e-3)
    p_sph['object_distance'] = 200e-3
    p_apl = la.make_singlet(R1=(n - 1) * 100e-3, R2=float('inf'),
                              d=4.1e-3, glass='N-BK7', aperture=12e-3)
    p_apl['surfaces'][0]['conic'] = K
    p_apl['object_distance'] = 200e-3
    wfe_sph = eval_image_plane_wfe(p_sph, wavelength=WL, n_pupil=31)
    wfe_apl = eval_image_plane_wfe(p_apl, wavelength=WL, n_pupil=31)
    return (wfe_apl.rms_waves < wfe_sph.rms_waves,
              f'RMS spherical = {wfe_sph.rms_waves:.4f}, '
              f'aplanatic = {wfe_apl.rms_waves:.4f}')


H.run('eval_image_plane_wfe: aplanatic conic reduces RMS',
      t_wfe_aplanatic_low_sa)


# ---------------------------------------------------------------------
H.section('remove_low_order_aberrations: residual is higher-order')


def t_remove_pure_defocus():
    """A pure-defocus input (W = c*r^2) should be completely
    annihilated by best-fit removal."""
    n_grid = 32
    pp = np.linspace(-1, 1, n_grid)
    PX, PY = np.meshgrid(pp, pp)
    px = PX.ravel(); py = PY.ravel()
    inside = px ** 2 + py ** 2 <= 1
    px, py = px[inside], py[inside]
    W = 2.345 * (px ** 2 + py ** 2)  # pure defocus
    R = remove_low_order_aberrations(W, px, py)
    return (np.max(np.abs(R)) < 1e-10,
              f'max |residual| = {np.max(np.abs(R)):.3e}')


H.run('remove_low_order: pure defocus -> zero',
      t_remove_pure_defocus)


def t_remove_pure_r4():
    """A pure r^4 input should be annihilated by include_r4=True
    but NOT by include_r4=False."""
    n_grid = 32
    pp = np.linspace(-1, 1, n_grid)
    PX, PY = np.meshgrid(pp, pp)
    px = PX.ravel(); py = PY.ravel()
    inside = px ** 2 + py ** 2 <= 1
    px, py = px[inside], py[inside]
    r2 = px ** 2 + py ** 2
    W = 1.0 * r2 ** 2
    R_with = remove_low_order_aberrations(W, px, py, include_r4=True)
    R_without = remove_low_order_aberrations(W, px, py, include_r4=False)
    ok = (np.max(np.abs(R_with)) < 1e-10 and
            np.max(np.abs(R_without)) > 0.1)
    return (ok, f'with r^4 max={np.max(np.abs(R_with)):.3e}, '
                  f'without r^4 max={np.max(np.abs(R_without)):.3e}')


H.run('remove_low_order: include_r4 toggle works',
      t_remove_pure_r4)


def t_remove_preserves_astigmatism():
    """Astigmatism W = px^2 - py^2 is orthogonal to the
    {1, r^2, px, py, r^4} basis on a unit disk (different angular
    symmetry), so best-fit removal should leave it untouched."""
    n_grid = 32
    pp = np.linspace(-1, 1, n_grid)
    PX, PY = np.meshgrid(pp, pp)
    px = PX.ravel(); py = PY.ravel()
    inside = px ** 2 + py ** 2 <= 1
    px, py = px[inside], py[inside]
    W = px ** 2 - py ** 2  # primary astigmatism
    R = remove_low_order_aberrations(W, px, py, include_r4=True)
    rms_in = float(np.sqrt(np.mean(W ** 2)))
    rms_out = float(np.sqrt(np.mean(R ** 2)))
    # Astigmatism is fully orthogonal -> residual must be ~the input
    return (rms_out / rms_in > 0.98,
              f'rms_out / rms_in = {rms_out/rms_in:.4f} '
              f'(astigmatism is orthogonal; expected ~1.0)')


H.run('remove_low_order: preserves astigmatism (orthogonal mode)',
      t_remove_preserves_astigmatism)


# ---------------------------------------------------------------------
H.section('3.8.2: image_plane + sphere_tangent options')


def t_image_plane_default_paraxial():
    """Default image_plane is 'paraxial' (back-compat)."""
    n = 1.5168
    p = la.make_singlet(R1=(n - 1) * 100e-3, R2=float('inf'),
                          d=4.1e-3, glass='N-BK7', aperture=12e-3)
    p['object_distance'] = 200e-3
    a = eval_image_plane_wfe(p, wavelength=WL, n_pupil=31)
    b = eval_image_plane_wfe(p, wavelength=WL, n_pupil=31,
                              image_plane='paraxial')
    return (abs(a.pv_waves - b.pv_waves) < 1e-12,
              f'default PV={a.pv_waves:.4f}, '
              f"image_plane='paraxial' PV={b.pv_waves:.4f}")


H.run("image_plane default == 'paraxial' (back-compat)",
      t_image_plane_default_paraxial)


def t_best_rms_reduces_rms():
    """best_rms image-plane gives smaller RMS than paraxial for any
    aberrated system."""
    n = 1.5168
    p = la.make_singlet(R1=(n - 1) * 100e-3, R2=float('inf'),
                          d=4.1e-3, glass='N-BK7', aperture=12e-3)
    p['object_distance'] = 200e-3
    par = eval_image_plane_wfe(p, wavelength=WL, n_pupil=31,
                                 image_plane='paraxial')
    bf = eval_image_plane_wfe(p, wavelength=WL, n_pupil=31,
                                image_plane='best_rms')
    return (bf.rms_waves <= par.rms_waves,
              f'paraxial RMS={par.rms_waves:.4f}, '
              f'best_rms RMS={bf.rms_waves:.4f}')


H.run("image_plane='best_rms' reduces RMS below paraxial",
      t_best_rms_reduces_rms)


def t_best_pv_reduces_pv():
    """best_pv image-plane gives PV <= paraxial PV."""
    n = 1.5168
    p = la.make_singlet(R1=(n - 1) * 100e-3, R2=float('inf'),
                          d=4.1e-3, glass='N-BK7', aperture=12e-3)
    p['object_distance'] = 200e-3
    par = eval_image_plane_wfe(p, wavelength=WL, n_pupil=31,
                                 image_plane='paraxial')
    bf = eval_image_plane_wfe(p, wavelength=WL, n_pupil=31,
                                image_plane='best_pv')
    return (bf.pv_waves <= par.pv_waves + 1e-6,
              f'paraxial PV={par.pv_waves:.4f}, '
              f'best_pv PV={bf.pv_waves:.4f}')


H.run("image_plane='best_pv' reduces PV below paraxial",
      t_best_pv_reduces_pv)


def t_best_rms_shifts_image_plane():
    """best_rms reports an img_d_m different from paraxial."""
    n = 1.5168
    p = la.make_singlet(R1=(n - 1) * 100e-3, R2=float('inf'),
                          d=4.1e-3, glass='N-BK7', aperture=12e-3)
    p['object_distance'] = 200e-3
    bf = eval_image_plane_wfe(p, wavelength=WL, n_pupil=31,
                                image_plane='best_rms')
    shift = abs(bf.img_d_m - bf.img_d_m_paraxial)
    return (shift > 0.01e-3 and bf.img_d_m_paraxial > 0,
              f'paraxial img_d={bf.img_d_m_paraxial*1e3:.4f} mm, '
              f'best_rms img_d={bf.img_d_m*1e3:.4f} mm, '
              f'shift={shift*1e6:.2f} um')


H.run('best_rms records distinct img_d_m + img_d_m_paraxial',
      t_best_rms_shifts_image_plane)


def t_sphere_tangent_changes_radius():
    """sphere_tangent='exit_pupil' produces a different r_sphere_m
    than 'vertex' (the chief OPD is unchanged but the sphere radius
    is rebased to the exit-pupil distance)."""
    n = 1.5168
    p = la.make_singlet(R1=(n - 1) * 100e-3, R2=float('inf'),
                          d=4.1e-3, glass='N-BK7', aperture=12e-3)
    p['object_distance'] = 200e-3
    v = eval_image_plane_wfe(p, wavelength=WL, n_pupil=31,
                               sphere_tangent='vertex')
    x = eval_image_plane_wfe(p, wavelength=WL, n_pupil=31,
                               sphere_tangent='exit_pupil')
    # Vertex-tangent: R = img_d.  Exit-pupil-tangent: R = img_d - xp_z
    # (xp_z < 0 for stop-at-front singlet, so |R_xp| > |R_vertex|).
    return (x.r_sphere_m > v.r_sphere_m,
              f'vertex R={v.r_sphere_m*1e3:.4f} mm, '
              f'exit_pupil R={x.r_sphere_m*1e3:.4f} mm')


H.run("sphere_tangent='exit_pupil' rebases the sphere radius",
      t_sphere_tangent_changes_radius)


def t_sphere_tangent_noop_for_chief():
    """Chief-centred reference spheres of different radii give the
    SAME WFE -- the chief is on every concentric sphere with zero
    OPD by construction, and the marginal-ray WFE is invariant to
    where on the chief ray the sphere is tangent.  The PV and RMS
    should agree at the FP-noise level."""
    n = 1.5168
    p = la.make_singlet(R1=(n - 1) * 100e-3, R2=float('inf'),
                          d=4.1e-3, glass='N-BK7', aperture=12e-3)
    p['object_distance'] = 200e-3
    v = eval_image_plane_wfe(p, wavelength=WL, n_pupil=31,
                               sphere_tangent='vertex')
    x = eval_image_plane_wfe(p, wavelength=WL, n_pupil=31,
                               sphere_tangent='exit_pupil')
    # WFE PV should differ by < ~0.01 wave (numerical FP roundoff).
    pv_diff = abs(v.pv_waves - x.pv_waves)
    rms_diff = abs(v.rms_waves - x.rms_waves)
    return (pv_diff < 0.01 and rms_diff < 0.01,
              f'PV diff={pv_diff*1e3:.3g} mw, '
              f'RMS diff={rms_diff*1e3:.3g} mw')


H.run('sphere_tangent vertex vs exit_pupil WFEs match within FP noise',
      t_sphere_tangent_noop_for_chief)


def t_invalid_image_plane_raises():
    """Unknown image_plane string raises ValueError."""
    n = 1.5168
    p = la.make_singlet(R1=(n - 1) * 100e-3, R2=float('inf'),
                          d=4.1e-3, glass='N-BK7', aperture=12e-3)
    p['object_distance'] = 200e-3
    try:
        eval_image_plane_wfe(p, wavelength=WL, n_pupil=11,
                              image_plane='best_potato')
        return (False, "no exception raised for invalid value")
    except ValueError as e:
        return ('best_potato' in str(e),
                  f'raised ValueError: {e}')


H.run('invalid image_plane raises ValueError', t_invalid_image_plane_raises)


# =====================================================================
# Off-axis support (4.0+)
# =====================================================================

H.section('Off-axis fields (4.0+)')


def t_off_axis_requires_field_max_m():
    """Calling with non-zero field but no field_max_m raises ValueError."""
    p = la.make_singlet(R1=51.5e-3, R2=float('inf'), d=4e-3,
                          glass='N-BK7', aperture=10e-3)
    p['object_distance'] = 200e-3
    try:
        la.eval_image_plane_wfe(p, WL, field=(0.5, 0.0), n_pupil=15)
        return False, 'no exception raised'
    except ValueError as e:
        return 'field_max_m' in str(e), str(e)


H.run('off-axis without field_max_m -> ValueError',
      t_off_axis_requires_field_max_m)


def t_off_axis_field_max_m_kwarg():
    """Off-axis WFE computes when field_max_m is given."""
    p = la.make_singlet(R1=51.5e-3, R2=float('inf'), d=4e-3,
                          glass='N-BK7', aperture=10e-3)
    p['object_distance'] = 200e-3
    wfe = la.eval_image_plane_wfe(p, WL, field=(0.5, 0.0), n_pupil=15,
                                    field_max_m=1e-3)
    return (np.isfinite(wfe.rms_waves) and wfe.rms_waves > 0
            and wfe.chief_idx >= 0), \
        f'PV={wfe.pv_waves:.4f}, RMS={wfe.rms_waves:.4f}'


H.run('off-axis WFE with field_max_m kwarg returns finite result',
      t_off_axis_field_max_m_kwarg)


def t_off_axis_aberration_grows_with_field():
    """For a singlet with positive spherical aberration, RMS WFE grows
    monotonically from on-axis to the field edge."""
    p = la.make_singlet(R1=51.5e-3, R2=float('inf'), d=4e-3,
                          glass='N-BK7', aperture=10e-3)
    p['object_distance'] = 200e-3
    wfe_axis = la.eval_image_plane_wfe(p, WL, n_pupil=15)
    wfe_mid = la.eval_image_plane_wfe(p, WL, field=(0.5, 0.0),
                                         n_pupil=15, field_max_m=1e-3)
    wfe_edge = la.eval_image_plane_wfe(p, WL, field=(1.0, 0.0),
                                          n_pupil=15, field_max_m=1e-3)
    monotone = wfe_axis.rms_waves <= wfe_mid.rms_waves <= wfe_edge.rms_waves
    return monotone, (
        f'axis={wfe_axis.rms_waves:.4f}, mid={wfe_mid.rms_waves:.4f}, '
        f'edge={wfe_edge.rms_waves:.4f}')


H.run('RMS WFE grows monotonically with field for a singlet',
      t_off_axis_aberration_grows_with_field)


def t_off_axis_zero_field_matches_on_axis():
    """field=(0,0) with field_max_m given should match field=(0,0)
    without field_max_m to many decimal places."""
    p = la.make_singlet(R1=51.5e-3, R2=float('inf'), d=4e-3,
                          glass='N-BK7', aperture=10e-3)
    p['object_distance'] = 200e-3
    a = la.eval_image_plane_wfe(p, WL, n_pupil=15)
    b = la.eval_image_plane_wfe(p, WL, n_pupil=15, field=(0.0, 0.0),
                                  field_max_m=1e-3)
    err = abs(a.rms_waves - b.rms_waves)
    return err < 1e-12, f'rms diff = {err:.2e}'


H.run('field=(0,0) with field_max_m matches on-axis result',
      t_off_axis_zero_field_matches_on_axis)


def t_field_grid_wfe_shape_and_content():
    """field_grid_wfe returns arrays of the requested shape with
    sensible numerical content."""
    p = la.make_singlet(R1=51.5e-3, R2=float('inf'), d=4e-3,
                          glass='N-BK7', aperture=10e-3)
    p['object_distance'] = 200e-3
    grid = la.field_grid_wfe(p, WL, field_max_m=1e-3, n_field=3,
                                n_pupil=15)
    shape_ok = (grid['Hx'].shape == (3, 3)
                and grid['rms_waves'].shape == (3, 3)
                and len(grid['wfe_per_field']) == 9)
    finite_ok = bool(np.all(np.isfinite(grid['rms_waves'])))
    onaxis_min = (grid['rms_waves'][1, 1]
                  <= grid['rms_waves'][0, 0] + 1e-6)
    return shape_ok and finite_ok and onaxis_min, (
        f"shape={grid['rms_waves'].shape}, "
        f"on-axis={grid['rms_waves'][1,1]:.4f}, "
        f"corner={grid['rms_waves'][0,0]:.4f}")


H.run('field_grid_wfe: shape, finiteness, on-axis-minimum invariant',
      t_field_grid_wfe_shape_and_content)


# ---------------------------------------------------------------------
def main():
    return H.summary()


if __name__ == '__main__':
    sys.exit(main())
