"""P9 (niche N10a / N10b): decenter / tilt in the RAY-based real-lens models.

P3 delivered the analytic decenter/tilt *phase* (correct centroid + coma
DIRECTION) but its single-plane screen cannot represent the transverse ray walk
between a thick element's surfaces, so the induced-coma SPOT was wrong: the model
EE80 NARROWED ~0.906x where ZOS + the geometric oracle both BROADEN ~1.02-1.03x.
P3 also established that ``apply_real_lens_traced`` and ``apply_real_lens_gbd``
IGNORED the decenter/tilt keys entirely (centered spot).

N10 threads a per-surface FIELD-FRAME decenter/tilt/freeform through the shared
low-level raytrace geometry (``_surface_sag_xy`` / ``_surface_sag_derivatives_xy``
/ ``_surface_normal``), so BOTH ray models carry the transverse walk-off (the true
induced coma) into the geometry + OPL naturally.  The convention is IDENTICAL to
the analytic displaced-pointwise model (``_disp_surface_z_grad``) and the
lumenairy-free ``geom_spot_decenter_oracle`` -- pinned by the cross-model
agreement test below.

**Amplitude honesty (2026-07-20, verifier round).**  The P9 build additionally
SWAPPED the traced amplitude leg to the bare input envelope for field-frame
elements, reporting a "1.097 broadening within 1.6% of ZOS".  The adversarial
verifier REFUTED that as a model-mixing artefact (forcing the input-envelope
amplitude on CENTERED geometry already widens the on-axis EE80 ~8% at ZERO
decenter).  The swap is REMOVED: the traced hybrid's grid-indexed amplitude
cannot carry the decentered walk-off (an asymmetric ray-density redistribution),
so its decentered-spot EE is amplitude-limited -- a genuine model limit of the P3
single-plane class.  The accurate decentered-coma reference is
``apply_real_lens_gbd`` (N10b): its beamlets carry the walk-off amplitude, so the
spot BROADENS under decenter.  ``traced`` remains correct + oracle-matched on
decenter GEOMETRY: centroid, sign-mirror, tilt.

**Grid-robustness (2026-07-20, second verifier round).**  A follow-up verifier
killed the N10b EE80 headline (1 mm ratio "1.034 matching ZOS 1.035 to 0.1%") as
grid-quantization noise: the ~18-24 um exit spot is only 2-3 px in radius on the
coarse INPUT-grid reconstruction, so the on-axis EE80 itself wandered ~11% across
dx and the 1 mm ratio swung 0.997 -> 1.034 -> 1.003.  The fix reconstructs the
evolved beamlets on a spot-RESOLVED fine grid and measures the RMS second-moment
radius (continuous over all pixels), whose broadening ratio is invariant to the
reconstruction pitch (to ~4 sig figs).  GBD BROADENS grid-robustly (RMS ratio
1.024 @1 mm, 1.093 @2 mm, both wavelengths); the common-mode-subtracted
in-quadrature coma RMS matches the geom oracle within ~15%.  ZOS POP corroborates
the DIRECTION (26.67 -> 27.60 EE80) but the sub-percent match was noise and is
withdrawn.

Gates (all runnable WITHOUT Zemax; the ZOS magnitude comparisons are recorded in
docs/audit_real_lens_displaced_2026_07_19.md):
  * CONVENTION -- raytrace field-frame sag/normal == analytic == geom oracle;
  * ZERO-DECENTER byte-identical (the regression pin);
  * N10a traced -- decenter GEOMETRY correct: centroid vs the geom oracle +
    sign-mirror + tilt; and the amplitude leg is NOT swapped (artefact-gone pin);
  * N10b GBD (the decenter reference) -- decenter BROADENS GRID-ROBUSTLY (RMS on
    a spot-resolved grid, ratio invariant to reconstruction pitch, coma RMS
    within ~15% of the geom oracle at two wavelengths) + power (frame
    completeness) + centroid + a GBD-vs-traced cross-oracle + zero-decenter
    byte-identical + sign-mirror.
"""
from __future__ import annotations

import importlib.util
import pathlib
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements._lens_real import _disp_surface_z_grad
from lumenairy.elements._lens_traced import _prescription_has_field_frame
from lumenairy.elements.lenses_gbd import (
    _GBD_AUTO_HUSIMI_THRESH,
    _auto_sample_step,
    _entrance_aperture_mask,
    _input_angular_spread,
    _prune_zero_beamlets,
)
from lumenairy.glass import GLASS_REGISTRY
from lumenairy.propagators.gbd import (
    apply_prescription_persurface_to_beamlets,
    decompose_field_to_beamlets,
    reconstruct_field_from_beamlets,
)
from lumenairy.raytrace import surfaces_from_prescription
from lumenairy.raytrace.surface import (
    _base_surface_sag_derivatives_xy,
    _base_surface_sag_xy,
    _field_frame_active,
    _field_frame_sag_and_grad,
    _surface_normal,
    _surface_sag_derivatives_xy,
    _surface_sag_xy,
)

GLASS_REGISTRY['_P9A'] = lambda wl: 1.5168
_Z_IMG = 49.162e-3   # paraxial image distance (exit vertex -> image) of the singlet


# ---- lumenairy-free geometric spot oracle (decenter / tilt) ---------------
_ORACLE_PATH = (pathlib.Path(__file__).resolve().parents[2]
                / 'validation' / 'oracles' / 'geom_spot_decenter_oracle.py')
_spec = importlib.util.spec_from_file_location('geom_spot_decenter_oracle',
                                               _ORACLE_PATH)
_oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oracle)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _singlet(wl=1.31e-6, dec=None, tilt=None, sag_callable=None):
    s0 = {'radius': 51.68e-3, 'thickness': 5e-3, 'glass_before': 'air',
          'glass_after': '_P9A', 'semi_diameter': 6e-3}
    if dec is not None:
        s0['decenter'] = dec
    if tilt is not None:
        s0['tilt'] = tilt
    if sag_callable is not None:
        s0['sag_callable'] = sag_callable
        s0['radius'] = float('inf')
    return {'wavelength': wl, 'aperture_diameter': 10e-3, 'surfaces': [
        s0,
        {'radius': -51.68e-3, 'thickness': 0.0, 'glass_before': '_P9A',
         'glass_after': 'air', 'semi_diameter': 6e-3}],
        'thicknesses': [5e-3], 'stop_index': 0}


def _gjob(wl=1.31e-6, dec_mm=(0.0, 0.0), tilt_mrad=(0.0, 0.0), w0_mm=4.0):
    return {'wavelength_um': wl * 1e6, 'aperture_mm': 10.0,
            'pop': {'w0_mm': w0_mm}, 'R_in_mm': None, 'surfaces': [
                {'radius_mm': 51.68, 'thickness_mm': 5.0, 'index': 1.5168,
                 'decenter_mm': list(dec_mm), 'tilt_mrad': list(tilt_mrad)},
                {'radius_mm': -51.68, 'thickness_mm': 49.162, 'index': 'air'}]}


def _gauss(N, dx, w0):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


def _traced(E0, presc, dx, wl):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return la.apply_real_lens_traced(E0, prescription=presc, wavelength=wl,
                                         dx=dx, on_undersample='silent')


def _gbd(E0, presc, dx, wl, bpa=48, diag=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return la.apply_real_lens_gbd(
            E0, prescription=presc, wavelength=wl, dx=dx,
            beamlets_per_aperture=bpa, output_plane_distance=_Z_IMG,
            diagnostics=diag)


def _gbd_evolve_at_image(E0, presc, dx, wl, bpa=64):
    """Decompose -> prune -> per-surface evolve to the image plane, returning the
    EVOLVED beamlet bundle (the physical GBD object).

    This replicates ``apply_real_lens_gbd``'s internals up to -- but NOT
    including -- its fixed input-grid reconstruction (``centre=(0,0)`` on the
    input ``Ny,Nx,dx``), so the caller can reconstruct on a spot-RESOLVED grid
    instead.  That decoupling is the fix for the verifier kill: the ~18-24 um
    exit spot is only 2-3 px in radius on the coarse input grid (dx=8-10 um), so
    an EE metric read there is grid-quantization noise (the on-axis EE80 wandered
    ~11% across dx).  The beamlets themselves are band-limited by their own
    waists, not by any grid, so sampling them finely is the honest measurement.
    Collimated input -> ``_input_angular_spread ~ 0`` -> axial launch, identical
    to ``apply_real_lens_gbd``'s ``direction_sampling='auto'`` there."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E_dec = E0 * _entrance_aperture_mask(E0, dx, dx, presc)
        ss = _auto_sample_step(E_dec, dx, presc, beamlets_per_aperture=bpa)
        uds = _input_angular_spread(E_dec, dx, dx, wl) > _GBD_AUTO_HUSIMI_THRESH
        bundle = decompose_field_to_beamlets(
            E_dec, dx, wavelength=wl, dy=dx, waist_factor=float(ss),
            sample_step=int(ss), direction_sampling=uds)
        bundle, _ = _prune_zero_beamlets(bundle)
        return apply_prescription_persurface_to_beamlets(
            bundle, presc, wl, z_image=_Z_IMG, jacobian='auto')


def _gbd_spot_rms(evolved, cx_um, wl, dxo_um=2.0, locwin_um=220.0):
    """RMS second-moment spot radius (um) + centroid-x (um) from reconstructing
    the evolved beamlets on a FINE grid of pitch ``dxo_um`` centred at
    ``(cx_um, 0)``.  The RMS integrates the intensity continuously over every
    pixel, so (unlike the single-radius EE80 threshold crossing on a 2-3 px spot)
    it is grid-robust -- invariant to the reconstruction pitch to ~4 sig figs."""
    dxo = dxo_um * 1e-6
    nloc = int(round(locwin_um * 1e-6 / dxo))
    nloc += nloc % 2
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E = np.asarray(reconstruct_field_from_beamlets(
            evolved, Ny=nloc, Nx=nloc, dx=dxo, dy=dxo,
            centre=(cx_um * 1e-6, 0.0), wavelength=wl, window=5.0))
    I = np.abs(E) ** 2
    ax = (np.arange(nloc) - nloc / 2) * dxo
    X, Y = np.meshgrid(ax + cx_um * 1e-6, ax)
    P = I.sum()
    xc = (I * X).sum() / P
    yc = (I * Y).sum() / P
    rr = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2)
    rms = np.sqrt((I * rr ** 2).sum() / P)
    return float(rms * 1e6), float(xc * 1e6)


def _img_metrics(E_exit, dx, wl, z=_Z_IMG, propagate=True):
    """EE80 (um) about the intensity centroid + centroid (um) + intensity."""
    if propagate:
        E = la.angular_spectrum_propagate(E_exit.astype(np.complex128), z, wl, dx)
    else:
        E = np.asarray(E_exit)
    I = np.abs(E) ** 2
    N = I.shape[0]
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    P = I.sum()
    xc = (I * X).sum() / P
    yc = (I * Y).sum() / P
    rr = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2).ravel()
    order = np.argsort(rr)
    cum = np.cumsum(I.ravel()[order])
    cum /= cum[-1]
    ee80 = float(np.interp(0.8, cum, rr[order])) * 1e6
    return ee80, float(xc * 1e6), float(yc * 1e6), I


# ===========================================================================
# CONVENTION -- the raytrace field-frame geometry == analytic == geom oracle
# ===========================================================================
def test_field_frame_geometry_matches_analytic_and_geom_oracle():
    """The N10a shared raytrace field-frame sag/normal helper reproduces BOTH
    the analytic displaced-pointwise ``_disp_surface_z_grad`` AND the
    lumenairy-free ``geom_spot_decenter_oracle._surface_z_grad`` to machine
    precision -- so traced / GBD / the analytic model / the oracle all agree on
    what a decenter + tilt MEANS (the cross-model convention pin)."""
    dec, tilt = (0.5e-3, -0.2e-3), (2e-3, 1e-3)
    surf = surfaces_from_prescription(_singlet(dec=dec, tilt=tilt))[0]
    surf_dict = {'radius': 51.68e-3, 'decenter': dec, 'tilt': tilt}
    osurf = {'_dec': dec, '_tilt': tilt, '_R': 51.68e-3, '_k': 0.0, '_asph': {}}
    x = np.linspace(-4e-3, 4e-3, 21)
    y = np.linspace(-3e-3, 3e-3, 21)
    X, Y = np.meshgrid(x, y)
    f_rt, fx_rt, fy_rt = _field_frame_sag_and_grad(X, Y, surf)
    f_an, fx_an, fy_an = _disp_surface_z_grad(surf_dict, X, Y)
    f_or, fx_or, fy_or = _oracle._surface_z_grad(osurf, X, Y)
    # raytrace vs analytic (both field-frame); analytic uses a relative-eps FD
    # radial derivative so the gradient agrees to ~1e-6, the sag to ~1e-12.
    assert np.nanmax(np.abs(f_rt - f_an)) < 1e-12
    assert np.nanmax(np.abs(fx_rt - fx_an)) < 1e-5
    assert np.nanmax(np.abs(fy_rt - fy_an)) < 1e-5
    # raytrace vs geom oracle (both analytic radial derivative): machine eps
    assert np.nanmax(np.abs(f_rt - f_or)) < 1e-15
    assert np.nanmax(np.abs(fx_rt - fx_or)) < 1e-13
    assert np.nanmax(np.abs(fy_rt - fy_or)) < 1e-13
    # the surface NORMAL follows n = (-fx, -fy, 1)/|.|
    nx, ny, nz = _surface_normal(X, Y, surf)
    nn = np.sqrt(fx_or ** 2 + fy_or ** 2 + 1.0)
    assert np.nanmax(np.abs(nx - (-fx_or / nn))) < 1e-13
    assert np.nanmax(np.abs(nz - (1.0 / nn))) < 1e-13


# ===========================================================================
# ZERO-DECENTER byte-identical (the regression pin)
# ===========================================================================
def test_zero_decenter_geometry_byte_identical():
    """A surface with no field-frame decenter/tilt (absent OR explicit (0,0))
    takes the pre-N10a base geometry path bit-for-bit."""
    x = np.linspace(-5e-3, 5e-3, 33)
    X, Y = np.meshgrid(x, x)
    for presc in (_singlet(), _singlet(dec=(0.0, 0.0), tilt=(0.0, 0.0))):
        s = surfaces_from_prescription(presc)[0]
        assert _field_frame_active(s) is False
        assert np.array_equal(_surface_sag_xy(X, Y, s),
                              _base_surface_sag_xy(X, Y, s))
        bx, by = _base_surface_sag_derivatives_xy(X, Y, s)
        dx_, dy_ = _surface_sag_derivatives_xy(X, Y, s)
        assert np.array_equal(dx_, bx) and np.array_equal(dy_, by)
    assert _prescription_has_field_frame(_singlet()) is False
    assert _prescription_has_field_frame(_singlet(dec=(0.0, 0.0))) is False
    assert _prescription_has_field_frame(_singlet(dec=(1e-3, 0.0))) is True
    assert _prescription_has_field_frame(_singlet(tilt=(1e-3, 0.0))) is True


def test_traced_zero_decenter_byte_identical():
    """``apply_real_lens_traced`` with an explicit decenter=(0,0) is bit-for-bit
    identical to the same prescription WITHOUT the key -- the field-frame code
    (geometry route + amplitude override) is fully inert at zero decenter."""
    N, dx, wl = 384, 8e-6, 1.31e-6
    E0 = _gauss(N, dx, 3e-3)
    a = _traced(E0, _singlet(), dx, wl)
    b = _traced(E0, _singlet(dec=(0.0, 0.0), tilt=(0.0, 0.0)), dx, wl)
    # numerical (not bit) identity -- traced cache-cold-vs-warm floats ~1 ULP
    # across cross-test ordering; the field-frame no-op is exact by construction.
    assert np.max(np.abs(np.asarray(a) - np.asarray(b))) <= \
        1e-10 * float(np.max(np.abs(np.asarray(b))))


def test_gbd_zero_decenter_byte_identical():
    """``apply_real_lens_gbd`` with decenter=(0,0) is bit-identical to no key."""
    N, dx, wl = 256, 12e-6, 1.31e-6
    E0 = _gauss(N, dx, 3e-3)
    a = _gbd(E0, _singlet(), dx, wl, bpa=24)
    b = _gbd(E0, _singlet(dec=(0.0, 0.0)), dx, wl, bpa=24)
    # numerical (not bit) identity -- reduction-order ~1 ULP across cross-test
    # ordering; the field-frame (0,0) key is a no-op by construction.
    assert np.max(np.abs(np.asarray(a) - np.asarray(b))) <= \
        1e-10 * float(np.max(np.abs(np.asarray(b))))


# ===========================================================================
# N10a TRACED -- centroid, sign-mirror, tilt (fast, Nyquist-compliant)
# ===========================================================================
def test_traced_decenter_centroid_matches_geom_oracle():
    """A 1 mm front-surface decenter shifts the traced PSF centroid to match the
    lumenairy-free geometric oracle within 5% (measured ~0.4%) -- the ray trace
    threads the decenter geometry (killing the P3 'centered spot' gap)."""
    N, dx, wl = 1024, 6e-6, 1.31e-6
    E0 = _gauss(N, dx, 4e-3)
    _, xc0, yc0, _ = _img_metrics(_traced(E0, _singlet(), dx, wl), dx, wl)
    assert abs(xc0) < 3.0 and abs(yc0) < 3.0
    _, xc, yc, _ = _img_metrics(
        _traced(E0, _singlet(dec=(1e-3, 0.0)), dx, wl), dx, wl)
    g = _oracle.geom_spot(_gjob(dec_mm=(1.0, 0.0)))
    assert xc > 0 and abs(yc) < 5.0
    rel = abs(xc - g['centroid_x_um']) / abs(g['centroid_x_um'])
    assert rel < 0.05, (xc, g['centroid_x_um'], rel)


def test_traced_sign_mirror_decenter():
    """+d and -d x-decenters produce mirror-image traced PSFs (centroid mirrors
    <1%, intensity mirror-L2 <3%)."""
    N, dx, wl = 512, 8e-6, 1.31e-6
    E0 = _gauss(N, dx, 3e-3)
    d = 0.6e-3
    _, xp, _, Ip = _img_metrics(
        _traced(E0, _singlet(dec=(d, 0.0)), dx, wl), dx, wl)
    _, xm, _, Im = _img_metrics(
        _traced(E0, _singlet(dec=(-d, 0.0)), dx, wl), dx, wl)
    assert abs(xp + xm) / abs(xp) < 1e-2
    Im_flip = np.roll(Im[:, ::-1], 1, axis=1)
    assert np.linalg.norm(Ip - Im_flip) / np.linalg.norm(Ip) < 3e-2


def test_traced_tilt_deflection_matches_rigid_rotation():
    """A 0.2 deg front-surface tilt deflects the traced PSF centroid; magnitude
    matches an INDEPENDENT rigid-body full-rotation ray trace to <10% (opposite
    sign is the differing 'positive tilt' definition, not an error)."""
    N, dx, wl = 1024, 6e-6, 1.31e-6
    E0 = _gauss(N, dx, 3e-3)
    tx = 0.2 * np.pi / 180.0
    _, xc, yc, _ = _img_metrics(
        _traced(E0, _singlet(tilt=(tx, 0.0)), dx, wl), dx, wl)
    rig = _oracle.rigid_tilt_centroid(_gjob(tilt_mrad=(tx * 1e3, 0.0)),
                                      n_side=61)
    assert abs(yc) < 5.0
    rel = abs(abs(xc) - abs(rig['centroid_x_um'])) / abs(rig['centroid_x_um'])
    assert rel < 0.10, (xc, rig['centroid_x_um'], rel)


# ===========================================================================
# N10a TRACED amplitude honesty -- the amplitude leg is NOT swapped (the
# verifier's kill #1: the P9 override widened the ON-AXIS EE80 ~8% at ZERO
# decenter, so its "1.097 broadening" was a model-mixing artefact).
# ===========================================================================
def test_traced_field_frame_amplitude_not_swapped():
    """FAIL-BEFORE / PASS-AFTER pin for the removed amplitude-swap artefact.

    The P9 field-frame amplitude override replaced ``|E_analytic|`` with the bare
    input envelope ``|E_in|`` for any field-frame prescription.  Forcing that on
    a GEOMETRICALLY-CENTERED element (a 1e-7 m decenter: centroid ~0.05 um) then
    widened the on-axis EE80 from 24.74 to ~26.8 um -- an ~8% jump with ZERO real
    decenter (grid-robust to N=3072), i.e. the reported decenter "broadening" was
    an amplitude-MODEL artefact, not induced coma.  With the override removed the
    amplitude leg is the standard self-consistent reconstruction, so a
    negligibly-small decenter is CONTINUOUS with the on-axis result.

    (Pre-fix this asserted ~1.08 and FAILED; post-fix ~1.00.)"""
    N, dx, wl = 1024, 6e-6, 1.31e-6
    E0 = _gauss(N, dx, 4e-3)
    m_on, _, _, _ = _img_metrics(_traced(E0, _singlet(wl=wl), dx, wl), dx, wl)
    m_tiny, xc, _, _ = _img_metrics(
        _traced(E0, _singlet(wl=wl, dec=(1e-7, 0.0)), dx, wl), dx, wl)
    assert abs(xc) < 1.0                                   # geometrically centered
    # continuous with the analytic-amplitude on-axis result (no ~8% swap jump).
    assert abs(m_tiny / m_on - 1.0) < 0.03, (m_on, m_tiny, m_tiny / m_on)


# ===========================================================================
# N10b GBD is the decentered-coma reference (beamlets carry the walk-off
# amplitude).  It BROADENS under decenter -- measured GRID-ROBUSTLY with the RMS
# second-moment radius on a spot-RESOLVED reconstruction (NOT the EE80 on the
# coarse input grid, which the verifier correctly killed as quantization noise
# on a 2-3 px spot).  Slow: N=1024, fast evolve, fine local reconstruction.
# ===========================================================================
@pytest.mark.slow
@pytest.mark.parametrize('wl', [1.31e-6, 0.633e-6])
def test_gbd_decenter_broadens_grid_robust_rms(wl):
    """THE N10b headline (plan gate b), GRID-ROBUST -- the verifier-corrected fix.

    The earlier draft measured EE80 on the COARSE input-grid reconstruction,
    where the ~18-24 um spot is only 2-3 px in radius, so its EE80 was
    grid-quantization noise: the ON-AXIS EE80 itself wandered ~11% across dx
    (17.94/18.98/19.97/19.74 um at dx=8/6/4/10 um) and the 1 mm "ratio 1.034
    matching ZOS 1.035 to 0.1%" rode that noise (it fell to 1.003 at dx=4 um).
    The adversarial verifier correctly killed that number.

    The fix reconstructs the evolved beamlets (the physical GBD object) on a
    spot-RESOLVED fine grid and measures the RMS second-moment radius (a
    continuous integral over all pixels, not a single-radius threshold crossing).
    The RMS broadening ratio is then invariant to the reconstruction pitch, so it
    does NOT evaporate under refinement:

      * BROADENS (RMS ratio > 1) at 1 mm AND 2 mm -- kills the P3 shrink;
      * GRID-ROBUST -- identical (to <0.5%) at two reconstruction pitches
        (measured to 4 sig figs, e.g. 1.0235 vs 1.0236 at dxo=3 vs 1.5 um);
      * MONOTONIC -- 2 mm broadens more than 1 mm (physically-ordered coma);
      * QUANTITATIVELY consistent with the geom oracle: the common-mode-
        subtracted in-quadrature coma RMS sqrt(RMS_dec^2 - RMS_on^2) matches the
        lumenairy-free geometric oracle within ~15% at both decenters.  (The raw
        RATIO 1.024 is diluted below the pure-geometric ratio 1.038 because GBD
        carries a diffraction/reconstruction core the geometric oracle does not;
        the coma contribution ON TOP -- what both models must agree on -- does.)
    """
    N, dx, w0 = 1024, 10e-6, 4e-3        # hw 5.12 mm holds the 5 mm aperture
    E0 = _gauss(N, dx, w0)
    ev0 = _gbd_evolve_at_image(E0, _singlet(wl=wl), dx, wl)
    ev1 = _gbd_evolve_at_image(E0, _singlet(wl=wl, dec=(1e-3, 0.0)), dx, wl)
    ev2 = _gbd_evolve_at_image(E0, _singlet(wl=wl, dec=(2e-3, 0.0)), dx, wl)
    # seed the fine-grid centre from a coarse centroid pass.
    _, c0 = _gbd_spot_rms(ev0, 0.0, wl, dxo_um=4.0)
    _, c1 = _gbd_spot_rms(ev1, 510.0, wl, dxo_um=4.0)
    _, c2 = _gbd_spot_rms(ev2, 1020.0, wl, dxo_um=4.0)
    assert abs(c0) < 3.0 and c1 > 300.0 and c2 > 900.0     # centroid shifted

    ratios1, ratios2 = [], []
    coma1 = coma2 = 0.0
    for dxo in (3.0, 1.5):                                 # two reconstruction pitches
        r0, _ = _gbd_spot_rms(ev0, c0, wl, dxo_um=dxo)
        r1, _ = _gbd_spot_rms(ev1, c1, wl, dxo_um=dxo)
        r2, _ = _gbd_spot_rms(ev2, c2, wl, dxo_um=dxo)
        ratios1.append(r1 / r0)
        ratios2.append(r2 / r0)
        coma1 = np.sqrt(max(r1 ** 2 - r0 ** 2, 0.0))
        coma2 = np.sqrt(max(r2 ** 2 - r0 ** 2, 0.0))

    # (1) BROADENS at both decenters -- kills the P3/N10a shrink.
    assert all(r > 1.01 for r in ratios1), (wl, ratios1)
    assert all(r > 1.05 for r in ratios2), (wl, ratios2)
    # (2) GRID-ROBUST: the ratio does NOT change with the reconstruction pitch
    #     (the verifier's kill was that the old input-grid EE80 swung ~11%).
    assert abs(ratios1[0] - ratios1[1]) < 5e-3, (wl, ratios1)
    assert abs(ratios2[0] - ratios2[1]) < 5e-3, (wl, ratios2)
    # (3) MONOTONIC: 2 mm broadens more than 1 mm.
    assert ratios2[-1] > ratios1[-1], (wl, ratios1, ratios2)
    # (4) in-quadrature coma RMS vs the geom oracle (common-mode-subtracted).
    g0 = _oracle.geom_spot(_gjob(wl=wl, dec_mm=(0.0, 0.0)))['rms_um']
    g1 = _oracle.geom_spot(_gjob(wl=wl, dec_mm=(1.0, 0.0)))['rms_um']
    g2 = _oracle.geom_spot(_gjob(wl=wl, dec_mm=(2.0, 0.0)))['rms_um']
    gc1 = np.sqrt(max(g1 ** 2 - g0 ** 2, 0.0))
    gc2 = np.sqrt(max(g2 ** 2 - g0 ** 2, 0.0))
    assert abs(coma1 / gc1 - 1.0) < 0.30, (wl, coma1, gc1)
    assert abs(coma2 / gc2 - 1.0) < 0.30, (wl, coma2, gc2)


# ===========================================================================
# N10b GBD -- centroid, broadening, power, cross-oracle vs traced, sign-mirror
# ===========================================================================
def test_gbd_decenter_centroid_and_power():
    """GBD threads the decenter into the beamlet base-ray trace + differential
    (ABCD) transfer: the reconstructed image-plane centroid matches the geom
    oracle within 5% (measured ~0.1%) and the frame conserves power (frame
    completeness > 0.98)."""
    N, dx, wl = 768, 10e-6, 1.31e-6
    E0 = _gauss(N, dx, 4e-3)
    diag0, diagd = {}, {}
    _, xc0, _, _ = _img_metrics(
        _gbd(E0, _singlet(), dx, wl, bpa=48, diag=diag0), dx, wl, propagate=False)
    _, xc, yc, _ = _img_metrics(
        _gbd(E0, _singlet(dec=(1e-3, 0.0)), dx, wl, bpa=48, diag=diagd),
        dx, wl, propagate=False)
    assert abs(xc0) < 3.0
    g = _oracle.geom_spot(_gjob(dec_mm=(1.0, 0.0)))
    assert abs(xc - g['centroid_x_um']) / abs(g['centroid_x_um']) < 0.05
    assert abs(yc) < 5.0
    # frame completeness (reconstructed / aperture-transmitted power) ~ 1.
    assert diag0['frame_completeness'] > 0.98
    assert diagd['frame_completeness'] > 0.98


@pytest.mark.slow
def test_gbd_decenter_broadens_and_matches_traced():
    """N10b cross-oracle: GBD BROADENS under decenter (killing the P3 shrink),
    measured with the grid-robust RMS spot radius on a spot-resolved grid, and
    its image-plane centroid agrees with the N10a-fixed traced result (the two
    independent ray models agree on the decenter GEOMETRY).  2 mm decenter."""
    N, dx, wl, d = 1024, 10e-6, 1.31e-6, 2e-3
    E0 = _gauss(N, dx, 4e-3)
    ev0 = _gbd_evolve_at_image(E0, _singlet(), dx, wl)
    evd = _gbd_evolve_at_image(E0, _singlet(dec=(d, 0.0)), dx, wl)
    r0, _ = _gbd_spot_rms(ev0, 0.0, wl, dxo_um=2.0)
    _, cx = _gbd_spot_rms(evd, 1020.0, wl, dxo_um=4.0)     # coarse centroid seed
    rd, _ = _gbd_spot_rms(evd, cx, wl, dxo_um=2.0)
    assert rd > r0, (r0, rd)                               # kills the shrink
    # centroid cross-check vs an independent N10a-fixed traced run.  The traced
    # ASM to the image plane needs the exit-Nyquist pitch (H3: exit NA ~0.1 ->
    # dx <= 6.5 um); at the GBD grid's dx=10 um the ASM aliases the centroid, so
    # the traced leg runs at its own dx=6 um (as the dedicated traced-centroid
    # test does).  GBD's fine-grid centroid is pitch-independent.
    dxt = 6e-6
    Et = _gauss(N, dxt, 4e-3)
    _, xt, _, _ = _img_metrics(
        _traced(Et, _singlet(dec=(d, 0.0)), dxt, wl), dxt, wl)
    assert abs(cx - xt) / abs(xt) < 0.05, (cx, xt)


def test_gbd_sign_mirror_decenter():
    """+d and -d decenters mirror the GBD image-plane centroid to <1%."""
    N, dx, wl = 512, 12e-6, 1.31e-6
    E0 = _gauss(N, dx, 3e-3)
    d = 0.6e-3
    _, xp, _, _ = _img_metrics(
        _gbd(E0, _singlet(dec=(d, 0.0)), dx, wl, bpa=40), dx, wl, propagate=False)
    _, xm, _, _ = _img_metrics(
        _gbd(E0, _singlet(dec=(-d, 0.0)), dx, wl, bpa=40), dx, wl, propagate=False)
    assert abs(xp + xm) / abs(xp) < 1e-2, (xp, xm)
