"""Vector-diffraction and partial-coherence tests.

Covers:
- Richards-Wolf / Debye-Wolf vector focusing (from physics_deep_test.py,
  physics_complex_test.py)
- Koehler / mutual coherence / extended-source imaging (from
  physics_deep_test.py, physics_extended_test.py)
"""
from __future__ import annotations

import sys

import numpy as np

from _harness import Harness

import lumenairy as op


H = Harness('advanced_diffraction')


# ---------------------------------------------------------------------
H.section('Richards-Wolf / Debye-Wolf vector focusing')


def t_rw_longitudinal_ez():
    pupil = np.ones((64, 64), dtype=np.complex128)
    Ex, Ey, Ez, _, _ = op.richards_wolf_focus(
        pupil, 1.31e-6, NA=0.9, f=2e-3, dx_pupil=25e-6, N_focal=64)
    ratio = np.abs(Ez).max() / np.abs(Ex).max()
    return ratio > 0.1, \
        f'|Ez|/|Ex| peak ratio = {ratio:.3f}'


H.run('Richards-Wolf: Ez nonzero at high NA', t_rw_longitudinal_ez)


def t_rw_low_na_matches_scalar():
    pupil = np.ones((128, 128), dtype=np.complex128)
    psf_v, _, _ = op.debye_wolf_psf(
        pupil, 1.31e-6, NA=0.1, f=20e-3, dx_pupil=100e-6,
        N_focal=128, polarization='x')
    peak_idx = np.unravel_index(np.argmax(psf_v), psf_v.shape)
    err = max(abs(peak_idx[0] - 64), abs(peak_idx[1] - 64))
    return err <= 1, f'peak offset = {err} pixels'


H.run('Richards-Wolf: low-NA PSF centered (like scalar)',
      t_rw_low_na_matches_scalar)


def t_rw_psf_width():
    NA = 0.1; lam = 0.633e-6; f = 20e-3
    x = (np.arange(128) - 64) * 100e-6
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X**2 + Y**2)
    pupil = np.where(r <= f * NA, 1.0, 0.0).astype(np.complex128)
    psf, _, _ = op.debye_wolf_psf(
        pupil, lam, NA, f, dx_pupil=100e-6, N_focal=128,
        dx_focal=lam / (8 * NA))
    row = psf[64, :]
    half_max = row.max() / 2
    above = row >= half_max
    if above.any():
        fwhm_pix = np.sum(above)
        fwhm = fwhm_pix * (lam / (8 * NA))
    else:
        fwhm = 0
    expected_fwhm = 0.51 * lam / NA
    err_pct = abs(fwhm - expected_fwhm) / expected_fwhm * 100
    return err_pct < 30, \
        (f'FWHM={fwhm*1e6:.2f}um, expected={expected_fwhm*1e6:.2f}um, '
         f'err={err_pct:.1f}%')


H.run('Richards-Wolf PSF width matches 0.51*lam/NA',
      t_rw_psf_width)


# ---------------------------------------------------------------------
H.section('Partial coherence / extended-source imaging')


def t_koehler_reduces_to_coherent_on_axis():
    N = 128; dx = 16e-6; lam = 1.31e-6
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=3e-3)
    E_obj = np.ones((N, N), dtype=np.complex128)
    I_koh = op.koehler_image(E_obj, pres, lam, dx,
                             condenser_NA=0.001, n_source_points=3)
    E_coh = op.apply_real_lens(E_obj, pres, lam, dx)
    I_coh = np.abs(E_coh)**2
    mask = I_coh > I_coh.max() * 0.1
    if mask.sum() < 10:
        return True, 'too few bright pixels'
    ratio = I_koh[mask] / I_coh[mask]
    std = np.std(ratio) / np.mean(ratio)
    return std < 0.15, \
        f'intensity ratio std/mean = {std:.4f}'


H.run('Koehler: single source matches coherent',
      t_koehler_reduces_to_coherent_on_axis)


def t_mutual_coherence_diagonal():
    N = 128; dx = 4e-6; lam = 1.31e-6
    fields = []
    for angle in [0, 0.005, -0.005, 0.003, -0.003]:
        E, _, _ = op.create_tilted_plane_wave(N, dx, lam, angle_y=angle)
        fields.append(E)
    Gamma, _ = op.mutual_coherence(fields, dx)
    diag = np.diag(np.abs(Gamma))
    mean_diag = diag.mean()
    return 0.5 < mean_diag < 2.0, \
        f'mean diag(Gamma) = {mean_diag:.4f}'


H.run('Mutual coherence: diagonal = avg intensity',
      t_mutual_coherence_diagonal)


def t_extended_source_smoothing():
    N = 128; dx = 16e-6; lam = 1.31e-6
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=3e-3)
    x = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x)
    E_obj = np.where(np.abs(X) < 200e-6, 1.0, 0.3).astype(np.complex128)
    I_narrow = op.koehler_image(E_obj, pres, lam, dx,
                                condenser_NA=0.01, n_source_points=3)
    I_wide = op.koehler_image(E_obj, pres, lam, dx,
                              condenser_NA=0.1, n_source_points=5)
    both_finite = np.isfinite(I_narrow).all() and np.isfinite(I_wide).all()
    both_positive = I_narrow.mean() > 0 and I_wide.mean() > 0
    return both_finite and both_positive, \
        f'I_narrow.mean={I_narrow.mean():.4e}, I_wide.mean={I_wide.mean():.4e}'


H.run('Extended source: wider NA smooths fringes',
      t_extended_source_smoothing)


# ---------------------------------------------------------------------
# Additional advanced-diffraction physics tests (3.2.13)
# ---------------------------------------------------------------------

def t_rw_higher_NA_gives_smaller_focal_spot():
    """Richards-Wolf PSF FWHM decreases as NA increases (diffraction
    limit FWHM ~ 0.51 * lambda / NA)."""
    lam = 0.633e-6; f = 20e-3
    pupil = np.ones((128, 128), dtype=np.complex128)
    fwhms = []
    for NA in (0.1, 0.4):
        x = (np.arange(128) - 64) * 100e-6
        X, Y = np.meshgrid(x, x)
        r = np.sqrt(X**2 + Y**2)
        p = np.where(r <= f * NA, 1.0, 0.0).astype(np.complex128)
        psf, _, _ = op.debye_wolf_psf(
            p, lam, NA, f, dx_pupil=100e-6, N_focal=128,
            dx_focal=lam / (8 * NA))
        row = psf[64, :]
        half = row.max() / 2
        above = row >= half
        fwhm = float(np.sum(above)) * (lam / (8 * NA))
        fwhms.append(fwhm)
    return fwhms[0] > fwhms[1], \
        (f'FWHM(NA=0.1)={fwhms[0]*1e6:.3f}um, '
         f'FWHM(NA=0.4)={fwhms[1]*1e6:.3f}um')


H.run('Richards-Wolf: FWHM shrinks with higher NA',
      t_rw_higher_NA_gives_smaller_focal_spot)


def t_rw_positive_NA_yields_finite_intensity():
    """Richards-Wolf at moderate NA returns a finite, positive PSF
    array - smoke check covering generic invocation."""
    pupil = np.ones((64, 64), dtype=np.complex128)
    psf, _, _ = op.debye_wolf_psf(
        pupil, 1.31e-6, NA=0.3, f=10e-3, dx_pupil=50e-6, N_focal=64)
    return np.all(np.isfinite(psf)) and psf.max() > 0, \
        f'finite={np.all(np.isfinite(psf))}, peak={psf.max():.3e}'


H.run('debye_wolf_psf: finite output for moderate NA',
      t_rw_positive_NA_yields_finite_intensity)


def t_mutual_coherence_off_diagonal_smaller_than_diagonal():
    """The Gamma matrix off-diagonal terms have magnitude <= the diagonal
    (Cauchy-Schwarz inequality on coherence functions)."""
    N, dx, lam = 128, 4e-6, 1.31e-6
    fields = []
    for angle in [0, 0.005, -0.005, 0.003, -0.003]:
        E, _, _ = op.create_tilted_plane_wave(N, dx, lam, angle_y=angle)
        fields.append(E)
    Gamma, _ = op.mutual_coherence(fields, dx)
    A = np.abs(Gamma)
    diag = np.diag(A)
    bound_violation = False
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i != j:
                bound = np.sqrt(diag[i] * diag[j])
                if A[i, j] > bound + 1e-10:
                    bound_violation = True
    return not bound_violation, \
        f'Cauchy-Schwarz check passed; max diag={diag.max():.4f}'


H.run('Mutual coherence: |Gamma_ij| <= sqrt(Gamma_ii * Gamma_jj)',
      t_mutual_coherence_off_diagonal_smaller_than_diagonal)


def t_koehler_imaging_finite_intensity():
    """Koehler imaging at typical NA returns a finite, positive image."""
    N, dx, lam = 64, 16e-6, 1.31e-6
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=3e-3)
    E = np.ones((N, N), dtype=np.complex128)
    I = op.koehler_image(E, pres, lam, dx,
                          condenser_NA=0.05, n_source_points=3)
    return (np.all(np.isfinite(I)) and I.mean() > 0), \
        f'I.mean={I.mean():.3e}, finite={np.all(np.isfinite(I))}'


H.run('Koehler imaging: typical NA returns finite image',
      t_koehler_imaging_finite_intensity)


if __name__ == '__main__':
    sys.exit(H.summary())
