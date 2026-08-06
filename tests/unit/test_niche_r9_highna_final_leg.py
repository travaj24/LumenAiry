"""R9 / audit F2 end-to-end (2026-07-22): EXACT (non-paraxial) high-NA final leg.

The paraxial carrier (Sziklas-Siegman) references a beam's wavefront to a
quadratic PARABOLA ``exp(i k r^2/2R)``.  On a strongly-converging FINAL leg
(NA -> 0.5) the true wavefront is the EXACT sphere ``S(R) =
sign(R)(sqrt(r^2+R^2)-|R|)``, and ``parabola - sphere`` reaches HUNDREDS of
radians of r^4 at the beam edge; re-enveloping with the paraxial ABCD ``R_out``
dumps that r^4 onto the paraxial envelope, which paraxial carrier propagation
CANNOT focus (only a few % encircled energy).

:func:`carrier_referenced_exact_focus_readout` fixes it: reference the field to
its own EXACT sphere (the two steep phases alias-cancel pointwise on the shared
grid, leaving a genuinely smooth envelope), resample that envelope onto a grid
that Nyquist-samples the exact sphere, reconstruct, and propagate to the image
with the EXISTING exact band-limited angular-spectrum Bluestein zoom
(:func:`angular_spectrum_propagate_mft`) -- no paraxial magnification/curvature.

These tests are SELF-CONTAINED (no Zemax): a clean converging EXACT-sphere
Gaussian at NA 0.30-0.46 is built on a COARSE grid that UNDER-samples the exit
sphere (reproducing the design-121 co-moving-grid condition), then focused

  * FAIL-BEFORE via the paraxial carrier path
    (:func:`carrier_referenced_focus_readout`) -> only a few % EE,
  * PASS-AFTER via the exact leg -> diffraction-limited (matches a fully
    resolved fine-grid ground truth), FWHM ~ lambda/(2 NA).
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy import (
    carrier_referenced_envelope,
    carrier_referenced_exact_focus_readout,
    carrier_referenced_focus_readout,
)
from lumenairy.propagators.mft import angular_spectrum_propagate_mft

_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL


def _skip_if_low_ram(n):
    try:
        import psutil
        if psutil.virtual_memory().available < 2 * (n * n * 16) + (1 << 30):
            pytest.skip(f'insufficient free RAM for the N={n} case')
    except Exception:
        pass


def _sphere(shape, dx, R):
    n = shape[-1]
    x = (np.arange(n) - n // 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    return np.sign(R) * (np.sqrt(r2 + R * R) - abs(R))


def _converging_exact_sphere(N, dx, w, R):
    """A converging Gaussian with an EXACT (non-paraxial) spherical wavefront."""
    S = _sphere((N, N), dx, R)
    x = (np.arange(N) - N // 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    return (np.exp(-r2 / w ** 2) * np.exp(1j * _K0 * S)).astype(np.complex128)


def _gauss_focus_waist(w, R):
    q = 1.0 / (1.0 / R - 1j * _WL / (np.pi * w ** 2))
    q_f = q + (-R)                                   # at the geometric focus
    return np.sqrt(-_WL / (np.pi * np.imag(1.0 / q_f)))


def _spot(E, dxo):
    I = np.abs(E) ** 2
    P = float(I.sum())
    n = I.shape[0]
    j, i = np.unravel_index(np.argmax(I), I.shape)
    xx = (np.arange(n) - i) * dxo
    yy = (np.arange(n) - j) * dxo
    rr = np.hypot(xx[None, :], yy[:, None])
    nb = int(0.45 * n)
    ring = np.clip((rr / dxo).astype(int), 0, nb)
    s = np.bincount(ring.ravel(), weights=I.ravel(), minlength=nb + 1)
    c = np.bincount(ring.ravel(), minlength=nb + 1)
    prof = (s[:nb] / np.maximum(c[:nb], 1)) / I[j, i]
    rb = (np.arange(nb) + 0.5) * dxo
    idx = np.where(prof < 0.5)[0]
    fwhm = 2.0 * rb[idx[0]] if len(idx) else np.nan
    offax = float(np.hypot((i - n / 2) * dxo, (j - n / 2) * dxo))
    return P, fwhm, offax, rr


def _ee_within(E, dxo, radius):
    P, _, _, rr = _spot(E, dxo)
    I = np.abs(E) ** 2
    return float(I[rr <= radius].sum()) / P


# ===========================================================================
# Core: exact leg focuses a high-NA converging sphere; paraxial fails
# ===========================================================================
@pytest.mark.parametrize('w, R', [
    (0.30e-3, -1.00e-3),     # NA = 0.30
    (0.35e-3, -0.77e-3),     # NA = 0.455 (the design-121 f/1.1 image leg)
])
def test_r9_exact_leg_focuses_highna_sphere(w, R):
    """PASS-AFTER: the exact leg focuses an NA 0.3-0.46 converging sphere to the
    diffraction limit; FAIL-BEFORE: the paraxial carrier path gives ~5%."""
    _skip_if_low_ram(2048)
    NA = w / abs(R)
    N, dx = 512, 3.0e-6
    zf = -R
    E = _converging_exact_sphere(N, dx, w, R)
    # the coarse grid UNDER-samples the exact exit sphere (the 121 condition)
    grad_px = _K0 * (w / np.sqrt(w * w + R * R)) * dx
    assert grad_px > np.pi, f'test setup must alias the exit sphere ({grad_px:.2f} rad/px)'

    w0 = _gauss_focus_waist(w, R)
    dx_out = w0 / 8.0
    N_out = 1024

    # D3 (2026-08-06): the PARAXIAL arm is the deliberate fail-before
    # reference and is read on the same wide common grid as the exact arm so
    # the two EE numbers are comparable.  Its Bluestein period is ~12 um
    # against a 178 um window, but every metric taken from it (EE inside
    # 2 w0 = 2.8 um, and the peak) sits well inside +/- period/2, so the
    # replicas in the outer window are not read.  Waived here, not silenced
    # globally; the default refusal is pinned in
    # test_niche_tight_focus_readout.
    E_par = np.asarray(carrier_referenced_focus_readout(
        carrier_referenced_envelope(E, R, _WL, dx), R, zf, _WL, dx,
        dx_out=dx_out, N_out=N_out, on_replica='ignore'))
    E_ex = np.asarray(carrier_referenced_exact_focus_readout(
        E, R, zf, _WL, dx, dx_out=dx_out, N_out=N_out))
    assert np.isfinite(E_ex).all()

    ee_par = _ee_within(E_par, dx_out, 2.0 * w0)
    ee_ex = _ee_within(E_ex, dx_out, 2.0 * w0)
    P, fwhm, offax, _ = _spot(E_ex, dx_out)

    # FAIL-BEFORE: paraxial carrier cannot focus this leg
    assert ee_par < 0.10, f'paraxial should fail (EE={ee_par:.3f})'
    # PASS-AFTER: exact leg reaches the diffraction limit
    assert ee_ex > 0.97, f'exact leg EE-in-2w0 = {ee_ex:.3f} (< 0.97)'
    assert ee_ex > 12.0 * ee_par, (ee_ex, ee_par)
    # FWHM near the diffraction limit lambda/(2 NA) (loose band -- a Gaussian's
    # FWHM is not exactly lambda/(2 NA); the precise match to a fully-resolved
    # focus is pinned by test_r9_exact_leg_matches_resolved_ground_truth).
    dl = _WL / (2.0 * NA)
    assert abs(fwhm - dl) / dl < 0.20, (fwhm, dl, NA)
    assert offax < 2.0 * dx_out, f'focus must be on-axis (offax={offax*1e6:.3f}um)'


def test_r9_exact_leg_matches_resolved_ground_truth():
    """The exact leg -- fed a COARSE, aliased converging sphere -- reproduces the
    fully-resolved fine-grid ASM focus (built directly, no aliasing) to a few
    %.  So the sphere-referencing genuinely de-aliases the field before the
    exact Bluestein zoom."""
    _skip_if_low_ram(2048)
    w, R = 0.35e-3, -0.77e-3
    N, dx, zf = 512, 3.0e-6, -R
    w0 = _gauss_focus_waist(w, R)
    dx_out, N_out = w0 / 8.0, 1024

    E_coarse = _converging_exact_sphere(N, dx, w, R)
    E_ex = np.asarray(carrier_referenced_exact_focus_readout(
        E_coarse, R, zf, _WL, dx, dx_out=dx_out, N_out=N_out,
        dx_fine=0.85e-6, N_fine=2048))

    # ground truth: build the sphere on the fine grid directly (no coarse alias)
    Nf, dxf = 2048, 0.85e-6
    E_gt_in = _converging_exact_sphere(Nf, dxf, w, R)
    E_gt = np.asarray(angular_spectrum_propagate_mft(
        E_gt_in, zf, _WL, dxf, dx_out, N_out, bandlimit=True))

    ee_ex = _ee_within(E_ex, dx_out, 2.0 * w0)
    ee_gt = _ee_within(E_gt, dx_out, 2.0 * w0)
    _, fw_ex, _, _ = _spot(E_ex, dx_out)
    _, fw_gt, _, _ = _spot(E_gt, dx_out)
    assert abs(ee_ex - ee_gt) < 0.03, (ee_ex, ee_gt)
    assert abs(fw_ex - fw_gt) / fw_gt < 0.05, (fw_ex, fw_gt)


def test_r9_exact_leg_collimated_matches_plain_mft():
    """A collimated carrier (R = +/-inf) has no sphere to reference -- the exact
    leg degenerates to a plain ASM-MFT of the input, byte-identical (to the FP
    floor) to calling :func:`angular_spectrum_propagate_mft` directly."""
    N, dx = 256, 4.0e-6
    x = (np.arange(N) - N // 2) * dx
    E = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / (0.3e-3) ** 2).astype(np.complex128)
    dx_out, N_out, z = 2.0e-6, 256, 20e-3
    E_ex = np.asarray(carrier_referenced_exact_focus_readout(
        E, np.inf, z, _WL, dx, dx_out=dx_out, N_out=N_out,
        dx_fine=dx, N_fine=N))                       # no upsample -> same grid
    E_ref = np.asarray(angular_spectrum_propagate_mft(
        E, z, _WL, dx, dx_out, N_out, bandlimit=True))
    rel = float(np.linalg.norm(E_ex - E_ref)) / (float(np.linalg.norm(E_ref)) + 1e-30)
    assert rel < 1e-9, f'collimated exact leg != plain MFT (rel={rel:.2e})'


# ===========================================================================
# Guards (the 4 dispatcher meta-audits require a guard-first 2-D scalar entry)
# ===========================================================================
def test_r9_exact_leg_guards_non_2d_input():
    """A non-2-D input is rejected by the shared ``_check_2d_scalar_field``
    first-line guard (dispatcher meta-audit contract)."""
    bad = np.ones(16, dtype=np.complex128)           # 1-D
    with pytest.raises((TypeError, ValueError)):
        carrier_referenced_exact_focus_readout(
            bad, -1e-3, 1e-3, _WL, 3e-6, dx_out=1e-6, N_out=64)


def test_r9_exact_leg_rejects_bad_args():
    """Guard the readout's grid / distance inputs."""
    E = _converging_exact_sphere(128, 4e-6, 0.2e-3, -1e-3)
    for kw in (dict(dx_out=-1.0, N_out=64),
               dict(dx_out=1e-6, N_out=0),
               dict(dx_out=1e-6, N_out=64, dx_fine=-1.0)):
        with pytest.raises(ValueError):
            carrier_referenced_exact_focus_readout(E, -1e-3, 1e-3, _WL, 4e-6, **kw)
    with pytest.raises(ValueError):
        carrier_referenced_exact_focus_readout(
            E, -1e-3, np.inf, _WL, 4e-6, dx_out=1e-6, N_out=64)


# ===========================================================================
# Orchestrator wiring: final_leg auto/exact/paraxial
# ===========================================================================
def _singlet(R1, R2, d, glass, ap, name='s'):
    gb, ga = ['air', glass], [glass, 'air']
    surfaces = [
        {'radius': R1, 'glass_before': gb[0], 'glass_after': ga[0],
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
        {'radius': R2, 'glass_before': gb[1], 'glass_after': ga[1],
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]
    return {'name': name, 'aperture_diameter': ap,
            'surfaces': surfaces, 'thicknesses': [d]}


def test_r9_orchestrator_auto_lowNA_byte_identical_to_paraxial():
    """final_leg='auto' on a LOW-NA final leg takes the paraxial path and is
    byte-identical (tolerance-pinned) to final_leg='paraxial' -- the default is
    unchanged from prior releases."""
    _skip_if_low_ram(1024)
    N, dx, w0 = 1024, 3e-6, 1.2e-3
    x = (np.arange(N) - N / 2) * dx
    env0 = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / w0 ** 2).astype(np.complex128)
    groups = [{'prescription': _singlet(60e-3, -60e-3, 4e-3, 'N-BK7', 6e-3, 'g1'),
               'gap_before': 0.0}]
    fr = dict(dx_out=0.2e-6, N_out=256, standoff=1.0e-3)
    R_out = la.propagators.carrier._paraxial_group_r_out(
        groups[0]['prescription'], np.inf, _WL)
    final = -R_out
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        res_auto = la.propagate_traced_carrier_chain(
            env0, groups, _WL, dx, r_in=np.inf, ray_subsample=4, n_workers=1,
            traced_kwargs=dict(parallel_amp=False), final_distance=final,
            focus_readout=fr, final_leg='auto')
        res_par = la.propagate_traced_carrier_chain(
            env0, groups, _WL, dx, r_in=np.inf, ray_subsample=4, n_workers=1,
            traced_kwargs=dict(parallel_amp=False), final_distance=final,
            focus_readout=fr, final_leg='paraxial')
    A, B = np.asarray(res_auto.field), np.asarray(res_par.field)
    assert A.shape == B.shape
    denom = float(np.linalg.norm(B)) + 1e-30
    assert float(np.linalg.norm(A - B)) / denom < 1e-12, 'auto != paraxial at low NA'
    assert not res_auto.stages[-1].get('exact_final', False)


def test_r9_orchestrator_exact_final_leg_runs_and_flags():
    """final_leg='exact' re-traces the final (high-NA) group on a fine grid and
    lands via the exact readout: it returns a finite field on the (N_out) focal
    grid, R is None, and the final stage is flagged exact_final.  'auto' selects
    it because the exit NA exceeds the threshold."""
    _skip_if_low_ram(2048)
    N, dx, w = 256, 10e-6, 0.6e-3
    x = (np.arange(N) - N / 2) * dx
    env0 = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / w ** 2).astype(np.complex128)
    # a strong biconvex singlet -> short EFL -> exit NA ~ 0.2 (> threshold)
    groups = [{'prescription': _singlet(3.1e-3, -3.1e-3, 1e-3, 'N-BK7', 2.4e-3, 'strong'),
               'gap_before': 0.0}]
    R_out = la.propagators.carrier._paraxial_group_r_out(
        groups[0]['prescription'], np.inf, _WL)
    na_exit = w / abs(R_out)
    assert na_exit > 0.15, f'test group must be high-NA (NA={na_exit:.3f})'
    fr = dict(dx_out=0.2e-6, N_out=256)
    tk = dict(parallel_amp=False, on_undersample='silent', on_noncollimated='silent')
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        res = la.propagate_traced_carrier_chain(
            env0, groups, _WL, dx, r_in=np.inf, ray_subsample=4, n_workers=1,
            traced_kwargs=tk, final_distance=-R_out, focus_readout=fr,
            final_leg='exact')
    E = np.asarray(res.field)
    assert E.shape == (256, 256)
    assert np.isfinite(E).all() and float(np.abs(E).max()) > 0.0
    assert res.R is None and res.dx == 0.2e-6
    assert res.stages[-1].get('exact_final', False) is True
    # 'auto' must pick the exact path for this NA (dx_fine sub-Nyquist of the
    # exit sphere) -- same exact_final flag.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        res_auto = la.propagate_traced_carrier_chain(
            env0, groups, _WL, dx, r_in=np.inf, ray_subsample=4, n_workers=1,
            traced_kwargs=tk, final_distance=-R_out, focus_readout=fr,
            final_leg='auto')
    assert res_auto.stages[-1].get('exact_final', False) is True


def test_r9_orchestrator_rejects_bad_final_leg():
    env0 = np.ones((32, 32), dtype=np.complex128)
    with pytest.raises(ValueError):
        la.propagate_traced_carrier_chain(env0, [], _WL, 3e-6, final_leg='nope')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
