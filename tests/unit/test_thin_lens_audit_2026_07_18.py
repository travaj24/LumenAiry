"""Thin-lens audit 2026-07-18 (docs/audit_asm_thinlens_focus_2026_07_18.md).

Regression suite for the two sign bugs + the two new capabilities:

Bug 1  -- ``lens_model='nonparaxial'`` with f < 0 had the WRONG sign:
          ``exp(1j*k*(f - sqrt(f**2+r**2)))`` expands to a CONVERGING
          quadratic ``-k r**2/(2|f|)`` for f < 0, so a diverging lens
          focused identically to its +|f| twin.
Bug 2  -- ``lens_model='aplanatic'`` had the WRONG quartic sign:
          ``-k*f*(1-sqrt(1-r**2/f**2)) ~ -k r**2/2f - k r**4/8f**3``
          where a converging sphere needs ``+k r**4/8f**3`` -- it
          DOUBLED paraxial's spherical aberration (focused WORSE).
Change 1 -- ``lens_model='stigmatic'``: conjugate-matched exact element
          ``phi = k*(S(R_out)-S(R_in))``, aberration-free under exact
          ASM at ANY conjugates.
Change 3 -- ``fresnel_tf_propagate``: same-grid Fresnel transfer-function
          step (z < 0 allowed) -- the matched-paraxial propagator that
          makes (paraxial lens x propagator) self-consistent, mirroring
          Zemax POP's pilot-beam convention.

Oracles are INDEPENDENT of the implementation: closed-form Taylor
coefficients of the spherical-wave phase, conjugate-symmetry invariants
(phi(-f) == -phi(+f)), and ABCD Gaussian-beam theory for the end-to-end
imaging checks.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy.propagators.fresnel import fresnel_tf_propagate

_WL = 1.0e-6
_K = 2.0 * np.pi / _WL


def _phase_grid(N, dx, f, model, **kw):
    E = np.ones((N, N), dtype=np.complex128)
    out = la.apply_thin_lens(E, f=f, wavelength=_WL, dx=dx,
                             lens_model=model, **kw)
    return out          # == the lens multiplier itself for unit input


def _second_moment_radius(I, dx, crop=None):
    """Intensity-weighted RMS radius; ``crop`` restricts to a centred
    (2*crop x 2*crop) window so the metric reads the FOCAL REGION rather
    than being dominated by the r^2-weighted faint outer field (a
    full-grid second moment weights a 1e-6-relative background at
    millimetre radii above a micron-scale spot)."""
    N = I.shape[0]
    if crop is not None:
        c = N // 2
        I = I[c - crop:c + crop, c - crop:c + crop]
        N = 2 * crop
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    tot = I.sum()
    xc = (I * X).sum() / tot
    yc = (I * Y).sum() / tot
    return np.sqrt(((I * ((X - xc) ** 2 + (Y - yc) ** 2)).sum() / tot))


# ---------------------------------------------------------------------------
# Bug 1 -- nonparaxial f < 0
# ---------------------------------------------------------------------------

def test_nonparaxial_f_positive_byte_identical_to_historical():
    """For f > 0 the corrected form is BYTE-identical to the historical
    ``exp(1j*k*(f - sqrt(f**2 + r**2)))`` -- no existing converging
    result changes."""
    N, dx, f = 256, 8e-6, 30e-3
    out = _phase_grid(N, dx, f, 'nonparaxial')
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r_sq = X ** 2 + Y ** 2
    hist = np.exp(1j * _K * (f - np.sqrt(f ** 2 + r_sq)))
    assert np.array_equal(out, hist)


def test_nonparaxial_negative_f_is_exact_conjugate_of_positive():
    """Physics invariant: an ideal diverging element is the exact phase
    CONJUGATE of its converging twin, phi(-f) == -phi(+f).  Pre-fix the
    two were EQUAL (f = -30 mm focused like f = +30 mm)."""
    N, dx = 256, 8e-6
    plus = _phase_grid(N, dx, +30e-3, 'nonparaxial')
    minus = _phase_grid(N, dx, -30e-3, 'nonparaxial')
    assert np.array_equal(minus, np.conj(plus))
    # decisive against the pre-fix behaviour:
    assert not np.allclose(minus, plus)


def test_nonparaxial_negative_f_does_not_focus_end_to_end():
    """The audit's mini-test in miniature: a collimated Gaussian through
    f = +15 mm forms a focus at z = 15 mm (huge on-axis peak); through
    f = -15 mm it must DIVERGE (peak far below even the unfocused
    input).  Pre-fix the two peaks were IDENTICAL."""
    N, dx = 512, 8e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / (0.8e-3) ** 2).astype(np.complex128)
    z = 15e-3
    peaks = {}
    for f in (+15e-3, -15e-3):
        E = la.apply_thin_lens(E0, f=f, wavelength=_WL, dx=dx,
                               lens_model='nonparaxial')
        Ez = la.angular_spectrum_propagate(E, z, _WL, dx)
        peaks[f] = float(np.max(np.abs(Ez) ** 2))
    assert peaks[+15e-3] > 50.0 * peaks[-15e-3], (
        f"f=-15mm must not focus like f=+15mm: peaks {peaks}")


# ---------------------------------------------------------------------------
# Bug 2 -- aplanatic quartic sign
# ---------------------------------------------------------------------------

def test_aplanatic_quartic_sign_is_positive():
    """Independent Taylor oracle: after removing the analytic quadratic
    ``-k r**2/2f``, the residual phase must be ``+k r**4/8f**3``
    (converging-sphere sign).  Pre-fix it was ``-k r**4/8f**3``."""
    N, dx, f = 512, 8e-6, 30e-3
    out = _phase_grid(N, dx, f, 'aplanatic')
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r_sq = X ** 2 + Y ** 2
    # remove the exact quadratic analytically; remainder is small (<1.2 rad)
    resid = np.angle(out * np.exp(1j * _K / (2 * f) * r_sq))
    r = np.sqrt(r_sq)
    band = (r > 1.2e-3) & (r < 1.8e-3)          # far enough out to resolve r^4
    expected = _K * r_sq[band] ** 2 / (8 * f ** 3)
    ratio = resid[band] / expected
    assert np.all(ratio > 0), "quartic residual must be POSITIVE (pre-fix: negative)"
    assert abs(np.median(ratio) - 1.0) < 0.05, (
        f"quartic residual must match +k r^4/8f^3; median ratio {np.median(ratio):.4f}")


def test_aplanatic_equals_stigmatic_sphere_inside_domain():
    """The corrected aplanatic phase IS the stigmatic sphere on r < |f|
    (unit phase outside -- the pinned v4.10 rim contract elsewhere)."""
    N, dx, f = 256, 8e-6, 0.8e-3          # small f so the domain edge is on-grid
    apl = _phase_grid(N, dx, f, 'aplanatic')
    non = _phase_grid(N, dx, f, 'nonparaxial')
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r_sq = X ** 2 + Y ** 2
    inside = r_sq / f ** 2 < 1.0
    assert np.array_equal(apl[inside], non[inside])
    assert np.array_equal(apl[~inside],
                          np.ones_like(apl[~inside]))


def test_aplanatic_no_longer_focuses_worse_than_paraxial():
    """The audit's NA~0.1 mini-test in miniature: pre-fix the aplanatic
    focal spot (9.1 um) was WORSE than paraxial (7.7 um); the corrected
    model must match nonparaxial (4.27 um class) and clearly beat
    paraxial."""
    N, dx = 1024, 4e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / (0.8e-3) ** 2).astype(np.complex128)
    f = 8e-3                                    # NA ~ w/f = 0.1
    radii = {}
    for model in ('paraxial', 'nonparaxial', 'aplanatic'):
        E = la.apply_thin_lens(E0, f=f, wavelength=_WL, dx=dx,
                               lens_model=model)
        Ez = la.angular_spectrum_propagate(E, f, _WL, dx)
        # focal-window metric (crop=64 -> +/-256 um around the focus);
        # measured: paraxial 3.378 um, nonparaxial == aplanatic 2.157 um
        radii[model] = _second_moment_radius(np.abs(Ez) ** 2, dx, crop=64)
    assert radii['aplanatic'] < 1.02 * radii['nonparaxial'], radii
    assert radii['aplanatic'] < 0.85 * radii['paraxial'], (
        f"corrected aplanatic must clearly beat paraxial: {radii}")


# ---------------------------------------------------------------------------
# Change 1 -- stigmatic model
# ---------------------------------------------------------------------------

def test_stigmatic_collimated_reduces_to_nonparaxial():
    """R_in = inf (default) must reduce EXACTLY to the corrected
    nonparaxial sphere -- both converging and diverging."""
    N, dx = 256, 8e-6
    for f in (+25e-3, -25e-3):
        stig = _phase_grid(N, dx, f, 'stigmatic')
        non = _phase_grid(N, dx, f, 'nonparaxial')
        assert np.array_equal(stig, non), f"f={f}"


def test_stigmatic_quartic_matches_conjugate_formula():
    """Independent Taylor oracle at FINITE conjugates: quadratic part is
    the lens equation -k r**2/2f; quartic part is
    ``-k r**4/8 * (1/R_out**3 - 1/R_in**3)`` -- the term 'paraxial'
    omits and 'nonparaxial' only nulls at infinite conjugates."""
    N, dx = 512, 8e-6
    f, R_in = 25e-3, 50e-3                       # 1/R_out = 1/50 - 1/25 => R_out = -50 mm
    R_out = 1.0 / (1.0 / R_in - 1.0 / f)
    out = _phase_grid(N, dx, f, 'stigmatic', conjugates=R_in)
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r_sq = X ** 2 + Y ** 2
    resid = np.angle(out * np.exp(1j * _K / (2 * f) * r_sq))
    r = np.sqrt(r_sq)
    band = (r > 1.2e-3) & (r < 1.8e-3)
    expected = -_K * r_sq[band] ** 2 / 8.0 * (1.0 / R_out ** 3 - 1.0 / R_in ** 3)
    ratio = resid[band] / expected
    assert abs(np.median(ratio) - 1.0) < 0.05, (
        f"stigmatic quartic must match the conjugate formula; "
        f"median ratio {np.median(ratio):.4f}")


def test_stigmatic_conjugates_guard():
    E = np.ones((64, 64), dtype=np.complex128)
    with pytest.raises(ValueError, match="only meaningful"):
        la.apply_thin_lens(E, f=0.03, wavelength=_WL, dx=8e-6,
                           lens_model='paraxial', conjugates=0.05)
    with pytest.raises(ValueError, match="nonzero"):
        la.apply_thin_lens(E, f=0.03, wavelength=_WL, dx=8e-6,
                           lens_model='stigmatic', conjugates=0.0)


def test_stigmatic_finite_conjugate_imaging_beats_paraxial_under_asm():
    """The audit's core claim end-to-end: at finite conjugates under the
    EXACT (ASM) propagator, the stigmatic element images a Gaussian at
    the ABCD-predicted waist while the paraxial element carries the
    uncompensated ``k r**4/8 (1/R_in^3 - 1/R_out^3)`` error.

    1:1 imaging, w0 = 4 um at s = 15 mm, f = 7.5 mm, lambda = 1 um:
    the ABCD image waist is w0 (unit magnification); the paraxial
    model's residual at the 1/e^2 ray is ~0.9 rad (14+ rad at 2w),
    which measurably broadens the second-moment spot."""
    N, dx = 4096, 1.2e-6
    w0, s, f = 4e-6, 15e-3, 7.5e-3
    zR = np.pi * w0 ** 2 / _WL
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)
    E_lens = la.angular_spectrum_propagate(E0, s, _WL, dx)
    # Gaussian wavefront radius at the lens (R > 0 diverging)
    R_in = s * (1.0 + (zR / s) ** 2)
    results = {}
    for model, kw in (('stigmatic', {'conjugates': R_in}),
                      ('paraxial', {})):
        E1 = la.apply_thin_lens(E_lens, f=f, wavelength=_WL, dx=dx,
                                lens_model=model, **kw)
        E_img = la.angular_spectrum_propagate(E1, s, _WL, dx)
        # second-moment radius of a Gaussian == w/ sqrt(2); focal-window
        # metric (crop=64 -> +/-76.8 um) so the r^2-weighted faint outer
        # field does not swamp the micron-scale spot comparison.
        results[model] = _second_moment_radius(np.abs(E_img) ** 2, dx,
                                               crop=64)
    w0_r2m = w0 / np.sqrt(2.0)
    assert abs(results['stigmatic'] - w0_r2m) / w0_r2m < 0.10, (
        f"stigmatic image must sit at the ABCD waist: "
        f"r2m={results['stigmatic']*1e6:.3f} um vs {w0_r2m*1e6:.3f} um")
    assert results['paraxial'] > 1.5 * results['stigmatic'], (
        f"paraxial x exact-ASM must show the uncompensated spherical "
        f"broadening the audit measured: {results}")


# ---------------------------------------------------------------------------
# Change 3 -- fresnel_tf_propagate (matched-paraxial chain step)
# ---------------------------------------------------------------------------

def test_fresnel_tf_z0_identity_and_roundtrip():
    rng = np.random.default_rng(7)
    E = (rng.standard_normal((128, 128))
         + 1j * rng.standard_normal((128, 128)))
    assert np.array_equal(fresnel_tf_propagate(E, 0.0, _WL, 8e-6), E)
    fwd = fresnel_tf_propagate(E, 5e-3, _WL, 8e-6)
    back = fresnel_tf_propagate(fwd, -5e-3, _WL, 8e-6)
    assert np.max(np.abs(back - E)) < 1e-10      # z<0 is the exact inverse


def test_fresnel_tf_gaussian_matches_abcd():
    """Free-space Gaussian expansion vs the paraxial ABCD (q-parameter)
    oracle -- the Fresnel TF is EXACT for this."""
    N, dx = 1024, 4e-6
    w0, z = 200e-6, 200e-3
    zR = np.pi * w0 ** 2 / _WL
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)
    Ez = fresnel_tf_propagate(E0, z, _WL, dx)
    w_pred = w0 * np.sqrt(1.0 + (z / zR) ** 2)
    r2m = _second_moment_radius(np.abs(Ez) ** 2, dx)
    assert abs(r2m - w_pred / np.sqrt(2.0)) / (w_pred / np.sqrt(2.0)) < 0.01


def test_matched_paraxial_pair_images_at_abcd_waist():
    """The audit's change-3 contract: (paraxial lens x Fresnel-TF
    propagator) is self-consistent -- the SAME 1:1 imaging geometry
    where paraxial x exact-ASM broadens by >1.5x must image at the ABCD
    waist under the matched propagator (the Zemax-POP-equivalent
    reference mode)."""
    N, dx = 4096, 1.2e-6
    w0, s, f = 4e-6, 15e-3, 7.5e-3
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)
    E_lens = fresnel_tf_propagate(E0, s, _WL, dx)
    E1 = la.apply_thin_lens(E_lens, f=f, wavelength=_WL, dx=dx,
                            lens_model='paraxial')
    E_img = fresnel_tf_propagate(E1, s, _WL, dx)
    # focal-window metric: the unbandlimited TF kernel's tiny aliased
    # background (~1e-6 relative) at millimetre radii would otherwise
    # dominate the r^2-weighted full-grid moment (measured: full-grid
    # 19.8 um vs windowed 2.835 um against the 2.828 um ABCD oracle).
    r2m = _second_moment_radius(np.abs(E_img) ** 2, dx, crop=64)
    w0_r2m = w0 / np.sqrt(2.0)
    assert abs(r2m - w0_r2m) / w0_r2m < 0.10, (
        f"matched paraxial pair must image at the ABCD waist: "
        f"r2m={r2m*1e6:.3f} um vs {w0_r2m*1e6:.3f} um")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
