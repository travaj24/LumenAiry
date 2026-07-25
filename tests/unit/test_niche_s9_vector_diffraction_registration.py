"""Regression pins for the S9 (review4a) vector-diffraction findings in
``lumenairy/propagators/vector_diffraction.py``.  Both are instances of the
patterns closed in the traced-carrier campaign.

S9-VD1 -- PATTERN #1 (interpolation / registration anchor).
    ``richards_wolf_focus`` pads (or crops) the pupil to ``N_focal`` and then
    runs ``fftshift(fft2(ifftshift(padded)))``.  ``ifftshift`` treats index
    ``L//2`` (FLOOR) as the array centre, so preserving the pupil's
    registration requires ``Np//2 -> N_focal//2``, i.e.
    ``pad = N_focal//2 - Np//2`` / ``c0 = Np//2 - N_focal//2``.  The old code
    used ``(N_focal - Np)//2`` / ``(Np - N_focal)//2``, which is the SAME
    integer for every same-parity pair but off by exactly one index when the
    parities differ -- (Np odd, N_focal even) on the pad branch and
    (Np even, N_focal odd) on the crop branch.  The one-index slip translates
    the whole masked+apodised pupil by one pupil pixel, giving the returned
    COMPLEX focal field a spurious linear phase of ``2*pi/N_focal`` rad per
    focal pixel (measured 0.098175 rad/px at (33, 64) and 0.369599 rad/px at
    (32, 17), matching ``2*pi*delta/N_focal`` to 6 digits; the two fields
    differ by 145% in L2).  Intensity (``debye_wolf_psf``) is blind to it;
    coherent superposition with a reference arm is not.

S9-VD2 -- PATTERN #3 (degenerate resample/crop-branch contract violation).
    ``N_focal < Np`` centre-CROPS the PUPIL, so a smaller focal grid silently
    truncates the APERTURE (and the effective NA), not just the focal
    sampling -- and it buys nothing, because the FFT-natural focal window
    ``N_focal*dx_focal = wavelength*f/dx_pupil`` does not depend on
    ``N_focal``.  Measured on NA=0.5 / f=4 mm / dx_pupil=20 um / Np=256
    (aperture radius 100 px): focal Parseval energy 0.504 / 0.279 / 0.123 of
    the full-pupil value at N_focal = 128 / 96 / 64, vs the predicted
    clipped-pupil area fractions 0.522 / 0.294 / 0.130.  The values are
    unchanged; the breach is now surfaced as a ``RuntimeWarning``.

CI-safe: tiny analytic pupils, no external assets, no optional backends.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.propagators.vector_diffraction import (
    debye_wolf_psf,
    richards_wolf_focus,
)

_LAM, _NA, _F, _DXP = 500e-9, 0.4, 4.0e-3, 20e-6


def _delta_pupil(Np: int) -> np.ndarray:
    """A single unit sample at the array's TRANSFORM centre ``Np//2``.

    Its transform is a constant (zero linear phase) field iff the pad/crop
    keeps that sample on the padded array's transform centre; a displacement
    of ``d`` indices shows up as a pure ``2*pi*d/N_focal`` rad/px ramp.
    """
    p = np.zeros((Np, Np), dtype=np.complex128)
    p[Np // 2, Np // 2] = 1.0
    return p


def _mean_phase_step(row: np.ndarray) -> float:
    """Mean phase increment per sample along ``row`` (robust to wrapping)."""
    return float(np.mean(np.angle(row[1:] * np.conj(row[:-1]))))


# ---------------------------------------------------------------------------
# S9-VD1: pad / crop DC registration
# ---------------------------------------------------------------------------

# (Np, N_focal): the first four are the mismatched-parity cases the old
# ``(N_focal - Np)//2`` / ``(Np - N_focal)//2`` offsets got wrong; the rest are
# same-parity controls that were already correct (and stay bit-identical).
@pytest.mark.parametrize('Np,Nf', [
    (33, 64), (33, 66), (35, 64),          # pad branch, Np odd  / Nf even
    (32, 17), (32, 19), (64, 33),          # crop branch, Np even / Nf odd
    (32, 64), (33, 65), (32, 32), (33, 33),  # same-parity controls
    (64, 32), (65, 33), (33, 16), (32, 16),
])
def test_pad_crop_preserves_transform_centre(Np, Nf):
    """A delta at the pupil's transform centre must produce a focal field with
    ZERO linear phase, for every (Np, N_focal) parity combination."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Ex, Ey, Ez, xf, yf = richards_wolf_focus(
            _delta_pupil(Np), _LAM, _NA, _F, _DXP, N_focal=Nf,
            polarization='x')
    Ex = np.asarray(Ex)
    assert Ex.shape == (Nf, Nf)
    c = Nf // 2
    # magnitude must be flat (a delta transforms to a constant)
    a = np.abs(Ex)
    assert a.max() > 0.0
    assert np.ptp(a) <= 1e-9 * a.max(), 'delta pupil must give a flat |E|'
    # ... and the phase must not ramp.  The pre-fix slip is exactly
    # 2*pi/Nf rad/px on the mismatched-parity cases; gate at 1/8 of that.
    gate = 0.125 * 2.0 * np.pi / Nf
    sx = _mean_phase_step(Ex[c, :])
    sy = _mean_phase_step(Ex[:, c])
    assert abs(sx) < gate, (Np, Nf, 'x ramp', sx, 2 * np.pi / Nf)
    assert abs(sy) < gate, (Np, Nf, 'y ramp', sy, 2 * np.pi / Nf)


def test_pad_crop_offsets_are_dc_preserving_integers():
    """White-box companion: the offsets used must equal ``Nf//2 - Np//2`` /
    ``Np//2 - Nf//2``, and must coincide with the legacy expressions for every
    same-parity pair (the bit-identity guarantee of the fix)."""
    for Np in range(16, 40):
        for Nf in range(16, 40):
            if Nf >= Np:
                need, legacy = Nf // 2 - Np // 2, (Nf - Np) // 2
                assert 0 <= need <= Nf - Np          # np.pad stays valid
            else:
                need, legacy = Np // 2 - Nf // 2, (Np - Nf) // 2
                assert 0 <= need and need + Nf <= Np  # crop stays in bounds
            if (Np % 2) == (Nf % 2):
                assert need == legacy, (Np, Nf, need, legacy)


def test_mismatched_parity_ramp_would_be_detected():
    """Sensitivity check: the SAME probe applied to the legacy offsets must
    show the ``2*pi/Nf`` rad/px ramp, so the pin above cannot pass vacuously."""
    for Np, Nf in [(33, 64), (32, 17)]:
        A = _delta_pupil(Np)
        if Nf >= Np:
            legacy = (Nf - Np) // 2
            B = np.zeros((Nf, Nf), dtype=np.complex128)
            B[legacy:legacy + Np, legacy:legacy + Np] = A
        else:
            legacy = (Np - Nf) // 2
            B = A[legacy:legacy + Nf, legacy:legacy + Nf]
        F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(B)))
        step = abs(_mean_phase_step(F[Nf // 2, :]))
        assert step == pytest.approx(2.0 * np.pi / Nf, rel=1e-9), (Np, Nf, step)


# ---------------------------------------------------------------------------
# S9-VD2: N_focal < Np silently clips the aperture
# ---------------------------------------------------------------------------

def _filled_pupil(Np, r_frac=0.78):
    """Circular pupil of radius ``r_frac * Np/2`` px (fills most of the grid)."""
    i = (np.arange(Np) - Np / 2) * _DXP
    R = np.hypot(*np.meshgrid(i, i))
    return (R <= r_frac * (Np / 2) * _DXP).astype(np.complex128)


def test_pupil_clipping_crop_warns_with_measured_loss():
    """``N_focal < Np`` on a filled pupil must warn, name the mechanism, and
    quote a non-trivial discarded-energy fraction."""
    Np = 64
    pupil = _filled_pupil(Np)
    with pytest.warns(RuntimeWarning, match='centre-CROPPED'):
        debye_wolf_psf(pupil, _LAM, _NA, _F, _DXP, N_focal=Np // 2)
    with pytest.warns(RuntimeWarning, match='effective NA'):
        debye_wolf_psf(pupil, _LAM, _NA, _F, _DXP, N_focal=Np // 2)


def test_no_warning_when_crop_discards_nothing():
    """A pupil whose support fits INSIDE the crop loses no energy, so the
    guard must stay silent (no false positives on zero-padded pupils)."""
    Np = 64
    pupil = _filled_pupil(Np, r_frac=0.40)      # support radius ~12.8 px
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        debye_wolf_psf(pupil, _LAM, _NA, _F, _DXP, N_focal=32)


def test_no_warning_on_the_default_and_padding_paths():
    Np = 32
    pupil = _filled_pupil(Np)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        debye_wolf_psf(pupil, _LAM, _NA, _F, _DXP)            # N_focal=None
        debye_wolf_psf(pupil, _LAM, _NA, _F, _DXP, N_focal=Np)
        debye_wolf_psf(pupil, _LAM, _NA, _F, _DXP, N_focal=2 * Np)


def test_crop_energy_loss_matches_clipped_pupil_prediction():
    """The physics claim behind the warning: the focal-plane Parseval energy
    of the cropped run tracks the pupil energy the crop RETAINS.  (The focal
    window extent ``N_focal*dx_focal = lam*f/dx_pupil`` is independent of
    ``N_focal``, so the comparison is apples-to-apples.)"""
    Np = 128
    pupil = _filled_pupil(Np, r_frac=0.78)      # radius ~50 px
    ref = None
    win = None
    for Nf in (Np, 96, 64, 48):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            psf, xf, yf = debye_wolf_psf(pupil, _LAM, _NA, _F, _DXP,
                                         N_focal=Nf)
        dxf = float(xf[1] - xf[0])
        # same physical focal window for every Nf -- the crop buys no FOV
        if win is None:
            win = Nf * dxf
        else:
            assert Nf * dxf == pytest.approx(win, rel=1e-12)
        E = float(psf.sum()) * dxf * dxf
        # predicted: fraction of the (apodisation-free) pupil energy kept
        c0 = Np // 2 - Nf // 2
        keep = (float(np.sum(np.abs(pupil[c0:c0 + Nf, c0:c0 + Nf]) ** 2))
                / float(np.sum(np.abs(pupil) ** 2)))
        if ref is None:
            ref = E
            assert keep == pytest.approx(1.0)
            continue
        assert E / ref == pytest.approx(keep, rel=0.10), (Nf, E / ref, keep)
        if Nf <= 64:                            # the crop really does clip
            assert keep < 0.60, (Nf, keep)
            assert E / ref < 0.60, (Nf, E / ref)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
