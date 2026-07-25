"""AUDIT_ADVERSARIAL_CODEBASE_2026_07_25 (P1): odd-N frequency-lattice pins
for the ASM / Fresnel propagator family.

The bug
-------
Every centred frequency grid in the family was built as
``(arange(N) - N/2) * df`` with a FLOAT anchor, then consumed through
``fftshift`` / ``ifftshift``.  Those permutations roll by whole samples
only -- they anchor DC at the INTEGER index ``N // 2``.  For even N the two
anchors coincide (``N // 2 == N / 2``) and the grids are identical; for ODD
N the float anchor labels every bin half a bin low, so ``ifftshift`` of the
grid yields ``f[0] = -df/2`` where the DFT's own bin 0 sits at exactly 0.
Every transfer function was therefore evaluated at ``f_true - df/2``, which
for the ASM / Fresnel kernels is a linear phase in f -- i.e. a lateral walk
of the propagated field by ``-lambda*z/(2*N*dx)``.

Measured before the fix (lambda = dx = 1 um, w0 = 20 um Gaussian, z = 2 mm,
exact Gaussian-ABCD oracle), N=257 vs N=256:

===============================  ==============  ==============
entry point                      N=257 rel err   N=256 rel err
===============================  ==============  ==============
angular_spectrum_propagate       2.593e-01       5.705e-05
fresnel_tf_propagate             2.592e-01       7.692e-06
angular_spectrum_propagate_mft   1.533e-01       5.705e-05
fresnel_propagate (single FFT)   1.719e-01       2.809e-16
fraunhofer_propagate             1.531e-01 [*]   2.787e-14 [*]
===============================  ==============  ==============

[*] vs ``fraunhofer_propagate_mft``, which has no far-field analytic
    Gaussian oracle on its own output grid.

with a -3.8916 px intensity-centroid walk at N=257 (closed form
``-lambda*z/(2*N*dx) = -3.8911 px``), and -8.0874 px of focal-spot walk in
the ``_build_asm_H_square`` Shack-Hartmann consumer path at Np=65.
``rayleigh_sommerfeld_propagate`` (even 2N pad) and
``fresnel_propagate_mft`` / ``fraunhofer_propagate_mft`` (explicit
``n - N/2`` Bluestein, no shifts) were already correct at odd N and act as
the cross-implementation discriminators here.

The fix
-------
1. Frequency lattices use the INTEGER anchor ``N // 2``, making them
   exactly ``fftshift(fftfreq(N, dx))`` for both parities
   (``fft_infra._get_or_make_freq_grids`` / ``_get_or_make_bandlimit``,
   ``asm._build_asm_H_square``, the ASM JAX H build, the tilted-ASM H,
   ``mft.angular_spectrum_propagate_mft``).
2. ``angular_spectrum_propagate_mft`` additionally carries the SAME anchor
   into its Bluestein ``n_centre_in_*`` (that argument is the frequency-bin
   centre) and folds the half-input-pixel gap between the ``ifftshift``
   origin (integer ``N // 2``) and this family's documented ``(n - N/2)``
   coordinate origin into ``k_centre_out_*``.
3. ``fresnel_propagate`` / ``fraunhofer_propagate`` keep their documented
   ``(n - N/2)`` grids and apply the exact shifted-DFT half-sample
   correction instead (``fresnel._centred_dft_halfpixel_args``), so they
   stay consistent with their MFT siblings.

SPATIAL centred grids elsewhere keep the float ``N/2`` convention: it is
this family's documented coordinate origin, the tilted-ASM carrier cancels
it exactly between demodulation and remodulation, and the discriminators
above evaluate it directly.

Even-N behaviour is bit-identical -- pinned below by recomputing the
pre-fix expression in-test.

Self-contained (analytic oracles + in-test fftfreq references only), ~10 s.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.propagators.asm import (
    _build_asm_H_square,
    angular_spectrum_propagate,
    angular_spectrum_propagate_tilted,
)
from lumenairy.propagators.fft_infra import (
    _get_or_make_bandlimit,
    _get_or_make_freq_grids,
)
from lumenairy.propagators.fresnel import (
    fraunhofer_propagate,
    fresnel_propagate,
    fresnel_tf_propagate,
)
from lumenairy.propagators.mft import (
    angular_spectrum_propagate_mft,
    fraunhofer_propagate_mft,
    fresnel_propagate_mft,
)

LAM = 1.0e-6
DX = 1.0e-6
W0 = 20.0e-6
Z = 2.0e-3
K = 2.0 * np.pi / LAM
ZR = np.pi * W0 ** 2 / LAM

# Grid sizes: paired even/odd so every odd-N assertion has an even-N twin
# at (nearly) the same window.  N=257 is the audit's measured case.
EVEN_N = (256, 512)
ODD_N = (257, 513)


# ---------------------------------------------------------------------------
# helpers -- the family's documented coordinate convention x = (n - N/2)*dx
# ---------------------------------------------------------------------------
def _axis(N, d=DX):
    return (np.arange(N) - N / 2) * d


def _gaussian(N, d=DX):
    x = _axis(N, d)
    X, Y = np.meshgrid(x, x, indexing='xy')
    return np.exp(-(X ** 2 + Y ** 2) / W0 ** 2).astype(np.complex128)


def _gaussian_abcd(N, z, d=DX):
    """Exact paraxial Gaussian-beam (ABCD / complex-q) oracle."""
    x = _axis(N, d)
    X, Y = np.meshgrid(x, x, indexing='xy')
    q0 = -1j * ZR
    q = z - 1j * ZR
    return (q0 / q) * np.exp(1j * K * z) * np.exp(1j * K * (X ** 2 + Y ** 2)
                                                  / (2 * q))


def _rel_err(got, ref):
    """Max |difference| over the support, normalised by the peak."""
    peak = np.abs(ref).max()
    mask = np.abs(ref) > 1e-5 * peak
    return float(np.max(np.abs(got[mask] - ref[mask]))) / peak


def _centroid_px(E, x):
    I = np.abs(E) ** 2
    prof = I.sum(axis=0)
    return float((prof * x).sum() / prof.sum()) / (x[1] - x[0])


def _d4sigma(E, x):
    I = np.abs(E) ** 2
    prof = I.sum(axis=0)
    tot = prof.sum()
    c = (prof * x).sum() / tot
    return 4.0 * float(np.sqrt((prof * (x - c) ** 2).sum() / tot))


# --- independent references built on the TRUE DFT bins (np.fft.fftfreq) ----
def _asm_reference(E, z, bandlimit):
    N = E.shape[0]
    fx = np.fft.fftfreq(N, DX)
    kx_sq = (2 * np.pi * fx) ** 2
    kz_sq = K ** 2 - kx_sq[None, :] - kx_sq[:, None]
    prop = kz_sq > 0
    H = np.where(prop, np.exp(1j * np.sqrt(np.where(prop, kz_sq, 0.0)) * z), 0)
    if bandlimit:
        f_max = N * DX / (2 * LAM * abs(z))
        H = H * ((np.abs(fx)[None, :] < f_max) & (np.abs(fx)[:, None] < f_max))
    return np.fft.ifft2(np.fft.fft2(E) * H)


def _fresnel_tf_reference(E, z):
    N = E.shape[0]
    kx_sq = (2 * np.pi * np.fft.fftfreq(N, DX)) ** 2
    phase = K * z - (z / (2 * K)) * (kx_sq[:, None] + kx_sq[None, :])
    return np.fft.ifft2(np.fft.fft2(E) * np.exp(1j * phase))


# ===========================================================================
# 1.  the lattice itself
# ===========================================================================
@pytest.mark.parametrize('N', [8, 9, 63, 64, 65, 256, 257])
def test_freq_grid_is_exactly_fftshift_fftfreq(N):
    """The cached (kx^2, ky^2) vectors must BE the true DFT bins, so that
    ``ifftshift`` puts exactly f = 0 in slot 0 for both parities."""
    kx_sq, ky_sq = _get_or_make_freq_grids(N, N, DX, DX, True)
    ref = (2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, DX))) ** 2
    assert np.array_equal(kx_sq, ref), 'kx^2 is not fftshift(fftfreq)^2'
    assert np.array_equal(ky_sq, ref), 'ky^2 is not fftshift(fftfreq)^2'
    # The consumers ifftshift these vectors; slot 0 must be DC exactly.
    assert float(np.fft.ifftshift(kx_sq)[0]) == 0.0, (
        'DC is not at natural bin 0 -- the odd-N half-bin offset is back')


@pytest.mark.parametrize('N', [9, 64, 65, 256, 257])
def test_bandlimit_masks_use_the_same_bin_labels(N):
    """The Matsushima masks multiply an H built from the grids above, so
    they must label the bins identically (integer DC anchor)."""
    bl_x, bl_y = _get_or_make_bandlimit(N, N, DX, DX, LAM, abs(Z), True)
    f = np.fft.fftshift(np.fft.fftfreq(N, DX))
    f_max = N * DX / (2 * LAM * abs(Z))
    assert np.array_equal(bl_x, np.abs(f) < f_max)
    assert np.array_equal(bl_y, np.abs(f) < f_max)


@pytest.mark.parametrize('N', [8, 9, 63, 64, 65, 256, 257, 512, 513])
def test_integer_anchor_is_bit_identical_for_even_and_fixes_odd(N):
    """The even-N bit-identity guard, computed from both expressions here.

    ``N // 2`` (the fix) vs ``N / 2`` (pre-fix): the two grids must be
    EXACTLY equal for even N -- that is what makes every even-N result
    byte-for-byte unchanged -- and must differ for odd N, where only the
    integer anchor reproduces ``fftfreq``.
    """
    df = 1.0 / (N * DX)
    old = (np.arange(N) - N / 2) * df          # pre-fix expression
    new = (np.arange(N) - N // 2) * df         # post-fix expression
    fftfreq_centred = np.fft.fftshift(np.fft.fftfreq(N, DX))

    if N % 2 == 0:
        assert np.array_equal(old, new), 'even N must be bit-identical'
        assert np.allclose(old, fftfreq_centred, rtol=0, atol=0)
    else:
        assert not np.array_equal(old, new), 'odd N must change'
        # The pre-fix grid is exactly half a bin low, everywhere.
        assert np.allclose(new - old, 0.5 * df, rtol=1e-12, atol=0.0)
        assert float(np.fft.ifftshift(old)[0]) == pytest.approx(-0.5 * df)
        assert float(np.fft.ifftshift(new)[0]) == 0.0
    # Only the integer anchor is the DFT's own lattice, for both parities.
    assert np.array_equal(new, fftfreq_centred)


def test_build_asm_H_square_even_n_matches_pre_fix_expression():
    """Even-N bit-identity guard one level up: the H builder itself.

    Rebuilds H here from the PRE-FIX float-anchor grid and requires exact
    equality at even N (and inequality at odd N).
    """
    def _H_from_anchor(N, anchor, z, bandlimit):
        fx = (np.arange(N, dtype=np.float64) - anchor) / (N * DX)
        kx_sq = (2 * np.pi * fx) ** 2
        kz_sq = K ** 2 - kx_sq[None, :] - kx_sq[:, None]
        prop = kz_sq > 0
        kz = np.where(prop, np.sqrt(np.where(prop, kz_sq, 0.0)), 0.0)
        H = np.where(prop, np.exp(1j * kz * z), 0.0).astype(np.complex128)
        if bandlimit:
            f_max = N * DX / (2 * LAM * abs(z))
            m = np.abs(fx) < f_max
            H = H * (m[None, :] & m[:, None]).astype(np.complex128)
        return H

    for N in (64, 256):
        H_lib = _build_asm_H_square(N, DX, Z, LAM, bandlimit=True)
        assert np.array_equal(H_lib, _H_from_anchor(N, N / 2, Z, True)), (
            f'even N={N} H changed -- the fix must be bit-identical there')
    for N in (65, 257):
        H_lib = _build_asm_H_square(N, DX, Z, LAM, bandlimit=True)
        assert not np.array_equal(H_lib, _H_from_anchor(N, N / 2, Z, True))
        assert np.array_equal(H_lib, _H_from_anchor(N, N // 2, Z, True))


# ===========================================================================
# 2.  the public entry points, odd N, vs the analytic oracle
# ===========================================================================
@pytest.mark.parametrize('N', EVEN_N + ODD_N)
@pytest.mark.parametrize('bandlimit', [False, True])
def test_asm_vs_gaussian_abcd_oracle(N, bandlimit):
    """Pre-fix: 2.593e-01 / -3.8916 px at N=257.  Post-fix the odd-N error
    is the SAME exact-vs-paraxial floor the even-N grid sits on (5.70e-05).
    """
    E = _gaussian(N)
    out = angular_spectrum_propagate(E, Z, LAM, DX, bandlimit=bandlimit)
    ref = _gaussian_abcd(N, Z)
    assert _rel_err(out, ref) < 1.0e-4
    assert abs(_centroid_px(out, _axis(N))) < 1.0e-3


@pytest.mark.parametrize('N', [64, 65, 255, 256, 257, 512, 513])
@pytest.mark.parametrize('bandlimit', [False, True])
def test_asm_equals_true_dft_bin_reference(N, bandlimit):
    """The sharpest form of the pin: the library must agree with an
    independent ``fftfreq``-lattice ASM to round-off for EVERY N.
    Pre-fix this failed at ~2.6e-01 for every odd N."""
    E = _gaussian(N)
    out = angular_spectrum_propagate(E, Z, LAM, DX, bandlimit=bandlimit)
    ref = _asm_reference(E, Z, bandlimit)
    assert np.max(np.abs(out - ref)) / np.abs(ref).max() < 1.0e-12


@pytest.mark.parametrize('N', EVEN_N + ODD_N)
def test_fresnel_tf_vs_oracle_and_reference(N):
    """Pre-fix: 2.592e-01 / -3.8911 px at N=257."""
    E = _gaussian(N)
    out = fresnel_tf_propagate(E, Z, LAM, DX)
    assert _rel_err(out, _gaussian_abcd(N, Z)) < 1.0e-4
    assert abs(_centroid_px(out, _axis(N))) < 1.0e-3
    ref = _fresnel_tf_reference(E, Z)
    assert np.max(np.abs(out - ref)) / np.abs(ref).max() < 1.0e-12


@pytest.mark.parametrize('N', [512, 513])
def test_fresnel_tf_is_the_exact_paraxial_kernel(N):
    """With the window wide enough to hold the tails, the matched-paraxial
    kernel must reproduce the paraxial oracle to round-off at BOTH
    parities (odd N pre-fix: 1.24e-01 at N=513)."""
    out = fresnel_tf_propagate(_gaussian(N), Z, LAM, DX)
    assert _rel_err(out, _gaussian_abcd(N, Z)) < 1.0e-11


@pytest.mark.parametrize('N', EVEN_N + ODD_N)
@pytest.mark.parametrize('bandlimit', [False, True])
def test_asm_mft_same_grid_vs_oracle_and_plain_asm(N, bandlimit):
    """Pre-fix at N=257: rel err 1.533e-01, centroid -3.3916 px (the
    -3.89 px kernel walk plus a half-pixel output-anchor slip)."""
    E = _gaussian(N)
    out = angular_spectrum_propagate_mft(E, Z, LAM, DX, DX, N,
                                        bandlimit=bandlimit)
    assert _rel_err(out, _gaussian_abcd(N, Z)) < 1.0e-4
    assert abs(_centroid_px(out, _axis(N))) < 1.0e-3
    # Same-grid ASM-MFT is documented to reproduce plain ASM to ~1e-12.
    plain = angular_spectrum_propagate(E, Z, LAM, DX, bandlimit=bandlimit)
    assert np.max(np.abs(out - plain)) / np.abs(plain).max() < 1.0e-12


@pytest.mark.parametrize('N', [256, 257])
@pytest.mark.parametrize('zoom', [3.0, 0.5])
def test_asm_mft_zoomed_output_grid_odd_n(N, zoom):
    """The zoom path exercises the output-centre half-pixel term: the
    Bluestein output grid no longer shares the input pitch."""
    dx_out = DX / zoom
    out = angular_spectrum_propagate_mft(_gaussian(N), Z, LAM, DX, dx_out, N,
                                         bandlimit=False)
    assert _rel_err(out, _gaussian_abcd(N, Z, dx_out)) < 1.0e-4


def test_asm_mft_off_axis_window_odd_n():
    """``centre_out`` must still land where it is asked to at odd N."""
    xc, yc = 7.0e-6, -3.0e-6
    N_out, dx_out = 96, DX / 2
    for N in (256, 257):
        out = angular_spectrum_propagate_mft(
            _gaussian(N), Z, LAM, DX, dx_out, N_out,
            centre_out=(xc, yc), bandlimit=False)
        x = _axis(N_out, dx_out) + xc
        y = _axis(N_out, dx_out) + yc
        X, Y = np.meshgrid(x, y, indexing='xy')
        q0 = -1j * ZR
        q = Z - 1j * ZR
        ref = (q0 / q) * np.exp(1j * K * Z) * np.exp(
            1j * K * (X ** 2 + Y ** 2) / (2 * q))
        assert _rel_err(out, ref) < 1.0e-4


@pytest.mark.parametrize('N', EVEN_N + ODD_N)
def test_fresnel_single_fft_vs_oracle_and_mft_sibling(N):
    """Pre-fix at N=257: rel err 1.719e-01 vs the oracle on its own output
    grid, with the intensity centroid a flat -0.5000 px off that grid --
    the returned field sat half an output pixel from the coordinates the
    caller is told to use.  ``fresnel_propagate_mft`` is the discriminator
    (correct at odd N pre-fix)."""
    E = _gaussian(N)
    out, dx_out, dy_out = fresnel_propagate(E, Z, LAM, DX)
    assert dx_out == pytest.approx(LAM * Z / (N * DX))
    assert dy_out == pytest.approx(dx_out)
    ref = _gaussian_abcd(N, Z, dx_out)
    assert _rel_err(out, ref) < 1.0e-11
    assert abs(_centroid_px(out, _axis(N, dx_out))) < 1.0e-3
    sib = fresnel_propagate_mft(E, Z, LAM, DX, dx_out, N)
    assert np.max(np.abs(out - sib)) / np.abs(sib).max() < 1.0e-11


@pytest.mark.parametrize('N', EVEN_N + ODD_N)
def test_fraunhofer_single_fft_vs_mft_sibling(N):
    """Fraunhofer is the far-field limit of Fresnel and shared the same
    ``fftshift`` / ``(n - N/2)`` mismatch verbatim (pre-fix rel err
    1.531e-01 at N=257, 2.787e-14 at N=256)."""
    E = _gaussian(N)
    out, dx_out, _ = fraunhofer_propagate(E, Z, LAM, DX)
    sib = fraunhofer_propagate_mft(E, Z, LAM, DX, dx_out, N)
    assert np.max(np.abs(out - sib)) / np.abs(sib).max() < 1.0e-11
    assert abs(_centroid_px(out, _axis(N, dx_out))) < 1.0e-3


@pytest.mark.parametrize('N', [257, 513])
def test_tilted_asm_odd_n_has_no_lateral_walk(N):
    """The tilted kernel's H is built centred and ``ifftshift``-ed, so it
    carried the same walk (-3.8914 um at N=257 for a carrier-free input,
    which must collapse to plain ASM exactly)."""
    E = _gaussian(N)
    for tilt in (0.0, 0.02):
        out = angular_spectrum_propagate_tilted(E, Z, LAM, DX, tilt_x=tilt,
                                                bandlimit=False)
        assert abs(_centroid_px(out, _axis(N))) < 1.0e-3


# ===========================================================================
# 3.  consumers and cross-parity consistency
# ===========================================================================
@pytest.mark.parametrize('Np', [63, 64, 65, 128, 129])
def test_build_asm_H_square_focal_spot_is_centred(Np):
    """The Shack-Hartmann sub-aperture consumer path
    (``analysis.detector.shack_hartmann`` -> ``_build_asm_H_square``).
    An ideal lens phase on a plane wave must focus on axis; pre-fix the
    focal spot walked -8.0874 px at Np=65 and -4.7245 px at Np=129
    (vs -0.1896 / -0.0404 px at the even twins)."""
    lam, dxp, f = 633e-9, 1.0e-6, 2.0e-3
    x = (np.arange(Np) - Np / 2) * dxp
    X, Y = np.meshgrid(x, x, indexing='xy')
    E = np.exp(-1j * np.pi / (lam * f) * (X ** 2 + Y ** 2)).astype(complex)
    H = _build_asm_H_square(Np, dxp, f, lam, bandlimit=False)
    spec = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E))) * H
    E_f = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spec)))
    I = np.abs(E_f) ** 2
    centroid_px = float((I.sum(axis=0) * x).sum() / I.sum()) / dxp
    # The even-N twins sit at -0.19 px (a real sampling artefact of this
    # coarse 1-um lens grid); odd N must now be in the same class, not 40x
    # worse.
    assert abs(centroid_px) < 0.5


def test_odd_and_even_grids_agree_on_physical_observables():
    """Same physical beam, same pitch, only the grid parity differs: the
    propagated centroid / width / power must agree.  Pre-fix the odd grids
    walked -3.89 px (N=257) and -1.94 px (N=513)."""
    obs = {}
    for N in (256, 257, 512, 513):
        E = _gaussian(N)
        out = angular_spectrum_propagate(E, Z, LAM, DX, bandlimit=False)
        x = _axis(N)
        p_in = float(np.sum(np.abs(E) ** 2)) * DX * DX
        p_out = float(np.sum(np.abs(out) ** 2)) * DX * DX
        obs[N] = (_centroid_px(out, x), _d4sigma(out, x), p_out / p_in)

    for N in (256, 257, 512, 513):
        centroid, width, power = obs[N]
        assert abs(centroid) < 1.0e-6, f'N={N} centroid {centroid}'
        assert power == pytest.approx(1.0, abs=1e-9), f'N={N} power {power}'
    # Widths must match across parity far tighter than any pixel scale.
    for even, odd in ((256, 257), (512, 513)):
        assert obs[odd][1] == pytest.approx(obs[even][1], rel=1e-9)
