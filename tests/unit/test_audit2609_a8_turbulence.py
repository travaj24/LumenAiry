"""WP-A8 regression pins for finding E3 of
``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`` section 5.

``generate_turbulence_screen`` filtered its white noise with
``sqrt(2 * PSD) * df``.  The real and imaginary noise draws are independent,
so taking the real part of the inverse transform costs no factor of two and
the ``sqrt(2)`` doubled the screen's variance and its structure function --
measured ratio 1.987 at r = 5 mm, i.e. an effective r0 of r0/2**(3/5) =
r0/1.516, so every AO / atmospheric run got ~1.5x the turbulence it asked
for at small separations.  The amplitude is now ``sqrt(PSD) * df``.

Oracles
-------
1. **The lattice's own exact structure function.**  With
   ``phi(r) = Re sum_k c_k exp(2 pi i f_k . r)`` and
   ``Var(Re c_k) = PSD_k df**2``, the expectation of
   ``<[phi(r+d) - phi(r)]**2>`` is ``sum_k PSD_k df**2 * 2 (1 - cos(2 pi
   f_k . d))`` exactly.  That sum is computed here from the PSD alone and
   knows nothing about the code's amplitude convention, which is the
   quantity under test.
2. **Kolmogorov** ``D(r) = 6.88 (r/r0)**(5/3)``, the continuum target.
3. **An explicit Schmidt-form reference screen** written out in this file
   (``ft_phase_screen``: ``cn = (randn + i randn) sqrt(PSD) del_f``).
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements import elements as E

gen = E.generate_turbulence_screen


def _lattice(N, dx, r0, L0=np.inf, l0=0.0):
    """The code's own centred frequency lattice and PSD, built independently
    here from the documented formula."""
    df = 1.0 / (N * dx)
    fx = (np.arange(N) - N // 2) * df
    FX, FY = np.meshgrid(fx, fx)
    f_sq = FX ** 2 + FY ** 2
    psd = 0.023 * r0 ** (-5.0 / 3.0) * (
        np.where(f_sq > 0, f_sq, 1.0) + 1.0 / L0 ** 2) ** (-11.0 / 6.0)
    if l0 > 0:
        psd = psd * np.exp(-(np.sqrt(f_sq) * l0 * 2 * np.pi / 5.92) ** 2)
    psd[N // 2, N // 2] = 0.0
    return FX, FY, psd, df


def _d_lattice(N, dx, r0, sep_px):
    """Exact expected structure function of that lattice at a pure-x lag."""
    FX, _, psd, df = _lattice(N, dx, r0)
    return float(np.sum(psd * df ** 2 * 2
                        * (1 - np.cos(2 * np.pi * FX * sep_px * dx))))


def _schmidt_reference(N, dx, r0, seed):
    """``ft_phase_screen`` (Schmidt 2010) written out: amplitude
    ``sqrt(PSD) * df``, no ``sqrt(2)``."""
    rng = np.random.default_rng(seed)
    _, _, psd, df = _lattice(N, dx, r0)
    noise = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    pf = noise * (np.sqrt(psd) * df)
    return np.real(
        np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(pf)))) * N ** 2


# ---------------------------------------------------------------------------
# Deterministic arm: the screen IS the Schmidt form, and is the pre-fix
# screen divided by sqrt(2).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('N', [64, 65, 128, 257])
def test_e3_screen_matches_the_schmidt_amplitude_form(N):
    dx, r0, seed = 1e-3, 0.1, 7
    got = gen(N, dx, r0, seed=seed)
    ref = _schmidt_reference(N, dx, r0, seed)
    assert np.array_equal(got, ref), (
        f"N={N}: max|screen - sqrt(PSD)*df reference| = "
        f"{np.max(np.abs(got - ref)):.6e}; the pre-fix screen differed from "
        f"this reference by the constant factor sqrt(2)")


def test_e3_screen_is_exactly_the_pre_fix_screen_over_sqrt_two():
    """Fail-before arm.  The change is a pure amplitude scale on the same
    RNG stream, so the relation is exact to float rounding: measured
    max|new*sqrt(2) - pre_fix| = 7.1e-15 on a screen whose peak is 12.9,
    i.e. 5.5e-16 relative -- at the float64 floor, 15 decades below the
    41 % discrepancy a missing sqrt(2) would produce.
    """
    N, dx, r0, seed = 256, 5e-3, 0.1, 3
    _, _, psd, df = _lattice(N, dx, r0)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    pre_fix = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        noise * (np.sqrt(2.0 * psd) * df))))) * N ** 2
    got = gen(N, dx, r0, seed=seed)
    rel = np.max(np.abs(got * np.sqrt(2.0) - pre_fix)) / np.max(
        np.abs(pre_fix))
    assert rel < 1e-13, rel
    # ... and the screen is NOT the pre-fix screen.
    assert np.max(np.abs(got - pre_fix)) / np.max(np.abs(pre_fix)) > 0.2


# ---------------------------------------------------------------------------
# Statistical arm: independent structure-function oracles.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sep_px', [1, 2, 4])
def test_e3_structure_function_matches_its_own_lattice(sep_px):
    """D_meas / D_lattice -> 1.

    Measured 2026-09-12, N = 256, dx = 5 mm, r0 = 0.1 m, six independent
    blocks of 24 seeds at sep = 2 px: mean 0.990, range [0.968, 1.031], i.e.
    a +-3.2 % estimator spread.  The bar [0.85, 1.15] sits ~4.7x that
    half-spread away on both sides, and the pre-fix value (exactly 2x, i.e.
    1.98) clears the upper bar by 72 % of the bar's own width.  With 24
    seeds the probability of a spurious excursion past 0.15 is far below any
    practical CI budget.
    """
    N, dx, r0, n_seeds = 256, 5e-3, 0.1, 24
    acc = 0.0
    for s in range(n_seeds):
        ph = gen(N, dx, r0, seed=90000 + s)
        acc += float(np.mean((ph[:, sep_px:] - ph[:, :-sep_px]) ** 2))
    ratio = (acc / n_seeds) / _d_lattice(N, dx, r0, sep_px)
    assert 0.85 < ratio < 1.15, (
        f"sep={sep_px}px: D_meas/D_lattice = {ratio:.4f} "
        f"(1.0 expected; 2.0 = the pre-fix sqrt(2))")


def test_e3_structure_function_vs_kolmogorov_is_below_one_not_above():
    """The continuum target.  An FFT screen is LOW against
    ``6.88 (r/r0)**(5/3)`` because the lattice cannot hold eddies larger
    than the grid: measured D/D_Kolm = 0.798 at r = 0.05 r0 (N = 512).
    Pre-fix it read 1.584 -- ABOVE the continuum target, which is
    unphysical for a band-limited screen and is the shape of the defect.
    The bar [0.6, 1.0] brackets the measured 0.80 with 25 % / 25 % margin
    and excludes the pre-fix 1.58 by 58 %.
    """
    N, dx, r0, n_seeds = 256, 5e-3, 0.1, 24
    sep_px = 1                       # r = 5 mm = 0.05 r0
    acc = 0.0
    for s in range(n_seeds):
        ph = gen(N, dx, r0, seed=91000 + s)
        acc += float(np.mean((ph[:, sep_px:] - ph[:, :-sep_px]) ** 2))
    r = sep_px * dx
    ratio = (acc / n_seeds) / (6.88 * (r / r0) ** (5.0 / 3.0))
    assert 0.6 < ratio < 1.0, (
        f"D_meas/D_Kolmogorov at r = 0.05 r0 = {ratio:.4f} "
        f"(0.80 expected for this lattice; 1.58 was the pre-fix reading)")


# ---------------------------------------------------------------------------
# Subharmonics
# ---------------------------------------------------------------------------

def test_e3_subharmonics_are_off_by_default_and_bit_identical_when_zero():
    a = gen(128, 5e-3, 0.1, seed=11)
    b = gen(128, 5e-3, 0.1, seed=11, subharmonics=0)
    assert np.array_equal(a, b)
    import inspect
    assert inspect.signature(gen).parameters['subharmonics'].default == 0


def test_e3_subharmonics_recover_the_large_scale_structure_function():
    """Three Lane levels close most of the FFT screen's low-frequency
    deficit.  Measured 2026-09-12 (N = 256, dx = 5 mm, r0 = 0.1 m, three
    independent blocks of 20 seeds), D/D_Kolmogorov:

        r/r0 = 0.4 : 0.661 [0.639, 0.703] off -> 0.857 [0.828, 0.893] on
        r/r0 = 1.6 : 0.464 [0.418, 0.538] off -> 0.774 [0.719, 0.827] on

    The bars below sit in the gap between the two measured block ranges:
    at r/r0 = 1.6 the OFF maximum is 0.538 and the ON minimum 0.719, so a
    0.62 / 0.65 pair has ~15 % clearance on each side.  Both arms are also
    asserted as a DECISION (on is closer to 1 than off) which needs no bar.
    """
    N, dx, r0, n_seeds = 256, 5e-3, 0.1, 20
    seps = (8, 32)                   # r/r0 = 0.4 and 1.6
    ratios = {}
    for n_sh in (0, 3):
        acc = np.zeros(len(seps))
        for s in range(n_seeds):
            ph = gen(N, dx, r0, seed=92000 + s, subharmonics=n_sh)
            for i, sp in enumerate(seps):
                acc[i] += float(np.mean((ph[:, sp:] - ph[:, :-sp]) ** 2))
        ratios[n_sh] = [
            (acc[i] / n_seeds) / (6.88 * ((sp * dx) / r0) ** (5.0 / 3.0))
            for i, sp in enumerate(seps)]
    for i, sp in enumerate(seps):
        off, on = ratios[0][i], ratios[3][i]
        assert abs(on - 1.0) < abs(off - 1.0), (
            f"r/r0={sp * dx / r0}: subharmonics made D/D_Kolm worse "
            f"({off:.3f} -> {on:.3f})")
    assert ratios[0][1] < 0.62, ratios[0][1]
    assert ratios[3][1] > 0.65, ratios[3][1]
    # The correction must not blow the small-scale end past the continuum.
    assert ratios[3][0] < 1.05, ratios[3][0]


def test_e3_subharmonic_sum_is_the_separable_form_of_the_direct_sum():
    """The 3x3 level is summed as ``e.T @ cn @ e``; that must equal the
    literal double loop over the nine frequencies.  Measured max abs
    difference 6.4e-14 on a screen whose peak is 36.4 (1.8e-15 relative) --
    float64 reassociation noise, 13 decades below any real difference.
    """
    N, dx, r0 = 256, 5e-3, 0.1
    got = E._turbulence_subharmonics(
        N, dx, r0, np.inf, 0.0, np.random.default_rng(11), 3)
    rng = np.random.default_rng(11)
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    low = np.zeros((N, N), dtype=complex)
    for p in range(1, 4):
        df_p = 1.0 / (3.0 ** p * N * dx)
        f_axis = np.array([-1.0, 0.0, 1.0]) * df_p
        FX, FY = np.meshgrid(f_axis, f_axis)
        f_sq = FX ** 2 + FY ** 2
        psd = E._turbulence_psd(np.where(f_sq > 0, f_sq, 1.0),
                                np.sqrt(f_sq), r0, np.inf, 0.0)
        psd[1, 1] = 0.0
        cn = ((rng.standard_normal((3, 3))
               + 1j * rng.standard_normal((3, 3))) * np.sqrt(psd) * df_p)
        for i in range(3):
            for j in range(3):
                low += cn[i, j] * np.exp(
                    2j * np.pi * (FX[i, j] * X + FY[i, j] * Y))
    ref = np.real(low)
    ref = ref - ref.mean()
    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 1e-12


@pytest.mark.parametrize('bad', [-1, 2.5, 'three', None])
def test_e3_subharmonics_is_validated_with_the_function_name_prefix(bad):
    with pytest.raises(ValueError, match=r'generate_turbulence_screen:'):
        gen(32, 5e-3, 0.1, seed=1, subharmonics=bad)


# ---------------------------------------------------------------------------
# Keep what the audit verified correct
# ---------------------------------------------------------------------------

def test_e3_psd_shape_is_untouched():
    """The audit verified the PSD SHAPE (0.023 f^-11/3 in cycles/m, the von
    Karman knee, the inner-scale cutoff and the integer ``N//2`` DC anchor);
    only the amplitude was wrong.  A finite outer scale must still cut the
    variance and a finite inner scale must cut it further, on the same seed.
    """
    N, dx, r0 = 256, 2e-3, 0.5
    kol = float(np.var(gen(N, dx, r0, seed=7)))
    vk = float(np.var(gen(N, dx, r0, L0=1.0, seed=7)))
    vk_inner = float(np.var(gen(N, dx, r0, L0=1.0, l0=5e-3, seed=7)))
    assert vk < kol, (vk, kol)
    assert vk_inner < vk, (vk_inner, vk)


def test_e3_shared_psd_helper_is_used_by_both_paths():
    """The FFT grid and the subharmonic grids must not carry two copies of
    the PSD formula (the audit's duplicated-scaffolding theme)."""
    import inspect
    src = inspect.getsource(gen)
    assert '_turbulence_psd(' in src
    assert '0.023' not in src.split('"""')[-1], (
        'the PSD constant is still inlined in the function body')
    sh = inspect.getsource(E._turbulence_subharmonics)
    assert '_turbulence_psd(' in sh and '0.023' not in sh
