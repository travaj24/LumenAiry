"""Byte-identity of the row-banded slant/fresnel refraction path.

``apply_real_lens`` evaluates the per-surface refraction pipeline (local
surface normal -> cos_ti / cos_tt -> refraction OPD -> phase screen ->
fresnel amplitude -> TIR mask) whole-grid, which materialises ~6 full-grid
float64 transients at once (~51 GB at N=32768).  ``sag_chunk_rows`` now
evaluates that pipeline in row-bands for the slant_correction / fresnel
case too, so only a (chunk_rows x Nx) slice is live at once.

Every banded op is pointwise EXCEPT the y-sag gradient, which is taken on
a 1-row halo so np.gradient's central differences match the whole-grid
result bit-for-bit; the numexpr phase-screen decision reuses the SAME
whole-E.size gate as the whole-grid path so the numexpr-vs-numpy choice
(and hence the last-bit result) is identical on both paths.  The banded
output is therefore BYTE-IDENTICAL to the whole-grid output -- pinned here
with ``np.array_equal`` (not ``allclose``).

Determinism note: apply_real_lens does an FFT-based ASM propagation
through the glass between surfaces, so byte-equality across separate calls
requires FFT plan determinism -- the fixture pins auto-promote off
(matching the production runner configuration and test_lens_chunked_sag).
"""
import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements._lens_real import apply_real_lens

LAM = 1.31e-6


@pytest.fixture(autouse=True)
def _deterministic_fft():
    """Pin FFT plan determinism across calls for byte-equality asserts."""
    la.set_fft_auto_promote(False)
    yield
    la.set_fft_auto_promote(True)


def _presc():
    """Real two-element singlet: a POWERED conic + asphere on both faces
    with an N-BK7 core, so the slant OPD (n2*sag/cos_tt - n1*sag/cos_ti)
    and the fresnel amplitude coefficients are non-trivial (the surface
    gradient makes cos_ti / cos_tt vary across the grid)."""
    return {
        'name': 't', 'aperture_diameter': 3e-3,
        'surfaces': [
            {'radius': 12e-3, 'conic': -0.4,
             'aspheric_coeffs': {4: 2e4, 6: 5e6},
             'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'radius': -15e-3, 'conic': 0.0, 'aspheric_coeffs': {4: -1e4},
             'glass_before': 'N-BK7', 'glass_after': 'air'},
        ],
        'thicknesses': [2.5e-3],
    }


def _tilted_field(N, dx, w=1.2e-3, tilt_deg=2.5, dtype=np.complex64):
    """A tilted Gaussian: real amplitude envelope x a linear phase ramp so
    the field the phase screen multiplies is a genuine complex array."""
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    k = 2 * np.pi / LAM
    env = np.exp(-(X ** 2 + Y ** 2) / w ** 2)
    phase = np.exp(1j * k * np.sin(np.deg2rad(tilt_deg)) * X)
    return (env * phase).astype(dtype)


# N=512 -> plain-numpy phase-screen path (E.size < 1Mi);
# N=1024 -> numexpr phase-screen path (E.size == 1Mi).  Both must be
# byte-identical between whole-grid and banded.
@pytest.mark.parametrize("dtype", [np.complex64,     # fresnel promotes -> c128
                                   np.complex128])   # no promotion
@pytest.mark.parametrize("N,dx", [(512, 6e-6), (1024, 3e-6)])
@pytest.mark.parametrize("slant,fres", [(True, False),    # slant only
                                        (False, True),    # fresnel only
                                        (True, True)])     # both
def test_slant_fresnel_band_byte_identical(N, dx, slant, fres, dtype):
    E = _tilted_field(N, dx, dtype=dtype)
    kw = dict(prescription=_presc(), wavelength=LAM, dx=dx,
              slant_correction=slant, fresnel=fres)
    E_whole = apply_real_lens(E.copy(), sag_chunk_rows=0, **kw)   # whole-grid
    E_band = apply_real_lens(E.copy(), sag_chunk_rows=64, **kw)   # row-banded
    # dtype AND values must match -- np.array_equal ignores dtype, and the
    # fresnel amplitude promotes complex64 -> complex128 exactly like the
    # whole-grid path, so pin both for a true byte-identity guarantee.
    assert E_whole.dtype == E_band.dtype, (
        f"N={N} slant={slant} fresnel={fres}: dtype "
        f"{E_whole.dtype} (whole) != {E_band.dtype} (band)")
    assert np.array_equal(E_whole, E_band), (
        f"N={N} slant={slant} fresnel={fres}: band64 != whole-grid")


@pytest.mark.parametrize("slant,fres", [(True, False),
                                        (False, True),
                                        (True, True)])
def test_ragged_last_band_byte_identical(slant, fres):
    """A band size that does NOT evenly divide N (70 into 512) exercises
    the ragged final band (rows 490:512) -- which also contains the true
    array edge row Ny-1 (one-sided gradient stencil)."""
    N, dx = 512, 6e-6
    E = _tilted_field(N, dx)
    kw = dict(prescription=_presc(), wavelength=LAM, dx=dx,
              slant_correction=slant, fresnel=fres)
    E_whole = apply_real_lens(E.copy(), sag_chunk_rows=0, **kw)
    E_band = apply_real_lens(E.copy(), sag_chunk_rows=70, **kw)
    assert E_whole.dtype == E_band.dtype
    assert np.array_equal(E_whole, E_band), (
        f"slant={slant} fresnel={fres}: ragged band70 != whole-grid")


def test_multiple_band_sizes_agree():
    """Several band sizes (divisor, non-divisor, single-row, all-rows) all
    reproduce the whole-grid result exactly for the combined slant+fresnel
    pipeline."""
    N, dx = 512, 6e-6
    E = _tilted_field(N, dx)
    kw = dict(prescription=_presc(), wavelength=LAM, dx=dx,
              slant_correction=True, fresnel=True)
    E_whole = apply_real_lens(E.copy(), sag_chunk_rows=0, **kw)
    for cr in (1, 3, 64, 70, 128, 511, 512, 999):
        E_band = apply_real_lens(E.copy(), sag_chunk_rows=cr, **kw)
        assert np.array_equal(E_whole, E_band), f"chunk_rows={cr} not byte-identical"


def test_banded_path_invokes_per_band_gradient(monkeypatch):
    """Prove the row-banded pipeline is actually taken (not silently falling
    through to the whole-grid path): the whole-grid path calls np.gradient
    ONCE per surface, the banded path once per BAND per surface.  Since
    xp is np on the CPU path, patching numpy.gradient observes both."""
    N, dx = 512, 6e-6
    E = _tilted_field(N, dx)
    kw = dict(prescription=_presc(), wavelength=LAM, dx=dx,
              slant_correction=True)

    counts = {'n': 0}
    real_grad = np.gradient

    def counting_grad(*a, **k):
        counts['n'] += 1
        return real_grad(*a, **k)

    monkeypatch.setattr(np, 'gradient', counting_grad)

    counts['n'] = 0
    apply_real_lens(E.copy(), sag_chunk_rows=0, **kw)      # whole-grid
    whole_calls = counts['n']

    counts['n'] = 0
    apply_real_lens(E.copy(), sag_chunk_rows=64, **kw)     # row-banded
    band_calls = counts['n']

    n_bands = (N + 64 - 1) // 64      # ceil(512 / 64) = 8 bands per surface
    assert whole_calls >= 1, "whole-grid path never took a sag gradient"
    assert band_calls == whole_calls * n_bands, (
        f"banded gradient count {band_calls} != {whole_calls} * {n_bands}; "
        "the per-band refraction path may not have executed")
