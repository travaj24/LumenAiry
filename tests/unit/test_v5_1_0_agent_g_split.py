"""v5.1.0 Agent G regression tests -- ``analysis/core.py`` 6-file split.

Closes the v5.1.0 ROADMAP "Architecture / housekeeping" 6-file-split
work for the analysis stack.  v5.0 shipped a single 4088-LOC
``analysis/core.py``; v5.1.0 carves it into six topical submodules:

* ``lumenairy/analysis/beam_stats.py``   -- centroid, D4sigma, power,
  diameter, M^2.
* ``lumenairy/analysis/strehl.py``       -- Strehl ratio (scalar +
  vector) and coupling efficiency (scalar + vector).
* ``lumenairy/analysis/psf_mtf_otf.py``  -- Fraunhofer PSF / OTF /
  MTF, encircled energy, MTF cutoff, Rayleigh / Sparrow / FWHM
  resolution.
* ``lumenairy/analysis/polychromatic.py`` -- multi-wavelength wrappers
  (``polychromatic_strehl``, ``polychromatic_psf``, ``ee_polychromatic``,
  ``chromatic_focal_shift``, ``radial_power_bands``).
* ``lumenairy/analysis/zernike.py``      -- OSA-indexed Zernike
  polynomials + decomposition + reconstruction + ``astigmatism_mag_angle``.
* ``lumenairy/analysis/opd.py``          -- ASM / OPD sampling
  diagnostics, ``opd_pv_rms``, ``wave_opd_1d/2d``,
  ``remove_wavefront_modes``, ``depth_of_focus``.

The constraint that this regression file pins is:

    **No public API change**: every historical import path that worked
    pre-v5.1.0 continues to work.  No numerics / algorithm change --
    purely mechanical reorganisation.

Three layers of import contract are exercised:

1. ``from lumenairy import X``           (top-level convenience)
2. ``from lumenairy.analysis import X``  (sub-package)
3. ``from lumenairy.analysis.core import X``  (legacy, pre-split path)

For each public symbol all three forms must resolve to the same object.
This is what a downstream consumer relying on any of those paths
actually sees, so identity-equality is the right invariant.

Author: Andrew Traverso -- v5.1.0 / Agent G.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Layer-1 / Layer-2 / Layer-3 import-path equality
# ---------------------------------------------------------------------------

# 41 public symbols that core.py exported before the split.  Built
# from the v5.0 ``__all__`` list literally so this is a complete
# enumeration.
_PUBLIC_SYMBOLS = (
    # beam statistics
    'beam_centroid', 'beam_d4sigma', 'beam_power', 'radial_power_bands',
    'beam_diameter',
    # Strehl
    'strehl_ratio', 'strehl_marechal', 'strehl_phase_integral',
    'strehl_vector',
    # coupling + beam quality
    'coupling_efficiency', 'coupling_efficiency_vector', 'M2',
    # sampling diagnostics
    'check_sampling_conditions', 'check_opd_sampling',
    # PSF / OTF / MTF
    'compute_psf', 'compute_otf', 'compute_mtf', 'mtf_radial',
    'mtf_cutoff',
    # Encircled energy / spec-sheet metrics
    'encircled_energy_curve', 'encircled_energy_radius',
    'ee_polychromatic',
    # Optical resolution metrics
    'rayleigh_resolution', 'sparrow_resolution', 'fwhm_resolution',
    # Depth of focus
    'depth_of_focus',
    # polychromatic
    'chromatic_focal_shift', 'polychromatic_strehl', 'polychromatic_psf',
    # Zernike basis + decomposition
    'zernike_index_to_nm', 'zernike_nm_to_index',
    'zernike_polynomial', 'zernike_basis_matrix',
    'zernike_decompose', 'zernike_reconstruct',
    'astigmatism_mag_angle',
    'clear_zernike_basis_cache',
    # OPD extraction / mode removal
    'remove_wavefront_modes', 'opd_pv_rms',
    'wave_opd_1d', 'wave_opd_2d',
)


@pytest.mark.parametrize('symbol', _PUBLIC_SYMBOLS)
def test_top_level_import_resolves(symbol: str) -> None:
    """``from lumenairy import X`` resolves to a non-None callable /
    attribute for every public symbol."""
    la = importlib.import_module('lumenairy')
    assert hasattr(la, symbol), (
        f"top-level lumenairy is missing public symbol {symbol!r}")
    assert getattr(la, symbol) is not None


@pytest.mark.parametrize('symbol', _PUBLIC_SYMBOLS)
def test_analysis_package_import_resolves(symbol: str) -> None:
    """``from lumenairy.analysis import X`` resolves for every public
    symbol."""
    ana = importlib.import_module('lumenairy.analysis')
    assert hasattr(ana, symbol), (
        f"lumenairy.analysis is missing public symbol {symbol!r}")


@pytest.mark.parametrize('symbol', _PUBLIC_SYMBOLS)
def test_legacy_core_import_resolves(symbol: str) -> None:
    """``from lumenairy.analysis.core import X`` (the pre-split path)
    continues to resolve for every public symbol."""
    core = importlib.import_module('lumenairy.analysis.core')
    assert hasattr(core, symbol), (
        f"lumenairy.analysis.core is missing public symbol {symbol!r}")


@pytest.mark.parametrize('symbol', _PUBLIC_SYMBOLS)
def test_three_paths_resolve_to_same_object(symbol: str) -> None:
    """All three import paths point at the SAME function / class
    object.  This is the bit-for-bit promise of "no public API
    change" -- not just that the names exist, but that they are
    interchangeable for ``is``-checks too."""
    la = importlib.import_module('lumenairy')
    ana = importlib.import_module('lumenairy.analysis')
    core = importlib.import_module('lumenairy.analysis.core')
    f_top = getattr(la, symbol)
    f_pkg = getattr(ana, symbol)
    f_core = getattr(core, symbol)
    assert f_top is f_pkg is f_core, (
        f"{symbol!r} differs across import paths -- "
        f"top={f_top!r}, pkg={f_pkg!r}, core={f_core!r}")


# ---------------------------------------------------------------------------
# Sub-module homes -- pin the new layout so future refactors flag here
# ---------------------------------------------------------------------------

_SUBMODULE_HOMES = {
    # (symbol, expected new submodule)
    'beam_centroid': 'lumenairy.analysis.beam_stats',
    'beam_d4sigma': 'lumenairy.analysis.beam_stats',
    'beam_power': 'lumenairy.analysis.beam_stats',
    'beam_diameter': 'lumenairy.analysis.beam_stats',
    'M2': 'lumenairy.analysis.beam_stats',

    'strehl_ratio': 'lumenairy.analysis.strehl',
    'strehl_marechal': 'lumenairy.analysis.strehl',
    'strehl_phase_integral': 'lumenairy.analysis.strehl',
    'strehl_vector': 'lumenairy.analysis.strehl',
    'coupling_efficiency': 'lumenairy.analysis.strehl',
    'coupling_efficiency_vector': 'lumenairy.analysis.strehl',

    'compute_psf': 'lumenairy.analysis.psf_mtf_otf',
    'compute_otf': 'lumenairy.analysis.psf_mtf_otf',
    'compute_mtf': 'lumenairy.analysis.psf_mtf_otf',
    'mtf_radial': 'lumenairy.analysis.psf_mtf_otf',
    'mtf_cutoff': 'lumenairy.analysis.psf_mtf_otf',
    'encircled_energy_curve': 'lumenairy.analysis.psf_mtf_otf',
    'encircled_energy_radius': 'lumenairy.analysis.psf_mtf_otf',
    'rayleigh_resolution': 'lumenairy.analysis.psf_mtf_otf',
    'sparrow_resolution': 'lumenairy.analysis.psf_mtf_otf',
    'fwhm_resolution': 'lumenairy.analysis.psf_mtf_otf',

    'chromatic_focal_shift': 'lumenairy.analysis.polychromatic',
    'polychromatic_strehl': 'lumenairy.analysis.polychromatic',
    'polychromatic_psf': 'lumenairy.analysis.polychromatic',
    'ee_polychromatic': 'lumenairy.analysis.polychromatic',
    'radial_power_bands': 'lumenairy.analysis.polychromatic',

    'zernike_index_to_nm': 'lumenairy.analysis.zernike',
    'zernike_nm_to_index': 'lumenairy.analysis.zernike',
    'zernike_polynomial': 'lumenairy.analysis.zernike',
    'zernike_basis_matrix': 'lumenairy.analysis.zernike',
    'zernike_decompose': 'lumenairy.analysis.zernike',
    'zernike_reconstruct': 'lumenairy.analysis.zernike',
    'clear_zernike_basis_cache': 'lumenairy.analysis.zernike',
    'astigmatism_mag_angle': 'lumenairy.analysis.zernike',

    'check_sampling_conditions': 'lumenairy.analysis.opd',
    'check_opd_sampling': 'lumenairy.analysis.opd',
    'remove_wavefront_modes': 'lumenairy.analysis.opd',
    'opd_pv_rms': 'lumenairy.analysis.opd',
    'wave_opd_1d': 'lumenairy.analysis.opd',
    'wave_opd_2d': 'lumenairy.analysis.opd',
    'depth_of_focus': 'lumenairy.analysis.opd',
}


@pytest.mark.parametrize('symbol,expected_module', list(_SUBMODULE_HOMES.items()))
def test_symbol_lives_in_expected_submodule(
    symbol: str, expected_module: str,
) -> None:
    """The ``__module__`` of every public symbol matches its NEW home.

    This pins the post-split layout so a future agent that moves
    ``strehl_vector`` (say) out of ``strehl.py`` updates this test
    deliberately.  Together with the three-path identity test above,
    this gives a precise "the re-export shell is wired through to
    the canonical home, and the canonical home hasn't moved" pin.
    """
    core = importlib.import_module('lumenairy.analysis.core')
    f = getattr(core, symbol)
    assert getattr(f, '__module__', None) == expected_module, (
        f"{symbol!r} should be defined in {expected_module!r}; "
        f"actually lives in {getattr(f, '__module__', None)!r}.")


# ---------------------------------------------------------------------------
# Smoke tests -- make sure each submodule executes its core numerical path
# without raising and produces sane values.  These are NOT replacements for
# the existing per-function unit tests; they merely confirm that the post-
# split re-exports are properly wired (no stale reference, no infinite
# import loop, no missing helper).
# ---------------------------------------------------------------------------

def _gaussian_field(N: int = 64, dx: float = 1.0e-6, w0: float = 10e-6):
    """Synthesise a 2-D Gaussian complex field for smoke tests."""
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)
    return E, dx, X, Y


def test_beam_stats_smoke() -> None:
    """beam_centroid / beam_d4sigma / beam_power / beam_diameter / M2
    on a Gaussian return non-NaN, physically-reasonable scalars."""
    from lumenairy.analysis.core import M2, beam_centroid, beam_d4sigma, beam_diameter, beam_power
    E, dx, _, _ = _gaussian_field()
    cx, cy = beam_centroid(E, dx)
    assert abs(cx) < dx and abs(cy) < dx, (cx, cy)
    d4x, d4y = beam_d4sigma(E, dx)
    # Gaussian D4sigma = 2 * w0 in each axis.
    assert d4x > 0 and d4y > 0
    p = beam_power(E, dx)
    assert p > 0.0
    d = beam_diameter(E, dx, threshold='1/e^2')
    assert d > 0.0
    m2x, m2y = M2(E, dx, 1.55e-6)
    # M^2 >= 1 modulo discretisation noise.
    assert m2x > 0.9 and m2y > 0.9


def test_strehl_smoke() -> None:
    """strehl_ratio / strehl_marechal / strehl_phase_integral /
    coupling_efficiency execute and produce values in [0, 1] on a
    Gaussian / unaberrated pupil."""
    from lumenairy.analysis.core import (
        coupling_efficiency,
        strehl_marechal,
        strehl_phase_integral,
        strehl_ratio,
    )
    E, dx, _, _ = _gaussian_field()
    # Strehl against itself should be 1.
    s_self = strehl_ratio(E, E, dx)
    assert abs(s_self - 1.0) < 1e-12, s_self
    # Marechal: 0 wave RMS -> S = 1.
    s_m = float(strehl_marechal(0.0))
    assert abs(s_m - 1.0) < 1e-12
    # Pupil-integral: unaberrated unit-flat-phase circular aperture
    # has S = 1.
    N = 128
    x = (np.arange(N) - N / 2) / (N / 2)
    X, Y = np.meshgrid(x, x)
    pupil = ((X ** 2 + Y ** 2) <= 1.0).astype(np.complex128)
    s_p = strehl_phase_integral(pupil)
    assert abs(s_p - 1.0) < 1e-12, s_p
    # Coupling with self = 1.
    eta = coupling_efficiency(E, E, dx)
    assert abs(eta - 1.0) < 1e-12


def test_psf_mtf_otf_smoke() -> None:
    """compute_psf / compute_otf / compute_mtf / mtf_radial /
    encircled_energy_curve all execute on a flat-phase circular
    aperture."""
    from lumenairy.analysis.core import (
        compute_mtf,
        compute_otf,
        compute_psf,
        encircled_energy_curve,
        encircled_energy_radius,
        fwhm_resolution,
        mtf_cutoff,
        mtf_radial,
        rayleigh_resolution,
    )
    N = 128
    dx_pupil = 10e-6  # 10 um
    x = (np.arange(N) - N / 2) * dx_pupil
    X, Y = np.meshgrid(x, x)
    D = 0.8e-3  # 0.8 mm aperture
    pupil = ((X ** 2 + Y ** 2) <= (D / 2) ** 2).astype(np.complex128)
    wl = 1.55e-6
    f_lens = 25e-3
    psf, dx_psf = compute_psf(pupil, wl, f_lens, dx_pupil)
    assert psf.shape == (N, N)
    assert dx_psf > 0
    otf = compute_otf(psf)
    assert otf.shape == (N, N)
    mtf = compute_mtf(psf)
    assert mtf.shape == (N, N)
    freq, mtf_rad = mtf_radial(mtf, dx_psf, wl, f_lens)
    assert freq.shape == mtf_rad.shape
    # The DC value is 1.
    assert abs(mtf_rad[0] - 1.0) < 1e-6
    # encircled-energy: increases monotonically.
    r, ee = encircled_energy_curve(psf.astype(np.complex128), dx_psf,
                                    n_radii=16)
    assert ee[-1] >= ee[0]
    # encircled-energy-radius: positive, finite.
    r84 = encircled_energy_radius(psf.astype(np.complex128), dx_psf)
    assert r84 > 0 and np.isfinite(r84)
    # MTF cutoff: positive.
    fc = mtf_cutoff(mtf_rad, freq)
    assert fc >= 0
    # FWHM resolution: positive on an Airy-like PSF.
    fwhm = fwhm_resolution(psf, dx_psf)
    assert fwhm > 0 and np.isfinite(fwhm)
    # Rayleigh resolution: typically positive for a circular-aperture
    # Airy PSF (or NaN for poorly-sampled cases).  Both are
    # acceptable.  Silence the runtime warning when the PSF lacks a
    # crisp first-ring zero -- the test only cares that the
    # re-export is wired through to the canonical home.
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter('ignore', RuntimeWarning)
        ray = rayleigh_resolution(psf, dx_psf, wl)
    assert np.isnan(ray) or ray > 0


def test_zernike_smoke() -> None:
    """Zernike round-trip: decompose then reconstruct on a synthetic
    OPD recovers the planted coefficients within fit noise."""
    from lumenairy.analysis.core import (
        astigmatism_mag_angle,
        clear_zernike_basis_cache,
        zernike_basis_matrix,
        zernike_decompose,
        zernike_index_to_nm,
        zernike_nm_to_index,
        zernike_polynomial,
        zernike_reconstruct,
    )
    # OSA index <-> (n, m) round-trip pins.
    assert zernike_index_to_nm(0) == (0, 0)
    assert zernike_index_to_nm(4) == (2, 0)  # defocus
    assert zernike_nm_to_index(2, 0) == 4
    # Build an OPD = pure defocus and decompose.
    N = 128
    dx = 1e-6
    aperture = 100e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r_pup = 0.5 * aperture
    rho = np.sqrt(X ** 2 + Y ** 2) / r_pup
    theta = np.arctan2(Y, X)
    # Plant 1 e-6 metres of pure defocus (OSA j=4, n=2, m=0)
    target = 1e-6
    Z4 = zernike_polynomial(2, 0, rho, theta)
    opd_in = (target * Z4) * (rho <= 1.0)
    clear_zernike_basis_cache()
    coeffs, names = zernike_decompose(opd_in, dx, aperture, n_modes=6)
    assert abs(coeffs[4] - target) / target < 1e-2, coeffs
    # Reconstruct and compare.  The decomposition + reconstruction
    # round-trip must be far smaller than the planted peak amplitude
    # of the OPD field.  Use the same pupil mask the reconstruct
    # function uses internally (r_sq <= 1.0 in the squared-radius
    # form) so a few boundary pixels don't mismatch on the
    # floating-point rounding of ``sqrt(r_sq) <= 1`` vs
    # ``r_sq <= 1``.  Precise relative tolerance is exercised by
    # the full-precision Zernike tests in test_v4_15_1_agent_a.py;
    # here we just smoke-check that the re-exported functions are
    # wired through to the canonical home.
    opd_back = zernike_reconstruct(coeffs, dx, opd_in.shape, aperture)
    pupil = ((X ** 2 + Y ** 2) / (r_pup ** 2)) <= 1.0
    err = float(np.abs(opd_in[pupil] - opd_back[pupil]).max())
    in_peak = float(np.abs(opd_in[pupil]).max())
    assert err < 1e-3 * in_peak, (err, in_peak)
    # Astigmatism magnitude / angle on a synthetic coefficient vector.
    c = np.zeros(6)
    c[3] = 0.3  # oblique
    c[5] = 0.4  # vertical
    mag, ang = astigmatism_mag_angle(c)
    assert abs(mag - 0.5) < 1e-9, mag
    assert -np.pi / 2 < ang <= np.pi / 2
    # basis_matrix smoke: returns the right shape.
    basis, mask = zernike_basis_matrix(6, X, Y, r_pup)
    assert basis.shape[1] == 6
    assert mask.shape == X.shape


def test_opd_smoke() -> None:
    """opd_pv_rms / wave_opd_1d / wave_opd_2d / check_opd_sampling /
    check_sampling_conditions / remove_wavefront_modes /
    depth_of_focus all execute without raising on a Gaussian beam."""
    from lumenairy.analysis.core import (
        check_opd_sampling,
        check_sampling_conditions,
        depth_of_focus,
        opd_pv_rms,
        remove_wavefront_modes,
        wave_opd_1d,
        wave_opd_2d,
    )
    E, dx, X, _ = _gaussian_field()
    pv, rms = opd_pv_rms(np.real(np.angle(E)))
    assert pv >= 0 and rms >= 0
    coord, opd = wave_opd_1d(E, dx, 1.55e-6, axis='x')
    assert coord.shape == opd.shape
    Xx, Yy, opd_map = wave_opd_2d(E, dx, 1.55e-6)
    assert opd_map.shape == E.shape
    res = check_opd_sampling(dx, 1.55e-6, 50e-6, 1.0e-3, verbose=False)
    assert isinstance(res, dict) and 'ok' in res
    samp = check_sampling_conditions(64, dx, 1.0e-3, 1.55e-6,
                                      verbose=False)
    assert isinstance(samp, dict)
    # Mode removal: feed a quadratic OPD; defocus is recovered.
    x = X[X.shape[0] // 2, :]
    opd_1d = (x ** 2) * 1e-3
    resid, coeffs = remove_wavefront_modes(x, opd_1d, modes='defocus')
    # Residual is much smaller than input PV.
    assert np.nanmax(np.abs(resid)) < 1e-3 * np.nanmax(np.abs(opd_1d)) + 1e-18
    # depth_of_focus: f/2 at 550 nm -> 8.8 um.
    dof = depth_of_focus(550e-9, 2.0)
    assert abs(dof - 8.8e-6) / 8.8e-6 < 1e-3


def test_polychromatic_radial_power_bands_smoke() -> None:
    """radial_power_bands -- the only ``polychromatic`` entry that's
    a pure NumPy helper (no prescription required).  The other
    ``polychromatic_*`` entries go through ``apply_real_lens`` and are
    exercised by the existing v4.15 / v4.16 integration suites."""
    from lumenairy.analysis.core import radial_power_bands
    E, dx, _, _ = _gaussian_field(N=128, dx=2e-6, w0=20e-6)
    radii = [10e-6, 20e-6, 50e-6]
    P = radial_power_bands(E, dx, radii)
    assert P.shape == (3,)
    # Monotonically increasing enclosed power.
    assert P[0] <= P[1] <= P[2]


# ---------------------------------------------------------------------------
# Submodule independence pin -- each new submodule imports cleanly when
# loaded in isolation (no hidden cross-file dependency that the shell file
# was masking).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('submodule', [
    'lumenairy.analysis.beam_stats',
    'lumenairy.analysis.strehl',
    'lumenairy.analysis.psf_mtf_otf',
    'lumenairy.analysis.polychromatic',
    'lumenairy.analysis.zernike',
    'lumenairy.analysis.opd',
])
def test_submodule_imports_cleanly(submodule: str) -> None:
    """Each new submodule imports without raising."""
    mod = importlib.import_module(submodule)
    assert mod is not None
    # And owns at least one public symbol.
    assert any(not name.startswith('_') for name in dir(mod))
