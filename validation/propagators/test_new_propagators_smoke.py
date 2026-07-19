"""S5-13 (AUDIT_V5_24_2) -- validation-suite smoke leg for the newest
propagators.

The v5.21+ caustic / spectral-element propagators (fga, traced-
multibranch, Maslov-Levin, Chebyshev, matrix-Fourier-transform, and the
partial-coherence ensemble driver) were covered under ``tests/unit`` but
had NO leg in the physics ``validation/`` suite, so a single command
(``run_all.py`` / ``pytest validation/``) never exercised them end-to-
end.  This file closes that gap with a MINIMAL smoke leg.

Where a cheap independent oracle exists it is used (not the code's own
formula):

  * Levin  -- int_0^1 e^{i w y} dy against the closed form
              (e^{iw} - 1) / (i w).
  * Chebyshev -- a height map built INSIDE the fit basis must be
              reconstructed with ~machine-zero residual.
  * MFT ASM -- Bluestein-CZT ASM on the natural grid must equal the
              FFT-based ``angular_spectrum_propagate`` (different
              algorithm, same math).
  * Ensemble -- ``propagate_ensemble`` must equal the hand-written
              per-realisation intensity average.

For the two heavy ray/swarm lens propagators (fga, multibranch) a cheap
independent oracle is not available, so the smoke check confirms they
run to a finite, correctly-shaped, energy-sane field at small N.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import sys

import numpy as np
from _harness import Harness

from lumenairy._math.chebyshev import chebyshev_fit_2d
from lumenairy._math.levin import levin1d_adaptive
from lumenairy.elements._lens_traced_multibranch import (
    apply_real_lens_traced_multibranch,
)
from lumenairy.propagators.asm import angular_spectrum_propagate
from lumenairy.propagators.ensemble import propagate_ensemble
from lumenairy.propagators.mft import angular_spectrum_propagate_mft

H = Harness('new_propagators_smoke')

_WL = 0.633e-6


def _singlet():
    """Plano-convex N-BK7 singlet, curved side first (flat exit)."""
    return {'name': 'pcx', 'aperture_diameter': 2.8e-3,
            'surfaces': [
                {'radius': 20e-3, 'conic': 0.0, 'glass_before': 'air',
                 'glass_after': 'N-BK7', 'semi_diameter': 1.4e-3},
                {'radius': np.inf, 'conic': 0.0, 'glass_before': 'N-BK7',
                 'glass_after': 'air', 'semi_diameter': 1.4e-3}],
            'thicknesses': [2.5e-3]}


def _collimated(N, dx, w=0.9e-3):
    xs = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(xs, xs)
    return np.exp(-(Xg ** 2 + Yg ** 2) / w ** 2).astype(np.complex128)


# --------------------------------------------------------------------- #
# Independent-oracle checks
# --------------------------------------------------------------------- #

def t_levin_matches_analytic():
    """Adaptive 1-D Levin integral of a pure oscillatory kernel matches
    the closed form to ~machine precision."""
    w = 40.0
    val = levin1d_adaptive(
        lambda y: w * y,
        lambda y: w * np.ones_like(np.asarray(y, float)),
        lambda y: np.ones_like(np.asarray(y, float)),
        0.0, 1.0, tol=1e-11)
    exact = (np.exp(1j * w) - 1.0) / (1j * w)
    err = abs(val - exact)
    return err < 1e-9, f'levin abs-err {err:.2e}'


def t_chebyshev_recovers_in_span():
    """A height map built inside the Chebyshev basis is reconstructed
    with machine-zero residual and the known coefficients."""
    N = 21
    xs = np.linspace(-3e-3, 3e-3, N)
    ys = np.linspace(-3e-3, 3e-3, N)
    ax, ay = xs.max(), ys.max()
    Xn, Yn = np.meshgrid(xs / ax, ys / ay, indexing='xy')
    z = 1.0 + 0.5 * Xn - 0.25 * Yn + 0.1 * (2 * Xn ** 2 - 1)
    coeffs, resid = chebyshev_fit_2d(
        xs, ys, z, n_max_x=3, n_max_y=3, return_residual=True)
    max_resid = float(np.nanmax(np.abs(resid)))
    ok_c = (abs(coeffs.get((0, 0), 0.0) - 1.0) < 1e-9
            and abs(coeffs.get((1, 0), 0.0) - 0.5) < 1e-9
            and abs(coeffs.get((0, 1), 0.0) + 0.25) < 1e-9
            and abs(coeffs.get((2, 0), 0.0) - 0.1) < 1e-9)
    return (max_resid < 1e-10 and ok_c,
            f'cheby resid {max_resid:.2e}, coeffs-ok {ok_c}')


def t_mft_asm_matches_fft_asm():
    """MFT (Bluestein-CZT) ASM on the natural output grid reproduces the
    FFT-based ASM -- an independent-algorithm cross-check."""
    N = 64
    dx = 8e-6
    E0 = _collimated(N, dx, w=40e-6)
    z = 2e-3
    E_fft = angular_spectrum_propagate(E0, z, _WL, dx)
    E_mft = angular_spectrum_propagate_mft(
        E0, z, _WL, dx_in=dx, dx_out=dx, N_out=N, centre_out=(0.0, 0.0))
    rel = np.linalg.norm(E_mft - E_fft) / np.linalg.norm(E_fft)
    return rel < 1e-9, f'mft vs fft relL2 {rel:.2e}'


def t_ensemble_matches_handloop():
    """``propagate_ensemble`` equals the hand-written per-realisation
    intensity average <|E_k|^2>_k."""
    N = 48
    dx = 8e-6
    E0 = _collimated(N, dx, w=40e-6)
    z = 2e-3
    rng = np.random.default_rng(0)
    K = 5
    ens = np.stack([E0 * np.exp(1j * rng.uniform(-np.pi, np.pi, E0.shape))
                    for _ in range(K)]).astype(np.complex128)
    I_helper = propagate_ensemble(
        ens, dx=dx, wavelength=_WL, propagator='asm', z=z,
        return_intensity=True)
    I_ref = np.mean(
        [np.abs(angular_spectrum_propagate(ens[k], z, _WL, dx)) ** 2
         for k in range(K)], axis=0)
    rel = np.linalg.norm(I_helper - I_ref) / np.linalg.norm(I_ref)
    return (I_helper.shape == (N, N) and rel < 1e-12,
            f'ensemble vs hand-loop relL2 {rel:.2e}')


# --------------------------------------------------------------------- #
# Heavy-propagator finite-output smoke (no cheap independent oracle)
# --------------------------------------------------------------------- #

def t_multibranch_runs_finite():
    """Multi-branch ray-traced lens field runs to a finite, correctly
    shaped, energy-sane field at small N."""
    N = 96
    dx = 20e-6
    u0 = _collimated(N, dx)
    E = apply_real_lens_traced_multibranch(
        u0, prescription=_singlet(), wavelength=_WL, dx=dx,
        output_plane_distance=25e-3, ray_subsample=3)
    E = np.asarray(E)
    power = float(np.sum(np.abs(E) ** 2))
    ok = (E.shape == (N, N) and np.all(np.isfinite(E))
          and np.isfinite(power) and power > 0.0)
    return ok, f'multibranch power {power:.3e}'


def t_fga_runs_finite():
    """FGA (Frozen Gaussian Approximation) lens field runs to a finite,
    correctly shaped field at small N.  Requires numba; a missing numba
    is reported as a pass-through skip so the leg stays green."""
    try:
        import numba  # noqa: F401
    except ImportError:
        return True, 'skipped: numba not installed'
    from lumenairy.propagators.fga import apply_real_lens_fga
    N = 64
    dx = 24e-6
    u0 = _collimated(N, dx)
    E = apply_real_lens_fga(
        u0, prescription=_singlet(), wavelength=_WL, dx=dx,
        output_plane_distance=25e-3)
    E = np.asarray(E)
    power = float(np.sum(np.abs(E) ** 2))
    ok = (E.shape == (N, N) and np.all(np.isfinite(E))
          and np.isfinite(power) and power > 0.0)
    return ok, f'fga power {power:.3e}'


def main():
    H.section('Independent-oracle checks')
    H.run('Levin 1-D integral matches closed form', t_levin_matches_analytic)
    H.run('Chebyshev fit recovers in-span map', t_chebyshev_recovers_in_span)
    H.run('MFT ASM matches FFT ASM on natural grid', t_mft_asm_matches_fft_asm)
    H.run('propagate_ensemble matches hand-loop', t_ensemble_matches_handloop)

    H.section('Heavy-propagator finite-output smoke')
    H.run('multibranch runs to finite field', t_multibranch_runs_finite)
    H.run('fga runs to finite field', t_fga_runs_finite)

    sys.exit(H.summary())


if __name__ == '__main__':
    main()
