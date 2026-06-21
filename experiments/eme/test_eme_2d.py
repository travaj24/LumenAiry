"""Validate the EME (eigenmode-expansion) 2-D lateral-cascade layer-mode solver.

The 2-D Bloch modes of a y-strip-sectioned crossed grating, built from 1-D-x
modes + a stable lateral S-matrix cascade, are checked against a direct 2-D
finite-difference eigensolve.  Because the EME is ANALYTIC in y while the 2-D-FD
is finite-difference in y, the 2-D-FD *converges to* the EME as ``Ny -> inf``
(the cleanest validation: the EME is the exact-y limit).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
from eme_2d import layer_modes, ref_2d_modes, strips_to_eps_xy

Lx = Ly = 1.0
k0 = 8.0
KY0 = np.pi                     # off the global band edge (cusp-free dispersion)


def _grating(Nx, e_lo, e_hi, duty=0.5):
    xg = (np.arange(Nx) + 0.5) / Nx
    return np.where(xg < duty, e_hi, e_lo).astype(float)


def test_uniform_matches_analytic():
    """Uniform layer: EME modes == eps*k0^2 - kx^2 - ky^2 (to FD accuracy)."""
    Nx = 28
    eps = np.full(Nx, 4.0)
    eme = layer_modes([(eps, 0.5), (eps, 0.5)], Lx, Nx, Ly, k0, (150, 256),
                      ky0=KY0)
    anal = sorted({4 * k0 ** 2 - (2 * np.pi * a) ** 2 - (KY0 + 2 * np.pi * b) ** 2
                   for a in range(-3, 4) for b in range(-3, 4)}, reverse=True)
    anal = [a for a in anal if a > 150]
    # each EME mode matches an analytic one (the small gap is the x-FD error)
    for q in eme[:3]:
        assert min(abs(q - a) for a in anal) < 0.5


def test_structured_2strip_converges_from_2dfd():
    """2-D-FD converges to the EME (analytic-y) as Ny grows -> the EME is the
    exact-y limit (the load-bearing validation)."""
    Nx = 28
    strips = [(_grating(Nx, 1.0, 4.0), 0.5), (np.full(Nx, 2.0), 0.5)]
    eme = layer_modes(strips, Lx, Nx, Ly, k0, (120, 256), ky0=KY0)[:3]
    assert len(eme) == 3
    prev = np.inf
    for Ny in (28, 56, 112):
        eps_xy = strips_to_eps_xy(strips, Lx, Nx, Ly, Ny)
        ref = ref_2d_modes(eps_xy, Lx, Ly, Nx, Ny, k0, ky0=KY0)
        ref = np.sort(ref[ref > 120][:3])[::-1]
        err = np.max(np.abs(ref - eme))
        assert err < prev or err < 1e-3        # monotone convergence to the EME
        prev = err
    assert prev < 0.05                          # converged to the EME


def test_high_contrast_pillar_multibasis():
    """The high-contrast pillar (deeply-localized modes) needs the multi-basis
    search; all modes match the 2-D-FD (Ny=200)."""
    Nx = 28
    epsH = np.full(Nx, 1.0)
    epsG = _grating(Nx, 1.0, 6.0)                # eps=6 pillar in eps=1 host
    strips = [(epsH, 0.25), (epsG, 0.5), (epsH, 0.25)]
    eme = layer_modes(strips, Lx, Nx, Ly, k0, (60, 384), ky0=KY0)[:6]
    ref = ref_2d_modes(strips_to_eps_xy(strips, Lx, Nx, Ly, 200),
                       Lx, Ly, Nx, 200, k0, ky0=KY0)
    ref = ref[ref > 60][:6]
    assert len(eme) == 6                         # found the localized modes too
    assert np.max(np.abs(eme - ref)) < 0.5       # all match (y-FD error ~0.1)


def test_no_duplicate_modes():
    """Modes found in several strip bases are deduped -- no near-duplicates."""
    Nx = 28
    strips = [(_grating(Nx, 1.0, 4.0), 0.5), (np.full(Nx, 2.0), 0.5)]
    eme = layer_modes(strips, Lx, Nx, Ly, k0, (120, 256), ky0=KY0)
    gaps = np.abs(np.diff(eme))
    assert np.all(gaps > 0.3)                    # no two reported modes coincide
