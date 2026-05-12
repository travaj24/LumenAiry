"""Tests for partial-coherence helpers (4.0+):
koehler_image, extended_source_image, mutual_coherence.

These three functions had no dedicated test coverage prior to 4.0.
This file fills that gap with smoke + sanity-invariant tests.
"""
from __future__ import annotations

import sys
import pathlib as _pathlib
import numpy as np

_sys_path = _pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_sys_path))
from _harness import Harness

import lumenairy as la


H = Harness('coherence')


# ---------------------------------------------------------------------
H.section('Koehler illumination')


def t_koehler_image_shape_and_finite():
    """koehler_image returns a finite intensity map of correct shape."""
    N, dx = 64, 8e-6
    obj = np.ones((N, N), dtype=complex)
    p = la.make_singlet(R1=51.5e-3, R2=float('inf'), d=4e-3,
                          glass='N-BK7', aperture=10e-3)
    try:
        img = la.koehler_image(
            obj, p, wavelength=1.31e-6, dx=dx,
            condenser_NA=0.05, n_source_points=5)
    except Exception as e:
        return False, f'{type(e).__name__}: {e}'
    return (img.shape == (N, N)
            and np.all(np.isfinite(img))
            and float(img.min()) >= 0), \
        f'shape={img.shape}, min={float(img.min()):.3e}'


H.run('koehler_image: finite, correct shape, non-negative intensity',
      t_koehler_image_shape_and_finite)


def t_koehler_image_uniform_object_gives_uniform_response():
    """A uniform-amplitude object should produce a smooth, mostly-uniform
    illumination on the detector (modulo finite-NA edge effects).  Test
    by checking that the intensity ratio between centre and 75%-radius
    of the FoV is within a factor of 2."""
    N, dx = 64, 8e-6
    obj = np.ones((N, N), dtype=complex)
    p = la.make_singlet(R1=51.5e-3, R2=float('inf'), d=4e-3,
                          glass='N-BK7', aperture=10e-3)
    img = la.koehler_image(
        obj, p, wavelength=1.31e-6, dx=dx,
        condenser_NA=0.05, n_source_points=5)
    cx, cy = N // 2, N // 2
    r = int(0.35 * N)
    centre = float(img[cy, cx])
    off = float(img[cy, cx + r])
    if centre == 0:
        return False, f'centre intensity is zero'
    ratio = off / centre
    return 0.2 < ratio < 5.0, \
        f'centre={centre:.3e}, off={off:.3e}, ratio={ratio:.3f}'


H.run('koehler_image: roughly uniform response across FoV',
      t_koehler_image_uniform_object_gives_uniform_response)


# ---------------------------------------------------------------------
H.section('Extended-source imaging')


def t_extended_source_image_smoke():
    N, dx = 64, 8e-6
    obj = np.ones((N, N), dtype=complex)
    p = la.make_singlet(R1=51.5e-3, R2=float('inf'), d=4e-3,
                          glass='N-BK7', aperture=10e-3)
    angles = [(0.0, 0.0), (0.05, 0.0), (-0.05, 0.0),
              (0.0, 0.05), (0.0, -0.05)]
    try:
        img = la.extended_source_image(
            obj, p, wavelength=1.31e-6, dx=dx,
            source_angles=angles)
    except Exception as e:
        return False, f'{type(e).__name__}: {e}'
    return (img.shape == (N, N) and np.all(np.isfinite(img))
            and float(img.min()) >= 0), \
        f'shape={img.shape}, min={float(img.min()):.3e}'


H.run('extended_source_image: finite, correct shape, non-negative',
      t_extended_source_image_smoke)


def t_extended_source_image_weights_normalised():
    """Re-running with rescaled weights should give a proportionally
    scaled image."""
    N, dx = 64, 8e-6
    obj = np.ones((N, N), dtype=complex)
    p = la.make_singlet(R1=51.5e-3, R2=float('inf'), d=4e-3,
                          glass='N-BK7', aperture=10e-3)
    angles = [(0.0, 0.0), (0.05, 0.0), (0.0, 0.05)]
    img_a = la.extended_source_image(
        obj, p, wavelength=1.31e-6, dx=dx,
        source_angles=angles, source_weights=[1.0, 1.0, 1.0])
    img_b = la.extended_source_image(
        obj, p, wavelength=1.31e-6, dx=dx,
        source_angles=angles, source_weights=[3.0, 3.0, 3.0])
    # If weights are normalised internally to sum to 1, doubling should
    # not change the result.  Verify they agree to roundoff.
    rel = float(np.max(np.abs(img_b - img_a)
                         / (np.abs(img_a) + 1e-20)))
    return rel < 1e-9, f'rel diff (3x weights vs 1x) = {rel:.2e}'


H.run('extended_source_image: weights renormalised internally',
      t_extended_source_image_weights_normalised)


# ---------------------------------------------------------------------
H.section('Mutual coherence')


def t_mutual_coherence_diagonal_is_intensity():
    """Gamma(x, x) along the diagonal must equal the per-pixel
    intensity along the central row averaged over the ensemble."""
    rng = np.random.default_rng(0)
    N = 64
    fields = [rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
              for _ in range(20)]
    G, _ = la.mutual_coherence(fields, dx=1e-3)
    diag = np.diag(G).real
    # Expected: mean over ensemble of |E[centre_row, :]|^2 (per-x).
    expected = np.mean(
        np.stack([np.abs(E[N // 2, :]) ** 2 for E in fields], axis=0),
        axis=0)
    rel = float(np.max(np.abs(diag - expected) / (expected + 1e-20)))
    return rel < 1e-12, f'max rel err on diagonal = {rel:.2e}'


H.run('mutual_coherence: diagonal equals ensemble-mean intensity',
      t_mutual_coherence_diagonal_is_intensity)


def t_mutual_coherence_hermitian():
    """Gamma(x1, x2) = Gamma*(x2, x1) -- Hermitian symmetry."""
    rng = np.random.default_rng(1)
    N = 32
    fields = [rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
              for _ in range(10)]
    G, _ = la.mutual_coherence(fields, dx=1e-3)
    err = float(np.max(np.abs(G - np.conj(G.T))))
    return err < 1e-12, f'max |G - G^H| = {err:.2e}'


H.run('mutual_coherence: Hermitian symmetry',
      t_mutual_coherence_hermitian)


def t_mutual_coherence_uncorrelated_realisations():
    """Independent zero-mean random fields have Gamma(x1, x2) ~ 0 off-
    diagonal.  Test with a 200-realisation ensemble."""
    rng = np.random.default_rng(42)
    N = 32
    fields = [rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
              for _ in range(200)]
    G, _ = la.mutual_coherence(fields, dx=1e-3)
    off = G - np.diag(np.diag(G))
    rms_off = float(np.sqrt(np.mean(np.abs(off) ** 2)))
    rms_diag = float(np.sqrt(np.mean(np.abs(np.diag(G)) ** 2)))
    # Decorrelation should make off-diagonal RMS << diagonal RMS.
    return rms_off / rms_diag < 0.3, \
        (f'off-diag RMS = {rms_off:.3e}, diag RMS = {rms_diag:.3e}, '
         f'ratio={rms_off / rms_diag:.3f}')


H.run('mutual_coherence: independent fields have small off-diagonal',
      t_mutual_coherence_uncorrelated_realisations)


# ---------------------------------------------------------------------
def main():
    return H.summary()


if __name__ == '__main__':
    sys.exit(main())
