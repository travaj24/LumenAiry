"""Validate the BOR-PMM M5 far-field core (Fourier-Bessel decomposition)."""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest
from scipy.special import jn_zeros, jv

from lumenairy.elements.bor.farfield import far_field_angles, fourier_bessel

pytestmark = pytest.mark.slow      # eig-heavy BOR-PMM convergence tests


def _grid(R, N):
    h = R / N
    return (np.arange(N) + 0.5) * h, h


def test_fourier_bessel_roundtrip():
    """A known sum of Fourier-Bessel modes is recovered exactly + reconstructs."""
    m, R, N = 1, 4.0, 400
    r, h = _grid(R, N)
    alpha = jn_zeros(m, 60)
    coeffs = {2: 1.3, 5: -0.7, 9: 0.4}
    f = sum(v * jv(m, alpha[k] * r / R) for k, v in coeffs.items())
    c, kt, norm = fourier_bessel(f, r, h, m, 60)
    for k, v in coeffs.items():
        assert abs(c[k].real - v) < 1e-3
    others = np.delete(c, list(coeffs))
    assert np.max(np.abs(others)) < 1e-5
    recon = np.sum([c[n] * jv(m, alpha[n] * r / R) for n in range(60)], axis=0)
    assert np.max(np.abs(recon - f)) < 1e-4


def test_parseval_power():
    """Parseval: INT|f|^2 r dr == sum |c_n|^2 N_n (the power normalization)."""
    m, R, N = 2, 5.0, 500
    r, h = _grid(R, N)
    rng_alpha = jn_zeros(m, 80)
    f = (0.9 * jv(m, rng_alpha[1] * r / R) - 0.5 * jv(m, rng_alpha[7] * r / R)
         + 0.3 * jv(m, rng_alpha[20] * r / R))
    c, kt, norm = fourier_bessel(f, r, h, m, 80)
    lhs = np.sum(np.abs(f) ** 2 * r * h)
    rhs = np.sum(np.abs(c) ** 2 * norm)
    assert abs(lhs - rhs) / lhs < 1e-6


def test_far_field_angles():
    """kt -> theta map: propagating kt give real angles, evanescent -> NaN."""
    eps, k0 = 2.25, 2.0
    kt = np.array([0.0, 1.0, 2.0, np.sqrt(eps) * k0 * 0.99,
                   np.sqrt(eps) * k0 * 1.5])
    theta, prop = far_field_angles(kt, eps, k0)
    assert prop[0] and theta[0] == 0.0           # kt=0 -> normal
    assert not prop[-1] and np.isnan(theta[-1])  # kt > sqrt(eps)k0 -> evanescent
    # monotonic angle with kt over the propagating set
    assert np.all(np.diff(theta[prop]) > 0)
