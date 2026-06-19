"""Validate the BOR-PMM radial eigensolver (Phase 0 / Milestone 1) against the
EXACT Bessel spectrum -- the cylindrical-metric + r=0-axis de-risking gate.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest
from radial_eigensolver import radial_spectrum
from scipy.special import jn_zeros, jnp_zeros, jv

R = 1.0


@pytest.mark.parametrize("m", [0, 1, 2, 3])
def test_tm_spectrum_matches_bessel_zeros(m):
    """Dirichlet (TM, E_z(R)=0): gamma R = j_{m,n}."""
    exact = jn_zeros(m, 6)
    g = np.sqrt(radial_spectrum(m, R, 14, 6, bc="dirichlet", n_low=6))
    assert np.max(np.abs(g - exact / R) / (exact / R)) < 1e-11


@pytest.mark.parametrize("m", [1, 2, 3])
def test_te_spectrum_matches_bessel_deriv_zeros(m):
    """Neumann (TE, H_z'(R)=0): gamma R = j'_{m,n}.  (m=0 has the trivial
    constant mode prepended, handled separately.)"""
    exact = jnp_zeros(m, 6)
    g = np.sqrt(radial_spectrum(m, R, 14, 6, bc="neumann", n_low=6))
    assert np.max(np.abs(g - exact / R) / (exact / R)) < 1e-11


@pytest.mark.parametrize("m", [1, 2])
def test_eigenfunctions_match_bessel_profiles(m):
    """The n-th eigenfunction equals J_m(j_{m,n} r / R) (up to scale/sign)."""
    zeros = jn_zeros(m, 4)
    w, vec, rg = radial_spectrum(m, R, 14, 6, bc="dirichlet", n_low=4,
                                 return_modes=True)
    for n in range(3):
        prof = vec[:, n]
        ref = jv(m, zeros[n] * rg / R)
        s = np.dot(prof, ref) / np.dot(prof, prof)
        rel = np.max(np.abs(s * prof - ref)) / np.max(np.abs(ref))
        assert rel < 1e-11


def test_stepindex_oracle_homogeneous_reduction():
    """The Milestone-2 step-index dispersion oracle must collapse to the
    homogeneous TM+TE Bessel spectrum when eps1 == eps2 -- a clean self-check
    of the 6x6 boundary matching + conventions, before it gates the coupled
    eigensolver."""
    from stepindex_oracle import stepindex_modes
    R, a, eps, k0, m = 1.0, 0.5, 4.0, 12.0, 1
    q2 = stepindex_modes(m, a, R, eps, eps, k0, n_modes=8) ** 2
    tm = eps * k0 ** 2 - (jn_zeros(m, 8) / R) ** 2
    te = eps * k0 ** 2 - (jnp_zeros(m, 8) / R) ** 2
    expected = np.concatenate([tm[tm > 0], te[te > 0]])
    for qq in q2:
        assert np.min(np.abs(expected - qq)) < 1e-6


def test_spectral_convergence():
    """Error drops geometrically with degree (spectral convergence)."""
    exact = jn_zeros(1, 4)[-1]
    errs = []
    for degree in (6, 9, 12):
        g = np.sqrt(radial_spectrum(1, R, 4, degree, bc="dirichlet",
                                    n_low=4))[-1]
        errs.append(abs(g - exact))
    # geometric (spectral) convergence: each refinement cuts the error by
    # at least ~5x (measured 25x then 10x)
    assert errs[1] < errs[0] * 0.2 and errs[2] < errs[1] * 0.2
