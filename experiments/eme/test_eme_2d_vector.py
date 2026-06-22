"""Validate the VECTOR (TE/TM) EME layer-mode solver (``eme_2d_vector.py``).

The full-Maxwell 2-D Bloch modes of a y-strip-sectioned crossed grating, built
from 1-D-x vector strip modes (Berreman-in-y) + the global block-``G`` lateral
interface residual, are checked against a direct Yee-staggered 2-D vector
finite-difference solve (``ref_2d_modes_vector``).  Because the EME is ANALYTIC in
y while the 2-D-FD is finite-difference in y, the 2-D-FD CONVERGES TO the EME as
``Ny -> inf`` (the EME is the exact-y limit) -- as in the scalar ``test_eme_2d.py``.

The mode-finder's validated regime is STRUCTURED layers (TE/TM split); a uniform
slab's high degeneracy makes its mode-finding unreliable (its dispersion is
validated here via the oracle instead).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
from eme_2d import strip_x_modes
from eme_2d_vector import (
    layer_vector_modes,
    mode_field_vec,
    ref_2d_modes_vector,
    strip_vector_modes,
    strips_to_eps_xy,
)

Lx = Ly = 1.0
k0 = 8.0
KY0 = np.pi                     # off the global band edge (cusp-free dispersion)


def _grating(Nx, e_lo, e_hi, duty=0.5):
    xg = (np.arange(Nx) + 0.5) / Nx
    return np.where(xg < duty, e_hi, e_lo).astype(float)


def _distinct(vals, rtol=3e-3):
    """Collapse near-equal values (e.g. the oracle's +-qz pairs) to one each."""
    out = []
    for v in np.sort(vals)[::-1]:
        if not out or abs(out[-1] - v) > rtol * max(abs(v), 1.0):
            out.append(v)
    return np.array(out)


def _oracle_band(strips, Nx, Ny, lo, hi, k=40):
    """Distinct physical oracle modes ``qz^2`` in ``(lo, hi)`` (reldiv-clean).
    Uses the sparse shift-invert oracle (~100x faster, returns distinct modes)
    centred on the band."""
    eps_xy = strips_to_eps_xy(strips, Lx, Nx, Ly, Ny)
    sigma = 1j * np.sqrt(0.5 * (lo + hi))
    qz2, _, reldiv = ref_2d_modes_vector(eps_xy, Lx, Ly, Nx, Ny, k0, ky0=KY0,
                                         return_vecs=True, k=k, sigma=sigma)
    return _distinct(qz2[(qz2 > lo) & (qz2 < hi) & (reldiv < 1e-2)])


def test_vector_strip_scalar_reduction():
    """At qz=0 the strip TE channel (Ez, E along the invariant z) reduces EXACTLY
    to the scalar ``eme_2d.strip_x_modes`` eigenvalues -- a byte-level check that
    the vector strip operator contains the scalar Helmholtz operator."""
    Nx = 28
    eps_x = _grating(Nx, 1.0, 4.0)
    lam_scalar, _ = strip_x_modes(eps_x, Lx, Nx, k0, 0.0)
    ky, _, _ = strip_vector_modes(eps_x, Lx, Nx, k0, 0.0, qz2=0.0)
    ky2 = (ky ** 2).real
    for lam in np.sort(lam_scalar)[::-1][:8]:
        assert np.min(np.abs(ky2 - lam)) < 1e-6


def test_vector_oracle_uniform_doubly_degenerate():
    """The 2-D vector FD oracle on a uniform layer gives the analytic plane-wave
    dispersion, each value DOUBLY degenerate (the +-qz pair), and is spurious-free
    (every physical mode has small reldiv)."""
    Nx = Ny = 16
    eps = 4.0
    eps_xy = np.full((Nx, Ny), eps, dtype=complex)
    qz2, _, reldiv = ref_2d_modes_vector(eps_xy, Lx, Ly, Nx, Ny, k0, ky0=KY0,
                                         return_vecs=True)
    top = np.sort(qz2[qz2 > 150])[::-1]
    anal = eps * k0 ** 2 - KY0 ** 2                  # (m,p)=(0,0): kx=0, ky=KY0
    assert abs(top[0] - anal) < 0.5                  # matches analytic dispersion
    assert np.sum(np.abs(qz2 - top[0]) < 1e-6) >= 2  # +-qz degeneracy
    assert np.max(reldiv[qz2 > 50]) < 1e-6           # spurious-free


def test_vector_structured_converges_from_2dfd():
    """The 2-D vector FD oracle converges to the EME (analytic-y) as Ny grows.
    Matched oracle->EME on the top oracle modes (robust to a spurious EME entry)."""
    Nx = 20
    strips = [(_grating(Nx, 1.0, 4.0), 0.5), (np.full(Nx, 2.0), 0.5)]
    eme = layer_vector_modes(strips, Lx, Nx, Ly, k0, (130, 256), ky0=KY0,
                             n_scan=400)
    prev = np.inf
    for Ny in (20, 40):
        ref = np.sort(_oracle_band(strips, Nx, Ny, 130, 256))[::-1][:2]
        err = max(min(abs(o - e) for e in eme) for o in ref)   # each oracle mode->EME
        assert err < prev or err < 1e-3                        # monotone -> the EME
        prev = err
    assert prev < 0.3                                          # converged (2nd order)


def test_vector_structured_completeness():
    """Full-band completeness regression (the test that catches the cascade
    conditioning bug the top-3 tests missed): the block-``G`` finder recovers
    the FULL band of a structured layer, not just the top modes.  The ill-
    conditioned Redheffer cascade residual found only ~2/16 of these modes."""
    Nx = 20
    strips = [(_grating(Nx, 1.0, 4.0), 0.5), (np.full(Nx, 2.0), 0.5)]
    ref = _oracle_band(strips, Nx, 56, 56, 259)
    eme = layer_vector_modes(strips, Lx, Nx, Ly, k0, (56, 259), ky0=KY0,
                             n_scan=500)
    recall = sum(min(abs(o - e) for e in eme) < 0.7 for o in ref)
    spurious = len(eme) - sum(min(abs(e - o) for o in ref) < 0.7 for e in eme)
    assert len(ref) >= 14                       # oracle finds the full band
    assert recall >= len(ref) - 3               # EME recovers all but a few (x-FD)
    assert recall >= 12                          # << the cascade's ~2; full band back
    assert spurious <= 2                         # rank-drop keeps it clean


def test_vector_no_duplicate_modes():
    """Reported modes are deduped -- no near-duplicates."""
    Nx = 20
    strips = [(_grating(Nx, 1.0, 4.0), 0.5), (np.full(Nx, 2.0), 0.5)]
    eme = layer_vector_modes(strips, Lx, Nx, Ly, k0, (130, 256), ky0=KY0,
                             n_scan=400)
    assert np.all(np.abs(np.diff(np.sort(eme))) > 0.3)


def test_vector_mode_field():
    """``mode_field_vec`` at a found mode returns a true mode field: the global
    block ``G`` is singular there (small ``sigma``) and the reconstructed
    tangential-E field is non-trivial."""
    Nx = 20
    strips = [(_grating(Nx, 1.0, 4.0), 0.5), (np.full(Nx, 2.0), 0.5)]
    eme = layer_vector_modes(strips, Lx, Nx, Ly, k0, (130, 256), ky0=KY0,
                             n_scan=400)
    qtop = _oracle_band(strips, Nx, 40, 130, 256)[0]        # a guaranteed-real mode
    q = eme[np.argmin(np.abs(eme - qtop))]                  # the EME mode matching it
    Ex, Ez, sigma = mode_field_vec(strips, Lx, Nx, Ly, k0, q, KY0, 40)
    assert sigma < 3e-3                          # confirmed a true mode (x-FD floor)
    assert max(np.abs(Ex).max(), np.abs(Ez).max()) > 1e-3   # non-trivial field
    assert Ex.shape == (Nx, 40) and Ez.shape == (Nx, 40)
