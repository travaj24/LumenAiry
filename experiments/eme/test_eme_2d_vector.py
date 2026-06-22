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
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from eme_2d import strip_x_modes
from eme_2d_vector import (
    eps_xy_to_strips,
    layer_vector_modes,
    layer_vector_modes_complex,
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


def _iso_tensor(eps_x):
    """Isotropic (Nx,3,3) tensor from a scalar/(Nx,) profile."""
    eps_x = np.atleast_1d(np.asarray(eps_x, dtype=complex))
    t = np.zeros((len(eps_x), 3, 3), dtype=complex)
    t[:, 0, 0] = t[:, 1, 1] = t[:, 2, 2] = eps_x
    return t


def _berreman_ky(eps33, qz):
    """Forward ``|ky|`` of a UNIFORM tensor strip via the role-swapped (y<->z)
    Berreman planar Delta -- the anisotropic oracle for the strip generator."""
    from lumenairy.elements.berreman import _berreman_delta
    epsb = np.asarray(eps33, dtype=complex)[[0, 2, 1]][:, [0, 2, 1]]
    gam = np.linalg.eigvals(_berreman_delta(epsb, 0.0, qz / k0))
    return np.sort(np.unique(np.round(np.abs(-1j * gam * k0), 5)))


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


def test_vector_geometry_wrapper():
    """eps_xy_to_strips (arbitrary eps(x,y) -> y-staircase) reproduces a hand-built
    strip list -- the EME accepts an arbitrary cell, not only a strip list."""
    Nx = 20

    def eps_fn(x, y):                            # == the hand-built 2-strip cell
        if y < 0.5:
            return 4.0 if x < 0.5 else 1.0
        return 2.0

    hand = [(_grating(Nx, 1.0, 4.0), 0.5), (np.full(Nx, 2.0), 0.5)]
    wrap = eps_xy_to_strips(eps_fn, Nx, 2, Lx, Ly)
    assert np.allclose(wrap[0][0], hand[0][0]) and np.allclose(wrap[1][0], hand[1][0])
    mh = np.sort(layer_vector_modes(hand, Lx, Nx, Ly, k0, (130, 256), ky0=KY0))[::-1]
    mw = np.sort(layer_vector_modes(wrap, Lx, Nx, Ly, k0, (130, 256), ky0=KY0))[::-1]
    assert np.allclose(mh[:3], mw[:3])           # wrapper modes == hand-built modes


def test_vector_verify_removes_spurious():
    """verify=True drops the ~1 spurious near-threshold candidate via an FD-oracle
    cross-check, without losing real modes (recall preserved, spurious -> 0)."""
    Nx = 20
    strips = [(_grating(Nx, 1.0, 4.0), 0.5), (np.full(Nx, 2.0), 0.5)]
    ref = _oracle_band(strips, Nx, 56, 56, 259)
    plain = layer_vector_modes(strips, Lx, Nx, Ly, k0, (56, 259), ky0=KY0, n_scan=500)
    verified = layer_vector_modes(strips, Lx, Nx, Ly, k0, (56, 259), ky0=KY0,
                                  n_scan=500, verify=True)
    rec = sum(min(abs(o - e) for e in verified) < 0.7 for o in ref)
    spur = len(verified) - sum(min(abs(e - o) for o in ref) < 0.7 for e in verified)
    assert len(verified) <= len(plain)           # verify only ever REMOVES candidates
    assert rec >= len(ref) - 1                    # real modes preserved
    assert spur == 0                              # spurious removed


def test_vector_multiplicity():
    """return_multiplicity reports the per-mode degeneracy (the rank-drop order) --
    1 for every mode of the non-degenerate (TE/TM-split) structured cell."""
    Nx = 20
    strips = [(_grating(Nx, 1.0, 4.0), 0.5), (np.full(Nx, 2.0), 0.5)]
    qz2, mult = layer_vector_modes(strips, Lx, Nx, Ly, k0, (130, 256), ky0=KY0,
                                   return_multiplicity=True)
    assert qz2.shape == mult.shape and mult.dtype.kind == "i"
    assert np.all(mult == 1)                     # non-degenerate -> all multiplicity 1


def test_vector_anisotropic_reduces_to_scalar():
    """An isotropic (Nx,3,3) tensor reproduces the SCALAR strip modes byte-exactly
    -- the load-bearing gate that the anisotropic generator derivation is correct
    (it reduces to the known-correct scalar code)."""
    Nx = 20
    eps_x = _grating(Nx, 2.0, 4.0)
    for qz2 in (0.0, 9.0, 36.0):
        ky_s = np.sort_complex(strip_vector_modes(eps_x, Lx, Nx, k0, 0.37, qz2)[0])
        ky_t = np.sort_complex(strip_vector_modes(_iso_tensor(eps_x), Lx, Nx, k0,
                                                  0.37, qz2)[0])
        assert np.max(np.abs(ky_s - ky_t)) < 1e-10


def test_vector_anisotropic_uniform_diagonal_vs_berreman():
    """A uniform DIAGONAL-birefringent strip's kx=0 modes match the role-swapped
    Berreman planar dispersion (ordinary + extraordinary)."""
    qz = 0.5 * k0
    eps33 = np.diag([2.0, 4.0, 3.0]).astype(complex)
    ky = strip_vector_modes(eps33[None, :, :], Lx, 1, k0, 0.0, qz ** 2)[0]
    got = np.sort(np.unique(np.round(np.abs(ky), 5)))
    assert np.allclose(got, _berreman_ky(eps33, qz))


def test_vector_anisotropic_exz_vs_berreman():
    """Out-of-plane exz/ezx coupling (symmetric AND asymmetric exz != ezx) matches
    the role-swapped Berreman oracle -- the novel anisotropy beyond diagonal."""
    qz = 0.5 * k0
    for exz, ezx in [(0.6, 0.6), (0.8, 0.5)]:
        e = np.array([[4, 0, exz], [0, 4, 0], [ezx, 0, 4]], dtype=complex)
        ky = strip_vector_modes(e[None, :, :], Lx, 1, k0, 0.0, qz ** 2)[0]
        got = np.sort(np.unique(np.round(np.abs(ky), 5)))
        assert np.allclose(got, _berreman_ky(e, qz))


def test_vector_anisotropic_eyz_raises():
    """eyz/ezy (yz) coupling is DEFERRED -- it breaks the block-anti-diagonal BC
    structure and was error-prone to derive, so it raises (no silent wrong answer)."""
    import pytest
    e = np.array([[4, 0, 0], [0, 4, 0.5], [0, 0.5, 4]], dtype=complex)
    with pytest.raises(NotImplementedError):
        strip_vector_modes(e[None, :, :], Lx, 1, k0, 0.0, 9.0)


def test_vector_anisotropic_oracle_returnvecs():
    """The tensor 2-D-FD oracle runs with return_vecs (the reldiv path) for a
    diagonal tensor -- regression for the ``is_tensor`` dtype check (a (N,3,3)
    eps was mis-detected and crashed the scalar reldiv branch)."""
    Nx = Ny = 10
    et = np.zeros((Nx, Ny, 3, 3), dtype=complex)
    et[:, :, 0, 0], et[:, :, 1, 1], et[:, :, 2, 2] = 2.0, 4.0, 3.0
    q, _, rd = ref_2d_modes_vector(et, Lx, Ly, Nx, Ny, k0, ky0=KY0,
                                   return_vecs=True, k=4)
    assert np.all(np.isfinite(rd))               # crashed before the fix
    assert rd[np.argmax(q)] < 1e-2               # physical mode is divergence-clean


def test_vector_banded_solver():
    """solver='banded' (the O(S) inverse-power sigma_min on the block-tridiagonal
    G) finds the SAME modes as the default dense solver -- the fine-y-staircase
    speedup (it grows with the strip count S)."""
    Nx = 16
    strips = [(_grating(Nx, 1.0, 4.0), 0.5), (np.full(Nx, 2.0), 0.5)]
    md = np.sort(layer_vector_modes(strips, Lx, Nx, Ly, k0, (130, 256),
                                    ky0=KY0))[::-1]
    mb = np.sort(layer_vector_modes(strips, Lx, Nx, Ly, k0, (130, 256), ky0=KY0,
                                    solver="banded"))[::-1]
    assert np.allclose(md[:3], mb[:3])


def test_vector_magnetic_mu():
    """Scalar permeability mu(x): the strip dispersion is eps*mu*k0^2 - kx^2 - ky^2,
    the oracle reproduces it (real + lossy), and mu=1 is byte-identical."""
    qz = 3.0
    for eps, mu in [(4.0, 2.0), (2.25, 1.8), (4.0, 0.6)]:
        ky = strip_vector_modes(np.full(1, eps, complex), Lx, 1, k0, 0.0, qz ** 2,
                                mu_x=mu)[0]
        assert abs(np.abs(ky).max() - np.sqrt(eps * mu * k0 ** 2 - qz ** 2)) < 1e-9
    Nx = Ny = 18                                     # lossy magnetic oracle: exact loss
    eps, mu = 4.0, 2.5 + 0.05j
    q = ref_2d_modes_vector(np.full((Nx, Ny), eps, complex), Lx, Ly, Nx, Ny, k0,
                            ky0=KY0, mu_xy=np.full((Nx, Ny), mu, complex),
                            return_complex=True)
    top = q[np.argmax(q.real)]
    assert abs(top.imag - (eps * mu * k0 ** 2 - KY0 ** 2).imag) < 1e-3
    # mu=1 is byte-identical to the no-mu strip modes
    e = _grating(16, 1.0, 4.0)
    k_no = np.sort_complex(strip_vector_modes(e, Lx, 16, k0, 0.37, 9.0)[0])
    k_one = np.sort_complex(strip_vector_modes(e, Lx, 16, k0, 0.37, 9.0, mu_x=1.0)[0])
    assert np.array_equal(k_no, k_one)


def test_vector_beyn_complex_lossy():
    """The SEEDED Beyn refiner reaches COMPLEX (lossy) qz^2 modes the real-axis
    scan structurally cannot: seed from the coarse complex oracle, refine to the
    EME's own complex mode (genuinely complex, tracking the oracle to the x-FD
    floor)."""
    Nx = 24
    xg = (np.arange(Nx) + 0.5) / Nx
    g = np.where(xg < 0.5, 4.0 + 0.08j, 1.0)         # complex-preserving (the loss)
    strips = [(g, 0.5), (np.full(Nx, 2.0 + 0j), 0.5)]
    eps_xy = strips_to_eps_xy(strips, Lx, Nx, Ly, 64)
    q, _, rd = ref_2d_modes_vector(eps_xy, Lx, Ly, Nx, 64, k0, ky0=KY0,
                                   return_vecs=True, k=12, sigma=1j * np.sqrt(190.0),
                                   return_complex=True)
    seeds = q[rd < 5e-2]
    seeds = seeds[np.argsort(seeds.real)[::-1]][:3]
    modes = layer_vector_modes_complex(strips, Lx, Nx, Ly, k0, seeds, ky0=KY0)
    assert len(modes) == 3
    assert np.all(np.abs(modes.imag) > 1e-2)         # genuinely complex (lossy)
    for sd in seeds:                                  # track the oracle seeds
        assert min(abs(sd - m) for m in modes) < 0.2


def test_vector_oracle_lossy_complex():
    """The FD oracle solves LOSSY layers exactly: ``return_complex=True`` gives the
    complex ``qz^2`` of a uniform lossy slab matching ``eps k0^2 - kx^2 - ky^2``
    (the modal loss ``Im(qz^2)`` is exact).  Default (real) path is unchanged."""
    Nx = Ny = 18
    eps = 4.0 + 0.05j
    q = ref_2d_modes_vector(np.full((Nx, Ny), eps, dtype=complex), Lx, Ly, Nx, Ny,
                            k0, ky0=KY0, return_complex=True)
    top = q[np.argmax(q.real)]
    anal = eps * k0 ** 2 - KY0 ** 2              # (0,0) mode: kx=0, ky=KY0
    assert abs(top.imag - anal.imag) < 1e-3      # modal loss exact
    assert abs(top.real - anal.real) < 0.5       # real part to x-FD accuracy
    qr = ref_2d_modes_vector(np.full((Nx, Ny), 4.0, dtype=complex), Lx, Ly, Nx, Ny,
                             k0, ky0=KY0)
    assert np.isrealobj(qr)                      # default path still real
