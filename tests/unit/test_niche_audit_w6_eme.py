"""EME physics-interior audit (W6, 2026-07-26) -- oracles + discriminating pins.

Territory: ``lumenairy/elements/eme/**`` -- the honest coverage gap of
``AUDIT_ADVERSARIAL_CODEBASE_2026_07_25.md`` (never numerically validated there;
the 2026-07-09 EME audit was read-only and found "none above nit level").

ORACLES built here (independent of the code under test):
  O1  analytic symmetric 3-layer slab dispersion, bisected from scratch
      (``_slab_betas``) -- the EME is ANALYTIC in y, so an x-uniform
      [clad|core|clad] strip stack must reproduce beta^2 to ~1e-8 relative;
  O2  lossless power conservation of the lateral cell S-matrix (with the strip
      basis Hermitian-orthonormal, ``sum_i Re(ky_i)(|S11[i,j]|^2 +
      |S21[i,j]|^2) == Re(ky_j)`` for every propagating input j);
  O3  the analytic Airy / Fabry-Perot slab (lossless AND lossy) for the
      mode-matching diffraction driver;
  O4  the independent 2-D-FD mode oracles already in the package, used as a
      recall/spurious cross-check at nonzero Bloch phase.

FINDINGS PINNED (each pin block verified to FAIL on a pre-fix worktree of
3a1da2b -- see the audit report):
  W6-1  CRITICAL: real eps at ``kx0 != 0`` was routed to ``eig`` although the
        strip operator is HERMITIAN there.  Roundoff-complex ``lam`` flipped
        ``np.sqrt``'s branch onto the exponentially GROWING lateral propagator
        and the bilinear normaliser left the basis non-orthonormal, so
        ``layer_modes`` returned 68 modes at kx0=0.37 of which 0/3 were real.
  W6-2  HIGH: ``mode_match`` carried ``exp(+|qz| depth)`` (backward amplitudes
        referenced at z=0), so ``cond(A) ~ exp(2|qz|max depth)`` and past ~1e16
        the answer collapsed to R_00=1 / T_00=0 for a HOMOGENEOUS medium, with
        ``energy = 1.000000`` masking it.
  W6-3  HIGH: ``diffraction_fd`` discarded ``Im(qz^2)``, so an absorbing slab
        reported ``energy = 1.000000``.
  W6-4  MEDIUM: the rasterizers lacked the layer finders' ``sum(h) == Ly``
        guard and silently left grid cells at ``eps = 0`` (-> inf/NaN oracle).
  W6-5  MEDIUM: a junk ``solver=`` value silently fell through to dense.
  W6-6  MEDIUM: the ``_DETECT_PPU`` detection grid was not scale-invariant.
  W6-7  MEDIUM: ``sigma`` was silently inert without ``k`` in both FD oracles.
  W6-8  LOW: the scalar sparse oracle lacked the fixed ARPACK ``v0`` its two
        siblings have (output depended on the global NumPy RNG).
  W6-9  LOW: lossy ``Im(qz^2)``-discard warning missing on the JAX scalar twin
        and on the NumPy VECTOR oracle (the scalar NumPy sibling warned).
  W6-10..14 LOW: silent empty result for n_scan<=1; opaque messages for a
        tensor grid / a zero-norm mode column / iters=0 / a traced Bloch phase.

Cross-platform tolerances are MEASURED, not aspirational: eigensolve-derived
quantities are compared with rel >= 1e-9 (observed BLAS drift 1e-13..1e-11) and
the analytic-slab comparison with rel 1e-6 (observed 2.4e-8 for the confined
modes, the rest being periodic-cell cladding truncation).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import warnings

import numpy as np
import pytest

from lumenairy.elements.eme import eme_2d, eme_2d_vector, eme_diffraction

PI = np.pi


# =========================================================================== #
#  ORACLE O1 -- analytic symmetric 3-layer slab, bisected independently        #
# =========================================================================== #
def _slab_betas(n1, n2, d, k0, nscan=40001):
    """All guided ``b = beta^2`` of a symmetric slab (core index ``n1``, thickness
    ``d``, cladding ``n2``) from the textbook dispersion equation, found by sign
    scan + bisection -- no package code involved.

        even:  kappa tan(kappa d/2) = gamma
        odd:  -kappa cot(kappa d/2) = gamma
        kappa = sqrt(n1^2 k0^2 - b),  gamma = sqrt(b - n2^2 k0^2)
    """
    lo, hi = (n2 * k0) ** 2, (n1 * k0) ** 2

    def F(b, parity):
        kap = np.sqrt(max(hi - b, 0.0))
        gam = np.sqrt(max(b - lo, 0.0))
        if parity == 0:
            return kap * np.sin(kap * d / 2) - gam * np.cos(kap * d / 2)
        return kap * np.cos(kap * d / 2) + gam * np.sin(kap * d / 2)

    out = []
    bs = np.linspace(lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo), nscan)
    for parity in (0, 1):
        fv = np.array([F(b, parity) for b in bs])
        for i in range(nscan - 1):
            if fv[i] * fv[i + 1] < 0:
                a, c = bs[i], bs[i + 1]
                for _ in range(120):
                    m = 0.5 * (a + c)
                    if F(a, parity) * F(m, parity) <= 0:
                        c = m
                    else:
                        a = m
                out.append(0.5 * (a + c))
    return np.sort(np.array(out))[::-1]


def _slab_strips(n1, n2, d, Ly, Nx):
    """[clad | core | clad] x-uniform y-strips, heights summing to Ly."""
    hc = (Ly - d) / 2.0
    return [(np.full(Nx, n2 ** 2), hc), (np.full(Nx, n1 ** 2), d),
            (np.full(Nx, n2 ** 2), hc)]


# =========================================================================== #
#  ORACLE O2 -- lossless power conservation of the lateral cell S-matrix       #
# =========================================================================== #
def _cell_flux_violation(strips, Lx, Nx, k0, kx0, qz2):
    """max relative violation of ``sum_i Re(ky_i)(|S11[i,j]|^2 + |S21[i,j]|^2)
    == Re(ky_j)`` over the propagating inputs j.  Both S-matrix ports live in
    strip 0's basis, so one ``diag(Re ky)`` metric serves both sides; the relation
    is exact for a lossless cell IFF the strip basis is orthonormal under the
    inner product the flux is written in (the Hermitian one)."""
    sm = [(eme_2d.strip_x_modes(e, Lx, Nx, k0, kx0), h) for e, h in strips]
    S11, _S12, S21, _S22 = eme_2d.cell_smatrix(sm, qz2)
    (lam0, _Phi0), _h = sm[0]
    ky = np.sqrt(np.asarray(lam0) - qz2 + 0j)
    scale = max(1.0, float(np.max(np.abs(ky))))
    K = np.where(np.abs(ky.imag) < 1e-9 * scale, ky.real, 0.0)
    worst = 0.0
    for j in np.nonzero(K > 1e-9)[0]:
        out = float(np.sum(K * (np.abs(S11[:, j]) ** 2 + np.abs(S21[:, j]) ** 2)))
        worst = max(worst, abs(out - K[j]) / K[j])
    return worst


# =========================================================================== #
#  ORACLE O3 -- analytic Airy / Fabry-Perot slab (lossless and lossy)          #
# =========================================================================== #
def _airy(n_sup, n_lay, n_sub, k0, depth, kt=0.0):
    """Scalar slab R / T (admittance = kz), the exact answer a UNIFORM layer's
    mode matching must reproduce."""
    kz_s = np.sqrt((n_sup * k0) ** 2 - kt ** 2 + 0j)
    kz_l = np.sqrt(complex(n_lay) ** 2 * k0 ** 2 - kt ** 2 + 0j)
    kz_b = np.sqrt((n_sub * k0) ** 2 - kt ** 2 + 0j)
    r1 = (kz_s - kz_l) / (kz_s + kz_l)
    r2 = (kz_l - kz_b) / (kz_l + kz_b)
    t1, t2 = 2 * kz_s / (kz_s + kz_l), 2 * kz_l / (kz_l + kz_b)
    ph = np.exp(1j * kz_l * depth)
    den = 1.0 + r1 * r2 * ph ** 2
    r, t = (r1 + r2 * ph ** 2) / den, t1 * t2 * ph / den
    return float(abs(r) ** 2), float((kz_b.real / kz_s.real) * abs(t) ** 2)


def _reference_cell(Nx=16):
    """The reference structured 2-strip cell used throughout the EME tests."""
    xg = (np.arange(Nx) + 0.5) / Nx
    grat = np.where(xg < 0.5, 4.0, 1.0)
    return [(grat, 0.5), (np.full(Nx, 2.0), 0.5)]


# =========================================================================== #
#  W6-1  the strip operator is HERMITIAN at any real kx0 (CRITICAL)            #
# =========================================================================== #
@pytest.mark.parametrize("kx0", [0.0, 0.37, 1.1, 3.0])
def test_w6_1_strip_operator_hermitian_and_basis_orthonormal(kx0):
    """Real eps -> ``A = A^H`` at ANY real kx0 (the wrap corners are conjugates),
    so ``eigh`` is the right solver: real ascending ``lam`` and an orthonormal
    ``Phi``.  PRE-FIX at kx0=0.37 this measured ``max|Phi^H Phi - I| = 43.2``
    with column 2-norms spread 1.01..6.65 (the complex-SYMMETRIC bilinear
    normaliser applied to a Hermitian operator)."""
    Nx = 16
    xg = (np.arange(Nx) + 0.5) / Nx
    for eps in (np.full(Nx, 2.0), np.where(xg < 0.5, 4.0, 1.0)):
        lam, Phi = eme_2d.strip_x_modes(eps, 1.0, Nx, 8.0, kx0)
        lam = np.asarray(lam)
        assert np.max(np.abs(lam.imag)) < 1e-12 * max(
            1.0, float(np.max(np.abs(lam.real))))           # real spectrum
        assert np.all(np.diff(lam.real) >= -1e-9)           # eigh -> ascending
        I = np.eye(Nx)
        assert np.max(np.abs(Phi.conj().T @ Phi - I)) < 1e-10   # orthonormal
        assert np.max(np.abs(Phi @ Phi.conj().T - I)) < 1e-10   # and complete
        assert np.allclose(np.linalg.norm(Phi, axis=0), 1.0, rtol=1e-10)


@pytest.mark.parametrize("kx0", [0.0, 0.37, 1.1])
def test_w6_1_no_growing_lateral_propagator(kx0):
    """The S-matrix cascade's whole premise: ``_prop`` must only ever carry
    DECAYING exponentials.  PRE-FIX, at kx0=0.37, 8-11 of 16 strip modes came
    back on the GROWING branch with ``max|exp(i ky h)| = 6.2e6`` -- the T-matrix
    blow-up S-matrices exist to avoid."""
    Nx, h = 16, 0.5
    xg = (np.arange(Nx) + 0.5) / Nx
    for eps in (np.full(Nx, 2.0), np.where(xg < 0.5, 4.0, 1.0),
                np.where(xg < 0.5, 4.0 + 0.2j, 1.0 + 0.0j)):
        for qz2 in (10.0, 40.0, 90.0, 200.0):
            lam, Phi = eme_2d.strip_x_modes(eps, 1.0, Nx, 8.0, kx0)
            _W, _V, ky = eme_2d._wv(lam, Phi, qz2)
            assert np.all(np.asarray(ky).imag >= -1e-12)
            assert np.max(np.abs(np.exp(1j * np.asarray(ky) * h))) < 1.0 + 1e-9


@pytest.mark.parametrize("kx0", [0.0, 0.37, 1.1])
@pytest.mark.parametrize("qz2", [10.0, 40.0, 90.0])
def test_w6_1_cell_smatrix_conserves_power(kx0, qz2):
    """ORACLE O2.  A lossless lateral cell must conserve power exactly.  PRE-FIX
    the STRUCTURED cell violated it by 1.5e-2 (kx0=0.37) and 1.8e-1 (kx0=1.1)
    while kx0=0 held at 1e-16 -- the signature of a non-orthonormal basis."""
    Nx = 16
    for strips in (_reference_cell(Nx),
                   [(np.full(Nx, 2.0), 0.5), (np.full(Nx, 2.0), 0.5)]):
        v = _cell_flux_violation(strips, 1.0, Nx, 8.0, kx0, qz2)
        assert v < 1e-10, f"lossless flux violated by {v:.3e}"


@pytest.mark.parametrize("kx0", [0.0, 0.37, 1.1])
@pytest.mark.parametrize("ky0", [0.0, PI])
def test_w6_1_layer_modes_recall_vs_fd_oracle(kx0, ky0):
    """ORACLE O4.  Every ``layer_modes`` root must be a genuine 2-D Bloch mode of
    the same cell (independent FD oracle) and every FD mode must be found.
    PRE-FIX at kx0=0.37/ky0=pi this returned 68 roots, recall 0/3, all spurious;
    the nearest FD eigenvalue to the best of them was 0.56 away."""
    Nx, Lx, Ly, k0 = 16, 1.0, 1.0, 8.0
    strips = _reference_cell(Nx)
    lo, hi = 120.0, 4.0 * k0 ** 2
    got = eme_2d.layer_modes(strips, Lx, Nx, Ly, k0, (lo, hi), kx0=kx0, ky0=ky0)
    eps_xy = eme_2d.strips_to_eps_xy(strips, Lx, Nx, Ly, 64)
    w = eme_2d.ref_2d_modes(eps_xy, Lx, Ly, Nx, 64, k0, kx0=kx0, ky0=ky0)
    ref = np.sort(w[(w >= lo) & (w <= hi)])[::-1]
    assert len(ref) >= 3
    # the EME is the Ny -> inf limit of the FD, so 0.5 in qz^2 is the y-FD gap
    for g in got:
        assert np.min(np.abs(ref - g)) < 0.5, f"spurious root qz2={g:.4f}"
    for r in ref:
        assert np.min(np.abs(got - r)) < 0.5, f"missed FD mode qz2={r:.4f}"


@pytest.mark.parametrize(
    "n1,n2,d,Ly,lam0,kx0",
    [(1.50, 1.45, 2.00, 8.0, 1.0, 0.0),
     (2.00, 1.00, 0.50, 6.0, 1.0, 0.0),
     (3.48, 1.44, 0.30, 4.0, 1.55, 0.0),
     (2.00, 1.00, 0.50, 6.0, 1.0, 0.37),
     (3.48, 1.44, 0.30, 4.0, 1.55, 1.10)])
def test_w6_1_layer_modes_matches_analytic_slab(n1, n2, d, Ly, lam0, kx0):
    """ORACLE O1.  An x-uniform [clad|core|clad] cell at Bloch phase ``kx0`` has
    ``qz^2 = beta^2_slab - kx0^2`` for the ``m = 0`` transverse order (the discrete
    x-Laplacian is EXACT on ``exp(i kx0 x)``), so the EME must reproduce the
    independently bisected slab dispersion.  The kx0 != 0 rows FAIL PRE-FIX
    (garbage roots); the kx0 = 0 rows are a standing physics lock."""
    Nx, Lx, k0 = 8, 1.0, 2 * PI / lam0
    an = _slab_betas(n1, n2, d, k0)
    assert len(an) >= 1
    want = an - kx0 ** 2
    strips = _slab_strips(n1, n2, d, Ly, Nx)
    lo = (n2 * k0) ** 2 - kx0 ** 2 + 1e-6
    hi = (n1 * k0) ** 2 - kx0 ** 2 - 1e-6
    got = eme_2d.layer_modes(strips, Lx, Nx, Ly, k0, (lo, hi), kx0=kx0,
                             n_scan=4000)
    for b in want:
        # confined modes land at ~2.4e-8 relative; the weakly-confined ones carry
        # the periodic-cell cladding-truncation error (the analytic slab has
        # INFINITE cladding), hence the 1e-4 relative window here
        d_rel = float(np.min(np.abs(got - b))) / abs(b)
        assert d_rel < 1e-4, f"analytic slab b={b:.8f} not found (rel {d_rel:.2e})"


def test_w6_1_analytic_slab_confined_mode_is_tight():
    """Same oracle, tightened on the WELL-CONFINED mode alone: 1e-6 relative
    (measured 2.4e-8 -- the EME is analytic in y, so only the exact x-FD and the
    root-finder tolerance enter)."""
    n1, n2, d, Ly, k0 = 2.0, 1.0, 0.5, 6.0, 2 * PI
    an = _slab_betas(n1, n2, d, k0)
    strips = _slab_strips(n1, n2, d, Ly, 8)
    got = eme_2d.layer_modes(strips, 1.0, 8, Ly, k0, ((n2 * k0) ** 2 + 1e-6,
                                                     (n1 * k0) ** 2 - 1e-6),
                             n_scan=4000)
    top = an[0]                                   # the fundamental (most confined)
    assert float(np.min(np.abs(got - top))) / top < 1e-6


# =========================================================================== #
#  W6-2  mode_match must not carry a growing exponential (HIGH)                #
# =========================================================================== #
@pytest.mark.parametrize("depth", [0.1, 2.0, 5.3, 8.0, 12.0])
def test_w6_2_index_matched_layer_transmits_unity_at_any_depth(depth):
    """A layer whose index EQUALS both half-spaces is not an interface at all:
    ``T_00 = 1``, ``R = 0``, at ANY depth.  PRE-FIX ``depth >= 5.3`` returned
    ``R_00 = 1.000000`` / ``T_00 = 0.000000`` -- total reflection from a
    HOMOGENEOUS medium -- with ``energy = 1.000000`` masking it (cond(A) had
    reached 1.4e17 through ``exp(+|qz| depth)``)."""
    Nx = Ny = 8
    k0 = 2 * PI / 1.3
    eps = np.full((Nx, Ny), 1.0)
    res = eme_diffraction.diffraction_fd(eps, 1.0, 1.0, Nx, Ny, k0, 1.0, 1.0,
                                         depth, 1, 1)
    i0 = res["orders"].index((0, 0))
    assert res["T"][i0] == pytest.approx(1.0, abs=1e-9)
    assert res["R"][i0] < 1e-12
    assert res["R"].sum() < 1e-12
    assert res["energy"] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("depth", [0.2, 2.0, 5.3, 8.0, 12.0])
def test_w6_2_uniform_slab_matches_airy_at_any_depth(depth):
    """ORACLE O3.  PRE-FIX ``n = 1.5`` collapsed to ``R_00 = 1.0`` at
    ``depth >= 8`` (analytic 0.146094 / 0.105215) and already carried a 3e-12
    error at depth 6; POST-FIX the error is ~5e-15 at every depth."""
    Nx = Ny = 8
    k0 = 2 * PI / 1.3
    n = 1.5
    Ra, Ta = _airy(1.0, n, 1.0, k0, depth)
    eps = np.full((Nx, Ny), n ** 2)
    res = eme_diffraction.diffraction_fd(eps, 1.0, 1.0, Nx, Ny, k0, 1.0, 1.0,
                                         depth, 1, 1)
    i0 = res["orders"].index((0, 0))
    assert res["R"][i0] == pytest.approx(Ra, abs=1e-9)
    assert res["T"][i0] == pytest.approx(Ta, abs=1e-9)
    assert res["energy"] == pytest.approx(1.0, abs=1e-9)


def test_w6_2_mode_match_matrix_stays_conditioned():
    """The mechanism, pinned directly: no ``exp(+|Im qz| depth)`` factor may reach
    the matched system, so the deep-slab solve stays well conditioned.  Probed
    through the public driver by checking that doubling the depth does NOT degrade
    the analytic agreement (pre-fix the error grew as exp(2|qz|max depth))."""
    Nx = Ny = 8
    k0 = 2 * PI / 1.3
    errs = []
    for depth in (1.0, 2.0, 4.0, 8.0, 16.0):
        Ra, _ = _airy(1.0, 1.5, 1.0, k0, depth)
        eps = np.full((Nx, Ny), 2.25)
        res = eme_diffraction.diffraction_fd(eps, 1.0, 1.0, Nx, Ny, k0, 1.0, 1.0,
                                            depth, 1, 1)
        errs.append(abs(res["R"][res["orders"].index((0, 0))] - Ra))
    assert max(errs) < 1e-9, f"depth-growing error {errs}"


# =========================================================================== #
#  W6-3  a lossy layer must absorb (HIGH)                                     #
# =========================================================================== #
@pytest.mark.parametrize("n_lay,depth", [(1.5 + 0.05j, 0.4), (1.5 + 0.05j, 1.5),
                                         (1.5 + 0.2j, 0.4), (1.5 + 0.2j, 4.0)])
def test_w6_3_lossy_slab_energy_matches_analytic(n_lay, depth):
    """ORACLE O3 with a complex index.  PRE-FIX ``diffraction_fd`` took
    ``Re(qz^2)`` and reported ``energy = 1.000000`` for every case -- e.g. at
    ``n = 1.5 + 0.2j``, ``depth = 4`` it claimed all the light emerged
    (``R_00 = 0.032362``) where the exact answer is ``R + T = 0.046505``
    (``R_00 = 0.046099``): 95% of the light is absorbed."""
    Nx = Ny = 8
    k0 = 2 * PI / 1.3
    Ra, Ta = _airy(1.0, n_lay, 1.0, k0, depth)
    eps = np.full((Nx, Ny), n_lay ** 2)
    res = eme_diffraction.diffraction_fd(eps, 1.0, 1.0, Nx, Ny, k0, 1.0, 1.0,
                                         depth, 1, 1)
    i0 = res["orders"].index((0, 0))
    assert res["R"][i0] == pytest.approx(Ra, abs=1e-8)
    assert res["T"][i0] == pytest.approx(Ta, abs=1e-8)
    assert res["energy"] == pytest.approx(Ra + Ta, abs=1e-8)
    assert res["energy"] < 1.0 - 1e-3            # it MUST absorb


def test_w6_3_lossless_layer_still_conserves_energy_exactly():
    """Guard on the W6-3 fix: a REAL eps takes the identical legacy path."""
    Nx = Ny = 8
    eps = np.full((Nx, Ny), 4.0)
    res = eme_diffraction.diffraction_fd(eps, 1.0, 1.0, Nx, Ny, 4.0, 1.0, 2.25,
                                         0.5, 1, 1)
    assert res["energy"] == pytest.approx(1.0, abs=1e-12)


# =========================================================================== #
#  W6-4  the rasterizers enforce sum(h) == Ly (MEDIUM)                        #
# =========================================================================== #
def test_w6_4_strips_to_eps_xy_rejects_short_heights():
    """PRE-FIX this silently produced a grid with 24/32 cells at ``eps = 0``,
    which the vector oracle's ``1/(k0 eps)`` turns into inf/NaN."""
    bad = [(np.full(4, 4.0), 0.25)]
    for fn in (eme_2d_vector.strips_to_eps_xy, eme_2d.strips_to_eps_xy):
        with pytest.raises(ValueError, match="heights sum to"):
            fn(bad, 1.0, 4, 1.0, 8)
    with pytest.raises(ValueError, match="heights sum to"):
        eme_2d_vector._strips_to_mu_xy([(np.full(4, 4.0), 0.25, 2.0)], 4, 1.0, 8)


def test_w6_4_consistent_heights_rasterize_with_no_zero_cells():
    """The valid case is unchanged and covers every y row."""
    g = eme_2d_vector.strips_to_eps_xy([(np.full(4, 4.0), 0.4),
                                        (np.full(4, 1.0), 0.6)], 1.0, 4, 1.0, 10)
    assert g.shape == (4, 10)
    assert np.min(np.abs(g)) > 0.0
    mu = eme_2d_vector._strips_to_mu_xy([(np.full(4, 4.0), 0.4, 2.0),
                                        (np.full(4, 1.0), 0.6, 2.0)], 4, 1.0, 10)
    assert np.all(mu == 2.0)


# =========================================================================== #
#  W6-5  junk solver selectors are rejected (MEDIUM)                          #
# =========================================================================== #
@pytest.mark.parametrize("junk", ["bananas", "DENSE", "sparse", "", None, 7])
def test_w6_5_junk_solver_rejected(junk):
    """PRE-FIX every one of these silently fell through to the dense path
    (measured bit-identical to ``solver='dense'``)."""
    strips = _reference_cell(8)
    with pytest.raises(ValueError, match="unknown solver"):
        eme_2d_vector.dispersion_vec(strips, 1.0, 8, 8.0, 0.0, 210.0, PI, 1.0,
                                     junk)
    with pytest.raises(ValueError, match="unknown solver"):
        eme_2d_vector.layer_vector_modes(strips, 1.0, 8, 1.0, 8.0, (200.0, 210.0),
                                        ky0=PI, n_scan=10, solver=junk)


def test_w6_5_both_valid_solvers_agree():
    """Verified-clean lock with its MEASURED tolerance: ``banded`` reproduces the
    dense ``sigma_min`` to 1.7e-3 relative on the reference cell (documented
    'same zeros')."""
    strips = _reference_cell(16)
    qs = np.linspace(150.0, 250.0, 11)
    dd = np.array([eme_2d_vector.dispersion_vec(strips, 1.0, 16, 8.0, 0.0, q, PI,
                                               1.0, "dense") for q in qs])
    bb = np.array([eme_2d_vector.dispersion_vec(strips, 1.0, 16, 8.0, 0.0, q, PI,
                                               1.0, "banded") for q in qs])
    assert np.max(np.abs(dd - bb) / np.maximum(dd, 1e-30)) < 1e-2


# =========================================================================== #
#  W6-6  the detection grid is scale-invariant (MEDIUM)                       #
# =========================================================================== #
def test_w6_6_detection_grid_size_is_length_unit_invariant():
    """The detection density is per unit of the DIMENSIONLESS ``(hi-lo)*Ly^2``.
    PRE-FIX it was per unit of the raw ``qz^2`` (units 1/length^2), so ONE
    physical cell was scanned at wildly different densities depending on the
    caller's length unit: the reference 1-um cell asked for 3944 points in um,
    400 (the ``n_scan`` floor -> UNDER-resolved) in nm, and 3.94e9 points -- a
    31.5 GB ``linspace``, i.e. a hang / MemoryError -- in mm."""
    f = eme_2d_vector._detect_grid_size
    ref = f(-739.5, 493.0, 1.0, 400)                # the reference cell in um
    for s in (1e-3, 1e-1, 1.0, 10.0, 1e3):          # the same cell, other units
        assert f(-739.5 / s ** 2, 493.0 / s ** 2, 1.0 * s, 400) == ref
    assert ref > 400                                # the PPU grid, not the floor
    assert f(0.0, 1.0, 1.0, 4096) == 4096           # n_scan stays a floor


def test_w6_6_scaled_cell_keeps_full_recall():
    """The behavioural consequence: the SAME physical cell expressed in a 10x
    larger length unit must recover the same modes.  PRE-FIX the scaled window
    (``hi-lo`` shrunk 100x) collapsed to a 9-point detection grid."""
    Nx, k0, Ly, Lx = 8, 8.0, 1.0, 1.0
    strips = _reference_cell(Nx)
    lo, hi = 150.0, 250.0
    base = eme_2d_vector.layer_vector_modes(
        strips, Lx, Nx, Ly, k0, (lo, hi), ky0=PI, n_scan=3)
    s = 10.0                                        # lengths x10 -> qz^2 / 100
    scaled = eme_2d_vector.layer_vector_modes(
        [(e, h * s) for e, h in strips], Lx * s, Nx, Ly * s, k0 / s,
        (lo / s ** 2, hi / s ** 2), ky0=PI / s, n_scan=3)
    assert len(base) >= 1
    assert len(scaled) >= 1, ("the scaled cell found NO modes -- the detection "
                              "grid collapsed with the length unit")
    # the physics is exactly scale-invariant, so every base mode must reappear
    for q in base:
        assert np.min(np.abs(scaled * s ** 2 - q)) < 3e-3 * abs(q)


def test_w6_6_pathological_window_warns_and_clamps(monkeypatch):
    """A window that would demand more than the cap warns loudly instead of
    silently trying to allocate it (the real cap is 200_000 points; it is lowered
    here so the pin costs nothing)."""
    monkeypatch.setattr(eme_2d_vector, "_DETECT_MAX", 40)
    strips = _reference_cell(4)
    with pytest.warns(UserWarning, match="CLAMPED"):
        eme_2d_vector.layer_vector_modes(strips, 1.0, 4, 1.0, 8.0,
                                        (200.0, 250.0), ky0=PI, n_scan=3)


# =========================================================================== #
#  W6-7  sigma is not silently inert (MEDIUM)                                 #
# =========================================================================== #
def test_w6_7_sigma_without_k_rejected():
    """PRE-FIX ``sigma`` was ignored on the dense path in BOTH oracles (measured
    bit-identical results with sigma=1e9), so a caller asking for a few modes
    near a shift silently got the full dense spectrum."""
    eps = np.full((6, 6), 2.0)
    with pytest.raises(ValueError, match="sigma only applies"):
        eme_2d.ref_2d_modes(eps, 1.0, 1.0, 6, 6, 4.0, sigma=1e9)
    with pytest.raises(ValueError, match="sigma only applies"):
        eme_2d_vector.ref_2d_modes_vector(eps, 1.0, 1.0, 6, 6, 4.0, sigma=1e9)
    # with k it is honoured (and the dense path without sigma still works)
    assert len(eme_2d.ref_2d_modes(eps, 1.0, 1.0, 6, 6, 4.0, k=4)) == 4
    assert len(eme_2d.ref_2d_modes(eps, 1.0, 1.0, 6, 6, 4.0)) == 36


# =========================================================================== #
#  W6-8  the sparse scalar oracle is RNG-independent (LOW)                    #
# =========================================================================== #
def test_w6_8_sparse_scalar_oracle_is_deterministic():
    """PRE-FIX ARPACK drew its start vector from NumPy's GLOBAL RNG, so this
    oracle's output depended on unrelated seeding elsewhere in the process
    (measured 1.7e-13 drift).  Both siblings (``ref_2d_modes_vector``,
    ``_fd_eig_dist``) already passed a fixed ``v0``."""
    eps = np.full((8, 8), 2.0)
    eps[2:5, 2:5] = 6.0
    np.random.seed(1)
    a = eme_2d.ref_2d_modes(eps, 1.0, 1.0, 8, 8, 4.0, k=4)
    np.random.seed(99999)
    b = eme_2d.ref_2d_modes(eps, 1.0, 1.0, 8, 8, 4.0, k=4)
    assert np.array_equal(a, b)


# =========================================================================== #
#  W6-9  lossy Im(qz^2)-discard warning parity (LOW)                          #
# =========================================================================== #
def test_w6_9_numpy_vector_oracle_warns_on_lossy_discard():
    """The scalar sibling warned; this one discarded silently."""
    eps = np.full((6, 6), 2.0 + 0.4j)
    with pytest.warns(UserWarning, match="DISCARDED"):
        eme_2d_vector.ref_2d_modes_vector(eps, 1.0, 1.0, 6, 6, 4.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")            # return_complex -> no warning
        eme_2d_vector.ref_2d_modes_vector(eps, 1.0, 1.0, 6, 6, 4.0,
                                          return_complex=True)


def test_w6_9_jax_scalar_twin_warns_on_lossy_discard():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    eps = np.full((5, 5), 2.0 + 0.4j)
    with pytest.warns(UserWarning, match="DISCARDED"):
        eme_2d.ref_2d_modes(jnp.asarray(eps), 1.0, 1.0, 5, 5, 4.0)


# =========================================================================== #
#  W6-10 .. W6-14  the LOW cluster (silent empty result / opaque messages)     #
# =========================================================================== #
@pytest.mark.parametrize("n_scan", [0, 1, 2])
def test_w6_10_tiny_n_scan_rejected(n_scan):
    """PRE-FIX ``n_scan <= 1`` silently returned an EMPTY mode array, which reads
    as 'no modes in this window'."""
    strips = _reference_cell(8)
    with pytest.raises(ValueError, match="n_scan"):
        eme_2d.layer_modes(strips, 1.0, 8, 1.0, 8.0, (150.0, 250.0), ky0=PI,
                           n_scan=n_scan)
    with pytest.raises(ValueError, match="n_scan"):
        eme_2d_vector.layer_vector_modes(strips, 1.0, 8, 1.0, 8.0, (150.0, 250.0),
                                        ky0=PI, n_scan=n_scan)


def test_w6_11_eps_xy_to_strips_tensor_grid_message():
    """PRE-FIX: ``ValueError: too many values to unpack (expected 2, got 4)``."""
    t = np.tile(np.eye(3) * 2.0, (8, 8, 1, 1))
    with pytest.raises(NotImplementedError, match="scalar isotropic"):
        eme_2d_vector.eps_xy_to_strips(t, 8, 4, 1.0, 1.0)
    # the scalar grid and the callable forms still work
    assert len(eme_2d_vector.eps_xy_to_strips(np.full((8, 8), 2.0), 8, 4,
                                              1.0, 1.0)) == 4
    assert len(eme_2d_vector.eps_xy_to_strips(lambda x, y: 2.0, 8, 4,
                                              1.0, 1.0)) == 4


def test_w6_12_sigma_min_invpow_zero_iters():
    """PRE-FIX: ``UnboundLocalError: cannot access local variable 'cur'``."""
    strips = _reference_cell(8)
    wvk = eme_2d_vector._strip_modes_at(strips, 1.0, 8, 8.0, 0.0, 200.0)
    Geq, _ = eme_2d_vector._global_block_G(wvk, np.exp(1j * PI))
    assert eme_2d_vector._sigma_min_invpow(Geq, iters=0) == 0.0
    assert eme_2d_vector._sigma_min_invpow(Geq) > 0.0


def test_w6_13_mode_match_zero_mode_column_message():
    """PRE-FIX: an opaque ``LinAlgError: SVD did not converge in Linear Least
    Squares`` from deep inside the solve."""
    orders = eme_diffraction.plane_wave_orders(0, 0)
    with pytest.raises(ValueError, match="zero-norm or non-finite"):
        eme_diffraction.mode_match(
            np.array([10.0]), np.zeros((16, 1), dtype=complex), orders,
            kx0=0.0, ky0=0.0, k0=4.0, eps_sup=1.0, eps_sub=1.0, depth=0.3,
            Lx=1.0, Ly=1.0, Nx=4, Ny=4)


def test_w6_14_jax_traced_bloch_phase_message():
    """PRE-FIX a traced ``kx0`` (which the dispatch's ``is_jax_array(kx0)`` test
    implies is supported) died with a bare ``ConcretizationTypeError`` naming only
    "the `float` function"."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    eps = np.full((4, 4), 2.0)

    def f(kx):
        return eme_2d.ref_2d_modes(jnp.asarray(eps), 1.0, 1.0, 4, 4, 4.0,
                                   kx0=kx)[0]

    with pytest.raises(NotImplementedError, match="CONCRETE"):
        jax.grad(f)(jnp.asarray(0.3))


# =========================================================================== #
#  W6-15  the vector path needs its scalar sibling's band-edge guard (MEDIUM)  #
# =========================================================================== #
def test_w6_15_vector_layer_finder_survives_a_band_edge_sample():
    """A ``qz2_range`` starting at 0 is perfectly natural, and for the reference
    cell ``qz^2 = 0`` puts the uniform ``eps = 2`` strip EXACTLY on a band edge
    (measured ``min|ky| = 0.000e+00`` at ``Nx = 8``, ``k0 = 8``).  The H-part is
    recovered as ``(C U)/(i ky)``, so that mode came back as NaN and the whole
    call died several frames later with
    ``ValueError: array must not contain infs or NaNs``.  The scalar sibling
    already skipped its analogous band-edge sample (audit P3-18)."""
    Nx = 8
    strips = _reference_cell(Nx)
    # the scalar sibling: already guarded (this passes pre- and post-fix)
    assert isinstance(eme_2d.layer_modes(strips, 1.0, Nx, 1.0, 8.0, (0.0, 1e5),
                                         ky0=PI, n_scan=50), np.ndarray)
    # the vector finder must now survive the same window
    q = eme_2d_vector.layer_vector_modes(strips, 1.0, Nx, 1.0, 8.0, (0.0, 256.0),
                                        ky0=PI, n_scan=50)
    assert isinstance(q, np.ndarray)
    assert np.all(np.isfinite(q))
    # and the single-point evaluator raises a NAMED, catchable LinAlgError rather
    # than producing NaN four frames downstream
    with pytest.raises(np.linalg.LinAlgError, match="band edge"):
        eme_2d_vector.strip_vector_modes(np.full(Nx, 2.0), 1.0, Nx, 8.0, 0.0, 0.0)
    with pytest.raises(np.linalg.LinAlgError, match="band edge"):
        eme_2d_vector.dispersion_vec(strips, 1.0, Nx, 8.0, 0.0, 0.0, PI, 1.0)


def test_w6_15_deeply_out_of_band_sample_is_named_not_a_nan():
    """The other non-evaluable sample: a ``qz^2`` so far outside the band that a
    hyper-evanescent strip mode's ``exp(+|ky| h)`` overflows to ``inf`` and the
    column equilibration makes it ``NaN`` (measured at ``qz^2 = 1e7``,
    ``max|ky| = 3.2e3``).  It used to raise the same opaque svdvals
    ``ValueError``."""
    strips = _reference_cell(8)
    with pytest.raises(np.linalg.LinAlgError, match="not finite"):
        eme_2d_vector.dispersion_vec(strips, 1.0, 8, 8.0, 0.0, 1e7, PI, 1.0)
    # inside the band it is unaffected
    assert eme_2d_vector.dispersion_vec(strips, 1.0, 8, 8.0, 0.0, 200.0,
                                        PI, 1.0) > 0.0


# =========================================================================== #
#  VERIFIED-CLEAN LOCKS (pass pre- AND post-fix -- they record what was        #
#  checked and cleared, with the measured cross-platform tolerances)           #
# =========================================================================== #
def test_clean_bloch_qep_residual_matches_derivation():
    """``M(qz^2) = -S12 t^2 + (I + S12 S21 - S22 S11) t - S21`` re-derived from
    the Bloch closure ``a' = t a, b' = t b`` on ``[b; a'] = S [a; b']``: a layer
    mode is exactly where ``M`` is singular, so the null vector of ``M`` must
    satisfy the closure to machine precision."""
    Nx, k0, Ly = 12, 8.0, 1.0
    strips = [(np.full(Nx, 2.0), 0.5), (np.full(Nx, 2.0), 0.5)]
    sm = [(eme_2d.strip_x_modes(e, 1.0, Nx, k0, 0.0), h) for e, h in strips]
    qz2 = 2.0 * k0 ** 2 - PI ** 2                     # exact mode at ky0 = pi
    S11, S12, S21, S22 = eme_2d.cell_smatrix(sm, qz2)
    t = np.exp(1j * PI * Ly)
    M = eme_2d._bloch_residual_M((S11, S12, S21, S22), t)
    a = np.linalg.svd(M)[2][-1].conj()
    I = np.eye(Nx, dtype=complex)
    assert np.linalg.norm(M @ a) < 1e-8
    # independent re-assembly of the same residual
    M2 = -S12 * t ** 2 + (I + S12 @ S21 - S22 @ S11) * t - S21
    assert np.allclose(M, M2, rtol=0, atol=1e-12)


def test_clean_jax_twins_match_numpy():
    """NumPy vs JAX parity for both eig-based oracles.  MEASURED: the scalar twin
    is bit-identical (0.0); the vector twin drifts up to 3.3e-12 absolute at
    kx0 != 0 (dense-eig ordering / BLAS)."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    Nx = Ny = 6
    eps = np.full((Nx, Ny), 2.0)
    eps[1:4, 1:4] = 6.0
    k0 = 2 * PI / 1.1
    for kx0, ky0 in ((0.0, 0.0), (0.37, 1.1)):
        a = eme_2d.ref_2d_modes(eps, 1.0, 1.0, Nx, Ny, k0, kx0=kx0, ky0=ky0)
        b = np.asarray(eme_2d.ref_2d_modes(jnp.asarray(eps), 1.0, 1.0, Nx, Ny,
                                          k0, kx0=kx0, ky0=ky0))
        assert np.max(np.abs(a - b)) < 1e-9
        c = eme_2d_vector.ref_2d_modes_vector(eps, 1.0, 1.0, Nx, Ny, k0,
                                             kx0=kx0, ky0=ky0)
        d = np.asarray(eme_2d_vector.ref_2d_modes_vector(
            jnp.asarray(eps), 1.0, 1.0, Nx, Ny, k0, kx0=kx0, ky0=ky0))
        assert np.max(np.abs(c - d)) < 1e-9


def test_clean_jax_frozen_operator_cache_key_is_complete():
    """Every geometry/Bloch argument the frozen operators depend on is in the
    cache key, and the LRU bound holds."""
    from lumenairy.elements.eme import _jax_modes
    pytest.importorskip("jax")
    _jax_modes._clear_frozen_cache()
    L1 = _jax_modes._frozen_helmholtz_L(4, 4, 1.0, 1.0, 0.0, 0.0)
    for args in ((4, 5, 1.0, 1.0, 0.0, 0.0), (4, 4, 2.0, 1.0, 0.0, 0.0),
                 (4, 4, 1.0, 2.0, 0.0, 0.0), (4, 4, 1.0, 1.0, 0.5, 0.0),
                 (4, 4, 1.0, 1.0, 0.0, 0.5)):
        L2 = _jax_modes._frozen_helmholtz_L(*args)
        assert L2.shape != L1.shape or not np.array_equal(L1, L2)
    Y1 = _jax_modes._frozen_yee_dense(4, 4, 1.0, 1.0, 0.0, 0.0)
    Y2 = _jax_modes._frozen_yee_dense(4, 4, 1.0, 1.0, 0.0, 0.7)
    assert not np.array_equal(Y1[2], Y2[2])
    for i in range(20):
        _jax_modes._frozen_helmholtz_L(4, 4, 1.0 + i, 1.0, 0.0, 0.0)
    assert len(_jax_modes._FROZEN_CACHE) <= _jax_modes._FROZEN_CACHE_SIZE


def test_clean_beyn_refiner_reaches_analytic_lossy_mode():
    """The SEEDED Beyn refiner recovers the analytic lossy uniform-cell
    ``qz^2 = eps k0^2 - ky0^2`` from a seed, to its documented x-FD-floored
    accuracy (MEASURED 2.3e-3 .. 1.0e-1 absolute -- it is a refiner, not a
    pole-sharp solver, exactly as its docstring says)."""
    k0 = 8.0
    for epsl in (2.0 + 0.02j, 2.0 + 0.1j):
        e = np.full(8, epsl)
        true = epsl * k0 ** 2 - PI ** 2
        m, s0 = eme_2d_vector.beyn_refine_complex([(e, 0.5), (e, 0.5)], 1.0, 8,
                                                  k0, PI, 1.0, true)
        assert m is not None and s0 > 1.0
        assert abs(m - true) < 0.2
        assert abs(m.imag) > 0.5 * abs(true.imag)     # genuinely complex


def test_clean_eyz_layer_finder_stays_gated():
    """The documented ``eyz``/``ezy`` LAYER gate still fires (the ``[W; -V]``
    backward mode is rigorously wrong there) -- a silent wrong answer would be
    worse than the NotImplementedError."""
    t = np.tile(np.eye(3) * 2.0, (6, 1, 1))
    t[:, 1, 2] = 0.3
    t[:, 2, 1] = 0.3
    with pytest.raises(NotImplementedError, match="eyz"):
        eme_2d_vector.layer_vector_modes([(t, 0.5), (t, 0.5)], 1.0, 6, 1.0, 8.0,
                                        (100.0, 130.0), ky0=PI, n_scan=10)


def test_clean_structured_diffraction_still_warns_and_does_not_converge():
    """The module's documented NEGATIVE RESULT is a basis problem, not the
    conditioning problem W6-2 fixed: with the stable reformulation in place a
    STRUCTURED layer still fails to converge, and still warns."""
    Lx = Ly = 2.0
    Nx, k0 = 16, 5.0
    xg = (np.arange(Nx) + 0.5) / Nx * Lx
    block = np.where((xg >= 0.5) & (xg < 1.3), 6.0, 1.0)
    strips = [(np.full(Nx, 1.0), 0.7), (block, 0.6), (np.full(Nx, 1.0), 0.7)]
    eps_xy = eme_2d_vector.strips_to_eps_xy(strips, Lx, Nx, Ly, 48)
    with pytest.warns(UserWarning, match="STRUCTURED"):
        eme_diffraction.diffraction_fd(eps_xy, Lx, Ly, Nx, 48, k0, 1.0, 2.25,
                                       0.4, 1, 1, kx0=0.0, ky0=2.0)
    energies, t00 = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for M in (1, 2, 3):
            res = eme_diffraction.diffraction_fd(eps_xy, Lx, Ly, Nx, 48, k0,
                                                1.0, 2.25, 0.4, M, M,
                                                kx0=0.0, ky0=2.0)
            energies.append(res["energy"])
            t00.append(res["T"][res["orders"].index((0, 0))])
    assert max(abs(e - 1.0) for e in energies) > 0.02
    assert max(t00) - min(t00) > 0.02
