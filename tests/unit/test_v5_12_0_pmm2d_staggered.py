"""v5.12.0 -- canonical no-floor 2-D crossed-grating PMM (Granet 2023 staggered
modified-Legendre basis), ``pmm_efficiency_2d_staggered``.

The defining property vs the FMM-floored hybrid ``pmm_efficiency_2d`` is that the
energy balance is ``n_orders``-INDEPENDENT (no Fourier floor) and tracks only the
modal degree -- exercised directly below.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.pmm.twod_staggered import pmm_efficiency_2d_staggered

_G = dict(period_x=0.8e-6, period_y=0.8e-6, depth=0.3e-6, wavelength=0.633e-6)


def _T00(o, T):
    return float(T[(o[:, 0] == 0) & (o[:, 1] == 0)][0])


def _R00(o, R):
    return float(R[(o[:, 0] == 0) & (o[:, 1] == 0)][0])


# quarter-cell square pillar (Nx=Ny=2, pillar in segment (0,0))
_PILLAR = np.array([[2.25, 1.0], [1.0, 1.0]], dtype=complex)


# ===================================================================
# Vacuum exactness
# ===================================================================
@pytest.mark.parametrize("n_orders", [3, 5, 7])
def test_vacuum_exact(n_orders):
    o, R, T = pmm_efficiency_2d_staggered(
        eps_cell=np.ones((2, 2)), n_substrate=1.0, n_superstrate=1.0,
        degree=6, n_orders=n_orders, **_G)
    assert abs(_T00(o, T) - 1.0) < 1e-10
    assert R.sum() < 1e-10
    assert abs(R.sum() + T.sum() - 1.0) < 1e-10


# ===================================================================
# THE no-floor signatures
# ===================================================================
def test_no_fourier_floor():
    """Energy balance is ROUND-OFF-IDENTICAL across n_orders -- the
    no-Fourier-floor property that distinguishes this from the FMM-floored
    hybrid.

    2026-08-15 (docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md, D6): this said
    "BYTE-IDENTICAL" and pinned ``max(vals) - min(vals) < 1e-13``.  The premise
    was false.  The four calls build DIFFERENT-SIZED Fourier order sets --
    MEASURED 49 / 121 / 225 / 361 orders, i.e. 7x7, 11x11, 15x15, 19x19 -- so
    the four energy sums are different-LENGTH floating-point reductions and
    their disagreement is pure reassociation, which moves with BLAS build, SIMD
    width and reduction blocking.  MEASURED spread 2.442e-15 (Win
    py3.14/np2.4.4) / 6.661e-15 (WSL py3.12/np2.5.1): a 2.7x cross-build spread
    on a hand-picked absolute bar, i.e. exactly the per-build fact this suite
    must not assert.

    THE PROPERTY, DERIVED IN THIS BUILD.  "No Fourier floor" means n_orders
    contributes nothing to the error -- the whole error is MODAL.  So bar the
    n_orders spread against the modal error the SAME run measures,
    ``modal_err = |vals[0] - 1|``: three decades below it is "n_orders
    contributes nothing".  MEASURED modal_err = 1.654013e-09 (Win) /
    1.653948e-09 (WSL), spread / modal_err = 1.48e-06 / 4.03e-06 -- 677x / 248x
    of headroom on the 1e-3 bar.

    TWO-SIDED.  Below: the FMM-floored hybrid this test exists to distinguish
    from moves its balance by an O(1) fraction of its total error when n_orders
    changes, which a 1e-3-of-modal-error bar rejects outright -- and because
    the bar is RELATIVE it tightens automatically as the modal error improves,
    where a fixed 1e-13 would go slack.  Above: it is floored at the
    reassociation limit of the LONGEST reduction actually performed
    (``n_max * eps``, 361 terms at ~1 ULP = 8.0e-14), so a future modal
    improvement can never make the test demand better arithmetic than float64
    can deliver.  The floor is 20x below the current bar and inactive today."""
    vals = []
    n_seen = []
    for no in (3, 5, 7, 9):
        o, R, T = pmm_efficiency_2d_staggered(
            eps_cell=_PILLAR, n_substrate=1.0, n_superstrate=1.0,
            degree=8, n_orders=no, **_G)
        vals.append(R.sum() + T.sum())
        n_seen.append(int(o.shape[0]))
    # the four solves really are different truncations -- otherwise the whole
    # claim would be vacuous rather than merely mis-stated
    assert n_seen == [49, 121, 225, 361], n_seen
    modal_err = abs(vals[0] - 1.0)
    bar = max(1e-3 * modal_err, max(n_seen) * np.finfo(float).eps)
    assert max(vals) - min(vals) < bar            # n_orders-independent
    assert modal_err < 1e-7                       # and small


def test_no_floor_in_degree():
    """Pillar energy error decreases monotonically with degree to round-off
    (no plateau).

    2026-08-15 (docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md, D6): the second
    bar was the absolute ``errs[2] < 1e-11``.  MEASURED errs[2] = 1.168e-13
    (Win py3.14/np2.4.4) / 2.774e-13 (WSL py3.12/np2.5.1) -- 86x / 36x with a
    2.4x cross-build spread, because by degree 10 the energy defect IS
    round-off: it is a cancellation residue whose magnitude is a property of
    the build's reduction order, not of the method.

    THE CLAIM is "no plateau" -- the error keeps FALLING with degree -- so
    state the last step against the previous step measured in the SAME run.
    MEASURED errs = 1.698721e-06 / 1.654015e-09 / 1.167955e-13 (Win) and
    1.698721e-06 / 1.653954e-09 / 2.774447e-13 (WSL); the degree 8 -> 10 step
    is a factor 7.06e-05 (Win) / 1.68e-04 (WSL), so the 1e-2 bar carries 142x /
    60x of headroom.  Two-sided: two decades per two degrees is still an
    unambiguous spectral descent, and a method that HAD hit a floor would
    return errs[2] ~ errs[1] and miss the bar by four decades.  The bar also
    cannot go slack as the solver improves -- it is relative to this build's
    own degree-8 error rather than to a number chosen in 2026."""
    errs = []
    for M in (6, 8, 10):
        o, R, T = pmm_efficiency_2d_staggered(
            eps_cell=_PILLAR, n_substrate=1.0, n_superstrate=1.0,
            degree=M, n_orders=5, **_G)
        errs.append(abs(R.sum() + T.sum() - 1.0))
    assert errs[1] < errs[0] and errs[2] < errs[1]
    assert errs[2] < 1e-2 * errs[1], errs


# ===================================================================
# Uniform-slab Fabry-Perot vs analytic (independent oracle)
# ===================================================================
def test_uniform_slab_fabry_perot():
    """A uniform layer must reproduce the analytic single-layer (characteristic
    matrix) reflectance/transmittance, with all non-zero orders vanishing."""
    n_sup, n_sub, n_f = 1.0, 1.5, 2.0
    o, R, T = pmm_efficiency_2d_staggered(
        eps_cell=np.full((2, 2), n_f ** 2), n_substrate=n_sub,
        n_superstrate=n_sup, degree=6, n_orders=4, **_G)
    # analytic normal-incidence slab (admittances ~ index):
    delta = 2.0 * np.pi * n_f * _G["depth"] / _G["wavelength"]
    B = np.cos(delta) + 1j * np.sin(delta) / n_f * n_sub
    Cc = 1j * n_f * np.sin(delta) + np.cos(delta) * n_sub
    r = (n_sup * B - Cc) / (n_sup * B + Cc)
    R_an = abs(r) ** 2
    T_an = 1.0 - R_an
    assert abs(_R00(o, R) - R_an) < 1e-9
    assert abs(_T00(o, T) - T_an) < 1e-9
    # non-zero orders carry no power for a uniform layer
    nz = ~((o[:, 0] == 0) & (o[:, 1] == 0))
    assert R[nz].sum() + T[nz].sum() < 1e-9


# ===================================================================
# Position invariance (lattice translation preserves per-order efficiency)
# ===================================================================
def test_position_invariance():
    """The pillar translated by half a period gives identical per-order
    efficiencies (a correctness check the FMM staircase cannot match exactly)."""
    base = dict(n_substrate=1.0, n_superstrate=1.0, degree=8, n_orders=5, **_G)
    o1, R1, T1 = pmm_efficiency_2d_staggered(eps_cell=_PILLAR, **base)
    pillar2 = np.array([[1.0, 1.0], [1.0, 2.25]], dtype=complex)  # shifted (1,1)
    o2, R2, T2 = pmm_efficiency_2d_staggered(eps_cell=pillar2, **base)
    assert np.array_equal(o1, o2)
    assert np.max(np.abs(R1 - R2)) < 1e-9
    assert np.max(np.abs(T1 - T2)) < 1e-9


# ===================================================================
# Physical absorption (R+T=1 in the lossless case is physics, not identity)
# ===================================================================
def test_lossy_pillar_absorbs():
    eps = np.array([[2.25 + 0.5j, 1.0], [1.0, 1.0]], dtype=complex)
    o, R, T = pmm_efficiency_2d_staggered(
        eps_cell=eps, n_substrate=1.0, n_superstrate=1.0, degree=8,
        n_orders=5, **_G)
    A = 1.0 - R.sum() - T.sum()
    assert A > 1e-3                       # genuine absorption
    assert A < 1.0


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_polarization_runs_and_conserves(pol):
    o, R, T = pmm_efficiency_2d_staggered(
        eps_cell=_PILLAR, n_substrate=1.0, n_superstrate=1.0, degree=8,
        n_orders=5, polarization=pol, **_G)
    assert abs(R.sum() + T.sum() - 1.0) < 1e-7


# ===================================================================
# Cross-check vs the Fourier method (brackets the same value)
# ===================================================================
def test_matches_rcwa():
    """PMM (converged, position-invariant) and a high-order RCWA-li agree on
    the dominant orders within RCWA's own truncation error on this hard
    high-contrast large-period case."""
    rcwa = pytest.importorskip("lumenairy.elements.rcwa")
    o, R, T = pmm_efficiency_2d_staggered(
        eps_cell=_PILLAR, n_substrate=1.0, n_superstrate=1.0, degree=12,
        n_orders=7, **_G)
    # rasterize the same quarter-cell pillar for RCWA
    n_ord = 15
    S = 8 * n_ord + 1
    e = np.ones((S, S), dtype=complex)
    xc = (np.arange(S) + 0.5) / S
    XX, YY = np.meshgrid(xc, xc, indexing="ij")
    e[(XX < 0.5) & (YY < 0.5)] = 2.25
    orc, Rr, Tr = rcwa.rcwa_efficiency_2d(
        _G["period_x"], _G["period_y"], e, 1.0, 1.0, _G["depth"],
        _G["wavelength"], polarization="te", n_orders_x=n_ord, n_orders_y=n_ord,
        formulation="li", stabilize=True)
    # 6e-3 (was 3e-3): the corrected sequential-rule 'li' oracle (audit F1,
    # 2026-06-10) moved T00 by ~1e-3 on this hard case; measured 4.0e-3.
    assert abs(_T00(o, T) - _T00(orc, Tr)) < 6e-3
    assert abs(R.sum() - Rr.sum()) < 6e-3


# ===================================================================
# PP2 perf invariant: the shared eps-free geometric eig reconstructs a
# homogeneous half-space identically to a fresh per-region eig
# ===================================================================
def test_pp2_shared_homogeneous_eig_matches_fresh_eig():
    """The half-space speed-up (one eps-free geometric eig serves BOTH half-spaces
    via ``g2 = g2_geo + eps``, same eigenvectors -- 3 region eigs -> 2) must yield
    GENUINE modes of each half-space operator.  Basis-invariant pin: the shifted
    modes (i) carry the SAME spectrum as a fresh per-region eig (as a set) and
    (ii) satisfy the region-b generalized eigen-relation ``L_b W = g2 G W`` to
    round-off.  (The interface S-matrix itself is NOT a valid check here -- it is
    expressed in region-b's mode-amplitude basis, whose ordering differs between
    the two eig routes; that ordering cancels only through the full solve, which
    ``test_uniform_slab_fabry_perot`` pins against an analytic oracle.)"""
    from lumenairy.elements.pmm.twod_staggered import (
        Granet2DTransverseE,
        _homog_geom_cache,
        _homog_region_modes,
        _region_modes,
    )

    px = py = 0.6e-6
    k0 = 2.0 * np.pi / 0.633e-6
    Nx = Ny = 2
    M = 4
    eps_ref, eps_b = 1.0 + 0j, 2.25 + 0j          # b != ref -> exercises the shift

    def _homog(eps):
        return Granet2DTransverseE(px, py, Nx, Ny, M, np.full((Nx, Ny), eps),
                                   alpha0x=0.0, alpha0y=0.0, k0=k0)

    sol_b = _homog(eps_b)
    L_b, G_b = sol_b.Lmat, -sol_b.Rmat

    # Fresh per-region eig at eps_b (the OLD 3-eig path) ...
    _Wbr, _Vbr, _l, g2_ref = _region_modes(sol_b)
    # ... vs the shared geometric eig built from a DIFFERENT eps_ref, shifted to eps_b
    geom = _homog_geom_cache(_homog(eps_ref))
    W0, g2_geo = geom[0], geom[1]
    g2_opt = g2_geo + eps_b

    # (i) same spectrum as a set
    s_ref = np.sort_complex(np.round(np.asarray(g2_ref), 8))
    s_opt = np.sort_complex(np.round(np.asarray(g2_opt), 8))
    assert np.max(np.abs(s_ref - s_opt)) < 1e-7

    # (ii) the shifted modes are genuine generalized eigenvectors of (L_b, G_b)
    resid = L_b @ W0 - (G_b @ W0) * g2_opt[None, :]
    scale = np.linalg.norm(L_b @ W0, axis=0) + 1.0
    assert np.max(np.linalg.norm(resid, axis=0) / scale) < 1e-9


# ===================================================================
# Input validation
# ===================================================================
def test_validation():
    base = dict(n_substrate=1.0, n_superstrate=1.0, **_G)
    with pytest.raises(ValueError):
        pmm_efficiency_2d_staggered(eps_cell=_PILLAR, polarization="xx", **base)
    with pytest.raises(ValueError):
        pmm_efficiency_2d_staggered(eps_cell=_PILLAR, degree=1, **base)
    with pytest.raises(ValueError):
        pmm_efficiency_2d_staggered(eps_cell=np.ones(3), **base)  # not 2-D
