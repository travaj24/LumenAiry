"""v5.6.1 RCWA even-parity-sector symmetry fast path (``symmetry=True``).

A centro-symmetric cell at normal incidence excites only even-parity modes, so
``rcwa_efficiency_2d`` can run the whole S-matrix recursion in the ``(N+1)``-d
even subspace instead of the full ``2N`` -- a correctness-preserving speed-up.

The feature is validated against the full ``2N`` solve (the established
reference): efficiencies must match to ~1e-12 (a different but physically
equivalent basis, hence not bit-for-bit), energy must be conserved, and every
non-applicable case (oblique incidence, a non-centro-symmetric cell, a uniform
layer) must transparently FALL BACK to -- and stay bit-identical with -- the
full solve.  An off-centre feature is covered specifically: it is symmetric
about its own centre, not sample 0, so the diagonal recentering gauge is
exercised.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.rcwa import (
    _even_basis_desc,
    _even_fold,
    _order_flip_perm,
    rcwa_efficiency_2d,
)

WL = 0.6e-6
PX = PY = 0.5e-6
DEPTH = 0.2e-6
EPS_HI = -3.0 + 4.0j          # lossy metal-like (PUBLIC Im > 0)
EPS_LO = 2.1


def _pillar(S, cx, cy, hw=0.22):
    """Square pillar of half-width ``hw`` centred at ``(cx, cy)`` (wrap-aware),
    so the cell is centro-symmetric about an arbitrary point."""
    x = (np.arange(S) + 0.5) / S - 0.5
    X, Y = np.meshgrid(x, x, indexing="ij")
    eps = np.full((S, S), EPS_LO, dtype=complex)
    dx = np.abs((X - cx + 0.5) % 1.0 - 0.5)
    dy = np.abs((Y - cy + 0.5) % 1.0 - 0.5)
    eps[(dx < hw) & (dy < hw)] = EPS_HI
    return eps


def _solve(cell, *, symmetry, pol="tm", formu="li", No=6, **kw):
    return rcwa_efficiency_2d(
        PX, PY, cell, 1.5, 1.0, DEPTH, WL, polarization=pol,
        n_orders_x=No, n_orders_y=No, formulation=formu, symmetry=symmetry,
        **kw)


# ---------------------------------------------------------------------------
# correctness: even-sector == full 2N solve (physically, ~1e-12)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pol", ["te", "tm"])
@pytest.mark.parametrize("cx,cy", [(0.0, 0.0), (0.13, -0.21), (-0.5, -0.5)])
def test_even_sector_matches_full_solve(pol, cx, cy):
    cell = _pillar(64, cx, cy)
    o0, R0, T0 = _solve(cell, symmetry=False, formu="laurent", pol=pol)
    o1, R1, T1 = _solve(cell, symmetry=True, formu="laurent", pol=pol)
    # physically identical, but a different (even-adapted, recentred) basis,
    # so ~1e-12 rather than bit-for-bit
    assert np.allclose(R0, R1, atol=1e-11, rtol=0)
    assert np.allclose(T0, T1, atol=1e-11, rtol=0)
    # and it is genuinely the fast path, not a silent fallback
    assert not (np.array_equal(R0, R1) and np.array_equal(T0, T1))


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_li_falls_back_bit_identical(pol):
    """Since v5.14.1 'li' is the Li-1997 SEQUENTIAL rule routed through the
    per-component tensor eigensolver, which _symmetric_solve_rt (hard-wired
    to the scalar core) cannot represent -- so symmetry=True must
    transparently FALL BACK bit-for-bit for 'li' (extending the even-sector
    fold to the per-component tensor operators is a roadmap perf item)."""
    cell = _pillar(64, 0.13, -0.21)
    o0, R0, T0 = _solve(cell, symmetry=False, formu="li", pol=pol)
    o1, R1, T1 = _solve(cell, symmetry=True, formu="li", pol=pol)
    assert np.array_equal(R0, R1) and np.array_equal(T0, T1)


def test_even_sector_conserves_energy_lossless():
    # lossless dielectric pillar (real index) -> sum R + T == 1
    cell = np.where(_pillar(64, 0.05, -0.17, hw=0.2) == EPS_HI,
                    6.0 + 0j, 1.0 + 0j)
    o, R, T = rcwa_efficiency_2d(PX, PY, cell, 1.0, 1.0, DEPTH, WL,
                                 polarization="tm", n_orders_x=6, n_orders_y=6,
                                 formulation="li", symmetry=True)
    assert abs(float(R.sum() + T.sum()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# transparent fallback: symmetry=True must equal the full solve bit-for-bit
# whenever the even-sector precondition does not hold
# ---------------------------------------------------------------------------

def test_oblique_incidence_falls_back_bit_identical():
    cell = _pillar(64, 0.0, 0.0)
    o0, R0, T0 = _solve(cell, symmetry=False, theta=0.25)
    o1, R1, T1 = _solve(cell, symmetry=True, theta=0.25)
    assert np.array_equal(R0, R1) and np.array_equal(T0, T1)


def test_non_centrosymmetric_falls_back_bit_identical():
    rng = np.random.default_rng(4)
    cell = (rng.uniform(1.0, 6.0, (64, 64))
            + 1j * rng.uniform(0.0, 2.0, (64, 64)))     # no symmetry centre
    o0, R0, T0 = _solve(cell, symmetry=False)
    o1, R1, T1 = _solve(cell, symmetry=True)
    assert np.array_equal(R0, R1) and np.array_equal(T0, T1)


def test_uniform_layer_falls_back_bit_identical():
    cell = np.full((64, 64), 2.25 + 0j)                  # laterally uniform
    o0, R0, T0 = _solve(cell, symmetry=False)
    o1, R1, T1 = _solve(cell, symmetry=True)
    assert np.array_equal(R0, R1) and np.array_equal(T0, T1)


def test_default_is_full_solve_bit_identical():
    cell = _pillar(64, 0.0, 0.0)
    o0, R0, T0 = _solve(cell, symmetry=False)
    o1, R1, T1 = rcwa_efficiency_2d(PX, PY, cell, 1.5, 1.0, DEPTH, WL,
                                    polarization="tm", n_orders_x=6,
                                    n_orders_y=6, formulation="li")
    assert np.array_equal(R0, R1) and np.array_equal(T0, T1)


# ---------------------------------------------------------------------------
# fold primitive: B^H A B reproduces the dense even compression
# ---------------------------------------------------------------------------

def test_even_fold_equals_dense_compression():
    # build a flip-invariant matrix and check _even_fold == B^H A B densely
    M = 3
    m = np.repeat(np.arange(-M, M + 1), 2 * M + 1)
    n = np.tile(np.arange(-M, M + 1), 2 * M + 1)
    N = m.size
    lut = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(m, n))}
    flip = np.array([lut[(-int(a), -int(b))] for a, b in zip(m, n)])
    rng = np.random.default_rng(0)
    # an order-flip-invariant 2N x 2N operator
    A = rng.standard_normal((2 * N, 2 * N)) + 1j * rng.standard_normal((2 * N, 2 * N))
    g = np.concatenate([flip, flip + N])
    A = 0.5 * (A + A[np.ix_(g, g)])               # symmetrize so J A J == A
    desc = _even_basis_desc(flip)
    Ae = _even_fold(A, desc, np)
    # dense even basis B (2N x Ne)
    i0, i1, c0, c1, n2 = desc
    ne = i0.size
    B = np.zeros((n2, ne), dtype=complex)
    B[i0, np.arange(ne)] += c0
    B[i1, np.arange(ne)] += c1
    assert np.allclose(Ae, B.conj().T @ A @ B, atol=1e-12)


def test_order_flip_perm_none_at_oblique():
    # at oblique incidence the order set is not closed under (kx,ky)->(-kx,-ky)
    kx = np.diag((np.arange(-3, 4) + 0.3).astype(complex))   # constant offset
    ky = np.diag(np.zeros(7, dtype=complex))
    assert _order_flip_perm(kx, ky) is None
    # at normal incidence it is a valid involution
    kx0 = np.diag(np.arange(-3, 4).astype(complex))
    perm = _order_flip_perm(kx0, ky)
    assert perm is not None and np.array_equal(perm[perm], np.arange(7))
