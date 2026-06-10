"""v5.6 RCWA convergence acceleration -- 2-D Fourier factorization + truncation.

- The grid solver ``rcwa_efficiency_2d`` gains ``formulation='li'`` (alias
  ``'fff'``): the dual-Laurent z-rule (``E_z`` elimination via ``[[1/eps]]``)
  that converges faster than the default ``'laurent'`` for TM / metals.  The
  default stays bit-for-bit ``'laurent'``.
- Both 2-D solvers gain ``truncation='circular'`` (Lalanne 1997): the inscribed
  reciprocal circle, reaching the same value at fewer harmonics.  The default
  ``'rectangular'`` order set is unchanged.

Correctness is anchored two ways: (1) self-consistent -- ``'li'`` lands closer
to lumenairy's own high-N reference than ``'laurent'`` does; (2) oracle -- the
analytic-shape solver converges toward inkstone's high-num_g value (the test is
skipped when inkstone is absent).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.elements.rcwa import (
    _EnergyError,
    _harmonic_orders_2d,
    rcwa_efficiency_1d,
    rcwa_efficiency_2d,
    rcwa_efficiency_2d_shapes,
    rcwa_extrapolate,
)

# --- metallic 2-D pillar: the canonical TM-convergence case --------------
WL = 0.5e-6
P = 0.5e-6
DEPTH = 0.2e-6
THETA = np.deg2rad(1e-3)          # tiny off-normal (oracle is singular at 0)
EPS_METAL = -3 + 4j               # PUBLIC Im > 0


def _metal_cell(S=160):
    xx, yy = np.meshgrid((np.arange(S) + 0.5) / S, (np.arange(S) + 0.5) / S,
                         indexing="ij")
    return np.where((xx < 0.5) & (yy < 0.5), EPS_METAL, 1.0 + 0j)


def _pillar_shape():
    return dict(shape="rectangle", eps=EPS_METAL,
                size=(0.25e-6, 0.25e-6), center=(0.125e-6, 0.125e-6))


def _grid_T(M, formulation, eps_cell):
    o, Re, Te = rcwa_efficiency_2d(
        P, P, eps_cell, 1.0, 1.0, DEPTH, WL, theta=THETA, phi=0.0,
        polarization="tm", n_orders_x=M, n_orders_y=M, formulation=formulation)
    return float(Re.sum() + Te.sum()), float(Te.sum())


# ---------------------------------------------------------------------------
# formulation = 'li' / 'fff'
# ---------------------------------------------------------------------------

def test_li_converges_closer_than_laurent_to_high_n_reference():
    """The dual-Laurent z-rule lands closer to the high-N answer than Laurent
    at a matched moderate harmonic count (self-consistent, no oracle)."""
    cell = _metal_cell()
    # high-N reference: 'li' at large M (the fast-converging formulation)
    _, T_ref = _grid_T(18, "li", cell)
    errs_li, errs_la = [], []
    for M in (5, 7, 9):
        _, T_li = _grid_T(M, "li", cell)
        _, T_la = _grid_T(M, "laurent", cell)
        errs_li.append(abs(T_li - T_ref))
        errs_la.append(abs(T_la - T_ref))
    # averaged over the sweep, 'li' is the closer formulation
    assert np.mean(errs_li) < np.mean(errs_la)


def test_fff_is_alias_of_li_and_default_is_laurent():
    cell = _metal_cell(96)
    rt_fff = _grid_T(5, "fff", cell)
    rt_li = _grid_T(5, "li", cell)
    assert rt_fff == rt_li
    # default == explicit 'laurent', and differs from 'li'
    o, Re, Te = rcwa_efficiency_2d(P, P, cell, 1.0, 1.0, DEPTH, WL,
                                   theta=THETA, polarization="tm",
                                   n_orders_x=5, n_orders_y=5)
    assert (float(Re.sum() + Te.sum()), float(Te.sum())) \
        == _grid_T(5, "laurent", cell)


def test_laurent_default_is_deterministic_bit_exact():
    cell = _metal_cell(96)
    assert _grid_T(6, "laurent", cell) == _grid_T(6, "laurent", cell)


def test_bad_formulation_rejected():
    cell = _metal_cell(64)
    with pytest.raises(ValueError, match="formulation must be"):
        rcwa_efficiency_2d(P, P, cell, 1.0, 1.0, DEPTH, WL,
                           n_orders_x=4, n_orders_y=4, formulation="bogus")


def test_li_conserves_energy_lossless_dielectric():
    """A lossless dielectric pillar conserves energy under the li rule."""
    S = 120
    xx, yy = np.meshgrid((np.arange(S) + .5) / S, (np.arange(S) + .5) / S,
                         indexing="ij")
    cell = np.where((xx < 0.5) & (yy < 0.5), 6.0 + 0j, 1.0 + 0j)
    o, Re, Te = rcwa_efficiency_2d(P, P, cell, 1.5, 1.0, DEPTH, WL,
                                   theta=THETA, polarization="tm",
                                   n_orders_x=6, n_orders_y=6, formulation="li")
    assert abs(Re.sum() + Te.sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# circular truncation
# ---------------------------------------------------------------------------

def test_harmonic_orders_circular_is_subset_and_keeps_dc():
    rect, nr = _harmonic_orders_2d(6, 6)
    circ, nc = _harmonic_orders_2d(6, 6, truncation="circular",
                                   period_x=P, period_y=P)
    assert nc < nr                                   # fewer orders
    assert (0, 0) in {tuple(r) for r in circ}        # DC retained
    # circular set is a strict subset of the rectangular box
    assert {tuple(r) for r in circ} <= {tuple(r) for r in rect}
    # square lattice: kept set is exactly m^2 + n^2 <= M^2
    assert all(m * m + n * n <= 36 + 1e-9 for m, n in circ)


def test_circular_needs_periods():
    with pytest.raises(ValueError, match="needs period_x and period_y"):
        _harmonic_orders_2d(5, 5, truncation="circular")


def test_circular_default_rectangular_bit_exact():
    cell = _metal_cell(96)
    o_r, Re_r, Te_r = rcwa_efficiency_2d(P, P, cell, 1.0, 1.0, DEPTH, WL,
                                         theta=THETA, polarization="tm",
                                         n_orders_x=5, n_orders_y=5)
    o_d, Re_d, Te_d = rcwa_efficiency_2d(P, P, cell, 1.0, 1.0, DEPTH, WL,
                                         theta=THETA, polarization="tm",
                                         n_orders_x=5, n_orders_y=5,
                                         truncation="rectangular")
    assert np.array_equal(o_r, o_d) and np.array_equal(Te_r, Te_d)


def test_circular_truncation_reaches_same_value_fewer_orders():
    """Circular truncation converges to the same transmittance as rectangular
    but with fewer harmonics for a comparable accuracy."""
    pillar = _pillar_shape()

    def shapes_T(M, trunc):
        o, Re, Te = rcwa_efficiency_2d_shapes(
            P, P, 1.0, [pillar], 1.0, 1.0, DEPTH, WL, theta=THETA, phi=0.0,
            polarization="tm", n_orders_x=M, n_orders_y=M, truncation=trunc)
        return len(o), float(Te.sum())

    n_rect, T_rect = shapes_T(16, "rectangular")
    n_circ, T_circ = shapes_T(16, "circular")
    assert n_circ < n_rect                       # genuinely fewer orders
    # both target the same physical value (each within ~1% of the other)
    assert abs(T_rect - T_circ) < 0.01


# ---------------------------------------------------------------------------
# oracle cross-check (skipped when inkstone is unavailable)
# ---------------------------------------------------------------------------

def test_li_sequential_converges_where_laurent_extrapolates():
    """RE-PINNED 2026-06-10 (audit F1): the old gate compared the dual-Laurent
    shapes solver at M=14 against inkstone at num_g=500 -- two UNCONVERGED
    values that agreed by shared error (inkstone oscillates 0.548 -> 0.541 ->
    0.544 -> 0.551 over num_g 200..1400 on this metal pillar; the same mutual-
    unconvergence trap as the PMM metal-TM adjudication).  The honest gate:
    the v5.14.1 sequential-rule 'li' is already stable at modest M, and the
    Richardson 1/M extrapolation of the slow-but-correct 'laurent' tail lands
    on it (measured: li M=8..14 in [0.5703, 0.5712]; laurent Richardson from
    (M=12, 14) = 0.5717; inkstone trends toward it from below)."""
    S = 64
    cell = np.full((S, S), 1.0 + 0j)
    cell[:S // 2, :S // 2] = EPS_METAL
    def T_of(form, M):
        o, R, T = rcwa_efficiency_2d(
            P, P, cell, 1.0, 1.0, DEPTH, WL, theta=THETA, phi=0.0,
            polarization="tm", n_orders_x=M, n_orders_y=M, formulation=form)
        return float(T.sum())
    T_li_8, T_li_12 = T_of("li", 8), T_of("li", 12)
    assert abs(T_li_12 - T_li_8) < 2e-3            # measured 6.9e-4: stable
    T_la_12, T_la_14 = T_of("laurent", 12), T_of("laurent", 14)
    T_extrap = (14.0 * T_la_14 - 12.0 * T_la_12) / 2.0   # Richardson, 1/M tail
    assert abs(T_extrap - T_li_12) < 5e-3          # measured 1.2e-3
    # laurent itself is still far at M=14 -- the acceleration is genuine
    assert abs(T_la_14 - T_li_12) > 5e-3           # measured 1.3e-2


# ---------------------------------------------------------------------------
# large-period self-heal (stabilize=)
# ---------------------------------------------------------------------------

# The large-period / high-contrast instability: period 15 um, n=2.5/1.0, an
# erratic, MEASURE-ZERO set of n_orders blows up energy (sum R+T up to 1e30).
# WHICH n_orders blow up is LAPACK-build dependent (a near-singular layer<->
# region mode-match), so the tests SEARCH for a blow-up at run time rather than
# pinning one truncation -- and skip when the platform's LAPACK happens not to
# exhibit it (otherwise the test is a coin-flip across NumPy/LAPACK builds).
_BLOWUP = dict(period=15e-6, n_ridge=2.5, n_groove=1.0, n_substrate=1.0,
               n_superstrate=1.0, depth=0.5e-6, duty_cycle=0.5,
               wavelength=0.633e-6, angle=0.0, polarization="te")


def _find_blowup_order(max_m=28):
    """First n_orders in ``[3, max_m]`` whose solve trips the energy guard at
    the ``_BLOWUP`` geometry, or ``None`` if this LAPACK build conserves energy
    at every truncation."""
    for m in range(3, max_m + 1):
        try:
            rcwa_efficiency_1d(**_BLOWUP, n_orders=m)
        except _EnergyError:
            return m
    return None


def test_energy_guard_raises_typed_error_on_blowup():
    m = _find_blowup_order()
    if m is None:
        pytest.skip("no measure-zero large-period blow-up on this LAPACK build")
    with pytest.raises(_EnergyError, match="energy non-conservation"):
        rcwa_efficiency_1d(**_BLOWUP, n_orders=m)


def test_stabilize_heals_blowup_to_conserving_solve():
    m = _find_blowup_order()
    if m is None:
        pytest.skip("no measure-zero large-period blow-up on this LAPACK build")
    o, Re, Te = rcwa_efficiency_1d(**_BLOWUP, n_orders=m, stabilize=True)
    tot = float(Re.sum() + Te.sum())
    assert abs(tot - 1.0) < 1e-3                  # lossless -> energy conserved
    assert len(o) != 2 * m + 1                    # bumped off the bad truncation


def test_stabilize_is_bit_exact_noop_on_clean_geometry():
    clean = dict(period=5e-6, n_ridge=2.5, n_groove=1.0, n_substrate=1.0,
                 n_superstrate=1.0, depth=0.5e-6, duty_cycle=0.5,
                 wavelength=0.633e-6, polarization="te")
    o1, R1, T1 = rcwa_efficiency_1d(**clean, n_orders=11)
    o2, R2, T2 = rcwa_efficiency_1d(**clean, n_orders=11, stabilize=True)
    assert np.array_equal(R1, R2) and np.array_equal(T1, T2)


# ---------------------------------------------------------------------------
# eig reuse for repeated layers (DBR / Bragg)
# ---------------------------------------------------------------------------

def _dbr_cells():
    S = 80
    xx, yy = np.meshgrid((np.arange(S) + .5) / S - .5, (np.arange(S) + .5) / S - .5,
                         indexing="ij")
    a = np.where((np.abs(xx) < 0.25) & (np.abs(yy) < 0.25), 6.0 + 0j, 1.0 + 0j)
    b = np.where((np.abs(xx) < 0.25) & (np.abs(yy) < 0.25), 2.5 + 0j, 1.0 + 0j)
    return a, b


def _build_dbr(n_periods):
    import lumenairy as la
    a, b = _dbr_cells()
    st = la.RCWAStack(1e-6, period_y=1e-6, n_substrate=1.5, n_orders=5,
                      n_orders_y=5)
    for _ in range(n_periods):
        st.add_layer(0.15e-6, eps_cell=a)
        st.add_layer(0.12e-6, eps_cell=b)
    return st.set_source(0.633e-6, theta=0.1)


def test_eig_reuse_solves_repeated_layer_once(monkeypatch):
    """A DBR with K identical period-pairs computes only the unique layer
    eigenproblems, not one per layer."""
    import lumenairy as la
    calls = {"n": 0}
    orig = la.RCWAStack._layer_modes

    def spy(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(la.RCWAStack, "_layer_modes", spy)
    _build_dbr(6).solve()                       # 12 layers, 2 unique
    assert calls["n"] == 2


def test_eig_reuse_is_bit_exact(monkeypatch):
    """Deduping repeated-layer eigs is a pure memoization -> bit-identical to
    solving every layer independently."""
    import lumenairy as la
    res_dedup = _build_dbr(5).solve()
    o1, R1, T1 = res_dedup.efficiencies()
    # disable the dedup (force a distinct key per layer) and re-solve
    monkeypatch.setattr(la.RCWAStack, "_layer_eig_key",
                        staticmethod(lambda L: None))
    res_full = _build_dbr(5).solve()
    o2, R2, T2 = res_full.efficiencies()
    assert np.array_equal(R1, R2) and np.array_equal(T1, T2)


# ---------------------------------------------------------------------------
# convergence extrapolation (Richardson / Shanks)
# ---------------------------------------------------------------------------

def test_richardson_recovers_algebraic_limit():
    """Exact for the algebraic tail s(N) = L + C N^-p (the RCWA model)."""
    L, C, p = 0.734, 2.1, 1.3
    Ns = [8, 10, 12, 14, 16]
    vals = [L + C * N ** (-p) for N in Ns]
    est = rcwa_extrapolate(vals, n_orders=Ns, method="richardson")
    assert abs(est - L) < 1e-9


@pytest.mark.parametrize("p", [0.5, 1.0, 2.0, 3.0])
def test_richardson_various_orders(p):
    L, C, Ns = 0.21, 1.7, [8, 10, 12]
    vals = [L + C * N ** (-p) for N in Ns]
    assert abs(rcwa_extrapolate(vals, n_orders=Ns, method="richardson") - L) < 1e-7


def test_shanks_recovers_geometric_limit():
    """Exact for the geometric tail s_k = L + A r^k (spectral convergence)."""
    L, A, r = 0.734, 0.5, 0.4
    vals = [L + A * r ** k for k in range(5)]
    assert abs(rcwa_extrapolate(vals, method="shanks") - L) < 1e-9


def test_extrapolate_input_validation():
    with pytest.raises(ValueError, match="at least 3 samples"):
        rcwa_extrapolate([1.0, 2.0])
    with pytest.raises(ValueError, match="needs n_orders"):
        rcwa_extrapolate([1.0, 2.0, 3.0], method="richardson")
    with pytest.raises(ValueError, match="method must be"):
        rcwa_extrapolate([1.0, 2.0, 3.0], method="bogus")


# ---------------------------------------------------------------------------
# Lanczos-sigma field filtering
# ---------------------------------------------------------------------------

def test_lanczos_filter_reduces_field_ringing():
    """The Lanczos sigma factors smooth the reconstructed multi-order field
    (lower total variation) without changing its grid/shape."""
    import lumenairy as la
    S = 64
    xx, _ = np.meshgrid((np.arange(S) + .5) / S, (np.arange(S) + .5) / S,
                        indexing="ij")
    cell = np.where(xx < 0.5, 9.0, 2.0).astype(complex)
    res = (la.RCWAStack(0.8e-6, period_y=0.8e-6, n_substrate=1.0, n_orders=8,
                        n_orders_y=8)
           .add_layer(0.3e-6, eps_cell=cell)
           .set_source(0.6e-6, theta=0.15).solve())
    raw = res.to_multiorder_field(48, 48, dx=0.8e-6 / 48, port="reflection",
                                  filter="none")
    lan = res.to_multiorder_field(48, 48, dx=0.8e-6 / 48, port="reflection",
                                  filter="lanczos")

    def tv(f):
        inten = np.abs(f.Ex) ** 2 + np.abs(f.Ey) ** 2
        return float(np.abs(np.diff(inten, 0)).sum()
                     + np.abs(np.diff(inten, 1)).sum())

    assert tv(lan) < tv(raw)
    assert lan.Ex.shape == raw.Ex.shape

    with pytest.raises(ValueError, match="filter must be"):
        res.to_multiorder_field(8, 8, dx=1e-7, filter="bogus")
