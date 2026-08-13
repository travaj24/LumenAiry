"""Unit tests for the rigorous (RCWA) DOE order table and its pipeline
decomposer -- ``validation/repro_traced_carrier_121/doe_rcwa_table.py`` and
``validation/pipeline/doe_rcwa.py``.

WHAT IS AND IS NOT TESTED HERE.  This file tests the INSTRUMENT: the structure
reconstruction (and the two conventions inside it that are silently-wrong
hazards), the energy closure, the convergence machinery, the angular quadrature,
the cache key discipline, and the decomposer's contract.  It does NOT test the
design-121 numbers -- those need the ``.zmx``, a 128-pixel Dammann cell and
hours of RCWA, and they are reported in
``docs/audits/BUILD_RCWA_DOE_TABLE_2026_08_12.md``.

Every fixture here is SYNTHETIC and TINY (8- and 12-pixel cells, periods of a
few wavelengths, ``n_orders <= 4``), so the file runs in seconds on a fresh
checkout with no Zemax file, no design-121 cache and no optional dependency.
That is possible because the table builder is design-agnostic: it takes a
phase-level map, not a design.

THE TWO ORACLES.  Where a claim can be checked against something that is not
RCWA, it is:

* the UNIFORM SLAB has a closed form (the Airy/Fabry-Perot amplitude), and it
  pins the whole stack machinery AND the transmitted-phase sign convention;
* the THIN-ELEMENT limit is an independent model, and a cell whose features are
  many wavelengths wide with a shallow relief must reproduce it.

No xfail, no skip.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / 'validation'),
           str(_REPO_ROOT / 'validation' / 'repro_traced_carrier_121')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import doe_rcwa_table as DT  # noqa: E402
from pipeline import doe_rcwa as PDR  # noqa: E402
from pipeline import sources as PS  # noqa: E402
from pipeline.spec import SpecError  # noqa: E402

LAM = 1.31e-6
N_GLASS = DT.N_FUSED_SILICA_1310

# ---------------------------------------------------------------------------
# BARS -- every one is set by the ENVELOPE RULE: measured on these fixtures,
# then given headroom, and the message says what was measured.
#
# A NOTE ON THE FIXTURES, because it is a real property of the instrument and
# not a testing convenience.  The solver's own energy guard names the regime
# "very large period / low index contrast", and this structure class sits in
# it: on 8-level fixtures at these periods the lossless closure wanders
# 1e-13 .. 1e-2 and isolated truncations blow up outright (measured; and the
# design-121 cell itself is unstable above n_orders = 5 -- see
# docs/audits/BUILD_RCWA_DOE_TABLE_2026_08_12.md).  The numeric tests here
# therefore use BINARY (2-level) relief, where the same sweep measured closure
# 1e-15 .. 1e-10 with no instability anywhere -- so a failure of one of these
# tests is a code defect, not the conditioning.  The 8-level structure is still
# exercised by every test that does not need a converged solve.
# ---------------------------------------------------------------------------
#: Lossless closure |R+T-1| on the BINARY fixtures.  Measured worst 6e-11 over
#: a 4-period x 3-index x 5-truncation sweep; 1e-8 is >2 decades of headroom.
#: REGIME-TIED (2026-08-13): that sweep and this bar are for the SMALL-period
#: fixtures (6-8 lambda).  The v5.35.3 release shard (py3.10 wheel LAPACK)
#: read 1.15e-8 on the 20-lambda TEA checkerboard while passing every
#: small-period site on the same run -- closure at large period varies with
#: the BLAS build (conditioning grows with period; the module note above
#: names 1e-2-scale pathology in the extreme).  So the 20-lambda fixtures
#: carry their own bar below: 2 decades over the worst cross-build reading
#: observed, 4 decades below the pathology it exists to catch.
BAR_CLOSURE = 1e-8
#: Lossless closure on the 20-LAMBDA fixtures (the TEA checkerboard and the
#: convergence ladder).  See the regime note on BAR_CLOSURE.
BAR_CLOSURE_20L = 1e-6
#: Slab amplitude against the analytic Airy form.  Measured ~1e-14 (it is the
#: same algebra evaluated two ways); 1e-9 is five decades of headroom.
BAR_SLAB = 1e-9
#: Thin-element agreement on a COARSE, SHALLOW cell.  Set in the test that uses
#: it, from the value measured there.
BAR_TEA_RATIO = 0.05


def _cell(n, seed=7, levels=8):
    """A deterministic multi-level cell that is DIFFERENT along x and y.

    Asymmetry is the point: a cell that happened to be symmetric could not
    detect the axis-order hazard :func:`DT.relief_heights` exists to prevent.
    """
    rng = np.random.default_rng(seed)
    lv = rng.integers(0, levels, size=(n, n))
    lv[0, :] = np.arange(n) % levels          # a distinctive first ROW  (y=0)
    lv[:, 0] = (2 * np.arange(n)) % levels    # a distinctive first COL  (x=0)
    return np.ascontiguousarray(lv)


def _struct(n=8, period_lam=6.0, n_doe=N_GLASS, seed=7, relief_sign=1,
            levels=8):
    return DT.DoeStructure(levels=_cell(n, seed, levels),
                           period=period_lam * LAM, wavelength=LAM,
                           n_doe=n_doe, n_levels=levels,
                           relief_sign=relief_sign, label='test')


def _binary(n=8, period_lam=6.0, n_doe=N_GLASS, seed=7):
    """The well-conditioned numeric fixture: 2-level relief (one RCWA slice)."""
    return _struct(n=n, period_lam=period_lam, n_doe=n_doe, seed=seed,
                   levels=2)


def _airy_t(n, thickness, wavelength):
    """Analytic normal-incidence transmission amplitude of a slab of index
    ``n`` in air.

    NO AIR-LEG DE-REFERENCE.  The solver's transmitted amplitude carries the
    FULL optical path through the stack, not the excess over an equal air leg;
    the two differ by exactly ``exp(-i k0 d)`` and this test is what measured
    that (the first version subtracted the leg and missed by 1.7576 rad =
    ``k0 d`` exactly)."""
    k0 = 2.0 * np.pi / wavelength
    r = (1.0 - n) / (1.0 + n)
    t1, t2 = 2.0 / (1.0 + n), 2.0 * n / (1.0 + n)
    ph = np.exp(1j * k0 * n * thickness)
    return t1 * t2 * ph / (1.0 - r * r * ph * ph)


# ===========================================================================
# (a) THE STRUCTURE -- reconstruction and its two conventions
# ===========================================================================
def test_levels_from_transmittance_round_trips_a_quantised_mask():
    lv = _cell(12)
    nf = np.exp(2j * np.pi * lv / 8.0)
    assert np.array_equal(DT.levels_from_transmittance(nf, 8), lv)


def test_levels_from_transmittance_refuses_an_unquantised_or_lossy_mask():
    """A silently ROUNDED level map is a silently different structure, so the
    constructor refuses rather than rounds."""
    lv = _cell(12)
    with pytest.raises(ValueError, match='not quantised'):
        DT.levels_from_transmittance(np.exp(1j * (2 * np.pi * lv / 8.0 + 0.31)),
                                     8)
    with pytest.raises(ValueError, match='phase-only'):
        DT.levels_from_transmittance(0.5 * np.exp(2j * np.pi * lv / 8.0), 8)
    with pytest.raises(ValueError, match='COMPLEX transmittance'):
        DT.levels_from_transmittance(2 * np.pi * lv / 8.0, 8)


def test_relief_heights_transposes_into_the_solvers_axis_order():
    """REGRESSION, silent-wrong class.  The Dammann cell is indexed
    ``[my, mx]`` (``_d121_common`` reads ``A[my + cy, mx + cx]``); the RCWA
    cell is indexed ``[x, y]`` (``_eps_convolution_2d``'s node contract).
    Handing one to the other untransposed solves the TRANSPOSED structure --
    energy-clean, convergent and wrong -- and on design 121's 8-wide-by-4-tall
    order block it reads ``sum |a|^2 = 0.4488`` against the true ``0.8851``."""
    st = _struct(n=8)
    H = DT.relief_heights(st)
    assert np.array_equal(H, st.levels.T)
    assert not np.array_equal(H, st.levels), (
        'the fixture cell is symmetric, so this test cannot see the hazard it '
        'exists to catch -- fix _cell(), not the assertion')


def test_relief_sign_minus_one_is_the_complementary_column_height():
    st = _struct(n=8, relief_sign=-1)
    assert np.array_equal(DT.relief_heights(st), 7 - st.levels.T)


def test_the_derived_relief_realises_one_full_wave_of_phase():
    """ASSUMPTION A2 in arithmetic: ``n_levels`` steps of ``dz`` is exactly
    ``2 pi`` of thin-element phase, so the staircase spans the design."""
    st = _struct()
    k0 = 2.0 * np.pi / LAM
    assert k0 * (st.n_doe - 1.0) * (st.dz * st.n_levels) == pytest.approx(
        2.0 * np.pi, rel=1e-12)
    assert st.relief_total == pytest.approx(7.0 * st.dz, rel=1e-15)


def test_the_structure_refuses_inputs_it_cannot_realise():
    with pytest.raises(ValueError, match='INTEGER level index'):
        DT.DoeStructure(levels=np.zeros((4, 4)), period=6 * LAM,
                        wavelength=LAM)
    with pytest.raises(ValueError, match='outside'):
        DT.DoeStructure(levels=np.full((4, 4), 9), period=6 * LAM,
                        wavelength=LAM)
    with pytest.raises(ValueError, match='SQUARE'):
        DT.DoeStructure(levels=np.zeros((4, 6), dtype=int), period=6 * LAM,
                        wavelength=LAM)
    with pytest.raises(ValueError, match='does not exceed the surrounding air'):
        DT.DoeStructure(levels=_cell(4), period=6 * LAM, wavelength=LAM,
                        n_doe=1.0)


# ===========================================================================
# (b) THE ANALYTIC ORACLE -- a uniform slab, in closed form
# ===========================================================================
@pytest.mark.parametrize('n_levels_of_glass', [1, 4, 7])
def test_a_uniform_cell_reproduces_the_analytic_slab_amplitude(
        n_levels_of_glass):
    """The whole stack machinery against a closed form.  A cell at ONE level is
    a plain slab, and its zeroth-order transmission is the Airy amplitude.
    This pins the layer stacking, the half-space handling, the eps-vs-index
    convention and the phase reference all at once -- none of which RCWA can
    check against itself."""
    lv = np.full((8, 8), int(n_levels_of_glass), dtype=int)
    st = DT.DoeStructure(levels=lv, period=6 * LAM, wavelength=LAM,
                         n_doe=N_GLASS, n_levels=8)
    # a uniform cell is a slab only if both half-spaces are air, so drive the
    # slab geometry directly rather than through the substrate-side stack.
    from lumenairy.elements.rcwa import RCWAStack
    thick = st.dz * n_levels_of_glass
    stack = RCWAStack(st.period, period_y=st.period, n_superstrate=1.0,
                      n_substrate=1.0, n_orders=2, n_orders_y=2)
    stack.add_layer(thick, eps_cell=np.full((9, 9), complex(N_GLASS ** 2)))
    res = stack.set_source(LAM, theta=0.0).solve()
    m = res.per_order_amplitudes('transmission')
    o = np.asarray(m['orders'])
    i0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    got = complex(np.asarray(m['Ex'])[0, i0])
    assert abs(got - _airy_t(N_GLASS, thick, LAM)) < BAR_SLAB, (
        f'uniform-cell zeroth-order transmission {got!r} is not the analytic '
        f'Airy amplitude {_airy_t(N_GLASS, thick, LAM)!r} for a '
        f'{thick * 1e6:.4f} um slab of n = {N_GLASS}')


def test_verify_relief_sign_measures_the_exp_minus_iwt_convention():
    """The phase-to-height sign is MEASURED, not assumed: adding material must
    ADVANCE the transmitted phase under ``exp(-i w t)``.  It is measured as a
    DERIVATIVE over a small thickness step, because the absolute phase is
    ambiguous on two counts -- the reference plane (the solver returns
    ``k0 n d``, not ``k0 (n-1) d``) and ``2 pi`` wrapping (``k0 n d`` is
    17.8 rad at the design relief)."""
    d_meas, d_exp, sign = DT.verify_relief_sign(N_GLASS, LAM)
    assert sign == 1, (
        f'the library no longer advances the transmitted phase with optical '
        f'thickness: a step that should move it {d_exp:+.4f} rad moved it '
        f'{d_meas:+.4f} rad.  Every relief in this module is built on that '
        f'sign; flip DoeStructure.relief_sign and re-key every table.')
    assert abs(d_exp) > 0.2, 'the probe step is too small to discriminate'
    assert d_meas == pytest.approx(d_exp, rel=0.15), (
        f'the measured phase step {d_meas:.4f} rad is not the predicted '
        f'{d_exp:.4f} rad within the slab-interface residual this fixture '
        f'measured (~0.3 %)')


# ===========================================================================
# (c) ENERGY CLOSURE + the y-invariance control
# ===========================================================================
def test_energy_closes_on_the_lossless_reconstructed_cell():
    st = _binary(n=8, period_lam=8.0)
    r = DT.solve_orders(st, 4)
    assert np.max(np.abs(r['closure'])) < BAR_CLOSURE, (
        f"R+T-1 = {r['closure']} on a provably lossless dielectric cell")
    assert np.all(r['eta_T'] >= -1e-14) and np.all(r['eta_R'] >= -1e-14)
    assert not r['stabilized'], (
        'this fixture is supposed to be a CLEAN solve -- a stabilized rung '
        'means the truncation returned is not the one requested')


def test_a_y_invariant_cell_puts_no_power_in_a_y_diffracted_order():
    """THE AXIS CONTROL, end to end.  A cell that varies only along x is a 1-D
    stripe grating: its ``(m, n != 0)`` orders are exactly decoupled and must
    carry nothing.  If the cell reached the solver transposed, the SAME
    structure would radiate into ``(0, n)`` instead -- so this catches the
    hazard through the physics rather than through an array identity."""
    row = (np.arange(8) >= 4).astype(int)      # ONE binary period per cell
    st = DT.DoeStructure(levels=np.tile(row, (8, 1)), period=6 * LAM,
                         wavelength=LAM, n_doe=N_GLASS, n_levels=2)
    #  ^ constant down axis 0 (= y of the Dammann convention) -> varies in x
    r = DT.solve_orders(st, 3)
    o, e = np.asarray(r['orders']), r['eta_T'][0]
    p_x = float(e[(o[:, 1] == 0) & (o[:, 0] != 0)].sum())
    p_y = float(e[o[:, 1] != 0].sum())
    assert p_x > 0.05, (
        f'the fixture is not diffracting in x at all (power {p_x:.3e}); it '
        f'cannot demonstrate the axis convention')
    assert p_y < 1e-12 * max(p_x, 1e-30) + 1e-14, (
        f'a y-invariant cell radiated {p_y:.3e} into y-diffracted orders '
        f'against {p_x:.3e} into x -- the cell reached the solver TRANSPOSED')


# ===========================================================================
# (d) THE THIN-ELEMENT LIMIT -- an independent model
# ===========================================================================
def _tea_split_error(n_levels, n_orders=4):
    """Worst RCWA-vs-thin-element SPLITTING deviation for a checkerboard whose
    phase step is ``2 pi / n_levels``, normalised by the strongest order.

    A CHECKERBOARD OF ONE PHASE STEP, not a full 2 pi element, because the two
    knobs cannot both be relaxed on a full-wave element: the relief that
    realises 2 pi is ``lambda / (n - 1)``, so a SHALLOW relief demands a HIGH
    index (strong walls) and a LOW index demands a DEEP relief (strong
    propagation).  Weakening the PHASE relaxes both at once, which is what puts
    the structure in the regime the thin-element model actually claims.
    """
    ix = np.arange(8) >= 4
    st = DT.DoeStructure(levels=(ix[:, None] ^ ix[None, :]).astype(int),
                         period=20 * LAM, wavelength=LAM, n_doe=N_GLASS,
                         n_levels=int(n_levels))
    want = np.array([(m, n) for n in (-1, 0, 1) for m in (-1, 0, 1)])
    r = DT.solve_orders(st, int(n_orders), want=want, on_unstable='raise')
    c = DT.compare_to_scalar(st, want, r['amp'][0], total_T=r['sum_T'][0])
    assert np.max(np.abs(r['closure'])) < BAR_CLOSURE_20L, (
        f"R+T-1 = {r['closure']} on the lossless 20-lambda TEA checkerboard "
        f"-- beyond even the large-period cross-build envelope")
    return float(np.max(np.abs(c['d_frac'])) / c['frac_scalar'].max())


def test_a_weak_phase_cell_reproduces_the_thin_element_split():
    """RCWA must converge to the scalar model where the scalar model is valid.

    SCORED ON THE SPLITTING RATIO, not on the raw efficiency: the scalar model
    is a unit-modulus phase screen with NO Fresnel reflection, so it cannot
    agree on absolute throughput with any real dielectric.  What the
    thin-element limit claims -- and what is tested -- is that the transmitted
    light is SPLIT between orders the same way."""
    e = _tea_split_error(32)
    assert e < 0.01, (
        f'RCWA splits the transmitted light differently from the thin-element '
        f'table by {e:.5f} of the strongest order on a checkerboard whose '
        f'phase step is only 2 pi / 32 = 0.196 rad, where the two models must '
        f'agree.  Measured 0.00166; bar 0.01.')


def test_the_thin_element_error_falls_away_as_the_phase_step_weakens():
    """THE TREND, which is the campaign's whole claim in one fixture: the
    thin-element model is not wrong or right, it degrades with the phase depth
    the element carries.  Measured on this checkerboard the splitting error
    runs 0.164 / 0.148 / 0.028 / 0.0068 / 0.0017 as the phase step goes
    pi -> pi/2 -> pi/4 -> pi/8 -> pi/16, i.e. TWO DECADES.  A model or
    convention defect that shifted the RCWA table would break the trend, not
    just the level."""
    errs = [_tea_split_error(k) for k in (2, 4, 8, 16, 32)]
    assert errs[0] > 20 * errs[-1], (
        f'the thin-element error no longer collapses with the phase step: '
        f'{[round(e, 5) for e in errs]} (pi .. pi/16).  Either RCWA and the '
        f'scalar table no longer share a limit, or the fixture stopped '
        f'spanning the regimes.')
    for a, b in zip(errs[1:], errs[2:]):        # monotone from pi/2 down
        assert b < a, (f'non-monotone approach to the thin-element limit: '
                       f'{[round(e, 5) for e in errs]}')


# ===========================================================================
# (e) CONVERGENCE
# ===========================================================================
def test_the_convergence_ladder_settles_and_reports_shrinking_deltas():
    """The ladder's contract: successive rungs report per-order deltas, and on
    a cell the truncation can actually resolve those deltas SHRINK and land
    under a bar.  The criterion is on the AMPLITUDES, not on total power -- a
    lossless cell conserves energy at every truncation, so energy proves
    nothing about the per-order split (the lossless trap)."""
    st = DT.DoeStructure(levels=_cell(8, levels=2), period=20 * LAM,
                         wavelength=LAM, n_doe=3.0, n_levels=2)
    want = np.array([(m, n) for n in (-1, 0, 1) for m in (-1, 0, 1)])
    rows = DT.convergence_ladder(st, want, (3, 4, 5, 6), log=None)
    assert len(rows) == 4 and 'd_amp_max' not in rows[0]
    d = [r['d_amp_max'] for r in rows[1:]]
    assert d[-1] < d[0], (
        f'the ladder is not converging on a resolvable fixture: rung-to-rung '
        f'amplitude movement went {d}')
    assert d[-1] < 0.02, f'last rung still moving by {d[-1]:.4f} in amplitude'
    for r in rows:
        assert np.max(np.abs(r['closure'])) < BAR_CLOSURE_20L, (
            f"R+T-1 = {r['closure']} on the lossless 20-lambda ladder fixture")


def test_the_ladder_reports_the_piston_removed_phase_movement_separately():
    """A uniform phase shift between rungs is not a convergence failure of the
    RELATIVE order phases, which is the only thing a coherent decomposer
    consumes -- so the ladder reports both and the relative one is the
    criterion."""
    st = DT.DoeStructure(levels=_cell(8, levels=2), period=20 * LAM,
                         wavelength=LAM, n_doe=3.0, n_levels=2)
    want = np.array([(m, n) for n in (-1, 0, 1) for m in (-1, 0, 1)])
    rows = DT.convergence_ladder(st, want, (3, 5), log=None)
    r = rows[1]
    assert set(('d_eta_max', 'd_amp_max', 'd_phase_max',
                'd_phase_rel_max', 'd_sumT')) <= set(r)
    assert 0.0 <= r['d_phase_rel_max'] <= np.pi + 1e-12
    # the DEFINING property, checked directly rather than through the ladder:
    # a global phase on one arm moves the raw phase delta and leaves the
    # piston-removed one alone.
    o, a_s = DT.scalar_table(st, want=want)
    a_r = np.asarray(rows[-1]['amp'][0])
    c0 = DT.compare_to_scalar(st, want, a_r)
    c1 = DT.compare_to_scalar(st, want, a_r * np.exp(0.7j))
    assert np.allclose(c1['d_phase_rel'], c0['d_phase_rel'], atol=1e-12)
    assert abs(float((c1['piston'] - c0['piston'] + np.pi) % (2 * np.pi)
                     - np.pi) - 0.7) < 1e-12


def test_the_cell_upsample_clears_the_solvers_aliasing_bound_exactly():
    """Nearest-neighbour replication is EXACT for a piecewise-constant relief,
    so raising the truncation past ``S/4`` costs geometry nothing."""
    assert DT._cell_upsample_for(8, 1) == 1
    assert DT._cell_upsample_for(8, 2) == 2       # needs 9 > 8
    assert DT._cell_upsample_for(128, 31) == 1    # needs 125 <= 128
    assert DT._cell_upsample_for(128, 32) == 2
    st = _struct(n=8)
    s1 = DT.build_stack(st, 1, cell_upsample=1)
    s2 = DT.build_stack(st, 1, cell_upsample=3)
    c1 = np.asarray(s1.layers[0].data)
    c3 = np.asarray(s2.layers[0].data)
    assert c3.shape == (24, 24)
    assert np.array_equal(c3, np.kron(c1, np.ones((3, 3))))


# ===========================================================================
# (f) THE ANGULAR GRID + THE AVERAGING
# ===========================================================================
def test_clenshaw_curtis_integrates_an_analytic_function_spectrally():
    """The quadrature that turns the angle table into one weight per order.
    On an analytic integrand it must beat the trapezoid rule badly enough to
    justify the Chebyshev choice, and it must be EXACT on a polynomial."""
    for n in (5, 9):
        t = DT.chebyshev_theta(1.0, n)
        w = DT.clenshaw_curtis_weights(n)
        assert float(w.sum()) == pytest.approx(1.0, abs=1e-13)
        assert float((w * t ** 2).sum()) == pytest.approx(1.0 / 3.0, abs=1e-13)
    exact = np.sin(3.0) / 3.0                    # INT_0^1 cos(3 t) dt
    t9 = DT.chebyshev_theta(1.0, 9)
    w9 = DT.clenshaw_curtis_weights(9)
    cc = abs(float((w9 * np.cos(3.0 * t9)).sum()) - exact)
    tu = np.linspace(0.0, 1.0, 9)
    tr = abs(float(np.trapezoid(np.cos(3.0 * tu), tu)) - exact)
    assert cc < 1e-6 and cc < tr / 100.0, (
        f'Clenshaw-Curtis error {cc:.3e} is not decisively better than the '
        f'9-point trapezoid {tr:.3e}; the Chebyshev grid is not earning its '
        f'justification')


def test_the_angle_grid_puts_theta_zero_first_and_carries_the_jacobian():
    th, ph, W = DT.angle_grid(5e-4, 3, 8)
    assert th[0] == 0.0 and th[-1] == pytest.approx(5e-4)
    assert len(ph) == 8 and ph[0] == 0.0
    assert W.shape == (3, 8)
    # sum of weights == INT theta dtheta dphi / (2 pi) over the disk
    assert float(W.sum()) == pytest.approx(0.5 * 5e-4 ** 2, rel=1e-12)


def test_coherent_and_incoherent_averages_agree_when_the_table_is_angle_flat():
    """``coherence`` is the MEASUREMENT of how much the collapse choice costs:
    it is exactly 1 when the amplitudes do not vary over the cone, so a table
    that is flat must give the two collapses bit-comparable answers."""
    K, nt, nph = 3, 3, 4
    a = np.array([0.4 + 0.1j, -0.2 + 0.3j, 0.5 - 0.05j])
    tab = {'orders': np.zeros((K, 2), dtype=int),
           'amp': np.broadcast_to(a, (nt, nph, 2, K)).copy(),
           'quad_weight': np.full((nt, nph), 0.25)}
    out = DT.beam_weighted_amplitudes(tab)
    assert np.allclose(out['amp_coherent'], a, atol=1e-15)
    assert np.allclose(out['amp_incoherent'], np.abs(a), atol=1e-15)
    assert np.allclose(out['coherence'], 1.0, atol=1e-14)


def test_a_phase_spread_across_the_cone_shows_up_as_lost_coherence():
    """The other direction, so the metric is not vacuous: a table whose phase
    swings across the cone must report coherence BELOW 1, and the coherent mean
    must lose exactly the power the incoherent one keeps."""
    a0 = 0.5 + 0.0j
    amp = np.zeros((2, 1, 2, 1), dtype=complex)
    amp[0, 0, :, 0] = a0
    amp[1, 0, :, 0] = a0 * np.exp(1j * np.pi / 2)
    tab = {'orders': np.zeros((1, 2), dtype=int), 'amp': amp,
           'quad_weight': np.full((2, 1), 0.5)}
    out = DT.beam_weighted_amplitudes(tab)
    assert float(out['coherence'][0]) == pytest.approx(0.5, abs=1e-12)
    assert abs(out['amp_coherent'][0]) < out['amp_incoherent'][0]


def test_the_gaussian_beam_weight_decays_over_the_cone():
    th, ph, _W = DT.angle_grid(5e-4, 4, 3)
    g = DT.gaussian_beam_angular_weight(th, ph, 2.5e-4)
    assert g.shape == (4, 3)
    assert g[0, 0] == pytest.approx(1.0)
    assert np.all(np.diff(g[:, 0]) < 0)
    assert np.allclose(g[:, 0][:, None], g)          # azimuth-uniform


# ===========================================================================
# (g) THE CACHE KEY DISCIPLINE
# ===========================================================================
def _tiny_table(tmp_path, **over):
    st = over.pop('struct', None) or _struct(n=8)
    kw = dict(n_orders=2, thetas=np.array([0.0]), phis=np.array([0.0]),
              cache_dir=str(tmp_path), max_workers=1, log=None)
    kw.update(over)
    want = np.array([(0, 0), (1, 0), (0, 1)])
    return st, want, DT.build_table(st, want, quad_weight=np.ones((1, 1)), **kw)


def test_build_table_round_trips_through_its_cache(tmp_path):
    st, want, t1 = _tiny_table(tmp_path)
    assert Path(t1['path']).exists()
    t2 = DT.build_table(st, want, 2, np.array([0.0]), np.array([0.0]),
                        cache_dir=str(tmp_path), max_workers=1, log=None,
                        quad_weight=np.ones((1, 1)))
    assert t2['digest'] == t1['digest'] and t2['path'] == t1['path']
    for f in ('amp', 'eta_T', 'eta_R', 'xpol', 'sum_R', 'sum_T', 'closure'):
        assert np.array_equal(np.asarray(t2[f]), np.asarray(t1[f])), f
    assert np.max(np.abs(np.asarray(t1['closure']))) < BAR_CLOSURE


def test_a_cache_file_that_does_not_carry_its_own_key_is_refused(tmp_path):
    """Belt AND braces, the ``_d121_common`` discipline: the hash names the
    file, the STORED key proves it, so a file copied onto a matching name is
    refused rather than used."""
    st, want, t1 = _tiny_table(tmp_path)
    d = dict(np.load(t1['path'], allow_pickle=False))
    d['key_json'] = np.array('{"schema": 0}')
    np.savez_compressed(t1['path'], **d)
    with pytest.raises(RuntimeError, match='does not carry the cache key'):
        DT.build_table(st, want, 2, np.array([0.0]), np.array([0.0]),
                       cache_dir=str(tmp_path), max_workers=1, log=None,
                       quad_weight=np.ones((1, 1)))


@pytest.mark.parametrize('field,value', [
    ('n_doe', 1.6), ('relief_sign', -1), ('n_levels', 4)])
def test_a_structure_change_re_keys_the_table(field, value, tmp_path):
    base = _struct(n=8)
    kw = dict(levels=base.levels, period=base.period, wavelength=base.wavelength,
              n_doe=base.n_doe, n_levels=base.n_levels,
              relief_sign=base.relief_sign)
    if field == 'n_levels':
        kw['levels'] = base.levels % 4
    kw[field] = value
    other = DT.DoeStructure(**kw)
    want = np.array([(0, 0), (1, 0)])
    a = DT.table_key(base, want, 2, [0.0], [0.0], formulation='laurent',
                     truncation='rectangular', cell_upsample=None)[1]
    b = DT.table_key(other, want, 2, [0.0], [0.0], formulation='laurent',
                     truncation='rectangular', cell_upsample=None)[1]
    assert a != b, f'changing {field} did not re-key the table'


@pytest.mark.parametrize('kw', [
    {'n_orders': 3}, {'thetas': [0.0, 1e-4]}, {'phis': [0.0, 1.0]},
    {'truncation': 'circular'}, {'formulation': 'li'}, {'cell_upsample': 2}])
def test_every_sweep_parameter_is_in_the_key(kw):
    st = _struct(n=8)
    want = np.array([(0, 0), (1, 0)])
    base = dict(n_orders=2, thetas=[0.0], phis=[0.0], formulation='laurent',
                truncation='rectangular', cell_upsample=None)
    a = DT.table_key(st, want, base.pop('n_orders'), base.pop('thetas'),
                     base.pop('phis'), **base)[1]
    b2 = dict(n_orders=2, thetas=[0.0], phis=[0.0], formulation='laurent',
              truncation='rectangular', cell_upsample=None)
    b2.update(kw)
    b = DT.table_key(st, want, b2.pop('n_orders'), b2.pop('thetas'),
                     b2.pop('phis'), **b2)[1]
    assert a != b, f'{kw} did not re-key the table'


def test_a_changed_level_map_re_keys_even_at_the_same_shape():
    a = _struct(n=8, seed=7)
    b = _struct(n=8, seed=8)
    assert a.key()['levels_sha256'] != b.key()['levels_sha256']


def test_the_library_source_hash_is_in_the_key():
    """The defect-D6 field: a library DEFAULT flip within one version changes
    this table and can never appear in a hand-spelled filename."""
    st = _struct(n=8)
    k, _d = DT.table_key(st, [(0, 0)], 2, [0.0], [0.0], formulation='laurent',
                         truncation='rectangular', cell_upsample=None)
    assert len(k['lumenairy_source_sha256']) == 64
    assert len(k['builder_sha256']) == 64
    assert k['structure']['n_doe'] == repr(float(st.n_doe))


# ===========================================================================
# (h) THE DECOMPOSER
# ===========================================================================
def test_the_rcwa_decomposer_is_registered_under_its_own_name():
    fn = PS.get_decomposer('design121_doe_rcwa')
    assert fn is PDR.decompose_design121_doe_rcwa
    assert PS.get_decomposer('design121_doe') is not fn, (
        'the RCWA decomposer must be a SEPARATE registration -- replacing the '
        'scalar one in place would make the A/B unspellable')


def test_an_unregistered_decomposer_still_names_the_new_one_in_its_message():
    with pytest.raises(SpecError, match='design121_doe_rcwa'):
        PS.get_decomposer('nope')


def test_rcwa_weights_returns_a_json_safe_record_and_matching_lengths(tmp_path,
                                                                     monkeypatch
                                                                     ):
    """A/B SMOKE.  The decomposer's weight source is driven end to end on a
    TINY structure (the design's own 128-pixel cell is hours of RCWA), and the
    contract it must satisfy for the pipeline is checked: one weight per order,
    a JSON-safe provenance record, and per-order diagnostics that carry the
    coherence the averaging choice is scored on."""
    import json
    monkeypatch.setattr(PDR, '_repro_dir', lambda: str(tmp_path))
    st = _binary(n=8, period_lam=8.0)
    params = {'n_orders': 4, 'n_theta': 2, 'n_phi': 2, 'theta_max': 1e-3,
              'beam_theta_rms': 5e-4, 'max_workers': 1}
    orders, w, rec, per = PDR.rcwa_weights(params, LAM, st.period,
                                           structure=st)
    assert orders.shape[1] == 2 and len(w) == len(orders) == len(per)
    assert w.dtype == complex and np.all(np.isfinite(w))
    json.dumps(rec)                       # must be checkpointable verbatim
    json.dumps(per)
    assert rec['averaging'] == 'coherent'
    assert rec['n_doe'] == pytest.approx(st.n_doe)
    assert rec['relief_total_m'] == pytest.approx(st.relief_total)
    assert rec['worst_closure'] < BAR_CLOSURE
    assert 0.0 < rec['sum_abs_w_sq'] <= 1.0
    assert all(0.0 <= p['coherence'] <= 1.0 + 1e-12 for p in per)


def test_the_incoherent_averaging_arm_preserves_more_power_than_coherent(
        tmp_path, monkeypatch):
    """The two collapses are DIFFERENT numbers and the module says so; the
    control is that the power-preserving one never carries LESS power."""
    monkeypatch.setattr(PDR, '_repro_dir', lambda: str(tmp_path))
    st = _binary(n=8, period_lam=8.0)
    base = {'n_orders': 4, 'n_theta': 2, 'n_phi': 2, 'theta_max': 5e-2,
            'beam_theta_rms': 3e-2, 'max_workers': 1}
    _o, wc, _r, _p = PDR.rcwa_weights(base, LAM, st.period, structure=st)
    _o, wi, _r, _p = PDR.rcwa_weights(dict(base, averaging='incoherent'), LAM,
                                      st.period, structure=st)
    assert np.all(np.abs(wi) >= np.abs(wc) - 1e-12)
    assert np.all(np.isreal(wi))


def test_the_decomposer_refuses_an_unknown_averaging_mode(tmp_path,
                                                          monkeypatch):
    monkeypatch.setattr(PDR, '_repro_dir', lambda: str(tmp_path))
    st = _binary(n=8, period_lam=8.0)
    with pytest.raises(SpecError, match='averaging'):
        PDR.rcwa_weights({'averaging': 'rms', 'n_orders': 4, 'n_theta': 1,
                          'n_phi': 1, 'max_workers': 1}, LAM, st.period,
                         structure=st)


def test_the_decomposer_refuses_to_build_a_missing_table_when_told_not_to(
        tmp_path, monkeypatch):
    """A run that must not silently spend an RCWA sweep inside its decompose
    stage can say so, and the refusal names the command that builds it."""
    monkeypatch.setattr(PDR, '_repro_dir', lambda: str(tmp_path))
    st = _binary(n=8, period_lam=8.0)
    with pytest.raises(SpecError, match='has not been built'):
        PDR.rcwa_weights({'build_if_missing': False, 'n_orders': 4,
                          'n_theta': 1, 'n_phi': 1, 'max_workers': 1}, LAM,
                         st.period, structure=st)
