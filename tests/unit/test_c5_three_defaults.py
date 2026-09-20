"""WP-C5 -- the three defaults the maintainer moved on 2026-09-20, and the
three opt-outs that still reproduce 5.48.x.

The decisions are ledger items 1.5 / 4.3, 1.8 and 1.7 of
``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/MAINTAINER_DECISIONS_2026_09.md``:

  1. ``carrier._GAP_KERNEL_ACCURACY_TAU = 1e-4`` -- ``gap_kernel='auto'``
     takes the paraxial kernel inside a derived band in
     ``k |z_eff| theta_env^4`` (near-focus at a fixed envelope angle,
     wide-envelope at a fixed distance -- see the round-2 section);
  2. ``gbd.DENSE_MEM_BUDGET_ACCOUNTING = 'measured'`` -- the dense
     reconstruction counts its memory honestly, and says so where the budget
     cannot be met at all;
  3. ``replica_fill='zero'`` -- a readout does not hand back periodic copies
     it never measured.

WHAT THIS FILE IS FOR, and what it deliberately is not.  Each item already has
a file that measures the BEHAVIOUR behind it
(``test_wave5_h2_near_focus_table.py``, ``test_wave5_gbd_dense_mem_budget.py``
and ``test_verify_b14_known_reds.py``, ``test_audit2609_a25_carrier_focus_readout.py``).
This file pins the DECISION: that the shipped default is the one that was
decided, that the previous behaviour is exactly one keyword or one constant
away, that the new behaviour is confined to the region it was measured on, and
-- for item 3, where the maintainer's stated concern was misapplication -- that
three specific ways of getting the confinement wrong are each caught by a
named id.

EVERY BAR HERE IS DERIVED AT RUNTIME from a quantity the running build
measures, two-sided, and premise-gated; no bar is a number copied from a
report.  Where a docstring quotes a measurement it is dated and says which
build it was taken on, and the assertion beside it is never that number.

Probes: ``validation/probe_c5_three_defaults/``.
"""
from __future__ import annotations

import inspect
import tracemalloc
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.propagators import carrier as CA
from lumenairy.propagators import gbd as G
from lumenairy.propagators.carrier import _collins_transport

# ===========================================================================
# Item 1 -- the kernel-departure switch (the band in k |z_eff| theta_env^4)
# ===========================================================================
#
# VERIFY-B4 F3's fixture: the one whose ladder actually approaches ``A = 0``,
# and therefore the only one of the campaign's two on which the rule can fire.
# Derived from the four quantities that define it; nothing below is quoted.
_F3 = dict(lam=1.064e-6, n=1024, dx=4e-6, w=0.30e-3, R=-40e-3,
           dx_out=5.6447e-06, n_out=128)


def _axis(n, d):
    return (np.arange(int(n), dtype=np.float64) - int(n) / 2) * float(d)


def _f3_env():
    x = _axis(_F3['n'], _F3['dx'])
    return np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
                  / _F3['w'] ** 2).astype(np.complex128)


def _f3_leg(dz, gap_kernel, st=None):
    """One F3 leg ``dz`` short of the geometric focus."""
    return _collins_transport(
        _f3_env(), _F3['R'], -_F3['R'] - dz, _F3['lam'], _F3['dx'], _F3['dx'],
        dx_out=_F3['dx_out'], dy_out=_F3['dx_out'], N_out_x=_F3['n_out'],
        N_out_y=_F3['n_out'], R_ref=float('inf'), gap_kernel=gap_kernel,
        on_collins_sampling='ignore', stats_out=st)


def _f3_oracle(dz):
    """The analytic Gaussian at the same plane, whole-function ``q`` form.

    ``1/q = 1/R + i lambda/(pi w^2)`` splits into the carrier and the
    amplitude with no cross term, so the input envelope IS the real Gaussian
    the fixture builds and the oracle needs nothing fitted.  It carries the
    absolute piston, so the comparison is piston-included.
    """
    lam, w, R = _F3['lam'], _F3['w'], _F3['R']
    q_in = 1.0 / (1.0 / R + 1j * lam / (np.pi * w ** 2))
    z = -R - dz
    k = 2.0 * np.pi / lam
    u = _axis(_F3['n_out'], _F3['dx_out'])
    xx, yy = np.meshgrid(u, u, indexing='xy')
    return (np.exp(1j * k * z) / (1.0 + z / q_in)
            * np.exp(1j * k * (xx ** 2 + yy ** 2) / (2.0 * (q_in + z))))


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _f3_theta_env():
    S = np.fft.fft2(np.ascontiguousarray(_f3_env(), dtype=np.complex128))
    return max(CA._collins_envelope_half_angle(S, _F3['dx'], _F3['dx'],
                                               _F3['lam']))


def test_the_kernel_switch_ships_armed_at_the_decided_tau():
    """THE DECISION.  ``tau = 1e-4`` is the value ledger 4.3 recommends and
    the maintainer took on 2026-09-20; it is an identity, not a tolerance,
    because every band this campaign quotes is that number.

    The law's own constant is checked beside it: ``sqrt(3/2)`` is the RMS
    moment of a quartic phase over a 2-D circular Gaussian, so it is exact
    arithmetic and is asserted against ``np.sqrt(1.5)`` rather than against a
    transcribed decimal.
    """
    assert CA._GAP_KERNEL_ACCURACY_TAU == 1e-4
    assert CA._QUARTIC_RMS_MOMENT == float(np.sqrt(1.5))


def test_the_rule_fires_only_inside_the_band_the_law_predicts():
    """CONFINEMENT, two-sided and against the closed form.

    The departure is linear in ``|z_eff|``, so the rule has an exact
    threshold, ``|z_eff| > 8 tau / (sqrt(3/2) k theta_env^4)``.  This id
    brackets the observed switch-over by bisection on the F3 ladder and
    compares the ``|z_eff|`` at which the resolution actually changes with
    that closed form.

    MEASURED 2026-09-20, both builds: the fixture's ``theta_env`` reads
    1.1289e-03 rad, the closed form gives 68.10 m, and the ladder switches at
    23.48 um short of focus.  The BAR is 1 % on the ratio of the two --
    nothing here is a reading: both sides are computed from the running
    build's own measurements on every call.

    PREMISE first: the ladder must actually straddle the threshold, or the
    bisection has nothing to find.
    """
    lam = _F3['lam']
    k = 2.0 * np.pi / lam
    th = _f3_theta_env()
    tau = float(CA._GAP_KERNEL_ACCURACY_TAU)
    z_predicted = 8.0 * tau / (CA._QUARTIC_RMS_MOMENT * k * th ** 4)

    def kernel_and_zeff(dz):
        st = {}
        _f3_leg(dz, 'auto', st)
        z_eff = abs(float(st['abcd'][1]) / float(st['abcd'][0]))
        return st['kernel'], z_eff

    k_near, z_near = kernel_and_zeff(1e-6)
    k_far, z_far = kernel_and_zeff(1e-3)
    assert (k_near, k_far) == ('fresnel', 'exact'), (
        f"PREMISE: the ladder does not straddle the threshold "
        f"({k_near} at 1 um, {k_far} at 1 mm); there is nothing to bracket")
    assert z_near > z_predicted > z_far, (
        f"PREMISE: the predicted threshold {z_predicted:.4g} m is not inside "
        f"the ladder's z_eff span [{z_far:.4g}, {z_near:.4g}] m")

    lo, hi = 1e-6, 1e-3                    # lo fires, hi does not
    for _ in range(40):
        mid = float(np.sqrt(lo * hi))
        if kernel_and_zeff(mid)[0] == 'fresnel':
            lo = mid
        else:
            hi = mid
    z_lo = kernel_and_zeff(lo)[1]
    z_hi = kernel_and_zeff(hi)[1]
    z_obs = float(np.sqrt(z_lo * z_hi))
    assert abs(z_obs / z_predicted - 1.0) < 1.0e-2, (
        f"the rule switches at |z_eff| = {z_obs:.6g} m against a closed-form "
        f"threshold of {z_predicted:.6g} m ({z_obs / z_predicted:.4f}x).  The "
        f"rule is keyed on something other than the law it documents.")
    # and the band expressed the way a caller reads it -- a distance to focus
    assert 1e-6 < float(np.sqrt(lo * hi)) < 1e-3


def test_the_fallback_replaces_the_departure_with_the_oracle_floor():
    """WHAT THE FLIP BUYS, against an independent oracle.

    Inside the band the exact-kernel refinement departs from the analytic
    Gaussian by the law's amount and the paraxial kernel holds the oracle's
    own floor.  The two-sided statement is that the DEFAULT now returns the
    second and not the first, and that it is bit-identical to an explicit
    ``gap_kernel='fresnel'`` there.

    MEASURED 2026-09-20, Windows py3.14 / WSL py3.12 agreeing to the printed
    digits: at 1 um short of focus the refinement reads 2.3496e-03 relative
    and the paraxial kernel 1.4568e-14, a gap of 11 decades.  The bars are
    derived from the run: the improvement must be at least 1e6, which is five
    decades under the measured 1.6e11 and five decades above a no-op.
    """
    ref = _f3_oracle(1e-6)
    auto = _f3_leg(1e-6, 'auto')
    fres = _f3_leg(1e-6, 'fresnel')
    exact = _f3_leg(1e-6, 'exact')
    r_auto, r_fres, r_exact = (_rel(auto, ref), _rel(fres, ref),
                               _rel(exact, ref))
    assert r_exact > 1e-4, (
        f"PREMISE: the exact-kernel refinement departs by only {r_exact:.3e} "
        f"here, under tau; this rung is outside the band and cannot "
        f"demonstrate the flip")
    assert r_fres / r_exact < 1e-6, (
        f"the paraxial kernel reads {r_fres:.3e} against the refinement's "
        f"{r_exact:.3e}; the gap the rule exists to close is not there")
    assert np.array_equal(np.asarray(auto), np.asarray(fres)), (
        "the default no longer returns exactly what gap_kernel='fresnel' "
        "returns on a leg inside the band")
    assert r_auto == pytest.approx(r_fres, rel=0, abs=0)


def test_an_explicit_exact_is_never_overridden_by_tau():
    """The D4 rule.  ``tau`` decides only what ``'auto'`` resolves to; a
    caller who names the exact kernel gets it, inside the band or not.
    Silently replacing a named request is the shape the ``gap_kernel``
    vocabulary gate exists to remove, and it would be worse here because the
    replacement is invisible in the returned array's shape and dtype."""
    st = {}
    _f3_leg(1e-6, 'exact', st)
    assert st['kernel'] == 'exact'
    assert st['kernel_departure'] > float(CA._GAP_KERNEL_ACCURACY_TAU), (
        "PREMISE: this rung is not over tau, so honouring 'exact' here "
        "proves nothing about the override")


def test_tau_none_is_the_opt_out_and_costs_nothing_to_take(monkeypatch):
    """THE OPT-OUT, two-sided.  With ``None`` the leg returns exactly what an
    explicit ``gap_kernel='exact'`` returns -- which is 5.48.x -- and the
    stats dict does not grow the ``'kernel_departure'`` key, so a probe that
    digests the stats dict sees no change either.

    ``test_verify_hyg2_round2.py`` asserts the stronger form (the rule's two
    functions are not CALLED at all); this id pins the consequence a caller
    cares about.
    """
    monkeypatch.setattr(CA, '_GAP_KERNEL_ACCURACY_TAU', None)
    st = {}
    out = _f3_leg(1e-6, 'auto', st)
    assert st['kernel'] == 'exact'
    assert 'kernel_departure' not in st
    assert np.array_equal(np.asarray(out),
                          np.asarray(_f3_leg(1e-6, 'exact')))


# ===========================================================================
# Item 2 -- the dense GBD memory budget
# ===========================================================================
_BUDGET_N = 256
_BUDGET_NB = 512


@pytest.fixture(scope='module')
def _bundle():
    rng = np.random.default_rng(0)
    n = _BUDGET_NB
    return G.BeamletBundle(
        positions=rng.normal(0.0, 2.0e-4, size=(n, 3)),
        directions=np.zeros((n, 3)),
        Q=np.full(n, 1.0 / (1.0e-3 - 0.02j), dtype=np.complex128),
        amplitude=(rng.normal(size=n)
                   + 1j * rng.normal(size=n)).astype(np.complex128),
        waist0=np.full(n, 1.0e-3))


@pytest.fixture
def _restore_accounting():
    old = G.DENSE_MEM_BUDGET_ACCOUNTING
    yield
    G.DENSE_MEM_BUDGET_ACCOUNTING = old


def _dense(bundle, mode, budget_mb, N=_BUDGET_N, chunk=4096, catch=False):
    """``(field, peak_bytes, messages)`` for one dense reconstruction."""
    G.DENSE_MEM_BUDGET_ACCOUNTING = mode
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            base = tracemalloc.get_traced_memory()[0]
            out = G.reconstruct_field_from_beamlets(
                bundle, Ny=N, Nx=N, dx=2.0e-6, wavelength=1.0e-6,
                chunk_beamlets=chunk, mem_budget_mb=budget_mb)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    msgs = [str(w.message) for w in rec] if catch else []
    return np.asarray(out), int(peak - base), msgs


def test_the_dense_accounting_ships_measured_and_legacy_stays_reachable():
    """THE DECISION, plus the two constants it chooses between.  The honest
    figure must stay above every reading the B14 ladder took (96.8 B per
    cell-column) and above the windowed sibling's own margin-carrying
    constant; the legacy figure must stay exactly what 5.21 through 5.48
    shipped, because that is the whole content of the word 'legacy'."""
    assert G.DENSE_MEM_BUDGET_ACCOUNTING == 'measured'
    assert G._DENSE_CELL_BYTES_LEGACY == 16.0
    assert G._DENSE_CELL_BYTES_MEASURED >= 96.8
    assert G._DENSE_CELL_BYTES_MEASURED >= G._WINDOWED_CELL_BYTES
    assert G._DENSE_MEM_BUDGET_ACCOUNTINGS == frozenset({'legacy',
                                                         'measured'})


def test_the_published_floor_is_an_upper_bound_on_the_one_column_peak(
        _bundle, _restore_accounting):
    """THE FLOOR, derived on the running build and compared with what the
    library publishes.

    The loop's peak is affine in the chunk -- ``peak/(Ny Nx) = fixed +
    c*chunk`` -- so three rungs at a budget too large to bind determine both
    terms and the third rung checks the model rather than fitting it.  The
    floor is the chunk = 1 value, the smallest transient the loop can have.

    MEASURED 2026-09-20 (both builds, 1024 beamlets): fixed 48.551 B/cell,
    c 96.000 B/cell-col, worst deviation from the affine model 5.2e-06 at
    N = 512.  ``_dense_budget_floor_bytes`` publishes ``Ny*Nx*(48 + 128)``,
    which is 1.22x that measured one-column peak.

    BARS, both derived here: the model must fit to better than 5 % (measured
    5e-06 to 1e-04, four decades of room, and a broken model misses by O(1)),
    and the published floor must sit ABOVE the measured one-column peak --
    otherwise it is not a floor -- and within 2x of it, otherwise it is a
    different quantity wearing the name.
    """
    # a discarded run first: the first tracemalloc window in a process also
    # catches whatever the import graph allocates lazily, which lands on
    # whichever chunk is measured first (measured 838 B/cell against a model
    # 144.5) and destroys the fit.
    _dense(_bundle, 'legacy', 1.0e7, chunk=1)
    rows = []
    for ch in (1, 4, 16):
        _f, peak, _m = _dense(_bundle, 'legacy', 1.0e7, chunk=ch)
        rows.append((ch, peak / float(_BUDGET_N * _BUDGET_N)))
    A = np.array([[1.0, float(ch)] for ch, _ in rows])
    y = np.array([v for _, v in rows])
    fixed, c = np.linalg.lstsq(A, y, rcond=None)[0]
    worst = float(np.max(np.abs(A @ np.array([fixed, c]) - y) / y))
    assert worst < 0.05, (
        f"the loop's peak is not affine in the chunk on this build "
        f"(worst deviation {worst:.3e} over {rows}); the floor below is "
        f"derived from that model and cannot be stated without it")
    assert c > 0.0 and fixed > 0.0

    one_col = float(_BUDGET_N * _BUDGET_N) * (fixed + c)
    published = G._dense_budget_floor_bytes(_BUDGET_N, _BUDGET_N)
    assert published >= one_col, (
        f"the published floor {published / 1e6:.3f} MB is BELOW the measured "
        f"one-column peak {one_col / 1e6:.3f} MB, so it is not a floor "
        f"(fixed {fixed:.2f} B/cell, c {c:.2f} B/cell-col)")
    assert published < 2.0 * one_col, (
        f"the published floor {published / 1e6:.3f} MB is more than twice "
        f"the measured one-column peak {one_col / 1e6:.3f} MB")
    # the helper is the ONE definition: the constants it is built from are
    # the ones the note documents, it scales with the GRID and not with the
    # beamlet count, and -- the part that matters -- it does not move with the
    # accounting switch.  A floor that read 'legacy' would sit at 4.19 MB on
    # this grid, under a loop that measurably cannot go below 9.58.
    assert published == pytest.approx(
        _BUDGET_N * _BUDGET_N * (G._DENSE_FIXED_CELL_BYTES
                                 + G._DENSE_CELL_BYTES_MEASURED), rel=0)
    assert G._dense_budget_floor_bytes(2 * _BUDGET_N, _BUDGET_N) ==         pytest.approx(2.0 * published, rel=0)
    assert G.DENSE_MEM_BUDGET_ACCOUNTING == 'legacy', (
        "PREMISE: the fit above left the switch on 'measured', so the line "
        "below cannot say the floor ignores it")
    assert G._dense_budget_floor_bytes(_BUDGET_N, _BUDGET_N) == published
    assert 'accounting' not in inspect.signature(
        G._dense_budget_floor_bytes).parameters


def test_a_budget_above_the_floor_bounds_the_loop(_bundle,
                                                  _restore_accounting):
    """THE CLAIM, over a SWEEP rather than at one cell.

    A single budget can flatter the arithmetic: the ratio peaks just after a
    chunk increment, so the honest statement is the worst ratio over a span of
    budgets above the floor.  Six multiples from 1.0x to 8.0x are measured and
    the bar is 1.0x on every one of them.

    MEASURED 2026-09-20 over ten multiples from 1.0x to 12x, 1024 beamlets:
    the ratio is worst at 1.5x the floor -- 0.917 at N = 256 and 0.911 at
    N = 512, the multiple at which the chunk has just stepped to 2 -- and
    settles to 0.75 as the budget grows.  The cross-build spread of a
    ``tracemalloc`` peak is about 7 %, so the worst cell has less headroom
    than a bar in this repository usually carries.  That is a property of the
    arithmetic and not of the test: the budget models ``chunk * 128`` B/cell
    while the loop also holds ``48`` B/cell outside the chunk, so the margin
    just after a chunk increment is what the 128-against-96 per-column margin
    has left over.  It is asserted here because it IS the claim; the SWEEP is
    what makes it a claim about the rule rather than about one lucky budget,
    and the failure message says which multiple broke.
    """
    floor = G._dense_budget_floor_bytes(_BUDGET_N, _BUDGET_N)
    _dense(_bundle, 'measured', 1.0e7, chunk=1)          # warm-up, discarded
    worst = (0.0, None)
    for mult in (1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
        mb = floor * mult / 1e6
        with warnings.catch_warnings():
            warnings.simplefilter('error')               # nothing may warn
            _f, peak, _m = _dense(_bundle, 'measured', mb)
        ratio = peak / (mb * 1e6)
        if ratio > worst[0]:
            worst = (ratio, mult)
        assert ratio < 1.0, (
            f"at {mult}x the floor ({mb:.3f} MB) the dense loop peaked at "
            f"{peak / 1e6:.3f} MB, {ratio:.3f}x the budget.  Above the floor "
            f"the budget is supposed to BOUND the loop.")
    assert worst[0] > 0.3, (
        f"PREMISE: the worst ratio over the sweep is only {worst[0]:.3f}, so "
        f"the budget is not the binding constraint anywhere on it and this "
        f"id is not measuring the chunk arithmetic")


def test_a_budget_below_the_floor_is_loud_and_names_the_floor(
        _bundle, _restore_accounting):
    """THE OTHER SIDE.  Below the floor the chunk has already bottomed out at
    one beamlet column, so no accounting constant can meet the request.  The
    old behaviour was to exceed it silently; the new one is to run and say so.

    Why a warning and not a refusal, measured: at the shipped
    ``mem_budget_mb=512`` default the floor binds from ``Ny*Nx >= 2.909e+06``
    cells -- any square grid from N = 1706 UP, the floor reading 511.6364 MB
    at N = 1705 against 512.2367 MB at N = 1706, so 1706 is the first square
    grid that binds (VERIFY-WP-C5 D4, re-measured 2026-09-20 on both builds;
    ``test_verify_c5_three_defaults.py`` derives the crossing at runtime) --
    so refusing would turn a call that completes today into a hard error on a
    DEFAULT path, and the one mitigation that keeps the grid
    (``window=5.0``) changes the returned field by its own ~1e-11 truncation
    and so cannot be applied for the caller.

    Two-sided: the notice fires below the floor and is SILENT above it (the
    sweep above runs under ``simplefilter('error')``, which is the other
    arm), it names the floor in a form the caller can act on, and the peak it
    is warning about really does exceed the budget.
    """
    floor = G._dense_budget_floor_bytes(_BUDGET_N, _BUDGET_N)
    mb = 0.5 * floor / 1e6
    _f, peak, msgs = _dense(_bundle, 'measured', mb, catch=True)
    hits = [m for m in msgs if 'mem_budget_mb' in m and 'floor' in m]
    assert len(hits) == 1, (
        f"expected exactly one floor notice at {mb:.3f} MB "
        f"(floor {floor / 1e6:.3f} MB); got {msgs}")
    msg = hits[0]
    assert ('%.6g' % (floor / 1e6)) in msg, (
        f"the notice does not quote the floor {floor / 1e6:.6g} MB: {msg}")
    assert 'window=5.0' in msg
    assert peak / (mb * 1e6) > 1.0, (
        f"PREMISE: the loop peaked at {peak / 1e6:.3f} MB against a "
        f"{mb:.3f} MB budget, i.e. it did NOT exceed it, so there was "
        f"nothing to warn about and this id proves nothing")
    # and 'legacy' says nothing: that mode never claimed to bound the loop,
    # and a caller who selects it has opted out of the claim.
    _f2, _p2, msgs2 = _dense(_bundle, 'legacy', mb, catch=True)
    assert not [m for m in msgs2 if 'floor' in m], msgs2


def test_the_two_accountings_differ_only_in_the_last_bits(
        _bundle, _restore_accounting):
    """THE COST of the flip, pinned.  The chunk boundary moves, so the order
    the per-chunk reductions are summed in moves, and floating-point addition
    is not associative.  Nothing else changes.

    MEASURED 2026-09-20: 2.1e-17 relative at N = 256 and 1.8e-18 at N = 512.
    The bar is 1e-12 -- five decades above the measurement and an UPPER bound
    on a quantity any real change of physics would push far past it.  The
    counter-pin is that the two are NOT identical: if they were, the flip
    would not have moved the chunk at all and the whole item would be a
    no-op.
    """
    a, _pa, _ma = _dense(_bundle, 'legacy', 512.0)
    c, _pc, _mc = _dense(_bundle, 'measured', 512.0)
    scale = float(np.max(np.abs(a)))
    assert scale > 0.0
    rel = float(np.max(np.abs(a - c))) / scale
    assert rel < 1.0e-12, (
        f"the two accountings differ by {rel:.3e} relative, far above the "
        f"summation-order round-off this can only be")
    assert not np.array_equal(a, c), (
        "the two accountings returned identical bytes, so the chunk did not "
        "move and the flip bought nothing on this cell")


def test_one_budget_twice_returns_the_same_bytes(_bundle,
                                                 _restore_accounting):
    """Determinism, under BOTH modes.  The chunk is a deterministic function
    of the budget and the grid, so two runs at one budget must agree to the
    bit -- the property that makes a byte-identity probe meaningful at all."""
    for mode in ('legacy', 'measured'):
        a, _p, _m = _dense(_bundle, mode, 37.0)
        b, _p2, _m2 = _dense(_bundle, mode, 37.0)
        assert np.array_equal(a, b), mode


def test_an_unknown_accounting_is_refused_by_name(_bundle,
                                                  _restore_accounting):
    """RESTATED for the new default (it read 'treated as legacy' through
    5.48.x).  While ``'legacy'`` was the default, falling through to it was
    the conservative choice.  Now it is the opposite: a typo would silently
    restore the six-fold under-count the default exists to remove.  Refused by
    name instead, the way the carrier module's ``gap_kernel`` and
    ``replica_fill`` gates refuse theirs, and the message has to name the
    knob and its whole vocabulary or a caller cannot act on it."""
    G.DENSE_MEM_BUDGET_ACCOUNTING = 'not-a-mode'
    with pytest.raises(ValueError) as ei:
        G.reconstruct_field_from_beamlets(
            _bundle, Ny=64, Nx=64, dx=2.0e-6, wavelength=1.0e-6,
            mem_budget_mb=512.0)
    msg = str(ei.value)
    assert 'DENSE_MEM_BUDGET_ACCOUNTING' in msg
    assert "'legacy'" in msg and "'measured'" in msg
    assert 'not-a-mode' in msg
    # and it is not reached when the budget arithmetic is not reached at all,
    # so a caller who never sets mem_budget_mb is not tripped by a stale
    # module attribute they did not know about.
    G.reconstruct_field_from_beamlets(
        _bundle, Ny=64, Nx=64, dx=2.0e-6, wavelength=1.0e-6,
        mem_budget_mb=0.0)


# ===========================================================================
# Item 3 -- the replica fill
# ===========================================================================
_WL3 = 1.064e-6
_R3 = -20.0e-3


def _pupil(n=512, dx=4.0e-6, w=0.5e-3):
    x = (np.arange(n) - n / 2) * dx
    E = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
               / w ** 2).astype(np.complex128)
    return E, dx, w


@pytest.fixture(scope='module')
def _readout_fixture():
    env, dx, w = _pupil()
    pd = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        la.carrier_referenced_focus_readout(
            env, _R3, -_R3, _WL3, dx,
            dx_out=(_WL3 * abs(_R3) / (np.pi * w)) / 8.0, N_out=16,
            _period_out=pd)
    return env, dx, w, min(pd['period'])


def _read(env, dx, dx_out, n_out, **kw):
    pd = {}
    kw.setdefault('on_replica', 'error')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        F = la.carrier_referenced_focus_readout(
            env, _R3, -_R3, _WL3, dx, dx_out=dx_out, N_out=n_out,
            _period_out=pd, **kw)
    return np.asarray(F), pd


def _inside_mask(period, dx_out, n, centre=(0.0, 0.0)):
    """The faithful region, from the period the TRANSFORM reported."""
    u = (np.arange(n) - n / 2.0) * float(dx_out)
    inx = np.abs(u + centre[0]) <= 0.5 * float(period[0]) * (1 + 1e-9)
    iny = np.abs(u + centre[1]) <= 0.5 * float(period[1]) * (1 + 1e-9)
    return np.logical_and(iny[:, None], inx[None, :]), inx, iny


def _assert_confined(F_rep, F_zero, period, dx_out, n, centre=(0.0, 0.0)):
    """THE PROPERTY, in one place so the mutation matrix below can aim at it.

    Under ``'zero'`` the readout must equal ``'repeat'`` EXACTLY inside one
    period of the transform's own origin and be EXACTLY zero outside it --
    nothing else, nowhere else.
    """
    mask, _inx, _iny = _inside_mask(period, dx_out, n, centre)
    assert mask.any() and (~mask).any(), (
        "PREMISE: this window is entirely inside or entirely outside one "
        "period, so it cannot demonstrate confinement")
    assert np.array_equal(F_rep[mask], F_zero[mask]), (
        "the two fills differ INSIDE one period -- the zeroing reached "
        "measurement")
    assert float(np.max(np.abs(F_zero[~mask]))) == 0.0, (
        "a sample outside one period survived the zeroing")
    assert float(np.max(np.abs(F_rep[~mask]))) > 0.0, (
        "PREMISE: 'repeat' is already zero outside the period on this "
        "fixture, so 'zero' has nothing to remove")


def test_the_replica_fill_ships_zero_on_every_readout_that_can_replicate():
    """THE DECISION, read from the signatures a caller actually calls --
    both public readouts and the private Collins readout the chain reaches
    when ``transport='collins'``.  Missing the third would leave one transport
    on the old default with nothing to say so."""
    for fn in (la.carrier_referenced_focus_readout,
               la.carrier_referenced_exact_focus_readout,
               CA._collins_focus_readout):
        assert inspect.signature(fn).parameters[
            'replica_fill'].default == 'zero', fn.__name__
    assert CA._REPLICA_FILLS == frozenset({'repeat', 'zero'})
    # and the knob is reachable from the chain and the multi entry point,
    # or 'the chain paths that reach them' would be a claim about nothing
    assert 'replica_fill' in CA._FOCUS_READOUT_KEYS
    assert 'replica_fill' in CA._OUTPUT_GRID_PASSTHROUGH


def test_a_faithful_window_is_returned_by_identity_under_the_new_default(
        _readout_fixture):
    """(a) THE WINDOW THE GUARD WOULD PASS IS UNTOUCHED.

    The region the fill governs is the exact complement of the guard's own
    condition, so it is EMPTY whenever ``2|centre_out| + N_out dx_out <=
    period``.  Asserted as object IDENTITY, not as equality: the short-circuit
    means no arithmetic runs at all, which is what makes 'every shipped
    faithful configuration is byte-identical' a structural statement rather
    than a measured one.
    """
    env, dx, w, per = _readout_fixture
    dxo = (_WL3 * abs(_R3) / (np.pi * w)) / 8.0
    n = 32
    assert n * dxo <= per, (
        f"PREMISE: the window {n * dxo:.4e} m is not inside one period "
        f"{per:.4e} m")
    F_def, pd = _read(env, dx, dxo, n)
    F_rep, _ = _read(env, dx, dxo, n, replica_fill='repeat')
    assert np.array_equal(F_def, F_rep)
    assert pd['faithful_samples'] == (n, n)
    assert CA._fill_readout_replicas(F_def, pd['period'], dxo, n,
                                     (0.0, 0.0), 'zero') is F_def


@pytest.mark.parametrize('ratio', [1.10, 1.60, 2.20])
def test_the_default_zeroes_outside_the_reported_period_and_nothing_else(
        _readout_fixture, ratio):
    """(b) CONFINEMENT, on a window that reaches past one period.

    Three window sizes: just past the bar, well past it, and past TWO periods
    (where a full-amplitude replica of the core lands inside the window and
    the argmax moves).  The boundary is taken from ``_period_out['period']``
    -- the period the transform reports from its own parameters -- and the
    faithful COUNT is checked against ``2*floor(period/2/dx_out) + 1`` of that
    reported period, so nothing here is keyed on a heuristic or a stored
    number.
    """
    env, dx, _w, per = _readout_fixture
    n = 256
    dxo = per * ratio / n
    F_rep, pd_rep = _read(env, dx, dxo, n, on_replica='ignore',
                          replica_fill='repeat')
    F_zero, pd_zero = _read(env, dx, dxo, n, on_replica='ignore',
                            replica_fill='zero')
    F_def, pd_def = _read(env, dx, dxo, n, on_replica='ignore')
    assert np.array_equal(F_def, F_zero), "the default is not 'zero'"
    assert not np.array_equal(F_def, F_rep), (
        "PREMISE: the two fills agree on this window, so it does not reach "
        "outside one period after all")
    _assert_confined(F_rep, F_zero, pd_rep['period'], dxo, n)
    # the boundary is the REPORTED period's own arithmetic
    expect = 2 * int(np.floor(0.5 * pd_rep['period'][0] / dxo
                              * (1 + 1e-9))) + 1
    assert pd_zero['faithful_samples'] == (expect, expect), (
        f"faithful_samples {pd_zero['faithful_samples']} against "
        f"2*floor(period/2/dx_out)+1 = {expect} of the reported period "
        f"{pd_rep['period'][0]:.6e} m")
    assert pd_zero['faithful_samples'] == pd_rep['faithful_samples']
    assert pd_def['faithful_samples'][0] < n


def test_the_faithful_zone_is_centred_on_the_field_not_on_the_window(
        _readout_fixture):
    """(b) again, OFF AXIS -- the case a heuristic would get wrong.

    ``E(u + period) == E(u)`` holds in ABSOLUTE output coordinates, so pushing
    ``centre_out`` off axis SPENDS the period rather than carrying it along.
    A fill keyed on the window's own centre would blank a symmetric band about
    ``centre_out`` and would pass every on-axis id in this file.
    """
    env, dx, _w, per = _readout_fixture
    n = 256
    dxo = per * 1.60 / n
    cen = (0.30 * per, 0.0)
    F_rep, pd = _read(env, dx, dxo, n, on_replica='ignore',
                      replica_fill='repeat', centre_out=cen)
    F_zero, pd0 = _read(env, dx, dxo, n, on_replica='ignore',
                        replica_fill='zero', centre_out=cen)
    _assert_confined(F_rep, F_zero, pd['period'], dxo, n, centre=cen)
    # the blanked region is NOT symmetric about the window centre: it is
    # symmetric about the field's origin, which sits 0.3 periods away.
    keep = np.abs(np.asarray(F_zero)).max(axis=0) > 0.0
    idx = np.flatnonzero(keep)
    lo, hi = int(idx[0]), int(idx[-1])
    assert (lo + hi) // 2 != n // 2, (
        f"the surviving band [{lo}, {hi}] is centred on the window "
        f"({n // 2}), not on the field's origin -- the fill is keyed on "
        f"centre_out rather than on the absolute coordinate")
    assert pd0['faithful_samples'][0] == int(keep.sum())


def test_the_result_names_the_faithful_count_and_the_fill_applied(
        _readout_fixture):
    """(c) WHAT THE RESULT SAYS.  A caller who gets an array back has to be
    able to tell how much of it is measurement and what the rest holds,
    without re-deriving the period.  Both keys are published on BOTH settings
    and on a faithful window as well as an oversized one, so neither has to be
    inferred from the other's absence."""
    env, dx, w, per = _readout_fixture
    n = 256
    for dxo, tag in ((per * 1.60 / n, 'oversized'),
                     ((_WL3 * abs(_R3) / (np.pi * w)) / 8.0, 'faithful')):
        n_out = n if tag == 'oversized' else 32
        for fill in ('repeat', 'zero', None):
            kw = {} if fill is None else {'replica_fill': fill}
            _F, pd = _read(env, dx, dxo, n_out, on_replica='ignore', **kw)
            assert pd['replica_fill'] == (fill or 'zero'), (tag, fill, pd)
            assert len(pd['faithful_samples']) == 2
            if tag == 'faithful':
                assert pd['faithful_samples'] == (n_out, n_out)


def test_the_chain_publishes_the_fill_beside_the_faithful_count():
    """(c) through the CHAIN, which is where a caller most often meets it: the
    per-stage dict has to carry the fill the readout applied beside the
    faithful count it already published, or a chain result cannot be read
    without re-running the readout."""
    from lumenairy.propagators.carrier import _publish_readout_containment
    stage = {}
    _publish_readout_containment(
        stage, {'faithful_samples': (249, 249), 'replica_fill': 'zero',
                'window_energy_frac': 0.99873})
    assert stage['readout_faithful_samples'] == (249, 249)
    assert stage['readout_replica_fill'] == 'zero'
    assert stage['readout_window_energy'] == 0.99873


def test_the_replica_refusal_is_unchanged_by_the_fill(_readout_fixture):
    """(d) THE REFUSAL IS NOT THE FILL'S BUSINESS.

    ``on_replica`` decides whether an oversized window is SERVED; the fill
    decides only what a served one contains.  Measured as two censuses over
    the same window ladder -- if any cell disagreed between the fills, the
    fill would be reaching into the decision.

    The ladder straddles the bar deliberately: 0.98 and 1.00 periods are
    served, 1.02 and up are refused, and that boundary is the guard's own
    ``2|centre_out| + N_out dx_out <= period``.
    """
    env, dx, _w, per = _readout_fixture
    n = 128
    census = {}
    for ratio in (0.50, 0.98, 1.00, 1.02, 1.60, 2.20):
        dxo = per * ratio / n
        for fill in ('repeat', 'zero'):
            try:
                _read(env, dx, dxo, n, on_replica='error',
                      replica_fill=fill)
                census[(ratio, fill)] = 'served'
            except RuntimeError as exc:
                assert 'ALIASES' in str(exc)
                census[(ratio, fill)] = 'refused'
    for ratio in (0.50, 0.98, 1.00, 1.02, 1.60, 2.20):
        assert census[(ratio, 'repeat')] == census[(ratio, 'zero')], (
            f"at {ratio}x the period the two fills disagree about whether the "
            f"window is served: {census}")
    assert census[(1.00, 'zero')] == 'served'
    assert census[(1.02, 'zero')] == 'refused'


def test_the_refusal_is_taken_before_the_fill_is_reached(_readout_fixture):
    """(f-iii) MUTATION: the fill applied on the REFUSAL path.

    If the fill ran before -- or instead of -- the guard, an oversized window
    would come back blanked rather than refused, which is the silent
    shrink-and-zero the D3 / V3 decisions were taken against.  Demonstrated
    rather than read: ``_fill_readout_replicas`` is replaced by a raising
    sentinel, and the refusal must still be the thing that comes out.
    """
    env, dx, _w, per = _readout_fixture
    n = 128
    dxo = per * 1.60 / n

    def boom(*a, **k):
        raise AssertionError('the fill ran on the refusal path')

    real = CA._fill_readout_replicas
    CA._fill_readout_replicas = boom
    try:
        with pytest.raises(RuntimeError, match='ALIASES'):
            _read(env, dx, dxo, n, on_replica='error')
    finally:
        CA._fill_readout_replicas = real
    # counter-pin: with the guard waived the sentinel IS reached, so the
    # arm above is about the ORDER and not about an unreachable call
    CA._fill_readout_replicas = boom
    try:
        with pytest.raises(AssertionError, match='refusal path'):
            _read(env, dx, dxo, n, on_replica='ignore')
    finally:
        CA._fill_readout_replicas = real


def test_zeroing_one_sample_inside_the_period_would_be_caught(
        _readout_fixture):
    """(f-i) MUTATION: the blanking reaches one sample too far.

    The most plausible way to get this wrong is an off-by-one in the
    comparison (``<`` for ``<=``, or a missing slack on a window that lands
    exactly on the period).  One extra blanked sample is invisible to a power
    budget and fatal to the contract, so the id that would catch it is
    demonstrated here on a hand-mutated array.
    """
    env, dx, _w, per = _readout_fixture
    n = 256
    dxo = per * 1.60 / n
    F_rep, pd = _read(env, dx, dxo, n, on_replica='ignore',
                      replica_fill='repeat')
    F_zero, _ = _read(env, dx, dxo, n, on_replica='ignore',
                      replica_fill='zero')
    _assert_confined(F_rep, F_zero, pd['period'], dxo, n)     # fail-after

    mask, inx, _iny = _inside_mask(pd['period'], dxo, n)
    edge = int(np.flatnonzero(inx)[-1])          # the last faithful column
    mutant = np.array(F_zero)
    mutant[:, edge] = 0.0
    assert float(np.max(np.abs(F_rep[:, edge]))) > 0.0, (
        "PREMISE: the last faithful column is already zero, so blanking it "
        "is not a mutation")
    with pytest.raises(AssertionError, match='INSIDE one period'):
        _assert_confined(F_rep, mutant, pd['period'], dxo, n)


def test_keying_the_fill_on_the_wrong_period_would_be_caught(
        _readout_fixture):
    """(f-ii) MUTATION: the fill keyed on a period that is not the
    transform's.

    The readout has more than one length that looks like a period -- the
    window ``N_out*dx_out``, the input grid's ``N*dx``, the stop-plane pitch's
    own ``N dx_stop`` -- and keying on the wrong one blanks a band of the
    wrong width while still looking like 'zeros in the wings'.  Built through
    the library's own fill function, so the mutant is a real answer the wrong
    key would have produced.
    """
    env, dx, _w, per = _readout_fixture
    n = 256
    dxo = per * 1.60 / n
    F_rep, pd = _read(env, dx, dxo, n, on_replica='ignore',
                      replica_fill='repeat')
    true_period = pd['period']
    for factor, why in ((0.5, 'half the period'), (1.5, 'half again')):
        wrong = (true_period[0] * factor, true_period[1] * factor)
        mutant = np.asarray(CA._fill_readout_replicas(
            np.array(F_rep), wrong, dxo, n, (0.0, 0.0), 'zero'))
        with pytest.raises(AssertionError):
            _assert_confined(F_rep, mutant, true_period, dxo, n)
    # and the RIGHT period passes the same helper, so the id above is about
    # the key and not about the helper being impossible to satisfy
    good = np.asarray(CA._fill_readout_replicas(
        np.array(F_rep), true_period, dxo, n, (0.0, 0.0), 'zero'))
    _assert_confined(F_rep, good, true_period, dxo, n)


def test_repeat_reproduces_the_previous_release_on_the_exact_readout():
    """The OPT-OUT on the other public readout.  The exact high-NA readout has
    the same Bluestein periodicity and its own period (the fine crop window),
    so the flip and its opt-out have to hold there too -- leaving one of the
    two readouts behind is the asymmetry that let the paraxial one ship
    without a guard in the first place."""
    n, dx, w, R = 512, 0.5e-6, 30e-6, -0.2e-3
    x = (np.arange(n) - n // 2) * dx
    r2 = x[:, None] ** 2 + x[None, :] ** 2
    S = np.sign(R) * (np.sqrt(r2 + R * R) - abs(R))
    E = (np.exp(-r2 / w ** 2)
         * np.exp(1j * 2.0 * np.pi / _WL3 * S)).astype(np.complex128)
    kw = dict(dx_out=0.05e-6, window_factor=4.0, on_replica='ignore')
    out = {}
    for fill in ('repeat', 'zero', None):
        pd = {}
        extra = {} if fill is None else {'replica_fill': fill}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out[fill] = np.asarray(la.carrier_referenced_exact_focus_readout(
                E, R, -R, _WL3, dx, N_out=3072, _period_out=pd, **kw,
                **extra))
        if fill is None:
            period, n_out = pd['period'], 3072
            assert n_out * 0.05e-6 > min(period), (
                f"PREMISE: the {n_out * 0.05e-6 * 1e6:.1f} um window fits "
                f"inside this readout's {min(period) * 1e6:.1f} um period "
                f"(the fine crop window), so there are no replicas to blank")
            assert pd['replica_fill'] == 'zero'
    assert np.array_equal(out[None], out['zero'])
    _assert_confined(out['repeat'], out['zero'], period, 0.05e-6, n_out)


# ===========================================================================
# ROUND 2 (VERIFY-WP-C5) -- the defects the verification raised, closed
# ===========================================================================
#
# Each id below closes one defect the independent verification raised.  None
# restates something already asserted above; each is the assertion that would
# have caught the defect while it was there, and each is two-sided.
# Measurements quoted in the docstrings were taken 2026-09-20 on Windows
# py3.14.6 / numpy 2.4.4 and WSL py3.12.3 / numpy 2.4.6 and agree to the
# digits printed; the bars beside them are derived from what the running
# build reads.  Probes: ``validation/probe_c5_round2/``.

#: A relay whose carrier is MISMATCHED to its beam, read out a QUARTER of the
#: way to the beam's focus.  The carrier (``fr * R0``) and the leg (``z``) are
#: fixed, so every arm below shares one ABCD and one ``z_eff``; the only thing
#: that moves is the INPUT beam's radius, which moves ``theta_env`` and
#: nothing else.  That is what separates the band from a distance.
_D6 = dict(lam=1.55e-6, n=1024, dx=6.25e-6, R0=-20.0e-3, fr=0.70,
           w_wide=0.80e-3, w_narrow=0.40e-3, nout=64)
_D6['z'] = 0.25 * abs(_D6['R0'])
_D6['dx_out'] = (_D6['lam'] * abs(_D6['R0']) / (np.pi * _D6['w_wide'])) / 8.0


def _d6_env(w):
    """``exp(-r^2/w^2)`` times the beam's own lens and the carrier's inverse,
    i.e. the RESIDUAL lens the mismatch leaves on the envelope."""
    k = 2.0 * np.pi / _D6['lam']
    g = _axis(_D6['n'], _D6['dx'])
    r2 = g[None, :] ** 2 + g[:, None] ** 2
    return (np.exp(-r2 / float(w) ** 2)
            * np.exp(1j * k * r2 / (2.0 * _D6['R0']))
            * np.exp(-1j * k * r2 / (2.0 * _D6['fr'] * _D6['R0']))).astype(
                np.complex128)


def _d6_leg(w, gap_kernel='auto', st=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(_collins_transport(
            _d6_env(w), _D6['fr'] * _D6['R0'], _D6['z'], _D6['lam'],
            _D6['dx'], _D6['dx'], dx_out=_D6['dx_out'],
            dy_out=_D6['dx_out'], N_out_x=_D6['nout'], N_out_y=_D6['nout'],
            R_ref=float('inf'), gap_kernel=gap_kernel,
            on_collins_sampling='ignore', stats_out=st))


def test_a_wide_envelope_leg_far_from_any_focus_falls_back():
    """D6.  THE RULE IS A BAND IN ``k |z_eff| theta_env^4``, NOT A DISTANCE.

    The Migration note and the CHANGELOG used to say the rule fires only for
    legs near the carrier's ``A = 0`` plane -- "within 23.5 um of the focus
    and nowhere else".  That is true of the two fixtures the law was fitted
    on and false of the rule: ``theta_env`` enters at the FOURTH power, so a
    wide envelope substitutes for a large ``z_eff``.

    This id holds the LEG fixed and moves only the envelope.  One carrier
    (``0.70 R0``), one readout plane (``z = 0.25 |R0|``), so both arms share
    ``A = 0.642857``, ``B = 5.000000e-03 m`` and ``z_eff = 7.777778e-03 m``
    to the bit and both sit 9.00 mm from that carrier's ``A = 0`` plane, with
    the beam still 48.6 (wide) / 12.2 (narrow) Rayleigh ranges short of its
    own focus.  Measured 2026-09-20 on both builds:

    ``w`` 0.80 mm: ``theta_env`` 1.715395e-02 rad, departure 4.179416e-04,
    resolves ``'fresnel'``; ``w`` 0.40 mm: ``theta_env`` 8.659722e-03 rad,
    departure 2.714408e-05, resolves ``'exact'`` -- a departure ratio of
    15.40 against an angle ratio of 1.9809, whose fourth power is 15.40.

    Nothing about the leg's geometry differs between the two rows, so a
    distance rule cannot tell them apart and this one does.  Both bars and
    the quartic are re-derived here.
    """
    tau = CA._GAP_KERNEL_ACCURACY_TAU
    assert tau is not None, (
        "PREMISE: the accuracy rule is disarmed on this tree, so there is no "
        "band to place")
    rows = {}
    for tag, w in (('wide', _D6['w_wide']), ('narrow', _D6['w_narrow'])):
        st = {}
        _d6_leg(w, 'auto', st)
        assert 'kernel_departure' in st, (
            f"the k4 gate did not resolve the {tag} arm to 'exact' "
            f"(k4={st.get('k4')}), so the accuracy rule never saw it and "
            f"this fixture cannot place the band")
        S = np.fft.fft2(np.ascontiguousarray(_d6_env(w), dtype=np.complex128))
        rows[tag] = dict(
            dep=float(st['kernel_departure']), kernel=st['kernel'],
            k4=float(st['k4']),
            theta=max(CA._collins_envelope_half_angle(
                S, _D6['dx'], _D6['dx'], _D6['lam'])))

    # (1) the leg is the SAME leg on both arms, and it is far from every
    #     focus there is -- derived from the fixture's geometry, not read
    #     back out of the transport.
    A = 1.0 + _D6['z'] / (_D6['fr'] * _D6['R0'])
    z_eff = _D6['z'] / A
    d0 = abs(_D6['z'] - (-_D6['fr'] * _D6['R0']))
    assert d0 > 1.0e-3, (
        f"the leg sits {d0 * 1e6:.1f} um from its A = 0 plane, which IS "
        f"near-focus, so this id is not demonstrating the scope it claims")
    z_R = np.pi * (_D6['lam'] * abs(_D6['R0'])
                   / (np.pi * _D6['w_wide'])) ** 2 / _D6['lam']
    assert (abs(_D6['R0']) - _D6['z']) / z_R > 20.0, (
        f"PREMISE: the readout plane is only "
        f"{(abs(_D6['R0']) - _D6['z']) / z_R:.1f} Rayleigh ranges from the "
        f"beam's own focus, so it IS near one")

    # (2) the decision, two-sided, on that one leg
    assert rows['wide']['kernel'] == 'fresnel', (
        f"the wide-envelope arm did NOT fall back: departure "
        f"{rows['wide']['dep']:.4e} against tau {tau:.1e}, theta_env "
        f"{rows['wide']['theta']:.4e} rad, z_eff {z_eff:.4e} m, "
        f"{d0 * 1e3:.2f} mm from the A = 0 plane")
    assert rows['narrow']['kernel'] == 'exact', (
        f"the NARROW-envelope arm on the SAME leg fell back too, so this "
        f"fixture no longer separates the angle from the distance: {rows}")
    assert rows['wide']['dep'] > tau > rows['narrow']['dep'], rows

    # (3) the separation is the QUARTIC, which is what makes the leg's own
    #     distance irrelevant.  The law is linear in |z_eff| and that factor
    #     is shared, so the departure ratio must be the angle ratio to the
    #     fourth -- an identity between two numbers this run measured.
    q = ((rows['wide']['theta'] / rows['narrow']['theta']) ** 4
         / (rows['wide']['dep'] / rows['narrow']['dep']))
    assert abs(q - 1.0) < 1e-6, (
        f"the two departures do not stand in the fourth-power ratio of the "
        f"two envelope angles ({q:.9f}), so they are not the same law on "
        f"the same leg: {rows}")
    assert rows['wide']['dep'] / rows['narrow']['dep'] > 4.0, (
        f"PREMISE: the two arms differ by only "
        f"{rows['wide']['dep'] / rows['narrow']['dep']:.2f}x, so this "
        f"fixture is sitting on the bar rather than either side of it")
    # (4) the REPRESENTABILITY gate sees none of it: both arms are decades
    #     under its bar of 1, which is the gap the accuracy rule closes.
    assert max(r['k4'] for r in rows.values()) < 1e-2, rows


def test_the_shipped_source_never_says_the_rule_is_off_while_it_is_armed():
    """D1 / D2 / D3.  THE COMMENTS A READER MEETS FIRST.

    Three stale statements shipped beside the armed rule: the in-function
    comment in ``_collins_transport`` and the departure law's docstring both
    said the rule was OFF BY DEFAULT, and one paragraph of
    ``_fill_readout_replicas``'s docstring began in column 0 inside an
    indented docstring, which makes ``inspect.cleandoc`` (and Sphinx) render
    every OTHER paragraph as a literal block.

    All three are properties of the shipped source, so all three are read off
    the running module rather than off a file path, and each arm carries a
    counter-arm on a synthetic stale copy -- otherwise the id would pass just
    as well if the phrase it greps for had been renamed away.

    Measured 2026-09-20 on both builds: with the column-0 paragraph present
    53 of the docstring's 55 non-blank lines come back from ``cleandoc`` with
    a four-space prefix; with it indented, 0 do.
    """
    stale = 'OFF BY DEFAULT'
    src = inspect.getsource(CA)
    tau_src = inspect.getsource(CA._collins_exact_kernel_departure)
    trans_src = inspect.getsource(CA._collins_transport)
    # the constant's own note is the run of ``#:`` lines above its assignment
    head = src.split('_GAP_KERNEL_ACCURACY_TAU = ')[0]
    note = head[head.rfind('\n\n'):]
    assert '#:' in note and 'tau' in note.lower(), (
        "PREMISE: the constant's note was not located, so arm 1 is reading "
        "the wrong text")

    def offenders(text):
        return [ln.strip() for ln in text.splitlines() if stale in ln.upper()]

    if CA._GAP_KERNEL_ACCURACY_TAU is not None:
        bad = {where: offenders(text)
               for where, text in (('the constant note', note),
                                   ('the departure law docstring', tau_src),
                                   ('_collins_transport', trans_src))
               if offenders(text)}
        assert not bad, (
            f"the shipped tau is {CA._GAP_KERNEL_ACCURACY_TAU!r}, i.e. the "
            f"rule is ARMED, and {len(bad)} place(s) beside it still say "
            f"{stale!r}: {bad}")
    # counter-arm: the grep IS discriminating.  The pre-round-2 wording put
    # back into a copy of the same text is caught.
    was = trans_src.replace('ACCURACY-KEYED FALLBACK, ARMED BY DEFAULT',
                            'ACCURACY-KEYED FALLBACK, OFF BY DEFAULT')
    assert was != trans_src, (
        "PREMISE: the comment this id watches has been reworded, so the "
        "counter-arm is not reconstructing the defect it claims")
    assert offenders(was), (
        "the grep does not catch the stale wording it was written for")

    # arm 2: the docstring renders as ONE block
    doc = inspect.cleandoc(CA._fill_readout_replicas.__doc__)
    body = [ln for ln in doc.splitlines() if ln.strip()]
    indented = [ln for ln in body if ln.startswith('    ')]
    assert not indented, (
        f"{len(indented)} of {len(body)} non-blank lines of "
        f"_fill_readout_replicas's docstring come back from cleandoc with a "
        f"four-space prefix, i.e. one paragraph sits at column 0 and the "
        f"rest render as a literal block: {indented[:2]}")
    # counter-arm: rebuild the SHAPE the defect had -- the docstring as it
    # sits in the source, four-space indented, with one paragraph left in
    # column 0 -- and check that the same reading flips.  Reconstructed from
    # the cleaned text rather than from ``__doc__``, because py3.13+ dedents
    # docstrings at COMPILE time (so ``__doc__`` is already clean on the
    # Windows build and still raw on the WSL one, and only this arm's own
    # reconstruction is the same on both).
    marker = "``'zero'`` IS THE DEFAULT"
    lines = doc.splitlines()
    hits = [k for k, ln in enumerate(lines) if ln.startswith(marker)]
    assert len(hits) == 1, (
        f"PREMISE: the paragraph this id watches was found {len(hits)} "
        f"times, so the counter-arm is not reconstructing the defect it "
        f"claims")
    raw_mut = '\n'.join(
        ln if (k == hits[0] or not ln.strip()) else '    ' + ln
        for k, ln in enumerate(lines))
    mbody = [ln for ln in inspect.cleandoc(raw_mut).splitlines() if ln.strip()]
    assert len([ln for ln in mbody
                if ln.startswith('    ')]) > 0.5 * len(mbody), (
        "the column-0 mutant does NOT make cleandoc indent the rest, so "
        "this id is not measuring what it says it is")


# -- D5: how each readout is periodic, and why the fill is right anyway -----
_D5 = dict(lam=1.55e-6, R=-25.0e-3, w=0.60e-3, n_in=512, dx_in=5.0e-6,
           n_out=256, window_periods=1.60)
_D5['period'] = _D5['lam'] * abs(_D5['R']) / _D5['dx_in']
_D5['dx_out'] = _D5['window_periods'] * _D5['period'] / _D5['n_out']
_D5['shift'] = int(round(_D5['period'] / _D5['dx_out']))


def _d5_pupil():
    x = _axis(_D5['n_in'], _D5['dx_in'])
    return np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
                  / _D5['w'] ** 2).astype(np.complex128)


def _d5_residuals(F, m):
    """Complex and modulus residuals at an integer shift of ``m`` samples,
    each divided by the array's own peak."""
    F = np.asarray(F)
    n = F.shape[1]
    sc = float(np.abs(F).max())
    A, B = F[:, m:], F[:, :n - m]
    return (float(np.max(np.abs(A - B))) / sc,
            float(np.max(np.abs(np.abs(A) - np.abs(B)))) / sc, sc)


def test_the_collins_replicas_are_aliases_up_to_a_known_phase():
    """D5.  WHAT THE FILL IS ACTUALLY JUSTIFIED BY.

    ``_fill_readout_replicas``'s docstring justified the blanking with
    "``E(u + period) == E(u)`` identically in ABSOLUTE output coordinates".
    That is true of the two readouts that finish on
    ``angular_spectrum_propagate_mft`` and NOT of ``_collins_focus_readout``,
    which the function also serves: its post-chirp ``exp(i k D x^2 / 2B)`` is
    quadratic in the absolute output coordinate, so a shift of one period
    multiplies the field by ``exp(i[2 pi u/dx_in + pi lambda z/dx_in^2])``
    instead of leaving it alone.

    Measured 2026-09-20 on both builds, on a 256-sample window at 1.60
    periods shifted by exactly one period (160 samples), 0.30 periods off
    axis: the complex residual is 6.96e-03 of the peak while the MODULUS
    residual is 2.01e-14, and the closed form holds to 7.17e-09 over 377
    sample pairs.  The paraxial readout on the same geometry reads 1.13e-13
    complex, i.e. the literal statement -- which is what says the difference
    is the Collins post-chirp and not the fixture.

    The behaviour the fill depends on is the MODULUS statement ("a replica is
    a full-amplitude image of the core"), and that is the arm with the
    decades under it.  Every bar here is a ratio between two numbers this run
    measured.
    """
    p, m, n = _D5['period'], _D5['shift'], _D5['n_out']
    assert abs(m * _D5['dx_out'] / p - 1.0) < 1e-12, (
        f"PREMISE: {m} samples is {m * _D5['dx_out'] / p:.6f} periods, not "
        f"one, so the shift below is not a replica shift")
    pd = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        F = np.asarray(CA._collins_focus_readout(
            _d5_pupil(), _D5['R'], -_D5['R'], _D5['lam'], _D5['dx_in'],
            _D5['dx_in'], dx_out=_D5['dx_out'], N_out=n,
            centre_out=(0.30 * p, 0.0), on_replica='ignore',
            on_collins_sampling='ignore', replica_fill='repeat',
            _period_out=pd))
    assert abs(min(pd['period']) / p - 1.0) < 1e-9, (
        f"PREMISE: the transport reports a period of {min(pd['period']):.6e} "
        f"m against the {p:.6e} m this fixture derives, so the shift is not "
        f"the period the library itself uses")
    c_res, a_res, peak = _d5_residuals(F, m)

    # (1) the MODULUS is periodic -- the statement the fill rests on
    assert a_res < 1e-10, (
        f"the modulus is NOT periodic at one period ({a_res:.4e} of the "
        f"peak), so 'a replica is a full-amplitude image of the core' is "
        f"not true here and the fill's premise is gone")
    # (2) the COMPLEX field is not, by six decades or more
    assert c_res / a_res > 1e6, (
        f"the complex residual {c_res:.4e} is not separated from the "
        f"modulus residual {a_res:.4e}, so this fixture does not "
        f"demonstrate the phase factor at all")
    # (3) and the phase it is off by is the CLOSED FORM
    u = _axis(n, _D5['dx_out']) + 0.30 * p
    pred = np.exp(1j * (2.0 * np.pi * u[:n - m] / _D5['dx_in']
                        + np.pi * _D5['lam'] * (-_D5['R'])
                        / _D5['dx_in'] ** 2))
    A, B = F[:, m:], F[:, :n - m]
    sel = np.abs(B) > 1e-6 * peak
    assert int(sel.sum()) > 300, (
        f"PREMISE: only {int(sel.sum())} sample pairs carry signal, so the "
        f"closed form below is checked on almost nothing")
    got = (A / np.where(np.abs(B) > 0, B, 1.0))[sel]
    err = float(np.max(np.abs(
        got - np.broadcast_to(pred[None, :], B.shape)[sel])))
    assert err < 1e-4 * c_res, (
        f"the ratio E(u+p)/E(u) is not exp(i[2 pi u/dx_in + pi lambda "
        f"z/dx_in^2]): worst error {err:.4e} over {int(sel.sum())} pairs, "
        f"against a complex residual of {c_res:.4e}")
    # (4) two-sided against the OTHER readout: the literal statement holds
    #     there, so the difference is the post-chirp and not the window
    pd2 = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        la.carrier_referenced_focus_readout(
            _d5_pupil(), _D5['R'], -_D5['R'], _D5['lam'], _D5['dx_in'],
            dx_out=1.0e-6, N_out=32, on_replica='ignore',
            on_focus_containment='ignore', _period_out=pd2)
        ps = float(min(pd2['period']))
        dxs = _D5['window_periods'] * ps / n
        Fs = np.asarray(la.carrier_referenced_focus_readout(
            _d5_pupil(), _D5['R'], -_D5['R'], _D5['lam'], _D5['dx_in'],
            dx_out=dxs, N_out=n, centre_out=(0.30 * ps, 0.0),
            on_replica='ignore', on_focus_containment='ignore',
            replica_fill='repeat', _period_out={}))
    s_res, _s_a, _s_pk = _d5_residuals(Fs, m)
    assert s_res < 1e-6 * c_res, (
        f"the paraxial readout is ALSO non-periodic in the complex field "
        f"({s_res:.4e} against the Collins readout's {c_res:.4e}), so this "
        f"id is reading a property of the window rather than of the Collins "
        f"post-chirp")


# -- D9: the refusal is taken before the fill, on ALL THREE call sites ------
def _refusal_sentinel():
    def boom(*a, **k):
        raise AssertionError('the fill ran on the refusal path')
    return boom


def test_the_refusal_precedes_the_fill_on_the_collins_readout():
    """D9.  THE SAME DEMONSTRATION ON ``_collins_focus_readout``.

    ``test_the_refusal_is_taken_before_the_fill_is_reached`` aims at a CALL
    SITE, and there are three of them, one per readout
    (``carrier.py`` 2772 / 4493 / 7235 at the branch tip).  That id exercises
    the paraxial one.  This one is the Collins site -- the readout the traced
    chain reaches on ``transport='collins'``, i.e. the route WP-C3 is about
    to make the default, whose refusal otherwise rests on a single
    pre-existing id in ``test_audit2609_b4_collins_transport.py``
    (``TestKellyGuard::test_the_period_is_the_input_grid_s_and_the_replica_guard_sees_it``).

    Same shape as the paraxial id: the fill is replaced by a raising sentinel
    and the REFUSAL must still be what comes out, with the counter-pin that
    the sentinel IS reached once the guard is waived -- so the arm is about
    the ORDER and not about an unreachable call.
    """
    p = _D5['period']
    n, dxo = 128, p * 1.60 / 128

    def call(on_replica):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return CA._collins_focus_readout(
                _d5_pupil(), _D5['R'], -_D5['R'], _D5['lam'], _D5['dx_in'],
                _D5['dx_in'], dx_out=dxo, N_out=n, on_replica=on_replica,
                on_collins_sampling='ignore')

    assert n * dxo > p, (
        f"PREMISE: the {n * dxo:.6e} m window fits inside the {p:.6e} m "
        f"period, so there is nothing to refuse")
    real = CA._fill_readout_replicas
    CA._fill_readout_replicas = _refusal_sentinel()
    try:
        with pytest.raises(RuntimeError, match='ALIASES'):
            call('error')
        with pytest.raises(AssertionError, match='refusal path'):
            call('ignore')
    finally:
        CA._fill_readout_replicas = real


def test_the_refusal_precedes_the_fill_on_the_exact_readout():
    """D9.  THE SAME DEMONSTRATION ON
    ``carrier_referenced_exact_focus_readout``.

    The third call site.  Before round 2 nothing in either C5 file caught a
    waived refusal here; the LIBRARY was covered (VERIFY-WP-C5 measured it
    per call site:
    ``test_fix_v1_v8_readout_guard_and_standoff.py::TestV3ExactReadout::test_one_period_off_the_chief_ray_is_refused``
    and
    ``test_niche_tight_focus_readout.py::test_the_exact_readout_guards_the_same_way_on_its_own_period``
    both fail on the mutant), and now this file is too.
    """
    n, dx, w, R = 512, 0.5e-6, 30e-6, -0.2e-3
    x = (np.arange(n) - n // 2) * dx
    r2 = x[:, None] ** 2 + x[None, :] ** 2
    S = np.sign(R) * (np.sqrt(r2 + R * R) - abs(R))
    E = (np.exp(-r2 / w ** 2)
         * np.exp(1j * 2.0 * np.pi / _WL3 * S)).astype(np.complex128)

    def call(on_replica, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return la.carrier_referenced_exact_focus_readout(
                E, R, -R, _WL3, dx, dx_out=0.05e-6, N_out=3072,
                window_factor=4.0, on_replica=on_replica, **kw)

    pd = {}
    call('ignore', _period_out=pd)
    assert 3072 * 0.05e-6 > min(pd['period']), (
        f"PREMISE: the {3072 * 0.05e-6 * 1e6:.1f} um window fits inside this "
        f"readout's {min(pd['period']) * 1e6:.1f} um period, so there is "
        f"nothing to refuse")
    real = CA._fill_readout_replicas
    CA._fill_readout_replicas = _refusal_sentinel()
    try:
        with pytest.raises(RuntimeError, match='ALIASES'):
            call('error')
        with pytest.raises(AssertionError, match='refusal path'):
            call('ignore')
    finally:
        CA._fill_readout_replicas = real
