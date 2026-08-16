"""P2C -- the PMM2DStackHybrid cascade fast path and its priced layer caches.

Build doc: ``docs/audits/BUILD_PMM2D_CASCADE_2026_08_16.md``.

What is under test
------------------
``cascade='fast'`` (the default) does two things the pre-P2C stack did not:

* **interface dedup** -- the per-interface mode-match S-matrix is memoized on
  the (above, below) modal-content key pair.  Same W/V bytes through the same
  LAPACK, so a hit is BYTE-IDENTICAL to a rebuild.
* **identical-run merge** -- a maximal run of ADJACENT layers sharing a modal
  key collapses to one propagation of the summed thickness.  Adjacent layers
  with the same modal basis ARE one thicker layer, so this is exact physics;
  numerically it replaces ``prop(t1) * ifc(A,A) * prop(t2)`` with
  ``prop(t1+t2)``, and the departure is bounded by how far ``ifc(A,A)`` sits
  from the exact no-op swap.

plus two priced, bounded, thread-safe caches (``_geom_cache`` for the
wavelength-independent nodal build, ``_eig_cache`` for the modal eigensolve)
that replaced two bare dicts, one of which grew without bound.

Every bar below is DERIVED at runtime from the running build's own measured
quantities (TESTING_STANDARDS rule 2 / rule 5); no test here pins a number
this campaign happened to read.
"""
import threading

import numpy as np
import pytest

from lumenairy.elements.pmm._core import _interface_smatrix
from lumenairy.elements.pmm._stack2d_cache import LayerCache, cached_nbytes
from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid

WL = 1.55e-6
PX = 0.9e-6


def _cell(hi, lo, s=6, r=2):
    c = np.full((s, s), complex(lo))
    c[r:s - r, r:s - r] = complex(hi)
    return c


def _stack(layers, *, theta=0.0, phi=0.0, degree=7, n_orders=4,
           cascade="fast", **kw):
    st = PMM2DStackHybrid(PX, n_superstrate=1.0, n_substrate=1.45,
                          degree=degree, n_orders=n_orders, symmetry=False,
                          cascade=cascade, **kw)
    for t, cell in layers:
        if np.isscalar(cell):
            st.add_layer(t, eps=cell)
        else:
            st.add_layer(t, eps_cell=cell)
    st.set_source(WL, theta=theta, phi=phi)
    return st


def _maxdiff(a, b):
    return max(float(np.max(np.abs(np.asarray(x) - np.asarray(y))))
               for x, y in zip(a[1:], b[1:]))


def _identity_residual(st):
    """The running build's OWN measured departure of ``ifc(A, A)`` from the
    exact no-op swap, for the first patterned layer of ``st``.

    This is the quantity the merge perturbs by, and it is a conditioning
    readout, not a tolerance: it is ``~cond(W) cond(V) eps_mach`` for this
    layer's modal basis, so it re-derives on every build and tracks any
    legitimate change in the modal machinery.
    """
    src = st._src
    wl = float(src["wavelength"])
    k0 = 2.0 * np.pi / wl
    n_orders = st.n_orders
    ox = np.arange(-n_orders, n_orders + 1)
    order_x = np.tile(ox, len(ox))
    order_y = np.repeat(ox, len(ox))
    nre = float(np.real(np.sqrt(np.conj(complex(st.n_sup) ** 2))))
    kx0 = nre * np.sin(src["theta"]) * np.cos(src["phi"])
    ky0 = nre * np.sin(src["theta"]) * np.sin(src["phi"])
    kxv = kx0 + order_x * (wl / st.period_x)
    kyv = ky0 + order_y * (wl / st.period_y)
    modes, _keys = st._layer_mode_sets(kxv, kyv, ox, ox, kx0, ky0, k0, wl)
    _k, W, V, _lam, _t = modes[0]
    S = _interface_smatrix(W, V, W, V)
    n = W.shape[0]
    eye, zero = np.eye(n), np.zeros((n, n))
    return max(float(np.abs(S[0] - zero).max()),
               float(np.abs(S[1] - eye).max()),
               float(np.abs(S[2] - eye).max()),
               float(np.abs(S[3] - zero).max()))


def _merge_bar(st, n_layers):
    """Derived agreement bar for fast-vs-monolithic.

    ``residual`` is measured above; the cascade applies at most ``2*n+2``
    Redheffer stars and interface matches to it, and ``1e3`` decades of
    headroom cover the star denominators' amplification (which the library
    already screens and refuses at ``_guarded_inverse``).  Both sides of the
    bar are checked in
    :func:`test_p2c_merge_bar_has_a_gap_on_both_sides`.
    """
    return 1.0e3 * (2 * n_layers + 2) * _identity_residual(st)


# ===================================================================== #
# 1. the merge: exact-physics claim against an INDEPENDENT oracle
# ===================================================================== #
def test_p2c_merged_run_equals_one_explicitly_thick_layer():
    """Two adjacent IDENTICAL layers of thickness t1, t2 must give the same
    answer as ONE layer of thickness t1+t2.

    The one-layer stack is an independent oracle: it has no run to merge, so
    it exercises neither the merge nor the interface dedup, and it is what the
    merge CLAIMS the two-layer stack is.  Bar derived from the running build's
    interface-identity residual.
    """
    cell = _cell(12.0, 2.1)
    t1, t2 = 0.22e-6, 0.17e-6
    two = _stack([(t1, cell), (t2, cell)])
    one = _stack([(t1 + t2, cell)])
    r_two, r_one = two.solve(), one.solve()
    bar = _merge_bar(two, 2)
    d = _maxdiff(r_two, r_one)
    assert d < bar, (
        f"merged run {d:.3e} vs single thick layer, bar {bar:.3e} "
        f"(residual {_identity_residual(two):.3e})")


@pytest.mark.parametrize("theta,phi", [(0.0, 0.0), (0.30, 0.0), (0.30, 0.7)])
def test_p2c_fast_matches_monolithic_across_incidence(theta, phi):
    """The fast cascade and the monolithic escape hatch agree within the
    derived bar at normal, oblique and conical incidence, on a stack that
    mixes a repeated run (mergeable) with distinct layers (not)."""
    a, b = _cell(12.0, 2.1), _cell(6.5, 2.1)
    layers = [(0.20e-6, a), (0.15e-6, a), (0.18e-6, b), (0.11e-6, 2.25),
              (0.13e-6, a)]
    fast = _stack(layers, theta=theta, phi=phi)
    mono = _stack(layers, theta=theta, phi=phi, cascade="monolithic")
    d = _maxdiff(fast.solve(), mono.solve())
    bar = _merge_bar(fast, len(layers))
    assert d < bar, f"fast vs monolithic {d:.3e} exceeds derived bar {bar:.3e}"


def test_p2c_distinct_layer_stacks_are_bit_for_bit_identical():
    """With NO two adjacent layers alike there is nothing to merge, so the
    fast path must be BIT-FOR-BIT equal to the monolithic one -- the interface
    dedup alone can never move a bit (same bytes, same LAPACK).

    This is the sharp half of the identity claim: it is asserted as exact
    equality, with no tolerance to be per-build about.
    """
    layers = [(0.20e-6, _cell(12.0, 2.1)), (0.18e-6, _cell(6.5, 2.1)),
              (0.16e-6, _cell(9.0, 2.4)), (0.14e-6, 2.25 + 0.01j),
              (0.12e-6, _cell(11.0, 3.0))]
    for theta, phi in ((0.0, 0.0), (0.30, 0.7)):
        rf = _stack(layers, theta=theta, phi=phi).solve()
        rm = _stack(layers, theta=theta, phi=phi,
                    cascade="monolithic").solve()
        for x, y, nm in zip(rf[1:], rm[1:], ("R", "T", "jones")):
            assert np.array_equal(np.asarray(x), np.asarray(y)), (
                f"{nm} differs at theta={theta}, phi={phi}: "
                f"max {np.max(np.abs(np.asarray(x) - np.asarray(y))):.3e}")


def test_p2c_merge_bar_has_a_gap_on_both_sides():
    """TESTING_STANDARDS rule 5: the bar must sit decades ABOVE the noise it
    tolerates and decades BELOW the smallest real signal it must catch.

    Below: the measured fast-vs-monolithic disagreement on a mergeable stack.
    Above: the SAME stack with one layer's permittivity nudged by the smallest
    amount that is still a physical change -- a defect the bar must never
    swallow.  Both numbers are measured HERE, on the running build.
    """
    cell = _cell(12.0, 2.1)
    layers = [(0.20e-6, cell), (0.15e-6, cell), (0.18e-6, _cell(6.5, 2.1))]
    fast = _stack(layers)
    mono = _stack(layers, cascade="monolithic")
    noise = _maxdiff(fast.solve(), mono.solve())
    bar = _merge_bar(fast, len(layers))

    # smallest real signal: perturb ONE pixel of ONE layer by 1e-6 (a change
    # far below any modelling tolerance, yet unambiguously physical)
    hurt = _cell(12.0, 2.1)
    hurt[2, 2] += 1e-6
    signal = _maxdiff(
        _stack([(0.20e-6, cell), (0.15e-6, hurt),
                (0.18e-6, _cell(6.5, 2.1))]).solve(),
        mono.solve())

    assert noise < bar, f"noise {noise:.3e} not below bar {bar:.3e}"
    assert bar * 100.0 < signal, (
        f"no gap above: bar {bar:.3e}, smallest real signal {signal:.3e} "
        f"(need >= 2 decades)")
    assert noise * 100.0 < bar, (
        f"no gap below: noise {noise:.3e}, bar {bar:.3e} "
        f"(need >= 2 decades)")


# ===================================================================== #
# 2. FAIL-BEFORE: cache invalidation -- one eps entry must split the solve
# ===================================================================== #
def test_p2c_one_eps_entry_apart_must_not_share_a_modal_solve():
    """ENGINEERED INJECTOR (standards rule 3).  Two layers identical except
    for ONE entry of one eps grid must not share a cached modal set.

    Two claims, both unconditional:

    1. the DECISION -- ``_mode_key`` separates them, and the eig cache holds
       two entries after the solve, never one;
    2. the CONSEQUENCE -- the answer moves.  The nudge is scanned up a ladder
       derived from the running build's own agreement bar rather than fixed,
       because 'a fixed +delta that lands harmlessly on some builds' is the
       parametrized-injection sub-shape TESTING_STANDARDS names.  Hard-fail
       only when the ladder is exhausted.
    """
    base = _cell(12.0, 2.1)
    st_ref = _stack([(0.20e-6, base), (0.20e-6, base)])
    ref = st_ref.solve()
    bar = _merge_bar(st_ref, 2)

    # --- claim 1: the DECISION, at the key level and at the cache level ----
    nudged = base.copy()
    nudged[2, 2] += 1e-9
    st2 = _stack([(0.20e-6, base), (0.20e-6, nudged)])
    keys = [st2._mode_key(L, 1.0, 0.0, 0.0, WL) for L in st2._layers]
    assert keys[0] != keys[1], (
        "a one-entry eps difference did not change the modal content key -- "
        "the two layers would share one eigensolve")
    st2.solve()
    assert len(st2._eig_cache) == 2, (
        f"eig cache holds {len(st2._eig_cache)} entries for two DIFFERENT "
        f"layers; a one-entry eps difference was collapsed into one solve")

    # --- claim 2: the CONSEQUENCE, scanned up a derived ladder -------------
    tried = []
    for k in range(1, 12):
        delta = bar * (10.0 ** k)
        hurt = base.copy()
        hurt[2, 2] += delta
        d = _maxdiff(_stack([(0.20e-6, base), (0.20e-6, hurt)]).solve(), ref)
        tried.append((delta, d))
        if d > 100.0 * bar:
            break
    else:
        raise AssertionError(
            "ladder exhausted: no eps nudge up to "
            f"{tried[-1][0]:.3e} moved the answer more than 100x the "
            f"agreement bar {bar:.3e} -- the two layers are being served one "
            f"modal set.  Ladder: {tried}")


def test_p2c_geom_key_and_mode_key_split_on_the_source():
    """The modal key must carry the SOURCE, not only the geometry: the same
    layer at two wavelengths (or two incidences) has different modes, so a
    key that omitted them would serve a stale eigensolve across a sweep.

    Asserted as a decision on the key, and confirmed by the answer moving."""
    cell = _cell(12.0, 2.1)
    st = _stack([(0.20e-6, cell)])
    L = st._layers[0]
    k_a = st._mode_key(L, 1.0, 0.0, 0.0, WL)
    assert k_a != st._mode_key(L, 1.0, 0.0, 0.0, WL * 1.01), "wavelength"
    assert k_a != st._mode_key(L, 1.01, 0.0, 0.0, WL), "k0"
    assert k_a != st._mode_key(L, 1.0, 0.11, 0.0, WL), "kx0"
    assert k_a != st._mode_key(L, 1.0, 0.0, 0.11, WL), "ky0"

    # and the same object re-solved at a second source must not reuse the first
    st.set_source(WL)
    r1 = st.solve()
    st.set_source(WL * 1.02)
    r2 = st.solve()
    assert _maxdiff(r1, r2) > 100.0 * _merge_bar(st, 1), (
        "re-solving at a different wavelength returned (near-)identical "
        "results -- the eig cache is serving a stale modal set")


# ===================================================================== #
# 3. FAIL-BEFORE: cascade ORDER
# ===================================================================== #
def test_p2c_cascade_respects_layer_order():
    """The cascade must be order-sensitive.  An asymmetric stack reversed
    must give a DIFFERENT answer -- if dedup/merge had collapsed the sequence
    into a multiset this would silently return the same numbers.

    The two orders are also each checked against their own monolithic path, so
    'different' cannot be a fast-path artefact.
    """
    a, b, c = _cell(12.0, 2.1), _cell(6.5, 2.1), _cell(9.0, 3.0)
    fwd = [(0.20e-6, a), (0.20e-6, a), (0.14e-6, b), (0.11e-6, c)]
    rev = list(reversed(fwd))
    r_f, r_r = _stack(fwd).solve(), _stack(rev).solve()
    bar = _merge_bar(_stack(fwd), 4)
    assert _maxdiff(r_f, r_r) > 100.0 * bar, (
        "reversing an asymmetric stack did not change the answer -- the "
        "cascade lost layer order")
    for layers, res in ((fwd, r_f), (rev, r_r)):
        assert _maxdiff(res, _stack(layers,
                                    cascade="monolithic").solve()) < bar


def test_p2c_merge_only_collapses_ADJACENT_runs():
    """A-B-A must NOT merge its two A layers: they are not adjacent.  Asserted
    on the cascade sequence itself (a decision), and on the answer differing
    from the A-A-B reordering (a consequence)."""
    a, b = _cell(12.0, 2.1), _cell(6.5, 2.1)
    st_aba = _stack([(0.20e-6, a), (0.14e-6, b), (0.20e-6, a)])
    st_aab = _stack([(0.20e-6, a), (0.20e-6, a), (0.14e-6, b)])

    def _seq_len(st):
        wl = float(st._src["wavelength"])
        k0 = 2.0 * np.pi / wl
        ox = np.arange(-st.n_orders, st.n_orders + 1)
        order_x, order_y = np.tile(ox, len(ox)), np.repeat(ox, len(ox))
        kxv = order_x * (wl / st.period_x)
        kyv = order_y * (wl / st.period_y)
        modes, keys = st._layer_mode_sets(kxv, kyv, ox, ox, 0.0, 0.0, k0, wl)
        return len(st._cascade_sequence(modes, keys, True))

    assert _seq_len(st_aba) == 3, "A-B-A must stay three cascade entries"
    assert _seq_len(st_aab) == 2, "A-A-B must merge to two cascade entries"
    bar = _merge_bar(st_aba, 3)
    assert _maxdiff(st_aba.solve(), st_aab.solve()) > 100.0 * bar


# ===================================================================== #
# 4. FAIL-BEFORE: the overflow regime a transfer matrix could not survive
# ===================================================================== #
def test_p2c_thick_lossy_layer_overflows_a_transfer_matrix_but_not_the_cascade():
    """ENGINEERED REGIME (standards rule 3/4).  A transfer-matrix formulation
    carries BOTH ``exp(-lam k0 t)`` and its reciprocal; the S-matrix recursion
    carries only the decaying one.  Build a thick lossy layer whose growing
    exponential overflows float64 ON THIS BUILD -- derived from the layer's
    own measured eigenvalues, not from a guessed thickness -- and show the
    cascade still returns finite, energy-sane results.
    """
    lossy = _cell(12.0 + 4.0j, 2.1 + 0.5j)
    probe = _stack([(0.20e-6, lossy)])
    wl = float(probe._src["wavelength"])
    k0 = 2.0 * np.pi / wl
    ox = np.arange(-probe.n_orders, probe.n_orders + 1)
    order_x, order_y = np.tile(ox, len(ox)), np.repeat(ox, len(ox))
    kxv, kyv = order_x * (wl / PX), order_y * (wl / PX)
    modes, _k = probe._layer_mode_sets(kxv, kyv, ox, ox, 0.0, 0.0, k0, wl)
    lam = modes[0][3]
    decay = float(np.max(np.real(lam)))
    assert decay > 0.0, "no decaying mode -- pick a lossier cell"

    # thickness at which the GROWING exponential exp(+lam k0 t) overflows:
    # ln(float64 max) = 709.78, so t > 709.78 / (decay * k0), with a decade of
    # margin so the claim does not sit on the overflow edge.
    t_over = 10.0 * np.log(np.finfo(float).max) / (decay * k0)
    with np.errstate(over="ignore"):
        grow = np.exp(decay * k0 * t_over)
    assert not np.isfinite(grow), (
        f"engineered thickness {t_over:.3e} m did not overflow the growing "
        f"exponential (got {grow:.3e}) -- the regime was not reached")

    st = _stack([(0.12e-6, _cell(12.0, 2.1)), (t_over, lossy),
                 (0.12e-6, _cell(12.0, 2.1))])
    _o, R, T, J = st.solve()
    for name, arr in (("R", R), ("T", T), ("jones", J)):
        assert np.all(np.isfinite(np.asarray(arr))), (
            f"{name} is not finite through a layer thick enough to overflow "
            f"a transfer matrix")
    # per-POLARIZATION energy accounting: sum orders WITHIN a pol, max over
    # pols -- never sum the two polarizations (the house convention).
    per_pol = np.asarray(R).sum(axis=1) + np.asarray(T).sum(axis=1)
    assert float(np.max(per_pol)) <= 1.0 + 1e-9, (
        f"absorbing stack returned R+T = {per_pol} > 1 per polarization")
    # essentially everything is absorbed in a layer that thick
    assert float(np.max(np.asarray(T).sum(axis=1))) < 1e-12, (
        "a layer thick enough to overflow a transfer matrix transmitted "
        "measurable power")
    # and the fast path did not invent this: the monolithic hatch agrees
    st_m = _stack([(0.12e-6, _cell(12.0, 2.1)), (t_over, lossy),
                   (0.12e-6, _cell(12.0, 2.1))], cascade="monolithic")
    assert _maxdiff(st.solve(), st_m.solve()) < _merge_bar(st, 3)


# ===================================================================== #
# 5. cache PRICING -- two-sided, with the precondition FORCED
# ===================================================================== #
def test_p2c_cache_is_priced_in_when_the_budget_allows():
    """Priced IN: with a budget that comfortably holds the working set, the
    caches retain every distinct layer and refuse nothing.  The budget is SET
    by the test (rule 4: force the precondition), not inherited from the box.
    """
    layers = [(0.20e-6, _cell(12.0 + 0.5 * i, 2.1)) for i in range(4)]
    st = _stack(layers, cache_max_bytes=4 * 1024 ** 3)
    st.solve()
    s = st.cache_stats()
    assert s["eig"]["entries"] == 4, s
    assert s["geom"]["entries"] == 4, s
    assert s["eig"]["refused"] == 0 and s["geom"]["refused"] == 0, s
    assert s["eig"]["evicted"] == 0 and s["geom"]["evicted"] == 0, s
    assert 0 < s["eig"]["nbytes"] <= s["eig"]["budget"], s


def test_p2c_cache_refuses_rather_than_degrades_when_priced_out():
    """Priced OUT: with a one-byte budget the caches must REFUSE to store and
    still return the SAME answer as an unpriced run.  Refusal costs
    recomputation, never correctness -- that is the whole contract.

    Two-sided with the test above; neither arm is a skip.
    """
    layers = [(0.20e-6, _cell(12.0 + 0.5 * i, 2.1)) for i in range(4)]
    rich = _stack(layers, cache_max_bytes=4 * 1024 ** 3).solve()
    poor_st = _stack(layers, cache_max_bytes=1)
    poor = poor_st.solve()
    s = poor_st.cache_stats()
    assert s["eig"]["entries"] == 0 and s["geom"]["entries"] == 0, s
    assert s["eig"]["refused"] > 0 and s["geom"]["refused"] > 0, s
    assert s["eig"]["nbytes"] == 0, s
    for x, y, nm in zip(rich[1:], poor[1:], ("R", "T", "jones")):
        assert np.array_equal(np.asarray(x), np.asarray(y)), (
            f"{nm} moved when the cache was priced out -- a refusal degraded "
            f"the answer instead of only costing recomputation")


def test_p2c_a_single_solve_never_evicts_its_own_working_set():
    """Generation guard: entries minted by the CURRENT solve are exempt from
    eviction, so a budget too small for the whole stack causes REFUSALS, never
    the pathological evict-then-immediately-remiss cycle.  Asserted as a
    decision on the counters plus an unchanged answer."""
    layers = [(0.20e-6, _cell(12.0 + 0.5 * i, 2.1)) for i in range(5)]
    big = _stack(layers, cache_max_bytes=4 * 1024 ** 3)
    ref = big.solve()
    one_entry = big.cache_stats()["eig"]["nbytes"] // 5
    tight = _stack(layers, cache_max_bytes=int(2.5 * one_entry))
    res = tight.solve()
    s = tight.cache_stats()
    assert s["eig"]["evicted"] == 0, (
        f"a single solve evicted {s['eig']['evicted']} of its own entries")
    assert s["eig"]["refused"] > 0, s
    for x, y in zip(ref[1:], res[1:]):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_p2c_geom_cache_no_longer_grows_without_bound_over_a_sweep():
    """The defect this replaced: a dispersive sweep minted one ``_geom_cache``
    entry per wavelength in a BARE DICT with no bound, so the footprint was
    LINEAR in sweep length.

    Fail-before (measured 2026-08-16 on the pre-P2C build, n_orders=4): a
    12-point sweep retained 8.41 MB and a 24-point sweep 16.82 MB -- exactly
    2x for exactly 2x the points.  Here the budget is FORCED small, and the
    claim is build-free: doubling the sweep must NOT double the footprint, and
    the footprint must never exceed the budget.
    """
    def _sweep(npts, budget):
        st = PMM2DStackHybrid(PX, n_substrate=1.45, degree=7, n_orders=4,
                              symmetry=False, cache_max_bytes=budget)
        st.add_layer(0.22e-6,
                     eps_cell=lambda w: _cell(12.0 + 3e6 * (w - WL), 2.1))
        st.solve_vs_wavelength(np.linspace(1.50e-6, 1.60e-6, npts),
                               max_workers=1)
        return st

    a = _sweep(6, 4 * 1024 ** 3)
    per_entry = a.cache_stats()["geom"]["nbytes"] / 6.0
    assert per_entry > 0
    budget = int(3.5 * per_entry)          # room for ~3 of the 24 points
    st = _sweep(24, budget)
    s = st.cache_stats()["geom"]
    assert s["nbytes"] <= budget, (
        f"cache held {s['nbytes']} bytes against a {budget}-byte budget")
    assert s["entries"] <= 4, (
        f"cache retained {s['entries']} entries for a 24-point sweep under a "
        f"~3-entry budget -- the bound is not being applied")
    assert s["evicted"] + s["refused"] > 0, (
        "the bound never engaged; the test did not reach the regime")


def test_p2c_cache_bytes_accounting_is_measured_not_estimated():
    """``cached_nbytes`` must count each distinct buffer ONCE (a value holding
    several views of one array is not double-charged) and must match the
    cache's running total."""
    base = np.zeros((32, 32), dtype=complex)
    val = (base, base[:16], {"a": base, "b": np.zeros((8, 8), dtype=complex)})
    assert cached_nbytes(val) == base.nbytes + 8 * 8 * 16
    c = LayerCache(max_bytes=10 ** 9)
    c.new_generation()
    c.put("k", val)
    assert c.nbytes == cached_nbytes(val)
    c.clear()
    assert c.nbytes == 0 and len(c) == 0


def test_p2c_cache_budget_tracks_the_library_ram_budget():
    """The budget is priced from :func:`lumenairy.memory.get_ram_budget` AT
    QUERY TIME, so a ``set_max_ram`` override applies immediately rather than
    being frozen at construction."""
    from lumenairy.memory import get_max_ram, set_max_ram
    c = LayerCache(budget_fraction=0.5, min_budget=1)
    prev = get_max_ram()
    try:
        set_max_ram(64 * 1024 ** 3)
        assert c.budget() == int(0.5 * 64 * 1024 ** 3)
        set_max_ram(8 * 1024 ** 3)
        assert c.budget() == int(0.5 * 8 * 1024 ** 3)
    finally:
        set_max_ram(prev)


# ===================================================================== #
# 6. reuse, threading, and concurrency
# ===================================================================== #
def test_p2c_second_solve_reuses_the_modal_sets():
    """Cross-call reuse: a second solve at the SAME source must hit the eig
    cache for every layer and rebuild nothing.  Asserted on the cache's own
    hit/miss counters (a decision), not on a wall-clock reading (which is a
    per-build fact)."""
    layers = [(0.20e-6, _cell(12.0 + 0.5 * i, 2.1)) for i in range(4)]
    st = _stack(layers, cache_max_bytes=4 * 1024 ** 3)
    r1 = st.solve()
    before = st.cache_stats()["eig"]["hits"]
    r2 = st.solve()
    after = st.cache_stats()["eig"]
    assert after["hits"] - before == 4, (
        f"second solve took {after['hits'] - before} cache hits for 4 layers")
    assert after["entries"] == 4, after
    for x, y in zip(r1[1:], r2[1:]):
        assert np.array_equal(np.asarray(x), np.asarray(y)), (
            "a cached re-solve is not byte-identical to the first solve")


def test_p2c_clear_cache_forces_a_byte_identical_recompute():
    """Dropping the caches must change nothing but the work done: the same
    LAPACK on the same bytes returns the same bits."""
    layers = [(0.20e-6, _cell(12.0 + 0.5 * i, 2.1)) for i in range(3)]
    st = _stack(layers)
    r1 = st.solve()
    st.clear_cache()
    assert len(st._eig_cache) == 0 and len(st._geom_cache) == 0
    r2 = st.solve()
    for x, y in zip(r1[1:], r2[1:]):
        assert np.array_equal(np.asarray(x), np.asarray(y))


@pytest.mark.parametrize("max_workers", [2, 4, 8])
def test_p2c_threaded_layer_eig_is_byte_identical_across_worker_counts(
        max_workers):
    """Inside the threaded contract, the worker COUNT must not move a bit.

    ``max_workers=1`` and ``max_workers=N`` both enter the SAME one
    process-wide BLAS cap and both store results by key from a list yielded in
    input order, so they are byte-identical to each other for any N.

    This is deliberately NOT compared against the ``max_workers=None``
    default: that path runs at the AMBIENT BLAS thread count, and a capped run
    and an uncapped run legitimately differ in the last bits when the two
    counts differ.  Asserting across them would be exactly the
    environment-dependent shape TESTING_STANDARDS S3 names -- and it is how
    the first version of this feature shipped a real bug (an uncapped serial
    branch beside a capped pooled one), which this test caught.
    """
    layers = [(0.20e-6 + 0.01e-6 * i, _cell(12.0 + 0.5 * i, 2.1 + 0.1 * i))
              for i in range(6)]
    ser = _stack(layers).solve(max_workers=1, blas_per_worker=1)
    par = _stack(layers).solve(max_workers=max_workers, blas_per_worker=1)
    for x, y, nm in zip(ser[1:], par[1:], ("R", "T", "jones")):
        assert np.array_equal(np.asarray(x), np.asarray(y)), (
            f"{nm} differs between max_workers=1 and {max_workers}")


def test_p2c_threaded_and_default_paths_agree_physically():
    """Across the two CONTRACTS (capped fan-out vs the uncapped default) the
    guarantee weakens from bit-equality to agreement within the derived bar --
    the difference is a BLAS reduction order, not a different cascade."""
    layers = [(0.20e-6 + 0.01e-6 * i, _cell(12.0 + 0.5 * i, 2.1 + 0.1 * i))
              for i in range(6)]
    d = _maxdiff(_stack(layers).solve(),
                 _stack(layers).solve(max_workers=4, blas_per_worker=1))
    bar = _merge_bar(_stack(layers), len(layers))
    assert d < bar, f"threaded vs default {d:.3e} exceeds derived bar {bar:.3e}"


def test_p2c_caches_survive_concurrent_solves_on_one_object():
    """The caches are shared, by design, with every ``solve_vs_wavelength``
    worker (the sweep hands out ``copy.copy`` shallow clones, which share the
    cache OBJECT -- measured on the pre-P2C build, where those dicts had no
    lock).  Concurrent solves must therefore return the serial bits.
    """
    layers = [(0.20e-6 + 0.01e-6 * i, _cell(12.0 + 0.5 * i, 2.1))
              for i in range(4)]
    st = _stack(layers)
    ref = st.solve()
    import copy as _copy
    out, err = [None] * 8, []

    def _work(i):
        try:
            sub = _copy.copy(st)
            sub._internal = None
            out[i] = sub.solve()
        except Exception as e:                              # noqa: BLE001
            err.append(repr(e))

    ths = [threading.Thread(target=_work, args=(i,)) for i in range(8)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert not err, err
    for i, r in enumerate(out):
        for x, y in zip(ref[1:], r[1:]):
            assert np.array_equal(np.asarray(x), np.asarray(y)), (
                f"concurrent solve {i} disagreed with the serial answer")


def test_p2c_dispersive_sweep_answers_are_unchanged_by_the_fast_path():
    """The sweep drives ``solve`` per point on shallow clones; fast and
    monolithic must agree there too, per point."""
    def _mk(cascade):
        st = PMM2DStackHybrid(PX, n_substrate=1.45, degree=7, n_orders=4,
                              symmetry=False, cascade=cascade)
        st.add_layer(0.22e-6,
                     eps_cell=lambda w: _cell(12.0 + 3e6 * (w - WL), 2.1))
        st.add_layer(0.22e-6,
                     eps_cell=lambda w: _cell(12.0 + 3e6 * (w - WL), 2.1))
        return st

    wls = np.linspace(1.50e-6, 1.60e-6, 5)
    of, Rf, Tf = _mk("fast").solve_vs_wavelength(wls, max_workers=1)
    om, Rm, Tm = _mk("monolithic").solve_vs_wavelength(wls, max_workers=1)
    bar = _merge_bar(_stack([(0.22e-6, _cell(12.0, 2.1))]), 2)
    assert np.array_equal(of, om)
    assert float(np.max(np.abs(Rf - Rm))) < bar
    assert float(np.max(np.abs(Tf - Tm))) < bar


# ===================================================================== #
# 7. API surface
# ===================================================================== #
def test_p2c_cascade_kwarg_is_validated():
    with pytest.raises(ValueError, match="cascade must be"):
        PMM2DStackHybrid(PX, cascade="turbo")


def test_p2c_retain_internal_keeps_the_per_layer_cascade():
    """``retain_internal`` indexes the partial cascades PER LAYER, so the
    merge must be off there -- and the retained machinery must still work on a
    stack whose adjacent layers WOULD have merged."""
    cell = _cell(12.0, 2.1)
    st = _stack([(0.20e-6, cell), (0.20e-6, cell), (0.14e-6, _cell(6.5, 2.1))])
    st.solve(retain_internal=True)
    a = st.layer_absorption()
    assert a.shape == (3, 2), a.shape
    f = st.internal_field(0.1e-6, component="E", nx=8)
    assert f["Ex"].shape[-1] == 8
    # with retain_internal the fast path must equal the monolithic BIT-FOR-BIT
    # (merge off, dedup byte-identical)
    r_f = st.solve(retain_internal=True)
    r_m = _stack([(0.20e-6, cell), (0.20e-6, cell), (0.14e-6, _cell(6.5, 2.1))],
                 cascade="monolithic").solve(retain_internal=True)
    for x, y, nm in zip(r_f[1:], r_m[1:], ("R", "T", "jones")):
        assert np.array_equal(np.asarray(x), np.asarray(y)), nm
