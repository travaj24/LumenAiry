"""The Newton process pool must serve BOTH ``newton_fit`` backends.

Before v5.30.1 the pool only knew how to rebuild a ``RectBivariateSpline``, so
``_invert_newton_parallel`` force-returned the serial path whenever
``newton_fit='polynomial'`` -- which is the DEFAULT.  The consequences were:

  * ``n_workers`` was a silent no-op in the default configuration.  Measured:
    n_workers=1/4/8 gave 11.2 / 11.3 / 11.4 s, i.e. 0.98-0.99x -- indistinguish-
    able from noise, with no warning that the knob did nothing.
  * whether you got parallel Newton depended on a knob (``newton_fit``) that is
    about FIT ACCURACY, not about parallelism, and that most callers never set.
    Choosing the default fit silently cost the pool speed-up.

The fix teaches the worker to serve whichever fit the caller chose.  The thing
that makes it safe -- and the thing this file guards -- is that it must remain
BIT-IDENTICAL to serial for both backends: evaluation is elementwise so
chunking cannot change it, and the worker mirrors the serial path's choice of
the combined value+gradient call (using separate ``.ev`` calls instead would
reorder the floating-point work and break identity).

This list used to have a third item -- "the Chebyshev fit is a deterministic
lstsq on identical data so every worker recovers the same coefficients" -- and
that sentence was itself a defect, stated as a safety argument.  It is the
second of the two below.

Since v5.32.3, the worker also mirrors the parent's choice of
IMPLEMENTATION of that call.  ``_Cheb2DEvaluator.ev_value_and_grad`` has two --
an ``@njit`` Chebyshev recurrence and a pure-xp Vandermonde contraction -- and
the worker used to pick between them from its own numba availability, in a
fresh interpreter, with nothing in the payload to say which one the parent had
taken.  ``test_pool_result_is_bit_identical_to_serial[polynomial]`` FAILED on
CI (ubuntu, py3.10) at ``max|delta| = 1.358e-11`` for exactly that reason, and
FIX_POOL_MEMORY sec 8.1 had ledgered the mechanism as a known conditional
before the pin closed it.  The payload now carries ``cheb_backend``; a worker
that cannot honour a pinned ``'numba'`` refuses the chunk (and the parent runs
it, where the pinned backend IS the local one) rather than answering in the
other order.

That pin was necessary and NOT SUFFICIENT.  ``[polynomial]`` went on failing on
CI at 1.341e-11 / 1.358e-11 in ALL FOUR python lanes afterwards, because the
worker was still REBUILDING the fit -- and building one runs
``_solve_lstsq_thread_safe``, i.e. ``A^T A`` over a ~78 000-row design matrix,
which OpenBLAS reduces in a THREAD-COUNT-DEPENDENT order.  Two processes on
byte-identical data therefore recover coefficients differing in the last bits
whenever their BLAS widths differ (MEASURED max|dc| 4.6e-15 -> 1.370e-11 of the
field, i.e. CI's number), and a spawn worker does not inherit its parent's
width: ``threadpoolctl``'s cap is process-global on OpenBLAS, so a long-lived
parent that has passed through a capped section is not at the environment
default a fresh interpreter starts at.  Since v5.33.0 the parent SHIPS its
built coefficients and the worker evaluates them, so there is no second solve
left to agree with.  See docs/audits/FIX_POOL_REBUILD_2026_08_08.md.

So the assertions here are unconditional, which is the only form in which
"bit-identical" means anything.

The COLD-tier tests below deliberately size the grid past ``_POOL_MIN_PIXELS``
so the pool engages on the very first call -- below that threshold everything
used to run serial and a pool/serial comparison would be vacuously true.

The WARM-tier tests (``_N_WARM`` / ``_RS_WARM``, 65 536 points per group --
design 121's own per-group count at ray_subsample=4) sit deliberately BELOW the
cold bar and ABOVE the warm bar.  That band is the whole reason the threshold
was split in two, and until v5.32.2 nothing in this file entered it: review D1
found the warm tier was UNREACHABLE from a cold process, because ``warm`` was
derived from ``_PERSISTENT_POOL is not None`` and the pool is only ever created
DOWNSTREAM of the gate that reads it.  A fresh process running the 121 shape
therefore made 0 pool dispatches and ran every group serial -- byte-for-byte
the behaviour the two-tier split was written to remove.  Each warm-tier test
starts from ``close_worker_pool()`` (the one entry point that returns the
process to a genuinely cold state) and asserts the DISPATCH COUNT, so it cannot
pass by accident on a process some earlier test happened to warm.

The promotion is gated on a MEASUREMENT rather than a point count, because at
65 536 points the serial Newton step costs 0.048 s on the default backend and
0.553 s on spline -- so reachability alone measured 5-7% SLOWER.  Finding V5
then showed that measurement was stored as a bare wall time against a worker
count: two spline inversions armed the gate and the next four POLYNOMIAL ones
at the same size all dispatched, re-admitting exactly that regression, and a
7x size drop did the same.  The last block below is that finding -- the
evidence is now keyed by (worker count, fit backend, point-count band), and
each of those three legs has a state-machine test replaying the verifier's own
measured numbers plus an end-to-end chain test with the cost bar neutralised
(so the assertion is about the KEY, not about the host's throughput).
"""
import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_traced as LT

_WL = 1.31e-6
# ray_subsample=2 on a 1024 grid gives (1024/2)^2 = 262144 Newton points,
# comfortably past the 200k pool threshold.
_N, _RS = 1024, 2
# ...and 512/2 gives (512/2)^2 = 65536, which is design 121's per-group Newton
# count at N=1024/ray_subsample=4: ABOVE the 8k warm bar, BELOW the 200k cold
# bar.  Keep these two shapes distinct -- the cold-tier and warm-tier tests
# are testing opposite halves of the policy.
_N_WARM, _RS_WARM = 512, 2
_W0 = 1.0e-3
# 2 workers is enough to prove dispatch and keeps the spawn cost (and the
# test's wall time) down; the policy is about WHETHER the pool engages, not
# about how wide it is.
_NW = 2


def _skip_if_low_ram():
    try:
        import psutil
        if psutil.virtual_memory().available < 3 * (1 << 30):
            pytest.skip('insufficient free RAM for the pool test')
    except ImportError:
        pass
    # PREMISE (FIX_POOL_MEMORY_2026_08_06): every dispatch-count assertion in
    # this file assumes the RESOURCE clamp added to _invert_newton_parallel
    # does not bind, because the clamp legitimately answers a pool-sized call
    # with fewer workers -- or with serial -- on a box that cannot hold the
    # pool.  Ask the shipped resolver rather than guessing a GB threshold, so
    # this guard tracks the model instead of drifting from it.  The fit-grid
    # term is priced at the largest grid these shapes build (a few hundred
    # squared); over-estimating it only makes the guard stricter.
    if LT._newton_resolve_workers(_NW, (_N // _RS) ** 2, 600 * 600) < _NW:
        pytest.skip('the Newton pool resource clamp binds on this box, so the '
                    'dispatch-count assertions would be testing the clamp')


def _doublet(ap):
    """Cemented doublet, same shape the design battery uses."""
    surfs, before = [], 'air'
    for R, g in ((61.5e-3, 'N-BK7'), (-45.0e-3, 'N-SF5'), (-128.0e-3, 'air')):
        surfs.append({'radius': R, 'glass_before': before, 'glass_after': g,
                      'conic': 0.0, 'radius_y': None, 'conic_y': None,
                      'aspheric_coeffs': None, 'aspheric_coeffs_y': None})
        before = g
    return {'name': 'doublet', 'aperture_diameter': ap,
            'thicknesses': [4.0e-3, 2.5e-3], 'surfaces': surfs}


def _run(fit, n_workers, n=None, rs=None, n_groups=1):
    n = _N if n is None else n
    rs = _RS if rs is None else rs
    ap = 1.2 * 2.0 * _W0
    dx = float(2.2 * max(ap, 3.0 * _W0) / n)
    x = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(x, x)
    env0 = np.exp(-(X ** 2 + Y ** 2) / _W0 ** 2).astype(np.complex128)
    groups = [{'prescription': _doublet(ap), 'gap_before': 0.0}
              for _ in range(n_groups)]
    res = la.propagate_traced_carrier_chain(
        env0, groups,
        _WL, dx, r_in=np.inf, ray_subsample=rs, n_workers=n_workers,
        final_distance=0.0,
        traced_kwargs=dict(parallel_amp=False, on_undersample='silent',
                           newton_fit=fit))
    return np.asarray(res.field)


class _DispatchSpy:
    """Count pool DISPATCHES from a cold start.

    ``_get_persistent_worker_pool`` is both the only site that creates the
    pool and is called exactly once per POOLED Newton inversion, so counting
    it answers "did this call parallelise?" without timing anything.  Entering
    the context returns the process to a cold state (no live pool, no pending
    promotion) so a count of 0 means "nothing pooled", not "someone else had
    already warmed us".
    """

    def __init__(self, promote_min_seconds=0.0):
        self.calls = []
        # The COST gate is calibrated in wall-seconds against a measured
        # per-dispatch overhead, so leaving it at its shipped value would make
        # every reachability assertion below a TIMING assertion on whatever
        # box CI happens to run -- i.e. flaky by construction.  These tests
        # pin the MECHANISM: 0.0 means "assume the work is worth pooling" and
        # a huge value means "assume it never is".  The shipped default is
        # exercised by ``test_the_cost_gate_default_is_a_real_bar`` and by the
        # measurement record in docs/audits/FIX_D1_POOL_2026_08_06.md.
        self.promote_min_seconds = float(promote_min_seconds)

    def __enter__(self):
        LT.close_worker_pool()
        self._orig = LT._get_persistent_worker_pool
        self._orig_min = LT._POOL_PROMOTE_MIN_SECONDS
        LT._POOL_PROMOTE_MIN_SECONDS = self.promote_min_seconds

        def _spy(n_workers):
            self.calls.append(int(n_workers))
            return self._orig(n_workers)

        LT._get_persistent_worker_pool = _spy
        return self

    def __exit__(self, *exc):
        LT._get_persistent_worker_pool = self._orig
        LT._POOL_PROMOTE_MIN_SECONDS = self._orig_min
        LT.close_worker_pool()
        return False

    @property
    def n(self):
        return len(self.calls)


def test_the_pool_threshold_is_actually_exceeded():
    """Guard the premise: if the grid ever shrinks below the pool threshold,
    the identity tests below stop testing the pool and start passing for free.

    The threshold is two-tier since v5.30.2 -- a COLD pool must amortise Windows
    spawn (measured crossover ~200k points; at 16k a cold pool is 1.62x SLOWER),
    while a WARM one only has per-chunk pickling to cover and wins down to 1k.
    This grid clears the COLD bar, so it engages the pool either way."""
    assert _N // _RS > 0
    pts = (_N // _RS) ** 2
    assert pts >= LT._POOL_MIN_PIXELS, (
        f'{pts} Newton points is below the cold threshold '
        f'_POOL_MIN_PIXELS={LT._POOL_MIN_PIXELS}: the pool may not engage and '
        'these tests would be vacuous')
    assert LT._POOL_MIN_PIXELS_WARM <= LT._POOL_MIN_PIXELS, (
        'the warm threshold must not exceed the cold one -- a live pool has '
        'strictly less overhead left to amortise')


def test_a_warm_pool_engages_sooner_than_a_cold_one():
    """The two-tier rule is the whole point: a multi-group chain calls
    apply_real_lens_traced once per group, so only the FIRST group is cold.  A
    single threshold would keep every later group serial too -- which is what
    happened to design-121-class chains at ray_subsample=4 (65k points/group,
    below the 200k cold bar) before this split.

    NOTE (review D1): this test only checks that the two CONSTANTS still
    straddle 65 536.  That was satisfied by construction while the warm tier
    was unreachable, and it proved nothing about reachability -- see
    ``test_a_cold_multi_group_chain_reaches_the_warm_tier`` for the arm that
    actually runs a cold chain and counts dispatches."""
    assert LT._POOL_MIN_PIXELS_WARM < LT._POOL_MIN_PIXELS, (
        'warm and cold thresholds are equal, so the split does nothing')
    # 65k is the measured design-121-class per-group count at rs=4: it must sit
    # below the warm bar (so later groups parallelise) and below the cold bar
    # (so the first group does not pay a losing spawn)
    assert LT._POOL_MIN_PIXELS_WARM < 65_536 < LT._POOL_MIN_PIXELS, (
        f'65536 points no longer straddles the two thresholds '
        f'({LT._POOL_MIN_PIXELS_WARM}, {LT._POOL_MIN_PIXELS}) -- the measured '
        'cold/warm tables that justified the split need redoing')


# ---------------------------------------------------------------------------
# WARM TIER (review D1).  Everything above this line runs at 262144 points and
# never leaves the cold tier; these are the tests that enter the 8k-200k band.
# ---------------------------------------------------------------------------

def test_the_warm_tier_shape_straddles_the_two_bars():
    """Premise guard for the warm-tier tests, the same way
    ``test_the_pool_threshold_is_actually_exceeded`` guards the cold ones.

    If this shape ever drifts above the cold bar the tests below stop proving
    the warm tier is reachable and start proving the cold tier works, which is
    already covered."""
    pts = (_N_WARM // _RS_WARM) ** 2
    assert pts == 65_536
    assert LT._POOL_MIN_PIXELS_WARM <= pts < LT._POOL_MIN_PIXELS, (
        f'{pts} Newton points no longer sits between the warm bar '
        f'{LT._POOL_MIN_PIXELS_WARM} and the cold bar {LT._POOL_MIN_PIXELS}')


# The point count every state-machine test below records its samples at, and
# asks its questions at, unless it is deliberately testing the SIZE axis.
# 65 536 is design 121's per-group Newton count at ray_subsample=4.
_PTS = 65_536
# The cost class the default path lands in on a numba box.  Spelled through the
# library helper rather than hard-coded, so a rename cannot make these tests
# quietly compare two labels that are both wrong.
_CLS_POLY = LT._newton_cost_class('polynomial')
_CLS_SPLINE = LT._newton_cost_class('spline')


def test_promotion_state_machine():
    """The "has been asked" flag, exercised directly -- no chain, no timing.

    This is the piece review D1 says was missing.  ``_pool_is_warm`` ("is
    alive") could only ever become true downstream of the gate that reads it;
    the promotion flag records a fact about the WORKLOAD instead, so a process
    that has already deferred a pool-sized inversion can promote itself."""
    LT.close_worker_pool()
    slow = 10.0 * LT._POOL_PROMOTE_MIN_SECONDS + 1.0
    ns = LT._POOL_PROMOTE_MIN_SAMPLES
    try:
        assert LT._pool_reuse_is_likely(4, _CLS_SPLINE, _PTS) is False, (
            'a cold process must not report a pending promotion')
        LT._note_pool_deferral(4, _CLS_SPLINE, _PTS, slow)
        assert LT._pool_reuse_is_likely(4, _CLS_SPLINE, _PTS) is False, (
            'ONE deferred call must not promote: the first Newton inversion '
            'in a process carries one-time numba warm-up (measured 0.637 s '
            'against a 0.048 s steady state), so it is not evidence about the '
            'work the pool would actually take over')
        for _ in range(ns - 1):
            LT._note_pool_deferral(4, _CLS_SPLINE, _PTS, slow)
        assert LT._pool_reuse_is_likely(4, _CLS_SPLINE, _PTS) is True, (
            f'{ns} deferred pool-sized calls must arm the promotion')
        # promotion is per worker count: a pool built for a different count
        # would be torn down and rebuilt, so it amortises nothing
        assert LT._pool_reuse_is_likely(8, _CLS_SPLINE, _PTS) is False
        for _ in range(ns):
            LT._note_pool_deferral(8, _CLS_SPLINE, _PTS, slow)
        assert LT._pool_reuse_is_likely(8, _CLS_SPLINE, _PTS) is True
        assert LT._pool_reuse_is_likely(4, _CLS_SPLINE, _PTS) is False, (
            'recording a new worker count must RESTART the measurement, not '
            'stack alongside the old one')
        # close_worker_pool is the documented "return to cold" entry point
        LT.close_worker_pool()
        assert LT._pool_reuse_is_likely(8, _CLS_SPLINE, _PTS) is False, (
            'close_worker_pool left a promotion armed, so the process is not '
            'actually cold afterwards')
    finally:
        LT.close_worker_pool()


def test_the_cost_gate_rejects_a_cheap_newton_step():
    """Point count is not cost, and this is the arm that says so.

    At a FIXED 65 536 Newton points the serial step measured 0.048 s on the
    default polynomial fit (its Chebyshev evaluator is an
    ``@njit(parallel=True)`` kernel, so it already uses every core), 0.553 s
    on spline and 0.95 s with numba unavailable -- a 20x spread at identical
    size, against a measured ~0.22 s per-dispatch pool overhead.  A promotion
    armed on size alone therefore made the 121-shape chain 5-7% SLOWER, which
    is the same trap the campaign's own notes describe: a threshold change
    that looks like a speed-up and is not."""
    LT.close_worker_pool()
    lo = 0.5 * LT._POOL_PROMOTE_MIN_SECONDS
    hi = 2.0 * LT._POOL_PROMOTE_MIN_SECONDS
    ns = LT._POOL_PROMOTE_MIN_SAMPLES
    try:
        for _ in range(ns + 1):
            LT._note_pool_deferral(4, _CLS_SPLINE, _PTS, lo)
        assert LT._pool_reuse_is_likely(4, _CLS_SPLINE, _PTS) is False, (
            'a Newton step below the cost bar must NOT promote however often '
            'it repeats: the pool would cost more per dispatch than the step '
            'it replaces')
        LT.close_worker_pool()
        for _ in range(ns):
            LT._note_pool_deferral(4, _CLS_SPLINE, _PTS, hi)
        assert LT._pool_reuse_is_likely(4, _CLS_SPLINE, _PTS) is True
        # the estimator is the MINIMUM, so one cheap sample withdraws the
        # promotion -- deliberately conservative, since over-promoting costs
        # every remaining call while under-promoting costs only the win
        LT._note_pool_deferral(4, _CLS_SPLINE, _PTS, lo)
        assert LT._pool_reuse_is_likely(4, _CLS_SPLINE, _PTS) is False, (
            'the promotion must track the MINIMUM deferred time; taking the '
            'max or the first sample promotes on numba warm-up')
    finally:
        LT.close_worker_pool()


# ---------------------------------------------------------------------------
# FINDING V5 -- the cost gate's evidence must be keyed by what determines cost
#
# The gate above shipped storing a BARE WALL TIME against a worker count: no
# fit backend, no point count.  Verified on the shipped gate, 4 workers, one
# process per block:
#
#   2 SPLINE groups at  65 536 pts  -> 0 dispatches, armed at 0.504 s
#   then 4 POLYNOMIAL groups
#        at the SAME    65 536 pts  -> 4 dispatches   (that step is 0.048 s,
#                                                      7.3x UNDER the bar)
#   2 SPLINE groups at 116 281 pts  -> 0 dispatches, armed at 1.036 s
#   then 4 groups at    16 384 pts  -> 4 dispatches   (~0.15 s, under the bar
#                                                      AND under one dispatch)
#
# i.e. the +5-7% regression the cost gate was written to reject, re-admitted
# for any process that touches two backends or two grid sizes -- and a
# mixed-backend process is a shipped idiom (spline is the fit-domain-free
# oracle in test_niche_d7 / c6 / c8 and the C11/C12 validation scripts).
# ---------------------------------------------------------------------------

def test_the_cost_class_separates_the_three_measured_backends():
    """The label the evidence is keyed by must actually separate the three
    regimes the measurement table distinguishes (0.048 / 0.553 / 0.95 s at a
    FIXED 65 536 points).  Two of them are both ``newton_fit='polynomial'``,
    so keying on ``newton_fit`` alone would merge a 20x cost difference."""
    orig = LT._NUMBA_AVAILABLE
    try:
        LT._NUMBA_AVAILABLE = True
        poly_numba = LT._newton_cost_class('polynomial')
        spline = LT._newton_cost_class('spline')
        LT._NUMBA_AVAILABLE = False
        poly_plain = LT._newton_cost_class('polynomial')
    finally:
        LT._NUMBA_AVAILABLE = orig
    assert len({poly_numba, spline, poly_plain}) == 3, (
        f'cost classes {poly_numba!r} / {spline!r} / {poly_plain!r} do not '
        f'separate the three measured regimes; a sample from one would '
        f'promote another')
    # numba availability is read at CALL time, not baked in at import, so a
    # process that loses the kernel lands in its own bucket instead of
    # inheriting the fast path's 0.048 s measurement
    expected = poly_numba if orig else poly_plain
    assert LT._newton_cost_class('polynomial') == expected
    # an unknown backend gets a bucket of its own rather than inheriting one
    assert LT._newton_cost_class('some_future_fit') not in (
        poly_numba, spline, poly_plain)


def test_a_spline_measurement_does_not_promote_the_polynomial_path():
    """FAIL-BEFORE (V5), state-machine half -- replayed with the verifier's own
    measured numbers so the assertion is arithmetic, not throughput.

    Two spline inversions at 0.504 s clear the 0.35 s bar.  The polynomial
    inversion that follows them, at the SAME 65 536 points, costs 0.048 s --
    so pooling it is a ~4.6x loss against the ~0.22 s dispatch.  Pre-fix the
    stored 0.504 s answered for it."""
    LT.close_worker_pool()
    try:
        for _ in range(LT._POOL_PROMOTE_MIN_SAMPLES):
            LT._note_pool_deferral(4, _CLS_SPLINE, _PTS, 0.504)
        assert LT._pool_reuse_is_likely(4, _CLS_SPLINE, _PTS) is True, (
            'the spline measurement must still arm its OWN class')
        assert LT._pool_reuse_is_likely(4, _CLS_POLY, _PTS) is False, (
            "a spline inversion's wall time promoted the polynomial path at "
            "the same point count; that path measures 0.048 s, 7.3x under "
            "the bar, and pooling it costs 5-7% (the regression the cost "
            "gate exists to reject)")
        # ...and the polynomial class then has to earn it from scratch: its
        # own cheap samples never arm, however many arrive
        for _ in range(LT._POOL_PROMOTE_MIN_SAMPLES + 2):
            LT._note_pool_deferral(4, _CLS_POLY, _PTS, 0.048)
        assert LT._pool_reuse_is_likely(4, _CLS_POLY, _PTS) is False
        # and recording the polynomial samples must not leave the spline
        # class armed on a bucket it no longer owns
        assert LT._pool_reuse_is_likely(4, _CLS_SPLINE, _PTS) is False, (
            'the pending measurement must belong to exactly one cost class')
    finally:
        LT.close_worker_pool()


def test_a_large_measurement_does_not_promote_a_much_smaller_call():
    """FAIL-BEFORE (V5), size half -- again with the verifier's numbers.

    1.036 s measured at 116 281 points says nothing about a 16 384-point call
    (7.1x fewer points, ~0.15 s, under BOTH the 0.35 s bar and the ~0.22 s
    dispatch cost).  The warm band spans 8 000-200 000 points, a 25x range, so
    "a pool-sized inversion was slow" is not a fact about every call in it."""
    LT.close_worker_pool()
    big, small = 116_281, 16_384
    r = LT._POOL_PROMOTE_SIZE_RATIO
    inside = 1.0 + 0.5 * (r - 1.0)          # strictly inside the band
    try:
        for _ in range(LT._POOL_PROMOTE_MIN_SAMPLES):
            LT._note_pool_deferral(4, _CLS_SPLINE, big, 1.036)
        assert LT._pool_reuse_is_likely(4, _CLS_SPLINE, big) is True
        assert LT._pool_reuse_is_likely(4, _CLS_SPLINE, small) is False, (
            f'a measurement taken at {big} points promoted a {small}-point '
            f'call ({big / small:.1f}x smaller), extrapolating a wall time '
            f'down without a size term')
        # the band is symmetric: a measurement does not answer for a call
        # several times LARGER either
        assert LT._pool_reuse_is_likely(
            4, _CLS_SPLINE, int(big * (r + 1.0))) is False
        # inside the band the sample IS reused, scaled to the query size --
        # otherwise the fix would just be "never promote"
        assert LT._pool_reuse_is_likely(
            4, _CLS_SPLINE, int(big / inside)) is True
    finally:
        LT.close_worker_pool()


def test_within_the_band_the_estimate_follows_the_point_count():
    """The band is not a second on/off switch: inside it the recorded sample
    is SCALED to the query size before it meets the bar.  So a sample that
    only just clears the bar stops clearing it once the call shrinks, even
    though both sizes are in the same band.

    Both the size probe and the margin are derived from the shipped ratio, so
    this stays a test of the RULE rather than of the constant."""
    LT.close_worker_pool()
    r = LT._POOL_PROMOTE_SIZE_RATIO
    assert r > 1.0
    f = 1.0 + 0.5 * (r - 1.0)               # strictly inside the band
    pts = 100_000
    # clears the bar by (1+f)/2 at ``pts``, which is less than f -- so the
    # same sample lands BELOW the bar once scaled down to pts / f
    secs = 0.5 * (1.0 + f) * LT._POOL_PROMOTE_MIN_SECONDS
    try:
        for _ in range(LT._POOL_PROMOTE_MIN_SAMPLES):
            LT._note_pool_deferral(4, _CLS_SPLINE, pts, secs)
        assert LT._pool_reuse_is_likely(4, _CLS_SPLINE, pts) is True
        assert LT._pool_reuse_is_likely(4, _CLS_SPLINE,
                                        int(pts / f)) is False, (
            'the recorded sample must be scaled to the query point count, '
            'not applied flat across the whole band')
        assert LT._pool_reuse_is_likely(4, _CLS_SPLINE,
                                        int(pts * f)) is True
    finally:
        LT.close_worker_pool()


def test_the_size_band_is_a_bounded_extrapolation():
    """Pin the bound itself.  Linear-in-points is a reasonable LOCAL model of
    the Newton step and a terrible 25x one (there is a fixed per-call
    fit/setup cost), so the scaling is only ever applied inside a factor of
    ``_POOL_PROMOTE_SIZE_RATIO``.  A ratio of 1 would make every size change
    re-measure (safe but wasteful); a huge one restores the defect."""
    assert 1.0 < LT._POOL_PROMOTE_SIZE_RATIO <= 4.0, (
        f'_POOL_PROMOTE_SIZE_RATIO={LT._POOL_PROMOTE_SIZE_RATIO} is outside '
        'the range docs/audits/FIX_V4_V5_2026_08_06.md justifies')
    r = LT._POOL_PROMOTE_SIZE_RATIO
    assert LT._pool_size_band_ok(1000, 1000) is True
    assert LT._pool_size_band_ok(int(1000 * r * 0.99), 1000) is True
    assert LT._pool_size_band_ok(int(1000 * r * 1.01) + 1, 1000) is False
    assert LT._pool_size_band_ok(1000, int(1000 * r * 1.01) + 1) is False
    # a cold record (anchor 0) can never satisfy the band, so it cannot be
    # mistaken for "any size matches"
    assert LT._pool_size_band_ok(1000, 0) is False


def test_the_cost_gate_default_is_a_real_bar():
    """The shipped constant must stay above the per-dispatch overhead it
    exists to clear.  0 would make the pool a pure tax on the default path;
    a huge value would make the warm tier unreachable again by another
    route."""
    assert 0.05 <= LT._POOL_PROMOTE_MIN_SECONDS <= 5.0, (
        f'_POOL_PROMOTE_MIN_SECONDS={LT._POOL_PROMOTE_MIN_SECONDS} is outside '
        'the range the measurements in docs/audits/FIX_D1_POOL_2026_08_06.md '
        'support (measured per-dispatch overhead ~0.22 s at 8 workers)')


def test_a_promotable_chain_does_not_pool_when_the_step_is_cheap():
    """End-to-end counterpart to the state-machine test: with the cost gate
    set unreachably high, a cold multi-group chain at the 121 shape must stay
    fully serial no matter how many groups it has."""
    _skip_if_low_ram()
    with _DispatchSpy(promote_min_seconds=1.0e9) as spy:
        f = _run('polynomial', _NW, n=_N_WARM, rs=_RS_WARM, n_groups=3)
        created = LT._PERSISTENT_POOL is not None
    assert np.isfinite(f).all()
    assert spy.n == 0, (
        f'{spy.n} dispatches with the cost gate closed; the promotion must be '
        f'gated on the MEASURED serial Newton time, not on point count alone')
    assert not created


def test_a_cold_multi_group_chain_reaches_the_warm_tier():
    """FAIL-BEFORE test for review D1.

    Runtime proof from the review, on the pre-fix code: a fresh process
    running 3 groups of 65 536 Newton points made **0** pool dispatches and
    never created a pool, because the only pool-creation site sits downstream
    of the gate that asks whether a pool exists.  Post-fix, and given work the
    cost gate deems worth pooling, the first group still runs serial (a
    one-shot must not pay a spawn it cannot amortise -- cold 16k measured
    1.62x SLOWER pooled) and every group after it pools.

    Asserting the exact count, not ``> 0``, is deliberate: it pins BOTH halves
    of the policy at once.  The cost gate is neutralised here so this stays a
    test of REACHABILITY rather than of the host's Newton throughput -- see
    ``_DispatchSpy`` and ``test_the_cost_gate_rejects_a_cheap_newton_step``."""
    _skip_if_low_ram()
    n_groups = 4
    n_serial = LT._POOL_PROMOTE_MIN_SAMPLES
    with _DispatchSpy() as spy:
        f = _run('polynomial', _NW, n=_N_WARM, rs=_RS_WARM, n_groups=n_groups)
    assert np.isfinite(f).all()
    assert spy.n == n_groups - n_serial, (
        f'{spy.n} pool dispatches across {n_groups} groups of '
        f'{(_N_WARM // _RS_WARM) ** 2} Newton points; expected '
        f'{n_groups - n_serial} (the first {n_serial} groups measure the '
        f'serial step, the rest pool).  0 means the warm tier is unreachable '
        f'from a cold process again (review D1); {n_groups} means a one-shot '
        f'call now pays a cold spawn.')


def test_a_spline_chain_does_not_promote_a_following_polynomial_chain():
    """END-TO-END fail-before for V5, on the sequence the verifier ran.

    The COST BAR is neutralised here (``_DispatchSpy`` default 0.0), which is
    what makes this a test of the KEYING rather than of the host's Newton
    throughput: with the bar at zero every deferral counts as "worth
    pooling", so the ONLY thing that can keep the polynomial groups off the
    pool is the fact that the evidence on record belongs to a different cost
    class.  Pre-fix, all of them dispatched on the spline phase's wall time;
    now the polynomial phase re-measures from scratch like any cold class,
    and only pools once it has its OWN ``_POOL_PROMOTE_MIN_SAMPLES``.

    At the SHIPPED bar the polynomial phase makes ZERO dispatches (its step is
    0.048 s against a 0.35 s bar) -- that arm is a timing statement about the
    box, so it lives in docs/audits/FIX_V4_V5_2026_08_06.md as a measurement,
    not here as an assertion."""
    _skip_if_low_ram()
    ns = LT._POOL_PROMOTE_MIN_SAMPLES
    n_poly = ns + 2
    kw = dict(n=_N_WARM, rs=_RS_WARM)
    with _DispatchSpy() as spy:
        _run('spline', _NW, n_groups=ns, **kw)
        n_spline = spy.n
        f = _run('polynomial', _NW, n_groups=n_poly, **kw)
        n_poly_disp = spy.n - n_spline
    assert np.isfinite(f).all()
    assert n_spline == 0, (
        f'the {ns} arming spline groups made {n_spline} dispatches; they are '
        f'supposed to MEASURE, not pool')
    assert n_poly_disp == n_poly - ns, (
        f'{n_poly_disp} of {n_poly} polynomial groups dispatched at the same '
        f'{(_N_WARM // _RS_WARM) ** 2} points after a SPLINE measurement '
        f'armed the gate; expected {n_poly - ns} (the polynomial class must '
        f'earn its own promotion).  {n_poly} means the spline wall time is '
        f'still answering for a backend that is 10x cheaper -- the +5-7% '
        f'regression the cost gate exists to reject.')


def test_a_large_chain_does_not_promote_a_following_small_chain():
    """END-TO-END fail-before for V5's size half, same backend throughout so
    only the point count changes: 65 536 points, then 16 384 (4x fewer, both
    inside the 8k-200k warm band).  Cost bar neutralised for the same reason
    as above."""
    _skip_if_low_ram()
    ns = LT._POOL_PROMOTE_MIN_SAMPLES
    n_small = ns + 2
    with _DispatchSpy() as spy:
        _run('polynomial', _NW, n=_N_WARM, rs=_RS_WARM, n_groups=ns)
        n_big = spy.n
        f = _run('polynomial', _NW, n=256, rs=2, n_groups=n_small)
        n_small_disp = spy.n - n_big
    assert np.isfinite(f).all()
    assert (256 // 2) ** 2 >= LT._POOL_MIN_PIXELS_WARM, (
        'the small phase dropped below the warm bar, so it would run serial '
        'for a reason that has nothing to do with the size band')
    assert n_big == 0
    assert n_small_disp == n_small - ns, (
        f'{n_small_disp} of {n_small} groups at {(256 // 2) ** 2} points '
        f'dispatched on a measurement taken at {(_N_WARM // _RS_WARM) ** 2} '
        f'points; expected {n_small - ns} (the smaller size band must earn '
        f'its own promotion)')


def test_a_single_sub_cold_bar_call_does_not_create_a_pool():
    """The other half of the policy, and the reason the fix is a promotion
    rather than simply lowering ``_POOL_MIN_PIXELS``.

    A one-shot call in the 8k-200k band must stay serial: the measured cold
    table has 16 384 points at 0.62x (a 1.6x SLOWDOWN) and 65 536 at 0.86x
    once the spawn is charged to the single call that pays it.  If someone
    "fixes" reachability by dropping the cold bar to the warm bar, this test
    fails."""
    _skip_if_low_ram()
    with _DispatchSpy() as spy:
        f = _run('polynomial', _NW, n=_N_WARM, rs=_RS_WARM, n_groups=1)
        created = LT._PERSISTENT_POOL is not None
    assert np.isfinite(f).all()
    assert spy.n == 0, (
        f'a single {(_N_WARM // _RS_WARM) ** 2}-point call dispatched to the '
        f'pool {spy.n} times; it must run serial and merely ARM the promotion')
    assert not created, 'a one-shot sub-cold-bar call created a worker pool'


@pytest.mark.parametrize('fit', ['polynomial', 'spline'])
def test_warm_tier_pool_result_is_bit_identical_to_serial(fit):
    """Bit-identity on the tier the fix newly makes reachable.

    The cold-tier identity test below has always run at 262144 points; nothing
    checked identity for a chain that reaches the pool by PROMOTION, where
    group 1 comes from the serial closure and groups 2+ come from the workers
    -- i.e. where a single output field is assembled from both paths.  The
    dispatch-count assertion is the liveness check: without it a regression
    that stopped pooling would make this test pass for free."""
    _skip_if_low_ram()
    n_groups = LT._POOL_PROMOTE_MIN_SAMPLES + 1
    kw = dict(n=_N_WARM, rs=_RS_WARM, n_groups=n_groups)
    with _DispatchSpy():
        serial = _run(fit, 1, **kw)
    with _DispatchSpy() as spy:
        pooled = _run(fit, _NW, **kw)
    assert spy.n == 1, (
        f'the pooled run made {spy.n} dispatches, so this comparison does not '
        f'exercise the warm tier')
    assert np.array_equal(serial, pooled), (
        f'newton_fit={fit!r}: warm-tier pool result differs from serial, '
        f'max|delta| = {np.abs(pooled - serial).max():.3e}')


def test_the_gate_consults_the_promotion_flag_not_only_the_live_pool():
    """Pin the MECHANISM, the way
    ``test_polynomial_is_no_longer_force_routed_to_serial`` pins the other
    defect: review D1's root cause was a gate whose only warmth signal was
    ``_PERSISTENT_POOL is not None``, a value that cannot become true before
    the gate runs.  A refactor that drops the promotion arm would restore the
    unreachable tier, and only the (slow) chain tests above would notice."""
    import inspect
    src = inspect.getsource(LT.apply_real_lens_traced)
    i = src.find('def _invert_newton_parallel')
    assert i != -1, 'could not locate _invert_newton_parallel'
    body = src[i:i + 6000]
    assert '_pool_reuse_is_likely' in body, (
        'the warm gate no longer consults the promotion flag: '
        '_PERSISTENT_POOL is not None can only become true downstream of this '
        'gate, so the warm tier would be unreachable from a cold process')
    assert '_note_pool_deferral' in body, (
        'nothing arms the promotion any more, so the second call cannot be '
        'promoted')
    # ...and the evidence must be keyed by what determines cost (V5).  Both
    # helpers take the cost class and the point count as REQUIRED arguments,
    # so a call site cannot be backend- or size-blind by omission -- but a
    # refactor could still pass a constant, so pin that the gate derives the
    # class from the resolved newton_fit and hands over this call's own size.
    assert '_newton_cost_class(newton_fit)' in body, (
        'the promotion gate no longer derives a cost class from the resolved '
        'newton_fit: a spline measurement would promote the 10x cheaper '
        'polynomial path again (finding V5)')
    for _call in ('_pool_reuse_is_likely(n_cpu, _cost_class, n_total)',
                  '_note_pool_deferral(n_cpu, _cost_class, n_total,'):
        assert _call in body, (
            f'{_call!r} is gone: the deferral evidence is no longer keyed by '
            f'this call\'s own backend and point count')


@pytest.mark.parametrize('fit', ['polynomial', 'spline'])
def test_pool_result_is_bit_identical_to_serial(fit):
    """The whole safety argument for parallel Newton: identical bits, not
    merely 'close'.  Runs for the DEFAULT polynomial backend too, which the
    pool refused to serve before v5.30.1."""
    _skip_if_low_ram()
    serial = _run(fit, 1)
    pooled = _run(fit, 4)
    assert np.array_equal(serial, pooled), (
        f'newton_fit={fit!r}: pool result differs from serial, '
        f'max|delta| = {np.abs(pooled - serial).max():.3e}')


def test_polynomial_is_no_longer_force_routed_to_serial():
    """Pin the specific defect: ``_invert_newton_parallel`` used to contain an
    early ``if newton_fit == 'polynomial': return _invert_newton(...)``.  A
    future refactor that reinstates it would silently halve throughput on the
    default path, and the bit-identity tests above would still pass -- so
    assert on the mechanism, not just the numbers."""
    import inspect
    src = inspect.getsource(LT.apply_real_lens_traced)
    i = src.find('def _invert_newton_parallel')
    assert i != -1, 'could not locate _invert_newton_parallel'
    body = src[i:i + 2600]
    assert "if newton_fit == 'polynomial':" not in body, (
        'the polynomial serial-force gate is back in '
        '_invert_newton_parallel: n_workers is a silent no-op again on the '
        'default fit')
    # the GPU bail-out is legitimate and must stay
    assert 'if use_gpu:' in body, (
        'the use_gpu in-process bail-out disappeared; CuPy arrays would be '
        'host-copied through the pool')


def test_worker_payload_carries_what_the_polynomial_fit_needs():
    """The worker can only rebuild the Chebyshev evaluators if the pickled
    payload actually carries the fit choice and its parameters."""
    import inspect
    src = inspect.getsource(LT.apply_real_lens_traced)
    for key in ("'newton_fit':", "'fit_poly_order':", "'fit_weights':"):
        assert key in src, f'worker payload is missing {key}'
    wsrc = inspect.getsource(LT._newton_invert_chunk)
    assert '_Cheb2DEvaluator' in wsrc, (
        'the pool worker cannot rebuild the polynomial fit')
    # backwards compatibility: an older payload with no key must still run
    assert "get('newton_fit', 'spline')" in wsrc, (
        'worker must default to spline for payloads written before the '
        'polynomial worker path existed')
    # and it must mirror the serial combined value+gradient route
    assert 'ev_value_and_grad' in wsrc, (
        'worker does not use the combined value+gradient call, so its '
        'floating-point order differs from serial')
    # ...and, since v5.32.3, the parent's resolved EVALUATOR BACKEND.  Mirroring
    # the call is not enough: ``ev_value_and_grad`` has two implementations of
    # one formula (njit Chebyshev recurrence / pure-xp Vandermonde) and the
    # worker used to pick between them from its OWN numba availability.  A
    # parent on the other branch then disagreed with its own pool -- MEASURED
    # 5.167e-14 on this file's shape, 1.358e-11 on CI, which is what
    # ``test_pool_result_is_bit_identical_to_serial[polynomial]`` failed by.
    assert "'cheb_backend':" in src or "'cheb_backend'] =" in src, (
        'the worker payload no longer pins the parent-resolved Chebyshev '
        'backend, so pool/serial bit-identity is conditional again on the '
        "workers happening to resolve the parent's evaluator")
    assert "get('cheb_backend'" in wsrc, (
        'the worker ignores the pinned backend and re-derives its own')
    assert 'NewtonWorkerBackendUnavailable' in wsrc, (
        'a worker that cannot honour a pinned numba backend must REFUSE the '
        'chunk; substituting the other floating-point order is the silent '
        'wrong answer this pin exists to prevent')
    # ...and, since v5.33.0, the parent's BUILT FIT.  Pinning the backend made
    # the two sides evaluate in the same ORDER; it did not stop the worker
    # RE-FITTING the polynomial from these grids, and that re-fit runs
    # ``_solve_lstsq_thread_safe`` -- ``A^T A`` over ~78 000 rows -- whose BLAS
    # reduction order depends on the thread width a fresh interpreter happens
    # to start at.  MEASURED: a worker at a different BLAS width moved the
    # field by up to 1.370e-11, which is the 1.341e-11 / 1.358e-11 this file's
    # own ``test_pool_result_is_bit_identical_to_serial[polynomial]`` kept
    # failing by on all four CI python lanes AFTER the backend pin shipped.
    # See docs/audits/FIX_POOL_REBUILD_2026_08_08.md.
    assert "'cheb_fit'] =" in src or "'cheb_fit':" in src, (
        "the worker payload no longer ships the parent's BUILT Chebyshev fit, "
        'so every worker re-solves the least squares in its own interpreter '
        'and pool/serial bit-identity is conditional on the two processes '
        'sharing a BLAS thread regime')
    assert "get('cheb_fit'" in wsrc, (
        'the worker ignores the shipped fit and re-fits from the grids')
    assert 'from_state' in wsrc, (
        'the worker no longer constructs its evaluators from the shipped '
        'coefficients; a rebuild is a least-squares solve, not an evaluation')
