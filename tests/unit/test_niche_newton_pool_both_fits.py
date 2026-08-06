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

The fix teaches the worker to rebuild whichever fit the caller chose, from the
same pickled grids.  The thing that makes it safe -- and the thing this file
guards -- is that it must remain BIT-IDENTICAL to serial for both backends: the
Chebyshev fit is a deterministic lstsq on identical data so every worker
recovers the same coefficients, evaluation is elementwise so chunking cannot
change it, and the worker mirrors the serial path's choice of the combined
value+gradient call (using separate ``.ev`` calls instead would reorder the
floating-point work and break identity).

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
        assert LT._pool_reuse_is_likely(4) is False, (
            'a cold process must not report a pending promotion')
        LT._note_pool_deferral(4, slow)
        assert LT._pool_reuse_is_likely(4) is False, (
            'ONE deferred call must not promote: the first Newton inversion '
            'in a process carries one-time numba warm-up (measured 0.637 s '
            'against a 0.048 s steady state), so it is not evidence about the '
            'work the pool would actually take over')
        for _ in range(ns - 1):
            LT._note_pool_deferral(4, slow)
        assert LT._pool_reuse_is_likely(4) is True, (
            f'{ns} deferred pool-sized calls must arm the promotion')
        # promotion is per worker count: a pool built for a different count
        # would be torn down and rebuilt, so it amortises nothing
        assert LT._pool_reuse_is_likely(8) is False
        for _ in range(ns):
            LT._note_pool_deferral(8, slow)
        assert LT._pool_reuse_is_likely(8) is True
        assert LT._pool_reuse_is_likely(4) is False, (
            'recording a new worker count must RESTART the measurement, not '
            'stack alongside the old one')
        # close_worker_pool is the documented "return to cold" entry point
        LT.close_worker_pool()
        assert LT._pool_reuse_is_likely(8) is False, (
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
            LT._note_pool_deferral(4, lo)
        assert LT._pool_reuse_is_likely(4) is False, (
            'a Newton step below the cost bar must NOT promote however often '
            'it repeats: the pool would cost more per dispatch than the step '
            'it replaces')
        LT.close_worker_pool()
        for _ in range(ns):
            LT._note_pool_deferral(4, hi)
        assert LT._pool_reuse_is_likely(4) is True
        # the estimator is the MINIMUM, so one cheap sample withdraws the
        # promotion -- deliberately conservative, since over-promoting costs
        # every remaining call while under-promoting costs only the win
        LT._note_pool_deferral(4, lo)
        assert LT._pool_reuse_is_likely(4) is False, (
            'the promotion must track the MINIMUM deferred time; taking the '
            'max or the first sample promotes on numba warm-up')
    finally:
        LT.close_worker_pool()


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
    body = src[i:i + 3600]
    assert '_pool_reuse_is_likely' in body, (
        'the warm gate no longer consults the promotion flag: '
        '_PERSISTENT_POOL is not None can only become true downstream of this '
        'gate, so the warm tier would be unreachable from a cold process')
    assert '_note_pool_deferral' in body, (
        'nothing arms the promotion any more, so the second call cannot be '
        'promoted')


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
