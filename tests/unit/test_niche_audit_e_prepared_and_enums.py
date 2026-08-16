"""Territory-E fixes from ``docs/audits/AUDIT_ADVERSARIAL_CODEBASE_2026_07_25.md``.

Six mechanisms, one file (all measured 2026-07-25 on a singlet; every number in
a docstring below is a MEASURED pre-fix / post-fix pair):

* **E-H2** ``_lens_traced``: the ProcessPool Newton worker hard-coded 12
  iterations, so ``newton_max_iters`` was INERT once the pool engaged (>=200k
  Newton points with ``newton_fit='spline'`` on the CPU path) and the pool never
  emitted the unconverged warning whose own advice is "increase
  newton_max_iters".  Pre-fix at N=512: cap 1 vs 12 BIT-IDENTICAL (0.0) and
  pool(cap=1) vs serial(cap=1) 5.98e-5 apart -- i.e. the OPL (pool) and the
  ray-density amplitude (always serial) could come from DIFFERENT Newton
  solutions.  Post-fix: cap 1 vs 12 differs by 5.98e-5, and pool vs serial
  agree to 4.4e-14 at BOTH caps (the pre-existing pool-vs-serial round-off,
  unchanged by this fix).
* **E-H3** ``prepare_real_lens`` / ``PreparedAnalyticLens`` hard-coded ASM /
  ``dy = dx`` / float64 geometry and never resolved the process-wide defaults:
  after ``set_default_wave_propagator('fresnel')`` the prepared object diverged
  from ``apply_real_lens`` by 49.6.
* **E-H4** ``PreparedTracedLens`` stored the UNRESOLVED ``None`` sentinels (the
  per-call amplitude leg re-resolved them, the frozen screen did not -> 49.6 on
  a global flip) and held ``prescription`` BY REFERENCE (an in-place edit gave a
  silent stale-OPL x new-amplitude hybrid, 0.71 from a correct rebuild).
* **E-M2** ``on_noncollimated='delegate'`` forwarded the RESOLVED
  ``sag_chunk_rows`` where all four sibling amp-leg call sites forward the raw
  kwarg, so the documented force-whole-grid sentinel ``0`` became
  ``None`` -> AUTO in the delegated call.
* **E-M3 / E-M4 / E-M5** unknown enum values were accepted silently
  (``on_noncollimated`` -> 'warn', ``inversion_method`` -> Newton) and
  ``amplitude_model='ray_density'`` silently zeroed ``newton_amp_mask_rel``
  where every neighbouring requirement in the same block raises.
* **E-L22** the delegate model swap discarded the traced-only physics kwargs
  with no diagnostic.
* **E-H7** ``_lens_thin`` ``lens_model='local_only'`` did the OPPOSITE of its
  docstring (it steers the sub-beam ONTO the axis and is bit-identical to
  ``paraxial(xc=0)`` up to a piston).  Behaviour is unchanged by design; the
  docstring is now true and marks the value deprecated.
"""
from __future__ import annotations

import copy
import pickle
import warnings

import numpy as np
import pytest

import lumenairy.elements._lens_traced as LT
from lumenairy.elements._lens_real import (
    apply_real_lens,
    get_lens_sag_dtype,
    prepare_real_lens,
    set_lens_sag_dtype,
)
from lumenairy.elements._lens_thin import apply_thin_lens
from lumenairy.elements._lens_traced import apply_real_lens_traced, prepare_real_lens_traced
from lumenairy.propagators.propagation import (
    get_default_dy,
    get_default_wave_propagator,
    set_default_dy,
    set_default_wave_propagator,
)

_WL = 632.8e-9


def _singlet(R1=50e-3, R2=-50e-3, d=4e-3, glass='N-BK7', ap=5e-3):
    return {'name': 'e-audit', 'aperture_diameter': ap, 'thicknesses': [d],
            'surfaces': [
                {'radius': R1, 'glass_before': 'air', 'glass_after': glass,
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
                {'radius': R2, 'glass_before': glass, 'glass_after': 'air',
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]}


def _gauss(N=128, dx=40e-6, w=1.2e-3):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128), dx


def _maxabs(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max())


@pytest.fixture
def restore_globals():
    """Restore every process-wide default this file flips, whatever happens."""
    wp = get_default_wave_propagator()
    dyv = get_default_dy()
    sd = get_lens_sag_dtype()
    yield
    set_default_wave_propagator(wp)
    set_default_dy(dyv)
    set_lens_sag_dtype(sd)


# ===========================================================================
# E-H2 -- the pool Newton worker honours newton_max_iters
# ===========================================================================
_POOL_N = 512          # 512^2 = 262144 >= the module's 200k pool threshold
_POOL_DX = 15e-6

# THE PRECONDITION, FORCED RATHER THAN HOPED FOR
# (``docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md`` S3, dated 2026-08-15).
#
# Every pin in this section is about what travels to a pool WORKER, so they
# need the pool to ENGAGE.  Engagement is a RESOURCE decision, not a library
# behaviour: :func:`~lumenairy.elements._lens_traced._newton_resolve_workers`
# prices the pool at ``_NEWTON_POOL_RAM_FRAC (0.5) * available_bytes -
# _NEWTON_POOL_MIN_FREE_GB (2 GB)`` against a measured ~1.85 GB per worker on
# this fixture.  A 2-core / 7 GB CI runner has ~4 GB available, so its budget
# is ``0.5 * 4 - 2 = 0.0 GB``, it CORRECTLY refuses every worker, and the run
# falls back to serial -- at which point the spy observations come back empty
# and the pins below read ``assert 0 >= 1`` and ``set() == {12}``.  That is
# what the ubuntu runners hit.  Asserting engagement unconditionally asserts
# the runner's RAM, not the library's behaviour; and the old response (a
# ``pytest.skip`` when the pool did not engage) meant the ENTIRE E-H2 section
# silently stopped running on exactly the boxes the release is gated on.
#
# The precondition is therefore FORCED, with the pricer's own documented test
# hook -- ``_newton_resolve_workers(..., _free_b=)``, "overrides the live
# memory read (tests only)".  Everything else in the pricing law still runs
# exactly as shipped (rule 1's unguarded-``__main__`` refusal, the per-worker
# memory model, the re-price-at-the-count-we-will-actually-run loop); only the
# byte count it is told about becomes deterministic.  Paired with an EXPLICIT
# worker count -- so the dispatch does not follow ``available_cpus()`` either
# -- the pool engages identically on a 2-core box and on a 24-core one.
#
# Priced with the shipped model at N=512 (262 144 Newton points, 2 workers,
# 1.85 GB/worker), MEASURED 2026-08-15 on this build:
#
#     _free_b     pool budget     workers resolved
#     64 GB       30.0 GB         2    <- engaged   (_POOL_PRICED_IN_B)
#      4 GB        0.0 GB         1    <- refused   (_POOL_PRICED_OUT_B,
#                                          the CI runner's own regime)
#
# 64 GB is 2.1x the budget the 2-worker dispatch needs (3.7 GB) with the whole
# ``_NEWTON_POOL_MIN_FREE_GB`` reserve still subtracted, and 4 GB is the
# largest round figure that still prices the pool OUT, so both arms sit well
# inside their regime rather than on a threshold.  Both directions are pinned
# by ``test_h2_pool_engagement_follows_the_ram_pricing_both_ways``.
_POOL_WORKERS = 2
_POOL_PRICED_IN_B = 64e9
_POOL_PRICED_OUT_B = 4e9

# THE WIRE SHAPE THIS SECTION READS, and why it needs a memo.
#
# E-H2's mechanism pin is "the RESOLVED cap travels in the pickled payload", so
# the spy below has to read the cap out of what the parent actually submits.
# That arg tuple has had two shapes:
#
#   historical            ``(knot_data, x_chunk, y_chunk)``
#   FIX_PERF_ROUND2 item 3 ``(payload_key, blob_or_None, x_chunk, y_chunk)``
#
# In the second shape the payload is pickled ONCE per dispatch instead of once
# per chunk, and ``blob is None`` means "the live workers already hold these
# exact bytes under this key" -- the KEY alone crosses the wire.  The cap is
# still in the pickled payload (that is exactly what the digest is taken over),
# but a key-only dispatch does not RE-SHIP it, and the key that stands for a
# given payload outlives any one ``_pool_run`` because the pool and the
# parent's residency belief are module-global.  So the key -> cap memo has to
# be module-global too; a per-run dict would go blind on the second dispatch of
# an already-resident payload.  Only the cap is memoised, never the payload, so
# this holds bytes rather than the ~2 MB of fit grids.
_CAP_BY_PAYLOAD_KEY: dict = {}


def _submitted_cap(args):
    """The ``newton_max_iters`` in the payload one ``submit`` call ships.

    Accepts both wire shapes above so this section pins the CAP, not the
    packaging.  A key-only dispatch for a key whose bytes were never seen
    returns a self-describing sentinel rather than raising, so a residency
    regression shows up as a failure in the one test that asserts on caps
    instead of an ERROR in every test the fixture feeds."""
    head = args[0]
    if isinstance(head, dict):
        # Historical shape: the payload dict rode in every chunk's tuple.
        return head.get('newton_max_iters', '<missing>')
    key, blob = head, args[1]
    assert isinstance(key, str) and key, f'unexpected payload key {key!r}'
    if blob is not None:
        assert isinstance(blob, (bytes, bytearray)), (
            f'payload blob should be pickled bytes, got {type(blob).__name__}')
        _CAP_BY_PAYLOAD_KEY[key] = pickle.loads(blob).get(
            'newton_max_iters', '<missing>')
    return _CAP_BY_PAYLOAD_KEY.get(key, f'<key-only, bytes unseen: {key}>')


def _pool_run(iters, n_workers=_POOL_WORKERS, free_b=_POOL_PRICED_IN_B):
    """One ``newton_fit='spline'`` call big enough to engage the process pool.

    Wraps the pool factory so the test can PROVE the pool was asked for (the
    whole finding is confined to that path) AND capture the pickled payloads
    the parent submits, which is where the resolved cap now travels.

    ``n_workers`` and ``free_b`` together make ENGAGEMENT deterministic on any
    box -- see the block above ``_POOL_WORKERS``.  ``free_b`` is fed to the
    shipped pricer through its own ``_free_b`` test hook, so the pricing law
    itself is unmodified; pass ``_POOL_PRICED_OUT_B`` to exercise the refusal.
    """
    presc = _singlet(R1=25e-3, R2=-25e-3, d=3e-3, ap=8e-3)
    E, dx = _gauss(N=_POOL_N, dx=_POOL_DX, w=2.0e-3)
    seen = {'pool': 0, 'caps': [], 'workers': [], 'sizes': []}
    orig = LT._get_persistent_worker_pool
    orig_resolve = LT._newton_resolve_workers

    def _priced(requested, n_total, fit_points, *a, **kw):
        """The shipped pricer, told a DETERMINISTIC number of free bytes.

        Also records the two SIZES the pricing model is applied to, so the
        pricing-law test can re-price exactly what this call priced instead
        of re-deriving the fit-grid size from the prescription.
        """
        kw['_free_b'] = free_b
        n = orig_resolve(requested, n_total, fit_points, *a, **kw)
        seen['workers'].append(int(n))
        seen['sizes'].append((int(n_total), int(fit_points)))
        return n

    class _ExecutorSpy:
        def __init__(self, inner):
            self._inner = inner

        def submit(self, fn, args, *a, **kw):
            seen['caps'].append(_submitted_cap(args))
            return self._inner.submit(fn, args, *a, **kw)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def _spy(n):
        seen['pool'] += 1
        return _ExecutorSpy(orig(n))

    LT._get_persistent_worker_pool = _spy
    LT._newton_resolve_workers = _priced
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            out = LT.apply_real_lens_traced(
                E, prescription=presc, wavelength=1.0e-6, dx=dx,
                ray_subsample=1, newton_max_iters=iters,
                newton_fit='spline', on_undersample='warn',
                on_noncollimated='off', n_workers=n_workers,
                parallel_amp=False, progress=None)
        msgs = [str(w.message) for w in rec]
    finally:
        LT._get_persistent_worker_pool = orig
        LT._newton_resolve_workers = orig_resolve
    return out, seen, msgs


@pytest.fixture(scope='module')
def _pool_runs():
    """The (expensive) N=512 spline runs this section needs, computed once:
    a warm-up plus pool/serial at cap 1 and cap 12.

    Engagement is FORCED (``_POOL_WORKERS`` / ``_POOL_PRICED_IN_B``), so this
    fixture produces the same observations on a 2-core CI runner as on a
    24-core workstation and there is nothing left for a ``skip`` to hide.  The
    two skips this replaces -- ``available_cpus() <= 1`` and "the pool did not
    engage on this box" -- were exactly how the section stopped running on the
    boxes that gate the release.
    """
    _pool_run(12)          # warm-up: the FIRST N=512 call in a process differs
                           # from every later one by ~4.4e-14 (FFT plan state in
                           # the analytic amplitude leg), so measure on warm
                           # state and keep the tolerances meaningful.
    pool_1, seen_1, msgs_1 = _pool_run(1)
    pool_12, seen_12, _ = _pool_run(12)
    serial_1, seen_serial, msgs_serial = _pool_run(1, n_workers=1)
    serial_12, _, _ = _pool_run(12, n_workers=1)
    LT.close_worker_pool()
    return dict(pool_1=pool_1, pool_12=pool_12, serial_1=serial_1,
                serial_12=serial_12, seen_1=seen_1, seen_12=seen_12,
                seen_serial=seen_serial, msgs_1=msgs_1,
                msgs_serial=msgs_serial)


def test_h2_pool_path_is_actually_engaged(_pool_runs):
    """Fixture guard: the finding only exists on the process-pool path, so the
    pins below are meaningless unless the pool was requested (and NOT requested
    for the ``n_workers=1`` control).

    This now holds on ANY box, because the fixture forces the precondition
    (``_POOL_WORKERS`` / ``_POOL_PRICED_IN_B``) instead of inheriting the
    runner's core count and free RAM.  2026-08-12 it did not: a 2-core / 7 GB
    ubuntu runner priced the pool out, fell back to serial, and this read
    ``assert 0 >= 1``.
    """
    assert _pool_runs['seen_1']['pool'] >= 1
    assert _pool_runs['seen_12']['pool'] >= 1
    assert _pool_runs['seen_serial']['pool'] == 0
    # ...and the forced pricing really did resolve the requested workers, so
    # "engaged" is not being read off a pool that ran with one worker.
    assert set(_pool_runs['seen_1']['workers']) == {_POOL_WORKERS}
    assert set(_pool_runs['seen_12']['workers']) == {_POOL_WORKERS}
    assert set(_pool_runs['seen_serial']['workers']) == {1}


def test_h2_the_ram_pricing_law_itself(_pool_runs):
    """The pricing ARITHMETIC, exercised straight on the shipped function --
    no lens call, so this arm costs nothing and covers the whole ladder.

    ``docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md`` S3.  This is what decides
    whether the section above has a pool to observe, so it is stated as its
    own claim rather than inferred from the observations.  Priced at the
    fixture's own size (N=512 -> 262 144 Newton points) with the same fit
    grid, measured 2026-08-15: ~1.85 GB per worker, so the budget
    ``0.5 * free - 2 GB`` admits 2 workers from 32 GB up and refuses below
    ~8 GB.  The 4 GB row is the 2-core / 7 GB CI runner's own regime.
    """
    # the exact sizes the fixture's pooled dispatch was priced at
    sizes = _pool_runs['seen_12']['sizes']
    assert sizes, 'the pricer was never consulted'
    n_total, fit_points = sizes[0]
    assert n_total == _POOL_N * _POOL_N, (n_total, _POOL_N)
    for free_b, want in ((64e9, _POOL_WORKERS), (32e9, _POOL_WORKERS),
                         (8e9, 1), (4e9, 1)):
        got = LT._newton_resolve_workers(
            _POOL_WORKERS, n_total, fit_points, on_pool_memory='silent',
            _free_b=free_b)
        assert got == want, (
            f'{free_b / 1e9:.0f} GB free priced {got} worker(s), expected '
            f'{want}; per-worker model says '
            f'{LT._newton_worker_bytes(n_total / _POOL_WORKERS, fit_points) / 1e9:.2f}'
            f' GB against a budget of {(0.5 * free_b - 2e9) / 1e9:.1f} GB')
    # the two arms the section actually uses must land on opposite sides
    assert LT._newton_resolve_workers(
        _POOL_WORKERS, n_total, fit_points, on_pool_memory='silent',
        _free_b=_POOL_PRICED_IN_B) == _POOL_WORKERS
    assert LT._newton_resolve_workers(
        _POOL_WORKERS, n_total, fit_points, on_pool_memory='silent',
        _free_b=_POOL_PRICED_OUT_B) == 1


def test_h2_pool_engagement_follows_the_ram_pricing_both_ways(_pool_runs):
    """The resource decision the fixture above pins DOWN, asserted here as its
    own claim -- in BOTH directions, on the shipped pricing law.

    ``docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md`` S3.  Splitting it out is
    the point: the E-H2 pins are about the PAYLOAD, and they should not double
    as an accidental (and box-dependent) assertion about RAM.  Here the RAM is
    the subject, so it is stated twice and deliberately:

    * priced IN (64 GB free -> 30.0 GB of pool budget against 1.85 GB/worker):
      the pool is built and the payload crosses the wire;
    * priced OUT (4 GB free -> 0.0 GB of budget, the 2-core/7 GB CI runner's
      own regime): the pool is REFUSED, no worker is ever asked for, and the
      call completes serially.

    The refusal is the library being right, so the answer must be unchanged by
    it -- the pool path is documented bit-identical to serial, and that is
    asserted here rather than assumed.
    """
    # PRICED IN: reuse the fixture's own pooled cap-12 run rather than paying
    # for a seventh N=512 solve -- it was produced at _POOL_PRICED_IN_B by
    # construction, and this test is about the OTHER side of the decision.
    out_in, seen_in = _pool_runs['pool_12'], _pool_runs['seen_12']
    out_out, seen_out, _ = _pool_run(12, free_b=_POOL_PRICED_OUT_B)
    LT.close_worker_pool()

    # priced in -> engaged, at the count that was asked for
    assert seen_in['pool'] >= 1
    assert set(seen_in['workers']) == {_POOL_WORKERS}
    assert seen_in['caps'], 'engaged but nothing was submitted'

    # priced out -> refused, before any worker is requested
    assert set(seen_out['workers']) == {1}, (
        f"the pricer allowed {sorted(set(seen_out['workers']))} worker(s) on "
        f"a {_POOL_PRICED_OUT_B / 1e9:.0f} GB budget; the refusal arm is no "
        f"longer exercising a refusal")
    assert seen_out['pool'] == 0
    assert seen_out['caps'] == []

    # ...and refusing cost nothing but wall time.  Same bar, same derivation,
    # as test_h2_pool_and_serial_share_one_newton_solution: the pool-vs-serial
    # residual is the ambient 4.4e-14 float floor and 1e-11 sits 250x above it.
    d = _maxabs(out_in, out_out)
    assert d < 1e-11, (
        f'the RAM refusal moved the answer by {d:.3e}; the clamp is a pure '
        f'resource decision and the pool path is bit-identical to serial')


def test_h2_resolved_cap_travels_in_the_pickled_payload(_pool_runs):
    """The mechanism: the ``_spline_data`` payload the parent submits to each
    worker must carry the RESOLVED cap.  Pre-fix the key did not exist and the
    worker used the module constant, which is exactly why the kwarg was
    inert.

    Stated as a SET, so what is pinned is the INVARIANT -- every payload that
    crosses the wire carries the resolved cap -- and not a submission COUNT,
    which is a function of the worker count and the chunking.  The emptiness
    guard is separate and explicit: an empty observation satisfies nothing,
    and on 2026-08-12 a serial fallback made this read ``set() == {12}``,
    which named the wrong defect.  Engagement is forced by the fixture (S3).
    """
    for tag, want in (('seen_1', 1), ('seen_12', 12)):
        caps = _pool_runs[tag]['caps']
        assert caps, (
            f'nothing was submitted for {tag}: the pool did not dispatch, so '
            f'this pin has no payload to read.  The engagement claim lives in '
            f'test_h2_pool_engagement_follows_the_ram_pricing_both_ways')
        assert set(caps) == {want}


def test_h2_newton_max_iters_is_live_on_the_pool_path(_pool_runs):
    """Headline pin: at N=512 with ``newton_fit='spline'`` the pool engages and
    ``newton_max_iters=1`` must NOT give the same field as ``12``.

    Pre-fix the two were bit-identical (max|dE| = 0.0 exactly, the audit's
    proof that the kwarg was inert); post-fix 5.98e-5.
    """
    d = _maxabs(_pool_runs['pool_1'], _pool_runs['pool_12'])
    assert d > 1e-8, (
        f'newton_max_iters is inert on the pool path: max|dE| = {d:.3e} '
        f'between cap 1 and cap 12')


def test_h2_pool_and_serial_share_one_newton_solution(_pool_runs):
    """The amp/phase-consistency half of E-H2: the ray-density amplitude leg
    always runs the SERIAL Newton closure while the OPL can run in the pool, so
    the two must solve the SAME problem at the SAME cap.

    Pre-fix at cap=1 they were **5.98e-5** apart (the pool silently ran 12
    iterations); post-fix they agree to the ambient pool-vs-serial float floor,
    measured 4.4e-14 at BOTH cap=1 and cap=12 -- i.e. the residual is the
    pre-existing round-off (identical at the default cap before and after this
    fix), not a cap mismatch.  1e-11 sits 250x above that floor and 6 orders
    below the pre-fix delta.
    """
    d1 = _maxabs(_pool_runs['pool_1'], _pool_runs['serial_1'])
    d12 = _maxabs(_pool_runs['pool_12'], _pool_runs['serial_12'])
    assert d1 < 1e-11, f'cap=1 pool vs serial: max|dE| = {d1:.3e}'
    assert d12 < 1e-11, f'cap=12 pool vs serial: max|dE| = {d12:.3e}'
    # ... and the tightened cap really did change the answer on BOTH paths, so
    # the agreement above is not the trivial "both ignored the cap".
    assert _maxabs(_pool_runs['serial_1'], _pool_runs['serial_12']) > 1e-8


def test_h2_pool_emits_the_same_unconverged_warning_as_serial(_pool_runs):
    """The pool used to be SILENT about unconverged pixels even though the
    serial path warns (and the message's own remedy is "increase
    newton_max_iters" -- the very knob the pool ignored).  Both paths must now
    report the same counts with the same wording."""
    def _unconv(msgs):
        return [m for m in msgs if 'did not converge' in m]

    pool_msgs = _unconv(_pool_runs['msgs_1'])
    serial_msgs = _unconv(_pool_runs['msgs_serial'])
    assert pool_msgs, _pool_runs['msgs_1']
    assert serial_msgs, _pool_runs['msgs_serial']
    assert pool_msgs[0] == serial_msgs[0]
    assert 'newton_max_iters' in pool_msgs[0]
    assert 'within 1 iterations' in pool_msgs[0]


def test_h2_worker_reads_the_cap_and_stays_backward_compatible():
    """Unit-level pin on the worker itself: it must read
    ``knot_data['newton_max_iters']`` and fall back to the module default when
    the key is absent (an older pickled payload).  Also pins the (opl,
    n_unconverged) return contract the parent's warning depends on."""
    n = 21
    xs = np.linspace(-1e-3, 1e-3, n)
    Xi, Yi = np.meshgrid(xs, xs, indexing='ij')
    payload = {
        'xs_in': xs,
        # A deliberately non-linear forward map so Newton needs >1 iteration.
        'x_out_grid': Xi * (1.0 + 4.0e4 * (Xi ** 2 + Yi ** 2)),
        'y_out_grid': Yi * (1.0 + 4.0e4 * (Xi ** 2 + Yi ** 2)),
        'opl_grid': (Xi ** 2 + Yi ** 2) * 1.0e3,
        'launch_radius': 1e-3, 'dx': 1e-7, 'bound': 0.999e-3,
        'inv_M_x': 1.0, 'inv_M_y': 1.0,
    }
    xq = np.array([6e-4, -6e-4, 3e-4])
    yq = np.array([0.0, 2e-4, -3e-4])
    opl_1, unconv_1 = LT._newton_invert_chunk(
        ({**payload, 'newton_max_iters': 1}, xq, yq))
    opl_12, unconv_12 = LT._newton_invert_chunk(
        ({**payload, 'newton_max_iters': 12}, xq, yq))
    opl_default, _ = LT._newton_invert_chunk((dict(payload), xq, yq))
    assert opl_1.shape == xq.shape
    assert unconv_1 > unconv_12          # a tighter cap leaves more active
    assert not np.array_equal(opl_1, opl_12)
    # No key -> the historical module default (12), byte-identical.
    assert np.array_equal(opl_default, opl_12)
    assert LT._NEWTON_MAX_ITERS == 12


def test_h2_cap_reader_tracks_every_wire_shape_the_worker_accepts():
    """Guard on the READER the pool pins above depend on.

    ``_submitted_cap`` has to decode whatever ``_invert_newton_parallel``
    submits; when FIX_PERF_ROUND2 item 3 replaced the embedded payload with
    ``(key, blob_or_None, ...)`` the old reader called ``.get`` on the digest
    STRING and every ``_pool_runs`` consumer died in setup.  This test fails
    instead, by name, if the wire shape moves again -- and it needs no pool, so
    it runs on a one-core box too.  Every shape here is one ``_newton_invert_
    chunk`` accepts, and the blob is built by the library's own producer."""
    n = 21
    xs = np.linspace(-1e-3, 1e-3, n)
    Xi, Yi = np.meshgrid(xs, xs, indexing='ij')
    payload = {
        'xs_in': xs,
        'x_out_grid': Xi * (1.0 + 4.0e4 * (Xi ** 2 + Yi ** 2)),
        'y_out_grid': Yi * (1.0 + 4.0e4 * (Xi ** 2 + Yi ** 2)),
        'opl_grid': (Xi ** 2 + Yi ** 2) * 1.0e3,
        'launch_radius': 1e-3, 'dx': 1e-7, 'bound': 0.999e-3,
        'inv_M_x': 1.0, 'inv_M_y': 1.0, 'newton_max_iters': 7,
    }
    xq = np.array([6e-4, -6e-4, 3e-4])
    yq = np.array([0.0, 2e-4, -3e-4])
    key, blob = LT._newton_payload_blob(payload)
    # Historical shape -- the payload dict itself.
    assert _submitted_cap((payload, xq, yq)) == 7
    # Round-2 shape, bytes attached: read out of the PICKLED payload.
    assert _submitted_cap((key, blob, xq, yq)) == 7
    # ...and the same bytes are what the worker runs, at that same cap.
    assert LT._newton_invert_chunk((key, blob, xq.copy(), yq.copy()))[0].shape \
        == xq.shape
    # Key-only re-send: answered from the memo the blob above populated, which
    # is the case a per-run cache would miss.
    assert _submitted_cap((key, None, xq, yq)) == 7
    # A key whose bytes were never seen degrades to a sentinel, not a raise:
    # the fixture must not turn a residency regression into five setup ERRORs.
    unseen = _submitted_cap(('00' * 16, None, xq, yq))
    assert isinstance(unseen, str) and unseen.startswith('<key-only')
    # A payload with no cap key at all (a pre-5.29.1 pickle) reads as missing
    # rather than silently as the module default.
    bare = {k: v for k, v in payload.items() if k != 'newton_max_iters'}
    assert _submitted_cap((bare, xq, yq)) == '<missing>'
    assert _submitted_cap(LT._newton_payload_blob(bare) + (xq, yq)) == '<missing>'


# ===========================================================================
# E-H3 -- prepare_real_lens freezes the defaults live at prepare time
# ===========================================================================
def test_h3_prepared_analytic_lens_freezes_the_wave_propagator(restore_globals):
    """Contract: a prepared object freezes the settings that were live when it
    was prepared.

    Pre-fix ``prepare_real_lens`` hard-coded ASM, so after flipping the global
    default the prepared object was 49.6 away from ``apply_real_lens`` -- while
    a freshly prepared object was ALSO 49.6 away (it never read the default at
    all).  Post-fix: the frozen object does not move (0.0), a fresh one matches
    the direct call exactly (0.0), and the direct call follows the global.
    """
    presc = _singlet()
    E, dx = _gauss()
    set_default_wave_propagator('asm')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prep_asm = prepare_real_lens(prescription=presc, wavelength=_WL,
                                     dx=dx, N=E.shape[0])
        assert prep_asm.wave_propagator == 'asm'
        out_asm = prep_asm(E)
        direct_asm = apply_real_lens(E, prescription=presc, wavelength=_WL,
                                     dx=dx)
        assert _maxabs(out_asm, direct_asm) == 0.0

        set_default_wave_propagator('fresnel')
        prep_fresnel = prepare_real_lens(prescription=presc, wavelength=_WL,
                                         dx=dx, N=E.shape[0])
        direct_fresnel = apply_real_lens(E, prescription=presc,
                                         wavelength=_WL, dx=dx)

    # the global flip is a REAL change (guards the fixture)
    assert _maxabs(direct_asm, direct_fresnel) > 1.0
    # 1. the already-prepared object is frozen
    assert _maxabs(prep_asm(E), out_asm) == 0.0
    # 2. ... and still equals apply_real_lens with its PREPARE-time settings
    assert _maxabs(prep_asm(E), apply_real_lens(
        E, prescription=presc, wavelength=_WL, dx=dx,
        wave_propagator='asm')) == 0.0
    # 3. a freshly prepared object adopts the new live default
    assert prep_fresnel.wave_propagator == 'fresnel'
    assert _maxabs(prep_fresnel(E), direct_fresnel) == 0.0


def test_h3_prepared_analytic_lens_resolves_sag_dtype_and_dy(restore_globals):
    """The other two process-wide knobs the prepared analytic lens ignored:
    ``set_lens_sag_dtype`` (pre-fix 1.45e-5 divergence) and
    ``set_default_dy`` (pre-fix 1.69)."""
    presc = _singlet()
    E, dx = _gauss()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        set_lens_sag_dtype(np.float32)
        prep32 = prepare_real_lens(prescription=presc, wavelength=_WL, dx=dx,
                                   N=E.shape[0])
        direct32 = apply_real_lens(E, prescription=presc, wavelength=_WL,
                                   dx=dx)
        assert prep32.sag_dtype == np.float32
        assert _maxabs(prep32(E), direct32) == 0.0
        set_lens_sag_dtype(np.float64)
        # frozen: flipping back does not move the float32-prepared object
        assert _maxabs(prep32(E), direct32) == 0.0

        set_default_dy(dx * 1.05)
        prep_dy = prepare_real_lens(prescription=presc, wavelength=_WL, dx=dx,
                                    N=E.shape[0])
        direct_dy = apply_real_lens(E, prescription=presc, wavelength=_WL,
                                    dx=dx)
        assert prep_dy._dy == pytest.approx(dx * 1.05)
        assert _maxabs(prep_dy(E), direct_dy) == 0.0


def test_h3_prepare_real_lens_rejects_an_unknown_propagator():
    with pytest.raises(ValueError, match='unknown wave_propagator'):
        prepare_real_lens(prescription=_singlet(), wavelength=_WL, dx=40e-6,
                          N=64, wave_propagator='not-a-propagator')


# ===========================================================================
# E-H4 -- PreparedTracedLens freezes defaults AND deep-copies the prescription
# ===========================================================================
_TRACED_PREP_KW = dict(ray_subsample=4, on_undersample='silent',
                       on_noncollimated='off')


def test_h4_prepared_traced_lens_freezes_the_resolved_defaults(restore_globals):
    """Pre-fix ``PreparedTracedLens`` stored the unresolved ``None`` sentinels,
    so the per-call analytic amplitude leg re-resolved them at CALL time while
    the frozen screen kept the prepare-time physics: flipping the global
    propagator moved the prepared output by 49.6 (and the sag dtype by
    1.45e-5).  Post-fix both are 0.0 and the resolved values are readable."""
    presc = _singlet()
    E, dx = _gauss()
    set_default_wave_propagator('asm')
    set_lens_sag_dtype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prep = prepare_real_lens_traced(prescription=presc, wavelength=_WL,
                                        dx=dx, N=E.shape[0], **_TRACED_PREP_KW)
        base = prep(E)
    assert prep.wave_propagator == 'asm'
    assert prep.sag_dtype == np.float64
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        set_default_wave_propagator('fresnel')
        assert _maxabs(prep(E), base) == 0.0
        set_default_wave_propagator('asm')
        set_lens_sag_dtype(np.float32)
        assert _maxabs(prep(E), base) == 0.0


def test_h4_prepared_traced_lens_deep_copies_the_prescription():
    """The optimizer / tolerancing pattern this class advertises: mutate the
    prescription in place and re-call.  Pre-fix the prepared lens held the
    caller's dict BY REFERENCE, so the cached (stale) OPL screen was paired
    with a freshly-computed amplitude for the NEW lens -- a silent hybrid,
    measured 1.77 from its own previous output and 0.71 from a correct
    rebuild.  Post-fix the prepared lens is immune (0.0) and a full rebuild is
    the only way to see the new lens."""
    presc = _singlet()
    E, dx = _gauss()
    rx_live = copy.deepcopy(presc)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prep = prepare_real_lens_traced(prescription=rx_live, wavelength=_WL,
                                        dx=dx, N=E.shape[0], **_TRACED_PREP_KW)
        before = prep(E)
        rx_live['surfaces'][0]['radius'] = 40e-3     # in-place edit
        after = prep(E)
        rebuilt = apply_real_lens_traced(
            E, prescription=rx_live, wavelength=_WL, dx=dx,
            **_TRACED_PREP_KW)
    assert prep.prescription is not rx_live
    assert prep.prescription['surfaces'][0]['radius'] == pytest.approx(50e-3)
    assert _maxabs(after, before) == 0.0
    # the mutation IS a different lens -- so the prepared screen must not
    # silently pretend to track it
    assert _maxabs(rebuilt, before) > 0.1


# ===========================================================================
# E-M2 / E-L22 -- the on_noncollimated='delegate' model swap
# ===========================================================================
def _noncollimated_field(N=128, dx=8.0e-6, w=0.25e-3, tilt=0.03):
    """A globally-tilted beam whose residual angular spread (0.03 rad) exceeds
    the collimated-reference threshold (0.02 rad) while staying below this
    grid's Nyquist tilt lam/(2 dx) = 0.0396 rad, so the wrapping-safe estimator
    reads it correctly and the guard fires."""
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    env = np.exp(-(X ** 2 + Y ** 2) / w ** 2)
    return (env * np.exp(1j * (2 * np.pi / _WL) * tilt * X)
            ).astype(np.complex128), dx


_DELEGATE_PRESC = _singlet(R1=10e-3, R2=-10e-3, d=1e-3, ap=0.8e-3)


def _delegate_forwarded_sag_chunk_rows(**kw):
    """Call the traced element on a non-collimated input with
    ``on_noncollimated='delegate'`` and capture what the delegated
    ``apply_real_lens`` was handed."""
    seen = []
    orig = LT.apply_real_lens

    def _spy(*a, **kwargs):
        seen.append(kwargs.get('sag_chunk_rows', '<missing>'))
        return orig(*a, **kwargs)

    E, dx = _noncollimated_field()
    LT.apply_real_lens = _spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            LT.apply_real_lens_traced(
                E, prescription=_DELEGATE_PRESC, wavelength=_WL, dx=dx,
                ray_subsample=4, on_undersample='silent',
                on_noncollimated='delegate', **kw)
    finally:
        LT.apply_real_lens = orig
    return seen


def test_m2_delegate_forwards_the_raw_sag_chunk_rows():
    """``sag_chunk_rows=0`` is the documented force-whole-grid sentinel.  The
    resolver maps ``0 -> None`` and ``apply_real_lens`` re-resolves ``None`` to
    AUTO, so forwarding the RESOLVED value (pre-fix) silently re-enabled row
    banding against an explicit opt-out.  All four sibling amp-leg call sites
    forward the RAW kwarg; the delegate must too.

    The single captured call also proves the delegate branch (not the amp legs,
    which would give two calls) is what ran.
    """
    assert _delegate_forwarded_sag_chunk_rows(sag_chunk_rows=0) == [0]
    assert _delegate_forwarded_sag_chunk_rows(sag_chunk_rows=32) == [32]
    assert _delegate_forwarded_sag_chunk_rows() == [None]


def test_l22_delegate_reports_the_discarded_physics_kwargs():
    """The model swap drops every traced-only knob (``apply_real_lens`` has no
    ray-trace leg).  Pre-fix that was silent; now the non-default ones are
    named, and an all-default delegate stays quiet."""
    E, dx = _noncollimated_field()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        apply_real_lens_traced(
            E, prescription=_DELEGATE_PRESC, wavelength=_WL, dx=dx,
            ray_subsample=4, on_undersample='silent',
            on_noncollimated='delegate', preserve_input_phase=False,
            newton_max_iters=5, newton_fit='spline', inversion_method='fit')
    dropped = [str(w.message) for w in rec if 'DISCARDED' in str(w.message)]
    assert dropped, [str(w.message) for w in rec]
    msg = dropped[0]
    for token in ('preserve_input_phase=False', "inversion_method='fit'",
                  "newton_fit='spline'", 'newton_max_iters=5',
                  'ray_subsample=4'):
        assert token in msg, (token, msg)
    assert 'apply_real_lens' in msg          # names the model it fell back to

    with warnings.catch_warnings(record=True) as rec2:
        warnings.simplefilter('always')
        apply_real_lens_traced(
            E, prescription=_DELEGATE_PRESC, wavelength=_WL, dx=dx,
            on_undersample='silent', on_noncollimated='delegate')
    assert not [w for w in rec2 if 'DISCARDED' in str(w.message)]


# ===========================================================================
# R1-WIRING -- the delegate branch forwards carrier=, and adjudicates it
# ===========================================================================
# VERIFY_ARCHITECTURE F5/P2-9: ZERO ``apply_real_lens`` calls made by the
# traced path forwarded ``carrier=``, so the screen-obliquity + R1 corrections
# were inert for every shipped consumer.  The delegate branch is the one site
# where the analytic model IS the answer (the amp legs subtract their own
# analytic phase straight back out), so it is the one that must forward -- but
# it fires precisely BECAUSE the carrier does not describe the field, so the
# forward is adjudicated by the guard's own F1 statistic: forward only when
# removing the carrier REDUCES the input's angular spread.


def _carriered_noncollimated_field(N=512, dx=4.0e-6, w=0.6e-3, tilt=0.05,
                                   f_div=0.02):
    """A tilted beam with a strong DIVERGENT leg on top: the carrier states the
    tilt, the divergence is the residual.  ``lam/(2 dx) = 0.164 rad`` here, so
    both the good carrier's residual (0.037) and the reversed carrier's (0.107)
    are inside the wrapping-safe estimator's unambiguous range."""
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    env = np.exp(-(X ** 2 + Y ** 2) / w ** 2)
    W = tilt * X + (X ** 2 + Y ** 2) / (2.0 * f_div)
    return (env * np.exp(1j * (2 * np.pi / _WL) * W)).astype(np.complex128), dx


def _delegate_out(E, dx, carrier):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = apply_real_lens_traced(
            E, prescription=_DELEGATE_PRESC, wavelength=_WL, dx=dx,
            carrier=carrier, on_undersample='silent',
            on_noncollimated='delegate')
    return out, [str(w.message) for w in rec]


def test_the_delegate_branch_forwards_a_carrier_that_describes_the_field():
    """The wiring: with a carrier the delegate's answer must be the CARRIERED
    analytic answer, byte for byte -- not the carrier-free one it used to be.
    That is the whole of F5: the feature was unreachable from the traced path.
    """
    import lumenairy as la
    E, dx = _carriered_noncollimated_field()
    car = la.TiltedCarrier(float('inf'), 0.05, 0.0)
    out, msgs = _delegate_out(E, dx, car)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        want = LT.apply_real_lens(
            E, prescription=_DELEGATE_PRESC, wavelength=_WL, dx=dx,
            carrier=car, on_screen_obliquity='silent')
        blind = LT.apply_real_lens(E, prescription=_DELEGATE_PRESC,
                                   wavelength=_WL, dx=dx)
    assert np.array_equal(out.view(np.uint8), want.view(np.uint8))
    assert not np.array_equal(out.view(np.uint8), blind.view(np.uint8))
    assert any('IS forwarded' in m for m in msgs), msgs


def test_the_delegate_branch_refuses_a_carrier_that_does_not_describe_it():
    """The adjudication, on the case that makes it necessary: a carrier of the
    RIGHT size and the WRONG sign.  The correction's cross term ``2 p0 . q``
    flips with it, so forwarding would be worse than not correcting at all --
    the same "wrong by twice the term" signature the refutation used.  The
    discriminator is the guard's OWN statistic (residual with the carrier
    removed vs the raw input tilt), not a new heuristic, and the refusal is
    stated in the warning rather than silent."""
    import lumenairy as la
    E, dx = _carriered_noncollimated_field()
    anti = la.TiltedCarrier(float('inf'), -0.05, 0.0)
    out, msgs = _delegate_out(E, dx, anti)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        blind = LT.apply_real_lens(E, prescription=_DELEGATE_PRESC,
                                   wavelength=_WL, dx=dx)
        wrong = LT.apply_real_lens(
            E, prescription=_DELEGATE_PRESC, wavelength=_WL, dx=dx,
            carrier=anti, on_screen_obliquity='silent')
    assert np.array_equal(out.view(np.uint8), blind.view(np.uint8))
    assert not np.array_equal(out.view(np.uint8), wrong.view(np.uint8))
    assert any('NOT forwarded' in m for m in msgs), msgs
    # ...and the statistic that decided it says what the docstring claims
    Wc, _l, _m = LT._tilted_carrier_parts(anti, *np.meshgrid(
        (np.arange(E.shape[1]) - E.shape[1] / 2) * dx,
        (np.arange(E.shape[0]) - E.shape[0] / 2) * dx))
    assert (LT._carrier_residual_rms(E, Wc, _WL, dx)
            > LT._input_tilt_stats(E, _WL, dx)[0])


def test_the_amp_legs_do_not_forward_the_carrier():
    """The adjudicated NON-forward, pinned structurally.  The traced exit is
    ``E_analytic * exp(i(k0 opl_traced - phase_analytic_lens))`` and BOTH
    factors come from ``apply_real_lens``, so the analytic lens phase -- the
    correction included -- is subtracted straight back out: forwarding to all
    the amp legs moves the traced answer by 8.7e-05 waves rms where the
    correction itself is 4.3e-03 (a 49x cancellation), and forwarding to only
    the ``E_in`` leg injects the FULL 4.3e-03 into an answer that is already
    exact.  Zero benefit, an asymmetry hazard, and 2.2-3.6x the analytic wall
    clock; see docs/audits/BUILD_R1_WIRING_2026_08_12.md S4."""
    import lumenairy as la
    seen = []
    orig = LT.apply_real_lens

    def _spy(*a, **kw):
        seen.append(kw.get('carrier', '<missing>'))
        return orig(*a, **kw)

    E, dx = _gauss(N=128, dx=8.0e-6, w=0.25e-3)
    LT.apply_real_lens = _spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            apply_real_lens_traced(
                E, prescription=_DELEGATE_PRESC, wavelength=_WL, dx=dx,
                ray_subsample=4, on_undersample='silent',
                on_noncollimated='off',
                carrier=la.TiltedCarrier(float('inf'), 0.01, 0.0))
    finally:
        LT.apply_real_lens = orig
    assert seen, 'no amp-leg call was made'
    assert all(v == '<missing>' for v in seen), seen


# ===========================================================================
# E-M3 / E-M4 / E-M5 -- enum membership guards (house rule: unknown -> raise)
# ===========================================================================
@pytest.mark.parametrize('bad', ['WARN', 'junk', 'nope', '', 42, None])
def test_m3_on_noncollimated_rejects_unknown_values(bad):
    """Pre-fix ANY value that was not 'off' / 'delegate' fell through to the
    'warn' branch, so a typo silently selected a policy."""
    E, dx = _gauss(N=64, dx=60e-6, w=0.8e-3)
    with pytest.raises(ValueError, match='on_noncollimated'):
        apply_real_lens_traced(
            E, prescription=_singlet(), wavelength=_WL, dx=dx,
            ray_subsample=8, on_undersample='silent', on_noncollimated=bad)


@pytest.mark.parametrize('alias', ['silent', 'ignore'])
def test_m3_silent_and_ignore_are_aliases_for_off(alias):
    """The sibling knobs in this signature spell suppression 'silent', so every
    in-repo call site that wanted the guard off wrote
    ``on_noncollimated='silent'`` -- and got the WARN branch, i.e. the opposite
    of the request.  The aliases must now actually suppress, and must give the
    same field as 'off'."""
    E, dx = _noncollimated_field()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out_alias = apply_real_lens_traced(
            E, prescription=_DELEGATE_PRESC, wavelength=_WL, dx=dx,
            ray_subsample=4, on_undersample='silent',
            on_noncollimated=alias)
    assert not [w for w in rec
                if 'residual angular spread' in str(w.message)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out_off = apply_real_lens_traced(
            E, prescription=_DELEGATE_PRESC, wavelength=_WL, dx=dx,
            ray_subsample=4, on_undersample='silent', on_noncollimated='off')
    assert np.array_equal(out_alias, out_off)


def test_m3_warn_still_warns():
    """Negative control: the default policy is unchanged."""
    E, dx = _noncollimated_field()
    with pytest.warns(RuntimeWarning, match='residual angular spread'):
        apply_real_lens_traced(
            E, prescription=_DELEGATE_PRESC, wavelength=_WL, dx=dx,
            ray_subsample=4, on_undersample='silent',
            on_noncollimated='warn')


@pytest.mark.parametrize('bad', ['NEWTON', 'junk', 'backward', '', 7, None])
def test_m4_inversion_method_rejects_unknown_values(bad):
    """Pre-fix any unrecognised ``inversion_method`` silently ran Newton (the
    dispatch is a chain of equality tests), so 'backward' -- a plausible typo
    for 'backward_trace' -- looked like it worked."""
    E, dx = _gauss(N=64, dx=60e-6, w=0.8e-3)
    with pytest.raises(ValueError, match='inversion_method'):
        apply_real_lens_traced(
            E, prescription=_singlet(), wavelength=_WL, dx=dx,
            ray_subsample=8, on_undersample='silent',
            on_noncollimated='off', inversion_method=bad)


@pytest.mark.parametrize('good', ['newton', 'fit'])
def test_m4_inversion_method_accepts_the_documented_values(good):
    E, dx = _gauss(N=64, dx=60e-6, w=0.8e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = apply_real_lens_traced(
            E, prescription=_singlet(), wavelength=_WL, dx=dx,
            ray_subsample=8, on_undersample='silent',
            on_noncollimated='off', inversion_method=good)
    assert np.isfinite(np.abs(out)).any()


def test_m4_inversion_method_documents_its_values():
    doc = apply_real_lens_traced.__doc__ or ''
    assert 'inversion_method :' in doc
    for v in ("'newton'", "'fit'", "'backward_trace'"):
        assert v in doc


def test_m5_ray_density_rejects_an_explicit_amp_mask():
    """``amplitude_model='ray_density'`` REQUIRES the full coarse Newton grid.
    Every other requirement in that block raises; this one silently zeroed the
    caller's value.  It now raises for an explicitly-requested mask while the
    shipped default (read as "not requested") and an explicit 0.0 still work."""
    E, dx = _gauss(N=64, dx=60e-6, w=0.8e-3)
    kw = dict(prescription=_singlet(), wavelength=_WL, dx=dx,
              ray_subsample=8, on_undersample='silent',
              on_noncollimated='off', amplitude_model='ray_density')
    with pytest.raises(ValueError, match='newton_amp_mask_rel'):
        apply_real_lens_traced(E, newton_amp_mask_rel=1e-2, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = apply_real_lens_traced(E, newton_amp_mask_rel=0.0, **kw)
        b = apply_real_lens_traced(E, **kw)
    assert np.array_equal(a, b)
    # The default is still the documented 1e-4 (the guard uses the module
    # constant rather than changing the public signature).
    import inspect
    assert inspect.signature(
        apply_real_lens_traced).parameters[
            'newton_amp_mask_rel'].default == 1e-4
    assert LT._NEWTON_AMP_MASK_REL_DEFAULT == 1e-4


# ===========================================================================
# E-H7 -- lens_model='local_only' documents what it actually does
# ===========================================================================
_H7 = dict(f=5.0e-3, wavelength=1.0e-6, dx=2.0e-6, xc=100e-6, yc=0.0)


def _phase_slope_at_lenslet_centre(model):
    N = 256
    E = np.ones((N, N), dtype=np.complex128)
    out = apply_thin_lens(E, lens_model=model, **_H7)
    ph = np.unwrap(np.unwrap(np.angle(out), axis=0), axis=1)
    row = N // 2
    j = int(round(_H7['xc'] / _H7['dx'] + N / 2))
    return (ph[row, j + 1] - ph[row, j - 1]) / (2 * _H7['dx'])


def test_h7_local_only_steers_onto_the_axis_and_paraxial_does_not():
    """The measured physics the docstring must describe: ``local_only``'s local
    phase gradient at the lenslet centre is ``-k*xc/f`` (-20.0 mrad for
    xc = 100 um, f = 5 mm), i.e. it steers the sub-beam ONTO the axis, while
    the plain decentered ``paraxial`` lens has exactly zero gradient there --
    the no-steer behaviour the old text attributed to ``local_only``."""
    k = 2 * np.pi / _H7['wavelength']
    g_local = _phase_slope_at_lenslet_centre('local_only')
    g_parax = _phase_slope_at_lenslet_centre('paraxial')
    assert g_parax == pytest.approx(0.0, abs=1e-6)
    assert g_local / k == pytest.approx(-_H7['xc'] / _H7['f'], rel=1e-6)


def test_h7_local_only_is_paraxial_at_the_origin_up_to_a_piston():
    """It is a dead duplicate: identical to ``paraxial(xc=yc=0)`` except for
    the constant piston ``-k/(2f)(xc^2+yc^2)`` (measured |ratio| spread
    7.8e-16, arg spread 7.1e-14 rad)."""
    N = 256
    E = np.ones((N, N), dtype=np.complex128)
    o_local = apply_thin_lens(E, lens_model='local_only', **_H7)
    o_axis = apply_thin_lens(E, lens_model='paraxial',
                             **{**_H7, 'xc': 0.0, 'yc': 0.0})
    ratio = o_local / o_axis
    assert np.ptp(np.abs(ratio)) < 1e-12
    assert np.ptp(np.angle(ratio)) < 1e-10
    k = 2 * np.pi / _H7['wavelength']
    expected_piston = -k / (2 * _H7['f']) * (_H7['xc'] ** 2 + _H7['yc'] ** 2)
    # compare WRAPPED (the analytic piston here is -2 pi exactly, which
    # np.angle reports as ~0)
    assert np.angle(ratio)[0, 0] == pytest.approx(
        float(np.angle(np.exp(1j * expected_piston))), abs=1e-9)


def test_h7_docstring_describes_the_actual_behaviour():
    """Pin the DOC fix (the audit's decision was doc-over-delete: it is a
    public enum value, so it gets a deprecation note, not a removal)."""
    doc = apply_thin_lens.__doc__ or ''
    i = doc.find("'local_only'")
    assert i > 0
    block = doc[i:i + 1600]
    low = block.lower()
    # says it steers, onto the axis
    assert 'steered' in low or 'steers' in low
    assert 'axis' in low
    # marks the deprecation and names the replacement / duplicate
    assert 'deprecated' in low
    assert 'paraxial' in low
    # and no longer claims the opposite
    assert 'without steering the beam' not in low
