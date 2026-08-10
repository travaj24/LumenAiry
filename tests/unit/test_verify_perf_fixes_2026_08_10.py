# Pins for the six defects an independent verification found on
# perf/traced-hotpath @ 7890b7d -- docs/audits/VERIFY_PERF_BRANCH_2026_08_10.md,
# fixed in docs/audits/FIX_VERIFY_PERF_2026_08_10.md.
#
# Every test here is written so that it FAILS on the pre-fix code.  Three of
# them would have failed on ONE platform only before the fix (D1 splits on
# whether ``np.dtype('c32')`` exists), so each is expressed as arithmetic that
# is identical on both arms rather than as a platform-conditional pin.
#
# Suite membership: these are unit-scale (no chain, no pool, no fine grid)
# except ``test_apply_real_lens_leaves_nothing_reachable_through_numexpr``,
# which runs ONE 1024^2 ``apply_real_lens`` because the retention it pins is a
# property of the shipped numexpr route and nothing smaller reaches it (the
# route has a 1 Mi-element size gate).
import gc
import inspect
import weakref

import numpy as np
import pytest

from lumenairy import memory as M
from lumenairy.analysis import beam_stats as BS
from lumenairy.analysis import zernike as Z
from lumenairy.elements import _lens_real as LR
from lumenairy.propagators import _bluestein as BL
from lumenairy.propagators import carrier as C
from lumenairy.propagators import fft_infra as FI

_WL = 1.31e-6


# ===========================================================================
# D1 -- ``estimate_asm_memory`` built the WRONG dtype code for the plan-buffer
#       branch, and failed differently on each platform.
# ===========================================================================
# ``cb`` is ALREADY the complex itemsize, so ``np.dtype(f'c{2 * cb}')`` asked
# for a dtype of twice the element size.  On Linux that is ``complex256`` and
# the workspace was priced at 32 B/element -- a 2.147 GB phantom that fails the
# 2 GB cap the true 1.074 GB workspace passes, so the estimator reported ONE
# resident buffer per plan key where the runtime builds TWO: a 43 % UNDER-
# estimate of a PUBLIC, documented BOUND at N = 8192 complex128, the shape
# ``fft_infra`` calls "this library's most common large shape".  On Windows
# ``'c32'`` raises and the except swallowed it into ``n_bufs = 2`` -- a no-op
# for complex128, and via the perfectly valid ``'c16'`` a 9.66 GB UNDER-
# estimate for complex64 at N = 12288.
#
# The A-6 pins sample N = 512 / 1024, where both codes return 2, which is why
# nothing saw it.  These pins sample where the predicate actually decides.


def _asm_terms(N, dtype, keys):
    """The estimator's own formula with the TRUE buffer count substituted."""
    cb = int(np.dtype(dtype).itemsize)
    npix = N * N
    n_bufs = int(FI._plan_entry_n_bufs((N, N), np.dtype(dtype)))
    return int(M._ASM_COMPLEX_ARRAYS * cb * npix
               + M._ASM_F64_GRID_ARRAYS * 8 * npix
               + keys * n_bufs * cb * npix
               + M._ASM_FIRST_CALL_FIXED_BYTES), n_bufs


@pytest.mark.parametrize('N,dtype,keys', [
    (512, 'complex64', 2),        # A-6 pin territory: n_bufs = 2 both ways
    (1024, 'complex128', 2),      # A-6 pin territory
    (8192, 'complex128', 8),      # the shape D1 under-priced on Linux
    (11000, 'complex128', 8),
    (12288, 'complex64', 8),      # the shape D1 under-priced on WINDOWS
    (16384, 'complex128', 2),     # true n_bufs = 1: the branch must BIND
])
def test_the_plan_buffer_term_uses_the_callers_dtype(N, dtype, keys):
    """The estimate must equal its own formula evaluated with the buffer count
    ``fft_infra`` will actually build for the CALLER's dtype -- exactly, to the
    byte, on every platform.

    Pre-fix this is False at (8192, complex128) on Linux and at
    (12288, complex64) everywhere, in the UNDER-estimating direction.
    """
    want, n_bufs = _asm_terms(N, dtype, keys)
    got = M.estimate_asm_memory(N, dtype, plan_cache_keys=keys)
    assert got == want, (
        f"N={N} {dtype} keys={keys}: estimate {got / 1e9:.3f} GB vs "
        f"{want / 1e9:.3f} GB at the TRUE n_bufs={n_bufs} -- the plan-buffer "
        f"branch is pricing a dtype the caller did not ask for")


def test_estimate_asm_memory_pins_the_8192_complex128_value():
    """The absolute number D1 got wrong.  8 plan keys x 2 resident workspaces
    x 16 B x 8192^2 is 17.18 GB of the total on its own; the defect reported
    one workspace and lost 8.59 GB of it."""
    got = M.estimate_asm_memory(8192, 'complex128', plan_cache_keys=8)
    assert abs(got / 1e9 - 19.762) < 0.001, (
        f"{got / 1e9:.3f} GB; the pre-fix Linux value was 11.172 GB")
    # ...and the same shape at complex64, which is where Windows lost it.
    got64 = M.estimate_asm_memory(12288, 'complex64', plan_cache_keys=8)
    assert abs(got64 / 1e9 - 22.648) < 0.001, (
        f"{got64 / 1e9:.3f} GB; the pre-fix value was 12.984 GB")


def test_the_plan_buffer_branch_is_live_and_not_a_swallowed_exception():
    """Move the cap so the SAME shape flips 2 -> 1 workspace: the estimate must
    move by exactly one workspace per key.

    This is the test that fails on Windows pre-fix for complex128, where the
    ``TypeError`` on ``'c32'`` made the whole branch a constant ``n_bufs = 2``
    -- the estimate did not move at all.
    """
    N, keys, cb = 8192, 4, 16
    one_ws = N * N * cb
    old = FI.get_fft_plan_max_bytes_per_buffer()
    try:
        FI.set_fft_plan_max_bytes_per_buffer(float('inf'))
        hi = M.estimate_asm_memory(N, 'complex128', plan_cache_keys=keys)
        assert FI._plan_entry_n_bufs((N, N), np.dtype('complex128')) == 2
        FI.set_fft_plan_max_bytes_per_buffer(one_ws // 2)
        lo = M.estimate_asm_memory(N, 'complex128', plan_cache_keys=keys)
        assert FI._plan_entry_n_bufs((N, N), np.dtype('complex128')) == 1
    finally:
        FI.set_fft_plan_max_bytes_per_buffer(old)
    assert hi - lo == keys * one_ws, (
        f"the estimate moved {(hi - lo) / 1e9:.3f} GB where one workspace per "
        f"key is {keys * one_ws / 1e9:.3f} GB -- the branch is not reading the "
        f"predicate it claims to")


def test_the_doubled_dtype_code_is_gone_from_the_source():
    """Named directly, because the defect is a plausible re-typing: ``cb`` is
    an itemsize, so ``c{2 * cb}`` is always wrong and ``c{cb}`` is only
    accidentally right."""
    src = inspect.getsource(M.estimate_asm_memory)
    code = ' '.join(ln.split('#')[0] for ln in src.splitlines())
    assert "c{2 * cb}" not in code and "c{2*cb}" not in code, (
        "the plan-buffer branch is doubling the complex itemsize again")
    assert '_plan_entry_n_bufs((N, N), np.dtype(complex_dtype))' in code


# ===========================================================================
# D2 -- ``del E_analytic`` did not free: numexpr retained the ``out=`` array.
# ===========================================================================
# ``numexpr.evaluate`` is ``validate`` + ``re_evaluate``, and ``validate``
# parks its kwargs -- ``out`` included -- in ``necompiler._numexpr_last`` so
# the replay has something to read.  ``apply_real_lens`` is the last numexpr
# caller in ``apply_real_lens_traced``, so the field it returns stayed
# reachable to the end of the CHAIN: MEASURED alive at the element's return,
# at the fine leg's return and at chain end, on the ray_density + remap +
# lattice route design 121 ships.  4.295 GB at n_fine = 16384.


def _singlet_prescription():
    from lumenairy import get_glass_index
    n = float(get_glass_index('N-BK7', _WL))
    def _s(radius, gb, ga, conic):
        return {'radius': float(radius), 'glass_before': gb,
                'glass_after': ga, 'conic': float(conic), 'radius_y': None,
                'conic_y': None, 'aspheric_coeffs': None,
                'aspheric_coeffs_y': None}
    return {'name': 'D6 fast singlet', 'aperture_diameter': 3.40e-3,
            'thicknesses': [1.5e-3],
            'surfaces': [_s(np.inf, 'air', 'N-BK7', 0.0),
                         _s(-(n - 1.0) * 3.00e-3, 'N-BK7', 'air', -n * n)]}


def _numexpr_or_skip():
    if not (LR.NUMEXPR_AVAILABLE and LR._ensure_numexpr_loaded()):
        pytest.skip('numexpr is not installed; the retention cannot occur')


def test_apply_real_lens_leaves_nothing_reachable_through_numexpr():
    """Run the real thing at a size that reaches the numexpr route, then drop
    the only name for the result: it must actually die.

    Pre-fix the weakref survives ``gc.collect()`` and its referrer is
    numexpr's 4-key kwargs dict (``out``, ``order``, ``casting``,
    ``ex_uses_vml``).
    """
    _numexpr_or_skip()
    N = 1024                                   # 1 Mi elements = the size gate
    assert N * N >= LR._NUMEXPR_MIN_SIZE
    dx = 6.0e-3 / N
    x = (np.arange(N) - N // 2) * dx
    E = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
               / (0.60e-3 ** 2)).astype(np.complex128)
    out = LR.apply_real_lens(E, prescription=_singlet_prescription(),
                             wavelength=_WL, dx=dx, sag_chunk_rows=0)
    assert out.size == N * N and np.isfinite(out).all()
    wr = weakref.ref(out)
    # The referrer form is informative where CPython tracks numexpr's kwargs
    # dict (py3.14/Windows) and vacuous where it does not (py3.12/Linux -- the
    # dict holds no GC-tracked values); the LIVENESS check below is the one
    # that binds on both, so both are asserted.
    referrers = [type(r).__name__ for r in gc.get_referrers(out)
                 if type(r).__name__ == 'dict' and 'ex_uses_vml' in r]
    assert not referrers, (
        "numexpr still holds the returned field as its last out= array")
    del out
    gc.collect()
    assert wr() is None, (
        "apply_real_lens's output survived its last reference -- something "
        "outside the caller's frame is holding a full grid")


def test_no_module_global_retains_a_full_grid_after_the_lens():
    """The census that credits ``-16.25 float64 grid equivalents`` sums
    ``f_locals``, so it is blind to an object whose NAME is gone.  This is the
    complementary check: after the element runs, no module global anywhere in
    the process may hold a grid-sized array."""
    _numexpr_or_skip()
    import sys
    N = 1024
    dx = 6.0e-3 / N
    x = (np.arange(N) - N // 2) * dx
    E = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
               / (0.60e-3 ** 2)).astype(np.complex128)
    out = LR.apply_real_lens(E, prescription=_singlet_prescription(),
                             wavelength=_WL, dx=dx, sag_chunk_rows=0)
    # ...the direct form first: no referrer of the returned field may BE a
    # module's __dict__ (gc.get_referrers, which is what a census cannot do).
    mod_dicts = {id(m.__dict__) for m in list(sys.modules.values())
                 if getattr(m, '__dict__', None) is not None}
    culprits = [r for r in gc.get_referrers(out)
                if isinstance(r, dict) and id(r) in mod_dicts]
    assert not culprits, (
        'a module global holds the field apply_real_lens just returned')
    del out, E, culprits
    gc.collect()
    held = []
    for name, mod in list(sys.modules.items()):
        d = getattr(mod, '__dict__', None)
        if not isinstance(d, dict):
            continue
        for k, v in list(d.items()):
            if isinstance(v, np.ndarray) and v.nbytes >= (N * N * 8):
                held.append(f'{name}.{k} ({v.nbytes / 1e6:.1f} MB)')
    assert not held, f"module globals retain grid-sized arrays: {held}"


def test_every_numexpr_out_site_drops_the_retention():
    """Structural, because the drain is easy to omit on a new site: each
    ``out=`` evaluate in this module must be followed by the drain."""
    src = inspect.getsource(LR)
    n_out = src.count('out=E,') + src.count('out=Eb)') + src.count('out=_Eb,')
    assert n_out == 3, (
        f'the numexpr out= site count changed ({n_out} != 3) -- add the drain '
        f'to the new one and update this pin')
    assert src.count('_drop_numexpr_out_retention()') >= n_out + 1, (
        'a numexpr out= site is not followed by _drop_numexpr_out_retention()')
    # ...and the drain must CLEAR the record, not merely unbind the
    # thread-local attribute: since numexpr 2.11 the payload lives in a
    # contextvars ContextVar, and ``del _numexpr_last.l`` leaves it reachable
    # (MEASURED: the weakref probe still read STILL ALIVE).
    dsrc = inspect.getsource(LR._drop_numexpr_out_retention)
    assert '.clear()' in dsrc and 'del _nc._numexpr_last.l' not in dsrc


def test_the_drain_actually_releases_the_out_array():
    """The drain in isolation, against numexpr directly.

    LIVENESS, not referrers.  ``gc.get_referrers`` cannot see this retention
    on every build: numexpr's kwargs dict holds only strings, a bool and a
    numeric ndarray, none of which are GC-tracked, so CPython leaves the dict
    itself untracked and ``get_referrers`` never reports it (MEASURED:
    ``gc.is_tracked(kwargs)`` is False on py3.12/Linux and the dict is
    invisible there, while it IS reported on py3.14/Windows).  A weakref sees
    the retention on both, which is also why the verifier's own probe used
    one.

    ``out`` is a SEPARATE buffer from the input, matching the shipped call
    where ``out=E`` is the field the caller then ``del``s.
    """
    _numexpr_or_skip()
    import numexpr
    a = np.zeros((1 << 10, 1 << 10), dtype=np.complex128)
    out = np.empty_like(a)
    numexpr.evaluate('a * 2', local_dict={'a': a}, out=out)
    wr = weakref.ref(out)
    del out
    gc.collect()
    # Whether numexpr retains out= is a LIBRARY-VERSION fact (>=2.11
    # ContextDict retains; some CI builds do not).  The load-bearing
    # invariant is unconditional: AFTER the drain nothing retains.  The
    # release demonstration runs only where the retention exists.
    retained = wr() is not None
    LR._drop_numexpr_out_retention()
    gc.collect()
    assert wr() is None, (
        'the drain did not release numexpr\'s reference to the out= array'
        if retained else
        'no retention on this numexpr build, yet the array is somehow '
        'alive after the drain -- a new retention path')


# ===========================================================================
# D3 -- an over-cap chirp-kernel entry was np.copy'd BEFORE it was rejected.
# ===========================================================================
# The byte cap converted a 4.86 GB retention into a 4.86 GB TRANSIENT at the
# window_factor = 7 geometry (L = 17424) its own comment works through -- on
# the run whose peak is the thing being defended.  The sibling this was
# modelled on, ``fft_infra._h_cache_store``, checks ``_entry_bytes(H)`` first
# and never copies.


def test_an_over_cap_chirp_entry_is_never_allocated():
    """Retention is 0 either way; only the TRACED PEAK separates the two
    orderings, so that is what this measures."""
    import tracemalloc
    BL._clear_h_fft_cache()
    L = 1024                                   # 16.8 MB, well over a 1 B cap
    H = np.zeros((L, L), dtype=np.complex128)
    old = BL._H_FFT_CACHE_MAX_BYTES_PER_ENTRY
    try:
        BL._H_FFT_CACHE_MAX_BYTES_PER_ENTRY = 1
        tracemalloc.start()
        base, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        BL._h_fft_cache_store(('probe', L), H)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        BL._H_FFT_CACHE_MAX_BYTES_PER_ENTRY = old
        BL._clear_h_fft_cache()
    assert BL._h_fft_cache_bytes() == 0, 'an over-cap entry was retained'
    assert peak - base < H.nbytes // 4, (
        f'rejecting an over-cap entry still allocated '
        f'{(peak - base) / 1e6:.3f} MB of a {H.nbytes / 1e6:.3f} MB entry -- '
        f'the copy is happening before the size test again')


def test_an_under_cap_chirp_entry_is_still_copied_and_stored():
    """The cap must not have cost the cache its reason to exist: an entry that
    fits is stored, byte-identical, and DECOUPLED from the caller's buffer
    (the pyFFTW double buffer hands out its own workspace)."""
    BL._clear_h_fft_cache()
    H = np.arange(64 * 64, dtype=np.complex128).reshape(64, 64)
    BL._h_fft_cache_store(('under', 64), H)
    got = BL._H_FFT_CACHE[('under', 64)]
    assert np.array_equal(got, H)
    assert got is not H and not np.shares_memory(got, H)
    assert BL._h_fft_cache_bytes() == H.nbytes
    BL._clear_h_fft_cache()


def test_the_chirp_store_tests_size_before_it_copies():
    """Structural: the sibling's ordering, in the sibling's words."""
    src = inspect.getsource(BL._h_fft_cache_store)
    i_cap = src.index('_H_FFT_CACHE_MAX_BYTES_PER_ENTRY')
    i_copy = src.index('np.copy(')
    assert i_cap < i_copy, (
        'the per-entry cap is tested AFTER the np.copy -- the cap then avoids '
        'a retention by paying for it')


# ===========================================================================
# D4 / D5 -- the clamp's cost model.
# ===========================================================================
# The measured rows and the 1.0-1.5x bound live in
# tests/unit/test_niche_d8_congruence_workers.py, which owns the clamp.  What
# is pinned HERE is the property that made D4 reportable in the first place:
# the model is an ENVELOPE over BOTH legs, and the two floors are different
# measured quantities rather than one constant applied outside its envelope.


def test_the_paraxial_leg_is_not_charged_the_exact_legs_floor():
    """``_FINE_GRID_BASE_BYTES``'s own note says it is a design-121-CLASS
    EXACT-leg figure.  Charging it to a worker that builds no fine grid took a
    16 GB-free box from 21 approved workers to ONE, against a MEASURED
    0.44-1.17 GB paraxial worker."""
    assert C._PARAXIAL_BASE_BYTES > 0, 'the paraxial floor is meant to be measured'
    assert C._PARAXIAL_BASE_BYTES < C._FINE_GRID_BASE_BYTES, (
        'the paraxial floor is not smaller than the exact leg\'s; the whole '
        'finding is that they are different quantities')
    n_px = 1024 * 1024
    par = C._fine_grid_peak_bytes(0, n_px=n_px)
    assert par == pytest.approx(
        C._MULTI_WORKER_GRID_FACTOR * n_px * 16.0 + C._PARAXIAL_BASE_BYTES)
    # ...and the exact leg still pays its own floor, at the same n_px.
    exact = C._fine_grid_peak_bytes(4096, n_px=n_px)
    assert exact - C._FINE_GRID_WORK_ARRAYS * 16.0 * 4096 ** 2 \
        - C._MULTI_WORKER_GRID_FACTOR * n_px * 16.0 == \
        pytest.approx(C._FINE_GRID_BASE_BYTES)


def test_the_paraxial_price_leaves_a_small_box_more_than_one_worker():
    """The regression D5 names, expressed as arithmetic rather than as a call
    into ``_multi_resolve_workers`` (whose answer depends on live free
    memory, which would make this test a report on what else the box is
    doing)."""
    reserve = 8e9
    for free, n_px, floor in ((16e9, 128 * 128, 4), (16e9, 1024 * 1024, 3)):
        pw = C._fine_grid_peak_bytes(0, n_px=n_px)
        allowed = int(max(1, (free - reserve) // pw))
        assert allowed >= floor, (
            f'a paraxial run with {free / 1e9:.0f} GB free and a '
            f'{n_px} px input gets {allowed} worker(s) at '
            f'{pw / 1e9:.3f} GB each')


def test_the_grid_ceiling_still_ignores_both_floors():
    """``frac`` IS the ceiling's allowance for everything that is not the fine
    grid.  Charging either floor there would double-count it."""
    src = inspect.getsource(C._fine_grid_ceiling)
    body = src.split('"""')[2]              # after the docstring
    body = ' '.join(ln.split('#')[0] for ln in body.splitlines())
    assert '_FINE_GRID_BASE_BYTES' not in body
    assert '_PARAXIAL_BASE_BYTES' not in body


# ===========================================================================
# D6 -- three numbers in comments/docstrings that disagreed with the code.
# ===========================================================================
def test_the_plan_buffer_cap_binds_where_the_comment_now_says():
    """The comment said 11586, which is the 2 GiB crossover; the constant is
    decimal 2e9 and binds at 11181."""
    dt = np.dtype('complex128')
    old = FI.get_fft_plan_max_bytes_per_buffer()
    try:
        FI.set_fft_plan_max_bytes_per_buffer(2_000_000_000)
        assert FI._plan_entry_n_bufs((11180, 11180), dt) == 2
        assert FI._plan_entry_n_bufs((11181, 11181), dt) == 1
    finally:
        FI.set_fft_plan_max_bytes_per_buffer(old)
    src = inspect.getsource(FI)
    head = src[:src.index('_PYFFTW_PLAN_MAX_BYTES_PER_BUFFER = ')]
    assert 'N >= 11181' in head, 'the cap comment does not name 11181'


def test_the_ram_budget_docstring_quotes_the_shipped_work_array_count():
    """It read ``= 16`` through two re-measurements of the constant."""
    doc = C.carrier_referenced_exact_focus_readout.__doc__ or ''
    assert f'= {C._FINE_GRID_WORK_ARRAYS} complex128 arrays' in doc, (
        'the ram_budget docstring quotes a stale _FINE_GRID_WORK_ARRAYS')


# ===========================================================================
# SIBLING SWEEP -- caches with a COUNT cap and no BYTE cap (VERIFY sec 5, B).
# ===========================================================================
# ``beam_stats._MESHGRID_CACHE`` holds a TUPLE OF TWO full (Ny, Nx) float64
# grids behind a count cap of 8: 34.4 GB at N = 16384, in a module global, for
# the life of the process.  ``zernike._ZERNIKE_BASIS_CACHE`` holds an
# (n_modes, Npix) float64 matrix behind a count cap of 32: 4.8 GB per entry at
# 36 modes on a 4096^2 pupil.  Both now carry the ``fft_infra._H_CACHE``
# per-entry + total byte caps.


def test_the_meshgrid_cache_refuses_an_over_cap_entry_but_still_returns_it():
    BS.clear_meshgrid_cache()
    old = BS._MESHGRID_CACHE_MAX_BYTES_PER_ENTRY
    try:
        BS._MESHGRID_CACHE_MAX_BYTES_PER_ENTRY = 1
        X, Y = BS._centered_meshgrid(np, 64, 64, 1.0, 1.0)
        assert X.shape == (64, 64) and Y.shape == (64, 64)
        assert BS.meshgrid_cache_bytes() == 0
        # value is unchanged by the cap
        xr = (np.arange(64) - 64 / 2) * 1.0
        Xr, Yr = np.meshgrid(xr, xr)
        assert np.array_equal(X, Xr) and np.array_equal(Y, Yr)
    finally:
        BS._MESHGRID_CACHE_MAX_BYTES_PER_ENTRY = old
        BS.clear_meshgrid_cache()


def test_the_meshgrid_cache_total_byte_cap_evicts():
    BS.clear_meshgrid_cache()
    old_t = BS._MESHGRID_CACHE_MAX_TOTAL_BYTES
    try:
        one = 2 * 64 * 64 * 8
        BS._MESHGRID_CACHE_MAX_TOTAL_BYTES = 2 * one + one // 2
        for k in range(6):
            BS._centered_meshgrid(np, 64, 64, 1.0 + k, 1.0)
        assert len(BS._MESHGRID_CACHE) <= 2, len(BS._MESHGRID_CACHE)
        assert BS.meshgrid_cache_bytes() <= BS._MESHGRID_CACHE_MAX_TOTAL_BYTES
        # ...and the cache still HITS inside the bound
        a = BS._centered_meshgrid(np, 64, 64, 6.0, 1.0)
        b = BS._centered_meshgrid(np, 64, 64, 6.0, 1.0)
        assert a[0] is b[0]
    finally:
        BS._MESHGRID_CACHE_MAX_TOTAL_BYTES = old_t
        BS.clear_meshgrid_cache()


def test_the_meshgrid_cache_keeps_its_count_cap_and_its_value():
    """The byte cap must not have changed what the cache RETURNS, nor the
    count bound that was already there."""
    BS.clear_meshgrid_cache()
    try:
        for k in range(BS._MESHGRID_CACHE_SIZE + 4):
            BS._centered_meshgrid(np, 32, 32, 1.0 + k, 1.0)
        assert len(BS._MESHGRID_CACHE) == BS._MESHGRID_CACHE_SIZE
    finally:
        BS.clear_meshgrid_cache()


def test_the_zernike_basis_cache_refuses_an_over_cap_entry_but_still_returns_it():
    Z.clear_zernike_basis_cache()
    old = Z._ZERNIKE_BASIS_CACHE_MAX_BYTES_PER_ENTRY
    try:
        n = 32
        xs = np.linspace(-1.0, 1.0, n)
        X, Y = np.meshgrid(xs, xs)
        ref, mref = Z.zernike_basis_matrix(6, X, Y, 1.0)
        Z.clear_zernike_basis_cache()
        Z._ZERNIKE_BASIS_CACHE_MAX_BYTES_PER_ENTRY = 1
        basis, mask = Z.zernike_basis_matrix(6, X, Y, 1.0)
        assert Z.zernike_basis_cache_bytes() == 0
        assert len(Z._ZERNIKE_BASIS_CACHE) == 0
        assert np.array_equal(basis, ref) and np.array_equal(mask, mref)
    finally:
        Z._ZERNIKE_BASIS_CACHE_MAX_BYTES_PER_ENTRY = old
        Z.clear_zernike_basis_cache()


def test_the_zernike_basis_cache_total_byte_cap_evicts():
    Z.clear_zernike_basis_cache()
    old_t = Z._ZERNIKE_BASIS_CACHE_MAX_TOTAL_BYTES
    try:
        n = 24
        xs = np.linspace(-1.0, 1.0, n)
        X, Y = np.meshgrid(xs, xs)
        b, m = Z.zernike_basis_matrix(6, X, Y, 1.0)
        one = b.nbytes + m.nbytes
        Z.clear_zernike_basis_cache()
        Z._ZERNIKE_BASIS_CACHE_MAX_TOTAL_BYTES = 2 * one + one // 2
        for r in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5):
            Z.zernike_basis_matrix(6, X, Y, r)
        assert len(Z._ZERNIKE_BASIS_CACHE) <= 2, len(Z._ZERNIKE_BASIS_CACHE)
        assert Z.zernike_basis_cache_bytes() <= \
            Z._ZERNIKE_BASIS_CACHE_MAX_TOTAL_BYTES
    finally:
        Z._ZERNIKE_BASIS_CACHE_MAX_TOTAL_BYTES = old_t
        Z.clear_zernike_basis_cache()


def test_both_sibling_caches_carry_the_libraries_own_cap_values():
    """Same policy, same numbers, as ``fft_infra._H_CACHE`` and
    ``_bluestein._H_FFT_CACHE``: a new policy per cache is how they drift."""
    for per, tot in ((BS._MESHGRID_CACHE_MAX_BYTES_PER_ENTRY,
                      BS._MESHGRID_CACHE_MAX_TOTAL_BYTES),
                     (Z._ZERNIKE_BASIS_CACHE_MAX_BYTES_PER_ENTRY,
                      Z._ZERNIKE_BASIS_CACHE_MAX_TOTAL_BYTES)):
        assert per == BL._H_FFT_CACHE_MAX_BYTES_PER_ENTRY
        assert tot == BL._H_FFT_CACHE_MAX_TOTAL_BYTES


# ===========================================================================
# SIBLING SWEEP -- harness scripts (VERIFY sec 5, class A).
# ===========================================================================
def test_capstone_stage_b_is_import_safe_and_blanket_free():
    """``capstone_stageB.py`` runs ``focus_scan_121.py`` under
    ``run_name='__main__'``, i.e. it BECOMES the runner's ``__main__``, and
    that runner asks for ``n_workers=8``.  Unguarded,
    ``_script_has_main_guard`` inspected THIS file, found no guard and forced
    the Newton pool serial -- the capstone measured a knob that was never
    applied."""
    import ast
    import os

    from lumenairy.elements._lens_traced import _script_has_main_guard
    here = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'validation',
        'repro_traced_carrier_121'))
    path = os.path.join(here, 'capstone_stageB.py')
    if not os.path.exists(path):        # pragma: no cover - trimmed checkout
        pytest.skip('validation/ is not present in this checkout')
    assert _script_has_main_guard(path), (
        'capstone_stageB.py has no top-level __main__ guard')
    with open(path, encoding='utf-8', errors='replace') as fh:
        tree = ast.parse(fh.read())
    blanket = [n for n in ast.walk(tree)
               if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Attribute)
               and n.func.attr in ('filterwarnings', 'simplefilter')
               and len(n.args) == 1 and not n.keywords
               and isinstance(n.args[0], ast.Constant)
               and n.args[0].value == 'ignore']
    assert not blanket, (
        'capstone_stageB.py carries a blanket warnings filter; the guard '
        'output it exists to read would be swallowed')
