"""FIX_RUNNER_OOM_2026_08_13 -- the v5.35.0 release-verify runner kill.

WHAT HAPPENED.  The v5.35.0 publish workflow's ``verify`` gate died twice, on
two different Python legs (3.12 and 3.10), at the SAME place: partway through
``tests/unit/test_niche_c1_consolidation.py``, with ``The runner has received a
shutdown signal``.  Not a timeout and not an assertion -- the kernel took the
runner service, which is what a hosted runner looks like when a test process
walks off the end of its RAM.

THE TEST IT DIED IN, and why it was invisible here.
``test_the_element_reports_the_measured_exit_na_and_the_aliased_power`` runs
``apply_real_lens_traced`` on a 2048^2 grid at ``ray_subsample=2``.  That is
2401^2 = 5,764,801 retained fit samples, and at the inverse map's degree 14
(P = 120 terms) ONE float64 design matrix is 5.53 GB.  Nothing bounded how many
of those were alive at once: the retired one-liners in ``_td_design`` /
``_td_design_grad`` / ``_Cheb2DEvaluator.__init__`` each held three or four
full copies (two fancy-index GATHERS plus their product), and the inverse-map
build ran its gradient designs with the design matrix still alive.  MEASURED
peak on this box, with ``psutil``'s AVAILABLE reading pinned to a runner-sized
6.5 GB: **32.9 GB**.  On the 128 GB dev box the same call simply took the
memory and passed, which is why every pre-merge gate was green.

WHAT CHANGED, in two layers:

* the design-matrix constructions are ROW-BLOCKED, which is a pure memory
  change -- elementwise products, no reduction, so no summation order moves and
  the bits are identical (tests 1-4 below, each against the retired expression
  recomputed here);
* ``build_inverse_map`` gained **GRAM**, a guard that prices the build against
  ``lumenairy.get_ram_budget()`` and REFUSES when it cannot fit, exactly as
  G1-G8 refuse.  Refusing keeps the shipped coarse-Newton + ``map_coordinates``
  path, so the refusal's output is byte-identical to ``TRACED_INVERSE_MAP =
  False`` -- asserted, not asserted-about (test 6).

Measured end to end at a pinned 6.5 GB budget: 32.9 GB -> 4.3 GB (test 8).
"""
from __future__ import annotations

import tracemalloc
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_imap as IM
from lumenairy.elements import _lens_traced as LT

_WL = 1.31e-6
_TKW = dict(on_undersample='silent', on_noncollimated='silent',
            on_aperture_beam='silent')


# ---------------------------------------------------------------------------
# helpers -- the retired expressions, recomputed here so the pins compare
# against the thing that shipped rather than against themselves
# ---------------------------------------------------------------------------
def _peak(fn, *a, _warm=None, **k):
    """``(result, peak_bytes)`` for one call.

    :mod:`tracemalloc` counts NumPy's allocator too (NumPy registers its own
    tracemalloc domain), so this is a DETERMINISTIC transient measurement --
    not an RSS sample that a garbage collector or an allocator arena can move
    under the test.

    ``_warm`` is an ``(args, kwargs)`` pair for ONE throwaway call on a tiny
    input, made BEFORE the window opens.  2026-08-15
    (``docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md`` S6 + S8.1, shape S4):
    tracemalloc counts EVERY allocation inside the window, and a module IMPORT
    is an allocation.
    The first :class:`~lumenairy.elements._lens_traced._Cheb2DEvaluator` call
    reaches the array-backend dispatcher, which first-imports ``cupy`` /
    ``cupyx`` / ``cupy_backends`` / ``cuda`` / ``ml_dtypes`` /
    ``_cython_3_1_4`` -- MEASURED 7.24 MB of traced heap on this box, more than
    the whole 5.06 MB margin the forward-fit pin below carries.  Whether that
    lands inside the window is a per-BUILD, per-ORDER fact (is CuPy installed?
    did an earlier test already touch the dispatcher?), which is exactly the
    thing a pin may not depend on: measured cold, the forward-fit window read
    315.7 MB against its 313.5 MB bar and FAILED on Windows py3.14 / np2.4.4 in
    a bare process, while the same source read 308.5 MB under ``pytest`` (where
    ``tests/conftest.py``'s leak guard happens to touch the dispatcher first)
    and 308.5 MB on WSL py3.12 / np2.5.1 (where CuPy is not installed at all).
    Warming moves that one-time import cost OUTSIDE the window, so the window
    measures arrays -- the thing these pins are about -- and reads the same
    number on every build.
    """
    if _warm is not None:
        _wa, _wk = _warm
        fn(*_wa, **_wk)
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        out = fn(*a, **k)
        _cur, pk = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return out, int(pk)


def _retired_td_design(ux, uy, degree, terms):
    """``_td_design`` exactly as it shipped through v5.35.0."""
    from numpy.polynomial.chebyshev import chebvander
    Vx = chebvander(np.asarray(ux, dtype=np.float64), degree)
    Vy = chebvander(np.asarray(uy, dtype=np.float64), degree)
    return Vx[:, terms[:, 0]] * Vy[:, terms[:, 1]]


def _retired_td_design_grad(ux, uy, degree, terms):
    """``_td_design_grad`` exactly as it shipped through v5.35.0."""
    from numpy.polynomial.chebyshev import chebvander
    Vx = chebvander(np.asarray(ux, dtype=np.float64), degree)
    Vy = chebvander(np.asarray(uy, dtype=np.float64), degree)
    Dx = IM._cheb_dvander(ux, degree)
    Dy = IM._cheb_dvander(uy, degree)
    return (Dx[:, terms[:, 0]] * Vy[:, terms[:, 1]],
            Vx[:, terms[:, 0]] * Dy[:, terms[:, 1]])


def _retired_cheb_coeffs(xs, ys, values, order, weights=None):
    """``_Cheb2DEvaluator.__init__``'s fit exactly as it shipped: the
    ``(Tu[K1] * Tv[K2]).reshape(n_terms, -1).T`` design, the same keep mask,
    the same solver."""
    mi = [(kx, ky) for kx in range(order + 1) for ky in range(order + 1 - kx)]
    K1 = np.asarray([m[0] for m in mi], dtype=np.int64)
    K2 = np.asarray([m[1] for m in mi], dtype=np.int64)
    xs_np = np.asarray(xs, dtype=np.float64)
    ys_np = np.asarray(ys, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    xmin, xmax = float(xs_np.min()), float(xs_np.max())
    ymin, ymax = float(ys_np.min()), float(ys_np.max())
    X, Y = np.meshgrid(xs_np, ys_np, indexing='ij')
    u = (2.0 * X - (xmin + xmax)) / (xmax - xmin)
    v = (2.0 * Y - (ymin + ymax)) / (ymax - ymin)
    Tu = LT._cheb_vand_2d(u, order, np)
    Tv = LT._cheb_vand_2d(v, order, np)
    A_full = (Tu[K1] * Tv[K2]).reshape(len(mi), -1).T
    flat = vals.ravel()
    finite = np.isfinite(flat)
    if weights is not None:
        w_flat = np.asarray(weights, dtype=np.float64).ravel()
        keep = finite & np.isfinite(w_flat) & (w_flat > 0.0)
        _all = bool(keep.all())
        A = np.ascontiguousarray(A_full if _all else A_full[keep, :])
        w_keep = w_flat if _all else w_flat[keep]
        A = A * w_keep[:, None]
        rhs = (flat if _all else flat[keep]) * w_keep
    elif finite.all():
        A, rhs = A_full, flat
    else:
        A, rhs = A_full[finite, :], flat[finite]
    return LT._solve_lstsq_thread_safe(A, rhs)


def _uv(n, seed=17):
    rng = np.random.default_rng(seed)
    return (rng.uniform(-1.0, 1.0, n), rng.uniform(-1.0, 1.0, n))


def _singlet(R1, R2, d, glass, ap, name):
    s = [{'radius': R1, 'glass_before': 'air', 'glass_after': glass,
          'conic': 0.0, 'radius_y': None, 'conic_y': None,
          'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
         {'radius': R2, 'glass_before': glass, 'glass_after': 'air',
          'conic': 0.0, 'radius_y': None, 'conic_y': None,
          'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]
    return {'name': name, 'aperture_diameter': ap, 'surfaces': s,
            'thicknesses': [d]}


def _gauss(n, dx, w):
    x = (np.arange(n) - n // 2) * dx
    return np.exp(-((x[None, :] ** 2 + x[:, None] ** 2) / w ** 2)
                  ).astype(np.complex128)


# ===========================================================================
# 1-2.  the blocked builders are BIT-identical to the ones they replaced
# ===========================================================================
@pytest.mark.parametrize('degree', [3, 6, 14])
def test_the_blocked_exit_design_is_bit_identical_to_the_retired_one_liner(
        degree, monkeypatch):
    """Same numbers, to the bit, across a chunk boundary.

    The chunk budget is forced small enough to guarantee SEVERAL blocks -- a
    blocking bug that dropped or duplicated a tail row would otherwise hide
    behind a single-block run at test sizes."""
    n = 5000
    ux, uy = _uv(n)
    terms = IM._td_terms(degree)
    monkeypatch.setattr(IM, '_IMAP_FIT_CHUNK_ENTRIES',
                        int(terms.shape[0]) * 700)
    blocks = list(IM._fit_row_blocks(n, int(terms.shape[0])))
    assert len(blocks) >= 6, blocks
    assert blocks[0][0] == 0 and blocks[-1][1] == n
    assert all(b[0] == a[1] for a, b in zip(blocks, blocks[1:])), blocks

    got = IM._td_design(ux, uy, degree, terms)
    ref = _retired_td_design(ux, uy, degree, terms)
    assert got.shape == ref.shape == (n, terms.shape[0])
    assert np.array_equal(got, ref), (
        'the blocked exit design moved bits: max |diff| = %.3e'
        % float(np.abs(got - ref).max()))


@pytest.mark.parametrize('degree', [4, 9])
def test_the_blocked_exit_gradient_design_is_bit_identical(degree,
                                                           monkeypatch):
    """Same, for the gradient designs the ``analytic_inverse`` Jacobian
    channel consumes -- the pair that carried the measured high-water mark."""
    n = 4096
    ux, uy = _uv(n, seed=5)
    terms = IM._td_terms(degree)
    monkeypatch.setattr(IM, '_IMAP_FIT_CHUNK_ENTRIES',
                        int(terms.shape[0]) * 500)
    Au, Av = IM._td_design_grad(ux, uy, degree, terms)
    Au_r, Av_r = _retired_td_design_grad(ux, uy, degree, terms)
    assert np.array_equal(Au, Au_r) and np.array_equal(Av, Av_r)


# ===========================================================================
# 3.  ...and they are the reason the transient collapsed
# ===========================================================================
def test_the_exit_design_transient_is_bounded_by_the_matrix_it_returns():
    """The whole point, measured: the blocked build's PEAK is the output plus
    block scratch, where the retired expression's peak was ~3x the output
    (two full gathers plus their product).

    The retired arm is measured here rather than quoted, so the pin
    demonstrates its own fail-before: if blocking regressed to the one-liner,
    the first assertion fails at the same numbers the second one passes."""
    n, degree = 300_000, 9
    terms = IM._td_terms(degree)
    P = int(terms.shape[0])
    out_b = n * P * 8
    ux, uy = _uv(n, seed=3)

    scratch = 3.0 * IM._IMAP_FIT_CHUNK_ENTRIES * 8
    # 2026-08-15 (``docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md`` S8.1, shape
    # S4): every window is WARMED, so it measures arrays, not first-imports (see
    # ``_peak``).  The bars are unchanged -- they are already derived from this
    # build's chunk constants -- but the MEASUREMENT is now build-free.
    # Measured warmed, IDENTICAL on Windows py3.14 / np2.4.4 and WSL py3.12 /
    # np2.5.1: pk_new = 283.406 MB against the 324.000 MB bar (132.000 MB
    # output + 192.000 MB block scratch), i.e. 12.5% BELOW the bar -- and
    # 2.15x ABOVE the 132.000 MB output alone, so the window is still holding
    # the block scratch it claims to and has not collapsed into a trivially
    # passing measurement.  Two-sided: the pin fails high if blocking regresses
    # (the retired arm below reads > 2.5x output) and fails low if the excess
    # ever stopped being a fixed block (the size-independence arm).  Cold, this
    # site happened to read 283.406 MB too -- it is not contaminated TODAY, but
    # it is the same window shape as the forward-fit pin that was, and nothing
    # stops a future first-import from landing in it.
    _wux, _wuy = _uv(8, seed=3)
    _warm_td = ((_wux, _wuy, degree, terms), {})
    _a, pk_new = _peak(IM._td_design, ux, uy, degree, terms, _warm=_warm_td)
    _b, pk_old = _peak(_retired_td_design, ux, uy, degree, terms,
                       _warm=_warm_td)
    assert pk_new < out_b + scratch, (
        'blocked design transient %.2f MB against a %.2f MB output + '
        '%.2f MB block scratch'
        % (pk_new / 1e6, out_b / 1e6, scratch / 1e6))
    assert pk_old > 2.5 * out_b, (
        'the retired expression no longer over-allocates, so this test can '
        'no longer see the fix: %.2f MB against %.2f MB'
        % (pk_old / 1e6, out_b / 1e6))

    # ...and the excess is FIXED, not proportional: that is what makes the
    # 5.53 GB design of the release-blocker call cost 5.53 GB and not 16.6.
    ux2, uy2 = _uv(2 * n, seed=4)
    _c, pk_2n = _peak(IM._td_design, ux2, uy2, degree, terms, _warm=_warm_td)
    excess_1, excess_2 = pk_new - out_b, pk_2n - 2 * out_b
    assert abs(excess_2 - excess_1) < 0.25 * excess_1, (
        'the blocked transient is not size-independent: %.1f MB excess at n, '
        '%.1f MB at 2n' % (excess_1 / 1e6, excess_2 / 1e6))


# ===========================================================================
# 4.  the forward Chebyshev fit: same bits, and the layout the SOLVE squares
# ===========================================================================
@pytest.mark.parametrize('weighted', [False, True])
def test_the_forward_fit_coefficients_are_bit_identical(weighted):
    """``_Cheb2DEvaluator`` builds its design C-contiguous now instead of as
    the transpose of a C-contiguous ``(n_terms, n)`` buffer.  That is only safe
    because ``_solve_lstsq_thread_safe`` opens with
    ``np.ascontiguousarray(A)`` -- it always squared a C-contiguous copy -- so
    the Gram it forms is the same array it always formed.  Asserted to the
    BIT against the retired construction, on both branches."""
    ax = np.linspace(-1.0e-3, 1.0e-3, 96)
    X, Y = np.meshgrid(ax, ax, indexing='ij')
    vals = (0.7 * X ** 2 - 1.3 * X * Y + 0.4 * Y ** 3
            + 2e-4 * np.cos(3.0e3 * X))
    w = None
    if weighted:
        w = np.exp(-((X / 6e-4) ** 2 + (Y / 6e-4) ** 2))
    ev = LT._Cheb2DEvaluator(ax, ax, vals, order=6, weights=w)
    ref = _retired_cheb_coeffs(ax, ax, vals, 6, weights=w)
    assert np.array_equal(np.asarray(ev.coeffs), np.asarray(ref)), (
        'the forward fit moved bits: max |diff| = %.3e'
        % float(np.abs(np.asarray(ev.coeffs) - np.asarray(ref)).max()))


def test_the_forward_fit_transient_is_bounded_by_its_design_matrix():
    """Same measurement as test 3, on the forward fit: the retired form held
    three full ``(n_terms, n)`` arrays plus the solver's own contiguous copy."""
    ax = np.linspace(-1.0e-3, 1.0e-3, 700)          # 490,000 samples
    X, Y = np.meshgrid(ax, ax, indexing='ij')
    vals = 0.5 * X ** 2 + 0.25 * Y ** 2
    n_terms = 28                                     # order 6, total degree
    out_b = ax.size ** 2 * n_terms * 8
    scratch = (3.0 * LT._CHEB_FIT_CHUNK_ENTRIES * 8
               + 3.0 * ax.size ** 2 * 8)             # + the Tu/Tv/u/v planes
    # 2026-08-15 (``docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md`` S6 + S8.1,
    # shape S4): THE site the warm-up exists for.  The two mounts' readings and
    # the fail-before are the S6 entry that opened this branch.
    #
    # The bar is unchanged (313.520 MB = 109.760 MB
    # design + 203.760 MB scratch, all derived from this build's chunk
    # constants); what changes is that the window no longer bills this pin for
    # the array-backend dispatcher's first-import.  Measured:
    #
    #     window          Windows py3.14/np2.4.4     WSL py3.12/np2.5.1
    #     cold (bare)     315.695 MB  -> FAILS       308.457 MB
    #     warmed          308.456 MB                 308.456 MB
    #
    # The cold Windows reading is 2.18 MB OVER the bar, and the 7.24 MB gap
    # between the two columns is exactly the cupy/cupyx/cuda/ml_dtypes import
    # -- a fact about which wheels the box has, not about how many arrays the
    # fit allocates.  Warmed, the two mounts agree to 1 kB.
    #
    # Two-sided headroom on the warmed number: 5.064 MB (1.62%) below the
    # 313.520 MB bar, and 2.81x ABOVE the 109.760 MB design matrix alone -- so
    # a regression that reintroduced even ONE extra full design copy
    # (+109.76 MB) fails the bar with 20x the margin, while a collapse to the
    # bare output would trip the retired-arm fail-before below.  1.62% is thin
    # in absolute terms but it is now a difference of two DERIVED array sizes
    # that read identically on both mounts, not a race against an import.
    _wax = np.linspace(-1.0e-3, 1.0e-3, 8)
    _wvals = 0.5 * _wax[:, None] ** 2 + 0.25 * _wax[None, :] ** 2
    _e, pk_new = _peak(LT._Cheb2DEvaluator, ax, ax, vals, order=6,
                       _warm=((_wax, _wax, _wvals), {'order': 2}))
    _c, pk_old = _peak(_retired_cheb_coeffs, ax, ax, vals, 6,
                       _warm=((_wax, _wax, _wvals, 2), {}))
    assert pk_new < out_b + scratch, (
        'forward-fit transient %.2f MB against a %.2f MB design + %.2f MB '
        'scratch' % (pk_new / 1e6, out_b / 1e6, scratch / 1e6))
    assert pk_old > 2.2 * out_b, (
        'the retired forward fit no longer over-allocates: %.2f vs %.2f MB'
        % (pk_old / 1e6, out_b / 1e6))


# ===========================================================================
# 5-6.  GRAM: the guard, and what refusing costs
# ===========================================================================
def _element_call(budget_b=None, flag=None, **over):
    """One ``apply_real_lens_traced`` at a pinned RAM budget, with the map's
    guard record returned."""
    presc = _singlet(0.030, -0.030, 3e-3, 'N-BK7', 9e-3, 'oom_pin')
    kw = dict(prescription=presc, wavelength=_WL, dx=16e-6,
              ray_subsample=4, n_workers=1, carrier=-0.06, **_TKW)
    kw.update(over)
    rec: dict = {}
    old_flag = IM.TRACED_INVERSE_MAP
    IM.inverse_map_cache_clear()
    try:
        if flag is not None:
            IM.TRACED_INVERSE_MAP = flag
        if budget_b is not None:
            _saved = IM._imap_ram_budget
            IM._imap_ram_budget = lambda: float(budget_b)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = np.asarray(la.apply_real_lens_traced(
                _gauss(384, 16e-6, 1.4e-3), _imap_out=rec, **kw))
    finally:
        if budget_b is not None:
            IM._imap_ram_budget = _saved
        IM.TRACED_INVERSE_MAP = old_flag
        IM.inverse_map_cache_clear()
    return out, rec


def test_gram_refuses_a_build_that_cannot_fit_the_box():
    """A box too small for the design matrix refuses at GRAM, names the
    numbers, and -- the fail-before -- the SAME call on a box that fits does
    not refuse."""
    _e_small, rec_small = _element_call(budget_b=32 * 1024 ** 2)
    assert rec_small.get('refused') == 'GRAM', rec_small
    assert 'set_max_ram' in rec_small['detail']
    assert rec_small['build_projected_gb'] > rec_small['build_budget_gb']

    _e_big, rec_big = _element_call(budget_b=512 * 1024 ** 3)
    assert rec_big.get('refused') != 'GRAM', rec_big
    assert rec_big['build_projected_gb'] <= rec_big['build_budget_gb']


def test_a_gram_refusal_is_byte_identical_to_the_fail_before_switch():
    """"Refuse, never degrade": the guard hands the caller the shipped
    coarse-Newton + ``map_coordinates`` path, which is exactly what
    ``TRACED_INVERSE_MAP = False`` hands it.  Byte-identical, or the guard is
    a third code path nobody validated."""
    E_gram, rec_gram = _element_call(budget_b=32 * 1024 ** 2)
    E_off, rec_off = _element_call(flag=False)
    assert rec_gram['refused'] == 'GRAM' and rec_off['engaged'] is False
    assert np.array_equal(E_gram, E_off), (
        'GRAM\'s fallback is not the fail-before path: max |diff| = %.3e'
        % float(np.abs(E_gram - E_off).max()))


def test_the_gram_projection_is_not_smaller_than_what_the_build_allocates():
    """The constant is MEASURED, and this is the measurement.  The build's two
    dominant transients are the design and the gradient pair; their peak must
    sit under the ``_IMAP_BUILD_PEAK_COPIES`` projection the guard prices
    with, or the guard would wave through a build that then OOMs."""
    n, degree = 200_000, 9
    terms = IM._td_terms(degree)
    P = int(terms.shape[0])
    ux, uy = _uv(n, seed=11)

    def _coexist(uxi=ux, uyi=uy):
        """The build's real high-water shape: the design matrix is ALIVE while
        the gradient pair is built (``_td_design_grad`` runs after the solve,
        with ``A`` still held for the residual report)."""
        A = IM._td_design(uxi, uyi, degree, terms)
        Au, Av = IM._td_design_grad(uxi, uyi, degree, terms)
        return float(A[0, 0] + Au[0, 0] + Av[0, 0])

    # 2026-08-15 (``docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md`` S8.1, shape
    # S4): warmed for the same reason as the two pins above -- an upper bar
    # measured with tracemalloc must not be able to bill a first-import to the
    # code under test.  ``_coexist`` takes
    # its sample vectors as defaulted arguments purely so a tiny warm-up call
    # can run the identical code path.  Measured warmed, both mounts: 438.678
    # MB against the guard's 544.000 MB projection (19.4% below) and 2.49x the
    # 176.000 MB two-copy floor the second arm requires -- unchanged from the
    # cold reading (438.679 MB), because this path never touches the array
    # dispatcher; the treatment is prophylactic, not a repair.
    _wux, _wuy = _uv(8, seed=11)
    _v, pk = _peak(_coexist, _warm=((_wux, _wuy), {}))
    projected = (IM._IMAP_BUILD_PEAK_COPIES * n * P * 8
                 + 3.0 * IM._IMAP_FIT_CHUNK_ENTRIES * 8)
    assert pk <= projected, (
        'the design + gradient high-water is %.1f MB against the guard\'s '
        '%.1f MB projection -- GRAM would wave through a build that OOMs'
        % (pk / 1e6, projected / 1e6))
    assert pk > 2.0 * n * P * 8, (
        'the measured high-water is below 2 copies, so this pin is no longer '
        'measuring the coexistence it claims to (%.1f MB)' % (pk / 1e6))


# ===========================================================================
# 7-8.  the call that killed the runner
# ===========================================================================
def test_the_release_blocker_call_fits_a_runner_sized_box():
    """THE regression pin, at the geometry that took the runner down:
    ``apply_real_lens_traced`` on 2048^2 at ``ray_subsample=2`` -- 5,764,801
    retained samples, a 5.53 GB design matrix at degree 14.

    Budget pinned to 6.5 GB (a hosted runner with the usual system share
    already gone).  Measured before this fix: 32.9 GB of transient, i.e. the
    kernel takes the process.  The bar below is 5.0 GB: comfortably above the
    ~2.3 GB this now allocates and comfortably below anything that could
    survive on a small box."""
    _sav = IM._imap_ram_budget
    IM._imap_ram_budget = lambda: 6.5 * 1024 ** 3
    IM.inverse_map_cache_clear()
    try:
        presc = _singlet(8.0e-3, -8.0e-3, 1.5e-3, 'N-BK7', 3.2e-3, 'c1-oom')
        E = _gauss(2048, 1.0e-6, 0.40e-3)

        def _call():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return np.asarray(la.apply_real_lens_traced(
                    E, prescription=presc, wavelength=_WL, dx=1.0e-6,
                    carrier=np.inf, ray_subsample=2, n_workers=1,
                    min_coarse_samples_per_aperture=0, **_TKW))

        out, pk = _peak(_call)
    finally:
        IM._imap_ram_budget = _sav
        IM.inverse_map_cache_clear()
    assert out.shape == (2048, 2048)
    assert np.isfinite(out).all()
    assert pk < 5.0e9, (
        'the exit-NA call allocates %.2f GB of transient on a 6.5 GB box; '
        'the v5.35.0 runner was killed at 32.9 GB' % (pk / 1e9))


def test_the_dev_box_still_gets_the_map_on_the_same_call():
    """...and the guard is a SMALL-box guard, not a retreat: the ordinary
    element call on a box with room still engages the map.  Without this the
    fix would read as "turn C15 off in CI" rather than "price it"."""
    _e, rec = _element_call(budget_b=512 * 1024 ** 3)
    assert rec.get('engaged') is True, rec
    assert rec.get('refused') is None, rec
