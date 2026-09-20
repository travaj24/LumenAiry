"""VERIFY-WP-C4 -- decision tests for the gaps the verification closed.

Companion to ``tests/unit/test_c4_mft_direct_default.py``, which gates the
shipped rule.  This file gates the things the verification MEASURED that that
file does not see.  Evidence:
``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-C4.md``
and ``validation/probe_verify_c4/``.

WHAT EACH ID IS FOR, and why it is written the way it is:

1.  **The anisotropy exposure (V-C4-D1).**  ``_MFT_DIRECT_MAX_RATIO`` was
    derived from a ladder of SQUARE shapes.  On a THIN input -- one axis long,
    the other short -- both output/input ratios can sit at 1/32 while the
    dense route is 1.1x to 9.7x SLOWER than the faster chirp-Z fallback, on
    BOTH builds and under THREE timing instruments -- interleaved, blocked,
    and fully single-threaded (measured 2026-09-20, worst readings in
    ``_MEASURED_SLOWER`` below).  A ratio cannot see this, because
    the quantity that decides it is the dense route's multiply-adds PER
    transcendental kernel entry, and that is not a function of the two ratios.
    The ids here gate the predictor and bound the exposure, and they are
    written so that ADDING the missing guard does not falsify them.

2.  **A y/x transposition at the call sites (V-C4-D4, WIDENED).**  The
    shipped file DOES catch one -- MEASURED 2026-09-20,
    ``validation/probe_verify_c4/v4_mutations.py`` mutation
    ``swapped_axis_arguments``, refused by
    ``test_auto_returns_the_route_the_rule_names_on_the_dense_side`` on all
    four parametrizations -- but on the strength of ONE ``_DENSE_SIDE`` entry,
    ``(512, 1024, 16, 32)``, whose two ratios are equal at the boundary so
    that exchanging the outputs takes it to ``(1/16, 1/64)``.  Nothing in that
    file states the property, so the coverage is one list edit away from
    disappearing.  The ids here drive four such shapes on both primitives and
    assert the premise -- that a partial swap changes the answer at all -- as
    its own claim.

3.  **The way back, at the ENTRY POINTS (V-C4-D2).**  The shipped file proves
    the way back on the PRIMITIVE.  Seven public entry points reach the rule
    without exposing a route keyword at all.

4.  **The warning, counted (V-C4-D5).**  The shipped file gates which ROUTE
    warns.  It does not gate how MANY times, which is what a guard moved to a
    new place can get wrong.

5.  **The bar's derivation (V-C4-D6).**  ``_phase_term_ratio`` calls the
    chirp-Z phase ``alpha*N_max^2``; the kernel the code builds is
    ``exp(i*pi*alpha*m^2)`` over ``|m| <= L - N_out``, which is
    ``alpha*(L-N_out)^2 / 2`` TURNS.  The ``/4`` absorbs the difference; this
    id pins that it still does.
"""
from __future__ import annotations

import inspect
import math
import warnings

import numpy as np
import pytest

from lumenairy.propagators import _bluestein as B
from lumenairy.propagators._bluestein import (_auto_selects_direct,
                                              _bluestein_2d,
                                              _bluestein_centred_2d,
                                              _clear_h_fft_cache,
                                              _MFT_DIRECT_ALWAYS,
                                              _MFT_DIRECT_MAX_RATIO,
                                              _MFT_DIRECT_NEVER)
from lumenairy.propagators.fft_infra import _fft2, _ifft2

TAU = 2.0 * math.pi


def _rand(ny, nx, seed=20260920):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((ny, nx))
            + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)


def _bits(a):
    return np.ascontiguousarray(np.asarray(a)).tobytes()


def _alpha(ny, nx, my, mx, budget=1.0e3):
    return budget / float(max(ny, nx, my, mx)) ** 2


# ---------------------------------------------------------------------------
# 1.  The anisotropy exposure
# ---------------------------------------------------------------------------

def _dense_work_per_kernel_entry(ny, nx, my, mx):
    """The dense route's multiply-adds per TRANSCENDENTAL kernel entry.

    Read off ``_direct_matrix_2d``: it builds two kernels of ``My*Ny`` and
    ``Mx*Nx`` entries, each one a ``numpy`` complex ``exp`` (tens of times the
    cost of a multiply-add), and then spends
    ``min(My*Ny*Nx + My*Nx*Mx, Ny*Nx*Mx + My*Ny*Mx)`` multiply-adds using
    them -- the same two costs the function itself compares to choose its
    association order.  The ratio of the two is what decides whether the
    kernel BUILD or the matrix products dominate, and it is a pure function of
    the four grid sizes, like the rule itself.

    It is NOT a function of the two ratios ``My/Ny`` and ``Mx/Nx``: at a fixed
    ratio pair it falls without bound as the short axis shrinks
    (``2048x2048 -> 64x64`` reads 1056, ``2048x64 -> 64x2`` reads 4.0, both at
    ratio 1/32), which is why no retune of ``_MFT_DIRECT_MAX_RATIO`` can reach
    this regime -- MEASURED at 1/64 as well, where ``4096x64 -> 64x1`` is
    still 1.07x to 1.49x slower.
    """
    entries = my * ny + mx * nx
    flops = min(my * ny * nx + my * nx * mx, ny * nx * mx + my * ny * mx)
    return flops / float(entries)


#: Shapes the SHIPPED rule captures at which the dense route was MEASURED
#: SLOWER than ``min(chirp-Z 2-D, separable)``, worst of three rounds of
#: best-of-nine, on 2026-09-20.  ``(shape): (WIN worst, WSL worst)``, each the
#: worst reading over BOTH instruments -- mine, which interleaves the routes,
#: and the branch's, which runs them in blocks.  The two agree about WHICH
#: shapes are slower at every entry.  Evidence:
#: ``validation/probe_verify_c4/v4_boundary_thin_{win,wsl}.json``,
#: ``..._thin_blocked_{win,wsl}.json`` and ``..._thin_w1_{win,wsl}.json`` (the
#: last with ``SCIPY_FFT_WORKERS = 1``, i.e. both sides single-threaded, which
#: removes the thread asymmetry as an explanation).
_MEASURED_SLOWER = {
    (2048, 128, 64, 4): (0.849, 1.250),
    (2048, 64, 64, 2): (1.432, 1.810),
    (2048, 32, 64, 1): (2.654, 2.293),
    (1024, 32, 32, 1): (1.581, 0.965),
    (64, 2048, 2, 64): (1.182, 1.615),
    (32, 2048, 1, 64): (2.180, 2.040),
    (4096, 64, 128, 2): (2.802, 9.685),
    (4096, 64, 64, 1): (1.487, None),
    (64, 4096, 1, 64): (1.102, None),
}

#: The largest ``work-per-kernel-entry`` at which the dense route was measured
#: SLOWER (``2048x128 -> 64x4``, 11.96) and the smallest at which it was
#: measured safe above it (``1024x128 -> 32x4``, 19.69).  A guard placed
#: anywhere in this open interval refuses every shape measured slower and
#: keeps every shape measured safe above it.  Stated as a pair so the gap is
#: visible; nothing here asserts a particular choice.
_WORK_RATIO_LARGEST_MEASURED_SLOWER = 12.0
_WORK_RATIO_SMALLEST_MEASURED_SAFE_ABOVE = 19.6

#: Captured shapes BELOW that threshold that were nonetheless measured safe
#: on both builds and under both instruments -- small absolute problems, where
#: the chirp-Z route's fixed costs dominate whatever the work ratio says.
#: ``(shape): (WIN worst, WSL worst)``, same runs and the same
#: worst-over-both-instruments convention as ``_MEASURED_SLOWER``.
#: They are listed, not predicted: the work ratio is a one-sided screen.
_MEASURED_SAFE_BELOW_THRESHOLD = {
    (1024, 64, 32, 2): (0.854, 0.768),
    (512, 64, 16, 2): (0.581, 0.525),
    (512, 32, 16, 1): (0.642, 0.486),
    (2048, 64, 32, 1): (0.742, None),
    (2048, 128, 32, 2): (0.478, None),
    (1024, 64, 16, 1): (0.383, None),
}


def test_the_dense_work_ratio_is_a_pure_function_of_the_four_grid_sizes():
    """The quantity a fix would have to read is as build-free as the rule.

    It reads four integers, so it cannot depend on the build, the backend or
    the clock -- the same property ``_auto_selects_direct`` rests on.  And it
    is NOT determined by the two ratios: two shapes with the SAME ratio pair
    differ in it by more than two decades, which is the whole finding.
    """
    square = _dense_work_per_kernel_entry(2048, 2048, 64, 64)
    thin = _dense_work_per_kernel_entry(2048, 64, 64, 2)
    # The two shapes have the SAME ratio pair, so no comparison against
    # _MFT_DIRECT_MAX_RATIO can tell them apart.  Asserted about the RATIOS
    # and not about what the rule currently answers, so a guard that DOES
    # separate them leaves this id true.
    assert 64 / 2048 == 2 / 64 == 1.0 / 32.0
    assert square / thin > 100.0, (
        f"two shapes with the SAME ratio pair (1/32, 1/32) "
        f"read {square:.1f} and {thin:.1f} multiply-adds per kernel entry; "
        f"if that spread has closed, re-measure before trusting this file")
    # integer-only, repeatable, no float state
    for _ in range(3):
        assert _dense_work_per_kernel_entry(2048, 64, 64, 2) == thin


def test_every_shape_the_rule_captures_is_work_dense_or_was_measured_slower():
    """THE EXPOSURE, bounded so that closing it does not falsify this id.

    For every shape on the ladder: if ``'auto'`` sends it to the dense route,
    then either its dense route does enough arithmetic per transcendental for
    that to have been measured safe, or the shape is one of the ones this
    verification MEASURED slower.  Today the second branch carries the thin
    shapes.  With the missing guard in place the first branch carries them --
    they are simply not captured any more -- and this id still passes.

    What it refuses is a THIRD state: the rule capturing a shape that is
    neither work-dense nor in the measured list, i.e. a widening of the
    captured region into unmeasured anisotropic territory.
    """
    ladder = list(_MEASURED_SLOWER) + [
        (2048, 2048, 64, 64), (1024, 1024, 32, 32), (512, 512, 16, 16),
        (2048, 512, 64, 16), (2048, 256, 64, 8), (1024, 128, 32, 4),
        (1024, 64, 32, 2), (512, 64, 16, 2), (512, 32, 16, 1),
        (768, 768, 24, 24), (1000, 1000, 25, 25), (64, 64, 2, 2),
        (2048, 2048, 16, 64), (2048, 1024, 64, 16),
    ]
    offenders = []
    for shape in ladder:
        if not _auto_selects_direct(*shape):
            continue
        w = _dense_work_per_kernel_entry(*shape)
        if w >= _WORK_RATIO_SMALLEST_MEASURED_SAFE_ABOVE:
            continue
        if (shape in _MEASURED_SLOWER
                or shape in _MEASURED_SAFE_BELOW_THRESHOLD):
            continue
        offenders.append((shape, round(w, 2)))
    assert not offenders, (
        f"method='auto' captures {offenders} -- shapes whose dense route "
        f"spends fewer than {_WORK_RATIO_SMALLEST_MEASURED_SAFE_ABOVE} "
        f"multiply-adds per transcendental kernel entry and which this "
        f"verification did not time.  Every such shape it DID time came out "
        f"1.1x to 9.7x SLOWER than the chirp-Z fallback on both builds.  "
        f"Either time these too, or gate the rule on the work ratio as well "
        f"as on the two grid ratios")


def test_retuning_the_ratio_cannot_reach_the_anisotropic_regime():
    """The report offers ``1/64`` as the value "with a two-fold margin at
    every shape".  It is not a remedy for this: MEASURED at 1/64,
    ``4096x64 -> 64x1`` is still 1.07 to 1.49 times slower.

    Asserted without a clock, and without reading what the rule answers, so
    the id survives the guard it argues for: at EVERY candidate constant a
    THIN shape and a SQUARE shape share the same two ratios exactly, while
    their multiply-adds per transcendental kernel entry straddle the largest
    value measured slower by more than two decades.  A rule that compares only
    against the constant cannot separate them; one that also reads the work
    ratio can.
    """
    for constant in (1.0 / 32.0, 1.0 / 64.0, 1.0 / 128.0):
        n = int(round(1.0 / constant))
        thin = (64 * n, n, 64, 1)
        square = (64 * n, 64 * n, 64, 64)
        assert (max(thin[2] / thin[0], thin[3] / thin[1])
                == max(square[2] / square[0], square[3] / square[1])
                == constant), (
            f"at 1/{n} the two probe shapes no longer share a ratio pair; "
            f"the construction behind this id has drifted")
        w_thin = _dense_work_per_kernel_entry(*thin)
        w_square = _dense_work_per_kernel_entry(*square)
        assert w_thin < _WORK_RATIO_LARGEST_MEASURED_SLOWER < w_square, (
            f"at 1/{n}: thin reads {w_thin:.1f} and square {w_square:.1f} "
            f"multiply-adds per kernel entry; they no longer straddle the "
            f"largest value measured slower "
            f"({_WORK_RATIO_LARGEST_MEASURED_SLOWER})")
        assert w_square / w_thin > 100.0


def test_the_derived_byte_counts_predict_where_the_memory_half_fails():
    """The report says the memory half "never argues against the rule
    anywhere".  It does, at the same thin shapes -- and the CODE says so
    before any measurement does.

    MEASURED 2026-09-20, ``tracemalloc`` peak, identical conclusion on both
    builds: at ``2048x64 -> 64x2`` the dense route peaks at 5.264 MB against
    the separable route's 4.399 MB (dense/cheapest = 1.20), while at every
    square captured shape it is 18x to 74x smaller.  This id asserts the
    prediction, from the two routes' own array sizes, so the exception is
    derived and not anecdotal.
    """
    from scipy.fft import next_fast_len

    def dense_live(ny, nx, my, mx):
        # two kernels, the larger kernel's float64 + complex128 build pair,
        # the intermediate and the output
        build = 24 * max(my * ny, mx * nx)
        kernels = 16 * (my * ny + mx * nx)
        y_first = (my * ny * nx + my * nx * mx) <= (ny * nx * mx
                                                    + my * ny * mx)
        inter = 16 * (my * nx if y_first else ny * mx)
        return max(build, kernels + inter + 16 * my * mx)

    def sep_live(ny, nx, my, mx):
        Ly = int(next_fast_len(ny + my - 1))
        Lx = int(next_fast_len(nx + mx - 1))
        return 16 * max(ny * Lx, my * Ly)

    for shape in ((2048, 2048, 64, 64), (1024, 1024, 32, 32),
                  (512, 512, 16, 16)):
        assert dense_live(*shape) < sep_live(*shape), (
            f"{shape}: the derived byte counts no longer put the dense route "
            f"below the separable one at a SQUARE captured shape")
    thin = (2048, 64, 64, 2)
    assert dense_live(*thin) > sep_live(*thin), (
        f"{thin}: the derived counts no longer predict the measured memory "
        f"exception (dense 5.264 MB against separable 4.399 MB on both "
        f"builds, 2026-09-20).  Re-measure before relaxing anything that "
        f"rests on 'the dense route is always the smaller one'")


# ---------------------------------------------------------------------------
# 2.  A y/x transposition at the call sites
# ---------------------------------------------------------------------------

#: Anisotropic shapes at which exchanging the two OUTPUT sizes flips what the
#: rule answers.  Exchanging BOTH pairs cannot flip it -- ``max`` over the two
#: ratios is symmetric under that -- so this is the only form of the swap a
#: test can see.
#: All four are WORK-DENSE (108 .. 435 multiply-adds per transcendental
#: kernel entry), so a work-ratio guard leaves them exactly where they are;
#: and they go both ways under the swap -- the first and third are captured
#: and become refused, the second and fourth are refused and become captured.
_TRANSPOSE_SENSITIVE = [(512, 1024, 16, 32), (512, 1024, 32, 16),
                        (1024, 2048, 32, 64), (256, 512, 8, 16)]


def test_the_rule_is_sensitive_to_a_y_x_transposition_at_all():
    """The premise the next id needs: at these shapes the swapped call gives
    a DIFFERENT answer, so a dispatch comparison there can see a
    transposition.  Without this, the next id could pass on a rule that
    ignores its arguments."""
    for (ny, nx, my, mx) in _TRANSPOSE_SENSITIVE:
        assert _auto_selects_direct(ny, nx, my, mx) \
            != _auto_selects_direct(ny, nx, mx, my), (
            f"{(ny, nx, my, mx)}: exchanging the two output sizes does not "
            f"change what the rule answers, so this shape cannot witness a "
            f"y/x transposition")


@pytest.mark.parametrize("primitive", ['plain', 'centred'])
def test_auto_dispatches_as_the_rule_says_at_transpose_sensitive_shapes(
        primitive):
    """The dispatch agrees with the rule at shapes where a y/x transposition
    in the WIRING would make it disagree.

    MEASURED 2026-09-20: mutating both call sites to pass
    ``(Ny_in, Nx_in, N_out_x, N_out_y)`` is refused by this id at all four of
    its shapes, and by the shipped file at its ONE transposition-sensitive
    shape.  The difference is the margin: the shipped coverage rests on a
    single ``_DENSE_SIDE`` entry that nobody has to keep.
    """
    fn = _bluestein_2d if primitive == 'plain' else _bluestein_centred_2d
    for (ny, nx, my, mx) in _TRANSPOSE_SENSITIVE:
        E = _rand(ny, nx)
        a = _alpha(ny, nx, my, mx)
        kw = dict(sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2)
        _clear_h_fft_cache()
        auto = fn(E, a, a * 1.5, my, mx, **kw)
        _clear_h_fft_cache()
        dense = fn(E, a, a * 1.5, my, mx, method='direct', **kw)
        _clear_h_fft_cache()
        chirp = fn(E, a, a * 1.5, my, mx, method='bluestein', **kw)
        matched = ('direct' if _bits(auto) == _bits(dense)
                   else 'bluestein' if _bits(auto) == _bits(chirp) else None)
        assert matched is not None, (
            f"{primitive} {ny}x{nx}->{my}x{mx}: 'auto' matched neither route")
        says = _auto_selects_direct(ny, nx, my, mx)
        assert (matched == 'direct') == says, (
            f"{primitive} {ny}x{nx}->{my}x{mx}: the rule says "
            f"{'direct' if says else 'bluestein'} and 'auto' returned "
            f"{matched} -- the four grid sizes are reaching the rule in the "
            f"wrong order")


# ---------------------------------------------------------------------------
# 3.  The way back, at the ENTRY POINTS
# ---------------------------------------------------------------------------

#: Public entry points that reach the MFT primitives and DO expose a route
#: keyword a caller can pass for one call.
_WITH_KEYWORD = ('fresnel_propagate_mft', 'fraunhofer_propagate_mft',
                 'angular_spectrum_propagate_mft', 'asm_propagate')

#: Public entry points that reach the MFT primitives and DO NOT.  MEASURED
#: 2026-09-20 archive-to-archive against ``49ddf4bd``: every one of these
#: MOVES at a shape the rule captures and is byte-identical at one it refuses
#: (``validation/probe_verify_c4/v4_entry_compare_win.json``).  The way back
#: offered for them is a PRIVATE MODULE CONSTANT, which is process-wide and is
#: not a keyword.  Listing them is the point: a new one appearing here, or one
#: of these growing a keyword, should be a decision and not a surprise.
_WITHOUT_KEYWORD = ('compute_psf', 'resample_field', 'propagate',
                    'carrier_referenced_focus_readout',
                    'carrier_referenced_exact_focus_readout',
                    're_reference', 'propagate_traced_carrier_chain')


def _route_keyword_of(fn):
    """Does this callable's OWN signature let a caller name the MFT route?

    ``method`` counts only when its documented vocabulary is the ROUTE
    vocabulary; ``compute_psf(method='fft'|'mft')`` and
    ``propagate(method='asm'|'fresnel'|...)`` name the propagator family and
    do NOT.  A ``**kwargs`` that is forwarded to one of the three MFT
    propagators does count, which is why ``asm_propagate`` is on the other
    list.
    """
    params = inspect.signature(fn).parameters
    p = params.get('method')
    if p is not None and p.default == 'auto':
        return 'method'
    if any(q.kind is inspect.Parameter.VAR_KEYWORD for q in params.values()) \
            and 'method' not in params:
        return '**kwargs -> method'
    return None


def test_the_public_entry_points_are_classified_by_their_way_back():
    """Every public entry point that reaches the MFT is on exactly one of the
    two lists, and its signature agrees with which list it is on.

    Fix-stable in the direction that matters: giving one of the unprotected
    entry points a route keyword makes ``_route_keyword_of`` return one, and
    the assertion below only requires that the WITH list all have one.  What
    it refuses is a new entry point on neither list.
    """
    import lumenairy as la
    for name in _WITH_KEYWORD:
        fn = getattr(la, name)
        assert _route_keyword_of(fn) is not None, (
            f"{name} no longer exposes a way to name the MFT route; the "
            f"campaign rule is that every moved public entry point has one")
    for name in _WITHOUT_KEYWORD:
        assert hasattr(la, name), (
            f"{name} is no longer public; re-derive this list from an AST "
            f"sweep of lumenairy/ rather than editing it")
    assert not set(_WITH_KEYWORD) & set(_WITHOUT_KEYWORD)


def test_the_keyword_way_back_reproduces_the_previous_dispatch_exactly():
    """For the entry points that HAVE a keyword: naming the previous route
    gives the same bytes the whole-process way back gives.

    This is the in-tree form of the archive-to-archive claim -- it compares
    ``method='bluestein'`` / ``'separable'`` against a no-keyword call with
    ``_MFT_DIRECT_MAX_RATIO = _MFT_DIRECT_NEVER``, which the shipped file
    proves is byte-identical to ``49ddf4bd``.  Measured on the DENSE side,
    where it is a claim.
    """
    from lumenairy.propagators.mft import (angular_spectrum_propagate_mft,
                                           fraunhofer_propagate_mft,
                                           fresnel_propagate_mft)
    wl, dx, z, n_in, n_out = 633e-9, 4e-6, 0.05, 256, 8
    dx_out = wl * z / (n_in * dx)
    E = _rand(n_in, n_in, seed=7) * 1e-3
    saved = B._MFT_DIRECT_MAX_RATIO
    try:
        assert _auto_selects_direct(n_in, n_in, n_out, n_out)
        for fn in (fresnel_propagate_mft, fraunhofer_propagate_mft,
                   angular_spectrum_propagate_mft):
            B._MFT_DIRECT_MAX_RATIO = saved
            named = fn(E, z, wl, dx, dx_out, n_out, method='bluestein')
            captured = fn(E, z, wl, dx, dx_out, n_out)
            B._MFT_DIRECT_MAX_RATIO = _MFT_DIRECT_NEVER
            by_constant = fn(E, z, wl, dx, dx_out, n_out)
            assert _bits(named) == _bits(by_constant), (
                f"{fn.__name__}: method='bluestein' and _MFT_DIRECT_NEVER do "
                f"not give the same bytes; one of the two ways back is not "
                f"the previous dispatch")
            assert _bits(captured) != _bits(by_constant), (
                f"{fn.__name__}: the captured shape did not move at all, so "
                f"this id is not testing a way back")
    finally:
        B._MFT_DIRECT_MAX_RATIO = saved


def test_the_entry_points_without_a_keyword_really_are_exposed():
    """The other half of the classification, MEASURED rather than asserted:
    the no-keyword entry points DO move when the rule fires, so the list
    above is a list of real exposures and not of theoretical ones.

    Driven through the constant, which is the only handle these callers have:
    the answer under ``_MFT_DIRECT_NEVER`` differs from the answer under the
    shipped constant at a shape the rule captures.
    """
    from lumenairy.analysis.psf_mtf_otf import compute_psf
    from lumenairy.propagators.mft import resample_field
    wl, dx, f_len = 633e-9, 4e-6, 0.05
    pup = np.exp(-((np.arange(256) - 128.0) ** 2)[:, None] / 60.0 ** 2
                 - ((np.arange(256) - 128.0) ** 2)[None, :] / 60.0 ** 2
                 ).astype(np.complex128)
    saved = B._MFT_DIRECT_MAX_RATIO
    try:
        assert _auto_selects_direct(256, 256, 8, 8)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            B._MFT_DIRECT_MAX_RATIO = saved
            psf_now, _ = compute_psf(pup, wl, f_len, dx, 8, method='mft')
            rs_now, _ = resample_field(pup, dx, dx * 32.0, 8,
                                       method='chirpz')
            B._MFT_DIRECT_MAX_RATIO = _MFT_DIRECT_NEVER
            psf_before, _ = compute_psf(pup, wl, f_len, dx, 8, method='mft')
            rs_before, _ = resample_field(pup, dx, dx * 32.0, 8,
                                          method='chirpz')
        assert _bits(psf_now) != _bits(psf_before), (
            "compute_psf(method='mft') does not move with the constant, so "
            "either the rule stopped reaching it or this fixture stopped "
            "being a captured shape")
        assert _bits(rs_now) != _bits(rs_before), (
            "resample_field(method='chirpz') does not move with the constant")
    finally:
        B._MFT_DIRECT_MAX_RATIO = saved


# ---------------------------------------------------------------------------
# 4.  The warning, counted
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("primitive", ['plain', 'centred'])
@pytest.mark.parametrize("separable", [False, True])
def test_the_phase_guard_warns_exactly_once_per_call(primitive, separable):
    """ONE warning per call, on both sides of the boundary, on both
    primitives, under both ``separable`` settings -- and none at all under the
    threshold.

    The guard moved: it used to sit inside one arm of ``_bluestein_2d`` and it
    now runs before ``'auto'`` chooses, and is ALSO called from
    ``_bluestein_centred_2d``'s dense arm.  A guard called from two places is
    exactly the shape that emits twice on the path that reaches both, and the
    shipped file counts ``>= 1`` rather than ``== 1``.
    """
    fn = _bluestein_2d if primitive == 'plain' else _bluestein_centred_2d
    for (ny, nx, my, mx) in ((96, 96, 3, 3), (64, 64, 8, 8),
                             (128, 64, 4, 2)):
        E = _rand(ny, nx, seed=99)
        over = _alpha(ny, nx, my, mx, budget=1.0e11)
        under = _alpha(ny, nx, my, mx, budget=1.0e3)
        kw = dict(sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2,
                  separable=separable)
        for label, a, expect in (('over', over, 1), ('under', under, 0)):
            for method, silent in ((None, False), ('direct', True),
                                   ('bluestein', False),
                                   ('separable', False)):
                _clear_h_fft_cache()
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    extra = {} if method is None else {'method': method}
                    fn(E, a, a, my, mx, **kw, **extra)
                got = [x for x in w if issubclass(x.category, RuntimeWarning)]
                want = 0 if (silent or expect == 0) else 1
                assert len(got) == want, (
                    f"{primitive} sep={separable} {ny}x{nx}->{my}x{mx} "
                    f"budget={label} method={method!r}: {len(got)} "
                    f"RuntimeWarning(s), expected {want}")


def test_the_warning_names_the_route_the_call_actually_took():
    """Two-sided, and the mutation it catches is named.

    MEASURED 2026-09-20: forcing ``on_dense=False`` at the ``_bluestein_2d``
    call site -- a message that advises ``method='direct'`` to a caller
    already on it -- is caught only by
    ``test_c4_mft_direct_default.py::test_the_default_flip_does_not_take_a_
    warning_away_from_a_caller``.  This id is the same claim on the CENTRED
    primitive and on an anisotropic shape, neither of which that id drives.
    """
    for (ny, nx, my, mx) in ((128, 64, 4, 2), (64, 64, 8, 8)):
        says = _auto_selects_direct(ny, nx, my, mx)
        E = _rand(ny, nx, seed=1234)
        a = _alpha(ny, nx, my, mx, budget=1.0e11)
        for fn in (_bluestein_2d, _bluestein_centred_2d):
            _clear_h_fft_cache()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                fn(E, a, a, my, mx, sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2)
            msgs = [str(x.message) for x in w
                    if issubclass(x.category, RuntimeWarning)]
            assert len(msgs) == 1, (
                f"{fn.__name__} {ny}x{nx}->{my}x{mx}: {len(msgs)} warnings")
            on_dense_text = "ALREADY on the dense route" in msgs[0]
            assert on_dense_text == says, (
                f"{fn.__name__} {ny}x{nx}->{my}x{mx}: the rule says "
                f"{'direct' if says else 'chirp-Z'} but the message "
                f"{'claims' if on_dense_text else 'does not claim'} the call "
                f"is on the dense route -- the message is describing a route "
                f"the call did not take")
            advises_direct = "method='direct' is the more accurate" in msgs[0]
            assert advises_direct == (not says), (
                f"{fn.__name__} {ny}x{nx}->{my}x{mx}: the message advises "
                f"method='direct' to a caller that is already on it")


# ---------------------------------------------------------------------------
# 5.  The bar's derivation
# ---------------------------------------------------------------------------

def test_the_shipped_phase_ratio_bar_is_conservative_against_the_kernels():
    """``R/4`` must stay at or below what the two kernels actually imply.

    Re-derived from the source, in TURNS (the unit the float64 product is
    formed in):

    * ``_bluestein_2d`` builds ``exp(i*sign*pi*alpha*m^2)`` over the PADDED
      kernel index, ``|m| <= L - N_out`` with
      ``L = next_fast_len(N_in + N_out - 1)``, so it spends
      ``alpha*(L - N_out)^2 / 2`` turns -- the ``pi`` is half a turn;
    * ``_direct_matrix_2d`` builds ``exp(i*sign*2*pi*alpha*n*k)``, so it
      spends ``alpha*(N_in - 1)*(N_out - 1)`` turns.

    ``_phase_term_ratio`` returns ``N_max^2 / ((N-1)(M-1))``, which is about
    TWICE the turn ratio.  The expected gap is the turn ratio times
    ``C_chirp/C_dense``, MEASURED in [1.48, 4.04] over ten decades of budget,
    so the bar ``R/4`` is conservative iff ``R/4 <= R_turns * 1.48``.  This id
    pins that inequality, so a future tightening of the ``/4`` cannot make the
    bar optimistic without someone re-deriving it.

    MEASURED 2026-09-20, both builds, at a budget of ~333: dense-side gaps
    21.5 .. 91.8 against bars 8.9 .. 32.3 -- clear by 2.4x to 5.7x; an
    impostor dense arm reads 0.995 .. 1.003, i.e. 10x to 16x UNDER the bar.
    """
    from scipy.fft import next_fast_len
    from tests.unit.test_c4_mft_direct_default import _phase_term_ratio
    c_min = 1.48
    for (n, m) in ((64, 2), (96, 3), (128, 4), (160, 5), (192, 6), (224, 7),
                   (256, 8), (320, 10), (128, 2)):
        L = int(next_fast_len(n + m - 1))
        m_max = max(L - m, n - 1, m - 1)
        r_turns = (float(m_max) ** 2 / 2.0) / (float(n - 1) * float(m - 1))
        r_shipped = _phase_term_ratio(n, m)
        assert r_shipped / 4.0 <= r_turns * c_min, (
            f"{n}->{m}: the shipped bar R/4 = {r_shipped / 4.0:.2f} is above "
            f"what the kernels imply, {r_turns:.2f} turns x the smallest "
            f"MEASURED constant ratio {c_min} = {r_turns * c_min:.2f}.  "
            f"Re-derive the bar before tightening it")
        assert r_shipped > r_turns, (
            f"{n}->{m}: the shipped R is no longer the larger of the two; "
            f"the factor this id exists to bound has changed sign")


def test_the_two_documented_ends_of_the_constant_still_mean_what_they_say():
    """A short control so the ids above cannot pass on a rule that has
    stopped consulting the constant at all."""
    saved = B._MFT_DIRECT_MAX_RATIO
    probes = [(2048, 64, 64, 2), (96, 96, 3, 3), (24, 24, 12, 12)]
    try:
        B._MFT_DIRECT_MAX_RATIO = _MFT_DIRECT_NEVER
        assert not any(_auto_selects_direct(*s) for s in probes)
        B._MFT_DIRECT_MAX_RATIO = _MFT_DIRECT_ALWAYS
        assert all(_auto_selects_direct(*s) for s in probes)
    finally:
        B._MFT_DIRECT_MAX_RATIO = saved
    assert 0.0 < float(_MFT_DIRECT_MAX_RATIO) < 1.0
