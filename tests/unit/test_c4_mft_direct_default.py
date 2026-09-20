"""WP-C4 -- ``method='auto'`` selects the dense matrix-Fourier route below
``M/N = 1/32``, and everything that makes that a decision rather than a drift.

WHAT SHIPPED.  Hygiene-2 built the dense route (``_direct_matrix_2d``) as ONE
``xp``-parametrised implementation behind ``method=``, opt-in, and measured it
the most accurate and the cheapest in memory of the three routes through the
MFT sum.  What it would not do was select it automatically, because the TIME
crossover is per-build.  v5.49.0 selects it from the SHAPE instead:
``_auto_selects_direct`` reads four grid sizes and TWO module constants,
``_MFT_DIRECT_MAX_RATIO`` and ``_MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY``, and
nothing else.

TWO CONDITIONS, AND WHY THE SECOND ONE EXISTS (round 2, VERIFY-WP-C4 D1).
The first is the two grid RATIOS.  The second counts the dense route's
multiply-adds per transcendental kernel entry, because a ratio cannot see a
THIN input: ``2048x2048 -> 64x64`` reads 1056 of them and ``2048x64 -> 64x2``
reads 4.0, and both sit at ratio exactly ``(1/32, 1/32)`` -- at which the
dense route was measured 1.1x to 13.0x SLOWER on both builds, and larger in
memory besides.  Shapes have to clear BOTH.

THE SIX CLAIMS IN THIS FILE, each stated as a helper so the mutation matrix
at the bottom can exercise the SAME assertion the shipped id does rather than
a paraphrase of it:

1.  the default is ``'auto'`` on both primitives and all three public entry
    points -- ``_claim_the_default_keyword_is_auto``;
2.  the boundary is asked FROM the constant and never typed as a number, and
    the two documented settings mean what they say --
    ``_claim_the_boundary_comes_from_the_constant``;
2b. the work screen refuses a thin input at the SAME ratio pair a square
    shape is captured at, and it too is asked from its constant --
    ``_claim_the_work_screen_refuses_a_thin_input``;
3.  at a shape on either side, ``'auto'`` returns the bytes of the route the
    rule names and never a third arithmetic --
    ``_claim_auto_dispatches_as_the_rule_says``;
4.  the way back is byte-identical -- ``_claim_the_way_back_is_byte_identical``;
5.  on the dense side the dense route is the more accurate one against a
    reference whose phase is reduced EXACTLY --
    ``_claim_the_dense_side_is_the_more_accurate_side``.

NOTHING HERE PINS A TIMING OR A BUILD.  The boundary constant is read, not
asserted to a number; the accuracy claim is a two-sided DECISION (which route
is closer to an exact reference, by how much, inside a derived bar) and not a
residual read off one build; and no dense-route byte is pinned anywhere,
because the dense route goes through BLAS and its last bits move with the
kernel.  The byte claims are all WITHIN one process: ``'auto'`` against a route
named explicitly in the same interpreter, which is a statement about dispatch
and not about arithmetic.  The archive-to-archive proof against 49ddf4bd is
``validation/probe_c4_mft_direct/c4_compare_{win,wsl}.json``.

Measurements behind the constant: WP-C4_MFT_DIRECT_DEFAULT_REPORT.md.

Author:  Andrew Traverso
"""
from __future__ import annotations

import inspect
import math
import warnings
from fractions import Fraction

import numpy as np
import pytest

from lumenairy.propagators import _bluestein as B
from lumenairy.propagators._bluestein import (
    _MFT_DIRECT_ALWAYS,
    _MFT_DIRECT_MAX_RATIO,
    _MFT_DIRECT_NEVER,
    _auto_selects_direct,
    _bluestein_2d,
    _bluestein_centred_2d,
    _clear_h_fft_cache,
)
from lumenairy.propagators.fft_infra import _fft2, _ifft2
from lumenairy.propagators.mft import (
    angular_spectrum_propagate_mft,
    fraunhofer_propagate_mft,
    fresnel_propagate_mft,
)

TAU = 2.0 * math.pi
WL = 633e-9

#: Shapes the rule sends to the dense route, and shapes it leaves on the
#: chirp-Z route.  Derived from the constant at import time, NOT typed: if the
#: maintainer retunes the boundary these lists follow it, which is the
#: difference between a test of the rule and a test of one value of it.
_DENSE_SIDE = [(ny, nx, my, mx) for (ny, nx, my, mx) in
               [(96, 96, 3, 3), (128, 128, 4, 4), (256, 256, 8, 8),
                (64, 64, 2, 2), (512, 1024, 16, 32)]
               if _auto_selects_direct(ny, nx, my, mx)]
#: ``(512, 32, 16, 1)`` is on this side for the SECOND condition and not the
#: first: both its ratios are exactly ``1/32``, so the ratio test admits it,
#: and its dense route spends 2.99 multiply-adds per transcendental kernel
#: entry, so the work screen refuses it.  It is here so the dispatch claim
#: covers the new condition on real bytes and not only on the predicate.
_CHIRP_SIDE = [(ny, nx, my, mx) for (ny, nx, my, mx) in
               [(24, 24, 12, 12), (64, 64, 8, 8), (32, 32, 16, 16),
                (28, 22, 15, 13), (512, 1024, 64, 32), (512, 32, 16, 1)]
               if not _auto_selects_direct(ny, nx, my, mx)]


def _bits(a):
    """Raw bytes plus dtype and shape -- no float ``==`` anywhere."""
    a = np.ascontiguousarray(a)
    return (str(a.dtype), a.shape, a.tobytes())


def _rand(ny, nx, seed=20260920):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((ny, nx))
            + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)


def _gauss(n, dx, w, seed=3):
    rng = np.random.default_rng(seed)
    x = (np.arange(n) - n / 2.0) * dx
    r2 = x[:, None] ** 2 + x[None, :] ** 2
    E = np.exp(-r2 / w ** 2).astype(np.complex128)
    return E * np.exp(1j * 0.3 * rng.standard_normal((n, n)))


def _alpha(ny, nx, my, mx, budget=1.0e3):
    """An ``alpha`` whose phase budget is ``budget`` -- six decades under the
    guard at ``1e3``, so no route pays a warning and nothing here measures the
    warnings machinery."""
    return budget / float(max(ny, nx, my, mx)) ** 2


def _exact_phase_rows(alpha, n_in, n_out):
    """``frac(alpha * n * k)`` in ``[-1/2, 1/2)``, reduced EXACTLY.

    ``alpha`` is a float64 and therefore an exact rational, ``n`` and ``k`` are
    integers, so in :class:`fractions.Fraction` the product and its fractional
    part are exact and the ONE float64 rounding lands on a number already
    inside ``[-1/2, 1/2)``.

    This is the one thing NO route does, and it is why the reference can see
    the routes' phase error at all: a reference that forms ``t = alpha*k*n`` in
    float64 and then reduces it commits the two roundings the dense route
    commits and agrees with it by CONSTRUCTION (VERIFY-WAVE5-HYGIENE2 round 2,
    D-1).
    """
    fa = Fraction(alpha)
    T = np.empty((n_out, n_in), dtype=np.float64)
    for k in range(n_out):
        fk = Fraction(k)
        for n in range(n_in):
            t = fa * fk * n
            t -= math.floor(t)                  # exact, into [0, 1)
            if t >= Fraction(1, 2):
                t -= 1                          # exact, into [-1/2, 1/2)
            T[k, n] = float(t)
    return T


def _exact_reference(E, alpha, M, sign=-1):
    """The same sum, correctly rounded in BOTH senses: exact phase, fsum."""
    ny, nx = E.shape
    Wy = np.exp(1j * sign * TAU * _exact_phase_rows(alpha, ny, M))
    Wx = np.exp(1j * sign * TAU * _exact_phase_rows(alpha, nx, M))
    out = np.empty((M, M), dtype=np.complex128)
    for ky in range(M):
        wy = Wy[ky]
        for kx in range(M):
            T = E * (wy[:, None] * Wx[kx][None, :])
            out[ky, kx] = complex(math.fsum(T.real.ravel()),
                                  math.fsum(T.imag.ravel()))
    return out


def _derived_bar(E, N, M, alpha, route):
    """The ABSOLUTE bar a route may not cross against the exact reference.

    Two sources, derived and summed -- nothing fitted:

    * SUMMATION.  Each output point sums ``n = Ny*Nx`` unit-modulus terms, so a
      summation of growth factor ``g`` commits at most ``g * eps * sum|E|``;
      ``g = 3*log2(L^2)`` for a chirp-Z route (three FFTs of length ``L^2``),
      ``g = sqrt(n)`` for the dense route's two BLAS products, and ``g = 1``
      for the ``math.fsum`` reference.
    * PHASE.  The route's phase argument ``t`` is a float64 product that has
      already lost its low bits, costing ``~eps*|t|`` of phase and therefore
      ``2*pi*eps*max|t|*sum|E|`` absolute.  ``max|t|`` is NOT the same for the
      two routes and that is the point: the chirp-Z route builds
      ``exp(i*pi*alpha*n^2)`` with ``n`` up to ``max(N, M)``, so it spends
      ``alpha*N_max^2``; the dense route builds ``exp(2*pi*i*alpha*n*k)`` with
      ``n < N`` and ``k < M``, so it spends only ``alpha*(N-1)*(M-1)``.
    """
    from scipy.fft import next_fast_len
    eps = float(np.finfo(np.float64).eps)
    n = int(E.size)
    s = float(np.sum(np.abs(E)))
    if route == 'dense':
        g = math.sqrt(float(n))
        max_t = abs(alpha) * float(N - 1) * float(max(M - 1, 1))
    else:
        L = int(next_fast_len(int(N + M - 1)))
        g = 3.0 * math.log2(float(L) ** 2)
        max_t = abs(alpha) * float(max(N, M)) ** 2
    return (g + 1.0 + TAU * max_t) * eps * s


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ===========================================================================
# The five claims, each callable so the mutation matrix can exercise it
# ===========================================================================

def _claim_the_default_keyword_is_auto():
    """``method`` defaults to ``'auto'`` on both primitives and all three
    public entry points -- read off the SIGNATURE, so a default changed in the
    body cannot pass."""
    for fn in (_bluestein_2d, _bluestein_centred_2d,
               fresnel_propagate_mft, fraunhofer_propagate_mft,
               angular_spectrum_propagate_mft):
        p = inspect.signature(fn).parameters.get('method')
        assert p is not None, f"{fn.__name__} has no method= keyword"
        assert p.default == 'auto', (
            f"{fn.__name__}'s method default is {p.default!r}, not 'auto'")


def _claim_the_boundary_comes_from_the_constant():
    """The boundary is READ from ``_MFT_DIRECT_MAX_RATIO``, and the two
    documented settings mean what the docstring says.

    No number is asserted.  What is asserted is the RELATION between the
    constant and the rule: a ratio just under the constant is on the dense
    side and a ratio just over it is not, for whatever the constant happens to
    be; and the two named settings give all-or-nothing.  That is what keeps the
    id alive if the maintainer retunes the boundary -- and what makes it fail
    if the rule stops consulting the constant at all.
    """
    r = float(_MFT_DIRECT_MAX_RATIO)
    assert 0.0 < r < 1.0, (
        f"_MFT_DIRECT_MAX_RATIO = {r!r} is not a ratio in (0, 1); a default "
        f"that selects the dense route at M >= N is not the measured rule")

    # A shape exactly AT the boundary is inside (the constant is the largest
    # ratio measured safe, and the comparison is <=).  The probe is SQUARE and
    # scaled by ``k`` so its multiply-adds per kernel entry, ``(N + M)/2``,
    # sit far above the second condition's constant whatever that constant is
    # -- this claim is about the RATIO, and the work screen has its own
    # (round 2, VERIFY-WP-C4 D1).
    n_at = int(round(1.0 / r))
    k = max(1, int(math.ceil(
        4.0 * float(B._MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY) / (n_at + 1.0))))
    assert _auto_selects_direct(k * n_at, k * n_at, k, k), (
        f"the boundary ratio 1/{n_at} itself is not on the dense side at a "
        f"work-dense square shape; the constant names a shape that was "
        f"measured and the comparison is <=")
    # one output sample MORE is outside
    assert not _auto_selects_direct(k * n_at, k * n_at, k + 1, k), (
        f"a ratio just past 1/{n_at} still takes the dense route")
    # ... and on the OTHER axis too: the max over the two decides
    assert not _auto_selects_direct(k * n_at, k * n_at, k, k + 1), (
        "the rule does not take the MAX over the two axes -- one axis past "
        "the boundary must be enough to refuse")

    saved = B._MFT_DIRECT_MAX_RATIO
    try:
        probes = [(96, 96, 3, 3), (24, 24, 12, 12), (64, 64, 64, 64),
                  (8, 8, 8, 8), (1024, 1024, 32, 32),
                  # a THIN shape the work screen refuses at the shipped
                  # constant: ``_MFT_DIRECT_ALWAYS`` has to override BOTH
                  # conditions, not just the ratio (round 2)
                  (2048, 64, 64, 2)]
        B._MFT_DIRECT_MAX_RATIO = _MFT_DIRECT_NEVER
        assert not any(_auto_selects_direct(*s) for s in probes), (
            "_MFT_DIRECT_NEVER does not mean never")
        B._MFT_DIRECT_MAX_RATIO = _MFT_DIRECT_ALWAYS
        assert all(_auto_selects_direct(*s) for s in probes), (
            "_MFT_DIRECT_ALWAYS does not mean always")
        for bad in (-1.0, float('nan')):
            B._MFT_DIRECT_MAX_RATIO = bad
            assert not any(_auto_selects_direct(*s) for s in probes), (
                f"a constant of {bad!r} selects the dense route somewhere; a "
                f"degenerate constant must mean NEVER, not 'whatever the "
                f"comparison does'")
    finally:
        B._MFT_DIRECT_MAX_RATIO = saved


def _dense_work_per_kernel_entry(ny, nx, my, mx):
    """Multiply-adds per transcendental kernel entry, from the code.

    :func:`~lumenairy.propagators._bluestein._direct_matrix_2d` builds
    ``My*Ny + Mx*Nx`` complex ``exp`` entries and then spends
    ``min(My*Ny*Nx + My*Nx*Mx, Ny*Nx*Mx + My*Ny*Mx)`` multiply-adds using them
    -- the two costs that function itself compares to pick its association
    order.  Four integers in, one number out; no float state, no clock.
    """
    entries = my * ny + mx * nx
    flops = min(my * ny * nx + my * nx * mx, ny * nx * mx + my * ny * mx)
    return flops / entries


def _claim_the_work_screen_refuses_a_thin_input():
    """The SECOND condition, asserted as a relation and never as a number.

    WHAT IT IS FOR (round 2, VERIFY-WP-C4 D1).  A rule that reads only the two
    grid ratios cannot separate ``2048x2048 -> 64x64`` from
    ``2048x64 -> 64x2``: the ratio pair is ``(1/32, 1/32)`` at both.  Their
    dense routes differ by more than two decades in multiply-adds per
    transcendental kernel entry (1056 against 4.0), and at the thin one the
    dense route was measured 1.1x to 13.0x SLOWER than ``min(chirp-Z 2-D,
    separable)`` on both builds, under a fully single-threaded instrument, and
    LARGER in memory besides (5.264 MB against 4.399 MB).  Ladder and margins:
    ``validation/probe_c4_round2/r2_workladder_all_{win,wsl}.json``.

    Nothing here types 16.  The shapes are BUILT from the two constants, so a
    maintainer who retunes either one keeps this id -- and a rule that stops
    consulting the work constant fails it.
    """
    r = float(_MFT_DIRECT_MAX_RATIO)
    w = float(B._MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY)
    assert w > 1.0, (
        f"_MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY = {w!r} screens nothing: the "
        f"dense route always spends at least one multiply-add per kernel "
        f"entry, so a value at or under 1 is the pre-round-2 exposure")
    n_at = int(round(1.0 / r))

    # A SQUARE shape at exactly the boundary ratio: its work/entry is
    # ``(N + M)/2 = k*(n_at + 1)/2``, so this ``k`` puts it at or above
    # ``2*w`` for whatever ``w`` is.  It must be CAPTURED.
    k = max(1, int(math.ceil(4.0 * w / (n_at + 1.0))))
    square = (k * n_at, k * n_at, k, k)
    # A THIN shape with the SAME ratio pair: one output sample on x against
    # ``n_at`` input samples.  Its work/entry FALLS with the scale (it tends
    # to ``n_at/k``), so the scale is SCANNED for one that clears the
    # constant downwards by 2x rather than assumed -- the relation has to
    # hold for whatever the maintainer sets the constant to, and hard-failing
    # only when the ladder is exhausted is what keeps that unconditional.
    thin = None
    for kt in range(1, 4097):
        cand = (kt * n_at, n_at, kt, 1)
        if _dense_work_per_kernel_entry(*cand) * 2.0 <= w:
            thin = cand
            break
    assert thin is not None, (
        f"no thin shape at the boundary ratio 1/{n_at} reaches half the work "
        f"constant {w!r} within a 4096x scale scan; the two constants are no "
        f"longer separable by a shape and this claim needs re-deriving")
    assert (max(square[2] / square[0], square[3] / square[1])
            == max(thin[2] / thin[0], thin[3] / thin[1]) == r), (
        f"the two probe shapes {square} and {thin} no longer share the "
        f"boundary ratio pair; the construction behind this claim has "
        f"drifted and it is no longer testing the SECOND condition")
    w_sq = _dense_work_per_kernel_entry(*square)
    w_thin = _dense_work_per_kernel_entry(*thin)
    assert w_thin < w <= w_sq, (
        f"the probe shapes read {w_thin:.2f} and {w_sq:.2f} multiply-adds "
        f"per kernel entry and no longer straddle the constant {w!r}; "
        f"re-derive the shapes before trusting the two assertions below")
    assert _auto_selects_direct(*square), (
        f"the work screen refuses the SQUARE shape {square} "
        f"({w_sq:.1f} multiply-adds per kernel entry, against a constant of "
        f"{w!r}) -- the screen is meant to cost the thin regime and nothing "
        f"else")
    assert not _auto_selects_direct(*thin), (
        f"the rule captures the THIN shape {thin}, whose dense route spends "
        f"only {w_thin:.2f} multiply-adds per transcendental kernel entry "
        f"against a constant of {w!r}.  It has the same ratio pair as "
        f"{square}, which is captured, so the two grid ratios cannot tell "
        f"them apart -- and at this shape family the dense route was "
        f"MEASURED 1.1x to 13.0x slower on both builds and larger in memory")

    # ... and the screen is a CONSTANT-driven relation, not a hard-coded
    # shape list: raise the constant past the square shape and it goes too.
    saved = B._MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY
    try:
        B._MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY = w_sq * 2.0
        assert not _auto_selects_direct(*square), (
            "the rule does not read _MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY: "
            "doubling it past the square shape's own work ratio left that "
            "shape captured")
        B._MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY = 0.0
        assert _auto_selects_direct(*thin), (
            "with the work constant at 0 the screen still refuses the thin "
            "shape, so it is refusing it for some other reason and this "
            "claim is not measuring the second condition")
    finally:
        B._MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY = saved


def _claim_auto_dispatches_as_the_rule_says(shapes, primitive, separable):
    """At every shape, ``'auto'``'s BYTES are one of the two named routes'
    bytes, and the one the rule names.

    The negative half is the one that matters: ``matched is None`` means
    ``'auto'`` produced arithmetic that is NEITHER route -- which is exactly
    what a centred ``'auto'`` would do if it went through the pre-chirp /
    post-chirp / constant decomposition and then into the dense core.
    """
    fn = _bluestein_2d if primitive == 'plain' else _bluestein_centred_2d
    prev = 'separable' if separable else 'bluestein'
    for (ny, nx, my, mx) in shapes:
        E = _rand(ny, nx)
        a = _alpha(ny, nx, my, mx)
        kw = dict(sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2,
                  separable=separable)
        _clear_h_fft_cache()
        auto = fn(E, a, a * 1.5, my, mx, **kw)
        _clear_h_fft_cache()
        dense = fn(E, a, a * 1.5, my, mx, method='direct', **kw)
        _clear_h_fft_cache()
        chirp = fn(E, a, a * 1.5, my, mx, method=prev, **kw)
        matched = ('direct' if _bits(auto) == _bits(dense)
                   else prev if _bits(auto) == _bits(chirp) else None)
        assert matched is not None, (
            f"{primitive} {ny}x{nx}->{my}x{mx} sep={separable}: "
            f"method='auto' returned bytes matching NEITHER 'direct' nor "
            f"{prev!r} -- a third arithmetic")
        says = _auto_selects_direct(ny, nx, my, mx)
        assert (matched == 'direct') == says, (
            f"{primitive} {ny}x{nx}->{my}x{mx} sep={separable}: the rule says "
            f"{'direct' if says else prev}, 'auto' returned {matched}")


def _claim_the_way_back_is_byte_identical(shapes):
    """Naming the previous route gives the previous route's bytes, and setting
    the constant to ``_MFT_DIRECT_NEVER`` gives them for the whole process.

    Measured ON the dense side, where it is a claim: off it, ``'auto'`` is the
    previous route anyway and the assertion would be vacuous.
    """
    saved = B._MFT_DIRECT_MAX_RATIO
    try:
        for (ny, nx, my, mx) in shapes:
            E = _rand(ny, nx)
            a = _alpha(ny, nx, my, mx)
            for sep, prev in ((False, 'bluestein'), (True, 'separable')):
                kw = dict(sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2,
                          separable=sep)
                B._MFT_DIRECT_MAX_RATIO = saved
                _clear_h_fft_cache()
                named = _bluestein_2d(E, a, a * 1.5, my, mx, method=prev,
                                      **kw)
                B._MFT_DIRECT_MAX_RATIO = _MFT_DIRECT_NEVER
                _clear_h_fft_cache()
                by_constant = _bluestein_2d(E, a, a * 1.5, my, mx, **kw)
                assert _bits(named) == _bits(by_constant), (
                    f"{ny}x{nx}->{my}x{mx} sep={sep}: "
                    f"_MFT_DIRECT_NEVER does not reproduce method={prev!r} "
                    f"byte for byte")
    finally:
        B._MFT_DIRECT_MAX_RATIO = saved


def _phase_term_ratio(N, M):
    """``max|t|_chirp / max|t|_dense`` -- the DERIVED lower bound on the
    accuracy gap between the two routes, read off their two kernels.

    UNITS, corrected 2026-09-20 (VERIFY-WP-C4 D6).  The chirp-Z route builds
    ``exp(sign*i*pi*alpha*m^2)`` over the PADDED index ``|m| <= L - N_out``
    with ``L = next_fast_len(N_in + N_out - 1)``, so its phase argument is
    ``pi*alpha*(L - N_out)^2`` RADIANS -- that is ``alpha*(L - N_out)^2 / 2``
    TURNS, because ``pi`` is half a turn, and ``L - N_out >= N - 1``.  The
    dense route builds ``exp(2*pi*i*alpha*n*k)`` with ``n < N`` and ``k < M``,
    which is ``alpha*(N-1)*(M-1)`` turns exactly.  The TURN ratio is therefore
    ``(L - N_out)^2 / (2*(N-1)*(M-1))``, and what this function returns,
    ``N_max^2 / ((N-1)(M-1))``, is about TWICE it: exactly
    ``2*N_max^2 / (L - N_out)^2`` times it, which is 2 up to ``(N/(N-1))^2``
    when ``L = N + M - 1`` and below 2 whenever ``next_fast_len`` pads
    further.  An earlier wording here called the chirp argument
    ``alpha*N_max^2`` and the ``/4`` in the claim below DERIVED; neither is
    right -- the factor of two is a slip, and the ``/4`` is a CHOSEN margin
    on top of it.  The BAR IS UNCHANGED, and still conservative: see the
    claim's own paragraph for the arithmetic.

    Both routes' error is ``C * eps * max|t|`` with the SAME law and different
    constants (VERIFY-WAVE5-HYGIENE2 round 2 D-1, re-measured here), so the
    gap is the TURN ratio times ``C_chirp / C_dense``.  ``alpha`` cancels,
    which is what makes this a property of the SHAPES and not of a fixture.
    """
    return float(max(N, M)) ** 2 / (float(N - 1) * float(max(M - 1, 1)))


def _claim_the_dense_side_is_the_more_accurate_side(shapes):
    """On the dense side, against a reference whose phase is reduced EXACTLY:
    both routes are inside their own derived bars, and the gap between them is
    at least the DERIVED one.

    THE GAP IS DERIVED; THE MARGIN IS CHOSEN -- and the two are labelled
    separately here, which they were not before (VERIFY-WP-C4 D6, corrected
    2026-09-20).  :func:`_phase_term_ratio` returns ``R``, which is about
    TWICE the ratio of the two routes' phase arguments in TURNS; write
    ``R_turns = R/2``.  The gap the two kernels imply is
    ``R_turns * C_chirp/C_dense``, with ``C_chirp/C_dense`` MEASURED in
    [1.481, 4.035] over ten decades of budget at the shipped N=24 -> M=12
    geometry (hygiene-2 round 3, reproduced 2026-09-20 on both builds).  The
    bar asserted is ``R / 4``, which IS ``R_turns / 2`` -- so it sits 3.0x to
    8.1x below the implied gap (``2 x 1.481`` to ``2 x 4.035``).  That factor
    is a chosen margin and not a derivation; the bar itself is unchanged and
    it has a gap on both sides:

    * ABOVE.  MEASURED 2026-09-20 at a budget of 1e3, identical on both
      builds: 96 -> 3 reads a gap of 51 against a bar of 12.1 (4.2x clear),
      128 -> 4 reads 130 against 10.8 (12x), 256 -> 8 reads 244 against 9.2
      (26x), 64 -> 2 reads 233 against 16.3 (14x).
    * BELOW.  A dense arm that quietly returned the chirp-Z answer instead
      reads a gap of ~1.0, which is 12x to 26x UNDER the bar -- which is what
      the ``dense_answer_is_really_separable`` mutation exercises.

    Asserting only ``mx_d < mx_c`` would NOT have that lower gap: the two
    chirp-Z arms differ from each other only in round-off, so which of them is
    nearer the reference at a given fixture is a coin flip and an impostor
    would pass half the time.
    """
    for (ny, nx, my, mx) in shapes:
        assert ny == nx and my == mx, (
            "the exact reference below is built square; pick a square shape")
        E = _rand(ny, nx, seed=496)
        a = _alpha(ny, nx, my, mx)
        ref = _exact_reference(E, a, my)
        _clear_h_fft_cache()
        auto = _bluestein_2d(E, a, a, my, mx, sign=-1, xp=np, fft2=_fft2,
                             ifft2=_ifft2)
        _clear_h_fft_cache()
        chirp = _bluestein_2d(E, a, a, my, mx, sign=-1, xp=np, fft2=_fft2,
                              ifft2=_ifft2, method='bluestein')
        _clear_h_fft_cache()
        dense = _bluestein_2d(E, a, a, my, mx, sign=-1, xp=np, fft2=_fft2,
                              ifft2=_ifft2, method='direct')
        bar_c = _derived_bar(E, ny, my, a, 'chirp')
        bar_d = _derived_bar(E, ny, my, a, 'dense')
        mx_c = float(np.max(np.abs(chirp - ref)))
        mx_d = float(np.max(np.abs(dense - ref)))
        assert mx_c < bar_c, (
            f"{ny}->{my}: the chirp-Z route is {mx_c:.3e} from the exact "
            f"reference, past its derived bar {bar_c:.3e}")
        assert mx_d < bar_d, (
            f"{ny}->{my}: the dense route is {mx_d:.3e} from the exact "
            f"reference, past its derived bar {bar_d:.3e}")
        R = _phase_term_ratio(ny, my)
        gap = mx_c / mx_d if mx_d > 0.0 else float('inf')
        assert gap > R / 4.0, (
            f"{ny}->{my}: the dense route is {mx_d:.3e} from the exact "
            f"reference against the chirp-Z route's {mx_c:.3e}, a gap of "
            f"{gap:.1f} -- under the derived bar R/4 = {R / 4.0:.1f} that the "
            f"two routes' phase arguments imply (R = {R:.1f}).  Either the "
            f"dense arm is not returning the dense answer, or the accuracy "
            f"half of the decision is gone and the rule needs re-deriving")
        assert _bits(auto) == _bits(dense), (
            f"{ny}->{my}: the rule says dense but 'auto' is not the dense "
            f"answer, so the accuracy above is not what a default caller gets")


# ===========================================================================
# 1.  The default, and the boundary
# ===========================================================================

def test_the_shipped_default_is_auto_on_every_entry_point():
    """The flip is in what ``'auto'`` DOES; the keyword itself must still be
    the default, or a caller reading the signature is reading fiction."""
    _claim_the_default_keyword_is_auto()


def test_the_boundary_is_asked_from_the_constant_and_the_two_settings_hold():
    """One constant, three documented behaviours: a ratio, always, never."""
    _claim_the_boundary_comes_from_the_constant()


def test_the_work_screen_refuses_a_thin_input_at_the_boundary_ratio():
    """The SECOND condition: a ratio cannot see a thin input, so the rule
    does not decide on the ratios alone (round 2, VERIFY-WP-C4 D1)."""
    _claim_the_work_screen_refuses_a_thin_input()


def test_the_work_ratio_is_a_pure_function_of_the_four_grid_sizes():
    """The quantity the second condition reads is as build-free as the first,
    and it is NOT determined by the two ratios -- which is why a second
    condition was needed at all.

    ``2048x2048 -> 64x64`` and ``2048x64 -> 64x2`` share the ratio pair
    ``(1/32, 1/32)`` exactly and differ in multiply-adds per transcendental
    kernel entry by more than two decades.  Asserted about the ratios and the
    counts, not about what the rule answers, so a future retune leaves it
    true.
    """
    square = _dense_work_per_kernel_entry(2048, 2048, 64, 64)
    thin = _dense_work_per_kernel_entry(2048, 64, 64, 2)
    assert 64 / 2048 == 2 / 64 == 1.0 / 32.0
    assert square / thin > 100.0, (
        f"the two shapes read {square:.1f} and {thin:.1f} multiply-adds per "
        f"kernel entry; if that spread has closed, the second condition is "
        f"no longer separating anything and the ladder needs re-measuring")
    for _ in range(3):                      # integers in, no float state
        assert _dense_work_per_kernel_entry(2048, 64, 64, 2) == thin
    # and it agrees with the rule's own arithmetic at the boundary
    assert not _auto_selects_direct(2048, 64, 64, 2)
    assert _auto_selects_direct(2048, 2048, 64, 64)


def test_the_selection_reads_nothing_but_the_four_grid_sizes():
    """PURITY: the decision is a function of the shape alone.

    The process is perturbed between calls in every way that is not a shape --
    the clock advanced, the thread-count environment rewritten both ways and
    removed, the library caches filled and dropped, the FFT backend switches
    flipped, the RNG advanced, another thread used -- and the answer must not
    move.  The FALSIFIER is in the id above: the one thing that DOES move it is
    the constant, and a perturbation set that could not change anything would
    not be evidence.

    (The full 30-shape sweep with the digests is
    ``validation/probe_c4_mft_direct/c4_rule_{win,wsl}.json``: 30/30 stable on
    both builds.)
    """
    import gc
    import os
    import threading
    import time

    from lumenairy._cache_registry import clear_all_registered_caches
    from lumenairy.propagators import fft_infra as fi

    keys = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'SCIPY_FFT_WORKERS', 'LUMENAIRY_MEM_BUDGET_MB')
    saved_env = {k: os.environ.get(k) for k in keys}
    saved_use = fi.USE_SCIPY_FFT
    saved_w = fi.SCIPY_FFT_WORKERS

    def _nudge_clock():
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 0.002:
            pass

    perturbations = [
        ('clock', _nudge_clock),
        ('env_hi', lambda: [os.environ.__setitem__(k, '8') for k in keys]),
        ('env_lo', lambda: [os.environ.__setitem__(k, '1') for k in keys]),
        ('env_gone', lambda: [os.environ.pop(k, None) for k in keys]),
        ('caches_dropped', lambda: (clear_all_registered_caches(),
                                    gc.collect())),
        ('rng_advanced',
         lambda: np.random.default_rng().standard_normal(4096)),
        ('scipy_fft_off', lambda: setattr(fi, 'USE_SCIPY_FFT', False)),
        ('scipy_fft_on', lambda: setattr(fi, 'USE_SCIPY_FFT', True)),
        ('workers_one', lambda: setattr(fi, 'SCIPY_FFT_WORKERS', 1)),
        ('workers_all', lambda: setattr(fi, 'SCIPY_FFT_WORKERS', -1)),
    ]
    shapes = _DENSE_SIDE + _CHIRP_SIDE + [(1024, 1024, 32, 32),
                                          (1024, 1024, 33, 32),
                                          # round 2: the second condition is
                                          # perturbed too, on both of its
                                          # sides at the SAME ratio pair
                                          (2048, 64, 64, 2),
                                          (2048, 2048, 64, 64),
                                          (160, 32, 5, 1)]
    try:
        for shape in shapes:
            first = _auto_selects_direct(*shape)
            for name, apply in perturbations:
                apply()
                got = _auto_selects_direct(*shape)
                assert got == first, (
                    f"the selection at {shape} moved from {first} to {got} "
                    f"under the perturbation {name!r} -- it is reading "
                    f"something that is not a grid size")
            box = []
            t = threading.Thread(
                target=lambda: box.append(_auto_selects_direct(*shape)))
            t.start()
            t.join()
            assert box[0] == first, (
                f"the selection at {shape} differs on another thread")
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        fi.USE_SCIPY_FFT = saved_use
        fi.SCIPY_FFT_WORKERS = saved_w


# ===========================================================================
# 2.  The dispatch, on both sides of the boundary
# ===========================================================================

@pytest.mark.parametrize("primitive", ('plain', 'centred'))
@pytest.mark.parametrize("separable", (False, True))
def test_auto_returns_the_route_the_rule_names_on_the_dense_side(primitive,
                                                                 separable):
    """The dense side: ``'auto'`` IS ``method='direct'``, byte for byte."""
    assert _DENSE_SIDE, "no dense-side shape survives the shipped constant"
    _claim_auto_dispatches_as_the_rule_says(_DENSE_SIDE, primitive, separable)


@pytest.mark.parametrize("primitive", ('plain', 'centred'))
@pytest.mark.parametrize("separable", (False, True))
def test_auto_returns_the_route_the_rule_names_on_the_chirp_side(primitive,
                                                                 separable):
    """The chirp side: ``'auto'`` is the route it always was, byte for byte,
    including the ``separable`` flag's own arm."""
    assert _CHIRP_SIDE, "no chirp-side shape survives the shipped constant"
    _claim_auto_dispatches_as_the_rule_says(_CHIRP_SIDE, primitive, separable)


def test_the_public_propagators_take_the_dense_route_at_a_small_output_grid():
    """The rule reaches the three public entry points, not only the
    primitives -- and it reaches them through the SAME decision, so a caller
    can predict the route from ``N_in`` and ``N_out`` without instrumenting
    anything."""
    N, dx, z, N_out = 512, 4e-6, 0.05, 16
    assert _auto_selects_direct(N, N, N_out, N_out), (
        "the fixture below is no longer on the dense side of the boundary")
    E = _gauss(N, dx, 12.0 * dx)
    dx_out = WL * z / (N * dx)
    for fn in (fresnel_propagate_mft, fraunhofer_propagate_mft,
               angular_spectrum_propagate_mft):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = fn(E, z, WL, dx, dx_out, N_out)
            d = fn(E, z, WL, dx, dx_out, N_out, method='direct')
            b = fn(E, z, WL, dx, dx_out, N_out, method='bluestein')
        assert _bits(a) == _bits(d), (
            f"{fn.__name__}: the default is not the dense route at "
            f"{N} -> {N_out}, where the rule says it is")
        assert _bits(a) != _bits(b), (
            f"{fn.__name__}: the default and the chirp-Z route are byte "
            f"identical at {N} -> {N_out}; either the flip did not happen or "
            f"the two routes have stopped being different arithmetic")


def test_the_public_propagators_are_unchanged_at_an_ordinary_focal_zoom_grid():
    """The other side of the same statement, and the one that bounds the blast
    radius: at the focal-zoom grids these propagators are written for
    (``M ~ N``), the default is still the chirp-Z route, byte for byte."""
    N, dx, z = 128, 8e-6, 0.05
    assert not _auto_selects_direct(N, N, N, N)
    E = _gauss(N, dx, 12.0 * dx)
    dx_out = (WL * z / (N * dx)) / 10.0
    for fn in (fresnel_propagate_mft, fraunhofer_propagate_mft,
               angular_spectrum_propagate_mft):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = fn(E, z, WL, dx, dx_out, N)
            b = fn(E, z, WL, dx, dx_out, N, method='bluestein')
        assert _bits(a) == _bits(b), (
            f"{fn.__name__}: an ordinary focal-zoom grid moved")


# ===========================================================================
# 3.  The way back
# ===========================================================================

def test_the_way_back_is_one_keyword_and_one_constant_and_both_are_exact():
    """Byte-identical under ``method='separable'`` / ``'bluestein'`` for one
    call, and under ``_MFT_DIRECT_MAX_RATIO = _MFT_DIRECT_NEVER`` for a whole
    process.  Asserted on the DENSE side, where it is a claim."""
    _claim_the_way_back_is_byte_identical(_DENSE_SIDE[:3])


# ===========================================================================
# 4.  The accuracy decision on the dense side
# ===========================================================================

def test_on_the_dense_side_the_selected_route_is_the_more_accurate_one():
    """The half of the decision that is not about time.

    Against a reference whose phase is reduced EXACTLY, at two dense-side
    shapes, both routes sit inside their own DERIVED bars and the dense route
    is the closer one.  MEASURED 2026-09-20 on both builds at a budget of 1e3:
    96 -> 3 reads 1.66e-13 (chirp-Z) against 3.82e-15 (dense), a factor of
    43.4, and 128 -> 4 reads 1.48e-13 against 3.94e-16, a factor of 376 -- but
    the factor is geometry-dependent and only its SIGN is asserted here.
    """
    shapes = [s for s in _DENSE_SIDE if s[0] == s[1] and s[2] == s[3]][:2]
    assert len(shapes) >= 2, "need two square dense-side shapes"
    _claim_the_dense_side_is_the_more_accurate_side(shapes)


def test_the_exact_reference_is_exact_where_float64_can_check_it():
    """THE PREMISE of the id above, and it has to be measured because a
    reference that shares a route's phase is the failure mode this whole
    construction exists to avoid (round 2, D-1).

    Two-sided.  At a DYADIC ``alpha`` the float64 product ``alpha*n*k`` is
    exact, so the exact reduction and the route's ``t - rint(t)`` must agree to
    the bit.  At a ``alpha`` whose product is lossy they must PART -- otherwise
    the reference is measuring itself.
    """
    def _naive(alpha, n_in, n_out):
        n = np.arange(n_in, dtype=np.float64)
        k = np.arange(n_out, dtype=np.float64)
        t = float(alpha) * k[:, None] * n[None, :]
        return t - np.rint(t)

    for a in (0.125, 0.5, 2.0 ** -7, 3.0 * 2.0 ** -5):
        d = _exact_phase_rows(a, 24, 12) - _naive(a, 24, 12)
        # a whole turn apart is the same complex number; rint's ties-to-even
        # resolves |t| = 1/2 the other way, which is the only legal difference
        assert float(np.max(np.abs(d - np.rint(d)))) == 0.0, (
            f"at the dyadic alpha {a!r} the exact reduction and the route's "
            f"differ by a fraction of a turn -- one of them is wrong")
    a_big = 1e12 / 24.0 ** 2
    d = _exact_phase_rows(a_big, 24, 12) - _naive(a_big, 24, 12)
    parted = float(np.max(np.abs(d - np.rint(d))))
    assert parted > 1e-6, (
        f"at a budget of 1e12 the exact reduction agrees with the route's "
        f"float64 one to {parted:.3e}; the reference is sharing the route's "
        f"phase and can see nothing")


def test_the_default_flip_does_not_take_a_warning_away_from_a_caller():
    """The diagnostic half.  On a shape the rule sends to the dense route, a
    phase budget past the threshold must still WARN under ``'auto'``, and the
    message must name the route actually taken.

    Before 5.49.0 the guard sat after the ``method='direct'`` early return, so
    only a chirp-Z call could reach it.  Selecting the dense route from the
    shapes would then have silenced a caller who was being warned -- which is
    the one thing a default flip may not do.  A caller who NAMES ``'direct'``
    is still silent, and that is the unchanged 5.48 decision, gated in
    ``test_wave5_h2_mft_direct.py``.

    EXACTLY ONE, not "at least one" (round 2, VERIFY-WP-C4 D5).  The guard has
    TWO call sites -- :func:`_bluestein_2d` and :func:`_bluestein_centred_2d`'s
    dense arm -- and a centred ``'auto'`` call at a REFUSED shape reaches the
    inner :func:`_bluestein_2d` while one at a CAPTURED shape returns before
    it.  The code is right and was measured right (exactly one warning at
    144 of 144 driven cases, both builds,
    ``validation/probe_verify_c4/v4_purity_{win,wsl}.json``), but a count of
    ">= 1" is precisely the assertion a two-call-site guard can pass while
    double-warning.  This id now COUNTS.  The same claim on a wider matrix is
    ``test_verify_c4_mft_direct.py::
    test_the_phase_guard_warns_exactly_once_per_call``.
    """
    ny = nx = 96
    my = mx = 3
    assert _auto_selects_direct(ny, nx, my, mx)
    E = _rand(ny, nx)
    a = _alpha(ny, nx, my, mx, budget=B._PHASE_BUDGET_MAX * 10.0)
    for fn in (_bluestein_2d, _bluestein_centred_2d):
        with pytest.warns(RuntimeWarning, match="chirp phase argument") as rec:
            _clear_h_fft_cache()
            fn(E, a, a, my, mx, sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2)
        budget_warnings = [w for w in rec
                           if issubclass(w.category, RuntimeWarning)
                           and 'chirp phase argument' in str(w.message)]
        assert len(budget_warnings) == 1, (
            f"{fn.__name__}: 'auto' emitted {len(budget_warnings)} chirp "
            f"phase-budget warnings, not exactly one.  The guard has two call "
            f"sites and a caller must hear it once per call: "
            f"{[str(w.message)[:60] for w in budget_warnings]}")
        assert 'ALREADY on the dense route' in str(
            budget_warnings[0].message), (
            f"{fn.__name__}: 'auto' warned but the message still advises "
            f"method='direct', which is the route it already took")
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _clear_h_fft_cache()
            fn(E, a, a, my, mx, sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2,
               method='direct')


# ===========================================================================
# 5.  The mutation matrix
# ===========================================================================
#
# Three mutations, each a thing that could plausibly happen to this rule, and
# each named with the id that catches it.  The mutation is applied to the
# SHIPPED module and the SHIPPED claim helper is called, so what is measured is
# the assertion the release actually carries -- not a paraphrase of it written
# to fail.

def _mutate_rule_inverted():
    """The rule answers the opposite of what it measured."""
    original = B._auto_selects_direct

    def inverted(ny, nx, my, mx):
        return not original(ny, nx, my, mx)
    B._auto_selects_direct = inverted
    return lambda: setattr(B, '_auto_selects_direct', original)


def _mutate_constant_silently_zero():
    """Somebody sets the boundary to 0 and the flip quietly never happens."""
    saved = B._MFT_DIRECT_MAX_RATIO
    B._MFT_DIRECT_MAX_RATIO = 0.0
    return lambda: setattr(B, '_MFT_DIRECT_MAX_RATIO', saved)


def _mutate_work_constant_silently_zero():
    """Somebody sets the SECOND constant to 0 and the thin-input screen
    quietly stops screening -- the pre-round-2 exposure, restored in silence.

    It is deliberately the same SHAPE of mistake as
    ``constant_silently_zero`` one condition down, and it needs its own claim
    for the same reason: with the screen off the rule and the dispatch still
    agree with each other perfectly, on a thin shape the dense route is
    measurably slower at.
    """
    saved = B._MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY
    B._MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY = 0.0
    return lambda: setattr(B, '_MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY', saved)


def _mutate_conjunction_or_instead_of_and():
    """The two conditions combined with OR instead of AND.

    The conjunction is the whole safety statement: a shape reaches the dense
    route only when NEITHER condition refuses it.  Under ``or`` a work-dense
    shape far PAST the ratio boundary is captured (``1024x1024 -> 512x512``
    reads 768 multiply-adds per kernel entry and a ratio of 1/2, where the
    dense route is 1.8x slower), and so is a thin shape at the boundary
    ratio.  This mutation replaces the whole rule rather than a constant,
    because the conjunction is not a constant.
    """
    original = B._auto_selects_direct

    def disjoined(ny, nx, my, mx):
        r = float(B._MFT_DIRECT_MAX_RATIO)
        if not (r > 0.0):
            return False
        ny, nx, my, mx = int(ny), int(nx), int(my), int(mx)
        if ny < 1 or nx < 1 or my < 1 or mx < 1:
            return False
        if r == float('inf'):
            return True
        by_ratio = max(my / ny, mx / nx) <= r
        entries = my * ny + mx * nx
        flops = min(my * ny * nx + my * nx * mx, ny * nx * mx + my * ny * mx)
        by_work = flops >= float(
            B._MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY) * entries
        return by_ratio or by_work
    B._auto_selects_direct = disjoined
    return lambda: setattr(B, '_auto_selects_direct', original)


def _mutate_dense_answer_is_really_separable():
    """The rule fires, the dispatch says 'direct', and what comes back is the
    separable chirp-Z answer -- the shape of a copy-paste in the arm."""
    original = B._direct_matrix_2d

    def impostor(E, alpha_x, alpha_y, N_out_y, N_out_x, *, sign, xp,
                 target_cdtype=None, **centres):
        if target_cdtype is None:
            target_cdtype = np.dtype(np.complex128)
        return B._bluestein_2d_separable(
            E, alpha_x, alpha_y, N_out_y, N_out_x, sign=sign,
            target_cdtype=np.dtype(target_cdtype))
    B._direct_matrix_2d = impostor
    return lambda: setattr(B, '_direct_matrix_2d', original)


#: ``{mutation: (apply, the NAMED claim that must catch it)}``.  Naming the
#: claim is the point: "something failed" degrades silently as claims are
#: added, while "THIS claim failed" fails loudly the moment that claim stops
#: discriminating.  The mapping is itself a measured statement -- each
#: mutation was run against all five claims and the one named is the one that
#: refused it, with the reason recorded in the id's docstring.
_MUTATIONS = {
    'rule_inverted': (_mutate_rule_inverted, 'dispatch_dense_side'),
    'constant_silently_zero': (_mutate_constant_silently_zero, 'boundary'),
    'work_constant_silently_zero': (
        _mutate_work_constant_silently_zero, 'work_screen'),
    'conjunction_or_instead_of_and': (
        _mutate_conjunction_or_instead_of_and, 'dispatch_chirp_side'),
    'dense_answer_is_really_separable': (
        _mutate_dense_answer_is_really_separable, 'accuracy'),
}


def _run_every_claim():
    """``[names of the claims that REFUSED]`` -- run them all, catch nothing
    else."""
    caught = []
    for claim, call in (
        ('boundary', _claim_the_boundary_comes_from_the_constant),
        ('work_screen', _claim_the_work_screen_refuses_a_thin_input),
        ('dispatch_dense_side',
         lambda: _claim_auto_dispatches_as_the_rule_says(
             _DENSE_SIDE[:2], 'plain', False)),
        ('dispatch_chirp_side',
         lambda: _claim_auto_dispatches_as_the_rule_says(
             _CHIRP_SIDE[:2], 'plain', False)),
        ('way_back',
         lambda: _claim_the_way_back_is_byte_identical(_DENSE_SIDE[:1])),
        ('accuracy',
         lambda: _claim_the_dense_side_is_the_more_accurate_side(
             [s for s in _DENSE_SIDE if s[0] == s[1] and s[2] == s[3]][:1])),
    ):
        try:
            call()
        except AssertionError:
            caught.append(claim)
    return caught


@pytest.mark.parametrize("name", sorted(_MUTATIONS))
def test_each_mutation_of_the_rule_is_caught_by_a_named_id(name):
    """Every mutation must be refused by the claim named beside it.

    WHICH CLAIM CATCHES WHICH, and why it is that one and not another:

    * ``rule_inverted`` -- the dispatch claim.  ``'auto'`` returns the chirp-Z
      bytes at a shape the rule names dense.  (The way-back claim catches it
      too, because ``_MFT_DIRECT_NEVER`` then means always.)
    * ``constant_silently_zero`` -- the BOUNDARY claim, and only that one.
      With the constant at zero the rule and the dispatch agree with each
      other perfectly; they agree on the wrong thing.  The dispatch claim
      cannot see it, which is exactly why a claim that reads the constant
      directly has to exist.
    * ``work_constant_silently_zero`` -- the WORK-SCREEN claim, and only that
      one, for exactly the reason ``constant_silently_zero`` needs the
      boundary claim: with the second constant at zero the rule and the
      dispatch agree with each other perfectly on a thin shape the dense
      route was measured 1.1x to 13.0x slower at.
    * ``conjunction_or_instead_of_and`` -- a DISPATCH claim, and that is a
      measurement rather than a preference.  This mutation replaces the rule
      FUNCTION on the module, and the claims that call ``_auto_selects_direct``
      by its imported name still hold the original -- the same reason
      ``rule_inverted`` is named for a dispatch claim.  What sees it is the
      library's own call: under ``or`` a chirp-side shape that is work-dense
      (``24x24 -> 12x12`` reads 18 multiply-adds per kernel entry) is captured
      although its ratio is 1/2, so ``'auto'`` returns the dense bytes where
      the rule's ratio half says chirp.  MEASURED 2026-09-20: caught by
      ``dispatch_chirp_side``.
    * ``dense_answer_is_really_separable`` -- the ACCURACY claim.  The dispatch
      claim cannot see this one either: ``'auto'`` and ``method='direct'`` go
      through the same arm, so they still agree byte for byte, and the rule
      still says dense.  What changes is the only thing that can change -- the
      answer's distance from an exact reference, which collapses from a gap of
      51..244 to ~1.  That is what the derived ``R/4`` bar is for.
    """
    apply, expected = _MUTATIONS[name]
    undo = apply()
    try:
        caught = _run_every_claim()
    finally:
        undo()
    assert caught, (
        f"the mutation {name!r} passed every shipped claim in this file; the "
        f"rule is not gated against it")
    assert expected in caught, (
        f"the mutation {name!r} was caught by {caught} but NOT by the claim "
        f"named for it, {expected!r}.  Either that claim has stopped "
        f"discriminating or the mapping is stale -- re-measure before "
        f"editing the expectation")


def test_the_mutation_matrix_is_not_passing_because_everything_fails():
    """The control the matrix above needs: with NO mutation applied, every
    claim it runs must PASS.  Without this, a file in which all five claims
    were broken would read as a perfect mutation score.
    """
    caught = _run_every_claim()
    assert caught == [], (
        f"with no mutation applied, {caught} already refuse; the mutation "
        f"matrix above is measuring a broken file, not a gated one")
