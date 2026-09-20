"""
Bluestein chirp-Z transform (internal helper).

Provides ``_bluestein_2d``, the building block used by Lumenairy's
matrix-Fourier-transform-based propagators

    fresnel_propagate_mft
    fraunhofer_propagate_mft
    angular_spectrum_propagate_mft

to evaluate a 2-D discrete Fourier integral on an arbitrary user-supplied
output grid (different size and pitch from the input grid) without
zero-padding the input.

Algorithm
---------
Given a 2-D input ``E[ny, nx]`` of shape ``(Ny_in, Nx_in)``, evaluate

.. math::

    F[k_y, k_x] = \\sum_{n_y=0}^{N_{y,\\rm in}-1}\\sum_{n_x=0}^{N_{x,\\rm in}-1}
        E[n_y, n_x]\\,
        \\exp\\!\\bigl(\\sigma\\,2\\pi\\,j\\,(\\alpha_x n_x k_x + \\alpha_y n_y k_y)\\bigr)

for ``kx`` in ``[0, N_out_x)`` and ``ky`` in ``[0, N_out_y)``, where
``sigma = +/- 1`` selects the inverse / forward transform direction
respectively.

Bluestein's identity ``n*k = (n^2 + k^2 - (n-k)^2) / 2`` lets us write

.. math::

    F[k] = e^{\\sigma\\pi j\\alpha k^2}
        \\bigl(g \\ast h\\bigr)[k],
        \\quad g[n] = E[n]\\,e^{\\sigma\\pi j\\alpha n^2},
        \\quad h[m] = e^{-\\sigma\\pi j\\alpha m^2}.

The convolution is computed with two zero-padded 2-D FFTs, giving total
cost ``O((N + M) \\log (N + M))`` per axis, where ``N = N_in`` and
``M = N_out``.  This is dramatically faster than a direct matrix-Fourier
transform for typical focal-zoom workflows.  ``O(N^2 M^2)`` is the cost of the
UNFACTORED four-index sum; the transform is separable, so the dense route
(:func:`_direct_matrix_2d`) evaluates it as two matrix products at
``O(M N^2 + M^2 N)``.  The default ``method='auto'`` now SELECTS that
dense route wherever it was measured never slower on either build -- both
output-over-input ratios at or under :data:`_MFT_DIRECT_MAX_RATIO` -- and takes
the chirp-Z reduction everywhere else; see :func:`_auto_selects_direct`.  The
measured time and memory crossover is tabulated in
``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/
WP-C4_MFT_DIRECT_DEFAULT_REPORT.md``, and the route's own measurements in the
``WAVE5_HYGIENE2_REPORT.md`` beside it.

Backends
--------
The 2-D FFTs are dispatched through caller-supplied ``fft2`` / ``ifft2``
callables, so the same helper drives the NumPy / pyFFTW / SciPy / CuPy /
JAX paths used by the rest of :mod:`lumenairy.propagators.propagation`.

The helper is internal (underscore prefix) -- end users should call the
public ``*_propagate_mft`` propagators in
:mod:`lumenairy.propagators.propagation` instead.

Author:  Andrew Traverso
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np
from scipy.fft import next_fast_len

# Chirp-kernel FFT cache: H_FFT = fft2(h_2d) depends only on
# (alpha_x, alpha_y, Nx_in, Ny_in, N_out_x, N_out_y, sign, dtype) -- not
# on the input field -- and costs one of the three FFTs per call.
# NumPy-default-backend entries only (see guard in :func:`_bluestein_2d`).
# Entries are np.copy'd on store AND on hit: the pyFFTW double-buffer
# hands out its internal output buffer, which the very next fft2/ifft2
# call overwrites -- there is zero buffer-ownership slack between the
# G_FFT / H_FFT pair.
_H_FFT_CACHE: 'OrderedDict[tuple, np.ndarray]' = OrderedDict()
_H_FFT_CACHE_MAXSIZE = 16
_H_FFT_CACHE_LOCK = threading.Lock()
_H_FFT_CACHE_HITS = 0

# v5.33.2 BYTE CAPS (audit AUDIT_TRACED_MEMORY_2026_08_09 row 7).  The count
# cap above was the ONLY bound, and one entry is ``L^2`` complex128 with
# ``L = next_fast_len(N_in + N_out - 1)`` -- i.e. it scales with the CALLER's
# grid, not with anything this module controls.  MEASURED at design 121's
# shipped readout shapes: ``N_fine`` 8192 / ``N_out`` 1024 -> ``L`` = 9216 ->
# **1.359 GB per entry, 21.7 GB across the 16**; at ``window_factor`` 7 the
# same geometry gives ``L`` = 17424 -> 4.858 GB per entry and **77.7 GB**.
#
# And on a fan that memory buys nothing.  The key carries ``alpha =
# dx_out/(N_in*dx_in)``, and ``dx_in = dx_fine`` differs per congruence (C-2
# measured per-order readout periods 4734.6..4738.3 um, a 0.08 % spread), so
# every order writes a NEW entry and hits NONE: ``hits = 0`` measured after a
# full production order, and reproduced directly with two orders whose alpha
# differs in the 5th digit (2 entries, 0 hits).
#
# ``fft_infra._H_CACHE`` -- the ASM transfer-function cache two files away --
# has carried exactly these two caps since 3.2.14.1, with a comment recording
# the identical lesson ("At N=32768 each H is 16 GB complex128; without this
# cap, an 8-entry cache can hold up to 128 GB").  The numbers here are that
# cache's, deliberately: this is bringing a sibling cache up to a standard the
# library already sets, not inventing a new policy.  A cache is a cache -- no
# accuracy consequence either way, the entry is recomputed on the next miss.
_H_FFT_CACHE_MAX_BYTES_PER_ENTRY = 2 * 1024 * 1024 * 1024   # 2 GB
_H_FFT_CACHE_MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024       # 8 GB


def _clear_h_fft_cache() -> None:
    """Drop every cached Bluestein chirp-kernel FFT."""
    global _H_FFT_CACHE_HITS
    with _H_FFT_CACHE_LOCK:
        _H_FFT_CACHE.clear()
        _H_FFT_CACHE_HITS = 0


def _h_fft_cache_bytes() -> int:
    """Total bytes currently retained by the chirp-kernel FFT cache."""
    with _H_FFT_CACHE_LOCK:
        return int(sum(int(v.nbytes) for v in _H_FFT_CACHE.values()))


def _h_fft_cache_store(cache_key, H_FFT) -> None:
    """Store one chirp-kernel FFT under the count AND byte bounds.

    Mirrors ``fft_infra._h_cache_store``: an entry larger than
    ``_H_FFT_CACHE_MAX_BYTES_PER_ENTRY`` is NOT stored at all (the transform
    still returns it -- only the retention is skipped), and after any store the
    oldest entries are evicted until both the count and the total-bytes bounds
    hold.  The globals are read at call time so a caller may retune them.

    One deliberate difference from the sibling: eviction stops at ONE entry.
    ``_h_cache_store`` will empty itself if a caller retunes the total cap
    below a single entry's size, which turns the cache into pure overhead (it
    stores, then immediately drops what it stored).  With the shipped caps
    (2 GiB/entry inside 8 GiB total) neither can reach that state.

    v5.33.3 (VERIFY_PERF_BRANCH_2026_08_10 D3): the size test reads
    ``H_FFT.nbytes`` and runs BEFORE the ``np.copy``, exactly as the sibling's
    ``_entry_bytes(H)`` does.  The first cut copied first and rejected second,
    which converted the retention the cap exists to avoid into an equally
    large TRANSIENT for an entry that is thrown away one line later -- 4.86 GB
    at the ``window_factor = 7`` geometry (``L = 17424``) the cap's own comment
    works through, allocated on the run whose peak is the thing being
    defended.  MEASURED with ``tracemalloc`` at a 1 B per-entry cap: retained
    +0.000 MB either way, traced peak +67.109 MB before / +0.000 MB after.
    """
    if int(getattr(H_FFT, 'nbytes', 0)) > int(_H_FFT_CACHE_MAX_BYTES_PER_ENTRY):
        return
    H = np.copy(H_FFT)
    with _H_FFT_CACHE_LOCK:
        _H_FFT_CACHE[cache_key] = H
        total = sum(int(v.nbytes) for v in _H_FFT_CACHE.values())
        while (len(_H_FFT_CACHE) > int(_H_FFT_CACHE_MAXSIZE)
               or total > int(_H_FFT_CACHE_MAX_TOTAL_BYTES)):
            if len(_H_FFT_CACHE) <= 1:
                break
            _, dropped = _H_FFT_CACHE.popitem(last=False)
            total -= int(dropped.nbytes)


try:
    from .._cache_registry import register_cache_clearer as _register_cache_clearer
    _register_cache_clearer('bluestein_h_fft', _clear_h_fft_cache)
except ImportError:
    pass


def _fft_1d(a, axis=-1, inverse=False):
    """1-D FFT along ``axis`` through the library's own NumPy-side dispatch.

    Mirrors ``fft_infra._scipy_or_numpy_fft2``'s choice (SciPy's threaded
    pocketfft when ``USE_SCIPY_FFT``, NumPy otherwise) so the separable
    Bluestein path honours the same backend switches as every other
    transform.  pyFFTW is not consulted: its cached plans are 2-D and a
    1-D plan family would double the resident workspace this module's
    separable path exists to remove.
    """
    from . import fft_infra as _fi
    if _fi.USE_SCIPY_FFT and _fi.SCIPY_FFT_AVAILABLE:
        fn = _fi._scipy_fft.ifft if inverse else _fi._scipy_fft.fft
        return fn(a, axis=axis, workers=_fi.SCIPY_FFT_WORKERS)
    fn = np.fft.ifft if inverse else np.fft.fft
    return fn(a, axis=axis)


def _bluestein_axis_1d(A, alpha, N_out, sign, target_cdtype, axis):
    """One chirp-Z pass along ``axis``: ``N_in -> N_out`` at rate ``alpha``.

    ``sum_n A[..., n] * exp(sign*2*pi*j*alpha*n*k)`` for ``k`` in
    ``[0, N_out)``, by the same Bluestein identity :func:`_bluestein_2d`
    uses -- pre-chirp, circular convolution with the folded kernel on
    ``L = next_fast_len(N_in + N_out - 1)``, post-chirp.  Chirp signals are
    built in float64 and cast to ``target_cdtype`` before multiplication,
    exactly as the 2-D primitive does (this is what keeps the float32 path's
    chirp phase from losing precision at large indices).

    The largest array here is ``(rest x L)`` rather than ``(L x L)``, which
    is the whole point: see :func:`_bluestein_2d`'s ``separable`` parameter.
    """
    A = np.moveaxis(A, axis, -1)
    n_in = int(A.shape[-1])
    N_out = int(N_out)
    alpha = float(alpha)

    n = np.arange(n_in, dtype=np.float64)
    k = np.arange(N_out, dtype=np.float64)
    pre = np.exp(1j * sign * np.pi * alpha * n * n).astype(target_cdtype,
                                                           copy=False)
    post = np.exp(1j * sign * np.pi * alpha * k * k).astype(target_cdtype,
                                                            copy=False)
    L = int(next_fast_len(int(n_in + N_out - 1)))
    m_idx = np.arange(L, dtype=np.int64)
    m_signed = np.where(m_idx < N_out, m_idx, m_idx - L).astype(np.float64)
    h = np.exp(-1j * sign * np.pi * alpha * m_signed
               * m_signed).astype(target_cdtype, copy=False)

    g = np.zeros(A.shape[:-1] + (L,), dtype=target_cdtype)
    g[..., :n_in] = A * pre
    G = _fft_1d(g, axis=-1)
    del g
    # in-place: the kernel product is where a second (rest x L) array would
    # otherwise appear, and (rest x L) is the whole point of this route.
    G *= _fft_1d(h)
    out = _fft_1d(G, axis=-1, inverse=True)[..., :N_out]
    del G
    out = out * post
    if out.dtype != target_cdtype:
        out = out.astype(target_cdtype)
    return np.moveaxis(out, -1, axis)


def _bluestein_2d_separable(E, alpha_x, alpha_y, N_out_y, N_out_x, *,
                            sign, target_cdtype):
    """:func:`_bluestein_2d` as two 1-D chirp-Z passes.  NumPy only."""
    F = _bluestein_axis_1d(E, alpha_x, N_out_x, sign, target_cdtype, -1)
    F = _bluestein_axis_1d(F, alpha_y, N_out_y, sign, target_cdtype, -2)
    return F


#: The three routes :func:`_bluestein_2d` can take through the SAME sum.
#: ``'bluestein'`` is the shipped default and is what every caller took before
#: the direct route existed; ``'separable'`` is the two-pass chirp-Z the
#: ``separable=True`` flag has selected since v5.33.2; ``'direct'`` is the
#: dense matrix-Fourier transform below.
_SUM_METHODS = ('bluestein', 'separable', 'direct')

#: float64 machine epsilon, named once so the budget below is visibly derived
#: from it rather than typed as a number.
_EPS64 = float(np.finfo(np.float64).eps)

#: The chirp phase budget at which the chirp-Z routes still return six
#: significant figures.  DERIVED, 2026-09-19 (VERIFY-WAVE5-HYGIENE2 V-D5):
#: against a ``math.fsum`` correctly-rounded reference at two geometries on
#: both builds, over 23 budgets spanning 11 decades, the chirp-Z relative L2
#: is LINEAR in the budget -- ``rel ~ eps * alpha * N_max^2`` -- so the budget
#: that leaves a relative error of 1e-6 is ``1e-6 / eps = 4.5e9``.
#:
#: THIS IS A CHANGE IN WARNING BEHAVIOUR AND NOT A CHANGE OF ANSWER.  No route
#: moves a byte; a caller who was between the old 1e15 and this 4.5e9 now
#: hears about an error they were already paying.  MEASURED at the old
#: threshold: a budget of 1e12 returned an answer wrong in the FOURTH
#: significant figure (rel 1.9e-04) and said nothing, and the first budget
#: that warned at all was 3.16e15, by which point the answer was 25 % wrong.
#: The natural MFT grids are nowhere near it: ``alpha = zoom/N`` there, so
#: ``budget = zoom*N ~ 1e4`` at N = 1024 with 10x zoom -- five decades of
#: silence for every shipped caller, which
#: ``tests/unit/test_wave5_h2_mft_direct.py`` asserts as its own claim.
_PHASE_BUDGET_MAX = 1e-6 / _EPS64                         # 4.503599627e+09


#: The shape boundary ``method='auto'`` takes the dense route at or below.
#:
#: The rule is ``max(N_out_y/Ny_in, N_out_x/Nx_in) <= _MFT_DIRECT_MAX_RATIO``
#: -- BOTH axes' output-over-input ratios have to sit at or under it -- and it
#: is read by :func:`_auto_selects_direct`, which takes the four grid sizes and
#: nothing else.
#:
#: DERIVED, 2026-09-20 (WP-C4), and not chosen.  The criterion the default
#: flip rests on is "the dense route is NEVER SLOWER on EITHER build",
#: so the constant is the largest ladder ratio at which that holds at EVERY
#: shape on BOTH builds.  Fresh ladder, ``N`` in {64,128,256,512,1024,2048} x
#: ``M`` in {16,32,64,128,256,512,1024} (42 shapes), best of five, cold, the
#: box's load recorded in the probe JSON; then the shapes that decide the
#: boundary re-measured on their own, THREE independent rounds of best-of-nine
#: each, the verdict taken on the WORST round.  The comparison is against
#: ``min(chirp-Z 2-D, separable)`` -- the FASTER of the two routes ``'auto'``
#: could otherwise have taken, because either can be the one a given caller
#: was on.  Worst dense-over-fallback ratio, over all three rounds:
#:
#:      M/N      1/64     1/32     1/16      1/8
#:      WIN     0.206    0.477    0.713    1.427
#:      WSL     0.561    0.954    1.450    2.982
#:
#: 1/16 is NOT safe: on WSL at ``N = 1024, M = 64`` the dense route is 1.38 to
#: 1.45 times SLOWER, in all three rounds.  (That is the shape the two earlier
#: campaigns already disagreed about -- ``WAVE5_HYGIENE2_REPORT.md`` read dense
#: losing there and ``VERIFY_WAVE5_HYGIENE2.md`` read it winning -- which is
#: why it was re-measured on its own rather than read off either.)  1/32 is the
#: conservative crossover BOTH builds support.  Its margin is thin at one shape
#: (WSL ``N = 1024, M = 32``, 0.844 / 0.877 / 0.954 -- dense faster by 5 to
#: 18 %) and wide everywhere else; 1/64 is the ratio with a two-fold margin at
#: every shape, if more headroom is ever wanted.  The full table, with the
#: load, is in ``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/
#: WP-C4_MFT_DIRECT_DEFAULT_REPORT.md``.
#:
#: WHY A RATIO AND NOT A TIME.  The TIME crossover is per-build -- the two
#: builds' crossovers differ by an octave, because scipy's pocketfft drives the
#: separable route's 1-D passes through its own worker pool on Linux
#: (``SCIPY_FFT_WORKERS = -1``, which ``OMP_NUM_THREADS=1`` does not
#: constrain).  A constant read off ONE build's clock is exactly the shape
#: ``docs/TESTING_STANDARDS.md`` calls S1.  This constant is instead the
#: INTERSECTION of the two builds' safe regions, it is compared against a
#: SHAPE at run time and never against a clock, and the MEMORY ordering --
#: which is build-free, and which puts the dense route cheapest at all 42
#: shapes on both builds -- never argues against it anywhere.
#:
#: THE TWO DOCUMENTED SETTINGS.  :data:`_MFT_DIRECT_ALWAYS` (``float('inf')``)
#: means ALWAYS: every shape takes the dense route.  :data:`_MFT_DIRECT_NEVER`
#: (``0.0``) means NEVER: the library goes back to the dispatch this keyword
#: had before this rule -- byte for byte, at every shape -- and so does any
#: value ``<= 0`` or
#: ``nan``.  Both are gated by ``tests/unit/test_c4_mft_direct_default.py``.
_MFT_DIRECT_MAX_RATIO = 1.0 / 32.0

#: The documented "always" and "never" settings of
#: :data:`_MFT_DIRECT_MAX_RATIO`, named so neither has to be typed as a float.
#: Assigning ``_MFT_DIRECT_MAX_RATIO = _MFT_DIRECT_NEVER`` is the supported way
#: back to the previous dispatch for a whole process; ``method='separable'``
#: / ``method='bluestein'`` is the way back for one call.
_MFT_DIRECT_ALWAYS = float('inf')
_MFT_DIRECT_NEVER = 0.0


#: The SECOND condition ``method='auto'`` has to clear, and the one a ratio
#: cannot see.
#:
#: WHY A RATIO IS NOT ENOUGH.  :func:`_direct_matrix_2d` builds
#: ``My*Ny + Mx*Nx`` transcendental kernel entries -- each a complex ``exp``,
#: tens of times the cost of a multiply-add -- and then spends
#: ``min(My*Ny*Nx + My*Nx*Mx, Ny*Nx*Mx + My*Ny*Mx)`` multiply-adds using them
#: (the two costs that function itself compares to pick its association
#: order).  The RATIO of those two numbers, the multiply-adds per kernel
#: entry, is what says whether the BUILD or the products dominate -- and it is
#: not a function of the two grid ratios.  ``2048x2048 -> 64x64`` reads
#: **1056** and ``2048x64 -> 64x2`` reads **4.0**, and both sit at ratio
#: exactly ``(1/32, 1/32)``.  For a square ``N -> M`` the quantity is
#: ``(N + M)/2``, so the ladder :data:`_MFT_DIRECT_MAX_RATIO` was derived from
#: -- square shapes only -- could not see the thin-input regime at all.
#:
#: DERIVED, 2026-09-20 (WP-C4 round 2, VERIFY-WP-C4 D1).  Ladder: 34 CAPTURED
#: anisotropic shapes spanning this quantity from 1.25 to 64 (both
#: orientations, several absolute sizes per decade) plus 17 square /
#: non-dyadic / mildly anisotropic control shapes; two independent rounds of
#: best-of-nine, routes INTERLEAVED with the order rotating per repeat, cold
#: before every repeat, ``fft_infra.SCIPY_FFT_WORKERS = 1`` so both sides are
#: single-threaded, verdict on the WORST round, on BOTH builds.  Against
#: ``min(chirp-Z 2-D, separable)``:
#:
#:  * the largest work/entry at which the dense route was measured SLOWER on
#:    EITHER build is **11.95** (``2048x128 -> 64x4``: 1.110 on WSL py3.12,
#:    0.924 on Windows py3.14);
#:  * the smallest at which it was measured safe on BOTH builds ABOVE that is
#:    **16.00** (``256x64 -> 4x1``: 0.310 / 0.294).
#:
#: 16.0 is therefore the LARGEST value that still captures every shape
#: measured safe above the slower region, and it clears the slower region by
#: **1.34x**.  It refuses 10 of the 10 shapes measured slower (by 1.11x to
#: 13.03x) and keeps 17 of 17 control shapes; the six shapes the whole shipped
#: suite drives read 264, 516, 520, 1028, 1044 and 2052, so none of them
#: moves.  Full table:
#: ``validation/probe_c4_round2/r2_workladder_all_{win,wsl}.json``.
#:
#: IT IS A ONE-SIDED SCREEN, NOT A CROSSOVER, and that costs coverage.  The
#: readings are not monotone in it -- ``512x32 -> 16x1`` reads 2.99 and is
#: safe (0.569 / 0.582), because at that absolute size the chirp-Z route's
#: fixed costs (planning, padding to ``next_fast_len``) dominate whatever the
#: asymptotic count says -- so a threshold refuses shapes that would have been
#: fine: 11 of the 34 thin shapes here were safe on both builds and are
#: refused anyway.  That is the correct direction for a rule whose premise is
#: "never slower": refusing a safe shape costs a few per cent of time, and
#: capturing an unsafe one cost up to 13x on this ladder.
#:
#: A non-positive or ``nan`` value means the screen refuses nothing that the
#: ratio admitted, which is the exposure this rule was widened to close,
#: and is what the
#: ``constant_silently_zero`` mutation in
#: ``tests/unit/test_c4_mft_direct_default.py`` exercises.
_MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY = 16.0


def _auto_selects_direct(Ny_in, Nx_in, N_out_y, N_out_x) -> bool:
    """Does ``method='auto'`` take the dense route at this shape?

    A pure function of the FOUR grid sizes and :data:`_MFT_DIRECT_MAX_RATIO`.
    It reads no clock, no environment variable, no thread count, no backend, no
    array CONTENTS and no module state other than that one constant -- so the
    route a call takes is reproducible from its signature alone, on every build
    and every backend.  That is the property the per-shape byte-identity claim
    rests on: "this shape goes to the previous route" is a statement about the
    shape, not about the run.  Gated by
    ``tests/unit/test_c4_mft_direct_default.py::
    test_the_selection_reads_nothing_but_the_four_grid_sizes``.

    Parameters
    ----------
    Ny_in, Nx_in : int
        Input grid size.
    N_out_y, N_out_x : int
        Output grid size.

    Returns
    -------
    bool
        ``True`` when BOTH conditions hold.  (1) Both per-axis ratios
        ``N_out / N_in`` sit at or below :data:`_MFT_DIRECT_MAX_RATIO`.  The
        MAX of the two is taken, which is the conservative reading on an
        anisotropic grid: the axis with the larger ratio decides, so a shape
        reaches the dense route only when NEITHER axis is past the boundary.
        (2) The dense route spends at least
        :data:`_MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY` multiply-adds per
        transcendental kernel entry.  A ratio cannot see a THIN input: at
        ratio ``(1/32, 1/32)`` the dense route was measured 1.1x to 13.0x
        SLOWER once the other axis is short, because it is then paying more
        transcendentals than multiply-adds.

    Notes
    -----
    The comparison is ``<=`` and the constant is the largest ratio MEASURED
    safe, rather than ``<`` against the first ratio measured unsafe, so the
    boundary ratio itself is inside the dense region and the constant names a
    shape that was actually timed.  Every ladder ratio is dyadic, so
    ``N_out / N_in`` at the boundary is exact in float64 and the comparison has
    no tie to resolve.

    A non-positive or ``nan`` constant means NEVER, and that is decided BEFORE
    the division, so a mis-set constant cannot reach the arithmetic.

    BOTH CONDITIONS ARE BUILD-FREE (WP-C4 round 2, VERIFY-WP-C4 D1).  The
    second one counts multiply-adds and kernel entries with the SAME
    expressions :func:`_direct_matrix_2d` uses to pick its association order,
    so it is four integers and one constant like the first -- no clock, no
    environment, no backend.
    """
    r = float(_MFT_DIRECT_MAX_RATIO)
    if not (r > 0.0):                    # 0.0, negative, or nan -> never
        return False
    ny, nx = int(Ny_in), int(Nx_in)
    my, mx = int(N_out_y), int(N_out_x)
    if ny < 1 or nx < 1 or my < 1 or mx < 1:
        return False
    if r == float('inf'):                # the documented "always"
        return True
    if not (max(my / ny, mx / nx) <= r):
        return False
    # A ratio cannot see a THIN input: at ratio (1/32, 1/32) the dense route
    # is 1.1x to 13.0x slower once the other axis is short, because it is then
    # paying more transcendentals than multiply-adds.  MEASURED 2026-09-20
    # (WP-C4 round 2, ``validation/probe_c4_round2/r2_workladder_*.json``);
    # the constant's own block has the ladder and the margin.
    entries = my * ny + mx * nx
    flops = min(my * ny * nx + my * nx * mx, ny * nx * mx + my * ny * mx)
    return flops >= float(_MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY) * entries


def _warn_phase_budget(alpha_x, alpha_y, Nx_in, Ny_in, N_out_x, N_out_y, *,
                       on_dense: bool, stacklevel: int) -> float:
    """The chirp phase-budget guard, in ONE place, and the budget it read.

    The guard's THRESHOLD and its LAW are unchanged (see
    :data:`_PHASE_BUDGET_MAX` and :func:`_bluestein_2d`'s Notes): the relative
    error of every route here is ``~eps * |alpha| * N_max^2``, linear in the
    budget over eleven decades, so the threshold is read off the law at the
    accuracy wanted rather than set at the point where the chirp wraps.

    WHAT WP-C4 CHANGED, and why.  Before the shape rule, the guard sat between the
    ``method='direct'`` early return and the chirp-Z arms, so only a chirp-Z
    call could reach it.  With ``'auto'`` now able to choose the dense route
    from the shapes, leaving it there would mean a caller who was being warned
    at a high budget goes SILENT on a shape the new rule captures -- a
    diagnostic removed by a default flip, which is the one thing a default flip
    may not do.  So ``'auto'`` evaluates the guard BEFORE it chooses, on the
    budget, and the message names the route it then takes.

    EXPLICIT ``method='direct'`` IS UNCHANGED AND STILL SILENT.  That is the
    shipped decision from hygiene-2 -- warning on the one route the warning's
    own advice names would be a false positive -- and it stays gated
    two-sidedly by ``tests/unit/test_wave5_h2_mft_direct.py::
    test_the_chirp_phase_guard_fires_on_the_chirp_route_and_not_the_dense_one``.

    ``on_dense=False`` reproduces the PREVIOUS message BYTE FOR BYTE, which
    is what keeps the byte-identity claim true for the fixtures that warn.
    """
    N_max = max(int(Nx_in), int(Ny_in), int(N_out_x), int(N_out_y))
    alpha_max = max(abs(float(alpha_x)), abs(float(alpha_y)))
    phase_budget = alpha_max * float(N_max) ** 2
    if phase_budget <= _PHASE_BUDGET_MAX:
        return phase_budget
    import warnings
    head = (
        f"Bluestein chirp phase argument ~{phase_budget:.1e} exceeds the "
        f"float64 chirp budget {_PHASE_BUDGET_MAX:.1e}; the chirp-Z "
        f"routes' relative error at this budget is "
        f"~{phase_budget * _EPS64:.1e}.  EVERY route here follows "
        f"rel ~ eps * budget with budget = |alpha| * N_max^2, so the way "
        f"out is a smaller BUDGET and not a different route: fewer "
        f"samples on whichever of N_in / N_out sets N_max, or a smaller "
        f"|alpha| -- for the MFT propagators alpha = dx*dx_out/(lambda "
        f"z), so a finer output pitch, a finer input pitch or a longer z "
        f"-- or a regular FFT propagator, which spends no such phase at "
        f"all.  ")
    if on_dense:
        tail = (
            f"This call is ALREADY on the dense route: method='auto' chose "
            f"it from the shapes alone (both N_out/N_in at or under "
            f"{float(_MFT_DIRECT_MAX_RATIO):.6g}), and it is the more "
            f"accurate of the two at the SAME budget -- but only by a "
            f"bounded factor (MEASURED 2026-09-20, 1.5x .. 11.8x over ten "
            f"decades of budget at N=24 -> M=12 on both index conventions "
            f"and both builds), so it is a smaller error and not an escape "
            f"from this one.  method='bluestein' / 'separable' name the "
            f"chirp-Z routes, which are the LESS accurate ones here.")
    else:
        tail = (
            "method='direct' is the more accurate route at the SAME "
            "budget, but only by a bounded factor (MEASURED 2026-09-20, "
            "1.5x .. 11.8x over ten decades of budget at N=24 -> M=12 on "
            "both index conventions and both builds), so it is a smaller "
            "error and not an escape from this one.")
    warnings.warn(head + tail, RuntimeWarning, stacklevel=stacklevel)
    return phase_budget



def _direct_matrix_2d(
    E,
    alpha_x: float,
    alpha_y: float,
    N_out_y: int,
    N_out_x: int,
    *,
    sign: int,
    xp,
    target_cdtype=None,
    n_centre_in_x: float = 0.0,
    n_centre_in_y: float = 0.0,
    k_centre_out_x: float = 0.0,
    k_centre_out_y: float = 0.0,
):
    """The direct matrix-Fourier transform: the SAME sum as
    :func:`_bluestein_2d` / :func:`_bluestein_centred_2d`, evaluated as two
    dense matrix products instead of a chirp-Z reduction.

    Computes::

        F[ky, kx] = sum_{ny, nx} E[ny, nx]
                    * exp(sign*2*pi*j * alpha_x * (nx - cIx) * (kx - cOx))
                    * exp(sign*2*pi*j * alpha_y * (ny - cIy) * (ky - cOy))

    which factors exactly as ``F = Wy . E . Wx^T`` with
    ``Wx[kx, nx] = exp(sign*2*pi*j*alpha_x*(nx - cIx)*(kx - cOx))`` and ``Wy``
    its y counterpart.  The non-centred primitive's convention is the case
    ``cIx = cIy = cOx = cOy = 0``, so this one function serves both.

    WHY IT EXISTS.  The module docstring and
    :func:`~lumenairy.propagators.mft.fresnel_propagate_mft`'s Notes have named
    the direct matrix-Fourier transform as the chirp-Z reduction's alternative
    since the MFT propagators were written, without shipping it.  Two things
    make it worth having as an OPT-IN rather than only as prose:

    * **Memory.**  The chirp-Z route pads to ``L = next_fast_len(N + M - 1)``
      per axis and holds several ``L``-sized working arrays; the dense route
      holds ``Mx*Nx + My*Ny`` kernel entries, one intermediate and the output,
      and no padding at all.  Below the crossover the dense route is the SMALLER
      one -- see ``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/
      WAVE5_HYGIENE2_REPORT.md`` for the measured table.
    * **Accuracy.**  The dense route reduces its phase argument modulo one turn
      before calling ``exp`` (below), so it does not spend float64 mantissa on
      a phase of ``pi*alpha*N^2`` radians the way the chirp signals do.  It is
      therefore the natural reference for the two chirp-Z reductions, and the
      report derives their agreement bar against it.  It is the more accurate
      route at a given budget by a BOUNDED factor and not by decades -- see
      the Notes below and :func:`_bluestein_2d`'s.

    THIS IS NOW THE DEFAULT ROUTE AT SMALL OUTPUT GRIDS (WP-C4).
    ``method='auto'`` -- the default on both primitives and on all three public
    MFT entry points -- selects it when both ``N_out / N_in`` ratios sit at or
    under :data:`_MFT_DIRECT_MAX_RATIO`, the largest ladder ratio at which the
    dense route was MEASURED never slower than the faster of the two chirp-Z
    fallbacks, on BOTH builds, in three independent rounds.  Everywhere else
    ``'auto'`` takes the route it always took, byte for byte.  The selection is
    :func:`_auto_selects_direct`, a pure function of the four grid sizes.

    It MOVES ANSWERS on the shapes it captures -- the three routes agree to
    round-off, not bit for bit -- which is why it carries a Migration note and
    why the way back is one keyword (``method='separable'`` or
    ``'bluestein'``) or one constant (``_MFT_DIRECT_MAX_RATIO =
    _MFT_DIRECT_NEVER``).  An earlier wording here said "It is NOT the default
    anywhere and nothing in the library selects it automatically"; the reason
    it gave -- "at the shapes the MFT propagators are written for the chirp-Z
    route wins on time by one to three orders of magnitude" -- is true at
    ``M ~ N`` and false by a factor of 2 to 20 at ``M <= N/32``, which is the
    region the rule captures and nothing else.

    Parameters
    ----------
    E : ndarray, complex 2-D
        Input of shape ``(Ny_in, Nx_in)`` in the ``xp`` namespace.
    alpha_x, alpha_y : float
        Sampling-rate parameters, as in :func:`_bluestein_2d`.
    N_out_y, N_out_x : int
        Output grid size.
    sign : int
        ``+1`` (inverse FT) or ``-1`` (forward FT).
    xp : module
        Array namespace -- ``numpy``, ``cupy`` or ``jax.numpy``.  The dense
        products run through ``xp.matmul``, so the whole evaluation stays on
        the caller's device / tracer.
    target_cdtype : numpy dtype, optional
        Complex dtype of the output.  Inferred from ``E`` when ``None``.
    n_centre_in_x, n_centre_in_y, k_centre_out_x, k_centre_out_y : float
        Input / output index centres.  Default ``0.0`` -- i.e. the
        non-centred convention of :func:`_bluestein_2d`.

    Returns
    -------
    F : ndarray, complex, shape ``(N_out_y, N_out_x)``.

    Notes
    -----
    **Phase construction, and why the budget still applies** (corrected
    2026-09-20, VERIFY-WAVE5-HYGIENE2 round 2 D-1).
    ``t = alpha*(n - cI)*(k - cO)`` is formed in float64 and then reduced by
    ``t - rint(t)``, which is EXACT for ``|t| <= 2**52`` (``rint(t)`` is an
    integer and the difference is a multiple of ``ulp(t)``, so no bit is lost
    in the subtraction).  Only the fractional turn reaches ``exp``.  The
    irreducible error is the two roundings in forming ``t`` itself, amplified
    by ``2*pi`` -- and THAT is the phase budget: ``eps * |t|`` is
    ``eps * |alpha| * n * k <= eps * budget``, so this route follows the same
    ``rel ~ eps * budget`` law :func:`_bluestein_2d`'s guard is derived from,
    with a smaller constant.  An earlier wording drew the opposite conclusion
    from the same two sentences ("so the ``pi*alpha*N^2`` phase-budget warning
    :func:`_bluestein_2d` carries does not apply to this route"); MEASURED
    against an exactly-reduced reference at N=24 -> M=12, this route reads
    6.9e-12 / 1.1e-07 / 8.6e-05 / 6.7e-02 at budgets 1e5 / 1e9 / 1e12 / 1e15,
    identical on both builds.  What it does NOT pay is the chirp signals'
    extra factor of 1.5 to 11.8.

    **Association order.**  The two products are taken in whichever order costs
    fewer multiply-adds, decided from the four grid sizes ALONE (a pure
    function of the shapes, so the same call always associates the same way):
    ``(Wy . E) . Wx^T`` costs ``My*Ny*Nx + My*Nx*Mx`` and ``Wy . (E . Wx^T)``
    costs ``Ny*Nx*Mx + My*Ny*Mx``.  The two orders differ in the last bits of
    the answer, which is the same statement the ``separable`` route carries.

    **Cost.**  ``O(My*Ny*Nx + My*Nx*Mx)`` multiply-adds -- ``O(N^3)`` for a
    square ``N = M`` grid, against the chirp-Z route's
    ``O(L^2 log L)``.  The module docstring's ``O(N^2 M^2)`` describes the
    UNFACTORED four-index sum; the transform is separable, so the dense route
    never has to pay that.
    """
    if sign not in (+1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign}")

    Ny_in, Nx_in = E.shape
    N_out_y = int(N_out_y)
    N_out_x = int(N_out_x)
    if N_out_y < 1 or N_out_x < 1:
        raise ValueError(
            f"N_out must be positive, got ({N_out_y}, {N_out_x})")

    if target_cdtype is None:
        target_cdtype = np.dtype(E.dtype) if xp.iscomplexobj(E) \
            else np.dtype(np.complex128)
    target_cdtype = np.dtype(target_cdtype)

    def _kernel(alpha, n_in, n_out, c_in, c_out):
        n = np.arange(int(n_in), dtype=np.float64) - float(c_in)
        k = np.arange(int(n_out), dtype=np.float64) - float(c_out)
        t = float(alpha) * k[:, None] * n[None, :]
        # Exact for |t| <= 2**52: rint(t) is an integer and t - rint(t) is a
        # multiple of ulp(t).  See the Notes.
        t = t - np.rint(t)
        W = np.exp(1j * sign * 2.0 * np.pi * t)
        return W.astype(target_cdtype, copy=False)

    Wx_np = _kernel(alpha_x, Nx_in, N_out_x, n_centre_in_x, k_centre_out_x)
    Wy_np = _kernel(alpha_y, Ny_in, N_out_y, n_centre_in_y, k_centre_out_y)
    if xp is np:
        Wx, Wy = Wx_np, Wy_np
    else:
        Wx, Wy = xp.asarray(Wx_np), xp.asarray(Wy_np)
    del Wx_np, Wy_np

    A = E if E.dtype == target_cdtype else E.astype(target_cdtype)
    cost_y_first = N_out_y * Ny_in * Nx_in + N_out_y * Nx_in * N_out_x
    cost_x_first = Ny_in * Nx_in * N_out_x + N_out_y * Ny_in * N_out_x
    if cost_y_first <= cost_x_first:
        F = xp.matmul(xp.matmul(Wy, A), Wx.T)
    else:
        F = xp.matmul(Wy, xp.matmul(A, Wx.T))
    if F.dtype != target_cdtype:
        F = F.astype(target_cdtype)
    return F


def _bluestein_2d(
    E,
    alpha_x: float,
    alpha_y: float,
    N_out_y: int,
    N_out_x: int,
    *,
    sign: int,
    xp,
    fft2,
    ifft2,
    target_cdtype=None,
    separable: bool = False,
    method: str = 'auto',
):
    """2-D Bluestein chirp-Z transform.

    Computes::

        F[ky, kx] = sum_{ny, nx} E[ny, nx]
                    * exp(sign * 2*pi*j * (alpha_x * nx * kx + alpha_y * ny * ky))

    over indices ``nx, ny, kx, ky`` all starting at zero.  Use
    propagator wrappers in :mod:`propagation` to handle centred
    conventions (``(n - N/2)`` shifts).

    Parameters
    ----------
    E : ndarray, complex 2-D
        Input array of shape ``(Ny_in, Nx_in)``.  The array library
        (NumPy / CuPy / JAX) is determined by ``xp``.
    alpha_x, alpha_y : float
        Sampling-rate parameters in the Bluestein sum (cycles per
        index pair on each axis).  For a Fresnel propagator,
        ``alpha = dx_in * dx_out / (lambda * z)``.
    N_out_y, N_out_x : int
        Output grid size.
    sign : int
        ``+1`` for the inverse-FT direction (e.g., the IFT step in
        :func:`angular_spectrum_propagate_mft`).
        ``-1`` for the forward-FT direction (e.g., the FT step in
        :func:`fresnel_propagate_mft`).
    xp : module
        Array namespace -- ``numpy``, ``cupy``, or ``jax.numpy``.  The
        same module that produced ``E``.
    fft2, ifft2 : callable
        2-D FFT and inverse FFT functions appropriate for ``xp``.  For
        NumPy this is typically ``lumenairy.propagators.propagation._fft2``
        (which dispatches to pyFFTW when applicable); for CuPy /
        JAX, pass ``cp.fft.fft2`` / ``jnp.fft.fft2`` etc.
    target_cdtype : numpy dtype, optional
        Complex dtype of the output array.  Inferred from ``E`` when
        ``None``.  Chirp signals are computed in float64 for accuracy
        and cast to ``target_cdtype`` before multiplication; this
        avoids float32 chirp-phase precision loss at large indices.
    separable : bool, default False
        Evaluate the SAME sum as two 1-D chirp-Z passes (v5.33.2, audit
        ``AUDIT_TRACED_MEMORY_2026_08_09`` row 6) instead of one 2-D
        convolution.  NumPy backends only -- with any other ``xp`` this is
        ignored and the 2-D path runs, because the 1-D transforms would have
        to be dispatched through a caller-supplied ``fft2`` that only does
        two axes at once.

        WHY IT EXISTS.  The 2-D path pads BOTH axes to
        ``L = next_fast_len(N_in + N_out - 1)``, so every working array is
        ``L^2`` -- and the transform is EXACTLY separable, which the code
        already knows (it builds the kernel as ``h_y[:, None] * h_x[None, :]``).
        Two 1-D passes give the same sum with a largest array of
        ``(N_in x L)``, and they also drop the ``L^2`` chirp-kernel cache
        entry: two length-``L`` vectors (0.15 MB) replace a 1.359 GB array at
        design 121's shipped readout shape.  MEASURED on a tapered beam:

        ======================  ==========  ==========  =========  =========
        N_in / N_out (L)        2-D peak    sep. peak   rel L2     time
        ======================  ==========  ==========  =========  =========
        2048 / 256   (L=2304)   0.854 GB    0.255 GB    8.6e-16    0.15x
        4096 / 1024  (L=5120)   3.412 GB    1.343 GB    9.1e-16    0.42x
        ======================  ==========  ==========  =========  =========

        ACCURACY: **not byte-identical** -- it is a different association
        order for the same sum, so the difference is round-off:
        ``rel L2 <= 9.1e-16``, ``max|delta|/max|F| 1.2e-15``, power ratio
        1.000000000000.  Callers that pin bits must leave this False; that is
        why it is opt-in here and why the one shipped consumer
        (:func:`~lumenairy.propagators.carrier.carrier_referenced_exact_focus_readout`)
        carries its own default-ON switch with the 2-D path one flag away.
    method : {'auto', 'bluestein', 'separable', 'direct'}, default 'auto'
        Which route through the SAME sum to take.

        ``'auto'`` (the default) DECIDES FROM THE SHAPES: it
        takes :func:`_direct_matrix_2d`, the dense matrix-Fourier transform,
        when :func:`_auto_selects_direct` says both ``N_out / N_in`` ratios sit
        at or under :data:`_MFT_DIRECT_MAX_RATIO`, and otherwise reproduces the
        historical dispatch exactly and byte for byte -- the separable two-pass
        route when ``separable=True`` and ``xp is numpy``, the 2-D convolution
        otherwise.  The decision is a pure function of the four grid sizes and
        that one constant: no clock, no environment, no backend, no array
        contents, so it is identical on every build and every backend.

        ``'bluestein'`` and ``'separable'`` name the two chirp-Z arms
        explicitly and are the WAY BACK for one call (``'separable'`` still
        falls back to the 2-D arm off NumPy, for the reason the ``separable``
        entry gives).  ``'direct'`` names the dense route explicitly, and
        differs from an ``'auto'`` that selected it in exactly one respect: it
        is SILENT at a phase budget past :data:`_PHASE_BUDGET_MAX`, where
        ``'auto'`` warns (see :func:`_warn_phase_budget`).

        The routes agree to round-off, NOT bit for bit, so a shape the rule
        captures MOVES in its last bits.  Setting
        ``_MFT_DIRECT_MAX_RATIO = _MFT_DIRECT_NEVER`` restores the previous
        dispatch for a whole process.  The measured crossover table is in
        ``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/
        WP-C4_MFT_DIRECT_DEFAULT_REPORT.md``.

    Returns
    -------
    F : ndarray, complex
        Output array of shape ``(N_out_y, N_out_x)`` in the same
        backend as ``E``.

    Notes
    -----
    Numerical sensitivity, MEASURED rather than reasoned about (2026-09-19).
    The chirp value ``exp(sign*pi*j*alpha*n^2)`` carries a phase up to
    ``pi * |alpha| * N_max^2``, and the error this costs is LINEAR in that
    budget -- there is no cliff to sit just below.  Against a ``math.fsum``
    correctly-rounded reference at N=24 M=12 and at N=48 M=24, over 23 budgets
    spanning 11 decades on both builds, the chirp-Z routes' relative L2 reads

        budget   1e8     1e10    1e12    1e14    1e15    1e17
        rel L2   1.5e-08 2.0e-06 1.9e-04 1.3e-02 2.5e-01 1.6e+00

    i.e. ``rel ~ eps * budget`` to within a small factor over the whole range.

    THE DENSE ROUTE OBEYS THE SAME LAW (corrected 2026-09-20,
    VERIFY-WAVE5-HYGIENE2 round 2 D-1).  An earlier wording here said it
    "reads 2.8e-16 .. 4.5e-16 at EVERY budget tested, because it reduces
    ``t`` by ``t - rint(t)`` before calling ``exp`` and therefore has no chirp
    phase to lose".  The premise is true and the conclusion is not: the
    REDUCTION is exact, but the float64 product ``alpha*(n - cI)*(k - cO)``
    that it reduces has already discarded the low bits of a value needing
    ~63 of them, and no later reduction recovers a bit that is gone.  So the
    dense route's phase error is ``~eps * |alpha| * n * k <= eps * budget``
    -- the same law, with a smaller constant.  The 2.8e-16 reading was a
    measurement of the INSTRUMENT: a ``math.fsum`` reference that forms
    ``t`` the same way agrees with the dense route by construction.

    MEASURED 2026-09-20 against a reference whose phase is reduced EXACTLY
    (``fractions.Fraction``), N=24 -> M=12, identical to the digit on both
    builds (``validation/probe_wave5_hyg2_round3/r3_budget_exact.py``):

        budget            1e5     1e9     1e12    1e15
        chirp-Z rel L2    2.8e-11 1.6e-07 1.9e-04 2.3e-01
        dense rel L2      6.9e-12 1.1e-07 8.6e-05 6.7e-02

    on the non-centred convention, with fitted slopes of 0.96 (chirp-Z) and
    0.99 (dense) over nine and ten decades.  The dense route is the more
    accurate of the two at every budget, by a factor of 1.5 .. 4.0 on that
    convention and 4.4 .. 11.8 on the centred one -- a bounded factor, not
    an immunity.  ``method='direct'`` therefore buys a smaller error at the
    same budget; only a smaller ``|alpha| * N_max^2`` buys a smaller budget.

    The guard's threshold is derived from that law and not from taste:
    ``_PHASE_BUDGET_MAX = 1e-6 / eps ~ 4.5e9`` is the budget at which six
    significant figures still remain.  The historical ``1e15`` was 7.5 decades
    late -- at a budget of 1e12 the route returned an answer wrong in the
    fourth significant figure and said nothing.  Shipped callers stay silent:
    at the natural MFT grids ``alpha = zoom/N``, so ``budget = zoom*N`` is of
    order 1e4 at N = 1024 with 10x zoom, five decades below the threshold.
    """
    if sign not in (+1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign}")
    if method not in ('auto',) + _SUM_METHODS:
        raise ValueError(
            f"method must be one of {('auto',) + _SUM_METHODS}, got "
            f"{method!r}")

    Ny_in, Nx_in = E.shape
    N_out_y = int(N_out_y)
    N_out_x = int(N_out_x)
    if N_out_y < 1 or N_out_x < 1:
        raise ValueError(
            f"N_out must be positive, got ({N_out_y}, {N_out_x})")

    if target_cdtype is None:
        target_cdtype = np.dtype(E.dtype) if xp.iscomplexobj(E) \
            else np.dtype(np.complex128)
    target_cdtype = np.dtype(target_cdtype)

    # ----- 0a) the dense route, asked for by name ---------------------------
    # Taken BEFORE the chirp phase-budget guard below, and SILENT.  The guard
    # is scoped to the chirp signals' float64 phase, which this route does not
    # build; what that scoping is NOT is a statement that this route is
    # accurate at that budget.  MEASURED 2026-09-20 (round 2 D-1) at 2.2x the
    # threshold on the shipped N=24 -> M=12 fixture: chirp-Z 2.028e-06 and
    # warning, dense 1.079e-06 and SILENT -- i.e. the dense route is quiet at
    # a budget where it has itself passed the 1e-6 the threshold is named for,
    # by a factor of 1.9 rather than by decades.  Widening the guard to a
    # caller who NAMED this route is a behaviour change owed to the maintainer
    # and is still an open item in ``docs/audits/
    # AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WAVE5_HYGIENE2_REPORT.md``;
    # the scoping is gated two-sidedly by ``tests/unit/
    # test_wave5_h2_mft_direct.py::
    # test_the_chirp_phase_guard_fires_on_the_chirp_route_and_not_the_dense_one``
    # so it cannot change silently either way.
    if method == 'direct':
        return _direct_matrix_2d(
            E, alpha_x, alpha_y, N_out_y, N_out_x,
            sign=sign, xp=xp, target_cdtype=target_cdtype)

    # ----- 0b) which arm 'auto' takes, decided from the SHAPES alone --------
    # WP-C4.  ``'auto'`` takes the dense route where it was measured
    # never slower on EITHER build -- see :func:`_auto_selects_direct` and
    # :data:`_MFT_DIRECT_MAX_RATIO`.  The decision is a pure function of the
    # four grid sizes and that one constant: no clock, no environment, no
    # backend, no array contents.  Every other shape takes the route this
    # dispatch has always taken, byte for byte.
    auto_direct = (method == 'auto'
                   and _auto_selects_direct(Ny_in, Nx_in, N_out_y, N_out_x))

    # Numerical-precision guard.  The chirp signal exp(sign*pi*j*alpha*n^2)
    # has phase up to pi * |alpha| * N_max^2, and the relative error that
    # costs is LINEAR in that budget (`rel ~ eps * budget`, measured over 11
    # decades at two geometries on both builds -- see the Notes).  The
    # threshold is therefore read off the law at the accuracy wanted rather
    # than set at the point where the chirp wraps incoherently.
    #
    # It runs BEFORE the ``'auto'`` dense return below, and not after, so that
    # the default flip cannot take a warning away from a caller who was
    # getting one: the budget is a property of the CALL, and both routes pay
    # it (round 2 D-1).  ``on_dense`` only changes which route the message's
    # last sentence names -- see :func:`_warn_phase_budget`.
    _warn_phase_budget(alpha_x, alpha_y, Nx_in, Ny_in, N_out_x, N_out_y,
                       on_dense=auto_direct, stacklevel=3)

    if auto_direct:
        return _direct_matrix_2d(
            E, alpha_x, alpha_y, N_out_y, N_out_x,
            sign=sign, xp=xp, target_cdtype=target_cdtype)

    # ----- 0) separable route (v5.33.2) --------------------------------------
    # Same sum, two 1-D passes, ``(N_in x L)`` instead of ``L^2``.  Taken
    # AFTER the precision guard above so both routes warn identically, and
    # only on NumPy (see the ``separable`` docstring entry).
    if (separable if method == 'auto' else method == 'separable') \
            and xp is np:
        return _bluestein_2d_separable(
            E, alpha_x, alpha_y, N_out_y, N_out_x,
            sign=sign, target_cdtype=target_cdtype)

    # ----- 1) Bluestein chirp signals (computed in float64 for accuracy) -----
    # Pre / post / kernel use the identity
    #   exp(sigma*2*pi*j*alpha*n*k)
    #     = exp(sigma*pi*j*alpha*n^2)
    #     * exp(sigma*pi*j*alpha*k^2)
    #     * exp(-sigma*pi*j*alpha*(n-k)^2)
    # so the convolution kernel has the OPPOSITE sign of the pre/post chirps.

    n_x = np.arange(Nx_in, dtype=np.float64)
    n_y = np.arange(Ny_in, dtype=np.float64)
    k_x = np.arange(N_out_x, dtype=np.float64)
    k_y = np.arange(N_out_y, dtype=np.float64)

    pre_x_np  = np.exp(1j * sign * np.pi * float(alpha_x) * n_x * n_x)
    pre_y_np  = np.exp(1j * sign * np.pi * float(alpha_y) * n_y * n_y)
    post_x_np = np.exp(1j * sign * np.pi * float(alpha_x) * k_x * k_x)
    post_y_np = np.exp(1j * sign * np.pi * float(alpha_y) * k_y * k_y)

    # ----- 2) Convolution kernel and FFT lengths -----
    # Linear convolution of length-N input with kernel covering m in
    # [-(N-1), M-1] requires output length >= N + M - 1; round up to a
    # cache-friendly FFT-fast length via scipy.fft.next_fast_len (numpy
    # types only -- this is integer arithmetic, no dispatch needed).
    Lx = int(next_fast_len(int(Nx_in + N_out_x - 1)))
    Ly = int(next_fast_len(int(Ny_in + N_out_y - 1)))

    # Fold the kernel into circular indexing on the padded length.
    # Indices [0 .. N_out-1] hold h[0..N_out-1]; indices
    # [Lx-(Nx_in-1) .. Lx-1] hold h[-(Nx_in-1)..-1].  The middle
    # "unused" region is set to whatever exp gives at those indices --
    # since g_pad is zero there, those values don't enter the answer.
    m_x_idx = np.arange(Lx, dtype=np.int64)
    m_x_signed = np.where(m_x_idx < N_out_x, m_x_idx, m_x_idx - Lx).astype(np.float64)
    h_x_np = np.exp(-1j * sign * np.pi * float(alpha_x) * m_x_signed * m_x_signed)

    m_y_idx = np.arange(Ly, dtype=np.int64)
    m_y_signed = np.where(m_y_idx < N_out_y, m_y_idx, m_y_idx - Ly).astype(np.float64)
    h_y_np = np.exp(-1j * sign * np.pi * float(alpha_y) * m_y_signed * m_y_signed)

    # ----- 3) Move to xp + cast to target dtype -----
    def _to_xp(arr_np_complex):
        a = arr_np_complex.astype(target_cdtype, copy=False)
        if xp is np:
            return a
        return xp.asarray(a)

    pre_x  = _to_xp(pre_x_np)
    pre_y  = _to_xp(pre_y_np)
    post_x = _to_xp(post_x_np)
    post_y = _to_xp(post_y_np)
    h_x    = _to_xp(h_x_np)
    h_y    = _to_xp(h_y_np)

    # ----- 4) Modulate input by the pre-chirp + zero-pad to (Ly, Lx) -----
    # Use xp.pad so the JAX path stays functional (no in-place .at[].set).
    g = E * (pre_y[:, None] * pre_x[None, :])
    g_pad = xp.pad(
        g,
        pad_width=((0, Ly - Ny_in), (0, Lx - Nx_in)),
        mode='constant',
        constant_values=0,
    )
    if g_pad.dtype != target_cdtype:
        g_pad = g_pad.astype(target_cdtype)

    # ----- 5) FFT-based circular convolution with the chirp kernel -----
    # The 2-D kernel is separable: h[my, mx] = h_y[my] * h_x[mx].
    G_FFT = fft2(g_pad)
    # Serve H_FFT from the module cache when possible.  NumPy default
    # path ONLY: the fft2 callable is caller-supplied (CuPy / JAX /
    # custom), and keying on anything else would pin device arrays in a
    # module-global or return results from a different transform.
    global _H_FFT_CACHE_HITS
    H_FFT = None
    cache_key = None
    if xp is np:
        from .fft_infra import _fft2 as _default_np_fft2
        if fft2 is _default_np_fft2:
            cache_key = (float(alpha_x), float(alpha_y), Nx_in, Ny_in,
                         N_out_x, N_out_y, sign, str(target_cdtype))
            with _H_FFT_CACHE_LOCK:
                cached = _H_FFT_CACHE.get(cache_key)
                if cached is not None:
                    _H_FFT_CACHE.move_to_end(cache_key)
                    _H_FFT_CACHE_HITS += 1
                    H_FFT = np.copy(cached)
    if H_FFT is None:
        h_2d = h_y[:, None] * h_x[None, :]
        H_FFT = fft2(h_2d)
        if cache_key is not None:
            _h_fft_cache_store(cache_key, H_FFT)
    CONV  = ifft2(G_FFT * H_FFT)

    # ----- 6) Extract the first (N_out_y, N_out_x) block and apply post-chirp -----
    block = CONV[:N_out_y, :N_out_x]
    F = block * (post_y[:, None] * post_x[None, :])

    # Ensure the output dtype matches the target.  Some backends promote
    # complex64 to complex128 through FFT pairs; force back here so the
    # caller's precision contract is honoured.
    if F.dtype != target_cdtype:
        F = F.astype(target_cdtype)
    return F


def _bluestein_centred_2d(
    E,
    alpha_x: float,
    alpha_y: float,
    N_out_y: int,
    N_out_x: int,
    *,
    n_centre_in_x: float | None = None,
    n_centre_in_y: float | None = None,
    k_centre_out_x: float | None = None,
    k_centre_out_y: float | None = None,
    sign: int,
    xp,
    fft2,
    ifft2,
    target_cdtype=None,
    separable: bool = False,
    method: str = 'auto',
):
    """2-D Bluestein with centred input AND output index conventions.

    Computes::

        F[ky, kx] = sum_{ny, nx} E[ny, nx]
                    * exp(sign*2*pi*j * alpha_x * (nx - cIx) * (kx - cOx))
                    * exp(sign*2*pi*j * alpha_y * (ny - cIy) * (ky - cOy))

    where ``cIx, cIy`` are input centres (default ``Nx_in/2, Ny_in/2``)
    and ``cOx, cOy`` are output centres (default ``Nx_out/2, Ny_out/2``).

    The non-centred primitive :func:`_bluestein_2d` computes
    ``sum E[n] * exp(sign*2*pi*j*alpha*n*k)``.  Centring expands as

    .. math::

        (n - cI)(k - cO) = nk - n\\cdot cO - k\\cdot cI + cI\\cdot cO,

    and the three correction factors fold cleanly into a pre-chirp
    (depends on ``n``), a post-chirp (depends on ``k``), and a
    multiplicative constant.

    Parameters
    ----------
    E : ndarray, complex 2-D
    alpha_x, alpha_y : float
    N_out_y, N_out_x : int
    n_centre_in_x, n_centre_in_y : float, optional
        Defaults to ``Nx_in / 2`` and ``Ny_in / 2`` respectively.  Pass
        a non-default value (e.g. ``0``) to use a left-anchored input.
    k_centre_out_x, k_centre_out_y : float, optional
        Defaults to ``N_out_x / 2`` and ``N_out_y / 2``.  Non-integer
        values are valid -- they correspond to a sub-pixel shift of the
        output centre.
    sign : int
        ``+1`` (inverse FT) or ``-1`` (forward FT).
    xp, fft2, ifft2, target_cdtype, separable
        Same as :func:`_bluestein_2d`.  The centring corrections are
        themselves separable (a pre-chirp in ``n``, a post-chirp in ``k`` and
        a constant), so ``separable`` changes only the core primitive.
    method : {'auto', 'bluestein', 'separable', 'direct'}, default 'auto'
        Same as :func:`_bluestein_2d`, including the shape rule, with
        one difference that matters: the dense route does NOT go through the
        pre-chirp / post-chirp / constant decomposition above, because
        :func:`_direct_matrix_2d` takes the index centres themselves and builds
        the centred kernel in one step.  The decomposition and the one-step
        build are the same sum; they are not the same bits.

        That is why the dense arm is taken HERE, before the decomposition, for
        BOTH ``'direct'`` and an ``'auto'`` the rule sends to it: routing an
        ``'auto'`` dense call through the decomposition and then into the dense
        core would produce a THIRD set of bits, matching neither the one-step
        dense build nor the chirp-Z route.  ``'auto'`` asks
        :func:`_auto_selects_direct` on the same four grid sizes the inner
        :func:`_bluestein_2d` call would ask it on, so the two primitives never
        disagree about which arm a shape takes.

    Returns
    -------
    F : ndarray, complex 2-D, shape ``(N_out_y, N_out_x)``.
    """
    # VOCABULARY BEFORE GEOMETRY, and in the SAME order as _bluestein_2d
    # (V-D17, 2026-09-19).  These two lines used to sit the other way round,
    # so ``_bluestein_centred_2d([[1+0j, 2+0j]], ..., method='bogus')`` raised
    # ``AttributeError: 'list' object has no attribute 'shape'`` where
    # ``_bluestein_2d`` with the same arguments raised the designed
    # ``ValueError`` -- and with ``sign=0, method='bogus'`` the two primitives
    # named DIFFERENT first errors.  A caller who mistypes a keyword should be
    # told which keyword, by whichever primitive they reached.
    if sign not in (+1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign}")
    if method not in ('auto',) + _SUM_METHODS:
        raise ValueError(
            f"method must be one of {('auto',) + _SUM_METHODS}, got "
            f"{method!r}")
    Ny_in, Nx_in = E.shape
    if n_centre_in_x is None:
        n_centre_in_x = Nx_in / 2.0
    if n_centre_in_y is None:
        n_centre_in_y = Ny_in / 2.0
    if k_centre_out_x is None:
        k_centre_out_x = N_out_x / 2.0
    if k_centre_out_y is None:
        k_centre_out_y = N_out_y / 2.0

    if target_cdtype is None:
        target_cdtype = np.dtype(E.dtype) if xp.iscomplexobj(E) \
            else np.dtype(np.complex128)
    target_cdtype = np.dtype(target_cdtype)

    # The dense route: the centred kernel in ONE build, no decomposition.
    # Reached by name (``method='direct'``) or by the shape rule under
    # ``'auto'`` -- and it has to be reached HERE in both cases, because going
    # through the pre-chirp / post-chirp / constant decomposition below and
    # then into the dense core would be a THIRD arithmetic, agreeing with
    # neither the one-step dense build nor the chirp-Z route bit for bit.
    # ``'auto'`` asks the same :func:`_auto_selects_direct` on the same four
    # grid sizes that :func:`_bluestein_2d` would ask on the decomposed call,
    # so the two primitives never disagree about which arm a shape takes.
    auto_direct = (method == 'auto'
                   and _auto_selects_direct(Ny_in, Nx_in, N_out_y, N_out_x))
    if method == 'direct' or auto_direct:
        if auto_direct:
            # Same reason as in :func:`_bluestein_2d`: the default flip may
            # not silence a diagnostic.  A caller who NAMED the route stays
            # silent, which is the unchanged shipped decision.
            _warn_phase_budget(alpha_x, alpha_y, Nx_in, Ny_in,
                               N_out_x, N_out_y, on_dense=True, stacklevel=3)
        return _direct_matrix_2d(
            E, alpha_x, alpha_y, N_out_y, N_out_x,
            sign=sign, xp=xp, target_cdtype=target_cdtype,
            n_centre_in_x=n_centre_in_x, n_centre_in_y=n_centre_in_y,
            k_centre_out_x=k_centre_out_x, k_centre_out_y=k_centre_out_y)

    n_x = np.arange(Nx_in, dtype=np.float64)
    n_y = np.arange(Ny_in, dtype=np.float64)
    k_x = np.arange(N_out_x, dtype=np.float64)
    k_y = np.arange(N_out_y, dtype=np.float64)

    # Decompose (n - cI)(k - cO) = nk - n*cO - k*cI + cI*cO.
    # Then exp(sign*2*pi*j*alpha*(n-cI)(k-cO))
    #   = [exp(sign*2*pi*j*alpha*n*k)]                       (Bluestein core)
    #   * [exp(-sign*2*pi*j*alpha*n*cO)]                      (pre-chirp on n)
    #   * [exp(-sign*2*pi*j*alpha*k*cI)]                      (post-chirp on k)
    #   * [exp(sign*2*pi*j*alpha*cI*cO)]                      (constant)

    pre_x_np  = np.exp(-1j * sign * 2 * np.pi * float(alpha_x)
                        * n_x * float(k_centre_out_x))
    pre_y_np  = np.exp(-1j * sign * 2 * np.pi * float(alpha_y)
                        * n_y * float(k_centre_out_y))
    post_x_np = np.exp(-1j * sign * 2 * np.pi * float(alpha_x)
                        * k_x * float(n_centre_in_x))
    post_y_np = np.exp(-1j * sign * 2 * np.pi * float(alpha_y)
                        * k_y * float(n_centre_in_y))
    const = (np.exp(1j * sign * 2 * np.pi * float(alpha_x)
                     * float(n_centre_in_x) * float(k_centre_out_x))
             * np.exp(1j * sign * 2 * np.pi * float(alpha_y)
                       * float(n_centre_in_y) * float(k_centre_out_y)))

    def _to_xp(arr_np_complex):
        a = arr_np_complex.astype(target_cdtype, copy=False)
        if xp is np:
            return a
        return xp.asarray(a)

    pre_x  = _to_xp(pre_x_np)
    pre_y  = _to_xp(pre_y_np)
    post_x = _to_xp(post_x_np)
    post_y = _to_xp(post_y_np)
    const_c = target_cdtype.type(const)

    # Modulate input by pre-chirp and call the Bluestein primitive.
    E_mod = E * (pre_y[:, None] * pre_x[None, :])
    F_core = _bluestein_2d(
        E_mod, alpha_x, alpha_y, N_out_y, N_out_x,
        sign=sign, xp=xp, fft2=fft2, ifft2=ifft2,
        target_cdtype=target_cdtype, separable=separable, method=method,
    )
    F = F_core * (post_y[:, None] * post_x[None, :]) * const_c
    return F


def _mft_route_kwargs(mft_method) -> dict:
    """``{}`` when the caller named no MFT route, ``{'method': ...}`` when
    they did -- the ONE place the ``mft_method=`` pass-through is turned into
    a call.

    WHY A HELPER AND NOT ``method=mft_method or 'auto'`` AT EACH SITE.  Seven
    public entry points (``compute_psf``, ``resample_field``, ``propagate``'s
    asm / fresnel / fraunhofer legs, both carrier focus readouts,
    ``re_reference`` and ``propagate_traced_carrier_chain``) already spend the
    name ``method`` on something else -- a sampler, a resampler, a propagator
    family -- so each of them exposes ``mft_method=`` instead and forwards it
    here.  ``None`` is the "the caller named nothing" sentinel and it STAMPS
    NOTHING: the keyword is left off the call entirely, so whatever the
    primitive's own default is at the time governs, and a future change to
    that default reaches these callers without eight edits.  Passing
    ``mft_method='auto'`` explicitly is NOT the same statement -- it pins the
    name -- even though today the two produce the same bytes.

    Gated by ``tests/unit/test_c4_round2_mft_method.py::
    test_mft_method_none_stamps_nothing_on_the_primitive``.
    """
    return {} if mft_method is None else {'method': mft_method}


__all__ = ['_bluestein_2d', '_bluestein_centred_2d',
           '_direct_matrix_2d', '_auto_selects_direct',
           '_mft_route_kwargs']
