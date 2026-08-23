"""Niche D14 (2026-08-22) -- the ``carrier='auto'`` fit is DETERMINISTIC BY
CONSTRUCTION at any BLAS thread count.

WHAT THIS FILE PINS, and why none of it could be written before.
``docs/audits/BUILD_LENS_32K_MEMORY_2026_08_22.md`` S4.2 caught
``carrier='auto'`` failing to reproduce ITSELF -- 2 distinct field hashes over
four identical N = 4096 calls, and again over eight calls on shipped 5.39.1 --
and traced it to a threaded BLAS reduction inside the fit that does not fix its
partitioning across calls.  While that held, NO byte-identity assertion on this
path was writable at all: an A/B across a call that varies run to run cannot
attribute anything to what was varied.  The tests below are that assertion,
finally available:

  1. THE NEW CLAIM -- the same hash at OMP/BLAS widths {1, 2, 4, 8} and over
     repeated calls, on ONE build, through BOTH consumers (the analytic
     ``apply_real_lens`` entry and a traced-path carrier fit).
  2. The fail-before, as a matched pair on whatever rung of a row-count ladder
     this build's BLAS actually splits.
  3. Deterministic-vs-shipped agreement inside a bar DERIVED from the legal
     partition family's own distance to a correctly-rounded oracle.
  4. Oracle-accuracy NON-REGRESSION on the fit's own design matrix.
  5. Footprint non-regression against the 32k wave's banded fit profile.
  6. THE SCOPE, pinned: the traced chain runs six least-squares solves and
     this niche owns ONE of them.  The end-to-end traced FIELD is therefore
     NOT claimed thread-invariant -- two 120-term residual-eikonal fits stay
     on the BLAS route by design and were measured moving with the width.
     Asserting the field would have passed intermittently.

=============================================================================
THE CROSS-BUILD NON-CLAIM.  THIS IS A COMMENT, NOT AN ASSERTION, ON PURPOSE.
=============================================================================
Nothing here claims -- and no test in this repository may assert -- that the
carrier fit produces the SAME BITS on two different builds or platforms.  It
does not, and it is not supposed to.  The per-block partials are ``np.sum``
over contiguous float64, whose pairwise block size, SIMD width and unroll are
properties of the NumPy build and the CPU it dispatched for; ``**``, ``sqrt``
and ``angle`` upstream of the fit go through the platform libm.  A different
libm or codegen may legally differ in the last ULP, and a test whose pass/fail
boundary sits inside that spread is per-build by the definition in
``docs/TESTING_STANDARDS.md``.  The property being pinned is exactly:

    ONE build, ANY thread count, ANY number of repeats -> the same bytes.

Every thread-invariance assertion below therefore compares hashes produced by
subprocesses of THIS interpreter against each other, never against a stored
constant.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap
import tracemalloc

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_traced as LT

_LAM = 1.31e-6
_WIDTHS = ('1', '2', '4', '8')
_REPS = 2

#: Repo root, taken from the LIVE import rather than from ``PYTHONPATH`` or
#: ``sys.path[0]``.  BUILD_LENS_32K_MEMORY S9 records the trap this closes: a
#: runner that inserts a sibling directory at ``sys.path[0]`` beats
#: ``PYTHONPATH`` outright, so a child launched with ``PYTHONPATH`` alone can
#: silently import a DIFFERENT library while looking correctly pinned.  Every
#: child below asserts its own ``lumenairy.__file__`` against this.
_ROOT = os.path.realpath(os.path.join(os.path.dirname(la.__file__), os.pardir))


def _child_env(width):
    """Environment for one arm: the BLAS width pinned in ALL THREE variables
    (OpenBLAS reads its own, and the OpenMP one is what a threadpool-capped
    parent would have moved), plus the library root."""
    env = dict(os.environ)
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS'):
        env[var] = str(width)
    env['PYTHONPATH'] = _ROOT + os.pathsep + env.get('PYTHONPATH', '')
    env['LUMENAIRY_ROOT'] = _ROOT
    env.pop('LUMENAIRY_MEM_BUDGET_MB', None)
    return env


_PREAMBLE = textwrap.dedent(
    """
    import hashlib, os, sys, warnings
    import numpy as np
    import lumenairy as la
    from lumenairy.elements import _lens_traced as LT
    _want = os.path.realpath(os.path.join(os.environ['LUMENAIRY_ROOT'],
                                          'lumenairy'))
    _got = os.path.realpath(os.path.dirname(la.__file__))
    assert _got == _want, 'imported %r, expected %r' % (_got, _want)
    LT.DETERMINISTIC_NORMAL_EQUATIONS = bool(int(sys.argv[1]))

    def _h(a):
        return hashlib.sha256(
            np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]
    """
)

#: The fixture the consumers share.  Two properties are load-bearing and both
#: are ASSERTED by the child, not assumed:
#:
#: * the fit must see enough rows for this build's BLAS to reach its threading
#:   threshold on the reduction -- measured on 2026-08-22 (Windows 11,
#:   py3.14.6, numpy 2.4.4 / scipy-openblas 0.3.31, Ryzen 9 5950X), the ``A.T
#:   @ b`` GEMV at 5 terms is width-invariant at 20 000 rows and splits from
#:   100 000 up.  This fixture produces 119 936;
#: * the carrier must stay BELOW the grid Nyquist tilt over the whole bright
#:   support, or R6's connected-core restriction throws the support away and
#:   the design matrix collapses to ~1 600 rows -- which is below the splitting
#:   threshold, and the fail-before then silently does not fire.  Caught doing
#:   exactly that with R = 0.045 m; R = 1.0 m keeps the whole support.
_FIXTURE = textwrap.dedent(
    """
    LAM, N, DX, R_CARRIER = 1.31e-6, 512, 30e-6, 1.0
    _x = (np.arange(N) - N / 2) * DX
    _X, _Y = np.meshgrid(_x, _x)
    _w = 80 * DX
    E0 = (np.exp(-(_X * _X + _Y * _Y) / (_w * _w))
          * np.exp(1j * (2 * np.pi / LAM) * (_X * _X + _Y * _Y)
                   / (2 * R_CARRIER))).astype(np.complex128)
    PRESC = {'wavelength': LAM, 'aperture_diameter': 14e-3, 'surfaces': [
        {'radius': 51.68e-3, 'thickness': 4e-3, 'glass_before': 'air',
         'glass_after': 'N-BK7', 'semi_diameter': 7e-3},
        {'radius': -51.68e-3, 'thickness': 0.0, 'glass_before': 'N-BK7',
         'glass_after': 'air', 'semi_diameter': 7e-3}],
        'thicknesses': [4e-3], 'stop_index': 0}
    """
)

_BODIES = {
    # the shared function itself -- W, and both gradient components
    'fit': """
    W, grad_fn, _wfn = LT._compute_carrier('auto', E0, LAM, DX, _X, _Y)
    L, M = grad_fn(_X, _Y)
    OUT = np.concatenate([np.ravel(W), np.ravel(L), np.ravel(M)])
    """,
    # CONSUMER 1: the analytic entry, carrier='auto' in apply_real_lens
    'analytic': """
    OUT = np.asarray(la.apply_real_lens(
        E0, prescription=PRESC, wavelength=LAM, dx=DX,
        surface_model='tangent_facet', carrier='auto'))
    """,
    # CONSUMER 2: a traced-path carrier fit -- the fit AS THE TRACED CHAIN
    # CALLS IT, captured through the real ``apply_real_lens_traced`` entry.
    #
    # WHY THE CAPTURE AND NOT THE RETURNED FIELD.  Measured (see
    # ``test_the_traced_chain_has_a_second_reduction_this_niche_does_not_own``)
    # the traced chain runs SIX least-squares solves, and the carrier fit is
    # one of them: two 120-term residual-eikonal fits stay on the BLAS route
    # by design, and those DO move with the thread width.  Hashing the exit
    # field would therefore be asserting a property this change does not
    # deliver -- and it would pass intermittently, which is worse.  What is
    # claimed, and checked here, is that the carrier fit reached through the
    # traced entry is bit-identical at every width.
    'traced': """
    _cc, _cap = LT._compute_carrier, []

    def _spy_cc(carrier, E_in, wavelength, dx, Xg, Yg, *a, **k):
        _r = _cc(carrier, E_in, wavelength, dx, Xg, Yg, *a, **k)
        if isinstance(carrier, str) and carrier == 'auto' and not _cap:
            _L, _M = _r[1](Xg, Yg)
            _cap.append(np.concatenate([np.ravel(np.asarray(_L)),
                                        np.ravel(np.asarray(_M))]))
        return _r

    LT._compute_carrier = _spy_cc
    try:
        la.apply_real_lens_traced(
            E0, prescription=PRESC, wavelength=LAM, dx=DX,
            carrier='auto', on_undersample='silent')
    finally:
        LT._compute_carrier = _cc
    assert _cap, 'the traced chain never reached the auto carrier fit'
    OUT = _cap[0]
    """,
}


def _run_consumer(consumer, deterministic, width, reps=_REPS, save=None):
    """Hash ``consumer``'s output ``reps`` times in ONE fresh interpreter at
    the given BLAS width.  Returns the list of hashes."""
    driver = textwrap.dedent(
        """
        _save = os.environ.get('D14_SAVE')
        for _r in range({reps}):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                __BODY__
            if _save and _r == 0:
                np.save(_save, np.asarray(OUT))
            print('HASH ' + _h(OUT))
        """
    ).format(reps=int(reps))
    # Splice at the placeholder's OWN indentation, read from the dedented
    # driver rather than assumed -- a hard-coded width silently produced a
    # body whose continuation lines were over-indented, which Python accepts
    # inside parentheses and rejects otherwise (two of the three consumers
    # ran; the third raised IndentationError).
    hit = next(ln for ln in driver.splitlines() if '__BODY__' in ln)
    pad = ' ' * (len(hit) - len(hit.lstrip()))
    body = textwrap.indent(
        textwrap.dedent(_BODIES[consumer]).strip(), pad).lstrip()
    code = _PREAMBLE + _FIXTURE + driver.replace('__BODY__', body)
    env = _child_env(width)
    if save is not None:
        env['D14_SAVE'] = str(save)
    proc = subprocess.run(
        [sys.executable, '-c', code, str(int(bool(deterministic)))],
        env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, (
        f"{consumer} arm (det={deterministic}, width={width}) failed:\n"
        f"{proc.stdout}\n{proc.stderr}")
    hashes = [ln.split()[1] for ln in proc.stdout.splitlines()
              if ln.startswith('HASH ')]
    assert len(hashes) == reps, proc.stdout + proc.stderr
    return hashes


# ===========================================================================
# 1.  THE NEW CLAIM: one build, any thread count, any number of repeats.
# ===========================================================================

@pytest.mark.slow
@pytest.mark.parametrize('consumer', ['fit', 'analytic', 'traced'])
def test_the_auto_carrier_hashes_the_same_at_every_thread_count(consumer):
    """THE ASSERTION THAT WAS IMPOSSIBLE TO WRITE BEFORE THIS FIX.

    Eight calls -- four BLAS widths x two repeats -- of the SAME call on the
    SAME data, each in a fresh interpreter with the width pinned before NumPy
    loads, must return one distinct SHA-256.  ``'fit'`` is the shared function
    (:func:`~lumenairy.elements._lens_traced._compute_carrier`) directly;
    ``'analytic'`` is the returned FIELD of ``apply_real_lens(carrier='auto',
    surface_model='tangent_facet')``; ``'traced'`` is the carrier fit as
    ``apply_real_lens_traced(carrier='auto')`` reaches it.  The fix lives in
    the shared function, so the consumers are VERIFICATION BREADTH, not
    duplicate code -- and all three were nondeterministic before it.

    Measured 2026-08-22 on the reference box, det OFF: three distinct hashes
    over the four widths for every consumer, grouping {1, 2} / {4} / {8} --
    which is the OpenBLAS GEMV's K-split, seen end to end.

    Compared against EACH OTHER, never against a stored hash: see this
    module's cross-build non-claim.
    """
    seen = {}
    for w in _WIDTHS:
        for i, h in enumerate(_run_consumer(consumer, True, w)):
            seen.setdefault(h, []).append(f'width={w} rep={i}')
    assert len(seen) == 1, (
        f"{consumer}: carrier='auto' is not thread-invariant -- "
        f"{len(seen)} distinct hashes over "
        f"{len(_WIDTHS) * _REPS} identical calls: "
        + '; '.join(f'{h}: {arms}' for h, arms in seen.items()))


def test_the_traced_chain_has_a_second_reduction_this_niche_does_not_own():
    """SCOPE, PINNED -- RESTATED 2026-08-23 BY NICHE D15, WHICH CLOSED IT.

    ORIGINALLY (D14) this test read: exactly ONE solve on the traced path is
    deterministic and it is the widest one.  That was the honest scope while
    the carrier fit was the only fit on the deterministic kernel, and it
    carried a REFUTATION of the tempting end-to-end claim -- "the traced field
    is the same at any thread count" was written, measured failing, and
    withdrawn, because the chain runs SIX solves and D14 owned one:

        (119936,   5)  deterministic   <- niche D14
        (  1457,  28)  BLAS   x3       invariant at this size, measured
        (  1337, 120)  BLAS   x2       MOVES with the width (measured: one
                                       value at OMP=1, another at 4 and 8)

    D15 put the other five on the deterministic route (see
    :data:`~lumenairy.elements._lens_traced.DETERMINISTIC_TRACED_FIT`), so
    "exactly one" is no longer the truth and asserting it would now be a test
    pinning a state the library has left.  TWO OF D14'S SENTENCES ABOUT THOSE
    FITS WERE ALSO WRONG AND THE RESTATEMENT RECORDS BOTH: they are the
    inverse-map exit fits in ``_lens_imap.build_inverse_map``, not
    residual-eikonal fits, and the 34x that priced them applied to a kernel
    that would not have fixed them anyway -- every one of them screens
    singular under C13 and takes a threaded QR, which no Gram kernel touches.

    WHAT THIS TEST NOW OWNS, and why it stays here rather than moving: the
    boundary between the two flags.  D14's flag must still gate the carrier
    fit and ONLY the carrier fit, so that ``DETERMINISTIC_NORMAL_EQUATIONS =
    False`` remains a clean fail-before for 5.41.0's bits independently of
    anything D15 does.  The end-to-end field claim lives in
    ``tests/unit/test_niche_d15_deterministic_traced_fit.py``.
    """
    x = (np.arange(256) - 128) * 30e-6
    X, Y = np.meshgrid(x, x)
    w = 40 * 30e-6
    E = (np.exp(-(X * X + Y * Y) / (w * w))
         * np.exp(1j * (2 * np.pi / _LAM) * (X * X + Y * Y) / 2.0)
         ).astype(np.complex128)
    presc = {'wavelength': _LAM, 'aperture_diameter': 14e-3, 'surfaces': [
        {'radius': 51.68e-3, 'thickness': 4e-3, 'glass_before': 'air',
         'glass_after': 'N-BK7', 'semi_diameter': 7e-3},
        {'radius': -51.68e-3, 'thickness': 0.0, 'glass_before': 'N-BK7',
         'glass_after': 'air', 'semi_diameter': 7e-3}],
        'thicknesses': [4e-3], 'stop_index': 0}

    seen = []
    orig = LT._solve_lstsq_thread_safe

    def _spy(A, b, deterministic=False):
        seen.append((np.shape(A), bool(deterministic)))
        return orig(A, b, deterministic=deterministic)

    # D15's flag OFF, so what is measured is D14's own scope and nothing
    # else.  The two flags are independent by construction and this is the
    # assertion that keeps them so.
    _d15 = LT.DETERMINISTIC_TRACED_FIT
    LT.DETERMINISTIC_TRACED_FIT = False
    LT._solve_lstsq_thread_safe = _spy
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            la.apply_real_lens_traced(E, prescription=presc, wavelength=_LAM,
                                      dx=30e-6, carrier='auto',
                                      on_undersample='silent')
    finally:
        LT._solve_lstsq_thread_safe = orig
        LT.DETERMINISTIC_TRACED_FIT = _d15

    det = [s for s, flag in seen if flag]
    blas = [s for s, flag in seen if not flag]
    assert len(det) == 1, (
        f'with DETERMINISTIC_TRACED_FIT off, expected exactly one '
        f'deterministic solve (the carrier fit); got {len(det)}: {seen}')
    assert blas, 'the traced chain no longer runs any BLAS-route fit'
    assert det[0][0] > max(s[0] for s in blas), (
        f'the carrier fit {det[0]} is no longer the widest reduction on this '
        f'path; the scope argument (it is the long one) needs re-deriving '
        f'against {blas}')

    # ...and with D15 on, NONE of them is left on the BLAS route.  That half
    # is what makes the end-to-end field claim available, and it is asserted
    # in full in the D15 file; here it is the other side of the boundary.
    seen2 = []

    def _spy2(A, b, deterministic=False):
        seen2.append((np.shape(A), bool(deterministic)))
        return orig(A, b, deterministic=deterministic)

    LT.DETERMINISTIC_TRACED_FIT = True
    LT._solve_lstsq_thread_safe = _spy2
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            la.apply_real_lens_traced(E, prescription=presc, wavelength=_LAM,
                                      dx=30e-6, carrier='auto',
                                      on_undersample='silent')
    finally:
        LT._solve_lstsq_thread_safe = orig
        LT.DETERMINISTIC_TRACED_FIT = _d15
    assert seen2 and not [s for s, f in seen2 if not f], (
        f'DETERMINISTIC_TRACED_FIT no longer covers the whole chain: {seen2}')


@pytest.mark.slow
def test_the_shipped_route_is_the_fail_before_wherever_this_build_splits():
    """The matched pair, run at the reduction level so the ladder is cheap.

    A fail-before that DEPENDS on a library's threading threshold cannot be
    asserted unconditionally -- a build whose BLAS never splits at these sizes
    is legitimate, and hard-failing on it would be exactly the
    environment-dependent precondition ``TESTING_STANDARDS`` S3 bans.  So this
    scans a LADDER of row counts and reports the pair: at every rung the
    deterministic kernel must be width-invariant (unconditional, asserted on
    every rung), and at the rungs where the shipped ``A.T @ A`` / ``A.T @ b``
    route is NOT, the pair is the fix.

    MEASURED 2026-08-22 on the reference box (Windows 11, py3.14.6, numpy
    2.4.4 / scipy-openblas 0.3.31 DYNAMIC_ARCH Haswell, Ryzen 9 5950X): the
    GEMV splits from 100 000 rows up, giving three distinct ``A.T @ b`` values
    over widths {1, 2}, {4}, {8}; the ``A.T @ A`` GEMM at 5 terms does NOT
    split at any of these sizes.  That last point is a REFUTATION of
    BUILD_LENS_32K_MEMORY S4.2's attribution to ``G`` and is why both products
    go through the deterministic kernel.
    """
    ladder = (20_000, 100_000, 1_000_000, 4_000_000)
    code = _PREAMBLE + textwrap.dedent(
        """
        rng = np.random.default_rng(20260822)
        for n in ({ladder}):
            x = rng.uniform(-1.5e-2, 1.5e-2, n)
            y = rng.uniform(-1.5e-2, 1.5e-2, n)
            A = np.ascontiguousarray(
                np.stack([np.ones(n), x, y, x * x, x * y], axis=1))
            A *= rng.uniform(0.0, 1.0, n)[:, None]
            b = A[:, 1] * 3.0 + rng.normal(0, 1e-9, n)
            Gd, rd = LT._det_normal_equations(A, b)
            print('ROW %d blas %s %s det %s %s'
                  % (n, _h(A.T @ A), _h(A.T @ b), _h(Gd), _h(rd)))
        """
    ).format(ladder=', '.join(str(v) for v in ladder))
    rows = {}
    for w in _WIDTHS:
        proc = subprocess.run([sys.executable, '-c', code, '0'],
                              env=_child_env(w), capture_output=True,
                              text=True, timeout=900)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        for ln in proc.stdout.splitlines():
            if not ln.startswith('ROW '):
                continue
            _, n, _, hG, hr, _, hGd, hrd = ln.split()
            rows.setdefault(int(n), []).append((hG, hr, hGd, hrd))

    split = []
    for n, arms in sorted(rows.items()):
        det = {(a[2], a[3]) for a in arms}
        assert len(det) == 1, (
            f'n={n}: the DETERMINISTIC normal equations moved with the BLAS '
            f'width -- {len(det)} distinct (G, rhs) pairs over '
            f'{len(_WIDTHS)} widths: {sorted(det)}')
        if len({(a[0], a[1]) for a in arms}) > 1:
            split.append(n)

    # Both branches assert something real.  Where the ladder found the split,
    # the pair above IS the fail-before.  Where it did not, the unconditional
    # half still ran on every rung -- and this build is simply one whose BLAS
    # keeps the reduction serial at these sizes, which is not a defect.
    assert rows, 'the ladder produced no rungs at all'
    if not split:
        pytest.xfail(  # pragma: no cover -- reference box splits from 1e5 up
            'this BLAS did not split the reduction at any rung of '
            f'{ladder}; the deterministic invariance was still asserted on '
            'every rung')


# ===========================================================================
# 2.  A correctly-rounded oracle, and what both routes measure against it.
# ===========================================================================

_SPLIT = float(2 ** 27 + 1)


def _two_product(a, b):
    """Exact ``a * b = hi + lo`` for float64 (Dekker; no FMA required)."""
    hi = a * b
    ca, cb = _SPLIT * a, _SPLIT * b
    ah, bh = ca - (ca - a), cb - (cb - b)
    al, bl = a - ah, b - bh
    return hi, (((ah * bh - hi) + ah * bl) + al * bh) + al * bl


def _exact_dot(a, b):
    """The CORRECT ROUNDING of the exact ``sum_i a_i b_i``.

    ``_two_product`` splits every product into an exactly-representable
    ``hi + lo`` pair, so the concatenated list sums to the exact mathematical
    value; ``math.fsum`` then returns its correctly-rounded float64.  This is
    an INDEPENDENT oracle in the sense S2 of ``TESTING_STANDARDS`` asks for --
    not another float64 summation order, and not a compensated accumulator
    whose own error would have to be bounded.
    """
    hi, lo = _two_product(np.asarray(a, np.float64), np.asarray(b, np.float64))
    return math.fsum(np.concatenate([hi.ravel(), lo.ravel()]).tolist())


def _oracle_normal_equations(A, b):
    A = np.asarray(A, np.float64)
    b = np.asarray(b, np.float64)
    m = A.shape[1]
    G = np.empty((m, m))
    r = np.empty(m)
    for p in range(m):
        for q in range(p, m):
            G[p, q] = G[q, p] = _exact_dot(A[:, p], A[:, q])
        r[p] = _exact_dot(A[:, p], b)
    return G, r


def test_the_oracle_is_an_oracle():
    """PROVE THE INSTRUMENT BEFORE USING IT.  Against exact rational
    arithmetic over the same float64 inputs -- ``Fraction`` products summed
    without rounding, then rounded once -- the two-product + ``fsum`` oracle
    must be BIT-IDENTICAL, not merely close.  A reference that is only
    'more accurate' cannot bound the thing it is refereeing.
    """
    from fractions import Fraction

    rng = np.random.default_rng(4242)
    n, m = 4000, 4
    A = np.ascontiguousarray(rng.normal(size=(n, m)) * 1e-2)
    b = np.ascontiguousarray(rng.normal(size=n))
    Go, ro = _oracle_normal_equations(A, b)
    cols = [[Fraction(v) for v in A[:, k]] for k in range(m)]
    bf = [Fraction(v) for v in b]
    for p in range(m):
        for q in range(p, m):
            want = float(sum(u * v for u, v in zip(cols[p], cols[q])))
            assert Go[p, q] == want, (p, q, Go[p, q], want)
        want = float(sum(u * v for u, v in zip(cols[p], bf)))
        assert ro[p] == want, (p, ro[p], want)


def _real_carrier_design_matrix():
    """The fit's OWN ``(A, b)``, captured from a real ``'auto'`` call rather
    than modelled -- the column scales (``1``, ``x`` ~ 1e-2, ``x^2`` ~ 1e-4)
    are what makes the reduction interesting and a synthetic stand-in would be
    a different matrix."""
    N, dx = 384, 30e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    w = 60 * dx
    E = (np.exp(-(X * X + Y * Y) / (w * w))
         * np.exp(1j * (2 * np.pi / _LAM) * (X * X + Y * Y) / 2.0)
         ).astype(np.complex128)
    box = {}
    orig = LT._solve_lstsq_thread_safe

    def _spy(A, b, deterministic=False):
        box.setdefault('A', np.ascontiguousarray(A, np.float64))
        box.setdefault('b', np.asarray(b, np.float64))
        return orig(A, b, deterministic=deterministic)

    LT._solve_lstsq_thread_safe = _spy
    try:
        LT._compute_carrier('auto', E, _LAM, dx, X, Y, need_W=False)
    finally:
        LT._solve_lstsq_thread_safe = orig
    return box['A'], box['b']


#: The legal-partition ladder.  A threaded BLAS reducing ``A^T A`` / ``A^T b``
#: splits the K dimension into contiguous chunks and adds the chunk results;
#: which ``k`` it picks is the scheduling fact this whole niche is about.
#: Reproducing the FAMILY in one process is what makes the envelope below
#: build-free: it is constructed through legal reorderings of the same exact
#: sum (``TESTING_STANDARDS`` S3, engineer the state) rather than waiting for
#: a particular BLAS to produce one.
_KSPLITS = (1, 2, 3, 4, 5, 8, 12, 16, 24, 32)


def _partitioned_normal_equations(A, b, k):
    n = A.shape[0]
    bnd = [round(i * n / k) for i in range(k + 1)]
    G = r = None
    for i in range(k):
        s = slice(bnd[i], bnd[i + 1])
        g, rr = A[s].T @ A[s], A[s].T @ b[s]
        G = g if G is None else G + g
        r = rr if r is None else r + rr
    return G, r


def _oracle_error_table(A, b):
    """``(det_errors, {k: errors})`` -- oracle-relative errors of ``G``, the
    right-hand side and the recovered coefficients, for the deterministic
    route and for every legal K-split."""
    Go, ro = _oracle_normal_equations(A, b)
    xo = np.linalg.solve(Go, ro)

    def _rel(x, ref):
        return float(np.max(np.abs(np.asarray(x) - ref)) / np.max(np.abs(ref)))

    def _errs(G, r):
        return (_rel(G, Go), _rel(r, ro), _rel(np.linalg.solve(G, r), xo))

    det = _errs(*LT._det_normal_equations(A, b))
    fam = {k: _errs(*_partitioned_normal_equations(A, b, k)) for k in _KSPLITS}
    return det, fam


def test_the_deterministic_route_is_at_least_as_close_to_the_truth():
    """ORACLE-ACCURACY NON-REGRESSION, on the fit's own design matrix.

    A fixed pairwise tree has BETTER rounding growth than a partitioned
    reduction (``log2`` of the block count against linear in the chunk
    length), so replacing nondeterminism with determinism must not cost
    accuracy -- and here it buys some.  The bar is the LEGAL-PARTITION FAMILY
    the shipped route draws from, measured in this run against a
    correctly-rounded oracle, so it carries no per-build constant.

    TWO CLAIMS, EACH WITH ITS OWN MEASURED GAP.

    * ``G`` and the right-hand side: at least as close as the WORST draw.
      A ``<= min`` here would be testing noise -- measured, the deterministic
      right-hand side is 1.4x WORSE than the luckiest draw on one fixture
      (2.57e-14 against 1.80e-14) and 6x better on the other.  Against the
      worst draw the margins are 3x / 14000x (``G``) and 25x / 890x (rhs).
    * the COEFFICIENTS -- the only thing the fit returns -- beat even the
      BEST draw, by 84x and 162x on the two fixtures (3.95e-18 against
      3.33e-16; 2.74e-18 against 4.44e-16), against a family spanning
      3.3e-16 .. 9.9e-15.  Two decades of gap below and above.

    All values measured 2026-08-22 (Windows 11, py3.14.6, numpy 2.4.4 /
    scipy-openblas 0.3.31, Ryzen 9 5950X) at A = 67 348 x 5 and 119 936 x 5.
    """
    A, b = _real_carrier_design_matrix()
    assert A.shape[0] > 20_000, f'fixture collapsed to {A.shape} rows'
    det, fam = _oracle_error_table(A, b)
    for i, name in enumerate(('G', 'rhs')):
        worst = max(v[i] for v in fam.values())
        assert det[i] <= worst, (
            f'{name}: deterministic {det[i]:.3e} is worse than every legal '
            f'partitioned draw (worst {worst:.3e})')
    best_coef = min(v[2] for v in fam.values())
    assert det[2] <= best_coef, (
        f'coefficients: deterministic {det[2]:.3e} against the best legal '
        f'partitioned draw {best_coef:.3e} -- the pairwise tree lost its '
        f'rounding-growth advantage')


def test_the_two_routes_agree_inside_the_derived_summation_envelope():
    """DETERMINISTIC-VS-SHIPPED AGREEMENT, bounded by a bar DERIVED in this
    run from the legal-partition family's own spread.

    The two are different summation orders of the same exact sum, so the
    defensible bar on their difference is the spread the shipped route itself
    exhibits over the orders a threaded BLAS may legally choose.  A fixed
    tolerance would pin a per-build reduction error; a bit-equality assertion
    would be FALSE by design, since the whole point is that the shipped order
    is not reproducible.

    Gap on the other side: the smallest coefficient difference that means
    anything here moves the recovered carrier radius by a part in 1e6, i.e.
    ~1e-6 relative -- nine decades above the bar this derives (measured
    3.9e-14 on 2026-08-22, against a gap of 3.4e-15).
    """
    A, b = _real_carrier_design_matrix()
    Go, ro = _oracle_normal_equations(A, b)
    xo = np.linalg.solve(Go, ro)
    scale = float(np.max(np.abs(xo)))
    det, fam = _oracle_error_table(A, b)

    x_det = LT._solve_lstsq_thread_safe(A, b, deterministic=True)
    x_nai = LT._solve_lstsq_thread_safe(A, b, deterministic=False)
    gap = float(np.max(np.abs(x_det - x_nai))) / scale
    # DERIVED: both answers are roundings of the same exact quantity, so they
    # cannot be further apart than the sum of their distances to it.  x4 for
    # the shared ``np.linalg.solve`` that follows the accumulation.
    bar = 4.0 * (det[2] + max(v[2] for v in fam.values()))
    assert gap <= bar, (
        f'the two routes differ by {gap:.3e}, past the derived envelope '
        f'{bar:.3e} -- that is not a summation-order difference')


@pytest.mark.slow
@pytest.mark.parametrize('consumer', ['fit', 'analytic', 'traced'])
def test_the_field_moves_only_at_the_summation_noise(consumer, tmp_path):
    """The same statement one level up: turning determinism ON changes the
    RETURNED FIELD only within the shipped route's own run-to-run spread.

    Both bars are measured in THIS run.  The shipped route's cross-width
    spread is the summation-order envelope, by definition -- it is the same
    call disagreeing with itself -- so a deterministic answer sitting inside
    it is an answer that could have come out of the shipped route on some
    thread count.  The absolute floor covers a build whose BLAS keeps the
    reduction serial (spread exactly 0), where the meaningful statement is
    that the two routes agree to far better than anything physical.

    Measured 2026-08-22 on the reference box, relative to the field peak:
    shipped cross-width spread 3.2e-15 (fit) / 2.8e-14 (analytic) / 1.6e-12
    (traced); ``|det - shipped|`` 3.2e-15 / 2.8e-14 / 1.6e-12 -- i.e. the
    deterministic answer lands INSIDE the spread in all three.  The floor sits
    ~3 decades above the largest of those and ~6 below any field difference
    that would change a physical readout.
    """
    naive, det = [], []
    for w in _WIDTHS:
        for tag, flag, bucket in (('n', False, naive), ('d', True, det)):
            path = tmp_path / f'{consumer}_{tag}_{w}.npy'
            _run_consumer(consumer, flag, w, reps=1, save=path)
            bucket.append(np.load(path))

    scale = float(np.max(np.abs(naive[0])))
    spread = max(float(np.max(np.abs(a - c)))
                 for a in naive for c in naive) / scale
    d_spread = max(float(np.max(np.abs(a - c)))
                   for a in det for c in det) / scale
    gap = max(float(np.max(np.abs(a - det[0]))) for a in naive) / scale

    assert d_spread == 0.0, (
        f'{consumer}: the deterministic arm disagreed with itself across '
        f'widths by {d_spread:.3e}')
    _FLOOR = 1e-9
    bar = max(10.0 * spread, _FLOOR)
    assert gap <= bar, (
        f'{consumer}: determinism moved the field by {gap:.3e} relative, '
        f'past the derived envelope {bar:.3e} (shipped cross-width spread '
        f'{spread:.3e})')


# ===========================================================================
# 3.  The kernel's own contracts.
# ===========================================================================

def test_the_partition_reads_the_term_count_and_nothing_else():
    """The block length is a function of ``n_terms`` ALONE.

    This is the whole determinism argument in one assertion.  A partition that
    consulted the row count, the free memory, the thread count or a cache
    probe would be a scheduling input, and the answer would move with it --
    which is the defect, re-introduced one level down.  Option (b) of the
    design brief (blocks sized so the partial falls below the BLAS threading
    threshold) is REFUSED for the same reason: that threshold is a build fact.
    """
    for n_terms in (1, 3, 5, 28, 66, 120):
        want = LT._det_block_rows(n_terms)
        assert want == LT._det_block_rows(n_terms), 'not a pure function'
        assert want >= LT._DET_GRAM_MIN_BLOCK_ROWS
        rng = np.random.default_rng(11 + n_terms)
        for n in (want // 3, want, 3 * want + 7):
            A = np.ascontiguousarray(rng.normal(size=(n, n_terms)))
            b = np.ascontiguousarray(rng.normal(size=n))
            g1, r1 = LT._det_normal_equations(A, b)
            g2, r2 = LT._det_normal_equations(A, b)
            assert np.array_equal(g1, g2) and np.array_equal(r1, r2)


def test_the_transposed_tile_is_a_speed_device_and_not_an_arithmetic_one():
    """The kernel copies each block into a contiguous ``(n_terms, rows)`` tile
    before taking the pair products.  That copy is there ONLY so the products
    read contiguous memory: at the SHIPPED block size it is 1.11x faster at 5
    terms and 1.89x at 66 (audit S2.3).  Run the identical scheme on the STRIDED columns
    of ``A`` and the bits must be the same, or the copy is silently part of
    the answer and the constant that sizes it is load-bearing in a second,
    undocumented way.
    """
    rng = np.random.default_rng(909)
    for n_terms, n in ((5, 40_000), (3, 9_000)):
        A = np.ascontiguousarray(rng.normal(size=(n, n_terms)) * 1e-2)
        b = np.ascontiguousarray(rng.normal(size=n))
        blk = LT._det_block_rows(n_terms)
        parts_g, parts_r = [], []
        for i0 in range(0, n, blk):
            S, bb = A[i0:i0 + blk], b[i0:i0 + blk]
            g = np.empty((n_terms, n_terms))
            r = np.empty(n_terms)
            for p in range(n_terms):
                cp = S[:, p]                       # STRIDED, no tile
                for q in range(p, n_terms):
                    g[p, q] = g[q, p] = float(np.sum(cp * S[:, q]))
                r[p] = float(np.sum(cp * bb))
            parts_g.append(g)
            parts_r.append(r)
        # the same fixed pairwise tree, written level by level
        for parts in (parts_g, parts_r):
            while len(parts) > 1:
                nxt = [parts[i] + parts[i + 1]
                       for i in range(0, len(parts) - 1, 2)]
                if len(parts) % 2:
                    nxt.append(parts[-1])
                parts[:] = nxt
        gd, rd = LT._det_normal_equations(A, b)
        assert np.array_equal(gd, parts_g[0]), n_terms
        assert np.array_equal(rd, parts_r[0]), n_terms


def test_the_flag_off_returns_the_shipped_bits_exactly():
    """``DETERMINISTIC_NORMAL_EQUATIONS = False`` -- and the ``deterministic``
    kwarg's default -- must restore ``G = A.T @ A`` / ``rhs = A.T @ b`` bit for
    bit.  That is the fail-before the layer map records, and it is also what
    keeps every OTHER caller of ``_solve_lstsq_thread_safe`` (the traced
    Chebyshev, coordinate and OPL fits, whose byte-identity contracts C1/C6/
    C8/C9 predate this) on its historical arithmetic: the deterministic kernel
    costs 34x at their 66-term shape (0.0106 s -> 0.3623 s at 141 471 rows,
    measured 2026-08-22), so it is scoped to the carrier fit alone.
    """
    rng = np.random.default_rng(5150)
    A = np.ascontiguousarray(rng.normal(size=(30_000, 6)) * 1e-2)
    b = np.ascontiguousarray(rng.normal(size=30_000))
    G, rhs = A.T @ A, A.T @ b
    want = LT._solve_lstsq_thread_safe(A, b)
    import scipy.linalg as sla
    ref = sla.cho_solve(sla.cho_factor(G, check_finite=False), rhs,
                        check_finite=False)
    assert np.array_equal(want, ref)
    assert np.array_equal(
        want, LT._solve_lstsq_thread_safe(A, b, deterministic=False))


def test_multi_rhs_matches_the_single_rhs_column_by_column():
    """``_solve_lstsq_thread_safe`` takes a 2-D ``b`` (the coordinate fits do),
    so the deterministic accumulator has to as well -- and column ``k`` of the
    multi-RHS answer must be the single-RHS answer for that column, bit for
    bit, or the two shapes are two different reductions."""
    rng = np.random.default_rng(2718)
    A = np.ascontiguousarray(rng.normal(size=(20_000, 4)) * 1e-2)
    B = np.ascontiguousarray(rng.normal(size=(20_000, 3)))
    G2, r2 = LT._det_normal_equations(A, B)
    assert np.array_equal(G2, G2.T), 'the Gram is not symmetric'
    assert r2.shape == (4, 3)
    for k in range(B.shape[1]):
        _, r1 = LT._det_normal_equations(A, np.ascontiguousarray(B[:, k]))
        assert r1.shape == (4,)
        assert np.array_equal(r1, r2[:, k]), k


def test_the_step_down_hole_is_declared_rather_than_hidden(recwarn):
    """THE ONE PLACE THE GUARANTEE DOES NOT REACH, asserted so it cannot be
    forgotten.  If the C13 screen finds the Gram numerically singular the
    solve reroutes to :func:`_solve_lstsq_qr`, whose ``geqrf`` over the full
    ``A`` is a threaded BLAS-3 factorisation -- so on that route the answer is
    NOT scheduling-independent.  A deterministic caller must be told, not
    quietly handed a weaker guarantee.

    RESTATED 2026-08-23 (niche D15), BECAUSE THE HOLE SHRANK.  D14 covered
    BOTH exits here: an outright rank-deficient Gram (Cholesky and LU both
    refuse) and one that factorises but screens singular under C13.  It read
    the second as a corner.  IT IS NOT: measured on the traced chain, EVERY
    non-carrier fit screens singular, so that exit is the default there and a
    guarantee that lapses on it is no guarantee at all.  D15 closed it -- a
    screened-singular Gram is now refined deterministically instead of
    rerouted -- so this test's C13 arm is INVERTED rather than deleted: the
    deterministic caller must now NOT be warned, and the same engineered
    fixture proves it.  The rank-deficient arm is unchanged, because there is
    no factorisation to refine through and the QR is still what happens.

    Each state is ENGINEERED rather than waited for: an exactly duplicated
    column makes the Gram rank-deficient outright, and a column perturbed by
    1e-6 relative makes one that factorises (Cholesky succeeds, the Gram
    stays positive-definite) but screens singular under C13 -- rcond 2.5e-13
    measured, five decades under the 1e-8 screen.  A merely RESCALED column
    would not do: ``_gram_rcond`` equilibrates the diagonal first, so scale is
    exactly what it is built to ignore.
    """
    import warnings

    rng = np.random.default_rng(1234)
    n = 12_000
    col = rng.normal(size=n)
    dup = np.ascontiguousarray(
        np.stack([np.ones(n), col, col, rng.normal(size=n)], axis=1))
    tiny = np.ascontiguousarray(
        np.stack([np.ones(n), col, col + 1e-6 * rng.normal(size=n),
                  rng.normal(size=n)], axis=1))
    b = np.ascontiguousarray(rng.normal(size=n))

    for tag, A, want_warn in (('rank-deficient', dup, True),
                              ('screened singular', tiny, False)):
        rcond = LT._gram_rcond(LT._det_normal_equations(A, b)[0])
        assert rcond < LT._LSTSQ_GRAM_RCOND_MIN, (
            f'{tag}: the fixture is not singular enough (rcond {rcond:.3e})')
        with warnings.catch_warnings(record=True) as got:
            warnings.simplefilter('always')
            LT._solve_lstsq_thread_safe(A, b, deterministic=True)
        msgs = [str(w.message) for w in got if w.category is RuntimeWarning]
        lapsed = [m for m in msgs if 'does NOT hold' in m]
        if want_warn:
            assert lapsed, (tag, msgs)
        else:
            assert not lapsed, (
                f'{tag}: D15 refines this exit deterministically, so the '
                f'guarantee does not lapse and must not be declared to: '
                f'{msgs}')

        # and the SHIPPED route stays silent on BOTH -- the warning is about
        # the promise this caller was given, not about the data.
        with warnings.catch_warnings(record=True) as got:
            warnings.simplefilter('always')
            LT._solve_lstsq_thread_safe(A, b, deterministic=False)
        assert not [w for w in got if w.category is RuntimeWarning], (
            tag, [str(w.message) for w in got])


# ===========================================================================
# 4.  Footprint: the 32k wave's banded fit profile is not regressed.
# ===========================================================================

def test_the_fit_footprint_is_not_regressed():
    """FOOTPRINT NON-REGRESSION against the 32k wave (BUILD_LENS_32K_MEMORY
    S2.3 / S3.2), measured with that audit's own instrument (``tracemalloc``
    peak, quoted in float64 GRIDS) and its own A/B shape.

    The deterministic kernel's live set is bounded BY CONSTRUCTION and the
    bound is computable from the two constants: one
    :data:`~lumenairy.elements._lens_traced._DET_GRAM_TILE_BYTES` tile, one
    row buffer, and -- because the partials are combined by a carry-stack
    rather than collected in a list -- ``O(log2(n / blk))`` partial Grams
    rather than ``O(n / blk)`` of them.  At the production shape that is ~9
    live partials, not 2 137.

    Measured 2026-08-22, N = 4096, the fit alone with ``need_W=False``:
    5% bright 10.125 grids OFF and ON; 21% 10.125 / 10.125; 59% 17.530 /
    17.530; 89% 24.800 / 24.800 -- delta 0.0000 grids to four decimals in
    every row.
    """
    N, dx = 1024, 30e-6
    grid = N * N * 8.0
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    w = 260 * dx
    E = (np.exp(-(X * X + Y * Y) / (w * w))
         * np.exp(1j * (2 * np.pi / _LAM) * (X * X + Y * Y) / 2.0)
         ).astype(np.complex128)
    del X, Y
    Xb = np.broadcast_to(x[None, :], (N, N))
    Yb = np.broadcast_to(x[:, None], (N, N))

    def _peak(flag):
        LT.DETERMINISTIC_NORMAL_EQUATIONS = flag
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            LT._compute_carrier('auto', E, _LAM, dx, Xb, Yb, need_W=False)
            return tracemalloc.get_traced_memory()[1] / grid
        finally:
            tracemalloc.stop()

    was = LT.DETERMINISTIC_NORMAL_EQUATIONS
    try:
        off = _peak(False)
        on = _peak(True)
    finally:
        LT.DETERMINISTIC_NORMAL_EQUATIONS = was

    # DERIVED allowance: the tile, the row buffer, the transposed RHS copy and
    # the carry stack, in grids -- everything the kernel can hold at once.
    blk = LT._det_block_rows(5)
    n_terms = 5
    live = (3 * blk * 8.0 + math.log2(max(N * N / blk, 2.0))
            * (n_terms * n_terms + n_terms) * 8.0)
    allowance = 4.0 * live / grid
    assert on <= off + allowance, (
        f'fit peak {on:.4f} grids with determinism on against {off:.4f} off; '
        f'the derived allowance is {allowance:.6f} grids')
