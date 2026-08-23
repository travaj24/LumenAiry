"""Niche D15 (2026-08-23) -- the TRACED EXIT FIELD is deterministic by
construction at any BLAS thread count.

WHAT THIS FILE PINS, and what D14 could not.
``docs/audits/BUILD_DETERMINISTIC_CARRIER_FIT_2026_08_22.md`` S7.3 recorded a
claim that was written, measured failing, and withdrawn: that
``apply_real_lens_traced(carrier='auto')`` returns the same exit field at
every thread count.  It did not, because the carrier fit was never that
path's only nondeterministic reduction -- only its longest.  D14 pinned the
scope instead and left the rest open.  This layer closes it, and the closure
needed TWO things, neither of which was a transplant of D14's kernel:

  1. A per-block partial that is affordable at 28 and 120 terms.  D14's is
     ``M(M+1)/2`` NumPy calls per block and costs 34-43x there.  A single
     ``np.einsum(..., optimize=False)`` per block is 7.7x cheaper AND
     BLAS-free -- but only at a 64-row block, because einsum accumulates
     SEQUENTIALLY where ``np.sum`` is pairwise and the naive transplant lands
     WORSE than the threaded route's worst legal partition (S5.1 of the
     audit; pinned by ``test_the_einsum_partial_is_at_least_as_close_...``).
  2. A deterministic replacement for the C13 QR step-down.  D14 called that
     step-down "the one hole" and warned on it.  MEASURED, IT IS THE DEFAULT
     ON THIS PATH: every non-carrier traced fit screens numerically singular
     (Gram rcond 1.6e-9 at 28 terms, 9.6e-11 at 120, against the 1e-8 screen)
     and takes a threaded ``dgeqrf``.  A deterministic Gram alone therefore
     changes nothing there, which is the second refutation this file carries.

=============================================================================
THE CROSS-BUILD NON-CLAIM.  THIS IS A COMMENT, NOT AN ASSERTION, ON PURPOSE.
=============================================================================
Nothing here claims -- and no test in this repository may assert -- that the
traced field produces the SAME BITS on two different builds or platforms.  It
does not, and it is not supposed to.  ``np.sum``'s pairwise block size and
einsum's ``sum_of_products`` unrolling are properties of the NumPy build and
the CPU it dispatched for; the ray trace upstream goes through the platform
libm.  A test whose pass/fail boundary sits inside that spread is per-build by
the definition in ``docs/TESTING_STANDARDS.md``.  The property pinned is:

    ONE build, ANY thread count, ANY number of repeats -> the same bytes.

Every thread-invariance assertion below therefore compares hashes produced by
subprocesses of THIS interpreter against each other, never against a stored
constant.
"""
from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
import textwrap
import tracemalloc
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_traced as LT

_LAM = 1.31e-6
_WIDTHS = ('1', '2', '4', '8')
_REPS = 2

#: Repo root from the LIVE import, not from ``PYTHONPATH``: BUILD_LENS_32K_
#: MEMORY S9 records that a runner inserting a sibling at ``sys.path[0]``
#: beats ``PYTHONPATH`` outright, so a child launched with ``PYTHONPATH``
#: alone can import a DIFFERENT library while looking correctly pinned.
_ROOT = os.path.realpath(os.path.join(os.path.dirname(la.__file__), os.pardir))

#: D14's oracle machinery, IMPORTED rather than re-derived.  A second copy of
#: an instrument is a second instrument, and the whole point of
#: ``test_the_oracle_is_an_oracle`` over there is that this one is proved.
_d14 = None


def _oracle():
    global _d14
    if _d14 is None:
        path = os.path.join(_ROOT, 'tests', 'unit',
                            'test_niche_d14_deterministic_carrier_fit.py')
        spec = importlib.util.spec_from_file_location('_d14_oracle', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _d14 = mod
    return _d14


def _child_env(width):
    """Environment for one arm: the BLAS width pinned in ALL FOUR variables
    (OpenBLAS reads its own; the OpenMP one is what a threadpool-capped parent
    would have moved), plus the library root the child asserts against."""
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
    LT.DETERMINISTIC_TRACED_FIT = bool(int(sys.argv[1]))

    def _h(a):
        return hashlib.sha256(
            np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]
    """
)

#: THE FIXTURE, and the one property that is load-bearing about it.
#:
#: ``ray_subsample=2`` is not decoration.  Measured over eight traced fixtures
#: (audit S2.2), the two ``build_inverse_map`` exit solves read different
#: coefficient bytes at widths {1,2} against {4,8} on ALL of them -- but on
#: seven the difference is swallowed downstream and the returned FIELD is
#: width-invariant anyway.  On this one it reaches the field: FOUR distinct
#: field hashes over the four widths on the shipped route.  A fixture whose
#: fail-before does not fire reads exactly like a fix, which is the trap D14
#: S1.2 recorded from the other side.
_FIXTURE = textwrap.dedent(
    """
    LAM, N, DX, SUB = 1.31e-6, 512, 30e-6, 2
    _x = (np.arange(N) - N / 2) * DX
    _X, _Y = np.meshgrid(_x, _x)
    _w = 80 * DX
    E0 = (np.exp(-(_X * _X + _Y * _Y) / (_w * _w))
          * np.exp(1j * (2 * np.pi / LAM) * (_X * _X + _Y * _Y) / 2.0)
          ).astype(np.complex128)
    PRESC = {'wavelength': LAM, 'aperture_diameter': 14e-3, 'surfaces': [
        {'radius': 51.68e-3, 'thickness': 4e-3, 'glass_before': 'air',
         'glass_after': 'N-BK7', 'semi_diameter': 7e-3},
        {'radius': -51.68e-3, 'thickness': 0.0, 'glass_before': 'N-BK7',
         'glass_after': 'air', 'semi_diameter': 7e-3}],
        'thicknesses': [4e-3], 'stop_index': 0}

    def CALL():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.asarray(la.apply_real_lens_traced(
                E0, prescription=PRESC, wavelength=LAM, dx=DX,
                carrier='auto', on_undersample='silent', ray_subsample=SUB))
    """
)

_BODY_FIELD = """
OUT = CALL()
"""


def _run_field(deterministic, width, reps=_REPS, save=None):
    """Hash the traced EXIT FIELD ``reps`` times in ONE fresh interpreter at
    the given BLAS width.  Returns the list of hashes."""
    driver = textwrap.dedent(
        """
        _save = os.environ.get('D15_SAVE')
        for _r in range({reps}):
            __BODY__
            if _save and _r == 0:
                np.save(_save, np.asarray(OUT))
            print('HASH ' + _h(np.nan_to_num(OUT)))
        """
    ).format(reps=int(reps))
    hit = next(ln for ln in driver.splitlines() if '__BODY__' in ln)
    pad = ' ' * (len(hit) - len(hit.lstrip()))
    body = textwrap.indent(textwrap.dedent(_BODY_FIELD).strip(), pad).lstrip()
    code = _PREAMBLE + _FIXTURE + driver.replace('__BODY__', body)
    env = _child_env(width)
    if save is not None:
        env['D15_SAVE'] = str(save)
    proc = subprocess.run(
        [sys.executable, '-c', code, str(int(bool(deterministic)))],
        env=env, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, (
        f'traced arm (det={deterministic}, width={width}) failed:\n'
        f'{proc.stdout}\n{proc.stderr}')
    hashes = [ln.split()[1] for ln in proc.stdout.splitlines()
              if ln.startswith('HASH ')]
    assert len(hashes) == reps, proc.stdout + proc.stderr
    return hashes


# ===========================================================================
# 1.  THE CLAIM D14 WITHDREW, NOW AVAILABLE.
# ===========================================================================

@pytest.mark.slow
def test_the_traced_exit_field_hashes_the_same_at_every_thread_count():
    """THE ASSERTION S7.3 OF THE D14 AUDIT COULD NOT WRITE.

    Eight calls of ``apply_real_lens_traced(carrier='auto')`` -- four BLAS
    widths x two repeats, each in a fresh interpreter with the width pinned
    before NumPy loads -- must return ONE distinct SHA-256 of the exit field.

    Asserted UNCONDITIONALLY, because unlike the fail-before below it does not
    depend on whether this build's BLAS happens to split anything: the
    property is delivered by construction (fixed block partition, BLAS-free
    per-block partial, fixed pairwise tree, and a deterministic replacement
    for the C13 step-down), so a build that never splits and a build that
    always does must both land on one hash.
    """
    seen = {}
    for w in _WIDTHS:
        seen[w] = _run_field(True, w)
    flat = sorted({h for hs in seen.values() for h in hs})
    assert len(flat) == 1, (
        'the deterministic traced field is not thread-invariant: '
        + repr(seen))


@pytest.mark.slow
def test_the_shipped_route_is_the_fail_before_on_this_fixture():
    """THE MATCHED PAIR -- and it is allowed to be inconclusive, on purpose.

    A fail-before that depends on a library's threading threshold cannot be
    asserted unconditionally: a build whose BLAS keeps these reductions serial
    is legitimate, and hard-failing on it would be exactly the
    environment-dependent precondition ``TESTING_STANDARDS`` S3 bans.  So the
    test asserts what is unconditional -- the deterministic arm reads one hash
    -- and REPORTS the shipped arm, xfailing with the evidence printed if this
    build does not split.

    MEASURED 2026-08-23 on the reference box (Windows 11, py3.14.6, numpy
    2.4.4 / scipy-openblas 0.3.31 DYNAMIC_ARCH Haswell, Ryzen 9 5950X):
    the shipped route reads FOUR distinct field hashes over widths
    {1, 2, 4, 8} on this fixture and the deterministic route reads one.
    """
    off = {w: _run_field(False, w, reps=1)[0] for w in _WIDTHS}
    on = {w: _run_field(True, w, reps=1)[0] for w in _WIDTHS}
    assert len(set(on.values())) == 1, ('deterministic arm split: ', on)
    if len(set(off.values())) == 1:
        pytest.xfail(
            'this build does not split any traced reduction at this fixture, '
            'so the matched pair is inconclusive here (the deterministic arm '
            f'still reads one hash unconditionally).  shipped: {off}')
    assert len(set(off.values())) > 1


# ===========================================================================
# 2.  THE KERNEL: why einsum, and what makes it deterministic.
# ===========================================================================

_EINSUM_PROBE = textwrap.dedent(
    """
    import hashlib, os, sys
    import numpy as np
    rng = np.random.default_rng(31415)
    for n, M in ((4096, 120), (200000, 66)):
        A = np.ascontiguousarray(rng.normal(size=(n, M)) * 1e-2)
        for tag, g in (('off', np.einsum('ri,rj->ij', A, A, optimize=False)),
                       ('on', np.einsum('ri,rj->ij', A, A, optimize=True)),
                       ('blas', A.T @ A)):
            print('E %d %d %s %s' % (
                n, M, tag,
                hashlib.sha256(np.ascontiguousarray(g).tobytes()
                               ).hexdigest()[:16]))
    """
)


@pytest.mark.slow
def test_einsum_is_blas_free_and_optimize_true_is_not():
    """THE DETERMINISM ARGUMENT FOR THE PARTIAL, AND ITS REFUTATION HALF.

    Two claims, both measured across pinned widths in fresh interpreters:

    * ``np.einsum(..., optimize=False)`` reads ONE hash at every width -- the
      same instrument D14 used to establish that ufunc reductions take no BLAS
      path.  That is what makes the per-block partial scheduling-independent
      BY CONSTRUCTION rather than by sitting below somebody's threshold.
    * ``optimize=True`` is NOT a harmless default.  einsum's optimizer routes
      this contraction through ``tensordot`` -> BLAS ``dgemm``: measured, it
      returns bytes IDENTICAL to ``A.T @ A`` and moves with the width exactly
      as the BLAS route does.  The shipped kernel passes ``optimize=False``
      explicitly for that reason, and this is the assertion that stops a
      future edit -- or a future NumPy default -- from silently undoing the
      whole layer.
    """
    rows = {}
    for w in _WIDTHS:
        proc = subprocess.run([sys.executable, '-c', _EINSUM_PROBE],
                              env=_child_env(w), capture_output=True,
                              text=True, timeout=900)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        for ln in proc.stdout.splitlines():
            if ln.startswith('E '):
                _, n, m, tag, h = ln.split()
                rows.setdefault((n, m, tag), set()).add(h)
    shapes = {(n, m) for n, m, _ in rows}
    assert shapes, 'the einsum probe produced no rows'
    for n, m in shapes:
        assert len(rows[(n, m, 'off')]) == 1, (
            f'einsum(optimize=False) at {n}x{m} is NOT width-invariant on '
            f'this build: {rows[(n, m, "off")]} -- the whole D15 partial '
            f'rests on this and the kernel must not ship without it')
    # optimize=True == BLAS is the refutation; it is allowed not to reproduce
    # on a build whose BLAS is serial, so it is reported, not asserted, when
    # the two do not move.
    same = [(n, m) for n, m in shapes
            if rows[(n, m, 'on')] == rows[(n, m, 'blas')]]
    assert same, (
        'optimize=True did not match the BLAS bytes at any shape on this '
        'build; the explicit optimize=False is then belt-and-braces rather '
        f'than load-bearing here.  rows={rows}')


def test_the_partial_route_reads_the_term_count_and_nothing_else():
    """THE DETERMINISM ARGUMENT, AS AN ASSERTION.

    Which partial runs, and how long its blocks are, must be functions of the
    TERM COUNT alone -- never of the row count, the free memory or the thread
    count, any of which would be a scheduling input and would put the defect
    back one level down.  Checked by running the same matrix twice at
    different row counts and by reconstructing the intended block list.
    """
    for n_terms in (2, 5, 7, 8, 12, 28, 66, 120):
        wide = n_terms >= LT._DET_EINSUM_MIN_TERMS
        blk = (LT._DET_EINSUM_BLOCK_ROWS if wide
               else LT._det_block_rows(n_terms))
        assert blk >= 1
        rng = np.random.default_rng(700 + n_terms)
        for n in (blk // 2 + 1, 3 * blk + 7, 11 * blk):
            A = np.ascontiguousarray(rng.normal(size=(n, n_terms)) * 1e-2)
            b = np.ascontiguousarray(rng.normal(size=n))
            # the same fixed pairwise tree, written level by level over the
            # SAME block list, must reproduce the kernel bit for bit
            parts_g, parts_r = [], []
            for i0 in range(0, n, blk):
                S, bb = A[i0:i0 + blk], b[i0:i0 + blk]
                Sc = np.ascontiguousarray(S)
                if wide:
                    g = np.einsum('ri,rj->ij', Sc, Sc, optimize=False)
                    r = np.einsum('ri,rk->ik', Sc,
                                  np.ascontiguousarray(bb).reshape(-1, 1),
                                  optimize=False).ravel()
                else:
                    g = np.empty((n_terms, n_terms))
                    r = np.empty(n_terms)
                    for p in range(n_terms):
                        cp = S[:, p]
                        for q in range(p, n_terms):
                            g[p, q] = g[q, p] = float(np.sum(cp * S[:, q]))
                        r[p] = float(np.sum(cp * bb))
                parts_g.append(g)
                parts_r.append(r)
            for parts in (parts_g, parts_r):
                while len(parts) > 1:
                    nxt = [parts[i] + parts[i + 1]
                           for i in range(0, len(parts) - 1, 2)]
                    if len(parts) % 2:
                        nxt.append(parts[-1])
                    parts[:] = nxt
            gd, rd = LT._det_normal_equations(A, b)
            assert np.array_equal(gd, parts_g[0]), (n_terms, n)
            assert np.array_equal(rd, parts_r[0]), (n_terms, n)


def test_the_gram_is_symmetric_in_both_partials():
    """The Cholesky must never see a Gram whose two triangles disagree in the
    last bit.  D14's partial mirrors the upper triangle explicitly; the einsum
    partial computes both halves, and that they come out bit-identical is a
    property of the contraction, not an assumption -- so it is checked."""
    rng = np.random.default_rng(88)
    for n, M in ((1337, 120), (1457, 28), (4000, 8), (20000, 5)):
        A = np.ascontiguousarray(rng.normal(size=(n, M)))
        b = np.ascontiguousarray(rng.normal(size=n))
        G, _ = LT._det_normal_equations(A, b)
        assert np.array_equal(G, G.T), (n, M)


def test_the_rhs_only_reduction_is_the_same_reduction():
    """:func:`~lumenairy.elements._lens_traced._det_at_b` exists so the
    refinement does not form a Gram it will not use.  It must be
    BIT-IDENTICAL to ``_det_normal_equations(A, b)[1]``: if it ever is not,
    the correction is computed by a different reduction from the solve it
    corrects, and the two would drift silently."""
    rng = np.random.default_rng(4096)
    for n, M, K in ((1337, 120, 3), (1457, 28, 1), (4000, 8, 2),
                    (20000, 7, 1), (119936, 5, 1), (37, 9, 1), (500, 12, 4)):
        A = np.ascontiguousarray(rng.normal(size=(n, M)))
        B = (np.ascontiguousarray(rng.normal(size=(n, K))) if K > 1
             else np.ascontiguousarray(rng.normal(size=n)))
        full = LT._det_normal_equations(A, B)[1]
        only = LT._det_at_b(A, B)
        assert only.shape == full.shape, (n, M, K)
        assert np.array_equal(full, only), (n, M, K)


# ===========================================================================
# 3.  ORACLE ACCURACY -- the naive transplant was WORSE, and that is pinned.
# ===========================================================================

def _errs_against_oracle(A, b):
    d14 = _oracle()
    Go, ro = d14._oracle_normal_equations(A, b)
    xo = np.linalg.solve(Go, ro)

    def rel(x, ref):
        return float(np.max(np.abs(np.asarray(x) - ref)) / np.max(np.abs(ref)))

    def errs(G, r):
        return (rel(G, Go), rel(r, ro), rel(np.linalg.solve(G, r), xo))

    det = errs(*LT._det_normal_equations(A, b))
    fam = [errs(*d14._partitioned_normal_equations(A, b, k))
           for k in d14._KSPLITS]
    return det, fam


def _wide_design(n, M, seed):
    """A total-degree / Chebyshev-shaped design: O(1) columns, a constant
    column, and enough of them that the Gram is the interesting object.  Not
    a synthetic stand-in for the fit -- the SHAPES are the traced fits'."""
    rng = np.random.default_rng(seed)
    A = np.ascontiguousarray(rng.uniform(-1.0, 1.0, size=(n, M)))
    A[:, 0] = 1.0
    return A, np.ascontiguousarray(rng.normal(size=n) * 1e-3)


@pytest.mark.parametrize('n,M', [(1337, 120), (1457, 28), (20000, 66)])
def test_the_einsum_partial_is_at_least_as_close_to_the_truth(n, M):
    """ORACLE-ACCURACY NON-REGRESSION at the TRACED fits' shapes.

    The bar is the LEGAL-PARTITION FAMILY the shipped BLAS route draws from,
    measured in this run against D14's correctly-rounded (Dekker two-product
    + ``math.fsum``) oracle, so it carries no per-build constant.  All three
    quantities are claimed against the family's WORST draw -- the same
    conservative bar D14 used for ``G`` and the right-hand side, and for the
    same reason: a ``<= best`` between two numbers of the same order is
    testing noise.

    THIS IS THE TEST THAT REFUTED THE OBVIOUS IMPLEMENTATION.  With the
    einsum partial over D14's 4096-row block it FAILS: einsum accumulates
    sequentially where ``np.sum`` is pairwise, and at (1337, 120) that reads
    G 7.65e-16 / rhs 1.64e-15 / coef 3.14e-15 against a family worst of
    1.70e-16 / 7.64e-16 / 9.82e-16 -- i.e. a determinism fix that pays for
    itself in error.  At the shipped 64-row block the same shape reads
    8.50e-17 / 3.27e-16 / 4.91e-16, at or below the family's BEST draw.
    """
    A, b = _wide_design(n, M, 20260823 + n)
    det, fam = _errs_against_oracle(A, b)
    worst = [max(f[i] for f in fam) for i in range(3)]
    best = [min(f[i] for f in fam) for i in range(3)]
    names = ('G', 'rhs', 'coefficients')
    for i, nm in enumerate(names):
        assert det[i] <= worst[i], (
            f'{nm} at {n}x{M}: deterministic {det[i]:.3e} is worse than the '
            f'legal-partition family\'s WORST draw {worst[i]:.3e} '
            f'(best {best[i]:.3e}) -- the reduction is trading accuracy for '
            f'determinism, which is not the trade this layer is allowed to '
            f'make')


def test_the_block_length_is_what_makes_that_true():
    """THE CONSTANT IS LOAD-BEARING, AND THE FAIL-BEFORE IS A VALUE OF IT.

    Re-runs the oracle comparison with the einsum block widened to D14's
    Gram-tile length.  At least one quantity must come out WORSE than at the
    shipped 64 rows, or ``_DET_EINSUM_BLOCK_ROWS`` is decoration and the
    audit's S5.1 measurement no longer holds on this build.
    """
    A, b = _wide_design(1337, 120, 20260823 + 1337)
    tight, _ = _errs_against_oracle(A, b)
    old = LT._DET_EINSUM_BLOCK_ROWS
    try:
        LT._DET_EINSUM_BLOCK_ROWS = int(LT._det_block_rows(120))
        loose, _ = _errs_against_oracle(A, b)
    finally:
        LT._DET_EINSUM_BLOCK_ROWS = old
    assert any(loose[i] > tight[i] for i in range(3)), (
        f'widening the einsum block from {old} to {LT._det_block_rows(120)} '
        f'rows did not degrade accuracy on this build (tight {tight}, loose '
        f'{loose}); the pinned constant needs re-deriving here')


# ===========================================================================
# 4.  THE C13 STEP-DOWN -- D14's "one hole", measured to be the DEFAULT here.
# ===========================================================================

def _singular_pair(n=12_000):
    """Two ENGINEERED states, D14's own fixtures.  An exactly duplicated
    column makes the Gram rank-deficient outright (Cholesky and LU both
    refuse); a column perturbed by 1e-6 relative makes one that factorises
    but screens singular under C13.  A merely RESCALED column would not do:
    ``_gram_rcond`` equilibrates the diagonal first."""
    rng = np.random.default_rng(1234)
    col = rng.normal(size=n)
    dup = np.ascontiguousarray(
        np.stack([np.ones(n), col, col, rng.normal(size=n)], axis=1))
    tiny = np.ascontiguousarray(
        np.stack([np.ones(n), col, col + 1e-6 * rng.normal(size=n),
                  rng.normal(size=n)], axis=1))
    return dup, tiny, np.ascontiguousarray(rng.normal(size=n))


def test_the_screened_singular_exit_no_longer_leaves_the_deterministic_route():
    """THE SECOND REFUTATION, PINNED.

    D14 declared the C13 step-down "the one hole" and warned the deterministic
    caller when it was taken, on the reading that it is a corner.  On the
    traced chain it is the DEFAULT -- every non-carrier fit screens singular
    -- so that reading would have made the whole layer inert.  A deterministic
    caller that reaches the screen now gets
    :func:`~lumenairy.elements._lens_traced._det_refine` instead of the
    threaded QR, and therefore must NOT be warned that its guarantee lapsed.
    """
    _dup, tiny, b = _singular_pair()
    G, _ = LT._det_normal_equations(tiny, b)
    rcond = LT._gram_rcond(G)
    assert rcond < LT._LSTSQ_GRAM_RCOND_MIN, (
        f'the fixture no longer screens singular (rcond {rcond:.3e})')
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter('always')
        x = LT._solve_lstsq_thread_safe(tiny, b, deterministic=True)
    msgs = [str(w.message) for w in got if w.category is RuntimeWarning]
    assert not any('does NOT hold' in m for m in msgs), (
        f'the deterministic route still declares a lapsed guarantee on the '
        f'C13 exit, which D15 closed: {msgs}')
    assert np.all(np.isfinite(x))
    # and it really is the refined answer, not the raw normal equations
    raw_G, raw_rhs = LT._det_normal_equations(tiny, b)
    import scipy.linalg as sla
    raw = sla.cho_solve(sla.cho_factor(raw_G, check_finite=False), raw_rhs,
                        check_finite=False)
    assert not np.array_equal(x, raw), (
        'the C13 exit returned the unrefined normal-equations answer')


def test_the_rank_deficient_hole_is_still_declared():
    """WHAT IS LEFT OF THE HOLE, asserted so it cannot be forgotten.

    A Gram so rank-deficient that Cholesky AND LU both refuse has no
    factorisation to refine through, so that exit still reroutes to
    :func:`~lumenairy.elements._lens_traced._solve_lstsq_qr` -- a threaded
    ``geqrf`` over the full ``A`` -- and the deterministic caller must still
    be TOLD, not quietly handed a weaker guarantee.  The shipped route stays
    silent: the warning is about the promise this caller was given, not about
    the data.
    """
    dup, _tiny, b = _singular_pair()
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter('always')
        LT._solve_lstsq_thread_safe(dup, b, deterministic=True)
    msgs = [str(w.message) for w in got if w.category is RuntimeWarning]
    assert any('does NOT hold' in m for m in msgs), msgs
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter('always')
        LT._solve_lstsq_thread_safe(dup, b, deterministic=False)
    assert not [w for w in got if w.category is RuntimeWarning], (
        [str(w.message) for w in got])


def test_the_refinement_refuses_where_it_does_not_converge():
    """THE THIRD REFUTATION, AND THE ONE A TEST SUITE CAUGHT.

    The first cut refined WHEREVER the C13 screen fired.  That broke
    ``test_niche_d7::test_c13_cures_the_hard_mask_fold_at_the_d7_order``: on
    the niche-D7 hard-mask design matrix the equilibrated Gram has a
    non-positive eigenvalue, refinement does not converge, and the route
    returned a fit missing the least-squares residual by 1.4e5x -- a
    determinism fix that would have shipped a wrong answer.

    So the refinement REFUSES, on a rule that is itself deterministic (it
    cannot score against the QR, whose residual moves with the width): the
    correction has to come out small relative to the answer.  Measured, the
    two populations are four decades apart in that quantity -- the traced
    fits correct by at most 9.8e-08, a system the normal equations cannot
    represent corrects by ~1.0.  This asserts BOTH sides of the corridor.
    """
    rng = np.random.default_rng(1234)
    n = 12_000
    col = rng.normal(size=n)
    b = np.ascontiguousarray(rng.normal(size=n))
    # diverging: a column perturbed by 1e-9 / 1e-12 makes a Gram the normal
    # equations cannot represent; refinement must refuse and the caller must
    # be told the guarantee lapsed
    for eps in (1e-9, 1e-12):
        A = np.ascontiguousarray(
            np.stack([np.ones(n), col, col + eps * rng.normal(size=n),
                      rng.normal(size=n)], axis=1))
        G, _ = LT._det_normal_equations(A, b)
        assert LT._gram_rcond(G) < LT._LSTSQ_GRAM_RCOND_MIN, eps
        with warnings.catch_warnings(record=True) as got:
            warnings.simplefilter('always')
            x = LT._solve_lstsq_thread_safe(A, b, deterministic=True)
        msgs = [str(w.message) for w in got if w.category is RuntimeWarning]
        assert any('does NOT hold' in m for m in msgs), (
            f'perturbation {eps}: refinement did not converge here (the '
            f'correction is O(1) of the answer) and the caller was not told; '
            f'got {msgs}')
        # and what came back is the QR, not the unconverged refinement
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r_x = LT._lstsq_residual(A, b, x)
            r_q = LT._lstsq_residual(A, b, LT._solve_lstsq_qr(A, b))
        assert r_x <= (1.0 + LT._LSTSQ_RESID_MARGIN) * r_q, (eps, r_x, r_q)

    # converging: the traced chain's own fits are all four decades inside the
    # bar, so the refusal cannot fire on them by accident
    for A, bb in _traced_fit_matrices():
        G, rhs = LT._det_normal_equations(A, bb)
        if LT._gram_rcond(G) >= LT._LSTSQ_GRAM_RCOND_MIN:
            continue
        import scipy.linalg as sla
        cf = sla.cho_factor(G, check_finite=False)
        B2 = np.asarray(bb, np.float64).reshape(np.shape(bb)[0], -1)
        x0 = sla.cho_solve(cf, rhs.reshape(A.shape[1], -1),
                           check_finite=False)
        d = sla.cho_solve(cf, LT._det_at_b(A, B2 - LT._det_matvec(A, x0)),
                          check_finite=False)
        ratio = float(np.max(np.abs(d))) / float(np.max(np.abs(x0)))
        assert ratio < LT._DET_REFINE_MAX_CORRECTION, (
            f'{A.shape}: the traced fit corrects by {ratio:.3e}, at or over '
            f'the {LT._DET_REFINE_MAX_CORRECTION:.0e} refusal bar -- the '
            f'corridor measured in S6.3 no longer holds on this build')


def _traced_fit_matrices():
    """The traced chain's OWN ``(A, b)`` pairs, captured from a real call
    rather than modelled -- a synthetic stand-in would have a different
    condition number, and the condition number is the whole subject here."""
    N, dx = 512, 30e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    w = 80 * dx
    E = (np.exp(-(X * X + Y * Y) / (w * w))
         * np.exp(1j * (2 * np.pi / _LAM) * (X * X + Y * Y) / 2.0)
         ).astype(np.complex128)
    presc = {'wavelength': _LAM, 'aperture_diameter': 14e-3, 'surfaces': [
        {'radius': 51.68e-3, 'thickness': 4e-3, 'glass_before': 'air',
         'glass_after': 'N-BK7', 'semi_diameter': 7e-3},
        {'radius': -51.68e-3, 'thickness': 0.0, 'glass_before': 'N-BK7',
         'glass_after': 'air', 'semi_diameter': 7e-3}],
        'thicknesses': [4e-3], 'stop_index': 0}
    cap = []
    orig = LT._solve_lstsq_thread_safe

    def spy(A, b, deterministic=False):
        cap.append((np.ascontiguousarray(A, np.float64),
                    np.array(b, dtype=np.float64)))
        return orig(A, b, deterministic=deterministic)

    LT._solve_lstsq_thread_safe = spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            la.apply_real_lens_traced(E, prescription=presc, wavelength=_LAM,
                                      dx=dx, carrier='auto',
                                      on_undersample='silent',
                                      ray_subsample=2)
    finally:
        LT._solve_lstsq_thread_safe = orig
    return [(A, b) for A, b in cap if A.shape[1] >= LT._DET_EINSUM_MIN_TERMS]


@pytest.mark.slow
def test_the_refined_step_down_fits_at_least_as_well_as_the_qr_it_replaces():
    """THE CLAIM THE REPLACEMENT HAS TO EARN, ON THE FITS' OWN MATRICES.

    C13 exists to lower ``||b - A x||`` where the Gram is ill-conditioned, and
    it decides between two candidates by SCORING them.  A deterministic route
    cannot keep that comparison: ``r_qr`` moves with the thread count, so a
    route that BRANCHES on it is not deterministic -- the nondeterminism just
    moves from the value into the choice.  The refinement is therefore taken
    unconditionally, and what it owes in exchange is this measurement.

    The bar is C13's OWN margin: a candidate has to BEAT the incumbent by
    ``_LSTSQ_RESID_MARGIN`` to displace it, so a replacement is acceptable
    while it is no worse than the QR by that much.  Measured 2026-08-23 it is
    far inside: at worst 1.0e-9 relative worse (a 28-term fit) and strictly
    BETTER on the two 120-term ones (8.445e-16 against QR's 9.941e-16,
    1.0230e-13 against 1.0242e-13).
    """
    mats = _traced_fit_matrices()
    assert mats, 'the traced chain ran no wide fit on this fixture'
    seen_screen = 0
    for A, b in mats:
        G, _ = LT._det_normal_equations(A, b)
        if LT._gram_rcond(G) >= LT._LSTSQ_GRAM_RCOND_MIN:
            continue
        seen_screen += 1
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            x_qr = LT._solve_lstsq_qr(A, b)
            x_det = LT._solve_lstsq_thread_safe(A, b, deterministic=True)
        r_qr = LT._lstsq_residual(A, b, x_qr)
        r_det = LT._lstsq_residual(A, b, x_det)
        assert r_det <= (1.0 + LT._LSTSQ_RESID_MARGIN) * r_qr, (
            f'{A.shape}: the deterministic refined answer fits WORSE than the '
            f'QR it replaces by more than C13\'s own margin '
            f'({r_det:.6e} vs {r_qr:.6e})')
    assert seen_screen, (
        'no traced fit screened singular on this build, so the replacement '
        'was never exercised; the S4 measurement needs re-deriving here')


# ===========================================================================
# 5.  SCOPE, AND WHAT MUST NOT MOVE.
# ===========================================================================

def test_every_least_squares_solve_on_the_traced_path_is_deterministic():
    """THE SCOPE, PINNED FROM THE OTHER SIDE.

    D14's ``test_the_traced_chain_has_a_second_reduction_this_niche_does_not_
    own`` pinned that exactly ONE of the traced chain's solves was
    deterministic.  Restated over there and completed here: with both flags
    on, ALL of them are.  A future change that adds a fit to this path, or
    withdraws one from the deterministic route, has to come back and say so.
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

    def spy(A, b, deterministic=False):
        seen.append((np.shape(A), bool(deterministic)))
        return orig(A, b, deterministic=deterministic)

    LT._solve_lstsq_thread_safe = spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            la.apply_real_lens_traced(E, prescription=presc, wavelength=_LAM,
                                      dx=30e-6, carrier='auto',
                                      on_undersample='silent')
    finally:
        LT._solve_lstsq_thread_safe = orig
    assert seen, 'the traced chain ran no least-squares solve at all'
    blas = [s for s, flag in seen if not flag]
    assert not blas, (
        f'{len(blas)} traced solve(s) still on the BLAS route: {blas} '
        f'(full census {seen})')


def test_the_flag_off_returns_the_shipped_bits_exactly():
    """``DETERMINISTIC_TRACED_FIT = False`` -- and the ``deterministic``
    kwarg's default -- must restore ``G = A.T @ A`` / ``rhs = A.T @ b`` bit
    for bit.  That is the fail-before the layer map records."""
    rng = np.random.default_rng(5151)
    A = np.ascontiguousarray(rng.normal(size=(30_000, 28)) * 1e-2)
    b = np.ascontiguousarray(rng.normal(size=30_000))
    G, rhs = A.T @ A, A.T @ b
    want = LT._solve_lstsq_thread_safe(A, b)
    import scipy.linalg as sla
    ref = sla.cho_solve(sla.cho_factor(G, check_finite=False), rhs,
                        check_finite=False)
    if LT._gram_rcond(G) >= LT._LSTSQ_GRAM_RCOND_MIN:
        assert np.array_equal(want, ref)
    assert np.array_equal(
        want, LT._solve_lstsq_thread_safe(A, b, deterministic=False))


def test_the_carrier_fit_keeps_d14s_arithmetic_exactly():
    """ACCEPTANCE (6): THE ANALYTIC PATH DOES NOT MOVE.

    ``_DET_EINSUM_MIN_TERMS`` keeps every fit below 8 terms -- the 5-term
    ``'auto'`` carrier fit among them -- on D14's ufunc partial over D14's
    block length.  Checked here by reconstructing that kernel from its two
    constants and comparing bit for bit, which is the same shape as D14's own
    ``test_the_partition_reads_the_term_count_and_nothing_else`` and does not
    need the 5.41.0 tree present to run.
    """
    assert LT._DET_EINSUM_MIN_TERMS > 5, (
        'the carrier fit is 5 terms; a threshold at or below it moves niche '
        'D14 bits and 5.41.0 released them')
    rng = np.random.default_rng(541)
    for n_terms in (2, 4, 5, 6, 7):
        blk = LT._det_block_rows(n_terms)
        n = 3 * blk + 17
        A = np.ascontiguousarray(rng.normal(size=(n, n_terms)) * 1e-2)
        b = np.ascontiguousarray(rng.normal(size=n))
        parts_g, parts_r = [], []
        for i0 in range(0, n, blk):
            S, bb = A[i0:i0 + blk], b[i0:i0 + blk]
            g = np.empty((n_terms, n_terms))
            r = np.empty(n_terms)
            for p in range(n_terms):
                cp = S[:, p]
                for q in range(p, n_terms):
                    g[p, q] = g[q, p] = float(np.sum(cp * S[:, q]))
                r[p] = float(np.sum(cp * bb))
            parts_g.append(g)
            parts_r.append(r)
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


def test_the_analytic_entry_is_inert_across_this_flag():
    """ACCEPTANCE (6), AT THE FIELD.  ``apply_real_lens`` runs no traced fit,
    so flipping :data:`DETERMINISTIC_TRACED_FIT` must be BYTE-INERT on its
    returned field -- both arms in one process, so nothing but the flag
    differs."""
    N, dx = 256, 30e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    w = 40 * dx
    E = (np.exp(-(X * X + Y * Y) / (w * w))
         * np.exp(1j * (2 * np.pi / _LAM) * (X * X + Y * Y) / 2.0)
         ).astype(np.complex128)
    presc = {'wavelength': _LAM, 'aperture_diameter': 14e-3, 'surfaces': [
        {'radius': 51.68e-3, 'thickness': 4e-3, 'glass_before': 'air',
         'glass_after': 'N-BK7', 'semi_diameter': 7e-3},
        {'radius': -51.68e-3, 'thickness': 0.0, 'glass_before': 'N-BK7',
         'glass_after': 'air', 'semi_diameter': 7e-3}],
        'thicknesses': [4e-3], 'stop_index': 0}
    out = []
    old = LT.DETERMINISTIC_TRACED_FIT
    try:
        for flag in (False, True):
            LT.DETERMINISTIC_TRACED_FIT = flag
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                out.append(np.asarray(la.apply_real_lens(
                    E, prescription=presc, wavelength=_LAM, dx=dx,
                    surface_model='tangent_facet', carrier='auto')))
    finally:
        LT.DETERMINISTIC_TRACED_FIT = old
    assert np.array_equal(out[0], out[1]), float(
        np.abs(out[0] - out[1]).max())


def test_multi_rhs_matches_the_single_rhs_column_by_column():
    """The inverse-map exit fit hands the solver a 3-column ``b``, so the
    einsum partial has to take one -- and column ``k`` of the multi-RHS answer
    must be the single-RHS answer for that column, bit for bit, or the two
    shapes are two different reductions."""
    rng = np.random.default_rng(2719)
    for M in (5, 28, 120):
        A = np.ascontiguousarray(rng.normal(size=(3000, M)) * 1e-2)
        B = np.ascontiguousarray(rng.normal(size=(3000, 3)))
        G2, r2 = LT._det_normal_equations(A, B)
        assert r2.shape == (M, 3)
        for k in range(3):
            _, r1 = LT._det_normal_equations(
                A, np.ascontiguousarray(B[:, k]))
            assert r1.shape == (M,)
            assert np.array_equal(r1, r2[:, k]), (M, k)


# ===========================================================================
# 6.  FOOTPRINT.
# ===========================================================================

def test_the_traced_fit_footprint_is_not_regressed():
    """FOOTPRINT NON-REGRESSION, against a DERIVED allowance rather than a
    pinned number.

    The einsum partial's live set is smaller than D14's by construction: the
    block is a VIEW of the design matrix (no transposed tile, no row buffer),
    and the carry-stack holds ``O(log2(n / blk))`` partial Grams rather than
    ``O(n / blk)`` of them.  The allowance below is computed from the two
    constants in force, so it moves when they do.
    """
    rng = np.random.default_rng(31)
    n, M = 200_000, 66
    A = np.ascontiguousarray(rng.normal(size=(n, M)) * 1e-2)
    b = np.ascontiguousarray(rng.normal(size=n))
    peaks = {}
    for tag in ('blas', 'det'):
        tracemalloc.start()
        if tag == 'blas':
            _G, _r = A.T @ A, A.T @ b
        else:
            _G, _r = LT._det_normal_equations(A, b)
        peaks[tag] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        del _G, _r
    blk = LT._DET_EINSUM_BLOCK_ROWS
    depth = max(1, int(math.ceil(math.log2(max(2, n / blk)))) + 1)
    # one block copy (a view in the shipped path, but bound the copy anyway),
    # the (M, M) and (M, K) partials at every live tree depth, and one
    # (M, M) + (M, K) result
    allow = (blk * M * 8.0
             + depth * (M * M + M) * 8.0
             + 2.0 * (M * M + M) * 8.0)
    assert peaks['det'] <= peaks['blas'] + allow, (
        f"deterministic peak {peaks['det']} exceeds the BLAS peak "
        f"{peaks['blas']} by more than the derived allowance {allow:.0f} B "
        f"(blk={blk}, M={M}, tree depth {depth})")
