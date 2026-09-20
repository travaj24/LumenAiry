"""Wave-5 hygiene-2, H2-1 (audit item 14) -- the direct-matrix MFT branch.

``lumenairy/propagators/mft.py``'s Notes and ``_bluestein.py``'s module
docstring have both named a "direct matrix-Fourier transform" as the chirp-Z
reduction's alternative since the MFT propagators were written, without
shipping one.  5.48.0 ships it as an OPT-IN: ``method='direct'`` on the three
public MFT entry points, :func:`~lumenairy.propagators._bluestein.
_direct_matrix_2d` underneath, and the shipped default untouched.

WHAT IS ASSERTED HERE, and what is only measured elsewhere.

Every claim in this file is a DECISION -- a route taken, a refusal, an
ordering, a residual against a bar with decades on both sides.  Nothing pins a
timing (the box runs other work; ``validation/probe_wave5_hyg2/
mft_direct_{win,wsl}.json`` carries the crossover table and the report states
its contention), and nothing pins a byte count read off one build.

The tolerance bars are DERIVED from the summation itself, not from a residual
somebody once saw.  Each output point is a sum of ``n = Ny*Nx`` terms whose
kernel has unit modulus, so its summation condition number is
``kappa = sum|E| / |F|`` and a summation whose growth factor is ``g`` commits
at most ``g * eps * sum|E|`` of ABSOLUTE error.  Two routes therefore agree to
``(g_a + g_b) * eps * sum|E|``, with

* ``g = log2(n/128) + 8`` for NumPy's pairwise summation (the reference),
* ``g = 3 * log2(L^2)`` for a chirp-Z route (three FFTs of length ``L^2``),
* ``g = sqrt(n)`` for the dense route's two BLAS products.

The bar is ABSOLUTE, so what is asserted against it is the MAX-ABS departure.
MEASURED 2026-09-15/19 on Windows py3.14 (numpy 2.4.4) and WSL py3.12
(numpy 2.4.6, scipy-openblas SkylakeX), at ``N = 16/32/48``: the dense route
sits ``2.17 .. 2.59`` decades inside its bar and the chirp-Z routes ``1.47 ..
1.98``.  The smallest real signal above is an O(1) wrong answer, so there are
fourteen decades on the other side.

THE MARGIN AND ITS RANGE OF VALIDITY (corrected 2026-09-19,
VERIFY-WAVE5-HYGIENE2 V-D8).  An earlier wording here said "the margin
narrowing with ``n``, as it must, because the growth factors are upper
bounds".  That is backwards twice over: a bar that grows faster than the error
it bounds gives a WIDENING margin, and the two routes do OPPOSITE things.
MEASURED over ``N -> M`` = 16->8, 48->24, 96->48, 128->64, 192->96, 256->128
on both builds:

* the DENSE route's margin WIDENS, 1.84 -> 3.60 decades, because its error
  grows like ``sqrt(n)`` against a bar that grows like ``n``;
* the chirp-Z routes' margin NARROWS, 1.93 -> 0.73 decades, because their
  error grows FASTER than ``3*log2(L^2) * eps * sum|E|``.

The bar is a bound for both at every shape measured -- it is never crossed up
to N = 256 on either build -- and the shapes this file exercises (max 32x24)
are 1.6-2.4 decades inside it, so the shipped assertions are sound.  What is
finite is the chirp-Z half's lifetime: extrapolating the measured trend, its
bar would first be crossed near ``N ~ 4000-5000``.  Nothing here goes near
that, and nothing in the formula says so, which is why it is written down.
The trend itself is gated, as a trend, by
``tests/unit/test_verify_wave5_hyg2.py::
test_the_two_reductions_margins_move_in_opposite_directions_with_n``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from lumenairy.propagators._bluestein import (
    _SUM_METHODS, _bluestein_2d, _bluestein_centred_2d, _clear_h_fft_cache,
    _direct_matrix_2d)
from lumenairy.propagators.fft_infra import _fft2, _ifft2
from lumenairy.propagators.mft import (
    angular_spectrum_propagate_mft, fraunhofer_propagate_mft,
    fresnel_propagate_mft)

WL = 633e-9
EPS = float(np.finfo(np.float64).eps)


def _bits(a):
    """The array's bytes, for an exact comparison.  ``ascontiguousarray``
    first: a non-contiguous result cannot be ``view``-ed to float64, and the
    copy preserves every bit."""
    return np.ascontiguousarray(a).view(np.float64)


def _rand(ny, nx, seed=20260915):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((ny, nx))
            + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)


def _gauss(n, dx, w, seed=3):
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    rng = np.random.default_rng(seed)
    speckle = 1.0 + 0.1 * (rng.standard_normal((n, n))
                           + 1j * rng.standard_normal((n, n)))
    return (np.exp(-(X ** 2 + Y ** 2) / (w * w)) * speckle).astype(
        np.complex128)


def _pairwise_reference(E, alpha_x, alpha_y, M_y, M_x, sign,
                        c_in=(0.0, 0.0), c_out=(0.0, 0.0)):
    """The SAME sum, one output point at a time, summed by ``np.sum``.

    ``np.sum`` over a contiguous array is PAIRWISE with an unrolled block of
    128, so its growth factor is ``log2(n/128) + 8`` -- logarithmic where a
    naive loop is linear.  No mpmath and no float128: the bar below is derived
    from that growth factor, which is what makes this a reference rather than
    a fourth implementation.

    The phase is reduced modulo one turn, as the shipped dense route does, so
    what separates this from the routes under test is the SUMMATION and
    nothing else.
    """
    ny, nx = E.shape
    n_x = np.arange(nx, dtype=np.float64) - float(c_in[0])
    n_y = np.arange(ny, dtype=np.float64) - float(c_in[1])
    out = np.empty((M_y, M_x), dtype=np.complex128)
    for ky in range(M_y):
        ty = alpha_y * (float(ky) - float(c_out[1])) * n_y
        wy = np.exp(1j * sign * 2.0 * np.pi * (ty - np.rint(ty)))
        for kx in range(M_x):
            tx = alpha_x * (float(kx) - float(c_out[0])) * n_x
            wx = np.exp(1j * sign * 2.0 * np.pi * (tx - np.rint(tx)))
            out[ky, kx] = np.sum(E * (wy[:, None] * wx[None, :]))
    return out


def _bars(E, M_y, M_x):
    """``(bar_chirp, bar_dense, sum_abs)`` -- the derived absolute bars."""
    from scipy.fft import next_fast_len
    ny, nx = E.shape
    n = ny * nx
    L = float(next_fast_len(int(max(ny, nx) + max(M_y, M_x) - 1)))
    g_pair = float(np.log2(max(n / 128.0, 2.0)) + 8.0)
    g_chirp = float(3.0 * np.log2(L * L))
    g_dense = float(np.sqrt(n))
    s = float(np.sum(np.abs(E)))
    return ((g_chirp + g_pair) * EPS * s, (g_dense + g_pair) * EPS * s, s)


# ===========================================================================
# 1.  The dense route computes the same sum as the chirp-Z reductions
# ===========================================================================

@pytest.mark.parametrize("sign", (-1, +1))
@pytest.mark.parametrize("shape", ((16, 16, 8, 8), (32, 24, 16, 20),
                                   (17, 9, 5, 23)))
def test_every_route_agrees_with_the_pairwise_reference(shape, sign):
    """All three routes against an independently summed float64 reference,
    each against its OWN derived bar.

    Not "they agree with each other": that would pass if all three shared a
    mistake.  The reference is a different association order over the same
    terms, and the bar is the two growth factors the two summations carry.
    """
    ny, nx, my, mx = shape
    E = _rand(ny, nx)
    alpha = 1.0 / 64.0
    ref = _pairwise_reference(E, alpha, alpha, my, mx, sign)
    bar_chirp, bar_dense, _ = _bars(E, my, mx)
    got = {}
    for method, kw in (('bluestein', dict(separable=False, method='auto')),
                       ('separable', dict(separable=True, method='auto')),
                       ('direct', dict(method='direct'))):
        _clear_h_fft_cache()
        F = _bluestein_2d(E, alpha, alpha, my, mx, sign=sign, xp=np,
                          fft2=_fft2, ifft2=_ifft2, **kw)
        got[method] = F
        assert F.shape == (my, mx)
        bar = bar_dense if method == 'direct' else bar_chirp
        err = float(np.max(np.abs(F - ref)))
        assert err < bar, (
            f"{method} departs from the pairwise reference by {err:.3e}, "
            f"past its derived bar {bar:.3e}")
    # ... and the two chirp-Z reductions agree with the dense one within the
    # SUM of their two bars, which is the statement "one sum, three orders".
    for a in ('bluestein', 'separable'):
        d = float(np.max(np.abs(got[a] - got['direct'])))
        both = bar_chirp + bar_dense
        assert d < both, (
            f"{a} and direct differ by {d:.3e}, past {both:.3e}")


def test_the_centred_primitive_is_the_same_sum_on_the_dense_route():
    """The centred convention, which is what every MFT propagator uses.

    The chirp-Z route reaches it by a pre-chirp / post-chirp / constant
    decomposition of ``(n - cI)(k - cO)``; the dense route builds the centred
    kernel in one step.  Same sum, different bits -- pinned against the
    reference with the centres carried through it.
    """
    E = _rand(20, 24)
    alpha = 1.0 / 48.0
    my, mx = 12, 10
    c_in = (24 / 2.0, 20 / 2.0)
    c_out = (mx / 2.0 - 1.3, my / 2.0)
    ref = _pairwise_reference(E, alpha, alpha, my, mx, -1,
                              c_in=c_in, c_out=c_out)
    bar_chirp, bar_dense, _ = _bars(E, my, mx)
    kw = dict(n_centre_in_x=c_in[0], n_centre_in_y=c_in[1],
              k_centre_out_x=c_out[0], k_centre_out_y=c_out[1],
              sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2)
    F_b = _bluestein_centred_2d(E, alpha, alpha, my, mx, **kw)
    F_d = _bluestein_centred_2d(E, alpha, alpha, my, mx, method='direct', **kw)
    assert float(np.max(np.abs(F_b - ref))) < bar_chirp
    assert float(np.max(np.abs(F_d - ref))) < bar_dense


def test_the_dense_kernel_serves_both_index_conventions():
    """ONE implementation, not two: the non-centred primitive's convention is
    the centred one at zero centres, and :func:`_direct_matrix_2d` is called
    with exactly that from both places."""
    E = _rand(12, 14)
    a = _direct_matrix_2d(E, 0.02, 0.03, 7, 9, sign=-1, xp=np)
    b = _direct_matrix_2d(E, 0.02, 0.03, 7, 9, sign=-1, xp=np,
                          n_centre_in_x=0.0, n_centre_in_y=0.0,
                          k_centre_out_x=0.0, k_centre_out_y=0.0)
    assert np.array_equal(_bits(a), _bits(b))


# ===========================================================================
# 2.  The shipped default is the route it always was
# ===========================================================================

def test_the_default_takes_the_chirp_z_route_bit_for_bit():
    """``method='auto'`` must be the pre-5.48 dispatch EXACTLY -- the same
    arm, the same bits -- not merely "close".

    Asserted as byte equality against the route named explicitly, on both
    settings of ``separable``, because the whole opt-in claim rests on the
    default not moving.  (The archive-to-archive proof against 5.47.0 is
    ``validation/probe_wave5_hyg2/mftbit_{win,wsl}_compare.json``; this is its
    in-suite counterpart, which an installed wheel can still run.)
    """
    E = _rand(28, 22)
    for sep, named in ((False, 'bluestein'), (True, 'separable')):
        _clear_h_fft_cache()
        auto = _bluestein_2d(E, 0.017, 0.019, 15, 13, sign=-1, xp=np,
                             fft2=_fft2, ifft2=_ifft2, separable=sep)
        _clear_h_fft_cache()
        explicit = _bluestein_2d(E, 0.017, 0.019, 15, 13, sign=-1, xp=np,
                                 fft2=_fft2, ifft2=_ifft2, method=named)
        assert np.array_equal(_bits(auto), _bits(explicit)), (
            f"method='auto' with separable={sep} is not method={named!r}")


@pytest.mark.parametrize("fn", (fresnel_propagate_mft,
                                fraunhofer_propagate_mft,
                                angular_spectrum_propagate_mft))
def test_the_propagators_default_is_byte_identical_to_omitting_the_keyword(fn):
    """A caller who never heard of ``method`` and a caller who passes its
    default must get the same bytes out of every public entry point."""
    E = _gauss(64, 8e-6, 60e-6)
    z = 2e-2
    dx_out = WL * z / (64 * 8e-6)
    a = fn(E, z, WL, 8e-6, dx_out, 64)
    b = fn(E, z, WL, 8e-6, dx_out, 64, method='auto')
    assert np.array_equal(_bits(a), _bits(b))


@pytest.mark.parametrize("fn", (fresnel_propagate_mft,
                                fraunhofer_propagate_mft,
                                angular_spectrum_propagate_mft))
def test_the_direct_route_reaches_the_same_physics(fn):
    """The opt-in route must be the same PROPAGATOR, not merely the same
    transform: the field it returns agrees with the default's to a bar derived
    from the transform's own summation, and its power is conserved to the same
    reading.

    The bar is formed on the transform stage's operands (the propagator's
    quadratic screens are elementwise and identical on both routes), so it is
    the two-route bar of the tests above carried through a unit-modulus
    prefactor -- stated relative to the field's own peak.
    """
    N, dx = 64, 8e-6
    E = _gauss(N, dx, 60e-6)
    z = 2e-2
    dx_out = WL * z / (N * dx)
    a = fn(E, z, WL, dx, dx_out, N)
    b = fn(E, z, WL, dx, dx_out, N, method='direct')
    bar_chirp, bar_dense, s = _bars(E, N, N)
    # relative to peak: the propagator's prefactor has unit modulus up to the
    # 1/(lambda z) scale, which divides out of a ratio taken against the
    # route's own peak.
    rel = float(np.max(np.abs(a - b)) / np.max(np.abs(a)))
    bar = (bar_chirp + bar_dense) / float(np.max(np.abs(E)))
    assert rel < bar, (f"{fn.__name__}: direct vs default {rel:.3e} past the "
                       f"derived bar {bar:.3e}")
    pa = float(np.sum(np.abs(a) ** 2))
    pb = float(np.sum(np.abs(b) ** 2))
    assert abs(pa - pb) / pa < 1e-12


# ===========================================================================
# 3.  The vocabulary is closed, and the guard belongs to the route that has it
# ===========================================================================

@pytest.mark.parametrize("bad", ('Direct', 'DIRECT', 'dense', '', None, 1))
def test_an_unknown_method_is_refused_not_silently_defaulted(bad):
    """Same rule as ``gap_kernel`` (defect D4): a vocabulary that falls through
    to a default on an unrecognised value turns a typo into a silent route
    change."""
    E = _rand(8, 8)
    with pytest.raises(ValueError, match="method must be one of"):
        _bluestein_2d(E, 0.01, 0.01, 4, 4, sign=-1, xp=np, fft2=_fft2,
                      ifft2=_ifft2, method=bad)
    with pytest.raises(ValueError, match="method must be one of"):
        _bluestein_centred_2d(E, 0.01, 0.01, 4, 4, sign=-1, xp=np,
                              fft2=_fft2, ifft2=_ifft2, method=bad)


def test_the_vocabulary_the_error_names_is_the_vocabulary_that_works():
    """The refusal message lists the accepted values; every one of them must
    actually be accepted.  A message that names a value the code rejects is
    worse than no message."""
    E = _rand(8, 8)
    for m in ('auto',) + tuple(_SUM_METHODS):
        out = _bluestein_2d(E, 0.01, 0.01, 4, 4, sign=-1, xp=np, fft2=_fft2,
                            ifft2=_ifft2, method=m)
        assert out.shape == (4, 4)


def test_the_two_primitives_report_the_same_first_error():
    """Vocabulary before geometry, and the SAME order in both primitives.

    MEASURED 2026-09-19 (VERIFY-WAVE5-HYGIENE2 V-D17):
    ``_bluestein_centred_2d([[1+0j, 2+0j]], ..., method='bogus')`` raised
    ``AttributeError: 'list' object has no attribute 'shape'`` because it read
    ``E.shape`` before it checked ``method``, while ``_bluestein_2d`` with the
    same arguments raised the designed ``ValueError``.  With
    ``sign=0, method='bogus'`` the two even named DIFFERENT first errors.  A
    caller who mistypes a keyword should be told which keyword, by whichever
    primitive they reached.

    The input is deliberately a LIST, not an array: that is what makes the
    ordering observable at all.
    """
    E = [[1 + 0j, 2 + 0j], [3 + 0j, 4 + 0j]]
    kw = dict(xp=np, fft2=_fft2, ifft2=_ifft2)
    for sign, method, needle in ((-1, 'bogus', 'method'),
                                 (0, 'bogus', 'sign'),
                                 (0, 'auto', 'sign')):
        seen = {}
        for name, fn in (('_bluestein_2d', _bluestein_2d),
                         ('_bluestein_centred_2d', _bluestein_centred_2d)):
            with pytest.raises(ValueError) as exc:
                fn(E, 0.01, 0.01, 2, 2, sign=sign, method=method, **kw)
            seen[name] = str(exc.value)
            assert needle in seen[name], (
                f"{name}(sign={sign}, method={method!r}) reported "
                f"{seen[name]!r}, which does not name {needle}")
        assert seen['_bluestein_2d'].split(' must ')[0] == \
            seen['_bluestein_centred_2d'].split(' must ')[0], (
            f"the two primitives disagree on which keyword is wrong first: "
            f"{seen}")


def _exact_phase_rows(alpha, n_in, n_out):
    """``frac(alpha * n * k)`` in ``[-1/2, 1/2)``, reduced EXACTLY.

    ``alpha`` is a float64 and therefore an exact rational; ``n`` and ``k``
    are integers.  Formed in :class:`fractions.Fraction`, the product and its
    fractional part are exact, so the ONE float64 rounding happens on a number
    already inside ``[-1/2, 1/2)`` and the phase keeps all 53 bits whatever
    the budget.

    This is the one thing the route does NOT do, and it is the whole reason
    this helper exists (VERIFY-WAVE5-HYGIENE2 round 2, D-1): a reference that
    forms ``t = alpha*k*n`` in float64 and then reduces it commits the same
    two roundings the route commits, and therefore agrees with the route by
    CONSTRUCTION at every budget.  The kernel is separable, so only
    ``n_in * n_out`` exact reductions are needed per axis.
    """
    from fractions import Fraction
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


def _fsum_reference(E, alpha, M, sign=-1):
    """The same sum again, correctly rounded in BOTH senses: an EXACT phase
    and a ``math.fsum`` summation.

    :func:`_pairwise_reference` is the right reference for comparing two
    SUMMATIONS, because it commits the same class of error as the routes it is
    compared against.  It is the wrong reference for measuring how much a
    PHASE costs, because at a large budget the reference's own phase is as
    wrong as the route's.

    REBUILT 2026-09-20 (VERIFY-WAVE5-HYGIENE2 round 2, D-1).  Until then this
    helper made only the SUMMATION correctly rounded and formed its phase as
    ``ty = alpha * ky * n_y`` followed by ``ty - rint(ty)`` -- the two
    roundings :func:`_direct_matrix_2d` itself commits.  Against it the dense
    route read 2.8e-16 .. 4.5e-16 at every budget, which was a measurement of
    the instrument and not of the route: it is what a reference that shares a
    route's phase reads about that route.  The phase is now reduced exactly
    (:func:`_exact_phase_rows`), which is what makes the two ids below able to
    see the dense route's own ``eps * budget`` error at all.
    """
    ny, nx = E.shape
    Wy = np.exp(1j * sign * 2.0 * np.pi * _exact_phase_rows(alpha, ny, M))
    Wx = np.exp(1j * sign * 2.0 * np.pi * _exact_phase_rows(alpha, nx, M))
    out = np.empty((M, M), dtype=np.complex128)
    for ky in range(M):
        wy = Wy[ky]
        for kx in range(M):
            T = E * (wy[:, None] * Wx[kx][None, :])
            out[ky, kx] = complex(math.fsum(T.real.ravel()),
                                  math.fsum(T.imag.ravel()))
    return out


def _naive_phase_reference(E, alpha, M, sign=-1):
    """The reference :func:`_fsum_reference` USED to be: correctly-rounded
    SUMMATION over a phase formed the way the route forms it.

    Kept, and used, as the premise arm of the id below: it is what makes
    "the reference is the instrument" a measurable statement rather than an
    argument.
    """
    ny, nx = E.shape
    n_x = np.arange(nx, dtype=np.float64)
    n_y = np.arange(ny, dtype=np.float64)
    out = np.empty((M, M), dtype=np.complex128)
    for ky in range(M):
        ty = alpha * ky * n_y
        wy = np.exp(1j * sign * 2.0 * np.pi * (ty - np.rint(ty)))
        for kx in range(M):
            tx = alpha * kx * n_x
            wx = np.exp(1j * sign * 2.0 * np.pi * (tx - np.rint(tx)))
            T = E * (wy[:, None] * wx[None, :])
            out[ky, kx] = complex(math.fsum(T.real.ravel()),
                                  math.fsum(T.imag.ravel()))
    return out


def _chirp_and_dense_at(budget, N=24, M=12, seed=5, naive_ref=False):
    """``(rel_chirp, warned_chirp, rel_dense, warned_dense)`` at one budget.

    ``naive_ref=True`` measures against :func:`_naive_phase_reference`
    instead -- the instrument arm, not a second claim.
    """
    import warnings as _w
    from lumenairy.propagators._bluestein import _bluestein_2d as _b2
    E = _rand(N, N, seed=seed)
    alpha = budget / float(N) ** 2
    ref = (_naive_phase_reference(E, alpha, M) if naive_ref
           else _fsum_reference(E, alpha, M))
    out = []
    for kw in ({}, {'method': 'direct'}):
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter('always')
            got = _b2(E, alpha, alpha, M, M, sign=-1, xp=np, fft2=_fft2,
                      ifft2=_ifft2, **kw)
        out.append(float(np.linalg.norm(got - ref) / np.linalg.norm(ref)))
        out.append(any('chirp phase' in str(c.message) for c in caught))
    return tuple(out)


def test_the_exact_phase_reference_is_exact_where_float64_can_check_it():
    """THE PREMISE the two ids below rest on: the reference's own phase.

    :func:`_exact_phase_rows` is only useful if it is genuinely a DIFFERENT
    instrument from the one the route carries, and genuinely the SAME
    transform.  Both directions are measured here, on the arithmetic itself
    rather than through a transform:

    * where the float64 product ``alpha*n*k`` is EXACT -- a dyadic ``alpha``,
      so no bit is thrown away -- the exact reduction and the route's
      reduction must agree exactly, modulo whole turns (``rint``'s
      ties-to-even resolves ``|t| = 1/2`` the other way, and that is a whole
      turn apart, not an error).  MEASURED 2026-09-20 on both builds: the
      fractional part of the difference is EXACTLY 0.0 at every one of four
      dyadic alphas;
    * where it is not exact -- the shipped ``alpha = 1e12/24^2`` -- they must
      PART, or the exact reference would be measuring nothing.  MEASURED
      5.293e-05 of a turn, identical on both builds.

    The gap between 0.0 and 5.3e-05 is the whole instrument; there is no bar
    here that noise could cross, because the first reading is exactly zero by
    construction and the second is eleven decades above float64 round-off.
    """
    for alpha in (0.125, 0.5, 2.0 ** -7, 3.0 * 2.0 ** -5):
        T_exact = _exact_phase_rows(alpha, 24, 12)
        n = np.arange(24, dtype=np.float64)
        k = np.arange(12, dtype=np.float64)
        t = alpha * k[:, None] * n[None, :]
        T_route = t - np.rint(t)
        d = T_exact - T_route
        assert np.array_equal(d, np.rint(d)), (
            f"at the dyadic alpha = {alpha!r} the float64 product alpha*n*k "
            f"is exactly representable, so the exact reduction and the "
            f"route's must differ by a whole number of turns; the worst "
            f"fractional departure is {np.max(np.abs(d - np.rint(d))):.3e}")
    alpha = 1e12 / 24.0 ** 2
    T_exact = _exact_phase_rows(alpha, 24, 12)
    t = alpha * np.arange(12.0)[:, None] * np.arange(24.0)[None, :]
    d = T_exact - (t - np.rint(t))
    parted = float(np.max(np.abs(d - np.rint(d))))
    assert parted > 1e-9, (
        f"at a budget of 1e12 the exact and float64 reductions agree to "
        f"{parted:.3e} of a turn; the exact reference would then be the same "
        f"instrument as the route and could not measure its phase error")


def test_both_routes_follow_the_budget_law_and_dense_wins_by_a_bounded_factor():
    """THE LAW the threshold is derived from, re-measured by the gate itself,
    against a reference whose phase is EXACT.

    The relative L2 of BOTH routes is LINEAR in the phase budget
    ``alpha * N_max^2`` -- there is no cliff to sit just below, and the
    historical "approaches float64 precision limit (1e15-1e16)" wording
    described a cliff that does not exist.

    CORRECTED 2026-09-20 (VERIFY-WAVE5-HYGIENE2 round 2, D-1).  This id used
    to be called ``..._and_dense_is_immune`` and asserted ``rd < 1e-14``.  It
    could not fail: :func:`_fsum_reference` formed its phase with the same two
    float64 roundings :func:`_direct_matrix_2d` commits, so the two agreed by
    construction.  With the reference's phase reduced exactly, MEASURED on the
    shipped N = 24 -> M = 12 fixture, IDENTICAL TO THE DIGIT on Windows
    py3.14 and WSL py3.12 (``validation/probe_wave5_hyg2_round3/
    r3_budget_exact.py``):

        budget            1e5       1e9       1e12      1e15
        chirp-Z rel L2    2.772e-11 1.635e-07 1.865e-04 2.285e-01
        DENSE rel L2      6.871e-12 1.104e-07 8.621e-05 6.739e-02
        eps * budget      2.220e-11 2.220e-07 2.220e-04 2.220e-01
        chirp / dense     4.035     1.481     2.163     3.391

    WHAT IS ASSERTED, and why each bar has a gap on both sides.

    1.  Both routes obey ``rel = C * eps * budget`` with ``C`` inside
        ``[0.03, 5]``.  MEASURED ``C``: 0.74 .. 1.25 (chirp-Z) and 0.30 ..
        0.50 (dense), so both sit a full decade inside each edge.  The
        refuted hypothesis -- the dense route immune, reading the summation
        floor of 3.2e-16 -- puts ``C`` at 1.4e-12 at a budget of 1e12, ten
        decades BELOW the lower edge.  The smallest real signal on the other
        side is an O(1) wrong answer, ``C = 4.5e10`` at the bottom rung, ten
        decades ABOVE the upper edge.
    2.  The fitted slope of each route over TEN decades of budget is 1 within
        0.1.  The band is derived from the rival hypotheses, not from taste:
        a budget-INDEPENDENT route fits slope 0 and a quadratic one fits
        slope 2, so 0.1 is one ninth of the distance to the nearer of them.
        MEASURED 0.992 (chirp-Z) and 0.998 (dense).
    3.  The dense route is the more accurate one at every budget, by a
        BOUNDED factor: ``1 < chirp/dense < 30``.  MEASURED 1.48 .. 4.04 here
        and 4.4 .. 11.8 on the centred index convention, against a factor of
        6.3e9 at a budget of 1e12 if the immunity reading were true -- eight
        decades above the upper edge.
    4.  PREMISE-GATED on the instrument: the same routes measured against the
        OLD naive-phase reference read the summation floor, which is what
        makes claim 3 a statement about the route rather than a reading of
        the reference.

    Nothing here pins the THRESHOLD; the next id does that, so this one stays
    true whatever the threshold becomes.
    """
    from lumenairy.propagators._bluestein import _EPS64
    budgets = (1e5, 1e9, 1e12, 1e15)
    rows = [(b,) + _chirp_and_dense_at(b) for b in budgets]
    for b, rc, _wc, rd, _wd in rows:
        pred = _EPS64 * b
        for name, r in (('chirp-Z', rc), ('DENSE', rd)):
            assert 0.03 * pred < r < 5.0 * pred, (
                f"at budget {b:.0e} the {name} relative error is {r:.3e}, "
                f"outside C in [0.03, 5] times eps*budget = {pred:.3e} "
                f"(C = {r / pred:.4f}).  Both routes follow that law -- the "
                f"dense one with a smaller constant, NOT with immunity")
        factor = rc / rd
        assert 1.0 < factor < 30.0, (
            f"at budget {b:.0e} the chirp-Z route is {factor:.3f}x the dense "
            f"route's error.  Below 1 the dense route stopped being the more "
            f"accurate one; above 30 it has stopped obeying the same law "
            f"(measured 1.48 .. 4.04 on this fixture, 2026-09-20)")
    x = np.log10([r[0] for r in rows])
    for name, col in (('chirp-Z', 1), ('dense', 3)):
        slope = float(np.polyfit(x, np.log10([r[col] for r in rows]), 1)[0])
        assert abs(slope - 1.0) < 0.1, (
            f"the {name} error grows as budget^{slope:.4f} over "
            f"{x.max() - x.min():.0f} decades, not linearly; slope 0 would "
            f"be a budget-independent route and slope 2 a quadratic law, and "
            f"the threshold's derivation rests on this being 1")
    assert x.max() - x.min() >= 5.0, (
        "PREMISE: the ladder spans fewer than five decades of budget, which "
        "is not enough to tell a linear law from a constant")

    # ... and the PREMISE for all of it: against the reference this file used
    # to carry, the dense route reads the summation floor at every budget --
    # i.e. the old reading measured the instrument.
    for b in (1e9, 1e12):
        _rc, _wc, rd_naive, _wd = _chirp_and_dense_at(b, naive_ref=True)
        assert rd_naive < 1e-14, (
            f"PREMISE: at budget {b:.0e} the dense route departs from a "
            f"reference that forms its phase the SAME way by {rd_naive:.3e}; "
            f"that reference is supposed to agree with it by construction, "
            f"and the contrast between the two references is what this id's "
            f"correction rests on")


def test_the_phase_budget_threshold_is_the_budget_that_keeps_six_figures():
    """The THRESHOLD, two-sided, against the law above.

    DERIVED: ``_PHASE_BUDGET_MAX = 1e-6 / eps = 4.5036e9`` is the budget at
    which six significant figures remain.  Both sides are asserted, because a
    threshold with only one side is a preference:

    * just BELOW it the routes are quiet AND still accurate -- MEASURED
      6.786e-07 at 0.9x the threshold, inside the 1e-6 the threshold is
      named for;
    * just ABOVE it the guard fires AND the error really has passed 1e-6 --
      MEASURED 2.028e-06 at 2.2x the threshold.

    Both readings RE-MEASURED 2026-09-20 against the exact-phase reference
    (round 2, D-1) and identical on both builds; the old 5.32e-07 / 1.90e-06
    were the same quantities against the naive-phase reference, which for the
    CHIRP route differs only in the last figure.

    WHY THE OLD 1e15 IS GONE (VERIFY-WAVE5-HYGIENE2 V-D5).  It was 7.5 decades
    late: at 1e12 the route returned an answer wrong in the fourth significant
    figure in silence, and the first budget that warned at all was 3.16e15, by
    which point the answer was 25 % wrong.  This is a change in WARNING
    behaviour, not a byte move -- no route's arithmetic changed -- and the
    quiet-caller side of it is the id after this one.
    """
    from lumenairy.propagators._bluestein import _EPS64, _PHASE_BUDGET_MAX

    # the derivation itself, so the constant cannot drift from its reason
    assert _PHASE_BUDGET_MAX * _EPS64 == pytest.approx(1e-6, rel=1e-12), (
        f"_PHASE_BUDGET_MAX = {_PHASE_BUDGET_MAX:.6e} is not 1e-6/eps; the "
        f"constant and the accuracy it claims have parted company")

    rc_lo, w_lo, rd_lo, wd_lo = _chirp_and_dense_at(_PHASE_BUDGET_MAX * 0.9)
    assert not w_lo and not wd_lo, (
        "the guard fires BELOW its own threshold")
    assert rc_lo < 1e-6, (
        f"at 0.9x the threshold the chirp-Z error is already {rc_lo:.3e}, "
        f"past the 1e-6 the threshold is named for; re-derive it")

    rc_hi, w_hi, rd_hi, wd_hi = _chirp_and_dense_at(_PHASE_BUDGET_MAX * 2.2)
    assert w_hi, "the guard did not fire above its threshold"
    assert rc_hi > 1e-6, (
        f"just above the threshold the chirp-Z error is {rc_hi:.3e}, still "
        f"inside 1e-6 -- the threshold would be firing early")

    # THE GUARD'S SCOPE, restated 2026-09-20 (round 2, D-1).  This used to
    # read `assert not wd_hi` with the reason "it has no chirp phase to lose"
    # and `assert rd_hi < 1e-14`.  The first is the shipped DECISION and
    # stays; the reason does not, and the second was the instrument.  What is
    # asserted now is the decision plus the gap it leaves, so the gap cannot
    # widen in silence: the dense route is quiet at 2.2x the threshold while
    # its OWN error there is past the 1e-6 the threshold is named for.
    # MEASURED 2026-09-20 on both builds: dense 1.079e-06 against chirp-Z
    # 2.028e-06, a factor of 1.880.  Widening the guard to the dense route is
    # a behaviour change owed to the maintainer and is an open item in
    # WAVE5_HYGIENE2_REPORT.md.
    assert not wd_hi, (
        "the dense route warned.  That may well be the better behaviour -- "
        "it is measured at 1.079e-06 at this budget -- but it is a "
        "behaviour change, and this id is what makes it a decision rather "
        "than a drift")
    assert 1e-8 < rd_hi < rc_hi, (
        f"at 2.2x the threshold the dense route reads {rd_hi:.3e} against "
        f"the chirp-Z route's {rc_hi:.3e}.  Below 1e-8 it would be back at "
        f"the summation floor (3.2e-16, the reading a reference sharing its "
        f"phase gives); at or above rc_hi it would have stopped being the "
        f"more accurate route.  MEASURED 1.079e-06, factor "
        f"{rc_hi / rd_hi:.3f}")
    assert rd_lo < rc_lo, (
        f"below the threshold the dense route ({rd_lo:.3e}) is no longer "
        f"the more accurate one ({rc_lo:.3e})")


def test_no_shipped_mft_grid_comes_near_the_phase_budget():
    """The other side of lowering a threshold: who does it make noisy?

    The budget a public MFT call spends is ``alpha * N_max^2`` with
    ``alpha = dx * dx_out / (lambda z)``, which the caller's own numbers give
    without instrumenting anything.  For the natural focal-zoom grids these
    propagators are written for, ``alpha ~ zoom/N`` and the budget is of order
    ``zoom * N``.  Every fixture below is asserted to sit at least three
    decades under the threshold AND to run without a RuntimeWarning, with
    warnings promoted to errors so a silent one cannot slip through.
    """
    import warnings as _w
    from lumenairy.propagators._bluestein import _PHASE_BUDGET_MAX
    from lumenairy.propagators.mft import (angular_spectrum_propagate_mft,
                                           fraunhofer_propagate_mft,
                                           fresnel_propagate_mft)
    wl = 633e-9
    cases = [
        # (N, dx, z, zoom, N_out)
        (256, 8e-6, 0.05, 10.0, 128),
        (512, 4e-6, 0.20, 4.0, 256),
        (1024, 2e-6, 0.01, 20.0, 128),
        (128, 20e-6, 1.00, 2.0, 128),
    ]
    for N, dx, z, zoom, N_out in cases:
        dx_out = (wl * z / (N * dx)) / zoom
        budget = (dx * dx_out / (wl * z)) * float(max(N, N_out)) ** 2
        assert budget < _PHASE_BUDGET_MAX / 1e3, (
            f"a natural MFT grid (N={N}, dx={dx:.1e}, z={z}, zoom={zoom}) "
            f"spends a phase budget of {budget:.3e}, within three decades of "
            f"the {_PHASE_BUDGET_MAX:.3e} threshold -- lowering the threshold "
            f"would make an ordinary caller noisy")
        E = _gauss(N, dx, 12.0 * dx)
        for fn in (fresnel_propagate_mft, fraunhofer_propagate_mft,
                   angular_spectrum_propagate_mft):
            with _w.catch_warnings():
                _w.simplefilter('ignore')
                _w.simplefilter('error', RuntimeWarning)
                fn(E, z, wl, dx, dx_out, N_out)


def test_the_chirp_phase_guard_fires_on_the_chirp_route_and_not_the_dense_one():
    """The phase-budget warning is SCOPED to the chirp signals' float64
    phase, which the dense route does not build.

    Two-sided: the chirp route warns and the dense route does not.  That is
    the shipped decision and this id is what keeps it one.

    WHAT THE SECOND HALF IS AND IS NOT (restated 2026-09-20,
    VERIFY-WAVE5-HYGIENE2 round 2, D-1).  ``max|dense - ref|`` here is
    measured against :func:`_pairwise_reference`, which forms its phase with
    the same two float64 roundings the dense route commits.  So this arm is a
    SUMMATION check -- the dense route's two BLAS products against a pairwise
    sum, inside the derived ``(g_dense + g_pair)*eps*sum|E|`` bar -- and NOT
    an accuracy claim about the phase.  It used to be described as "the dense
    route stays at the summation floor", which read as the latter.  The
    dense route's own phase error at this budget is enormous (the law is
    ``rel ~ eps * budget`` for BOTH routes, and this budget is 1e17); the
    exact-reference id above is where that is measured.
    """
    N, M = 24, 12
    E = _rand(N, N, seed=77)
    alpha = 1e17 / float(N) ** 2
    ref = _pairwise_reference(E, alpha, alpha, M, M, -1)
    _, bar_dense, _ = _bars(E, M, M)

    with pytest.warns(RuntimeWarning, match="chirp phase argument"):
        chirp = _bluestein_2d(E, alpha, alpha, M, M, sign=-1, xp=np,
                              fft2=_fft2, ifft2=_ifft2)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        dense = _bluestein_2d(E, alpha, alpha, M, M, sign=-1, xp=np,
                              fft2=_fft2, ifft2=_ifft2, method='direct')
    assert float(np.max(np.abs(dense - ref))) < bar_dense
    rel_chirp = float(np.linalg.norm(chirp - ref) / np.linalg.norm(ref))
    assert rel_chirp > 0.1, (
        f"the chirp route survived a 1e17 phase budget (rel {rel_chirp:.3e}); "
        f"the premise of this comparison is gone -- re-derive the budget")


# ===========================================================================
# 4.  The dense route's own structure
# ===========================================================================

def test_the_association_order_is_a_function_of_the_shapes_alone():
    """Two products, taken in whichever order costs fewer multiply-adds.

    Two claims, because either alone is weak:

    1. DETERMINISM -- the same call associates the same way every time, so the
       same bits come back.  (The rule reads only the four grid sizes, so this
       is structural, not luck.)
    2. The rule is the one the docstring states -- the CHEAPER order.  Checked
       on a RECTANGULAR shape where the two costs differ (on a square
       ``Ny=Nx``, ``My=Mx`` grid they are algebraically equal, so a square
       shape cannot tell the two rules apart and is no test at all), by
       building both associations here and asserting the function returns the
       cheaper one bit for bit.
    """
    E = _rand(48, 12)
    my, mx = 40, 6
    ny, nx = E.shape
    runs = [_direct_matrix_2d(E, 0.013, 0.011, my, mx, sign=-1, xp=np)
            for _ in range(3)]
    for r in runs[1:]:
        assert np.array_equal(_bits(runs[0]), _bits(r))

    cost_y_first = my * ny * nx + my * nx * mx
    cost_x_first = ny * nx * mx + my * ny * mx
    assert cost_y_first != cost_x_first, (
        "this shape cannot discriminate the two association rules; pick "
        "another")

    def _kernel(alpha, n_in, n_out, c_in, c_out):
        n = np.arange(int(n_in), dtype=np.float64) - float(c_in)
        k = np.arange(int(n_out), dtype=np.float64) - float(c_out)
        t = float(alpha) * k[:, None] * n[None, :]
        return np.exp(1j * -1 * 2.0 * np.pi * (t - np.rint(t)))

    # the primitive's signature is (E, alpha_x, alpha_y, N_out_y, N_out_x),
    # so the call above passes alpha_x = 0.013 and alpha_y = 0.011
    Wx = _kernel(0.013, nx, mx, 0.0, 0.0)
    Wy = _kernel(0.011, ny, my, 0.0, 0.0)
    y_first = (Wy @ E) @ Wx.T
    x_first = Wy @ (E @ Wx.T)
    cheaper = y_first if cost_y_first <= cost_x_first else x_first
    dearer = x_first if cost_y_first <= cost_x_first else y_first
    assert np.array_equal(_bits(runs[0]), _bits(cheaper)), (
        "the dense route did not take the cheaper association")
    # and the two orders really are different bits, so the claim above is not
    # vacuous on this fixture
    assert not np.array_equal(_bits(cheaper), _bits(dearer)), (
        "the two associations agree bit for bit here, so asserting which one "
        "was taken proves nothing -- pick a shape where they do not")


def test_the_phase_reduction_is_exact_so_a_huge_index_product_is_not_lost():
    """``t - rint(t)`` is exact for ``|t| <= 2**52``: the REDUCTION loses
    nothing.

    Checked on the arithmetic itself rather than through a transform: the
    reduced value must be EXACTLY representable, i.e. adding back the integer
    returns the original bit for bit.

    WHAT THIS DOES NOT SAY (restated 2026-09-20, round 2 D-1).  An earlier
    wording added "that is what lets the dense route carry a phase budget the
    chirp route cannot".  It does not: the reduction is exact, but the
    float64 PRODUCT ``alpha*(n - cI)*(k - cO)`` handed to it has already
    thrown away the low bits of a value needing ~63 of them, and an exact
    reduction of an inexact number is still inexact.  What the exactness of
    this step buys is that the dense route's error is ``eps*|t|`` and not
    more.
    """
    rng = np.random.default_rng(11)
    t = rng.uniform(-1e12, 1e12, size=4096)
    r = t - np.rint(t)
    assert np.all(np.abs(r) <= 0.5)
    assert np.array_equal(r + np.rint(t), t)


def test_the_dense_route_honours_the_caller_s_complex_dtype():
    """No silent upcast (the v4.14.1 contract), on the new route too."""
    E = _rand(16, 16).astype(np.complex64)
    out = _bluestein_2d(E, 0.01, 0.01, 8, 8, sign=-1, xp=np, fft2=_fft2,
                        ifft2=_ifft2, method='direct')
    assert out.dtype == np.complex64
    out64 = _bluestein_2d(_rand(16, 16), 0.01, 0.01, 8, 8, sign=-1, xp=np,
                          fft2=_fft2, ifft2=_ifft2, method='direct')
    assert out64.dtype == np.complex128


def test_the_dense_route_rejects_the_same_bad_inputs_as_the_chirp_route():
    """A route that accepts what its sibling refuses is a hole in the guard,
    not a feature."""
    E = _rand(8, 8)
    for kw in (dict(sign=0), dict(sign=2)):
        with pytest.raises(ValueError, match="sign must be"):
            _direct_matrix_2d(E, 0.01, 0.01, 4, 4, xp=np, **kw)
    with pytest.raises(ValueError, match="N_out must be positive"):
        _direct_matrix_2d(E, 0.01, 0.01, 0, 4, sign=-1, xp=np)


def test_the_dense_route_uses_no_fft_at_all():
    """The structural property behind a capability the timing table does not
    show: the dense route needs no transform, so it runs wherever ``xp.matmul``
    does -- including on a box whose FFT library is broken.

    MEASURED 2026-09-19 on this box, whose cuFFT DLL is broken (the reason
    ``tests/unit/test_niche_k2_carrier_backends.py`` skips its propagating CuPy
    arms): ``_direct_matrix_2d`` with ``xp=cupy`` returns complex128 agreeing
    with the NumPy route to 3.79e-16 relative.  That is recorded in the report
    rather than asserted here, because asserting it would mean skipping on a
    resource check.  What IS asserted is the property that makes it possible,
    which every box can check.
    """
    import inspect
    sig = inspect.signature(_direct_matrix_2d)
    assert 'fft2' not in sig.parameters and 'ifft2' not in sig.parameters, (
        "the dense route took an FFT callable; it is supposed to need none")
    src = inspect.getsource(_direct_matrix_2d)
    body = src.split('"""')[-1]            # past the docstring
    for needle in ('fft', 'next_fast_len', 'pad('):
        assert needle not in body, (
            f"the dense route's body mentions {needle!r}; it is supposed to be "
            f"two matrix products and nothing else")
    # four occurrences: two products in each of the two association
    # arms, of which exactly one arm runs per call
    assert body.count('matmul') == 4


def test_the_dense_route_is_cheaper_in_memory_at_the_shapes_it_is_for():
    """The MEMORY claim, as an ordering rather than a byte count.

    The byte counts move with the numpy version, the FFT backend and whether
    pyFFTW holds a plan; the ORDERING does not, because it follows from the
    padding: the chirp-Z route works on ``L = next_fast_len(N + M - 1)`` per
    axis and holds several ``L^2`` arrays, while the dense route holds two
    ``M x N`` kernels, one intermediate and the output, and pads nothing.

    Asserted on the two routes' own ANALYTIC counts, derived here from the
    same source-reading the probe uses, at the shapes the focal-zoom workflow
    actually runs (``M`` well below ``N``).  The measured ``tracemalloc``
    peaks are in ``validation/probe_wave5_hyg2/mft_direct_{win,wsl}.json``.
    """
    from scipy.fft import next_fast_len
    for (N, M) in ((256, 32), (512, 64), (1024, 128), (1024, 512)):
        L = int(next_fast_len(N + M - 1))
        chirp = 16 * (6 * L * L + N * N)
        dense = 16 * (2 * M * N + M * N + M * M) + 8 * M * N
        assert dense < chirp, (N, M, dense, chirp)
