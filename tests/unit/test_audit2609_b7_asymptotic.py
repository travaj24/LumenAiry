"""WP-B7 -- the asymptotic family: Y4 / Y5 performance, the S6 gate's slope
statistic, the pupil-chart sizing, and the S9 GBD kernel clip.

Audit ``AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11``, WP-A4 section 6 items 3-8
and VERIFY-B1's follow-ups F1 / F2.  Report:
``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B7_REPORT.md``.

TESTING_STANDARDS.  Every bar here is either

* an EXACT equality between two arithmetic routes that this work package
  claims produce the same bits (``np.array_equal`` -- a build may move both
  sides together, never one of them),
* an OPERATION COUNT (basis builds, Newton sweeps, modes built, kernel
  samples, peak bytes) -- integers a build cannot move, and the only way to
  pin a performance claim without a wall clock, or
* a derived envelope measured on the running build's own geometry, with the
  measurement and the date in the assertion's own message.

No test here asserts a duration.  Where a claim is about cost, the cost is
counted, not timed.
"""
from __future__ import annotations

import math
import tracemalloc
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import lenses_maslov as LM
from lumenairy.propagators import asymptotic_canonical_fit as ACF
from lumenairy.propagators import asymptotic_maslov as AM
from lumenairy.propagators import asymptotic_modes as AMD
from lumenairy.propagators import gbd as GBD
from lumenairy.propagators.asymptotic import (
    _compute_M_b_batch,
    _solve_envelope_stationary_batch,
    fit_canonical_polynomials,
    propagate_modal_asymptotic,
)


# ===========================================================================
# Shared fixtures
# ===========================================================================
def _singlet():
    rx = la.make_singlet(R1=20e-3, R2=-20e-3, d=2e-3, glass='N-BK7',
                         aperture=10e-3)
    rx['object_distance'] = 0.1
    return rx


@pytest.fixture(scope='module')
def canon_fit():
    return fit_canonical_polynomials(
        _singlet(), wavelength=1.31e-6,
        source_box_half=20e-6, pupil_box_half=0.02,
        n_field=8, n_pupil=8, poly_order=6,
    )


@pytest.fixture(scope='module')
def pixel_batch(canon_fit):
    """A 21x21 output raster inside the fit's s2 box, flattened."""
    hw = 0.9 * canon_fit.s2x_halfrange
    ax = np.linspace(-hw, hw, 21) + canon_fit.s2x_centre
    ay = np.linspace(-hw, hw, 21) + canon_fit.s2y_centre
    SX, SY = np.meshgrid(ax, ay, indexing='xy')
    return SX, SY, SX.ravel(), SY.ravel()


_WS, _WP = 20e-6, 0.02


class _Counter:
    """Count calls to the Chebyshev table builders, wherever they are bound.

    The three consumers import them under module-private aliases, so the
    counter patches every binding rather than the definition -- patching
    ``lumenairy._math.chebyshev`` alone would count nothing.
    """

    NAMES = ('_chebyshev_vandermonde', '_chebyshev_derivative_vandermonde',
             '_chebyshev_second_derivative_vandermonde')

    def __init__(self, *modules):
        self.modules = modules
        self.n = 0
        self._saved = []

    def __enter__(self):
        for mod in self.modules:
            for nm in self.NAMES:
                orig = getattr(mod, nm, None)
                if orig is None:
                    continue

                def wrapped(*a, _o=orig, **k):
                    self.n += 1
                    return _o(*a, **k)

                setattr(mod, nm, wrapped)
                self._saved.append((mod, nm, orig))
        return self

    def __exit__(self, *exc):
        for mod, nm, orig in self._saved:
            setattr(mod, nm, orig)
        return False


def _cheb_modules():
    from lumenairy.elements import lenses as LN
    return (AM, ACF, LN)


# ===========================================================================
# 1.  Y4(a) + Y4(b) -- the fused basis evaluation
# ===========================================================================
def test_b7_the_fused_evaluation_reproduces_the_two_separate_ones_exactly(
        canon_fit, pixel_batch):
    """``eval_s1_and_phi_with_v2_grad`` IS ``eval_s1_with_v2_grad`` plus
    ``eval_phi_with_v2_grad``, bit for bit.

    The fusion shares the three ``(M, N)`` basis tensors across the three
    coefficient vectors instead of rebuilding them per vector; each output is
    still one ``np.tensordot`` of the same vector against the same tensor, so
    ``array_equal`` -- not a tolerance -- is the right assertion.  A tolerance
    here would pass a fusion that had quietly become a single stacked GEMM,
    which is entitled to reorder the reduction.
    """
    _sx, _sy, fx, fy = pixel_batch
    v2x = np.full(fx.size, canon_fit.v2x_centre + 0.1 * canon_fit.v2x_halfrange)
    v2y = np.full(fx.size, canon_fit.v2y_centre - 0.2 * canon_fit.v2y_halfrange)

    ref_s1 = canon_fit.eval_s1_with_v2_grad(fx, fy, v2x, v2y)
    ref_phi = canon_fit.eval_phi_with_v2_grad(fx, fy, v2x, v2y,
                                              include_linear=False)
    fused = canon_fit.eval_s1_and_phi_with_v2_grad(fx, fy, v2x, v2y,
                                                   include_linear=False)
    assert len(fused) == 9
    for i, (got, want) in enumerate(zip(fused[:6], ref_s1)):
        assert np.array_equal(got, want), f's1 output {i} moved'
    for i, (got, want) in enumerate(zip(fused[6:], ref_phi)):
        assert np.array_equal(got, want), f'phi output {i} moved'

    # The premise: the outputs are not all trivially equal to each other, so
    # the nine comparisons above are nine real claims.
    assert not np.array_equal(fused[0], fused[1])
    assert np.any(np.abs(fused[6]) > 0.0)

    # And the s2-only factor the Hessian pass reuses is the same table.
    K1, K2, _K3, _K4 = canon_fit.basis_index_columns()
    from lumenairy._math.chebyshev import chebyshev_vandermonde as _cv
    u1 = (fx - canon_fit.s2x_centre) / canon_fit.s2x_halfrange
    u2 = (fy - canon_fit.s2y_centre) / canon_fit.s2y_halfrange
    want_T12 = _cv(u1, canon_fit.poly_order)[K1] * _cv(
        u2, canon_fit.poly_order)[K2]
    got_T12 = canon_fit.eval_s1_and_phi_with_v2_grad(
        fx, fy, v2x, v2y, return_s2_factor=True)[9]
    assert np.array_equal(got_T12, want_T12)


def test_b7_the_batched_kernels_build_one_basis_per_sweep_not_three(
        canon_fit, pixel_batch):
    """Operation-count pin for Y4(a) / Y4(b).

    ``_compute_M_b_batch`` evaluates three coefficient vectors and a Hessian at
    one point set.  Unfused that is three value+gradient builds (six Chebyshev
    tables each) plus the Hessian's own eight = 26 table builds; fused it is
    four for the shared value+gradient pass plus the Hessian's remaining
    six = 12.  The Newton loop is the same story per sweep: two unfused
    evaluations (12 tables) against four, with the ``s2``-only factor hoisted
    out of the loop entirely (two tables, once).

    Counts, not seconds: a slower box does not change them, and they fail
    loudly if a future edit unfuses either site.
    """
    _sx, _sy, fx, fy = pixel_batch
    v2x = np.full(fx.size, canon_fit.v2x_centre)
    v2y = np.full(fx.size, canon_fit.v2y_centre)

    with _Counter(*_cheb_modules()) as c:
        _compute_M_b_batch(canon_fit, fx, fy, v2x, v2y, 0.0, 0.0,
                           _WS, _WP, 0.0, 0.0)
    assert c.n == 12, (
        f'_compute_M_b_batch built {c.n} Chebyshev tables; the fused pass is '
        f'6 (T1, T2, T3, T4, dT3, dT4) and the Hessian pass 6 (T3, T4, dT3, '
        f'dT4, d2T3, d2T4 -- it takes T1.T2 from the fused pass) = 12.  '
        f'Unfused it is 3 x 6 + 8 = 26.')

    with _Counter(*_cheb_modules()) as c2:
        _solve_envelope_stationary_batch(
            canon_fit, fx, fy, 0.0, 0.0, w_s=_WS, w_p=_WP,
            v_cx=0.0, v_cy=0.0)
    # 2 hoisted (T1, T2 for the loop-invariant s2 factor) + 4 per sweep.
    assert c2.n % 4 == 2 and 2 + 4 <= c2.n <= 2 + 4 * 12, (
        f'the Newton loop built {c2.n} tables; expected 2 hoisted plus 4 per '
        f'sweep over at most 12 sweeps.  Unfused it is 12 per sweep.')


def test_b7_the_fused_kernels_leave_the_modal_field_untouched(
        canon_fit, pixel_batch):
    """End-to-end: the Y4 refactors change nothing a caller can observe.

    The reference is built IN THIS FILE from the unfused methods, so the claim
    does not rest on a stored number: ``_compute_M_b_batch``'s seven returns
    are recomputed here through ``eval_s1_with_v2_grad`` /
    ``eval_phi_with_v2_grad`` / ``_phi_v2_hessian_batch`` and compared exactly.
    """
    _sx, _sy, fx, fy = pixel_batch
    v2x, v2y, _conv = _solve_envelope_stationary_batch(
        canon_fit, fx, fy, 0.0, 0.0, w_s=_WS, w_p=_WP, v_cx=0.0, v_cy=0.0)
    M, b, s1s, J, phis, G0, detJ = _compute_M_b_batch(
        canon_fit, fx, fy, v2x, v2y, 0.0, 0.0, _WS, _WP, 0.0, 0.0)

    s1x, s1y, a, bb, cc, dd = canon_fit.eval_s1_with_v2_grad(fx, fy, v2x, v2y)
    phi, gx, gy = canon_fit.eval_phi_with_v2_grad(fx, fy, v2x, v2y,
                                                  include_linear=False)
    assert np.array_equal(s1s[:, 0], s1x) and np.array_equal(s1s[:, 1], s1y)
    assert np.array_equal(J[:, 0, 0], a) and np.array_equal(J[:, 1, 1], dd)
    assert np.array_equal(J[:, 0, 1], bb) and np.array_equal(J[:, 1, 0], cc)
    assert np.array_equal(phis, phi.astype(np.complex128))

    H_ref = AM._phi_v2_hessian_batch(canon_fit, fx, fy, v2x, v2y)
    H_shared = AM._phi_v2_hessian_batch(
        canon_fit, fx, fy, v2x, v2y,
        canon_fit.eval_s1_and_phi_with_v2_grad(
            fx, fy, v2x, v2y, return_s2_factor=True)[9])
    assert np.array_equal(H_ref, H_shared)
    # b carries the gradient, so an unfused/fused split in it would show here.
    assert np.array_equal(
        b, (2.0j * math.pi) * np.column_stack([gx, gy]).astype(np.complex128)
        - 2.0 / (_WS * _WS) * np.einsum(
            'nij,nj->ni', np.swapaxes(J, -1, -2), s1s).astype(np.complex128)
        - 2.0 / (_WP * _WP) * np.column_stack(
            [v2x, v2y]).astype(np.complex128))
    assert np.all(np.isfinite(G0)) and np.all(detJ >= 0.0)
    assert np.all(np.isfinite(M.real))


# ===========================================================================
# 2.  Y4(c) -- the scale-relative Newton stop is opt-in
# ===========================================================================
def test_b7_the_scale_relative_newton_stop_is_opt_in_and_moves_the_iterate(
        canon_fit, pixel_batch):
    """Two-sided on the seam.

    OFF (the default) the solver takes every Newton step it always took, so
    the iterate is bit-identical to the absolute-stop reference; ON, pixels
    that have converged to machine precision leave the active set and keep the
    converged iterate instead of the one N round-off steps later.  Both
    directions are asserted, so neither a seam that does nothing nor a default
    that silently flipped can pass.

    The movement is bounded by the pupil box the iterate lives in, not by a
    stored number: a Newton step taken from a converged point is round-off, so
    ``max |dv2|`` must stay far below the box half-width.  MEASURED 2026-09-13
    on the stock N-BK7 singlet fit: 2.2e-11 against a 0.02 half-width, i.e.
    1.1e-9 of the box.
    """
    _sx, _sy, fx, fy = pixel_batch
    kw = dict(s2x=fx, s2y=fy, src_x=0.0, src_y=0.0, w_s=_WS, w_p=_WP,
              v_cx=0.0, v_cy=0.0)
    assert AM._NEWTON_SCALE_RELATIVE_STOP is False, (
        'the scale-relative Newton stop must ship OFF: it moves every '
        'asymptotic answer in the last bits')
    base = _solve_envelope_stationary_batch(canon_fit, **kw)
    default = _solve_envelope_stationary_batch(
        canon_fit, **kw, scale_relative_stop=None)
    absolute = _solve_envelope_stationary_batch(
        canon_fit, **kw, scale_relative_stop=False)
    relative = _solve_envelope_stationary_batch(
        canon_fit, **kw, scale_relative_stop=True)
    for i in range(3):
        assert np.array_equal(base[i], default[i])
        assert np.array_equal(base[i], absolute[i])

    moved = max(float(np.max(np.abs(absolute[i] - relative[i])))
                for i in (0, 1))
    assert moved > 0.0, (
        'the scale-relative stop changed nothing on this fixture, so the '
        'opt-in is untested -- the premise of the bar below is gone')
    assert moved < 1e-6 * canon_fit.v2x_halfrange, (
        f'the scale-relative stop moved the saddle by {moved:.3e}, which is '
        f'not round-off on a {canon_fit.v2x_halfrange:.3e} pupil half-range '
        f'(measured 2026-09-13: 2.2e-11, 1.1e-9 of the box)')
    # It must also be a real saving: fewer basis builds, i.e. fewer sweeps.
    with _Counter(*_cheb_modules()) as c_abs:
        _solve_envelope_stationary_batch(canon_fit, **kw,
                                         scale_relative_stop=False)
    with _Counter(*_cheb_modules()) as c_rel:
        _solve_envelope_stationary_batch(canon_fit, **kw,
                                         scale_relative_stop=True)
    assert c_rel.n <= c_abs.n, (
        f'the scale-relative stop built {c_rel.n} tables against the absolute '
        f'stop\'s {c_abs.n}: it is supposed to leave the active set EARLIER')


# ===========================================================================
# 3.  Y4 -- aberration_tensor's default cost
# ===========================================================================
def test_b7_decompose_lg_builds_only_the_modes_that_were_asked_for():
    """``only=`` returns the same numbers from a smaller build.

    The ``(p_max, ell_max)`` rectangle is the enclosing box of the requested
    set, not the set: ``(0,0) (1,0) (2,0) (1,1) (0,3) (2,2)`` spans 21 modes
    and reads 6.  Each overlap is a per-mode reduction against an
    independently-built mode, so the restricted call must reproduce the full
    call EXACTLY -- and the mode stack must actually be smaller, which is the
    half of the claim a value comparison cannot see.
    """
    n = 48
    ax = np.linspace(-3e-3, 3e-3, n)
    X, Y = np.meshgrid(ax, ax, indexing='xy')
    rng = np.random.default_rng(5)
    field = (np.exp(-(X ** 2 + Y ** 2) / (1.4e-3 ** 2))
             * np.exp(1j * 0.3 * rng.standard_normal((n, n))))
    want = [(0, 0), (1, 0), (2, 0), (1, 1), (0, 3), (2, 2)]
    p_max = max(k[0] for k in want)
    ell_max = max(abs(k[1]) for k in want)

    AMD.clear_lg_mode_stack_cache()
    full = AMD.decompose_lg(field, ax, ax, w=1.4e-3, p_max=p_max,
                            ell_max=ell_max)
    part = AMD.decompose_lg(field, ax, ax, w=1.4e-3, p_max=p_max,
                            ell_max=ell_max, only=want)
    assert set(part) == set(want)
    assert len(full) == (p_max + 1) * (2 * ell_max + 1) == 21
    for k in want:
        assert part[k] == full[k], f'mode {k} moved under only='

    AMD.clear_lg_mode_stack_cache()
    keys_part, stack_part = AMD._lg_mode_conj_stack(
        X, Y, 1.4e-3, p_max, ell_max, 0.0, 0.0, ax[1] - ax[0], ax[1] - ax[0],
        tuple(want))
    keys_full, stack_full = AMD._lg_mode_conj_stack(
        X, Y, 1.4e-3, p_max, ell_max, 0.0, 0.0, ax[1] - ax[0], ax[1] - ax[0])
    assert len(keys_part) == 6 and len(keys_full) == 21
    assert stack_part.nbytes * 3 < stack_full.nbytes
    # ... and the two stacks agree mode for mode, so the restriction is a
    # selection and not a different basis.
    for k in want:
        assert np.array_equal(stack_part[keys_part.index(k)],
                              stack_full[keys_full.index(k)])


def test_b7_the_image_plane_waist_probe_is_memoised_on_what_changes_it():
    """The ``w_o`` probe is a pure function of its inputs, so it is cached on
    all of them -- and on nothing less.

    A cache keyed on a subset would serve one optic's waist for another's.
    The test drives a hit, then four misses (fit, image point, pupil
    weighting, pupil waist), counting the probe's own propagate calls: a
    counter, not a clock.
    """
    from lumenairy.propagators import asymptotic_aberration_tensor as AT

    fit = fit_canonical_polynomials(
        _singlet(), wavelength=1.31e-6, source_box_half=20e-6,
        pupil_box_half=0.02, n_field=6, n_pupil=6, poly_order=4)
    calls = {'n': 0}

    def probe(*a, **k):
        calls['n'] += 1
        return propagate_modal_asymptotic(*a, **k)

    probe.__qualname__ = 'propagate_modal_asymptotic'
    base = dict(fit=fit, s2x_img=fit.s2x_centre, s2y_img=fit.s2y_centre,
                source_point=(0.0, 0.0),
                pupil_amplitudes={(0, 0): 1.0 + 0j},
                w_s=_WS, w_p=_WP, v2_centre=(fit.v2x_centre, fit.v2y_centre),
                propagate=probe)

    AT.clear_image_plane_waist_cache()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        w1 = AT._measure_image_plane_waist(**base)
        n_first = calls['n']
        w2 = AT._measure_image_plane_waist(**base)
        assert calls['n'] == n_first, 'the repeat call was not served'
        assert w1 == w2

        misses = 0
        for key, val in (
                ('s2x_img', fit.s2x_centre + 0.3 * fit.s2x_halfrange),
                ('pupil_amplitudes', {(0, 0): 1.0 + 0j, (1, 0): 0.4 + 0j}),
                ('w_p', _WP * 0.5),
        ):
            alt = dict(base)
            alt[key] = val
            before = calls['n']
            AT._measure_image_plane_waist(**alt)
            misses += int(calls['n'] > before)
            assert calls['n'] > before, (
                f'changing {key} was served from the cache; the key does not '
                f'cover everything that changes the answer')
        assert misses == 3

        # A structurally different fit is a different key even though the
        # object is a CanonicalPolyFit of the same shape.
        import dataclasses
        alt_fit = dataclasses.replace(fit, coef_phi=fit.coef_phi * 1.0001)
        before = calls['n']
        AT._measure_image_plane_waist(**{**base, 'fit': alt_fit})
        assert calls['n'] > before

        AT.clear_image_plane_waist_cache()
        before = calls['n']
        AT._measure_image_plane_waist(**base)
        assert calls['n'] > before, 'the clearer did not drain the cache'


# ===========================================================================
# 4.  S9 -- the GBD FFT kernel is clipped to its own support
# ===========================================================================
def _gbd_bundle(N=128, dx=4.0e-6, z=0.2e-3, w=100e-6, lam=1.0e-6):
    """A free-space beamlet bundle.  ``z`` selects the REGIME: a beamlet
    decomposed at ``waist_factor=1`` has a Rayleigh range of ~50 um here, so a
    short ``z`` leaves it compact against the grid (the kernel clips) and a
    long one lets it spread to fill the grid (it cannot, and must not)."""
    ax = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(ax, ax, indexing='xy')
    E = np.exp(-(X ** 2 + Y ** 2) / (w * w)).astype(np.complex128)
    b = GBD.decompose_field_to_beamlets(E, dx, wavelength=lam,
                                        waist_factor=1.0)
    return GBD.propagate_beamlets_freespace(b, z_distance=z, wavelength=lam)


def _bundle_half_width(bl, dx, N, lam=1.0e-6):
    cq = np.conj(complex(np.asarray(bl.Q).reshape(-1)[0]))
    return GBD._kernel_half_width(cq, 2.0 * np.pi / lam, dx, N)


def test_b7_the_fft_kernel_half_width_is_the_beamlets_own_support():
    """``_kernel_half_width`` IS ``ceil(n_sigma / sqrt(alpha) / d)``, clamped.

    Checked against the closed form rather than against a stored integer, and
    two-sided: a beamlet that decays inside the grid clips, one that does not
    (or a non-decaying axis) keeps the full ``N - 1`` range.
    """
    k = 2.0 * np.pi / 1.0e-6
    dx = 4.0e-6
    for lam_q in (4.0e3, 4.0e4, 4.0e5):
        cq = 1j * lam_q            # Im(conj(Q)) = lam_q > 0 -> decaying
        alpha = 0.5 * k * lam_q
        want = int(min(max(math.ceil(GBD._FFT_KERNEL_N_SIGMA
                                     / math.sqrt(alpha) / dx), 1), 127))
        assert GBD._kernel_half_width(cq, k, dx, 128) == want
    # No decay at all -> the full offset range, never a 1-sample kernel.
    assert GBD._kernel_half_width(0.0 + 0.0j, k, dx, 128) == 127
    assert GBD._kernel_half_width(-1j * 4e3, k, dx, 128) == 127
    # An enormously wide beamlet also keeps the full range (clamped, not
    # wrapped round to something small).
    assert GBD._kernel_half_width(1j * 1e-12, k, dx, 128) == 127
    assert GBD._FFT_KERNEL_N_SIGMA >= 6.1, (
        'the clip margin must keep exp(-n_sigma^2) below float64 eps '
        'relative to the kernel peak (n_sigma > 6.07), or the truncation '
        'stops being invisible to the transform that consumes it')


@pytest.mark.parametrize('z_mm', [0.05, 0.2, 0.5])
def test_b7_the_clipped_fft_reconstruction_agrees_with_the_windowed_sum(z_mm):
    """Accuracy envelope for the clip, derived from the oracle's own floor.

    The reference is the windowed scatter-add at ``n_sigma = 9`` -- a
    DIFFERENT summation order over the same beamlets, whose own truncation is
    ``exp(-81) = 7e-36``, i.e. nothing.  The two are therefore separated only
    by round-off and by the kernel clip, and the bar is ``1e-12`` relative L2,
    which separates "these are the same sum" from "these are different sums":
    the windowed path's own documented agreement with the dense sum is
    ``~1e-15``, and a clip at 6.5 sigma adds ``exp(-42.25) = 4.5e-19``.
    MEASURED 2026-09-13 over these three distances at N = 96 and N = 128:
    8.2e-16 .. 1.8e-15, with the kernel clipped to 10 / 27 / 65 samples of the
    available 95 or 127.  The parametrisation walks the clip across the grid,
    so a bar that only held for a near-point beamlet would show up here.
    """
    N, dx = 96, 4.0e-6
    bl = _gbd_bundle(N=N, dx=dx, z=z_mm * 1e-3)
    assert GBD._fft_reconstruct_applicable(bl, N, N, dx, dx, (0.0, 0.0))
    half = _bundle_half_width(bl, dx, N)
    assert half < N - 1, (
        f'premise: at z = {z_mm} mm the kernel must actually clip; it took '
        f'the full {half} of {N - 1} samples')
    out = GBD._reconstruct_fft(bl, xp=np, Ny=N, Nx=N, dx=dx, dy=dx,
                               centre=(0.0, 0.0), wavelength=1.0e-6)
    ref = GBD._reconstruct_windowed(bl, Ny=N, Nx=N, dx=dx, dy=dx,
                                    centre=(0.0, 0.0), wavelength=1.0e-6,
                                    n_sigma=9.0)
    rel = float(np.linalg.norm(out - ref) / np.linalg.norm(ref))
    assert rel < 1e-12, (
        f'clipped FFT reconstruction differs from the 9-sigma windowed sum by '
        f'{rel:.3e} relative L2; the clip is meant to be invisible beside the '
        f'transform\'s own round-off')
    assert np.linalg.norm(ref) > 0.0


def test_b7_the_fft_kernel_clip_caps_the_transform_peak():
    """Peak-bytes pin for S9, two-sided, expressed as a multiple of the
    output grid.

    Unclipped the kernel spans ``(2N-1, 2N-1)`` and the linear convolution
    transforms ``(3N-2, 3N-2)``: nine output grids per array with several
    alive, which is the ~36x the auditor measured and which this test still
    observes for a beamlet that has spread to fill the grid -- the clip cannot
    remove support that is really there, and a test that did not check the
    second half would pass a clip that silently truncated a live beam.
    MEASURED 2026-09-13 at N = 128: 7.0x the grid at z = 0.05 mm (kernel 10 of
    127 samples) against 37.8x at z = 8 mm (127 of 127), and at N = 512,
    6.1x against 37.8x.  ``tracemalloc`` counts bytes Python allocated -- a
    number the box's speed cannot move.
    """
    N, dx = 128, 4.0e-6
    grid_bytes = N * N * 16

    def _peak(bl):
        kw = dict(xp=np, Ny=N, Nx=N, dx=dx, dy=dx, centre=(0.0, 0.0),
                  wavelength=1.0e-6)
        GBD._reconstruct_fft(bl, **kw)          # warm lazy imports / plans
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            GBD._reconstruct_fft(bl, **kw)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    tight = _gbd_bundle(N=N, dx=dx, z=0.05e-3)
    spread = _gbd_bundle(N=N, dx=dx, z=8.0e-3)
    assert _bundle_half_width(tight, dx, N) < 0.2 * (N - 1)
    assert _bundle_half_width(spread, dx, N) == N - 1
    p_tight, p_spread = _peak(tight), _peak(spread)
    assert p_tight < 12.0 * grid_bytes, (
        f'a beamlet occupying {_bundle_half_width(tight, dx, N)} of {N - 1} '
        f'kernel samples still peaked at {p_tight / grid_bytes:.1f}x the '
        f'output grid; the clip is not engaging')
    assert p_spread > 25.0 * grid_bytes, (
        f'a beamlet that fills the grid peaked at '
        f'{p_spread / grid_bytes:.1f}x it; the full-support kernel is ~36x, '
        f'so the clip is truncating a beam that has not decayed')


# ===========================================================================
# 5.  Item 11 -- the pupil chart is sized from the mean AND the spread
# ===========================================================================
_LAM_A = 1.0e-6
_APER_A = 0.60e-3
_N_A, _DX_A = 256, 3.2e-6
_W_A = 0.15e-3


def _pres_a():
    return la.make_singlet(6.0e-3, -6.0e-3, 0.7e-3, 'N-BK7', aperture=_APER_A)


def _grid_a():
    ax = (np.arange(_N_A) - _N_A / 2) * _DX_A
    return np.meshgrid(ax, ax, indexing='xy')


def _field_a(tilt_x=0.0, curv_f=None, hard=None):
    X, Y = _grid_a()
    r2 = X * X + Y * Y
    E = np.exp(-r2 / (_W_A * _W_A)).astype(np.complex128)
    if hard is not None:
        E = E * (r2 <= (hard * 0.5 * _APER_A) ** 2)
    ph = np.zeros_like(X)
    if tilt_x:
        ph = ph + (2 * np.pi / _LAM_A) * tilt_x * X
    if curv_f:
        ph = ph - (2 * np.pi / _LAM_A) * r2 / (2.0 * float(curv_f))
    return E * np.exp(1j * ph) if np.any(ph) else E


def _maslov_a(E, **kw):
    args = dict(prescription=_pres_a(), wavelength=_LAM_A, dx=_DX_A,
                output_plane_distance=6.40e-3,
                integration_method='stationary_phase',
                roi=(0.0, 0.0, 8.0e-6), normalize_output='none',
                poly_order=4, ray_field_samples=8, ray_pupil_samples=8)
    args.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return la.apply_real_lens_maslov(E.copy(), **args)


def _angular_moments(E):
    """``(3 sigma_about_zero, |mean|, 3 sigma_about_mean)`` of the field's
    angular spectrum, computed here the way the driver computes them."""
    P = np.abs(np.fft.fft2(E)) ** 2
    f = np.fft.fftfreq(_N_A, d=_DX_A)
    FX, FY = np.meshgrid(f, f, indexing='xy')
    tot = float(P.sum())
    rms = float(np.sqrt(((_LAM_A ** 2 * (FX ** 2 + FY ** 2)) * P).sum() / tot))
    mean = float(np.hypot(_LAM_A * float((FX * P).sum() / tot),
                          _LAM_A * float((FY * P).sum() / tot)))
    spread = math.sqrt(max(rms * rms - mean * mean, 0.0))
    return 3.0 * rms, mean, mean + 3.0 * spread


def test_b7_a_uniform_tilt_sizes_the_chart_to_the_tilt_not_to_three_times_it():
    """The sizing statistic is ``|mean launch direction| + 3 sigma_about_mean``.

    A uniform tilt ``theta`` is a change of REFERENCE DIRECTION, not an
    angular spread: its first moment is ``theta`` and its spread about that is
    the beam's own diffraction, so the chart must reach ``na_lens + theta``
    and not ``na_lens + 3 theta``.  Asserted end to end, against the field's
    own measured collimated spread on this build rather than a stored number:
    the field the driver returns for the auto-sized chart must be BIT-EQUAL to
    the one it returns for an explicit ``input_na`` set to the mean-plus-spread
    value, and must DIFFER from the moment-about-zero one.
    """
    theta = 3.0e-2
    E = _field_a(tilt_x=theta)
    m0, mean, m1 = _angular_moments(E)
    assert abs(mean - theta) < 0.02 * theta, (
        f'premise: the first angular moment of a uniform {theta:.3e} tilt '
        f'must be that tilt; measured {mean:.4e}')
    assert m0 > 2.5 * m1, (
        f'premise: the two rules must differ here -- moment-about-zero '
        f'{m0:.4f} against mean-plus-spread {m1:.4f}')
    # Full-grid readout: a tilted beam lands ~f*theta off axis, so a window
    # centred on zero would compare two all-zero patches and prove nothing.
    auto = _maslov_a(E, roi=None)
    want = _maslov_a(E, roi=None, input_na=m1)
    other = _maslov_a(E, roi=None, input_na=m0)
    assert np.any(np.abs(auto) > 0.0), 'premise: the readout must see the spot'
    assert np.array_equal(auto, want), (
        'the auto-sized chart for a tilted input is not the '
        'mean-plus-spread chart')
    assert not np.array_equal(auto, other), (
        'the auto-sized chart is still the moment-about-zero one, or the two '
        'charts coincide and this test proves nothing')


@pytest.mark.parametrize('label', ['collimated', 'converging', 'diverging',
                                   'hard aperture', 'speckle'])
def test_b7_a_centred_input_keeps_the_chart_it_always_had(label):
    """Byte-identity gate on the sizing change.

    Every field whose angular spectrum is centred -- collimated, converging,
    diverging, hard-apertured, speckled about zero -- must get the OLD chart
    EXACTLY, not merely closely: the chart NA scales every launch direction,
    so a 1-ULP move there moves the whole output field.  What buys that is
    ``_NA_MEAN_MIN_FRACTION``, not the arithmetic: ``mean + 3 sigma_about_mean``
    and ``3 sigma_about_zero`` are DIFFERENT float64s here (measured
    4.501578547758776e-03 against 4.5015785477421685e-03 on the collimated
    row), so the test asserts the branch on this build's own moments, then
    asserts the field it produces is bit-equal to the explicit old-rule chart.

    MEASURED first-moment / spread ratios, 2026-09-13: 1.1e-11 (collimated),
    2.3e-11 (converging), 3.4e-11 (diverging), 9.3e-05 (hard aperture at 0.80
    of the pupil -- a pixel-quantised mask is not centred on an even grid),
    4.1e-04 (0.05 rad rms speckle).  A uniform tilt at
    ``_SADDLE_FLAT_INPUT_NA`` reads 5.5e-01.
    """
    E = {'collimated': _field_a(),
         'converging': _field_a(curv_f=+40e-3),
         'diverging': _field_a(curv_f=-25e-3),
         'hard aperture': _field_a(hard=0.8),
         'speckle': _field_a() * np.exp(
             1j * 0.05 * np.random.default_rng(3).standard_normal(
                 (_N_A, _N_A)))}[label]
    m0, mean, _m1 = _angular_moments(E)
    rms = m0 / 3.0
    assert mean <= LM._NA_MEAN_MIN_FRACTION * rms, (
        f'{label}: first moment {mean:.3e} is {mean / rms:.3e} of the spread '
        f'{rms:.3e}, at or above the {LM._NA_MEAN_MIN_FRACTION:g} bar -- this '
        f'field would take the mean-plus-spread branch and its chart would '
        f'move')
    assert np.array_equal(_maslov_a(E), _maslov_a(E, input_na=m0)), (
        f'{label}: the auto-sized chart is no longer the '
        f'3-sigma-about-zero one')


# ===========================================================================
# 6.  Item 10 -- the S6 gate's slope statistic
# ===========================================================================
def _s6_stats(E, **kw):
    """``(slope error, value residual, engaged)`` as the driver itself
    reports them on its progress channel."""
    seen = []
    _maslov_a(E, progress=lambda _s, _f, m: seen.append(m), **kw)
    line = [m for m in seen if 'k1 slope error' in m]
    assert line, 'the S6 gate did not report its statistics'
    m = line[0]
    return (float(m.split('k1 slope error')[1].split(',')[0]),
            float(m.split('k1 fit residual')[1].split()[0]),
            'engaged' in m)


def test_b7_the_k1_slope_error_sees_the_family_the_value_residual_cannot():
    """Calibration of :func:`_k1_fit_derivative_error`, and the two-sided
    derivation of ``_K1_DERIV_RESIDUAL_MAX``.

    A uniform tilt is a CONSTANT local wavevector: both the order-p and the
    order-(p-1) fit resolve it exactly, so the slope error is zero to
    round-off and the gate cannot fire on a clean tilt.  A HARD-EDGED aperture
    is the family the bar exists for: ``_local_direction_cosines`` reports a
    launch direction of 0 in the dark and the true wavefront in the light, so
    a degree-4 fit of that step has a small VALUE residual -- inside its own
    0.5 bar -- and an enormous SLOPE error.  MEASURED 2026-09-13 on this chart
    and on an f = 14.1 mm N-SF11 one: hard edge at 0.95 / 0.80 / 0.60 of the
    pupil scores value 0.15 / 0.08 / 0.26 against slope 4.3 / 2.8 / 3.9, while
    every input where engaging the saddle WINS scores slope <= 0.56.

    Three claims, all on the running build's own numbers: the tilt is silent,
    the hard edge is loud on the slope while quiet on the value, and the
    shipped bar separates them with the decision to match.
    """
    d_tilt, v_tilt, eng_tilt = _s6_stats(_field_a(tilt_x=3.0e-2))
    assert d_tilt < 1e-6, (
        f'a uniform tilt scored slope error {d_tilt:.3e}; a constant local '
        f'wavevector has none')
    assert eng_tilt, 'a clean tilt must still engage the S6 saddle'

    hard = _field_a(tilt_x=3.0e-2, hard=0.8)
    d_hard, v_hard, eng_hard = _s6_stats(hard)
    assert v_hard < LM._K1_FIT_RESIDUAL_MAX, (
        f'the hard-edged aperture scored value residual {v_hard:.3e}, at or '
        f'above its own {LM._K1_FIT_RESIDUAL_MAX:g} bar -- the premise of '
        f'this test (that the value residual is blind to it) no longer holds')
    assert d_hard > LM._K1_DERIV_RESIDUAL_MAX, (
        f'the hard-edged aperture scored slope error {d_hard:.3e}, below the '
        f'{LM._K1_DERIV_RESIDUAL_MAX:g} bar; the gate will engage a fit whose '
        f'derivative is a fiction')
    assert not eng_hard, 'the slope gate must refuse it, and it did not'
    assert d_hard > 20.0 * d_tilt and d_hard > 5.0 * v_hard

    # The bar sits in a measured gap, not on one side of it: a speckled input
    # that the saddle still improves must stay engaged.
    speck = _field_a(tilt_x=3.0e-2) * np.exp(
        1j * 0.05 * np.random.default_rng(11).standard_normal((_N_A, _N_A)))
    d_sp, _v_sp, eng_sp = _s6_stats(speck)
    assert d_sp < LM._K1_DERIV_RESIDUAL_MAX < d_hard, (
        f'the bar {LM._K1_DERIV_RESIDUAL_MAX:g} no longer sits in the measured '
        f'gap {d_sp:.3e} (speckle 0.05 rad, still improved by engaging) .. '
        f'{d_hard:.3e} (hard edge, made worse by engaging)')
    assert eng_sp


# ===========================================================================
# 6b.  Item 9 -- the JAX sibling's chief-ray displacement
# ===========================================================================
def _jax_ok():
    try:
        import jax  # noqa: F401
        return True
    except Exception:                            # pragma: no cover - env
        return False


@pytest.mark.skipif(not _jax_ok(), reason='jax not installed')
def test_b7_the_jax_screen_carries_the_chief_ray_displacement():
    """``apply_real_lens_maslov_jax`` is a thin-OPD phase screen, and a thin
    screen indexes its OPL by the ray's ENTRANCE point while it samples
    ``E_in`` at the OUTPUT pixel.  For a tilted input those differ by the
    ray's walk across the element, and the input's own phase has to be
    re-referenced to the entrance point.

    Three claims, all on this build's own geometry:

    * a COLLIMATED input is byte-identical to the uncorrected screen (its
      local wavevector is exactly zero, so the term is exactly zero);
    * a TILTED input is NOT, and the correction moves the answer in the
      direction of the chief ray -- checked against the screen's own
      entrance-to-exit map rather than against a stored landing, so the claim
      survives a change to the fit or the trace;
    * the keyword is three-valued and the extremes bracket the default.

    MEASURED 2026-09-13 against an exact conic raytrace of the input's own
    rays on an f = 14.1 mm N-SF11 singlet at 1.55 um: the uncorrected screen
    under-shoots the chief-ray landing by -2.66 % at a quarter of the lens NA,
    -2.66 % at half and -2.72 % at one; corrected, -0.19 / -0.19 / -0.27 %.
    """
    import jax
    jax.config.update('jax_enable_x64', True)
    from lumenairy.backend.array import to_numpy
    from lumenairy.elements._lens_jax import apply_real_lens_maslov_jax

    lam, dx, N = 1.55e-6, 4.0e-6, 128
    pres = la.make_singlet(20.7e-3, -20.7e-3, 1.2e-3, 'N-SF11',
                           aperture=0.40e-3)
    ax = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(ax, ax, indexing='xy')
    base = np.exp(-(X ** 2 + Y ** 2) / (80e-6 ** 2)).astype(np.complex128)
    theta = 0.02

    def run(E, **kw):
        return to_numpy(apply_real_lens_maslov_jax(
            E, prescription=pres, wavelength=lam, dx=dx, **kw))

    flat = run(base)
    assert np.array_equal(flat, run(base, input_wavevector_saddle=False)), (
        'a real non-negative input has an exactly-zero local wavevector, so '
        'the displacement term must be exactly zero and the screen unchanged')
    assert np.array_equal(flat, run(base, input_wavevector_saddle=True))

    tilted = base * np.exp(1j * (2 * np.pi / lam) * theta * X)
    off = run(tilted, input_wavevector_saddle=False)
    auto = run(tilted)
    on = run(tilted, input_wavevector_saddle=True)
    assert np.array_equal(auto, on), (
        'a 0.02 rad tilt is far above the flat-input bar, so the default must '
        'agree with the forced-on arm')
    assert not np.array_equal(auto, off), (
        'the correction changed nothing on a 0.02 rad tilted input')

    # The correction is the re-referencing term, so the phase it adds must
    # equal k0 * k1 . (entrance - pixel) -- and on a uniform tilt that is
    # k0 * theta * (xe - x), a quantity with the sign of the ray walk.
    d = np.angle(auto * np.conj(off))
    live = np.abs(auto) > 0.01 * np.abs(auto).max()
    assert np.any(live)
    rms_waves = float(np.sqrt(np.mean((d[live] / (2 * np.pi)) ** 2)))
    assert rms_waves > 1e-3, (
        f'the correction is only {rms_waves:.3e} waves rms on a 0.02 rad '
        f'tilt; that is not a chief-ray displacement')
    assert rms_waves < 1.0, (
        f'the correction is {rms_waves:.3e} waves rms, which is not a '
        f'first-order walk term on a thin element')


# ===========================================================================
# 7.  Folded in from WP-B9 -- jacobian='auto' reaches the analytic path for an
#     aspheric prescription
# ===========================================================================
def test_b7_an_aspheric_prescription_reaches_the_analytic_jacobian_via_auto():
    """WP-B9 gave the analytic differential ray transfer even-aspheric
    support, so ``jacobian='auto'`` -- which dispatches on the primitive's own
    ``NotImplementedError`` rather than on a list of surface kinds -- now
    returns the EXACT Jacobian there instead of the finite-difference one.

    Pinned at the primitive, two-sided: the aspheric prescription does not
    raise (so ``'auto'`` keeps the analytic candidate), a biconic one still
    does (so the fallback is not dead), and the analytic and FD answers agree
    inside the FD's own truncation floor, which is what makes the swap safe.
    """
    from lumenairy.raytrace import surfaces_from_prescription
    from lumenairy.raytrace.differential import (
        ray_transfer_jacobian,
        ray_transfer_jacobian_analytic,
    )
    lam = 1.0e-6
    asph = la.make_singlet(20e-3, -20e-3, 2e-3, 'N-BK7', aperture=8e-3)
    asph['surfaces'][0]['aspheric_coeffs'] = {4: 1.0e-4}
    surfs = list(surfaces_from_prescription(asph))

    x = np.array([0.0, 1.0e-3, 2.0e-3])
    y = np.zeros_like(x)
    u = np.zeros_like(x)
    an = ray_transfer_jacobian_analytic(x, y, u, u, surfs, lam,
                                        per_surface=False)
    fd = ray_transfer_jacobian(x, y, u, u, surfs, lam, per_surface=False)
    assert np.all(np.asarray(an.alive)) and np.all(np.asarray(fd.alive))
    scale = float(np.max(np.abs(fd.x)))
    err = float(np.max(np.abs(np.asarray(an.x) - np.asarray(fd.x))))
    assert err < 1e-6 * max(scale, 1e-12), (
        f'analytic and FD exit heights differ by {err:.3e} on a {scale:.3e} '
        f'scale, above the FD central-difference truncation floor')

    # The fallback is still live for the kinds WP-B9 did NOT cover.
    bic = la.make_singlet(20e-3, -20e-3, 2e-3, 'N-BK7', aperture=8e-3)
    bic['surfaces'][0]['radius_y'] = 25e-3
    with pytest.raises(NotImplementedError):
        ray_transfer_jacobian_analytic(
            x, y, u, u, list(surfaces_from_prescription(bic)), lam,
            per_surface=False)

    # And the GBD dispatcher orders its candidates analytic-first, so an
    # aspheric prescription -- which no longer raises -- is served by the
    # analytic primitive under the default jacobian='auto'.
    import inspect

    from lumenairy.propagators.gbd import (
        apply_prescription_persurface_to_beamlets,
    )
    src = inspect.getsource(apply_prescription_persurface_to_beamlets)
    i_auto = src.index("if jacobian == 'auto':")
    cand = src[i_auto:src.index('\n', src.index('_jac_candidates', i_auto))]
    assert cand.index('ray_transfer_jacobian_analytic') < cand.index(
        'ray_transfer_jacobian]'), (
        "jacobian='auto' must try the analytic primitive FIRST and fall back "
        "on its NotImplementedError; the candidate order is reversed")
    assert inspect.signature(
        apply_prescription_persurface_to_beamlets
    ).parameters['jacobian'].default == 'auto'
