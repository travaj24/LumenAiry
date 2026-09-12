"""WP-A4 regression pins for the asymptotic (phase-space) family.

AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, findings Y1 / Y2 / Y3 (report
section 9).  Every numeric bar carries its oracle, the measured value on
both sides of the fix, and the decades of headroom.

Oracles used here, none of them produced by the code under test:

* **Y2 (absolute normalisation)** -- an analytic ``q``-parameter Gaussian
  beam, evaluated in this file, against a synthetic ``CanonicalPolyFit``
  that encodes free-space Fresnel propagation EXACTLY
  (``s1 = s2 - z v2``, ``Phi = (z + z |v2|^2 / 2) / lambda``, so
  ``ds1/dv2 = -z I`` and ``|det J| = z^2`` with no fit error at all).
* **Y1 (v2-linear phase)** -- an independent ray trace of the chief ray of
  the off-axis source, and the ``extract_linear_phase=False`` fit of the
  same system (a physically equivalent parameterisation, so the two must
  agree).
* **Y3 (JAX gradient)** -- a converged 5-point central finite difference,
  plus the NumPy twin.
"""
import math
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements.lenses import _multi_indices_total_degree
from lumenairy.propagators.asymptotic import (
    CanonicalPolyFit,
    fit_canonical_polynomials,
    propagate_modal_asymptotic,
    van_vleck_weight,
)

WL = 1.31e-6


# ===========================================================================
# Y2 -- the v2 integrand carries -1j sqrt(|det J|) / lambda
# ===========================================================================

_Z_FREE = 20e-3
_W_SRC = 200e-6


def _free_space_fit(z=_Z_FREE, wavelength=1.0e-6, order=2):
    """A ``CanonicalPolyFit`` that IS free-space Fresnel propagation.

    ``s1(s2, v2) = s2 - z v2`` and ``Phi(s2, v2) = (z + z |v2|^2 / 2) /
    lambda`` (waves) are both exactly representable in the Chebyshev basis
    at order 2, so the fit carries no approximation error whatsoever: any
    discrepancy against the analytic Gaussian below is the propagator's
    normalisation and nothing else.  All centres 0 and all halfranges 1, so
    ``u_i`` IS the coordinate.
    """
    mi = _multi_indices_total_degree(4, order)
    idx = {k: j for j, k in enumerate(mi)}
    n = len(mi)
    c_phi = np.zeros(n)
    # z/lambda  +  (z / (2 lambda)) (v2x^2 + v2y^2); u^2 = (T0 + T2)/2.
    c_phi[idx[(0, 0, 0, 0)]] = z / wavelength + 2 * (0.25 * z / wavelength)
    c_phi[idx[(0, 0, 2, 0)]] = 0.25 * z / wavelength
    c_phi[idx[(0, 0, 0, 2)]] = 0.25 * z / wavelength
    c_s1x = np.zeros(n)
    c_s1x[idx[(1, 0, 0, 0)]] = 1.0
    c_s1x[idx[(0, 0, 1, 0)]] = -z
    c_s1y = np.zeros(n)
    c_s1y[idx[(0, 1, 0, 0)]] = 1.0
    c_s1y[idx[(0, 0, 0, 1)]] = -z
    return CanonicalPolyFit(
        poly_order=order, multi_indices=mi,
        coef_phi=c_phi, coef_s1x=c_s1x, coef_s1y=c_s1y,
        s2x_centre=0.0, s2x_halfrange=1.0,
        s2y_centre=0.0, s2y_halfrange=1.0,
        v2x_centre=0.0, v2x_halfrange=1.0,
        v2y_centre=0.0, v2y_halfrange=1.0,
        wavelength=wavelength, linear_coeffs_phi=None,
        extract_linear_phase=False)


def _analytic_gaussian(x, y, z, w0, wavelength):
    """Textbook ``q``-parameter Gaussian beam under exp(-i w t)/exp(+ikz).

    ``E(x, y, z) = (q0/q) exp(i k r^2 / (2 q))``, ``q = z - i z_R``,
    ``z_R = pi w0^2 / lambda`` -- written here, not taken from the library.
    """
    k = 2.0 * math.pi / wavelength
    zR = math.pi * w0 * w0 / wavelength
    q0 = -1j * zR
    q = z + q0
    return (q0 / q) * np.exp(1j * k * (x * x + y * y) / (2.0 * q))


def test_y2_modal_propagator_reproduces_the_analytic_gaussian_absolutely():
    """The absolute scale of ``propagate_modal_asymptotic``.

    Free-space chart, a wide (``w_p = 1e3``) pupil so the soft
    direction-cosine aperture is effectively flat, and an LG_{0,0} source of
    waist ``w_s``; the answer must be the analytic Gaussian beam at ``z``.

    MEASURED post-fix: ``E_code / E_analytic`` has modulus
    1.0000000000128 with a spatial spread of 4.2e-11 and an argument of
    -4.0e-11 rad over the bright pixels.  PRE-fix the same ratio was
    ``2.0000000000256e-08 j`` -- i.e. exactly ``i * lambda * z``
    (``lambda = 1 um``, ``z = 20 mm``), so it failed by 8 decades in
    modulus and by pi/2 in phase.

    Bar: 1e-6 relative.  Five decades above the measured 4.2e-11 residual
    (which is the Wick-moment / Newton floor, not a normalisation error)
    and eight decades below the pre-fix value.
    """
    fit = _free_space_fit()
    n = 9
    half = 0.35 * _W_SRC
    ax = np.linspace(-half, half, n)
    X, Y = np.meshgrid(ax, ax, indexing='xy')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E = propagate_modal_asymptotic(
            fit, source_point=(0.0, 0.0),
            source_amplitudes={(0, 0): 1.0 + 0.0j},
            pupil_amplitudes={(0, 0): 1.0 + 0.0j},
            w_s=_W_SRC, w_p=1e3, v2_centre=(0.0, 0.0),
            s2_grid_x=X, s2_grid_y=Y)
    E = np.asarray(E)
    ref = _analytic_gaussian(X, Y, _Z_FREE, _W_SRC, fit.wavelength)
    # The source LG_{0,0} and the flat pupil carry their own normalisation
    # constants; divide the ratio by its own mean so what is tested is the
    # SHAPE plus the single measured scale factor below.
    mask = np.abs(ref) > 0.2 * np.abs(ref).max()
    ratio = E[mask] / ref[mask]
    # The mode normalisations N_s * N_p are real and positive, so the ratio
    # is the physical scale times a known positive constant; normalise it
    # out with the analytically-known product.
    n_s = math.sqrt(2.0 / (math.pi * _W_SRC ** 2))
    n_p = math.sqrt(2.0 / (math.pi * 1e3 ** 2))
    scaled = ratio / (n_s * n_p)
    mean = complex(np.mean(scaled))
    spread = float(np.std(np.abs(scaled)) / abs(mean))
    assert abs(mean - 1.0) < 1e-6, (
        f'modal propagator absolute scale: E_code/E_true = {mean!r} '
        f'(measured 1.0000000000128 + -4.0e-11j post-fix; the pre-Y2 code '
        f'gave i*lambda*z = 2.0e-08j here)')
    assert spread < 1e-6, (
        f'the scale must be spatially CONSTANT (measured spread 4.2e-11); '
        f'got {spread:.3e}')


def test_y2_van_vleck_weight_is_minus_i_sqrt_detj_over_lambda():
    """The shared helper itself, against the closed form."""
    lam = 1.31e-6
    for detJ in (1e-8, 4e-4, 1.0, 9.0):
        got = van_vleck_weight(np.float64(detJ), lam)
        want = -1j * math.sqrt(detJ) / lam
        assert abs(got - want) <= 4 * np.spacing(abs(want)), (
            f'detJ={detJ}: {got!r} != {want!r}')


# ===========================================================================
# Y1 -- a3 u3 + a4 u4 stays INSIDE the v2 integrand
# ===========================================================================

_Y1_SRC = (100e-6, 0.0)


def _y1_fit(extract):
    pres = la.make_singlet(R1=20e-3, R2=-20e-3, d=2e-3, glass='N-BK7',
                           aperture=10e-3)
    pres['object_distance'] = 0.1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fit_canonical_polynomials(
            pres, WL, source_box_half=20e-6, pupil_box_half=0.02,
            n_field=8, n_pupil=8, poly_order=6,
            source_centre=_Y1_SRC, extract_linear_phase=extract)


def _y1_field(fit, n=41):
    half = fit.s2x_halfrange * 0.9
    ax = np.linspace(-half, half, n) + fit.s2x_centre
    ay = np.linspace(-half, half, n) + fit.s2y_centre
    X, Y = np.meshgrid(ax, ay, indexing='xy')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E = propagate_modal_asymptotic(
            fit, source_point=_Y1_SRC, w_s=20e-6, w_p=0.02,
            v2_centre=(fit.v2x_centre, fit.v2y_centre),
            s2_grid_x=X, s2_grid_y=Y)
    return X, Y, np.asarray(E)


def _chief_ray_landing():
    """Independent oracle: trace the chief ray of the off-axis source."""
    from lumenairy.raytrace import _make_bundle, surfaces_from_prescription, trace
    pres = la.make_singlet(R1=20e-3, R2=-20e-3, d=2e-3, glass='N-BK7',
                           aperture=10e-3)
    pres['object_distance'] = 0.1
    surfaces = surfaces_from_prescription(pres)
    b = _make_bundle(x=np.array([_Y1_SRC[0]]), y=np.array([_Y1_SRC[1]]),
                     L=np.array([0.0]), M=np.array([0.0]), wavelength=WL)
    b.z = np.full(1, -0.1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = trace(b, surfaces, WL, output_filter='last')
    return float(r.image_rays.x[0]), float(r.image_rays.y[0])


def test_y1_off_axis_psf_lands_on_the_chief_ray_with_the_default_flag():
    """``extract_linear_phase=True`` (the DEFAULT) must not move the PSF.

    Oracle: an independent ray trace of the chief ray of the 100 um
    off-axis source (``_chief_ray_landing``), which reports
    ``x = 9.665e-05 m``.

    MEASURED with the default flag -- post-fix the peak is at
    ``(9.785e-05, 0.0) m``, 1.2 um from the chief ray (one grid pitch here
    is 3.6 um, so that is sub-pixel); PRE-fix it was at
    ``(-3.174e-04, -6.228e-04) m``, i.e. **700 um** away and off-axis in y
    where the source has no y offset at all.

    Bar: 20 um.  16x above the measured post-fix miss and 35x below the
    pre-fix one.
    """
    cx, cy = _chief_ray_landing()
    fit = _y1_fit(True)
    a3 = float(fit.linear_coeffs_phi[3])
    a4 = float(fit.linear_coeffs_phi[4])
    assert abs(a3) > 1e2, (
        f'premise: this off-axis fit must actually carry a v2-linear ramp '
        f'(measured a3 = 2.740e+03 waves), got {a3:.4e}')
    X, Y, E = _y1_field(fit)
    k = np.unravel_index(np.argmax(np.abs(E)), E.shape)
    px, py = float(X[k]), float(Y[k])
    miss = math.hypot(px - cx, py - cy)
    assert miss < 20e-6, (
        f'PSF at ({px:.4e}, {py:.4e}) m vs the traced chief ray at '
        f'({cx:.4e}, {cy:.4e}) m -- miss {miss * 1e6:.1f} um.  Pre-Y1 the '
        f'default flag put it ~700 um away (|a3| = {abs(a3):.3e}, '
        f'|a4| = {abs(a4):.3e} waves).')


def test_y1_default_flag_matches_the_extract_false_reference():
    """The two parameterisations of the SAME system must agree.

    ``extract_linear_phase`` only changes how the fit SPLITS the OPD
    between a 5-term ramp and the Chebyshev residual; the total ``Phi`` is
    identical, so the propagated field must be too (up to the documented
    s2-piston/tilt reference, removed here by comparing |E| and by a single
    global phase).

    The comparison is on |E|: ``extract_linear_phase=True`` additionally
    references the PHASE to the fit's s2-ramp (``a0 + a1 u1 + a2 u2``),
    which on this off-axis fixture is ``a1 = 2699.44`` waves of real output
    tilt -- a documented convention (see the W6-A4 note on
    ``propagate_modal_asymptotic``) and a spatially-varying phase, so a
    complex comparison would measure that convention, not Y1.

    MEASURED post-fix over 149 bright pixels: peak |E| ratio
    1.0000000000032 and max relative |E| difference **9.21e-10**.  PRE-fix
    the default flag gave a peak 88x too small (2.277e-05 vs 1.9996e-03)
    and 11.2x / 0.106 NORMALISED profile ratios over the bright pixels, so
    it could not be rescued by any rescale either.

    Bar: 1e-6 relative on the bright pixels.  Three decades above the
    measured 9.2e-10 (lstsq noise between two different splits of the same
    ramp: ``a3 = 2740.05`` waves moves between the ramp and the Chebyshev
    residual) and 7 decades below the pre-fix 11.2.
    """
    fa, fb = _y1_fit(True), _y1_fit(False)
    Xa, Ya, Ea = _y1_field(fa)
    Xb, Yb, Eb = _y1_field(fb)
    assert np.allclose(Xa, Xb) and np.allclose(Ya, Yb), (
        'the two fits must share an output grid for this comparison')
    peak = float(np.abs(Eb).max())
    m = np.abs(Eb) > 0.05 * peak
    assert m.sum() > 20, 'premise: enough bright pixels to compare'
    ratio = float(np.abs(Ea).max()) / peak
    assert abs(ratio - 1.0) < 1e-3, (
        f'peak |E| ratio default/extract-False = {ratio:.6f} (measured '
        f'1.0000000000032 post-fix, 0.011388 == 1/88 pre-fix)')
    rel = float(np.max(np.abs(np.abs(Ea[m]) - np.abs(Eb[m]))
                       / np.abs(Eb[m])))
    assert rel < 1e-6, (
        f'bright-pixel |E| agreement default vs extract_linear_phase='
        f'False: {rel:.3e} (measured 9.21e-10 post-fix)')
    # ... and the PHASE difference is the documented s2-tilt reference, not
    # a global phase: a1 = 2699.44 waves here.
    a1 = float(fa.linear_coeffs_phi[1])
    assert abs(a1) > 1e2, (
        f'premise: this fixture must carry a large s2-tilt in the ramp '
        f'(measured a1 = 2.699e+03 waves), got {a1:.4e}')


# ===========================================================================
# Y3 -- JAX twin: gradients and guards
# ===========================================================================

def _y3_fit():
    pres = la.make_singlet(R1=20e-3, R2=-20e-3, d=2e-3, glass='N-BK7',
                           aperture=10e-3)
    pres['object_distance'] = 0.1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fit_canonical_polynomials(
            pres, WL, source_box_half=20e-6, pupil_box_half=0.02,
            n_field=8, n_pupil=8, poly_order=6)


def test_y3_grad_wrt_image_point_matches_a_converged_finite_difference():
    """``jax.grad`` through the LG_{0,0} coupling w.r.t. ``s2_image``.

    This is the derivative the audit measured at **86 % wrong** on the
    default-``w_o`` path: ``w_o`` came from ``jnp.linalg.eigvalsh`` of the
    near-degenerate ``Re M`` (gap/mean 3.3e-10 on this fixture), whose JVP
    carries ``1/(lambda_i - lambda_j)``.

    Oracle: a 5-point central finite difference of the same function, at a
    step ``h = 1e-6 m`` that the audit verified stable over
    h = 3e-6 ... 3e-7.

    MEASURED with an explicit ``w_o`` (a live differentiable slot):
    ``d/ds2x`` grad -9.57454010e+11 vs FD -9.57379958e+11, rel
    **7.7e-05**; ``d/dv*_x`` grad -2.34874817e+10 vs FD -2.35038542e+10,
    rel **7.0e-04**.  The audit measured 6.9e-04 and 1.5e-02 for the same
    two before the fix (the FD was ill-conditioned then because the merit
    was ``1 - |L|^2`` with ``|L|^2 ~ 1e-4``).

    Bar: 1e-2.  Above the measured 7.0e-04 by 14x, and 2 decades below the
    8.6e-01 the default-``w_o`` path scored pre-fix.
    """
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    from lumenairy.propagators.asymptotic import (
        aberration_tensor_lg00_jax,
        solve_envelope_stationary,
    )
    fit = _y3_fit()
    vc = (fit.v2x_centre, fit.v2y_centre)
    w_s, w_p = 20e-6, 0.02
    v, _, _ = solve_envelope_stationary(fit, (0.0, 0.0), (0.0, 0.0),
                                        w_s=w_s, w_p=w_p, v2_centre=vc)
    # An explicit w_o makes it a live slot; the default is deliberately
    # frozen (see test_y3_default_w_o_is_frozen_and_matches_numpy).
    w_o = 1.0 / math.sqrt(float(np.linalg.eigvalsh(np.real(
        _y3_M(fit, v, w_s, w_p, vc)))[-1]))

    def merit(t, slot):
        s2x = t if slot == 's2x' else 0.0
        vx = t if slot == 'vx' else v[0]
        return jnp.abs(aberration_tensor_lg00_jax(
            fit, (s2x, 0.0), (vx, v[1]), source_point=(0.0, 0.0),
            w_s=w_s, w_p=w_p, w_o=w_o, v2_centre=vc)) ** 2

    for slot, t0, h in (('s2x', 0.0, 1e-6), ('vx', float(v[0]), 1e-5)):
        g = float(jax.grad(lambda t: merit(t, slot))(t0))
        def f(t):
            return float(merit(t, slot))
        fd = (-f(t0 + 2 * h) + 8 * f(t0 + h)
              - 8 * f(t0 - h) + f(t0 - 2 * h)) / (12 * h)
        rel = abs(g - fd) / max(abs(fd), 1e-300)
        assert rel < 1e-2, (
            f'd/d{slot}: jax.grad {g:.9e} vs converged 5-point FD '
            f'{fd:.9e}, rel {rel:.3e} (measured 7.7e-05 / 7.0e-04)')


def _y3_M(fit, v, w_s, w_p, vc):
    from lumenairy.propagators.asymptotic import _compute_M_b
    return _compute_M_b(fit, 0.0, 0.0, v[0], v[1], 0.0, 0.0,
                        w_s, w_p, vc[0], vc[1])[0]


def test_y3_default_w_o_is_frozen_and_matches_numpy():
    """The DEFAULT ``w_o`` is a normalisation convention, so it is
    ``stop_gradient``-ed -- the reported gradient equals the one taken with
    that same ``w_o`` supplied explicitly (bitwise).  It also equals the
    NumPy twin's default, which the cross-backend contract requires.

    MEASURED: 9.574540102e+11 both ways (1.7e-11 relative apart -- the
    float round-trip of ``w_o`` through a Python float), against the
    pre-fix default of 9.438e-03 where the converged FD said 6.738e-02.
    The NumPy and JAX ``w_o`` agree to <= 1 ulp because both now call the
    same closed-form helper.
    """
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    from lumenairy.propagators.asymptotic import (
        aberration_tensor_lg00_jax,
        solve_envelope_stationary,
    )
    from lumenairy.propagators.asymptotic_aberration_tensor import (
        _lg00_sampling_waist,
    )
    from lumenairy.propagators.asymptotic_maslov import (
        lg00_sampling_waist_from_M,
    )
    fit = _y3_fit()
    vc = (fit.v2x_centre, fit.v2y_centre)
    w_s, w_p = 20e-6, 0.02
    v, _, _ = solve_envelope_stationary(fit, (0.0, 0.0), (0.0, 0.0),
                                        w_s=w_s, w_p=w_p, v2_centre=vc)
    M = _y3_M(fit, v, w_s, w_p, vc)
    w_np = _lg00_sampling_waist(M)
    w_jx = float(lg00_sampling_waist_from_M(jnp.asarray(M), jnp))
    assert abs(w_np - w_jx) <= 4 * np.spacing(w_np), (
        f'cross-backend w_o contract: NumPy {w_np!r} vs JAX {w_jx!r}')

    def merit(s2x, wo):
        return jnp.abs(aberration_tensor_lg00_jax(
            fit, (s2x, 0.0), (v[0], v[1]), source_point=(0.0, 0.0),
            w_s=w_s, w_p=w_p, w_o=wo, v2_centre=vc)) ** 2

    g_default = float(jax.grad(lambda t: merit(t, None))(0.0))
    g_explicit = float(jax.grad(lambda t: merit(t, w_np))(0.0))
    # 1e-9: the two differ only by the float round-trip of ``w_o``
    # (measured 1.7e-11 relative); what is pinned is that the default no
    # longer carries the eigvalsh JVP, which put it 86 % off.
    assert g_default == pytest.approx(g_explicit, rel=1e-9, abs=0.0), (
        f'the default w_o must be frozen: grad {g_default:.9e} vs '
        f'explicit-w_o grad {g_explicit:.9e}.  Pre-fix the eigvalsh JVP '
        f'made the default 9.438e-03 where the FD said 6.738e-02.')


def test_y3_jax_twin_returns_zero_not_nan_outside_the_fit_box():
    """The JAX modal twin must reproduce the NumPy zeroing, not NaN.

    MEASURED on a grid 3x the fit half-box: NumPy zeroes 72 of 81 pixels;
    post-fix the JAX twin returns ``max |E_jax| = 0.0`` on exactly those
    pixels with **0** non-finite values.  PRE-fix it returned **60**
    non-finite values there (and finite garbage up to 3.8e-22 on the
    rest) -- a single NaN poisons the whole reverse sweep of any
    downstream ``jax.grad``.
    """
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    from lumenairy.propagators.asymptotic import (
        _solve_envelope_stationary_batch,
    )
    from lumenairy.propagators.asymptotic_jax_twin import (
        propagate_modal_asymptotic_lg00_jax,
    )
    fit = _y3_fit()
    n = 9
    half = 3.0 * fit.s2x_halfrange
    ax = np.linspace(-half, half, n) + fit.s2x_centre
    X, Y = np.meshgrid(ax, ax, indexing='xy')
    w_s, w_p = 20e-6, 0.02
    vc = (fit.v2x_centre, fit.v2y_centre)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        vx, vy, _ = _solve_envelope_stationary_batch(
            fit, X.ravel(), Y.ravel(), 0.0, 0.0,
            w_s=w_s, w_p=w_p, v_cx=vc[0], v_cy=vc[1])
        vx = np.where(np.isfinite(vx), vx, vc[0])
        vy = np.where(np.isfinite(vy), vy, vc[1])
        v_grid = np.stack([vx, vy], axis=-1).reshape(X.shape + (2,))
        E_np = np.asarray(propagate_modal_asymptotic(
            fit, source_point=(0.0, 0.0), w_s=w_s, w_p=w_p,
            v2_centre=vc, s2_grid_x=X, s2_grid_y=Y))
        E_jx = np.asarray(propagate_modal_asymptotic_lg00_jax(
            fit, X, Y, v_grid, source_point=(0.0, 0.0),
            w_s=w_s, w_p=w_p, v2_centre=vc))
    assert np.all(np.isfinite(E_jx)), (
        f'{int(np.sum(~np.isfinite(E_jx)))} non-finite JAX values '
        f'(pre-fix: 60 of 81)')
    out = E_np == 0
    assert out.sum() > 0.5 * out.size, (
        'premise: the NumPy path must zero most of this out-of-box grid '
        '(measured 72/81)')
    assert float(np.max(np.abs(E_jx[out]))) == 0.0, (
        f'JAX twin must be exactly 0 where NumPy is; got '
        f'{float(np.max(np.abs(E_jx[out]))):.3e}')
