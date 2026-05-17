"""Correctness pins for the v4.14.0 audit (Agent 2 scope).

Two perf wins:

* **2A** :func:`lumenairy.elements.coatings.coating_reflectance` now
  builds the per-layer characteristic-matrix stack in a single
  vectorised (n_wv, n_layers, 2, 2) pass instead of rebuilding it
  inside an outer Python loop over wavelengths.  Snell's real-only
  angle chain is wavelength-independent in the documented
  approximation (n.imag dropped, lambda absent), so it is walked
  exactly once.  The tournament reduction now collapses over the
  layer axis with batched np.matmul.  Test pins the batched output
  against an inline scalar-wavelength reference at atol=1e-12 for
  R, T, and phase_r on a 50-layer AR stack across 200 wavelengths;
  also pins scalar-wavelength input -> scalar output (back-compat
  shape contract).

* **2B** :func:`lumenairy.elements.lenses._evaluate_polynomial_4d_and_grad34`
  is now vectorised over basis terms via three ``np.tensordot``
  calls, mirroring the sibling :func:`_evaluate_polynomial_4d`.
  The Python loop with the ``if c == 0`` early-exit dominated the
  apply_real_lens_maslov Newton iteration cost at M=70 basis terms.
  Test pins (f, df3, df4) against a scalar-loop reference at 1e-14
  on a 30-term polynomial evaluated on a 16x16x16x16 grid, and
  exercises a real Maslov Newton inversion to confirm the
  optimisation result is bit-near-exact vs the pre-perf scalar
  loop.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from lumenairy.elements.coatings import coating_reflectance
from lumenairy.elements.lenses import (
    _chebyshev_vandermonde,
    _chebyshev_derivative_vandermonde,
    _multi_indices_total_degree,
    _evaluate_polynomial_4d_and_grad34,
)


# ---------------------------------------------------------------------------
# 2A: coating_reflectance wavelength-batch correctness pin
# ---------------------------------------------------------------------------


def _scalar_loop_coating_reference(layers, wavelengths, angle=0.0,
                                     n_substrate=1.52, n_ambient=1.0,
                                     polarization='avg'):
    """Pre-v4.14.0 scalar-wavelength characteristic-matrix walker.

    Inlines the exact algorithm used before the wavelength batch axis
    was added, including the TIR cap at 0.9999, the n.imag-dropped
    Snell step, and the Macleod p-pol sign convention.  This is the
    pin reference for the 50-layer x 200-wavelength batch test.
    """
    wavelengths = np.atleast_1d(np.asarray(wavelengths, dtype=np.float64))
    n_wv = wavelengths.size
    R = np.empty(n_wv)
    T = np.empty(n_wv)
    phase_r = np.empty(n_wv)
    pols = ['s', 'p'] if polarization == 'avg' else [polarization]
    for iw, lam in enumerate(wavelengths):
        rs, ts = [], []
        eta_sub_by_pol, eta_amb_by_pol = {}, {}
        for pol in pols:
            M = np.eye(2, dtype=np.complex128)
            theta_prev = angle
            n_prev = complex(n_ambient)
            for n_layer, d in layers:
                n_layer = complex(n_layer)
                sin_t = n_prev.real * math.sin(theta_prev) / n_layer.real
                sin_t = min(sin_t, 0.9999)
                cos_t = math.sqrt(1 - sin_t * sin_t)
                delta = 2 * math.pi * n_layer * d * cos_t / lam
                if pol == 's':
                    eta = n_layer * cos_t
                else:
                    eta = n_layer / cos_t
                Mj = np.array([
                    [np.cos(delta), -1j * np.sin(delta) / eta],
                    [-1j * eta * np.sin(delta), np.cos(delta)],
                ], dtype=np.complex128)
                M = M @ Mj
                theta_prev = math.asin(sin_t)
                n_prev = n_layer
            sin_sub = (n_prev.real * math.sin(theta_prev)
                       / complex(n_substrate).real)
            sin_sub = min(sin_sub, 0.9999)
            cos_sub = math.sqrt(1 - sin_sub * sin_sub)
            cos_angle = math.cos(angle)
            if pol == 's':
                eta_sub = complex(n_substrate) * cos_sub
                eta_amb = complex(n_ambient) * cos_angle
            else:
                eta_sub = complex(n_substrate) / cos_sub
                eta_amb = complex(n_ambient) / cos_angle
            eta_sub_by_pol[pol] = eta_sub
            eta_amb_by_pol[pol] = eta_amb
            B = M[0, 0] + M[0, 1] * eta_sub
            C = M[1, 0] + M[1, 1] * eta_sub
            r = (eta_amb * B - C) / (eta_amb * B + C)
            t_amp = 2.0 * eta_amb / (eta_amb * B + C)
            rs.append(r)
            ts.append(t_amp)
        if polarization == 'avg':
            R_val = 0.5 * (abs(rs[0]) ** 2 + abs(rs[1]) ** 2)
            # v4.14.1 (audit P1-NEW-2): aggregate phases via complex
            # sum before taking the angle.  Pre-v4.14.1 form
            # ``0.5 * (np.angle(rs[0]) + np.angle(rs[1]))`` is off by
            # +/-pi at Brewster because r_p sign-flips through zero;
            # the unwrapped arithmetic average of two angles separated
            # by ~pi is wrong.  This reference helper has been updated
            # to match the corrected coatings.py implementation.
            phase_val = np.angle(0.5 * (rs[0] + rs[1]))
            _eta_sub_s = eta_sub_by_pol['s']
            _eta_amb_s = eta_amb_by_pol['s']
            _eta_sub_p = eta_sub_by_pol['p']
            _eta_amb_p = eta_amb_by_pol['p']
            T_s = ((_eta_sub_s.real / max(_eta_amb_s.real, 1e-30))
                    * abs(ts[0]) ** 2)
            T_p = ((_eta_sub_p.real / max(_eta_amb_p.real, 1e-30))
                    * abs(ts[1]) ** 2)
            T_val = 0.5 * (T_s + T_p)
        else:
            R_val = abs(rs[0]) ** 2
            phase_val = np.angle(rs[0])
            T_val = ((eta_sub.real / max(eta_amb.real, 1e-30))
                      * abs(ts[0]) ** 2)
        R[iw] = R_val
        T[iw] = max(0.0, T_val)
        phase_r[iw] = phase_val
    return R, T, phase_r


def _build_50layer_ar_stack(lam0=1.0e-6):
    """50-layer alternating high/low-index quarter-wave AR-like stack."""
    n_H = 2.30  # TiO2-like
    n_L = 1.46  # SiO2-like
    d_H = lam0 / (4 * n_H)
    d_L = lam0 / (4 * n_L)
    layers = []
    for i in range(50):
        if i % 2 == 0:
            layers.append((n_L, d_L))
        else:
            layers.append((n_H, d_H))
    return layers


class Test2A_CoatingWavelengthBatch:
    """Pin the wavelength-batch refactor against a scalar-loop reference."""

    def test_50layer_200wv_normal_incidence_avg_pol(self):
        layers = _build_50layer_ar_stack(1.0e-6)
        wavelengths = np.linspace(0.8e-6, 1.2e-6, 200)
        R_ref, T_ref, ph_ref = _scalar_loop_coating_reference(
            layers, wavelengths, angle=0.0,
            n_substrate=1.52, n_ambient=1.0, polarization='avg')
        R_new, T_new, ph_new = coating_reflectance(
            layers, wavelengths, angle=0.0,
            n_substrate=1.52, n_ambient=1.0, polarization='avg')
        np.testing.assert_allclose(R_new, R_ref, atol=1e-12,
                                    err_msg='R mismatch')
        np.testing.assert_allclose(T_new, T_ref, atol=1e-12,
                                    err_msg='T mismatch')
        np.testing.assert_allclose(ph_new, ph_ref, atol=1e-12,
                                    err_msg='phase_r mismatch')

    def test_50layer_200wv_oblique_aoi_s_pol(self):
        layers = _build_50layer_ar_stack(1.0e-6)
        wavelengths = np.linspace(0.8e-6, 1.2e-6, 200)
        angle = math.radians(30.0)
        R_ref, T_ref, ph_ref = _scalar_loop_coating_reference(
            layers, wavelengths, angle=angle,
            n_substrate=1.52, n_ambient=1.0, polarization='s')
        R_new, T_new, ph_new = coating_reflectance(
            layers, wavelengths, angle=angle,
            n_substrate=1.52, n_ambient=1.0, polarization='s')
        np.testing.assert_allclose(R_new, R_ref, atol=1e-12)
        np.testing.assert_allclose(T_new, T_ref, atol=1e-12)
        np.testing.assert_allclose(ph_new, ph_ref, atol=1e-12)

    def test_50layer_200wv_oblique_aoi_p_pol(self):
        layers = _build_50layer_ar_stack(1.0e-6)
        wavelengths = np.linspace(0.8e-6, 1.2e-6, 200)
        angle = math.radians(30.0)
        R_ref, T_ref, ph_ref = _scalar_loop_coating_reference(
            layers, wavelengths, angle=angle,
            n_substrate=1.52, n_ambient=1.0, polarization='p')
        R_new, T_new, ph_new = coating_reflectance(
            layers, wavelengths, angle=angle,
            n_substrate=1.52, n_ambient=1.0, polarization='p')
        np.testing.assert_allclose(R_new, R_ref, atol=1e-12)
        np.testing.assert_allclose(T_new, T_ref, atol=1e-12)
        np.testing.assert_allclose(ph_new, ph_ref, atol=1e-12)

    def test_complex_index_lossy_stack(self):
        """Documented approximation: n.imag dropped at the Snell step
        but kept in the phase-thickness delta; this test pins that the
        new batched code reproduces the scalar-loop output for an
        absorbing layer (so any future complex-Snell rewrite has a
        regression checkpoint)."""
        layers = [
            (1.46, 100e-9),                    # SiO2 spacer
            (2.30 + 0.01j, 80e-9),             # mildly absorbing high-index
            (1.46, 100e-9),
        ]
        wavelengths = np.linspace(0.9e-6, 1.1e-6, 50)
        R_ref, T_ref, ph_ref = _scalar_loop_coating_reference(
            layers, wavelengths, angle=0.1,
            n_substrate=1.52, n_ambient=1.0, polarization='avg')
        R_new, T_new, ph_new = coating_reflectance(
            layers, wavelengths, angle=0.1,
            n_substrate=1.52, n_ambient=1.0, polarization='avg')
        np.testing.assert_allclose(R_new, R_ref, atol=1e-12)
        np.testing.assert_allclose(T_new, T_ref, atol=1e-12)
        np.testing.assert_allclose(ph_new, ph_ref, atol=1e-12)

    def test_empty_stack_uncoated_fresnel(self):
        """Zero layers: characteristic matrix is identity, result is
        the bare Fresnel reflectance at the ambient/substrate
        interface.  Pins that the n_layers=0 fast path still produces
        a wavelength-shaped array."""
        wavelengths = np.linspace(0.9e-6, 1.1e-6, 11)
        R_ref, T_ref, ph_ref = _scalar_loop_coating_reference(
            [], wavelengths, angle=0.0,
            n_substrate=1.5, n_ambient=1.0, polarization='s')
        R_new, T_new, ph_new = coating_reflectance(
            [], wavelengths, angle=0.0,
            n_substrate=1.5, n_ambient=1.0, polarization='s')
        np.testing.assert_allclose(R_new, R_ref, atol=1e-14)
        np.testing.assert_allclose(T_new, T_ref, atol=1e-14)
        np.testing.assert_allclose(ph_new, ph_ref, atol=1e-14)
        # Fresnel result is wavelength-independent (in this approx),
        # so all entries should match the analytical R = ((n1-n2)/(n1+n2))^2.
        r_fresnel = ((1.0 - 1.5) / (1.0 + 1.5)) ** 2
        np.testing.assert_allclose(R_new, r_fresnel, atol=1e-14)

    def test_singleton_array_returns_array(self):
        """Length-1 array input -> length-1 array output.  Pins the
        existing back-compat: all existing callers pass [lam] or
        np.array([lam]) and expect ndarray return."""
        layers = [(1.38, 1.0e-6 / (4 * 1.38))]
        R, T, ph = coating_reflectance(
            layers, np.array([1.0e-6]), angle=0.0,
            n_substrate=1.5, n_ambient=1.0, polarization='avg')
        assert isinstance(R, np.ndarray) and R.shape == (1,)
        assert isinstance(T, np.ndarray) and T.shape == (1,)
        assert isinstance(ph, np.ndarray) and ph.shape == (1,)

    def test_scalar_input_returns_scalar(self):
        """0-d / Python-scalar input -> Python-float output (new in
        v4.14.0).  Lets callers write
        ``R, T, p = coating_reflectance(layers, lam, ...)`` and use
        the scalar directly without indexing."""
        layers = [(1.38, 1.0e-6 / (4 * 1.38))]
        R, T, ph = coating_reflectance(
            layers, 1.0e-6, angle=0.0,
            n_substrate=1.5, n_ambient=1.0, polarization='avg')
        assert isinstance(R, float)
        assert isinstance(T, float)
        assert isinstance(ph, float)
        # And the value must match what the array-input path produces.
        R_arr, T_arr, ph_arr = coating_reflectance(
            layers, np.array([1.0e-6]), angle=0.0,
            n_substrate=1.5, n_ambient=1.0, polarization='avg')
        assert abs(R - float(R_arr[0])) < 1e-14
        assert abs(T - float(T_arr[0])) < 1e-14
        assert abs(ph - float(ph_arr[0])) < 1e-14


# ---------------------------------------------------------------------------
# 2B: _evaluate_polynomial_4d_and_grad34 vectorisation correctness pin
# ---------------------------------------------------------------------------


def _scalar_loop_poly34_reference(coeffs, multi_indices, u1, u2, u3, u4,
                                    max_order):
    """Pre-v4.14.0 scalar-basis-loop polynomial-and-grad evaluator.

    Inlined here as the pin reference for the tensordot vectorisation.
    """
    shape = np.broadcast(u1, u2, u3, u4).shape
    T1 = _chebyshev_vandermonde(u1, max_order)
    T2 = _chebyshev_vandermonde(u2, max_order)
    T3 = _chebyshev_vandermonde(u3, max_order)
    T4 = _chebyshev_vandermonde(u4, max_order)
    dT3 = _chebyshev_derivative_vandermonde(u3, max_order)
    dT4 = _chebyshev_derivative_vandermonde(u4, max_order)
    f = np.zeros(shape, dtype=np.float64)
    df3 = np.zeros(shape, dtype=np.float64)
    df4 = np.zeros(shape, dtype=np.float64)
    for c, (k1, k2, k3, k4) in zip(coeffs, multi_indices):
        if c == 0.0:
            continue
        T12 = T1[k1] * T2[k2]
        f = f + c * T12 * T3[k3] * T4[k4]
        df3 = df3 + c * T12 * dT3[k3] * T4[k4]
        df4 = df4 + c * T12 * T3[k3] * dT4[k4]
    return f, df3, df4


class Test2B_PolynomialGradVectorise:
    """Pin the basis-vectorised gradient evaluator against the loop."""

    def test_random_30term_degree4_16grid(self):
        max_order = 4
        multi_indices = _multi_indices_total_degree(4, max_order)
        # Pick 30 random basis terms with non-trivial coefficients;
        # leave the rest at zero so the early-exit branch in the
        # scalar reference (`if c == 0: continue`) is also covered.
        rng = np.random.default_rng(0xC0FFEE)
        coeffs = np.zeros(len(multi_indices), dtype=np.float64)
        idx = rng.choice(len(multi_indices), size=30, replace=False)
        coeffs[idx] = rng.standard_normal(30)
        # 16x16x16x16 evaluation grid in [-0.9, 0.9]^4
        ax = np.linspace(-0.9, 0.9, 16)
        u1, u2, u3, u4 = np.meshgrid(ax, ax, ax, ax, indexing='ij')
        f_ref, d3_ref, d4_ref = _scalar_loop_poly34_reference(
            coeffs, multi_indices, u1, u2, u3, u4, max_order)
        f_new, d3_new, d4_new = _evaluate_polynomial_4d_and_grad34(
            coeffs, multi_indices, u1, u2, u3, u4, max_order)
        np.testing.assert_allclose(f_new, f_ref, atol=1e-14, rtol=0,
                                    err_msg='f mismatch')
        np.testing.assert_allclose(d3_new, d3_ref, atol=1e-14, rtol=0,
                                    err_msg='df/du3 mismatch')
        np.testing.assert_allclose(d4_new, d4_ref, atol=1e-14, rtol=0,
                                    err_msg='df/du4 mismatch')

    def test_full_70term_degree4_64grid(self):
        """Mirror the M=70-ish Newton-step workload that actually
        runs inside apply_real_lens_maslov: full total-degree<=4
        4D basis (70 terms), non-zero coefficients."""
        max_order = 4
        multi_indices = _multi_indices_total_degree(4, max_order)
        rng = np.random.default_rng(0xDECAFBAD)
        coeffs = rng.standard_normal(len(multi_indices))
        ax = np.linspace(-0.8, 0.8, 8)  # 8^4 = 4096 grid points
        u1, u2, u3, u4 = np.meshgrid(ax, ax, ax, ax, indexing='ij')
        f_ref, d3_ref, d4_ref = _scalar_loop_poly34_reference(
            coeffs, multi_indices, u1, u2, u3, u4, max_order)
        f_new, d3_new, d4_new = _evaluate_polynomial_4d_and_grad34(
            coeffs, multi_indices, u1, u2, u3, u4, max_order)
        np.testing.assert_allclose(f_new, f_ref, atol=1e-14, rtol=0)
        np.testing.assert_allclose(d3_new, d3_ref, atol=1e-14, rtol=0)
        np.testing.assert_allclose(d4_new, d4_ref, atol=1e-14, rtol=0)

    def test_real_maslov_newton_inversion_bit_near_exact(self):
        """End-to-end check: run apply_real_lens_maslov on a small
        spherical singlet and confirm the field magnitude/phase is
        bit-near-exact against the scalar-loop reference evaluator.

        The grad34 helper is only called inside Newton iterations of
        apply_real_lens_maslov, so the only practical way to exercise
        it on the actual call path is to run the propagator.  We
        monkeypatch the lenses module so the scalar reference is used
        in one run and the vectorised version in another, comparing
        the final fields.
        """
        import lumenairy as lm
        from lumenairy.elements import lenses as _lmod

        # Small problem to keep the test fast.
        N = 32
        dx = 1.0e-5
        wavelength = 1.31e-6
        x = (np.arange(N) - N // 2) * dx
        xx, yy = np.meshgrid(x, x, indexing='ij')
        # Simple Gaussian beam
        w0 = 100e-6
        E = np.exp(-(xx ** 2 + yy ** 2) / w0 ** 2).astype(np.complex128)

        prescription = lm.make_singlet(R1=50e-3, R2=-50e-3, d=3e-3,
                                        glass='N-BK7', aperture=0.3e-3)

        # Cheap ray-field / pupil sampling for fast tests.
        kw = dict(prescription=prescription, wavelength=wavelength, dx=dx,
                   ray_field_samples=8, ray_pupil_samples=8, n_v2=8)

        # Run with the vectorised (current) version.  Filter the
        # aperture-vs-grid UserWarning so the test output is clean.
        import warnings as _warn
        with _warn.catch_warnings():
            _warn.simplefilter('ignore')
            E_vec = lm.apply_real_lens_maslov(E, **kw)

        # Now swap in the scalar-loop reference and run again.
        original = _lmod._evaluate_polynomial_4d_and_grad34
        _lmod._evaluate_polynomial_4d_and_grad34 = (
            _scalar_loop_poly34_reference)
        try:
            # The apply_real_lens_maslov path looks up the helper via
            # the module-level binding in lenses_maslov, which itself
            # imports from elements.lenses.  Patch both name spaces.
            from lumenairy.elements import lenses_maslov as _lm_mod
            had_lm = hasattr(_lm_mod,
                              '_evaluate_polynomial_4d_and_grad34')
            if had_lm:
                _orig_lm = _lm_mod._evaluate_polynomial_4d_and_grad34
                _lm_mod._evaluate_polynomial_4d_and_grad34 = (
                    _scalar_loop_poly34_reference)
            from lumenairy.propagators import asymptotic as _asy_mod
            had_asy = hasattr(_asy_mod,
                               '_evaluate_polynomial_4d_and_grad34')
            if had_asy:
                _orig_asy = _asy_mod._evaluate_polynomial_4d_and_grad34
                _asy_mod._evaluate_polynomial_4d_and_grad34 = (
                    _scalar_loop_poly34_reference)
            try:
                with _warn.catch_warnings():
                    _warn.simplefilter('ignore')
                    E_loop = lm.apply_real_lens_maslov(E, **kw)
            finally:
                if had_lm:
                    _lm_mod._evaluate_polynomial_4d_and_grad34 = _orig_lm
                if had_asy:
                    _asy_mod._evaluate_polynomial_4d_and_grad34 = _orig_asy
        finally:
            _lmod._evaluate_polynomial_4d_and_grad34 = original

        # The Newton iterations may not converge to identical floating
        # point values because the scalar loop and tensordot path
        # accumulate sums in different orders; pin a tight relative
        # tolerance instead of bit-exact.
        max_abs = max(np.max(np.abs(E_vec)), 1e-30)
        rel_diff = np.max(np.abs(E_vec - E_loop)) / max_abs
        assert rel_diff < 1e-10, (
            f'Maslov field differs by {rel_diff:.3e} (rel) between '
            f'scalar-loop and tensordot-vectorised grad34 evaluator; '
            f'expected < 1e-10.'
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
