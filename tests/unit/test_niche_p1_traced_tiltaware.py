"""Accuracy-niches P1 / N5 (2026-07-19): traced ``tilt_aware_rays`` entrance
eikonal.

Background
----------
The hammer H6 fix restored the carrier's ENTRANCE-plane eikonal ``k0*W(x_in)`` on
the ``carrier=`` path of :func:`apply_real_lens_traced`.  N5 asks whether the
``tilt_aware_rays=True`` path has the same omission.

Measured verdict (this file pins it):

* On the DEFAULT ``preserve_input_phase=True`` the ``tilt_aware`` path was NOT
  collapsing -- a diverging/tilted input already focuses at its true image,
  because ``E_analytic = apply_real_lens(E_in)`` carries the input eikonal and
  the tilt_aware reference leg is a PLANE WAVE (it does not subtract the eikonal,
  unlike the carrier path's ``exp(i*k0*W)`` leg that made the H6 collapse surface
  on the default path).  The plan's stated fail-before does not reproduce there;
  the acceptance gate (tilt_aware agrees with carrier at the ABCD focus,
  EE100 > 0.99) was already met.
* The genuine omission is confined to the NON-default legacy
  ``preserve_input_phase=False`` mode (documented "discards input phase"), which
  builds the exit phase from ``opl_traced`` ALONE.  There the ray tracer
  accumulates OPL only from the entrance plane forward, so a diverging/tilted
  input collapses to the collimated focal plane ``f`` -- while the carrier path
  (H6) carries the eikonal in BOTH modes.

The N5 fix threads an auto-fit carrier eikonal through the SAME carrier plumbing
(reference leg + H6 entrance-eikonal OPL term) whenever ``tilt_aware_rays=True``
and no explicit carrier is set, so the exit wavefront carries the input
congruence in both preserve modes.  It engages only when the fitted eikonal is
non-trivial, so a (near-)collimated tilt_aware input keeps the byte-identical
plane-wave path.

Oracle: ABCD Gaussian q-trace through the dual-oracle singlet (R = +/-51.68 mm,
t = 5 mm, n = 1.5168, lambda = 1.31 um) -- independent of the traced code.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_traced as LT

_WL = 1.31e-6
_N_GLASS = 1.5168
_R1, _R2, _TC = 51.68e-3, -51.68e-3, 5e-3
_F_PLANE = 49.163e-3            # collimated focus (BFL), dual-oracle number

# Model glass for THIS module only: registered and removed by
# tests/conftest.py::_module_glass_registry_guard.
MODULE_GLASSES = {'_N5_FIX_GLASS': lambda wl: _N_GLASS}


def _singlet():
    return {
        'wavelength': _WL, 'aperture_diameter': 20e-3,
        'surfaces': [
            {'radius': _R1, 'thickness': _TC, 'glass_before': 'air',
             'glass_after': '_N5_FIX_GLASS', 'semi_diameter': 10e-3},
            {'radius': _R2, 'thickness': 0.0, 'glass_before': '_N5_FIX_GLASS',
             'glass_after': 'air', 'semi_diameter': 10e-3},
        ],
        'thicknesses': [_TC], 'stop_index': 0,
    }


def _lens_abcd():
    def refr(R, n1, n2):
        return np.array([[1.0, 0.0], [-(n2 - n1) / (R * n2), n1 / n2]])

    def trans(d):
        return np.array([[1.0, d], [0.0, 1.0]])

    return refr(_R2, _N_GLASS, 1.0) @ trans(_TC) @ refr(_R1, 1.0, _N_GLASS)


def _q_trace_image(R_in, w_L):
    q_inv = 1.0 / R_in - 1j * _WL / (np.pi * w_L ** 2)
    q0 = 1.0 / q_inv
    M = _lens_abcd()
    q1 = (M[0, 0] * q0 + M[0, 1]) / (M[1, 0] * q0 + M[1, 1])
    return float(-q1.real)


def _diverging_input(N, dx, w_L, R_in):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r_sq = X ** 2 + Y ** 2
    k = 2 * np.pi / _WL
    return (np.exp(-r_sq / w_L ** 2)
            * np.exp(1j * k * r_sq / (2.0 * R_in))).astype(np.complex128)


def _ee100_at(E_exit, dx, z):
    E = la.angular_spectrum_propagate(E_exit, z, _WL, dx)
    I = np.abs(E) ** 2
    N = I.shape[0]
    x = (np.arange(N) - N / 2) * dx
    j, i = np.unravel_index(np.argmax(I), I.shape)
    X, Y = np.meshgrid(x - x[i], x - x[j])
    r = np.sqrt(X ** 2 + Y ** 2)
    return float(I[r <= 100e-6].sum() / I.sum())


def _traced(E0, dx, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')      # H3 nyquist / F1 guards
        return la.apply_real_lens_traced(
            E0, prescription=_singlet(), wavelength=_WL, dx=dx, **kw)


# ---------------------------------------------------------------------------
# Core fix: preserve_input_phase=False + tilt_aware on a diverging input.
# Fail-before (gate forced off via monkeypatch = the pre-N5 behaviour) collapses
# to the collimated focus f; the fix (gate on) focuses at the ABCD image.
# ---------------------------------------------------------------------------
def test_n5_tiltaware_pip_false_collapse_fixed(monkeypatch):
    N, dx, w_L, R_in = 1024, 5e-6, 3e-3, 150e-3
    z_img = _q_trace_image(R_in, w_L)
    E0 = _diverging_input(N, dx, w_L, R_in)

    # FAIL-BEFORE: force the pre-N5 plane-wave path (threshold = inf keeps the
    # auto-carrier from engaging) -> the entrance eikonal is dropped and the
    # diverging input collapses to the collimated focal plane f.
    monkeypatch.setattr(LT, '_TILT_EIKONAL_MIN_RAD', np.inf)
    E_old = _traced(E0, dx, tilt_aware_rays=True, preserve_input_phase=False)
    ee_img_old = _ee100_at(E_old, dx, z_img)
    ee_f_old = _ee100_at(E_old, dx, _F_PLANE)
    assert ee_f_old > ee_img_old and ee_img_old < 0.2, (
        f"pre-N5 pip=False tilt_aware must collapse to f "
        f"(EE_img={ee_img_old:.3f}, EE_f={ee_f_old:.3f})")

    # PASS-AFTER: restore the fix -> the entrance eikonal rides the OPL and the
    # beam focuses at the ABCD image, beating the collimated plane f.
    monkeypatch.setattr(LT, '_TILT_EIKONAL_MIN_RAD', 1e-2)
    E_new = _traced(E0, dx, tilt_aware_rays=True, preserve_input_phase=False)
    ee_img = _ee100_at(E_new, dx, z_img)
    ee_f = _ee100_at(E_new, dx, _F_PLANE)
    assert ee_img > 0.9, (
        f"pip=False tilt_aware must focus at z_img={z_img*1e3:.2f}mm after N5 "
        f"(EE100={ee_img:.3f})")
    assert ee_img > ee_f, (ee_img, ee_f)


# ---------------------------------------------------------------------------
# Acceptance gate (plan N5): on the DEFAULT preserve_input_phase=True the
# tilt_aware and carrier paths agree at the ABCD focus, EE100 > 0.99.
# ---------------------------------------------------------------------------
def test_n5_tiltaware_agrees_with_carrier_at_abcd_focus():
    N, dx, w_L, R_in = 2048, 5e-6, 3e-3, 150e-3
    z_img = _q_trace_image(R_in, w_L)
    E0 = _diverging_input(N, dx, w_L, R_in)
    ee_tilt = _ee100_at(_traced(E0, dx, tilt_aware_rays=True), dx, z_img)
    ee_carr = _ee100_at(_traced(E0, dx, carrier='auto'), dx, z_img)
    assert ee_tilt > 0.99, f"tilt_aware EE100@z_img={ee_tilt:.4f} <= 0.99"
    assert ee_carr > 0.99, f"carrier EE100@z_img={ee_carr:.4f} <= 0.99"
    assert abs(ee_tilt - ee_carr) < 0.02, (ee_tilt, ee_carr)


# ---------------------------------------------------------------------------
# Byte-identical: a (near-)collimated tilt_aware input fits W == 0, so the gate
# is OFF and the fix is a strict no-op.  Two independent proofs, in both preserve
# modes:
#   (1) the N5 auto-carrier path (tilt_aware=True, gate naturally off) equals the
#       gate-forced-off path (monkeypatch threshold=inf) -- the pre-N5 plane-wave
#       reference -- to the traced FP floor (the codebase's byte-identical bar for
#       traced is rel < 1e-12; numba parallel/fastmath + FFTW plan state add ULP
#       noise across a varying-N session, so array_equal is too strict here);
#   (2) tilt_aware=True equals tilt_aware=False (the axial path is untouched by
#       N5 -- the carrier block only fires for tilt_aware or an explicit carrier),
#       which for a collimated field launches identically (grad(phase)=0).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('pip', [True, False])
def test_n5_collimated_tiltaware_byte_identical(monkeypatch, pip):
    N, dx, w_L = 512, 8e-6, 3e-3
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / w_L ** 2).astype(np.complex128)

    def _rel(A, B):
        m = np.abs(A) > 1e-3 * np.abs(A).max()
        return float(np.linalg.norm((A - B)[m])) / (float(np.linalg.norm(A[m])) + 1e-30)

    # (2) tilt_aware == axial (untouched path) for a collimated field.
    E_tilt = _traced(E0, dx, tilt_aware_rays=True, preserve_input_phase=pip)
    E_axial = _traced(E0, dx, tilt_aware_rays=False, preserve_input_phase=pip)
    assert _rel(E_tilt, E_axial) < 1e-12, (
        f"collimated tilt_aware must equal the untouched axial path (rel="
        f"{_rel(E_tilt, E_axial):.2e})")

    # (1) natural (gate off) == gate-forced-off (pre-N5 plane-wave path).
    monkeypatch.setattr(LT, '_TILT_EIKONAL_MIN_RAD', np.inf)
    E_forced_off = _traced(E0, dx, tilt_aware_rays=True, preserve_input_phase=pip)
    assert _rel(E_tilt, E_forced_off) < 1e-12, (
        "collimated tilt_aware must take the byte-identical plane-wave path "
        f"(rel={_rel(E_tilt, E_forced_off):.2e})")


# ---------------------------------------------------------------------------
# Adversarial (plan N5): tilt_aware AND an explicit carrier requested together
# must NOT double-count W -- the beam still focuses at the ABCD image with the
# same EE as either alone (a double-count would smear or shift it).
# ---------------------------------------------------------------------------
def test_n5_tiltaware_plus_carrier_no_double_count():
    N, dx, w_L, R_in = 1024, 5e-6, 3e-3, 150e-3
    z_img = _q_trace_image(R_in, w_L)
    E0 = _diverging_input(N, dx, w_L, R_in)
    ee = {}
    for name, kw in (('both', {'tilt_aware_rays': True, 'carrier': R_in}),
                     ('carrier', {'carrier': R_in}),
                     ('tilt', {'tilt_aware_rays': True})):
        ee[name] = _ee100_at(_traced(E0, dx, **kw), dx, z_img)
    assert ee['both'] > 0.9, f"tilt+carrier collapsed: EE100={ee['both']:.3f}"
    # no double-count: 'both' tracks 'carrier' (not smeared/shifted to a
    # different plane); a doubled W would drop EE well below either.
    assert abs(ee['both'] - ee['carrier']) < 0.03, ee
    assert abs(ee['both'] - ee['tilt']) < 0.03, ee


# ---------------------------------------------------------------------------
# Unit check of the engage gate: a flat-wavefront input fits W == 0 exactly
# (gate off); a diverging input fits a large eikonal (gate on).
# ---------------------------------------------------------------------------
def test_n5_engage_gate_separates_collimated_from_diverging():
    N, dx, w_L = 512, 8e-6, 3e-3
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    k0 = 2 * np.pi / _WL
    flat = np.exp(-(X ** 2 + Y ** 2) / w_L ** 2).astype(np.complex128)
    Wf, _, _ = LT._compute_carrier('auto', flat, _WL, dx, X, Y)
    assert float(np.nanmax(np.abs(Wf))) * k0 <= LT._TILT_EIKONAL_MIN_RAD

    phased = (flat * np.exp(1j * 0.9))          # global phase -> still flat
    Wp, _, _ = LT._compute_carrier('auto', phased, _WL, dx, X, Y)
    assert float(np.nanmax(np.abs(Wp))) * k0 <= LT._TILT_EIKONAL_MIN_RAD

    div = _diverging_input(N, dx, w_L, 150e-3)
    Wd, _, _ = LT._compute_carrier('auto', div, _WL, dx, X, Y)
    mag = np.abs(div)
    bright = mag > 0.05 * mag.max()
    assert float(np.nanmax(np.abs(Wd[bright]))) * k0 > LT._TILT_EIKONAL_MIN_RAD


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
