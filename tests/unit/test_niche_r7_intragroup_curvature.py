"""R7 / audit F2 (2026-07-21) -- intra-group exit-curvature fidelity of
``apply_real_lens_traced(carrier=...)`` on a thick group.

Audit finding F2: with the correct explicit carrier a THICK, strongly-focusing
group still left a SMOOTH exit-wavefront error -- an exit-CURVATURE (defocus)
term dominating on triplets and steep input curvature, plus a genuine r^4
residual at the steepest conjugate -- accumulating ~6.8 rad rms over the 8-group
design-121 relay (Strehl ~ 0 -> the observed pedestal).  Two independent root
causes were fixed:

* the GLOBAL tensor-Chebyshev fit of the entrance->exit map + OPL aliased the
  far marginal rays' out-of-basis high order (r^8+) into the low-order (defocus)
  coefficients -> a carrier-gated fit-domain restriction drops those rays;
* the coarse-grid Newton OPL was LINEARLY upsampled -> cubic upsampling on the
  carrier path removes the outer-beam interpolation residual;
* the float ``carrier`` conjugate used the PARAXIAL parabola ``r^2/(2s)`` rather
  than the EXACT point-source sphere -> a large r^4 on a steep conjugate; the
  exact sphere makes the reference leg, ray-launch cosines and H6 eikonal agree.

These tests are SELF-CONTAINED (no Zemax): a synthetic thick triplet with a
steep spherical carrier, validated against an INLINE exact meridional
raytrace + eikonal oracle (lumenairy-free), mirroring the repro oracle at
``validation/repro_traced_carrier_121/traced_group_oracle.py``.  A flat window
stays exact (the regression guard: windows were exact both before and after the
fix, so any change there is a convention break).

fail-before / pass-after: with the R7 fit-domain restriction disabled the
synthetic triplet reads ~1.8 rad (reproducing F2); with it, < 0.02 rad.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.glass import get_glass_index

_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL


def _n_of(g):
    return 1.0 if g in (None, '', 'air', 'AIR') else float(get_glass_index(g, _WL))


def _build_triplet(radii, glasses, thicks, aperture):
    """Four-surface (three-element) group prescription; ``radii`` may be inf."""
    gb = ['air', glasses[0], glasses[1], glasses[2]]
    ga = [glasses[0], glasses[1], glasses[2], 'air']
    surfaces = [
        {'radius': radii[i], 'glass_before': gb[i], 'glass_after': ga[i],
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None}
        for i in range(4)
    ]
    return {'name': 'r7-synth', 'aperture_diameter': aperture,
            'surfaces': surfaces, 'thicknesses': list(thicks)}


def _group_geometry(pres):
    surfs = []
    z = 0.0
    th = pres['thicknesses']
    for i, s in enumerate(pres['surfaces']):
        R_s = s['radius'] if s['radius'] else np.inf
        surfs.append((z, R_s, _n_of(s['glass_before']), _n_of(s['glass_after'])))
        z += th[i] if i < len(th) else 0.0
    return surfs, surfs[-1][0]


def _oracle_phase(surfs, z_exit, R_in, x_max, npts=600):
    """Exact meridional trace from the on-axis carrier point source through the
    spherical surfaces (exact Snell), eikonal to the exit vertex plane; returns
    ``(x_exit, phi)`` with the on-axis piston removed (mirrors the repro
    oracle).  Signed-t continuation for concave-front surfaces whose cap lies
    behind the vertex plane."""
    x0s = np.linspace(1e-9, x_max, npts)
    xe = np.full(npts, np.nan)
    ph = np.full(npts, np.nan)
    for j, x0 in enumerate(x0s):
        P = np.array([x0, 0.0])
        if np.isinf(R_in):
            u = np.array([0.0, 1.0])
        else:
            u = np.array([x0, R_in]) / np.hypot(x0, R_in)
            if R_in < 0:
                u = np.array([-x0, -R_in]) / np.hypot(x0, R_in)
        opl = (np.sign(R_in) * (np.hypot(x0, R_in) - abs(R_in))
               if np.isfinite(R_in) else 0.0)
        ok = True
        for (zv, R_s, n1, n2) in surfs:
            if np.isfinite(R_s):
                C = np.array([0.0, zv + R_s])
                oc = P - C
                b = np.dot(oc, u)
                c = np.dot(oc, oc) - R_s * R_s
                disc = b * b - c
                if disc < 0:
                    ok = False
                    break
                t1, t2 = -b - np.sqrt(disc), -b + np.sqrt(disc)
                t = min((t1, t2), key=lambda tt: abs((P + tt * u)[1] - zv))
                P2 = P + t * u
                nv = np.sign(R_s) * (C - P2) / np.linalg.norm(C - P2)
            else:
                if abs(u[1]) < 1e-12:
                    ok = False
                    break
                t = (zv - P[1]) / u[1]
                P2 = P + t * u
                nv = np.array([0.0, 1.0])
            opl += n1 * t
            c1 = np.dot(u, nv)
            eta = n1 / n2
            disc2 = 1.0 - eta * eta * (1.0 - c1 * c1)
            if disc2 < 0:
                ok = False
                break
            u = eta * u + (np.sqrt(disc2) - eta * c1) * nv
            u = u / np.linalg.norm(u)
            P = P2
        if not ok:
            continue
        n_last = surfs[-1][3]
        t = (z_exit - P[1]) / u[1]
        opl += n_last * t
        xe[j] = (P + t * u)[0]
        ph[j] = _K0 * opl
    good = np.isfinite(xe)
    xe, ph = xe[good], ph[good]
    p2 = np.polyfit(xe[:20], ph[:20], 2)
    return xe, ph - np.polyval(p2, 0.0)


def _exit_rms(pres, R_in, w_, N=1024, **kw):
    """Apply the traced group to a carrier Gaussian and return the RMS exit-
    phase residual (rad, piston removed) vs the meridional oracle over r < w."""
    surfs, z_exit = _group_geometry(pres)
    dx = 4.2 * w_ / N * 2
    x = (np.arange(N) - N // 2) * dx
    r2g = x[None, :] ** 2 + x[:, None] ** 2
    env = np.exp(-r2g / w_ ** 2)
    Sin = (np.sign(R_in) * (np.sqrt(r2g + R_in * R_in) - abs(R_in))
           if np.isfinite(R_in) else 0.0)
    E_in = (env * np.exp(1j * _K0 * Sin)).astype(np.complex128)
    xe, ph = _oracle_phase(surfs, z_exit, R_in, x_max=3.2 * w_)
    with warnings.catch_warnings():
        # a steep exit converges beyond the coarse grid Nyquist (expected H3
        # halo-aliasing advisory) -- this test measures the CORE (r<w) wavefront
        warnings.simplefilter('ignore', RuntimeWarning)
        E_out = la.apply_real_lens_traced(
            E_in, prescription=pres, wavelength=_WL, dx=dx,
            ray_subsample=4, n_workers=1, parallel_amp=False, **kw)
    rr = np.sqrt(r2g)
    ph_o = np.interp(rr, xe, ph, left=ph[0], right=np.nan)
    res = np.angle(E_out * np.exp(-1j * ph_o))
    mask = (rr < w_) & np.isfinite(ph_o) & (np.abs(E_out) > 0.05 * np.abs(E_out).max())
    res0 = res - np.median(res[rr < 0.05 * w_])
    rm = res0[mask]
    # circular RMS (mean-phase removed), matching the repro metric
    m = np.angle(np.mean(np.exp(1j * rm)))
    return float(np.sqrt(np.mean(np.angle(np.exp(1j * (rm - m))) ** 2)))


# A strong, thick focusing triplet with a wide aperture -- the launch grid
# (1.5x the aperture radius) then reaches far past the beam, so the marginal
# rays' high-order aberration is what the F2 global-fit contamination aliased
# into defocus (reproduced at ~1.8 rad with the R7 restriction disabled).
_TRIPLET = _build_triplet(
    radii=[40e-3, -35e-3, 120e-3, -45e-3],
    glasses=['N-BK7', 'N-SF2', 'N-BK7'],
    thicks=[7e-3, 5e-3, 7e-3], aperture=24e-3)

# Three flat windows (no refracting power): the OPL is a pure piston, so the
# exit wavefront must be exact regardless of the carrier -- the regression guard.
_WINDOW = _build_triplet(
    radii=[np.inf, np.inf, np.inf, np.inf],
    glasses=['N-BK7', 'N-BK7', 'N-BK7'],
    thicks=[3e-3, 3e-3, 3e-3], aperture=24e-3)


def _skip_if_low_ram(N):
    """Skip a big-N case when free RAM is tight (defensive; N=1024 is small)."""
    try:
        import psutil
        if psutil.virtual_memory().available < 2 * (N * N * 16) + (1 << 30):
            pytest.skip('insufficient free RAM for the traced N={} case'.format(N))
    except Exception:
        pass


@pytest.mark.parametrize('R_in, w', [
    (55e-3, 4.0e-3),    # steep DIVERGING carrier (w/R = 0.073)
    (-30e-3, 3.5e-3),   # steep CONVERGING (w/R = 0.117 -> exact-sphere r^4)
])
def test_r7_thick_triplet_steep_carrier_exit_wavefront(R_in, w):
    """A thick focusing triplet under a steep spherical carrier must reproduce
    the exact meridional exit wavefront to < 0.1 rad rms (audit F2 gate).
    Pre-fix (global-fit defocus + paraxial-carrier r^4) this read ~1.8 rad."""
    _skip_if_low_ram(1024)
    rms = _exit_rms(_TRIPLET, R_in, w, carrier=R_in)
    assert rms < 0.1, (
        f"traced triplet exit-wavefront rms {rms:.3f} rad exceeds the F2 gate "
        f"(0.1) for carrier R_in={R_in*1e3:.1f} mm -- the intra-group "
        f"curvature error is not corrected")


def test_r7_flat_window_stays_exact_with_carrier():
    """A flat window (no refracting power) is a pure-piston OPL, so the exit
    wavefront is exact both before and after the R7 fix -- the regression guard
    that the carrier-gated fit restriction / exact sphere never perturb a
    benign group."""
    _skip_if_low_ram(1024)
    rms = _exit_rms(_WINDOW, 55e-3, 4.0e-3, carrier=55e-3)
    assert rms < 0.02, (
        f"flat-window exit wavefront rms {rms:.4f} rad -- the carrier path "
        f"perturbed a pure-piston group (convention break)")


def test_r7_window_carrier_matches_no_carrier():
    """On a flat window a spherical carrier changes only the (cancelling)
    reference legs, so the exit field must match the plane-wave-referenced
    (carrier=None) field on the SAME collimated input -- a default byte-level
    consistency check that the carrier plumbing is a no-op on a benign group."""
    _skip_if_low_ram(1024)
    N, w = 1024, 4.0e-3
    dx = 4.2 * w / N * 2
    x = (np.arange(N) - N // 2) * dx
    r2g = x[None, :] ** 2 + x[:, None] ** 2
    E_in = np.exp(-r2g / w ** 2).astype(np.complex128)   # collimated Gaussian
    common = dict(prescription=_WINDOW, wavelength=_WL, dx=dx,
                  ray_subsample=4, n_workers=1, parallel_amp=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        E_none = la.apply_real_lens_traced(E_in, **common)
        # a very-long-conjugate carrier is ~collimated: the exit must agree
        E_car = la.apply_real_lens_traced(E_in, carrier=5.0e3, **common)
    num = float(np.linalg.norm(E_none - E_car))
    den = float(np.linalg.norm(E_none)) + 1e-30
    assert num / den < 1e-3, (
        f"window carrier vs no-carrier relative mismatch {num/den:.2e} -- the "
        f"carrier path is not a no-op on a benign flat group")


def test_r7_compute_carrier_is_exact_sphere():
    """Unit test of the R7 exact-sphere fix in ``_compute_carrier``: a float
    conjugate must build the EXACT point-source sphere
    ``W = sign(s)(sqrt(r^2+s^2)-|s|)`` (not the paraxial ``r^2/2s``), with the
    ray-launch cosines and the entrance eikonal ``w_fn`` consistent."""
    from lumenairy.elements._lens_traced import _compute_carrier
    N, dx = 64, 5e-5
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    for s in (60e-3, -30e-3):
        W, grad_fn, w_fn = _compute_carrier(s, None, _WL, dx, X, Y)
        r2 = X ** 2 + Y ** 2
        W_exact = np.sign(s) * (np.sqrt(r2 + s * s) - abs(s))
        assert np.allclose(W, W_exact, rtol=0, atol=1e-15)
        # w_fn must reproduce W_full
        assert np.allclose(w_fn(X, Y), W_exact, rtol=0, atol=1e-15)
        # grad = dW/dx = sign(s) x / sqrt(r^2+s^2) (the ray direction cosine)
        Lx, My = grad_fn(X, Y)
        rho = np.sqrt(r2 + s * s)
        assert np.allclose(Lx, np.sign(s) * X / rho, rtol=0, atol=1e-15)
        assert np.allclose(My, np.sign(s) * Y / rho, rtol=0, atol=1e-15)
        # the exact sphere must DIFFER from the old paraxial parabola off-axis
        W_paraxial = r2 / (2.0 * s)
        assert not np.allclose(W, W_paraxial, atol=1e-9)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
