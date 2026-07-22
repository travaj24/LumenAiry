"""R6 / audit F1 (2026-07-21): ``apply_real_lens_traced(carrier='auto')`` must
robustly recover a spherical carrier R from a strongly-diverging / coarsely-
sampled input.

Before the fix the ``'auto'`` least-squares gradient fit silently degraded to
~inf (no carrier) on exactly that input class: the nearest-neighbour phase-
increment tilt reading ALIASES (wraps past +-pi) wherever the local carrier
tilt exceeds the grid Nyquist tilt ``lambda/(2 dx)``, and those wrapped
(near-zero-mean) samples -- the great majority of the bright support on a coarse
grid -- pulled the fitted 1/R toward 0.  The full-scale 121 chain therefore ran
as if the carrier machinery (hammer H6) had never landed.

The fix restricts the fit to the CONNECTED un-aliased core: the component,
seeded at the brightest pixel (the beam centre), whose local tilt stays below
the Nyquist tilt.  The central parabola alone fixes the spherical R; the wrapped
alias rings are separate connected components and are excluded.

These tests are SELF-CONTAINED / SYNTHETIC -- no .zmx, no Zemax.  A spherical-
carrier input is built directly and validated against an INLINE exact meridional
raytrace + eikonal oracle (a lumenairy-free mirror of
``validation/repro_traced_carrier_121/traced_group_oracle.py``).  The full 121
end-to-end acceptance runs LOCALLY against that repro; here we reproduce the F1
signature on a synthetic single-group prescription so CI (which has no .zmx) can
gate the fix.
"""
import warnings

import numpy as np
import pytest

from lumenairy import get_glass_index
from lumenairy.elements._lens_traced import (
    _AUTO_CARRIER_ALIAS_FRAC,
    _TILT_EIKONAL_MIN_RAD,
    _compute_carrier,
    apply_real_lens_traced,
)
from lumenairy.memory import available_memory_bytes

LAM = 1.31e-6
K0 = 2.0 * np.pi / LAM


# ---------------------------------------------------------------------------
# Synthetic inputs + inline (lumenairy-free) meridional oracle
# ---------------------------------------------------------------------------
def _spherical_input(N, dx, w, R):
    """Gaussian envelope x a FULL-sphere carrier ``exp(i k0 S(R))`` with
    ``S = sign(R)(sqrt(r^2+R^2)-|R|)`` -- identical to the repro oracle's
    input.  ``R>0`` diverging (point source in front of the plane)."""
    x = (np.arange(N) - N // 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    env = np.exp(-r2 / w ** 2)
    if np.isinf(R):
        s = 0.0
    else:
        s = np.sign(R) * (np.sqrt(r2 + R * R) - abs(R))
    return (env * np.exp(1j * K0 * s)).astype(np.complex128), x


def _singlet(R1, R2, t, glass='N-BK7', ap=8e-3):
    return {
        'name': 'r6-synthetic', 'aperture_diameter': ap,
        'surfaces': [
            {'radius': R1, 'conic': 0.0, 'glass_before': 'air',
             'glass_after': glass, 'semi_diameter': ap / 2},
            {'radius': R2, 'conic': 0.0, 'glass_before': glass,
             'glass_after': 'air', 'semi_diameter': ap / 2},
        ],
        'thicknesses': [t],
    }


def _oracle_exit_phase(surfs, z_exit, R_in, x_max, npts=300):
    """Exact meridional trace from the carrier point source through the
    spherical surfaces to the exit vertex plane; returns (x_exit, phi).

    ``surfs`` = list of ``(z_vertex_rel, R_surface, n_before, n_after)``.
    Lumenairy-free: vector Snell + eikonal only (mirrors the repro oracle).
    """
    x0s = np.linspace(1e-9, x_max, npts)
    xe = np.full(npts, np.nan)
    ph = np.full(npts, np.nan)
    for j, x0 in enumerate(x0s):
        p = np.array([x0, 0.0])
        if np.isinf(R_in):
            u = np.array([0.0, 1.0])
            opl = 0.0
        else:
            u = np.array([x0, R_in]) / np.hypot(x0, R_in)
            if R_in < 0:
                u = np.array([-x0, -R_in]) / np.hypot(x0, R_in)
            opl = np.sign(R_in) * (np.hypot(x0, R_in) - abs(R_in))
        ok = True
        for (zv, r_s, n1, n2) in surfs:
            if np.isfinite(r_s):
                cen = np.array([0.0, zv + r_s])
                oc = p - cen
                b = np.dot(oc, u)
                c = np.dot(oc, oc) - r_s * r_s
                disc = b * b - c
                if disc < 0:
                    ok = False
                    break
                t1, t2 = -b - np.sqrt(disc), -b + np.sqrt(disc)
                t = min((t1, t2), key=lambda tt: abs((p + tt * u)[1] - zv))
                p2 = p + t * u
                nv = np.sign(r_s) * (cen - p2) / np.linalg.norm(cen - p2)
            else:
                if abs(u[1]) < 1e-12:
                    ok = False
                    break
                t = (zv - p[1]) / u[1]
                p2 = p + t * u
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
            p = p2
        if not ok:
            continue
        n_last = surfs[-1][3]
        t = (z_exit - p[1]) / u[1]
        opl += n_last * t
        xe[j] = (p + t * u)[0]
        ph[j] = K0 * opl
    good = np.isfinite(xe)
    xe, ph = xe[good], ph[good]
    # axial reference: extrapolate to x_exit -> 0 (quadratic in x)
    p2 = np.polyfit(xe[:20], ph[:20], 2)
    return xe, ph - np.polyval(p2, 0.0)


def _residual_metrics(E_out, x, xe, ph, w):
    """Pointwise residual vs the oracle over r < w (piston removed).
    Returns (rms_residual, r4_residual) -- the two F1-table columns."""
    N = E_out.shape[0]
    rr = np.hypot(x[None, :], x[:, None])
    ph_o = np.interp(rr, xe, ph, left=ph[0], right=np.nan)
    res = np.angle(E_out * np.exp(-1j * ph_o))
    res0 = res - np.median(res[rr < 0.05 * w])
    mask = (rr < w) & np.isfinite(ph_o) & (
        np.abs(E_out) > 0.05 * np.abs(E_out).max())
    resm = res0[mask]
    rms = float(np.sqrt(np.mean(np.angle(np.exp(
        1j * (resm - np.angle(np.mean(np.exp(1j * resm)))))) ** 2)))
    selr = np.abs(x) < 0.5 * w
    rowu = np.unwrap(res0[N // 2, selr])
    rowu = rowu - np.polyval(np.polyfit(x[selr], rowu, 2), x[selr])
    r4 = float(np.std(rowu))
    return rms, r4


def _grad_recovered_R(E, dx, w):
    """Recover the spherical R the way the ray trace consumes it: from the
    carrier's transverse GRADIENT (L = dW/dx = x/R) at several core radii."""
    x = (np.arange(E.shape[0]) - E.shape[0] // 2) * dx
    X, Y = np.meshgrid(x, x)
    _W, grad_fn, _w = _compute_carrier('auto', E, LAM, dx, X, Y)
    xq = np.array([0.1, 0.2, 0.3, 0.4]) * w
    yq = np.zeros_like(xq)
    lq, _mq = grad_fn(xq, yq)
    return xq / lq   # = R at each radius (constant for a true sphere)


# ---------------------------------------------------------------------------
# F1 core: auto recovers R on an ALIASED spherical input (fail-before/pass-after)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('R_in,w,N', [
    (153.37e-3, 6.0e-3, 2048),   # 121 S5-S7 conditions (the audit's exact case)
    (30.0e-3, 1.6e-3, 512),      # steep + coarse: aliased beyond r ~ 0.75 mm
    (100.0e-3, 3.0e-3, 512),
    (-51.5e-3, 4.1e-3, 1024),    # converging carrier
])
def test_auto_recovers_spherical_R_on_aliased_input(R_in, w, N):
    if available_memory_bytes() < 1 * (1 << 30):
        pytest.skip("insufficient free RAM")
    dx = 8.4 * w / N              # 8.4 w window -> the repro's sampling
    nyq_tilt = LAM / (2.0 * dx)
    edge_tilt = abs(w / R_in)
    # Precondition: the input MUST be aliased (edge tilt over Nyquist) -- i.e.
    # exactly the class the old fit failed on.  If not, the test is vacuous.
    assert edge_tilt > nyq_tilt, (
        f"test input not aliased (edge {edge_tilt:.4f} <= nyq {nyq_tilt:.4f})")
    E, _x = _spherical_input(N, dx, w, R_in)
    Rvals = _grad_recovered_R(E, dx, w)
    # (1) auto recovers R to a few percent (the OLD code returned R ~7-600x too
    # large here -> this bound FAILS before the fix, PASSES after).
    Rmed = np.median(Rvals)
    err = abs(Rmed - R_in) / abs(R_in)
    assert err < 0.05, f"auto R={Rmed*1e3:.3f}mm vs {R_in*1e3:.3f}mm (err {err:.1%})"
    # (2) the recovered carrier is a clean sphere: grad is linear in radius, so
    # R is (nearly) the same at every core radius (no spurious tilt term).
    spread = (Rvals.max() - Rvals.min()) / abs(Rmed)
    assert spread < 0.05, f"grad non-linear (spread {spread:.1%}): {Rvals*1e3}"


# ---------------------------------------------------------------------------
# F1 acceptance mirrored end-to-end: auto == explicit R, both << no-carrier,
# validated against the inline meridional oracle on a synthetic singlet.
# ---------------------------------------------------------------------------
def test_auto_matches_explicit_endtoend_vs_oracle():
    if available_memory_bytes() < 1 * (1 << 30):
        pytest.skip("insufficient free RAM")
    R_in, w, N = 30.0e-3, 1.6e-3, 512
    dx = 8.4 * w / N
    # Aliased precondition (the old fit fails here):
    assert abs(w / R_in) > LAM / (2.0 * dx)
    t_glass = 3.0e-3
    rx = _singlet(35e-3, -35e-3, t_glass, ap=8e-3)
    n_g = float(get_glass_index('N-BK7', LAM))
    surfs = [(0.0, 35e-3, 1.0, n_g), (t_glass, -35e-3, n_g, 1.0)]
    xe, ph = _oracle_exit_phase(surfs, t_glass, R_in, x_max=0.9 * w)

    E_in, x = _spherical_input(N, dx, w, R_in)
    out = {}
    for name, kw in (('explicit', {'carrier': R_in}),
                     ('auto', {'carrier': 'auto'}),
                     ('none', {'carrier': None})):
        eo = apply_real_lens_traced(
            E_in, prescription=rx, wavelength=LAM, dx=dx, ray_subsample=4,
            n_workers=4, parallel_amp=False, on_noncollimated='off', **kw)
        out[name] = _residual_metrics(eo, x, xe, ph, w)

    rms_e, r4_e = out['explicit']
    rms_a, r4_a = out['auto']
    rms_n, r4_n = out['none']
    # explicit carrier is the reference truth: it matches the oracle tightly.
    assert r4_e < 0.03 and rms_e < 0.1, out
    # F1 fix: 'auto' recovers the SAME correction as explicit R ...
    assert abs(r4_a - r4_e) < 0.01, f"auto r4 {r4_a:.4f} != explicit {r4_e:.4f}"
    assert abs(rms_a - rms_e) < 0.05, f"auto rms {rms_a:.4f} != explicit {rms_e:.4f}"
    # ... and both are MUCH better than the no-carrier (plane-wave) reference,
    # which is the degraded state 'auto' used to silently fall back to.
    assert r4_n > 3 * r4_a, out
    assert rms_n > 3 * rms_a, out


# ---------------------------------------------------------------------------
# Collimated -> no carrier (byte-identical), both at the fit and end-to-end.
# ---------------------------------------------------------------------------
def test_collimated_auto_is_byte_identical_no_carrier():
    N, dx, w = 512, 8e-6, 1.2e-3
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    flat = np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)
    # (1) fit level: a flat (real) field fits W == 0 EXACTLY.
    W, _g, _w = _compute_carrier('auto', flat, LAM, dx, X, Y)
    assert np.array_equal(W, np.zeros_like(W))
    # a global phase is still flat -> W ~ 0 (below the engage floor).
    W2, _g2, _w2 = _compute_carrier('auto', flat * np.exp(1j * 0.9),
                                    LAM, dx, X, Y)
    assert float(np.abs(W2).max()) * K0 <= _TILT_EIKONAL_MIN_RAD
    # (2) end-to-end: carrier='auto' == carrier=None on a collimated input.
    if available_memory_bytes() < 1 * (1 << 30):
        pytest.skip("insufficient free RAM")
    rx = _singlet(40e-3, -40e-3, 3e-3, ap=4e-3)
    common = dict(prescription=rx, wavelength=LAM, dx=dx, ray_subsample=4,
                  n_workers=4, parallel_amp=False)
    e_auto = apply_real_lens_traced(flat, carrier='auto', **common)
    e_none = apply_real_lens_traced(flat, carrier=None, **common)
    scale = float(np.abs(e_none).max())
    assert float(np.abs(e_auto - e_none).max()) <= 1e-10 * scale


# ---------------------------------------------------------------------------
# Regression guard for the alias-trigger: a WELL-SAMPLED input keeps the full-
# support fit (the core restriction stays disengaged), so a multi-emitter field
# is NOT collapsed onto a single beamlet's connected component.
# ---------------------------------------------------------------------------
def test_wellsampled_multiemitter_not_collapsed():
    # Two well-separated, well-sampled Gaussian beamlets with OPPOSITE tilt.
    # The global 'auto' carrier must average them (~zero net tilt at centre);
    # collapsing onto one beamlet would return that beamlet's (nonzero) tilt.
    N, dx = 512, 6e-6
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    wl = 0.25e-3
    sep = 0.9e-3
    tilt = 0.01                       # << Nyquist tilt (well sampled)
    assert tilt < 0.5 * LAM / (2.0 * dx)
    g1 = np.exp(-((X - sep) ** 2 + Y ** 2) / wl ** 2) * np.exp(
        1j * K0 * (+tilt) * X)
    g2 = np.exp(-((X + sep) ** 2 + Y ** 2) / wl ** 2) * np.exp(
        1j * K0 * (-tilt) * X)
    E = (g1 + g2).astype(np.complex128)
    # trigger precondition: essentially nothing is aliased here.
    gphx = np.angle(np.roll(E, -1, 1) * np.conj(E)) / (K0 * dx)
    gphy = np.angle(np.roll(E, -1, 0) * np.conj(E)) / (K0 * dx)
    mask = np.abs(E) > 0.05 * np.abs(E).max()
    aliased = mask & (np.hypot(gphx, gphy) >= 0.5 * LAM / (2.0 * dx))
    assert int(aliased.sum()) <= _AUTO_CARRIER_ALIAS_FRAC * int(mask.sum())
    _W, grad_fn, _w = _compute_carrier('auto', E, LAM, dx, X, Y)
    lc, mc = grad_fn(np.array([0.0]), np.array([0.0]))
    # averaged (not collapsed onto a single +tilt or -tilt beamlet):
    assert abs(float(lc[0])) < 0.3 * tilt, (
        f"multi-emitter carrier collapsed: centre tilt {float(lc[0]):.4f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
