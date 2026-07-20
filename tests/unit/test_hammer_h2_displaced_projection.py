"""Hammer-audit H2(a) (2026-07-19): ray-angle-aware ``surface_model='displaced'``.

The default per-surface phase screen imprints the paraxial thin-element OPD
``(n2 - n1) * sag(r)`` at the transverse coordinate ``r``, blind to the
INCOMING RAY ANGLE.  On a plano-convex singlet that OPD is orientation-
INVARIANT -- the curved-first ("good") and flat-first ("bad") prescriptions
imprint the identical ``(n_glass - 1)|sag(r)|`` map -- so the analytic model
cannot reproduce the textbook ~4x spherical-aberration penalty between the
two orientations (dual-oracle r2m 43.2 vs 127.6 um; the thin screen gives
60.4/60.9).

``surface_model='displaced'`` replaces the screen with the eikonal-correct
piston OPD ``(n2 cos(alpha_out) - n1 cos(alpha_in)) * sag(r)``, cosines of the
TRUE ray angle to the z-axis from a collimated meridional ray fan.  The
cosines carry the incoming-ray-angle obliquity and BREAK the orientation
symmetry, matching both hammer-campaign oracles.

Oracle provenance (2026-07-19 hammer campaign, dual-oracle: Zemax
OpticStudio POP via ZOS-API + an independent grid-free Debye/Huygens
integral, agreeing 0.5-8%; scripts in the session scratchpad
debye_oracle.py / zos_oracle.py; lambda=1.31 um, n=1.5168, w0=5 mm
collimated):

  * f/5 biconvex R=+/-51.68 mm, t=5 mm, image 49.163 mm:
        r2m = 64.98 um (Debye) / 64.7 um (POP); EE50=15.17, EE80=55.22.
  * plano-convex "good" (curved R=+26.71 toward source), image 48.372 mm:
        r2m = 43.2 um.
  * plano-convex "bad" (flat toward source, curved R=-26.71 toward image),
        image 51.669 mm: r2m = 127.6 um.

SAMPLING NOTE: the exit converging wavefront (NA_exit~0.24 on the f/5 case)
must be Nyquist-sampled -- dx <= lambda/(2 NA_exit) ~ 2.7 um -- or the
windowed r2m ALIASES LOW (the same exit-NA rule as traced, hammer H3).  The
~40 um analytic "plateau" the 2026-07-18 audit reported for this class was a
dx=6 um undersampling artefact (traced itself reads 40.9 um there, 64.8 um at
dx<=3 um), NOT a model floor -- so the converged validations below run at
N=8192 / dx=3 um.  The fast tests use a reduced-NA config that IS Nyquist-ok
at dx=5 um and cross-check against the independent traced propagator.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy.glass import GLASS_REGISTRY

_WL = 1.31e-6
GLASS_REGISTRY['_H2_DISP_GLASS'] = lambda wl: 1.5168
_G = '_H2_DISP_GLASS'


# ---------------------------------------------------------------------------
# Prescriptions
# ---------------------------------------------------------------------------
def _biconvex_f5():
    return {'wavelength': _WL, 'aperture_diameter': 24e-3,
            'surfaces': [
                {'radius': 51.68e-3, 'thickness': 5e-3, 'glass_before': 'air',
                 'glass_after': _G, 'semi_diameter': 12e-3},
                {'radius': -51.68e-3, 'thickness': 0.0, 'glass_before': _G,
                 'glass_after': 'air', 'semi_diameter': 12e-3}],
            'thicknesses': [5e-3], 'stop_index': 0}


def _pcx(good, aperture=24e-3, semi=12e-3):
    """Plano-convex singlet.  good=True -> curved (R=+26.71) toward source
    (low SA); good=False -> flat toward source, curved (R=-26.71) toward
    image (~4x SA)."""
    if good:
        surfs = [
            {'radius': 26.71e-3, 'thickness': 5e-3, 'glass_before': 'air',
             'glass_after': _G, 'semi_diameter': semi},
            {'radius': np.inf, 'thickness': 0.0, 'glass_before': _G,
             'glass_after': 'air', 'semi_diameter': semi}]
    else:
        surfs = [
            {'radius': np.inf, 'thickness': 5e-3, 'glass_before': 'air',
             'glass_after': _G, 'semi_diameter': semi},
            {'radius': -26.71e-3, 'thickness': 0.0, 'glass_before': _G,
             'glass_after': 'air', 'semi_diameter': semi}]
    return {'wavelength': _WL, 'aperture_diameter': aperture,
            'surfaces': surfs, 'thicknesses': [5e-3], 'stop_index': 0}


# image distances (behind exit vertex) for the plano-convex conjugates
_PCX_ZIMG = {True: 48.372e-3, False: 51.669e-3}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _gauss(N, dx, w0):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return x, np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


def _r2m(E_exit, dx, z_img, win=500e-6):
    """fga_f5-style windowed second-moment radius [m] at the image plane."""
    E = la.angular_spectrum_propagate(E_exit, z_img, _WL, dx)
    I = np.abs(E) ** 2
    N = I.shape[0]
    x = (np.arange(N) - N / 2) * dx
    j, i = np.unravel_index(np.argmax(I), I.shape)
    X, Y = np.meshgrid(x - x[i], x - x[j])
    r = np.sqrt(X ** 2 + Y ** 2)
    m = r <= win
    return float(np.sqrt((I * r ** 2)[m].sum() / I[m].sum()))


def _ee(E_exit, dx, z_img, frac, win=500e-6):
    E = la.angular_spectrum_propagate(E_exit, z_img, _WL, dx)
    I = np.abs(E) ** 2
    N = I.shape[0]
    x = (np.arange(N) - N / 2) * dx
    j, i = np.unravel_index(np.argmax(I), I.shape)
    X, Y = np.meshgrid(x - x[i], x - x[j])
    r = np.sqrt(X ** 2 + Y ** 2)
    rb = np.linspace(0, win, 251)
    cum = np.array([I[r <= rr].sum() for rr in rb]) / I.sum()
    return float(np.interp(frac, cum, rb))


# ===========================================================================
# FAST tests
# ===========================================================================
def test_default_path_is_byte_identical_to_thin():
    """DEFAULT UNCHANGED: omitting surface_model (or 'thin') must be
    byte-for-byte identical to prior releases -- pins the no-regression
    contract for the shipped default."""
    N, dx = 1024, 8e-6
    _, E0 = _gauss(N, dx, 0.5e-3)
    p = _biconvex_f5()
    a = la.apply_real_lens(E0, prescription=p, wavelength=_WL, dx=dx)
    b = la.apply_real_lens(E0, prescription=p, wavelength=_WL, dx=dx,
                           surface_model='thin')
    assert np.array_equal(a, b), (
        f"surface_model='thin' must equal the default byte-for-byte; "
        f"max|diff|={np.abs(a - b).max()}")


def test_thin_screen_cannot_split_pcx_orientation():
    """FAIL-BEFORE anchor: the paraxial thin screen imprints an
    orientation-INVARIANT OPD, so the good and bad plano-convex
    orientations give ~equal r2m (the structural defect H2 quantified:
    60.4/60.9 vs the dual-oracle 43.2/127.6).  This is what 'displaced'
    fixes."""
    N, dx = 2048, 5e-6
    _, E0 = _gauss(N, dx, 2e-3)
    r = {g: _r2m(la.apply_real_lens(E0, prescription=_pcx(g, aperture=10e-3,
                                                          semi=6e-3),
                                    wavelength=_WL, dx=dx,
                                    surface_model='thin'),
                 dx, _PCX_ZIMG[g], win=200e-6) for g in (True, False)}
    ratio = r[True] / r[False]
    assert ratio > 0.9, (
        f"thin screen should NOT split orientation (documented defect): "
        f"good={r[True]*1e6:.2f} bad={r[False]*1e6:.2f} ratio={ratio:.3f}")


def test_displaced_splits_pcx_and_matches_traced_propagator():
    """PASS-AFTER + INDEPENDENT ORACLE: at Nyquist-compliant sampling the
    'displaced' model splits the plano-convex orientations (good < bad) and
    agrees with the fully-independent per-pixel ray-traced propagator
    (apply_real_lens_traced) to within a few percent on BOTH orientations.
    Reduced-NA config (w0=2 mm, 10 mm aperture) so dx=5 um is Nyquist-ok."""
    N, dx = 2048, 5e-6
    _, E0 = _gauss(N, dx, 2e-3)
    disp, trac = {}, {}
    for g in (True, False):
        p = _pcx(g, aperture=10e-3, semi=6e-3)
        z = _PCX_ZIMG[g]
        disp[g] = _r2m(la.apply_real_lens(E0, prescription=p, wavelength=_WL,
                                          dx=dx, surface_model='displaced'),
                       dx, z, win=200e-6)
        trac[g] = _r2m(la.apply_real_lens_traced(E0, prescription=p,
                                                 wavelength=_WL, dx=dx,
                                                 on_undersample='silent'),
                       dx, z, win=200e-6)
    # orientation split in the correct direction
    assert disp[True] < 0.8 * disp[False], (
        f"displaced must split orientation good<bad: good={disp[True]*1e6:.2f} "
        f"bad={disp[False]*1e6:.2f}")
    # independent-propagator agreement (traced), both orientations
    for g in (True, False):
        rel = abs(disp[g] - trac[g]) / trac[g]
        assert rel < 0.06, (
            f"displaced vs traced ({'good' if g else 'bad'}): "
            f"{disp[g]*1e6:.2f} vs {trac[g]*1e6:.2f} um (rel {rel:.3f})")


def test_displaced_benign_regime_matches_thin_and_oracle():
    """The ray-angle correction scales as the beam obliquity (~r^4 SA) and
    VANISHES for a near-axial (benign) beam: a w0=0.5 mm beam through the f/5
    singlet must stay at the thin-model / Zemax value (~29.98 um) and not
    regress.  Independent oracle: ZOS POP 29.98 um for this config."""
    N, dx = 1024, 8e-6
    _, E0 = _gauss(N, dx, 0.5e-3)
    p = _biconvex_f5()
    z = 49.163e-3
    r_thin = _r2m(la.apply_real_lens(E0, prescription=p, wavelength=_WL,
                                     dx=dx, surface_model='thin'), dx, z)
    r_disp = _r2m(la.apply_real_lens(E0, prescription=p, wavelength=_WL,
                                     dx=dx, surface_model='displaced'), dx, z)
    assert abs(r_disp - r_thin) / r_thin < 0.03, (
        f"benign: displaced {r_disp*1e6:.3f} must ~= thin {r_thin*1e6:.3f} um")
    assert abs(r_disp - 29.98e-6) / 29.98e-6 < 0.10


def test_displaced_rejects_unsupported_combinations():
    """The displaced model supplies its own ray-angle OPD and a
    rotationally-symmetric collimated fan; incompatible modifiers /
    prescriptions must raise loudly, not silently mis-model."""
    N, dx = 256, 8e-6
    _, E0 = _gauss(N, dx, 0.5e-3)
    p = _biconvex_f5()
    for kw in ({'slant_correction': True}, {'fresnel': True},
               {'seidel_correction': True}, {'absorption': True},
               {'surface_frame': True}):
        with pytest.raises(ValueError):
            la.apply_real_lens(E0, prescription=p, wavelength=_WL, dx=dx,
                               surface_model='displaced', **kw)
    with pytest.raises(ValueError):
        la.apply_real_lens(E0, prescription=p, wavelength=_WL, dx=dx,
                           surface_model='displaced', wave_propagator='fresnel')
    with pytest.raises(ValueError):
        la.apply_real_lens(E0, prescription=p, wavelength=_WL, dx=dx,
                           surface_model='not_a_model')
    # biconic radius_y -> NotImplementedError (unsupported on both paths)
    p_bic = _biconvex_f5()
    p_bic['surfaces'][0]['radius_y'] = 60e-3
    with pytest.raises(NotImplementedError):
        la.apply_real_lens(E0, prescription=p_bic, wavelength=_WL, dx=dx,
                           surface_model='displaced')
    # decenter: the MERIDIONAL (rotationally-symmetric) fan cannot represent it
    # and raises; the default 'auto' now routes it to the P10 2-D transverse-walk
    # remap (test_niche_p10_*) -- the pointwise screen is the opt-in peer
    # (test_niche_p3_*) -- so it must NOT raise there.
    p_dec = _biconvex_f5()
    p_dec['surfaces'][0]['decenter'] = (1e-3, 0.0)
    with pytest.raises(NotImplementedError):
        la.apply_real_lens(E0, prescription=p_dec, wavelength=_WL, dx=dx,
                           surface_model='displaced',
                           displaced_obliquity='meridional')
    out = la.apply_real_lens(E0, prescription=p_dec, wavelength=_WL, dx=dx,
                             surface_model='displaced')      # 'auto' -> 2-D remap
    assert np.all(np.isfinite(out))


# ===========================================================================
# SLOW tests (converged, Nyquist-compliant dx=3 um) -- headline dual-oracle
# ===========================================================================
@pytest.mark.slow      # N=8192 whole-grid (exit-NA Nyquist sampling)
def test_displaced_matches_dual_oracle_f5_converged():
    """HEADLINE: at converged dx=3 um the 'displaced' f/5 biconvex r2m lands
    within ~10% of the dual-oracle 64.98 um (task window 58-72 um) -- and the
    EE50/EE80 profile matches too (15.17/55.22 um).  The thin/slant screens
    plateau at 25/50 um; the walk-off-only path (rejected) over-corrects to
    92 um.  Oracle: grid-free Debye integral 64.98 um (see module docstring)."""
    N, dx = 8192, 3e-6
    _, E0 = _gauss(N, dx, 5e-3)
    p = _biconvex_f5()
    z = 49.163e-3
    E_exit = la.apply_real_lens(E0, prescription=p, wavelength=_WL, dx=dx,
                                surface_model='displaced')
    r2m = _r2m(E_exit, dx, z)
    assert 58e-6 < r2m < 72e-6, (
        f"displaced f/5 r2m={r2m*1e6:.2f} um outside oracle window [58,72] "
        f"(Debye 64.98)")
    e50 = _ee(E_exit, dx, z, 0.5)
    e80 = _ee(E_exit, dx, z, 0.8)
    assert abs(e50 - 15.17e-6) < 4e-6, f"EE50={e50*1e6:.2f} vs oracle 15.17"
    assert abs(e80 - 55.22e-6) < 8e-6, f"EE80={e80*1e6:.2f} vs oracle 55.22"


@pytest.mark.slow      # N=8192 whole-grid x2 orientations
def test_displaced_matches_dual_oracle_pcx_split_converged():
    """HEADLINE: at converged dx=3 um the 'displaced' model reproduces the
    plano-convex orientation split -- good ~42 um (oracle 43.2), bad ~127 um
    (oracle 127.6), ratio ~0.33 vs the oracle 0.339.  The thin screen gives
    ratio ~0.99 (no split)."""
    N, dx = 8192, 3e-6
    _, E0 = _gauss(N, dx, 5e-3)
    r = {}
    for g in (True, False):
        p = _pcx(g)
        E_exit = la.apply_real_lens(E0, prescription=p, wavelength=_WL, dx=dx,
                                    surface_model='displaced')
        r[g] = _r2m(E_exit, dx, _PCX_ZIMG[g])
    assert 36e-6 < r[True] < 50e-6, f"PCX good r2m={r[True]*1e6:.2f} vs 43.2"
    assert 110e-6 < r[False] < 145e-6, f"PCX bad r2m={r[False]*1e6:.2f} vs 127.6"
    ratio = r[True] / r[False]
    assert 0.28 < ratio < 0.42, (
        f"split ratio {ratio:.3f} vs oracle 0.339 (thin ~0.99)")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
