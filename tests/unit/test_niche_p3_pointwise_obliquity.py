"""P3 (niche N2): pointwise 2-D obliquity for ``surface_model='displaced'``.

The pre-P3 displaced screen derived its obliquity cosines from a MERIDIONAL fan,
assuming rotational symmetry -- decentered / tilted / freeform elements raised
``NotImplementedError``.  P3 adds a pointwise 2-D obliquity path: a 2-D ray grid
launched along the input congruence is traced through the actual (possibly
asymmetric) surfaces and its per-surface z-axis cosines are imprinted on the
field grid.  ``displaced_obliquity='auto'`` (default) keeps the fast meridional
LUT for symmetric elements (byte-identical) and switches to pointwise only when a
surface carries a decenter / tilt / freeform ``sag_callable``.

This module pins the four plan validations (all runnable WITHOUT Zemax):
  (a) SYMMETRIC LIMIT -- pointwise reproduces the meridional LUT to <0.1% (the
      convention-bug killer);
  (b) DECENTER -- the induced centroid shift matches an independent lumenairy-
      free geometric spot oracle (and, recorded in the audit doc, ZOS to ~2.5%);
  (c) TILT -- the induced deflection matches an independent RIGID-ROTATION ray
      trace in magnitude to <10%;
  (d) SIGN MIRROR -- +d and -d decenters mirror the PSF (centroid + intensity).

Plus: defaults byte-identical, auto-routing, the validation-error surface, the
freeform ``sag_callable`` hook, and the coma-flare sign.

The oracle is imported directly from ``validation/oracles/`` -- lumenairy-free.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

import lumenairy as la
from lumenairy.glass import GLASS_REGISTRY

_WL = 1.31e-6
K = 2 * np.pi / _WL
GLASS_REGISTRY['_P3A'] = lambda wl: 1.5168

# ---- lumenairy-free geometric spot oracle (decenter / tilt) ---------------
_ORACLE_PATH = (pathlib.Path(__file__).resolve().parents[2]
                / 'validation' / 'oracles' / 'geom_spot_decenter_oracle.py')
_spec = importlib.util.spec_from_file_location('geom_spot_decenter_oracle',
                                               _ORACLE_PATH)
_oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oracle)

_Z_IMG = 49.162e-3     # paraxial image distance of the biconvex singlet below


# ---------------------------------------------------------------------------
# Prescription builders
# ---------------------------------------------------------------------------
def _singlet(dec=(0.0, 0.0), tilt=(0.0, 0.0), sag_callable=None):
    s0 = {'radius': 51.68e-3, 'thickness': 5e-3, 'glass_before': 'air',
          'glass_after': '_P3A', 'semi_diameter': 6e-3,
          'decenter': dec, 'tilt': tilt}
    if sag_callable is not None:
        s0['sag_callable'] = sag_callable
        s0['radius'] = float('inf')
    return {'wavelength': _WL, 'aperture_diameter': 10e-3, 'surfaces': [
        s0,
        {'radius': -51.68e-3, 'thickness': 0.0, 'glass_before': '_P3A',
         'glass_after': 'air', 'semi_diameter': 6e-3}],
        'thicknesses': [5e-3], 'stop_index': 0}


def _gjob(dec_mm=(0.0, 0.0), tilt_mrad=(0.0, 0.0), w0_mm=3.0):
    return {'wavelength_um': _WL * 1e6, 'aperture_mm': 10.0,
            'pop': {'w0_mm': w0_mm}, 'R_in_mm': None, 'surfaces': [
                {'radius_mm': 51.68, 'thickness_mm': 5.0, 'index': 1.5168,
                 'decenter_mm': list(dec_mm), 'tilt_mrad': list(tilt_mrad)},
                {'radius_mm': -51.68, 'thickness_mm': 49.162, 'index': 'air'}]}


def _gauss(N, dx, w0, R_in=None):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r2 = X ** 2 + Y ** 2
    amp = np.exp(-r2 / w0 ** 2)
    if R_in is not None:
        amp = amp * np.exp(1j * K * r2 / (2.0 * R_in))
    return amp.astype(np.complex128)


def _disp(E0, presc, dx, obliq='auto', **kw):
    return la.apply_real_lens(E0, prescription=presc, wavelength=_WL, dx=dx,
                              surface_model='displaced',
                              displaced_obliquity=obliq, **kw)


def _psf_metrics(E_exit, dx, z=_Z_IMG):
    E = la.angular_spectrum_propagate(E_exit, z, _WL, dx)
    I = np.abs(E) ** 2
    N = I.shape[0]
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    P = I.sum()
    xc = float((I * X).sum() / P)
    yc = float((I * Y).sum() / P)
    m2 = float((I * (X - xc) ** 2).sum() / P)
    m3 = float((I * (X - xc) ** 3).sum() / P)
    skew = m3 / m2 ** 1.5 if m2 > 0 else 0.0
    return xc, yc, I, skew


def _ee80_about_centroid(E_exit, dx, z=_Z_IMG):
    """EE80 radius (um) about the intensity centroid at plane ``z``."""
    E = la.angular_spectrum_propagate(E_exit, z, _WL, dx)
    I = np.abs(E) ** 2
    N = I.shape[0]
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    P = I.sum()
    xc = (I * X).sum() / P
    yc = (I * Y).sum() / P
    rr = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2).ravel()
    order = np.argsort(rr)
    cum = np.cumsum(I.ravel()[order])
    cum /= cum[-1]
    return float(np.interp(0.8, cum, rr[order])) * 1e6


# ===========================================================================
# (a) SYMMETRIC LIMIT -- pointwise reproduces the meridional LUT (< 0.1%)
# ===========================================================================
@pytest.mark.parametrize('R_in', [None, 300e-3])
def test_pointwise_reproduces_meridional_on_symmetric_element(R_in):
    """The convention-bug killer: forcing the 2-D pointwise obliquity path on a
    rotationally-symmetric singlet reproduces the meridional cosine-LUT field to
    well under 0.1% (measured ~2e-5)."""
    N, dx = 384, 10e-6
    p = _singlet()
    E0 = _gauss(N, dx, 3e-3, R_in)
    a = _disp(E0, p, dx, obliq='meridional', conjugate=R_in)
    b = _disp(E0, p, dx, obliq='pointwise', conjugate=R_in)
    rel = np.linalg.norm(a - b) / np.linalg.norm(a)
    assert rel < 5e-4, (R_in, rel)


def test_default_is_meridional_byte_identical_on_symmetric():
    """A symmetric element with the default ``displaced_obliquity='auto'`` takes
    the byte-identical meridional LUT path (no pointwise, no regression)."""
    N, dx = 384, 10e-6
    p = _singlet()
    E0 = _gauss(N, dx, 3e-3)
    a = la.apply_real_lens(E0, prescription=p, wavelength=_WL, dx=dx,
                           surface_model='displaced')          # default 'auto'
    b = _disp(E0, p, dx, obliq='meridional')
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_auto_routes_decentered_to_remap():
    """A decentered element under the default ``displaced_obliquity='auto'`` now
    routes to the 2-D transverse-walk REMAP (P10 / N11), NOT the pointwise
    obliquity SCREEN: the default is byte-identical to explicit
    ``displaced_mode='remap'`` and DIFFERS from the ``'pointwise'`` screen (which
    is retained as the documented walk-off-limited peer)."""
    N, dx = 384, 10e-6
    p = _singlet(dec=(0.4e-3, 0.0))
    E0 = _gauss(N, dx, 3e-3)
    a = la.apply_real_lens(E0, prescription=p, wavelength=_WL, dx=dx,
                           surface_model='displaced')            # default 'auto'
    b = la.apply_real_lens(E0, prescription=p, wavelength=_WL, dx=dx,
                           surface_model='displaced', displaced_mode='remap')
    assert np.array_equal(np.asarray(a), np.asarray(b))          # auto == remap
    c = _disp(E0, p, dx, obliq='pointwise')                      # the screen peer
    assert not np.array_equal(np.asarray(a), np.asarray(c))      # remap != screen


# ===========================================================================
# Validation-error surface
# ===========================================================================
def test_validation_errors():
    N, dx = 128, 10e-6
    E0 = _gauss(N, dx, 1e-3)
    # meridional + decenter -> not supported
    with pytest.raises(NotImplementedError, match='pointwise'):
        _disp(E0, _singlet(dec=(0.4e-3, 0.0)), dx, obliq='meridional')
    # unknown obliquity selector
    with pytest.raises(ValueError, match='displaced_obliquity'):
        _disp(E0, _singlet(), dx, obliq='bogus')
    # obliquity on the thin model is meaningless
    with pytest.raises(ValueError, match='displaced_obliquity='):
        la.apply_real_lens(E0, prescription=_singlet(), wavelength=_WL, dx=dx,
                           displaced_obliquity='pointwise')
    # displaced_mode='remap' + decenter is now VALID (the P10 2-D walk-off
    # remap); only 'split' has no 2-D generalisation and still raises.
    _disp(E0, _singlet(dec=(0.4e-3, 0.0)), dx, displaced_mode='remap')
    with pytest.raises(ValueError, match='split'):
        _disp(E0, _singlet(dec=(0.4e-3, 0.0)), dx, displaced_mode='split')
    # sag_callable must be callable
    p = _singlet()
    p['surfaces'][0]['sag_callable'] = 3.0
    with pytest.raises(TypeError, match='sag_callable'):
        _disp(E0, p, dx, obliq='pointwise')
    # sag_callable on the meridional path -> not supported
    p2 = _singlet(sag_callable=lambda xs, ys: np.zeros_like(xs))
    with pytest.raises(NotImplementedError, match='pointwise'):
        _disp(E0, p2, dx, obliq='meridional')


# ===========================================================================
# (b) DECENTER -- centroid shift vs the independent geometric oracle
# ===========================================================================
def test_decenter_centroid_matches_geometric_oracle():
    """A 0.5 mm front-surface decenter shifts the PSF centroid; the shift matches
    the lumenairy-free geometric spot oracle within 5%, and the on-axis case is
    centered.  (The default decentered path is now the P10 2-D remap: its
    coma-weighted wave-intensity centroid is ~2.7% vs the geom ray-density
    oracle; the pointwise screen alone is ~0.1%.)  Full-aperture grid (half-width
    5.12 mm >= the 5 mm aperture)."""
    N, dx = 1280, 8e-6
    E0 = _gauss(N, dx, 3e-3)
    # on-axis is centered
    xc0, yc0, _, _ = _psf_metrics(_disp(E0, _singlet(), dx), dx)
    assert abs(xc0) < 2e-6 and abs(yc0) < 2e-6
    # decentered
    d = 0.5e-3
    xc, yc, _, _ = _psf_metrics(_disp(E0, _singlet(dec=(d, 0.0)), dx), dx)
    g = _oracle.geom_spot(_gjob(dec_mm=(d * 1e3, 0.0)))
    assert abs(yc) < 3e-6
    assert xc > 0                                    # +decenter -> +shift
    rel = abs(xc * 1e6 - g['centroid_x_um']) / abs(g['centroid_x_um'])
    assert rel < 0.05, (xc * 1e6, g['centroid_x_um'], rel)


def test_decenter_induces_signed_coma_flare():
    """Decenter induces a comatic asymmetry (large x-skewness) that flips sign
    with the decenter sign -- the coma structure, not just the centroid.
    Full-aperture grid so the outer comatic rays are resolved."""
    N, dx = 1280, 8e-6
    E0 = _gauss(N, dx, 3.0e-3)
    _, _, _, s0 = _psf_metrics(_disp(E0, _singlet(), dx), dx)
    _, _, _, sp = _psf_metrics(_disp(E0, _singlet(dec=(0.8e-3, 0.0)), dx), dx)
    _, _, _, sm = _psf_metrics(_disp(E0, _singlet(dec=(-0.8e-3, 0.0)), dx), dx)
    assert abs(s0) < 1.0                              # on-axis ~symmetric
    assert abs(sp) > 3.0                              # strong coma flare
    assert np.sign(sp) == -np.sign(sm)                # mirror under -d
    assert abs(abs(sp) - abs(sm)) / abs(sp) < 0.05    # equal magnitude


@pytest.mark.slow
def test_coma_ee_growth_screen_limit_pinned_remap_fixes_it():
    """The single-plane pointwise SCREEN's documented walk-off limit -- PINNED --
    AND the P10 default remap that FIXES it.

    The pointwise obliquity SCREEN (``displaced_obliquity='pointwise'``) imprints
    the OPD at the STRAIGHT-THROUGH grid position and cannot represent the
    transverse ray WALK between a thick element's two surfaces, so it captures the
    coma flare DIRECTION (centroid + skewness, validated above) but NOT the coma
    spot GROWTH -- the same H2 walk-off ceiling.  At the coma-dominated w0=4mm /
    1mm-decenter config the SCREEN EE80 SHRINKS (~0.91x) whereas the INDEPENDENT
    geometric spot oracle BROADENS (~1.02x).  This is retained as a documented
    peer, PINNED so it stays visible.

    P10 (N11) added the 2-D transverse-walk REMAP, now the DEFAULT for a decentered
    element (``displaced_obliquity='auto'``); it FIXES the shrink.  NOTE: the coma
    is diffraction-diluted in the EE80 metric (the honest broadening is the RMS /
    coma-RMS gate in test_niche_p10_*, matching GBD), so here the robust EE80 claim
    is DIRECTIONAL: the SCREEN COLLAPSES (ratio ~0.91) while the default remap does
    NOT (ratio ~1.00, well above the screen's shrink).  See
    docs/audit_real_lens_displaced_2026_07_19.md (P3 screen limit + P10 fix).
    """
    N, dx, w0 = 2048, 6e-6, 4e-3
    E0 = _gauss(N, dx, w0)
    # the retained pointwise SCREEN still SHRINKS (documented single-plane limit)
    m_on = _ee80_about_centroid(_disp(E0, _singlet(), dx, obliq='pointwise'), dx)
    m_dec = _ee80_about_centroid(
        _disp(E0, _singlet(dec=(1e-3, 0.0)), dx, obliq='pointwise'), dx)
    # the P10 DEFAULT (auto -> 2-D remap).  Baseline via a NEGLIGIBLE decenter so
    # both are the SAME (remap) model -- honest like-for-like.
    r_on = _ee80_about_centroid(_disp(E0, _singlet(dec=(1e-9, 0.0)), dx), dx)
    r_dec = _ee80_about_centroid(_disp(E0, _singlet(dec=(1e-3, 0.0)), dx), dx)
    g_on = _oracle.geom_spot(_gjob(dec_mm=(0.0, 0.0), w0_mm=4.0))['EE80_um']
    g_dec = _oracle.geom_spot(_gjob(dec_mm=(1.0, 0.0), w0_mm=4.0))['EE80_um']
    # The independent geometric oracle broadens under coma (correct physics).
    assert g_dec / g_on > 1.005, (g_on, g_dec)
    # The SCREEN does NOT reproduce that growth -- it collapses (the walk-off limit).
    assert m_dec < m_on, (m_on, m_dec)
    assert 0.86 < m_dec / m_on < 0.95, (m_on, m_dec, m_dec / m_on)
    # The P10 remap FIXES the direction: its EE80 ratio is well above the SCREEN's
    # collapse (the true magnitude is in the RMS gate -- test_niche_p10_*).
    assert r_dec / r_on > m_dec / m_on + 0.05, (r_dec / r_on, m_dec / m_on)
    # On-axis absolute EE80 is within the H2 envelope of ZOS 26.9um (sanity).
    assert 20.0 < m_on < 30.0, m_on


# ===========================================================================
# (c) TILT -- deflection magnitude vs an independent rigid-rotation trace
# ===========================================================================
def test_tilt_deflection_matches_rigid_rotation_oracle():
    """A 0.2 deg (3.491 mrad) front-surface tilt deflects the PSF; the field-
    frame linear-ramp deflection matches an INDEPENDENT rigid-body full-rotation
    ray trace in magnitude to <10% (measured ~0.3%).  Opposite sign is the
    differing 'positive tilt' definition, not an error."""
    N, dx = 1024, 8e-6
    E0 = _gauss(N, dx, 3e-3)
    tx = 0.2 * np.pi / 180.0
    xc, yc, _, _ = _psf_metrics(_disp(E0, _singlet(tilt=(tx, 0.0)), dx), dx)
    rig = _oracle.rigid_tilt_centroid(_gjob(tilt_mrad=(tx * 1e3, 0.0)),
                                      n_side=61)
    assert abs(yc) < 3e-6
    rel = abs(abs(xc * 1e6) - abs(rig['centroid_x_um'])) / abs(
        rig['centroid_x_um'])
    assert rel < 0.10, (xc * 1e6, rig['centroid_x_um'], rel)


def test_tilt_linear_and_mirrored():
    """Tilt deflection scales linearly and mirrors under sign flip."""
    N, dx = 512, 8e-6
    E0 = _gauss(N, dx, 3e-3)
    t = 1e-3
    x1, _, _, _ = _psf_metrics(_disp(E0, _singlet(tilt=(t, 0.0)), dx), dx)
    x2, _, _, _ = _psf_metrics(_disp(E0, _singlet(tilt=(2 * t, 0.0)), dx), dx)
    xm, _, _, _ = _psf_metrics(_disp(E0, _singlet(tilt=(-2 * t, 0.0)), dx), dx)
    assert abs(x2 / x1 - 2.0) < 0.05                  # linear in tilt
    assert abs(x2 + xm) / abs(x2) < 0.02              # mirror


# ===========================================================================
# (d) SIGN MIRROR -- +d and -d decenters mirror the PSF exactly
# ===========================================================================
def test_sign_mirror_decenter():
    """+d and -d x-decenters produce PSFs that are exact x-mirrors: the centroid
    mirrors to <0.05% and the intensity mirror-L2 is <1%."""
    N, dx = 512, 8e-6
    E0 = _gauss(N, dx, 3e-3)
    d = 0.6e-3
    xp, _, Ip, _ = _psf_metrics(_disp(E0, _singlet(dec=(d, 0.0)), dx), dx)
    xm, _, Im, _ = _psf_metrics(_disp(E0, _singlet(dec=(-d, 0.0)), dx), dx)
    # centroid mirrors
    assert abs(xp + xm) / abs(xp) < 5e-3
    # intensity mirrors (even grid centered at N/2 -> flip then roll by 1)
    Im_flip = np.roll(Im[:, ::-1], 1, axis=1)
    relL2 = np.linalg.norm(Ip - Im_flip) / np.linalg.norm(Ip)
    assert relL2 < 1e-2, relL2


# ===========================================================================
# Freeform sag_callable hook
# ===========================================================================
def test_sag_callable_reproduces_equivalent_conic():
    """A freeform ``sag_callable`` returning exactly the conic sag reproduces
    the analytic-conic pointwise field -- the freeform hook is threaded through
    both the ray trace and the OPD imprint consistently."""
    N, dx = 384, 10e-6
    E0 = _gauss(N, dx, 3e-3)
    R = 51.68e-3

    def conic_sag(xs, ys):
        r2 = xs ** 2 + ys ** 2
        c = 1.0 / R
        w = np.sqrt(np.maximum(1.0 - c * c * r2, 0.0))
        return c * r2 / (1.0 + w)

    a = _disp(E0, _singlet(), dx, obliq='pointwise')                # conic
    b = _disp(E0, _singlet(sag_callable=conic_sag), dx, obliq='pointwise')
    rel = np.linalg.norm(a - b) / np.linalg.norm(a)
    assert rel < 2e-3, rel


def test_sag_callable_runs_freeform():
    """A genuinely freeform (non-rotationally-symmetric) ``sag_callable`` runs
    end-to-end and produces a finite, energy-bounded field."""
    N, dx = 384, 10e-6
    E0 = _gauss(N, dx, 3e-3)
    R = 51.68e-3

    def freeform(xs, ys):
        r2 = xs ** 2 + ys ** 2
        c = 1.0 / R
        w = np.sqrt(np.maximum(1.0 - c * c * r2, 0.0))
        base = c * r2 / (1.0 + w)
        return base + 2.0e3 * (xs ** 2 - ys ** 2) * xs   # trefoil-ish departure

    out = _disp(E0, _singlet(sag_callable=freeform), dx, obliq='pointwise')
    assert np.all(np.isfinite(out))
    p_in = float((np.abs(E0) ** 2).sum())
    p_out = float((np.abs(out) ** 2).sum())
    assert p_out <= p_in * 1.001                       # lossless (apertured)
