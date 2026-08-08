"""P11 (niche N12): opt-in ray-density (Jacobian) amplitude for the traced lens.

``apply_real_lens_traced(amplitude_model='ray_density')`` replaces the historical
hybrid exit magnitude (``|E_analytic|`` from a single ``apply_real_lens`` ASM
call) with the geometric ray-tube energy-conserving amplitude
``|E_in(x_in)| / sqrt(|det J|)``, ``J = d(x_out,y_out)/d(x_in,y_in)`` the exit
ray-map Jacobian (analytic gradient of the entrance->exit fit), placed at the
exit ray position with the traced OPL phase.  Default ``'screen'`` is
byte-identical to prior releases.

WHAT THIS PHASE ESTABLISHED (measured, adversarial):

* The ray-density amplitude is a valid, ENERGY-CONSERVING (< 0.5% away from
  folds) geometric ray-tube amplitude, and it is DECENTER-STABLE where the
  screen ``apply_real_lens`` amplitude leaks energy (~9% at a 2 mm decenter).
* It is CAUSTIC-SAFE: ``det J -> 0`` at a fold is detected (absolute floor +
  |det J| dynamic-range + adjacent sign change), the amplitude is CAPPED
  (never inf/nan), and a one-time ``RuntimeWarning`` steers to GBD/FGA.  On the
  ``caustic_fold`` ground truth (fold DOWNSTREAM of the traced exit vertex) the
  exit-vertex ray-density field ASM-propagated to the fold plane matches the
  reference within ~3%.
* HONEST LIMIT on decentered coma (the N12 premise, REFUTED for the traced
  OUTPUT plane).  The traced output is the exit VERTEX, where the ray map is
  nearly the identity (``det J`` median ~0.93, spread ~0.7% -> amplitude
  modulation ~0.3%): the coma redistribution is a DOWNSTREAM (propagation-to-
  image) effect carried by the PHASE, and at the image plane ``det J -> 0`` (a
  caustic) where single-branch ray density is unreliable (flagged).  So
  ``ray_density`` broadens the decentered spot only marginally (RMS ~1.03-1.05,
  killing any shrink) and TRACKS the screen -- it does NOT make the focal-plane
  decentered PSF match the geometric oracle to 15%.  ``apply_real_lens_gbd``
  (N10b), whose beamlets carry the image-plane ray density, remains the
  decentered-coma reference.  This test PINS that limit.

All gates run WITHOUT Zemax (geometric + Debye lumenairy-free oracles + the
caustic_fold ground-truth npz); the ZOS numbers live in the audit doc.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import warnings

import numpy as np
import pytest

import lumenairy as la

# Model glass for THIS module only: registered and removed by
# tests/conftest.py::_module_glass_registry_guard.
MODULE_GLASSES = {'_P11A': lambda wl: 1.5168,
                  '_P11SLOW': lambda wl: 1.5168,
                  '_P11CAU': lambda wl: 1.5168}

_WL = 1.31e-6
_Z_IMG = 49.162e-3     # f/5 singlet paraxial image distance (exit vertex -> image)

# ---- lumenairy-free oracles -----------------------------------------------
_ROOT = pathlib.Path(__file__).resolve().parents[2] / 'validation' / 'oracles'


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_debye = _load('debye_oracle_v3')


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _singlet(dec=None):
    s0 = {'radius': 51.68e-3, 'thickness': 5e-3, 'glass_before': 'air',
          'glass_after': '_P11A', 'semi_diameter': 6e-3}
    if dec is not None:
        s0['decenter'] = dec
    return {'wavelength': _WL, 'aperture_diameter': 10e-3, 'surfaces': [
        s0,
        {'radius': -51.68e-3, 'thickness': 0.0, 'glass_before': '_P11A',
         'glass_after': 'air', 'semi_diameter': 6e-3}],
        'thicknesses': [5e-3], 'stop_index': 0}


def _singlet_to_focus(dec=None):
    """Singlet + a flat dummy surface AT the paraxial image, so the traced
    OUTPUT plane IS the (near-caustic) focus -- the caustic-at-output stress."""
    s0 = {'radius': 51.68e-3, 'thickness': 5e-3, 'glass_before': 'air',
          'glass_after': '_P11A', 'semi_diameter': 6e-3}
    if dec is not None:
        s0['decenter'] = dec
    return {'wavelength': _WL, 'aperture_diameter': 10e-3, 'surfaces': [
        s0,
        {'radius': -51.68e-3, 'thickness': _Z_IMG, 'glass_before': '_P11A',
         'glass_after': 'air', 'semi_diameter': 6e-3},
        {'radius': float('inf'), 'thickness': 0.0, 'glass_before': 'air',
         'glass_after': 'air', 'semi_diameter': 80e-3}],
        'thicknesses': [5e-3, _Z_IMG], 'stop_index': 0}


def _gjob(dec_mm=(0.0, 0.0), w0_mm=4.0):
    return {'wavelength_um': _WL * 1e6, 'aperture_mm': 10.0,
            'pop': {'w0_mm': w0_mm}, 'R_in_mm': None, 'window_um': 200.0,
            'surfaces': [
                {'radius_mm': 51.68, 'thickness_mm': 5.0, 'index': 1.5168,
                 'decenter_mm': list(dec_mm), 'tilt_mrad': [0.0, 0.0]},
                {'radius_mm': -51.68, 'thickness_mm': 49.162, 'index': 'air'}]}


def _gauss(N, dx, w0):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


def _rlt(E0, p, dx, model='screen', **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens_traced(
            E0, prescription=p, wavelength=_WL, dx=dx,
            on_undersample='silent', amplitude_model=model, **kw))


def _img(E_exit, dx, z=_Z_IMG, win=200e-6):
    """Image-plane metrics about the intensity centroid (ASM to z)."""
    E = la.angular_spectrum_propagate(E_exit.astype(np.complex128), z, _WL, dx)
    I = np.abs(E) ** 2
    N = I.shape[0]
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    P = I.sum()
    xc = (I * X).sum() / P
    yc = (I * Y).sum() / P
    r = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2)
    m = r <= win
    rms = np.sqrt((I[m] * r[m] ** 2).sum() / I[m].sum()) * 1e6
    rr = r.ravel()
    o = np.argsort(rr)
    cum = np.cumsum(I.ravel()[o])
    cum /= cum[-1]
    ee80 = float(np.interp(0.8, cum, rr[o])) * 1e6
    ee50 = float(np.interp(0.5, cum, rr[o])) * 1e6
    return {'rms': rms, 'ee80': ee80, 'ee50': ee50,
            'xc': float(xc * 1e6), 'yc': float(yc * 1e6), 'I': I}


# ===========================================================================
# DEFAULT byte-identical (the regression pin)
# ===========================================================================
def test_default_amplitude_model_is_screen_byte_identical():
    """``amplitude_model`` defaults to ``'screen'`` (the same model as before this
    feature); ``'ray_density'`` is a genuinely different field.

    The default and explicit ``'screen'`` calls are the SAME numerical model, so
    they agree to machine precision.  They are bit-for-bit identical in a
    warm-cache process, but two separate traced calls can float at the ~1 ULP
    level by cache-cold-vs-warm reduction order depending on what ran earlier in
    the process (the same cross-test sensitivity documented for the displaced
    zero-decenter pin); assert numerical identity, not bit-identity.  The
    campaign verifier separately confirmed the default is byte-identical to the
    pre-P11 committed result (max|diff| = 0.0)."""
    N, dx = 512, 8e-6
    E0 = _gauss(N, dx, 4e-3)
    default = _rlt(E0, _singlet(), dx)
    screen = _rlt(E0, _singlet(), dx, 'screen')
    rd = _rlt(E0, _singlet(), dx, 'ray_density')
    assert np.max(np.abs(default - screen)) <= 1e-10 * float(np.max(np.abs(screen)))
    assert not np.array_equal(default, rd)


def test_ray_density_finite_and_preserves_complex_dtype():
    """The ray-density field is finite (no inf/nan) and preserves the caller's
    complex dtype (complex64 stays complex64)."""
    N, dx = 384, 8e-6
    for dt in (np.complex128, np.complex64):
        E0 = _gauss(N, dx, 3e-3).astype(dt)
        out = _rlt(E0, _singlet(), dx, 'ray_density')
        assert out.dtype == dt
        assert np.all(np.isfinite(out))


def test_invalid_and_forbidden_combinations_raise():
    """``amplitude_model`` validation + the documented incompatibilities."""
    N, dx = 128, 10e-6
    E0 = _gauss(N, dx, 1e-3)
    with pytest.raises(ValueError, match='amplitude_model'):
        _rlt(E0, _singlet(), dx, 'bogus')
    with pytest.raises(ValueError, match='return_screen'):
        _rlt(E0, _singlet(), dx, 'ray_density', return_screen=True)
    with pytest.raises(ValueError, match='inversion_method'):
        _rlt(E0, _singlet(), dx, 'ray_density', inversion_method='fit')


# ===========================================================================
# ENERGY closure (< 0.5% away from folds) + decenter stability
# ===========================================================================
def test_energy_conservation_and_decenter_stability():
    """The sqrt(|det J|) weighting is energy-exact in the geometric limit: the
    ray-density power equals the aperture-transmitted input power to < 0.5% at
    the (fold-free) exit vertex, and -- unlike the screen ``apply_real_lens``
    amplitude, which leaks energy for a decentered element -- it is STABLE under
    decenter (no silent renormalisation)."""
    N, dx = 1280, 8e-6                       # hw 5.12 mm holds the 5 mm aperture
    E0 = _gauss(N, dx, 4e-3)
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    pin = float((np.abs(E0[(X ** 2 + Y ** 2) <= (5e-3) ** 2]) ** 2).sum())
    for dec in (None, (2e-3, 0.0)):
        rd = _rlt(E0, _singlet(dec), dx, 'ray_density')
        assert abs(float((np.abs(rd) ** 2).sum()) / pin - 1.0) < 5e-3, dec
    # contrast: the screen amplitude LEAKS energy for the decentered element.
    sc2 = _rlt(E0, _singlet((2e-3, 0.0)), dx, 'screen')
    assert float((np.abs(sc2) ** 2).sum()) / pin < 0.97       # screen loses > 3%


# ===========================================================================
# COLLIMATED unaberrated -- Jacobian ~ const -> ray_density ~ screen (Airy)
# ===========================================================================
def test_collimated_slow_lens_tracks_screen():
    """On a slow, near-unaberrated singlet the exit ray map is nearly a uniform
    magnification (det J ~ const), so the ray-density amplitude reduces to a
    scaled input envelope and reproduces the screen (Airy-limited) field over
    the bright support -- the sanity limit."""
    N, dx = 1024, 4e-6
    slow = {'wavelength': _WL, 'aperture_diameter': 4e-3, 'surfaces': [
        {'radius': 103.36e-3, 'thickness': 3e-3, 'glass_before': 'air',
         'glass_after': '_P11SLOW', 'semi_diameter': 2.5e-3},
        {'radius': -103.36e-3, 'thickness': 0.0, 'glass_before': '_P11SLOW',
         'glass_after': 'air', 'semi_diameter': 2.5e-3}],
        'thicknesses': [3e-3], 'stop_index': 0}
    E0 = _gauss(N, dx, 1.2e-3)
    rd = np.abs(_rlt(E0, slow, dx, 'ray_density'))
    sc = np.abs(_rlt(E0, slow, dx, 'screen'))
    m = sc > 0.02 * sc.max()
    assert np.abs(rd[m] - sc[m]).max() / sc[m].max() < 0.10


# ===========================================================================
# SIGN-MIRROR (+d / -d)
# ===========================================================================
def test_ray_density_sign_mirror_decenter():
    """+d and -d decenters produce mirror-image ray-density PSFs: the centroid
    mirrors (< 1%) and the intensity mirror-L2 is < 3% (the robust P9-style
    metric; the single-radius EE80 threshold crossing is quantization-noisy on a
    few-pixel spot and is not gated)."""
    N, dx = 512, 8e-6
    E0 = _gauss(N, dx, 3e-3)
    d = 0.6e-3
    mp = _img(_rlt(E0, _singlet((d, 0.0)), dx, 'ray_density'), dx)
    mm = _img(_rlt(E0, _singlet((-d, 0.0)), dx, 'ray_density'), dx)
    assert abs(mp['xc'] + mm['xc']) / abs(mp['xc']) < 1e-2
    Im_flip = np.roll(mm['I'][:, ::-1], 1, axis=1)
    assert np.linalg.norm(mp['I'] - Im_flip) / np.linalg.norm(mp['I']) < 3e-2


# ===========================================================================
# ON-AXIS aberrated -- oracle-consistent with the Debye diffraction oracle
# ===========================================================================
def test_on_axis_aberrated_oracle_consistent_with_debye():
    """For the on-axis f/5 singlet the ray-density amplitude is oracle-CONSISTENT
    with the lumenairy-free Debye diffraction oracle: it tracks the validated
    screen leg (det J ~ const at the exit vertex) and, like screen, trends toward
    the Debye EE as the exit-NA-Nyquist sampling tightens (H3) -- it is not worse
    than the incumbent.  (The absolute EE is exit-NA-sampling-limited at this
    modest dx; both models under-read the SA halo identically.)"""
    N, dx = 2048, 3e-6                        # dx <~ exit-NA Nyquist (~6.5 um)
    E0 = _gauss(N, dx, 4e-3)
    rd = _img(_rlt(E0, _singlet(), dx, 'ray_density'), dx)
    sc = _img(_rlt(E0, _singlet(), dx, 'screen'), dx)
    obj = _debye.evaluate(_gjob())
    # ray_density tracks the validated screen leg to a few percent.
    assert abs(rd['ee50'] / sc['ee50'] - 1.0) < 0.05
    assert abs(rd['ee80'] / sc['ee80'] - 1.0) < 0.05
    # both are the same fraction of the Debye diffraction EE (H3-sampling-limited
    # low at this dx, converging up as dx tightens -- documented).
    assert 0.4 < rd['ee80'] / obj['huy_EE80_um'] < 1.15


# ===========================================================================
# CAUSTIC -- fold DOWNSTREAM of the exit vertex: no blow-up, matches the
# ground-truth field after ASM to the fold plane (validation d).
# ===========================================================================
@pytest.mark.slow
def test_caustic_fold_no_blowup_matches_reference():
    """On the ``caustic_fold`` ground truth the geometric ray map FOLDS at a
    through-focus plane 4.37 mm DOWNSTREAM of the traced exit vertex.  At the
    exit vertex itself the map is single-valued (no fold warning), so the
    ray-density exit field is well-conditioned and FINITE; ASM-propagating it to
    the fold plane -- a wave method that handles the multi-valued caustic --
    reproduces the dense direct-RS reference (r2m / EE50 / EE80) within ~8%, with
    NO inf/nan."""
    ref = np.load(_ROOT / 'caustic_fold_ref.npz', allow_pickle=True)
    meta = json.loads(str(ref['metrics']))
    Nc, dxc = int(ref['N']), float(ref['dx'])          # 1280, 2 um
    opd = float(ref['output_plane_distance'])          # 4.3704 mm vertex->fold
    caustic_p = {'wavelength': _WL, 'aperture_diameter': 1.4e-3, 'surfaces': [
        {'radius': 2.7e-3, 'thickness': 1.0e-3, 'glass_before': 'air',
         'glass_after': '_P11CAU', 'semi_diameter': 0.75e-3},
        {'radius': float('inf'), 'thickness': 0.0, 'glass_before': '_P11CAU',
         'glass_after': 'air', 'semi_diameter': 0.75e-3}],
        'thicknesses': [1.0e-3], 'stop_index': 0}
    Ec = _gauss(Nc, dxc, 0.55e-3)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        E_exit = np.asarray(la.apply_real_lens_traced(
            Ec, prescription=caustic_p, wavelength=_WL, dx=dxc,
            on_undersample='silent', amplitude_model='ray_density'))
        # the exit vertex is UPSTREAM of the fold -> no caustic warning there.
        assert not any('fold caustic' in str(w.message) for w in rec)
    assert np.all(np.isfinite(E_exit))
    E_fold = la.angular_spectrum_propagate(E_exit, opd, _WL, dxc)
    assert np.all(np.isfinite(E_fold))
    # axis-centred windowed metrics (rotational caustic).
    I = np.abs(E_fold) ** 2
    xx = (np.arange(Nc) - Nc / 2) * dxc
    Xg, Yg = np.meshgrid(xx, xx)
    r = np.sqrt(Xg ** 2 + Yg ** 2)
    m = r <= 60e-6
    Iw, rw, Pw = I[m], r[m], float(I[m].sum())
    r2m = np.sqrt((Iw * rw ** 2).sum() / Pw) * 1e6
    rb = np.linspace(0, 60e-6, 601)
    cum = np.array([Iw[rw <= rr].sum() for rr in rb]) / Pw
    e50 = float(np.interp(0.5, cum, rb)) * 1e6
    e80 = float(np.interp(0.8, cum, rb)) * 1e6
    assert abs(r2m / meta['grid_r2m_um'] - 1.0) < 0.08, (r2m, meta['grid_r2m_um'])
    assert abs(e50 / meta['grid_EE50_um'] - 1.0) < 0.08, (e50, meta['grid_EE50_um'])
    assert abs(e80 / meta['grid_EE80_um'] - 1.0) < 0.10, (e80, meta['grid_EE80_um'])


def test_caustic_at_output_plane_detected_and_finite():
    """When the traced OUTPUT plane is placed AT a focus (a dummy surface at the
    paraxial image), the exit ray map itself is a near-caustic (det J -> 0):
    single-branch ray density under-resolves the singular spot.  The mode must
    DETECT it (one-time RuntimeWarning steering to GBD/FGA) and stay FINITE --
    never inf/nan."""
    N, dx = 768, 6e-6
    E0 = _gauss(N, dx, 4e-3)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = np.asarray(la.apply_real_lens_traced(
            E0, prescription=_singlet_to_focus((1e-3, 0.0)), wavelength=_WL,
            dx=dx, on_undersample='silent', amplitude_model='ray_density'))
    assert np.all(np.isfinite(out))
    assert not np.any(np.isinf(out))
    assert any('fold caustic' in str(w.message) for w in rec), \
        'caustic-at-output must warn'


# ===========================================================================
# DECENTER -- ray_density BROADENS (kills any shrink) grid-robustly, but at the
# exit vertex the Jacobian is ~const so it TRACKS the screen and UNDER-represents
# the geometric coma: the honest N12 limit (GBD remains the reference).
# ===========================================================================
@pytest.mark.slow
def test_decenter_broadens_grid_robust_but_tracks_screen():
    """The ray-density decentered spot RMS BROADENS (> 1, monotonic in decenter,
    grid-robust), so it never SHRINKS.  But because the traced output is the exit
    VERTEX (det J ~ const there), it TRACKS the screen to within ~2% -- the
    exit-vertex amplitude cannot carry the downstream coma.  Pinned as the honest
    envelope (the accurate decentered-coma model is GBD, N10b)."""
    ratios1, ratios2 = [], []
    for N, dx in ((1024, 6e-6), (1536, 4e-6)):
        E0 = _gauss(N, dx, 4e-3)
        on = _img(_rlt(E0, _singlet((1e-9, 0.0)), dx, 'ray_density'), dx)['rms']
        r1 = _img(_rlt(E0, _singlet((1e-3, 0.0)), dx, 'ray_density'), dx)['rms']
        r2 = _img(_rlt(E0, _singlet((2e-3, 0.0)), dx, 'ray_density'), dx)['rms']
        assert r1 > on and r2 > on, (N, on, r1, r2)          # BROADENS (no shrink)
        assert r2 / on > r1 / on, (N, r1, r2)                # monotonic
        ratios1.append(r1 / on)
        ratios2.append(r2 / on)
        # THE honest limit pin: the ray-density decentered PSF TRACKS the screen
        # PSF (RMS broadening ratio agrees within 3%) -- the exit-vertex Jacobian
        # is near-constant (~0.3% amplitude modulation), so the exit-plane
        # amplitude cannot carry the downstream coma.  ``apply_real_lens_gbd``
        # (N10b), whose beamlets carry the image-plane ray density, is the
        # decentered-coma reference (P9); ``ray_density`` does not replace it.
        s_on = _img(_rlt(E0, _singlet((1e-9, 0.0)), dx, 'screen'), dx)['rms']
        s2 = _img(_rlt(E0, _singlet((2e-3, 0.0)), dx, 'screen'), dx)['rms']
        assert abs((r2 / on) - (s2 / s_on)) < 0.03, (N, r2 / on, s2 / s_on)
    # grid-robust broadening ratios (RMS is a continuous integral -- P9 lesson).
    assert abs(ratios1[0] - ratios1[1]) < 0.01, ratios1
    assert ratios1[-1] > 1.01 and ratios2[-1] > 1.03, (ratios1, ratios2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
