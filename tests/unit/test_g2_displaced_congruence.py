"""G2 Task 1 (2026-07-19): congruence-aware ``surface_model='displaced'``.

The pre-G2 displaced screen launched its per-surface obliquity fan COLLIMATED,
exact only for a collimated input.  G2 launches the fan along the INPUT
CONGRUENCE selected by ``conjugate=`` (None collimated / scalar R_in / 'auto' /
wavefront ndarray), so the second-and-later surfaces of a lens see the true
converging/diverging incidence.

Independent oracle (inline, lumenairy-free): an EXACT meridional geometric ray
trace launched along the same congruence, whose transverse ray aberration at the
image plane is the ground-truth aberrated spot (r2m >> diffraction here).  This
is the robust oracle for an aberrated finite-conjugate spot -- the far-field
ring-Huygens sum mis-weights the exit-pupil measure for a non-collimated
congruence (session finding; see docs/audit_real_lens_displaced_2026_07_19.md
G2 section).

Headline fail-before/pass-after (test_congruence_beats_collimated_on_doublet):
on the cemented DOUBLET with a diverging input the COLLIMATED fan grossly
UNDER-represents the aberration (it applies the wrong obliquity, which spuriously
"corrects" the spot); the CONGRUENCE fan tracks the geometric caustic.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy.glass import GLASS_REGISTRY

_WL = 1.31e-6
K = 2 * np.pi / _WL
# Model glass for THIS module only: registered and removed by
# tests/conftest.py::_module_glass_registry_guard.
MODULE_GLASSES = {'_G2_A': lambda wl: 1.5168,
                  '_G2_B': lambda wl: 1.6200}


# ---------------------------------------------------------------------------
# Prescriptions (G1 matrix: M6 f/5 singlet, M1 cemented doublet)
# ---------------------------------------------------------------------------
def _singlet():
    return {'wavelength': _WL, 'aperture_diameter': 24e-3,
            'surfaces': [
                {'radius': 51.68e-3, 'thickness': 5e-3, 'glass_before': 'air',
                 'glass_after': '_G2_A', 'semi_diameter': 12e-3},
                {'radius': -51.68e-3, 'thickness': 0.0, 'glass_before': '_G2_A',
                 'glass_after': 'air', 'semi_diameter': 12e-3}],
            'thicknesses': [5e-3], 'stop_index': 0}


def _doublet():
    return {'wavelength': _WL, 'aperture_diameter': 40e-3,
            'surfaces': [
                {'radius': 61.5e-3, 'thickness': 6e-3, 'glass_before': 'air',
                 'glass_after': '_G2_A', 'semi_diameter': 20e-3},
                {'radius': -44.5e-3, 'thickness': 2.5e-3, 'glass_before': '_G2_A',
                 'glass_after': '_G2_B', 'semi_diameter': 20e-3},
                {'radius': -129e-3, 'thickness': 0.0, 'glass_before': '_G2_B',
                 'glass_after': 'air', 'semi_diameter': 20e-3}],
            'thicknesses': [6e-3, 2.5e-3], 'stop_index': 0}


def _n(glass):
    return 1.0 if glass == 'air' else float(GLASS_REGISTRY[glass](_WL))


# ---------------------------------------------------------------------------
# Inline independent oracle: exact meridional geometric ray trace + paraxial
# image + transverse-aberration spot (Gaussian-apodized).  Spherical surfaces.
# ---------------------------------------------------------------------------
def _paraxial_image(presc, R_in):
    surfs, th = presc['surfaces'], presc['thicknesses']
    h = 1.0
    u = (1.0 / R_in if R_in is not None else 0.0)
    for i, s in enumerate(surfs):
        R = s['radius']
        n1, n2 = _n(s['glass_before']), _n(s['glass_after'])
        phi = 0.0 if (R == 0 or not np.isfinite(R)) else (n2 - n1) / R
        u = (n1 * u - h * phi) / n2
        if i < len(surfs) - 1:
            h = h + u * th[i]
    return -h / u          # BFD from last vertex


def _trace(h, surfs, thicknesses, R_in):
    p = np.array([0.0, h])
    if R_in is None:
        d = np.array([1.0, 0.0])
    else:
        d = np.array([1.0, h / R_in])
        d = d / np.linalg.norm(d)
    n_in, z_v = 1.0, 0.0
    for i, s in enumerate(surfs):
        R = s['radius']
        n_out = _n(s['glass_after'])
        if R == 0 or not np.isfinite(R):
            t = (z_v - p[0]) / d[0]
            p = p + t * d
            nrm = np.array([1.0, 0.0])
        else:
            c = np.array([z_v + R, 0.0])
            oc = p - c
            b = np.dot(oc, d)
            disc = b * b - (np.dot(oc, oc) - R * R)
            if disc < 0:
                return None
            sq = np.sqrt(disc)
            t = (-b - sq) if R > 0 else (-b + sq)
            p = p + t * d
            nrm = (p - c) / R
            if nrm[0] < 0:
                nrm = -nrm
        cos_i = np.dot(d, nrm)
        eta = n_in / n_out
        s2 = eta * eta * (1 - cos_i * cos_i)
        if s2 > 1:
            return None
        cost = np.sqrt(1 - s2)
        d = eta * d + (cost - eta * cos_i) * nrm
        d = d / np.linalg.norm(d)
        n_in = n_out
        if i < len(thicknesses):
            z_v += thicknesses[i]
    return p, d


def _geo_spot(presc, R_in, w0, z_img):
    """Energy-weighted geometric r2m + EE80 [m] of the transverse ray
    aberration at ``z_img`` (paraxial image), Gaussian apodization w0."""
    surfs, thicknesses = presc['surfaces'], presc['thicknesses']
    z_exit = sum(thicknesses)
    hs = np.linspace(1e-7, 1.9 * w0, 3000)
    yi, wi = [], []
    for h in hs:
        r = _trace(h, surfs, thicknesses, R_in)
        if r is None:
            continue
        p, d = r
        t = (z_exit + z_img - p[0]) / d[0]
        y = p[1] + t * d[1]
        yi.append(abs(y))
        wi.append(np.exp(-2 * h ** 2 / w0 ** 2) * h)
    yi = np.array(yi)
    wi = np.array(wi)
    r2m = float(np.sqrt(np.sum(wi * yi ** 2) / np.sum(wi)))
    order = np.argsort(yi)
    cum = np.cumsum(wi[order]) / np.sum(wi)
    ee80 = float(np.interp(0.8, cum, yi[order]))
    return r2m, ee80


# ---------------------------------------------------------------------------
# Field / metric helpers
# ---------------------------------------------------------------------------
def _gc(N, dx, w0, R_in):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r2 = X ** 2 + Y ** 2
    amp = np.exp(-r2 / w0 ** 2)
    if R_in is None:
        return amp.astype(np.complex128)
    return (amp * np.exp(1j * K * r2 / (2 * R_in))).astype(np.complex128)


def _wave_ee80(E_exit, dx, z_img, win):
    E = la.angular_spectrum_propagate(E_exit, z_img, _WL, dx)
    I = np.abs(E) ** 2
    N = I.shape[0]
    x = (np.arange(N) - N / 2) * dx
    j, i = np.unravel_index(np.argmax(I), I.shape)
    X, Y = np.meshgrid(x - x[i], x - x[j])
    r = np.sqrt(X ** 2 + Y ** 2)
    rb = np.linspace(0, win, 401)
    cum = np.array([I[r <= rr].sum() for rr in rb]) / I.sum()
    return float(np.interp(0.8, cum, rb))


# ===========================================================================
# FAST tests
# ===========================================================================
def test_conjugate_none_is_byte_identical_to_default_displaced():
    """conjugate=None (collimated congruence) must equal the pre-G2 displaced
    fan byte-for-byte -- the collimated path is unchanged (regression pin)."""
    N, dx = 1024, 8e-6
    E0 = _gc(N, dx, 0.5e-3, None)
    p = _singlet()
    a = la.apply_real_lens(E0, prescription=p, wavelength=_WL, dx=dx,
                           surface_model='displaced')
    b = la.apply_real_lens(E0, prescription=p, wavelength=_WL, dx=dx,
                           surface_model='displaced', conjugate=None)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_thin_plus_conjugate_raises():
    """conjugate= is meaningless for the default 'thin' screen -> raise."""
    N, dx = 256, 8e-6
    E0 = _gc(N, dx, 0.5e-3, None)
    with pytest.raises(ValueError, match='conjugate'):
        la.apply_real_lens(E0, prescription=_singlet(), wavelength=_WL, dx=dx,
                           conjugate=150e-3)
    with pytest.raises(ValueError):
        la.apply_real_lens(E0, prescription=_singlet(), wavelength=_WL, dx=dx,
                           surface_model='displaced', conjugate=0.0)


def test_scalar_conjugate_changes_the_screen():
    """A finite conjugate must change the obliquity fan (hence the field) vs the
    collimated fan on a beam that actually samples the obliquity."""
    N, dx = 1024, 8e-6
    E0 = _gc(N, dx, 3e-3, 150e-3)
    p = _singlet()
    a = la.apply_real_lens(E0, prescription=p, wavelength=_WL, dx=dx,
                           surface_model='displaced', conjugate=None)
    b = la.apply_real_lens(E0, prescription=p, wavelength=_WL, dx=dx,
                           surface_model='displaced', conjugate=150e-3)
    assert not np.array_equal(np.asarray(a), np.asarray(b))


def test_auto_matches_scalar_for_matched_divergent_input():
    """conjugate='auto' fits the carrier from E_in; for a clean diverging
    Gaussian of known R_in it must reproduce the scalar-conjugate screen."""
    N, dx = 1024, 6e-6
    R_in = 150e-3
    E0 = _gc(N, dx, 3e-3, R_in)
    p = _singlet()
    a = la.apply_real_lens(E0, prescription=p, wavelength=_WL, dx=dx,
                           surface_model='displaced', conjugate=R_in)
    b = la.apply_real_lens(E0, prescription=p, wavelength=_WL, dx=dx,
                           surface_model='displaced', conjugate='auto')
    rel = np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(a).max()
    assert rel < 5e-3, f"auto vs scalar rel {rel:.2e}"


def test_lut_cache_bounded_and_registered():
    """The displaced cosine-LUT cache is registered with the central registry
    and bounded/drainable (G1 cache conventions)."""
    from lumenairy._cache_registry import list_registered_cache_clearers
    from lumenairy.elements._lens_real import (
        _DISPLACED_LUT_CACHE,
        _DISPLACED_LUT_CACHE_MAX,
        clear_displaced_lut_cache,
    )
    assert 'displaced_cos_luts' in list_registered_cache_clearers()
    N, dx = 256, 8e-6
    E0 = _gc(N, dx, 2e-3, None)
    p = _singlet()
    clear_displaced_lut_cache()
    for R in (100e-3, 120e-3, 140e-3, 160e-3, 180e-3, 200e-3,
              220e-3, 240e-3, 260e-3, 280e-3):
        la.apply_real_lens(E0, prescription=p, wavelength=_WL, dx=dx,
                           surface_model='displaced', conjugate=R)
    assert len(_DISPLACED_LUT_CACHE) <= _DISPLACED_LUT_CACHE_MAX
    clear_displaced_lut_cache()
    assert len(_DISPLACED_LUT_CACHE) == 0


def test_congruence_beats_collimated_on_doublet():
    """HEADLINE fail-before/pass-after + independent oracle.  Cemented DOUBLET,
    diverging input R_in=+150 mm: the COLLIMATED fan grossly under-represents
    the geometric aberration (wrong obliquity compounds across the 3 surfaces);
    the CONGRUENCE fan tracks it.  Oracle: inline geometric ray-trace spot."""
    N, dx = 4096, 5e-6
    R_in = 150e-3
    w0 = 8e-3
    p = _doublet()
    z_img = _paraxial_image(p, R_in)
    E0 = _gc(N, dx, w0, R_in)
    win = 400e-6
    ee_collim = _wave_ee80(la.apply_real_lens(
        E0, prescription=p, wavelength=_WL, dx=dx,
        surface_model='displaced', conjugate=None), dx, z_img, win)
    ee_congr = _wave_ee80(la.apply_real_lens(
        E0, prescription=p, wavelength=_WL, dx=dx,
        surface_model='displaced', conjugate=R_in), dx, z_img, win)
    geo_r2m, geo_ee80 = _geo_spot(p, R_in, w0, z_img)
    # collimated fan under-represents the true aberration by >3x
    assert ee_collim < 0.3 * geo_ee80, (
        f"collim EE80 {ee_collim*1e6:.1f} vs geo {geo_ee80*1e6:.1f}")
    # congruence fan tracks the geometric caustic (materially closer)
    assert ee_congr > 0.6 * geo_ee80, (
        f"congr EE80 {ee_congr*1e6:.1f} vs geo {geo_ee80*1e6:.1f}")
    # and is a large, unambiguous improvement over the collimated fan
    assert ee_congr > 3.0 * ee_collim


@pytest.mark.slow
def test_congruence_within_envelope_on_singlet():
    """Singlet f/5, diverging R_in=+150 mm, w0=5 mm: the congruence fan lands
    within ~15% of the geometric-caustic oracle on EE80 (moderate finite-
    conjugate aberration -- the regime where the phase-screen family reaches
    the oracle; extreme aberration retains the H2 walk-off floor)."""
    N, dx = 4096, 3e-6
    R_in = 150e-3
    w0 = 5e-3
    p = _singlet()
    z_img = _paraxial_image(p, R_in)
    E0 = _gc(N, dx, w0, R_in)
    ee_congr = _wave_ee80(la.apply_real_lens(
        E0, prescription=p, wavelength=_WL, dx=dx,
        surface_model='displaced', conjugate=R_in), dx, z_img, 400e-6)
    _, geo_ee80 = _geo_spot(p, R_in, w0, z_img)
    assert abs(ee_congr - geo_ee80) / geo_ee80 < 0.15, (
        f"congr EE80 {ee_congr*1e6:.1f} vs geo {geo_ee80*1e6:.1f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
