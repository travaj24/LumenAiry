"""P2 (niche N1): extreme finite-conjugate ``surface_model='displaced'``.

The pre-P2 displaced screen was reported at "~0.50x oracle r2m" on a negative-
lens real focus and "~0.58x" on virtual-image back-propagation, attributed to a
transverse-walk model floor.  P2 REFUTES that attribution with the P0-fixed
congruence DIFFRACTION oracle (``validation/oracles/debye_oracle_v3.py``): the
prior floor was measured against the GEOMETRIC ray-density spot, which
over-estimates the true wave spot by ~2x near the reconvergence caustic (and was
compounded by grid truncation).  Against the diffraction-faithful oracle the
SHIPPED screen is already within ~4-8% on every extreme case.

This module pins:
  * defaults byte-identical (displaced == displaced_mode='screen');
  * displaced_mode validation;
  * the two experimental candidates run + conserve energy;
  * the diffraction oracle is trustworthy (reproduces the ABCD Gaussian limit);
  * FAIL-BEFORE / PASS-AFTER: vs the GEOMETRIC spot the models read ~0.5x (the
    "floor"); vs the DIFFRACTION oracle screen + remap + split are within 15%.

Oracle is imported directly from ``validation/oracles/`` -- lumenairy-free.
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
# Model glass for THIS module only: registered and removed by
# tests/conftest.py::_module_glass_registry_guard.
MODULE_GLASSES = {'_P2_A': lambda wl: 1.5168,
                  '_P2_B': lambda wl: 1.6200}

# ---- import the lumenairy-free diffraction oracle -------------------------
_ORACLE_PATH = (pathlib.Path(__file__).resolve().parents[2]
                / 'validation' / 'oracles' / 'debye_oracle_v3.py')
_spec = importlib.util.spec_from_file_location('debye_oracle_v3', _ORACLE_PATH)
_oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oracle)


# ---------------------------------------------------------------------------
# Prescriptions (mirror the G1/G2 matrix + the P2 negative-lens cases)
# ---------------------------------------------------------------------------
def _singlet():
    return {'wavelength': _WL, 'aperture_diameter': 24e-3, 'surfaces': [
        {'radius': 51.68e-3, 'thickness': 5e-3, 'glass_before': 'air',
         'glass_after': '_P2_A', 'semi_diameter': 12e-3},
        {'radius': -51.68e-3, 'thickness': 0.0, 'glass_before': '_P2_A',
         'glass_after': 'air', 'semi_diameter': 12e-3}],
        'thicknesses': [5e-3], 'stop_index': 0}


def _doublet():
    return {'wavelength': _WL, 'aperture_diameter': 40e-3, 'surfaces': [
        {'radius': 61.5e-3, 'thickness': 6e-3, 'glass_before': 'air',
         'glass_after': '_P2_A', 'semi_diameter': 20e-3},
        {'radius': -44.5e-3, 'thickness': 2.5e-3, 'glass_before': '_P2_A',
         'glass_after': '_P2_B', 'semi_diameter': 20e-3},
        {'radius': -129e-3, 'thickness': 0.0, 'glass_before': '_P2_B',
         'glass_after': 'air', 'semi_diameter': 20e-3}],
        'thicknesses': [6e-3, 2.5e-3], 'stop_index': 0}


def _biconcave(R1=-50e-3, R2=50e-3, tc=3e-3):
    return {'wavelength': _WL, 'aperture_diameter': 20e-3, 'surfaces': [
        {'radius': R1, 'thickness': tc, 'glass_before': 'air',
         'glass_after': '_P2_A', 'semi_diameter': 10e-3},
        {'radius': R2, 'thickness': 0.0, 'glass_before': '_P2_A',
         'glass_after': 'air', 'semi_diameter': 10e-3}],
        'thicknesses': [tc], 'stop_index': 0}


def _n(glass):
    return 1.0 if glass == 'air' else float(GLASS_REGISTRY[glass](_WL))


def _paraxial_image(presc, R_in):
    surfs, th = presc['surfaces'], presc['thicknesses']
    h, u = 1.0, (1.0 / R_in if R_in is not None else 0.0)
    for i, s in enumerate(surfs):
        R = s['radius']
        n1, n2 = _n(s['glass_before']), _n(s['glass_after'])
        phi = 0.0 if (R == 0 or not np.isfinite(R)) else (n2 - n1) / R
        u = (n1 * u - h * phi) / n2
        if i < len(surfs) - 1:
            h = h + u * th[i]
    return -h / u


def _gaussian(N, dx, w0, R_in):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r2 = X ** 2 + Y ** 2
    amp = np.exp(-r2 / w0 ** 2)
    if R_in is None:
        return amp.astype(np.complex128)
    return (amp * np.exp(1j * K * r2 / (2.0 * R_in))).astype(np.complex128)


def _wave_metrics(E_exit, dx, z_img, win=400e-6):
    E = la.angular_spectrum_propagate(E_exit, z_img, _WL, dx)
    I = np.abs(E) ** 2
    N = I.shape[0]
    x = (np.arange(N) - N / 2) * dx
    j, i = np.unravel_index(np.argmax(I), I.shape)
    X, Y = np.meshgrid(x - x[i], x - x[j])
    r = np.sqrt(X ** 2 + Y ** 2)
    m = r <= win
    Iw, rw, Pw = I[m], r[m], I[m].sum()
    r2m = float(np.sqrt(np.sum(Iw * rw ** 2) / Pw))
    rb = np.linspace(0, win, 601)
    cum = np.array([Iw[rw <= rr].sum() for rr in rb]) / Pw
    return r2m, float(np.interp(0.8, cum, rb))


def _oracle_job(presc, R_in, w0, z_img):
    """Build the debye_oracle_v3 job (mm units) from a lumenairy prescription;
    the last surface's thickness carries the image distance."""
    surfs = []
    for i, s in enumerate(presc['surfaces']):
        t = (presc['thicknesses'][i] if i < len(presc['thicknesses'])
             else z_img)
        idx = _n(s['glass_after'])
        surfs.append({'radius_mm': (s['radius'] * 1e3 if np.isfinite(s['radius'])
                                    and s['radius'] != 0 else 0),
                      'thickness_mm': t * 1e3,
                      'index': 'air' if idx == 1.0 else idx})
    return {'wavelength_um': _WL * 1e6,
            'R_in_mm': None if R_in is None else R_in * 1e3,
            'aperture_mm': presc['aperture_diameter'] * 1e3,
            'pop': {'w0_mm': w0 * 1e3}, 'surfaces': surfs}


def _disp(E0, presc, R_in, dx, mode):
    return la.apply_real_lens(E0, prescription=presc, wavelength=_WL, dx=dx,
                              surface_model='displaced', conjugate=R_in,
                              displaced_mode=mode)


# ===========================================================================
# FAST -- defaults, validation, energy, oracle self-check
# ===========================================================================
def test_default_displaced_is_screen_byte_identical():
    """surface_model='displaced' with no displaced_mode == displaced_mode=
    'screen' byte-for-byte -- both collimated and scalar-conjugate (the G1/G2
    contract, extended to the new kwarg default)."""
    N, dx = 512, 8e-6
    p = _singlet()
    for R_in in (None, 150e-3):
        E0 = _gaussian(N, dx, 2e-3, R_in)
        a = la.apply_real_lens(E0, prescription=p, wavelength=_WL, dx=dx,
                               surface_model='displaced', conjugate=R_in)
        b = _disp(E0, p, R_in, dx, 'screen')
        # Same numerical model (default == 'screen'); the first (cache-cold)
        # displaced call builds the cosine LUT via a reduction whose order
        # differs ~1 ULP from the cached-retrieve path, so bit-identity is not
        # robust to cross-test cache warmth -- assert numerical identity.
        assert np.max(np.abs(np.asarray(a) - np.asarray(b))) <= \
            1e-10 * float(np.max(np.abs(np.asarray(b)))), R_in


def test_thin_default_unchanged_by_displaced_mode_default():
    """The default 'thin' path is byte-identical whether displaced_mode is
    omitted or passed as its 'screen' default."""
    N, dx = 512, 8e-6
    E0 = _gaussian(N, dx, 0.5e-3, None)
    p = _singlet()
    a = la.apply_real_lens(E0, prescription=p, wavelength=_WL, dx=dx)
    b = la.apply_real_lens(E0, prescription=p, wavelength=_WL, dx=dx,
                           displaced_mode='screen')
    # numerical (not bit) identity -- cache-warmth ~1 ULP, see the sibling pin
    assert np.max(np.abs(np.asarray(a) - np.asarray(b))) <= \
        1e-10 * float(np.max(np.abs(np.asarray(b))))


def test_displaced_mode_validation():
    N, dx = 256, 8e-6
    E0 = _gaussian(N, dx, 0.5e-3, None)
    p = _singlet()
    with pytest.raises(ValueError, match='unknown displaced_mode'):
        la.apply_real_lens(E0, prescription=p, wavelength=_WL, dx=dx,
                           surface_model='displaced', displaced_mode='bogus')
    with pytest.raises(ValueError, match='displaced_mode='):
        la.apply_real_lens(E0, prescription=p, wavelength=_WL, dx=dx,
                           displaced_mode='remap')     # thin + non-screen


def test_remap_split_conserve_energy():
    """The remap's energy Jacobian and the split's reduced-distance propagation
    conserve throughput -- total exit power matches the screen path (all three
    are lossless within the traced pupil) to <2%."""
    N, dx = 1024, 6e-6
    R_in, w0 = 150e-3, 3e-3
    p = _singlet()
    E0 = _gaussian(N, dx, w0, R_in)
    p_screen = float(np.sum(np.abs(_disp(E0, p, R_in, dx, 'screen')) ** 2))
    for mode in ('remap', 'split'):
        pm = float(np.sum(np.abs(_disp(E0, p, R_in, dx, mode)) ** 2))
        assert abs(pm - p_screen) / p_screen < 0.02, (mode, pm, p_screen)


def test_remap_phase_continuity_and_coherent_focus():
    """Phase-continuity probe on the remap (plan N1: 'coherent fringe test').

    (1) The remap exit field is PHASE-CONTINUOUS: across the illuminated pupil
        the adjacent-pixel phase step stays below Nyquist (< pi), so the
        coordinate warp + energy Jacobian + eikonal did not scramble the phase.
    (2) COHERENT FOCUS consistency vs an independent coherent propagator (the
        validated 'screen' path -- `traced` over-blurs this diverging large beam,
        so it is not the fringe reference here): both focus to the SAME spot
        (EE80 within 15%) with a comparable peak.  A phase-discontinuous remap
        could not focus coherently."""
    N, dx = 2048, 6e-6
    R_in, w0 = 150e-3, 2e-3          # hw = 6.14mm = 3.1 w0
    p = _singlet()
    z_img = _paraxial_image(p, R_in)
    E0 = _gaussian(N, dx, w0, R_in)
    E_remap = np.asarray(_disp(E0, p, R_in, dx, 'remap'))
    # (1) phase continuity across the illuminated pupil (central row)
    row = E_remap[N // 2]
    amp = np.abs(row)
    bright = amp > 0.02 * amp.max()
    ph = np.angle(row[bright])
    step = np.abs(np.diff(np.unwrap(ph)))
    assert step.max() < np.pi, f"phase step {step.max():.3f} exceeds Nyquist"
    # (2) coherent-focus consistency with the screen propagator
    r_s, ee_s = _wave_metrics(_disp(E0, p, R_in, dx, 'screen'), dx, z_img)
    r_r, ee_r = _wave_metrics(E_remap, dx, z_img)
    assert abs(ee_r - ee_s) / ee_s < 0.15, (ee_r, ee_s)
    pk_s = np.abs(la.angular_spectrum_propagate(
        _disp(E0, p, R_in, dx, 'screen'), z_img, _WL, dx)).max()
    pk_r = np.abs(la.angular_spectrum_propagate(E_remap, z_img, _WL, dx)).max()
    assert 0.5 < pk_r / pk_s < 2.0, (pk_r, pk_s)   # coherent focusing preserved


def test_diffraction_oracle_reproduces_abcd_limit():
    """INDEPENDENT trust check on the P0 oracle: for a SMALL (diffraction-
    limited) beam its EE80 must reproduce the ABCD Gaussian image size
    (EE80 = 0.897 * w_img) to <8%.  This is what makes the oracle's aberrated
    verdicts credible."""
    for presc, R_in in ((_singlet(), 150e-3), (_biconcave(), -35e-3)):
        z_img = _paraxial_image(presc, R_in)
        for w0 in (0.4e-3, 0.8e-3):
            job = _oracle_job(presc, R_in, w0, z_img)
            job['window_um'] = 200.0
            job['rho_max_um'] = 300.0
            huy_ee = _oracle.evaluate(job)['huy_EE80_um'] * 1e-6
            w_img = _abcd_image_waist(presc, R_in, w0)
            ee_abcd = 0.897 * w_img
            assert abs(huy_ee - ee_abcd) / ee_abcd < 0.08, (
                presc['surfaces'][0]['radius'], w0, huy_ee, ee_abcd)


def _abcd_image_waist(presc, R_in, w0):
    surfs, th = presc['surfaces'], presc['thicknesses']
    M = np.eye(2)
    for i, s in enumerate(surfs):
        n1, n2 = _n(s['glass_before']), _n(s['glass_after'])
        R = s['radius'] if s['radius'] != 0 else np.inf
        P = -((n2 - n1) / R / n2) if np.isfinite(R) else 0.0
        M = np.array([[1, 0], [P, n1 / n2]]) @ M
        if i < len(th):
            M = np.array([[1, th[i] / n2], [0, 1]]) @ M
    q_inv = (1.0 / R_in if R_in is not None else 0.0) - 1j * _WL / (np.pi * w0 ** 2)
    q = 1.0 / q_inv
    A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    q_ex = (A * q + B) / (C * q + D)
    q_w = q_ex - np.real(q_ex)
    return float(np.sqrt(-_WL / (np.pi * np.imag(1.0 / q_w))))


# ===========================================================================
# SLOW -- oracle-backed extreme-conjugate accuracy (the P2 headline)
# ===========================================================================
@pytest.mark.slow
@pytest.mark.parametrize('name,presc_fn,R_in,w0,N,dx', [
    ('M5real', _biconcave, -35e-3, 4e-3, 4096, 5e-6),   # negative-lens real focus
    ('M1', _doublet, 150e-3, 8e-3, 8192, 5.2e-6),        # cemented doublet, diverging
])
def test_extreme_conjugate_within_15pct_of_diffraction_oracle(
        name, presc_fn, R_in, w0, N, dx):
    """HEADLINE fail-before / pass-after.

    FAIL-BEFORE: vs the GEOMETRIC ray-density spot the screen reads ~0.5-0.8x
    (the documented "floor"), and the diffraction oracle itself is far below the
    geometric spot (the geometric metric over-estimates the wave spot).

    PASS-AFTER: vs the diffraction-faithful oracle the shipped screen AND both
    experimental candidates (remap, split) are within 15% -- the "floor" was an
    oracle artefact, not a model limit."""
    # This validation builds several N^2 complex128 fields (input + three
    # displaced modes each ASM-propagated to the image + the diffraction oracle).
    # At N=8192 the peak (~10-13 GB) exceeds a standard 16 GB CI runner -- observed
    # as a runner OOM/shutdown -- so skip on memory-constrained hosts.  It is a
    # memory-bound accuracy check, not a regression guard, and runs on >=24 GB
    # machines (where the campaign originally validated it).
    from lumenairy.memory import available_memory_bytes
    est_peak_gb = (N * N * 16 * 12) / 1e9
    avail_gb = available_memory_bytes() / 1e9
    if avail_gb < est_peak_gb + 4.0:
        pytest.skip(
            f"{name}: needs ~{est_peak_gb:.0f} GB peak (N={N}); only "
            f"{avail_gb:.0f} GB available -- memory-bound accuracy check")
    presc = presc_fn()
    z_img = _paraxial_image(presc, R_in)
    huy = _oracle.evaluate(_oracle_job(presc, R_in, w0, z_img))
    huy_ee = huy['huy_EE80_um'] * 1e-6
    geo_ee = huy['geo_EE80_um'] * 1e-6
    # reframing: the geometric spot over-estimates the true wave spot
    assert huy_ee / geo_ee < 0.95, (name, huy_ee, geo_ee)
    E0 = _gaussian(N, dx, w0, R_in)
    ee = {m: _wave_metrics(_disp(E0, presc, R_in, dx, m), dx, z_img)[1]
          for m in ('screen', 'remap', 'split')}
    # fail-before anchor: vs the geometric spot the screen looks far short
    assert ee['screen'] / geo_ee < 0.85, (name, ee['screen'], geo_ee)
    # pass-after: within 15% of the diffraction-faithful oracle, all modes
    for m in ('screen', 'remap', 'split'):
        assert abs(ee[m] - huy_ee) / huy_ee < 0.15, (name, m, ee[m], huy_ee)


@pytest.mark.slow
def test_M5_virtual_image_backprop_within_15pct():
    """Negative lens over-diverging a converging input to a VIRTUAL (upstream)
    image (R_in=-60mm): the exit field is ASM back-propagated to the virtual
    plane.

    ORACLE (independent of lumenairy): the forward ring-Huygens (+R0) kernel
    CANNOT reach an upstream image -- run naively it blows up to ~715 um with
    only ~25% of energy in-window.  So the P0 oracle (``debye_oracle_v3``)
    routes the virtual case (``zrel < 0``) to a diffraction BACK-propagation
    (``huy_method == 'asm_backprop'``): it builds the EXACT ray-traced exit
    field (energy-conserving amplitude + eikonal phase) and angular-spectrum
    back-propagates it.  That exit-field construction is entirely separate from
    lumenairy's phase-screen family, so it is a genuine independent oracle.

    FAIL-BEFORE: vs the GEOMETRIC ray-density spot the screen reads ~0.6x (the
    documented "0.58x floor").  PASS-AFTER: vs the diffraction back-prop oracle
    the shipped ``screen`` (and both experimental candidates) are within 15% --
    the floor was a geometric-spot artefact, not a model limit.
    """
    presc, R_in, w0, N, dx = _biconcave(-50e-3, 50e-3), -60e-3, 4e-3, 4096, 5e-6
    z_img = _paraxial_image(presc, R_in)          # negative -> virtual, upstream
    assert z_img < 0, z_img
    # independent diffraction oracle at the virtual plane: exact ray-traced exit
    # field, ASM back-propagated on the SAME grid the model uses (so the only
    # difference is the exit-field construction, not the propagator/grid).
    job = _oracle_job(presc, R_in, w0, z_img)
    job['asm_N'], job['asm_dx_um'] = N, dx * 1e6
    huy = _oracle.evaluate(job)
    assert huy['huy_method'] == 'asm_backprop', huy['huy_method']
    huy_r2m = huy['huy_r2m_um'] * 1e-6
    geo_r2m = huy['geo_r2m_um'] * 1e-6            # over-estimating geometric spot
    E0 = _gaussian(N, dx, w0, R_in)
    r2m = {m: _wave_metrics(_disp(E0, presc, R_in, dx, m), dx, z_img)[0]
           for m in ('screen', 'remap', 'split')}
    # fail-before anchor: vs the geometric spot the screen reads well under 1x
    assert r2m['screen'] / geo_r2m < 0.75, (r2m['screen'], geo_r2m)
    # pass-after: every mode within 15% of the INDEPENDENT diffraction back-prop
    for m in ('screen', 'remap', 'split'):
        assert abs(r2m[m] - huy_r2m) / huy_r2m < 0.15, (m, r2m[m], huy_r2m)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
