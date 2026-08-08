"""P8 (niche N9): capstone -- composed end-to-end + full-aperture fast case.

Every piece of the accuracy-niches campaign is oracle-validated in isolation
(P1-P7); this module pins the COMPOSITION that nobody had demonstrated:

  * STEP A (N0.2) -- the ZOS Huygens-PSF oracle mode.  The Zemax comparison is
    recorded in ``validation/oracles/README.md`` + the plan RESULTS section (it
    needs a licence); the UNIT test here pins the lumenairy-free half -- the
    analytic Airy relation and the ``debye_oracle_v3`` uniform-pupil reference
    that the Huygens PSF matched to <1% (matched metric window).
  * STEP B (N9) -- a generic production-class chain (fast doublet + relay)
    propagated END-TO-END through the NEW stack: per-group traced (H6/N5) for the
    doublet, a carrier-referenced gap leg (N7), the universal-gated fast path for
    the relay (routes to the displaced phase-screen), and a carrier-referenced
    FINAL-FOCUS leg -- vs ``debye_oracle_v3``.
  * STEP C (N9) -- the full-aperture f/2 biconvex (M4) at w0=9 mm reconstructed
    AT FOCUS via the pilot-beam (GBD) machinery on a MODEST grid, where a fixed
    grid at the same N cannot resolve the ~0.64-NA exit and the exact route needs
    N ~ 21000 (budget-infeasible).
  * STEP D -- energy closure + grid-halving audit on the composed primitives.

The oracle is imported directly from ``validation/oracles/`` -- lumenairy-free.
Unit tests run WITHOUT Zemax.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

import lumenairy as la

_WL = 1.31e-6
_WL_VIS = 0.633e-6
_NA = 1.5168        # crown-class model glass
_NB = 1.6200        # flint-class model glass
# Model glass for THIS module only: registered and removed by
# tests/conftest.py::_module_glass_registry_guard.
MODULE_GLASSES = {'_P8A': lambda wl: _NA,
                  '_P8B': lambda wl: _NB}

_ORACLE_PATH = (pathlib.Path(__file__).resolve().parents[2]
                / 'validation' / 'oracles' / 'debye_oracle_v3.py')
_spec = importlib.util.spec_from_file_location('debye_oracle_v3', _ORACLE_PATH)
_oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oracle)


# ---------------------------------------------------------------------------
# Helpers (mm-unit surface lists for the oracle; SI prescriptions for lumenairy)
# ---------------------------------------------------------------------------
def _singlet_presc(R1, R2, d, glass, ap):
    return {'name': 's', 'aperture_diameter': ap, 'surfaces': [
        {'radius': R1, 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': 'air', 'glass_after': glass},
        {'radius': R2, 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': glass, 'glass_after': 'air'}],
        'thicknesses': [d]}


def _surfs_mm_singlet(R1, R2, d, n):
    return [{'radius_mm': R1 * 1e3, 'thickness_mm': d * 1e3, 'index': n,
             'conic': 0.0},
            {'radius_mm': R2 * 1e3, 'thickness_mm': 0.0, 'index': 'air',
             'conic': 0.0}]


def _surfs_mm_doublet(R1, R2, R3, d1, d2, n1, n2, gap):
    return [{'radius_mm': R1 * 1e3, 'thickness_mm': d1 * 1e3, 'index': n1},
            {'radius_mm': R2 * 1e3, 'thickness_mm': d2 * 1e3, 'index': n2},
            {'radius_mm': R3 * 1e3, 'thickness_mm': gap * 1e3, 'index': 'air'}]


def _paraxial_img_efl(surfs_mm, R_in_mm=None):
    """(image dist from last vertex, EFL) mm; y-nu marginal trace."""
    h, u = 1.0, (0.0 if R_in_mm is None else 1.0 / R_in_mm)
    hE, uE, nbE, nb = 1.0, 0.0, 1.0, 1.0
    ns = len(surfs_mm)
    for i, s in enumerate(surfs_mm):
        R = s['radius_mm']
        n2 = 1.0 if s.get('index', 'air') == 'air' else float(s['index'])
        phi = 0.0 if (R == 0 or not np.isfinite(R)) else (n2 - nb) / R
        u = (nb * u - h * phi) / n2 if n2 else u
        uE = (nbE * uE - hE * phi) / n2 if n2 else uE
        if i < ns - 1:
            h += u * s['thickness_mm']
            hE += uE * s['thickness_mm']
        nb, nbE = n2, n2
    img = (-h / u) if u else np.inf
    efl = (-1.0 / uE) if uE else np.inf
    return img, efl


def _oracle_job(surfs_mm, wl, ap_mm, img_mm, w0_mm, R_in_mm=None, **extra):
    S = [dict(s) for s in surfs_mm]
    S[-1]['thickness_mm'] = img_mm
    job = {'wavelength_um': wl * 1e6, 'aperture_mm': ap_mm, 'R_in_mm': R_in_mm,
           'pop': {'w0_mm': w0_mm}, 'surfaces': S}
    job.update(extra)
    return job


def _gauss(N, dx, w0, R_in=None):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r2 = X ** 2 + Y ** 2
    amp = np.exp(-r2 / w0 ** 2)
    if R_in is None:
        return amp.astype(np.complex128)
    K = 2 * np.pi / _WL
    return (amp * np.exp(1j * K * r2 / (2 * R_in))).astype(np.complex128)


def _ee_metrics(I, dx, win=400e-6):
    n = I.shape[0]
    x = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(x, x)
    jp, ip = np.unravel_index(np.argmax(I), I.shape)
    R = np.sqrt((X - x[ip]) ** 2 + (Y - x[jp]) ** 2)
    m = R <= win
    Iw, Rw, Pw = I[m], R[m], I[m].sum()
    rb = np.linspace(0, win, 600)
    cum = np.array([Iw[Rw <= rr].sum() for rr in rb]) / Pw
    r2m = float(np.sqrt((Iw * Rw ** 2).sum() / Pw))
    return r2m, float(np.interp(0.5, cum, rb)), float(np.interp(0.8, cum, rb))


# ===========================================================================
# STEP A -- Huygens-PSF oracle (lumenairy-free half; Zemax half in the report)
# ===========================================================================
def test_stepA_airy_first_zero_relation():
    """The N0.2 Airy acceptance uses first_zero = 1.22*lambda*(EFL/EPD).  Pin the
    relation the ZOS Huygens-PSF mode was validated against (Zemax measured
    40.32 um vs this 40.15 um, 0.4%, Strehl 1.0 -- recorded in the report)."""
    efl_mm, epd_mm = 100.497, 4.0
    airy_um = 1.22 * (_WL * 1e6) * (efl_mm / epd_mm)
    assert airy_um == pytest.approx(40.15, abs=0.1)


@pytest.mark.slow
def test_stepA_debye_uniform_pupil_reference_stable():
    """The ZOS Huygens-PSF cross-check compares to ``debye_oracle_v3`` in a
    UNIFORM-pupil configuration (large Gaussian w0 -> flat over the clipped
    aperture).  Pin that lumenairy-free reference: the f/2 equiconvex singlet
    stopped to f/4 (aperture 12 mm) reproduces the recorded EE radii (the Zemax
    Huygens PSF matched these to EE50 1.9% / EE80 0.72% / EE95 0.17% at a matched
    110-um metric window -- see validation/oracles/README.md)."""
    R = 51.68e-3
    surfs = _surfs_mm_singlet(R, -R, 5e-3, _NA)
    img_mm, _ = _paraxial_img_efl(surfs)
    job = _oracle_job(surfs, _WL, 12.0, img_mm, 100.0, R_in_mm=None,
                      n_fan=12000, n_rho=6000, rho_max_um=110.0, window_um=110.0)
    r = _oracle.evaluate(job)
    assert r['huy_method'] == 'ring_huygens'
    # recorded matched-window Debye reference (deterministic)
    assert r['huy_EE50_um'] == pytest.approx(38.99, rel=0.03)
    assert r['huy_EE80_um'] == pytest.approx(79.20, rel=0.03)
    assert r['huy_EE95_um'] == pytest.approx(99.29, rel=0.03)


# ===========================================================================
# STEP B -- composed doublet + relay end-to-end vs the Debye oracle
# ===========================================================================
def _run_composed_chain(WL, N=4096, dx=2.0e-6):
    """Propagate a collimated Gaussian through a weak doublet + relay singlet
    with the NEW stack, landing at the paraxial image plane.  Returns
    (r2m, EE50, EE80) [m] of the final spot + the routed relay method."""
    ap, w0 = 6.0e-3, 1.5e-3
    D = dict(R1=90e-3, R2=-60e-3, R3=-350e-3, d1=4e-3, d2=2e-3)
    GAP = 50e-3
    RL = dict(R1=41.3e-3, R2=-41.3e-3, d=3e-3)
    doublet = {'name': 'g1', 'aperture_diameter': ap, 'surfaces': [
        {'radius': D['R1'], 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': 'air', 'glass_after': '_P8A'},
        {'radius': D['R2'], 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': '_P8A', 'glass_after': '_P8B'},
        {'radius': D['R3'], 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': '_P8B', 'glass_after': 'air'}],
        'thicknesses': [D['d1'], D['d2']]}
    relay = _singlet_presc(RL['R1'], RL['R2'], RL['d'], '_P8A', ap)

    full = (_surfs_mm_doublet(D['R1'], D['R2'], D['R3'], D['d1'], D['d2'],
                              _NA, _NB, GAP)
            + _surfs_mm_singlet(RL['R1'], RL['R2'], RL['d'], _NA))
    img_mm, _ = _paraxial_img_efl(full)

    E0 = _gauss(N, dx, w0, None)
    E1 = np.asarray(la.apply_real_lens_traced(
        E0, prescription=doublet, wavelength=WL, dx=dx, on_undersample='warn'))
    R1 = la.carrier_referenced_fit_radius(E1, WL, dx)
    env1 = la.carrier_referenced_envelope(E1, R1, WL, dx)
    env2, R2, dx2 = la.propagate_carrier_referenced(env1, R1, GAP, WL, dx)
    E_relay = np.asarray(la.carrier_referenced_reconstruct(
        np.asarray(env2), R2, WL, dx2))
    E3, method = la.apply_real_lens_universal(
        E_relay, prescription=relay, wavelength=WL, dx=dx2, return_method=True)
    E3 = np.asarray(E3)
    R3 = la.carrier_referenced_fit_radius(E3, WL, dx2)
    env3 = la.carrier_referenced_envelope(E3, R3, WL, dx2)
    env4, _R4, dx4 = la.propagate_carrier_referenced(
        env3, R3, img_mm * 1e-3, WL, dx2)
    r2m, e50, e80 = _ee_metrics(np.abs(np.asarray(env4)) ** 2, dx4, win=200e-6)
    return (r2m, e50, e80), method, (full, img_mm, ap, w0)


@pytest.mark.slow
def test_stepB_composed_doublet_relay_matches_debye():
    """The COMPOSITION (traced doublet + carrier gap leg + universal-gated relay +
    carrier final-focus leg) reproduces the independent Debye diffraction oracle
    to within 6% on EE50/EE80 at the paraxial image -- the integration nobody had
    demonstrated.  Also asserts the relay routed to the analytic phase-screen (the
    displaced fast path the N8 gate selects at its low relay NA)."""
    try:
        (r2m, e50, e80), method, (full, img_mm, ap, w0) = _run_composed_chain(_WL)
        assert method == 'phase_screen', method
        job = _oracle_job(full, _WL, ap * 1e3, img_mm, w0 * 1e3, R_in_mm=None,
                          n_fan=6000, n_rho=3000, rho_max_um=200.0,
                          window_um=200.0)
        d = _oracle.evaluate(job)
        assert d['huy_method'] == 'ring_huygens'
        assert e80 / (d['huy_EE80_um'] * 1e-6) == pytest.approx(1.0, abs=0.06)
        assert e50 / (d['huy_EE50_um'] * 1e-6) == pytest.approx(1.0, abs=0.06)
    finally:
        la.clear_asm_caches()


# ===========================================================================
# STEP C -- full-aperture f/2 biconvex to focus via the pilot beam (GBD)
# ===========================================================================
@pytest.mark.slow
def test_stepC_full_aperture_f2_pilot_beam_focus():
    """The full-aperture f/2 M4 biconvex at w0=9 mm reconstructed AT FOCUS via the
    GBD pilot beam on a MODEST N=3072 grid matches ``debye_oracle_v3`` (Gaussian
    w0=9 mm) EE80 to within 8%, with frame completeness > 0.95 -- feasible where
    the exact route needs N ~ 21000.  FAIL-BEFORE anchor: a fixed-grid phase-
    screen + ASM at the SAME modest N under-samples the ~0.64-NA exit and reads
    < 0.5x (aliased garbage), which is exactly why the pilot beam is needed."""
    R, ap, w0 = 51.68e-3, 22.0e-3, 9.0e-3
    presc = _singlet_presc(R, -R, 5e-3, '_P8A', ap)
    surfs = _surfs_mm_singlet(R, -R, 5e-3, _NA)
    img_mm, efl_mm = _paraxial_img_efl(surfs)
    bfd = img_mm * 1e-3

    job = _oracle_job(surfs, _WL, ap * 1e3, img_mm, w0 * 1e3, R_in_mm=None,
                      n_fan=8000, n_rho=3500, rho_max_um=400.0, window_um=400.0)
    d = _oracle.evaluate(job)
    d_e80 = d['huy_EE80_um'] * 1e-6

    N, dx = 3072, 8.0e-6
    E0 = _gauss(N, dx, w0, None)
    try:
        diag = {}
        E_focus = np.asarray(la.apply_real_lens_gbd(
            E0, prescription=presc, wavelength=_WL, dx=dx,
            beamlets_per_aperture=64, output_plane_distance=bfd,
            diagnostics=diag))
        _r, _e50, e80 = _ee_metrics(np.abs(E_focus) ** 2, dx, win=400e-6)
        assert diag['frame_completeness'] > 0.95, diag['frame_completeness']
        assert e80 / d_e80 == pytest.approx(1.0, abs=0.08), (e80, d_e80)

        # fail-before: fixed-grid thin+ASM at the SAME N cannot resolve the exit
        E_thin = np.asarray(la.apply_real_lens(
            E0, prescription=presc, wavelength=_WL, dx=dx,
            surface_model='thin'))
        E_asm = np.asarray(la.angular_spectrum_propagate(E_thin, bfd, _WL, dx))
        _r2, _e502, e80_fixed = _ee_metrics(np.abs(E_asm) ** 2, dx, win=400e-6)
        assert e80_fixed / d_e80 < 0.5, (e80_fixed, d_e80)

        # the exact fixed grid would need N well beyond the modest one used
        na_exit = 0.64
        n_needed = ap / (_WL / (2 * na_exit))
        assert n_needed > 3 * N
    finally:
        la.clear_asm_caches()


# ===========================================================================
# STEP D -- energy closure + grid-halving audit on the composed primitives
# ===========================================================================
def test_stepD_carrier_leg_energy_closure():
    """A carrier-referenced leg conserves total power on the co-moving area
    element (dx_out^2) -- the energy audit the capstone regression applies to
    every leg."""
    N, dx, w0 = 512, 4e-6, 120e-6
    env = _gauss(N, dx, w0, None)
    p_in = float(np.sum(np.abs(env) ** 2)) * dx * dx
    env_out, _R, dx_out = la.propagate_carrier_referenced(env, -40e-3, 12e-3,
                                                          _WL, dx)
    p_out = float(np.sum(np.abs(np.asarray(env_out)) ** 2)) * dx_out * dx_out
    assert p_out == pytest.approx(p_in, rel=2e-3)


def test_stepD_grid_halving_focus_metric_stable():
    """Halving dx (doubling N over the same extent) on the pilot-beam focus of a
    moderate singlet changes EE80 by < 3% -- the grid-convergence audit the final
    adversarial pass runs on every headline number."""
    R, ap, w0 = 51.68e-3, 8.0e-3, 3.0e-3          # f/6.4, tractable
    presc = _singlet_presc(R, -R, 5e-3, '_P8A', ap)
    surfs = _surfs_mm_singlet(R, -R, 5e-3, _NA)
    img_mm, _ = _paraxial_img_efl(surfs)
    bfd = img_mm * 1e-3
    e80s = []
    try:
        for N, dx in ((1536, 12e-6), (3072, 6e-6)):
            E0 = _gauss(N, dx, w0, None)
            E = np.asarray(la.apply_real_lens_gbd(
                E0, prescription=presc, wavelength=_WL, dx=dx,
                beamlets_per_aperture=48, output_plane_distance=bfd))
            _r, _e50, e80 = _ee_metrics(np.abs(E) ** 2, dx, win=200e-6)
            e80s.append(e80)
        assert abs(e80s[1] - e80s[0]) / e80s[0] < 0.03, e80s
    finally:
        la.clear_asm_caches()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
