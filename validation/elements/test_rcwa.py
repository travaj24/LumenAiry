"""Validation of the native RCWA module :mod:`lumenairy.elements.rcwa`.

Validation strategy (self-contained -- no external/GPL RCWA oracle is used
here; the GPL codes grcwa / inkstone are used only in the developer scratch
suite for absolute cross-checks):

* **Uniform-layer limit == analytic Airy thin-film.**  A grating with
  ``n_ridge == n_groove`` is a homogeneous slab; its rigorous zeroth-order
  reflectance must equal the exact Fresnel/Airy result for TE AND TM at
  normal and oblique incidence.  This exercises the entire machinery
  (eigenproblem, branch selection, all four S-matrix blocks, the star
  product, the efficiency formula) against a closed form.
* **Uniform-layer limit == the library's own TMM** (``coating_reflectance``)
  -- an independent in-library cross-check.
* **Energy conservation.**  Lossless ``sum(R)+sum(T) == 1``; lossy metal
  ``sum(R)+sum(T)+A == 1`` with ``A > 0`` (the absorptance sign is the
  exp(-i w t) / n=n+i kappa convention bridge).
* **High-order stability.**  The Re>=0 layer-eigenvalue branch keeps the
  solver convergent (no exponential blow-up) to large truncation.
* **Diffraction-order labelling.**  ``kx_m = kx0 + m lambda/period``; a
  symmetric grating at normal incidence is +/-m symmetric, an asymmetric
  duty cycle is not.
* **API contracts** -- shapes, validation errors, the wavelength sweep.
"""
from __future__ import annotations

import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from _harness import Harness

import numpy as np

import lumenairy as la
from lumenairy.elements.rcwa import (
    rcwa_efficiency_1d,
    rcwa_efficiency_vs_wavelength,
)

H = Harness('rcwa')

WL = 0.633e-6


# ----------------------------------------------------------------------------
# Analytic single-film Airy reference (exp(-i w t)/exp(+i k z)).
# ----------------------------------------------------------------------------

def _airy_R(n_sup, n_film, n_sub, d, wl, angle, pol):
    """Exact single-film power reflectance for TE/TM."""
    C = np.complex128
    kx = np.real(n_sup) * np.sin(angle)

    def kz(n):
        return np.sqrt(C(n) ** 2 - kx ** 2)
    kzs, kzf, kzt = kz(n_sup), kz(n_film), kz(n_sub)
    if pol == 'te':
        r01 = (kzs - kzf) / (kzs + kzf)
        r12 = (kzf - kzt) / (kzf + kzt)
    else:
        e0, e1, e2 = C(n_sup) ** 2, C(n_film) ** 2, C(n_sub) ** 2
        r01 = (e1 * kzs - e0 * kzf) / (e1 * kzs + e0 * kzf)
        r12 = (e2 * kzf - e1 * kzt) / (e2 * kzf + e1 * kzt)
    ph = np.exp(2j * (2 * np.pi / wl) * kzf * d)
    r = (r01 + r12 * ph) / (1 + r01 * r12 * ph)
    return float(np.abs(r) ** 2)


# ----------------------------------------------------------------------------
# 1. Uniform-layer limit == analytic Airy
# ----------------------------------------------------------------------------

def t_uniform_airy():
    cases = [
        (1.0, 1.5, 1.5, 0.40e-6, 0.0),
        (1.0, 2.2, 1.5, 0.30e-6, 0.0),
        (1.0, 2.2, 1.5, 0.30e-6, np.deg2rad(30)),
        (1.5, 1.0, 1.5, 0.50e-6, np.deg2rad(15)),
        (1.0, 1.8, 1.0, 0.70e-6, np.deg2rad(45)),
    ]
    worst = 0.0
    for n_sup, n_film, n_sub, d, ang in cases:
        for pol in ('te', 'tm'):
            o, R, T = rcwa_efficiency_1d(
                0.5e-6, n_film, n_film, n_sub, n_sup, d, 0.5, WL,
                angle=ang, polarization=pol, n_orders=5)
            R0 = R[len(o) // 2]
            worst = max(worst, abs(R0 - _airy_R(n_sup, n_film, n_sub, d, WL, ang, pol)))
    return worst < 1e-9, f'max |R_rcwa - R_airy| = {worst:.2e}'


# ----------------------------------------------------------------------------
# 2. Uniform-layer limit == library TMM (coating_reflectance)
# ----------------------------------------------------------------------------

def t_uniform_vs_coatings_tmm():
    worst = 0.0
    for n_film, d, ang, pol in [(2.2, 0.3e-6, 0.0, 'te'),
                                (2.2, 0.3e-6, np.deg2rad(25), 'te'),
                                (1.9, 0.45e-6, np.deg2rad(20), 'tm')]:
        Rc, Tc, Ac = la.coating_reflectance(
            [(n_film, d)], WL, angle=ang, n_substrate=1.5, n_ambient=1.0,
            polarization=('s' if pol == 'te' else 'p'))
        o, R, T = rcwa_efficiency_1d(
            0.5e-6, n_film, n_film, 1.5, 1.0, d, 0.5, WL,
            angle=ang, polarization=pol, n_orders=4)
        worst = max(worst, abs(R[len(o) // 2] - float(np.atleast_1d(Rc)[0])))
    return worst < 1e-9, f'max |R_rcwa - R_TMM| = {worst:.2e}'


# ----------------------------------------------------------------------------
# 3. Energy conservation (lossless)
# ----------------------------------------------------------------------------

def t_energy_lossless():
    worst = 0.0
    for ang in (0.0, np.deg2rad(20)):
        for pol in ('te', 'tm'):
            for M in (5, 11, 21):
                o, R, T = rcwa_efficiency_1d(
                    1.2e-6, 2.0, 1.0, 1.5, 1.0, 0.5e-6, 0.4, WL,
                    angle=ang, polarization=pol, n_orders=M)
                worst = max(worst, abs(R.sum() + T.sum() - 1.0))
    return worst < 1e-9, f'max |1 - (sumR+sumT)| = {worst:.2e}'


# ----------------------------------------------------------------------------
# 4. Metal grating: energy with positive absorptance (loss-sign bridge)
# ----------------------------------------------------------------------------

def t_metal_positive_absorptance():
    n_ag = 0.056 + 4.28j  # silver @ 633 nm (n + i kappa)
    min_A = 1.0
    worst_energy = 0.0
    for pol in ('te', 'tm'):
        o, R, T = rcwa_efficiency_1d(
            0.6e-6, n_ag, 1.0, 1.5, 1.0, 0.12e-6, 0.5, WL,
            angle=np.deg2rad(25), polarization=pol, n_orders=31)
        A = 1.0 - R.sum() - T.sum()
        min_A = min(min_A, A)
        # All orders must be finite and non-negative.
        worst_energy = max(worst_energy, 0.0 if np.all(np.isfinite(R)) and
                           np.all(np.isfinite(T)) else 1.0)
    ok = (min_A > 1e-3) and (worst_energy == 0.0)
    return ok, f'min absorptance = {min_A:.4f} (must be > 0), all finite={worst_energy==0.0}'


# ----------------------------------------------------------------------------
# 5. High-order stability (no blow-up; convergence)
# ----------------------------------------------------------------------------

def t_high_order_stable():
    vals = []
    for M in (5, 11, 21, 31):
        o, R, T = rcwa_efficiency_1d(
            1.2e-6, 2.0, 1.0, 1.5, 1.0, 0.5e-6, 0.4, WL,
            angle=0.0, polarization='tm', n_orders=M)
        vals.append(R[len(o) // 2])
        if not (np.all(np.isfinite(R)) and abs(R.sum() + T.sum() - 1) < 1e-8):
            return False, f'M={M}: energy/finiteness failed (blow-up)'
    drift = abs(vals[-1] - vals[-2])
    return drift < 1e-3, f'R0 converged to {vals[-1]:.6f}, |M31-M21| drift = {drift:.2e}'


# ----------------------------------------------------------------------------
# 6. Diffraction-order labelling / symmetry
# ----------------------------------------------------------------------------

def t_order_symmetry():
    # A single binary ridge is mirror-symmetric about its own centre, so at
    # NORMAL incidence the efficiencies are +/-m symmetric for any duty.
    o, R, T = rcwa_efficiency_1d(
        2.0e-6, 1.6, 1.0, 1.0, 1.0, 0.6e-6, 0.3, WL,
        angle=0.0, polarization='te', n_orders=8)
    mid = len(o) // 2
    sym_err = max(abs(R[mid + m] - R[mid - m]) for m in range(1, 6))
    # At OBLIQUE incidence the mirror symmetry is broken: +m and -m
    # diffract into different directions and carry different power.  This
    # also pins the kx_m = kx0 + m lambda/period order-sign convention
    # (a wrong sign would mirror-flip which side is brighter).
    o2, R2, T2 = rcwa_efficiency_1d(
        2.0e-6, 1.6, 1.0, 1.0, 1.0, 0.6e-6, 0.3, WL,
        angle=np.deg2rad(18), polarization='te', n_orders=8)
    mid2 = len(o2) // 2
    asym = max(abs(R2[mid2 + m] - R2[mid2 - m]) for m in range(1, 6))
    ok = (sym_err < 1e-9) and (asym > 1e-3)
    return ok, (f'normal +/-m err={sym_err:.1e} (->0), oblique spread='
                f'{asym:.1e} (>0)')


# ----------------------------------------------------------------------------
# 7. API contracts
# ----------------------------------------------------------------------------

def t_api_validation():
    msgs = []
    for kw, label in [(dict(polarization='circular'), 'bad polarization'),
                      (dict(duty_cycle=1.5), 'bad duty_cycle'),
                      (dict(formulation='spectral'), 'bad formulation')]:
        try:
            base = dict(period=1e-6, n_ridge=2.0, n_groove=1.0, n_substrate=1.5,
                        n_superstrate=1.0, depth=0.4e-6, duty_cycle=0.5,
                        wavelength=WL)
            base.update(kw)
            rcwa_efficiency_1d(**base)
            msgs.append(f'{label}: no error')
        except ValueError:
            pass
        except Exception as e:
            msgs.append(f'{label}: wrong type {type(e).__name__}')
    return not msgs, ('; '.join(msgs) or 'all bad inputs raise ValueError')


def t_shapes():
    o, R, T = rcwa_efficiency_1d(1e-6, 2.0, 1.0, 1.5, 1.0, 0.4e-6, 0.5, WL,
                                 n_orders=7)
    ok = (o.shape == (15,) == R.shape == T.shape and o[0] == -7 and o[-1] == 7
          and np.all(R >= 0) and np.all(T >= 0))
    return ok, f'orders {o.shape}, R {R.shape}, T {T.shape}'


# ----------------------------------------------------------------------------
# 8. Wavelength sweep
# ----------------------------------------------------------------------------

def t_wavelength_sweep():
    wls = np.linspace(0.5e-6, 0.7e-6, 9)
    eff = rcwa_efficiency_vs_wavelength(
        1.4e-6, 2.0, 1.0, 1.5, 1.0, 0.5e-6, 0.5, wls,
        order=1, polarization='te', n_orders=11, quantity='transmitted')
    # Must equal the per-call value at each wavelength.
    worst = 0.0
    for i, w in enumerate(wls):
        o, R, T = rcwa_efficiency_1d(1.4e-6, 2.0, 1.0, 1.5, 1.0, 0.5e-6, 0.5,
                                     float(w), polarization='te', n_orders=11)
        worst = max(worst, abs(eff[i] - T[np.searchsorted(o, 1)]))
    scalar = rcwa_efficiency_vs_wavelength(
        1.4e-6, 2.0, 1.0, 1.5, 1.0, 0.5e-6, 0.5, 0.6e-6, order=1)
    ok = (eff.shape == (9,)) and worst < 1e-12 and np.isscalar(scalar)
    return ok, f'sweep shape {eff.shape}, max dev from per-call {worst:.1e}, scalar={np.isscalar(scalar)}'


H.run('uniform layer == analytic Airy (TE+TM, oblique)', t_uniform_airy)
H.run('uniform layer == library TMM (coating_reflectance)', t_uniform_vs_coatings_tmm)
H.run('energy conservation (lossless)', t_energy_lossless)
H.run('metal grating: positive absorptance + finite', t_metal_positive_absorptance)
H.run('high-order stability + convergence to M=31', t_high_order_stable)
H.run('diffraction-order labelling / symmetry', t_order_symmetry)
H.run('API: bad inputs raise ValueError', t_api_validation)
H.run('API: output shapes and order range', t_shapes)
H.run('wavelength sweep matches per-call + scalar return', t_wavelength_sweep)


if __name__ == '__main__':
    import sys
    sys.exit(H.summary())
