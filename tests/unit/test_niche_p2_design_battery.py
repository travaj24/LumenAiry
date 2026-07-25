"""P2 DESIGN BATTERY for :func:`lumenairy.propagate_traced_carrier_chain`
(audit ``docs/audits/AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24.md`` §5 P2,
"Design test battery ... replacing reliance on the 121 alone").

WHY THIS FILE EXISTS
--------------------
Every fidelity claim about the traced-carrier chain had been anchored on ONE
design (the design-121 relay, which needs a proprietary ``.zmx``).  This file
is the CI-safe, self-contained replacement: four designs -- a fast singlet, a
cemented achromat, a 3-group air-spaced triplet and the E4-style corrected
relay -- each at two beam sizes (two exit NAs) and two aperture:beam ratios
that straddle the aperture:beam cliff, gated against an INLINE exact
meridional-raytrace + eikonal oracle (pure ray trace, grid-independent) and,
for the through-focus cells, against the analytic Gaussian waist.

It pins the CHAIN's v5.29 defaults (``carrier_reference='sphere'`` +
``amplitude_model='ray_density'`` + ``preserve_input_phase='remap'`` +
``fit_radius_beam_factor=2.0``) across that envelope.

KNOWN-GOOD ENVELOPE (measured 2026-07-25, N=1024, ray_subsample=4)
------------------------------------------------------------------
* **Aperture:beam** -- 1.2x to 2.5x the beam 1/e^2 diameter, WITH the default
  cliff guard.  Without it (``fit_radius_beam_factor=None``) two of the twelve
  cells collapse to a random exit phase (1.82 rad rms = pi/sqrt(3)); see
  ``test_battery_cliff_cells_need_the_guard``.  Below ~1.5x the aperture
  TRUNCATES the beam (a real, physical effect: the focus broadens to ~1.2x the
  untruncated Gaussian FWHM and EE within 2 waists drops 99.8 -> 92%).
* **Exit NA** -- 0.013 to 0.20 (the largest here is the fast singlet at
  w0 = 1.2 mm).  The design-121 acceptance covers 0.152 with a high-NA final
  leg; NA > ~0.5 is out of scope for this file (it needs the exact final leg
  plus a memory budget -- see test_niche_r9_highna_final_leg.py).
* **Wavefront agreement** -- chain exit-wavefront rms tracks the ray oracle to
  within ``max(0.15 rad, 0.35 x rms_oracle)`` on every cell, INCLUDING the
  deliberately-aberrated fast singlet at 0.79 rad rms (i.e. the chain
  reproduces a bad design's aberration, it does not flatter it).
* **Through focus** -- an unclipped, well-corrected cell lands within 1.10x of
  the analytic Gaussian FWHM with >= 95% of the launched power inside two
  waists.

The oracle is the same construction as
``test_niche_e4_corrected_relay_oracle.py`` (documented there); the
through-focus scan follows ``validation/repro_traced_carrier_121/focus_scan_121.py``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy import carrier_referenced_envelope
from lumenairy.glass import get_glass_index
from lumenairy.propagators.carrier import _paraxial_group_r_out
from lumenairy.propagators.mft import angular_spectrum_propagate_mft

_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL
_N = 1024
_RS = 4


def _skip_if_low_ram(n=_N):
    try:
        import psutil
        if psutil.virtual_memory().available < 4 * (n * n * 16) + (1 << 30):
            pytest.skip(f'insufficient free RAM for the N={n} battery')
    except ImportError:
        pass


def _n_of(g):
    return 1.0 if g in (None, '', 'air', 'AIR') else float(get_glass_index(g, _WL))


def _presc(name, radii, thick, glasses, ap):
    """Prescription from surface radii + the medium AFTER each surface."""
    surfs = []
    before = 'air'
    for R, g in zip(radii, glasses):
        surfs.append({'radius': R, 'glass_before': before, 'glass_after': g,
                      'conic': 0.0, 'radius_y': None, 'conic_y': None,
                      'aspheric_coeffs': None, 'aspheric_coeffs_y': None})
        before = g
    return {'name': name, 'aperture_diameter': ap, 'thicknesses': list(thick),
            'surfaces': surfs}


# ---------------------------------------------------------------------------
# The four designs.  Each returns [(prescription, gap_before), ...].
# ---------------------------------------------------------------------------
def _d_singlet(ap):
    """Fast (f/4.5-class) biconvex BK7 singlet -- deliberately UNCORRECTED, so
    the oracle reports real spherical aberration the chain has to reproduce."""
    return [(_presc('fast_singlet', [9e-3, -9e-3], [3e-3], ['N-BK7', 'air'],
                    ap), 0.0)]


def _d_doublet(ap):
    """Cemented achromat (N-BK7 / N-SF5), f ~ 50 mm -- well corrected."""
    return [(_presc('doublet', [31.6e-3, -25.1e-3, -172.0e-3],
                    [5.5e-3, 2.0e-3], ['N-BK7', 'N-SF5', 'air'], ap), 0.0)]


def _d_triplet(ap):
    """Three air-spaced singlets as three CHAIN GROUPS (positive / negative /
    positive).  Not hand-optimised: the oracle supplies whatever aberration the
    design has and the chain must match it, which is the actual property under
    test.  Also exercises two inter-group Sziklas-Siegman legs."""
    return [(_presc('t1', [22.0e-3, -100.0e-3], [3.0e-3], ['N-BK7', 'air'],
                    ap), 0.0),
            (_presc('t2', [-30.0e-3, 30.0e-3], [2.0e-3], ['N-SF5', 'air'],
                    ap), 4e-3),
            (_presc('t3', [40.0e-3, -22.0e-3], [3.0e-3], ['N-BK7', 'air'],
                    ap), 4e-3)]


def _d_relay(ap):
    """The E4 corrected relay: fast biconvex + tuned negative corrector."""
    return [(_presc('g1', [18e-3, -18e-3], [3e-3], ['N-BK7', 'air'], ap), 0.0),
            (_presc('g2', [-9.342e-3, 36.508e-3], [3e-3], ['N-BK7', 'air'],
                    ap), 5e-3)]


# ---------------------------------------------------------------------------
# Inline exact meridional-raytrace + eikonal oracle (grid-independent truth).
# ---------------------------------------------------------------------------
def _system_surfaces(groups_with_gaps):
    surfs = []
    z = 0.0
    for (p, gap) in groups_with_gaps:
        z += gap
        th = p['thicknesses']
        for i, s in enumerate(p['surfaces']):
            R_s = s['radius'] if s['radius'] else np.inf
            surfs.append((z, R_s, _n_of(s['glass_before']),
                          _n_of(s['glass_after'])))
            if i < len(p['surfaces']) - 1:
                z += th[i] if i < len(th) else 0.0
    return surfs, surfs[-1][0]


def _trace_collimated(x0, surfs):
    P = np.array([x0, 0.0]); u = np.array([0.0, 1.0]); opl = 0.0
    for (zv, R_s, n1, n2) in surfs:
        if np.isfinite(R_s):
            C = np.array([0.0, zv + R_s]); oc = P - C
            b = np.dot(oc, u); c = np.dot(oc, oc) - R_s * R_s
            disc = b * b - c
            if disc < 0:
                return None
            t = min((-b - np.sqrt(disc), -b + np.sqrt(disc)),
                    key=lambda tt: abs((P + tt * u)[1] - zv))
            P2 = P + t * u
            nv = np.sign(R_s) * (C - P2) / np.linalg.norm(C - P2)
        else:
            if abs(u[1]) < 1e-12:
                return None
            t = (zv - P[1]) / u[1]; P2 = P + t * u; nv = np.array([0.0, 1.0])
        opl += n1 * t
        c1 = np.dot(u, nv); eta = n1 / n2
        d2 = 1.0 - eta * eta * (1.0 - c1 * c1)
        if d2 < 0:
            return None
        u = eta * u + (np.sqrt(d2) - eta * c1) * nv
        u = u / np.linalg.norm(u); P = P2
    return P, u, opl


def _oracle_rms(groups_with_gaps, x_max, w0, npts=240):
    """Gaussian-weighted exit-pupil wavefront rms (RADIANS) vs a reference
    sphere on the paraxial image, piston+defocus removed."""
    surfs, z_last = _system_surfaces(groups_with_gaps)
    n_final = surfs[-1][3]
    near = _trace_collimated(x_max * 0.999, surfs)
    if near is None:
        return None
    Pn, un, _ = near
    z_f = Pn[1] + (-Pn[0] / un[0]) * un[1]
    F = np.array([0.0, z_f]); rho = z_f - z_last
    x0s = np.linspace(1e-9, x_max, npts)
    xe = np.full(npts, np.nan); W = np.full(npts, np.nan)
    for j, x0 in enumerate(x0s):
        tr = _trace_collimated(x0, surfs)
        if tr is None:
            continue
        P, u, opl = tr
        oc = P - F; b = np.dot(oc, u); c = np.dot(oc, oc) - rho * rho
        disc = b * b - c
        if disc < 0:
            continue
        t = min((-b - np.sqrt(disc), -b + np.sqrt(disc)), key=lambda tt: abs(tt))
        opl += n_final * t
        xe[j] = (P + t * u)[0]; W[j] = opl
    good = np.isfinite(W); x0g = x0s[good]; xe, W = xe[good], W[good]
    wgt = np.exp(-2.0 * (x0g / w0) ** 2)
    W = W - W[0]
    A = np.vstack([np.ones_like(xe), xe ** 2]).T
    sw = np.sqrt(wgt)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], W * sw, rcond=None)
    W = W - A @ coef
    return float(_K0 * np.sqrt(np.sum(wgt * W ** 2) / np.sum(wgt)))


# ---------------------------------------------------------------------------
# Chain runners (cached -- cells are shared between tests).
# ---------------------------------------------------------------------------
_CACHE = {}


def _grid(ap, w0):
    dx = float(2.2 * max(ap, 3.0 * w0) / _N)
    return _N, dx


def _launch(ap, w0):
    N, dx = _grid(ap, w0)
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128), dx


def _run_chain(design, w0, ratio, guard='default', **kw):
    ap = ratio * 2.0 * w0
    gwg = design(ap)
    groups = [{'prescription': p, 'gap_before': g} for (p, g) in gwg]
    env0, dx = _launch(ap, w0)
    tkw = dict(parallel_amp=False, on_undersample='silent')
    if guard != 'default':
        tkw['fit_radius_beam_factor'] = guard
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return la.propagate_traced_carrier_chain(
            env0, groups, _WL, dx, r_in=np.inf, ray_subsample=_RS, n_workers=1,
            traced_kwargs=tkw, **kw), gwg, env0, dx


def _chain_exit_rms(design, w0, ratio, guard='default'):
    """Exit-wavefront rms (radians) at the last group's exit vertex, referenced
    to the paraxial exit sphere with piston/tilt/defocus removed -- the
    focus-plane-INDEPENDENT fidelity metric."""
    key = (design.__name__, w0, ratio, guard)
    if key in _CACHE:
        return _CACHE[key]
    res, _, _, _ = _run_chain(design, w0, ratio, guard, final_distance=0.0)
    env = carrier_referenced_envelope(np.asarray(res.field), float(res.R), _WL,
                                      float(res.dx))
    n = env.shape[0]
    xx = (np.arange(n) - n / 2) * float(res.dx)
    XX, YY = np.meshgrid(xx, xx)
    amp = np.abs(env)
    mask = amp > 0.10 * amp.max()
    ph = np.angle(env)
    w = amp[mask] ** 2
    A = np.stack([np.ones(int(mask.sum())), XX[mask], YY[mask],
                  XX[mask] ** 2 + YY[mask] ** 2], 1)
    coef, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], ph[mask] * np.sqrt(w),
                               rcond=None)
    resid = np.angle(np.exp(1j * (ph[mask] - A @ coef)))
    out = float(np.sqrt(np.sum(w * resid ** 2) / np.sum(w)))
    _CACHE[key] = out
    return out


def _exit_carrier_radius(gwg):
    """Paraxial exit carrier radius of the whole train for a collimated input
    (the geometric focus sits ``-R`` past the last vertex)."""
    R = np.inf
    for (p, g) in gwg:
        if np.isfinite(R):
            R = R + g
        R = _paraxial_group_r_out(p, R, _WL)
    return float(R)


def _through_focus(design, w0, ratio, guard='default', dx_out=0.5e-6,
                   n_out=512, n_steps=9):
    """Best-focus metrics from a through-focus scan around the geometric focus,
    plus the analytic Gaussian prediction.  Returns
    ``(fwhm, fwhm_theory, {n_waists: EE}, dz_best)``."""
    ap = ratio * 2.0 * w0
    gwg = design(ap)
    R = _exit_carrier_radius(gwg)
    res, _, env0, dx = _run_chain(
        design, w0, ratio, guard, final_distance=-R,
        focus_readout=dict(dx_out=dx_out, N_out=n_out))
    P_in = float((np.abs(env0) ** 2).sum()) * dx * dx
    E0 = np.asarray(res.field)
    w_exit = float(res.stages[-1]['w'])
    w_f = _WL * abs(R) / (np.pi * w_exit)
    fwhm_th = w_f * np.sqrt(2.0 * np.log(2.0))
    zR = np.pi * w_f * w_f / _WL
    best = (0.0, -1.0, None)
    for dz in np.linspace(-zR, zR, n_steps):
        Ez = E0 if dz == 0 else angular_spectrum_propagate_mft(
            E0, float(dz), _WL, dx_out, dx_out, n_out)
        I = np.abs(Ez) ** 2
        pk = float(I.max())
        if pk > best[1]:
            best = (float(dz), pk, I)
    dz, pk, I = best
    iy, ix = np.unravel_index(int(np.argmax(I)), I.shape)
    xx = (np.arange(n_out) - ix) * dx_out
    yy = (np.arange(n_out) - iy) * dx_out
    rr = np.sqrt(xx[None, :] ** 2 + yy[:, None] ** 2)
    nb = n_out // 2
    ring = np.clip((rr / dx_out).astype(int), 0, nb)
    s = np.bincount(ring.ravel(), weights=I.ravel(), minlength=nb + 1)
    cnt = np.bincount(ring.ravel(), minlength=nb + 1)
    prof = (s[:nb] / np.maximum(cnt[:nb], 1)) / I[iy, ix]
    rb = (np.arange(nb) + 0.5) * dx_out
    idx = np.where(prof < 0.5)[0]
    fwhm = float(2 * rb[idx[0]]) if len(idx) else float('nan')
    ee = {m: float(I[rr <= m * w_f].sum()) * dx_out ** 2 / P_in
          for m in (1, 2, 3)}
    return fwhm, float(fwhm_th), ee, dz


# ===========================================================================
# 1. Wavefront: the chain tracks the ray oracle across the whole battery.
# ===========================================================================
_CELLS = [
    ('singlet', _d_singlet, 0.6e-3, 1.2), ('singlet', _d_singlet, 0.6e-3, 2.5),
    ('singlet', _d_singlet, 1.2e-3, 1.2), ('singlet', _d_singlet, 1.2e-3, 2.5),
    ('doublet', _d_doublet, 1.0e-3, 1.2), ('doublet', _d_doublet, 1.0e-3, 2.5),
    ('doublet', _d_doublet, 2.0e-3, 1.2), ('doublet', _d_doublet, 2.0e-3, 2.5),
    ('triplet', _d_triplet, 1.6e-3, 1.2), ('triplet', _d_triplet, 1.6e-3, 2.5),
    ('relay', _d_relay, 2.0e-3, 1.2), ('relay', _d_relay, 2.0e-3, 2.5),
]


@pytest.mark.parametrize('name,design,w0,ratio', _CELLS,
                         ids=[f'{c[0]}-w{c[2] * 1e3:g}mm-ap{c[3]:g}x'
                              for c in _CELLS])
def test_battery_wavefront_matches_ray_oracle(name, design, w0, ratio):
    """With the chain's v5.29 defaults, the traced exit wavefront agrees with
    the independent meridional ray oracle to within
    ``max(0.15 rad, 0.35 x rms_oracle)`` on every battery cell -- including the
    fast singlet, whose 0.79 rad of real spherical aberration the chain must
    REPRODUCE rather than flatter.

    Measured 2026-07-25 (rms rad, oracle -> chain):
    singlet w0.6 1.2x 0.049 -> 0.025 | 2.5x 0.049 -> 0.054
    singlet w1.2 1.2x 0.795 -> 0.946 | 2.5x 0.795 -> 1.018
    doublet w1.0 1.2x 0.001 -> 0.003 | 2.5x 0.001 -> 0.009
    doublet w2.0 1.2x 0.009 -> 0.016 | 2.5x 0.009 -> 0.041
    triplet w1.6 1.2x 0.032 -> 0.028 | 2.5x 0.032 -> 0.046
    relay   w2.0 1.2x 0.003 -> 0.019 | 2.5x 0.003 -> 0.100
    """
    _skip_if_low_ram()
    ap = ratio * 2.0 * w0
    orc = _oracle_rms(design(ap), 1.5 * w0, w0)
    assert orc is not None, 'oracle trace failed (design is unphysical?)'
    rms = _chain_exit_rms(design, w0, ratio)
    tol = max(0.15, 0.35 * orc)
    assert abs(rms - orc) <= tol, (
        f'{name} w0={w0 * 1e3:g}mm ap={ratio:g}x: chain exit rms {rms:.4f} rad '
        f'vs oracle {orc:.4f} rad (tolerance {tol:.4f})')


# ===========================================================================
# 2. The cliff cells: the guard is what keeps the envelope inside tolerance.
# ===========================================================================
@pytest.mark.parametrize('name,design,w0', [('singlet', _d_singlet, 1.2e-3),
                                            ('relay', _d_relay, 2.0e-3)],
                         ids=['singlet-w1.2mm', 'relay-w2mm'])
def test_battery_cliff_cells_need_the_guard(name, design, w0):
    """Two battery cells sit ON the aperture:beam cliff at 2.5x: with the
    pre-v5.29 aperture-only ray-fit domain the exit phase degenerates to NOISE
    (1.82 rad rms = pi/sqrt(3), a uniformly random wrapped phase), and the
    default beam-relative fit domain is what keeps them physical.

    Measured 2026-07-25 (rms rad, guard off -> on): singlet 1.824 -> 1.018,
    relay 1.822 -> 0.100."""
    _skip_if_low_ram()
    off = _chain_exit_rms(design, w0, 2.5, guard=None)
    on = _chain_exit_rms(design, w0, 2.5)
    assert off > 1.5, (
        f'{name}: expected the un-guarded cliff (>1.5 rad rms), got {off:.4f}')
    assert on < 0.7 * off, (
        f'{name}: guard must recover the cliff cell ({off:.4f} -> {on:.4f} rad)')


# ===========================================================================
# 3. Through focus (the mandatory fidelity methodology) vs the analytic waist.
# ===========================================================================
def test_battery_through_focus_unclipped_doublet_matches_gaussian():
    """An unclipped (2.5x aperture), well-corrected cell reproduces the
    ANALYTIC Gaussian focus: best-focus FWHM within 10% of
    ``1.177 * lambda|R|/(pi w_exit)`` and >= 95% of the launched power inside
    two waists.  Measured 2026-07-25: FWHM 18.5 um vs 17.4 um theory (1.062x),
    EE1w 86.0% / EE2w 99.7% / EE3w 99.8%."""
    _skip_if_low_ram()
    fwhm, fwhm_th, ee, _ = _through_focus(_d_doublet, 2.0e-3, 2.5)
    assert fwhm / fwhm_th < 1.10, (fwhm, fwhm_th)
    assert ee[2] > 0.95, ee
    assert ee[3] >= ee[2], ee


def test_battery_through_focus_truncated_aperture_broadens_predictably():
    """At 1.2x the beam diameter the aperture TRUNCATES the beam -- a real,
    physical effect, not a numerical defect: the focus broadens to ~1.2x the
    untruncated Gaussian FWHM and the two-waist encircled energy drops from
    ~99.8% to ~92% (power moves into the diffraction rings).  Pinned so a
    future regression cannot hide inside 'it's just truncation'.  Measured
    2026-07-25: FWHM 23.5 um vs 19.2 um theory (1.226x), EE2w 92.3%."""
    _skip_if_low_ram()
    fwhm, fwhm_th, ee, _ = _through_focus(_d_doublet, 2.0e-3, 1.2)
    assert 1.10 < fwhm / fwhm_th < 1.40, (fwhm, fwhm_th)
    assert 0.85 < ee[2] < 0.97, ee


def test_battery_through_focus_relay_cliff_is_a_focal_catastrophe():
    """The aperture:beam cliff expressed in FOCAL terms (the metric a user
    actually reads): on the corrected relay at a 2.5x aperture, the un-guarded
    chain puts 0.2% of the launched power inside one waist -- the focus is
    simply gone -- while the default guard delivers 81%.  This is the silent
    mis-report the P2 guard exists to prevent.  Measured 2026-07-25."""
    _skip_if_low_ram()
    fw_on, th_on, ee_on, _ = _through_focus(_d_relay, 2.0e-3, 2.5)
    fw_off, th_off, ee_off, _ = _through_focus(_d_relay, 2.0e-3, 2.5,
                                               guard=None)
    assert ee_on[1] > 0.7, ee_on
    assert ee_off[1] < 0.10, ee_off
    assert fw_on / th_on < 1.25, (fw_on, th_on)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
