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
from lumenairy.propagators import carrier as C
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


def _n_of(g, wl=_WL):
    return (1.0 if g in (None, '', 'air', 'AIR')
            else float(get_glass_index(g, wl)))


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
def _system_surfaces(groups_with_gaps, wl=_WL):
    surfs = []
    z = 0.0
    for (p, gap) in groups_with_gaps:
        z += gap
        th = p['thicknesses']
        for i, s in enumerate(p['surfaces']):
            R_s = s['radius'] if s['radius'] else np.inf
            surfs.append((z, R_s, _n_of(s['glass_before'], wl),
                          _n_of(s['glass_after'], wl)))
            if i < len(p['surfaces']) - 1:
                z += th[i] if i < len(th) else 0.0
    return surfs, surfs[-1][0]


def _trace_collimated(x0, surfs):
    P = np.array([x0, 0.0])
    u = np.array([0.0, 1.0])
    opl = 0.0
    for (zv, R_s, n1, n2) in surfs:
        if np.isfinite(R_s):
            C = np.array([0.0, zv + R_s])
            oc = P - C
            b = np.dot(oc, u)
            c = np.dot(oc, oc) - R_s * R_s
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
            t = (zv - P[1]) / u[1]
            P2 = P + t * u
            nv = np.array([0.0, 1.0])
        opl += n1 * t
        c1 = np.dot(u, nv)
        eta = n1 / n2
        d2 = 1.0 - eta * eta * (1.0 - c1 * c1)
        if d2 < 0:
            return None
        u = eta * u + (np.sqrt(d2) - eta * c1) * nv
        u = u / np.linalg.norm(u)
        P = P2
    return P, u, opl


def _oracle_rms(groups_with_gaps, x_max, w0, npts=240, wl=_WL):
    """Gaussian-weighted exit-pupil wavefront rms (RADIANS) vs a reference
    sphere on the paraxial image, piston+defocus removed."""
    surfs, z_last = _system_surfaces(groups_with_gaps, wl)
    n_final = surfs[-1][3]
    near = _trace_collimated(x_max * 0.999, surfs)
    if near is None:
        return None
    Pn, un, _ = near
    z_f = Pn[1] + (-Pn[0] / un[0]) * un[1]
    F = np.array([0.0, z_f])
    rho = z_f - z_last
    x0s = np.linspace(1e-9, x_max, npts)
    xe = np.full(npts, np.nan)
    W = np.full(npts, np.nan)
    for j, x0 in enumerate(x0s):
        tr = _trace_collimated(x0, surfs)
        if tr is None:
            continue
        P, u, opl = tr
        oc = P - F
        b = np.dot(oc, u)
        c = np.dot(oc, oc) - rho * rho
        disc = b * b - c
        if disc < 0:
            continue
        t = min((-b - np.sqrt(disc), -b + np.sqrt(disc)), key=lambda tt: abs(tt))
        opl += n_final * t
        xe[j] = (P + t * u)[0]
        W[j] = opl
    good = np.isfinite(W)
    x0g = x0s[good]
    xe, W = xe[good], W[good]
    wgt = np.exp(-2.0 * (x0g / w0) ** 2)
    W = W - W[0]
    A = np.vstack([np.ones_like(xe), xe ** 2]).T
    sw = np.sqrt(wgt)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], W * sw, rcond=None)
    W = W - A @ coef
    return float((2.0 * np.pi / wl)
                 * np.sqrt(np.sum(wgt * W ** 2) / np.sum(wgt)))


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


def _run_chain(design, w0, ratio, guard='default', wl=_WL, **kw):
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
            env0, groups, wl, dx, r_in=np.inf, ray_subsample=_RS, n_workers=1,
            traced_kwargs=tkw, **kw), gwg, env0, dx


def _wavefront_rms(env, dx_out):
    """Amplitude-weighted exit-wavefront rms (RADIANS) of a carrier-referenced
    envelope, with piston, tilt and defocus removed.

    INVARIANT, and it is the point of the function: the returned number does
    NOT move when ``env`` is multiplied by a global unit phasor.  Pinned by
    ``test_wavefront_rms_is_invariant_under_a_global_piston``.
    """
    n = env.shape[0]
    xx = (np.arange(n) - n / 2) * float(dx_out)
    XX, YY = np.meshgrid(xx, xx)
    amp = np.abs(env)
    mask = amp > 0.10 * amp.max()
    w = amp[mask] ** 2
    ph = np.angle(env)[mask]
    # ---- PISTON REFERENCE (docs/audits/FIX_PR29_BLAST_2026_08_11.md S1) ----
    # ``ph`` is a CIRCULAR variable read off a wrapped ``np.angle``; the linear
    # least-squares below is not.  Its piston COLUMN can slide the model, but
    # nothing in it can undo a +-2 pi branch jump in the DATA -- so before this
    # reference the metric was piston-invariant only by LUCK: whichever global
    # phase the element happened to return had to leave the beam's phase
    # cluster clear of the +-pi cut.  ``apply_real_lens_traced`` now returns an
    # ABSOLUTE optical path (docs/audits/FIX_TILT_QUADRATIC_OPL_2026_08_11.md),
    # i.e. the global phase is the element's own ``k0 * OPL`` (~5e+04 rad),
    # whose value mod 2 pi is arbitrary as far as a WAVEFRONT metric is
    # concerned.  Two cells duly landed on the cut (the triplet at 98.6 % of
    # its masked pixels within 0.35 rad of +-pi) and the fit returned noise.
    #
    # De-rotate the amplitude-weighted CIRCULAR MEAN first.  The circular mean
    # is equivariant -- ``ph -> ph + c`` gives ``mu -> mu + c`` -- so
    # ``ph - mu``, and therefore every number below, is EXACTLY invariant under
    # ``env -> env * exp(1j c)``, which is what this docstring always claimed.
    #
    # Measured across all 12 battery cells at v5.34.0 vs this branch: the
    # de-rotated rms is unchanged to <= 1.3e-16 relative on 8 cells and
    # <= 3e-08 on 3 more; the 12th (the relay at 2.5x, sitting on the cliff
    # shoulder) moves 1.5e-03, which is FFT non-commutativity through the
    # transport legs, not physics.
    mu = np.angle(np.sum(w * np.exp(1j * ph)))
    ph = np.angle(np.exp(1j * (ph - mu)))
    A = np.stack([np.ones(int(mask.sum())), XX[mask], YY[mask],
                  XX[mask] ** 2 + YY[mask] ** 2], 1)
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], ph * sw, rcond=None)
    resid = np.angle(np.exp(1j * (ph - A @ coef)))
    return float(np.sqrt(np.sum(w * resid ** 2) / np.sum(w)))


def _chain_exit_rms(design, w0, ratio, guard='default', wl=_WL):
    """Exit-wavefront rms (radians) at the last group's exit vertex, referenced
    to the paraxial exit sphere with piston/tilt/defocus removed -- the
    focus-plane-INDEPENDENT fidelity metric."""
    key = (design.__name__, w0, ratio, guard, wl)
    if key in _CACHE:
        return _CACHE[key]
    res, _, _, _ = _run_chain(design, w0, ratio, guard, wl=wl,
                              final_distance=0.0)
    env = carrier_referenced_envelope(np.asarray(res.field), float(res.R), wl,
                                      float(res.dx))
    out = _wavefront_rms(env, float(res.dx))
    _CACHE[key] = out
    return out


def _exit_carrier_radius(gwg, wl=_WL):
    """Paraxial exit carrier radius of the whole train for a collimated input
    (the geometric focus sits ``-R`` past the last vertex)."""
    R = np.inf
    for (p, g) in gwg:
        if np.isfinite(R):
            R = R + g
        R = _paraxial_group_r_out(p, R, wl)
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
        # D3 (2026-08-06): the through-focus scan below re-propagates this
        # field with its own MFT, and every metric it takes (FWHM, EE inside a
        # few waists) is confined to the core.  The requested window exceeds
        # one Bluestein period, as it always has; waived rather than shrunk
        # because n_out also sizes the scan's own transform.
        focus_readout=dict(dx_out=dx_out, N_out=n_out,
                           on_replica='ignore'))
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
# 0. The metric itself.  Everything below scores the chain with
#    ``_wavefront_rms``; if that number can move without the WAVEFRONT moving,
#    every row in the battery is measuring the wrong thing.
# ===========================================================================
def test_wavefront_rms_is_invariant_under_a_global_piston():
    """A global unit phasor is not a wavefront error -- it is the reference
    the metric is supposed to remove -- so ``_wavefront_rms`` must return the
    same number for ``env`` and for ``env * exp(1j c)`` at every ``c``.

    This is not hypothetical bookkeeping.  ``apply_real_lens_traced`` now
    returns an ABSOLUTE optical path (FIX_TILT_QUADRATIC_OPL_2026_08_11), so
    the exit field carries the element's own ``k0 * OPL`` -- tens of thousands
    of radians whose residue mod 2 pi is, to a wavefront metric, arbitrary.
    A linear least-squares run directly on wrapped ``np.angle`` output cannot
    absorb that: its piston column moves the MODEL, but the +-pi branch cut
    has already folded the DATA.  Measured with the pre-fix construction, this
    fixture's rms swings 0.2384 -> 1.3922 rad across the 17-point piston sweep
    below (a 5.8x spread on a quantity that must not move at all) -- the same
    failure that made two battery cells and the E4 7 mm cell report fiction.
    With the circular-mean reference the sweep is flat to 3.9e-16 rad.  The
    oracle here is exactness itself: the answer cannot depend on ``c``.
    """
    n, dxo, w0 = 256, 5.0e-6, 3.0e-4
    xx = (np.arange(n) - n / 2) * dxo
    XX, YY = np.meshgrid(xx, xx)
    u = (XX ** 2 + YY ** 2) / w0 ** 2
    # Gaussian amplitude + a real, sizeable spherical aberration, so the
    # residual genuinely spans more than one 2 pi branch.
    env = np.exp(-u) * np.exp(1j * 0.9 * u ** 2)
    base = _wavefront_rms(env, dxo)
    assert base > 0.2, f'fixture is too flat to test wrapping ({base:.4f})'
    vals = [_wavefront_rms(env * np.exp(1j * c), dxo)
            for c in np.linspace(0.0, 2.0 * np.pi, 17)]
    assert max(vals) - min(vals) < 1e-9, (
        f'wavefront rms is piston-DEPENDENT: {min(vals):.6f} to '
        f'{max(vals):.6f} rad over a 2 pi global-phase sweep (base '
        f'{base:.6f})')


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

    Measured 2026-08-11 (rms rad, oracle -> chain).  These are the
    PISTON-INVARIANT values: they were re-measured at v5.34.0 and on the
    absolute-OPL branch and agree to <= 1.3e-16 relative on 8 of the 12 cells
    and <= 3e-08 on 3 more, so the column below is a property of the chain and
    not of whatever global phase the element happens to return.  Four cells
    move against the 2026-07-25 record (singlet w1.2 0.946 -> 0.840 and
    1.018 -> 0.921; triplet 0.028 -> 0.031 and 0.046 -> 0.038): those old
    numbers were read off wrapped ``np.angle`` output without a piston
    reference, so they were contaminated by branch-cut folding -- see
    ``_wavefront_rms`` and docs/audits/FIX_PR29_BLAST_2026_08_11.md S1.

    singlet w0.6 1.2x 0.049 -> 0.025 | 2.5x 0.049 -> 0.054
    singlet w1.2 1.2x 0.795 -> 0.840 | 2.5x 0.795 -> 0.921
    doublet w1.0 1.2x 0.001 -> 0.003 | 2.5x 0.001 -> 0.009
    doublet w2.0 1.2x 0.009 -> 0.016 | 2.5x 0.009 -> 0.041
    triplet w1.6 1.2x 0.032 -> 0.031 | 2.5x 0.032 -> 0.038
    relay   w2.0 1.2x 0.003 -> 0.021 | 2.5x 0.003 -> 0.100
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

    Measured 2026-08-11 with the piston-invariant metric (rms rad, guard
    off -> on): singlet 1.7978 -> 0.9213 (0.512 of the un-guarded value,
    against a 0.7 bar), relay 1.8030 -> 0.1000 (0.056).  The un-guarded values
    sit at pi/sqrt(3) = 1.8138 to within 1 %, which is what a UNIFORMLY RANDOM
    wrapped phase reads -- the property that identifies the cliff."""
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


# ===========================================================================
# 5. SECOND WAVELENGTH.  The chain must be a general workhorse, not a
#    1.31 um / design-121 special case.
# ===========================================================================
# Everything above runs at _WL = 1.31 um.  A propagator that only ever gets
# exercised at one wavelength can hide two whole classes of error, because
# wavelength enters the traced chain through two INDEPENDENT doors:
#
#   * the OPTICS: glass indices are dispersive, so at a different wavelength
#     these are genuinely different systems with different focal lengths and
#     different amounts of real aberration.  The ray oracle re-derives its own
#     truth from the same dispersion data, so if the chain's index lookup, its
#     paraxial R_out bookkeeping, or its ray/eikonal fit were subtly tied to
#     1.31 um, oracle and chain would part company here and nowhere else.
#   * the PROPAGATION: k = 2 pi / lambda sets both the carrier phase and the
#     Nyquist limit of a fixed grid.  At 0.633 um the same dx is ~2.07x coarser
#     in units of lambda, so the sampling margin genuinely tightens.
#
# 0.633 um is chosen deliberately: it is a real HeNe line, it is short enough to
# move the indices measurably, and it sits far outside the design band, so it
# also demonstrates the chain degrades honestly rather than silently.
_WL2 = 0.633e-6

_CELLS_WL2 = [
    ('singlet', _d_singlet, 0.6e-3, 1.2),
    ('doublet', _d_doublet, 1.0e-3, 1.2),
    ('triplet', _d_triplet, 1.6e-3, 1.2),
    ('relay', _d_relay, 2.0e-3, 1.2),
]


def test_second_wavelength_is_actually_a_different_optical_system():
    """Guard the premise: if the glasses were non-dispersive (or the index
    lookup ignored its wavelength argument) the tests below would be a re-run of
    the 1.31 um battery wearing a different label, and would prove nothing."""
    moved = False
    for design in (_d_singlet, _d_doublet, _d_triplet, _d_relay):
        gwg = design(2.4e-3)
        for (p, _gap) in gwg:
            for s in p['surfaces']:
                for g in (s['glass_before'], s['glass_after']):
                    if g in (None, '', 'air', 'AIR'):
                        continue
                    n1, n2 = _n_of(g, _WL), _n_of(g, _WL2)
                    assert n1 > 1.0 and n2 > 1.0
                    if abs(n2 - n1) > 1e-4:
                        moved = True
    assert moved, (
        'no glass index moved between 1.31 um and 0.633 um -- the dispersion '
        'data or the index lookup is ignoring wavelength, so the '
        'second-wavelength battery below is vacuous')


@pytest.mark.parametrize('name,design,w0,ratio', _CELLS_WL2,
                         ids=[f'{c[0]}-w{c[2] * 1e3:g}mm-ap{c[3]:g}x-633nm'
                              for c in _CELLS_WL2])
def test_battery_wavefront_matches_ray_oracle_at_second_wavelength(
        name, design, w0, ratio):
    """The SAME agreement criterion as the 1.31 um battery, at 0.633 um.

    This is the test that says ``apply_real_lens_traced`` is a general
    propagator: the oracle independently re-traces the dispersed system, and the
    chain has to land on it with no wavelength-specific tuning anywhere.

    Tolerance is deliberately the identical ``max(0.15 rad, 0.35 x rms_oracle)``
    used at 1.31 um -- an absolute floor plus a fraction of however much real
    aberration the system has at THIS wavelength.  Note the oracle values are
    much larger here (a 1.31 um design run at 0.633 um is heavily aberrated),
    which is the honest physics, not a failure.
    """
    _skip_if_low_ram()
    ap = ratio * 2.0 * w0
    orc = _oracle_rms(design(ap), 1.5 * w0, w0, wl=_WL2)
    assert orc is not None, 'oracle trace failed at 0.633 um'
    rms = _chain_exit_rms(design, w0, ratio, wl=_WL2)
    tol = max(0.15, 0.35 * orc)
    assert abs(rms - orc) <= tol, (
        f'{name} @633nm w0={w0 * 1e3:g}mm ap={ratio:g}x: chain exit rms '
        f'{rms:.4f} rad vs oracle {orc:.4f} rad (tolerance {tol:.4f})')


def test_shorter_wavelength_reports_more_aberration_on_the_same_glass():
    """Direction-of-effect check that does not depend on either tolerance: the
    same physical singlet, measured in waves, must be WORSE at 0.633 um than at
    1.31 um -- both because the index is higher and because a wave is smaller.
    A chain that normalised this away would be hiding real wavefront error."""
    ap = 1.2 * 2.0 * 0.6e-3
    o_ir = _oracle_rms(_d_singlet(ap), 1.5 * 0.6e-3, 0.6e-3, wl=_WL)
    o_vis = _oracle_rms(_d_singlet(ap), 1.5 * 0.6e-3, 0.6e-3, wl=_WL2)
    assert o_ir is not None and o_vis is not None
    assert o_vis > o_ir, (
        f'oracle: 633nm {o_vis:.4f} rad should exceed 1310nm {o_ir:.4f} rad')
    c_ir = _chain_exit_rms(_d_singlet, 0.6e-3, 1.2, wl=_WL)
    c_vis = _chain_exit_rms(_d_singlet, 0.6e-3, 1.2, wl=_WL2)
    assert c_vis > c_ir, (
        f'chain: 633nm {c_vis:.4f} rad should exceed 1310nm {c_ir:.4f} rad')


# ===========================================================================
# 6. GUARD CALIBRATION across designs AND wavelengths.
# ===========================================================================
# The three gap-guard thresholds (_GAP_SAG_TOL_DEFAULT, _GAP_NA_TOL,
# _GAP_ENV_PHI_TOL_DEFAULT) were originally calibrated on design 121 alone.  A
# threshold calibrated on ONE system is indistinguishable from a threshold
# fitted to that system's quirks.  This pins the other half of the calibration:
# healthy, well-behaved systems must sit comfortably UNDER every threshold, on
# more than one design and in more than one wavelength band -- otherwise the
# guards emit false positives on ordinary optics and users learn to silence
# them, which is worse than not having them.
#
# Measured 2026-08-05 across triplet + relay x {1310 nm, 633 nm} (6 gap legs):
#   gap_na              max 0.1009  vs threshold 0.60  ->   6x headroom
#   gap_env_phi_drop    max 0.0028  vs threshold 0.30  -> 107x headroom
#   gap_env_theta       max 0.0256 rad
# The 633 nm legs show the larger phi_drop, as they must: the frame-dropped
# term is k |z_eff| theta^4 / 8 and k is 2.07x larger there.
@pytest.mark.parametrize('wl,band', [(_WL, '1310nm'), (_WL2, '633nm')])
def test_gap_guard_thresholds_have_headroom_on_healthy_designs(wl, band):
    _skip_if_low_ram()
    seen = 0
    for design, w0 in ((_d_triplet, 1.6e-3), (_d_relay, 2.0e-3)):
        res, _, _, _ = _run_chain(design, w0, 1.2, wl=wl, final_distance=0.0,
                                  on_gap_paraxial='warn', on_gap_frame='warn')
        for s in [st for st in res.stages if 'gap_env_theta' in st]:
            seen += 1
            na = float(s.get('gap_na') or 0.0)
            phi = float(s.get('gap_env_phi_drop') or 0.0)
            th = float(s['gap_env_theta'])
            # the observable must actually have been measured, not defaulted
            assert th > 0.0, (
                f'{design.__name__} @{band}: gap_env_theta is exactly 0 -- the '
                'frame observable was skipped, so this calibration is vacuous')
            assert na < 0.5 * C._GAP_NA_TOL, (
                f'{design.__name__} @{band}: healthy leg gap_na={na:.4f} has '
                f'less than 2x headroom under _GAP_NA_TOL={C._GAP_NA_TOL}')
            assert phi < 0.5 * C._GAP_ENV_PHI_TOL_DEFAULT, (
                f'{design.__name__} @{band}: healthy leg phi_drop={phi:.4f} has '
                f'less than 2x headroom under the frame tolerance')
    assert seen >= 3, f'expected >=3 gap legs to calibrate on, saw {seen}'
