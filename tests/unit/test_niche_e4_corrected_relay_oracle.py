"""Corrected-relay regression (audit
``docs/audits/AUDIT_WAVEFRONT_AWARE_RAY_LAUNCH_2026_07_23.md`` §4c).

HISTORY / WHY THIS FILE CHANGED
-------------------------------
This file began (2026-07-23) as the "generality gate" for the experimental
``wavefront_aware`` (carrier-relative residual) ray launch (roadmap Part E).
That feature was **reverted** (2026-07-24) — it did not help (and hurt the real
121) — so all ``wavefront_aware`` assertions are gone.  What remains is the
DURABLE, feature-independent finding that the Part E investigation uncovered:

  The traced-carrier chain reproduces a well-posed corrected relay to the
  **diffraction limit** — PROVIDED the group aperture is matched to the beam.
  With an oversized aperture (aperture >> beam) the traced OPL fit is corrupted
  by wildly-aberrated marginal rays and the focus collapses (a sharp cliff at
  ~1.5x the beam diameter on this relay).  The original "chain fails corrected
  relays" reading was an ARTIFACT of an oversized (2.5x-beam) aperture.

P2 UPDATE (2026-07-25) -- MECHANISM PINNED DOWN, AND GUARDED
------------------------------------------------------------
The cliff was localised (audit
``AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24.md`` §4 work): it is carried
ENTIRELY by group 1 (the fast biconvex singlet) -- G2's aperture can be 2.5x its
beam with Strehl 0.963 -- and it is a pure PHASE (OPL-fit) effect, identical
under every ``amplitude_model`` / ``preserve_input_phase`` /
``carrier_reference`` combination.  The exit residual is 1.80 rad rms =
pi/sqrt(3), i.e. a uniformly RANDOM wrapped phase across the whole beam
(core included), not an outer-annulus aberration.

Why group 1: the chain hands its first group ``carrier=r_in=inf``, whose exact
sphere eikonal is non-finite, so the carrier never ENGAGES and the R7
carrier-gated fit-domain restriction (``_CARRIER_FIT_RADIUS_FRAC``) is inactive
there -- the order-6 forward-map / OPL fit then spans the whole launch SQUARE
including its corners, at ``sqrt(2) * 0.75 * aperture`` = 3.7x the beam radius
for a 7 mm aperture.  Sweeping ``_CARRIER_FIT_RADIUS_FRAC`` 0.5 -> 0.21 changes
nothing (Strehl 0.1052 -> 0.1073) precisely because that mask never applies on
this path.

The P2 fix is ``fit_radius_beam_factor`` (chain default 2.0): the same
NaN-masking mechanism, sized to the BEAM and applied on both paths, leaving
``launch_radius`` / the Newton ``bound`` / the out-of-domain threshold alone so
that no field energy is clipped.

Self-contained (no ``.zmx``): a 2-group corrected relay (a fast biconvex singlet
G1 that leaves spherical aberration, + a negative corrector G2 whose radii were
least-squares-tuned against the inline oracle to null it), validated by an
inline exact meridional-raytrace + eikonal oracle, then propagated through
:func:`lumenairy.propagate_traced_carrier_chain`.  Focus quality is scored by
the focus-plane-INDEPENDENT exit-wavefront rms -> Marechal Strehl (no fragile
focus-plane search).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy import carrier_referenced_envelope
from lumenairy.glass import get_glass_index

_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL
_W0 = 2.0e-3            # collimated input 1/e amplitude radius
_GAP = 5e-3


def _skip_if_low_ram(n):
    try:
        import psutil
        if psutil.virtual_memory().available < 2 * (n * n * 16) + (1 << 30):
            pytest.skip(f'insufficient free RAM for the N={n} case')
    except Exception:
        pass


def _n_of(g):
    return 1.0 if g in (None, '', 'air', 'AIR') else float(get_glass_index(g, _WL))


def _singlet(R1, R2, d, glass, ap):
    return {'name': 's', 'aperture_diameter': ap, 'thicknesses': [d], 'surfaces': [
        {'radius': R1, 'glass_before': 'air', 'glass_after': glass, 'conic': 0.0,
         'radius_y': None, 'conic_y': None, 'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
        {'radius': R2, 'glass_before': glass, 'glass_after': 'air', 'conic': 0.0,
         'radius_y': None, 'conic_y': None, 'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]}


def _groups(ap):
    return [{'prescription': _singlet(18e-3, -18e-3, 3e-3, 'N-BK7', ap), 'gap_before': 0.0},
            {'prescription': _singlet(-9.342e-3, 36.508e-3, 3e-3, 'N-BK7', ap), 'gap_before': _GAP}]


# ---------------------------------------------------------------------------
# Inline exact meridional-raytrace + eikonal oracle (design validation only).
# ---------------------------------------------------------------------------
def _system_surfaces(groups_with_gaps):
    surfs = []
    z = 0.0
    for (pres, gap) in groups_with_gaps:
        z += gap
        th = pres['thicknesses']
        for i, s in enumerate(pres['surfaces']):
            R_s = s['radius'] if s['radius'] else np.inf
            surfs.append((z, R_s, _n_of(s['glass_before']), _n_of(s['glass_after'])))
            if i < len(pres['surfaces']) - 1:
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


def _oracle_strehl(groups_with_gaps, x_max, w0, npts=300):
    """Exit-pupil wavefront-error rms (waves) at best focus vs a reference sphere
    on the paraxial image (piston+defocus removed) -> Marechal Strehl.  Pure ray
    trace, grid-independent."""
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
    rms = float(np.sqrt(np.sum(wgt * W ** 2) / np.sum(wgt)))
    return float(np.exp(-(_K0 * rms) ** 2))


# ---------------------------------------------------------------------------
# Chain exit-wavefront Strehl (focus-plane INDEPENDENT).
# ---------------------------------------------------------------------------
_STREHL_CACHE = {}


def _chain_strehl(ap, N=1536, dx=7e-6, guard='default'):
    """Exit-wavefront Strehl of the 2-group relay at group aperture ``ap``.

    ``guard='default'`` uses the chain's own defaults, which since v5.29
    include the P2 aperture:beam cliff guard (``fit_radius_beam_factor=2.0``,
    a BEAM-relative ray-fit domain).  ``guard=None`` restores the pre-v5.29
    aperture-only ray-fit domain (the configuration the cliff was measured
    in).  Cached: a call costs ~8 s and several tests share cells."""
    key = (float(ap), int(N), float(dx), guard)
    if key in _STREHL_CACHE:
        return _STREHL_CACHE[key]
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    env0 = np.exp(-(X ** 2 + Y ** 2) / _W0 ** 2).astype(np.complex128)
    tkw = dict(parallel_amp=False)
    if guard != 'default':
        tkw['fit_radius_beam_factor'] = guard
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = la.propagate_traced_carrier_chain(
            env0, _groups(ap), _WL, dx, r_in=np.inf, ray_subsample=4, n_workers=1,
            traced_kwargs=tkw, final_distance=0.0)
    env = carrier_referenced_envelope(np.asarray(res.field), float(res.R), _WL, float(res.dx))
    dxo = float(res.dx)
    n = env.shape[0]; xx = (np.arange(n) - n / 2) * dxo; XX, YY = np.meshgrid(xx, xx)
    amp = np.abs(env); mask = amp > 0.10 * amp.max(); ph = np.angle(env); w = amp[mask] ** 2
    A = np.stack([np.ones(int(mask.sum())), XX[mask], YY[mask], (XX[mask] ** 2 + YY[mask] ** 2)], 1)
    coef, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], ph[mask] * np.sqrt(w), rcond=None)
    resid = np.angle(np.exp(1j * (ph[mask] - A @ coef)))
    rms = np.sqrt(np.sum(w * resid ** 2) / np.sum(w))
    out = float(np.exp(-rms ** 2))
    _STREHL_CACHE[key] = out
    return out


# ===========================================================================
# 1. The construction really is a corrected relay (independent ray oracle).
# ===========================================================================
def test_e4_oracle_validates_corrected_relay():
    """The inline meridional oracle confirms the design is a genuine corrected
    relay: G1 alone is heavily aberrated; G1+G2 is diffraction-limited.  Pure
    ray trace, grid-independent."""
    # Pupil radius = 1.5x the beam 1/e radius = the beam's actual footprint
    # (matches the amplitude support the chain-Strehl metric samples).
    x_max = 1.5 * _W0
    g1 = _singlet(18e-3, -18e-3, 3e-3, 'N-BK7', 10e-3)
    s_g1 = _oracle_strehl([(g1, 0.0)], x_max, _W0)
    s_relay = _oracle_strehl([(_groups(10e-3)[0]['prescription'], 0.0),
                              (_groups(10e-3)[1]['prescription'], _GAP)], x_max, _W0)
    assert s_g1 is not None and s_relay is not None
    assert s_g1 < 0.5, f"G1 alone must be clearly aberrated (Strehl {s_g1:.3f})"
    assert s_relay > 0.9, f"corrected relay oracle Strehl {s_relay:.3f} < 0.9"
    assert s_relay > 2.0 * s_g1, (s_relay, s_g1)


# ===========================================================================
# 2. At a BEAM-MATCHED aperture the chain reproduces the DL focus.
# ===========================================================================
def test_e4_chain_diffraction_limited_at_matched_aperture():
    """With the group aperture matched to the beam (6 mm = 1.5x beam diameter),
    the plain sphere-only carrier chain reproduces the corrected relay to the
    diffraction limit (exit-wavefront Strehl > 0.9) -- i.e. the chain does NOT
    fail corrected relays.  (Measured ~0.997 at the audit date.)"""
    _skip_if_low_ram(1536)
    st = _chain_strehl(6e-3)
    assert st > 0.9, (
        f"sphere-only chain at a beam-matched aperture must be near-"
        f"diffraction-limited (got Strehl {st:.3f})")


# ===========================================================================
# 3. Oversized aperture corrupts the traced fit (documents the artifact/cliff)
#    -- with the P2 guard DISABLED, i.e. the pre-v5.29 fit domain.
# ===========================================================================
def test_e4_oversized_aperture_corrupts_fit_without_guard():
    """Regression guard for the audit §4c finding: with the pre-v5.29
    aperture-only ray-fit domain (``fit_radius_beam_factor=None``) an oversized
    aperture (10 mm = 2.5x beam diameter) feeds wildly-aberrated marginal rays
    into the traced OPL fit and collapses the focus (Strehl << the
    matched-aperture value) -- the artifact that originally masqueraded as a
    'chain fails corrected relays' result.  Beam is unchanged; only the
    aperture grows.  Measured 2026-07-25: 0.998 (6 mm) -> 0.039 (10 mm)."""
    _skip_if_low_ram(1536)
    st_matched = _chain_strehl(6e-3, guard=None)
    st_oversized = _chain_strehl(10e-3, guard=None)
    assert st_oversized < 0.3, (
        f"oversized aperture should collapse the traced fit (got Strehl "
        f"{st_oversized:.3f})")
    assert st_matched > 3.0 * st_oversized, (st_matched, st_oversized)


# ===========================================================================
# 4. P2 aperture:beam cliff GUARD -- the chain default recovers the cliff.
# ===========================================================================
@pytest.mark.parametrize('ap_mm', [7.0, 8.0, 10.0])
def test_e4_cliff_guard_recovers_oversized_aperture(ap_mm):
    """The chain's default beam-relative ray-fit domain
    (``fit_radius_beam_factor=2.0``, audit
    AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24 §4) recovers the whole cliff:
    every aperture from 1.75x to 2.5x the beam diameter comes back to
    diffraction-limited instead of collapsing.

    Measured 2026-07-25 (Strehl, guard off -> guard on):
    7 mm 0.105 -> 0.9995, 8 mm 0.042 -> 0.9995, 10 mm 0.039 -> 0.9995.

    KNOWN-GOOD ENVELOPE: aperture:beam-diameter ratios 1.0-2.5x on this fast
    (f/4.5) singlet-led relay; the guard's own recovery is flat for
    ``fit_radius_beam_factor`` in 1.5-2.5 (3.0 starts to re-admit marginal
    rays: 0.959 at 10 mm)."""
    _skip_if_low_ram(1536)
    st_guarded = _chain_strehl(ap_mm * 1e-3)
    st_unguarded = _chain_strehl(ap_mm * 1e-3, guard=None)
    assert st_guarded > 0.9, (
        f"the cliff guard must restore a near-diffraction-limited focus at a "
        f"{ap_mm} mm aperture (got Strehl {st_guarded:.4f})")
    assert st_guarded > 5.0 * st_unguarded, (st_guarded, st_unguarded)


def test_e4_cliff_guard_leaves_precliff_apertures_alone():
    """The guard must not disturb the PRE-cliff (aperture <= 1.5x beam
    diameter) results it was not needed for.  Measured 2026-07-25: 4 mm
    0.9986 -> 0.9986, 5 mm 0.9997 -> 0.9996, 6 mm 0.9978 -> 0.9993 (the 6 mm
    cell improves slightly -- it already sat on the cliff's shoulder)."""
    _skip_if_low_ram(1536)
    for ap_mm in (4.0, 5.0, 6.0):
        off = _chain_strehl(ap_mm * 1e-3, guard=None)
        on = _chain_strehl(ap_mm * 1e-3)
        assert on > 0.99, (ap_mm, on)
        assert on > off - 0.005, (
            f"guard degraded the pre-cliff {ap_mm} mm aperture: "
            f"{off:.4f} -> {on:.4f}")


def test_e4_cliff_guard_conserves_energy():
    """The guard restricts the ray-FIT domain only -- it must NOT vignette
    (the failure mode of clamping the aperture, audit
    AUDIT_WAVEFRONT_AWARE_RAY_LAUNCH_2026_07_23 §4c.3, where a 3x-beam
    aperture clamp cost the design-121 22 EE6 points of real halo power).
    Exit power with and without the guard must agree to <0.1%."""
    _skip_if_low_ram(1536)
    N, dx = 1536, 7e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    env0 = np.exp(-(X ** 2 + Y ** 2) / _W0 ** 2).astype(np.complex128)
    pw = {}
    for guard in ('default', None):
        tkw = dict(parallel_amp=False)
        if guard is None:
            tkw['fit_radius_beam_factor'] = None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = la.propagate_traced_carrier_chain(
                env0, _groups(10e-3), _WL, dx, r_in=np.inf, ray_subsample=4,
                n_workers=1, traced_kwargs=tkw, final_distance=0.0)
        f = np.asarray(res.field)
        pw[guard] = float((np.abs(f) ** 2).sum()) * float(res.dx) ** 2
    rel = abs(pw['default'] - pw[None]) / pw[None]
    assert rel < 1e-3, f"guard changed transmitted power by {rel:.2e}"


def test_e4_aperture_beam_warning_fires_and_is_silenceable():
    """The warn-only half of the guard: with the beam-relative fit domain OFF
    and an aperture well above 1.5x the beam diameter, a RuntimeWarning names
    the ratio and the cliff; ``on_aperture_beam='silent'`` suppresses it, and
    an engaged ``fit_radius_beam_factor`` suppresses it too (the regime is
    guarded, so there is nothing to warn about)."""
    N, dx = 512, 20e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X ** 2 + Y ** 2) / _W0 ** 2).astype(np.complex128)
    presc = _singlet(18e-3, -18e-3, 3e-3, 'N-BK7', 9e-3)
    common = dict(prescription=presc, wavelength=_WL, dx=dx, ray_subsample=4,
                  n_workers=1, parallel_amp=False, on_undersample='silent',
                  on_noncollimated='off')
    with pytest.warns(RuntimeWarning, match='beam 1/e\\^2 diameter'):
        la.apply_real_lens_traced(E, fit_radius_beam_factor=None, **common)
    for kw in (dict(on_aperture_beam='silent'),
               dict(fit_radius_beam_factor=2.0)):
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            la.apply_real_lens_traced(E, **dict(common, **kw))
        assert not [w for w in wl if 'beam 1/e^2 diameter' in str(w.message)], (
            f'cliff warning should be suppressed by {kw}')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
