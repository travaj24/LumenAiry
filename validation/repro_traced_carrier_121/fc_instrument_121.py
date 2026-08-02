# D121 FINAL CLOSURE -- the INSTRUMENT, and its own error band.
#
# ``hybrid_localize_121.py`` scores the chain against the exact-ray oracle by
# running BOTH through ``rs_spot``.  But the two arms reach that common readout
# by different routes, and this study's first job is to price the routes rather
# than the physics.  Every knob below moves the MEASUREMENT with the FIELD held
# fixed; anything that moves EE3 is instrument, not chain.
#
# Both arms call ``hybrid_localize_121.rs_spot`` VERBATIM (imported, not
# copied), so the readout is bit-identical between them and between every row
# here and the campaign's existing tables.
#
# Knobs (env):
#   ARM     'oracle' (launch at the DOE plane) | 'chain' (launch at the chain's
#           group-5 exit) | 'chainA' (chain, but with the readout's own launch
#           built from an EXACT-sphere split instead of the parabola one)
#   CARRY   oracle only: 1 carries the DOE-plane residual phase of ``env_doe``
#           (the chain always does; the oracle arm of hybrid_localize does NOT)
#   STRIP   chain only: 1 feeds the chain |env_doe| (residual phase removed),
#           i.e. the SAME field the CARRY=0 oracle launches
#   UP      chain only: band-limited Fourier upsample of the exit envelope
#   TAPER   chain only: 'on' (shipped) | 'off' (T==1) | a float r_safe scale
#   NL, CLIP, NOUT, DXO, BACK, ORD, RN, RS   as in hybrid_localize_121
#
# Diagnostics.  The integrand-step statistic ``hybrid_localize`` prints is
# computed from a 2-D-UNWRAPPED launch phase, which carries spurious 2*pi
# jumps (DIAG_LAST_GROUP_DECENTRE artefact 7).  The RS kernel only ever sees
# ``exp(i*ph0)``, so those jumps cannot touch the RESULT -- but they do
# dominate the DIAGNOSTIC.  This script prints the WRAPPED-difference version
# alongside, which is the honest one.
#
# usage:  ARM=chain UP=2 python fc_instrument_121.py
import hashlib
import os
import sys
import time
import warnings

import numpy as np

# ``approx_common`` (imported transitively by ``approx_ablate_121``) DEFAULTS
# ``LUMEN_PIN`` to a frozen v5.31 export in the 2026-07-31 study's scratchpad,
# and that directory still exists on this box: importing it first silently
# measures v5.31.  This study measures the WORKING TREE, so the pin is forced
# off before any such import and the provenance is printed per run.
os.environ.setdefault('LUMEN_PIN', '0')

import _d121_common as C                                       # noqa: E402
import approx_ablate_121 as AB                                 # noqa: E402
import hybrid_localize_121 as H                                # noqa: E402
import lumenairy                                               # noqa: E402
import lumenairy.elements._lens_traced as _LT                  # noqa: E402
import lumenairy.propagators.carrier as _CM                    # noqa: E402
from approx_common import Patch                                # noqa: E402
from exact_ray_oracle_121 import _bilinear, _phase_gradient    # noqa: E402
from lumenairy.propagators.carrier import (                    # noqa: E402
    _envelope_amp_radius, carrier_referenced_envelope)
from lumenairy.raytrace import RayBundle, trace                # noqa: E402

LAM = C.LAM
K0 = 2 * np.pi / LAM
_HERE = os.path.dirname(os.path.abspath(__file__))


def _provenance():
    """Print the library actually imported, with the sha256 of the two files
    every number in this study depends on."""
    out = [f"lumenairy {lumenairy.__version__} @ {lumenairy.__file__}"]
    for mod in (_LT, _CM):
        h = hashlib.sha256(open(mod.__file__, 'rb').read()).hexdigest()[:16]
        out.append(f"  {os.path.basename(mod.__file__)} {h}")
    return '\n'.join(out)


def _free_surfaces(dist, back):
    return H._free_surfaces(dist, back)


def _wrapped_step(x0, y0, amp, ph0, p, q, surfs, back, xci, yci, nl,
                  n_out, dx_out):
    """HONEST integrand-step statistic: the same quantity ``rs_spot`` reports,
    but differenced with a WRAPPED difference so a 2*pi jump in the launch
    phase (which the RS kernel cannot see, since it only consumes
    ``exp(i*ph0)``) cannot inflate it.

    Returns ``(p50, p999, max)`` in cycles, amplitude-weighted exactly as
    ``rs_spot`` weights."""
    n0 = np.sqrt(np.maximum(1.0 - p * p - q * q, 0.0))
    rb = RayBundle(x=x0.copy(), y=y0.copy(), z=np.zeros_like(x0),
                   L=p.copy(), M=q.copy(), N=n0.copy(), wavelength=LAM,
                   alive=np.ones(x0.size, bool), opd=np.zeros_like(x0))
    ir = trace(rb, surfs, LAM, output_filter='last').image_rays
    alive = np.asarray(ir.alive, bool)
    xe = np.asarray(ir.x)
    ye = np.asarray(ir.y)
    opd = np.asarray(ir.opd)
    Ne = np.asarray(ir.N)
    XE = xe.reshape(nl, nl)
    YE = ye.reshape(nl, nl)
    hx = float(x0.reshape(nl, nl)[0, 1] - x0.reshape(nl, nl)[0, 0])
    hy = float(y0.reshape(nl, nl)[1, 0] - y0.reshape(nl, nl)[0, 0])
    J = np.abs(np.gradient(XE, hx, axis=1) * np.gradient(YE, hy, axis=0)
               - np.gradient(XE, hy, axis=0) * np.gradient(YE, hx, axis=1)
               ).ravel()
    good = alive & (amp > 0) & np.isfinite(opd) & (J > 0) & (Ne > 0)
    W = np.zeros(x0.size)
    W[good] = (amp[good] * np.sqrt(n0[good] * J[good] / Ne[good])
               * abs(hx * hy))
    ax = (np.arange(n_out) - (n_out - 1) / 2.0) * dx_out
    px, py = xci + ax[0], yci + ax[0]
    phi = np.full(x0.size, np.nan)
    phi[good] = (K0 * opd[good] + ph0[good]
                 + K0 * np.sqrt((px - xe[good]) ** 2 + (py - ye[good]) ** 2
                                + back ** 2))
    PH = phi.reshape(nl, nl)
    WG = W.reshape(nl, nl)
    wm = max(W.max(), 1e-300)

    def _w(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    st = np.concatenate([
        (np.abs(_w(np.diff(PH, axis=0))) * np.minimum(WG[1:], WG[:-1])).ravel(),
        (np.abs(_w(np.diff(PH, axis=1)))
         * np.minimum(WG[:, 1:], WG[:, :-1])).ravel()]) / wm
    st = st[np.isfinite(st)]
    if not st.size:
        return np.nan, np.nan, np.nan
    return (float(np.percentile(st, 50)) / (2 * np.pi),
            float(np.percentile(st, 99.9)) / (2 * np.pi),
            float(st.max()) / (2 * np.pi))


def oracle_launch(env_doe, R_doe, dx_doe, L, M, nl, clip, carry, post, back):
    """The n=0 launch of ``hybrid_localize_121``, with the DOE-plane residual
    phase optionally carried (``carry``).  The shipped n=0 arm does NOT carry
    it: it launches |env_doe| on a PERFECT sphere of radius ``R_doe``."""
    w = _envelope_amp_radius(env_doe, dx_doe, dx_doe)
    h = 2 * clip * w / (nl - 1)
    t = (np.arange(nl) - (nl - 1) / 2.0) * h
    X, Y = np.meshgrid(t, t)
    x0, y0 = X.ravel(), Y.ravel()
    amp = _bilinear(np.abs(env_doe), dx_doe, x0, y0)
    den = np.sqrt(R_doe ** 2 + x0 ** 2 + y0 ** 2)
    p = x0 / den * np.sign(R_doe) + L
    q = y0 / den * np.sign(R_doe) + M
    ph0 = K0 * (L * x0 + M * y0 + np.sign(R_doe) * (den - abs(R_doe)))
    step_ph = 0.0
    if carry:
        gx, gy, step_ph = _phase_gradient(env_doe, dx_doe)
        p = p + _bilinear(gx, dx_doe, x0, y0) / K0
        q = q + _bilinear(gy, dx_doe, x0, y0) / K0
        if carry == 1:
            # UNWRAP-FREE (this study).  ``exact_ray_oracle_121.oracle_spot``'s
            # ``carry_phase`` arm bilinear-interpolates a GLOBAL 2-D unwrap of
            # ``env_doe``.  95 % of that grid is dark skirt whose phase is
            # numerical noise, and ``np.unwrap`` along a row propagates that
            # noise across the beam -- the artefact DIAG_LAST_GROUP_DECENTRE
            # section 8 item 2 killed in its own instruments.  Measured here:
            # the unwrapped arm reads EE3 9.9-52.0 with NO convergence in the
            # launch density, against 90.05 (converged) for this branch.
            # The RS kernel only ever consumes ``exp(i*ph0)``, so interpolating
            # the UNIT PHASOR and taking its argument is exact for a residual
            # this smooth (its own per-pixel step is 0.0049 rad).
            u = env_doe / np.maximum(np.abs(env_doe), 1e-300)
            ph0 = ph0 + np.angle(_bilinear(u, dx_doe, x0, y0))
        else:
            ph = np.unwrap(np.unwrap(np.angle(env_doe), axis=1), axis=0)
            ph0 = ph0 + _bilinear(ph, dx_doe, x0, y0)
    return (x0, y0, amp, ph0, p, q, C.post_surfaces(post, back_off=back),
            nl, h, w, step_ph)


def chain_launch(res, L, M, nl0, clip, up, post, back, split='parabola',
                 ngroups=None):
    """The n=6 launch of ``hybrid_localize_121``, verbatim, plus ``split``:

    'parabola' -- the shipped route.  The analytic part of the launch phase is
        the PARABOLA r^2/(2R) + the tilt ramp; everything else (including the
        sphere-minus-parabola quartic) is in the finite-differenced residual.
    'sphere'   -- the SAME field, split against the EXACT sphere instead, so
        the quartic is analytic and the finite-differenced part is smaller.
        A pure re-partition: ``ph0`` is the same total phase either way, so any
        difference is the finite-difference LAUNCH DIRECTION, i.e. instrument.
    'exact'    -- split against the FULL niche-C5 tilted eikonal, the exact
        wavefront of the congruence the chain is carrying:
        ``W = sign(R)(sqrt((u + R L/N)^2 + (v + R M/N)^2 + R^2) - |R|/N)``.
        This also removes the C5 exactness term ``D`` (the coma/astigmatism the
        'sphere' split leaves in the finite-differenced residual: measured
        0.15-0.19 rad/px at design 121's 46-51 mrad orders).  Reduces to
        'sphere' EXACTLY at L = M = 0 -- asserted as a null below.
    """
    fld, Rk, dxk = np.asarray(res.field), float(res.R), float(res.dx)
    st = [s for s in res.stages if not s.get('target')][-1]
    Lk, Mk = st.get('L_out', L), st.get('M_out', M)
    xck, yck = st.get('x_c_out', 0.0), st.get('y_c_out', 0.0)
    envk = carrier_referenced_envelope(fld, Rk, LAM, dxk)
    nn = envk.shape[0]
    u = (np.arange(nn) - nn / 2) * dxk
    envk = envk * np.exp(-1j * K0 * (Lk * u[None, :] + Mk * u[:, None]))
    if split in ('sphere', 'exact'):
        # move the (reference - parabola) part OUT of the finite-differenced
        # residual and into the analytic part.  The tilt ramp is already off
        # (line above), so 'sphere' adds back (S - parabola) and 'exact' adds
        # back the whole C5 eikonal minus (parabola + ramp).
        r2 = u[None, :] ** 2 + u[:, None] ** 2
        if split == 'sphere':
            Wref = np.sign(Rk) * (np.sqrt(r2 + Rk * Rk) - abs(Rk))
        else:
            npar = np.sqrt(max(1.0 - Lk * Lk - Mk * Mk, 0.0))
            uu = u[None, :] + Rk * Lk / npar
            vv = u[:, None] + Rk * Mk / npar
            Wref = np.sign(Rk) * (np.sqrt(uu * uu + vv * vv + Rk * Rk)
                                  - abs(Rk) / npar)
            Wref = Wref - (Lk * u[None, :] + Mk * u[:, None])
        envk = envk * np.exp(-1j * K0 * (Wref - r2 / (2.0 * Rk)))
    envk = H._fourier_up(envk, up)
    dxk_u = dxk / up
    nn = envk.shape[0]
    wk = _envelope_amp_radius(envk, dxk_u, dxk_u)
    half = int(np.ceil(clip * wk / dxk_u))
    c = nn // 2
    i0 = max(c - half, 0)
    i1 = min(c + half + 1, nn)
    envk = envk[i0:i1, i0:i1]
    nn = envk.shape[0]
    u = (np.arange(nn) - (c - i0)) * dxk_u
    gx, gy, step_ph = _phase_gradient(envk, dxk_u)
    phres = np.unwrap(np.unwrap(np.angle(envk), axis=1), axis=0)
    sd = max(1, int(np.floor(nn / nl0)))
    envk = envk[::sd, ::sd]
    gx, gy, phres = gx[::sd, ::sd], gy[::sd, ::sd], phres[::sd, ::sd]
    u = u[::sd]
    nn = envk.shape[0]
    U, V = np.meshgrid(u, u)
    x0 = (U + xck).ravel()
    y0 = (V + yck).ravel()
    amp = np.abs(envk).ravel()
    if split == 'sphere':
        den = np.sqrt(Rk * Rk + U ** 2 + V ** 2)
        p = (U / den * np.sign(Rk)).ravel() + Lk + gx.ravel() / K0
        q = (V / den * np.sign(Rk)).ravel() + Mk + gy.ravel() / K0
        ph0 = (K0 * (np.sign(Rk) * (den - abs(Rk))).ravel()
               + K0 * (Lk * U.ravel() + Mk * V.ravel()) + phres.ravel())
    elif split == 'exact':
        # W and its EXACT gradient (the congruence's own direction cosines):
        # dW/du = sign(R) * (u + R L/N) / sqrt(.), which -> L as u -> 0.
        npar = np.sqrt(max(1.0 - Lk * Lk - Mk * Mk, 0.0))
        uu = U + Rk * Lk / npar
        vv = V + Rk * Mk / npar
        den = np.sqrt(uu * uu + vv * vv + Rk * Rk)
        p = (np.sign(Rk) * uu / den).ravel() + gx.ravel() / K0
        q = (np.sign(Rk) * vv / den).ravel() + gy.ravel() / K0
        ph0 = (K0 * (np.sign(Rk) * (den - abs(Rk) / npar)).ravel()
               + phres.ravel())
    else:
        p = (U / Rk).ravel() + Lk + gx.ravel() / K0
        q = (V / Rk).ravel() + Mk + gy.ravel() / K0
        ph0 = K0 * ((U ** 2 + V ** 2).ravel() / (2 * Rk)
                    + Lk * U.ravel() + Mk * V.ravel()) + phres.ravel()
    ng = len(post) if ngroups is None else int(ngroups)
    # the hybrid localisation: the chain did ``ng`` groups, the exact-ray +
    # Rayleigh-Sommerfeld oracle finishes the rest (``hybrid_localize_121``'s
    # ``n_chain`` semantics, so the step between ng and ng+1 is the cost of
    # putting that group through the chain).
    surfs = (_free_surfaces(C.TRAILING, back) if ng >= len(post)
             else C.post_surfaces(post[ng:], back_off=back))
    return (x0, y0, amp, ph0, p, q, surfs, nn, dxk_u * sd, wk, step_ph)


def _taper_patch(spec):
    """``TAPER`` -> the patch list for ``Patch``.

    NOTE (2026-08-02).  Once niche C9 landed, the LIBRARY DEFAULT became the
    exact (untapered) conversion, so ``TAPER='on'`` -- "leave the library
    alone" -- silently stopped meaning "the taper".  Every arm is therefore
    pinned through the C9 FLAG here rather than through the default:
    ``'on'`` forces ``SPHERE_PARAB_CONVERSION_EXACT = False`` (the historical
    cos^2 taper, i.e. v5.32.0) and ``'off'`` forces it True.  A float still
    scales ``r_safe`` through the monkeypatch, which needs the taper present.

    (Caught by ``fc_sampling_121.py``, whose taper-on and taper-off rows came
    back byte-identical after the library default flipped.  Every number taken
    BEFORE the flip used the monkeypatch and is unaffected -- the two routes
    are checked against each other in the audit's S7.3.)
    """
    if spec in ('on', '1', ''):
        return [(_CM, 'SPHERE_PARAB_CONVERSION_EXACT', False)]
    if spec in ('off', '0'):
        return [(_CM, 'SPHERE_PARAB_CONVERSION_EXACT', True)]
    return (_taper_scale_patch(float(spec))
            + [(_CM, 'SPHERE_PARAB_CONVERSION_EXACT', False)])


def _taper_scale_patch(f):
    """``_sphere_parab_conversion`` with ``r_safe`` scaled by ``f`` and NOTHING
    else changed -- ``approx_post_c6.p_sphere_taper_scale`` VERBATIM, copied
    here rather than imported so this script does not pull in that module's
    heavy import-time setup.  ``f = 1`` must reproduce the shipped taper
    exactly, which is asserted by running it alongside ``TAPER='on'``."""
    def q(shape, dx, wavelength, R, sign, w_beam=None, centre=(0.0, 0.0)):
        if not np.isfinite(R) or R == 0.0:
            return None
        n, ny = int(shape[-1]), int(shape[-2])
        x = (np.arange(n, dtype=np.float64) - n / 2) * dx - float(centre[0])
        y = (np.arange(ny, dtype=np.float64) - ny / 2) * dx - float(centre[1])
        r2 = x[None, :] ** 2 + y[:, None] ** 2
        k = 2.0 * np.pi / wavelength
        diff = _CM._exact_sphere_eikonal((ny, n), dx, dx, wavelength, R,
                                         centre=centre) - r2 / (2.0 * R)
        r_safe = f * (abs(R) ** 3 * wavelength / dx) ** (1.0 / 3.0)
        t = np.clip((np.sqrt(r2) - 0.75 * r_safe) / (0.25 * r_safe), 0.0, 1.0)
        return np.exp(sign * 1j * k * diff * np.cos(0.5 * np.pi * t) ** 2)
    return [(_CM, '_sphere_parab_conversion', q)]


def run_chain(post, env_doe, R_doe, dx_doe, L, M, rs, taper, strip,
              ngroups=None):
    e = np.abs(env_doe).astype(np.complex128) if strip else env_doe
    grp = post if ngroups is None else post[:int(ngroups)]
    with Patch(_taper_patch(taper)):
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            res = C.la.propagate_traced_carrier_chain(
                e, [dict(g) for g in grp], LAM, dx_doe,
                r_in=C.la.TiltedCarrier(R_doe, L, M),
                ray_subsample=rs, n_workers=8, final_distance=0.0,
                final_leg='paraxial', on_decentred_fit='ignore')
    seen = {}
    for w in wl:
        t = str(w.message)[:110]
        seen[t] = seen.get(t, 0) + 1
    return res, seen


def main():
    arm = os.environ.get('ARM', 'chain')
    m, n = (int(v) for v in os.environ.get('ORD', '0,0').split(','))
    rn = int(os.environ.get('RN', '1024'))
    rs = int(os.environ.get('RS', '4'))
    up = int(os.environ.get('UP', '1'))
    nl0 = int(os.environ.get('NL', '9999'))
    clip = float(os.environ.get('CLIP', '3.0'))
    nout = int(os.environ.get('NOUT', '61'))
    dxo = float(os.environ.get('DXO', '0.4')) * 1e-6
    back = float(os.environ.get('BACK', '5.0')) * 1e-3
    carry = int(os.environ.get('CARRY', '0'))
    strip = int(os.environ.get('STRIP', '0'))
    taper = os.environ.get('TAPER', 'on')
    split = os.environ.get('SPLIT', 'parabola')
    ng = os.environ.get('NG')
    ng = None if ng in (None, '') else int(ng)
    tag = os.environ.get('TAG', '')
    print(_provenance())

    _pre, post, _g, period = C.geometry()
    env_doe, R_doe, dx_doe, _P = C.chain_a(n=rn, rs=rs)
    L = m * LAM / period
    M = n * LAM / period
    ch = trace(RayBundle(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
                         L=np.array([L]), M=np.array([M]),
                         N=np.array([np.sqrt(1 - L * L - M * M)]),
                         wavelength=LAM, alive=np.ones(1, bool),
                         opd=np.zeros(1)),
               C.post_surfaces(post), LAM, output_filter='last').image_rays
    xci, yci = float(ch.x[0]), float(ch.y[0])

    t0 = time.time()
    warns = {}
    if arm == 'oracle':
        nl0 = int(os.environ.get('NL', '161'))
        (x0, y0, amp, ph0, p, q, surfs, nl, h, w,
         step_ph) = oracle_launch(env_doe, R_doe, dx_doe, L, M, nl0, clip,
                                  carry, post, back)
    else:
        res, warns = run_chain(post, env_doe, R_doe, dx_doe, L, M, rs,
                               taper, strip, ng)
        (x0, y0, amp, ph0, p, q, surfs, nl, h, w,
         step_ph) = chain_launch(res, L, M, nl0, clip, up, post, back, split,
                                 ng)
    r = H.rs_spot(x0, y0, amp, ph0, p, q, surfs, back, xci, yci,
                  dx_out=dxo, n_out=nout, nl=nl)
    wp50, wp999, wmx = _wrapped_step(x0, y0, amp, ph0, p, q, surfs, back,
                                     xci, yci, nl, nout, dxo)
    dig = hashlib.sha256(np.ascontiguousarray(r['I']).tobytes()
                         ).hexdigest()[:16]
    for t, k in warns.items():
        print(f"     [warn x{k}] {t}")
    print(f"ARM={arm:6s} ORD=({m:+d},{n:+d}) TAG={tag:12s} "
          f"UP={up} NL={nl0} CLIP={clip} NOUT={nout} DXO={dxo * 1e6} "
          f"BACK={back * 1e3} CARRY={carry} STRIP={strip} TAPER={taper} "
          f"SPLIT={split} NG={ng if ng is not None else 'all'}")
    print(f"   grid {nl}^2 @ {h * 1e6:.3f} um  w {w * 1e6:.1f} um  "
          f"{r['n_rays']} rays  live {r['live'] * 100:.4f} %  "
          f"[{time.time() - t0:.0f}s]")
    print(f"   step: envelope max {step_ph:.4f} rad | integrand UNWRAPPED "
          f"p50/p99.9/max {r['step_md']:.4f}/{r['step_w']:.4f}/"
          f"{r['step_mx']:.3g} | WRAPPED (honest) "
          f"{wp50:.4f}/{wp999:.4f}/{wmx:.4f} cycles")
    print(f"   FWHM {r['fwhm'] * 1e6:6.3f} um   EE3 {r['ee3'] * 100:7.4f}   "
          f"EE6 {r['ee6'] * 100:7.4f}   EE12 {r['ee12'] * 100:7.4f}   "
          f"sha {dig}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
