# C11 -- THE PHYSICAL DECENTRE GATE, on SYNTHETIC designs.
#
# The niche-C1 gate (``_DECENTRE_GATE_PIXELS`` / ``_DECENTRE_GATE_W_FRAC``)
# chooses, per element call, between
#
#   CONCENTRIC   the historical path: fit disc about the GRID ORIGIN with
#                radius ``frbf * sqrt(2 c^2 + w^2)``, hard NaN sample mask,
#                fit order ``newton_poly_order``;
#   OFF-CENTRE   the D1/D7 path: fit disc about the BEAM with radius
#                ``frbf * w``, weighted restriction, fit order
#                ``_DECENTRED_FIT_POLY_ORDER``.
#
# 0.05 w was chosen only to kill a discontinuity at NULL decentre, not from
# physics.  This measures WHERE the two branches actually cross over, on
# geometries that share nothing with design 121, against a reference that
# shares nothing with either branch: ``newton_fit='spline'``, a LOCAL bicubic
# of the traced map which skips the polynomial fit and its disc restriction
# entirely (the disc block is gated on ``newton_fit != 'spline'``).
#
# SELF-CONTAINED: synthetic N-BK7 singlets + the niche-D6 ``K = -n^2`` conic
# built inline.  No .zmx, no design asset, no library edit -- both branches are
# forced through the module attributes every call site resolves, inside
# try/finally.
#
# usage:  GEOMS=f6,f3,f6w,d6 python c11_synth_sweep.py
#         GEOMS=f6 US='0 0.05 0.2 0.5 0.8' python c11_synth_sweep.py
import os
import sys
import time
import warnings

import numpy as np

os.environ.setdefault('LUMEN_PIN', '0')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lumenairy as la                                          # noqa: E402
import lumenairy.elements._lens_traced as LT                    # noqa: E402

WL = 1.31e-6
TKW = dict(on_undersample='silent', on_noncollimated='silent',
           on_aperture_beam='silent')


# ---------------------------------------------------------------------------
# geometries (inline; nothing loaded from disk)
# ---------------------------------------------------------------------------
def _singlet(R1, R2, d, glass, ap, name):
    surf = [{'radius': R1, 'glass_before': 'air', 'glass_after': glass,
             'conic': 0.0, 'radius_y': None, 'conic_y': None,
             'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
            {'radius': R2, 'glass_before': glass, 'glass_after': 'air',
             'conic': 0.0, 'radius_y': None, 'conic_y': None,
             'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]
    return {'name': name, 'aperture_diameter': ap, 'surfaces': surf,
            'thicknesses': [d]}


def _conic_singlet(f=3.00e-3):
    """niche D6's stand-in: flat entrance + ``K = -n^2`` conic exit, the exact
    Fermat singlet for a collimated bundle (stigmatic at EVERY decentre, so its
    truth is decentre-invariant)."""
    n = float(la.get_glass_index('N-BK7', WL))
    ap, th = 3.40e-3, 1.5e-3

    def s(radius, gb, ga, conic):
        return {'radius': float(radius), 'glass_before': gb,
                'glass_after': ga, 'conic': float(conic), 'radius_y': None,
                'conic_y': None, 'aspheric_coeffs': None,
                'aspheric_coeffs_y': None}

    return {'name': 'd6 conic', 'aperture_diameter': ap,
            'thicknesses': [th],
            'surfaces': [s(np.inf, 'air', 'N-BK7', 0.0),
                         s(-(n - 1.0) * f, 'N-BK7', 'air', -n * n)]}


#: label -> (prescription, N, dx, w, ray_subsample, frbf)
GEOMS = {
    # C1 item 1's own f/6 fixture (launch radius 7.5 mm, beam 1.0 mm)
    'f6': (_singlet(60e-3, -60e-3, 3.0e-3, 'N-BK7', 10e-3, 'c11-f6'),
           512, 30e-6, 1.0e-3, 8, 2.0),
    # C1 item 1's FAST fixture -- the geometry where the pre-C1 null step was
    # largest (8.3e-6 of peak)
    'f3': (_singlet(30e-3, -30e-3, 3.0e-3, 'N-BK7', 10e-3, 'c11-f3'),
           512, 30e-6, 1.0e-3, 8, 2.0),
    # the geometry the C1 note's OWN crossover sweep used: same f/6 singlet at
    # w = 1.4 mm, so the fit disc (2.8 mm) sits inside a 5.0 mm semi-aperture
    # and |c| + r really does reach new territory
    'f6w': (_singlet(60e-3, -60e-3, 3.0e-3, 'N-BK7', 10e-3, 'c11-f6w'),
            512, 30e-6, 1.4e-3, 8, 2.0),
    # niche D6's conic (fast, exit NA 0.20, aperture:beam 2.83)
    'd6': (_conic_singlet(), 1024, 6.0e-3 / 1024, 0.60e-3, 8, 2.0),
    # ... the same conic on a grid whose Nyquist direction cosine
    # (lambda / 2 dx = 0.224) clears its 0.20 exit NA, so the spline oracle
    # can converge at all (niche C2's caveat: it returns an ALL-ZERO field
    # when it cannot, and with on_undersample='silent' says nothing)
    'd6f': (_conic_singlet(), 2048, 6.0e-3 / 2048, 0.60e-3, 8, 2.0),
    # ... and a SLOWER conic (f = 6 mm -> exit NA 0.10) so the ASPHERIC
    # generality claim does not rest on a single NA
    'd6s': (_conic_singlet(6.0e-3), 1024, 6.0e-3 / 1024, 0.60e-3, 8, 2.0),
}


def gauss(n, dx, w, cx=0.0, cy=0.0):
    x = (np.arange(n) - n // 2) * dx
    return np.exp(-(((x[None, :] - cx) ** 2 + (x[:, None] - cy) ** 2)
                    / w ** 2)).astype(np.complex128)


class Force(object):
    """Force the branch through the library's own documented selectors."""

    def __init__(self, arm, gate=None):
        self.arm, self.gate = arm, gate

    def __enter__(self):
        self.old = (LT._DECENTRE_GATE_PIXELS, LT._DECENTRE_GATE_W_FRAC)
        if self.arm == 'conc':
            LT._DECENTRE_GATE_W_FRAC = float('inf')
        elif self.arm == 'off':
            LT._DECENTRE_GATE_PIXELS = 0.0
            LT._DECENTRE_GATE_W_FRAC = 0.0
        elif self.arm == 'gate':
            LT._DECENTRE_GATE_W_FRAC = float(self.gate)
        elif self.arm != 'ship':
            raise ValueError(self.arm)
        return self

    def __exit__(self, *e):
        LT._DECENTRE_GATE_PIXELS, LT._DECENTRE_GATE_W_FRAC = self.old
        return False


class FitSpy(object):
    """Capture every ``_Cheb2DEvaluator`` the call builds: the launch axes, the
    traced values, the order and the weights.  Used for the DIRECT
    DISCRIMINATOR arm (compare the two candidate fits' beam-weighted residuals
    without ever leaving the decision site)."""

    def __enter__(self):
        self.seen = []
        self._orig = LT._Cheb2DEvaluator.__init__
        seen = self.seen

        def spy(zelf, xs_in, ys_in, values, order=6, xp=None, weights=None):
            self._orig(zelf, xs_in, ys_in, values, order=order, xp=xp,
                       weights=weights)
            seen.append({'ev': zelf, 'xs': np.asarray(xs_in),
                         'ys': np.asarray(ys_in),
                         'values': np.asarray(values), 'order': int(order),
                         'weights': (None if weights is None
                                     else np.asarray(weights))})

        LT._Cheb2DEvaluator.__init__ = spy
        return self

    def __exit__(self, *e):
        LT._Cheb2DEvaluator.__init__ = self._orig
        return False


#: arm -> (branch to force, extra element kwargs).  ``off6`` / ``conc10``
#: SEPARATE the two things the branch changes together -- the fit DOMAIN (disc
#: centre + weighted vs hard-mask restriction) and the fit ORDER -- which S5 of
#: D121_RESIDUAL_CLOSURE names as unseparated.
ARM_KW = {
    'conc': ('conc', {}),
    'off': ('off', {}),
    'off6': ('off', {'decentred_fit_poly_order': 6}),
    'conc10': ('conc', {'newton_poly_order': 10}),
}


def run(presc, n, dx, w, sub, frbf, c, arm, gate=None, fit='polynomial',
        spy=False):
    branch, extra = ARM_KW.get(arm, (arm, {}))
    kw = dict(prescription=presc, wavelength=WL, dx=dx, ray_subsample=sub,
              n_workers=1, fit_radius_beam_factor=frbf, carrier=np.inf,
              beam_centre=(c, 0.0), newton_fit=fit, **TKW)
    kw.update(extra)
    E = gauss(n, dx, w, c, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if spy:
            with Force(branch, gate), FitSpy() as sp:
                out = np.asarray(la.apply_real_lens_traced(E, **kw))
            return out, sp.seen
        with Force(branch, gate):
            return np.asarray(la.apply_real_lens_traced(E, **kw)), None


# ---------------------------------------------------------------------------
# scoring: unwrap-free, piston + tilt removed, power weighted
# ---------------------------------------------------------------------------
def score(E, Eref, X, Y, thresh=0.02, n_iter=40):
    """Equivalent rms wavefront error of ``E`` against ``Eref`` in WAVES, with
    piston and tilt removed WITHOUT any phase unwrapping (a fixed point of the
    weighted-phasor alignment), plus the amplitude error, the sup-norm C1 used,
    and a fold/ghost detector.

    Piston is irrelevant (a global phase) and tilt is a lateral shift of the
    spot the readout centroids out, so both are projected away -- what is left
    is what actually costs image quality.
    """
    a = np.abs(Eref)
    pk = float(a.max())
    if not (pk > 0):
        return None
    keep = a > thresh * pk
    if not keep.any():
        return None
    ww = (a[keep] ** 2)
    psi = np.angle(E[keep] * np.conj(Eref[keep]))
    Xk, Yk = X[keep], Y[keep]
    A = np.stack([Xk, Yk], axis=1)
    Aw = A * ww[:, None]
    G = Aw.T @ A
    p = q = 0.0
    e = np.zeros_like(psi)
    for _ in range(n_iter):
        r = psi - p * Xk - q * Yk
        c0 = np.angle((ww * np.exp(1j * r)).sum())
        e = np.angle(np.exp(1j * (r - c0)))
        try:
            d = np.linalg.solve(G, Aw.T @ e)
        except np.linalg.LinAlgError:
            break
        p += float(d[0])
        q += float(d[1])
        if abs(float(d[0])) * np.ptp(Xk) + abs(float(d[1])) * np.ptp(Yk) < 1e-9:
            break
    sig = float(np.sqrt((ww * e ** 2).sum() / ww.sum()))
    # sampling adequacy: the power-weighted p99.9 wrapped nearest-neighbour
    # step of the RESIDUAL must be far below pi (a bare max saturates at pi on
    # a handful of skirt pixels)
    R = np.zeros_like(a)
    R[keep] = e
    K = keep
    dxs, wts = [], []
    for sl0, sl1 in (((slice(None), slice(1, None)),
                      (slice(None), slice(None, -1))),
                     ((slice(1, None), slice(None)),
                      (slice(None, -1), slice(None)))):
        kk = K[sl0] & K[sl1]
        if kk.any():
            dxs.append(np.abs(np.angle(np.exp(1j * (R[sl0] - R[sl1])))[kk]))
            wts.append(np.minimum(a[sl0] ** 2, a[sl1] ** 2)[kk])
    if dxs:
        v = np.concatenate(dxs)
        wv = np.concatenate(wts)
        o = np.argsort(v)
        cw = np.cumsum(wv[o])
        cw /= max(float(cw[-1]), 1e-300)
        p999 = float(v[o][np.searchsorted(cw, 0.999)])
    else:
        p999 = 0.0
    amp = float(np.sqrt((ww * ((np.abs(E[keep]) - a[keep]) / pk) ** 2).sum()
                        / ww.sum()))
    dE = float(np.abs(E - Eref).max()) / pk
    dark = a < 1e-4 * pk
    ghost = (float(np.abs(E[dark]).max()) / pk) if dark.any() else 0.0
    return {'sig_waves': sig / (2 * np.pi), 'strehl': float(np.exp(-sig ** 2)),
            'amp_rms': amp, 'maxdE': dE, 'ghost': ghost, 'p999': p999,
            'n_keep': int(keep.sum())}


# ---------------------------------------------------------------------------
# the decision-site quantities and the candidate predictors
# ---------------------------------------------------------------------------
def decision_site(presc, n, dx, w_nom, frbf, c, E=None):
    """Everything the library has IN HAND at the branch decision, computed the
    way the library computes it."""
    if E is None:
        E = gauss(n, dx, w_nom, c, 0.0)
    ap = presc.get('aperture_diameter')
    Lr = (0.5 * float(ap) * 1.50) if ap is not None else 0.5 * n * dx
    w_beam = float(LT._input_beam_amp_radius(E, dx, dx, centre=(c, 0.0)))
    w_orig = float(LT._input_beam_amp_radius(E, dx, dx, centre=None))
    r_beam = min(frbf * w_beam, Lr)          # off-centre disc radius
    r_conc = min(frbf * w_orig, Lr)          # concentric disc radius
    # the carrier-gated R7 disc: only live when an ENGAGED carrier puts the
    # call on ``_r7_carrier_path`` (a plane-wave ``carrier=np.inf`` does not),
    # so it is reported but not folded into the radii here
    r_geom = LT._CARRIER_FIT_RADIUS_FRAC * Lr
    # the disc actually used, each way (the library's ``_fit_r_max``)
    R_conc = r_conc
    R_off = r_beam
    return {'c': c, 'u': (c / w_beam if w_beam > 0 else 0.0), 'w': w_beam,
            'w_orig': w_orig, 'Lr': Lr, 'r_geom': r_geom,
            'r_beam': r_beam, 'r_conc': r_conc, 'R_conc': R_conc,
            'R_off': R_off,
            # P1: DISC INFLATION.  the concentric disc is sized from the
            # ORIGIN-referenced second moment, which reads sqrt(2c^2+w^2); the
            # ratio to the radius the caller ASKED for is the inflation
            'infl': (r_conc / (frbf * w_beam)) if w_beam > 0 else np.inf,
            # P2: REACH.  how far each fit domain runs into the aperture
            'reach_conc': R_conc, 'reach_off': c + r_beam,
            'reach_ratio': (R_conc / (c + r_beam)) if (c + r_beam) > 0 else 1.0}


def discriminate(presc, n, dx, w, sub, frbf, c):
    """THE DIRECT DISCRIMINATOR (approach D of the brief).

    At the decision site the rays are ALREADY traced, so both candidate fits
    can be built from the SAME samples and compared before either is used.
    This builds them (via the two branches the library already implements),
    evaluates each on the coarse launch lattice, and scores it against the
    traced samples themselves with the BEAM's own intensity as the weight --
    i.e. "which polynomial reproduces the traced map where the light is".

    Returns the per-branch weighted rms of the forward map (metres) and of the
    OPL (waves), plus the discriminator's pick.
    """
    ec, sc = run(presc, n, dx, w, sub, frbf, c, 'conc', spy=True)
    eo, so = run(presc, n, dx, w, sub, frbf, c, 'off', spy=True)
    if len(sc) < 3 or len(so) < 3:
        return None
    sc, so = sc[-3:], so[-3:]                     # x_out, y_out, opl
    xs, ys = so[0]['xs'], so[0]['ys']
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    # the OFF-CENTRE arm keeps every traced sample (the weighted restriction),
    # so ITS ``values`` are the traced map itself -- the common truth both
    # fits are scored against
    wgt = np.exp(-2.0 * (((X - c) ** 2 + Y ** 2) / (w * w)))
    out = {}
    for name, S in (('conc', sc), ('off', so)):
        r_xy, r_opl = 0.0, 0.0
        for k, rec in enumerate(S):
            truth = so[k]['values']
            pred = np.asarray(rec['ev'].ev(X, Y))
            ok = np.isfinite(truth) & np.isfinite(pred)
            ww = np.where(ok, wgt, 0.0)
            tot = float(ww.sum())
            if tot <= 0:
                continue
            r = np.where(ok, pred - truth, 0.0)
            v = float(np.sqrt((ww * r * r).sum() / tot))
            if k < 2:
                r_xy = float(np.hypot(r_xy, v))
            else:
                r_opl = v
        out[name] = {'xy_rms': r_xy, 'opl_waves': r_opl / WL,
                     'order': S[0]['order'],
                     'weighted': S[0]['weights'] is not None}
    out['pick'] = ('conc' if out['conc']['opl_waves'] < out['off']['opl_waves']
                   else 'off')
    out['pick_xy'] = ('conc' if out['conc']['xy_rms'] < out['off']['xy_rms']
                      else 'off')
    return out


def main():
    geoms = os.environ.get('GEOMS', 'f6,f3,f6w,d6').split(',')
    arms = os.environ.get('ARMS', 'conc,off').split(',')
    do_d = os.environ.get('DISCRIM', '1') not in ('0', '')
    us = [float(v) for v in os.environ.get(
        'US', '0 0.02 0.05 0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.75 1.0 1.25 1.5'
    ).split()]
    import hashlib
    print(f"lumenairy {la.__version__} @ {la.__file__}")
    print(f"  _lens_traced.py "
          f"{hashlib.sha256(open(LT.__file__, 'rb').read()).hexdigest()[:16]}"
          f"   gate ships at pixels={LT._DECENTRE_GATE_PIXELS} "
          f"w_frac={LT._DECENTRE_GATE_W_FRAC}", flush=True)
    for g in geoms:
        presc, n, dx, w, sub, frbf = GEOMS[g]
        print(f"\n########## {g}: {presc['name']}  N={n} dx={dx*1e6:.3f} um "
              f"w={w*1e3:.3f} mm  aperture={presc['aperture_diameter']*1e3:.3f}"
              f" mm  rs={sub} frbf={frbf} ##########", flush=True)
        hdr = (f"{'|c|/w':>7} {'infl':>6} {'reachR':>7} "
               + ''.join(f"{'sig ' + a:>10}" for a in arms)
               + f" {'FLOOR':>10}"
               + ''.join(f"{'dE ' + a:>10}" for a in arms)
               + f" {'ghost_c':>9} {'p999':>6} {'WINNER':>7}"
               + ('' if not do_d else
                  f" | {'D:opl_c':>9} {'D:opl_o':>9} {'PICK':>5} {'':>4}"))
        print(hdr)
        print('-' * len(hdr), flush=True)
        for u in us:
            t0 = time.time()
            c = u * w
            ds = decision_site(presc, n, dx, w, frbf, c)
            ref, _ = run(presc, n, dx, w, sub, frbf, c, 'conc', fit='spline')
            ref2, _ = run(presc, n, dx, w, sub, frbf, c, 'off', fit='spline')
            if not np.isfinite(ref).all() or float(np.abs(ref).max()) <= 0:
                print(f"{u:7.3f}   SPLINE ORACLE UNUSABLE (all-zero / "
                      f"non-finite) -- skipped", flush=True)
                continue
            ax = (np.arange(n) - n / 2) * dx
            X, Y = np.meshgrid(ax, ax)
            res = {}
            for a in arms:
                res[a] = score(run(presc, n, dx, w, sub, frbf, c, a)[0],
                               ref, X, Y)
            fl = score(ref2, ref, X, Y)
            sc, so = res['conc'], res['off']
            win = ('tie' if abs(sc['sig_waves'] - so['sig_waves'])
                   <= fl['sig_waves'] else
                   ('conc' if sc['sig_waves'] < so['sig_waves'] else 'off'))
            dsc = discriminate(presc, n, dx, w, sub, frbf, c) if do_d else None
            print(f"{ds['u']:7.3f} {ds['infl']:6.3f} {ds['reach_ratio']:7.3f} "
                  + ''.join(f"{res[a]['sig_waves']:10.3e}" for a in arms)
                  + f" {fl['sig_waves']:10.3e}"
                  + ''.join(f"{res[a]['maxdE']:10.2e}" for a in arms)
                  + f" {sc['ghost']:9.2e} "
                  f"{max(res[a]['p999'] for a in arms):6.3f} {win:>7}"
                  + ('' if dsc is None else
                     f" | {dsc['conc']['opl_waves']:9.3e} "
                     f"{dsc['off']['opl_waves']:9.3e} {dsc['pick']:>5} "
                     f"{'OK' if dsc['pick'] == win or win == 'tie' else 'MISS'}")
                  + f"   [{time.time()-t0:.0f}s]", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
