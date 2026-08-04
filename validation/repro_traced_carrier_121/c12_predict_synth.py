# C12 -- the PHYSICS-DERIVED crossover predictor, on the synthetic geometries.
#
# Niche C11 left the concentric-vs-off-centre ray-fit branch ARBITRATED: build
# both candidates at the fit site, score each against the traced OPL over the
# beam, take the smaller.  It works (42/42 against the spline oracle) but it is
# a measurement, not a model: it cannot say WHY the crossover sits at 0.55 w on
# an f/3 singlet and at 0 w on an f/6 one, and it cannot say where it sits at
# any decentre other than the one in front of it.
#
# THE MODEL (derived in docs/audits/C12_PHYSICS_FIT_SELECTION_2026_08_03.md S2).
# The traced OPL is a fixed function of the ENTRANCE position; the beam's
# decentre moves neither it nor the launch grid.  Fit it once on the launch box
# at order Q and split its Chebyshev coefficients at each candidate's own
# order:
#
#     W_>m   = the part of the traced OPL of total degree > m
#
# A least-squares fit of total degree m reproduces the degree-<=m part EXACTLY,
# so each candidate's residual is the residual of fitting ITS OWN spectral tail
# -- identically, not approximately:
#
#     (I - Pi_m) W  ==  (I - Pi_m) W_>m
#
# The tail is decentre-free, so the whole u-dependence of the CONCENTRIC
# candidate is the inflation of its disc,
#
#     R_conc(u) = frbf * sqrt(2 c^2 + w^2) = frbf * w * rho,  rho = sqrt(1+2u^2)
#
# and the OFF-CENTRE candidate's disc and the beam translate together, so its
# residual is flat.  Evaluating the two at any u is then arithmetic on a fixed
# surrogate -- no ray trace, no second decentre -- and the crossover u* is the
# root of E_conc(u) = E_off.  Its closed form follows from the tail's own
# spectral first moment
#
#     m_eff = sum_{n>p} n (S_n sigma^n)^2 / sum_{n>p} (S_n sigma^n)^2
#     rho*  = rho(u) * (E_off / E_conc(u))^(1/m_eff)
#     u*    = sqrt((rho*^2 - 1) / 2)
#
# with NO fitted constant anywhere: S_n is the lens's own measured spectrum,
# sigma = frbf*w/R_box and rho are geometry, and p / P / eps are library
# constants.
#
# SELF-CONTAINED: reuses c11_synth_sweep's inline N-BK7 singlets.  No library
# edit -- the traced OPL is captured with a script-side spy and every candidate
# is scored through the library's OWN ``_decentred_fit_restriction`` /
# ``_decentred_fit_score``.
#
# usage:  GEOMS=f6,f3,f6w python c12_predict_synth.py
import hashlib
import os
import sys
import time
import warnings

import numpy as np

os.environ.setdefault('LUMEN_PIN', '0')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c11_synth_sweep as SS                                    # noqa: E402
import lumenairy as la                                          # noqa: E402
import lumenairy.elements._lens_traced as LT                    # noqa: E402

WL = SS.WL


class _Stop(Exception):
    pass


class OplSpy(object):
    """Capture the launch axes + the UNMASKED traced OPL grid, then abort.

    Run with the OFF-CENTRE branch forced, whose weighted restriction keeps
    every traced sample -- so the third evaluator's ``values`` IS the traced
    map, carrying no NaN mask of either candidate's making.
    """

    def __enter__(self):
        self.seen = []
        self._orig = LT._Cheb2DEvaluator.__init__
        seen = self.seen

        def spy(zelf, xs_in, ys_in, values, order=6, xp=None, weights=None):
            self._orig(zelf, xs_in, ys_in, values, order=order, xp=xp,
                       weights=weights)
            seen.append({'ev': zelf, 'xs': np.asarray(xs_in),
                         'values': np.asarray(values), 'order': int(order),
                         'weights': (None if weights is None
                                     else np.asarray(weights))})
            if len(seen) >= 3:
                raise _Stop()

        LT._Cheb2DEvaluator.__init__ = spy
        return self

    def __exit__(self, *e):
        LT._Cheb2DEvaluator.__init__ = self._orig
        return False


def capture(presc, n, dx, w, sub, frbf, c):
    """One traced call, aborted at the fit site."""
    kw = dict(prescription=presc, wavelength=WL, dx=dx, ray_subsample=sub,
              n_workers=1, fit_radius_beam_factor=frbf, carrier=np.inf,
              beam_centre=(c, 0.0), newton_fit='polynomial', **SS.TKW)
    E = SS.gauss(n, dx, w, c, 0.0)
    old = (LT._DECENTRE_GATE_PIXELS, LT._DECENTRE_GATE_W_FRAC)
    LT._DECENTRE_GATE_PIXELS = 0.0
    LT._DECENTRE_GATE_W_FRAC = 0.0
    try:
        with OplSpy() as sp, warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                la.apply_real_lens_traced(E, **kw)
            except _Stop:
                pass
    finally:
        LT._DECENTRE_GATE_PIXELS, LT._DECENTRE_GATE_W_FRAC = old
    if len(sp.seen) < 3:
        return None
    return {'xs': sp.seen[0]['xs'], 'opl': sp.seen[2]['values'], 'E': E}


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
def spectrum_and_tails(xs, opl, q, orders):
    """ONE order-``q`` box fit of the traced OPL -> its degree-shell spectrum
    ``S_n`` and, for each requested order ``m``, the surrogate ``W_>m``."""
    ev = LT._Cheb2DEvaluator(xs, xs, opl, order=int(q))
    co = np.asarray(ev.coeffs, dtype=np.float64).ravel()
    deg = np.asarray([a + b for a, b in ev._mi], dtype=np.intp)
    S = np.zeros(int(q) + 1)
    for n in range(int(q) + 1):
        sel = deg == n
        if sel.any():
            S[n] = float(np.sqrt(float((co[sel] ** 2).sum())))
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    tails = {}
    for m in orders:
        c2 = co.copy()
        c2[deg <= int(m)] = 0.0
        ev.coeffs = ev.xp.asarray(c2)
        tails[int(m)] = np.asarray(ev.ev(X, Y))
    ev.coeffs = ev.xp.asarray(co)
    return S, tails


def spectral_moment(S, m, sigma):
    """``m_eff``: the tail's own spectral first moment at the beam-disc scale.

    It is the exponent of the inflation law -- ``d log T / d log rho`` at
    ``rho = 1`` -- and it is a property of the SPECTRUM, not of a fit.
    """
    S = np.asarray(S, dtype=np.float64)
    num = den = 0.0
    for n in range(int(m) + 1, S.size):
        e = (S[n] * sigma ** n) ** 2
        num += n * e
        den += e
    return (num / den) if den > 0 else float(m + 2)


def model_scores(xs, tails, w, c, frbf, Lr, r_geom, p, P, u):
    """The two candidates' modelled residuals at decentre ``u``, from the
    surrogate alone -- no trace, no data."""
    cx = u * w
    rho = float(np.sqrt(1.0 + 2.0 * u * u))
    r_off = min(frbf * w, Lr)
    r_conc = min(frbf * w * rho, Lr)
    if r_geom is not None:
        r_conc = min(r_conc, r_geom)
    r2 = xs[:, None] ** 2 + xs[None, :] ** 2
    disc_o = ((xs[:, None] - cx) ** 2 + xs[None, :] ** 2) <= r_off ** 2
    if r_geom is not None:
        both = disc_o & (r2 <= r_geom ** 2)
        if int(both.sum()) >= LT._CARRIER_FIT_MIN_SAMPLES:
            disc_o = both
    disc_c = r2 <= r_conc ** 2
    wgt = np.exp(-2.0 * (((xs[:, None] - cx) ** 2 + xs[None, :] ** 2)
                         / (w * w)))
    wo, oo = LT._decentred_fit_restriction(disc_o, True, p, P)
    wc, oc = LT._decentred_fit_restriction(disc_c, False, p, P)
    e_c = LT._decentred_fit_score(xs, tails[p], wgt, disc_c, wc, oc)
    e_o = LT._decentred_fit_score(xs, tails[P], wgt, disc_o, wo, oo)
    return e_c, e_o, rho, disc_c, disc_o, wgt, wc, oc, wo, oo


def crossover(xs, tails, w, frbf, Lr, r_geom, p, P, umax=4.0):
    """Root of ``E_conc(u) = E_off(u)`` on the model.  ``0.0`` when the
    off-centre candidate already wins at zero decentre."""
    e_c0, e_o0 = model_scores(xs, tails, w, 0.0, frbf, Lr, r_geom,
                              p, P, 0.0)[:2]
    if not (e_c0 < e_o0):
        return 0.0
    lo, hi = 0.0, float(umax)
    ec, eo = model_scores(xs, tails, w, 0.0, frbf, Lr, r_geom, p, P, hi)[:2]
    if ec < eo:
        return float('inf')
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        ec, eo = model_scores(xs, tails, w, 0.0, frbf, Lr, r_geom,
                              p, P, mid)[:2]
        if ec < eo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    geoms = os.environ.get('GEOMS', 'f6,f3,f6w').split(',')
    us = [float(v) for v in os.environ.get(
        'US', '0.02 0.1 0.2 0.4 0.5 0.6 0.75 1.0 1.5').split()]
    Q = int(os.environ.get('Q', '16'))
    p = int(os.environ.get('P0', '6'))
    P = int(LT._DECENTRED_FIT_POLY_ORDER)
    print(f"lumenairy {la.__version__} @ {la.__file__}")
    print(f"  _lens_traced.py "
          f"{hashlib.sha256(open(LT.__file__, 'rb').read()).hexdigest()[:16]}"
          f"  p={p} P={P} eps={LT._FIT_DISC_OUTSIDE_WEIGHT_REL} Q={Q}",
          flush=True)
    for g in geoms:
        presc, n, dx, w0, sub, frbf = SS.GEOMS[g]
        Lr = 0.5 * float(presc['aperture_diameter']) * 1.50
        t0 = time.time()
        # ONE capture, at a single decentre.  The traced map does not depend on
        # it (plane-wave reference, no C6 launch) -- asserted below.
        cap = capture(presc, n, dx, w0, sub, frbf, 0.2 * w0)
        cap2 = capture(presc, n, dx, w0, sub, frbf, 1.0 * w0)
        same = bool(np.array_equal(cap['opl'], cap2['opl']))
        xs, opl = cap['xs'], cap['opl']
        R_box = 0.5 * float(xs.max() - xs.min())
        w = float(LT._input_beam_amp_radius(cap['E'], dx, dx,
                                            centre=(0.2 * w0, 0.0)))
        S, tails = spectrum_and_tails(xs, opl, Q, (p, P))
        sigma = min(frbf * w, Lr) / R_box
        m_eff = spectral_moment(S, p, sigma)
        ustar = crossover(xs, tails, w, frbf, Lr, None, p, P)
        print(f"\n##### {g}  w={w*1e3:.4f}mm  R_box={R_box*1e3:.3f}mm  "
              f"sigma={sigma:.4f}  m_eff={m_eff:.3f}  "
              f"traced map decentre-INVARIANT: {same}", flush=True)
        print(f"      shells S_n: " + ' '.join(f"{v:.2e}" for v in S),
              flush=True)
        print(f"      PREDICTED CROSSOVER  u* = {ustar:.4f}", flush=True)
        hdr = (f"{'|c|/w':>7} {'rho':>6} {'E_c mdl':>10} {'E_o mdl':>10} "
               f"{'mdl pick':>9} | {'E_c meas':>10} {'E_o meas':>10} "
               f"{'arb pick':>9} | {'K_c':>7} {'K_o':>7} "
               f"{'u* closed':>10}")
        print(hdr)
        print('-' * len(hdr), flush=True)
        for u in us:
            e_c, e_o, rho, disc_c, disc_o, wgt, wc, oc, wo, oo = model_scores(
                xs, tails, w, 0.0, frbf, Lr, None, p, P, u)
            capu = capture(presc, n, dx, w0, sub, frbf, u * w0)
            s_c = LT._decentred_fit_score(capu['xs'], capu['opl'], wgt,
                                          disc_c, wc, oc)
            s_o = LT._decentred_fit_score(capu['xs'], capu['opl'], wgt,
                                          disc_o, wo, oo)
            rr = float(np.sqrt(1.0 + 2.0 * u * u))
            uc = (rr * (s_o / max(s_c, 1e-300)) ** (1.0 / m_eff)) ** 2
            uclosed = float(np.sqrt(max(uc - 1.0, 0.0) / 2.0))
            print(f"{u:7.3f} {rho:6.3f} {e_c:10.3e} {e_o:10.3e} "
                  f"{'conc' if e_c <= e_o else 'off':>9} | "
                  f"{s_c:10.3e} {s_o:10.3e} "
                  f"{'conc' if s_c <= s_o else 'off':>9} | "
                  f"{s_c/max(e_c,1e-300):7.3f} {s_o/max(e_o,1e-300):7.3f} "
                  f"{uclosed:10.4f}", flush=True)
        print(f"   [{time.time()-t0:.0f}s]", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
