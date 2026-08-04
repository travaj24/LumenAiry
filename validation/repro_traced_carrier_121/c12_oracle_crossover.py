# C12 -- the THREE-DESIGN prediction table: predictor vs arbiter vs oracle.
#
# For each synthetic geometry this bisects, on the SAME fixture and in the same
# process, three crossovers:
#
#   ORACLE     where the exit field itself stops preferring the concentric
#              branch, scored against ``newton_fit='spline'`` -- a LOCAL
#              bicubic of the traced map that skips the polynomial fit and its
#              disc restriction entirely, so it is independent of both
#              candidates.  This is the truth the whole question is about;
#   ARBITER    niche C11: where the two candidates' measured beam-weighted OPL
#              residuals cross;
#   PREDICTOR  niche C12: the closed-form ``u*`` the library itself computes
#              from the lens's own spectral tail and the disc-inflation law,
#              read out of ``_decentred_fit_crossover`` in flight.
#
# The predictor's ``u*`` is a NUMBER the library produces once per call; the
# other two have to be bisected by re-running the whole element, which is why
# only the predictor can be quoted per design.
#
# LOCAL-ONLY; no library edit.
#
# usage:  GEOMS=f6,f3,f6w python c12_oracle_crossover.py
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


class Read(object):
    """Read the predictor's own ``u*`` and the arbiter's own two scores out of
    a live call, without changing either."""

    def __enter__(self):
        self.ustar = []
        self.scores = []
        self._x = LT._decentred_fit_crossover
        self._s = LT._decentred_fit_score
        ustar, scores = self.ustar, self.scores

        def xspy(u, ec, eo, m):
            v = self._x(u, ec, eo, m)
            ustar.append((float(u), float(ec), float(eo), float(m), float(v)))
            return v

        def sspy(xs, opl, wgt, disc, wts, order):
            v = self._s(xs, opl, wgt, disc, wts, order)
            scores.append(float(v))
            return v

        LT._decentred_fit_crossover = xspy
        LT._decentred_fit_score = sspy
        return self

    def __exit__(self, *e):
        LT._decentred_fit_crossover = self._x
        LT._decentred_fit_score = self._s
        return False


def call(presc, n, dx, w, sub, frbf, c, fit='polynomial', **flags):
    kw = dict(prescription=presc, wavelength=SS.WL, dx=dx, ray_subsample=sub,
              n_workers=1, fit_radius_beam_factor=frbf, carrier=np.inf,
              beam_centre=(c, 0.0), newton_fit=fit, **SS.TKW)
    E = SS.gauss(n, dx, w, c, 0.0)
    old = {k: getattr(LT, k) for k in flags}
    for k, v in flags.items():
        setattr(LT, k, v)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with Read() as r:
                out = np.asarray(la.apply_real_lens_traced(E, **kw))
        return out, r
    finally:
        for k, v in old.items():
            setattr(LT, k, v)


def bisect(f, lo, hi, n=14):
    """Smallest ``u`` at which ``f(u)`` stops being True (``f`` True below)."""
    if not f(lo):
        return 0.0
    if f(hi):
        return float('inf')
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if f(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    geoms = os.environ.get('GEOMS', 'f6,f3,f6w').split(',')
    nb = int(os.environ.get('NB', '12'))
    print(f"lumenairy {la.__version__}")
    print(f"  _lens_traced.py "
          f"{hashlib.sha256(open(LT.__file__, 'rb').read()).hexdigest()[:16]}",
          flush=True)
    hdr = (f"{'geometry':>10} {'ORACLE u*':>20} {'ARBITER u*':>11} "
           f"{'PREDICTOR u*':>13} {'m_eff':>7} {'resolved':>9}")
    print()
    print(hdr)
    print('-' * len(hdr), flush=True)
    for g in geoms:
        presc, n, dx, w, sub, frbf = SS.GEOMS[g]
        t0 = time.time()
        ax = (np.arange(n) - n / 2) * dx
        X, Y = np.meshgrid(ax, ax)

        def oracle_conc(u):
            c = u * w
            ref, _ = SS.run(presc, n, dx, w, sub, frbf, c, 'conc',
                            fit='spline')
            if not np.isfinite(ref).all() or float(np.abs(ref).max()) <= 0:
                return False
            sc = SS.score(SS.run(presc, n, dx, w, sub, frbf, c, 'conc')[0],
                          ref, X, Y)
            so = SS.score(SS.run(presc, n, dx, w, sub, frbf, c, 'off')[0],
                          ref, X, Y)
            return bool(sc['sig_waves'] < so['sig_waves'])

        def arb_conc(u):
            _o, r = call(presc, n, dx, w, sub, frbf, u * w,
                         DECENTRED_FIT_ARBITER=True)
            if len(r.scores) < 2:
                return False
            return bool(r.scores[1] <= r.scores[0])

        state = {}

        def pred_conc(u):
            _o, r = call(presc, n, dx, w, sub, frbf, u * w,
                         DECENTRED_FIT_PREDICTOR=True)
            if not r.ustar:
                return False
            uu, ec, _eo, m, us = r.ustar[-1]
            state['m'] = m
            state['res'] = ('yes' if (len(r.scores) >= 2
                                      and ec != r.scores[1]) else 'no')
            state['ustar'] = us
            return bool(uu <= us)

        # ``lo`` sits ABOVE niche C1's null gate (0.05 w): below it neither the
        # arbiter nor the predictor runs at all, and a bisection started there
        # would report "off-centre" for a call that never made a choice.
        u_or = bisect(oracle_conc, 1e-4, 1.5, nb)
        u_ar = bisect(arb_conc, 0.06, 1.5, nb)
        u_pr = bisect(pred_conc, 0.06, 1.5, nb)
        print(f"{g:>10} {u_or:>20.4f} {u_ar:>11.4f} {u_pr:>13.4f} "
              f"{state.get('m', float('nan')):>7.3f} "
              f"{state.get('res', '?'):>9}   [{time.time()-t0:.0f}s]",
              flush=True)
    print("\nORACLE u* is bisected on the EXIT FIELD against the "
          "fit-domain-free spline\nreference; 0.0000 means the off-centre "
          "branch already wins at the first\nrepresentable decentre.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
