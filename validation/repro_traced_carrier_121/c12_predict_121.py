# C12 -- the physics predictor on DESIGN 121's own element calls.
#
# Replays the arbiter inputs captured by ``c12_arb_trace_121.py`` (in process,
# with the flag ON, so they are the SHIPPED decision's own arguments) and
# evaluates the C12 spectral model on them:
#
#   * the traced OPL's degree-shell spectrum ``S_n`` over the launch box, from
#     ONE order-Q fit;
#   * each candidate's residual re-derived from its OWN spectral tail
#     ``W_>m`` -- ``K = measured / modelled`` is the model's error, and it
#     carries no fitted constant;
#   * the tail's spectral first moment ``m_eff``, which is the exponent of the
#     disc-inflation law;
#   * the closed-form crossover
#         rho* = rho(u) (E_off / E_conc)^(1/m_eff),  u* = sqrt((rho*^2-1)/2).
#
# The per-group ``u*`` is then checked against the BRACKET the shipped
# arbiter's own verdicts give: a group that reads CONCENTRIC at ``u`` and
# OFF-CENTRE at ``u'`` > ``u`` brackets its crossover in ``(u, u')``.
#
# usage:  python c12_predict_121.py
import os
import sys

import numpy as np

os.environ.setdefault('LUMEN_PIN', '0')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c12_predict_synth as PS                                  # noqa: E402
import lumenairy.elements._lens_traced as LT                    # noqa: E402

#: capture tag -> (order label, index of the FIRST group that is arbitrated).
#: The chain has six post-DOE groups; the ones below niche C1's null gate never
#: reach the arbiter, so the k-th captured pair is group ``k + first``.
TAGS = [('m1_0', '(-1,0)', 2), ('m2_0', '(-2,0)', 1), ('m3_0', '(-3,0)', 1),
        ('m4_0', '(-4,0)', 0), ('m4_m2', '(-4,-2)', 0)]


def load(tag):
    fn = f"_c12_arb_{tag}.npz"
    if not os.path.exists(fn):
        return None
    z = np.load(fn)
    n = int(z['n'])
    return [{'xs': z[f"xs{k}"], 'opl': z[f"opl{k}"], 'wgt': z[f"wgt{k}"],
             'disc': z[f"disc{k}"],
             'weights': (z[f"wts{k}"] if bool(z[f"hasw{k}"]) else None),
             'order': int(z[f"ord{k}"]), 'score': float(z[f"sc{k}"])}
            for k in range(n)]


def beam_from_weight(xs, g):
    """Recover the chief-ray offset and the beam radius from the shipped
    scorer's own weight array ``g = exp(-2 |r - c|^2 / w^2)``."""
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    tot = float(g.sum())
    cx = float((g * X).sum() / tot)
    cy = float((g * Y).sum() / tot)
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    return cx, cy, float(np.sqrt(2.0 * float((g * r2).sum()) / tot))


def main():
    Q = int(os.environ.get('Q', '14'))
    p = int(os.environ.get('P0', '6'))
    P = int(LT._DECENTRED_FIT_POLY_ORDER)
    print(f"  p={p} P={P} Q={Q}")
    hdr = (f"{'order':>8} {'grp':>4} {'|c|/w':>7} {'m_eff':>6} "
           f"{'E_c mdl':>10} {'E_c meas':>10} {'K_c':>6} "
           f"{'E_o mdl':>10} {'E_o meas':>10} {'K_o':>6} "
           f"{'PICK':>5} {'u* mdl':>7} {'u* meas':>8}")
    print()
    print(hdr)
    print('-' * len(hdr), flush=True)
    ustars = {}
    for tag, lab, first in TAGS:
        rec = load(tag)
        if rec is None:
            print(f"  (no capture for {tag})")
            continue
        for k in range(0, len(rec) - 1, 2):
            o_, c_ = rec[k], rec[k + 1]
            xs = o_['xs']
            cx, cy, w = beam_from_weight(xs, o_['wgt'])
            u = float(np.hypot(cx, cy)) / w
            rho = float(np.sqrt(1.0 + 2.0 * u * u))
            R_box = 0.5 * float(xs.max() - xs.min())
            S, tails = PS.spectrum_and_tails(xs, o_['opl'], Q,
                                             (c_['order'], o_['order']))
            # sigma: the beam disc in box units, read off the OFF-CENTRE disc
            r_off = float(np.sqrt(float(o_['disc'].sum())
                                  / np.pi)) * float(xs[1] - xs[0])
            sigma = r_off / R_box
            m_eff = PS.spectral_moment(S, c_['order'], sigma)
            e_c = LT._decentred_fit_score(xs, tails[c_['order']], o_['wgt'],
                                          c_['disc'], c_['weights'],
                                          c_['order'])
            e_o = LT._decentred_fit_score(xs, tails[o_['order']], o_['wgt'],
                                          o_['disc'], o_['weights'],
                                          o_['order'])
            s_c, s_o = c_['score'], o_['score']

            def _us(ec, eo):
                if not (ec > 0 and np.isfinite(ec) and np.isfinite(eo)):
                    return float('nan')
                rr = rho * (eo / ec) ** (1.0 / m_eff)
                return float(np.sqrt(max(rr * rr - 1.0, 0.0) / 2.0))

            grp = k // 2 + first
            ustars.setdefault(grp, []).append((lab, u, _us(s_c, s_o),
                                               'conc' if s_c <= s_o else 'off'))
            print(f"{lab:>8} {grp:>4} {u:>7.4f} {m_eff:>6.2f} "
                  f"{e_c:>10.3e} {s_c:>10.3e} {s_c/max(e_c,1e-300):>6.2f} "
                  f"{e_o:>10.3e} {s_o:>10.3e} {s_o/max(e_o,1e-300):>6.2f} "
                  f"{'conc' if s_c <= s_o else 'off':>5} "
                  f"{_us(e_c, e_o):>7.4f} {_us(s_c, s_o):>8.4f}", flush=True)
    print("\nPER-GROUP CROSSOVER: closed form vs the bracket the shipped "
          "arbiter's\nown verdicts give (last CONC decentre, first OFF one).",
          flush=True)
    print(f"{'grp':>4} {'u* closed (per order)':>44} {'mean':>8} "
          f"{'bracket lo':>11} {'bracket hi':>11}")
    for grp in sorted(ustars):
        rows = sorted(ustars[grp], key=lambda r: r[1])
        vals = [r[2] for r in rows if np.isfinite(r[2])]
        lo = max([r[1] for r in rows if r[3] == 'conc'], default=0.0)
        hi = min([r[1] for r in rows if r[3] == 'off'], default=float('inf'))
        print(f"{grp:>4} " + ' '.join(f"{v:6.3f}" for v in vals).rjust(44)
              + f" {np.mean(vals):8.3f} {lo:11.4f} {hi:11.4f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
