# Conservation audit (2026-07-31): SAMPLING ADEQUACY, stated properly.
#
# ``energy_hull_121.py``'s first pass reported the element input's
# amplitude-weighted nn-step p99.9 at 3.13-3.14 rad -- essentially pi -- which
# would invalidate every wave statement in this audit.  Before that is
# reported as a finding it has to be shown not to be a property of the METRIC.
# Two ways it could be one:
#   (a) the weighted percentile is reached in the numerically-zero surround,
#       where the phase is random but the weight is not exactly 0;
#   (b) the step is taken across the whole grid rather than the bright core.
# Both are tested here by restricting to an amplitude contour and reporting the
# median / p99 / p99.9 / max of the SAME statistic on the same field.
#
# The distinction matters for the audit's conclusions: a POWER measurement
# (P/Pin, halo shells, second moment of a returned array) does not care about
# phase sampling at all, while anything that propagates the field does.
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _d121_common as C                                          # noqa: E402
import probe_c6_element as E6                                     # noqa: E402
import probe_ghost_c6 as G6                                       # noqa: E402
import wfe_probe_common as P                                      # noqa: E402
from lumenairy.elements._lens_traced import (TiltedCarrier,       # noqa: E402
                                             _input_beam_amp_radius)


def steps(E, sel):
    """Wrapped nearest-neighbour phase steps of ``E`` and their power weights,
    restricted to pairs where BOTH pixels satisfy ``sel``."""
    ph = np.angle(E)
    w = np.abs(E) ** 2
    dx_ = np.abs(np.angle(np.exp(1j * (ph[:, 1:] - ph[:, :-1]))))
    dy_ = np.abs(np.angle(np.exp(1j * (ph[1:, :] - ph[:-1, :]))))
    kx = sel[:, 1:] & sel[:, :-1]
    ky = sel[1:, :] & sel[:-1, :]
    v = np.concatenate([dx_[kx], dy_[ky]])
    ww = np.concatenate([np.minimum(w[:, 1:], w[:, :-1])[kx],
                         np.minimum(w[1:, :], w[:-1, :])[ky]])
    return v, ww


def wq(v, ww, q):
    if v.size == 0:
        return float('nan')
    o = np.argsort(v)
    cw = np.cumsum(ww[o])
    if cw[-1] <= 0:
        return float('nan')
    return float(v[o][np.searchsorted(cw / cw[-1], q)])


def main():
    orders = [tuple(int(v) for v in o.split(','))
              for o in os.environ.get('ORDERS', '0,0 -4,-2').split()]
    _pre, post, _g, _p = C.geometry()
    presc = post[-1]['prescription']
    surfs = P.element_surfaces(presc)
    print("Amplitude-weighted wrapped nearest-neighbour phase step [rad], "
          "limit pi = 3.1416.")
    print("Rows: the element INPUT (co-moving envelope) and the element "
          "OUTPUT, per amplitude contour.")
    print()
    hdr = (f"{'order':>8} {'field':>7} {'contour':>12} {'npix':>9} "
           f"{'p50':>8} {'p99':>8} {'p99.9':>8} {'max':>8} {'pow frac':>10}")
    print(hdr)
    print('-' * len(hdr))
    for (m, n) in orders:
        E_in, _Eo, carv, dx = E6.get_call(m, n)
        car = TiltedCarrier(*carv)
        F, _d, _f, _t = None, None, None, None
        import energy_guard_probe as GP
        F, _d, _f, _t = GP.element(E_in, presc, dx, car, 4, True)
        for tag, E in (('E_in', E_in), ('E_out', np.asarray(F))):
            a = np.abs(E)
            tot = float((a ** 2).sum())
            for lbl, thr in (('all', 0.0), ('>1e-6 pk', 1e-6),
                             ('>0.05 pk', 0.05), ('>0.37 pk', np.exp(-1.0))):
                sel = a > thr * a.max()
                v, ww = steps(E, sel)
                pf = float((a[sel] ** 2).sum()) / tot if tot > 0 else 0.0
                print(f"{f'({m},{n})':>8} {tag:>7} {lbl:>12} {int(sel.sum()):>9} "
                      f"{wq(v, ww, 0.50):>8.4f} {wq(v, ww, 0.99):>8.4f} "
                      f"{wq(v, ww, 0.999):>8.4f} "
                      f"{(float(v.max()) if v.size else float('nan')):>8.4f} "
                      f"{pf:>10.6f}")
        print()
    print("Note: a POWER measurement (P/Pin, halo shells, second moment of a "
          "RETURNED array) is unaffected by phase sampling.")
    print("      These numbers bound what may be said about PROPAGATING these "
          "fields, not about their energy content.")


if __name__ == '__main__':
    sys.exit(main())
