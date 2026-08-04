# C12 -- the per-GROUP branch pattern, priced at the chain level.
#
# The C11 arbiter's verdict is a per-group decision, but every chain-level
# number the campaign has is per ORDER: it can say "the arbiter costs 0.026
# points at (-1,0)" and not WHICH group's pick costs it.  This runs the chain
# with the branch FORCED per arbitrated group, so the attribution is direct.
#
# A pattern is one character per arbitrated element call, in call order:
#
#     'c'  force CONCENTRIC     'o'  force OFF-CENTRE     '-'  the arbiter's
#                                                              own verdict
#
# Forcing is done at the DECISION, not at the gate: the two candidate scores
# are returned as 0 / 1 in the wanted direction, so both candidates are still
# BUILT exactly as shipped and everything upstream of the fit site -- including
# niche C6's residual-eikonal domain, which is committed before the trace --
# is untouched.  That is what separates this from ``rc_gate_121.py``'s forced
# arms, which move four things at once.
#
# Same instrument, readout and scoring as ``rc_gate_121.py`` /
# ``c11_gate_arms_121.py``, so the numbers are directly comparable.
#
# SCRIPT-SIDE ONLY -- no library file is edited by this runner.
#
# usage:  ORDERS='-1,0' PATTERNS='C11=-,allc=cccc,allo=oooo,g4o=ccoc' \
#             python c12_group_arms_121.py
import hashlib
import os
import sys
import time

import numpy as np

os.environ.setdefault('LUMEN_PIN', '0')

import _d121_common as C                                       # noqa: E402
import fc_instrument_121 as FI                                 # noqa: E402
import hybrid_localize_121 as H                                # noqa: E402
import rc_readout_121 as RD                                    # noqa: E402
import lumenairy.elements._lens_traced as _LT                  # noqa: E402
from lumenairy.raytrace import RayBundle, trace                # noqa: E402

LAM = C.LAM
BACK = 5.0e-3


class Force(object):
    """Force the arbiter's verdict per arbitrated call.

    ``pattern`` is one character per call; ``'-'`` (or running off the end of
    the pattern) leaves the library's own scores alone.  ``None`` disables the
    arbiter entirely (the v5.32 pure-gate selector).
    """

    def __init__(self, pattern):
        self.pattern = pattern

    def __enter__(self):
        self._old_flag = _LT.DECENTRED_FIT_ARBITER
        self._orig = _LT._decentred_fit_score
        self.calls = []
        if self.pattern is None:
            _LT.DECENTRED_FIT_ARBITER = False
            return self
        _LT.DECENTRED_FIT_ARBITER = True
        pat = self.pattern
        calls = self.calls
        state = {'n': 0}

        def spy(xs_in, opl_grid, weight, disc, weights, order):
            v = self._orig(xs_in, opl_grid, weight, disc, weights, order)
            k = state['n']
            state['n'] = k + 1
            grp, half = k // 2, k % 2       # half 0 = off-centre, 1 = conc
            ch = pat[grp] if grp < len(pat) else '-'
            calls.append((grp, half, ch, float(v)))
            if ch == 'c':                   # concentric must win the <= test
                return 1.0 if half == 0 else 0.0
            if ch == 'o':
                return 0.0 if half == 0 else 1.0
            return v

        _LT._decentred_fit_score = spy
        return self

    def __exit__(self, *e):
        _LT._decentred_fit_score = self._orig
        _LT.DECENTRED_FIT_ARBITER = self._old_flag
        return False


def main():
    orders = os.environ.get('ORDERS', '-1,0').split()
    spec = os.environ.get('PATTERNS', 'v532=OFF,C11=-')
    arms = []
    for tok in spec.split(','):
        name, _, pat = tok.partition('=')
        arms.append((name, None if pat.upper() == 'OFF' else pat))
    rn, rs, clip, nlo = 1024, 4, 3.0, 321
    print(FI._provenance(), flush=True)
    _pre, post, _g, period = C.geometry()
    env, R, dx, _P = C.chain_a(n=rn, rs=rs)
    print(f"\n{'order':>8} {'oracle':>9}" + ''.join(
        f"{n:>12}" for n, _ in arms) + ''.join(
        f"{'res ' + n:>12}" for n, _ in arms), flush=True)
    for o in orders:
        m, n = (int(v) for v in o.split(','))
        L, M = m * LAM / period, n * LAM / period
        ch = trace(RayBundle(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
                             L=np.array([L]), M=np.array([M]),
                             N=np.array([np.sqrt(1 - L * L - M * M)]),
                             wavelength=LAM, alive=np.ones(1, bool),
                             opd=np.zeros(1)),
                   C.post_surfaces(post), LAM,
                   output_filter='last').image_rays
        xci, yci = float(ch.x[0]), float(ch.y[0])
        t0 = time.time()
        a = FI.oracle_launch(env, R, dx, L, M, nlo, clip, 1, post, BACK)
        r = H.rs_spot(*a[:7], BACK, xci, yci, dx_out=0.4e-6, n_out=61,
                      nl=a[7])
        I = np.ascontiguousarray(r['I'])
        cx, cy = RD.centroid(I, r['ax'])
        orc = RD.ee(I, r['ax'], cx, cy, 3e-6, 'area') * 100
        vals, shas, seen = [], [], []
        for _name, pat in arms:
            with Force(pat) as f:
                res, _w = FI.run_chain(post, env, R, dx, L, M, rs, 'off', 0)
                b = FI.chain_launch(res, L, M, 9999, clip, 1, post, BACK,
                                    'exact')
            seen.append(''.join(ch2 for _g, h, ch2, _v in f.calls if h == 0))
            rr = H.rs_spot(*b[:7], BACK, xci, yci, dx_out=0.4e-6, n_out=61,
                           nl=b[7])
            J = np.ascontiguousarray(rr['I'])
            ccx, ccy = RD.centroid(J, rr['ax'])
            vals.append(RD.ee(J, rr['ax'], ccx, ccy, 3e-6, 'area') * 100)
            shas.append(hashlib.sha256(J.tobytes()).hexdigest()[:8])
        print(f"{o:>8} {orc:9.4f}" + ''.join(f"{v:12.4f}" for v in vals)
              + ''.join(f"{orc - v:12.4f}" for v in vals)
              + f"   shas {'/'.join(shas)}"
              + f"  applied {'/'.join(s or '.' for s in seen)}"
              + f"  [{time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
