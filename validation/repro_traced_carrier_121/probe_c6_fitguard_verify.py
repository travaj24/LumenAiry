# BYTE-IDENTITY / FAIL-BEFORE verification for
# ``REMAP_STATIONARY_PHASE_FIT_GUARD`` (the niche-C6 on-axis-ghost fix).
#
# Three claims have to hold, and each is checked with ``np.array_equal`` on the
# element's returned complex field -- not a tolerance:
#
#   1. FAIL-BEFORE.  With the flag ``False`` the patched tree reproduces the
#      pre-patch tree BIT FOR BIT, on every order.
#   2. TILTED UNCHANGED.  Every off-centre order already takes the weighted
#      branch, so the fix cannot touch it: patched == pre-patch with the flag
#      ON as well.  This is what protects the C6 recovery.
#   3. C6-OFF UNCHANGED.  With ``REMAP_STATIONARY_PHASE_LAUNCH = False`` the
#      guard is never engaged (``_resid_eik is None``), so the whole legacy
#      path -- untilted included -- is bit-identical.
#
# and one that must NOT hold: the on-axis field with the flag ON must DIFFER
# from pre-patch (otherwise the fix is inert).
#
# usage, from validation/repro_traced_carrier_121/ :
#   PIN=<pre-patch tree> TAG=pre python probe_c6_fitguard_verify.py
#   TAG=post python probe_c6_fitguard_verify.py
#   CMP=pre,post python probe_c6_fitguard_verify.py
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# The pin must be prepended BEFORE lumenairy is imported anywhere (the audit's
# artefact 2: _d121_common prepends the live repo at import time).
_PIN = os.environ.get('PIN', '')
if _PIN and os.path.isdir(os.path.join(_PIN, 'lumenairy')):
    sys.path.insert(0, _PIN)

import lumenairy.elements._lens_traced as LT                     # noqa: E402


def _compare():
    a, b = os.environ['CMP'].split(',')
    da = np.load(os.path.join(HERE, f'_fitguard_{a}.npz'))
    db = np.load(os.path.join(HERE, f'_fitguard_{b}.npz'))
    keys = [k for k in da.files if not k.startswith('_')]
    print("%-26s %-12s %-12s  %s" % ('case', a, b, 'array_equal / max|dE|'))
    print('-' * 74)
    for k in sorted(keys):
        if k not in db.files:
            print("%-26s  (missing in %s)" % (k, b))
            continue
        eq = bool(np.array_equal(da[k], db[k]))
        mx = float(np.abs(da[k] - db[k]).max())
        print("%-26s %-12s %-12s  %-5s  max|dE| %.3e"
              % (k, a, b, str(eq), mx))
    return 0


def main():
    if os.environ.get('CMP'):
        return _compare()
    warnings.filterwarnings('ignore')
    import hashlib
    import probe_c6_element as E6
    import probe_ghost_c6 as G
    import wfe_probe_common as P
    import _d121_common as C
    from lumenairy.elements._lens_traced import TiltedCarrier

    tag = os.environ.get('TAG', 'post')
    rs = int(os.environ.get('RS', '4'))
    orders = [tuple(int(v) for v in o.split(','))
              for o in os.environ.get('ORDERS',
                                      '0,0 -2,0 -4,0 -4,-2').split()]
    has_guard = hasattr(LT, 'REMAP_STATIONARY_PHASE_FIT_GUARD')
    print("tag %s   lib %s\n    sha256 %s   guard flag present: %s"
          % (tag, LT.__file__,
             hashlib.sha256(open(LT.__file__, 'rb').read()).hexdigest()[:16],
             has_guard))
    _pre, post, _g, _p = C.geometry()
    presc = post[-1]['prescription']
    out = {}
    for (m, n) in orders:
        E_in, _Eo, carv, dx = E6.get_call(m, n, rs=rs)
        car = TiltedCarrier(*carv)
        cases = [('c6on', dict(flag=True)), ('c6off', dict(flag=False))]
        for lbl, kw in cases:
            out[f'{m}_{n}__{lbl}'] = G.element(E_in, presc, dx, car, rs,
                                               **kw)[0]
        if has_guard:
            old = LT.REMAP_STATIONARY_PHASE_FIT_GUARD
            LT.REMAP_STATIONARY_PHASE_FIT_GUARD = False
            try:
                out[f'{m}_{n}__guardoff'] = G.element(
                    E_in, presc, dx, car, rs, flag=True)[0]
            finally:
                LT.REMAP_STATIONARY_PHASE_FIT_GUARD = old
            # the fix's TARGET configuration, reached the old way (forced
            # decentre gates) -- patched c6on must equal this exactly
            out[f'{m}_{n}__forcedec'] = G.element(
                E_in, presc, dx, car, rs, flag=True, force_dec=True)[0]
        print("  done order (%d,%d)" % (m, n), flush=True)
    np.savez(os.path.join(HERE, f'_fitguard_{tag}.npz'), **out)
    print("wrote _fitguard_%s.npz (%d cases)" % (tag, len(out)))
    # in-process cross-check where possible
    for (m, n) in orders:
        if has_guard:
            k1, k2 = f'{m}_{n}__c6on', f'{m}_{n}__forcedec'
            print("  (%d,%d) c6on == forced-decentre target: %s"
                  % (m, n, np.array_equal(out[k1], out[k2])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
