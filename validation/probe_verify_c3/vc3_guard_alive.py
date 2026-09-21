"""VERIFY-WP-C3 CLAIM 10 -- the Kelly guard is NOT dead on the new default.

A caller-NAMED output lattice has no complementary form to fall back to
(``tf_available`` requires ``dx_out is None``), so the guard must still speak.
"""
import warnings
import numpy as np
import lumenairy.propagators.carrier as CA

LAM = 1.064e-6


def gauss(n, dx, w):
    x = (np.arange(n) - n / 2) * dx
    return np.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / w ** 2).astype(complex)


def run(tag, **kw):
    env = gauss(512, 8e-6, 0.5e-3)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        try:
            CA.propagate_carrier_referenced(env, np.inf, 5e-3, LAM, 8e-6, **kw)
            err = None
        except Exception as exc:                            # noqa: BLE001
            err = '%s: %s' % (type(exc).__name__, str(exc)[:90])
    kelly = [str(w.message) for w in rec if 'Kelly' in str(w.message)
             or 'sampling' in str(w.message).lower()]
    print('%-46s kelly=%d  err=%s' % (tag, len(kelly), err))
    for m in kelly[:1]:
        print('      ', m[:200].replace('\n', ' '))
    return len(kelly)


print('transport default =', CA.propagate_carrier_referenced.__wrapped__.__defaults__
      if hasattr(CA.propagate_carrier_referenced, '__wrapped__') else 'n/a')
n1 = run('UNNAMED lattice (default, fallback available)')
n2 = run('NAMED dx_out (no fallback)', dx_out=2.0e-6)
n3 = run('NAMED dx_out, explicit collins', dx_out=2.0e-6, transport='collins')
n4 = run('NAMED carrier_out', carrier_out=0.2)
print('GUARD ALIVE (named lattice warns):', bool(n2 or n3 or n4))
