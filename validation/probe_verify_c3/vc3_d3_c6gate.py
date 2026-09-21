"""Why ||E6-E4|| is EXACTLY 0.0 on the flipped default: does the niche-C6
stationary-phase launch engage at all?"""
import sys, warnings
import numpy as np
sys.path.insert(0, 'tests/unit'); sys.path.insert(0, 'tests')
import lumenairy.elements._lens_traced as _lt
import test_niche_d3_guards as D3

calls = []
_orig = _lt._fit_residual_eikonal
def _spy(*a, **k):
    calls.append(k.get('degree', '<default>'))
    return _orig(*a, **k)
_lt._fit_residual_eikonal = _spy

def run(tr, degree):
    calls.clear()
    _deg = _lt._REMAP_RESID_EIKONAL_DEGREE
    _lch = _lt.REMAP_STATIONARY_PHASE_LAUNCH
    _lt._REMAP_RESID_EIKONAL_DEGREE = degree
    _lt.REMAP_STATIONARY_PHASE_LAUNCH = True
    kw = {} if tr is None else {'transport': tr}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = D3._chain(D3._mux_fan(0.023), quiet=True, focus_readout=None,
                          on_multi_congruence='ignore', **kw)
    finally:
        _lt._REMAP_RESID_EIKONAL_DEGREE = _deg
        _lt.REMAP_STATIONARY_PHASE_LAUNCH = _lch
    print('transport=%-9s degree=%d  _fit_residual_eikonal calls=%d  '
          'norm=%.6g' % (str(tr), degree, len(calls),
                         float(np.linalg.norm(r.field))))
    return r.field

for tr in ('sziklas', 'collins'):
    a = run(tr, 6); b = run(tr, 4)
    print('   -> max|E6-E4| =', float(np.max(np.abs(a - b))))
