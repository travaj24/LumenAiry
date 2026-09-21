"""VERIFY-WP-C3 ROUND 2, item 3 -- the three quantities the five d3
SEPARATION ids place bars on, measured directly on whichever tree is bound.

    bad6 = _linearity_error(0.023) at the shipped residual-eikonal degree
    good6 = _linearity_error(0.0005) at the same
    bad4 = _linearity_error(0.023) at degree 4

The shipped bars are ``bad6 > 5 * good6`` and ``bad4 > 2 * bad6``.  Reading
them on the branch and on the C3 x C5 merge says whether the merge moved the
quantity the bars are about, or only re-ran the assertion.
"""
import importlib.util
import json
import os
import pathlib
import sys

TREE = os.path.abspath(os.environ['VC3_TREE'])
sys.path.insert(0, TREE)
import lumenairy                                              # noqa: E402
import lumenairy.propagators.carrier as C                     # noqa: E402

assert os.path.abspath(lumenairy.__file__).startswith(TREE)
sys.path.insert(0, str(pathlib.Path(TREE) / 'tests' / 'unit'))
_P = pathlib.Path(TREE) / 'tests' / 'unit' / 'test_niche_d3_guards.py'
_spec = importlib.util.spec_from_file_location('d3mod', _P)
D3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D3)

_n = {'collins_legs': 0, 'tau_evaluated': 0, 'tau_fired': 0}
_real_ct, _real_dep = C._collins_transport, C._collins_exact_kernel_departure


def _ct(*a, **k):
    _n['collins_legs'] += 1
    return _real_ct(*a, **k)


def _dep(z, th, wl):
    d = _real_dep(z, th, wl)
    _n['tau_evaluated'] += 1
    tau = C._GAP_KERNEL_ACCURACY_TAU
    if tau is not None and d > float(tau):
        _n['tau_fired'] += 1
    return d


C._collins_transport = _ct
C._collins_exact_kernel_departure = _dep

import lumenairy.elements._lens_traced as _lt                 # noqa: E402

bad6 = D3._linearity_error(0.023)
good6 = D3._linearity_error(0.0005)
_deg = _lt._REMAP_RESID_EIKONAL_DEGREE
_lt._REMAP_RESID_EIKONAL_DEGREE = 4
try:
    bad4 = D3._linearity_error(0.023)
finally:
    _lt._REMAP_RESID_EIKONAL_DEGREE = _deg

out = {'tree': TREE, 'lumenairy': lumenairy.__file__,
       'python': sys.version.split()[0],
       'tau': C._GAP_KERNEL_ACCURACY_TAU,
       'chain_default_transport': __import__('inspect').signature(
           C.propagate_traced_carrier_chain).parameters['transport'].default,
       'bad6': bad6, 'good6': good6, 'bad4': bad4,
       'sep_bad6_over_good6': bad6 / good6, 'bar_sep': 5.0,
       'ratio_bad4_over_bad6': bad4 / bad6, 'bar_deg': 2.0,
       'collins_counts': dict(_n)}
tag = os.environ.get('VC3_TAG', 'x')
p = os.path.join(os.environ['VC3_OUT'], f'vr2_d3_bars_{tag}.json')
with open(p, 'w', encoding='utf-8') as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1))
