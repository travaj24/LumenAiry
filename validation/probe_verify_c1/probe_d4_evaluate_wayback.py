"""VERIFY-C1 round 2, defect D4 -- ``lumenairy.evaluate``'s way back.

``_prescription_to_elements`` emits ``{'type': 'aperture', 'shape':
'circular', 'params': {'diameter': D}}`` for every ``is_stop=True`` surface,
with no ``'edge'`` key, so ``evaluate``'s answer moved with the 5.49.0
default and -- until round 2 -- the only route to the pre-5.49 answer was the
PRIVATE builder plus a hand-driven ``propagate_through_system``.  Round 2 gave
``evaluate`` an ``aperture_edge=`` / ``aperture_edge_samples=`` keyword.

This probe digests the returned field on one Zemax-shape prescription with a
STOP surface, so the claim "``aperture_edge='hard'`` restores the parent's
answer BIT FOR BIT" is checked archive-to-archive rather than inside one
working copy.  Run it against each tree from INSIDE that tree, so
``lumenairy`` cannot bind elsewhere; the digest of ``lumenairy.__file__`` is
printed every time.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=<tree> python validation/probe_verify_c1/probe_d4_evaluate_wayback.py <tag>

writes ``validation/probe_verify_c1/d4_evaluate_<tag>.json`` next to this
file when that directory exists, else beside the CWD.

Arms:
  pre_default     ``evaluate(rx, src)``                       -- old tree only
  default         ``evaluate(rx, src)``                       -- new default
  private_hard    the pre-round-2 way back (private builder)  -- every tree
  kw_hard         ``evaluate(..., aperture_edge='hard')``     -- new tree only
  kw_gray4        ``evaluate(..., aperture_edge='gray',
                                  aperture_edge_samples=4)``  -- new tree only

``kw_hard`` must equal ``private_hard`` here AND the old tree's
``pre_default``; ``kw_gray4`` must equal ``default``, since 4 is the default.
"""
import hashlib
import json
import os
import platform
import sys

import numpy as np

import lumenairy as la

RX = {
    'elements': [
        {'surf_num': 1, 'element_type': 'surface', 'radius': np.inf,
         'glass_after': 'air', 'is_stop': True, 'semi_diameter': 1.2e-3,
         'comment': 'STOP'},
        {'surf_num': 2, 'element_type': 'surface', 'radius': 0.05,
         'glass_after': 'N-BK7', 'semi_diameter': 2e-3},
        {'surf_num': 3, 'element_type': 'surface', 'radius': -0.05,
         'glass_after': 'air', 'semi_diameter': 2e-3},
    ],
    'all_thicknesses': [2e-3, 3e-3, 20e-3],
    'aperture_diameter': 2.4e-3,
}


def _src():
    return la.Source.gaussian(N=128, dx=40e-6, wavelength=633e-9, w0=1.0e-3)


def _digest(arr):
    a = np.ascontiguousarray(np.asarray(arr))
    return {'sha256': hashlib.sha256(a.tobytes()).hexdigest(),
            'dtype': str(a.dtype), 'shape': list(a.shape)}


def _private_hard():
    """The ONLY way back before round 2: reach into the private builder,
    inject the key ``evaluate`` had no argument for, drive the chain."""
    from lumenairy.propagators.system import (_prescription_to_elements,
                                              propagate_through_system)
    elements = [dict(e, edge='hard') if e.get('type') == 'aperture' else e
                for e in _prescription_to_elements(RX)]
    out = propagate_through_system(np.asarray(_src().E), elements, 633e-9,
                                   dx=40e-6)
    return np.asarray(out[0] if isinstance(out, tuple) else out)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'run'
    print('lumenairy.__file__ =', lumenairy_file := la.__file__)
    print('python             =', sys.version.split()[0], platform.platform())
    print('numpy              =', np.__version__)

    arms = {}
    arms['default'] = _digest(la.evaluate(RX, _src()).field)
    arms['private_hard'] = _digest(_private_hard())

    import inspect
    params = inspect.signature(la.evaluate).parameters
    has_kw = 'aperture_edge' in params
    print('evaluate exposes aperture_edge =', has_kw)
    if has_kw:
        arms['kw_hard'] = _digest(
            la.evaluate(RX, _src(), aperture_edge='hard').field)
        arms['kw_gray4'] = _digest(
            la.evaluate(RX, _src(), aperture_edge='gray',
                        aperture_edge_samples=4).field)
        arms['kw_gray1'] = _digest(
            la.evaluate(RX, _src(), aperture_edge='gray',
                        aperture_edge_samples=1).field)
        refusals = {}
        for label, kw in (('bad_edge', {'aperture_edge': 'soft'}),
                          ('bad_samples', {'aperture_edge_samples': 2.5})):
            try:
                la.evaluate(RX, _src(), **kw)
                refusals[label] = 'ACCEPTED'
            except (ValueError, TypeError) as e:
                refusals[label] = f'{type(e).__name__}: {e}'
        arms['_refusals'] = refusals

    for k, v in sorted(arms.items()):
        if k.startswith('_'):
            print(f'  {k}: {v}')
        else:
            print(f'  {k:14s} {v["sha256"][:16]}  {v["dtype"]} {v["shape"]}')

    out = {'tag': tag, 'lumenairy': lumenairy_file,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'platform': platform.platform(),
           'exposes_aperture_edge': has_kw, 'arms': arms}
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, f'd4_evaluate_{tag}.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print('wrote', path)


if __name__ == '__main__':
    main()
