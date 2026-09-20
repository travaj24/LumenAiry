"""WP-C2 item 1 -- the DERIVED floor of the ModalAsymptotic comparison.

The per-pixel disagreement between the library's batched cold-start
evaluation and the test's inline scalar one is dense and ~1e-8 relative
(see modal_mechanism.py: the two saddle locations agree to 7e-18, so it
is NOT a basin flip).  Two independent float64 evaluations of the same
mathematical field can only agree to ``eps * kappa``, where ``kappa`` is
the field's amplification of a relative perturbation of its own inputs.

This probe MEASURES kappa on the running build with a finite-difference
ladder over the fit's polynomial coefficients, checks that the response
is linear in the perturbation (so the ladder is measuring conditioning
and not a threshold), and reports ``eps * kappa``.

Usage:  LUMENAIRY_ROOT=<root> python modal_conditioning.py <out.json>
"""
import dataclasses
import importlib.util
import json
import os
import sys

import numpy as np

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)

import lumenairy as la  # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    out_path = sys.argv[1]
    tp = _load('_tp', os.path.join(_ROOT, 'tests', 'unit',
                                   'test_audit_propagation.py'))
    fit = tp._build_singlet_fit()
    N = 32
    ax = np.linspace(-5e-6, 5e-6, N)
    S2X, S2Y = np.meshgrid(ax, ax, indexing='xy')
    kw = dict(source_point=(0.0, 0.0),
              source_amplitudes={(0, 0): 1.0 + 0.0j},
              pupil_amplitudes={(0, 0): 1.0 + 0.0j},
              w_s=50e-6, w_p=0.02, v2_centre=(0.0, 0.0),
              s2_grid_x=S2X, s2_grid_y=S2Y)
    base = tp.propagate_modal_asymptotic(fit, **kw)
    peak = float(np.max(np.abs(base)))
    rng = np.random.default_rng(20260920)
    r = rng.normal(size=np.asarray(fit.coef_phi).shape)
    r /= np.linalg.norm(r)
    rows = []
    for delta in (1e-13, 1e-12, 1e-11, 1e-10):
        f2 = dataclasses.replace(
            fit, coef_phi=np.asarray(fit.coef_phi) * (1.0 + delta * r))
        alt = tp.propagate_modal_asymptotic(f2, **kw)
        chg = float(np.max(np.abs(alt - base))) / peak
        rows.append(dict(delta=delta, rel_change=chg, kappa=chg / delta))
        print(f'delta={delta:.1e} rel_change={chg:.4e} '
              f'kappa={chg / delta:.4e}', flush=True)
    kappas = [row['kappa'] for row in rows]
    eps = float(np.finfo(np.float64).eps)
    out = dict(
        peak=peak, rows=rows,
        kappa_median=float(np.median(kappas)),
        kappa_spread=float(max(kappas) / min(kappas)),
        eps=eps,
        floor_eps_kappa=eps * float(np.median(kappas)),
    )
    print('floor eps*kappa =', out['floor_eps_kappa'])
    meta = dict(python=sys.version, numpy=np.__version__,
                lumenairy=la.__version__, platform=sys.platform)
    with open(out_path, 'w') as fh:
        json.dump(dict(meta=meta, out=out), fh, indent=1)


if __name__ == '__main__':
    main()
