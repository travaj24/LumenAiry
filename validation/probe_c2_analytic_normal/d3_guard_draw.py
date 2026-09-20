"""WP-C2 item 5 -- is the d3 guard's degree-attribution ratio a property of
the library, or a draw?

``tests/unit/test_niche_d3_guards.py`` has two arms that moved under this
work package's defaults:

* ``test_the_separation_survives_the_c10_residual_degree_and_is_caused_by_it``
  claim 2, ``bad4 > 2.0 * bad6`` (measured 15x-29x when it was written,
  1.17x here);
* ``test_the_residual_degree_moves_the_multiplexed_route_only_through_c6``,
  ``moved > 1.0`` (measured 39.8 / 43.9 when written, 0.836 here).

Both are measured with niche C6's stationary-phase launch ENGAGED on a
MULTIPLEXED 2x2 order fan.  The file's own docstring for the sibling
``test_c13_makes_the_d3_separation_build_independent`` says what that
state is: the C6 residual-eikonal fit "explains NONE of its own data at
EVERY degree 1-6", returns a model whose gradient reaches ``|grad a| =
974`` against a physical maximum of 1, and "perturbing ONLY that fit's
coefficients by a relative 1e-12 ... moves ``|mux|`` by 163x".  That
sibling's CONDITION was moved (2026-08-08) precisely because "with that
launch ENGAGED this fixture cannot measure a solver property at all".

So the question this probe answers is not "did WP-C2 break it" but
"can these two magnitudes carry a bar at all".  It measures each arm

1. at the shipped 5.49 defaults,
2. with the pre-5.49 keywords forced back on ``trace`` /
   ``trace_world``'s ``__defaults__`` -- if the readings return, the
   route is the trigger,
3. under a ONE-ULP nudge of the input field at fixed defaults -- a
   perturbation no library behaviour can be attributed to.  If a 1-ULP
   input nudge swings the ratio by a comparable factor, the quantity is
   a draw and no bar on it is a property of the library.

Usage:  LUMENAIRY_ROOT=<root> python d3_guard_draw.py <out.json>
"""
import importlib
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

trace_mod = importlib.import_module('lumenairy.raytrace.trace')
wtrace_mod = importlib.import_module('lumenairy.raytrace.world_trace')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _set_defaults(renorm, sph):
    for fn in (trace_mod.trace, wtrace_mod.trace_world):
        d = list(fn.__defaults__)
        d[-2], d[-1] = renorm, sph
        fn.__defaults__ = tuple(d)


def _arms(d3, nudge=0.0):
    """``(bad6, good6, bad4, moved)`` on the module's own fixtures."""
    lt = d3._lens_traced

    def lin(tilt):
        X, Y = d3._grid(d3._CN, d3._CDX)
        G = d3._gauss(d3._CN, d3._CDX, d3._CW)
        if nudge:
            to = np.inf if nudge > 0 else -np.inf
            G = np.asarray(G)
            if np.iscomplexobj(G):
                G = (np.nextafter(G.real, to)
                     + 1j * np.nextafter(G.imag, to))
            else:
                G = np.nextafter(G, to)
        parts = [G * np.exp(1j * d3._K0 * tilt * (sx * X + sy * Y))
                 for sx in (-1, 1) for sy in (-1, 1)]
        kw = dict(focus_readout=None, on_multi_congruence='ignore')
        ref = None
        for p in parts:
            f = d3._chain(p, quiet=True, **kw).field
            ref = f if ref is None else ref + f
        mux = d3._chain(sum(parts), quiet=True, **kw).field
        return float(np.linalg.norm(mux - ref) / np.linalg.norm(ref))

    bad6 = lin(0.023)
    good6 = lin(0.0005)
    deg = lt._REMAP_RESID_EIKONAL_DEGREE
    lt._REMAP_RESID_EIKONAL_DEGREE = 4
    try:
        bad4 = lin(0.023)
    finally:
        lt._REMAP_RESID_EIKONAL_DEGREE = deg
    on6 = d3._mux_chain_field(0.023, degree=6, launch=True)
    on4 = d3._mux_chain_field(0.023, degree=4, launch=True)
    off6 = d3._mux_chain_field(0.023, degree=6, launch=False)
    off4 = d3._mux_chain_field(0.023, degree=4, launch=False)
    ref = float(np.linalg.norm(on6))
    return dict(
        bad6=bad6, good6=good6, bad4=bad4,
        separation=bad6 / good6 if good6 else float('inf'),
        degree_ratio=bad4 / bad6 if bad6 else float('inf'),
        moved=float(np.linalg.norm(on6 - on4)) / ref if ref else float('nan'),
        launch_off_inert=bool(np.array_equal(off6, off4)),
        launch_off_max_delta=float(np.max(np.abs(off6 - off4))),
    )


def main():
    out_path = sys.argv[1]
    d3 = _load('_d3', os.path.join(_ROOT, 'tests', 'unit',
                                   'test_niche_d3_guards.py'))
    shipped = (trace_mod.trace.__defaults__[-2],
               trace_mod.trace.__defaults__[-1])
    res = {}
    cases = [
        ('shipped 5.49 defaults', shipped, 0.0),
        ('pre-5.49 defaults forced', ('surface', 'generic'), 0.0),
        ('shipped defaults, input nudged 1 ULP up', shipped, +1.0),
        ('shipped defaults, input nudged 1 ULP down', shipped, -1.0),
        ('pre-5.49 defaults, input nudged 1 ULP up', ('surface', 'generic'),
         +1.0),
    ]
    only = os.environ.get('D3_ONLY')
    if only:
        cases = [c for c in cases if only in c[0]]
    for label, defaults, nudge in cases:
        _set_defaults(*defaults)
        r = _arms(d3, nudge=nudge)
        res[label] = dict(r, defaults=list(defaults), nudge=nudge)
        print('%-44s bad6=%.4f good6=%.4f sep=%.1fx bad4=%.4f '
              'bad4/bad6=%.3fx moved=%.4f inert_off=%s'
              % (label, r['bad6'], r['good6'], r['separation'], r['bad4'],
                 r['degree_ratio'], r['moved'], r['launch_off_inert']),
              flush=True)
    _set_defaults(*shipped)
    ratios = [v['degree_ratio'] for v in res.values()]
    moveds = [v['moved'] for v in res.values()]
    summary = dict(
        degree_ratio_range=[min(ratios), max(ratios)],
        degree_ratio_spread=max(ratios) / min(ratios),
        moved_range=[min(moveds), max(moveds)],
        moved_spread=max(moveds) / min(moveds),
        inert_off_everywhere=all(v['launch_off_inert']
                                 for v in res.values()),
        separation_range=[min(v['separation'] for v in res.values()),
                          max(v['separation'] for v in res.values())],
    )
    for k, v in summary.items():
        print(k, v)
    meta = dict(python=sys.version, numpy=np.__version__,
                lumenairy=la.__version__, platform=sys.platform)
    with open(out_path, 'w') as fh:
        json.dump(dict(meta=meta, summary=summary, results=res), fh, indent=1)
    print('wrote', out_path)


if __name__ == '__main__':
    main()
