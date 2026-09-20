"""VERIFY-WP-C2 items 2 and 3 -- a DETERMINISTIC instrument: count the
array work each switch adds and removes.

Why this exists.  Three timing instruments were tried against the two
switches on this box and all three moved under load: batched CPU time (the
15.6 ms Windows tick quantises a 3.5 ms body at 0.9 %, which read the
9.6 % renormalise block as exactly ZERO), wall-clock minima (the same two
arms came out 9.6 % apart on one run and 5.7 % apart WITH THE SIGN REVERSED
on the next -- ``renormalize=True`` cannot be faster than
``renormalize=False``, it does strictly more work, so that reading is the
instrument failing), and cProfile shares (stable, but blind to code inlined
in ``_refract``).

This instrument does not measure time at all.  A ``ndarray`` subclass
intercepts ``__array_ufunc__`` and counts every element-wise operation and
every element touched, so the answer is a COUNT: the same on both builds,
the same under any load, and reproducible to the last digit.  Time is then
predicted with one first-order model -- equal cost per element-op -- which
is stated as a model, not measured.

Usage: ``LUMENAIRY_ROOT=<root> python vc2_opcount.py OUT.json``
"""
import json
import os
import sys

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)

import numpy as np                                            # noqa: E402
import lumenairy as la                                        # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__

from lumenairy.raytrace.surface import RayBundle, Surface     # noqa: E402
from lumenairy.raytrace.trace import trace                    # noqa: E402

WL = 587.5618e-9
N_RAYS = int(os.environ.get('VC2_N', '4096'))

_TALLY = {'ops': 0, 'elements': 0, 'by_ufunc': {}}


def _strip(o):
    if isinstance(o, np.ndarray):
        return o.view(np.ndarray)
    if isinstance(o, (list, tuple)):
        return type(o)(_strip(x) for x in o)
    return o


def _wrap(o):
    if isinstance(o, np.ndarray):
        return o.view(Counted)
    if isinstance(o, (list, tuple)):
        return type(o)(_wrap(x) for x in o)
    return o


def _tally(name, inputs):
    n = 0
    for a in inputs:
        if isinstance(a, np.ndarray) and a.ndim:
            n = max(n, a.size)
        elif isinstance(a, (list, tuple)):
            for b in a:
                if isinstance(b, np.ndarray) and b.ndim:
                    n = max(n, b.size)
    _TALLY['ops'] += 1
    _TALLY['elements'] += n
    d = _TALLY['by_ufunc'].setdefault(name, [0, 0])
    d[0] += 1
    d[1] += n


class Counted(np.ndarray):
    """An ndarray that tallies every element-wise operation applied to it.

    Both protocols are implemented.  ``__array_ufunc__`` alone is not
    enough: ``np.where`` is dispatched through ``__array_function__`` and,
    without it, returns a BASE ndarray -- after which every downstream
    operation on that value is invisible, which silently under-counts
    exactly the branch that uses ``np.where`` most.
    """

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        _tally(f'{ufunc.__name__}.{method}', inputs)
        kw = dict(kwargs)
        if 'out' in kw:
            kw['out'] = _strip(kw['out'])
        return _wrap(getattr(ufunc, method)(*[_strip(a) for a in inputs],
                                            **kw))

    def __array_function__(self, func, types, args, kwargs):
        _tally(f'{getattr(func, "__name__", str(func))}.fn', args)
        impl = getattr(func, '_implementation', func)
        return _wrap(impl(*_strip(args),
                          **{k: _strip(v) for k, v in kwargs.items()}))


class _NpShim:
    """``numpy`` with ``asarray`` / ``asanyarray`` re-viewing as Counted.

    ``np.asarray`` on a subclass returns a BASE ndarray by contract, and
    ``surface._surface_sag_derivative`` opens with exactly that call -- so
    without this shim the GENERIC normal route stops being counted at its
    first line while the closed form keeps being counted, and the census
    comes out backwards.  Installed only on the three raytrace modules,
    and removed again afterwards.
    """

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def asarray(a, *args, **kwargs):
        return _wrap(np.asarray(a, *args, **kwargs))

    @staticmethod
    def asanyarray(a, *args, **kwargs):
        return _wrap(np.asanyarray(a, *args, **kwargs))


def _reset():
    _TALLY['ops'] = 0
    _TALLY['elements'] = 0
    _TALLY['by_ufunc'] = {}


def _snapshot():
    return dict(ops=_TALLY['ops'], elements=_TALLY['elements'],
                by_ufunc={k: list(v) for k, v in
                          sorted(_TALLY['by_ufunc'].items())})


def _s(R, th, gb, ga, sd, conic=0.0, asph=None, mirror=False):
    return Surface(radius=R, conic=conic, aspheric_coeffs=asph, thickness=th,
                   glass_before=gb, glass_after=ga, semi_diameter=sd,
                   is_mirror=mirror)


def _ladder(n_pairs):
    out = []
    for j in range(n_pairs):
        out.append(_s(0.0800 + 0.003 * j, 0.0055, 'air', 'N-BK7', 0.011))
        out.append(_s(-0.0900 - 0.003 * j, 0.0090, 'N-BK7', 'air', 0.011))
    out.append(_s(0.2500, 0.0300, 'air', 'N-BK7', 0.011))
    return out


PRES = {
    'doublet3': [_s(0.0517, 0.0090, 'air', 'N-BK7', 0.0125),
                 _s(-0.0345, 0.0025, 'N-BK7', 'N-SF5', 0.0125),
                 _s(-0.1200, 0.0400, 'N-SF5', 'air', 0.0125)],
    'stack5': _ladder(2),
    'stack7': _ladder(3),
    'stack9': _ladder(4),
    'ladder13': _ladder(6),
    'two_mirror': [_s(-0.3000, -0.1000, 'air', 'air', 0.060, mirror=True),
                   _s(-0.0900, 0.2000, 'air', 'air', 0.020, mirror=True)],
    'conic3_control': [_s(0.0517, 0.0090, 'air', 'N-BK7', 0.0125,
                          conic=-0.6),
                       _s(-0.0345, 0.0025, 'N-BK7', 'N-SF5', 0.0125,
                          conic=-1.2),
                       _s(-0.1200, 0.0400, 'N-SF5', 'air', 0.0125,
                          conic=0.4)],
    'asphere3_control': [_s(0.0517, 0.0090, 'air', 'N-BK7', 0.0125,
                            asph={4: 1e3}),
                         _s(-0.0345, 0.0025, 'N-BK7', 'N-SF5', 0.0125,
                            asph={4: -5e2}),
                         _s(-0.1200, 0.0400, 'N-SF5', 'air', 0.0125,
                            asph={4: 2e2})],
}


def _bundle(n):
    rng = np.random.default_rng(20260920)
    x = rng.uniform(-0.0070, 0.0070, n).view(Counted)
    y = rng.uniform(-0.0070, 0.0070, n).view(Counted)
    return RayBundle(x=x, y=y, z=np.zeros(n).view(Counted),
                     L=np.zeros(n).view(Counted),
                     M=np.zeros(n).view(Counted),
                     N=np.ones(n).view(Counted), wavelength=WL,
                     alive=np.ones(n, dtype=bool).view(Counted),
                     opd=np.zeros(n).view(Counted))


SHIMMED = ('lumenairy.raytrace.intersection', 'lumenairy.raytrace.surface',
           'lumenairy.raytrace.trace', 'lumenairy.raytrace._conic_core',
           'lumenairy.raytrace.exit_vertex')


def _count(surfs, **kw):
    import importlib
    mods = [importlib.import_module(m) for m in SHIMMED]
    mods = [m for m in mods if hasattr(m, 'np')]
    saved = [m.np for m in mods]
    rb = _bundle(N_RAYS)
    trace(rb, surfs, WL, output_filter='last', **kw)   # warm the caches
    shim = _NpShim()
    for m in mods:
        m.np = shim
    try:
        _reset()
        rb = _bundle(N_RAYS)
        trace(rb, surfs, WL, output_filter='last', **kw)
        snap = _snapshot()
    finally:
        for m, s in zip(mods, saved):
            m.np = s
    return snap


def main(out_path):
    res = dict(meta=dict(python=sys.version.split()[0],
                         numpy=np.__version__, lumenairy=la.__version__,
                         file=la.__file__, n_rays=N_RAYS))
    per = {}
    for name, surfs in PRES.items():
        n_s = len(surfs)
        c = {}
        for rn in ('surface', 'exit'):
            c[f'renorm_{rn}'] = _count(surfs, sphere_normal='analytic',
                                       renormalize=rn)
        for sn in ('generic', 'analytic'):
            c[f'normal_{sn}'] = _count(surfs, sphere_normal=sn,
                                       renormalize='exit')
        rs, re_ = c['renorm_surface'], c['renorm_exit']
        ng, na = c['normal_generic'], c['normal_analytic']
        per[name] = dict(
            n_surfaces=n_s,
            renorm=dict(
                ops_surface=rs['ops'], ops_exit=re_['ops'],
                elements_surface=rs['elements'],
                elements_exit=re_['elements'],
                ops_saved=rs['ops'] - re_['ops'],
                elements_saved=rs['elements'] - re_['elements'],
                elements_saved_fraction=((rs['elements'] - re_['elements'])
                                         / rs['elements']),
                predicted_speedup=(rs['elements'] / re_['elements'])),
            normal=dict(
                ops_generic=ng['ops'], ops_analytic=na['ops'],
                elements_generic=ng['elements'],
                elements_analytic=na['elements'],
                ops_saved=ng['ops'] - na['ops'],
                elements_saved=ng['elements'] - na['elements'],
                elements_saved_fraction=((ng['elements'] - na['elements'])
                                         / ng['elements']),
                predicted_speedup=(ng['elements'] / na['elements'])),
            by_ufunc_renorm_delta={
                k: [rs['by_ufunc'].get(k, [0, 0])[0]
                    - re_['by_ufunc'].get(k, [0, 0])[0],
                    rs['by_ufunc'].get(k, [0, 0])[1]
                    - re_['by_ufunc'].get(k, [0, 0])[1]]
                for k in sorted(set(rs['by_ufunc']) | set(re_['by_ufunc']))
                if (rs['by_ufunc'].get(k, [0, 0])
                    != re_['by_ufunc'].get(k, [0, 0]))})
    res['per_prescription'] = per

    real = [k for k in per if not k.endswith('_control')]
    res['summary'] = dict(
        renorm_predicted_speedup={k: per[k]['renorm']['predicted_speedup']
                                  for k in per},
        normal_predicted_speedup={k: per[k]['normal']['predicted_speedup']
                                  for k in per},
        renorm_range=[min(per[k]['renorm']['predicted_speedup']
                          for k in real),
                      max(per[k]['renorm']['predicted_speedup']
                          for k in real)],
        normal_range=[min(per[k]['normal']['predicted_speedup']
                          for k in real),
                      max(per[k]['normal']['predicted_speedup']
                          for k in real)],
        controls=[dict(name=k,
                       renorm=per[k]['renorm']['predicted_speedup'],
                       normal=per[k]['normal']['predicted_speedup'])
                  for k in per if k.endswith('_control')])
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(f"{'prescription':18s} {'S':>3s} {'renorm x':>9s} "
          f"{'saved el':>10s} {'normal x':>9s} {'saved el':>10s}")
    for k, v in per.items():
        print(f"{k:18s} {v['n_surfaces']:3d} "
              f"{v['renorm']['predicted_speedup']:9.4f} "
              f"{v['renorm']['elements_saved']:10d} "
              f"{v['normal']['predicted_speedup']:9.4f} "
              f"{v['normal']['elements_saved']:10d}")
    print()
    print('renorm ufunc delta (surface - exit), stack7:',
          json.dumps(per['stack7']['by_ufunc_renorm_delta']))


if __name__ == '__main__':
    main(sys.argv[1])
