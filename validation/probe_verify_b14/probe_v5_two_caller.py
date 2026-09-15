"""VERIFY-B14 arm 5 -- warning attribution at TWO library depths, measured on
a battery of warn sites rather than on one.

THE INSTRUMENT.  ``warnings.catch_warnings(record=True)`` records the
``filename`` / ``lineno`` that the emitting call's ``stacklevel`` resolved to.
A warning is CORRECTLY attributed when that filename is this probe file (the
nearest frame outside the package) and MISATTRIBUTED when it is a file under
``lumenairy/``.  No guess about how deep any entry point is is needed.

THE SECOND DEPTH, without editing the package.  ``caller_stacklevel()``
classifies a frame as library code by
``os.path.abspath(frame.f_code.co_filename).startswith(<package dir>)``, so a
function COMPILED with a ``co_filename`` under the package directory is a
library frame to every consumer of that rule: ``warnings`` counts it like any
other frame and the walker treats it like any other library frame.  That gives
a synthetic but faithful extra library frame for EVERY site, not only for the
sites that happen to have a second real entry point.

The instrument is validated against the one pair that has a genuine two-depth
route -- ``propagate_carrier_referenced`` reached directly, and the same warn
site reached through ``carrier_referenced_focus_readout`` -- so the synthetic
frame's verdict can be compared with a real one on the same tree.
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

import lumenairy as la
from lumenairy.propagators import carrier as CA, carrier_field as CF, system as SY

_PKG = os.path.abspath(os.path.dirname(la.__file__))
_ME = os.path.abspath(__file__)


def _library_frame(nest):
    """A callable forwarding ``(fn, args, kwargs)`` through ``nest`` frames
    whose ``co_filename`` AND ``__file__`` sit inside the package directory.

    The synthetic frames are the INNERMOST ones -- the deepest of them calls
    ``fn`` itself -- so they lie between this file and the library entry
    point, which is where a real intermediate library frame would be.
    """
    if nest <= 0:
        return lambda fn, a, kw: fn(*a, **kw)
    fake = os.path.join(_PKG, 'propagators', '_verify_b14_synth_depth.py')

    def _compile(body):
        code = compile(body, fake, 'exec')
        # ``warnings`` reads the reported filename from the frame's GLOBALS
        # (``__file__``); ``caller_stacklevel`` reads ``co_filename``.  The
        # synthetic frame has to look like library code to both.
        ns = {'_inner': cur, '__file__': fake,
              '__name__': 'lumenairy.propagators._verify_b14_synth_depth'}
        exec(code, ns)
        return ns['_w']

    cur = None
    cur = _compile('def _w(fn, a, kw):\n    return fn(*a, **kw)\n')
    for _ in range(nest - 1):
        cur = _compile('def _w(fn, a, kw):\n    return _inner(fn, a, kw)\n')
    return cur


def _run(fn, args, kwargs, nest):
    fwd = _library_frame(nest)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            out = fwd(fn, args, kwargs)
            err = None
        except Exception as exc:
            out, err = None, type(exc).__name__ + ': ' + str(exc)[:120]
    rows = []
    for w in caught:
        f = os.path.abspath(w.filename)
        rows.append({'file': os.path.basename(f), 'lineno': w.lineno,
                     'cat': w.category.__name__,
                     'in_package': bool(f.startswith(_PKG)),
                     'is_probe': bool(f == _ME),
                     'msg': str(w.message)[:90]})
    return out, err, rows


def _env(n=64, dx=8e-6, w=60e-6):
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    return np.exp(-(X ** 2 + Y ** 2) / (w ** 2)).astype(np.complex128)


def _cf(env, dx, wl):
    return CF.CarrierField(
        envelope=env.copy(),
        grid=CF.FieldGrid(shape=env.shape, dx=dx, dy=dx, origin=(0.0, 0.0)),
        carrier=CF.CarrierSpec(R=-0.05, centre=(0.0, 0.0), tilt=(0.0, 0.0),
                               piston=0.0),
        wavelength=wl)


def _cases():
    env = _env()
    dx, wl = 8e-6, 633e-9
    big = _env(n=128, dx=4e-6, w=120e-6)
    return {
        'carrier.tilt_inert_direct_entry': (
            CA.propagate_carrier_referenced, (env, -0.05, 5e-3),
            dict(wavelength=wl, dx=dx, gap_kernel='fresnel',
                 tilt=(0.12, 0.0)), 'INERT on this call'),
        'carrier.tilt_inert_via_focus_readout': (
            CA.carrier_referenced_focus_readout, (env, -0.05, 5e-3),
            dict(wavelength=wl, dx=dx, gap_kernel='fresnel', tilt=(0.12, 0.0),
                 dx_out=dx, N_out=32), 'INERT on this call'),
        'carrier.fit_radius_aliased': (
            CA.carrier_referenced_fit_radius, (env,),
            dict(wavelength=wl, dx=dx, on_aliased='warn'), None),
        'carrier.focus_containment': (
            CA.carrier_referenced_focus_readout, (env, -0.05, 4.99e-2),
            dict(wavelength=wl, dx=dx, dx_out=dx / 8.0, N_out=64,
                 on_focus_containment='warn', on_replica='warn'), None),
        'carrier.replica': (
            CA.carrier_referenced_focus_readout, (env, -0.05, 5e-3),
            dict(wavelength=wl, dx=dx, dx_out=dx * 8.0, N_out=256,
                 on_replica='warn', on_focus_containment='warn'), None),
        'carrier.collins_sampling': (
            CA.propagate_carrier_referenced, (env, -0.05, 5e-3),
            dict(wavelength=wl, dx=dx, transport='collins',
                 on_collins_sampling='warn', dx_out=dx / 64.0), None),
        'carrier.exact_focus_readout': (
            CA.carrier_referenced_exact_focus_readout, (big, -0.05, 4.9e-2),
            dict(wavelength=wl, dx=4e-6, dx_out=4e-6 * 40, N_out=256,
                 on_readout_window='warn', on_replica='warn',
                 on_n_fine_cap='warn', n_fine_cap=64), None),
        'carrier_field.setattr_deprecation': (
            (lambda f: setattr(f, 'wavelength', float(f.wavelength))),
            (_cf(env, dx, wl),), {}, None),
        'carrier_field.re_reference_guards': (
            CF.re_reference,
            (_cf(env, dx, wl),
             CF.CarrierSpec(R=2e-4, centre=(0.0, 0.0), tilt=(0.0, 0.0),
                            piston=0.0),
             CF.FieldGrid(shape=(64, 64), dx=dx, dy=dx, origin=(0.0, 0.0))),
            dict(on_nyquist='warn', on_window='warn'), None),
        'system.tilted_method': (
            SY.propagate_through_system, (env,),
            dict(elements=[{'type': 'propagate', 'z': 0.02, 'tilt_x': 0.05,
                            'method': 'fresnel'}], wavelength=wl, dx=dx),
            'method'),
        'system.fresnel_leg': (
            SY.propagate_through_system, (env,),
            dict(elements=[{'type': 'propagate', 'z': 0.5,
                            'method': 'fresnel'}], wavelength=wl, dx=dx),
            None),
        'system.fresnel_leg_long': (
            SY.propagate_through_system, (_env(n=128, dx=2e-6, w=8e-6),),
            dict(elements=[{'type': 'propagate', 'z': 0.05,
                            'method': 'fresnel'}], wavelength=wl, dx=2e-6),
            None),
        'system.prescription_doe': (
            SY._prescription_to_elements,
            ([{'type': 'doe', 'phase': None}],), {}, None),
    }


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'v5_two_caller.json'
    res = {'lumenairy_file': la.__file__, 'version': la.__version__,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'probe_file': _ME, 'package_root': _PKG, 'cases': {}}
    for name, (fn, a, kw, needle) in _cases().items():
        entry = {'arms': {}}
        for nest, label in ((0, 'direct'), (1, 'deeper1'), (2, 'deeper2')):
            try:
                _out, err, rows = _run(fn, a, dict(kw), nest)
            except Exception as exc:
                entry['arms'][label] = {
                    'fatal': type(exc).__name__ + ': ' + str(exc)[:120]}
                continue
            sel = [r for r in rows if needle is None or needle in r['msg']]
            entry['arms'][label] = {
                'err': err, 'n_all': len(rows), 'n_selected': len(sel),
                'misattributed': sum(1 for r in sel if r['in_package']),
                'rows': sel[:6]}
        res['cases'][name] = entry
        print('%-40s ' % name + '  '.join(
            '%s n=%s bad=%s' % (k, v.get('n_selected', '-'),
                                v.get('misattributed', '-'))
            for k, v in entry['arms'].items()), flush=True)
    sel = sum(v.get('n_selected', 0) for c in res['cases'].values()
              for v in c['arms'].values())
    bad = sum(v.get('misattributed', 0) for c in res['cases'].values()
              for v in c['arms'].values())
    res['total_selected'] = sel
    res['total_misattributed'] = bad
    print('TOTAL selected %d, misattributed %d' % (sel, bad))
    print('lumenairy:', la.__file__)
    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
