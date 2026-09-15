"""Warning-attribution instrument for ``lumenairy/propagators/carrier.py``.

The claim under measurement (handoff 4.4): a warning raised inside the library
must name the USER'S frame -- the nearest frame outside the package -- at every
call depth.  A literal ``stacklevel`` encodes ONE depth, so it is right for one
call path and silently wrong for the others; the failure is invisible in a
normal run because the warning still fires, it just points at library source.

THE INSTRUMENT.  ``warnings.warn`` is wrapped for the duration of each call.
For every warning raised from inside the package the wrapper computes two
things from the live stack:

  * TARGET -- the frame the call's own ``stacklevel`` actually names, counted
    the way ``warnings.warn`` counts it (1 = the frame that calls ``warn``);
  * TRUTH  -- the first frame outside the package directory, which is what
    :func:`lumenairy.elements._lens_kernels.caller_stacklevel` returns and what
    the user means by "where did this come from".

A site is MISATTRIBUTED when the two disagree.  This needs no guess about how
deep any particular entry point is, and it scores every warning that fires.

THE TWO CALLERS.  Every cell is driven twice: once directly from this file and
once through ``propagate_traced_carrier_chain``-style nesting, so the same warn
site is reached at two different library depths.  A literal that is right at
one depth is wrong at the other, and the comparison is what the b11 ratchet's
rationale is about.
"""
import json
import os
import pathlib
import sys
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import lumenairy as la  # noqa: E402
from lumenairy.propagators import carrier as CA  # noqa: E402

PKG = os.path.abspath(os.path.dirname(la.__file__))
ME = pathlib.Path(__file__).name
_REAL_WARN = warnings.warn


def _frame_at(depth):
    """The frame ``warnings.warn`` would attribute to at ``stacklevel=depth``,
    starting from the frame that called our wrapper (depth 1)."""
    # 0=_frame_at, 1=_wrapped_warn, 2=the frame that called warn == depth 1
    f = sys._getframe(2)
    d = 1
    while f is not None and d < depth:
        f = f.f_back
        d += 1
    return f


def _first_outside():
    f = sys._getframe(2)
    d = 1
    last_inside = 1
    while f is not None and d < 64:
        if not os.path.abspath(f.f_code.co_filename).startswith(PKG):
            return d, f
        last_inside, f, d = d, f.f_back, d + 1
    return last_inside, None


def _install(records):
    def _wrapped_warn(message, category=UserWarning, stacklevel=1, **kw):
        here = sys._getframe(1)
        if os.path.abspath(here.f_code.co_filename).startswith(PKG):
            tgt = _frame_at(int(stacklevel))
            truth_depth, truth = _first_outside()
            records.append({
                'warn_site': os.path.basename(here.f_code.co_filename),
                'warn_line': here.f_lineno,
                'stacklevel': int(stacklevel),
                'named_file': (os.path.basename(tgt.f_code.co_filename)
                               if tgt is not None else None),
                'named_line': tgt.f_lineno if tgt is not None else None,
                'truth_file': (os.path.basename(truth.f_code.co_filename)
                               if truth is not None else None),
                'truth_line': truth.f_lineno if truth is not None else None,
                'truth_stacklevel': truth_depth,
                'named_is_library': (
                    tgt is not None
                    and os.path.abspath(
                        tgt.f_code.co_filename).startswith(PKG)),
                'message': str(message)[:90],
            })
        # +1 compensates for THIS wrapper's own frame, so the emitted
        # warning lands exactly where it would without the instrument.
        return _REAL_WARN(message, category,
                          stacklevel=int(stacklevel) + 1, **kw)
    warnings.warn = _wrapped_warn


def _uninstall():
    warnings.warn = _REAL_WARN


def _drive(fn, *a, **kw):
    rec = []
    _install(rec)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('always')
            try:
                fn(*a, **kw)
                err = None
            except Exception as exc:                  # noqa: BLE001
                err = f'{type(exc).__name__}: {exc}'
    finally:
        _uninstall()
    return rec, err


def _wrap1(fn, *a, **kw):
    return fn(*a, **kw)


def _gauss(n, dx, w):
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    return np.exp(-(X ** 2 + Y ** 2) / (w * w)).astype(np.complex128)


def _cells():
    E = _gauss(64, 8e-6, 60e-6)
    base = dict(wavelength=633e-9, dx=8e-6)
    return [
        ('tilt_inert_fresnel', CA.propagate_carrier_referenced,
         (E, -0.05, 5e-3), dict(base, gap_kernel='fresnel', tilt=(0.12, 0.0))),
        ('tilt_inert_sas', CA.propagate_carrier_referenced,
         (E, -0.05, 5e-3), dict(base, gap_kernel='sas', tilt=(0.12, 0.0))),
        ('collins_long_leg', CA.propagate_carrier_referenced,
         (E, -0.02, 0.05), dict(base, transport='collins',
                                on_collins_sampling='warn')),
        ('sziklas_long_leg', CA.propagate_carrier_referenced,
         (E, -0.02, 0.05), dict(base, transport='sziklas',
                                on_collins_sampling='warn')),
        ('coarse_grid', CA.propagate_carrier_referenced,
         (_gauss(32, 40e-6, 120e-6), -0.30, 0.20),
         dict(wavelength=1.55e-6, dx=40e-6)),
        # THE DECISIVE PAIR.  ``carrier_referenced_focus_readout`` calls
        # ``propagate_carrier_referenced`` internally (carrier.py:3987) and
        # forwards ``gap_kernel`` and ``tilt``, so the SAME warn site is
        # reached one LIBRARY frame deeper than the direct call above.  A
        # literal stacklevel cannot be right for both.
        ('tilt_inert_via_focus_readout', CA.carrier_referenced_focus_readout,
         (E, -0.05, 5e-3), dict(base, dx_out=8e-6, N_out=32,
                                gap_kernel='fresnel', tilt=(0.12, 0.0))),
    ]


def main():
    assert 'lum_reds' in la.__file__, la.__file__
    out = {'lumenairy_file': la.__file__, 'version': la.__version__,
           'package_root': PKG, 'probe_file': ME,
           'env': {k: os.environ.get(k) for k in
                   ('OPENBLAS_CORETYPE', 'OPENBLAS_NUM_THREADS')},
           'cells': {}}
    all_rec = []
    for label, fn, a, kw in _cells():
        if not a and not kw:
            continue
        direct, e1 = _drive(fn, *a, **kw)
        wrapped, e2 = _drive(_wrap1, fn, *a, **kw)
        for r in direct:
            r['arm'] = f'{label}:direct'
        for r in wrapped:
            r['arm'] = f'{label}:wrapped'
        all_rec += direct + wrapped
        out['cells'][label] = {
            'error_direct': e1, 'error_wrapped': e2,
            'n_direct': len(direct), 'n_wrapped': len(wrapped),
            'direct': direct, 'wrapped': wrapped,
            'misattributed_direct': [r['warn_line'] for r in direct
                                     if r['named_is_library']],
            'misattributed_wrapped': [r['warn_line'] for r in wrapped
                                      if r['named_is_library']],
        }
        print(label, 'direct', len(direct), 'wrapped', len(wrapped),
              'misattributed',
              len(out['cells'][label]['misattributed_direct'])
              + len(out['cells'][label]['misattributed_wrapped']))

    carrier_rec = [r for r in all_rec if r['warn_site'] == 'carrier.py']
    bad = [r for r in carrier_rec if r['named_is_library']]
    out['carrier_warnings_seen'] = len(carrier_rec)
    out['carrier_sites_seen'] = sorted({r['warn_line'] for r in carrier_rec})
    out['carrier_misattributed'] = bad
    out['carrier_misattributed_lines'] = sorted({r['warn_line'] for r in bad})
    out['all_warnings_seen'] = len(all_rec)
    out['all_misattributed_lines'] = sorted(
        {(r['warn_site'], r['warn_line']) for r in all_rec
         if r['named_is_library']})
    print('carrier.py warnings seen:', len(carrier_rec),
          'at lines', out['carrier_sites_seen'])
    print('carrier.py MISATTRIBUTED lines:', out['carrier_misattributed_lines'])
    for r in bad:
        print('  line', r['warn_line'], 'stacklevel', r['stacklevel'],
              '-> names', r['named_file'] + ':' + str(r['named_line']),
              '| truth', str(r['truth_file']) + ':' + str(r['truth_line']),
              '(stacklevel', r['truth_stacklevel'], ')')
    print('ALL misattributed (file, line):', out['all_misattributed_lines'])
    tag = os.environ.get('PROBE_TAG', 'default')
    dest = pathlib.Path(__file__).parent / f'carrier_attribution_{tag}.json'
    dest.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print('wrote', dest)


if __name__ == '__main__':
    main()
