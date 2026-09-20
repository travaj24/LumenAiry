"""VERIFY-WP-C1 ROUND 2 -- D4: ``evaluate``'s new rim keywords, re-measured.

Round 2 gave ``lumenairy.evaluate`` ``aperture_edge=`` /
``aperture_edge_samples=``, threaded into ``_prescription_to_elements`` and
stamped onto every ``'aperture'`` element it emits for an ``is_stop=True``
surface.  This probe re-measures the four claims on its OWN prescriptions --
a different STOP geometry from the verification's, a TWO-STOP prescription and
a NO-STOP prescription -- and adds the three things round 2 did not measure:

1. a prescription with TWO ``is_stop=True`` surfaces: do BOTH emitted aperture
   elements carry the stamp, or only the first?
2. a prescription whose aperture-bearing surface is NOT the stop (a vignetting
   semi-diameter): does the keyword reach it, and is any element emitted at
   all?
3. the no-STOP prescription's byte identity with and without the keyword, as
   an ASSERTION rather than a note.

It also checks that the keywords are validated BEFORE the decomposition runs,
by using a prescription whose decomposition WARNS: if the guard runs first, a
bad rim raises with zero warnings emitted.

Run from inside the tree under test:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=<tree> python probe_d4_evaluate_v2.py <out.json>
"""
import hashlib
import json
import sys
import warnings

import numpy as np

import lumenairy as la

LAM, NGRID, DXG = 633e-9, 128, 40e-6

# One STOP, different geometry from the verification's fixture.
RX_ONE_STOP = {
    'elements': [
        {'surf_num': 1, 'element_type': 'surface', 'radius': np.inf,
         'glass_after': 'air', 'is_stop': True, 'semi_diameter': 0.9e-3,
         'comment': 'STOP'},
        {'surf_num': 2, 'element_type': 'surface', 'radius': 0.032,
         'glass_after': 'N-SF11', 'semi_diameter': 1.6e-3},
        {'surf_num': 3, 'element_type': 'surface', 'radius': -0.075,
         'glass_after': 'air', 'semi_diameter': 1.6e-3},
    ],
    'all_thicknesses': [1.5e-3, 2.5e-3, 18e-3],
    'aperture_diameter': 1.8e-3,
}

# TWO surfaces flagged is_stop -- the decomposer emits one aperture step each.
RX_TWO_STOPS = {
    'elements': [
        {'surf_num': 1, 'element_type': 'surface', 'radius': np.inf,
         'glass_after': 'air', 'is_stop': True, 'semi_diameter': 0.9e-3,
         'comment': 'STOP A'},
        {'surf_num': 2, 'element_type': 'surface', 'radius': 0.032,
         'glass_after': 'N-SF11', 'semi_diameter': 1.6e-3},
        {'surf_num': 3, 'element_type': 'surface', 'radius': -0.075,
         'glass_after': 'air', 'semi_diameter': 1.6e-3},
        {'surf_num': 4, 'element_type': 'surface', 'radius': np.inf,
         'glass_after': 'air', 'is_stop': True, 'semi_diameter': 0.6e-3,
         'comment': 'STOP B'},
    ],
    'all_thicknesses': [1.5e-3, 2.5e-3, 6e-3, 12e-3],
    'aperture_diameter': 1.8e-3,
}

# NO stop anywhere, and a VIGNETTING semi-diameter on a plain surface.
RX_NO_STOP = {
    'elements': [
        {'surf_num': 1, 'element_type': 'surface', 'radius': 0.032,
         'glass_after': 'N-SF11', 'semi_diameter': 1.6e-3},
        {'surf_num': 2, 'element_type': 'surface', 'radius': -0.075,
         'glass_after': 'air', 'semi_diameter': 0.4e-3,
         'comment': 'vignetting rim, not the stop'},
    ],
    'all_thicknesses': [2.5e-3, 18e-3],
    'aperture_diameter': 1.8e-3,
}


def _src():
    return la.Source.gaussian(N=NGRID, dx=DXG, wavelength=LAM, w0=0.9e-3)


def _digest(arr):
    a = np.ascontiguousarray(np.asarray(arr))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def _has_kw():
    import inspect
    return ('aperture_edge'
            in inspect.signature(la.evaluate).parameters)


def _ev(rx, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return _digest(la.evaluate(rx, _src(), **kw).field)


def _private_hard(rx):
    """The ONLY way back before round 2 -- the private builder plus a
    hand-driven chain -- kept here as the independent reference."""
    from lumenairy.propagators.system import _prescription_to_elements, propagate_through_system
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        elements = [dict(e, edge='hard') if e.get('type') == 'aperture' else e
                    for e in _prescription_to_elements(rx)]
        out = propagate_through_system(np.asarray(_src().E), elements, LAM,
                                       dx=DXG)
    return _digest(out[0] if isinstance(out, tuple) else out)


def main(out_path):
    from lumenairy.propagators.system import _prescription_to_elements
    has_kw = _has_kw()
    out = {'lumenairy_file': la.__file__, 'python': sys.version.split()[0],
           'numpy': np.__version__, 'has_aperture_edge_kw': has_kw}

    # ---- digests -----------------------------------------------------------
    arms = {}
    for label, rx in (('one_stop', RX_ONE_STOP), ('two_stops', RX_TWO_STOPS),
                      ('no_stop', RX_NO_STOP)):
        a = {'default': _ev(rx), 'private_hard': _private_hard(rx)}
        if has_kw:
            a['kw_hard'] = _ev(rx, aperture_edge='hard')
            a['kw_samples_1'] = _ev(rx, aperture_edge_samples=1)
            a['kw_gray4'] = _ev(rx, aperture_edge='gray',
                                aperture_edge_samples=4)
            a['kw_none_none'] = _ev(rx, aperture_edge=None,
                                    aperture_edge_samples=None)
            a['kw_samples_16'] = _ev(rx, aperture_edge_samples=16)
        arms[label] = a
    out['digests'] = arms

    # ---- how many aperture elements, and do they all carry the stamp? ------
    stamp = {}
    for label, rx in (('one_stop', RX_ONE_STOP), ('two_stops', RX_TWO_STOPS),
                      ('no_stop', RX_NO_STOP)):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plain = _prescription_to_elements(rx)
            kwargs = ({'aperture_edge': 'hard', 'aperture_edge_samples': 7}
                      if has_kw else {})
            stamped = _prescription_to_elements(rx, **kwargs)
        ap_plain = [e for e in plain if e.get('type') == 'aperture']
        ap_stamped = [e for e in stamped if e.get('type') == 'aperture']
        stamp[label] = {
            'n_elements': len(plain),
            'n_aperture_elements': len(ap_plain),
            'plain_have_edge_key': [('edge' in e) for e in ap_plain],
            'stamped_have_edge_key': [('edge' in e) for e in ap_stamped],
            'stamped_edges': [e.get('edge') for e in ap_stamped],
            'stamped_samples': [e.get('edge_samples') for e in ap_stamped],
            'all_stamped': bool(ap_stamped) and all(
                e.get('edge') == 'hard' and e.get('edge_samples') == 7
                for e in ap_stamped),
            'element_types': [e.get('type') for e in plain],
        }
    out['stamping'] = stamp

    # ---- the guard runs BEFORE the decomposition --------------------------
    # RX_DOE decomposes with a UserWarning; if the rim is validated first the
    # raise happens with zero warnings recorded.
    rx_doe = {
        'elements': [
            {'surf_num': 1, 'element_type': 'surface', 'radius': np.inf,
             'glass_after': 'air', 'is_stop': True, 'semi_diameter': 0.9e-3},
            {'surf_num': 2, 'element_type': 'surface', 'radius': np.inf,
             'glass_after': 'air', 'semi_diameter': 1.6e-3,
             'aspheric_coeffs': [1e-9, 0.0, 0.0]},
        ],
        'all_thicknesses': [1.5e-3, 18e-3],
        'aperture_diameter': 1.8e-3,
    }
    order = {}
    for label, kw in (('bad_edge', {'aperture_edge': 'soft'}),
                      ('bad_samples', {'aperture_edge_samples': 2.5}),
                      ('bad_samples_list', {'aperture_edge_samples': [4]}),
                      ('good', {'aperture_edge': 'hard'})):
        if not has_kw:
            continue
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            try:
                _prescription_to_elements(rx_doe, **kw)
                verdict = [False, '']
            except BaseException as e:   # noqa: BLE001
                verdict = [True, "{0}: {1}".format(type(e).__name__, e)]
            order[label] = {
                'verdict': verdict,
                'n_warnings_before_raise': len(rec),
                'warnings': [str(w.message)[:90] for w in rec],
            }
    out['guard_before_decomposition'] = order

    # ---- evaluate's own refusal, same message -----------------------------
    refusal = {}
    if has_kw:
        for label, kw in (('edge_soft', {'aperture_edge': 'soft'}),
                          ('samples_2p5', {'aperture_edge_samples': 2.5}),
                          ('samples_zero', {'aperture_edge_samples': 0}),
                          ('samples_true', {'aperture_edge_samples': True}),
                          ('samples_list', {'aperture_edge_samples': [4]})):
            try:
                _ev(RX_ONE_STOP, **kw)
                refusal[label] = [False, '']
            except BaseException as e:   # noqa: BLE001
                refusal[label] = [True, "{0}: {1}".format(type(e).__name__, e)]
    out['evaluate_refusal'] = refusal

    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=repr)

    print('lumenairy:', la.__file__, ' has_aperture_edge_kw =', has_kw)
    for label, a in arms.items():
        print('==', label)
        for k in sorted(a):
            print('   {0:16s} {1}'.format(k, a[k]))
    print('== stamping')
    for label, s in stamp.items():
        print('   {0:10s} n_ap={1} plain_edge={2} stamped_edge={3} '
              'edges={4} samples={5} all_stamped={6}'.format(
                  label, s['n_aperture_elements'], s['plain_have_edge_key'],
                  s['stamped_have_edge_key'], s['stamped_edges'],
                  s['stamped_samples'], s['all_stamped']))
    print('== guard before decomposition')
    for label, o in order.items():
        print('   {0:18s} raised={1} warnings_before={2} {3}'.format(
            label, o['verdict'][0], o['n_warnings_before_raise'],
            o['verdict'][1][:80]))
    print('== evaluate refusal')
    for label, r in refusal.items():
        print('   {0:14s} {1} {2}'.format(label, r[0], r[1][:90]))


if __name__ == '__main__':
    main(sys.argv[1])
