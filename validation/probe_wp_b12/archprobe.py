"""WP-B12 -- the archive-to-archive byte-identity proof.

Run INSIDE an extracted ``git archive <commit> lumenairy`` tree, in a CHILD
process whose ``cwd`` and ``PYTHONPATH`` are that tree, with
``lumenairy.__file__`` asserted to live under it.  Never through pytest, never
against the shared working tree (WP-B7b section 0's rule).

Prints one JSON object: SHA-256 of every FGA field it computes.  The driver
(``archprobe_run.sh``) runs it in the parent tree and in the head tree and
diffs the two:

* the FLAT-last-surface prescription must give the SAME digest in both;
* the curved ones must differ.

Usage: ``python archprobe.py <out.json>``
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings

import numpy as np

_LAM_A = 0.633e-6
_LAM_B = 1.064e-6


def _sha(a):
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def _grid(n, dx, w0):
    xs = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(xs, xs)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


def _singlet(R1, R2, t, glass, semi, name):
    return {'name': name, 'aperture_diameter': 2 * semi,
            'surfaces': [
                {'radius': R1, 'conic': 0.0, 'thickness': t,
                 'glass_before': 'air', 'glass_after': glass,
                 'semi_diameter': semi},
                {'radius': R2, 'conic': 0.0, 'thickness': 0.0,
                 'glass_before': glass, 'glass_after': 'air',
                 'semi_diameter': semi}],
            'thicknesses': [t], 'stop_index': 0}


CASES = {
    # name: (prescription, wavelength, N, dx, w0, output_plane_distance)
    'flat_last_vertex': (
        _singlet(1.45e-3, float('inf'), 0.55e-3, 'N-LASF9', 0.262e-3, 'flat'),
        0.850e-6, 256, 2.0e-6, 190e-6, 0.0),
    'flat_last_focus': (
        _singlet(1.45e-3, float('inf'), 0.55e-3, 'N-LASF9', 0.262e-3, 'flat'),
        0.850e-6, 256, 2.0e-6, 190e-6, 1.43034e-3),
    'curved_b7b_focus': (
        _singlet(1.6e-3, -1.6e-3, 0.60e-3, 'N-SF11', 0.15e-3, 'b7b'),
        _LAM_A, 192, 1.8e-6, 80e-6, 0.92595e-3),
    'curved_baf10_vertex': (
        _singlet(2.10e-3, -2.10e-3, 0.70e-3, 'N-BAF10', 0.20e-3, 'baf10'),
        _LAM_B, 192, 2.55e-6, 105e-6, 0.0),
}


def main(out_path):
    import lumenairy
    root = os.path.abspath(os.getcwd())
    f = os.path.abspath(lumenairy.__file__)
    assert f.startswith(root), (f, root)
    import lumenairy as la
    res = {'tree': root, 'lumenairy_file': f, 'version': la.__version__,
           'python': sys.version.split()[0], 'digests': {}}
    for name, (presc, lam, n, dx, w0, opd) in CASES.items():
        E = _grid(n, dx, w0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fld = la.apply_real_lens_fga(E, prescription=presc, wavelength=lam,
                                         dx=dx, output_plane_distance=opd)
            zone = la.propagators.fga._caustic_zone(E, dx, presc, lam)
        res['digests'][name] = _sha(fld)
        res['digests'][name + '__caustic_zone'] = (
            None if zone is None else [float(zone[0]), float(zone[1])])
    with open(out_path, 'w', encoding='cp1252', errors='replace') as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
