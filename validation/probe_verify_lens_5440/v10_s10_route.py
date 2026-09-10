"""V10 -- the ONE difference the CHANGELOG names, on the S10 fixture itself.

The 5.44.0 note says a banded call at the shipped default used to select the
incumbent coarse-Newton inversion, "measured 2.19e-02 relative on the niche-S10
carrier fixture".  This reproduces that fixture (the singlet, grid and carrier
of ``test_niche_s10_sibling_patterns.py::
test_row_band_assembly_matches_whole_grid_under_a_carrier``) on BOTH builds and
reports:

  whole   sag_chunk_rows=0    -- the evaluator on both builds
  band    sag_chunk_rows=32   -- coarse-Newton on v5.43.0, evaluator on 5.44.0
  rel     max|band - whole| / max|whole|   (the test file's own _rel)

The cross-build claim -- that the 5.44.0 BANDED field is the v5.43.0
WHOLE-GRID field, not a third answer -- is read off the hashes.

Usage:  python v10_s10_route.py <out.json>
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fix  # noqa: E402

_WL = 1.31e-6


def _singlet(R1, R2, d, glass, ap):
    def s(r, gb, ga):
        return {'radius': r, 'glass_before': gb, 'glass_after': ga,
                'conic': 0.0, 'radius_y': None, 'conic_y': None,
                'aspheric_coeffs': None, 'aspheric_coeffs_y': None}
    return {'name': 's10', 'aperture_diameter': ap, 'thicknesses': [d],
            'surfaces': [s(R1, 'air', glass), s(R2, glass, 'air')]}


def _gauss(N, dx, w, R, sphere=True):
    x = (np.arange(N) - N // 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    k = 2 * np.pi / _WL
    ph = (np.exp(1j * k * (np.sqrt(r2 + R * R) - abs(R)))
          if sphere else np.exp(1j * k * r2 / (2.0 * R)))
    return (np.exp(-r2 / w ** 2) * ph).astype(np.complex128)


def _rel(a, b):
    return float(np.max(np.abs(a - b))
                 / max(float(np.max(np.abs(b))), 1e-300))


def main():
    la = _fix.banner()
    prev = la.get_fft_auto_promote()
    la.set_fft_auto_promote(False)
    presc = _singlet(6.0e-3, -6.0e-3, 1.0e-3, 'N-BK7', 1.6e-3)
    N, dx, conj = 256, 4.0e-6, 30.0e-3
    E = _gauss(N, dx, 250e-6, conj, sphere=True)
    out = {'version': la.__version__, 'arms': {}}
    for tag, carrier in (('carrier', conj), ('carrier_none', None)):
        kw = dict(prescription=presc, wavelength=_WL, dx=dx, ray_subsample=8,
                  carrier=carrier, on_undersample='silent',
                  on_noncollimated='off', on_aperture_beam='silent',
                  parallel_amp=False, n_workers=1)
        fields = {}
        for rows in (0, 32):
            from lumenairy.elements import _lens_imap as IM
            IM.inverse_map_cache_clear()
            rec = {}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                f = np.asarray(la.apply_real_lens_traced(
                    E, sag_chunk_rows=rows, _imap_out=rec, **kw))
            IM.inverse_map_cache_clear()
            fields[rows] = (f, rec)
        (fw, rw), (fb, rb) = fields[0], fields[32]
        r = _rel(fb, fw)
        out['arms'][tag] = {
            'whole_hash': _fix.h(fw), 'band_hash': _fix.h(fb),
            'whole_engaged': bool(rw.get('engaged', False)),
            'band_engaged': bool(rb.get('engaged', False)),
            'band_gate_open': bool(rb.get('gate_open', False)),
            'rel_band_vs_whole': r,
            'bit_equal': bool(np.array_equal(fw, fb))}
        print('%-14s whole=%s(eng=%s) band=%s(eng=%s gate=%s) rel=%.4e '
              'bit_eq=%s' % (tag, _fix.h(fw), out['arms'][tag]['whole_engaged'],
                             _fix.h(fb), out['arms'][tag]['band_engaged'],
                             out['arms'][tag]['band_gate_open'], r,
                             out['arms'][tag]['bit_equal']), flush=True)
    la.set_fft_auto_promote(prev)
    with open(sys.argv[1], 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', sys.argv[1])


if __name__ == '__main__':
    main()
