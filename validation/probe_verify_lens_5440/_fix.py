"""Shared fixtures for the 5.44.0 lens/carrier verification probes.

Deliberately independent of tests/unit/test_banded_ray_density_and_inverse_map.py:
a different singlet, a different wavelength, a different grid, its own carrier,
its own decentred and caustic-bearing variants.  Runs unchanged on v5.43.0 and
on 50824e9 (nothing here touches an API the change introduced).
"""
from __future__ import annotations

import hashlib
import os
import sys
import warnings

import numpy as np

WL = 1.064e-6


def banner():
    import lumenairy as la
    print(f"# lumenairy.__file__ = {la.__file__}", flush=True)
    print(f"# lumenairy.__version__ = {la.__version__}", flush=True)
    print(f"# numpy {np.__version__}  python {sys.version.split()[0]}",
          flush=True)
    print(f"# OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')} "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    return la


def _surf(radius, gb, ga, ca=None):
    d = {'radius': radius, 'glass_before': gb, 'glass_after': ga,
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None}
    if ca is not None:
        d['clear_aperture'] = ca
    return d


def presc_singlet(ap=11e-3):
    """Biconvex N-SF11 singlet -- a different glass and a different f/# from
    every fixture in the shipped tests."""
    return {'name': 'verify_singlet', 'aperture_diameter': ap,
            'surfaces': [_surf(0.0255, 'air', 'N-SF11'),
                         _surf(-0.0410, 'N-SF11', 'air')],
            'thicknesses': [2.6e-3]}


def presc_strong(ap=11e-3):
    """A deliberately STRONG / high-NA biconvex singlet: spherical aberration
    large enough to fold the ray map (the caustic-bearing fixture)."""
    return {'name': 'verify_strong', 'aperture_diameter': ap,
            'surfaces': [_surf(0.0090, 'air', 'N-SF11'),
                         _surf(-0.0090, 'N-SF11', 'air')],
            'thicknesses': [5.0e-3]}


def gauss(n, dx, w, x0=0.0, y0=0.0, dtype=np.complex128):
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    return np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / w ** 2).astype(dtype)


def h(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(a.tobytes()).hexdigest()[:24]


def run(la, E, kw, rows, imap_out=True):
    """One traced call with the inverse-map cache cold; returns
    (field, record, sorted warning message prefixes)."""
    from lumenairy.elements import _lens_imap as IM
    IM.inverse_map_cache_clear()
    rec = {} if imap_out else None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        kw2 = dict(kw)
        if rows is not None:
            kw2['sag_chunk_rows'] = rows
        if rec is not None:
            kw2['_imap_out'] = rec
        out = la.apply_real_lens_traced(E, **kw2)
    IM.inverse_map_cache_clear()
    msgs = sorted(str(w.message)[:70] for w in caught)
    return np.asarray(out), (rec or {}), msgs
