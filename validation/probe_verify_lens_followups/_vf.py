"""Fixtures for the INDEPENDENT verification of the seven 5.44.0 lens
follow-ups (D1-D7).

Deliberately independent of BOTH the shipped tests and of
``validation/probe_verify_lens_5440/_fix.py`` / ``probe_fix_lens_5440``:
a different glass pair (N-BAF10 / N-LAK22 rather than N-SF11 / N-BK7), a
different wavelength (1.55 um rather than 1.064 / 1.31 um), a different grid
pitch, a different f/#, its own beams.  Nothing here touches an API the
change introduced, so every script runs unchanged on v5.44.0 and on the
follow-up branch.

Every script banners ``lumenairy.__file__`` and REFUSES to run if the import
did not come from the tree named on the command line (``--tree``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings

import numpy as np

WL = 1.55e-6


# ---------------------------------------------------------------------------
# arm identity
# ---------------------------------------------------------------------------
def banner(expect_tree=None):
    """Import lumenairy, print its provenance, and REFUSE a mismatched arm."""
    import lumenairy as la
    f = os.path.abspath(la.__file__)
    print(f"# lumenairy.__file__    = {f}", flush=True)
    print(f"# lumenairy.__version__ = {la.__version__}", flush=True)
    print(f"# python {sys.version.split()[0]}  numpy {np.__version__}",
          flush=True)
    print(f"# OMP={os.environ.get('OMP_NUM_THREADS')} "
          f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')} "
          f"MKL={os.environ.get('MKL_NUM_THREADS')}", flush=True)
    if expect_tree:
        want = os.path.abspath(expect_tree).replace('\\', '/').rstrip('/')
        got = f.replace('\\', '/')
        if not got.lower().startswith(want.lower() + '/'):
            raise SystemExit(f"ARM MISMATCH: imported {got}, expected a tree "
                             f"under {want}")
    return la


def argp(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument('out', help='output JSON path')
    p.add_argument('--tree', default=None, help='tree the import MUST come '
                                                'from')
    p.add_argument('--tag', default='', help='free-form arm label')
    return p


def dump(args, payload):
    import lumenairy as la
    payload = dict(payload)
    payload['_arm'] = {
        'lumenairy_file': os.path.abspath(la.__file__).replace('\\', '/'),
        'lumenairy_version': la.__version__,
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'tag': args.tag,
        'platform': sys.platform,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='cp1252', errors='replace') as fh:
        json.dump(payload, fh, indent=1, sort_keys=True, default=_default)
    print(f"# wrote {args.out}", flush=True)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(repr(type(o)))


def free_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().available / 2 ** 30, 2)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------
def h(a):
    """sha256 of the exact bytes (24 hex chars)."""
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(a.tobytes()).hexdigest()[:24]


def field_record(E):
    """Everything about a returned field that a bit-identity claim needs."""
    E = np.asarray(E)
    fin = np.isfinite(E)
    return {
        'hash': h(E),
        'dtype': str(E.dtype),
        'shape': list(E.shape),
        'sum_abs2': float(np.sum(np.abs(E[fin]) ** 2)),
        'max_abs': float(np.max(np.abs(E[fin])) if fin.any() else np.nan),
        'n_nonfinite': int((~fin).sum()),
    }


def rec_record(rec):
    """The diagnostic dict, JSON-clean and hashed where it is an array."""
    out = {}
    for k in sorted(rec):
        v = rec[k]
        if isinstance(v, np.ndarray):
            out[k] = {'hash': h(v), 'dtype': str(v.dtype),
                      'shape': list(v.shape),
                      'values': [float(z) for z in np.ravel(v)[:8]]}
        elif isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        elif isinstance(v, (float, np.floating)):
            out[k] = float(v)
        elif v is None:
            out[k] = None
        elif isinstance(v, tuple):
            out[k] = 'tuple:' + repr([np.asarray(z).tolist()
                                      if isinstance(z, np.ndarray) else z
                                      for z in v])[:200]
        else:
            out[k] = repr(v)[:200]
    return out


# ---------------------------------------------------------------------------
# prescriptions -- MY OWN
# ---------------------------------------------------------------------------
def _s(radius, gb, ga, ca=None):
    d = {'radius': radius, 'glass_before': gb, 'glass_after': ga,
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None}
    if ca is not None:
        d['clear_aperture'] = ca
    return d


def presc_meniscus(ap=6.0e-3):
    """Positive MENISCUS in N-BAF10 -- neither of the two biconvex fixtures
    the builder's probes use, and a different glass."""
    return {'name': 'vfollow_meniscus', 'aperture_diameter': ap,
            'surfaces': [_s(0.0180, 'air', 'N-BAF10'),
                         _s(0.0620, 'N-BAF10', 'air')],
            'thicknesses': [3.1e-3]}


def presc_fast(ap=6.0e-3):
    """A FAST biconvex N-LAK22 -- spherical aberration heavy enough to fold
    the ray map (the caustic / coarse-Newton fixture)."""
    return {'name': 'vfollow_fast', 'aperture_diameter': ap,
            'surfaces': [_s(0.0072, 'air', 'N-LAK22'),
                         _s(-0.0110, 'N-LAK22', 'air')],
            'thicknesses': [4.4e-3]}


def presc_doublet(ap=6.0e-3):
    """A cemented DOUBLET -- three surfaces, two glasses, nothing else here
    has one."""
    return {'name': 'vfollow_doublet', 'aperture_diameter': ap,
            'surfaces': [_s(0.0262, 'air', 'N-LAK22'),
                         _s(-0.0195, 'N-LAK22', 'N-SF6'),
                         _s(-0.0910, 'N-SF6', 'air')],
            'thicknesses': [3.4e-3, 1.6e-3]}


# ---------------------------------------------------------------------------
# beams
# ---------------------------------------------------------------------------
def gauss(n, dx, w, x0=0.0, y0=0.0, dtype=np.complex128):
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    return np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / w ** 2).astype(dtype)


def sph(n, dx, w, R, x0=0.0, y0=0.0, dtype=np.complex128):
    """Gaussian carrying a real spherical carrier of radius ``R``."""
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    r2 = (X - x0) ** 2 + (Y - y0) ** 2
    k = 2 * np.pi / WL
    return (np.exp(-r2 / w ** 2) * np.exp(1j * k * r2 / (2.0 * R))
            ).astype(dtype)


def speckled(n, dx, w, seed=7, dtype=np.complex128):
    """A structured (non-smooth) beam -- amplitude ripple + phase ripple, so
    a hash cannot be satisfied by a smooth-field coincidence."""
    rng = np.random.default_rng(seed)
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    r2 = X ** 2 + Y ** 2
    amp = np.exp(-r2 / w ** 2) * (1.0 + 0.08 * rng.standard_normal((n, n)))
    ph = 0.35 * rng.standard_normal((n, n))
    return (amp * np.exp(1j * ph)).astype(dtype)


# ---------------------------------------------------------------------------
# one traced call, cache-cold, with every notice captured
# ---------------------------------------------------------------------------
def run_traced(la, E, kw, rows, imap_out=True, probe_rc=None):
    """Returns (field, record, warning list [(msg70, filename, lineno)])."""
    from lumenairy.elements import _lens_imap as IM
    IM.inverse_map_cache_clear()
    rec = {} if imap_out else None
    if rec is not None and probe_rc is not None:
        rec['probe_rc'] = probe_rc
    kw2 = dict(kw)
    if rows is not None:
        kw2['sag_chunk_rows'] = rows
    if rec is not None:
        kw2['_imap_out'] = rec
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = la.apply_real_lens_traced(E, **kw2)
    IM.inverse_map_cache_clear()
    wl = sorted((str(w.message)[:70], os.path.basename(str(w.filename)),
                 int(w.lineno)) for w in caught)
    return np.asarray(out), (rec if rec is not None else {}), wl
