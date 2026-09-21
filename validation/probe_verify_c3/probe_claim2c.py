"""VERIFY-WP-C3 CLAIM 2(c) -- the three route keys are ABSENT on
``transport='sziklas'``, proved by BIT IDENTITY of the whole ``stages`` list
against the PRE-FLIP tree.

    python probe_claim2c.py <tree> <out.json> <arm>

``<arm>`` is ``base`` (pass NO ``transport=``; at 49ddf4bd the default IS
``'sziklas'``) or ``branch`` (pass ``transport='sziklas'`` explicitly).  Run as
a CHILD PROCESS with ``cwd`` and ``PYTHONPATH`` set to its own tree root --
a ``lumenairy/`` in cwd wins over PYTHONPATH, so the anchor is asserted and
printed before anything is measured.

Each record folds, in order: the returned field's raw bytes, the returned
carrier and pitch, the returned object's type name, ``repr(stages)`` AND a
canonical per-key fold of every stage, and every warning in EMISSION order.
"""
from __future__ import annotations

import hashlib
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
_ARM = sys.argv[3]
sys.path.insert(0, _TREE)

import numpy as np                                     # noqa: E402
import vclib as V                                      # noqa: E402

V.anchor(_TREE)

import lumenairy.propagators.carrier as CA             # noqa: E402

WL = 1.31e-6
TKW = dict(on_undersample='silent', on_noncollimated='silent')


def singlet(R1, R2, d, glass, ap, name):
    return {'name': name, 'aperture_diameter': ap, 'thicknesses': [d],
            'surfaces': [
                {'radius': R1, 'glass_before': 'air', 'glass_after': glass,
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
                {'radius': R2, 'glass_before': glass, 'glass_after': 'air',
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]}


def fixture(n=256):
    dx = 60e-6 * 256 / n
    g = V.axis(n, dx)
    e = np.exp(-((g[None, :] ** 2 + g[:, None] ** 2) / 4.5e-3 ** 2)
               ).astype(np.complex128)
    p = singlet(60e-3, -60e-3, 3e-3, 'N-BK7', 14e-3, 'p')
    return e, dx, 60e-3, [{'prescription': p, 'gap_before': 20e-3},
                          {'prescription': p, 'gap_before': 10e-3}]


def _canon(o):
    """A canonical, order-stable string for anything a stage holds."""
    if isinstance(o, dict):
        return '{' + ','.join(f'{k!r}:{_canon(v)}' for k, v in sorted(
            o.items(), key=lambda kv: str(kv[0]))) + '}'
    if isinstance(o, (list, tuple)):
        return type(o).__name__ + '(' + ','.join(_canon(v) for v in o) + ')'
    if isinstance(o, np.ndarray):
        return 'nd' + str(o.shape) + str(o.dtype) + hashlib.sha256(
            np.ascontiguousarray(o).tobytes()).hexdigest()
    if isinstance(o, float):
        return repr(float(o))
    return repr(o)


CONFIGS = {
    'bare_final_leg': dict(final_distance=8e-3),
    'bare_final_leg_zero': dict(final_distance=0.0),
    'readout': dict(final_distance=8e-3,
                    focus_readout=dict(dx_out=0.5e-6, N_out=64)),
    'readout_standoff': dict(final_distance=8e-3,
                             focus_readout=dict(dx_out=0.5e-6, N_out=64,
                                                standoff=2e-3)),
    'readout_containment': dict(
        final_distance=8e-3,
        focus_readout=dict(dx_out=0.5e-6, N_out=64,
                           on_focus_containment='ignore')),
    'readout_fd0': dict(final_distance=0.0,
                        focus_readout=dict(dx_out=0.5e-6, N_out=64)),
    'readout_centre_out': dict(
        final_distance=8e-3,
        focus_readout=dict(dx_out=0.5e-6, N_out=64, centre_out=(2e-6, -1e-6))),
    'readout_N512': dict(final_distance=8e-3, _n=512,
                         focus_readout=dict(dx_out=0.5e-6, N_out=64)),
    'readout_fresnel': dict(final_distance=8e-3, gap_kernel='fresnel',
                            focus_readout=dict(dx_out=0.5e-6, N_out=64)),
}

rec_out = {}
for tag, cfg in CONFIGS.items():
    cfg = dict(cfg)
    n = cfg.pop('_n', 256)
    e, dx, ri, gr = fixture(n)
    kw = dict(r_in=ri, ray_subsample=16, n_workers=1, traced_kwargs=TKW,
              final_leg='paraxial')
    kw.update(cfg)
    if _ARM == 'branch':
        kw['transport'] = 'sziklas'
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try:
            r = CA.propagate_traced_carrier_chain(e, gr, WL, dx, **kw)
            err = None
        except Exception as ex:                         # noqa: BLE001
            r, err = None, f'{type(ex).__name__}: {ex}'
        wmsgs = [f'{wi.category.__name__}: {wi.message}' for wi in w]
    if r is None:
        rec_out[tag] = {'raised': err, 'warnings': wmsgs}
        continue
    fld = np.ascontiguousarray(np.asarray(r.field))
    h = hashlib.sha256()
    h.update(fld.tobytes())
    h.update(str(fld.dtype).encode())
    h.update(str(fld.shape).encode())
    h.update(repr(r.R).encode())
    h.update(repr(r.dx).encode())
    h.update(type(r).__name__.encode())
    h.update(repr(r.stages).encode())
    h.update(_canon(r.stages).encode())
    h.update('||'.join(wmsgs).encode())
    rec_out[tag] = {
        'digest': h.hexdigest(),
        'field_digest': hashlib.sha256(fld.tobytes()).hexdigest(),
        'stages_repr_sha': hashlib.sha256(
            repr(r.stages).encode()).hexdigest(),
        'stages_repr_len': len(repr(r.stages)),
        'stages_canon_sha': hashlib.sha256(_canon(r.stages).encode(
        )).hexdigest(),
        'stage_keys': [sorted(st.keys()) for st in r.stages],
        'n_stages': len(r.stages),
        'R': repr(r.R), 'dx': repr(r.dx),
        'n_warnings': len(wmsgs),
        'warnings_sha': hashlib.sha256('||'.join(wmsgs).encode()).hexdigest(),
    }

V.write_json(sys.argv[2], {'arm': _ARM, 'tree': _TREE, 'records': rec_out})
for k, v in rec_out.items():
    print(f"  {k:24s} {v.get('digest', v.get('raised'))}")
