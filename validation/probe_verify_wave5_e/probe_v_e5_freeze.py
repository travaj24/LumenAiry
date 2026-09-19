"""VERIFY-WAVE5-E / E5 (O-1): my own clipped-fan fixtures, PRE vs POST.

Runs unchanged in the PRE tree (``ede07f30``) and the POST tree.  For each
surface class x backend x aperture over-fill it reports, against the PRODUCTION
bundle operator ``TraceResult.at_exit_vertex()``:

  * ``missed_drift``          how far the differential projection moved rows
                              that never reached the last surface (0 = frozen)
  * ``eq_at_exit_vertex_*``   do the FROZEN rows equal ``at_exit_vertex``'s own
                              frozen state, bit for bit?  (the point of O-1)
  * ``reached_digest``        sha256 of the rows that DID reach, so PRE and
                              POST can be compared byte for byte
  * ``n_companion_only``      rows base-alive but FD-companion-dead (O-4)
"""
import copy
import hashlib
import json
import sys
import warnings

import numpy as np

import lumenairy
from lumenairy import raytrace as rt
from lumenairy.raytrace import surfaces_from_prescription
from lumenairy.raytrace.differential import ray_transfer_jacobian, ray_transfer_jacobian_analytic

_LAM = 1.03e-6
_SEMI = 0.15e-3
_T = 0.55e-3
_GLASS = 'N-SSK8'
_N_RAYS = 121

# MY fixtures -- different radii / conics / tilts from the item-E test file, so
# this is a re-measurement and not a re-run of the builder's cells.
_CLASSES = {
    'conic': dict(last={'radius': -1.05e-3, 'conic': -0.35}, tilt=0.0),
    'asphere': dict(last={'radius': -0.980e-3, 'conic': -0.42,
                          'aspheric_coeffs': {4: 1.4e9, 6: -2.6e16}},
                    tilt=0.0),
    'biconic': dict(last={'radius': -0.930e-3, 'radius_y': -1.31e-3,
                          'conic': 0.0, 'conic_y': 0.0}, tilt=0.0),
    'mirror': dict(last={'radius': -3.1e-3, 'conic': 0.0,
                         'glass_after': 'MIRROR'}, tilt=0.0),
    'oblique': dict(last={'radius': -1.62e-3, 'conic': 0.0}, tilt=8.0),
    'flat': dict(last={'radius': np.inf, 'conic': 0.0}, tilt=0.0),
}
_ANALYTIC = ('conic', 'asphere', 'mirror', 'oblique', 'flat')
_OVERS = (1.6, 2.2, 3.0, 4.0)


def _presc(last, glass_after='air'):
    last = dict(last)
    ga = last.pop('glass_after', glass_after)
    s1 = {'radius': np.inf, 'conic': 0.0, 'thickness': _T,
          'glass_before': 'air', 'glass_after': _GLASS,
          'semi_diameter': _SEMI}
    s2 = {'conic': 0.0, 'thickness': 0.0, 'glass_before': _GLASS,
          'glass_after': ga, 'semi_diameter': _SEMI}
    s2.update(last)
    return {'name': 'v_wave5e', 'aperture_diameter': 2 * _SEMI,
            'surfaces': [s1, s2], 'thicknesses': [_T], 'stop_index': 0}


def _surfs(name):
    s = [copy.copy(x)
         for x in surfaces_from_prescription(_presc(_CLASSES[name]['last']))]
    s[-1].thickness = 0.0
    return s


def _fan(name, over):
    h = np.linspace(-over * _SEMI, over * _SEMI, _N_RAYS)
    z = np.zeros(_N_RAYS)
    ux = np.full(_N_RAYS, np.tan(np.deg2rad(_CLASSES[name]['tilt'])))
    return h, z.copy(), ux, z.copy()


def _arr(v):
    return np.asarray(v, dtype=np.float64)


def _dig(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(_arr(a)).tobytes())
    return h.hexdigest()[:16]


def _bundle(name, over):
    surfs = _surfs(name)
    h, y, ux, uy = _fan(name, over)
    nz = 1.0 / np.sqrt(1.0 + ux ** 2 + uy ** 2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        b = rt.RayBundle(x=h.copy(), y=y.copy(), z=np.zeros(_N_RAYS),
                         L=ux * nz, M=uy * nz, N=nz, wavelength=_LAM,
                         alive=np.ones(_N_RAYS, bool), opd=np.zeros(_N_RAYS))
        res = rt.trace(b, surfs, _LAM)
        return res.image_rays, res.at_exit_vertex()


def _transfers(name, backend, over):
    surfs = _surfs(name)
    h, y, ux, uy = _fan(name, over)
    fn = (ray_transfer_jacobian if backend == 'fd'
          else ray_transfer_jacobian_analytic)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        srf = fn(h.copy(), y.copy(), ux.copy(), uy.copy(), surfs, _LAM,
                 reference='surface')
        vtx = fn(h.copy(), y.copy(), ux.copy(), uy.copy(), surfs, _LAM,
                 reference='exit_vertex')
    return srf, vtx


def _jac_rows(j, mask):
    j = _arr(j)
    if j.ndim == 3:
        return j[mask]
    return j[:, mask]


def main():
    out = dict(lumenairy_file=lumenairy.__file__,
               version=lumenairy.__version__, python=sys.version.split()[0],
               platform=sys.platform, numpy=np.__version__, cells=[])
    for name in sorted(_CLASSES):
        for over in _OVERS:
            img, ev = _bundle(name, over)
            reached = np.asarray(img.alive, bool)
            missed = ~reached
            for backend in ('fd', 'analytic'):
                cell = dict(fixture=name, backend=backend, over=over,
                            n_rays=_N_RAYS, n_reached=int(reached.sum()),
                            n_missed=int(missed.sum()))
                if backend == 'analytic' and name not in _ANALYTIC:
                    cell['skipped'] = 'analytic refuses biconic'
                    out['cells'].append(cell)
                    continue
                try:
                    srf, vtx = _transfers(name, backend, over)
                except NotImplementedError as exc:
                    cell['skipped'] = 'NotImplementedError %s' % str(exc)[:70]
                    out['cells'].append(cell)
                    continue
                talive = np.asarray(srf.alive, bool)
                cell['n_transfer_alive'] = int(talive.sum())
                cell['n_companion_only'] = int((reached & ~talive).sum())
                dmax = {}
                for f in ('x', 'y', 'ux', 'uy', 'opd'):
                    a, b = _arr(getattr(srf, f)), _arr(getattr(vtx, f))
                    dmax[f] = (float(np.nanmax(np.abs(a[missed] - b[missed])))
                               if missed.any() else 0.0)
                if missed.any():
                    ja = _jac_rows(srf.jacobian, missed)
                    jb = _jac_rows(vtx.jacobian, missed)
                    dj = np.abs(ja - jb)
                    dmax['jacobian'] = float(np.nanmax(dj)) if dj.size else 0.0
                    dmax['jacobian_nonfinite_pre'] = int(
                        np.count_nonzero(~np.isfinite(ja)))
                    dmax['jacobian_nonfinite_post'] = int(
                        np.count_nonzero(~np.isfinite(jb)))
                else:
                    dmax['jacobian'] = 0.0
                cell['missed_drift'] = dmax
                cell['missed_frozen'] = bool(
                    max(dmax[k] for k in
                        ('x', 'y', 'ux', 'uy', 'opd', 'jacobian')) == 0.0)
                if missed.any():
                    cell['eq_at_exit_vertex_opd'] = float(np.nanmax(np.abs(
                        _arr(vtx.opd)[missed] - _arr(ev.opd)[missed])))
                    cell['eq_at_exit_vertex_x'] = float(np.nanmax(np.abs(
                        _arr(vtx.x)[missed] - _arr(ev.x)[missed])))
                    cell['eq_at_exit_vertex_bitexact'] = bool(
                        np.array_equal(
                            _arr(vtx.opd)[missed].view(np.uint64),
                            _arr(ev.opd)[missed].view(np.uint64))
                        and np.array_equal(
                            _arr(vtx.x)[missed].view(np.uint64),
                            _arr(ev.x)[missed].view(np.uint64)))
                if reached.any():
                    cell['reached_digest'] = _dig(
                        _arr(vtx.x)[reached], _arr(vtx.y)[reached],
                        _arr(vtx.ux)[reached], _arr(vtx.uy)[reached],
                        _arr(vtx.opd)[reached],
                        _jac_rows(vtx.jacobian, reached))
                    co = reached & ~talive
                    if co.any():
                        cell['companion_only_vs_ev_opd'] = float(np.nanmax(
                            np.abs(_arr(vtx.opd)[co] - _arr(ev.opd)[co])))
                out['cells'].append(cell)
    scored = [c for c in out['cells'] if 'missed_frozen' in c]
    out['summary'] = dict(
        n_cells=len(out['cells']), n_scored=len(scored),
        n_frozen=sum(c['missed_frozen'] for c in scored),
        n_unfrozen=sum(not c['missed_frozen'] for c in scored),
        max_missed_opd_drift=max(
            (c['missed_drift']['opd'] for c in scored), default=0.0),
        max_missed_jac_drift=max(
            (c['missed_drift']['jacobian'] for c in scored), default=0.0),
        n_eq_at_exit_vertex=sum(
            1 for c in scored if c.get('eq_at_exit_vertex_bitexact')),
        n_with_companion_only=sum(
            1 for c in scored if c.get('n_companion_only', 0) > 0),
        total_companion_only=sum(
            c.get('n_companion_only', 0) for c in scored),
        jac_nonfinite_post=sum(
            c['missed_drift'].get('jacobian_nonfinite_post', 0)
            for c in scored),
        jac_nonfinite_pre=sum(
            c['missed_drift'].get('jacobian_nonfinite_pre', 0)
            for c in scored),
    )
    print(json.dumps(out, indent=1))


main()
