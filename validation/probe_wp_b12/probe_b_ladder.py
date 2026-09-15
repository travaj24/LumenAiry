"""WP-B12 probe B -- the oracle ladder: FGA against an independent
Rayleigh-Sommerfeld oracle, at the exit vertex and at the traced best focus,
before and after the exit-vertex projection.

The "after" arms are produced HERE by wrapping the two differential-transfer
primitives (a probe-local projection written from the probe's own sag
formulae, importing nothing of the proposed library edit), so the ladder can
be measured BEFORE any library code is touched and re-measured after it.

Arms
----
``shipped``          the tree as it stands
``proj_state``       base-ray state projected to the exit-vertex plane
``proj_state_jac``   state AND the 4x4 Jacobian projected (the exact
                     composite transfer to the vertex PLANE)
``native``           no wrapper: what the tree itself does.  On a fixed tree
                     ``native == shipped``; after the repair lands it is the
                     arm that must match ``proj_state_jac``.

Run with OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b12_common import (  # noqa: E402
    FIXTURES,
    _dsag,
    _sag,
    assert_tree,
    build_tag,
    dump,
    fidelity,
)

ROOT = os.environ.get('B12_TREE', os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
ONLY = [k for k in os.environ.get('B12_ONLY', '').split(',') if k]


# ---------------------------------------------------------------------------
# The probe's own projection (independent of the library edit under test).
# ---------------------------------------------------------------------------
def _project(dt, last, wavelength, with_jacobian):
    from lumenairy.raytrace.differential import DifferentialTransfer
    R = float(getattr(last, 'radius', np.inf))
    k = float(getattr(last, 'conic', 0.0) or 0.0)
    asph = getattr(last, 'aspheric_coeffs', None)
    x = np.asarray(dt.x, float)
    y = np.asarray(dt.y, float)
    ux = np.asarray(dt.ux, float)
    uy = np.asarray(dt.uy, float)
    h = np.sqrt(x * x + y * y)
    s = _sag(h, R, k, asph)
    ds = _dsag(h, R, k, asph)
    with np.errstate(invalid='ignore', divide='ignore'):
        sx = np.where(h > 0, ds * x / np.where(h > 0, h, 1.0), 0.0)
        sy = np.where(h > 0, ds * y / np.where(h > 0, h, 1.0), 0.0)
    sec = np.sqrt(1.0 + ux * ux + uy * uy)
    J = np.asarray(dt.jacobian, float)
    if with_jacobian and J.ndim == 3:
        P = np.tile(np.eye(4), (x.size, 1, 1))
        P[:, 0, 0] = 1.0 - sx * ux
        P[:, 0, 1] = -sy * ux
        P[:, 0, 2] = -s
        P[:, 1, 0] = -sx * uy
        P[:, 1, 1] = 1.0 - sy * uy
        P[:, 1, 3] = -s
        J = P @ J
    return DifferentialTransfer(
        jacobian=J, x=x - s * ux, y=y - s * uy, ux=ux, uy=uy,
        opd=np.asarray(dt.opd, float) - s * sec, alive=dt.alive)


def _wrap(base, with_jacobian):
    def wrapped(x, y, ux, uy, surfaces, wavelength, **kw):
        # Force the LAST-SURFACE reference on the base call so the arm always
        # measures the probe's own projection exactly once -- on a repaired
        # tree the caller asks for reference='exit_vertex' and the base would
        # otherwise project too.
        per_surface = kw.get('per_surface', False)
        kw = {k: v for k, v in kw.items() if k != 'reference'}
        try:
            dt = base(x, y, ux, uy, surfaces, wavelength,
                      reference='surface', **kw)
        except TypeError:
            dt = base(x, y, ux, uy, surfaces, wavelength, **kw)
        if per_surface:
            return dt
        return _project(dt, surfaces[-1], wavelength, with_jacobian)
    wrapped.__name__ = f'{base.__name__}_projected'
    return wrapped


def _pin_surface(base):
    """Force the LAST-SURFACE reference and project nothing -- exactly what
    the tree did before WP-B12, reproducible on a repaired tree."""
    def wrapped(*a, **kw):
        kw['reference'] = 'surface'
        return base(*a, **kw)
    wrapped.__name__ = f'{base.__name__}_surface'
    return wrapped


class Arm:
    """Context manager that installs a projecting wrapper on both primitives."""

    def __init__(self, mode):
        self.mode = mode

    def __enter__(self):
        from lumenairy.raytrace import differential as D
        self._saved = (D.ray_transfer_jacobian, D.ray_transfer_jacobian_analytic)
        if self.mode in ('proj_state', 'proj_state_jac'):
            wj = self.mode == 'proj_state_jac'
            D.ray_transfer_jacobian = _wrap(self._saved[0], wj)
            D.ray_transfer_jacobian_analytic = _wrap(self._saved[1], wj)
        elif self.mode == 'pre_b12':
            D.ray_transfer_jacobian = _pin_surface(self._saved[0])
            D.ray_transfer_jacobian_analytic = _pin_surface(self._saved[1])
        return self

    def __exit__(self, *a):
        from lumenairy.raytrace import differential as D
        D.ray_transfer_jacobian, D.ray_transfer_jacobian_analytic = self._saved
        return False


def caustic_zone_oracle(fx, E, n_rays=25):
    """The caustic zone of the SAME estimator ``_caustic_zone`` implements --
    equally-radius-spaced meridional fan over the field's illuminated support,
    5th-95th percentile of the axial crossings -- but traced by the probe's own
    exact conic trace and referenced to the exit-VERTEX plane.

    Independent of ``fga.py`` in the trace and in the reference plane; it
    shares only the estimator's DEFINITION, which is what is being scored.
    """
    from b12_common import trace_meridional
    N = E.shape[-1]
    cx = N // 2
    xgrid = (np.arange(N) - cx) * fx.dx
    row = np.abs(E[E.shape[0] // 2, :])
    xs_h, amp_h = xgrid[cx:], row[cx:]
    good = amp_h > 0.05 * amp_h.max()
    rr = np.linspace(xs_h[good][0], xs_h[good][-1], n_rays)
    rr = rr[rr > 0]
    _xs, _us, _ols, xv, uv, _olv, _n = trace_meridional(
        rr, fx.oracle_surfaces(), fx.lam)
    z = -xv / uv
    z = z[z > 0]
    return float(np.percentile(z, 5)), float(np.percentile(z, 95))


def main():
    assert_tree(ROOT)
    import lumenairy as la
    from lumenairy.propagators.fga import _caustic_zone
    out = {'build': build_tag(), 'version': la.__version__, 'fixtures': {}}
    keys = ONLY or list(FIXTURES)
    for key in keys:
        fx = FIXTURES[key]
        rec = {'note': fx.note}
        p = fx.prescription()
        E = fx.beam()
        zf = fx.best_focus()
        rec['best_focus_m'] = zf
        rec['caustic_zone_oracle_m'] = caustic_zone_oracle(fx, E)
        planes = {'vertex': 0.0, 'focus': zf}
        t0 = time.time()
        oracles = {nm: fx.oracle_field(z) for nm, z in planes.items()}
        rec['oracle_seconds'] = time.time() - t0
        rec['planes'] = {}
        for nm, z in planes.items():
            orc = oracles[nm]
            row = {'z_m': z,
                   'oracle_power': float(np.sum(np.abs(orc) ** 2))}
            for mode in ('native', 'pre_b12', 'proj_state',
                         'proj_state_jac'):
                t1 = time.time()
                with Arm(mode), warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    fld = la.apply_real_lens_fga(
                        E, prescription=p, wavelength=fx.lam, dx=fx.dx,
                        output_plane_distance=z)
                row[mode] = {
                    'fidelity': fidelity(fld, orc),
                    'power_ratio': float(np.sum(np.abs(fld) ** 2)
                                         / np.sum(np.abs(orc) ** 2)),
                    'seconds': time.time() - t1,
                    'sha': _sha(fld),
                }
            rec['planes'][nm] = row
            print(f'  {key:24s} {nm:7s} pre {row["pre_b12"]["fidelity"]:.4f} '
                  f'(P {row["pre_b12"]["power_ratio"]:.3f})  -> native '
                  f'{row["native"]["fidelity"]:.4f} '
                  f'(P {row["native"]["power_ratio"]:.3f})   proj '
                  f'{row["proj_state"]["fidelity"]:.6f}  proj+J '
                  f'{row["proj_state_jac"]["fidelity"]:.6f}  bytes '
                  f'{"IDENTICAL" if row["pre_b12"]["sha"] == row["native"]["sha"] else "changed"}')
        # the caustic zone, three arms
        rec['caustic_zone'] = {}
        for mode in ('native', 'pre_b12', 'proj_state',
                     'proj_state_jac'):
            with Arm(mode), warnings.catch_warnings():
                warnings.simplefilter('ignore')
                cz = _caustic_zone(E, fx.dx, p, fx.lam)
            rec['caustic_zone'][mode] = (None if cz is None
                                         else [float(cz[0]), float(cz[1])])
        print(f'  {key:24s} caustic zone oracle '
              f'{rec["caustic_zone_oracle_m"]}\n'
              f'  {"":24s}   pre_b12 {rec["caustic_zone"]["pre_b12"]}\n'
              f'  {"":24s}   native  {rec["caustic_zone"]["native"]}')
        out['fixtures'][key] = rec
    dump(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      f'probe_b_ladder_{sys.platform}_'
                      f'{sys.version_info.major}{sys.version_info.minor}.json'),
         out)


def _sha(a):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


if __name__ == '__main__':
    main()
