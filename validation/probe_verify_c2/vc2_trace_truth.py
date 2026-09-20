"""VERIFY-WP-C2 items 2 and 3 -- what the two switches change NUMERICALLY,
measured END TO END against a 60-digit ``decimal`` trace oracle.

Nothing in ``probe_c2_analytic_normal`` answers this: its ladder measures
the DIFFERENCE between the two renormalise modes, which says how far apart
they are but not which one is closer to the truth.  This probe measures both
against an independent oracle.

The oracle replicates the library's GEOMETRY exactly (ray-sphere
intersection, vector Snell with the outward sphere normal, vertex-plane
transfer, ``opd += n * t``) in ``decimal`` at 60 significant digits.  It is
fed the EXACT float64 prescription, and every input ray direction is
``(0, 0, 1)``, which is exactly unit in float64 -- so in exact arithmetic
the refracted directions stay exactly unit at every surface and the oracle's
``t`` and ``opd`` are unambiguous.  (Only an axial launch has an exactly-unit
float64 direction: a dyadic ``(L, N)`` with ``L**2 + N**2 = 1`` exactly forces
``L = 0``.  Height is swept instead, which is the marginal-ray sweep the rim
question needs anyway.)

Three claims are separated:

1. ``|d| - 1`` at every history bundle under each mode (what ``'exit'``
   actually changes);
2. the END-TO-END error of each of the four ``(renormalize, sphere_normal)``
   combinations against the oracle, in position and in OPL;
3. the SENSITIVITY of ``t`` and ``opd`` to a direction-norm drift, measured
   by injecting a drift of ``1e-12`` (nine decades above the noise) and
   scaling -- which turns "the quadratic assumes ``a = |d|**2 = 1``" from an
   argument into a coefficient.

Usage: ``LUMENAIRY_ROOT=<root> python vc2_trace_truth.py OUT.json``
"""
import json
import math
import os
import sys
from decimal import Decimal, getcontext

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)

import numpy as np                                            # noqa: E402
import lumenairy as la                                        # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__

from lumenairy.raytrace.surface import RayBundle, Surface     # noqa: E402
from lumenairy.raytrace.trace import trace                    # noqa: E402
from lumenairy.glass import get_glass_index         # noqa: E402

getcontext().prec = int(os.environ.get('VC2_PREC', '60'))

WL = 587.5618e-9
D = Decimal


# ------------------------------------------------------------------ oracle

def _sqrtD(v):
    return v.sqrt()


def _oracle_trace(surfs, idx, x0, y0, z0):
    """Exact-arithmetic trace of ONE axial ray.  Returns (x, y, z, opd)."""
    px, py, pz = D(x0), D(y0), D(z0)
    dl, dm, dn = D(0), D(0), D(1)
    opd = D(0)
    for i, s in enumerate(surfs):
        n1, n2 = D(idx[i][0]), D(idx[i][1])
        R = D(s.radius)
        # --- intersect the sphere centred at (0, 0, R) ---
        ax, ay, az = px, py, pz - R
        a = dl * dl + dm * dm + dn * dn
        b = 2 * (dl * ax + dm * ay + dn * az)
        c = ax * ax + ay * ay + az * az - R * R
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        sq = _sqrtD(disc)
        t1 = (-b - sq) / (2 * a)
        t2 = (-b + sq) / (2 * a)
        t = t1 if abs(t1) <= abs(t2) else t2
        px += dl * t
        py += dm * t
        pz += dn * t
        opd += n1 * t
        # --- outward sphere normal n = -(P - C)/R ---
        nx = -px / R
        ny = -py / R
        nz = -(pz - R) / R
        # --- vector Snell, normal oriented against the ray ---
        dnn = dl * nx + dm * ny + dn * nz
        if dnn > 0:
            nx, ny, nz = -nx, -ny, -nz
        cos_i = -(dl * nx + dm * ny + dn * nz)
        if s.is_mirror:
            dl = dl + 2 * cos_i * nx
            dm = dm + 2 * cos_i * ny
            dn = dn + 2 * cos_i * nz
        else:
            eta = n1 / n2
            disc_r = 1 - eta * eta * (1 - cos_i * cos_i)
            if disc_r < 0:
                return None
            root = _sqrtD(disc_r)
            k = eta * cos_i - root
            dl = eta * dl + k * nx
            dm = eta * dm + k * ny
            dn = eta * dn + k * nz
        # --- transfer to the next vertex plane ---
        if i < len(surfs) - 1:
            th = D(s.thickness)
            if th != 0:
                tt = (th - pz) / dn
                px += dl * tt
                py += dm * tt
                opd += D(idx[i][1]) * tt
            pz = D(0)
    return px, py, pz, opd


# --------------------------------------------------------- prescriptions

def _s(R, th, gb, ga, sd, mirror=False):
    return Surface(radius=R, conic=0.0, thickness=th, glass_before=gb,
                   glass_after=ga, semi_diameter=sd, is_mirror=mirror)


def doublet():
    return [_s(0.0517, 0.0090, 'air', 'N-BK7', 0.0125),
            _s(-0.0345, 0.0025, 'N-BK7', 'N-SF5', 0.0125),
            _s(-0.1200, 0.0400, 'N-SF5', 'air', 0.0125)]


def stack7():
    out = []
    for j in range(3):
        out.append(_s(0.0800 + 0.004 * j, 0.0055, 'air', 'N-BK7', 0.011))
        out.append(_s(-0.0900 - 0.004 * j, 0.0090, 'N-BK7', 'air', 0.011))
    out.append(_s(0.2500, 0.0300, 'air', 'N-BK7', 0.011))
    return out


def ladder(n_pairs):
    out = []
    for j in range(n_pairs):
        out.append(_s(0.0800 + 0.003 * j, 0.0055, 'air', 'N-BK7', 0.011))
        out.append(_s(-0.0900 - 0.003 * j, 0.0090, 'N-BK7', 'air', 0.011))
    out.append(_s(0.2500, 0.0300, 'air', 'N-BK7', 0.011))
    return out


PRESCRIPTIONS = {'doublet3': doublet(), 'stack7': stack7()}
for _n in (1, 2, 3, 4, 5, 6):
    PRESCRIPTIONS[f'ladder{2 * _n + 1}'] = ladder(_n)


def _bundle(n, hmax):
    h = np.linspace(-hmax, hmax, n)
    z = np.zeros(n)
    return RayBundle(x=h.copy(), y=(0.37 * h).copy(), z=z.copy(),
                     L=np.zeros(n), M=np.zeros(n), N=np.ones(n),
                     wavelength=WL, alive=np.ones(n, dtype=bool),
                     opd=np.zeros(n))


def _indices(surfs):
    return [(get_glass_index(s.glass_before, WL),
             get_glass_index(s.glass_after, WL)) for s in surfs]


# ------------------------------------------------------------------- main

def main(out_path):
    res = {'meta': dict(python=sys.version.split()[0], numpy=np.__version__,
                        lumenairy=la.__version__, file=la.__file__,
                        prec=getcontext().prec)}

    # --- 1 + 2: end-to-end error against the oracle -----------------
    combos = [('surface', 'generic'), ('surface', 'analytic'),
              ('exit', 'generic'), ('exit', 'analytic')]
    endtoend = {}
    for name in ('doublet3', 'stack7', 'ladder13'):
        surfs = PRESCRIPTIONS[name]
        idx = _indices(surfs)
        rb = _bundle(9, 0.0080)
        truth = [_oracle_trace(surfs, idx, float(rb.x[i]), float(rb.y[i]),
                               float(rb.z[i])) for i in range(rb.n_rays)]
        keep = [i for i, t in enumerate(truth) if t is not None]
        per = {}
        for rn, sn in combos:
            out = trace(rb, surfs, WL, output_filter='last',
                        renormalize=rn, sphere_normal=sn)
            ir = out.image_rays
            dx = dy = dop = 0.0
            for i in keep:
                if not bool(ir.alive[i]):
                    continue
                tx, ty, _tz, top = truth[i]
                dx = max(dx, abs(float(ir.x[i]) - float(tx)))
                dy = max(dy, abs(float(ir.y[i]) - float(ty)))
                dop = max(dop, abs(float(ir.opd[i]) - float(top)))
            per[f'{rn}/{sn}'] = dict(dx=dx, dy=dy, dopd=dop,
                                     n_alive=int(np.sum(ir.alive)))
        # which mode is closer, per axis
        endtoend[name] = dict(
            n_surfaces=len(surfs), n_rays_compared=len(keep), per=per,
            renorm_effect_at_generic=dict(
                pos=(per['exit/generic']['dx']
                     - per['surface/generic']['dx']),
                opd=(per['exit/generic']['dopd']
                     - per['surface/generic']['dopd'])),
            renorm_effect_at_analytic=dict(
                pos=(per['exit/analytic']['dx']
                     - per['surface/analytic']['dx']),
                opd=(per['exit/analytic']['dopd']
                     - per['surface/analytic']['dopd'])),
            normal_effect_at_surface=dict(
                pos=(per['surface/analytic']['dx']
                     - per['surface/generic']['dx']),
                opd=(per['surface/analytic']['dopd']
                     - per['surface/generic']['dopd'])),
        )
    res['end_to_end_vs_oracle'] = endtoend

    # --- the |d| - 1 ladder ------------------------------------------
    drift = {}
    for n_pairs in (1, 2, 3, 4, 5, 6):
        name = f'ladder{2 * n_pairs + 1}'
        surfs = PRESCRIPTIONS[name]
        rb = _bundle(4000, 0.0080)
        row = {}
        for rn in ('surface', 'exit'):
            out = trace(rb, surfs, WL, output_filter='all',
                        renormalize=rn, sphere_normal='analytic')
            hist = out.ray_history
            worst_hist = 0.0
            for b in hist[:-1]:
                m = np.sqrt(b.L ** 2 + b.M ** 2 + b.N ** 2)[b.alive]
                if m.size:
                    worst_hist = max(worst_hist, float(np.max(np.abs(m - 1))))
            fb = hist[-1]
            mf = np.sqrt(fb.L ** 2 + fb.M ** 2 + fb.N ** 2)[fb.alive]
            row[rn] = dict(history=worst_hist,
                           final=float(np.max(np.abs(mf - 1))))
        n_s = len(surfs)
        row['n_surfaces'] = n_s
        row['history_over_n_eps'] = (row['exit']['history']
                                     / (n_s * 2.220446049250313e-16))
        drift[name] = row
    res['direction_norm_drift'] = drift

    # --- 3: the sensitivity coefficient ------------------------------
    # Inject a direction-norm drift of DELTA on the bundle entering a
    # spherical stack, and measure how far the answer moves.  The
    # ray-sphere quadratic hard-codes a = |d|**2 = 1 and the OPL leg is
    # n * t (parametric, not geometric), so a norm drift is a FIRST-ORDER
    # error in both.  Sizing it nine decades above the noise makes the
    # coefficient measurable; it is then scaled back.
    surfs = PRESCRIPTIONS['stack7']
    sens = {}
    for delta in (1e-12, 1e-10):
        rb0 = _bundle(200, 0.0080)
        rb1 = _bundle(200, 0.0080)
        rb1.L = rb1.L * (1.0 + delta)
        rb1.M = rb1.M * (1.0 + delta)
        rb1.N = rb1.N * (1.0 + delta)
        a = trace(rb0, surfs, WL, output_filter='last',
                  renormalize='surface', sphere_normal='analytic').image_rays
        b = trace(rb1, surfs, WL, output_filter='last',
                  renormalize='surface', sphere_normal='analytic').image_rays
        m = a.alive & b.alive
        dpos = float(np.max(np.abs(a.x[m] - b.x[m])))
        dopd = float(np.max(np.abs(a.opd[m] - b.opd[m])))
        sens[f'delta={delta:g}'] = dict(
            d_pos=dpos, d_opd=dopd,
            d_pos_per_unit_drift=dpos / delta,
            d_opd_per_unit_drift=dopd / delta)
    # predicted error from the MEASURED drift of the 13-surface ladder
    meas = drift['ladder13']['exit']['history']
    k_pos = sens['delta=1e-12']['d_pos_per_unit_drift']
    k_opd = sens['delta=1e-12']['d_opd_per_unit_drift']
    sens['measured_history_drift_13'] = meas
    sens['predicted_pos_error_from_that_drift'] = k_pos * meas
    sens['predicted_opd_error_from_that_drift'] = k_opd * meas
    sens['linearity_check'] = dict(
        ratio_pos=(sens['delta=1e-10']['d_pos_per_unit_drift']
                   / max(k_pos, 1e-300)),
        ratio_opd=(sens['delta=1e-10']['d_opd_per_unit_drift']
                   / max(k_opd, 1e-300)))
    res['norm_drift_sensitivity'] = sens

    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(json.dumps(res, indent=1, sort_keys=True))
    return res


if __name__ == '__main__':
    main(sys.argv[1])
    assert math.isfinite(1.0)
