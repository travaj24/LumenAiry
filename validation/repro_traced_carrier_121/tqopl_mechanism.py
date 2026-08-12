# TILT-QUADRATIC OPL -- mechanism probe.
#
# PROBE_CHAIN_LADDER_PISTON_2026_08_11 S3.7 measures the chain reproducing a
# CONSTANT 95.2 % of the tilt-quadratic chief-ray optical path at design 121's
# FIRST post-DOE group, invariant to every grid / lattice / element-fit lever.
# This script tests ONE named closed form for the missing 4.8 %:
#
#   ``apply_real_lens_traced`` references its traced OPL grid to the ray
#   launched at the OPTICAL AXIS of the launch lattice
#   (``opl_grid -= opl_grid[i_axis, i_axis]`` with ``xs_in`` axis-centred),
#   and that constant is never re-applied.  The removed constant is
#
#       Lam(0) = W(0, 0) + P(0, 0)
#
#   -- the carrier eikonal at the axis (H6 term, ``final.opd += W(h)``) plus
#   the geometric path of the axis ray through the group.  BOTH depend on the
#   tilt, so the element hands back an exit field whose chief-ray piston is
#   short by ``k0 * [Lam_theta(0) - Lam_0(0)]``, a pure theta^2 term.
#
# Everything here is computed from ray traces and closed forms -- it does NOT
# run the chain.  It reads the MEASURED pistons out of the artifacts
# ``probe_ladder_run_121.py`` already wrote (``PL_NGROUPS=1 PL_TSCALE=s``).
#
# THIS IS A FAIL-BEFORE INSTRUMENT.  It scores the DEFICIT against its
# predicted closed form, so it is only meaningful against artifacts written by
# a library that still HAS the defect -- i.e. ``_probe_ladder/`` (pre-fix).
# Pointed at post-fix artifacts (``PL_OUT=_probe_tqopl``) the deficit column
# collapses to the numerical floor and ``pred/def`` is meaningless, which is
# the pass-after statement, not a failure of this script.  The verdict tables
# for both arms are in
# ``docs/audits/FIX_TILT_QUADRATIC_OPL_2026_08_11.md`` S2 / S4.
#
# Usage:  python tqopl_mechanism.py
import dataclasses
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import _d121_common as d121  # noqa: E402

from lumenairy.elements._lens_traced import TiltedCarrier, _tilted_carrier_parts  # noqa: E402
from lumenairy.raytrace import Surface, make_ray, trace  # noqa: E402
from lumenairy.raytrace.trace import surfaces_from_prescription  # noqa: E402

OUT = os.environ.get('PL_OUT', os.path.join(_HERE, '_probe_ladder'))
K0 = 2.0 * np.pi / d121.LAM


def _load(tag):
    p = os.path.join(OUT, '%s.json' % tag)
    if not os.path.exists(p):
        return None
    with open(p, encoding='cp1252') as fh:
        return json.load(fh)


def _chief_surfs(post, ng):
    """The oracle's surface list: DOE plane -> exit vertex of group ng-1."""
    return d121.post_surfaces(post[:ng], trailing=0.0)


def _group_surfs(presc):
    """The group ALONE, front vertex plane -> back vertex plane -- exactly the
    span ``apply_real_lens_traced`` traces (it launches on the entrance plane
    and corrects every ray to the flat exit vertex plane)."""
    sf = [dataclasses.replace(s, semi_diameter=np.inf)
          for s in surfaces_from_prescription(presc)]
    sf[-1] = dataclasses.replace(sf[-1], thickness=0.0)
    sf.append(Surface(radius=np.inf, conic=0.0, semi_diameter=np.inf,
                      glass_before='air', glass_after='air', is_mirror=False,
                      thickness=0.0, label='_vertex'))
    return sf


def _trace_one(surfs, x, y, L, M):
    im = trace(make_ray(x, y, L, M, wavelength=d121.LAM), surfs,
               d121.LAM, output_filter='last').image_rays
    return (float(np.asarray(im.opd).ravel()[0]),
            float(np.asarray(im.x).ravel()[0]),
            float(np.asarray(im.y).ravel()[0]),
            float(np.asarray(im.L).ravel()[0]),
            float(np.asarray(im.M).ravel()[0]))


def _unwrap_to(x, ref):
    """Lift the wrapped ``x`` onto ``ref``'s branch."""
    return x + 2.0 * np.pi * round((ref - x) / (2.0 * np.pi))


if __name__ == '__main__':
    pre, post, gap_to_doe, period = d121.geometry()
    gap1 = float(post[0]['gap_before'])
    gsurf = _group_surfs(post[0]['prescription'])
    csurf = _chief_surfs(post, 1)

    # R at the DOE plane -- read from any artifact (all share chain A).
    R_doe = None
    for p in sorted(glob.glob(os.path.join(OUT, '*.json'))):
        with open(p, encoding='cp1252') as fh:
            R_doe = float(json.load(fh)['chain_a']['R_doe'])
        break
    print('design 121: DOE->group1 gap %.6f mm   R_doe %.6f mm   '
          'period %.4f um' % (gap1 * 1e3, R_doe * 1e3, period * 1e6))
    print('group 1 surfaces: %d  (%s)'
          % (len(gsurf) - 1,
             ', '.join('%s' % (s.glass_after) for s in gsurf[:-1])))
    print()

    def _lam0(L, M):
        """The constant ``apply_real_lens_traced`` removes and never restores:
        the AXIS launch ray's total entrance-referenced OPL."""
        ob = 1.0 / np.sqrt(1.0 - L * L - M * M)
        x_c, y_c = L * gap1 * ob, M * gap1 * ob
        R_use = R_doe + gap1
        spec = TiltedCarrier(R_use, L, M, x_c, y_c)
        W0, gL, gM = _tilted_carrier_parts(spec, 0.0, 0.0)
        P0 = _trace_one(gsurf, 0.0, 0.0, float(gL), float(gM))[0]
        return float(W0) + P0, float(W0), P0, x_c, y_c

    lam0_ref, W0_ref, P0_ref, _, _ = _lam0(0.0, 0.0)

    # reference run: on-axis order (0,0) at PL_NGROUPS=1
    ref = _load('n1024_dx2_o0_0_g1')
    if ref is None:
        raise SystemExit('missing reference artifact n1024_dx2_o0_0_g1')
    pist_ref = float(ref['aperture']['piston_c'])
    opl_ref = _trace_one(csurf, 0.0, 0.0, 0.0, 0.0)[0]

    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, 'n1024_dx2_o*_g1*.json'))):
        d = json.load(open(p, encoding='cp1252'))
        if int(d.get('n_groups_b', 6)) != 1:
            continue
        L, M = float(d.get('tilt_L', 0.0)), float(d.get('tilt_M', 0.0))
        if L == 0.0 and M == 0.0:
            continue
        if d.get('x0'):
            continue
        tag = d['tag']
        if 'rs1' in tag:                  # a lattice control, kept separate
            pass
        opl, xo, yo, Lo, Mo = _trace_one(csurf, 0.0, 0.0, L, M)
        orc = K0 * (opl - opl_ref)                       # unwrapped truth
        mes = _unwrap_to(float(d['aperture']['piston_c']) - pist_ref, orc)
        lam0, W0, P0, x_c, y_c = _lam0(L, M)
        pred = -K0 * (lam0 - lam0_ref)                   # predicted deficit
        rows.append((tag, float(np.hypot(L, M)), orc, mes, mes - orc, pred,
                     W0, P0 - P0_ref, x_c))

    rows.sort(key=lambda r: r[1])
    print('%-28s %9s %13s %13s %13s %13s %9s' % (
        'run', 'tilt mrad', 'oracle rad', 'measured rad', 'deficit rad',
        'predicted rad', 'pred/def'))
    for (tag, th, orc, mes, dfc, pred, W0, dP0, x_c) in rows:
        print('%-28s %9.4f %13.6e %13.6e %13.6e %13.6e %9.5f' % (
            tag, th * 1e3, orc, mes, dfc, pred,
            (pred / dfc) if dfc else float('nan')))
    print()
    print('%-28s %9s %13s %13s %13s %9s' % (
        'run', 'tilt mrad', 'k0*W(0) rad', 'k0*dP(0) rad', 'x_c mm',
        'meas/oracle'))
    for (tag, th, orc, mes, dfc, pred, W0, dP0, x_c) in rows:
        print('%-28s %9.4f %13.6e %13.6e %13.6f %9.7f' % (
            tag, th * 1e3, K0 * W0, K0 * dP0, x_c * 1e3, mes / orc))
