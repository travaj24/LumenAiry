"""VERIFY-WP-C3 CLAIM 9 -- my own design-121 driver.

    python v_d121.py <tree> <out.json> <N> [force]

``force`` is '', 'sziklas' or 'collins': it OVERRIDES the K1 route decision
(by substituting the module's own ``_collins_readout_k1`` with a constant) so
the COST of a route flip can be measured directly at the same N.

Beyond the report's table this driver also records, for the readout's K1, the
pieces it is built from (r_x, r_y, th_x, th_y, A, B), my own independent
recomputation of K1 from those pieces, and -- the point -- the CUMSUM MARGIN
at the containment index on each of the four marginals, i.e. how much the
marginal power would have to move for the containment radius to jump by one
sample, together with what such a jump would do to K1.  The containment radius
is quantised to the lattice, so K1 cannot drift: it is either identical or it
steps.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, _TREE)

import vlib  # noqa: E402
import numpy as np  # noqa: E402

import lumenairy as la  # noqa: E402
import lumenairy.propagators.carrier as CA  # noqa: E402

vlib.anchor(_TREE)

_D121 = os.path.join(_TREE, 'validation', 'repro_traced_carrier_121')

LAM = 1.31e-6
W0 = 4e-6
TRAILING = 7.7058e-3
DXO = 0.25e-6
NOUT = 256


def _metrics(field, dx_out):
    I = np.abs(np.asarray(field)) ** 2
    n = I.shape[-1]
    ax = (np.arange(n) - n / 2.0) * dx_out
    pk = float(I.max())
    iy, ix = np.unravel_index(int(np.argmax(I)), I.shape)
    row = I[iy]
    half = pk / 2.0
    xs = []
    for d in (-1, 1):
        j = ix
        while 0 < j < n - 1 and row[j] > half:
            j += d
        a, b = row[j], row[j - d]
        t = 0.0 if b == a else (half - a) / (b - a)
        xs.append(ax[j] + t * (ax[j - d] - ax[j]))
    r2 = (ax - ax[ix])[None, :] ** 2 + (ax - ax[iy])[:, None] ** 2
    tot = float(I.sum())
    ee = {'ee%d' % int(r * 1e6): float(I[r2 <= r * r].sum() / tot * 100.0)
          for r in (3e-6, 6e-6, 12e-6)}
    return dict(peak=pk, fwhm_um=abs(xs[1] - xs[0]) * 1e6,
                power=tot * dx_out * dx_out,
                centroid_x_um=float((I.sum(axis=0) * ax).sum() / tot * 1e6),
                **ee)


# --- my own containment measurement, written here, not imported -------------
def _marginals(E):
    A = np.asarray(E)
    P = np.abs(A) ** 2
    return P.sum(axis=0), P.sum(axis=1)


def _contain(P, coord, frac):
    """Smallest radius about 0 holding 1-frac of P.

    Beyond the radius this returns everything a DISCRETE decision needs: the
    chosen index, the cumsum margins either side of the threshold (relative to
    the total and in ULPs of it), and -- the number that actually decides a
    route flip -- the headroom before the RADIUS itself steps.  The sorted
    magnitudes come in +/- pairs for an FFT axis, so the index can move by one
    without the radius moving at all; what matters is the cumulative power that
    must shift before the next DISTINCT magnitude is reached, in either
    direction.
    """
    tot = float(P.sum())
    eps = float(np.finfo(np.float64).eps)
    d = np.abs(np.asarray(coord, dtype=np.float64))
    order = np.argsort(d, kind='stable')
    ds = d[order]
    c = np.cumsum(P[order])
    target = (1.0 - float(frac)) * tot
    i = int(np.searchsorted(c, target, side='left'))
    i = min(i, ds.size - 1)
    lo = float(c[i - 1]) if i > 0 else 0.0
    hi = float(c[i])
    r0 = float(ds[i])
    # UP: the last index sharing this radius; the radius grows only once the
    # threshold passes its cumsum.
    iu = i
    while iu + 1 < ds.size and ds[iu + 1] == r0:
        iu += 1
    r_up = float(ds[iu + 1]) if iu + 1 < ds.size else r0
    head_up = (float(c[iu]) - target) / tot
    # DOWN: the first index sharing this radius; the radius shrinks once the
    # threshold falls below the cumsum just before it.
    idn = i
    while idn - 1 >= 0 and ds[idn - 1] == r0:
        idn -= 1
    r_dn = float(ds[idn - 1]) if idn - 1 >= 0 else r0
    head_dn = (target - (float(c[idn - 1]) if idn >= 1 else 0.0)) / tot
    return dict(radius=r0, index=i, index_first=idn, index_last=iu,
                saturated=bool(i == ds.size - 1),
                margin_below=(target - lo) / tot,
                margin_above=(hi - target) / tot,
                margin_below_ulps=(target - lo) / tot / eps,
                margin_above_ulps=(hi - target) / tot / eps,
                radius_up=r_up, radius_down=r_dn,
                headroom_up=head_up, headroom_down=head_dn,
                headroom_up_ulps=head_up / eps,
                headroom_down_ulps=head_dn / eps,
                cumsum_window=[float((c[j] - target) / tot)
                               for j in range(max(0, i - 4),
                                              min(ds.size, i + 7))],
                radius_window=[float(ds[j])
                               for j in range(max(0, i - 4),
                                              min(ds.size, i + 7))])


class Rec(object):
    def __init__(self):
        self.calls = []


def _install_recorder(rec, force):
    orig = CA._collins_readout_k1

    def patched(env, R, z, wavelength, dx, dy):
        k1 = orig(env, R, z, wavelength, dx, dy)
        frac = CA._COLLINS_TAIL_FRAC
        R_x, R_y, _ = CA._parse_carrier(R, 'v')
        Ax, B, _, _ = CA._collins_envelope_abcd(R_x, z, np.inf)
        Ay, _, _, _ = CA._collins_envelope_abcd(R_y, z, np.inf)
        E = np.asarray(env)
        Ny, Nx = E.shape[-2], E.shape[-1]
        x = (np.arange(Nx) - Nx / 2) * dx
        y = (np.arange(Ny) - Ny / 2) * dy
        Px, Py = _marginals(E)
        S = np.fft.fft2(np.ascontiguousarray(E, dtype=np.complex128))
        Sx, Sy = _marginals(S)
        fx = np.fft.fftfreq(Nx, d=dx)
        fy = np.fft.fftfreq(Ny, d=dy)
        cx = _contain(Px, x, frac)
        cy = _contain(Py, y, frac)
        ax_ = _contain(Sx, fx, frac)
        ay_ = _contain(Sy, fy, frac)
        th_x = ax_['radius'] * wavelength
        th_y = ay_['radius'] * wavelength
        k1x = 2.0 * dx * (abs(Ax) * cx['radius'] / abs(B) + th_x) / wavelength
        k1y = 2.0 * dy * (abs(Ay) * cy['radius'] / abs(B) + th_y) / wavelength
        # THE DECOMPOSITION.  angle_term = 2 dx theta / lambda is bounded
        # above by EXACTLY 1 (theta is a GRID coordinate of |fftfreq|, whose
        # outermost bin is 1/(2 dx)) and quantised in steps of exactly 2/N.
        # So K1 <= 1 is reachable only when the angular support lands at least
        # ceil(space_term * N/2) FFT bins INSIDE the Nyquist edge -- and a
        # grid-CLIPPED angular support makes K1 > 1 identically.
        space_x = 2.0 * dx * abs(Ax) * cx['radius'] / abs(B) / wavelength
        angle_x = 2.0 * dx * th_x / wavelength
        space_y = 2.0 * dy * abs(Ay) * cy['radius'] / abs(B) / wavelength
        angle_y = 2.0 * dy * th_y / wavelength
        jx = int(round(angle_x * Nx / 2.0))
        jy = int(round(angle_y * Ny / 2.0))
        d_r_x = (2.0 * dx * abs(Ax) * (cx['radius_up'] - cx['radius'])
                 / abs(B) / wavelength)
        d_th_x = 2.0 * dx * (ax_['radius_up'] - ax_['radius'])
        rec.calls.append(dict(
            space_term_x=space_x, angle_term_x=angle_x,
            space_term_y=space_y, angle_term_y=angle_y,
            angle_bin_jx=jx, angle_bin_jy=jy, nyquist_bin=Nx // 2,
            angle_clipped_x=bool(angle_x == 1.0),
            angle_clipped_y=bool(angle_y == 1.0),
            bins_inside_nyquist_x=Nx // 2 - jx,
            k1_if_angle_steps_up=float(space_x + angle_x + 2.0 / Nx),
            k1_if_angle_steps_down=float(space_x + angle_x - 2.0 / Nx),
            k1_library=float(k1), k1_mine=float(max(k1x, k1y)),
            k1x=float(k1x), k1y=float(k1y),
            Ax=float(Ax), Ay=float(Ay), B=float(B),
            dx=float(dx), dy=float(dy), Nx=int(Nx), Ny=int(Ny),
            R_x=float(R_x), R_y=float(R_y), z=float(z), frac=float(frac),
            space_x=cx, space_y=cy, angle_x=ax_, angle_y=ay_,
            th_x=th_x, th_y=th_y,
            k1_step_if_r_moves_one_sample=float(d_r_x),
            k1_step_if_theta_moves_one_sample=float(d_th_x),
            env_l2=float(np.linalg.norm(E)),
            env_absmax=float(np.abs(E).max()),
        ))
        if force == 'sziklas':
            return 2.0
        if force == 'collins':
            return 0.5
        return k1

    CA._collins_readout_k1 = patched
    return orig


def main():
    out_path = sys.argv[2]
    N = int(sys.argv[3])
    force = sys.argv[4] if len(sys.argv) > 4 else ''

    rec = {'build': vlib.build_tag(), 'env': vlib.env_tag(), 'tree': _TREE,
           'lumenairy_pre': la.__file__, 'N': N, 'force_route': force,
           'numpy': np.__version__}
    sys.path.insert(0, _D121)
    # tx_design_study_sim.py (imported by _d121_common) ASSERTS that the
    # resolved lumenairy lives under LUMENAIRY_ROOT, defaulting to the sibling
    # D:\...\Lumenairy checkout.  Without this the import dies -- which is why
    # the package's own probe_d121_acceptance.py only runs with the variable
    # already exported in the shell.  Set it to the tree under test.
    os.environ['LUMENAIRY_ROOT'] = _TREE
    rec['LUMENAIRY_ROOT'] = _TREE
    os.environ.setdefault(
        'D121_ROOT',
        'D:' + os.sep + os.path.join(
            'Metacept', 'Neurophos', 'Python_Test_Scripts',
            'Free_Space_Optics'))
    import _d121_common as D  # noqa: E402
    # THE TRAP: _d121_common inserts <D121_ROOT>/Lumenairy at sys.path[0].
    # lumenairy is already imported, so the anchor must STILL hold -- re-read.
    rec['lumenairy_post'] = la.__file__
    rec['anchor_held'] = os.path.abspath(la.__file__).lower().startswith(
        _TREE.lower() + os.sep)
    assert rec['anchor_held'], rec['lumenairy_post']
    rec['zmx'] = D.ZMX
    rec['carrier_file'] = CA.__file__

    pre, post, gap_to_doe, period = D.geometry()
    rec['doe_period_um'] = period * 1e6
    rec['n_groups_pre'] = len(pre)
    rec['n_groups_post'] = len(post)
    groups = list(pre)
    if post:
        post = [dict(post[0], gap_before=post[0]['gap_before'] + gap_to_doe)] \
            + list(post[1:])
        groups += post

    zR = np.pi * W0 * W0 / LAM
    z1 = 2e-3
    w_z1 = W0 * np.sqrt(1.0 + (z1 / zR) ** 2)
    R1 = z1 * (1.0 + (zR / z1) ** 2)
    dx0 = 1.0e-6 * 2048.0 / N
    x = (np.arange(N) - N // 2) * dx0
    env = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
                 / (w_z1 * w_z1)).astype(np.complex128)
    rec['launch'] = {'w0_um': W0 * 1e6, 'z1_mm': z1 * 1e3,
                     'w_z1_mm': w_z1 * 1e3, 'R1_mm': R1 * 1e3,
                     'dx0_um': dx0 * 1e6}
    rec['env_digest'] = float(np.linalg.norm(env))

    fr = dict(dx_out=DXO, N_out=NOUT)
    tkw = dict(on_undersample='silent', on_noncollimated='silent')
    rows = []
    transports = ('sziklas', 'collins') if not force else ('collins',)
    if len(sys.argv) > 5 and sys.argv[5]:
        transports = tuple(sys.argv[5].split(','))
    rec['transports'] = list(transports)
    for tr in transports:
        r = Rec()
        orig = _install_recorder(r, force if tr == 'collins' else '')
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                res = CA.propagate_traced_carrier_chain(
                    env, groups, LAM, dx0, r_in=R1, ray_subsample=4,
                    n_workers=1, traced_kwargs=tkw, final_leg='paraxial',
                    final_distance=TRAILING, focus_readout=fr, transport=tr)
                row = {'transport': tr, 'raised': None,
                       'seconds': round(time.perf_counter() - t0, 2)}
                row.update(_metrics(res.field, DXO))
                st = res.stages[-1]
                row['readout_route'] = st.get('readout_route')
                row['readout_route_k1'] = st.get('readout_route_k1')
                row['readout_route_reason'] = st.get('readout_route_reason')
                row['n_stages'] = len(res.stages)
                row['n_kelly'] = len([q for q in w
                                      if 'under-sampled' in str(q.message)])
                row['all_warnings'] = sorted({str(q.message)[:120]
                                              for q in w})
        except Exception as exc:  # noqa: BLE001
            row = {'transport': tr,
                   'raised': ('%s: %s' % (type(exc).__name__, exc))[:400],
                   'seconds': round(time.perf_counter() - t0, 2)}
        finally:
            CA._collins_readout_k1 = orig
        row['k1_calls'] = r.calls
        rows.append(row)
    rec['rows'] = rows
    vlib.write_json(rec, out_path)
    for r in rows:
        if r.get('raised'):
            print('[v_d121] %-8s RAISED %s' % (r['transport'], r['raised']))
        else:
            print('[v_d121] N=%d %-8s force=%r FWHM %.6f  EE3 %.4f  '
                  'EE6 %.4f  route %s  K1 %r  kelly %d  %.1fs'
                  % (N, r['transport'], force, r['fwhm_um'], r['ee3'],
                     r['ee6'], r['readout_route'], r['readout_route_k1'],
                     r['n_kelly'], r['seconds']))
            for c in r['k1_calls']:
                print('    K1 lib=%r mine=%r k1x=%r k1y=%r'
                      % (c['k1_library'], c['k1_mine'], c['k1x'], c['k1y']))
                print('    space=%r angle=%r  j=%d/%d clipped=%s  '
                      'angle-tie headroom up=%.4e down=%.4e (rel power)'
                      % (c['space_term_x'], c['angle_term_x'],
                         c['angle_bin_jx'], c['nyquist_bin'],
                         c['angle_clipped_x'], c['angle_x']['headroom_up'],
                         c['angle_x']['headroom_down']))


if __name__ == '__main__':
    main()
