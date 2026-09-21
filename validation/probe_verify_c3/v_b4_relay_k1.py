"""VERIFY-WP-C3 -- the K1 decomposition on WP-B4's OWN two-group relay, at
four grids.

    python v_b4_relay_k1.py <tree> <out.json>

``_collins_readout_k1``'s docstring says K1 "falls only as 1/dx, so N ~ 16000
would be needed to sample it".  K1 = space_term + angle_term, and the ANGLE
term is 2 dx theta / lambda with theta a GRID coordinate of |fftfreq| -- whose
outermost bin is exactly 1/(2 dx).  So the angle term is bounded above by
EXACTLY 1 and quantised in steps of exactly 2/N.  If the angular support is
grid-CLIPPED the angle term is exactly 1 and K1 > 1 at EVERY N, and no grid
refinement can reach the one-step form.  This measures which of the two is
happening on the relay the docstring is written about.
"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, _TREE)

import vlib  # noqa: E402
import numpy as np  # noqa: E402

import lumenairy.propagators.carrier as CA  # noqa: E402

vlib.anchor(_TREE)

WL = 1.31e-6
FINAL = 8e-3


def _singlet(R1, R2, d, glass, ap, name):
    return {'name': name, 'aperture_diameter': ap, 'thicknesses': [d],
            'surfaces': [
                {'radius': R1, 'glass_before': 'air', 'glass_after': glass,
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
                {'radius': R2, 'glass_before': glass, 'glass_after': 'air',
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]}


def main():
    out = sys.argv[2]
    rows = []
    presc = _singlet(60e-3, -60e-3, 3e-3, 'N-BK7', 14e-3, 'p')
    groups = [{'prescription': presc, 'gap_before': 20e-3},
              {'prescription': presc, 'gap_before': 10e-3}]
    tkw = dict(on_undersample='silent', on_noncollimated='silent')
    for N in (256, 512, 1024, 2048):
        dx = 60e-6 * 256.0 / N          # same physical window as the fixture
        g = (np.arange(N) - N // 2) * dx
        env = np.exp(-((g[None, :] ** 2 + g[:, None] ** 2) / 4.5e-3 ** 2)
                     ).astype(np.complex128)
        cap = {}

        orig = CA._collins_readout_k1

        def patched(e, R, z, wl, dxx, dyy, _o=orig, _c=cap):
            k1 = _o(e, R, z, wl, dxx, dyy)
            R_x, _R_y, _ = CA._parse_carrier(R, 'v')
            Ax, B, _, _ = CA._collins_envelope_abcd(R_x, z, np.inf)
            r_x, _r_y, th_x, _th_y = CA._collins_input_box(
                e, dxx, dyy, wl, CA._COLLINS_TAIL_FRAC)
            nx = int(np.shape(e)[-1])
            space = 2.0 * dxx * abs(Ax) * r_x / abs(B) / wl
            angle = 2.0 * dxx * th_x / wl
            _c.update(k1=float(k1), space=float(space), angle=float(angle),
                      angle_is_exactly_one=bool(angle == 1.0),
                      j=int(round(angle * nx / 2.0)), nyquist_bin=nx // 2,
                      Ax=float(Ax), B=float(B), r_x=float(r_x),
                      th_x=float(th_x), exit_dx=float(dxx), exit_N=nx,
                      R_exit=float(R_x))
            return k1

        CA._collins_readout_k1 = patched
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                res = CA.propagate_traced_carrier_chain(
                    env, groups, WL, dx, r_in=60e-3, ray_subsample=16,
                    n_workers=1, traced_kwargs=tkw, final_leg='paraxial',
                    final_distance=FINAL,
                    focus_readout=dict(dx_out=0.5e-6, N_out=64),
                    transport='collins')
            st = res.stages[-1]
            row = dict(cap)
            row.update(N=N, dx_in=dx, route=st.get('readout_route'),
                       stage_k1=st.get('readout_route_k1'),
                       reason=st.get('readout_route_reason'),
                       n_kelly=len([q for q in w
                                    if 'under-sampled' in str(q.message)]))
        except Exception as exc:                      # noqa: BLE001
            row = dict(N=N, raised=('%s: %s'
                                    % (type(exc).__name__, exc))[:300])
            print('    RAISED %s' % row['raised'])
        finally:
            CA._collins_readout_k1 = orig
        rows.append(row)
        print('[b4] N=%-5d route=%-8s K1=%-20r space=%-22r angle=%-18r '
              'angle==1 %s  j=%s/%s'
              % (N, row.get('route'), row.get('k1'), row.get('space'),
                 row.get('angle'), row.get('angle_is_exactly_one'),
                 row.get('j'), row.get('nyquist_bin')))
    vlib.write_json({'build': vlib.build_tag(), 'tree': _TREE,
                     'carrier_file': CA.__file__, 'rows': rows}, out)


if __name__ == '__main__':
    main()
