"""Round 2 (V-D19) -- the two-group chain's quadrature decision, measured.

``tests/unit/test_audit2609_b4_collins_transport.py::TestGateCTwoGroupChain::
test_the_two_transports_agree_on_this_chain`` asserted
``np.array_equal(sziklas_field, collins_field)`` on a 2048x2048 chain.  That is
bit-equal only while every leg stays in the TRANSFER-FUNCTION half of the
quadrature split, which is a MEASURED condition (``K1 > 1``) the assertion did
not make -- testing-standards shape S1/S5.  In one 48-file sweep the two arms
differed in the sixth significant figure (~8.5e-06 relative), which is a
quadrature switch and not round-off; it did not reproduce.

This probe records the numbers the restated assertion is derived from: the
per-leg ``K1`` / ``K3`` readings and their distance from the threshold of 1,
the quadrature form each leg took, and the actual disagreement between the two
arms.

    python r2_b4_chain.py <tree> <out.json>
"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'probe_wave5_hyg2'))
import hlib                                                  # noqa: E402

import numpy as np                                           # noqa: E402


def main(tree, out):
    hlib.anchor(tree)
    import lumenairy.propagators.carrier as C
    from lumenairy.elements import apply_real_lens_traced       # noqa: F401
    from lumenairy.glass import GLASS_REGISTRY
    from lumenairy.raytrace.seidel import system_abcd_prescription

    lam, ng = 1.31e-6, 1.5168
    GLASS_REGISTRY['_B4GLASS'] = (lambda wl: ng)
    sd = 10e-3

    def presc():
        return {'surfaces': [
            {'radius': 51.68e-3, 'glass_before': 'air',
             'glass_after': '_B4GLASS', 'semi_diameter': sd},
            {'radius': -51.68e-3, 'glass_before': '_B4GLASS',
             'glass_after': 'air', 'semi_diameter': sd}],
            'thicknesses': [5e-3], 'aperture_diameter': 2 * sd,
            'stop_index': 0}

    M, _, _, _ = system_abcd_prescription(presc(), lam)
    w0, z1 = 6.0e-6, 30e-3
    zR = np.pi * w0 ** 2 / lam
    r_in = z1 * (1.0 + (zR / z1) ** 2)
    w_l = w0 * np.sqrt(1.0 + (z1 / zR) ** 2)
    n = 2048
    dx = 2 * 3.0 * w_l / n
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    env0 = np.exp(-(X ** 2 + Y ** 2) / (w_l * w_l)).astype(np.complex128)
    tk = dict(amplitude_model='ray_density', preserve_input_phase='remap',
              remap_sampling='full')
    gap = 40e-3
    R_a = (M[0, 0] * r_in + M[0, 1]) / (M[1, 0] * r_in + M[1, 1])
    R_b = R_a + gap
    R_c = (M[0, 0] * R_b + M[0, 1]) / (M[1, 0] * R_b + M[1, 1])
    groups = [{'prescription': presc(), 'gap_before': 0.0},
              {'prescription': presc(), 'gap_before': gap}]

    res = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for tr in ('sziklas', 'collins'):
            r = C.propagate_traced_carrier_chain(
                env0, groups, lam, dx, r_in=r_in, ray_subsample=2,
                final_distance=-R_c * 0.5, final_leg='paraxial',
                traced_kwargs=tk, carrier_reference='sphere', transport=tr)
            res[tr] = r
    a = np.asarray(res['sziklas'].field)
    b = np.asarray(res['collins'].field)
    peak = float(np.max(np.abs(a)))
    stages = []
    for st in res['collins'].stages:
        if st.get('collins_form'):
            stages.append({
                'form': st.get('collins_form'),
                'k1': [float(v) for v in (st.get('collins_k1') or (0.0, 0.0))],
                'k3': [float(v) for v in (st.get('collins_k3') or (0.0, 0.0))],
            })
    rec = {
        'peak': peak,
        'bit_equal': bool(np.array_equal(a, b)),
        'max_abs_diff': float(np.max(np.abs(a - b))),
        'max_abs_diff_over_peak': float(np.max(np.abs(a - b)) / peak),
        'rel_l2': float(np.linalg.norm(a - b) / np.linalg.norm(a)),
        'stages': stages,
        'decision_margins': [max(max(s['k1']), max(s['k3'])) for s in stages],
        'shape': list(a.shape),
        'dtype': str(a.dtype),
    }
    hlib.write_json(rec, out)
    print(f"  bit_equal={rec['bit_equal']}  "
          f"max|a-b|/peak={rec['max_abs_diff_over_peak']:.6e}  "
          f"rel_l2={rec['rel_l2']:.6e}", file=sys.stderr)
    for s, m in zip(stages, rec['decision_margins']):
        print(f"  form={s['form']}  k1={s['k1']}  k3={s['k3']}  "
              f"margin={m:.6f}", file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
