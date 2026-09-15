"""The readings every claim rests on, on ONE build, with no oracle.

Run on both builds and diff: the oracle scoring is expensive and
build-independent (it never imports lumenairy), so what has to be shown on
two builds is the READING and the DECISION.

Usage:  python vreadings.py <out.json>
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import vfixtures as FX

#: (id, fixture, builder?, z_um, why)
CASES = [
    ('V_1758', 'V', True, 1758.0, "VERIFY-B7b's fold ring, healthy"),
    ('V_1761', 'V', True, 1761.0,
     'claim 2: branch sum 1.0636 above the bar, completion 1.0021 below it'),
    ('V_1768', 'V', True, 1768.0, 'the blow-up window'),
    ('F_1063', 'F_alt', True, 1063.0, 'healthy'),
    ('F_1073', 'F_alt', True, 1073.0, 'the LARGEST returned reading, 1.0221'),
    ('F_1074', 'F_alt', True, 1074.0, 'the SMALLEST refused reading, 1.092'),
    ('F_1076', 'F_alt', True, 1076.0, "D1's plane"),
    ('F_1080', 'F_alt', True, 1080.0, "D2's plane, coarse grid"),
    ('W_4870', 'W', False, 4870.0, 'my plano-first singlet, healthy fold'),
    ('W_4940', 'W', False, 4940.0, 'blow-up'),
    ('X_960', 'X', False, 960.0, 'NA 0.3865 fold ring, healthy'),
    ('X_992p22', 'X', False, 992.22, 'one pixel before the onset'),
    ('X_992p30', 'X', False, 992.30,
     'THE FALSE REFUSAL: reads 1.0624, oracle fidelity 0.9578'),
    ('X_992p40', 'X', False, 992.40, 'healthy again'),
    ('X_993', 'X', False, 993.0, 'refused, fidelity 0.9333'),
    ('Y_1970', 'Y', False, 1970.0, 'oblate-conic singlet, healthy fold'),
    ('Y_2221p2', 'Y', False, 2221.2, 'fallback, refused at fidelity 0.9492'),
    ('Y_2221p4', 'Y', False, 2221.4, 'fallback, returned at fidelity 0.9500'),
    ('Z_3442', 'Z', False, 3442.0,
     'cemented doublet: RETURNED at 1.0565, inside the claimed gap'),
    ('Z_3510', 'Z', False, 3510.0,
     'R-5: bracket 1.021 (healthy) but continuity 3.03, fidelity 0.647'),
    ('C_16000', 'C', False, 16000.0, 'the CONVERGED control: must read 1'),
    ('C_24000', 'C', False, 24000.0, 'the converged control, past focus'),
    ('C_22800', 'C', False, 22800.0,
     'a MISS: reads 1.002 and is returned at fidelity 0.464'),
]


def main():
    out = sys.argv[1]
    import lumenairy
    from lumenairy.elements import _lens_traced_uniform as U
    print('lumenairy.__file__ =', lumenairy.__file__, flush=True)
    cmax = float(U._MB_PIXEL_CONTINUITY_MAX)
    cmin = float(U._MB_PIXEL_CONTINUITY_MIN)
    pmax = float(U._MB_POWER_RATIO_MAX)
    U._MB_PIXEL_CONTINUITY_MAX = float('inf')
    U._MB_PIXEL_CONTINUITY_MIN = 0.0
    U._MB_POWER_RATIO_MAX = float('inf')
    U._MB_POWER_RATIO_MIN = 0.0
    rows = []
    for cid, name, is_b, z_um, why in CASES:
        fx = FX.builder(name) if is_b else FX.FIXTURES[name]
        E = FX.input_field(fx)
        row = dict(id=cid, fixture=name, z_um=z_um, why=why, N=fx['N'],
                   dx_um=fx['dx'] * 1e6)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                Eo, d = U.apply_real_lens_traced_uniform(
                    E, prescription=fx['prescription'],
                    wavelength=fx['wavelength'], dx=fx['dx'],
                    output_plane_distance=z_um * 1e-6,
                    return_diagnostics=True)
        except Exception as exc:                            # noqa: BLE001
            row['error'] = f'{type(exc).__name__}: {str(exc)[:120]}'
            rows.append(row)
            print(json.dumps(row), flush=True)
            continue
        for k in ('pixel_continuity', 'multibranch_pixel_continuity',
                  'pixel_continuity_of', 'multibranch_power_ratio_bracketed',
                  'reason', 'fell_back', 'n_branch_max'):
            v = d.get(k)
            row[k] = (float(v) if isinstance(v, (int, float, np.floating))
                      and not isinstance(v, bool) else v)
        c = row['pixel_continuity']
        row['shipped_decision'] = ('not_measured' if c is None else
                                   'REFUSED' if c > cmax else
                                   'not_converged_loss' if c < cmin else 'ok')
        row['grid_power'] = float(d.get('grid_power') or 0.0)
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(out, 'w', encoding='cp1252') as f:
        json.dump(dict(lumenairy=lumenairy.__file__,
                       python=sys.version.split()[0], numpy=np.__version__,
                       bars=dict(continuity_max=cmax, continuity_min=cmin,
                                 power_max=pmax), rows=rows), f, indent=1)
    print('DONE', out, flush=True)


if __name__ == '__main__':
    main()
