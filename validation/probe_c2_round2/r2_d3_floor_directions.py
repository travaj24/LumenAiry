"""WP-C2 round 2, defect D8 -- the d3 arm-2 floor's own spread over one-ULP
directions.

``test_niche_d3_guards.py::test_the_residual_degree_moves_the_multiplexed_route_only_through_c6``
bars ``moved > 3.0 * noise``, with ``noise`` measured from ONE perturbation
(``np.nextafter(..., +inf)`` on both parts of the input envelope).
VERIFY-WP-C2 measured that floor over four one-ULP directions and found a
3.22x (Windows) / 4.79x (WSL) spread -- at or above the 3.0 multiplier sitting
on it -- so the floor was a draw as much as the magnitude it replaced.

This probe measures the same four directions on the running build, plus the
natural control the arm never asserted (the same degree twice, which must read
exactly 0), and reports the margin at the WORST floor.

Usage:  OMP_NUM_THREADS=1 ... python r2_d3_floor_directions.py --root <tree>
            --out r2_d3_floor_win.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import warnings


def _load_tests(root):
    path = os.path.join(root, 'tests', 'unit', 'test_niche_d3_guards.py')
    spec = importlib.util.spec_from_file_location('_r2_d3', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    import lumenairy
    assert os.path.abspath(lumenairy.__file__).startswith(root)
    import numpy as np
    print('lumenairy.__file__ =', lumenairy.__file__)
    print('numpy', np.__version__, 'python', sys.version.split()[0])

    d3 = _load_tests(root)
    payload = {'lumenairy_file': lumenairy.__file__,
               'python': sys.version.split()[0],
               'numpy': np.__version__}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        on6 = d3._mux_chain_field(0.023, degree=6, launch=True)
        on4 = d3._mux_chain_field(0.023, degree=4, launch=True)
        ref = float(np.linalg.norm(on6))
        moved = float(np.linalg.norm(on6 - on4)) / ref
        floors = {}
        for kind in ('up', 'down', 'real', 'one_element'):
            floors[kind] = d3._mux_last_bit_noise(
                0.023, degree=6, launch=True, kind=kind)
        # the control the arm never asserted: the same degree twice
        again = d3._mux_chain_field(0.023, degree=6, launch=True)
        control = float(np.linalg.norm(again - on6)) / ref

        # and the sibling arm's floor, over the same directions
        bad6 = d3._linearity_error(0.023)
        sib = {}
        for kind in ('up', 'down', 'real', 'one_element'):
            sib[kind] = abs(
                d3._linearity_error(0.023, nudge=kind) - bad6) / max(
                    bad6, 1e-300)
        d3._lens_traced._REMAP_RESID_EIKONAL_DEGREE = 4
        try:
            bad4 = d3._linearity_error(0.023)
        finally:
            d3._lens_traced._REMAP_RESID_EIKONAL_DEGREE = 6
        degree_effect = abs(bad4 - bad6) / max(bad6, 1e-300)

    payload['arm2'] = {
        'moved': moved,
        'floors': floors,
        'floor_max': max(floors.values()),
        'floor_min': min(floors.values()),
        'floor_spread': max(floors.values()) / min(floors.values()),
        'margin_at_worst_floor': moved / max(floors.values()),
        'margin_at_shipped_floor': moved / floors['up'],
        'same_degree_twice': control,
    }
    payload['arm1'] = {
        'bad6': bad6, 'bad4': bad4,
        'degree_effect': degree_effect,
        'floors': sib,
        'floor_max': max(sib.values()),
        'floor_min': min(sib.values()),
        'floor_spread': (max(sib.values()) / min(sib.values())
                         if min(sib.values()) > 0 else float('inf')),
        'margin_at_worst_floor': degree_effect / max(sib.values()),
        'margin_at_shipped_floor': degree_effect / sib['up'],
    }
    for arm in ('arm1', 'arm2'):
        r = payload[arm]
        print(f'{arm}: floors {r["floors"]}')
        print(f'  spread {r["floor_spread"]:.3f}x  '
              f'margin at worst floor {r["margin_at_worst_floor"]:.2f}x  '
              f'(at the shipped single draw '
              f'{r["margin_at_shipped_floor"]:.2f}x)')
    print('arm2 same-degree-twice control:',
          payload['arm2']['same_degree_twice'])

    with open(args.out, 'w', encoding='ascii') as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
