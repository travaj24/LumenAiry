"""WP-C2 round 2, defect D2 -- the ModalAsymptotic pin's ``kappa`` was drawn
from ONE RANDOM DIRECTION in coefficient space.

VERIFY-WP-C2 measured 6.2423e+06 with the shipped seed and 1.6406e+06 with
another, a factor of 3.8, so the margin the WP-C2 report presents as
11.1x-13.7x reads 2.92x-3.59x along the other direction.  A bar whose value
depends on a draw is a bar whose strictness is arbitrary.

This probe computes the TRUE worst case instead: the full finite-difference
Jacobian of ``propagate_modal_asymptotic`` with respect to a RELATIVE move of
each of the 70 phase coefficients, and from it the induced ``inf <- 2`` norm
-- the largest per-pixel response to a unit-2-norm relative coefficient
perturbation, maximised over directions.  It reports that number, the
direction that attains it, the readings along the sampled directions, and the
margins the two arms then have.

Usage:  OMP_NUM_THREADS=1 ... python r2_conditioning_kappa.py --root <tree>
            --out r2_conditioning_win.json
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import sys
import warnings

EPS = 2.0 ** -52


def _load_tests(root):
    path = os.path.join(root, 'tests', 'unit', 'test_audit_propagation.py')
    spec = importlib.util.spec_from_file_location('_r2_tap', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _jacobian(pma, fit, kwargs, base, peak, delta):
    import numpy as np
    coef = np.asarray(fit.coef_phi, dtype=float)
    cols = np.empty((np.asarray(base).size, coef.size), dtype=np.complex128)
    for j in range(coef.size):
        bump = np.zeros_like(coef)
        bump[j] = delta
        alt = dataclasses.replace(fit, coef_phi=coef * (1.0 + bump))
        cols[:, j] = (np.asarray(pma(alt, **kwargs)).ravel()
                      - np.asarray(base).ravel()) / (peak * delta)
    return cols


def _worst(cols):
    """(kappa, unit direction) -- the induced ``inf <- 2`` norm of ``cols``."""
    import numpy as np
    a, b = cols.real, cols.imag
    aa = np.einsum('ij,ij->i', a, a)
    bb = np.einsum('ij,ij->i', b, b)
    ab = np.einsum('ij,ij->i', a, b)
    tr, det = aa + bb, aa * bb - ab * ab
    lam = 0.5 * (tr + np.sqrt(np.maximum(tr * tr - 4.0 * det, 0.0)))
    i = int(np.argmax(lam))
    u, s, vt = np.linalg.svd(np.stack([a[i], b[i]]), full_matrices=False)
    return float(s[0]), vt[0], i


def _along(pma, fit, kwargs, base, peak, direction, deltas):
    import numpy as np
    coef = np.asarray(fit.coef_phi, dtype=float)
    out = []
    for d in deltas:
        alt = dataclasses.replace(fit, coef_phi=coef * (1.0 + d * direction))
        moved = pma(alt, **kwargs)
        out.append(float(np.max(np.abs(moved - base))) / peak / d)
    return out


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

    tap = _load_tests(root)
    pma = tap.propagate_modal_asymptotic
    fit = tap._build_singlet_fit()

    N = 32
    s2 = np.linspace(-5e-6, 5e-6, N)
    S2X, S2Y = np.meshgrid(s2, s2, indexing='xy')
    cases = {
        'LG_(0,0)': dict(
            source_point=(0.0, 0.0),
            source_amplitudes={(0, 0): 1.0 + 0.0j},
            pupil_amplitudes={(0, 0): 1.0 + 0.0j},
            w_s=50e-6, w_p=0.02, v2_centre=(0.0, 0.0),
            s2_grid_x=S2X, s2_grid_y=S2Y),
        '4-mode LG_p0': dict(
            source_point=(0.0, 0.0),
            source_amplitudes={(0, 0): 1.0 + 0.0j, (1, 0): 0.3 - 0.1j,
                               (2, 0): 0.05 + 0.02j},
            pupil_amplitudes={(0, 0): 1.0 + 0.0j, (1, 0): 0.2 + 0.0j},
            w_s=50e-6, w_p=0.02, v2_centre=(0.0, 0.0),
            s2_grid_x=S2X, s2_grid_y=S2Y),
    }

    payload = {'lumenairy_file': lumenairy.__file__,
               'python': sys.version.split()[0],
               'numpy': np.__version__, 'cases': {}}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for label, kwargs in cases.items():
            base = pma(fit, **kwargs)
            cold = tap.TestAuditFixesV4_14_0_agent_1_1APropagateModalAsymptoticStillBitEqual()
            ref = cold._cold_start_reference_propagate_modal_asymptotic(
                fit, kwargs['source_amplitudes'], kwargs['pupil_amplitudes'],
                kwargs['w_s'], kwargs['w_p'],
                kwargs['s2_grid_x'], kwargs['s2_grid_y'])
            peak = float(np.max(np.abs(ref)))
            reading = float(np.max(np.abs(base - ref))) / max(peak, 1.0)

            cols = _jacobian(pma, fit, kwargs, base, peak, 1e-11)
            kappa_true, direction, pixel = _worst(cols)
            ladder = _along(pma, fit, kwargs, base, peak, direction,
                            (1e-12, 1e-11, 1e-10))

            sampled = {}
            for seed in (20260920, 7770001, 424242):
                rng = np.random.default_rng(seed)
                d = rng.normal(size=np.asarray(fit.coef_phi).shape)
                d /= np.linalg.norm(d)
                sampled[str(seed)] = _along(pma, fit, kwargs, base, peak, d,
                                            (1e-12, 1e-11, 1e-10))
            med = {k: float(np.median(v)) for k, v in sampled.items()}
            row = {
                'kappa_true_worst_direction': kappa_true,
                'kappa_true_pixel': pixel,
                'kappa_along_worst_direction_ladder': ladder,
                'kappa_sampled_medians': med,
                'sampled_spread': max(med.values()) / min(med.values()),
                'true_over_shipped_seed': kappa_true / med['20260920'],
                'floor_eps_kappa_true': EPS * kappa_true,
                'bar_100x': 100.0 * EPS * kappa_true,
                'reading_relative': reading,
                'reading_over_floor': reading / (EPS * kappa_true),
                'margin_bar_over_reading':
                    100.0 * EPS * kappa_true / reading,
                'peak': peak,
            }
            payload['cases'][label] = row
            print(f'{label}: kappa_true {kappa_true:.4e} '
                  f'(shipped seed {med["20260920"]:.4e}, '
                  f'x{row["true_over_shipped_seed"]:.2f}), '
                  f'reading {reading:.3e} = '
                  f'{row["reading_over_floor"]:.2f} floors, '
                  f'margin {row["margin_bar_over_reading"]:.1f}x')
            print(f'    ladder along the worst direction: '
                  f'{[f"{v:.4e}" for v in ladder]}')

    with open(args.out, 'w', encoding='ascii') as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
