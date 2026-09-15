"""The ORACLE's own floor on the VERIFY fixtures.

Everything in Task A rests on
``validation/oracles/caustic_fold_truth.py``, so its own convergence and its
own energy closure are measured here before any verdict is taken on it: the
ray-fan quadrature (``n_fan``), the radial sampling (``n_rho``), the
full-plane energy closure against the launched power, and -- because the
azimuthal integral is taken in the Debye ``J0`` form, exact only to
``O(k (y rho / R0^2)^2)`` -- the size of that expansion parameter at the radius
where the field actually lives.

Reported per fixture as the relative L2 change of the (N, N) reference field
between successive refinements; a verdict on a member whose disagreement with
the oracle is smaller than this floor is not a verdict.

Usage: python probe5_oraclefloor.py <out.json> <FIXTURE>:<z_um> [...]
"""
# ruff: noqa: E402, I001  (sys.path is set up between the imports)
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixtures as FX
import oracle as OR


def rel_l2(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def main(argv):
    out = argv[1]
    res = {'python': sys.version.split()[0], 'numpy': np.__version__,
           'rows': []}
    for spec in argv[2:]:
        name, zu = spec.split(':')
        fx = FX.FIXTURES[name]
        z = float(zu) * 1e-6
        ref = {}
        for nf, nr in ((6000, 2400), (12000, 2400), (6000, 4800),
                       (12000, 4800)):
            E, rho, E_rho, ex = OR.oracle_field(fx, z, n_fan=nf, n_rho=nr)
            ref[(nf, nr)] = (E, float(
                2.0 * np.pi * np.trapezoid(np.abs(E_rho) ** 2 * rho, rho)
                / ex['P_in']))
        base = ref[(6000, 2400)][0]
        # the Debye J0 expansion parameter at the radius carrying the light
        N, dx = fx['N'], fx['dx']
        x = (np.arange(N) - N / 2.0) * dx
        X, Y = np.meshgrid(x, x)
        r = np.sqrt(X ** 2 + Y ** 2)
        I = np.abs(base) ** 2
        r_eff = float(np.sqrt(np.sum(I * r ** 2) / max(np.sum(I), 1e-300)))
        y_max = 0.5 * float(fx['prescription']['aperture_diameter']) * 0.98
        k = 2.0 * np.pi / fx['wavelength']
        R0 = z
        debye = float(k * (y_max * r_eff) ** 2 / (2.0 * R0 ** 3))
        res['rows'].append({
            'fixture': name, 'z_um': float(zu), 'note': fx['note'],
            'n_fan_6000_to_12000_relL2': rel_l2(ref[(12000, 2400)][0], base),
            'n_rho_2400_to_4800_relL2': rel_l2(ref[(6000, 4800)][0], base),
            'both_relL2': rel_l2(ref[(12000, 4800)][0], base),
            'energy_closure': ref[(6000, 2400)][1],
            'rms_radius_um': r_eff * 1e6,
            'debye_J0_phase_error_rad': debye,
            'NA_marginal': y_max / z,
        })
        print(json.dumps(res['rows'][-1], indent=1), flush=True)
        with open(out, 'w') as fh:
            json.dump(res, fh, indent=1, default=str)


if __name__ == '__main__':
    main(sys.argv)
