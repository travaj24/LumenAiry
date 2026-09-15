"""The oracle's floor, and the NA ceiling VERIFY-WP-B7c's D5 recorded.

Per fixture and plane this measures

* the Debye ``J0`` phase error the shared oracle's azimuthal form drops,
  ``k (y rho)^2 / (2 R0^3)`` at the rms radius the light occupies -- D5's
  column, reproduced;
* the DIFFERENCE between the shared oracle and the EXACT azimuthal quadrature
  of ``oracle.rs_integral_exact`` at the same plane, in fidelity and in power.
  That difference IS the ceiling, measured rather than bounded;
* the exact quadrature's own convergence in ``n_phi`` and ``n_fan``, so the
  difference above can be attributed;
* energy closure of both.

Usage::

    python oraclefloor.py <out.json> <FIXTURE>:<z_um> [...]
"""
# ruff: noqa: E402, I001
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixtures as FX                                     # noqa: E402
import oracle as OR                                       # noqa: E402

_ORACLES = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'oracles')
if _ORACLES not in sys.path:
    sys.path.insert(0, _ORACLES)
import caustic_fold_truth as cft                          # noqa: E402


def one(name, zum, n_fan=4000, safety=1.5):
    fx = FX.FIXTURES[name]
    z = zum * 1e-6
    job = OR.job_for(fx, z, n_fan=n_fan)
    h, ys, opl, amp, z_exit, ex = cft.build_exit_field(job, n_fan=n_fan)
    rho = OR.rho_grid(fx, n_inner=800, n_outer=200)
    t0 = time.perf_counter()
    E_deb = cft._rs_integral(h, ys, opl, amp, z, fx['wavelength'], rho)
    t_deb = time.perf_counter() - t0
    rc = max(OR.core_radius(rho, E_deb), 4.0 * fx['dx'])
    core = rho <= rc
    t0 = time.perf_counter()
    E_ex, n_used = OR.rs_integral_exact(
        ys, opl, amp, z, fx['wavelength'], rho[core], phi_safety=safety,
        return_n_phi=True)
    t_ex = time.perf_counter() - t0
    E_ex2 = OR.rs_integral_exact(ys, opl, amp, z, fx['wavelength'],
                                 rho[core], phi_safety=2.0 * safety)
    job2 = OR.job_for(fx, z, n_fan=2 * n_fan)
    h2, ys2, opl2, amp2, _ze2, _ex2 = cft.build_exit_field(job2,
                                                           n_fan=2 * n_fan)
    E_ex_f2 = OR.rs_integral_exact(ys2, opl2, amp2, z, fx['wavelength'],
                                   rho[core], phi_safety=safety)

    def relL2(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    w = np.abs(E_deb) ** 2 * rho
    tot = float(np.trapezoid(w, rho))
    rms = OR.rms_radius(rho, E_deb)
    # the two fields on the fixture's own 2-D grid, built the same way
    N, dx = fx['N'], fx['dx']
    E_hyb = E_deb.copy()
    E_hyb[core] = E_ex
    A_deb = cft.radial_to_2d(rho, E_deb, N, dx)
    A_ex = cft.radial_to_2d(rho, E_hyb, N, dx)
    return {
        'fixture': name, 'z_um': zum, 'N': N, 'dx': dx,
        'y_max_over_z': float(np.nanmax(np.abs(ys)) / z),
        'rms_radius_um': rms * 1e6, 'r_core_um': rc * 1e6,
        'core_energy_fraction': (float(np.trapezoid(w[core], rho[core]))
                                 / tot if tot > 0 else None),
        'debye_J0_phase_error_rad': OR.debye_phase_error(
            ys, z, fx['wavelength'], rms),
        'closure_debye': float(2.0 * np.pi * tot / ex['P_in']),
        'n_phi_min': int(n_used.min()), 'n_phi_max': int(n_used.max()),
        'exact_phi_safety_x2_relL2': relL2(E_ex, E_ex2),
        'exact_n_fan_x2_relL2': relL2(E_ex, E_ex_f2),
        'debye_vs_exact_core_relL2': relL2(E_deb[core], E_ex),
        'debye_vs_exact_fidelity_2d': OR.fidelity(A_deb, A_ex),
        'debye_vs_exact_power_2d': (OR.power(A_deb, dx)
                                    / OR.power(A_ex, dx)),
        't_debye_s': t_deb, 't_exact_s': t_ex,
        'n_fan': n_fan, 'n_rho': int(rho.size), 'n_core': int(core.sum()),
    }


def main(argv):
    out = argv[1]
    res = {'python': sys.version.split()[0], 'numpy': np.__version__,
           'rows': []}
    for spec in argv[2:]:
        name, zum = spec.split(':')
        r = one(name, float(zum))
        res['rows'].append(r)
        print(f"{r['fixture']:7s} z={r['z_um']:9.2f} "
              f"y/z={r['y_max_over_z']:.3f} rms={r['rms_radius_um']:7.2f}um "
              f"J0err={r['debye_J0_phase_error_rad']:8.4f}rad | "
              f"core {r['r_core_um']:6.1f}um "
              f"({r['core_energy_fraction']:.5f}) n_phi "
              f"{r['n_phi_min']}..{r['n_phi_max']} | "
              f"deb-vs-exact relL2={r['debye_vs_exact_core_relL2']:.4f} "
              f"fid={r['debye_vs_exact_fidelity_2d']:.5f} "
              f"pow={r['debye_vs_exact_power_2d']:.5f} | conv: safety x2 "
              f"{r['exact_phi_safety_x2_relL2']:.2e} nfan x2 "
              f"{r['exact_n_fan_x2_relL2']:.2e} closure "
              f"{r['closure_debye']:.5f} t={r['t_exact_s']:.0f}s", flush=True)
        with open(out, 'w') as fh:
            json.dump(res, fh, indent=1, default=str)


if __name__ == '__main__':
    main(sys.argv)
