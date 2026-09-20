"""VERIFY-WP-B7c round 3 -- the ORACLE FLOOR at full radius (claim 9 / E6).

E6 of the round-2 verification is that the builder's "Debye ``J0`` vs exact"
column substituted the exact azimuth only INSIDE the 99.95 %-energy core,
outside which the two compared fields are identical by construction -- while
the dropped quadratic term grows with radius.  Round 3 re-measured it at full
radius and reports 0.0563 at round 2's own ``Q`` row against a published
0.0003 and a fitted rule predicting 3.2e-4.  It also CORRECTS the round-2
verification the other way, on the ``J0`` arm's ENERGY CLOSURE: 0.99992 /
0.99913 from a well-resolved radial profile against the verification's 0.9972
/ 0.9741, which round 3 attributes to that verification's own 256-radius
reconstruction losing energy on BOTH arms equally (0.894 each).

Everything above is re-measured here, with MY exact quadrature (Gauss-Legendre
azimuth, Simpson radial), and both halves of the adjudication are taken:

* the FULL-RADIUS and the CORE-CONFINED relative L2 on the SAME pointwise
  sample of the output grid's own pixels, so the difference between them is
  the core confinement and nothing else;
* the energy closure of all three arms at TWO radial resolutions, so the
  claim "a coarse profile loses energy on both arms equally" is a measurement
  and not an attribution.

``eps`` is this module's own: the largest phase the Debye step DROPS, i.e.
``k y^2 rho^2 / (2 R0^3)`` evaluated at ``y_max`` and at the field's RMS
radius -- which is how the round-2 verification quoted it ("0.0009 rad at the
rms radius and several radians at the grid corner").

Usage:  python v3floor.py <out.json> <optic>:<z_um> [<optic>:<z_um> ...]
"""
from __future__ import annotations

import json
import sys

import numpy as np
import v3fixtures as FX
import v3oracle as OR
import v3scan as S

N_FAN = 3001
N_SAMPLE_BINS = 24
N_PER_BIN = 20
N_PROFILE = 600


def pixel_sample(N, dx, seed=20260919, n_bins=N_SAMPLE_BINS,
                 per_bin=N_PER_BIN):
    """A radius-stratified, importance-weighted sample of the output grid's
    OWN pixels.

    Equal-radius bins, up to ``per_bin`` pixels drawn per bin, each carrying
    weight ``N_bin / n_drawn`` so the weighted sums are unbiased estimates of
    the full-grid sums.  Nothing is interpolated on either arm.
    """
    rng = np.random.default_rng(seed)
    x = (np.arange(int(N)) - int(N) / 2.0) * float(dx)
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X * X + Y * Y).ravel()
    edges = np.linspace(0.0, r.max() * (1 + 1e-12), n_bins + 1)
    idx, w = [], []
    for i in range(n_bins):
        m = np.nonzero((r >= edges[i]) & (r < edges[i + 1]))[0]
        if m.size == 0:
            continue
        take = m if m.size <= per_bin else rng.choice(m, per_bin,
                                                      replace=False)
        idx.append(take)
        w.append(np.full(take.size, m.size / take.size))
    idx = np.concatenate(idx)
    return idx, np.concatenate(w), r[idx]


def radial_profile_power(ef, z, wl, kind, rho, **kw):
    """``INT |E(rho)|^2 2 pi rho drho`` from a well-resolved radial profile."""
    E = (OR.rs_exact(ef, z, wl, rho, **kw) if kind == 'exact'
         else OR.rs_j0(ef, z, wl, rho))
    f = np.abs(E) ** 2 * rho
    return float(2.0 * np.pi * np.trapezoid(f, rho)), E


def one_row(name, z_um, refine=3):
    fx = FX.FIXTURES[name]
    wl, N, dx = fx['wavelength'], fx['N'], fx['dx']
    z = float(z_um) * 1e-6
    ef = OR.exit_field(fx['prescription'], wl, fx['w0'], n_fan=N_FAN)
    k = 2.0 * np.pi / wl
    ymax = float(np.nanmax(np.abs(ef['y'])))

    idx, w, rs = pixel_sample(N, dx)
    E_ex = OR.rs_exact(ef, z, wl, rs)
    E_j0 = OR.rs_j0(ef, z, wl, rs)
    E_asm_2d = OR.asm_field(ef, z, wl, N, dx, refine=refine,
                            Nf=S.asm_grid(fx, refine))
    E_asm = E_asm_2d.ravel()[idx]

    # the 99.95 %-energy core radius, from a well-resolved radial profile
    rho = np.linspace(0.0, float(rs.max()), N_PROFILE)
    P_ex, prof_ex = radial_profile_power(ef, z, wl, 'exact', rho)
    P_j0, prof_j0 = radial_profile_power(ef, z, wl, 'j0', rho)
    cw = np.abs(prof_ex) ** 2 * rho
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (cw[1:] + cw[:-1])
                                           * np.diff(rho))])
    r_core = float(np.interp(0.9995 * cum[-1], cum, rho))

    # the rms radius of the exact field, where eps is quoted
    r_rms = float(np.sqrt(np.trapezoid(np.abs(prof_ex) ** 2 * rho ** 3, rho)
                          / max(np.trapezoid(np.abs(prof_ex) ** 2 * rho, rho),
                                1e-300)))
    R0 = np.sqrt(z * z + ymax * ymax + r_rms * r_rms)
    eps = float(k * ymax ** 2 * r_rms ** 2 / (2.0 * R0 ** 3))
    R0c = np.sqrt(z * z + ymax * ymax + float(rs.max()) ** 2)
    eps_corner = float(k * ymax ** 2 * float(rs.max()) ** 2 / (2.0 * R0c ** 3))

    den = float(np.sqrt(np.sum(w * np.abs(E_ex) ** 2)))
    core = rs <= r_core
    row = dict(
        optic=name, z_um=float(z_um), N=N, dx_um=dx * 1e6, wavelength=wl,
        y_max_over_z=ymax / z, r_core_um=r_core * 1e6, r_rms_um=r_rms * 1e6,
        eps_rms=eps, eps_corner=eps_corner, rule_037_eps=0.37 * eps,
        n_sample=int(rs.size),
        j0_vs_exact_full=OR.rel_l2(E_j0, E_ex, w),
        j0_vs_exact_core=float(np.sqrt(np.sum(
            w[core] * np.abs(E_j0[core] - E_ex[core]) ** 2)) / den),
        j0_fidelity=OR.w_fidelity(E_j0, E_ex, w),
        asm_vs_exact_full=OR.rel_l2(E_asm, E_ex, w),
        asm_fidelity=OR.w_fidelity(E_asm, E_ex, w),
        P_in=float(ef['P_in']),
        closure_exact_profile=P_ex / ef['P_in'],
        closure_j0_profile=P_j0 / ef['P_in'],
        closure_asm_grid=OR.power(E_asm_2d, dx) / ef['P_in'],
        n_profile=N_PROFILE, n_fan=N_FAN, refine=refine)

    # --- convergence controls ------------------------------------------
    E_ex2 = OR.rs_exact(ef, z, wl, rs, safety=4.0)
    row['ctrl_exact_safety_x2'] = OR.rel_l2(E_ex2, E_ex, w)
    ef2 = OR.exit_field(fx['prescription'], wl, fx['w0'], n_fan=2 * N_FAN + 1)
    E_ex3 = OR.rs_exact(ef2, z, wl, rs)
    row['ctrl_exact_rings_x2'] = OR.rel_l2(E_ex3, E_ex, w)
    E_asm2 = OR.asm_field(ef, z, wl, N, dx, refine=refine + 1,
                          Nf=S.asm_grid(fx, refine + 1)).ravel()[idx]
    row['ctrl_asm_refine_plus1'] = OR.rel_l2(E_asm2, E_asm, w)

    # --- the round-2 verification's own reconstruction, reproduced ------
    for nr in (256, 600, 1200):
        r2 = np.linspace(0.0, float(rs.max()), nr)
        pe, _ = radial_profile_power(ef, z, wl, 'exact', r2)
        pj, _ = radial_profile_power(ef, z, wl, 'j0', r2)
        row[f'closure_exact_{nr}radii'] = pe / ef['P_in']
        row[f'closure_j0_{nr}radii'] = pj / ef['P_in']
    return row


def main():
    out = sys.argv[1]
    import lumenairy
    print('lumenairy.__file__ =', lumenairy.__file__, flush=True)
    rows = []
    for spec in sys.argv[2:]:
        nm, z = spec.split(':')
        print('--- row', spec, flush=True)
        r = one_row(nm, float(z))
        rows.append(r)
        print(json.dumps(r), flush=True)
    with open(out, 'w', encoding='cp1252') as f:
        json.dump(dict(lumenairy=lumenairy.__file__,
                       python=sys.version.split()[0], numpy=np.__version__,
                       rows=rows), f, indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
