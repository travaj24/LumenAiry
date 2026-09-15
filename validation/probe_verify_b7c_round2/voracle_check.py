"""Cross-validate the verifier's THREE propagators against each other.

Claim (6) is the ``J0`` question, and it is asked here from two independent
directions: the exact azimuthal quadrature (the SAME integral, exact in
``phi``) and a band-limited angular-spectrum propagation (a different
integral altogether, exact for the scalar Helmholtz equation given the
boundary field).  If the two agree with each other and BOTH disagree with the
``J0`` form by the same amount, the disagreement is the ``J0`` truncation and
not either method's error.

The exact arm is O(n_ring x n_rho x n_phi) with ``n_phi`` running to
thousands, so the J0-vs-exact comparison is taken on a REDUCED ring count and
radial sampling, with the same reduction applied to the ``J0`` arm -- a like
for like comparison -- and both reductions carry their own convergence
control.

Usage:  python voracle_check.py <out.json> <fixture|B:fixture> <z_um> [...]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import vfixtures as FX
import vroracle as OR


def _asm_grid(fx, refine):
    """Fine grid covering 2.2x the clear aperture with the coarse pixel
    centres an exact subset."""
    N, dx = fx['N'], fx['dx']
    need = 2.2 * float(fx['prescription']['aperture_diameter'])
    dxf = dx / refine
    Nf = max(int(np.ceil(need / dxf)), refine * N)
    if (Nf - refine * N) % 2:
        Nf += 1
    return Nf


def run(name, z, n_fan=6000, n_rho=1600, n_fan_cmp=1500, n_rho_cmp=320,
        safety=6.0, use_builder=False):
    fx = FX.builder(name) if use_builder else FX.FIXTURES[name]
    presc, wl = fx['prescription'], fx['wavelength']
    N, dx, w0 = fx['N'], fx['dx'], fx['w0']
    rho_max = 0.5 * N * dx * np.sqrt(2.0) * 1.001

    # ---- production fields: J0 (the shared oracle's form) and ASM --------
    h, y, opl, amp, yl, P_in = OR.exit_field(presc, wl, w0, z, n_fan=n_fan)
    rho = np.linspace(0.0, rho_max, n_rho)
    t0 = time.time()
    E_j0 = OR.rs_j0(y, opl, amp, z, wl, rho)
    t_j0 = time.time() - t0
    A_j0 = OR.to_2d(rho, E_j0, N, dx)
    Nf4, Nf6 = _asm_grid(fx, 4), _asm_grid(fx, 6)
    t0 = time.time()
    A_asm = OR.asm_field(y, opl, amp, z, wl, N, dx, refine=4, Nf=Nf4)
    t_asm = time.time() - t0
    A_asm6 = OR.asm_field(y, opl, amp, z, wl, N, dx, refine=6, Nf=Nf6)
    h2, y2, o2, a2, _, _ = OR.exit_field(presc, wl, w0, z, n_fan=2 * n_fan)
    A_asmf = OR.asm_field(y2, o2, a2, z, wl, N, dx, refine=4, Nf=Nf4)

    # ---- the J0-vs-exact comparison, like for like ----------------------
    hc, yc, oc, ac, _, _ = OR.exit_field(presc, wl, w0, z, n_fan=n_fan_cmp)
    rc_ = np.linspace(0.0, rho_max, n_rho_cmp)
    Ec_j0 = OR.rs_j0(yc, oc, ac, z, wl, rc_)
    t0 = time.time()
    Ec_ex = OR.rs_exact(yc, oc, ac, z, wl, rc_, safety=safety)
    t_ex = time.time() - t0
    Ec_ex2 = OR.rs_exact(yc, oc, ac, z, wl, rc_, safety=2.0 * safety)
    hf, yf, of_, af, _, _ = OR.exit_field(presc, wl, w0, z,
                                          n_fan=2 * n_fan_cmp)
    Ec_exf = OR.rs_exact(yf, of_, af, z, wl, rc_, safety=safety)
    Ec_j0f = OR.rs_j0(yf, of_, af, z, wl, rc_)

    C_j0 = OR.to_2d(rc_, Ec_j0, N, dx)
    C_ex = OR.to_2d(rc_, Ec_ex, N, dx)
    C_exasm = A_asm      # the ASM arm is already exact; compare to it too

    ymax = float(np.max(np.abs(y)))
    k = 2.0 * np.pi / wl
    rms = float(np.sqrt(np.sum(np.abs(E_j0) ** 2 * rho ** 3)
                        / max(np.sum(np.abs(E_j0) ** 2 * rho), 1e-300)))
    R0 = np.sqrt(z * z + ymax * ymax + rms * rms)
    eps = float(k * ymax ** 2 * rms ** 2 / (2.0 * R0 ** 3))

    return dict(
        fixture=name, builder=use_builder, z_um=z * 1e6, N=N, dx_um=dx * 1e6,
        ymax_over_z=ymax / z, rms_radius_um=rms * 1e6, eps_rad=eps,
        n_fan=n_fan, n_fan_cmp=n_fan_cmp, n_rho_cmp=n_rho_cmp,
        Nf_refine4=Nf4, Nf_refine6=Nf6,
        t_j0_s=t_j0, t_exact_s=t_ex, t_asm_s=t_asm,
        # --- the claim: the J0 form against an exact azimuth ------------
        j0_vs_exact_relL2=OR.rel_l2(C_j0, C_ex),
        j0_vs_exact_fidelity=OR.fidelity(C_j0, C_ex),
        j0_vs_exact_radial_relL2=float(
            np.linalg.norm((Ec_j0 - Ec_ex) * np.sqrt(rc_))
            / max(np.linalg.norm(Ec_ex * np.sqrt(rc_)), 1e-300)),
        # --- the same claim from the other direction --------------------
        j0_vs_asm_relL2=OR.rel_l2(A_j0, A_asm),
        j0_vs_asm_fidelity=OR.fidelity(A_j0, A_asm),
        exact_vs_asm_relL2=OR.rel_l2(C_ex, C_exasm),
        exact_vs_asm_fidelity=OR.fidelity(C_ex, C_exasm),
        # --- convergence controls ---------------------------------------
        exact_safety_doubled_relL2=OR.rel_l2(OR.to_2d(rc_, Ec_ex2, N, dx),
                                             C_ex),
        exact_nfan_doubled_relL2=OR.rel_l2(OR.to_2d(rc_, Ec_exf, N, dx),
                                           C_ex),
        j0_nfan_doubled_relL2=OR.rel_l2(OR.to_2d(rc_, Ec_j0f, N, dx), C_j0),
        asm_refine_4_to_6_relL2=OR.rel_l2(A_asm6, A_asm),
        asm_nfan_doubled_relL2=OR.rel_l2(A_asmf, A_asm),
        # --- energy closure ---------------------------------------------
        power_j0_over_Pin=OR.power(A_j0, dx) / P_in,
        power_asm_over_Pin=OR.power(A_asm, dx) / P_in,
        P_in=P_in,
    )


def main():
    out = sys.argv[1]
    rows = []
    import lumenairy
    print('lumenairy.__file__ =', lumenairy.__file__, flush=True)
    i = 2
    while i < len(sys.argv):
        tok = sys.argv[i]
        use_b = tok.startswith('B:')
        name = tok[2:] if use_b else tok
        i += 1
        while i < len(sys.argv) and (sys.argv[i][0].isdigit()
                                     or sys.argv[i][0] == '.'):
            z = float(sys.argv[i]) * 1e-6
            i += 1
            r = run(name, z, use_builder=use_b)
            rows.append(r)
            print(json.dumps(r), flush=True)
            with open(out, 'w', encoding='cp1252') as f:
                json.dump(rows, f, indent=1)
    print('DONE', out, flush=True)


if __name__ == '__main__':
    main()
