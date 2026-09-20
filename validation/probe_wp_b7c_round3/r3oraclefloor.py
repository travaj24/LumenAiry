"""E6 -- the oracle floor measured at FULL RADIUS, and the scoring arm's own
bracket.

Two separate questions, deliberately not mixed:

**(a) how good is the arm this round SCORES with.**  Every fidelity in the
round-3 population is taken against the band-limited angular spectrum, which
is an exact solution of the scalar Helmholtz equation for the traced exit
field -- no Debye, paraxial or azimuthal approximation at any radius.  Its own
error is bounded here by an INDEPENDENT integral: the exact azimuthal
quadrature of the Rayleigh-Sommerfeld integral, applied at every radius out to
the grid corner, plus each arm's own convergence control.

**(b) what the round-2 oracle-floor table's "Debye vs exact" column says when
the exact arm is not confined to the core.**  Round 2 published
``rel L2 ~ 0.37 eps`` "holding over three decades of eps" from a column whose
exact substitution ran only inside the 99.95 %-energy radius, outside which
the two compared fields are identically equal by construction -- while the
dropped quadratic azimuthal term GROWS with radius.  The same rows are
re-measured here with the substitution everywhere.

**Why the comparison is taken at PIXEL radii and not on a radial profile.**
Reconstructing a 2-D field from a radial profile interpolates, and near a
caustic the radial fringe spacing falls to a few microns, so a profile coarse
enough to afford the exact quadrature aliases -- which shows up as an energy
closure of 0.89 on BOTH radial arms while the angular spectrum closes at
0.999.  That is the reconstruction's error, not the propagator's.  So every
comparison here is taken POINTWISE at a stratified sample of the output grid's
own pixels: the exact quadrature is evaluated at those pixels' radii and the
angular spectrum is read at those pixels, with no interpolation on either
side.

The sample is IMPORTANCE-WEIGHTED, because these fields put almost all of
their energy in a fraction of a percent of the grid's area: pixels are
stratified into twenty-four bins of equal RADIUS (not of equal count), an
equal number is drawn from each, and each draw carries the weight
``N_bin / n_drawn``.  That makes the sampled sums unbiased estimators of the
2-D sums while giving the bright core its own bins -- a uniform pixel sample
would put three or four samples in the core and estimate the L2 norm from
them.

Usage:
    python r3oraclefloor.py <out.json> [optic:z_um ...]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import r3fixtures as FX
import r3oracle as OR
import r3scan as SC

#: (optic, z [um]) -- one plane per optic across the whole NA axis, including
#: the two the round-2 report and its verification each published a number for
#: (``Q`` z = 5680, ``Y`` z = 1980) so the three measurements can be put side
#: by side.
DEFAULT = ['Q:5680', 'S:3214.78', 'W:4900', 'M:2201.74', 'Y:1980',
           'F_alt:1056', 'G:950', 'P:888', 'X:960', 'HN:1900', 'AS:3000',
           'MC:2600']


def _sample_pixels(N, dx, n_want=480, n_bins=24, seed=20260919):
    """A radius-stratified, importance-weighted sample of the output grid's
    pixels, with the weight that makes a sampled sum an unbiased estimator of
    the whole-grid sum."""
    rng = np.random.default_rng(seed)
    x = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X * X + Y * Y).ravel()
    edges = np.linspace(0.0, float(R.max()) * (1.0 + 1e-12), n_bins + 1)
    per = max(1, n_want // n_bins)
    idx, wts = [], []
    for b in range(n_bins):
        inb = np.nonzero((R >= edges[b]) & (R < edges[b + 1]))[0]
        if inb.size == 0:
            continue
        m = min(per, inb.size)
        pick = rng.choice(inb, size=m, replace=False)
        idx.extend(pick.tolist())
        wts.extend([inb.size / float(m)] * m)
    idx = np.asarray(idx, dtype=np.int64)
    w = np.asarray(wts, dtype=float)
    order = np.argsort(R[idx])
    idx, w = idx[order], w[order]
    return idx, R[idx], w


def _wrel_l2(a, b, w):
    a, b, w = np.asarray(a), np.asarray(b), np.asarray(w, dtype=float)
    num = float(np.sum(w * np.abs(a - b) ** 2))
    den = float(np.sum(w * np.abs(b) ** 2))
    return float(np.sqrt(num / max(den, 1e-300)))


def _wfid(a, b, w):
    a, b, w = np.asarray(a), np.asarray(b), np.asarray(w, dtype=float)
    num = abs(complex(np.sum(w * np.conj(a) * b)))
    den = np.sqrt(float(np.sum(w * np.abs(a) ** 2))
                  * float(np.sum(w * np.abs(b) ** 2)))
    return float(num / max(den, 1e-300))


def one(fx, z, n_fan=6000, n_sample=480, n_ctrl=48, safety=6.0, refine=4,
        n_rho_profile=4000):
    wl, N, dx, w0 = fx['wavelength'], fx['N'], fx['dx'], fx['w0']
    h, y, opl, amp, yl, P_in = OR.exit_field(fx['prescription'], wl, w0, z,
                                             n_fan=n_fan)
    idx, rr, w = _sample_pixels(N, dx, n_sample)

    t0 = time.time()
    s_j0 = OR.rs_j0(y, opl, amp, z, wl, rr)
    t_j0 = time.time() - t0
    t0 = time.time()
    s_ex = OR.rs_exact(y, opl, amp, z, wl, rr, safety=safety)
    t_ex = time.time() - t0
    t0 = time.time()
    E_asm = OR.asm_field(y, opl, amp, z, wl, N, dx, refine=refine,
                         Nf=SC._asm_grid(fx, refine))
    t_asm = time.time() - t0
    s_asm = np.asarray(E_asm).ravel()[idx]

    # the round-2 column, reproduced on the SAME sample: exact only inside the
    # 99.95 %-energy core, ``J0`` outside, where the two are equal by
    # construction
    rho_p = np.linspace(0.0, float(rr.max()), 1200)
    E_p = OR.rs_j0(y, opl, amp, z, wl, rho_p)
    rc = OR.core_radius(rho_p, E_p, frac=0.9995)
    s_core = np.where(rr <= rc, s_ex, s_j0)

    # convergence controls, on a coarse sub-sample: what they bound is each
    # quadrature's own error, not the sample's resolution
    sub = np.linspace(0, rr.size - 1, min(n_ctrl, rr.size)).astype(np.int64)
    rc_r, wc_r = rr[sub], w[sub]
    e_a = OR.rs_exact(y, opl, amp, z, wl, rc_r, safety=safety)
    e_b = OR.rs_exact(y, opl, amp, z, wl, rc_r, safety=2.0 * safety)
    h2, y2, opl2, amp2, _, _ = OR.exit_field(fx['prescription'], wl, w0, z,
                                             n_fan=2 * n_fan)
    e_c = OR.rs_exact(y2, opl2, amp2, z, wl, rc_r, safety=safety)
    E_asm6 = OR.asm_field(y, opl, amp, z, wl, N, dx, refine=refine + 2,
                          Nf=SC._asm_grid(fx, refine + 2))
    E_asm_rings = OR.asm_field(y2, opl2, amp2, z, wl, N, dx, refine=refine,
                               Nf=SC._asm_grid(fx, refine))

    ym = float(np.max(np.abs(y)))
    wp = np.abs(E_p) ** 2 * rho_p
    rms = float(np.sqrt(max(np.trapezoid(wp * rho_p * rho_p, rho_p), 0.0)
                        / max(np.trapezoid(wp, rho_p), 1e-300)))
    R0 = float(np.sqrt(z * z + ym * ym + rms * rms))
    eps = float((2.0 * np.pi / wl) * ym * ym * rms * rms / (2.0 * R0 ** 3))
    # the J0 arm's own ENERGY CLOSURE, from a well-resolved radial profile
    prof = np.linspace(0.0, float(rr.max()), n_rho_profile)
    E_prof = OR.rs_j0(y, opl, amp, z, wl, prof)
    j0_energy = float(2.0 * np.pi * np.trapezoid(np.abs(E_prof) ** 2 * prof,
                                                 prof))
    return dict(
        fixture=fx['name'], z_um=z * 1e6, N=N, dx_um=dx * 1e6, n_sample=rr.size,
        y_max=ym, y_max_over_z=ym / z, rms_radius=rms, eps=eps,
        core_radius=rc, rho_max=float(rr.max()), exact_n_fan=n_fan,
        j0_vs_exact_FULL_relL2=_wrel_l2(s_j0, s_ex, w),
        j0_vs_exact_FULL_fidelity=_wfid(s_j0, s_ex, w),
        j0_vs_exact_CORE_ONLY_relL2=_wrel_l2(s_j0, s_core, w),
        asm_vs_exact_relL2=_wrel_l2(s_asm, s_ex, w),
        asm_vs_exact_fidelity=_wfid(s_asm, s_ex, w),
        j0_vs_asm_relL2=_wrel_l2(s_j0, s_asm, w),
        exact_conv_safety_x2=_wrel_l2(e_b, e_a, wc_r),
        exact_conv_rings_x2=_wrel_l2(e_c, e_a, wc_r),
        asm_conv_refine_p2=OR.rel_l2(E_asm6, E_asm),
        asm_conv_rings_x2=OR.rel_l2(E_asm_rings, E_asm),
        P_in=P_in,
        j0_closure=j0_energy / max(P_in, 1e-300),
        asm_closure=OR.power(E_asm, dx) / max(P_in, 1e-300),
        seconds=dict(j0=t_j0, exact=t_ex, asm=t_asm))


def main():
    out = sys.argv[1]
    items = sys.argv[2:] or DEFAULT
    rows = []
    for it in items:
        nm, zs = it.split(':')
        fx = FX.FIXTURES[nm]
        try:
            r = one(fx, float(zs) * 1e-6)
        except Exception as exc:                            # noqa: BLE001
            r = dict(fixture=nm, z_um=float(zs),
                     error=f'{type(exc).__name__}: {exc}')
        rows.append(r)
        print(json.dumps(r, default=float), flush=True)
    ok = [r for r in rows if 'error' not in r]
    summary = dict(
        n=len(ok),
        asm_vs_exact_relL2_range=([min(r['asm_vs_exact_relL2'] for r in ok),
                                   max(r['asm_vs_exact_relL2'] for r in ok)]
                                  if ok else None),
        asm_vs_exact_fidelity_min=min((r['asm_vs_exact_fidelity']
                                       for r in ok), default=None))
    with open(out, 'w', encoding='cp1252') as f:
        json.dump(dict(summary=summary, rows=rows), f, indent=1, default=float)
    print(json.dumps(summary, default=float))
    print('wrote', out)


if __name__ == '__main__':
    main()
