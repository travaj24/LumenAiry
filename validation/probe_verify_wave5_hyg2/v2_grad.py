"""(6) THE GRADIENT LADDER, on MY merit and MY ladder.

MY merit is NOT the author's transported power.  It is the single-point
intensity at the output origin,

    M(a) = |E_out[N/2, N/2]|^2,   a real,

which has a CLOSED-FORM gradient on this transport and is not power-
conserving, so the author's "proportional to the input amplitude" shape
check cannot pass here by accident.  At the output origin the post-chirp and
the chirp-Z phase are both 1, so

    E_out(0,0) = pref * sum_ij a_ij chirp(u_i) chirp(v_j),
    pref = exp(i k B) dx dy / (i lambda B),
    chirp(u) = exp(i k u^2 / (2 B/A)),

hence  dM/da_ij = 2 Re( conj(E_out(0,0)) * pref * chirp(u_i) chirp(v_j) ) --
a Fresnel chirp pattern, built here from plain NumPy and not from the
module's own helpers.

MY floor, derived: a central difference carries truncation ~|P'''| h^2/6 and
round-off ~eps|P|/h; the sum is minimised at h ~ (3 eps |P| / |P'''|)^(1/3),
where the ABSOLUTE error is ~ (eps|P|)^(2/3) |P'''|^(1/3) / 2 and the error
RELATIVE to P' is ~ eps^(2/3) up to the (P, P', P''') scale factors, which
are measured here rather than assumed.

    python v2_grad.py <tree> <out.json>
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                            # noqa: E402
from vlib import anchor, build_tag, write_json                 # noqa: E402

WL, N, DX = 1.064e-6, 96, 5.5e-6
R_IN, Z, R_REF, W0 = -0.028, 2.3e-3, -0.0215, 41e-6
LADDER = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)


def main():
    tree, out_path = sys.argv[1], sys.argv[2]
    anchor(tree)
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import lumenairy.propagators.carrier as CA

    ax = (np.arange(N) - N // 2) * DX
    X, Y = np.meshgrid(ax, ax)
    a0 = np.exp(-(X ** 2 + Y ** 2) / (W0 * W0))
    amp0 = jnp.asarray(a0)
    res = {'build': build_tag(), 'jax_version': jax.__version__,
           'merit': 'single-point output intensity |E_out[N/2,N/2]|^2',
           'ladder': list(LADDER)}

    KW = dict(dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N, R_ref=R_REF,
              gap_kernel='fresnel', on_collins_sampling='ignore')

    def merit(a):
        out = CA._collins_transport(a.astype(jnp.complex128), R_IN, Z, WL,
                                    DX, DX, **KW)
        c = out[N // 2, N // 2]
        return jnp.real(c) ** 2 + jnp.imag(c) ** 2

    g = np.asarray(jax.grad(merit)(amp0))
    res['grad_finite'] = bool(np.all(np.isfinite(g)))
    res['grad_max_abs'] = float(np.max(np.abs(g)))
    res['grad_std_over_max'] = float(np.std(g) / np.max(np.abs(g)))
    res['falsification_not_zero'] = bool(np.max(np.abs(g)) > 0.0)
    res['falsification_not_constant'] = bool(
        np.std(g) / np.max(np.abs(g)) > 1e-3)

    P0 = float(merit(amp0))
    res['P0'] = P0

    # ---- the ladder, on FOUR pixels (peak, mid, tail, off-diagonal) -----
    pts = {'peak': (N // 2, N // 2),
           'mid': (N // 2 + 5, N // 2 - 3),
           'tail': (N // 2 + 14, N // 2 + 14),
           'offdiag': (N // 2 - 9, N // 2 + 2)}
    eps = float(np.finfo(np.float64).eps)
    ladders = {}
    for pname, ij in pts.items():
        rows = []
        for h in LADDER:
            def at(sign):
                ap = a0.copy()
                ap[ij] += sign * h
                return float(merit(jnp.asarray(ap)))
            fp, fm = at(+1), at(-1)
            fd = (fp - fm) / (2.0 * h)
            rel = abs(fd - g[ij]) / abs(g[ij])
            rows.append({'h': h, 'fd': fd, 'rel': rel})
        best = min(rows, key=lambda r: r['rel'])
        # is the U-curve's upturn visible?  (the smallest h must be WORSE
        # than the best, by a clear margin)
        upturn = rows[-1]['rel'] > 10.0 * best['rel']
        coarse_up = rows[0]['rel'] > 2.0 * best['rel']
        ladders[pname] = {
            'ij': list(ij), 'grad': float(g[ij]), 'rows': rows,
            'best_rel': best['rel'], 'best_h': best['h'],
            'upturn_at_small_h_visible': bool(upturn),
            'truncation_arm_visible_at_large_h': bool(coarse_up),
            'rel_at_smallest_h': rows[-1]['rel'],
            'rel_at_largest_h': rows[0]['rel']}
    res['ladders'] = ladders

    # ---- MY floor, from the measured P, P', P''' -------------------------
    ij = pts['peak']
    hq = 1e-3

    def f_at(d):
        ap = a0.copy()
        ap[ij] += d
        return float(merit(jnp.asarray(ap)))
    # third derivative by a five-point stencil
    d3 = ((-0.5) * f_at(-2 * hq) + f_at(-hq) - f_at(hq)
          + 0.5 * f_at(2 * hq)) / (hq ** 3)
    res['P_third_derivative'] = d3
    h_opt = (3.0 * eps * abs(P0) / max(abs(d3), 1e-300)) ** (1.0 / 3.0)
    abs_floor = (abs(d3) * h_opt ** 2 / 6.0
                 + eps * abs(P0) / max(h_opt, 1e-300))
    rel_floor = abs_floor / abs(g[ij])
    res['floor'] = {
        'eps_two_thirds': eps ** (2.0 / 3.0),
        'h_opt_derived': h_opt,
        'abs_floor_derived': abs_floor,
        'rel_floor_derived': rel_floor,
        'bar_10x_rel_floor': 10.0 * rel_floor,
        'bar_10x_eps23': 10.0 * eps ** (2.0 / 3.0)}
    res['best_over_all_pixels'] = min(v['best_rel'] for v in ladders.values())
    res['factor_inside_derived_bar'] = (
        10.0 * rel_floor / res['best_over_all_pixels'])
    res['factor_inside_eps23_bar'] = (
        10.0 * eps ** (2.0 / 3.0) / res['best_over_all_pixels'])
    res['inside_derived_bar'] = bool(
        res['best_over_all_pixels'] < 10.0 * rel_floor)

    # ---- a merit with a NON-ZERO third derivative, so the FULL U shows --
    # M(a) = |out_c|^2 is exactly QUADRATIC in a single real pixel (out is
    # linear in a), so its central difference has ZERO truncation error and
    # only the round-off arm of the U exists.  The cube has a real P''',
    # so the truncation arm must appear at large h -- which is what makes the
    # ladder a two-armed instrument rather than a one-armed one.
    def merit3(a):
        out = CA._collins_transport(a.astype(jnp.complex128), R_IN, Z, WL,
                                    DX, DX, **KW)
        c = out[N // 2, N // 2]
        return (jnp.real(c) ** 2 + jnp.imag(c) ** 2) ** 3

    g3 = np.asarray(jax.grad(merit3)(amp0))
    ij3 = pts['peak']
    rows3 = []
    for h in LADDER:
        def at3(sign):
            ap = a0.copy()
            ap[ij3] += sign * h
            return float(merit3(jnp.asarray(ap)))
        fd = (at3(+1) - at3(-1)) / (2.0 * h)
        rows3.append({'h': h, 'fd': fd,
                      'rel': abs(fd - g3[ij3]) / abs(g3[ij3])})
    best3 = min(rows3, key=lambda r: r['rel'])
    res['cubic_merit_ladder'] = {
        'grad': float(g3[ij3]), 'rows': rows3,
        'best_rel': best3['rel'], 'best_h': best3['h'],
        'rel_at_largest_h': rows3[0]['rel'],
        'rel_at_smallest_h': rows3[-1]['rel'],
        'truncation_arm_visible': bool(rows3[0]['rel'] > 10.0
                                       * best3['rel']),
        'roundoff_arm_visible': bool(rows3[-1]['rel'] > 10.0
                                     * best3['rel']),
        'full_U_visible': bool(rows3[0]['rel'] > 10.0 * best3['rel']
                               and rows3[-1]['rel'] > 10.0 * best3['rel'])}

    # ---- the ANALYTIC shape of the gradient ------------------------------
    A, B, _C, D = CA._collins_envelope_abcd(R_IN, Z, R_REF)
    k = 2.0 * np.pi / WL
    u = (np.arange(N, dtype=np.float64) - N / 2.0) * DX
    chirp = np.exp(1j * k * (u * u) / (2.0 * (B / A)))
    pref = np.exp(1j * k * B) * DX * DX / (1j * WL * B)
    out = np.asarray(CA._collins_transport(
        a0.astype(np.complex128), R_IN, Z, WL, DX, DX, **KW))
    c = complex(out[N // 2, N // 2])
    K = pref * chirp[None, :] * chirp[:, None]
    g_analytic = 2.0 * np.real(np.conj(c) * K)
    res['analytic_shape'] = {
        'abcd': [A, B, _C, D],
        'max_abs_diff': float(np.max(np.abs(g - g_analytic))),
        'rel_l2': float(np.linalg.norm(g - g_analytic)
                        / np.linalg.norm(g_analytic)),
        'correlation': float(np.corrcoef(g.ravel(),
                                         g_analytic.ravel())[0, 1]),
        'is_chirp_not_amplitude': float(np.corrcoef(
            g.ravel(), a0.ravel())[0, 1]),
        'centre_value_check': [float(g[ij]), float(g_analytic[ij])]}

    write_json(res, out_path)
    printable = dict(res)
    printable['ladders'] = {kk: {k2: v2 for k2, v2 in vv.items()}
                            for kk, vv in ladders.items()}
    print(json.dumps(printable, indent=1, default=str))


if __name__ == '__main__':
    main()
