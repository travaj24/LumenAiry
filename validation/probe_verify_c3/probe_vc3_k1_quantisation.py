"""VERIFY-WP-C3 -- is ``_collins_readout_k1 <= 1`` a decision on a CONTINUOUS
quantity, or on a STAIRCASE?

``_collins_readout_k1`` is
    K1 = 2 dx (|A| r / |B|) / lambda   +   2 dx theta / lambda
and both ``r`` and ``theta`` come from ``_collins_containment_radius``, which
returns ``d[order[i]]`` -- a GRID COORDINATE, with no interpolation.  So

  * theta is quantised in units of ``lambda/(N dx)``  => the SECOND term is
    quantised in steps of exactly ``2/N`` and lives in [0, 1] (it is the
    envelope's own angular fill against the grid Nyquist, and the outermost
    ``fftfreq`` bin is exactly ``1/(2 dx)``);
  * r is quantised in units of ``dx``                 => the FIRST term is
    quantised in steps of ``2 dx^2 |A| / (lambda |B|)``.

and the index ``i`` is a ``searchsorted`` of a CUMULATIVE SUM of
``|FFT(env)|^2`` against a fixed ``1 - _COLLINS_TAIL_FRAC`` threshold -- a
large reduction compared to a fixed bar, which is shape S4/S5 of
``docs/TESTING_STANDARDS.md``.

This probe measures the step size, shows the staircase, and shows that a
perturbation FAR below any physical significance moves K1 by a whole step and
FLIPS THE ROUTE.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import lumenairy  # noqa: E402
from lumenairy.propagators import carrier as C  # noqa: E402

TREE = os.path.abspath(os.environ.get('VC3_TREE', os.getcwd()))
assert os.path.abspath(lumenairy.__file__).startswith(TREE), (
    lumenairy.__file__, TREE)

LAM = 1.064e-6


def gauss(n, dx, w):
    x = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(x, x, indexing='ij')
    return np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)


def terms(env, R, z, dx):
    Rx, Ry, _ = C._parse_carrier(R, 'p')
    A, B, _, _ = C._collins_envelope_abcd(Rx, z, np.inf)
    r, _, th, _ = C._collins_input_box(env, dx, dx, LAM, C._COLLINS_TAIL_FRAC)
    t_sp = 2.0 * dx * abs(A) * r / abs(B) / LAM
    t_an = 2.0 * dx * th / LAM
    return dict(A=A, B=B, r=r, theta=th, space_term=t_sp, angle_term=t_an,
                k1_sum=t_sp + t_an,
                k1_lib=C._collins_readout_k1(env, R, z, LAM, dx, dx))


def main():
    out = {'lumenairy': lumenairy.__file__, 'python': sys.version.split()[0],
           'numpy': np.__version__}

    # ---- 1. the ANGLE term is bounded by 1 and quantised in steps of 2/N ----
    N, dx = 512, 4.0e-6
    R, z = -40e-3, 6.0e-3
    lad = []
    for w in np.linspace(60e-6, 400e-6, 341):
        t = terms(gauss(N, dx, float(w)), R, z, dx)
        lad.append(dict(w_um=float(w) * 1e6, angle_term=t['angle_term'],
                        space_term=t['space_term'], k1=t['k1_lib']))
    ang = np.array([e['angle_term'] for e in lad])
    uniq = np.unique(np.round(ang * N / 2.0, 9))
    out['angle_term_is_quantised'] = dict(
        N=N, step_expected=2.0 / N,
        all_terms_are_integer_multiples_of_step=bool(
            np.allclose(uniq, np.round(uniq), atol=1e-9)),
        max_angle_term=float(ang.max()), n_distinct=int(uniq.size),
        distinct_j_values=[float(v) for v in uniq[:25]])

    # ---- 2. the STAIRCASE at the K1 = 1 boundary -------------------------
    k1 = np.array([e['k1'] for e in lad])
    w_um = np.array([e['w_um'] for e in lad])
    cross = np.where(np.diff(np.sign(k1 - 1.0)) != 0)[0]
    steps = np.abs(np.diff(k1))
    steps = steps[steps > 1e-12]
    out['staircase'] = dict(
        k1_min=float(k1.min()), k1_max=float(k1.max()),
        n_crossings_of_1=int(cross.size),
        smallest_nonzero_jump=float(steps.min()) if steps.size else None,
        jump_at_each_crossing=[
            dict(w_um_lo=float(w_um[i]), w_um_hi=float(w_um[i + 1]),
                 k1_lo=float(k1[i]), k1_hi=float(k1[i + 1]),
                 jump=float(k1[i + 1] - k1[i])) for i in cross],
        ladder=lad[::10])

    # ---- 3. THE DECISION FLIPS ON A PERTURBATION FAR BELOW SIGNIFICANCE ---
    # find a w whose K1 sits just BELOW 1, then perturb the envelope at a
    # relative level no physical model could care about, and re-read K1.
    below = [e for e in lad if e['k1'] <= 1.0]
    flips = []
    if below:
        e0 = max(below, key=lambda e: e['k1'])
        w0 = e0['w_um'] * 1e-6
        env0 = gauss(N, dx, w0)
        rng = np.random.default_rng(20260920)
        for lvl in (1e-6, 1e-8, 1e-10, 1e-12, 1e-14):
            pert = env0 * (1.0 + lvl * rng.standard_normal(env0.shape))
            t = terms(pert, R, z, dx)
            flips.append(dict(
                rel_perturbation=lvl, k1_before=e0['k1'], k1_after=t['k1_lib'],
                delta=t['k1_after'] if False else t['k1_lib'] - e0['k1'],
                route_before='collins' if e0['k1'] <= 1.0 else 'sziklas',
                route_after='collins' if t['k1_lib'] <= 1.0 else 'sziklas'))
        out['perturbation_flips'] = dict(
            w_um=e0['w_um'], k1_before=e0['k1'],
            margin_below_1=1.0 - e0['k1'],
            quantisation_step=2.0 / N,
            margin_over_step=(1.0 - e0['k1']) / (2.0 / N),
            rows=flips)

    # ---- 4. the same margin arithmetic for the SHIPPED design-121 reading -
    out['design121_margin_arithmetic'] = {
        'reported_k1_N1024': 0.9995812250283026,
        'margin_below_1': 1.0 - 0.9995812250283026,
        'angle_quantisation_step_at_N1024': 2.0 / 1024,
        'margin_in_units_of_one_angle_bin':
            (1.0 - 0.9995812250283026) / (2.0 / 1024),
        'note': 'a margin smaller than one quantisation step of the quantity '
                'the threshold is taken on'}

    tag = os.environ.get('VC3_TAG', 'x')
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     f'k1_quantisation_{tag}.json')
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'staircase'},
                     indent=1))
    print('staircase:', json.dumps(
        {k: v for k, v in out['staircase'].items() if k != 'ladder'}, indent=1))
    print('WROTE', p)


if __name__ == '__main__':
    main()
