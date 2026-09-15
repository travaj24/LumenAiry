"""WAVE5-E item E3 -- re-derive the ULP bar of
``test_audit2609_a4_verify_maslov_asymptotic.py::
test_s10_vector_normalisation_is_one_joint_scale_for_the_pair``
from first principles, and record what every build reads.

THE PROPERTY.  ``apply_real_lens_maslov_vector``'s ``normalize_output`` must
apply ONE joint scale ``s`` to the (E_x, E_y) pair, so the polarization power
ratio ``P_x / P_y`` is untouched.

WHY THE SHIPPED BAR IS NOT DERIVED.  The test asserts ``<= 4 ULP`` with a
docstring origin of "0 ULP for 'power', 1 ULP for 'peak'", i.e. the bar is a
round number one decade above one process's reading.  Its stated derivation --
"``(a s)^2 / (b s)^2`` ... is exact up to the rounding of the two products" --
ignores the larger of the two rounding sources: each power is a SUM of
``N*N = 9216`` non-negative terms, and the scaled and unscaled sums do not
round the same way even though the terms differ by an exact common factor.

WHAT THIS PROBE MEASURES, per build and per BLAS kernel arm:

  1. the three modes' ratios and their ULP deltas from ``'none'``;
  2. the SUMMATION's own uncertainty on this data, measured at runtime by
     summing the same non-negative terms through five different orderings /
     algorithms (pairwise C order, pairwise over each axis, both reversals,
     and ``math.fsum`` as the exact reference) and taking the spread.  All
     terms are non-negative, so the sum is perfectly conditioned
     (``sum|t| / |sum t| = 1``) and this spread IS the rounding, not a
     cancellation artefact;
  3. the derived bar: the ratio's relative uncertainty is the sum of the two
     legs' relative summation uncertainties, converted to ULP of the ratio by
     ``r0 / np.spacing(r0)``;
  4. the two-sided arm: what an INDEPENDENT per-leg normalisation (the pre-fix
     defect) does to the same ratio, in the same ULP units.

Usage:  python probe_e3_maslov_ulp.py <out.json>
"""
from __future__ import annotations

import json
import math
import os
import sys
import warnings

import numpy as np


def _sum_variants(v):
    """The same non-negative terms summed five ways, plus the exact value.

    ``np.sum`` is pairwise, so reversing the array and reducing per axis give
    genuinely different summation TREES over the identical multiset of terms.
    Their spread about ``math.fsum`` is the summation's own rounding on this
    data, measured rather than bounded.
    """
    flat = np.ascontiguousarray(v).ravel()
    exact = math.fsum(flat.tolist())
    variants = {
        'pairwise_c': float(np.sum(v)),
        'pairwise_axis0_then_1': float(np.sum(np.sum(v, axis=0))),
        'pairwise_axis1_then_0': float(np.sum(np.sum(v, axis=1))),
        'pairwise_reversed': float(np.sum(flat[::-1])),
        'pairwise_transposed': float(np.sum(np.ascontiguousarray(v.T))),
    }
    spread = max(abs(x - exact) for x in variants.values())
    return exact, variants, spread


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'e3_maslov_ulp.json'
    import lumenairy as la
    from lumenairy.elements.lenses_maslov import apply_real_lens_maslov_vector

    res = {'lumenairy_file': la.__file__, 'python': sys.version.split()[0],
           'numpy': np.__version__, 'platform': sys.platform,
           'env': {k: os.environ.get(k) for k in
                   ('OPENBLAS_CORETYPE', 'OPENBLAS_NUM_THREADS',
                    'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}}
    print('lumenairy.__file__ =', la.__file__, flush=True)
    print('OPENBLAS_CORETYPE =', os.environ.get('OPENBLAS_CORETYPE'),
          ' OPENBLAS_NUM_THREADS =', os.environ.get('OPENBLAS_NUM_THREADS'),
          flush=True)
    try:
        import threadpoolctl
        res['threadpool'] = [
            {k: d.get(k) for k in ('user_api', 'internal_api', 'architecture',
                                   'num_threads', 'version')}
            for d in threadpoolctl.threadpool_info()]
        for d in res['threadpool']:
            print('  threadpool:', d, flush=True)
    except Exception as exc:                                # pragma: no cover
        res['threadpool'] = 'unavailable: %s' % exc

    # ---- the fixture, verbatim from the test file -------------------------
    wl, N, dxg = 1.55e-6, 96, 16e-6
    xs = (np.arange(N) - N // 2) * dxg
    X, Y = np.meshgrid(xs, xs)
    amp = np.exp(-(X ** 2 + Y ** 2) / (0.55e-3) ** 2)
    E_vec = np.stack([(0.8 * amp).astype(np.complex128),
                      (0.6 * amp).astype(np.complex128)], axis=0)
    pres = {'surfaces': [
        {'radius': 8e-3, 'conic': 0.0, 'glass_before': 'air',
         'glass_after': 'N-BK7'},
        {'radius': -8e-3, 'conic': 0.0, 'glass_before': 'N-BK7',
         'glass_after': 'air'}],
        'thicknesses': [2.5e-3], 'aperture_diameter': 2.4e-3}
    kw = dict(prescription=pres, wavelength=wl, dx=dxg,
              integration_method='quadrature', n_v2=48, poly_order=4)

    res['N'] = N
    res['n_terms_per_sum'] = int(N * N)
    res['modes'] = {}
    outs = {}
    for mode in ('none', 'power', 'peak'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = apply_real_lens_maslov_vector(
                E_vec.copy(), normalize_output=mode, **kw)
        outs[mode] = out
        tx = np.abs(out[0]) ** 2
        ty = np.abs(out[1]) ** 2
        px, py = float(np.sum(tx)), float(np.sum(ty))
        ex, vx, sx = _sum_variants(tx)
        ey, vy, sy = _sum_variants(ty)
        res['modes'][mode] = {
            'px': px, 'py': py, 'ratio': px / py, 'power': px + py,
            'px_exact_fsum': ex, 'py_exact_fsum': ey,
            'px_variants': vx, 'py_variants': vy,
            'px_spread_abs': sx, 'py_spread_abs': sy,
            'px_spread_rel': sx / ex if ex else None,
            'py_spread_rel': sy / ey if ey else None,
        }
        print('mode=%-6s ratio=%.17g  px=%.17g py=%.17g  '
              'sum-spread rel x=%.3e y=%.3e'
              % (mode, px / py, px, py,
                 res['modes'][mode]['px_spread_rel'],
                 res['modes'][mode]['py_spread_rel']), flush=True)

    r0 = res['modes']['none']['ratio']
    ulp = float(np.spacing(r0))
    res['r0'] = r0
    res['spacing_r0'] = ulp
    res['readings_ulp'] = {}
    for mode in ('power', 'peak'):
        d = abs(res['modes'][mode]['ratio'] - r0) / ulp
        res['readings_ulp'][mode] = d
        print('reading: %-6s %.3f ULP' % (mode, d), flush=True)

    # ---- the derived bar --------------------------------------------------
    # The ratio's relative uncertainty is the sum of the two legs' own
    # summation uncertainties (they are independent reductions), and the
    # per-element product rounding rides inside them because the variants are
    # taken on the ALREADY SCALED terms.  Convert to ULP of the ratio.
    worst_rel = 0.0
    for mode in ('none', 'power', 'peak'):
        m = res['modes'][mode]
        worst_rel = max(worst_rel, m['px_spread_rel'] + m['py_spread_rel'])
    res['summation_rel_uncertainty'] = worst_rel
    res['derived_bar_ulp_raw'] = worst_rel * r0 / ulp
    print('derived summation uncertainty: rel %.4e  = %.3f ULP of the ratio'
          % (worst_rel, res['derived_bar_ulp_raw']), flush=True)

    # ---- the two-sided arm: an INDEPENDENT per-leg scale -------------------
    # The pre-fix defect normalised each scalar leg on its own.  Reproduced
    # here on the 'none' output, so the arm is engineered from the running
    # build's own numbers rather than quoted.
    ox, oy = outs['none'][0], outs['none'][1]
    px0 = float(np.sum(np.abs(ox) ** 2))
    py0 = float(np.sum(np.abs(oy) ** 2))
    pxin = float(np.sum(np.abs(E_vec[0]) ** 2))
    pyin = float(np.sum(np.abs(E_vec[1]) ** 2))
    ix = ox * float(np.sqrt(pxin / px0))
    iy = oy * float(np.sqrt(pyin / py0))
    r_indep = (float(np.sum(np.abs(ix) ** 2))
               / float(np.sum(np.abs(iy) ** 2)))
    res['independent_per_leg'] = {
        'ratio': r_indep,
        'delta_ulp': abs(r_indep - r0) / ulp,
        'input_ratio': pxin / pyin,
        'r0_vs_input_rel': abs(r0 / (pxin / pyin) - 1.0)}
    print('two-sided arm: independent per-leg normalisation moves the ratio '
          'by %.4e ULP (to %.17g, the input ratio %.17g)'
          % (res['independent_per_leg']['delta_ulp'], r_indep, pxin / pyin),
          flush=True)
    res['decades_of_gap'] = (
        math.log10(res['independent_per_leg']['delta_ulp']
                   / max(res['derived_bar_ulp_raw'], 1e-30))
        if res['independent_per_leg']['delta_ulp'] > 0 else None)
    print('decades between the derived bar and the real signal: %.2f'
          % res['decades_of_gap'], flush=True)

    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(res, fh, indent=1)
    print('wrote', out_path, flush=True)


if __name__ == '__main__':
    main()
