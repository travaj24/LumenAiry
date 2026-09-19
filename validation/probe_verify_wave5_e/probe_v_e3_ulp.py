"""VERIFY-WAVE5-E / E3: the Maslov joint-scale ULP bar, re-derived and re-run.

Two fixtures, so the bar's two-sidedness is not a property of one cell:

  * ``pinned``  -- the fixture the pin itself uses (f/3.3 N-BK7 biconvex,
    96^2, P_x/P_y = 1.7777777777777779), so the recorded readings
    1.7777776632250892 and -6.4436e-08 can be checked;
  * ``mine``    -- a different biconvex (different radii, thickness, aperture,
    grid, wavelength and input ratio), so the bar is exercised somewhere the
    pin has never been run.

Per fixture it reports the three modes' ratios, the ULP deltas, the derived
bar broken into its two terms, and the INDEPENDENT-per-leg reconstruction (the
pre-fix behaviour).  Run once per ``OPENBLAS_CORETYPE`` x thread-count arm.
"""
import json
import math
import os
import sys
import warnings

import numpy as np

import lumenairy
from lumenairy import apply_real_lens_maslov_vector


def _spread(terms):
    v = np.asarray(terms, dtype=np.float64)
    flat = np.ascontiguousarray(v).ravel()
    exact = math.fsum(flat.tolist())
    trees = (float(np.sum(v)),
             float(np.sum(np.sum(v, axis=0))),
             float(np.sum(np.sum(v, axis=1))),
             float(np.sum(flat[::-1])),
             float(np.sum(np.ascontiguousarray(v.T))))
    return (max(abs(t - exact) for t in trees) / exact) if exact else 0.0


def _fixture_pinned():
    wl, N, dxg = 1.55e-6, 96, 16e-6
    xs = (np.arange(N) - N // 2) * dxg
    X, Y = np.meshgrid(xs, xs)
    amp = np.exp(-(X ** 2 + Y ** 2) / (0.55e-3) ** 2)
    E = np.stack([(0.8 * amp).astype(np.complex128),
                  (0.6 * amp).astype(np.complex128)], axis=0)
    pres = {'surfaces': [
        {'radius': 8e-3, 'conic': 0.0, 'glass_before': 'air',
         'glass_after': 'N-BK7'},
        {'radius': -8e-3, 'conic': 0.0, 'glass_before': 'N-BK7',
         'glass_after': 'air'}],
        'thicknesses': [2.5e-3], 'aperture_diameter': 2.4e-3}
    return E, dict(prescription=pres, wavelength=wl, dx=dxg,
                   integration_method='quadrature', n_v2=48, poly_order=4)


def _fixture_mine():
    wl, N, dxg = 1.03e-6, 128, 12e-6
    xs = (np.arange(N) - N // 2) * dxg
    X, Y = np.meshgrid(xs, xs)
    amp = np.exp(-(X ** 2 + Y ** 2) / (0.42e-3) ** 2)
    E = np.stack([(0.93 * amp).astype(np.complex128),
                  (0.37 * amp).astype(np.complex128)], axis=0)
    pres = {'surfaces': [
        {'radius': 6.5e-3, 'conic': 0.0, 'glass_before': 'air',
         'glass_after': 'N-SF11'},
        {'radius': -11.0e-3, 'conic': 0.0, 'glass_before': 'N-SF11',
         'glass_after': 'air'}],
        'thicknesses': [2.0e-3], 'aperture_diameter': 2.0e-3}
    return E, dict(prescription=pres, wavelength=wl, dx=dxg,
                   integration_method='quadrature', n_v2=48, poly_order=4)


def run(name, mk):
    E_vec, kw = mk()
    ratios, powers, terms, outs = {}, {}, {}, {}
    for mode in ('none', 'power', 'peak'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = apply_real_lens_maslov_vector(
                E_vec.copy(), normalize_output=mode, **kw)
        outs[mode] = out
        tx, ty = np.abs(out[0]) ** 2, np.abs(out[1]) ** 2
        px, py = float(np.sum(tx)), float(np.sum(ty))
        ratios[mode], powers[mode] = px / py, px + py
        terms[mode] = (tx, ty)
    r0 = ratios['none']
    ulp = float(np.spacing(r0))
    u = float(np.finfo(np.float64).eps) / 2.0
    product_rel = 6.0 * u
    reduction_rel = max(_spread(terms[m][0]) + _spread(terms[m][1])
                        for m in ('none', 'power', 'peak'))
    bar_ulp = 4.0 * (product_rel + reduction_rel) * r0 / ulp
    ox, oy = outs['none'][0], outs['none'][1]
    px0 = float(np.sum(np.abs(ox) ** 2))
    py0 = float(np.sum(np.abs(oy) ** 2))
    pxin = float(np.sum(np.abs(E_vec[0]) ** 2))
    pyin = float(np.sum(np.abs(E_vec[1]) ** 2))
    ix = ox * float(np.sqrt(pxin / px0))
    iy = oy * float(np.sqrt(pyin / py0))
    r_indep = (float(np.sum(np.abs(ix) ** 2))
               / float(np.sum(np.abs(iy) ** 2)))
    r_in = pxin / pyin
    return dict(
        fixture=name, r_none=r0, r_power=ratios['power'], r_peak=ratios['peak'],
        ulp=ulp, r_in=r_in,
        dev_from_input=r0 / r_in - 1.0,
        delta_power_ulp=abs(ratios['power'] - r0) / ulp,
        delta_peak_ulp=abs(ratios['peak'] - r0) / ulp,
        products_ulp=product_rel * r0 / ulp,
        reduction_ulp=reduction_rel * r0 / ulp,
        bar_ulp=bar_ulp,
        indep_ulp=abs(r_indep - r0) / ulp,
        indep_over_bar=(abs(r_indep - r0) / ulp) / bar_ulp,
        bar_over_widest_reading=bar_ulp / max(
            1e-300, abs(ratios['power'] - r0) / ulp,
            abs(ratios['peak'] - r0) / ulp),
        power_total=powers['power'], none_over_power=powers['none'] /
        powers['power'],
    )


def main():
    try:
        import threadpoolctl
        pools = [dict(prefix=p.get('prefix'), arch=p.get('architecture'),
                      threads=p.get('num_threads'),
                      version=p.get('version'))
                 for p in threadpoolctl.threadpool_info()]
    except Exception as exc:                              # pragma: no cover
        pools = [{'error': str(exc)}]
    out = dict(lumenairy_file=lumenairy.__file__, python=sys.version.split()[0],
               platform=sys.platform, numpy=np.__version__,
               coretype=os.environ.get('OPENBLAS_CORETYPE'),
               threads={k: os.environ.get(k) for k in
                        ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                         'MKL_NUM_THREADS')},
               threadpools=pools,
               results=[run('pinned', _fixture_pinned),
                        run('mine', _fixture_mine)])
    print(json.dumps(out, indent=1))


main()
