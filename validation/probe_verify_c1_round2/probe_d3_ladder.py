"""VERIFY-WP-C1 ROUND 2 -- D3: the convergence ladder, re-run twice.

Round 2 restates the doc claim as "the hard edge is first order at best and its
step orders are erratic" and backs it with VERIFY-C1's optic (lambda = 1064 nm,
a = 62.5 um, window 400 um, z = 4.0 mm RS / 2.5 mm HF), claiming hard step
orders 1.680 / 0.203 / 1.359 (RS), mean 1.0804, grey mean 1.9369, and HF hard
mean 1.0709 (a correction of the 1.06 printed in VERIFY_WP-C1.md sec. 2.1).

This probe:

1. RE-MEASURES that optic with its own code, to check the four digits;
2. runs a THIRD optic (lambda = 532 nm, a = 150 um, window 900 um,
   z = 30 mm RS / 15 mm HF -- a different wavelength, radius, window and
   Fresnel number again) to test whether "first order at best, erratic steps"
   is a general statement or two optics' behaviour.

Oracle is the closed form on axis behind a circular aperture,

    U(0,0,z) = exp(i k z) - (z/r_a) exp(i k r_a),   r_a = sqrt(z^2 + a^2)

which depends on no discretisation.  The RS spatial kernel's alias threshold
2 W^2 / (N lambda) is largest at the coarsest N; the probe RECORDS the check
at every N rather than assuming it.

Run:  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONPATH=<tree> python probe_d3_ladder.py <out.json>
"""
import json
import math
import sys
import time

import numpy as np

import lumenairy
from lumenairy.elements.elements import apply_aperture
from lumenairy.propagators.hf import (
    propagate_huygens_fresnel_with_opl_callable,
)
from lumenairy.propagators.rs import rayleigh_sommerfeld_propagate

NS = (128, 256, 512, 1024)

OPTICS = {
    # VERIFY-C1's optic -- the one round 2 quotes four digits of
    'verify_c1': dict(lam=1064e-9, a=62.5e-6, window=400e-6,
                      z_rs=4.0e-3, z_hf=2.5e-3),
    # this round's own third optic
    'round2_verify': dict(lam=532e-9, a=150e-6, window=900e-6,
                          z_rs=30.0e-3, z_hf=15.0e-3),
}


def closed_form(lam, a, z):
    k = 2.0 * np.pi / lam
    r_a = math.sqrt(z * z + a * a)
    return np.exp(1j * k * z) - (z / r_a) * np.exp(1j * k * r_a)


def build(N, opt, edge_kw):
    dx = opt['window'] / N
    E = np.ones((N, N), dtype=complex)
    return apply_aperture(E, dx, shape='circular',
                          params={'diameter': 2.0 * opt['a']}, **edge_kw), dx


def rs_on_axis(N, opt, edge_kw):
    E_in, dx = build(N, opt, edge_kw)
    E_out = rayleigh_sommerfeld_propagate(
        E_in, z=opt['z_rs'], wavelength=opt['lam'], dx=dx, kernel='spatial')
    return complex(np.asarray(E_out)[N // 2, N // 2])


def hf_on_axis(N, opt, edge_kw):
    E_in, dx = build(N, opt, edge_kw)
    z, lam = opt['z_hf'], opt['lam']

    def opl_fn(s1x, s1y, s2x, s2y):
        return np.sqrt((s1x - s2x) ** 2 + (s1y - s2y) ** 2 + z * z) / lam

    out = propagate_huygens_fresnel_with_opl_callable(
        E_in, opl_fn=opl_fn,
        output_grid_x=np.array([0.0]), output_grid_y=np.array([0.0]),
        input_grid_dx=dx)
    return complex(np.reshape(np.asarray(out), (-1,))[0])


ARMS = {'hard': {'edge': 'hard'}, 'gray': {'edge': 'gray'}, 'default': {}}


def run_optic(name, opt):
    truth_rs = closed_form(opt['lam'], opt['a'], opt['z_rs'])
    truth_hf = closed_form(opt['lam'], opt['a'], opt['z_hf'])
    res = {'params': dict(opt), 'truth_rs': [truth_rs.real, truth_rs.imag],
           'truth_hf': [truth_hf.real, truth_hf.imag], 'errs': {}, 'alias': {}}
    for kernel, fn, truth, z in (('RS', rs_on_axis, truth_rs, opt['z_rs']),
                                 ('HF', hf_on_axis, truth_hf, opt['z_hf'])):
        for arm, kw in ARMS.items():
            errs, vals = [], []
            for N in NS:
                t0 = time.time()
                u = fn(N, opt, kw)
                vals.append([u.real, u.imag])
                errs.append(abs(u - truth) / abs(truth))
                print("   {0} {1} {2} N={3} err={4:.6e} ({5:.1f}s)".format(
                    name, kernel, arm, N, errs[-1], time.time() - t0),
                    flush=True)
            res['errs']['{0}_{1}'.format(kernel, arm)] = {
                'errs': errs, 'vals': vals,
                'orders': [math.log2(errs[i] / errs[i + 1])
                           for i in range(len(errs) - 1)],
                'gain': errs[0] / errs[-1],
                'mean_order': math.log2(errs[0] / errs[-1]) / (len(errs) - 1),
            }
        if kernel == 'RS':
            for N in NS:
                dx = opt['window'] / N
                res['alias']['N{0}'.format(N)] = {
                    'threshold_m': 2.0 * N * dx * dx / opt['lam'],
                    'z_m': z,
                    'clears': z > 2.0 * N * dx * dx / opt['lam']}
    # bit identity of the default arm
    for kernel in ('RS', 'HF'):
        d = res['errs']['{0}_default'.format(kernel)]['vals']
        res['{0}_default_is'.format(kernel)] = [
            arm for arm in ('hard', 'gray')
            if res['errs']['{0}_{1}'.format(kernel, arm)]['vals'] == d]
    return res


def main(out_path):
    out = {'lumenairy_file': lumenairy.__file__,
           'python': sys.version.split()[0],
           'numpy': np.__version__, 'optics': {}}
    for name, opt in OPTICS.items():
        print('==', name, flush=True)
        out['optics'][name] = run_optic(name, opt)
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=repr)
    print('lumenairy:', lumenairy.__file__)
    for name, r in out['optics'].items():
        print('===', name, r['params'])
        for key, v in sorted(r['errs'].items()):
            print("  {0:12s} errs={1}  orders={2}  gain={3:.4f}x  "
                  "mean={4:.4f}".format(
                      key,
                      ' '.join('{0:.4e}'.format(e) for e in v['errs']),
                      ' '.join('{0:.3f}'.format(o) for o in v['orders']),
                      v['gain'], v['mean_order']))
        print("  RS default is bit-identical to", r['RS_default_is'],
              " HF default is", r['HF_default_is'])
        print("  alias:", json.dumps(r['alias']))


if __name__ == '__main__':
    main(sys.argv[1])
