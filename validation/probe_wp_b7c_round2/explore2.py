"""Which continuity is the predictive one?

Three candidate readings of the same pixel-halving control, measured side by
side against the oracle-scored outcome:

* ``C_mb``  -- the branch sum's whole-grid deposited power at dx over dx/2
  (what the shipped arbiter reads);
* ``C_uni`` -- the same ratio taken on the COMPLETED field, i.e. the thing the
  caller actually receives;
* ``C_bright`` -- the branch sum's ratio restricted to the pixels the
  completion KEEPS verbatim, i.e. inside the caustic ring and outside the
  fold band the CFU swap rewrites.

Usage::

    python explore2.py <out.json> <FIXTURE>:<z_um> [...]
"""
# ruff: noqa: E402, I001
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixtures as FX                                     # noqa: E402

try:
    from lumenairy.elements._lens_traced_multibranch import (  # noqa: E402
        _multibranch_render)
except ImportError:
    from lumenairy.elements._lens_traced_multibranch import (  # noqa: E402
        apply_real_lens_traced_multibranch)

    def _multibranch_render(E, *, pixel_halving_arbiter=False, **kw):
        return apply_real_lens_traced_multibranch(E, **kw)
from lumenairy.elements._lens_traced_uniform import (      # noqa: E402
    apply_real_lens_traced_uniform)


def _power(E, dx):
    return float(np.sum(np.abs(np.asarray(E)) ** 2)) * dx * dx


def _radii(N, dx):
    x = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    return np.sqrt(X ** 2 + Y ** 2)


def one(name, zum):
    fx = FX.FIXTURES[name]
    z = zum * 1e-6
    N, dx = fx['N'], fx['dx']
    out = {'fixture': name, 'z_um': zum}
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        E1, d1 = _multibranch_render(
            FX.gauss(N, dx, fx['w0']), prescription=fx['prescription'],
            wavelength=fx['wavelength'], dx=dx, output_plane_distance=z,
            return_diagnostics=True, pixel_halving_arbiter=True)
        E2, d2 = _multibranch_render(
            FX.gauss(2 * N, dx / 2, fx['w0']),
            prescription=fx['prescription'], wavelength=fx['wavelength'],
            dx=dx / 2, output_plane_distance=z, ray_subsample=4,
            return_diagnostics=True, pixel_halving_arbiter=False)
    out['C_mb'] = d1.get('pixel_continuity')
    out['n_branch_max'] = d1.get('n_branch_max')
    vals = [float(v) for v in (d1.get('power_ratio'),
                               d1.get('power_ratio_triangles'))
            if v is not None and np.isfinite(v)]
    out['bracket'] = min(vals) if vals else None

    # the completion at both pitches
    def uni(n, d, sub):
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                E, ud = apply_real_lens_traced_uniform(
                    FX.gauss(n, d, fx['w0']),
                    prescription=fx['prescription'],
                    wavelength=fx['wavelength'], dx=d,
                    output_plane_distance=z, ray_subsample=sub,
                    return_diagnostics=True)
            return _power(E, d), ud.get('reason'), ud.get('r_c')
        except RuntimeError as e:
            return None, 'RAISE:' + str(e)[:40], None

    p1, r1, rc1 = uni(N, dx, 2)
    p2, r2, rc2 = uni(2 * N, dx / 2, 4)
    out['uni_reason_coarse'] = r1
    out['uni_reason_fine'] = r2
    out['r_c_um'] = (rc1 * 1e6) if rc1 else None
    out['C_uni'] = (p1 / p2) if (p1 and p2) else None

    # bright-side-only branch-sum continuity: pixels strictly INSIDE the
    # caustic ring and outside the fold band the CFU swap rewrites
    if rc1:
        l_a = None
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                _E, ud = apply_real_lens_traced_uniform(
                    FX.gauss(N, dx, fx['w0']),
                    prescription=fx['prescription'],
                    wavelength=fx['wavelength'], dx=dx,
                    output_plane_distance=z, return_diagnostics=True)
            l_a = ud.get('l_airy')
        except RuntimeError:
            pass
        if l_a:
            cut = max(rc1 - 3.0 * l_a, 0.0)
            m1 = _radii(N, dx) < cut
            m2 = _radii(2 * N, dx / 2) < cut
            q1 = float(np.sum(np.abs(np.asarray(E1)[m1]) ** 2)) * dx * dx
            q2 = float(np.sum(np.abs(np.asarray(E2)[m2]) ** 2)) * (dx / 2) ** 2
            out['C_bright'] = (q1 / q2) if q2 > 0 else None
            out['bright_cut_um'] = cut * 1e6
    return out


def main(argv):
    rows = []
    print(f"{'fx':7s} {'z_um':>9s} {'bracket':>9s} {'C_mb':>8s} "
          f"{'C_uni':>8s} {'C_bright':>9s} {'nbr':>5s}  reasons")
    for spec in argv[2:]:
        name, zum = spec.split(':')
        r = one(name, float(zum))
        rows.append(r)
        print(f"{r['fixture']:7s} {r['z_um']:9.2f} {r['bracket']!s:>9.9} "
              f"{r['C_mb']!s:>8.8} {r['C_uni']!s:>8.8} "
              f"{r.get('C_bright')!s:>9.9} {r['n_branch_max']!s:>5} "
              f"{r['uni_reason_coarse']!s:.22} | {r['uni_reason_fine']!s:.22}",
              flush=True)
        with open(argv[1], 'w') as fh:
            json.dump({'rows': rows}, fh, indent=1, default=str)


if __name__ == '__main__':
    main(sys.argv)
