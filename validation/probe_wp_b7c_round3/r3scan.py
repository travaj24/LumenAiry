"""WP-B7c round 3 -- the plane scan.

Same design as the round-2 verifier's ``vscan.py``, which this round keeps
deliberately: ONE call per plane, on ONE tree, with both refusal bars patched
to ``inf`` at run time (``lumenairy/`` is never edited), so the READING and
the FIELD it would have refused come from the SAME call.  The shipped
decision is re-derived afterwards from constants imported BEFORE the patch.

Three things are new here:

* every plane is scored against a FULL-RADIUS oracle (the angular spectrum,
  which makes no Debye, paraxial or azimuthal approximation at any radius).
  E6 of the round-2 verification is that the builder's oracle-floor column
  substituted its exact arm only inside the 99.95 %-energy CORE, outside
  which the compared fields are identical by construction; ``r3oracle`` also
  carries ``exact_full_radius`` so that the ASM arm can be bracketed by an
  exact QUADRATURE at every radius on a sample of planes
  (``r3oraclefloor.py``);
* the FALLBACK-route candidate readings of R3-4 are recorded on every plane
  (the launched-power bracket, the two power ratios, the branch census, the
  fold-scale diagnostics and the fraction of the returned energy in the
  outermost ring of the window), so the question "does anything already in
  the diagnostics order the fallback route's fidelity?" is answered from the
  population rather than from an opinion;
* ``--oracle none`` re-takes the READINGS only.  The R3-3 pixel-centre
  alignment changes the half-pitch render and therefore the reading; it does
  NOT change the returned field (the coarse render and the completion are
  untouched), so the oracle column measured once on the pre-fix tree is
  still the fidelity of the post-fix field, and only the reading has to be
  re-measured.  ``r3join.py`` checks that invariant per plane before it
  joins (the returned power must be bit-equal).

Usage:
    python r3scan.py <optic> <z-spec> <out.json> [--oracle asm|none]
                     [--sub N] [--refine N]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings

import numpy as np
import r3fixtures as FX
import r3oracle as OR


def _bars():
    from lumenairy.elements import _lens_traced_uniform as U
    return (float(U._MB_PIXEL_CONTINUITY_MAX),
            float(U._MB_PIXEL_CONTINUITY_MIN),
            float(U._MB_POWER_RATIO_MAX), float(U._MB_POWER_RATIO_MIN))


def _disable_bars():
    from lumenairy.elements import _lens_traced_uniform as U
    U._MB_PIXEL_CONTINUITY_MAX = float('inf')
    U._MB_PIXEL_CONTINUITY_MIN = 0.0
    U._MB_POWER_RATIO_MAX = float('inf')
    U._MB_POWER_RATIO_MIN = 0.0


def _asm_grid(fx, refine):
    need = 2.2 * float(fx['prescription']['aperture_diameter'])
    dxf = fx['dx'] / refine
    Nf = max(int(np.ceil(need / dxf)), refine * fx['N'])
    if (Nf - refine * fx['N']) % 2:
        Nf += 1
    return Nf


def oracle_2d(fx, z, kind='asm', n_fan=6000, refine=4):
    presc, wl = fx['prescription'], fx['wavelength']
    N, dx, w0 = fx['N'], fx['dx'], fx['w0']
    h, y, opl, amp, yl, P_in = OR.exit_field(presc, wl, w0, z, n_fan=n_fan)
    E = OR.asm_field(y, opl, amp, z, wl, N, dx, refine=refine,
                     Nf=_asm_grid(fx, refine))
    return E, P_in


#: every diagnostic key the population, the bar-cost table and the R3-4
#: fallback question read.  Recorded whole so a later question can be asked
#: of the JSON without re-measuring.
_KEYS = ('pixel_continuity', 'pixel_continuity_of',
         'pixel_continuity_decision', 'multibranch_pixel_continuity',
         'pixel_halved_power', 'power_ratio', 'power_ratio_triangles',
         'multibranch_power_ratio_bracketed', 'power_ratio_decision',
         'reason', 'fell_back', 'n_branch_max', 'n_degenerate',
         'zeta_extrapolation', 'grid_power', 'launched_power',
         'launched_power_triangles', 'r_c', 'kappa', 'fit_residual',
         'fit_halfwidth', 'l_airy', 'dark_fill_depth', 'zeta_curvature',
         'zeta_linear_range', 'zeta_linear_resid', 'n_turn')


def _edge_fraction(E, dx, frac=0.10):
    """Fraction of the returned energy in the outermost ``frac`` of the
    window's half-width -- the "is the field leaving the render window?"
    candidate of R3-4."""
    A = np.abs(np.asarray(E)) ** 2
    N = A.shape[0]
    x = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X * X + Y * Y)
    rmax = 0.5 * N * dx
    tot = float(A.sum())
    if tot <= 0.0:
        return None
    return float(A[r > (1.0 - frac) * rmax].sum() / tot)


def scan(fx, zs, oracle='asm', ray_subsample=2, n_fan_lib=4000, refine=4):
    from lumenairy.elements import _lens_traced_uniform as U
    cmax, cmin, pmax, pmin = _bars()
    _disable_bars()
    rows = []
    try:
        for z in zs:
            E_in = FX.input_field(fx)
            row = dict(fixture=fx['name'], z_um=z * 1e6, N=fx['N'],
                       dx_um=fx['dx'] * 1e6, ray_subsample=ray_subsample)
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    E, d = U.apply_real_lens_traced_uniform(
                        E_in, prescription=fx['prescription'],
                        wavelength=fx['wavelength'], dx=fx['dx'],
                        output_plane_distance=float(z),
                        ray_subsample=ray_subsample, n_fan=n_fan_lib,
                        return_diagnostics=True)
                row['warnings'] = sorted({str(x.message)[:70] for x in w})
            except Exception as exc:                        # noqa: BLE001
                row['error'] = f'{type(exc).__name__}: {str(exc)[:200]}'
                rows.append(row)
                print(json.dumps(row), flush=True)
                continue
            for k in _KEYS:
                v = d.get(k)
                row[k] = (float(v) if isinstance(v, (int, float, np.floating))
                          and not isinstance(v, bool) else v)
            row['returned_power'] = OR.power(E, fx['dx'])
            row['edge_fraction'] = _edge_fraction(E, fx['dx'])
            c = row.get('pixel_continuity')
            if c is None:
                row['shipped_decision'] = 'not_measured'
            elif c > cmax:
                row['shipped_decision'] = 'REFUSED'
            elif c < cmin:
                row['shipped_decision'] = 'not_converged_loss'
            else:
                row['shipped_decision'] = 'ok'
            b = row.get('multibranch_power_ratio_bracketed')
            row['power_arm_refuses'] = bool(b is not None and b > pmax)
            if oracle == 'none':
                rows.append(row)
                print(json.dumps(row), flush=True)
                continue
            E_or, P_in = oracle_2d(fx, z, refine=refine)
            row['fidelity'] = OR.fidelity(E, E_or)
            row['oracle_power'] = OR.power(E_or, fx['dx'])
            row['power_over_oracle'] = (row['returned_power']
                                        / max(row['oracle_power'], 1e-300))
            row['oracle'] = 'asm_full_radius'
            row['oracle_refine'] = refine
            row['P_in'] = P_in
            row['oracle_closure'] = row['oracle_power'] / max(P_in, 1e-300)
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        U._MB_PIXEL_CONTINUITY_MAX, U._MB_PIXEL_CONTINUITY_MIN = cmax, cmin
        U._MB_POWER_RATIO_MAX, U._MB_POWER_RATIO_MIN = pmax, pmin
    return rows


def parse_zs(spec):
    out = []
    for part in spec.split(','):
        if ':' in part:
            a, b, n = part.split(':')
            out.extend(np.linspace(float(a), float(b), int(n)).tolist())
        else:
            out.append(float(part))
    return [z * 1e-6 for z in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fixture')
    ap.add_argument('zs')
    ap.add_argument('out')
    ap.add_argument('--oracle', default='asm')
    ap.add_argument('--sub', type=int, default=2)
    ap.add_argument('--refine', type=int, default=4)
    a = ap.parse_args()
    import lumenairy
    print('lumenairy.__file__ =', lumenairy.__file__, flush=True)
    fx = FX.FIXTURES[a.fixture]
    rows = scan(fx, parse_zs(a.zs), oracle=a.oracle, ray_subsample=a.sub,
                refine=a.refine)
    with open(a.out, 'w', encoding='cp1252') as f:
        json.dump(dict(lumenairy=lumenairy.__file__,
                       python=sys.version.split()[0],
                       numpy=np.__version__, fixture=fx['name'],
                       note=fx.get('note'), oracle=a.oracle,
                       refine=a.refine, rows=rows), f, indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
