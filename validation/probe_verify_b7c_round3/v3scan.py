"""VERIFY-WP-B7c round 3 -- the plane scan.

ONE library call per plane, with BOTH refusal bars lifted at run time (the
module attributes are patched; ``lumenairy/`` is never edited), so the READING
and the FIELD the shipped bar would have refused come from the SAME call.  The
shipped decision is re-derived afterwards from constants captured BEFORE the
patch.

Every plane is scored against :mod:`v3oracle`'s band-limited ANGULAR SPECTRUM
at FULL radius -- no Debye step, no paraxial step, no azimuthal step, and no
core confinement, which is what E6 of the round-2 verification is about.  The
oracle's own refinement is a command-line knob so that the convergence control
(``--refine``) is a measurement and not an assumption.

Usage:
    python v3scan.py <optic> <z-spec> <out.json> [--oracle asm|none]
                     [--sub N] [--refine N] [--fan N]

``z-spec`` is a comma-separated list of ``a:b:n`` ranges and bare values, in
micrometres.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings

import numpy as np
import v3fixtures as FX
import v3oracle as OR

#: every diagnostic the population, the cost table and the fallback question
#: read.  Recorded whole so a later question can be asked of the JSON without
#: re-measuring.
KEYS = ('pixel_continuity', 'pixel_continuity_of',
        'pixel_continuity_decision', 'pixel_continuity_scope',
        'multibranch_pixel_continuity', 'pixel_halved_power', 'grid_power',
        'power_ratio', 'power_ratio_triangles', 'power_ratio_decision',
        'multibranch_power_ratio', 'multibranch_power_ratio_bracketed',
        'reason', 'fell_back', 'n_branch_max', 'n_triangles_degenerate',
        'n_turn', 'zeta_extrapolation', 'zeta_curvature', 'zeta_linear_range',
        'zeta_linear_resid', 'r_c', 'kappa', 'fit_residual', 'fit_halfwidth',
        'l_airy', 'dark_fill_depth')


def _num(v):
    if v is None or isinstance(v, (str, bool)):
        return v
    a = np.asarray(v)
    if a.ndim == 0 and a.dtype.kind in 'fiu':
        return float(a)
    if a.ndim == 0:
        return None
    return None


def bars():
    from lumenairy.elements import _lens_traced_uniform as U
    return dict(cmax=float(U._MB_PIXEL_CONTINUITY_MAX),
                cmin=float(U._MB_PIXEL_CONTINUITY_MIN),
                pmax=float(U._MB_POWER_RATIO_MAX),
                pmin=float(U._MB_POWER_RATIO_MIN))


def lift_bars():
    from lumenairy.elements import _lens_traced_uniform as U
    U._MB_PIXEL_CONTINUITY_MAX = float('inf')
    U._MB_PIXEL_CONTINUITY_MIN = 0.0
    U._MB_POWER_RATIO_MAX = float('inf')
    U._MB_POWER_RATIO_MIN = 0.0


def restore_bars(b):
    from lumenairy.elements import _lens_traced_uniform as U
    U._MB_PIXEL_CONTINUITY_MAX = b['cmax']
    U._MB_PIXEL_CONTINUITY_MIN = b['cmin']
    U._MB_POWER_RATIO_MAX = b['pmax']
    U._MB_POWER_RATIO_MIN = b['pmin']


def asm_grid(fx, refine):
    """Fine-grid width, in fine pixels.

    The fine window has to hold (a) the whole exit aperture, because that is
    where the source lives, and (b) the output window, because that is where
    the field is read; and the Matsushima band limit already removes every
    plane wave whose ray would traverse more than half the window in ``z``,
    so wrap-around is bounded rather than padded away.  ``1.15 x`` the larger
    of the two plus a third of the output window is what that comes to, and
    the ``--refine`` control measures whether it is enough by moving it.
    """
    dxf = float(fx['dx']) / int(refine)
    span = max(float(fx['prescription']['aperture_diameter']),
               fx['N'] * fx['dx'])
    # ``or_window_mult`` widens it where the Matsushima band limit, not the
    # source, is what sets the window: the limit removes every plane wave
    # whose ray traverses more than half the window in ``z``, so a LONG
    # propagation needs a wider window to carry the same angles.  Measured at
    # the ``Q`` oracle-floor row (``asmbandlimit_win.json``): the relative L2
    # against the exact azimuthal quadrature falls from 0.0144 to 0.0020 the
    # moment the limit clears the largest source-to-pixel angle, and does not
    # move after.
    need = (1.15 * span + 0.35 * fx['N'] * fx['dx']) * float(
        fx.get('or_window_mult', 1.0))
    Nf = max(int(np.ceil(need / dxf)), int(refine) * int(fx['N']))
    if (Nf - int(refine) * int(fx['N'])) % 2:
        Nf += 1
    return Nf


def oracle_field(fx, z, refine=3, n_fan=6001):
    ef = OR.exit_field(fx['prescription'], fx['wavelength'], fx['w0'],
                       n_fan=n_fan)
    E = OR.asm_field(ef, float(z), fx['wavelength'], fx['N'], fx['dx'],
                     refine=int(refine), Nf=asm_grid(fx, refine))
    return E, ef['P_in']


def edge_fraction(E, dx, frac=0.10):
    """Fraction of the returned energy in the outermost tenth of the window's
    half-width -- one of the fallback-ordering candidates."""
    A = np.abs(np.asarray(E)) ** 2
    N = A.shape[0]
    x = (np.arange(N) - N / 2.0) * float(dx)
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X * X + Y * Y)
    tot = float(A.sum())
    if tot <= 0.0:
        return None
    return float(A[r > (1.0 - frac) * (0.5 * N * float(dx))].sum() / tot)


def scan(fx, zs, oracle='asm', ray_subsample=2, n_fan_lib=4000, refine=3,
         n_fan_or=6001):
    from lumenairy.elements import _lens_traced_uniform as U
    b = bars()
    lift_bars()
    rows = []
    try:
        for z in zs:
            E_in = FX.input_field(fx)
            row = dict(fixture=fx['name'], z_um=float(z) * 1e6, N=fx['N'],
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
            for k in KEYS:
                v = d.get(k)
                row[k] = v if isinstance(v, (str, bool)) or v is None \
                    else _num(v)
            row['returned_power'] = OR.power(E, fx['dx'])
            row['edge_fraction'] = edge_fraction(E, fx['dx'])
            c = row.get('pixel_continuity')
            if c is None:
                row['shipped_decision'] = 'not_measured'
            elif c > b['cmax']:
                row['shipped_decision'] = 'REFUSED'
            elif c < b['cmin']:
                row['shipped_decision'] = 'not_converged_loss'
            else:
                row['shipped_decision'] = 'ok'
            pr = row.get('multibranch_power_ratio_bracketed')
            # only the GAIN side of the launched-power bracket refuses; the
            # loss side is reported (``power_ratio_decision='energy_loss'``)
            row['power_arm_refuses'] = bool(pr is not None and pr > b['pmax'])
            row['power_arm_loss'] = bool(pr is not None and pr < b['pmin'])
            row['shipped_returns'] = bool(
                not row['power_arm_refuses']
                and row['shipped_decision'] != 'REFUSED')
            row['route'] = ('fold_ring' if (row.get('reason') == 'fold_ring'
                                            and not row.get('fell_back'))
                            else 'fallback')
            if oracle != 'none':
                E_or, P_in = oracle_field(fx, z, refine=refine,
                                          n_fan=n_fan_or)
                row['fidelity'] = OR.fidelity(E, E_or)
                row['oracle_power'] = OR.power(E_or, fx['dx'])
                row['power_over_oracle'] = (
                    row['returned_power'] / max(row['oracle_power'], 1e-300))
                row['oracle'] = 'asm_full_radius'
                row['oracle_refine'] = int(refine)
                row['P_in'] = float(P_in)
                row['oracle_closure'] = (row['oracle_power']
                                         / max(float(P_in), 1e-300))
                del E_or
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        restore_bars(b)
    return rows


def parse_zs(spec):
    out = []
    for part in str(spec).split(','):
        if not part:
            continue
        if ':' in part:
            a, bb, n = part.split(':')
            out.extend(np.linspace(float(a), float(bb), int(n)).tolist())
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
    ap.add_argument('--refine', type=int, default=3)
    ap.add_argument('--fan', type=int, default=6001)
    a = ap.parse_args()
    import lumenairy
    print('lumenairy.__file__ =', lumenairy.__file__, flush=True)
    fx = FX.FIXTURES[a.fixture]
    rows = scan(fx, parse_zs(a.zs), oracle=a.oracle, ray_subsample=a.sub,
                refine=a.refine, n_fan_or=a.fan)
    with open(a.out, 'w', encoding='cp1252') as f:
        json.dump(dict(lumenairy=lumenairy.__file__,
                       python=sys.version.split()[0], numpy=np.__version__,
                       fixture=fx['name'], note=fx.get('note'),
                       bars=bars(), oracle=a.oracle, refine=a.refine,
                       n_fan_oracle=a.fan, rows=rows), f, indent=1)
    print('wrote', a.out, len(rows), 'rows')


if __name__ == '__main__':
    main()
