"""VERIFY-C1 probe 1 -- an INDEPENDENT convergence ladder for apply_aperture's
``edge`` default, on a different optic from WP-B11 / WP-C1's.

WP-B11 sec. 2.9 and the WP-C1 report both measure lambda = 633 nm, a = 100 um,
window 512 um, z = 16 mm (RS) / 5 mm (HF).  Re-running that geometry would
re-read their number, not re-decide their claim.  This probe uses

    lambda  = 1064 nm      (different source)
    a       =  62.5 um     (different radius)
    window  = 400 um       (different window, so a different D/dx at every N)
    z_RS    = 4.0 mm       (Fresnel number a^2/(lambda z) = 0.918, a different
                            diffraction regime from WP-B11's)
    z_HF    = 2.5 mm

and the same CLOSED-FORM on-axis oracle, which depends on no discretisation:

    U(0,0,z) = exp(i k z) - (z/r_a) exp(i k r_a),   r_a = sqrt(z^2 + a^2)

RS alias threshold check: the spatial kernel wants z > 2 N dx^2 / lambda =
2 W^2 / (N lambda), which is LARGEST at the coarsest grid: 2.35 mm at N = 128.
z_RS = 4.0 mm clears it at every N, and the probe records that.

Three arms per kernel: ``edge='hard'``, ``edge='gray'``, and the DEFAULT arm
called with no keyword at all, with a BIT-IDENTITY test of the default arm
against each named arm (not a tolerance).

It also runs, on the same optic:
  * the ``edge_samples`` ladder 1/2/4/8/16 at the mid grid, from BOTH sides;
  * an exact-AREA oracle on apertures the ladder does not cover -- a CENTRED
    circle, a DECENTRED circle (centre at a non-half-integer pixel offset) and
    an ANAMORPHIC rectangle on an anamorphic grid (dy != dx) -- where the truth
    is the analytic area of the shape, a pure geometry fact, together with the
    two-sided POWER band [area - n_rim dx dy / 4, area].

Usage:  python -P validation/probe_verify_c1/probe_ladder_v.py <tag>
writes  validation/probe_verify_c1/ladder_v_<tag>.json
"""
import json
import os
import platform
import sys

import numpy as np

import lumenairy
from lumenairy.elements.elements import apply_aperture
from lumenairy.propagators.hf import (
    propagate_huygens_fresnel_with_opl_callable,
)
from lumenairy.propagators.rs import rayleigh_sommerfeld_propagate

LAM = 1064e-9
A = 62.5e-6          # aperture RADIUS [m]
WINDOW = 400e-6      # full window [m]
NS = (128, 256, 512, 1024)
Z_RS = 4.0e-3
Z_HF = 2.5e-3


def on_axis_closed_form(z):
    k = 2.0 * np.pi / LAM
    r_a = np.sqrt(z * z + A * A)
    return np.exp(1j * k * z) - (z / r_a) * np.exp(1j * k * r_a)


def build_input(N, edge_kw):
    dx = WINDOW / N
    E = np.ones((N, N), dtype=complex)
    return apply_aperture(E, dx, shape='circular',
                          params={'diameter': 2.0 * A}, **edge_kw), dx


def rs_on_axis(N, edge_kw):
    E_in, dx = build_input(N, edge_kw)
    E_out = rayleigh_sommerfeld_propagate(
        E_in, z=Z_RS, wavelength=LAM, dx=dx, kernel='spatial')
    return complex(E_out[N // 2, N // 2])


def hf_on_axis(N, edge_kw):
    E_in, dx = build_input(N, edge_kw)
    z = Z_HF

    def opl_fn(s1x, s1y, s2x, s2y):
        return np.sqrt((s1x - s2x) ** 2 + (s1y - s2y) ** 2 + z * z) / LAM

    out = propagate_huygens_fresnel_with_opl_callable(
        E_in, opl_fn=opl_fn,
        output_grid_x=np.array([0.0]), output_grid_y=np.array([0.0]),
        input_grid_dx=dx)
    return complex(np.reshape(out, (-1,))[0])


def orders(errs):
    return [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]


def ladder(fn, z):
    exact = on_axis_closed_form(z)
    arms = {'default': {}, 'hard': {'edge': 'hard'}, 'gray': {'edge': 'gray'}}
    rows = {}
    raw = {}
    for name, kw in arms.items():
        errs = []
        vals = []
        for N in NS:
            v = fn(N, kw)
            errs.append(float(abs(v - exact) / abs(exact)))
            vals.append((v.real, v.imag))
        rows[name] = {'err': errs, 'order': orders(errs),
                      'value': [[a, b] for a, b in vals]}
        raw[name] = vals
    rows['_default_bit_identical_to'] = sorted(
        n for n in ('hard', 'gray') if raw['default'] == raw[n])
    rows['_exact'] = [exact.real, exact.imag]
    return rows


def samples_ladder(N):
    """edge_samples 1/2/4/8/16 at one grid, RS kernel, same optic."""
    exact = on_axis_closed_form(Z_RS)
    out = {}
    for n_sub in (1, 2, 4, 8, 16):
        v = rs_on_axis(N, {'edge': 'gray', 'edge_samples': n_sub})
        out[str(n_sub)] = float(abs(v - exact) / abs(exact))
    v_hard = rs_on_axis(N, {'edge': 'hard'})
    out['_hard'] = float(abs(v_hard - exact) / abs(exact))
    out['_n_sub_1_is_bitwise_hard'] = bool(
        rs_on_axis(N, {'edge': 'gray', 'edge_samples': 1}) == v_hard)
    out['_gain_1_to_2'] = out['1'] / out['2']
    out['_gain_2_to_4'] = out['2'] / out['4']
    out['_gain_4_to_8'] = out['4'] / out['8']
    out['_gain_8_to_16'] = out['8'] / out['16']
    return out


# ---------------------------------------------------------------- area oracle
def _area_readings(N, dx, dy, shape, params, xc, yc, analytic_area):
    """Transmitted AREA (linear sum of the amplitude mask) and POWER
    (quadratic sum) against the analytic area, on both arms and the default."""
    E = np.ones((N, N), dtype=complex)
    out = {'analytic_area_m2': analytic_area}
    for name, kw in (('hard', {'edge': 'hard'}), ('gray', {'edge': 'gray'}),
                     ('default', {})):
        m = apply_aperture(E, dx, shape=shape, params=params, xc=xc, yc=yc,
                           dy=dy, **kw).real
        area = float(m.sum() * dx * dy)
        power = float((np.abs(m) ** 2).sum() * dx * dy)
        n_rim = int(np.count_nonzero((m > 0) & (m < 1)))
        bound = n_rim * dx * dy / 4.0
        out[name] = {
            'area': area,
            'area_rel_err': area / analytic_area - 1.0,
            'power': power,
            'power_rel_err': power / analytic_area - 1.0,
            'n_rim': n_rim,
            'rim_bound_m2': bound,
            'power_in_band': bool(area - bound <= power <= area),
            'band_lo_slack': power - (area - bound),
            'band_hi_slack': area - power,
            'sum_f_minus_f2_m2': float((m - m ** 2).sum() * dx * dy),
        }
    return out


def area_cases():
    cases = {}
    N = 512
    W = WINDOW
    dx = W / N
    cases['centred_circle'] = _area_readings(
        N, dx, dx, 'circular', {'diameter': 2 * A}, 0.0, 0.0, np.pi * A * A)
    # DECENTRED circle -- centre at (+3.37 dx, -1.83 dx): neither rim lands on
    # a lattice symmetry, so no cancellation can flatter either arm.
    cases['decentred_circle'] = _area_readings(
        N, dx, dx, 'circular', {'diameter': 2 * A},
        3.37 * dx, -1.83 * dx, np.pi * A * A)
    # ANAMORPHIC rectangle on an ANAMORPHIC grid (dy = 1.6 dx), decentred.
    Wx, Wy = 173.0e-6, 91.0e-6
    dyy = 1.6 * dx
    cases['anamorphic_rect_decentred'] = _area_readings(
        N, dx, dyy, 'rectangular', {'width_x': Wx, 'width_y': Wy},
        2.5 * dx, 1.25 * dyy, Wx * Wy)
    cases['anamorphic_rect_centred'] = _area_readings(
        N, dx, dyy, 'rectangular', {'width_x': Wx, 'width_y': Wy},
        0.0, 0.0, Wx * Wy)
    # Annular, decentred, on the square grid.
    cases['decentred_annulus'] = _area_readings(
        N, dx, dx, 'annular',
        {'inner_diameter': 40e-6, 'outer_diameter': 130e-6},
        1.7 * dx, 0.9 * dx, np.pi * ((65e-6) ** 2 - (20e-6) ** 2))
    lad = {}
    for n in (128, 256, 512, 1024):
        d = W / n
        r = _area_readings(n, d, d, 'circular', {'diameter': 2 * A},
                           3.37 * d, -1.83 * d, np.pi * A * A)
        lad[str(n)] = {'hard': r['hard']['area_rel_err'],
                       'gray': r['gray']['area_rel_err'],
                       'power_gray_rel': r['gray']['power_rel_err'],
                       'n_rim': r['gray']['n_rim'],
                       'rim_bound_over_area':
                           r['gray']['rim_bound_m2'] / (np.pi * A * A)}
    cases['_decentred_area_ladder'] = lad
    return cases


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'run'
    import inspect
    sig = inspect.signature(apply_aperture)
    thr = {str(n): 2.0 * (WINDOW ** 2) / (n * LAM) for n in NS}
    out = {
        'tag': tag,
        'lumenairy_file': lumenairy.__file__,
        'lumenairy_version': getattr(lumenairy, '__version__', '?'),
        'python': sys.version,
        'numpy': np.__version__,
        'platform': platform.platform(),
        'threads': {k: os.environ.get(k) for k in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'MKL_NUM_THREADS')},
        'geometry': {'lambda_m': LAM, 'a_m': A, 'window_m': WINDOW,
                     'z_rs_m': Z_RS, 'z_hf_m': Z_HF, 'N': list(NS),
                     'fresnel_number_rs': (A * A) / (LAM * Z_RS)},
        'rs_alias_threshold_m': thr,
        'rs_z_clears_threshold': all(Z_RS > v for v in thr.values()),
        'signature_default_edge': sig.parameters['edge'].default,
        'signature_default_edge_samples':
            sig.parameters['edge_samples'].default,
        'rs': ladder(rs_on_axis, Z_RS),
        'hf': ladder(hf_on_axis, Z_HF),
        'edge_samples_ladder_N512': samples_ladder(512),
        'area': area_cases(),
    }
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, 'ladder_v_%s.json' % tag)
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print('lumenairy.__file__ =', lumenairy.__file__)
    print('signature edge default =', out['signature_default_edge'])
    print('RS default bit-identical to', out['rs']['_default_bit_identical_to'])
    print('HF default bit-identical to', out['hf']['_default_bit_identical_to'])
    print('RS hard err', out['rs']['hard']['err'])
    print('RS gray err', out['rs']['gray']['err'])
    print('RS hard ord', out['rs']['hard']['order'])
    print('RS gray ord', out['rs']['gray']['order'])
    print('HF hard err', out['hf']['hard']['err'])
    print('HF gray err', out['hf']['gray']['err'])
    print('HF hard ord', out['hf']['hard']['order'])
    print('HF gray ord', out['hf']['gray']['order'])
    print('wrote', path)


if __name__ == '__main__':
    main()
