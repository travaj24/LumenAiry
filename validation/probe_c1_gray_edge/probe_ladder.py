"""WP-C1 probe 1 -- the oracle ladder that decides `apply_aperture(edge=)`'s default.

Reproduces WP-B11 section 2.9's measurement on both spatial kernels, and adds a
third arm the report did not need: the DEFAULT arm, called with no ``edge=``
keyword at all.  Run at the parent commit the default arm coincides with the
``'hard'`` arm; run after the flip it coincides with the ``'gray'`` arm.  That
is the whole content of the default move, measured rather than asserted.

Geometry (WP-B11 sec. 2.9, WP-B3 sec. 3.2): lambda = 633 nm, circular aperture
of radius a = 100 um centred on a 512 um window, N = 128 / 256 / 512 / 1024.
The oracle is the exact on-axis RS-I field behind a circular aperture in a
plane screen illuminated by a unit-amplitude plane wave,

    U(0, 0, z) = exp(i k z) - (z / r_a) exp(i k r_a),   r_a = sqrt(z^2 + a^2)

(Born & Wolf sec. 8.3.2 / Goodman eq. 3-43 evaluated on axis), which is a
CLOSED FORM, not another discretisation: nothing in it depends on N, on the
kernel, or on the build.  The reported error is |U_num - U_exact| / |U_exact|
at the on-axis pixel.

Two kernels:
  * ``rayleigh_sommerfeld_propagate(kernel='spatial')`` at z = 16 mm (above the
    spatial kernel's own alias threshold 2 N dx^2 / lambda at every N here),
  * ``propagate_huygens_fresnel_with_opl_callable`` at z = 5 mm with the exact
    spherical OPL, evaluated at ONE on-axis output point.

Usage:  python validation/probe_c1_gray_edge/probe_ladder.py <tag>
writes  validation/probe_c1_gray_edge/ladder_<tag>.json
"""
import json
import os
import sys
import platform

import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

import lumenairy  # noqa: E402
from lumenairy.elements.elements import apply_aperture  # noqa: E402
from lumenairy.propagators.rs import rayleigh_sommerfeld_propagate  # noqa: E402
from lumenairy.propagators.hf import (  # noqa: E402
    propagate_huygens_fresnel_with_opl_callable,
)

WAVELENGTH = 633e-9
A = 100e-6          # aperture RADIUS [m]
WINDOW = 512e-6     # full window [m]
NS = (128, 256, 512, 1024)
Z_RS = 16e-3
Z_HF = 5e-3


def on_axis_closed_form(z):
    """Exact on-axis RS-I field behind the circular aperture."""
    k = 2.0 * np.pi / WAVELENGTH
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
        E_in, z=Z_RS, wavelength=WAVELENGTH, dx=dx, kernel='spatial')
    return complex(E_out[N // 2, N // 2])


def hf_on_axis(N, edge_kw):
    E_in, dx = build_input(N, edge_kw)
    lam = WAVELENGTH
    z = Z_HF

    def opl_fn(s1x, s1y, s2x, s2y):
        # WAVES, per this entry point's units contract.
        return np.sqrt((s1x - s2x) ** 2 + (s1y - s2y) ** 2 + z * z) / lam

    out = propagate_huygens_fresnel_with_opl_callable(
        E_in, opl_fn=opl_fn,
        output_grid_x=np.array([0.0]), output_grid_y=np.array([0.0]),
        input_grid_dx=dx)
    return complex(np.reshape(out, (-1,))[0])


def orders(errs):
    """log2 ratio between successive rows (the grid halves each step)."""
    return [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]


def ladder(fn, z, arm_kwargs):
    exact = on_axis_closed_form(z)
    rows = {}
    for name, kw in arm_kwargs.items():
        errs, vals = [], []
        for N in NS:
            v = fn(N, kw)
            errs.append(float(abs(v - exact) / abs(exact)))
            vals.append([v.real, v.imag])
        rows[name] = {'err': errs, 'order': orders(errs), 'value': vals}
    rows['_exact'] = [exact.real, exact.imag]
    return rows


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'run'
    arms = {
        'default': {},                                   # no edge= keyword
        'hard': {'edge': 'hard'},
        'gray': {'edge': 'gray'},
    }
    import inspect
    sig = inspect.signature(apply_aperture)
    out = {
        'tag': tag,
        'lumenairy_file': lumenairy.__file__,
        'lumenairy_version': getattr(lumenairy, '__version__', '?'),
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'numpy': np.__version__,
        'signature_default_edge': sig.parameters['edge'].default,
        'signature_default_edge_samples': sig.parameters['edge_samples'].default,
        'geometry': {'wavelength': WAVELENGTH, 'a': A, 'window': WINDOW,
                     'N': list(NS), 'z_rs': Z_RS, 'z_hf': Z_HF},
        'rs_spatial': ladder(rs_on_axis, Z_RS, arms),
        'hf_quadrature': ladder(hf_on_axis, Z_HF, arms),
    }
    # Is the default arm the same COMPLEX NUMBER as one of the named arms?
    for kern in ('rs_spatial', 'hf_quadrature'):
        d = out[kern]['default']['value']
        out[kern]['default_matches'] = [
            nm for nm in ('hard', 'gray')
            if out[kern][nm]['value'] == d]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f'ladder_{tag}.json')
    with open(path, 'w', encoding='cp1252') as f:
        json.dump(out, f, indent=2)
    print(f"lumenairy.__file__ = {lumenairy.__file__}")
    print(f"signature edge default = {out['signature_default_edge']!r}, "
          f"edge_samples = {out['signature_default_edge_samples']!r}")
    for kern in ('rs_spatial', 'hf_quadrature'):
        print(f"\n--- {kern} ---")
        print(f"{'N':>6} {'hard':>13} {'gray':>13} {'default':>13}")
        for i, N in enumerate(NS):
            print(f"{N:>6} {out[kern]['hard']['err'][i]:>13.4e} "
                  f"{out[kern]['gray']['err'][i]:>13.4e} "
                  f"{out[kern]['default']['err'][i]:>13.4e}")
        print(f"{'order':>6} "
              f"{'/'.join(f'{o:.3f}' for o in out[kern]['hard']['order'])}   "
              f"{'/'.join(f'{o:.3f}' for o in out[kern]['gray']['order'])}   "
              f"{'/'.join(f'{o:.3f}' for o in out[kern]['default']['order'])}")
        print(f"default is bit-identical to: {out[kern]['default_matches']}")
    print(f"\nwrote {path}")


if __name__ == '__main__':
    main()
