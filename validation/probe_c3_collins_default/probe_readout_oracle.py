"""WP-C3 -- is the one-step Collins focus readout RIGHT on a chain's own exit
pitch, or is the Sziklas readout?  Decided against an oracle that is neither.

Run as a CHILD process bound to ONE tree::

    python probe_readout_oracle.py <tree> <out.json>

WHY THIS PROBE EXISTS.  ``probe_oracle_ladders.py`` measured the chain's
image-plane readout through both transports on WP-B4's own two-group relay
fixture and read a peak of 1.0743 on ``'sziklas'`` against 9017.1 on
``'collins'`` -- a factor of 8394, with the Kelly guard's K1 condition
warning on the Collins arm and nothing warning on the other.  A ratio of that
size is not an accuracy difference; one of the two answers is wrong, and
neither transport can be the judge of the other.

THE ORACLE.  The chain is stopped at its EXIT PLANE (``final_distance=0``,
no readout), which hands back an envelope, its carrier and its pitch.  The
physical field there is rebuilt here, and then propagated the final leg by a
DENSE separable matrix Fresnel transform written in this file: for each output
sample ``x``,

    u(x) = exp(i k z)/(i lambda z) * sum_u u_in(u) exp(i k (x - u)^2 / 2z) du

evaluated as two ``(N_out x N_in)`` matrix products, one per axis.  No FFT, no
chirp-Z, no Bluestein, no zoom, no standoff plane -- and therefore none of the
sampling conditions any of those carry.  It is O(N_out N_in) per axis and, on
this fixture, exact to the summation's own rounding: the ONLY approximation in
it is the paraxial (Fresnel) kernel, which is the same kernel BOTH transports'
readouts use, so it cannot favour either.

The convergence of the oracle itself is demonstrated rather than asserted: the
same sum is evaluated at the fixture's pitch and at a refined input pitch
obtained by zero-padding-free re-launch, and the two are compared.  If the
oracle had not converged, the comparison below would be between two unknowns.
"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, _TREE)

import clib  # noqa: E402

import numpy as np  # noqa: E402

clib.anchor(_TREE)

import lumenairy.propagators.carrier as CA          # noqa: E402

WL = 1.31e-6
_TKW = dict(on_undersample='silent', on_noncollimated='silent')


def _relay_fixture(n=256, dx=60e-6, w=4.5e-3):
    x = clib.axis(n, dx)
    X, Y = np.meshgrid(x, x)
    env = np.exp(-(X ** 2 + Y ** 2) / (w * w)).astype(np.complex128)
    presc = {'name': 'p', 'aperture_diameter': 14e-3, 'thicknesses': [3e-3],
             'surfaces': [
                 {'radius': 60e-3, 'glass_before': 'air',
                  'glass_after': 'N-BK7', 'conic': 0.0, 'radius_y': None,
                  'conic_y': None, 'aspheric_coeffs': None,
                  'aspheric_coeffs_y': None},
                 {'radius': -60e-3, 'glass_before': 'N-BK7',
                  'glass_after': 'air', 'conic': 0.0, 'radius_y': None,
                  'conic_y': None, 'aspheric_coeffs': None,
                  'aspheric_coeffs_y': None}]}
    return env, dx, 60e-3, [{'prescription': presc, 'gap_before': 20e-3},
                            {'prescription': presc, 'gap_before': 10e-3}]


def _dense_fresnel(u_in, dx_in, dy_in, z, wavelength, dx_out, n_out,
                   centre_out=(0.0, 0.0)):
    """The Fresnel integral as two dense matrix products.  Independent of
    every transform in the library."""
    k = 2.0 * np.pi / wavelength
    ny, nx = u_in.shape[-2], u_in.shape[-1]
    xi = clib.axis(nx, dx_in)
    yi = clib.axis(ny, dy_in)
    xo = clib.axis(n_out, dx_out) + float(centre_out[0])
    yo = clib.axis(n_out, dx_out) + float(centre_out[1])
    Wx = np.exp(1j * k * (xo[:, None] - xi[None, :]) ** 2 / (2.0 * z))
    Wy = np.exp(1j * k * (yo[:, None] - yi[None, :]) ** 2 / (2.0 * z))
    out = Wy @ (u_in @ Wx.T)
    return out * (np.exp(1j * k * z) / (1j * wavelength * z)
                  * dx_in * dy_in)


def _metrics(field, dx_out):
    a = np.abs(np.asarray(field)) ** 2
    tot = float(a.sum())
    return {'peak': float(a.max()), 'power': tot * dx_out * dx_out,
            'l2': float(np.linalg.norm(np.asarray(field)))}


def main():
    out_path = sys.argv[2]
    rec = {'build': clib.build_tag(), 'tree': _TREE,
           'carrier_file': CA.__file__, 'wavelength': WL}

    z_final = 8e-3
    fr = dict(dx_out=0.5e-6, N_out=64)

    for n_in, dx_in in ((256, 60e-6), (512, 30e-6)):
        env, dx, r_in, groups = _relay_fixture(n=n_in, dx=dx_in)
        base = dict(r_in=r_in, ray_subsample=16, n_workers=1,
                    traced_kwargs=_TKW, final_leg='paraxial')
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            exit_res = CA.propagate_traced_carrier_chain(
                env, groups, WL, dx, final_distance=0.0, **base)
        dxe = float(exit_res.dx if not isinstance(exit_res.dx, tuple)
                    else exit_res.dx[0])
        Rexit = (exit_res.R if not isinstance(exit_res.R, tuple)
                 else exit_res.R[0])
        u_exit = clib.reconstruct_field(exit_res.field, Rexit, dxe, dxe, WL)

        ref = _dense_fresnel(u_exit, dxe, dxe, z_final, WL,
                             fr['dx_out'], fr['N_out'])

        arms = {}
        for tr in ('sziklas', 'collins'):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                try:
                    res = CA.propagate_traced_carrier_chain(
                        env, groups, WL, dx, final_distance=z_final,
                        focus_readout=fr, transport=tr, **base)
                    got = np.asarray(res.field)
                    arms[tr] = dict(
                        _metrics(got, fr['dx_out']),
                        rel_l2_vs_oracle=clib.rel_l2(got, ref),
                        rel_l2_piston_free=clib.rel_l2_piston_free(got, ref),
                        raised=None,
                        kelly=[str(x.message)[:200] for x in w
                               if 'under-sampled' in str(x.message)])
                except Exception as exc:               # noqa: BLE001
                    arms[tr] = {'raised': f'{type(exc).__name__}: {exc}'}

        rec[f'N{n_in}'] = {
            'n_in': n_in, 'dx_in': dx_in, 'dx_exit': dxe,
            'R_exit': float(Rexit), 'z_final': z_final,
            'oracle': _metrics(ref, fr['dx_out']),
            'arms': arms,
        }

    # The oracle's own convergence: the SAME physical readout evaluated from
    # two different chain grids.  If these two agree far better than either
    # arm's disagreement with them, the oracle is the fixed point.
    a = rec['N256']['oracle']
    b = rec['N512']['oracle']
    rec['oracle_convergence'] = {
        'peak_ratio_512_over_256': b['peak'] / max(a['peak'], 1e-300),
        'power_ratio_512_over_256': b['power'] / max(a['power'], 1e-300),
    }
    clib.write_json(rec, out_path)
    for key in ('N256', 'N512'):
        r = rec[key]
        print(f"[{key}] oracle peak {r['oracle']['peak']:.6g}  "
              + '  '.join(
                  f"{t}: " + (v['raised'] if v.get('raised')
                              else f"peak {v['peak']:.6g} "
                                   f"rel {v['rel_l2_vs_oracle']:.3e} "
                                   f"kelly {len(v['kelly'])}")
                  for t, v in r['arms'].items()))
    print(f"[oracle convergence] {rec['oracle_convergence']}")


if __name__ == '__main__':
    main()
