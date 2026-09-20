"""WP-C5 round 2 (VERIFY-WP-C5 D6) -- the accuracy rule's band, re-measured.

The Migration note and the CHANGELOG described the rule as a NEAR-FOCUS rule.
The criterion is ``sqrt(3/2) k |z_eff| theta_env^4 / 8 > tau``, in which the
envelope's angle enters at the FOURTH power, so a wide envelope substitutes
for a large ``z_eff``.  This probe re-measures, on the running build:

  * ``mismatch_ladder`` -- the verification's own family (one physical beam
    re-enveloped against three mismatched carriers, readout AT the beam's
    focus): the leg FURTHEST from its own ``A = 0`` plane is the one that
    falls back;
  * ``quarter_distance`` -- the same beam read out a QUARTER of the way to
    its focus, where the rule still fires at ``fr = 0.70`` and does not at
    ``fr = 0.80``, which is further from ``A = 0``;
  * ``same_leg_width`` -- ONE leg (one carrier, one readout plane, one
    ``z_eff``) with only the input beam's radius moved, which is what
    separates the angle from the distance with nothing else varying;
  * ``b4_leg`` -- the one leg in the library's own suite that the rule moves.

Usage (BLAS pinned on the COMMAND LINE)::

    PYTHONPATH=<tree> python validation/probe_c5_round2/r2_d6_band.py OUT.json
"""
import os
import sys
import warnings

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), 'probe_verify_c5'))

import numpy as np                                            # noqa: E402

from _vd import write                                         # noqa: E402

from lumenairy.propagators import carrier as CA               # noqa: E402
from lumenairy.propagators.carrier import _collins_transport  # noqa: E402


def ax(n, d):
    return (np.arange(int(n), dtype=np.float64) - int(n) / 2) * float(d)


# ---- the verification's relay family -------------------------------------
M = dict(lam=1.55e-6, n=1024, w=0.80e-3, na=0.04, ext=4.0, nout=64)
M['R0'] = -M['w'] / M['na']
M['dx'] = 2.0 * M['ext'] * M['w'] / M['n']
M['w0'] = M['lam'] * abs(M['R0']) / (np.pi * M['w'])
M['dxo'] = M['w0'] / 8.0


def relay_env(fr, w=None):
    k = 2.0 * np.pi / M['lam']
    g = ax(M['n'], M['dx'])
    r2 = g[None, :] ** 2 + g[:, None] ** 2
    return (np.exp(-r2 / float(w if w else M['w']) ** 2)
            * np.exp(1j * k * r2 / (2.0 * M['R0']))
            * np.exp(-1j * k * r2 / (2.0 * fr * M['R0']))).astype(
                np.complex128)


def theta_of(E, dx, lam):
    S = np.fft.fft2(np.ascontiguousarray(E, dtype=np.complex128))
    return max(CA._collins_envelope_half_angle(S, dx, dx, lam))


def run_leg(E, R_carrier, z, lam, dx, dxo, nout, gap_kernel='auto'):
    st = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _collins_transport(E, R_carrier, z, lam, dx, dx, dx_out=dxo,
                           dy_out=dxo, N_out_x=nout, N_out_y=nout,
                           R_ref=float('inf'), gap_kernel=gap_kernel,
                           on_collins_sampling='ignore', stats_out=st)
    A = 1.0 + float(z) / float(R_carrier)
    return dict(kernel=st['kernel'], k4=float(st['k4']),
                departure=float(st.get('kernel_departure', float('nan'))),
                A=A, B=float(z), z_eff=float(z) / A,
                d_to_A0=abs(float(z) - (-float(R_carrier))))


# ---- the b4 fixture, reproduced from the suite ----------------------------
#: ``test_audit2609_b4_collins_transport.py``'s ``mismatch_matrix``, from its
#: own definition: WP-A6's fixture, lambda 1.31 um, N 1024, w_in 1 mm,
#: NA 0.05 (R0 = -20 mm), ext 4, readout at ``z = -R0``.
B4 = dict(lam=1.31e-6, n=1024, w=1.0e-3, na=0.05, ext=4.0, nout=64)
B4['R0'] = -B4['w'] / B4['na']
B4['dx'] = 2.0 * B4['ext'] * B4['w'] / B4['n']
B4['w0'] = B4['lam'] * abs(B4['R0']) / (np.pi * B4['w'])
B4['dxo'] = B4['w0'] / 8.0


def b4_env(fr):
    k = 2.0 * np.pi / B4['lam']
    g = ax(B4['n'], B4['dx'])
    r2 = g[None, :] ** 2 + g[:, None] ** 2
    return (np.exp(-r2 / B4['w'] ** 2)
            * np.exp(1j * k * r2 / (2.0 * B4['R0']))
            * np.exp(-1j * k * r2 / (2.0 * fr * B4['R0']))).astype(
                np.complex128)


def main(out_path):
    warnings.simplefilter('ignore')
    res = {'tau': CA._GAP_KERNEL_ACCURACY_TAU}

    # (1) the relay ladder, readout AT the beam's focus
    rows = {}
    for fr in (0.85, 0.90, 0.95):
        E = relay_env(fr)
        r = run_leg(E, fr * M['R0'], -M['R0'], M['lam'], M['dx'], M['dxo'],
                    M['nout'])
        r['theta_env'] = theta_of(E, M['dx'], M['lam'])
        rows['fr=%.2f' % fr] = r
    res['mismatch_ladder'] = rows

    # (2) the same beam read out a QUARTER of the way to its focus
    z = 0.25 * abs(M['R0'])
    z_R = np.pi * M['w0'] ** 2 / M['lam']
    rows = {}
    for fr in (0.70, 0.80):
        E = relay_env(fr)
        r = run_leg(E, fr * M['R0'], z, M['lam'], M['dx'], M['dxo'],
                    M['nout'])
        r['theta_env'] = theta_of(E, M['dx'], M['lam'])
        r['beam_radius_here'] = M['w'] * (1.0 - z / abs(M['R0']))
        r['rayleigh_ranges_to_focus'] = (abs(M['R0']) - z) / z_R
        rows['fr=%.2f' % fr] = r
    res['quarter_distance'] = rows

    # (3) ONE leg, only the envelope's width moved
    D = dict(lam=1.55e-6, n=1024, dx=6.25e-6, R0=-20.0e-3, fr=0.70, nout=64)
    D['z'] = 0.25 * abs(D['R0'])
    D['dxo'] = (D['lam'] * abs(D['R0']) / (np.pi * 0.80e-3)) / 8.0
    rows = {}
    for w in (0.80e-3, 0.60e-3, 0.40e-3, 0.30e-3):
        k = 2.0 * np.pi / D['lam']
        g = ax(D['n'], D['dx'])
        r2 = g[None, :] ** 2 + g[:, None] ** 2
        E = (np.exp(-r2 / w ** 2) * np.exp(1j * k * r2 / (2.0 * D['R0']))
             * np.exp(-1j * k * r2 / (2.0 * D['fr'] * D['R0']))).astype(
                 np.complex128)
        r = run_leg(E, D['fr'] * D['R0'], D['z'], D['lam'], D['dx'], D['dxo'],
                    D['nout'])
        r['theta_env'] = theta_of(E, D['dx'], D['lam'])
        w0 = D['lam'] * abs(D['R0']) / (np.pi * w)
        r['rayleigh_ranges_to_focus'] = (
            (abs(D['R0']) - D['z']) / (np.pi * w0 ** 2 / D['lam']))
        r['beam_radius_here'] = w * (1.0 - D['z'] / abs(D['R0']))
        r['w_in'] = w
        rows['w=%.2fmm' % (w * 1e3)] = r
    res['same_leg_width'] = rows
    wide, narrow = rows['w=0.80mm'], rows['w=0.40mm']
    res['same_leg_quartic'] = dict(
        theta_ratio=wide['theta_env'] / narrow['theta_env'],
        departure_ratio=wide['departure'] / narrow['departure'],
        quartic_identity=((wide['theta_env'] / narrow['theta_env']) ** 4
                          / (wide['departure'] / narrow['departure'])),
        same_z_eff=(wide['z_eff'] == narrow['z_eff']),
        z_eff=wide['z_eff'], d_to_A0=wide['d_to_A0'])

    # (4) the b4 leg the suite actually moves
    rows = {}
    for fr in (0.90, 0.95):
        E = b4_env(fr)
        r = run_leg(E, fr * B4['R0'], -B4['R0'], B4['lam'], B4['dx'],
                    B4['dxo'], B4['nout'])
        r['theta_env'] = theta_of(E, B4['dx'], B4['lam'])
        rows['fr=%.2f' % fr] = r
    res['b4_leg'] = rows

    write(out_path, res)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1
         else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'r2_d6_band_win.json'))
