"""WP-C5 item 1 -- the near-focus exact-kernel switch, measured.

Emits one JSON with

  * ``tau``            -- the running tree's ``_GAP_KERNEL_ACCURACY_TAU``
  * ``h2_ladder``      -- the hygiene-2 ladder: resolved kernel + departure
  * ``f3_ladder``      -- VERIFY-B4 F3's ladder: resolved kernel, departure,
                          and relative L2 against the analytic Gaussian for
                          ``'auto'`` / ``'fresnel'`` / ``'exact'``
  * ``band``           -- the band edge (distance to focus where the predicted
                          departure crosses tau) found by bisection on each
                          fixture, plus the closed-form ``|z_eff|`` threshold
  * ``digests``        -- byte digests of a fixed leg set, for the
                          archive-to-archive comparison

Usage (BLAS pinned on the COMMAND LINE)::

    PYTHONPATH=<tree> python validation/probe_c5_three_defaults/c5_item1_kernel.py OUT.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                             # noqa: E402

from _digest import dig, write                                 # noqa: E402

from lumenairy.propagators import carrier as CA                # noqa: E402
from lumenairy.propagators.carrier import _collins_transport   # noqa: E402

# --------------------------------------------------------------------------
# fixture A -- the hygiene-2 ladder (derived, exactly as its test derives it)
# --------------------------------------------------------------------------
W0, THETA, F = 15.915e-6, 20.0e-3, 20.0e-3
LAM = float(np.pi * W0 * THETA)
ZR = float(np.pi * W0 ** 2 / LAM)
N_IN, DX_IN = 512, 8e-6

# --------------------------------------------------------------------------
# fixture B -- VERIFY-B4 F3's, the one whose ladder approaches A = 0
# --------------------------------------------------------------------------
F3 = dict(lam=1.064e-6, n=1024, dx=4e-6, w=0.30e-3, R=-40e-3,
          dx_out=5.6447e-06, n_out=128)


def axis(n, d):
    return (np.arange(int(n), dtype=np.float64) - int(n) / 2) * float(d)


def gauss(n, dx, w):
    x = axis(n, dx)
    return np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
                  / w ** 2).astype(np.complex128)


def q_field(n, dx, q_in, z, lam):
    k = 2.0 * np.pi / lam
    xo = axis(n, dx)
    xx, yy = np.meshgrid(xo, xo, indexing='xy')
    r2 = xx ** 2 + yy ** 2
    return (np.exp(1j * k * z) / (1.0 + z / q_in)
            * np.exp(1j * k * r2 / (2.0 * (q_in + z))))


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# --- fixture A helpers ------------------------------------------------------
def h2_env():
    q = complex(-F, -ZR)
    w_in = float(np.sqrt(LAM / (np.pi * np.imag(1.0 / q))))
    R_in = 1.0 / float(np.real(1.0 / q))
    return q, R_in, w_in, gauss(N_IN, DX_IN, w_in)


def h2_leg(d, gk, st=None, from_focus=False):
    """``d`` short of the WAIST (the published ladder) or, with
    ``from_focus``, short of the CARRIER's own ``A = 0`` plane."""
    _q, R_in, _w, env = h2_env()
    z0 = (-R_in) if from_focus else F
    return _collins_transport(
        env, R_in, z0 - d, LAM, DX_IN, DX_IN, dx_out=DX_IN, dy_out=DX_IN,
        N_out_x=N_IN, N_out_y=N_IN, R_ref=float('inf'), gap_kernel=gk,
        on_collins_sampling='ignore', stats_out=st)


# --- fixture B helpers ------------------------------------------------------
def f3_env():
    return gauss(F3['n'], F3['dx'], F3['w'])


def f3_leg(dz, gk, st=None):
    return _collins_transport(
        f3_env(), F3['R'], -F3['R'] - dz, F3['lam'], F3['dx'], F3['dx'],
        dx_out=F3['dx_out'], dy_out=F3['dx_out'], N_out_x=F3['n_out'],
        N_out_y=F3['n_out'], R_ref=float('inf'), gap_kernel=gk,
        on_collins_sampling='ignore', stats_out=st)


def f3_oracle(dz):
    lam, w, R = F3['lam'], F3['w'], F3['R']
    q_in = 1.0 / (1.0 / R + 1j * lam / (np.pi * w ** 2))
    return q_field(F3['n_out'], F3['dx_out'], q_in, -R - dz, lam)


def predicted(st, theta_env, lam):
    z_eff = abs(float(st['abcd'][1]) / float(st['abcd'][0]))
    return CA._collins_exact_kernel_departure(z_eff, theta_env, lam), z_eff


def theta_env_of(env, dx, lam):
    S = np.fft.fft2(np.ascontiguousarray(env, dtype=np.complex128))
    return max(CA._collins_envelope_half_angle(S, dx, dx, lam))


def band_edge(leg, lo, hi, theta_env, lam, tau):
    """Bisect for the distance-to-focus at which the PREDICTED departure
    crosses ``tau``.  Monotone because the departure is linear in |z_eff| and
    |z_eff| falls monotonically with the distance to ``A = 0``."""
    def dep(d):
        st = {}
        leg(d, 'fresnel', st)
        return predicted(st, theta_env, lam)[0]

    if dep(lo) < tau or dep(hi) > tau:
        return None
    for _ in range(60):
        mid = float(np.sqrt(lo * hi))
        if dep(mid) > tau:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def main(out):
    tau = CA._GAP_KERNEL_ACCURACY_TAU
    res = {'tau': repr(tau)}
    digests = {}

    _q, _R, _w, env_a = h2_env()
    th_a = theta_env_of(env_a, DX_IN, LAM)
    th_b = theta_env_of(f3_env(), F3['dx'], F3['lam'])
    res['theta_env'] = {'h2': th_a, 'f3': th_b}

    # ---- fixture A: the published ladder (distances from the WAIST) -------
    h2 = []
    for d in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 5e-3):
        st = {}
        fld = h2_leg(d, 'auto', st)
        pred, z_eff = predicted(st, th_a, LAM)
        h2.append(dict(d=d, kernel=st['kernel'], k4=st['k4'], z_eff=z_eff,
                       departure=st.get('kernel_departure'), predicted=pred))
        digests['h2/auto/%.0e' % d] = dig(fld)
    res['h2_ladder'] = h2

    # ---- fixture A walked to its OWN carrier focus -----------------------
    h2f = []
    for d in (1e-7, 3e-7, 1e-6, 3e-6, 1e-5):
        st = {}
        h2_leg(d, 'fresnel', st, from_focus=True)
        pred, z_eff = predicted(st, th_a, LAM)
        h2f.append(dict(d=d, z_eff=z_eff, predicted=pred))
    res['h2_ladder_from_carrier_focus'] = h2f

    # ---- fixture B: the F3 ladder, with the analytic oracle ---------------
    f3 = []
    for dz in (1e-6, 1e-5, 1e-4, 1e-3, 5e-3):
        row = dict(dz=dz)
        ref = f3_oracle(dz)
        for gk in ('auto', 'fresnel', 'exact'):
            st = {}
            fld = f3_leg(dz, gk, st)
            row[gk] = dict(kernel=st['kernel'], k4=st['k4'],
                           departure=st.get('kernel_departure'),
                           rel_oracle=rel(fld, ref))
            digests['f3/%s/%.0e' % (gk, dz)] = dig(fld)
        st = {}
        f3_leg(dz, 'fresnel', st)
        row['predicted'], row['z_eff'] = predicted(st, th_b, F3['lam'])
        f3.append(row)
    res['f3_ladder'] = f3

    # ---- the band ---------------------------------------------------------
    tau_f = 1e-4 if tau is None else float(tau)
    band = {'tau_used': tau_f}
    band['f3_edge_m'] = band_edge(f3_leg, 1e-7, 1e-2, th_b, F3['lam'], tau_f)
    band['h2_waist_edge_m'] = band_edge(
        h2_leg, 1e-8, 1e-2, th_a, LAM, tau_f)
    band['h2_focus_edge_m'] = band_edge(
        lambda d, gk, st: h2_leg(d, gk, st, from_focus=True),
        1e-9, 1e-2, th_a, LAM, tau_f)
    # the closed form: the rule fires where |z_eff| > 8 tau / (C k theta^4)
    for nm, th, lam in (('f3', th_b, F3['lam']), ('h2', th_a, LAM)):
        k = 2.0 * np.pi / lam
        band['%s_z_eff_threshold_m' % nm] = float(
            8.0 * tau_f / (CA._QUARTIC_RMS_MOMENT * k * th ** 4))
    res['band'] = band

    # ---- explicit 'exact' is honoured over tau ---------------------------
    st = {}
    f3_leg(1e-6, 'exact', st)
    res['explicit_exact_kernel_at_1um'] = st['kernel']

    # ---- the opt-out: tau = None ----------------------------------------
    old = CA._GAP_KERNEL_ACCURACY_TAU
    try:
        CA._GAP_KERNEL_ACCURACY_TAU = None
        for d in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 5e-3):
            digests['optout/h2/auto/%.0e' % d] = dig(h2_leg(d, 'auto'))
        for dz in (1e-6, 1e-5, 1e-4, 1e-3, 5e-3):
            for gk in ('auto', 'fresnel', 'exact'):
                digests['optout/f3/%s/%.0e' % (gk, dz)] = dig(f3_leg(dz, gk))
    finally:
        CA._GAP_KERNEL_ACCURACY_TAU = old

    res['digests'] = digests
    write(out, res)


if __name__ == '__main__':
    main(sys.argv[1])
