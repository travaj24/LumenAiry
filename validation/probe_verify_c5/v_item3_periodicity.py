"""VERIFY-WP-C5 defect D5 -- how each readout's reconstruction is periodic.

``_fill_readout_replicas`` justifies its geometry with "``E(u + period) ==
E(u)`` identically in ABSOLUTE output coordinates".  This probe measures that
statement on all three readouts the function serves, on the complex field AND
on its modulus, at a shift of exactly one period.

It also checks the closed form the Collins readout actually obeys,

    E(u + period) = exp(i [2 pi u / dx_in + pi lambda z / dx_in^2]) E(u)

which is its post-chirp ``exp(i k D x^2 / 2B)`` evaluated at ``x`` and
``x + period`` -- a phase that is unity only where ``u / dx_in`` is an integer.

Usage (BLAS pinned on the COMMAND LINE)::

    PYTHONPATH=<tree> python validation/probe_verify_c5/v_item3_periodicity.py OUT.json
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                            # noqa: E402

from _vd import write                                         # noqa: E402

import lumenairy as la                                        # noqa: E402
from lumenairy.propagators import carrier as C                # noqa: E402

WL, RMAG, W_IN = 1.55e-6, -25.0e-3, 0.60e-3
N_IN, DX_IN = 512, 5.0e-6
P = WL * abs(RMAG) / DX_IN


def ax(n, d):
    return (np.arange(int(n), dtype=np.float64) - int(n) / 2) * float(d)


def pupil():
    x = ax(N_IN, DX_IN)
    return np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
                  / W_IN ** 2).astype(np.complex128)


def residuals(F, m):
    """Complex and modulus residuals at an integer shift ``m``, / peak."""
    F = np.asarray(F)
    n = F.shape[1]
    sc = float(np.abs(F).max())
    A, B = F[:, m:], F[:, :n - m]
    return (float(np.max(np.abs(A - B))) / sc,
            float(np.max(np.abs(np.abs(A) - np.abs(B)))) / sc, sc)


def main(out_path):
    warnings.simplefilter('ignore')
    res = {}
    n, wp, m = 256, 1.60, 160          # 256/1.60 = 160 samples per period

    # --- the Collins readout -------------------------------------------
    dxo = wp * P / n
    for tag, c in (('on_axis', 0.0), ('off_0.30p', 0.30 * P)):
        pd = {}
        F = np.asarray(C._collins_focus_readout(
            pupil(), RMAG, -RMAG, WL, DX_IN, DX_IN, dx_out=dxo, N_out=n,
            centre_out=(c, 0.0), on_replica='ignore',
            on_collins_sampling='ignore', replica_fill='repeat',
            _period_out=pd))
        rc, ra, sc = residuals(F, m)
        row = dict(period=float(min(pd['period'])), dx_out=dxo, shift=m,
                   complex_resid=rc, modulus_resid=ra, peak=sc)
        if tag == 'off_0.30p':
            # the closed form, on the samples that carry signal
            u = ax(n, dxo) + c
            pred = np.exp(1j * (2.0 * np.pi * u[:n - m] / DX_IN
                                + np.pi * WL * (-RMAG) / DX_IN ** 2))
            A, B = F[:, m:], F[:, :n - m]
            sel = np.abs(B) > 1e-6 * sc
            got = (A / np.where(np.abs(B) > 0, B, 1))[sel]
            pr = np.broadcast_to(pred[None, :], B.shape)[sel]
            row['closed_form_max_abs_error'] = float(np.max(np.abs(got - pr)))
            row['closed_form_n_pairs'] = int(sel.sum())
            row['pi_lambda_z_over_dx2_in_pi'] = float(
                WL * (-RMAG) / DX_IN ** 2)
        res['collins/' + tag] = row

    # --- the paraxial (Sziklas) readout ---------------------------------
    pd0 = {}
    la.carrier_referenced_focus_readout(
        pupil(), RMAG, -RMAG, WL, DX_IN, dx_out=1.0e-6, N_out=32,
        on_replica='ignore', on_focus_containment='ignore', _period_out=pd0)
    ps = float(min(pd0['period']))
    dxs = wp * ps / n
    for tag, c in (('on_axis', 0.0), ('off_0.30p', 0.30 * ps)):
        pd = {}
        F = np.asarray(la.carrier_referenced_focus_readout(
            pupil(), RMAG, -RMAG, WL, DX_IN, dx_out=dxs, N_out=n,
            centre_out=(c, 0.0), on_replica='ignore',
            on_focus_containment='ignore', replica_fill='repeat',
            _period_out=pd))
        rc, ra, sc = residuals(F, m)
        res['sziklas/' + tag] = dict(period=float(min(pd['period'])),
                                     dx_out=dxs, shift=m, complex_resid=rc,
                                     modulus_resid=ra, peak=sc)

    # --- the exact readout ----------------------------------------------
    n2, dx2, w2, R2 = 512, 0.5e-6, 30e-6, -0.2e-3
    x = ax(n2, dx2)
    r2 = x[:, None] ** 2 + x[None, :] ** 2
    S = np.sign(R2) * (np.sqrt(r2 + R2 * R2) - abs(R2))
    E = (np.exp(-r2 / w2 ** 2)
         * np.exp(1j * 2.0 * np.pi / WL * S)).astype(np.complex128)
    pd = {}
    F = np.asarray(la.carrier_referenced_exact_focus_readout(
        E, R2, -R2, WL, dx2, dx_out=0.05e-6, N_out=3072, window_factor=4.0,
        on_replica='ignore', replica_fill='repeat', _period_out=pd))
    pe = float(min(pd['period']))
    me = int(round(pe / 0.05e-6))
    rc, ra, sc = residuals(F, me)
    res['exact/on_axis'] = dict(period=pe, dx_out=0.05e-6, shift=me,
                                shift_in_periods=me * 0.05e-6 / pe,
                                complex_resid=rc, modulus_resid=ra, peak=sc)

    write(out_path, {'periodicity': res})


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1
         else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'v_item3_periodicity_win.json'))
