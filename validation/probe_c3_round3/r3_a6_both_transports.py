"""WP-C3 round 3 -- measure the 8 a6 ids on BOTH transports against oracles
that owe nothing to either quadrature.

ORACLES
  O1  the analytic Gaussian-ABCD focal field (whole-function q form), written
      here; it is the truth the two ids in ``TestVerifyC1AgainstTheAnalyticFocus``
      already grade against.
  O2  the EXACT second-moment law  <r^2>(z) = <r^2> + 2 z <r.theta> + z^2 <theta^2>
      read off the INPUT (exit) field by spectral moments -- no propagator at
      all.  It gives the beam's true amplitude radius at the stop plane, which
      is what the containment guard's ratio is a ratio OF.
  O3  power conservation through the leg.

    python c3d_a6_measure.py <tree> <out.json>
"""
from __future__ import annotations

import json
import os
import platform
import sys
import warnings

import numpy as np

TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, TREE)
import lumenairy                                            # noqa: E402
from lumenairy.propagators import carrier as C              # noqa: E402

assert os.path.abspath(lumenairy.__file__).lower().startswith(TREE.lower()), (
    lumenairy.__file__, TREE)
print('[bind] lumenairy.__file__ =', lumenairy.__file__, flush=True)
import inspect                                              # noqa: E402
_DEF = inspect.signature(
    C.carrier_referenced_focus_readout).parameters['transport'].default
print('[bind] readout transport default =', _DEF, flush=True)

OUT = {'tree': TREE, 'lumenairy': lumenairy.__file__,
       'python': sys.version.split()[0], 'numpy': np.__version__,
       'platform': platform.platform(),
       'readout_default_transport': _DEF}


# ---------------------------------------------------------------------------
# oracles
# ---------------------------------------------------------------------------
def grid(n, dx):
    return (np.arange(n, dtype=np.float64) - n / 2) * dx


def gauss(n, dx, w):
    g = grid(n, dx)
    return np.exp(-((g[None, :] ** 2 + g[:, None] ** 2) / w ** 2))


def abcd_field(xo, w_in, r_beam, z, lam):
    kk = 2.0 * np.pi / lam
    inv_q = 1.0 / r_beam + 1j * lam / (np.pi * w_in ** 2)
    q = 1.0 / inv_q
    q2 = q + z
    wz = np.sqrt(lam / (np.pi * np.imag(1.0 / q2)))
    r2 = xo[None, :] ** 2 + xo[:, None] ** 2
    E = (np.exp(1j * kk * z) * (q / q2)
         * np.exp(1j * kk * r2 / (2.0 * q2)))
    return E, wz


def moment_law(E, dx, lam, z):
    """O2: the exact free-space second-moment law, read off ``E`` alone.

    Returns (w_of_z, w_of_0) as 1/e AMPLITUDE radii, the same convention
    ``_envelope_amp_radius`` uses (w = sqrt(2 <r^2>)).
    """
    E = np.asarray(E)
    n = E.shape[-1]
    x = grid(n, dx)
    I = np.abs(E) ** 2
    P = I.sum()
    r2 = float((I * (x[None, :] ** 2 + x[:, None] ** 2)).sum()) / P
    # <theta^2> from the angular spectrum (carries both phase tilt and
    # diffraction; no propagator involved).
    F = np.fft.fft2(E)
    S = np.abs(F) ** 2
    fx = np.fft.fftfreq(n, dx)
    th2 = float((S * (lam * lam)
                 * (fx[None, :] ** 2 + fx[:, None] ** 2)).sum()) / S.sum()
    # <r.theta> = (1/k) Im[ int r . E* grad E ] / P
    k = 2.0 * np.pi / lam
    gy, gx = np.gradient(E, dx, dx)
    rt = float(np.imag((np.conj(E) * (x[None, :] * gx + x[:, None] * gy)).sum())
               ) / P / k
    r2z = r2 + 2.0 * z * rt + z * z * th2
    return float(np.sqrt(2.0 * max(r2z, 0.0))), float(np.sqrt(2.0 * r2))


def run_readout(env, R, z, lam, dx, transport, **kw):
    pd = {}
    msgs = []
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        try:
            f = C.carrier_referenced_focus_readout(
                env, R, z, lam, dx, transport=transport, _period_out=pd, **kw)
        except Exception as exc:                            # noqa: BLE001
            return dict(raised=type(exc).__name__, msg=str(exc)[:200],
                        pd={k: _j(v) for k, v in pd.items()})
        msgs = [str(x.message)[:120] for x in wl]
    return dict(raised=None, field=f, pd={k: _j(v) for k, v in pd.items()},
                warnings=msgs)


def _j(v):
    if isinstance(v, tuple):
        return [float(x) for x in v]
    if v is None or isinstance(v, str):
        return v
    return float(v)


def grade(f, truth):
    pist = np.angle(np.vdot(truth, f))
    rel = float(np.linalg.norm(f * np.exp(-1j * pist) - truth)
                / np.linalg.norm(truth))
    peak = float((np.abs(f) ** 2).max() / (np.abs(truth) ** 2).max())
    return peak, rel


def pre_fix_standoff(env, r, z, lam, dx):
    real = C._beam_containment_standoff
    try:
        C._beam_containment_standoff = lambda *a, **k: 0.0
        return C._default_focus_standoff(env, r, z, lam, dx)
    finally:
        C._beam_containment_standoff = real


TR = ('sziklas', 'collins')


# ===========================================================================
# A. the a6 (test_audit2609_a6_carrier.py) C1 fixture -- ids 1 and 2
# ===========================================================================
LAM_A = 1.31e-6
K0_A = 2.0 * np.pi / LAM_A


def c1_fixture(n=1024, w_in=1.0e-3, na=0.05, ext=4.0):
    r0 = -w_in / na
    dx = 2.0 * ext * w_in / n
    g = grid(n, dx)
    r2 = g[None, :] ** 2 + g[:, None] ** 2
    e = np.exp(-r2 / w_in ** 2) * np.exp(1j * K0_A * r2 / (2.0 * r0))
    w0 = LAM_A * abs(r0) / (np.pi * w_in)
    return e, r0, dx, w0, w_in


def sec_A():
    e_phys, r0, dx, w0, w_in = c1_fixture()
    z = -r0
    xo = grid(64, w0 / 8.0)
    truth, wz = abcd_field(xo, w_in, r0, z, LAM_A)
    rows = {}
    for frac in (0.98, 0.95, 0.90):
        r = frac * r0
        env = C.carrier_referenced_envelope(e_phys, r, LAM_A, dx)
        s_old = pre_fix_standoff(env, r, z, LAM_A, dx)
        z_stop = z - abs(s_old)
        w_true_stop, w_true_in = moment_law(e_phys, dx, LAM_A, z_stop)
        row = {'frac': frac, 'pre_fix_standoff': float(s_old),
               'z_stop': float(z_stop),
               'w_true_at_stop_um': w_true_stop * 1e6,
               'w_true_at_input_um': w_true_in * 1e6}
        for tr in TR:
            got = run_readout(env, r, z, LAM_A, dx, tr,
                              dx_out=w0 / 8.0, N_out=64, standoff=s_old,
                              on_replica='ignore',
                              on_focus_containment='ignore')
            d = {'pd': got['pd'], 'warnings': got['warnings'],
                 'raised': got['raised']}
            if got['raised'] is None:
                pk, rel = grade(got['field'], truth)
                d['peak_vs_abcd'] = pk
                d['relL2_vs_abcd'] = rel
                # O2: what half-width did the leg's grid offer, against the
                # TRUE beam radius there?
                pdd = got['pd']
                d['half_over_w_true'] = (
                    None if 'containment' not in pdd else
                    float(pdd['containment'] * pdd.get('containment', 1.0) * 0))
            rows.setdefault(tr, []).append(d)
        # what grid each leg returned
        for tr in TR:
            cr = C.propagate_carrier_referenced(env, r, z_stop, LAM_A, dx,
                                                transport=tr)
            dxs = cr.dx[0] if isinstance(cr.dx, tuple) else cr.dx
            nn = min(np.shape(cr.env)[-2:])
            half = 0.5 * nn * float(dxs)
            row[f'{tr}_stop_dx_um'] = float(dxs) * 1e6
            row[f'{tr}_stop_half_um'] = half * 1e6
            row[f'{tr}_half_over_w_true'] = half / w_true_stop
            row[f'{tr}_power_ratio'] = float(
                (np.abs(np.asarray(cr.env)) ** 2).sum() * float(dxs) ** 2
                / ((np.abs(env) ** 2).sum() * dx * dx))
        rows.setdefault('geom', []).append(row)
    # the guard: does it fire, and is the peak actually collapsed?
    guard = []
    for frac in (0.95, 0.90):
        r = frac * r0
        env = C.carrier_referenced_envelope(e_phys, r, LAM_A, dx)
        s_old = pre_fix_standoff(env, r, z, LAM_A, dx)
        ent = {'frac': frac}
        for tr in TR:
            strict = run_readout(env, r, z, LAM_A, dx, tr,
                                 dx_out=w0 / 8.0, N_out=64, standoff=s_old,
                                 on_replica='ignore')
            ent[f'{tr}_default_disposition'] = strict['raised'] or 'returned'
            ig = run_readout(env, r, z, LAM_A, dx, tr,
                             dx_out=w0 / 8.0, N_out=64, standoff=s_old,
                             on_replica='ignore',
                             on_focus_containment='ignore')
            if ig['raised'] is None:
                pk, rel = grade(ig['field'], truth)
                ent[f'{tr}_peak_vs_abcd'] = pk
                ent[f'{tr}_relL2_vs_abcd'] = rel
                ent[f'{tr}_containment'] = ig['pd'].get('containment')
                ent[f'{tr}_containment_model'] = ig['pd'].get(
                    'containment_model')
        guard.append(ent)
    return {'rows': rows, 'guard': guard,
            'w0_um': w0 * 1e6, 'wz_oracle_um': wz * 1e6}


# ===========================================================================
# B. the verify-a6 downward-quadratic fixture -- id 3
# ===========================================================================
def sec_B():
    lam = 0.85e-6
    k = 2.0 * np.pi / lam
    n, w2, ext, r = 512, 200e-6, 6.0, -20e-3
    inv_res = -60.0
    dx = 2.0 * ext * w2 / n
    g = grid(n, dx)
    r2 = g[None, :] ** 2 + g[:, None] ** 2
    env = np.exp(-r2 / w2 ** 2) * np.exp(1j * k * r2 * 0.5 * inv_res)
    z = -r
    cen = C._envelope_amp_centroid(env, dx, dx)
    w_env = C._envelope_amp_radius(env, dx, dx, centre=cen)
    half = 0.5 * n * dx - max(abs(cen[0]), abs(cen[1]))
    inv_fit = C._fit_carrier_inv(env, lam, dx, dx, axis=None,
                                 estimator='increment', centre=cen)
    s_new = C._beam_containment_standoff(env, r, z, lam, dx, w_env, cen,
                                         half, inv_env=inv_fit)
    s_res = C._default_focus_standoff(env, r, z, lam, dx)
    out = {'w_env_um': w_env * 1e6, 'half_um': half * 1e6,
           'inv_fit': float(inv_fit),
           's_beam_um': float(s_new) * 1e6,
           's_resolved_um': float(s_res) * 1e6,
           'M': float(C._FOCUS_STANDOFF_MARGIN)}
    # the envelope is referenced to the carrier r; the PHYSICAL field is
    # env * exp(i k r^2 / 2r).
    e_phys = env * np.exp(1j * k * r2 / (2.0 * r))
    for tr in TR:
        got = run_readout(env, r, z, lam, dx, tr, dx_out=1e-6, N_out=48,
                          on_replica='ignore', on_focus_containment='warn')
        out[f'{tr}_pd'] = got['pd']
        out[f'{tr}_warnings'] = got['warnings']
        out[f'{tr}_raised'] = got['raised']
        z_stop = z - abs(s_res)
        w_true, _ = moment_law(e_phys, dx, lam, z_stop)
        cr = C.propagate_carrier_referenced(env, r, z_stop, lam, dx,
                                            transport=tr)
        dxs = cr.dx[0] if isinstance(cr.dx, tuple) else cr.dx
        nn = min(np.shape(cr.env)[-2:])
        out[f'{tr}_stop_half_over_w_true'] = (0.5 * nn * float(dxs)) / w_true
        out[f'{tr}_stop_dx_um'] = float(dxs) * 1e6
        out['w_true_at_stop_um'] = w_true * 1e6
    # the FAIL-BEFORE gate (alpha > 0 restored)
    real = C._beam_containment_standoff

    def _gated(*a, **kw):
        r_ = a[1]
        zeta_cf = -float(r_)
        w_, half_ = a[5], a[7]
        iv = kw.get('inv_env')
        if iv is None:
            iv = 0.0
        cc = 1.0 / float(r_) + float(iv)
        zr_ = np.pi * w_ * w_ / a[3]
        al = (half_ ** 2 / zeta_cf ** 2
              - (C._FOCUS_STANDOFF_MARGIN * w_) ** 2
              * (cc * cc + 1.0 / zr_ ** 2))
        return real(*a, **kw) if al > 0.0 else 0.0
    try:
        C._beam_containment_standoff = _gated
        s_pre = C._default_focus_standoff(env, r, z, lam, dx)
        out['pre_fix_gate_standoff_um'] = float(s_pre) * 1e6
        for tr in TR:
            got = run_readout(env, r, z, lam, dx, tr, dx_out=1e-6, N_out=48,
                              on_replica='ignore')
            out[f'{tr}_gated_default'] = got['raised'] or 'returned'
            out[f'{tr}_gated_pd'] = got['pd']
    finally:
        C._beam_containment_standoff = real
    return out


# ===========================================================================
# C. the analytic-focus fixtures -- ids 4-8
# ===========================================================================
LAM_C = 0.85e-6


def c_fixture(n=1024, w_in=0.6e-3, na=0.08, ext=4.0):
    k = 2.0 * np.pi / LAM_C
    r0 = -w_in / na
    dx = 2.0 * ext * w_in / n
    g = grid(n, dx)
    r2 = g[None, :] ** 2 + g[:, None] ** 2
    e = np.exp(-r2 / w_in ** 2) * np.exp(1j * k * r2 / (2.0 * r0))
    return e, r0, dx, w_in


def sec_C(ext, fracs):
    lam = LAM_C
    e_phys, r0, dx, w_in = c_fixture(ext=ext)
    z = -r0
    _, w0 = abcd_field(np.zeros(1), w_in, r0, z, lam)
    n_out, dx_out = 96, w0 / 8.0
    xo = grid(n_out, dx_out)
    truth, _ = abcd_field(xo, w_in, r0, z, lam)
    rows = []
    for frac in fracs:
        r = frac * r0
        env = C.carrier_referenced_envelope(e_phys, r, lam, dx)
        s_pre = pre_fix_standoff(env, r, z, lam, dx)
        row = {'frac': frac, 'pre_fix_standoff_um': float(s_pre) * 1e6}
        for tr in TR:
            # post-fix (resolved) leg, default dispositions apart from replica
            got = run_readout(env, r, z, lam, dx, tr, dx_out=dx_out,
                              N_out=n_out, on_replica='ignore')
            row[f'{tr}_post_raised'] = got['raised']
            if got['raised'] is None:
                pk, rel = grade(got['field'], truth)
                row[f'{tr}_post_peak'] = pk
                row[f'{tr}_post_relL2'] = rel
                row[f'{tr}_post_pd'] = got['pd']
                row[f'{tr}_post_warnings'] = got['warnings']
            # pre-fix leg, guard waived
            got2 = run_readout(env, r, z, lam, dx, tr, dx_out=dx_out,
                               N_out=n_out, standoff=s_pre,
                               on_replica='ignore',
                               on_focus_containment='ignore')
            if got2['raised'] is None:
                pk, rel = grade(got2['field'], truth)
                row[f'{tr}_pre_peak'] = pk
                row[f'{tr}_pre_relL2'] = rel
                row[f'{tr}_pre_pd'] = got2['pd']
            else:
                row[f'{tr}_pre_raised'] = got2['raised']
            # pre-fix leg, DEFAULT disposition (does the guard refuse?)
            got3 = run_readout(env, r, z, lam, dx, tr, dx_out=dx_out,
                               N_out=n_out, standoff=s_pre,
                               on_replica='ignore')
            row[f'{tr}_pre_default'] = got3['raised'] or 'returned'
            # O2 on the pre-fix stop plane
            z_stop = z - abs(s_pre)
            w_true, _ = moment_law(e_phys, dx, lam, z_stop)
            cr = C.propagate_carrier_referenced(env, r, z_stop, lam, dx,
                                                transport=tr)
            dxs = cr.dx[0] if isinstance(cr.dx, tuple) else cr.dx
            nn = min(np.shape(cr.env)[-2:])
            row[f'{tr}_pre_half_over_w_true'] = (0.5 * nn * float(dxs)) / w_true
            row[f'{tr}_pre_stop_dx_um'] = float(dxs) * 1e6
            row[f'{tr}_pre_power_ratio'] = float(
                (np.abs(np.asarray(cr.env)) ** 2).sum() * float(dxs) ** 2
                / ((np.abs(env) ** 2).sum() * dx * dx))
            row['w_true_at_pre_stop_um'] = w_true * 1e6
        rows.append(row)
    return {'ext': ext, 'w0_um': w0 * 1e6, 'dx_out_um': dx_out * 1e6,
            'rows': rows}


if __name__ == '__main__':
    OUT['A_c1_fixture'] = sec_A()
    print('A done', flush=True)
    OUT['B_downward_quadratic'] = sec_B()
    print('B done', flush=True)
    OUT['C_analytic_ext4'] = sec_C(4.0, (1.00, 0.99, 0.97, 0.93, 0.90))
    print('C4 done', flush=True)
    OUT['C_analytic_ext3'] = sec_C(3.0, (1.00, 0.99, 0.97, 0.93))
    print('C3 done', flush=True)
    with open(sys.argv[2], 'w', encoding='utf-8') as fh:
        json.dump(OUT, fh, indent=1, default=str)
    print('WROTE', sys.argv[2])
