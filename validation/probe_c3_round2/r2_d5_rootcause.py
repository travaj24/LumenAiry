"""WP-C3 round 2 -- D5 root cause.

For the ORDINARY two-group relay of probe_vc3_newraise_indep.py, read the
CHAIN EXIT state the readout is handed on both transports, then the Sziklas
readout's standoff leg and the containment it reads from each exit.

Nothing here propagates an oracle; it reads the library's own state so the
question "is the Collins exit's support larger, or is the base's smaller an
aliasing artefact" can be answered from the exit fields themselves via the
exact free-space second-moment law.
"""
import json
import os
import sys
import warnings

import numpy as np

TREE = os.path.abspath(os.environ['VC3_TREE'])
sys.path.insert(0, TREE)
import lumenairy  # noqa: E402
from lumenairy.propagators import carrier as C  # noqa: E402

assert os.path.abspath(lumenairy.__file__).startswith(TREE), (
    lumenairy.__file__, TREE)
LAM = 1.31e-6
TKW = dict(on_undersample='silent', on_noncollimated='silent')


def singlet(r1, r2, t, glass, ap):
    return {'name': 'p', 'aperture_diameter': ap, 'thicknesses': [t],
            'surfaces': [
                {'radius': r1, 'glass_before': 'air', 'glass_after': glass,
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
                {'radius': r2, 'glass_before': glass, 'glass_after': 'air',
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]}


def gauss(n, dx, w):
    x = (np.arange(n) - n // 2) * dx
    xx, yy = np.meshgrid(x, x, indexing='ij')
    return np.exp(-(xx ** 2 + yy ** 2) / w ** 2).astype(np.complex128)


def amp_radius(a, dx):
    ii = np.abs(np.asarray(a)) ** 2
    n = ii.shape[0]
    ax = (np.arange(n) - n // 2) * dx
    xx, yy = np.meshgrid(ax, ax, indexing='ij')
    return float(np.sqrt(((xx ** 2 + yy ** 2) * ii).sum() / ii.sum()))


def moment_law(env, R, dx, z):
    """<r^2>(z) = <r^2> + 2 z <r.theta> + z^2 <theta^2>, exact in free space,
    read off the PHYSICAL field (carrier reconstructed)."""
    n = np.shape(env)[0]
    ax = (np.arange(n) - n // 2) * dx
    xx, yy = np.meshgrid(ax, ax, indexing='ij')
    kk = 2.0 * np.pi / LAM
    rx, _ry, _ = C._parse_carrier(R, 'moment')
    phys = np.asarray(env) * (
        1.0 if not np.isfinite(rx)
        else np.exp(1j * kk * (xx ** 2 + yy ** 2) / (2.0 * rx)))
    ii = np.abs(phys) ** 2
    r2 = float(((xx ** 2 + yy ** 2) * ii).sum() / ii.sum())
    spec = np.fft.fft2(phys)
    ps = np.abs(spec) ** 2
    fr = np.fft.fftfreq(n, d=dx)
    fx, fy = np.meshgrid(fr, fr, indexing='ij')
    th2 = float((((LAM * fx) ** 2 + (LAM * fy) ** 2) * ps).sum() / ps.sum())
    mix = float(np.imag(np.sum(np.conj(phys) * (
        xx * np.fft.ifft2(spec * (2j * np.pi * fx))
        + yy * np.fft.ifft2(spec * (2j * np.pi * fy)))))
        / ii.sum() / kk)
    return dict(r2m=float(np.sqrt(r2)),
                r2m_at_z=float(np.sqrt(max(r2 + 2 * z * mix + z * z * th2,
                                           0.0))),
                theta_rms=float(np.sqrt(th2)))


def spy_exit(n, dx, w, transport, fd):
    """Capture exactly what the chain hands the readout."""
    p = singlet(120e-3, -120e-3, 6e-3, 'N-BK7', 25.4e-3)
    groups = [{'prescription': p, 'gap_before': 20e-3},
              {'prescription': p, 'gap_before': 15e-3}]
    seen = {}
    real_sz = C.carrier_referenced_focus_readout
    real_col = C._collins_focus_readout

    def sz(env, R, z, wl_, dx_, **kw):
        seen.update(route='sziklas', env=np.array(env), R=R, z=z,
                    dx=float(dx_), kw=sorted(kw))
        return real_sz(env, R, z, wl_, dx_, **kw)

    def col(env, R, z, wl_, dx_, dy_, **kw):
        seen.update(route='collins', env=np.array(env), R=R, z=z,
                    dx=float(dx_), kw=sorted(kw))
        return real_col(env, R, z, wl_, dx_, dy_, **kw)

    C.carrier_referenced_focus_readout = sz
    C._collins_focus_readout = col
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter('always')
            try:
                res = C.propagate_traced_carrier_chain(
                    gauss(n, dx, w), groups, LAM, dx, r_in=np.inf,
                    ray_subsample=16, n_workers=1, traced_kwargs=TKW,
                    final_leg='paraxial', final_distance=fd,
                    focus_readout=dict(dx_out=0.5e-6, N_out=64),
                    transport=transport)
                outcome, msg = 'returned', ''
                peak = float(np.max(np.abs(res.field) ** 2))
            except BaseException as exc:            # noqa: BLE001
                outcome, msg, peak = 'raised', str(exc)[:220], None
        warns = [str(x.message)[:70] for x in wlist
                 if 'carrier' in str(x.filename)]
    finally:
        C.carrier_referenced_focus_readout = real_sz
        C._collins_focus_readout = real_col
    row = dict(outcome=outcome, msg=msg, peak=peak, warns=warns)
    if seen:
        env, R, dxe = seen['env'], seen['R'], seen['dx']
        rx, _ry, _ = C._parse_carrier(R, 'x')
        r_x, _r2, th_x, _t2 = C._collins_input_box(
            env, dxe, dxe, LAM, C._COLLINS_TAIL_FRAC)
        row.update(readout_route_taken=seen['route'], exit_dx_um=dxe * 1e6,
                   exit_R_mm=float(rx) * 1e3,
                   exit_halfwidth_um=0.5 * env.shape[0] * dxe * 1e6,
                   exit_box_r_um=r_x * 1e6, exit_box_theta_mrad=th_x * 1e3,
                   exit_amp_radius_um=amp_radius(env, dxe) * 1e6,
                   exit_env_power=float((np.abs(env) ** 2).sum()) * dxe * dxe,
                   readout_kw=seen['kw'])
        row.update({('moment_' + k): v
                    for k, v in moment_law(env, R, dxe, fd).items()})
        cen = C._envelope_amp_centroid(env, dxe, dxe)
        wenv = C._envelope_amp_radius(env, dxe, dxe, centre=cen)
        inv = C._fit_carrier_inv(env, LAM, dxe, dxe, axis=None,
                                 estimator='increment', centre=cen)
        so = C._default_focus_standoff(env, float(rx), fd, LAM, dxe,
                                       inv_env=inv, cen=cen, w_env=wenv)
        row.update(w_env_um=wenv * 1e6, inv_env=float(inv),
                   standoff_mm=float(so) * 1e3)
        for name in ('sziklas', 'collins'):
            try:
                zs = fd - np.copysign(min(float(so), abs(fd)), fd)
                cr = C.propagate_carrier_referenced(
                    env, float(rx), zs, LAM, dxe, transport=name)
                dxs = cr.dx[0] if isinstance(cr.dx, tuple) else cr.dx
                ce = C._envelope_amp_centroid(cr.env, dxs, dxs)
                ws = C._envelope_amp_radius(cr.env, dxs, dxs, centre=ce)
                half = 0.5 * min(np.shape(cr.env)) * float(dxs) - max(
                    abs(ce[0]), abs(ce[1]))
                row['stop_' + name] = dict(
                    dx_stop_um=float(dxs) * 1e6, w_stop_um=ws * 1e6,
                    half_stop_um=half * 1e6, containment=half / ws,
                    R_stop_mm=float(cr.R[0] if isinstance(cr.R, tuple)
                                    else cr.R) * 1e3)
            except BaseException as exc:            # noqa: BLE001
                row['stop_' + name] = dict(exc=type(exc).__name__,
                                           msg=str(exc)[:120])
    return row


def main():
    out = {'tree': TREE, 'lumenairy': lumenairy.__file__,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'rows': []}
    nan = float('nan')
    for n, dx, w in ((512, 20e-6, 2.0e-3), (256, 20e-6, 2.0e-3)):
        for fd in (5e-3, 8e-3):
            for tr in ('sziklas', 'collins'):
                r = spy_exit(n, dx, w, tr, fd)
                r.update(N=n, dx_um=dx * 1e6, w_mm=w * 1e3, fd_mm=fd * 1e3,
                         transport=tr)
                out['rows'].append(r)
                print(f"N={n} fd={fd * 1e3:.0f}mm {tr:8s} {r['outcome']:8s} "
                      f"route={r.get('readout_route_taken')} "
                      f"exit dx={r.get('exit_dx_um', nan):9.4f}um "
                      f"R={r.get('exit_R_mm', nan):10.4f}mm "
                      f"half={r.get('exit_halfwidth_um', nan):10.2f}um "
                      f"amp_r={r.get('exit_amp_radius_um', nan):9.2f}um "
                      f"box_r={r.get('exit_box_r_um', nan):9.2f}um "
                      f"th={r.get('exit_box_theta_mrad', nan):8.4f}mrad "
                      f"P={r.get('exit_env_power', nan):.6e}")
                print(f"      moment r2m(exit)={r.get('moment_r2m', nan) * 1e6:9.2f}um"
                      f" r2m(@fd)={r.get('moment_r2m_at_z', nan) * 1e6:9.2f}um"
                      f"  standoff={r.get('standoff_mm', nan):.4f}mm "
                      f"w_env={r.get('w_env_um', nan):.2f}um")
                for nm in ('sziklas', 'collins'):
                    print(f"      stop[{nm}] = {r.get('stop_' + nm)}")
                if r['outcome'] == 'raised':
                    print(f"      MSG {r['msg'][:200]}")
    tag = os.environ.get('VC3_TAG', 'x')
    p = os.path.join(os.environ['VC3_OUT'], f'r2_d5_rootcause_{tag}.json')
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, default=str)
    print('WROTE', p)


if __name__ == '__main__':
    main()
