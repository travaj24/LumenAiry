"""VERIFY-WP-C3 CLAIM 8 -- the CuPy arm, re-measured independently.

    python v_cupy.py <tree> <out.json>

My own grids, my own bar.  Nothing is imported from
``validation/probe_c3_collins_default``.
"""
from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, _TREE)

import vlib  # noqa: E402
import numpy as np  # noqa: E402

import lumenairy.propagators.carrier as CA  # noqa: E402

vlib.anchor(_TREE)

WL = 1.064e-6          # NOT the probe's 633 nm
EPS = float(np.finfo(np.float64).eps)


def _premise():
    out = {'importable': False, 'version': None, 'n_devices': None,
           'device_name': None, 'elementwise_ok': False,
           'elementwise_error': None, 'fft_ok': False, 'fft_error': None,
           'cupy_fft_import_error': None, 'cuda_runtime_version': None}
    try:
        import cupy as cp
    except Exception as exc:                          # noqa: BLE001
        out['import_error'] = '%s: %s' % (type(exc).__name__, exc)
        return out, None
    out['importable'] = True
    out['version'] = cp.__version__
    try:
        out['n_devices'] = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:                          # noqa: BLE001
        out['n_devices'] = '%s: %s' % (type(exc).__name__, exc)
    try:
        out['cuda_runtime_version'] = int(cp.cuda.runtime.runtimeGetVersion())
    except Exception as exc:                          # noqa: BLE001
        out['cuda_runtime_version'] = str(exc)
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        nm = props['name']
        out['device_name'] = nm.decode() if isinstance(nm, bytes) else str(nm)
    except Exception as exc:                          # noqa: BLE001
        out['device_name'] = '%s: %s' % (type(exc).__name__, exc)
    # elementwise: a REAL kernel, a reduction and a transcendental, not ones()
    try:
        a = cp.asarray(np.linspace(-3.0, 3.0, 4096).reshape(64, 64))
        b = cp.exp(1j * a) * 2.0 + 1.0
        s = complex(cp.asnumpy(b.sum()))
        assert np.isfinite(s.real)
        out['elementwise_ok'] = True
    except Exception as exc:                          # noqa: BLE001
        out['elementwise_error'] = '%s: %s' % (type(exc).__name__, exc)
        return out, cp
    try:
        cp.asnumpy(cp.fft.fft2(cp.ones((8, 8), dtype=cp.complex128)))
        out['fft_ok'] = True
    except Exception as exc:                          # noqa: BLE001
        out['fft_error'] = ('%s: %s' % (type(exc).__name__, exc))[:300]
    try:
        import cupy.fft  # noqa: F401
        out['cupy_fft_import_error'] = None
    except Exception as exc:                          # noqa: BLE001
        out['cupy_fft_import_error'] = (
            '%s: %s' % (type(exc).__name__, exc))[:300]
    return out, cp


def _rel(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    n = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / n) if n else float('nan')


def _row(name, d, h, **extra):
    d = np.asarray(d)
    h = np.asarray(h)
    m = float(np.abs(d - h).max())
    r = dict(helper=name, rel_l2=_rel(d, h), max_abs=m, ulps_of_one=m / EPS,
             bitwise_equal=bool(np.array_equal(d.view(np.float64)
                                               if d.dtype.kind == 'c' else d,
                                               h.view(np.float64)
                                               if h.dtype.kind == 'c' else h)))
    r.update(extra)
    return r


def _device_helpers(cp):
    rows = []
    # MY grids: different n, different pitch, different R, and one POSITIVE R.
    for n, dx, R in ((96, 5.5e-6, -0.031), (384, 2.75e-6, 0.017),
                     (1024, 1.9e-6, -0.0075), (2048, 0.85e-6, -0.11)):
        d = cp.asnumpy(CA._collins_axis_chirp(n, dx, WL, R, bld=cp))
        h = CA._collins_axis_chirp(n, dx, WL, R, bld=np)
        rows.append(_row('_collins_axis_chirp', d, h, n=n, dx=dx, R=R,
                         max_phase_rad=float(2 * np.pi / WL
                                             * ((n / 2 * dx) ** 2)
                                             / (2 * abs(R)))))
    # offset + complex64 narrowing arms, which the package's probe never drove
    d = cp.asnumpy(CA._collins_axis_chirp(512, 2.0e-6, WL, -0.02,
                                          offset=7.3e-6, bld=cp))
    h = CA._collins_axis_chirp(512, 2.0e-6, WL, -0.02, offset=7.3e-6, bld=np)
    rows.append(_row('_collins_axis_chirp[offset]', d, h, n=512))
    d = cp.asnumpy(CA._collins_axis_chirp(512, 2.0e-6, WL, -0.02,
                                          dtype=np.complex64, bld=cp))
    h = CA._collins_axis_chirp(512, 2.0e-6, WL, -0.02, dtype=np.complex64,
                               bld=np)
    rows.append(_row('_collins_axis_chirp[complex64]', d, h, n=512))

    # _tf_phase_to_H: my own argument range, both target dtypes.
    for nn, lo, hi in ((256, -1.0e5, 1.0e5), (768, -3.0e3, 3.0e3)):
        arg = np.linspace(lo, hi, nn * nn).reshape(nn, nn)
        Hd = cp.asnumpy(CA._tf_phase_to_H(cp.asarray(arg), np.complex128,
                                          cp, False, cp))
        Hh = CA._tf_phase_to_H(arg, np.complex128, np, False, np)
        rows.append(_row('_tf_phase_to_H[c128]', Hd, Hh, n=nn,
                         arg_max=float(np.abs(arg).max())))
    arg = np.linspace(-3.0e4, 3.0e4, 256 * 256).reshape(256, 256)
    Hd = cp.asnumpy(CA._tf_phase_to_H(cp.asarray(arg), np.complex64,
                                      cp, False, cp))
    Hh = CA._tf_phase_to_H(arg, np.complex64, np, False, np)
    rows.append(_row('_tf_phase_to_H[c64]', Hd.astype(np.complex128),
                     Hh.astype(np.complex128), n=256,
                     note='NumPy arm ignores target dtype (xp is np branch) '
                          'so this row compares a float32 device build to a '
                          'float64 host build ON PURPOSE'))

    # _exact_dispersion_phase: my own axes, and THREE tilts incl. a big one.
    k = 2.0 * np.pi / WL
    for nq, dq in ((200, 3.1e-6), (512, 1.4e-6)):
        qx = 2.0 * np.pi * np.fft.fftfreq(nq, d=dq)
        qy = 2.0 * np.pi * np.fft.fftfreq(nq + 8, d=dq * 1.3)
        for tilt in ((0.0, 0.0), (0.02, -0.01), (0.31, 0.22)):
            pd = cp.asnumpy(CA._exact_dispersion_phase(
                cp.asarray(qx), cp.asarray(qy), k, tilt, cp, 'v'))
            ph = CA._exact_dispersion_phase(qx, qy, k, tilt, np, 'v')
            rows.append(_row('_exact_dispersion_phase', pd, ph, n=nq,
                             tilt=list(tilt)))
    return rows


def _host_side_readings(cp):
    n, dx = 128, 4.0e-6
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    E = (np.exp(-(X ** 2 + Y ** 2) / (90e-6 ** 2))
         * np.exp(0.7j * X / 1e-4)).astype(np.complex128)
    Ed = cp.asarray(E)
    out = {}
    out['space_support_device'] = list(CA._collins_space_support(Ed, dx, dx,
                                                                 1e-6))
    out['space_support_host'] = list(CA._collins_space_support(E, dx, dx,
                                                               1e-6))
    xp, is_jax, bld = CA._backend_of(Ed)
    out['backend_of'] = {'xp': getattr(xp, '__name__', str(xp)),
                         'is_jax': bool(is_jax),
                         'bld': getattr(bld, '__name__', str(bld)),
                         'bld_is_xp': bld is xp}
    try:
        CA._collins_input_box(Ed, dx, dx, WL, 1e-6)
        out['input_box'] = 'ran'
    except Exception as exc:                          # noqa: BLE001
        out['input_box'] = '%s: %s' % (type(exc).__name__, str(exc)[:220])
    return out


def _call(label, fn):
    rec = {'label': label}
    try:
        r = fn()
        rec['outcome'] = 'ran'
        rec['result_namespace'] = type(r).__module__.split('.')[0]
    except Exception as exc:                          # noqa: BLE001
        rec['outcome'] = 'raised'
        rec['error_type'] = type(exc).__name__
        rec['error'] = str(exc)[:400]
        tb = traceback.extract_tb(exc.__traceback__)
        rec['raised_in'] = ['%s:%d %s' % (os.path.basename(f.filename),
                                          f.lineno, f.name) for f in tb[-6:]]
        msg = '%s: %s' % (type(exc).__name__, exc)
        rec['names_cufft'] = 'cufft' in msg.lower()
        rec['is_implicit_conversion_TypeError'] = (
            type(exc).__name__ == 'TypeError'
            and 'implicit conversion' in msg.lower())
        rec['reached_device_transform'] = (
            type(exc).__name__ == 'ImportError' and 'cufft' in msg.lower())
    return rec


def _public_legs(cp):
    n, dx = 64, 8e-6
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    E = np.exp(-(X ** 2 + Y ** 2) / (60e-6 ** 2)).astype(np.complex128)
    Ed = cp.asarray(E)
    kw = dict(transport='collins', gap_kernel='fresnel',
              on_collins_sampling='ignore')
    out = []
    out.append(_call(
        'propagate_carrier_referenced(device, collins, fresnel)',
        lambda: CA.propagate_carrier_referenced(
            Ed, -0.05, 5e-3, WL, dx, **kw).env))
    out.append(_call(
        'propagate_carrier_referenced(device, collins, exact)',
        lambda: CA.propagate_carrier_referenced(
            Ed, -0.05, 5e-3, WL, dx, transport='collins', gap_kernel='exact',
            on_collins_sampling='ignore').env))
    out.append(_call(
        'propagate_carrier_referenced(device, sziklas)',
        lambda: CA.propagate_carrier_referenced(
            Ed, -0.05, 5e-3, WL, dx, transport='sziklas').env))
    out.append(_call(
        '_collins_focus_readout(device)',
        lambda: CA._collins_focus_readout(
            Ed, -0.05, 5e-3, WL, dx, dx, dx_out=2e-6, N_out=64,
            on_collins_sampling='ignore', on_replica='ignore')))
    out.append(_call(
        '_collins_transport(device)',
        lambda: CA._collins_transport(
            Ed, -0.05, 5e-3, WL, dx, dx, dx_out=dx, dy_out=dx,
            N_out_x=n, N_out_y=n, R_ref=np.inf,
            on_collins_sampling='ignore')))
    out.append(_call(
        '_collins_carrier_leg(device)',
        lambda: CA._collins_carrier_leg(
            Ed, -0.05, 5e-3, WL, dx, dx, on_collins_sampling='ignore')))
    out.append(_call(
        '_collins_input_box(device)',
        lambda: CA._collins_input_box(Ed, dx, dx, WL, 1e-6)))
    out.append(_call(
        '_collins_readout_k1(device)',
        lambda: CA._collins_readout_k1(Ed, -0.05, 5e-3, WL, dx, dx)))
    # THE CHAIN with a device array -- a one-group chain is enough to reach
    # the same transport.
    out.append(_call(
        'propagate_traced_carrier_chain(device, collins)',
        lambda: CA.propagate_traced_carrier_chain(
            Ed, [{'gap_before': 5e-3}], WL, dx, r_in=-0.05,
            transport='collins',
            on_collins_sampling='ignore').field))
    return out


def main():
    out_path = sys.argv[2]
    prem, cp = _premise()
    rec = {'build': vlib.build_tag(), 'env': vlib.env_tag(), 'tree': _TREE,
           'carrier_file': CA.__file__, 'premise': prem,
           'eps': EPS, 'bar_ulps_of_one': 8.0}
    if cp is not None and prem['elementwise_ok']:
        rec['device_helpers'] = _device_helpers(cp)
        rec['host_side'] = _host_side_readings(cp)
        rec['public_legs'] = _public_legs(cp)
        worst = max(r['ulps_of_one'] for r in rec['device_helpers']
                    if 'c64' not in r['helper'])
        rec['worst_ulps_of_one_excl_c64'] = worst
    else:
        rec['device_helpers'] = None
        rec['host_side'] = None
        rec['public_legs'] = None
        # the no-CuPy decision, asserted rather than skipped
        import inspect
        rec['no_cupy_decision'] = {
            'backend_of_bld_is_np': CA._backend_of(
                np.zeros((4, 4), dtype=np.complex128))[2] is np,
            'axis_chirp_bld_default_is_np': inspect.signature(
                CA._collins_axis_chirp).parameters['bld'].default is np}
    vlib.write_json(rec, out_path)
    print('[v_cupy] build=%s cupy=%s ver=%s devices=%s elementwise=%s fft=%s'
          % (rec['build'], prem['importable'], prem['version'],
             prem['n_devices'], prem['elementwise_ok'], prem['fft_ok']))
    for r in (rec['device_helpers'] or []):
        print('   %-34s rel=%.4e  maxabs=%.4e  %.3f ULP  bitwise=%s'
              % (r['helper'], r['rel_l2'], r['max_abs'], r['ulps_of_one'],
                 r['bitwise_equal']))
    for r in (rec['public_legs'] or []):
        print('   LEG %-52s %s %s %s' % (
            r['label'], r['outcome'], r.get('error_type', ''),
            ('cufft' if r.get('names_cufft') else
             ('IMPLICIT-CONVERSION' if
              r.get('is_implicit_conversion_TypeError') else
              r.get('error', '')[:70]))))


if __name__ == '__main__':
    main()
