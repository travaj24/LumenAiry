"""V9 -- DURABILITY audit of the two new test files: every constant measured.

For each bar the shipped tests assert, this re-measures the quantity on the
SHIPPED FIXTURE (not on a fixture of mine) and reports the margin, so the
same script can be run on a second build and the two compared.

Also re-derives the quantities the test DOCSTRINGS record as measurements --
the phase arguments '3.6e+02 rad' / '1.3e+04 rad' and the control errors
'1.5e-05' / '4.9e-04' -- because a recorded measurement that does not
reproduce is a durability defect of its own.

Usage:  python v9_durability.py <out.json>
"""
from __future__ import annotations

import json
import os
import sys
import tracemalloc
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fix                                                    # noqa: E402

# ---- the shipped fixtures, copied verbatim from the two test files --------
_WL = 1.31e-6
_K = 2 * np.pi / _WL
_N, _DX, _W = 384, 16e-6, 1.4e-3
_SUB = 4
_RC = -0.06


def _surf(radius, gb, ga):
    return {'radius': radius, 'glass_before': gb, 'glass_after': ga,
            'conic': 0.0, 'radius_y': None, 'conic_y': None,
            'aspheric_coeffs': None, 'aspheric_coeffs_y': None}


def _presc():
    return {'name': 'band_rd_singlet', 'aperture_diameter': 9e-3,
            'surfaces': [_surf(0.030, 'air', 'N-BK7'),
                         _surf(-0.030, 'N-BK7', 'air')],
            'thicknesses': [3e-3]}


def _field(n=_N, dx=_DX, w=_W):
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    return np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)


def _base_kw(**over):
    kw = dict(prescription=_presc(), wavelength=_WL, dx=_DX,
              ray_subsample=_SUB, n_workers=1, on_undersample='silent',
              on_noncollimated='off', on_aperture_beam='silent',
              parallel_amp=False, carrier=_RC)
    kw.update(over)
    return kw


def _helpers(C, N, dx, R):
    shape = (N, N)
    L, M = 0.0515, -0.02
    x0, y0 = 1.3e-3, -0.4e-3
    return {
        '_radial_carrier_phase': lambda dt: C._radial_carrier_phase(
            shape, dx, dx, _WL, R, +1, dtype=dt),
        '_tilt_ramp': lambda dt: C._tilt_ramp(
            shape, dx, _WL, L, M, x0, y0, -1, dtype=dt),
        '_tilt_exactness_phase': lambda dt: C._tilt_exactness_phase(
            shape, dx, dx, _WL, R, L, M, +1, centre=(x0, y0), dtype=dt),
        '_sphere_parab_conversion': lambda dt: C._sphere_parab_conversion(
            shape, dx, _WL, R, +1, centre=(x0, y0), dtype=dt),
    }


def _smooth_env(n, dtype=np.complex128):
    """test_niche_perf_round2's fixture helper, reproduced."""
    x = (np.arange(n) - n / 2) / n
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    return np.exp(-r2 / 0.05).astype(dtype)


def _readout_field(N, dx, R, dtype):
    x = (np.arange(N) - N // 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    E = (np.exp(-r2 / (0.25e-3 ** 2))
         * np.exp(1j * _K * (-(np.sqrt(r2 + R * R) - abs(R)))))
    return E.astype(dtype)


def _singlet(R1, R2, d, glass, ap, name='s'):
    return {'name': name, 'aperture_diameter': ap, 'thicknesses': [d],
            'surfaces': [
                {'radius': R1, 'glass_before': 'air', 'glass_after': glass,
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
                {'radius': R2, 'glass_before': glass, 'glass_after': 'air',
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]}


def main():
    la = _fix.banner()
    from lumenairy.propagators import carrier as C
    from lumenairy.elements import _lens_imap as IM
    prev = la.get_fft_auto_promote()
    la.set_fft_auto_promote(False)
    out = {'version': la.__version__, 'numpy': np.__version__,
           'python': sys.version.split()[0], 'bars': {}}

    def rec(name, measured, bar, sense, note=''):
        margin = (bar / measured if sense == 'upper' and measured > 0
                  else (measured / bar if sense == 'lower' and bar > 0
                        else None))
        out['bars'][name] = {'measured': measured, 'bar': bar,
                             'sense': sense, 'margin': margin, 'note': note}
        print('%-52s measured=%-12.4g bar=%-10.4g margin=%s  %s'
              % (name, measured, bar,
                 ('%.1fx' % margin) if margin else '--', note), flush=True)

    # ---- 1. _C64_PHASOR_TOL = 2.5e-7 on the two shipped rungs ------------
    for R, N, dx in ((45.9e-3, 512, 1.806e-6), (5.0e-3, 1024, 1.806e-6)):
        worst = 0.0
        x = (np.arange(N) - N / 2) * dx
        r2 = x[None, :] ** 2 + x[:, None] ** 2
        argmax = float((_K * r2 / (2.0 * R)).max())
        for nm, fn in _helpers(C, N, dx, R).items():
            ref = fn(np.complex128)
            if ref is None:
                continue
            got = fn(np.complex64)
            worst = max(worst, float(np.abs(
                ref - got.astype(np.complex128)).max()))
        rec('C64_PHASOR_TOL@R=%g,N=%d' % (R, N), worst, 2.5e-7, 'upper',
            'true argmax %.4g rad (docstring says 3.6e+02 / 1.3e+04)'
            % argmax)
        out['bars']['C64_PHASOR_TOL@R=%g,N=%d' % (R, N)]['argmax_rad'] = argmax

    # ---- 2. the float32-ARGUMENT control ratios (bars 10x / 100x) --------
    for R, N, dx, ratio_bar in ((45.9e-3, 512, 1.806e-6, 10.0),
                                (5.0e-3, 1024, 1.806e-6, 100.0)):
        x = (np.arange(N) - N / 2) * dx
        r2 = x[None, :] ** 2 + x[:, None] ** 2
        ref = np.exp(1j * _K * r2 / (2.0 * R))
        good = C._radial_carrier_phase((N, N), dx, dx, _WL, R, +1,
                                       dtype=np.complex64)
        naive = np.exp(1j * (_K * r2 / (2.0 * R)).astype(np.float32))
        e_good = float(np.abs(ref - good.astype(np.complex128)).max())
        e_naive = float(np.abs(ref - naive.astype(np.complex128)).max())
        r_meas = e_naive / e_good
        rec('f32arg_control_ratio@R=%g' % R, r_meas, ratio_bar, 'lower',
            'e_good %.3e  e_naive %.3e (docstring records 1.5e-05 / 4.9e-04 '
            'and ratios 360x / 11600x)' % (e_good, e_naive))
        out['bars']['f32arg_control_ratio@R=%g' % R].update(
            {'e_good': e_good, 'e_naive': e_naive})

    # ---- 3. _fourier_upsample_crop rel L2 (bar 1e-5) --------------------
    N, dx = 512, 1.806e-6
    rng = np.random.default_rng(1)
    x = (np.arange(N) - N / 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    env = (np.exp(-r2 / (0.15 * N * dx) ** 2)
           * (1 + 0.05 * rng.standard_normal((N, N)))).astype(np.complex128)
    worst = 0.0
    for nc, nf in ((N // 2, N), (N, N // 2)):
        ref = C._fourier_upsample_crop(env, nc, nf)
        got = C._fourier_upsample_crop(env.astype(np.complex64), nc, nf)
        worst = max(worst, float(np.linalg.norm(ref - got)
                                 / np.linalg.norm(ref)))
    rec('upsample_crop_rel_L2', worst, 1e-5, 'upper',
        'docstring records 2.0e-07 at N=2048; the test runs N=512')

    # ---- 4. the exact focus readout (bar 1e-4) --------------------------
    N, dx, R = 256, 2.0e-6, -2.0e-3
    kw = dict(dx_out=0.1e-6, N_out=64, window_factor=4.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = np.asarray(C.carrier_referenced_exact_focus_readout(
            _readout_field(N, dx, R, np.complex128), R, -R, _WL, dx, **kw))
        b = np.asarray(C.carrier_referenced_exact_focus_readout(
            _readout_field(N, dx, R, np.complex64), R, -R, _WL, dx, **kw))
    rel = float(np.linalg.norm(a - b) / np.linalg.norm(a))
    rec('exact_focus_readout_rel_L2', rel, 1e-4, 'upper',
        'a.dtype=%s b.dtype=%s; the docstring records NO number' % (a.dtype,
                                                                   b.dtype))

    # ---- 5. the one-group chain (bar 1e-4) ------------------------------
    Nc, dxc = 512, 30e-6
    w, R_in = 4.5e-3, 60e-3
    presc = _singlet(60.0e-3, -60.0e-3, 3.0e-3, 'N-BK7', 14.0e-3, 'sph')
    xg = (np.arange(Nc) - Nc // 2) * dxc
    r2g = xg[None, :] ** 2 + xg[:, None] ** 2
    envg = np.exp(-r2g / w ** 2).astype(np.complex128)

    def _runchain(e):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.asarray(la.propagate_traced_carrier_chain(
                e, [{'prescription': presc, 'gap_before': 0.0}], _WL, dxc,
                r_in=R_in, ray_subsample=8, n_workers=1,
                traced_kwargs=dict(on_undersample='silent',
                                   on_noncollimated='silent')).field)
    ca = _runchain(envg)
    cb = _runchain(envg.astype(np.complex64))
    relc = float(np.linalg.norm(ca - cb) / np.linalg.norm(ca))
    rec('one_group_chain_rel_L2', relc, 1e-4, 'upper',
        'a.dtype=%s b.dtype=%s' % (ca.dtype, cb.dtype))

    # ---- 6. the memory bar: p_whole - p_band >= 6 grids ------------------
    Nm, dxm, subm = 512, 12e-6, 16

    def _peak(rows):
        E = _field(Nm, dxm)
        kwm = _base_kw(dx=dxm, ray_subsample=subm,
                       amplitude_model='ray_density',
                       preserve_input_phase=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            IM.inverse_map_cache_clear()
            la.apply_real_lens_traced(E, sag_chunk_rows=rows, **kwm)
            IM.inverse_map_cache_clear()
            r = {}
            tracemalloc.start()
            tracemalloc.reset_peak()
            la.apply_real_lens_traced(E, sag_chunk_rows=rows, _imap_out=r,
                                      **kwm)
            _, pk = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            IM.inverse_map_cache_clear()
        return pk, bool(r.get('engaged', False))
    grid = 8 * Nm * Nm
    pw, ew = _peak(0)
    pb, eb = _peak(32)
    rec('band_memory_saving_grids', (pw - pb) / grid, 6.0, 'lower',
        'whole %.2f grids (eng=%s), band %.2f grids (eng=%s); '
        'the test docstring records 22.6 and 10.2'
        % (pw / grid, ew, pb / grid, eb))
    out['bars']['band_memory_saving_grids'].update(
        {'whole_grids': pw / grid, 'band_grids': pb / grid,
         'whole_engaged': ew, 'band_engaged': eb})

    # ---- 7. the ROUTE-CHANGE size on the shipped fixture -----------------
    kwr = _base_kw(amplitude_model='ray_density',
                   preserve_input_phase='remap', remap_sampling='full')
    m, rm, _ = _fix.run(la, _field(), kwr, 64)
    i, ri, _ = _fix.run(la, _field(), dict(kwr, inverse_map=False), 64)
    relr = float(np.linalg.norm(m - i) / np.linalg.norm(m))
    out['bars']['route_change_rel_on_shipped_fixture'] = {
        'measured': relr, 'bar': None, 'sense': 'informational',
        'note': 'model engaged=%s incumbent engaged=%s; CHANGELOG cites '
                '2.19e-02 on the S10 fixture' % (rm.get('engaged'),
                                                 ri.get('engaged'))}
    print('%-52s measured=%-12.4g (CHANGELOG cites 2.19e-02 on S10)'
          % ('route_change_rel_on_shipped_fixture', relr), flush=True)

    la.set_fft_auto_promote(prev)
    with open(sys.argv[1], 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', sys.argv[1])


if __name__ == '__main__':
    main()
