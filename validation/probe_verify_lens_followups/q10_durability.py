"""Q10 (D4/D5) -- every recorded constant of the three touched test files,
re-measured on both builds.

Task 5 and task 6.  The claims under test are about what
``tests/unit/test_mixed_precision_carrier_helpers.py`` RECORDS, so the
fixtures here are that file's fixtures (same N, dx, R, same beams) -- the
measurement is independent, the fixture cannot be.  Everything is reported
with the bar it is asserted against and the margin on BOTH sides.

Measured:

* ``_C64_PHASOR_TOL`` (2.5e-7) against the four helpers at R=45.9 mm / N=512
  and R=5 mm / N=1024, and against the float32-ARGUMENT control at the same
  fixtures -- so the "3.95x below the control" statement can be checked;
* the two fixtures' MAX PHASE ARGUMENT (docstring: 22.34 / 820.19 rad);
* the control errors and the control/shipped ratios (23.5x / 727x) against
  the 10x / 100x bars;
* the upsample-crop rel L2 at the test's own N=512 fixture (bar 1e-5);
* the exact focus readout rel L2 (bar, tightened, 1e-6);
* the one-group chain rel L2 (bar, tightened, 1e-6);
* the banded memory saving in float64 grids (bar >= 6) from
  ``test_banded_ray_density_and_inverse_map.py``.

Usage: python q10_durability.py <out.json> --tree <arm tree>
"""
from __future__ import annotations

import os
import sys
import tracemalloc
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vf  # noqa: E402

_WL = 1.31e-6                     # the test file's wavelength
_K = 2 * np.pi / _WL


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
    args = _vf.argp(__doc__).parse_args()
    la = _vf.banner(args.tree)
    from lumenairy.propagators import carrier as C
    prev = la.get_fft_auto_promote()
    la.set_fft_auto_promote(False)
    out = {}
    try:
        # ---- 1. the phasor tolerance and its control -----------------------
        for R, N, dx, ratio_bar in ((45.9e-3, 512, 1.806e-6, 10.0),
                                    (5.0e-3, 1024, 1.806e-6, 100.0)):
            tag = f'R={R * 1e3:g}mm_N={N}'
            row = {'ratio_bar': ratio_bar, 'tol_bar': 2.5e-7}
            errs = {}
            for name, fn in _helpers(C, N, dx, R).items():
                ref = fn(np.complex128)
                got = fn(np.complex64)
                errs[name] = float(np.abs(
                    ref - got.astype(np.complex128)).max())
            row['phasor_err'] = errs
            row['phasor_err_max'] = max(errs.values())
            row['tol_margin'] = 2.5e-7 / row['phasor_err_max']
            x = (np.arange(N) - N / 2) * dx
            r2 = x[None, :] ** 2 + x[:, None] ** 2
            arg = _K * r2 / (2.0 * R)
            row['max_argument_rad'] = float(np.abs(arg).max())
            ref = np.exp(1j * arg)
            good = C._radial_carrier_phase((N, N), dx, dx, _WL, R, +1,
                                           dtype=np.complex64)
            naive = np.exp(1j * arg.astype(np.float32))
            row['e_good'] = float(np.abs(
                ref - good.astype(np.complex128)).max())
            row['e_naive_control'] = float(np.abs(
                ref - naive.astype(np.complex128)).max())
            row['control_ratio'] = row['e_naive_control'] / row['e_good']
            row['ratio_margin'] = row['control_ratio'] / ratio_bar
            row['control_over_tol'] = row['e_naive_control'] / 2.5e-7
            out[f'phasor_{tag}'] = row
            print(f"  {tag}: max_arg={row['max_argument_rad']:.4f} rad  "
                  f"c64_err={row['e_good']:.5e}  control="
                  f"{row['e_naive_control']:.5e}  ratio="
                  f"{row['control_ratio']:.3f}x (bar {ratio_bar}x, margin "
                  f"{row['ratio_margin']:.3f}x)  control/tol="
                  f"{row['control_over_tol']:.3f}x", flush=True)

        # ---- 2. the crop's rel L2 at the test's own fixture -----------------
        N, dx = 512, 1.806e-6
        rng = np.random.default_rng(1)
        x = (np.arange(N) - N / 2) * dx
        r2 = x[None, :] ** 2 + x[:, None] ** 2
        env = (np.exp(-r2 / (0.15 * N * dx) ** 2)
               * (1 + 0.05 * rng.standard_normal((N, N)))
               ).astype(np.complex128)
        crop = {}
        for nc, nf in ((N // 2, N), (N, N // 2)):
            ref = C._fourier_upsample_crop(env, nc, nf)
            got = C._fourier_upsample_crop(env.astype(np.complex64), nc, nf)
            crop[f'{nc}->{nf}'] = float(np.linalg.norm(ref - got)
                                        / np.linalg.norm(ref))
        out['crop_rel_l2'] = {'values': crop, 'bar': 1e-5,
                              'margin': 1e-5 / max(crop.values())}
        print(f"  crop rel L2 {crop}  bar 1e-5  margin "
              f"{out['crop_rel_l2']['margin']:.2f}x", flush=True)

        # ---- 3. the exact focus readout ------------------------------------
        Nr, dxr, Rr = 256, 2.0e-6, -2.0e-3
        kwr = dict(dx_out=0.1e-6, N_out=64, window_factor=4.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = np.asarray(C.carrier_referenced_exact_focus_readout(
                _readout_field(Nr, dxr, Rr, np.complex128), Rr, -Rr, _WL,
                dxr, **kwr))
            b = np.asarray(C.carrier_referenced_exact_focus_readout(
                _readout_field(Nr, dxr, Rr, np.complex64), Rr, -Rr, _WL,
                dxr, **kwr))
        rel = float(np.linalg.norm(a - b) / np.linalg.norm(a))
        out['readout_rel_l2'] = {'value': rel, 'bar': 1e-6,
                                 'margin': 1e-6 / rel,
                                 'dtype_a': str(a.dtype),
                                 'dtype_b': str(b.dtype)}
        print(f"  readout rel L2 {rel:.6e}  bar 1e-6  margin "
              f"{1e-6 / rel:.3f}x", flush=True)

        # ---- 4. the one-group chain ----------------------------------------
        Nc, dxc = 512, 30e-6
        w, R_in = 4.5e-3, 60e-3
        presc = _singlet(60.0e-3, -60.0e-3, 3.0e-3, 'N-BK7', 14.0e-3, 'sph')
        xc = (np.arange(Nc) - Nc // 2) * dxc
        r2c = xc[None, :] ** 2 + xc[:, None] ** 2
        envc = np.exp(-r2c / w ** 2).astype(np.complex128)

        def _run(e):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = la.propagate_traced_carrier_chain(
                    e, [{'prescription': presc, 'gap_before': 0.0}], _WL,
                    dxc, r_in=R_in, ray_subsample=8, n_workers=1,
                    traced_kwargs=dict(on_undersample='silent',
                                       on_noncollimated='silent'))
            return np.asarray(res.field)

        ca = _run(envc)
        cb = _run(envc.astype(np.complex64))
        relc = float(np.linalg.norm(ca - cb) / np.linalg.norm(ca))
        out['chain_rel_l2'] = {'value': relc, 'bar': 1e-6,
                               'margin': 1e-6 / relc,
                               'dtype_a': str(ca.dtype),
                               'dtype_b': str(cb.dtype)}
        print(f"  one-group chain rel L2 {relc:.6e}  bar 1e-6  margin "
              f"{1e-6 / relc:.3f}x", flush=True)

        # ---- 5. the banded memory saving in float64 grids ------------------
        out['band_memory'] = band_memory(la)
        print(f"  band memory saving {out['band_memory']['saving_grids']:.6f}"
              f" grids (bar 6, margin "
              f"{out['band_memory']['saving_grids'] / 6.0:.3f}x)", flush=True)
    finally:
        la.set_fft_auto_promote(prev)
    _vf.dump(args, {'cases': out, 'free_gb': _vf.free_gb()})


def band_memory(la):
    """The test file's own N=512 / sub=16 ray-density + evaluator pin."""
    from lumenairy.elements import _lens_imap as IM
    N, dx, sub = 512, 12e-6, 16
    E = _vf.gauss(N, dx, 0.9e-3)
    kw = dict(prescription=_vf.presc_meniscus(ap=4.4e-3), wavelength=_vf.WL,
              dx=dx, ray_subsample=sub, n_workers=1, on_undersample='silent',
              on_noncollimated='off', on_aperture_beam='silent',
              parallel_amp=False, amplitude_model='ray_density')

    def _peak(rows):
        IM.inverse_map_cache_clear()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            la.apply_real_lens_traced(E, sag_chunk_rows=rows, **kw)   # warm
        IM.inverse_map_cache_clear()
        tracemalloc.start()
        tracemalloc.reset_peak()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            la.apply_real_lens_traced(E, sag_chunk_rows=rows, **kw)
        _, pk = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        IM.inverse_map_cache_clear()
        return pk / (8.0 * N * N)

    whole = _peak(0)
    band = _peak(32)
    return {'whole_grids': whole, 'band_grids': band,
            'saving_grids': whole - band, 'bar': 6.0}


if __name__ == '__main__':
    main()
