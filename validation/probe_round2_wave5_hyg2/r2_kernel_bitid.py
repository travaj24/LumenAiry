"""Round 2 (VERIFY-WAVE5-HYGIENE2) -- archive-to-archive BIT IDENTITY of the
NumPy path across the V-D22 consolidation and the V-D3 public-leg threading.

WHAT IT PROVES.  ``lumenairy/propagators/carrier.py`` used to carry THREE
transcriptions of the exact non-paraxial dispersion (V-D22) and demoted an
eager JAX array to host NumPy at the public ``transport='collins'`` leg
(V-D3).  Both are changed.  Neither may move a single byte on the NumPy path,
so this probe drives every carrier / Collins / Sziklas fixture reachable from
the public and private surfaces on BOTH trees and compares the digests key by
key.

HOW IT IS GATED.  One harness (``validation/probe_wave5_hyg2/hlib.py``,
imported from the WORKTREE so both arms fold their results identically),
``hlib.anchor`` refusing to continue unless ``lumenairy.__file__`` resolves
under the tree named on the command line, and a digest that folds the returned
object's type and bytes -- or the exception type and message -- plus every
warning in EMISSION order.  NaN and signed zero are compared as bytes; no float
``==`` appears in the digest path.

    python r2_kernel_bitid.py <tree> <out.json>
"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'probe_wave5_hyg2'))
import hlib                                                  # noqa: E402

import numpy as np                                           # noqa: E402

WL = 633e-9
WL2 = 1.064e-6


def _gauss(n, dx, w, dtype=np.complex128, cx=0.0, cy=0.0):
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (w * w)).astype(dtype)


def _structured(n, dx, w):
    """A field with real structure in BOTH domains, so a summation-order
    change anywhere in the chain has something to move."""
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    env = np.exp(-(X ** 2 + Y ** 2) / (w * w))
    env = env * (1.0 + 0.3 * np.cos(2.0 * np.pi * X / (6.0 * dx)))
    return (env * np.exp(1j * 2.0 * np.pi * (0.13 * X + 0.07 * Y) / (8 * dx))
            ).astype(np.complex128)


def main(tree, out):
    lum = hlib.anchor(tree)
    import lumenairy.propagators.carrier as CA
    P = hlib.Probe()
    P.add('meta::version', lum.__version__)

    # ---- 1. the consolidated kernel, called DIRECTLY at every site --------
    # _exact_dispersion_phase did not exist on base, so the three sites are
    # driven through their PUBLIC-facing wrappers, which is what has to be
    # byte-identical.
    for n, dx in ((16, 8e-6), (32, 8e-6), (64, 4e-6), (33, 8e-6),
                  (48, 2.5e-6)):
        env = _structured(n, dx, 10.0 * dx)
        for z in (1e-3, 5e-3, -2e-3, 1e-6):
            for tilt in ((0.0, 0.0), (0.12, 0.03), (-0.4, 0.55),
                         (0.0, 0.31)):
                P.call(f'tf::exact::n{n}_dx{dx:.1e}_z{z:.0e}_t{tilt}',
                       CA._exact_envelope_tf_step, env, z, WL, dx, dx,
                       tilt=tilt)
        # the refusal, whose MESSAGE is part of the contract
        for bad in ((1.0, 0.0), (0.8, 0.8), (float('nan'), 0.0)):
            P.call(f'tf::exact::refuse::n{n}_{bad}',
                   CA._exact_envelope_tf_step, env, 1e-3, WL, dx, dx,
                   tilt=bad)
        # complex64 and a non-square pitch
        P.call(f'tf::exact::c64::n{n}', CA._exact_envelope_tf_step,
               env.astype(np.complex64), 2e-3, WL, dx, dx)
        P.call(f'tf::exact::anisopitch::n{n}', CA._exact_envelope_tf_step,
               env, 2e-3, WL, dx, 1.3 * dx)

    # ---- 2. _exact_tf_2d_xp on the NumPy namespace -----------------------
    # Never taken by a NumPy field in production (the module keeps
    # _exact_envelope_tf_step for its pyFFTW fast FFT), but it is the site the
    # consolidation changed most, and on xp = np it is exercisable directly.
    for n, dx in ((16, 8e-6), (32, 4e-6), (33, 8e-6)):
        env = _structured(n, dx, 10.0 * dx)
        for tilt in ((0.0, 0.0), (0.12, 0.03), (0.0, 0.31)):
            P.call(f'tf::xp::n{n}_t{tilt}', CA._exact_tf_2d_xp, env, 3e-3,
                   WL, dx, dx, tilt, np, False, np)
        P.call(f'tf::xp::refuse::n{n}', CA._exact_tf_2d_xp, env, 3e-3, WL,
               dx, dx, (0.9, 0.9), np, False, np)

    # ---- 3. the Collins exact-kernel correction --------------------------
    from lumenairy.propagators.fft_infra import _fft2
    for n, dx in ((16, 8e-6), (32, 8e-6), (64, 4e-6), (33, 8e-6)):
        env = _structured(n, dx, 10.0 * dx)
        S = _fft2(np.ascontiguousarray(env, dtype=np.complex128))
        for z_eff in (0.06, 3.03, 12.27, -1600.0):
            for tilt in ((0.0, 0.0), (0.12, 0.03), (0.0, 0.31)):
                P.call(f'corr::n{n}_ze{z_eff}_t{tilt}',
                       CA._collins_exact_kernel_correction, S, z_eff, WL,
                       dx, dx, tilt)
        P.call(f'corr::refuse::n{n}', CA._collins_exact_kernel_correction,
               S, 1.0, WL, dx, dx, (1.2, 0.0))
        P.call(f'corr::anisopitch::n{n}',
               CA._collins_exact_kernel_correction, S, 3.0, WL, dx,
               1.3 * dx, (0.0, 0.0))

    # ---- 4. the private transport, every spelling ------------------------
    N, DX = 64, 8e-6
    env = _gauss(N, DX, 60e-6)
    envs = _structured(N, DX, 60e-6)
    for tag, e in (('gauss', env), ('struct', envs),
                   ('c64', env.astype(np.complex64))):
        for gk in ('auto', 'fresnel', 'exact'):
            for R_ref in (-0.045, float('inf')):
                P.call(f'transport::{tag}_{gk}_R{R_ref}',
                       CA._collins_transport, e, -0.05, 5e-3, WL, DX, DX,
                       dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N,
                       R_ref=R_ref, gap_kernel=gk,
                       on_collins_sampling='ignore')
        P.call(f'transport::{tag}::astig',
               CA._collins_transport, e, (-0.05, -0.08), 5e-3, WL, DX, DX,
               dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N,
               R_ref=(-0.045, -0.07), gap_kernel='fresnel',
               on_collins_sampling='ignore')
        P.call(f'transport::{tag}::astig_exact_refused',
               CA._collins_transport, e, (-0.05, -0.08), 5e-3, WL, DX, DX,
               dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N,
               R_ref=(-0.045, -0.07), gap_kernel='exact',
               on_collins_sampling='ignore')
        P.call(f'transport::{tag}::tilted',
               CA._collins_transport, e, -0.05, 5e-3, WL, DX, DX,
               dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N, R_ref=-0.045,
               gap_kernel='exact', tilt=(0.12, 0.03),
               on_collins_sampling='ignore')
        P.call(f'transport::{tag}::offcentre',
               CA._collins_transport, e, -0.05, 5e-3, WL, DX, DX,
               dx_out=1.7 * DX, dy_out=1.3 * DX, N_out_x=48, N_out_y=40,
               R_ref=-0.045, centre_out=(3 * DX, -2 * DX),
               gap_kernel='auto', on_collins_sampling='ignore')
    # the stats dict and the guard's three dispositions on a violating leg
    for disp in ('ignore', 'warn', 'error'):
        st = {}
        P.call(f'transport::guard::{disp}', CA._collins_transport,
               _gauss(N, 40e-6, 500e-6), -0.30, 0.29, 1.55e-6, 40e-6, 40e-6,
               dx_out=40e-6 * 16, dy_out=40e-6 * 16, N_out_x=N, N_out_y=N,
               R_ref=float('inf'), gap_kernel='auto',
               on_collins_sampling=disp, stats_out=st, check_period=True)
        P.add(f'transport::guard::{disp}::stats', st)
    P.call('transport::B0', CA._collins_transport, env, -0.05, 0.05, WL,
           DX, DX, dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N,
           R_ref=float('inf'), gap_kernel='fresnel',
           on_collins_sampling='ignore')

    # ---- 5. the PUBLIC entry, both transports, on NumPy ------------------
    for transport in ('collins', 'sziklas'):
        for gk in ('auto', 'fresnel', 'exact'):
            for tag, e in (('gauss', env), ('struct', envs),
                           ('c64', env.astype(np.complex64))):
                kw = dict(wavelength=WL, dx=DX, transport=transport,
                          gap_kernel=gk)
                if transport == 'collins':
                    kw['on_collins_sampling'] = 'ignore'
                P.call(f'public::{transport}_{gk}_{tag}',
                       CA.propagate_carrier_referenced, e, -0.05, 5e-3, **kw)
        # a tilted carrier, a second wavelength and a coarse grid
        kw = dict(wavelength=WL2, dx=4e-6, transport=transport,
                  gap_kernel='auto', tilt=(0.09, -0.02))
        if transport == 'collins':
            kw['on_collins_sampling'] = 'ignore'
        P.call(f'public::{transport}::tilt_wl2',
               CA.propagate_carrier_referenced, _gauss(128, 4e-6, 30e-5),
               -0.04, 1e-3, **kw)
        # a collimated leg (R = inf), which has no pre-chirp at all
        kw = dict(wavelength=WL, dx=DX, transport=transport,
                  gap_kernel='auto')
        if transport == 'collins':
            kw['on_collins_sampling'] = 'ignore'
        P.call(f'public::{transport}::collimated',
               CA.propagate_carrier_referenced, env, float('inf'), 5e-3,
               **kw)
        # z == 0, the identity short-circuit
        P.call(f'public::{transport}::z0',
               CA.propagate_carrier_referenced, env, -0.05, 0.0, **kw)

    # ---- 6. the two focus readouts and the near-focus bridge -------------
    for tilt in ((0.0, 0.0), (0.02, -0.01)):
        P.call(f'readout::exact::t{tilt}',
               CA.carrier_referenced_exact_focus_readout,
               _gauss(256, 8e-6, 400e-6), -0.020031715, 0.020 - 1e-6, 1.0e-6,
               8e-6, dx_out=15.915e-6 / 8, N_out=128, tilt=tilt,
               on_replica='ignore')
    for d in (1e-6, 1e-4, 3e-4, 5e-3):
        for transport in ('collins', 'sziklas'):
            kw = dict(wavelength=1.0e-6, dx=8e-6, transport=transport,
                      gap_kernel='auto')
            if transport == 'collins':
                kw['on_collins_sampling'] = 'ignore'
            P.call(f'nearfocus::{transport}::d{d:.0e}',
                   CA.propagate_carrier_referenced,
                   _gauss(512, 8e-6, 400.03e-6), -0.020031715,
                   0.020 - d, **kw)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pass
    P.write(out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
