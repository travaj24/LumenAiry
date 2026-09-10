"""V6 -- the complex64 CHAIN: dtype hand-offs, error growth, energy honesty.

(1) DTYPE INSTRUMENTATION.  Every one of the six named helpers plus
    ``_fourier_upsample_crop`` and ``carrier_referenced_exact_focus_readout``
    is wrapped and its input / output dtypes logged during a REAL two-group
    ``propagate_traced_carrier_chain`` run with a complex64 envelope.  Any
    complex128 return on the c64 arm is an UPCAST and is reported.

(2) SIX-LEG SYNTHETIC CHAIN, built here (not the prototypes'): one leg =
    de-chirp by a carrier phase, a band-limited resample round trip,
    re-chirp, tilt ramp.  Relative L2 of the c64 arm against the c128 arm,
    per leg.

(3) THE ENERGY HONESTY BAR.  The campaign bar is 4e-05 RELATIVE on chain
    energy readouts (ADJUDICATION_NFC_8192_2026_08_10: throughput, capture,
    per-frame power, ``power_exit``).  Power is quadratic in the field, so a
    field relative-L2 of ``e`` moves a power ratio by at most ~``2e``; the
    field-side bar is therefore 2e-05.  Both are measured on the REAL
    two-group chain, not only on the synthetic one.

(4) ``_fourier_upsample_crop`` DOUBLE-ROUNDING: the c64 path against a
    complex128 transform of the SAME c64 samples narrowed ONCE at the end.

Usage:  python v6_c64_chain.py <out.json>
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fix  # noqa: E402

WL = _fix.WL
K = 2 * np.pi / WL


def _singlet(R1, R2, d, glass, ap, name):
    def s(r, gb, ga):
        return {'radius': r, 'glass_before': gb, 'glass_after': ga,
                'conic': 0.0, 'radius_y': None, 'conic_y': None,
                'aspheric_coeffs': None, 'aspheric_coeffs_y': None}
    return {'name': name, 'aperture_diameter': ap, 'thicknesses': [d],
            'surfaces': [s(R1, 'air', glass), s(R2, glass, 'air')]}


# ---------------------------------------------------------------- (1) dtypes
_NAMES = ['_radial_carrier_phase', '_tilt_ramp', '_tilt_exactness_phase',
          '_sphere_parab_conversion', '_fourier_upsample_crop',
          'carrier_referenced_exact_focus_readout', '_phasor_rows']


def instrument(C, log):
    """Wrap the six helpers + the crop + the readout, logging dtypes."""
    saved = {}
    for nm in _NAMES:
        fn = getattr(C, nm, None)
        if fn is None:
            continue
        saved[nm] = fn

        def mk(nm, fn):
            def wrapper(*a, **kw):
                out = fn(*a, **kw)
                ain = [str(np.asarray(z).dtype) for z in a
                       if isinstance(z, np.ndarray) and np.iscomplexobj(z)]
                dt = kw.get('dtype')
                o = getattr(out, 'dtype', None)
                if o is None and hasattr(out, 'field'):
                    o = getattr(np.asarray(out.field), 'dtype', None)
                log.append({'fn': nm, 'in': ain,
                            'dtype_kw': (None if dt is None else str(dt)),
                            'out': (None if o is None else str(o))})
                return out
            return wrapper
        setattr(C, nm, mk(nm, fn))
    return saved


def restore(C, saved):
    for nm, fn in saved.items():
        setattr(C, nm, fn)


def real_chain(la, env, dx):
    presc1 = _singlet(55.0e-3, -55.0e-3, 3.0e-3, 'N-BK7', 13.0e-3, 'g1')
    presc2 = _singlet(40.0e-3, -1e12, 2.5e-3, 'N-SF11', 13.0e-3, 'g2')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return la.propagate_traced_carrier_chain(
            env, [{'prescription': presc1, 'gap_before': 0.0},
                  {'prescription': presc2, 'gap_before': 12.0e-3}],
            WL, dx, r_in=55e-3, ray_subsample=8, n_workers=1,
            traced_kwargs=dict(on_undersample='silent',
                               on_noncollimated='silent'))


# ------------------------------------------------------------- (2) six legs
def six_leg(N, dx, R, n_legs=6):
    from lumenairy.propagators import carrier as C
    x = (np.arange(N) - N / 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    env0 = np.exp(-r2 / (0.18 * N * dx) ** 2).astype(np.complex128)
    arms = []
    for dt in (np.complex128, np.complex64):
        E = env0.astype(dt)
        traj = []
        for _ in range(n_legs):
            ph = C._radial_carrier_phase((N, N), dx, dx, WL, R, -1,
                                         dtype=np.asarray(E).dtype)
            E = np.asarray(E) * ph
            E = C._fourier_upsample_crop(np.asarray(E), N, N)
            ph = C._radial_carrier_phase((N, N), dx, dx, WL, R, +1,
                                         dtype=np.asarray(E).dtype)
            E = np.asarray(E) * ph
            rp = C._tilt_ramp((N, N), dx, WL, 0.02, -0.011, 0.0, 0.0, +1,
                              dtype=np.asarray(E).dtype)
            E = np.asarray(E) * rp
            traj.append(np.asarray(E).copy())
        arms.append((traj, str(np.asarray(E).dtype)))
    ref, dt_ref = arms[0]
    got, dt_got = arms[1]
    rel = [float(np.linalg.norm(r - g.astype(np.complex128))
                 / np.linalg.norm(r)) for r, g in zip(ref, got)]
    return {'rel_per_leg': rel, 'dtype_c128_arm': dt_ref,
            'dtype_c64_arm': dt_got, 'slope_per_leg': rel[-1] / len(rel)}


def main():
    la = _fix.banner()
    from lumenairy.propagators import carrier as C
    out = {'version': la.__version__}

    # ---- (4) the crop's narrowing -------------------------------------
    N, dx = 512, 1.9e-6
    rng = np.random.default_rng(7)
    x = (np.arange(N) - N / 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    env = (np.exp(-r2 / (0.15 * N * dx) ** 2)
           * (1 + 0.05 * rng.standard_normal((N, N)))).astype(np.complex128)
    crop = {}
    e32 = float(np.finfo(np.float32).eps)
    for nc, nf in ((N // 2, N), (N, N // 2), (N, N)):
        e64 = env.astype(np.complex64)
        got = C._fourier_upsample_crop(e64.copy(), nc, nf)
        ref1 = C._fourier_upsample_crop(e64.astype(np.complex128), nc, nf)
        ref128 = C._fourier_upsample_crop(env.copy(), nc, nf)
        d = float(np.abs(np.asarray(got).astype(np.complex128) - ref1).max())
        scale = float(np.abs(ref1).max())
        key = str(nc) + '->' + str(nf)
        crop[key] = {
            'out_dtype': str(np.asarray(got).dtype),
            'max_abs_diff_vs_single_rounding': d,
            'peak': scale,
            'eps32_of_peak': e32 * scale,
            'double_rounded': bool(d > 1.5 * e32 * scale),
            'rel_l2_vs_c128_input': float(
                np.linalg.norm(np.asarray(got).astype(np.complex128) - ref128)
                / np.linalg.norm(ref128))}
        print('crop ' + key + ': dtype=' + crop[key]['out_dtype']
              + ' max|c64 - narrow-once|=%.3e' % d
              + '  eps32*peak=%.3e' % (e32 * scale)
              + '  double_rounded=' + str(crop[key]['double_rounded'])
              + '  rel_vs_c128_in=%.3e' % crop[key]['rel_l2_vs_c128_input'],
              flush=True)
    out['upsample_crop'] = crop

    # ---- (2) six legs --------------------------------------------------
    sl = six_leg(512, 1.9e-6, 40.0e-3)
    out['six_leg'] = sl
    print('six-leg rel L2 per leg: '
          + ' '.join('%.3e' % v for v in sl['rel_per_leg']), flush=True)
    print('  dtypes: c128 arm ' + sl['dtype_c128_arm']
          + ', c64 arm ' + sl['dtype_c64_arm'], flush=True)

    # ---- (1) + (3) the real two-group chain ----------------------------
    Nc, dxc, w = 512, 26e-6, 3.6e-3
    xc = (np.arange(Nc) - Nc // 2) * dxc
    r2c = xc[None, :] ** 2 + xc[:, None] ** 2
    env_c = np.exp(-r2c / w ** 2).astype(np.complex128)
    res = {}
    for tag, e in (('c128', env_c), ('c64', env_c.astype(np.complex64))):
        log = []
        saved = instrument(C, log)
        try:
            r = real_chain(la, e, dxc)
        finally:
            restore(C, saved)
        f = np.asarray(r.field)
        res[tag] = {'field': f, 'dtype': str(f.dtype),
                    'power': float(np.sum(np.abs(f) ** 2)), 'log': log}
        ups = ([d for d in log if d['out'] == 'complex128']
               if tag == 'c64' else [])
        print('chain ' + tag + ': field dtype ' + str(f.dtype)
              + '  helper calls %d' % len(log)
              + '  complex128 returns %d' % len(ups), flush=True)
        for d in ups[:20]:
            print('   UPCAST:', d, flush=True)
    a, b = res['c128']['field'], res['c64']['field']
    rel = float(np.linalg.norm(a - b.astype(np.complex128))
                / np.linalg.norm(a))
    p_rel = (abs(res['c64']['power'] - res['c128']['power'])
             / res['c128']['power'])
    out['real_chain'] = {
        'N': Nc, 'dx': dxc,
        'dtype_c128_arm': res['c128']['dtype'],
        'dtype_c64_arm': res['c64']['dtype'],
        'rel_l2_field': rel, 'rel_power': p_rel,
        'field_bar_2e_5_met': bool(rel <= 2e-5),
        'energy_bar_4e_5_met': bool(p_rel <= 4e-5),
        'upcasts_on_c64_arm': [d for d in res['c64']['log']
                               if d['out'] == 'complex128'],
        'call_log_c64': res['c64']['log'],
        'call_log_c128': res['c128']['log']}
    print('real 2-group chain: rel L2 field = %.3e (bar 2e-5), '
          'rel power = %.3e (bar 4e-5)' % (rel, p_rel), flush=True)
    with open(sys.argv[1], 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', sys.argv[1])


if __name__ == '__main__':
    main()
