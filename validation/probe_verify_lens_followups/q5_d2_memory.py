"""Q5 (D2) -- the per-call transient, and the full-grid complex128 census.

Task 2.  Three measurements, my own fixtures:

1. whole-call ``tracemalloc`` peak of ONE ``carrier_referenced_envelope`` /
   ``carrier_referenced_reconstruct`` call, warm, at N = 1024 / 2048 / 4096,
   scalar / astigmatic / single-axis carrier, complex64 and complex128.
   ``tracemalloc`` counts REQUESTED sizes, so these are fixed by shapes and
   dtypes and must reproduce to the byte on any build.  N=1024 is the
   "below N ~ 1414 the 32 MB band IS the grid" statement: no transient
   saving there, only the narrower output.
2. every reference-phase helper's return, per two-group carrier chain,
   logged by name / dtype / size / caller frame -- the "5 full-grid
   complex128 phasors -> 0" count.
3. the returned FIELD hashed in every case, so the memory change is shown
   value-free.

Usage: python q5_d2_memory.py <out.json> --tree <arm tree>
"""
from __future__ import annotations

import os
import sys
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vf  # noqa: E402

MIB = 2 ** 20
CARRIERS = {'scalar': 71e-3, 'astig': (71e-3, -59e-3), 'one_axis': (71e-3,
                                                                    np.inf)}


def _field(N, dx, dtype, seed=17):
    """A speckled envelope -- not a smooth Gaussian, so nothing can be
    satisfied by a degenerate array."""
    rng = np.random.default_rng(seed)
    x = (np.arange(N) - N / 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    a = np.exp(-r2 / (0.3 * N * dx) ** 2)
    a = a * (1.0 + 0.05 * rng.standard_normal((N, N)))
    return a.astype(dtype)


def peaks(C, N, dx, R, dtype, fn_name):
    fn = getattr(C, fn_name)
    E = _field(N, dx, dtype)
    out = fn(E, R, _vf.WL, dx)                       # warm
    h0 = _vf.h(np.asarray(out))
    del out
    tracemalloc.start()
    tracemalloc.reset_peak()
    out = fn(E, R, _vf.WL, dx)
    _, pk = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    o = np.asarray(out)
    assert _vf.h(o) == h0
    return {'peak_bytes': int(pk), 'peak_MiB': round(pk / MIB, 2),
            'peak_c64_grids': round(pk / (8.0 * N * N), 3),
            'out_dtype': str(o.dtype), 'hash': h0}


def census(la, C, N, dx):
    """Every reference-phase helper return in a TWO-GROUP carrier chain."""
    import warnings
    names = ('_radial_carrier_phase', '_axis_carrier_phase', '_phasor_rows',
             '_tilt_ramp', '_tilt_exactness_phase',
             '_sphere_parab_conversion', '_narrow_rows')
    log = []
    saved = {}
    for nm in names:
        real = getattr(C, nm, None)
        if real is None:
            continue
        saved[nm] = real

        def mk(real=real, nm=nm):
            def wrap(*a, **kw):
                o = real(*a, **kw)
                if o is not None:
                    import sys as _s
                    fr = _s._getframe(1)
                    log.append({'helper': nm, 'dtype': str(np.dtype(o.dtype)),
                                'size': int(np.size(o)),
                                'caller': fr.f_code.co_name,
                                'dtype_kw': str(kw.get('dtype', 'positional'))
                                })
                return o
            return wrap
        setattr(C, nm, mk())
    try:
        res = {}
        g1 = {'prescription': _vf.presc_meniscus(), 'gap_before': 0.0}
        g2 = {'prescription': _vf.presc_doublet(), 'gap_before': 22e-3}
        for dt in (np.complex128, np.complex64):
            log.clear()
            E = _field(N, dx, dt, seed=23)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                r = la.propagate_traced_carrier_chain(
                    E, [g1, g2], _vf.WL, dx, r_in=0.075, ray_subsample=4,
                    n_workers=1,
                    traced_kwargs=dict(on_undersample='silent',
                                       on_noncollimated='silent',
                                       on_aperture_beam='silent',
                                       parallel_amp=False),
                    on_multi_congruence='ignore', on_na_proximity='ignore',
                    on_decentred_fit='ignore', on_gap_paraxial='ignore',
                    on_gap_frame='ignore', on_rs_fine_clamp='ignore',
                    on_ram_cap='ignore')
            wide = [d for d in log
                    if d['dtype'] == 'complex128' and d['size'] >= N * N]
            by_sig = {}
            for d in log:
                k = (f"{d['helper']}|caller={d['caller']}|"
                     f"dtype_kw={d['dtype_kw']}|out={d['dtype']}|"
                     f"full_grid={d['size'] >= N * N}")
                by_sig[k] = by_sig.get(k, 0) + 1
            res[str(np.dtype(dt))] = {
                'n_helper_returns': len(log),
                'n_full_grid_c128': len(wide),
                'bytes_full_grid_c128': int(sum(d['size'] * 16
                                                for d in wide)),
                'by_signature': dict(sorted(by_sig.items())),
                'field_hash': _vf.h(np.asarray(r.field)),
                'field_dtype': str(np.asarray(r.field).dtype),
            }
            print(f"  chain {np.dtype(dt)}: {len(log)} helper returns, "
                  f"{len(wide)} FULL-GRID complex128 "
                  f"({sum(d['size'] * 16 for d in wide) / MIB:.1f} MiB), "
                  f"field {res[str(np.dtype(dt))]['field_hash']}", flush=True)
        return res
    finally:
        for nm, real in saved.items():
            setattr(C, nm, real)


def main():
    args = _vf.argp(__doc__).parse_args()
    la = _vf.banner(args.tree)
    from lumenairy.propagators import carrier as C
    print(f"# free RAM {_vf.free_gb()} GB   "
          f"_PHASOR_BAND_BYTES={C._PHASOR_BAND_BYTES}", flush=True)
    out = {'transient': {}, 'census': {},
           'band_bytes': float(C._PHASOR_BAND_BYTES)}
    for N, dx in ((1024, 6.0e-6), (2048, 3.0e-6), (4096, 1.5e-6)):
        for cname, R in CARRIERS.items():
            for fn_name in ('carrier_referenced_envelope',
                            'carrier_referenced_reconstruct'):
                for dt in (np.complex128, np.complex64):
                    k = f'N={N}|{cname}|{fn_name[18:]}|{np.dtype(dt)}'
                    out['transient'][k] = peaks(C, N, dx, R, dt, fn_name)
                    print(f"  {k:52s} peak "
                          f"{out['transient'][k]['peak_MiB']:8.2f} MiB "
                          f"({out['transient'][k]['peak_c64_grids']:.3f} "
                          f"c64 grids)  {out['transient'][k]['hash']}",
                          flush=True)
    out['census'] = census(la, C, 288, 11e-6)
    _vf.dump(args, {**out, 'free_gb': _vf.free_gb()})


if __name__ == '__main__':
    main()
