"""Q9 (D3) -- the single-precision transform pair, measured and judged.

Task 4.  D3 is a DECISION (keep the single-precision pair) resting on four
numbers.  Re-measured here on my own chain and my own crop fixture:

* a real two-group traced carrier chain WITH an exact focus readout -- the
  only shape that reaches ``_fourier_upsample_crop`` -- in three arms:
  A = complex64 with the shipped single-precision pair, B = complex64 with the
  pair forced to complex128 and narrowed ONCE on return (a wrapper around the
  public helper, so nothing internal is assumed), C = complex128;
* the same chain with ``final_leg='paraxial'``, to count how many times the
  plain chain reaches the crop at all;
* the crop alone against a narrow-once reference over n_fine 256 -> 2048, to
  test the ~sqrt(log2 N) growth statement;
* ``numpy.fft.fft2`` of a complex64 array, to check the premise (numpy >= 2.0
  returns complex64) on this build.

Usage: python q9_d3_fft.py <out.json> --tree <arm tree>
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vf  # noqa: E402

N, DX = 288, 11e-6
W = 0.95e-3
SUB = 4


def _chain(la, groups, E, **over):
    kw = dict(r_in=0.075, ray_subsample=SUB, n_workers=1,
              traced_kwargs=dict(on_undersample='silent',
                                 on_noncollimated='silent',
                                 on_aperture_beam='silent',
                                 parallel_amp=False),
              on_multi_congruence='ignore', on_na_proximity='ignore',
              on_decentred_fit='ignore', on_gap_paraxial='ignore',
              on_gap_frame='ignore', on_rs_fine_clamp='ignore',
              on_ram_cap='ignore')
    kw.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return la.propagate_traced_carrier_chain(E, groups, _vf.WL, DX, **kw)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a.astype(np.complex128)
                                - b.astype(np.complex128))
                 / np.linalg.norm(a.astype(np.complex128)))


def _relpow(a, b):
    pa = float(np.sum(np.abs(np.asarray(a).astype(np.complex128)) ** 2))
    pb = float(np.sum(np.abs(np.asarray(b).astype(np.complex128)) ** 2))
    return abs(pa - pb) / pa


def main():
    args = _vf.argp(__doc__).parse_args()
    la = _vf.banner(args.tree)
    from lumenairy.propagators import carrier as C
    out = {}

    # ---- the premise -------------------------------------------------------
    z = np.ones((8, 8), dtype=np.complex64)
    out['numpy_fft2_complex64_returns'] = str(np.fft.fft2(z).dtype)
    out['numpy_version'] = np.__version__
    print(f"  np.fft.fft2(complex64).dtype = "
          f"{out['numpy_fft2_complex64_returns']}", flush=True)

    g1 = {'prescription': _vf.presc_meniscus(), 'gap_before': 0.0}
    g2 = {'prescription': _vf.presc_doublet(), 'gap_before': 22e-3}
    E = _vf.gauss(N, DX, W)

    real = C._fourier_upsample_crop
    calls = {'n': 0}

    def counting(env, nc, nf):
        calls['n'] += 1
        return real(env, nc, nf)

    def forced(env, nc, nf):
        """The pair in complex128, narrowed ONCE on return."""
        calls['n'] += 1
        dt = np.asarray(env).dtype
        r = real(np.asarray(env).astype(np.complex128), nc, nf)
        return np.asarray(r).astype(dt)

    for tag, leg, dist in (('exact', 'exact', 4.0e-3),
                           ('paraxial', 'paraxial', 4.0e-3)):
        arms = {}
        for arm, wrapper, dt in (('A_shipped_c64', counting, np.complex64),
                                 ('B_forced_c128_c64', forced, np.complex64),
                                 ('C_c128', counting, np.complex128)):
            calls['n'] = 0
            C._fourier_upsample_crop = wrapper
            try:
                r = _chain(la, [g1, g2], E.astype(dt), final_leg=leg,
                           final_distance=dist, carrier_reference='sphere')
                f = np.asarray(r.field)
            finally:
                C._fourier_upsample_crop = real
            arms[arm] = {'field': _vf.field_record(f), 'crop_calls':
                         calls['n'], 'array': f}
            print(f"  chain[{tag}] {arm:18s} dtype={f.dtype} "
                  f"crop_calls={calls['n']} hash={_vf.h(f)}", flush=True)
        blk = {a: {k: v for k, v in arms[a].items() if k != 'array'}
               for a in arms}
        cc = arms['C_c128']['array']
        blk['A_vs_C_rel_l2'] = _rel(cc, arms['A_shipped_c64']['array'])
        blk['A_vs_C_rel_power'] = _relpow(cc, arms['A_shipped_c64']['array'])
        blk['B_vs_C_rel_l2'] = _rel(cc, arms['B_forced_c128_c64']['array'])
        blk['B_vs_C_rel_power'] = _relpow(cc, arms['B_forced_c128_c64']
                                          ['array'])
        blk['A_vs_B_rel_l2'] = _rel(arms['B_forced_c128_c64']['array'],
                                    arms['A_shipped_c64']['array'])
        blk['B_better_than_A'] = bool(blk['B_vs_C_rel_l2']
                                      < blk['A_vs_C_rel_l2'])
        out[f'chain_{tag}'] = blk
        print(f"  chain[{tag}]  A vs C rel L2 {blk['A_vs_C_rel_l2']:.4e} / "
              f"power {blk['A_vs_C_rel_power']:.4e}   "
              f"B vs C {blk['B_vs_C_rel_l2']:.4e} / "
              f"{blk['B_vs_C_rel_power']:.4e}   "
              f"B better? {blk['B_better_than_A']}", flush=True)

    # ---- the crop alone, against a narrow-once reference --------------------
    lad = {}
    for nc, nf in ((128, 256), (256, 512), (512, 1024), (1024, 2048)):
        env = _vf.speckled(nc, 1.9e-6, 0.15 * nc * 1.9e-6, seed=9)
        e64 = env.astype(np.complex64)
        got = np.asarray(real(e64, nc, nf))
        ref_once = np.asarray(real(env, nc, nf)).astype(np.complex64)
        ref128 = np.asarray(real(env, nc, nf))
        peak = float(np.abs(ref128).max())
        lad[f'{nc}->{nf}'] = {
            'max_abs_diff_vs_narrow_once': float(np.abs(
                got.astype(np.complex128) - ref_once.astype(
                    np.complex128)).max()),
            'eps32_x_peak': float(np.finfo(np.float32).eps * peak),
            'rel_l2_vs_c128_input': _rel(ref128, got),
            'ratio_to_eps32_peak': float(np.abs(
                got.astype(np.complex128)
                - ref_once.astype(np.complex128)).max()
                / (np.finfo(np.float32).eps * peak)),
            'sqrt_log2_nf': float(np.sqrt(np.log2(nf))),
        }
        print(f"  crop {nc}->{nf}: max|c64-narrow_once|="
              f"{lad[f'{nc}->{nf}']['max_abs_diff_vs_narrow_once']:.4e}  "
              f"eps32*peak={lad[f'{nc}->{nf}']['eps32_x_peak']:.4e}  "
              f"ratio={lad[f'{nc}->{nf}']['ratio_to_eps32_peak']:.2f}  "
              f"relL2={lad[f'{nc}->{nf}']['rel_l2_vs_c128_input']:.4e}",
              flush=True)
    out['crop_ladder'] = lad
    r0 = lad['128->256']['rel_l2_vs_c128_input']
    r1 = lad['1024->2048']['rel_l2_vs_c128_input']
    out['ladder_growth_over_4_octaves'] = r1 / r0
    out['sqrt_log2_growth_predicted'] = (np.sqrt(np.log2(2048))
                                         / np.sqrt(np.log2(256)))
    print(f"  ladder growth {r1 / r0:.4f} vs sqrt(log2) prediction "
          f"{out['sqrt_log2_growth_predicted']:.4f}", flush=True)
    _vf.dump(args, {**out, 'free_gb': _vf.free_gb()})


if __name__ == '__main__':
    main()
