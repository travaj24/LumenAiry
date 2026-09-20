"""(7) _fft2_pair IDENTITY, and whether the STATED CONSEQUENCE obtains.

The claim: routing the NumPy path through ``backend.fft2`` would break
``_bluestein_2d``'s chirp-kernel cache, which is keyed on
``fft2 is fft_infra._fft2``, and would "silently turn the cache off for every
Collins leg".

Measured here: (b) the identity itself; (c) whether the cache ACTUALLY
engages on a Collins leg as shipped, whether it engages when the raw
callables reach ``_bluestein_2d``'s 2-D arm, and whether a wrapper turns it
off -- counting stores and hits rather than asserting.

    python v2_fftpair.py <tree> <out.json>
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                            # noqa: E402
from vlib import anchor, build_tag, write_json                 # noqa: E402

WL, N, DX, R_IN, Z, R_REF = 1.064e-6, 96, 5.5e-6, -0.028, 2.3e-3, -0.0215


def main():
    tree, out_path = sys.argv[1], sys.argv[2]
    anchor(tree)
    import lumenairy.propagators.carrier as CA
    import lumenairy.propagators._bluestein as BL
    from lumenairy.propagators.fft_infra import _fft2, _ifft2
    from lumenairy.backend import fft2 as backend_fft2
    res = {'build': build_tag(), 'tree': tree}

    # ---- (a) the cache key, quoted from the source ----------------------
    src = open(BL.__file__, encoding='cp1252', errors='replace').read()
    lines = src.splitlines()
    idx = [i for i, ln in enumerate(lines) if 'cache_key = (' in ln]
    quote = []
    if idx:
        i0 = max(0, idx[0] - 8)
        quote = ['%d: %s' % (i + 1, lines[i])
                 for i in range(i0, min(len(lines), idx[0] + 12))]
    res['a_cache_key_source'] = quote
    res['a_key_condition_lineno'] = [i + 1 for i, ln in enumerate(lines)
                                     if 'if fft2 is _default_np_fft2' in ln]
    res['a_separable_gate_lineno'] = [
        i + 1 for i, ln in enumerate(lines)
        if "if (separable if method == 'auto' else method == 'separable')"
        in ln]

    # ---- (b) identity ----------------------------------------------------
    f, i = CA._fft2_pair(np, False)
    res['b_numpy_pair_is_fft_infra'] = {'fft2_is__fft2': f is _fft2,
                                        'ifft2_is__ifft2': i is _ifft2,
                                        'fft2_is_backend_fft2':
                                            f is backend_fft2}
    try:
        import jax.numpy as jnp
        fj, ij = CA._fft2_pair(jnp, True)
        res['b_jax_pair'] = {'fft2_is_jnp_fft_fft2': fj is jnp.fft.fft2,
                             'ifft2_is_jnp_fft_ifft2': ij is jnp.fft.ifft2}
    except ImportError:
        res['b_jax_pair'] = 'jax unavailable'
    res['b_backend_fft2_is_distinct_object'] = backend_fft2 is not _fft2

    # value agreement, bit for bit, NumPy side
    ax = (np.arange(N) - N // 2) * DX
    X, Y = np.meshgrid(ax, ax)
    env = np.exp(-(X ** 2 + Y ** 2) / (41e-6) ** 2).astype(np.complex128)
    a = np.ascontiguousarray(f(np.ascontiguousarray(env))).copy()
    b = np.ascontiguousarray(backend_fft2(np.ascontiguousarray(env))).copy()
    res['b_value_bit_equal'] = bool(np.array_equal(a.view(np.float64),
                                                   b.view(np.float64)))

    # ---- (c) DOES THE CACHE ENGAGE ON A COLLINS LEG? ---------------------
    def snap():
        return (len(BL._H_FFT_CACHE), int(BL._H_FFT_CACHE_HITS),
                int(BL._h_fft_cache_bytes()))

    KW = dict(dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N, R_ref=R_REF,
              gap_kernel='fresnel', on_collins_sampling='ignore')

    def run_leg(n=3):
        for _ in range(n):
            CA._collins_transport(env, R_IN, Z, WL, DX, DX, **KW)

    BL._clear_h_fft_cache()
    before = snap()
    run_leg()
    after = snap()
    res['c_shipped_separable_flag'] = bool(
        CA._EXACT_READOUT_SEPARABLE_BLUESTEIN)
    res['c_collins_leg_as_shipped'] = {
        'before': before, 'after': after,
        'entries_stored': after[0] - before[0],
        'hits_gained': after[1] - before[1],
        'cache_engaged': bool(after[0] > before[0] or after[1] > before[1])}

    # now with the separable route OFF, so _bluestein_2d's 2-D arm runs
    keep = CA._EXACT_READOUT_SEPARABLE_BLUESTEIN
    try:
        CA._EXACT_READOUT_SEPARABLE_BLUESTEIN = False
        BL._clear_h_fft_cache()
        before = snap()
        run_leg()
        after = snap()
        res['c_collins_leg_separable_off'] = {
            'before': before, 'after': after,
            'entries_stored': after[0] - before[0],
            'hits_gained': after[1] - before[1],
            'cache_engaged': bool(after[0] > before[0]
                                  or after[1] > before[1])}
    finally:
        CA._EXACT_READOUT_SEPARABLE_BLUESTEIN = keep

    # ---- (c2) the stated consequence, demonstrated at the primitive ------
    # Same call, three spellings of the fft2 callable, 2-D arm forced.
    g = np.ascontiguousarray(env, dtype=np.complex128)
    alpha = float(DX) * float(DX) / (WL * (Z))

    def wrapper_fft2(x):
        return _fft2(x)

    prim = {}
    for name, (ff, ii) in (
            ('raw_fft_infra', (_fft2, _ifft2)),
            ('backend_fft2_wrapper', (backend_fft2, _ifft2)),
            ('local_lambda_wrapper', (wrapper_fft2, _ifft2))):
        BL._clear_h_fft_cache()
        b0 = snap()
        outs = []
        for _ in range(3):
            outs.append(np.asarray(BL._bluestein_centred_2d(
                g, alpha, alpha, N, N, k_centre_out_x=N / 2.0,
                k_centre_out_y=N / 2.0, sign=-1, xp=np, fft2=ff, ifft2=ii,
                target_cdtype=np.complex128, separable=False)).copy())
        b1 = snap()
        prim[name] = {
            'before': b0, 'after': b1,
            'entries_stored': b1[0] - b0[0], 'hits_gained': b1[1] - b0[1],
            'cache_engaged': bool(b1[0] > b0[0] or b1[1] > b0[1]),
            'bytes_retained': b1[2],
            'answers_bit_equal_across_repeats': bool(
                np.array_equal(outs[0].view(np.float64),
                               outs[2].view(np.float64)))}
    # and do the three spellings agree on the ANSWER?
    BL._clear_h_fft_cache()
    ref = np.asarray(BL._bluestein_centred_2d(
        g, alpha, alpha, N, N, k_centre_out_x=N / 2.0, k_centre_out_y=N / 2.0,
        sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2,
        target_cdtype=np.complex128, separable=False)).copy()
    BL._clear_h_fft_cache()
    wrp = np.asarray(BL._bluestein_centred_2d(
        g, alpha, alpha, N, N, k_centre_out_x=N / 2.0, k_centre_out_y=N / 2.0,
        sign=-1, xp=np, fft2=wrapper_fft2, ifft2=_ifft2,
        target_cdtype=np.complex128, separable=False)).copy()
    prim['answers_identical_raw_vs_wrapper'] = bool(np.array_equal(
        ref.view(np.float64), wrp.view(np.float64)))
    res['c2_primitive'] = prim

    # ---- (c3) what the cache is worth when it DOES engage ----------------
    import time
    BL._clear_h_fft_cache()
    t = []
    for label in ('cold', 'warm'):
        t0 = time.perf_counter()
        for _ in range(5):
            BL._bluestein_centred_2d(
                g, alpha, alpha, N, N, k_centre_out_x=N / 2.0,
                k_centre_out_y=N / 2.0, sign=-1, xp=np, fft2=_fft2,
                ifft2=_ifft2, target_cdtype=np.complex128, separable=False)
        t.append((label, time.perf_counter() - t0))
    res['c3_timing_raw_callable_s'] = dict(t)
    BL._clear_h_fft_cache()
    t = []
    for label in ('cold', 'warm'):
        t0 = time.perf_counter()
        for _ in range(5):
            BL._bluestein_centred_2d(
                g, alpha, alpha, N, N, k_centre_out_x=N / 2.0,
                k_centre_out_y=N / 2.0, sign=-1, xp=np, fft2=wrapper_fft2,
                ifft2=_ifft2, target_cdtype=np.complex128, separable=False)
        t.append((label, time.perf_counter() - t0))
    res['c3_timing_wrapper_s'] = dict(t)
    res['c3_cache_state_after_timing'] = snap()

    write_json(res, out_path)
    print(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main()
