"""Round 3 (round 2, D-3 / D-4 / D-5) -- the two finite-difference ladders and
the cross-backend FFT spread, re-measured on both builds.

    PYTHONPATH=<tree> python r3_fd_ladders.py <tree> OUT.json

Three readings the round-2 addendum states and this round restates:

``D-3`` the QUADRATIC merit's third-difference estimate over its own floor,
    at every rung of its ladder (the addendum quotes one rung and labels it
    with another's number).
``D-4`` the CUBIC control's same statistic over ITS ladder, and one decade
    further, where a third difference's ``h^-3`` round-off floor swamps any
    real ``P'''``.
``D-5`` the cross-backend FFT spread the bar in
    ``test_wave5_h2_collins_jax.py::_fft_spread_bar`` is built from, what the
    bar returns, and whether ``np.fft.fft2`` and ``jnp.fft.fft2`` agree bit
    for bit at the shipped shape.
"""
import json
import os
import sys

import numpy as np


def anchor(tree):
    import lumenairy
    got = os.path.realpath(lumenairy.__file__)
    want = os.path.realpath(tree)
    try:
        same = os.path.commonpath([got, want]) == want
    except ValueError:                      # different drives on Windows
        same = False
    if not same:
        raise SystemExit(f"WRONG TREE: {got!r} is not under {want!r}")
    return lumenairy


def main(tree, out_path):
    lumenairy = anchor(tree)
    import jax

    # The module's own autouse fixture forces float64; a probe that skipped it
    # would be measuring dtype policy instead of the port.  MEASURED once
    # without it: every ladder reading came back at complex64 precision.
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from tests.unit.test_wave5_h2_collins_jax import (
        CA,
        DX,
        R_IN,
        R_REF,
        WL,
        N,
        Z,
        _fft_spread_bar,
        _gauss,
        _merit_factory,
    )

    eps = float(np.finfo(np.float64).eps)
    res = {'lumenairy_file': lumenairy.__file__,
           'platform': sys.platform, 'python': sys.version.split()[0],
           'jax': jax.__version__, 'numpy': np.__version__, 'eps': eps}

    def ladder(merit, hs):
        amp0 = jnp.asarray(np.real(_gauss()))
        a = np.asarray(amp0)
        ij = np.unravel_index(int(np.argmax(a)), a.shape)
        P0 = float(merit(amp0))
        g_ij = float(np.asarray(jax.grad(merit)(amp0))[ij])
        rows = []
        for h in hs:
            def _at(sign, mult=1, h=h):
                ap = a.copy()
                ap[ij] += sign * mult * h
                return float(merit(jnp.asarray(ap)))
            fd = (_at(+1) - _at(-1)) / (2.0 * h)
            p3 = (_at(+1, 2) - 2 * _at(+1) + 2 * _at(-1) - _at(-1, 2)) \
                / (2.0 * h ** 3)
            floor = eps * abs(P0) / h ** 3
            rows.append({'h': h, 'p3': p3, 'floor': floor,
                         'over_floor': abs(p3) / floor,
                         'exactly_zero': p3 == 0.0,
                         'rel_fd_vs_grad': abs(fd - g_ij) / abs(g_ij)})
        return {'P0': P0, 'g_ij': g_ij, 'rows': rows}

    # --- D-3: the shipped QUADRATIC merit, its own ladder -------------------
    quad = _merit_factory(gap_kernel='fresnel', on_collins_sampling='ignore')
    res['D3_quadratic'] = ladder(quad, (1e-1, 1e-2, 1e-3, 1e-4))

    # --- D-4: the CUBIC control, its ladder and one decade further ----------
    def cubic(amp):
        out = CA._collins_transport(
            amp.astype(jnp.complex128), R_IN, Z, WL, DX, DX, dx_out=DX,
            dy_out=DX, N_out_x=N, N_out_y=N, R_ref=R_REF,
            gap_kernel='fresnel', on_collins_sampling='ignore')
        return (jnp.abs(out[N // 2, N // 2]) ** 2) ** 3

    res['D4_cubic'] = ladder(cubic, (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4))

    # --- D-5: the cross-backend bar and where its spread comes from ---------
    # The SHIPPED fixture, i.e. the one every call site passes.
    from lumenairy.propagators.fft_infra import _fft2
    env = _gauss()
    E = np.ascontiguousarray(env, dtype=np.complex128)
    lib = np.asarray(_fft2(E))
    npy = np.fft.fft2(E)
    jxy = np.asarray(jnp.fft.fft2(jnp.asarray(E, dtype=jnp.complex128)))
    nrm = float(np.linalg.norm(lib))

    def r(a, b):
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / nrm)

    bar = float(_fft_spread_bar(env))
    spread = r(lib, jxy)
    res['D5'] = {
        'np_vs_jnp_bitwise_identical': bool(
            np.array_equal(npy.view(np.float64), jxy.view(np.float64))),
        'np_vs_jnp_rel': r(npy, jxy),
        'lib_vs_np_rel': r(lib, npy),
        'lib_vs_jnp_spread': spread,
        'six_times_spread': 6.0 * spread,
        'bar': bar,
        'thirty_two_eps': 32.0 * eps,
        'bar_is_exactly_32_eps': bar == 32.0 * eps,
        'floor_dominates_by': bar / (6.0 * spread) if spread else None,
    }

    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=str)

    print(f"lumenairy={lumenairy.__file__}  ({res['platform']}, "
          f"py{res['python']}, jax {res['jax']}, numpy {res['numpy']})")
    for tag in ('D3_quadratic', 'D4_cubic'):
        d = res[tag]
        print(f"--- {tag}  P0={d['P0']:.6e}  g={d['g_ij']:.6e} ---")
        print("   h        p3            over floor     exactly 0?  "
              "|fd-g|/|g|")
        for row in d['rows']:
            print(f"   {row['h']:.0e}  {row['p3']:+.5e}  "
                  f"{row['over_floor']:14.4f}  {str(row['exactly_zero']):>5}  "
                  f"  {row['rel_fd_vs_grad']:.4e}")
    d = res['D5']
    print("--- D5 cross-backend FFT ---")
    print(f"   np.fft.fft2 vs jnp.fft.fft2 bitwise identical: "
          f"{d['np_vs_jnp_bitwise_identical']}  (rel {d['np_vs_jnp_rel']:.4e})")
    print(f"   library _fft2 vs np {d['lib_vs_np_rel']:.4e}, vs jnp (the "
          f"SPREAD) {d['lib_vs_jnp_spread']:.4e}")
    print(f"   6 x spread = {d['six_times_spread']:.4e};  bar = {d['bar']!r}; "
          f"32*eps = {d['thirty_two_eps']!r}; equal: "
          f"{d['bar_is_exactly_32_eps']}; floor dominates by "
          f"{d['floor_dominates_by']:.3f}x")
    print(f"-> {out_path}")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
