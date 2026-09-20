"""WP-C4 task 6 -- the rule applies identically on every backend.

THE CLAIM.  :func:`_auto_selects_direct` reads four integers, so the route a
shape takes cannot depend on the array namespace; and the dense route needs no
FFT at all (two ``xp.matmul`` calls), so it is reachable on a backend whose FFT
is unusable.  Both halves are measured here rather than argued:

* the SELECTION, asked on NumPy, on JAX and on CuPy for the same 30 shapes, is
  the same list of answers;
* the DISPATCH, driven through the primitives with ``xp=jax.numpy`` /
  ``xp=cupy``, matches what the rule says on that backend, and agrees with the
  NumPy answer to a derived bar rather than bit for bit (a different device's
  BLAS is entitled to different last bits -- this file never pins them);
* JAX is exercised eager AND under ``jit``, because a tracer is where a rule
  that read anything but a static shape would break: a traced call has no
  concrete array to look at, and ``_auto_selects_direct`` takes Python ints
  from ``E.shape``, which stay static under a trace.

PREMISE-GATED, NOT SKIPPED.  A backend that is not installed, or a device that
this box cannot reach, is RECORDED as a premise reading with the exception it
raised -- never turned into a silent pass.  This box's cuFFT DLL is broken
(the reason ``tests/unit/test_niche_k2_carrier_backends.py`` skips its
propagating CuPy arms), which is itself the point: the dense route does not
use it.

    PYTHONPATH=<tree> python c4_backends.py <tree> OUT.json
"""
from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c4lib  # noqa: E402

SHAPES = [
    (32, 32, 1, 1), (64, 64, 2, 2), (96, 96, 3, 3), (128, 128, 4, 4),
    (256, 256, 8, 8), (512, 512, 16, 16), (1024, 1024, 32, 32),
    (33, 33, 1, 1), (65, 65, 2, 2), (320, 320, 10, 10),
    (512, 512, 17, 16), (1024, 1024, 33, 32), (256, 256, 9, 8),
    (64, 64, 4, 4), (48, 48, 24, 24), (28, 22, 15, 13), (24, 24, 12, 12),
    (128, 128, 128, 128), (64, 64, 256, 256), (16, 16, 8, 8),
    (512, 1024, 16, 32), (512, 1024, 16, 64), (512, 1024, 64, 32),
    (1024, 512, 32, 16), (768, 768, 24, 24), (768, 768, 25, 24),
    (100, 200, 3, 6), (100, 200, 3, 7), (200, 100, 6, 3), (7, 5, 1, 1),
]

#: shapes small enough to actually DRIVE on a device, both sides of the rule
DRIVE = [(64, 64, 2, 2), (128, 128, 4, 4), (256, 256, 8, 8), (96, 96, 3, 3),
         (64, 64, 8, 8), (24, 24, 12, 12), (48, 48, 24, 24), (32, 32, 16, 16)]


def _bar(np, E, N, M, eps):
    """``(g_dense + g_dense) * eps * sum|E|`` -- two BLAS sandwiches on two
    different kernels, so the bar is twice one route's growth.  ABSOLUTE."""
    import math
    n = int(E.size)
    return 2.0 * math.sqrt(float(n)) * eps * float(np.sum(np.abs(E)))


def main(tree, out_path):
    lum = c4lib.anchor(tree)
    import numpy as np
    from lumenairy.propagators import _bluestein as B

    eps = float(np.finfo(np.float64).eps)
    out = {'build': c4lib.build_tag(), 'python': sys.version.split()[0],
           'lumenairy_file': lum.__file__,
           'ratio': float(B._MFT_DIRECT_MAX_RATIO),
           'numpy_selection': [bool(B._auto_selects_direct(*s))
                               for s in SHAPES],
           'shapes': [list(s) for s in SHAPES],
           'backends': {}}

    # ---------------- JAX --------------------------------------------------
    jx = {'premise': None}
    try:
        import jax
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        jx['premise'] = 'available'
        jx['version'] = jax.__version__
        jx['devices'] = [str(d) for d in jax.devices()]
        jx['x64'] = bool(jax.config.jax_enable_x64)
        # (a) the SELECTION is namespace-free: same four ints, same answers
        jx['selection'] = [bool(B._auto_selects_direct(*s)) for s in SHAPES]
        jx['selection_matches_numpy'] = (jx['selection']
                                         == out['numpy_selection'])
        # (b) the DISPATCH, driven on the tracer's own namespace
        rows = []
        for (ny, nx, my, mx) in DRIVE:
            rng = np.random.default_rng(7)
            E = (rng.standard_normal((ny, nx))
                 + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)
            alpha = 1.0e3 / float(max(ny, nx, my, mx)) ** 2
            says = bool(B._auto_selects_direct(ny, nx, my, mx))
            kwj = dict(sign=-1, xp=jnp, fft2=jnp.fft.fft2,
                       ifft2=jnp.fft.ifft2)
            Ej = jnp.asarray(E)
            eager = np.asarray(B._bluestein_2d(Ej, alpha, alpha, my, mx,
                                               **kwj))
            dense_j = np.asarray(B._bluestein_2d(
                Ej, alpha, alpha, my, mx, method='direct', **kwj))
            chirp_j = np.asarray(B._bluestein_2d(
                Ej, alpha, alpha, my, mx, method='bluestein', **kwj))

            def _call(a):
                return B._bluestein_2d(a, alpha, alpha, my, mx, **kwj)

            jitted = np.asarray(jax.jit(_call)(Ej))
            ref_np = B._bluestein_2d(E, alpha, alpha, my, mx, sign=-1, xp=np,
                                     fft2=np.fft.fft2, ifft2=np.fft.ifft2,
                                     method='direct' if says else 'bluestein')
            bar = _bar(np, E, max(ny, nx), max(my, mx), eps)
            rows.append({
                'shape': [ny, nx, my, mx], 'rule_says_direct': says,
                'eager_matches': ('direct'
                                  if np.array_equal(eager, dense_j)
                                  else 'bluestein'
                                  if np.array_equal(eager, chirp_j)
                                  else 'NONE'),
                'jit_matches': ('direct' if np.array_equal(jitted, dense_j)
                                else 'bluestein'
                                if np.array_equal(jitted, chirp_j)
                                else 'NONE'),
                'eager_vs_numpy_max_abs': float(np.max(np.abs(eager
                                                              - ref_np))),
                'jit_vs_eager_max_abs': float(np.max(np.abs(jitted - eager))),
                'derived_bar': bar,
                'inside_bar': bool(np.max(np.abs(eager - ref_np)) < bar)})
            del E, Ej
        jx['dispatch'] = rows
        jx['dispatch_as_rule_says'] = sum(
            1 for r in rows
            if (r['eager_matches'] == 'direct') == r['rule_says_direct'])
        jx['jit_agrees_with_eager'] = sum(
            1 for r in rows if r['jit_matches'] == r['eager_matches'])
        jx['inside_bar'] = sum(1 for r in rows if r['inside_bar'])
        jx['n'] = len(rows)
    except Exception as exc:                        # noqa: BLE001 -- recorded
        jx['premise'] = f"{type(exc).__name__}: {exc}"
        jx['traceback'] = traceback.format_exc()[-1500:]
    out['backends']['jax'] = jx

    # ---------------- CuPy -------------------------------------------------
    cu = {'premise': None}
    try:
        import cupy as cp
        cu['version'] = cp.__version__
        # the DEVICE reading is its own premise, taken before anything else
        dev = cp.cuda.runtime.getDeviceProperties(0)
        cu['device'] = str(dev.get('name'))
        _ = cp.asarray(np.ones(4)) + 1          # does the device actually run?
        cu['premise'] = 'available'
        cu['fft_premise'] = None
        try:
            cp.fft.fft2(cp.asarray(np.ones((8, 8), dtype=np.complex128)))
            cu['fft_premise'] = 'cuFFT works'
        except Exception as exc:                    # noqa: BLE001 -- recorded
            cu['fft_premise'] = f"cuFFT BROKEN: {type(exc).__name__}: {exc}"
        cu['selection'] = [bool(B._auto_selects_direct(*s)) for s in SHAPES]
        cu['selection_matches_numpy'] = (cu['selection']
                                         == out['numpy_selection'])
        rows = []
        for (ny, nx, my, mx) in DRIVE:
            rng = np.random.default_rng(7)
            E = (rng.standard_normal((ny, nx))
                 + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)
            alpha = 1.0e3 / float(max(ny, nx, my, mx)) ** 2
            says = bool(B._auto_selects_direct(ny, nx, my, mx))
            Ec = cp.asarray(E)
            row = {'shape': [ny, nx, my, mx], 'rule_says_direct': says}
            # The dense route needs NO FFT, so it is reachable here even with
            # a broken cuFFT; the chirp arm is attempted and its failure, if
            # any, is recorded rather than hidden.
            try:
                got = cp.asnumpy(B._bluestein_2d(
                    Ec, alpha, alpha, my, mx, sign=-1, xp=cp,
                    fft2=cp.fft.fft2, ifft2=cp.fft.ifft2))
                dense_c = cp.asnumpy(B._bluestein_2d(
                    Ec, alpha, alpha, my, mx, sign=-1, xp=cp,
                    fft2=cp.fft.fft2, ifft2=cp.fft.ifft2, method='direct'))
                row['auto_is_dense_bits'] = bool(np.array_equal(got, dense_c))
                ref_np = B._bluestein_2d(
                    E, alpha, alpha, my, mx, sign=-1, xp=np,
                    fft2=np.fft.fft2, ifft2=np.fft.ifft2,
                    method='direct' if says else 'bluestein')
                bar = _bar(np, E, max(ny, nx), max(my, mx), eps)
                row['auto_vs_numpy_max_abs'] = float(
                    np.max(np.abs(got - ref_np)))
                row['derived_bar'] = bar
                row['inside_bar'] = bool(
                    np.max(np.abs(got - ref_np)) < bar)
                row['ran'] = True
            except Exception as exc:                # noqa: BLE001 -- recorded
                row['ran'] = False
                row['error'] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            del E, Ec
        cu['dispatch'] = rows
        cu['n_ran'] = sum(1 for r in rows if r.get('ran'))
        cu['n'] = len(rows)
        cu['n_dense_side_ran'] = sum(1 for r in rows
                                     if r.get('ran') and r['rule_says_direct'])
    except Exception as exc:                        # noqa: BLE001 -- recorded
        cu['premise'] = f"{type(exc).__name__}: {exc}"
    out['backends']['cupy'] = cu

    c4lib.write_json(out, out_path)
    print(f"numpy selection: {sum(out['numpy_selection'])} of "
          f"{len(SHAPES)} shapes take the dense route")
    for name, b in out['backends'].items():
        print(f"{name}: premise={b['premise']!r} "
              f"selection_matches_numpy={b.get('selection_matches_numpy')}")
        if name == 'jax' and b.get('dispatch'):
            print(f"   dispatch as the rule says {b['dispatch_as_rule_says']}"
                  f"/{b['n']}, jit agrees with eager "
                  f"{b['jit_agrees_with_eager']}/{b['n']}, inside the derived "
                  f"bar {b['inside_bar']}/{b['n']}")
        if name == 'cupy' and b.get('dispatch'):
            print(f"   fft premise: {b.get('fft_premise')}")
            print(f"   ran {b['n_ran']}/{b['n']} "
                  f"({b['n_dense_side_ran']} of them dense-side)")
            for r in b['dispatch']:
                print(f"     {r['shape']} says_direct={r['rule_says_direct']} "
                      f"ran={r.get('ran')} "
                      + (f"auto_is_dense_bits={r.get('auto_is_dense_bits')} "
                         f"vs numpy {r.get('auto_vs_numpy_max_abs'):.3e} "
                         f"bar {r.get('derived_bar'):.3e}"
                         if r.get('ran') else r.get('error', '')[:100]))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
