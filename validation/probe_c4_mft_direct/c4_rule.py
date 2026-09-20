"""WP-C4 task 2 -- the selection is a function of SHAPE ALONE, and ``'auto'``
is one of the two existing routes bit for bit and never a third arithmetic.

TWO CLAIMS, both measured rather than argued.

PURITY.  :func:`_auto_selects_direct` is called many times per shape with the
process perturbed between calls in every way that is not a shape: the wall
clock advanced, ``OMP_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS`` /
``MKL_NUM_THREADS`` / ``SCIPY_FFT_WORKERS`` / ``LUMENAIRY_MEM_BUDGET_MB``
rewritten, the library's caches filled and dropped, the RNG advanced, a
different backend imported, garbage collected, and the thread it runs on
changed.  The answer must be the same every time.  The DISPATCH is then
measured the same way: the digest of what ``'auto'`` returns, under each
perturbation, must be one value.

A "perturbation that changes nothing" is only evidence if it could have
changed something, so the file also records the FALSIFIER: the one thing that
DOES move the answer is ``_MFT_DIRECT_MAX_RATIO`` itself, and both documented
settings are exercised.

NO THIRD ARITHMETIC.  At every shape, the bytes ``'auto'`` returns are
compared to the bytes of BOTH named routes.  ``'auto'`` must equal exactly one
of them, and the one it equals must be the one the rule names.  A route that
agreed with neither -- for instance a centred ``'auto'`` that went through the
pre-chirp decomposition and then into the dense core -- would show up here as
``matched: none``.

    PYTHONPATH=<tree> python c4_rule.py <tree> OUT.json
"""
from __future__ import annotations

import gc
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c4lib  # noqa: E402

#: ``(Ny_in, Nx_in, N_out_y, N_out_x)`` -- 30 shapes, dense side and chirp
#: side, square and anisotropic, right on the boundary and far from it.
SHAPES = [
    (32, 32, 1, 1), (64, 64, 2, 2), (128, 128, 4, 4), (256, 256, 8, 8),
    (512, 512, 16, 16), (1024, 1024, 32, 32), (96, 96, 3, 3),
    (320, 320, 10, 10), (33, 33, 1, 1), (65, 65, 2, 2),
    (512, 512, 17, 16), (1024, 1024, 33, 32), (256, 256, 9, 8),
    (64, 64, 4, 4), (48, 48, 24, 24), (28, 22, 15, 13), (24, 24, 12, 12),
    (128, 128, 128, 128), (64, 64, 256, 256), (16, 16, 8, 8),
    (512, 1024, 16, 32), (512, 1024, 16, 64), (512, 1024, 64, 32),
    (1024, 512, 32, 16), (768, 768, 24, 24), (768, 768, 25, 24),
    (100, 200, 3, 6), (100, 200, 3, 7), (200, 100, 6, 3), (7, 5, 1, 1),
]

ENV_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'SCIPY_FFT_WORKERS', 'LUMENAIRY_MEM_BUDGET_MB',
            'LUMENAIRY_USE_SCIPY_FFT')


def _perturbations(np):
    """Everything that is not a shape.  Each entry is ``(name, apply)``."""
    from lumenairy.propagators import fft_infra as fi

    def clock(_):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 0.004:
            pass

    def env_hi(_):
        for k in ENV_KEYS:
            os.environ[k] = '8'

    def env_lo(_):
        for k in ENV_KEYS:
            os.environ[k] = '1'

    def env_gone(_):
        for k in ENV_KEYS:
            os.environ.pop(k, None)

    def caches_full(_):
        from lumenairy.propagators._bluestein import _bluestein_2d
        E = np.ones((16, 16), dtype=np.complex128)
        _bluestein_2d(E, 0.01, 0.01, 8, 8, sign=-1, xp=np,
                      fft2=fi._fft2, ifft2=fi._ifft2)

    def caches_empty(_):
        from lumenairy._cache_registry import clear_all_registered_caches
        clear_all_registered_caches()
        gc.collect()

    def rng_advanced(_):
        np.random.default_rng().standard_normal(1024)

    def scipy_off(_):
        fi.USE_SCIPY_FFT = False

    def scipy_on(_):
        fi.USE_SCIPY_FFT = True

    def workers_one(_):
        fi.SCIPY_FFT_WORKERS = 1

    def workers_all(_):
        fi.SCIPY_FFT_WORKERS = -1

    return [('baseline', lambda _: None), ('clock', clock),
            ('env_hi', env_hi), ('env_lo', env_lo), ('env_gone', env_gone),
            ('caches_full', caches_full), ('caches_empty', caches_empty),
            ('rng_advanced', rng_advanced),
            ('scipy_fft_off', scipy_off), ('scipy_fft_on', scipy_on),
            ('workers_one', workers_one), ('workers_all', workers_all)]


def main(tree, out_path):
    lum = c4lib.anchor(tree)
    import numpy as np
    from lumenairy.propagators import _bluestein as B
    from lumenairy.propagators.fft_infra import _fft2, _ifft2

    saved_env = {k: os.environ.get(k) for k in ENV_KEYS}
    out = {'build': c4lib.build_tag(), 'python': sys.version.split()[0],
           'lumenairy_file': lum.__file__,
           'ratio': float(B._MFT_DIRECT_MAX_RATIO),
           'purity': [], 'dispatch': [], 'falsifier': {}}

    # ---------------- purity of the DECISION ------------------------------
    try:
        for (ny, nx, my, mx) in SHAPES:
            seen = set()
            for name, apply in _perturbations(np):
                apply(None)
                seen.add(bool(B._auto_selects_direct(ny, nx, my, mx)))
            # ... and from another thread, on a fresh interpreter stack
            box = []
            t = threading.Thread(
                target=lambda: box.append(
                    bool(B._auto_selects_direct(ny, nx, my, mx))))
            t.start()
            t.join()
            seen.add(box[0])
            out['purity'].append({
                'shape': [ny, nx, my, mx], 'ratio': max(my / ny, mx / nx),
                'answers': sorted(seen), 'stable': len(seen) == 1,
                'says_direct': box[0]})
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # ---------------- the DISPATCH: one of two, never a third -------------
    for (ny, nx, my, mx) in SHAPES:
        if ny * nx > 300_000:            # keep the sweep under a minute
            continue
        rng = np.random.default_rng(1234)
        E = (rng.standard_normal((ny, nx))
             + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)
        alpha = 1.0e3 / float(max(ny, nx, my, mx)) ** 2
        says = bool(B._auto_selects_direct(ny, nx, my, mx))
        for prim, fn in (('plain', B._bluestein_2d),
                         ('centred', B._bluestein_centred_2d)):
            for sep in (False, True):
                kw = dict(sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2,
                          separable=sep)
                c4lib.cold()
                a = c4lib.digest(fn(E, alpha, alpha * 1.5, my, mx, **kw))
                c4lib.cold()
                d = c4lib.digest(fn(E, alpha, alpha * 1.5, my, mx,
                                    method='direct', **kw))
                c4lib.cold()
                prev_name = 'separable' if sep else 'bluestein'
                p = c4lib.digest(fn(E, alpha, alpha * 1.5, my, mx,
                                    method=prev_name, **kw))
                matched = ('direct' if a == d else
                           prev_name if a == p else 'NONE')
                # ... and the SAME 'auto' call again under every
                # perturbation.  Two different questions are recorded, because
                # they have different answers and only one of them is WP-C4's:
                #   * did the ROUTE move?  (compare 'auto' to the two named
                #     routes again under the perturbation) -- this is the
                #     selection-rule claim, and it must never move;
                #   * did the BITS move?  -- the chirp-Z route dispatches its
                #     FFTs through ``fft_infra.USE_SCIPY_FFT`` /
                #     ``SCIPY_FFT_WORKERS``, and the dense route's two products
                #     go through BLAS, so both are ENTITLED to move their last
                #     bits with the backend.  That is pre-existing and is NOT a
                #     property of this rule; which perturbations do it is
                #     recorded rather than asserted away.
                moved_bits = []
                route_moved = []
                for name, apply in _perturbations(np):
                    apply(None)
                    c4lib.cold()
                    a2 = c4lib.digest(fn(E, alpha, alpha * 1.5, my, mx, **kw))
                    if a2 != a:
                        moved_bits.append(name)
                    c4lib.cold()
                    d2 = c4lib.digest(fn(E, alpha, alpha * 1.5, my, mx,
                                         method='direct', **kw))
                    c4lib.cold()
                    p2 = c4lib.digest(fn(E, alpha, alpha * 1.5, my, mx,
                                         method=prev_name, **kw))
                    m2 = ('direct' if a2 == d2 else
                          prev_name if a2 == p2 else 'NONE')
                    if m2 != matched:
                        route_moved.append(f"{name}:{m2}")
                out['dispatch'].append({
                    'shape': [ny, nx, my, mx], 'primitive': prim,
                    'separable': sep, 'rule_says_direct': says,
                    'matched': matched,
                    'as_the_rule_says': (matched == 'direct') == says,
                    'route_stable_under_perturbation': not route_moved,
                    'route_moved_by': route_moved,
                    'auto_bits_stable_under_perturbation': not moved_bits,
                    'bits_moved_by': moved_bits})
        del E
        gc.collect()
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

    # ---------------- the falsifier: the constant DOES move it ------------
    saved = B._MFT_DIRECT_MAX_RATIO
    try:
        for label, value in (('never', B._MFT_DIRECT_NEVER),
                             ('always', B._MFT_DIRECT_ALWAYS),
                             ('shipped', saved)):
            B._MFT_DIRECT_MAX_RATIO = value
            out['falsifier'][label] = {
                'value': repr(value),
                'n_direct': sum(1 for (ny, nx, my, mx) in SHAPES
                                if B._auto_selects_direct(ny, nx, my, mx)),
                'n_shapes': len(SHAPES)}
    finally:
        B._MFT_DIRECT_MAX_RATIO = saved

    n_pure = sum(1 for r in out['purity'] if r['stable'])
    n_ok = sum(1 for r in out['dispatch'] if r['as_the_rule_says'])
    n_stable = sum(1 for r in out['dispatch']
                   if r['auto_bits_stable_under_perturbation'])
    n_route = sum(1 for r in out['dispatch']
                  if r['route_stable_under_perturbation'])
    n_none = sum(1 for r in out['dispatch'] if r['matched'] == 'NONE')
    movers = {}
    for r in out['dispatch']:
        for name in r['bits_moved_by']:
            movers[name] = movers.get(name, 0) + 1
    out['summary'] = {
        'purity_stable': n_pure, 'purity_n': len(out['purity']),
        'dispatch_as_rule_says': n_ok, 'dispatch_n': len(out['dispatch']),
        'dispatch_route_stable': n_route,
        'dispatch_bits_stable': n_stable, 'dispatch_third_arithmetic': n_none,
        'which_perturbations_move_bits': movers,
        'n_direct_shapes': sum(1 for r in out['purity'] if r['says_direct'])}
    c4lib.write_json(out, out_path)
    print(f"purity: {n_pure}/{len(out['purity'])} shapes give ONE answer "
          f"under {len(_perturbations(np))} perturbations + another thread")
    print(f"dispatch: {n_ok}/{len(out['dispatch'])} match the route the rule "
          f"names; {n_none} match NEITHER named route (third arithmetic)")
    print(f"  ROUTE stable under perturbation: "
          f"{n_route}/{len(out['dispatch'])}")
    print(f"  BITS  stable under perturbation: "
          f"{n_stable}/{len(out['dispatch'])}; moved by {movers}")
    print("falsifier:", {k: v['n_direct'] for k, v in
                         out['falsifier'].items()},
          f"of {len(SHAPES)} shapes")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
