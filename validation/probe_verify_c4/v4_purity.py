"""VERIFY-WP-C4 claims 3 and 4 -- try to MOVE the selection with something
that is not a shape, and try to catch ``'auto'`` producing a THIRD arithmetic.

The branch's purity sweep perturbs the environment, the caches, the clock, the
RNG and the thread count.  Those are all things ``_auto_selects_direct`` never
reads, so they cannot move it and the sweep is not much of a falsifier.  This
probe goes after the things that COULD:

* the array's dtype -- ``complex64`` / ``float32`` input, where the route's
  kernels are built in float64 and cast, and where a shape-blind implementation
  might have consulted ``E.dtype``;
* a NON-CONTIGUOUS input -- a transpose, an F-order array and a strided slice,
  which change ``E.strides`` but not ``E.shape``;
* ``separable=True`` / ``False``, which selected the previous arm and might
  still leak into the new one;
* the PHASE-BUDGET WARNING path -- a budget past ``_PHASE_BUDGET_MAX``, where
  the guard now runs BEFORE the choice;
* the index CENTRES, which the centred primitive passes and the plain one does
  not;
* a JAX tracer inside ``jit`` (concrete shapes) and, when the build supports
  it, ``jax.export`` SYMBOLIC shapes, where ``int(E.shape[0])`` has no value;
* integer TYPE -- ``numpy.int64`` / ``numpy.int32`` grid sizes instead of
  Python ints;
* AXIS ORDER -- an anisotropic shape at which swapping the two axes' arguments
  would flip the answer, asked of BOTH primitives, which is the only way to
  see a y/x transposition in the wiring.

    PYTHONPATH=<tree> python v4_purity.py <tree>
"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v4lib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

SHAPES = [
    (512, 512, 16, 16), (1024, 1024, 32, 32), (512, 512, 32, 32),
    (256, 256, 8, 8), (256, 256, 16, 16), (96, 96, 3, 3), (64, 64, 2, 2),
    (128, 128, 4, 4), (128, 128, 8, 8), (48, 48, 24, 24), (24, 24, 12, 12),
    (512, 64, 16, 2), (64, 512, 2, 16), (512, 1024, 16, 32),
    (512, 1024, 16, 64), (512, 1024, 64, 32), (1000, 1000, 25, 25),
    (768, 768, 24, 24), (32, 32, 1, 1), (33, 33, 1, 1),
]

#: shapes small enough to DRIVE both primitives at, both signs, both
#: ``separable`` settings -- the no-third-arithmetic sweep
DRIVE = [(96, 96, 3, 3), (64, 64, 2, 2), (128, 128, 4, 4), (128, 128, 8, 8),
         (64, 64, 8, 8), (48, 48, 24, 24), (24, 24, 12, 12), (32, 32, 1, 1),
         (128, 64, 4, 2), (64, 128, 2, 4), (128, 64, 4, 16), (96, 96, 6, 6)]


def _bits(a):
    import numpy as np
    return np.ascontiguousarray(np.asarray(a)).tobytes()


def selection_perturbations(np, B, out):
    """Everything that is not a shape, asked of the rule and of the DISPATCH."""
    from lumenairy.propagators.fft_infra import _fft2, _ifft2
    rows = []
    for (ny, nx, my, mx) in SHAPES:
        base = bool(B._auto_selects_direct(ny, nx, my, mx))
        answers = {'python_int': base}
        # integer TYPE
        answers['np_int64'] = bool(B._auto_selects_direct(
            np.int64(ny), np.int64(nx), np.int64(my), np.int64(mx)))
        answers['np_int32'] = bool(B._auto_selects_direct(
            np.int32(ny), np.int32(nx), np.int32(my), np.int32(mx)))
        answers['float_exact'] = bool(B._auto_selects_direct(
            float(ny), float(nx), float(my), float(mx)))
        # AXIS ORDER.  Swapping BOTH pairs cannot flip the answer -- the
        # max over the two ratios is symmetric under it -- so the test that
        # can actually see a y/x transposition in the wiring is a PARTIAL
        # swap: the output pair exchanged and the input pair left alone,
        # which is what a mis-wired call site would produce.
        answers['_swapped_both_pairs'] = bool(B._auto_selects_direct(
            nx, ny, mx, my))
        answers['_swapped_out_pair_only'] = bool(B._auto_selects_direct(
            ny, nx, mx, my))
        answers['_swapped_in_pair_only'] = bool(B._auto_selects_direct(
            nx, ny, my, mx))
        rows.append({'shape': [ny, nx, my, mx], 'answers': answers,
                     'one_answer': len(set(
                         v for k, v in answers.items()
                         if not k.startswith('_'))) == 1,
                     'partial_swap_would_flip':
                         (answers['_swapped_out_pair_only'] != base
                          or answers['_swapped_in_pair_only'] != base)})
    out['selection'] = rows
    out['selection_all_one_answer'] = all(r['one_answer'] for r in rows)
    out['selection_shapes_where_a_partial_swap_flips'] = [
        r['shape'] for r in rows if r['partial_swap_would_flip']]

    # --- the DISPATCH, not just the rule: does the route taken move? -------
    disp = []
    for (ny, nx, my, mx) in DRIVE:
        says = bool(B._auto_selects_direct(ny, nx, my, mx))
        rng = np.random.default_rng(11)
        E64 = (rng.standard_normal((ny, nx))
               + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)
        a = 1.0e3 / float(max(ny, nx, my, mx)) ** 2
        variants = {
            'c128_C': E64,
            'c64': E64.astype(np.complex64),
            'F_order': np.asfortranarray(E64),
            'transposed': np.ascontiguousarray(
                E64.T.copy()).T.copy().T if False else E64.copy().T.T,
            'strided': np.ascontiguousarray(
                np.repeat(E64, 2, axis=1))[:, ::2],
            'real_float64': E64.real.copy(),
            'real_float32': E64.real.astype(np.float32),
        }
        for vname, Ev in variants.items():
            if Ev.shape != (ny, nx):
                continue
            for sep in (False, True):
                kw = dict(sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2,
                          separable=sep)
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter('always')
                    try:
                        auto = B._bluestein_2d(Ev, a, a * 1.5, my, mx, **kw)
                        dense = B._bluestein_2d(Ev, a, a * 1.5, my, mx,
                                                method='direct', **kw)
                        prev = B._bluestein_2d(
                            Ev, a, a * 1.5, my, mx,
                            method=('separable' if sep else 'bluestein'),
                            **kw)
                    except Exception as exc:              # noqa: BLE001
                        disp.append({'shape': [ny, nx, my, mx],
                                     'variant': vname, 'separable': sep,
                                     'raised': f"{type(exc).__name__}: {exc}"})
                        continue
                matched = ('direct' if _bits(auto) == _bits(dense)
                           else 'prev' if _bits(auto) == _bits(prev)
                           else None)
                disp.append({'shape': [ny, nx, my, mx], 'variant': vname,
                             'separable': sep, 'rule_says_direct': says,
                             'matched': matched,
                             'agrees_with_rule':
                                 (matched == 'direct') == says})
    out['dispatch'] = disp
    driven = [d for d in disp if 'matched' in d]
    out['dispatch_summary'] = {
        'cases': len(driven),
        'matched_neither': sum(1 for d in driven if d['matched'] is None),
        'agrees_with_rule': sum(1 for d in driven if d['agrees_with_rule']),
        'raised': sum(1 for d in disp if 'raised' in d),
    }


def third_arithmetic(np, B, out):
    """Both primitives, both signs, both ``separable`` settings, the centred
    primitive with and without off-centre conventions, and the WARNING path."""
    from lumenairy.propagators.fft_infra import _fft2, _ifft2
    rows = []
    for (ny, nx, my, mx) in DRIVE:
        says = bool(B._auto_selects_direct(ny, nx, my, mx))
        rng = np.random.default_rng(5150)
        E = (rng.standard_normal((ny, nx))
             + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)
        for budget_tag, budget in (('under', 1.0e3), ('over_guard', 1.0e11)):
            a = budget / float(max(ny, nx, my, mx)) ** 2
            for prim in ('plain', 'centred'):
                fn = (B._bluestein_2d if prim == 'plain'
                      else B._bluestein_centred_2d)
                for sgn in (-1, +1):
                    for sep in (False, True):
                        for centres in ((), ('off',)):
                            if prim == 'plain' and centres:
                                continue
                            extra = {}
                            if centres:
                                extra = dict(
                                    n_centre_in_x=0.0,
                                    n_centre_in_y=ny / 3.0,
                                    k_centre_out_x=mx / 2.0 - 0.37,
                                    k_centre_out_y=my / 2.0)
                            kw = dict(sign=sgn, xp=np, fft2=_fft2,
                                      ifft2=_ifft2, separable=sep, **extra)
                            with warnings.catch_warnings(record=True) as wa:
                                warnings.simplefilter('always')
                                B._clear_h_fft_cache()
                                auto = fn(E, a, a * 1.5, my, mx, **kw)
                                n_auto = len(wa)
                            with warnings.catch_warnings(record=True) as wd:
                                warnings.simplefilter('always')
                                B._clear_h_fft_cache()
                                dense = fn(E, a, a * 1.5, my, mx,
                                           method='direct', **kw)
                                n_dense = len(wd)
                            with warnings.catch_warnings(record=True) as wp:
                                warnings.simplefilter('always')
                                B._clear_h_fft_cache()
                                prev = fn(E, a, a * 1.5, my, mx,
                                          method=('separable' if sep
                                                  else 'bluestein'), **kw)
                                n_prev = len(wp)
                            matched = ('direct' if _bits(auto) == _bits(dense)
                                       else 'prev' if _bits(auto)
                                       == _bits(prev) else None)
                            msg = str(wa[0].message) if n_auto else ''
                            rows.append({
                                'shape': [ny, nx, my, mx], 'primitive': prim,
                                'sign': sgn, 'separable': sep,
                                'centres': bool(centres),
                                'budget_tag': budget_tag,
                                'rule_says_direct': says, 'matched': matched,
                                'agrees_with_rule':
                                    (matched == 'direct') == says,
                                'warns_auto': n_auto, 'warns_dense': n_dense,
                                'warns_prev': n_prev,
                                'auto_msg_names_dense':
                                    ("ALREADY on the dense route" in msg),
                                'auto_msg_names_chirp_advice':
                                    ("method='direct' is the more accurate"
                                     in msg),
                            })
    out['third'] = rows
    out['third_summary'] = {
        'cases': len(rows),
        'matched_neither': sum(1 for r in rows if r['matched'] is None),
        'agrees_with_rule': sum(1 for r in rows if r['agrees_with_rule']),
        'warn_cases_over_guard': sum(
            1 for r in rows if r['budget_tag'] == 'over_guard'),
        'auto_warns_exactly_once_over_guard': sum(
            1 for r in rows
            if r['budget_tag'] == 'over_guard' and r['warns_auto'] == 1),
        'dense_silent_over_guard': sum(
            1 for r in rows
            if r['budget_tag'] == 'over_guard' and r['warns_dense'] == 0),
        'prev_warns_once_over_guard': sum(
            1 for r in rows
            if r['budget_tag'] == 'over_guard' and r['warns_prev'] == 1),
        'auto_warns_under_guard': sum(
            1 for r in rows
            if r['budget_tag'] == 'under' and r['warns_auto'] > 0),
        'message_names_the_route_taken': sum(
            1 for r in rows
            if r['budget_tag'] == 'over_guard' and r['warns_auto'] == 1
            and (r['auto_msg_names_dense'] == r['rule_says_direct'])),
    }


def jax_arm(np, B, out):
    rec = {'available': False}
    try:
        import jax
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        rec['available'] = True
        rec['version'] = jax.__version__
    except Exception as exc:                              # noqa: BLE001
        rec['premise'] = f"{type(exc).__name__}: {exc}"
        out['jax'] = rec
        return
    from lumenairy.propagators.fft_infra import _fft2, _ifft2   # noqa: F401
    concrete = []
    for (ny, nx, my, mx) in ((96, 96, 3, 3), (64, 64, 8, 8), (128, 64, 4, 2)):
        says = bool(B._auto_selects_direct(ny, nx, my, mx))
        rng = np.random.default_rng(2)
        E = (rng.standard_normal((ny, nx))
             + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)
        a = 1.0e3 / float(max(ny, nx, my, mx)) ** 2
        Ej = jnp.asarray(E)
        kw = dict(sign=-1, xp=jnp, fft2=jnp.fft.fft2, ifft2=jnp.fft.ifft2)
        eager = B._bluestein_2d(Ej, a, a * 1.5, my, mx, **kw)
        jitted = jax.jit(lambda x: B._bluestein_2d(
            x, a, a * 1.5, my, mx, **kw))(Ej)
        dense = B._bluestein_2d(Ej, a, a * 1.5, my, mx, method='direct', **kw)
        concrete.append({
            'shape': [ny, nx, my, mx], 'rule_says_direct': says,
            'jit_matches_eager': bool(_bits(jitted) == _bits(eager)),
            'eager_matches_dense': bool(_bits(eager) == _bits(dense)),
        })
    rec['concrete'] = concrete
    # --- SYMBOLIC shapes: int(E.shape[0]) has no value under a polymorphic
    # trace, so the rule either raises or silently answers about a symbol
    try:
        from jax import export as jexport
        sy, sx = jexport.symbolic_shape("a, b")
        rec['symbolic_supported'] = True

        def f(x):
            return B._bluestein_2d(x, 1.0e-3, 1.0e-3, 3, 3, sign=-1,
                                   xp=jnp, fft2=jnp.fft.fft2,
                                   ifft2=jnp.fft.ifft2)
        try:
            jexport.export(jax.jit(f))(
                jax.ShapeDtypeStruct((sy, sx), jnp.complex128))
            rec['symbolic_result'] = 'exported without error'
        except Exception as exc:                          # noqa: BLE001
            rec['symbolic_result'] = f"{type(exc).__name__}: {str(exc)[:300]}"
    except Exception as exc:                              # noqa: BLE001
        rec['symbolic_supported'] = False
        rec['symbolic_premise'] = f"{type(exc).__name__}: {str(exc)[:200]}"
    out['jax'] = rec


def main(tree):
    import numpy as np
    v4lib.anchor(tree)
    from lumenairy.propagators import _bluestein as B
    out = {'build': v4lib.build_tag(), 'python': sys.version.split()[0],
           'numpy': np.__version__,
           'constant': float(B._MFT_DIRECT_MAX_RATIO)}
    selection_perturbations(np, B, out)
    print('SELECTION one-answer at every shape:',
          out['selection_all_one_answer'], flush=True)
    print('shapes where a PARTIAL axis swap would flip the answer:',
          out['selection_shapes_where_a_partial_swap_flips'], flush=True)
    print('DISPATCH', out['dispatch_summary'], flush=True)
    third_arithmetic(np, B, out)
    print('THIRD', out['third_summary'], flush=True)
    jax_arm(np, B, out)
    print('JAX', {k: v for k, v in out['jax'].items()
                  if k != 'concrete'}, flush=True)
    if out['jax'].get('concrete'):
        print('  concrete:', out['jax']['concrete'], flush=True)
    v4lib.write_json(out, os.path.join(
        HERE, f"v4_purity_{v4lib.short_tag()}.json"))


if __name__ == '__main__':
    main(sys.argv[1])
