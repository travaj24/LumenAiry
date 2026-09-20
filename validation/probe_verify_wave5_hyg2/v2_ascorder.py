"""(8) _as_c_order: the jax.numpy gap, the CuPy dtype= call, byte identity.

    python v2_ascorder.py <tree> <out.json>
"""
from __future__ import annotations

import inspect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                            # noqa: E402
from vlib import anchor, build_tag, write_json                 # noqa: E402


def main():
    tree, out_path = sys.argv[1], sys.argv[2]
    anchor(tree)
    import lumenairy.propagators.carrier as CA
    res = {'build': build_tag(), 'tree': tree}

    # ---- jax.numpy has no ascontiguousarray? ----------------------------
    try:
        import jax
        import jax.numpy as jnp
        res['jax_version'] = jax.__version__
        res['jnp_has_ascontiguousarray'] = hasattr(jnp, 'ascontiguousarray')
        res['jnp_dir_matches'] = sorted(
            n for n in dir(jnp) if 'contig' in n.lower())
        res['jnp_has_asarray'] = hasattr(jnp, 'asarray')
        jax.config.update('jax_enable_x64', True)
        a = jnp.asarray(np.arange(12.0).reshape(3, 4))
        out = CA._as_c_order(a, np.complex128, jnp)
        res['jax_as_c_order'] = {'dtype': str(out.dtype),
                                 'type': type(out).__name__,
                                 'took_asarray_branch':
                                     not hasattr(jnp, 'ascontiguousarray')}
        # the transposed (non-C) case: JAX has no strides to fix
        res['jax_transposed_ok'] = str(
            CA._as_c_order(a.T, np.complex128, jnp).dtype)
    except ImportError as exc:
        res['jax_version'] = 'unavailable: %s' % exc

    # ---- CuPy accepts dtype= ? -------------------------------------------
    cup = {}
    try:
        import cupy as cp
        cup['version'] = cp.__version__
        cup['has_ascontiguousarray'] = hasattr(cp, 'ascontiguousarray')
        cup['signature'] = str(inspect.signature(cp.ascontiguousarray)) \
            if hasattr(cp, 'ascontiguousarray') else None
        try:
            a = cp.asarray(np.arange(12.0).reshape(3, 4))
            o = cp.ascontiguousarray(a, dtype=cp.complex128)
            cup['dtype_kwarg_accepted'] = True
            cup['result_dtype'] = str(o.dtype)
            cup['c_contiguous'] = bool(o.flags.c_contiguous)
            o2 = CA._as_c_order(a.T, np.complex128, cp)
            cup['as_c_order_dtype'] = str(o2.dtype)
            cup['as_c_order_c_contiguous'] = bool(o2.flags.c_contiguous)
            cup['matches_cupy_asnumpy'] = bool(np.array_equal(
                cp.asnumpy(o2),
                np.ascontiguousarray(np.arange(12.0).reshape(3, 4).T,
                                     dtype=np.complex128)))
        except BaseException as exc:                        # noqa: BLE001
            cup['dtype_kwarg_accepted'] = False
            cup['error'] = '%s: %s' % (type(exc).__name__, str(exc)[:200])
    except ImportError as exc:
        cup['available'] = False
        cup['error'] = str(exc)[:120]
    res['cupy'] = cup

    # ---- NumPy path BYTE-identical through it ----------------------------
    rng = np.random.default_rng(7)
    cases = {}
    base = (rng.standard_normal((37, 41))
            + 1j * rng.standard_normal((37, 41)))
    inputs = {
        'c_contiguous_c128': np.ascontiguousarray(base),
        'f_order_c128': np.asfortranarray(base),
        'transposed_c128': base.T,
        'sliced_c128': base[::2, ::3],
        'real_f64': np.real(base).copy(),
        'c64': base.astype(np.complex64),
        'int32': (np.real(base) * 100).astype(np.int32),
        'scalar_like': np.array(3.5),
    }
    for name, arr in inputs.items():
        for dt in (np.complex128, np.complex64):
            got = CA._as_c_order(arr, dt, np)
            want = np.ascontiguousarray(arr, dtype=dt)
            cases['%s->%s' % (name, np.dtype(dt).name)] = {
                'byte_identical': bool(
                    np.array_equal(np.ascontiguousarray(got).view(np.uint8),
                                   np.ascontiguousarray(want).view(np.uint8))),
                'dtype_ok': str(got.dtype) == str(want.dtype),
                'c_contiguous': bool(np.asarray(got).flags.c_contiguous),
                'is_same_object_as_input': got is arr,
            }
    res['numpy_byte_identity'] = cases
    res['numpy_all_byte_identical'] = all(
        v['byte_identical'] and v['dtype_ok'] and v['c_contiguous']
        for v in cases.values())

    # the historical spelling the branch replaced, at the two live sites
    res['source_sites'] = []
    src = open(CA.__file__, encoding='cp1252', errors='replace').read()
    for i, ln in enumerate(src.splitlines(), 1):
        if '_as_c_order(' in ln:
            res['source_sites'].append('%d: %s' % (i, ln.strip()))

    write_json(res, out_path)
    print(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main()
