"""(3) MEASURE the 'three helpers needed nothing' claim.

The author states that giving ``_collins_space_support`` /
``_collins_angle_support`` an ``xp`` and reducing ON-DEVICE "would change the
NumPy summation order and break the bit-identity contract".  On the NumPy
path an on-device reduction IS ``np.sum(|A|^2, axis=...)`` over the whole
array, so the claim is directly testable against the shipped banded host
accumulation.

Also checks whether ``_collins_sampling_stats`` really takes only Python
floats (signature, returned types, and an AST scan of its body).

    python v2_marginals.py <tree> <out.json>
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                            # noqa: E402
from vlib import anchor, build_tag, write_json                 # noqa: E402


def ulp_spread(a, b):
    """Max |ULP| difference between two float64 arrays."""
    ai = np.asarray(a, dtype=np.float64).view(np.int64)
    bi = np.asarray(b, dtype=np.float64).view(np.int64)
    ai = np.where(ai < 0, np.int64(np.iinfo(np.int64).min) - ai, ai)
    bi = np.where(bi < 0, np.int64(np.iinfo(np.int64).min) - bi, bi)
    return int(np.max(np.abs(ai - bi)))


def banded_vs_whole(CA, A):
    """(a) the shipped banded host accumulation, (b) one whole-array sum."""
    Px_b, Py_b = CA._collins_power_marginals(A)
    M = np.abs(np.asarray(A)) ** 2
    Px_w = M.sum(axis=0)
    Py_w = M.sum(axis=1)
    return (Px_b, Py_b), (Px_w, Py_w)


def compare(tag, banded, whole):
    (Px_b, Py_b), (Px_w, Py_w) = banded, whole
    out = {}
    for name, b, w in (('Px', Px_b, Px_w), ('Py', Py_b, Py_w)):
        bb = np.ascontiguousarray(b, dtype=np.float64)
        ww = np.ascontiguousarray(w, dtype=np.float64)
        ident = bool(np.array_equal(bb.view(np.uint8), ww.view(np.uint8)))
        maxabs = float(np.max(np.abs(bb - ww)))
        rel = float(np.max(np.abs(bb - ww) / np.maximum(np.abs(ww), 1e-300)))
        out[name] = {'bit_identical': ident, 'max_abs': maxabs,
                     'max_rel': rel, 'max_ulp': ulp_spread(bb, ww)}
    return {tag: out}


def main():
    tree, out_path = sys.argv[1], sys.argv[2]
    anchor(tree)
    import lumenairy.propagators.carrier as CA
    res = {'build': build_tag(), 'tree': tree,
           'PHASOR_BAND_BYTES': float(CA._PHASOR_BAND_BYTES)}

    # --- how many bands does the shipped loop actually take? --------------
    bands = {}
    for nx in (48, 64, 256, 1024, 2048, 4096, 8192):
        rows = max(1, int(CA._PHASOR_BAND_BYTES // max(8 * nx, 1)))
        bands['nx=%d' % nx] = {'rows_per_band': rows,
                               'bands_at_ny=nx': int(np.ceil(nx / rows))}
    res['band_arithmetic'] = bands

    # --- fixture 1: an ordinary Collins fixture (ONE band) ----------------
    rng = np.random.default_rng(20260919)
    n = 64
    ax = (np.arange(n) - n // 2) * 8e-6
    X, Y = np.meshgrid(ax, ax)
    env = (np.exp(-(X ** 2 + Y ** 2) / (55e-6) ** 2)
           * np.exp(1j * 7.0 * (X + Y) / 8e-6)).astype(np.complex128)
    res.update(compare('f1_gauss_64_one_band', *banded_vs_whole(CA, env)))
    res['f1_bands'] = int(np.ceil(
        n / max(1, int(CA._PHASOR_BAND_BYTES // max(8 * n, 1)))))

    # a spectrum (what _collins_angle_support actually reduces)
    from lumenairy.propagators.fft_infra import _fft2
    S = _fft2(np.ascontiguousarray(env, dtype=np.complex128))
    res.update(compare('f1_spectrum_64', *banded_vs_whole(CA, S)))

    # a wide-dynamic-range random field -- the worst case for cancellation
    big = (rng.standard_normal((n, n)) * 10.0 ** rng.uniform(-8, 8, (n, n))
           + 1j * rng.standard_normal((n, n))).astype(np.complex128)
    res.update(compare('f1_random_wide_dynamic_64', *banded_vs_whole(CA, big)))

    # --- fixture 2: FORCE more than one band ------------------------------
    # nx = 4096 gives rows = 976 -> several bands at ny = 4096, which is the
    # only regime in which the banding can move a bit at all.
    nx = 4096
    ny = 3 * max(1, int(CA._PHASOR_BAND_BYTES // max(8 * nx, 1))) + 7
    rowsb = max(1, int(CA._PHASOR_BAND_BYTES // max(8 * nx, 1)))
    wide = (rng.standard_normal((ny, nx))
            * 10.0 ** rng.uniform(-6, 6, (ny, nx))).astype(np.complex128)
    res['f2_shape'] = [ny, nx]
    res['f2_rows_per_band'] = rowsb
    res['f2_bands'] = int(np.ceil(ny / rowsb))
    res.update(compare('f2_multiband_4096', *banded_vs_whole(CA, wide)))
    del wide

    # --- and a shrunken band, to isolate the ORDER effect alone -----------
    # Same array, band size forced small: this is the only thing an on-device
    # whole-array reduction changes.
    keep = CA._PHASOR_BAND_BYTES
    try:
        CA._PHASOR_BAND_BYTES = 8.0 * 64 * 4      # 4 rows per band at nx=64
        res['f3_rows_per_band'] = max(
            1, int(CA._PHASOR_BAND_BYTES // max(8 * n, 1)))
        res.update(compare('f3_forced_4row_bands_64',
                           *banded_vs_whole(CA, big)))
        res.update(compare('f3_forced_4row_bands_gauss',
                           *banded_vs_whole(CA, env)))
    finally:
        CA._PHASOR_BAND_BYTES = keep

    # --- does the DOWNSTREAM reading move at all? -------------------------
    # _collins_space_support is what the guard actually reads.
    def support_with(bandbytes, A):
        k = CA._PHASOR_BAND_BYTES
        try:
            CA._PHASOR_BAND_BYTES = bandbytes
            return CA._collins_space_support(A, 8e-6, 8e-6, 1e-3)
        finally:
            CA._PHASOR_BAND_BYTES = k
    res['f4_support_shipped'] = support_with(keep, big)
    res['f4_support_4row_bands'] = support_with(8.0 * 64 * 4, big)
    res['f4_support_identical'] = (res['f4_support_shipped']
                                   == res['f4_support_4row_bands'])

    # --- _collins_sampling_stats: only Python floats? ---------------------
    sig = inspect.signature(CA._collins_sampling_stats)
    res['stats_signature'] = str(sig)
    res['stats_param_kinds'] = {k: str(v.kind)
                                for k, v in sig.parameters.items()}
    st = CA._collins_sampling_stats(0.84, 3.1e-3, -27.0, 1.0, 8e-6, 8e-6,
                                    1.6e-4, 1.6e-4, 1.1e-2, 1.1e-2, 8e-6,
                                    8e-6, 64, 64, (0.0, 0.0), 633e-9)
    res['stats_return_types'] = {k: type(v).__name__ for k, v in st.items()}
    res['stats_all_python_scalars'] = all(
        isinstance(v, (float, int, str, tuple)) for v in st.values())
    # AST of the body: which np.* attributes does it touch?
    src = inspect.getsource(CA._collins_sampling_stats)
    body = ast.parse(src.lstrip())
    np_uses = sorted({
        ast.unparse(node) for node in ast.walk(body)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name) and node.value.id == 'np'})
    res['stats_np_attribute_uses'] = np_uses
    calls = sorted({node.func.id for node in ast.walk(body)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)})
    res['stats_bare_calls'] = calls
    # behaviour when handed an ARRAY (does it silently vectorise?)
    try:
        bad = CA._collins_sampling_stats(
            np.array([0.84, 0.5]), 3.1e-3, -27.0, 1.0, 8e-6, 8e-6,
            np.array([1.6e-4, 2e-4]), 1.6e-4, 1.1e-2, 1.1e-2, 8e-6, 8e-6,
            64, 64, (0.0, 0.0), 633e-9)
        res['stats_array_input'] = {
            'raised': False,
            'k1_type': type(bad['k1'][0]).__name__,
            'worst_type': type(bad['worst']).__name__}
    except BaseException as exc:                               # noqa: BLE001
        res['stats_array_input'] = {'raised': True,
                                    'type': type(exc).__name__,
                                    'msg': str(exc)[:200]}

    write_json(res, out_path)
    print(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main()
