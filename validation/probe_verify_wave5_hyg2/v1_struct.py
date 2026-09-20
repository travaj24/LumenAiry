"""v1: (6) ONE xp-parametrised dense implementation, and (8) the vocabulary
gate -- structural AND dynamic.

Usage::  python v1_struct.py <tree> <out.json>
"""
from __future__ import annotations

import os
import sys
import tracemalloc
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                            # noqa: E402
from vlib import anchor, build_tag, write_json                # noqa: E402

TREE = sys.argv[1]
OUT = sys.argv[2]
anchor(TREE)

import lumenairy.propagators._bluestein as BL                 # noqa: E402
from lumenairy.propagators._bluestein import (                # noqa: E402
    _SUM_METHODS, _bluestein_2d, _bluestein_centred_2d)
from lumenairy.propagators.fft_infra import _fft2, _ifft2     # noqa: E402
from lumenairy.propagators.mft import (                       # noqa: E402
    angular_spectrum_propagate_mft, fraunhofer_propagate_mft,
    fresnel_propagate_mft)

WL = 633e-9
R = {'build': build_tag(), 'tree': TREE}


def rand(ny, nx, seed=5150):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((ny, nx))
            + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)


def bits(a):
    return np.ascontiguousarray(a).view(np.float64)


# ===================== (6a) both primitives call the ONE function ==========
calls = []
_real = BL._direct_matrix_2d


def _spy(*a, **kw):
    calls.append({'n_pos': len(a),
                  'kw': {k: (v if isinstance(v, (int, float, str))
                             else type(v).__name__)
                         for k, v in kw.items()}})
    return _real(*a, **kw)


E = rand(20, 24)
BL._direct_matrix_2d = _spy
try:
    _bluestein_2d(E, 1 / 48., 1 / 48., 12, 10, sign=-1, xp=np, fft2=_fft2,
                  ifft2=_ifft2, method='direct')
    R['spy_bluestein_2d'] = list(calls)
    calls.clear()
    _bluestein_centred_2d(E, 1 / 48., 1 / 48., 12, 10, sign=-1, xp=np,
                          fft2=_fft2, ifft2=_ifft2, method='direct',
                          n_centre_in_x=3.5, n_centre_in_y=2.25,
                          k_centre_out_x=1.5, k_centre_out_y=-0.75)
    R['spy_bluestein_centred_2d'] = list(calls)
    calls.clear()
    # and the centred one at its DEFAULTS (what the propagators use)
    _bluestein_centred_2d(E, 1 / 48., 1 / 48., 12, 10, sign=-1, xp=np,
                          fft2=_fft2, ifft2=_ifft2, method='direct')
    R['spy_centred_defaults'] = list(calls)
    calls.clear()
    # the chirp arms must NOT reach it
    _bluestein_2d(E, 1 / 48., 1 / 48., 12, 10, sign=-1, xp=np, fft2=_fft2,
                  ifft2=_ifft2, method='auto')
    _bluestein_centred_2d(E, 1 / 48., 1 / 48., 12, 10, sign=-1, xp=np,
                          fft2=_fft2, ifft2=_ifft2, method='separable')
    R['spy_chirp_arms_calls'] = len(calls)
finally:
    BL._direct_matrix_2d = _real

# ===================== (6b) zero centres == non-centred, BIT for BIT =======
rows = {}
for (ny, nx, my, mx, ax_, ay_, sgn) in ((20, 24, 12, 10, 1 / 48., 1 / 48., -1),
                                        (17, 9, 5, 23, .013, .011, +1),
                                        (32, 32, 16, 16, .02, .03, -1)):
    Ei = rand(ny, nx, seed=ny * 100 + nx)
    a = _bluestein_2d(Ei, ax_, ay_, my, mx, sign=sgn, xp=np, fft2=_fft2,
                      ifft2=_ifft2, method='direct')
    b = _bluestein_centred_2d(Ei, ax_, ay_, my, mx, sign=sgn, xp=np,
                              fft2=_fft2, ifft2=_ifft2, method='direct',
                              n_centre_in_x=0.0, n_centre_in_y=0.0,
                              k_centre_out_x=0.0, k_centre_out_y=0.0)
    rows[f'{ny}x{nx}->{my}x{mx}.sign{sgn}'] = {
        'bit_identical': bool(np.array_equal(bits(a), bits(b))),
        'max_abs_diff': float(np.max(np.abs(a - b)))}
R['zero_centre_bit_identity'] = rows

# ===================== (8) vocabulary gate =================================
BAD = ('Direct', 'DIRECT', 'dense', 'direct ', ' direct', '', None, 1, 3.0,
       b'direct', ('direct',), 'directt', 'auto ', 'AUTO')
gate = {}
Ev = rand(8, 8)
for bad in BAD:
    row = {}
    for name, call in (
        ('b2d', lambda m: _bluestein_2d(Ev, .01, .01, 4, 4, sign=-1, xp=np,
                                        fft2=_fft2, ifft2=_ifft2, method=m)),
        ('bc2d', lambda m: _bluestein_centred_2d(
            Ev, .01, .01, 4, 4, sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2,
            method=m)),
    ):
        try:
            call(bad)
            row[name] = 'NO RAISE'
        except BaseException as exc:                          # noqa: BLE001
            row[name] = f'{type(exc).__name__}: {str(exc)[:70]}'
    gate[repr(bad)] = row
R['primitive_gate'] = gate

Eg = np.ones((32, 32), dtype=np.complex128)
nat = WL * 2e-2 / (32 * 8e-6)
pub = {}
for fn in (fresnel_propagate_mft, fraunhofer_propagate_mft,
           angular_spectrum_propagate_mft):
    row = {}
    for bad in ('Direct', 'dense', '', None, 1, 'auto '):
        try:
            with warnings.catch_warnings(record=True) as cw:
                warnings.simplefilter('always')
                fn(Eg, 2e-2, WL, 8e-6, nat, 32, method=bad)
            row[repr(bad)] = 'NO RAISE'
        except BaseException as exc:                          # noqa: BLE001
            row[repr(bad)] = (f'{type(exc).__name__}: {str(exc)[:60]} '
                              f'| warnings_first={[w.category.__name__ for w in cw]}')
    pub[fn.__name__] = row
R['public_gate'] = pub

# --- does the public entry point validate BEFORE expensive work? ----------
# Peak traced memory of a FAILING call vs a SUCCEEDING one, N = 512.
Nbig = 512
Eb = np.ones((Nbig, Nbig), dtype=np.complex128)
natb = WL * 2e-2 / (Nbig * 8e-6)
cost = {}
for fn in (fresnel_propagate_mft, fraunhofer_propagate_mft,
           angular_spectrum_propagate_mft):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tracemalloc.start()
        tracemalloc.reset_peak()
        try:
            fn(Eb, 2e-2, WL, 8e-6, natb, 64, method='bogus')
        except ValueError:
            pass
        _, peak_fail = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tracemalloc.start()
        tracemalloc.reset_peak()
        out = fn(Eb, 2e-2, WL, 8e-6, natb, 64)
        _, peak_ok = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del out
    cost[fn.__name__] = {'peak_MB_on_invalid_method': peak_fail / 1e6,
                         'peak_MB_on_valid_call': peak_ok / 1e6,
                         'fraction_of_valid_work_done':
                             peak_fail / max(peak_ok, 1.0)}
R['public_gate_cost'] = cost

# --- ORDER of validation inside the two primitives -------------------------
order = {}
# invalid sign AND invalid method -> which error?
try:
    _bluestein_2d(Ev, .01, .01, 4, 4, sign=0, xp=np, fft2=_fft2,
                  ifft2=_ifft2, method='bogus')
    order['b2d.sign0_method_bogus'] = 'NO RAISE'
except BaseException as exc:                                  # noqa: BLE001
    order['b2d.sign0_method_bogus'] = f'{type(exc).__name__}: {exc}'
try:
    _bluestein_centred_2d(Ev, .01, .01, 4, 4, sign=0, xp=np, fft2=_fft2,
                          ifft2=_ifft2, method='bogus')
    order['bc2d.sign0_method_bogus'] = 'NO RAISE'
except BaseException as exc:                                  # noqa: BLE001
    order['bc2d.sign0_method_bogus'] = f'{type(exc).__name__}: {exc}'
# invalid N_out AND invalid method
try:
    _bluestein_2d(Ev, .01, .01, 0, 4, sign=-1, xp=np, fft2=_fft2,
                  ifft2=_ifft2, method='bogus')
    order['b2d.nout0_method_bogus'] = 'NO RAISE'
except BaseException as exc:                                  # noqa: BLE001
    order['b2d.nout0_method_bogus'] = f'{type(exc).__name__}: {exc}'
try:
    _bluestein_centred_2d(Ev, .01, .01, 0, 4, sign=-1, xp=np, fft2=_fft2,
                          ifft2=_ifft2, method='bogus')
    order['bc2d.nout0_method_bogus'] = f'NO RAISE (returned)'
except BaseException as exc:                                  # noqa: BLE001
    order['bc2d.nout0_method_bogus'] = f'{type(exc).__name__}: {exc}'
# a phase budget that WOULD warn, with an invalid method: does it warn first?
with warnings.catch_warnings(record=True) as cw:
    warnings.simplefilter('always')
    try:
        _bluestein_2d(rand(24, 24), 1e17 / 576., 1e17 / 576., 12, 12,
                      sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2,
                      method='bogus')
        order['b2d.warnbudget_method_bogus'] = 'NO RAISE'
    except BaseException as exc:                              # noqa: BLE001
        order['b2d.warnbudget_method_bogus'] = (
            f'{type(exc).__name__} with '
            f'{[w.category.__name__ for w in cw]} emitted first')
# non-array E: does ``E.shape`` (read BEFORE the method check in the centred
# primitive) mask the vocabulary error?
for nm, fnc in (('b2d', _bluestein_2d), ('bc2d', _bluestein_centred_2d)):
    try:
        fnc([[1 + 0j, 2 + 0j]], .01, .01, 2, 2, sign=-1, xp=np, fft2=_fft2,
            ifft2=_ifft2, method='bogus')
        order[f'{nm}.listE_method_bogus'] = 'NO RAISE'
    except BaseException as exc:                              # noqa: BLE001
        order[f'{nm}.listE_method_bogus'] = f'{type(exc).__name__}: {str(exc)[:90]}'
# and with a VALID method, for contrast
for nm, fnc in (('b2d', _bluestein_2d), ('bc2d', _bluestein_centred_2d)):
    try:
        fnc([[1 + 0j, 2 + 0j]], .01, .01, 2, 2, sign=-1, xp=np, fft2=_fft2,
            ifft2=_ifft2, method='direct')
        order[f'{nm}.listE_method_direct'] = 'NO RAISE'
    except BaseException as exc:                              # noqa: BLE001
        order[f'{nm}.listE_method_direct'] = f'{type(exc).__name__}: {str(exc)[:90]}'
R['validation_order'] = order
R['SUM_METHODS'] = list(_SUM_METHODS)

# --- every value the message names really works ---------------------------
works = {}
for m in ('auto',) + tuple(_SUM_METHODS):
    try:
        o1 = _bluestein_2d(Ev, .01, .01, 4, 4, sign=-1, xp=np, fft2=_fft2,
                           ifft2=_ifft2, method=m)
        o2 = _bluestein_centred_2d(Ev, .01, .01, 4, 4, sign=-1, xp=np,
                                   fft2=_fft2, ifft2=_ifft2, method=m)
        works[m] = [str(o1.shape), str(o2.shape)]
    except BaseException as exc:                              # noqa: BLE001
        works[m] = f'{type(exc).__name__}: {exc}'
R['vocabulary_round_trip'] = works

# --- JAX arm of the ONE implementation ------------------------------------
try:
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    Ej = rand(24, 20)
    Fn = BL._direct_matrix_2d(Ej, .013, .011, 12, 10, sign=-1, xp=np)
    Fj = np.asarray(BL._direct_matrix_2d(jnp.asarray(Ej), .013, .011, 12, 10,
                                         sign=-1, xp=jnp))
    R['jax'] = {'version': jax.__version__,
                'rel_L2': float(np.linalg.norm(Fj - Fn)
                                / np.linalg.norm(Fn)),
                'dtype': str(Fj.dtype),
                'bit_identical': bool(np.array_equal(bits(Fn), bits(Fj)))}
    jf = jax.jit(lambda e: BL._direct_matrix_2d(e, .013, .011, 12, 10,
                                                sign=-1, xp=jnp))
    Fjit = np.asarray(jf(jnp.asarray(Ej)))
    R['jax']['jit_rel_L2'] = float(np.linalg.norm(Fjit - Fn)
                                   / np.linalg.norm(Fn))
except Exception as exc:                                      # noqa: BLE001
    R['jax'] = f'unavailable: {type(exc).__name__}: {str(exc)[:160]}'

write_json(R, OUT)
