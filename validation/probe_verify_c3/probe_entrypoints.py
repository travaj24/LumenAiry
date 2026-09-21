"""VERIFY-WP-C3 claim 4b -- the WAY-BACK table, measured.

For each PUBLIC entry point that the AST census (``census_callers.py``) shows
reaching a flipped default, this probe answers three questions by RUNNING the
code on one tree:

* does the entry point's answer MOVE (default vs the pre-flip arithmetic)?
* is ``transport='sziklas'`` accepted, and does it RESTORE the answer?
* what happens on the shapes the CHANGELOG / Migration-Guide make claims
  about -- the JAX trace, the stop-plane keys on an explicit
  ``transport='collins'``, ``final_distance = 0``, and ``A == 0``.

    python probe_entrypoints.py <tree> <out.json>

Everything is reported as a MEASUREMENT (outcome, exception type, message
head, array digest, peak, pitch) rather than as a pass/fail, so the two trees'
JSONs are compared by eye and by the driver.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, _TREE)

import numpy as np                                            # noqa: E402

import vlib3                                                  # noqa: E402

vlib3.bind(_TREE)

import lumenairy as la                                        # noqa: E402
import lumenairy.propagators.carrier as CA                    # noqa: E402

WL = 633e-9
WL_IR = 1.31e-6
TKW = dict(on_undersample='silent', on_noncollimated='silent')
OUT = {}


def gauss(n, dx, w, dtype=np.complex128):
    x = (np.arange(n) - n / 2.0) * dx
    return np.exp(-((x[:, None] / w) ** 2 + (x[None, :] / w) ** 2)).astype(
        dtype)


def _plano_convex():
    return {'name': 'pcx', 'aperture_diameter': 16e-3, 'thicknesses': [5e-3],
            'surfaces': [
                {'radius': 52e-3, 'glass_before': 'air',
                 'glass_after': 'N-BK7', 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None},
                {'radius': np.inf, 'glass_before': 'N-BK7',
                 'glass_after': 'air', 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None}]}


def _biconvex():
    return {'name': 'bcx', 'aperture_diameter': 16e-3, 'thicknesses': [4e-3],
            'surfaces': [
                {'radius': 75e-3, 'glass_before': 'air',
                 'glass_after': 'N-BK7', 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None},
                {'radius': -75e-3, 'glass_before': 'N-BK7',
                 'glass_after': 'air', 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None}]}


GROUPS = [{'prescription': _plano_convex(), 'gap_before': 18e-3},
          {'prescription': _biconvex(), 'gap_before': 12e-3}]


def arr_stats(a):
    a = np.ascontiguousarray(np.asarray(a))
    return {'sha': hashlib.sha256(a.tobytes()).hexdigest()[:16],
            'dtype': str(a.dtype), 'shape': list(a.shape),
            'peak': float(np.nanmax(np.abs(a)) ** 2),
            'power': float(np.nansum(np.abs(a) ** 2)),
            'nan': int(np.count_nonzero(~np.isfinite(a)))}


def run(tag, fn, *a, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            val = fn(*a, **kw)
            rec = {'outcome': 'ok', 'type': type(val).__name__}
            try:
                rec['value'] = describe(val)
            except Exception as exc:                          # noqa: BLE001
                rec['value'] = 'undescribable: %r' % (exc,)
        except BaseException as exc:                          # noqa: BLE001
            rec = {'outcome': 'raised', 'type': type(exc).__name__,
                   'message': str(exc)[:400]}
    rec['warnings'] = [(w.category.__name__, str(w.message)[:200])
                       for w in caught]
    OUT[tag] = rec
    return rec


def describe(val):
    if hasattr(val, 'env') and hasattr(val, 'R') and hasattr(val, 'dx'):
        return {'kind': 'CarrierReferencedField', 'env': arr_stats(val.env),
                'R': repr(val.R), 'dx': repr(val.dx)}
    if hasattr(val, 'field') and hasattr(val, 'stages'):
        return {'kind': 'chain', 'field': arr_stats(val.field),
                'R': repr(val.R), 'dx': repr(val.dx),
                'n_stages': len(val.stages),
                'stages_sha': hashlib.sha256(
                    repr(val.stages).encode()).hexdigest()[:16],
                'readout_route': [s.get('readout_route') for s in val.stages
                                  if isinstance(s, dict)
                                  and 'readout_route' in s],
                'readout_route_k1': [s.get('readout_route_k1')
                                     for s in val.stages
                                     if isinstance(s, dict)
                                     and 'readout_route_k1' in s],
                'readout_route_reason': [s.get('readout_route_reason')
                                         for s in val.stages
                                         if isinstance(s, dict)
                                         and 'readout_route_reason' in s]}
    if hasattr(val, 'field') and hasattr(val, 'centre'):
        return {'kind': 'multi', 'field': arr_stats(val.field),
                'dx': repr(val.dx), 'centre': repr(val.centre)}
    if isinstance(val, np.ndarray):
        return arr_stats(val)
    if hasattr(val, 'shape') and hasattr(val, 'dtype'):
        return dict(arr_stats(np.asarray(val)),
                    backend=type(val).__module__.split('.')[0])
    return repr(val)[:400]


# ---------------------------------------------------------------------------
# A.  signature introspection -- which PUBLIC names take ``transport``?
# ---------------------------------------------------------------------------
def sec_signatures():
    import inspect as _ins
    names = ('propagate_carrier_referenced', 'propagate_traced_carrier_chain',
             'propagate_traced_carrier_chain_multi',
             'carrier_referenced_focus_readout',
             'carrier_referenced_exact_focus_readout',
             'carrier_referenced_reconstruct', 'carrier_referenced_envelope',
             'carrier_referenced_fit_radius', 'carrier_referenced_aperture')
    sig = {}
    for n in names:
        fn = getattr(la, n, None)
        if fn is None:
            sig[n] = {'exported': False}
            continue
        p = _ins.signature(fn).parameters
        sig[n] = {'exported': True, 'takes_transport': 'transport' in p,
                  'transport_default': (repr(p['transport'].default)
                                        if 'transport' in p else None)}
    OUT['A-signatures'] = sig
    # and the whole public surface, for the record
    OUT['A-public-names-with-transport'] = sorted(
        n for n in getattr(la, '__all__', [])
        if callable(getattr(la, n, None))
        and _has_transport(getattr(la, n)))


def _has_transport(fn):
    import inspect as _ins
    try:
        return 'transport' in _ins.signature(fn).parameters
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# B.  does each entry point MOVE, and does the keyword restore it?
# ---------------------------------------------------------------------------
def sec_moves():
    # a leg with N dx^2 <= lambda |z_eff| -- the documented moving condition
    E = gauss(64, 8e-6, 60e-6)
    for tag, R, z in (('B1-step-converging', -0.05, 5e-3),
                      ('B1-step-focuscross', -0.02, 0.0205),
                      ('B1-step-collimated', np.inf, 5e-3),
                      ('B1-step-astig', (-0.05, -0.08), 5e-3),
                      ('B1-step-A0-exact', -0.02, 0.02)):
        run(tag + '-default', CA.propagate_carrier_referenced,
            E, R, z, WL, 8e-6)
        run(tag + '-sziklas', CA.propagate_carrier_referenced,
            E, R, z, WL, 8e-6, transport='sziklas')
        run(tag + '-collins', CA.propagate_carrier_referenced,
            E, R, z, WL, 8e-6, transport='collins')

    env = gauss(256, 55e-6, 4.0e-3)
    base = dict(r_in=-70e-3, ray_subsample=16, n_workers=1,
                traced_kwargs=TKW, final_leg='paraxial')
    fr = dict(dx_out=0.4e-6, N_out=64)
    for tag, kw in (('B2-chain-readout',
                     dict(final_distance=9e-3, focus_readout=fr)),
                    ('B2-chain-bare', dict(final_distance=9e-3)),
                    ('B2-chain-readout-zero',
                     dict(final_distance=0.0, focus_readout=fr)),
                    ('B2-chain-readout-standoff',
                     dict(final_distance=9e-3,
                          focus_readout=dict(fr, standoff=2e-3)))):
        run(tag + '-default', CA.propagate_traced_carrier_chain,
            env, GROUPS, WL_IR, 55e-6, **dict(base, **kw))
        run(tag + '-sziklas', CA.propagate_traced_carrier_chain,
            env, GROUPS, WL_IR, 55e-6, transport='sziklas',
            **dict(base, **kw))
        run(tag + '-collins', CA.propagate_traced_carrier_chain,
            env, GROUPS, WL_IR, 55e-6, transport='collins',
            **dict(base, **kw))

    one = [{'field': env, 'carrier': -70e-3}]
    mbase = dict(ray_subsample=16, n_workers=1, traced_kwargs=TKW,
                 final_leg='paraxial', final_distance=9e-3)
    run('B3-multi-K1-default', CA.propagate_traced_carrier_chain_multi,
        one, GROUPS, WL_IR, 55e-6, output_grid=fr, **mbase)
    run('B3-multi-K1-sziklas', CA.propagate_traced_carrier_chain_multi,
        one, GROUPS, WL_IR, 55e-6, output_grid=fr, transport='sziklas',
        **mbase)
    run('B3-multi-K1-collins', CA.propagate_traced_carrier_chain_multi,
        one, GROUPS, WL_IR, 55e-6, output_grid=fr, transport='collins',
        **mbase)
    run('B3-multi-K1-standoff-default',
        CA.propagate_traced_carrier_chain_multi, one, GROUPS, WL_IR, 55e-6,
        output_grid=dict(fr, standoff=2e-3), **mbase)

    # the two PUBLIC readouts: no ``transport`` at all, so the only question
    # is whether they move between the trees.
    Ef = gauss(128, 4e-6, 120e-6)
    run('B4-focus-readout', CA.carrier_referenced_focus_readout,
        Ef, -0.03, 0.03, WL, 4e-6, dx_out=2e-7, N_out=32,
        on_replica='ignore')
    run('B4-focus-readout-standoff', CA.carrier_referenced_focus_readout,
        Ef, -0.03, 0.03, WL, 4e-6, dx_out=2e-7, N_out=32, standoff=1e-3,
        on_replica='ignore')
    run('B4-exact-focus-readout',
        CA.carrier_referenced_exact_focus_readout,
        Ef, -0.03, 0.03, WL, 4e-6, dx_out=2e-7, N_out=32,
        on_replica='ignore')
    # final_leg='exact' -- claimed unmoved on either setting
    run('B5-final-leg-exact-default', CA.propagate_traced_carrier_chain,
        env, GROUPS, WL_IR, 55e-6, r_in=-70e-3, ray_subsample=16,
        n_workers=1, traced_kwargs=TKW, final_leg='exact',
        final_distance=9e-3, focus_readout=fr)
    run('B5-final-leg-exact-sziklas', CA.propagate_traced_carrier_chain,
        env, GROUPS, WL_IR, 55e-6, r_in=-70e-3, ray_subsample=16,
        n_workers=1, traced_kwargs=TKW, final_leg='exact',
        final_distance=9e-3, focus_readout=fr, transport='sziklas')


# ---------------------------------------------------------------------------
# C.  the JAX trace -- the CHANGELOG's ONE documented new refusal
# ---------------------------------------------------------------------------
def sec_jax_trace():
    try:
        import jax
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
    except Exception as exc:                                   # noqa: BLE001
        OUT['C-jax'] = {'available': False, 'why': repr(exc)[:200]}
        return
    OUT['C-jax'] = {'available': True, 'version': jax.__version__}
    amp = np.exp(-((np.arange(64) - 32.0) / 8.0) ** 2)
    A = jnp.asarray(np.outer(amp, amp))

    def merit_default(a):
        out = CA.propagate_carrier_referenced(
            a.astype(jnp.complex128), -0.05, 5e-3, WL, 8e-6)
        return jnp.sum(jnp.abs(out.env) ** 2)

    def merit_sziklas(a):
        out = CA.propagate_carrier_referenced(
            a.astype(jnp.complex128), -0.05, 5e-3, WL, 8e-6,
            transport='sziklas')
        return jnp.sum(jnp.abs(out.env) ** 2)

    run('C1-grad-default', lambda: np.asarray(jax.grad(merit_default)(A)))
    run('C1-grad-sziklas', lambda: np.asarray(jax.grad(merit_sziklas)(A)))
    run('C2-jit-default', lambda: np.asarray(
        jax.jit(lambda a: CA.propagate_carrier_referenced(
            a.astype(jnp.complex128), -0.05, 5e-3, WL, 8e-6).env)(A)))
    run('C2-jit-sziklas', lambda: np.asarray(
        jax.jit(lambda a: CA.propagate_carrier_referenced(
            a.astype(jnp.complex128), -0.05, 5e-3, WL, 8e-6,
            transport='sziklas').env)(A)))
    run('C3-eager-jax-default', lambda: describe(
        CA.propagate_carrier_referenced(
            jnp.asarray(A, dtype=jnp.complex128), -0.05, 5e-3, WL, 8e-6)))


# ---------------------------------------------------------------------------
# D.  the contract shapes the docs make claims about
# ---------------------------------------------------------------------------
def sec_contracts():
    env = gauss(256, 55e-6, 4.0e-3)
    base = dict(r_in=-70e-3, ray_subsample=16, n_workers=1,
                traced_kwargs=TKW, final_leg='paraxial')
    fr = dict(dx_out=0.4e-6, N_out=64)
    # explicit collins + the stop-plane keys: REFUSED before, SELECTS after
    run('D1-collins-standoff', CA.propagate_traced_carrier_chain,
        env, GROUPS, WL_IR, 55e-6, transport='collins',
        final_distance=9e-3, focus_readout=dict(fr, standoff=2e-3), **base)
    run('D2-collins-containment', CA.propagate_traced_carrier_chain,
        env, GROUPS, WL_IR, 55e-6, transport='collins',
        final_distance=9e-3,
        focus_readout=dict(fr, on_focus_containment='warn'), **base)
    run('D3-collins-zero-distance', CA.propagate_traced_carrier_chain,
        env, GROUPS, WL_IR, 55e-6, transport='collins',
        final_distance=0.0, focus_readout=fr, **base)
    one = [{'field': env, 'carrier': -70e-3}]
    run('D4-multi-collins-standoff',
        CA.propagate_traced_carrier_chain_multi, one, GROUPS, WL_IR, 55e-6,
        output_grid=dict(fr, standoff=2e-3), transport='collins',
        ray_subsample=16, n_workers=1, traced_kwargs=TKW,
        final_leg='paraxial', final_distance=9e-3)
    # the transport vocabulary gate's own message
    run('D5-bad-transport', CA.propagate_carrier_referenced,
        gauss(32, 8e-6, 60e-6), -0.05, 5e-3, WL, 8e-6, transport='Sziklas')
    # the Migration-Guide's near-focus recipe
    run('D6-gap-kernel-fresnel-recipe', CA.propagate_traced_carrier_chain,
        env, GROUPS, WL_IR, 55e-6, gap_kernel='fresnel',
        final_distance=9e-3, focus_readout=fr, **base)


def main():
    sec_signatures()
    sec_moves()
    sec_jax_trace()
    sec_contracts()
    OUT['_meta'] = {'tree': _TREE, 'build': vlib3.build_tag(),
                    'lumenairy': la.__version__,
                    'python': sys.version.split()[0]}
    with open(sys.argv[2], 'w', encoding='utf-8') as fh:
        json.dump(OUT, fh, indent=1, sort_keys=True, default=str)
    sys.stdout.write('[probe_entrypoints] %d records -> %s\n'
                     % (len(OUT), sys.argv[2]))


if __name__ == '__main__':
    main()
