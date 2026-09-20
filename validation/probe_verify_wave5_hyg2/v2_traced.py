"""(5) THE TRACED REFUSAL -- confirm, refute, and hunt for the hole.

    python v2_traced.py <tree> <out.json>
"""
from __future__ import annotations

import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                            # noqa: E402
from vlib import anchor, build_tag, write_json                 # noqa: E402

WL = 1.064e-6
N = 96
DX = 5.5e-6
R_IN = -0.028
Z = 2.3e-3
R_REF = -0.0215


def gauss(n=N, dx=DX, w=41e-6):
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    return np.exp(-(X ** 2 + Y ** 2) / (w * w))


def main():
    tree, out_path = sys.argv[1], sys.argv[2]
    anchor(tree)
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import lumenairy.propagators.carrier as CA
    import lumenairy.propagators._bluestein as BL

    res = {'build': build_tag(), 'tree': tree,
           'jax_version': jax.__version__}
    amp = jnp.asarray(gauss())
    envj = jnp.asarray(gauss(), dtype=jnp.complex128)
    KW = dict(dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N, R_ref=R_REF)

    def call(e, **kw):
        return CA._collins_transport(e, R_IN, Z, WL, DX, DX, **dict(KW, **kw))

    # ---- (a) the full 3 x 3 x 2 matrix under grad AND jit ----------------
    matrix = {}
    for gk in ('auto', 'fresnel', 'exact'):
        for ocs in ('error', 'warn', 'ignore'):
            for so in (None, 'dict'):
                stats = None if so is None else {}

                def merit(a, gk=gk, ocs=ocs, stats=stats):
                    out = call(a.astype(jnp.complex128), gap_kernel=gk,
                               on_collins_sampling=ocs, stats_out=stats)
                    return jnp.sum(jnp.abs(out) ** 2)

                cell = {}
                for mode in ('grad', 'jit'):
                    try:
                        if mode == 'grad':
                            jax.grad(merit)(amp)
                        else:
                            jax.jit(merit)(amp)
                        cell[mode] = 'RAN'
                    except BaseException as exc:            # noqa: BLE001
                        cell[mode] = '%s: %s' % (type(exc).__name__,
                                                 str(exc)[:110])
                matrix['gk=%s|ocs=%s|stats=%s' % (gk, ocs, so)] = cell
    res['a_matrix'] = matrix
    res['a_ran_cells'] = sorted(k for k, v in matrix.items()
                                if v['grad'] == 'RAN' and v['jit'] == 'RAN')

    # ---- (b) does the message name BOTH ways out? ------------------------
    msgs = {}
    for gk, ocs in (('auto', 'warn'), ('auto', 'ignore'), ('fresnel', 'warn'),
                    ('exact', 'error')):
        def m(a, gk=gk, ocs=ocs):
            return jnp.sum(jnp.abs(call(a.astype(jnp.complex128),
                                        gap_kernel=gk,
                                        on_collins_sampling=ocs)) ** 2)
        try:
            jax.grad(m)(amp)
            msgs['%s|%s' % (gk, ocs)] = {'raised': False}
        except BaseException as exc:                        # noqa: BLE001
            s = str(exc)
            msgs['%s|%s' % (gk, ocs)] = {
                'raised': True, 'type': type(exc).__name__,
                'names_fresnel': "gap_kernel='fresnel'" in s,
                'names_ignore': "on_collins_sampling='ignore'" in s,
                'says_Tracer': 'Tracer' in s,
                'says_trace_safe': 'trace-safe' in s,
                'msg': s[:900]}
    res['b_messages'] = msgs

    # ---- (c) an EAGER jax array measures normally, warning included ------
    wide = gauss(n=64, dx=40e-6, w=460e-6)
    ew = jnp.asarray(wide, dtype=jnp.complex128)
    st = {}
    out = CA._collins_transport(ew, -0.30, 0.29, 1.55e-6, 40e-6, 40e-6,
                                dx_out=40e-6 * 16, dy_out=40e-6 * 16,
                                N_out_x=64, N_out_y=64, R_ref=np.inf,
                                gap_kernel='auto',
                                on_collins_sampling='ignore', stats_out=st)
    res['c_eager_jax'] = {
        'is_tracer': bool(CA._is_traced(ew)),
        'out_type': type(out).__name__, 'out_shape': list(out.shape),
        'stats_keys': sorted(st), 'kernel': st.get('kernel'),
        'k4': st.get('k4'), 'worst': st.get('worst')}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        CA._collins_transport(ew, -0.30, 0.29, 1.55e-6, 40e-6, 40e-6,
                              dx_out=40e-6 * 16, dy_out=40e-6 * 16,
                              N_out_x=64, N_out_y=64, R_ref=np.inf,
                              gap_kernel='auto', on_collins_sampling='warn')
    res['c_eager_jax_warnings'] = [(w.category.__name__, str(w.message)[:160])
                                   for w in caught]

    # eager CuPy
    cup = {'available': False}
    try:
        import cupy as cp
        cup['available'] = True
        cup['version'] = cp.__version__
        from lumenairy.propagators import fft_infra as FI
        cup['fft_infra_CUPY_AVAILABLE'] = bool(
            getattr(FI, 'CUPY_AVAILABLE', None))
        cup['device_count'] = int(cp.cuda.runtime.getDeviceCount())
        ec = cp.asarray(wide, dtype=cp.complex128)
        st2 = {}
        o2 = CA._collins_transport(ec, -0.30, 0.29, 1.55e-6, 40e-6, 40e-6,
                                   dx_out=40e-6 * 16, dy_out=40e-6 * 16,
                                   N_out_x=64, N_out_y=64, R_ref=np.inf,
                                   gap_kernel='auto',
                                   on_collins_sampling='ignore',
                                   stats_out=st2)
        cup['ran'] = True
        cup['out_type'] = type(o2).__name__
        cup['kernel'] = st2.get('kernel')
        cup['rel_vs_numpy'] = float(np.linalg.norm(
            cp.asnumpy(o2) - np.asarray(CA._collins_transport(
                wide.astype(np.complex128), -0.30, 0.29, 1.55e-6, 40e-6,
                40e-6, dx_out=40e-6 * 16, dy_out=40e-6 * 16, N_out_x=64,
                N_out_y=64, R_ref=np.inf, gap_kernel='auto',
                on_collins_sampling='ignore')))
            / np.linalg.norm(cp.asnumpy(o2)))
    except BaseException as exc:                            # noqa: BLE001
        cup['ran'] = False
        cup['error'] = '%s: %s' % (type(exc).__name__, str(exc)[:220])
    res['c_eager_cupy'] = cup

    # ---- (d) NO PARTIAL STATE after a refusing jax.grad ------------------
    def snap():
        return {
            'h_fft_cache_len': len(BL._H_FFT_CACHE),
            'h_fft_cache_hits': int(BL._H_FFT_CACHE_HITS),
            'h_fft_cache_bytes': int(BL._h_fft_cache_bytes()),
            'warning_filters': len(warnings.filters),
            'warning_filters_repr': repr(warnings.filters)[:400],
        }
    before = snap()
    stats_probe = {'SENTINEL': 1}

    def bad(a):
        return jnp.sum(jnp.abs(call(a.astype(jnp.complex128),
                                    gap_kernel='auto',
                                    on_collins_sampling='warn',
                                    stats_out=stats_probe)) ** 2)
    exc_info = {}
    try:
        jax.grad(bad)(amp)
        exc_info['raised'] = False
    except BaseException as exc:                            # noqa: BLE001
        chain, e = [], exc
        for _ in range(6):
            chain.append(type(e).__name__)
            nxt = e.__cause__ or e.__context__
            if nxt is None or nxt is e:
                break
            e = nxt
        exc_info = {
            'raised': True, 'type': type(exc).__name__,
            'is_ValueError': isinstance(exc, ValueError),
            'cause_chain': chain,
            'has_jax_wrapper': any('JaxStackTrace' in c for c in chain),
            'args_types': [type(a).__name__ for a in exc.args],
        }
    after = snap()
    res['d_partial_state'] = {
        'exception': exc_info,
        'stats_out_untouched': stats_probe == {'SENTINEL': 1},
        'stats_out_after': dict(stats_probe),
        'cache_before': before, 'cache_after': after,
        'cache_unchanged': (before['h_fft_cache_len']
                            == after['h_fft_cache_len']
                            and before['h_fft_cache_hits']
                            == after['h_fft_cache_hits']
                            and before['h_fft_cache_bytes']
                            == after['h_fft_cache_bytes']),
        'warning_filters_unchanged':
            before['warning_filters_repr'] == after['warning_filters_repr'],
    }
    # a subsequent SUCCESSFUL jax.grad on the same function shape
    def good(a):
        return jnp.sum(jnp.abs(call(a.astype(jnp.complex128),
                                    gap_kernel='fresnel',
                                    on_collins_sampling='ignore')) ** 2)
    try:
        g = np.asarray(jax.grad(good)(amp))
        res['d_subsequent_grad'] = {
            'ok': True, 'finite': bool(np.all(np.isfinite(g))),
            'max_abs': float(np.max(np.abs(g)))}
    except BaseException as exc:                            # noqa: BLE001
        res['d_subsequent_grad'] = {'ok': False,
                                    'err': '%s: %s' % (type(exc).__name__,
                                                       str(exc)[:200])}
    # and a leaked-tracer check: re-run the refusing call, then use jnp again
    try:
        jax.grad(bad)(amp)
    except BaseException:                                   # noqa: BLE001
        pass
    res['d_jnp_still_usable'] = float(
        np.asarray(jnp.sum(jnp.asarray([1.0, 2.0]))))

    # ---- (e) HUNT: a combination that should refuse but does not ---------
    hunt = {}

    # e1. instrument the two measurement helpers: are they EVER reached
    #     under a trace on the allowed spelling?
    seen = {'stats': 0, 'check': 0, 'space': 0, 'angle': 0}
    o_stats = CA._collins_sampling_stats
    o_check = CA._check_collins_sampling
    o_space = CA._collins_space_support
    o_angle = CA._collins_angle_support

    def wrap(name, fn):
        def w(*a, **k):
            seen[name] += 1
            return fn(*a, **k)
        return w
    CA._collins_sampling_stats = wrap('stats', o_stats)
    CA._check_collins_sampling = wrap('check', o_check)
    CA._collins_space_support = wrap('space', o_space)
    CA._collins_angle_support = wrap('angle', o_angle)
    try:
        jax.grad(good)(amp)
        jax.jit(good)(amp)
        hunt['e1_measurement_helpers_reached_under_trace'] = dict(seen)
    finally:
        CA._collins_sampling_stats = o_stats
        CA._check_collins_sampling = o_check
        CA._collins_space_support = o_space
        CA._collins_angle_support = o_angle

    # e2. ASTIGMATIC carrier + gap_kernel='auto' under a trace.  The blocked
    #     list guards the kernel clause with `Ax == Ay`, so an astigmatic
    #     'auto' may slip through the refusal.
    def astig(a):
        out = CA._collins_transport(
            a.astype(jnp.complex128), (-0.028, -0.041), Z, WL, DX, DX,
            dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N,
            R_ref=(-0.0215, -0.031), gap_kernel='auto',
            on_collins_sampling='ignore')
        return jnp.sum(jnp.abs(out) ** 2)
    try:
        ga = np.asarray(jax.grad(astig)(amp))
        hunt['e2_astigmatic_auto_under_grad'] = {
            'refused': False, 'max_abs_grad': float(np.max(np.abs(ga)))}
    except BaseException as exc:                            # noqa: BLE001
        hunt['e2_astigmatic_auto_under_grad'] = {
            'refused': True, 'type': type(exc).__name__,
            'msg': str(exc)[:200]}
    # what does the EAGER astigmatic 'auto' resolve to?  (is it the same?)
    st3 = {}
    CA._collins_transport(gauss().astype(np.complex128), (-0.028, -0.041), Z,
                          WL, DX, DX, dx_out=DX, dy_out=DX, N_out_x=N,
                          N_out_y=N, R_ref=(-0.0215, -0.031),
                          gap_kernel='auto', on_collins_sampling='ignore',
                          stats_out=st3)
    hunt['e2_eager_astigmatic_kernel'] = st3.get('kernel')

    # e2b. astigmatic 'auto' + on_collins_sampling='warn' under a trace:
    #      only the guard clause should block.
    def astig_warn(a):
        out = CA._collins_transport(
            a.astype(jnp.complex128), (-0.028, -0.041), Z, WL, DX, DX,
            dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N,
            R_ref=(-0.0215, -0.031), gap_kernel='auto',
            on_collins_sampling='warn')
        return jnp.sum(jnp.abs(out) ** 2)
    try:
        jax.grad(astig_warn)(amp)
        hunt['e2b_astig_auto_warn'] = 'RAN'
    except BaseException as exc:                            # noqa: BLE001
        s = str(exc)
        hunt['e2b_astig_auto_warn'] = {
            'type': type(exc).__name__,
            'names_gap_kernel': 'gap_kernel' in s,
            'names_ocs': 'on_collins_sampling' in s,
            'n_decisions': s.split('asks for ')[-1][:3] if 'asks for ' in s
            else None}

    # e3. is _is_traced checked on env ONLY?  Trace w.r.t. z / R_in / dx with
    #     a CONCRETE envelope.
    envc = gauss().astype(np.complex128)
    for name, fn in (
        ('z', lambda t: jnp.sum(jnp.abs(CA._collins_transport(
            envc, R_IN, t, WL, DX, DX, **dict(KW, gap_kernel='fresnel',
                                              on_collins_sampling='ignore')
        )) ** 2)),
        ('R_in', lambda t: jnp.sum(jnp.abs(CA._collins_transport(
            envc, t, Z, WL, DX, DX, **dict(KW, gap_kernel='fresnel',
                                           on_collins_sampling='ignore')
        )) ** 2)),
        ('dx', lambda t: jnp.sum(jnp.abs(CA._collins_transport(
            envc, R_IN, Z, WL, t, t, **dict(KW, gap_kernel='fresnel',
                                            on_collins_sampling='ignore')
        )) ** 2)),
    ):
        try:
            v = float(np.asarray(jax.grad(fn)(
                jnp.asarray({'z': Z, 'R_in': R_IN, 'dx': DX}[name]))))
            hunt['e3_traced_scalar_' + name] = {'refused': False, 'grad': v}
        except BaseException as exc:                        # noqa: BLE001
            hunt['e3_traced_scalar_' + name] = {
                'refused': True, 'type': type(exc).__name__,
                'mentions_collins': '_collins' in str(exc),
                'mentions_Tracer_refusal': 'trace-safe' in str(exc),
                'msg': str(exc)[:260]}
        # and with a TRACED envelope at the same time
        try:
            def both(pair, name=name):
                a, t = pair
                kw = dict(KW, gap_kernel='fresnel',
                          on_collins_sampling='ignore')
                e = a.astype(jnp.complex128)
                if name == 'z':
                    o = CA._collins_transport(e, R_IN, t, WL, DX, DX, **kw)
                elif name == 'R_in':
                    o = CA._collins_transport(e, t, Z, WL, DX, DX, **kw)
                else:
                    o = CA._collins_transport(e, R_IN, Z, WL, t, t, **kw)
                return jnp.sum(jnp.abs(o) ** 2)
            jax.grad(both)((amp,
                            jnp.asarray({'z': Z, 'R_in': R_IN,
                                         'dx': DX}[name])))
            hunt['e3_traced_env_and_' + name] = 'RAN'
        except BaseException as exc:                        # noqa: BLE001
            hunt['e3_traced_env_and_' + name] = '%s: %s' % (
                type(exc).__name__, str(exc)[:160])

    # e4. jit ARGUMENT vs CLOSED-OVER CONSTANT
    def f_arg(e):
        return CA._collins_transport(e, R_IN, Z, WL, DX, DX,
                                     **dict(KW, gap_kernel='auto',
                                            on_collins_sampling='ignore'))
    try:
        jax.jit(f_arg)(envj)
        hunt['e4_jit_argument'] = 'RAN (no refusal)'
    except BaseException as exc:                            # noqa: BLE001
        hunt['e4_jit_argument'] = '%s: %s' % (type(exc).__name__,
                                              str(exc)[:120])

    def f_closed():
        return CA._collins_transport(envj, R_IN, Z, WL, DX, DX,
                                     **dict(KW, gap_kernel='auto',
                                            on_collins_sampling='ignore'))
    try:
        o = jax.jit(f_closed)()
        eager = CA._collins_transport(envj, R_IN, Z, WL, DX, DX,
                                      **dict(KW, gap_kernel='auto',
                                             on_collins_sampling='ignore'))
        hunt['e4_jit_closed_over_constant'] = {
            'refused': False,
            'rel_vs_eager': float(np.linalg.norm(np.asarray(o)
                                                 - np.asarray(eager))
                                  / np.linalg.norm(np.asarray(eager)))}
    except BaseException as exc:                            # noqa: BLE001
        hunt['e4_jit_closed_over_constant'] = '%s: %s' % (
            type(exc).__name__, str(exc)[:160])

    # e5. other trace kinds: vmap, linearize, vjp
    for kind in ('vmap', 'linearize', 'vjp'):
        def m(a):
            return jnp.sum(jnp.abs(call(a.astype(jnp.complex128),
                                        gap_kernel='auto',
                                        on_collins_sampling='warn')) ** 2)
        try:
            if kind == 'vmap':
                jax.vmap(m)(jnp.stack([amp, amp]))
            elif kind == 'linearize':
                jax.linearize(m, amp)
            else:
                jax.vjp(m, amp)
            hunt['e5_' + kind] = 'RAN (no refusal)'
        except BaseException as exc:                        # noqa: BLE001
            hunt['e5_' + kind] = '%s: %s' % (type(exc).__name__,
                                             str(exc)[:110])

    # e6. THE PUBLIC ROUTE.  propagate_carrier_referenced(transport='collins')
    #     -> _collins_carrier_leg, which does np.asarray(env) at its top.
    pub = {}
    pj = CA.propagate_carrier_referenced(envj, R_IN, Z, WL, DX,
                                         transport='collins',
                                         gap_kernel='fresnel',
                                         on_collins_sampling='ignore')
    pn = CA.propagate_carrier_referenced(np.asarray(envj), R_IN, Z, WL, DX,
                                         transport='collins',
                                         gap_kernel='fresnel',
                                         on_collins_sampling='ignore')
    pub['jax_in_returns_type'] = type(pj.env).__name__
    pub['jax_in_returns_module'] = type(pj.env).__module__
    pub['numpy_in_returns_type'] = type(pn.env).__name__
    pub['bitwise_equal_to_numpy_arm'] = bool(np.array_equal(
        np.ascontiguousarray(np.asarray(pj.env)).view(np.float64),
        np.ascontiguousarray(np.asarray(pn.env)).view(np.float64)))

    def pub_merit(a):
        r = CA.propagate_carrier_referenced(
            a.astype(jnp.complex128), R_IN, Z, WL, DX, transport='collins',
            gap_kernel='fresnel', on_collins_sampling='ignore')
        return jnp.sum(jnp.abs(r.env) ** 2)
    try:
        jax.grad(pub_merit)(amp)
        pub['grad_through_public_entry'] = 'RAN'
    except BaseException as exc:                            # noqa: BLE001
        pub['grad_through_public_entry'] = '%s: %s' % (type(exc).__name__,
                                                       str(exc)[:220])
        pub['grad_gives_designed_refusal'] = 'trace-safe' in str(exc)
    # the same question for the exact-focus readout, which does NOT convert
    def ro_merit(a):
        o = CA._collins_focus_readout(
            a.astype(jnp.complex128), R_IN, Z, WL, DX, DX, dx_out=2.0e-6,
            N_out=64, gap_kernel='fresnel', on_replica='ignore',
            on_collins_sampling='ignore')
        return jnp.sum(jnp.abs(o) ** 2)
    try:
        gg = np.asarray(jax.grad(ro_merit)(amp))
        pub['grad_through_focus_readout'] = {
            'ok': True, 'max_abs': float(np.max(np.abs(gg)))}
    except BaseException as exc:                            # noqa: BLE001
        pub['grad_through_focus_readout'] = '%s: %s' % (type(exc).__name__,
                                                        str(exc)[:220])
    try:
        import cupy as cp
        CA.propagate_carrier_referenced(
            cp.asarray(gauss(), dtype=cp.complex128), R_IN, Z, WL, DX,
            transport='collins', gap_kernel='fresnel',
            on_collins_sampling='ignore')
        pub['cupy_through_public_entry'] = 'RAN'
    except BaseException as exc:                            # noqa: BLE001
        pub['cupy_through_public_entry'] = '%s: %s' % (type(exc).__name__,
                                                       str(exc)[:200])
    hunt['e6_public_entry'] = pub

    res['e_hunt'] = hunt
    write_json(res, out_path)
    print(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main()
