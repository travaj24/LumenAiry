"""(5e, follow-up) jax.jit with the envelope as a CLOSED-OVER CONSTANT.

``_is_traced`` inspects the ENVELOPE only.  A concrete jnp array closed over
by a jitted function is not a Tracer, so the refusal does not fire -- but
inside a jit trace every jnp op is STAGED, so the measurement FFT's output
IS a Tracer and ``_collins_power_marginals``'s ``to_numpy`` hits it.

    python v2_closedover.py <tree> <out.json>
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
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import lumenairy.propagators.carrier as CA

    ax = (np.arange(N) - N // 2) * DX
    X, Y = np.meshgrid(ax, ax)
    env = np.exp(-(X ** 2 + Y ** 2) / (41e-6) ** 2).astype(np.complex128)
    envj = jnp.asarray(env)
    res = {'build': build_tag(), 'jax_version': jax.__version__}

    KW = dict(dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N, R_ref=R_REF)
    cells = {}
    for gk in ('fresnel', 'auto'):
        for ocs in ('ignore', 'warn'):
            for holder, e in (('jax_const', envj), ('numpy_const', env)):
                def f(gk=gk, ocs=ocs, e=e):
                    return CA._collins_transport(
                        e, R_IN, Z, WL, DX, DX,
                        **dict(KW, gap_kernel=gk, on_collins_sampling=ocs))
                key = '%s|%s|closed_%s' % (gk, ocs, holder)
                try:
                    o = jax.jit(f)()
                    eager = CA._collins_transport(
                        e, R_IN, Z, WL, DX, DX,
                        **dict(KW, gap_kernel=gk, on_collins_sampling=ocs))
                    cells[key] = {
                        'ran': True,
                        'rel_vs_eager': float(
                            np.linalg.norm(np.asarray(o) - np.asarray(eager))
                            / np.linalg.norm(np.asarray(eager)))}
                except BaseException as exc:               # noqa: BLE001
                    cells[key] = {
                        'ran': False, 'type': type(exc).__name__,
                        'is_designed_refusal': 'trace-safe' in str(exc),
                        'msg': str(exc)[:200]}
    res['jit_closed_over'] = cells

    # and the same as an ARGUMENT, for the contrast
    arg = {}
    for gk in ('fresnel', 'auto'):
        for ocs in ('ignore', 'warn'):
            def g(e, gk=gk, ocs=ocs):
                return CA._collins_transport(
                    e, R_IN, Z, WL, DX, DX,
                    **dict(KW, gap_kernel=gk, on_collins_sampling=ocs))
            key = '%s|%s|argument' % (gk, ocs)
            try:
                jax.jit(g)(envj)
                arg[key] = {'ran': True}
            except BaseException as exc:                   # noqa: BLE001
                arg[key] = {'ran': False, 'type': type(exc).__name__,
                            'is_designed_refusal': 'trace-safe' in str(exc)}
    res['jit_argument'] = arg
    write_json(res, out_path)
    print(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main()
