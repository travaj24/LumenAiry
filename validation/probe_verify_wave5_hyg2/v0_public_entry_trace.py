"""Does the TRACED refusal reach the PUBLIC entry point, or only the private
transport?

H2-2's refusal lives in ``_collins_transport``.  A caller who reaches the
Collins leg through ``propagate_carrier_referenced(transport='collins')``
under ``jax.jit`` / ``jax.grad`` meets whatever the public chain does FIRST --
and the chain has its own measuring steps (the focus-crossing test, the
near-focus bridge test, the auto carrier fit).  This probe records, for each
spelling, WHAT is raised and WHERE, so "the transport refuses with a message
naming the two ways out" can be checked at the surface a user actually calls.
"""
from __future__ import annotations
import os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
vlib.anchor(os.path.join(ROOT, 'lumenairy'))

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from lumenairy.propagators.carrier import (
    propagate_carrier_referenced, _collins_transport)

WL, N, DX, R_IN, Z, R_REF = 633e-9, 64, 8e-6, -0.05, 5e-3, -0.045
ax = (np.arange(N) - N // 2) * DX
X, Y = np.meshgrid(ax, ax)
ENV = np.exp(-(X**2 + Y**2) / (60e-6)**2).astype(np.complex128)

rows = []


def attempt(label, fn, *a):
    try:
        v = fn(*a)
        rows.append({'case': label, 'outcome': 'ok',
                     'type': type(v).__name__})
        print(f"{label:52s} OK ({type(v).__name__})")
    except BaseException as exc:                    # noqa: BLE001 -- recorded
        tb = traceback.extract_tb(exc.__traceback__)
        inside = [f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}"
                  for f in tb if 'lumenairy' in f.filename]
        msg = str(exc)
        rows.append({'case': label, 'outcome': 'raised',
                     'type': type(exc).__name__, 'msg': msg[:900],
                     'lumenairy_frames': inside[-4:],
                     'names_gap_kernel': 'gap_kernel' in msg,
                     'names_on_collins_sampling':
                         'on_collins_sampling' in msg,
                     'names_fresnel': "'fresnel'" in msg,
                     'names_ignore': "'ignore'" in msg})
        print(f"{label:52s} {type(exc).__name__}: {msg[:150]!r}")
        print(f"{'':52s}   frames: {inside[-3:]}")


# --- the PRIVATE transport, which is where the refusal lives ---------------
def priv(kw):
    def f(e):
        return _collins_transport(e, R_IN, Z, WL, DX, DX, dx_out=DX,
                                  dy_out=DX, N_out_x=N, N_out_y=N,
                                  R_ref=R_REF, **kw)
    return f


attempt("private, jit, defaults",
        lambda: jax.jit(priv({}))(jnp.asarray(ENV)))
attempt("private, jit, gap=fresnel + sampling=ignore",
        lambda: jax.jit(priv(dict(gap_kernel='fresnel',
                                  on_collins_sampling='ignore')))(
            jnp.asarray(ENV)))

# --- the PUBLIC entry, which is what a user calls --------------------------
def pub(kw):
    def f(e):
        return propagate_carrier_referenced(
            e, R_IN, Z, WL, DX, transport='collins', **kw).env
    return f


attempt("public, jit, transport='collins', defaults",
        lambda: jax.jit(pub({}))(jnp.asarray(ENV)))
attempt("public, jit, transport='collins', fresnel+ignore",
        lambda: jax.jit(pub(dict(gap_kernel='fresnel',
                                 on_collins_sampling='ignore')))(
            jnp.asarray(ENV)))
attempt("public, jit, transport='collins', fresnel+ignore+carrier_out=inf",
        lambda: jax.jit(pub(dict(gap_kernel='fresnel',
                                 on_collins_sampling='ignore',
                                 carrier_out=float('inf'))))(
            jnp.asarray(ENV)))
attempt("public, grad, transport='collins', fresnel+ignore",
        lambda: jax.grad(lambda a: jnp.sum(jnp.abs(
            pub(dict(gap_kernel='fresnel', on_collins_sampling='ignore'))(
                a.astype(jnp.complex128)))**2))(jnp.asarray(np.real(ENV))))
attempt("public, jit, default transport (sziklas), defaults",
        lambda: jax.jit(lambda e: propagate_carrier_referenced(
            e, R_IN, Z, WL, DX).env)(jnp.asarray(ENV)))

# --- a traced SCALAR with a concrete envelope ------------------------------
attempt("private, grad wrt z (concrete envelope)",
        lambda: jax.grad(lambda z: jnp.sum(jnp.abs(_collins_transport(
            jnp.asarray(ENV), R_IN, z, WL, DX, DX, dx_out=DX, dy_out=DX,
            N_out_x=N, N_out_y=N, R_ref=R_REF, gap_kernel='fresnel',
            on_collins_sampling='ignore'))**2))(Z))

# --- no partial state: a refusing grad, then a working one -----------------
try:
    jax.grad(lambda a: jnp.sum(jnp.abs(priv({})(a.astype(jnp.complex128)))**2)
             )(jnp.asarray(np.real(ENV)))
    after = 'no-raise'
except Exception as exc:                            # noqa: BLE001 -- recorded
    after = f"{type(exc).__name__}"
g = jax.grad(lambda a: jnp.sum(jnp.abs(priv(
    dict(gap_kernel='fresnel', on_collins_sampling='ignore'))(
        a.astype(jnp.complex128)))**2))(jnp.asarray(np.real(ENV)))
rows.append({'case': 'a working grad AFTER a refusing one',
             'refusal_type': after,
             'recovered': bool(np.all(np.isfinite(np.asarray(g)))),
             'grad_max': float(np.max(np.abs(np.asarray(g))))})
print(f"after a refusing grad ({after}) a working grad recovers: "
      f"{np.all(np.isfinite(np.asarray(g)))}")

vlib.write_json({'build': vlib.build_tag(), 'rows': rows},
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"v0_public_entry_trace_"
                             f"{vlib.build_tag().split('-')[0].lower()}.json"))
