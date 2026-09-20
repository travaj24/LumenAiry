"""H2-2 says the Collins transport "runs on NumPy, CuPy and JAX, selected by
the array the caller passes".  WHERE does that hold -- at the private
``_collins_transport``, or at the surfaces a caller actually reaches?

For each public spelling this probe records the TYPE of the array that comes
back for a NumPy, an eager-JAX and (where present) a CuPy input.  A JAX input
that comes back as ``numpy.ndarray`` means the leg round-tripped through the
host, which is the thing the package says it stopped doing.
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
import lumenairy.propagators.carrier as CA

WL, N, DX, R_IN, Z, R_REF = 633e-9, 64, 8e-6, -0.05, 5e-3, -0.045
ax = (np.arange(N) - N // 2) * DX
X, Y = np.meshgrid(ax, ax)
ENV = np.exp(-(X**2 + Y**2) / (60e-6)**2).astype(np.complex128)

try:
    import cupy
    CUPY = cupy.asarray(ENV)
except Exception as exc:                            # noqa: BLE001 -- recorded
    cupy, CUPY = None, None
    print(f"[cupy] unavailable: {type(exc).__name__}: {exc}")

rows = []


def run(label, fn, arr):
    try:
        v = fn(arr)
        e = getattr(v, 'env', v)
        rows.append({'case': label, 'in': type(arr).__name__,
                     'out': type(e).__name__, 'module': type(e).__module__,
                     'outcome': 'ok'})
        print(f"{label:46s} in={type(arr).__name__:16s} "
              f"out={type(e).__module__}.{type(e).__name__}")
    except BaseException as exc:                    # noqa: BLE001 -- recorded
        tb = traceback.extract_tb(exc.__traceback__)
        fr = [f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}"
              for f in tb if 'lumenairy' in f.filename]
        rows.append({'case': label, 'in': type(arr).__name__,
                     'outcome': 'raised', 'type': type(exc).__name__,
                     'msg': str(exc)[:400], 'frames': fr[-3:]})
        print(f"{label:46s} in={type(arr).__name__:16s} "
              f"RAISED {type(exc).__name__}: {str(exc)[:110]!r}  {fr[-2:]}")


def priv(e):
    return CA._collins_transport(e, R_IN, Z, WL, DX, DX, dx_out=DX, dy_out=DX,
                                 N_out_x=N, N_out_y=N, R_ref=R_REF,
                                 gap_kernel='fresnel',
                                 on_collins_sampling='ignore')


def pub(e):
    return CA.propagate_carrier_referenced(
        e, R_IN, Z, WL, DX, transport='collins', gap_kernel='fresnel',
        on_collins_sampling='ignore')


def pub_sziklas(e):
    return CA.propagate_carrier_referenced(e, R_IN, Z, WL, DX,
                                           gap_kernel='fresnel')


for arr, tag in ((ENV, 'numpy'), (jnp.asarray(ENV), 'jax-eager')) + \
        (((CUPY, 'cupy'),) if CUPY is not None else ()):
    run(f"_collins_transport  [{tag}]", priv, arr)
    run(f"public transport='collins'  [{tag}]", pub, arr)
    run(f"public default transport (sziklas)  [{tag}]", pub_sziklas, arr)

vlib.write_json({'build': vlib.build_tag(), 'rows': rows,
                 'cupy_present': CUPY is not None},
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"v0_backend_reach_"
                             f"{vlib.build_tag().split('-')[0].lower()}.json"))
