"""Shared fixtures of the Phase E3 (JAX twin) probes.

Every probe imports this first: it pins the BLAS threads (must already be set
on the command line), asserts that ``lumenairy`` is the worktree's, enables
JAX x64, and provides the reference fixtures and the traced-map builders.
"""
import json
import os
import sys
import time
import warnings

ROOT = os.path.normcase(os.path.abspath(
    os.environ.get("LUM_TREE", "C:/tmp/lum_curved_e3")))
import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

assert os.path.normcase(os.path.abspath(lumenairy.__file__)).startswith(ROOT), (
    f"lumenairy imported from {lumenairy.__file__}, not {ROOT}")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    assert os.environ.get(_k) == "1", f"{_k} must be 1 on the command line"

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from lumenairy.elements.pmm import (  # noqa: E402
    PMM2DStackPure,
    _curvemap as CM,  # noqa: E402
    twod_staggered as TS,  # noqa: E402,F401
)
from lumenairy.elements.pmm._jax_twod_staggered import (  # noqa: E402,F401
    StagJaxTwin,
)

HERE = os.path.dirname(os.path.abspath(__file__))
WL, P = 1.0, 1.2
DEG = np.pi / 180.0
warnings.simplefilter("ignore")


def dump(name, obj):
    obj = dict(obj)
    obj["env"] = {"python": sys.version.split()[0], "numpy": np.__version__,
                  "jax": jax.__version__, "lumenairy": lumenairy.__file__,
                  "platform": sys.platform}
    with open(os.path.join(HERE, name), "w") as f:
        json.dump(obj, f, indent=1, default=float)


def tic():
    return time.perf_counter()


# --------------------------------------------------------------------------
# rectangular pillar on a 3 x 3 grid: walls x [0, cx-w/2, cx+w/2, P]
# --------------------------------------------------------------------------
RECT = dict(cx=0.6, cy=0.6, w=0.5, h=0.4, eps=4.0, depth=0.4)


def rect_walls(w, h, cx=RECT["cx"], cy=RECT["cy"]):
    return (np.array([0.0, cx - w / 2, cx + w / 2, P]),
            np.array([0.0, cy - h / 2, cy + h / 2, P]))


def rect_cell(eps=RECT["eps"]):
    c = np.ones((3, 3), complex)
    c[1, 1] = eps
    return c


def rect_stack(M, w=RECT["w"], h=RECT["h"], n_orders=2, theta=0.0, phi=0.0,
               cmap=None, eps=RECT["eps"], depth=RECT["depth"]):
    """The NumPy stack of the rectangular pillar: through the shapes route
    (a :class:`Rect` -- the shipped UNMAPPED solver on its own walls), or on
    the explicit map ``cmap`` (an ``eps_cell`` layer on the map's grid)."""
    from lumenairy.elements.pmm import Rect
    kw = {} if cmap is None else {"cmap": cmap}
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=M,
                        n_orders=n_orders, **kw)
    if cmap is None:
        st.add_layer(depth, shapes=[Rect(RECT["cx"], RECT["cy"], w, h, eps)],
                     background_eps=1.0)
    else:
        st.add_layer(depth, eps_cell=rect_cell(eps))
    st.set_source(WL, theta=theta, phi=phi)
    return st


def rect_traced_map(ref_map, w, h=RECT["h"], cx=RECT["cx"], cy=RECT["cy"]):
    """A TRACED transfinite map on the frozen grid of ``ref_map`` whose
    interior vertex images sit at the rectangle of width ``w`` (straight
    edges: bilinear, here affine, cells)."""
    xs = jnp.stack([jnp.asarray(0.0), cx - w / 2, cx + w / 2,
                    jnp.asarray(P)])
    ys = jnp.stack([jnp.asarray(0.0), cy - h / 2, cy + h / 2,
                    jnp.asarray(P)])
    V = jnp.stack([jnp.broadcast_to(xs[:, None], (4, 4)),
                   jnp.broadcast_to(ys[None, :], (4, 4))], axis=-1)
    return CM.TransfiniteMap._traced(ref_map.u_bounds, ref_map.v_bounds, V,
                                     {}, [])


# --------------------------------------------------------------------------
# circular pillar on the Phase B 3 x 3 layout
# --------------------------------------------------------------------------
CIRC = dict(c=0.6, r=0.36, eps=4.0, depth=0.5)


def circle_curves(r, c=CIRC["c"]):
    cc = (c, c)
    return {("h", 1, 1): CM.Arc(cc, r, 225 * DEG, 315 * DEG),
            ("h", 1, 2): CM.Arc(cc, r, 135 * DEG, 45 * DEG),
            ("v", 1, 1): CM.Arc(cc, r, 225 * DEG, 135 * DEG),
            ("v", 2, 1): CM.Arc(cc, r, -45 * DEG, 45 * DEG)}


def circle_traced_map(ref_map, r, c=CIRC["c"]):
    hh = r / np.sqrt(2.0)
    xs = jnp.stack([jnp.asarray(0.0), c - hh, c + hh, jnp.asarray(P)])
    V = jnp.stack([jnp.broadcast_to(xs[:, None], (4, 4)),
                   jnp.broadcast_to(xs[None, :], (4, 4))], axis=-1)
    return CM.TransfiniteMap._traced(ref_map.u_bounds, ref_map.v_bounds, V,
                                     circle_curves(r, c),
                                     ref_map.singular_vertices)


def circle_stack(M, r=CIRC["r"], n_orders=2, theta=0.0, phi=0.0,
                 eps=CIRC["eps"], depth=CIRC["depth"]):
    """The NumPy stack of the circular pillar through the SHAPES route (its
    own grid at radius ``r``)."""
    from lumenairy.elements.pmm import Circle
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=M,
                        n_orders=n_orders)
    st.add_layer(depth, shapes=[Circle(CIRC["c"], CIRC["c"], r, eps)],
                 background_eps=1.0)
    st.set_source(WL, theta=theta, phi=phi)
    return st


def rel(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def absd(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
