"""V1 -- ``pmm_jones_2d(..., slant=...)`` on the JAX dispatch.

The JAX branch of ``pmm_jones_2d`` never reads ``slant``: it hands off to
``_pmm_jones_2d_cell_jax`` without it, and ``slant = _norm_slant_pair(...)``
is not even reached until AFTER the branch.  So every traced route returns the
VERTICAL answer, silently.

This probe enumerates ALL SEVEN members of the dispatch tuple
``(eps_tensor_cell, n_substrate, n_superstrate, depth, wavelength, theta,
phi)`` and records, per route, whether it SOLVES or REFUSES and -- when it
solves -- how it reads against the NumPy VERTICAL and the NumPy SLANTED
answers.  Three controls must keep solving on both arms:

  * a VERTICAL traced solve (``slant=None``) -- bit-identical hashes;
  * a CONSTANT-TILE "slanted" cell, whose slant is a pure coordinate change of
    a uniform medium (measured here, not assumed) -- bit-identical hashes;
  * ``jax.grad`` AD against a central FD on a vertical traced depth.
"""
from __future__ import annotations

import math
import sys
import warnings

import _lib as L
import numpy as np

PX = PY = 0.85e-6
WL = 0.58e-6
DEPTH = 0.42e-6
NSUP, NSUB = 1.0, 1.5
TSL = 0.5
TH, PH = math.radians(25.0), math.radians(0.0)
NORD, DEG = 3, 5

# a 6 x 4 x-ASYMMETRIC, y-varying scalar cell, promoted to (6, 4, 3, 3)
BASE = np.array([
    [2.10, 2.10, 1.30, 1.30],
    [3.05, 2.60, 1.30, 1.72],
    [1.30, 1.30, 1.30, 1.30],
    [1.30, 2.44, 2.44, 1.30],
    [1.30, 1.30, 1.95, 1.95],
    [1.30, 1.30, 1.30, 1.30],
], dtype=float)
LAYOUT = np.arange(BASE.size).reshape(BASE.shape)


def _tensor(scal):
    T = np.zeros(np.shape(scal) + (3, 3), dtype=complex)
    for i in range(3):
        T[..., i, i] = scal
    return T


CELL = _tensor(BASE)
CONST = _tensor(np.full(BASE.shape, 2.20))


def _call(*, cell=CELL, n_sub=NSUB, n_sup=NSUP, depth=DEPTH, wl=WL,
          theta=TH, phi=PH, slant=(TSL, 0.0)):
    from lumenairy.elements.pmm.twod_jones import pmm_jones_2d
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pmm_jones_2d(PX, PY, cell, n_sub, n_sup, depth, wl,
                            theta=theta, phi=phi, n_orders=NORD, degree=DEG,
                            slant=slant, region_layout=LAYOUT)


def _cmp(a, b):
    return dict(dR=L.dmax(a[1], b[1]), dT=L.dmax(a[2], b[2]),
                dJones=L.dmax(a[3], b[3]))


def _hashes(out):
    return dict(orders=L.sha(out[0]), R=L.sha(out[1]), T=L.sha(out[2]),
                J=L.sha(out[3]))


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    res = {}
    np_sl = _call()
    np_vt = _call(slant=None)
    res["numpy_slanted_vs_vertical"] = _cmp(np_sl, np_vt)

    # ---- the seven traced routes -------------------------------------------
    routes = {
        "eps_tensor_cell": dict(cell=jnp.asarray(CELL)),
        "n_substrate": dict(n_sub=jnp.asarray(NSUB)),
        "n_superstrate": dict(n_sup=jnp.asarray(NSUP)),
        "depth": dict(depth=jnp.asarray(DEPTH)),
        "wavelength": dict(wl=jnp.asarray(WL)),
        "theta": dict(theta=jnp.asarray(TH)),
        "phi": dict(phi=jnp.asarray(PH)),
    }
    out = {}
    for name, kw in routes.items():
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                r = _call(**kw)
            out[name] = dict(outcome="SOLVED_SILENTLY",
                             warnings=[str(x.message)[:100] for x in w],
                             vs_numpy_VERTICAL=_cmp(r, np_vt),
                             vs_numpy_SLANTED=_cmp(r, np_sl))
        except NotImplementedError as e:
            out[name] = dict(outcome="REFUSED", msg=str(e)[:400])
        except Exception as e:                            # noqa: BLE001
            out[name] = dict(outcome="RAISE_OTHER", exc=type(e).__name__,
                             msg=str(e)[:300])
    res["routes"] = out

    # ---- controls -----------------------------------------------------------
    ctl = {}
    # (i) the VERTICAL traced solve
    try:
        v = _call(depth=jnp.asarray(DEPTH), slant=None)
        ctl["vertical_traced_depth"] = dict(
            outcome="SOLVED", hashes=_hashes(v), vs_numpy=_cmp(v, np_vt))
    except Exception as e:                                # noqa: BLE001
        ctl["vertical_traced_depth"] = dict(outcome="RAISE",
                                            exc=type(e).__name__,
                                            msg=str(e)[:300])
    # (ii) a CONSTANT-TILE slanted cell.  First MEASURE that the slant really
    #      is a no-op there on the NumPy path (all four returns are
    #      anchor-free: R / T are unimodular-invariant and the Jones is the
    #      REFLECTION one).
    c_sl = _call(cell=CONST)
    c_vt = _call(cell=CONST, slant=None)
    ctl["const_tile_numpy_slant_is_a_noop"] = _cmp(c_sl, c_vt)
    ctl["const_tile_numpy_hashes_equal"] = (_hashes(c_sl) == _hashes(c_vt))
    try:
        cj = _call(cell=CONST, depth=jnp.asarray(DEPTH))
        ctl["const_tile_traced_depth"] = dict(
            outcome="SOLVED", hashes=_hashes(cj), vs_numpy_slanted=_cmp(cj, c_sl),
            vs_numpy_vertical=_cmp(cj, c_vt))
    except Exception as e:                                # noqa: BLE001
        ctl["const_tile_traced_depth"] = dict(outcome="RAISE",
                                              exc=type(e).__name__,
                                              msg=str(e)[:300])
    # (iii) AD vs FD on a VERTICAL traced depth
    try:
        from lumenairy.elements.pmm.twod_jones import pmm_jones_2d

        def f(d):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                o = pmm_jones_2d(PX, PY, CELL, NSUB, NSUP, d, WL, theta=TH,
                                 phi=PH, n_orders=NORD, degree=DEG,
                                 slant=None, region_layout=LAYOUT)
            return o[2].sum()
        g = float(jax.grad(f)(jnp.asarray(DEPTH)))
        h = 1e-11
        fd = float((f(jnp.asarray(DEPTH + h)) - f(jnp.asarray(DEPTH - h)))
                   / (2 * h))
        ctl["grad_vertical_traced_depth"] = dict(
            ad=g, fd=fd, rel=abs(g - fd) / max(abs(fd), 1e-30))
    except Exception as e:                                # noqa: BLE001
        ctl["grad_vertical_traced_depth"] = dict(outcome="RAISE",
                                                 exc=type(e).__name__,
                                                 msg=str(e)[:300])
    res["controls"] = ctl

    import jax as _j
    res["jax_version"] = _j.__version__
    for k, v in res["routes"].items():
        print("%-18s %-16s %s" % (k, v["outcome"],
                                  v.get("vs_numpy_SLANTED", "")))
    print("controls:")
    for k, v in ctl.items():
        print("   %-34s %s" % (k, v))
    L.dump("v1_routes", res,
           suffix=(sys.argv[1] if len(sys.argv) > 1 else ""))


if __name__ == "__main__":
    main()
