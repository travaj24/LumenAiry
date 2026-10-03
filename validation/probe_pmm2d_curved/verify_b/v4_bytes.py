"""V4 -- B1 byte identity on THIS verifier's own no-map fixtures (not Phase
A's 109): SHA-256 of every output of a set of unmapped solves, run once on
the PRE tree (``git archive 539ce4a3``) and once on this tree; then compare.

  (cd PRE && PYTHONPATH=PRE python .../v4_bytes.py PRE pre)
  PYTHONPATH=THIS python v4_bytes.py THIS post
  python v4_bytes.py compare
Fixtures (all cmap=None): tensor + magnetic + slanted + absorbing operator
assemblies on non-uniform 4 x 4 walls; a 3-layer conical stack (absorbing,
tensor uniform layer, patterned) with retained internals + absorption; a
per-layer-grids stack with a mortar; the direct pmm_jones / pmm_efficiency
functions; Basis1D matrices.
Output: v4_bytes_<tag>.json, v4_bytes_compare.json
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def h(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(a.tobytes() + str(a.dtype).encode()
                          + str(a.shape).encode()).hexdigest()


def run(root, tag):
    import lumenairy
    assert os.path.normcase(os.path.abspath(lumenairy.__file__)).startswith(
        os.path.normcase(os.path.abspath(root))), lumenairy.__file__
    from lumenairy.elements.pmm import (
        PMM2DStackPure,
        pmm_efficiency_2d_staggered,
        pmm_jones_2d_staggered,
        twod_staggered as TS,
    )
    out = {"lumenairy": lumenairy.__file__}
    P = 1.1
    k0 = 2 * np.pi / 0.95
    wx = np.array([0.0, 0.2, 0.5, 0.8, P])
    wy = np.array([0.0, 0.3, 0.6, 0.85, P])
    rng = np.random.default_rng(7)
    eps = 1.0 + 3.0 * rng.random((4, 4)) + 0.2j * rng.random((4, 4))
    # operators: scalar, tensor, magnetic, slanted
    epsT = np.zeros((4, 4, 3, 3), complex)
    epsT[..., 0, 0] = eps
    epsT[..., 1, 1] = eps * 1.1
    epsT[..., 0, 1] = epsT[..., 1, 0] = 0.15
    epsT[..., 2, 2] = eps * 0.9
    mu = 1.0 + 0.3 * rng.random((4, 4))
    for nm, kw in (("scalar", dict(eps_cell=eps)),
                   ("tensor", dict(eps_cell=epsT)),
                   ("magnetic", dict(eps_cell=eps, mu_cell=mu)),
                   ("slant", dict(eps_cell=eps, slant=(0.2, -0.1)))):
        e = kw.pop("eps_cell")
        try:
            s = TS.Granet2DTransverseE(P, P, wx, wy, 5, e, alpha0x=0.7,
                                       alpha0y=-0.4, k0=k0, **kw)
            for a in ("Lmat", "Rmat", "Stt", "Schur"):
                v = getattr(s, a, None)
                out[f"op_{nm}_{a}"] = None if v is None else h(v)
        except Exception as ex:                           # noqa: BLE001
            out[f"op_{nm}"] = "EXC " + type(ex).__name__ + str(ex)[:60]
    # 3-layer conical stack, integer grid
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.5, n_modes=5,
                        n_orders=3)
    st.add_layer(0.15, eps=2.2 + 0.05j)
    st.add_layer(0.10, eps=np.array([[2.0, 0.1, 0], [0.1, 2.3, 0],
                                     [0, 0, 2.1]], complex))
    cell = np.ones((3, 3), complex)
    cell[1, 1] = 4.0 + 0.1j
    cell[0, 2] = 2.5
    st.add_layer(0.30, eps_cell=cell)
    st.set_source(0.95, theta=0.35, phi=0.6)
    o, R, T, J = st.solve(retain_internal=True)
    out.update(stack3_R=h(R), stack3_T=h(T), stack3_J=h(J), stack3_o=h(o),
               stack3_abs=h(st.layer_absorption()))
    # per-layer grids (a mortar between two different wall sets)
    st2 = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                         n_modes=6, n_orders=3, layer_grids="per-layer")
    st2.add_layer(0.2, eps_cell=np.array([[1.0, 3.0], [3.0, 1.0]], complex),
                  x_walls=[0.55], y_walls=[0.55])
    st2.add_layer(0.25, eps_cell=cell, x_walls=[0.3, 0.8], y_walls=[0.4, 0.7])
    st2.set_source(1.0, theta=0.1, phi=0.2)
    o, R, T = st2.solve(jones=False)
    out.update(stackpl_R=h(R), stackpl_T=h(T))
    # direct functions
    r = pmm_jones_2d_staggered(P, P, cell, 1.45, 1.0, 0.4, 1.0, n_modes=5,
                               n_orders=3, theta=0.2, phi=0.3)
    out["jones_direct"] = [h(np.asarray(x)) for x in (
        r if isinstance(r, tuple) else (r,)) if not isinstance(x, dict)]
    r = pmm_efficiency_2d_staggered(P, P, cell, 1.45, 1.0, 0.4, 1.0,
                                    n_modes=5, n_orders=3, polarization="tm")
    out["eff_direct"] = [h(np.asarray(x)) for x in (
        r if isinstance(r, tuple) else (r,)) if not isinstance(x, dict)]
    b = TS.Basis1D(P, wx, 6, np.exp(-0.3j))
    out["basis"] = [h(getattr(b, a)) for a in sorted(vars(b))
                    if isinstance(getattr(b, a), np.ndarray)]
    with open(os.path.join(HERE, f"v4_bytes_{tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(tag, len(out), "entries", flush=True)


def compare(sfx=""):
    a = json.load(open(os.path.join(HERE, f"v4_bytes_pre{sfx}.json")))
    b = json.load(open(os.path.join(HERE, f"v4_bytes_post{sfx}.json")))
    n = eq = 0
    diff = []
    for k in a:
        if k == "lumenairy":
            continue
        va, vb = a[k], b.get(k)
        va = va if isinstance(va, list) else [va]
        vb = vb if isinstance(vb, list) else [vb]
        for i, (x, y) in enumerate(zip(va, vb)):
            n += 1
            if x == y:
                eq += 1
            else:
                diff.append(f"{k}[{i}]")
    res = {"hashes": n, "identical": eq, "differ": diff,
           "pre": a["lumenairy"], "post": b["lumenairy"]}
    print(res)
    with open(os.path.join(HERE, f"v4_bytes_compare{sfx}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(sys.argv[2] if len(sys.argv) > 2 else "")
    else:
        run(sys.argv[1], sys.argv[2])
