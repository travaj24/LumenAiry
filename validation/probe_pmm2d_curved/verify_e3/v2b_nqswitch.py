"""V2b: the frozen twin at a point where the NumPy solver's ADAPTIVE node
count CHANGES (circle, M = 4: _stag_map_nodes 16 -> 32 between r = 0.38 and
0.40, v2a_nq_scan.json).

    python v2b_nqswitch.py

1. bisect r* (the switch) to 1e-7;
2. the NumPy solve on both sides of r* and with the node count FORCED to 16
   / 32 at the same r: the jump = the quadrature error the criterion
   tolerates;
3. the NumPy FD ladder CENTRED on r* (h / P = 1e-2 .. 1e-6): a straddling
   difference is contaminated by jump / 2h;
4. the twin frozen at r0 = 0.36 (16 nodes) evaluated beyond the switch (r* +
   1e-4, 0.42, 0.46, 0.50): AD vs FD(twin), vs FD(numpy forced 16), vs
   FD(numpy forced 32); the value vs NumPy(16) / NumPy(32); the gradient's
   discretisation error FD(numpy, M = 4) - FD(numpy, M = 6) at the same r.
"""
from _ve3 import TS, WL, P, PMM2DStackPure, dump, jax, jnp, ladder, np, tic

from lumenairy.elements.pmm import Circle
from lumenairy.elements.pmm.shapes2d import _merge

M = 4
EPS, DEP = 4.0, 0.5
out = {"M": M}
orig_nodes = TS._stag_map_nodes


def nq_at(r, m=M):
    U, V, cm, _c, _i, _m = _merge(P, P, [("l", [Circle(0.6, 0.6, r, EPS)],
                                          1.0, None)])
    return int(orig_nodes(TS.Basis1D(P, cm.u_walls, m),
                          TS.Basis1D(P, cm.v_walls, m), cm, m))


lo, hi = 0.38, 0.40
assert nq_at(lo) == 16 and nq_at(hi) == 32
while hi - lo > 1e-7:
    mid = 0.5 * (lo + hi)
    if nq_at(mid) == 16:
        lo = mid
    else:
        hi = mid
rs = 0.5 * (lo + hi)
out["r_switch"] = rs
out["nq_below_above"] = [nq_at(rs - 1e-6), nq_at(rs + 1e-6)]
print("r* =", rs, out["nq_below_above"], flush=True)


def build(r, m=M, backend="numpy"):
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=m, n_orders=2, backend=backend)
    st.add_layer(DEP, shapes=[Circle(0.6, 0.6, r, EPS)], background_eps=1.0)
    st.set_source(WL)
    return st


def q(R, T):
    return np.array([R[0, 12], T[0, 12]])


def np_solve(r, force=None, m=M):
    if force is not None:
        TS._stag_map_nodes = lambda *a, **k: force
    try:
        o, R, T, J = build(r, m).solve()
    finally:
        TS._stag_map_nodes = orig_nodes
    return q(R, T)


# 2. the jump
v16 = np_solve(rs, 16)
v32 = np_solve(rs, 32)
out["jump_at_rstar_16_vs_32"] = float(np.max(np.abs(v16 - v32)))
out["np_rstar_minus_plus"] = [np_solve(rs - 1e-9).tolist(),
                              np_solve(rs + 1e-9).tolist()]
# 3. FD centred on r*
steps = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
rows = [((np_solve(rs + h * P) - np_solve(rs - h * P)) / (2 * h * P))
        for h in steps]
rows16 = [((np_solve(rs + h * P, 16) - np_solve(rs - h * P, 16))
           / (2 * h * P)) for h in steps]
out["FD_numpy_straddling"] = [r.tolist() for r in rows]
out["FD_numpy_forced16"] = [r.tolist() for r in rows16]
out["straddle_minus_forced16"] = [float(np.max(np.abs(a - b)))
                                  for a, b in zip(rows, rows16)]
out["steps"] = steps
print("straddle - forced16:", out["straddle_minus_forced16"], flush=True)

# 4. the twin frozen at 0.36
st = build(0.36, backend="jax")
tw = st.jax_twin()
out["twin_frozen_nq"] = int(tw.sol_h._qrule.n) if hasattr(
    tw.sol_h._qrule, "n") else repr(tw.sol_h._qrule)


def f(r):
    p = tw.params()
    p["layers"][0]["shapes"] = [Circle(0.6, 0.6, r, EPS)]
    _o, R, T, J = st.solve(params=p)
    return jnp.stack([R[0, 12], T[0, 12]])


fj = jax.jit(f)
gj = jax.jit(jax.jacrev(f))
res = {}
for r in (rs + 1e-4, 0.42, 0.46, 0.50):
    t = tic()
    ad = np.asarray(gj(r))
    _rows, fdt, _c, rat_t = ladder(fj, r, [1e-3, 3e-4, 1e-4], P)
    _rows, fd16, _c, rat16 = ladder(lambda x: np_solve(x, 16), r,
                                    [1e-3, 3e-4, 1e-4], P)
    _rows, fd32, _c, rat32 = ladder(lambda x: np_solve(x, 32), r,
                                    [1e-3, 3e-4, 1e-4], P)
    _rows, fd6, _c, _r6 = ladder(lambda x: np_solve(x, m=6), r,
                                 [1e-3, 3e-4, 1e-4], P)
    val = np.asarray(fj(r))
    sc = float(np.max(np.abs(fd32)))
    res[repr(r)] = {
        "nq_numpy": nq_at(r), "AD": ad.tolist(), "FD_twin": fdt.tolist(),
        "FD_numpy16": fd16.tolist(), "FD_numpy32": fd32.tolist(),
        "FD_numpy_M6": fd6.tolist(),
        "AD_vs_FDtwin": float(np.max(np.abs(ad - fdt))) / sc,
        "AD_vs_FDnumpy32": float(np.max(np.abs(ad - fd32))) / sc,
        "FDnumpy16_vs_32": float(np.max(np.abs(fd16 - fd32))) / sc,
        "grad_discretisation_M4_vs_M6": float(np.max(np.abs(fd32 - fd6)))
        / sc,
        "value_twin_vs_numpy16": float(np.max(np.abs(val - np_solve(r, 16)))),
        "value_twin_vs_numpy32": float(np.max(np.abs(val - np_solve(r, 32)))),
        "value_numpy16_vs_32": float(np.max(np.abs(np_solve(r, 16)
                                                    - np_solve(r, 32)))),
        "value_discretisation_M4_vs_M6": float(np.max(np.abs(
            np_solve(r) - np_solve(r, m=6)))),
        "s": tic() - t}
    print(r, res[repr(r)], flush=True)
out["twin_beyond_switch"] = res
print(dump("v2b_nqswitch.json", out))
