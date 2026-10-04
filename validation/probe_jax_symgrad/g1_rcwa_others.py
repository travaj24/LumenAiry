"""G1: the other RCWA JAX entries at the four-fold symmetric pixel cell --
``rcwa_jones_2d`` (tensor cell) and ``RCWAStack`` (JAX eps_cell) -- AD
(jit(jacrev)) vs a premise-checked NumPy Richardson FD (ladder 1e-3 / 3e-4 /
1e-4, Richardson of the last two; h^2 premise ratio -> 11.375, see
``_dcommon.premise_of``), on whichever tree ``LUM_TREE`` names.

    python g1_rcwa_others.py

Cell (a1's): 15 x 15 pixels, centre block eps 4, side blocks 1.5, corners 1.
n_orders 3 x 3, P 1.2, wl 1, depth 0.45, n_substrate 1, n_superstrate 1.45
(the stated values; NB a1_rcwa.py's call passes 1.45 as n_SUBSTRATE).

Cases
  jones_x      rcwa_jones_2d, eps * I tensor cell, t * I on the two x-side
               blocks (breaks C4 -> C2v): R, T of (0,0), (+-1,0), both
               incident columns (x, y) -> 12 outputs.
  jones_all    the same, t on all four side blocks (symmetry-KEEPING control).
  jones_conical  jones_x at theta 0.2, phi 0.3.
  stack1_x     RCWAStack, one patterned layer (scalar eps_cell), t on the
               x-sides; rows x / y of R, T (x = TM, y = TE at phi 0).
  stack2_x     the same + a uniform spacer eps 2.25, thickness 0.1.
  stack1_all   stack1 with t on all four sides (symmetry-keeping control).
  stack2p_x    the cell + a second patterned layer (the unperturbed base
               cell, thickness 0.2); t on layer 1's x-sides.
NumPy FD runs with symmetry=False (the full solve, no even-sector fold).
NB a TRACED tensor cell goes to rcwa_jones_2d's GENERAL (out-of-plane) branch
(it cannot be inspected for off-plane components), so every jones gradient
here is the out-of-plane path; the in-plane branch runs only on concrete JAX
input (forward; see g2).
Per case: the eig spectra of the jit route (exact-cluster members at gap
<= 1e-12 max|lam|), forward parity JAX-jit vs NumPy, the sweep, and the
gauge test (two seeds).
"""
import traceback

from _dcommon import capture, gauge, parity, sweep
from _h import dump, jax, jnp, np

import lumenairy.elements.rcwa._core as RCcore
from lumenairy.elements.rcwa import (
    RCWAStack,
    rcwa_efficiency_2d_shapes,
    rcwa_jones_2d,
)

P, WL, D = 1.2, 1.0, 0.45
NSUB, NSUP = 1.0, 1.45
base3 = np.array([[1.0, 1.5, 1.0], [1.5, 4.0, 1.5], [1.0, 1.5, 1.0]])
xs3 = np.zeros((3, 3))
xs3[0, 1] = xs3[2, 1] = 1.0
all3 = np.zeros((3, 3))
all3[0, 1] = all3[2, 1] = all3[1, 0] = all3[1, 2] = 1.0
BASE, DX, DALL = (np.kron(a, np.ones((5, 5))) for a in (base3, xs3, all3))
I3 = np.eye(3)
ORD = ((0, 0), (1, 0), (-1, 0))


def _idx(o):
    o = np.asarray(o)
    return [int(np.nonzero((o[:, 0] == a) & (o[:, 1] == b))[0][0])
            for a, b in ORD]


def pack(R, T, idx, xp):
    return xp.concatenate([xp.stack([R[p][i] for i in idx]) for p in (0, 1)]
                          + [xp.stack([T[p][i] for i in idx]) for p in (0, 1)])


# output layout: R_x(3) R_y(3) T_x(3) T_y(3)
COLS = {"x": [0, 1, 2, 6, 7, 8], "y": [3, 4, 5, 9, 10, 11]}


def jones(t, xp, dirn, theta=0.0, phi=0.0):
    eps = (xp.asarray(BASE) + t * xp.asarray(dirn)).astype(complex)
    cell = eps[:, :, None, None] * xp.asarray(I3)
    kw = {} if xp is jnp else {"symmetry": False}
    return rcwa_jones_2d(P, P, cell, NSUB, NSUP, D, WL, theta=theta, phi=phi,
                         n_orders_x=3, n_orders_y=3, **kw)


def stack(t, xp, dirn, second=None):
    if xp is jnp and CLEAR[0]:
        # work-around of the homogeneous-mode cache leak (section 0): the
        # module cache keeps arrays built INSIDE a jit trace; a later trace
        # with the same geometry key reads a dead tracer.  Cleared at trace
        # time so each trace computes its own half-space modes.
        RCcore._clear_rcwa_caches()
    eps = (xp.asarray(BASE) + t * xp.asarray(dirn)).astype(complex)
    st = RCWAStack(P, period_y=P, n_superstrate=NSUP, n_substrate=NSUB,
                   n_orders=3, n_orders_y=3)
    st.add_layer(D, eps_cell=eps)
    if second == "spacer":
        st.add_layer(0.1, eps=2.25)
    elif second == "patterned":
        st.add_layer(0.2, eps_cell=xp.asarray(BASE).astype(complex))
    st.set_source(WL)
    res = st.solve(symmetry=False) if xp is np else st.solve()
    o, R, T = res.efficiencies()
    return o, R, T


def make(kind, **kw):
    runner = jones if kind == "jones" else stack
    o = runner(0.0, np, **kw)[0]
    idx = _idx(o)

    def fj(t):
        r = runner(t, jnp, **kw)
        return pack(r[1], r[2], idx, jnp)

    def fn(t):
        r = runner(float(t), np, **kw)
        return np.asarray(pack(r[1], r[2], idx, np))
    return fj, fn


CASES = {
    "jones_x": ("jones", {"dirn": DX}, (0.0, 1e-6, 1e-3)),
    "jones_all": ("jones", {"dirn": DALL}, (0.0,)),
    "jones_conical": ("jones", {"dirn": DX, "theta": 0.2, "phi": 0.3},
                      (0.0,)),
    "stack1_x": ("stack", {"dirn": DX}, (0.0, 1e-6)),
    "stack1_all": ("stack", {"dirn": DALL}, (0.0,)),
    "stack2_x": ("stack", {"dirn": DX, "second": "spacer"}, (0.0, 1e-6)),
    "stack2p_x": ("stack", {"dirn": DX, "second": "patterned"}, (0.0,)),
}

CLEAR = [False]
out = {"cases": {}}
# 0. the RCWAStack homogeneous-mode cache vs two successive jit traces
# (fresh cache, NO work-around): jit(f) then jit(jacrev(f)) at the same point.
RCcore._clear_rcwa_caches()
_fj, _fn = make("stack", dirn=DX)
leak = {}
for lab, fn_ in (("jit_f", lambda: jax.jit(_fj)(jnp.asarray(0.0))),
                 ("jit_jacrev_f", lambda: jax.jit(jax.jacrev(_fj))(
                     jnp.asarray(0.0))),
                 ("eager_f", lambda: _fj(jnp.asarray(0.0)))):
    try:
        jax.block_until_ready(fn_())
        leak[lab] = "ok"
    except Exception as e:
        leak[lab] = f"{type(e).__name__}: " + str(e).splitlines()[0][:200]
    print("cache-leak repro", lab, leak[lab], flush=True)
out["stack_cache_leak_repro"] = leak
CLEAR[0] = True
RCcore._clear_rcwa_caches()
for name, (kind, kw, xs) in CASES.items():
    rec = {}
    try:
        fj, fn = make(kind, **kw)
        sp = capture(lambda fj=fj: jax.jit(fj)(jnp.asarray(0.0)))
        rec["spectrum"] = sp
        print("spectrum", name, [(r["n"], round(r["max_abs"], 3),
                                  "%.1e" % r["min_rel_gap"],
                                  r["members_below_1e-12"],
                                  r["members_below_1e-8"]) for r in sp],
              flush=True)
        rec["parity"] = parity(fj, fn, 0.0)
        rec["sweep"] = sweep(fj, fn, xs, label=name)
        for x, s in rec["sweep"].items():
            g, fd = np.asarray(s["AD"]), np.asarray(s["FD"])
            s["rel_err_by_col"] = {
                c: float(np.max(np.abs(g[ii] - fd[ii]))
                         / max(np.max(np.abs(fd[ii])), 1e-300))
                for c, ii in COLS.items()}
            print("  ", name, "x=" + x, "by col", {
                c: "%.2e" % v for c, v in s["rel_err_by_col"].items()},
                flush=True)
        rec["gauge"] = gauge(fj, 0.0)
        print("parity %.2e" % rec["parity"], "gauge", rec["gauge"],
              flush=True)
    except Exception as e:  # report, do not retry
        rec["error"] = "".join(traceback.format_exception(e))[-3000:]
        print("ERROR", name, rec["error"], flush=True)
    out["cases"][name] = rec

# rcwa_efficiency_2d_shapes: no JAX path (expected NotImplementedError)
try:
    rcwa_efficiency_2d_shapes(
        P, P, jnp.asarray(1.0 + 0j),
        [{"shape": "rectangle", "eps": 4.0, "size": (0.4, 0.4),
          "center": (0.6, 0.6)}], NSUB, NSUP, D, WL, n_orders_x=3,
        n_orders_y=3)
    out["shapes_jax"] = "NO EXCEPTION"
except Exception as e:
    out["shapes_jax"] = f"{type(e).__name__}: {e}"
print("shapes_jax", out["shapes_jax"], flush=True)
print(dump("g1_rcwa_others.json", out))
