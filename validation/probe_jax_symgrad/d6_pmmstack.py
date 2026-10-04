"""D6: the 1-D PMM STACK JAX twins (pmm/_jax_stack.py) -- ``PMMStack.solve``
with a traced angle, d / d(angle) at EXACTLY normal incidence.

* ``_pmm_stack_solve_jax`` (layer_grids='shared'; eig site ~315: the shared
  geometric half-space ``Kx2`` + every layer's ``Mbig`` through
  ``_jpmm_sem_modes_tensor``),
* ``_pmm_stack_solve_jax_perlayer`` (layer_grids='per-layer'; eig site ~566:
  the half-space ``Kx2`` on the FIRST and LAST layer grids + every layer's
  ``Mbig``).

Configurations (metres; P 1.2 um, wl 1 um, n_sup 1.45, n_sub 1, degree 12):
  shared_1L    one layer [(0.5, 4), (0.5, 1)], t 0.45 um  (the d5 fixture)
  shared_2L    + layer 2 [(0.1, 1), (0.3, 2.25), (0.6, 1)], t 0.15 um
               (both layers mirror-symmetric about x = 0.25 P)
  shared_spacer  L1 (t 0.3 um) + a UNIFORM spacer eps 2.25, t 0.1 um: the
               spacer's coupled ``Mbig`` (= eps I - blockdiag(Kx2, Kx2) on
               the shared mesh) carries the same exact +-m pairs as Kx2
  perlayer_3L  + layer 3 [(0.2, 1), (0.1, 4), (0.7, 1)], t 0.1 um, on
               per-layer grids (window +-1: three DIFFERENT grids, the
               half-spaces on the first / last; mortar interfaces)
               (all layers mirror-symmetric about x = 0.25 P; the layer-3
               WINDOW grid {0, .1, .2, .3, .4} is NOT -- no wall at .5 --
               so the discretization breaks the mirror identity there at
               ~7e-5 relative, the FD row shows it)
Outputs: R and T of the +-1 orders for incident Ex and Ey (8); mirror
identity d R_{+1} = - d R_{-1} at normal incidence.
Control: d / d(thickness of layer 1) at angle 0.
"""
from _dcommon import capture, gauge, parity, sweep
from _h import dump, jax, jnp, np

from lumenairy.elements.pmm import PMMStack

P, WL = 1.2e-6, 1.0e-6
L1 = [(0.5, 4.0), (0.5, 1.0)]
L2 = [(0.1, 1.0), (0.3, 2.25), (0.6, 1.0)]
L3 = [(0.2, 1.0), (0.1, 4.0), (0.7, 1.0)]
CFG = {"shared_1L": ("shared", [(0.45e-6, L1)]),
       "shared_2L": ("shared", [(0.3e-6, L1), (0.15e-6, L2)]),
       "shared_spacer": ("shared", [(0.3e-6, L1), (0.1e-6, [(1.0, 2.25)])]),
       "perlayer_3L": ("per-layer", [(0.3e-6, L1), (0.15e-6, L2),
                                     (0.1e-6, L3)])}
MIRROR = [(0, 1), (2, 3), (4, 5), (6, 7)]


def solve(grids, layers, angle, t1=None):
    st = PMMStack(P, n_substrate=1.0, n_superstrate=1.45, degree=12,
                  layer_grids=grids)
    for i, (t, segs) in enumerate(layers):
        st.add_layer(t1 if (i == 0 and t1 is not None) else t,
                     segments=segs)
    st.set_source(WL, angle=angle)
    return st.solve()


_o, *_r = solve("shared", CFG["shared_1L"][1], 0.0)
_o = np.asarray(_o)
IDX = [int(np.nonzero(_o == m)[0][0]) for m in (1, -1)]


def pack(R, T, xp):
    return xp.concatenate([xp.stack([R[p][i] for p in (0, 1) for i in IDX]),
                           xp.stack([T[p][i] for p in (0, 1) for i in IDX])])


def f_angle(grids, layers, xp):
    def f(a):
        _o, R, T, _J = solve(grids, layers, a if xp is jnp else float(a))
        return pack(R, T, xp)
    return f


def f_t1(grids, layers, xp):
    def f(t):
        _o, R, T, _J = solve(grids, layers,
                             jnp.asarray(0.0) if xp is jnp else 0.0,
                             t1=(t if xp is jnp else float(t)))
        return pack(R, T, xp)
    return f


out = {"orders_idx": IDX, "spectrum": {}, "parity": {}, "sweep": {},
       "gauge": {}}
for name, (grids, layers) in CFG.items():
    o2 = np.asarray(solve(grids, layers, 0.0)[0])
    assert [int(np.nonzero(o2 == m)[0][0]) for m in (1, -1)] == IDX
    for a in (0.0, 1e-5):
        sp = capture(lambda a=a, g=grids, ly=layers: jax.jit(
            f_angle(g, ly, jnp))(jnp.asarray(a)))
        out["spectrum"][f"{name}_{a!r}"] = sp
        print("spectrum", name, a, [(r["n"], round(r["max_abs"], 3),
                                     "%.1e" % r["min_rel_gap"],
                                     r["members_below_1e-12"],
                                     r["members_below_1e-8"]) for r in sp],
              flush=True)
    fj, fn = f_angle(grids, layers, jnp), f_angle(grids, layers, np)
    rec = {"parity": parity(fj, fn, 0.0)}
    rec["sweep"] = sweep(fj, fn, (0.0, 1e-5, 1e-3), mirror_pairs=MIRROR,
                         label=name + " angle")
    rec["gauge"] = gauge(fj, 0.0)
    print("parity", rec["parity"], "gauge", rec["gauge"], flush=True)
    t0 = layers[0][0]
    fj, fn = f_t1(grids, layers, jnp), f_t1(grids, layers, np)
    rec["t1_sweep"] = sweep(fj, fn, (t0,), scale=t0, label=name + " t1")
    out["sweep"][name] = rec
print(dump("d6_pmmstack.json", out))
