"""D5: the JAX ``pmm_jones_1d`` twin (``_pmm_jones_1d_jax`` ->
``_jpmm_jones_solve``, pmm/_core.py), d / d(angle) at EXACTLY normal
incidence on a symmetric binary grating.

P 1.2, ridge eps 4 I / groove I (in-plane isotropic tensors), n_sup 1.45,
n_sub 1, depth 0.45, duty 0.5, wl 1, degree 12, stabilize=False.  Outputs:
R and T of the +-1 orders for incident Ex and Ey (8).  Two eig sites per
solve: the layer's coupled 2n ``Mbig`` (``_jpmm_sem_modes_tensor``) and the
half-spaces' shared geometric ``Kx2`` (``_juniform_geo_eig``).

Mirror identity at normal incidence on a symmetric grating:
d R_{+1} / d angle = - d R_{-1} / d angle (and for T), per polarization.
Control: d / d(depth) at angle 0 (symmetry-keeping).
"""
from _dcommon import capture, gauge, parity, sweep
from _h import dump, jax, jnp, np

from lumenairy.elements.pmm import pmm_jones_1d

P, WL = 1.2, 1.0
ER, EG = 4.0 * np.eye(3, dtype=complex), np.eye(3, dtype=complex)
o, *_ = pmm_jones_1d(P, ER, EG, 1.0, 1.45, 0.45, 0.5, WL, degree=12,
                     stabilize=False)
o = np.asarray(o)
IDX = [int(np.nonzero(o == m)[0][0]) for m in (1, -1)]
# output layout: [R_x(+1), R_x(-1), R_y(+1), R_y(-1), T_x(+1), ...]
MIRROR = [(0, 1), (2, 3), (4, 5), (6, 7)]


def pack(R, T, xp):
    return xp.concatenate([xp.stack([R[p][i] for p in (0, 1) for i in IDX]),
                           xp.stack([T[p][i] for p in (0, 1) for i in IDX])])


def f_angle(xp):
    def f(t):
        _o, R, T, _J = pmm_jones_1d(
            P, ER, EG, 1.0, 1.45, 0.45, 0.5, WL,
            angle=(t if xp is jnp else float(t)), degree=12, stabilize=False)
        return pack(R, T, xp)
    return f


def f_depth(xp):
    def f(d):
        _o, R, T, _J = pmm_jones_1d(
            P, ER, EG, 1.0, 1.45, (d if xp is jnp else float(d)), 0.5, WL,
            angle=(jnp.asarray(0.0) if xp is jnp else 0.0), degree=12,
            stabilize=False)
        return pack(R, T, xp)
    return f


out = {"orders_idx": IDX, "spectrum": {}, "parity": {}, "sweep": {},
       "gauge": {}}
for a in (0.0, 1e-5, 1e-3):
    sp = capture(lambda a=a: jax.jit(f_angle(jnp))(jnp.asarray(a)))
    out["spectrum"][f"angle_{a!r}"] = sp
    print("spectrum", a, [(r["n"], round(r["max_abs"], 3),
                           "%.1e" % r["min_rel_gap"],
                           r["members_below_1e-12"]) for r in sp], flush=True)

fj, fn = f_angle(jnp), f_angle(np)
out["parity"]["angle"] = parity(fj, fn, 0.0)
out["sweep"]["angle"] = sweep(fj, fn, (0.0, 1e-5, 1e-3), mirror_pairs=MIRROR,
                              label="angle")
out["gauge"]["angle"] = gauge(fj, 0.0)
print("parity", out["parity"]["angle"], "gauge", out["gauge"]["angle"],
      flush=True)

fj, fn = f_depth(jnp), f_depth(np)
out["parity"]["depth"] = parity(fj, fn, 0.45)
out["sweep"]["depth"] = sweep(fj, fn, (0.45,), label="depth")
out["gauge"]["depth"] = gauge(fj, 0.45)
print("parity", out["parity"]["depth"], "gauge", out["gauge"]["depth"],
      flush=True)
print(dump("d5_pmmjones1d.json", out))
