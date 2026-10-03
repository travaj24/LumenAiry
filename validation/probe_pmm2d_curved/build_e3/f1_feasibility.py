"""E3 FEASIBILITY PROBE (deliverable 1, run BEFORE the rest of the twin).

    python f1_feasibility.py rect|circle M [theta_deg phi_deg]

rect   -- the 3 x 3 rectangular pillar (w 0.5, h 0.4 in a 1.2 cell, eps 4,
          depth 0.4, wl 1.0, n 1.0 / 1.45): forward parity of the twin vs the
          NumPy stack (R / T / Jones), the gradient of R00 (and T00) w.r.t.
          the pillar WIDTH vs converged central finite differences, compile
          and run times.  The width enters as a TRACED transfinite map on the
          frozen 3 x 3 grid (the interior vertex images move; straight edges,
          so every cell is affine).
circle -- the Phase B 3 x 3 circle (r 0.36, eps 4, depth 0.5): the same with
          the gradient w.r.t. the RADIUS -- the arcs and the vertex images are
          functions of r, the Jacobian is analytic, the four corner (Duffy)
          cells keep their rule.

Two finite-difference references per parameter:
  FD(twin)  central differences of the twin's own forward (the SAME discrete
            function the gradient differentiates -- frozen grid, frozen node
            counts);
  FD(numpy) central differences of the shipped NumPy solve, whose grid moves
            with the parameter (the Rect / Circle shape route).
Each on a step ladder; the converged value is the rung pair with the
smallest change, and convergence (a rung-to-rung change below the AD-vs-FD
gap) is recorded as the premise.  Output f1_<case>_M<M>[_obl].json.
"""
import sys

from _e3common import (  # noqa
    CIRC,
    CM,
    DEG,
    RECT,
    P,
    StagJaxTwin,
    absd,
    circle_stack,
    circle_traced_map,
    dump,
    jax,
    jnp,
    np,
    rect_stack,
    rect_traced_map,
    rel,
    tic,
)

case = sys.argv[1]
M = int(sys.argv[2])
th = float(sys.argv[3]) * DEG if len(sys.argv) > 3 else 0.0
ph = float(sys.argv[4]) * DEG if len(sys.argv) > 4 else 0.0
out = {"case": case, "M": M, "theta": th, "phi": ph}

if case == "rect":
    p_ref = RECT["w"]

    def np_stack(w):
        return rect_stack(M, w=w, theta=th, phi=ph)
    st0 = np_stack(p_ref)
    t = tic()
    o, R, T, J = st0.solve()
    out["numpy_solve_s"] = tic() - t
    # (a) the UNMAPPED twin (the kron route; traced materials only)
    t = tic()
    tw_s = StagJaxTwin(np_stack(p_ref), geometry="static")
    out["template_static_s"] = tic() - t
    _o, Rs, Ts, Js = tw_s.solve()
    out["parity_static"] = {"R": absd(Rs, R), "T": absd(Ts, T),
                            "J": absd(Js, J)}
    # (b) the MAPPED twin (identity transfinite map on the rect walls)
    t = tic()
    tw = StagJaxTwin(np_stack(p_ref), geometry="mapped")
    out["template_mapped_s"] = tic() - t
    _o, Rm, Tm, Jm = tw.solve()
    out["parity_mapped_vs_unmapped_numpy"] = {
        "R": absd(Rm, R), "T": absd(Tm, T), "J": absd(Jm, J)}
    # the NumPy solve of the SAME discretisation (explicit identity map)
    xw, yw = tw.cmap_ref.u_bounds, tw.cmap_ref.v_bounds
    stI = rect_stack(M, w=p_ref, theta=th, phi=ph,
                     cmap=CM.TransfiniteMap(xw, yw))
    _o, RI, TI, JI = stI.solve()
    out["parity_mapped_vs_mapped_numpy"] = {
        "R": absd(Rm, RI), "T": absd(Tm, TI), "J": absd(Jm, JI)}
    out["numpy_mapped_vs_unmapped"] = {"R": absd(RI, R), "T": absd(TI, T),
                                       "J": absd(JI, J)}

    def build_map(w):
        return rect_traced_map(tw.cmap_ref, w)
elif case == "circle":
    p_ref = CIRC["r"]

    def np_stack(r):
        return circle_stack(M, r=r, theta=th, phi=ph)
    st0 = np_stack(p_ref)
    t = tic()
    o, R, T, J = st0.solve()
    out["numpy_solve_s"] = tic() - t
    t = tic()
    tw = StagJaxTwin(np_stack(p_ref))
    out["template_mapped_s"] = tic() - t
    _o, Rm, Tm, Jm = tw.solve()
    out["parity_mapped_vs_mapped_numpy"] = {
        "R": absd(Rm, R), "T": absd(Tm, T), "J": absd(Jm, J)}

    def build_map(r):
        return circle_traced_map(tw.cmap_ref, r)
else:
    raise SystemExit("case must be rect or circle")

p0 = tw.p0


def f(x):
    _o, Rx, Tx, Jx = tw.solve(cmap=build_map(x))
    return jnp.stack([Rx[0, p0], Tx[0, p0]])


# the traced map AT the reference value is the reference map
_o, Rr, Tr, Jr = tw.solve(cmap=build_map(jnp.asarray(p_ref)))
out["traced_map_at_ref_vs_ref_map"] = {"R": absd(Rr, Rm), "T": absd(Tr, Tm),
                                       "J": absd(Jr, Jm)}
fj = jax.jit(f)
t = tic()
v0 = np.asarray(fj(p_ref))
out["jit_forward_first_s"] = tic() - t
t = tic()
v0b = np.asarray(fj(p_ref + 1e-9))
out["jit_forward_second_s"] = tic() - t
out["value_at_ref"] = v0.tolist()
gj = jax.jit(jax.jacrev(f))
t = tic()
g = np.asarray(gj(p_ref))
out["jit_grad_first_s"] = tic() - t
t = tic()
g2 = np.asarray(gj(p_ref + 1e-9))
out["jit_grad_second_s"] = tic() - t
out["AD"] = g.tolist()
del v0b, g2

steps = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
fd_twin, fd_np = [], []
for hstep in steps:
    h = hstep * P
    fp = np.asarray(fj(p_ref + h))
    fm = np.asarray(fj(p_ref - h))
    fd_twin.append(((fp - fm) / (2 * h)).tolist())
    _o, Rp, Tp, _J = np_stack(p_ref + h).solve()
    _o, Rn, Tn, _J = np_stack(p_ref - h).solve()
    fd_np.append([float((Rp[0, p0] - Rn[0, p0]) / (2 * h)),
                  float((Tp[0, p0] - Tn[0, p0]) / (2 * h))])
fd_twin = np.asarray(fd_twin)
fd_np = np.asarray(fd_np)
out["steps_over_P"] = steps
out["FD_twin"] = fd_twin.tolist()
out["FD_numpy"] = fd_np.tolist()


def converged(fd):
    ch = np.max(np.abs(np.diff(fd, axis=0)), axis=1)
    k = int(np.argmin(ch))
    return fd[k + 1], float(ch[k]), k


for nm, fd in (("twin", fd_twin), ("numpy", fd_np)):
    best, change, k = converged(fd)
    out[f"FD_{nm}_converged"] = best.tolist()
    out[f"FD_{nm}_rung_change"] = change
    out[f"FD_{nm}_rung"] = [steps[k], steps[k + 1]]
    out[f"AD_vs_FD_{nm}_rel"] = (np.abs(g - best) / np.abs(best)).tolist()
print(dump(f"f1_{case}_M{M}{'_obl' if th else ''}.json", out) or "")
for k in ("parity_static", "parity_mapped_vs_unmapped_numpy",
          "parity_mapped_vs_mapped_numpy", "traced_map_at_ref_vs_ref_map",
          "AD", "FD_twin_converged", "FD_numpy_converged", "AD_vs_FD_twin_rel",
          "AD_vs_FD_numpy_rel", "FD_twin_rung_change", "FD_numpy_rung_change",
          "jit_forward_first_s", "jit_forward_second_s", "jit_grad_first_s",
          "jit_grad_second_s", "numpy_solve_s"):
    if k in out:
        print(k, out[k])
