"""D: the twins NOT changed by the fix, measured with the verifier's own
fixtures (one 'defective' family, one 'correct' family, one unmeasured RCWA
entry):

  pmm_jones_1d  d / d(angle) at exactly 0 (and 1e-5 rad), isotropic tensors
                on the verifier grating (ridge 2.3^2 I, groove 1.35^2 I,
                P 0.85, duty 0.4, depth 0.31, n_sup 1, n_sub 1.6, degree 10)
  BOR-SEM       BORStack(basis='sem'), Rbig 2.5, n_hs 1.3, eps 4.5, thk 0.4,
                degree 6, m = 0 and 2, d / d(in-plane anisotropy) at 0
  rcwa_jones_2d the verifier C4v post as isotropic tensors (eps I), n_orders 2,
                d / d(t) for t I added on the x-widening pixels, and t added
                to eps_xy = eps_yx on the post
AD = jit(jacrev); FD = premise-checked Richardson (h halving).
"""
import sys
import warnings

import _fix
from _v import dump, fd_halving, jax, jnp, np, rel

from lumenairy.elements.pmm import pmm_jones_1d
from lumenairy.elements.rcwa import rcwa_jones_2d

out = {}
which = set(sys.argv[1:]) or {"jones1d", "borsem", "jones2d"}

if "jones1d" in which:
    ER = 2.3 ** 2 * np.eye(3, dtype=complex)
    EG = 1.35 ** 2 * np.eye(3, dtype=complex)

    def fj(t, xp):
        o, R, T, J = pmm_jones_1d(0.85, ER, EG, 1.6, 1.0, 0.31, 0.4, 1.0,
                                  angle=(jnp.asarray(t) if xp is jnp
                                         else float(t)),
                                  degree=10, stabilize=False)
        return xp.concatenate([xp.ravel(xp.asarray(R)),
                               xp.ravel(xp.asarray(T))]).real
    for x0 in (0.0, 1e-5, 1e-3):
        g = np.asarray(jax.jit(jax.jacrev(lambda t: fj(t, jnp)))(x0))
        fd, prem = fd_halving(lambda t: fj(t, np), x0, h0=4e-3)
        out[f"pmm_jones_1d_angle_{x0:g}"] = {"rel": rel(g, fd),
                                             "premise": prem,
                                             "fd_max": float(np.max(np.abs(fd)))}
        print("jones1d", x0, out[f"pmm_jones_1d_angle_{x0:g}"], flush=True)

if "borsem" in which:
    from lumenairy.elements.bor.bor_stack import BORStack
    E0 = 4.5

    def solve(m, tri):
        s = BORStack(2.5, m, n_superstrate=1.3, n_substrate=1.3, basis="sem",
                     degree=6, n_mesh_cap=2.6)
        s.add_layer(0.4, segments=[(2.5, tri)])
        s.set_source(wavelength=2 * np.pi / 2.2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return s.solve()

    for m in (0, 2):
        r0 = solve(m, jnp.asarray([E0, E0, E0], dtype=complex))
        inc = np.nonzero(np.asarray(r0["inc_mask"]) > 0.5)[0][:3]

        def fb(d, m=m, inc=inc):
            d = jnp.asarray(d).astype(jnp.complex128)
            r = solve(m, jnp.stack([E0 + d, E0 - d, E0 + 0.0 * d]))
            R, T = jnp.asarray(r["R"]), jnp.asarray(r["T"])
            return jnp.concatenate([jnp.stack([jnp.sum(R), jnp.sum(T)]),
                                    R[inc], T[inc]]).real
        g = np.asarray(jax.jit(jax.jacrev(fb))(0.0))
        fd, prem = fd_halving(lambda x: np.asarray(fb(jnp.asarray(x))), 0.0,
                              h0=2e-2)
        out[f"borsem_m{m}"] = {"rel": rel(g, fd), "premise": prem,
                               "fd_max": float(np.max(np.abs(fd))),
                               "n_inc": int(inc.size)}
        print("borsem", m, out[f"borsem_m{m}"], flush=True)

if "jones2d" in which:
    iso_cell = (_fix.BASE[..., None, None] * np.eye(3)).astype(complex)
    DX = (_fix.D_X[..., None, None] * np.eye(3)).astype(complex)
    XY = np.zeros((24, 24, 3, 3), complex)
    XY[..., 0, 1] = XY[..., 1, 0] = _fix.post
    for pname, D in (("x_widen", DX), ("eps_xy_post", XY)):
        for kw in ({}, {"formulation": "li"}):
            def fr(t, xp, D=D, kw=kw):
                cell = xp.asarray(iso_cell) + t * xp.asarray(D)
                extra = {} if xp is jnp else {"symmetry": False}
                o, R, T, J = rcwa_jones_2d(0.9, 0.9, cell, 1.52, 1.0, 0.37,
                                           1.0, n_orders_x=2, n_orders_y=2,
                                           **kw, **extra)
                return xp.concatenate([xp.ravel(xp.asarray(R)),
                                       xp.ravel(xp.asarray(T))]).real
            g = np.asarray(jax.jit(jax.jacrev(lambda t: fr(t, jnp)))(0.0))
            fd, prem = fd_halving(lambda t: fr(t, np), 0.0, h0=2e-2)
            g1 = np.asarray(jax.jit(jax.jacrev(lambda t: fr(t, jnp)))(1e-4))
            fd1, prem1 = fd_halving(lambda t: fr(t, np), 1e-4, h0=2e-2)
            key = f"rcwa_jones_2d_{pname}_{kw.get('formulation', 'laurent')}"
            out[key] = {"rel": rel(g, fd), "premise": prem,
                        "fd_max": float(np.max(np.abs(fd))),
                        "rel_off_1e-4": rel(g1, fd1), "premise_off": prem1}
            print(key, out[key], flush=True)
dump("d_twins" + ("_" + "_".join(sorted(which)) if len(which) < 3 else ""),
     out)
