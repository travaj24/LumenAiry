"""B5 -- the three JAX twins: forward answers, NumPy parity, and gradients.

The three JAX modules (``_jax_twod``, ``_jax_stack2d``, ``_jax_twod_jones``)
each define ``_sqrt_decay`` in a NESTED scope, so unlike the NumPy copies they
cannot be A/B-ed by monkeypatch inside one interpreter.  This probe therefore
records the ABSOLUTE numbers, and the PRE/POST comparison is made ACROSS the
two runs of the same file (before and after the consolidation), keyed on the
arm stamp each JSON carries.

Three claims are measured:

* FORWARD -- each traced surface's per-order (R, T) and its lossless closure;
* PARITY -- the JAX twin against its NumPy sibling on the same geometry, which
  is the claim the consolidation must not weaken (the twins must trace the SAME
  body the NumPy path executes);
* GRADIENT -- ``jax.grad`` of a scalar functional of the solve, against a
  central finite difference of the same functional, so a change to the branch
  test that broke differentiability or moved a derivative would show.

Usage:  OPENBLAS_NUM_THREADS=<n> PYTHONPATH=. python b5_jax.py <out.json>
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_fixtures as F  # noqa: E402

WL, PX, D = F.WL, F.PX, F.DEPTH
NS, NS_OFF, HOST = 1.5, 1.63, 2.25
WEAK = HOST * (1.0 + 1e-6)
S = 6
LAYOUT = np.zeros((S, S), dtype=np.int64)
LAYOUT[2:4, 2:4] = 1


def _np_cell(pillar, host=HOST):
    c = np.full((S, S), host + 0j)
    c[2:4, 2:4] = pillar
    return c


def main():
    F.require_local_tree()
    out = sys.argv[1] if len(sys.argv) > 1 else "b5.json"
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    from lumenairy.elements.pmm import (
        PMM2DStackHybrid,
        pmm_efficiency_2d_cell,
        pmm_jones_2d,
    )

    def jcell(pillar, host=HOST):
        c = jnp.asarray(_np_cell(0.0, 0.0))
        c = c.at[LAYOUT == 0].set(jnp.asarray(host + 0j))
        c = c.at[LAYOUT == 1].set(jnp.asarray(pillar))
        return c

    res = {}

    def record(name, fn_np, fn_jx, kind):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                rn = fn_np()
                rj = fn_jx()
            except Exception as exc:
                res[name] = dict(raised=type(exc).__name__,
                                 message=repr(exc)[:200])
                print("%-26s RAISED %s" % (name, repr(exc)[:100]))
                return
        vn = F.rt_vec(rn)
        vj = F.rt_vec(rj)
        cl_n = F.closure_jones(rn) if kind == "jones" else F.closure_eff(rn)
        cl_j = F.closure_jones(rj) if kind == "jones" else F.closure_eff(rj)
        res[name] = dict(raised=None, numpy_rt=vn.tolist(), jax_rt=vj.tolist(),
                         numpy_closure=cl_n, jax_closure=cl_j,
                         parity=F.motion(vn, vj))
        print("%-26s np.closure %+.4e  jax.closure %+.4e  parity %.3e"
              % (name, cl_n, cl_j, F.motion(vn, vj)))

    # ---- _jax_twod: pmm_efficiency_2d_cell ---------------------------------
    for tag, pillar, nsub in (("cellweak_region", WEAK, NS),
                              ("cellweak_none", WEAK, NS_OFF),
                              ("cellstrong_region", 6.0, NS)):
        record("jaxtwod_" + tag,
               (lambda p=pillar, n=nsub: pmm_efficiency_2d_cell(
                   PX, PX, _np_cell(p), n, 1.0, D, WL, degree=7, n_orders=4)),
               (lambda p=pillar, n=nsub: pmm_efficiency_2d_cell(
                   PX, PX, jcell(p), n, 1.0, D, WL, degree=7, n_orders=4,
                   region_layout=LAYOUT)),
               "eff")

    # ---- _jax_stack2d: PMM2DStackHybrid with a traced cell ------------------
    def stack(traced, spacer, nsub, pillar):
        st = PMM2DStackHybrid(PX, PX, n_substrate=nsub, n_superstrate=1.0,
                              degree=7, n_orders=3, symmetry=False)
        if spacer:
            st.add_layer(0.1e-6, eps=HOST)
        if traced:
            st.add_layer(D, eps_cell=jcell(pillar), region_layout=LAYOUT)
        else:
            st.add_layer(D, eps_cell=_np_cell(pillar))
        if spacer:
            st.add_layer(0.1e-6, eps=HOST)
        return st.set_source(WL, theta=0.0).solve()

    for tag, spacer, nsub, pillar in (("region", False, NS, WEAK),
                                      ("both", True, NS, WEAK),
                                      ("none", False, NS_OFF, WEAK),
                                      ("strong_both", True, NS, 6.0)):
        record("jaxstack_" + tag,
               (lambda s=spacer, n=nsub, p=pillar: stack(False, s, n, p)),
               (lambda s=spacer, n=nsub, p=pillar: stack(True, s, n, p)),
               "jones")

    # ---- _jax_twod_jones: pmm_jones_2d with a traced tensor cell -----------
    #      (its LAYER runs through the shared rcwa generator; the copy under
    #      test here serves its REGION modes)
    JLAY = np.zeros((4, 4), dtype=np.int64)
    JLAY[:2, :2] = 1

    def _tens(xp, no, ne, twist):
        no2, ne2 = no ** 2, ne ** 2
        c0, s0 = np.cos(twist), np.sin(twist)
        return xp.asarray(
            [[ne2 * c0 * c0 + no2 * s0 * s0, (ne2 - no2) * c0 * s0, 0.0],
             [(ne2 - no2) * c0 * s0, ne2 * s0 * s0 + no2 * c0 * c0, 0.0],
             [0.0, 0.0, no2]], dtype=xp.complex128)

    def _jones_cell(xp, host, no, ne):
        cell = xp.zeros(JLAY.shape + (3, 3), dtype=xp.complex128)
        for r, t in enumerate([_tens(xp, host ** 0.5, host ** 0.5, 0.0),
                               _tens(xp, no, ne, 0.7)]):
            mask = xp.asarray((JLAY == r).astype(np.float64))[..., None, None]
            cell = cell + mask * xp.asarray(t)[None, None]
        return cell

    def jones_np(nsub):
        return pmm_jones_2d(PX, PX, np.asarray(_jones_cell(np, HOST, 1.5,
                                                           1.7)),
                            nsub, 1.0, D, WL, degree=7, n_orders=2)

    def jones_jx(nsub):
        return pmm_jones_2d(PX, PX, _jones_cell(jnp, HOST, 1.5, 1.7), nsub,
                            1.0, D, WL, degree=7, n_orders=2,
                            region_layout=JLAY)

    for tag, nsub in (("region", NS), ("none", NS_OFF)):
        record("jaxjones_" + tag, lambda n=nsub: jones_np(n),
               lambda n=nsub: jones_jx(n), "jones")

    # ---- GRADIENTS ---------------------------------------------------------
    grads = {}

    def grad_pair(name, f, x0, h):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                g = complex(jax.grad(f)(jnp.asarray(x0)))
                fd = (float(f(jnp.asarray(x0 + h)))
                      - float(f(jnp.asarray(x0 - h)))) / (2.0 * h)
            except Exception as exc:
                grads[name] = dict(raised=type(exc).__name__,
                                   message=repr(exc)[:200])
                print("%-26s GRAD RAISED %s" % (name, repr(exc)[:90]))
                return
        grads[name] = dict(grad=[g.real, g.imag], fd=fd,
                           rel=(abs(g.real - fd) / max(abs(fd), 1e-30)))
        print("%-26s grad %+.12e  fd %+.12e  rel %.3e"
              % (name, g.real, fd, grads[name]["rel"]))

    def sumT_cell(e1, nsub=NS_OFF):
        _o, _R, T = pmm_efficiency_2d_cell(
            PX, PX, jcell(e1), nsub, 1.0, D, WL, degree=7, n_orders=4,
            region_layout=LAYOUT)
        return jnp.sum(T)

    # the brief's OFF-coincidence lossless fixture: the derivative there must
    # not move, because nothing on that solve is on a coincidence at all
    grad_pair("grad_cell_offcoinc", sumT_cell, 6.0 + 0j, 1e-5)
    grad_pair("grad_cell_region",
              lambda e: sumT_cell(e, NS), 6.0 + 0j, 1e-5)

    def sumT_stack(eps_u):
        st = PMM2DStackHybrid(PX, PX, n_substrate=NS_OFF, n_superstrate=1.0,
                              degree=7, n_orders=3, symmetry=False)
        st.add_layer(D, eps_cell=_np_cell(6.0))
        st.add_layer(0.1e-6, eps=eps_u)
        _o, _R, T, _J = st.set_source(WL, theta=0.0).solve()
        return jnp.sum(jnp.asarray(T))

    grad_pair("grad_stack_offcoinc", sumT_stack, 2.4 + 0j, 1e-5)

    F.dump(out, dict(surfaces=res, grads=grads,
                     jax_version=jax.__version__,
                     openblas_num_threads=os.environ.get(
                         "OPENBLAS_NUM_THREADS", "unpinned")))


if __name__ == "__main__":
    main()
