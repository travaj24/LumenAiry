"""TASK 3 -- V1: the JAX dispatch of ``pmm_jones_2d`` on a SLANTED cell.

Three questions, all answered by measurement on this verification's OWN
fixture (a ``(5, 4, 3, 3)`` x-asymmetric cell over ``px = 0.92 um`` /
``py = 0.80 um`` at ``wl = 0.61 um``, ``slant = (0.45, 0)``, oblique 28 deg --
none of the fix's numbers):

1. do all SEVEN members of the traced-input dispatch tuple SOLVE SILENTLY with
   the VERTICAL answer on the pre-fix trees, and REFUSE post-fix;
2. is there an EIGHTH route -- a traced ``slant``, ``jax.jit`` / ``vmap`` /
   ``grad`` wrappers, the hybrid stack's own dispatch, the pure staggered
   entry, the private jnp twin -- that still reaches the twin with a shear it
   cannot carry;
3. is the CONSTANT-tile slanted cell a measured no-op at oblique AND conical
   incidence and for a genuinely ANISOTROPIC constant tile, not only for the
   isotropic normal-ish case the fix measured?
"""
from __future__ import annotations

import os
import sys
import time
import warnings

if os.environ.get("LUM_ARM_TREE"):
    sys.path.insert(0, os.environ["LUM_ARM_TREE"])
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

import _lib  # noqa: E402
import numpy as np  # noqa: E402

UM = 1e-6
SLANT = (0.45, 0.0)
THETA = float(np.deg2rad(28.0))
NAMES = ("eps_tensor_cell", "n_substrate", "n_superstrate", "depth",
         "wavelength", "theta", "phi")


def cell(sx=5, sy=4, const=None, aniso=False):
    """``(sx, sy, 3, 3)`` in-plane tensor cell, patterned on BOTH axes (the
    jnp twin's own precondition), or CONSTANT-valued when ``const`` is set."""
    c = np.zeros((sx, sy, 3, 3), dtype=complex)
    for i in range(sx):
        for j in range(sy):
            if const is not None:
                m = np.eye(3, dtype=complex) * const
                if aniso:
                    m[0, 0], m[1, 1], m[2, 2] = 2.9, 2.1, 2.5
                    m[0, 1] = m[1, 0] = 0.42
                c[i, j] = m
            else:
                v = 1.30 + 1.75 * ((i * 3 + j * 2) % 7) / 6.0
                c[i, j] = np.eye(3) * v
                c[i, j, 0, 1] = c[i, j, 1, 0] = 0.07 * ((i + 2 * j) % 3)
    return c


def layout(sx=5, sy=4):
    return np.arange(sx * sy, dtype=int).reshape(sx, sy)


BASE = dict(period_x=0.92 * UM, period_y=0.80 * UM, n_substrate=1.55,
            n_superstrate=1.0, depth=0.37 * UM, wavelength=0.61 * UM,
            theta=THETA, phi=0.0, n_orders=3, degree=5)


def _d(a, b):
    return dict(dR=_lib.rel(a[1], b[1]), dT=_lib.rel(a[2], b[2]),
                dJones=_lib.rel(a[3], b[3]),
                sha=_lib.sha(np.asarray(a[0]), np.asarray(a[1]),
                             np.asarray(a[2]), np.asarray(a[3])))


def main():                                                  # noqa: C901
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from lumenairy.elements.pmm import PMM2DStackHybrid, pmm_jones_2d

    out = {}
    C = cell()
    LAY = layout()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        np_vert = pmm_jones_2d(eps_tensor_cell=C, slant=None, **BASE)
        np_slant = pmm_jones_2d(eps_tensor_cell=C, slant=SLANT, **BASE)
    out["numpy_warnings"] = [str(w.message)[:150] for w in caught]
    out["numpy_vertical_vs_slanted"] = _d(np_vert, np_slant)
    out["numpy_vertical_sha"] = _lib.sha(*[np.asarray(x) for x in np_vert])
    out["numpy_slanted_sha"] = _lib.sha(*[np.asarray(x) for x in np_slant])

    # ---- the SEVEN dispatch members ------------------------------------
    seven = {}
    for nm in NAMES:
        kw = dict(BASE)
        kw["eps_tensor_cell"] = C
        kw["region_layout"] = LAY
        kw["slant"] = SLANT
        if nm == "eps_tensor_cell":
            kw[nm] = jnp.asarray(C)
        else:
            kw[nm] = jnp.asarray(float(kw[nm] if nm != "phi" else 0.0))
        rec = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t = time.time()
            try:
                res = pmm_jones_2d(**kw)
                rec["outcome"] = "SOLVED"
                rec.update(vs_numpy_vertical=_d(res, np_vert),
                           vs_numpy_slanted=_d(res, np_slant))
            except Exception as exc:                            # noqa: BLE001
                rec["outcome"] = type(exc).__name__
                rec["message"] = str(exc)
            rec["secs"] = round(time.time() - t, 2)
        rec["warnings"] = [str(w.message)[:200] for w in caught]
        seven[nm] = rec
        print(f"  seven/{nm:16s} {rec['outcome']}")
    out["seven"] = seven

    # ---- CONSTANT-tile no-op, three mounts and an ANISOTROPIC tile -----
    noop = {}
    for tag, kwx, cst in (
            ("iso_ob28", dict(), cell(const=2.20)),
            ("iso_normal", dict(theta=0.0), cell(const=2.20)),
            ("iso_conical", dict(theta=THETA, phi=float(np.deg2rad(37.0))),
             cell(const=2.20)),
            ("aniso_ob28", dict(), cell(const=2.20, aniso=True)),
            ("aniso_conical", dict(theta=THETA,
                                   phi=float(np.deg2rad(37.0))),
             cell(const=2.20, aniso=True))):
        kw = dict(BASE)
        kw.update(kwx)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = pmm_jones_2d(eps_tensor_cell=cst, slant=None, **kw)
            s = pmm_jones_2d(eps_tensor_cell=cst, slant=SLANT, **kw)
            big = pmm_jones_2d(eps_tensor_cell=cst, slant=(1.7, 0.9), **kw)
            rec = dict(slant_vs_vertical=_d(s, v),
                       bigslant_vs_vertical=_d(big, v))
            try:
                j = pmm_jones_2d(eps_tensor_cell=jnp.asarray(cst),
                                 slant=SLANT, region_layout=LAY, **kw)
                rec["traced_outcome"] = "SOLVED"
                rec["traced_vs_numpy_vertical"] = _d(j, v)
            except Exception as exc:                            # noqa: BLE001
                rec["traced_outcome"] = type(exc).__name__
                rec["traced_message"] = str(exc)[:200]
        noop[tag] = rec
        print(f"  noop/{tag:16s} dR={rec['slant_vs_vertical']['dR']:.3e} "
              f"traced={rec['traced_outcome']}")
    out["constant_tile_noop"] = noop

    # ---- the EIGHTH-ROUTE HUNT -----------------------------------------
    hunt = {}

    def _probe(tag, fn):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                r = fn()
                hunt[tag] = dict(outcome="SOLVED", **r)
            except Exception as exc:                            # noqa: BLE001
                hunt[tag] = dict(outcome=type(exc).__name__,
                                 message=str(exc)[:400])
        hunt[tag]["warnings"] = [str(w.message)[:200] for w in caught]
        print(f"  hunt/{tag:34s} {hunt[tag]['outcome']}")

    # 8a: only the SLANT is a jnp array (concrete), everything else NumPy.
    def _h_slant_concrete():
        r = pmm_jones_2d(eps_tensor_cell=C, slant=(jnp.asarray(0.45),
                                                   jnp.asarray(0.0)), **BASE)
        return dict(vs_numpy_slanted=_d(r, np_slant),
                    vs_numpy_vertical=_d(r, np_vert))
    _probe("slant_is_a_concrete_jnp_array", _h_slant_concrete)

    # 8b: the slant TRACED under jit, everything else NumPy.
    def _h_slant_traced():
        f = jax.jit(lambda s: pmm_jones_2d(eps_tensor_cell=C, slant=(s, 0.0),
                                           region_layout=LAY, **BASE)[1])
        r = f(jnp.asarray(0.45))
        return dict(R_sha=_lib.sha(np.asarray(r)))
    _probe("slant_traced_under_jit", _h_slant_traced)

    # 8c: jax.jit of the whole NumPy wrapper over the DEPTH.
    def _h_jit_depth():
        kw = {k: v for k, v in BASE.items() if k != "depth"}
        f = jax.jit(lambda d: pmm_jones_2d(eps_tensor_cell=C, depth=d,
                                           slant=SLANT, region_layout=LAY,
                                           **kw)[1])
        r = f(jnp.asarray(BASE["depth"]))
        return dict(R_sha=_lib.sha(np.asarray(r)))
    _probe("jit_over_depth", _h_jit_depth)

    # 8d: vmap over a batch of wavelengths.
    def _h_vmap_wl():
        kw = {k: v for k, v in BASE.items() if k != "wavelength"}
        f = jax.vmap(lambda w: pmm_jones_2d(eps_tensor_cell=C, wavelength=w,
                                            slant=SLANT, region_layout=LAY,
                                            **kw)[1])
        r = f(jnp.asarray([0.60e-6, 0.61e-6]))
        return dict(R_sha=_lib.sha(np.asarray(r)))
    _probe("vmap_over_wavelength", _h_vmap_wl)

    # 8e: jax.grad through the slanted call.
    def _h_grad_depth():
        kw = {k: v for k, v in BASE.items() if k != "depth"}
        g = jax.grad(lambda d: jnp.sum(
            pmm_jones_2d(eps_tensor_cell=C, depth=d, slant=SLANT,
                         region_layout=LAY, **kw)[2]))
        return dict(grad=float(g(jnp.asarray(BASE["depth"]))))
    _probe("grad_over_depth", _h_grad_depth)

    # 8f: the PRIVATE jnp twin, called directly with a slanted geometry.
    def _h_twin_direct():
        import inspect

        from lumenairy.elements.pmm._jax_twod_jones import _pmm_jones_2d_cell_jax
        sig = list(inspect.signature(_pmm_jones_2d_cell_jax)
                   .parameters)
        return dict(signature=sig, has_slant=("slant" in sig))
    _probe("private_twin_signature", _h_twin_direct)

    # 8g: the HYBRID stack's own JAX dispatch, slanted layer + traced
    #     thickness / wavelength / theta / half-space index / uniform eps.
    def _hyb(traced):
        def go():
            s = PMM2DStackHybrid(
                0.92 * UM, 0.80 * UM,
                n_substrate=(jnp.asarray(1.55) if traced == "n_sub"
                             else 1.55),
                degree=5, n_orders=3)
            s.add_layer(jnp.asarray(0.37e-6) if traced == "t" else 0.37 * UM,
                        eps_tensor_cell=C, slant=SLANT)
            s.set_source(
                jnp.asarray(0.61e-6) if traced == "wl" else 0.61 * UM,
                theta=(jnp.asarray(THETA) if traced == "theta" else THETA))
            r = s.solve()
            return dict(R_sha=_lib.sha(np.asarray(r[1])))
        return go
    for tr in ("t", "wl", "theta", "n_sub"):
        _probe(f"hybrid_stack_traced_{tr}", _hyb(tr))

    # 8h: a SCALAR (non-tensor) traced hybrid layer carrying a slant.
    def _h_hyb_scalar():
        s = PMM2DStackHybrid(0.92 * UM, 0.80 * UM, n_substrate=1.55,
                             degree=5, n_orders=3)
        e = np.full((5, 4), 1.30)
        e[1:3, :] = 3.05
        s.add_layer(jnp.asarray(0.37e-6), eps_cell=e, slant=SLANT)
        s.set_source(0.61 * UM, theta=THETA)
        r = s.solve()
        return dict(R_sha=_lib.sha(np.asarray(r[1])))
    _probe("hybrid_stack_scalar_cell_traced_t", _h_hyb_scalar)

    # 8i: the PURE staggered single-layer entry with a traced input + slant.
    def _h_stag():
        from lumenairy.elements.pmm import pmm_jones_2d_staggered
        e = np.full((4, 4), 1.30)
        e[1:3, :] = 3.05
        r = pmm_jones_2d_staggered(
            0.92 * UM, 0.80 * UM, e, 1.55, 1.0,
            jnp.asarray(0.37e-6), 0.61 * UM, theta=THETA,
            n_modes=3, n_orders=2, slant=SLANT)
        return dict(R_sha=_lib.sha(np.asarray(r[1])))
    _probe("pure_staggered_traced_depth", _h_stag)

    # 8j: a y-only slant on the traced dispatch.
    def _h_yslant():
        r = pmm_jones_2d(eps_tensor_cell=C, slant=(0.0, 0.45),
                         region_layout=LAY,
                         **{**BASE, "depth": jnp.asarray(BASE["depth"])})
        return dict(R_sha=_lib.sha(np.asarray(r[1])))
    _probe("traced_depth_y_only_slant", _h_yslant)

    # 8k: an EXPLICIT zero slant on the traced dispatch must still solve.
    def _h_zeroslant():
        r = pmm_jones_2d(eps_tensor_cell=C, slant=(0.0, 0.0),
                         region_layout=LAY,
                         **{**BASE, "depth": jnp.asarray(BASE["depth"])})
        return dict(vs_numpy_vertical=_d(r, np_vert))
    _probe("traced_depth_explicit_zero_slant", _h_zeroslant)

    # 8l: a slant BELOW the no-op tolerance in the tile (a cell that varies
    #     by 5e-13, i.e. inside _SLANT_NOOP_TOL) -- declared a no-op.
    def _h_tiny_var():
        c2 = cell(const=2.20)
        c2[0, 0, 0, 0] += 5e-13
        v = pmm_jones_2d(eps_tensor_cell=c2, slant=None, **BASE)
        s = pmm_jones_2d(eps_tensor_cell=c2, slant=SLANT, **BASE)
        j = pmm_jones_2d(eps_tensor_cell=jnp.asarray(c2), slant=SLANT,
                         region_layout=LAY, **BASE)
        return dict(numpy_slant_vs_vertical=_d(s, v),
                    traced_vs_numpy_slanted=_d(j, s),
                    traced_vs_numpy_vertical=_d(j, v))
    _probe("tile_varies_by_5e-13_declared_noop", _h_tiny_var)

    # 8m: a tile varying by 5e-11 -- just ABOVE the tolerance.
    def _h_above_tol():
        c2 = cell(const=2.20)
        c2[0, 0, 0, 0] += 5e-11
        j = pmm_jones_2d(eps_tensor_cell=jnp.asarray(c2), slant=SLANT,
                         region_layout=LAY, **BASE)
        return dict(R_sha=_lib.sha(np.asarray(j[1])))
    _probe("tile_varies_by_5e-11_above_tol", _h_above_tol)

    # 8n: the VERTICAL traced control still differentiates (AD vs central FD).
    def _h_grad_vertical():
        kw = {k: v for k, v in BASE.items() if k != "depth"}

        def f(d):
            return jnp.sum(pmm_jones_2d(eps_tensor_cell=C, depth=d,
                                        slant=None, region_layout=LAY,
                                        **kw)[2])
        d0 = BASE["depth"]
        g = float(jax.grad(f)(jnp.asarray(d0)))
        h = 1e-11
        f0 = float(f(jnp.asarray(d0)))
        fd = float((f(jnp.asarray(d0 + h)) - f(jnp.asarray(d0 - h)))
                   / (2.0 * h))
        return dict(ad=g, fd=fd, rel=abs(g - fd) / max(abs(fd), 1e-300),
                    fd_floor=float(np.finfo(float).eps * abs(f0) / h))
    _probe("vertical_traced_grad_vs_fd", _h_grad_vertical)

    # 8o: the fix's OWN control shape -- a CONSTANT NumPy cell with a
    #     TRACED DEPTH.  ``_slanted_cell_is_a_frame_noop`` can inspect a
    #     NumPy cell, so this must still SOLVE and match the NumPy vertical
    #     call bit for bit.
    def _h_const_numpy_traced_depth():
        cst = cell(const=2.20)
        kw = {k: v for k, v in BASE.items() if k != "depth"}
        v = pmm_jones_2d(eps_tensor_cell=cst, slant=None, **BASE)
        j = pmm_jones_2d(eps_tensor_cell=cst, slant=SLANT,
                         depth=jnp.asarray(BASE["depth"]),
                         region_layout=LAY, **kw)
        return dict(vs_numpy_vertical=_d(j, v),
                    sha_equal=bool(_lib.sha(*[np.asarray(x) for x in j])
                                   == _lib.sha(*[np.asarray(x)
                                                 for x in v])))
    _probe("const_numpy_cell_traced_depth", _h_const_numpy_traced_depth)

    # 8p: a CONCRETE (not traced) jnp cell that is CONSTANT.  It is
    #     inspectable -- ``np.asarray`` on it is exact -- but
    #     ``is_jax_array`` cannot tell it from a tracer, so the refusal
    #     covers it too.  Recorded as a SCOPE boundary, not a defect.
    def _h_const_concrete_jnp_cell():
        cst = cell(const=2.20)
        return dict(numpy_vertical_sha=_lib.sha(*[
            np.asarray(x) for x in pmm_jones_2d(
                eps_tensor_cell=cst, slant=None, **BASE)]),
            traced=_lib.sha(*[np.asarray(x) for x in pmm_jones_2d(
                eps_tensor_cell=jnp.asarray(cst), slant=SLANT,
                region_layout=LAY, **BASE)]))
    _probe("const_concrete_jnp_cell_slanted", _h_const_concrete_jnp_cell)

    # 8q: the same concrete jnp cell with NO slant must still solve.
    def _h_concrete_jnp_cell_vertical():
        r = pmm_jones_2d(eps_tensor_cell=jnp.asarray(C), slant=None,
                         region_layout=LAY, **BASE)
        return dict(vs_numpy_vertical=_d(r, np_vert))
    _probe("concrete_jnp_cell_vertical", _h_concrete_jnp_cell_vertical)

    # 8r: the PURE staggered entry under a REAL tracer (jit).  That engine
    #     has no jnp twin at all, so the question is whether it fails LOUDLY
    #     rather than silently dropping the shear.
    def _h_stag_jit():
        from lumenairy.elements.pmm import pmm_jones_2d_staggered
        e = np.full((4, 4), 1.30)
        e[1:3, :] = 3.05
        f = jax.jit(lambda d: pmm_jones_2d_staggered(
            0.92 * UM, 0.80 * UM, e, 1.55, 1.0, d, 0.61 * UM, theta=THETA,
            n_modes=3, n_orders=2, slant=SLANT)[1])
        return dict(R_sha=_lib.sha(np.asarray(f(jnp.asarray(0.37e-6)))))
    _probe("pure_staggered_under_jit", _h_stag_jit)

    # 8s: ``PMM2DStackPure.solve`` with a concrete jnp thickness + slant.
    def _h_pure_stack_traced():
        from lumenairy.elements.pmm import PMM2DStackPure
        e = np.full((4, 4), 1.30)
        e[1:3, :] = 3.05
        st = PMM2DStackPure(0.92 * UM, 0.80 * UM, n_substrate=1.55,
                            n_modes=3, n_orders=2)
        st.add_layer(jnp.asarray(0.37e-6), eps_cell=e, slant=SLANT)
        st.set_source(0.61 * UM, theta=THETA)
        return dict(R_sha=_lib.sha(np.asarray(st.solve()[1])))
    _probe("pure_stack_concrete_jnp_thickness", _h_pure_stack_traced)

    out["eighth_route_hunt"] = hunt
    _lib.save("t3_v1_jax", out)


if __name__ == "__main__":
    main()
