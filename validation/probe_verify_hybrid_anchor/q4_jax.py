"""Q4 -- the JAX refusal (fix 2 / audit O3), re-measured from scratch.

On the WITHOUT arm (5.44.0) every traced scalar that routes
``PMM2DStackHybrid.solve`` to the jnp twin must be shown to return the
VERTICAL answer SILENTLY on a stack holding a slanted PATTERNED layer; on the
shipped arm every one of them must RAISE.  Two controls must still solve.

A SEVENTH route the fix's own census lists (a traced UNIFORM eps) is included,
and one route the fix did NOT close is measured beside them:
``pmm_jones_2d(..., slant=..., <traced>)``, whose JAX dispatch never looks at
``slant`` either.
"""
from __future__ import annotations

import math
import warnings

import _lib as L
import numpy as np

CELL = L.BASE                       # 6 x 4, x-asymmetric
CONST = np.full((6, 4), 2.20)
TX, D = 0.5, L.DTHICK
TH, PH = math.radians(25.0), math.radians(0.0)


def _jax():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    return jax, jnp


def _stack(*, slanted, cell=CELL, t=D, wl=L.WL, th=TH, ph=PH,
           n_sup=L.NSUP, n_sub=L.NSUB, film_eps=None):
    from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid
    st = PMM2DStackHybrid(L.PX, L.PY, n_superstrate=n_sup,
                          n_substrate=n_sub, n_orders=5)
    st.add_layer(t, eps_cell=cell, slant=((TX, 0.0) if slanted else None))
    if film_eps is not None:
        st.add_layer(0.10e-6, eps=film_eps)
    st.set_source(wl, theta=th, phi=ph)
    return st


def _solve(st):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = st.solve()
    return out, [str(x.message)[:120] for x in w]


def _cmp(out, ref):
    return dict(dR=float(np.max(np.abs(np.asarray(out[1])
                                       - np.asarray(ref[1])))),
                dT=float(np.max(np.abs(np.asarray(out[2])
                                       - np.asarray(ref[2])))),
                dJones=float(np.max(np.abs(np.asarray(out[3])
                                           - np.asarray(ref[3])))))


def routes(jnp):
    """Every traced input that reaches the jnp twin, each with the SLANTED
    patterned layer present."""
    return {
        "traced_thickness": lambda: _stack(slanted=True,
                                           t=jnp.asarray(D)),
        "traced_wavelength": lambda: _stack(slanted=True,
                                            wl=jnp.asarray(L.WL)),
        "traced_theta": lambda: _stack(slanted=True, th=jnp.asarray(TH)),
        "traced_phi": lambda: _stack(slanted=True, ph=jnp.asarray(PH)),
        "traced_n_substrate": lambda: _stack(slanted=True,
                                             n_sub=jnp.asarray(L.NSUB + 0j)),
        "traced_n_superstrate": lambda: _stack(slanted=True,
                                               n_sup=jnp.asarray(L.NSUP + 0j)),
        "traced_uniform_eps": lambda: _stack(slanted=True,
                                             film_eps=jnp.asarray(2.30 + 0j)),
    }


def main():
    jax, jnp = _jax()
    a = L.arm()
    res = dict(jax_version=jax.__version__,
               x64=bool(jax.config.read("jax_enable_x64")))

    # the two NumPy references
    ref_slanted, _ = _solve(_stack(slanted=True))
    ref_vertical, _ = _solve(_stack(slanted=False))
    res["numpy_slanted_vs_vertical"] = _cmp(ref_slanted, ref_vertical)

    rows = {}
    for name, mk in routes(jnp).items():
        try:
            st = mk()
        except Exception as e:                        # noqa: BLE001
            rows[name] = dict(outcome="RAISE_AT_BUILD",
                              exc=type(e).__name__, msg=str(e)[:200])
            continue
        try:
            out, warns = _solve(st)
        except NotImplementedError as e:
            rows[name] = dict(outcome="REFUSED",
                              exc="NotImplementedError", msg=str(e)[:160])
            continue
        except Exception as e:                        # noqa: BLE001
            rows[name] = dict(outcome="RAISE_OTHER",
                              exc=type(e).__name__, msg=str(e)[:200])
            continue
        # a "traced_uniform_eps" stack has an extra film -- re-reference it
        if name == "traced_uniform_eps":
            rs, _ = _solve(_stack(slanted=True, film_eps=2.30))
            rv, _ = _solve(_stack(slanted=False, film_eps=2.30))
        else:
            rs, rv = ref_slanted, ref_vertical
        rows[name] = dict(
            outcome="SOLVED_SILENTLY",
            vs_vertical=_cmp(out, rv), vs_correct_slanted=_cmp(out, rs),
            closure=float(np.max(np.asarray(out[1]).sum(axis=1)
                                 + np.asarray(out[2]).sum(axis=1))),
            warnings=warns)
    res["routes"] = rows

    # ---- CONTROLS: what must STILL solve on the shipped arm ---------------
    ctl = {}
    try:
        st = _stack(slanted=False, t=jnp.asarray(D))
        out, _w = _solve(st)
        ctl["traced_thickness_VERTICAL"] = dict(
            outcome="SOLVED", vs_numpy=_cmp(out, ref_vertical),
            closure=float(np.max(np.asarray(out[1]).sum(axis=1)
                                 + np.asarray(out[2]).sum(axis=1))))
    except Exception as e:                            # noqa: BLE001
        ctl["traced_thickness_VERTICAL"] = dict(outcome="RAISE",
                                                exc=type(e).__name__,
                                                msg=str(e)[:200])
    try:
        stn = _stack(slanted=True, cell=CONST)         # constant tile + slant
        refc, _w = _solve(_stack(slanted=False, cell=CONST))
        st = _stack(slanted=True, cell=CONST, t=jnp.asarray(D))
        out, _w = _solve(st)
        ctl["traced_thickness_CONSTTILE_slant"] = dict(
            outcome="SOLVED", vs_numpy_vertical=_cmp(out, refc),
            closure=float(np.max(np.asarray(out[1]).sum(axis=1)
                                 + np.asarray(out[2]).sum(axis=1))))
        del stn
    except Exception as e:                            # noqa: BLE001
        ctl["traced_thickness_CONSTTILE_slant"] = dict(
            outcome="RAISE", exc=type(e).__name__, msg=str(e)[:200])
    # a GRADIENT through the vertical control -- the surface must still work
    try:
        def f(t):
            s = _stack(slanted=False, t=t)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                o = s.solve()
            return o[2].sum()
        g = jax.grad(f)(jnp.asarray(D))
        h = 1e-10
        fp = f(jnp.asarray(D + h))
        fm = f(jnp.asarray(D - h))
        fd = float((fp - fm) / (2 * h))
        ctl["grad_vertical_traced_thickness"] = dict(
            ad=float(g), fd=fd,
            rel=abs(float(g) - fd) / max(abs(fd), 1e-30))
    except Exception as e:                            # noqa: BLE001
        ctl["grad_vertical_traced_thickness"] = dict(
            outcome="RAISE", exc=type(e).__name__, msg=str(e)[:200])
    res["controls"] = ctl

    # ---- the OTHER slant-carrying JAX route: pmm_jones_2d -----------------
    from lumenairy.elements.pmm.twod_jones import pmm_jones_2d
    T = np.zeros((6, 4, 3, 3), dtype=complex)
    for i in range(3):
        T[..., i, i] = CELL
    layout = np.arange(24).reshape(6, 4)

    def _pj(depth, slant):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pmm_jones_2d(L.PX, L.PY, T, L.NSUB, L.NSUP, depth, L.WL,
                                theta=TH, phi=PH, n_orders=3, degree=5,
                                slant=slant, region_layout=layout)
    try:
        np_sl = _pj(D, (TX, 0.0))
        np_vt = _pj(D, None)
        jx_sl = _pj(jnp.asarray(D), (TX, 0.0))
        res["pmm_jones_2d_route"] = dict(
            outcome="SOLVED_SILENTLY",
            jax_slant_vs_numpy_VERTICAL=_cmp(jx_sl, np_vt),
            jax_slant_vs_numpy_SLANTED=_cmp(jx_sl, np_sl),
            numpy_slanted_vs_vertical=_cmp(np_sl, np_vt))
    except NotImplementedError as e:
        res["pmm_jones_2d_route"] = dict(outcome="REFUSED", msg=str(e)[:200])
    except Exception as e:                            # noqa: BLE001
        res["pmm_jones_2d_route"] = dict(outcome="RAISE_OTHER",
                                         exc=type(e).__name__,
                                         msg=str(e)[:250])

    for k, v in res["routes"].items():
        print("%-24s %s" % (k, v.get("outcome")),
              v.get("vs_correct_slanted", ""), v.get("vs_vertical", ""))
    print("controls:", res["controls"])
    print("pmm_jones_2d:", res["pmm_jones_2d_route"])
    print("arm:", a["arm"])
    L.dump("q4_jax", res)


if __name__ == "__main__":
    main()
