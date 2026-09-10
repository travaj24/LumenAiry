"""P3 -- the BIT-IDENTITY gate for the hybrid frame-anchor fix.

The fix must be invisible everywhere the hybrid does NOT enter a sheared frame,
and must leave ``R``, ``T`` and the REFLECTION Jones alone even where it does.
This probe hashes every returned array with sha256 so the claim is byte-level,
never tolerance-level, and it runs against BOTH trees:

  * the fix worktree ``C:/tmp/lum_hyb`` -- run it with ``FIXSTAGE=pre`` before
    the library edit and ``FIXSTAGE=post`` after;
  * the READ-ONLY main clone ``D:/Metacept/.../Lumenairy`` (``arm() == 'base'``)
    -- the independent reference, which by construction cannot have the fix.

``p3_compare.py`` diffs the JSONs.  Row ``slanted_patterned_*`` is the one row
whose ``jones_transmission`` hash is EXPECTED to change (that is the fix); its
``R`` / ``T`` / ``jones_reflection`` hashes must not.
"""
import numpy as np

from _lib import arm, dump, sha  # noqa: I001

from lumenairy.elements.pmm import PMM2DStackHybrid

WL = 0.68e-6
PX = PY = 1.20e-6
DEP = 0.55e-6
NSUP, NSUB = 1.0, 1.5
NORD = 5
OBL = (np.deg2rad(25.0), 0.0)
CON = (np.deg2rad(25.0), np.deg2rad(40.0))
NRM = (0.0, 0.0)


def scalar_cell(n=6):
    c = np.ones((n, n), dtype=complex)
    c[:, 0:n // 2] = np.repeat(np.array([4.0, 4.0, 2.0, 1.0, 1.0, 1.0]),
                               n // 6)[:, None]
    return c


def centro_cell(n=6):
    c = np.ones((n, n), dtype=complex)
    c[2:4, 2:4] = 6.0
    return c


def inplane_tensor(n=4):
    t = np.zeros((n, n, 3, 3), dtype=complex)
    for i in range(n):
        for j in range(n):
            e = 4.0 if (i < n // 2) else 2.0
            t[i, j] = np.diag([e, e * 1.15, e])
            t[i, j, 0, 1] = t[i, j, 1, 0] = 0.25
    return t


def oop_tensor(n=4):
    t = inplane_tensor(n)
    t[..., 0, 2] = t[..., 2, 0] = 0.35
    return t


def _st(**kw):
    kw.setdefault("n_orders", NORD)
    return PMM2DStackHybrid(PX, PY, n_superstrate=NSUP, n_substrate=NSUB, **kw)


def _row(st, mount, jt=True):
    st.set_source(WL, theta=mount[0], phi=mount[1])
    o, R, T, J = st.solve()
    d = dict(orders=sha(o), R=sha(R), T=sha(T), jones_reflection=sha(J),
             R_sum=float(np.sum(R)), T_sum=float(np.sum(T)))
    if jt:
        d["jones_transmission"] = sha(st.jones_transmission())
        a = st.per_order_amplitudes("transmission")
        d["per_order_transmission"] = sha(a["Ex"], a["Ey"])
        b = st.per_order_amplitudes("reflection")
        d["per_order_reflection"] = sha(b["Ex"], b["Ey"])
    return d


def fixtures():
    f = {}

    def add(name, build, mount):
        st = build()
        f[name] = _row(st, mount)

    # -- 1..3 plain VERTICAL scalar patterned layer, three mounts ----------
    for tag, mount in (("normal", NRM), ("oblique", OBL), ("conical", CON)):
        add(f"vertical_scalar_{tag}",
            lambda: _st().add_layer(DEP, eps_cell=scalar_cell()), mount)
    # -- 4 VERTICAL in-plane tensor ---------------------------------------
    add("vertical_inplane_tensor_conical",
        lambda: _st().add_layer(DEP, eps_tensor_cell=inplane_tensor()), CON)
    # -- 5 VERTICAL out-of-plane tensor (the 4N generator, no slant) -------
    add("vertical_oop_tensor_oblique",
        lambda: _st().add_layer(DEP, eps_tensor_cell=oop_tensor()), OBL)
    # -- 6 three-layer vertical stack --------------------------------------

    def three():
        st = _st()
        st.add_layer(0.20e-6, eps=2.10)
        st.add_layer(DEP, eps_cell=scalar_cell())
        st.add_layer(0.30e-6, eps=3.20)
        return st
    add("vertical_three_layer_oblique", three, OBL)
    # -- 7 uniform-only stack ----------------------------------------------

    def films():
        st = _st()
        st.add_layer(0.25e-6, eps=2.25)
        st.add_layer(0.35e-6, eps=4.00)
        return st
    add("uniform_films_conical", films, CON)
    # -- 8/9 the even-parity fold, on and off -------------------------------
    add("centro_symmetry_auto_normal",
        lambda: _st(symmetry="auto").add_layer(DEP, eps_cell=centro_cell()),
        NRM)
    add("centro_symmetry_off_normal",
        lambda: _st(symmetry=False).add_layer(DEP, eps_cell=centro_cell()),
        NRM)
    # -- 10 tapered pillars (an auto-sliced z-staircase) --------------------
    add("tapered_pillars_oblique",
        lambda: _st().add_tapered_pillars(
            DEP, pillars=[((0.6e-6, 0.6e-6), (0.40e-6, 0.40e-6),
                           (0.24e-6, 0.24e-6), 6.0)],
            eps_host=1.0, n_slices=4), OBL)
    # -- 11/12 the other cascade strategies ---------------------------------
    def three_casc(c):
        st = _st(cascade=c)
        st.add_layer(0.20e-6, eps=2.10)
        st.add_layer(DEP, eps_cell=scalar_cell())
        st.add_layer(0.30e-6, eps=3.20)
        return st
    for casc in ("fused", "tree"):
        add(f"cascade_{casc}_oblique", lambda c=casc: three_casc(c), OBL)
    # -- 13/14 SLANTED UNIFORM layers (never enter a frame) -----------------
    add("slanted_uniform_x_oblique",
        lambda: _st().add_layer(DEP, eps=2.25, slant=(0.5, 0.0)), OBL)
    add("slanted_uniform_diag_conical",
        lambda: _st().add_layer(DEP, eps=2.25, slant=(0.4, 0.3)), CON)
    # -- 15 a CONSTANT-tile "patterned" slanted layer (also no frame) -------
    add("slanted_constant_tile_oblique",
        lambda: _st().add_layer(DEP, eps_cell=np.full((6, 6), 2.25 + 0j),
                                slant=(0.5, 0.0)), OBL)
    # -- 16 slanted uniform BELOW a vertical pattern ------------------------

    def mixed():
        st = _st()
        st.add_layer(DEP, eps_cell=scalar_cell())
        st.add_layer(0.30e-6, eps=3.20, slant=(0.6, 0.0))
        return st
    add("vertical_pattern_over_slanted_uniform_conical", mixed, CON)
    # -- 17/18 the SLANTED PATTERNED rows: R / T / reflection must not move,
    #          jones_transmission is the one hash the fix is allowed to change
    add("slanted_patterned_oblique",
        lambda: _st().add_layer(DEP, eps_cell=scalar_cell(), slant=(0.5, 0.0)),
        OBL)
    add("slanted_patterned_conical",
        lambda: _st().add_layer(DEP, eps_cell=scalar_cell(), slant=(0.4, 0.3)),
        CON)
    add("slanted_patterned_normal",
        lambda: _st().add_layer(DEP, eps_cell=scalar_cell(), slant=(0.5, 0.0)),
        NRM)
    add("slanted_tensor_inplane_oblique",
        lambda: _st().add_layer(DEP, eps_tensor_cell=inplane_tensor(),
                                slant=(0.5, 0.0)), OBL)
    # -- 19 a wavelength SWEEP (per-wavelength solves on private clones) ----
    st = _st()
    st.add_layer(DEP, eps_cell=scalar_cell())
    st.set_source(WL, theta=OBL[0], phi=OBL[1])
    o, R, T, J = st.solve_vs_wavelength([0.62e-6, 0.68e-6, 0.74e-6],
                                        jones=True)
    f["sweep_three_wavelengths_oblique"] = dict(
        orders=sha(o), R=sha(R), T=sha(T), jones_reflection=sha(J),
        R_sum=float(np.sum(R)), T_sum=float(np.sum(T)))
    return f


def main():
    f = fixtures()
    for k, v in sorted(f.items()):
        print(f"{k:48s} R {v['R'][:12]}  Jr {v['jones_reflection'][:12]}  "
              f"Jt {v.get('jones_transmission', '-' * 12)[:12]}")
    dump("p3_bit_identity", f)
    print("arm", arm(), "rows", len(f))


if __name__ == "__main__":
    main()
