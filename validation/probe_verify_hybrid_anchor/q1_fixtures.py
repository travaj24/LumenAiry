"""Q1 -- the WITHOUT-ARM bit-identity fixture set (27 fixtures, 7 keys each).

Run on BOTH builds (``lum_vhyb`` = shipped, ``lum_v5440`` = the released
5.44.0 read-only worktree) and diff with ``q1_compare.py``.  The claim under
test: everything except the TRANSMITTED amplitudes and ``jones_transmission``
on a slanted PATTERNED layer is bit-identical to the released library.

Every fixture is built here, from this file's own cells -- none of the fix's
own probe fixtures are imported.
"""
from __future__ import annotations

import math
import warnings

import _lib as L
import numpy as np

CENTRO = np.array([                     # centro-symmetric -> the even fold
    [1.20, 1.20, 1.20, 1.20],
    [1.20, 2.60, 2.60, 1.20],
    [1.20, 2.60, 2.60, 1.20],
    [1.20, 1.20, 1.20, 1.20],
], dtype=float)

LOSSY = L.BASE + 0.15j                  # PUBLIC Im(eps) > 0 = loss


def _tensor_cell(base, *, exy=0.0, exz=0.0, ezx=0.0):
    """(Sx, Sy, 3, 3) from a scalar cell, with optional off-diagonals."""
    Sx, Sy = base.shape
    T = np.zeros((Sx, Sy, 3, 3), dtype=complex)
    for i in range(3):
        T[..., i, i] = base
    if exy:
        T[..., 0, 1] = exy * (base - base.min())
        T[..., 1, 0] = -exy * (base - base.min())
    if exz:
        T[..., 0, 2] = exz * (base - base.min())
    if ezx:
        T[..., 2, 0] = ezx * (base - base.min())
    return T


def _st(**kw):
    return L.hybrid(**kw)


def _solve(st, mount, wl=L.WL):
    th, ph = L.MOUNTS[mount]
    st.set_source(wl, theta=th, phi=ph)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return st.solve()


def _keys(st, out):
    orders, R, T, jr = out
    d = dict(orders=L.sha(np.asarray(orders)), R=L.sha(R), T=L.sha(T),
             jones_reflection=L.sha(jr))
    try:
        d["jones_transmission"] = L.sha(st.jones_transmission())
    except Exception as e:                       # noqa: BLE001
        d["jones_transmission"] = "RAISE:" + type(e).__name__
    for port, tag in (("transmission", "per_order_transmission"),
                      ("reflection", "per_order_reflection")):
        try:
            a = st.per_order_amplitudes(port)
            d[tag] = L.sha(np.concatenate([a["Ex"], a["Ey"]], axis=1))
        except Exception as e:                   # noqa: BLE001
            d[tag] = "RAISE:" + type(e).__name__
    return d


# ------------------------------------------------------------ the fixtures
def fixtures():
    F = {}
    d = L.DTHICK

    # --- 1..3 vertical scalar, three mounts -------------------------------
    for m in ("normal", "oblique25", "conical25_40"):
        st = _st(n_orders=5)
        st.add_layer(d, eps_cell=L.BASE)
        F["vert_scalar_" + m] = (st, m)

    # --- 4 vertical IN-PLANE tensor ---------------------------------------
    st = _st(n_orders=5)
    st.add_layer(d, eps_tensor_cell=_tensor_cell(L.BASE, exy=0.25j))
    F["vert_tensor_inplane_oblique25"] = (st, "oblique25")

    # --- 5 vertical OUT-OF-PLANE tensor (the 4N generator, no slant) ------
    st = _st(n_orders=5)
    st.add_layer(d, eps_tensor_cell=_tensor_cell(L.BASE, exz=0.30, ezx=0.30))
    F["vert_tensor_oop_oblique25"] = (st, "oblique25")

    # --- 6 three-layer vertical stack (pattern / film / pattern) ----------
    st = _st(n_orders=5)
    st.add_layer(0.20e-6, eps_cell=L.BASE)
    st.add_layer(0.13e-6, eps=2.25)
    st.add_layer(0.17e-6, eps_cell=L.LOWER)
    F["vert_three_layer_conical"] = (st, "conical25_40")

    # --- 7 uniform films only ---------------------------------------------
    st = _st(n_orders=5)
    for t, e in ((0.11e-6, 2.10), (0.19e-6, 1.60), (0.07e-6, 3.40)):
        st.add_layer(t, eps=e)
    F["uniform_films_oblique25"] = (st, "oblique25")

    # --- 8/9 the even-parity fold, both switches --------------------------
    st = _st(n_orders=5, symmetry="auto")
    st.add_layer(d, eps_cell=CENTRO)
    F["centro_symmetry_auto_normal"] = (st, "normal")
    st = _st(n_orders=5, symmetry=False)
    st.add_layer(d, eps_cell=CENTRO)
    F["centro_symmetry_false_normal"] = (st, "normal")

    # --- 10/11 tapered pillars --------------------------------------------
    st = _st(n_orders=5)
    st.add_tapered_pillar(d, eps_pillar=3.30, eps_host=1.10,
                          x_bounds_bottom=(0.18e-6, 0.66e-6),
                          y_bounds_bottom=(0.20e-6, 0.70e-6),
                          x_bounds_top=(0.28e-6, 0.56e-6),
                          y_bounds_top=(0.30e-6, 0.60e-6), n_slices=6)
    F["tapered_pillar_oblique25"] = (st, "oblique25")
    st = _st(n_orders=5)
    st.add_tapered_pillars(
        d, eps_host=1.10, n_slices=5,
        pillars=[((0.22e-6, 0.24e-6), (0.16e-6, 0.20e-6),
                  (0.26e-6, 0.30e-6), 3.30),
                 ((0.66e-6, 0.62e-6), (0.24e-6, 0.22e-6),
                  (0.24e-6, 0.22e-6), 2.40)])
    F["tapered_pillars_normal"] = (st, "normal")

    # --- 12..14 the three non-default cascades ----------------------------
    for cas in ("fused", "tree", "monolithic"):
        st = _st(n_orders=5, cascade=cas)
        st.add_layer(0.15e-6, eps_cell=L.BASE)
        st.add_layer(0.10e-6, eps=2.05)
        st.add_layer(0.15e-6, eps_cell=L.LOWER)
        F["cascade_" + cas + "_oblique25"] = (st, "oblique25")

    # --- 15/16 slanted UNIFORM layers (must be no-ops) --------------------
    st = _st(n_orders=5)
    st.add_layer(d, eps=2.20, slant=(0.7, 0.0))
    F["slanted_uniform_x_oblique25"] = (st, "oblique25")
    st = _st(n_orders=5)
    st.add_layer(d, eps=2.20, slant=(0.4, 0.3))
    F["slanted_uniform_diag_conical"] = (st, "conical25_40")
    # ... and their VERTICAL twin, for the cross-fixture identity
    st = _st(n_orders=5)
    st.add_layer(d, eps=2.20)
    F["vertical_film_2p20_oblique25"] = (st, "oblique25")

    # --- 17 CONSTANT-tile slanted patterned layer -------------------------
    st = _st(n_orders=5)
    st.add_layer(d, eps_cell=np.full((6, 4), 2.20), slant=(0.7, 0.0))
    F["slanted_consttile_oblique25"] = (st, "oblique25")

    # --- 18 a vertical pattern OVER a slanted uniform film ----------------
    st = _st(n_orders=5)
    st.add_layer(0.22e-6, eps_cell=L.BASE)
    st.add_layer(0.23e-6, eps=2.20, slant=(0.7, 0.0))
    F["vert_pattern_over_slanted_film_oblique25"] = (st, "oblique25")

    # --- 19..21 slanted PATTERNED, three mounts, QUARTER walk -------------
    for m in ("normal", "oblique25", "conical25_40"):
        st = _st(n_orders=5)
        st.add_layer(d, eps_cell=L.BASE, slant=(0.5, 0.0))   # 0.5*0.45 = P/4
        F["slanted_patterned_quarter_" + m] = (st, m)

    # --- 22 slanted PATTERNED at a HALF walk, NORMAL (the P_0 exemption
    #        AND the half-walk conjugate degeneracy) ----------------------
    st = _st(n_orders=5)
    st.add_layer(d, eps_cell=L.BASE, slant=(1.0, 0.0))       # 1.0*0.45 = P/2
    F["slanted_patterned_half_normal"] = (st, "normal")

    # --- 23 slanted PATTERNED, in-plane TENSOR ----------------------------
    st = _st(n_orders=5)
    st.add_layer(d, eps_tensor_cell=_tensor_cell(L.BASE, exy=0.25j),
                 slant=(0.5, 0.0))
    F["slanted_tensor_inplane_oblique25"] = (st, "oblique25")

    # --- 24 two slanted layers, DIFFERENT slants and thicknesses ----------
    st = _st(n_orders=5)
    st.add_layer(0.30e-6, eps_cell=L.BASE, slant=(0.5, 0.0))
    st.add_layer(0.15e-6, eps_cell=L.LOWER, slant=(-0.3, 0.2))
    F["two_slanted_layers_conical"] = (st, "conical25_40")

    # --- 25 a LOSSY patterned layer ---------------------------------------
    st = _st(n_orders=5)
    st.add_layer(d, eps_cell=LOSSY)
    F["vert_lossy_oblique25"] = (st, "oblique25")

    # --- 26 a slanted patterned layer OVER a reflecting film --------------
    st = _st(n_orders=5)
    st.add_layer(d, eps_cell=L.BASE, slant=(0.5, 0.0))
    st.add_layer(0.12e-6, eps=3.60)
    F["slanted_over_film_oblique25"] = (st, "oblique25")

    # --- 27 a slant with a Y component ------------------------------------
    st = _st(n_orders=5)
    st.add_layer(d, eps_cell=L.BASE, slant=(0.25, 0.5))
    F["slanted_patterned_xy_conical"] = (st, "conical25_40")
    return F


def sweep_fixture():
    st = _st(n_orders=5)
    st.add_layer(0.22e-6, eps_cell=L.BASE)
    st.add_layer(0.13e-6, eps=2.25)
    st.set_source(L.WL, theta=math.radians(25.0), phi=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = st.solve_vs_wavelength(
            [0.60e-6, 0.62e-6, 0.66e-6], jones=True)
    return dict(orders=L.sha(np.asarray(o)), R=L.sha(R), T=L.sha(T),
                jones=L.sha(J))


def main():
    out = {}
    for name, (st, mount) in fixtures().items():
        res = _solve(st, mount)
        out[name] = _keys(st, res)
        out[name]["_R_sum"] = float(np.asarray(res[1]).sum())
        out[name]["_T_sum"] = float(np.asarray(res[2]).sum())
    out["sweep_3wl"] = sweep_fixture()
    n_hash = sum(1 for f in out.values() for k in f if not k.startswith("_"))
    print("fixtures %d, hashed keys %d" % (len(out), n_hash))
    L.dump("q1_hashes", dict(fixtures=out, n_fixtures=len(out),
                             n_hashed_keys=n_hash))


if __name__ == "__main__":
    main()
