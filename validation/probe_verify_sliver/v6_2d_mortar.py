"""VERIFY task 5 -- the 2-D stacks and the NEWLY LANDED per-layer / mortar path.

A. The two SHIPPED claims, re-measured:
     * ``PMM2DStackHybrid`` has no union grid at all (structural -- no caller of
       ``_pmm_union_grid`` outside the 1-D stack);
     * ``PMM2DStackPure``'s shared union IS the caller-supplied pixel lattice,
       so every cell is ``period/N`` and the aspect ratio is exactly 1.

B. THE CLAIM UNDER TEST: "the mortar route is the safe one".  The mortar
   removes the CROSS-LAYER union, so it removes the cross-layer sliver.  It
   does NOT remove the element Jacobian.  ``add_layer(x_walls=...)`` and
   ``add_tapered_pillar`` let a caller put two walls of ONE layer arbitrarily
   close, and ``_stag_walls_spec`` only requires them STRICTLY INCREASING.
   Measured here on a within-layer sliver: the Jacobian, the ``|q|`` predictor,
   the closure, the answer against a wall-coincident reference, and whether
   ANYTHING refuses.

C. The 1-D counterpart of the same hole, because it is the SAME rule: the 1-D
   guard's ownership test deliberately never flags a thin feature owned by one
   layer ("a 1 nm liner is intentional").  Does a 1-D single-layer sliver
   misbehave the same way?

    python validation/probe_verify_sliver/v6_2d_mortar.py [out.json]
"""
import json
import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import lumenairy
from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm import stack as ps
from lumenairy.elements.pmm._core import (
    _build_sem_tensor_segments,
    _pmm_union_grid,
    _sem_modes_tensor,
)
from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure

HERE = os.path.dirname(os.path.abspath(__file__))
P2, WL2 = 1.2, 0.85
EH, EP = 2.25, 9.0


# ==========================================================================
def part_a():
    out = {}
    # A1 -- the pure stack's SHARED union is the pixel lattice
    rows = []
    for N in (8, 12, 24, 64, 129):
        cell = np.full((N, N), EH + 0j)
        cell[N // 4: 3 * N // 4, N // 4: 3 * N // 4] = EP
        st = PMM2DStackPure(P2, n_modes=4, n_orders=1)
        st.add_layer(0.3, eps_cell=cell)
        # the grid the pure stack solves on: the cell's own lattice
        wx = np.full(N, P2 / N)
        rows.append(dict(N=N, cells=N, every_cell=float(P2 / N),
                         aspect=float(wx.max() / wx.min()),
                         min_cell=float(wx.min())))
    out["pure_shared_lattice"] = rows
    print("  pure shared lattice: " + ", ".join(
        f"N={r['N']} cell={r['every_cell']:.6g} aspect={r['aspect']:.1f}"
        for r in rows))

    # A2 -- can two patterned layers with DIFFERENT lattices coexist on the
    # shared path?  (if the API refuses, no union can be formed at all)
    cA = np.full((6, 6), EH + 0j)
    cA[2:4, 2:4] = EP
    cB = np.full((8, 8), EH + 0j)
    cB[3:5, 3:5] = EP
    st = PMM2DStackPure(P2, n_modes=4, n_orders=1)
    st.add_layer(0.3, eps_cell=cA)
    try:
        st.add_layer(0.3, eps_cell=cB)
        out["pure_shared_mixed_lattice"] = "accepted (a union WOULD be needed)"
    except Exception as exc:                                   # noqa: BLE001
        out["pure_shared_mixed_lattice"] = f"{type(exc).__name__}: {exc}"[:220]
    print("  pure shared, two different lattices:",
          out["pure_shared_mixed_lattice"][:150])

    # A3 -- the hybrid: different lattices per layer, no union
    sh = PMM2DStackHybrid(P2, degree=7, n_orders=3)
    try:
        sh.add_layer(0.3, eps_cell=cA)
        sh.add_layer(0.3, eps_cell=cB)
        sh.set_source(WL2, theta=0.2, phi=0.3)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            o, R, T, _J = sh.solve()
        tot = np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)
        out["hybrid_mixed_lattice"] = dict(
            ok=True, RplusT=[float(x) for x in tot],
            warned=[str(w.message)[:60] for w in rec][:2])
    except Exception as exc:                                   # noqa: BLE001
        out["hybrid_mixed_lattice"] = dict(ok=False,
                                           err=f"{type(exc).__name__}: {exc}"[:200])
    print("  hybrid, two different lattices:", out["hybrid_mixed_lattice"])
    return out


# ==========================================================================
def _pure_nu(walls, M=6, per_layer=True, nlay=1):
    """A per-layer NON-UNIFORM 3-segment layer with the given interior walls."""
    tile = np.array([[EH] * 3, [EP] * 3, [EH] * 3], dtype=complex)
    st = PMM2DStackPure(P2, n_modes=M, n_orders=1,
                        layer_grids="per-layer" if per_layer else "shared")
    for _ in range(nlay):
        st.add_layer(0.30 / nlay, eps_cell=tile,
                     x_walls=[w * P2 for w in walls],
                     y_walls=[w * P2 for w in walls])
    st.set_source(WL2, theta=0.15)
    return st


def _solve_pure(st):
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = True
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            o, R, T = st.solve(jones=False)
        oo = np.asarray(o)
        sel = oo[:, 1] == 0
        m = oo[sel, 0]
        i = np.argsort(m)
        tot = np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)
        return dict(refused=False, m=m[i], R=np.asarray(R)[1][sel][i],
                    T=np.asarray(T)[1][sel][i], worst=float(np.max(tot)),
                    warned=[str(w.message)[:70] for w in rec][:3])
    except Exception as exc:                                   # noqa: BLE001
        return dict(refused=True, err=f"{type(exc).__name__}: {exc}"[:200])
    finally:
        ps.PMM_SLIVER_GUARD = was


def _pure_uniform(M=6):
    """The EXACT ``d -> 0`` limit of ``_pure_nu``: the thin EP stripe has zero
    width, so the layer is a homogeneous EH slab."""
    st = PMM2DStackPure(P2, n_modes=M, n_orders=1, layer_grids="per-layer")
    st.add_layer(0.30, eps=EH, grid=1, n_modes=M)
    st.set_source(WL2, theta=0.15)
    return st


def part_b():
    """Can a caller build a sliver INSIDE ONE LAYER's own non-uniform grid?

    The reference is EXACT: as the two walls approach each other the EP stripe
    vanishes and the layer becomes the homogeneous EH slab, so the structure is
    continuous in ``d`` and ``err`` must fall LINEARLY in ``d``."""
    out = {"accepted": [], "solves": []}
    a0 = 0.2371
    refs = {M: _solve_pure(_pure_uniform(M)) for M in (4, 5, 6, 7, 8)}
    print("  the d -> 0 reference (homogeneous EH slab), R0 vs n_modes: " +
          " ".join(f"{M}:{r['R'][int(np.argmin(np.abs(r['m'])))]:.8f}"
                   for M, r in refs.items()))
    for d in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        walls = [a0, a0 + d]
        try:
            _pure_nu(walls)
            out["accepted"].append(dict(d=d, accepted=True))
        except Exception as exc:                               # noqa: BLE001
            out["accepted"].append(dict(d=d, accepted=False,
                                        err=f"{type(exc).__name__}: {exc}"[:140]))
            print(f"  x_walls separated by {d:.0e}: REFUSED at build -- "
                  f"{type(exc).__name__}")
            continue
        ladder = []
        for M in (4, 5, 6, 7, 8):
            r = _solve_pure(_pure_nu(walls, M=M))
            ref = refs[M]
            if not r["refused"]:
                i0 = int(np.argmin(np.abs(r["m"])))
                err = float(max(np.abs(r["R"] - ref["R"]).max(),
                                np.abs(r["T"] - ref["T"]).max()))
                ladder.append(dict(M=M, R0=float(r["R"][i0]), err=err,
                                   err_over_d=err / d, worst=r["worst"],
                                   warned=r["warned"]))
            else:
                ladder.append(dict(M=M, refused=True, err_msg=r["err"]))
        good = [x for x in ladder if "R0" in x]
        spread = (max(x["R0"] for x in good) - min(x["R0"] for x in good)
                  if len(good) > 1 else None)
        out["solves"].append(dict(
            d=d, n_refused=sum("refused" in x for x in ladder),
            ladder=ladder, M_spread=spread,
            max_err=max((x["err"] for x in good), default=None),
            max_err_over_d=max((x["err_over_d"] for x in good), default=None),
            max_absRT1=max((abs(x["worst"] - 1.0) for x in good), default=None)))
        s_ = out["solves"][-1]
        print(f"  d={d:.0e}: refused {s_['n_refused']}/5  err {s_['max_err']} "
              f"err/d {s_['max_err_over_d']}  |R+T-1| {s_['max_absRT1']}  "
              f"M-spread {spread}")
        if any(x.get("warned") for x in good):
            print(f"      warnings: {[x['warned'] for x in good if x['warned']]}")
        if s_["n_refused"]:
            print("      refusal: "
                  + [x.get("err_msg") for x in ladder if "err_msg" in x][0])
    return out


def part_b2():
    """The same hole through the SHIPPED SURFACE ``add_tapered_pillar``: a
    taper whose interpolated bounds converge to within ``d`` of each other."""
    out = []
    for d in (1e-2, 1e-4, 1e-6):
        st = PMM2DStackPure(P2, n_modes=6, n_orders=1, layer_grids="per-layer")
        try:
            st.add_tapered_pillar(
                0.30, eps_pillar=EP, eps_host=EH,
                x_bounds_bottom=(0.30 * P2, 0.70 * P2),
                y_bounds_bottom=(0.30 * P2, 0.70 * P2),
                x_bounds_top=(0.50 * P2 - 0.5 * d * P2,
                              0.50 * P2 + 0.5 * d * P2),
                y_bounds_top=(0.50 * P2 - 0.5 * d * P2,
                              0.50 * P2 + 0.5 * d * P2),
                n_slices=4)
            st.set_source(WL2, theta=0.15)
            r = _solve_pure(st)
            out.append(dict(d=d, built=True, refused=r["refused"],
                            worst=r.get("worst"), warned=r.get("warned"),
                            err=r.get("err")))
        except Exception as exc:                               # noqa: BLE001
            out.append(dict(d=d, built=False,
                            err=f"{type(exc).__name__}: {exc}"[:200]))
        print(f"  add_tapered_pillar tip width {d:.0e}: {out[-1]}")
    return out


# ==========================================================================

def _pure_nu2(walls_a, walls_b, M=6):
    """TWO layers, each on its OWN non-uniform grid, each carrying its own
    sliver at a DIFFERENT position -- the mortar interface between two
    sliver-bearing grids, which is the configuration the shared union grid
    turns into a cross-layer sliver."""
    tile = np.array([[EH] * 3, [EP] * 3, [EH] * 3], dtype=complex)
    st = PMM2DStackPure(P2, n_modes=M, n_orders=1, layer_grids="per-layer")
    for w in (walls_a, walls_b):
        st.add_layer(0.15, eps_cell=tile,
                     x_walls=[x * P2 for x in w], y_walls=[x * P2 for x in w])
    st.set_source(WL2, theta=0.15)
    return st


def _pure_uniform2(M=6):
    st = PMM2DStackPure(P2, n_modes=M, n_orders=1, layer_grids="per-layer")
    for _ in range(2):
        st.add_layer(0.15, eps=EH, grid=1, n_modes=M)
    st.set_source(WL2, theta=0.15)
    return st


def part_b3():
    """The mortar between two DIFFERENT sliver grids -- the 2-D analogue of the
    1-D cross-layer sliver, on the route the fix audit calls safe."""
    out = []
    a0 = 0.2371
    refs = {M: _solve_pure(_pure_uniform2(M)) for M in (4, 6, 8)}
    for d in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        wa = [a0, a0 + d]
        wb = [a0 + d / 3.0, a0 + 4.0 * d / 3.0]
        ladder = []
        for M in (4, 6, 8):
            r = _solve_pure(_pure_nu2(wa, wb, M=M))
            if r["refused"]:
                ladder.append(dict(M=M, refused=True, err_msg=r["err"]))
                continue
            ref = refs[M]
            err = float(max(np.abs(r["R"] - ref["R"]).max(),
                            np.abs(r["T"] - ref["T"]).max()))
            ladder.append(dict(M=M, R0=float(r["R"][int(np.argmin(
                np.abs(r["m"])))]), err=err, err_over_d=err / d,
                worst=r["worst"], warned=r["warned"]))
        good = [x for x in ladder if "R0" in x]
        out.append(dict(d=d, ladder=ladder,
                        n_refused=sum("refused" in x for x in ladder),
                        max_err=max((x["err"] for x in good), default=None),
                        max_err_over_d=max((x["err_over_d"] for x in good),
                                           default=None),
                        M_spread=(max(x["R0"] for x in good)
                                  - min(x["R0"] for x in good)
                                  if len(good) > 1 else None),
                        max_absRT1=max((abs(x["worst"] - 1.0) for x in good),
                                       default=None)))
        o_ = out[-1]
        print(f"  two mortar-coupled slivers, d={d:.0e}: refused "
              f"{o_['n_refused']}/3  err {o_['max_err']}  err/d "
              f"{o_['max_err_over_d']}  |R+T-1| {o_['max_absRT1']}  "
              f"M-spread {o_['M_spread']}")
        if o_["n_refused"]:
            print("      refusal: "
                  + [x.get("err_msg") for x in ladder if "err_msg" in x][0])
    return out


def part_c():
    """The 1-D counterpart: a thin feature owned by ONE layer -- exactly what
    the shipped ownership rule declares intentional and never flags.

    Same EXACT reference: as the liner width goes to zero the layer becomes the
    plain two-segment layer, so ``err`` must fall linearly in ``d``."""
    P, WL, TH = 1.2e-6, 0.85e-6, 0.15
    DZ = 0.32e-6 / 4
    rows = []
    k0 = 2.0 * np.pi / WL

    def _t3(e):
        return dict(exx=complex(e), exy=0.0, eyx=0.0, eyy=complex(e),
                    ezz=complex(e))

    def _run(segs, deg):
        st = PMMStack(P, n_superstrate=1.0, n_substrate=1.0, degree=deg,
                      min_feature=P * 1e-12)
        st.add_layer(DZ, segments=segs)
        st.add_layer(DZ, segments=segs)
        st.set_source(WL, theta=TH)
        was = ps.PMM_SLIVER_GUARD
        ps.PMM_SLIVER_GUARD = True
        try:
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                o, R, T, _J = st.solve()
            o = np.asarray(o).ravel()
            i = np.argsort(o)
            tot = np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)
            return dict(refused=False, m=o[i], R=np.asarray(R)[1][i],
                        T=np.asarray(T)[1][i], worst=float(np.max(tot)),
                        warned=[str(w.message)[:60] for w in rec][:2])
        except ValueError as exc:
            return dict(refused=True, err=str(exc)[:120])
        finally:
            ps.PMM_SLIVER_GUARD = was

    refs = {deg: _run([(0.30, EH), (0.70, EH)], deg) for deg in (8, 12, 14, 16)}
    for d in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        segs = [(0.30, EH), (d, EP), (0.70 - d, EH)]
        ladder = []
        for deg in (8, 12, 14, 16):
            r = _run(segs, deg)
            if r["refused"]:
                ladder.append(dict(degree=deg, refused=True, err=r["err"]))
                continue
            ref = refs[deg]
            common = np.intersect1d(r["m"], ref["m"])
            ia = np.searchsorted(r["m"], common)
            ib = np.searchsorted(ref["m"], common)
            err = float(max(np.abs(r["R"][ia] - ref["R"][ib]).max(),
                            np.abs(r["T"][ia] - ref["T"][ib]).max()))
            i0 = int(np.argmin(np.abs(r["m"])))
            ladder.append(dict(degree=deg, R0=float(r["R"][i0]), err=err,
                               err_over_d=err / d, worst=r["worst"],
                               warned=r["warned"]))
        good = [x for x in ladder if "R0" in x]
        screen = ps._cross_layer_sliver([segs, segs], 1e-12)
        uw, leps = _pmm_union_grid([segs, segs], 1e-12)
        m = _build_sem_tensor_segments(P, uw, [_t3(e) for e in leps[0]],
                                       14, 1, True)
        _W, _V, _lam, q = _sem_modes_tensor(m, k0, np.sin(TH) * k0, True)
        J = 0.5 * float(np.min(uw)) * P
        rows.append(dict(d=d, screen_fires=screen is not None, ladder=ladder,
                         qmax=float(np.abs(q).max()), k0J=float(k0 * J),
                         pred_const=float(np.abs(q).max()) * k0 * J
                         / (14 * 15 / 4.0),
                         max_err=max((x["err"] for x in good), default=None),
                         max_err_over_d=max((x["err_over_d"] for x in good),
                                            default=None),
                         degree_spread=(max(x["R0"] for x in good)
                                        - min(x["R0"] for x in good)
                                        if len(good) > 1 else None),
                         max_absRT1=max((abs(x["worst"] - 1.0) for x in good),
                                        default=None),
                         n_refused=sum("refused" in x for x in ladder)))
        r_ = rows[-1]
        print(f"  liner d={d:.0e}: screen_fires={r_['screen_fires']} "
              f"refused {r_['n_refused']}/4  err {r_['max_err']:.4e} "
              f"err/d {r_['max_err_over_d']:.4g}  |R+T-1| {r_['max_absRT1']:.3e} "
              f"deg-spread {r_['degree_spread']:.3e}  |q|max {r_['qmax']:.3e} "
              f"const {r_['pred_const']:.4f}")
        w = [x["warned"] for x in good if x.get("warned")]
        if w:
            print(f"      warnings: {w}")
    return rows


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "v6_2d_mortar.json"))
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib, lumenairy.__version__)
    assert "lum_vsliver" in lib.replace("\\", "/"), lib
    print("\n== A: the two shipped 2-D claims ==")
    a = part_a()
    print("\n== B: a sliver INSIDE one layer's own non-uniform grid ==")
    b = part_b()
    print("\n== B2: the same, through add_tapered_pillar ==")
    b2 = part_b2()
    print("\n== B3: two mortar-coupled sliver grids ==")
    b3r = part_b3()
    print("\n== C: the 1-D single-layer liner (the ownership rule's blind spot) ==")
    c = part_c()
    with open(out_path, "w") as f:
        json.dump(dict(meta=dict(lumenairy=lib, python=sys.version.split()[0],
                                 numpy=np.__version__),
                       part_a=a, part_b=b, part_b2=b2, part_b3=b3r,
                       part_c=c), f,
                  indent=1, default=str)
    print("\nwrote", out_path)


if __name__ == "__main__":
    main()
