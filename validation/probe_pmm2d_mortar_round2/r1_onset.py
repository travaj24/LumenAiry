"""R1 -- the D1 ONSET MAP: an intra-layer sliver under the mortar.

``python r1_onset.py [ladder cond fixtures shared]``

D1 (VERIFY_PMM2D_STAGGERED_MORTAR_2026_09_11 S6.3) reports that two walls
``delta`` apart INSIDE ONE layer's own non-uniform grid, with that layer
mortar-coupled to neighbours on OTHER grids, wander by up to 7.8e-03 absolute
in ``R(0,0)`` with the lossless closure pinned.  This probe maps the onset in
the quantity a guard can actually read.

THE INSTRUMENT.  The middle layer is ALL HOST, so the DEVICE is independent of
``delta`` entirely -- the walls are element boundaries in a continuous medium.
The exact reference is therefore the SAME stack with the middle layer carried
on a ONE-segment grid, converged in ``M``.  A sound discretisation must have
``err(delta, M) -> 0`` as ``M`` grows AT EVERY ``delta``; the defect is the
``delta`` at which that stops being true.  That is a DECISION (does the ladder
converge?), not a reading, and it is what an onset has to be measured in.

Sections
  ``ladder``   err vs the converged reference, over (delta, M), 3 fixtures.
  ``cond``     what the mortar does with the sliver's spurious modes: cond of
               the two mortar solve operators and of the cross-mass factors,
               with the 1/w exponent FITTED.
  ``shared``   can the SHARED-grid (union) path with ``x_walls`` be driven into
               the same band?
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
import warnings

import numpy as np

import lumenairy
from lumenairy.elements.pmm import _core as _pc
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
from lumenairy.elements.pmm.twod_staggered import (
    Granet2DTransverseE,
    StagCrossOps,
    StagGridOps,
    _region_modes,
    _stag_kron_apply,
)

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT), lumenairy.__file__
print(f"[arm] lumenairy = {lumenairy.__file__}", flush=True)

_C = complex
RES = {}
P = 1.2
WL = 0.85
TH, PH = 0.15, 0.35
EPS_P, EPS_H = 9.0, 2.25
T0 = time.time()


def _log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def _solve(st, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = st.solve(**kw)
    return r, [str(x.message)[:80] for x in w]


def _r00(o, R, T):
    p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    return (float(R[0, p0]),
            float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0))))


# ------------------------------------------------------------------ fixtures
def _tile(n=3):
    t = np.full((n, n), _C(EPS_H))
    t[n // 2, n // 2] = _C(EPS_P)
    return t


#: (name, (outer walls of layer 0), (outer walls of layer 2), thickness, eps_p)
FIXTURES = {
    # the verifier's own S6.3 fixture
    "v_s63": dict(w0=(0.21, 0.68), w2=(0.33, 0.79), t=0.06, epsp=9.0,
                  th=0.15, ph=0.35, wl=0.85, per=1.2),
    # a second period / wavelength / contrast, walls at other spacings
    "alt": dict(w0=(0.2371, 0.6183), w2=(0.3117, 0.7402), t=0.09, epsp=6.0,
                th=0.28, ph=0.11, wl=0.62, per=0.9),
    # normal incidence, high contrast, thicker slices
    "normal": dict(w0=(0.17, 0.55), w2=(0.41, 0.83), t=0.13, epsp=12.25,
                   th=0.0, ph=0.0, wl=1.05, per=1.4),
}


def _build(fx, delta, M, *, middle_uniform=False):
    """delta<0 -> the reference (middle layer on ONE segment)."""
    per = fx["per"]
    st = PMM2DStackPure(per, n_modes=M, n_orders=1, layer_grids="per-layer")
    tl = np.full((3, 3), _C(EPS_H))
    tl[1, 1] = _C(fx["epsp"])
    st.add_layer(fx["t"], eps_cell=tl,
                 x_walls=[fx["w0"][0] * per, fx["w0"][1] * per],
                 y_walls=[fx["w0"][0] * per, fx["w0"][1] * per])
    if middle_uniform:
        st.add_layer(fx["t"], eps=EPS_H, grid=1)
    else:
        host = np.full((3, 3), _C(EPS_H))
        st.add_layer(fx["t"], eps_cell=host,
                     x_walls=[(0.5 - delta / 2) * per, (0.5 + delta / 2) * per],
                     y_walls=[(0.5 - delta / 2) * per, (0.5 + delta / 2) * per])
    st.add_layer(fx["t"], eps_cell=tl,
                 x_walls=[fx["w2"][0] * per, fx["w2"][1] * per],
                 y_walls=[fx["w2"][0] * per, fx["w2"][1] * per])
    st.set_source(fx["wl"], theta=fx["th"], phi=fx["ph"])
    return st


DELTAS = (3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5)
MS = (4, 5, 6, 7)


def sec_ladder():
    out = {}
    for name, fx in FIXTURES.items():
        rows = {}
        # ---- the converged reference: middle layer on ONE segment ----------
        ref = {}
        for M in MS + (8,):
            try:
                (o, R, T), w = _solve(_build(fx, 0.0, M, middle_uniform=True),
                                      jones=False)
                r, c = _r00(o, R, T)
                ref[M] = {"R00": r, "closure": c, "warnings": w}
            except Exception as exc:                      # noqa: BLE001
                ref[M] = {"REFUSED": f"{type(exc).__name__}: {str(exc)[:100]}"}
        ok = [M for M in ref if "R00" in ref[M]]
        r_ref = ref[max(ok)]["R00"]
        selfgap = abs(ref[max(ok)]["R00"] - ref[sorted(ok)[-2]]["R00"])
        _log(f"{name}: reference R00 = {r_ref:.9f} (M={max(ok)}), "
             f"self-gap {selfgap:.3e}, closure "
             f"{ref[max(ok)]['closure']:.2e}")
        rows["reference"] = ref
        rows["ref_R00"] = r_ref
        rows["ref_selfgap"] = selfgap
        # ---- the delta ladder ----------------------------------------------
        grid = {}
        for delta in DELTAS:
            rec = {}
            for M in MS:
                t0 = time.time()
                try:
                    (o, R, T), w = _solve(_build(fx, delta, M), jones=False)
                    r, c = _r00(o, R, T)
                    rec[str(M)] = {"R00": r, "closure": c,
                                   "err": abs(r - r_ref),
                                   "warnings": w, "wall": time.time() - t0}
                except Exception as exc:                  # noqa: BLE001
                    rec[str(M)] = {
                        "REFUSED": f"{type(exc).__name__}: {str(exc)[:100]}",
                        "wall": time.time() - t0}
            grid[f"{delta:g}"] = rec
            errs = [rec[str(M)].get("err") for M in MS]
            _log(f"{name}: delta={delta:.0e}  err(M=4..7) = "
                 + " ".join("REF" if e is None else f"{e:.2e}" for e in errs)
                 + "   clo "
                 + " ".join(f"{rec[str(M)].get('closure', float('nan')):.1e}"
                            for M in MS))
        rows["grid"] = grid
        out[name] = rows
    RES["ladder"] = out


# ------------------------------------------------------------------- cond
def _mortar_cond(fx, delta, M):
    """cond_2 of the two mortar solve operators and of the cross-masses at
    the (layer0 | sliver) interface, computed by re-forming exactly what
    ``_interface_smatrix_mortar_2d`` forms."""
    per = fx["per"]
    k0 = 2 * np.pi / fx["wl"]
    n_sup = 1.0
    kx0 = k0 * n_sup * np.sin(fx["th"]) * np.cos(fx["ph"])
    ky0 = k0 * n_sup * np.sin(fx["th"]) * np.sin(fx["ph"])
    taux, tauy = np.exp(-1j * kx0 * per), np.exp(-1j * ky0 * per)
    wa = np.array([0.0, fx["w0"][0] * per, fx["w0"][1] * per, per])
    wb = np.array([0.0, (0.5 - delta / 2) * per, (0.5 + delta / 2) * per, per])
    ga = StagGridOps(per, per, wa, wa, M, taux, tauy)
    gb = StagGridOps(per, per, wb, wb, M, taux, tauy)
    cr = StagCrossOps(ga, gb)
    tl = np.full((3, 3), _C(EPS_H))
    tl[1, 1] = _C(fx["epsp"])
    host = np.full((3, 3), _C(EPS_H))
    sa = Granet2DTransverseE(per, per, wa, wa, M, tl,
                             alpha0x=kx0, alpha0y=ky0, k0=k0)
    sb = Granet2DTransverseE(per, per, wb, wb, M, host,
                             alpha0x=kx0, alpha0y=ky0, k0=k0)
    Wa, Va, lam_a, _ = _region_modes(sa)
    Wb, Vb, lam_b, _ = _region_modes(sb)
    lhsE = _pc._stag_blk2_apply(gb.V1, gb.V2, Wb, gb.qq, _stag_kron_apply)
    hb_a, hb_c = _pc._stag_h_blocks(ga, cr)
    lhsH = _pc._stag_blk2_apply(hb_a[0], hb_a[1], Va, ga.qq, _stag_kron_apply)
    # the SLIVER grid is gb; its own H-row operator is what the OTHER mortar
    # (sliver | layer2) puts in the ``a`` slot, so measure it too
    hb_b, _hb_c2 = _pc._stag_h_blocks(gb, cr)
    lhsH_b = _pc._stag_blk2_apply(hb_b[0], hb_b[1], Vb, gb.qq, _stag_kron_apply)

    def c2(A):
        try:
            return float(np.linalg.cond(np.asarray(A)))
        except Exception:                                # noqa: BLE001
            return float("inf")

    return {
        "delta": delta, "M": M,
        "Jmin_over_d": float(np.min(np.diff(wb)) / per),
        "k0_Jmin": float(k0 * 0.5 * np.min(np.diff(wb))),
        "lam_max_sliver": float(np.max(np.abs(lam_b))),
        "lam_max_plain": float(np.max(np.abs(lam_a))),
        "cond_lhsE_B": c2(lhsE),
        "cond_lhsH_A": c2(lhsH),
        "cond_lhsH_B": c2(lhsH_b),
        "cond_C1x": c2(cr.C1[1]), "cond_C1y": c2(cr.C1[0]),
        "cond_C2x": c2(cr.C2[1]), "cond_C2y": c2(cr.C2[0]),
        "cond_Wb": c2(Wb), "cond_Vb": c2(Vb),
        "cond_Wa": c2(Wa), "cond_Va": c2(Va),
        "cond_massE_B": c2(np.kron(gb.V1[0], gb.V1[1])),
    }


def sec_cond():
    out = {}
    for name in ("v_s63", "alt"):
        fx = FIXTURES[name]
        rows = {}
        for delta in DELTAS + (1e-6,):
            for M in (4, 6):
                r = _mortar_cond(fx, delta, M)
                rows[f"d{delta:g}_M{M}"] = r
            r4 = rows[f"d{delta:g}_M4"]
            _log(f"{name} cond delta={delta:.0e}: k0J={r4['k0_Jmin']:.2e} "
                 f"|lam|max {r4['lam_max_sliver']:.2e}  lhsE {r4['cond_lhsE_B']:.2e} "
                 f"lhsH_A {r4['cond_lhsH_A']:.2e} lhsH_B {r4['cond_lhsH_B']:.2e} "
                 f"C1x {r4['cond_C1x']:.2e} Vb {r4['cond_Vb']:.2e}")
        # fit exponents in 1/delta over the tail
        fits = {}
        tail = [d for d in DELTAS + (1e-6,) if d <= 1e-2]
        for key in ("cond_lhsE_B", "cond_lhsH_A", "cond_lhsH_B", "cond_C1x",
                    "cond_C2x", "cond_Wb", "cond_Vb", "lam_max_sliver",
                    "cond_massE_B"):
            for M in (4, 6):
                y = np.array([rows[f"d{d:g}_M{M}"][key] for d in tail])
                x = np.array([1.0 / d for d in tail])
                good = np.isfinite(y) & (y > 0)
                if good.sum() >= 3:
                    p = np.polyfit(np.log(x[good]), np.log(y[good]), 1)
                    fits[f"{key}_M{M}"] = float(p[0])
        rows["exponent_in_inv_delta"] = fits
        _log(f"{name} FITTED exponents in 1/delta: "
             + ", ".join(f"{k}={v:.3f}" for k, v in fits.items()
                         if k.endswith("_M6")))
        out[name] = rows
    RES["cond"] = out


# ------------------------------------------------------------------- shared
def sec_shared():
    """Can the SHARED (union pixel-lattice) path be driven into the band?

    ``layer_grids='shared'`` takes a common ``(Nx, Ny)`` PIXEL lattice, so its
    cells are ``period/N`` and the aspect ratio is exactly 1 -- the sliver is
    not constructible there.  Measured directly: does ``add_layer(x_walls=...)``
    even reach the shared path?"""
    out = {}
    fx = FIXTURES["v_s63"]
    per = fx["per"]
    for spelling in ("shared_with_walls", "shared_default"):
        try:
            st = PMM2DStackPure(per, n_modes=5, n_orders=1,
                                layer_grids="shared")
            host = np.full((3, 3), _C(EPS_H))
            if spelling == "shared_with_walls":
                st.add_layer(0.06, eps_cell=host,
                             x_walls=[0.4999 * per, 0.5001 * per],
                             y_walls=[0.4999 * per, 0.5001 * per])
            else:
                st.add_layer(0.06, eps_cell=host, grid=3)
            st.set_source(fx["wl"], theta=fx["th"], phi=fx["ph"])
            (o, R, T), w = _solve(st, jones=False)
            r, c = _r00(o, R, T)
            out[spelling] = {"R00": r, "closure": c, "warnings": w}
        except Exception as exc:                            # noqa: BLE001
            out[spelling] = {"REFUSED": f"{type(exc).__name__}: {str(exc)[:200]}"}
        _log(f"shared/{spelling}: {out[spelling]}")
    # and the geometric fact: what widths can the shared lattice produce?
    aspects = {}
    for N in (3, 5, 8, 13, 24):
        b = np.linspace(0.0, per, N + 1)
        aspects[str(N)] = {"min_over_d": float(np.min(np.diff(b)) / per),
                           "aspect": float(np.max(np.diff(b))
                                           / np.min(np.diff(b)))}
    out["shared_lattice_widths"] = aspects
    _log(f"shared lattice widths: {aspects}")
    RES["shared"] = out


SECTIONS = {"ladder": sec_ladder, "cond": sec_cond, "shared": sec_shared}


def main():
    for w in (sys.argv[1:] or list(SECTIONS)):
        SECTIONS[w]()
    tag = os.environ.get("R_TAG", "")
    path = os.path.join(HERE, f"r1_onset{('_' + tag) if tag else ''}.json")
    with open(path, "w") as fh:
        json.dump(RES, fh, indent=1, sort_keys=True, default=float)
    _log(f"wrote {path}")


if __name__ == "__main__":
    main()
