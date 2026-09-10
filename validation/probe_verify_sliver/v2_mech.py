"""VERIFY task 2 -- the MECHANISM, re-measured on my own fixture.

Three independent things:

A. On MY fixture (period 0.9 um, wl 0.62 um, theta 0.21, eps 2.0 / 6.5, walls
   0.311 / 0.688) at 3 degrees x 4 sliver widths: the nodal ``Kx^2`` norm, the
   modal ``|q|max``, the predictor constant ``|q|max k0 J / (N(N+1)/4)``, the
   interface mode-match conditioning and the largest interface S entry -- with
   the observed power of ``1/w`` fitted between consecutive widths, so the
   claimed exponents are read off the data and not assumed.

B. On the O-11 fixture verbatim (period 1.2 um, wl 0.85 um, theta 0.15, walls
   0.27865 / 0.62505): the degree-12-vs-14 SELF-GAP that ``f5f_attrib.py``
   printed, on BOTH ``layer_grids`` spellings, plus the stack's own ``R+T`` and
   whether ``_warn_stack_energy`` fires -- the two corrections to O-11.

C. Whether the per-layer window at halfwidth 1 IS the union on a 2-layer stack
   (grid-level, not just answer-level) and what happens on a LONGER stack.

    python validation/probe_verify_sliver/v2_mech.py [out.json]
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
    _interface_smatrix,
    _pmm_union_grid,
    _safe_inv,
    _sem_modes_tensor,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- MY fixture ----------------
MP, MWL, MTH = 0.9e-6, 0.62e-6, 0.21
MEH, MEP = 2.0, 6.5
MA, MB = 0.311, 0.688
MDZ = 70e-9

# ---------------- the O-11 fixture, verbatim ----------------
OP, OWL, OTH = 1.2e-6, 0.85e-6, 0.15
OEH, OEP = 2.25, 9.0
OA, OB = 0.27865, 0.62505
ODZ = 0.32e-6 / 4


def _segs(a, b, eh, ep):
    return [(a, eh), (b - a, ep), (1.0 - b, eh)]


def _t3(e):
    return dict(exx=complex(e), exy=0.0, eyx=0.0, eyy=complex(e),
                ezz=complex(e))


# ==========================================================================
# A -- the scaling laws on my own fixture
# ==========================================================================
def part_a():
    k0 = 2.0 * np.pi / MWL
    kx0 = np.sin(MTH) * k0
    rows = []
    for delta in (1e-2, 1e-3, 1e-4, 3e-5):
        segs = [_segs(MA, MB, MEH, MEP),
                _segs(MA - delta, MB + delta, MEH, MEP)]
        uw, leps = _pmm_union_grid(segs, 1e-12)     # snap dormant
        w = float(np.min(uw))
        J = 0.5 * w * MP
        for deg in (10, 14, 18):
            mats = [_build_sem_tensor_segments(
                MP, uw, [_t3(e) for e in leps[i]], deg, 1, True)
                for i in (0, 1)]
            S0 = mats[0]["S0"]
            iS0 = _safe_inv(S0)
            Kx2 = (iS0 @ mats[0]["stiff"]["one"]) / (k0 * k0)
            out = []
            for m in mats:
                W, V, lam, q = _sem_modes_tensor(m, k0, kx0, True)
                out.append((W, V, lam, q))
            qmax = max(float(np.abs(o[3]).max()) for o in out)
            Wa, Va = out[0][0], out[0][1]
            Wb, Vb = out[1][0], out[1][1]
            a_ = np.linalg.solve(Wb, Wa)
            b_ = np.linalg.solve(Vb, Va)
            cond_iface = float(np.linalg.cond(a_ + b_))
            cond_Wa = float(np.linalg.cond(Wa))
            cond_Wb = float(np.linalg.cond(Wb))
            cond_Va = float(np.linalg.cond(Va))
            cond_Vb = float(np.linalg.cond(Vb))
            # the FULL mode-match block system, which is what the fix audit's
            # "cond interface" column is: [[Wa, -Wb], [Va, Vb]].
            cond_block = float(np.linalg.cond(
                np.block([[Wa, -Wb], [Va, Vb]])))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    S = _interface_smatrix(Wa, Va, Wb, Vb)
                    smax = float(max(np.abs(np.asarray(x)).max() for x in S))
                except Exception as exc:              # noqa: BLE001
                    smax = float("nan")
                    print("   interface refused:", type(exc).__name__)
            rows.append(dict(
                delta=delta, w=w, n_cells=int(uw.size), k0J=float(k0 * J),
                degree=deg,
                cond_S0=float(np.linalg.cond(S0)),
                norm_Kx2=float(np.linalg.norm(Kx2, 2)),
                qmax=qmax,
                pred_const=qmax * k0 * J / (deg * (deg + 1) / 4.0),
                cond_iface=cond_iface, cond_block=cond_block,
                cond_Wa=cond_Wa, cond_Wb=cond_Wb,
                cond_Va=cond_Va, cond_Vb=cond_Vb, smax=smax))
            print(f"  delta={delta:.1e} w={w:.3e} deg={deg:2d} "
                  f"k0J={k0*J:.4e} cond(S0)={rows[-1]['cond_S0']:.3e} "
                  f"|Kx2|={rows[-1]['norm_Kx2']:.4e} |q|max={qmax:.4e} "
                  f"const={rows[-1]['pred_const']:.4f} "
                  f"cond(a+b)={cond_iface:.3e} cond(blk)={cond_block:.3e} "
                  f"cond(Vb)={cond_Vb:.3e} maxS={smax:.4e}", flush=True)
    return rows


def _powers(rows):
    """Fit the exponent p in X ~ w^-p between consecutive widths, per degree."""
    out = {}
    for deg in sorted({r["degree"] for r in rows}):
        rs = [r for r in rows if r["degree"] == deg]
        rs.sort(key=lambda r: -r["w"])
        for key in ("cond_S0", "norm_Kx2", "qmax", "cond_iface",
                    "cond_block", "cond_Wa", "cond_Wb", "cond_Va", "cond_Vb"):
            ps_ = []
            for i in range(len(rs) - 1):
                x0, x1 = rs[i][key], rs[i + 1][key]
                w0, w1 = rs[i]["w"], rs[i + 1]["w"]
                if x0 > 0 and x1 > 0:
                    ps_.append(float(np.log(x1 / x0) / np.log(w0 / w1)))
            out[f"deg{deg}_{key}"] = ps_
    return out


# ==========================================================================
# B -- the O-11 self-gaps, the closure and the warning
# ==========================================================================
def _orc(frames, deg, per_layer=False, mf=None, guard=True):
    kw = dict(layer_grids="per-layer") if per_layer else {}
    if mf is not None:
        kw["min_feature"] = mf
    s = PMMStack(OP, n_superstrate=1.0, n_substrate=1.0, degree=deg, **kw)
    for (a, b) in frames:
        s.add_layer(ODZ, segments=_segs(a, b, OEH, OEP))
    s.set_source(OWL, theta=OTH)
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = bool(guard)
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            o, R, T = s.solve()[:3]
    finally:
        ps.PMM_SLIVER_GUARD = was
    o = np.asarray(o).ravel()
    i = np.argsort(o)
    msgs = [str(x.message) for x in rec]
    return o[i], R[1][i], T[1][i], msgs


def part_b():
    ref = _orc([(OA, OB), (OA, OB)], 14, guard=False)
    rows = []
    for d in (1e-2, 2.6e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 0.0):
        fr = [(OA, OB), (OA - d, OB + d)]
        m14, R14, T14, msg14 = _orc(fr, 14, guard=False)
        _m, R12, T12, _ = _orc(fr, 12, guard=False)
        sg = float(max(np.abs(R14 - R12).max(), np.abs(T14 - T12).max()))
        try:
            mp14, RP14, TP14, _ = _orc(fr, 14, per_layer=True, guard=False)
            _m, RP12, TP12, _ = _orc(fr, 12, per_layer=True, guard=False)
            sgp = float(max(np.abs(RP14 - RP12).max(),
                            np.abs(TP14 - TP12).max()))
            plgap = float(max(np.abs(R14 - RP14).max(),
                              np.abs(T14 - TP14).max()))
        except Exception as exc:                              # noqa: BLE001
            sgp, plgap = float("nan"), float("nan")
            print("  per-layer:", type(exc).__name__, str(exc)[:80])
        shift = float(max(np.abs(R14 - ref[1]).max(), np.abs(T14 - ref[2]).max()))
        tot = float(R14.sum() + T14.sum())
        warned = any("energy not conserved" in m for m in msg14)
        # and with the guard ARMED, does it refuse?
        try:
            _orc(fr, 14, guard=True)
            refused = False
        except ValueError as exc:
            refused = "NEAR-COINCIDENT-WALL SLIVER" in str(exc)
        rows.append(dict(delta=d, sliver_m=d * OP, selfgap_shared=sg,
                         selfgap_perlayer=sgp, perlayer_vs_shared=plgap,
                         shift_from_ref=shift, RplusT=tot,
                         abs_RT_minus_1=abs(tot - 1.0),
                         energy_warned=bool(warned), refused=bool(refused),
                         warnings=msg14[:2]))
        print(f"  delta={d:.2e} sg_shared={sg:.3e} sg_PL={sgp:.3e} "
              f"PLvsSH={plgap:.3e} shift={shift:.3e} R+T={tot:.6g} "
              f"warned={warned} refused={refused}", flush=True)
    return rows


# ==========================================================================
# C -- is the per-layer window IS the union?  grid-level, 2 vs 5 layers
# ==========================================================================
def part_c():
    out = {}
    from lumenairy.elements.pmm._core import _perlayer_window_grids
    for nlay in (2, 3, 5):
        frames = [(OA - 1e-4 * i, OB + 1e-4 * i) for i in range(nlay)]
        segs = [_segs(a, b, OEH, OEP) for (a, b) in frames]
        uw, _le = _pmm_union_grid(segs, 1e-12)
        try:
            g = _perlayer_window_grids(segs, 1e-12, halfwidth=1)
            widths = [np.asarray(x[0]) for x in g]
            same = [bool(len(w) == len(uw) and np.array_equal(w, uw))
                    for w in widths]
        except Exception as exc:                              # noqa: BLE001
            same = f"{type(exc).__name__}: {exc}"
            widths = []
        out[f"n{nlay}"] = dict(union_cells=int(uw.size),
                               union_min=float(uw.min()),
                               per_layer_equals_union=same,
                               per_layer_cells=[int(np.size(w)) for w in widths])
        print(f"  n_layers={nlay}: union cells={uw.size} min={uw.min():.3e} "
              f"per-layer==union: {same}", flush=True)
    return out


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "v2_mech.json"))
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib, lumenairy.__version__)
    assert "lum_vsliver" in lib.replace("\\", "/"), lib
    print("\n== A: scaling on my own fixture ==")
    a = part_a()
    p = _powers(a)
    print("\n  fitted exponents (X ~ w^-p):")
    for k in sorted(p):
        print(f"    {k:26s} {['%.3f' % v for v in p[k]]}")
    print("\n== B: the O-11 fixture, self-gaps + closure + warning ==")
    b = part_b()
    print("\n== C: per-layer window vs the union ==")
    c = part_c()
    with open(out_path, "w") as f:
        json.dump(dict(meta=dict(lumenairy=lib, python=sys.version.split()[0],
                                 numpy=np.__version__),
                       part_a=a, powers=p, part_b=b, part_c=c), f, indent=1)
    print("\nwrote", out_path)


if __name__ == "__main__":
    main()
