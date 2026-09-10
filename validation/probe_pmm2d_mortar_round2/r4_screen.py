"""R4 -- D2: a FREE, RIGOROUS conditioning screen for the mortar's solves.

``python r4_screen.py [pop]``

The three ``np.linalg.solve`` calls the mortar adds (``_core.py`` E-row, H-row
and the generalized twin) are unguarded: below ``delta ~ 1e-7`` they raise a
bare ``LinAlgError: Singular matrix``, and above it they return numbers from an
operator whose ``cond_2`` has already reached 1e+13 (R1).

A residual screen measures nothing on a backward-stable ``solve`` (M1), and an
exact condition number costs a second O(n^3) factorisation.  What IS free and
RIGOROUS is a LOWER BOUND on the condition number read off the solve's own
output::

    X = A^-1 B   =>   ||X||_F <= ||A^-1||_2 ||B||_F
                 =>   ||A^-1||_2 >= ||X||_F / ||B||_F
                 and  ||A||_2 >= ||A||_F / sqrt(n)
                 =>   cond_2(A) >= ||A||_F ||X||_F / (sqrt(n) ||B||_F) =: g

Three Frobenius norms, O(n^2), no factorisation, and ``g > bar`` PROVES
``cond_2(A) > bar``.  This probe measures ``g`` and the exact ``cond_2`` over a
HEALTHY population (every grid pair the shipped fixtures build) and a SLIVER
population, so the bar can be set with the gap on both sides measured.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time

import numpy as np
import scipy.linalg as sla

import lumenairy
from lumenairy.elements.pmm import _core as _pc
from lumenairy.elements.pmm import twod_staggered as _ts
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
T0 = time.time()


def _log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def _lb(A, B, X):
    """The FREE lower bound g on cond_2(A)."""
    nb = float(np.linalg.norm(B))
    if nb <= 0.0:
        return 0.0
    n = A.shape[0]
    return (float(np.linalg.norm(A)) * float(np.linalg.norm(X))
            / (nb * np.sqrt(n)))


def _pair(per, wl, th, ph, wa, wb, cella, cellb, M):
    """Both mortar solve operators, their right-hand sides, the solve and the
    two instruments, for one (grid a | grid b) interface."""
    k0 = 2 * np.pi / wl
    kx0 = k0 * np.sin(th) * np.cos(ph)
    ky0 = k0 * np.sin(th) * np.sin(ph)
    taux, tauy = np.exp(-1j * kx0 * per), np.exp(-1j * ky0 * per)
    ga = StagGridOps(per, per, wa[0], wa[1], M, taux, tauy)
    gb = StagGridOps(per, per, wb[0], wb[1], M, taux, tauy)
    cr = StagCrossOps(ga, gb)
    sa = Granet2DTransverseE(per, per, wa[0], wa[1], M, cella,
                             alpha0x=kx0, alpha0y=ky0, k0=k0)
    sb = Granet2DTransverseE(per, per, wb[0], wb[1], M, cellb,
                             alpha0x=kx0, alpha0y=ky0, k0=k0)
    Wa, Va, _la, _g = _region_modes(sa)
    Wb, Vb, _lb2, _g2 = _region_modes(sb)
    out = {}
    lhsE = _pc._stag_blk2_apply(gb.V1, gb.V2, Wb, gb.qq, _stag_kron_apply)
    rhsE = _pc._stag_blk2_apply(cr.C1H(), cr.C2H(), Wa, ga.qq, _stag_kron_apply)
    hb_a, hb_c = _pc._stag_h_blocks(ga, cr)
    lhsH = _pc._stag_blk2_apply(hb_a[0], hb_a[1], Va, ga.qq, _stag_kron_apply)
    rhsH = _pc._stag_blk2_apply(hb_c[0], hb_c[1], Vb, gb.qq, _stag_kron_apply)
    for tag, (L, R) in (("E", (lhsE, rhsE)), ("H", (lhsH, rhsH))):
        rec = {"n": int(L.shape[0])}
        # (i) the EXACT reference
        rec["cond2"] = float(np.linalg.cond(L))
        # (ii) the LAPACK 1-norm rcond estimate off the SAME LU the solve does
        try:
            lu, piv = sla.lu_factor(L)
            gecon = sla.get_lapack_funcs("gecon", (L,))
            anorm = float(np.max(np.sum(np.abs(L), axis=0)))
            rc, info = gecon(lu, anorm)
            rec["rcond_gecon"] = float(rc)
            rec["gecon_info"] = int(info)
            X = sla.lu_solve((lu, piv), R)
            rec["lu_solve_bitident"] = bool(
                np.array_equal(X, np.linalg.solve(L, R)))
            rec["g"] = _lb(L, R, X)
        except Exception as exc:                            # noqa: BLE001
            rec["RAISED"] = f"{type(exc).__name__}: {str(exc)[:60]}"
        out[tag] = rec
    # and the guarded site the build already screens
    try:
        A = np.linalg.solve(lhsE, rhsE)
        B = np.linalg.solve(lhsH, rhsH)
        BA = B @ A
        I = np.eye(BA.shape[0], dtype=_C)
        out["IplusBA"] = {"cond2": float(np.linalg.cond(I + BA))}
    except Exception as exc:                                # noqa: BLE001
        out["IplusBA"] = {"RAISED": f"{type(exc).__name__}"}
    out["min_seg_frac_a"] = float(np.min(np.diff(np.asarray(wa[0]))) / per) \
        if np.ndim(wa[0]) else 1.0 / int(wa[0])
    out["min_seg_frac_b"] = float(np.min(np.diff(np.asarray(wb[0]))) / per) \
        if np.ndim(wb[0]) else 1.0 / int(wb[0])
    return out


def _full(per, w):
    return np.array([0.0] + [x * per for x in w] + [per])


def sec_pop():
    P, WL, TH, PH = 1.2, 0.85, 0.15, 0.35
    EP, EH = 9.0, 2.25

    def tile(n=3, ep=EP):
        t = np.full((n, n), _C(EH))
        t[n // 2, n // 2] = _C(ep)
        return t

    host = np.full((3, 3), _C(EH))
    healthy, sliver = {}, {}

    # ---- HEALTHY 1: the shipped mortar test's TAPER, every adjacent pair ----
    xb0, xb1 = (0.1873, 0.7241), (0.2917, 0.6109)
    slices = []
    for s in range(4):
        z = 1.0 - (s + 0.5) / 4
        slices.append((xb0[0] + (xb1[0] - xb0[0]) * z,
                       xb0[1] + (xb1[1] - xb0[1]) * z))
    for M in (4, 5, 6):
        for i in range(3):
            wa = (_full(P, slices[i]), _full(P, slices[i]))
            wb = (_full(P, slices[i + 1]), _full(P, slices[i + 1]))
            healthy[f"taper_M{M}_{i}"] = _pair(P, WL, TH, PH, wa, wb,
                                               tile(), tile(), M)
    # ---- HEALTHY 2: the shipped NON-CONFORMING pairs -----------------------
    cases = {
        "nu_vs_uniform3": ((0.2371, 0.6183), 3),
        "nu_vs_nu": ((0.2371, 0.6183), (0.3117, 0.7402)),
        "nu_vs_nu2": ((0.21, 0.55), (0.33, 0.78)),
        "uniform2_vs_uniform3": (2, 3),
        "uniform3_vs_uniform5": (3, 5),
        "nested": ((0.25, 0.75), (0.125, 0.25, 0.75, 0.875)),
        "conforming": ((0.2371, 0.6183), (0.2371, 0.6183)),
        "wall_0.4": ((0.4,), (0.6,)),
    }
    for name, (a, b) in cases.items():
        for M in (4, 6):
            wa = (a if np.ndim(a) == 0 else _full(P, a),) * 2
            wbb = (b if np.ndim(b) == 0 else _full(P, b),) * 2
            na = a if np.ndim(a) == 0 else len(a) + 1
            nb = b if np.ndim(b) == 0 else len(b) + 1
            ca = tile(na) if na % 2 else np.full((na, na), _C(EH))
            cb = tile(nb) if nb % 2 else np.full((nb, nb), _C(EH))
            try:
                healthy[f"{name}_M{M}"] = _pair(P, WL, TH, PH, wa, wbb,
                                                ca, cb, M)
            except Exception as exc:                        # noqa: BLE001
                healthy[f"{name}_M{M}"] = {"BUILD": f"{type(exc).__name__}: "
                                                    f"{str(exc)[:80]}"}
    # ---- HEALTHY 3: a genuinely FINE but ordinary feature -------------------
    for frac in (0.10, 0.05, 0.03, 0.02, 0.01):
        for M in (4, 6):
            a = (0.5 - frac / 2, 0.5 + frac / 2)
            healthy[f"fine{frac:g}_M{M}"] = _pair(
                P, WL, TH, PH,
                (_full(P, (0.21, 0.68)),) * 2, (_full(P, a),) * 2,
                tile(), host, M)
    # ---- SLIVER -------------------------------------------------------------
    for delta in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6, 1e-7):
        for M in (4, 6):
            a = (0.5 - delta / 2, 0.5 + delta / 2)
            sliver[f"d{delta:g}_M{M}"] = _pair(
                P, WL, TH, PH,
                (_full(P, (0.21, 0.68)),) * 2, (_full(P, a),) * 2,
                tile(), host, M)
    for name, d in (("healthy", healthy), ("sliver", sliver)):
        for k, v in d.items():
            if "BUILD" in v:
                _log(f"{name} {k}: {v['BUILD']}")
                continue
            _log(f"{name:8s} {k:22s} n={v['E'].get('n')}  "
                 f"E rc={v['E'].get('rcond_gecon', float('nan')):.2e} c2="
                 f"{v['E'].get('cond2', float('nan')):.2e} g="
                 f"{v['E'].get('g', float('nan')):.2f}  "
                 f"H rc={v['H'].get('rcond_gecon', float('nan')):.2e} c2="
                 f"{v['H'].get('cond2', float('nan')):.2e} g="
                 f"{v['H'].get('g', float('nan')):.2f}  "
                 f"I+BA c2={v['IplusBA'].get('cond2', float('nan')):.2e}"
                 + ("  RAISED" if any("RAISED" in v[t] for t in ("E", "H"))
                    else "")
                 + ("" if all(v[t].get("lu_solve_bitident", True)
                              for t in ("E", "H")) else "  !!NOT-BITIDENT"))
    RES["pop"] = {"healthy": healthy, "sliver": sliver}
    # ---- the two populations, summarised -----------------------------------
    def _gs(d):
        out = []
        for v in d.values():
            if "BUILD" in v:
                continue
            for t in ("E", "H"):
                if "rcond_gecon" in v[t]:
                    out.append((v[t]["g"], v[t]["cond2"],
                                v[t]["rcond_gecon"], v[t]["n"],
                                v[t].get("lu_solve_bitident")))
        return out

    hg = _gs(healthy)
    sg = _gs(sliver)
    summ = {
        "healthy_max_g": max(r[0] for r in hg),
        "healthy_min_g": min(r[0] for r in hg),
        "sliver_max_g": max(r[0] for r in sg),
        "sliver_min_g": min(r[0] for r in sg),
        "healthy_max_cond2": max(r[1] for r in hg),
        "healthy_min_rcond_gecon": min(r[2] for r in hg),
        "sliver_min_rcond_gecon": min(r[2] for r in sg),
        "worst_gecon_vs_cond2_ratio": max(r[1] * r[2] for r in hg + sg),
        "best_gecon_vs_cond2_ratio": min(r[1] * r[2] for r in hg + sg),
        "lu_solve_bitidentical": all(r[4] for r in hg + sg),
        "n_healthy_solves": len(hg), "n_sliver_solves": len(sg),
    }
    RES["summary"] = summ
    _log(f"SUMMARY {summ}")


SECTIONS = {"pop": sec_pop}


def main():
    # These probes MAP the refused band on purpose, so the round-2 guards are
    # lifted for their duration.  Library code never does this.
    _ts.PMM2D_STAG_MIN_SEG_GUARD = False
    prev_rc, _pc._MORTAR_RCOND_REFUSE = _pc._MORTAR_RCOND_REFUSE, 0.0
    try:
        for w in (sys.argv[1:] or list(SECTIONS)):
            SECTIONS[w]()
    finally:
        _ts.PMM2D_STAG_MIN_SEG_GUARD = True
        _pc._MORTAR_RCOND_REFUSE = prev_rc
    tag = os.environ.get("R_TAG", "")
    path = os.path.join(HERE, f"r4_screen{('_' + tag) if tag else ''}.json")
    with open(path, "w") as fh:
        json.dump(RES, fh, indent=1, sort_keys=True, default=float)
    _log(f"wrote {path}")


if __name__ == "__main__":
    main()
