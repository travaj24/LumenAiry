"""VERIFY round 2, task 5 -- D3, the projector's quadrature order, re-measured.

``python v5_d3.py need rule kernel integer candidate nqmap``

``need``       the SMALLEST Gauss order that reaches a refined rule's OWN
               floor, over MY OWN (omega, M) points, with the bar derived at
               each point from the reference rule's self-drift.
``rule``       the shipped ``_stag_quad_order`` scored against that
               requirement (margin per cell).
``kernel``     the relative kernel error BEFORE (the fixed ``2M+8`` rule) and
               AFTER, on long segments up to 0.96 d.
``integer``    the shipped rule returns exactly ``2M+8`` on every uniform
               lattice ``M = 3..14`` x ``N = 1..60`` with ``|alpha0| <= G/2``.
``candidate``  the REJECTED candidate ``max(2M+8, ceil(0.75 w) + M + 8)`` on
               the same 720 cells.
``nqmap``      what the rule hands out on ORDINARY per-layer geometries --
               i.e. how often D3 changes anything at all.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from numpy.polynomial.legendre import leggauss  # noqa: E402

import lumenairy  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as _ts  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Basis1D,
    _modleg_value_deriv,
    _stag_fourier_projection,
    _stag_quad_order,
)

HERE = os.path.dirname(os.path.abspath(__file__))
print(f"[arm] lumenairy = {lumenairy.__file__} v{lumenairy.__version__}",
      flush=True)
TAG = os.environ.get("V5_TAG", "win")
T0 = time.time()
RES = {}


def _log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def _kernel(M, omega, nq):
    """INT_-1^1 Ltilde_a(u) exp(i omega u) du by an nq-node Gauss rule, for
    every modified-Legendre index a.  This IS the projector's inner sum with
    the segment's affine map folded into ``omega``."""
    xg, wg = leggauss(nq)
    V, _ = _modleg_value_deriv(M, xg)
    return (np.exp(1j * omega * xg) * wg) @ V.T


# =================================================================== need
def sec_need():
    """The bar is the REFERENCE rule's own self-drift, per point."""
    out = {}
    Ms = (3, 4, 5, 6, 7, 8, 10, 12)
    omegas = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
    tab = {}
    for M in Ms:
        row = {}
        for w in omegas:
            nref = int(max(200, 4 * w + 60))
            ref = _kernel(M, w, nref)
            drift = np.max(np.abs(_kernel(M, w, nref + 37) - ref))
            scale = max(np.max(np.abs(ref)), 1e-300)
            bar = max(3.0 * drift, 0.0) / scale
            need = None
            for nq in range(2, 200):
                e = np.max(np.abs(_kernel(M, w, nq) - ref)) / scale
                if e <= bar:
                    need = nq
                    break
            row[f"{w:g}"] = dict(need=need, bar=float(bar),
                                 self_drift=float(drift / scale))
        tab[str(M)] = row
        _log(f"M={M:2d}: need(omega=0..128) = "
             + " ".join(str(row[f'{w:g}']['need']) for w in omegas)
             + f"   (bars {min(row[f'{w:g}']['bar'] for w in omegas):.1e}"
               f" .. {max(row[f'{w:g}']['bar'] for w in omegas):.1e})")
    out["table"] = tab
    fits = {}
    for M in Ms:
        x = np.array([w for w in omegas])
        y = np.array([tab[str(M)][f"{w:g}"]["need"] for w in omegas],
                     dtype=float)
        p = np.polyfit(x, y, 1)
        fits[str(M)] = dict(slope=float(p[0]), intercept=float(p[1]))
    out["fits"] = fits
    sl = [v["slope"] for v in fits.values()]
    ic = [v["intercept"] for v in fits.values()]
    pM = np.polyfit(np.array(Ms, dtype=float), np.array(ic), 1)
    out["slope_range"] = [float(min(sl)), float(max(sl))]
    out["slope_spread_pct"] = float(100 * (max(sl) - min(sl)) / np.mean(sl))
    out["intercept_vs_M"] = [float(pM[0]), float(pM[1])]
    _log(f"FITTED slope a = {min(sl):.4f} .. {max(sl):.4f} "
         f"(spread {out['slope_spread_pct']:.1f} % over M = 3..12); "
         f"intercept b = {pM[1]:.2f} + {pM[0]:.3f} M")
    _log(f"SHIPPED constants: omega coefficient {_ts._STAG_QUAD_OMEGA}, "
         f"M coefficient {_ts._STAG_QUAD_M}, constant {_ts._STAG_QUAD_CONST}"
         f"  -> envelope of the measurement: "
         f"{_ts._STAG_QUAD_OMEGA > max(sl)} / "
         f"{_ts._STAG_QUAD_M >= pM[0]} / {_ts._STAG_QUAD_CONST >= pM[1]}")
    RES["need"] = out


# =================================================================== rule
def sec_rule():
    need = RES.get("need", {}).get("table")
    if need is None:
        p = os.path.join(HERE, f"v5_d3_{TAG}.json")
        need = json.load(open(p))["need"]["table"]
    worst = None
    cells = 0
    bad = []
    for M, row in need.items():
        for w, rec in row.items():
            if rec["need"] is None:
                continue
            cells += 1
            got = _stag_quad_order(int(M), float(w))
            margin = got - rec["need"]
            if worst is None or margin < worst:
                worst = margin
            if margin < 0:
                bad.append((M, w, got, rec["need"]))
    _log(f"RULE vs the measured requirement: {cells} cells, WORST margin "
         f"{worst} nodes, violations {len(bad)}")
    RES["rule"] = dict(cells=cells, worst_margin=worst, violations=bad)


# ================================================================= kernel
def sec_kernel():
    """The projector on a 3-segment grid with ONE very long segment, before
    (the fixed 2M+8 rule) and after (the shipped per-segment rule)."""
    d = 1.0
    G = 2.0 * np.pi / d
    a0 = 0.37 * G
    out = {}
    for longest in (0.33, 0.62, 0.91, 0.96):
        rest = (1.0 - longest) / 2.0
        walls = np.array([0.0, rest * d, (rest + longest) * d, d])
        for M in (4, 6, 8):
            for mmax in (3, 7):
                b = Basis1D(d, walls, M)
                orders = np.arange(-mmax, mmax + 1)
                asm = _stag_fourier_projection(b, orders, a0)
                after = np.asarray(asm(b.B))
                # BEFORE: the historical single 2M+8 rule, reimplemented
                before = _fixed_rule_projection(b, orders, a0, 2 * M + 8)
                # reference: an 8x refined rule
                ref = _fixed_rule_projection(b, orders, a0, 8 * (2 * M + 8))
                sc = np.max(np.abs(ref))
                out[f"L{longest}_M{M}_m{mmax}"] = dict(
                    before=float(np.max(np.abs(before - ref)) / sc),
                    after=float(np.max(np.abs(after - ref)) / sc),
                    nq=[_stag_quad_order(M, float(np.max(np.abs(
                        orders * G + a0))) * b.Jn[s]) for s in range(b.N)])
        r3 = out[f"L{longest}_M4_m7"]
        _log(f"longest segment {longest:.2f} d, M=4, |m|<=7: BEFORE "
             f"{r3['before']:.2e}  AFTER {r3['after']:.2e}   nq {r3['nq']}")
    RES["kernel"] = out


def _fixed_rule_projection(basis, orders, a0, nq):
    """The pre-2026-09-11 projector: ONE Gauss rule of order ``nq`` for every
    segment.  Reimplemented so the BEFORE arm exists on this build."""
    d, N, M = basis.d, basis.N, basis.M
    G = 2.0 * np.pi / d
    xb = basis.xb
    xg, wg = leggauss(nq)
    Vref, _ = _modleg_value_deriv(M, xg)
    orders = np.asarray(orders)
    T_local = np.zeros((len(orders), N, M), dtype=complex)
    for seg in range(N):
        J = basis.Jn[seg]
        xphys = 0.5 * (xb[seg] + xb[seg + 1]) + J * xg
        phase = np.exp(1j * np.outer(orders * G + a0, xphys))
        T_local[:, seg, :] = (J / d) * (phase * wg) @ Vref.T
    S = np.array(basis.B)
    return np.einsum("msa,jsa->mj", T_local, S)


# ================================================================ integer
def sec_integer():
    """720 cells: M = 3..14 x N = 1..60, |alpha0| <= G/2."""
    n_cells = ok = 0
    viol = []
    for M in range(3, 15):
        for N in range(1, 61):
            n_cells += 1
            d = 1.0
            G = 2.0 * np.pi / d
            a0 = 0.5 * G           # the worst |alpha0| the contract allows
            mmax = int(_ts._STAG_ORDER_CAP) if hasattr(
                _ts, "_STAG_ORDER_CAP") else None
            # the far-field order cap the shipped path allows
            m_hi = (M - 1) * N // 2 if mmax is None else mmax
            kmax = abs(m_hi * G + a0)
            J = d / (2.0 * N)
            got = _stag_quad_order(M, kmax * J)
            if got == 2 * M + 8:
                ok += 1
            else:
                viol.append((M, N, got, 2 * M + 8))
    _log(f"INTEGER lattices: {ok}/{n_cells} return exactly 2M+8; "
         f"{len(viol)} violations"
         + (f"  first {viol[:3]}" if viol else ""))
    RES["integer"] = dict(cells=n_cells, ok=ok, violations=viol[:20],
                          n_violations=len(viol))


# ============================================================== candidate
def sec_candidate():
    def cand(M, omega):
        return max(2 * M + 8, int(np.ceil(0.75 * omega)) + M + 8)

    n_cells = ok = 0
    for M in range(3, 15):
        for N in range(1, 61):
            n_cells += 1
            d, G = 1.0, 2.0 * np.pi
            a0 = 0.5 * G
            m_hi = (M - 1) * N // 2
            kmax = abs(m_hi * G + a0)
            J = d / (2.0 * N)
            if cand(M, kmax * J) == 2 * M + 8:
                ok += 1
    _log(f"REJECTED candidate max(2M+8, ceil(0.75 w) + M + 8): {ok}/{n_cells}"
         f" return 2M+8, i.e. {n_cells - ok} cells MOVE")
    # and does it clear the requirement?
    need = RES.get("need", {}).get("table")
    if need is None:
        try:
            need = json.load(open(os.path.join(
                HERE, f"v5_d3_{TAG}.json")))["need"]["table"]
        except Exception:                                # noqa: BLE001
            need = None
    worst = None
    if need:
        for M, row in need.items():
            for w, rec in row.items():
                if rec["need"] is None:
                    continue
                m = cand(int(M), float(w)) - rec["need"]
                worst = m if worst is None else min(worst, m)
        _log(f"   the candidate's worst margin against the measured "
             f"requirement: {worst} nodes")
    RES["candidate"] = dict(cells=n_cells, ok=ok, moved=n_cells - ok,
                            worst_margin=worst)


# ================================================================== nqmap
def sec_nqmap():
    """How often does D3 change anything on an ORDINARY geometry?"""
    d = 1.0
    G = 2.0 * np.pi / d
    rows = {}
    geoms = {
        "conforming 0.2371/0.6183": [0.0, 0.2371, 0.6183, 1.0],
        "non-conforming 0.3117/0.7402": [0.0, 0.3117, 0.7402, 1.0],
        "duty-1/3": [0.0, 1 / 3, 2 / 3, 1.0],
        "nested 0.125/0.25/0.75/0.875": [0.0, 0.125, 0.25, 0.75, 0.875, 1.0],
        "taper slice 0.22/0.70": [0.0, 0.22, 0.70, 1.0],
        "one wall 0.4": [0.0, 0.4, 1.0],
        "long segment 0.80": [0.0, 0.10, 0.90, 1.0],
        "long segment 0.91": [0.0, 0.045, 0.955, 1.0],
        "long segment 0.96": [0.0, 0.02, 0.98, 1.0],
        "pillar edge 0.002": [0.0, 0.002, 0.998, 1.0],
    }
    for name, w in geoms.items():
        w = np.asarray(w) * d
        for M in (4, 6, 8):
            for mmax in (3, 7):
                a0 = 0.37 * G
                kmax = float(np.max(np.abs(np.arange(-mmax, mmax + 1) * G
                                           + a0)))
                Jn = 0.5 * np.diff(w)
                nq = [_stag_quad_order(M, kmax * J) for J in Jn]
                rows[f"{name}|M{M}|m{mmax}"] = dict(
                    nq=nq, base=2 * M + 8, moved=any(x != 2 * M + 8
                                                     for x in nq))
        r = rows[f"{name}|M4|m7"]
        _log(f"{name:32s} M=4 |m|<=7: nq {r['nq']}  base {r['base']}  "
             f"MOVED={r['moved']}")
    n_moved = sum(1 for v in rows.values() if v["moved"])
    _log(f"D3 changes the rule on {n_moved} of {len(rows)} (geometry, M, "
         f"order) cells")
    RES["nqmap"] = dict(rows=rows, n_moved=n_moved, n_cells=len(rows))


SECTIONS = {"need": sec_need, "rule": sec_rule, "kernel": sec_kernel,
            "integer": sec_integer, "candidate": sec_candidate,
            "nqmap": sec_nqmap}

if __name__ == "__main__":
    for s in (sys.argv[1:] or list(SECTIONS)):
        _log(f"=== section {s} ===")
        SECTIONS[s]()
    p = os.path.join(HERE, f"v5_d3_{TAG}.json")
    old = {}
    if os.path.exists(p):
        try:
            old = json.load(open(p))
        except Exception:                                # noqa: BLE001
            old = {}
    old.update(RES)
    old["_lumenairy"] = lumenairy.__file__
    with open(p, "w") as fh:
        json.dump(old, fh, indent=1, default=str)
    _log(f"wrote {p}")
