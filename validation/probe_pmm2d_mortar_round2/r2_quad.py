"""R2 -- D3: the far-field projector's per-segment quadrature order.

``python r2_quad.py [need rule ladder]``

``_stag_fourier_projection`` integrates ``Ltilde_a(u) e^{i (mG + alpha0) x}``
with a FIXED ``nq = 2M + 8`` Gauss rule per segment.  On the uniform lattice a
segment is ``d/N`` long, so the HALF-PHASE it carries,

    omega_n = |m G + alpha0| * J_n          (J_n = (x_{n+1} - x_n) / 2)

is bounded by the order cap; with arbitrary walls a single segment can be
almost the whole period and ``omega`` is unbounded.

``need``  MEASURES the smallest ``nq`` that reaches machine precision, over a
          grid of ``(omega, M)``, so the rule is FITTED and not assumed.
``rule``  scores a candidate rule: does it reproduce ``2M + 8`` EXACTLY on
          every integer-N grid the order cap allows, and does it clear the
          measured requirement everywhere?
``ladder`` the shipped-vs-refined kernel error before and after.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
import sys
import time

import numpy as np
from numpy.polynomial.legendre import leggauss

import lumenairy
from lumenairy.elements.pmm.twod_staggered import (
    Basis1D,
    _modleg_value_deriv,
    _stag_fourier_projection,
)

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT), lumenairy.__file__
print(f"[arm] lumenairy = {lumenairy.__file__}", flush=True)

_C = complex
RES = {}
T0 = time.time()


def _log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


def _kernel(M, omega, nq):
    """``INT_{-1}^{1} Ltilde_a(u) e^{i omega u} du`` for a = 0..M-1, by an
    ``nq``-point Gauss rule.  This IS the segment integral of
    ``_stag_fourier_projection`` up to the constant ``(J/d) e^{i k x_c}``."""
    xg, wg = leggauss(nq)
    V, _ = _modleg_value_deriv(M, xg)
    return (np.exp(1j * omega * xg) * wg) @ V.T


#: The kernel error that matters is ABSOLUTE against the projector's own O(1)
#: scale, not relative to an integral that decays like a Bessel function: at
#: ``omega = 96`` the true value is ~1e-9, and asking for 14 relative digits
#: THERE is asking for 23 absolute ones, which no rule delivers and no caller
#: needs.  The scale is therefore fixed at the ``omega = 0`` magnitude.
_ABS_BAR = 1e-15

_OMEGAS = (0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0,
           96.0, 128.0)


def sec_need():
    """The smallest nq whose kernel matches a refined rule to _ABS_BAR of the
    projector's own omega=0 scale."""
    out = {}
    for M in (3, 4, 5, 6, 7, 8, 10, 12):
        scale0 = float(np.max(np.abs(_kernel(M, 0.0, 4 * M + 40)))) or 1.0
        row = {}
        for omega in _OMEGAS:
            nref = max(240, int(3 * omega) + 4 * M + 80)
            ref = _kernel(M, omega, nref)
            ref2 = _kernel(M, omega, nref + 37)
            drift = float(np.max(np.abs(ref - ref2))) / scale0
            # BAR derived from the ORACLE's own floor: the reference rule
            # and a 37-node-finer one agree only to ``drift``, so nothing
            # below ~that is a statement about quadrature.
            bar = max(_ABS_BAR, 20.0 * drift)
            need = None
            for nq in range(2, 240):
                e = float(np.max(np.abs(_kernel(M, omega, nq) - ref))) / scale0
                if e < bar:
                    need = nq
                    break
            row[f"{omega:g}"] = {"need": need, "ref_drift": drift, "bar": bar}
        out[str(M)] = row
        _log("M=%2d  need(nq): " % M
             + " ".join(f"{k}:{v['need']}" for k, v in row.items())
             + "   worst ref drift "
             + f"{max(v['ref_drift'] for v in row.values()):.1e}")
    fits = {}
    for M, row in out.items():
        ok = [(k, v) for k, v in row.items() if v["need"] is not None]
        om = np.array([float(k) for k, _v in ok])
        nq = np.array([float(v["need"]) for _k, v in ok])
        A = np.vstack([om, np.ones_like(om)]).T
        sl, ic = np.linalg.lstsq(A, nq, rcond=None)[0]
        fits[M] = {"slope": float(sl), "intercept": float(ic),
                   "max_over_fit": float(np.max(nq - (sl * om + ic))),
                   "need_at_0": float(row["0"]["need"])}
    RES["need"] = {"grid": out, "fits": fits, "abs_bar": _ABS_BAR}
    _log("fits: " + ", ".join(f"M{M}: {f['slope']:.3f}*om+{f['intercept']:.2f}"
                              for M, f in fits.items()))


# --------------------------------------------------------------- the rule
def _nq_rule(M, omega):
    """CANDIDATE.  ``2M + 8`` unless the segment's own half-phase needs more.

    The oscillatory factor ``e^{i omega u}`` costs a Gauss rule about
    ``omega / 2`` nodes beyond what the polynomial factor alone needs (the
    measured slope, ``sec_need``); the shipped ``2M + 8`` already carries the
    polynomial factor and a large constant reserve, so the rule only has to
    top it up once ``omega`` outruns that reserve."""
    return max(2 * M + 8, int(math.ceil(0.75 * omega)) + M + 8)


def sec_rule():
    """Two-sided: (a) the rule clears the MEASURED requirement; (b) it returns
    EXACTLY 2M+8 on every integer-N grid the shipped order cap allows."""
    out = {}
    need = RES.get("need", {}).get("grid")
    if need is None:
        sec_need()
        need = RES["need"]["grid"]
    # (a) clears the requirement, with margin
    marg = {}
    for M, row in need.items():
        for k, v in row.items():
            r = _nq_rule(int(M), float(k))
            if v["need"] is None:
                continue
            marg[f"M{M}_om{k}"] = {"need": v["need"], "rule": r,
                                   "margin": r - v["need"]}
    worst = min(v["margin"] for v in marg.values())
    _log(f"(a) rule vs measured need: worst margin {worst} nodes "
         f"(over {len(marg)} cells)")
    out["margin"] = marg
    out["worst_margin_nodes"] = worst
    # (b) integer-N bit-identity: on a uniform grid the largest |m| the shipped
    #     cap allows is (N(M-1) - 1)//2, and alpha0 lies in the first BZ so
    #     |alpha0| <= G/2.
    ident = {}
    bad = []
    for M in range(3, 15):
        for N in range(1, 61):
            q = N * (M - 1)
            mmax = (q - 1) // 2
            # omega = |m G + alpha0| * J,  J = d/(2N),  G = 2 pi / d
            om = (mmax + 0.5) * 2.0 * np.pi * (1.0 / (2.0 * N))
            r = _nq_rule(M, om)
            ident[f"M{M}_N{N}"] = {"omega_max": float(om), "nq": r,
                                   "shipped": 2 * M + 8}
            if r != 2 * M + 8:
                bad.append((M, N, om, r))
    out["uniform_identity"] = {"cells": len(ident), "violations": len(bad),
                               "worst": bad[:10]}
    _log(f"(b) integer-N grids M=3..14 x N=1..60: {len(ident)} cells, "
         f"{len(bad)} where the rule != 2M+8"
         + (f"   WORST {bad[:3]}" if bad else ""))
    RES["rule"] = out


# ------------------------------------------------------------- the ladder
def _proj_ref(basis, orders, alpha0, mult):
    """``_stag_fourier_projection``'s T_local with the rule multiplied."""
    d, N, M = basis.d, basis.N, basis.M
    G = 2.0 * np.pi / d
    xb = basis.xb
    nq = int(mult * (2 * M + 8))
    xg, wg = leggauss(nq)
    Vref, _ = _modleg_value_deriv(M, xg)
    orders = np.asarray(orders)
    T = np.zeros((len(orders), N, M), dtype=_C)
    for seg in range(N):
        J = basis.Jn[seg]
        xphys = 0.5 * (xb[seg] + xb[seg + 1]) + J * xg
        phase = np.exp(1j * np.outer(orders * G + alpha0, xphys))
        T[:, seg, :] = (J / d) * (phase * wg) @ Vref.T
    return T


def _proj_shipped(basis, orders, alpha0):
    asm = _stag_fourier_projection(basis, orders, alpha0)
    return asm(basis.B), asm(basis.Btilde)


def sec_ladder():
    """Kernel error of the SHIPPED projector against an 8x-refined rule, over
    the (longest segment, M, m_max) grid the verifier used."""
    d = 1.0
    out = {}
    for frac in (0.3333333333333333, 0.5, 0.62, 0.8, 0.91, 0.96):
        for M in (4, 6, 8):
            for mmax in (3, 7):
                if abs(frac - 1.0 / 3.0) < 1e-12:
                    xb = np.array([0.0, d / 3.0, 2.0 * d / 3.0, d])
                else:
                    rest = (1.0 - frac) / 2.0
                    xb = np.array([0.0, rest * d, (rest + frac) * d, d])
                b = Basis1D(d, xb, M, tau=np.exp(-0.41j))
                orders = np.arange(-mmax, mmax + 1)
                a0 = 0.37 * 2 * np.pi / d
                B1, B2 = _proj_shipped(b, orders, a0)
                Tref = _proj_ref(b, orders, a0, 8)
                S1 = np.einsum("msa,jsa->mj", Tref, np.array(b.B))
                S2 = np.einsum("msa,jsa->mj", Tref, np.array(b.Btilde))
                sc = max(float(np.max(np.abs(S1))), float(np.max(np.abs(S2))))
                e = max(float(np.max(np.abs(B1 - S1))),
                        float(np.max(np.abs(B2 - S2)))) / sc
                out[f"f{frac:.2f}_M{M}_m{mmax}"] = {
                    "err_rel": e,
                    "omega_max": float((mmax + 0.37) * 2 * np.pi
                                       * 0.5 * frac * d / d),
                    "nq_shipped": 2 * M + 8,
                    "nq_rule": _nq_rule(
                        M, (mmax + 0.37) * 2 * np.pi * 0.5 * frac)}
        _log(f"frac={frac:.2f}: "
             + "  ".join(f"M{M}m{m}={out[f'f{frac:.2f}_M{M}_m{m}']['err_rel']:.1e}"
                         f"(nq {out[f'f{frac:.2f}_M{M}_m{m}']['nq_shipped']}"
                         f"->{out[f'f{frac:.2f}_M{M}_m{m}']['nq_rule']})"
                         for M in (4, 6, 8) for m in (3, 7)))
    RES["ladder"] = out


SECTIONS = {"need": sec_need, "rule": sec_rule, "ladder": sec_ladder}


def main():
    for w in (sys.argv[1:] or list(SECTIONS)):
        SECTIONS[w]()
    tag = os.environ.get("R_TAG", "")
    path = os.path.join(HERE, f"r2_quad{('_' + tag) if tag else ''}.json")
    with open(path, "w") as fh:
        json.dump(RES, fh, indent=1, sort_keys=True, default=float)
    _log(f"wrote {path}")


if __name__ == "__main__":
    main()
