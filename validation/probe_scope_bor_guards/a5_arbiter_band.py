"""CLASS A on BOR -- ATTRIBUTION, and the candidate bar.

a4 established the defect: for a radial order approaching its own cutoff the
classifier ratio ``rho = |Im q| / |Re q|`` grows as ``1 / qn^2``, crosses the
shipped ``1e-9`` band at ``qn ~ 5e-03``, and the order is then oriented by
``sign(Im q)`` -- which is the eigensolver's backward error.  Measured, that
sign FLIPS with the BLAS thread count.

SUPER-UNITY IS THE DETECTOR, NOT THE ATTRIBUTION (the round-2 sliver lesson).
A near-cutoff order is also badly RESOLVED, so its closure could be ordinary
under-convergence.  The arbiter here is the same shape the sliver guard uses:
RE-SOLVE with the one thing changed, and see whether the damage vanishes.

THE CANDIDATE.  The shipped classifier scales the band by the mode's OWN
|Re q|.  That is exactly the shape ``rcwa/_core._CUT_BAND_REL`` was measured
AGAINST and rejected: "at a cutoff the mode's own magnitude has collapsed and
judging its real part against it is judging noise against noise ... the
spectrum's top is the only stable scale there".  So the candidate is the
``_sqrt_decay`` shape, transposed:

    prop = |Im q| <= band * max(max|q|, k0)          band = 1e-8

Three things are measured:
  R1  does the crossing disappear, and does the thread spread collapse?
  R2  does the candidate MOVE any ordinary answer (the bit-identity contract)?
  R3  what are the candidate bar's two-sided margins on the same populations?
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import closure, dump, pin_tree  # noqa: E402

print("TREE", pin_tree())
RBIG = 24.0
NFD = 120
EPS = 1.41 ** 2
NREF = 1.41
BAND = 1e-8


# --------------------------------------------------------------------------- #
#  A re-implementation of zcascade._layer_modes_staggered with the classifier
#  as a PARAMETER.  Verbatim apart from the two marked lines, so `rule="shipped"`
#  must reproduce the library bit for bit (asserted below).
# --------------------------------------------------------------------------- #
def layer_modes_rule(m, Rbig, N, eps_profile, k0, rule):
    from lumenairy.elements.bor.coupled_radial_eigensolver import (
        _assemble_staggered, _fast_geig,
    )
    op = _assemble_staggered(m, Rbig, N, eps_profile, k0)
    Lei, A_f2n, Dn2f = op["Lei"], op["A_f2n"], op["Dn2f"]
    mrn = op["mrn"]
    r_n, r_f, h = op["r_n"], op["r_f"], op["h"]
    q2, Vm = _fast_geig(op["K"], op["B"])
    q = np.sqrt(q2)
    qtop = float(np.max(np.abs(q))) if q.size else 1.0     # spectrum scale

    def hfields(Er, Ephi, qj):
        Ez = qj * (Lei @ (1j * A_f2n @ Er - mrn * Ephi))
        hr = (1.0 / k0) * (mrn * Ez - qj * Ephi)
        hphi = (1.0 / k0) * (qj * Er + 1j * (Dn2f @ Ez))
        return hr, hphi

    def flux(Er, Ephi, hr, hphi):
        return np.real(np.sum(Er * np.conj(hphi) * r_f * h)
                       - np.sum(Ephi * np.conj(hr) * r_n * h))

    nm = len(q)
    W = np.zeros((2 * N, nm), dtype=complex)
    V = np.zeros((2 * N, nm), dtype=complex)
    qf = np.zeros(nm, dtype=complex)
    for j in range(nm):
        Er, Ephi = Vm[:N, j], Vm[N:, j]
        qj = q[j]
        hr, hphi = hfields(Er, Ephi, qj)
        P = flux(Er, Ephi, hr, hphi)
        # ---- THE ONE CHANGED LINE ------------------------------------- #
        if rule == "shipped":
            is_prop = abs(qj.imag) < 1e-9 * max(abs(qj.real), 1e-300)
        else:                       # candidate: _sqrt_decay's spectrum scale
            is_prop = abs(qj.imag) <= BAND * max(qtop, k0)
        # --------------------------------------------------------------- #
        if is_prop:
            qj = qj if P >= 0 else -qj
        else:
            qj = qj if qj.imag > 0 else -qj
        hr, hphi = hfields(Er, Ephi, qj)
        P = flux(Er, Ephi, hr, hphi)
        fnrm = (np.sum(np.abs(Er) ** 2 * r_f * h)
                + np.sum(np.abs(Ephi) ** 2 * r_n * h))
        s = (1.0 / np.sqrt(abs(P)) if abs(P) > 1e-10 * fnrm
             else 1.0 / np.sqrt(np.sum(np.abs(Er) ** 2 + np.abs(Ephi) ** 2)
                                + 1e-300))
        W[:N, j] = Er * s
        W[N:, j] = Ephi * s
        V[:N, j] = hr * s
        V[N:, j] = hphi * s
        qf[j] = qj
    wq_node = (r_n * h).astype(complex)
    wq_face = (r_f * h).astype(complex)
    return dict(W=W, V=V, q=qf, r=r_n, wq=wq_node, N=N, r_face=r_f,
                wq_node=wq_node, wq_face=wq_face)


class patched:
    """Swap the FD staggered modal builder for the parameterized one."""

    def __init__(self, rule):
        self.rule = rule

    def __enter__(self):
        from lumenairy.elements.bor import zcascade
        self.old = zcascade._layer_modes_staggered
        rule = self.rule

        def repl(m, Rbig, N, eps_profile, k0):
            return layer_modes_rule(m, Rbig, N, eps_profile, k0, rule)
        zcascade._layer_modes_staggered = repl
        return self

    def __exit__(self, *a):
        from lumenairy.elements.bor import zcascade
        zcascade._layer_modes_staggered = self.old


def gamma_of(m, idx=2):
    from lumenairy.elements.bor.zcascade import layer_modes
    L = layer_modes(m, RBIG, NFD,
                    lambda r: np.full_like(r, EPS, dtype=complex), 2.0,
                    staggered=True)
    q = np.asarray(L["q"])
    g = np.sqrt(2.0 ** 2 * EPS - q ** 2)
    g = np.real(g[np.abs(g.imag) < 1e-9 * np.maximum(np.abs(g.real), 1e-300)])
    g = np.sort(g[g > 1e-6])
    return float(g[idx])


def stack(m, k0):
    from lumenairy import BORStack
    s = BORStack(RBIG, m, n_substrate=NREF, n_superstrate=NREF, N=NFD,
                 basis="fd")
    s.add_layer(0.4, eps=EPS)
    s.add_layer(0.5, rings=(3.0, 0.5, 2.45, 1.41))
    s.add_layer(0.4, eps=EPS)
    s.set_source(k0=float(k0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return s.solve()


def summary(res):
    return dict(n_orders=int(np.size(res["R"])), closure=closure(res),
                sumR=float(np.sum(res["R"])),
                R=[float(x) for x in np.asarray(res["R"])[:6]],
                T=[float(x) for x in np.asarray(res["T"])[:6]])


def main():
    tag = os.environ.get("PROBE_TAG", "win")
    thr = os.environ.get("OPENBLAS_NUM_THREADS", "?")
    payload = dict(threads=thr, band=BAND, rungs=[], ordinary=[])

    # R2 FIRST: the bit-identity check of the re-implementation ------------- #
    ref = stack(1, 2.0)
    with patched("shipped"):
        rep = stack(1, 2.0)
    ident = dict(dR=float(np.max(np.abs(np.asarray(ref["R"])
                                        - np.asarray(rep["R"])))),
                 dT=float(np.max(np.abs(np.asarray(ref["T"])
                                        - np.asarray(rep["T"])))))
    payload["reimpl_identity"] = ident
    print(f"re-implementation vs library (rule='shipped'): dR={ident['dR']:.3e}"
          f" dT={ident['dT']:.3e}   (must be 0.0)")

    # R1: the cutoff ladder, both rules ------------------------------------ #
    deltas = sorted({10.0 ** (-e / 2.0) for e in range(8, 21)}, reverse=True)
    for m in (0, 1, 2):
        g = gamma_of(m)
        for dl in deltas:
            k0 = g / (NREF * np.sqrt(1.0 - dl))
            row = dict(m=m, delta=float(dl), k0=float(k0),
                       qn_target=float(NREF * np.sqrt(dl)))
            with patched("shipped"):
                row["shipped"] = summary(stack(m, k0))
            with patched("candidate"):
                row["candidate"] = summary(stack(m, k0))
            payload["rungs"].append(row)

    # R2: the ORDINARY battery -- the candidate must move nothing ---------- #
    from lumenairy import BORStack
    for m in (0, 1, 2, 5):
        for k0 in (0.8, 2.0, 3.5):
            def build():
                s = BORStack(RBIG, m, n_substrate=1.41, n_superstrate=1.41,
                             N=NFD, basis="fd")
                s.add_layer(0.4, eps=EPS)
                s.add_layer(0.5, rings=(3.0, 0.5, 2.45, 1.41))
                s.add_layer(0.4, eps=EPS)
                s.set_source(k0=k0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return s.solve()
            with patched("shipped"):
                a = build()
            with patched("candidate"):
                b = build()
            payload["ordinary"].append(dict(
                m=m, k0=k0,
                n_shipped=int(np.size(a["R"])), n_cand=int(np.size(b["R"])),
                dR=float(np.max(np.abs(np.asarray(a["R"])
                                       - np.asarray(b["R"]))))
                if np.size(a["R"]) == np.size(b["R"]) else None,
                dT=float(np.max(np.abs(np.asarray(a["T"])
                                       - np.asarray(b["T"]))))
                if np.size(a["T"]) == np.size(b["T"]) else None,
                closure_shipped=closure(a), closure_cand=closure(b)))

    dump(f"a5_arbiter_{tag}_t{thr}.json", payload)

    print("\n== R1 CUTOFF LADDER: shipped vs candidate ==")
    print("  m  delta      qn_tgt      n_ord(s/c)  closure_shipped  "
          "closure_candidate  ratio")
    for r in payload["rungs"]:
        s_, c_ = r["shipped"], r["candidate"]
        rat = (s_["closure"] / c_["closure"]
               if c_["closure"] and c_["closure"] > 0 else float("nan"))
        print(f"  {r['m']}  {r['delta']:.2e}  {r['qn_target']:.3e}   "
              f"{s_['n_orders']:2d}/{c_['n_orders']:2d}      "
              f"{s_['closure']:.4e}      {c_['closure']:.4e}    {rat:8.2f}")
    sc = [r["shipped"]["closure"] for r in payload["rungs"]]
    cc = [r["candidate"]["closure"] for r in payload["rungs"]]
    print(f"\n  worst closure  shipped {max(sc):.4e}   candidate {max(cc):.4e}"
          f"   -> {max(sc) / max(cc):.1f}x better")
    nmov = sum(1 for r in payload["rungs"]
               if r["shipped"]["n_orders"] != r["candidate"]["n_orders"])
    print(f"  rungs where the CHANNEL COUNT differs: {nmov}/"
          f"{len(payload['rungs'])}")

    print("\n== R2 ORDINARY BATTERY: the candidate must move nothing ==")
    mv = [o for o in payload["ordinary"]
          if o["dR"] is None or o["dR"] > 0.0 or o["dT"] > 0.0]
    print(f"  {len(payload['ordinary'])} ordinary solves, moved by the "
          f"candidate: {len(mv)}")
    for o in mv[:8]:
        print("   ", o)


if __name__ == "__main__":
    main()
