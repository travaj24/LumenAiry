"""P1 -- the spectrum of ONE frozen tapered slab, conservative vs M5 form.

The M5 spike (``docs/audits/PMM_M5_2D_FEASIBILITY_2026_08_04.md`` S3.6)
found that its frozen taper pencil has a COMPLEX fundamental on a lossless
cell and the symmetry ``q -> -conj(q)`` instead of ``q -> -q`` and
``q -> conj(q)``, so no forward/backward selector could classify the modes.
Its pencil kept, inside the slab, the term ``d(Q_ww)/dw d_w E =
S'(u) d_w E`` (the w-derivative of the frozen metric).  The CONSERVATIVE
frozen slab (the virtual medium at one height, nothing else) has the pencil
``A0 + beta i(C1 - C2) - beta^2 Mww`` with ``i(C1 - C2)`` Hermitian, so its
spectrum must be closed under ``beta -> conj(beta)`` (flux conservation)
and ``beta -> -beta`` (reciprocity).

Measured on the P2 strong taper (sidewall 14.9 deg) frozen at w = 0.1 h,
0.5 h and 0.9 h, TE and TM, degree 16: the largest |Im beta| over the
propagating set, and the two pairing residuals.

Run:  PYTHONPATH=<worktree> OMP_NUM_THREADS=2 python p1_spectrum.py
Writes p1_spectrum.json.
"""
from __future__ import annotations

import os

import _zcommon as zc
import numpy as np
import scipy.linalg as sla
from p2_taper_ladder import DEG, EPS_G, EPS_R, WL, C, H, P, xw_factory

HERE = os.path.dirname(os.path.abspath(__file__))


def spectrum(ops, n):
    I = np.eye(n, dtype=complex)
    Z = np.zeros((n, n), dtype=complex)
    beta = sla.eig(np.block([[Z, I], [ops["A0"], ops["A1"]]]),
                   np.block([[I, Z], [Z, ops["A2"]]]), right=False)
    return beta[np.isfinite(beta)]


def pair_res(beta, f):
    other = f(beta)
    return float(np.max([np.min(np.abs(beta - o)) for o in other])
                 / max(np.max(np.abs(beta)), 1.0))


def main():
    info = zc.assert_tree()
    dt, db = 0.30, 0.70
    dm = 0.5 * (dt + db) * P
    uw = [0.0, C - 0.5 * dm, C + 0.5 * dm, P]
    k0 = 2 * np.pi / WL
    out = dict(info=info)
    for pol in ("te", "tm"):
        for zf in (0.1, 0.5, 0.9):
            mesh = zc.Mesh1D(P, uw, [EPS_G, EPS_R, EPS_G], DEG)
            tmap = zc.TaperMap(uw, xw_factory(dt, db), H)
            row = {}
            for form in ("conservative", "m5"):
                ops = zc.region_ops(mesh, tmap, zf * H, pol, k0,
                                    m5=(form == "m5"))
                b = spectrum(ops, mesh.n)
                q = b / k0
                # the fundamental: largest real part
                i0 = int(np.argmax(q.real))
                row[form] = dict(
                    fundamental=[float(q[i0].real), float(q[i0].imag)],
                    min_abs_im_over_top4=float(np.min(np.abs(
                        q[np.argsort(-q.real)[:4]].imag))),
                    res_minus_q=pair_res(q, lambda x: -x),
                    res_conj_q=pair_res(q, np.conj),
                    res_minus_conj_q=pair_res(q, lambda x: -np.conj(x)))
                print(f"{pol} w={zf}h {form:12s}: q0 = {q[i0]:.10f}  "
                      f"|q+(-q)| {row[form]['res_minus_q']:.1e}  "
                      f"|q-conj q| {row[form]['res_conj_q']:.1e}  "
                      f"|q+conj q| {row[form]['res_minus_conj_q']:.1e}")
            out[f"{pol}_w{zf}"] = row
    zc.dump(os.path.join(HERE, "p1_spectrum.json"), out)


if __name__ == "__main__":
    main()
