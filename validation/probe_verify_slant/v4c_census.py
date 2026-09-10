"""V4c -- THE MODE CENSUS on a slanted region: the forward/backward split, the
spectral radius, and whether anything in the forward set GROWS.

The library's own split lives inside ``_region_modes_oop``, and
``_select_forward_flux`` REBALANCES to exactly ``2 q^2`` unconditionally -- so
reading its output can never fail.  This probe therefore classifies the modes
ITSELF, from the eigenvectors, by the physical rule (net Poynting z-flux summed
over all harmonics; decay sign for the flux-null ones) and reports the count
BEFORE any rebalance.  That is the measurement the shipped ``RuntimeError``
cannot make.

Also measured, per row: ``min Re(lam_forward)`` (a lossless region's forward set
must not contain a grower; a lossy one's must be strictly positive), the
spectral radius ``max |q|`` and how it grows with the slant (the 1-D convection
form's runaway is the thing being excluded), and the worst forward propagator
magnitude over a thick layer.

Cells include a LOSSY dielectric and a METAL (``eps = -20 + 1.5i``), because a
metal's modes are exactly the ones a flux-sign rule can misclassify.
"""
import numpy as np
import scipy.linalg as sla

from _lib import arm, dump  # noqa: I001

from lumenairy.elements.pmm.twod_staggered import Granet2DTransverseE
from lumenairy.elements.rcwa._core import uniaxial_tensor

WL = 1.0
K0 = 2.0 * np.pi
PX = PY = 1.2
NG = 2
M = 6
AX0 = float(np.sin(np.deg2rad(20.0)) * np.cos(np.deg2rad(35.0)))
AY0 = float(np.sin(np.deg2rad(20.0)) * np.sin(np.deg2rad(35.0)))

TIL = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0), phi=np.deg2rad(25.0))
AIR = np.eye(3, dtype=complex)


def scalar_cell(e):
    c = np.ones((NG, NG), dtype=complex)
    c[0, 0] = e
    return c


def tensor_cell(t):
    c = np.zeros((NG, NG, 3, 3), dtype=complex)
    c[:, :] = AIR
    c[0, 0] = t
    return c


T20 = float(np.tan(np.deg2rad(20.0)))
T45 = 1.0
T60 = float(np.tan(np.deg2rad(60.0)))
SLANTS = {"vertical": None, "x20": (T20, 0.0), "x45": (T45, 0.0),
          "diag45": (T45 / np.sqrt(2), T45 / np.sqrt(2)), "x60": (T60, 0.0)}
CELLS = {
    "scalar4": (scalar_cell(4.0), True),
    "oop_uniaxial": (tensor_cell(TIL), True),
    "high_contrast12": (scalar_cell(12.0), True),
    "lossy4_06i": (scalar_cell(4.0 + 0.6j), False),
    "metal": (scalar_cell(-20.0 + 1.5j), False),
}


def census(sol):
    """Classify the 4q^2 modes by the PHYSICAL flux rule, before any
    rebalance, from the whitened eigenvectors (the construction
    ``_region_modes_oop`` uses)."""
    A, B = sol.Agen, sol.Bgen
    qq = sol.q * sol.q
    Lc = np.linalg.cholesky(B)
    Ah = sla.solve_triangular(Lc, A, lower=True)
    Ah = sla.solve_triangular(Lc, Ah.conj().T, lower=True).conj().T
    qv, Y = np.linalg.eig(Ah)
    X = sla.solve_triangular(Lc.conj().T, Y, lower=False)
    W = X[:2 * qq, :]
    Gst = X[2 * qq:, :]
    L1 = np.linalg.cholesky(B[:qq, :qq]).conj().T
    L2 = np.linalg.cholesky(B[qq:2 * qq, qq:2 * qq]).conj().T
    V = np.concatenate([L1 @ W[:qq], L2 @ W[qq:], L2 @ Gst[:qq],
                        L1 @ Gst[qq:]], axis=0)
    nrm = np.linalg.norm(V, axis=0)
    V = V / np.where(nrm == 0.0, 1.0, nrm)[None, :]
    N = qq
    Ex, Ey = V[:N], V[N:2 * N]
    Hx, Hy = V[2 * N:3 * N] / 1j, V[3 * N:4 * N] / 1j
    Sz = np.real(np.sum(Ex * np.conj(Hy) - Ey * np.conj(Hx), axis=0))
    lam = -1j * qv
    gre = np.real(lam)
    mx = max(1.0, float(np.max(np.abs(Sz))))
    carries = np.abs(Sz) > 1e-9 * mx
    carries &= ~((np.abs(Sz) < 3e-3 * mx) & (np.abs(gre) > 0.1))
    carries &= ~(np.abs(gre) > 0.5)
    fwd = np.where(carries, Sz > 0, gre > 0)
    nf = int(np.sum(fwd))
    lam_f = lam[fwd]
    return dict(dim=int(4 * qq), want=int(2 * qq), n_forward=nf,
                n_backward=int(4 * qq - nf),
                split_exact=bool(nf == 2 * qq),
                min_re_lam_f=float(np.min(np.real(lam_f))) if nf else None,
                max_abs_q=float(np.max(np.abs(qv))),
                max_fwd_growth_1lam=float(np.max(np.abs(
                    np.exp(-lam_f * K0 * 1.0)))) if nf else None,
                max_fwd_growth_3lam=float(np.max(np.abs(
                    np.exp(-lam_f * K0 * 3.0)))) if nf else None)


def main():
    out = {"rows": {}, "config": dict(px=PX, M=M, grid=NG, ax0=AX0, ay0=AY0)}
    for cname, (cell, lossless) in CELLS.items():
        for sname, sl in SLANTS.items():
            if sl is None and cell.ndim == 2:
                continue          # a vertical scalar cell has no 4q^2 pencil
            sol = Granet2DTransverseE(PX, PY, NG, NG, M, cell,
                                      alpha0x=AX0 * K0, alpha0y=AY0 * K0,
                                      k0=K0, slant=sl)
            if not sol.offplane:
                continue
            row = census(sol)
            row["lossless"] = lossless
            out["rows"][f"{cname}/{sname}"] = row
            print(f"{cname:16s}/{sname:8s} split {row['n_forward']}/"
                  f"{row['n_backward']} (want {row['want']}) "
                  f"{'EXACT' if row['split_exact'] else '*** OFF ***'}  "
                  f"minRe(lam_f) {row['min_re_lam_f']:+.2e}  max|q| "
                  f"{row['max_abs_q']:.3f}  growth1 "
                  f"{row['max_fwd_growth_1lam']:.4e}  growth3 "
                  f"{row['max_fwd_growth_3lam']:.4e}")
    # growth of the spectral radius with slant, on one cell
    base = out["rows"].get("oop_uniaxial/vertical")
    if base:
        for sname in SLANTS:
            k = f"oop_uniaxial/{sname}"
            if k in out["rows"]:
                out["rows"][k]["radius_ratio_vs_vertical"] = (
                    out["rows"][k]["max_abs_q"] / base["max_abs_q"])
        print("radius ratio vs vertical: " + "  ".join(
            f"{s} {out['rows'][f'oop_uniaxial/{s}']['radius_ratio_vs_vertical']:.3f}"
            for s in SLANTS if f"oop_uniaxial/{s}" in out["rows"]))
    dump("v4c_census", out)
    print("arm", arm())


if __name__ == "__main__":
    main()
