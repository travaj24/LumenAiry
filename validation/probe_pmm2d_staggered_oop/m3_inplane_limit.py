"""M3 -- IN-PLANE LIMIT: do the OOP candidates REDUCE to the shipped E-form?

With the cross terms set to zero, each candidate must be compared against the
shipped in-plane discretization on the SAME cell.  Two questions, measured
separately:

  T1  spectra: distance between the candidate's eigenvalue set and the E-form's
      ``{+q, -q}`` set (the E-form solves for ``q^2``);
  T2  observables: R / T / Jones of a full slab solve through the candidate vs
      through the shipped ``pmm_efficiency_2d_staggered`` (scalar) and vs the
      probe's own E-form path -- an EQUIVALENT discretization must agree to the
      eigensolver's noise, not merely converge to the same answer;
  T3  the ALGEBRAIC reduction of candidate (d): its ``P2/P1/P0`` blocks with the
      OOP entries zeroed must rebuild the shipped ``Lmat``/``Rmat`` EXACTLY
      (this is a derivation check, not a convergence claim).

Run:
  cd /c/tmp/lum_aniso_oop && PYTHONPATH=/c/tmp/lum_aniso_oop \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python validation/probe_pmm2d_staggered_oop/m3_inplane_limit.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import probe_common as pc  # noqa: E402

from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE,
    pmm_efficiency_2d_staggered,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

WL = 1.0
PX = PY = 0.9
K0 = 2.0 * np.pi / WL
DEPTH = 0.35


def cells(kind, Nx=2):
    """In-plane test cells (no OOP entries anywhere)."""
    if kind == "isotropic":
        e = np.zeros((Nx, Nx, 3, 3), dtype=complex)
        sc = np.array([[2.25, 1.0], [1.0, 4.0 + 0.2j]])
        for i in range(Nx):
            for j in range(Nx):
                e[i, j] = sc[i % 2, j % 2] * np.eye(3)
        return e, sc
    if kind == "inplane_uniaxial":
        t = pc.uniaxial(1.5, 1.7, 90.0, azim_deg=30.0)     # axis in the plane
        e = np.zeros((Nx, Nx, 3, 3), dtype=complex)
        for i in range(Nx):
            for j in range(Nx):
                e[i, j] = t if (i + j) % 2 == 0 else 2.25 * np.eye(3)
        return e, None
    if kind == "gyrotropic":
        t = np.array([[2.25, 0.5j, 0], [-0.5j, 2.25, 0], [0, 0, 2.0]],
                     dtype=complex)
        e = np.zeros((Nx, Nx, 3, 3), dtype=complex)
        for i in range(Nx):
            for j in range(Nx):
                e[i, j] = t if (i + j) % 2 == 0 else np.eye(3)
        return e, None
    raise ValueError(kind)


def main():
    pc.banner("M3 -- in-plane limit")
    R = {}
    print("\n## T1  spectra: candidate vs the probe's E-form (2 q^2 -> +/-q)")
    print("   cell               M  cand  dim   max_over_cand min_over_eform")
    for kind in ("isotropic", "inplane_uniaxial", "gyrotropic"):
        ec, _ = cells(kind)
        for M in (5, 6, 7):
            for th, ph in ((0.0, 0.0), (np.deg2rad(25.0), np.deg2rad(40.0))):
                kx0, ky0 = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph)
                cell = pc.StaggeredCell(PX, PY, ec, M, K0, kx0, ky0)
                We, Ve, qe, _ = pc.eform_modes(cell)
                qref = np.concatenate([qe, -qe])
                for cand in ("a", "d"):
                    if cand == "a":
                        _, _, qv, _ = pc.modes_a(cell)
                    else:
                        _, _, qv, _, _ = pc.modes_d(cell)
                        keep = np.abs(qv) > 1e-6 * max(
                            1.0, float(np.max(np.abs(qv))))
                        qv = qv[keep]
                    d = float(np.max([np.min(np.abs(qref - z)) for z in qv]))
                    tag = f"{kind}|M{M}|{'nrm' if th == 0 else 'obl'}|{cand}"
                    print(f"   {kind:18s} {M:2d}  ({cand})  {qv.size:5d} "
                          f"{'normal' if th == 0 else 'oblique'}   {d:.3e}")
                    R[f"T1_{tag}"] = dict(dim=int(qv.size), maxdist=d)

    print("\n## T2  observables: candidate vs E-form vs SHIPPED scalar solver")
    ec, sc = cells("isotropic")
    for th, ph in ((0.0, 0.0), (np.deg2rad(25.0), np.deg2rad(40.0))):
        for M in (6, 7):
            out = {}
            for cand in ("a", "d", "eform"):
                _o, Rm, Tm, J = pc.solve_slab(PX, PY, ec, 1.5, 1.0, DEPTH, WL,
                                              M=M, theta=th, phi=ph,
                                              candidate=cand)
                out[cand] = (Rm.sum(axis=1), Tm.sum(axis=1), J)
            # shipped scalar solver (te / tm rows are a different basis than
            # Ex / Ey, so compare the TOTALS only)
            ship = {}
            for pol in ("te", "tm"):
                r = pmm_efficiency_2d_staggered(
                    PX, PY, sc, 1.5, 1.0, DEPTH, WL, degree=M, n_orders=8,
                    polarization=pol, theta=th, phi=ph)
                ship[pol] = (float(np.sum(r[1])), float(np.sum(r[2])))
            dRa = float(np.max(np.abs(out["a"][0] - out["eform"][0])))
            dRd = float(np.max(np.abs(out["d"][0] - out["eform"][0])))
            dJa = float(np.max(np.abs(out["a"][2] - out["eform"][2])))
            dJd = float(np.max(np.abs(out["d"][2] - out["eform"][2])))
            print(f"   theta={np.rad2deg(th):4.0f} M={M}: "
                  f"|R(a)-R(eform)| = {dRa:.2e}  |R(d)-R(eform)| = {dRd:.2e}  "
                  f"|J(a)-J(eform)| = {dJa:.2e}  |J(d)-J(eform)| = {dJd:.2e}")
            if th == 0.0:
                # at normal incidence the shipped te/tm rows ARE Ey/Ex
                print(f"            shipped te (Ey) R = {ship['te'][0]:.10f}  "
                      f"probe row1 R = {out['eform'][0][1]:.10f}  "
                      f"diff = {abs(ship['te'][0]-out['eform'][0][1]):.2e}")
                print(f"            shipped tm (Ex) R = {ship['tm'][0]:.10f}  "
                      f"probe row0 R = {out['eform'][0][0]:.10f}  "
                      f"diff = {abs(ship['tm'][0]-out['eform'][0][0]):.2e}")
                R[f"T2_ship_normal_M{M}"] = dict(
                    te=abs(ship["te"][0] - out["eform"][0][1]),
                    tm=abs(ship["tm"][0] - out["eform"][0][0]))
            R[f"T2_theta{int(np.rad2deg(th))}_M{M}"] = dict(
                dRa=dRa, dRd=dRd, dJa=dJa, dJd=dJd)

    print("\n## T3  ALGEBRAIC reduction of candidate (d) to the shipped E-form")
    print("   (P0, P1 with the OOP entries zeroed must rebuild Lmat / Rmat)")
    for kind in ("isotropic", "inplane_uniaxial", "gyrotropic"):
        ec, sc = cells(kind)
        for th, ph in ((0.0, 0.0), (np.deg2rad(25.0), np.deg2rad(40.0))):
            kx0, ky0 = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph)
            cell = pc.StaggeredCell(PX, PY, ec, 6, K0, kx0, ky0)
            L, G = pc.eform_operators(cell)
            P2, P1, P0 = pc.pencil_d(cell)
            qq = cell.qq
            # eliminate e3 from the (OOP-free) row 3: i q A33 e3 = Kdiv e_t
            Kdiv = -P0[2 * qq:, :2 * qq]
            A33 = P1[2 * qq:, 2 * qq:] / 1j
            Ktz = P1[:2 * qq, 2 * qq:] / 1j
            Lrec = -P0[:2 * qq, :2 * qq] - Ktz @ np.linalg.solve(A33, Kdiv)
            Grec = P2[:2 * qq, :2 * qq]
            dL = float(np.max(np.abs(Lrec - L))) / float(np.max(np.abs(L)))
            dG = float(np.max(np.abs(Grec - G))) / float(np.max(np.abs(G)))
            if sc is not None:
                sol = Granet2DTransverseE(PX, PY, 2, 2, 6, sc,
                                          alpha0x=-K0 * kx0,
                                          alpha0y=-K0 * ky0, k0=K0)
                dS = float(np.max(np.abs(Lrec - sol.Lmat))) / float(
                    np.max(np.abs(sol.Lmat)))
            else:
                dS = float("nan")
            print(f"   {kind:18s} {'normal ' if th == 0 else 'oblique'}  "
                  f"rel|L_rec - L_probe| = {dL:.2e}   rel|G_rec - G| = "
                  f"{dG:.2e}   rel|L_rec - Lmat_shipped| = {dS:.2e}")
            R[f"T3_{kind}_{'nrm' if th == 0 else 'obl'}"] = dict(
                dL=dL, dG=dG, dShipped=dS)

    with open(os.path.join(OUT, "m3_inplane_limit.json"), "w") as f:
        json.dump(R, f, indent=1, default=str)
    print("\nwrote results/m3_inplane_limit.json")


if __name__ == "__main__":
    main()
