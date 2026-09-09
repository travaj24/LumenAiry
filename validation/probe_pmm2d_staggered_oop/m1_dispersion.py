"""M1 -- UNIFORM-SLAB DISPERSION: the gate that catches a sign / factor error.

For a UNIFORM cell every physical mode is a plane wave on a Bloch harmonic, so
the generator's spectrum must be the union over harmonics ``(m, n)`` of the
four exact roots of ``det(k k^T - |k|^2 I + eps) = 0`` -- to the accuracy with
which the staggered polynomial basis resolves that harmonic (the basis is
POLYNOMIAL, not Fourier, so this is a convergent statement, not an identity;
see T1b for the resolvable window).

  T1   per-harmonic residual on an M-ladder, candidates (a) and (d),
       normal AND oblique;
  T1b  the discrete 1-D channel wavenumbers vs the exact harmonics -- how many
       harmonics the basis actually carries at each M;
  T2   ASSIGNMENT census: optimal 1-1 matching of the whole spectrum against
       the 4q^2 exact roots of the resolvable window; everything off-branch is
       listed with its flux and |q|;
  T2b  is candidate (d)'s physical spectrum the SAME as candidate (a)'s?
  T3   CONVENTION ARBITRATION with the assembly deliberately mis-signed.  The
       discriminator is ``sum`` of the four (0,0) roots: exactly zero for any
       in-plane tensor, non-zero only through the OOP coupling (this is the
       AUDIT_OOP_GENERATOR_FACTOR_I asymmetric-extraordinary-pair test).

Run:
  cd /c/tmp/lum_aniso_oop && PYTHONPATH=/c/tmp/lum_aniso_oop \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python validation/probe_pmm2d_staggered_oop/m1_dispersion.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import probe_common as pc  # noqa: E402
import scipy.linalg as sla  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

WL = 1.0
PX = PY = 0.9
K0 = 2.0 * np.pi / WL


def spectrum(cell, cand):
    if cand == "a":
        W, V, qv, _e3 = pc.modes_a(cell)
        return W, V, qv, 0
    W, V, qv, _e3, n_inf = pc.modes_d(cell)
    return W, V, qv, n_inf


def channel_mu(b):
    """Discrete 1-D Bloch wavenumbers^2 carried by the staggered basis: the
    generalized eig of ``<dBtil|dBtil>`` against ``<Btil|Btil>``."""
    Lt = np.array(b.Btilde)
    Bt = np.array(b.B)
    mm = np.einsum("isa,ab,jsb->ij", np.conj(Lt), b.m_ref, Lt) * b.J
    cbt = np.einsum("isa,ab,jsb->ij", np.conj(Bt), b.c_ref.T, Lt)
    mbb = np.einsum("isa,ab,jsb->ij", np.conj(Bt), b.m_ref, Bt) * b.J
    stiff = cbt.conj().T @ np.linalg.solve(mbb, cbt)
    return np.real(sla.eig(stiff, mm)[0])


def resolvable_roots(eps33, kx0, ky0, q):
    """The ``4 q^2`` EXACT roots of the ``q x q`` harmonics with the smallest
    ``|kx|, |ky|`` -- the window the staggered basis can carry."""
    ms = np.arange(-60, 61)
    kx = kx0 + ms * (WL / PX)
    ky = ky0 + ms * (WL / PY)
    mx = ms[np.argsort(np.abs(kx))[:q]]
    my = ms[np.argsort(np.abs(ky))[:q]]
    out = []
    for m in mx:
        for n in my:
            out.extend(complex(z) for z in pc.exact_kz_roots(
                eps33, kx0 + m * (WL / PX), ky0 + n * (WL / PY)))
    return np.array(out)


def channel_u(basis, k0v):
    """SIGNED discrete transverse wavenumbers ``u_j`` (in ``k0`` units) that the
    staggered basis actually carries: ``|u_j| = sqrt(mu_j)/k0`` from the 1-D
    Galerkin Laplacian, the sign taken from the exact harmonic it pairs with
    (the two orderings are monotone, so the pairing is unambiguous)."""
    mu = np.sort(np.sqrt(np.abs(channel_mu(basis)))) / k0v
    ms = np.arange(-80, 81)
    kx = np.sort(np.abs(basis._K0 + ms * (WL / basis.d)))[:mu.size]  # magnitude
    sgn = []
    kall = basis._K0 + ms * (WL / basis.d)
    order = np.argsort(np.abs(kall))[:mu.size]
    for j in range(mu.size):
        sgn.append(np.sign(kall[order[j]]) or 1.0)
    del kx
    return mu * np.array(sgn)


def discrete_channel_roots(eps33, bx, by, k0v):
    """The ``4 q^2`` roots of the quartic on the DISCRETE channel wavenumbers.

    For a uniform cell the staggered generator block-diagonalizes into
    ``(jx, jy)`` channels whose transverse symbols are the discrete
    ``u_jx, v_jy``; if the generator carries the exact plane-wave symbol, its
    spectrum IS this set (independently of how well the basis resolves the
    physical harmonics)."""
    us = channel_u(bx, k0v)
    vs = channel_u(by, k0v)
    out = []
    for u in us:
        for v in vs:
            out.extend(complex(z) for z in pc.exact_kz_roots(eps33, u, v))
    return np.array(out)


def per_harmonic(eps33, cand, Nx, M, th, ph, mlist):
    kx0, ky0 = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph)
    cell = pc.StaggeredCell(PX, PY, pc.tile(eps33, Nx, Nx), M, K0, kx0, ky0)
    t0 = time.perf_counter()
    _W, _V, qv, n_inf = spectrum(cell, cand)
    dt = time.perf_counter() - t0
    out = {}
    for (m, n) in mlist:
        r4 = pc.exact_kz_roots(eps33, kx0 + m * (WL / PX),
                               ky0 + n * (WL / PY))
        out[(m, n)] = max(float(np.min(np.abs(qv - r)) / max(abs(r), 1.0))
                          for r in r4)
    return out, qv.size, dt, n_inf


def census(eps33, cand, Nx, M, th, ph, off_cut=0.3,
           root_set="exact"):
    kx0, ky0 = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph)
    cell = pc.StaggeredCell(PX, PY, pc.tile(eps33, Nx, Nx), M, K0, kx0, ky0)
    W, V, qv, n_inf = spectrum(cell, cand)
    flux = pc.modal_flux(cell, W, V)
    fl = np.abs(flux) / max(float(np.max(np.abs(flux))), 1e-30)
    rv = (resolvable_roots(eps33, kx0, ky0, cell.q) if root_set == "exact"
          else discrete_channel_roots(eps33, cell.bx, cell.by, K0))
    Cst = np.abs(qv[:, None] - rv[None, :])
    nroot = rv.size
    if Cst.shape[0] > nroot:                       # (d): 5q^2 modes, 4q^2 roots
        Cst = np.concatenate(
            [Cst, np.full((Cst.shape[0], Cst.shape[0] - nroot), 1e6)], axis=1)
    ri, ci = linear_sum_assignment(Cst)
    real = ci < nroot
    dist = Cst[ri[real], ci[real]]
    off = dist > off_cut
    offi = np.concatenate([ri[real][off], ri[~real]])
    return dict(dim=int(qv.size), n_inf=int(n_inf), nroot=int(nroot),
                med=float(np.median(dist)), mx=float(np.max(dist)),
                off=int(offi.size),
                off_flux=float(np.max(fl[offi])) if offi.size else 0.0,
                off_absq_max=float(np.max(np.abs(qv[offi]))) if offi.size else 0.0,
                off_absq_min=float(np.min(np.abs(qv[offi]))) if offi.size else 0.0,
                off_req_max=(float(np.max(np.abs(np.real(qv[offi]))))
                             if offi.size else 0.0),
                off_decay_min=(float(np.min(np.abs(np.imag(qv[offi]))))
                               if offi.size else 0.0))


def _conj_oop(e):
    o = e.copy()
    for i, j in ((0, 2), (1, 2), (2, 0), (2, 1)):
        o[i, j] = np.conj(e[i, j])
    return o


def _tr_oop(e):
    o = e.copy()
    o[0, 2], o[2, 0] = e[2, 0], e[0, 2]
    o[1, 2], o[2, 1] = e[2, 1], e[1, 2]
    return o


def _neg_oop(e):
    return e * np.array([[1, 1, -1], [1, 1, -1], [-1, -1, 1]])


def _drop_oop(e):
    return e * np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])


def main():
    pc.banner("M1 -- uniform-slab dispersion")
    R = {}
    TIL = pc.uniaxial(1.5, 1.7, 35.0)
    TILA = pc.uniaxial(1.5, 1.7, 35.0, azim_deg=40.0)
    TILL = pc.uniaxial(1.5, 1.7, 35.0, loss=0.08)
    ASYM = pc.uniaxial(1.5, 1.7, 35.0, loss=0.08).copy()
    ASYM[0, 2] += 0.11j                      # e13 != e31 (non-reciprocal-like)
    ASYM[2, 0] -= 0.11j
    print("\ntilted uniaxial (no=1.5, ne=1.7, 35 deg):")
    print(np.array2string(TIL, precision=5))
    mlist = [(0, 0), (1, 0), (0, 1), (-1, 0), (1, 1), (-1, -1), (2, 0)]

    print("\n## T1  per-harmonic max residual (nearest generator eigenvalue)")
    for cand, Ms in (("a", (5, 6, 7, 8, 9, 10)), ("d", (5, 6, 7, 8))):
        for tag, (th, ph) in (("normal", (0.0, 0.0)),
                              ("oblique25", (np.deg2rad(25.0),
                                             np.deg2rad(40.0)))):
            print(f"\n  candidate ({cand})  {tag}  Nx=2  tilted uniaxial 35deg")
            print("    M   dim    eig[s] " +
                  " ".join(f"({m:+d},{n:+d})" for m, n in mlist))
            for M in Ms:
                tb, dim, dt, ni = per_harmonic(TIL, cand, 2, M, th, ph, mlist)
                print(f"    {M:2d} {dim:5d} {dt:8.2f} " +
                      " ".join(f"{tb[k]:8.1e}" for k in mlist))
                R[f"T1_{cand}_{tag}_M{M}"] = dict(
                    dim=int(dim), eig_s=dt, n_inf=int(ni),
                    **{f"h{m}_{n}": float(tb[(m, n)]) for m, n in mlist})

    print("\n## T1b  discrete 1-D channel wavenumbers vs the exact harmonics")
    for tag, th, ph in (("normal", 0.0, 0.0),
                        ("oblique25", np.deg2rad(25.0), np.deg2rad(40.0))):
        kx0 = np.sin(th) * np.cos(ph)
        print(f"    {tag}:   M   q   max|sqrt(mu_j) - |kx_m||   "
              f"resolved(<1e-8)")
        for M in (5, 6, 7, 8, 9, 10):
            b = pc.Basis1D(PX, 2, M, np.exp(1j * K0 * kx0 * PX))
            b._K0 = kx0                      # in k0 units
            mu = channel_mu(b)
            ms = np.arange(-40, 41)
            ex = np.sort(np.abs(kx0 + ms * (WL / PX)))[:mu.size]
            got = np.sort(np.sqrt(np.abs(mu))) / K0
            err = np.abs(got - ex)
            print(f"            {M:2d} {b.dim:3d}      {np.max(err):.2e}"
                  f"              {int((err < 1e-8).sum())}/{mu.size}")
            R[f"T1b_{tag}_M{M}"] = dict(q=int(b.dim), maxerr=float(np.max(err)),
                                        resolved=int((err < 1e-8).sum()))

    print("\n## T2  ASSIGNMENT census (optimal 1-1 matching of the whole "
          "spectrum).")
    print("       root set 'discrete' = the quartic on the DISCRETE channel "
          "wavenumbers the")
    print("       basis carries; 'exact' = the physical harmonics (adds the "
          "basis resolution")
    print("       error).  off = matched distance > 0.3.")
    print("   cand case          rootset  dim n_inf nroot  med.dist  max.dist"
          "   off |flux|rel min|q|off  min Re(lam) off")
    for cand in ("a", "d"):
        for tag, eps33, th, ph in (
                ("normal/tilt35", TIL, 0.0, 0.0),
                ("obl25/tilt35", TIL, np.deg2rad(25.0), np.deg2rad(40.0)),
                ("obl25/azim40", TILA, np.deg2rad(25.0), np.deg2rad(40.0)),
                ("obl25/lossy", TILL, np.deg2rad(25.0), np.deg2rad(40.0))):
            M = 8 if cand == "a" else 7
            for rs in ("discrete", "exact"):
                r = census(eps33, cand, 2, M, th, ph, root_set=rs)
                print(f"   ({cand})  {tag:14s} {rs:8s} {r['dim']:5d} "
                      f"{r['n_inf']:5d} {r['nroot']:5d}  {r['med']:.2e}  "
                      f"{r['mx']:.2e} {r['off']:5d}  {r['off_flux']:.1e}  "
                      f"{r['off_absq_min']:.2e}  {r['off_decay_min']:.2e}")
                R[f"T2_{cand}_{tag}_{rs}"] = r

    print("\n## T2b  candidate (a) vs (d): is the PHYSICAL spectrum identical?")
    for tag, th, ph in (("normal", 0.0, 0.0),
                        ("oblique25", np.deg2rad(25.0), np.deg2rad(40.0))):
        for M in (5, 6, 7):
            kx0, ky0 = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph)
            cell = pc.StaggeredCell(PX, PY, pc.tile(TIL, 2, 2), M, K0, kx0, ky0)
            _, _, qa, _ = pc.modes_a(cell)
            _, _, qd, _, _ = pc.modes_d(cell)
            da = np.array([np.min(np.abs(qd - z)) for z in qa])
            dd = np.array([np.min(np.abs(qa - z)) for z in qd])
            ex = dd > 1e-6
            rng = (f"[{np.min(np.abs(qd[ex])):.2e},{np.max(np.abs(qd[ex])):.2e}]"
                   if ex.any() else "-")
            print(f"    {tag:10s} M={M} 4q^2={qa.size} 5q^2={qd.size}  "
                  f"max_a min_d|qa-qd| = {np.max(da):.2e}  "
                  f"(d)-only: {int(ex.sum())} (q^2={cell.qq}) |q| {rng}")
            R[f"T2b_{tag}_M{M}"] = dict(
                na=int(qa.size), nd=int(qd.size), maxd=float(np.max(da)),
                d_only=int(ex.sum()))

    print("\n## T3  convention arbitration (assembly mis-signed; exact roots "
          "unchanged).")
    print("   DISCRIMINATOR: sum of the four (0,0) roots -- exactly 0 for any "
          "in-plane\n   tensor, non-zero only through the OOP coupling.")
    th, ph = np.deg2rad(25.0), np.deg2rad(40.0)
    kx0, ky0 = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph)
    variants = (("reference", lambda e: e), ("negate_oop", _neg_oop),
                ("conj_oop", _conj_oop), ("transpose_oop", _tr_oop),
                ("drop_oop", _drop_oop))
    for pname, EPS in (("lossy_tilt35", TILL), ("asym_oop", ASYM)):
        rr = np.sort_complex(pc.exact_kz_roots(EPS, kx0, ky0))
        ssum = complex(np.sum(rr))
        print(f"\n   probe tensor {pname}:  e13={EPS[0,2]:.4f} "
              f"e31={EPS[2,0]:.4f}")
        print(f"   exact (0,0) roots {np.array2string(rr, precision=6)}")
        print(f"   exact sum(roots) = {ssum:.6f}")
        print("   variant         cand  h(0,0)    h(+1,0)   "
              "sum(4 matched q)        |err|")
        for name, fn in variants:
            for cand in ("a", "d"):
                cell = pc.StaggeredCell(PX, PY, pc.tile(fn(EPS.copy()), 2, 2),
                                        8 if cand == "a" else 7, K0, kx0, ky0)
                _W, _V, qv, _ = spectrum(cell, cand)
                ds = []
                for m, n in ((0, 0), (1, 0)):
                    r4 = pc.exact_kz_roots(EPS, kx0 + m * (WL / PX),
                                           ky0 + n * (WL / PY))
                    ds.append(max(float(np.min(np.abs(qv - r))
                                        / max(abs(r), 1.0)) for r in r4))
                gs = complex(np.sum([qv[np.argmin(np.abs(qv - r))]
                                     for r in rr]))
                print(f"   {name:15s} ({cand})  {ds[0]:.2e}  {ds[1]:.2e}  "
                      f"{gs.real:+.6f}{gs.imag:+.6f}j   {abs(gs - ssum):.2e}")
                R[f"T3_{pname}_{name}_{cand}"] = dict(
                    h00=ds[0], h10=ds[1], sum_err=float(abs(gs - ssum)))
        R[f"T3_{pname}_exact_sum"] = str(ssum)

    with open(os.path.join(OUT, "m1_dispersion.json"), "w") as f:
        json.dump(R, f, indent=1, default=str)
    print("\nwrote results/m1_dispersion.json")


if __name__ == "__main__":
    main()
