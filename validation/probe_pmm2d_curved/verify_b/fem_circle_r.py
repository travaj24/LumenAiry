"""[VERIFY-B copy of ../fem/fem_circle.py, only change: --rad sets the pillar radius (nm) and
results go to fem_r<rad>_results.jsonl here.]  Independent 3-D FEM oracle: periodic circular-pillar grating, per-order diffraction efficiencies.

Fixture (nm): lambda 1000; square lattice P = 1200; circular pillar r = 360 (eps 4) in air (eps 1),
height 500; superstrate air; substrate n = 1.45.  Normal incidence from the superstrate, E along y.

Method: plain NGSolve (the engine DynaMeta's solve_fem wraps), written out here so the per-order
Fourier extraction has access to the solved field.  Same formulation as DynaMeta solve_fem:
scattered field w.r.t. the analytic air/substrate Fresnel background, HCurl (Nedelec) elements,
complex-stretch PML half-spaces top and bottom, QUARTER cell [0, P/2]^2 with mirror walls:
  x = 0, P/2 : PMC (natural BC; Ex odd, Ey/Ez even in x)
  y = 0, P/2 : PEC (Dirichlet;  Ex/Ez odd, Ey even in y)
Curved (high-order geometry) elements: mesh.Curve(curve_order) on the cylinder surface.

Per-order extraction: the full-cell Fourier coefficient of order (m, n) is recovered from the
quarter cell with the parity kernels
  Ey_mn =  (4/P^2) Int_Q Ey cos(G m x) cos(G n y)
  Ex_mn = -(4/P^2) Int_Q Ex sin(G m x) sin(G n y)
integrated EXACTLY (element quadrature) over three homogeneous slabs in each buffer, then a
per-order two-wave (up/down) least-squares fit across the slabs.  The down-going (PML-reflected)
amplitude is reported as a diagnostic.  Efficiency = |E|^2 Re(kz_mn) / kz_inc with
Ez = -/+ (kx Ex + ky Ey) / kz.  ASCII-only source.
"""
import argparse
import json
import math
import os
import time

import numpy as np

WL = 1000.0
P = 1200.0
Q = P / 2.0
RAD = 360.0
H = 500.0
EPS_P = 4.0
N_SUB = 1.45
K0 = 2.0 * math.pi / WL
G = 2.0 * math.pi / P
ORD = [(m, n) for m in range(3) for n in range(3)]          # distinct (|m|, |n|) up to 2


def mult(m, n):
    return (2 if m else 1) * (2 if n else 1)


def build(hscale=1.0, dpml=900.0, buf=450.0, nslab=3, h_edge=None, verbose=True, dm_names=False):
    import netgen.occ as occ
    import ngsolve as ng
    hp, ha, hs, hpml = 110.0 * hscale, 150.0 * hscale, 130.0 * hscale, 120.0 * hscale
    solids = []

    def box(z0, z1):
        return occ.Box(occ.Pnt(0, 0, z0), occ.Pnt(Q, Q, z1))

    def add(s, name, mh):
        s.mat(name)
        s.maxh = mh
        solids.append(s)
    zs = -buf
    add(box(zs - dpml, zs), "pml_bot", hpml)
    dz = buf / nslab
    for i in range(nslab):
        add(box(-buf + i * dz, -buf + (i + 1) * dz), "substrate" if dm_names else "sub%d" % i, hs)
    cyl = occ.Cylinder(occ.Pnt(0, 0, 0), occ.Z, r=RAD, h=H)
    pil = cyl * box(0.0, H)
    add(pil, "pillar", hp)
    add(box(0.0, H) - pil, "host", ha)
    for i in range(nslab):
        add(box(H + i * dz, H + (i + 1) * dz), "superstrate" if dm_names else "sup%d" % i, ha)
    zt = H + buf
    add(box(zt, zt + dpml), "pml_top", hpml)
    glued = occ.Glue(solids)
    tol = 1e-6 * Q
    for f in glued.faces:
        c = f.center
        if abs(c.x) < tol or abs(c.x - Q) < tol:
            f.name = "sym_x" if dm_names else "pmc"
        elif abs(c.y) < tol or abs(c.y - Q) < tol:
            f.name = "sym_y" if dm_names else "pec"
        else:
            f.name = "default"
    for s in glued.solids:                       # maxh on the GLUED shape
        nm = s.name
        s.maxh = {"pillar": hp, "host": ha, "pml_bot": hpml, "pml_top": hpml}.get(
            nm, hs if nm.startswith("sub") else ha)
    if h_edge:
        for e in glued.edges:                    # the two circular pillar rims
            c = e.center
            rr = math.hypot(c.x, c.y)
            if (abs(c.z) < tol or abs(c.z - H) < tol) and 0.3 * RAD < rr < RAD * 1.01 and \
                    c.x > tol and c.y > tol:
                e.maxh = h_edge
    geo = occ.OCCGeometry(glued)
    mesh = ng.Mesh(geo.GenerateMesh(maxh=max(ha, hs, hpml)))
    if verbose:
        print("mesh: %d vertices, %d tets, materials %s, bnds %s" % (
            mesh.nv, mesh.ne, sorted(set(mesh.GetMaterials())), sorted(set(mesh.GetBoundaries()))),
            flush=True)
    return mesh, dict(zs=zs, zt=zt, buf=buf, nslab=nslab, dz=dz, dpml=dpml)


def solve(mesh, lay, p=2, curve=None, alpha=2.0, solver="sparsecholesky", nthreads=6,
          rtol=1e-10, verbose=True):
    import ngsolve as ng
    ng.SetNumThreads(nthreads)
    curve = p if curve is None else curve
    mesh.Curve(curve)
    try:
        mesh.UnSetPML("pml_top")
        mesh.UnSetPML("pml_bot")
    except Exception:
        pass
    mesh.SetPML(ng.pml.HalfSpace(point=(0, 0, lay["zt"]), normal=(0, 0, 1), alpha=1j * alpha), "pml_top")
    mesh.SetPML(ng.pml.HalfSpace(point=(0, 0, lay["zs"]), normal=(0, 0, -1), alpha=1j * alpha), "pml_bot")
    fes = ng.HCurl(mesh, order=p, complex=True, dirichlet="pec")
    u, v = fes.TnT()
    epsd = {"pillar": EPS_P, "pml_bot": N_SUB ** 2}
    epsd.update({"sub%d" % i: N_SUB ** 2 for i in range(lay["nslab"])})
    eps = mesh.MaterialCF(epsd, default=1.0)
    kz_s = K0
    kz_sub = K0 * N_SUB
    r_f = (kz_s - kz_sub) / (kz_s + kz_sub)             # z_int = 0
    t_f = 2.0 * kz_s / (kz_s + kz_sub)
    bg_sup = ng.exp(-1j * kz_s * ng.z) + r_f * ng.exp(1j * kz_s * ng.z)
    Ebg = ng.CoefficientFunction((0.0, bg_sup, 0.0))    # valid in z > 0 (the pillar)
    cond = solver != "umfpack"
    a = ng.BilinearForm(fes, symmetric=True, condense=cond)
    a += (ng.curl(u) * ng.curl(v) - K0 ** 2 * eps * u * v) * ng.dx
    f = ng.LinearForm(fes)
    f += (K0 ** 2 * (EPS_P - 1.0) * Ebg * v) * ng.dx(definedon=mesh.Materials("pillar"))
    pre = ng.Preconditioner(a, "bddc") if solver.startswith("bddc") else None
    gfu = ng.GridFunction(fes)
    t0 = time.time()
    info = dict(ndof=fes.ndof, nfree=int(sum(fes.FreeDofs(cond))))
    if verbose:
        print("  p=%d curve=%d ndof %d (condensed free %d), solver %s" % (
            p, curve, fes.ndof, info["nfree"], solver), flush=True)
    with ng.TaskManager():
        a.Assemble()
        f.Assemble()
        if solver in ("sparsecholesky", "umfpack"):
            inv = a.mat.Inverse(fes.FreeDofs(cond), inverse=solver)
            if cond:
                f.vec.data += a.harmonic_extension_trans * f.vec
                gfu.vec.data = inv * f.vec
                gfu.vec.data += a.harmonic_extension * gfu.vec
                gfu.vec.data += a.inner_solve * f.vec
            else:
                gfu.vec.data = inv * f.vec
        else:
            f.vec.data += a.harmonic_extension_trans * f.vec
            ng.solvers.GMRes(A=a.mat, b=f.vec, pre=pre.mat, x=gfu.vec, tol=rtol, maxsteps=3000,
                             printrates=False)
            gfu.vec.data += a.harmonic_extension * gfu.vec
            gfu.vec.data += a.inner_solve * f.vec
    info["solve_s"] = time.time() - t0
    # residual on the full (non-condensed) system as an independent check
    a2 = ng.BilinearForm(fes, symmetric=True)
    a2 += (ng.curl(u) * ng.curl(v) - K0 ** 2 * eps * u * v) * ng.dx
    f2 = ng.LinearForm(fes)
    f2 += (K0 ** 2 * (EPS_P - 1.0) * Ebg * v) * ng.dx(definedon=mesh.Materials("pillar"))
    with ng.TaskManager():
        a2.Assemble()
        f2.Assemble()
    r = f2.vec.CreateVector()
    r.data = f2.vec - a2.mat * gfu.vec
    fd = np.fromiter(fes.FreeDofs(), dtype=bool, count=len(r))
    info["relres"] = float(np.linalg.norm(r.FV().NumPy()[fd]) / np.linalg.norm(f2.vec.FV().NumPy()[fd]))
    del a2, f2
    return gfu, dict(r_f=r_f, t_f=t_f, **info)


def slab_basis(kz, z0, z1, sgn):
    """Slab average of exp(sgn i kz z) over [z0, z1]."""
    if abs(kz) < 1e-14:
        return 1.0
    return (np.exp(sgn * 1j * kz * z1) - np.exp(sgn * 1j * kz * z0)) / (sgn * 1j * kz * (z1 - z0))


def extract(mesh, gfu, lay, p, bg):
    import ngsolve as ng
    kers = []
    for (m, n) in ORD:
        kers.append(gfu[1] * ng.cos(G * m * ng.x) * ng.cos(G * n * ng.y))
        kers.append(gfu[0] * ng.sin(G * m * ng.x) * ng.sin(G * n * ng.y))
    vec = ng.CoefficientFunction(tuple(kers))
    iorder = 2 * p + 6
    out = {"R": {}, "T": {}, "diag": {}}
    for side in ("sup", "sub"):
        z_lo = H if side == "sup" else lay["zs"]
        zb = [(z_lo + i * lay["dz"], z_lo + (i + 1) * lay["dz"]) for i in range(lay["nslab"])]
        vals = []
        for i in range(lay["nslab"]):
            with ng.TaskManager():
                I = ng.Integrate(vec, mesh, order=iorder, definedon=mesh.Materials("%s%d" % (side, i)))
            vals.append(np.asarray(I, complex) * 4.0 / (P * P * lay["dz"]))
        vals = np.array(vals)                       # [slab, 2*len(ORD)]
        nmed = 1.0 if side == "sup" else N_SUB
        for k, (m, n) in enumerate(ORD):
            kx, ky = G * m, G * n
            kz = np.sqrt(complex((K0 * nmed) ** 2 - kx * kx - ky * ky))
            if kz.imag < 0:
                kz = -kz
            prop = abs(kz.imag) < 1e-12
            # outgoing: up (+) in sup, down (-) in sub
            so = +1 if side == "sup" else -1
            M = np.array([[slab_basis(kz, z0, z1, so), slab_basis(kz, z0, z1, -so)] for (z0, z1) in zb])
            Ey_s, Ex_s = vals[:, 2 * k], -vals[:, 2 * k + 1]
            cy, resy, *_ = np.linalg.lstsq(M, Ey_s, rcond=None)
            cx, resx, *_ = np.linalg.lstsq(M, Ex_s, rcond=None)
            Ey_o, Ex_o = complex(cy[0]), complex(cx[0])
            Ey_i, Ex_i = complex(cy[1]), complex(cx[1])
            if (m, n) == (0, 0):
                Ey_o += bg["r_f"] if side == "sup" else bg["t_f"]
            fit_res = float(max(np.linalg.norm(M @ cy - Ey_s), np.linalg.norm(M @ cx - Ex_s)) /
                            max(np.linalg.norm(Ey_s) + np.linalg.norm(Ex_s), 1e-300))
            key = "%d,%d" % (m, n)
            if prop:
                Ez = -so * (kx * Ex_o + ky * Ey_o) / kz
                eff = (abs(Ex_o) ** 2 + abs(Ey_o) ** 2 + abs(Ez) ** 2) * kz.real / K0
                # incoming (reflected-back) wave power as a PML diagnostic
                Ez_i = so * (kx * Ex_i + ky * Ey_i) / kz
                eff_in = (abs(Ex_i) ** 2 + abs(Ey_i) ** 2 + abs(Ez_i) ** 2) * kz.real / K0
            else:
                eff, eff_in = 0.0, None
            out["R" if side == "sup" else "T"][key] = dict(
                eff=float(eff), propagating=bool(prop), Ex=[Ex_o.real, Ex_o.imag],
                Ey=[Ey_o.real, Ey_o.imag], incoming_eff=eff_in, fit_relres=fit_res)
    # INDEPENDENT energy check: z-Poynting flux of the reflected / transmitted field (scattered +
    # analytic background part), volume-averaged over each buffer slab (flux is z-independent in a
    # lossless source-free slab); no Fourier projection involved.
    Erf = gfu + ng.CoefficientFunction((0.0, bg["r_f"] * ng.exp(1j * K0 * ng.z), 0.0))
    Etr = gfu + ng.CoefficientFunction((0.0, bg["t_f"] * ng.exp(-1j * K0 * N_SUB * ng.z), 0.0))
    cE_r = ng.curl(gfu) + ng.CoefficientFunction((-1j * K0 * bg["r_f"] * ng.exp(1j * K0 * ng.z), 0.0, 0.0))
    cE_t = ng.curl(gfu) + ng.CoefficientFunction((1j * K0 * N_SUB * bg["t_f"] * ng.exp(-1j * K0 * N_SUB * ng.z), 0.0, 0.0))

    def sz(E, cE):
        return 1j * (E[0] * ng.Conj(cE[1]) - E[1] * ng.Conj(cE[0]))
    area = Q * Q
    fl = {"sup": [], "sub": []}
    for side, E, cE in (("sup", Erf, cE_r), ("sub", Etr, cE_t)):
        for i in range(lay["nslab"]):
            with ng.TaskManager():
                v = ng.Integrate(sz(E, cE), mesh, order=iorder, definedon=mesh.Materials("%s%d" % (side, i)))
            fl[side].append(float(np.real(v)) / (area * lay["dz"] * K0))
    out["R_flux"] = fl["sup"]
    out["T_flux"] = [-x for x in fl["sub"]]
    Rs = sum(mult(*map(int, k.split(","))) * v["eff"] for k, v in out["R"].items())
    Ts = sum(mult(*map(int, k.split(","))) * v["eff"] for k, v in out["T"].items())
    out["sumR"], out["sumT"], out["RplusT"] = Rs, Ts, Rs + Ts
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rad", type=float, default=360.0)
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--p", type=int, nargs="+", default=[2])
    ap.add_argument("--curve", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--dpml", type=float, default=900.0)
    ap.add_argument("--hedge", type=float, default=None)
    ap.add_argument("--solver", default="sparsecholesky")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--dry", action="store_true")
    A = ap.parse_args()
    global RAD
    RAD = A.rad
    A.out = "fem_r%d_results.jsonl" % int(round(RAD))
    import ngsolve as ng
    t0 = time.time()
    mesh, lay = build(A.h, dpml=A.dpml, h_edge=A.hedge)
    tmesh = time.time() - t0
    if A.dry:
        for p in A.p:
            fes = ng.HCurl(mesh, order=p, complex=True, dirichlet="pec")
            print("p=%d ndof %d" % (p, fes.ndof))
        return
    for p in A.p:
        t1 = time.time()
        gfu, info = solve(mesh, lay, p=p, curve=A.curve, alpha=A.alpha, solver=A.solver,
                          nthreads=A.threads)
        res = extract(mesh, gfu, lay, p, info)
        rec = dict(rad=RAD, tag=A.tag, hscale=A.h, p=p, curve=(p if A.curve is None else A.curve),
                   alpha=A.alpha, dpml=A.dpml, hedge=A.hedge, ntet=mesh.ne, nv=mesh.nv,
                   mesh_s=tmesh, total_s=time.time() - t1, solver=A.solver, **info, **res)
        rec.pop("r_f")
        rec.pop("t_f")
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), A.out), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print("[%s h%.2f p%d] ndof %d  %.0f s  relres %.1e  R00 %.6f R10 %.6f R01 %.6f | "
              "T00 %.6f T10 %.6f T01 %.6f T11 %.6f | sumR %.6f sumT %.6f R+T %.7f | flux R %.6f T %.6f" % (
                  A.tag, A.h, p, info["ndof"], rec["total_s"], info["relres"],
                  res["R"]["0,0"]["eff"], res["R"]["1,0"]["eff"], res["R"]["0,1"]["eff"],
                  res["T"]["0,0"]["eff"], res["T"]["1,0"]["eff"], res["T"]["0,1"]["eff"],
                  res["T"]["1,1"]["eff"], res["sumR"], res["sumT"], res["RplusT"], np.mean(res["R_flux"]), np.mean(res["T_flux"])), flush=True)
        del gfu


if __name__ == "__main__":
    main()
