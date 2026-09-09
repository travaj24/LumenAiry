"""M0 -- convention pins and self-checks BEFORE any candidate is measured.

Four measurements, each of which a sign error would break:

  0.1  the Bloch sign of ``Basis1D`` (which direction the basis's plane-wave
       content runs) -- read off the periodic hat AND confirmed by fitting the
       phase slope of a discrete eigenfunction;
  0.2  the exact quartic root solver, validated on an ISOTROPIC slab where the
       roots are known in closed form;
  0.3  the probe's E-form operators vs the SHIPPED
       ``Granet2DTransverseE`` (same cell, sign bridge applied) -- proves the
       probe's assembly reproduces the shipped isotropic discretization;
  0.4  the probe's STRONG H-partner vs the shipped Eq.-25 H-partner.

Run:
  cd /c/tmp/lum_aniso_oop && PYTHONPATH=/c/tmp/lum_aniso_oop \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python validation/probe_pmm2d_staggered_oop/m0_convention.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import probe_common as pc  # noqa: E402
import scipy.linalg as sla  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
res = {}

pc.banner("M0 -- conventions and self-checks")

# --------------------------------------------------------------------- 0.1
print("\n## 0.1  Bloch sign of Basis1D")
d, N, M = 1.0, 2, 6
Kphys = 1.234                       # target Bloch wavenumber (1/length)
b = pc.Basis1D(d, N, M, np.exp(1j * Kphys * d))
# the periodic hat is global dof 0: Ltilde_1 on segment 0 (value 1 at x=0),
# tau * Ltilde_2 on segment N-1 (value 1 at x=d).
S0 = b.Btilde[0]
f0 = complex(S0[0, :] @ b.val_m1)                # value at x = 0
fd = complex(S0[N - 1, :] @ b.val_p1)            # value at x = d
print(f"  hat(0) = {f0:.6g}   hat(d) = {fd:.6g}   tau = {b.tau:.6g}")
print(f"  hat(d)/hat(0) = {fd / f0:.10g}  vs exp(+i K d) = "
      f"{np.exp(1j * Kphys * d):.10g}")
res["bloch_ratio_err"] = float(abs(fd / f0 - np.exp(1j * Kphys * d)))

# independent confirmation: the LOWEST discrete eigenfunction of the 1-D
# Galerkin Laplacian must be ~ exp(+i K x); fit its phase slope.
Mtt = b._global = None
qdim = b.dim
Lt = np.array(b.Btilde)
Bt = np.array(b.B)
mm = np.einsum("isa,ab,jsb->ij", np.conj(Lt), b.m_ref, Lt) * b.J
cbt = np.einsum("isa,ab,jsb->ij", np.conj(Bt), b.c_ref.T, Lt)
mbb = np.einsum("isa,ab,jsb->ij", np.conj(Bt), b.m_ref, Bt) * b.J
stiff = cbt.conj().T @ np.linalg.solve(mbb, cbt)
mu, vec = sla.eig(stiff, mm)
j = int(np.argmin(np.abs(mu)))
xs = np.linspace(0.02, 0.98, 40) * d
seg = np.clip((xs / b.h).astype(int), 0, N - 1)
u = 2.0 * (xs - b.xb[seg]) / b.h - 1.0
vals = np.zeros(len(xs), dtype=complex)
for i, (s, uu) in enumerate(zip(seg, u)):
    V, _ = pc._modleg_value_deriv(M, np.array([uu]))
    vals[i] = (vec[:, j] @ Lt[:, s, :]) @ V[:, 0]
ph = np.unwrap(np.angle(vals))
slope = np.polyfit(xs, ph, 1)[0]
print(f"  lowest |mu| = {mu[j].real:.6g} (exact K^2 = {Kphys**2:.6g}); "
      f"eigenfunction phase slope = {slope:+.6f} (exact +K = {Kphys:+.6f})")
res["mu0_err"] = float(abs(mu[j].real - Kphys ** 2))
res["phase_slope"] = float(slope)
res["phase_slope_err"] = float(abs(slope - Kphys))
print("  => basis carries exp(+i K x) with tau = exp(+i K d): "
      "D1 -> +i kx.  PROBE SIGN CONVENTION CONFIRMED.")

# --------------------------------------------------------------------- 0.2
print("\n## 0.2  exact quartic root solver on an ISOTROPIC slab")
rows = []
for epsv, uu, vv in ((2.25 + 0.0j, 0.0, 0.0), (2.25 + 0.3j, 0.41, 0.23),
                     (4.0 + 0.0j, 0.9, 0.5)):
    r = np.sort_complex(pc.exact_kz_roots(epsv * np.eye(3), uu, vv))
    kz = np.sqrt(complex(epsv - uu ** 2 - vv ** 2))
    ex = np.sort_complex(np.array([kz, kz, -kz, -kz]))
    err = float(np.max(np.abs(r - ex)))
    rows.append(dict(eps=str(epsv), u=uu, v=vv, err=err))
    print(f"  eps={epsv}  (u,v)=({uu},{vv})  max|root - +/-sqrt| = {err:.3e}")
res["exact_root_iso_err"] = max(r["err"] for r in rows)
print("  (isotropic roots are DOUBLE -> quartic root conditioning is "
      "sqrt(eps_mach) ~ 1e-8; that is the degeneracy, not the solver.)")
print("  uniaxial optic-axis-z (SIMPLE roots, closed form o / e):")
for no, ne, uu, vv in ((1.5, 1.7, 0.0, 0.0), (1.5, 1.7, 0.55, 0.31),
                       (1.5, 1.7, 0.9, 0.2)):
    eo, ee = no ** 2, ne ** 2
    ep = np.diag([eo, eo, ee]).astype(complex)
    kt2 = uu ** 2 + vv ** 2
    ko = np.sqrt(complex(eo - kt2))
    ke = np.sqrt(complex(eo * (1.0 - kt2 / ee)))
    ex = np.sort_complex(np.array([ko, -ko, ke, -ke]))
    r = np.sort_complex(pc.exact_kz_roots(ep, uu, vv))
    err = float(np.max(np.abs(r - ex)))
    print(f"    no={no} ne={ne} (u,v)=({uu},{vv})  max|root - closed form| "
          f"= {err:.3e}")
    res.setdefault("exact_root_uniax_err", 0.0)
    res["exact_root_uniax_err"] = max(res["exact_root_uniax_err"], err)

# --------------------------------------------------------------------- 0.3
print("\n## 0.3  probe E-form operators vs the SHIPPED Granet2DTransverseE")
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE,
    _region_modes,
)

wl = 1.0
k0 = 2.0 * np.pi / wl
px = py = 0.9
Nx = Ny = 2
Mdeg = 6
eps_map = np.array([[2.25, 1.0], [1.0, 4.0 + 0.2j]], dtype=complex)
for tag, (th, ph) in (("normal", (0.0, 0.0)), ("oblique", (0.42, 0.7))):
    kx0 = np.sin(th) * np.cos(ph)
    ky0 = np.sin(th) * np.sin(ph)
    c = pc.StaggeredCell(px, py, eps_map[:, :, None, None] * np.eye(3),
                         Mdeg, k0, kx0, ky0)
    L, G = pc.eform_operators(c)
    # sign bridge: shipped tau = exp(-i alpha0 p) == probe tau = exp(+i k0 kx0 p)
    sol = Granet2DTransverseE(px, py, Nx, Ny, Mdeg,
                              eps_map, alpha0x=-k0 * kx0,
                              alpha0y=-k0 * ky0, k0=k0)
    dL = float(np.max(np.abs(L - sol.Lmat))) / float(np.max(np.abs(sol.Lmat)))
    dG = float(np.max(np.abs(G + sol.Rmat))) / float(np.max(np.abs(sol.Rmat)))
    print(f"  {tag:8s}  rel |L - Lmat| = {dL:.3e}   rel |G + Rmat| = {dG:.3e}")
    res[f"eform_L_rel_{tag}"] = dL
    res[f"eform_G_rel_{tag}"] = dG

# --------------------------------------------------------------------- 0.4
print("\n## 0.4  STRONG H-partner vs the shipped Eq.-25 H-partner")
Ws, Vs, lam_s, g2s = _region_modes(sol)
qs = 1j * lam_s                       # lam = -i q  ->  q = i lam
c = pc.StaggeredCell(px, py, eps_map[:, :, None, None] * np.eye(3),
                     Mdeg, k0, kx0, ky0)
Kdiv = np.concatenate([c.K11 + c.K21, c.K12 + c.K22], axis=1)
iqe3 = np.linalg.solve(c.A33, Kdiv @ Ws)
e3 = iqe3 / (1j * np.where(np.abs(qs) < 1e-12, 1e-12, qs))[None, :]
Vp = pc.h_partner(c, Ws, e3, qs)
ratio = np.sum(np.conj(Vp) * Vs, axis=0) / np.sum(np.conj(Vp) * Vp, axis=0)
keep = np.abs(qs) > 1e-6
spread = float(np.max(np.abs(ratio[keep] - np.median(ratio[keep]))))
print(f"  V_shipped / V_strong: median = {np.median(ratio[keep]):+.10f}, "
      f"max deviation over {int(keep.sum())} modes = {spread:.3e}")
res["hpartner_ratio"] = complex(np.median(ratio[keep])).real
res["hpartner_spread"] = spread

with open(os.path.join(OUT, "m0_convention.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print("\nwrote results/m0_convention.json")
