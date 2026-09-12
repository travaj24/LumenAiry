import numpy as np, warnings, time
t0 = time.perf_counter()
from lumenairy.elements.bor import (BORStack, guided_modes, fiber_modes,
                                    radial_coupled_modes, radial_spectrum)
from scipy.special import jn_zeros, jnp_zeros
print("import %.1f s" % (time.perf_counter() - t0), flush=True)

print("=== B1) guided_modes vs the exact fiber oracle (V=2.4) ===", flush=True)
lam = 1.55
k0 = 2*np.pi/lam
n1, n2 = 1.45, 1.44
V = 2.4
a = V/(k0*np.sqrt(n1**2-n2**2))
q_or = fiber_modes(1, a, n1**2, n2**2, k0)
neff_or = q_or[0]/k0
print(f"   a={a:.6f}  oracle HE11 neff={neff_or:.12f}", flush=True)
for Rb in (4*a, 8*a):
    for N in (150, 300, 600):
        for stag in (True, False):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = guided_modes(1, a, Rb, N, n1**2, n2**2, k0)
                q = np.atleast_1d(np.asarray(
                    out[0] if isinstance(out, tuple) else out))
                qr = np.real(q)
                band = qr[(qr > n2*k0) & (qr < n1*k0)]
                best = np.max(band)/k0 if band.size else float('nan')
                print(f"   Rbig={Rb/a:.0f}a N={N:4d}: n_band={band.size} "
                      f"neff={best:.12f} err={best-neff_or:+.3e} "
                      f"rel={abs(best-neff_or)/neff_or:.3e}", flush=True)
                break
            except Exception as e:
                print(f"   Rbig={Rb/a:.0f}a N={N} RAISED {type(e).__name__}: "
                      f"{str(e)[:200]}", flush=True)
                break

print("\n=== B1b) M1 SEM radial spectrum vs Bessel zeros ===", flush=True)
for m in (0, 1, 3):
    for bc, zf in (("dirichlet", jn_zeros), ("neumann", jnp_zeros)):
        w = radial_spectrum(m, 1.0, 8, 12, bc=bc, n_low=5)
        gam = np.sqrt(np.abs(w))
        an = zf(m, 6)
        an = an[an > 1e-12][:5] if bc == "neumann" and m == 0 else an[:5]
        print(f"   m={m} {bc:9s}: max rel err = "
              f"{np.max(np.abs(gam[:len(an)]-an)/an):.3e}", flush=True)

print("\n=== B4b) BOR reciprocity on the PROPAGATING channel block ===",
      flush=True)
S = 1e6
k0r = 2.0*np.pi/(1.0e-6*S)
s = BORStack(Rbig=24e-6*S, m=1, N=128, n_superstrate=1.41+0j,
             n_substrate=1.41+0j)
s.add_layer(0.5e-6*S, rings=(3.0e-6*S, 0.5, 2.45+0j, 1.41+0j))
s.set_source(k0=k0r)
res = s.solve()
S11, S12, S21, S22 = res["S"]
inc = np.asarray(res["inc"])
out = np.asarray(res["out"])
print(f"   inc dtype={inc.dtype} shape={inc.shape} sum={np.sum(np.real(inc))}",
      flush=True)
ii = np.where(np.real(inc) > 0.5)[0]
oo = np.where(np.real(out) > 0.5)[0]
A21 = np.asarray(S21)[np.ix_(oo, ii)]
A12 = np.asarray(S12)[np.ix_(ii, oo)]
print(f"   propagating block: |S21 - S12^T| / max|S21| = "
      f"{np.max(np.abs(A21 - A12.T))/max(np.max(np.abs(A21)),1e-30):.3e}",
      flush=True)
A11 = np.asarray(S11)[np.ix_(ii, ii)]
print(f"   |S11 - S11^T| / max|S11| (prop block) = "
      f"{np.max(np.abs(A11 - A11.T))/max(np.max(np.abs(A11)),1e-30):.3e}",
      flush=True)
# unitarity of the propagating S (lossless index-matched)
U = np.block([[A11, A12], [A21, np.asarray(S22)[np.ix_(oo, oo)]]])
print(f"   |U^H U - I| = {np.max(np.abs(U.conj().T@U - np.eye(U.shape[0]))):.3e}",
      flush=True)
e = np.asarray(res["energy"], float)
print(f"   n_ch={len(ii)}/{len(oo)}  max|E-1|={np.max(np.abs(e-1)):.3e}",
      flush=True)
