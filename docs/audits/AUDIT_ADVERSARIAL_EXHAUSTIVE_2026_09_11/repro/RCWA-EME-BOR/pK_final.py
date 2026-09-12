import numpy as np, warnings, time
t0 = time.perf_counter()
from lumenairy.elements.rcwa import rcwa_efficiency_1d, rcwa_efficiency_2d
from lumenairy.elements.bor import (BORStack, guided_modes, fiber_modes,
                                    radial_coupled_modes)
print("import %.1f s" % (time.perf_counter() - t0), flush=True)

print("=== V) 2-D(n_orders_y=0) vs 1-D: convergence in CELL PIXELS Sx ===",
      flush=True)
duty = 0.5
base = {}
for pol in ("te", "tm"):
    o1, R1, T1 = rcwa_efficiency_1d(0.5e-6, 2.0, 1.0, 1.5, 1.0, 0.3e-6, duty,
                                    0.633e-6, angle=0.0, polarization=pol,
                                    n_orders=8, formulation='li')
    base[pol] = (R1, T1)
for Sx in (64, 128, 256, 512, 1024, 2048, 4096):
    xg = np.arange(Sx)/Sx
    cell = np.where(xg < duty, 4.0, 1.0)[:, None]
    row = []
    for pol in ("te", "tm"):
        R1, T1 = base[pol]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = rcwa_efficiency_2d(0.5e-6, 0.5e-6, cell, 1.5, 1.0, 0.3e-6,
                                   0.633e-6, theta=0.0, phi=0.0,
                                   polarization=pol, n_orders_x=8,
                                   n_orders_y=0, formulation='li')
        oo, R2, T2 = np.asarray(r[0]), np.asarray(r[1]), np.asarray(r[2])
        sel = {int(v[0]): i for i, v in enumerate(oo)}
        dR = max(abs(R1[8+m] - R2[sel[m]]) for m in (-1, 0, 1) if m in sel)
        dT = max(abs(T1[8+m] - T2[sel[m]]) for m in (-1, 0, 1) if m in sel)
        row.append((pol, dR, dT))
    print("   Sx=%5d  " % Sx +
          "  ".join(f"{p}: dR={a:.2e} dT={b:.2e}" for p, a, b in row),
          flush=True)

print("\n=== B4b) BOR reciprocity / unitarity on the propagating block ===",
      flush=True)
S = 1e6
k0r = 2.0*np.pi/(1.0e-6*S)
s = BORStack(Rbig=12e-6*S, m=1, N=96, n_superstrate=1.41+0j,
             n_substrate=1.41+0j)
s.add_layer(0.5e-6*S, rings=(3.0e-6*S, 0.5, 2.45+0j, 1.41+0j))
s.set_source(k0=k0r)
res = s.solve()
S11, S12, S21, S22 = res["S"]
inc = np.real(np.asarray(res["inc"]))
out = np.real(np.asarray(res["out"]))
ii = np.where(inc > 0.5)[0]
oo = np.where(out > 0.5)[0]
A11 = np.asarray(S11)[np.ix_(ii, ii)]
A12 = np.asarray(S12)[np.ix_(ii, oo)]
A21 = np.asarray(S21)[np.ix_(oo, ii)]
A22 = np.asarray(S22)[np.ix_(oo, oo)]
print(f"   n_inc={len(ii)} n_out={len(oo)}", flush=True)
print(f"   |S21-S12^T|/max|S21| = "
      f"{np.max(np.abs(A21-A12.T))/max(np.max(np.abs(A21)),1e-30):.3e}",
      flush=True)
print(f"   |S11-S11^T|/max|S11| = "
      f"{np.max(np.abs(A11-A11.T))/max(np.max(np.abs(A11)),1e-30):.3e}",
      flush=True)
U = np.block([[A11, A12], [A21, A22]])
print(f"   |U^H U - I| = "
      f"{np.max(np.abs(U.conj().T@U-np.eye(U.shape[0]))):.3e}", flush=True)
e = np.asarray(res["energy"], float)
print(f"   max|E-1| = {np.max(np.abs(e-1)):.3e}", flush=True)

print("\n=== B1c) guided_modes debug on the V=2.4 fiber ===", flush=True)
lam = 1.55
k0 = 2*np.pi/lam
n1, n2 = 1.45, 1.44
V = 2.4
a = V/(k0*np.sqrt(n1**2-n2**2))
neff_or = fiber_modes(1, a, n1**2, n2**2, k0)[0]/k0
print(f"   oracle neff={neff_or:.12f}   guided window neff in "
      f"({n2:.4f}, {n1:.4f})", flush=True)
for Rb, N in ((6*a, 300), (6*a, 600), (12*a, 600)):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        def ep(rr):
            return np.where(rr <= a, n1**2, n2**2).astype(complex)
        md = radial_coupled_modes(1, Rb, N, ep, k0, staggered=True)
    q = np.array([m_["q"] for m_ in md])
    rd = np.array([m_["reldiv"] for m_ in md])
    sel = (q.real/k0 > n2) & (q.real/k0 < n1) & (np.abs(q.imag) < 1e-6*k0)
    print(f"   Rbig={Rb/a:.0f}a N={N}: {sel.sum()} modes in the guided window; "
          f"neff={np.sort(q.real[sel]/k0)[::-1][:3]}  reldiv="
          f"{rd[sel][:3] if sel.sum() else []}", flush=True)
    if sel.sum():
        best = np.max(q.real[sel])/k0
        print(f"      best neff={best:.12f} err={best-neff_or:+.3e} "
              f"rel={abs(best-neff_or)/neff_or:.3e}", flush=True)
    gm = guided_modes(1, a, Rb, N, n1**2, n2**2, k0)
    print(f"      guided_modes() returned {len(gm)} modes", flush=True)
    if gm:
        qq = np.array([g["q"] if isinstance(g, dict) else g for g in gm])
        print(f"      neff = {np.real(qq)/k0}", flush=True)
