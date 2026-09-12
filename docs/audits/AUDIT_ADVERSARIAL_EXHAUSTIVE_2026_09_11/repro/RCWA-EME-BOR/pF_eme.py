import numpy as np, warnings, time
t0 = time.perf_counter()
from lumenairy.elements.eme import eme_2d as E2
from lumenairy.elements.eme import eme_2d_vector as EV
from lumenairy.elements.eme import eme_diffraction as ED
print("import %.1f s" % (time.perf_counter() - t0), flush=True)

print("\n=== E1) strip_x_modes vs ANALYTIC slab-waveguide TE0 (open->PEC box) ===",
      flush=True)
# 1-D Bloch cell of width Lx with a high-index slab of width a centred:
# for Lx >> a and kx0=0, the lowest eigenvalue lam ~ (n_eff k0)^2 of the slab
# TE0 mode.  Analytic characteristic eq: tan(kx a/2) = gamma/kx with
# kx^2 = n1^2 k0^2 - beta^2, gamma^2 = beta^2 - n2^2 k0^2.
from scipy.optimize import brentq
lam0 = 1.0
k0 = 2*np.pi/lam0
n1, n2 = 1.50, 1.45
aw = 0.8                                     # slab width


def char_even(beta):
    kx = np.sqrt(max(n1**2*k0**2 - beta**2, 0.0))
    ga = np.sqrt(max(beta**2 - n2**2*k0**2, 0.0))
    return kx*np.tan(kx*aw/2) - ga


lo, hi = n2*k0*(1+1e-12), n1*k0*(1-1e-12)
bs = np.linspace(lo, hi, 200001)
vs = np.array([char_even(b) for b in bs])
roots = [brentq(char_even, bs[i], bs[i+1], xtol=1e-16, rtol=8.9e-16)
         for i in range(len(bs)-1)
         if np.isfinite(vs[i]) and np.isfinite(vs[i+1]) and vs[i]*vs[i+1] < 0
         and abs(vs[i]) < 1e3 and abs(vs[i+1]) < 1e3]
beta0 = max(roots)
print(f"   analytic TE0 beta = {beta0:.12f}  neff = {beta0/k0:.12f}", flush=True)
Lx = 12.0
for Nx in (240, 480, 960, 1920, 3840):
    x = (np.arange(Nx)+0.5)/Nx*Lx
    eps = np.where(np.abs(x-Lx/2) < aw/2, n1**2, n2**2)
    lam, Phi = E2.strip_x_modes(eps, Lx, Nx, k0, 0.0)
    b = np.sqrt(np.max(lam.real))
    print(f"   Nx={Nx:5d}: beta_FD={b:.12f}  err={b-beta0:+.3e}  "
          f"rel={abs(b-beta0)/beta0:.3e}", flush=True)

print("\n=== E2) strip_x_modes orthonormality / Hermiticity at kx0 != 0 ===",
      flush=True)
for kx0 in (0.0, 0.37):
    Nx = 128
    x = (np.arange(Nx)+0.5)/Nx*Lx
    eps = np.where(np.abs(x-Lx/2) < aw/2, n1**2, n2**2)
    lam, Phi = E2.strip_x_modes(eps, Lx, Nx, k0, kx0)
    print(f"   kx0={kx0}: max|Phi^H Phi - I| = "
          f"{np.max(np.abs(Phi.conj().T@Phi - np.eye(Nx))):.3e}  "
          f"max|Im lam| = {np.max(np.abs(lam.imag)):.3e}", flush=True)
    epsl = eps*(1+0.001j)
    lam2, Phi2 = E2.strip_x_modes(epsl, Lx, Nx, k0, kx0)
    print(f"        lossy: max|Im lam| = {np.max(np.abs(np.imag(lam2))):.3e}",
          flush=True)

print("\n=== E3) mode_match vs ANALYTIC Airy slab (uniform layer) ===",
      flush=True)
Lxc, Lyc = 1.0, 1.0
Nx, Ny = 16, 16
k0c = 2*np.pi/0.633
for nf, depth in ((1.5, 0.2), (1.5, 2.0), (1.5, 5.3), (1.5+0.2j, 4.0)):
    eps_xy = np.full((Nx, Ny), complex(nf)**2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ED.diffraction_fd(eps_xy, Lxc, Lyc, Nx, Ny, k0c, 1.0, 1.0,
                                depth, 1, 1)
    i0 = res["orders"].index((0, 0))
    # analytic Airy for a slab in vacuum, normal incidence
    n0, ns = 1.0, 1.0
    nfc = complex(nf)
    r1 = (n0-nfc)/(n0+nfc); r2 = (nfc-ns)/(nfc+ns)
    t1 = 2*n0/(n0+nfc); t2 = 2*nfc/(nfc+ns)
    ph = np.exp(1j*k0c*nfc*depth)
    rA = (r1 + r2*ph**2)/(1 + r1*r2*ph**2)
    tA = (t1*t2*ph)/(1 + r1*r2*ph**2)
    print(f"   n={nf} d={depth}: R_fd={res['R'][i0]:.12f} R_an={abs(rA)**2:.12f}"
          f"  T_fd={res['T'][i0]:.12f} T_an={abs(tA)**2:.12f}"
          f"  energy={res['energy']:.12f}", flush=True)

print("\n=== E4) EME layer_modes vs the 2-D FD oracle (structured cell) ===",
      flush=True)
Lx2, Ly2, Nx2 = 1.0, 1.0, 16
k02 = 8.0
strips = [(np.where(np.abs((np.arange(Nx2)+0.5)/Nx2 - 0.5) < 0.25, 4.0, 1.0),
           0.5),
          (np.full(Nx2, 1.0), 0.5)]
t = time.perf_counter()
eps_xy = E2.strips_to_eps_xy(strips, Lx2, Nx2, Ly2, 32)
ref = E2.ref_2d_modes(eps_xy, Lx2, Ly2, Nx2, 32, k02)
top = ref[:8]
qs = E2.layer_modes(strips, Lx2, Nx2, Ly2, k02, (0.0, 4.0*k02**2), n_scan=400)
print(f"   ({time.perf_counter()-t:.1f}s) EME modes in (0, 4k0^2): {qs}",
      flush=True)
print(f"   FD oracle top-8 qz^2 (Ny=32): {top}", flush=True)
for q in qs:
    d = np.min(np.abs(ref - q))
    print(f"      EME qz2={q:.6f} -> nearest FD {ref[np.argmin(np.abs(ref-q))]:.6f}"
          f"  |d|={d:.3e}  rel={d/max(abs(q),1e-30):.3e}", flush=True)

print("\n=== E5) vector strip modes: uniform-strip analytic dispersion ===",
      flush=True)
Nx3 = 24
epsu = np.full(Nx3, 4.0)
for qz2 in (0.0, 9.0):
    ky, W, V = EV.strip_vector_modes(epsu, 1.0, Nx3, 8.0, 0.0, qz2)
    # analytic: ky^2 = eps k0^2 - kx^2 - qz^2 with kx = 2 pi m / Lx
    ms = np.arange(-Nx3//2, Nx3//2)
    kx = 2*np.pi*ms
    an = np.sqrt(np.asarray(4.0*64.0 - kx**2 - qz2, dtype=complex))
    an = np.where(an.imag < 0, -an, an)
    a = np.sort_complex(np.round(np.sort_complex(ky), 9))
    print(f"   qz2={qz2}: n_fwd={len(ky)} "
          f"first few |ky| = {np.sort(np.abs(ky))[:6]}", flush=True)
    print(f"      analytic |ky| (first few) = {np.sort(np.abs(an))[:6]}",
          flush=True)
