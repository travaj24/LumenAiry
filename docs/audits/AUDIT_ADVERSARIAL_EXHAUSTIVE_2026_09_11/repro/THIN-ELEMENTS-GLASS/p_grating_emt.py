import numpy as np, sys, warnings
sys.path.insert(0,r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.thin_grating import thin_grating_efficiency_1d
warnings.simplefilter('ignore')

print("=== thin_grating_efficiency_1d vs direct FFT of the transmittance ===")
wl=1.0e-6; P=20e-6; f=0.5
n_r, n_g, n_sub, n_sup = 1.5, 1.0, 1.0, 1.0
depth = wl/(2*(n_r-n_g))                     # pi phase step
orders, R, T = thin_grating_efficiency_1d(P, n_r, n_g, n_sub, n_sup, depth, f, wl, n_orders=31)
M=8192; xs=(np.arange(M)+0.5)/M*P
k0=2*np.pi/wl
t = np.where(xs < f*P, np.exp(1j*k0*(n_r-n_sub)*depth), np.exp(1j*k0*(n_g-n_sub)*depth))
tm = np.fft.fft(t)/M
def num_eff(m): return abs(tm[m % M])**2
print(f"  pi-step, 50% duty: eta_0 code={T[31]:.6f} fft={num_eff(0):.6f} ;"
      f" eta_+1 code={T[32]:.6f} fft={num_eff(1):.6f} ; 4/pi^2 = {4/np.pi**2:.6f}")
print(f"  eta_+3 code={T[34]:.6f} fft={num_eff(3):.6f} ; 4/(9 pi^2)={4/(9*np.pi**2):.6f}")
print(f"  sum T (propagating only) = {T.sum():.6f} ; sum |t_m|^2 over all kept = {sum(num_eff(m) for m in orders):.6f}")

# sinusoidal grating -> Raman-Nath Bessel check using the binary model? no: check a blazed/binary sweep
print("\n  binary duty/phase sweep: code vs fft (max abs diff over orders -5..5)")
for f_ in (0.25, 0.4, 0.5, 0.7):
    for phi in (0.5*np.pi, np.pi, 2.0*np.pi, 3.0*np.pi):
        dep = phi/(k0*(n_r-n_g))
        o,Rr,Tt = thin_grating_efficiency_1d(P, n_r, n_g, n_sub, n_sup, dep, f_, wl, n_orders=31)
        t = np.where(xs < f_*P, np.exp(1j*k0*(n_r-n_sub)*dep), np.exp(1j*k0*(n_g-n_sub)*dep))
        tm = np.fft.fft(t)/M
        err = max(abs(Tt[31+m]-abs(tm[m%M])**2) for m in range(-5,6))
        print(f"    f={f_:.2f} phi={phi/np.pi:.1f}pi   max|code-fft| = {err:.2e}   sum T={Tt.sum():.6f}")

print("\n  Q (Klein-Cook) validity guard present?  Q = 2*pi*lambda*d/(n*Lambda^2)")
for P_ in (20e-6, 2e-6, 1.0e-6):
    dep = 10e-6
    Q = 2*np.pi*wl*dep/(1.0*P_**2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        o,Rr,Tt = thin_grating_efficiency_1d(P_, n_r, n_g, n_sub, n_sup, dep, 0.5, wl, n_orders=11)
    print(f"    Lambda={P_*1e6:5.1f}um depth=10um  Q={Q:8.2f} (thin needs Q<1)  warnings={len(w)}  sumT={Tt.sum():.4f}")

print("\n=== EMT: Rytov vs RCWA ===")
try:
    from lumenairy.elements.rcwa import rcwa_efficiency_1d
    have=True
except Exception as e:
    print("  rcwa_efficiency_1d import failed:", e); have=False
from lumenairy.elements.emt import rytov_tensor
n1, n2, fill = 3.5, 1.0, 0.5
e1, e2 = n1**2, n2**2
for order in (0,2):
    for ratio in (0.05, 0.1, 0.3):
        P = ratio*wl
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            Tn = rytov_tensor(e1, e2, fill, period=P, wavelength=wl, order=order)
        n_perp = np.sqrt(Tn[0,0]).real   # E across the lamellae (x) = TM
        n_par  = np.sqrt(Tn[1,1]).real   # E along the lamellae (y) = TE
        n_te0 = np.sqrt(fill*e1+(1-fill)*e2); n_tm0 = np.sqrt(1/(fill/e1+(1-fill)/e2))
        print(f"  order={order} P/lam={ratio:4.2f}: n_par(TE)={n_par:.6f} n_perp(TM)={n_perp:.6f}"
              f"   [0th-order TE={n_te0:.6f} TM={n_tm0:.6f}]  warns={len(w)}")
