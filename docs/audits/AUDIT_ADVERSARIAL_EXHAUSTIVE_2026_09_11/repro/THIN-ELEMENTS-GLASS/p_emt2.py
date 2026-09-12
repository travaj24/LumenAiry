import numpy as np, sys, warnings
sys.path.insert(0,r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
warnings.simplefilter('ignore')
from lumenairy.elements.rcwa import rcwa_efficiency_1d
from lumenairy.elements.emt import rytov_tensor

wl=1.0e-6; n1,n2,fill=2.0,1.0,0.5; e1,e2=n1**2,n2**2
d=0.8e-6                       # slab thickness

def slab_T(n, d, wl, n_in=1.0, n_out=1.0):
    """Exact single-layer transmittance |t|^2 * (n_out/n_in) at normal incidence."""
    k0=2*np.pi/wl; b=k0*n*d
    M=np.array([[np.cos(b), 1j*np.sin(b)/n],[1j*n*np.sin(b), np.cos(b)]], dtype=complex)
    t=2*n_in/((M[0,0]+M[0,1]*n_out)*n_in + (M[1,0]+M[1,1]*n_out))
    return float(abs(t)**2*(n_out/n_in))

Tn0=rytov_tensor(e1,e2,fill,order=0)
n_par=np.sqrt(Tn0[1,1]).real   # arithmetic, E ALONG grooves
n_perp=np.sqrt(Tn0[0,0]).real  # harmonic,  E ACROSS grooves
print(f"Rytov0: n(arith, E||grooves)={n_par:.6f}   n(harm, E across)={n_perp:.6f}")
print(f"slab T: arith={slab_T(n_par,d,wl):.6f}  harm={slab_T(n_perp,d,wl):.6f}\n")
print(f"{'P/lam':>6s} {'pol':>4s} {'T0_RCWA':>10s} {'T_slab(arith)':>14s} {'T_slab(harm)':>13s}  -> which EMT branch")
for ratio in (0.02, 0.05, 0.1, 0.3):
    P=ratio*wl
    for pol in ('te','tm'):
        o,R,T = rcwa_efficiency_1d(P, n1, n2, 1.0, 1.0, d, fill, wl,
                                   polarization=pol, n_orders=25)
        T0=float(T[np.argmin(np.abs(o))])
        ta=slab_T(n_par,d,wl); th=slab_T(n_perp,d,wl)
        which = 'ARITH(par)' if abs(T0-ta)<abs(T0-th) else 'HARM(perp)'
        print(f"{ratio:6.2f} {pol:>4s} {T0:10.6f} {ta:14.6f} {th:13.6f}  -> {which}"
              f"  |dT|_best={min(abs(T0-ta),abs(T0-th)):.2e}")
# order-2 improvement at P/lam=0.3
P=0.3*wl
Tn2=rytov_tensor(e1,e2,fill,period=P,wavelength=wl,order=2)
print(f"\norder-2 at P/lam=0.3: n_par={np.sqrt(Tn2[1,1]).real:.6f} n_perp={np.sqrt(Tn2[0,0]).real:.6f}")
for pol,neff0,neff2 in (('te',n_par,np.sqrt(Tn2[1,1]).real),('tm',n_perp,np.sqrt(Tn2[0,0]).real)):
    o,R,T=rcwa_efficiency_1d(P,n1,n2,1.0,1.0,d,fill,wl,polarization=pol,n_orders=25)
    T0=float(T[np.argmin(np.abs(o))])
    print(f"  {pol}: RCWA T0={T0:.6f}  slab(order0)={slab_T(neff0,d,wl):.6f} (d={T0-slab_T(neff0,d,wl):+.2e})"
          f"  slab(order2)={slab_T(neff2,d,wl):.6f} (d={T0-slab_T(neff2,d,wl):+.2e})")
