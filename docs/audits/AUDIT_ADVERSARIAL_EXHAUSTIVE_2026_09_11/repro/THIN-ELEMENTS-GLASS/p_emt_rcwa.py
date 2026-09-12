import numpy as np, sys, warnings
sys.path.insert(0,r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
warnings.simplefilter('ignore')
from lumenairy.elements.rcwa import rcwa_jones_1d
from lumenairy.elements.emt import rytov_tensor

wl=1.0e-6; n1,n2,fill = 2.0, 1.0, 0.5
e1,e2 = n1**2, n2**2
I3=np.eye(3)

def neff_from_rcwa(P, pol, d1=0.5e-6, d2=1.0e-6, n_orders=41):
    """Effective index from the 0th-order transmitted phase slope between two depths."""
    phases=[]
    for d in (d1,d2):
        out = rcwa_jones_1d(P, e1*I3, e2*I3, 1.0, 1.0, d, fill, wl,
                            n_orders=n_orders, return_jones_transmission=True)
        J = out[-1]
        J = np.asarray(J)
        # Jones (2x2): index 0 = s/TE, 1 = p/TM  (library convention)
        t = J[0,0] if pol=='te' else J[1,1]
        phases.append(np.angle(t))
    dphi = np.unwrap(np.array([0.0, phases[1]-phases[0]]))[1]
    # unwrap manually: true dphi = k0*(n_eff-1)*(d2-d1) mod 2pi ; use a guess to pick branch
    k0=2*np.pi/wl
    best=None
    for m in range(-4,5):
        cand = (dphi + 2*np.pi*m)/(k0*(d2-d1)) + 1.0
        if 0.9 <= cand <= 2.5:
            if best is None or abs(cand-1.5)<abs(best-1.5): best=cand
    return best

Tn0 = rytov_tensor(e1, e2, fill, order=0)
n_te0 = np.sqrt(Tn0[1,1]).real     # arithmetic (E along grooves)
n_tm0 = np.sqrt(Tn0[0,0]).real     # harmonic  (E across grooves)
print(f"Rytov 0th order: n_TE(par,arith)={n_te0:.6f}  n_TM(perp,harm)={n_tm0:.6f}")
for ratio in (0.05, 0.1, 0.3):
    P = ratio*wl
    Tn2 = rytov_tensor(e1, e2, fill, period=P, wavelength=wl, order=2)
    n_te2 = np.sqrt(Tn2[1,1]).real; n_tm2 = np.sqrt(Tn2[0,0]).real
    r_te = neff_from_rcwa(P,'te'); r_tm = neff_from_rcwa(P,'tm')
    print(f"  P/lam={ratio:4.2f}  RCWA n_eff: TE={r_te}  TM={r_tm}")
    if r_te: print(f"            Rytov0 TE={n_te0:.6f} (d={r_te-n_te0:+.2e})   Rytov2 TE={n_te2:.6f} (d={r_te-n_te2:+.2e})")
    if r_tm: print(f"            Rytov0 TM={n_tm0:.6f} (d={r_tm-n_tm0:+.2e})   Rytov2 TM={n_tm2:.6f} (d={r_tm-n_tm2:+.2e})")
