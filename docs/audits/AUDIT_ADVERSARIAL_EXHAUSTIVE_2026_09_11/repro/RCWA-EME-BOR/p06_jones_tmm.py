import sys, numpy as np, warnings
sys.path.insert(0, r'docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/RCWA-EME-BOR')
from tmm import tmm_jones
from lumenairy.elements.rcwa import rcwa_jones_1d, RCWAStack
np.set_printoptions(precision=10, suppress=False, linewidth=150)

wl=0.633e-6; P=0.4e-6   # sub-wavelength period, zeroth order only
def fmt(z): return f"{abs(z):.10f} < {np.angle(z):+.9f}"

print("=== A) rcwa_jones_1d on an UNPATTERNED film (ridge==groove) vs TMM ===")
for nf,nsub,d in ((2.0,1.5,0.25e-6), (1.45,1.5,0.4e-6), (2.0+0.05j,1.5,0.25e-6)):
  for ang in (0.0, 45.0):
    E=np.eye(3)
    out=rcwa_jones_1d(P, (nf**2)*E, (nf**2)*E, nsub, 1.0, d, 0.5, wl,
                      angle=np.deg2rad(ang), n_orders=5, return_jones_transmission=True)
    o,Rq,Tq,Jr,Jt = out
    t=tmm_jones([1.0,nf,nsub],[d],wl,np.deg2rad(ang))
    print(f" nf={nf} nsub={nsub} ang={ang}")
    print(f"   RCWA Jr = [[{fmt(Jr[0,0])}, {fmt(Jr[0,1])}],[{fmt(Jr[1,0])}, {fmt(Jr[1,1])}]]")
    print(f"   TMM  rxx= {fmt(t['rxx'])}  (-rxx = {fmt(-t['rxx'])})   ryy= {fmt(t['ryy'])}")
    print(f"   RCWA Jt = [[{fmt(Jt[0,0])}, {fmt(Jt[0,1])}],[{fmt(Jt[1,0])}, {fmt(Jt[1,1])}]]")
    print(f"   TMM  txx= {fmt(t['txx'])}  (-txx = {fmt(-t['txx'])})   tyy= {fmt(t['tyy'])}")
    print(f"   |dJr_xx|={abs(Jr[0,0]-t['rxx']):.3e} / flipped {abs(Jr[0,0]+t['rxx']):.3e} ;"
          f" |dJr_yy|={abs(Jr[1,1]-t['ryy']):.3e}")
    print(f"   |dJt_xx|={abs(Jt[0,0]-t['txx']):.3e} / flipped {abs(Jt[0,0]+t['txx']):.3e} ;"
          f" |dJt_yy|={abs(Jt[1,1]-t['tyy']):.3e}")
