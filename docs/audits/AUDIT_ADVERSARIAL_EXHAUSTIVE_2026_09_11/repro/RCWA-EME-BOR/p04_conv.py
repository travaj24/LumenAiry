import sys, numpy as np, warnings, time
sys.path.insert(0, r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/RCWA-EME-BOR")
from oracle1d import oracle_1d
from lumenairy.elements.rcwa import rcwa_efficiency_1d

print("A) oracle robustness at exact Wood point (lam=Lam), TM, vs truncation")
for M in (11,21,41,81,121):
    m,R,T,_,_=oracle_1d(1e-6,2.04,1.0,1.0,1.0,1e-6,0.5,1e-6,pol="tm",M=M)
    print(f"   M={M:4d} orcR0={R[M]:.12f} closure={R.sum()+T.sum()-1:+.2e}")

print("B) does the nudge warn?")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    rcwa_efficiency_1d(1e-6,2.04,1.0,1.0,1.0,1e-6,0.5,1e-6,polarization="tm",n_orders=21)
    print("   warnings:", [str(x.message)[:90] for x in w])

print("C) Li 1996 metallic-TM convergence: Ag lamellar, lam=0.6328um, Lam=0.5um, d=0.2um, duty=0.5, normal")
# silver at 632.8 nm: n = 0.135 + 3.99i
args=dict(period=0.5e-6,n_ridge=0.135+3.99j,n_groove=1.0,n_substrate=1.5,n_superstrate=1.0,
          depth=0.2e-6,duty_cycle=0.5,wavelength=0.6328e-6)
hdr=f"   {'N':>5s} {'R0(li)':>16s} {'R0(laurent)':>16s} {'A(li)':>12s} {'A(laurent)':>12s}"
print(hdr)
prev=None
for M in (5,11,21,41,61,81,101,151,201):
    o,Rl,Tl=rcwa_efficiency_1d(**args,polarization="tm",n_orders=M,formulation="li")
    o,Ra,Ta=rcwa_efficiency_1d(**args,polarization="tm",n_orders=M,formulation="laurent")
    print(f"   {2*M+1:5d} {Rl[M]:16.12f} {Ra[M]:16.12f} {1-Rl.sum()-Tl.sum():12.3e} {1-Ra.sum()-Ta.sum():12.3e}")
