import sys, numpy as np
sys.path.insert(0, r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/RCWA-EME-BOR")
from oracle1d import oracle_1d
from lumenairy.elements.rcwa import rcwa_efficiency_1d
np.set_printoptions(precision=12, suppress=False)
cases = [
  dict(period=1.0e-6, n_ridge=2.0, n_groove=1.0, n_sub=1.5, n_sup=1.0, depth=0.4e-6, duty=0.5, wl=0.633e-6),
  dict(period=1.0e-6, n_ridge=2.04,n_groove=1.0, n_sub=1.0, n_sup=1.0, depth=1.0e-6, duty=0.5, wl=1.0e-6),
  dict(period=1.2e-6, n_ridge=3.5, n_groove=1.45,n_sub=1.45,n_sup=1.0, depth=0.6e-6, duty=0.35,wl=0.55e-6),
  dict(period=0.8e-6, n_ridge=0.22+6.71j, n_groove=1.0, n_sub=1.5, n_sup=1.0, depth=0.3e-6, duty=0.5, wl=0.6328e-6),
]
for ci,c in enumerate(cases):
  for pol in ("te","tm"):
    for ang in (0.0, 25.0, 55.0):
      M=21
      m,Ro,To,_,_ = oracle_1d(c["period"],c["n_ridge"],c["n_groove"],c["n_sub"],c["n_sup"],
                              c["depth"],c["duty"],c["wl"],angle=np.deg2rad(ang),pol=pol,M=M)
      o,R,T = rcwa_efficiency_1d(c["period"],c["n_ridge"],c["n_groove"],c["n_sub"],c["n_sup"],
                                 c["depth"],c["duty"],c["wl"],angle=np.deg2rad(ang),
                                 polarization=pol,n_orders=M,formulation="li")
      dR=np.max(np.abs(R-Ro)); dT=np.max(np.abs(T-To))
      print(f"case{ci} pol={pol} ang={ang:4.1f}  maxdR={dR:.3e} maxdT={dT:.3e}  "
            f"lib R0={R[M]:.10f} orc R0={Ro[M]:.10f} | lib T0={T[M]:.10f} orc T0={To[M]:.10f}")
