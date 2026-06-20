import sys; sys.path.insert(0,".")
from fiber_oracle import fiber_modes

m,a,k0=1,1.0,2.0; eps1,eps2=1.5**2,1.0**2
qs=fiber_modes(m,a,eps1,eps2,k0)
print("EXACT fiber oracle q (m=1):", [f"{q:.6f}" for q in qs])
