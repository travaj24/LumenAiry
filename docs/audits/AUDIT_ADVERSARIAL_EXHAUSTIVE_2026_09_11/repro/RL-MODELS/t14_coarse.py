"""Probe 3: the pointwise cos-grid is built on a FIXED 384-sample coarse grid
spanning the WHOLE FIELD EXTENT, then bilinearly upsampled.  So its resolution
INSIDE the pupil degrades linearly with grid padding."""
import numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements._lens_real import _build_displaced_cos_grid
lam=0.55e-6
s=[dict(radius=+19.6e-3, glass_before='AIR',   glass_after='N-BK7', decenter=(1e-9,0.0)),
   dict(radius=-27.4e-3, glass_before='N-BK7', glass_after='AIR',   decenter=(1e-9,0.0))]
r_max=1.0e-3
print(" window[mm]  coarse pitch[um]  samples across the 2mm pupil |"
      "  max|cos_out - reference| inside pupil (surface 1)")
ref=None
for N, dx in [(512, 4e-6), (512, 8e-6), (512, 16e-6), (512, 32e-6), (512, 64e-6)]:
    ax=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(ax,ax)
    g=_build_displaced_cos_grid(s,[2.5e-3],lam,r_max,N,N,dx,dx,n_coarse=384)
    gh=_build_displaced_cos_grid(s,[2.5e-3],lam,r_max,N,N,dx,dx,n_coarse=4096)
    pup=(X**2+Y**2)<=r_max**2
    win=N*dx
    pitch=win/383
    err=float(np.max(np.abs(g[1][1][pup]-gh[1][1][pup])))
    print(f" {win*1e3:9.3f}  {pitch*1e6:14.2f}  {2*r_max/pitch:26.1f} | {err:.4e}")
