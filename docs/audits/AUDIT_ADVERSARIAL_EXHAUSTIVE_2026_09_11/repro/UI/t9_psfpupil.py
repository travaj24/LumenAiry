import sys, numpy as np
sys.path.insert(0, r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/UI/stub")
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import numpy as np
from lumenairy.ui.model import SystemModel, Element, SurfaceRow
m = SystemModel()
sr1 = SurfaceRow(radius=50.0, thickness=3.0, glass='N-BK7', semi_diameter=12.7)
sr2 = SurfaceRow(radius=np.inf, thickness=0.0, glass='', semi_diameter=12.7)
m.insert_element(1, Element(0,'L1','Singlet',distance_mm=10.0, surfaces=[sr1,sr2]))
m.elements[-1].distance_mm = 0.0   # let it use paraxial BFL
res = m.run_trace(num_rings=8, rays_per_ring=36)
r = res.image_rays
print("image_rays attrs:", [a for a in dir(r) if not a.startswith('_')])
print("has .opl:", hasattr(r,'opl'), " has .opd:", hasattr(r,'opd'))
alive = r.alive
x = r.x[alive]; y = r.y[alive]
print("n alive:", alive.sum())
print("image-plane ray radius: max |r| = %.6e m  (EPD/2 = %.6e m)" %
      (float(np.max(np.hypot(x,y))), m.epd_m/2))
N=256; ap=m.epd_m; dx=ap/N
ix = ((x/dx)+N/2).astype(int); iy=((y/dx)+N/2).astype(int)
inb = (ix>=0)&(ix<N)&(iy>=0)&(iy<N)
cnt = np.zeros((N,N),int); np.add.at(cnt,(iy[inb],ix[inb]),1)
print("distinct pupil pixels filled by psf_mtf_dock._load_from_raytrace:", int((cnt>0).sum()), "of", N*N)
