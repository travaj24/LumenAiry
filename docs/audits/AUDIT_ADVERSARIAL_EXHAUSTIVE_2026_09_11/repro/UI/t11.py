import sys, numpy as np
from lumenairy.ui.model import SystemModel, Element, SurfaceRow
m = SystemModel()
mir = SurfaceRow(np.inf, 0.0, '', 25.0, surf_type='Mirror')
m.insert_element(1, Element(0,'M1','Mirror',distance_mm=50.0, surfaces=[mir], tilt_x=45.0))
a=[SurfaceRow(50.0,3.0,'N-BK7',12.7), SurfaceRow(np.inf,0.0,'',12.7)]
m.insert_element(2, Element(0,'L1','Singlet',distance_mm=40.0,surfaces=a, tilt_x=45.0))
ts = m.build_trace_surfaces()
print('local trace surfaces (cb?, R, thickness_m):')
for s in ts:
    print('   cb=%-5s R=%-8s t=%s' % (s.is_coordbrk, s.radius, s.thickness))
p = m.to_prescription()
print('legacy surfaces count:', len(p['surfaces']), ' thicknesses:', p['thicknesses'])
print('total gap in local list  :', sum(s.thickness for s in ts))
print('total gap in thicknesses :', sum(p['thicknesses']))
print('LOST (the cb_post air gap):', sum(s.thickness for s in ts) - sum(p["thicknesses"]))
