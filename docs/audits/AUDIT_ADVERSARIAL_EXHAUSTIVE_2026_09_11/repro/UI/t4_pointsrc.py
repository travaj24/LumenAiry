import sys, numpy as np
sys.path.insert(0, r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/UI/stub")
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import numpy as np
from lumenairy.ui.model import SystemModel, Element, SurfaceRow, SourceDefinition

m = SystemModel()
sr1 = SurfaceRow(radius=50.0, thickness=3.0, glass='N-BK7', semi_diameter=12.7)
sr2 = SurfaceRow(radius=np.inf, thickness=0.0, glass='', semi_diameter=12.7)
m.insert_element(1, Element(0,'L1','Singlet',distance_mm=200.0, surfaces=[sr1,sr2]))
m.elements[0].source = SourceDefinition('point_source', object_distance_mm=200.0)
m.elements[-1].distance_mm = 200.0
res = m.run_trace(num_rings=3, rays_per_ring=8)
x = res.x if hasattr(res,'x') else None
print("trace result type:", type(res).__name__)
import inspect
print("fields:", [f for f in dir(res) if not f.startswith('_')][:40])
# Inspect the launched bundle directly by monkeypatching
import lumenairy.raytrace as rt
captured = {}
orig = rt._make_bundle
def spy(x,y,L,M,wv):
    captured['x']=np.array(x); captured['y']=np.array(y)
    captured['L']=np.array(L); captured['M']=np.array(M)
    return orig(x,y,L,M,wv)
import lumenairy.ui.model as UM
# model does `from ..raytrace import _make_bundle` INSIDE the function
rt._make_bundle = spy
m.run_trace(num_rings=3, rays_per_ring=8)
L=captured['L']; M=captured['M']
print("\nn rays:", len(L))
print("unique (L,M) pairs:", len(set(zip(np.round(L,12), np.round(M,12)))))
print("L =", np.round(L,6))
print("M =", np.round(M,6))
print("L==M everywhere:", np.allclose(L, M))
