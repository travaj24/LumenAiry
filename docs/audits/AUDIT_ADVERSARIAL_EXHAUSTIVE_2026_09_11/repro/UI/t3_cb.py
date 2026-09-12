import sys, numpy as np
sys.path.insert(0, r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/UI/stub")
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la
from lumenairy.ui.model import SystemModel, Element, SurfaceRow
from lumenairy.raytrace.world import world_surfaces_from_prescription, _rot_x, _rot_y

np.set_printoptions(precision=6, suppress=True)

m = SystemModel()
sr1 = SurfaceRow(radius=50.0, thickness=3.0, glass='N-BK7', semi_diameter=12.7)
sr2 = SurfaceRow(radius=np.inf, thickness=0.0, glass='', semi_diameter=12.7)
e = Element(0, 'L1', 'Singlet', distance_mm=10.0, surfaces=[sr1, sr2],
            tilt_x=30.0, tilt_y=0.0, decenter_x=2.0, decenter_y=1.0)
m.insert_element(1, e)

print("CONVENTIONS: +90 tilt_x puts new local +z at world -y")
Rt = _rot_x(np.radians(90.0))
print("  world._rot_x(90)[:,2] =", Rt[:,2])

m.recompute_element_frames()
print("\nUI element frames:")
for el in m.elements:
    print(f"  {el.name:10s} origin={el.origin}  z_axis={el.R[:,2]}")

# +90 test on the UI side
m2 = SystemModel()
e2 = Element(0, 'L1', 'Singlet', distance_mm=10.0,
             surfaces=[SurfaceRow(radius=np.inf, thickness=0.0)], tilt_x=90.0)
m2.insert_element(1, e2)
print("  UI tilt_x=+90 -> z_axis =", m2.elements[1].R[:,2])

# Compare UI world frames vs library world_surfaces_from_prescription on
# the UI's own exported prescription.
pres = m.to_prescription()
print("\nexported coord_breaks:", pres['coord_breaks'])
print("exported all_thicknesses:", pres['all_thicknesses'])
ws_lib = world_surfaces_from_prescription(pres)
ws_ui = m.build_trace_surfaces_world()
print("\nn world surfaces lib/ui:", len(ws_lib), len(ws_ui))
for i,(a,b) in enumerate(zip(ws_lib, ws_ui)):
    print(f"  S{i}: lib origin={a.world_origin*1e3} ui origin={b.world_origin*1e3}")
    print(f"       lib z={a.world_R[:,2]}  ui z={b.world_R[:,2]}")
