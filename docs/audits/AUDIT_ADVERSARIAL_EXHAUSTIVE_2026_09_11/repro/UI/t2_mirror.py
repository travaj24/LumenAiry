import sys, numpy as np
sys.path.insert(0, r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/UI/stub")
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la
from lumenairy.ui.model import SystemModel, Element, SurfaceRow
from lumenairy.raytrace import surfaces_from_prescription, system_abcd

m = SystemModel()
# Concave mirror R=-200 mm at 50mm
mir = SurfaceRow(radius=-200.0, thickness=0.0, glass='', semi_diameter=25.0,
                 surf_type='Mirror')
em = Element(0, 'M1', 'Mirror', distance_mm=50.0, surfaces=[mir])
m.insert_element(1, em)
# add a lens AFTER the mirror to check semi-diameter index alignment
sr1 = SurfaceRow(radius=80.0, thickness=4.0, glass='N-BK7', semi_diameter=6.0)
sr2 = SurfaceRow(radius=np.inf, thickness=0.0, glass='', semi_diameter=6.0)
m.insert_element(2, Element(0,'Lens 2','Singlet', distance_mm=-30.0, surfaces=[sr1,sr2]))
pres = m.to_prescription()
print("legacy surfaces:")
for s in pres['surfaces']:
    print("  ", {k:v for k,v in s.items() if k in ('radius','glass_before','glass_after')},
          "is_mirror key present:", 'is_mirror' in s, " semi_diameter key present:", 'semi_diameter' in s)
print("thicknesses:", pres['thicknesses'])
print("elements types:", [e['element_type'] for e in pres['elements']])
print("elements semi_diameter:", [e.get('semi_diameter') for e in pres['elements']])

surfs = surfaces_from_prescription(pres)
print("\nsurfaces_from_prescription -> ")
for s in surfs:
    print(f"   R={s.radius} is_mirror={s.is_mirror} sd={s.semi_diameter} t={s.thickness} gb={s.glass_before} ga={s.glass_after}")

print("\nUI internal trace surfaces (build_trace_surfaces):")
for s in m.build_trace_surfaces():
    print(f"   R={s.radius} is_mirror={s.is_mirror} sd={s.semi_diameter} t={s.thickness} cb={s.is_coordbrk}")

a,e,b,f = system_abcd(surfs, 1310e-9)
print("\nABCD from EXPORTED prescription: efl=",e," bfl=",b)
print("ABCD from UI internal surfaces  : efl=",m.efl_mm*1e-3," bfl=",m.bfl_mm*1e-3)
