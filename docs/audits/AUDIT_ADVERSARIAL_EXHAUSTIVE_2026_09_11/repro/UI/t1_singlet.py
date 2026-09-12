import sys, os, numpy as np
sys.path.insert(0, r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/UI/stub")
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la
from lumenairy.ui.model import SystemModel, Element, SurfaceRow
from lumenairy.raytrace import surfaces_from_prescription, system_abcd

m = SystemModel()
# a BK7 singlet: R1=50mm, R2=inf, d=3mm, CA dia 25.4mm
sr1 = SurfaceRow(radius=50.0, thickness=3.0, glass='N-BK7', semi_diameter=12.7)
sr2 = SurfaceRow(radius=np.inf, thickness=0.0, glass='', semi_diameter=12.7)
e = Element(0, 'Lens 1', 'Singlet', distance_mm=10.0, surfaces=[sr1, sr2])
m.insert_element(1, e)
m.epd_mm = 25.4
m.wavelength_nm = 1310.0
m._invalidate()

pres_ui = m.to_prescription()
import pprint
print("=== UI prescription ===")
pprint.pprint(pres_ui)

pres_lib = la.make_singlet(R1=50e-3, R2=float('inf'), d=3e-3, glass='N-BK7', aperture=25.4e-3)
print("\n=== lib make_singlet ===")
pprint.pprint(pres_lib)

s_ui = surfaces_from_prescription(pres_ui)
s_lib = surfaces_from_prescription(pres_lib)
print("\nn surf ui/lib:", len(s_ui), len(s_lib))
for a,b in zip(s_ui, s_lib):
    print(f"  ui R={a.radius!r} t={a.thickness!r} sd={a.semi_diameter!r} gb={a.glass_before} ga={a.glass_after} mir={a.is_mirror}")
    print(f"  lib R={b.radius!r} t={b.thickness!r} sd={b.semi_diameter!r} gb={b.glass_before} ga={b.glass_after} mir={b.is_mirror}")

abcd_u, efl_u, bfl_u, ffl_u = system_abcd(s_ui, 1310e-9)
abcd_l, efl_l, bfl_l, ffl_l = system_abcd(s_lib, 1310e-9)
print("\nEFL ui  =", efl_u, " BFL=", bfl_u)
print("EFL lib =", efl_l, " BFL=", bfl_l)
print("model.efl_mm =", m.efl_mm, "model.bfl_mm =", m.bfl_mm)
