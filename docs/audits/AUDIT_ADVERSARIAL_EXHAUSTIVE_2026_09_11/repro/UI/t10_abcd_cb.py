import sys, numpy as np
sys.path.insert(0, r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/UI/stub")
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import numpy as np
from lumenairy.ui.model import SystemModel, Element, SurfaceRow
from lumenairy.raytrace import system_abcd, find_paraxial_focus

def two_lens(tilt=0.0):
    m = SystemModel()
    a=[SurfaceRow(50.0,3.0,'N-BK7',12.7), SurfaceRow(np.inf,0.0,'',12.7)]
    b=[SurfaceRow(80.0,3.0,'N-BK7',12.7), SurfaceRow(np.inf,0.0,'',12.7)]
    m.insert_element(1, Element(0,'L1','Singlet',distance_mm=10.0,surfaces=a))
    m.insert_element(2, Element(0,'L2','Singlet',distance_mm=40.0,surfaces=b, tilt_x=tilt))
    return m

print("=== (A) coord-break Surfaces inside system_abcd ===")
m0 = two_lens(0.0); m1 = two_lens(1e-6)  # infinitesimal tilt -> physically identical
print("untilted  EFL/BFL mm:", round(m0.efl_mm,6), round(m0.bfl_mm,6))
print("tilt 1e-6 EFL/BFL mm:", round(m1.efl_mm,6), round(m1.bfl_mm,6))
print("n surfaces in local list: untilted", len(m0.build_trace_surfaces()),
      " tilted", len(m1.build_trace_surfaces()))
for s in m1.build_trace_surfaces():
    print("   cb=%s R=%s t=%s" % (s.is_coordbrk, s.radius, s.thickness))

print()
print("=== (B) world surface list drops inter-element air gaps ===")
wl = m0.build_trace_surfaces_world()
print("world thicknesses (m):", [s.thickness for s in wl])
print("local thicknesses (m):", [s.thickness for s in m0.build_trace_surfaces()])
print("find_paraxial_focus(world) =", find_paraxial_focus(wl, m0.wavelength_m))
print("find_paraxial_focus(local) =", find_paraxial_focus(m0.build_trace_surfaces(), m0.wavelength_m))
_,e,b,_ = system_abcd(wl, m0.wavelength_m); print("system_abcd(world) efl,bfl:", e, b)
_,e2,b2,_ = system_abcd(m0.build_trace_surfaces(), m0.wavelength_m); print("system_abcd(local) efl,bfl:", e2, b2)
