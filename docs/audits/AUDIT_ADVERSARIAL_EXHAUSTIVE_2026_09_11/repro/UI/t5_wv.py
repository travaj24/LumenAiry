import sys, numpy as np
sys.path.insert(0, r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/UI/stub")
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.ui.model import SystemModel, SourceDefinition

m = SystemModel()
m.elements[0].source = SourceDefinition('point_source', object_distance_mm=50.0)
print("before: model.wavelength_nm =", m.wavelength_nm,
      " source.wavelength_nm =", m.source.wavelength_nm)
m.set_wavelength(632.8)
print("after set_wavelength(632.8): model =", m.wavelength_nm,
      " source =", m.source.wavelength_nm)
s = m.source.to_source(N=64, dx_m=2e-6)
print("Source built at wavelength:", s.wavelength, " model.wavelength_m:", m.wavelength_m)
print("MISMATCH" if abs(s.wavelength - m.wavelength_m) > 1e-15 else "ok")

# Does it matter physically? point_source carries a spherical phase ~ exp(i k r^2/2z)
s2 = SourceDefinition('point_source', object_distance_mm=50.0, wavelength_nm=632.8).to_source(N=64, dx_m=2e-6)
import numpy as np
print("max |phase diff| (rad) between stale-wv and correct-wv source:",
      float(np.max(np.abs(np.angle(s.E * np.conj(s2.E))))))
print("_suppress_history ever set True?  grep result below")
