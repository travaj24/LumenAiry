import sys, numpy as np, warnings
sys.path.insert(0, r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/UI/stub")
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la
from lumenairy.ui.model import SystemModel, Element, SurfaceRow
from lumenairy.elements.lenses import apply_real_lens

# System: flat fold mirror at 100mm then a lens.  Compare apply_real_lens on
# the UI's exported prescription vs the equivalent with is_mirror preserved.
m = SystemModel()
mir = SurfaceRow(radius=np.inf, thickness=0.0, glass='', semi_diameter=25.0, surf_type='Mirror')
m.insert_element(1, Element(0,'M1','Mirror',distance_mm=50.0, surfaces=[mir]))
sr1 = SurfaceRow(radius=50.0, thickness=3.0, glass='N-BK7', semi_diameter=12.7)
sr2 = SurfaceRow(radius=np.inf, thickness=0.0, glass='', semi_diameter=12.7)
m.insert_element(2, Element(0,'L1','Singlet',distance_mm=-40.0, surfaces=[sr1,sr2]))
pres = m.to_prescription()
print("thicknesses (legacy key):", pres['thicknesses'])
print("any is_mirror in surfaces:", any('is_mirror' in s for s in pres['surfaces']))
E = np.ones((128,128), dtype=complex)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    try:
        out = apply_real_lens(E, prescription=pres, wavelength=1.31e-6, dx=4e-6)
        print("apply_real_lens on UI prescription: OK (mirror treated as refracting air-air)")
    except Exception as e:
        print("apply_real_lens raises:", type(e).__name__, e)
