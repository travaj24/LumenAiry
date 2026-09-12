import sys
sys.path.insert(0, r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/UI/stub")
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.ui.model import SystemModel, SourceDefinition
m = SystemModel()
print("SystemModel.source is property:", isinstance(type(m).source, property))
print("has setter:", type(m).source.fset)
try:
    m.source = SourceDefinition('gaussian', wavelength_nm=1310.0)
    print("assignment SUCCEEDED (unexpected)")
except Exception as e:
    print("assignment FAILS:", type(e).__name__, e)
