import sys
sys.path.insert(0, r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/UI/stub")
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.ui.model import SourceDefinition
# emulate element_table._apply_source_params: every field cast to float
kwargs = {'emitter_pitch_mm':0.05, 'emitter_nx':float(12), 'emitter_ny':float(12),
          'emitter_waist_mm':0.009, 'field_angle_x_deg':0.0, 'field_angle_y_deg':0.0}
src = SourceDefinition('emitter_array', **kwargs)
print("emitter_nx type:", type(src.emitter_nx).__name__, src.emitter_nx)
try:
    s = src.to_source(N=64, dx_m=2e-6)
    print("to_source OK", s.E.shape)
except Exception as e:
    print("to_source RAISES:", type(e).__name__, e)
# also confirm wavelength default reset
print("wavelength_nm after edit-rebuild:", src.wavelength_nm)
print("polarization after edit-rebuild:", src.polarization)
