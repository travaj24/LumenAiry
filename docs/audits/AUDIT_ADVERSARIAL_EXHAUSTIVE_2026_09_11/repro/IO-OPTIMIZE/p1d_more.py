import os, sys, warnings
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
TMP = r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/IO-OPTIMIZE"
import numpy as np, lumenairy as la

# PARAXIAL surface dropped silently?
rxf2 = la.load_zemax_zmx(os.path.join(TMP,'paraxial2.zmx'))
print("PARAXIAL+glass: object_distance =", rxf2['object_distance'], " elements surf_nums =",
      [e['surf_num'] for e in rxf2['elements']], " stop_index=", rxf2['stop_index'],
      " aperture=", rxf2['aperture_diameter'])

# same but with a TOROIDAL (also powered, also air-to-air) upstream
txt = open(os.path.join(TMP,'paraxial2.zmx')).read().replace('TYPE PARAXIAL','TYPE EVENASPH')
p = os.path.join(TMP,'evenasph_upstream.zmx'); open(p,'w').write(txt)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    r = la.load_zemax_zmx(p)
print("EVENASPH air-to-air upstream: elements surf_nums =", [e['surf_num'] for e in r['elements']],
      "warnings:", [str(x.message)[:60] for x in w])

# ---- tilt convention: build world rotation for tilt_x=+90 and check local +z -> world -y
from lumenairy.raytrace import world as W
import inspect
src = inspect.getsource(W._apply_coord_break)
print("---- world._apply_coord_break ----")
print(src[:2200])
