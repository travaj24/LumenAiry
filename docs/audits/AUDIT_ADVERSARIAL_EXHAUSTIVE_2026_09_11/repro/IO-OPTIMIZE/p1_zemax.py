import os, sys, warnings, json
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
TMP = r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/IO-OPTIMIZE"
import numpy as np
import lumenairy as la
from lumenairy.raytrace.seidel import system_abcd_prescription

# ---- (a) AC254-100-A doublet, UNIT MM, STOP on surf 1, DIAM semi-dia, WAVM um
ZMX_A = """VERS 210000 0 123 0 0
MODE SEQ
NAME AC254-100-A
UNIT MM X W X CM MR CPMM
ENPD 25.400000
GCAT SCHOTT MISC
WAVM 1 0.587560 1.0
PWAV 1
SURF 0
  TYPE STANDARD
  CURV 0 0 0 0 0 ""
  DISZ INFINITY
  DIAM 0 0 0 0 1 ""
SURF 1
  STOP
  TYPE STANDARD
  CURV 0.01593625498 0 0 0 0 ""
  DISZ 4.0
  GLAS N-BAF10 0 0 1.67003 47.11 0 0 0 0 0 0
  DIAM 12.7 1 0 0 1 ""
SURF 2
  TYPE STANDARD
  CURV -0.02187705097 0 0 0 0 ""
  DISZ 2.5
  GLAS N-SF6HT 0 0 1.80518 25.36 0 0 0 0 0 0
  DIAM 12.7 1 0 0 1 ""
SURF 3
  TYPE STANDARD
  CURV -0.0077984091 0 0 0 0 ""
  DISZ 95.0
  DIAM 12.7 1 0 0 1 ""
SURF 4
  TYPE STANDARD
  CURV 0 0 0 0 0 ""
  DISZ 0.0
  DIAM 0 0 0 0 1 ""
BLNK
"""
p = os.path.join(TMP, 'ac254_100_a.zmx')
open(p, 'w', encoding='utf-8').write(ZMX_A)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    rx = la.load_zemax_zmx(p)
    ws = [str(x.message)[:90] for x in w]
print("=== (a) AC254-100-A ===")
print(" radii(mm):", [f"{s['radius']*1e3:.4f}" for s in rx['surfaces']])
print(" thick(mm):", [f"{t*1e3:.4f}" for t in rx['thicknesses']])
print(" glasses:", [(s['glass_before'], s['glass_after']) for s in rx['surfaces']])
print(" aperture(mm):", rx['aperture_diameter']*1e3, " stop_index:", rx['stop_index'])
print(" object_distance:", rx['object_distance'])
print(" warnings:", ws)
abcd, efl, bfl, ffl = system_abcd_prescription(rx, 587.6e-9)
print(f" EFL = {efl*1e3:.4f} mm  (vendor 100.1)   BFL={bfl*1e3:.4f} FFL={ffl*1e3:.4f}")

# ---- (h) UTF-16 LE with BOM
p16 = os.path.join(TMP, 'ac254_utf16.zmx')
with open(p16, 'wb') as f:
    f.write(b'\xff\xfe' + ZMX_A.replace('\n', '\r\n').encode('utf-16-le'))
try:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        rx16 = la.load_zemax_zmx(p16)
    same = [abs(a['radius']-b['radius']) for a, b in zip(rx['surfaces'], rx16['surfaces'])]
    print("=== (h) UTF-16LE+BOM: OK, radius diffs:", same, " ap:", rx16['aperture_diameter'])
except Exception as e:
    print("=== (h) UTF-16LE+BOM FAILED:", type(e).__name__, e)

# UTF-16 BE with BOM (rarer)
pbe = os.path.join(TMP, 'ac254_utf16be.zmx')
with open(pbe, 'wb') as f:
    f.write(b'\xfe\xff' + ZMX_A.encode('utf-16-be'))
try:
    rxbe = la.load_zemax_zmx(pbe)
    print("=== UTF-16BE: OK")
except Exception as e:
    print("=== UTF-16BE:", type(e).__name__, str(e)[:110])

# ---- (e) UNIT IN and CM
for unit, sc in (('IN', 25.4e-3), ('CM', 1e-2), ('M', 1.0), ('BOGUS', 1e-3)):
    txt = ZMX_A.replace('UNIT MM X W X CM MR CPMM', f'UNIT {unit} X W X CM MR CPMM')
    pu = os.path.join(TMP, f'unit_{unit}.zmx')
    open(pu, 'w', encoding='utf-8').write(txt)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r = la.load_zemax_zmx(pu)
    print(f"=== (e) UNIT {unit}: R1 = {r['surfaces'][0]['radius']:.6g} m "
          f"(expect {62.75e-3*sc/1e-3:.6g}) warn={[str(x.message)[:50] for x in w]}")

# ---- (b) EVENASPH PARM mapping + unit scaling
ZMX_B = """VERS 210000 0 123 0 0
MODE SEQ
UNIT MM X W X CM MR CPMM
SURF 0
  TYPE STANDARD
  CURV 0
  DISZ INFINITY
SURF 1
  TYPE EVENASPH
  STOP
  CURV 0.02
  DISZ 5.0
  GLAS N-BK7 0 0 1.5168 64.17 0 0 0 0 0 0
  PARM 1 1.0e-3
  PARM 2 -1.234e-6
  PARM 3 5.678e-10
  PARM 8 1.0e-20
  DIAM 12.7
SURF 2
  TYPE STANDARD
  CURV 0
  DISZ 50.0
  DIAM 12.7
SURF 3
  TYPE STANDARD
  CURV 0
  DISZ 0
  DIAM 12.7
"""
pb = os.path.join(TMP, 'evenasph.zmx')
open(pb, 'w', encoding='utf-8').write(ZMX_B)
rxb = la.load_zemax_zmx(pb)
ac = rxb['surfaces'][0]['aspheric_coeffs']
print("=== (b) EVENASPH coeffs (power -> value in m^(1-p)):", ac)
print("   expected: {2: 1.0e-3/1e-3=1.0, 4: -1.234e-6*1e9=-1234.0, 6: 5.678e-10*1e15=5.678e5, 16: 1e-20*1e45=1e25}")
# sag check at r = 5 mm
r_m = 5e-3
sag_lib = sum(v*r_m**p for p, v in ac.items())
r_mm = 5.0
sag_zmx_mm = 1.0e-3*r_mm**2 + -1.234e-6*r_mm**4 + 5.678e-10*r_mm**6 + 1.0e-20*r_mm**16
print(f"   sag(lib)={sag_lib*1e3:.9f} mm   sag(zemax-native)={sag_zmx_mm:.9f} mm")
