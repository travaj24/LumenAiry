import os, sys, warnings
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
TMP = r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/IO-OPTIMIZE"
import numpy as np, lumenairy as la
from lumenairy.raytrace.seidel import system_abcd_prescription

def W(fn,*a,**k):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always'); r=fn(*a,**k)
    return r,[str(x.message)[:120] for x in w]

# A genuine-style CODE V .seq for the same doublet the .zmx test used.
# CODE V lens units: DIM M = MILLIMETRES (vendor), C = cm, I = inches.
SEQ = """! AC254-100 doublet, CODE V sequence file
LEN NEW
RDM
DIM M
WL 587.6
REF 1
SO
  RDY INFINITY
  THI INFINITY
S1
  STO
  RDY 62.75
  THI 4.0
  GLA N-BAF10
S2
  RDY -45.71
  THI 2.5
  GLA N-SF6HT
S3
  RDY -128.23
  THI 95.0
SI
  RDY INFINITY
  THI 0.0
GO
END
"""
p = os.path.join(TMP,'doublet_dimM.seq'); open(p,'w').write(SEQ)
rx, ws = W(la.load_codev_seq, p)
print("=== DIM M (CODE V = millimetres) ===")
print("  radii as loaded [m]:", [s['radius'] for s in rx['surfaces']])
print("  -> radii in mm:", [f"{s['radius']*1e3:.2f}" for s in rx['surfaces']])
print("  thicknesses[m]:", rx['thicknesses'], " aperture:", rx['aperture_diameter'])
print("  wavelength:", rx.get('wavelength'), " stop:", rx.get('stop_index'), " bfl:", rx.get('back_focal_length'))
print("  warnings:", ws)
_,efl,_,_ = system_abcd_prescription(rx, 587.6e-9)
print(f"  EFL = {efl:.4f} m   (should be ~0.0722 m if DIM M were read as mm)")

for dim, expect in (('C', 1e-2), ('I', 0.0254), ('MM', 1e-3), ('IN', 0.0254)):
    q = os.path.join(TMP, f'doublet_dim{dim}.seq'); open(q,'w').write(SEQ.replace('DIM M\n', f'DIM {dim}\n'))
    r,_ = W(la.load_codev_seq, q)
    print(f"  DIM {dim:<3s}: R1 loaded = {r['surfaces'][0]['radius']:.6g} m   (CODE V means {62.75*expect:.6g} m)")

# --- surface with no RDY: radius comes back None ---
SEQ2 = """LEN NEW
DIM M
WL 587.6
S1
  RDY 62.75
  THI 4.0
  GLA N-BK7
S2
  THI 10.0
S3
  RDY -50.0
  THI 90.0
SI
  RDY INFINITY
  THI 0.0
END
"""
q = os.path.join(TMP,'no_rdy.seq'); open(q,'w').write(SEQ2)
r2,_ = W(la.load_codev_seq, q)
print("=== surface without RDY: radii =", [s['radius'] for s in r2['surfaces']])
try:
    system_abcd_prescription(r2, 587.6e-9)
    print("    system_abcd: OK")
except Exception as e:
    print("    system_abcd RAISES:", type(e).__name__, str(e)[:130])

# --- K conic + A/B/C/D aspheres + REFL mirror: silently dropped? ---
SEQ3 = """LEN NEW
DIM M
WL 587.6
S1
  STO
  RDY 62.75
  THI 4.0
  GLA N-BK7
  ASP
  K -1.0
  A 1.234E-07
  B -5.0E-11
S2
  RDY -50.0
  THI 30.0
  REFL
S3
  RDY INFINITY
  THI 50.0
  RMD REFL
SI
  RDY INFINITY
  THI 0.0
END
"""
q = os.path.join(TMP,'asph_refl.seq'); open(q,'w').write(SEQ3)
r3,w3 = W(la.load_codev_seq, q)
print("=== K/A/B/REFL/RMD ===")
print("  conics:", [s['conic'] for s in r3['surfaces']], " asph:", [s['aspheric_coeffs'] for s in r3['surfaces']])
print("  glasses:", [(s['glass_before'], s['glass_after']) for s in r3['surfaces']])
print("  has elements key:", 'elements' in r3, " has_mirrors:", la.has_mirrors(r3))
print("  warnings:", w3)

# --- cross-format diff: same doublet .zmx vs .seq ---
rz = la.load_zemax_zmx(os.path.join(TMP,'ac254_100_a.zmx'))
q = os.path.join(TMP,'doublet_dimMM.seq')
rc,_ = W(la.load_codev_seq, q)
print("=== .zmx vs .seq (DIM MM so the scale matches) ===")
print("  zmx keys:", sorted(rz.keys()))
print("  seq keys:", sorted(rc.keys()))
for i,(a,b) in enumerate(zip(rz['surfaces'], rc['surfaces'])):
    print(f"   S{i}: dR={a['radius']-b['radius']:.3e}  zmx_keys-seq_keys="
          f"{sorted(set(a)-set(b))} seq-zmx={sorted(set(b)-set(a))}")
print("  thickness diff:", [a-b for a,b in zip(rz['thicknesses'], rc['thicknesses'])])
print("  aperture zmx/seq:", rz['aperture_diameter'], rc['aperture_diameter'])
