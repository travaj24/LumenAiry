import os, sys, warnings
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
TMP = r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/IO-OPTIMIZE"
import numpy as np, lumenairy as la

def W(fn, *a, **k):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r = fn(*a, **k)
    return r, [str(x.message)[:130] for x in w]

# ---- (c) COORDBRK + MIRROR ----
ZMX_C = """VERS 210000 0 123 0 0
MODE SEQ
UNIT MM X W X CM MR CPMM
SURF 0
  TYPE STANDARD
  CURV 0
  DISZ INFINITY
SURF 1
  TYPE STANDARD
  STOP
  CURV 0.01
  DISZ 5.0
  GLAS N-BK7 0 0 1.5168 64.17 0 0 0 0 0 0
  DIAM 12.7
SURF 2
  TYPE STANDARD
  CURV 0
  DISZ 30.0
  DIAM 12.7
SURF 3
  TYPE COORDBRK
  CURV 0
  PARM 1 1.5
  PARM 2 -2.5
  PARM 3 45.0
  PARM 4 10.0
  PARM 5 90.0
  PARM 6 1
  DISZ 7.0
SURF 4
  TYPE STANDARD
  CURV -0.005
  DISZ -40.0
  GLAS MIRROR
  DIAM 25.0
SURF 5
  TYPE STANDARD
  CURV 0
  DISZ 0
  DIAM 12.7
"""
p = os.path.join(TMP, 'cb_mirror.zmx'); open(p,'w',encoding='utf-8').write(ZMX_C)
rx, ws = W(la.load_zemax_zmx, p)
print("=== (c) COORDBRK + MIRROR ===")
print(" elements:", [(e['element_type'], e['surf_num'], f"R={e['radius']:.5g}") for e in rx['elements']])
print(" all_thicknesses(mm):", [f"{t*1e3:.3f}" for t in rx['all_thicknesses']])
print(" lens thicknesses(mm):", [f"{t*1e3:.3f}" for t in rx['thicknesses']])
print(" coord_breaks:", rx['coord_breaks'])
print(" aperture(mm):", rx['aperture_diameter']*1e3, " stop_index:", rx['stop_index'])
print(" warnings:", ws)
print(" has_mirrors:", la.has_mirrors(rx))
legs = la.split_prescription_at_mirrors(rx)
print(" legs:", [(l['kind'], (len(l['prescription']['surfaces']) if l['kind']=='refractive' else l.get('distance_in'), l.get('distance_out'))) for l in legs])

# tilt convention: +90 deg tilt_x must send local +z to world -y
from lumenairy.raytrace import world as _w
import inspect
print(" _apply_coord_break in raytrace.world:", hasattr(_w, '_apply_coord_break'))

# ---- (d) glass name aliasing ----
from lumenairy.glass import GLASS_REGISTRY, get_glass_index
print("=== (d) glass names ===")
for g in ('N-BK7','BK7','SF11','N-SF11','SF6','N-SF6HT','N-BAF10','BAF10','FUSED_SILICA','SILICA','F2','N-F2','SCHOTT_N-BK7'):
    inreg = g in GLASS_REGISTRY
    try:
        n = get_glass_index(g, 587.6e-9)
    except Exception as e:
        n = f"ERR {type(e).__name__}"
    print(f"   {g:16s} in_registry={inreg!s:5s} n(d)={n}")

# ---- (f) PARAXIAL ideal lens ----
ZMX_F = """VERS 210000 0 123 0 0
MODE SEQ
UNIT MM X W X CM MR CPMM
SURF 0
  TYPE STANDARD
  CURV 0
  DISZ INFINITY
SURF 1
  TYPE PARAXIAL
  STOP
  CURV 0
  PARM 1 100.0
  PARM 2 1
  DISZ 100.0
  DIAM 12.7
SURF 2
  TYPE STANDARD
  CURV 0
  DISZ 0.0
  DIAM 12.7
"""
p = os.path.join(TMP, 'paraxial.zmx'); open(p,'w',encoding='utf-8').write(ZMX_F)
try:
    rxf, wf = W(la.load_zemax_zmx, p)
    print("=== (f) PARAXIAL:", [(e['element_type'], e['radius']) for e in rxf['elements']], wf)
except Exception as e:
    print("=== (f) PARAXIAL raises:", type(e).__name__, str(e)[:150])

# with glass present so the window is non-empty
ZMX_F2 = ZMX_F.replace('SURF 2\n  TYPE STANDARD\n  CURV 0\n  DISZ 0.0\n  DIAM 12.7\n',
 """SURF 2
  TYPE STANDARD
  CURV 0.01
  DISZ 3.0
  GLAS N-BK7 0 0 1.5 50 0 0 0 0 0 0
  DIAM 12.7
SURF 3
  TYPE STANDARD
  CURV 0
  DISZ 50.0
  DIAM 12.7
SURF 4
  TYPE STANDARD
  CURV 0
  DISZ 0
  DIAM 12.7
""")
p = os.path.join(TMP, 'paraxial2.zmx'); open(p,'w',encoding='utf-8').write(ZMX_F2)
rxf2, wf2 = W(la.load_zemax_zmx, p)
print("=== (f2) PARAXIAL+glass: elements:", [(e['element_type'], e['surf_num'], e['radius']) for e in rxf2['elements']])
print("    warnings:", wf2)

# ---- (g) TOROIDAL / BICONICX ----
for tp, parms in (('TOROIDAL', {1: 100.0, 2: 0.0}), ('BICONICX', {1: 0.02, 2: -0.5})):
    body = "\n".join(f"  PARM {k} {v}" for k, v in parms.items())
    ZMX_G = f"""VERS 210000 0 123 0 0
MODE SEQ
UNIT MM X W X CM MR CPMM
SURF 0
  TYPE STANDARD
  CURV 0
  DISZ INFINITY
SURF 1
  TYPE {tp}
  STOP
  CURV 0.02
  DISZ 5.0
  GLAS N-BK7 0 0 1.5 50 0 0 0 0 0 0
{body}
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
    p = os.path.join(TMP, f'{tp}.zmx'); open(p,'w',encoding='utf-8').write(ZMX_G)
    r, wg = W(la.load_zemax_zmx, p)
    s0 = r['surfaces'][0]
    print(f"=== (g) {tp}: radius={s0['radius']:.5g} radius_y={s0.get('radius_y')} asph={s0['aspheric_coeffs']}")
    print(f"    keys={sorted(s0.keys())}")
    print(f"    warn={wg}")

# ---- (i) semi-diameter -> clear aperture factor 2
print("=== (i) DIAM 12.7 (semi) -> aperture_diameter:", rx['aperture_diameter'], "m  (expect 0.0254 when stop DIAM=12.7)")

# ---- (j) multi-config MNUM/MCON
ZMX_J = """VERS 210000 0 123 0 0
MODE SEQ
UNIT MM X W X CM MR CPMM
MNUM 3 1
SURF 0
  TYPE STANDARD
  CURV 0
  DISZ INFINITY
SURF 1
  TYPE STANDARD
  STOP
  CURV 0.02
  DISZ 5.0
  GLAS N-BK7 0 0 1.5 50 0 0 0 0 0 0
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
MCON 1 3 2 50.0 0 0
MCON 2 3 2 75.0 0 0
MCON 3 3 2 100.0 0 0
"""
p = os.path.join(TMP, 'multicfg.zmx'); open(p,'w',encoding='utf-8').write(ZMX_J)
rj, wj = W(la.load_zemax_zmx, p)
print("=== (j) multiconfig: keys=", sorted(rj.keys()))
print("    thicknesses:", rj['thicknesses'], " warnings:", wj)
