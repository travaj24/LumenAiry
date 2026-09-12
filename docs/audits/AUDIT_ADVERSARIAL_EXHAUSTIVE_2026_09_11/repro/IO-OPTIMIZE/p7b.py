import os, sys, warnings, ast, copy
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
TMP = r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/IO-OPTIMIZE"
import numpy as np, lumenairy as la
from lumenairy.io.codegen import generate_simulation_script
def W(fn,*a,**k):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always'); r=fn(*a,**k)
    return r,[str(x.message)[:100] for x in w]

# ---- whitespace-free injection payload ----
EVIL = "X'];__import__(\"builtins\").print(\"PWNED-FROM-GLASS-NAME\");_z=['y"
ZMX = f"""VERS 210000 0 123 0 0
MODE SEQ
UNIT MM X W X CM MR CPMM
SURF 0
  TYPE STANDARD
  CURV 0
  DISZ INFINITY
SURF 1
  TYPE STANDARD
  STOP
  CURV 0.02
  DISZ 5.0
  GLAS {EVIL} 0 0 1.5 50 0 0 0 0 0 0
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
p = os.path.join(TMP, 'evil2.zmx'); open(p,'w',encoding='utf-8').write(ZMX)
rx,_ = W(la.load_zemax_zmx, p)
code,_ = W(generate_simulation_script, rx, wavelength=1.31e-6, N=64,
           include_plotting=False, include_analysis=False)
gl = [l for l in code.splitlines() if 'GLASS_REGISTRY' in l]
print("INJECTED LINE:", gl[0] if gl else '(none)')
try:
    ast.parse(code); print("  -> generated script PARSES OK")
    head = code.split('# ' + '='*70 + '\n# LENS PRESCRIPTIONS')[0]
    exec(compile('\n'.join(gl), 'gen', 'exec'), {'la': type('X',(object,),{'GLASS_REGISTRY':{}})()})
    print("  -> the GLASS_REGISTRY lines EXECUTED the injected payload above")
except SyntaxError as e:
    print("  -> SyntaxError:", e)
except Exception as e:
    print("  -> exec raised after payload:", type(e).__name__, e)

# ---- inf/nan emission (with element_type so codegen accepts it) ----
rx2 = la.make_singlet(50e-3, float('-inf'), 4e-3, 'N-BK7', aperture=25e-3)
for s in rx2['surfaces']: s['element_type'] = 'surface'; s['semi_diameter'] = 12.5e-3
rx2['surfaces'][0]['conic'] = float('inf')
rx2['surfaces'][0]['aspheric_coeffs'] = {4: float('nan'), 6: 1e5}
rx2['elements'] = rx2['surfaces']; rx2['all_thicknesses'] = rx2['thicknesses']
code2,_ = W(generate_simulation_script, rx2, wavelength=1.31e-6, N=64, include_plotting=False)
for ln in code2.splitlines():
    if '"radius"' in ln or '"conic"' in ln or 'aspheric_coeffs' in ln:
        print("  GEN2>", ln.strip())
try:
    exec(compile(code2.split('# SIMULATION')[0], 'g', 'exec'), {})
    print("  -> prescription block executes")
except Exception as e:
    print("  -> prescription block FAILS:", type(e).__name__, e)

# ---- normalize_prescription -> codegen contract ----
rxn = la.normalize_prescription(la.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7'))
rxn['wavelength'] = 1.31e-6
try:
    generate_simulation_script(rxn, N=64, include_plotting=False)
    print("normalize_prescription -> codegen: OK")
except Exception as e:
    print("normalize_prescription -> codegen: RAISES", type(e).__name__, e)

# ---- scale_prescription identities ----
rxq = la.load_zemax_zmx('validation/real_lens_opd/zemax_prescriptions/AC254_100_C.zmx')
rxq['surfaces'][0]['freeform_type'] = 'q_bfs'
rxq['surfaces'][0]['q_bfs_coeffs'] = [1e-6, 2e-7]
rxq['surfaces'][0]['r_max'] = 7.5e-3
rxq['back_focal_length'] = 0.084
rxq['diffractives'] = [{'type': 'grating', 'period': 2e-6, 'order': 1,
                        'origin': (1e-3, 0.0), 'gap_before': 1e-2, 'gap_after': 2e-2,
                        'semi_diameter': 6e-3, 'angle_deg': 0.0}]
s = 0.25
a = la.scale_prescription(rxq, s)
b = la.scale_prescription(a, 1/s)
print("scale round-trip max |dR|:", max(abs(x['radius']-y['radius']) for x,y in zip(b['surfaces'], rxq['surfaces'])))
print("scale round-trip max |dt|:", max(abs(x-y) for x,y in zip(b['thicknesses'], rxq['thicknesses'])))
print("scaled r_max:", a['surfaces'][0]['r_max'], " expected", 7.5e-3*s)
print("scaled q_bfs_coeffs:", a['surfaces'][0]['q_bfs_coeffs'], " expected", [1e-6*s, 2e-7*s])
print("scaled back_focal_length:", a.get('back_focal_length'), " expected", 0.084*s)
print("scaled diffractive period:", a['diffractives'][0]['period'], " expected", 2e-6*s)
print("scaled diffractive gap_before:", a['diffractives'][0]['gap_before'], " expected", 1e-2*s)
# aspheric scaling identity
rxa = la.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7')
rxa['surfaces'][0]['aspheric_coeffs'] = {4: 1.2e5, 6: -3.4e9}
sa = la.scale_prescription(rxa, s)
h = 3e-3
sag0 = sum(v*h**k for k,v in rxa['surfaces'][0]['aspheric_coeffs'].items())
sag1 = sum(v*(s*h)**k for k,v in sa['surfaces'][0]['aspheric_coeffs'].items())
print(f"aspheric self-similarity: sag(s*h)/sag(h) = {sag1/sag0:.17g} (expect {s})")
