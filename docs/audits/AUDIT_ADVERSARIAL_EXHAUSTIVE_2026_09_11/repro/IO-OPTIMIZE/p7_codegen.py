import os, sys, warnings, ast, copy
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
TMP = r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/IO-OPTIMIZE"
import numpy as np, lumenairy as la
from lumenairy.io.codegen import generate_simulation_script

def W(fn,*a,**k):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always'); r=fn(*a,**k)
    return r,[str(x.message)[:100] for x in w]

# --- 1. INJECTION via GLAS name (no whitespace allowed, but quotes are) ---
EVIL = "EVIL');__import__('os').system('echo PWNED');#"
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
p = os.path.join(TMP, 'evil.zmx'); open(p,'w',encoding='utf-8').write(ZMX)
rx,_ = W(la.load_zemax_zmx, p)
print("parsed glass:", rx['surfaces'][0]['glass_after'])
code,_ = W(generate_simulation_script, rx, wavelength=1.31e-6, N=64, include_plotting=False)
for ln in code.splitlines():
    if 'GLASS_REGISTRY' in ln or 'Applying' in ln:
        print("  GEN>", ln)
try:
    ast.parse(code)
    print("  -> generated script PARSES (injection is live code)")
except SyntaxError as e:
    print("  -> SyntaxError:", e)

# --- 2. inf / nan conic, -inf radius ---
rx2 = la.make_singlet(50e-3, float('-inf'), 4e-3, 'N-BK7', aperture=25e-3)
rx2 = la.normalize_prescription(rx2)
rx2['surfaces'][0]['conic'] = float('inf')
rx2['surfaces'][0]['aspheric_coeffs'] = {4: float('nan'), 6: 1e5}
rx2['wavelength'] = 1.31e-6
code2,_ = W(generate_simulation_script, rx2, N=64, include_plotting=False)
for ln in code2.splitlines():
    if '"radius"' in ln or '"conic"' in ln or 'aspheric_coeffs' in ln:
        print("  GEN2>", ln.strip())
try:
    ast.parse(code2); print("  -> parses")
except SyntaxError as e:
    print("  -> SyntaxError:", e)
ns = {}
try:
    exec(compile(code2.split('def run_simulation')[0], 'g', 'exec'), ns)
    print("  -> prescription block EXECUTES; LENS_1_RX radius =", ns['LENS_1_RX']['surfaces'][1]['radius'],
          " conic=", ns['LENS_1_RX']['surfaces'][0]['conic'])
except Exception as e:
    print("  -> exec of generated prescription block FAILS:", type(e).__name__, e)

# --- 3. name with a quote / newline in export_zemax_zmx and codegen ---
rx3 = la.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7')
rx3['name'] = 'bad\nSURF 99\n  CURV 0.5\nNAME x'
q = os.path.join(TMP, 'namebomb.zmx')
la.export_zemax_zmx(rx3, q, wavelength=1.31e-6)
txt = open(q).read()
print("export_zemax_zmx NAME injection -> first 6 lines:")
for ln in txt.splitlines()[:6]: print("   ", repr(ln))
rt,_ = W(la.load_zemax_zmx, q)
print("   reloaded surf count:", len(rt['elements']), "radii:", [s['radius'] for s in rt['surfaces']])

# --- 4. codegen round-trip fidelity on a clean prescription ---
rxc = la.load_zemax_zmx('validation/real_lens_opd/zemax_prescriptions/AC254_100_C.zmx')
code3,_ = W(generate_simulation_script, rxc, wavelength=1.31e-6, N=64, include_plotting=False)
ns3 = {}
exec(compile(code3.split('def run_simulation')[0], 'g', 'exec'), ns3)
gen = ns3['LENS_1_RX']
print("round-trip radii diff:", [a['radius']-b['radius'] for a,b in zip(gen['surfaces'], rxc['surfaces'])])
print("round-trip thick diff:", [a-b for a,b in zip(gen['thicknesses'], rxc['thicknesses'])])
print("glass:", [(s['glass_before'],s['glass_after']) for s in gen['surfaces']])
