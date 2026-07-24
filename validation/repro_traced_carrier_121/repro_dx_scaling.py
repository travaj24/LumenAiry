# dx-scaling repro for AUDIT_TRACED_CHAIN_DX_SCALING_2026_07_22 (F-A/F-B):
# design-121 traced chain (no DOE) on the v5.28 stack --
# propagate_traced_carrier_chain (R8) + R9 exact final leg (NA~0.46).
# Env knobs: RN = chain N; DX0 = launch pitch in metres (omit for the
# extent-preserving default 1.0e-6*2048/RN, which walks the F-B divergence
# and, at RN > 16384, the F-A n_fine-cap energy bug); RS = ray_subsample.
# Audit table rows: (RN,DX0,RS) = (1024,2e-6,2) (2048,1e-6,4) (2048,1e-6,8)
# (4096,.5e-6,8) (8192,.25e-6,4) (28672,extent-preserving,4).
# Requires the design-study runner + zmx alongside (paths below), as
# carrier_chain_121.py does.
import ast, re, sys, time, warnings
import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy")
sys.path.insert(0, r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Reverse_Symmetric_ASM")
import lumenairy as la
from lumenairy import GLASS_REGISTRY, GLASS_VALIDITY, SELLMEIER_COEFFICIENTS

RUNNER = (r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics"
          r"\Reverse_Symmetric_ASM\run_poc_119_120_v518.py")
ZMX = (r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics"
       r"\Reverse_Symmetric_ASM\tx4designstudy121\20260707 dll Tx02-MSOP16.zmx")
lam = 1.31e-6
w0 = 4e-6
OBJ_GAP = 47.906284e-3
TRAILING = 7.7058e-3

src = open(RUNNER, 'r', encoding='utf-8', errors='ignore').read()
m = re.search(r'_NEW_GLASSES\s*=\s*\{', src)
i0 = m.end() - 1
d = 0
for i in range(i0, len(src)):
    if src[i] == '{': d += 1
    elif src[i] == '}':
        d -= 1
        if d == 0: break
for g, (c, r, nd) in ast.literal_eval(src[i0:i+1]).items():
    if g not in SELLMEIER_COEFFICIENTS:
        SELLMEIER_COEFFICIENTS[g] = c; GLASS_REGISTRY[g] = '__sellmeier__'; GLASS_VALIDITY[g] = r

import tx_design_study_sim as sim
rx = la.load_zemax_zmx(ZMX)
events = sim.group_elements_into_lenses(rx)

# group specs: lens events only; DOE (disabled) folds its gap into the next
groups = []
pending_gap = OBJ_GAP
first = True
z1 = 2e-3
for ev in events:
    if ev['type'] != 'lens':
        pending_gap += ev['z_before'] if not first else 0.0
        continue
    gap = ev['z_before'] if not first else pending_gap - z1
    if not first:
        gap += pending_gap - (ev['z_before'] if False else 0.0) if False else 0.0
    groups.append({'prescription': ev['prescription'],
                   'gap_before': gap + (0.0 if first else 0.0)})
    first = False
# fix: fold DOE gap (51.54) into S14-S15's gap (7.0) explicitly
gaps = []
pend = 0.0
first = True
groups = []
for ev in events:
    if ev['type'] != 'lens':
        pend += ev['z_before']
        continue
    g = (OBJ_GAP - z1) if first else (ev['z_before'] + pend)
    pend = 0.0
    groups.append({'prescription': ev['prescription'], 'gap_before': g})
    first = False
print("groups:", [(gr['prescription'].get('name', '?'), f"{gr['gap_before']*1e3:.3f}mm")
                  for gr in groups])

import os
N = int(os.environ.get('RN','2048'))
RS = int(os.environ.get('RS','4'))
# Optional final-leg knobs (defaults preserve the original behavior):
# NFC = n_fine_cap for the pre-readout re-trace (default 16384)
# WF  = window_factor for re-trace + readout (default 7.0)
# RNF = PIN the exact readout's internal N_fine (default: unpinned -- the
#       readout sizes itself to Nyquist, up to 32768^2 = 16 GiB at 121
#       conditions; pin to 16384 to bound memory on smaller boxes)
NFC = int(os.environ.get('NFC', '16384'))
WF = float(os.environ.get('WF', '7.0'))
RNF = os.environ.get('RNF', '')
# PIP=0 -> pass traced_kwargs={'preserve_input_phase': False} to every group
# (defect-A diagnostic, AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24 S3); unset
# (default) leaves the chain's call untouched.
PIP = os.environ.get('PIP', '')
_fr = {'dx_out': 0.05e-6, 'N_out': 1024, 'n_fine_cap': NFC,
       'window_factor': WF}
if RNF:
    _fr['N_fine'] = int(RNF)
_tkw = {'preserve_input_phase': False} if PIP == '0' else None
zR = np.pi*w0*w0/lam
w_z1 = w0*np.sqrt(1 + (z1/zR)**2)
R1 = z1*(1 + (zR/z1)**2)
# extent-preserving default (dx0 ~ 1/N) is dx-DIVERGENT (F-B) and hits the
# n_fine cap bug (F-A) at N>16384; DX0 pins the launch pitch instead
# (pitch-preserving: validated per-stage dx at any N, N adds guard band).
dx0 = float(os.environ.get('DX0', 1.0e-6*2048/N))
x = (np.arange(N) - N//2)*dx0
env0 = np.exp(-(x[None,:]**2 + x[:,None]**2)/w_z1**2).astype(np.complex128)
P_in = float(np.sum(np.abs(env0)**2))*dx0*dx0

t0 = time.time()
_chain_kw = {} if _tkw is None else {'traced_kwargs': _tkw}
res = la.propagate_traced_carrier_chain(
    env0, groups, lam, dx0, r_in=R1, ray_subsample=RS, n_workers=8,
    final_distance=TRAILING,
    focus_readout=_fr,
    final_leg='auto', **_chain_kw)
print("chain done", f"{time.time()-t0:.0f}s")
for st in res.stages:
    print(f"  stage {st.get('name','?'):<14} dx={st['dx']*1e6:8.4f}um "
          f"w={st['w']*1e3:8.4f}mm power={st['power']:.6e} "
          f"R_in={st['R_in']*1e3:+9.3f}mm R_out={st['R_out']*1e3:+9.3f}mm"
          + ("  [exact_final]" if st.get('exact_final') else ""))
E = res.field if hasattr(res, 'field') else (res.image if hasattr(res, 'image') else res[0])
dxo = 0.05e-6
I = np.abs(E)**2
iy, ix = np.unravel_index(np.argmax(I), I.shape)
print(f"peak offset: ({(ix-512)*dxo*1e6:+.2f}, {(iy-512)*dxo*1e6:+.2f}) um")
xx = (np.arange(1024)-ix)*dxo; yy = (np.arange(1024)-iy)*dxo
rr = np.sqrt(xx[None,:]**2 + yy[:,None]**2)
nb = 500
ring = np.clip((rr/dxo).astype(int), 0, nb)
s = np.bincount(ring.ravel(), weights=I.ravel(), minlength=nb+1)
cnt = np.bincount(ring.ravel(), minlength=nb+1)
prof = (s[:nb]/np.maximum(cnt[:nb],1))/I[iy,ix]
rb = (np.arange(nb)+0.5)*dxo
idx = np.where(prof < 0.5)[0]
fwhm = 2*rb[idx[0]] if len(idx) else np.nan
ee = {r: float(I[rr <= r*1e-6].sum())*dxo*dxo/P_in for r in (3, 6, 12)}
fn = f'runA_field_N{N}.npy' if RS == 4 else f'diagA_N{N}_rs{RS}.npy'
np.save(fn, E)
Pwin = float(I.sum())*dxo*dxo/P_in
print(f"RUN A N={N} dx0={dx0*1e6:.4f}um rs={RS} nfc={NFC} wf={WF} rnf={RNF or 'auto'} "
      f"pip={PIP or '1'} (traced noDOE, exact final leg): "
      f"FWHM={fwhm*1e6:.2f}um EE3={ee[3]*100:.1f}% EE6={ee[6]*100:.1f}% EE12={ee[12]*100:.1f}% "
      f"window-total={Pwin*100:.1f}% "
      f"(refs: Zemax 2.74um; audit-R9 table 4.05um/52.1/69.7/73.6)")
