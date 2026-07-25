# Converged-config chain + THROUGH-FOCUS scan: every prior metric in this
# campaign was taken exactly at the fixed MSoP plane (final_distance =
# TRAILING); with NA 0.152 the focused Rayleigh range is ~18 um, so
# focus-position errors of tens of um confound cross-config comparisons.
# This runner executes the chain (ray_density + preserve_input_phase=False,
# post lattice-fix), then scans the readout field through +/-DZ um (plain
# fixed-grid Bluestein re-propagation of the readout plane) and reports
# at-plane AND best-focus metrics.
import ast, os, re, sys, time, warnings
import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy")
sys.path.insert(0, r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Reverse_Symmetric_ASM")
import lumenairy as la
from lumenairy import GLASS_REGISTRY, GLASS_VALIDITY, SELLMEIER_COEFFICIENTS
from lumenairy.propagators.mft import angular_spectrum_propagate_mft

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
z1 = 2e-3
gapsum = 0.0
first = True
groups = []
for ev in events:
    if ev['type'] != 'lens':
        gapsum += ev['z_before']
        continue
    g = (OBJ_GAP - z1) if first else (ev['z_before'] + gapsum)
    gapsum = 0.0
    groups.append({'prescription': ev['prescription'], 'gap_before': g})
    first = False

N = int(os.environ.get('RN', '2048'))
RS = int(os.environ.get('RS', '4'))
NFC = int(os.environ.get('NFC', '8192'))
WF = float(os.environ.get('WF', '4.0'))
NOUT = int(os.environ.get('NOUT', '2048'))
AM = os.environ.get('AM', 'ray_density')
PIP = os.environ.get('PIP', '0')
_tkw = {}
if AM:
    _tkw['amplitude_model'] = AM
if PIP == '0':
    _tkw['preserve_input_phase'] = False
elif PIP == 'remap':
    _tkw['preserve_input_phase'] = 'remap'
_fr = {'dx_out': 0.05e-6, 'N_out': NOUT, 'n_fine_cap': NFC,
       'window_factor': WF}
zR = np.pi * w0 * w0 / lam
w_z1 = w0 * np.sqrt(1 + (z1 / zR) ** 2)
R1 = z1 * (1 + (zR / z1) ** 2)
dx0 = float(os.environ.get('DX0', 1.0e-6 * 2048 / N))
x = (np.arange(N) - N // 2) * dx0
env0 = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / w_z1 ** 2
              ).astype(np.complex128)
P_in = float(np.sum(np.abs(env0) ** 2)) * dx0 * dx0

t0 = time.time()
res = la.propagate_traced_carrier_chain(
    env0, groups, lam, dx0, r_in=R1, ray_subsample=RS, n_workers=8,
    final_distance=TRAILING, focus_readout=_fr, final_leg='auto',
    **({'traced_kwargs': _tkw} if _tkw else {}))
print(f"chain done {time.time()-t0:.0f}s")

dxo = 0.05e-6


def metrics(E):
    E = np.asarray(E)
    n = E.shape[-1]
    I = np.abs(E) ** 2
    iy, ix = np.unravel_index(np.argmax(I), I.shape)
    xx = (np.arange(n) - ix) * dxo
    yy = (np.arange(n) - iy) * dxo
    rr = np.sqrt(xx[None, :] ** 2 + yy[:, None] ** 2)
    nb = 500
    ring = np.clip((rr / dxo).astype(int), 0, nb)
    s = np.bincount(ring.ravel(), weights=I.ravel(), minlength=nb + 1)
    cnt = np.bincount(ring.ravel(), minlength=nb + 1)
    prof = (s[:nb] / np.maximum(cnt[:nb], 1)) / I[iy, ix]
    rb = (np.arange(nb) + 0.5) * dxo
    idx = np.where(prof < 0.5)[0]
    fwhm = 2 * rb[idx[0]] if len(idx) else np.nan
    ee = {r: float(I[rr <= r * 1e-6].sum()) * dxo * dxo / P_in
          for r in (3, 6, 12)}
    off = ((ix - n // 2) * dxo, (iy - n // 2) * dxo)
    return fwhm, ee, float(I.max()), off


E0 = np.asarray(res.field)
fw, ee, pk, off = metrics(E0)
print(f"AT-PLANE: FWHM={fw*1e6:.3f}um EE3={ee[3]*100:.1f}% EE6={ee[6]*100:.1f}% "
      f"EE12={ee[12]*100:.1f}% off=({off[0]*1e6:+.2f},{off[1]*1e6:+.2f})um")

best = (0.0, -1.0, None)
for dz_um in range(-80, 81, 5):
    if dz_um == 0:
        Ez = E0
    else:
        Ez = angular_spectrum_propagate_mft(
            E0, dz_um * 1e-6, lam, dxo, dxo, NOUT)
    fw, ee, pk, off = metrics(Ez)
    if ee[6] > best[1]:
        best = (dz_um, ee[6], (fw, ee, pk, off))
    print(f"  dz={dz_um:+4d}um: FWHM={fw*1e6:6.3f} EE3={ee[3]*100:5.1f} "
          f"EE6={ee[6]*100:5.1f} EE12={ee[12]*100:5.1f} pk={pk:.3e}",
          flush=True)
fw, ee, pk, off = best[2]
print(f"BEST-FOCUS dz={best[0]:+d}um: FWHM={fw*1e6:.3f}um EE3={ee[3]*100:.1f}% "
      f"EE6={ee[6]*100:.1f}% EE12={ee[12]*100:.1f}% "
      f"off=({off[0]*1e6:+.2f},{off[1]*1e6:+.2f})um")
print("  targets: FWHM 3.223um EE3 91.0% EE6 100.0% (POP waist 2.737um radius)")
