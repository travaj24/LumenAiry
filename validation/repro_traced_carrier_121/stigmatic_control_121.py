# Stigmatic control for AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24 P0.1:
# run the design-121 chain GEOMETRY (same groups, gaps, apertures, launch,
# readout) through the SAME carrier machinery -- propagate_traced_carrier_chain
# with its gap transports, paraxial reconstruct/envelope hand-offs, the
# _fine_trace_group_exit resample+retrace, and the exact Bluestein readout --
# but with apply_real_lens_traced monkeypatched to an IDEAL STIGMATIC element:
# an exact sphere(R_in) -> sphere(R_out) phase map (|T| = 1, energy-conserving
# by construction) plus the group's hard aperture.  R_out comes from the same
# paraxial ABCD the orchestrator itself uses, so the hand-off radii are
# byte-identical to a real chain run.
#
# Localization logic (audit S5, P0.1):
#   * stigmatic CONVERGES (EE6 stable, window <= ~1.0) while traced diverges
#       -> the defect is in apply_real_lens_traced (OPL fit / interpolation
#          under grid refinement).
#   * stigmatic ALSO diverges -> the defect is in the shared machinery
#       (carrier gap transport / envelope-reconstruct / resample / readout).
#
# Env knobs (mirror repro_dx_scaling.py):
#   RN  = chain N                     (default 2048)
#   DX0 = launch pitch in metres      (default extent-preserving 1.0e-6*2048/RN)
#   RS  = ray_subsample               (default 4; inert for the ideal element)
#   NFC = n_fine_cap                  (default 16384)
#   WF  = window_factor               (default 7.0)
# Axis A (chain grid):    sweep RN with the extent-preserving default DX0.
# Axis B (final-leg res): fix RN=2048, DX0=1e-6, sweep NFC (and optionally WF).
import ast, os, re, sys, time, warnings
import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy")
sys.path.insert(0, r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Reverse_Symmetric_ASM")
import lumenairy as la
from lumenairy import GLASS_REGISTRY, GLASS_VALIDITY, SELLMEIER_COEFFICIENTS
from lumenairy.propagators.carrier import (_exact_sphere_eikonal,
                                           _paraxial_group_r_out)

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

# group specs: lens events only; the disabled DOE folds its gap into the next
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
print("groups:", [(gr['prescription'].get('name', '?'),
                   f"{gr['gap_before']*1e3:.3f}mm") for gr in groups])

# ---- ideal stigmatic element (monkeypatch target) -------------------------
N_CALLS = {'n': 0}

def _aperture_mask(shape, dx, prescription):
    ap = (prescription.get('aperture_diameter')
          if isinstance(prescription, dict) else None)
    if ap is None or not np.isfinite(ap) or ap <= 0:
        return None
    n = shape[-1]
    x = (np.arange(n) - n // 2) * dx
    rr2 = x[None, :] ** 2 + x[:, None] ** 2
    return rr2 <= (0.5 * float(ap)) ** 2

def ideal_stigmatic_traced(E_full, *, prescription, wavelength, dx,
                           carrier=None, ray_subsample=None, n_workers=None,
                           **kw):
    """Exact sphere(R_in)->sphere(R_out) thin phase map + hard aperture.
    |T| = 1 pointwise, so any energy loss is the aperture's and any energy
    GAIN downstream is the machinery's."""
    E = np.asarray(E_full)
    k = 2.0 * np.pi / wavelength
    R_in = np.inf if carrier is None else float(carrier)
    R_out = _paraxial_group_r_out(prescription, R_in, wavelength)
    ph = np.zeros(E.shape, dtype=np.float64)
    if np.isfinite(R_in) and R_in != 0.0:
        ph -= _exact_sphere_eikonal(E.shape, dx, dx, wavelength, R_in)
    if np.isfinite(R_out) and R_out != 0.0:
        ph += _exact_sphere_eikonal(E.shape, dx, dx, wavelength, R_out)
    out = E * np.exp(1j * k * ph)
    msk = _aperture_mask(E.shape, dx, prescription)
    if msk is not None:
        out = out * msk
    N_CALLS['n'] += 1
    return out

import lumenairy.elements as _el
_el.apply_real_lens_traced = ideal_stigmatic_traced   # both chain call sites
                                                      # import lazily at call

# ---- chain run (identical to repro_dx_scaling.py) --------------------------
N = int(os.environ.get('RN', '2048'))
RS = int(os.environ.get('RS', '4'))
NFC = int(os.environ.get('NFC', '16384'))
WF = float(os.environ.get('WF', '7.0'))
# RNF: pin the exact readout's internal N_fine (memory bound); '' = unpinned.
# NOTE: the thin stigmatic stub does not compress the beam through the last
# group (no thick-group amplitude mapping), so the readout would size itself
# to 32768^2 (16 GiB) unpinned -- pin RNF=16384 for sweeps.
RNF = os.environ.get('RNF', '')
_fr = {'dx_out': 0.05e-6, 'N_out': 1024, 'n_fine_cap': NFC,
       'window_factor': WF}
if RNF:
    _fr['N_fine'] = int(RNF)
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
    final_distance=TRAILING,
    focus_readout=_fr,
    final_leg='auto')
print(f"chain done {time.time()-t0:.0f}s  (ideal-element calls: {N_CALLS['n']})")
for st in res.stages:
    print(f"  stage {st.get('name','?'):<14} dx={st['dx']*1e6:8.4f}um "
          f"w={st['w']*1e3:8.4f}mm power={st['power']:.6e} "
          f"R_in={st['R_in']*1e3:+9.3f}mm R_out={st['R_out']*1e3:+9.3f}mm"
          + ("  [exact_final]" if st.get('exact_final') else ""))

E = res.field
dxo = 0.05e-6
I = np.abs(E) ** 2
iy, ix = np.unravel_index(np.argmax(I), I.shape)
print(f"peak offset: ({(ix-512)*dxo*1e6:+.2f}, {(iy-512)*dxo*1e6:+.2f}) um")
xx = (np.arange(1024) - ix) * dxo
yy = (np.arange(1024) - iy) * dxo
rr = np.sqrt(xx[None, :] ** 2 + yy[:, None] ** 2)
nb = 500
ring = np.clip((rr / dxo).astype(int), 0, nb)
s = np.bincount(ring.ravel(), weights=I.ravel(), minlength=nb + 1)
cnt = np.bincount(ring.ravel(), minlength=nb + 1)
prof = (s[:nb] / np.maximum(cnt[:nb], 1)) / I[iy, ix]
rb = (np.arange(nb) + 0.5) * dxo
idx = np.where(prof < 0.5)[0]
fwhm = 2 * rb[idx[0]] if len(idx) else np.nan
ee = {r: float(I[rr <= r * 1e-6].sum()) * dxo * dxo / P_in for r in (3, 6, 12)}
Pwin = float(I.sum()) * dxo * dxo / P_in
print(f"STIGMATIC N={N} dx0={dx0*1e6:.4f}um rs={RS} nfc={NFC} wf={WF} "
      f"rnf={RNF or 'auto'}: "
      f"FWHM={fwhm*1e6:.2f}um EE3={ee[3]*100:.1f}% EE6={ee[6]*100:.1f}% "
      f"EE12={ee[12]*100:.1f}% window-total={Pwin*100:.1f}% "
      f"(stigmatic exp20 ref: 2.97um at plane)")
