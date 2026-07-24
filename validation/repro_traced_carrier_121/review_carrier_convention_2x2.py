# ADVERSARIAL REVIEW - stigmatic-control VARIANTS.  Same chain machinery as
# validation/repro_traced_carrier_121/stigmatic_control_121.py (which this is a
# scratch copy of), but the monkeypatched ideal element has 4 selectable modes:
#
#   MODE=sphere_frozen  sphere(R_in)->sphere(R_out) phase map, amplitude frozen
#                       == the audit's P0.1 control, verbatim
#   MODE=sphere_remap   same phase map, but the COMPLEX carrier-frame envelope is
#                       geometrically remapped by m = A + B/R_in  == the audit
#                       s5 FIX DESIGN applied to an ideal element
#   MODE=reset_frozen   |E| * exp(i k W_sphere(R_out))  (input phase discarded:
#                       a perfectly-corrected element) with frozen amplitude
#   MODE=reset_remap    remapped |E| * exp(i k W_sphere(R_out)) -- the MACHINERY
#                       UPPER BOUND: every element is perfect AND every beam
#                       size is the design q-trace size
#
# Env knobs as the original: RN DX0 RS NFC WF RNF, plus MODE.
import ast, os, re, sys, time, warnings
import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy")
sys.path.insert(0, r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Reverse_Symmetric_ASM")
import lumenairy as la
from lumenairy import GLASS_REGISTRY, GLASS_VALIDITY, SELLMEIER_COEFFICIENTS
from lumenairy.propagators.carrier import (_exact_sphere_eikonal,
                                           _paraxial_group_r_out)
from lumenairy.raytrace.seidel import system_abcd_prescription
from scipy.ndimage import map_coordinates

RUNNER = (r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics"
          r"\Reverse_Symmetric_ASM\run_poc_119_120_v518.py")
ZMX = (r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics"
       r"\Reverse_Symmetric_ASM\tx4designstudy121\20260707 dll Tx02-MSOP16.zmx")
lam = 1.31e-6
w0 = 4e-6
OBJ_GAP = 47.906284e-3
TRAILING = 7.7058e-3
MODE = os.environ.get('MODE', 'sphere_frozen')

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


def _remap(arr, mag):
    """band-limited-ish geometric remap  out(r) = in(r/m)/|m|  (power exact)."""
    n = arr.shape[-1]
    c = n // 2
    ii, jj = np.indices(arr.shape, dtype=np.float64)
    ii = (ii - c) / mag + c
    jj = (jj - c) / mag + c
    if np.iscomplexobj(arr):
        re = map_coordinates(arr.real, [ii, jj], order=3, mode='constant',
                             cval=0.0)
        im = map_coordinates(arr.imag, [ii, jj], order=3, mode='constant',
                             cval=0.0)
        out = (re + 1j * im)
    else:
        out = map_coordinates(arr, [ii, jj], order=3, mode='constant', cval=0.0)
    return out / abs(mag)


def ideal_stigmatic_traced(E_full, *, prescription, wavelength, dx,
                           carrier=None, ray_subsample=None, n_workers=None,
                           **kw):
    E = np.asarray(E_full)
    k = 2.0 * np.pi / wavelength
    R_in = np.inf if carrier is None else float(carrier)
    R_out = _paraxial_group_r_out(prescription, R_in, wavelength)
    S_in = (_exact_sphere_eikonal(E.shape, dx, dx, wavelength, R_in)
            if (np.isfinite(R_in) and R_in != 0.0) else 0.0)
    S_out = (_exact_sphere_eikonal(E.shape, dx, dx, wavelength, R_out)
             if (np.isfinite(R_out) and R_out != 0.0) else 0.0)
    M, _e, _b, _f = system_abcd_prescription(prescription, wavelength)
    A, B = float(M[0, 0]), float(M[0, 1])
    mag = A + (B / R_in if np.isfinite(R_in) else 0.0)

    if MODE == 'sphere_frozen':
        out = E * np.exp(1j * k * (S_out - S_in))
    elif MODE == 'sphere_remap':
        env = E * np.exp(-1j * k * S_in)
        out = _remap(env, mag) * np.exp(1j * k * S_out)
    elif MODE == 'reset_frozen':
        out = np.abs(E) * np.exp(1j * k * S_out)
    elif MODE == 'reset_remap':
        out = _remap(np.abs(E), mag) * np.exp(1j * k * S_out)
    else:
        raise SystemExit(f'bad MODE {MODE}')
    msk = _aperture_mask(E.shape, dx, prescription)
    if msk is not None:
        out = out * msk
    N_CALLS['n'] += 1
    print(f"    [{MODE}] call {N_CALLS['n']}  dx={dx*1e6:8.3f}um N={E.shape[-1]}"
          f" R_in={R_in*1e3:+11.3f} R_out={R_out*1e3:+11.3f} m={mag:.5f}",
          flush=True)
    return out.astype(np.complex128)


import lumenairy.elements as _el
_el.apply_real_lens_traced = ideal_stigmatic_traced


# ---- REVIEW: optionally swap the CHAIN's parabolic carrier convention for the
# EXACT SPHERE (the convention apply_real_lens_traced's H6 eikonal /
# _reference_input / ray-launch already use, per R7/F2).
if os.environ.get('SPHERE_CHAIN', '0') == '1':
    import lumenairy.propagators.carrier as _C

    def _recon_sph(E_env, R_carrier, wavelength, dx, dy=None):
        R = float(R_carrier)
        if not np.isfinite(R) or R == 0.0:
            return np.array(E_env)
        k = 2.0 * np.pi / wavelength
        S = _exact_sphere_eikonal(E_env.shape, dx, dy or dx, wavelength, R)
        return E_env * np.exp(1j * k * S)

    def _env_sph(E_full, R_carrier, wavelength, dx, dy=None):
        R = float(R_carrier)
        if not np.isfinite(R) or R == 0.0:
            return np.array(E_full)
        k = 2.0 * np.pi / wavelength
        S = _exact_sphere_eikonal(E_full.shape, dx, dy or dx, wavelength, R)
        return E_full * np.exp(-1j * k * S)

    _C.carrier_referenced_reconstruct = _recon_sph
    _C.carrier_referenced_envelope = _env_sph
    print('*** CHAIN carrier convention patched to the EXACT SPHERE ***')


# ---- REVIEW: measure the exit wavefront the exact readout consumes, against
# the EXACT sphere(R_out) it references.  Fits piston + r^2 + r^4 + r^6 on a
# radial cut inside the beam and reports the r^4 content in rad at r = w.
if os.environ.get('MEASURE', '0') == '1':
    import lumenairy.propagators.carrier as _CM
    _orig_ro = _CM.carrier_referenced_exact_focus_readout

    def _measured_ro(E_full, R_carrier, z, wavelength, dx, **kw):
        R = float(R_carrier)
        k = 2.0 * np.pi / wavelength
        S = _exact_sphere_eikonal(E_full.shape, dx, dx, wavelength, R)
        env = np.asarray(E_full) * np.exp(-1j * k * S)
        n = env.shape[-1]
        c = n // 2
        I = np.abs(env) ** 2
        xg = (np.arange(n) - c) * dx
        tot = I.sum()
        w_amp = np.sqrt(2.0 * (I * (xg[None, :] ** 2 + xg[:, None] ** 2)).sum()
                        / tot)
        # radial cut (+x), amplitude-gated, unwrapped
        row = env[c, c:]
        rr = xg[c:]
        amp = np.abs(row)
        sel = (rr < 1.15 * w_amp) & (amp > 0.02 * amp.max())
        ph = np.unwrap(np.angle(row[sel]))
        rs = rr[sel]
        Vd = np.stack([np.ones_like(rs), rs ** 2], axis=1)
        cd = np.linalg.lstsq(Vd, ph, rcond=None)[0]
        res_def = ph - Vd @ cd
        V = np.stack([np.ones_like(rs), rs ** 2, rs ** 4, rs ** 6], axis=1)
        cf = np.linalg.lstsq(V, ph, rcond=None)[0]
        pred = k * (w_amp ** 4) / (8.0 * R ** 3)
        print("  [MEASURE] exit-plane wavefront vs EXACT sphere(R_out="
              f"{R*1e3:+.4f}mm), dx={dx*1e6:.4f}um, w_amp={w_amp*1e3:.4f}mm")
        print(f"      r^4 coeff -> {cf[2]*w_amp**4:+8.3f} rad at r=w   "
              f"r^6 -> {cf[3]*w_amp**6:+8.3f} rad at r=w")
        print(f"      rms after removing piston+defocus: "
              f"{np.std(res_def):8.3f} rad   peak-valley "
              f"{res_def.max()-res_def.min():8.3f} rad")
        print(f"      (parabola-vs-sphere prediction for THIS plane: "
              f"{pred:+.3f} rad at r=w)")
        return _orig_ro(E_full, R_carrier, z, wavelength, dx, **kw)

    _CM.carrier_referenced_exact_focus_readout = _measured_ro

N = int(os.environ.get('RN', '2048'))
RS = int(os.environ.get('RS', '4'))
NFC = int(os.environ.get('NFC', '8192'))
WF = float(os.environ.get('WF', '4.0'))
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
    final_distance=TRAILING, focus_readout=_fr, final_leg='auto')
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
print(f"MODE={MODE} N={N} dx0={dx0*1e6:.4f}um nfc={NFC} wf={WF} "
      f"rnf={RNF or 'auto'}: FWHM={fwhm*1e6:.3f}um EE3={ee[3]*100:.1f}% "
      f"EE6={ee[6]*100:.1f}% EE12={ee[12]*100:.1f}% "
      f"window-total={Pwin*100:.1f}%")
print("  TRUE q-trace target: waist 2.737um -> FWHM 3.223um, EE3 91.0%, "
      "EE6 100.0%   |   frozen-w target: FWHM 1.093um")
