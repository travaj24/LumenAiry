# ADVERSARIAL REVIEW - the REAL traced 121 chain, with an optional swap of the
# CHAIN's carrier convention from the paraxial parabola
# (carrier_referenced_reconstruct/envelope, exp(i k r^2/2R)) to the EXACT SPHERE
# S(R) = sign(R)(sqrt(r^2+R^2)-|R|) -- the convention apply_real_lens_traced's
# own carrier legs (R7/F2 _compute_carrier W, the H6 entrance eikonal,
# _reference_input) and carrier_referenced_exact_focus_readout already use.
# Scratch copy of validation/repro_traced_carrier_121/repro_dx_scaling.py
# (no .npy dump).  Env: RN DX0 RS NFC WF RNF PIP SPHERE_CHAIN.
import ast, os, re, sys, time, warnings
import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy")
sys.path.insert(0, r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Reverse_Symmetric_ASM")
import lumenairy as la
from lumenairy import GLASS_REGISTRY, GLASS_VALIDITY, SELLMEIER_COEFFICIENTS
from lumenairy.propagators.carrier import _exact_sphere_eikonal

RUNNER = (r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics"
          r"\Reverse_Symmetric_ASM\run_poc_119_120_v518.py")
ZMX = (r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics"
       r"\Reverse_Symmetric_ASM\tx4designstudy121\20260707 dll Tx02-MSOP16.zmx")
lam = 1.31e-6
w0 = 4e-6
OBJ_GAP = 47.906284e-3
TRAILING = 7.7058e-3
z1 = 2e-3

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

SPH = os.environ.get('SPHERE_CHAIN', '0') == '1'
if SPH:
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
    print('*** CHAIN carrier convention = EXACT SPHERE ***')
else:
    print('*** CHAIN carrier convention = PARABOLA (library default) ***')


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
PIP = os.environ.get('PIP', '')
_fr = {'dx_out': 0.05e-6, 'N_out': 1024, 'n_fine_cap': NFC,
       'window_factor': WF}
if RNF:
    _fr['N_fine'] = int(RNF)
_tkw = {'preserve_input_phase': False} if PIP == '0' else None
zR = np.pi * w0 * w0 / lam
w_z1 = w0 * np.sqrt(1 + (z1 / zR) ** 2)
R1 = z1 * (1 + (zR / z1) ** 2)
dx0 = float(os.environ.get('DX0', 1.0e-6 * 2048 / N))
x = (np.arange(N) - N // 2) * dx0
env0 = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / w_z1 ** 2
              ).astype(np.complex128)
P_in = float(np.sum(np.abs(env0) ** 2)) * dx0 * dx0

t0 = time.time()
_chain_kw = {} if _tkw is None else {'traced_kwargs': _tkw}
res = la.propagate_traced_carrier_chain(
    env0, groups, lam, dx0, r_in=R1, ray_subsample=RS, n_workers=8,
    final_distance=TRAILING, focus_readout=_fr, final_leg='auto', **_chain_kw)
print("chain done", f"{time.time()-t0:.0f}s")
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
print(f"REAL-TRACED sphere_chain={int(SPH)} N={N} dx0={dx0*1e6:.4f}um rs={RS} "
      f"nfc={NFC} wf={WF} rnf={RNF or 'auto'} pip={PIP or '1'}: "
      f"FWHM={fwhm*1e6:.3f}um EE3={ee[3]*100:.1f}% EE6={ee[6]*100:.1f}% "
      f"EE12={ee[12]*100:.1f}% window-total={Pwin*100:.1f}%")
print("  TRUE target: waist 2.737um -> FWHM 3.223um, EE3 91.0%, EE6 100.0%;"
      " machinery ceiling at this readout plane: FWHM 3.550um EE3 87.4%")
