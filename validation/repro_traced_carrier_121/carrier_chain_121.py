# Carrier-referenced (pilot-beam) traced chain for design 121 (P8-style
# composition): envelope legs via propagate_carrier_referenced, per-group
# apply_real_lens_traced(carrier=R_in) at reconstructed planes, re-envelope
# with the q-trace R_out.  N=2048 co-moving grid.  Per-stage w checks vs the
# physical q-trace.
import ast, re, sys, time, warnings
import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy")
sys.path.insert(0, r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Reverse_Symmetric_ASM")
import lumenairy as la
from lumenairy import GLASS_REGISTRY, GLASS_VALIDITY, SELLMEIER_COEFFICIENTS, get_glass_index

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
depth = 0
for i in range(i0, len(src)):
    if src[i] == '{': depth += 1
    elif src[i] == '}':
        depth -= 1
        if depth == 0: break
for gname, (coefs, rng, nd) in ast.literal_eval(src[i0:i+1]).items():
    if gname not in SELLMEIER_COEFFICIENTS:
        SELLMEIER_COEFFICIENTS[gname] = coefs
        GLASS_REGISTRY[gname] = '__sellmeier__'
        GLASS_VALIDITY[gname] = rng

import tx_design_study_sim as sim
rx = la.load_zemax_zmx(ZMX)
events = sim.group_elements_into_lenses(rx)
lens_events = [ev for ev in events if ev['type'] == 'lens']
print("lens events:", [(ev['name'], f"{ev['z_before']*1e3:.2f}mm") for ev in lens_events])

# ---- physical q-trace with group-boundary snapshots ----
def n_of(g):
    return 1.0 if g in (None, '', 'air', 'AIR') else float(get_glass_index(g, lam))

elements = rx['elements']; th = rx['thicknesses']
group_bounds = [(ev['surf_start'], ev['surf_end']) for ev in lens_events]
q = 1j*np.pi*w0*w0/lam
z = 0.0; gap = OBJ_GAP
R_at = {}   # (surf_num,'in'/'out') -> R
def R_of(q):
    iq = 1.0/q
    return np.inf if abs(iq.real) < 1e-30 else 1.0/iq.real
for i, el in enumerate(elements):
    n1 = n_of(el['glass_before']); n2 = n_of(el['glass_after'])
    q = q + gap
    for (s0, s1_) in group_bounds:
        if el['surf_num'] == s0:
            R_at[(s0, 'in')] = R_of(q)
    R_s = el['radius'] if el['radius'] else np.inf
    if el['surf_num'] in (9, 11):
        R_s = np.inf
    if n1 != n2 or np.isfinite(R_s):
        pw = 0.0 if not np.isfinite(R_s) else (n2 - n1)/R_s
        q = n2/(n1*(1.0/q) - pw)
    for (s0, s1_) in group_bounds:
        if el['surf_num'] == s1_:
            R_at[(s1_, 'out')] = R_of(q)
    gap = th[i] if i < len(th) else 0.0
print("R_in/R_out per group [mm]:")
for ev in lens_events:
    print(f"  {ev['name']:<16} R_in={R_at[(ev['surf_start'],'in')]*1e3:+10.3f}  "
          f"R_out={R_at[(ev['surf_end'],'out')]*1e3:+10.3f}")

# ---- carrier chain ----
N = 2048
z1 = 2e-3                       # hand-off from waist: analytic Gaussian at z1
zR = np.pi*w0*w0/lam
w_z1 = w0*np.sqrt(1 + (z1/zR)**2)
R1 = z1*(1 + (zR/z1)**2)
dx = 1.0e-6
x = (np.arange(N) - N//2) * dx
env = np.exp(-(x[None,:]**2 + x[:,None]**2)/w_z1**2).astype(np.complex128)
R = R1
print(f"launch: w={w_z1*1e6:.2f}um R={R*1e3:.4f}mm dx={dx*1e6:.2f}um")

def wmeas(env, dx):
    I = np.abs(env)**2
    P = I.sum()
    xx = (np.arange(env.shape[1]) - env.shape[1]//2)*dx
    cx = (I.sum(0)*xx).sum()/P
    vx = (I.sum(0)*(xx-cx)**2).sum()/P
    return 2*np.sqrt(vx)          # 1/e^2 radius for Gaussian = 2*sigma

def step(env, R, dx, z, label):
    cr = la.propagate_carrier_referenced(env, R, z, lam, dx)
    env2, R2, dx2 = cr.env if hasattr(cr, 'env') else cr[0], cr.R if hasattr(cr, 'R') else cr[1], cr.dx if hasattr(cr, 'dx') else cr[2]
    if isinstance(dx2, tuple): dx2 = dx2[0]
    if isinstance(R2, tuple): R2 = R2[0]
    print(f"  [leg {label}] z={z*1e3:+8.3f}mm -> R={R2*1e3:+9.3f}mm dx={dx2*1e6:7.3f}um "
          f"w={wmeas(env2, dx2)*1e3:7.3f}mm grid={N*dx2*1e3:7.2f}mm", flush=True)
    return env2, R2, dx2

gaps = [ev['z_before'] for ev in events]      # per-event air gap before it
gaps[0] = OBJ_GAP
t0 = time.time()
first = True
for ev, gap_ev in zip(events, gaps):
    legz = gap_ev - z1 if first else gap_ev
    first = False
    if abs(legz) > 1e-12:
        env, R, dx = step(env, R, dx, legz, f"-> {ev['name']}")
    if ev['type'] != 'lens':
        print(f"  [skip] {ev['name']} (DOE disabled)")
        continue
    Rin_t = R_at[(ev['surf_start'], 'in')]
    Rout_t = R_at[(ev['surf_end'], 'out')]
    print(f"  [lens {ev['name']}] carrier R={R*1e3:+.3f}mm (q-trace {Rin_t*1e3:+.3f}mm)", flush=True)
    E_full = la.carrier_referenced_reconstruct(env, R, lam, dx)
    E_exit = la.apply_real_lens_traced(
        E_full, prescription=ev['prescription'], wavelength=lam, dx=dx,
        ray_subsample=4, n_workers=8, carrier=R)
    env = la.carrier_referenced_envelope(E_exit, Rout_t, lam, dx)
    R = Rout_t
    print(f"    exit: R_out={R*1e3:+.3f}mm w={wmeas(env, dx)*1e3:.3f}mm "
          f"P={np.sum(np.abs(env)**2)*dx*dx:.4e}", flush=True)

# land 0.5 mm SHORT of the MSoP plane, then finish with fine ASM zoom
z_short = 0.5e-3
env, R, dx = step(env, R, dx, TRAILING - z_short, "trailing-0.5mm")
E = la.carrier_referenced_reconstruct(env, R, lam, dx)
P_tot = float(np.sum(np.abs(E)**2))*dx*dx
print(f"  pre-plane grid: dx={dx*1e6:.3f}um window={N*dx*1e3:.3f}mm P={P_tot:.4e}")

def fine_metrics(Ef, dxf):
    I = np.abs(Ef)**2
    iy, ix = np.unravel_index(np.argmax(I), I.shape)
    n = I.shape[0]
    xx = (np.arange(n)-ix)*dxf; yy = (np.arange(n)-iy)*dxf
    rr = np.sqrt(xx[None,:]**2 + yy[:,None]**2)
    nb = int(30e-6/dxf)
    ring = np.clip((rr/dxf).astype(int), 0, nb)
    sums = np.bincount(ring.ravel(), weights=I.ravel(), minlength=nb+1)
    cnts = np.bincount(ring.ravel(), minlength=nb+1)
    prof = (sums[:nb]/np.maximum(cnts[:nb],1))/I[iy,ix]
    rb = (np.arange(nb)+0.5)*dxf
    def cross(level):
        idx = np.where(prof < level)[0]
        if len(idx) == 0: return np.nan
        i_ = idx[0]
        if i_ == 0: return rb[0]
        fr = (prof[i_-1]-level)/(prof[i_-1]-prof[i_])
        return rb[i_-1] + fr*(rb[i_]-rb[i_-1])
    ee = {r: float(I[rr <= r*1e-6].sum())*dxf*dxf/P_tot for r in (3,6,12)}
    return cross(np.exp(-2)), 2*cross(0.5), ee

best = (None, -1, None)
for dz_um in np.arange(-200, 701, 25):
    zt = z_short + dz_um*1e-6
    Ef = la.angular_spectrum_propagate_mft(E, zt, lam, dx, 0.05e-6, 1024)
    pk = float(np.max(np.abs(Ef)**2))
    if dz_um == 0:
        w_n, fw_n, ee_n = fine_metrics(Ef, 0.05e-6)
        print(f"AT MSoP: w={w_n*1e6:.3f}um FWHM={fw_n*1e6:.3f}um "
              f"EE3={ee_n[3]*100:.1f}% EE6={ee_n[6]*100:.1f}% EE12={ee_n[12]*100:.1f}%")
    if pk > best[1]:
        best = (dz_um, pk, Ef)
w_b, fw_b, ee_b = fine_metrics(best[2], 0.05e-6)
print(f"BEST dz={best[0]:+d}um from MSoP: w={w_b*1e6:.3f}um FWHM={fw_b*1e6:.3f}um "
      f"EE3={ee_b[3]*100:.1f}% EE6={ee_b[6]*100:.1f}% EE12={ee_b[12]*100:.1f}%")
print(f"(Zemax 2.736um waist / stigmatic exp20 2.97um at plane)  total {time.time()-t0:.0f}s")
