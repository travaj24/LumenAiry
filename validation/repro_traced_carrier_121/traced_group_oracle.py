# Per-group fidelity audit of apply_real_lens_traced(carrier=R_in) on the 121:
# for each group, launch the chain's actual carrier Gaussian (w, R_in from the
# physical q-trace), apply the traced group, and compare the exit field
# POINTWISE against an exact meridional raytrace oracle (lumenairy-free):
# rays from the carrier point source through the group's spherical surfaces,
# exact eikonal to the exit vertex plane.
# Reports per group: rms residual phase over the beam, best-fit exit-curvature
# error vs ABCD, r^4 (aberration) residual, and high-frequency (scramble) rms.
import ast, re, sys, warnings
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
k0 = 2*np.pi/lam
w0 = 4e-6
OBJ_GAP = 47.906284e-3

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
lens_events = [ev for ev in events if ev['type'] == 'lens']
print("prescription keys of group 1:", sorted(lens_events[0]['prescription'].keys()))

def n_of(g):
    return 1.0 if g in (None, '', 'air', 'AIR') else float(get_glass_index(g, lam))

# physical q-trace for (w, R_in) at each group front vertex + R_out ABCD
elements = rx['elements']; th = rx['thicknesses']
q = 1j*np.pi*w0*w0/lam
gap = OBJ_GAP
ginfo = {}
def wR(q):
    iq = 1.0/q
    R = np.inf if abs(iq.real) < 1e-30 else 1.0/iq.real
    return np.sqrt(-lam/(np.pi*iq.imag)), R
for i, el in enumerate(elements):
    n1 = n_of(el['glass_before']); n2 = n_of(el['glass_after'])
    q = q + gap
    for ev in lens_events:
        if el['surf_num'] == ev['surf_start']:
            w_, R_ = wR(q)
            ginfo[ev['name']] = {'w': w_, 'R_in': R_}
    R_s = el['radius'] if el['radius'] else np.inf
    if el['surf_num'] in (9, 11): R_s = np.inf
    if n1 != n2 or np.isfinite(R_s):
        pw = 0.0 if not np.isfinite(R_s) else (n2 - n1)/R_s
        q = n2/(n1*(1.0/q) - pw)
    for ev in lens_events:
        if el['surf_num'] == ev['surf_end']:
            ginfo[ev['name']]['R_out'] = wR(q)[1]
    gap = th[i] if i < len(th) else 0.0

# geometry per group for the oracle: (z_vertex_rel, R_s, n1, n2) per surface
def group_geometry(ev):
    surfs = []
    z = 0.0
    for i, el in enumerate(elements):
        if ev['surf_start'] <= el['surf_num'] <= ev['surf_end']:
            R_s = el['radius'] if el['radius'] else np.inf
            surfs.append((z, R_s, n_of(el['glass_before']), n_of(el['glass_after'])))
            z += th[i] if i < len(th) else 0.0
    z_exit = surfs[-1][0]
    return surfs, z_exit

def oracle_phase(surfs, z_exit, R_in, x_max, npts=600):
    """Exact meridional trace: entrance height x0 -> (x_exit, phi_total)."""
    x0s = np.linspace(1e-9, x_max, npts)
    xe = np.full(npts, np.nan); ph = np.full(npts, np.nan)
    for j, x0 in enumerate(x0s):
        P = np.array([x0, 0.0])
        if np.isinf(R_in):
            u = np.array([0.0, 1.0])
        else:
            u = np.array([x0, R_in]) / np.hypot(x0, R_in)
            if R_in < 0:
                u = np.array([-x0, -R_in]) / np.hypot(x0, R_in)
        opl = (np.sign(R_in)*(np.hypot(x0, R_in) - abs(R_in))
               if np.isfinite(R_in) else 0.0)   # entrance spherical eikonal
        ok = True
        for (zv, R_s, n1, n2) in surfs:
            if np.isfinite(R_s):
                C = np.array([0.0, zv + R_s])
                oc = P - C
                b = np.dot(oc, u); c = np.dot(oc, oc) - R_s*R_s
                disc = b*b - c
                if disc < 0: ok = False; break
                t1, t2 = -b - np.sqrt(disc), -b + np.sqrt(disc)
                # nearest-to-vertex intersection; SIGNED t (eikonal
                # continuation: a concave front surface lies behind the
                # vertex plane where the input field is defined)
                t = min((t1, t2), key=lambda tt: abs((P + tt*u)[1] - zv))
                P2 = P + t*u
                nv = np.sign(R_s)*(C - P2)/np.linalg.norm(C - P2)
            else:
                if abs(u[1]) < 1e-12: ok = False; break
                t = (zv - P[1])/u[1]
                P2 = P + t*u
                nv = np.array([0.0, 1.0])
            opl += n1*t
            c1 = np.dot(u, nv)
            eta = n1/n2
            disc2 = 1.0 - eta*eta*(1.0 - c1*c1)
            if disc2 < 0: ok = False; break
            u = eta*u + (np.sqrt(disc2) - eta*c1)*nv
            u = u/np.linalg.norm(u)
            P = P2
        if not ok: continue
        n_last = surfs[-1][3]
        t = (z_exit - P[1])/u[1]
        opl += n_last*t
        xe[j] = (P + t*u)[0]
        ph[j] = k0*opl
    good = np.isfinite(xe)
    xe, ph = xe[good], ph[good]
    # axial reference: extrapolate to x_exit->0 with a quadratic in x
    p2 = np.polyfit(xe[:20], ph[:20], 2)
    return xe, ph - np.polyval(p2, 0.0)

N = 2048
VARIANTS = [{}]                    # default: carrier=R_in, subsample 4
if len(sys.argv) > 1 and sys.argv[1] == 'variants':
    # knob sensitivity on the worst group only (S5-S7)
    lens_events = [ev for ev in lens_events if ev['name'] == 'Lens S5-S7']
    VARIANTS = [
        {'label': 'carrier=R subsample=4'},
        {'label': 'carrier=R subsample=16', 'ray_subsample': 16},
        {'label': "carrier='auto'", 'carrier': 'auto'},
        {'label': 'carrier=R tilt_aware', 'tilt_aware_rays': True},
        {'label': 'NO carrier (plane-wave ref)', 'carrier': None},
    ]
print(f"\n{'group':<14}{'R_in mm':>10}{'w mm':>7}{'rms_res':>9}{'dRout %':>9}"
      f"{'rms_r4+':>9}{'rms_hf':>8}  (rad over r<w; hf = scramble)")
for ev in lens_events:
    gi = ginfo[ev['name']]
    w_, R_in, R_out = gi['w'], gi['R_in'], gi['R_out']
    surfs, z_exit = group_geometry(ev)
    dx = 4.2*w_/N * 2         # grid = 8.4w
    x = (np.arange(N) - N//2)*dx
    r2g = x[None,:]**2 + x[:,None]**2
    env = np.exp(-r2g/w_**2)
    Sin = np.sign(R_in)*(np.sqrt(r2g + R_in*R_in) - abs(R_in)) if np.isfinite(R_in) else 0.0
    E_in = (env*np.exp(1j*k0*Sin)).astype(np.complex128)
    xe, ph = oracle_phase(surfs, z_exit, R_in, x_max=3.2*w_)
    for var in VARIANTS:
      kw = dict(ray_subsample=var.get('ray_subsample', 4), n_workers=8)
      if 'carrier' in var:
          if var['carrier'] is not None:
              kw['carrier'] = var['carrier']
      else:
          kw['carrier'] = R_in
      if var.get('tilt_aware_rays'):
          kw['tilt_aware_rays'] = True
      E_out = la.apply_real_lens_traced(
          E_in, prescription=ev['prescription'], wavelength=lam, dx=dx, **kw)
      if 'label' in var:
          print(f"  --- {var['label']}")
      # oracle phase on the radial grid
      rr = np.sqrt(r2g)
      ph_o = np.interp(rr, xe, ph, left=ph[0], right=np.nan)
      res = np.angle(E_out * np.exp(-1j*ph_o))
      mask = (rr < w_) & np.isfinite(ph_o) & (np.abs(E_out) > 0.05*np.abs(E_out).max())
      res0 = res - np.median(res[rr < 0.05*w_])   # piston off
      resm = res0[mask]
      rms = np.sqrt(np.mean(np.angle(np.exp(1j*(resm - np.angle(np.mean(np.exp(1j*resm))))))**2))
      row = res0[N//2, :]
      selr = (np.abs(x) < 0.5*w_)
      rowu = np.unwrap(row[selr])
      c2 = np.polyfit(x[selr], rowu, 2)[0]        # res ~ c2 x^2
      dCurv = 2*c2/k0                              # delta(1/R)
      dRout_pct = 100*abs(dCurv*R_out) if np.isfinite(R_out) else 0.0
      row4 = rowu - np.polyval(np.polyfit(x[selr], rowu, 2), x[selr])
      rms4 = np.std(row4)
      ker = np.ones(21)/21
      hf = rowu - np.convolve(rowu, ker, mode='same')
      rms_hf = np.std(hf[10:-10])
      print(f"{ev['name']:<14}{R_in*1e3:>10.2f}{w_*1e3:>7.2f}{rms:>9.3f}{dRout_pct:>9.3f}"
            f"{rms4:>9.3f}{rms_hf:>8.3f}", flush=True)
