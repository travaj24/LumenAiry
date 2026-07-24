# Per-group dx-convergence probe (AUDIT_TRACED_PRODUCTION_READINESS
# 2026-07-24 P0 follow-on; recommended by AUDIT_TRACED_CHAIN_DX_SCALING
# 2026-07-22 F-B): for a chosen 121 group, hold the physical window (8.4 w),
# the launch (carrier Gaussian from the q-trace), and the PHYSICAL ray pitch
# fixed, and sweep the GRID PITCH ALONE (N = 512..8192, dx = 8.4w/N, ray
# subsample scaled with N).  Compare the traced exit POINTWISE against the
# exact meridional-ray oracle at every dx.  If the residual GROWS as dx -> 0,
# the F-B divergence lives inside apply_real_lens_traced; flat residuals
# would clear the element.
#
# Env knobs: GROUPS = comma list of group names to probe (default
# 'Lens S3-S4,Lens S5-S7'); NS = comma list of N values (default
# '512,1024,2048,4096,8192').
import ast, os, re, sys, time, warnings
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
k0 = 2 * np.pi / lam
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

def n_of(g):
    return 1.0 if g in (None, '', 'air', 'AIR') else float(get_glass_index(g, lam))

elements = rx['elements']; th = rx['thicknesses']
q = 1j * np.pi * w0 * w0 / lam
gap = OBJ_GAP
ginfo = {}
def wR(q):
    iq = 1.0 / q
    R = np.inf if abs(iq.real) < 1e-30 else 1.0 / iq.real
    return np.sqrt(-lam / (np.pi * iq.imag)), R
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
        pw = 0.0 if not np.isfinite(R_s) else (n2 - n1) / R_s
        q = n2 / (n1 * (1.0 / q) - pw)
    for ev in lens_events:
        if el['surf_num'] == ev['surf_end']:
            ginfo[ev['name']]['R_out'] = wR(q)[1]
    gap = th[i] if i < len(th) else 0.0

def group_geometry(ev):
    surfs = []
    z = 0.0
    for i, el in enumerate(elements):
        if ev['surf_start'] <= el['surf_num'] <= ev['surf_end']:
            R_s = el['radius'] if el['radius'] else np.inf
            surfs.append((z, R_s, n_of(el['glass_before']),
                          n_of(el['glass_after'])))
            z += th[i] if i < len(th) else 0.0
    return surfs, surfs[-1][0]

def oracle_phase(surfs, z_exit, R_in, x_max, npts=600):
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
        opl = (np.sign(R_in) * (np.hypot(x0, R_in) - abs(R_in))
               if np.isfinite(R_in) else 0.0)
        ok = True
        for (zv, R_s, n1, n2) in surfs:
            if np.isfinite(R_s):
                C = np.array([0.0, zv + R_s])
                oc = P - C
                b = np.dot(oc, u); c = np.dot(oc, oc) - R_s * R_s
                disc = b * b - c
                if disc < 0: ok = False; break
                t1, t2 = -b - np.sqrt(disc), -b + np.sqrt(disc)
                t = min((t1, t2), key=lambda tt: abs((P + tt * u)[1] - zv))
                P2 = P + t * u
                nv = np.sign(R_s) * (C - P2) / np.linalg.norm(C - P2)
            else:
                if abs(u[1]) < 1e-12: ok = False; break
                t = (zv - P[1]) / u[1]
                P2 = P + t * u
                nv = np.array([0.0, 1.0])
            opl += n1 * t
            c1 = np.dot(u, nv)
            eta = n1 / n2
            disc2 = 1.0 - eta * eta * (1.0 - c1 * c1)
            if disc2 < 0: ok = False; break
            u = eta * u + (np.sqrt(disc2) - eta * c1) * nv
            u = u / np.linalg.norm(u)
            P = P2
        if not ok: continue
        n_last = surfs[-1][3]
        t = (z_exit - P[1]) / u[1]
        opl += n_last * t
        xe[j] = (P + t * u)[0]
        ph[j] = k0 * opl
    good = np.isfinite(xe)
    xe, ph = xe[good], ph[good]
    p2 = np.polyfit(xe[:20], ph[:20], 2)
    return xe, ph - np.polyval(p2, 0.0)

GROUPS = os.environ.get('GROUPS', 'Lens S3-S4,Lens S5-S7').split(',')
NS = [int(s) for s in os.environ.get('NS', '512,1024,2048,4096,8192').split(',')]
REF_N, REF_RS = 2048, 4          # the oracle script's validated config

print(f"\n{'group':<14}{'N':>6}{'dx um':>9}{'rs':>4}{'ray-pitch um':>13}"
      f"{'rms_res':>9}{'dRout %':>9}{'rms_r4+':>9}{'rms_hf':>8}{'P_out/P_in':>11}"
      f"{'w_out/w_in':>11}  (rad over r<w)")
for name in GROUPS:
    ev = next(e for e in lens_events if e['name'] == name)
    gi = ginfo[name]
    w_, R_in, R_out = gi['w'], gi['R_in'], gi['R_out']
    surfs, z_exit = group_geometry(ev)
    xe, ph = oracle_phase(surfs, z_exit, R_in, x_max=3.2 * w_)
    for N in NS:
        dx = 8.4 * w_ / N
        rs = max(1, int(round(REF_RS * N / REF_N)))   # pitch-preserving
        x = (np.arange(N) - N // 2) * dx
        r2g = x[None, :] ** 2 + x[:, None] ** 2
        env = np.exp(-r2g / w_ ** 2)
        Sin = (np.sign(R_in) * (np.sqrt(r2g + R_in * R_in) - abs(R_in))
               if np.isfinite(R_in) else 0.0)
        E_in = (env * np.exp(1j * k0 * Sin)).astype(np.complex128)
        P_in = float(np.sum(np.abs(E_in) ** 2)) * dx * dx
        t0 = time.time()
        E_out = la.apply_real_lens_traced(
            E_in, prescription=ev['prescription'], wavelength=lam, dx=dx,
            carrier=R_in, ray_subsample=rs, n_workers=8)
        E_out = np.asarray(E_out)
        P_out = float(np.sum(np.abs(E_out) ** 2)) * dx * dx
        rr = np.sqrt(r2g)
        ph_o = np.interp(rr, xe, ph, left=ph[0], right=np.nan)
        res = np.angle(E_out * np.exp(-1j * ph_o))
        mask = ((rr < w_) & np.isfinite(ph_o)
                & (np.abs(E_out) > 0.05 * np.abs(E_out).max()))
        res0 = res - np.median(res[rr < 0.05 * w_])
        resm = res0[mask]
        rms = np.sqrt(np.mean(np.angle(np.exp(1j * (
            resm - np.angle(np.mean(np.exp(1j * resm)))))) ** 2))
        row = res0[N // 2, :]
        selr = (np.abs(x) < 0.5 * w_)
        rowu = np.unwrap(row[selr])
        c2 = np.polyfit(x[selr], rowu, 2)[0]
        dCurv = 2 * c2 / k0
        dRout_pct = 100 * abs(dCurv * R_out) if np.isfinite(R_out) else 0.0
        row4 = rowu - np.polyval(np.polyfit(x[selr], rowu, 2), x[selr])
        rms4 = np.std(row4)
        ker = np.ones(21) / 21
        hf = rowu - np.convolve(rowu, ker, mode='same')
        rms_hf = np.std(hf[10:-10])
        # exit-amplitude second-moment radius vs entrance (blur indicator)
        I_o = np.abs(E_out) ** 2
        vx = float((I_o.sum(axis=0) * x ** 2).sum() / I_o.sum())
        I_i = np.abs(E_in) ** 2
        vxi = float((I_i.sum(axis=0) * x ** 2).sum() / I_i.sum())
        print(f"{name:<14}{N:>6}{dx*1e6:>9.3f}{rs:>4}{rs*dx*1e6:>13.3f}"
              f"{rms:>9.3f}{dRout_pct:>9.3f}{rms4:>9.3f}{rms_hf:>8.3f}"
              f"{P_out/P_in:>11.6f}{np.sqrt(vx/vxi):>11.6f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
