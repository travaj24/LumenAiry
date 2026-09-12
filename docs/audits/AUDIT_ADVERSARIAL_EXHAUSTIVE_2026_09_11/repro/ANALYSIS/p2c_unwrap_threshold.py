"""ANALYSIS probe 2c: minimal repro + threshold of the wave_opd_2d column-slip bug.

Pupil = flat circular aperture + A waves (PV) of primary coma.  No defocus.
Sampling: 400 samples across the pupil -> max phase step is tiny.
"""
import sys, warnings, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Free_Space_Optics" if False else
                r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.analysis.opd import wave_opd_2d
from lumenairy.analysis.zernike import zernike_decompose

lam = 633e-9; k0 = 2*np.pi/lam
N, dx, ap = 512, 1e-6, 400e-6
x = (np.arange(N)-N/2)*dx
X, Y = np.meshgrid(x, x)
R = np.sqrt(X**2+Y**2); rho = R/(ap/2); th = np.arctan2(Y, X)
inside = R <= ap/2
# OSA (3,1) horizontal coma, RMS-normalised: sqrt(8)(3rho^3-2rho)cos(th)
coma = np.sqrt(8)*(3*rho**3 - 2*rho)*np.cos(th)

print(" A[waves rms]  max|dphi|/sample   max|OPD err|   rms err   %pix wrong   Zernike c8 err")
print(" " + "-"*88)
for A in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0):
    opd_true = A*lam*coma
    E = inside*np.exp(1j*k0*opd_true)
    ph = k0*opd_true
    gx = np.abs(np.diff(ph, axis=1))[inside[:, :-1] & inside[:, 1:]].max()
    gy = np.abs(np.diff(ph, axis=0))[inside[:-1] & inside[1:]].max()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, _, opd = wave_opd_2d(E, dx, lam, aperture=ap)
    m = np.isfinite(opd)
    d = opd - opd_true
    d = d - np.median(d[m])
    pct = 100*np.mean(np.abs(d[m]) > 0.4*lam)
    # Zernike fit of the returned map vs the truth
    c, names = zernike_decompose(np.where(m, opd, np.nan), dx, ap, n_modes=15)
    print(f"  {A:6.2f}      {max(gx,gy):8.4f} rad   {np.abs(d[m]).max()/lam:9.3f} w "
          f"{np.sqrt((d[m]**2).mean())/lam:8.3f} w  {pct:7.2f}%   "
          f"c8={c[8]/lam:+8.4f} w (true {A:+.4f})")

print()
print("=== Same, but with f_ref supplied (the optimizer's call pattern) ===")
f = 0.02
for A in (0.2, 0.5, 1.0, 2.0):
    opd_true = -R**2/(2*f) + A*lam*coma
    E = inside*np.exp(1j*k0*opd_true)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, _, opd = wave_opd_2d(E, dx, lam, aperture=ap, focal_length=f, f_ref=f)
    m = np.isfinite(opd)
    d = opd - opd_true; d = d - np.median(d[m])
    c, _ = zernike_decompose(np.where(m, opd, np.nan), dx, ap, n_modes=15)
    print(f"  coma={A:4.2f} w rms -> max|err|={np.abs(d[m]).max()/lam:8.3f} w, "
          f"%wrong={100*np.mean(np.abs(d[m])>0.4*lam):6.2f}%, "
          f"fitted c8={c[8]/lam:+8.4f} w (true {A:+.4f}), "
          f"fitted RMS-WFE={np.sqrt((c[1:]**2).sum())/lam:.4f} w (true {A:.4f})")

print()
print("=== Proposed fix check: mask-aware unwrap (skimage) vs current ===")
try:
    from skimage.restoration import unwrap_phase
    have_sk = True
except Exception as e:
    have_sk = False
    print("  skimage not available:", e)
A = 1.0
opd_true = A*lam*coma
E = inside*np.exp(1j*k0*opd_true)
ph_w = np.angle(E)
# current algorithm
cur = np.unwrap(np.unwrap(ph_w, axis=1), axis=0)/k0
dc = (cur-opd_true); dc -= np.median(dc[inside])
print(f"  current row-then-col : max|err| = {np.abs(dc[inside]).max()/lam:.3f} waves")
if have_sk:
    ma = np.ma.array(ph_w, mask=~inside)
    fx = np.asarray(unwrap_phase(ma))/k0
    dfx = (fx-opd_true); dfx -= np.median(dfx[inside])
    print(f"  skimage masked unwrap: max|err| = {np.abs(dfx[inside]).max()/lam:.3e} waves")
# cheap in-library fix: unwrap the CENTRAL column first, then each row
def fix_unwrap(ph, mask):
    out = np.array(ph, dtype=float)
    Nr, Nc = ph.shape
    jc = Nc//2
    col = np.unwrap(ph[:, jc])
    out2 = np.unwrap(ph, axis=1)              # rows, each with own offset
    # re-anchor each row so out2[:, jc] matches the unwrapped centre column
    out2 = out2 + (col - out2[:, jc])[:, None]
    return out2
fx2 = fix_unwrap(ph_w, inside)/k0
d2 = (fx2-opd_true); d2 -= np.median(d2[inside])
print(f"  centre-column anchor : max|err| = {np.abs(d2[inside]).max()/lam:.3e} waves")
