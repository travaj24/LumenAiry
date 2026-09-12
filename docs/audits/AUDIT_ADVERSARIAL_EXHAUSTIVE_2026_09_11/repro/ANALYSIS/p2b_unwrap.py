"""ANALYSIS probe 2b: dissect the wave_opd_2d row-then-column unwrap failure."""
import sys, warnings, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.analysis.opd import wave_opd_2d

lam = 633e-9; k0 = 2*np.pi/lam

def build(N, dx, f, ap, extra=None, pad_zero=True):
    x = (np.arange(N)-N/2)*dx
    X, Y = np.meshgrid(x, x)
    R2 = X**2+Y**2
    ph = -k0*R2/(2*f)
    if extra is not None:
        ph = ph + extra(X, Y)
    amp = (R2 <= (ap/2)**2).astype(float) if pad_zero else np.ones_like(R2)
    return amp*np.exp(1j*ph), X, Y, ph

def report(tag, N, dx, f, ap, extra=None, aperture_arg=True, f_ref=None):
    E, X, Y, ph = build(N, dx, f, ap, extra)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, _, opd = wave_opd_2d(E, dx, lam, aperture=(ap if aperture_arg else None), f_ref=f_ref)
    exact = ph/k0
    m = np.isfinite(opd)
    d = (opd-exact)
    d = d - np.median(d[m])
    dw = d/lam
    print(f"{tag}")
    print(f"   max|err| = {np.abs(dw[m]).max():.4f} waves ; "
          f"is integer? {np.allclose(np.round(dw[m]), dw[m], atol=1e-6)}")
    # structure: is the error constant per column? per row?
    W = np.where(m, np.round(dw), np.nan)
    ncol_uniform = 0; ncol_tot = 0
    for j in range(N):
        v = W[:, j]; v = v[np.isfinite(v)]
        if v.size:
            ncol_tot += 1
            if np.all(v == v[0]): ncol_uniform += 1
    nrow_uniform = 0; nrow_tot = 0
    for i in range(N):
        v = W[i, :]; v = v[np.isfinite(v)]
        if v.size:
            nrow_tot += 1
            if np.all(v == v[0]): nrow_uniform += 1
    print(f"   columns with a single constant wave-offset: {ncol_uniform}/{ncol_tot}")
    print(f"   rows    with a single constant wave-offset: {nrow_uniform}/{nrow_tot}")
    vals, cnts = np.unique(W[m], return_counts=True)
    print(f"   offset histogram (waves): {dict(zip(vals.astype(int).tolist(), cnts.tolist()))}")
    # max per-sample phase step inside the pupil (the Nyquist quantity)
    gx = np.abs(np.diff(ph, axis=1))[m[:, :-1] & m[:, 1:]]
    gy = np.abs(np.diff(ph, axis=0))[m[:-1, :] & m[1:, :]]
    print(f"   max in-pupil |dphi| per sample: x={gx.max():.4f} rad  y={gy.max():.4f} rad  (pi={np.pi:.4f})")
    print()

ap = 400e-6
print("--- 1. pure defocus, 20 waves edge OPD, dx = dx_max/4 (4x Nyquist margin) ---")
f = (ap/2)**2/(2*20*lam); dxm = lam*f/ap; dx = dxm/4
N = int(np.ceil(ap/dx/2))*2+64
report(f"  N={N} dx={dx*1e6:.3f}um f={f*1e3:.3f}mm", N, dx, f, ap)

print("--- 2. same but grid EXACTLY the aperture (no zero border) ---")
N2 = int(np.ceil(ap/dx/2))*2
report(f"  N={N2}", N2, dx, f, ap)

print("--- 3. defocus + 3 waves trefoil (aberrated), f=2 m, dx=1um, N=512 ---")
def tre(X, Y):
    rho = np.sqrt(X**2+Y**2)/(ap/2); th = np.arctan2(Y, X)
    return k0*3*lam*np.sqrt(8)*rho**3*np.cos(3*th)
report("  ", 512, 1e-6, 2.0, ap, extra=tre)

print("--- 4. same as 3 but NO defocus (trefoil only) ---")
report("  ", 512, 1e-6, 1e12, ap, extra=tre)

print("--- 5. plain 15.8-wave defocus f=2m dx=1um N=512 (probe E geometry) ---")
report("  ", 512, 1e-6, 2.0, ap)
