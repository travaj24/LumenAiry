"""RW test 2: (a) first-zero convergence vs focal sampling,
                (b) NA=0.9 longitudinal energy fraction."""
import sys, gc, warnings
import numpy as np
from scipy.special import j1, jn_zeros
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.vector_diffraction import richards_wolf_focus

lam = 633e-9


def first_zero(I, xf, yf, k, NA):
    Xf, Yf = np.meshgrid(xf, yf)
    Rf = np.hypot(Xf, Yf)
    r1 = jn_zeros(1, 1)[0]/(k*NA)
    In = I/I.max()
    nb, rmax = 1200, 3*r1
    bins = np.linspace(0, rmax, nb+1)
    idx = np.digitize(Rf.ravel(), bins)-1
    ok = (idx >= 0) & (idx < nb)
    prof = np.bincount(idx[ok], weights=In.ravel()[ok], minlength=nb)
    cnt = np.bincount(idx[ok], minlength=nb)
    good = cnt > 0
    p = (prof/np.maximum(cnt, 1))[good]
    rr = (0.5*(bins[:-1]+bins[1:]))[good]
    for i in range(2, len(p)-2):
        if p[i] < p[i-1] and p[i] <= p[i+1] and p[i] < 0.05:
            d = p[i-1]-2*p[i]+p[i+1]
            s = 0.5*(p[i-1]-p[i+1])/d if d else 0.0
            return rr[i]+s*(rr[i+1]-rr[i]), r1
    return np.nan, r1


print("=== (a) low-NA first-zero convergence, identical pupil ===")
NA, f, Np, dxp = 0.05, 5e-3, 512, 2e-6
k = 2*np.pi/lam
x = (np.arange(Np)-Np/2)*dxp
X, Y = np.meshgrid(x, x)
pupil = (np.hypot(X, Y) <= f*NA).astype(np.complex128)
for Nf in (1024, 2048, 4096):
    Ex, Ey, Ez, xf, yf = richards_wolf_focus(pupil, lam, NA, f, dxp,
                                             N_focal=Nf, polarization='x')
    I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
    r0, r1 = first_zero(I, xf, yf, k, NA)
    dxf = lam*f/(Nf*dxp)
    print(f"  N_focal={Nf:5d}  dx_focal={dxf*1e9:7.1f} nm  "
          f"r0/(lam/NA)={r0/(lam/NA):.5f}  (theory 0.60976)  "
          f"rel err {abs(r0-r1)/r1:.3e}")
    del Ex, Ey, Ez, I; gc.collect()

print()
print("=== (b) NA=0.9 (n=1) uniform aplanatic pupil, x-pol ===")
NA, f = 0.9, 1e-3
for (Np, dxp, Nf) in ((512, 4e-6, 1024), (512, 4e-6, 2048), (1024, 2e-6, 2048)):
    x = (np.arange(Np)-Np/2)*dxp
    X, Y = np.meshgrid(x, x)
    pupil = (np.hypot(X, Y) <= f*NA).astype(np.complex128)
    half = Np*dxp/2
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        Ex, Ey, Ez, xf, yf = richards_wolf_focus(pupil, lam, NA, f, dxp,
                                                 N_focal=Nf, polarization='x')
    ix = np.abs(Ex)**2; iy = np.abs(Ey)**2; iz = np.abs(Ez)**2
    tot = ix.sum()+iy.sum()+iz.sum()
    dxf = lam*f/(Nf*dxp)
    # energy fractions over the whole focal-plane window
    print(f"  Np={Np} dxp={dxp*1e6:.1f}um half={half*1e6:.0f}um rim={f*NA*1e6:.0f}um "
          f"N_focal={Nf} dx_focal={dxf*1e9:.1f}nm")
    print(f"     whole-window  |Ex|^2 {ix.sum()/tot:.4f}  |Ey|^2 {iy.sum()/tot:.5f}"
          f"  |Ez|^2 {iz.sum()/tot:.4f}")
    # restricted to the central Airy disc r < 0.61 lam/NA and to r < 2 lam
    Xf, Yf = np.meshgrid(xf, yf); Rf = np.hypot(Xf, Yf)
    for lab, m in (("r<0.61lam/NA", Rf <= 0.61*lam/NA),
                   ("r<1.0 lam   ", Rf <= lam),
                   ("r<3.0 lam   ", Rf <= 3*lam)):
        t = ix[m].sum()+iy[m].sum()+iz[m].sum()
        print(f"     {lab}  |Ex|^2 {ix[m].sum()/t:.4f}  |Ey|^2 {iy[m].sum()/t:.5f}"
              f"  |Ez|^2 {iz[m].sum()/t:.4f}")
    c = Nf//2
    print(f"     on-axis focal point: |Ex|^2 {ix[c,c]:.4e} |Ez|^2 {iz[c,c]:.4e}  "
          f"peak|Ez|^2/peak|Ex|^2 = {iz.max()/ix.max():.4f}")
    del Ex, Ey, Ez, ix, iy, iz; gc.collect()
