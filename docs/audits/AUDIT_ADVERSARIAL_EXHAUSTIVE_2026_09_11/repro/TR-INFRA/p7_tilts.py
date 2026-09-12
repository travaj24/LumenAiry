"""TR-INFRA probe 7: _sample_local_tilts sign, half-pixel centring, np.roll
wrap contamination, clipping, amplitude weighting."""
import numpy as np
from lumenairy.elements import _lens_traced as T

lam = 1.31e-6; k0 = 2*np.pi/lam
N = 256; dx = 4e-6
x = (np.arange(N) - N/2)*dx
X, Y = np.meshgrid(x, x, indexing='xy')       # row = y, col = x

print("=== 7a  TILTED PLANE WAVE (uniform amplitude) -- sign + wrap pollution ===")
L0, M0 = 0.03, -0.02
E = np.exp(1j*k0*(L0*X + M0*Y)).astype(np.complex128)
for sig in (0.0, 4.0):
    L, M = T._sample_local_tilts(E, lam, dx, X, Y, smooth_sigma_px=sig)
    core = (np.abs(X) < 0.35*N*dx/2) & (np.abs(Y) < 0.35*N*dx/2)
    print(f"  sigma={sig}: core max|L-L0|={np.abs(L-L0)[core].max():.3e} "
          f"max|M-M0|={np.abs(M-M0)[core].max():.3e}  "
          f"WHOLE-GRID max|L-L0|={np.abs(L-L0).max():.3e} (L0={L0})")
    # how far the wrap contamination reaches in from the edge
    bad = np.abs(L-L0) > 0.01*abs(L0)
    if bad.any():
        cols = np.where(bad.any(axis=0))[0]
        rows = np.where(bad.any(axis=1))[0]
        print(f"           contaminated columns {cols.min()}..{cols.max()} of {N},"
              f" rows {rows.min()}..{rows.max()}  ({100*bad.mean():.1f}% of pixels)")

print("\n=== 7b  UNIFORM-AMPLITUDE SPHERICAL wave -- half-pixel centring bias ===")
for Rc in (0.10, 0.05):
    r2 = X**2+Y**2
    Wt = np.sqrt(r2+Rc*Rc)-abs(Rc)
    E = np.exp(1j*k0*Wt)*np.exp(-r2/(0.20e-3)**2)
    L, M = T._sample_local_tilts(E, lam, dx, X, Y, smooth_sigma_px=0.0)
    Lt = X/np.sqrt(r2+Rc*Rc)
    # the exact FORWARD-difference reading at pixel i estimates dW/dx at i+1/2
    Lt_mid = (np.sqrt((X+dx)**2+Y**2+Rc*Rc)-np.sqrt(r2+Rc*Rc))/dx
    m = (np.abs(X)<0.15e-3)&(np.abs(Y)<0.15e-3)&(np.abs(X)>0.02e-3)
    print(f"  R={Rc} m: mean (L_sampled - L_true_at_pixel) = {np.mean((L-Lt)[m]):+.4e}"
          f"   predicted half-pixel bias dx/(2R) = {dx/(2*Rc):+.4e}")
    print(f"            mean (L_sampled - L_true_at_MIDPOINT) = {np.mean((L-Lt_mid)[m]):+.4e}"
          f"   (typical |L| = {np.abs(Lt)[m].mean():.3e}, rel bias = "
          f"{np.mean((L-Lt)[m])/np.abs(Lt)[m].mean()*100:+.3f} %)")

print("\n=== 7c  clipping to max_sin and amplitude weighting ===")
E = (np.exp(1j*k0*(0.8*X))*np.exp(-(X**2+Y**2)/(0.2e-3)**2)).astype(np.complex128)
L, M = T._sample_local_tilts(E, lam, dx, X, Y, max_sin=0.5, smooth_sigma_px=0.0)
print(f"  launch tilt 0.8 (aliased at dx={dx*1e6} um, Nyquist sin = {lam/(2*dx):.3f}):"
      f"  returned max|L| = {np.abs(L).max():.4f} (clip at 0.5) -- silently WRONG, no warning")
