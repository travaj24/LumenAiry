"""TR-INFRA probe 8: _input_beam_amp_radius, _carrier_residual_rms,
_input_tilt_stats, _fit_residual_eikonal, ndarray-carrier index rounding."""
import numpy as np
from lumenairy.elements import _lens_traced as T

lam=1.31e-6; k0=2*np.pi/lam
N=512; dx=8e-6
x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x,indexing='xy')

print("=== 8a  _input_beam_amp_radius: w = sqrt(2<r^2>) on a known Gaussian ===")
for w0 in (0.3e-3, 0.8e-3):
    for xc in (0.0, 0.5e-3):
        E=np.exp(-((X-xc)**2+Y**2)/w0**2)
        w_o = T._input_beam_amp_radius(E, dx)
        w_c = T._input_beam_amp_radius(E, dx, centre=(xc,0.0))
        print(f"  w0={w0*1e3:.2f} mm xc={xc*1e3:.2f} mm: about ORIGIN {w_o*1e3:.5f} mm"
              f" (analytic sqrt(2xc^2+w0^2)={np.sqrt(2*xc**2+w0**2)*1e3:.5f})"
              f"   about CENTRE {w_c*1e3:.5f} mm (true {w0*1e3:.5f})")

print("\n=== 8b  _carrier_residual_rms / _input_tilt_stats on a tilted plane wave ===")
E=np.exp(1j*k0*(0.03*X-0.02*Y))*np.exp(-(X**2+Y**2)/(0.8e-3)**2)
print(f"  _carrier_residual_rms(no carrier) = {T._carrier_residual_rms(E,None,lam,dx):.6f}"
      f"   (expect hypot(0.03,0.02)={np.hypot(0.03,0.02):.6f})")
ts=T._input_tilt_stats(E,lam,dx)
print(f"  _input_tilt_stats = tilt_rms {ts[0]:.6f}, coherence_ratio {ts[1]:.6f} (expect ~1)")
W = 0.03*X-0.02*Y
print(f"  _carrier_residual_rms(with the exact carrier) = "
      f"{T._carrier_residual_rms(E,W,lam,dx):.3e}  (expect ~0)")

print("\n=== 8c  ndarray-carrier grad lookup: FLOOR vs NEAREST index ===")
Rc=0.05; r2=X**2+Y**2
Wg = np.sqrt(r2+Rc*Rc)-Rc
_,gfn,wfn = T._compute_carrier(Wg, E, lam, dx, X, Y)
xq=np.linspace(-0.6e-3,0.6e-3,4001); yq=np.zeros_like(xq)
L,_=gfn(xq,yq)
Lt = xq/np.sqrt(xq**2+Rc**2)
# the implementation uses .astype(int64) (TRUNCATION), so the sampled index is
# floor(xq/dx + N/2) -- compare against both conventions
idx_floor=np.clip((xq/dx+N/2).astype(np.int64),0,N-1)
idx_round=np.clip(np.rint(xq/dx+N/2).astype(np.int64),0,N-1)
xs_floor=(idx_floor-N/2)*dx; xs_round=(idx_round-N/2)*dx
print(f"  mean (x_sampled - x_query)/dx  with the shipped .astype(int64): "
      f"{np.mean(xs_floor-xq)/dx:+.4f} px   with np.rint: {np.mean(xs_round-xq)/dx:+.4f} px")
print(f"  mean (L_returned - L_true) = {np.mean(L-Lt):+.4e}"
      f"   (half-pixel prediction -0.5*dx/R = {-0.5*dx/Rc:+.4e})")
print(f"  max |L_returned - L_true|  = {np.abs(L-Lt).max():.4e}  (|L| span {np.abs(Lt).max():.4e})")

print("\n=== 8d  _fit_residual_eikonal: recovery of a KNOWN r^4 residual ===")
w0=0.4e-3
a_true = 2.0e-6*((X**2+Y**2)/w0**2)**2        # metres, r^4 residual
Wc = np.zeros_like(X)
E2 = np.exp(-(X**2+Y**2)/w0**2)*np.exp(1j*k0*(Wc+a_true))
m = T._fit_residual_eikonal(E2, Wc, lam, dx, dx, (0.0,0.0), w0)
print(f"  model: {None if m is None else m.diag}")
if m is not None:
    xs2=np.linspace(-2*w0,2*w0,201); Xs,Ys=np.meshgrid(xs2,xs2)
    a_fit=m.value(Xs,Ys); a_tr=2.0e-6*((Xs**2+Ys**2)/w0**2)**2
    # the potential is fitted from GRADIENTS, so it is known only up to a piston
    d=(a_fit-a_tr); d=d-d[100,100]
    disc=(Xs**2+Ys**2)<=(2*w0)**2
    print(f"  max|a_fit - a_true| (piston-removed) over the fit disc = {np.abs(d)[disc].max():.4e} m"
          f" = {np.abs(d)[disc].max()/lam:.5f} waves   (a_true PV = {a_tr[disc].max():.3e} m)")
    gx,gy=m.grad(Xs,Ys)
    gtx = 2.0e-6*4*(Xs**2+Ys**2)*Xs/w0**4
    print(f"  max|grad_x err| = {np.abs(gx-gtx)[disc].max():.4e} (|grad| span {np.abs(gtx)[disc].max():.4e})")
    # RADIAL FREEZE continuity
    r=np.sqrt(Xs**2+Ys**2); shell=np.abs(r-m.r_fit)<2e-6
    print(f"  r_fit={m.r_fit*1e3:.4f} mm r_freeze={m.diag['r_freeze']*1e3:.4f} mm;"
          f"  value is C0 across the freeze: {np.isfinite(a_fit[shell]).all()}")
