import numpy as np
from lumenairy.elements import _lens_traced as T
print("=== TiltedCarrier: W(x0,y0)==0, grad W(x0,y0)==(L,M), eikonal |grad W|<=1 ===")
for spec in (T.TiltedCarrier(R=0.05, L=0.05, M=-0.03, x0=1e-3, y0=-2e-3),
             T.TiltedCarrier(R=-0.05, L=0.2, M=0.1),
             T.TiltedCarrier(R=np.inf, L=0.05, M=0.02, x0=3e-4),
             T.TiltedCarrier(R=0.02, L=0.6, M=0.6)):
    try:
        Wq, Lq, Mq = T._tilted_carrier_parts(spec, np.array([float(spec.x0)]),
                                             np.array([float(spec.y0)]))
    except ValueError as e:
        print(f"  R={spec.R:+.4g} L={spec.L} M={spec.M}: raises -> {e}")
        continue
    print(f"  R={spec.R:+.4g} L={spec.L:+.3f} M={spec.M:+.3f}: W(x0,y0)={Wq[0]:+.3e}"
          f"  grad=({Lq[0]:+.12f},{Mq[0]:+.12f})  err=({Lq[0]-spec.L:+.2e},{Mq[0]-spec.M:+.2e})")
    xs=np.linspace(-2e-3,2e-3,9); Xg,Yg=np.meshgrid(xs,xs)
    Wg,Lg,Mg=T._tilted_carrier_parts(spec,Xg+spec.x0,Yg+spec.y0)
    print(f"     max |grad W| over a 4 mm patch = {np.hypot(Lg,Mg).max():.9f}  (must be <= 1)")
    # does W solve the eikonal equation exactly?  |grad W| should be exactly 1
    # only for a POINT-SOURCE eikonal measured along the true ray; check that
    # the analytic gradient equals a finite difference of W
    h=1e-9
    Wp,_,_=T._tilted_carrier_parts(spec,Xg+spec.x0+h,Yg+spec.y0)
    Wm,_,_=T._tilted_carrier_parts(spec,Xg+spec.x0-h,Yg+spec.y0)
    fd=(Wp-Wm)/(2*h)
    print(f"     max|L_analytic - dW/dx(FD)| = {np.abs(Lg-fd).max():.3e}"
          f"   (scale {np.abs(fd).max():.3e})")

print("\n=== C5 exact-eikonal vs pre-C5 sphere+ramp (fail-before flag) ===")
spec=T.TiltedCarrier(R=-0.02446, L=0.0549, M=0.0, x0=0.0, y0=0.0)
xs=np.linspace(-3.63e-3,3.63e-3,201); Xg,Yg=np.meshgrid(xs,xs)
W1,_,_=T._tilted_carrier_parts(spec,Xg,Yg)
T.TILTED_CARRIER_EXACT_EIKONAL=False
W0,_,_=T._tilted_carrier_parts(spec,Xg,Yg)
T.TILTED_CARRIER_EXACT_EIKONAL=True
lam=1.31e-6
print(f"  max|W_exact - W_sphere+ramp| = {np.abs(W1-W0).max():.4e} m"
      f" = {np.abs(W1-W0).max()/lam:.3f} waves  (docstring claims 2.5 waves within one beam radius)")
