"""TR-INFRA probe 10: _newton_invert_chunk on a synthetic map with a known
inverse -- convergence, tolerance units, non-converged handling."""
import numpy as np
from lumenairy.elements import _lens_traced as T

# synthetic FORWARD map:  x_out = m*x_in + a*x_in^3 ; y_out = m*y_in + a*y_in^3
# OPL = c * (x_in^2 + y_in^2)     (so we can check the inverted OPL exactly)
L = 1.0e-2            # launch radius (m)
n_launch = 161
xs_in = np.linspace(-L, L, n_launch)
Xi, Yi = np.meshgrid(xs_in, xs_in, indexing='ij')
m_, a_, c_ = -0.5, 8.0e2, 3.0e-2
x_out = m_*Xi + a_*Xi**3
y_out = m_*Yi + a_*Yi**3
opl   = c_*(Xi**2 + Yi**2)

dx = 2.0e-5
knot = dict(xs_in=xs_in, x_out_grid=x_out, y_out_grid=y_out, opl_grid=opl,
            launch_radius=L, dx=dx, bound=L*0.999,
            inv_M_x=1.0/m_, inv_M_y=1.0/m_,
            newton_fit='polynomial', fit_poly_order=6, fit_weights=None,
            cheb_backend=T._resolved_cheb_backend('polynomial'),
            newton_max_iters=T._NEWTON_MAX_ITERS)
# ship the parent's built fit (the shipped payload shape)
Sx=T._Cheb2DEvaluator(xs_in,xs_in,x_out,order=6)
Sy=T._Cheb2DEvaluator(xs_in,xs_in,y_out,order=6)
So=T._Cheb2DEvaluator(xs_in,xs_in,opl ,order=6)
knot['cheb_fit']=T._cheb_fit_payload(Sx,Sy,So)

# query points = the exact images of a known set of entrance points
xin = np.linspace(-0.6*L, 0.6*L, 41)
Q1, Q2 = np.meshgrid(xin, xin, indexing='ij')
qx = (m_*Q1 + a_*Q1**3).ravel()
qy = (m_*Q2 + a_*Q2**3).ravel()
opl_true = (c_*(Q1**2+Q2**2)).ravel()

out, n_unconv = T._newton_invert_chunk((knot, qx, qy))
fin = np.isfinite(out)
print(f"  points={qx.size}  non-finite (out-of-domain)={int((~fin).sum())}  n_unconverged={n_unconv}")
print(f"  max|OPL_newton - OPL_true| = {np.abs(out[fin]-opl_true[fin]).max():.4e} m"
      f"   (OPL span {opl_true.max()-opl_true.min():.4e} m)")
print(f"  rel = {np.abs(out[fin]-opl_true[fin]).max()/(opl_true.max()-opl_true.min()):.3e}")

print("\n  --- convergence tolerance is 0.01*dx in POSITION (m), on the EXIT side ---")
print(f"      tol = {0.01*dx:.3e} m; the map's local |dx_out/dx_in| ~ {abs(m_):.2f},"
      f" so the ENTRANCE-side accuracy is tol/|dx_out/dx_in| = {0.01*dx/abs(m_):.3e} m"
      f" = {0.01/abs(m_):.3f} wave-grid pixels")

print("\n  --- FOLDED map: what happens to non-converged / multi-root points? ---")
# make the map fold inside the launch disc: x_out = x - b x^3 with b large
b_ = 3.0e4
x_out2 = Xi - b_*Xi**3
y_out2 = Yi - b_*Yi**3
knot2 = dict(knot); knot2['x_out_grid']=x_out2; knot2['y_out_grid']=y_out2
knot2['inv_M_x']=1.0; knot2['inv_M_y']=1.0
Sx2=T._Cheb2DEvaluator(xs_in,xs_in,x_out2,order=6)
Sy2=T._Cheb2DEvaluator(xs_in,xs_in,y_out2,order=6)
knot2['cheb_fit']=T._cheb_fit_payload(Sx2,Sy2,So)
qx2 = np.linspace(-3e-3, 3e-3, 201); qy2 = np.zeros_like(qx2)
out2, nu2 = T._newton_invert_chunk((knot2, qx2, qy2))
print(f"      folded map: {int(np.isnan(out2).sum())}/{out2.size} NaN, n_unconverged={nu2}")
print(f"      -> non-converged points that stayed INSIDE the domain return a FINITE"
      f" OPL at the last iterate with NO NaN and NO flag on the value itself")
# verify: are any of the finite outputs actually wrong?
fin2=np.isfinite(out2)
print(f"      finite outputs: {int(fin2.sum())};  of those, how many are at a point"
      f" whose forward image does NOT match the query?")
