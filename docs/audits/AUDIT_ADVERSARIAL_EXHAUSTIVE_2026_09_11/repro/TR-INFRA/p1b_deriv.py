"""TR-INFRA probe 1b: _Cheb2DEvaluator derivatives vs numpy.polynomial.chebyshev
and finite differences; numba-vs-numpy agreement; out-of-domain behaviour."""
import numpy as np
from numpy.polynomial import chebyshev as NC
from lumenairy.elements import _lens_traced as T

# ---- recurrence check against numpy.polynomial.chebyshev -------------------
print("=== T_k and T'_k recurrences vs numpy.polynomial.chebyshev ===")
for maxk in (6, 10, 14):
    u = np.array([-1.0, -0.9999, -0.5, 0.0, 1e-17, 0.5, 0.9999, 1.0])
    Tv = T._cheb_vand_2d(u, maxk, np)
    Dv = T._cheb_deriv_vand_2d(u, maxk, np)
    eT = eD = 0.0
    for k in range(maxk+1):
        c = np.zeros(k+1); c[k] = 1.0
        eT = max(eT, np.abs(Tv[k]-NC.chebval(u,c)).max())
        eD = max(eD, np.abs(Dv[k]-NC.chebval(u,NC.chebder(c))).max())
    print(f"  max_k={maxk:2d}  max|dT|={eT:.3e}  max|dT'|={eD:.3e}")

# ---- evaluator derivative vs central finite differences -------------------
print("\n=== _Cheb2DEvaluator.ev(dx=1/dy=1) vs central finite differences ===")
n = 81; R = 1.0
xs = np.linspace(-R, R, n)
X, Y = np.meshgrid(xs, xs, indexing='ij')
rng = np.random.default_rng(3)
vals = np.sin(1.3*X)*np.cos(0.7*Y) + 0.2*X*Y**2
ev = T._Cheb2DEvaluator(xs, xs, vals, order=8)
for lab, pts in (('interior', (np.array([0.0, 0.25, -0.4]), np.array([0.1, -0.3, 0.6]))),
                 ('boundary', (np.array([ 1.0, -1.0, 1.0]), np.array([0.0,  1.0, -1.0]))),
                 ('outside(1.5R)', (np.array([1.5, -1.5, 0.0]), np.array([0.0, 1.5, -1.5])))):
    xq, yq = pts
    h = 1e-6
    fx_fd = (ev.ev(xq+h, yq) - ev.ev(xq-h, yq))/(2*h)
    fy_fd = (ev.ev(xq, yq+h) - ev.ev(xq, yq-h))/(2*h)
    f, fx, fy = ev.ev_value_and_grad(xq, yq)
    sc = max(np.abs(fx_fd).max(), np.abs(fy_fd).max(), 1e-30)
    print(f"  {lab:14s} max|fx-fd|/scale={np.abs(fx-fx_fd).max()/sc:.3e}"
          f"  max|fy-fd|/scale={np.abs(fy-fy_fd).max()/sc:.3e}   f={f}")

# ---- numba vs numpy backend agreement ------------------------------------
print("\n=== numba kernel vs pure-numpy fallback ===")
rng = np.random.default_rng(7)
for order in (6, 8, 10, 12):
    vals = (np.exp(-(X**2+Y**2)) + 0.3*X**3*Y + 0.05*np.cos(4*X))
    ev = T._Cheb2DEvaluator(xs, xs, vals, order=order)
    st = T._cheb_fit_state(ev)
    e_nb = T._Cheb2DEvaluator.from_state(st, backend='numba')
    e_np = T._Cheb2DEvaluator.from_state(st, backend='numpy')
    xq = rng.uniform(-1.2, 1.2, 20000); yq = rng.uniform(-1.2, 1.2, 20000)
    f1,gx1,gy1 = e_nb.ev_value_and_grad(xq,yq)
    f2,gx2,gy2 = e_np.ev_value_and_grad(xq,yq)
    def rel(a,b): 
        s=max(np.abs(b).max(),1e-300); return np.abs(a-b).max()/s
    print(f"  order {order:2d}: rel df={rel(f1,f2):.3e}  rel dfx={rel(gx1,gx2):.3e}"
          f"  rel dfy={rel(gy1,gy2):.3e}  bitwise equal: {np.array_equal(f1,f2)}")

# ---- out-of-domain growth -------------------------------------------------
print("\n=== EXTRAPOLATION growth of |T_k(u)| (no clamping in _to_u) ===")
for order in (6, 8, 10, 12):
    for uu in (1.0, 1.2, 1.5, 2.0, 3.0):
        tv = T._cheb_vand_2d(np.array([uu]), order, np)[:,0]
        print(f"  order {order:2d} u={uu:4.1f}  max|T_k|={np.abs(tv).max():.4e}"
              f"  max|T'_k|={np.abs(T._cheb_deriv_vand_2d(np.array([uu]),order,np)[:,0]).max():.4e}")
    print()
