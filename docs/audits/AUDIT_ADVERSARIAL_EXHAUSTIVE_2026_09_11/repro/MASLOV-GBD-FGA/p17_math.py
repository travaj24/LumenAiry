"""Probe 17: _math/chebyshev vs numpy.polynomial; _math/levin vs brute force."""
import numpy as np, sys, time
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from numpy.polynomial import chebyshev as C
from lumenairy._math.chebyshev import (chebyshev_vandermonde as V,
                                       chebyshev_derivative_vandermonde as Vd,
                                       chebyshev_second_derivative_vandermonde as Vdd)
from lumenairy._math.levin import levin1d_adaptive, levin2d

print("### chebyshev helpers vs numpy.polynomial (u in [-1,1], and OUTSIDE) ###")
for lo, hi, tag in ((-1.0, 1.0, 'inside'), (-1.5, 1.5, 'outside [-1,1]')):
    u = np.linspace(lo, hi, 401)
    for nmax in (8, 16, 32):
        T, Tp, Tpp = V(u, nmax), Vd(u, nmax), Vdd(u, nmax)
        eT=eP=ePP=0.0
        for n in range(nmax+1):
            c = np.zeros(n+1); c[n]=1.0
            eT  = max(eT,  np.max(np.abs(T[n]-C.chebval(u,c))))
            eP  = max(eP,  np.max(np.abs(Tp[n]-C.chebval(u,C.chebder(c)))) if n>0 else 0.0)
            ePP = max(ePP, np.max(np.abs(Tpp[n]-C.chebval(u,C.chebder(c,2)))) if n>1 else 0.0)
        print(f"  {tag:14s} nmax={nmax:3d}: max|T-ref|={eT:.2e}  max|T'-ref|={eP:.2e}  max|T''-ref|={ePP:.2e}")

print("\n### levin1d_adaptive: I = int_0^1 f(x) exp(i w g(x)) dx ###")
from scipy.integrate import quad
def brute(f, g, w, a, b, n=4_000_001):
    x = np.linspace(a, b, n)
    y = f(x)*np.exp(1j*w*g(x))
    # Simpson
    h = (b-a)/(n-1)
    s = y[0]+y[-1]+4*y[1:-1:2].sum()+2*y[2:-2:2].sum()
    return s*h/3.0
f  = lambda x: 1.0/(1.0+x**2)
print("  NO stationary point: g(x) = x   (g' = 1)")
for w in (1e2, 1e3, 1e4):
    g = lambda x, w=w: x
    gy = lambda x: np.ones_like(x)
    t=time.perf_counter()
    I = levin1d_adaptive(lambda x, w=w: w*x, lambda x, w=w: w*np.ones_like(x), f, 0.0, 1.0, tol=1e-12)
    ref = brute(f, lambda x:x, w, 0.0, 1.0)
    print(f"    w={w:8.0f}: levin={I:.12g}  ref={ref:.12g}  relerr={abs(I-ref)/abs(ref):.3e}"
          f"  ({time.perf_counter()-t:.2f}s)")
print("  WITH a stationary point at x=0.5: g(x) = (x-0.5)^2")
for w in (1e2, 1e3, 1e4):
    t=time.perf_counter()
    I = levin1d_adaptive(lambda x, w=w: w*(x-0.5)**2,
                         lambda x, w=w: 2*w*(x-0.5), f, 0.0, 1.0, tol=1e-12)
    ref = brute(f, lambda x:(x-0.5)**2, w, 0.0, 1.0)
    print(f"    w={w:8.0f}: levin={I:.12g}  ref={ref:.12g}  relerr={abs(I-ref)/abs(ref):.3e}"
          f"  ({time.perf_counter()-t:.2f}s)")
