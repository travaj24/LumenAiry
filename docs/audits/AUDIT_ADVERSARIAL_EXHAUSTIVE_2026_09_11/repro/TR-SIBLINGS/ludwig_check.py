"""Independent check of ludwig_fold + the KMAH -pi/2 sign, against the EXACT
cubic-phase (Airy) integral and against direct numerical quadrature.

Canonical fold:  phi(s) = s^3/3 - zeta*s + A ,  f(s) = a0 + a1*s
  exact:  I = e^{ikA} 2pi [ a0 k^-1/3 Ai(-k^2/3 zeta) - i a1 k^-2/3 Ai'(...) ]
  saddles: s=+sqrt(zeta) -> S- = A - (2/3)z^{3/2}, phi''>0 -> e^{+i pi/4}
           s=-sqrt(zeta) -> S+ = A + (2/3)z^{3/2}, phi''<0 -> e^{-i pi/4}
"""
import numpy as np
from scipy.special import airy
from lumenairy.elements._lens_traced_multibranch import ludwig_fold

def exact(k, A, z, a0, a1):
    ai, aip, _, _ = airy(-(k**(2/3))*z)
    return np.exp(1j*k*A)*2*np.pi*(a0*k**(-1/3)*ai - 1j*a1*k**(-2/3)*aip)

def quad(k, A, z, a0, a1, smax=60.0, n=4_000_001):
    s = np.linspace(-smax, smax, n)
    # Gaussian damping to make the oscillatory tails converge; eps chosen so the
    # saddle region is untouched.
    eps = 1e-4
    f = (a0 + a1*s)*np.exp(1j*k*(s**3/3 - z*s + A))*np.exp(-eps*s**2)
    return np.trapezoid(f, s)

def geometric_pair(k, A, z, a0, a1):
    sz = np.sqrt(z)
    Sm = A - (2/3)*z**1.5
    Sp = A + (2/3)*z**1.5
    pref = np.sqrt(2*np.pi/(k*2*sz))
    Am = (a0 + a1*sz)*pref*np.exp(1j*np.pi/4)     # S- branch, phi''>0
    Ap = (a0 - a1*sz)*pref*np.exp(-1j*np.pi/4)    # S+ branch, phi''<0  (KMAH -pi/2 rel.)
    return Sp, Sm, Ap, Am

print('=== ludwig_fold vs exact CFU/Airy (bright side) ===')
for k in (1.0e3, 1.0e5, 1.0e7):
    for z in (2.0e-3, 1e-2, 5e-2):
        for (a0, a1) in ((1.0, 0.0), (1.0, 0.3), (0.2+0.1j, -0.7+0.2j)):
            Sp, Sm, Ap, Am = geometric_pair(k, 0.0, z, a0, a1)
            u = ludwig_fold(k, Sp, Sm, Ap, Am)
            e = exact(k, 0.0, z, a0, a1)
            print(' k=%.0e z=%.3g a=(%s,%s)  |ludwig-exact|/|exact| = %.3e'
                  % (k, z, a0, a1, abs(u-e)/abs(e)))
print()
print('=== sanity: exact vs brute-force quadrature (k=1e3) ===')
for z in (5e-2, 2e-1):
    for (a0, a1) in ((1.0, 0.0), (1.0, 0.3)):
        e = exact(1.0e3, 0.0, z, a0, a1); q = quad(1.0e3, 0.0, z, a0, a1)
        print('  z=%.3g a=(%g,%g)  exact %s  quad %s  rel %.3e'
              % (z, a0, a1, np.round(e, 8), np.round(q, 8), abs(e-q)/abs(e)))
print()
print('=== KMAH SIGN: plain 2-branch sum with exp(-i pi/2) on the S+ branch ===')
print('    vs the far-bright-side asymptote of the exact Airy form')
k = 1.0e6
for z in (1e-3, 3e-3, 1e-2):
    a0, a1 = 1.0, 0.0
    Sp, Sm, Ap, Am = geometric_pair(k, 0.0, z, a0, a1)
    plain = Ap*np.exp(1j*k*Sp) + Am*np.exp(1j*k*Sm)
    e = exact(k, 0.0, z, a0, a1)
    # wrong-sign control: +pi/2 instead of -pi/2 on the S+ branch
    wrong = (Ap*np.exp(1j*np.pi/2)*np.exp(1j*k*Sp)
             + Am*np.exp(1j*k*Sm))*np.exp(-1j*np.pi/2)*0 + \
            (a0*np.sqrt(2*np.pi/(k*2*np.sqrt(z)))*np.exp(+1j*np.pi/4)*np.exp(1j*k*Sp)
             + Am*np.exp(1j*k*Sm))
    print('  z=%.3g  k^{2/3}z=%.1f  |plain(-pi/2)-exact|/|exact| = %.3e   '
          '|plain(+pi/2)-exact|/|exact| = %.3e'
          % (z, k**(2/3)*z, abs(plain-e)/abs(e), abs(wrong-e)/abs(e)))
