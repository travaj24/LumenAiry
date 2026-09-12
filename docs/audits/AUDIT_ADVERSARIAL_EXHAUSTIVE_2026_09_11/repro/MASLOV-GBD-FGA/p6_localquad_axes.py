"""Probe 6: is the local_quadrature sampling window aligned/scaled correctly?

Synthetic chart: OPD(u3,u4) = 0.5*A*u3^2 + 0.5*B*u4^2 (waves, normalised coords),
s1x = c*u3, s1y = c*u4, E_in == 1.  Then the integral the integrators are
supposed to evaluate is

   I = int E_in * exp(2 pi i OPD) * |det ds1/dv2| d^2 v2
     = c^2/(vx_h*vy_h) * (vx_h*vy_h) * int exp(i pi (A u3^2 + B u4^2)) du3 du4
     = c^2 * exp(i pi (sgnA + sgnB)/4) / sqrt(|A B|)          (infinite limits)
"""
import numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.lenses_maslov import (_integrate_local_quadrature,
                                              _integrate_stationary_phase)
from lumenairy.elements.lenses import _multi_indices_total_degree

poly_order = 2
mi = _multi_indices_total_degree(4, poly_order)
K1 = np.array([k[0] for k in mi], np.int64); K2 = np.array([k[1] for k in mi], np.int64)
K3 = np.array([k[2] for k in mi], np.int64); K4 = np.array([k[3] for k in mi], np.int64)
idx = {k: j for j, k in enumerate(mi)}

vx_h = vy_h = 1.0            # normalised == physical v2
c = 1.0

def make_coefs(A, B):
    # u^2 = (T0 + T2)/2   ->  0.5*A*u3^2 = 0.25*A*(T0(u3) + T2(u3))
    co = np.zeros(len(mi))
    co[idx[(0,0,0,0)]] += 0.25*A + 0.25*B
    co[idx[(0,0,2,0)]] += 0.25*A
    co[idx[(0,0,0,2)]] += 0.25*B
    sx = np.zeros(len(mi)); sx[idx[(0,0,1,0)]] = c      # s1x = c*T1(u3) = c*u3
    sy = np.zeros(len(mi)); sy[idx[(0,0,0,1)]] = c
    return co, sx, sy

def sample_E(s1x, s1y):
    return np.ones_like(s1x, dtype=np.complex128)

def prog(*a, **k): pass

N_out = 1
u_s2x_out = np.zeros((1,1)); u_s2y_out = np.zeros((1,1)); inbox = np.array([True])

print(f"{'A':>8} {'B':>8} | {'exact':>26} | {'local_quad':>26} {'relerr':>9} | "
      f"{'stat_phase':>26} {'relerr':>9}")
for A, B in [(40.,40.), (40.,4.), (4.,40.), (200.,4.), (4.,200.), (100.,10.)]:
    co, sx, sy = make_coefs(A, B)
    exact = c**2*np.exp(1j*np.pi*(np.sign(A)+np.sign(B))/4)/np.sqrt(abs(A*B))
    lq = _integrate_local_quadrature(co, sx, sy, K1, K2, K3, K4, poly_order,
            N_out, u_s2x_out, u_s2y_out, inbox, vx_h, vy_h, sample_E,
            30, 1e-12, 24, 4.0, prog, False)[0,0]
    sp = _integrate_stationary_phase(co, sx, sy, mi, K1, K2, K3, K4, poly_order,
            N_out, u_s2x_out, u_s2y_out, inbox, vx_h, vy_h, sample_E,
            30, 1e-12, prog, False)[0,0]
    print(f"{A:8.1f} {B:8.1f} | {exact:26.6g} | {lq:26.6g} {abs(lq-exact)/abs(exact):9.2e} | "
          f"{sp:26.6g} {abs(sp-exact)/abs(exact):9.2e}")
