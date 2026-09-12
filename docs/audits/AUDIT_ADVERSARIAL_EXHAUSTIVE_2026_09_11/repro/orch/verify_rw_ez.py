"""Orchestrator re-verification of PROP-HF's P0: richards_wolf_focus E_z sign.

Independent oracle (derived from scratch, no Novotny-Hecht sign conventions
borrowed): a ray through exit-pupil point at azimuth phi_p, radius f*sin(theta),
travels toward the focus along s = (-sin(t)cos(p), -sin(t)sin(p), cos(t)).
The x-polarised input's radial component rides the rigid rotation
rho_hat -> e_theta = (cos(t)cos(p), cos(t)sin(p), +sin(t)) (perpendicular to s,
continuous from rho_hat as t->0); the azimuthal component is unchanged.
Strength vector a = cos(p) e_theta - sin(p) phi_hat.
On the focal x-axis (x>0, y=0, z=0):  E_z/E_x = -2i I01 / (I00 + I02)
with I00 = int sqrt(c) s (1+c) J0(k x s) dt, I01 = int sqrt(c) s^2 J1(k x s) dt,
I02 = int sqrt(c) s (1-c) J2(k x s) dt.  Negative-imaginary for small x>0.
"""
import sys, time
import numpy as np
from scipy.integrate import quad
from scipy.special import j0, j1, jv

t0 = time.time()
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.vector_diffraction import richards_wolf_focus
print(f"import {time.time()-t0:.1f}s", flush=True)

wl, NA, f = 633e-9, 0.5, 4e-3
Np = 256
dx_p = 2.0 * f * NA / Np * 1.05     # array spans the rim with 5 % margin
pupil = np.ones((Np, Np), dtype=complex)
Ex, Ey, Ez, xf, yf = richards_wolf_focus(pupil, wl, NA, f, dx_p, polarization='x')
print("shapes", Ex.shape, "dx_focal", xf[1]-xf[0], flush=True)
k = 2*np.pi/wl
tmax = np.arcsin(NA)

def oracle_ratio(x):
    I00 = quad(lambda t: np.sqrt(np.cos(t))*np.sin(t)*(1+np.cos(t))*j0(k*x*np.sin(t)), 0, tmax, limit=400)[0]
    I01 = quad(lambda t: np.sqrt(np.cos(t))*np.sin(t)**2*j1(k*x*np.sin(t)), 0, tmax, limit=400)[0]
    I02 = quad(lambda t: np.sqrt(np.cos(t))*np.sin(t)*(1-np.cos(t))*jv(2, k*x*np.sin(t)), 0, tmax, limit=400)[0]
    return -2j*I01/(I00+I02)

c = Np//2
print("  x_f [um]     code Ez/Ex (row y=0, col x)        oracle Ez/Ex          code/oracle")
for kk in (1, 2, 3, 5):
    x = xf[c+kk]
    r_code = Ez[c, c+kk]/Ex[c, c+kk]
    r_or = oracle_ratio(x)
    print(f"  {x*1e6:8.3f}   {r_code.real:+.5f}{r_code.imag:+.5f}j   {r_or.real:+.5f}{r_or.imag:+.5f}j   {(r_code/r_or).real:+.4f}{(r_code/r_or).imag:+.4f}j")
# orientation sanity: Ez should be ~0 along the y-axis (x=0) and odd in x
print("Ez along y-axis / |Ez|max:", np.abs(Ez[c+3, c])/np.abs(Ez).max())
print("Ez odd in x?  Ez[c,c+3] + Ez[c,c-3] rel:", abs(Ez[c, c+3]+Ez[c, c-3])/abs(Ez[c, c+3]))
print("Ex even in x? rel:", abs(Ex[c, c+3]-Ex[c, c-3])/abs(Ex[c, c+3]))
print(f"total {time.time()-t0:.1f}s")
