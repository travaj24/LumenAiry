"""Probe 1d: structure of the through-focus residual phase error."""
import sys, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.carrier import (
    propagate_carrier_referenced, carrier_referenced_reconstruct,
    carrier_referenced_envelope)
import lumenairy.propagators.carrier as C

wl = 1.31e-6
def gauss(x, w0, z, wl):
    k = 2*np.pi/wl; zR = np.pi*w0**2/wl
    X, Y = np.meshgrid(x, x, indexing='xy'); r2 = X**2 + Y**2
    w = w0*np.sqrt(1+(z/zR)**2); invR = 0.0 if z == 0 else z/(z**2 + zR**2)
    return (w0/w)*np.exp(-r2/w**2)*np.exp(1j*(k*z + k*r2*invR/2.0 - np.arctan2(z, zR)))

w0 = 4e-6; zR = np.pi*w0**2/wl; z0 = -30e-3
w_in = w0*np.sqrt(1+(z0/zR)**2); R0 = z0*(1+(zR/z0)**2)
N = 2048; ext = 5.0
dx = (2*ext*w_in)/N
x = (np.arange(N)-N/2)*dx
E0 = gauss(x, w0, z0, wl)
env0 = carrier_referenced_envelope(E0, R0, wl, dx)
z = 60e-3
for gk in ('auto', 'fresnel'):
    env, R, dxo = propagate_carrier_referenced(env0, R0, z, wl, dx, gap_kernel=gk)
    Ef = carrier_referenced_reconstruct(env, R, wl, dxo)
    xo = (np.arange(N)-N/2)*dxo
    Ea = gauss(xo, w0, z0+z, wl)
    dphi = np.angle(Ef*np.conj(Ea))
    A = np.abs(Ea); W = A**2
    X, Y = np.meshgrid(xo, xo, indexing='xy'); r2 = X**2+Y**2
    # weighted LSQ fit  dphi ~ a + b r2
    M = np.stack([np.ones(r2.size), r2.ravel()], 1)
    Wv = W.ravel()
    ATA = M.T @ (M*Wv[:,None]); ATb = M.T @ (Wv*dphi.ravel())
    c = np.linalg.solve(ATA, ATb)
    res = dphi.ravel() - M@c
    rms_res = np.sqrt((Wv*res**2).sum()/Wv.sum())
    rms_tot = np.sqrt((Wv*(dphi.ravel()-np.average(dphi.ravel(),weights=Wv))**2).sum()/Wv.sum())
    k = 2*np.pi/wl
    R_equiv = k/(2*c[1]) if c[1] != 0 else np.inf
    print(f"gk={gk:8s} piston={c[0]:+.5f} rad  quad coeff={c[1]:+.5g} rad/m^2 -> dR^-1 equiv R={R_equiv:.6g} m")
    print(f"          amp-weighted phase RMS total={rms_tot:.4e}  after removing piston+quadratic={rms_res:.4e}")
    # encircled energy / second moment comparison
    r2m_c = np.sqrt((np.abs(Ef)**2*r2).sum()/ (np.abs(Ef)**2).sum())
    r2m_a = np.sqrt((np.abs(Ea)**2*r2).sum()/ (np.abs(Ea)**2).sum())
    print(f"          r2m carrier={r2m_c*1e6:.5f}um analytic={r2m_a*1e6:.5f}um  rel={r2m_c/r2m_a-1:+.3e}")
