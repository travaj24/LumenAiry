"""Probe 1: Sziklas-Siegman carrier step vs analytic Gaussian + plain ASM."""
import sys, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.carrier import (
    propagate_carrier_referenced, carrier_referenced_reconstruct,
    carrier_referenced_envelope, carrier_referenced_fit_radius)
from lumenairy.propagators.propagation import angular_spectrum_propagate

wl = 1.31e-6
k = 2*np.pi/wl

def gaussian_field(x, y, w0, z, wl):
    """Analytic Gaussian, waist w0 at z=0, exp(-i omega t)/exp(+ikz) forward."""
    k = 2*np.pi/wl
    zR = np.pi*w0**2/wl
    X, Y = np.meshgrid(x, y, indexing='xy')
    r2 = X**2 + Y**2
    w = w0*np.sqrt(1+(z/zR)**2)
    if z == 0:
        invR = 0.0
    else:
        invR = z/(z**2 + zR**2)     # 1/R, R = z(1+(zR/z)^2)
    gouy = np.arctan2(z, zR)
    return (w0/w)*np.exp(-r2/w**2)*np.exp(1j*(k*z + k*r2*invR/2.0 - gouy))

# diverging: waist at z = -z0 so at plane 0 we have R>0
w0 = 30e-6
zR = np.pi*w0**2/wl
z0 = 6*zR                 # plane 0 is 6 zR past waist
N = 512
w_at0 = w0*np.sqrt(1+(z0/zR)**2)
R0 = z0*(1+(zR/z0)**2)
print(f"w0={w0*1e6:.3f}um zR={zR*1e6:.1f}um z0={z0*1e6:.1f}um w(0)={w_at0*1e6:.2f}um R0={R0*1e6:.2f}um NA~{w0/zR:.4f}")
dx = 4.0*w_at0/N*2       # grid half-width = 4 w
dx = (8.0*w_at0)/N
x = (np.arange(N)-N/2)*dx
y = x.copy()
E0 = gaussian_field(x, y, w0, z0, wl)

# envelope w.r.t. parabolic carrier R0
env0 = carrier_referenced_envelope(E0, R0, wl, dx)
print("env0 residual phase p-v (rad, within 2w):",
      np.ptp(np.angle(env0)[np.abs(env0)>0.05*np.abs(env0).max()]))

for z in (0.5*z0, 2.0*z0, 10.0*z0):
    env, R, dxo = propagate_carrier_referenced(env0, R0, z, wl, dx)
    Efull = carrier_referenced_reconstruct(env, R, wl, dxo)
    xo = (np.arange(N)-N/2)*dxo
    Eana = gaussian_field(xo, xo, w0, z0+z, wl)
    m = (R0+z)/R0
    # relative L2
    rel = np.linalg.norm(Efull-Eana)/np.linalg.norm(Eana)
    # phase RMS over bright support
    br = np.abs(Eana) > 0.05*np.abs(Eana).max()
    dphi = np.angle(Efull[br]*np.conj(Eana[br]))
    dphi -= np.mean(dphi)
    # power
    p_c = (np.abs(Efull)**2).sum()*dxo*dxo
    p_a = (np.abs(Eana)**2).sum()*dxo*dxo
    print(f"z={z*1e6:9.1f}um m={m:8.4f} dx_out={dxo*1e9:8.2f}nm relL2={rel:.3e} "
          f"phaseRMS={np.std(dphi):.3e} rad  P_carrier/P_ana={p_c/p_a:.6f}")
