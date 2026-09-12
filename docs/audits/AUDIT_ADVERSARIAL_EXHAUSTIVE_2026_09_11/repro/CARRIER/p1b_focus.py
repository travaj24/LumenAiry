"""Probe 1b: converging carrier THROUGH focus (auto-split bridge)."""
import sys, warnings, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.carrier import (
    propagate_carrier_referenced, carrier_referenced_reconstruct,
    carrier_referenced_envelope)

wl = 1.31e-6
def gaussian_field(x, y, w0, z, wl):
    k = 2*np.pi/wl
    zR = np.pi*w0**2/wl
    X, Y = np.meshgrid(x, y, indexing='xy')
    r2 = X**2 + Y**2
    w = w0*np.sqrt(1+(z/zR)**2)
    invR = 0.0 if z == 0 else z/(z**2 + zR**2)
    gouy = np.arctan2(z, zR)
    return (w0/w)*np.exp(-r2/w**2)*np.exp(1j*(k*z + k*r2*invR/2.0 - gouy))

w0 = 4e-6
zR = np.pi*w0**2/wl
z0 = -30e-3                     # start plane 30 mm BEFORE the waist
w_in = w0*np.sqrt(1+(z0/zR)**2)
R0 = z0*(1+(zR/z0)**2)          # negative => converging
N = 2048
dx = (2*2.6*w_in)/N
x = (np.arange(N)-N/2)*dx
print(f"w0={w0*1e6}um zR={zR*1e6:.2f}um w_in={w_in*1e3:.4f}mm R0={R0*1e3:.4f}mm dx={dx*1e6:.4f}um half/w={0.5*N*dx/w_in:.2f}")
E0 = gaussian_field(x, x, w0, z0, wl)
env0 = carrier_referenced_envelope(E0, R0, wl, dx)

for z in (30e-3, 30e-3+6*zR, 60e-3):
    with warnings.catch_warnings(record=True) as wl_:
        warnings.simplefilter('always')
        env, R, dxo = propagate_carrier_referenced(env0, R0, z, wl, dx)
        warns = [str(w.message)[:90] for w in wl_]
    Efull = carrier_referenced_reconstruct(env, R, wl, dxo)
    xo = (np.arange(N)-N/2)*dxo
    Eana = gaussian_field(xo, xo, w0, z0+z, wl)
    rel = np.linalg.norm(Efull-Eana)/np.linalg.norm(Eana)
    # peak & r2m windowed
    Ic = np.abs(Efull)**2; Ia = np.abs(Eana)**2
    print(f"z={z*1e3:8.4f}mm z_final={(z0+z)*1e6:9.2f}um R_out={R:.6g} dx_out={dxo*1e9:9.3f}nm "
          f"relL2={rel:.3e} peak_ratio={Ic.max()/Ia.max():.6f} P={Ic.sum()/Ia.sum():.6f}")
    if warns: print("   warnings:", warns)
