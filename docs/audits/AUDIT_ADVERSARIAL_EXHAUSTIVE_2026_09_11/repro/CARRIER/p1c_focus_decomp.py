"""Probe 1c: decompose the through-focus error into amplitude / phase / piston."""
import sys, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.carrier import (
    propagate_carrier_referenced, carrier_referenced_reconstruct,
    carrier_referenced_envelope)

wl = 1.31e-6
def gaussian_field(x, y, w0, z, wl):
    k = 2*np.pi/wl; zR = np.pi*w0**2/wl
    X, Y = np.meshgrid(x, y, indexing='xy'); r2 = X**2 + Y**2
    w = w0*np.sqrt(1+(z/zR)**2)
    invR = 0.0 if z == 0 else z/(z**2 + zR**2)
    return (w0/w)*np.exp(-r2/w**2)*np.exp(1j*(k*z + k*r2*invR/2.0 - np.arctan2(z, zR)))

w0 = 4e-6; zR = np.pi*w0**2/wl; z0 = -30e-3
w_in = w0*np.sqrt(1+(z0/zR)**2); R0 = z0*(1+(zR/z0)**2)

for ext in (2.6, 3.5, 5.0):
  for N in (2048, 4096):
    dx = (2*ext*w_in)/N
    x = (np.arange(N)-N/2)*dx
    E0 = gaussian_field(x, x, w0, z0, wl)
    env0 = carrier_referenced_envelope(E0, R0, wl, dx)
    z = 60e-3
    env, R, dxo = propagate_carrier_referenced(env0, R0, z, wl, dx)
    Ef = carrier_referenced_reconstruct(env, R, wl, dxo)
    xo = (np.arange(N)-N/2)*dxo
    Ea = gaussian_field(xo, xo, w0, z0+z, wl)
    rel = np.linalg.norm(Ef-Ea)/np.linalg.norm(Ea)
    amp = np.linalg.norm(np.abs(Ef)-np.abs(Ea))/np.linalg.norm(np.abs(Ea))
    br = np.abs(Ea) > 0.05*np.abs(Ea).max()
    dp = np.angle(Ef[br]*np.conj(Ea[br]))
    pist = np.angle(np.vdot(Ea, Ef))
    relp = np.linalg.norm(Ef*np.exp(-1j*pist)-Ea)/np.linalg.norm(Ea)
    print(f"ext={ext:4.1f} N={N:5d} dx={dx*1e6:7.4f}um relL2={rel:.3e} relL2_pistonfree={relp:.3e} "
          f"ampL2={amp:.3e} piston={pist:+.4f}rad phRMS={np.std(dp-np.mean(dp)):.3e} P={(np.abs(Ef)**2).sum()/(np.abs(Ea)**2).sum():.6f}")
